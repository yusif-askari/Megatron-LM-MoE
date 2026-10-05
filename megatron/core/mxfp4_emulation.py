"""Emulated MXFP4 GEMMs for GPUs without FP4 tensor cores (GH200).

Each GEMM operand is quantized to OCP MXFP4 (blocks of 32 along the contraction dim, one
power-of-two e8m0 scale per block, e2m1 elements, round-to-nearest-even, saturating at 6)
and dequantized back to bf16. The product is then a plain bf16 GEMM with fp32 accumulation.
Every e2m1 value times a power-of-two scale is exact in bf16, so the result is what an
MXFP4 tensor-core GEMM computes; only the speed is not.

Every GEMM quantizes along its own contraction dim, as on hardware (and as in ALTO's
``mxfp4_base`` recipe, which has no 2-D blocks, Hadamard transform or stochastic rounding):
    fprop  y  = x  · Wᵀ   x along K,  W along K
    dgrad  dx = dy · W    dy along N, W along N
    wgrad  dW = dyᵀ · x   dy and x along the tokens of each expert (each expert's segment
                          zero-padded to a multiple of 32, as an aligned grouped GEMM would)

Scale rules (``--mxfp4-scale-rule``):
    floor  OCP spec / tensorcast default: e = floor(log2 amax) - 2. The block max lands in
           [4, 8)·s and is clipped to 6·s whenever it is above 6·s (~42% of blocks).
    alto   ALTO ``blockscale_selection="default"`` (AMD-AGI/ALTO rad/lpt,
           alto/kernels/fp4/mxfp4/mxfp_quantization.py::_calculate_scales): the same, but
           e + 1 when amax/s >= 7. Implemented bit for bit: amax in the operand dtype, +0.25 on
           its significand, keep the carried exponent. Matches ALTO's Triton kernel exactly on
           fp32 input (mxfp4-scale-selection/test_alto_kernel_gh200.py).
"""
import itertools
import math

import torch

BLOCK = 32
RULES = ("floor", "alto")
_E8M0_MIN = -126            # ALTO clamps the biased scale to >= 1


def scale_exponent(amax: torch.Tensor, rule: str) -> torch.Tensor:
    """Unbiased exponent e of the block scale 2^e, as float32. ``amax`` keeps the operand dtype."""
    if rule == "floor":
        a = amax.float()
        e = (torch.frexp(a).exponent - 1 - 2).float()      # exact floor(log2 a) - 2
        e = torch.where(a > 0, e, torch.full_like(e, _E8M0_MIN))
    elif rule == "alto":
        if amax.dtype == torch.bfloat16:
            bits = amax.view(torch.int16).to(torch.int32) & 0x7FFF
            e = (((bits + (1 << 5)) & (0x1FF << 7)) >> 7) - 127 - 2
        else:
            bits = amax.float().view(torch.int32).to(torch.int64) & 0x7FFFFFFF
            e = (((bits + (1 << 21)) & (0x1FF << 23)) >> 23) - 127 - 2
        e = e.float()
    else:
        raise ValueError(f"unknown MXFP4 scale rule {rule!r}, expected one of {RULES}")
    return e.clamp(min=_E8M0_MIN)


def _qdq_lastdim(x: torch.Tensor, rule: str) -> torch.Tensor:
    """Quantize-dequantize along the last dim (a multiple of 32). Returns x's dtype."""
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // BLOCK, BLOCK)
    amax = xb.abs().amax(-1, keepdim=True)
    s = torch.exp2(scale_exponent(amax, rule))
    y = xb.float() / s
    a = y.abs().clamp(max=6.0)
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    q = torch.copysign((torch.round(a / step) * step).clamp(max=6.0), y)   # round: half to even
    return (q * s).to(x.dtype).reshape(x.shape)


_qdq_compiled = None


def qdq(x: torch.Tensor, rule: str, dim: int = -1) -> torch.Tensor:
    """MXFP4 quantize-dequantize of ``x`` in blocks of 32 along ``dim`` (must divide by 32)."""
    global _qdq_compiled
    if _qdq_compiled is None:
        try:
            _qdq_compiled = torch.compile(_qdq_lastdim, dynamic=True)
        except Exception:  # noqa: BLE001  (no inductor: fall back to eager)
            _qdq_compiled = _qdq_lastdim
    dim = dim % x.ndim
    if x.shape[dim] % BLOCK:
        raise ValueError(f"MXFP4 contraction dim {x.shape[dim]} is not a multiple of {BLOCK}")
    if dim == x.ndim - 1:
        return _qdq_compiled(x.contiguous(), rule)
    xt = x.transpose(dim, -1).contiguous()
    return _qdq_compiled(xt, rule).transpose(dim, -1)


def _offsets(counts, device):
    """Cumulative segment ends, the ``offs`` argument of torch._grouped_mm."""
    return torch.tensor(list(itertools.accumulate(counts)), device=device, dtype=torch.int32)


class _MXFP4GroupedLinear(torch.autograd.Function):
    """y[e] = Q(x[e]) · Q(W_e)ᵀ for the token segments of each local expert (no bias)."""

    @staticmethod
    def forward(ctx, x, m_splits, rule, *weights):
        n_tok = sum(m_splits)
        assert x.shape[0] == n_tok, (x.shape, n_tok)
        ctx.m_splits, ctx.rule = list(m_splits), rule
        ctx.save_for_backward(x, *weights)
        xq = qdq(x, rule, dim=-1)                                           # [T, K]
        wq = qdq(torch.stack(weights), rule, dim=-1)                        # [E, N, K] along K
        offs = _offsets(ctx.m_splits, x.device)
        return torch._grouped_mm(xq, wq.transpose(1, 2), offs=offs)          # [T, N]

    @staticmethod
    def backward(ctx, dy):
        x, *weights = ctx.saved_tensors
        rule, m_splits = ctx.rule, ctx.m_splits
        dy = dy.contiguous()
        offs = _offsets(m_splits, x.device)
        dx = None
        if ctx.needs_input_grad[0]:
            dyq = qdq(dy, rule, dim=-1)                                     # along N
            wq = qdq(torch.stack(weights), rule, dim=1)                     # [E, N, K] along N
            dx = torch._grouped_mm(dyq, wq, offs=offs)                       # [T, K]
        # wgrad: contraction over each expert's tokens -> pad every segment to a multiple of 32
        padded = [int(math.ceil(m / BLOCK) * BLOCK) for m in m_splits]
        if sum(padded) != sum(m_splits):
            starts = [0] + list(itertools.accumulate(padded))[:-1]
            rows = torch.cat([torch.arange(m, device=x.device) + p0
                              for m, p0 in zip(m_splits, starts)])
            xp = x.new_zeros(sum(padded), x.shape[1]).index_copy_(0, rows, x)
            dyp = dy.new_zeros(sum(padded), dy.shape[1]).index_copy_(0, rows, dy)
        else:
            xp, dyp = x, dy
        xq_t = qdq(xp, rule, dim=0)                                          # [Tp, K] along tokens
        dyq_t = qdq(dyp, rule, dim=0)                                        # [Tp, N] along tokens
        poffs = _offsets(padded, x.device)
        # bf16 output: _grouped_mm cannot emit fp32 from bf16 inputs on sm90. The extra bf16
        # rounding of dW (~2^-9 relative) is negligible next to the MXFP4 operand error (~15%).
        dw = torch._grouped_mm(dyq_t.t(), xq_t, offs=poffs)                  # [E, N, K]
        grads = []
        for e, w in enumerate(weights):
            if hasattr(w, "main_grad") and getattr(w, "grad_added_to_main_grad", None) is not None:
                # Megatron DDP + gradient-accumulation fusion: accumulate in fp32 like TE does
                w.main_grad.add_(dw[e].float())
                w.grad_added_to_main_grad = True
                grads.append(_dummy_grad(w))
            else:
                grads.append(dw[e].to(w.dtype))
        return (dx, None, None, *grads)


_DUMMY = {}


def _dummy_grad(w):
    """Placeholder grad (like TE's get_dummy_wgrad): DDP's hook needs param.grad to be set, and
    skips adding it because grad_added_to_main_grad is True."""
    key = (tuple(w.shape), w.dtype, w.device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
    return _DUMMY[key]


def grouped_linear(x: torch.Tensor, m_splits, weights, rule: str) -> torch.Tensor:
    """Emulated-MXFP4 grouped GEMM over the local experts' token segments."""
    return _MXFP4GroupedLinear.apply(x, list(m_splits), rule, *weights)


def te_grouped_linear_forward(module, x: torch.Tensor, m_splits, rule: str):
    """Drop-in for ``TEGroupedLinear(x, m_splits)`` (no bias, TP = 1): returns (output, None)."""
    assert getattr(module, "use_bias", False) is False, "MXFP4 emulation supports bias-free experts"
    weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
    return grouped_linear(x, m_splits, weights, rule), None
