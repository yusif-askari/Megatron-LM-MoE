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

Scopes (``--mxfp4-emulate``):
    experts     the routed experts' fc1/fc2 (TEGroupedMLP)
    all_linear  every TE linear inside the decoder blocks as well: attention / KDA projections,
                dense and shared-expert MLPs, MoE latent projections, with their fused input
                norm computed in fp32 first. Embedding, output layer and router stay bf16 (they
                are not TE linears), as do norms, softmax and attention / KDA internals.

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


def _qdq_dim(x: torch.Tensor, rule: str, dim: int) -> torch.Tensor:
    """Quantize-dequantize in blocks of 32 along ``dim`` (a multiple of 32), keeping x's layout:
    the dim is split into (n/32, 32) and the block max taken over the 32-axis, so no transpose
    copy is needed for the dgrad/wgrad axes. Returns x's dtype."""
    xb = x.unflatten(dim, (x.shape[dim] // BLOCK, BLOCK))
    amax = xb.abs().amax(dim + 1, keepdim=True)
    e = scale_exponent(amax, rule)
    # every divisor is a power of two, so multiplying by its exact reciprocal is bit-identical
    y = xb.float() * torch.exp2(-e)
    a = y.abs().clamp(max=6.0)
    big = a >= 4
    small = a < 2
    inv_step = torch.where(small, 2.0, torch.where(big, 0.5, 1.0))        # 1 / grid spacing
    step = torch.where(small, 0.5, torch.where(big, 2.0, 1.0))
    q = torch.copysign((torch.round(a * inv_step) * step).clamp(max=6.0), y)  # half to even
    return (q * torch.exp2(e)).to(x.dtype).flatten(dim, dim + 1)


_qdq_compiled = None


def qdq(x: torch.Tensor, rule: str, dim: int = -1) -> torch.Tensor:
    """MXFP4 quantize-dequantize of ``x`` in blocks of 32 along ``dim``. A ragged last block is
    zero-padded (a zero never raises the block max; hardware pads too)."""
    global _qdq_compiled
    if _qdq_compiled is None:
        # One compiled graph per variant (rank, axis, rule, ragged or not): a full model needs
        # a few dozen. Past dynamo's default limit of 8 the rest would silently run eager
        # (several times slower), so raise it once; each variant still compiles only once.
        import torch._dynamo as _dynamo  # "import torch._dynamo" would make torch a local name

        cfg = _dynamo.config
        for name, floor in (("recompile_limit", 256), ("cache_size_limit", 256),
                            ("accumulated_recompile_limit", 4096),
                            ("accumulated_cache_size_limit", 4096)):
            if hasattr(cfg, name):
                setattr(cfg, name, max(getattr(cfg, name), floor))
        try:
            _qdq_compiled = torch.compile(_qdq_dim, dynamic=True)
        except Exception:  # noqa: BLE001  (no inductor: fall back to eager)
            _qdq_compiled = _qdq_dim
    dim = dim % x.ndim
    n = x.shape[dim]
    pad = (-n) % BLOCK
    if pad:
        widths = [0, 0] * (x.ndim - 1 - dim) + [0, pad]
        x = torch.nn.functional.pad(x, widths)
    q = _qdq_compiled(x, rule, dim)
    return q.narrow(dim, 0, n) if pad else q


# Quantized weights are cached within one training step: weights only change in
# optimizer.step, while a step runs several micro-batches. The cache is off unless
# training.train_step brackets the step with new_step() / end_step(), so evaluation, tests and
# anything else always quantize afresh. Keys use the weight objects' identity, which is only
# safe within a step.
_WCACHE = {}
_CACHE_ON = False


def new_step():
    """Start a training step: clear the quantized-weight cache and enable it."""
    global _CACHE_ON
    _WCACHE.clear()
    _CACHE_ON = True


def end_step():
    """After optimizer.step: the weights changed, so clear and disable the cache."""
    global _CACHE_ON
    _WCACHE.clear()
    _CACHE_ON = False


def _cached(key, make):
    if not _CACHE_ON:
        return make()
    out = _WCACHE.get(key)
    if out is None:
        out = make()
        _WCACHE[key] = out
    return out


def _to_device(values, device, dtype):
    """Host list -> device tensor without a stream sync (pinned, non-blocking copy)."""
    return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)


def _offsets(counts, device):
    """Cumulative segment ends, the ``offs`` argument of torch._grouped_mm."""
    return _to_device(list(itertools.accumulate(counts)), device, torch.int32)


class _MXFP4GroupedLinear(torch.autograd.Function):
    """y[e] = Q(x[e]) · Q(W_e)ᵀ for the token segments of each local expert (no bias)."""

    @staticmethod
    def forward(ctx, x, m_splits, rule, *weights):
        n_tok = sum(m_splits)
        assert x.shape[0] == n_tok, (x.shape, n_tok)
        ctx.m_splits, ctx.rule = list(m_splits), rule
        ctx.save_for_backward(x, *weights)
        xq = qdq(x, rule, dim=-1)                                           # [T, K]
        wq = _cached((id(weights[0]), rule, "K"),
                     lambda: qdq(torch.stack(weights), rule, dim=-1))       # [E, N, K] along K
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
            wq = _cached((id(weights[0]), rule, "N"),
                         lambda: qdq(torch.stack(weights), rule, dim=1))    # [E, N, K] along N
            dx = torch._grouped_mm(dyq, wq, offs=offs)                       # [T, K]
        # wgrad: contraction over each expert's tokens -> pad every segment to a multiple of 32
        padded = [int(math.ceil(m / BLOCK) * BLOCK) for m in m_splits]
        if sum(padded) != sum(m_splits):
            # destination row of token t of expert e: t + (padded start - unpadded start of e)
            shift = [p - u for p, u in zip([0] + list(itertools.accumulate(padded))[:-1],
                                           [0] + list(itertools.accumulate(m_splits))[:-1])]
            n_tok = sum(m_splits)
            rows = torch.arange(n_tok, device=x.device) + torch.repeat_interleave(
                _to_device(shift, x.device, torch.int64),
                _to_device(m_splits, x.device, torch.int64), output_size=n_tok)
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


class _MXFP4Linear(torch.autograd.Function):
    """y = Q(x) · Q(W)ᵀ for x [T, K], W [N, K] (no bias)."""

    @staticmethod
    def forward(ctx, x, w, rule):
        ctx.rule = rule
        ctx.save_for_backward(x, w)
        wq = _cached((id(w), rule, "K"), lambda: qdq(w, rule, dim=1))
        return qdq(x, rule, dim=-1) @ wq.t()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        rule = ctx.rule
        dy = dy.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            wq = _cached((id(w), rule, "N"), lambda: qdq(w, rule, dim=0))
            dx = qdq(dy, rule, dim=-1) @ wq
        dw = qdq(dy, rule, dim=0).t() @ qdq(x, rule, dim=0)                 # along tokens
        if hasattr(w, "main_grad") and getattr(w, "grad_added_to_main_grad", None) is not None:
            w.main_grad.add_(dw.float())
            w.grad_added_to_main_grad = True
            return dx, _dummy_grad(w), None
        return dx, dw.to(w.dtype), None


def linear(x: torch.Tensor, w: torch.Tensor, rule: str) -> torch.Tensor:
    """Emulated-MXFP4 x · Wᵀ over the last dim of x (any leading shape)."""
    lead = x.shape[:-1]
    y = _MXFP4Linear.apply(x.reshape(-1, x.shape[-1]), w, rule)
    return y.reshape(*lead, w.shape[0])


def _fused_norm(module, x):
    """The input norm of a TE LayerNormLinear, in fp32 (same parameters as the fused kernel)."""
    xf = x.float()
    gamma = module.layer_norm_weight.float()
    if getattr(module, "zero_centered_gamma", False):
        gamma = gamma + 1.0
    if module.normalization == "RMSNorm":
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + module.eps) * gamma
    else:
        y = torch.nn.functional.layer_norm(xf, xf.shape[-1:], None, None, module.eps) * gamma
        if getattr(module, "layer_norm_bias", None) is not None:
            y = y + module.layer_norm_bias.float()
    return y.to(x.dtype)


def _te_linear_forward(module, x, rule, fused_norm):
    if fused_norm:
        x = _fused_norm(module, x)
    return linear(x, module.weight, rule), None


def apply_to_model(model, rule: str) -> list:
    """--mxfp4-emulate all_linear: route every TE (LayerNorm)Linear inside the decoder blocks
    through the emulated MXFP4 GEMM. Returns the patched module names."""
    import types

    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TELinear,
    )

    patched = []
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise ValueError("--mxfp4-emulate all_linear expects a model with a .decoder block")
    for name, mod in decoder.named_modules():
        fused = isinstance(mod, TELayerNormColumnParallelLinear)
        if not (fused or isinstance(mod, TELinear)):
            continue
        if getattr(mod, "use_bias", False):
            raise ValueError(f"--mxfp4-emulate all_linear: {name} has a bias (use --disable-bias-linear)")
        if getattr(mod, "tp_size", 1) != 1 or getattr(mod, "sequence_parallel", False):
            raise ValueError(f"--mxfp4-emulate all_linear needs TP = 1 ({name})")
        mod.forward = types.MethodType(
            lambda self, x, _f=fused: _te_linear_forward(self, x, rule, _f), mod)
        patched.append(name)
    return patched
