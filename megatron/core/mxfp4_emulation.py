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

Recipes (``--mxfp4-recipe``):
    plain  the above, round-to-nearest everywhere (ALTO's ``mxfp4_base``).
    alto   ALTO's training recipe (``lpt_recipe``: use_2dblock_w, use_hadamard, use_sr_grad):
             fprop  Q(x) along K  ·  Q2D(W)ᵀ, W in 32x32 tiles
             dgrad  Q_sr(dy) along N  ·  Q2D(W)   (the same tiled weight, no re-quantization)
             wgrad  Q_sr(H·dy)ᵀ  ·  Q(H·x), along tokens, H a random orthonormal 32x32
                    Hadamard applied to every 32-token block, redrawn every optimizer step
           (Q_sr: stochastic rounding, unbiased). (H·dy)ᵀ(H·x) = dyᵀ·x, so the rotation only
           spreads outliers across a block before quantizing.

Scopes (``--mxfp4-emulate``):
    experts     the routed experts' fc1/fc2 (TEGroupedMLP)
    all_linear  every TE linear inside the decoder blocks as well: attention / KDA projections,
                dense and shared-expert MLPs, MoE latent projections, with their fused input
                norm computed in fp32 first. Embedding, output layer and router stay bf16 (they
                are not TE linears), as do norms, softmax and attention / KDA internals.

Scale rules (``--mxfp4-scale-rule``):
    floor  OCP spec / tensorcast default: e = floor(log2 amax) - 2. The block max lands in
           [4, 8)·s and is clipped to 6·s whenever it is above 6·s (~42% of blocks).
    none   no quantization at all: the emulated GEMM path in plain bf16, to check that the
           plumbing (patching, norms, main_grad, weight cache) reproduces the bf16 run.
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
RULES = ("floor", "alto", "none")   # "none": no quantization, to check the plumbing
_E8M0_MIN = -126            # ALTO clamps the biased scale to >= 1


def scale_exponent(amax: torch.Tensor, rule: str) -> torch.Tensor:
    """Unbiased exponent e of the block scale 2^e, as float32. ``amax`` keeps the operand dtype."""
    # Both rules read the exponent field of amax directly. floor(log2 amax) is the biased exponent
    # - 127 for normal numbers; zero and subnormal amax have field 0 and land on the clamp below
    # (scale 2^-126, the block quantizes to ~0). Do not use torch.frexp here: on CUDA it returns
    # a garbage exponent for subnormal and inf amax, which gave a scale of 2^huge = inf and a
    # whole block of 0 * inf = NaN.
    if rule not in ("floor", "alto"):
        raise ValueError(f"unknown MXFP4 scale rule {rule!r}, expected one of {RULES}")
    if amax.dtype == torch.bfloat16:
        bits = amax.view(torch.int16).to(torch.int32) & 0x7FFF
        carry = (1 << 5) if rule == "alto" else 0          # alto: +0.25 on the significand
        e = (((bits + carry) & (0x1FF << 7)) >> 7) - 127 - 2
    else:
        bits = amax.float().view(torch.int32).to(torch.int64) & 0x7FFFFFFF
        carry = (1 << 21) if rule == "alto" else 0
        e = (((bits + carry) & (0x1FF << 23)) >> 23) - 127 - 2
    e = e.float()
    return e.clamp(min=_E8M0_MIN)


def _round_e2m1(a: torch.Tensor, sr: bool) -> torch.Tensor:
    """|y| clamped to [0, 6] onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}: round-to-nearest-
    even, or (sr) stochastic rounding between the two neighbours, which is unbiased."""
    big = a >= 4
    small = a < 2
    inv_step = torch.where(small, 2.0, torch.where(big, 0.5, 1.0))        # 1 / grid spacing
    step = torch.where(small, 0.5, torch.where(big, 2.0, 1.0))
    t = a * inv_step                                   # powers of two: exact
    if sr:
        lo = torch.floor(t)
        r = lo + (torch.rand_like(t) < (t - lo)).to(t.dtype)
    else:
        r = torch.round(t)                             # half to even (= even e2m1 code)
    return (r * step).clamp(max=6.0)


def _qdq_dim(x: torch.Tensor, rule: str, dim: int, sr: bool) -> torch.Tensor:
    """Quantize-dequantize in blocks of 32 along ``dim`` (a multiple of 32), keeping x's layout:
    the dim is split into (n/32, 32) and the block max taken over the 32-axis, so no transpose
    copy is needed for the dgrad/wgrad axes. Returns x's dtype."""
    xb = x.unflatten(dim, (x.shape[dim] // BLOCK, BLOCK))
    amax = xb.abs().amax(dim + 1, keepdim=True)
    e = scale_exponent(amax, rule)
    # every divisor is a power of two, so multiplying by its exact reciprocal is bit-identical
    y = xb.float() * torch.exp2(-e)
    q = torch.copysign(_round_e2m1(y.abs().clamp(max=6.0), sr), y)
    return (q * torch.exp2(e)).to(x.dtype).flatten(dim, dim + 1)


def _qdq_tile(x: torch.Tensor, rule: str) -> torch.Tensor:
    """Quantize-dequantize in 32x32 tiles over the last two dims (both multiples of 32), round-
    to-nearest. ALTO's 2-D weight blocks: one scale per tile, so the same quantized weight
    serves the forward (contraction over K) and the dgrad (contraction over N)."""
    n, k = x.shape[-2], x.shape[-1]
    xb = x.unflatten(-1, (k // BLOCK, BLOCK)).unflatten(-3, (n // BLOCK, BLOCK))
    amax = xb.abs().amax(dim=(-3, -1), keepdim=True)              # [..., n/32, 1, k/32, 1]
    e = scale_exponent(amax, rule)
    y = xb.float() * torch.exp2(-e)
    q = torch.copysign(_round_e2m1(y.abs().clamp(max=6.0), False), y)
    return (q * torch.exp2(e)).to(x.dtype).flatten(-2, -1).flatten(-3, -2)


_COMPILED = {}
_FORCE_RNE = False      # tests only: replace stochastic rounding by round-to-nearest


def _compiled(fn):
    """torch.compile'd version of fn (eager if inductor is unavailable)."""
    if fn not in _COMPILED:
        # One compiled graph per variant (rank, axis, rule, sr, ragged or not): a full model
        # needs a few dozen. Past dynamo's default limit of 8 the rest would silently run eager
        # (several times slower), so raise it once; each variant still compiles only once.
        import torch._dynamo as _dynamo  # "import torch._dynamo" would make torch a local name

        cfg = _dynamo.config
        for name, floor in (("recompile_limit", 256), ("cache_size_limit", 256),
                            ("accumulated_recompile_limit", 4096),
                            ("accumulated_cache_size_limit", 4096)):
            if hasattr(cfg, name):
                setattr(cfg, name, max(getattr(cfg, name), floor))
        try:
            _COMPILED[fn] = torch.compile(fn, dynamic=True)
        except Exception:  # noqa: BLE001  (no inductor: fall back to eager)
            _COMPILED[fn] = fn
    return _COMPILED[fn]


def qdq(x: torch.Tensor, rule: str, dim: int = -1, sr: bool = False) -> torch.Tensor:
    """MXFP4 quantize-dequantize of ``x`` in blocks of 32 along ``dim``; ``sr``: stochastic
    rounding. A ragged last block is zero-padded (a zero never raises the block max; hardware
    pads too)."""
    if rule == "none":          # plumbing check: the emulated path with quantization switched off
        return x
    sr = sr and not _FORCE_RNE
    dim = dim % x.ndim
    n = x.shape[dim]
    pad = (-n) % BLOCK
    if pad:
        widths = [0, 0] * (x.ndim - 1 - dim) + [0, pad]
        x = torch.nn.functional.pad(x, widths)
    # A plain, detached tensor: otherwise dynamo compiles separate versions for nn.Parameter
    # vs Tensor inputs and for requires_grad True/False (27 recompiles per rank in the 5B
    # smoke). Gradients never flow through qdq; the autograd Functions below own them.
    q = _compiled(_qdq_dim)(x.detach(), rule, dim, sr)
    return q.narrow(dim, 0, n) if pad else q


def qdq_tile(x: torch.Tensor, rule: str) -> torch.Tensor:
    """MXFP4 quantize-dequantize in 32x32 tiles over the last two dims (ragged edges padded)."""
    if rule == "none":
        return x
    n, k = x.shape[-2], x.shape[-1]
    pn, pk = (-n) % BLOCK, (-k) % BLOCK
    if pn or pk:
        x = torch.nn.functional.pad(x, (0, pk, 0, pn))
    q = _compiled(_qdq_tile)(x.detach(), rule)
    return q[..., :n, :k] if (pn or pk) else q


# ---- random Hadamard rotation along tokens (ALTO's wgrad transform) -------------------------
_H_SIZE = 32
_H = {}                                   # device -> this step's rotation
_H_GEN = torch.Generator().manual_seed(20261006)


def _sylvester(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def hadamard(device) -> torch.Tensor:
    """This step's 32x32 random orthonormal Hadamard rotation (as ALTO's HadamardFactory with
    randomized=True: a Hadamard matrix with random signs, rows and columns permuted alike, /√32).
    new_step() draws a fresh one, like ALTO's hadamard_refresh="step"."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:      # "cuda" and "cuda:0": one matrix
        device = torch.device("cuda", torch.cuda.current_device())
    h = _H.get(device)
    if h is None:
        signs = torch.randint(0, 2, (_H_SIZE,), generator=_H_GEN).float() * 2 - 1
        m = _sylvester(_H_SIZE) * signs                       # H · diag(signs)
        perm = torch.randperm(_H_SIZE, generator=_H_GEN)
        h = (m[perm][:, perm] / math.sqrt(_H_SIZE)).to(device)
        _H[device] = h
    return h


def rotate_tokens(x: torch.Tensor) -> torch.Tensor:
    """H applied to every block of 32 consecutive rows (tokens) of x [T, K], T % 32 == 0, in
    x's dtype (as ALTO). For wgrad, (H·dy)ᵀ(H·x) = dyᵀ·x exactly, since H is orthonormal."""
    t, k = x.shape
    h = hadamard(x.device).to(x.dtype)
    return torch.matmul(h, x.reshape(t // _H_SIZE, _H_SIZE, k)).reshape(t, k)


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[0])) if rows > x.shape[0] else x


# Quantized weights are cached within one training step: weights only change in
# optimizer.step, while a step runs several micro-batches. The cache is off unless
# training.train_step brackets the step with new_step() / end_step(), so evaluation, tests and
# anything else always quantize afresh. Keys use the weight objects' identity, which is only
# safe within a step.
_WCACHE = {}
_CACHE_ON = False


def new_step():
    """Start a training step: clear the quantized-weight cache and enable it, and draw this
    step's Hadamard rotation."""
    global _CACHE_ON
    _WCACHE.clear()
    _H.clear()
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


RECIPES = ("plain", "alto")


def _weight_fwd(w_key, make_w, rule, recipe):
    """Quantized weight for the forward GEMM: 1-D blocks along K (plain) or 32x32 tiles (alto)."""
    if recipe == "alto":
        return _cached((w_key, rule, "tile"), lambda: qdq_tile(make_w(), rule))
    return _cached((w_key, rule, "K"), lambda: qdq(make_w(), rule, dim=-1))


def _weight_dgrad(w_key, make_w, rule, recipe, n_dim):
    """Quantized weight for the dgrad GEMM: 1-D blocks along N (plain) or the same 32x32 tiles
    as the forward (alto: one quantized weight for both, as with ALTO's 2-D weight blocks)."""
    if recipe == "alto":
        return _cached((w_key, rule, "tile"), lambda: qdq_tile(make_w(), rule))
    return _cached((w_key, rule, "N"), lambda: qdq(make_w(), rule, dim=n_dim))


def _wgrad_operands(xp, dyp, rule, recipe):
    """Token-axis operands of the wgrad GEMM (rows already a multiple of 32, expert segments
    32-aligned). alto: random Hadamard along tokens on both, stochastic rounding of dy."""
    if recipe == "alto":
        xp, dyp = rotate_tokens(xp), rotate_tokens(dyp)
    return qdq(xp, rule, dim=0), qdq(dyp, rule, dim=0, sr=recipe == "alto")


def _accumulate_wgrad(w, dw):
    """Weight gradient into Megatron's fp32 main_grad (DDP's grad_added_to_main_grad contract,
    as TE does) or, outside DDP, as a normal grad."""
    if hasattr(w, "main_grad") and getattr(w, "grad_added_to_main_grad", None) is not None:
        w.main_grad.add_(dw.float())
        w.grad_added_to_main_grad = True
        return _dummy_grad(w)
    return dw.to(w.dtype)


class _MXFP4GroupedLinear(torch.autograd.Function):
    """y[e] = Q(x[e]) · Q(W_e)ᵀ for the token segments of each local expert (no bias)."""

    @staticmethod
    def forward(ctx, x, m_splits, rule, recipe, *weights):
        n_tok = sum(m_splits)
        assert x.shape[0] == n_tok, (x.shape, n_tok)
        ctx.m_splits, ctx.rule, ctx.recipe = list(m_splits), rule, recipe
        ctx.save_for_backward(x, *weights)
        xq = qdq(x, rule, dim=-1)                                           # [T, K] along K
        wq = _weight_fwd(id(weights[0]), lambda: torch.stack(weights), rule, recipe)  # [E, N, K]
        offs = _offsets(ctx.m_splits, x.device)
        return torch._grouped_mm(xq, wq.transpose(1, 2), offs=offs)          # [T, N]

    @staticmethod
    def backward(ctx, dy):
        x, *weights = ctx.saved_tensors
        rule, recipe, m_splits = ctx.rule, ctx.recipe, ctx.m_splits
        dy = dy.contiguous()
        offs = _offsets(m_splits, x.device)
        dx = None
        if ctx.needs_input_grad[0]:
            dyq = qdq(dy, rule, dim=-1, sr=recipe == "alto")                # along N
            wq = _weight_dgrad(id(weights[0]), lambda: torch.stack(weights), rule, recipe, 1)
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
        xq_t, dyq_t = _wgrad_operands(xp, dyp, rule, recipe)                 # along tokens
        poffs = _offsets(padded, x.device)
        # bf16 output: _grouped_mm cannot emit fp32 from bf16 inputs on sm90. The extra bf16
        # rounding of dW (~2^-9 relative) is negligible next to the MXFP4 operand error (~15%).
        dw = torch._grouped_mm(dyq_t.t(), xq_t, offs=poffs)                  # [E, N, K]
        grads = [_accumulate_wgrad(w, dw[e]) for e, w in enumerate(weights)]
        return (dx, None, None, None, *grads)


_DUMMY = {}


def _dummy_grad(w):
    """Placeholder grad (like TE's get_dummy_wgrad): DDP's hook needs param.grad to be set, and
    skips adding it because grad_added_to_main_grad is True."""
    key = (tuple(w.shape), w.dtype, w.device)
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(w.shape, dtype=w.dtype, device=w.device)
    return _DUMMY[key]


def grouped_linear(x: torch.Tensor, m_splits, weights, rule: str, recipe: str = "plain"):
    """Emulated-MXFP4 grouped GEMM over the local experts' token segments."""
    return _MXFP4GroupedLinear.apply(x, list(m_splits), rule, recipe, *weights)


def te_grouped_linear_forward(module, x: torch.Tensor, m_splits, rule: str, recipe: str = "plain"):
    """Drop-in for ``TEGroupedLinear(x, m_splits)`` (no bias, TP = 1): returns (output, None)."""
    assert getattr(module, "use_bias", False) is False, "MXFP4 emulation supports bias-free experts"
    weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
    return grouped_linear(x, m_splits, weights, rule, recipe), None


class _MXFP4Linear(torch.autograd.Function):
    """y = Q(x) · Q(W)ᵀ for x [T, K], W [N, K] (no bias)."""

    @staticmethod
    def forward(ctx, x, w, rule, recipe):
        ctx.rule, ctx.recipe = rule, recipe
        ctx.save_for_backward(x, w)
        wq = _weight_fwd(id(w), lambda: w, rule, recipe)
        return qdq(x, rule, dim=-1) @ wq.t()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        rule, recipe = ctx.rule, ctx.recipe
        dy = dy.contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            wq = _weight_dgrad(id(w), lambda: w, rule, recipe, 0)
            dx = qdq(dy, rule, dim=-1, sr=recipe == "alto") @ wq
        rows = int(math.ceil(x.shape[0] / BLOCK) * BLOCK)
        xq_t, dyq_t = _wgrad_operands(_pad_rows(x, rows), _pad_rows(dy, rows), rule, recipe)
        dw = dyq_t.t() @ xq_t                                                 # along tokens
        return dx, _accumulate_wgrad(w, dw), None, None


def linear(x: torch.Tensor, w: torch.Tensor, rule: str, recipe: str = "plain") -> torch.Tensor:
    """Emulated-MXFP4 x · Wᵀ over the last dim of x (any leading shape)."""
    lead = x.shape[:-1]
    y = _MXFP4Linear.apply(x.reshape(-1, x.shape[-1]), w, rule, recipe)
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


def _te_linear_forward(module, x, rule, recipe, fused_norm):
    if fused_norm:
        x = _fused_norm(module, x)
    return linear(x, module.weight, rule, recipe), None


def apply_to_model(model, rule: str, recipe: str = "plain") -> list:
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
            lambda self, x, _f=fused: _te_linear_forward(self, x, rule, recipe, _f), mod)
        patched.append(name)
    return patched
