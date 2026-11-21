"""
Per-layer statistics for the activation-gate sweep (--log-glu-stats).

On logging iterations only (the training loop calls begin_step() before and
reduce_and_log() after), the generic GLU paths record forward/backward tensor
statistics, the transformer layers record the residual stream, the MoE router
records expert load and the GPT model records logit magnitudes. Everything is
accumulated locally, all-reduced once per logging step (SUM for sums/counts, MAX
for maxima) and written to TensorBoard / wandb. Nothing about training changes,
and non-logging iterations pay only an `is_active()` check.

For a GLU  h = z1 * f(z2)  (z2 = gate input, f = the gate, h = input of fc2):

  moments    RMS, absmax, crest factor (absmax/RMS), kurtosis E[x^4]/E[x^2]^2
             of z1, z2, f(z2), h and of the gradients dz1, dz2.
  blk32      per-block kurtosis mean(x^4)/mean(x^2)^2 (about 0) of every 1x32 block along the
             last dim (the MXFP4 block), for every tensor above and the residual / logits:
             mean, p50/p90/p99 (from a 248-bin histogram over [1, 32]), max, fraction of
             blocks above 8 and 16, and the mean block absmax/rms. Tags gate-blk32/...
             (residual/blk32_..., logits/blk32_...). Bounded by 32; a Gaussian block gives ~2.8.
  mxfp4alto  MXFP4 exactly as the training emulation applies it (mxfp4_emulation: alto e8m0
             scale from the block amax in the tensor's dtype, 1x32 along the last dim): RN flush
             fraction and relerr, and for stochastic rounding (what the alto recipe uses on
             gradients) the mean probability that a non-zero element becomes 0 and the
             expected relerr, both analytic (no sampling). For h and dz2.
  fp4blk     dead zone relative to each 1x32 block's OWN largest slope |f'(z2)| (the dgrad block
             of dz2) with the real alto scale: fraction of slopes RN would flush, mean SR
             zero probability; plus the mean block-max slope and the fraction of blocks whose max
             slope is > 1 (the fixed-floor deadzone_fp4 assumes it is 1).
  fake-quant low precision is simulated on the bf16 training tensors, blocked
             along the last (hidden-unit) dim as in the forward / dgrad GEMM
             operands: FP8 e4m3 1x128 blocks (DeepSeek-V3), MXFP4 1x32 with an
             E8M0 scale, NVFP4 1x16 with an E4M3 block scale and an FP32 tensor
             scale. Logged: flush-to-zero fraction of non-zero elements and the
             relative L2 error, for h and dz2.
  dead zone  fraction of z2 with |f'(z2)| below the format floor (smallest
             representable / largest, the paper's Fig. 7 convention), and the
             fraction of hidden units (per expert) that are in the dead zone for
             more than 99% of their tokens.
  outliers   outlier-channel ratio: per expert, max / median over its hidden units
             of the per-unit RMS of h (the "stripes" of per-channel heatmaps);
             token-weighted mean over experts with >= 32 tokens (and the worst
             expert as ..._max). Alongside, per expert: fraction of channels with
             RMS > 4x median (how many outliers), p90/median of the channel RMS
             (uniformity of the bulk) and the energy share of the top-1% channels
             (concentration), all token-weighted; and
             max|x| / median|x| of h and dz2 per call (exact max; median of a uniform random
             1M-element subsample), mean and max over calls.
  nvfp4 scale fraction of NVFP4 blocks whose E4M3 block scale underflows to 0
             (whole block lost) or is subnormal (< 2^-6, reduced scale precision).

Once per (site, layer) per logging step, on one expert / up to 2048 tokens:
  gemm       relative error of the actual matmul outputs under each recipe vs an
             fp32 reference: fc2 Fprop  h @ W2^T  (activations 1xB, weights
             FP8 128x128 / MXFP4 1x32 / NVFP4 16x16), fc1 Dgrad  dz @ W1  (NVFP4
             with stochastic rounding on dz), and fc1 Wgrad  dz^T @ x  blocked
             along tokens (NVFP4 = 16x16 random Hadamard + SR on dz, as in the
             NVIDIA recipe; nvfp4_rht = RHT only; nvfp4_plain = neither). The fc1 input x is
             checked by recomputing fc1 in fp32; backward GEMMs are skipped when it
             does not match (e.g. fused-LayerNorm fc1 in the dense layer).
  weights    flush + relative error of W1 (fc1) and W2 (fc2) quantized with the
             weight recipes above (up to 8 local experts).
  alignment  per-unit |cos| between gate and value weight rows of W1 and per-unit
             |corr(z1, z2)| over tokens: the SwiGLU weight alignment that makes the
             output quadratic (Fishman et al., arXiv 2409.12517).

Sites: "routed" (MoE experts), "shared" (shared expert), "dense" (dense MLP
layers), "resid" (transformer-layer output), "logits" (first 64 positions of the
LM head output) and "router" (tokens per expert: CV, max/mean, dead experts).

Tags: layer averages under gate-quant/, gate-outliers/, gate-deadzone/, gate-moments/,
gemm-quant/, weight-quant/, gate-align/, residual/, logits/, router/ (e.g. gate-quant/routed/h_mxfp4_flush); a curated subset
per layer under layers-<section>/.../L<nn>.
"""
import math

import torch
import torch.distributed as dist
import torch.nn.functional as Fn

# ---- low-precision simulators ------------------------------------------------
FORMATS = ("fp8", "mxfp4", "nvfp4")
_BLOCK = {"fp8": 128, "mxfp4": 32, "nvfp4": 16}
_FP8_MAX = 448.0
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# dead-zone floors on |f'(z)|: smallest representable / largest in a block
FLOORS = {"fp8": 2.0 ** -9 / _FP8_MAX, "fp4": 0.5 / 6.0}
_DEAD_UNIT_FRAC = 0.99
_MIN_CH_TOKENS = 32          # experts with fewer tokens are skipped for the outlier-channel ratio
_OCH_K = 4.0                 # a channel counts as an outlier if its RMS > 4x the expert's median
_OCH_TOP = 0.01              # top-1% channels for the energy-share metric
_CHUNK = 1 << 24            # elements per chunk (bounds the fp32 temporaries)
_LOGIT_POSITIONS = 64
_WGT_2D = {"fp8": 128, "nvfp4": 16}   # weight blocks: DeepSeek-V3 128x128, NVFP4 16x16; MXFP4 1x32
_GEMM_ROWS = 2048                      # tokens per GEMM-error sample
_MIN_ROWS = 32
_MAX_WEIGHT_EXPERTS = 8                # local experts per rank for weight / alignment stats
_MEDIAN_SAMPLE = 1 << 20
_FP8_SUBNORMAL = 2.0 ** -6
_BLK = 32                    # block size of the per-block kurtosis (the MXFP4 block)
_BLK_BINS = 248              # histogram of per-block kurtosis over [1, 32], bin width 0.125
_BLK_THRESH = (8.0, 16.0)    # logged: fraction of blocks with kurtosis above these
# per-block max|x| / median|x|: histogram of log2(ratio) over [0, 12.4), bin 0.05 (same bin count as
# the kurtosis histogram); MXFP4 RN flushes |x| < max/16..max/28, so ratio > 16 / > 28 means the
# block's median element is flushed (sometimes / always)
_MM_LOG2_MAX = 12.4
_MM_THRESH = (16.0, 28.0)

GLU_SITES = ("routed", "shared", "dense")
GLU_TENSORS = ("z1", "z2", "g", "h", "dz1", "dz2")
QUANT_TENSORS = ("h", "dz2")


def _blocks(x, b):
    pad = (-x.shape[-1]) % b
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, b)


def blk32_stats(cf):
    """Per-block (1x32 along the last dim, the MXFP4 block) kurtosis and crest of float32 cf.
    kappa = mean(x^4) / mean(x^2)^2 about 0 (the FP4 scale is set from the block absmax, not
    the spread about the mean); crest = absmax / rms. Both are computed on x / absmax so tiny
    gradients do not underflow. All-zero blocks are skipped. 1 <= kappa <= 32, 1 <= crest <= sqrt(32).
    Returns (n_blocks, sum kappa, sum crest, max kappa, histogram of kappa [_BLK_BINS])."""
    xb = _blocks(cf, _BLK)
    amax = xb.abs().amax(-1, keepdim=True)
    live = amax.squeeze(-1) > 0
    y = (xb / amax.clamp(min=1e-30))[live]
    y2 = y * y
    m2, m4 = y2.mean(-1), (y2 * y2).mean(-1)
    kap = m4 / (m2 * m2)
    crest = m2.rsqrt()
    idx = ((kap - 1.0) * (_BLK_BINS / (_BLK - 1.0))).long().clamp(0, _BLK_BINS - 1)
    hist = torch.bincount(idx, minlength=_BLK_BINS).double()
    kmax = kap.max() if kap.numel() else torch.zeros((), device=cf.device)
    return (live.sum(dtype=torch.float64), kap.sum(dtype=torch.float64),
            crest.sum(dtype=torch.float64), kmax.float(), hist)


def blk32_maxmed(cf):
    """Per-block (1x32 along the last dim) max|x| / median|x| of float32 cf, median = mean of the
    16th and 17th smallest |x|. All-zero blocks are skipped; a zero median counts as +inf (top bin
    and above every threshold). Returns (n_blocks, histogram of log2(ratio) [_BLK_BINS],
    {threshold: number of blocks with ratio > threshold})."""
    a = _blocks(cf, _BLK).abs()
    live = a.amax(-1) > 0
    s = a[live].sort(-1).values
    med = 0.5 * (s[:, _BLK // 2 - 1] + s[:, _BLK // 2])
    r = torch.where(med > 0, s[:, -1] / med.clamp(min=1e-38), torch.full_like(med, float("inf")))
    idx = (torch.log2(r) * (_BLK_BINS / _MM_LOG2_MAX)).clamp(0, _BLK_BINS - 1).long()
    hist = torch.bincount(idx, minlength=_BLK_BINS).double()
    return (live.sum(dtype=torch.float64), hist,
            {t: (r > t).sum(dtype=torch.float64) for t in _MM_THRESH})


def _mx_scaled(x):
    """x [..., D] (training dtype) -> (blocks of x in fp32, blocks of x / 2^e) with the e8m0
    exponent e of the training emulation (scale rule "alto", 1x32 along D, amax in x's dtype)."""
    from megatron.core.mxfp4_emulation import scale_exponent
    xb = _blocks(x, _BLK)
    e = scale_exponent(xb.abs().amax(-1, keepdim=True), "alto")
    xf = xb.float()
    return xf, xf * torch.exp2(-e), torch.exp2(e)


def mx_round_stats(y):
    """Rounding of scaled MXFP4 blocks y (= x / 2^e) as the training emulation does it:
    elements clamp at 6, then round onto {0, .5, 1, 1.5, 2, 3, 4, 6}, to nearest (forward) or
    stochastically (gradients, recipe alto). Per element, in units of the block scale:
      rn_flush  RN sends a non-zero value to 0 (|y| <= 0.25, a tie rounds to the even code 0)
      sr_pzero  probability that SR sends it to 0: 1 - |y| / 0.5 for |y| < 0.5, else 0
      sr_var    SR variance (a - lo)(hi - a) plus the clamp error above 6 (SR is unbiased
                below 6, so this is its whole mean-square error)
      rn_err2   RN squared error
    Returns per-element tensors (nonzero mask, rn_flush mask, sr_pzero, sr_var, rn_err2); the
    last two are in units of scale^2 (the caller multiplies by the block scale squared)."""
    a = y.abs()
    nz = a > 0
    ac = a.clamp(max=6.0)
    step = torch.where(ac < 2, 0.5, torch.where(ac < 4, 1.0, 2.0))
    lo = torch.floor(ac / step) * step
    sr_var = (ac - lo) * (lo + step - ac) + (a - ac) ** 2
    rn = torch.round(ac / step) * step
    rn_err2 = (rn - a) ** 2
    rn_flush = nz & (ac <= 0.25)
    sr_pzero = torch.where(nz & (ac < 0.5), 1.0 - 2.0 * ac, torch.zeros_like(ac))
    return nz, rn_flush, sr_pzero, sr_var, rn_err2


def _hist_quantile(h, q, lo=1.0, hi=float(_BLK)):
    """q-quantile from a histogram over [lo, hi] (linear inside a bin); default: kappa over [1, 32]."""
    tot = sum(h)
    if tot <= 0:
        return None
    w, target, acc = (hi - lo) / len(h), q * tot, 0.0
    for i, c in enumerate(h):
        if c > 0 and acc + c >= target:
            return lo + w * (i + (target - acc) / c)
        acc += c
    return hi


def _round_e2m1(y):
    grid = torch.tensor(_E2M1_GRID, device=y.device, dtype=y.dtype)
    mids = torch.tensor(_E2M1_MIDS, device=y.device, dtype=y.dtype)
    return torch.sign(y) * grid[torch.bucketize(y.abs().clamp(max=6.0).contiguous(), mids)]


def _sr_e2m1(y, gen):
    """Stochastic rounding onto the e2m1 grid (unbiased: E[round(y)] = y for |y| <= 6)."""
    grid = torch.tensor(_E2M1_GRID, device=y.device, dtype=y.dtype)
    a = y.abs().clamp(max=6.0)
    i = (torch.bucketize(a.contiguous(), grid, right=True) - 1).clamp(0, len(_E2M1_GRID) - 1)
    lo, hi = grid[i], grid[(i + 1).clamp(max=len(_E2M1_GRID) - 1)]
    p = torch.where(hi > lo, (a - lo) / (hi - lo), torch.zeros_like(a))
    u = torch.rand(a.shape, device=a.device, dtype=a.dtype, generator=gen)
    return torch.sign(y) * torch.where(u < p, hi, lo)


def _scale_q(xb, fmt, tensor_amax, sr=False, gen=None):
    """Quantize-dequantize blocks xb [..., nblocks, blocksize] (scale per block)."""
    amax = xb.abs().amax(-1, keepdim=True)
    rnd = (lambda y: _sr_e2m1(y, gen)) if sr else _round_e2m1
    if fmt == "fp8":
        s = (amax / _FP8_MAX).clamp(min=1e-30)
        return (xb / s).to(torch.float8_e4m3fn).float() * s
    if fmt == "mxfp4":
        s = torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2.0)
        return rnd(xb / s) * s
    if fmt == "nvfp4":
        s_t = (tensor_amax / (_FP8_MAX * 6.0)).clamp(min=1e-30)
        s_b = (amax / (6.0 * s_t)).to(torch.float8_e4m3fn).float()
        s = torch.where(s_b > 0, s_b, torch.ones_like(s_b)) * s_t
        return rnd(xb / s) * s
    raise ValueError(fmt)


def fake_quant(x, fmt, tensor_amax=None, sr=False, gen=None):
    """Quantize-dequantize float32 x [..., D] blocked along D. Returns (x_blocks, q_blocks)."""
    xb = _blocks(x, _BLOCK[fmt])
    ta = x.abs().amax() if tensor_amax is None else tensor_amax
    return xb, _scale_q(xb, fmt, ta, sr, gen)


def qdq(x, fmt, sr=False, gen=None):
    """Quantize-dequantize along the last dim; float32 result with x's shape."""
    D = x.shape[-1]
    _, q = fake_quant(x.float(), fmt, None, sr, gen)
    return q.reshape(*x.shape[:-1], -1)[..., :D]


def qdq_weight(w, fmt, contract_dim=1):
    """Weight quantize-dequantize: FP8 128x128 / NVFP4 16x16 2D blocks (transpose-invariant),
    MXFP4 1x32 blocks along the GEMM's contraction dim of w."""
    w = w.float()
    if fmt not in _WGT_2D:
        return qdq(w, fmt) if contract_dim == 1 else qdq(w.t(), fmt).t()
    b, (N, K) = _WGT_2D[fmt], w.shape
    wp = Fn.pad(w, (0, (-K) % b, 0, (-N) % b))
    Np, Kp = wp.shape
    blk = wp.reshape(Np // b, b, Kp // b, b).permute(0, 2, 1, 3).reshape(Np // b, Kp // b, b * b)
    q = _scale_q(blk, fmt, w.abs().amax())
    return q.reshape(Np // b, Kp // b, b, b).permute(0, 2, 1, 3).reshape(Np, Kp)[:N, :K]


_HAD = {}


def _rht16(x):
    """Random Hadamard transform (16x16, one fixed random sign vector) along the last dim,
    which must be a multiple of 16. Orthogonal: rht(A) @ rht(B).T == A @ B.T."""
    R = _HAD.get(x.device)
    if R is None:
        H = torch.ones(1, 1)
        while H.shape[0] < 16:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        g = torch.Generator().manual_seed(16)
        d = torch.randint(0, 2, (16,), generator=g).float() * 2 - 1
        R = _HAD[x.device] = ((H / 4.0) * d).to(x.device)       # H @ diag(d)
    return (x.reshape(*x.shape[:-1], -1, 16) @ R.t()).reshape(x.shape)


# ---- tracker -----------------------------------------------------------------
class _Tracker:
    def __init__(self):
        self.active = False
        self.layer = 0
        self.num_slots = 2          # index 0 = model-level (logits), 1..L = layers
        self.num_experts = 1
        self.sums, self.maxs, self.vecs, self.hists = {}, {}, {}, {}
        self.done = set()           # (site, layer) with once-per-step stats already taken
        self.fc1_in = None          # fc1 input of the GLU about to run (see stash_fc1_input)
        self.gen = None             # private RNG for stochastic rounding (training RNG untouched)


_T = _Tracker()


def is_active():
    return _T.active


def set_layer(layer_number):
    _T.layer = int(layer_number) if layer_number is not None else 0


def begin_step(active, num_layers, num_experts=None):
    _T.active = bool(active)
    _T.num_slots = int(num_layers) + 2
    _T.num_experts = max(int(num_experts or 1), 1)
    _T.sums.clear(); _T.maxs.clear(); _T.vecs.clear(); _T.hists.clear()
    _T.done.clear(); _T.fc1_in = None


def _gen():
    if _T.gen is None:
        _T.gen = torch.Generator(device="cuda")
        _T.gen.manual_seed(20260926 + (dist.get_rank() if dist.is_initialized() else 0))
    return _T.gen


def _slot(layer):
    return min(max(int(layer), 0), _T.num_slots - 1)


def _add(key, layer, v):
    t = _T.sums.get(key)
    if t is None:
        t = _T.sums[key] = torch.zeros(_T.num_slots, dtype=torch.float64, device="cuda")
    t[_slot(layer)] += v


def _max(key, layer, v):
    t = _T.maxs.get(key)
    if t is None:
        t = _T.maxs[key] = torch.zeros(_T.num_slots, dtype=torch.float64, device="cuda")
    i = _slot(layer)
    t[i] = torch.maximum(t[i], torch.as_tensor(v, dtype=torch.float64, device=t.device))


def _vec(key, layer, v):
    t = _T.vecs.get(key)
    if t is None:
        t = _T.vecs[key] = torch.zeros(_T.num_slots, _T.num_experts, dtype=torch.float64,
                                       device="cuda")
    n = min(v.numel(), _T.num_experts)
    t[_slot(layer), :n] += v[:n].double()


def _hist(key, layer, h):
    t = _T.hists.get(key)
    if t is None:
        t = _T.hists[key] = torch.zeros(_T.num_slots, _BLK_BINS, dtype=torch.float64,
                                        device="cuda")
    t[_slot(layer)] += h


def _tensor_stats(prefix, layer, x, quant=False):
    x2 = x.detach().reshape(-1, x.shape[-1])
    if x2.numel() == 0:
        return
    tmax = x2.abs().amax().float()
    _max(prefix + "/absmax", layer, tmax)
    _add(prefix + "/count", layer, float(x2.numel()))
    if quant:
        flat = x2.reshape(-1)
        if flat.numel() > _MEDIAN_SAMPLE:   # uniform random subsample: a fixed stride aliases
            flat = flat[torch.randint(flat.numel(), (_MEDIAN_SAMPLE,), device=flat.device,
                                      generator=_gen())]      # with the row length (columns)
        med = flat.abs().float().median()
        if med > 0:
            r = tmax / med
            _add(prefix + "/mom_sum", layer, r.double())
            _add(prefix + "/mom_n", layer, 1.0)
            _max(prefix + "/mom_max", layer, r)
    for c in x2.split(max(1, _CHUNK // x2.shape[-1])):
        cf = c.float()
        sq = cf * cf
        _add(prefix + "/sumsq", layer, sq.sum(dtype=torch.float64))
        _add(prefix + "/sum4", layer, (sq * sq).sum(dtype=torch.float64))
        del sq
        n, ks, cs, km, hb = blk32_stats(cf)
        _add(prefix + "/blk/n", layer, n)
        _add(prefix + "/blk/ksum", layer, ks)
        _add(prefix + "/blk/crsum", layer, cs)
        _max(prefix + "/blk/kmax", layer, km)
        _hist(prefix + "/blk/hist", layer, hb)
        if quant:
            nm, hm, gt = blk32_maxmed(cf)
            _add(prefix + "/mm/n", layer, nm)
            _hist(prefix + "/mm/hist", layer, hm)
            for t, v in gt.items():
                _add(f"{prefix}/mm/gt{t:g}", layer, v)
            # MXFP4 as the training emulation does it (alto scale, RN forward / SR gradients)
            _, y, s = _mx_scaled(c)
            nz, fl, pz, var, e2 = mx_round_stats(y)
            s2 = s * s
            _add(f"{prefix}/mxa/nonzero", layer, nz.sum(dtype=torch.float64))
            _add(f"{prefix}/mxa/rn_flushed", layer, fl.sum(dtype=torch.float64))
            _add(f"{prefix}/mxa/sr_pzero", layer, pz.sum(dtype=torch.float64))
            _add(f"{prefix}/mxa/sr_errsq", layer, (var * s2).sum(dtype=torch.float64))
            _add(f"{prefix}/mxa/rn_errsq", layer, (e2 * s2).sum(dtype=torch.float64))
            del y, nz, fl, pz, var, e2
            for fmt in FORMATS:
                xb, q = fake_quant(cf, fmt, tmax)
                nz = xb != 0
                _add(f"{prefix}/{fmt}/nonzero", layer, nz.sum(dtype=torch.float64))
                _add(f"{prefix}/{fmt}/flushed", layer, (nz & (q == 0)).sum(dtype=torch.float64))
                _add(f"{prefix}/{fmt}/errsq", layer, (q - xb).pow(2).sum(dtype=torch.float64))
                if fmt == "nvfp4":
                    bmax = xb.abs().amax(-1)
                    s_b = (bmax / (6.0 * (tmax / (_FP8_MAX * 6.0)).clamp(min=1e-30))).to(
                        torch.float8_e4m3fn).float()
                    live = bmax > 0
                    _add(f"{prefix}/nvfp4/blocks", layer, live.sum(dtype=torch.float64))
                    _add(f"{prefix}/nvfp4/scale_underflow", layer,
                         (live & (s_b == 0)).sum(dtype=torch.float64))
                    _add(f"{prefix}/nvfp4/scale_subnormal", layer,
                         (live & (s_b > 0) & (s_b < _FP8_SUBNORMAL)).sum(dtype=torch.float64))


def _unit_stats(site, layer, z2, h, gate_fn, tokens_per_expert):
    """Dead-zone / dead-unit fractions from f'(z2), and the outlier-channel ratio of h."""
    F_ = z2.shape[-1]
    z2d, hd = z2.detach().reshape(-1, F_), h.detach().reshape(-1, F_)
    N, dev = z2d.shape[0], z2d.device
    if N == 0:
        return
    if tokens_per_expert is not None:
        counts = torch.as_tensor(tokens_per_expert, device=dev).long().flatten()
    else:
        counts = torch.tensor([N], device=dev, dtype=torch.long)
    eid = torch.repeat_interleave(torch.arange(counts.numel(), device=dev), counts)[:N]
    if eid.numel() < N:                     # rows past sum(counts) (padding): drop
        N = eid.numel()
    E_ = counts.numel()
    dead = {k: torch.zeros(E_, F_, device=dev) for k in FLOORS}
    hsq = torch.zeros(E_, F_, device=dev)
    rows = max(1, _CHUNK // F_)
    for s in range(0, N, rows):
        ec = eid[s:s + rows]
        with torch.enable_grad():
            zg = z2d[s:s + rows].float().requires_grad_(True)
            gv = gate_fn(zg, expert_ids=ec) if getattr(gate_fn, "per_expert_gate", False) \
                else gate_fn(zg)
            (gp,) = torch.autograd.grad(gv.sum(), zg)
        agp = gp.abs()
        # dead zone relative to each block's own largest slope (1x32 along the hidden units of
        # one token = the dgrad block layout of dz2), with the real e8m0 scale: RN flush and
        # the probability that SR sends the slope factor to 0. Plus the mean block-max slope.
        _, ys, _ = _mx_scaled(gp)
        nzs, fls, pzs, _, _ = mx_round_stats(ys)
        _add(f"{site}/z2/fpblk_n", layer, nzs.sum(dtype=torch.float64))
        _add(f"{site}/z2/fpblk_rn", layer, fls.sum(dtype=torch.float64))
        _add(f"{site}/z2/fpblk_sr", layer, pzs.sum(dtype=torch.float64))
        bmax = _blocks(agp, _BLK).amax(-1)
        _add(f"{site}/z2/fpblk_max_sum", layer, bmax.sum(dtype=torch.float64))
        _add(f"{site}/z2/fpblk_blocks", layer, float(bmax.numel()))
        _add(f"{site}/z2/fpblk_max_gt1", layer, (bmax > 1).sum(dtype=torch.float64))
        del ys, nzs, fls, pzs, bmax
        for k, fl in FLOORS.items():
            m = (agp < fl).float()
            _add(f"{site}/z2/deadzone_{k}", layer, m.sum(dtype=torch.float64))
            dead[k].index_add_(0, ec, m)
        hc = hd[s:s + rows].float()
        hsq.index_add_(0, ec, hc * hc)
    live = counts > 0
    cnt = counts.clamp(min=1).float().unsqueeze(1)
    _add(f"{site}/units", layer, (live.sum() * F_).double())
    for k in FLOORS:
        is_dead = (dead[k] / cnt > _DEAD_UNIT_FRAC) & live.unsqueeze(1)
        _add(f"{site}/dead_units_{k}", layer, is_dead.sum(dtype=torch.float64))
    # outlier-channel ratio WITHIN each expert (channels of different experts never share a
    # quantization block), skipping experts whose per-channel RMS rests on too few tokens.
    elig = counts >= _MIN_CH_TOKENS
    if elig.any():
        rms = (hsq[elig] / cnt[elig]).sqrt()                            # [E_elig, F]
        ratio = rms.max(1).values / rms.median(1).values.clamp(min=1e-30)
        w = counts[elig].double()
        _add(f"{site}/h/och_ratio_wsum", layer, (ratio.double() * w).sum())
        _add(f"{site}/h/och_ratio_w", layer, w.sum())
        _max(f"{site}/h/och_ratio_max", layer, ratio.max())
        # how many outlier channels, how uniform the bulk is, how concentrated the energy is
        med = rms.median(1, keepdim=True).values.clamp(min=1e-30)
        e = rms * rms
        k = max(1, math.ceil(_OCH_TOP * F_))
        extra = {
            "outlier_frac": (rms > _OCH_K * med).float().mean(1),
            "p90_over_median": torch.quantile(rms, 0.9, dim=1) / med.squeeze(1),
            "top1pct_energy": e.topk(k, dim=1).values.sum(1) / e.sum(1).clamp(min=1e-30),
        }
        for name, v in extra.items():
            _add(f"{site}/h/och_{name}_wsum", layer, (v.double() * w).sum())


def _get_weights(module, e):
    """(W1 [2F, K], W2 [K', F]) of local expert e (grouped experts) or of an MLP, as fp32."""
    out = []
    for name in ("linear_fc1", "linear_fc2"):
        lin, w = getattr(module, name, None), None
        if lin is not None:
            w = getattr(lin, f"weight{e}", None) if e is not None else None
            if w is None:
                w = getattr(lin, "weight", None)
        try:
            out.append(w.detach().float() if torch.is_tensor(w) and w.dim() == 2 else None)
        except Exception:  # noqa: BLE001  (e.g. a quantized parameter type)
            out.append(None)
    return out


def _stat(key, layer, vals):
    _add(key + "/sum", layer, vals.sum(dtype=torch.float64))
    _add(key + "/n", layer, float(vals.numel()))
    _max(key + "/max", layer, vals.max())


def _gemm_err(key, layer, ref, approx):
    _add(key + "/errsq", layer, (approx - ref).pow(2).sum(dtype=torch.float64))
    _add(key + "/refsq", layer, ref.pow(2).sum(dtype=torch.float64))


def _weight_q(prefix, layer, w, contract_dim):
    nz = w != 0
    for fmt in FORMATS:
        q = qdq_weight(w, fmt, contract_dim)
        _add(f"{prefix}_{fmt}/nonzero", layer, nz.sum(dtype=torch.float64))
        _add(f"{prefix}_{fmt}/flushed", layer, (nz & (q == 0)).sum(dtype=torch.float64))
        _add(f"{prefix}_{fmt}/errsq", layer, (q - w).pow(2).sum(dtype=torch.float64))
        _add(f"{prefix}_{fmt}/refsq", layer, w.pow(2).sum(dtype=torch.float64))


def _colcorr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-30)


def _deep_stats(site, layer, z1, z2, h, fc1_out, fc1_in, module, tokens_per_expert):
    F_ = z2.shape[-1]
    z1d, z2d, hd = (t.detach().reshape(-1, F_) for t in (z1, z2, h))
    N = z2d.shape[0]
    grouped = tokens_per_expert is not None
    counts = [int(c) for c in torch.as_tensor(tokens_per_expert).flatten().tolist()] \
        if grouped else [N]
    starts = [sum(counts[:i]) for i in range(len(counts))]
    # weight alignment + weight quantization over up to 8 local experts
    for e in (range(min(len(counts), _MAX_WEIGHT_EXPERTS)) if grouped else [None]):
        w1, w2 = _get_weights(module, e) if module is not None else (None, None)
        if w1 is not None and w1.shape[0] == 2 * F_:
            g, v = w1[:F_], w1[F_:]
            cos = (g * v).sum(1) / (g.norm(dim=1) * v.norm(dim=1)).clamp(min=1e-30)
            _stat(f"{site}/align/w_cos_abs", layer, cos.abs())
            _stat(f"{site}/align/w_cos", layer, cos)
            _weight_q(f"{site}/wq/fc1", layer, w1, 1)
        if w2 is not None and w2.shape[1] == F_:
            _weight_q(f"{site}/wq/fc2", layer, w2, 1)
    # GEMM errors + activation correlation on one expert's tokens
    cand = [i for i, c in enumerate(counts) if c >= _MIN_ROWS and starts[i] + c <= N]
    if not cand:
        return
    e = cand[0]
    rows = slice(starts[e], starts[e] + min(counts[e], _GEMM_ROWS))
    _stat(f"{site}/align/z_corr_abs", layer,
          _colcorr(z1d[rows].float(), z2d[rows].float()).abs())
    w1, w2 = _get_weights(module, e if grouped else None) if module is not None else (None, None)
    if w2 is not None and w2.shape[1] == F_:
        hs = hd[rows].float()
        ref = hs @ w2.t()
        for fmt in FORMATS:
            _gemm_err(f"{site}/gemm/fc2_fprop_{fmt}", layer, ref,
                      qdq(hs, fmt) @ qdq_weight(w2, fmt, 1).t())
    if (w1 is None or w1.shape[0] != 2 * F_ or fc1_out is None or fc1_in is None
            or not fc1_out.requires_grad or fc1_in.shape[-1] != w1.shape[1]):
        return
    xin = fc1_in.detach().reshape(-1, fc1_in.shape[-1])
    if xin.shape[0] != N:
        return
    xin = xin[rows].float()
    out = fc1_out.detach().reshape(-1, 2 * F_)[rows].float()
    mism = ((xin @ w1.t() - out).norm() / out.norm().clamp(min=1e-30)).item()
    if mism > 0.02:                 # stashed input is not the real GEMM input: skip
        _add(f"{site}/gemm/skipped", layer, 1.0)
        return
    fc1_out.register_hook(
        lambda g, s=site, l=layer, r=rows, x=xin, w=w1: _bwd_gemm_hook(s, l, r, x, w, g))


def _bwd_gemm_hook(site, layer, rows, xin, w1, grad):
    if not _T.active or grad is None:
        return
    with torch.no_grad():
        dz = grad.detach().reshape(-1, grad.shape[-1])[rows].float()      # [n, 2F]
        ref = dz @ w1                                                     # Dgrad, contract 2F
        for fmt in FORMATS:
            A = qdq(dz, fmt, sr=fmt == "nvfp4", gen=_gen())
            _gemm_err(f"{site}/gemm/fc1_dgrad_{fmt}", layer, ref, A @ qdq_weight(w1, fmt, 0))
        pad = (-dz.shape[0]) % 16                                         # Wgrad, contract tokens
        dzT, xT = Fn.pad(dz.t(), (0, pad)), Fn.pad(xin.t(), (0, pad))
        ref = dzT @ xT.t()
        # nvfp4 = full NVIDIA recipe (RHT + SR on dz); nvfp4_rht = RHT only; nvfp4_plain = neither.
        # SR raises the single-sample error but removes bias across steps, so compare the three.
        for name, fmt, rht, sr in (("fp8", "fp8", False, False), ("mxfp4", "mxfp4", False, False),
                                   ("nvfp4", "nvfp4", True, True),
                                   ("nvfp4_rht", "nvfp4", True, False),
                                   ("nvfp4_plain", "nvfp4", False, False)):
            a, b = (_rht16(dzT), _rht16(xT)) if rht else (dzT, xT)
            approx = qdq(a, fmt, sr=sr, gen=_gen()) @ qdq(b, fmt).t()
            _gemm_err(f"{site}/gemm/fc1_wgrad_{name}", layer, ref, approx)


def _gate_params(site, layer, gate_mod, owner):
    """Learned gate coefficients (one per local expert), placed at their global expert index so
    the SUM-reduce yields every expert's value once per replica: mean / std / min / max."""
    E_loc = gate_mod.num_local_experts
    off = 0
    if site == "routed" and getattr(owner, "ep_group", None) is not None:
        off = owner.ep_group.rank() * E_loc
    E = max(_T.num_experts, off + E_loc)
    if E > _T.num_experts:                  # dense / shared MLP before begin_step knew experts
        return
    for name, v in gate_mod.coefficients().items():
        buf = torch.zeros(E, dtype=torch.float64, device=v.device)
        cnt = torch.zeros_like(buf)
        buf[off:off + E_loc] = v.detach().double().flatten()
        cnt[off:off + E_loc] = 1.0
        _vec(f"{site}/param/{name}/sum", layer, buf)
        _vec(f"{site}/param/{name}/n", layer, cnt)


def _grad_hook(site, name, layer, grad):
    if _T.active and grad is not None:
        with torch.no_grad():
            _tensor_stats(f"{site}/{name}", layer, grad, quant=name in QUANT_TENSORS)


# ---- recording entry points (all no-ops unless active) -------------------------
def stash_fc1_input(x):
    """Called right before fc1 so the next record_glu() knows the GEMM input."""
    if _T.active:
        _T.fc1_in = x


def record_glu(site, z1, z2, gate, h, gate_fn, tokens_per_expert=None, fc1_out=None,
               module=None):
    """Called from the generic GLU paths with z1 (value), z2 (gate input), gate = f(z2), h,
    the full fc1 output [z2 | z1] and the owning MLP / experts module (for its weights)."""
    if not _T.active:
        return
    layer = _T.layer
    fc1_in, _T.fc1_in = _T.fc1_in, None
    with torch.no_grad():
        _tensor_stats(f"{site}/z1", layer, z1)
        _tensor_stats(f"{site}/z2", layer, z2)
        _tensor_stats(f"{site}/g", layer, gate)
        _tensor_stats(f"{site}/h", layer, h, quant=True)
        _unit_stats(site, layer, z2, h, gate_fn, tokens_per_expert)
        if (site, layer) not in _T.done:
            _T.done.add((site, layer))
            if getattr(gate_fn, "per_expert_gate", False):
                _gate_params(site, layer, gate_fn, module)
            _deep_stats(site, layer, z1, z2, h, fc1_out, fc1_in, module, tokens_per_expert)
    if z2.requires_grad:
        z2.register_hook(lambda g, s=site, l=layer: _grad_hook(s, "dz2", l, g))
    if z1.requires_grad:
        z1.register_hook(lambda g, s=site, l=layer: _grad_hook(s, "dz1", l, g))


def record_moments(prefix, x, layer=None):
    if not _T.active or not torch.is_tensor(x):
        return
    with torch.no_grad():
        _tensor_stats(prefix, _T.layer if layer is None else layer, x)


def record_logits(logits):
    if _T.active and torch.is_tensor(logits):
        record_moments("logits", logits[:_LOGIT_POSITIONS], layer=0)


def record_router(layer_number, routing_map):
    if not _T.active or routing_map is None:
        return
    with torch.no_grad():
        _vec("router/tokens_per_expert", layer_number or 0, routing_map.sum(0))


# ---- reduction + logging -----------------------------------------------------
def _union_keys():
    mine = (sorted(_T.sums), sorted(_T.maxs), sorted(_T.vecs), sorted(_T.hists))
    if not (dist.is_available() and dist.is_initialized()):
        return mine
    allk = [None] * dist.get_world_size()
    dist.all_gather_object(allk, mine)
    return tuple(sorted(set().union(*[k[i] for k in allk])) for i in range(4))


def _reduced():
    ks, km, kv, kh = _union_keys()
    for k in kh:
        if k not in _T.hists:
            _T.hists[k] = torch.zeros(_T.num_slots, _BLK_BINS, dtype=torch.float64, device="cuda")
    if kh:
        flat_h = torch.cat([_T.hists[k].flatten() for k in kh])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(flat_h, op=dist.ReduceOp.SUM)
        fh, L = flat_h.cpu().tolist(), _T.num_slots
        H = {k: [fh[(j * L + l) * _BLK_BINS:(j * L + l + 1) * _BLK_BINS] for l in range(L)]
             for j, k in enumerate(kh)}
    else:
        H = {}
    for k in ks:
        if k not in _T.sums:
            _T.sums[k] = torch.zeros(_T.num_slots, dtype=torch.float64, device="cuda")
    for k in km:
        if k not in _T.maxs:
            _T.maxs[k] = torch.zeros(_T.num_slots, dtype=torch.float64, device="cuda")
    for k in kv:
        if k not in _T.vecs:
            _T.vecs[k] = torch.zeros(_T.num_slots, _T.num_experts, dtype=torch.float64,
                                     device="cuda")
    flat_s = torch.cat([_T.sums[k] for k in ks] + [_T.vecs[k].flatten() for k in kv]) \
        if (ks or kv) else torch.zeros(0, dtype=torch.float64, device="cuda")
    flat_m = torch.cat([_T.maxs[k] for k in km]) if km else \
        torch.zeros(0, dtype=torch.float64, device="cuda")
    if dist.is_available() and dist.is_initialized():
        if flat_s.numel():
            dist.all_reduce(flat_s, op=dist.ReduceOp.SUM)
        if flat_m.numel():
            dist.all_reduce(flat_m, op=dist.ReduceOp.MAX)
    S, M, V = {}, {}, {}
    fs, fm = flat_s.cpu().tolist(), flat_m.cpu().tolist()
    L, E, o = _T.num_slots, _T.num_experts, 0
    for k in ks:
        S[k] = fs[o:o + L]; o += L
    for k in kv:
        V[k] = [fs[o + i * E:o + (i + 1) * E] for i in range(L)]; o += L * E
    for i, k in enumerate(km):
        M[k] = fm[i * L:(i + 1) * L]
    V["__hist__"] = H
    return S, M, V


def _metrics(S, M, V):
    """-> {(site, metric): {slot: value}} for every slot with data."""
    out = {}
    H = V.pop("__hist__", {})

    def put(site, metric, slot, val):
        if val is not None and math.isfinite(val):
            out.setdefault((site, metric), {})[slot] = val

    def moments(site, tensor, prefix):
        cnt = S.get(prefix + "/count")
        if cnt is None:
            return
        for l, c in enumerate(cnt):
            if c <= 0:
                continue
            ms = S[prefix + "/sumsq"][l] / c
            rms = math.sqrt(ms)
            amax = M.get(prefix + "/absmax", [0.0] * len(cnt))[l]
            put(site, f"{tensor}/rms", l, rms)
            put(site, f"{tensor}/absmax", l, amax)
            if rms > 0:
                put(site, f"{tensor}/crest", l, amax / rms)
                put(site, f"{tensor}/kurtosis", l, (S[prefix + "/sum4"][l] / c) / ms ** 2)
            nb = S.get(prefix + "/blk/n")
            if nb is not None and nb[l] > 0:          # per-block (1x32) kurtosis / crest
                put(site, f"{tensor}/blk32_kurtosis", l, S[prefix + "/blk/ksum"][l] / nb[l])
                put(site, f"{tensor}/blk32_amax_over_rms", l, S[prefix + "/blk/crsum"][l] / nb[l])
                put(site, f"{tensor}/blk32_kurtosis_max", l, M[prefix + "/blk/kmax"][l])
                h = H.get(prefix + "/blk/hist")
                if h is not None:
                    for q, name in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
                        put(site, f"{tensor}/blk32_kurtosis_{name}", l, _hist_quantile(h[l], q))
                    w = (_BLK - 1.0) / _BLK_BINS
                    tot = sum(h[l])
                    for t in _BLK_THRESH:          # thresholds sit on bin edges
                        i0 = int(round((t - 1.0) / w))
                        put(site, f"{tensor}/blk32_frac_kurt_gt{t:g}", l, sum(h[l][i0:]) / tot)
            nm = S.get(prefix + "/mm/n")              # per-block max / median
            if nm is not None and nm[l] > 0:
                h = H.get(prefix + "/mm/hist")
                if h is not None:
                    for q, name in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
                        v = _hist_quantile(h[l], q, 0.0, _MM_LOG2_MAX)
                        put(site, f"{tensor}/blk32_max_over_median_{name}", l,
                            None if v is None else 2.0 ** v)
                for t in _MM_THRESH:
                    put(site, f"{tensor}/blk32_frac_maxmed_gt{t:g}", l, S[f"{prefix}/mm/gt{t:g}"][l] / nm[l])
            mn = S.get(prefix + "/mom_n")
            if mn is not None and mn[l] > 0:
                put(site, f"{tensor}/max_over_median", l, S[prefix + "/mom_sum"][l] / mn[l])
                put(site, f"{tensor}/max_over_median_max", l, M[prefix + "/mom_max"][l])
            blk = S.get(prefix + "/nvfp4/blocks")
            if blk is not None and blk[l] > 0:
                for k in ("scale_underflow", "scale_subnormal"):
                    put(site, f"{tensor}/nvfp4_{k}", l, S[f"{prefix}/nvfp4/{k}"][l] / blk[l])
            nz = S.get(f"{prefix}/mxa/nonzero")     # MXFP4 exactly as trained (alto scale)
            if nz is not None and nz[l] > 0:
                put(site, f"{tensor}/mxfp4alto_rn_flush", l, S[f"{prefix}/mxa/rn_flushed"][l] / nz[l])
                put(site, f"{tensor}/mxfp4alto_sr_zero_prob", l, S[f"{prefix}/mxa/sr_pzero"][l] / nz[l])
                if S[prefix + "/sumsq"][l] > 0:
                    for r in ("rn", "sr"):
                        put(site, f"{tensor}/mxfp4alto_{r}_relerr", l,
                            math.sqrt(S[f"{prefix}/mxa/{r}_errsq"][l] / S[prefix + "/sumsq"][l]))
            for fmt in FORMATS:
                nz = S.get(f"{prefix}/{fmt}/nonzero")
                if nz is None or nz[l] <= 0:
                    continue
                put(site, f"{tensor}/{fmt}_flush", l, S[f"{prefix}/{fmt}/flushed"][l] / nz[l])
                if S[prefix + "/sumsq"][l] > 0:
                    put(site, f"{tensor}/{fmt}_relerr", l,
                        math.sqrt(S[f"{prefix}/{fmt}/errsq"][l] / S[prefix + "/sumsq"][l]))

    for site in GLU_SITES:
        for t in GLU_TENSORS:
            moments(site, t, f"{site}/{t}")
        fn_, fb_ = S.get(f"{site}/z2/fpblk_n"), S.get(f"{site}/z2/fpblk_blocks")
        for l in range(len(fn_ or [])):
            if fn_[l] > 0:
                put(site, "deadzone_fp4blk_rn", l, S[f"{site}/z2/fpblk_rn"][l] / fn_[l])
                put(site, "deadzone_fp4blk_sr", l, S[f"{site}/z2/fpblk_sr"][l] / fn_[l])
            if fb_ is not None and fb_[l] > 0:
                put(site, "deadzone_ref_slope_blkmax", l, S[f"{site}/z2/fpblk_max_sum"][l] / fb_[l])
                put(site, "deadzone_ref_frac_blkmax_gt1", l, S[f"{site}/z2/fpblk_max_gt1"][l] / fb_[l])
        zc, units = S.get(f"{site}/z2/count"), S.get(f"{site}/units")
        for k in FLOORS:
            dz, du = S.get(f"{site}/z2/deadzone_{k}"), S.get(f"{site}/dead_units_{k}")
            for l in range(len(zc or [])):
                if zc[l] > 0 and dz is not None:
                    put(site, f"deadzone_{k}", l, dz[l] / zc[l])
                if units is not None and units[l] > 0 and du is not None:
                    put(site, f"dead_units_{k}", l, du[l] / units[l])
        n, sm, mx = (S.get(f"{site}/h/och_ratio_w"), S.get(f"{site}/h/och_ratio_wsum"),
                     M.get(f"{site}/h/och_ratio_max"))
        for l in range(len(n or [])):
            if n[l] > 0:
                put(site, "h/outlier_channel_ratio", l, sm[l] / n[l])
                put(site, "h/outlier_channel_ratio_max", l, mx[l])
                for name in ("outlier_frac", "p90_over_median", "top1pct_energy"):
                    v = S.get(f"{site}/h/och_{name}_wsum")
                    if v is not None:
                        put(site, f"h/och_{name}", l, v[l] / n[l])
        for key in S:
            if not key.startswith(site + "/"):
                continue
            rest = key[len(site) + 1:]
            for l in range(len(S[key])):
                if rest.startswith(("gemm/", "wq/")) and rest.endswith("/errsq"):
                    base = rest[:-len("/errsq")]
                    ref = S[f"{site}/{base}/refsq"][l]
                    if ref > 0:
                        put(site, f"{base}_relerr", l, math.sqrt(S[key][l] / ref))
                    dot, apx = S.get(f"{site}/{base}/dot"), S.get(f"{site}/{base}/apxsq")
                    if dot is not None and ref > 0 and apx[l] > 0:   # emulated MXFP4 GEMMs
                        put(site, f"{base}_cos", l, dot[l] / math.sqrt(ref * apx[l]))
                        put(site, f"{base}_proj", l, dot[l] / ref)
                        if S[key][l] > 0:
                            put(site, f"{base}_snr", l, math.sqrt(ref / S[key][l]))
                    if rest.startswith("wq/") and S[f"{site}/{base}/nonzero"][l] > 0:
                        put(site, f"{base}_flush", l,
                            S[f"{site}/{base}/flushed"][l] / S[f"{site}/{base}/nonzero"][l])
                elif rest.startswith("align/") and rest.endswith("/sum"):
                    base, nn = rest[:-len("/sum")], S[f"{site}/{rest[:-len('/sum')]}/n"][l]
                    if nn > 0:
                        put(site, f"{base}_mean", l, S[key][l] / nn)
                        mk = M.get(f"{site}/{base}/max")
                        if mk is not None and base.endswith("_abs"):
                            put(site, f"{base}_max", l, mk[l])
                elif rest == "gemm/skipped" and S[key][l] > 0:
                    put(site, "gemm/skipped_samples", l, S[key][l])
        for key in V:                       # learned gate coefficients (exp_pglu / exp_xrlglu)
            if not (key.startswith(site + "/param/") and key.endswith("/sum")):
                continue
            name = key[len(site) + 7:-4]
            for l, (vs, ns) in enumerate(zip(V[key], V[key[:-4] + "/n"])):
                vals = [v / n for v, n in zip(vs, ns) if n > 0]
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                put(site, f"param/{name}_mean", l, mean)
                put(site, f"param/{name}_std", l,
                    math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
                put(site, f"param/{name}_min", l, min(vals))
                put(site, f"param/{name}_max", l, max(vals))
    moments("resid", "x", "resid")
    moments("logits", "x", "logits")
    for l, vec in enumerate(V.get("router/tokens_per_expert", [])):
        tot = sum(vec)
        if tot <= 0:
            continue
        mean = tot / len(vec)
        var = sum((v - mean) ** 2 for v in vec) / len(vec)
        put("router", "load_cv", l, math.sqrt(var) / mean)
        put("router", "load_max_over_mean", l, max(vec) / mean)
        put("router", "dead_experts", l, sum(1 for v in vec if v == 0) / len(vec))
    return out


_MAX_LIKE = ("absmax", "crest", "_max")

# Tag layout: TensorBoard and wandb both group by the first path segment, so each
# topic gets its own small section. Layer averages go to <topic>/..., and a curated
# subset of metrics is also logged per layer to layers-<topic>/.../L<nn>.
_SITE_SECTION = {"resid": "residual", "logits": "logits", "router": "router"}
_PER_LAYER = {
    "h/rms", "h/crest", "h/kurtosis", "h/mxfp4_flush", "h/nvfp4_flush", "dz2/rms",
    "dz2/mxfp4_flush", "deadzone_fp4", "dead_units_fp4", "h/outlier_channel_ratio",
    "x/rms", "x/absmax", "load_cv", "dead_experts", "h/max_over_median",
    "h/nvfp4_scale_underflow", "gemm/fc2_fprop_nvfp4_relerr", "gemm/fc1_dgrad_nvfp4_relerr",
    "gemm/fc1_wgrad_nvfp4_relerr", "wq/fc1_nvfp4_relerr", "wq/fc2_nvfp4_relerr",
    "align/w_cos_abs_mean", "align/z_corr_abs_mean", "h/och_outlier_frac",
    "h/och_p90_over_median", "h/och_top1pct_energy",
    "h/blk32_kurtosis", "h/blk32_kurtosis_p99", "h/blk32_frac_kurt_gt8",
    "dz2/blk32_kurtosis", "dz2/blk32_kurtosis_p99", "dz1/blk32_kurtosis", "x/blk32_kurtosis",
    "deadzone_fp4blk_rn", "deadzone_fp4blk_sr", "deadzone_ref_slope_blkmax",
    "dz2/mxfp4alto_sr_zero_prob", "dz2/mxfp4alto_sr_relerr",
    "h/mxfp4alto_rn_flush", "h/mxfp4alto_rn_relerr",
    *(f"{t}/blk32_{m}" for t in ("h", "dz2") for m in (
        "max_over_median_p50", "max_over_median_p90", "max_over_median_p99",
        "frac_maxmed_gt16", "frac_maxmed_gt28")),
    *(f"gemm/mx_{lin}_{st}_{m}" for lin in ("fc1", "fc2")
      for st in ("fprop", "dgrad", "wgrad", "wgradstep") for m in ("relerr", "cos", "proj", "snr")),
}
_PREFIX_SECTION = {"gemm/": "gemm-quant", "wq/": "weight-quant", "align/": "gate-align",
                   "param/": "learned-gate"}


def _tag(site, metric):
    """(site, metric) -> (section, leaf)."""
    if site in _SITE_SECTION:
        return _SITE_SECTION[site], metric.replace("x/", "")
    for pre, sec in _PREFIX_SECTION.items():
        if metric.startswith(pre):
            return sec, f"{site}/{metric[len(pre):]}"
    if "blk32_" in metric:
        return "gate-blk32", f"{site}/{metric.replace('/', '_')}"
    if metric.endswith(("_flush", "_relerr", "_underflow", "_subnormal", "_zero_prob")):
        section = "gate-quant"
    elif metric.startswith(("deadzone", "dead_units")):
        section = "gate-deadzone"
    elif metric.endswith(("crest", "kurtosis")) or "outlier_channel" in metric \
            or "max_over_median" in metric or metric.startswith("h/och_"):
        section = "gate-outliers"
    else:
        section = "gate-moments"
    return section, f"{site}/{metric.replace('/', '_')}"


def reduce_and_log(iteration, writer=None, wandb_writer=None):
    """All ranks must call this on the same iterations. Writes layer-average + per-layer tags."""
    if not _T.active:
        return
    import sys
    mx = sys.modules.get("megatron.core.mxfp4_emulation")
    if mx is not None:          # full-step weight-gradient error of the emulated MXFP4 GEMMs
        with torch.no_grad():
            mx.flush_step_stats(_add)
    _T.active = False
    S, M, V = _reduced()
    metrics = _metrics(S, M, V)
    scalars = {}
    for (site, metric), per_slot in metrics.items():
        section, leaf = _tag(site, metric)
        # dense MLPs and the LM head occupy a single slot: their average is already per-layer
        per_layer = metric in _PER_LAYER or (metric.startswith("param/") and (
            metric.endswith("_mean") or (site == "routed" and metric.endswith(("_min", "_max")))))
        if per_layer and site not in ("dense", "logits"):
            for slot, val in per_slot.items():
                if slot > 0:
                    scalars[f"layers-{section}/{leaf}/L{slot:02d}"] = val
        vals = list(per_slot.values())
        if metric.startswith("param/") and metric.endswith("_min"):
            agg = min(vals)
        elif any(metric.endswith(s) or s in metric for s in _MAX_LIKE):
            agg = max(vals)
        else:
            agg = sum(vals) / len(vals)
        scalars[f"{section}/{leaf}"] = agg
    if writer is not None:
        for tag, val in scalars.items():
            writer.add_scalar(tag, val, iteration)
    if wandb_writer is not None:
        wandb_writer.log(scalars, iteration)
    rank_last = (not dist.is_initialized()) or dist.get_rank() == dist.get_world_size() - 1
    if rank_last:
        def g(k):
            v = scalars.get(k)
            return "n/a" if v is None else f"{v:.4g}"
        print(f"[glu-stats] iter {iteration} | {len(scalars)} scalars | routed h: rms "
              f"{g('gate-moments/routed/h_rms')} crest {g('gate-outliers/routed/h_crest')} kurt "
              f"{g('gate-outliers/routed/h_kurtosis')} fp8/mxfp4/nvfp4 flush "
              f"{g('gate-quant/routed/h_fp8_flush')}/{g('gate-quant/routed/h_mxfp4_flush')}/"
              f"{g('gate-quant/routed/h_nvfp4_flush')} | routed dz2 mxfp4 flush "
              f"{g('gate-quant/routed/dz2_mxfp4_flush')} | dead units fp4 routed/shared/dense "
              f"{g('gate-deadzone/routed/dead_units_fp4')}/{g('gate-deadzone/shared/dead_units_fp4')}/"
              f"{g('gate-deadzone/dense/dead_units_fp4')} | resid rms {g('residual/rms')} | logits "
              f"absmax {g('logits/absmax')} | router cv {g('router/load_cv')} | nvfp4 GEMM "
              f"relerr fprop/dgrad/wgrad {g('gemm-quant/routed/fc2_fprop_nvfp4_relerr')}/"
              f"{g('gemm-quant/routed/fc1_dgrad_nvfp4_relerr')}/"
              f"{g('gemm-quant/routed/fc1_wgrad_nvfp4_relerr')} (plain "
              f"{g('gemm-quant/routed/fc1_wgrad_nvfp4_plain_relerr')}) | w-align "
              f"{g('gate-align/routed/w_cos_abs_mean')} z-corr "
              f"{g('gate-align/routed/z_corr_abs_mean')} | h max/median "
              f"{g('gate-outliers/routed/h_max_over_median')} | blk32 kurt h mean/p99 "
              f"{g('gate-blk32/routed/h_blk32_kurtosis')}/{g('gate-blk32/routed/h_blk32_kurtosis_p99')}"
              f" dz2 {g('gate-blk32/routed/dz2_blk32_kurtosis')}/"
              f"{g('gate-blk32/routed/dz2_blk32_kurtosis_p99')} | routed deadzone fp4 floor/blk-rn/"
              f"blk-sr {g('gate-deadzone/routed/deadzone_fp4')}/"
              f"{g('gate-deadzone/routed/deadzone_fp4blk_rn')}/{g('gate-deadzone/routed/deadzone_fp4blk_sr')}"
              f" slope blkmax {g('gate-deadzone/routed/deadzone_ref_slope_blkmax')} | dz2 alto SR "
              f"p0 {g('gate-quant/routed/dz2_mxfp4alto_sr_zero_prob')} relerr "
              f"{g('gate-quant/routed/dz2_mxfp4alto_sr_relerr')}",
              flush=True)
    return scalars
