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
  outliers   outlier-channel ratio: max / median over hidden units of the
             per-unit RMS of h (the "stripes" of per-channel heatmaps).

Sites: "routed" (MoE experts), "shared" (shared expert), "dense" (dense MLP
layers), "resid" (transformer-layer output), "logits" (first 64 positions of the
LM head output) and "router" (tokens per expert: CV, max/mean, dead experts).

Tags: layer averages under gate-quant/, gate-outliers/, gate-deadzone/, gate-moments/,
residual/, logits/, router/ (e.g. gate-quant/routed/h_mxfp4_flush); a curated subset
per layer under layers-<section>/.../L<nn>.
"""
import math

import torch
import torch.distributed as dist

# ---- low-precision simulators ------------------------------------------------
FORMATS = ("fp8", "mxfp4", "nvfp4")
_BLOCK = {"fp8": 128, "mxfp4": 32, "nvfp4": 16}
_FP8_MAX = 448.0
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# dead-zone floors on |f'(z)|: smallest representable / largest in a block
FLOORS = {"fp8": 2.0 ** -9 / _FP8_MAX, "fp4": 0.5 / 6.0}
_DEAD_UNIT_FRAC = 0.99
_CHUNK = 1 << 24            # elements per chunk (bounds the fp32 temporaries)
_LOGIT_POSITIONS = 64

GLU_SITES = ("routed", "shared", "dense")
GLU_TENSORS = ("z1", "z2", "g", "h", "dz1", "dz2")
QUANT_TENSORS = ("h", "dz2")


def _blocks(x, b):
    pad = (-x.shape[-1]) % b
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, b)


def _round_e2m1(y):
    grid = torch.tensor(_E2M1_GRID, device=y.device, dtype=y.dtype)
    mids = torch.tensor(_E2M1_MIDS, device=y.device, dtype=y.dtype)
    return torch.sign(y) * grid[torch.bucketize(y.abs().clamp(max=6.0), mids)]


def fake_quant(x, fmt, tensor_amax=None):
    """Quantize-dequantize float32 x [..., D] blocked along D. Returns (x_blocks, q_blocks)."""
    xb = _blocks(x, _BLOCK[fmt])
    amax = xb.abs().amax(-1, keepdim=True)
    if fmt == "fp8":
        s = (amax / _FP8_MAX).clamp(min=1e-30)
        q = (xb / s).to(torch.float8_e4m3fn).float() * s
    elif fmt == "mxfp4":
        s = torch.exp2(torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2.0)
        q = _round_e2m1(xb / s) * s
    elif fmt == "nvfp4":
        ta = x.abs().amax() if tensor_amax is None else tensor_amax
        s_t = (ta / (_FP8_MAX * 6.0)).clamp(min=1e-30)
        s_b = (amax / (6.0 * s_t)).to(torch.float8_e4m3fn).float()
        s = torch.where(s_b > 0, s_b, torch.ones_like(s_b)) * s_t
        q = _round_e2m1(xb / s) * s
    else:
        raise ValueError(fmt)
    return xb, q


# ---- tracker -----------------------------------------------------------------
class _Tracker:
    def __init__(self):
        self.active = False
        self.layer = 0
        self.num_slots = 2          # index 0 = model-level (logits), 1..L = layers
        self.num_experts = 1
        self.sums, self.maxs, self.vecs = {}, {}, {}


_T = _Tracker()


def is_active():
    return _T.active


def set_layer(layer_number):
    _T.layer = int(layer_number) if layer_number is not None else 0


def begin_step(active, num_layers, num_experts=None):
    _T.active = bool(active)
    _T.num_slots = int(num_layers) + 2
    _T.num_experts = max(int(num_experts or 1), 1)
    _T.sums.clear(); _T.maxs.clear(); _T.vecs.clear()


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


def _tensor_stats(prefix, layer, x, quant=False):
    x2 = x.detach().reshape(-1, x.shape[-1])
    if x2.numel() == 0:
        return
    tmax = x2.abs().amax().float()
    _max(prefix + "/absmax", layer, tmax)
    _add(prefix + "/count", layer, float(x2.numel()))
    for c in x2.split(max(1, _CHUNK // x2.shape[-1])):
        cf = c.float()
        sq = cf * cf
        _add(prefix + "/sumsq", layer, sq.sum(dtype=torch.float64))
        _add(prefix + "/sum4", layer, (sq * sq).sum(dtype=torch.float64))
        if quant:
            for fmt in FORMATS:
                xb, q = fake_quant(cf, fmt, tmax)
                nz = xb != 0
                _add(f"{prefix}/{fmt}/nonzero", layer, nz.sum(dtype=torch.float64))
                _add(f"{prefix}/{fmt}/flushed", layer, (nz & (q == 0)).sum(dtype=torch.float64))
                _add(f"{prefix}/{fmt}/errsq", layer, (q - xb).pow(2).sum(dtype=torch.float64))


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
            (gp,) = torch.autograd.grad(gate_fn(zg).sum(), zg)
        agp = gp.abs()
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
    rms = (hsq / cnt).sqrt()[live].flatten()
    if rms.numel():
        ratio = rms.max() / rms.median().clamp(min=1e-30)
        _add(f"{site}/h/och_ratio_sum", layer, ratio.double())
        _add(f"{site}/h/och_ratio_n", layer, 1.0)
        _max(f"{site}/h/och_ratio_max", layer, ratio)


def _grad_hook(site, name, layer, grad):
    if _T.active and grad is not None:
        with torch.no_grad():
            _tensor_stats(f"{site}/{name}", layer, grad, quant=name in QUANT_TENSORS)


# ---- recording entry points (all no-ops unless active) -------------------------
def record_glu(site, z1, z2, gate, h, gate_fn, tokens_per_expert=None):
    """Called from the generic GLU paths with z1 (value), z2 (gate input), gate = f(z2), h."""
    if not _T.active:
        return
    layer = _T.layer
    with torch.no_grad():
        _tensor_stats(f"{site}/z1", layer, z1)
        _tensor_stats(f"{site}/z2", layer, z2)
        _tensor_stats(f"{site}/g", layer, gate)
        _tensor_stats(f"{site}/h", layer, h, quant=True)
        _unit_stats(site, layer, z2, h, gate_fn, tokens_per_expert)
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
    mine = (sorted(_T.sums), sorted(_T.maxs), sorted(_T.vecs))
    if not (dist.is_available() and dist.is_initialized()):
        return mine
    allk = [None] * dist.get_world_size()
    dist.all_gather_object(allk, mine)
    return tuple(sorted(set().union(*[k[i] for k in allk])) for i in range(3))


def _reduced():
    ks, km, kv = _union_keys()
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
    return S, M, V


def _metrics(S, M, V):
    """-> {(site, metric): {slot: value}} for every slot with data."""
    out = {}

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
        zc, units = S.get(f"{site}/z2/count"), S.get(f"{site}/units")
        for k in FLOORS:
            dz, du = S.get(f"{site}/z2/deadzone_{k}"), S.get(f"{site}/dead_units_{k}")
            for l in range(len(zc or [])):
                if zc[l] > 0 and dz is not None:
                    put(site, f"deadzone_{k}", l, dz[l] / zc[l])
                if units is not None and units[l] > 0 and du is not None:
                    put(site, f"dead_units_{k}", l, du[l] / units[l])
        n, sm, mx = (S.get(f"{site}/h/och_ratio_n"), S.get(f"{site}/h/och_ratio_sum"),
                     M.get(f"{site}/h/och_ratio_max"))
        for l in range(len(n or [])):
            if n[l] > 0:
                put(site, "h/outlier_channel_ratio", l, sm[l] / n[l])
                put(site, "h/outlier_channel_ratio_max", l, mx[l])
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
    "x/rms", "x/absmax", "load_cv", "dead_experts",
}


def _tag(site, metric):
    """(site, metric) -> (section, leaf)."""
    if site in _SITE_SECTION:
        return _SITE_SECTION[site], metric.replace("x/", "")
    if metric.endswith(("_flush", "_relerr")):
        section = "gate-quant"
    elif metric.startswith(("deadzone", "dead_units")):
        section = "gate-deadzone"
    elif metric.endswith(("crest", "kurtosis")) or "outlier_channel" in metric:
        section = "gate-outliers"
    else:
        section = "gate-moments"
    return section, f"{site}/{metric.replace('/', '_')}"


def reduce_and_log(iteration, writer=None, wandb_writer=None):
    """All ranks must call this on the same iterations. Writes layer-average + per-layer tags."""
    if not _T.active:
        return
    _T.active = False
    S, M, V = _reduced()
    metrics = _metrics(S, M, V)
    scalars = {}
    for (site, metric), per_slot in metrics.items():
        section, leaf = _tag(site, metric)
        # dense MLPs and the LM head occupy a single slot: their average is already per-layer
        if metric in _PER_LAYER and site not in ("dense", "logits"):
            for slot, val in per_slot.items():
                if slot > 0:
                    scalars[f"layers-{section}/{leaf}/L{slot:02d}"] = val
        vals = list(per_slot.values())
        agg = max(vals) if any(metric.endswith(s) or s in metric for s in _MAX_LIKE) \
            else sum(vals) / len(vals)
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
              f"absmax {g('logits/absmax')} | router cv {g('router/load_cv')}",
              flush=True)
    return scalars
