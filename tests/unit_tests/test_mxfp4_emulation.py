"""megatron/core/mxfp4_emulation.py: quantizer rules and the emulated MXFP4 grouped GEMM."""
import time

import pytest
import torch

from megatron.core import mxfp4_emulation as mx

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def ref_qdq_lastdim(x, rule):
    """Independent reference: bucketize onto the e2m1 grid (ties go to the even code)."""
    xb = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = xb.abs().amax(-1, keepdim=True)
    mant, ex = torch.frexp(amax)                             # amax = mant * 2^ex, mant in [0.5, 1)
    e = (ex - 3).float()                                     # floor(log2 amax) - 2
    if rule == "alto":                                       # bump when the significand >= 1.75
        e = e + (2 * mant >= 1.75).float()
    e = torch.where(amax > 0, e, torch.full_like(e, -126.0))
    s = torch.exp2(e.clamp(min=-126))
    y = (xb / s)
    a = y.abs().clamp(max=6.0)
    g, mids = GRID.to(x.device), MIDS.to(x.device)
    i = torch.bucketize(a.contiguous(), mids)                # a == mid -> lower index
    tie = (i < 7) & (a == mids[i.clamp(max=6)])
    i = torch.where(tie & (i % 2 == 1), i + 1, i)           # tie -> even code
    q = torch.sign(y) * g[i]
    return (q * s).to(x.dtype).reshape(x.shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rule", ["floor", "alto"])
def test_qdq_matches_reference(dtype, rule):
    torch.manual_seed(0)
    x = torch.randn(512, 1024, device="cuda") * torch.exp(torch.randn(512, 1, device="cuda") * 2)
    x[torch.rand_like(x) < 1e-3] *= 40
    x[:3, :64] = 0.0                                         # all-zero blocks
    x = x.to(dtype)
    assert torch.equal(mx.qdq(x, rule), ref_qdq_lastdim(x, rule))
    # along dim 0 == transpose, quantize, transpose back
    assert torch.equal(mx.qdq(x, rule, dim=0), ref_qdq_lastdim(x.t().contiguous(), rule).t())


def test_alto_clips_less_than_floor():
    x = torch.zeros(1, 32, device="cuda")
    x[0, 0], x[0, 1] = 7.5, 1.0                              # block max at 7.5 * s
    assert float(mx.qdq(x, "floor")[0, 0]) == 6.0            # clipped
    assert float(mx.qdq(x, "alto")[0, 0]) == 8.0             # next scale, rounded up


def ref_grouped(x, m_splits, ws, rule, dy):
    """Per-expert loop with explicit quantization of every GEMM operand, fp32 math."""
    ys, dxs, dws, o = [], [], [], 0
    for m, w in zip(m_splits, ws):
        xe, dye = x[o:o + m].float(), dy[o:o + m].float()
        o += m
        ys.append(mx.qdq(xe, rule).float() @ mx.qdq(w, rule, 1).float().t())
        dxs.append(mx.qdq(dye, rule).float() @ mx.qdq(w, rule, 0).float())
        mp = -(-m // 32) * 32
        xp = torch.zeros(mp, xe.shape[1], device=x.device); xp[:m] = xe
        dyp = torch.zeros(mp, dye.shape[1], device=x.device); dyp[:m] = dye
        xp, dyp = xp.to(x.dtype), dyp.to(x.dtype)
        dws.append(mx.qdq(dyp, rule, 0).float().t() @ mx.qdq(xp, rule, 0).float())
    return torch.cat(ys), torch.cat(dxs), dws


@pytest.mark.parametrize("rule", ["floor", "alto"])
def test_grouped_linear_fwd_bwd(rule):
    torch.manual_seed(1)
    m_splits = [37, 0, 64, 5, 130]                           # empty and non-aligned segments
    K, N = 640, 1408
    x = torch.randn(sum(m_splits), K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ws = [(torch.randn(N, K, device="cuda") / 25).to(torch.bfloat16).requires_grad_()
          for _ in m_splits]
    y = mx.grouped_linear(x, m_splits, ws, rule)
    dy = torch.randn_like(y)
    y.backward(dy)
    y_ref, dx_ref, dw_ref = ref_grouped(x.detach(), m_splits, [w.detach() for w in ws], rule, dy)
    rel = lambda a, b: float((a.float() - b).norm() / b.norm().clamp(min=1e-12))
    assert rel(y, y_ref) < 1e-2
    assert rel(x.grad, dx_ref) < 1e-2
    for w, d in zip(ws, dw_ref):
        if d.abs().sum() > 0:
            assert rel(w.grad, d) < 1e-2
        else:
            assert float(w.grad.float().abs().sum()) == 0.0


def test_main_grad_path():
    torch.manual_seed(2)
    m_splits = [40, 24]
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(96, 64, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in m_splits]
    for w in ws:
        w.main_grad = torch.zeros(w.shape, device="cuda", dtype=torch.float32)
        w.grad_added_to_main_grad = False
    y = mx.grouped_linear(x, m_splits, ws, "alto")
    dy = torch.randn_like(y)
    y.backward(dy)
    _, _, dw_ref = ref_grouped(x, m_splits, [w.detach() for w in ws], "alto", dy)
    for w, d in zip(ws, dw_ref):
        assert w.grad_added_to_main_grad is True
        assert float((w.main_grad - d).norm() / d.norm()) < 1e-2
        assert float(w.grad.abs().sum()) == 0.0               # placeholder only


def test_speed_vs_te_grouped_linear():
    """5B expert shapes per EP rank: 64 local experts, ~1.5k tokens each, fc1 640 -> 1408."""
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(3)
    E, K, N = 64, 640, 1408
    m_splits = torch.multinomial(torch.ones(E), 98304, replacement=True).bincount(minlength=E).tolist()
    x = torch.randn(sum(m_splits), K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    lin = te.GroupedLinear(E, K, N, bias=False, params_dtype=torch.bfloat16, device="cuda")
    ws = [getattr(lin, f"weight{i}") for i in range(E)]

    def step(f):
        y = f()
        y.backward(torch.ones_like(y))
        x.grad = None
        for w in ws:
            w.grad = None

    def bench(f, n=10):
        step(f); torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            step(f)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n * 1e3

    t_te = bench(lambda: lin(x, m_splits))
    t_mx = bench(lambda: mx.grouped_linear(x, m_splits, ws, "alto"))
    print(f"\nfc1 fwd+bwd, {sum(m_splits)} tokens: TE bf16 {t_te:.2f} ms, emulated MXFP4 {t_mx:.2f} ms "
          f"({t_mx / t_te:.2f}x)")


@pytest.mark.parametrize("rule", ["floor", "alto"])
def test_dense_linear_ragged_out_dim(rule):
    """KDA in_proj-like: N = 4106 is not a multiple of 32 (ragged last block in dgrad)."""
    torch.manual_seed(4)
    T, K, N = 256, 1280, 4106
    x = torch.randn(4, T // 4, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(N, K, device="cuda") / 36).to(torch.bfloat16).requires_grad_()
    y = mx.linear(x, w, rule)
    assert y.shape == (4, T // 4, N)
    dy = torch.randn_like(y)
    y.backward(dy)
    x2, dy2 = x.detach().reshape(T, K), dy.reshape(T, N)
    pad = (-N) % 32
    dyp = torch.nn.functional.pad(dy2, (0, pad))
    wp = torch.nn.functional.pad(w.detach().t(), (0, pad)).t()           # [N+pad, K]
    y_ref = mx.qdq(x2, rule).float() @ mx.qdq(w.detach(), rule, 1).float().t()
    dx_ref = (mx.qdq(dyp, rule).float() @ mx.qdq(wp, rule, 0).float())
    dw_ref = mx.qdq(dy2, rule, 0).float().t() @ mx.qdq(x2, rule, 0).float()
    rel = lambda a, b: float((a.float() - b).norm() / b.norm())
    assert rel(y.reshape(T, N), y_ref) < 1e-2
    assert rel(x.grad.reshape(T, K), dx_ref) < 1e-2
    assert rel(w.grad, dw_ref) < 1e-2
    # the padded quantization equals quantizing a zero-extended tensor and slicing
    assert torch.equal(mx.qdq(dy2, rule), mx.qdq(dyp, rule)[:, :N])


def test_fused_norm_matches_te():
    """_fused_norm reproduces TE LayerNormLinear's own normalized input."""
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(5)
    for zc in (False, True):
        mod = te.LayerNormLinear(1280, 512, bias=False, normalization="RMSNorm", eps=1e-5,
                                 zero_centered_gamma=zc, return_layernorm_output=True,
                                 params_dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            mod.layer_norm_weight.copy_(torch.rand(1280, device="cuda") * 0.5 + (0 if zc else 0.75))
        x = torch.randn(64, 1280, device="cuda", dtype=torch.bfloat16)
        _, ln_out = mod(x)
        ours = mx._fused_norm(mod, x)
        assert float((ours.float() - ln_out.float()).abs().max()) <= 2 ** -6 * float(ln_out.abs().max())


def test_weight_cache_within_step():
    torch.manual_seed(6)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    ref = mx.linear(x, w, "alto")                     # cache off: fresh quantization
    mx.new_step()
    try:
        y1 = mx.linear(x, w, "alto")
        y2 = mx.linear(x, w, "alto")                  # served from the cache
        assert torch.equal(y1, ref) and torch.equal(y2, ref)
        with torch.no_grad():
            w.mul_(2.0)                               # would be stale if the cache stayed on
    finally:
        mx.end_step()
    assert torch.equal(mx.linear(x, w, "alto"), 2.0 * ref)


# ---- ALTO recipe: stochastic rounding, 32x32 weight tiles, Hadamard along tokens -------------
def test_sr_unbiased_and_on_grid():
    """Blocks built so nothing clips (block max exactly 6 * 2^j): SR must then be unbiased and
    land only on the two grid neighbours of each value."""
    torch.manual_seed(7)
    j = torch.randint(-3, 4, (64, 8, 1), device="cuda").float()
    s = torch.exp2(j)                                              # the block scale
    x = (torch.rand(64, 8, 32, device="cuda") * 12 - 6) * s       # |x/s| < 6
    x[..., 0] = 6.0 * s[..., 0]                                    # block max = 6 s -> scale s
    x = x.reshape(64, 256)
    s = s.expand(64, 8, 32).reshape(64, 256)
    samples = torch.stack([mx.qdq(x, "floor", sr=True) for _ in range(400)])
    y = samples / s
    a = (x / s).abs()
    step = torch.where(a < 2, 0.5, torch.where(a < 4, 1.0, 2.0))
    lo = torch.floor(a / step) * step
    assert torch.isin(y.abs(), GRID.to(x.device)).all()
    assert ((y.abs() == lo) | (y.abs() == (lo + step).clamp(max=6.0))).all()
    # unbiased: per element, |mean - x| within ~5 sd of the Monte-Carlo error (sd <= step/40)
    err = ((samples.mean(0) - x) / s).abs() / step
    assert float(err.max()) < 0.13
    assert float(((samples.mean(0) - x) / s).mean().abs()) < 2e-3   # no systematic bias


def ref_tile(w, rule):
    """Independent 32x32-tile reference: each tile flattened to 1024 values, one scale. Ragged
    edges: zero-pad to whole tiles, quantize, slice back."""
    n0, k0 = w.shape[-2:]
    if n0 % 32 or k0 % 32:
        wp = torch.nn.functional.pad(w, (0, (-k0) % 32, 0, (-n0) % 32))
        return ref_tile(wp, rule)[..., :n0, :k0]
    n, k = w.shape[-2:]
    t = w.float().reshape(*w.shape[:-2], n // 32, 32, k // 32, 32).transpose(-3, -2)
    flat = t.reshape(*t.shape[:-2], 1024)
    amax = flat.abs().amax(-1, keepdim=True)
    mant, ex = torch.frexp(amax)
    e = (ex - 3).float()
    if rule == "alto":
        e = e + (2 * mant >= 1.75).float()
    e = torch.where(amax > 0, e, torch.full_like(e, -126.0)).clamp(min=-126)
    sc = torch.exp2(e)
    a = (flat / sc).abs().clamp(max=6.0)
    g, mids = GRID.to(w.device), MIDS.to(w.device)
    i = torch.bucketize(a.contiguous(), mids)
    tie = (i < 7) & (a == mids[i.clamp(max=6)])
    i = torch.where(tie & (i % 2 == 1), i + 1, i)
    q = (torch.sign(flat) * g[i] * sc).reshape(t.shape).transpose(-3, -2)
    return q.reshape(w.shape).to(w.dtype)


@pytest.mark.parametrize("rule", ["floor", "alto"])
def test_tile_matches_reference(rule):
    torch.manual_seed(8)
    w = (torch.randn(3, 1408, 640, device="cuda") * torch.exp(torch.randn(3, 1408, 1, device="cuda")))
    w = w.to(torch.bfloat16)
    assert torch.equal(mx.qdq_tile(w, rule), ref_tile(w, rule))


def test_hadamard_orthonormal_and_wgrad_invariant():
    mx.new_step()
    try:
        h = mx.hadamard(torch.device("cuda"))
        assert h is mx.hadamard(torch.device("cuda", torch.cuda.current_device()))  # one per step
        hd = h.double()                                       # fp64: the container runs fp32 as TF32
        assert float((hd @ hd.t() - torch.eye(32, device="cuda", dtype=torch.float64)).abs().max()) < 1e-6
        assert torch.allclose(h.abs(), torch.full_like(h, 32 ** -0.5))
        x = torch.randn(256, 64, device="cuda", dtype=torch.float64)
        dy = torch.randn(256, 48, device="cuda", dtype=torch.float64)
        hx, hdy = mx.rotate_tokens(x), mx.rotate_tokens(dy)
        assert torch.allclose(hdy.t() @ hx, dy.t() @ x, atol=1e-9)
        h1 = h.clone()
    finally:
        mx.end_step()
    mx.new_step()
    try:
        assert not torch.equal(mx.hadamard(torch.device("cuda")), h1)   # redrawn each step
    finally:
        mx.end_step()


def ref_alto_grouped(x, m_splits, ws, rule, dy):
    """ALTO recipe per expert, round-to-nearest (stochastic rounding switched off)."""
    ys, dxs, dws, o = [], [], [], 0
    for m, w in zip(m_splits, ws):
        xe, dye = x[o:o + m], dy[o:o + m]
        o += m
        wt = ref_tile(w, rule).float()
        ys.append(mx.qdq(xe, rule).float() @ wt.t())
        dxs.append(mx.qdq(dye, rule).float() @ wt)
        mp = -(-m // 32) * 32
        xp = torch.nn.functional.pad(xe, (0, 0, 0, mp - m))
        dyp = torch.nn.functional.pad(dye, (0, 0, 0, mp - m))
        if mp:
            xp, dyp = mx.rotate_tokens(xp), mx.rotate_tokens(dyp)
        dws.append(mx.qdq(dyp, rule, 0).float().t() @ mx.qdq(xp, rule, 0).float())
    return torch.cat(ys), torch.cat(dxs), dws


@pytest.mark.parametrize("rule", ["floor", "alto"])
def test_alto_recipe_grouped_and_dense(rule, monkeypatch):
    monkeypatch.setattr(mx, "_FORCE_RNE", True)
    torch.manual_seed(9)
    m_splits = [37, 0, 64, 5, 130]
    K, N = 640, 1408
    x = torch.randn(sum(m_splits), K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ws = [(torch.randn(N, K, device="cuda") / 25).to(torch.bfloat16).requires_grad_()
          for _ in m_splits]
    mx.new_step()
    try:
        y = mx.grouped_linear(x, m_splits, ws, rule, recipe="alto")
        dy = torch.randn_like(y)
        y.backward(dy)
        y_ref, dx_ref, dw_ref = ref_alto_grouped(x.detach(), m_splits, [w.detach() for w in ws],
                                                 rule, dy)
        # dense: one "expert" with all tokens, ragged N
        xd = torch.randn(96, 1280, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        wd = (torch.randn(4106, 1280, device="cuda") / 36).to(torch.bfloat16).requires_grad_()
        yd = mx.linear(xd, wd, rule, recipe="alto")
        dyd = torch.randn_like(yd)
        yd.backward(dyd)
        yd_ref, dxd_ref, dwd_ref = ref_alto_grouped(xd.detach(), [96], [wd.detach()], rule, dyd)
    finally:
        mx.end_step()
    rel = lambda a, b: float((a.float() - b).norm() / b.norm().clamp(min=1e-12))
    assert rel(y, y_ref) < 1e-2 and rel(x.grad, dx_ref) < 1e-2
    for w, d in zip(ws, dw_ref):
        assert rel(w.grad, d) < 1e-2 if d.abs().sum() > 0 else float(w.grad.float().abs().sum()) == 0
    assert rel(yd, yd_ref) < 1e-2 and rel(xd.grad, dxd_ref) < 1e-2 and rel(wd.grad, dwd_ref[0]) < 1e-2


def test_alto_recipe_sr_is_used_in_backward():
    """With stochastic rounding on, two identical backward passes give different dx."""
    torch.manual_seed(10)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = (torch.randn(512, 256, device="cuda") / 16).to(torch.bfloat16).requires_grad_()
    dy = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
    grads = []
    for _ in range(2):
        x.grad = None
        mx.linear(x, w, "alto", recipe="alto").backward(dy)
        grads.append(x.grad.clone())
    assert not torch.equal(grads[0], grads[1])

