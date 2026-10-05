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
