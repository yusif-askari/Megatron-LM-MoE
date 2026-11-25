"""exp_pglu_lr (exp_gates.ExpPGLULR): PGLU with separate tail exponents left / right of the peak,
g'(x) = (1 + (b - x))^-pL (x < b), (1 + (x - b))^-pR (x >= b), g(x) = H(x - b) - H(-b), on fused kernels
(fp32 coefficients, h rounded to bf16 once, analytic backward, per-expert dL/drho_L, dL/drho_R, dL/db).
Checked against fp64 autograd of the formula, on the eager and the torch.compile path.

Self-contained (loads exp_gates.py by path); needs one GPU:
    srun ... python tests/unit_tests/transformer/test_exp_pglu_lr_fused.py
"""
import contextlib
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
spec = importlib.util.spec_from_file_location("exp_gates", os.path.join(REPO, "megatron/core/exp_gates.py"))
eg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg)
DEV = "cuda"
COUNTS = [0, 37, 128, 300, 5, 248, 129, 64, 200, 90]                 # 10 local experts, one empty
# per expert: (pL, pR, b): b > 0, b < 0, b = 0, pL = pR, extremes (0.001 .. 4), the 3B regime (p ~ 0.001, b ~ 10)
PLR = [(1.0, 1.0, 1.0), (3.0, 0.3, 0.5), (0.5, 2.0, -1.0), (1.5, 0.05, 2.0), (2.0, 2.0, 0.0),
       (0.001, 0.001, 5.0), (4.0, 0.2, -3.0), (0.3, 0.3, 10.0), (1.0001, 0.999, 0.25), (0.02, 1.7, -0.6)]


@contextlib.contextmanager
def compiled():
    """Run the fused functions under torch.compile (as in training, where Megatron's jit_fuser is torch.compile)."""
    eg._COMPILED.clear()
    orig = eg._fused
    cache = {}
    eg._fused = lambda fn: cache.setdefault(fn, torch.compile(fn, dynamic=True))
    try:
        yield
    finally:
        eg._fused = orig
        eg._COMPILED.clear()


def _F64(u, p):
    """F_p(u) in fp64 (odd; autograd slope (1 + |u|)^-p, 1 at u = 0)."""
    def G(t):
        l = torch.log1p(t)
        z = (1.0 - p) * l
        small = z.abs() < 1e-6
        zs = torch.where(small, torch.ones_like(z), z)
        return l * torch.where(small, 1.0 + z * (0.5 + z / 6.0), torch.expm1(zs) / zs)
    zero = torch.zeros_like(u)
    return torch.where(u >= 0, G(torch.where(u >= 0, u, zero)), -G(torch.where(u < 0, -u, zero)))


def _H64(u, pl, pr):
    return torch.where(u < 0, _F64(u, pl), _F64(u, pr))


def _truth(y, prob, raw_pl, raw_pr, b, counts):
    leaf = lambda t: t.double().detach().requires_grad_()
    lv = dict(y=leaf(y), prob=leaf(prob), raw_pl=leaf(raw_pl), raw_pr=leaf(raw_pr), b=leaf(b))
    rep = lambda t: torch.repeat_interleave(t, torch.as_tensor(counts, device=y.device)).unsqueeze(-1)
    pl, pr, br = rep(F.softplus(lv["raw_pl"])), rep(F.softplus(lv["raw_pr"])), rep(lv["b"])
    g, v = lv["y"].chunk(2, -1)
    gate = _H64(g - br, pl, pr) - _H64(-br, pl, pr)
    return gate * v * lv["prob"], lv


def _rel(a, r):
    return ((a.double() - r).norm() / r.norm().clamp_min(1e-300)).item()


def _model(plr):
    m = eg.ExpPGLULR(num_local_experts=len(plr)).to(DEV)
    t = torch.tensor(plr, device=DEV)
    with torch.no_grad():
        m.raw_pl.copy_(torch.log(torch.expm1(t[:, 0]))); m.raw_pr.copy_(torch.log(torch.expm1(t[:, 1])))
        m.b.copy_(t[:, 2])
    return m


def _inputs(seed, b, scale):
    torch.manual_seed(seed)
    n, f = sum(COUNTS), 256
    y = (torch.randn(n, 2 * f, device=DEV) * scale).bfloat16()
    br = torch.repeat_interleave(b, torch.tensor(COUNTS, device=DEV)).unsqueeze(-1).expand(n, f)
    near = torch.rand(n, f, device=DEV) < 0.2                       # near the peak
    y[:, :f][near] = (br[near] + torch.randn_like(br[near])).bfloat16()
    tie = torch.rand(n, f, device=DEV) < 0.02                       # exactly at bf16(b)
    y[:, :f][tie] = br[tie].bfloat16()
    y[:, :f][torch.rand(n, f, device=DEV) < 0.01] = 0
    prob = torch.rand(n, 1, device=DEV) * 0.3
    dh = (torch.randn(n, f, device=DEV) * 1e-2).bfloat16()
    return y, prob, dh


def _check(m, y, prob, dh, tol_param=1e-4):
    yy, pr = y.clone().requires_grad_(True), prob.clone().requires_grad_(True)
    h = m.weighted_glu(yy, pr, COUNTS)
    assert h.dtype == torch.bfloat16
    h.backward(dh)
    h_t, lv = _truth(y, prob, m.raw_pl.detach(), m.raw_pr.detach(), m.b.detach(), COUNTS)
    h_t.backward(dh.double())
    hb = h_t.detach().to(torch.bfloat16)
    mism = (h != hb).float().mean().item()
    ulp = torch.exp2(torch.floor(torch.log2(hb.float().abs().clamp_min(2.0 ** -126))) - 7)    # one bf16 ulp
    maxulp = ((h.float() - h_t.detach().float()).abs() / ulp).max().item()
    assert mism < 5e-4, mism                                         # one rounding: ties only
    assert maxulp <= 0.51, maxulp
    assert _rel(yy.grad, lv["y"].grad) < 1.8e-3, _rel(yy.grad, lv["y"].grad)
    assert _rel(pr.grad, lv["prob"].grad) < 1e-5
    for name in ("raw_pl", "raw_pr", "b"):
        g = getattr(m, name).grad
        r = _rel(g, lv[name].grad)
        assert r < tol_param, (name, r)
        nz = lv[name].grad.abs() > 1e-12 * lv[name].grad.abs().max()
        assert (torch.sign(g[nz]) == torch.sign(lv[name].grad[nz].float())).all(), name
        assert g[0].item() == 0.0, name                              # empty expert
        assert getattr(m, name).dtype == torch.float32 and getattr(m, name).keep_fp32
    return mism, maxulp


def test_vs_fp64_eager_and_compiled():
    for scale in (2.0, 100.0):
        for mode in ("eager", "compiled"):
            ctx = compiled() if mode == "compiled" else contextlib.nullcontext()
            with ctx:
                m = _model(PLR)
                y, prob, dh = _inputs(1, m.b.detach(), scale)
                mism, maxulp = _check(m, y, prob, dh)
            print(f"   scale {scale:5.0f} {mode:8s}: h != bf16(truth) {mism:.1e}, max {maxulp:.2f} ulp")


def test_equal_exponents_is_pglu():
    """pL = pR: the same h as exp_pglu (bitwise) and the same gradients up to the slope's rounding
    (dL/drho_L + dL/drho_R = dL/drho)."""
    torch.manual_seed(3)
    p = torch.tensor([1.0, 0.3, 2.5, 0.001, 1.6, 1.0, 0.7, 4.0, 1.2, 0.05], device=DEV)
    b = torch.tensor([1.0, -0.7, 3.0, 5.0, 0.0, 1.3, -2.0, 6.0, 0.25, 10.0], device=DEV)
    lr = _model([(float(a), float(a), float(c)) for a, c in zip(p, b)])
    pg = eg.ExpPGLU(num_local_experts=len(COUNTS)).to(DEV)
    with torch.no_grad():
        pg.raw_p.copy_(torch.log(torch.expm1(p))); pg.b.copy_(b)
    y, prob, dh = _inputs(4, b, 5.0)
    outs = []
    for m in (lr, pg):
        yy, pr = y.clone().requires_grad_(True), prob.clone().requires_grad_(True)
        h = m.weighted_glu(yy, pr, COUNTS)
        h.backward(dh)
        outs.append((h, yy.grad, pr.grad, m.b.grad))
    assert torch.equal(outs[0][0], outs[1][0])                       # forward: same code path, bitwise
    # backward: exp_pglu_lr takes the slope as exp(-p l) (exp_pglu: (e^z - 1 + 1) / (1 + w), which loses digits
    # where the slope is << 1), so dL/dy may differ at bf16 rounding ties only
    dmis = (outs[0][1] != outs[1][1]).float().mean().item()
    print(f"   dL/dy differs from exp_pglu in {dmis:.1e} of entries, relative {_rel(outs[0][1], outs[1][1].double()):.1e}")
    assert dmis < 1e-3 and _rel(outs[0][1], outs[1][1].double()) < 1e-3
    assert _rel(outs[0][2], outs[1][2].double()) < 1e-6
    assert _rel(outs[0][3], outs[1][3].double()) < 2e-5
    assert _rel(lr.raw_pl.grad + lr.raw_pr.grad, pg.raw_p.grad.double()) < 1e-5


def test_slope_continuous_and_formula():
    """g'(x) from the kernel = (1 + |x - b|)^-p_side, = 1 at x = b, and the gate alone matches fp64."""
    m = _model(PLR)
    E = len(PLR)
    b = m.b.detach()
    eps = torch.tensor([-1e-3, 0.0, 1e-3], device=DEV)
    x = (b.unsqueeze(-1) + eps).reshape(-1)                         # around each peak
    xs = torch.cat([x, torch.linspace(-50, 50, 301, device=DEV).repeat(E)])
    ids = torch.cat([torch.arange(E, device=DEV).repeat_interleave(3), torch.arange(E, device=DEV).repeat_interleave(301)])
    # reference from the kernel's own fp32 parameters (softplus of the fp32 rho) and fp32 inputs, in fp64
    pl = F.softplus(m.raw_pl.detach().double())[ids]
    pr = F.softplus(m.raw_pr.detach().double())[ids]
    bb = m.b.detach().double()[ids]
    # analytic slope through the backward kernel: d/dx of sum(g(x)) for a single-column input, v = 1, prob = 1
    coef = eg.pglu_lr_coefficients(m.raw_pl, m.raw_pr, m.b).index_select(0, ids)
    g_in = xs.unsqueeze(-1).float()
    d_g, _, _, _ = eg._pglu_lr_bwd(torch.ones_like(g_in), g_in, torch.ones_like(g_in), coef, torch.ones(1, 1, device=DEV))
    slope = d_g.double().squeeze(-1)
    want = torch.where(xs.double() < bb, (1 + (bb - xs.double())) ** (-pl), (1 + (xs.double() - bb)) ** (-pr))
    err = ((slope - want).abs() / want).max().item()
    print(f"   slope vs fp64: max relative error {err:.1e}")
    assert err < 1e-5                                               # fp32 arithmetic
    at_b = slope[:3 * E].reshape(E, 3)
    assert ((at_b - 1.0).abs() < 1e-2).all(), at_b                  # continuous: -> 1 from both sides
    gate = m(xs, expert_ids=ids).double()
    g64 = _H64(xs.double() - bb, pl, pr) - _H64(-bb, pl, pr)
    assert ((gate - g64).abs() / g64.abs().clamp_min(1e-3)).max().item() < 1e-5
    assert m(torch.zeros(E, device=DEV), expert_ids=torch.arange(E, device=DEV)).abs().max().item() == 0.0  # g(0) = 0


def test_single_expert_and_coefficients():
    """Dense / shared MLP (one expert, no probs), init == exp_pglu at init, coefficients() for glu-stats."""
    m = eg.GATES["exp_pglu_lr"]().to(DEV)
    c = m.coefficients()
    assert set(c) == {"pL", "pR", "b"} and abs(c["pL"].item() - 1) < 1e-6 and abs(c["pR"].item() - 1) < 1e-6
    torch.manual_seed(5)
    y = (torch.randn(1000, 512, device=DEV) * 3).bfloat16()
    pg = eg.ExpPGLU().to(DEV)
    assert torch.equal(m.weighted_glu(y), pg.weighted_glu(y))
    with torch.no_grad():
        m.raw_pl.fill_(1.2); m.raw_pr.fill_(-2.0); m.b.fill_(0.8)
    yy = y.clone().requires_grad_(True)
    h = m.weighted_glu(yy)
    dh = torch.randn_like(h)
    h.backward(dh)
    h_t, lv = _truth(y, torch.ones(1000, 1, device=DEV), m.raw_pl.detach(), m.raw_pr.detach(), m.b.detach(), [1000])
    h_t.backward(dh.double())
    assert (h != h_t.detach().to(torch.bfloat16)).float().mean().item() < 5e-4
    assert _rel(yy.grad, lv["y"].grad) < 1.8e-3
    for name in ("raw_pl", "raw_pr", "b"):
        assert _rel(getattr(m, name).grad, lv[name].grad) < 1e-4, name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback; traceback.print_exc()
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
