"""exp_bglu_p{p}, exp_bglu_p{p}_c, exp_bglu_p{p}_cs and exp_pglu_c on the fused pglu kernels (exp_gates.py):
fp32 coefficients (keep_fp32), h = bf16((g(y_gate) + c y_gate) * y_value * prob) rounded once, analytic backward,
per-expert dL/db, dL/dc (and dL/drho for exp_pglu_c). Checked against fp64 autograd of the same formula.

Self-contained (loads exp_gates.py by path); needs one GPU:
    srun ... python tests/unit_tests/transformer/test_exp_bglu_fused.py
"""
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eg = _load("exp_gates", os.path.join(REPO, "megatron/core/exp_gates.py"))
DEV = "cuda"
D = torch.float64
COUNTS = [0, 37, 128, 300, 5, 248, 129, 64]          # 8 local experts, one empty


def _bump_integral64(u, p):
    """F_p(u) in fp64 (autograd gives the true slope 1 at u = 0)."""
    def G(t):
        l = torch.log1p(t)
        z = (1.0 - p) * l
        small = z.abs() < 1e-6
        zs = torch.where(small, torch.ones_like(z), z)
        return l * torch.where(small, 1.0 + z * (0.5 + z / 6.0), torch.expm1(zs) / zs)
    zero = torch.zeros_like(u)
    return torch.where(u >= 0, G(torch.where(u >= 0, u, zero)), -G(torch.where(u < 0, -u, zero)))


def _truth(y, prob, b, counts, p=None, raw_p=None, c=None, raw_c=None):
    """fp64 autograd of h = (F_p(g - b) - F_p(-b) + c g) * v * prob over expert segments.
    p: fixed exponent (float) or raw_p (learnable, p = softplus); c: [E] or raw_c (c = softplus(raw_c))."""
    leaf = lambda t: t.double().detach().requires_grad_()
    y64, pr64, b64 = leaf(y), leaf(prob), leaf(b)
    cnt = torch.as_tensor(counts, device=y.device)
    rep = lambda t: torch.repeat_interleave(t, cnt).unsqueeze(-1)
    leaves = {"y": y64, "prob": pr64, "b": b64}
    if raw_p is not None:
        leaves["raw_p"] = leaf(raw_p)
        pr = rep(F.softplus(leaves["raw_p"]))
    else:
        pr = p
    g, v = y64.chunk(2, -1)
    br = rep(b64)
    gate = _bump_integral64(g - br, pr) - _bump_integral64(-br, pr)
    if c is not None:
        leaves["c"] = leaf(c)
        gate = gate + rep(leaves["c"]) * g
    if raw_c is not None:
        leaves["raw_c"] = leaf(raw_c)
        gate = gate + rep(F.softplus(leaves["raw_c"])) * g
    return gate * v * pr64, leaves


def _inputs(seed, b):
    torch.manual_seed(seed)
    n, f = sum(COUNTS), 256
    y = (torch.randn(n, 2 * f, device=DEV) * 2.0).bfloat16()
    br = torch.repeat_interleave(b, torch.tensor(COUNTS, device=DEV)).unsqueeze(-1).expand(n, f)
    tie = torch.rand(n, f, device=DEV) < 0.02                       # 2% of gate inputs exactly at b
    y[:, :f][tie] = br[tie].bfloat16()
    prob = torch.rand(n, 1, device=DEV) * 0.3
    dh = (torch.randn(n, f, device=DEV) * 1e-2).bfloat16()
    return y, prob, dh


def _rel(a, r):
    return ((a.double() - r).norm() / r.norm()).item()


B_VALS = [1.0, 0.0, -0.7, 3.0, 1.3, 0.25, 1.0, 6.0]
C_VALS = [0.0, 0.05, -0.02, 0.3, 0.1, 0.0, 0.01, 1.0]


def _check_training_path(m, y, prob, dh, truth_kw, names):
    yy, pr = y.clone().requires_grad_(True), prob.clone().requires_grad_(True)
    h = m.weighted_glu(yy, pr, COUNTS)
    assert h.dtype == torch.bfloat16
    h.backward(dh)
    h_t, leaves = _truth(y, prob, counts=COUNTS, **truth_kw)
    h_t.backward(dh.double())
    mism = (h != h_t.to(torch.bfloat16)).float().mean().item()
    assert mism < 5e-4, mism                    # one rounding: one-ulp differences at rounding ties only
    assert _rel(yy.grad, leaves["y"].grad) < 1.8e-3                 # bf16 rounding of d(y)
    assert _rel(pr.grad, leaves["prob"].grad) < 1e-5
    for name in names:
        g = getattr(m, name).grad
        assert _rel(g, leaves[name].grad) < 1e-5, (type(m).__name__, name, _rel(g, leaves[name].grad))
        assert g[0].item() == 0.0                                   # empty expert
        prm = getattr(m, name)
        assert prm.dtype == torch.float32 and prm.keep_fp32, name


def test_bglu_c_weighted_glu_vs_fp64():
    for p in (2.0, 0.5, 1.0, 1.5):
        cls = eg.GATES[f"exp_bglu_p{eg._f(p)}_c"]
        b, c = torch.tensor(B_VALS, device=DEV), torch.tensor(C_VALS, device=DEV)
        y, prob, dh = _inputs(0, b)
        m = cls(num_local_experts=len(COUNTS)).to(DEV)
        with torch.no_grad():
            m.b.copy_(b); m.c.copy_(c)
        _check_training_path(m, y, prob, dh, dict(b=b, p=p, c=c), ("b", "c"))


def test_bglu_weighted_glu_vs_fp64():
    for p in (2.0, 1.0):
        b = torch.tensor(B_VALS, device=DEV)
        y, prob, dh = _inputs(1, b)
        m = eg.GATES[f"exp_bglu_p{eg._f(p)}"](num_local_experts=len(COUNTS)).to(DEV)
        with torch.no_grad():
            m.b.copy_(b)
        _check_training_path(m, y, prob, dh, dict(b=b, p=p), ("b",))


def test_pglu_c_weighted_glu_vs_fp64():
    b, c = torch.tensor(B_VALS, device=DEV), torch.tensor(C_VALS, device=DEV)
    raw_p = torch.log(torch.expm1(torch.tensor([1.0, 0.3, 2.5, 1.0001, 0.999, 1.6, 1.0, 4.0], device=DEV)))
    y, prob, dh = _inputs(2, b)
    m = eg.ExpPGLUC(num_local_experts=len(COUNTS)).to(DEV)
    with torch.no_grad():
        m.raw_p.copy_(raw_p); m.b.copy_(b); m.c.copy_(c)
    _check_training_path(m, y, prob, dh, dict(b=b, raw_p=raw_p, c=c), ("raw_p", "b", "c"))


def test_bglu_cs_weighted_glu_vs_fp64():
    b = torch.tensor(B_VALS, device=DEV)
    raw_c = torch.tensor([-4.0, -2.0, 0.0, 1.0, -4.0, -3.0, 0.5, -1.0], device=DEV)
    y, prob, dh = _inputs(3, b)
    m = eg.GATES["exp_bglu_p2_cs"](num_local_experts=len(COUNTS)).to(DEV)
    with torch.no_grad():
        m.b.copy_(b); m.raw_c.copy_(raw_c)
    _check_training_path(m, y, prob, dh, dict(b=b, p=2.0, raw_c=raw_c), ("b", "raw_c"))


def test_gate_alone_vs_fp64_and_reference():
    """forward(x) (the --log-glu-stats path): value and dL/dx, dL/db, dL/dc against fp64, per expert, by
    tokens_per_expert and by explicit expert_ids; and against the old plain-autograd implementation."""
    torch.manual_seed(4)
    E = len(COUNTS)
    b, c = torch.tensor(B_VALS, device=DEV), torch.tensor(C_VALS, device=DEV)
    m = eg.GATES["exp_bglu_p2_c"](num_local_experts=E).to(DEV)
    with torch.no_grad():
        m.b.copy_(b); m.c.copy_(c)
    x = (torch.randn(sum(COUNTS), 64, device=DEV) * 4).double().requires_grad_(True)
    y = m(x, COUNTS)
    w = torch.randn_like(y)
    (y * w).sum().backward()
    ones = torch.ones_like(x)
    h_t, leaves = _truth(torch.cat([x.detach(), ones], -1), torch.ones(x.shape[0], 1, device=DEV, dtype=D),
                         b, COUNTS, p=2.0, c=c)
    (h_t * w).sum().backward()
    assert (y - h_t).abs().max().item() < 2e-6 * max(1.0, h_t.abs().max().item())
    assert _rel(x.grad, leaves["y"].grad[:, :64]) < 1e-5
    assert _rel(m.b.grad, leaves["b"].grad) < 1e-5 and _rel(m.c.grad, leaves["c"].grad) < 1e-5
    ids = torch.repeat_interleave(torch.arange(E, device=DEV), torch.tensor(COUNTS, device=DEV))
    assert (m(x.detach(), expert_ids=ids) - y.detach()).abs().max().item() == 0.0
    # old implementation (fp32 math from the fp32 parameters, autograd): same gate
    xs = x.detach().float()
    ref = eg._pglu_c_ref(xs, torch.full((xs.shape[0], 1), 2.0, device=DEV),
                         b[ids].unsqueeze(-1), c[ids].unsqueeze(-1))
    assert (m(xs, COUNTS) - ref).abs().max().item() < 1e-4 * max(1.0, ref.abs().max().item())


def test_c0_equals_parent_bitwise():
    """At c = 0 exp_bglu_p2_c trains exactly like exp_bglu_p2 (h and all gradients bitwise)."""
    b = torch.tensor(B_VALS, device=DEV)
    y, prob, dh = _inputs(5, b)
    out = []
    for name in ("exp_bglu_p2", "exp_bglu_p2_c"):
        m = eg.GATES[name](num_local_experts=len(COUNTS)).to(DEV)
        with torch.no_grad():
            m.b.copy_(b)
        yy, pr = y.clone().requires_grad_(True), prob.clone().requires_grad_(True)
        h = m.weighted_glu(yy, pr, COUNTS)
        h.backward(dh)
        out.append((h, yy.grad, pr.grad, m.b.grad))
    for a, r in zip(*out):
        assert torch.equal(a, r)


def test_single_expert_dense_and_compiled():
    """One expert (dense / shared MLP, no probs), and the kernels under torch.compile."""
    torch.manual_seed(6)
    m = eg.GATES["exp_bglu_p2_c"]().to(DEV)
    with torch.no_grad():
        m.b.fill_(1.7); m.c.fill_(0.04)
    y = (torch.randn(1000, 512, device=DEV) * 3).bfloat16().requires_grad_(True)
    h = m.weighted_glu(y)
    dh = torch.randn_like(h)
    h.backward(dh)
    h_t, leaves = _truth(y.detach(), torch.ones(1000, 1, device=DEV), torch.tensor([1.7], device=DEV), [1000],
                         p=2.0, c=torch.tensor([0.04], device=DEV))
    h_t.backward(dh.double())
    assert (h != h_t.to(torch.bfloat16)).float().mean().item() < 5e-4
    assert _rel(y.grad, leaves["y"].grad) < 1.8e-3
    assert _rel(m.b.grad, leaves["b"].grad) < 1e-5 and _rel(m.c.grad, leaves["c"].grad) < 1e-5
    cr = eg.bglu_coefficients(m.b, 2.0)
    g, v = y.detach().chunk(2, -1)
    cc = m.c.float().reshape(1, 1)
    one = torch.ones(1, 1, device=DEV)
    he, hc = eg._pglu_fwd(g, v, cr, one, cc), torch.compile(eg._pglu_fwd, dynamic=True)(g, v, cr, one, cc)
    assert (he != hc).float().mean().item() < 1e-3
    be = eg._pglu_bwd(dh, g, v, cr, one, cc)
    bc = torch.compile(eg._pglu_bwd, dynamic=True)(dh, g, v, cr, one, cc)
    for a, r in zip(be, bc):
        assert torch.isfinite(r.float()).all()
        assert ((a.float() - r.float()).norm() / a.float().norm()) < 1e-3


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
