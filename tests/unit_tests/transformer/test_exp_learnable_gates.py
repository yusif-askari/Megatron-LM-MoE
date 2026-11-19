"""Learnable activation-sweep gates (exp_pglu / exp_xrlglu) and their --log-glu-stats hooks.

exp_pglu evaluates in fp32 from an fp32 coefficient table (p = softplus(rho), b and derived constants;
see the pglu section of exp_gates.py), so its tolerances are fp32-level even for fp64 inputs.

Self-contained (loads exp_gates.py / glu_stats.py by path); needs one GPU:
    srun ... python tests/unit_tests/transformer/test_exp_learnable_gates.py
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


gs = _load("glu_stats", os.path.join(REPO, "megatron/core/transformer/glu_stats.py"))
eg = _load("exp_gates", os.path.join(REPO, "megatron/core/exp_gates.py"))
DEV = "cuda"
D = torch.float64
X = torch.linspace(-12, 12, 4001, dtype=torch.float64)


def _pglu(p, b):
    m = eg.ExpPGLU(p_init=p, b_init=b).double()
    return m


def test_pglu_matches_builder_at_init():
    """p=1, b=1 is exactly the fixed bump exp_l0m1r0_p1_b1."""
    ref = eg.GATES["exp_l0m1r0_p1_b1"](X)
    m = _pglu(1.0, 1.0)
    assert m.b.item() == 1.0 and abs(m.coefficients()["p"].item() - 1.0) < 1e-7   # fp32 init
    assert (m(X) - ref).abs().max() < 1e-6


def test_pglu_derivative_and_zero():
    for p in (0.5, 0.999, 1.0, 1.0005, 1.5, 2.0):
        for b in (0.0, 1.0, -0.7):
            x = X.clone().requires_grad_(True)
            m = _pglu(p, b)
            y = m(x)
            (gx,) = torch.autograd.grad(y.sum(), x)
            pp, bb = F.softplus(m.raw_p).item(), m.b.item()     # the fp32-stored init values
            ref = (1 + (X - bb).abs()) ** (-pp)
            assert ((gx - ref).abs() / ref).max() < 2e-6, (p, b)
            assert abs(_pglu(p, b)(torch.zeros(1, dtype=torch.float64)).item()) < 1e-12
            xb = torch.tensor([bb], dtype=torch.float64, requires_grad=True)      # slope 1 at x = b
            (sb,) = torch.autograd.grad(m(xb).sum(), xb)
            assert sb.item() == 1.0, (p, b, sb.item())


def _bump_integral64(u, p):
    """F_p(u) in fp64 as G(u) for u >= 0 and -G(-u) for u < 0, G(t) = l exprel((1-p) l), l = ln(1+t):
    autograd gives the true slope 1 at u = 0."""
    def G(t):
        l = torch.log1p(t)
        z = (1.0 - p) * l
        small = z.abs() < 1e-6
        zs = torch.where(small, torch.ones_like(z), z)
        return l * torch.where(small, 1.0 + z * (0.5 + z / 6.0), torch.expm1(zs) / zs)
    zero = torch.zeros_like(u)
    return torch.where(u >= 0, G(torch.where(u >= 0, u, zero)), -G(torch.where(u < 0, -u, zero)))


def _truth(y, prob, raw_p, b, counts):
    """fp64 autograd of h = (F_p(g - b) - F_p(-b)) * v * prob over expert segments."""
    y64 = y.double().detach().requires_grad_()
    pr64 = prob.double().detach().requires_grad_()
    rp64 = raw_p.double().detach().requires_grad_()
    b64 = b.double().detach().requires_grad_()
    c = torch.as_tensor(counts, device=y.device)
    p = torch.repeat_interleave(F.softplus(rp64), c).unsqueeze(-1)
    br = torch.repeat_interleave(b64, c).unsqueeze(-1)
    g, v = y64.chunk(2, -1)
    h = (_bump_integral64(g - br, p) - _bump_integral64(-br, p)) * v * pr64
    return h, (y64, pr64, rp64, b64)


def test_pglu_param_grads_at_p1():
    """dL/drho, dL/db finite and right at and next to p = 1 (the old where(p==1, log, pow) form gave a NaN
    dL/dp at p = 1), from the gate alone (forward) against the fp64 truth."""
    for p in (1.0, 1.00005, 0.99995):
        m = _pglu(p, 1.0)
        m(X).sum().backward()
        gp, gb = m.raw_p.grad.item(), m.b.grad.item()
        rp, bb = m.raw_p.detach().float(), m.b.detach().float()
        h, (_, _, rp64, b64) = _truth(torch.stack([X, torch.ones_like(X)], -1), torch.ones(len(X), 1, dtype=D),
                                      rp, bb, [len(X)])
        h.sum().backward()
        assert math.isfinite(gp) and math.isfinite(gb)
        assert abs(gp - rp64.grad.item()) <= 1e-5 * abs(rp64.grad.item()), (p, gp, rp64.grad.item())
        assert abs(gb - b64.grad.item()) <= 1e-5 * abs(b64.grad.item()), (p, gb, b64.grad.item())


def test_pglu_series_branch_continuous():
    """Value continuous across the |z| = 0.5 series / closed-form switch of exprel (and its phi)."""
    x = torch.tensor([3.0], dtype=torch.float64)
    l = math.log1p(3.0)
    for z in (0.4999, 0.5001, -0.4999, -0.5001):
        m = _pglu(1.0, 0.0)
        with torch.no_grad():                                 # set p = 1 - z/l
            m.raw_p.fill_(math.log(math.expm1(1.0 - z / l)))
        q = 1.0 - F.softplus(m.raw_p.float()).double().item()
        exact = ((1 + 3.0) ** q - 1) / q
        assert abs(m(x).item() - exact) < 2e-6 * abs(exact), (z, m(x).item(), exact)


def test_pglu_weighted_glu_vs_fp64():
    """The training path h = bf16(g(y_gate) y_value prob) (one rounding, analytic backward) over 8 experts with
    uneven counts (one empty), p = 1 exactly and near it, b = 0 and negative, 2% of gate inputs exactly at b."""
    torch.manual_seed(0)
    counts = [0, 37, 128, 300, 5, 248, 129, 64]
    n, f, E = sum(counts), 256, len(counts)
    y = (torch.randn(n, 2 * f, device=DEV) * 2.0).bfloat16()
    p_vals = torch.tensor([1.0, 0.3, 2.5, 1.0001, 0.999, 1.6, 1.0, 4.0], device=DEV)
    b_vals = torch.tensor([1.0, 0.0, -0.7, 3.0, 1.3, 0.25, 1.0, 6.0], device=DEV)
    raw_p, b = torch.log(torch.expm1(p_vals)), b_vals
    br = torch.repeat_interleave(b, torch.tensor(counts, device=DEV)).unsqueeze(-1).expand(n, f)
    tie = torch.rand(n, f, device=DEV) < 0.02
    y[:, :f][tie] = br[tie].bfloat16()
    prob = torch.rand(n, 1, device=DEV) * 0.3
    dh = (torch.randn(n, f, device=DEV) * 1e-2).bfloat16()
    m = eg.ExpPGLU(num_local_experts=E).to(DEV)
    with torch.no_grad():
        m.raw_p.copy_(raw_p); m.b.copy_(b)
    yy, pr = y.clone().requires_grad_(True), prob.clone().requires_grad_(True)
    h = m.weighted_glu(yy, pr, counts)
    assert h.dtype == torch.bfloat16
    h.backward(dh)
    h_t, leaves = _truth(y, prob, raw_p, b, counts)
    h_t.backward(dh.double())
    rel = lambda a, r: ((a.double() - r).norm() / r.norm()).item()
    assert (h != h_t.to(torch.bfloat16)).float().mean().item() < 2e-4
    assert rel(yy.grad, leaves[0].grad) < 1.8e-3
    assert rel(pr.grad, leaves[1].grad) < 1e-5
    assert rel(m.raw_p.grad, leaves[2].grad) < 1e-5
    assert rel(m.b.grad, leaves[3].grad) < 1e-5
    assert m.raw_p.grad[0].item() == 0.0 and m.b.grad[0].item() == 0.0          # empty expert
    assert m.raw_p.dtype == torch.float32 and m.raw_p.keep_fp32 and m.b.keep_fp32


def test_xrlglu_alpha0_is_rlglu_and_tails():
    ref = eg.GATES["exp_l0m0.5r1_p1_b0"](X)
    assert (eg.ExpXRLGLU(alpha_init=0.0).double()(X) - ref).abs().max() < 1e-12
    a = 0.3
    x = torch.tensor([-1e6, 0.0, 1e6], dtype=torch.float64, requires_grad=True)
    (gx,) = torch.autograd.grad(eg.ExpXRLGLU(alpha_init=a).double()(x).sum(), x)
    assert abs(gx[0] + a) < 1e-5 and abs(gx[2] - 1) < 1e-5     # g'(-inf) = -alpha, g'(inf) = 1


def _per_expert_reference(mod_cls, kw, vals, x, tpe):
    """Row block e evaluated with a single-expert module holding expert e's coefficients."""
    outs, s = [], 0
    for e, n in enumerate(tpe):
        m = mod_cls(**kw).to(DEV)
        with torch.no_grad():
            for name, v in vals.items():
                getattr(m, name).fill_(v[e])
        outs.append(m(x[s:s + n])); s += n
    return torch.cat(outs)


def test_per_expert_mapping_and_grads():
    torch.manual_seed(0)
    tpe = [5, 0, 7, 3]
    E, N, Fd = len(tpe), sum(tpe), 16
    for cls, vals in ((eg.ExpPGLU, {"raw_p": [0.2, 0.5, 0.9, 1.4], "b": [1.0, -0.3, 0.4, 2.0]}),
                      (eg.ExpXRLGLU, {"alpha": [0.3, -0.1, 0.7, 0.05]})):
        m = cls(num_local_experts=E).to(DEV)
        with torch.no_grad():
            for name, v in vals.items():
                getattr(m, name).copy_(torch.tensor(v))
        x = torch.randn(N, Fd, device=DEV) * 4
        y = m(x, tpe)
        ref = _per_expert_reference(cls, {}, vals, x, tpe)
        assert (y - ref).abs().max() < 1e-5, cls.__name__
        # explicit row ids (glu_stats path) agree
        ids = torch.repeat_interleave(torch.arange(E, device=DEV), torch.tensor(tpe, device=DEV))
        assert (m(x, expert_ids=ids) - y).abs().max() < 1e-6
        # coefficient grads: expert e only sees its own rows; empty expert gets zero grad
        w = torch.randn_like(y)
        (y * w).sum().backward()
        s = 0
        for e, n in enumerate(tpe):
            me = cls().to(DEV)
            with torch.no_grad():
                for name, v in vals.items():
                    getattr(me, name).fill_(v[e])
            (me(x[s:s + n]) * w[s:s + n]).sum().backward()
            s += n
            for name in vals:
                ge, gm = getattr(me, name).grad.item(), getattr(m, name).grad[e].item()
                assert abs(ge - gm) <= 1e-4 * max(1.0, abs(ge)), (cls.__name__, name, e, ge, gm)
        for name in vals:
            assert getattr(m, name).grad[1].item() == 0.0


def test_bf16_io_and_compiled_matches_eager():
    torch.manual_seed(1)
    tpe = [300, 212]
    x = (torch.randn(sum(tpe), 256, device=DEV) * 5).to(torch.bfloat16)
    for cls in (eg.ExpPGLU, eg.ExpXRLGLU):
        m = cls(num_local_experts=2).to(DEV)
        with torch.no_grad():
            for prm in m.parameters():
                prm.add_(torch.tensor([0.1, -0.2], device=DEV))
        y = m(x, tpe)
        assert y.dtype == torch.bfloat16
        ref = m(x.double(), tpe)
        assert ((y.double() - ref).abs() / (ref.abs() + 1)).max() < 1e-2
    # pglu: its fused forward / backward functions, eager against torch.compile
    m = eg.ExpPGLU(num_local_experts=2).to(DEV)
    cr = eg._rows(eg.pglu_coefficients(m.raw_p, m.b), tpe, x.shape[0])
    v = (torch.randn_like(x.float()) * 2).bfloat16()
    pr = torch.rand(x.shape[0], 1, device=DEV)
    dh = torch.randn_like(x)
    he, hc = eg._pglu_fwd(x, v, cr, pr), torch.compile(eg._pglu_fwd, dynamic=True)(x, v, cr, pr)
    assert (he != hc).float().mean().item() < 1e-3 and (he.float() - hc.float()).abs().max() < 2e-2
    be = eg._pglu_bwd(dh, x, v, cr, pr)
    bc = torch.compile(eg._pglu_bwd, dynamic=True)(dh, x, v, cr, pr)
    for a, b in zip(be, bc):
        assert torch.isfinite(b.float()).all()
        assert ((a.float() - b.float()).norm() / a.float().norm()) < 1e-3
    for cls in (eg.ExpXRLGLU,):
        m = cls(num_local_experts=2).to(DEV)
        with torch.no_grad():
            for prm in m.parameters():
                prm.add_(torch.tensor([0.1, -0.2], device=DEV))
        # the training path wraps the math in torch.compile (megatron jit_fuser)
        fn = eg._xrlglu
        cfn = torch.compile(fn, dynamic=True)
        args = [m._rows(m.alpha, x, tpe)]
        xe = x.clone().requires_grad_(True); xc = x.clone().requires_grad_(True)
        ye, yc = fn(xe, *args), cfn(xc, *args)
        assert (ye.float() - yc.float()).abs().max() < 2e-2
        ge = torch.autograd.grad(ye.float().sum(), [xe] + [a for a in args if a.requires_grad])
        gc = torch.autograd.grad(yc.float().sum(), [xc] + [a for a in args if a.requires_grad])
        for a, b in zip(ge, gc):
            assert torch.isfinite(b).all()
            assert ((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-6)) < 2e-2


class _Cfg:
    def __init__(self, ep):
        self.expert_model_parallel_size = ep


def test_expert_parallel_params_not_dense_allreduced():
    m = eg.ExpPGLU(config=_Cfg(4), num_local_experts=8)
    assert all(getattr(p, "allreduce", True) is False for p in m.parameters())
    for cfg, E in ((_Cfg(1), 8), (_Cfg(4), 1), (None, 1)):
        m = eg.ExpXRLGLU(config=cfg, num_local_experts=E)
        assert all(getattr(p, "allreduce", True) for p in m.parameters())


class _Owner:
    class _G:
        def __init__(self, r): self.r = r
        def rank(self): return self.r
    def __init__(self, r): self.ep_group = self._G(r)


def test_glu_stats_learned_params_and_deadzone():
    """Learned coefficients land in learned-gate/*, and the dead zone uses each expert's own p."""
    torch.manual_seed(0)
    E, Fd = 2, 32
    tpe = [512, 512]
    m = eg.ExpPGLU(num_local_experts=E).to(DEV)
    with torch.no_grad():                                  # expert 0: p=1 ; expert 1: p=3
        m.raw_p.copy_(torch.log(torch.expm1(torch.tensor([1.0, 3.0]))))
        m.b.copy_(torch.tensor([0.0, 0.5]))
    gs.begin_step(True, num_layers=1, num_experts=4)
    gs.set_layer(1)
    z2 = torch.full((sum(tpe), Fd), 5.0, device=DEV)      # |x-b| ~ 5: p=1 live, p=3 dead (fp4)
    z1 = torch.randn_like(z2)
    g = m(z2, tpe)
    gs.record_glu("routed", z1, z2, g, g * z1, m, tpe, module=_Owner(1))   # EP rank 1 -> 2,3
    sc = gs.reduce_and_log(iteration=1)
    # f'(5) = 1/6 (p=1, live) ; (1/5.5)^3 = 0.006 < 1/12 (p=3, dead) -> half the entries dead
    assert abs(sc["layers-gate-deadzone/routed/deadzone_fp4/L01"] - 0.5) < 1e-9
    assert abs(sc["learned-gate/routed/p_mean"] - 2.0) < 1e-5
    assert abs(sc["learned-gate/routed/p_min"] - 1.0) < 1e-5
    assert abs(sc["learned-gate/routed/p_max"] - 3.0) < 1e-5
    assert abs(sc["learned-gate/routed/p_std"] - 1.0) < 1e-5
    assert abs(sc["layers-learned-gate/routed/b_mean/L01"] - 0.25) < 1e-6
    assert "layers-learned-gate/routed/p_max/L01" in sc
    assert not any(k.startswith("layers-learned-gate/routed/p_std") for k in sc)


def test_glu_stats_fixed_gate_unchanged():
    """A plain callable still goes through gate_fn(z) (no expert_ids)."""
    gs.begin_step(True, num_layers=1, num_experts=1)
    gs.set_layer(1)
    z2 = torch.randn(256, 16, device=DEV)
    z1 = torch.randn_like(z2)
    f = eg.GATES["exp_l0m1r0_p1_b0"]
    gs.record_glu("dense", z1, z2, f(z2), f(z2) * z1, f, None)
    sc = gs.reduce_and_log(iteration=1)
    assert "gate-deadzone/dense/deadzone_fp4" in sc
    assert not any(k.startswith("learned-gate") for k in sc)


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
