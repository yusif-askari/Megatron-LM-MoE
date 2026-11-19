"""Bump gates of the activation sweep (exp_l0m{M}r0_p{p}_b{b}, exp_gates.BumpGate).

A (standalone; loads exp_gates.py by path): value and slope against the fp64 builder formula, including
  x = b where autograd through _builder_gate gives slope 0 instead of M; the fused GLU
  h = bf16(g(x_glu) x_linear prob) and its backward against fp64 truth, eager and under torch.compile
  (training: jit_fuser = torch.compile).
B (needs Megatron + one GPU; torchrun --nproc_per_node=1 -m pytest ...): TEGroupedMLP and MLP route the
  bump gates to weighted_glu, with the router prob / per-token scale applied exactly once.
"""
import importlib.util
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_spec = importlib.util.spec_from_file_location("exp_gates_bump_test",
                                               os.path.join(REPO, "megatron/core/exp_gates.py"))
eg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eg)

D = torch.float64
BUMPS = {n: prm for n, (fn, prm) in eg.BUILDER_GATES.items() if isinstance(fn, eg.BumpGate)}
NEED_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _raw(f):
    return getattr(f, "_torchdynamo_orig_callable", f)


@pytest.fixture(params=["eager", "compiled"])
def mode(request, monkeypatch):
    for name in ("_bump_val", "_bump_val_bwd", "_bump_glu_fwd", "_bump_glu_bwd"):
        f = _raw(getattr(eg, name))
        monkeypatch.setattr(eg, name, f if request.param == "eager" else torch.compile(f))
    torch._dynamo.reset()
    return request.param


def _truth(L, M, R, p, b):
    def val(x):
        return eg._antideriv(x - b, L, M, R, p) - eg._antideriv(torch.full_like(x, -b), L, M, R, p)

    def slope(x):
        return eg._grad(x - b, L, M, R, p)          # explicit form: M at x = b
    return val, slope


def test_registry():
    assert sorted(BUMPS) == sorted(["exp_l0m1r0_p0.5_b0", "exp_l0m1r0_p0.5_b1", "exp_l0m1r0_p1_b0",
                                    "exp_l0m1r0_p1_b1", "exp_l0m1r0_p2_b0", "exp_l0m1r0_p2_b1",
                                    "exp_l0m1r0_p1.5_b0"])
    for n, (fn, (L, M, R, p, b)) in eg.BUILDER_GATES.items():      # others keep the builder path
        assert isinstance(fn, eg.BumpGate) == (L == 0.0 and R == 0.0)
        assert eg.GATES[n] is fn


def test_value_and_slope_fp64(mode):
    """fp64 in, fp64 out: g(0) = 0, g and g' agree with the builder's formulas; slope M at x = b."""
    x = torch.cat([torch.linspace(-12, 12, 4001, dtype=D),
                   torch.tensor([0.0, 1.0, -1.0, 1e-9, -1e-9, 1.0 + 1e-12], dtype=D)])
    for name, (L, M, R, p, b) in BUMPS.items():
        val, slope = _truth(L, M, R, p, b)
        xx = x.clone().requires_grad_(True)
        g = eg.GATES[name](xx)
        (gp,) = torch.autograd.grad(g.sum(), xx)
        assert g.dtype == D
        ref = val(x)                     # exprel series: < 1e-9 relative (sized for fp32)
        assert ((g - ref).abs() / (1.0 + ref.abs())).max() < 2e-9, name
        assert (gp - slope(x)).abs().max() < 1e-12, name
        assert g[x == 0.0].abs().max() == 0.0, name
        xb = torch.tensor([b], dtype=D, requires_grad=True)
        (sb,) = torch.autograd.grad(eg.GATES[name](xb).sum(), xb)
        assert sb.item() == M, (name, sb.item())
        # the builder path this replaces: autograd gives L = 0 at x = b
        xo = torch.tensor([b], dtype=D, requires_grad=True)
        (so,) = torch.autograd.grad(_raw(eg._builder_gate)(xo, L, M, R, p, b).sum(), xo)
        assert so.item() == L


def test_no_cancellation_near_zero(mode):
    """b != 0: g(x) for tiny x is ~ x B^-p M; the subtraction F_p(x - b) - F_p(-b) in fp32 lost it."""
    x = torch.tensor([1e-6, -3e-5, 2e-4, 1e-3], dtype=torch.float32)
    for name, (L, M, R, p, b) in BUMPS.items():
        if b == 0.0:
            continue
        ref = _truth(L, M, R, p, b)[0](x.double())
        got = eg.GATES[name](x).double()
        assert ((got - ref).abs() / ref.abs()).max() < 1e-6, name


def _inputs(dev, n=2048, f=704, ties=0.02):
    gen = torch.Generator(device=dev).manual_seed(1)
    y = (torch.randn(n, 2 * f, device=dev, generator=gen) * 2.0).bfloat16()
    y[:, :f] = (y[:, :f].float() * torch.rand(n, 1, device=dev, generator=gen).mul(4)).bfloat16()
    prob = torch.rand(n, 1, device=dev, generator=gen) * 0.3
    dh = (torch.randn(n, f, device=dev, generator=gen) * 1e-2).bfloat16()
    tie = torch.rand(n, f, device=dev, generator=gen) < ties          # gate inputs exactly at b = 1
    return y, prob, dh, tie


def _rel(a, r):
    return ((a.double() - r).norm() / r.norm()).item()


@NEED_GPU
def test_weighted_glu_vs_fp64(mode):
    """bf16 fc1 output -> h with one rounding; backward near the bf16 floor (~1.7e-3), also with 2%
    of the gate inputs exactly at x = b = 1 (the old path dropped their input gradient)."""
    dev = torch.device("cuda")
    y0, prob, dh, tie = _inputs(dev)
    f = y0.shape[1] // 2
    for name, (L, M, R, p, b) in BUMPS.items():
        val, slope = _truth(L, M, R, p, b)
        y = y0.clone()
        y[:, :f][tie] = b                                              # ties at x = b
        y.requires_grad_(True)
        pr = prob.clone().requires_grad_(True)
        g, v = torch.chunk(y, 2, dim=-1)
        h = eg.GATES[name].weighted_glu(g, v, pr)
        assert h.dtype == torch.bfloat16
        h.backward(dh)
        x64, v64, p64, dh64 = g.detach().to(D), v.detach().to(D), prob.to(D), dh.to(D)
        g64 = val(x64)
        h_t = g64 * v64 * p64
        dg_t = dh64 * p64 * v64 * slope(x64)
        dv_t = dh64 * p64 * g64
        dp_t = (dh64 * v64 * g64).sum(-1, keepdim=True)
        mis = (h != h_t.to(torch.bfloat16)).float().mean().item()
        dyg, dyv = y.grad[:, :f], y.grad[:, f:]
        assert mis < 2e-4, (name, mode, mis)
        assert _rel(h, h_t) < 1.8e-3, name
        assert _rel(dyg, dg_t) < 1.8e-3, (name, mode, _rel(dyg, dg_t))
        assert _rel(dyv, dv_t) < 1.8e-3, name
        assert _rel(dyg[tie], dg_t[tie]) < 2e-3, (name, mode, _rel(dyg[tie], dg_t[tie]))
        assert _rel(pr.grad, dp_t) < 1e-5, (name, mode, _rel(pr.grad, dp_t))


@NEED_GPU
def test_weighted_glu_without_prob(mode):
    dev = torch.device("cuda")
    y, _, dh, _ = _inputs(dev, n=256, f=128)
    gate = eg.GATES["exp_l0m1r0_p2_b1"]
    g, v = torch.chunk(y, 2, dim=-1)
    ones = torch.ones(y.shape[0], 1, device=dev)
    assert torch.equal(gate.weighted_glu(g, v), gate.weighted_glu(g, v, ones))


# ---- B: wiring in Megatron (TEGroupedMLP, MLP) ------------------------------------------------
try:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils
    from megatron.core import exp_gates as meg_eg
    HAVE_MEGATRON = True
except Exception:                                                     # pragma: no cover
    HAVE_MEGATRON = False


class _Plain:
    """The same gate without weighted_glu: forces the generic GLU path (gate * x_linear, * prob)."""

    def __init__(self, gate):
        self.gate = gate

    def __call__(self, x):
        return self.gate(x)


def _config(gate, **kw):
    return TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4, ffn_hidden_size=96,
                             add_bias_linear=False, gated_linear_unit=True, activation_func=gate,
                             bias_activation_fusion=False, bf16=True, params_dtype=torch.bfloat16, **kw)


def _close(a, b, tol=2e-2):
    if torch.equal(a, b):                                             # e.g. an expert with no tokens
        return True
    return ((a.float() - b.float()).norm() / b.float().norm()).item() < tol


def _run_both(module, gate, call, params, extra):
    """Forward/backward with the fused bump path and with the generic path; returns both results."""
    out = {}
    for tag, fn in (("fused", gate), ("plain", _Plain(gate))):
        module.config.activation_func = fn
        for prm in params + extra:
            prm.grad = None
        o = call()
        o.float().square().sum().backward()
        out[tag] = (o.detach(), [prm.grad.clone() for prm in params + extra])
    module.config.activation_func = gate
    return out


@NEED_GPU
@pytest.mark.skipif(not HAVE_MEGATRON, reason="Megatron not importable")
class TestWiring:

    def setup_method(self, method):
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        self.gate = meg_eg.GATES["exp_l0m1r0_p1_b1"]

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_grouped_mlp(self, monkeypatch):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules)
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer
        cfg = _config(self.gate, num_moe_experts=4, moe_grouped_gemm=True, moe_router_topk=2)
        layer = MoELayer(cfg, get_gpt_layer_with_transformer_engine_submodules(
            4, moe_grouped_gemm=True).mlp.submodules).cuda()
        experts = layer.experts
        assert isinstance(experts, TEGroupedMLP)
        calls = []
        orig = meg_eg.BumpGate.weighted_glu
        monkeypatch.setattr(meg_eg.BumpGate, "weighted_glu",
                            lambda s, *a: calls.append(1) or orig(s, *a))
        tpe = torch.tensor([37, 0, 64, 27])
        n = int(tpe.sum())
        x = (torch.randn(n, 64, device="cuda") * 0.5).bfloat16()
        probs = (torch.rand(n, device="cuda") * 0.9 + 0.05).requires_grad_(True)
        params = [prm for prm in experts.parameters()]
        res = _run_both(experts, self.gate,
                        lambda: experts(x, tokens_per_expert=tpe, permuted_probs=probs)[0],
                        params, [probs])
        assert len(calls) == 1                                        # fused path taken once
        (of, gf), (op, gp) = res["fused"], res["plain"]
        assert _close(of, op), "router prob applied a different number of times"
        for a, b in zip(gf, gp):
            assert _close(a, b), "gradient mismatch"

    def test_dense_mlp(self, monkeypatch):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.transformer.mlp import MLP
        mlp = MLP(_config(self.gate), get_gpt_layer_local_submodules().mlp.submodules).cuda()
        calls = []
        orig = meg_eg.BumpGate.weighted_glu
        monkeypatch.setattr(meg_eg.BumpGate, "weighted_glu",
                            lambda s, *a: calls.append(1) or orig(s, *a))
        x = (torch.randn(16, 3, 64, device="cuda") * 0.5).bfloat16()
        scale = (torch.rand(16, 3, device="cuda") + 0.1).requires_grad_(True)
        params = [prm for prm in mlp.parameters()]
        for kw, extra in (({}, []), ({"per_token_scale": scale}, [scale])):
            calls.clear()
            res = _run_both(mlp, self.gate, lambda: mlp(x, **kw)[0], params, extra)
            assert len(calls) == 1
            (of, gf), (op, gp) = res["fused"], res["plain"]
            assert _close(of, op), kw.keys()
            for a, b in zip(gf, gp):
                assert _close(a, b), kw.keys()


# ---- exp_pglu wiring (learnable gate: exp_gates.ExpPGLU.weighted_glu) ------------------------------
class _NoFused(torch.nn.Module):
    """The same ExpPGLU without weighted_glu: forces the generic GLU path (gate, * x_linear, * prob)."""

    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        return self.gate(x, tokens_per_expert, expert_ids)


def _pglu_config(**kw):
    return TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4, ffn_hidden_size=96,
                             add_bias_linear=False, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
                             exp_learnable_gate="exp_pglu", bias_activation_fusion=False, bf16=True,
                             params_dtype=torch.bfloat16, **kw)


def _run_pglu_both(module, call, params):
    gate, out = module.exp_gate, {}
    for tag in ("fused", "plain"):
        module.exp_gate = gate if tag == "fused" else _NoFused(gate)
        for prm in params:
            prm.grad = None
        o = call()
        o.float().square().sum().backward()
        out[tag] = (o.detach(), [prm.grad.clone() for prm in params])
    module.exp_gate = gate
    return out


@NEED_GPU
@pytest.mark.skipif(not HAVE_MEGATRON, reason="Megatron not importable")
class TestPGLUWiring:

    def setup_method(self, method):
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _count(monkeypatch):
        calls = []
        orig = meg_eg.ExpPGLU.weighted_glu
        monkeypatch.setattr(meg_eg.ExpPGLU, "weighted_glu",
                            lambda s, *a, **k: calls.append(1) or orig(s, *a, **k))
        return calls

    def test_grouped_mlp(self, monkeypatch):
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_submodules)
        from megatron.core.transformer.moe.moe_layer import MoELayer
        cfg = _pglu_config(num_moe_experts=4, moe_grouped_gemm=True, moe_router_topk=2)
        layer = MoELayer(cfg, get_gpt_layer_with_transformer_engine_submodules(
            4, moe_grouped_gemm=True).mlp.submodules).cuda()
        experts = layer.experts
        with torch.no_grad():
            experts.exp_gate.raw_p.copy_(torch.tensor([0.5413, -0.3, 1.2, 0.0]))   # p = 1, ...
            experts.exp_gate.b.copy_(torch.tensor([1.0, 0.0, -0.5, 0.3]))
        calls = self._count(monkeypatch)
        tpe = torch.tensor([37, 0, 64, 27])
        n = int(tpe.sum())
        x = (torch.randn(n, 64, device="cuda") * 0.5).bfloat16()
        probs = (torch.rand(n, device="cuda") * 0.9 + 0.05).requires_grad_(True)
        params = [prm for prm in experts.parameters()] + [probs]
        res = _run_pglu_both(experts, lambda: experts(x, tokens_per_expert=tpe, permuted_probs=probs)[0],
                             params)
        assert len(calls) == 1
        (of, gf), (op, gp) = res["fused"], res["plain"]
        assert _close(of, op), "router prob applied a different number of times"
        for a, b in zip(gf, gp):
            assert _close(a, b), "gradient mismatch"

    def test_dense_mlp(self, monkeypatch):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.transformer.mlp import MLP
        mlp = MLP(_pglu_config(), get_gpt_layer_local_submodules().mlp.submodules).cuda()
        with torch.no_grad():
            mlp.exp_gate.raw_p.fill_(0.2)
            mlp.exp_gate.b.fill_(0.7)
        calls = self._count(monkeypatch)
        x = (torch.randn(16, 3, 64, device="cuda") * 0.5).bfloat16()
        scale = (torch.rand(16, 3, device="cuda") + 0.1).requires_grad_(True)
        params = [prm for prm in mlp.parameters()]
        for kw, extra in (({}, []), ({"per_token_scale": scale}, [scale])):
            calls.clear()
            res = _run_pglu_both(mlp, lambda: mlp(x, **kw)[0], params + extra)
            assert len(calls) == 1
            (of, gf), (op, gp) = res["fused"], res["plain"]
            assert _close(of, op), kw.keys()
            for a, b in zip(gf, gp):
                assert _close(a, b), kw.keys()

    def test_float16module_keeps_fp32(self):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.transformer.mlp import MLP
        from megatron.core.transformer.module import Float16Module
        cfg = _pglu_config()
        mlp = MLP(cfg, get_gpt_layer_local_submodules().mlp.submodules)
        with torch.no_grad():
            mlp.exp_gate.b.fill_(1.0 + 2.0 ** -10)                    # not representable in bf16
        fm = Float16Module(cfg, mlp)
        assert fm.module.exp_gate.raw_p.dtype == torch.float32
        assert fm.module.exp_gate.b.dtype == torch.float32 and fm.module.exp_gate.b[0].item() == 1.0 + 2.0 ** -10
        assert fm.module.linear_fc1.weight.dtype == torch.bfloat16
