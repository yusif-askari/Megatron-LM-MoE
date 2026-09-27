"""
Numerical checks for megatron/core/transformer/glu_stats.py (the --log-glu-stats module).

glu_stats only depends on torch, so this loads it by path and runs without the rest of
Megatron. Needs a GPU (the tracker allocates on cuda); torchao is optional (used as the
reference for the MXFP4 / NVFP4 simulators when available).

    python tests/unit_tests/transformer/test_glu_stats.py
"""
import importlib.util
import math
import os
import sys

import torch

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


def test_e2m1_grid_exact():
    g = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -6], device=DEV)
    assert torch.equal(gs._round_e2m1(g), g)
    assert gs._round_e2m1(torch.tensor([7.0, 0.2, 0.3, 5.2], device=DEV)).tolist() == [6, 0, .5, 6]


def test_fp8_blockwise_error():
    x = torch.randn(64, 448, device=DEV)
    xb, q = gs.fake_quant(x, "fp8")
    rel = ((q - xb).norm() / xb.norm()).item()
    assert rel < 0.05, rel                       # e4m3: ~3 mantissa bits


def _torchao_mx(x):
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    try:
        t = MXTensor.to_mx(x, torch.float4_e2m1fn_x2, 32)
    except TypeError:
        t = MXTensor.to_mx(x, torch.float4_e2m1fn_x2, block_size=32)
    return t.dequantize(torch.float32)


def test_mxfp4_matches_torchao():
    try:
        import torchao  # noqa: F401
    except ImportError:
        print("  (torchao not installed: skipping MXFP4 reference check)")
        return
    x = torch.randn(256, 512, device=DEV) * torch.logspace(-3, 1, 512, device=DEV)
    _, q = gs.fake_quant(x, "mxfp4")
    ref = _torchao_mx(x)
    q = q.reshape_as(x)
    mism = (q != ref).float().mean().item()
    print(f"  MXFP4 vs torchao: mismatch fraction {mism:.2e}, max|diff| {(q-ref).abs().max().item():.3g}")
    assert mism < 1e-3, mism


def test_nvfp4_matches_torchao():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor, per_tensor_amax_to_scale
    except ImportError:
        print("  (torchao NVFP4 not available: skipping NVFP4 reference check)")
        return
    x = torch.randn(256, 512, device=DEV) * torch.logspace(-3, 1, 512, device=DEV)
    amax = x.abs().amax()
    _, q = gs.fake_quant(x, "nvfp4", amax)
    t = NVFP4Tensor.to_nvfp4(x, per_tensor_scale=per_tensor_amax_to_scale(amax))
    ref = t.dequantize(torch.float32)
    q = q.reshape_as(x)
    mism = (q != ref).float().mean().item()
    rel = ((q - ref).norm() / ref.norm()).item()
    print(f"  NVFP4 vs torchao: mismatch fraction {mism:.2e}, rel diff {rel:.2e}")
    assert mism < 1e-2 and rel < 1e-2, (mism, rel)


def test_tracker_end_to_end():
    torch.manual_seed(0)
    E, F, rows = 4, 64, 4096                       # 4 experts x 1024 tokens, 64 hidden units
    tpe = [rows // E] * E
    gate = eg.GATES["exp_l0m1r0_p1_b0"]            # p=1 bump: |f'| = 1/(1+|x|)
    gs.begin_step(True, num_layers=2, num_experts=16)   # router vector width (16 toy experts)
    gs.set_layer(1)
    sigma = 20.0
    z2 = (torch.randn(rows, F, device=DEV) * sigma)
    z2[:, 0] = 50.0                                # unit 0 of every expert: always in the fp4 dead zone
    z2 = z2.requires_grad_(True)
    z1 = torch.randn(rows, F, device=DEV).requires_grad_(True)
    g = gate(z2)
    h = g * z1
    h_planted = h.clone()
    gs.record_glu("routed", z1, z2, g, h, gate, tpe)
    (h.pow(2).sum()).backward()                    # fires the dz1 / dz2 hooks
    rm = torch.zeros(rows, 16, dtype=torch.bool, device=DEV)
    rm[:, :8] = True                               # 8 of 16 experts get every token, 8 get none
    gs.record_router(1, rm)
    gs.set_layer(2)
    gs.record_moments("resid", torch.randn(8, 2, 32, device=DEV) * 3.0)
    gs.record_logits(torch.randn(128, 2, 100, device=DEV))
    sc = gs.reduce_and_log(iteration=5)
    assert sc and all(math.isfinite(v) for v in sc.values()), "non-finite scalars"

    # dead zone (fp4 floor 1/12): |f'| < 1/12  <=>  |x| > 11; unit 0 is always dead.
    p = 2 * (1 - 0.5 * (1 + math.erf(11 / sigma / math.sqrt(2))))
    expect = (1 / F) * 1.0 + (1 - 1 / F) * p
    got = sc["layers-gate-deadzone/routed/deadzone_fp4/L01"]
    print(f"  dead zone fp4: got {got:.4f} expected {expect:.4f}")
    assert abs(got - expect) < 0.01, (got, expect)
    du = sc["layers-gate-deadzone/routed/dead_units_fp4/L01"]
    print(f"  dead units fp4: got {du:.4f} expected {1/F:.4f} (one unit per expert)")
    assert abs(du - 1 / F) < 1e-6, du
    # moments of h against a direct computation
    rms = h_planted.float().pow(2).mean().sqrt().item()
    assert abs(sc["layers-gate-moments/routed/h_rms/L01"] - rms) / rms < 1e-4
    assert "gate-quant/routed/dz2_mxfp4_flush" in sc and "gate-moments/routed/dz1_rms" in sc
    # router: 8/16 experts idle -> dead 0.5, loads {rows, 0} -> cv 1, max/mean 2
    assert abs(sc["layers-router/dead_experts/L01"] - 0.5) < 1e-9
    assert abs(sc["layers-router/load_cv/L01"] - 1.0) < 1e-9
    assert abs(sc["router/load_max_over_mean"] - 2.0) < 1e-9
    assert abs(sc["layers-residual/rms/L02"] - 3.0) < 0.3
    assert "logits/absmax" in sc and not any(k.startswith("glu/") for k in sc)
    assert not gs.is_active()


def test_outlier_channel_ratio():
    torch.manual_seed(1)
    F = 128
    gs.begin_step(True, num_layers=1, num_experts=1)
    gs.set_layer(1)
    z2 = torch.randn(2048, F, device=DEV)
    z1 = torch.randn(2048, F, device=DEV)
    gate = eg.GATES["exp_l0m1r0_p1_b0"]
    g = gate(z2)
    h = g * z1
    h[:, 7] *= 100.0                                # one outlier channel
    gs.record_glu("dense", z1, z2, g, h, gate, None)
    sc = gs.reduce_and_log(iteration=1)
    r = sc["gate-outliers/dense/h_outlier_channel_ratio"]
    print(f"  outlier-channel ratio with a x100 channel: {r:.1f}")
    assert 60 < r < 160, r


def test_inactive_is_noop():
    gs.begin_step(False, num_layers=1)
    x = torch.randn(4, 8, device=DEV)
    gs.record_glu("dense", x, x, x, x, torch.nn.functional.silu, None)
    assert not gs._T.sums and gs.reduce_and_log(1) is None


def test_weight_2d_blocks_transpose_invariant():
    torch.manual_seed(2)
    w = torch.randn(64, 48, device=DEV) * torch.logspace(-2, 1, 48, device=DEV)
    for fmt in ("nvfp4", "fp8"):
        assert torch.equal(gs.qdq_weight(w, fmt).t(), gs.qdq_weight(w.t(), fmt)), fmt
    # MXFP4 is 1D: blocks follow the contraction dim
    assert torch.equal(gs.qdq_weight(w, "mxfp4", 0), gs.qdq(w.t(), "mxfp4").t())


def test_rht_is_orthogonal():
    torch.manual_seed(3)
    a, b = torch.randn(8, 64, device=DEV), torch.randn(5, 64, device=DEV)
    ref = a @ b.t()
    got = gs._rht16(a) @ gs._rht16(b).t()
    assert ((got - ref).norm() / ref.norm()).item() < 1e-5


def test_stochastic_rounding_unbiased():
    g = torch.Generator(device=DEV).manual_seed(0)
    y = torch.full((400_000,), 0.3, device=DEV)
    m = gs._sr_e2m1(y, g).mean().item()
    print(f"  SR mean of 0.3: {m:.4f}")
    assert abs(m - 0.3) < 0.003, m
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -.5, -6], device=DEV)
    assert torch.equal(gs._sr_e2m1(grid, g), grid)


def test_nvfp4_scale_underflow_and_max_over_median():
    gs.begin_step(True, num_layers=1)
    x = torch.ones(1, 32, device=DEV)
    x[0, :16] = 1e-8                     # block 0: tiny, its E4M3 scale underflows
    x[0, 16] = 1e4                       # block 1 holds the tensor max
    gs._tensor_stats("t", 1, x, quant=True)
    S = gs._T.sums
    assert S["t/nvfp4/blocks"][1].item() == 2
    assert S["t/nvfp4/scale_underflow"][1].item() == 1
    gs.begin_step(True, num_layers=1)
    y = torch.ones(1, 4096, device=DEV); y[0, 7] = 100.0
    gs._tensor_stats("u", 1, y, quant=True)
    r = gs._T.sums["u/mom_sum"][1].item()
    assert abs(r - 100.0) < 1e-6, r


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_experts_step(stash_input=True, corrupt_input=False):
    torch.manual_seed(4)
    F, K, H, n = 32, 48, 40, 256
    gate = eg.GATES["exp_l0m1r0_p1_b1"]
    W1 = [torch.randn(2 * F, K, device=DEV) / K ** .5 for _ in range(2)]
    W1[0][F] = W1[0][0]                  # expert 0, unit 0: gate row == value row
    W1 = [w.requires_grad_(True) for w in W1]
    W2 = [(torch.randn(H, F, device=DEV) / F ** .5).requires_grad_(True) for _ in range(2)]
    mod = _NS(linear_fc1=_NS(weight0=W1[0], weight1=W1[1]),
              linear_fc2=_NS(weight0=W2[0], weight1=W2[1]))
    x = torch.randn(2 * n, K, device=DEV)
    out = torch.cat([x[:n] @ W1[0].t(), x[n:] @ W1[1].t()])
    z2, z1 = torch.chunk(out, 2, dim=-1)
    g = gate(z2); h = g * z1
    gs.begin_step(True, num_layers=2, num_experts=2)
    gs.set_layer(1)
    if stash_input:
        gs.stash_fc1_input(x * 3.0 if corrupt_input else x)
    gs.record_glu("routed", z1, z2, g, h, gate, [n, n], fc1_out=out, module=mod)
    y = torch.cat([h[:n] @ W2[0].t(), h[n:] @ W2[1].t()])
    y.pow(2).sum().backward()
    return gs.reduce_and_log(iteration=7)


def test_gemm_weight_align_end_to_end():
    sc = _fake_experts_step()
    assert all(math.isfinite(v) for v in sc.values())
    wc, zc = sc["gate-align/routed/w_cos_abs_max"], sc["gate-align/routed/z_corr_abs_max"]
    print(f"  planted unit: max|cos(Wg,Wv)| {wc:.4f}  max|corr(z1,z2)| {zc:.4f}")
    assert abs(wc - 1) < 1e-5 and abs(zc - 1) < 1e-4
    assert sc["gate-align/routed/w_cos_abs_mean"] < 0.5
    e = {k.split("/")[-1]: v for k, v in sc.items() if k.startswith("gemm-quant/routed/")}
    print("  GEMM relerr: " + ", ".join(f"{k.replace('_relerr', '')} {v:.3g}"
                                          for k, v in sorted(e.items())))
    for op in ("fc2_fprop", "fc1_dgrad", "fc1_wgrad"):
        assert e[f"{op}_fp8_relerr"] < e[f"{op}_nvfp4_relerr"] < 0.5, op
        assert e[f"{op}_nvfp4_relerr"] < e[f"{op}_mxfp4_relerr"] * 1.5, op
    assert "fc1_wgrad_nvfp4_plain_relerr" in e and "fc1_wgrad_nvfp4_rht_relerr" in e
    assert "skipped_samples" not in e
    for w in ("fc1", "fc2"):
        assert sc[f"weight-quant/routed/{w}_fp8_relerr"] < sc[f"weight-quant/routed/{w}_nvfp4_relerr"]
    assert "layers-gemm-quant/routed/fc1_wgrad_nvfp4_relerr/L01" in sc


def test_gemm_skipped_when_input_mismatch():
    sc = _fake_experts_step(corrupt_input=True)
    assert sc["gemm-quant/routed/skipped_samples"] >= 1
    assert "gemm-quant/routed/fc1_dgrad_nvfp4_relerr" not in sc
    assert "gemm-quant/routed/fc2_fprop_nvfp4_relerr" in sc      # needs no fc1 input


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
