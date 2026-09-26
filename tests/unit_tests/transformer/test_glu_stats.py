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
