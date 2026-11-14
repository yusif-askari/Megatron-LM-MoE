"""Routed-expert MLP fc2(weighted SwiGLU(fc1(x), p)) with the fused forward (mxfp4_alto.fused_swiglu_mlp)
against the unfused path (fc1, Megatron's weighted_bias_swiglu_impl, fc2), on the hardware backend and the
emulation, at the 1.5B and 5B latent-MoE shapes; then the time of both hardware paths.

1. Stochastic rounding off (mxfp4_emulation._FORCE_RNE): out, dx, dprobs, dW1, dW2 compared bitwise:
   fused vs unfused (hardware), unfused hardware vs emulation.
2. Time (CUDA events, median): forward only and forward + backward, 64 local experts, realistic token
   counts, weight cache on (as in training, one step = many micro-batches).

Run: ALTO_DIR=... python tests/unit_tests/test_mxfp4_fused_swiglu.py out.json   (one gfx950 GPU)."""
import json
import math
import statistics
import sys

import torch

from megatron.core import mxfp4_alto
from megatron.core import mxfp4_emulation as emu
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

dev = torch.device("cuda", 0)
g = torch.Generator(device=dev).manual_seed(0)


class _Bf16Grouped(torch.autograd.Function):
    """bf16 grouped GEMM with torch._grouped_mm (forward, dgrad, wgrad), stacked weights [E, N, K]."""

    @staticmethod
    def forward(ctx, x, w, offs):
        ctx.save_for_backward(x, w, offs)
        return torch._grouped_mm(x, w.transpose(1, 2), offs=offs)

    @staticmethod
    def backward(ctx, dy):
        x, w, offs = ctx.saved_tensors
        return torch._grouped_mm(dy, w, offs=offs), torch._grouped_mm(dy.t(), x, offs=offs), None


def bf16_torch(x, m, p, w1s, w2s, offs):
    """bf16 baseline: torch._grouped_mm fc1, Megatron's weighted SwiGLU, fc2."""
    return _Bf16Grouped.apply(weighted_bias_swiglu_impl(_Bf16Grouped.apply(x, w1s, offs), None, p), w2s, offs)


def bf16_te(x, m, p, gl1, gl2):
    """bf16 baseline as in a bf16 training run: TransformerEngine GroupedLinear (what TEGroupedMLP uses),
    Megatron's weighted SwiGLU, GroupedLinear."""
    return gl2(weighted_bias_swiglu_impl(gl1(x, m), None, p), m)


def unfused(x, m, p, w1, w2):
    y = emu.grouped_linear(x, m, w1, "alto", "alto")
    h = weighted_bias_swiglu_impl(y, None, p)
    return emu.grouped_linear(h, m, w2, "alto", "alto")


def fused(x, m, p, w1, w2):
    mxfp4_alto.FUSED_SWIGLU_BWD = True
    return mxfp4_alto.fused_swiglu_mlp(x, m, p, w1, w2)


def fused_fwd_only(x, m, p, w1, w2):
    mxfp4_alto.FUSED_SWIGLU_BWD = False
    return mxfp4_alto.fused_swiglu_mlp(x, m, p, w1, w2)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def run(fn, backend, x, m, p, w1, w2, dout):
    emu.set_backend(backend, "alto", "alto")
    xs = x.detach().clone().requires_grad_()
    ps = p.detach().clone().requires_grad_()
    w1s = [w.detach().clone().requires_grad_() for w in w1]
    w2s = [w.detach().clone().requires_grad_() for w in w2]
    out = fn(xs, m, ps, w1s, w2s)
    out.backward(dout)
    return {"out": out.detach(), "dx": xs.grad, "dprobs": ps.grad,
            "dw1": torch.stack([w.grad for w in w1s]), "dw2": torch.stack([w.grad for w in w2s])}


def operands(m, d, f, n_exp):
    T = sum(m)
    x = torch.randn(T, d, device=dev, generator=g).bfloat16()
    p = torch.rand(T, 1, device=dev, generator=g) * 0.3          # router probs (fp32), [T, 1]
    w1 = [(torch.randn(2 * f, d, device=dev, generator=g) * d ** -0.5).bfloat16() for _ in range(n_exp)]
    w2 = [(torch.randn(d, f, device=dev, generator=g) * f ** -0.5).bfloat16() for _ in range(n_exp)]
    dout = (torch.randn(T, d, device=dev, generator=g) * 1e-3).bfloat16()
    return x, p, w1, w2, dout


def mism(a, b):
    return {k: int((a[k] != b[k]).sum()) for k in a}


def diff_size(a, b):
    """For mismatching elements: largest |a - b| in units of the bf16 spacing (ulp) at |b|, and the
    largest |a - b| relative to the tensor's largest |b|."""
    out = {}
    for k in a:
        x, y = a[k].float(), b[k].float()
        d = (x - y).abs()
        if not (d > 0).any():
            continue
        ulp = torch.exp2(torch.floor(torch.log2(y.abs().clamp_min(1e-30))) - 7)
        out[k] = {"max_ulps": (d / ulp).max().item(), "max_rel_to_tensor_max": (d.max() / y.abs().max()).item()}
    return out


def timed(fn, warm=5, iters=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


res = {}
for name, (d, f) in {"1.5b (latent 384, expert 448)": (384, 448), "5b (latent 640, expert 704)": (640, 704)}.items():
    # ---- exactness, 16 experts with uneven counts (0, small, not multiples of 128)
    m = [0, 5, 37, 128, 300, 1000, 2048, 129, 64, 31, 4000, 700, 1, 255, 513, 3000]
    x, p, w1, w2, dout = operands(m, d, f, len(m))
    emu._FORCE_RNE = True
    emu.new_step()
    a = run(fused, "alto", x, m, p, w1, w2, dout)
    a1 = run(fused_fwd_only, "alto", x, m, p, w1, w2, dout)
    b = run(unfused, "alto", x, m, p, w1, w2, dout)
    c = run(unfused, "emulate", x, m, p, w1, w2, dout)
    emu.end_step()
    # stochastic rounding on: gradient error against the round-to-nearest result, fused vs unfused
    emu._FORCE_RNE = False
    sr = {"fused": [], "unfused": []}
    for rep in range(4):
        emu.new_step()
        for label, fn in (("fused", fused), ("unfused", unfused)):
            s = run(fn, "alto", x, m, p, w1, w2, dout)
            sr[label].append([rel(s[k], b[k]) for k in ("dx", "dprobs", "dw1", "dw2")])
        emu.end_step()
    sr_mean = {k: torch.tensor(v).mean(0).tolist() for k, v in sr.items()}
    r = {"fused_vs_unfused_alto": mism(a, b), "fused_fwd_only_vs_unfused_alto": mism(a1, b),
         "fused_vs_unfused_dprobs_rel": rel(a["dprobs"], b["dprobs"]),
         "sr_rel_err_vs_rne (dx, dprobs, dw1, dw2)": sr_mean,
         "sr_err_ratio_fused_over_unfused": [round(f / u, 4) for f, u in zip(sr_mean["fused"], sr_mean["unfused"])],
         "unfused_alto_vs_emulation": mism(b, c),
         "unfused_alto_vs_emulation_size": diff_size(b, c),
         "max_abs_out_fused_vs_unfused": (a["out"].float() - b["out"].float()).abs().max().item()}
    # ---- time: 64 local experts, routed rows per GPU per micro-batch as in the runs
    T = 131072 if d == 384 else 98304
    wts = torch.exp(0.8 * torch.randn(64, generator=torch.Generator().manual_seed(1)))
    m = torch.multinomial(wts / wts.sum(), T, replacement=True,
                          generator=torch.Generator().manual_seed(2)).bincount(minlength=64).tolist()
    x, p, w1, w2, dout = operands(m, d, f, 64)
    emu.set_backend("alto", "alto", "alto")
    xs = x.clone().requires_grad_(); ps = p.clone().requires_grad_()
    w1s = [w.clone().requires_grad_() for w in w1]; w2s = [w.clone().requires_grad_() for w in w2]
    # bf16 baselines (same operands)
    offs = torch.tensor(list(__import__("itertools").accumulate(m)), device=dev, dtype=torch.int32)
    w1st = torch.stack(w1).requires_grad_(); w2st = torch.stack(w2).requires_grad_()
    bf = {"bf16_torch_grouped_mm": (lambda: bf16_torch(xs, m, ps, w1st, w2st, offs),
                                    lambda: bf16_torch(x, m, p, w1st, w2st, offs), [w1st, w2st])}
    try:
        import transformer_engine.pytorch as te
        gl1 = te.GroupedLinear(64, d, 2 * f, bias=False, params_dtype=torch.bfloat16, device="cuda")
        gl2 = te.GroupedLinear(64, f, d, bias=False, params_dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            for e in range(64):
                getattr(gl1, f"weight{e}").copy_(w1[e]); getattr(gl2, f"weight{e}").copy_(w2[e])
        bf["bf16_te_grouped_linear"] = (lambda: bf16_te(xs, m, ps, gl1, gl2), lambda: bf16_te(x, m, p, gl1, gl2),
                                        list(gl1.parameters()) + list(gl2.parameters()))
    except ImportError:
        pass
    for label, (fb, f_, params) in bf.items():
        def fwd():
            with torch.no_grad():
                f_()

        def fwd_bwd():
            for t_ in [xs, ps, *params]:
                t_.grad = None
            fb().backward(dout)
        r[f"{label}_fwd_ms"] = timed(fwd)
        r[f"{label}_fwd_bwd_ms"] = timed(fwd_bwd)
    emu.new_step()                                   # weight cache on: one step, many micro-batches
    for label, fn in (("unfused", unfused), ("fused_fwd_only", fused_fwd_only), ("fused", fused)):
        def fwd():
            with torch.no_grad():
                fn(x, m, p, w1, w2)

        def fwd_bwd():
            for t in [xs, ps, *w1s, *w2s]:
                t.grad = None
            fn(xs, m, ps, w1s, w2s).backward(dout)
        r[f"{label}_fwd_ms"] = timed(fwd)
        r[f"{label}_fwd_bwd_ms"] = timed(fwd_bwd)
    emu.end_step()
    r["speedup_fwd"] = r["unfused_fwd_ms"] / r["fused_fwd_ms"]
    base = "bf16_te_grouped_linear" if "bf16_te_grouped_linear_fwd_ms" in r else "bf16_torch_grouped_mm"
    r["vs_bf16_baseline"] = base
    for v in ("unfused", "fused_fwd_only", "fused"):
        r[f"{v}_vs_bf16_fwd"] = r[f"{base}_fwd_ms"] / r[f"{v}_fwd_ms"]
        r[f"{v}_vs_bf16_fwd_bwd"] = r[f"{base}_fwd_bwd_ms"] / r[f"{v}_fwd_bwd_ms"]
    r["speedup_fwd_bwd_fused_fwd_only"] = r["unfused_fwd_bwd_ms"] / r["fused_fwd_only_fwd_bwd_ms"]
    r["speedup_fwd_bwd"] = r["unfused_fwd_bwd_ms"] / r["fused_fwd_bwd_ms"]
    r["routed_rows"] = T
    res[name] = r
    print(name, json.dumps(r), flush=True)

ok = all(v == 0 for r in res.values() for k, v in r["fused_vs_unfused_alto"].items() if k != "dprobs")
ok = ok and all(v == 0 for r in res.values() for v in r["fused_fwd_only_vs_unfused_alto"].values())
ok = ok and all(r["fused_vs_unfused_dprobs_rel"] < 1e-5 for r in res.values())
ok = ok and all(0.9 < q < 1.1 for r in res.values() for q in r["sr_err_ratio_fused_over_unfused"])
print("PASS (fused = unfused bitwise except d(probs) at fp32 rounding; SR statistics equal)" if ok else "FAIL")
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_mxfp4_fused_swiglu.json", "w"), indent=1)
sys.exit(0 if ok else 1)
