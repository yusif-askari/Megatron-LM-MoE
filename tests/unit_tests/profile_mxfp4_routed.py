"""Per-kernel GPU time of the routed-expert MLP (fc1 -> weighted SwiGLU -> fc2, forward + backward), MXFP4 fused
path with row maps vs bf16 TransformerEngine, at the 1.5B and 5B shapes; and the cost of keeping y (fc1 output)
for the backward: the fused fc1 kernel timed with and without writing y.

torch.profiler, ITERS calls of forward + backward after warm-up; GPU kernel time per call (sum over the
kernel's launches / ITERS). Run: ALTO_DIR=... python profile_mxfp4_routed.py out.json   (one gfx950 GPU)."""
import collections
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mxfp4_fused_swiglu as T  # noqa: E402
from megatron.core import mxfp4_alto  # noqa: E402
from megatron.core import mxfp4_emulation as emu  # noqa: E402

ITERS, WARM = 20, 10
dev = T.dev


def short(name):
    for key, label in (("_dual_quant", "dual quantization (x or dout)"),
                       ("_fc1_swiglu_quant", "fused fc1 + SwiGLU + Q(h)  [fwd]"),
                       ("_fc2_dgrad_swiglu_bwd", "fused fc2 dgrad + SwiGLU bwd + Q(dy1)  [bwd]"),
                       ("grouped_gemm_forward", "fc2 GEMM  [fwd]"),
                       ("backward_dx", "fc1 dgrad GEMM  [bwd]"),
                       ("wgrad_seg", "wgrad GEMM (fc1 + fc2)  [bwd]"),
                       ("backward_dw", "wgrad GEMM (fc1 + fc2)  [bwd]"),
                       ("_wgrad", "wgrad GEMM (fc1 + fc2)  [bwd]"),
                       ("convert", "weight quantization"),
                       ("_quantize", "weight quantization")):
        if key in name:
            return label
    return name[:90]


def kernel_table(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    per = collections.defaultdict(float)
    launches = collections.defaultdict(int)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name] += e.device_time / 1000.0          # us -> ms
            launches[e.name] += 1
    rows = sorted(((n, t / ITERS, launches[n] / ITERS) for n, t in per.items()), key=lambda r: -r[1])
    grouped = collections.defaultdict(float)
    for n, t, _ in rows:
        grouped[short(n)] += t
    return {"total_ms": sum(t for _, t, _ in rows),
            "grouped_ms": dict(sorted(grouped.items(), key=lambda kv: -kv[1])),
            "kernels": [{"name": n[:160], "ms_per_call": t, "launches_per_call": c} for n, t, c in rows]}


def time_med(fn, n=50, rounds=5):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(n):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        meds.append(statistics.median(ts))
    return {"median_ms": statistics.median(meds), "min_ms": min(meds), "max_ms": max(meds)}


def main():
    import transformer_engine.pytorch as te
    mxfp4_alto._alto()                     # registers the "alto" package from $ALTO_DIR
    from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.fused_swiglu import fc1_swiglu_quant

    res = {"device": torch.cuda.get_device_name(0), "host": os.uname().nodename, "iters": ITERS}
    for name, (d, f, rows) in {"1.5b": (384, 448, 131072), "5b": (640, 704, 98304)}.items():
        wts = torch.exp(0.8 * torch.randn(64, generator=torch.Generator().manual_seed(1)))
        m = torch.multinomial(wts / wts.sum(), rows, replacement=True,
                              generator=torch.Generator().manual_seed(2)).bincount(minlength=64).tolist()
        x, p, w1, w2, dout = T.operands(m, d, f, 64)
        xs = x.clone().requires_grad_(); ps = p.clone().requires_grad_()
        w1s = [w.clone().requires_grad_() for w in w1]; w2s = [w.clone().requires_grad_() for w in w2]
        gl1 = te.GroupedLinear(64, d, 2 * f, bias=False, params_dtype=torch.bfloat16, device="cuda")
        gl2 = te.GroupedLinear(64, f, d, bias=False, params_dtype=torch.bfloat16, device="cuda")
        te_params = list(gl1.parameters()) + list(gl2.parameters())
        emu.set_backend("alto", "alto", "alto")
        emu.new_step()

        def fused_fb():
            for t_ in [xs, ps, *w1s, *w2s]:
                t_.grad = None
            T.fused(xs, m, ps, w1s, w2s).backward(dout)

        def te_fb():
            for t_ in [xs, ps, *te_params]:
                t_.grad = None
            T.bf16_te(xs, m, ps, gl1, gl2).backward(dout)

        r = {"mxfp4_fused_row_maps": kernel_table(fused_fb), "bf16_te": kernel_table(te_fb)}

        # cost of keeping y: the fused fc1 kernel alone, with and without the y store
        A = mxfp4_alto._alto()
        rows_, idx, tp = mxfp4_alto._layout(m, dev)
        src = mxfp4_alto._src_rows(rows_, tp)
        h = mxfp4_alto._h_bf16(dev)
        xq, xsc, _, _ = A.dual_quantize(x, h, use_sr=False, src_rows=src)
        w1q, w1sc = mxfp4_alto._stack_tile(w1, "fc1")
        pp = p.reshape(-1).contiguous()
        r["fc1_kernel_with_y"] = time_med(lambda: fc1_swiglu_quant(xq, xsc, w1q, w1sc, idx, pp, h, src_rows=src))
        r["fc1_kernel_without_y"] = time_med(
            lambda: fc1_swiglu_quant(xq, xsc, w1q, w1sc, idx, pp, h, src_rows=src, store_y=False))
        r["y_bytes_MB"] = tp * 2 * f * 2 / 1e6
        r["Tp"] = tp
        emu.end_step()
        res[name] = r
        print(name, json.dumps({"fused_total": r["mxfp4_fused_row_maps"]["total_ms"],
                                "fused_grouped": r["mxfp4_fused_row_maps"]["grouped_ms"],
                                "bf16_total": r["bf16_te"]["total_ms"],
                                "fc1_with_y": r["fc1_kernel_with_y"], "fc1_without_y": r["fc1_kernel_without_y"],
                                "y_MB": r["y_bytes_MB"]}), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
