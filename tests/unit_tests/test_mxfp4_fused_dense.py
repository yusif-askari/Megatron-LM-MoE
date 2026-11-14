"""Shared expert / dense MLP fc2(SwiGLU(fc1(x))) on the fused kernels (mxfp4_alto.fused_swiglu_dense) against the
unfused MXFP4 path it replaces (MXFP4 linear, Megatron's bias_swiglu_impl, MXFP4 linear).

1. Round to nearest: out, dx, dW1, dW2 compared bitwise (plain .grad, no main_grad), and with main_grad
   (gradient-accumulation fusion on both paths): main_grad compared bitwise.
2. Stochastic rounding on: gradient error vs the round-to-nearest result, fused / unfused ratio (4 repeats).
3. Time of forward + backward with main_grad (as in training), interleaved, 5 rounds x 50 calls.
Shapes: 16384 tokens per micro-batch; 1.5B shared expert (768 -> 2x448), 1.5B dense MLP (768 -> 2x1920),
5B shared expert (1280 -> 2x704).  Run: ALTO_DIR=... python test_mxfp4_fused_dense.py out.json"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatron.core import mxfp4_alto  # noqa: E402
from megatron.core import mxfp4_emulation as emu  # noqa: E402
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl  # noqa: E402

dev = torch.device("cuda", 0)


_KEEP = []


def unfused(x, w1, w2):
    return mxfp4_alto.linear(bias_swiglu_impl(mxfp4_alto.linear(x, w1), None), w2)


def fused(x, w1, w2):
    return mxfp4_alto.fused_swiglu_dense(x, w1, w2)


def params(ws, main_grad):
    out = []
    for w in ws:
        p = w.detach().clone().requires_grad_()
        if main_grad:
            p.main_grad = torch.zeros(p.shape, dtype=torch.float32, device=dev)
            p.grad_added_to_main_grad = False
        out.append(p)
    return out


def run(fn, x, w1, w2, dout, main_grad=False):
    xs = x.detach().clone().requires_grad_()
    a, b = params([w1, w2], main_grad)
    _KEEP.extend([a, b])         # the per-step weight caches are keyed by id(): keep the copies alive
    out = fn(xs, a, b)
    out.backward(dout)
    r = {"out": out.detach(), "dx": xs.grad}
    for k, w in (("dw1", a), ("dw2", b)):
        if main_grad:
            r[k] = w.main_grad.clone() if w.grad_added_to_main_grad else w.grad.float()
        else:
            r[k] = w.grad
    return r


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def main():
    emu.set_backend("alto", "alto", "alto")
    mxfp4_alto._alto()
    res, ok = {}, True
    g = torch.Generator(device=dev).manual_seed(0)
    for name, (d, f) in {"1.5b shared expert": (768, 448), "1.5b dense MLP": (768, 1920),
                         "5b shared expert": (1280, 704)}.items():
        T = 16384
        x = torch.randn(T, d, device=dev, generator=g).bfloat16()
        w1 = (torch.randn(2 * f, d, device=dev, generator=g) * d ** -0.5).bfloat16()
        w2 = (torch.randn(d, f, device=dev, generator=g) * f ** -0.5).bfloat16()
        dout = (torch.randn(T, d, device=dev, generator=g) * 1e-3).bfloat16()
        emu._FORCE_RNE = True
        emu.new_step()
        a, b = run(fused, x, w1, w2, dout), run(unfused, x, w1, w2, dout)
        am, bm = run(fused, x, w1, w2, dout, True), run(unfused, x, w1, w2, dout, True)
        emu.end_step()
        r = {"fused_vs_unfused": {k: int((a[k] != b[k]).sum()) for k in a},
             "fused_vs_unfused_main_grad": {k: int((am[k] != bm[k]).sum()) for k in am}}
        ok = ok and all(v == 0 for v in r["fused_vs_unfused"].values())
        ok = ok and all(v == 0 for v in r["fused_vs_unfused_main_grad"].values())
        emu._FORCE_RNE = False
        sr = {"fused": [], "unfused": []}
        for _ in range(4):
            emu.new_step()
            for label, fn in (("fused", fused), ("unfused", unfused)):
                s = run(fn, x, w1, w2, dout)
                sr[label].append([rel(s[k], b[k]) for k in ("dx", "dw1", "dw2")])
            emu.end_step()
        mean = {k: torch.tensor(v).mean(0).tolist() for k, v in sr.items()}
        r["sr_err_ratio_fused_over_unfused (dx, dw1, dw2)"] = [round(p / q, 4) for p, q in
                                                              zip(mean["fused"], mean["unfused"])]
        ok = ok and all(0.9 < q < 1.1 for q in r["sr_err_ratio_fused_over_unfused (dx, dw1, dw2)"])

        # time, with main_grad (gradient-accumulation fusion on both), weight cache on
        emu.new_step()
        xs = x.clone().requires_grad_()
        pa, pb = params([w1, w2], True)

        def mk(fn):
            def call():
                xs.grad = None
                for w in (pa, pb):
                    w.grad, w.grad_added_to_main_grad = None, False
                fn(xs, pa, pb).backward(dout)
                for w in (pa, pb):               # what DDP's hook does if the grad was not accumulated
                    if not w.grad_added_to_main_grad:
                        w.main_grad.add_(w.grad)
            return call
        variants = {"unfused": mk(unfused), "fused": mk(fused)}
        for fn in variants.values():
            for _ in range(10):
                fn()
        torch.cuda.synchronize()
        rounds = {k: [] for k in variants}
        for _ in range(5):
            for k, fn in variants.items():
                ts = []
                for _ in range(50):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))  # noqa
                rounds[k].append(statistics.median(ts))
        emu.end_step()
        r["fwd_bwd_ms"] = {k: round(statistics.median(v), 4) for k, v in rounds.items()}
        r["per_round_speedup"] = [round(u / q, 3) for u, q in zip(rounds["unfused"], rounds["fused"])]
        res[name] = r
        print(name, json.dumps(r), flush=True)
    print("PASS (fused = unfused bitwise; SR statistics equal)" if ok else "FAIL")
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_mxfp4_fused_dense.json", "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
