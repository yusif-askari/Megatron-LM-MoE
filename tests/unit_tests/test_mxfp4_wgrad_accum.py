"""Gradient-accumulation fusion of the MXFP4 backend (mxfp4_alto.WGRAD_ACCUM): the wgrad GEMMs add their fp32
result into the weights' main_grad (as TE's linears under Megatron's --gradient-accumulation-fusion) instead
of returning a bf16 dW that is then added per expert.

Weights carry Megatron's DDP attributes (main_grad fp32, grad_added_to_main_grad). Round to nearest
(mxfp4_emulation._FORCE_RNE) so dW is deterministic. For the fused routed MLP, the unfused grouped linear and
the dense linear (shared expert / dense MLP):
  1. from main_grad = 0, the new main_grad rounded to bf16 equals the old path's bf16 dW bitwise
     (same fp32 accumulator; old: rounded to bf16 first);
  2. from a random main_grad M0: new main_grad == M0 + (fp32 dW from step 1) bitwise (one fp32 add);
  3. forward output and input gradients identical to the old path; grad_added_to_main_grad set.
Then: time of forward + backward of the routed MLP, old vs new (interleaved, 5 rounds x 50 calls).
Run: ALTO_DIR=... python test_mxfp4_wgrad_accum.py out.json   (one gfx950 GPU)."""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mxfp4_fused_swiglu as T  # noqa: E402
from megatron.core import mxfp4_alto  # noqa: E402
from megatron.core import mxfp4_emulation as emu  # noqa: E402

dev = T.dev


def with_main_grad(ws, init):
    out = []
    for w in ws:
        p = w.detach().clone().requires_grad_()
        p.main_grad = init(p)
        p.grad_added_to_main_grad = False
        out.append(p)
    return out


def run(kind, accum, x, m, p, w1, w2, dout, init):
    mxfp4_alto.WGRAD_ACCUM = accum
    mxfp4_alto._alto()
    from alto.kernels.fp4.mxfp4 import mxfp_linear
    mxfp_linear.WGRAD_INTO_MAIN_GRAD = accum
    xs = x.detach().clone().requires_grad_()
    ps = p.detach().clone().requires_grad_()
    a, b = with_main_grad(w1, init), with_main_grad(w2, init)
    T._KEEP.extend(a + b)        # the per-step weight caches are keyed by id(): keep the copies alive
    if kind == "fused":
        out = mxfp4_alto.fused_swiglu_mlp(xs, m, ps, a, b)
    elif kind == "unfused":
        out = T.unfused(xs, m, ps, a, b)
    else:                                                   # dense: x [T, K] -> fc1 [2F, K] -> fc2 [K, 2F]
        out = mxfp4_alto.linear(mxfp4_alto.linear(xs, a[0]), b[0])
    out.backward(dout)
    ws = a + b
    # without DDP nothing adds .grad into main_grad: the old path's dW is in .grad (or already in main_grad)
    mg = []
    for w in ws:
        g = w.main_grad.clone()
        if not w.grad_added_to_main_grad:
            g += w.grad.float()
        mg.append(g)
    return {"out": out.detach(), "dx": xs.grad, "mg": torch.cat([g.flatten() for g in mg]),
            "flags": all(w.grad_added_to_main_grad for w in ws) if accum else True}


def main():
    emu.set_backend("alto", "alto", "alto")
    res = {}
    ok = True
    for name, (d, f) in {"1.5b": (384, 448), "5b": (640, 704)}.items():
        m = [0, 5, 37, 128, 300, 1000, 2048, 129, 64, 31, 4000, 700, 1, 255, 513, 3000]
        x, p, w1, w2, dout = T.operands(m, d, f, len(m))
        emu._FORCE_RNE = True
        emu.new_step()
        r = {}
        for kind in ("fused", "unfused", "dense"):
            ww1, ww2 = (w1, w2) if kind != "dense" else (w1[:1], [torch.randn(d, 2 * f, device=dev).bfloat16()
                                                                    * (2 * f) ** -0.5])
            zero = lambda w: torch.zeros(w.shape, dtype=torch.float32, device=dev)
            g0 = torch.Generator(device=dev).manual_seed(5)
            rnd = lambda w: torch.randn(w.shape, dtype=torch.float32, device=dev, generator=g0) * 1e-2
            old0 = run(kind, False, x, m, p, ww1, ww2, dout, zero)
            new0 = run(kind, True, x, m, p, ww1, ww2, dout, zero)
            g0.manual_seed(5); oldr = run(kind, False, x, m, p, ww1, ww2, dout, rnd)   # noqa: E702
            g0.manual_seed(5); m0 = torch.cat([rnd(w).flatten() for w in ww1 + ww2])          # noqa: E702
            g0.manual_seed(5); newr = run(kind, True, x, m, p, ww1, ww2, dout, rnd)   # noqa: E702
            c = {
                "bf16(new from 0) != old dW (elements)": int((new0["mg"].bfloat16() != old0["mg"].bfloat16()).sum()),
                "new from M0 != M0 + fp32 dW (elements)": int((newr["mg"] != m0 + new0["mg"]).sum()),
                "out mismatches": int((new0["out"] != old0["out"]).sum()),
                "dx mismatches": int((new0["dx"] != old0["dx"]).sum()),
                "grad_added_to_main_grad set": new0["flags"],
                "old vs new main_grad from M0, max rel": ((oldr["mg"] - newr["mg"]).norm() /
                                                          newr["mg"].norm()).item(),
            }
            ok = ok and c["grad_added_to_main_grad set"] and all(
                v == 0 for k, v in c.items() if "mismatch" in k or "!=" in k)
            r[kind] = c
        emu.end_step()

        # ---- time: routed MLP forward + backward, 64 experts, realistic rows; old vs new interleaved
        T_rows = 131072 if d == 384 else 98304
        wts = torch.exp(0.8 * torch.randn(64, generator=torch.Generator().manual_seed(1)))
        mm = torch.multinomial(wts / wts.sum(), T_rows, replacement=True,
                               generator=torch.Generator().manual_seed(2)).bincount(minlength=64).tolist()
        x, p, w1, w2, dout = T.operands(mm, d, f, 64)
        emu._FORCE_RNE = False
        emu.new_step()
        zero = lambda w: torch.zeros(w.shape, dtype=torch.float32, device=dev)
        xs = x.clone().requires_grad_(); ps = p.clone().requires_grad_()    # noqa: E702
        a, b = with_main_grad(w1, zero), with_main_grad(w2, zero)

        def step(accum):
            def fn():
                mxfp4_alto.WGRAD_ACCUM = accum
                for t_ in [xs, ps, *a, *b]:
                    t_.grad = None
                for w in a + b:
                    w.grad_added_to_main_grad = False
                mxfp4_alto.fused_swiglu_mlp(xs, mm, ps, a, b).backward(dout)
                if not accum:                     # what DDP's hook does with the returned grads
                    for w in a + b:
                        if not w.grad_added_to_main_grad:
                            w.main_grad.add_(w.grad)
            return fn
        variants = {"old (bf16 dW, per-expert add)": step(False), "new (wgrad into main_grad)": step(True)}
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
        r["fwd_bwd_ms"] = {k: {"median": statistics.median(v), "rounds": [round(t, 4) for t in v]}
                           for k, v in rounds.items()}
        ks = list(variants)
        r["per_round_speedup"] = [round(o / n, 3) for o, n in zip(rounds[ks[0]], rounds[ks[1]])]
        res[name] = r
        print(name, json.dumps(r), flush=True)
    print("PASS" if ok else "FAIL")
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_mxfp4_wgrad_accum.json", "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
