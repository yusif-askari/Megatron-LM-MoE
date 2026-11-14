"""Timing only: routed-expert MLP fc2(weighted SwiGLU(fc1(x), p)), bf16 (TransformerEngine GroupedLinear,
torch._grouped_mm) vs MXFP4 (unfused, fused forward, fused forward + backward). The variants are interleaved:
ROUNDS rounds, each timing every variant for ITERS calls (after WARM warm-up calls), so a slow period of the
shared node affects all variants alike. Reports per variant all round medians, their median and min-max.
Run: ALTO_DIR=... python bench_mxfp4_fused_swiglu.py out.json   (one gfx950 GPU)."""
import itertools, json, os, statistics, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mxfp4_fused_swiglu as T  # noqa: E402  (reuses its functions)
from megatron.core import mxfp4_emulation as emu  # noqa: E402
import transformer_engine.pytorch as te  # noqa: E402

ROUNDS, ITERS, WARM = 5, 50, 10
dev = T.dev


def time_calls(fn, n):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


res = {"device": torch.cuda.get_device_name(0), "host": os.uname().nodename, "rounds": ROUNDS, "iters": ITERS}
for name, (d, f, rows) in {"1.5b": (384, 448, 131072), "5b": (640, 704, 98304)}.items():
    wts = torch.exp(0.8 * torch.randn(64, generator=torch.Generator().manual_seed(1)))
    m = torch.multinomial(wts / wts.sum(), rows, replacement=True,
                          generator=torch.Generator().manual_seed(2)).bincount(minlength=64).tolist()
    x, p, w1, w2, dout = T.operands(m, d, f, 64)
    xs = x.clone().requires_grad_(); ps = p.clone().requires_grad_()
    w1s = [w.clone().requires_grad_() for w in w1]; w2s = [w.clone().requires_grad_() for w in w2]
    offs = torch.tensor(list(itertools.accumulate(m)), device=dev, dtype=torch.int32)
    w1st = torch.stack(w1).requires_grad_(); w2st = torch.stack(w2).requires_grad_()
    gl1 = te.GroupedLinear(64, d, 2 * f, bias=False, params_dtype=torch.bfloat16, device="cuda")
    gl2 = te.GroupedLinear(64, f, d, bias=False, params_dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        for e in range(64):
            getattr(gl1, f"weight{e}").copy_(w1[e]); getattr(gl2, f"weight{e}").copy_(w2[e])
    te_params = list(gl1.parameters()) + list(gl2.parameters())
    emu.set_backend("alto", "alto", "alto")
    emu.new_step()                               # weight cache on, as within a training step

    def mk(call_g, call_ng, params):
        def fwd():
            with torch.no_grad():
                call_ng()

        def fwd_bwd():
            for t_ in [xs, ps, *params]:
                t_.grad = None
            call_g().backward(dout)
        return fwd, fwd_bwd
    variants = {
        "bf16_te_grouped_linear": mk(lambda: T.bf16_te(xs, m, ps, gl1, gl2), lambda: T.bf16_te(x, m, p, gl1, gl2), te_params),
        "bf16_torch_grouped_mm": mk(lambda: T.bf16_torch(xs, m, ps, w1st, w2st, offs),
                                    lambda: T.bf16_torch(x, m, p, w1st, w2st, offs), [w1st, w2st]),
        "mxfp4_unfused": mk(lambda: T.unfused(xs, m, ps, w1s, w2s), lambda: T.unfused(x, m, p, w1, w2), w1s + w2s),
        "mxfp4_fused_padded": mk(lambda: T.fused_padded(xs, m, ps, w1s, w2s),
                                 lambda: T.fused_padded(x, m, p, w1, w2), w1s + w2s),
        "mxfp4_fused_fwd_only": mk(lambda: T.fused_fwd_only(xs, m, ps, w1s, w2s),
                                   lambda: T.fused_fwd_only(x, m, p, w1, w2), w1s + w2s),
        "mxfp4_fused": mk(lambda: T.fused(xs, m, ps, w1s, w2s), lambda: T.fused(x, m, p, w1, w2), w1s + w2s),
    }
    for fwd, fb in variants.values():          # warm-up (compiles, autotunes)
        for _ in range(WARM):
            fwd(); fb()
    torch.cuda.synchronize()
    rounds = {k: {"fwd": [], "fwd_bwd": []} for k in variants}
    for _ in range(ROUNDS):
        for k, (fwd, fb) in variants.items():
            rounds[k]["fwd"].append(time_calls(fwd, ITERS))
            rounds[k]["fwd_bwd"].append(time_calls(fb, ITERS))
    emu.end_step()
    r = {}
    for k, v in rounds.items():
        r[k] = {ph: {"median_ms": statistics.median(v[ph]), "min_ms": min(v[ph]), "max_ms": max(v[ph]),
                     "rounds_ms": [round(t, 4) for t in v[ph]]} for ph in v}
    base = r["bf16_te_grouped_linear"]
    for k in r:
        r[k]["speedup_vs_bf16_te"] = {ph: base[ph]["median_ms"] / r[k][ph]["median_ms"] for ph in ("fwd", "fwd_bwd")}
    res[name] = r
    print(name, json.dumps({k: {ph: round(r[k][ph]["median_ms"], 3) for ph in ("fwd", "fwd_bwd")} for k in r}), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
