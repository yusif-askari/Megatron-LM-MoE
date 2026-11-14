"""GPU kernel time and launches per call, fused vs unfused dense MLP (shared expert / dense MLP shapes), and the
host time per call (CPU wall of the Python call, no sync).  Run: ALTO_DIR=... python profile_mxfp4_dense.py"""
import collections, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mxfp4_fused_dense as D  # noqa: E402
from megatron.core import mxfp4_alto  # noqa: E402
from megatron.core import mxfp4_emulation as emu  # noqa: E402

dev = D.dev
emu.set_backend("alto", "alto", "alto"); mxfp4_alto._alto()
for name, (d, f) in {"1.5b shared expert": (768, 448), "1.5b dense MLP": (768, 1920)}.items():
    T = 16384
    x = torch.randn(T, d, device=dev).bfloat16().requires_grad_()
    w1, w2 = D.params([(torch.randn(2 * f, d, device=dev) * d ** -0.5).bfloat16(),
                       (torch.randn(d, f, device=dev) * f ** -0.5).bfloat16()], True)
    dout = (torch.randn(T, d, device=dev) * 1e-3).bfloat16()
    emu.new_step()
    for label, fn in (("unfused", D.unfused), ("fused", D.fused)):
        def call():
            x.grad = None
            for w in (w1, w2):
                w.grad, w.grad_added_to_main_grad = None, False
            fn(x, w1, w2).backward(dout)
            for w in (w1, w2):
                if not w.grad_added_to_main_grad:
                    w.main_grad.add_(w.grad)
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        n = 30
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                call()
            torch.cuda.synchronize()
        per = collections.defaultdict(lambda: [0.0, 0])
        for e in prof.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                per[e.name][0] += e.device_time / 1000 / n; per[e.name][1] += 1 / n
        # host time: queue many calls without syncing (GPU runs behind), measure CPU wall per call
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n):
            call()
        t1 = time.perf_counter(); torch.cuda.synchronize(); t2 = time.perf_counter()
        print(f"{name} | {label}: GPU kernels {sum(v[0] for v in per.values()):.3f} ms in "
              f"{sum(v[1] for v in per.values()):.0f} launches; host {1000 * (t1 - t0) / n:.3f} ms/call; "
              f"end-to-end {1000 * (t2 - t0) / n:.3f} ms/call", flush=True)
        for k, v in sorted(per.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f"      {v[0]:.4f} ms x{v[1]:.0f}  {k[:90]}")
    emu.end_step()
