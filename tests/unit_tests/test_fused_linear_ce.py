"""Fused output layer + cross entropy (megatron/core/fusions/fused_linear_cross_entropy.py) against Megatron's path:
logits = h · Wᵀ (bf16), fused_vocab_parallel_cross_entropy (--cross-entropy-loss-fusion, native), autograd.

N = 16384 tokens (one micro-batch: 8192 × 2), V = 200064 (the run's tokenizer), H = 768 (1.5B) and 1280 (5B);
10% of the tokens masked; the loss gradient as Megatron's loss_func makes it (mask · c). Two weight scales:
small logits (init-like) and spread logits (std ≈ 4, like a trained model).
Reported: loss (max abs diff), dh and dW (relative difference), the main_grad path, the mask · c check, time
of forward + backward (interleaved, 5 rounds × 10 calls) and peak memory.
Run: python test_fused_linear_ce.py out.json   (one GPU)."""
import json
import os
import statistics
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("FUSED_LINEAR_CE_CHECK", "1")
from megatron.core.fusions import fused_linear_cross_entropy as F  # noqa: E402
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy  # noqa: E402

dev = torch.device("cuda", 0)


def reference(h, w, tgt, tp):
    logits = torch.matmul(h, w.t())                                   # [N, V] bf16
    return fused_vocab_parallel_cross_entropy(logits.unsqueeze(1), tgt.unsqueeze(1), tp).view(-1)


def fused(h, w, tgt, mask):
    return F._FusedLinearCrossEntropy.apply(h, w, tgt, mask)


def grads(fn, h, w, mask, main_grad=False):
    hh = h.detach().clone().requires_grad_()
    ww = w.detach().clone().requires_grad_()
    if main_grad:
        ww.main_grad = torch.zeros(ww.shape, dtype=torch.float32, device=dev)
        ww.grad_added_to_main_grad = False
    loss = fn(hh, ww)
    c = 1.0 / mask.sum()
    (loss * mask * c).sum().backward()
    dw = ww.main_grad if (main_grad and ww.grad_added_to_main_grad) else ww.grad.float()
    return loss.detach(), hh.grad.float(), dw


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29533", rank=0, world_size=1)
    tp = dist.new_group([0])
    g = torch.Generator(device=dev).manual_seed(0)
    N, V = 16384, 200064
    res, ok = {}, True
    for H in (768, 1280):
        for wscale_name, wscale in (("init-like logits", 0.02), ("spread logits", 4.0 / H ** 0.5)):
            h = torch.randn(N, H, device=dev, generator=g).bfloat16()
            w = (torch.randn(V, H, device=dev, generator=g) * wscale).bfloat16()
            tgt = torch.randint(0, V, (N,), device=dev, generator=g)
            mask = (torch.rand(N, device=dev, generator=g) > 0.1).float()
            lr, dhr, dwr = grads(lambda a, b: reference(a, b, tgt, tp), h, w, mask)
            lf, dhf, dwf = grads(lambda a, b: fused(a, b, tgt, mask), h, w, mask)
            _, _, dwm = grads(lambda a, b: fused(a, b, tgt, mask), h, w, mask, main_grad=True)
            r = {"loss max abs diff": (lf - lr).abs().max().item(), "loss mean (ref)": lr.mean().item(),
                 "dh rel diff": rel(dhf, dhr), "dW rel diff": rel(dwf, dwr),
                 "dW via main_grad rel diff": rel(dwm, dwr)}
            ok = ok and r["loss max abs diff"] < 1e-3 and r["dh rel diff"] < 1e-2 and r["dW rel diff"] < 1e-2 \
                and r["dW via main_grad rel diff"] < 1e-2
            res[f"H={H}, {wscale_name}"] = r
            print(f"H={H}, {wscale_name}", json.dumps(r), flush=True)
        # time + memory (spread logits case), forward + backward
        F._CHECK = False
        hh = h.detach().clone().requires_grad_()
        ww = w.detach().clone().requires_grad_()
        c = 1.0 / mask.sum()
        variants = {"megatron (logits + fused CE)": lambda: (reference(hh, ww, tgt, tp) * mask * c).sum().backward(),
                    "fused linear CE": lambda: (fused(hh, ww, tgt, mask) * mask * c).sum().backward()}
        mem = {}
        for k, fn in variants.items():
            hh.grad = ww.grad = None
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()        # noqa: E702
            base = torch.cuda.memory_allocated()
            fn(); torch.cuda.synchronize()                                        # noqa: E702
            mem[k] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2)
            for _ in range(3):
                hh.grad = ww.grad = None
                fn()
        rounds = {k: [] for k in variants}
        for _ in range(5):
            for k, fn in variants.items():
                ts = []
                for _ in range(10):
                    hh.grad = ww.grad = None
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))  # noqa
                rounds[k].append(statistics.median(ts))
        F._CHECK = True
        ks = list(variants)
        t = {"fwd_bwd_ms": {k: round(statistics.median(v), 2) for k, v in rounds.items()},
             "per_round_speedup": [round(a / b, 3) for a, b in zip(rounds[ks[0]], rounds[ks[1]])],
             "peak_extra_memory_GiB": mem}
        res[f"H={H} time"] = t
        print(f"H={H} time", json.dumps(t), flush=True)
    print("PASS" if ok else "FAIL")
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_fused_linear_ce.json", "w"), indent=1)
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
