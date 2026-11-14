"""--mxfp4-backend alto (ALTO's MXFP4 kernels, gfx950) against the emulation (--mxfp4-backend emulate)
through the Megatron entry points (mxfp4_emulation.grouped_linear / linear), on the shapes of the
kda-1.5b latent-MoE rung: routed fc1 384 -> 2*448, routed fc2 448 -> 384, dense / shared fc1
768 -> 2*448 and 768 -> 2*1920, fc2 back to 768.

1. Stochastic rounding off in both (mxfp4_emulation._FORCE_RNE): y, dx, dW must be bit-identical.
2. Stochastic rounding on: dx and dW differ by the random draws only, so their relative error
   against the bf16 product must match the emulation's (both unbiased, same variance).

Run: ALTO_DIR=... python tests/unit_tests/test_mxfp4_alto.py   (one gfx950 GPU)."""
import json
import sys

import torch

from megatron.core import mxfp4_emulation as emu

dev = torch.device("cuda", 0)
torch.manual_seed(0)


def run(backend, fn, inputs, dy):
    emu.set_backend(backend, "alto", "alto")
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    y = fn(*leaves)
    y.backward(dy)
    return [y.detach()] + [t.grad.detach() for t in leaves]


def grouped_fn(m_splits):
    return lambda x, *ws: emu.grouped_linear(x, m_splits, list(ws), "alto", "alto")


def dense_fn(x, w):
    return emu.linear(x, w, "alto", "alto")


def rel(a, b):
    """Relative error; 0 when both are exactly zero (an expert with no tokens has dW = 0)."""
    d, n = (a.float() - b.float()).norm(), b.float().norm()
    return 0.0 if n == 0 and d == 0 else (d / n).item()


def case(name, fn, inputs, dy, ref):
    out = {}
    emu._FORCE_RNE = True
    emu.new_step()
    a = run("emulate", fn, inputs, dy)
    b = run("alto", fn, inputs, dy)
    labels = ["y", "dx"] + [f"dw{i}" for i in range(len(inputs) - 1)]
    out["rne_mismatches"] = {k: int((p != q).sum()) for k, p, q in zip(labels, a, b)}
    out["rne_bit_exact"] = all(v == 0 for v in out["rne_mismatches"].values())
    emu.end_step()
    # stochastic rounding: relative error vs the bf16 reference, both backends, 4 draws each
    emu._FORCE_RNE = False
    errs = {"emulate": [], "alto": []}
    for rep in range(4):
        emu.new_step()                      # same H for both backends within a draw
        for be in ("emulate", "alto"):
            r = run(be, fn, inputs, dy)
            errs[be].append([rel(p, q) for p, q in zip(r[1:], ref[1:])])
        emu.end_step()
    for be in errs:
        t = torch.tensor(errs[be])
        out[f"sr_rel_err_{be}"] = dict(zip(labels[1:], [round(v, 5) for v in t.mean(0).tolist()]))
    out["sr_err_ratio_alto_over_emulate"] = {
        k: round(out["sr_rel_err_alto"][k] / out["sr_rel_err_emulate"][k], 4) if out["sr_rel_err_emulate"][k]
        else 1.0 for k in labels[1:]}
    print(name, json.dumps(out), flush=True)
    return out


def reference(fn_bf16, inputs, dy):
    leaves = [t.detach().clone().requires_grad_() for t in inputs]
    y = fn_bf16(*leaves)
    y.backward(dy)
    return [y.detach()] + [t.grad.detach() for t in leaves]


res = {}
# routed experts: 16 local experts with uneven counts (0, small, not multiples of 128 / 32)
for name, (n, k) in {"routed_fc1": (896, 384), "routed_fc2": (384, 448)}.items():
    m_splits = [0, 5, 37, 128, 300, 1000, 2048, 129, 64, 31, 4000, 700, 1, 255, 513, 3000]
    T = sum(m_splits)
    x = torch.randn(T, k, device=dev).bfloat16()
    ws = [(torch.randn(n, k, device=dev) * 0.05).bfloat16() for _ in m_splits]
    dy = (torch.randn(T, n, device=dev) * 1e-3).bfloat16()

    def bf16(x, *w, _m=m_splits):
        return torch.cat([xi @ wi.t() for xi, wi in zip(torch.split(x, _m), w)])
    ref = reference(bf16, [x, *ws], dy)
    res[name] = case(name, grouped_fn(m_splits), [x, *ws], dy, ref)

# dense linears (shared expert, dense MLP of layer 0); 16384 tokens = MBS 4 x 4096
for name, (n, k) in {"shared_fc1": (896, 768), "shared_fc2": (768, 448),
                     "dense_fc1": (3840, 768), "dense_fc2": (768, 1920)}.items():
    x = torch.randn(16384, k, device=dev).bfloat16()
    w = (torch.randn(n, k, device=dev) * 0.05).bfloat16()
    dy = (torch.randn(16384, n, device=dev) * 1e-3).bfloat16()
    ref = reference(lambda a, b: a @ b.t(), [x, w], dy)
    res[name] = case(name, dense_fn, [x, w], dy, ref)

ok = all(r["rne_bit_exact"] for r in res.values()) and all(
    0.9 < v < 1.1 for r in res.values() for v in r["sr_err_ratio_alto_over_emulate"].values())
print("PASS" if ok else "FAIL")
json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_mxfp4_alto.json", "w"), indent=1)
sys.exit(0 if ok else 1)
