"""Fused MXFP4 routed MLP on tokens that are not sorted by expert (mxfp4_alto.chunk_perm / fused_swiglu_mlp(perm=),
MXFP4_SORT_FOLD): bit-identical to sorting the tokens with sort_chunks_by_idxs first and un-sorting the output.
Received layout as after the expert-parallel all-to-all: chunks ordered by (source rank, local expert); the experts
need them ordered by (local expert, source rank). SwiGLU and pglu, 5B shape (latent 640, expert 704), 8 source ranks x
32 local experts with uneven chunk sizes (some empty). Same seeds and the same Hadamard rotation for both runs.
Run: ALTO_DIR=... python tests/unit_tests/test_mxfp4_sort_fold.py out.json   (one gfx950 GPU)."""
import json
import sys

import torch

from megatron.core import exp_gates, mxfp4_alto
from megatron.core import mxfp4_emulation as emu
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs


def main():
    dev = torch.device("cuda", 0)
    R, E, d, f = 8, 32, 640, 704
    g = torch.Generator().manual_seed(0)
    sizes = torch.randint(0, 1100, (R, E), generator=g)
    sizes[torch.rand(R, E, generator=g) < 0.08] = 0                      # empty chunks
    sizes[:, 5] = 0                                                      # an expert without tokens
    split = sizes.ravel().to(dev)                                        # received order: (rank, expert)
    idx_sort = torch.arange(R * E, device=dev).reshape(-1, E).T.ravel()  # -> (expert, rank)
    idx_restore = torch.arange(R * E, device=dev).reshape(E, -1).T.ravel()
    m = sizes.sum(0).tolist()                                            # tokens per local expert
    T = int(sizes.sum())
    gd = torch.Generator(device=dev).manual_seed(1)
    x = torch.randn(T, d, device=dev, generator=gd).bfloat16()
    p = torch.rand(T, 1, device=dev, generator=gd) * 0.3
    w1 = [(torch.randn(2 * f, d, device=dev, generator=gd) * d ** -0.5 * 2).bfloat16() for _ in range(E)]
    w2 = [(torch.randn(d, f, device=dev, generator=gd) * f ** -0.5).bfloat16() for _ in range(E)]
    dout = (torch.randn(T, d, device=dev, generator=gd) * 1e-3).bfloat16()
    emu.set_backend("alto", "alto", "alto")

    def run(fold, gate):
        # new_step() draws the step's random Hadamard rotation (token-direction quantization of the weight-gradient
        # operands) from emu._H_GEN, which torch.manual_seed does not reset: same rotation for both runs
        emu._H_GEN.manual_seed(20261006)
        emu.new_step()
        xs = x.clone().requires_grad_()
        ps = p.clone().requires_grad_()
        a1 = [w.clone().requires_grad_() for w in w1]
        a2 = [w.clone().requires_grad_() for w in w2]
        gm = None
        if gate:
            gm = exp_gates.ExpPGLU(num_local_experts=E).to(dev)
            with torch.no_grad():
                gm.raw_p.copy_(torch.log(torch.expm1(torch.tensor([1.0, 0.7, 1.4, 2.0], device=dev)[torch.arange(E, device=dev) % 4])))
                gm.b.copy_(torch.tensor([1.0, 0.5, 2.0, -0.5], device=dev)[torch.arange(E, device=dev) % 4])
        torch.manual_seed(1234)
        if fold:
            # fold >= 2: the row maps from one kernel (mxfp4_alto.CHUNK_KERNEL, triton_fast.chunk_layout)
            mxfp4_alto.CHUNK_KERNEL = fold >= 2
            perm = mxfp4_alto.chunk_order(split, idx_sort, T, dev, E)
            assert isinstance(perm, tuple) == (fold >= 2)
            out = mxfp4_alto.fused_swiglu_mlp(xs, m, ps, a1, a2, gate=gm, perm=perm)
        else:
            xsrt, psrt = sort_chunks_by_idxs(xs, split, idx_sort, probs=ps, fused=False)
            o = mxfp4_alto.fused_swiglu_mlp(xsrt, m, psrt, a1, a2, gate=gm)
            out, _ = sort_chunks_by_idxs(o, sizes.T.ravel().to(dev), idx_restore, fused=False)
        out.backward(dout)
        emu.end_step()
        r = {"out": out.detach(), "dx": xs.grad, "dprobs": ps.grad, "dw1": torch.stack([w.grad for w in a1]),
             "dw2": torch.stack([w.grad for w in a2])}
        if gate:
            r["d_raw_p"], r["d_b"] = gm.raw_p.grad, gm.b.grad
        return r

    res, ok = {"tokens": T, "chunks": R * E, "empty_chunks": int((sizes == 0).sum())}, True
    # the map itself against the reference sort
    perm = mxfp4_alto.chunk_perm(split, idx_sort, T, dev)
    ref, _ = sort_chunks_by_idxs(torch.arange(T, device=dev).unsqueeze(1), split, idx_sort, fused=False)
    res["perm_equals_reference_sort"] = bool(torch.equal(perm.long(), ref.squeeze(1)))
    ok &= res["perm_equals_reference_sort"]
    for gate in (False, True):
        b = run(False, gate)
        for fold, tag in ((1, ""), (2, "_chunk_kernel")):
            a = run(fold, gate)
            cmp = {k: {"mismatch": int((a[k].float() != b[k].float()).sum()), "n": a[k].numel(),
                       "nonfinite": int((~torch.isfinite(a[k].float())).sum())} for k in a}
            # d(probs), d_raw_p, d_b are fp32 sums of the same terms in the same order: identical as well
            res[("pglu" if gate else "swiglu") + tag] = cmp
            ok &= all(v["mismatch"] == 0 and v["nonfinite"] == 0 for v in cmp.values())
            print(("pglu" if gate else "swiglu") + tag, cmp, flush=True)
    res["pass"] = bool(ok)
    print("PASS" if ok else "FAIL")
    if len(sys.argv) > 1:
        json.dump(res, open(sys.argv[1], "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
