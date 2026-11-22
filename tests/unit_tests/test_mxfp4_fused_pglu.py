"""pglu (learnable per-expert p, b; megatron/core/exp_gates.py) in the MLP and on the fused MXFP4 kernels.

A. Math, against an fp64 autograd truth (no quantization): h = gate(y_gate) * y_value * prob with
   per-expert p (including p = 1 exactly) and b, for
     new  = exp_gates.ExpPGLU.weighted_glu (fp32, one bf16 rounding, analytic backward), and
     old  = the Megatron-LM-MoE version (gate, gate * value, * prob each rounded to bf16; autograd).
   Reported: h and dy mismatches against the bf16-rounded truth, relative errors of d(prob), dp, db.
B. MXFP4 routed MLP (1.5B and 5B shapes, 16 experts with uneven counts) and dense MLP (one expert), stochastic
   rounding off: fused kernels (GATE=1) against the unfused path (ALTO GEMMs + exp_gates reference), and the
   fused forward with the reference backward on fp32 dh. Row maps against padded copies.
C. Time at the 5B shape, 64 local experts: SwiGLU fused, pglu fused, pglu unfused (forward, forward + backward).

Run: ALTO_DIR=... python tests/unit_tests/test_mxfp4_fused_pglu.py out.json  (one gfx950 GPU); part A alone on
any device: python tests/unit_tests/test_mxfp4_fused_pglu.py --math-only [cpu]."""
import json
import statistics
import sys

import torch
import torch.nn.functional as F

from megatron.core import exp_gates


# ---------------------------------------------------------------------------------------------- A. math
def _bump_integral64(u, p):
    """F_p(u) in fp64, written as G(u) for u >= 0 and -G(-u) for u < 0 with G(t) = l exprel((1-p) l),
    l = ln(1+t): autograd then gives the true slope dF/du = 1 at u = 0 (sign(u) * l(|u|) gives 0 there)."""
    def G(t):
        l = torch.log1p(t)
        z = (1.0 - p) * l
        small = z.abs() < 1e-6
        zs = torch.where(small, torch.ones_like(z), z)
        return l * torch.where(small, 1.0 + z * (0.5 + z / 6.0), torch.expm1(zs) / zs)
    # torch.where routes the gradient (clamp's gradient at its bound differs between torch versions)
    zero = torch.zeros_like(u)
    return torch.where(u >= 0, G(torch.where(u >= 0, u, zero)), -G(torch.where(u < 0, -u, zero)))


def truth(y, prob, raw_p, b, counts):
    """fp64 autograd of h = (F_p(g - b) - F_p(-b)) * v * prob."""
    y64 = y.double().detach().requires_grad_()
    pr64 = prob.double().detach().requires_grad_()
    rp64 = raw_p.double().detach().requires_grad_()
    b64 = b.double().detach().requires_grad_()
    c = torch.as_tensor(counts, device=y.device)
    p = torch.repeat_interleave(F.softplus(rp64), c).unsqueeze(-1)
    br = torch.repeat_interleave(b64, c).unsqueeze(-1)
    g, v = y64.chunk(2, -1)
    h = (_bump_integral64(g - br, p) - _bump_integral64(-br, p)) * v * pr64
    return h, (y64, pr64, rp64, b64)


def old_lm_moe(y, prob, raw_p, b, counts):
    """Megatron-LM-MoE: gate = _pglu(x_glu, p, b) (fp32 math, bf16 out), out = gate * x_linear (bf16),
    h = (out * probs).to(bf16); autograd backward; coefficients in bf16 (Float16Module)."""
    rp = raw_p.bfloat16().detach().requires_grad_()
    bb = b.bfloat16().detach().requires_grad_()
    yy = y.detach().requires_grad_()
    pr = prob.detach().requires_grad_()
    c = torch.as_tensor(counts, device=y.device)
    p = F.softplus(torch.repeat_interleave(rp.float(), c)).unsqueeze(-1)
    br = torch.repeat_interleave(bb.float(), c).unsqueeze(-1)
    g, v = yy.chunk(2, -1)

    def bi(u, p):          # Megatron-LM-MoE _bump_integral (exprel Taylor for |z| < 1e-3)
        l = torch.log1p(u.abs())
        z = (1.0 - p) * l
        small = z.abs() < 1e-3
        zs = torch.where(small, torch.ones_like(z), z)
        return torch.sign(u) * l * torch.where(small, 1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)),
                                               torch.expm1(zs) / zs)
    gate = (bi(g.float() - br, p) - bi(-br, p)).to(g.dtype)
    out = gate * v
    h = (out * pr).to(out.dtype)
    return h, (yy, pr, rp, bb)


def new_megatron(y, prob, raw_p, b, counts):
    m = exp_gates.ExpPGLU(num_local_experts=len(counts)).to(y.device)
    with torch.no_grad():
        m.raw_p.copy_(raw_p)
        m.b.copy_(b)
    yy = y.detach().requires_grad_()
    pr = prob.detach().requires_grad_()
    return m.weighted_glu(yy, pr, counts), (yy, pr, m.raw_p, m.b)


def math_case(dev, counts, f, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    n, E = sum(counts), len(counts)
    y = (torch.randn(n, 2 * f, device=dev, generator=gen) * 2.0).bfloat16()
    y[:, :f] = (y[:, :f].float() * torch.rand(n, 1, device=dev, generator=gen).mul(4)).bfloat16()  # wide |g|
    prob = torch.rand(n, 1, device=dev, generator=gen) * 0.3
    p_vals = torch.tensor([1.0, 0.3, 2.5, 1.0001, 0.999, 1.6, 0.05, 4.0], device=dev)
    b_vals = torch.tensor([1.0, 0.0, -0.7, 3.0, 1.3, 0.25, -2.0, 6.0], device=dev)
    p = p_vals[torch.arange(E) % 8]
    raw_p = torch.log(torch.expm1(p))
    b = b_vals[torch.arange(E) % 8]
    dh = (torch.randn(n, f, device=dev, generator=gen) * 1e-2).bfloat16()

    h_t, leaves_t = truth(y, prob, raw_p, b, counts)
    h_t.backward(dh.double())
    gt = [t.grad for t in leaves_t]
    br = torch.repeat_interleave(b, torch.as_tensor(counts, device=dev)).unsqueeze(-1)
    out = {"elements_with_g_equal_b (bf16 g, slope 1 there; autograd of sign(u) l gives 0)":
           int((y[:, :f].float() == br.bfloat16().float()).sum())}
    for label, fn in (("new", new_megatron), ("old_lm_moe", old_lm_moe)):
        h, leaves = fn(y, prob, raw_p, b, counts)
        h.backward(dh)
        g = [t.grad for t in leaves]
        h_ref = h_t.to(torch.bfloat16)
        dy_ref = gt[0].to(torch.bfloat16)
        rel = lambda a, r: ((a.double() - r).norm() / r.norm()).item()
        out[label] = {
            "h_mismatch_vs_rounded_truth": int((h != h_ref).sum()), "h_elements": h.numel(),
            "h_rel_err": rel(h, h_t.detach()),
            "dy_mismatch_vs_rounded_truth": int((g[0] != dy_ref).sum()),
            "dy_rel_err": rel(g[0], gt[0]),
            "dprob_rel_err": rel(g[1], gt[1]),
            "d_raw_p_rel_err": rel(g[2], gt[2]),
            "d_b_rel_err": rel(g[3], gt[3]),
            "d_raw_p_max_rel_err_per_expert": ((g[2].double() - gt[2]).abs() / gt[2].abs().clamp_min(1e-30)).max().item(),
            "d_b_max_rel_err_per_expert": ((g[3].double() - gt[3]).abs() / gt[3].abs().clamp_min(1e-30)).max().item(),
        }
    return out


def part_a(dev):
    res = {"16 experts, F=704": math_case(dev, [0, 5, 37, 128, 300, 100, 248, 129, 64, 31, 400, 70, 1, 255, 113, 300],
                                          704),
           "1 expert (dense), F=704": math_case(dev, [2048], 704, seed=1)}
    ok = all(r["new"]["d_raw_p_rel_err"] < 1e-5 and r["new"]["d_b_rel_err"] < 1e-5 and r["new"]["dprob_rel_err"] < 1e-5
             and r["new"]["h_mismatch_vs_rounded_truth"] <= r["old_lm_moe"]["h_mismatch_vs_rounded_truth"]
             and r["new"]["dy_mismatch_vs_rounded_truth"] <= r["old_lm_moe"]["dy_mismatch_vs_rounded_truth"]
             for r in res.values())
    return res, ok


# ------------------------------------------------------------------------------------------- B. MXFP4 MLP
def part_b_and_c():
    from megatron.core import mxfp4_alto
    from megatron.core import mxfp4_emulation as emu
    from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl

    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(0)
    keep = []

    def gate_module(E):
        m = exp_gates.ExpPGLU(num_local_experts=E).to(dev)
        pv = torch.tensor([1.0, 0.7, 1.4, 2.0], device=dev)[torch.arange(E, device=dev) % 4]
        bv = torch.tensor([1.0, 0.5, 2.0, -0.5], device=dev)[torch.arange(E, device=dev) % 4]
        with torch.no_grad():
            m.raw_p.copy_(torch.log(torch.expm1(pv)))
            m.b.copy_(bv)
        return m

    def unfused(x, m, p, w1, w2, gm):
        y = emu.grouped_linear(x, m, w1, "alto", "alto")
        h = gm.weighted_glu(y, p, m)
        return emu.grouped_linear(h, m, w2, "alto", "alto")

    def fused(x, m, p, w1, w2, gm):
        mxfp4_alto.FUSED_SWIGLU_BWD = True
        return mxfp4_alto.fused_swiglu_mlp(x, m, p, w1, w2, gate=gm)

    def fused_fwd_only(x, m, p, w1, w2, gm):
        # the flag is read in backward(): it stays off until the next fused() call turns it on
        mxfp4_alto.FUSED_SWIGLU_BWD = False
        return mxfp4_alto.fused_swiglu_mlp(x, m, p, w1, w2, gate=gm)

    def padded(fn):
        def g(*a):
            mxfp4_alto.ROW_MAP = False
            try:
                return fn(*a)
            finally:
                mxfp4_alto.ROW_MAP = True
        return g

    def run(fn, x, m, p, w1, w2, gm, dout):
        emu.set_backend("alto", "alto", "alto")
        xs = x.detach().clone().requires_grad_()
        ps = p.detach().clone().requires_grad_() if p is not None else None
        w1s = [w.detach().clone().requires_grad_() for w in w1]
        w2s = [w.detach().clone().requires_grad_() for w in w2]
        g2 = gate_module(gm.num_local_experts)
        with torch.no_grad():
            g2.raw_p.copy_(gm.raw_p)
            g2.b.copy_(gm.b)
        keep.extend(w1s + w2s)
        out = fn(xs, m, ps, w1s, w2s, g2)
        out.backward(dout)
        r = {"out": out.detach(), "dx": xs.grad, "dw1": torch.stack([w.grad for w in w1s]),
             "dw2": torch.stack([w.grad for w in w2s]), "d_raw_p": g2.raw_p.grad, "d_b": g2.b.grad}
        if ps is not None:
            r["dprobs"] = ps.grad
        return r

    def operands(m, d, f, n_exp, weighted=True):
        T = sum(m)
        x = torch.randn(T, d, device=dev, generator=gen).bfloat16()
        p = torch.rand(T, 1, device=dev, generator=gen) * 0.3 if weighted else None
        w1 = [(torch.randn(2 * f, d, device=dev, generator=gen) * d ** -0.5 * 2).bfloat16() for _ in range(n_exp)]
        w2 = [(torch.randn(d, f, device=dev, generator=gen) * f ** -0.5).bfloat16() for _ in range(n_exp)]
        dout = (torch.randn(T, d, device=dev, generator=gen) * 1e-3).bfloat16()
        return x, p, w1, w2, dout

    def cmp(a, b):
        out = {}
        for k in a:
            x, y = a[k].float(), b[k].float()
            out[k] = {"mismatch": int((x != y).sum()), "n": x.numel(),
                      "rel": ((x - y).norm() / y.norm().clamp_min(1e-30)).item()}
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
        r = {}
        m = [0, 5, 37, 128, 300, 1000, 2048, 129, 64, 31, 4000, 700, 1, 255, 513, 3000]
        x, p, w1, w2, dout = operands(m, d, f, len(m))
        gm = gate_module(len(m))
        emu._FORCE_RNE = True
        emu.new_step()
        a = run(fused, x, m, p, w1, w2, gm, dout)
        a1 = run(fused_fwd_only, x, m, p, w1, w2, gm, dout)
        ap = run(padded(fused), x, m, p, w1, w2, gm, dout)
        b = run(unfused, x, m, p, w1, w2, gm, dout)
        emu.end_step()
        r["routed: fused vs fused_fwd_only (same forward; fp32-dh backward: kernel vs exp_gates)"] = cmp(a, a1)
        r["routed: fused vs unfused (exp_gates on ALTO GEMMs; its dh is bf16)"] = cmp(a, b)
        r["routed: row maps vs padded copies (fused)"] = cmp(a, ap)
        # dense / shared expert: one expert, rows a multiple of 128 (no padding)
        md = [4096]
        x, _, w1, w2, dout = operands(md, d, f, 1, weighted=False)
        gd = gate_module(1)
        emu.new_step()
        a = run(lambda x_, m_, p_, w1_, w2_, g_: mxfp4_alto.fused_swiglu_dense(x_, w1_[0], w2_[0], gate=g_),
                x, md, None, w1, w2, gd, dout)
        b = run(lambda x_, m_, p_, w1_, w2_, g_: emu.linear(
                    g_.weighted_glu(emu.linear(x_, w1_[0], "alto", "alto"), None), w2_[0], "alto", "alto"),
                x, md, None, w1, w2, gd, dout)
        emu.end_step()
        r["dense: fused vs unfused"] = cmp(a, b)
        emu._FORCE_RNE = False

        if d == 640:
            # ---- C. time: 64 local experts, routed rows per GPU per micro-batch as in the 5B runs
            T = 98304
            wts = torch.exp(0.8 * torch.randn(64, generator=torch.Generator().manual_seed(1)))
            m = torch.multinomial(wts / wts.sum(), T, replacement=True,
                                  generator=torch.Generator().manual_seed(2)).bincount(minlength=64).tolist()
            x, p, w1, w2, dout = operands(m, d, f, 64)
            gm = gate_module(64)
            xs = x.clone().requires_grad_(); ps = p.clone().requires_grad_()
            w1s = [w.clone().requires_grad_() for w in w1]; w2s = [w.clone().requires_grad_() for w in w2]
            emu.set_backend("alto", "alto", "alto")
            emu.new_step()
            cases = {"swiglu_fused": lambda x_, p_, a_, b_: mxfp4_alto.fused_swiglu_mlp(x_, m, p_, a_, b_),
                     "pglu_fused": lambda x_, p_, a_, b_: fused(x_, m, p_, a_, b_, gm),
                     "pglu_unfused": lambda x_, p_, a_, b_: unfused(x_, m, p_, a_, b_, gm),
                     "swiglu_unfused": lambda x_, p_, a_, b_: emu.grouped_linear(weighted_bias_swiglu_impl(
                         emu.grouped_linear(x_, m, a_, "alto", "alto"), None, p_), m, b_, "alto", "alto")}
            for label, fn in cases.items():
                def fwd():
                    with torch.no_grad():
                        fn(x, p, w1, w2)

                def fwd_bwd():
                    for t in [xs, ps, *w1s, *w2s, gm.raw_p, gm.b]:
                        t.grad = None
                    fn(xs, ps, w1s, w2s).backward(dout)
                r[f"time_{label}_fwd_ms"] = timed(fwd)
                r[f"time_{label}_fwd_bwd_ms"] = timed(fwd_bwd)
            # the two fused kernels alone, SwiGLU (GATE=0) vs pglu (GATE=1)
            from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm import fused_swiglu as FS
            A = mxfp4_alto._alto()
            rows_, idx, tp = mxfp4_alto._layout(m, dev)
            src = mxfp4_alto._src_rows(rows_, tp)
            hh = mxfp4_alto._h_bf16(dev)
            xq, xsc, _, _ = A.dual_quantize(x, hh, use_sr=False, src_rows=src)
            w1q, w1sc = mxfp4_alto._stack_tile(w1, "fc1")
            w2q, w2sc = mxfp4_alto._stack_tile(w2, "fc2")
            dq, dsc, _, _ = A.dual_quantize(dout, hh, use_sr=True, src_rows=src)
            pp = p.reshape(-1).contiguous()
            import itertools, math
            prm = (gm.raw_p.detach(), gm.b.detach())
            offs = torch.tensor([0, *itertools.accumulate(math.ceil(c / 128) * 128 for c in m)], dtype=torch.int32,
                                device=dev)
            y1 = FS.fc1_swiglu_quant(xq, xsc, w1q, w1sc, idx, pp, hh, src_rows=src, pglu=prm)[0]
            for label, gc in (("swiglu", None), ("pglu", prm)):
                r[f"kernel_fc1_{label}_ms"] = timed(
                    lambda: FS.fc1_swiglu_quant(xq, xsc, w1q, w1sc, idx, pp, hh, src_rows=src, pglu=gc), 10, 50)
                r[f"kernel_bwd_{label}_ms"] = timed(
                    lambda: FS.fc2_dgrad_swiglu_bwd(dq, dsc, w2q, w2sc, idx, y1, pp, hh, use_sr=True, src_rows=src,
                                                    pglu=gc, pglu_row_offsets=offs if gc else None), 10, 50)
            r["kernel_configs"] = {"fc1": str(FS._fc1_swiglu_quant_kernel.best_config),
                                   "bwd": str(FS._fc2_dgrad_swiglu_bwd_kernel.best_config)}
            try:          # registers / spills of every compiled variant (GATE, tile config)
                regs = []
                for kname, k in (("fc1", FS._fc1_swiglu_quant_kernel), ("bwd", FS._fc2_dgrad_swiglu_bwd_kernel)):
                    for cache in k.fn.device_caches.values():
                        for ck in cache[0].values():
                            md = ck.metadata
                            regs.append(f"{kname} GATE={md.constants.get('GATE', '?') if hasattr(md, 'constants') else '?'}"
                                        f" warps={md.num_warps} regs={ck.n_regs} spills={ck.n_spills}")
                r["kernel_registers"] = sorted(set(regs))
            except Exception as e:  # noqa: BLE001
                r["kernel_registers"] = f"unavailable: {e!r}"
            emu.end_step()
            r["routed_rows"] = T
        res[name] = r
        print(name, json.dumps(r), flush=True)
    return res


def main():
    if "--math-only" in sys.argv:
        dev = torch.device("cpu") if "cpu" in sys.argv else torch.device("cuda", 0)
        res, ok = part_a(dev)
        print(json.dumps(res, indent=1))
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    from megatron.core.jit import enable_jit_fuser
    enable_jit_fuser()                     # the reference runs torch.compile'd, as in training
    res = {}
    res["A_math_vs_fp64_truth"], ok_a = part_a(torch.device("cuda", 0))
    print("A", json.dumps(res["A_math_vs_fp64_truth"]), flush=True)
    res["B_C_mxfp4"] = part_b_and_c()
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "test_mxfp4_fused_pglu.json", "w"), indent=1)
    print("A (math vs fp64 truth):", "PASS" if ok_a else "FAIL")
    print("B: see the mismatch / rel numbers (fused vs reference)")
    sys.exit(0 if ok_a else 1)


if __name__ == "__main__":
    main()
