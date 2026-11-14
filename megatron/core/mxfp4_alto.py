"""Hardware MXFP4 backend of --mxfp4-emulate (--mxfp4-backend alto): the same GEMMs, recipe and
per-step weight cache as megatron/core/mxfp4_emulation.py, computed with AMD ALTO's MXFP4 kernels
on gfx950 (MI350X / MI355X: e2m1 elements, e8m0 scales, native MXFP4 matrix instructions).

Only the emulation's ``alto`` recipe with the ``alto`` scale rule has a hardware form:
  forward  y  = Q_K(x) · Q_tile(W)ᵀ          x along K, W in 32x32 tiles
  dgrad    dx = Q_N,sr(dy) · Q_tile(W)        dy along N with stochastic rounding, same W tiles
  wgrad    dW = Q_T,sr(H·dy)ᵀ · Q_T(H·x)      along tokens after this step's 32x32 Hadamard H
where Q is MXFP4 quantization (blocks of 32, scale = ALTO's default rule) and H is the
emulation's per-step rotation (mxfp4_emulation.hadamard), so both backends draw the same H.
Deterministic parts are bit-identical to the emulation; the stochastic rounding uses the
kernels' own random numbers, so it matches the emulation in distribution only.

ALTO is imported from $ALTO_DIR (a checkout with the MI350X patches) without running its
package __init__ (which pulls training-framework dependencies the kernels do not need).
"""

import itertools
import math
import os
import sys
import types

import torch

from megatron.core import mxfp4_emulation as emu

_ALTO = None
ALIGN = 128          # rows per expert segment in ALTO's grouped layout


def _sr() -> bool:
    """Stochastic rounding of the gradient operands (the recipe). Tests switch it off together
    with the emulation's (mxfp4_emulation._FORCE_RNE) to compare both backends bit for bit."""
    return not emu._FORCE_RNE


def _alto():
    """Import the ALTO kernels once; returns a namespace with what this module uses."""
    global _ALTO
    if _ALTO is None:
        if "alto" not in sys.modules:
            root = os.environ.get("ALTO_DIR")
            if not root:
                raise RuntimeError("--mxfp4-backend alto needs ALTO_DIR (path of the ALTO checkout)")
            pkg = types.ModuleType("alto")
            pkg.__path__ = [os.path.join(root, "alto")]
            sys.modules["alto"] = pkg
        from alto.kernels.fp4.mxfp4 import mxfp_quantization  # noqa: F401  (registers the ops)
        from alto.kernels.fp4.mxfp4 import mxfp_grouped_gemm  # noqa: F401  (registers the ops)
        from alto.kernels.fp4.mxfp4 import mxfp_weight_cache
        from alto.kernels.fp4.mxfp4._eager_ops import ops
        from alto.kernels.fp4.mxfp4.mxfp_dual_quant import dual_quantize
        from alto.kernels.fp4.mxfp4.mxfp_linear import MXFP4LinearFunction
        from alto.kernels.hadamard_transform.transform import HadamardTransform

        assert mxfp_grouped_gemm.ALIGN_SIZE_M == ALIGN, mxfp_grouped_gemm.ALIGN_SIZE_M
        _ALTO = types.SimpleNamespace(ops=ops, dual_quantize=dual_quantize, wcache=mxfp_weight_cache,
                                      Linear=MXFP4LinearFunction, HadamardTransform=HadamardTransform)
    return _ALTO


def check(rule: str, recipe: str):
    if rule != "alto" or recipe != "alto":
        raise ValueError("--mxfp4-backend alto implements --mxfp4-recipe alto with "
                         f"--mxfp4-scale-rule alto only (got rule={rule}, recipe={recipe})")
    if emu._PASS_ONLY is not None:
        raise ValueError("MXFP4_PASS_ONLY is an emulation-only debug switch")
    _alto()


def new_step():
    """Called from mxfp4_emulation.new_step(): ALTO's own quantized-weight cache follows the step."""
    _alto().wcache.new_step()


def end_step():
    _alto().wcache.end_step()


def _h_bf16(device):
    """This step's Hadamard rotation (the emulation's), in bf16 as ALTO applies it."""
    key = ("h_bf16", torch.device(device))
    return emu._cached(key, lambda: emu.hadamard(device).to(torch.bfloat16).contiguous())


# ---- routed experts ------------------------------------------------------------------------
def _layout(m_splits, device):
    """ALTO's grouped layout: every expert segment padded to a multiple of 128 rows.
    Returns (rows: destination row of each token [T] int64, expert index of each padded row
    [Tp] int32, Tp)."""
    padded = [int(math.ceil(m / ALIGN) * ALIGN) for m in m_splits]
    tp = sum(padded)
    starts_p = [0] + list(itertools.accumulate(padded))[:-1]
    starts_u = [0] + list(itertools.accumulate(m_splits))[:-1]
    shift = [p - u for p, u in zip(starts_p, starts_u)]
    n_tok = sum(m_splits)
    counts = emu._to_device(m_splits, device, torch.int64)
    rows = torch.arange(n_tok, device=device) + torch.repeat_interleave(
        emu._to_device(shift, device, torch.int64), counts, output_size=n_tok)
    idx = torch.repeat_interleave(torch.arange(len(m_splits), device=device, dtype=torch.int32),
                                  emu._to_device(padded, device, torch.int64), output_size=tp)
    return rows, idx, tp


def _scatter_rows(x, rows, tp):
    return x.new_zeros(tp, x.shape[1]).index_copy_(0, rows, x)


class _AltoGroupedLinear(torch.autograd.Function):
    """y[e] = Q(x[e]) · Q(W_e)ᵀ over the local experts' token segments, on ALTO's kernels."""

    @staticmethod
    def forward(ctx, x, m_splits, *weights):
        A = _alto()
        m_splits = list(m_splits)
        assert x.shape[0] == sum(m_splits), (x.shape, sum(m_splits))
        x = x.contiguous()
        rows, idx, tp = _layout(m_splits, x.device)
        xp = _scatter_rows(x, rows, tp)
        h = _h_bf16(x.device)
        # x along K for the forward GEMM and Q(H·x) along tokens for the wgrad, in one pass
        xq, xs, xq_t, xs_t = A.dual_quantize(xp, h, use_sr=False, blockscale_selection="default")
        # stacked weights [E, N, K] and their 32x32-tile quantization, once per training step
        # (keyed by the first weight's identity, as the emulation; the cache keeps the stack
        # alive, so ALTO's own data_ptr-keyed cache cannot confuse two layers' stacks)
        w = emu._cached((id(weights[0]), "alto_stack"), lambda: torch.stack(weights))
        wq, ws = emu._cached((id(weights[0]), "alto_tile"), lambda: A.ops.convert_to_mxfp4(
            w, axis=-1, is_2d_block=True, blockscale_selection="default"))
        yp = A.ops.mxfp4_grouped_gemm_forward(
            inputs=xq, expert_weights=wq, expert_indices=idx, input_scales=xs, weight_scales=ws,
            trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, output_dtype=x.dtype)
        ctx.save_for_backward(xq_t, xs_t, wq, ws, idx, rows, *weights)
        ctx.tp, ctx.n_experts, ctx.dtype = tp, len(weights), x.dtype
        return yp.index_select(0, rows)

    @staticmethod
    def backward(ctx, dy):
        A = _alto()
        xq_t, xs_t, wq, ws, idx, rows, *weights = ctx.saved_tensors
        dyp = _scatter_rows(dy.contiguous(), rows, ctx.tp)
        h = _h_bf16(dy.device)
        # dy along N (dgrad) and Q_sr(H·dy) along tokens (wgrad), both stochastically rounded
        dyq, dys, dyq_t, dys_t = A.dual_quantize(dyp, h, use_sr=_sr(), blockscale_selection="default")
        dx = None
        if ctx.needs_input_grad[0]:
            dxp = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=dyq, expert_weights=wq, expert_indices=idx, go_scales=dys,
                expert_weight_scales=ws, trans_weights=True, use_2dblock_x=False, use_2dblock_w=True,
                k_pack_w=False, output_dtype=ctx.dtype)
            dx = dxp.index_select(0, rows)
        dw = A.ops.mxfp4_grouped_gemm_backward_weights(
            grad_output=dyq_t, inputs=xq_t, expert_indices=idx, num_experts=ctx.n_experts,
            go_scales=dys_t, input_scales=xs_t, trans_weights=True, use_2dblock_go=False,
            use_2dblock_x=False, k_pack_go=True, k_pack_x=True, output_dtype=ctx.dtype)
        grads = [emu._accumulate_wgrad(w, dw[e]) for e, w in enumerate(weights)]
        return (dx, None, *grads)


def grouped_linear(x, m_splits, weights):
    return _AltoGroupedLinear.apply(x, list(m_splits), *weights)


# ---- dense linears (shared expert, dense MLP) ----------------------------------------------
class _StepHadamard:
    """ALTO HadamardTransform (its only state is .weight) holding this step's emulation rotation."""

    def get(self, device):
        t = object.__new__(_alto().HadamardTransform)
        t.weight = _h_bf16(device)
        return t


_HT = _StepHadamard()


def linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [..., K] · Wᵀ on ALTO's dense MXFP4 GEMM (tokens padded to a multiple of 32)."""
    A = _alto()
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    t = x2.shape[0]
    tp = int(math.ceil(t / emu.BLOCK) * emu.BLOCK)
    if tp != t:
        x2 = emu._pad_rows(x2, tp)
    # (x, weight, use_2dblock_x, use_2dblock_w, use_sr_grad, use_dge, clip_mode,
    #  blockscale_selection, use_macro_block_scaling, hadamard_transform, module_id)
    y = A.Linear.apply(x2, w, False, True, _sr(), False, "none", "default", False,
                       _HT.get(x.device), None)
    return y[:t].reshape(*lead, w.shape[0])


# ---- routed-expert MLP with fc1 + SwiGLU + fc2-input quantization fused (forward) -----------
FUSED_SWIGLU = os.environ.get("MXFP4_FUSED_SWIGLU", "1") == "1"
# backward of the fused MLP: fc2 dgrad + weighted SwiGLU backward + fc1 gradient quantization in one kernel
FUSED_SWIGLU_BWD = os.environ.get("MXFP4_FUSED_SWIGLU_BWD", "1") == "1"


def _stack_tile(weights, tag):
    """Stacked weights [E, N, K] and their 32x32-tile quantization, once per training step."""
    A = _alto()
    w = emu._cached((id(weights[0]), tag + "_stack"), lambda: torch.stack(weights))
    return emu._cached((id(weights[0]), tag + "_tile"), lambda: A.ops.convert_to_mxfp4(
        w, axis=-1, is_2d_block=True, blockscale_selection="default"))


class _AltoFusedSwiGLUMLP(torch.autograd.Function):
    """out = fc2(SwiGLU(fc1(x)) * p) over the local experts' token segments (no bias), as
    TEGroupedMLP computes it with weighted SwiGLU, on ALTO's kernels with the forward fused:
    fc1 GEMM -> bf16 y -> h = bf16(silu(g) v p) -> Q(h), Q(H h) in one kernel (fused_swiglu.py).
    Backward: fc2 dgrad/wgrad, Megatron's weighted_swiglu_back, fc1 dgrad/wgrad (unfused)."""

    @staticmethod
    def forward(ctx, x, probs, m_splits, n_fc1, *weights):
        A = _alto()
        from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.fused_swiglu import fc1_swiglu_quant

        m_splits = list(m_splits)
        w1, w2 = weights[:n_fc1], weights[n_fc1:]
        x = x.contiguous()
        rows, idx, tp = _layout(m_splits, x.device)
        xp = _scatter_rows(x, rows, tp)
        pp = torch.zeros(tp, dtype=probs.dtype, device=x.device).index_copy_(0, rows, probs.reshape(-1))
        h = _h_bf16(x.device)
        xq, xs, xq_t, xs_t = A.dual_quantize(xp, h, use_sr=False, blockscale_selection="default")
        w1q, w1s = _stack_tile(w1, "fc1")
        y1, hq, hs, hq_t, hs_t = fc1_swiglu_quant(xq, xs, w1q, w1s, idx, pp, h)
        w2q, w2s = _stack_tile(w2, "fc2")
        outp = A.ops.mxfp4_grouped_gemm_forward(
            inputs=hq, expert_weights=w2q, expert_indices=idx, input_scales=hs, weight_scales=w2s,
            trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, output_dtype=x.dtype)
        ctx.save_for_backward(xq_t, xs_t, w1q, w1s, y1, pp, hq_t, hs_t, w2q, w2s, idx, rows, *weights)
        ctx.tp, ctx.n_fc1, ctx.dtype, ctx.p_shape, ctx.p_dtype = tp, n_fc1, x.dtype, probs.shape, probs.dtype
        return outp.index_select(0, rows)

    @staticmethod
    def backward(ctx, dout):
        A = _alto()
        from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back

        xq_t, xs_t, w1q, w1s, y1, pp, hq_t, hs_t, w2q, w2s, idx, rows, *weights = ctx.saved_tensors
        w1, w2 = weights[:ctx.n_fc1], weights[ctx.n_fc1:]
        h = _h_bf16(dout.device)
        gemm = dict(trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, k_pack_w=False,
                    output_dtype=ctx.dtype)
        wg = dict(trans_weights=True, use_2dblock_go=False, use_2dblock_x=False, k_pack_go=True,
                  k_pack_x=True, output_dtype=ctx.dtype)
        # fc2
        dp = _scatter_rows(dout.contiguous(), rows, ctx.tp)
        dq, ds, dq_t, ds_t = A.dual_quantize(dp, h, use_sr=_sr(), blockscale_selection="default")
        dw2 = A.ops.mxfp4_grouped_gemm_backward_weights(
            grad_output=dq_t, inputs=hq_t, expert_indices=idx, num_experts=len(w2), go_scales=ds_t,
            input_scales=hs_t, **wg)
        if FUSED_SWIGLU_BWD:
            # fc2 dgrad -> weighted SwiGLU backward -> Q_sr(dy1), Q_sr(H dy1) in one kernel
            from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.fused_swiglu import fc2_dgrad_swiglu_bwd

            gq, gs, gq_t, gs_t, dpp = fc2_dgrad_swiglu_bwd(dq, ds, w2q, w2s, idx, y1, pp, h, use_sr=_sr())
        else:
            dh = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=dq, expert_weights=w2q, expert_indices=idx, go_scales=ds,
                expert_weight_scales=w2s, **gemm)
            # weighted SwiGLU (Megatron's backward on the saved bf16 y1 and the probs)
            dy1, dpp = weighted_swiglu_back(dh, y1, pp.unsqueeze(-1))
            gq, gs, gq_t, gs_t = A.dual_quantize(dy1.contiguous(), h, use_sr=_sr(),
                                                 blockscale_selection="default")
        # fc1
        dx = None
        if ctx.needs_input_grad[0]:
            dxp = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=gq, expert_weights=w1q, expert_indices=idx, go_scales=gs,
                expert_weight_scales=w1s, **gemm)
            dx = dxp.index_select(0, rows)
        dw1 = A.ops.mxfp4_grouped_gemm_backward_weights(
            grad_output=gq_t, inputs=xq_t, expert_indices=idx, num_experts=len(w1), go_scales=gs_t,
            input_scales=xs_t, **wg)
        dprobs = dpp.reshape(-1).index_select(0, rows).reshape(ctx.p_shape).to(ctx.p_dtype)
        grads = [emu._accumulate_wgrad(w, dw1[e]) for e, w in enumerate(w1)]
        grads += [emu._accumulate_wgrad(w, dw2[e]) for e, w in enumerate(w2)]
        return (dx, dprobs, None, None, *grads)


def fused_swiglu_mlp(x, m_splits, probs, w1, w2):
    """Routed-expert MLP fc2(weighted SwiGLU(fc1(x), probs)) with the fused forward."""
    return _AltoFusedSwiGLUMLP.apply(x, probs, list(m_splits), len(w1), *w1, *w2)
