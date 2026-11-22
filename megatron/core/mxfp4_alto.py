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
# Gradient-accumulation fusion, as TE's linears do under Megatron's (default) --gradient-accumulation-fusion:
# the wgrad GEMMs add their fp32 result directly into the weights' main_grad instead of returning a bf16 dW
# that is then copied and added per expert. MXFP4_WGRAD_ACCUM=0: the previous path (bf16 dW + add).
WGRAD_ACCUM = os.environ.get("MXFP4_WGRAD_ACCUM", "1") == "1"
_PTRS = {}


def _sr() -> bool:
    """Stochastic rounding of the gradient operands (the recipe). Tests switch it off together
    with the emulation's (mxfp4_emulation._FORCE_RNE) to compare both backends bit for bit."""
    return not emu._FORCE_RNE


# ---- Triton launch path: less Python per kernel launch -------------------------------------------------
# A training step launches ~6,300 Triton kernels (5B, 8 GPUs, rank 0), ~80 us of Python each (cProfile of the fused
# MLP call, train/bench/bench_pglu_host.py): the autotuner's key (str(dtype) of every argument, hasattr on every
# scalar) and the per-tensor specialization of the AMD backend (an `import torch`, two environment lookups, hasattr
# and isinstance per tensor). Both are replaced by equivalents that produce the same keys:
#  - HIPBackend.get_tensor_specialization: "D" if aligned to 16 bytes and align requested, "S" if buffer ops are on
#    and the tensor's storage is within 2 GB (same tests, without the import and the environment lookups; the
#    AMDGCN_USE_BUFFER_OPS switch is read once);
#  - Autotuner.run, only when the key is already cached and the config has no pre-hook: the same key built with a
#    type test in front of hasattr and cached dtype strings; otherwise the original method runs.
# MXFP4_FAST_TRITON=1 installs it; the default leaves Triton untouched. Measured: host time of one fused MLP call
# -0.10 .. -0.16 ms (bench_pglu_host.py, job 532479); 5B training step, two back-to-back sets on one node (jobs
# 532486-9, 532676-9), median ms without / with: pglu 3409, 3412 / 3386, 3386; SwiGLU 3355, 3364 / 3365, 3394.
# So -24 ms for pglu both times, no gain (+10, +30 ms) for SwiGLU: not on by default.
# Later, with sort folding, Triton RMSNorm and the bookkeeping kernels of moe/triton_fast.py on (more of the step is
# then limited by host time between kernel launches), SwiGLU, without / with: 3,129 / 3,083 and 3,035 / 3,003 (job
# 535995), 3,020 / 2,987 and 2,985 (job 536090): -31 .. -46 ms in four comparisons. Still opt-in, because it replaces
# two methods of the Triton library.
FAST_TRITON = os.environ.get("MXFP4_FAST_TRITON", "0") == "1"
_TRITON_ORIG = {}


def fast_triton(on=True):
    """Install (on) or remove the faster Triton launch path described above."""
    from triton import knobs
    from triton.backends.amd.compiler import HIPBackend
    from triton.runtime.autotuner import Autotuner

    if not on:
        if _TRITON_ORIG:
            HIPBackend.get_tensor_specialization = staticmethod(_TRITON_ORIG["spec"])
            Autotuner.run = _TRITON_ORIG["run"]
            _TRITON_ORIG.clear()
        return
    if _TRITON_ORIG:
        return
    orig_spec = HIPBackend.get_tensor_specialization
    orig_run = Autotuner.run
    _TRITON_ORIG.update(spec=orig_spec, run=orig_run)
    use_buffer_ops = bool(knobs.amd.use_buffer_ops)
    max_int32 = 2**31 - 1
    tensor = torch.Tensor

    def spec(arg, **kwargs):
        if isinstance(arg, tensor):
            ret = "D" if (arg.data_ptr() % 16 == 0 and kwargs.get("align", False)) else ""
            if use_buffer_ops and arg.untyped_storage().size() <= max_int32:
                ret += "S"
            return ret
        return orig_spec(arg, **kwargs)

    scalars = (int, float, bool, str, type(None))      # exact types without a dtype attribute
    dtype_str, cfg_kwargs, name_sets = {}, {}, {}

    def run(self, *args, **kwargs):
        if len(self.configs) > 1:
            names = name_sets.get(id(self))
            if names is None:
                names = name_sets[id(self)] = frozenset(self.arg_names)
            nargs = dict(zip(self.arg_names, args))
            _args = nargs
            for k in kwargs:
                if k in names:
                    _args = {**nargs, **{k: v for k, v in kwargs.items() if k in names}}
                    break
            key = [_args[k] for k in self.keys if k in _args]
            for arg in _args.values():
                if type(arg) not in scalars and hasattr(arg, "dtype"):
                    dt = arg.dtype
                    s_ = dtype_str.get(dt)
                    if s_ is None:
                        s_ = dtype_str[dt] = str(dt)
                    key.append(s_)
            config = self.cache.get(tuple(key))
            if config is not None and config.pre_hook is None:
                self.nargs = nargs
                self.best_config = config
                ck = cfg_kwargs.get(id(config))
                if ck is None:
                    ck = cfg_kwargs[id(config)] = config.all_kwargs()
                ret = self.fn.run(*args, **kwargs, **ck)
                self.nargs = None
                return ret
        return orig_run(self, *args, **kwargs)

    HIPBackend.get_tensor_specialization = staticmethod(spec)
    Autotuner.run = run


def _alto():
    """Import the ALTO kernels once; returns a namespace with what this module uses."""
    global _ALTO
    if _ALTO is None:
        if FAST_TRITON:
            fast_triton(True)
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

        from alto.kernels.fp4.mxfp4 import mxfp_linear
        from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.wgrad_segmented import wgrad_accumulate

        assert mxfp_grouped_gemm.ALIGN_SIZE_M == ALIGN, mxfp_grouped_gemm.ALIGN_SIZE_M
        mxfp_linear.WGRAD_INTO_MAIN_GRAD = WGRAD_ACCUM       # dense linears: wgrad GEMM adds into main_grad
        _ALTO = types.SimpleNamespace(ops=ops, dual_quantize=dual_quantize, wcache=mxfp_weight_cache,
                                      Linear=MXFP4LinearFunction, HadamardTransform=HadamardTransform,
                                      wgrad_accumulate=wgrad_accumulate)
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


_IDX0 = {}


def _idx_unpadded(m_splits, device):
    """Expert index of every row when no segment needs padding (all multiples of 128)."""
    if len(m_splits) == 1:            # shared expert / dense MLP: one expert, cached
        key = (m_splits[0], torch.device(device))
        if key not in _IDX0:
            _IDX0[key] = torch.zeros(m_splits[0], dtype=torch.int32, device=device)
        return _IDX0[key]
    return torch.repeat_interleave(torch.arange(len(m_splits), device=device, dtype=torch.int32),
                                   emu._to_device(m_splits, device, torch.int64), output_size=sum(m_splits))


def _scatter_rows(x, rows, tp):
    return x.new_zeros(tp, x.shape[1]).index_copy_(0, rows, x)


# The padded layout without padded tensors: the kernels that read Megatron's bf16 tensors (the dual
# quantization of x and dout, the probs in the fused SwiGLU) gather rows on load, and the GEMMs whose bf16
# results go back to Megatron (fc2 forward, fc1 dgrad, d(probs)) scatter rows on store, through
# src[m] = token of grouped row m (-1: padding). Only ALTO-internal tensors (FP4 operands, y1) keep the
# padded rows. MXFP4_ROW_MAP=0: scatter to / gather from padded bf16 copies as before (same results).
ROW_MAP = os.environ.get("MXFP4_ROW_MAP", "1") == "1"


def _src_rows(rows, tp, perm=None):
    """Inverse of the layout's row map: token index of every grouped row, -1 on padding rows [Tp] int32.
    perm [T] int32 (optional): row of the caller's buffer holding expert-sorted token t (chunk_perm); the map
    then points into that buffer, so the tokens need not be sorted by expert first."""
    n = rows.shape[0]
    return torch.full((tp,), -1, dtype=torch.int32, device=rows.device).index_copy_(
        0, rows, perm if perm is not None else torch.arange(n, dtype=torch.int32, device=rows.device))


# MXFP4_SORT_FOLD=1 (default): the all-to-all token dispatcher does not sort the received tokens by local expert
# (and does not un-sort the expert output); the fused routed MLP reads and writes them in the received order
# through its row map. Removes TE's chunk sort (forward, backward, both directions) and its map kernels. The
# grouped layout the kernels see is the same, so the results are unchanged. 0: sort as before.
SORT_FOLD = os.environ.get("MXFP4_SORT_FOLD", "1") == "1"


def chunk_perm(split_sizes, sorted_idxs, n, device):
    """sort_chunks_by_idxs(input, split_sizes, sorted_idxs) as an index map: output row t = input row perm[t].
    split_sizes [C]: lengths of the consecutive chunks of the input; sorted_idxs [C]: chunk order of the output;
    n = number of rows (sum of the sizes, passed to avoid a host sync). Returns perm [n] int32."""
    sizes = split_sizes.to(device=device, dtype=torch.int64)
    idxs = sorted_idxs.to(device=device, dtype=torch.int64)
    starts = sizes.cumsum(0) - sizes                       # first input row of each chunk
    ls = sizes.index_select(0, idxs)                       # chunk lengths in output order
    dst = ls.cumsum(0) - ls                                # first output row of each chunk
    shift = starts.index_select(0, idxs) - dst
    return (torch.repeat_interleave(shift, ls, output_size=n)
            + torch.arange(n, device=device)).to(torch.int32)


# MXFP4_CHUNK_KERNEL=1: with SORT_FOLD, the row maps (src, idx) come from one Triton kernel that reads the chunk
# sizes (megatron/core/transformer/moe/triton_fast.py chunk_layout) instead of chunk_perm + _layout + _src_rows
# (about 25 small torch operations issued while the GPU has nothing queued). Same maps. Default on.
CHUNK_KERNEL = os.environ.get("MXFP4_CHUNK_KERNEL", "1") == "1"
_RANK_EXPERT_ORDER = {}


def chunk_order(split_sizes, sorted_idxs, n, device, n_experts):
    """The token order argument of fused_swiglu_mlp(perm=) for the dispatcher's skipped sort
    sort_chunks_by_idxs(input, split_sizes, sorted_idxs): chunk_perm's index map, or with CHUNK_KERNEL the tuple
    ("chunks", sizes on the device, number of source ranks) when the chunks arrive ordered by (source rank, local
    expert) and are wanted by (local expert, source rank) - the all-to-all dispatcher's order (checked once per
    sorted_idxs tensor)."""
    if CHUNK_KERNEL:
        c = sorted_idxs.numel()
        r = c // n_experts
        hit = _RANK_EXPERT_ORDER.get(id(sorted_idxs))
        if hit is None or hit[0] is not sorted_idxs:
            ok = c == r * n_experts and bool(torch.equal(
                sorted_idxs.cpu(), torch.arange(c).reshape(r, n_experts).T.ravel()))
            hit = _RANK_EXPERT_ORDER[id(sorted_idxs)] = (sorted_idxs, ok)
        if hit[1]:
            return ("chunks", split_sizes.to(device=device, non_blocking=True), r)
    return chunk_perm(split_sizes, sorted_idxs, n, device)


class _AltoGroupedLinear(torch.autograd.Function):
    """y[e] = Q(x[e]) · Q(W_e)ᵀ over the local experts' token segments, on ALTO's kernels."""

    @staticmethod
    def forward(ctx, x, m_splits, *weights):
        A = _alto()
        m_splits = list(m_splits)
        assert x.shape[0] == sum(m_splits), (x.shape, sum(m_splits))
        x = x.contiguous()
        rows, idx, tp = _layout(m_splits, x.device)
        src = _src_rows(rows, tp) if ROW_MAP else None
        h = _h_bf16(x.device)
        # x along K for the forward GEMM and Q(H·x) along tokens for the wgrad, in one pass
        if ROW_MAP:
            xq, xs, xq_t, xs_t = A.dual_quantize(x, h, use_sr=False, blockscale_selection="default", src_rows=src)
        else:
            xp = _scatter_rows(x, rows, tp)
            xq, xs, xq_t, xs_t = A.dual_quantize(xp, h, use_sr=False, blockscale_selection="default")
        # stacked weights [E, N, K] and their 32x32-tile quantization, once per training step
        # (keyed by the first weight's identity, as the emulation; the cache keeps the stack
        # alive, so ALTO's own data_ptr-keyed cache cannot confuse two layers' stacks)
        w = emu._cached((id(weights[0]), "alto_stack"), lambda: torch.stack(weights))
        wq, ws = emu._cached((id(weights[0]), "alto_tile"), lambda: A.ops.convert_to_mxfp4(
            w, axis=-1, is_2d_block=True, blockscale_selection="default"))
        yp = A.ops.mxfp4_grouped_gemm_forward(
            inputs=xq, expert_weights=wq, expert_indices=idx, input_scales=xs, weight_scales=ws,
            trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, output_dtype=x.dtype,
            **(dict(out_rows=src, out_len=x.shape[0]) if ROW_MAP else {}))
        ctx.save_for_backward(xq_t, xs_t, wq, ws, idx, src if ROW_MAP else rows, *weights)
        ctx.tp, ctx.n_experts, ctx.dtype, ctx.row_map = tp, len(weights), x.dtype, ROW_MAP
        return yp if ROW_MAP else yp.index_select(0, rows)

    @staticmethod
    def backward(ctx, dy):
        A = _alto()
        xq_t, xs_t, wq, ws, idx, rows, *weights = ctx.saved_tensors      # rows: src map if ctx.row_map
        h = _h_bf16(dy.device)
        # dy along N (dgrad) and Q_sr(H·dy) along tokens (wgrad), both stochastically rounded
        if ctx.row_map:
            dyq, dys, dyq_t, dys_t = A.dual_quantize(dy.contiguous(), h, use_sr=_sr(),
                                                     blockscale_selection="default", src_rows=rows)
        else:
            dyp = _scatter_rows(dy.contiguous(), rows, ctx.tp)
            dyq, dys, dyq_t, dys_t = A.dual_quantize(dyp, h, use_sr=_sr(), blockscale_selection="default")
        dx = None
        if ctx.needs_input_grad[0]:
            dxp = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=dyq, expert_weights=wq, expert_indices=idx, go_scales=dys,
                expert_weight_scales=ws, trans_weights=True, use_2dblock_x=False, use_2dblock_w=True,
                k_pack_w=False, output_dtype=ctx.dtype,
                **(dict(out_rows=rows, out_len=dy.shape[0]) if ctx.row_map else {}))
            dx = dxp if ctx.row_map else dxp.index_select(0, rows)
        grads = _grouped_wgrad(dyq_t, dys_t, xq_t, xs_t, idx, weights, ctx.dtype)
        return (dx, None, *grads)


def _main_grad_ok(weights):
    w0 = weights[0]
    if not (hasattr(w0, "main_grad") and getattr(w0, "grad_added_to_main_grad", None) is not None):
        return False
    mg0 = w0.main_grad
    return all(hasattr(w, "main_grad") and getattr(w, "grad_added_to_main_grad", None) is not None
               and w.main_grad.shape == w.shape == mg0.shape and w.main_grad.stride() == mg0.stride()
               and w.main_grad.dtype == mg0.dtype for w in weights) and mg0.dtype in (torch.float32,
                                                                                  torch.bfloat16)


def _grouped_wgrad(go_t, go_s, x_t, x_s, idx, weights, dtype):
    """Weight gradients of the local experts: dW[e] = Q(H·dy)_eᵀ · Q(H·x)_e (ALTO's grouped wgrad), returned
    as the autograd grads of `weights`. With WGRAD_ACCUM and Megatron's main_grad, the kernel adds its fp32
    result into every expert's main_grad in place and the returned grads are placeholders (fresh, so
    autograd's AccumulateGrad takes them without a copy; DDP skips them: grad_added_to_main_grad)."""
    A = _alto()
    if len(weights) == 1:
        # one expert (shared expert, dense MLP): the dense MXFP4 wgrad GEMM (the kernel the unfused dense path
        # uses); the grouped kernel parallelises over experts and would run few programs over all tokens
        from alto.kernels.fp4.mxfp4.mxfp_linear import blockwise_mxfp4_gemm_accumulate_

        w = weights[0]
        if WGRAD_ACCUM and _main_grad_ok(weights):
            blockwise_mxfp4_gemm_accumulate_(go_t, go_s, x_t, x_s, w.main_grad)
            w.grad_added_to_main_grad = True
            return [torch.empty_like(w)]
        dw = A.ops.blockwise_mxfp4_gemm(go_t, go_s, x_t, x_s, trans_a=True, output_dtype=dtype)
        return [emu._accumulate_wgrad(w, dw)]
    if WGRAD_ACCUM and _main_grad_ok(weights):
        ptrs = tuple(w.main_grad.data_ptr() for w in weights)
        t = _PTRS.get(ptrs)
        if t is None:
            t = _PTRS[ptrs] = torch.tensor(ptrs, dtype=torch.int64, device=go_t.device)
        A.wgrad_accumulate(go_t, x_t, idx, go_s, x_s, t, weights[0].main_grad)
        for w in weights:
            w.grad_added_to_main_grad = True
        return list(torch.empty((len(weights), *weights[0].shape), dtype=weights[0].dtype,
                                device=go_t.device).unbind(0))
    dw = A.ops.mxfp4_grouped_gemm_backward_weights(
        grad_output=go_t, inputs=x_t, expert_indices=idx, num_experts=len(weights), go_scales=go_s,
        input_scales=x_s, trans_weights=True, use_2dblock_go=False, use_2dblock_x=False, k_pack_go=True,
        k_pack_x=True, output_dtype=dtype)
    return [emu._accumulate_wgrad(w, dw[e]) for e, w in enumerate(weights)]


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
    Backward: fc2 wgrad; fc2 dgrad + SwiGLU backward + Q(dy1) in one kernel (FUSED_SWIGLU_BWD); fc1 dgrad/wgrad.
    With raw_p, gb (exp_gates.ExpPGLU parameters, one per local expert) the gate is pglu instead of silu
    (kernels with GATE=1; megatron/core/exp_gates.py is the reference) and their gradients are returned."""

    @staticmethod
    def forward(ctx, x, probs, raw_p, gb, perm, m_splits, n_fc1, *weights):
        A = _alto()
        from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.fused_swiglu import fc1_swiglu_quant

        m_splits = list(m_splits)
        w1, w2 = weights[:n_fc1], weights[n_fc1:]
        x = x.contiguous()
        # segments already 128-aligned (e.g. the shared expert / dense MLP: one expert, 16384 tokens):
        # the grouped layout is the token order itself, no maps and no copies
        # perm (chunk_perm): x and probs are in the dispatcher's received order, not sorted by expert
        nopad = perm is None and all(m % ALIGN == 0 for m in m_splits)
        assert perm is None or ROW_MAP, "token order map needs MXFP4_ROW_MAP=1"
        chunks = perm if isinstance(perm, tuple) else None    # chunk_order with CHUNK_KERNEL: maps from one kernel
        if nopad:
            rows, idx, tp = None, _idx_unpadded(m_splits, x.device), x.shape[0]
        elif chunks is not None:
            from megatron.core.transformer.moe import triton_fast

            rows = None
            src, idx, tp = triton_fast.chunk_layout(chunks[1], chunks[2], m_splits, x.device, ALIGN)
        else:
            rows, idx, tp = _layout(m_splits, x.device)
        use_map = ROW_MAP and not nopad
        h = _h_bf16(x.device)
        weighted = probs is not None
        if not weighted:      # plain SwiGLU (shared expert, dense MLP): p = 1 (h = silu(g) v exactly)
            probs = _ones(x.shape[0], x.device)
        if nopad:
            src = None
            pp = probs.reshape(-1).contiguous()
            xq, xs, xq_t, xs_t = A.dual_quantize(x, h, use_sr=False, blockscale_selection="default")
        elif use_map:
            # no padded copies: the kernels read x and probs through src, fc2 writes out in token order
            if chunks is None:
                src = _src_rows(rows, tp, perm)
            pp = probs.reshape(-1).contiguous()
            xq, xs, xq_t, xs_t = A.dual_quantize(x, h, use_sr=False, blockscale_selection="default",
                                                 src_rows=src)
        else:
            src = None
            xp = _scatter_rows(x, rows, tp)
            pp = torch.zeros(tp, dtype=probs.dtype, device=x.device).index_copy_(0, rows, probs.reshape(-1))
            xq, xs, xq_t, xs_t = A.dual_quantize(xp, h, use_sr=False, blockscale_selection="default")
        w1q, w1s = _stack_tile(w1, "fc1")
        pglu = (raw_p, gb) if raw_p is not None else None      # read by the kernel only (no autograd in here)
        y1, hq, hs, hq_t, hs_t = fc1_swiglu_quant(xq, xs, w1q, w1s, idx, pp, h, src_rows=src, pglu=pglu)
        w2q, w2s = _stack_tile(w2, "fc2")
        outp = A.ops.mxfp4_grouped_gemm_forward(
            inputs=hq, expert_weights=w2q, expert_indices=idx, input_scales=hs, weight_scales=w2s,
            trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, output_dtype=x.dtype,
            **(dict(out_rows=src, out_len=x.shape[0]) if use_map else {}))
        ctx.save_for_backward(xq_t, xs_t, w1q, w1s, y1, pp, hq_t, hs_t, w2q, w2s, idx,
                              src if use_map else rows, raw_p, gb, *weights)
        # row counts of the experts' segments in the grouped layout (for the pglu coefficient gradients)
        ctx.seg = m_splits if nopad else [int(math.ceil(m / ALIGN) * ALIGN) for m in m_splits]
        # (the fused backward kernel finds the segments from idx; only the unfused backward uses ctx.seg)
        ctx.tp, ctx.n_fc1, ctx.dtype, ctx.p_shape, ctx.p_dtype = tp, n_fc1, x.dtype, probs.shape, probs.dtype
        ctx.row_map, ctx.nopad, ctx.weighted = use_map, nopad, weighted
        return outp if (use_map or nopad) else outp.index_select(0, rows)

    @staticmethod
    def backward(ctx, dout):
        A = _alto()
        from megatron.core.fusions.fused_bias_swiglu import weighted_swiglu_back

        xq_t, xs_t, w1q, w1s, y1, pp, hq_t, hs_t, w2q, w2s, idx, rows, raw_p, gb, *weights = ctx.saved_tensors
        w1, w2 = weights[:ctx.n_fc1], weights[ctx.n_fc1:]
        pglu = raw_p is not None
        h = _h_bf16(dout.device)
        gemm = dict(trans_weights=True, use_2dblock_x=False, use_2dblock_w=True, k_pack_w=False,
                    output_dtype=ctx.dtype)
        rm = ctx.row_map                   # rows = src map (token of each grouped row) if rm, else token -> row
        src = rows if rm else None
        direct = rm or ctx.nopad           # results already in token order
        # fc2
        if ctx.nopad:
            dq, ds, dq_t, ds_t = A.dual_quantize(dout.contiguous(), h, use_sr=_sr(),
                                                 blockscale_selection="default")
        elif rm:
            dq, ds, dq_t, ds_t = A.dual_quantize(dout.contiguous(), h, use_sr=_sr(),
                                                 blockscale_selection="default", src_rows=src)
        else:
            dp = _scatter_rows(dout.contiguous(), rows, ctx.tp)
            dq, ds, dq_t, ds_t = A.dual_quantize(dp, h, use_sr=_sr(), blockscale_selection="default")
        grads2 = _grouped_wgrad(dq_t, ds_t, hq_t, hs_t, idx, w2, ctx.dtype)
        if FUSED_SWIGLU_BWD:
            # fc2 dgrad -> weighted SwiGLU backward -> Q_sr(dy1), Q_sr(H dy1) in one kernel
            from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.fused_swiglu import fc2_dgrad_swiglu_bwd

            gq, gs, gq_t, gs_t, dpp, *dpg = fc2_dgrad_swiglu_bwd(
                dq, ds, w2q, w2s, idx, y1, pp, h, use_sr=_sr(), src_rows=src,
                pglu=(raw_p, gb) if pglu else None)
        else:
            if rm:                         # Megatron's backward works on the padded rows: padded probs
                pp = torch.zeros(ctx.tp, dtype=pp.dtype, device=pp.device).index_copy_(
                    0, (src >= 0).nonzero().squeeze(1), pp.index_select(0, src[src >= 0].long()))
            dh = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=dq, expert_weights=w2q, expert_indices=idx, go_scales=ds,
                expert_weight_scales=w2s, **{**gemm, **(dict(output_dtype=torch.float32) if pglu else {})})
            if pglu:
                # pglu backward (Megatron's reference, exp_gates) on fp32 dh, per grouped row
                from megatron.core import exp_gates

                coef = exp_gates.pglu_coefficients(raw_p, gb)
                dy1, dpp, s = exp_gates.pglu_bwd_rows(dh, y1, pp.unsqueeze(-1), coef.index_select(0, idx.long()))
                dpg = exp_gates.pglu_param_grads(raw_p, exp_gates.segment_sum(s, ctx.seg))
            else:
                # weighted SwiGLU (Megatron's backward on the saved bf16 y1 and the probs)
                dy1, dpp = weighted_swiglu_back(dh, y1, pp.unsqueeze(-1))
            if rm:
                dpp = dpp.reshape(-1)[src >= 0].new_zeros(dout.shape[0]).index_copy_(
                    0, src[src >= 0].long(), dpp.reshape(-1)[src >= 0])
            gq, gs, gq_t, gs_t = A.dual_quantize(dy1.contiguous(), h, use_sr=_sr(),
                                                 blockscale_selection="default")
        # fc1
        dx = None
        if ctx.needs_input_grad[0]:
            dxp = A.ops.mxfp4_grouped_gemm_backward_inputs(
                grad_output=gq, expert_weights=w1q, expert_indices=idx, go_scales=gs,
                expert_weight_scales=w1s, **gemm,
                **(dict(out_rows=src, out_len=dout.shape[0]) if rm else {}))
            dx = dxp if direct else dxp.index_select(0, rows)
        grads1 = _grouped_wgrad(gq_t, gs_t, xq_t, xs_t, idx, w1, ctx.dtype)
        dprobs = None
        if ctx.weighted:
            dpp = dpp.reshape(-1)
            dprobs = (dpp if direct else dpp.index_select(0, rows)).reshape(ctx.p_shape).to(ctx.p_dtype)
        grads = grads1 + grads2
        d_raw = d_b = None
        if pglu:
            d_raw, d_b = dpg[0], dpg[1]
            if d_raw.dtype != raw_p.dtype or d_b.dtype != gb.dtype:
                d_raw, d_b = d_raw.to(raw_p.dtype), d_b.to(gb.dtype)
        return (dx, dprobs, d_raw, d_b, None, None, None, *grads)


_ONES = {}


def _ones(n, device):
    key = (n, torch.device(device))
    if key not in _ONES:
        _ONES[key] = torch.ones(n, dtype=torch.float32, device=device)
    return _ONES[key]


def _gate_params(gate):
    """(raw_p, b) of an exp_gates.ExpPGLU module, (None, None) for SwiGLU."""
    if gate is None:
        return None, None
    from megatron.core import exp_gates

    assert isinstance(gate, exp_gates.ExpPGLU), f"fused MXFP4 MLP: unsupported gate {type(gate).__name__}"
    return gate.raw_p, gate.b


def fused_swiglu_dense(x, w1, w2, gate=None):
    """Dense MLP fc2(SwiGLU(fc1(x))) (shared expert, dense MLP layer) on the fused kernels of the routed
    MLP, as one expert holding all tokens: x [T, K], w1 [2F, K], w2 [K, F]. Same numerics as the unfused
    MXFP4 path (MXFP4 linear, Megatron's bias_swiglu_impl, MXFP4 linear). gate: an exp_gates.ExpPGLU
    (pglu instead of SwiGLU) or None."""
    return _AltoFusedSwiGLUMLP.apply(x, None, *_gate_params(gate), None, [x.shape[0]], 1, w1, w2)


def fuse_dense_mlps(decoder, patched):
    """--mxfp4-emulate all_linear on this backend: MLP / SharedExpertMLP modules whose fc1 and fc2 are both
    MXFP4 (names in `patched`) and compute exactly fc1 -> SwiGLU -> fc2 run on fused_swiglu_dense (fc1
    epilogue: SwiGLU + fc2 input quantization; backward: fc2 dgrad + SwiGLU backward + fc1 gradient
    quantization). Returns the names of the fused modules."""
    import types

    import torch.nn.functional as F

    from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    if not FUSED_SWIGLU:
        return []
    names = set(patched)
    fused = []
    for name, mod in decoder.named_modules():
        if not isinstance(mod, MLP) or f"{name}.linear_fc1" not in names or f"{name}.linear_fc2" not in names:
            continue
        c = mod.config
        pglu = c.exp_learnable_gate == 'exp_pglu'      # pglu (one p, b per MLP) instead of SwiGLU
        if not (c.gated_linear_unit and mod.activation_func == F.silu and (c.bias_activation_fusion or pglu)
                and not c.use_te_activation_func and c.activation_func_clamp_value is None
                and c.glu_linear_offset == 0.0 and not c.activation_func_fp8_input_store
                and not (c.cpu_offloading and c.cpu_offloading_activations) and not (c.fp8 or c.fp4)
                and not c.add_bias_linear):
            continue
        norm = isinstance(mod.linear_fc1, TELayerNormColumnParallelLinear)
        shared = isinstance(mod, SharedExpertMLP)

        def fused_mlp(self, hidden_states, _norm=norm):
            """fc2(gate(fc1(norm(x)))) on the fused kernels (also called by SharedExpertMLP's overlap path)."""
            x = emu._fused_norm(self.linear_fc1, hidden_states) if _norm else hidden_states
            lead = x.shape[:-1]
            return fused_swiglu_dense(x.reshape(-1, x.shape[-1]), self.linear_fc1.weight,
                                      self.linear_fc2.weight, gate=getattr(self, "exp_gate", None)).reshape(*lead, -1)

        mod._mxfp4_fused_mlp = types.MethodType(fused_mlp, mod)

        def forward(self, hidden_states, per_token_scale=None, _norm=norm, _shared=shared, **kwargs):
            assert per_token_scale is None, "fused MXFP4 MLP: per_token_scale is not supported"
            out = self._mxfp4_fused_mlp(hidden_states)
            if not _shared:
                return out, None
            if self.use_shared_expert_gate:
                out = out * torch.sigmoid(torch.nn.functional.linear(hidden_states, self.gate_weight))
            return out

        mod.forward = types.MethodType(forward, mod)
        fused.append(name)
    return fused


def fused_swiglu_mlp(x, m_splits, probs, w1, w2, gate=None, perm=None):
    """Routed-expert MLP fc2(weighted SwiGLU(fc1(x), probs)) on the fused kernels; gate: an exp_gates.ExpPGLU
    with one (p, b) per local expert (weighted pglu instead of SwiGLU) or None. perm (chunk_perm, optional): x
    and probs are not sorted by expert; expert-sorted token t is row perm[t]. The output has x's row order."""
    return _AltoFusedSwiGLUMLP.apply(x, probs, *_gate_params(gate), perm, list(m_splits), len(w1), *w1, *w2)
