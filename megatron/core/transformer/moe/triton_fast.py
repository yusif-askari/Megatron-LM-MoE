"""Faster Triton kernels for three bookkeeping steps of the MoE layer. Same results as the code they replace.

1. make_row_id_map: the map Transformer Engine's fused token permute / unpermute read
   (transformer_engine.pytorch.triton.permutation.make_row_id_map: three kernels - a cumsum per (expert, 1024-token
   block) into a [tokens, experts] int32 buffer, a second pass in which every program loads the counts of all
   earlier blocks, and a 256-element bitonic sort per token to move the routed experts to the front).
   Here: the per-block counts with three torch ops and one kernel that works on [BT tokens, all experts] tiles,
   ranks the tokens of each expert with a cumsum along the tokens and compacts the routed experts of each token
   with a cumsum along the experts (no sort, no intermediate buffer).
   MOE_FAST_ROW_ID_MAP=1 installs it in place of Transformer Engine's function (install()).

2. qb_histogram: the per-expert histogram of the quantile-balancing router
   (moe_utils.compute_qb_histogram: about ten elementwise torch kernels and a scatter_add_ of tokens x experts
   int64 ones). Here: one kernel that computes the bin of every (token, expert) score and counts it with an
   atomic add. MOE_FAST_QB_HIST=1 makes compute_qb_histogram use it.

3. chunk_layout: the row maps of the fused MXFP4 routed MLP when the received tokens are not sorted by expert
   (megatron/core/mxfp4_alto.py, MXFP4_CHUNK_KERNEL=1), in one kernel.

All three are on by default (set the variable to 0 to get the replaced code back).
Equality with the replaced code and time per call: train/bench/bench_moe_maps.py (optional-project, job 535391);
tests/unit_tests/test_mxfp4_sort_fold.py for chunk_layout inside the fused MLP (job 535392).
5B training step, 8 GPUs, SwiGLU, sort folding and Triton RMSNorm on, median of steps 10-30 (job 535995):
without 3,134 and 3,123 ms; row-id map alone 3,107; histogram alone 3,115; chunk_layout alone 3,049; all three
3,031 and 3,038."""
import os

import torch

FAST_ROW_ID_MAP = os.environ.get("MOE_FAST_ROW_ID_MAP", "1") == "1"
FAST_QB_HIST = os.environ.get("MOE_FAST_QB_HIST", "1") == "1"
# tile sizes / warps (benchmarked in bench_moe_maps.py)
RIM_BT = int(os.environ.get("MOE_RIM_BT", "32"))
RIM_WARPS = int(os.environ.get("MOE_RIM_WARPS", "4"))
QB_BT = int(os.environ.get("MOE_QB_BT", "64"))
QB_BE = int(os.environ.get("MOE_QB_BE", "64"))
QB_WARPS = int(os.environ.get("MOE_QB_WARPS", "4"))
QB_I32 = os.environ.get("MOE_QB_I32", "1") == "1"       # count in int32 (one cast afterwards) or directly in int64
# Host-time switches (off by default until measured in a training step; see the functions below):
# MOE_RIM_PREFIX_KERNEL=1: the running counts that make_row_id_map needs come from one kernel instead of five small
#   torch operations (cast of the whole routing map to int32, block sum, transpose copy, cumulative sum = 2 kernels).
# MOE_DIRECT_PERMUTE=1: Megatron's fused permute / unpermute call Transformer Engine's Triton functions through a
#   plain torch.autograd.Function (permute_with_probs, unpermute below) instead of through TE's torch.library
#   custom operators, whose dispatch (operator call, generated autograd wrapper, redispatch, backend wrapper) runs
#   on the host before the first kernel of every call.
# MOE_HOST_SPLITS=1 (token_dispatcher.MoEAlltoAllTokenDispatcher._preprocess_host_splits): the token counts of all
#   ranks go to the host in one copy and the split sizes are computed there, instead of five small kernels, four
#   device-to-host copies and a side stream.
HOST_SPLITS = os.environ.get("MOE_HOST_SPLITS", "1") == "1"
# MOE_QB_RANGE_IN_KERNEL=1: the histogram kernel computes its bin range from beta itself, instead of nine small
#   torch operations (max, min, negations, additions, division, stack) launched before it.
QB_RANGE = os.environ.get("MOE_QB_RANGE_IN_KERNEL", "1") == "1"
RIM_PREFIX = os.environ.get("MOE_RIM_PREFIX_KERNEL", "1") == "1"
# prefix kernel tiles (bench_moe_maps.py, job 536713: make_row_id_map 51 us with the torch operations, 39 us with
# one expert per program and 128 blocks per iteration; 86 us with 8 experts and 16 blocks)
RIM_EB = int(os.environ.get("MOE_RIM_EB", "1"))            # experts per program of the prefix kernel
RIM_PWARPS = int(os.environ.get("MOE_RIM_PWARPS", "4"))
RIM_NB = int(os.environ.get("MOE_RIM_NB", "128"))          # token blocks per iteration of the prefix kernel
DIRECT_PERMUTE = os.environ.get("MOE_DIRECT_PERMUTE", "1") == "1"

_K = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl

    @triton.jit
    def _row_id_map_kernel(map_ptr, incl_ptr, tot_ptr, out_ptr, stride_mt, stride_me, stride_ot, stride_oe, n_blk,
                           E: tl.constexpr, EP: tl.constexpr, BT: tl.constexpr, LOCAL: tl.constexpr):
        # one program: BT consecutive tokens x all experts
        blk = tl.program_id(0)
        t = blk * BT + tl.arange(0, BT)
        e = tl.arange(0, EP)
        em = e < E
        m = tl.load(map_ptr + t[:, None] * stride_mt + e[None, :] * stride_me, mask=em[None, :], other=0).to(tl.int32)
        m = tl.where(m != 0, 1, 0)
        rank = tl.cumsum(m, axis=0)                   # tokens of this block, up to and including t, routed to e
        col = tl.sum(m, axis=0)
        # incl[e, blk]: routed (token, expert) pairs of all earlier experts + of expert e up to the end of this block
        if LOCAL:
            # incl_ptr (from _rim_prefix_kernel): pairs of expert e alone up to the end of this block; the pairs of
            # all earlier experts are the exclusive running sum of the experts' totals
            tot = tl.load(tot_ptr + e, mask=em, other=0)
            first = tl.cumsum(tot, axis=0) - tot
            base = first + tl.load(incl_ptr + e.to(tl.int64) * n_blk + blk, mask=em, other=0) - col
        else:
            incl = tl.load(incl_ptr + e.to(tl.int64) * n_blk + blk, mask=em, other=0)
            base = (incl - col).to(tl.int32)
        row = base[None, :] + rank - 1                # destination row of (t, e) in the permuted tokens
        pos = tl.cumsum(m, axis=1)                    # routed experts of token t up to and including e
        n = tl.sum(m, axis=1)
        slot = n[:, None] - pos                       # largest destination row (= largest expert) first, as TE's sort
        keep = (m != 0) & em[None, :]
        o = out_ptr + t[:, None] * stride_ot
        tl.store(o + slot * stride_oe, row, mask=keep)
        tl.store(o + (E + slot) * stride_oe, e[None, :] + tl.zeros((BT, EP), tl.int32), mask=keep)
        tl.store(out_ptr + t * stride_ot + (2 * E) * stride_oe, n)

    @triton.jit
    def _rim_prefix_kernel(map_ptr, incl_ptr, tot_ptr, stride_mt, stride_me, n_blk, E,
                           BT: tl.constexpr, EB: tl.constexpr, NB: tl.constexpr):
        # one program: EB experts, all tokens in order, NB blocks of BT tokens per iteration.
        # incl[e, blk] (int32) = tokens routed to expert e in the token blocks 0 .. blk; tot[e] = all its tokens.
        e = tl.program_id(0) * EB + tl.arange(0, EB)
        em = e < E
        carry = tl.zeros((EB,), tl.int32)
        for b0 in tl.range(0, n_blk, NB):
            t = b0 * BT + tl.arange(0, NB * BT)
            live = (t < n_blk * BT)[:, None] & em[None, :]
            m = tl.load(map_ptr + t[:, None] * stride_mt + e[None, :] * stride_me, mask=live, other=0)
            m = tl.where(m.to(tl.int32) != 0, 1, 0)
            cnt = tl.sum(tl.reshape(m, (NB, BT, EB)), axis=1)             # tokens per (block, expert)
            run = tl.cumsum(cnt, axis=0) + carry[None, :]
            blk = b0 + tl.arange(0, NB)
            tl.store(incl_ptr + e[None, :].to(tl.int64) * n_blk + blk[:, None], run,
                     mask=(blk < n_blk)[:, None] & em[None, :])
            carry += tl.sum(cnt, axis=0)
        tl.store(tot_ptr + e, carry, mask=em)

    @triton.jit
    def _qb_hist_kernel(scores_ptr, alpha_ptr, pad_ptr, lw_ptr, hist_ptr, stride_st, stride_se, stride_a, stride_p,
                        T, E, num_bins,
                        BT: tl.constexpr, BE: tl.constexpr, HAS_PAD: tl.constexpr, I32: tl.constexpr,
                        RANGE: tl.constexpr, EP: tl.constexpr):
        # one program: BT tokens x BE experts of the scores
        t = tl.program_id(0) * BT + tl.arange(0, BT)
        e = tl.program_id(1) * BE + tl.arange(0, BE)
        tm = t < T
        em = e < E
        s = tl.load(scores_ptr + t[:, None] * stride_st + e[None, :] * stride_se, mask=tm[:, None] & em[None, :],
                    other=0.0)
        a = tl.load(alpha_ptr + t * stride_a, mask=tm, other=0.0)
        if RANGE:
            # lw_ptr = beta [E] (fp32): the bin range in compute_qb_histogram's arithmetic,
            # lower = -max(beta) - 1, upper = -min(beta) + 1, width = (upper - lower) / num_bins.
            # torch divides a device tensor by a Python number as a product with the fp32 reciprocal, so the
            # width is (upper - lower) * (1 / num_bins); a true division differs in the last bit for some ranges
            # (bench_moe_maps.py, job 536750).
            eb = tl.arange(0, EP)
            lower = -tl.max(tl.load(lw_ptr + eb, mask=eb < E, other=float("-inf")), axis=0) - 1.0
            upper = -tl.min(tl.load(lw_ptr + eb, mask=eb < E, other=float("inf")), axis=0) + 1.0
            width = (upper - lower) * (1.0 / num_bins.to(tl.float32))
        else:
            lower = tl.load(lw_ptr)
            width = tl.load(lw_ptr + 1)
        # compute_qb_histogram's fp32 arithmetic, in the same order: (alpha - scores - lower) / width, clamp, truncate
        b = ((a[:, None] - s) - lower) / width
        b = tl.minimum(tl.maximum(b, 0.0), (num_bins - 1).to(tl.float32))
        idx = b.to(tl.int32)
        keep = tm[:, None] & em[None, :]
        if HAS_PAD:
            pad = tl.load(pad_ptr + t * stride_p, mask=tm, other=1).to(tl.int32)
            keep = keep & (pad == 0)[:, None]
        dst = hist_ptr + e[None, :].to(tl.int64) * num_bins + idx
        if I32:
            tl.atomic_add(dst, tl.full((BT, BE), 1, tl.int32), mask=keep)
        else:
            tl.atomic_add(dst, tl.full((BT, BE), 1, tl.int64), mask=keep)

    @triton.jit
    def _chunk_layout_kernel(sizes_ptr, src_ptr, idx_ptr, R, E,
                             RP: tl.constexpr, EP: tl.constexpr, ALIGN: tl.constexpr, BM: tl.constexpr):
        # one program: the padded row segment of local expert e0
        e0 = tl.program_id(0)
        r = tl.arange(0, RP)
        e = tl.arange(0, EP)
        live = (r < R)[:, None] & (e < E)[None, :]
        s = tl.load(sizes_ptr + r[:, None] * E + e[None, :], mask=live, other=0).to(tl.int32)   # chunk sizes [R, E]
        tot = tl.sum(s, axis=0)                                   # tokens of each expert
        pad = ((tot + (ALIGN - 1)) // ALIGN) * ALIGN              # rows of each expert's padded segment
        seg0 = tl.sum(tl.where(e < e0, pad, 0))                   # first row of this expert's segment
        n_e = tl.sum(tl.where(e == e0, tot, 0))
        n_pad = tl.sum(tl.where(e == e0, pad, 0))
        col = tl.sum(tl.where((e == e0)[None, :], s, 0), axis=1)  # sizes of this expert's chunks, by source rank
        cum = tl.cumsum(col, axis=0)
        # first received row of chunk (r, e0): the sizes of all chunks before it in (rank, expert) order
        flat = tl.reshape(r[:, None] * E + e[None, :] + tl.zeros((RP, EP), tl.int32), (1, RP * EP))
        sf = tl.reshape(s, (1, RP * EP))
        key = r * E + e0
        start = tl.sum(tl.where(flat < key[:, None], sf, 0), axis=1)
        delta = start - (cum - col)                               # received row = position in the expert + delta
        for i in tl.range(0, n_pad, BM):
            j = i + tl.arange(0, BM)
            rr = tl.sum(tl.where(cum[None, :] <= j[:, None], 1, 0), axis=1)        # source rank of position j
            d = tl.sum(tl.where(r[None, :] == rr[:, None], delta[None, :], 0), axis=1)
            m = j < n_pad
            tl.store(src_ptr + seg0 + j, tl.where(j < n_e, j + d, -1), mask=m)
            tl.store(idx_ptr + seg0 + j, tl.full((BM,), 0, tl.int32) + e0, mask=m)

    _K.update(triton=triton, rim=_row_id_map_kernel, rimp=_rim_prefix_kernel, qb=_qb_hist_kernel,
              cl=_chunk_layout_kernel)
    return _K


_TE_MAKE_ROW_ID_MAP = None


def make_row_id_map(routing_map: torch.Tensor, num_tokens: int, num_experts: int, bt=None, warps=None,
                    prefix=None):
    """Transformer Engine's make_row_id_map (same arguments, same map in every entry the permute / unpermute
    kernels read): [num_tokens, 2 * num_experts + 1] int32; per token the destination rows of its routed experts
    in descending order, at offset num_experts the experts of those rows, last the number of routed experts."""
    bt = bt or RIM_BT
    if num_tokens % bt != 0 or routing_map.dim() != 2 or num_tokens == 0:
        return _TE_MAKE_ROW_ID_MAP(routing_map, num_tokens, num_experts)
    K = _kernels()
    n_blk = num_tokens // bt
    dev = routing_map.device
    out = torch.empty((num_tokens, num_experts * 2 + 1), dtype=torch.int32, device=dev)
    local = RIM_PREFIX if prefix is None else prefix
    if local:
        # running counts per expert from one kernel (its programs walk the token blocks in order)
        incl = torch.empty(num_experts * n_blk, dtype=torch.int32, device=dev)
        tot = torch.empty(num_experts, dtype=torch.int32, device=dev)
        K["rimp"][(K["triton"].cdiv(num_experts, RIM_EB),)](
            routing_map, incl, tot, routing_map.stride(0), routing_map.stride(1), n_blk, num_experts,
            BT=bt, EB=RIM_EB, NB=RIM_NB, num_warps=RIM_PWARPS)
    else:
        # routed tokens per (block, expert) -> running total in (expert, block) order
        cnt = routing_map.reshape(n_blk, bt, num_experts).sum(dim=1, dtype=torch.int32)
        incl = cnt.t().reshape(-1).cumsum(0)                                   # int64 [num_experts * n_blk]
        tot = incl
    K["rim"][(n_blk,)](
        routing_map, incl, tot, out, routing_map.stride(0), routing_map.stride(1), out.stride(0), out.stride(1),
        n_blk, E=num_experts, EP=K["triton"].next_power_of_2(num_experts), BT=bt, LOCAL=local,
        num_warps=warps or RIM_WARPS)
    return out


def install():
    """Replace Transformer Engine's make_row_id_map (its permutation code calls it through the module)."""
    global _TE_MAKE_ROW_ID_MAP
    import transformer_engine.pytorch.triton.permutation as tp

    if _TE_MAKE_ROW_ID_MAP is None:
        _TE_MAKE_ROW_ID_MAP = tp.make_row_id_map
    tp.make_row_id_map = make_row_id_map


def uninstall():
    import transformer_engine.pytorch.triton.permutation as tp

    if _TE_MAKE_ROW_ID_MAP is not None:
        tp.make_row_id_map = _TE_MAKE_ROW_ID_MAP


def te_make_row_id_map():
    """Transformer Engine's own function (also after install())."""
    import transformer_engine.pytorch.triton.permutation as tp

    return _TE_MAKE_ROW_ID_MAP or tp.make_row_id_map


_TP = None


def _te_perm():
    global _TP
    if _TP is None:
        import transformer_engine.pytorch.triton.permutation as tp

        _TP = tp
    return _TP


class _PermuteWithProbs(torch.autograd.Function):
    """transformer_engine.pytorch.permutation.moe_permute_with_probs (mask map, no FP8, no padding): the same
    Triton functions in the same order as TE's operators te_moe::permute_mask_map_fwd / _bwd."""

    @staticmethod
    def forward(ctx, inp, probs, routing_map, num_out_tokens):
        tp = _te_perm()
        num_tokens, hidden = inp.shape
        num_experts = routing_map.size(1)
        row_id_map = tp.make_row_id_map(routing_map, num_tokens, num_experts)
        out, _, permuted_probs = tp.permute_with_mask_map(
            inp, row_id_map, probs, None, None, num_tokens, num_experts, num_out_tokens, hidden, None)
        ctx.save_for_backward(row_id_map)
        ctx.dims = (num_tokens, num_experts, hidden)
        ctx.mark_non_differentiable(row_id_map)
        return out, permuted_probs, row_id_map

    @staticmethod
    def backward(ctx, grad_out, grad_probs, _grad_map):
        (row_id_map,) = ctx.saved_tensors
        num_tokens, num_experts, hidden = ctx.dims
        act_grad, probs_grad = _te_perm().unpermute_with_mask_map(
            grad_out, row_id_map, None, grad_probs if ctx.needs_input_grad[1] else None, None,
            num_tokens, num_experts, hidden)
        return act_grad, probs_grad, None, None


class _Unpermute(torch.autograd.Function):
    """transformer_engine.pytorch.permutation.moe_unpermute (mask map, no merging probs, no FP8, no padding):
    TE's operators te_moe::unpermute_mask_map_fwd / _bwd_no_probs."""

    @staticmethod
    def forward(ctx, inp, row_id_map, num_tokens, hidden):
        num_experts = (row_id_map.size(1) - 1) // 2
        out, _ = _te_perm().unpermute_with_mask_map(inp, row_id_map, None, None, None, num_tokens, num_experts, hidden)
        ctx.save_for_backward(row_id_map)
        ctx.dims = (num_tokens, num_experts, inp.size(0), hidden)
        return out

    @staticmethod
    def backward(ctx, grad):
        (row_id_map,) = ctx.saved_tensors
        num_tokens, num_experts, num_permuted, hidden = ctx.dims
        act_grad, _, _ = _te_perm().permute_with_mask_map(
            grad, row_id_map, None, None, None, num_tokens, num_experts, num_permuted, hidden, None)
        return act_grad, None, None, None


def direct_ok(x):
    """Inputs the direct functions take: a plain (not quantized) non-empty device tensor."""
    return type(x) is torch.Tensor and x.is_cuda and x.numel() > 0


def permute_with_probs(inp, probs, routing_map, num_out_tokens):
    """(permuted tokens, permuted probs, row_id_map), as TE's moe_permute_with_probs."""
    return _PermuteWithProbs.apply(inp, probs, routing_map, num_out_tokens)


def unpermute(inp, row_id_map, restore_shape):
    """Tokens back in their original order (rows of the same token summed), as TE's moe_unpermute without
    merging probs."""
    num_tokens, hidden = restore_shape
    return _Unpermute.apply(inp, row_id_map, num_tokens, hidden)


def qb_histogram(scores, alpha, beta, num_bins, padding_mask=None, bt=None, be=None, warps=None, i32=None,
                 in_kernel_range=None):
    """moe_utils.compute_qb_histogram in one kernel: [num_experts, num_bins] int64 counts of the bin of
    alpha_i - scores_ij over the tokens i (rows with padding_mask True are left out). scores fp32 [T, E]."""
    K = _kernels()
    triton = K["triton"]
    bt, be = bt or QB_BT, be or QB_BE
    i32 = QB_I32 if i32 is None else i32
    T, E = scores.shape
    rng = QB_RANGE if in_kernel_range is None else in_kernel_range
    rng = rng and beta.dim() == 1 and beta.is_contiguous() and beta.numel() == E
    if rng:
        lw = beta          # the kernel takes the bin range from beta itself
    else:
        lower = -beta.max() - 1.0
        upper = -beta.min() + 1.0
        width = (upper - lower) / num_bins
        lw = torch.stack((lower, width)).to(torch.float32)
    hist = torch.zeros(E * num_bins, dtype=torch.int32 if i32 else torch.int64, device=scores.device)
    pad = padding_mask if padding_mask is not None else alpha
    K["qb"][(triton.cdiv(T, bt), triton.cdiv(E, be))](
        scores, alpha, pad, lw, hist, scores.stride(0), scores.stride(1), alpha.stride(0), pad.stride(0),
        T, E, num_bins,
        BT=bt, BE=be, HAS_PAD=padding_mask is not None, I32=i32, RANGE=rng, EP=triton.next_power_of_2(E),
        num_warps=warps or QB_WARPS)
    return (hist.to(torch.int64) if i32 else hist).reshape(E, num_bins)


def chunk_layout(sizes, n_ranks, m_splits, device, align=128):
    """Row maps of the fused MXFP4 routed MLP for tokens received as chunks ordered by (source rank, local
    expert), in one kernel (mxfp4_alto: _layout + chunk_perm + _src_rows, about 25 torch operations).
    sizes [n_ranks * E] (device tensor): chunk lengths in the received order; m_splits: tokens per local expert
    (host ints). Returns (src [Tp] int32: received row of every row of the 128-aligned grouped layout, -1 on
    padding rows; idx [Tp] int32: local expert of every grouped row; Tp)."""
    K = _kernels()
    triton = K["triton"]
    E = len(m_splits)
    tp = sum(-(-m // align) * align for m in m_splits)
    src = torch.empty(tp, dtype=torch.int32, device=device)
    idx = torch.empty(tp, dtype=torch.int32, device=device)
    if tp:
        K["cl"][(E,)](sizes, src, idx, n_ranks, E, RP=triton.next_power_of_2(n_ranks),
                      EP=triton.next_power_of_2(E), ALIGN=align, BM=1024, num_warps=4)
    return src, idx, tp
