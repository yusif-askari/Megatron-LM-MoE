# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Output layer + cross entropy without materialising the full logits (training, TP = 1).

The unfused path computes logits = h · Wᵀ for all N tokens of a micro-batch ([N, V] bf16; with
N = 16384 and V = 200064 that is 6.6 GB). The vocab-parallel cross entropy then upcasts them to
fp32 and makes several passes over them, and the backward writes the logits gradient ([N, V]
bf16) for the input-gradient and weight-gradient GEMMs.

Here the tokens are processed in chunks of CHUNK rows. Per chunk:
  logits_c = h_c · Wᵀ      bf16 GEMM into one [CHUNK, V] buffer that every chunk reuses
  one Triton kernel, one program per row (3 reads of the row, 1 write, in place):
      m = max_j l_j,  e_j = exp(l_j - m),  S = Σ e_j,  S_o = Σ_{j≠t} e_j          (fp32)
      loss = softplus(log S_o - (l_t - m))     (as fused_vocab_parallel_cross_entropy)
      grad_j = e_j / S (j ≠ t),  grad_t = -S_o / S,  times loss_mask   -> written over logits_c
  dh_c = grad_c · W        bf16 GEMM
  dW_buf += grad_cᵀ · h_c  fp32 accumulation (wgrad_gemm_accum_fp32 when available)

The gradient is computed in the forward. This is valid because the training loss reduces the
per-token losses as Σ loss_i · mask_i and then applies only scalar factors, so the gradient that
arrives at loss_i in the backward is mask_i · c with one scalar c per micro-batch. The backward
computes c = Σ g / Σ mask (on the device, no synchronisation), returns c · dh and adds c · dW_buf
into W.main_grad (gradient-accumulation fusion) or returns it as the gradient of W. A loss that
weights tokens differently must not use this path; FUSED_LINEAR_CE_CHECK=1 asserts g == mask · c
in the backward (this synchronises; for tests).

Same arithmetic as the unfused path up to the reduction order over the vocabulary and the place
where c is applied (after the bf16 GEMMs instead of before).

Switch: MEGATRON_FUSED_LINEAR_CE=1 (off by default). GPTModel uses the fused path only where it
computes the same thing, see GPTModel._postprocess. FUSED_LINEAR_CE_CHUNK sets the chunk size
(default 4096 rows).
"""
import os

import torch

CHUNK = int(os.environ.get("FUSED_LINEAR_CE_CHUNK", "4096"))
_CHECK = os.environ.get("FUSED_LINEAR_CE_CHECK", "0") == "1"
_KERNEL = None


def enabled() -> bool:
    """True if MEGATRON_FUSED_LINEAR_CE=1."""
    return os.environ.get("MEGATRON_FUSED_LINEAR_CE", "0") == "1"


def _ce_rows_kernel():
    """The Triton kernel, built on first use (Triton is imported only when the fused path runs)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def ce_rows(logits_ptr, target_ptr, mask_ptr, loss_ptr, stride_row, V, BLOCK_V: tl.constexpr):
        row = tl.program_id(0)
        base = logits_ptr + row.to(tl.int64) * stride_row
        t = tl.load(target_ptr + row).to(tl.int32)
        w = tl.load(mask_ptr + row).to(tl.float32)
        offs = tl.arange(0, BLOCK_V)
        # pass 1: row maximum (fp32 of the stored logits)
        mx = tl.full((BLOCK_V,), float("-inf"), tl.float32)
        for v0 in range(0, V, BLOCK_V):
            x = tl.load(base + v0 + offs, mask=v0 + offs < V, other=float("-inf")).to(tl.float32)
            mx = tl.maximum(mx, x)
        m = tl.max(mx, axis=0)
        # pass 2: sums of exp(l - m), over all entries and over all entries but the target
        s_all = tl.zeros((BLOCK_V,), tl.float32)
        s_oth = tl.zeros((BLOCK_V,), tl.float32)
        for v0 in range(0, V, BLOCK_V):
            idx = v0 + offs
            x = tl.load(base + idx, mask=idx < V, other=float("-inf")).to(tl.float32)
            e = tl.exp(x - m)
            s_all += e
            s_oth += tl.where(idx == t, 0.0, e)
        S = tl.sum(s_all, axis=0)
        So = tl.sum(s_oth, axis=0)
        lt = tl.load(base + t).to(tl.float32)
        z = tl.log(So) - (lt - m)
        loss = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))  # softplus (torch: threshold 20)
        tl.store(loss_ptr + row, loss)
        # pass 3: gradient (softmax, target entry -S_o / S) times the loss mask, over the logits
        inv = 1.0 / S
        tgrad = -So * inv
        for v0 in range(0, V, BLOCK_V):
            idx = v0 + offs
            x = tl.load(base + idx, mask=idx < V, other=0.0).to(tl.float32)
            gr = tl.where(idx == t, tgrad, tl.exp(x - m) * inv) * w
            tl.store(base + idx, gr.to(logits_ptr.dtype.element_ty), mask=idx < V)

    _KERNEL = ce_rows
    return _KERNEL


def _wgrad_accum(h, d, buf):
    """buf (fp32 [V, H]) += dᵀ · h, as the output layer accumulates its weight gradient."""
    try:
        import fused_weight_gradient_mlp_cuda
    except ImportError:
        buf.add_(torch.matmul(d.t(), h).float())
        return
    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(h, d, buf)


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, target, mask):
        # h [N, H], weight [V, H] (same dtype), target [N] integer, mask [N]
        N, H = h.shape
        V = weight.shape[0]
        h = h.contiguous()
        target = target.contiguous()
        maskf = mask.to(torch.float32).contiguous()
        loss = torch.empty(N, dtype=torch.float32, device=h.device)
        dh = torch.empty_like(h)
        dw = torch.zeros((V, H), dtype=torch.float32, device=h.device)
        C = max(1, min(CHUNK, N))
        buf = torch.empty((C, V), dtype=h.dtype, device=h.device)
        kernel = _ce_rows_kernel()
        for s in range(0, N, C):
            e = min(s + C, N)
            lc = buf[: e - s]
            torch.matmul(h[s:e], weight.t(), out=lc)
            kernel[(e - s,)](
                lc, target[s:e], maskf[s:e], loss[s:e], lc.stride(0), V, BLOCK_V=4096, num_warps=8
            )
            torch.matmul(lc, weight, out=dh[s:e])
            _wgrad_accum(h[s:e], lc, dw)
        ctx.save_for_backward(dh, dw, maskf)
        ctx.weight = weight
        return loss

    @staticmethod
    def backward(ctx, g):
        dh, dw, maskf = ctx.saved_tensors
        weight = ctx.weight
        g = g.reshape(-1).float()
        c = g.sum() / maskf.sum().clamp_min(1.0)  # 0-dim device tensor, no synchronisation
        if _CHECK:
            assert torch.allclose(
                g, maskf * c, rtol=1e-6, atol=0.0
            ), "loss gradient is not mask * c"
        dx = (dh.float() * c).to(dh.dtype)
        dw.mul_(c)
        if hasattr(weight, "main_grad") and hasattr(weight, "grad_added_to_main_grad"):
            # As LinearWithGradAccumulationAndAsyncCommunication: the gradient goes into main_grad
            # and the returned .grad is a placeholder that DistributedDataParallel's hook skips.
            # With weights shared with the embedding (zero_out_wgrad) the hook does add .grad, so
            # the placeholder must be zero.
            weight.main_grad.add_(dw)
            weight.grad_added_to_main_grad = True
            if getattr(weight, "zero_out_wgrad", False):
                dweight = torch.zeros_like(weight)
            else:
                dweight = torch.empty_like(weight)
        else:
            dweight = dw.to(weight.dtype)
        return dx, dweight, None, None


def fused_linear_cross_entropy(hidden, weight, labels, loss_mask):
    """Per-token cross entropy of logits = hidden · weightᵀ without materialising all logits.

    Args:
        hidden (Tensor): [s, b, H], the input of the output layer.
        weight (Tensor): [V, H], the weight of the output layer (no bias).
        labels (Tensor): [b, s], target token ids.
        loss_mask (Tensor): [b, s], the mask the training loss multiplies the per-token losses with.

    Returns:
        Tensor: the loss [b, s] in fp32, as LanguageModule.compute_language_model_loss returns it.
    """
    s, b, H = hidden.shape
    tgt = labels.transpose(0, 1).reshape(-1)
    msk = loss_mask.transpose(0, 1).reshape(-1)
    loss = _FusedLinearCrossEntropy.apply(hidden.reshape(s * b, H), weight, tgt, msk)
    return loss.view(s, b).transpose(0, 1).contiguous()
