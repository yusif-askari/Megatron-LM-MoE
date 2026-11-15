"""Output layer + cross entropy without materialising the full logits (training, TP = 1).

Megatron computes logits = h · Wᵀ for all N tokens of a micro-batch ([N, V] bf16; N = 16384, V = 200064:
6.6 GB), then the fused vocab-parallel cross entropy upcasts them to fp32 and makes several passes over them,
and the backward writes the logits gradient ([N, V] bf16) for the dgrad / wgrad GEMMs.

Here the tokens are processed in chunks of CHUNK rows. Per chunk:
  logits_c = h_c · Wᵀ                     bf16 GEMM, [C, V] (one chunk buffer, reused)
  one Triton pass per row (3 reads of the row, the later two from cache, 1 write, in place):
      m = max_j l_j,  e_j = exp(l_j - m),  S = Σ e_j,  S_o = Σ_{j≠t} e_j          (fp32)
      loss = softplus(log S_o - (l_t - m))                                         (Megatron's formula)
      grad_j = e_j / S (j ≠ t),  grad_t = -S_o / S,  times loss_mask              -> bf16, over logits_c
  dh_c = grad_c · W                        bf16 GEMM
  dW_buf += grad_cᵀ · h_c                  fp32 accumulation (Megatron's wgrad_gemm_accum_fp32)
The gradient is computed in the forward: Megatron's loss_func reduces the per-token losses as
Σ loss_i · mask_i and then applies only scalar factors, so the gradient arriving at loss_i is mask_i · c with
one scalar c per micro-batch. The backward computes c = Σ g / Σ mask (on the device, no sync) and returns
c · dh, adding c · dW_buf into W.main_grad (gradient-accumulation fusion) or returning it as W's grad.
FUSED_LINEAR_CE_CHECK=1 verifies g == mask · c in the backward (synchronises; for testing).

Same arithmetic as Megatron's path up to reduction order (max / sums over the vocabulary in a different
order) and the place where c is applied (after the bf16 GEMMs instead of before).
"""
import os

import torch
import triton
import triton.language as tl

CHUNK = int(os.environ.get("FUSED_LINEAR_CE_CHUNK", "4096"))
_CHECK = os.environ.get("FUSED_LINEAR_CE_CHECK", "0") == "1"


@triton.jit
def _ce_rows_kernel(logits_ptr, target_ptr, mask_ptr, loss_ptr, stride_row, V,
                    BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride_row
    t = tl.load(target_ptr + row).to(tl.int32)
    w = tl.load(mask_ptr + row).to(tl.float32)
    offs = tl.arange(0, BLOCK_V)
    # pass 1: row max (fp32 of the bf16 logits)
    mx = tl.full((BLOCK_V,), float("-inf"), tl.float32)
    for v0 in range(0, V, BLOCK_V):
        x = tl.load(base + v0 + offs, mask=v0 + offs < V, other=float("-inf")).to(tl.float32)
        mx = tl.maximum(mx, x)
    m = tl.max(mx, axis=0)
    # pass 2: sums of exp(l - m), all and excluding the target
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
    loss = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(z)))        # softplus (torch: threshold 20)
    tl.store(loss_ptr + row, loss)
    # pass 3: gradient (softmax, target entry -S_o/S) times the loss mask, in place as bf16
    inv = 1.0 / S
    tgrad = -So * inv
    for v0 in range(0, V, BLOCK_V):
        idx = v0 + offs
        x = tl.load(base + idx, mask=idx < V, other=0.0).to(tl.float32)
        gr = tl.where(idx == t, tgrad, tl.exp(x - m) * inv) * w
        tl.store(base + idx, gr.to(logits_ptr.dtype.element_ty), mask=idx < V)


def _wgrad_accum(h, d, buf):
    """buf (fp32 [V, H]) += dᵀ · h, as Megatron's output layer accumulates its weight gradient."""
    try:
        import fused_weight_gradient_mlp_cuda

        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(h, d, buf)
    except ImportError:
        buf.add_(torch.matmul(d.t(), h).float())


class _FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, weight, target, mask):
        # h [N, H] bf16, weight [V, H], target [N] int, mask [N]
        N, H = h.shape
        V = weight.shape[0]
        h = h.contiguous()
        target = target.contiguous()
        maskf = mask.to(torch.float32).contiguous()
        loss = torch.empty(N, dtype=torch.float32, device=h.device)
        dh = torch.empty_like(h)
        dw = torch.zeros((V, H), dtype=torch.float32, device=h.device)
        C = min(CHUNK, N)
        buf = torch.empty((C, V), dtype=h.dtype, device=h.device)
        for s in range(0, N, C):
            e = min(s + C, N)
            lc = buf[: e - s]
            torch.matmul(h[s:e], weight.t(), out=lc)
            _ce_rows_kernel[(e - s,)](lc, target[s:e], maskf[s:e], loss[s:e], lc.stride(0), V,
                                      BLOCK_V=4096, num_warps=8)
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
        c = g.sum() / maskf.sum().clamp_min(1.0)               # 0-dim device tensor, no sync
        if _CHECK:
            assert torch.allclose(g, maskf * c, rtol=1e-6, atol=0.0), "loss gradient is not mask * c"
        dx = (dh.float() * c).to(dh.dtype)
        dw.mul_(c)
        if hasattr(weight, "main_grad") and getattr(weight, "grad_added_to_main_grad", None) is not None:
            weight.main_grad.add_(dw)
            weight.grad_added_to_main_grad = True
            dweight = torch.empty_like(weight)                   # placeholder (DDP skips it)
        else:
            dweight = dw.to(weight.dtype)
        return dx, dweight, None, None


def fused_linear_cross_entropy(hidden, weight, labels, loss_mask):
    """Per-token cross entropy of logits = hidden · weightᵀ without materialising all logits.
    hidden [s, b, H], labels / loss_mask [b, s]. Returns the loss [b, s] fp32 (as
    LanguageModule.compute_language_model_loss)."""
    s, b, H = hidden.shape
    tgt = labels.transpose(0, 1).reshape(-1)
    msk = loss_mask.transpose(0, 1).reshape(-1)
    loss = _FusedLinearCrossEntropy.apply(hidden.reshape(s * b, H), weight, tgt, msk)
    return loss.view(s, b).transpose(0, 1).contiguous()


def enabled() -> bool:
    return os.environ.get("MEGATRON_FUSED_LINEAR_CE", "0") == "1"
