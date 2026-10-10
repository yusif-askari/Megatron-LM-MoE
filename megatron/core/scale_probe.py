"""SCALE_PROBE=<dir> (debug): is the MXFP4 "grow" push on norm weights a bias of the
straight-through gradient, or does the FP4 loss itself prefer larger norm weights?

At the first training step, on a fixed set of cached batches and the loaded weights:
  1. finite differences: scale a group of norm weights by lambda and measure the actual loss,
     forward only, once with MXFP4 on (the loss the network trains on) and once with
     quantization off (bf16 reference);
  2. the gradient prediction: one forward/backward at lambda = 1, in both modes, giving
     dL/dlog(lambda) = sum_groups <gamma, dL/dgamma>.
The bf16 pair (finite difference vs gradient) calibrates the gradient units, since both are
the true derivative there. The FP4 gradient (straight-through) is then compared with the FP4
finite difference (the true derivative of the FP4 loss). Writes probe.json and stops the job.

Groups: pre_mlp (norm in front of the MoE), fp4_path (every norm around an FP4 GEMM),
qk_control (q/k norms, which only feed bf16 attention math).
"""
import json
import math
import os

import torch

PROBE_DIR = os.environ.get("SCALE_PROBE", "")
N_BATCHES = int(os.environ.get("SCALE_PROBE_BATCHES", "4"))
LAMBDAS = (0.9, 0.95, 1.05, 1.1)
GROUPS = {
    "pre_mlp": ("pre_mlp_layernorm.weight",),
    "fp4_path": ("pre_mlp_layernorm.weight", "post_mlp_layernorm.weight",
                 "post_self_attn_layernorm.weight", "in_proj.layer_norm_weight",
                 "linear_qkv.layer_norm_weight", "mlp.linear_fc1.layer_norm_weight",
                 "self_attention.out_norm.weight"),
    "qk_control": ("q_layernorm.weight", "k_layernorm.weight"),
}


def _params(model, suffixes):
    return [p for chunk in model for n, p in chunk.named_parameters()
            if n.endswith(suffixes) and ".decoder." in f".{n}"]


def run(forward_step_func, data_iterator, model, forward_backward_func, num_microbatches,
        seq_length, micro_batch_size, decoder_seq_length):
    from megatron.core import mxfp4_emulation, parallel_state

    rank = torch.distributed.get_rank()
    dp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    items = [next(data_iterator) for _ in range(N_BATCHES * num_microbatches)]
    buffers = [(b, b.detach().clone()) for chunk in model for b in chunk.buffers()]

    def passes(forward_only):
        """Mean loss over all cached batches, in ONE schedule call (so the gradient all-reduce
        runs once), with the same token-weighted reduction as train_step."""
        for buf, saved in buffers:              # router balancing state etc.: identical every pass
            buf.data.copy_(saved)
        mxfp4_emulation.new_step()              # drop the per-step quantized-weight cache
        ctx = torch.no_grad() if forward_only else torch.enable_grad()
        with ctx:
            losses = forward_backward_func(
                forward_step_func=forward_step_func, data_iterator=iter(items), model=model,
                num_microbatches=len(items), seq_length=seq_length,
                micro_batch_size=micro_batch_size, decoder_seq_length=decoder_seq_length,
                forward_only=forward_only)
        val = torch.vstack([x["lm loss"].view(-1) for x in losses]).sum(0).float()
        torch.distributed.all_reduce(val, group=dp_group)
        return float(val[0] / val[1])

    out = {"n_batches": N_BATCHES, "lambdas": LAMBDAS, "modes": {}}
    for mode in ("bf16", "mxfp4"):
        mxfp4_emulation._RULE_OVERRIDE = "none" if mode == "bf16" else None
        res = {"loss": {}, "grad": {}}
        base = passes(True)
        for g, suf in GROUPS.items():
            ps = _params(model, suf)
            saved = [p.detach().clone() for p in ps]
            curve = {1.0: base}
            for lam in LAMBDAS:
                for p, s in zip(ps, saved):
                    p.data.copy_(s * lam)
                curve[lam] = passes(True)
            for p, s in zip(ps, saved):
                p.data.copy_(s)
            fd = {f"{a}-{b}": (curve[b] - curve[a]) / (math.log(b) - math.log(a))
                  for a, b in ((0.95, 1.05), (0.9, 1.1))}
            res["loss"][g] = {"curve": {str(k): v for k, v in sorted(curve.items())}, "fd_slope": fd,
                              "n_params": len(ps)}
        for chunk in model:
            chunk.zero_grad_buffer()
        passes(False)                           # forward/backward at lambda = 1, grads reduced
        for g, suf in GROUPS.items():
            s = 0.0
            for p in _params(model, suf):
                gr = getattr(p, "main_grad", None)
                gr = p.grad if gr is None else gr
                s += float((p.detach().float() * gr.float().view_as(p)).sum())
            res["grad"][g] = s
        for chunk in model:
            chunk.zero_grad_buffer()
        out["modes"][mode] = res
    mxfp4_emulation._RULE_OVERRIDE = None

    # Calibrate the gradient units once: in bf16 the finite difference and the gradient are the
    # same derivative. The normalization (microbatch / data-parallel averaging) is the same for
    # every parameter, so one constant from the largest group serves all groups.
    fd_ref = out["modes"]["bf16"]["loss"]["fp4_path"]["fd_slope"]["0.95-1.05"]
    gd_ref = out["modes"]["bf16"]["grad"]["fp4_path"]
    c = fd_ref / gd_ref if gd_ref else float("nan")
    summ = {"calib": c}
    for g in GROUPS:
        summ[g] = {"bf16_fd": out["modes"]["bf16"]["loss"][g]["fd_slope"]["0.95-1.05"],
                   "bf16_grad": c * out["modes"]["bf16"]["grad"][g],
                   "mxfp4_fd": out["modes"]["mxfp4"]["loss"][g]["fd_slope"]["0.95-1.05"],
                   "mxfp4_fd_wide": out["modes"]["mxfp4"]["loss"][g]["fd_slope"]["0.9-1.1"],
                   "mxfp4_ste_grad": c * out["modes"]["mxfp4"]["grad"][g]}
    out["summary"] = summ
    if rank == 0:
        os.makedirs(PROBE_DIR, exist_ok=True)
        json.dump(out, open(os.path.join(PROBE_DIR, "probe.json"), "w"), indent=1)
        print("[SCALE_PROBE] " + json.dumps(summ), flush=True)
    torch.distributed.barrier()
    os._exit(0)
