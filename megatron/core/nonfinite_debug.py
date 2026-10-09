"""Debug-only tracer for the first inf/NaN in the forward pass (NONFINITE_CHECK=1).

Every module in the decoder records, without a host sync, whether its output tensors contain
a non-finite value. Each MoE router (before the token dispatch, where bad routing would crash
with an index-out-of-bounds assert) and the end of each model chunk's forward sync once, and on
the first hit print, per rank:
  - the iteration and the first modules (forward order) whose output was non-finite,
  - router outputs that are malformed (non-finite probs, a token not routed to exactly top-k),
  - every parameter that is itself non-finite,
then raise, so the run stops at the cause instead of in a later kernel.
"""
import os

import torch

ENABLED = os.environ.get("NONFINITE_CHECK", "0") == "1"
_FLAGS = []  # (module name, device bool tensor) since the last check


def _tensors(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


def _record(name):
    def hook(module, inputs, out):
        for t in _tensors(out):
            if t.is_floating_point() and t.numel():
                _FLAGS.append((name, ~torch.isfinite(t).all()))
    return hook


def _iteration():
    try:
        from megatron.training.global_vars import get_args
        return getattr(get_args(), "curr_iteration", "?")
    except Exception:
        return "?"


def _fail(where, chunk, extra=""):
    bad = [n for n, f in _FLAGS if f.item()]
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    params = [n for n, p in list(chunk.named_parameters()) + list(chunk.named_buffers())
              if p.is_floating_point() and not torch.isfinite(p).all().item()]
    msg = (f"[NONFINITE] rank {rank} iteration {_iteration()} at {where}: "
           f"{len(bad)} module outputs non-finite; first in forward order: {bad[:12]}; "
           f"non-finite parameters/buffers ({len(params)}): {params[:12]}{extra}")
    print(msg, flush=True)
    raise RuntimeError(msg)


def _check(where, chunk, extra_bad=None):
    flags = [f for _, f in _FLAGS]
    hit = torch.stack(flags).any() if flags else torch.zeros((), dtype=torch.bool)
    if extra_bad is not None:
        hit = hit | extra_bad[0]
    if hit.item():
        _fail(where, chunk, extra_bad[1]() if extra_bad is not None and extra_bad[0].item() else "")
    _FLAGS.clear()


def install(chunk):
    """Register the hooks on one model chunk; returns the number of hooked modules."""
    from megatron.core.transformer.moe.router import TopKRouter

    n = 0
    for name, m in chunk.named_modules():
        if ".decoder." not in f".{name}." and not name.endswith("output_layer"):
            continue
        m.register_forward_hook(_record(name))
        n += 1
        if isinstance(m, TopKRouter):
            def router_pre_hook(module, inputs, name=name):
                # before routing: a bad index from top-k would assert inside the router itself
                x = inputs[0]
                beta = getattr(module, "qb_beta", None)
                bad = ~torch.isfinite(x).all()
                if isinstance(beta, torch.Tensor):
                    bad = bad | ~torch.isfinite(beta).all()

                def detail():
                    return (f"; router input {name}: finite={torch.isfinite(x).all().item()} "
                            f"absmax={x.float().abs().nan_to_num(posinf=-1).max().item():.3g}; "
                            f"qb_beta finite="
                            f"{torch.isfinite(beta).all().item() if isinstance(beta, torch.Tensor) else 'n/a'}")

                _check(f"before router {name}", chunk, (bad, detail))
            m.register_forward_pre_hook(router_pre_hook)

            def router_hook(module, inputs, out, name=name):
                probs, routing_map = out[0], out[1]
                per_token = routing_map.sum(-1)
                k = module.topk
                bad = (~torch.isfinite(probs).all()) | (per_token != k).any()

                def detail():
                    x = inputs[0]
                    return (f"; router {name}: input finite={torch.isfinite(x).all().item()} "
                            f"absmax={x.float().abs().nan_to_num(posinf=-1).max().item():.3g}, "
                            f"probs finite={torch.isfinite(probs).all().item()}, tokens with "
                            f"!= {k} experts: {(per_token != k).sum().item()}/{per_token.numel()}")

                _check(f"router {name}", chunk, (bad, detail))
            m.register_forward_hook(router_hook)
    chunk.register_forward_hook(lambda module, inputs, out: _check("end of forward", chunk))
    return n
