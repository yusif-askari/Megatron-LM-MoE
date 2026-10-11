"""NORM_FREEZE=<substr,...> (experiment): keep every parameter whose name contains one of the
substrings at the value it has when training (re)starts, e.g. the norm weights in front of the
MXFP4 GEMMs after a bf16 phase.

The parameters stay in the model and in the optimizer (so a checkpoint and its optimizer state
load unchanged); after every optimizer step both the bf16 model copy and, on the rank that owns
it, the fp32 main copy are put back to the saved value. The forward pass therefore always sees
the frozen value and the saved checkpoints hold it.
"""
import os

import torch

PATTERNS = tuple(p for p in os.environ.get("NORM_FREEZE", "").split(",") if p)
ENABLED = bool(PATTERNS)
_FROZEN = []        # (model param, saved bf16 value, [fp32 main copies], saved fp32 value)


def _main_copies(optimizer, ids):
    """fp32 main copies of the given model params, from any (chained) Megatron optimizer."""
    found = {}
    stack = [optimizer]
    while stack:
        opt = stack.pop()
        stack.extend(getattr(opt, "chained_optimizers", []))
        for model_group, main_group in zip(getattr(opt, "float16_groups", []),
                                           getattr(opt, "fp32_from_float16_groups", [])):
            for p, m in zip(model_group, main_group):
                if id(p) in ids:
                    found.setdefault(id(p), []).append(m)
    return found


def setup(model, optimizer):
    """Record the values to keep. Call once, after the checkpoint is loaded."""
    params = [(n, p) for chunk in model for n, p in chunk.named_parameters()
              if any(s in n for s in PATTERNS)]
    unmatched = [s for s in PATTERNS if not any(s in n for n, _ in params)]
    if unmatched:
        raise ValueError(f"NORM_FREEZE: no parameter matches {unmatched}")
    mains = _main_copies(optimizer, {id(p) for _, p in params})
    for _, p in params:
        ms = mains.get(id(p), [])
        _FROZEN.append((p, p.detach().clone(), ms, p.detach().float().clone()))
    if torch.distributed.get_rank() == 0:
        names = sorted({n.split("layers.")[-1].split(".", 1)[-1] for n, _ in params})
        print(f"[NORM_FREEZE] {len(params)} parameters frozen ({', '.join(names)}); "
              f"rank 0 holds {sum(len(m) for m in mains.values())} fp32 main copies", flush=True)


@torch.no_grad()
def restore():
    """After optimizer.step: put the frozen values back."""
    for p, saved, mains, saved32 in _FROZEN:
        p.copy_(saved)
        for m in mains:
            m.copy_(saved32)
