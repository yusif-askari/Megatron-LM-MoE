"""NORM_FIXED_RMS=<substr,...> (experiment): norm weights with a fixed size.

Every parameter v whose name contains one of the substrings is used in the forward pass as

    gamma_eff = r0 * v / rms(v),    r0 = rms(v) when training (re)starts,

so rms(gamma_eff) = r0 always while its direction (the balance between channels) is learned.
The gradient reaching v is orthogonal to v: the "grow" push of the MXFP4 straight-through
gradient, which acts along v, is removed before the optimizer sees it.

The module's own forward runs unchanged with gamma_eff swapped in for v (TE RMSNorm, the fused
input norm of the MXFP4-emulated LayerNormLinear, FLA's gated RMSNorm alike). v stays the stored
parameter, so checkpoints and optimizer state keep their layout. r0 is written to
<save>/norm_rms_targets.json and read back on a resume.
"""
import json
import os

import torch
from torch.nn.utils.stateless import _reparametrize_module

PATTERNS = tuple(p for p in os.environ.get("NORM_FIXED_RMS", "").split(",") if p)
ENABLED = bool(PATTERNS)
_DONE = False


def effective(v: torch.Tensor, r0: float) -> torch.Tensor:
    vf = v.float()
    return (vf * (r0 * torch.rsqrt(vf.pow(2).mean()))).to(v.dtype)


def _wrap(module, pname, r0):
    orig = module.forward

    def forward(*args, **kwargs):
        v = getattr(module, pname)
        with _reparametrize_module(module, {pname: effective(v, r0)}):
            return orig(*args, **kwargs)

    module.forward = forward


def setup(model, save_dir=None):
    """Wrap the matching norms. Call once, after the checkpoint is loaded."""
    global _DONE
    path = os.path.join(save_dir, "norm_rms_targets.json") if save_dir else None
    saved = json.load(open(path)) if path and os.path.exists(path) else {}
    targets, unmatched = {}, set(PATTERNS)
    for chunk in model:
        for mname, module in chunk.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                full = f"{mname}.{pname}"
                hits = [s for s in PATTERNS if s in full]
                if not hits:
                    continue
                unmatched.difference_update(hits)
                r0 = saved.get(full, float(p.detach().float().pow(2).mean().sqrt()))
                targets[full] = r0
                _wrap(module, pname, r0)
    if unmatched:
        raise ValueError(f"NORM_FIXED_RMS: no parameter matches {sorted(unmatched)}")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        if path and not saved:
            os.makedirs(save_dir, exist_ok=True)
            json.dump(targets, open(path, "w"), indent=1)
        vals = list(targets.values())
        print(f"[NORM_FIXED_RMS] {len(targets)} norm weights with fixed rms "
              f"(r0 {'read from' if saved else 'set now, saved to'} {path}; "
              f"mean {sum(vals) / len(vals):.3f}, min {min(vals):.3f}, max {max(vals):.3f})",
              flush=True)
    _DONE = True
