"""
Full builder gate sweep for the activation experiments.

Covers EVERY gate the supervisor's builder produces over the meaningful grid
    L, R in {-1, 0, 1},  M(=y0) in {0, 1/2, 1},  power p in {1/2, 1, 2},  shift b in {0, 1}
minus the *redundant* ones:
  - degenerate: constant gradient (L==M==R) -> linear/dead gate;
  - weight-absorbable duplicates: output-sign (L,M,R)->(-L,-M,-R), input-sign ->(-R,-M,-L),
    both, and positive output scale (so the pure scaling (0,1/2,0)=1/2*(0,1,0) is dropped);
  - power sampled at 3 rates only where a tail decays (an endpoint == 0); linear-both-sides
    shapes fix p=1; shift b in {0,1} only for the bounded bump (both endpoints 0).

Names spell out the parameters: exp_l{L}m{M}r{R}_p{power}_b{shift}. The `exp_` prefix keeps
them clear of the gates already in Megatron-LM-MoE. Gates are GLU gates: g(x) applied to the
gate half, framework does g(x_glu)*x_linear; all pinned g(0)=0. Also included: a structural
extension (asymmetric tails), learnable gates, and baselines.

Ported from Megatron-LM-MoE (megatron/core/exp_gates.py) with the learnable pglu rewritten: one bf16
rounding from the fc1 output to the fc2 input, analytic fp32 backward, fp32 coefficients (see the
pglu section). exp_xrlglu is not ported.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from megatron.core.jit import jit_fuser
    from megatron.core.transformer.module import MegatronModule
except Exception:
    def jit_fuser(fn): return fn
    class MegatronModule(nn.Module):                 # standalone fallback (swallows config)
        def __init__(self, config=None): super().__init__()


# ---- the builder (supervisor's gradient_function / gate_function, inlined) ----
def _grad(x, L, M, R, p):
    """g'(x): -> L at -inf, M at 0, R at +inf; kernel k = 1/(1+|x|)^p."""
    k = 1.0 / (1.0 + x.abs()) ** p
    return torch.where(x <= 0, L + (M - L) * k, R + (M - R) * k)

def _antideriv(u, L, M, R, p):
    """A(u) = int_0^u g'(t) dt with A(0)=0."""
    s = 1.0 + u.abs()
    g = torch.log(s) if p == 1.0 else (torch.pow(s, 1.0 - p) - 1.0) / (1.0 - p)
    return torch.where(u <= 0, L * u - (M - L) * g, R * u + (M - R) * g)

@jit_fuser
def _builder_gate(x, L, M, R, p, b):
    """gate(x) = int_0^x g'(t-b) dt, pinned gate(0)=0."""
    return _antideriv(x - b, L, M, R, p) - _antideriv(-b * torch.ones_like(x), L, M, R, p)


# ---- enumerate the non-redundant (L, M, R, power, shift) combos ---------------
_ENDS, _MIDS, _POWERS = [-1.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 1.0, 2.0]

def _canon(t):
    L, M, R = t
    orbit = [(L, M, R), (-L, -M, -R), (-R, -M, -L), (R, M, L)]   # output/input sign symmetries
    cands = [o for o in orbit if o[1] >= 0]                      # prefer M >= 0
    return max([o for o in cands if o[2] >= o[0]] or cands)      # prefer R >= L (rising)

def _shapes():
    keep = set()
    for L in _ENDS:
        for M in _MIDS:
            for R in _ENDS:
                c = _canon((L, M, R))
                if c[0] == c[1] == c[2]:                         # degenerate (constant grad)
                    continue
                if c[0] == c[2] == 0.0 and c[1] == 0.5:          # scale dup of (0,1,0)
                    continue
                keep.add(c)
    return sorted(keep, key=lambda t: (-(t[0] == 0) - (t[2] == 0), t))

def _f(v):                                                       # format, avoiding "-0"
    return f"{0.0 if v == 0 else v:g}"

def _tail(end, M, p):
    if end != 0.0: return "const"
    return "1/x^2" if M == 0 else {0.5: "1/sqrt", 1.0: "1/x", 1.5: "1/x^1.5", 2.0: "1/x^2"}[p]

def _growth(L, R, p):
    if L != 0.0 or R != 0.0: return "linear"                     # a nonzero tail -> linear output
    return {0.5: "sqrt", 1.0: "log", 1.5: "bounded", 2.0: "bounded"}[p]

# each entry: name -> (callable, (L, M, R, p, b))
BUILDER_GATES = {}

def _register(L, M, R, p, b):
    name = f"exp_l{_f(L)}m{_f(M)}r{_f(R)}_p{_f(p)}_b{int(b)}"
    BUILDER_GATES[name] = ((lambda x, L=L, M=M, R=R, p=p, b=b: _builder_gate(x, L, M, R, p, b)),
                           (L, M, R, p, b))

for (L, M, R) in _shapes():
    zeros = (L == 0.0) + (R == 0.0)
    powers = _POWERS if zeros >= 1 else [1.0]                    # power only where a tail decays
    shifts = [0.0, 1.0] if zeros == 2 else [0.0]                 # shift only the bounded bump
    for p in powers:
        for b in shifts:
            _register(L, M, R, p, b)

# off-grid powers added as sweep follow-ups (the {1/2, 1, 2} grid above stays the catalogue)
_register(0.0, 1.0, 0.0, 1.5, 0.0)       # bump p=3/2: g -> +-2 (bounded); FP4 dead zone |x| > 4.24


# ---- structural extension (outside the single-power builder) ------------------
@jit_fuser
def exp_arglu(x):
    """Asymmetric tails: rsqrt (slow) left for survival + sharp 1/x^2 settle to linear
    right (R=1). Decouples criterion (a) from (b) by side; g(0)=0."""
    xm, xp = x.clamp(max=0.0), x.clamp(min=0.0)
    return torch.where(x <= 0.0, 1.0 - torch.sqrt(1.0 - xm), xp - 0.5 + 0.5 / (1.0 + xp))


# ---- learnable gate: pglu ------------------------------------------------------
# Per expert (one pair per dense / shared MLP): p = softplus(rho) > 0 and a shift b,
#     g'(x) = (1 + |x - b|)^(-p),   g(x) = F_p(x - b) - F_p(-b),   g(0) = 0,
#     F_p(u) = sign(u) ((1 + |u|)^q - 1) / q = sign(u) l exprel(q l),  l = ln(1 + |u|),  q = 1 - p.
# The GLU output is h = g(y_gate) * y_value * prob (prob: the router probability, 1 for dense MLPs).
#
# Numerics. This is also the reference of the fused MXFP4 kernels (megatron/core/mxfp4_alto.py,
# ALTO fused_swiglu.py with GATE=1), which evaluate the same expressions.
#  - Forward: fp32 from the bf16 fc1 output y = [y_gate | y_value] to h, rounded to bf16 once:
#        h = bf16( g(y_gate) * y_value * prob )
#    (Megatron-LM-MoE's version rounds the gate, gate * value and the prob product to bf16 one by one).
#  - No cancellation in g(x) = F_p(x - b) - F_p(-b). When x - b and -b have the same sign s (x between 0
#    and b, or beyond 0 away from b), with B = 1 + |b| and 1 + |x - b| = B (1 + w), w = s x / B:
#        g(x) = s B^q L exprel(q L),  L = ln(1 + w)
#    (the difference of the two F_p terms, ~F_p(-b) each, cost ~eps |F_p(-b)| / |g(x)| relative near x = 0).
#    Otherwise the two terms have opposite signs and are added as they are. The derivatives use the
#    same split: g'(x) = B^-p (1 + w)^-p, dg/db = B^-p - g'(x) = -B^-p expm1(-p L), and dg/dp below.
#  - The slope at x = b is 1 (autograd of sign(u) * ln(1 + |u|) gives 0 there: Megatron-LM-MoE dropped
#    the x-gradient and part of the b-gradient wherever the bf16 gate input equalled the bf16 b).
#  - Backward: analytic, fp32 from dh. d(y_gate) and d(y_value) are rounded to bf16 once each; d(prob),
#    dp and db are fp32 per-row sums; per expert, differences of fp64 prefix sums (deterministic).
#  - exprel(z) and d exprel(z)/dz = phi(z) = (e^z (z - 1) + 1) / z^2 use their series for |z| < 0.5 and the
#    closed forms (from exp) beyond. No expm1: torch.compile lowers it to exp(z) - 1, ~1e-3 relative error at
#    |z| = 1e-4 (p ~ 1). phi's closed form loses ~4 eps / |z| (Megatron-LM-MoE's autograd through
#    expm1(z)/z did that down to |z| = 1e-3).
#  - rho and b stay fp32 parameters (keep_fp32: Float16Module does not cast them to bf16), so the forward
#    uses the optimizer's value, not a bf16 rounding of it (bf16 spacing at b = 1 is 2^-7).
_PHI_SERIES = (0.5, 1.0 / 3.0, 1.0 / 8.0, 1.0 / 30.0, 1.0 / 144.0, 1.0 / 840.0, 1.0 / 5760.0, 1.0 / 45360.0)
# per-expert coefficient columns (pglu_coefficients; the kernels read the same [E, 9] fp32 table)
PGLU_NCOEF = 9           # p, b, C = F_p(-b), s = sign(-b), B = 1 + |b|, B^q, K = B^-p, ln B, dC/dp


_EXPREL_SERIES = (1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0, 1.0 / 40320.0,
                  1.0 / 362880.0)


def _exprel_series(z):
    c = _EXPREL_SERIES
    return c[0] + z * (c[1] + z * (c[2] + z * (c[3] + z * (c[4] + z * (c[5] + z * (c[6] + z * (c[7] + z * c[8])))))))


def _exprel(z):
    """(exprel(z), expm1(z)) with exprel(z) = (e^z - 1) / z: series for |z| < 0.5 (truncation < 1e-9 relative),
    (exp(z) - 1) / z beyond. Built from exp only: torch.compile lowers expm1 to exp(z) - 1 (CPU inductor),
    which loses ~eps/|z| relative near 0, the p ~ 1 regime pglu starts in."""
    small = z.abs() < 0.5
    zs = torch.where(small, torch.ones_like(z), z)
    ex = torch.where(small, _exprel_series(z), (torch.exp(zs) - 1.0) / zs)
    return ex, z * ex


def _phi(z, em1):
    """(e^z (z - 1) + 1) / z^2 = d exprel(z) / dz, with em1 = expm1(z)."""
    small = z.abs() < 0.5
    zs = torch.where(small, torch.ones_like(z), z)
    c = _PHI_SERIES
    series = c[0] + z * (c[1] + z * (c[2] + z * (c[3] + z * (c[4] + z * (c[5] + z * (c[6] + z * c[7]))))))
    return torch.where(small, series, (zs * em1 + zs - em1) / (zs * zs))


def pglu_coefficients(raw_p, b):
    """Per-expert coefficient table [E, PGLU_NCOEF] fp32 (columns: see PGLU_NCOEF); one fused kernel."""
    return _fused(_pglu_coefficients)(raw_p, b)


def _pglu_coefficients(raw_p, b):
    p = F.softplus(raw_p.float())
    b = b.float()
    q = 1.0 - p
    lb = torch.log1p(b.abs())
    zb = q * lb
    exb, em1b = _exprel(zb)
    s = torch.sign(-b)
    c = s * lb * exb
    dc_dp = -s * lb * lb * _phi(zb, em1b)
    return torch.stack([p, b, c, s, 1.0 + b.abs(), em1b + 1.0, torch.exp(-p * lb), lb, dc_dp], dim=-1)


def _pglu_eval(x, cr):
    """fp32 gate input x and coefficient columns cr [.., 9] -> gate and the terms its derivatives reuse."""
    p, b, c, s, bb, bq = (cr[..., i:i + 1] for i in range(6))
    u = x - b
    su = torch.sign(u)
    same = (s != 0) & (su == s)
    w = torch.where(same, s * x / bb, u.abs())
    l = torch.log1p(w)
    z = (1.0 - p) * l
    ex, em1 = _exprel(z)
    gate = torch.where(same, s * bq * l * ex, su * l * ex - c)
    return gate, same, su, w, l, z, em1, ex


def _pglu_fwd(g, v, cr, prob):
    gate = _pglu_eval(g.float(), cr)[0]
    return (gate * v.float() * prob).to(g.dtype)


def _pglu_bwd(dh, g, v, cr, prob):
    vf, dhf = v.float(), dh.float()
    gate, same, su, w, l, z, em1, ex = _pglu_eval(g.float(), cr)
    p, s, bq, k, lb, dc_dp = (cr[..., i:i + 1] for i in (0, 3, 5, 6, 7, 8))
    d_out = dhf * prob
    d_v = (d_out * gate).to(g.dtype)
    dgate = d_out * vf                                  # dL/dg(x)
    gp1 = (em1 + 1.0) / (1.0 + w)                       # (1 + w)^-p
    gp = torch.where(same, k * gp1, gp1)                # g'(x) = (1 + |x - b|)^-p
    d_g = (dgate * gp).to(g.dtype)
    phi = _phi(z, em1)
    dg_dp = torch.where(same, -s * bq * l * (lb * ex + l * phi), -su * l * l * phi - dc_dp)
    # dg/db = K - g'(x); same side: -K expm1(-p L), with (1 + w)^-p = gp1 = e^zp (series near zp = 0)
    zp = -p * l
    dg_db = torch.where(same, torch.where(zp.abs() < 0.5, -k * zp * _exprel_series(zp), k * (1.0 - gp1)), k - gp1)
    dprob = (gate * vf * dhf).sum(-1, keepdim=True)
    sums = torch.stack([(dgate * dg_dp).sum(-1), (dgate * dg_db).sum(-1)], -1)
    return d_g, d_v, dprob, sums


_COMPILED = {}


def _fused(fn):
    """fn through Megatron's jit fuser as configured when first called (torch.compile in training)."""
    if fn not in _COMPILED:
        from megatron.core import jit as _jit
        f = _jit.jit_fuser
        _COMPILED[fn] = f(fn, dynamic=True) if f is torch.compile else (fn if f is torch.jit.script else f(fn))
    return _COMPILED[fn]


def pglu_bwd_rows(dh, y, prob, cr):
    """Backward of h = bf16(g(y_gate) * y_value * prob) for rows with coefficient rows cr ([N, 9] or [1, 9]):
    dy [N, 2F] (bf16 halves), dprob [N, 1] fp32, sums [N, 2] fp32 = per-row sums of dL/dp, dL/db."""
    g, v = torch.chunk(y, 2, dim=-1)
    d_g, d_v, dprob, sums = _fused(_pglu_bwd)(dh, g, v, cr, prob)
    return torch.cat([d_g, d_v], dim=-1), dprob, sums


def segment_sum(s, counts):
    """Sums of the rows of s [N, k] over consecutive segments of the given lengths -> [E, k] fp64,
    as differences of fp64 prefix sums (deterministic, no atomics)."""
    if len(counts) == 1:
        return s.double().sum(0, keepdim=True)
    # prefix sums along the contiguous dim ([k, N]; ROCm's cumsum over dim 0 of [N, k] is ~70x slower)
    cs = F.pad(s.t().double().contiguous().cumsum(1), (1, 0))
    bounds = torch.tensor([0, *itertools.accumulate(counts)], device=s.device)
    return (cs[:, bounds[1:]] - cs[:, bounds[:-1]]).t()


def pglu_param_grads(raw_p, sums):
    """dL/drho, dL/db [E] fp32 from the per-expert sums [E, 2] (dL/dp, dL/db); dp/drho = sigmoid(rho)."""
    dp, db = sums.unbind(-1)
    return (dp * torch.sigmoid(raw_p.double())).float(), db.float()


_ONE = {}


def _one(device):
    if device not in _ONE:
        _ONE[device] = torch.ones(1, dtype=torch.float32, device=device)
    return _ONE[device]


def _rows(coef, counts, n):
    """Per-expert table [E, 9] -> per-row [N, 9] (or [1, 9] for one expert)."""
    if len(counts) == 1:
        return coef
    return torch.repeat_interleave(coef, torch.as_tensor(counts, device=coef.device), dim=0, output_size=n)


class _WeightedPGLU(torch.autograd.Function):
    """h = bf16(g(y_gate) * y_value * prob) over expert-sorted rows (segments of `counts` rows)."""

    @staticmethod
    def forward(ctx, y, prob, raw_p, b, counts):
        g, v = torch.chunk(y, 2, dim=-1)
        pr = prob.float() if prob is not None else _one(y.device)
        cr = _rows(pglu_coefficients(raw_p, b), counts, y.shape[0])
        h = _fused(_pglu_fwd)(g, v, cr, pr)
        ctx.save_for_backward(y, prob, raw_p, b)
        ctx.counts = counts
        return h

    @staticmethod
    def backward(ctx, dh):
        y, prob, raw_p, b = ctx.saved_tensors
        pr = prob.float() if prob is not None else _one(y.device)
        cr = _rows(pglu_coefficients(raw_p, b), ctx.counts, y.shape[0])
        dy, dprob, sums = pglu_bwd_rows(dh, y, pr, cr)
        d_raw, d_b = pglu_param_grads(raw_p, segment_sum(sums, ctx.counts))
        dprob = dprob.to(prob.dtype).reshape(prob.shape) if prob is not None and ctx.needs_input_grad[1] else None
        return dy, dprob, d_raw.to(raw_p.dtype), d_b.to(b.dtype), None


class _PerExpertGate(MegatronModule):
    per_expert_gate = True

    def __init__(self, config=None, num_local_experts=1):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # routed experts under EP: each rank owns different experts, so the coefficient
        # grads must reduce over the expert-data-parallel group, like the expert weights.
        self._expert_parallel = (num_local_experts > 1 and config is not None
                                 and getattr(config, "expert_model_parallel_size", 1) > 1)

    def _param(self, value):
        t = nn.Parameter(torch.full((self.num_local_experts,), float(value), dtype=torch.float32))
        t.keep_fp32 = True                     # Float16Module leaves it fp32
        if self._expert_parallel:
            t.allreduce = False
        return t


class ExpPGLU(_PerExpertGate):
    """Learnable tail exponent p = softplus(rho) > 0 and shift b (per expert):
    g'(x) = 1/(1+|x-b|)^p, g(x) = F_p(x-b) - F_p(-b), g(0) = 0. The default init (p=1, b=1)
    is exactly exp_l0m1r0_p1_b1. The MLPs call weighted_glu (fc1 output -> fc2 input)."""

    def __init__(self, config=None, p_init=1.0, b_init=1.0, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.raw_p = self._param(torch.log(torch.expm1(torch.tensor(float(p_init)))).item())
        self.b = self._param(b_init)

    def coefficients(self):
        return {"p": F.softplus(self.raw_p.float()), "b": self.b.float()}

    def _counts(self, n, tokens_per_expert):
        if self.num_local_experts == 1:
            return [n]
        if tokens_per_expert is None:
            raise ValueError(f"{type(self).__name__} with {self.num_local_experts} local experts "
                             "needs tokens_per_expert")
        return [int(t) for t in tokens_per_expert]

    def weighted_glu(self, y, probs=None, tokens_per_expert=None):
        """h = bf16(gate(y_gate) * y_value * probs) for the fc1 output y [N, 2F] (expert-sorted rows),
        probs [N, 1] or None."""
        return _WeightedPGLU.apply(y, probs, self.raw_p, self.b, self._counts(y.shape[0], tokens_per_expert))

    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        """The gate alone, g(x) in x.dtype (analysis; training uses weighted_glu)."""
        coef = pglu_coefficients(self.raw_p, self.b)
        if expert_ids is not None:
            cr = coef[expert_ids]
        else:
            cr = _rows(coef, self._counts(x.shape[0], tokens_per_expert), x.shape[0])
        return _pglu_eval(x.float(), cr)[0].to(x.dtype)


# ---- baselines ---------------------------------------------------------------
@jit_fuser
def exp_swiglu(x): return F.silu(x)
@jit_fuser
def exp_situ(x):   return torch.sigmoid(x) * torch.tanh(x)
@jit_fuser
def exp_ssglu(x):  return x * (0.5 + 0.5 * F.softsign(x))


# ---- master registry: name -> callable (or class) ----------------------------
GATES = {name: fn for name, (fn, _) in BUILDER_GATES.items()}
LEARNABLE_GATES = {"exp_pglu": ExpPGLU}
GATES.update(exp_arglu=exp_arglu, **LEARNABLE_GATES,
             exp_swiglu=exp_swiglu, exp_situ=exp_situ, exp_ssglu=exp_ssglu)
# exp_xrlglu (Megatron-LM-MoE) is not ported to this fork.
