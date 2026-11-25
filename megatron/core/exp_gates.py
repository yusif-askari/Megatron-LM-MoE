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

Self-test (g(0)=0 and autograd == analytic g' for every builder gate):
    uv run python new_gates.py
"""
import itertools
import os
import math

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


# ---- bump gates (L = R = 0): fp32 GLU rounded once, analytic backward -----------------------
# For L = R = 0 the builder gate is
#     g(x) = M (F_p(x - b) - F_p(-b)),   g'(x) = M (1 + |x - b|)^-p,
#     F_p(u) = sign(u) ((1 + |u|)^q - 1) / q = sign(u) l exprel(q l),   l = ln(1 + |u|),  q = 1 - p
# (exp_pglu with p and b fixed). _builder_gate in the generic GLU path has three defects on these
# gates, all fixed here:
#  - slope at x = b: autograd through abs / where gives L = 0 at u = 0; the true slope is M. bf16
#    gate inputs equal to b = 1 are common (874 of 1.44M elements in the audit), and those rows
#    lost their input gradient (d(y_gate) relative error 3.8e-2 .. 8.2e-2 on b = 1 bumps);
#  - three bf16 roundings (gate, gate * x_linear, * prob): here h = bf16(g(x_glu) x_linear prob)
#    is evaluated in fp32 and rounded once; d(x_glu), d(x_linear) are rounded once each and
#    d(prob) is an fp32 row sum;
#  - cancellation of F_p(x - b) - F_p(-b) near x = 0 (b != 0): where x - b and -b have the same
#    sign s, with B = 1 + |b| and 1 + |x - b| = B (1 + w), w = s x / B,
#        g(x) = M s B^q L exprel(q L),   L = ln(1 + w),   g'(x) = M B^-p (1 + w)^-p;
#    otherwise the two terms have opposite signs and are added as they are.
# exprel is built from exp (series for |z| < 0.5): torch.compile lowers expm1 to exp(z) - 1 on CPU.
# Same expressions as exp_pglu's reference in the MXFP4 fork (megatron-main, pglu section).
_EXPREL_SERIES = (1.0, 1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0,
                  1.0 / 40320.0, 1.0 / 362880.0)


def _exprel_exp(z):
    """(e^z - 1) / z: series for |z| < 0.5 (truncation < 1e-9 relative), (exp(z) - 1) / z beyond."""
    c = _EXPREL_SERIES
    series = c[0] + z * (c[1] + z * (c[2] + z * (c[3] + z * (c[4] + z * (c[5] + z * (c[6] + z * (
        c[7] + z * c[8])))))))
    small = z.abs() < 0.5
    zs = torch.where(small, torch.ones_like(z), z)
    return torch.where(small, series, (torch.exp(zs) - 1.0) / zs)


def _bump_eval(x, M, p, b, s, B, Bq, K, c):
    """x (fp32 or wider) -> (g(x), g'(x)). Constants (BumpGate.k): s = sign(-b), B = 1 + |b|,
    Bq = B^q, K = B^-p, c = F_p(-b)."""
    q = 1.0 - p
    u = x - b
    su = torch.sign(u)
    if s == 0.0:                                     # b = 0: g = M F_p(x), nothing subtracted
        l = torch.log1p(u.abs())
        return M * su * l * _exprel_exp(q * l), M * torch.exp(-p * l)
    same = su == s
    w = torch.where(same, (s / B) * x, u.abs())
    l = torch.log1p(w)
    ex = _exprel_exp(q * l)
    gate = torch.where(same, (s * Bq) * l * ex, su * l * ex - c)
    gp1 = torch.exp(-p * l)                          # (1 + w)^-p; = 1 at x = b
    return M * gate, M * torch.where(same, K * gp1, gp1)


def _wide(t):
    return t.to(torch.promote_types(t.dtype, torch.float32))


@jit_fuser
def _bump_val(x, M, p, b, s, B, Bq, K, c):
    return _bump_eval(_wide(x), M, p, b, s, B, Bq, K, c)[0].to(x.dtype)


@jit_fuser
def _bump_val_bwd(dy, x, M, p, b, s, B, Bq, K, c):
    return (_wide(dy) * _bump_eval(_wide(x), M, p, b, s, B, Bq, K, c)[1]).to(x.dtype)


@jit_fuser
def _bump_glu_fwd(g, v, prob, M, p, b, s, B, Bq, K, c):
    gate = _bump_eval(g.float(), M, p, b, s, B, Bq, K, c)[0]
    return (gate * v.float() * prob).to(g.dtype)


@jit_fuser
def _bump_glu_bwd(dh, g, v, prob, M, p, b, s, B, Bq, K, c):
    gate, gp = _bump_eval(g.float(), M, p, b, s, B, Bq, K, c)
    vf, dhf = v.float(), dh.float()
    d_out = dhf * prob
    d_g = (d_out * vf * gp).to(g.dtype)
    d_v = (d_out * gate).to(v.dtype)
    dprob = (gate * vf * dhf).sum(-1, keepdim=True)
    return d_g, d_v, dprob


class _BumpGateFn(torch.autograd.Function):
    """The gate alone, g(x) in x.dtype, with the analytic slope (analysis, --log-glu-stats)."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x)
        ctx.k = k
        return _bump_val(x, *k)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return _bump_val_bwd(dy, x, *ctx.k), None


_ONE = {}


def _one(device):
    if device not in _ONE:
        _ONE[device] = torch.ones(1, dtype=torch.float32, device=device)
    return _ONE[device]


class _BumpGLU(torch.autograd.Function):
    """h = bf16(g(x_glu) * x_linear * prob), fp32 inside, prob [N, 1] or None (= 1)."""

    @staticmethod
    def forward(ctx, g, v, prob, k):
        pr = prob.float() if prob is not None else _one(g.device)
        ctx.save_for_backward(g, v, prob)
        ctx.k = k
        return _bump_glu_fwd(g, v, pr, *k)

    @staticmethod
    def backward(ctx, dh):
        g, v, prob = ctx.saved_tensors
        pr = prob.float() if prob is not None else _one(g.device)
        d_g, d_v, dprob = _bump_glu_bwd(dh, g, v, pr, *ctx.k)
        if prob is not None and ctx.needs_input_grad[2]:
            dprob = dprob.to(prob.dtype).reshape(prob.shape)
        else:
            dprob = None
        return d_g, d_v, dprob, None


class BumpGate:
    """Builder gate with L = R = 0 (exp_l0m{M}r0_p{p}_b{b}). Called on x: the gate alone, g(x) in
    x.dtype. The GLU paths call weighted_glu(x_glu, x_linear, probs) instead (fc1 output -> fc2
    input, one rounding)."""
    per_expert_gate = False

    def __init__(self, M, p, b, name):
        M, p, b = float(M), float(p), float(b)
        q = 1.0 - p
        lb = math.log1p(abs(b))
        Fb = lb if q == 0.0 else math.expm1(q * lb) / q          # F_p(|b|)
        s = 0.0 if b == 0.0 else -math.copysign(1.0, b)           # sign(-b)
        B = 1.0 + abs(b)
        self.k = (M, p, b, s, B, B ** q, B ** -p, s * Fb)
        self.__name__ = name

    def __call__(self, x):
        return _BumpGateFn.apply(x, self.k)

    def weighted_glu(self, x_glu, x_linear, probs=None):
        return _BumpGLU.apply(x_glu, x_linear, probs, self.k)

    def __repr__(self):
        return f"BumpGate({self.__name__})"


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
    if L == 0.0 and R == 0.0:                                    # bump: fused, see BumpGate
        fn = BumpGate(M, p, b, name)
    else:
        fn = lambda x, L=L, M=M, R=R, p=p, b=b: _builder_gate(x, L, M, R, p, b)
    BUILDER_GATES[name] = (fn, (L, M, R, p, b))

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
# Numerics. Ported from the MXFP4 fork (megatron-main), where this is also the reference of the fused
# MXFP4 kernels (ALTO fused_swiglu.py with GATE=1).
#  - Forward: fp32 from the bf16 fc1 output y = [y_gate | y_value] to h, rounded to bf16 once:
#        h = bf16( g(y_gate) * y_value * prob )
#    (the earlier version here rounded the gate, gate * value and the prob product to bf16 one by one).
#  - No cancellation in g(x) = F_p(x - b) - F_p(-b). When x - b and -b have the same sign s (x between 0
#    and b, or beyond 0 away from b), with B = 1 + |b| and 1 + |x - b| = B (1 + w), w = s x / B:
#        g(x) = s B^q L exprel(q L),  L = ln(1 + w)
#    (the difference of the two F_p terms, ~F_p(-b) each, cost ~eps |F_p(-b)| / |g(x)| relative near x = 0).
#    Otherwise the two terms have opposite signs and are added as they are. The derivatives use the
#    same split: g'(x) = B^-p (1 + w)^-p, dg/db = B^-p - g'(x) = -B^-p expm1(-p L), and dg/dp below.
#  - The slope at x = b is 1 (autograd of sign(u) * ln(1 + |u|) gives 0 there: the earlier version dropped
#    the x-gradient and part of the b-gradient wherever the bf16 gate input equalled the bf16 b).
#  - Backward: analytic, fp32 from dh. d(y_gate) and d(y_value) are rounded to bf16 once each; d(prob),
#    dp and db are fp32 per-row sums; per expert, differences of fp64 prefix sums (deterministic).
#  - exprel(z) and d exprel(z)/dz = phi(z) = (e^z (z - 1) + 1) / z^2 use their series for |z| < 0.5 and the
#    closed forms (from exp) beyond. No expm1: torch.compile lowers it to exp(z) - 1, ~1e-3 relative error at
#    |z| = 1e-4 (p ~ 1). phi's closed form loses ~4 eps / |z| (the earlier version's autograd through
#    expm1(z)/z did that down to |z| = 1e-3).
#  - rho and b stay fp32 parameters (keep_fp32: Float16Module does not cast them to bf16), so the forward
#    uses the optimizer's value, not a bf16 rounding of it (bf16 spacing at b = 1 is 2^-7).
_PHI_SERIES = (0.5, 1.0 / 3.0, 1.0 / 8.0, 1.0 / 30.0, 1.0 / 144.0, 1.0 / 840.0, 1.0 / 5760.0, 1.0 / 45360.0)
# per-expert coefficient columns (pglu_coefficients; the kernels read the same [E, 9] fp32 table)
PGLU_NCOEF = 9           # p, b, C = F_p(-b), s = sign(-b), B = 1 + |b|, B^q, K = B^-p, ln B, dC/dp


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


def bglu_coefficients(b, p):
    """The same table for a fixed tail exponent p (python float; exp_bglu_p{p}, exp_bglu_p{p}_c)."""
    return _fused(_bglu_coefficients)(b, p)


def _coefficient_table(raw_p, b, p_fixed):
    return pglu_coefficients(raw_p, b) if raw_p is not None else bglu_coefficients(b, p_fixed)


def _pglu_coefficients(raw_p, b):
    return _coefficients(F.softplus(raw_p.float()), b)


def _bglu_coefficients(b, p):
    return _coefficients(torch.full_like(b, p, dtype=torch.float32), b)


def _coefficients(p, b):
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


# cc (optional): the per-row slope shift c of exp_pglu_c / exp_bglu_p{p}_c ([N, 1] or [1, 1] fp32):
#     gate = g(x) + c x,  gate' = g'(x) + c,  dgate/dc = x  (third column of the per-row sums).
def _pglu_fwd(g, v, cr, prob, cc=None):
    gf = g.float()
    gate = _pglu_eval(gf, cr)[0]
    if cc is not None:
        gate = gate + cc * gf
    return (gate * v.float() * prob).to(g.dtype)


def _pglu_bwd(dh, g, v, cr, prob, cc=None):
    vf, dhf, gf = v.float(), dh.float(), g.float()
    gate, same, su, w, l, z, em1, ex = _pglu_eval(gf, cr)
    if cc is not None:
        gate = gate + cc * gf
    p, s, bq, k, lb, dc_dp = (cr[..., i:i + 1] for i in (0, 3, 5, 6, 7, 8))
    d_out = dhf * prob
    d_v = (d_out * gate).to(g.dtype)
    dgate = d_out * vf                                  # dL/dg(x)
    gp1 = (em1 + 1.0) / (1.0 + w)                       # (1 + w)^-p
    gp = torch.where(same, k * gp1, gp1)                # g'(x) = (1 + |x - b|)^-p
    if cc is not None:
        gp = gp + cc
    d_g = (dgate * gp).to(g.dtype)
    phi = _phi(z, em1)
    dg_dp = torch.where(same, -s * bq * l * (lb * ex + l * phi), -su * l * l * phi - dc_dp)
    # dg/db = K - g'(x); same side: -K expm1(-p L), with (1 + w)^-p = gp1 = e^zp (series near zp = 0)
    zp = -p * l
    dg_db = torch.where(same, torch.where(zp.abs() < 0.5, -k * zp * _exprel_series(zp), k * (1.0 - gp1)), k - gp1)
    dprob = (gate * vf * dhf).sum(-1, keepdim=True)
    cols = [(dgate * dg_dp).sum(-1), (dgate * dg_db).sum(-1)]
    if cc is not None:
        cols.append((dgate * gf).sum(-1))
    sums = torch.stack(cols, -1)
    return d_g, d_v, dprob, sums


_COMPILED = {}


def _fused(fn):
    """fn through Megatron's jit fuser as configured when first called (torch.compile in training)."""
    if fn not in _COMPILED:
        try:
            from megatron.core import jit as _jit
            f = _jit.jit_fuser
        except Exception:                            # standalone (file loaded by path)
            f = jit_fuser
        _COMPILED[fn] = f(fn, dynamic=True) if f is torch.compile else (fn if f is torch.jit.script else f(fn))
    return _COMPILED[fn]


def pglu_bwd_rows(dh, y, prob, cr, cc=None):
    """Backward of h = bf16(g(y_gate) * y_value * prob) for rows with coefficient rows cr ([N, 9] or [1, 9]):
    dy [N, 2F] (bf16 halves), dprob [N, 1] fp32, sums [N, 2] fp32 = per-row sums of dL/dp, dL/db
    ([N, 3] with the slope shift cc: plus dL/dc)."""
    g, v = torch.chunk(y, 2, dim=-1)
    d_g, d_v, dprob, sums = _fused(_pglu_bwd)(dh, g, v, cr, prob, cc)
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
    """dL/drho, dL/db [E] fp32 from the per-expert sums [E, 2+] (dL/dp, dL/db, ...); dp/drho = sigmoid(rho)."""
    dp, db = sums[..., 0], sums[..., 1]
    return (dp * torch.sigmoid(raw_p.double())).float(), db.float()


def _rows(coef, counts, n):
    """Per-expert table [E, k] -> per-row [N, k] (or [1, k] for one expert)."""
    if len(counts) == 1:
        return coef
    return torch.repeat_interleave(coef, torch.as_tensor(counts, device=coef.device), dim=0, output_size=n)


def _param_grads(tot, raw_p, b, c, need_p, need_c):
    """Per-expert sums tot [E, 2 or 3] fp64 -> (dL/drho or None, dL/db, dL/dc or None) in the parameters' dtypes."""
    if raw_p is not None and need_p:
        d_raw, d_b = pglu_param_grads(raw_p, tot)
        d_raw = d_raw.to(raw_p.dtype)
    else:
        d_raw, d_b = None, tot[..., 1].float()
    d_c = tot[..., 2].float().to(c.dtype) if c is not None and need_c else None
    return d_raw, d_b.to(b.dtype), d_c


class _WeightedPGLU(torch.autograd.Function):
    """h = bf16(gate(y_gate) * y_value * prob) over expert-sorted rows (segments of `counts` rows), for the
    pglu family: gate = g(x) (+ c x if c is given), with p = softplus(raw_p) (exp_pglu*) or, when raw_p is
    None, the fixed exponent p_fixed (exp_bglu_p{p}*). raw_p, b, c: [E] fp32 (c may be a function of a
    parameter, e.g. softplus for exp_bglu_p{p}_cs; autograd takes dL/dc from here)."""

    @staticmethod
    def forward(ctx, y, prob, raw_p, b, counts, p_fixed=None, c=None):
        g, v = torch.chunk(y, 2, dim=-1)
        pr = prob.float() if prob is not None else _one(y.device)
        cr = _rows(_coefficient_table(raw_p, b, p_fixed), counts, y.shape[0])
        cc = None if c is None else _rows(c.float().unsqueeze(-1), counts, y.shape[0])
        h = _fused(_pglu_fwd)(g, v, cr, pr, cc)
        ctx.save_for_backward(y, prob, raw_p, b, c)
        ctx.counts, ctx.p_fixed = counts, p_fixed
        return h

    @staticmethod
    def backward(ctx, dh):
        y, prob, raw_p, b, c = ctx.saved_tensors
        pr = prob.float() if prob is not None else _one(y.device)
        cr = _rows(_coefficient_table(raw_p, b, ctx.p_fixed), ctx.counts, y.shape[0])
        cc = None if c is None else _rows(c.float().unsqueeze(-1), ctx.counts, y.shape[0])
        dy, dprob, sums = pglu_bwd_rows(dh, y, pr, cr, cc)
        d_raw, d_b, d_c = _param_grads(segment_sum(sums, ctx.counts), raw_p, b, c,
                                       ctx.needs_input_grad[2], ctx.needs_input_grad[6])
        dprob = dprob.to(prob.dtype).reshape(prob.shape) if prob is not None and ctx.needs_input_grad[1] else None
        return dy, dprob, d_raw, d_b, None, None, d_c


class _PGLUGate(torch.autograd.Function):
    """The gate alone, gate(x) in x.dtype (analysis: --log-glu-stats, tests), for rows of local expert idx [N]
    (None: one expert): fp32 inside, the analytic slope (1 + c at x = b), dL/drho, dL/db, dL/dc summed per
    expert. raw_p / p_fixed / c as in _WeightedPGLU."""

    @staticmethod
    def forward(ctx, x, raw_p, b, idx, p_fixed=None, c=None):
        coef = _coefficient_table(raw_p, b, p_fixed)
        cr = coef if idx is None else coef.index_select(0, idx)
        ctx.save_for_backward(x, raw_p, b, idx, c)
        ctx.p_fixed = p_fixed
        xs = x.unsqueeze(-1) if x.dim() == 1 else x            # 1-D x: one row per element
        xf = xs.float()
        gate = _pglu_eval(xf, cr)[0]
        if c is not None:
            gate = gate + _gate_rows(c, idx) * xf
        return gate.to(x.dtype).reshape(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x, raw_p, b, idx, c = ctx.saved_tensors
        coef = _coefficient_table(raw_p, b, ctx.p_fixed)
        cr = coef if idx is None else coef.index_select(0, idx)
        cc = None if c is None else _gate_rows(c, idx)
        xs = x.unsqueeze(-1) if x.dim() == 1 else x
        dys = dy.reshape(xs.shape)
        d_x, _, _, sums = _fused(_pglu_bwd)(dys, xs, torch.ones_like(xs), cr, _one(x.device), cc)
        d_x = d_x.reshape(x.shape)
        d_raw = d_b = d_c = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[5]:
            s = sums.reshape(-1, sums.shape[-1]).double()
            if idx is None:
                tot = s.sum(0, keepdim=True)
            else:
                tot = torch.zeros(b.numel(), s.shape[-1], dtype=torch.float64, device=s.device).index_add_(0, idx, s)
            d_raw, d_b, d_c = _param_grads(tot, raw_p, b, c, ctx.needs_input_grad[1], ctx.needs_input_grad[5])
        return d_x, d_raw, d_b, None, None, d_c


def _gate_rows(c, idx):
    """Slope shift c [E] -> [1, 1] (one expert, idx None) or [N, 1] (rows of local expert idx), fp32."""
    cf = c.float()
    return cf.reshape(1, 1) if idx is None else cf.index_select(0, idx).unsqueeze(-1)


# ---- learnable gates ---------------------------------------------------------
# One coefficient set per local expert (num_local_experts=1 for dense / shared MLPs, and for
# standalone use). forward(x, tokens_per_expert) maps each expert's coefficients onto its rows
# of the expert-concatenated [N, F] input; forward(x, expert_ids=ids) does the same from
# explicit row ids (used by --log-glu-stats). Math runs in (at least) fp32, output in x.dtype.
# Plain-autograd reference of the pglu family (the implementation before the fused kernels above);
# used only by the self-test below and the unit tests.
def _ref_exprel(z):
    """(e^z - 1)/z with its Taylor series near 0, so value and d/dz stay finite at z = 0."""
    small = z.abs() < 1e-3
    zs = torch.where(small, torch.ones_like(z), z)
    return torch.where(small, 1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)), torch.expm1(zs) / zs)

def _ref_log1p_rel(a):
    """ln(1+a)/a for a >= 0, with its Taylor series near 0 (value 1 at a = 0)."""
    small = a < 1e-4
    as_ = torch.where(small, torch.ones_like(a), a)
    return torch.where(small, 1.0 - a * (0.5 - a / 3.0), torch.log1p(as_) / as_)

def _ref_bump_integral(u, p):
    """F_p(u) = sign(u) ((1+|u|)^(1-p) - 1)/(1-p)  (sign(u) ln(1+|u|) at p = 1), written as
    sign(u) l exprel((1-p) l) with l = ln(1+|u|): smooth in p, so dF/dp is exact at p = 1.
    sign(u) l is computed as u * (l/|u|): same value, but autograd gives F_p'(0) = 1 (with
    sign(u) * log1p(|u|) both factors have derivative 0 at u = 0, so dF/du came out 0 at x = b)."""
    a = u.abs()
    l = torch.log1p(a)
    return u * _ref_log1p_rel(a) * _ref_exprel((1.0 - p) * l)

@jit_fuser
def _pglu_ref(x, p, b):
    xf = x.to(p.dtype)
    return (_ref_bump_integral(xf - b, p) - _ref_bump_integral(-b, p)).to(x.dtype)

@jit_fuser
def _xrlglu(x, a):
    xf = x.to(a.dtype)
    return ((1.0 + a) * (F.relu(xf) - 0.5 * torch.log1p(xf.abs())) - a * xf).to(x.dtype)

class _PerExpertGate(MegatronModule):
    per_expert_gate = True                            # glu_stats: call with expert_ids=

    def __init__(self, config=None, num_local_experts=1):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # routed experts under EP: each rank owns different experts, so the coefficient
        # grads must reduce over the expert-data-parallel group, like the expert weights.
        self._expert_parallel = (num_local_experts > 1 and config is not None
                                 and getattr(config, "expert_model_parallel_size", 1) > 1)

    def _param(self, value, keep_fp32=False):
        t = nn.Parameter(torch.full((self.num_local_experts,), float(value), dtype=torch.float32))
        if keep_fp32:
            t.keep_fp32 = True                         # Float16Module leaves it fp32
        if self._expert_parallel:
            t.allreduce = False
        return t

    def _rows(self, t, x, tokens_per_expert=None, expert_ids=None):
        """(E,) coefficient -> tensor broadcastable against x ([N, F] -> [N, 1])."""
        t = t.to(torch.promote_types(x.dtype, torch.float32))
        if expert_ids is not None:
            return t[expert_ids].unsqueeze(-1)
        if self.num_local_experts == 1:
            return t
        if tokens_per_expert is None:
            raise ValueError(f"{type(self).__name__} with {self.num_local_experts} local experts "
                             "needs tokens_per_expert")
        n = x.shape[0]
        counts = torch.as_tensor(tokens_per_expert, device=x.device)
        return torch.repeat_interleave(t, counts, output_size=n).unsqueeze(-1)

# PGLU_DUMP_DIR (analysis only; unset in training): the first call with grad enabled of each routed-expert gate
# whose MoE-layer index (0 = first MoE layer, in call order) is in PGLU_DUMP_LAYERS (default 0) saves its real
# inputs and coefficients, the h it produced, and dL/dh and dL/dy from the backward, to
# <dir>/rank<r>_moe<i>.pt (scripts: optional-project/sweep-results/3b-large-scale/scripts/check_pglu_real.py).
_DUMP_DIR = os.environ.get("PGLU_DUMP_DIR")
_DUMP_LAYERS = {int(i) for i in os.environ.get("PGLU_DUMP_LAYERS", "0").split(",") if i.strip()}
_DUMP_ORDER = {}


def _dump_once(gate, y, probs, counts, h):
    idx = _DUMP_ORDER[id(gate)]
    if idx not in _DUMP_LAYERS or getattr(gate, "_dumped", False):
        return
    gate._dumped = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    raw_p, p_fixed, c = gate._gate_args()
    rec = {"y": y.detach().cpu(), "probs": None if probs is None else probs.detach().cpu(), "counts": list(counts),
           "raw_p": None if raw_p is None else raw_p.detach().float().cpu(), "b": gate.b.detach().float().cpu(),
           "p_fixed": p_fixed, "c": None if c is None else c.detach().float().cpu(), "h": h.detach().cpu()}
    path = os.path.join(_DUMP_DIR, f"rank{rank}_moe{idx}.pt")

    def on_dh(g):
        rec["dh"] = g.detach().cpu()
        if not y.requires_grad:
            torch.save(rec, path)

    def on_dy(g):
        rec["dy"] = g.detach().cpu()
        torch.save(rec, path)
        print(f"[PGLU_DUMP] wrote {path}: rows {y.shape[0]}, F {y.shape[1] // 2}", flush=True)

    h.register_hook(on_dh)
    if y.requires_grad:
        y.register_hook(on_dy)


class _PGLUFamilyGate(_PerExpertGate):
    """The pglu family on the fused kernels above (fp32 coefficients, h rounded to bf16 once, analytic
    backward). Subclasses set the parameters and _gate_args() -> (raw_p or None, fixed p or None, c or None).
    The MLPs call weighted_glu (fc1 output -> fc2 input); forward(x) is the gate alone (analysis)."""

    def _gate_args(self):
        raise NotImplementedError

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
        raw_p, p_fixed, c = self._gate_args()
        counts = self._counts(y.shape[0], tokens_per_expert)
        h = _WeightedPGLU.apply(y, probs, raw_p, self.b, counts, p_fixed, c)
        if _DUMP_DIR and self.num_local_experts > 1:
            _DUMP_ORDER.setdefault(id(self), len(_DUMP_ORDER))      # forward order, also under recompute
            if torch.is_grad_enabled():
                _dump_once(self, y, probs, counts, h)
        return h

    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        """The gate alone, gate(x) in x.dtype (analysis; training uses weighted_glu)."""
        if expert_ids is not None:
            idx = expert_ids.long()
        elif self.num_local_experts == 1:
            idx = None
        else:
            counts = torch.as_tensor(self._counts(x.shape[0], tokens_per_expert), device=x.device)
            idx = torch.repeat_interleave(torch.arange(self.num_local_experts, device=x.device), counts,
                                          output_size=x.shape[0])
        raw_p, p_fixed, c = self._gate_args()
        return _PGLUGate.apply(x, raw_p, self.b, idx, p_fixed, c)


class ExpPGLU(_PGLUFamilyGate):
    """Learnable tail exponent p = softplus(rho) > 0 and shift b (per expert):
    g'(x) = 1/(1+|x-b|)^p, g(x) = F_p(x-b) - F_p(-b), g(0) = 0. The default init (p=1, b=1)
    is exactly exp_l0m1r0_p1_b1. rho and b are fp32 parameters (keep_fp32). See the pglu section above
    for the numerics."""

    def __init__(self, config=None, p_init=1.0, b_init=1.0, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.raw_p = self._param(torch.log(torch.expm1(torch.tensor(float(p_init)))).item(), keep_fp32=True)
        self.b = self._param(b_init, keep_fp32=True)

    def coefficients(self):
        return {"p": F.softplus(self.raw_p.float()), "b": self.b.float()}

    def _gate_args(self):
        return self.raw_p, None, None


class ExpBGLU(_PGLUFamilyGate):
    """PGLU with the tail exponent FIXED and only the shift b learnable (per expert):
    g'(x) = 1/(1+|x-b|)^p, g(x) = F_p(x-b) - F_p(-b), g(0) = 0, p = the class's `p`.
    Same kernels as ExpPGLU with p fixed; the default init b=1 is exactly the builder
    gate exp_l0m1r0_p{p}_b1. One subclass per exponent (exp_bglu_p{p}), because --exp-gate
    detects learnable gates with isinstance(fn, type)."""
    p = 1.0

    def __init__(self, config=None, b_init=1.0, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.b = self._param(b_init, keep_fp32=True)

    def coefficients(self):
        return {"b": self.b.float()}

    def _gate_args(self):
        return None, float(self.p), None

# exp_bglu_p{p}: the builder power grid {1/2, 1, 2} plus the 3/2 follow-up.
BGLU_GATES = {f"exp_bglu_p{_f(p)}": type(f"ExpBGLU_p{_f(p)}", (ExpBGLU,), {"p": p})
              for p in (0.5, 1.0, 1.5, 2.0)}


# ---- learnable vertical shift c of the gradient (exp_pglu_c, exp_bglu_p{p}_c) ----
# g'(x) = 1/(1+|x-b|)^p + c  =>  g(x) = F_p(x-b) - F_p(-b) + c x,  g(0) = 0.
# c is one unconstrained scalar per expert, init 0, so at init the gate is exactly its parent
# (exp_pglu / exp_bglu_p{p}). For c > 0 the tails have slope -> c instead of 0 (no FP4 dead zone
# once c > 0.5/6); for fixed c the gate is the builder shape (L, M, R) = (c, 1 + c, c).
# Same fused kernels as pglu, with c x added in fp32 before the single rounding (cc in _pglu_fwd / _pglu_bwd).
def _pglu_c_ref(x, p, b, c):
    """Plain-autograd reference of g(x) + c x (self-test / unit tests)."""
    xf = x.to(p.dtype)
    return (_ref_bump_integral(xf - b, p) - _ref_bump_integral(-b, p) + c * xf).to(x.dtype)

class ExpPGLUC(ExpPGLU):
    """ExpPGLU plus a learnable vertical shift c of the gradient (per expert):
    g'(x) = 1/(1+|x-b|)^p + c, g(x) = F_p(x-b) - F_p(-b) + c x. Init c = 0 == exp_pglu."""
    def __init__(self, config=None, p_init=1.0, b_init=1.0, c_init=0.0, num_local_experts=1):
        super().__init__(config, p_init=p_init, b_init=b_init, num_local_experts=num_local_experts)
        self.c = self._param(c_init, keep_fp32=True)
    def coefficients(self):
        return {**super().coefficients(), "c": self.c.float()}
    def _gate_args(self):
        return self.raw_p, None, self.c

class ExpBGLUC(ExpBGLU):
    """ExpBGLU (fixed p, learnable b) plus a learnable vertical shift c of the gradient:
    g'(x) = 1/(1+|x-b|)^p + c, g(x) = F_p(x-b) - F_p(-b) + c x. Init c = 0 == exp_bglu_p{p}."""
    def __init__(self, config=None, b_init=1.0, c_init=0.0, num_local_experts=1):
        super().__init__(config, b_init=b_init, num_local_experts=num_local_experts)
        self.c = self._param(c_init, keep_fp32=True)
    def coefficients(self):
        return {**super().coefficients(), "c": self.c.float()}
    def _gate_args(self):
        return None, float(self.p), self.c

BGLUC_GATES = {f"exp_bglu_p{_f(p)}_c": type(f"ExpBGLUC_p{_f(p)}", (ExpBGLUC,), {"p": p})
               for p in (0.5, 1.0, 1.5, 2.0)}

class ExpBGLUCS(ExpBGLU):
    """ExpBGLUC with the slope shift constrained positive: c = softplus(gamma) > 0 (per expert), so the gate
    is always increasing: g'(x) = 1/(1+|x-b|)^p + softplus(gamma). gamma_init = -4 -> c = 0.018 at init
    (c cannot start at exactly 0; the parent exp_bglu_p{p} is the limit gamma -> -inf)."""
    gamma_init = -4.0
    def __init__(self, config=None, b_init=1.0, gamma_init=None, num_local_experts=1):
        super().__init__(config, b_init=b_init, num_local_experts=num_local_experts)
        self.raw_c = self._param(self.gamma_init if gamma_init is None else gamma_init, keep_fp32=True)
    def coefficients(self):
        return {**super().coefficients(), "c": F.softplus(self.raw_c.float())}
    def _gate_args(self):
        return None, float(self.p), F.softplus(self.raw_c)

BGLUCS_GATES = {f"exp_bglu_p{_f(p)}_cs": type(f"ExpBGLUCS_p{_f(p)}", (ExpBGLUCS,), {"p": p})
                for p in (0.5, 1.0, 1.5, 2.0)}

class ExpXRLGLU(_PerExpertGate):
    """Learnable negative-dip depth alpha (expanded-range trick on RLGLU), per expert: g'
    stretched from (0,1) to (-alpha,1). g(0) = 0; alpha = 0 is exp_l0m0.5r1_p1_b0 (RLGLU)."""
    def __init__(self, config=None, alpha_init=0.3, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.alpha = self._param(alpha_init)
    def coefficients(self):
        return {"alpha": self.alpha.float()}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        return _xrlglu(x, self._rows(self.alpha, x, tokens_per_expert, expert_ids))


# ---- baselines ---------------------------------------------------------------
@jit_fuser
def exp_swiglu(x): return F.silu(x)
@jit_fuser
def exp_situ(x):   return torch.sigmoid(x) * torch.tanh(x)
@jit_fuser
def exp_ssglu(x):  return x * (0.5 + 0.5 * F.softsign(x))


# ---- master registry: name -> callable (or class) ----------------------------
GATES = {name: fn for name, (fn, _) in BUILDER_GATES.items()}
LEARNABLE_GATES = {"exp_pglu": ExpPGLU, "exp_xrlglu": ExpXRLGLU, **BGLU_GATES,
                   "exp_pglu_c": ExpPGLUC, **BGLUC_GATES, **BGLUCS_GATES}
GATES.update(exp_arglu=exp_arglu, **LEARNABLE_GATES,
             exp_swiglu=exp_swiglu, exp_situ=exp_situ, exp_ssglu=exp_ssglu)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    x = torch.linspace(-8, 8, 6001)
    EXISTING = {"exp_l0m1r0_p1_b1", "exp_l0m1r0_p2_b1", "exp_l0m0.5r1_p1_b0",   # lglu/sssglu/rlglu
                "exp_swiglu", "exp_situ", "exp_ssglu"}

    def check(f):                                     # -> (g(0), max|autograd g' - reference|)
        xi = x.clone().requires_grad_(True)
        y = f(xi); (ga,) = torch.autograd.grad(y.sum(), xi)
        return y[len(x) // 2].item(), ga.detach()

    rows, worst = [], 0.0
    worst32 = 0.0     # the pglu family computes in fp32 (fused kernels): fp32-level tolerance below
    for name, (fn, (L, M, R, p, b)) in BUILDER_GATES.items():
        g0, ga = check(fn)
        err = (ga - _grad(x - b, L, M, R, p)).abs().max().item()
        worst = max(worst, err, abs(g0))
        fam = ("bump", "half", "linear")[2 - (L == 0) - (R == 0)]
        rows.append((fam, name, f"({_f(L)},{_f(M)},{_f(R)})", _f(p), str(int(b)),
                     f"{_tail(L, M, p)} / {_tail(R, M, p)}", _growth(L, R, p),
                     " " if name in EXISTING else "*"))
    for name, tail, out in [("exp_arglu", "1/sqrt / 1/x^2", "linear"),
                            ("exp_pglu", "1/x^p learned", "learned"),
                            ("exp_xrlglu", "1/x + dip", "linear"),
                            ("exp_swiglu", "exp / exp", "linear"),
                            ("exp_situ", "exp / exp", "bounded"),
                            ("exp_ssglu", "1/x^2 / 1/x^2", "linear")]:
        rows.append(("extra", name, "-", "-", "-", tail, out, " " if name in EXISTING else "*"))

    # exp_bglu_p{p}: g(0)=0; at init (b=1) == builder exp_l0m1r0_p{p}_b1 and == exp_pglu(p, b=1);
    # autograd d/dx == 1/(1+|x-b|)^p; d/db matches a central finite difference; per-expert rows.
    for name, cls in BGLU_GATES.items():
        p = cls.p
        gate = cls()
        g0, ga = check(lambda t: gate(t))
        ref_fn = lambda t: _builder_gate(t, 0.0, 1.0, 0.0, p, 1.0)
        err = max(abs(g0), (ga - _grad(x - 1.0, 0.0, 1.0, 0.0, p)).abs().max().item(),
                  (gate(x) - ref_fn(x)).abs().max().item(),
                  (gate(x) - ExpPGLU(p_init=p, b_init=1.0)(x)).abs().max().item())
        gate.b.data.fill_(0.7)
        (db,) = torch.autograd.grad(gate(x).sum(), gate.b)
        b64 = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)          # fp64 autograd of the reference
        (fd,) = torch.autograd.grad(_pglu_ref(x.double(), torch.tensor(p, dtype=torch.float64), b64).sum(), b64)
        err = max(err, abs(db.item() - fd.item()) / max(1.0, abs(fd.item())))
        g2 = cls(num_local_experts=2)                 # per-expert mapping via tokens_per_expert
        g2.b.data = torch.tensor([1.0, 0.5])
        xx = torch.linspace(-3, 3, 10).unsqueeze(-1)
        out = g2(xx, tokens_per_expert=[4, 6])
        err = max(err, (out[:4] - cls(b_init=1.0)(xx[:4])).abs().max().item(),
                  (out[4:] - cls(b_init=0.5)(xx[4:])).abs().max().item())
        worst32 = max(worst32, err)
        rows.append(("extra", name, "(0,1,0)", _f(p), "learned", f"1/x^{_f(p)} / 1/x^{_f(p)}",
                     _growth(0.0, 0.0, p), "*"))

    # exp_pglu_c / exp_bglu_p{p}_c: learnable vertical shift c of the gradient.
    #  (1) c = 0 reproduces the parent gate exactly;
    #  (2) autograd d/dx == 1/(1+|x-b|)^p + c   (the gradient we want);
    #  (3) the gate == the numerical integral of that gradient from 0 (cumulative trapezoid), so the
    #      closed form is the right antiderivative, independently of autograd; g(0) = 0;
    #  (4) d/dc == sum(x) exactly; d/db, d/dp (pglu_c) match central finite differences;
    #  (5) per-expert rows via tokens_per_expert and via expert_ids.
    def trapz_cum(y, xs):
        seg = 0.5 * (y[1:] + y[:-1]) * (xs[1:] - xs[:-1])
        out = torch.cat([torch.zeros(1), torch.cumsum(seg, 0)])
        return out - out[len(xs) // 2]                    # anchor the integral at x = 0
    xf = torch.linspace(-40, 40, 400001)                  # fine grid for the quadrature check
    cases = [("exp_pglu_c", ExpPGLUC, p, b, c) for p, b, c in
             ((0.33, 6.37, 0.0), (0.33, 6.37, 0.2), (1.0, 1.0, -0.15), (1.41, 2.56, 0.083), (2.0, -0.7, 0.5))]
    cases += [(n, cls, cls.p, b, c) for n, cls in BGLUC_GATES.items() for b, c in ((1.0, 0.0), (2.89, 0.1))]
    for name, cls, p, b, c in cases:
        if cls is ExpPGLUC:
            gate = cls(p_init=p, b_init=b, c_init=c)
            parent = ExpPGLU(p_init=p, b_init=b)
        else:
            gate = cls(b_init=b, c_init=c)
            parent = BGLU_GATES[name[:-2]](b_init=b)
        g0, ga = check(lambda t: gate(t))
        kern = (1.0 + (x - b).abs()) ** (-p)
        err = max(abs(g0), (ga - (kern + c)).abs().max().item())                       # (2)
        if c == 0.0:
            err = max(err, (gate(x) - parent(x)).abs().max().item())                   # (1)
        else:
            err = max(err, (gate(x) - parent(x) - c * x).abs().max().item())
        with torch.no_grad():                                                           # (3)
            yq = trapz_cum((1.0 + (xf - b).abs()) ** (-p) + c, xf)
            err = max(err, (gate(xf) - yq).abs().max().item() / 10.0)                   # O(h^2) quadrature
        (dc,) = torch.autograd.grad(gate(x).sum(), gate.c)                              # (4)
        err = max(err, abs(dc.item() - x.sum().item()))
        pnames = ["b", "raw_p"] if cls is ExpPGLUC else ["b"]
        for pname in pnames:                                    # vs fp64 autograd of the reference
            (an,) = torch.autograd.grad(gate(x).sum(), getattr(gate, pname))
            v64 = {k: getattr(gate, k).detach().double().clone().requires_grad_(k == pname) for k in pnames}
            pp = F.softplus(v64["raw_p"]) if cls is ExpPGLUC else torch.tensor(p, dtype=torch.float64)
            ref = _pglu_c_ref(x.double(), pp, v64["b"], torch.tensor(c, dtype=torch.float64)).sum()
            (rg,) = torch.autograd.grad(ref, v64[pname])
            err = max(err, abs(an.item() - rg.item()) / max(1.0, abs(rg.item())))
        g2 = (cls(num_local_experts=2, p_init=p) if cls is ExpPGLUC else cls(num_local_experts=2))   # (5)
        g2.b.data = torch.tensor([b, 0.5]); g2.c.data = torch.tensor([c, -0.1])
        one = lambda bb, cc: (cls(p_init=p, b_init=bb, c_init=cc) if cls is ExpPGLUC else cls(b_init=bb, c_init=cc))
        xx = torch.linspace(-3, 3, 10).unsqueeze(-1)
        out = g2(xx, tokens_per_expert=[4, 6])
        ids = torch.tensor([0] * 4 + [1] * 6)
        err = max(err, (out[:4] - one(b, c)(xx[:4])).abs().max().item(),
                  (out[4:] - one(0.5, -0.1)(xx[4:])).abs().max().item(),
                  (g2(xx, expert_ids=ids) - out).abs().max().item())
        worst32 = max(worst32, err)
        print(f"  {name:<16} p={p:<5g} b={b:<5g} c={c:<6g} max err {err:.1e}")
    # exp_bglu_p{p}_cs (c = softplus(gamma)): == exp_bglu_p{p}_c at c = softplus(gamma) (value and d/dx);
    # c > 0 so g' > 0 everywhere (monotone); d/dgamma == sigmoid(gamma) * sum(x); default c(init) = softplus(-4);
    # per-expert rows.
    for name, cls in BGLUCS_GATES.items():
        p = cls.p
        for b, gam in ((1.0, -4.0), (2.89, -2.44), (-0.7, 1.3)):
            gate = cls(b_init=b, gamma_init=gam)
            c = float(F.softplus(torch.tensor(gam)))
            ref = BGLUC_GATES[name[:-1]](b_init=b, c_init=c)
            g0, ga = check(lambda t: gate(t))
            err = max(abs(g0), (gate(x) - ref(x)).abs().max().item(),
                      (ga - ((1.0 + (x - b).abs()) ** (-p) + c)).abs().max().item())
            if ga.min().item() <= 0.0:
                err = max(err, 1.0)                                       # must be monotone
            (dg,) = torch.autograd.grad(gate(x).sum(), gate.raw_c)
            err = max(err, abs(dg.item() - torch.sigmoid(torch.tensor(gam)).item() * x.sum().item()))
            g2 = cls(num_local_experts=2)
            g2.b.data = torch.tensor([b, 0.5]); g2.raw_c.data = torch.tensor([gam, -1.0])
            xx = torch.linspace(-3, 3, 10).unsqueeze(-1)
            out = g2(xx, tokens_per_expert=[4, 6])
            err = max(err, (out[:4] - cls(b_init=b, gamma_init=gam)(xx[:4])).abs().max().item(),
                      (out[4:] - cls(b_init=0.5, gamma_init=-1.0)(xx[4:])).abs().max().item())
            worst32 = max(worst32, err)
            print(f"  {name:<17} p={p:<5g} b={b:<5g} gamma={gam:<6g} c={c:.4f}  max err {err:.1e}")
        rows.append(("extra", name, "(c,1+c,c)", _f(p), "learned", f"1/x^{_f(p)} + softplus", "linear", "*"))
    assert abs(float(F.softplus(BGLUCS_GATES["exp_bglu_p2_cs"]().raw_c)) - 0.018149) < 1e-5
    rows.append(("extra", "exp_pglu_c", "(c,1+c,c)", "learned", "learned", "1/x^p + c learned", "linear (c≠0)", "*"))
    for name, cls in BGLUC_GATES.items():
        rows.append(("extra", name, "(c,1+c,c)", _f(cls.p), "learned", f"1/x^{_f(cls.p)} + c", "linear (c≠0)", "*"))

    n_new = sum(r[-1] == "*" for r in rows)
    print(f"\n  {len(_shapes())} shapes -> {len(BUILDER_GATES)} builder gates + {len(rows) - len(BUILDER_GATES)} extra = {len(rows)} gates"
          f"   ({n_new} new, {len(rows) - n_new} baselines/twins);  all g(0)=0, autograd == g'\n")
    hdr = ("gate", "(L,M,R)", "power", "shift", "'grad tail L/R'", "output")
    cells = lambda r: r[1:7]                          # skip family (r[0]) and new-marker (r[7])
    W = [max(len(str(c)) for c in col) for col in zip(hdr, *map(cells, rows))]
    show = lambda cs: "  ".join(f"{str(v):<{w}}" for v, w in zip(cs, W))
    for fam in ("bump", "half", "linear", "extra"):
        grp = [r for r in rows if r[0] == fam]
        print(f"── {fam}  ({len(grp)})")
        print("   " + show(hdr))
        for r in grp:
            print(f" {r[7]} " + show(cells(r)))
        print()
    print(f"worst |g(0)| or |autograd - g'|: {worst:.1e} (fp64 gates), {worst32:.1e} (pglu family, fp32 kernels)"
          f"  => {'ALL OK' if worst < 1e-6 and worst32 < 2e-5 else 'FAIL'}")
