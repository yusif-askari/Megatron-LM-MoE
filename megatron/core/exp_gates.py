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


# ---- learnable gates ---------------------------------------------------------
# One coefficient set per local expert (num_local_experts=1 for dense / shared MLPs, and for
# standalone use). forward(x, tokens_per_expert) maps each expert's coefficients onto its rows
# of the expert-concatenated [N, F] input; forward(x, expert_ids=ids) does the same from
# explicit row ids (used by --log-glu-stats). Math runs in (at least) fp32, output in x.dtype.
def _exprel(z):
    """(e^z - 1)/z with its Taylor series near 0, so value and d/dz stay finite at z = 0."""
    small = z.abs() < 1e-3
    zs = torch.where(small, torch.ones_like(z), z)
    return torch.where(small, 1.0 + z * (0.5 + z * (1.0 / 6.0 + z / 24.0)), torch.expm1(zs) / zs)

def _log1p_rel(a):
    """ln(1+a)/a for a >= 0, with its Taylor series near 0 (value 1 at a = 0)."""
    small = a < 1e-4
    as_ = torch.where(small, torch.ones_like(a), a)
    return torch.where(small, 1.0 - a * (0.5 - a / 3.0), torch.log1p(as_) / as_)

def _bump_integral(u, p):
    """F_p(u) = sign(u) ((1+|u|)^(1-p) - 1)/(1-p)  (sign(u) ln(1+|u|) at p = 1), written as
    sign(u) l exprel((1-p) l) with l = ln(1+|u|): smooth in p, so dF/dp is exact at p = 1.
    sign(u) l is computed as u * (l/|u|): same value, but autograd gives F_p'(0) = 1 (with
    sign(u) * log1p(|u|) both factors have derivative 0 at u = 0, so dF/du came out 0 at x = b)."""
    a = u.abs()
    l = torch.log1p(a)
    return u * _log1p_rel(a) * _exprel((1.0 - p) * l)

@jit_fuser
def _pglu(x, p, b):
    xf = x.to(p.dtype)
    return (_bump_integral(xf - b, p) - _bump_integral(-b, p)).to(x.dtype)

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

    def _param(self, value):
        t = nn.Parameter(torch.full((self.num_local_experts,), float(value)))
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

class ExpPGLU(_PerExpertGate):
    """Learnable tail exponent p = softplus(rho) > 0 and shift b (per expert):
    g'(x) = 1/(1+|x-b|)^p, g(x) = F_p(x-b) - F_p(-b), g(0) = 0. The default init (p=1, b=1)
    is exactly exp_l0m1r0_p1_b1."""
    def __init__(self, config=None, p_init=1.0, b_init=1.0, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.raw_p = self._param(torch.log(torch.expm1(torch.tensor(float(p_init)))).item())
        self.b = self._param(b_init)
    def coefficients(self):
        return {"p": F.softplus(self.raw_p.float()), "b": self.b.float()}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        p = F.softplus(self._rows(self.raw_p, x, tokens_per_expert, expert_ids))
        return _pglu(x, p, self._rows(self.b, x, tokens_per_expert, expert_ids))

class ExpBGLU(_PerExpertGate):
    """PGLU with the tail exponent FIXED and only the shift b learnable (per expert):
    g'(x) = 1/(1+|x-b|)^p, g(x) = F_p(x-b) - F_p(-b), g(0) = 0, p = the class's `p`.
    Same math as ExpPGLU (_pglu) with p frozen; the default init b=1 is exactly the builder
    gate exp_l0m1r0_p{p}_b1. One subclass per exponent (exp_bglu_p{p}), because --exp-gate
    detects learnable gates with isinstance(fn, type)."""
    p = 1.0
    def __init__(self, config=None, b_init=1.0, num_local_experts=1):
        super().__init__(config, num_local_experts)
        self.b = self._param(b_init)
    def coefficients(self):
        return {"b": self.b.float()}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        b = self._rows(self.b, x, tokens_per_expert, expert_ids)
        return _pglu(x, torch.full_like(b, self.p), b)

# exp_bglu_p{p}: the builder power grid {1/2, 1, 2} plus the 3/2 follow-up.
BGLU_GATES = {f"exp_bglu_p{_f(p)}": type(f"ExpBGLU_p{_f(p)}", (ExpBGLU,), {"p": p})
              for p in (0.5, 1.0, 1.5, 2.0)}


# ---- learnable vertical shift c of the gradient (exp_pglu_c, exp_bglu_p{p}_c) ----
# g'(x) = 1/(1+|x-b|)^p + c  =>  g(x) = F_p(x-b) - F_p(-b) + c x,  g(0) = 0.
# c is one unconstrained scalar per expert, init 0, so at init the gate is exactly its parent
# (exp_pglu / exp_bglu_p{p}). For c > 0 the tails have slope -> c instead of 0 (no FP4 dead zone
# once c > 0.5/6); for fixed c the gate is the builder shape (L, M, R) = (c, 1 + c, c).
@jit_fuser
def _pglu_c(x, p, b, c):
    xf = x.to(p.dtype)
    return (_bump_integral(xf - b, p) - _bump_integral(-b, p) + c * xf).to(x.dtype)

class ExpPGLUC(ExpPGLU):
    """ExpPGLU plus a learnable vertical shift c of the gradient (per expert):
    g'(x) = 1/(1+|x-b|)^p + c, g(x) = F_p(x-b) - F_p(-b) + c x. Init c = 0 == exp_pglu."""
    def __init__(self, config=None, p_init=1.0, b_init=1.0, c_init=0.0, num_local_experts=1):
        super().__init__(config, p_init=p_init, b_init=b_init, num_local_experts=num_local_experts)
        self.c = self._param(c_init)
    def coefficients(self):
        return {**super().coefficients(), "c": self.c.float()}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        p = F.softplus(self._rows(self.raw_p, x, tokens_per_expert, expert_ids))
        return _pglu_c(x, p, self._rows(self.b, x, tokens_per_expert, expert_ids),
                       self._rows(self.c, x, tokens_per_expert, expert_ids))

class ExpBGLUC(ExpBGLU):
    """ExpBGLU (fixed p, learnable b) plus a learnable vertical shift c of the gradient:
    g'(x) = 1/(1+|x-b|)^p + c, g(x) = F_p(x-b) - F_p(-b) + c x. Init c = 0 == exp_bglu_p{p}."""
    def __init__(self, config=None, b_init=1.0, c_init=0.0, num_local_experts=1):
        super().__init__(config, b_init=b_init, num_local_experts=num_local_experts)
        self.c = self._param(c_init)
    def coefficients(self):
        return {**super().coefficients(), "c": self.c.float()}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        b = self._rows(self.b, x, tokens_per_expert, expert_ids)
        return _pglu_c(x, torch.full_like(b, self.p), b,
                       self._rows(self.c, x, tokens_per_expert, expert_ids))

BGLUC_GATES = {f"exp_bglu_p{_f(p)}_c": type(f"ExpBGLUC_p{_f(p)}", (ExpBGLUC,), {"p": p})
               for p in (0.5, 1.0, 1.5, 2.0)}

class ExpBGLUCS(ExpBGLU):
    """ExpBGLUC with the slope shift constrained positive: c = softplus(gamma) > 0 (per expert), so the gate
    is always increasing: g'(x) = 1/(1+|x-b|)^p + softplus(gamma). gamma_init = -4 -> c = 0.018 at init
    (c cannot start at exactly 0; the parent exp_bglu_p{p} is the limit gamma -> -inf)."""
    gamma_init = -4.0
    def __init__(self, config=None, b_init=1.0, gamma_init=None, num_local_experts=1):
        super().__init__(config, b_init=b_init, num_local_experts=num_local_experts)
        self.raw_c = self._param(self.gamma_init if gamma_init is None else gamma_init)
    def coefficients(self):
        return {**super().coefficients(), "c": F.softplus(self.raw_c.float())}
    def forward(self, x, tokens_per_expert=None, expert_ids=None):
        b = self._rows(self.b, x, tokens_per_expert, expert_ids)
        c = F.softplus(self._rows(self.raw_c, x, tokens_per_expert, expert_ids))
        return _pglu_c(x, torch.full_like(b, self.p), b, c)

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
        h = 1e-6
        fd = ((_pglu(x, torch.tensor(p), torch.tensor(0.7 + h)).sum()
               - _pglu(x, torch.tensor(p), torch.tensor(0.7 - h)).sum()) / (2 * h))
        err = max(err, abs(db.item() - fd.item()) / max(1.0, abs(fd.item())))
        g2 = cls(num_local_experts=2)                 # per-expert mapping via tokens_per_expert
        g2.b.data = torch.tensor([1.0, 0.5])
        xx = torch.linspace(-3, 3, 10).unsqueeze(-1)
        out = g2(xx, tokens_per_expert=[4, 6])
        err = max(err, (out[:4] - cls(b_init=1.0)(xx[:4])).abs().max().item(),
                  (out[4:] - cls(b_init=0.5)(xx[4:])).abs().max().item())
        worst = max(worst, err)
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
        h = 1e-6
        for pname in (["b", "raw_p"] if cls is ExpPGLUC else ["b"]):
            prm = getattr(gate, pname)
            (an,) = torch.autograd.grad(gate(x).sum(), prm)
            with torch.no_grad():
                prm += h; up = gate(x).sum().item()
                prm -= 2 * h; dn = gate(x).sum().item()
                prm += h
            err = max(err, abs(an.item() - (up - dn) / (2 * h)) / max(1.0, abs(an.item())))
        g2 = (cls(num_local_experts=2, p_init=p) if cls is ExpPGLUC else cls(num_local_experts=2))   # (5)
        g2.b.data = torch.tensor([b, 0.5]); g2.c.data = torch.tensor([c, -0.1])
        one = lambda bb, cc: (cls(p_init=p, b_init=bb, c_init=cc) if cls is ExpPGLUC else cls(b_init=bb, c_init=cc))
        xx = torch.linspace(-3, 3, 10).unsqueeze(-1)
        out = g2(xx, tokens_per_expert=[4, 6])
        ids = torch.tensor([0] * 4 + [1] * 6)
        err = max(err, (out[:4] - one(b, c)(xx[:4])).abs().max().item(),
                  (out[4:] - one(0.5, -0.1)(xx[4:])).abs().max().item(),
                  (g2(xx, expert_ids=ids) - out).abs().max().item())
        worst = max(worst, err)
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
            worst = max(worst, err)
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
    print(f"worst |g(0)| or |autograd - g'|: {worst:.1e}  => {'ALL OK' if worst < 1e-6 else 'FAIL'}")
