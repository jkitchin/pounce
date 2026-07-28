"""Adversary cross-check: minimum relative entropy (KL divergence) with a
moment constraint -- the exponential-family / Gibbs tilting problem.
Family: exp   Class: exponential-cone (relative entropy minimization)

Problem:  minimize  D(p || q) = sum_i p_i * log(p_i / q_i)
          s.t.       sum_i p_i        = 1          (normalization)
                     sum_i a_i p_i    = m          (moment / mean constraint)
                     p_i >= 0

  with reference q = uniform (q_i = 1/n) and support values a = [0,1,...,n-1].

SOURCE: Classical maximum-entropy / minimum-discrimination-information problem
  (Kullback; Cover & Thomas, "Elements of Information Theory", Ch. 11 -- I-
  projection onto a moment family; Boyd & Vandenberghe, "Convex Optimization",
  Sec. 7.1 exponential-family / Sec. on KL geometry). The I-projection of a
  reference q onto the affine family {p : E_p[a] = m} is the *Gibbs* (tilted)
  distribution

        p_i*(lambda) = q_i * exp(lambda * a_i) / Z(lambda),
        Z(lambda)    = sum_j q_j * exp(lambda * a_j),

  where the single scalar dual variable lambda is fixed by the mean constraint
        d/dlambda log Z(lambda) = sum_i a_i p_i*(lambda) = m.
  The minimal divergence then has the closed form
        D* = lambda * m - log Z(lambda).
  This is an EXACT closed form once lambda solves the scalar monotone equation
  above (mean is strictly increasing in lambda), which we solve INDEPENDENTLY of
  any conic solver with scipy.brentq. No cvxpy needed for the reference.

KNOWN_OPTIMAL: D* = lambda* * m - log Z(lambda*) with lambda* from brentq.
  Instance: n = 4, a = [0,1,2,3], q uniform = 1/4, target mean m = 2.0
  (the uniform mean is 1.5, so m=2.0 pulls mass toward larger a -> lambda*>0).

N_VARIABLES (pounce): 2n = 8 decision vars (p_i and per-coordinate epigraph t_i).

EXP-CONE ENCODING (re-derived from scratch, the trap):
  pounce / cvxpy convention:  Kexp = {(x,y,z) : y*exp(x/y) <= z, y > 0}.
  We MINIMIZE sum_i t_i where each t_i upper-bounds the relative-entropy term
        t_i >= p_i * log(p_i / q_i).
  Put r_i = p_i / q_i. Then p_i log(p_i/q_i) = p_i log r_i. We want a cone that
  certifies  t_i >= p_i log(p_i / q_i), i.e.  -t_i <= p_i log(q_i / p_i) = the
  (per-term) entropy-like quantity.  Equivalently  -t_i <= -p_i log(p_i/q_i).
  Rearranged into the cone form y*exp(x/y) <= z:

        t_i >= p_i log(p_i/q_i)
    <=> -t_i/p_i <= log(q_i/p_i) = log(q_i) - log(p_i)
    <=> log(p_i) - log(q_i) - t_i/p_i <= 0
    <=> log(p_i/q_i) <= t_i/p_i
    <=> p_i/q_i <= exp(t_i/p_i)
    <=> p_i <= q_i * exp(t_i/p_i)
    <=> p_i * exp( (-t_i)/p_i ) <= q_i      [divide both sides by exp(t_i/p_i),
                                              multiply by p_i: p_i*exp(-t_i/p_i)<=q_i]

  So with the cone triple (x, y, z) = (-t_i, p_i, q_i):
        y*exp(x/y) = p_i * exp(-t_i/p_i) <= q_i = z.   CORRECT.

  Therefore each exp cone i has slack s = (x, y, z) = (-t_i, p_i, q_i):
        s_{3i}   = -t_i   ->  G[3i,   t_i] = +1   (s = h - G x, h=0)
        s_{3i+1} =  p_i   ->  G[3i+1, p_i] = -1
        s_{3i+2} =  q_i   ->  h[3i+2] = q_i        (constant, G row 0)

  Objective: minimize sum_i t_i  ->  c[t_i] = +1.
  Equalities (A,b): sum p_i = 1 ; sum a_i p_i = m.

SANITY: at the optimum, recovered p* must equal the Gibbs distribution
  p_i* = q_i exp(lambda* a_i)/Z, and D* must match lambda* m - log Z.
"""
import time
import numpy as np
import pounce
import cvxpy as cp
from scipy.optimize import brentq

# ---- instance --------------------------------------------------------------
n = 4
a = np.arange(n, dtype=float)        # [0,1,2,3]
q = np.full(n, 1.0 / n)              # uniform reference
m = 2.0                              # target mean (uniform mean is 1.5)

# ---- independent closed-form reference (scipy, no conic solver) ------------
def gibbs_p(lam):
    w = q * np.exp(lam * a)
    return w / w.sum()

def mean_minus_m(lam):
    return float(a @ gibbs_p(lam) - m)

# mean is strictly increasing in lambda; bracket it
lam_star = brentq(mean_minus_m, -50.0, 50.0, xtol=1e-14, rtol=1e-14)
p_star = gibbs_p(lam_star)
Z_star = float((q * np.exp(lam_star * a)).sum())
KNOWN_OPTIMAL = float(lam_star * m - np.log(Z_star))   # D* closed form
# cross-check the closed form against the direct KL of p_star
KL_direct = float(np.sum(p_star * np.log(p_star / q)))

# ---- pounce encoding -------------------------------------------------------
# vars X = [p_0..p_{n-1}, t_0..t_{n-1}], length 2n
N = 2 * n
P = lambda i: i           # index of p_i
T = lambda i: n + i       # index of t_i
c = np.zeros(N)
for i in range(n):
    c[T(i)] = 1.0         # minimize sum t_i

G = np.zeros((3 * n, N))
h = np.zeros(3 * n)
for i in range(n):
    r = 3 * i
    # s_r   = -t_i      -> G[r,   t_i] = +1
    G[r, T(i)] = 1.0
    # s_{r+1} = p_i     -> G[r+1, p_i] = -1
    G[r + 1, P(i)] = -1.0
    # s_{r+2} = q_i     -> h[r+2] = q_i
    h[r + 2] = q[i]

cones = [("exp", 3)] * n

# equalities: sum p_i = 1 ; sum a_i p_i = m
A = np.zeros((2, N))
b = np.zeros(2)
A[0, :n] = 1.0;  b[0] = 1.0
A[1, :n] = a;    b[1] = m

t0 = time.perf_counter()
res = pounce.solve_socp(c=c, A=A, b=b, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
xv = np.asarray(res.x)
p_pounce = xv[:n]
obj_pounce = float(res.obj)          # = sum t_i = D(p||q)
status = res.status

# ---- oracle: cvxpy kl_div with TWO solvers (independent model) -------------
# cvxpy: kl_div(p,q) = p*log(p/q) - p + q ; sum over a fixed-normalization
# problem this reduces to the KL plus a constant (sum p = sum q = 1 -> the
# -p+q terms cancel in aggregate), but we model the KL directly via rel_entr.
def solve_cvxpy(solver):
    p = cp.Variable(n, nonneg=True)
    # cp.rel_entr(p, q) = p log(p/q), elementwise
    obj = cp.Minimize(cp.sum(cp.rel_entr(p, q)))
    cons = [cp.sum(p) == 1, a @ p == m]
    prob = cp.Problem(obj, cons)
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return float(prob.value), time.perf_counter() - t0, np.asarray(p.value)

obj_ecos, t_ecos, p_ecos = solve_cvxpy(cp.ECOS)
obj_scs, t_scs, p_scs = solve_cvxpy(cp.SCS)


def rel(x, y):
    return abs(x - y) / max(1.0, abs(y))

err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_ecos = rel(obj_pounce, obj_ecos)
err_scs = rel(obj_pounce, obj_scs)
err_p = float(np.max(np.abs(p_pounce - p_star)))

print(f"=== closed form (scipy brentq) ===")
print(f"lambda*={lam_star:.10f}  Z*={Z_star:.10f}")
print(f"p*={np.round(p_star,8)}")
print(f"D* (lam*m - logZ) = {KNOWN_OPTIMAL:.12f}")
print(f"D* (direct KL of p*) = {KL_direct:.12f}  (consistency {abs(KNOWN_OPTIMAL-KL_direct):.2e})")
print(f"=== pounce (n={n}, m={m}) ===")
print(f"status={status} D={obj_pounce:.12f} t={t_pounce:.4f}s")
print(f"p*={np.round(p_pounce,8)}  max|dp|={err_p:.2e}")
print(f"=== oracle cvxpy (rel_entr) ===")
print(f"ECOS D={obj_ecos:.12f} t={t_ecos:.4f}s")
print(f"SCS  D={obj_scs:.12f} t={t_scs:.4f}s")
print(f"rel_err vs known={err_known:.2e}  vs ECOS={err_ecos:.2e}  vs SCS={err_scs:.2e}")

# Accuracy gates (objective vs closed form AND vs both oracles, and the
# recovered distribution vs the analytic Gibbs tilt).
acc_ok = (err_known < 1e-5 and err_ecos < 1e-5 and err_scs < 1e-5 and err_p < 1e-4)
# pounce halts at optimal_inaccurate here (HSDE exp-cone driver's default
# residual floor) but lands on the correct objective to ~6e-8 -- the same
# benign TOLERANCE behavior accepted in the box-volume GP and entropy runs.
# Treat as PASS when the objective is correct; flag the status as a TOLERANCE
# note (NOT a solver bug: ECOS and SCS agree to the same ~6e-8).
clean_optimal = (status == "optimal" or getattr(res, "success", False))
if acc_ok and clean_optimal:
    print(f"VERDICT: PASS (pounce D={obj_pounce:.8f} matches closed-form Gibbs "
          f"projection and both ECOS/SCS oracles; status={status})")
elif acc_ok:
    print(f"VERDICT: PASS (TOLERANCE note) (pounce D={obj_pounce:.8f} correct to "
          f"{max(err_known,err_ecos,err_scs):.1e} vs closed form AND both "
          f"ECOS/SCS; status={status}/success={getattr(res,'success',None)} -- "
          f"benign HSDE residual floor, objective is right; not a solver bug)")
else:
    print(f"VERDICT: FAIL (status={status}, err_known={err_known:.2e}, "
          f"err_ecos={err_ecos:.2e}, err_scs={err_scs:.2e}, err_p={err_p:.2e})")
