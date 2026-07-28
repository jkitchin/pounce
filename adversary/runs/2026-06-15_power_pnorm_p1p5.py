"""Adversary cross-check: p-norm minimization via the 3-D power cone, p = 1.5.
Family: power   Class: p-norm minimization over a linear equality.

DISTINCT from the prior p=3 run: here p=1.5 (in (1,2)), so q=3 and the
bounded geom-mean exponent is alpha = 1/p = 2/3 (the p=3 run had alpha=1/3).
This exercises the power cone in a different regime (the bounded slack carries
the LARGER exponent, 2/3), and a different Holder conjugate q=3.

Problem:
  minimize ||x||_{1.5}  s.t.  a^T x = b.
Holder closed form: for fixed a^T x = b, min ||x||_p = |b| / ||a||_q,
1/p + 1/q = 1. With p=1.5, q = p/(p-1) = 3.  min = |b| / ||a||_3.
The minimizer (Holder equality case) is
  x_i* = sign-consistent with a_i, with |x_i| proportional to |a_i|^{q-1};
  concretely x_i* = (b/||a||_q^q) * sign(a_i) * |a_i|^{q-1}.
This is exact and solver-independent (Holder's inequality, tight iff
|x_i|^p proportional to |a_i|^q).

Power-cone encoding of ||x||_p <= t:
  exists z_i >= 0, sum_i z_i = t, |x_i|^p <= z_i * t^{p-1}
  <=> |x_i| <= z_i^{1/p} t^{1-1/p}.
  pounce cone {(s0,s1,s2): |s0| <= s1^alpha s2^{1-alpha}, s1,s2>=0}, FIRST
  slack s0 is the BOUNDED side.  Map s0 = x_i, s1 = z_i, s2 = t,
  alpha = 1/p = 2/3:  |x_i| <= z_i^{2/3} t^{1/3} = z_i^{1/p} t^{1-1/p}. OK.

Variables: x (n), z (n), t (1)  -> 2n+1 vars.
Cones: n power cones (one per coord), each triple (x_i, z_i, t), alpha=2/3.
Equalities: a^T x = b ; sum_i z_i - t = 0.
"""
import time
import numpy as np

p = 1.5
q = p / (p - 1.0)            # = 3.0
a = np.array([1.0, -2.0, 0.5, 3.0, -1.5])
b = 4.0
n = a.size

KNOWN_OPTIMAL = abs(b) / np.linalg.norm(a, q)
# Holder-tight minimizer: |x_i|^p ~ |a_i|^q, sign(x_i)=sign(a_i*b)
aq = np.abs(a) ** q
x_star = (b / aq.sum()) * np.sign(a) * np.abs(a) ** (q - 1.0)

# variable layout: [x0..x_{n-1}, z0..z_{n-1}, t]
nv = 2 * n + 1
ix = lambda i: i
iz = lambda i: n + i
it = 2 * n

# objective: minimize t
c = np.zeros(nv)
c[it] = 1.0

# Inequality / cone block G x <= h, slack s = h - G x lives in cones.
# For each coord i: power cone slack (s0,s1,s2) = (x_i, z_i, t).
#   s0 = x_i  => s = h - G x = x_i => G row = -e_{x_i}, h=0
#   s1 = z_i  => G row = -e_{z_i}, h=0
#   s2 = t    => G row = -e_t,     h=0
rows = []
h = []
for i in range(n):
    r0 = np.zeros(nv); r0[ix(i)] = -1.0; rows.append(r0); h.append(0.0)  # x_i
    r1 = np.zeros(nv); r1[iz(i)] = -1.0; rows.append(r1); h.append(0.0)  # z_i
    r2 = np.zeros(nv); r2[it] = -1.0;    rows.append(r2); h.append(0.0)  # t
G = np.array(rows)
h = np.array(h)
cones = [("pow", 1.0 / p)] * n   # alpha = 2/3

# Equalities A x = bvec
A = np.zeros((2, nv))
A[0, ix(0):ix(0) + n] = a          # a^T x = b
A[1, n:2 * n] = 1.0                # sum z_i
A[1, it] = -1.0                    # - t = 0
bvec = np.array([b, 0.0])

# --- pounce conic IPM ---
import pounce
t0 = time.perf_counter()
r = pounce.solve_socp(c=c, A=A, b=bvec, G=G, h=h, cones=cones, tol=1e-10)
t_pounce = time.perf_counter() - t0
xv = np.asarray(r.x)
obj_pounce = r.obj
status = r.status
x_sol = xv[:n]

# --- oracle: cvxpy pnorm, two solvers ---
import cvxpy as cp


def solve_cvxpy(solver):
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.pnorm(x, p)), [a @ x == b])
    t0 = time.perf_counter()
    prob.solve(solver=solver)
    return prob.value, time.perf_counter() - t0, np.asarray(x.value)


val_scs, t_scs, x_scs = solve_cvxpy(cp.SCS)
val_cla, t_cla, x_cla = solve_cvxpy(cp.CLARABEL)


def rel(a_, b_):
    return abs(a_ - b_) / max(1.0, abs(b_))


err_known = rel(obj_pounce, KNOWN_OPTIMAL)
err_scs = rel(obj_pounce, val_scs)
err_cla = rel(obj_pounce, val_cla)

# verify feasibility of pounce solution
feas_eq = abs(a @ x_sol - b)
normp = np.linalg.norm(x_sol, p)
err_x = float(np.max(np.abs(x_sol - x_star)))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"  x={x_sol}")
print(f"  x*={x_star}")
print(f"  ||x||_1.5={normp:.10e}  a^T x - b = {feas_eq:.2e}  x_err={err_x:.2e}")
print("=== oracle cvxpy/SCS ===")
print(f"obj={val_scs:.10e} t={t_scs:.4f}s x={x_scs}")
print("=== oracle cvxpy/CLARABEL ===")
print(f"obj={val_cla:.10e} t={t_cla:.4f}s x={x_cla}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"rel_err vs_known={err_known:.2e}  vs_SCS={err_scs:.2e}  vs_CLARABEL={err_cla:.2e}")

ok = ((status in ("optimal", "optimal_inaccurate") or r.success)
      and err_known < 1e-5
      and err_scs < 1e-4 and err_cla < 1e-4
      and feas_eq < 1e-6)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={err_known:.2e}, "
      f"feas={feas_eq:.2e}, err_scs={err_scs:.2e}, err_cla={err_cla:.2e})")
