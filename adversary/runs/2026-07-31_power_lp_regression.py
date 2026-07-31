"""Adversary cross-check: L_p regression (p=1.5) via the power cone
Family: power   Class: power-cone epigraph on an affine residual (regression,
        not a bare p-norm-of-x constraint like prior probes)
Source: L_p-norm approximation / regression, minimize ||Ax-b||_p, is a
        standard convex program -- see Boyd & Vandenberghe, "Convex
        Optimization" (2004) Sec 6.1.1 (norm approximation), and the MOSEK
        Modeling Cookbook Sec 3.2.4 "Power cone" for the p-norm-as-power-cone
        reduction: |y_i| <= t_i, t_i^{1/p} * 1^{1-1/p} >= |r_i|/... encoded
        per-residual as (r_i, u_i, r_aux_i) in a 3-D power cone with
        alpha=1/p, plus a shared "1" tied via t = sum(r_aux_i); minimizing
        the epigraph variable t is equivalent to minimizing ||r||_p^p, whose
        minimizer coincides with minimizing ||r||_p for a fixed p.
Known optimal: none published for this instance; oracle-only (cvxpy native
        p-norm atom, which internally uses the power cone via a different
        code path, plus an independent scipy.optimize smooth-NLP solve).
"""
import numpy as np
import time

m, n = 6, 3
p = 1.5
np.random.seed(0)
A = np.array(
    [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, -1.0, 0.5],
        [-1.0, 2.0, 1.0],
        [0.5, 0.5, -1.0],
    ]
)
b = np.array([2.0, 1.0, -1.0, 3.0, 0.5, -0.5])

# --- pounce: solve_socp, minimize sum_i r_aux_i s.t. |r_i| <= r_aux_i^{1/p} * w_i^{1-1/p}
# pounce's pow cone: (x, y, z) with |x| <= y^alpha z^(1-alpha), alpha=1/p.
# Variables: x (n), r (m, residual = Ax - b), u (m, epigraph per-term), w (scalar aux = 1)
from pounce import solve_socp

alpha = 1.0 / p
nvar = n + m + m + 1  # x, r, u, w


def xidx(i):
    return i


def ridx(i):
    return n + i


def uidx(i):
    return n + m + i


def widx():
    return n + m + m


# equality: r = A x - b  ->  r - A x = -b
A_eq = np.zeros((m, nvar))
for i in range(m):
    A_eq[i, ridx(i)] = 1.0
    A_eq[i, :n] = -A[i, :]
b_eq = -b.copy()

# equality: w = 1
A_eq2 = np.zeros((1, nvar))
A_eq2[0, widx()] = 1.0
A_full = np.vstack([A_eq, A_eq2])
b_full = np.concatenate([b_eq, [1.0]])

# G,h,cones: per i, pow cone (r_i, u_i, w) with alpha=1/p: |r_i| <= u_i^alpha * w^(1-alpha)
rows = 3 * m
G = np.zeros((rows, nvar))
h = np.zeros(rows)
cones = []
row = 0
for i in range(m):
    G[row, ridx(i)] = -1.0
    row += 1
    G[row, uidx(i)] = -1.0
    row += 1
    G[row, widx()] = -1.0
    row += 1
    cones.append(("pow", alpha))

c = np.zeros(nvar)
for i in range(m):
    c[uidx(i)] = 1.0  # minimize sum u_i  (== ||r||_p^p up to monotone reparam)

t0 = time.perf_counter()
r = solve_socp(c=c, A=A_full, b=b_full, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
z_pounce = np.asarray(r.x)
x_pounce = z_pounce[:n]
resid_pounce = A @ x_pounce - b
pnorm_pounce = np.sum(np.abs(resid_pounce) ** p) ** (1.0 / p)
status = r.status

# --- oracle 1: cvxpy native p-norm atom (independent code path) ---
import cvxpy as cp

xv = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.pnorm(A @ xv - b, p)))
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle = xv.value
pnorm_oracle = prob.value

# --- oracle 2: scipy SLSQP, smooth unconstrained NLP on ||Ax-b||_p directly ---
from scipy.optimize import minimize


def pnorm_obj(x_):
    return np.sum(np.abs(A @ x_ - b) ** p) ** (1.0 / p)


res = minimize(pnorm_obj, x0=np.zeros(n), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
x_scipy = res.x
pnorm_scipy = res.fun


def rel(a, b_):
    return abs(a - b_) / max(1.0, abs(b_))


err_cvxpy = rel(pnorm_pounce, pnorm_oracle)
err_scipy = rel(pnorm_pounce, pnorm_scipy)

print("=== pounce (power cone) ===")
print(f"status={status} ||r||_p={pnorm_pounce:.10e} t={t_pounce:.4f}s x={x_pounce}")
print("=== oracle: cvxpy CLARABEL (native pnorm atom) ===")
print(f"||r||_p={pnorm_oracle:.10e} t={t_oracle:.4f}s x={x_oracle}")
print("=== oracle: scipy Nelder-Mead (smooth NLP, no cone machinery) ===")
print(f"||r||_p={pnorm_scipy:.10e} success={res.success} x={x_scipy}")
print(f"err_vs_cvxpy={err_cvxpy:.2e} err_vs_scipy={err_scipy:.2e}")

ok = status == "optimal" and err_cvxpy < 1e-4 and err_scipy < 1e-3
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status})")
