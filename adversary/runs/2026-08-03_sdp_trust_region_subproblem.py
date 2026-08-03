"""Adversary cross-check: Trust-region subproblem (TRS) via SDP relaxation
Family: sdp   Class: single dense PSD cone (Shor/moment relaxation), indefinite B
Source: Moré & Sorensen (1983) secular-equation TRS algorithm (closed-form
    optimality conditions for min 0.5 x'Bx + g'x s.t. ||x||<=Delta); tightness
    of the SDP/Shor relaxation for a single ball constraint is the classical
    S-lemma result (no duality gap for one quadratic constraint -- see Boyd &
    Vandenberghe App. B, or Fortin & Wolkowicz 2004 "The trust region
    subproblem and semidefinite programming").
Known optimal: value of the secular-equation TRS solution (below), which the
    S-lemma guarantees equals the SDP relaxation's optimum exactly (B has an
    indefinite spectrum, so the *unrelaxed, nonconvex* TRS is not itself
    convex -- the point of this probe is exactly that the convex SDP
    relaxation still recovers the true nonconvex optimum here).

Problem instance (n=2):
    B = diag(1, -2)   (indefinite: eigenvalues 1, -2)
    g = [1, 1]
    Delta = 1
    minimize   0.5 x'Bx + g'x
    subject to ||x||_2 <= Delta

SDP (Shor) relaxation lifts x -> (x, X) with X ~ xx^T:
    minimize   0.5<B,X> + g'x
    subject to trace(X) <= Delta^2
               [[1, x'],[x, X]] >= 0   (Schur complement <=> X - xx' >= 0)

pounce encoding: decision vector v = [x1, x2, X11, X21, X22] (5 dims).
PSD block is the 3x3 moment matrix M = [[1,x1,x2],[x1,X11,X21],[x2,X21,X22]].
svec (pounce convention): lower triangle, column-major, off-diag * sqrt(2):
    s = [M00, sqrt2*M10, sqrt2*M20, M11, sqrt2*M21, M22]
      = [1,   sqrt2*x1,  sqrt2*x2,  X11, sqrt2*X21,  X22]
so with s = h - G v:  h=[1,0,0,0,0,0], G picks off -sqrt2*x1 etc (see below).
Second cone row: trace(X) = X11+X22 <= Delta^2 (nonneg cone, 1 row).
"""
import time
import numpy as np
from scipy.optimize import brentq

KNOWN_OPTIMAL = None  # computed below from the secular equation (closed form)

B = np.diag([1.0, -2.0])
g = np.array([1.0, 1.0])
Delta = 1.0
n = 2

# --- "known optimal" oracle: Moré-Sorensen secular equation (closed form) ---
# B is diagonal here, so (B + lam*I)^-1 is diagonal too.
# x(lam) = -(B + lam*I)^-1 g, valid/PD for lam > -lambda_min(B) = 2.
lam_min_B = float(np.linalg.eigvalsh(B)[0])  # = -2
lo = max(0.0, -lam_min_B) + 1e-9


def norm_x_of_lambda(lam):
    x = -g / (np.diag(B) + lam)
    return np.linalg.norm(x)


def secular(lam):
    return norm_x_of_lambda(lam) - Delta


# bracket a root above `lo` (standard, non-hard case; g has full support so no
# hard case here)
hi = lo + 1.0
while secular(hi) > 0:
    hi *= 2.0
lam_star = brentq(secular, lo, hi, xtol=1e-14, rtol=1e-14)
x_star = -g / (np.diag(B) + lam_star)
KNOWN_OPTIMAL = 0.5 * x_star @ B @ x_star + g @ x_star

# --- pounce SDP relaxation ---
r2 = np.sqrt(2.0)
N = n + 1  # 3x3 moment matrix
idx = {}
k = 0
for j in range(N):
    for i in range(j, N):
        idx[(i, j)] = k
        k += 1
svec_dim = k  # 6

h = np.zeros(svec_dim)
h[idx[(0, 0)]] = 1.0  # M00 = 1 (constant)

# variable order: v = [x1, x2, X11, X21, X22]
nvar = 2 + 3
G_psd = np.zeros((svec_dim, nvar))
G_psd[idx[(1, 0)], 0] = -r2   # M10 = x1  -> s = h - G v = r2*x1  => G[.,0] = -r2
G_psd[idx[(2, 0)], 1] = -r2   # M20 = x2
G_psd[idx[(1, 1)], 2] = -1.0  # M11 = X11
G_psd[idx[(2, 1)], 3] = -r2   # M21 = X21
G_psd[idx[(2, 2)], 4] = -1.0  # M22 = X22

# trace(X) = X11 + X22 <= Delta^2
G_trace = np.array([[0.0, 0.0, 1.0, 0.0, 1.0]])
h_trace = np.array([Delta**2])

G = np.vstack([G_psd, G_trace])
h_full = np.concatenate([h, h_trace])
cones = [("psd", N), ("nonneg", 1)]

# objective: 0.5<B,X> + g'x = 0.5*(B11*X11 + 2*B12*X21 + B22*X22) + g1*x1 + g2*x2
c = np.array([g[0], g[1], 0.5 * B[0, 0], B[0, 1], 0.5 * B[1, 1]])

import pounce

t0 = time.perf_counter()
r = pounce.solve_socp(c=c, G=G, h=h_full, cones=cones)
t_pounce = time.perf_counter() - t0
v = np.asarray(r.x, dtype=float)
x_p = v[:2]
X_p = np.array([[v[2], v[3]], [v[3], v[4]]])
obj_pounce = float(r.obj)
status = str(r.status)

# rank-1 tightness check: X should equal x x' at the (tight) optimum
rank1_gap = float(np.linalg.norm(X_p - np.outer(x_p, x_p), ord="fro"))
eig_M = np.linalg.eigvalsh(np.block([[np.array([[1.0]]), x_p.reshape(1, 2)],
                                      [x_p.reshape(2, 1), X_p]]))
min_eig_M = float(eig_M.min())

# --- independent oracle: cvxpy building the identical SDP ---
import cvxpy as cp

Xc = cp.Variable((2, 2), symmetric=True)
xc = cp.Variable(2)
M = cp.bmat([[np.array([[1.0]]), cp.reshape(xc, (1, 2))],
             [cp.reshape(xc, (2, 1)), Xc]])
constraints = [M >> 0, cp.trace(Xc) <= Delta**2]
obj_c = 0.5 * cp.trace(B @ Xc) + g @ xc
prob = cp.Problem(cp.Minimize(obj_c), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
obj_oracle = float(prob.value)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_p - x_star, np.inf))

print("=== pounce (SDP/PSD-cone relaxation of TRS) ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"x*={x_p} rank1_gap(X - xx')={rank1_gap:.2e} min_eig(moment matrix)={min_eig_M:.2e}")
print("=== oracle: cvxpy CLARABEL (identical SDP) ===")
print(f"obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print("=== known optimal: Moré-Sorensen secular equation (closed form) ===")
print(f"lambda*={lam_star:.10e} x*={x_star} known_optimal={KNOWN_OPTIMAL:.10e}")
print(f"obj_err_vs_known={obj_err_known:.2e} obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err_vs_known={x_err:.2e}")

ok = (
    status in ("optimal", "optimal_inaccurate")
    and obj_err_known < 1e-4
    and obj_err_oracle < 1e-4
    and min_eig_M > -1e-6
    and rank1_gap < 1e-3
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, err_known={obj_err_known:.2e}, "
      f"err_oracle={obj_err_oracle:.2e}, rank1_gap={rank1_gap:.2e})")
