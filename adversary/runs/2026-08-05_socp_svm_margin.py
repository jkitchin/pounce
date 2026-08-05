"""Adversary cross-check: hard-margin SVM maximum-margin classifier via SOCP
Family: socp   Class: SOC epigraph + linear (nonneg) constraints, symmetric closed-form
Source: Boyd & Vandenberghe, Convex Optimization, Sec 8.6.1 "Linear
    discrimination" / classic maximum-margin SVM (Vapnik). The hard-margin
    SVM is posed as minimizing ||w||_2 (an SOC epigraph, NOT the usual
    1/2||w||^2 QP form) subject to the margin constraints, to exercise
    pounce's actual conic (solve_socp) entry point rather than the QP path
    already covered many times in the log.

Problem: 4 points in R^2, symmetric by construction so the optimal
separating hyperplane and margin are known exactly by inspection:

    class +1: (1, 1), (1, -1)
    class -1: (-1, 1), (-1, -1)

By symmetry the maximum-margin hyperplane is x=0 (w=(1,0), b=0), and every
point lies exactly on its margin (all four are support vectors):
    y_i (w'x_i + b) = 1  for i=1..4

    minimize    t                       (t = ||w||_2, an SOC epigraph)
    subject to  ||w||_2 <= t
                y_i (w'x_i + b) >= 1,  i=1..4

Known optimal: t* = ||w*||_2 = 1  (w*=(1,0), b*=0), all 4 margin
constraints active (verified by direct substitution below).

pounce encoding (solve_socp): variables v=[t, w1, w2, b] (n=4).
cones = [("soc", 3), ("nonneg", 4)]:
  - soc block (dim 3): s=(t, w1, w2), t >= ||(w1,w2)||
      row0: s0 = t         -> G_row = [-1,0,0,0], h=0
      row1: s1 = w1        -> G_row = [0,-1,0,0], h=0
      row2: s2 = w2         -> G_row = [0,0,-1,0], h=0
  - nonneg block (dim 4): s_i = y_i(w'x_i+b) - 1 >= 0
      G_row_i = -[0, y_i*x_i1, y_i*x_i2, y_i],  h_i = -1
"""
import time
import numpy as np

X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
Y = np.array([1.0, 1.0, -1.0, -1.0])

KNOWN_OPTIMAL = 1.0
KNOWN_W = np.array([1.0, 0.0])
KNOWN_B = 0.0
assert np.allclose(Y * (X @ KNOWN_W + KNOWN_B), 1.0), "closed-form check failed"

n = 4  # [t, w1, w2, b]

G_soc = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
])
h_soc = np.zeros(3)

G_margin = np.zeros((4, n))
h_margin = -np.ones(4)
for i in range(4):
    G_margin[i, 1] = -Y[i] * X[i, 0]
    G_margin[i, 2] = -Y[i] * X[i, 1]
    G_margin[i, 3] = -Y[i]

G = np.vstack([G_soc, G_margin])
h = np.concatenate([h_soc, h_margin])
c = np.array([1.0, 0.0, 0.0, 0.0])  # minimize t

cones = [("soc", 3), ("nonneg", 4)]

from pounce.qp import solve_socp

t0 = time.perf_counter()
r = solve_socp(c=c, G=G, h=h, cones=cones)
t_pounce = time.perf_counter() - t0
t_pounce_x, obj_pounce, status = r.x, r.obj, r.status

# --- oracle: cvxpy (CLARABEL + SCS) ---
import cvxpy as cp

v = cp.Variable(n)
t_, w1_, w2_, b_ = v[0], v[1], v[2], v[3]
constraints = [cp.norm(v[1:3], 2) <= t_]
for i in range(4):
    constraints.append(Y[i] * (X[i, 0] * w1_ + X[i, 1] * w2_ + b_) >= 1)
prob = cp.Problem(cp.Minimize(t_), constraints)
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = v.value, prob.value

prob2 = cp.Problem(cp.Minimize(t_), constraints)
prob2.solve(solver=cp.SCS, eps=1e-9)
obj_oracle2 = prob2.value


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


x_pounce = np.asarray(t_pounce_x)
w_pounce = x_pounce[1:3]
b_pounce = x_pounce[3]

obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
w_err_known = float(np.linalg.norm(w_pounce - KNOWN_W, np.inf))
w_err_oracle = float(np.linalg.norm(w_pounce - np.asarray(x_oracle)[1:3], np.inf))
oracle_agree = rel(obj_oracle, obj_oracle2)

print("=== pounce (solve_socp) ===")
print(f"status={status} obj={obj_pounce:.10e} w={w_pounce} b={b_pounce:.10e} t={t_pounce:.4f}s")
print("=== oracle CLARABEL ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"=== oracle SCS === obj={obj_oracle2:.10e} (CLARABEL/SCS agreement {oracle_agree:.2e})")
print(f"known_optimal={KNOWN_OPTIMAL} rel_err_vs_known={obj_err_known:.2e} w_err_vs_known={w_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} w_err_vs_oracle={w_err_oracle:.2e}")

# margin activity check on pounce's own solution
margins = Y * (X @ w_pounce + b_pounce)
print(f"margins at pounce solution (should all be ~1.0): {margins}")

ok = (
    (status in ("optimal",) or getattr(r, "success", False))
    and obj_err_known < 1e-4
    and obj_err_oracle < 1e-4
    and w_err_known < 1e-4
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err_known:.2e})")
