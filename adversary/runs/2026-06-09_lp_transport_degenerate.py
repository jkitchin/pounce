"""Adversary cross-check: degenerate balanced transportation LP (equality constraints)
Family: lp   Class: degenerate equality-constrained LP (transportation)

Source: Classic balanced transportation problem. Transportation problems are
the canonical example of *primal degeneracy*: a balanced m-source/n-sink
problem has m+n equality rows but rank m+n-1, so any basic feasible solution
has m+n-1 basic cells and the rest are degenerate (basic at zero). This one
is deliberately built so the unique optimum has a degenerate vertex.

  2 sources (supply s), 3 sinks (demand d), balanced sum s = sum d.
  Variables x_ij = flow from source i to sink j  (i in 0..1, j in 0..2).
  minimize  sum c_ij x_ij
  subject to  sum_j x_ij = s_i   (supply, equality)
              sum_i x_ij = d_j   (demand, equality)
              x >= 0

Data (degenerate by construction):
  supply  s = [10, 10]                (total 20)
  demand  d = [10, 5, 5]              (total 20)
  cost matrix C =
     [ 1, 2, 3 ]
     [ 4, 2, 1 ]   (row i, col j)

  Northwest/least-cost reasoning gives optimum:
    x00=10 (cost1), then source0 exhausted; source1 covers d1=5 (cost2),
    d2=5 (cost1):  x11=5, x12=5.  d0 already met by x00.
    => x = [[10,0,0],[0,5,5]], obj = 10*1 + 5*2 + 5*1 = 25.
  Note d0=10 == s0=10 forces x00=10 and x01=x02=x10=0 -> several zero
  basic cells -> degenerate vertex. Oracle is the authority on optimum.

One supply (or demand) equality row is redundant in a balanced problem
(rank deficient). We keep ALL equality rows to stress pounce's handling of
a rank-deficient A; pounce's presolve must cope. lb=0, no upper bounds.
"""
import time
import numpy as np

KNOWN_OPTIMAL = 25.0
# x flattened row-major: [x00,x01,x02,x10,x11,x12]
X_STAR = np.array([10.0, 0.0, 0.0, 0.0, 5.0, 5.0])

C = np.array([[1.0, 2.0, 3.0],
              [4.0, 2.0, 1.0]])
s = np.array([10.0, 10.0])
d = np.array([10.0, 5.0, 5.0])
m, n = 2, 3
nv = m * n

c = C.flatten()

# Build equality matrix A x = b : 2 supply rows + 3 demand rows = 5 rows
A_rows = []
b_vals = []
# supply rows: sum_j x_ij = s_i
for i in range(m):
    row = np.zeros(nv)
    for j in range(n):
        row[i * n + j] = 1.0
    A_rows.append(row)
    b_vals.append(s[i])
# demand rows: sum_i x_ij = d_j
for j in range(n):
    row = np.zeros(nv)
    for i in range(m):
        row[i * n + j] = 1.0
    A_rows.append(row)
    b_vals.append(d[j])
A = np.array(A_rows)
b = np.array(b_vals)
lb = np.zeros(nv)

# --- pounce ---
import pounce
t0 = time.perf_counter()
res = pounce.solve_qp(P=None, c=c, A=A, b=b, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = np.asarray(res.x), res.obj, res.status

# --- oracle 1: scipy linprog (A_eq x = b_eq) ---
from scipy.optimize import linprog
t0 = time.perf_counter()
lp = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * nv)
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = lp.x, lp.fun

# --- oracle 2: cvxpy ---
import cvxpy as cp
xv = cp.Variable(nv)
prob = cp.Problem(cp.Minimize(c @ xv), [A @ xv == b, xv >= 0])
prob.solve(solver=cp.CLARABEL)
obj_cvx = prob.value


def rel(a, b_):
    return abs(a - b_) / max(1.0, abs(b_))


obj_err = rel(obj_pounce, obj_oracle)
# x may differ if alternate optima; compare objective primarily.
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

# feasibility of pounce solution
eq_res = float(np.linalg.norm(A @ x_pounce - b, np.inf))
neg = float(x_pounce.min())

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"x={x_pounce}")
print(f"eq_residual(inf)={eq_res:.2e} min_x={neg:.2e}")
print("=== oracle (linprog) ===")
print(f"status={lp.status} obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"x={x_oracle}")
print(f"cvxpy_obj={obj_cvx:.10e}")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err(may differ if alt opt)={x_err:.2e}")

ok = (status == "optimal" or getattr(res, "success", False)) \
    and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4 \
    and eq_res < 1e-5 and neg > -1e-6
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, eq_res={eq_res:.2e})")
