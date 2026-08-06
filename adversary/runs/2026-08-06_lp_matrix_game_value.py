"""Adversary cross-check: value of a 2x2 zero-sum matrix game via LP
Family: lp   Class: minimax / two-person zero-sum game LP (von Neumann)
Source: von Neumann minimax theorem; standard LP formulation of matrix
        games, e.g. Chvatal, "Linear Programming" (1983), Ch. 15; Dantzig,
        "Linear Programming and Extensions" Ch. 24.
        Payoff matrix to the row player:
            A = [[ 2, -1],
                 [-1,  1]]
        No saddle point in pure strategies (maximin = -1 != minimax = 1),
        so the value is achieved by mixed strategies. For a 2x2 game the
        value has the closed form
            v* = (a11 a22 - a12 a21) / (a11 + a22 - a12 - a21)
        Row player's LP:  max v  s.t.  v <= p . A[:,j]  for each column j,
                                       sum(p) = 1, p >= 0
Known optimal: v* = (2*1 - (-1)*(-1)) / (2 + 1 - (-1) - (-1)) = 1/5 = 0.2
               p* = ((a22-a21)/D, (a11-a12)/D) = (2/5, 3/5), D = 5
"""
import time
import numpy as np
from scipy.optimize import linprog

A = np.array([[2.0, -1.0], [-1.0, 1.0]])
m = 2  # row player strategies
D = A[0, 0] + A[1, 1] - A[0, 1] - A[1, 0]
KNOWN_V = (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / D
KNOWN_P = np.array([(A[1, 1] - A[1, 0]) / D, (A[0, 0] - A[0, 1]) / D])

# Row player's LP: variables x = (p_0, p_1, v). maximize v
# <=>  minimize -v
# s.t.  for each column j: -A[:,j].p + v <= 0   (i.e. v <= sum_i p_i A[i,j])
#       sum p_i = 1,  p_i >= 0,  v free
n = m + 1  # p_0, p_1, v
c = np.zeros(n)
c[-1] = -1.0  # minimize -v == maximize v

G = np.zeros((2, n))  # 2 columns -> 2 inequality rows
for j in range(2):
    G[j, :m] = -A[:, j]
    G[j, -1] = 1.0
h = np.zeros(2)

A_eq = np.zeros((1, n))
A_eq[0, :m] = 1.0
b_eq = np.array([1.0])

lb = np.array([0.0, 0.0, -np.inf])
ub = np.array([np.inf, np.inf, np.inf])

# --- pounce ---
from pounce import solve_qp
t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A_eq, b=b_eq, G=G, h=h, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = r.x, r.obj, r.status
v_pounce = x_pounce[-1]
p_pounce = x_pounce[:m]

# --- oracle: scipy.optimize.linprog (HiGHS) ---
t0 = time.perf_counter()
res = linprog(c, A_ub=G, b_ub=h, A_eq=A_eq, b_eq=b_eq,
              bounds=[(0, None), (0, None), (None, None)], method="highs")
t_oracle = time.perf_counter() - t0
v_oracle = res.x[-1]
p_oracle = res.x[:m]
obj_oracle = res.fun


def rel(u, v):
    return abs(u - v) / max(1.0, abs(v))


v_err_known = rel(v_pounce, KNOWN_V)
v_err_oracle = rel(v_pounce, v_oracle)
p_err = float(np.linalg.norm(p_pounce - KNOWN_P, np.inf))
obj_err = rel(obj_pounce, obj_oracle)

print("=== pounce ===")
print(f"status={status} v={v_pounce:.10e} p={p_pounce} t={t_pounce:.4f}s")
print("=== oracle (scipy.optimize.linprog/HiGHS) ===")
print(f"v={v_oracle:.10e} p={p_oracle} t={t_oracle:.4f}s")
print(f"known_optimal_v={KNOWN_V:.10e} known_p={KNOWN_P}")
print(f"v_err_vs_known={v_err_known:.2e} v_err_vs_oracle={v_err_oracle:.2e} p_inf_err_vs_known={p_err:.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e}")

ok = (status == "optimal") and v_err_known < 1e-4 and v_err_oracle < 1e-4 and p_err < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, v_err_known={v_err_known:.2e})")
