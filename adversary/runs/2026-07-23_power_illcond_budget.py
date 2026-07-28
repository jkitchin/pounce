"""Adversary cross-check: geometric-mean maximization under a budget, extreme scale
Family: power   Class: extreme-scale power cone
Source: max x s.t. |x| <= y^.5 z^.5, y+z<=S, y,z>=0.  For alpha=.5: y=z=S/2, x*=S/2.
Known optimal: x* = S/2.  S=2e8 => x*=1e8.
Direction: budget/objective span 8 orders of magnitude; tests power-cone HSDE scaling.
"""
import time, numpy as np

S = 2e8
x_star = S / 2.0    # 1e8
print(f"S={S:.1e} x*={x_star:.10e}")

# --- pounce power:  vars z=(x,y,w) ---
from pounce import solve_socp
c = np.array([-1.0, 0.0, 0.0])           # maximize x
G = np.zeros((4, 3)); h = np.zeros(4)
# power cone (x,y,w) in Kpow(0.5): |x| <= y^.5 w^.5
G[0, 0] = -1.0   # s0 = x
G[1, 1] = -1.0   # s1 = y
G[2, 2] = -1.0   # s2 = w
# budget y + w <= S
G[3, 1] = 1.0; G[3, 2] = 1.0; h[3] = S   # s3 = S - y - w >= 0
t0 = time.perf_counter()
r = solve_socp(P=None, c=c, G=G, h=h, cones=[("pow", 0.5), ("nonneg", 1)])
t_pounce = time.perf_counter() - t0
x_p = float(np.asarray(r.x)[0]); status = r.status

# --- oracle: cvxpy PowCone3D ---
import cvxpy as cp
xv = cp.Variable(); yv = cp.Variable(); wv = cp.Variable()
# cvxpy PowCone3D(x,y,z,alpha):  x^alpha y^(1-alpha) >= |z|, x,y>=0
cons = [cp.constraints.PowCone3D(yv, wv, xv, 0.5), yv + wv <= S, yv >= 0, wv >= 0]
prob = cp.Problem(cp.Maximize(xv), cons)
t0 = time.perf_counter()
prob.solve(solver=cp.SCS)
t_oracle = time.perf_counter() - t0
x_o = float(xv.value)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
err_ref = rel(x_p, x_star); err_oracle = rel(x_p, x_o)
print("=== pounce ===");  print(f"status={status} x={x_p:.10e} t={t_pounce:.4f}s")
print("=== oracle(cvxpy SCS) ==="); print(f"x={x_o:.10e} t={t_oracle:.4f}s")
print("=== reference ==="); print(f"x*={x_star:.10e}")
print(f"x_err_vs_ref={err_ref:.2e} x_err_vs_oracle={err_oracle:.2e}")
ok = (status in ("optimal",) or getattr(r, "success", False)) and err_ref < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, x_err_ref={err_ref:.2e})")
