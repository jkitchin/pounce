"""Adversary cross-check: geometric program with an extreme coefficient
Family: exp   Class: extreme-scale GP
Source: min x + K/x  (x>0),  optimum x*=sqrt(K), f*=2*sqrt(K).  K=1e12 => f*=2e6.
Known optimal: 2e6 exactly (analytic AM-GM).
Direction: coefficient K spans 12 orders of magnitude; tests exp-cone HSDE scaling.
"""
import time, numpy as np

K = 1e12
lnK = np.log(K)
x_star = np.sqrt(K)          # 1e6
f_star = 2.0 * np.sqrt(K)    # 2e6
print(f"K={K:.1e} x*={x_star:.6e} f*={f_star:.10e}  lnK={lnK:.6f}")

# --- pounce exp:  min t1 + t2
#   (u, 1, t1) in Kexp  =>  t1 >= e^u = x
#   (lnK - u, 1, t2) in Kexp  =>  t2 >= e^{lnK-u} = K e^{-u} = K/x
# vars z = (u, t1, t2)
from pounce import solve_socp
c = np.array([0.0, 1.0, 1.0])
G = np.zeros((6, 3)); h = np.zeros(6)
# cone 1: s0=u, s1=1, s2=t1
G[0, 0] = -1.0                # s0 = u
h[1] = 1.0                    # s1 = 1
G[2, 1] = -1.0               # s2 = t1
# cone 2: s0 = lnK - u, s1=1, s2=t2
G[3, 0] = 1.0; h[3] = lnK    # s3 = lnK - u
h[4] = 1.0                    # s4 = 1
G[5, 2] = -1.0               # s5 = t2
t0 = time.perf_counter()
r = solve_socp(P=None, c=c, G=G, h=h, cones=[("exp", 3), ("exp", 3)])
t_pounce = time.perf_counter() - t0
u = float(np.asarray(r.x)[0]); x_p = np.exp(u)
obj_p = float(r.obj)
status = r.status

# --- oracle: cvxpy exp cone ---
import cvxpy as cp
uc = cp.Variable(); t1 = cp.Variable(); t2 = cp.Variable()
cons = [cp.constraints.ExpCone(uc, 1.0, t1), cp.constraints.ExpCone(lnK - uc, 1.0, t2)]
prob = cp.Problem(cp.Minimize(t1 + t2), cons)
t0 = time.perf_counter()
prob.solve(solver=cp.ECOS)
t_oracle = time.perf_counter() - t0
obj_o = float(prob.value)

def rel(a, b): return abs(a - b) / max(1.0, abs(b))
obj_err_ref = rel(obj_p, f_star)
obj_err_oracle = rel(obj_p, obj_o)
x_err = rel(x_p, x_star)

print("=== pounce ===");  print(f"status={status} obj={obj_p:.10e} x=e^u={x_p:.6e} t={t_pounce:.4f}s")
print("=== oracle(cvxpy ECOS) ==="); print(f"obj={obj_o:.10e} t={t_oracle:.4f}s")
print("=== reference ==="); print(f"f*={f_star:.10e} x*={x_star:.6e}")
print(f"obj_err_vs_ref={obj_err_ref:.2e} obj_err_vs_oracle={obj_err_oracle:.2e} x_rel_err={x_err:.2e}")

ok = (status in ("optimal",) or getattr(r, "success", False)) and obj_err_ref < 1e-5
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err_ref={obj_err_ref:.2e})")
