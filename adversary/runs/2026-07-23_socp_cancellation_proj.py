"""Adversary cross-check: SOC projection onto a hyperplane with catastrophic cancellation
Family: socp   Class: extreme-scale / cancellation
Source: min ||x-a|| s.t. e'x=0.  Closed form x* = a - (e'a/e'e) e.
Known optimal: computed in float64 from the closed form; objective ||x*-a||.
Direction: a has +1e10 and -1e10 components (e'a is a catastrophic cancellation).
"""
import time, numpy as np

n = 4
a = np.array([1e10, 1.0, 1e-10, -1e10])
e = np.ones(n)
proj = a - (e @ a) / (e @ e) * e         # closed-form projection
res_ref = float(np.linalg.norm(proj - a))
print(f"e'a={e@a:.6e}  x*={proj}  res*={res_ref:.10e}")

# --- pounce SOC: min t s.t. (t, x-a) in SOC, e'x=0 ---
# vars z=(x0..x3, t). Aeq z = 0 with Aeq=[e,0]; slack (t, x-a) in SOC dim n+1.
from pounce import solve_socp
nv = n + 1
c = np.zeros(nv); c[-1] = 1.0
Aeq = np.zeros((1, nv)); Aeq[0, :n] = e; beq = np.zeros(1)
G = np.zeros((n + 1, nv)); h = np.zeros(n + 1)
G[0, -1] = -1.0                # s0 = t
G[1:, :n] = -np.eye(n)         # s[1:] = h - G x = a... need s[1:] = x - a
h[1:] = -a                     # s[1:] = h - (-I)x = x - a  -> h=-a? check: s = h - Gx = -a -(-I x)=x - a. yes
t0 = time.perf_counter()
r = solve_socp(P=None, c=c, A=Aeq, b=beq, G=G, h=h, cones=[("soc", n + 1)])
t_pounce = time.perf_counter() - t0
x_p = np.asarray(r.x)[:n]; status = r.status
res_p = float(np.linalg.norm(x_p - a))

# --- oracle: cvxpy ---
import cvxpy as cp
xc = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(xc - a, 2)), [e @ xc == 0])
t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
t_oracle = time.perf_counter() - t0
x_o = np.asarray(xc.value); res_o = float(np.linalg.norm(x_o - a))

def rel(u, v): return abs(u - v) / max(1.0, abs(v))
obj_err = rel(res_p, res_ref)
x_err = float(np.linalg.norm(x_p - proj, np.inf))
feas = float(abs(e @ x_p))
print("=== pounce ===");  print(f"status={status} res={res_p:.10e} eq_resid={feas:.2e} t={t_pounce:.4f}s x={x_p}")
print("=== oracle(cvxpy) ==="); print(f"res={res_o:.10e} t={t_oracle:.4f}s x={x_o}")
print("=== reference ==="); print(f"res*={res_ref:.10e} x*={proj}")
print(f"obj_err_vs_ref={obj_err:.2e} x_inf_err_vs_ref={x_err:.2e} eq_feas={feas:.2e}")
ok = (status in ("optimal",) or getattr(r, "success", False)) and obj_err < 1e-4 and feas < 1e-3
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e}, eq_feas={feas:.2e})")
