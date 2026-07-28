"""Adversary cross-check: Hock-Schittkowski problem 71 (HS071)
Family: nlp   Class: inequality + equality constrained, bounded
Source: Hock & Schittkowski, "Test Examples for Nonlinear Programming
        Codes", problem 71. Also the canonical Ipopt tutorial example.
Known optimal: f* = 17.0140173   x* = (1.0, 4.743, 3.821, 1.379)
"""
import time
import numpy as np

KNOWN_OPTIMAL = 17.0140172891520078

# --- problem data ---
def f(x):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

# scipy-style constraints: ineq fun >= 0, eq fun == 0
constraints = [
    {"type": "ineq", "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0},
    {"type": "eq", "fun": lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40.0},
]
bounds = [(1.0, 5.0)] * 4
x0 = np.array([1.0, 5.0, 5.0, 1.0])

# --- pounce ---
import pounce
t0 = time.perf_counter()
res = pounce.minimize(f, x0, bounds=bounds, constraints=constraints,
                      options={"solver_selection": "nlp"})
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = res.x, res.fun, res.status

# --- oracle: Pyomo + Ipopt ---
import pyomo.environ as pyo
m = pyo.ConcreteModel()
m.x = pyo.Var(range(4), bounds=(1.0, 5.0))
m.x[0] = 1.0; m.x[1] = 5.0; m.x[2] = 5.0; m.x[3] = 1.0
m.obj = pyo.Objective(expr=m.x[0]*m.x[3]*(m.x[0]+m.x[1]+m.x[2]) + m.x[2])
m.c1 = pyo.Constraint(expr=m.x[0]*m.x[1]*m.x[2]*m.x[3] >= 25.0)
m.c2 = pyo.Constraint(expr=sum(m.x[i]**2 for i in range(4)) == 40.0)
t0 = time.perf_counter()
pyo.SolverFactory("ipopt").solve(m)
t_oracle = time.perf_counter() - t0
x_oracle = np.array([pyo.value(m.x[i]) for i in range(4)])
obj_oracle = pyo.value(m.obj)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))

obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} x={x_pounce} t={t_pounce:.4f}s nit={res.nit}")
print("=== oracle (Ipopt) ===")
print(f"obj={obj_oracle:.10e} x={x_oracle} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = res.success and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
