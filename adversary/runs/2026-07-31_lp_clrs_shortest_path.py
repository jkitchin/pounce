"""Adversary cross-check: CLRS shortest-path-as-LP (min-cost flow)
Family: lp   Class: network-flow (min-cost flow LP, TU polytope -> integral optimum)
Source: Cormen, Leiserson, Rivest, Stein, "Introduction to Algorithms" 3rd ed.,
        Figure 24.6 (single-source shortest paths, Dijkstra's algorithm example).
        Vertices {s,t,x,y,z}; directed edges/weights:
        s->t:10, s->y:5, t->x:1, t->y:2, x->z:4, y->t:3, y->x:9, y->z:2,
        z->x:6, z->s:7.
        Published shortest-path distances from s (CLRS Section 24.3, worked
        example): d(s)=0, d(y)=5, d(z)=7, d(t)=8, d(x)=9.
Known optimal: shortest s->x distance = 9 (path s->y->t->x: 5+3+1=9)
"""
import numpy as np
from scipy.optimize import linprog

KNOWN_OPTIMAL = 9.0

# Vertices in order: s=0, t=1, x=2, y=3, z=4
nodes = ["s", "t", "x", "y", "z"]
idx = {n: i for i, n in enumerate(nodes)}

edges = [
    ("s", "t", 10.0),
    ("s", "y", 5.0),
    ("t", "x", 1.0),
    ("t", "y", 2.0),
    ("x", "z", 4.0),
    ("y", "t", 3.0),
    ("y", "x", 9.0),
    ("y", "z", 2.0),
    ("z", "x", 6.0),
    ("z", "s", 7.0),
]
m = len(edges)
n = len(nodes)
c = np.array([w for (_, _, w) in edges])

# Node-arc incidence: conservation A_eq @ f = b_eq
# row per node: sum(outflow) - sum(inflow) = supply (source=+1, sink=-1, else 0)
A_eq = np.zeros((n, m))
for j, (u, v, _w) in enumerate(edges):
    A_eq[idx[u], j] += 1.0
    A_eq[idx[v], j] -= 1.0

b_eq = np.zeros(n)
b_eq[idx["s"]] = 1.0
b_eq[idx["x"]] = -1.0

# --- pounce (LP: solve_qp with P=None) ---
from pounce import solve_qp
import time

t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A_eq, b=b_eq, lb=np.zeros(m), ub=None)
t_pounce = time.perf_counter() - t0
x_pounce, obj_pounce, status = np.asarray(r.x), r.obj, r.status

# --- oracle: scipy.optimize.linprog (HiGHS) ---
t0 = time.perf_counter()
res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * m, method="highs")
t_oracle = time.perf_counter() - t0
x_oracle, obj_oracle = res.x, res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

print("=== pounce ===")
print(f"status={status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print("=== oracle (scipy HiGHS) ===")
print(f"status={res.status} obj={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={rel(obj_pounce, KNOWN_OPTIMAL):.2e}")
print(f"obj_err_vs_oracle={obj_err:.2e} x_inf_err={x_err:.2e}")

ok = (status == "optimal") and obj_err < 1e-4 and rel(obj_pounce, KNOWN_OPTIMAL) < 1e-4
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={status}, obj_err={obj_err:.2e})")
