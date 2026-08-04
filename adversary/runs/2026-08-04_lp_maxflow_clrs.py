"""Adversary cross-check: maximum-flow LP (CLRS flow network)
Family: lp   Class: network-flow LP, max-flow via conservation + capacity
    (fresh class for lp -- prior lp runs covered shortest-path-as-LP,
    transportation/assignment, free-variable, badly-scaled/Klee-Minty,
    degenerate/dual-degenerate, status-reporting, duality/shadow-prices; none
    used the maximum-flow LP formulation with per-edge capacity bounds and
    node conservation equalities).
Source: Cormen, Leiserson, Rivest, Stein, "Introduction to Algorithms" 3rd ed.,
    Section 26.1, Figure 26.1 (the canonical s -> v1,v2,v3,v4 -> t flow
    network). CLRS proves (Sec 26.1/26.2 worked example) the maximum flow
    value is 23.

Formulation (edges e = (u, v) with capacity cap_e, flow variable f_e):
    maximize   f(s,v1) + f(s,v2)                 (flow leaving s)
    subject to capacity: 0 <= f_e <= cap_e for all e
               conservation at v1: f(s,v1) + f(v2,v1) - f(v1,v3) = 0
               conservation at v2: f(s,v2) + f(v3,v2) - f(v2,v1) - f(v2,v4) = 0
               conservation at v3: f(v1,v3) + f(v4,v3) - f(v3,v2) - f(v3,t) = 0
               conservation at v4: f(v2,v4) - f(v4,v3) - f(v4,t) = 0

Edges (index: (u,v), capacity), taken directly from CLRS Fig 26.1:
    e0 (s ,v1) 16   e1 (s ,v2) 13   e2 (v1,v3) 12   e3 (v2,v1) 4
    e4 (v2,v4) 14   e5 (v3,v2) 9    e6 (v3,t ) 20   e7 (v4,v3) 7
    e8 (v4,t ) 4

KNOWN_OPTIMAL (max flow) = 23  (CLRS, worked example, min-cut {s,v1,v2} vs
    {v3,v4,t} has capacity f(v1,v3)+f(v2,v4)+ ... ; CLRS derives 23 directly).
"""
import time
import numpy as np
from scipy.optimize import linprog
from pounce import solve_qp

KNOWN_OPTIMAL = 23.0

# edges: (name, capacity)
edges = [
    ("s_v1", 16.0), ("s_v2", 13.0), ("v1_v3", 12.0), ("v2_v1", 4.0),
    ("v2_v4", 14.0), ("v3_v2", 9.0), ("v3_t", 20.0), ("v4_v3", 7.0),
    ("v4_t", 4.0),
]
idx = {name: i for i, (name, _) in enumerate(edges)}
nvar = len(edges)
cap = np.array([c for _, c in edges])

# maximize f(s_v1)+f(s_v2)  <=>  minimize -(f(s_v1)+f(s_v2))
c = np.zeros(nvar)
c[idx["s_v1"]] = -1.0
c[idx["s_v2"]] = -1.0

# conservation equalities (net flow = 0) at v1, v2, v3, v4
def row(pos, neg):
    r = np.zeros(nvar)
    for name in pos:
        r[idx[name]] += 1.0
    for name in neg:
        r[idx[name]] -= 1.0
    return r

A = np.array([
    row(["s_v1", "v2_v1"], ["v1_v3"]),                    # v1
    row(["s_v2", "v3_v2"], ["v2_v1", "v2_v4"]),            # v2
    row(["v1_v3", "v4_v3"], ["v3_v2", "v3_t"]),             # v3
    row(["v2_v4"], ["v4_v3", "v4_t"]),                      # v4
])
b = np.zeros(4)

lb = np.zeros(nvar)
ub = cap.copy()

# --- pounce ---
t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A, b=b, lb=lb, ub=ub)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x)
obj_pounce = -r.obj  # maximize sense
status = r.status

# --- oracle: scipy.optimize.linprog (HiGHS), independent LP codebase ---
t0 = time.perf_counter()
res = linprog(c, A_eq=A, b_eq=b, bounds=list(zip(lb, ub)), method="highs")
t_oracle = time.perf_counter() - t0
x_oracle = res.x
obj_oracle = -res.fun


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_oracle = rel(obj_pounce, obj_oracle)
x_err = float(np.linalg.norm(x_pounce - x_oracle, np.inf))

# sanity: conservation & capacity feasibility of pounce's answer, checked
# independently of both pounce's own status flag and the LP oracle
feas_cap = bool(np.all(x_pounce >= -1e-7) and np.all(x_pounce <= cap + 1e-7))
feas_cons = float(np.max(np.abs(A @ x_pounce - b)))

print("=== pounce ===")
print(f"status={status} max_flow={obj_pounce:.10e} t={t_pounce:.4f}s")
print(f"edges: " + ", ".join(f"{n}={v:.4f}" for (n, _), v in zip(edges, x_pounce)))
print("=== oracle (scipy linprog/HiGHS) ===")
print(f"success={res.success} max_flow={obj_oracle:.10e} t={t_oracle:.4f}s")
print(f"known_optimal={KNOWN_OPTIMAL:.10e} rel_err_vs_known={obj_err_known:.2e}")
print(f"obj_err_vs_oracle={obj_err_oracle:.2e} x_inf_err_vs_oracle={x_err:.2e}")
print(f"feas_capacity={feas_cap} feas_conservation_max_resid={feas_cons:.2e}")

ok = (
    status in ("optimal", "solved")
    and obj_err_known < 1e-6
    and obj_err_oracle < 1e-6
    and feas_cap
    and feas_cons < 1e-6
)
print("VERDICT: PASS" if ok else
      f"VERDICT: FAIL (status={status}, obj_err_known={obj_err_known:.2e}, "
      f"feas_cap={feas_cap}, feas_cons={feas_cons:.2e})")
