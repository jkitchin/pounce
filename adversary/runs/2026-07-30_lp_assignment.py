"""Adversary cross-check: bipartite assignment problem, LP relaxation
Family: lp   Class: network-flow / totally-unimodular LP (new class for lp;
prior lp runs covered bounded 2-var, diet/blending, degenerate transportation,
Klee-Minty, badly-scaled, dual-degenerate, status battery, duality invariants,
shadow prices, reduced costs, unboundedness -- none used an assignment-problem
constraint structure or the Hungarian algorithm as an independent combinatorial
oracle).

Problem: classic 5x5 linear assignment problem.
    minimize   sum_{i,j} c_ij x_ij
    subject to sum_j x_ij = 1   for each row i      (each worker assigned once)
               sum_i x_ij = 1   for each col j      (each job filled once)
               x_ij >= 0                            (NO integrality constraint)

The constraint matrix of an assignment problem is totally unimodular (it is
the incidence matrix of a bipartite graph), so the LP relaxation's optimal
basic feasible solutions are automatically integral (Birkhoff-von Neumann /
TU theory) -- the LP relaxation optimum EQUALS the combinatorial assignment
optimum, with no need to round. This makes the Hungarian algorithm
(scipy.optimize.linear_sum_assignment, an O(n^3) combinatorial method
completely unrelated to simplex/interior-point LP solvers) a genuinely
independent oracle for an LP.

SOURCE: standard OR textbook LP (e.g. Bertsimas & Tsitsiklis, "Introduction to
Linear Optimization", Ch. 1 network-flow / assignment example; Hungarian
algorithm: Kuhn 1955, "The Hungarian method for the assignment problem",
Naval Research Logistics Quarterly 2(1-2):83-97). Cost matrix fixed here with
a hardcoded seed so the "known optimal" is reproducible without re-deriving.

KNOWN_OPTIMAL: computed at runtime by scipy.optimize.linear_sum_assignment
(Hungarian algorithm) -- independent of both pounce and scipy.linprog.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

rng = np.random.default_rng(20260730)
N = 5
C = rng.integers(1, 50, size=(N, N)).astype(float)

# --- oracle #1: Hungarian algorithm (combinatorial, NOT an LP solver) ---
t0 = time.perf_counter()
row_ind, col_ind = linear_sum_assignment(C)
t_hungarian = time.perf_counter() - t0
KNOWN_OPTIMAL = float(C[row_ind, col_ind].sum())
X_STAR = np.zeros((N, N))
X_STAR[row_ind, col_ind] = 1.0

# --- build the LP relaxation in standard equality form for pounce ---
# variable order: x_00, x_01, ..., x_0(N-1), x_10, ..., x_(N-1)(N-1)  (row-major)
n = N * N
c = C.flatten()
A = np.zeros((2 * N, n))
for i in range(N):
    A[i, i * N:(i + 1) * N] = 1.0          # row-sum i
for j in range(N):
    for i in range(N):
        A[N + j, i * N + j] = 1.0          # col-sum j
b = np.ones(2 * N)
lb = np.zeros(n)
# NOTE: no upper bound of 1 imposed -- TU + nonnegativity + the two equality
# families is enough to force x_ij in {0,1} at any LP vertex, so omitting an
# explicit ub=1 is deliberate, not an oversight (adds a test that pounce's LP
# path doesn't need the upper bound to reach the integral vertex).

from pounce import solve_qp

t0 = time.perf_counter()
r = solve_qp(P=None, c=c, A=A, b=b, lb=lb)
t_pounce = time.perf_counter() - t0
x_pounce = np.asarray(r.x).reshape(N, N)
obj_pounce = float(r.obj)

# --- oracle #2: scipy.optimize.linprog (HiGHS), independent LP solver ---
t0 = time.perf_counter()
res = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs")
t_linprog = time.perf_counter() - t0
obj_linprog = float(res.fun)
x_linprog = np.asarray(res.x).reshape(N, N)


def rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


obj_err_known = rel(obj_pounce, KNOWN_OPTIMAL)
obj_err_linprog = rel(obj_pounce, obj_linprog)
linprog_vs_known = rel(obj_linprog, KNOWN_OPTIMAL)
x_err_vs_known = float(np.linalg.norm(x_pounce - X_STAR, np.inf))
# Fractional-vertex check: did pounce actually land on an INTEGRAL vertex
# (as TU theory guarantees), or some other optimal fractional point on a
# degenerate face? Report it; a fractional-but-optimal-objective point would
# still PASS on objective but is worth flagging.
max_fractional_gap = float(np.min([np.abs(x_pounce - 0), np.abs(x_pounce - 1)], axis=0).max())

print("=== assignment problem, 5x5, LP relaxation ===")
print(f"cost matrix:\n{C}")
print(f"Hungarian (oracle #1, combinatorial): obj={KNOWN_OPTIMAL:.10e} t={t_hungarian:.5f}s")
print(f"assignment: {list(zip(row_ind.tolist(), col_ind.tolist()))}")
print("-- pounce (solve_qp, LP: P=None) --")
print(f"status={r.status} obj={obj_pounce:.10e} t={t_pounce:.4f}s")
print("-- scipy.optimize.linprog (HiGHS, oracle #2) --")
print(f"status={res.status} obj={obj_linprog:.10e} t={t_linprog:.4f}s")
print(f"obj_err vs known(Hungarian) = {obj_err_known:.2e}")
print(f"obj_err vs linprog(HiGHS)   = {obj_err_linprog:.2e}")
print(f"linprog vs known(Hungarian) = {linprog_vs_known:.2e}  (sanity: both LP oracles agree)")
print(f"x_inf_err vs Hungarian assignment matrix = {x_err_vs_known:.2e}")
print(f"max distance of any x_ij from {{0,1}} (integrality check) = {max_fractional_gap:.2e}")

TOL = 1e-6
ok = (
    r.status == "optimal"
    and obj_err_known < TOL
    and obj_err_linprog < TOL
    and x_err_vs_known < 1e-5
)
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (status={r.status}, obj_err_known={obj_err_known:.2e})")
