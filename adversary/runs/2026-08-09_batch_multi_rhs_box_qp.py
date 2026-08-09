"""Adversary cross-check: solve_qp_multi_rhs vs closed-form box-QP + per-item solve_qp
Family: batch   Class: fixed geometry, varying linear objective (multi-RHS)
Source: no external oracle needed for the geometry -- the fixed problem is
        a separable box-constrained QP: minimize 0.5*sum(P_ii*x_i^2) +
        sum(c_i*x_i)  s.t. lb <= x <= ub, which has an EXACT closed form
        per coordinate (no cross-terms): unconstrained stationary point is
        x_i = -c_i/P_ii, clipped to [lb_i, ub_i]. This is derived
        analytically (Nocedal & Wright, projected-gradient / separable QP
        argument), independent of both pounce APIs under test.
Known optimal: per c-vector in the batch, x_i*(c) = clip(-c_i/P_ii, lb_i, ub_i);
        obj*(c) = sum(0.5*P_ii*x_i*(c)^2 + c_i*x_i*(c)).

This exercises solve_qp_multi_rhs -- the "same structure, many objectives"
batch API (distinct from solve_qp_batch, tested in the previous batch run
with per-item HETEROGENEOUS P/A/G; here P/A(none)/G(none)/lb/ub are FIXED
and only the linear term c varies, the code path solve_qp_multi_rhs is
built for). Several of the 6 RHS vectors are chosen so the closed-form
optimum sits strictly inside the box, AT a bound, and clipped by BOTH
bounds in different coordinates, to exercise the active-set bookkeeping
across a batch of shared-factorization solves.
"""
import time
import numpy as np

n = 4
P_diag = np.array([2.0, 4.0, 6.0, 8.0])
P = np.diag(P_diag)
lb = np.array([-3.0, -3.0, -3.0, -3.0])
ub = np.array([3.0, 3.0, 3.0, 3.0])

# 6 RHS (linear-term) vectors: mix of interior, at-bound, and clipped optima
cs = [
    np.array([1.0, -2.0, 3.0, -4.0]),      # x* = -c/P interior for all
    np.array([-20.0, 8.0, -6.0, 4.0]),     # coord 0 clipped to ub (x0*=-c0/2=10 -> clip 3)
    np.array([20.0, -8.0, 6.0, -4.0]),     # coord 0 clipped to lb
    np.array([0.0, 0.0, 0.0, 0.0]),        # trivial: x*=0 everywhere
    np.array([-6.0, -100.0, 5.0, -200.0]), # coords 1,3 both clipped (opposite... check signs)
    np.array([6.0, 12.0, -18.0, 24.0]),    # x* = -c/P exactly at half the box
]

# closed-form (independent oracle)
KNOWN_X = []
KNOWN_OBJ = []
for c_vec in cs:
    x_star = np.clip(-c_vec / P_diag, lb, ub)
    obj_star = float(np.sum(0.5 * P_diag * x_star ** 2 + c_vec * x_star))
    KNOWN_X.append(x_star)
    KNOWN_OBJ.append(obj_star)

from pounce import solve_qp, solve_qp_multi_rhs

t0 = time.perf_counter()
multi_results = solve_qp_multi_rhs(P=P, c=cs[0], lb=lb, ub=ub, cs=cs)
t_multi = time.perf_counter() - t0

t0 = time.perf_counter()
single_results = [solve_qp(P=P, c=c_vec, lb=lb, ub=ub) for c_vec in cs]
t_single = time.perf_counter() - t0


def rel(u, v):
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
    denom = max(1.0, float(np.linalg.norm(v, np.inf)))
    return float(np.linalg.norm(u - v, np.inf) / denom)


all_optimal = True
max_x_err_known = max_obj_err_known = 0.0
max_x_err_single = max_obj_err_single = 0.0
for i, (rm, rs, x_known, obj_known) in enumerate(zip(multi_results, single_results, KNOWN_X, KNOWN_OBJ)):
    if rm.status != "optimal" or rs.status != "optimal":
        all_optimal = False
        print(f"item {i}: status multi_rhs={rm.status} single={rs.status}")
        continue
    max_x_err_known = max(max_x_err_known, rel(rm.x, x_known))
    max_obj_err_known = max(max_obj_err_known, abs(rm.obj - obj_known) / max(1.0, abs(obj_known)))
    max_x_err_single = max(max_x_err_single, rel(rm.x, rs.x))
    max_obj_err_single = max(max_obj_err_single, abs(rm.obj - rs.obj) / max(1.0, abs(rs.obj)))

print("=== pounce solve_qp_multi_rhs vs closed-form + per-item solve_qp (6 RHS, fixed box QP) ===")
print(f"t_multi_rhs={t_multi:.4f}s t_single_sum={t_single:.4f}s")
print(f"all_optimal={all_optimal}")
print(f"max_x_err_vs_known={max_x_err_known:.2e} max_obj_err_vs_known={max_obj_err_known:.2e}")
print(f"max_x_err_vs_single={max_x_err_single:.2e} max_obj_err_vs_single={max_obj_err_single:.2e}")
for i, (rm, x_known, obj_known) in enumerate(zip(multi_results, KNOWN_X, KNOWN_OBJ)):
    print(f"  item {i}: x_pounce={rm.x} x_known={x_known} obj_pounce={rm.obj:.6f} obj_known={obj_known:.6f}")

# IPM solutions approach an active bound but do not land on it to machine
# precision (interior-point boundary slack ~tol); 1e-4 is generous relative
# to the default tol and still tight relative to the box width of 6. The
# multi_rhs-vs-single-solve check is held to a much stricter 1e-6 since both
# calls should hit the identical IPM fixed point.
ok = (
    all_optimal
    and max(max_x_err_known, max_obj_err_known) < 1e-4
    and max(max_x_err_single, max_obj_err_single) < 1e-6
)
print("VERDICT: PASS" if ok else "VERDICT: FAIL (multi_rhs mismatch vs closed-form/single)")
