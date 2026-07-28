"""Adversary cross-check: singular (PSD, rank-deficient) Hessian + redundant equality rows.

Family: qp   Class: degeneracy / constraint-qualification failure / non-unique optimum

Source: constructed instance in the standard degenerate-QP style of
  Nocedal & Wright, *Numerical Optimization* 2e, Ch. 16.1-16.2 (equality-constrained
  QP; reduced/null-space method, singular reduced Hessian) and Ch. 16.8
  (rank-deficient constraint Jacobian / redundant constraints), and
  Boyd & Vandenberghe, *Convex Optimization*, Sec. 4.4 / 10.1 (QP with
  positive-SEMI-definite P: solution set is an affine set, not a point).

Problem (n = 5):

    minimize    1/2 x' P x + c' x,   P = M'M,  M = [[1,1,0,0,0],
                                                    [0,0,1,1,0]]
    subject to  A x = b

  P = diag-blocks [[1,1],[1,1]], [[1,1],[1,1]], [0]  -> rank(P) = 2, nullity 3.
  c = [1, 2, 3, 4, -1]

  A rows (4x5, rank 2 -- rows 3,4 are exact linear combinations of rows 1,2):
     a1 = [1, 0, 1, 0, 1]   b1 = 3
     a2 = [0, 1, 0, 1, 1]   b2 = 2
     a3 = a1                b3 = 3        (exact duplicate)
     a4 = 2*a1 - a2         b4 = 4        (redundant combination)

  null(P) INTERSECT null(A) = span{ d = (1,-1,-1,1,0) }, and c'd = 1-2-3+4 = 0,
  so the QP is bounded below and its OPTIMAL SET IS A LINE:  X* = x_ref + t*d.

  Because the minimizer is not unique, correctness is graded on
    (1) objective value,          (2) feasibility A x = b,
    (3) MEMBERSHIP in X*:  Z'(P x + c) = 0 with Z a basis of null(A),
  and NOT on x equality against any particular oracle point.

Three cases are run so a failure can be attributed:
  case "semidef_P"   : singular P, full-rank A (rows 1,2 only)
  case "dup_rows"    : nonsingular P, rank-deficient A (all 4 rows)
  case "both"        : singular P AND rank-deficient A   <-- the headline case
"""

import time

import numpy as np

np.set_printoptions(precision=6, suppress=True)

M = np.array([[1.0, 1, 0, 0, 0], [0, 0, 1, 1, 0]])
P_SING = M.T @ M                       # rank 2, PSD, singular
P_PD = P_SING + np.eye(5)              # SPD control
C = np.array([1.0, 2.0, 3.0, 4.0, -1.0])

A_FULL = np.array([[1.0, 0, 1, 0, 1], [0.0, 1, 0, 1, 1]])
B_FULL = np.array([3.0, 2.0])
A_DUP = np.vstack([A_FULL, A_FULL[0], 2 * A_FULL[0] - A_FULL[1]])
B_DUP = np.array([3.0, 2.0, 3.0, 2 * 3.0 - 2.0])

D_NULL = np.array([1.0, -1.0, -1.0, 1.0, 0.0])   # null(P) cap null(A_FULL)

# Stress variant: 8 rows of rank 2, including a badly scaled redundant row
# (1e6 * a1) and a near-duplicate pair, on top of the singular P.
A_STRESS = np.vstack([
    A_FULL,
    A_FULL[0],
    2 * A_FULL[0] - A_FULL[1],
    1e6 * A_FULL[0],
    -3 * A_FULL[1],
    A_FULL[0] + A_FULL[1],
    1e-6 * A_FULL[1],
])
B_STRESS = np.array([3.0, 2.0, 3.0, 4.0, 3e6, -6.0, 5.0, 2e-6])

CASES = [
    ("semidef_P", P_SING, A_FULL, B_FULL),
    ("dup_rows", P_PD, A_DUP, B_DUP),
    ("both", P_SING, A_DUP, B_DUP),
    ("stress8x5", P_SING, A_STRESS, B_STRESS),
]


def obj(P, x):
    return 0.5 * float(x @ P @ x) + float(C @ x)


def nullspace(A, rtol=1e-10):
    _, s, vt = np.linalg.svd(A)
    tol = (s[0] if s.size else 0.0) * rtol
    r = int((s > tol).sum())
    return vt[r:].T


def analytic_optimum(P, A, b):
    """Exact reduced (null-space) solution. Returns (x_ref, p_star, Z)."""
    Z = nullspace(A)
    x_p = np.linalg.lstsq(A, b, rcond=None)[0]        # min-norm particular point
    H = Z.T @ P @ Z                                    # reduced Hessian (may be singular)
    g = Z.T @ (P @ x_p + C)
    u = np.linalg.lstsq(H, -g, rcond=None)[0]          # min-norm reduced step
    x_ref = x_p + Z @ u
    return x_ref, obj(P, x_ref), Z


def grade(name, P, A, b, x, p_star, Z, label):
    x = np.asarray(x, float).ravel()
    feas = float(np.max(np.abs(A @ x - b))) if A.size else 0.0
    o = obj(P, x)
    obj_err = abs(o - p_star) / max(1.0, abs(p_star))
    # membership in the optimal set: reduced gradient must vanish
    memb = float(np.max(np.abs(Z.T @ (P @ x + C)))) if Z.size else 0.0
    print(
        f"    {label:<10s} obj={o:+.12e}  obj_rel_err={obj_err:.2e}  "
        f"feas_inf={feas:.2e}  reduced_grad_inf={memb:.2e}"
    )
    return obj_err, feas, memb


import cvxpy as cp  # noqa: E402
from pounce import solve_qp  # noqa: E402

all_ok = True
rows = []

for name, P, A, b in CASES:
    print(f"\n=== case: {name}  rank(P)={np.linalg.matrix_rank(P)}  "
          f"A:{A.shape} rank(A)={np.linalg.matrix_rank(A)} ===")
    x_ref, p_star, Z = analytic_optimum(P, A, b)
    dim_opt = nullspace(np.vstack([P, A])).shape[1]
    print(f"    analytic p* = {p_star:+.12e}   dim(optimal set) = {dim_opt}")

    # --- pounce ---
    t0 = time.perf_counter()
    r = solve_qp(P=P, c=C, A=A, b=b)
    t_p = time.perf_counter() - t0
    status = r.status
    print(f"    pounce status={status} iters={getattr(r, 'iterations', '?')} t={t_p:.4f}s")
    e_p = grade(name, P, A, b, r.x, p_star, Z, "pounce")

    # --- oracles ---
    ora = {}
    for solver in ("CLARABEL", "OSQP"):
        xv = cp.Variable(5)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(xv, cp.psd_wrap(P)) + C @ xv), [A @ xv == b]
        )
        t0 = time.perf_counter()
        try:
            prob.solve(solver=getattr(cp, solver))
        except Exception as exc:  # noqa: BLE001
            print(f"    {solver:<10s} EXCEPTION {type(exc).__name__}: {exc}")
            continue
        t_o = time.perf_counter() - t0
        if xv.value is None:
            print(f"    {solver:<10s} status={prob.status} (no solution)")
            continue
        print(f"    {solver} status={prob.status} t={t_o:.4f}s")
        ora[solver] = (grade(name, P, A, b, xv.value, p_star, Z, solver), t_o)

    # direction check: pounce's answer differs from the analytic point only
    # along the optimal-set direction (only meaningful when dim_opt > 0)
    if dim_opt > 0:
        delta = np.asarray(r.x, float).ravel() - x_ref
        Zopt = nullspace(np.vstack([P, A]))
        resid = delta - Zopt @ (Zopt.T @ delta)
        print(f"    ||x_pounce - x_ref|| = {np.linalg.norm(delta):.3e}, "
              f"component OUTSIDE optimal-set direction = {np.linalg.norm(resid):.3e}")

    ok = (
        status in ("optimal", "Solved", "solved")
        and e_p[0] < 1e-6
        and e_p[1] < 1e-6
        and e_p[2] < 1e-6
    )
    all_ok &= ok
    rows.append((name, status, e_p, t_p, ok))
    print(f"    case verdict: {'PASS' if ok else 'FAIL'}")

print("\n--- summary ---")
for name, status, (oe, fe, me), t, ok in rows:
    print(f"{name:<12s} status={status:<10s} obj_err={oe:.2e} feas={fe:.2e} "
          f"redgrad={me:.2e} t={t:.4f}s {'PASS' if ok else 'FAIL'}")

print("VERDICT: PASS" if all_ok else "VERDICT: FAIL")
