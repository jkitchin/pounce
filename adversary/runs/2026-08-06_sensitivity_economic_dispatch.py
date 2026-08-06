"""Adversary cross-check: economic-dispatch QP sensitivity dP_i/dD
Family: sensitivity   Class: parametric dx/db, applied power-systems QP
Source: classic economic-dispatch "equal incremental cost" criterion, e.g.
        Wood, Wollenberg & Sheble, "Power Generation, Operation, and
        Control", Ch. 3. Quadratic generator cost C_i(P_i) = a_i P_i^2 +
        b_i P_i, minimized subject to power balance sum(P_i) = D and box
        limits Pmin <= P_i <= Pmax.
        At an interior optimum (no P_i at a bound) all marginal costs are
        equal: 2 a_i P_i + b_i = lambda for every i. Differentiating the
        KKT system w.r.t. D gives the closed form
            dP_i/dD = (1/(2 a_i)) / sum_j (1/(2 a_j))
        independent of D and b, as long as the active set (which units are
        at a bound) does not change.
Known optimal: dP_i/dD given by the closed form above; cross-checked
        against pounce's QpSensitivity.parametric_step AND a central
        finite-difference re-solve of the QP at D +/- delta.
"""
import numpy as np

# 4 generators, quadratic cost a_i P_i^2 + b_i P_i (n.b. P = a*x^2+b*x form
# means the QP Hessian diagonal is 2*a_i, matching solve_qp's 1/2 x'Px + c'x)
a = np.array([0.02, 0.03, 0.015, 0.025])   # $/MW^2 cost curvature
b = np.array([20.0, 18.0, 22.0, 19.0])     # $/MW linear cost
Pmin = np.array([10.0, 10.0, 10.0, 10.0])
Pmax = np.array([300.0, 300.0, 300.0, 300.0])   # slack -- no unit at a bound
D0 = 400.0
n = len(a)

P_qp = np.diag(2.0 * a)
c_qp = b.copy()
A_eq = np.ones((1, n))
b_eq = np.array([D0])

from pounce.qp import QpSensitivity
sens = QpSensitivity(P=P_qp, c=c_qp, A=A_eq, b=b_eq, lb=Pmin, ub=Pmax)
assert not sens.ill_conditioned, "unexpectedly ill-conditioned KKT"
assert not sens.weakly_active_indices.inequalities and not sens.weakly_active_indices.bounds, \
    "unexpected weakly-active constraint"

delta_D = 1.0
dP_pounce = sens.parametric_step([0], [delta_D])

# --- closed-form oracle ---
inv2a = 1.0 / (2.0 * a)
dP_closed_form = inv2a / inv2a.sum()

# --- independent oracle: central finite-difference re-solve via pounce's
# OWN solve_qp is disallowed (that would be pounce-vs-pounce); use the
# closed-form KKT system solved directly with numpy as the true oracle,
# plus a finite-difference re-solve computed by directly inverting the
# bordered KKT matrix (independent of pounce's sensitivity machinery).
def solve_dispatch_numpy(D):
    # KKT: [2A, 1; 1^T, 0] [P; lambda] = [-b; D]   (unconstrained-bounds case)
    K = np.zeros((n + 1, n + 1))
    K[:n, :n] = 2.0 * np.diag(a)
    K[:n, n] = 1.0
    K[n, :n] = 1.0
    rhs = np.concatenate([-b, [D]])
    sol = np.linalg.solve(K, rhs)
    return sol[:n]


P_plus = solve_dispatch_numpy(D0 + delta_D)
P_minus = solve_dispatch_numpy(D0 - delta_D)
dP_fd = (P_plus - P_minus) / (2.0 * delta_D)
P0_numpy = solve_dispatch_numpy(D0)


def relvec(u, v):
    return float(np.linalg.norm(u - v, np.inf) / max(1.0, np.linalg.norm(v, np.inf)))


err_vs_closed_form = relvec(dP_pounce, dP_closed_form)
err_vs_fd = relvec(dP_pounce, dP_fd)
x0_err = relvec(sens.x, P0_numpy)

print("=== pounce (QpSensitivity.parametric_step) ===")
print(f"x0={sens.x}")
print(f"dP/dD={dP_pounce}")
print("=== independent oracles ===")
print(f"closed_form dP/dD={dP_closed_form}")
print(f"central-FD (bordered-KKT re-solve, numpy) dP/dD={dP_fd}")
print(f"x0_err_vs_numpy_KKT={x0_err:.2e}")
print(f"err_vs_closed_form={err_vs_closed_form:.2e} err_vs_finite_difference={err_vs_fd:.2e}")

ok = err_vs_closed_form < 1e-6 and err_vs_fd < 1e-6 and x0_err < 1e-8
print("VERDICT: PASS" if ok else f"VERDICT: FAIL (err_closed_form={err_vs_closed_form:.2e}, err_fd={err_vs_fd:.2e})")
