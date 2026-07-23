"""P4 -- Strictly convex equality-constrained QP (Markowitz min-variance).

The most airtight dual check available: on the convex path the primal AND the
multipliers have a CLOSED FORM (the KKT linear system), so the duals are not
merely self-consistent between solvers -- they match an analytic value, sign
included. Four routes are compared:

    1. closed form   -- solve  [[P, A'],[A, 0]] [x; -y] = [-c; b]
    2. pounce.solve_qp
    3. cvxpy (CLARABEL)
    4. IPOPT via Pyomo (dual suffix)

Problem: minimum-variance portfolio over 5 assets with a budget constraint
sum(x)=1 and a target-return constraint mu'x = r_target. Both equality
multipliers are interpretable shadow prices (marginal variance per unit budget
/ per unit required return). No sign bound is active, so this isolates the
equality-multiplier convention.

    min 0.5 x' P x        s.t.  1' x = 1,  mu' x = r_target
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from _common import abs_err, dump_result, setup_pounce

setup_pounce()

# ── problem data (fixed, SPD covariance) ───────────────────────────────────────
# A recognizable 5-asset covariance matrix: D^T D + small ridge => SPD.
_rng = np.random.default_rng(0)
_L = np.array([
    [0.20, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.18, 0.00, 0.00, 0.00],
    [0.03, 0.04, 0.15, 0.00, 0.00],
    [0.02, 0.03, 0.05, 0.22, 0.00],
    [0.04, 0.02, 0.03, 0.06, 0.17],
])
P = _L @ _L.T + 0.01 * np.eye(5)   # SPD
c = np.zeros(5)                    # pure minimum-variance objective
mu = np.array([0.10, 0.12, 0.15, 0.09, 0.13])   # expected returns
R_TARGET = 0.12
A = np.vstack([np.ones(5), mu])   # 2 x 5
b = np.array([1.0, R_TARGET])
N, M = 5, 2


def closed_form():
    """Exact KKT solve. Convention: L = 0.5 x'Px + c'x + y'(Ax - b), so
    stationarity is Px + c + A'y = 0 and the KKT system is
    [[P, A'],[A, 0]] [x; y] = [-c; b]."""
    K = np.block([[P, A.T], [A, np.zeros((M, M))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(K, rhs)
    x = sol[:N]
    y = sol[N:]
    return x, y


def kkt_stationarity(x, y):
    """|| Px + c + A'y ||_inf -- must vanish for the returned duals."""
    return float(np.max(np.abs(P @ x + c + A.T @ y)))


def solve_pounce():
    from pounce.qp import solve_qp
    r = solve_qp(P=P, c=c, A=A, b=b)
    # pounce returns y for  Px + c + A'y = 0  (same convention as closed_form)
    return np.asarray(r.x), np.asarray(r.y), float(r.obj), r.status


def solve_cvxpy():
    x = cp.Variable(N)
    con = [A @ x == b]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P))), con)
    prob.solve(solver=cp.CLARABEL)
    # cvxpy dual of (A x == b): for Lagrangian L = f + nu'(Ax-b) it returns
    # nu, matching our +A'y convention.
    y = np.asarray(con[0].dual_value)
    return np.asarray(x.value), y, float(prob.value), prob.status


def solve_ipopt():
    m = pyo.ConcreteModel("minvar")
    m.I = pyo.RangeSet(0, N - 1)
    m.x = pyo.Var(m.I, initialize=0.2)
    m.obj = pyo.Objective(
        expr=0.5 * sum(P[i, j] * m.x[i] * m.x[j]
                       for i in range(N) for j in range(N)))
    m.budget = pyo.Constraint(expr=sum(m.x[i] for i in range(N)) == 1.0)
    m.ret = pyo.Constraint(
        expr=sum(mu[i] * m.x[i] for i in range(N)) == R_TARGET)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    SolverFactory("ipopt").solve(m, options={"tol": 1e-11})
    x = np.array([pyo.value(m.x[i]) for i in range(N)])
    # AMPL marginal m.dual[con] = d obj*/d rhs. For  A x = b  the multiplier
    # in the +A'y convention is  y = -m.dual  (marginal is -y).
    y = -np.array([float(m.dual[m.budget]), float(m.dual[m.ret])])
    return x, y, float(pyo.value(m.obj)), "ipopt"


def main():
    xc, yc = closed_form()
    xp, yp, op, sp = solve_pounce()
    xv, yv, ov, sv = solve_cvxpy()
    xi, yi, oi, si = solve_ipopt()

    obj_cf = float(0.5 * xc @ P @ xc)

    def rowcmp(name, x, y, obj, status):
        return {
            "route": name,
            "status": status,
            "objective": obj,
            "x": x.tolist(),
            "y_budget": float(y[0]),
            "y_return": float(y[1]),
            "x_maxabs_vs_closed_form": float(np.max(np.abs(x - xc))),
            "y_maxabs_vs_closed_form": float(np.max(np.abs(y - yc))),
            "obj_absdiff_vs_closed_form": abs_err(obj, obj_cf),
            "kkt_stationarity_inf": kkt_stationarity(x, y),
        }

    payload = {
        "problem": "convex_QP_minvariance",
        "n_vars": N, "n_eq": M,
        "closed_form": {
            "x": xc.tolist(), "y_budget": float(yc[0]),
            "y_return": float(yc[1]), "objective": obj_cf,
            "kkt_stationarity_inf": kkt_stationarity(xc, yc),
        },
        "routes": {
            "pounce": rowcmp("pounce.solve_qp", xp, yp, op, sp),
            "cvxpy_clarabel": rowcmp("cvxpy(CLARABEL)", xv, yv, ov, sv),
            "ipopt": rowcmp("ipopt", xi, yi, oi, si),
        },
        "multiplier_signs": {
            "closed_form": [float(np.sign(yc[0])), float(np.sign(yc[1]))],
            "pounce": [float(np.sign(yp[0])), float(np.sign(yp[1]))],
            "cvxpy": [float(np.sign(yv[0])), float(np.sign(yv[1]))],
            "ipopt": [float(np.sign(yi[0])), float(np.sign(yi[1]))],
            "all_agree": bool(
                np.allclose(np.sign(yc), np.sign(yp)) and
                np.allclose(np.sign(yc), np.sign(yv)) and
                np.allclose(np.sign(yc), np.sign(yi))),
        },
    }
    import json
    print(json.dumps(payload, indent=2))
    dump_result("p4_qp", payload)


if __name__ == "__main__":
    main()
