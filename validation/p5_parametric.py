"""P5 -- Parametric sensitivity dx/dp and dObj/dp vs finite differences.

A SECOND, independent sensitivity mechanism (distinct from P1's covariance):
the parametric derivative of the solution and objective with respect to a
model parameter p, delivered by pyomo_pounce's declare_sens_param / gradient
(the sIPOPT computation from the held KKT factorization). Two things are
checked against convention-free oracles:

  (a) ENVELOPE THEOREM -- pounce's constraint multiplier equals dObj*/dp (up
      to sign), matched to a central finite difference AND to the analytic
      value.
  (b) dx/dp from the sensitivity backsolve matches a central finite
      difference to O(delta^2): a step-size sweep shows the error shrinking
      quadratically as delta -> 0.

Problem (smooth, nonlinear x*(p) so the FD error is genuinely O(delta^2)):

    min  -x1 - x2   s.t.  x1^2 + x2^2 == p,   x1,x2 >= 0

    x*(p) = (sqrt(p/2), sqrt(p/2)),  Obj*(p) = -sqrt(2p),
    multiplier (L = f + lam*(x1^2+x2^2-p)) lam = 1/sqrt(2p),
    dObj*/dp = -lam = -1/sqrt(2p)   (envelope theorem).
"""
from __future__ import annotations

import math

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import pyomo_pounce
from pyomo_pounce import declare_sens_param, gradient
from _common import abs_err, dump_result, setup_pounce

setup_pounce()

P0 = 2.0  # x* = (1, 1), Obj* = -2, lam = 1/2


def analytic(p):
    return {
        "x1": math.sqrt(p / 2.0),
        "x2": math.sqrt(p / 2.0),
        "obj": -math.sqrt(2.0 * p),
        "lam": 1.0 / math.sqrt(2.0 * p),      # L = f + lam*(g - p)
        "dObj_dp": -1.0 / math.sqrt(2.0 * p),  # envelope: dObj*/dp = -lam
        "dx1_dp": 1.0 / (2.0 * math.sqrt(2.0 * p)),
    }


def build(p_value):
    m = pyo.ConcreteModel("param")
    m.x1 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.x2 = pyo.Var(bounds=(0, None), initialize=1.0)
    m.p = pyo.Param(initialize=p_value, mutable=True)
    m.obj = pyo.Objective(expr=-m.x1 - m.x2)
    m.con = pyo.Constraint(expr=m.x1 ** 2 + m.x2 ** 2 == m.p)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve_ipopt(p_value):
    m = build(p_value)
    SolverFactory("ipopt").solve(m, options={"tol": 1e-10})
    return (pyo.value(m.x1), pyo.value(m.x2), pyo.value(m.obj),
            float(m.dual[m.con]))


def solve_pounce_plain(p_value):
    """Plain pounce solve (CLI/ASL) for x, obj and the constraint dual."""
    m = build(p_value)
    SolverFactory("pounce").solve(m)
    return (pyo.value(m.x1), pyo.value(m.x2), pyo.value(m.obj),
            float(m.dual[m.con]))


def main():
    ana = analytic(P0)

    # ── pounce parametric sensitivity session ──────────────────────────────
    m = build(P0)
    declare_sens_param(m.p)
    res = SolverFactory("pounce").solve(m)
    x1 = pyo.value(m.x1)
    x2 = pyo.value(m.x2)
    obj = pyo.value(m.obj)
    dual_con = float(m.dual[m.con]) if m.con in m.dual else None
    dx1_dp = gradient(m.x1, wrt=m.p)
    dx2_dp = gradient(m.x2, wrt=m.p)

    # ── convention-free central-FD oracle, step-size sweep ─────────────────
    deltas = [1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3]
    sweep = []
    for d in deltas:
        xp1, xp2, objp, _ = solve_ipopt(P0 + d)
        xm1, xm2, objm, _ = solve_ipopt(P0 - d)
        fd_dx1 = (xp1 - xm1) / (2 * d)
        fd_dobj = (objp - objm) / (2 * d)
        sweep.append({
            "delta": d,
            "fd_dx1_dp": fd_dx1,
            "fd_dObj_dp": fd_dobj,
            "err_dx1_vs_pounce": abs(fd_dx1 - dx1_dp),
            "err_dx1_vs_analytic": abs(fd_dx1 - ana["dx1_dp"]),
            "err_dObj_vs_analytic": abs(fd_dobj - ana["dObj_dp"]),
        })

    # quadratic-convergence ratio: halving delta should quarter the FD error
    ratios = []
    for i in range(1, len(sweep)):
        e_prev = sweep[i - 1]["err_dx1_vs_analytic"]
        e_cur = sweep[i]["err_dx1_vs_analytic"]
        if e_cur > 0:
            ratios.append(e_prev / e_cur)

    # ── envelope theorem: multiplier vs dObj/dp ────────────────────────────
    # AMPL marginal m.dual[con] = dObj*/d(rhs=p). Compare to FD dObj/dp and to
    # the analytic dObj*/dp = -lam. The finest-delta FD is the reference.
    fd_dobj_best = sweep[-1]["fd_dObj_dp"]
    _, _, _, ip_dual = solve_ipopt(P0)
    _, _, _, po_dual = solve_pounce_plain(P0)

    payload = {
        "problem": "parametric_NLP",
        "p0": P0,
        "analytic": ana,
        "pounce": {
            "status": str(res.solver.termination_condition),
            "x1": x1, "x2": x2, "obj": obj,
            "dual_con": dual_con,
            "dx1_dp_sensitivity": dx1_dp,
            "dx2_dp_sensitivity": dx2_dp,
        },
        "fd_sweep": sweep,
        "quadratic_convergence_ratios_err_dx1": ratios,
        "envelope": {
            "analytic_dObj_dp": ana["dObj_dp"],
            "fd_dObj_dp_finest": fd_dobj_best,
            "pounce_marginal_dual_con": po_dual,
            "ipopt_marginal_dual_con": ip_dual,
            # AMPL marginal should equal dObj/dp (envelope) directly
            "pounce_dual_vs_fd_dObj_dp": abs_err(po_dual, fd_dobj_best),
            "pounce_dual_vs_analytic_dObj_dp": abs_err(po_dual,
                                                       ana["dObj_dp"]),
            "pounce_vs_ipopt_dual": abs_err(po_dual, ip_dual),
        },
        "agreement": {
            "dx1_dp_pounce_vs_analytic": abs_err(dx1_dp, ana["dx1_dp"]),
            "dx1_dp_pounce_vs_fd_finest": abs(dx1_dp - sweep[-1]["fd_dx1_dp"]),
            "x_maxabs_vs_analytic": max(abs_err(x1, ana["x1"]),
                                        abs_err(x2, ana["x2"])),
        },
    }
    import json
    print(json.dumps(payload, indent=2, default=float))
    dump_result("p5_parametric", payload)


if __name__ == "__main__":
    main()
