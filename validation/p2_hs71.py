"""P2 -- Hock-Schittkowski problem 71.

Independent, published ground truth (H&S 1981) for BOTH the primal solution
and the constraint multipliers. Because the reference multipliers come from a
source that predates and is independent of any modern interior-point code, an
agreement here cannot be an artifact of two solvers sharing a convention. This
is the load-bearing check for the #271/#272 constraint-dual SIGN fix.

    min  x1*x4*(x1+x2+x3) + x3
    s.t. x1*x2*x3*x4 >= 25            (c1, inequality)
         x1^2+x2^2+x3^2+x4^2 == 40    (c2, equality)
         1 <= xi <= 5,  x0 = (1,5,5,1)

Two independent oracles are used besides the two solvers:
  * published (H&S 1981): f* = 17.0140173, x* = (1, 4.7429994, 3.8211503, 1.3794082)
  * a machine-precision solve of the KKT system (the active set is x1 on its
    lower bound, c1 active, c2 the equality), giving f*, x*, and the exact
    multipliers to ~1e-14. This is computed here, uses NEITHER solver, and
    fixes the sign convention analytically.
"""
from __future__ import annotations

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.optimize import fsolve

from _common import abs_err, dump_result, rel_err, setup_pounce

setup_pounce()  # pin the correct pounce binary (see _common.setup_pounce)

# Published optimum (Hock & Schittkowski 1981).
F_PUB = 17.0140173
X_PUB = (1.0, 4.7429994, 3.8211503, 1.3794082)


# ── analytic KKT oracle (uses neither solver) ──────────────────────────────────
def _grads(x):
    x1, x2, x3, x4 = x
    gf = np.array([x4 * (2 * x1 + x2 + x3), x1 * x4, x1 * x4 + 1.0,
                   x1 * (x1 + x2 + x3)])
    g1 = np.array([x2 * x3 * x4, x1 * x3 * x4, x1 * x2 * x4, x1 * x2 * x3])
    g2 = np.array([2 * x1, 2 * x2, 2 * x3, 2 * x4])
    return gf, g1, g2


def exact_kkt():
    """Solve the KKT system to machine precision. Active set: x1 at lower
    bound, c1 active, c2 equality. L = f - mu1 g1 - mu2 g2 - z1 (x1 - 1)."""
    def F(v):
        x = v[:4]
        mu1, mu2, z1 = v[4], v[5], v[6]
        gf, g1, g2 = _grads(x)
        stat = gf - mu1 * g1 - mu2 * g2 - z1 * np.array([1.0, 0, 0, 0])
        x1, x2, x3, x4 = x
        return [*stat, x1 - 1.0, x1 * x2 * x3 * x4 - 25.0,
                x1**2 + x2**2 + x3**2 + x4**2 - 40.0]

    v = fsolve(F, [1, 4.743, 3.821, 1.379, 0.552, -0.161, 1.088], xtol=1e-14)
    x = v[:4]
    f = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    return {"f": float(f), "x": [float(t) for t in x],
            "mu1": float(v[4]), "mu2": float(v[5]), "z1": float(v[6])}


def stationarity_residual(x, mu1, mu2):
    """KKT stationarity residual using the solver's reported multipliers,
    with the x1 (active-bound) component projected out -- the bound
    multiplier z1 absorbs exactly that direction, so the free-space residual
    must vanish if (mu1, mu2) are the correct constraint multipliers."""
    gf, g1, g2 = _grads(x)
    r = gf - mu1 * g1 - mu2 * g2
    r_free = r.copy()
    r_free[0] = 0.0  # project out the bound-active x1 direction
    return float(np.linalg.norm(r_free))


# ── model / solves ─────────────────────────────────────────────────────────────
def build():
    m = pyo.ConcreteModel("HS71")
    m.x = pyo.Var([1, 2, 3, 4], bounds=(1.0, 5.0))
    m.x[1] = 1.0
    m.x[2] = 5.0
    m.x[3] = 5.0
    m.x[4] = 1.0
    m.obj = pyo.Objective(
        expr=m.x[1] * m.x[4] * (m.x[1] + m.x[2] + m.x[3]) + m.x[3])
    m.c1 = pyo.Constraint(expr=m.x[1] * m.x[2] * m.x[3] * m.x[4] >= 25.0)
    m.c2 = pyo.Constraint(
        expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2 + m.x[4] ** 2 == 40.0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve(which: str) -> dict:
    m = build()
    opt = SolverFactory(which)
    if which == "ipopt":
        opt.options["tol"] = 1e-9  # well-converged reference
    res = opt.solve(m)
    x = [pyo.value(m.x[i]) for i in (1, 2, 3, 4)]
    dc1 = float(m.dual[m.c1])
    dc2 = float(m.dual[m.c2])
    return {
        "solver": which,
        "status": str(res.solver.termination_condition),
        "objective": pyo.value(m.obj),
        "x": x,
        "dual_c1": dc1,
        "dual_c2": dc2,
        "kkt_stationarity_residual": stationarity_residual(x, dc1, dc2),
    }


def main():
    ex = exact_kkt()
    p = solve("pounce")
    i = solve("ipopt")

    def errs(s):
        return {
            "obj_err_vs_exact": abs_err(s["objective"], ex["f"]),
            "x_maxabs_vs_exact": max(abs_err(a, b)
                                     for a, b in zip(s["x"], ex["x"])),
            "dual_c1_err_vs_exact": abs_err(s["dual_c1"], ex["mu1"]),
            "dual_c2_err_vs_exact": abs_err(s["dual_c2"], ex["mu2"]),
            "dual_c1_sign_matches_exact":
                (s["dual_c1"] > 0) == (ex["mu1"] > 0),
            "dual_c2_sign_matches_exact":
                (s["dual_c2"] > 0) == (ex["mu2"] > 0),
        }

    payload = {
        "problem": "HS71",
        "published": {"f_star": F_PUB, "x_star": list(X_PUB),
                      "source": "Hock & Schittkowski 1981"},
        "exact_kkt_oracle": ex,
        "pounce": p,
        "ipopt": i,
        "agreement": {
            "obj_absdiff_pounce_vs_ipopt": abs_err(
                p["objective"], i["objective"]),
            "x_maxabs_pounce_vs_ipopt": max(
                abs_err(a, b) for a, b in zip(p["x"], i["x"])),
            "dual_c1_pounce": p["dual_c1"],
            "dual_c1_ipopt": i["dual_c1"],
            "dual_c1_exact": ex["mu1"],
            "dual_c1_absdiff_pounce_vs_ipopt":
                abs_err(p["dual_c1"], i["dual_c1"]),
            "dual_c2_pounce": p["dual_c2"],
            "dual_c2_ipopt": i["dual_c2"],
            "dual_c2_exact": ex["mu2"],
            "dual_c2_absdiff_pounce_vs_ipopt":
                abs_err(p["dual_c2"], i["dual_c2"]),
            "pounce_vs_exact": errs(p),
            "ipopt_vs_exact": errs(i),
            "all_three_dual_c1_same_sign":
                (p["dual_c1"] > 0) == (i["dual_c1"] > 0) == (ex["mu1"] > 0),
            "all_three_dual_c2_same_sign":
                (p["dual_c2"] < 0) == (i["dual_c2"] < 0) == (ex["mu2"] < 0),
        },
    }
    import json
    print(json.dumps(payload, indent=2))
    dump_result("p2_hs71", payload)


if __name__ == "__main__":
    main()
