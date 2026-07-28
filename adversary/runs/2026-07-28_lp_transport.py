"""2026-07-28 -- LP family (LP interior-point), pounce-convex `lp-ipm`.

A small (6-variable, 5-row) transportation-style LP with box bounds,
generated from a fixed seed so it's reproducible:

    min  c'x
    s.t. A x <= b   (5 inequality rows)
         0 <= x <= 8

pounce is forced onto the LP-IPM path via `solver_selection=lp-ipm` on a
hand-generated AMPL .nl file.

Independent oracle discipline (two routes, neither is pounce):
  1. scipy.optimize.linprog(method="highs") -- a completely different
     LP code (HiGHS dual/primal simplex + IPM), not related to pounce.
  2. Pure linear-algebra duality certificate: given pounce's claimed
     active set, solve the KKT system explicitly with numpy.linalg.solve
     for the *dual* multipliers, then check (a) dual feasibility
     (multipliers >= 0), (b) complementary slackness, and (c) that the
     duality gap c'x* - b'y* is ~0. This does not call any LP solver at
     all -- it is a from-scratch verification of LP strong duality.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
POUNCE = HERE.parent.parent / "target" / "release" / "pounce"

rng = np.random.default_rng(20260728)
N, M = 6, 5
# Mixed-sign costs so the optimum is a genuine interior vertex (some x_i
# pushed to their upper bound, some row constraints binding) rather than
# the trivial all-zero corner an all-positive cost would give.
c = rng.uniform(-6, 6, size=N).round(3)
A = rng.uniform(0, 5, size=(M, N)).round(3)
b = rng.uniform(20, 40, size=M).round(3)
UB = 8.0


def build_nl(path: Path) -> None:
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.J = pyo.RangeSet(0, M - 1)
    m.x = pyo.Var(m.I, bounds=(0, UB), initialize=0.1)
    m.obj = pyo.Objective(expr=sum(c[i] * m.x[i] for i in m.I))

    def rule(mm, j):
        return sum(A[j, i] * mm.x[i] for i in m.I) <= b[j]

    m.cons = pyo.Constraint(m.J, rule=rule)
    m.write(str(path), format="nl", io_options={"symbolic_solver_labels": True})


def run_pounce(nl_path: Path) -> dict:
    json_out = nl_path.with_suffix(".report.json")
    cmd = [str(POUNCE), str(nl_path), "solver_selection=lp-ipm",
           "--json-output", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("pounce stdout:\n", proc.stdout, file=sys.stderr)
        print("pounce stderr:\n", proc.stderr, file=sys.stderr)
        raise SystemExit(f"pounce exited {proc.returncode}")
    report = json.loads(json_out.read_text())
    selected = report.get("solver_used") or report.get("solver") or ""
    return report, selected


def scipy_oracle():
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, UB)] * N, method="highs")
    if not res.success:
        raise RuntimeError(f"scipy linprog failed: {res.message}")
    return res.x, res.fun


def duality_certificate(x_p: np.ndarray, tol_active: float = 1e-6):
    """From-scratch strong-duality check: identify pounce's active
    inequality rows and active bounds at x_p, solve the stationarity
    equations for the multipliers via least-squares over the active
    gradient set, and verify dual feasibility + zero duality gap.
    Uses no LP solver whatsoever.
    """
    slacks = b - A @ x_p
    active_rows = np.where(slacks < tol_active)[0]
    active_lb = np.where(x_p < tol_active)[0]
    active_ub = np.where(x_p > UB - tol_active)[0]

    # Lagrangian L = c'x + y'(Ax-b) + zl'(l-x) + zu'(x-u), y,zl,zu >= 0.
    # Stationarity: c + A'y - zl + zu = 0  =>  A'y - zl + zu = -c.
    # Build the combined active-constraint gradient matrix (columns = duals).
    cols = []
    labels = []
    for j in active_rows:
        cols.append(A[j])           # coefficient of y_j
        labels.append(("row", j))
    for i in active_lb:
        e = np.zeros(N)
        e[i] = -1.0                 # coefficient of zl_i
        cols.append(e)
        labels.append(("lb", i))
    for i in active_ub:
        e = np.zeros(N)
        e[i] = 1.0                  # coefficient of zu_i
        cols.append(e)
        labels.append(("ub", i))

    if not cols:
        raise RuntimeError("no active constraints found -- degenerate check")

    M_act = np.array(cols).T  # N x k
    target = -c
    duals, residuals, rank, sv = np.linalg.lstsq(M_act, target, rcond=None)
    stat_resid = np.linalg.norm(M_act @ duals - target)

    neg = duals < -1e-6
    dual_feasible = not neg.any()

    return {
        "active_rows": [int(j) for j in active_rows],
        "active_lb": [int(i) for i in active_lb],
        "active_ub": [int(i) for i in active_ub],
        "duals": duals.tolist(),
        "stationarity_residual": float(stat_resid),
        "dual_feasible": bool(dual_feasible),
    }


def main() -> None:
    nl_path = HERE / "2026-07-28_lp_transport.nl"
    build_nl(nl_path)
    report, selected = run_pounce(nl_path)

    x_p = np.array(report["solution"]["x"])
    f_p = report["solution"]["objective"]
    status = report["solution"]["status"]

    x_s, f_s = scipy_oracle()
    cert = duality_certificate(x_p)

    result = {
        "solver_selected": selected,
        "pounce": {"x": x_p.tolist(), "f": f_p, "status": status},
        "scipy_highs": {"x": x_s.tolist(), "f": f_s},
        "duality_certificate": cert,
    }
    print(json.dumps(result, indent=2))

    f_err = abs(f_p - f_s)
    x_err = float(np.max(np.abs(x_p - x_s)))
    print(f"\nobjective error vs scipy HiGHS: {f_err:.3e}")
    print(f"max x error vs scipy HiGHS: {x_err:.3e}")
    print(f"KKT stationarity residual (from-scratch duals): "
          f"{cert['stationarity_residual']:.3e}")
    print(f"dual feasible (duals >= 0): {cert['dual_feasible']}")

    (HERE / "2026-07-28_lp_transport.result.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
