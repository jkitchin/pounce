"""2026-07-28 -- conic (SOCP) family, pounce-convex `socp` (QCQP -> SOC).

Minimize a linear function over a Euclidean ball:

    min  c'x
    s.t. x1^2 + x2^2 + x3^2 + x4^2 <= r^2

This is a convex QCQP (one quadratic <=-upper-bound constraint, PSD
quadratic form), which pounce's classifier routes to the conic solver
under `solver_selection=socp` (reformulated internally to a second-order
cone).

Oracle #1 (closed form, no solver at all): minimizing a linear functional
over a ball is elementary vector geometry -- the minimizer sits on the
boundary in the direction opposite the gradient:

    x* = -r * c / ||c||_2,   f* = -r * ||c||_2

Oracle #2 is cvxpy (Clarabel), expressing the same SOC constraint natively
via cp.SOC / cp.norm, a fully independent conic solver stack.

pounce is forced onto the conic path via `solver_selection=socp` on a
hand-generated AMPL .nl file.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
POUNCE = HERE.parent.parent / "target" / "release" / "pounce"

rng = np.random.default_rng(4224)
N = 4
c = rng.uniform(-5, 5, size=N).round(3)
R = 3.0


def build_nl(path: Path) -> None:
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.x = pyo.Var(m.I, bounds=(-10, 10), initialize=0.1)
    m.obj = pyo.Objective(expr=sum(c[i] * m.x[i] for i in m.I))
    m.ball = pyo.Constraint(expr=sum(m.x[i] ** 2 for i in m.I) <= R ** 2)
    m.write(str(path), format="nl", io_options={"symbolic_solver_labels": True})


def run_pounce(nl_path: Path) -> dict:
    json_out = nl_path.with_suffix(".report.json")
    cmd = [str(POUNCE), str(nl_path), "solver_selection=socp",
           "--json-output", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("pounce stdout:\n", proc.stdout, file=sys.stderr)
        print("pounce stderr:\n", proc.stderr, file=sys.stderr)
        raise SystemExit(f"pounce exited {proc.returncode}")
    return json.loads(json_out.read_text())


def closed_form():
    norm_c = np.linalg.norm(c)
    x = -R * c / norm_c
    f = float(c @ x)
    return x, f


def cvxpy_oracle():
    x = cp.Variable(N)
    obj = cp.Minimize(c @ x)
    cons = [cp.norm(x, 2) <= R]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"cvxpy status {prob.status}")
    return x.value, float(prob.value)


def main() -> None:
    nl_path = HERE / "2026-07-28_socp_linear-over-ball.nl"
    build_nl(nl_path)
    report = run_pounce(nl_path)

    x_p = np.array(report["solution"]["x"])
    f_p = report["solution"]["objective"]
    status = report["solution"]["status"]

    x_c, f_c = closed_form()
    x_cvx, f_cvx = cvxpy_oracle()

    result = {
        "pounce": {"x": x_p.tolist(), "f": f_p, "status": status},
        "closed_form": {"x": x_c.tolist(), "f": f_c},
        "cvxpy_clarabel": {"x": x_cvx.tolist(), "f": f_cvx},
    }
    print(json.dumps(result, indent=2))

    f_err = max(abs(f_p - f_c), abs(f_p - f_cvx))
    x_err = max(float(np.max(np.abs(x_p - x_c))), float(np.max(np.abs(x_p - x_cvx))))
    print(f"\nmax objective error vs 2 independent oracles: {f_err:.3e}")
    print(f"max x error vs 2 independent oracles: {x_err:.3e}")

    (HERE / "2026-07-28_socp_linear-over-ball.result.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
