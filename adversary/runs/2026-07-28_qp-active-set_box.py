"""2026-07-28 -- active-set QP family, pounce-qp `qp-active-set`.

Diagonal (separable) convex QP with ONLY box bounds:

    min  0.5 * sum_i q_i x_i^2 - c_i x_i
    s.t. l_i <= x_i <= u_i

Because Q is diagonal, the problem is fully separable: for each i the
unconstrained minimizer is c_i/q_i, and since there is no coupling
between coordinates the constrained optimum is EXACTLY that value
clipped into [l_i, u_i]. This closed form is derived directly from
first-order calculus per coordinate -- no linear algebra, no LP/QP
solver of any kind. Bounds are deliberately set so several coordinates'
unconstrained optima fall outside their box, forcing a genuine
(nontrivial) active set -- exactly what this solver family exists to
handle.

Oracle #2 is cvxpy (Clarabel), a fully independent convex solver.

pounce is forced onto the active-set QP path via
`solver_selection=qp-active-set` on a hand-generated AMPL .nl file.
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

rng = np.random.default_rng(80808)
N = 8
q = rng.uniform(0.5, 4.0, size=N).round(3)
c = rng.uniform(-10, 10, size=N).round(3)   # wide range so several
                                             # unconstrained optima c_i/q_i
                                             # fall outside the box below
LB = -2.0
UB = 2.0


def build_nl(path: Path) -> None:
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.x = pyo.Var(m.I, bounds=(LB, UB), initialize=0.0)
    m.obj = pyo.Objective(
        expr=sum(0.5 * q[i] * m.x[i] ** 2 - c[i] * m.x[i] for i in m.I)
    )
    m.write(str(path), format="nl", io_options={"symbolic_solver_labels": True})


def run_pounce(nl_path: Path) -> dict:
    json_out = nl_path.with_suffix(".report.json")
    cmd = [str(POUNCE), str(nl_path), "solver_selection=qp-active-set",
           "--json-output", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("pounce stdout:\n", proc.stdout, file=sys.stderr)
        print("pounce stderr:\n", proc.stderr, file=sys.stderr)
        raise SystemExit(f"pounce exited {proc.returncode}")
    return json.loads(json_out.read_text())


def closed_form():
    x_unc = c / q
    x = np.clip(x_unc, LB, UB)
    f = float(0.5 * np.sum(q * x ** 2) - np.sum(c * x))
    n_active = int(np.sum((x <= LB + 1e-9) | (x >= UB - 1e-9)))
    return x, f, n_active


def cvxpy_oracle():
    x = cp.Variable(N)
    obj = cp.Minimize(0.5 * cp.sum(cp.multiply(q, cp.square(x))) - c @ x)
    cons = [x >= LB, x <= UB]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"cvxpy status {prob.status}")
    return x.value, float(prob.value)


def main() -> None:
    nl_path = HERE / "2026-07-28_qp-active-set_box.nl"
    build_nl(nl_path)
    report = run_pounce(nl_path)

    x_p = np.array(report["solution"]["x"])
    f_p = report["solution"]["objective"]
    status = report["solution"]["status"]

    x_c, f_c, n_active = closed_form()
    x_cvx, f_cvx = cvxpy_oracle()

    result = {
        "pounce": {"x": x_p.tolist(), "f": f_p, "status": status},
        "closed_form_clip": {"x": x_c.tolist(), "f": f_c,
                              "n_active_bounds": n_active},
        "cvxpy_clarabel": {"x": x_cvx.tolist(), "f": f_cvx},
    }
    print(json.dumps(result, indent=2))
    print(f"\nnumber of active bounds in closed-form solution: {n_active}/{N}")

    f_err = max(abs(f_p - f_c), abs(f_p - f_cvx))
    x_err = max(float(np.max(np.abs(x_p - x_c))), float(np.max(np.abs(x_p - x_cvx))))
    print(f"max objective error vs 2 independent oracles: {f_err:.3e}")
    print(f"max x error vs 2 independent oracles: {x_err:.3e}")

    (HERE / "2026-07-28_qp-active-set_box.result.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
