"""2026-07-28 -- QP family (convex interior-point), pounce-convex `qp-ipm`.

Diagonal (separable), strictly convex QP with a single linear equality:

    min  0.5 * sum_i q_i x_i^2 - c_i x_i
    s.t. a'x = r

Because Q = diag(q) is diagonal, the KKT stationarity for a *fixed*
multiplier lambda gives x_i(lambda) = (c_i + lambda*a_i) / q_i in closed
form; substituting into a'x=r yields a single SCALAR linear equation for
lambda, solvable exactly by hand (no linear-algebra package, let alone an
LP/QP solver). That closed form is oracle #1.

Oracle #2 is cvxpy (OSQP/Clarabel backend) -- a completely independent
convex solver stack.

pounce is forced onto the convex QP-IPM path via
`solver_selection=qp-ipm` on a hand-generated AMPL .nl file (CLI-driven,
since the Python extension isn't built in this sandbox).
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

rng = np.random.default_rng(7182026)
N = 6
q = rng.uniform(0.5, 5.0, size=N).round(3)   # diagonal Hessian entries > 0
c = rng.uniform(-4, 4, size=N).round(3)
a = rng.uniform(0.5, 3.0, size=N).round(3)
R_TARGET = 4.0


def build_nl(path: Path) -> None:
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, N - 1)
    m.x = pyo.Var(m.I, initialize=0.0)
    m.obj = pyo.Objective(
        expr=sum(0.5 * q[i] * m.x[i] ** 2 - c[i] * m.x[i] for i in m.I)
    )
    m.eq = pyo.Constraint(expr=sum(a[i] * m.x[i] for i in m.I) == R_TARGET)
    m.write(str(path), format="nl", io_options={"symbolic_solver_labels": True})


def run_pounce(nl_path: Path) -> dict:
    json_out = nl_path.with_suffix(".report.json")
    cmd = [str(POUNCE), str(nl_path), "solver_selection=qp-ipm",
           "--json-output", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("pounce stdout:\n", proc.stdout, file=sys.stderr)
        print("pounce stderr:\n", proc.stderr, file=sys.stderr)
        raise SystemExit(f"pounce exited {proc.returncode}")
    return json.loads(json_out.read_text())


def closed_form():
    """x_i(lambda) = (c_i + lambda*a_i)/q_i ; solve scalar eq a'x(lambda)=R."""
    # a' x(lambda) = sum a_i(c_i + lambda a_i)/q_i = sum(a_i c_i/q_i)
    #              + lambda * sum(a_i^2/q_i) = R
    num = np.sum(a * c / q)
    den = np.sum(a ** 2 / q)
    lam = (R_TARGET - num) / den
    x = (c + lam * a) / q
    f = float(0.5 * np.sum(q * x ** 2) - np.sum(c * x))
    return x, f, lam


def cvxpy_oracle():
    x = cp.Variable(N)
    obj = cp.Minimize(0.5 * cp.sum(cp.multiply(q, cp.square(x))) - c @ x)
    cons = [a @ x == R_TARGET]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"cvxpy status {prob.status}")
    return x.value, float(prob.value)


def main() -> None:
    nl_path = HERE / "2026-07-28_qp_equality.nl"
    build_nl(nl_path)
    report = run_pounce(nl_path)

    x_p = np.array(report["solution"]["x"])
    f_p = report["solution"]["objective"]
    status = report["solution"]["status"]

    x_c, f_c, lam = closed_form()
    x_cvx, f_cvx = cvxpy_oracle()

    result = {
        "pounce": {"x": x_p.tolist(), "f": f_p, "status": status},
        "closed_form": {"x": x_c.tolist(), "f": f_c, "lambda": lam},
        "cvxpy_clarabel": {"x": x_cvx.tolist(), "f": f_cvx},
    }
    print(json.dumps(result, indent=2))

    f_err = max(abs(f_p - f_c), abs(f_p - f_cvx))
    x_err = max(float(np.max(np.abs(x_p - x_c))), float(np.max(np.abs(x_p - x_cvx))))
    print(f"\nmax objective error vs 2 independent oracles: {f_err:.3e}")
    print(f"max x error vs 2 independent oracles: {x_err:.3e}")

    (HERE / "2026-07-28_qp_equality.result.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
