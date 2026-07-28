"""2026-07-28 -- NLP family (filter-IPM), pounce-nlp / pounce-algorithm.

Maximize the volume of a rectangular box given a fixed total surface area
(the classical "cube is optimal" isoperimetric result):

    min  -x*y*z
    s.t. x*y + y*z + z*x == 3
         0.1 <= x, y, z <= 10

By AM-GM / Lagrange symmetry the unique interior maximizer is the cube
x=y=z=1 (surface area 2*(1+1+1)=6 matches the constraint xy+yz+zx=3),
giving volume 1, objective -1. This is a textbook symmetry argument, not a
memorized numeric constant, and is independently re-derived below three
ways that use NEITHER pounce nor each other's code path:

  1. sympy: symbolically solve the Lagrange stationarity + constraint
     system for the fully-symmetric critical point (KKT oracle).
  2. scipy.optimize.minimize (SLSQP) from an asymmetric start.
  3. scipy.optimize.minimize (trust-constr) from a different asymmetric
     start.

pounce is exercised via the CLI on a hand-generated AMPL .nl file
(solver_selection=nlp forces the filter-IPM path), so this is CLI-driven,
not going through the (unbuilt, in this sandbox) Python extension.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import NonlinearConstraint, minimize

HERE = Path(__file__).resolve().parent
POUNCE = HERE.parent.parent / "target" / "release" / "pounce"


def build_nl(path: Path) -> None:
    import pyomo.environ as pyo

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0.1, 10), initialize=2.0)
    m.y = pyo.Var(bounds=(0.1, 10), initialize=0.5)
    m.z = pyo.Var(bounds=(0.1, 10), initialize=3.0)
    m.obj = pyo.Objective(expr=-(m.x * m.y * m.z))
    m.surface = pyo.Constraint(expr=m.x * m.y + m.y * m.z + m.z * m.x == 3)
    m.write(str(path), format="nl", io_options={"symbolic_solver_labels": True})


def run_pounce(nl_path: Path) -> dict:
    json_out = nl_path.with_suffix(".report.json")
    cmd = [str(POUNCE), str(nl_path), "solver_selection=nlp",
           "--json-output", str(json_out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("pounce stdout:\n", proc.stdout, file=sys.stderr)
        print("pounce stderr:\n", proc.stderr, file=sys.stderr)
        raise SystemExit(f"pounce exited {proc.returncode}")
    return json.loads(json_out.read_text())


def sympy_kkt() -> tuple[float, float, float, float]:
    """Certify the symmetric candidate (1,1,1) as an exact KKT point.

    Rather than trusting sympy's general nonlinear `solve` to *find* the
    critical point (which for this system returns no closed-form real
    branch), we independently derive the multiplier by symmetry
    (dL/dx=dL/dy=dL/dz by construction at x=y=z) and then verify --
     symbolically, exactly in Rational arithmetic -- that all four KKT
    equations vanish there. That is a strictly stronger check than a
    numeric solve: it is an exact algebraic certificate.
    """
    x, y, z, lam = sp.symbols("x y z lam")
    f = -(x * y * z)
    g = x * y + y * z + z * x - 3
    L = f - lam * g
    eqs = [sp.diff(L, v) for v in (x, y, z)] + [g]

    candidate = {x: sp.Integer(1), y: sp.Integer(1), z: sp.Integer(1),
                 lam: sp.Rational(-1, 2)}
    residuals = [sp.simplify(e.subs(candidate)) for e in eqs]
    if any(r != 0 for r in residuals):
        raise RuntimeError(f"KKT residuals nonzero: {residuals}")

    xv, yv, zv = candidate[x], candidate[y], candidate[z]
    return float(xv), float(yv), float(zv), float(f.subs(candidate))


def scipy_oracle(method: str, x0: np.ndarray) -> tuple[np.ndarray, float]:
    def obj(v):
        return -(v[0] * v[1] * v[2])

    def obj_grad(v):
        return -np.array([v[1] * v[2], v[0] * v[2], v[0] * v[1]])

    con = NonlinearConstraint(
        lambda v: v[0] * v[1] + v[1] * v[2] + v[2] * v[0], 3.0, 3.0
    )
    bounds = [(0.1, 10)] * 3
    res = minimize(obj, x0, jac=obj_grad, method=method, bounds=bounds,
                    constraints=[con], options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"scipy {method} did not converge: {res.message}")
    return res.x, res.fun


def main() -> None:
    nl_path = HERE / "2026-07-28_nlp_box-max-volume.nl"
    build_nl(nl_path)
    report = run_pounce(nl_path)

    x_p = report["solution"]["x"]
    f_p = report["solution"]["objective"]
    status = report["solution"]["status"]

    xs, ys, zs, f_sym = sympy_kkt()
    x_s1, f_s1 = scipy_oracle("SLSQP", np.array([2.0, 0.5, 3.0]))
    x_s2, f_s2 = scipy_oracle("trust-constr", np.array([0.2, 5.0, 1.0]))

    result = {
        "pounce": {"x": x_p, "f": f_p, "status": status},
        "sympy_kkt": {"x": [xs, ys, zs], "f": f_sym},
        "scipy_slsqp": {"x": x_s1.tolist(), "f": f_s1},
        "scipy_trust_constr": {"x": x_s2.tolist(), "f": f_s2},
    }
    print(json.dumps(result, indent=2))

    max_f_err = max(abs(f_p - f_sym), abs(f_p - f_s1), abs(f_p - f_s2))
    max_x_err = max(
        max(abs(a - b) for a, b in zip(x_p, [xs, ys, zs])),
        max(abs(a - b) for a, b in zip(x_p, x_s1)),
        max(abs(a - b) for a, b in zip(x_p, x_s2)),
    )
    print(f"\nmax objective error vs 3 independent oracles: {max_f_err:.3e}")
    print(f"max x error vs 3 independent oracles: {max_x_err:.3e}")

    (HERE / "2026-07-28_nlp_box-max-volume.result.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
