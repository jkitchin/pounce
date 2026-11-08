"""Regenerate `qp_python.json`, the Python side of `tests/qp_python_parity.rs`
(gh#1147 S3-I3).

Each row solves one tiny model through Python's `pounce.qp.solve_qp` or
`pounce.qp.solve_socp` with the row's keyword options, and records two things:

* `model` -- the exact triplets and vectors the frontend hands the binding
  (`pounce.qp._lower_triangle_coo` / `_coo`, in their order, as `_build` does).
  The Rust test feeds these to the `PounceQp*` handle, so both sides solve the
  same `QpProblem`, triplet order included.
* `result` -- what the parity test holds the C API to: `status`, `obj`,
  `iters`, `x`, `y`, `z`, `z_lb`, `z_ub`, the eight `residuals`, `tau`,
  `kappa` and `certificate_scale` (`None` -> JSON null).

Reals are written by Python's `json` (shortest round-trip `repr`), and the
Rust side parses them with serde_json's `float_roundtrip`, so a value is
compared to the bit. An infinite bound entry is written as null (JSON has no
infinity) and read back as the matching infinity. NaN never occurs in a row
(`allow_nan=False` would refuse it).

The references are exact only on the platform they were generated on (Linux
x86_64; the test gates its bitwise comparison on that). Run from the
repository root, with the extension built from this tree (`make python-ext`):

    python crates/pounce-cinterface/tests/common/gen_qp_python.py
"""

import json
import math
from pathlib import Path

import numpy as np

from pounce import qp as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / "qp_python.json"

SQRT2 = 2.0**0.5

# Shared models, as `solve_qp` / `solve_socp` keyword arguments.

# min -x0 - 2 x1  s.t.  x0 - x1 = 0, x0 + x1 <= 1, x >= 0  ->  (1/2, 1/2), -3/2.
LP = dict(c=[-1.0, -2.0], A=[[1.0, -1.0]], b=[0.0], G=[[1.0, 1.0]], h=[1.0], lb=[0.0, 0.0])
# min 1/2 |x|^2 - x0 - x1  s.t.  x0 + x1 <= 1, 0 <= x <= 10  ->  (1/2, 1/2), -3/4.
QP = dict(
    P=np.eye(2), c=[-1.0, -1.0], G=[[1.0, 1.0]], h=[1.0], lb=[0.0, 0.0], ub=[10.0, 10.0]
)
# min 1/2 xT [[2,-1],[-1,2]] x - x0 - x1, unconstrained  ->  (1, 1), -1.
QP_COUPLED = dict(P=[[2.0, -1.0], [-1.0, 2.0]], c=[-1.0, -1.0])
# min 1/2 x0^2 - 1/2 x1^2 over [-1, 1]^2  ->  local minimum (0, +-1), -1/2.
INDEFINITE = dict(P=[[1.0, 0.0], [0.0, -1.0]], c=[0.0, 0.0], lb=[-1.0, -1.0], ub=[1.0, 1.0])
# -x0 <= -1 with x0 <= 0: primal infeasible.
QP_INFEASIBLE = dict(c=[1.0], G=[[-1.0]], h=[-1.0], ub=[0.0])
# min -x0 over x0 >= 0: unbounded.
QP_UNBOUNDED = dict(c=[-1.0], lb=[0.0])

# Distance from (2, -1) to x1 + x2 = 0: (t, x1 - 2, x2 + 1) in Q^3 -> 1/sqrt2.
SOC_DISTANCE = dict(
    c=[1.0, 0.0, 0.0],
    A=[[0.0, 1.0, 1.0]],
    b=[0.0],
    G=-np.eye(3),
    h=[0.0, -2.0, 1.0],
    cones=[("soc", 3)],
)
# python/tests/test_socp.py:22 (test_mixed_orthant_and_soc): (1, 1), -2.
G_MIXED = [[1.0, 0.0], [0.0, 0.0], [0.0, -1.0]]

ROWS = [
    # ---- solve_qp: LP / convex QP on each engine and switch -------------
    ("lp_ipm", "qp", LP, {}),
    ("lp_presolve_crossover", "qp", LP, dict(qp_presolve=True, qp_crossover=True)),
    ("lp_direct", "qp", LP, dict(qp_hsde=False)),
    ("lp_active_set", "qp", LP, dict(method="active-set")),
    ("qp_ipm", "qp", QP, {}),
    ("qp_ipm_tau", "qp", QP, dict(tau=0.9, tau_max=0.99)),
    ("qp_ipm_reg_no_equilibrate", "qp", QP, dict(qp_reg=1e-8, qp_equilibrate=False, tol=1e-10)),
    ("qp_active_set_presolve", "qp", QP, dict(method="active-set", qp_presolve=True)),
    ("qp_coupled", "qp", QP_COUPLED, {}),
    ("indefinite_active_set", "qp", INDEFINITE, dict(method="active-set")),
    ("qp_iteration_limit", "qp", QP, dict(max_iter=1)),
    ("qp_primal_infeasible", "qp", QP_INFEASIBLE, {}),
    ("qp_dual_infeasible", "qp", QP_UNBOUNDED, {}),
    # ---- solve_socp: one model per cone kind, and the statuses ------------
    # min -x0 - x1 s.t. x0 <= 1, x1 <= 2 (nonneg 2): (1, 2), -3.
    (
        "nonneg",
        "socp",
        dict(c=[-1.0, -1.0], G=np.eye(2), h=[1.0, 2.0], cones=[("nonneg", 2)]),
        {},
    ),
    ("soc_distance", "socp", SOC_DISTANCE, {}),
    ("soc_distance_tol", "socp", SOC_DISTANCE, dict(tol=1e-10)),
    # python/tests/test_socp.py:15 (test_projection_onto_soc): P = I with a
    # cone, (3/2, 3/2, 0), -9/4.
    (
        "soc_projection",
        "socp",
        dict(P=np.eye(3), c=[-1.0, -2.0, 0.0], G=-np.eye(3), h=[0.0, 0.0, 0.0], cones=[("soc", 3)]),
        {},
    ),
    (
        "nonneg_and_soc",
        "socp",
        dict(c=[-1.0, -1.0], G=G_MIXED, h=[1.0, 1.0, 0.0], cones=[("nonneg", 1), ("soc", 2)]),
        {},
    ),
    # python/tests/test_socp.py:43 (test_exp_cone_geometric_program): 2.
    (
        "exp_geometric_program",
        "socp",
        dict(
            c=[0.0, 1.0, 1.0],
            G=[
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            h=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            cones=[("exp", 3), ("exp", 3)],
        ),
        {},
    ),
    # python/tests/test_socp.py:96 (test_soc_mixed_with_exp): (5, e), 5 + e.
    (
        "soc_and_exp",
        "socp",
        dict(
            c=[1.0, 1.0],
            G=[[-1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, -1.0]],
            h=[0.0, 3.0, 4.0, 1.0, 1.0, 0.0],
            cones=[("soc", 3), ("exp", 3)],
        ),
        {},
    ),
    # python/tests/test_socp.py:115 (test_power_cone_known_optimum), alpha 0.3:
    # x0 = 2^0.3 * 0.5^0.7.
    (
        "power",
        "socp",
        dict(
            c=[-1.0, 0.0, 0.0],
            A=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            b=[2.0, 0.5],
            G=-np.eye(3),
            h=[0.0, 0.0, 0.0],
            cones=[("pow", 0.3)],
        ),
        {},
    ),
    # python/tests/test_socp.py:152 (test_psd_min_eigenvalue_offdiagonal):
    # lambda_min([[2,1],[1,2]]) = 1, obj -1.
    (
        "psd",
        "socp",
        dict(c=[-1.0], G=[[1.0], [0.0], [1.0]], h=[2.0, SQRT2, 2.0], cones=[("psd", 2)]),
        {},
    ),
    # t <= -1 (nonneg) with (t, x) in Q^2: primal infeasible.
    (
        "soc_primal_infeasible",
        "socp",
        dict(c=[0.0, 0.0], G=[[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]], h=[-1.0, 0.0, 0.0],
             cones=[("nonneg", 1), ("soc", 2)]),
        {},
    ),
    # min -x subject to (t, x) in Q^2, t free: unbounded.
    (
        "soc_dual_infeasible",
        "socp",
        dict(c=[0.0, -1.0], G=-np.eye(2), h=[0.0, 0.0], cones=[("soc", 2)]),
        {},
    ),
    ("soc_iteration_limit", "socp", SOC_DISTANCE, dict(max_iter=2)),
]


def finite_or_null(v):
    return [None if math.isinf(x) else float(x) for x in v]


def model_of(kind, m):
    """The binding's inputs, exactly as `pounce.qp._build` computes them."""
    c = np.asarray(m["c"], dtype=np.float64).ravel()
    n = c.shape[0]
    pr, pc, pv = pq._lower_triangle_coo(m.get("P"), n)
    ar, ac, av = pq._coo(m.get("A"), n, "A")
    gr, gc, gv = pq._coo(m.get("G"), n, "G")
    vec = lambda k: [] if m.get(k) is None else np.asarray(m[k], np.float64).ravel().tolist()  # noqa: E731
    out = dict(
        n=n,
        m_eq=len(vec("b")),
        m_ineq=len(vec("h")),
        p=[pr, pc, pv],
        c=c.tolist(),
        a=[ar, ac, av],
        b=vec("b"),
        g=[gr, gc, gv],
        h=vec("h"),
        lb=finite_or_null(vec("lb")),
        ub=finite_or_null(vec("ub")),
        cones=[],
    )
    if kind == "socp":
        out["cones"] = [[k, float(v)] for k, v in pq._normalize_cones(m["cones"])]
    return out


def result_of(r):
    f = lambda v: None if v is None else float(v)  # noqa: E731
    vec = lambda v: [float(x) for x in np.asarray(v).ravel()]  # noqa: E731
    res = r.residuals
    return dict(
        status=r.status,
        obj=float(r.obj),
        iters=int(r.iters),
        x=vec(r.x),
        y=vec(r.y),
        z=vec(r.z),
        z_lb=vec(r.z_lb),
        z_ub=vec(r.z_ub),
        residuals={k: float(res[k]) for k in sorted(res)},
        tau=f(r.tau),
        kappa=f(r.kappa),
        certificate_scale=f(r.certificate_scale),
    )


def main():
    rows = {}
    for name, kind, model, opts in ROWS:
        assert name not in rows, name
        kw = {k: v for k, v in model.items() if k != "cones"}
        if kind == "qp":
            r = pq.solve_qp(**kw, **opts)
        else:
            r = pq.solve_socp(**kw, cones=model["cones"], **opts)
        rows[name] = dict(
            entry=kind,
            options={k: v for k, v in opts.items()},
            model=model_of(kind, model),
            result=result_of(r),
        )
    doc = dict(
        generator="crates/pounce-cinterface/tests/common/gen_qp_python.py",
        platform="linux-x86_64",
        rows=rows,
    )
    OUT.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
