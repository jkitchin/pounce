"""gh#987 item 6: the l1 exact-penalty loop reports every pass.

Each pass of the rho loop is a full solve that resets the statistics, so
`iter_count` and the report used to describe only the last one (a solve that
ran 18 + 22 + 22 + 26 iterations reported 26).  Now `iter_count` is the total,
and `statistics.passes` gives (rho, iterations, violation, status) per pass.
"""

import json
import os
import tempfile

import numpy as np
import pytest

pounce = pytest.importorskip("pounce")


class Mpcc:
    """min (x-1)^2 + (y-1)^2   s.t.  x*y = 0,  x, y >= 0  (complementarity)."""

    def objective(self, v):
        return (v[0] - 1.0) ** 2 + (v[1] - 1.0) ** 2

    def gradient(self, v):
        return np.array([2.0 * (v[0] - 1.0), 2.0 * (v[1] - 1.0)])

    def constraints(self, v):
        return np.array([v[0] * v[1]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, v):
        return np.array([v[1], v[0]])

    def hessianstructure(self):
        return np.array([0, 1, 1]), np.array([0, 0, 1])

    def hessian(self, v, lam, of):
        return np.array([2.0 * of, lam[0], 2.0 * of])


def _solve(opts):
    p = pounce.Problem(n=2, m=1, problem_obj=Mpcc(), lb=[0, 0], ub=[10, 10], cl=[0.0], cu=[0.0])
    p.add_option("print_level", 0)
    for k, v in opts.items():
        p.add_option(k, v)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.json")
        x, info = p.solve(x0=np.array([0.5, 0.5]), report_path=path, report_detail="full")
        rep = json.load(open(path))
    return x, info, rep


def test_iter_count_is_the_total_over_the_rho_passes():
    _x, info, rep = _solve(
        {"l1_exact_penalty_barrier": "yes", "l1_penalty_init": 1e-3, "bound_relax_factor": 0.0}
    )
    passes = rep["statistics"].get("passes")
    assert passes, "an l1 solve must carry a per-pass summary"
    assert len(passes) >= 2, f"expected the rho loop to escalate at least once: {passes}"
    total = sum(p["iterations"] for p in passes)
    assert info["iter_count"] == total
    assert rep["statistics"]["iteration_count"] == total
    rhos = [p["rho"] for p in passes]
    assert rhos == sorted(rhos) and rhos[0] == pytest.approx(1e-3)
    # The rows of every pass are in the report, and `first_row` marks the seams.
    assert len(rep["iterations"]) == total + len(passes)  # each pass logs iter 0..N
    assert [p["first_row"] for p in passes] == np.cumsum([0] + [p["iterations"] + 1 for p in passes[:-1]]).tolist()
    # The last pass produced the returned point; its violation is the model's own.
    assert passes[-1]["constraint_violation"] is not None


def test_a_plain_solve_has_no_pass_summary():
    _x, info, rep = _solve({})
    assert "passes" not in rep["statistics"]
    assert info["iter_count"] == rep["statistics"]["iteration_count"]
