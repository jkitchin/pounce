"""gh#987 item 3: the report carries the *unregularized* inertia.

`linear_solver.last_inertia` is the inertia of the factorization after the
interior-point method's shift (delta_w) was added, so a solve that stops at a
local maximum reads (n, 0, 0) there. `last_inertia_unregularized` is what the
KKT matrix itself looked like when tried with no shift.
"""

import json
import os
import tempfile

import numpy as np
import pytest

pounce = pytest.importorskip("pounce")


class Bistable:  # the origin is a local maximum of f
    def objective(self, x):
        return (x[0] ** 2 - 1) ** 2 + (x[1] ** 2 - 1) ** 2 + 0.4 * x[0] * x[1]

    def gradient(self, x):
        return np.array(
            [4 * x[0] * (x[0] ** 2 - 1) + 0.4 * x[1], 4 * x[1] * (x[1] ** 2 - 1) + 0.4 * x[0]]
        )

    def hessianstructure(self):
        return (np.array([0, 1, 1]), np.array([0, 0, 1]))

    def hessian(self, x, lam, of):
        return of * np.array([12 * x[0] ** 2 - 4, 0.4, 12 * x[1] ** 2 - 4])


def _solve(x0, **opts):
    p = pounce.Problem(n=2, m=0, problem_obj=Bistable(), lb=[-2, -2], ub=[2, 2], cl=[], cu=[])
    p.add_option("print_level", 0)
    for k, v in opts.items():
        p.add_option(k, v)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r.json")
        x, info = p.solve(x0=np.asarray(x0, dtype=float), report_path=path, report_detail="full")
        rep = json.load(open(path))
    return x, info, rep


def test_a_local_maximum_reads_indefinite_before_the_shift():
    x, info, rep = _solve([0.0, 0.0], neg_curv_escapes=0)
    ls = rep["linear_solver"]
    assert np.allclose(x, 0.0)
    # What the shift made it...
    assert ls["last_inertia"] == [2, 0, 0]
    assert rep["iterations"][-1]["regularization"] > 0
    # ...and what it was.
    assert ls["last_inertia_unregularized"] == [0, 2, 0]
    assert info["linear_solver"]["last_inertia_unregularized"] == (0, 2, 0)


def test_a_regular_minimum_needs_no_shift():
    x, _info, rep = _solve([1.2, -1.1])
    ls = rep["linear_solver"]
    assert abs(abs(x[0]) - 1.0) < 0.2
    assert ls["last_inertia_unregularized"] == [2, 0, 0]
    assert ls["last_inertia"] == [2, 0, 0]
