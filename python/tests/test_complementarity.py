"""Declared complementarity through the Python API: `Problem(...,
complementarity=[(row, var), ...])` solves through the complementarity
driver (`pounce_restoration::mpcc_driver`) and reports which formulation
answered and the answer's stationarity class in `info["complementarity"]`.

A pair `(row, var)` means `x_var` and `g_row(x)` are complementary, each
measured from its one finite bound: `0 <= x_j - lb_j  ⟂  g_i(x) - cl_i >= 0`
(or the upper-bound orientation). The models are the ones the Rust driver
tests and the CLI tests use, so the three entry points answer the same
questions.
"""
from pathlib import Path

import numpy as np
import pytest

import pounce

FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "pounce-cli" / "tests" / "fixtures_mpcc"
INF = 2e19


class QpecSmall:
    """min (x-1)^2 + (y1-1)^2 + y2^2,  0 <= x <= 2,
    0 <= y1 ⟂ 2 y1 - 1 - x >= 0,   0 <= y2 ⟂ x + 2 y2 - 1 >= 0.
    Answer (1, 1, 0), the second pair biactive there; S-stationary."""

    def objective(self, z):
        return (z[0] - 1) ** 2 + (z[1] - 1) ** 2 + z[2] ** 2

    def gradient(self, z):
        return np.array([2 * (z[0] - 1), 2 * (z[1] - 1), 2 * z[2]])

    def constraints(self, z):
        return np.array([2 * z[1] - 1 - z[0], z[0] + 2 * z[2] - 1])

    def jacobianstructure(self):
        return np.array([0, 0, 1, 1]), np.array([0, 1, 0, 2])

    def jacobian(self, z):
        return np.array([-1.0, 2.0, 1.0, 2.0])

    def hessianstructure(self):
        return np.array([0, 1, 2]), np.array([0, 1, 2])

    def hessian(self, z, lagrange, obj_factor):
        return obj_factor * np.array([2.0, 2.0, 2.0])


def qpec_problem(**kw):
    return pounce.Problem(
        n=3,
        m=2,
        problem_obj=QpecSmall(),
        lb=[0.0, 0.0, 0.0],
        ub=[2.0, INF, INF],
        cl=[0.0, 0.0],
        cu=[INF, INF],
        **kw,
    )


class Ralph1:
    """min 2x - y,  x >= 0,  0 <= y ⟂ y - x >= 0. Answer: the origin, which
    admits no S-stationary multiplier (M-stationary only)."""

    def objective(self, z):
        return 2 * z[0] - z[1]

    def gradient(self, z):
        return np.array([2.0, -1.0])

    def constraints(self, z):
        return np.array([z[1] - z[0]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, z):
        return np.array([-1.0, 1.0])

    def hessianstructure(self):
        return np.array([], dtype=int), np.array([], dtype=int)

    def hessian(self, z, lagrange, obj_factor):
        return np.array([])


@pytest.mark.parametrize("x0", [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [2.0, 1.5, 0.0]])
def test_qpec_small_is_solved_and_certified(x0):
    prob = qpec_problem(complementarity=[(0, 1), (1, 2)])
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array(x0))
    np.testing.assert_allclose(x, [1.0, 1.0, 0.0], atol=1e-4)
    cc = info["complementarity"]
    assert cc["accepted_by"] is not None, cc
    assert cc["max_abs_product"] <= 1e-4
    assert cc["certificate"]["class"] == "S", cc
    assert cc["certificate"]["n_biactive"] == 1, cc
    assert "stationarity: S" in cc["summary"]


def test_a_converged_point_that_is_not_strongly_stationary_says_so():
    prob = pounce.Problem(
        n=2, m=1, problem_obj=Ralph1(), lb=[0.0, 0.0], ub=[INF, INF], cl=[0.0], cu=[INF],
        complementarity=[(0, 1)],
    )
    prob.add_option("print_level", 0)
    x, info = prob.solve(np.array([1.0, 2.0]))
    np.testing.assert_allclose(x, [0.0, 0.0], atol=1e-4)
    cc = info["complementarity"]
    assert cc["certificate"]["class"] == "W", cc
    # A W point is reported as a fallback, not as an answer.
    assert cc["accepted_by"] is None, cc
    assert cc["fallback"] is not None, cc


def test_without_pairs_nothing_changes():
    prob = qpec_problem()
    prob.add_option("print_level", 0)
    _, info = prob.solve(np.array([0.0, 0.5, 0.5]))
    assert info["complementarity"] is None
    assert prob.complementarity == []


def test_a_pair_it_cannot_lower_raises_with_the_reason():
    # x is bounded on both sides: a mixed pair, not lowered.
    prob = qpec_problem(complementarity=[(0, 0)])
    prob.add_option("print_level", 0)
    with pytest.raises(ValueError, match="one-sided"):
        prob.solve(np.array([0.0, 0.5, 0.5]))


def test_read_nl_exposes_pairs_and_the_batch_path_refuses_them():
    nl = pounce.read_nl(str(FIXTURES / "qpec_small.nl"))
    assert nl.complementarity == [(0, 1, 1), (1, 2, 1)]
    with pytest.raises(ValueError, match="complementarity"):
        pounce.solve_nlp_batch([nl], [nl.x0])
    # The extension refuses too: the wrapper's check is not the only guard.
    from pounce import _pounce

    with pytest.raises(ValueError, match="complementarity"):
        _pounce.solve_nlp_batch([nl])
    prob = qpec_problem(complementarity=[(0, 1)])
    with pytest.raises(ValueError, match="complementarity"):
        pounce.solve_nlp_batch([prob], [np.array([0.0, 0.5, 0.5])])
