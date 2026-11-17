"""`nlp_scaling_untagged_constr` through `Problem.set_problem_scaling`.

Two equality rows at the start x = (1, 1, 1):

* row 0, ``2 x0 + x1 = 0``, residual 3, factor 0.5;
* row 1, ``1000 x0 + x1 = 5``, residual 996, row max 1000.

Stopped at ``max_iter=0``, ``final_constr_viol`` is the larger scaled
residual, so it reads off the factor row 1 got: 1 leaves 996, and the
gradient-based factor 100 / 1000 = 0.1 gives 99.6. The `.nl` path's test,
``crates/pounce-cli/tests/untagged_constr_scaling.rs``, uses the same rows.
"""

import math

import numpy as np
import pytest

import pounce


class TwoRows:
    def objective(self, x):
        return float(x @ x)

    def gradient(self, x):
        return 2.0 * x

    def constraints(self, x):
        return np.array([2.0 * x[0] + x[1], 1000.0 * x[0] + x[1]])

    def jacobianstructure(self):
        return (np.array([0, 0, 1, 1], dtype=np.int64),
                np.array([0, 1, 0, 1], dtype=np.int64))

    def jacobian(self, x):
        return np.array([2.0, 1.0, 1000.0, 1.0])

    def hessianstructure(self):
        idx = np.arange(3, dtype=np.int64)
        return idx, idx

    def hessian(self, x, lagrange, obj_factor):
        return np.full(3, 2.0 * obj_factor)


def _problem(rule, given, x_scaling=None):
    prob = pounce.Problem(
        n=3, m=2, problem_obj=TwoRows(),
        lb=[-1e19] * 3, ub=[1e19] * 3, cl=[0.0, 5.0], cu=[0.0, 5.0],
    )
    for k, v in [("nlp_scaling_method", "user-scaling"),
                 ("nlp_scaling_untagged_constr", rule),
                 ("solver_selection", "nlp"),
                 ("max_iter", 0),
                 ("print_level", 0),
                 ("sb", "yes")]:
        prob.add_option(k, v)
    prob.set_problem_scaling(1.0, x_scaling=x_scaling, g_scaling=[0.5, 1.0],
                             g_scaling_given=given)
    return prob


def _violation(rule, given, x_scaling=None):
    _, info = pounce.Solver(_problem(rule, given, x_scaling)).solve(x0=np.ones(3))
    assert info["final_unscaled_constr_viol"] == pytest.approx(996.0)
    return info["final_constr_viol"]


def test_a_row_marked_not_given_gets_the_gradient_based_factor():
    assert _violation("gradient-based", [True, False]) == pytest.approx(99.6)


def test_the_default_keeps_the_factor_written():
    assert _violation("none", [True, False]) == pytest.approx(996.0)


def test_without_flags_every_row_counts_as_given():
    assert _violation("gradient-based", None) == pytest.approx(996.0)


def test_the_rule_reads_the_jacobian_after_the_variable_factors():
    # x0 scaled by 2 halves its column: row 1's max is 500, factor 0.2.
    got = _violation("gradient-based", [True, False], x_scaling=[2.0, 1.0, 1.0])
    assert got == pytest.approx(199.2)


def test_get_problem_scaling_reports_nan_for_a_row_without_a_factor():
    _, _, g = _problem("gradient-based", [True, False]).get_problem_scaling()
    assert g[0] == 0.5
    assert math.isnan(g[1])


def test_flags_of_the_wrong_length_are_refused():
    with pytest.raises(ValueError, match="g_scaling_given"):
        _problem("gradient-based", [True])
