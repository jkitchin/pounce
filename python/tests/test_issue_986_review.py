"""gh#986 review fixes, end to end through the Python front end.

* item 10: the issue's SQP repro -- a child that fixes a variable warm-starts
  from the parent's working set (mapped through the fixed-variable
  elimination) and the published working set is in the caller's space;
* item 11: a raising ``intermediate`` reports ``Callback_Error`` from the
  batch drivers too, not ``User_Requested_Stop``;
* item 12: the callback convention is read from the signature once: a body
  that raises ``TypeError`` is not re-run, and a ``*args`` callback is called
  positionally in cyipopt's argument order.
"""
import numpy as np
import pytest

import pounce


class TinyQP:
    """min (x1-0.6)^2 + (x2-0.3)^2  s.t.  x1 + x2 <= 0.8"""

    def objective(self, x):
        return (x[0] - 0.6) ** 2 + (x[1] - 0.3) ** 2

    def gradient(self, x):
        return np.array([2 * (x[0] - 0.6), 2 * (x[1] - 0.3)])

    def constraints(self, x):
        return np.array([x[0] + x[1]])

    def jacobianstructure(self):
        return np.array([0, 0]), np.array([0, 1])

    def jacobian(self, x):
        return np.array([1.0, 1.0])

    def hessianstructure(self):
        return np.array([0, 1]), np.array([0, 1])

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * np.array([2.0, 2.0])


def _sqp(lb, ub):
    p = pounce.Problem(n=2, m=1, problem_obj=TinyQP(), lb=lb, ub=ub,
                       cl=[-1e20], cu=[0.8])
    p.add_option("print_level", 0)
    p.add_option("algorithm", "active-set-sqp")
    return p


def test_sqp_child_with_a_fixed_variable_warm_starts_and_publishes_full_space(caplog):
    xp, ip = _sqp([0, 0], [1, 1]).solve(x0=np.array([0.5, 0.5]))
    assert ip["status_msg"] == "Solve_Succeeded"
    assert ip["working_set"] is not None
    xc, ic = _sqp([0, 0], [0, 1]).solve(x0=np.array([0.0, 0.3]),
                                        working_set=ip["working_set"])
    assert ic["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(xc, [0.0, 0.3], atol=1e-6)
    # Published in the caller's space, the fixed variable marked Fixed (3):
    # the first pass returned None here (a cold solve's reduced set).
    ws = ic["working_set"]
    assert ws is not None
    bounds = np.asarray(ws[0])
    assert bounds.shape == (2,) and bounds[0] == 3
    # A sibling fixing the *other* variable takes the child's set onto the
    # right column.
    xs, is_ = _sqp([0, 0.3], [1, 0.3]).solve(x0=np.array([0.5, 0.3]),
                                            working_set=ws)
    assert is_["status_msg"] == "Solve_Succeeded"
    np.testing.assert_allclose(xs, [0.5, 0.3], atol=1e-6)


class Quad:
    def objective(self, x):
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

    def gradient(self, x):
        return np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])

    def hessianstructure(self):
        return np.array([0, 1]), np.array([0, 1])

    def hessian(self, x, lam, of):
        return of * np.array([2.0, 2.0])


def _quad_problem(obj):
    p = pounce.Problem(n=2, m=0, problem_obj=obj, lb=[-5, -5], ub=[5, 5],
                       cl=[], cu=[])
    p.add_option("print_level", 0)
    return p


class Raises(Quad):
    def __init__(self):
        self.calls = 0

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        self.calls += 1
        raise TypeError("got an unexpected keyword argument 'from_the_body'")


def test_a_body_raising_typeerror_runs_once_and_is_a_callback_error():
    obj = Raises()
    x, info = _quad_problem(obj).solve(x0=np.zeros(2))
    assert obj.calls == 1, "the first pass retried the body positionally"
    assert info["status_msg"] == "Callback_Error"
    assert info["status"] == -198
    assert "from_the_body" in info["callback_error"]


class StarArgs(Quad):
    def __init__(self):
        self.seen = []

    def intermediate(self, *args):
        self.seen.append(args)
        return True


def test_a_star_args_callback_is_called_positionally_in_cyipopt_order():
    obj = StarArgs()
    x, info = _quad_problem(obj).solve(x0=np.zeros(2))
    assert info["status_msg"] == "Solve_Succeeded"
    assert len(obj.seen) >= 2
    for k, args in enumerate(obj.seen):
        # alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
        # regularization_size, alpha_du, alpha_pr, ls_trials
        assert len(args) == 11
        assert args[0] in (0, 1)
        assert args[1] == k                     # iter_count, in order
        assert isinstance(args[10], int)        # ls_trials
        assert args[5] > 0                      # mu
    # obj_value at iteration 0 is f(x0) = 1 + 4.
    assert obj.seen[0][2] == pytest.approx(5.0)


class Renamed(Quad):
    """cyipopt's parameter list under other names: positional too."""

    def __init__(self):
        self.iters = []

    def intermediate(self, mode, it, f, pr, du, mu, dn, rg, adu, apr, ls):
        self.iters.append(it)
        return True


def test_the_cyipopt_parameter_list_under_other_names_is_positional():
    obj = Renamed()
    x, info = _quad_problem(obj).solve(x0=np.zeros(2))
    assert info["status_msg"] == "Solve_Succeeded"
    assert obj.iters == list(range(len(obj.iters)))


class Boom(Quad):
    def intermediate(self, **kw):
        raise RuntimeError("boom in a batch")


def test_batch_reports_a_raising_callback_as_callback_error():
    probs = [_quad_problem(Boom()), _quad_problem(Quad())]
    results = pounce.solve_nlp_batch(probs, x0s=[np.zeros(2), np.zeros(2)],
                                     parallel=False)
    (_, bad), (_, good) = results
    assert bad["status_msg"] == "Callback_Error"
    assert bad["status"] == -198
    assert "boom in a batch" in bad["callback_error"]
    assert good["status_msg"] == "Solve_Succeeded"
    assert "callback_error" not in good
