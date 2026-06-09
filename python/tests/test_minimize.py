"""Smoke tests for the scipy.optimize-style facade."""

import numpy as np
import pytest

import pounce


def test_minimize_rosenbrock():
    def rosen(x):
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    def grad(x):
        n = x.size
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1 - x[:-1])
        g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
        return g

    def hess(x):
        n = x.size
        H = np.zeros((n, n))
        # Standard analytic Hessian of the chained 2-term Rosenbrock.
        H[np.arange(n - 1), np.arange(n - 1)] += (
            1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
        )
        H[np.arange(1, n), np.arange(1, n)] += 200.0
        off = -400.0 * x[:-1]
        H[np.arange(n - 1), np.arange(1, n)] += off
        H[np.arange(1, n), np.arange(n - 1)] += off
        return H

    res = pounce.minimize(
        rosen, x0=np.zeros(4), jac=grad, hess=hess,
        options={"tol": 1e-8, "print_level": 0},
    )
    assert res.success
    np.testing.assert_allclose(res.x, np.ones(4), atol=1e-4)


def test_minimize_eq_constraint():
    """min  x[0]^2 + x[1]^2   s.t.   x[0] + x[1] = 1   →   x* = (.5, .5), f* = .5."""
    def f(x):
        return float(x @ x)

    def grad(x):
        return 2.0 * x

    def c_fun(x):
        return np.array([x[0] + x[1] - 1.0])

    def c_jac(x):
        return np.array([[1.0, 1.0]])

    res = pounce.minimize(
        f, x0=np.zeros(2), jac=grad,
        constraints=[{"type": "eq", "fun": c_fun, "jac": c_jac}],
        options={"tol": 1e-10, "print_level": 0},
    )
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(res.fun, 0.5, atol=1e-8)


def test_minimize_rejects_wrong_length_bounds():
    """A too-short ``bounds`` list used to silently leave trailing variables
    unbounded (and, in the sampling searches, broadcast one box across several);
    it now raises a clear ValueError up front, like scipy."""
    def f(x):
        return float(x @ x)

    def grad(x):
        return 2.0 * x

    # two variables, one bound pair -> rejected
    with pytest.raises(ValueError, match="bounds has 1 entry but the problem has 2"):
        pounce.minimize(f, x0=np.zeros(2), jac=grad, bounds=[(-1, 1)],
                        options={"print_level": 0})

    # too-long list is rejected too (plural wording)
    with pytest.raises(ValueError, match="bounds has 3 entries but the problem has 2"):
        pounce.minimize(f, x0=np.zeros(2), jac=grad,
                        bounds=[(-1, 1), (-1, 1), (-1, 1)],
                        options={"print_level": 0})

    # correct length still works
    res = pounce.minimize(f, x0=np.array([0.5, 0.5]), jac=grad,
                          bounds=[(-1, 1), (-1, 1)],
                          options={"tol": 1e-10, "print_level": 0})
    assert res.success
    np.testing.assert_allclose(res.x, [0.0, 0.0], atol=1e-6)


def test_minimize_promotes_scalar_x0():
    """A scalar / 0-d ``x0`` is promoted to 1-D (like scipy), so a
    single-variable problem can be written ``minimize(f, 1.5)`` instead of
    tripping ``TypeError: iteration over a 0-d array``."""
    f = lambda x: float(x[0]) ** 2
    g = lambda x: np.array([2.0 * x[0]])
    for x0 in (1.5, np.array(1.5)):
        res = pounce.minimize(f, x0=x0, jac=g, options={"tol": 1e-10, "print_level": 0})
        assert res.success
        assert res.x.shape == (1,)
        np.testing.assert_allclose(res.x, [0.0], atol=1e-6)


def test_minimize_rejects_reversed_bounds():
    """A reversed ``(low, high)`` pair (low > high) used to silently produce an
    infeasible box; it now raises a clear ValueError. A fixed bound (low ==
    high) is still allowed."""
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    with pytest.raises(ValueError, match=r"bounds\[0\] is reversed"):
        pounce.minimize(f, x0=np.zeros(2), jac=g, bounds=[(1.0, -1.0), (-2, 2)],
                        options={"print_level": 0})
    # low == high (a fixed variable) is permitted
    res = pounce.minimize(f, x0=np.array([0.3, 0.3]), jac=g,
                          bounds=[(0.5, 0.5), (-2, 2)],
                          options={"tol": 1e-10, "print_level": 0})
    np.testing.assert_allclose(res.x[0], 0.5, atol=1e-6)


def test_nlp_success_status_includes_acceptable_level():
    """gh #119 regression (mapping). ``success`` for the NLP path must count
    ``Solved_To_Acceptable_Level`` (status 1) as a success, not only
    ``Solve_Succeeded`` (status 0) — matching Ipopt/cyipopt, scipy, and pounce's
    own differentiable ``_OK_STATUS``. Infeasible/tiny-step/etc. stay failures."""
    from pounce._minimize import _NLP_SUCCESS_STATUS

    assert 0 in _NLP_SUCCESS_STATUS          # Solve_Succeeded
    assert 1 in _NLP_SUCCESS_STATUS          # Solved_To_Acceptable_Level
    assert 2 not in _NLP_SUCCESS_STATUS      # Infeasible_Problem_Detected
    assert 3 not in _NLP_SUCCESS_STATUS      # Search_Direction_Becomes_Too_Small


def test_minimize_acceptable_level_reports_success():
    """gh #119 regression (end-to-end). HS071 from the (2,2,2,2) start converges
    to the acceptable level (status 1) rather than the tight tolerance; pounce
    used to flag that ``success=False`` at the verified optimum. The acceptable
    solve must now report ``success=True``."""
    def f(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    constraints = [
        {"type": "ineq", "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0},
        {"type": "eq", "fun": lambda x: float(x @ x) - 40.0},
    ]
    res = pounce.minimize(
        f, x0=np.array([2.0, 2.0, 2.0, 2.0]), bounds=[(1.0, 5.0)] * 4,
        constraints=constraints, options={"solver_selection": "nlp", "print_level": 0},
    )
    # Reaches the known HS071 optimum f* = 17.0140173 ...
    np.testing.assert_allclose(res.fun, 17.0140172891520078, atol=1e-5)
    # ... and terminates acceptable-or-better, which must read as success.
    assert res.status in (0, 1)
    assert res.success


def test_minimize_rejects_malformed_constraint_dicts():
    """Malformed constraint dicts used to raise a bare ``KeyError``; they now
    raise a clear ValueError naming the problem."""
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    opts = {"print_level": 0}
    with pytest.raises(ValueError, match="missing required key"):
        pounce.minimize(f, np.ones(2), jac=g,
                        constraints={"fun": lambda x: x[0] - x[1]}, options=opts)
    with pytest.raises(ValueError, match="missing required key"):
        pounce.minimize(f, np.ones(2), jac=g,
                        constraints={"type": "eq"}, options=opts)
    with pytest.raises(ValueError, match="must be a dict"):
        pounce.minimize(f, np.ones(2), jac=g, constraints=["bad"], options=opts)
    with pytest.raises(ValueError, match="must be callable"):
        pounce.minimize(f, np.ones(2), jac=g,
                        constraints={"type": "eq", "fun": 3.0}, options=opts)
