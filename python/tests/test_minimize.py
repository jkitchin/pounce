"""Smoke tests for the scipy.optimize-style facade."""

import numpy as np
import scipy.optimize as opt
from scipy import sparse
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
        rosen,
        x0=np.zeros(4),
        jac=grad,
        hess=hess,
        tol=1e-8,
        print_level=0,
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
        f,
        x0=np.zeros(2),
        jac=grad,
        constraints=[{"type": "eq", "fun": c_fun, "jac": c_jac}],
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(res.fun, 0.5, atol=1e-8)


# -- The mixture-quadratic fixture is shared by the LinearConstraint tests --


def _mixture_quadratic():
    """min 0.5 * sum((x - target)^2) s.t. x[0]+x[1]+x[2] = 1.

    Analytic optimum: project ``target`` onto the simplex hyperplane, which is
    ``target - mean(target - 1/3)`` per coord, i.e. shift so the sum is 1.
    """
    target = np.array([0.7, 0.1, 0.4])

    def f(x):
        return 0.5 * float(((x - target) ** 2).sum())

    def grad(x):
        return x - target

    shift = (target.sum() - 1.0) / 3.0
    expected = target - shift
    return f, grad, target, expected


def test_minimize_scipy_routing_via_callable_method():
    """`scipy.optimize.minimize(method=pounce.minimize, …)` works end-to-end.

    Exercises the `_custom` dispatch path: scipy calls pounce with all
    standard kwargs (including ``hessp=None``) plus ``**options`` splat.
    """
    f, grad, target, _ = _mixture_quadratic()
    res = opt.minimize(
        f,
        x0=np.full(3, 1.0 / 3),
        jac=grad,
        method=pounce.minimize,
        options={"tol": 1e-8, "print_level": 0, "max_iter": 200},
    )
    assert res.success
    # Unconstrained → reaches target.
    np.testing.assert_allclose(res.x, target, atol=1e-5)


def test_minimize_linear_constraint_sparse_A():
    """LinearConstraint with a scipy.sparse COO matrix carries through to Ipopt."""
    f, grad, _, expected = _mixture_quadratic()
    A = sparse.coo_array(np.array([[1.0, 1.0, 1.0]]))
    lc = opt.LinearConstraint(A, lb=1.0, ub=1.0)
    res = pounce.minimize(
        f,
        x0=np.full(3, 1.0 / 3),
        jac=grad,
        constraints=lc,
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, expected, atol=1e-6)
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)


def test_minimize_linear_constraint_dense_A():
    """LinearConstraint with a dense numpy A gets COO-ified internally."""
    f, grad, _, expected = _mixture_quadratic()
    A = np.array([[1.0, 1.0, 1.0]])
    lc = opt.LinearConstraint(A, lb=1.0, ub=1.0)
    res = pounce.minimize(
        f,
        x0=np.full(3, 1.0 / 3),
        jac=grad,
        constraints=[lc],
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, expected, atol=1e-6)
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)


def test_minimize_mixed_linear_and_dict_constraints():
    """A list mixing LinearConstraint + legacy dict resolves to one feasible solution."""
    f, grad, _, _ = _mixture_quadratic()
    # Linear equality: x[0]+x[1]+x[2] = 1.
    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), lb=1.0, ub=1.0)

    # Dict inequality: x[0] >= 0.2.
    def c_fun(x):
        return np.array([x[0] - 0.2])

    def c_jac(x):
        return np.array([[1.0, 0.0, 0.0]])

    res = pounce.minimize(
        f,
        x0=np.array([0.3, 0.3, 0.4]),
        jac=grad,
        constraints=[lc, {"type": "ineq", "fun": c_fun, "jac": c_jac}],
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)
    assert res.x[0] >= 0.2 - 1e-8


def test_minimize_absorbs_scipy_default_kwargs():
    """Calling with the kwargs scipy's ``_custom`` dispatch always sends
    (including ``hessp=None``) must not TypeError. ``None`` values are filtered
    out before reaching Ipopt; real Ipopt options pass through unchanged.
    """
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f,
        x0=np.full(3, 1.0 / 3),
        args=(),  # scipy default
        jac=grad,
        hess=None,  # scipy default
        hessp=None,  # not declared on the signature; absorbed by **options
        bounds=None,
        constraints=None,
        callback=None,
        tol=1e-8,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-5)


def test_minimize_unknown_option_raises_at_solve():
    """Real Ipopt-option typos surface as RuntimeError at solve() time.

    This is intentional: it's the only way a user finds out they mistyped
    a real Ipopt option name (silently dropping would hide the mistake).
    """
    f, grad, _, _ = _mixture_quadratic()
    with pytest.raises(RuntimeError, match="Unknown option"):
        pounce.minimize(
            f,
            x0=np.full(3, 1.0 / 3),
            jac=grad,
            tol_typo=1e-8,
            print_level=0,
        )
