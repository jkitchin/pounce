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


# -- scipy.optimize.Bounds input ----------------------------------------------


def test_minimize_accepts_scipy_bounds_object():
    """``bounds`` may be a ``scipy.optimize.Bounds`` instance, not just a list."""
    f, grad, target, _ = _mixture_quadratic()
    # Box: x_i in [0.2, 0.6]. Target [0.7, 0.1, 0.4] gets clipped to [0.6, 0.2, 0.4].
    bnds = opt.Bounds(lb=np.array([0.2, 0.2, 0.2]), ub=np.array([0.6, 0.6, 0.6]))
    res = pounce.minimize(
        f, x0=np.full(3, 0.4), jac=grad, bounds=bnds, tol=1e-10, print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, [0.6, 0.2, 0.4], atol=1e-6)


def test_minimize_accepts_scipy_bounds_with_scalar_broadcast():
    """``Bounds(lb=0.0, ub=1.0)`` (scalar) should broadcast to n."""
    f, grad, target, _ = _mixture_quadratic()
    bnds = opt.Bounds(lb=0.0, ub=1.0)
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, bounds=bnds, tol=1e-10, print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


def test_minimize_accepts_scipy_bounds_keep_feasible_ignored():
    """``keep_feasible=True`` must not raise — silently honored by the barrier."""
    f, grad, target, _ = _mixture_quadratic()
    bnds = opt.Bounds(lb=0.0, ub=1.0, keep_feasible=True)
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, bounds=bnds, tol=1e-10, print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


# -- scipy → Ipopt option-name synonyms ---------------------------------------


def test_minimize_maxiter_synonym():
    """``maxiter`` (scipy) translates to Ipopt ``max_iter`` so user options work."""
    f, grad, _, _ = _mixture_quadratic()
    # maxiter=1 should leave the solver short of convergence on Rosenbrock-ish problems,
    # but on this convex quadratic 1 step still converges. So instead just check the
    # option is *accepted*.
    res = pounce.minimize(
        f, x0=np.full(3, 1.0 / 3), jac=grad, maxiter=200, print_level=0,
    )
    assert res.success


def test_minimize_gtol_ftol_xtol_synonyms():
    """``gtol`` / ``ftol`` / ``xtol`` all map to Ipopt's single ``tol``."""
    f, grad, target, _ = _mixture_quadratic()
    for key in ("gtol", "ftol", "xtol"):
        res = pounce.minimize(
            f, x0=np.full(3, 0.5), jac=grad, print_level=0, **{key: 1e-10},
        )
        assert res.success, f"{key} synonym failed"
        np.testing.assert_allclose(res.x, target, atol=1e-6)


def test_minimize_disp_synonym():
    """``disp=False`` (scipy bool) translates to ``print_level=0`` (Ipopt int)."""
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, disp=False,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


def test_minimize_iprint_synonym():
    """``iprint`` translates to Ipopt's ``print_level``."""
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, iprint=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


def test_minimize_maxcor_synonym():
    """``maxcor`` translates to Ipopt's ``limited_memory_max_history``."""
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, maxcor=8, print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


# -- nfev / njev counters on the result ----------------------------------------


def test_minimize_populates_nfev_and_njev():
    """The result should expose scipy-standard ``nfev`` / ``njev`` counters."""
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f, x0=np.full(3, 0.5), jac=grad, tol=1e-10, print_level=0,
    )
    assert res.success
    # At least one objective and one gradient evaluation must have happened.
    assert res.nfev > 0
    assert res.njev > 0
    # And they should be on the order of magnitude of iteration count, not 0.
    assert res.nfev >= res.nit


def test_minimize_jac_true_counts_both_eval_modes():
    """With ``jac=True`` a single fun(x) call returns (f, g); both counters tick."""
    call_count = {"n": 0}

    def fg(x):
        call_count["n"] += 1
        return 0.5 * float((x ** 2).sum()), x

    res = pounce.minimize(
        fg, x0=np.array([1.0, 2.0, 3.0]), jac=True, tol=1e-10, print_level=0,
    )
    assert res.success
    # Counters should both be non-zero; the single-pass cache reuse means
    # ``call_count["n"]`` is roughly equal to ``nfev`` (Ipopt calls objective,
    # which calls fg, which the cache then serves to gradient).
    assert res.nfev > 0
    assert res.njev > 0
