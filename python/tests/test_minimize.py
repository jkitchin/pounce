"""Smoke tests for the scipy.optimize-style facade."""

import warnings

import numpy as np
import pytest
import scipy.optimize as opt
from scipy import sparse

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


def test_minimize_reports_timing_breakdown():
    """pounce#180 item 3: the result exposes a per-subsystem wall-clock
    breakdown (``res.timing`` / ``res.info["timing"]``) plus a top-level
    ``res.wall_time``, so a caller can attribute solve runtime (func /
    gradient / Jacobian / Hessian eval time, factorization vs back-solve)
    without patching the solver.

    The *detailed* breakdown is opt-in via ``timing_statistics="yes"``
    (issue #190): running the per-subsystem timers unconditionally costs
    two ``getrusage`` syscalls per timed section, so by default only
    ``overall_alg`` / ``wall_time`` are populated and the rest read
    ``0.0``. The breakdown schema (keys present, non-negative, the
    linear-algebra total equal to the sum of its parts) holds either way.
    """

    def f(x):
        return float(x @ x)

    def grad(x):
        return 2.0 * x

    res = pounce.minimize(
        f,
        x0=np.array([1.0, 2.0, 3.0]),
        jac=grad,
        print_level=0,
        timing_statistics="yes",
    )
    assert res.success

    # Top-level convenience fields.
    assert np.isfinite(res.wall_time)
    assert res.wall_time >= 0.0
    assert isinstance(res.timing, dict)
    # Same breakdown is mirrored into the info dict.
    assert res.info["timing"] == res.timing
    assert res.info["wall_time"] == res.wall_time

    # Advertised keys are present, non-negative, and the linear-algebra
    # total is the exact sum of its factorization / back-solve parts.
    expected_keys = {
        "overall_alg",
        "linear_system_total",
        "linear_system_factorization",
        "linear_system_back_solve",
        "function_evaluations_total",
        "eval_objective",
        "eval_gradient",
        "eval_constraints",
        "eval_constraint_jacobian",
        "eval_lagrangian_hessian",
    }
    assert expected_keys <= set(res.timing)
    for key in expected_keys:
        assert res.timing[key] >= 0.0, key
    assert res.timing["linear_system_total"] == pytest.approx(
        res.timing["linear_system_factorization"]
        + res.timing["linear_system_back_solve"]
    )


def test_minimize_timing_breakdown_gated_by_default():
    """Issue #190: with ``timing_statistics`` at its default (``no``), the
    detailed per-subsystem timers are disabled — the solve stops paying two
    ``getrusage`` syscalls per timed section on fast objectives. The
    breakdown schema is still present (keys exist), ``overall_alg`` /
    ``wall_time`` are still populated, but every detailed key reads ``0.0``.
    """

    def f(x):
        return float(x @ x)

    def grad(x):
        return 2.0 * x

    res = pounce.minimize(f, x0=np.array([1.0, 2.0, 3.0]), jac=grad, print_level=0)
    assert res.success

    # Overall totals survive the gating.
    assert res.wall_time >= 0.0
    assert res.timing["overall_alg"] >= 0.0

    # Detailed per-callback / linear-algebra timers are gated off ⇒ 0.0.
    for key in (
        "eval_objective",
        "eval_gradient",
        "function_evaluations_total",
        "linear_system_factorization",
        "linear_system_back_solve",
    ):
        assert res.timing[key] == 0.0, (
            f"{key} accumulated despite timing_statistics=no (#190)"
        )


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


# -- Sparse Jacobians from dict constraints (linear & nonlinear) -------------


def test_wrap_constraints_dict_sparse_jac_declares_sparse_structure():
    """A dict constraint whose ``jac`` returns a scipy-sparse matrix declares the
    matrix's COO structure (nnz), not the dense m*n grid."""
    from pounce._minimize import _wrap_constraints

    n = 6
    # Two rows, block-diagonal: row0 touches cols 0,1; row1 touches cols 2,3.
    A = sparse.coo_array(
        np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    )
    con = {"type": "ineq", "fun": lambda x: A @ x, "jac": lambda x: A}
    m, g, jac_values, cl, cu, jr, jc = _wrap_constraints([con], n, x0=np.zeros(n))

    assert m == 2
    # Sparse: 4 nonzeros, NOT the dense 2*6 = 12.
    assert jr.size == 4 and jc.size == 4
    # Structure matches A's (canonicalized, row-major) COO triplet.
    np.testing.assert_array_equal(jr, [0, 0, 1, 1])
    np.testing.assert_array_equal(jc, [0, 1, 2, 3])
    np.testing.assert_allclose(jac_values(np.arange(n, dtype=float)), [1, 1, 1, 1])


def test_minimize_dict_sparse_jac_linear_matches_linear_constraint():
    """A linear equality given as a dict with a constant coo jac reaches the same
    optimum as the equivalent LinearConstraint."""
    f, grad, _, expected = _mixture_quadratic()
    A = sparse.coo_array(np.array([[1.0, 1.0, 1.0]]))
    kw = dict(x0=np.full(3, 1.0 / 3), jac=grad, tol=1e-10, print_level=0)

    res_lc = pounce.minimize(f, constraints=opt.LinearConstraint(A, 1.0, 1.0), **kw)
    res_sd = pounce.minimize(
        f,
        constraints={
            "type": "eq",
            "fun": lambda x: np.array([x.sum() - 1.0]),
            "jac": lambda x: A,
        },
        **kw,
    )

    assert res_lc.success and res_sd.success
    np.testing.assert_allclose(res_sd.x, expected, atol=1e-6)
    np.testing.assert_allclose(res_sd.x, res_lc.x, atol=1e-6)


def test_minimize_dict_sparse_jac_nonlinear_matches_dense():
    """A nonlinear inequality whose jac returns a (varying) coo matrix solves to
    the same point as the dense-jac form."""
    f, grad, _, _ = _mixture_quadratic()

    # Active nonlinear ineq (pounce dict 'ineq' means fun(x) >= 0):
    #   x[0]^2 + x[2]^2 - 0.7 >= 0  → jac = [2 x0, 0, 2 x2] (nonzeros at cols 0, 2).
    def c_fun(x):
        return np.array([x[0] ** 2 + x[2] ** 2 - 0.7])

    def c_jac_dense(x):
        return np.array([[2 * x[0], 0.0, 2 * x[2]]])

    def c_jac_sparse(x):
        return sparse.coo_array(
            (np.array([2 * x[0], 2 * x[2]]), (np.array([0, 0]), np.array([0, 2]))),
            shape=(1, 3),
        )

    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), 1.0, 1.0)
    kw = dict(x0=np.array([0.3, 0.3, 0.4]), jac=grad, tol=1e-9, print_level=0)

    res_dense = pounce.minimize(
        f, constraints=[lc, {"type": "ineq", "fun": c_fun, "jac": c_jac_dense}], **kw
    )
    res_sparse = pounce.minimize(
        f, constraints=[lc, {"type": "ineq", "fun": c_fun, "jac": c_jac_sparse}], **kw
    )

    assert res_dense.success and res_sparse.success
    np.testing.assert_allclose(res_sparse.x.sum(), 1.0, atol=1e-7)
    assert res_sparse.x[0] ** 2 + res_sparse.x[2] ** 2 >= 0.7 - 1e-6
    # Identical jac *values* → identical iterates; only the representation differs.
    np.testing.assert_allclose(res_sparse.x, res_dense.x, atol=1e-5)


def test_minimize_dict_sparse_jac_pattern_change_raises():
    """Pounce requires a fixed sparsity pattern; a jac whose nnz changes between
    the x0 probe and the solve raises a clear error."""
    from pounce._minimize import _wrap_constraints

    n = 3
    calls = {"k": 0}

    def jac(x):
        calls["k"] += 1
        # 1 nonzero at probe (build), 2 thereafter → pattern change.
        if calls["k"] == 1:
            return sparse.coo_array(([1.0], ([0], [0])), shape=(1, n))
        return sparse.coo_array(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, n))

    con = {"type": "ineq", "fun": lambda x: np.array([x[0]]), "jac": jac}
    _, _, jac_values, _, _, jr, _ = _wrap_constraints([con], n, x0=np.zeros(n))
    assert jr.size == 1  # declared from the probe
    with pytest.raises(ValueError, match="sparsity pattern changed"):
        jac_values(np.ones(n))


def test_minimize_dict_sparse_jac_position_change_raises():
    """A jac that keeps the *same nnz count* but moves a nonzero to a different
    column must also raise — otherwise its values would be silently misaligned
    against the build-time structure and fed to Ipopt as wrong derivatives."""
    from pounce._minimize import _wrap_constraints

    n = 3
    calls = {"k": 0}

    def jac(x):
        calls["k"] += 1
        # Same count (2 nonzeros) at every call, but the second column moves
        # from col 1 (probe/build) to col 2 (solve) → position change.
        if calls["k"] == 1:
            return sparse.coo_array(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, n))
        return sparse.coo_array(([1.0, 1.0], ([0, 0], [0, 2])), shape=(1, n))

    con = {"type": "ineq", "fun": lambda x: np.array([x[0]]), "jac": jac}
    _, _, jac_values, _, _, jr, jc = _wrap_constraints([con], n, x0=np.zeros(n))
    np.testing.assert_array_equal(jc, [0, 1])  # declared from the probe
    with pytest.raises(ValueError, match="sparsity pattern changed"):
        jac_values(np.ones(n))


def test_minimize_dict_dense_jac_still_dense():
    """Regression: a dict jac returning a dense ndarray keeps the dense pattern."""
    from pounce._minimize import _wrap_constraints

    n = 4
    con = {
        "type": "ineq",
        "fun": lambda x: np.array([x[0] - 0.2]),
        "jac": lambda x: np.array([[1.0, 0.0, 0.0, 0.0]]),
    }
    m, _, _, _, _, jr, jc = _wrap_constraints([con], n, x0=np.zeros(n))
    assert m == 1 and jr.size == n  # dense 1*n pattern, unchanged


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
        f,
        x0=np.full(3, 0.4),
        jac=grad,
        bounds=bnds,
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, [0.6, 0.2, 0.4], atol=1e-6)


def test_minimize_accepts_scipy_bounds_with_scalar_broadcast():
    """``Bounds(lb=0.0, ub=1.0)`` (scalar) should broadcast to n."""
    f, grad, target, _ = _mixture_quadratic()
    bnds = opt.Bounds(lb=0.0, ub=1.0)
    res = pounce.minimize(
        f,
        x0=np.full(3, 0.5),
        jac=grad,
        bounds=bnds,
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


def test_minimize_accepts_scipy_bounds_keep_feasible_ignored():
    """``keep_feasible=True`` must not raise — silently honored by the barrier."""
    f, grad, target, _ = _mixture_quadratic()
    bnds = opt.Bounds(lb=0.0, ub=1.0, keep_feasible=True)
    res = pounce.minimize(
        f,
        x0=np.full(3, 0.5),
        jac=grad,
        bounds=bnds,
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    np.testing.assert_allclose(res.x, target, atol=1e-6)


# -- scipy → Ipopt option-name synonyms ---------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("maxiter", 200),  # → max_iter
        ("gtol", 1e-10),  # → tol  (gradient tolerance synonym)
        ("ftol", 1e-10),  # → tol  (function tolerance synonym)
        ("xtol", 1e-10),  # → tol  (x-step tolerance synonym)
        ("disp", False),  # → print_level=0
        ("iprint", 0),  # → print_level
        ("maxcor", 8),  # → limited_memory_max_history
    ],
)
def test_minimize_scipy_option_synonyms(key, value):
    """Each scipy-canonical option translates to its Ipopt equivalent and is
    accepted without error, reaching the same optimum as the bare call."""
    f, grad, target, _ = _mixture_quadratic()
    kwargs = {"jac": grad, "print_level": 0, key: value}
    res = pounce.minimize(f, x0=np.full(3, 1.0 / 3), **kwargs)
    assert res.success, f"{key}={value!r} was not accepted as a scipy synonym"
    np.testing.assert_allclose(res.x, target, atol=1e-5)


# -- nfev / njev counters on the result ----------------------------------------


def test_minimize_populates_nfev_and_njev():
    """The result should expose scipy-standard ``nfev`` / ``njev`` counters."""
    f, grad, target, _ = _mixture_quadratic()
    res = pounce.minimize(
        f,
        x0=np.full(3, 0.5),
        jac=grad,
        tol=1e-10,
        print_level=0,
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
        return 0.5 * float((x**2).sum()), x

    res = pounce.minimize(
        fg,
        x0=np.array([1.0, 2.0, 3.0]),
        jac=True,
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    # Counters should both be non-zero; the single-pass cache reuse means
    # ``call_count["n"]`` is roughly equal to ``nfev`` (Ipopt calls objective,
    # which calls fg, which the cache then serves to gradient).
    assert res.nfev > 0
    assert res.njev > 0


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
        pounce.minimize(
            f, x0=np.zeros(2), jac=grad, bounds=[(-1, 1)], options={"print_level": 0}
        )

    # too-long list is rejected too (plural wording)
    with pytest.raises(ValueError, match="bounds has 3 entries but the problem has 2"):
        pounce.minimize(
            f,
            x0=np.zeros(2),
            jac=grad,
            bounds=[(-1, 1), (-1, 1), (-1, 1)],
            options={"print_level": 0},
        )

    # correct length still works
    res = pounce.minimize(
        f,
        x0=np.array([0.5, 0.5]),
        jac=grad,
        bounds=[(-1, 1), (-1, 1)],
        options={"tol": 1e-10, "print_level": 0},
    )
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
        pounce.minimize(
            f,
            x0=np.zeros(2),
            jac=g,
            bounds=[(1.0, -1.0), (-2, 2)],
            options={"print_level": 0},
        )
    # low == high (a fixed variable) is permitted
    res = pounce.minimize(
        f,
        x0=np.array([0.3, 0.3]),
        jac=g,
        bounds=[(0.5, 0.5), (-2, 2)],
        options={"tol": 1e-10, "print_level": 0},
    )
    np.testing.assert_allclose(res.x[0], 0.5, atol=1e-6)


def test_normalize_bounds_rejects_nan():
    """#265: a NaN bound used to sail through the reversed-bound check (``lb >
    ub`` is False against NaN) and behave as a silent 'no bound'. It now raises,
    on both the pair-list and ``scipy.optimize.Bounds`` paths. ±inf/None — the
    real unbounded spellings — must still pass."""
    from pounce._minimize import _normalize_bounds

    # pair-list path
    with pytest.raises(ValueError, match="NaN"):
        _normalize_bounds([(float("nan"), 10.0)], 1)
    # scipy Bounds path
    with pytest.raises(ValueError, match="NaN"):
        _normalize_bounds(opt.Bounds(np.nan, 10.0), 1)

    # ±inf and None stay legal (they are the one-sided encoding this function
    # itself produces); use np.isnan, never ~np.isfinite.
    lb, ub = _normalize_bounds([(-np.inf, np.inf)], 1)
    np.testing.assert_array_equal(lb, [-np.inf])
    np.testing.assert_array_equal(ub, [np.inf])
    lb, ub = _normalize_bounds([(None, None)], 1)
    np.testing.assert_array_equal(lb, [-np.inf])
    np.testing.assert_array_equal(ub, [np.inf])


def test_minimize_rejects_nan_bounds_end_to_end():
    """A NaN bound reaches the public ``minimize`` API and now raises instead of
    silently returning a 'successful' solve (#265)."""
    with pytest.raises(ValueError, match="NaN"):
        pounce.minimize(
            lambda v: (v[0] - 3.0) ** 2,
            x0=[0.0],
            bounds=[(float("nan"), 10.0)],
            options={"print_level": 0},
        )


def test_minimize_nan_gradient_reports_honest_failure():
    """#292: a ``jac`` that returns NaN used to be laundered into a
    ``Solve_Succeeded`` at ``x0`` with a *finite* ``fun`` — the most dangerous
    silent-success shape, since the caller gets no signal. The max-norm behind
    the dual-infeasibility measure silently drops NaN (``NaN > m`` is False), so
    the KKT error read 0.0 and the solve declared itself optimal. It must now
    surface an honest non-success status."""
    res = pounce.minimize(
        lambda x: float(x[0] ** 2),
        np.array([0.5]),
        jac=lambda x: np.array([np.nan]),
        options={"print_level": 0},
    )
    assert not res.success
    assert res.message != "Solve_Succeeded"
    assert res.status != 0


def test_minimize_inf_gradient_reports_honest_failure():
    """#292 sibling: an Inf gradient (Inf is *not* laundered by the max-norm,
    but verify it still fails honestly and did not regress)."""
    for bad in (np.inf, -np.inf):
        res = pounce.minimize(
            lambda x: float(x[0] ** 2),
            np.array([0.5]),
            jac=lambda x, bad=bad: np.array([bad]),
            options={"print_level": 0},
        )
        assert not res.success, f"Inf gradient ({bad}) reported success"
        assert res.status != 0


def test_minimize_nan_constraint_jacobian_reports_honest_failure():
    """#292: a NaN in the *constraint* Jacobian enters the Lagrangian gradient
    through the Jᵀy term and was laundered by the same max-norm, again yielding
    a bogus ``Solve_Succeeded``. It must now fail honestly."""

    def f(x):
        return float(x @ x)

    res = pounce.minimize(
        f,
        x0=np.array([0.5, 0.5]),
        jac=lambda x: 2.0 * x,
        constraints=[
            {
                "type": "eq",
                "fun": lambda x: np.array([x[0] + x[1] - 1.0]),
                "jac": lambda x: np.array([[np.nan, 1.0]]),
            }
        ],
        options={"print_level": 0},
    )
    assert not res.success
    assert res.status != 0


def test_minimize_nan_objective_stays_honest():
    """#292 contrast (guard against over-correction): a ``fun`` that returns NaN
    already failed honestly (the objective value reaches a finiteness check the
    gradient norm bypassed). The fix must not change that — it must still report
    a non-success status, not ``Solve_Succeeded``."""
    res = pounce.minimize(
        lambda x: float(np.nan),
        np.array([0.5]),
        jac=lambda x: np.array([2.0 * x[0]]),
        options={"print_level": 0},
    )
    assert not res.success
    assert res.status != 0


def test_minimize_finite_solve_still_succeeds_to_optimum():
    """#292 no-regression: with everything finite, a normal unconstrained solve
    still converges to the known optimum with ``Solve_Succeeded``."""
    res = pounce.minimize(
        lambda x: float((x[0] - 3.0) ** 2),
        np.array([0.5]),
        jac=lambda x: np.array([2.0 * (x[0] - 3.0)]),
        tol=1e-10,
        options={"print_level": 0},
    )
    assert res.success
    assert res.message == "Solve_Succeeded"
    np.testing.assert_allclose(res.x, [3.0], atol=1e-6)


def test_nlp_success_status_includes_acceptable_level():
    """gh #119 regression (mapping). ``success`` for the NLP path must count
    ``Solved_To_Acceptable_Level`` (status 1) as a success, not only
    ``Solve_Succeeded`` (status 0) — matching Ipopt/cyipopt, scipy, and pounce's
    own differentiable ``_OK_STATUS``. Infeasible/tiny-step/etc. stay failures."""
    from pounce._minimize import _NLP_SUCCESS_STATUS

    assert 0 in _NLP_SUCCESS_STATUS  # Solve_Succeeded
    assert 1 in _NLP_SUCCESS_STATUS  # Solved_To_Acceptable_Level
    assert 2 not in _NLP_SUCCESS_STATUS  # Infeasible_Problem_Detected
    assert 3 not in _NLP_SUCCESS_STATUS  # Search_Direction_Becomes_Too_Small


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # this test deliberately uses the FD fallback
        res = pounce.minimize(
            f,
            x0=np.array([2.0, 2.0, 2.0, 2.0]),
            bounds=[(1.0, 5.0)] * 4,
            constraints=constraints,
            options={"solver_selection": "nlp", "print_level": 0},
        )
    # Reaches the known HS071 optimum f* = 17.0140173 ...
    np.testing.assert_allclose(res.fun, 17.0140172891520078, atol=1e-5)
    # ... and terminates acceptable-or-better, which must read as success.
    assert res.status in (0, 1)
    assert res.success


def test_finite_diff_grad_is_central_and_accurate():
    """gh #123 (C). The FD fallback is now a *central* difference: its error on a
    smooth function is ``O(h^2)``, orders of magnitude tighter than the old
    one-sided ``O(h)``. Check it against an analytic gradient at a noise floor a
    forward difference (step ~1.49e-8, error ~1e-8) could never reach."""
    from pounce._minimize import _finite_diff_grad

    # f(x) = sum(x^3)  =>  grad = 3 x^2, an analytic reference.
    f = lambda x: float(np.sum(x**3))
    x = np.array([0.7, -1.3, 2.1])
    g_fd = _finite_diff_grad(f, x)
    g_exact = 3.0 * x**2
    # Central differences clear ~1e-9 here; a forward difference would sit ~1e-7.
    np.testing.assert_allclose(g_fd, g_exact, rtol=1e-7, atol=1e-8)


def test_minimize_fd_path_converges_from_documented_start():
    """gh #119 facet 2 / gh #123 (C + E). HS071 from the *documented* start
    (1, 5, 5, 1) with NO analytic jac used to crawl to a tiny-step exit
    (status 3) and report ``success=False`` at the verified optimum, because the
    forward-difference noise floor sat right at the tight ``tol=1e-8``. With
    central differences the dual infeasibility clears the tolerance and the
    finite-difference solve now reports success at the known optimum."""

    def f(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    constraints = [
        {"type": "ineq", "fun": lambda x: x[0] * x[1] * x[2] * x[3] - 25.0},
        {"type": "eq", "fun": lambda x: float(x @ x) - 40.0},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # FD-fallback warning is asserted elsewhere
        res = pounce.minimize(
            f,
            x0=np.array([1.0, 5.0, 5.0, 1.0]),
            bounds=[(1.0, 5.0)] * 4,
            constraints=constraints,
            options={"solver_selection": "nlp", "print_level": 0},
        )
    np.testing.assert_allclose(res.fun, 17.0140172891520078, atol=1e-5)
    assert res.success
    # The fix also exposes the final NLP error to the info dict (gh #123, E).
    assert np.isfinite(res.info["final_kkt_error"])
    assert res.info["final_kkt_error"] <= 1e-6


def test_minimize_warns_on_finite_difference_fallback():
    """gh #123 (D). Omitting analytic derivatives makes the NLP path
    finite-difference them, which is slower / less accurate. That must surface as
    a ``UserWarning`` naming the remedies — not happen silently."""
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    opts = {"solver_selection": "nlp", "tol": 1e-10, "print_level": 0}

    # No jac -> warns, naming the objective gradient and the autodiff remedy.
    with pytest.warns(UserWarning, match="objective gradient"):
        pounce.minimize(f, x0=np.array([1.0, 1.0]), options=opts)

    # Constraint without 'jac' -> warns about the constraint Jacobian even when
    # the objective jac is supplied.
    with pytest.warns(UserWarning, match="constraint Jacobian"):
        pounce.minimize(
            f,
            x0=np.array([1.0, 1.0]),
            jac=g,
            constraints=[{"type": "eq", "fun": lambda x: x[0] + x[1] - 1.0}],
            options=opts,
        )

    # Fully analytic -> no warning at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pounce.minimize(
            f,
            x0=np.array([1.0, 1.0]),
            jac=g,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: x[0] + x[1] - 1.0,
                    "jac": lambda x: np.array([[1.0, 1.0]]),
                }
            ],
            options=opts,
        )


def test_minimize_rejects_malformed_constraint_dicts():
    """Malformed constraint dicts used to raise a bare ``KeyError``; they now
    raise a clear ValueError naming the problem."""
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    opts = {"print_level": 0}
    with pytest.raises(ValueError, match="missing required key"):
        pounce.minimize(
            f,
            np.ones(2),
            jac=g,
            constraints={"fun": lambda x: x[0] - x[1]},
            options=opts,
        )
    with pytest.raises(ValueError, match="missing required key"):
        pounce.minimize(f, np.ones(2), jac=g, constraints={"type": "eq"}, options=opts)
    with pytest.raises(ValueError, match="must be a dict"):
        pounce.minimize(f, np.ones(2), jac=g, constraints=["bad"], options=opts)
    with pytest.raises(ValueError, match="must be callable"):
        pounce.minimize(
            f, np.ones(2), jac=g, constraints={"type": "eq", "fun": 3.0}, options=opts
        )


# -- Opt-in routing (default is NLP after the auto-route default flip) --------


def test_minimize_default_does_not_route():
    """Default is ``solver_selection="nlp"`` — even a convex QP that the router
    would happily detect must stay on the NLP path unless the caller opts in."""
    fun = lambda x: x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1]
    jac = lambda x: np.array([2 * x[0] - 3, 2 * x[1] - 4])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    res = pounce.minimize(
        fun,
        [0.5, 0.5],
        jac=jac,
        hess=hess,
        bounds=[(0, 1), (0, 1)],
    )
    # No routing — `info["solver"]` is only set when the router fires.
    assert res.info.get("solver") is None
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)


def test_minimize_solver_selection_auto_routes_convex_qp():
    """Explicit ``solver_selection="auto"`` restores the probe-then-route path
    and dispatches a textbook convex QP to the QP-IPM."""
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    jac = lambda x: 2 * x
    hess = lambda x: 2 * np.eye(2)
    res = pounce.minimize(
        fun,
        [1.0, 1.0],
        jac=jac,
        hess=hess,
        bounds=[(-1, 1), (-1, 1)],
        solver_selection="auto",
    )
    assert res.info.get("solver") == "qp-ipm"


def test_routed_result_exposes_eval_counters():
    """A routed (convex) result must still carry the scipy-standard
    ``nfev``/``njev``/``nhev`` attributes — accessing them must not
    ``AttributeError`` just because a different backend ran. The convex solver
    consumes the extracted quadratic form, so the counts are 0."""
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    jac = lambda x: 2 * x
    hess = lambda x: 2 * np.eye(2)
    res = pounce.minimize(
        fun,
        [1.0, 1.0],
        jac=jac,
        hess=hess,
        bounds=[(-1, 1), (-1, 1)],
        solver_selection="auto",
    )
    assert res.info.get("solver") == "qp-ipm"
    assert res.nfev == 0 and res.njev == 0 and res.nhev == 0


def test_result_subscript_falls_back_to_info():
    """Back-compat: ``res["<info-key>"]`` resolves the nested ``info`` mapping,
    preserving the pre-#97 subscript access (the old bespoke dataclass did this
    via ``__getitem__``). Top-level keys still win; a genuine miss raises."""
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    jac = lambda x: 2 * x
    hess = lambda x: 2 * np.eye(2)
    res = pounce.minimize(
        fun,
        [1.0, 1.0],
        jac=jac,
        hess=hess,
        bounds=[(-1, 1), (-1, 1)],
        solver_selection="auto",
    )
    # nested info key reachable via subscript and attribute
    assert res["solver"] == "qp-ipm"
    assert res.solver == "qp-ipm"
    # top-level key still resolves directly
    assert res["nfev"] == 0
    # a key in neither place still raises KeyError
    with pytest.raises(KeyError):
        res["definitely_not_a_key"]


def test_minimize_solver_selection_qp_ipm_with_linear_constraint():
    """``solver_selection="qp-ipm"`` accepts a ``LinearConstraint`` end-to-end
    and dispatches to the convex IPM (rather than probing or falling back)."""
    target = np.array([0.7, 0.1, 0.4])
    fun = lambda x: 0.5 * float(((x - target) ** 2).sum())
    jac = lambda x: x - target
    hess = lambda x: np.eye(3)
    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), lb=1.0, ub=1.0)
    res = pounce.minimize(
        fun,
        np.full(3, 1.0 / 3),
        jac=jac,
        hess=hess,
        constraints=lc,
        solver_selection="qp-ipm",
    )
    assert res.info.get("solver") == "qp-ipm"
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)


def test_minimize_callback_fires_with_linear_constraint():
    """Coverage gap: callback + ``LinearConstraint`` together. Each was tested
    in isolation; the combination exercises the callback shim under our COO
    constraint path."""
    f, grad, _, _ = _mixture_quadratic()
    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), lb=1.0, ub=1.0)
    xs = []
    res = pounce.minimize(
        f,
        x0=np.full(3, 1.0 / 3),
        jac=grad,
        constraints=lc,
        callback=lambda xk: xs.append(xk.copy()),
        tol=1e-10,
        print_level=0,
    )
    assert res.success
    assert len(xs) >= 1


def test_wrap_constraints_probes_at_x0_not_origin():
    """L47: constraint sizing must probe at the user's x0, not at the
    origin. A constraint undefined at 0 (e.g. ``log``) but defined at a
    feasible start used to fail before the solve began."""
    from pounce._minimize import _wrap_constraints

    calls = []

    def con(x):
        calls.append(np.array(x, dtype=float))
        # Undefined at the origin; finite at x0 = [1, 1].
        return np.log(x)

    x0 = np.array([1.0, 1.0])
    m, g, jac, cl, cu, _, _ = _wrap_constraints([{"type": "ineq", "fun": con}], 2, x0)

    # Probe happened at x0, so it saw a finite value (log(1) == 0).
    assert m == 2
    assert calls, "constraint should have been probed once for sizing"
    np.testing.assert_allclose(calls[0], x0)
    assert np.all(np.isfinite(g(x0)))

    # Probing at the origin (the old behavior) would have yielded -inf.
    with np.errstate(divide="ignore"):
        m0, g0, _, _, _, _, _ = _wrap_constraints(
            [{"type": "ineq", "fun": con}], 2, None
        )
        assert np.any(np.isneginf(g0(np.zeros(2))))


def test_wrap_constraints_fd_jac_uses_probed_sizes():
    """L47 (part 2): the FD Jacobian must not re-evaluate the constraint
    function purely to recount rows — the per-constraint size learned at
    probe time is reused, and the assembled Jacobian has the right shape.

    Our representation returns ``jac_values`` (a flat nnz vector) plus the COO
    ``(jac_rows, jac_cols)`` structure, so we reconstruct the dense ``(3, 2)``
    matrix to check the property.
    """
    from pounce._minimize import _wrap_constraints

    def con(x):
        # 3 outputs from 2 inputs -> Jacobian must be (3, 2).
        return np.array([x[0], x[1], x[0] + x[1]])

    x0 = np.array([0.5, 0.5])
    m, g, jac_values, cl, cu, jac_rows, jac_cols = _wrap_constraints(
        [{"type": "eq", "fun": con}], 2, x0
    )
    assert m == 3
    J = sparse.coo_array((jac_values(x0), (jac_rows, jac_cols)), shape=(m, 2)).toarray()
    assert J.shape == (3, 2)


class _FakeProblem:
    """Stand-in for the native ``Problem`` so the NLP path can run to its
    warnings without the compiled extension."""

    def __init__(self, **kwargs):
        self._x0 = None

    def add_option(self, key, value):
        pass

    def solve(self, x0):
        info = {
            "status": 0,
            "status_msg": "Solve_Succeeded",
            "obj_val": 0.0,
            "iter_count": 1,
            "final_kkt_error": 0.0,
        }
        return np.asarray(x0, dtype=float), info


def test_hess_ignored_with_constraints_warns(monkeypatch):
    """L48: a user-supplied ``hess`` cannot be honored once constraints are
    present (the wrapper can't form the constraint-curvature term of the
    Lagrangian Hessian), so the solver silently fell back to L-BFGS. It must
    now warn."""
    import pounce._minimize as M

    monkeypatch.setattr(M, "Problem", _FakeProblem)

    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    H = lambda x: 2.0 * np.eye(2)
    con = {
        "type": "eq",
        "fun": lambda x: x[0] - x[1],
        "jac": lambda x: np.array([[1.0, -1.0]]),
    }

    # solver_selection='nlp' forces the general NLP path (no convex routing).
    with pytest.warns(UserWarning, match="ignores the supplied 'hess'"):
        M.minimize(
            f,
            np.ones(2),
            jac=g,
            hess=H,
            constraints=con,
            options={"solver_selection": "nlp", "print_level": 0},
        )

    # Unconstrained: hess is honored, so no such warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M.minimize(
            f,
            np.ones(2),
            jac=g,
            hess=H,
            options={"solver_selection": "nlp", "print_level": 0},
        )


def _no_hess_warning(rec):
    """True if no 'ignores the supplied hess' warning is in the record."""
    return not any("ignores the supplied 'hess'" in str(w.message) for w in rec)


def test_hess_used_with_linear_constraint():
    """A user ``hess`` IS honored when all constraints are linear (the
    constraint-curvature term of the Lagrangian Hessian is zero, so the
    objective Hessian is the Lagrangian Hessian). No warning; the exact
    Hessian is actually used (``nhev > 0``)."""
    target = np.array([0.7, 0.1, 0.4])
    f = lambda x: 0.5 * float(((x - target) ** 2).sum())
    g = lambda x: x - target
    H = lambda x: np.eye(3)
    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), lb=1.0, ub=1.0)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = pounce.minimize(
            f,
            np.full(3, 1.0 / 3),
            jac=g,
            hess=H,
            constraints=lc,
            tol=1e-10,
            print_level=0,
        )
    assert _no_hess_warning(rec), "hess must not be dropped for linear constraints"
    assert res.success
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)
    assert res.nhev > 0, "the exact Hessian was not used (fell back to L-BFGS)"


def test_hess_used_with_mixed_eq_ineq_linear_constraint():
    """A single LinearConstraint carrying both an equality and an inequality
    row still counts as linear → the Hessian is used."""
    target = np.array([0.6, 0.1, 0.5])
    f = lambda x: 0.5 * float(((x - target) ** 2).sum())
    g = lambda x: x - target
    H = lambda x: np.eye(3)
    # row 0: sum == 1 (lb==ub); row 1: x0 - x1 >= 0 (lb=0, ub=+inf)
    A = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
    lc = opt.LinearConstraint(A, lb=[1.0, 0.0], ub=[1.0, np.inf])

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = pounce.minimize(
            f,
            np.full(3, 1.0 / 3),
            jac=g,
            hess=H,
            constraints=lc,
            tol=1e-10,
            print_level=0,
        )
    assert _no_hess_warning(rec)
    assert res.success
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)
    assert res.x[0] - res.x[1] >= -1e-8
    assert res.nhev > 0


def test_hess_still_dropped_with_dict_constraint():
    """A dict constraint is treated as nonlinear-by-policy (no probing), so a
    supplied ``hess`` is still dropped + warned, and the exact Hessian is NOT
    used (``nhev == 0`` → L-BFGS)."""
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x
    H = lambda x: 2.0 * np.eye(2)
    con = {
        "type": "eq",
        "fun": lambda x: x[0] + x[1] - 1.0,
        "jac": lambda x: np.array([[1.0, 1.0]]),
    }

    with pytest.warns(UserWarning, match="ignores the supplied 'hess'"):
        res = pounce.minimize(
            f, np.full(2, 0.5), jac=g, hess=H, constraints=con, tol=1e-10, print_level=0
        )
    assert res.success
    assert res.nhev == 0, "dict constraint must not trigger exact-Hessian use"


def test_hess_with_jac_true_and_linear_constraint():
    """``jac=True`` (fun returns (f, g)) composes with a separate ``hess`` and a
    LinearConstraint: the (f, g) cache and the exact Hessian are both used."""
    target = np.array([0.7, 0.1, 0.4])

    def fg(x):
        return 0.5 * float(((x - target) ** 2).sum()), x - target

    H = lambda x: np.eye(3)
    lc = opt.LinearConstraint(np.array([[1.0, 1.0, 1.0]]), lb=1.0, ub=1.0)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = pounce.minimize(
            fg,
            np.full(3, 1.0 / 3),
            jac=True,
            hess=H,
            constraints=lc,
            tol=1e-10,
            print_level=0,
        )
    assert _no_hess_warning(rec)
    assert res.success
    np.testing.assert_allclose(res.x.sum(), 1.0, atol=1e-8)
    assert res.nhev > 0 and res.nfev > 0


def test_convex_route_warns_on_dropped_options(monkeypatch):
    """L48: the dedicated convex (LP/QP) router honors only tol/max_iter, so
    NLP-only options like ``acceptable_tol``/``print_level`` are dropped. That
    must warn rather than happen silently."""
    import pounce._minimize as M

    class _Extract:
        kind = "qp"

    sentinel = M.OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        success=True,
        status=0,
        message="optimal",
        nit=1,
        info={"solver": "qp-ipm"},
    )

    monkeypatch.setattr(M, "classify_and_extract", lambda **kw: _Extract())
    monkeypatch.setattr(M, "_solve_via_convex", lambda ex, opts: sentinel)

    f = lambda x: float(x @ x)
    with pytest.warns(UserWarning, match="had no effect|were ignored|acceptable_tol"):
        res = M.minimize(
            f,
            np.ones(2),
            options={
                "solver_selection": "qp-ipm",
                "acceptable_tol": 1e-9,
                "print_level": 3,
            },
        )
    assert res is sentinel

    # Only honored options (tol/max_iter) -> no warning.
    monkeypatch.setattr(M, "classify_and_extract", lambda **kw: _Extract())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        M.minimize(
            f,
            np.ones(2),
            options={"solver_selection": "qp-ipm", "tol": 1e-8, "max_iter": 50},
        )


def test_convex_route_warns_on_disp(monkeypatch):
    """L48 residual: ``_solve_via_convex``/``_solve_via_socp`` read only
    ``tol``/``max_iter``, so ``disp=True`` is dropped on the convex routes just
    like ``print_level`` — it must trigger the same dropped-options warning
    rather than be listed as honored."""
    import pounce._minimize as M

    class _Extract:
        kind = "qp"

    sentinel = M.OptimizeResult(
        x=np.zeros(2),
        fun=0.0,
        success=True,
        status=0,
        message="optimal",
        nit=1,
        info={"solver": "qp-ipm"},
    )

    monkeypatch.setattr(M, "classify_and_extract", lambda **kw: _Extract())
    monkeypatch.setattr(M, "_solve_via_convex", lambda ex, opts: sentinel)

    f = lambda x: float(x @ x)
    with pytest.warns(UserWarning, match="disp"):
        res = M.minimize(
            f, np.ones(2), options={"solver_selection": "qp-ipm", "disp": True}
        )
    assert res is sentinel


class _UserStopProblem:
    """Fake native Problem that returns ``User_Requested_Stop`` (status 5) —
    what the bridge reports when the user's ``intermediate`` callback aborts or
    crashes (M32) — together with a *small* final KKT error."""

    def __init__(self, **kwargs):
        pass

    def add_option(self, key, value):
        pass

    def solve(self, x0):
        info = {
            "status": 5,  # User_Requested_Stop
            "status_msg": "User_Requested_Stop",
            "obj_val": 1.0,
            "iter_count": 3,
            "final_kkt_error": 1e-12,  # coincidentally below acceptable_tol
        }
        return np.asarray(x0, dtype=float), info


def test_user_requested_stop_is_not_success_despite_small_kkt(monkeypatch):
    """L50: the KKT-error fallback must not upgrade a ``User_Requested_Stop``
    to ``success=True``. A callback that aborted (or crashed, via M32) is an
    external stop, not a numerical stall at an acceptable point — even when the
    last computed KKT error happens to be below ``acceptable_tol``."""
    import pounce._minimize as M

    monkeypatch.setattr(M, "Problem", _UserStopProblem)

    f = lambda x: float(x @ x)
    res = M.minimize(
        f, np.ones(2), options={"solver_selection": "nlp", "print_level": 0}
    )
    assert res.status == 5
    assert res.success is False, "User_Requested_Stop must not be reported as success"

    # Control: a genuine numerical stall (Search_Direction_Becomes_Too_Small,
    # status 3) with the same small KKT error IS still upgraded to success.
    class _StallProblem(_UserStopProblem):
        def solve(self, x0):
            x, info = super().solve(x0)
            info["status"] = 3
            info["status_msg"] = "Search_Direction_Becomes_Too_Small"
            return x, info

    monkeypatch.setattr(M, "Problem", _StallProblem)
    res2 = M.minimize(
        f, np.ones(2), options={"solver_selection": "nlp", "print_level": 0}
    )
    assert res2.status == 3
    assert res2.success is True, "an acceptable-KKT numerical stall stays a success"


def test_non_mapping_options_is_rejected_not_silently_dropped():
    """A non-Mapping ``options=`` must raise, not be discarded.

    Found by an adversary probe following gh #213. ``options=`` was popped and
    merged only when it was a ``Mapping``; anything else fell through and the
    solve ran on defaults. That drops *every* option the caller passed --
    tolerances, iteration limits, ``solver_selection`` -- and still returns a
    plausible answer, so the mistake is invisible. Same failure mode as #213,
    one level up.
    """
    import pounce._minimize as M

    fun = lambda x: float(x @ x)
    jac = lambda x: 2.0 * x
    for bad in ([("tol", 1e-9)], ("tol", 1e-9), "tol=1e-9", 42):
        with pytest.raises(TypeError, match="must be a mapping"):
            M.minimize(fun, np.ones(2), jac=jac, options=bad)

    # The supported forms are untouched.
    assert M.minimize(fun, np.ones(2), jac=jac, options={"tol": 1e-9}).success
    assert M.minimize(fun, np.ones(2), jac=jac).success


def test_refused_problem_does_not_report_success():
    """A problem the solver *declines* must not come back ``success=True``.

    ``minimize`` upgrades a non-success status to success when the final KKT
    error is within the acceptable tolerance -- right for a solve that stalled
    near a good point. It used to fire for problems the solver had refused
    outright, because ``SolveStatistics`` defaulted its residuals to ``0.0``
    and "never computed" was indistinguishable from "converged exactly". An
    over-determined NLP came back ``Not_Enough_Degrees_Of_Freedom`` *and*
    ``success=True``, with an ``x`` outside its own declared bounds.

    The residuals now default to NaN, so the ``isfinite`` guard on that
    fallback does what its comment always claimed.
    """
    # 2 variables, 3 equality constraints.
    res = pounce.minimize(
        lambda v: (v[1] - 1.0) ** 2 + v[0] ** 2,
        [0.5, 0.0],
        jac=lambda v: np.array([2 * v[0], 2 * (v[1] - 1.0)]),
        bounds=[(0.3, 0.7), (-1e20, 1e20)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda v: np.array([v[0] - 5.0, v[0] + 5.0]),
                "jac": lambda v: np.array([[1.0, 0.0], [1.0, 0.0]]),
            },
            {
                "type": "eq",
                "fun": lambda v: np.array([v[1] - v[0] + 0.5]),
                "jac": lambda v: np.array([[-1.0, 1.0]]),
            },
        ],
    )
    assert not res.success, "a refused problem must not report success"
    assert res.info["status"] == -10  # Not_Enough_Degrees_Of_Freedom
    assert np.isnan(res.info["final_kkt_error"]), (
        "an uncomputed residual must be NaN, not 0.0 -- a zero here is what "
        "made the acceptable-KKT fallback fire on a refused solve"
    )
def test_active_set_sqp_honors_hessian_approximation():
    """An SQP solve must not demand a Hessian the caller never supplied.

    ``hessian_approximation=limited-memory`` is what a frontend sets when no
    second derivatives are available -- ``minimize`` does it automatically and
    warns that it has. That option was only read on the IPM path, so an
    active-set SQP solve fell back to its ``Exact`` default and asked the NLP
    for a Lagrangian Hessian nobody provided. The resulting zero Hessian turns
    the QP subproblem into an LP, unbounded along any null-space direction of
    the active constraints, and the solve died with ``Internal_Error`` on a
    problem the IPM solves without complaint.
    """
    fun = lambda v: (v[0] - 3.0) ** 2 + (v[1] - 2.0) ** 2
    jac = lambda v: np.array([2 * (v[0] - 3.0), 2 * (v[1] - 2.0)])
    con = [
        {
            "type": "ineq",
            "fun": lambda v: np.array([4.0 - v[0] - v[1]]),
            "jac": lambda v: np.array([[-1.0, -1.0]]),
        }
    ]
    # Constrained minimum: project (3, 2) onto x0 + x1 = 4.
    expected = np.array([2.5, 1.5])

    sqp = pounce.minimize(
        fun, [0.0, 0.0], jac=jac, constraints=con, solver_selection="qp-active-set"
    )
    assert sqp.success, f"active-set SQP failed: {sqp.message}"
    np.testing.assert_allclose(sqp.x, expected, atol=1e-6)

    # Same answer as the interior-point path.
    ipm = pounce.minimize(fun, [0.0, 0.0], jac=jac, constraints=con)
    np.testing.assert_allclose(sqp.x, ipm.x, atol=1e-6)

    # An explicit sqp_hessian still overrides the inferred default.
    for source in ("lbfgs", "damped-bfgs"):
        r = pounce.minimize(
            fun,
            [0.0, 0.0],
            jac=jac,
            constraints=con,
            solver_selection="qp-active-set",
            sqp_hessian=source,
        )
        assert r.success
        np.testing.assert_allclose(r.x, expected, atol=1e-6)


def test_active_set_sqp_exact_without_hessian_downgrades_not_internal_error():
    """Explicit ``sqp_hessian=exact`` with no available Hessian must fall back
    to L-BFGS, not die with ``Internal_Error`` (issue #348).

    The automatic ``hessian_approximation=limited-memory`` downgrade is applied
    first, but an explicit ``sqp_hessian=exact`` in the user options is applied
    *after* it and re-enabled the ``Exact`` source with no Hessian behind it:
    the driver evaluated a zero Lagrangian Hessian, so the QP subproblem went
    unbounded and the solve died with ``Internal_Error``. The fix downgrades an
    exact request to L-BFGS (with a warning) whenever the problem exposes no
    ``hessian`` / ``hessianstructure``.
    """
    fun = lambda v: (v[0] - 3.0) ** 2 + (v[1] - 2.0) ** 2
    jac = lambda v: np.array([2 * (v[0] - 3.0), 2 * (v[1] - 2.0)])
    hess = lambda v: 2.0 * np.eye(2)
    expected = np.array([2.5, 1.5])  # project (3, 2) onto x0 + x1 = 4

    # (a) dict constraint (nonlinear-by-policy) — previously Internal_Error,
    #     whether or not a hess is supplied (the facade attaches no Hessian).
    dict_con = [
        {
            "type": "ineq",
            "fun": lambda v: np.array([4.0 - v[0] - v[1]]),
            "jac": lambda v: np.array([[-1.0, -1.0]]),
        }
    ]
    for hh in (hess, None):
        with pytest.warns(UserWarning, match="exact Hessian was requested"):
            r = pounce.minimize(
                fun,
                [0.0, 0.0],
                jac=jac,
                hess=hh,
                constraints=dict_con,
                solver_selection="qp-active-set",
                sqp_hessian="exact",
            )
        assert r.success, f"exact-without-hessian must not fail: {r.message}"
        np.testing.assert_allclose(r.x, expected, atol=1e-6)
        assert r.nhev == 0, "downgrade to L-BFGS means the exact Hessian is unused"

    # (b) genuinely nonlinear constraint + hess: same downgrade, still solves.
    nl_con = [
        {
            "type": "eq",
            "fun": lambda v: v[0] ** 2 + v[1] ** 2 - 2.0,
            "jac": lambda v: np.array([[2 * v[0], 2 * v[1]]]),
        }
    ]
    with pytest.warns(UserWarning, match="exact Hessian was requested"):
        r = pounce.minimize(
            fun,
            [1.0, 1.0],
            jac=jac,
            hess=hess,
            constraints=nl_con,
            solver_selection="qp-active-set",
            sqp_hessian="exact",
        )
    assert r.success, f"exact-without-hessian (nonlinear) must not fail: {r.message}"

    # (c) control: an explicit LinearConstraint keeps the exact Hessian — no
    #     downgrade, no #348 warning, and the Hessian is actually used.
    lc = opt.LinearConstraint(np.array([[1.0, 1.0]]), lb=-np.inf, ub=4.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = pounce.minimize(
            fun,
            [0.0, 0.0],
            jac=jac,
            hess=hess,
            constraints=lc,
            solver_selection="qp-active-set",
            sqp_hessian="exact",
        )
    assert r.success
    np.testing.assert_allclose(r.x, expected, atol=1e-6)
    assert not any(
        "exact Hessian was requested" in str(w.message) for w in rec
    ), "exact must be honored (not downgraded) when a Hessian is available"
    assert r.nhev > 0, "the exact Hessian should actually be used here"


def test_uncomputed_objective_is_nan_not_zero():
    """An objective that was never evaluated must not be reported as ``0.0``.

    Same sentinel problem the residuals had: ``0.0`` is a perfectly ordinary
    objective value, so it cannot signal "never computed". A refused solve
    never reaches ``finalize_solution``, so ``obj_val`` used to come back as a
    fabricated zero.

    Lower stakes than the residual case -- nothing *decides* anything from the
    objective, it is only reported -- but one rule ("uncomputed is NaN") beats
    remembering which fields lie.
    """
    # 2 variables, 3 equality constraints -> refused.
    res = pounce.minimize(
        lambda v: (v[1] - 1.0) ** 2 + v[0] ** 2,
        [0.5, 0.0],
        jac=lambda v: np.array([2 * v[0], 2 * (v[1] - 1.0)]),
        bounds=[(0.3, 0.7), (-1e20, 1e20)],
        constraints=[
            {
                "type": "eq",
                "fun": lambda v: np.array([v[0] - 5.0, v[0] + 5.0]),
                "jac": lambda v: np.array([[1.0, 0.0], [1.0, 0.0]]),
            },
            {
                "type": "eq",
                "fun": lambda v: np.array([v[1] - v[0] + 0.5]),
                "jac": lambda v: np.array([[-1.0, 1.0]]),
            },
        ],
    )
    assert not res.success
    assert np.isnan(res.info["obj_val"]), (
        "a solve that evaluated nothing must not report an objective of 0.0"
    )

    # ...and a solve whose objective genuinely *is* zero still reports zero.
    ok = pounce.minimize(
        lambda v: float(v @ v),
        [1.0, 1.0],
        jac=lambda v: 2.0 * v,
        bounds=[(-5, 5), (-5, 5)],
    )
    assert ok.success
    assert not np.isnan(ok.info["obj_val"])
    assert ok.info["obj_val"] == pytest.approx(0.0, abs=1e-12)


# --- gh #275: non-finite inputs must not report success ------------------


def test_minimize_rejects_nonfinite_x0():
    """A NaN/inf x0 must raise, not report a successful solve.

    Every convergence test is a comparison against a tolerance, and
    comparisons against NaN are False — including the ones that would have
    rejected the iterate. Before #275 the loop fell straight through to
    "converged" at iteration zero and returned ``success=True, status=0,
    message="Solve_Succeeded", fun=nan, nit=0``.
    """
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x

    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match=r"x0\[0\] is .*must be finite"):
            pounce.minimize(f, np.array([bad, 1.0]), jac=g)

    # A non-first bad entry is reported at its own index.
    with pytest.raises(ValueError, match=r"x0\[1\] is nan"):
        pounce.minimize(f, np.array([1.0, np.nan]), jac=g)


def test_minimize_rejects_bounds_no_finite_value_can_satisfy():
    """``lb = +inf`` / ``ub = -inf`` admit no finite point.

    ``lb > ub`` catches the mixed spellings (``lower=+inf`` with a finite
    upper) but not ``lb == ub == ±inf``, where the comparison is False — so
    that box silently admitted any x.
    """
    f = lambda x: float(x @ x)
    g = lambda x: 2.0 * x

    with pytest.raises(ValueError, match=r"bounds\[0\] has lower bound inf"):
        pounce.minimize(f, np.array([0.5]), jac=g, bounds=[(np.inf, np.inf)])

    with pytest.raises(ValueError, match=r"bounds\[0\] has upper bound -inf"):
        pounce.minimize(f, np.array([0.5]), jac=g, bounds=[(-np.inf, -np.inf)])


def test_minimize_still_accepts_legitimate_infinite_bounds():
    """±inf on the *absent* side stays legal — that is the one-sided encoding."""
    f = lambda x: float((x[0] - 3.0) ** 2)
    g = lambda x: np.array([2.0 * (x[0] - 3.0)])

    r = pounce.minimize(f, np.array([0.5]), jac=g, bounds=[(-np.inf, np.inf)])
    assert r.success
    assert r.x[0] == pytest.approx(3.0, abs=1e-6)

    r2 = pounce.minimize(f, np.array([0.5]), jac=g, bounds=[(0.0, np.inf)])
    assert r2.success
    assert r2.x[0] == pytest.approx(3.0, abs=1e-6)

def test_minimize_extreme_objective_scale_monotone_reaches_bound_not_false_success():
    """gh #327: ``min 1/x`` over ``x in [1e-12, 10]`` from x0 = 1e-12.

    The objective is monotone decreasing, so the true optimum is the upper bound
    x = 10, f = 0.1. The enormous initial gradient (``-1/x**2 = -1e24`` at x0)
    pins pounce's gradient scaling at its 1e-8 floor, which deflates the scaled
    KKT error below ``tol`` all along the trajectory. Previously the default
    quasi-Newton path stopped at a non-stationary interior point x ≈ 2.84
    (f ≈ 0.35, |grad| = 0.124, neither bound active) and reported
    ``success=True, status=0`` — a false certificate. The masked-certificate veto
    now keeps the point the continuation actually reaches (the bound optimum),
    matching every independent oracle (ipopt MA57 / L-BFGS, scipy L-BFGS-B).
    """
    f = lambda z: 1.0 / z[0]
    g = lambda z: np.array([-1.0 / z[0] ** 2])

    r = pounce.minimize(
        f,
        np.array([1e-12]),
        jac=g,
        bounds=[(1e-12, 10.0)],
        solver_selection="nlp",
    )
    # Must reach the true bound optimum, not the interior false success.
    assert r.x[0] == pytest.approx(10.0, abs=1e-2)
    assert r.fun == pytest.approx(0.1, abs=1e-3)
    # And never report success at the non-stationary interior point x ≈ 2.84.
    assert r.x[0] > 5.0, (
        f"regressed to the gh #327 interior false optimum: x={r.x[0]}, f={r.fun}"
    )
