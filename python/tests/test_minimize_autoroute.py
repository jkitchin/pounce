"""Auto-routing of ``pounce.minimize`` to the convex LP/QP solver.

``minimize`` takes opaque callables, so the router (``pounce._route``) probes
them, fits a linear/quadratic model, and validates it at held-out points
before dispatching to ``solve_qp``. These tests pin the two correctness
properties that matter: genuine LP/convex-QP problems route (and report the
right objective, constant included), while nonlinear / nonconvex problems
stay on the NLP path — the router never silently sends them to the QP solver.
"""

import numpy as np
import pytest

from pounce import minimize


def _routed_to(res):
    """The convex selector a result was routed through, or ``None`` for NLP."""
    return res.info.get("solver")


def test_convex_qp_routes_and_recovers_objective_constant():
    # min x0² + x1² − 3x0 − 4x1 + 5  s.t. 0 ≤ x ≤ 1  → x*=(1,1), f*=0.
    # The +5 constant lives only in `fun`; the QP solver never sees it, so the
    # reported objective must add it back (the Finding-#1 issue, Python side).
    fun = lambda x: x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1] + 5.0
    jac = lambda x: np.array([2 * x[0] - 3, 2 * x[1] - 4])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    res = minimize(fun, [0.5, 0.5], jac=jac, hess=hess, bounds=[(0, 1), (0, 1)],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "qp-ipm"
    assert res.info["problem_class"] == "convex_qp"
    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-6)
    assert res.fun == pytest.approx(0.0, abs=1e-6)  # constant folded back in
    assert res.info["obj_constant"] == pytest.approx(5.0)


def test_lp_routes_to_lp_selector():
    # min −x0 − 2x1  s.t.  x0 + x1 ≤ 1,  x ≥ 0  → x*=(0,1), f*=−2.
    fun = lambda x: -x[0] - 2 * x[1]
    con = {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]}  # ≥ 0
    res = minimize(fun, [0.1, 0.1], bounds=[(0, None), (0, None)], constraints=con,
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "lp-ipm"
    assert res.info["problem_class"] == "lp"
    np.testing.assert_allclose(res.x, [0.0, 1.0], atol=1e-6)
    assert res.fun == pytest.approx(-2.0, abs=1e-6)


def test_routed_qp_matches_nlp_solve():
    # The router must be transparent: forcing NLP gives the same optimum.
    fun = lambda x: x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1]
    jac = lambda x: np.array([2 * x[0] - 3, 2 * x[1] - 4])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    kw = dict(jac=jac, hess=hess, bounds=[(0, 1), (0, 1)])

    auto = minimize(fun, [0.5, 0.5], options={"solver_selection": "auto"}, **kw)
    nlp = minimize(fun, [0.5, 0.5], options={"solver_selection": "nlp"}, **kw)

    assert _routed_to(auto) == "qp-ipm"
    assert _routed_to(nlp) is None  # forced onto the NLP path
    np.testing.assert_allclose(auto.x, nlp.x, atol=1e-6)
    assert auto.fun == pytest.approx(nlp.fun, abs=1e-6)


def test_point_cache_stores_defensive_copies():
    """M34: the router's probe cache must store copies, not the user's return
    object. A ``jac``/``hess`` callable that reuses one output buffer across
    calls would otherwise mutate earlier cache entries in place and poison the
    routers' probe data."""
    # The cache itself is pure Python (numpy only), but importing it pulls in
    # the pounce package __init__ and thus the native extension; skip when
    # that's unavailable rather than fail on an unrelated import.
    route = pytest.importorskip("pounce._route")

    buf = np.zeros(2)
    n_calls = [0]

    def jac(x):
        n_calls[0] += 1
        buf[:] = x  # reuse ONE buffer: the previous return value is mutated
        return buf

    cached = route._point_cache(jac)
    x1 = np.array([1.0, 2.0])
    x2 = np.array([3.0, 4.0])
    v1 = cached(x1)
    v2 = cached(x2)  # mutates `buf` in place
    # The second call must not have poisoned the first cached value...
    np.testing.assert_array_equal(np.asarray(v1), [1.0, 2.0])
    np.testing.assert_array_equal(np.asarray(v2), [3.0, 4.0])
    # ...and a cache hit returns the stored copy without re-evaluating.
    np.testing.assert_array_equal(np.asarray(cached(x1)), [1.0, 2.0])
    assert n_calls[0] == 2

    # Scalars (a cached `fun` value) are stored as 0-d arrays, which the
    # routers' `float(...)` / `np.asarray(...)` consumers accept.
    fun = route._point_cache(lambda x: 7.5)
    assert float(fun(x1)) == 7.5
    assert float(fun(x1)) == 7.5  # cache hit
    # None (absent jac/hess) still passes through untouched.
    assert route._point_cache(None) is None


def test_nonlinear_objective_stays_on_nlp():
    # Rosenbrock: quartic, not a quadratic — must NOT be routed to the QP solver.
    fun = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    jac = lambda x: np.array(
        [
            -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
            200 * (x[1] - x[0] ** 2),
        ]
    )
    res = minimize(fun, [-1.2, 1.0], jac=jac, options={"solver_selection": "auto"})

    assert _routed_to(res) is None
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-4)


def test_nonconvex_qp_stays_on_nlp():
    # Indefinite Hessian diag(−2, 2): a *nonconvex* QP. The convex solver would
    # be wrong here, so the router must reject it and fall back to NLP.
    fun = lambda x: -(x[0] ** 2) + x[1] ** 2
    jac = lambda x: np.array([-2 * x[0], 2 * x[1]])
    hess = lambda x: np.array([[-2.0, 0.0], [0.0, 2.0]])
    res = minimize(fun, [0.5, 0.5], jac=jac, hess=hess, bounds=[(0, 1), (0, 1)],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) is None


def test_forced_lp_on_nonlinear_raises():
    fun = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    with pytest.raises(ValueError):
        minimize(fun, [-1.2, 1.0], options={"solver_selection": "lp-ipm"})


def test_forced_qp_on_nonlinear_raises():
    fun = lambda x: x[0] ** 4 + x[1] ** 2
    with pytest.raises(ValueError):
        minimize(fun, [1.0, 1.0], options={"solver_selection": "qp-ipm"})


def test_finite_difference_qp_routes_without_user_derivatives():
    # No jac/hess supplied: the router fits the quadratic by finite differences
    # and the held-out validation confirms it. min ½‖x−a‖² style box QP.
    a = np.array([0.3, 0.7])
    fun = lambda x: float((x[0] - a[0]) ** 2 + (x[1] - a[1]) ** 2)
    res = minimize(fun, [0.0, 0.0], bounds=[(0, 1), (0, 1)],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "qp-ipm"
    np.testing.assert_allclose(res.x, a, atol=1e-5)


def test_auto_route_probes_objective_once_not_twice():
    # M34: on the `auto` path both the LP/QP router and the SOCP/QCQP router run
    # in sequence, each finite-differencing the *same* objective at the *same*
    # probe points (identical seed). A shared point-cache (wired in `_minimize`
    # via `_route._point_cache`) makes the second router's probes cache hits, so
    # the routing overhead is one router's worth of `fun` calls, not two.
    from pounce._route import classify_and_extract

    n = 5
    x0 = np.full(n, 0.3)

    def _counting_quartic():
        calls = {"n": 0}

        def fun(x):
            calls["n"] += 1
            return float(np.sum(np.asarray(x) ** 4))  # quartic → NLP route

        return fun, calls

    # Auto path: both routers probe, then the problem falls through to NLP.
    # (pounce defaults to solver_selection="nlp"; opt in explicitly to route.)
    f_auto, c_auto = _counting_quartic()
    minimize(f_auto, x0, options={"solver_selection": "auto"})
    # NLP-forced path: no routing, so the difference isolates the routing cost.
    f_nlp, c_nlp = _counting_quartic()
    minimize(f_nlp, x0, options={"solver_selection": "nlp"})
    # One router in isolation: the unit the shared cache should collapse to.
    f_one, c_one = _counting_quartic()
    classify_and_extract(
        fun=f_one,
        jac=None,
        hess=None,
        lb=None,
        ub=None,
        m=0,
        g_combined=None,
        jac_combined=None,
        cl=None,
        cu=None,
        x0=x0,
        rtol=1e-5,
    )

    routing_overhead = c_auto["n"] - c_nlp["n"]
    # Post-fix the overhead equals a single router's probe count; pre-fix (no
    # shared cache) it was 2× because each router re-probed from scratch.
    assert routing_overhead == c_one["n"]
