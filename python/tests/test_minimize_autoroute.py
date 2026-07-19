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


def test_convex_route_warns_dropped_callback():
    # issue #196 (related): the convex/SOCP routers consume the extracted
    # quadratic form and never call back into Python, so a user `callback`
    # cannot fire on that route. `callback` is a named argument (not in
    # `options`), so it must be surfaced explicitly in the dropped-options
    # warning rather than silently ignored.
    fun = lambda x: x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1] + 5.0
    jac = lambda x: np.array([2 * x[0] - 3, 2 * x[1] - 4])
    hess = lambda x: np.array([[2.0, 0.0], [0.0, 2.0]])
    seen = []
    with pytest.warns(UserWarning, match=r"callback \(argument\)"):
        res = minimize(
            fun, [0.5, 0.5], jac=jac, hess=hess, bounds=[(0, 1), (0, 1)],
            callback=lambda xk: seen.append(1),
            options={"solver_selection": "auto"},
        )
    assert _routed_to(res) == "qp-ipm"  # still took the convex fast path
    assert seen == []  # callback did not fire — exactly what the warning says


def test_args_are_bound_into_convex_router_probes():
    # A parameterized convex QP min (x0 − c)² + x1² with args=(c,). The router
    # probes fun/jac/hess as bare f(x), so before the fix `args` was dropped:
    # `auto` silently fell back to NLP (never routed) and a forced `qp-ipm` was
    # wrongly rejected as "not convex". Both must now route and land x0 = c.
    fun = lambda x, c: (x[0] - c) ** 2 + x[1] ** 2
    jac = lambda x, c: np.array([2 * (x[0] - c), 2 * x[1]])
    hess = lambda x, c: np.array([[2.0, 0.0], [0.0, 2.0]])
    kw = dict(args=(3.0,), jac=jac, hess=hess, bounds=[(-10, 10), (-10, 10)])

    auto = minimize(fun, [0.0, 0.0], options={"solver_selection": "auto"}, **kw)
    assert _routed_to(auto) == "qp-ipm"  # args-bound probe now detects the QP
    np.testing.assert_allclose(auto.x, [3.0, 0.0], atol=1e-6)

    # Forced convex must no longer spuriously reject a genuinely convex QP.
    forced = minimize(fun, [0.0, 0.0], options={"solver_selection": "qp-ipm"}, **kw)
    assert _routed_to(forced) == "qp-ipm"
    np.testing.assert_allclose(forced.x, [3.0, 0.0], atol=1e-6)


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


def test_unbounded_lp_reports_unbounded_not_iteration_limit():
    """gh #160: an unbounded LP routed to the convex solver must report a
    distinct unbounded status, not a generic iteration limit.

        min -x0 - x1  s.t. x0 - x1 <= 1,  x0, x1 >= 0
    is unbounded along x0 = x1 + 1, x1 -> inf. The NLP path can only hit
    max_iter here (its iterates grow ~linearly, never reaching
    diverging_iterates_tol), so LP callers route to the LP solver, whose HSDE
    returns a dual-infeasibility certificate => primal unbounded.
    """
    fun = lambda x: -x[0] - x[1]
    jac = lambda x: np.array([-1.0, -1.0])
    con = {"type": "ineq",
           "fun": lambda x: 1.0 - (x[0] - x[1]),   # con(x) >= 0
           "jac": lambda x: np.array([-1.0, 1.0])}
    res = minimize(fun, [0.5, 0.5], jac=jac, bounds=[(0.0, None), (0.0, None)],
                   constraints=con,
                   options={"solver_selection": "lp-ipm", "max_iter": 3000})

    assert _routed_to(res) == "lp-ipm"          # took the LP path, not NLP
    assert not res.success
    assert res.status == 3                       # scipy-linprog: 3 == unbounded
    assert "unbounded" in res.message.lower()    # distinct, not "max iterations"
    # Raw HSDE certificate still available for programmatic callers.
    assert res.info["status"] == "dual_infeasible"


# --- gh #213: solver_selection must be validated, not silently dropped -------

def _qp_problem():
    """min ½xᵀPx + cᵀx  s.t.  x0 + x1 >= 1  — a convex QP with an active
    constraint at the optimum, so the SQP and IPM engines take visibly
    different iteration counts."""
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    c = np.array([-2.0, -2.0])
    con = [{"type": "ineq",
            "fun": lambda x: 1.0 - (x[0] + x[1]),
            "jac": lambda x: np.array([-1.0, -1.0])}]
    return (lambda x: 0.5 * x @ P @ x + c @ x,
            lambda x: P @ x + c,
            con)


@pytest.mark.parametrize("bad", ["qp_ipm", "totally-bogus-solver", "", "QP_IPM"])
def test_invalid_solver_selection_raises(bad):
    """An unrecognized selector must raise, not fall through to the NLP path.

    This is the gh #213 defect. Silent fallback is the dangerous failure mode
    precisely because it still returns a *correct* answer on easy problems: a
    typo (`qp_ipm` for `qp-ipm`) benchmarks or ships an engine the caller never
    asked for, with nothing in the result to reveal it.
    """
    fun, jac, con = _qp_problem()
    with pytest.raises(ValueError, match="not a valid selector"):
        minimize(fun, np.zeros(2), jac=jac, constraints=con,
                 options={"solver_selection": bad})


def test_qp_active_set_reaches_the_sqp_engine():
    """`qp-active-set` is a valid CLI/library selector and must actually
    dispatch, not be swallowed.

    Before the fix it was indistinguishable from a bogus value: both fell
    through to the filter-IPM. The backend treats it as equivalent to
    `algorithm=active-set-sqp`, so pinning it against that spelling proves the
    option reached the Rust side rather than merely being accepted by Python.
    """
    fun, jac, con = _qp_problem()
    kw = dict(jac=jac, constraints=con)

    sel = minimize(fun, np.zeros(2), options={"solver_selection": "qp-active-set"}, **kw)
    algo = minimize(fun, np.zeros(2), options={"algorithm": "active-set-sqp"}, **kw)
    nlp = minimize(fun, np.zeros(2), options={"solver_selection": "nlp"}, **kw)

    assert sel.nit == algo.nit, "qp-active-set must select the same engine as algorithm=active-set-sqp"
    assert np.allclose(sel.x, algo.x)
    assert sel.nit != nlp.nit, "qp-active-set must be distinguishable from the NLP path"
    assert np.allclose(sel.x, nlp.x, atol=1e-6), "both engines must still solve it"


def test_solver_selection_is_case_insensitive():
    """The Rust side compares with `eq_ignore_ascii_case`; match it, so a
    selector that works on the CLI is not rejected here on casing alone."""
    fun, jac, con = _qp_problem()
    res = minimize(fun, np.zeros(2), jac=jac, constraints=con,
                   options={"solver_selection": "QP-Active-Set"})
    assert res.success


def test_solver_selection_values_match_rust():
    """Drift guard: the Python whitelist must equal the Rust registry.

    A hardcoded list would silently diverge the moment a selector is added on
    the Rust side — and the failure would be exactly the bug this test suite
    exists to prevent, a valid selector rejected (or a dropped one accepted) by
    the facade alone. Parse the registration instead.
    """
    import re
    from pathlib import Path

    from pounce._minimize import _SOLVER_SELECTION_VALUES

    src = (Path(__file__).resolve().parents[2]
           / "crates/pounce-algorithm/src/upstream_options.rs").read_text()
    block = re.search(
        r'add_string_option\(\s*"solver_selection".*?\n        \],', src, re.S
    )
    assert block, "could not locate the solver_selection registration in Rust"
    rust_values = set(re.findall(r'^\s*\(\s*\n?\s*"([a-z-]+)",', block.group(0), re.M))

    assert rust_values, "parsed no values; the registration format changed"
    assert rust_values == set(_SOLVER_SELECTION_VALUES), (
        f"Python whitelist {sorted(_SOLVER_SELECTION_VALUES)} != "
        f"Rust registry {sorted(rust_values)}"
    )
