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
    """`qp-active-set` must dispatch to the **convex active-set driver**, the
    same engine the CLI reaches with `solver_selection=qp-active-set`.

    It originally fell through to the filter-IPM, indistinguishable from a bogus
    value. The first fix forwarded it to the backend, which ran the *SQP outer
    loop* — so the same selector named two different algorithms depending on
    whether you called the CLI or `minimize`. It now takes the same Python-side
    convex extraction as `qp-ipm` and dispatches to the active-set engine, so
    the two surfaces agree.

    The tell is `nfev`: the convex route consumes the extracted quadratic form
    and calls back into Python exactly once — the final `fun(x)` that makes the
    reported objective the objective *at the returned point* rather than the
    extracted model's value (gh #939) — while both the NLP and SQP paths report
    many.
    """
    fun, jac, con = _qp_problem()
    kw = dict(jac=jac, constraints=con)

    sel = minimize(fun, np.zeros(2), options={"solver_selection": "qp-active-set"}, **kw)
    ipm = minimize(fun, np.zeros(2), options={"solver_selection": "qp-ipm"}, **kw)
    nlp = minimize(fun, np.zeros(2), options={"solver_selection": "nlp"}, **kw)

    assert sel.nfev == 1, "qp-active-set must route to the convex driver, not a callback path"
    assert nlp.nfev > 0, "the NLP path must be distinguishable by callback count"
    # Same problem, same optimum, whichever convex engine ran.
    assert np.allclose(sel.x, ipm.x, atol=1e-6)
    assert sel.success
    assert np.allclose(sel.x, nlp.x, atol=1e-6), "both engines must still solve it"


def test_qp_active_set_solves_an_indefinite_qp(recwarn):
    """gh #786: `qp-active-set` is the one selector with an engine for a
    nonconvex QP, so `minimize` must route one there instead of refusing.

    `min ½xᵀPx + cᵀx` over `[−1, 1]²` with `P = diag(−2, 1)`,
    `c = (0.5, −0.5)` is separable: the concave `x₀` coordinate bottoms out at
    the endpoint `x₀ = −1` (`−1.5`, against `−0.5` at `x₀ = +1`) and the convex
    `x₁` one at `x₁ = 0.5` (`−0.125`), so `f* = −1.625` at `(−1, 0.5)`.

    `nfev == 1` is the tell that the convex driver ran: it consumes the
    extracted quadratic form and calls back into Python only for the final
    `fun(x)` at the returned point (gh #939).
    """
    P = np.diag([-2.0, 1.0])
    c = np.array([0.5, -0.5])

    def fun(x):
        return 0.5 * float(x @ P @ x) + float(c @ x)

    def jac(x):
        return P @ x + c

    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    res = minimize(
        fun,
        np.zeros(2),
        jac=jac,
        bounds=bounds,
        options={"solver_selection": "qp-active-set"},
    )
    assert res.success
    assert res.nfev == 1, "must route to the convex driver, not a callback path"
    assert res.fun == pytest.approx(-1.625, abs=1e-6)
    assert np.allclose(res.x, [-1.0, 0.5], atol=1e-6)
    assert res.info["problem_class"] == "nonconvex_qp"

    # `auto` must NOT take that route: the detection is an inference, and the
    # general NLP path is the safer default for a nonconvex model.
    auto = minimize(fun, np.zeros(2), jac=jac, bounds=bounds,
                    options={"solver_selection": "auto"})
    assert auto.nfev > 1, "auto must leave a nonconvex QP on the NLP path"


def test_qp_active_set_still_refuses_a_nonlinear_constraint():
    """The lift is about the *objective's* curvature. A curved constraint is
    not something this engine controls, and the extractor would drop it — so
    the selector must still refuse, and say which half failed."""
    def fun(x):
        return float(x[0] * x[1])

    def jac(x):
        return np.array([x[1], x[0]])

    con = [{"type": "ineq", "fun": lambda x: 1.0 - float(x @ x)}]
    with pytest.raises(ValueError, match="linear constraints"):
        minimize(fun, np.zeros(2), jac=jac, constraints=con,
                 options={"solver_selection": "qp-active-set"})


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


def test_jac_true_routes_like_a_separate_gradient_callable():
    # scipy's `jac=True` spelling: `fun(x)` returns `(f, grad)`. The routers
    # probe `fun` for a float and call `jac(x)` for a vector, so before the fix
    # they were handed a tuple-returning `fun` and the bare bool `True`; every
    # probe raised, the routers' catch-all read that as "not convex", and the
    # same convex QP that routes under `jac=callable` was rejected. (#750)
    #
    # min ½‖x‖² − x0 − 2x1 on [−5, 5]²  → x* = (1, 2), interior to the box.
    P, c = np.eye(2), np.array([-1.0, -2.0])
    f = lambda x: float(0.5 * x @ P @ x + c @ x)
    g = lambda x: P @ x + c
    fg = lambda x: (f(x), g(x))
    bounds = [(-5.0, 5.0)] * 2

    separate = minimize(f, np.zeros(2), jac=g, bounds=bounds,
                        options={"solver_selection": "qp-ipm"})
    packed = minimize(fg, np.zeros(2), jac=True, bounds=bounds,
                      options={"solver_selection": "qp-ipm"})

    assert _routed_to(separate) == _routed_to(packed) == "qp-ipm"
    np.testing.assert_allclose(packed.x, [1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(packed.x, separate.x, atol=1e-8)
    assert packed.fun == pytest.approx(separate.fun, abs=1e-8)

    # `auto` must take the convex fast path too, not fall through to NLP.
    auto = minimize(fg, np.zeros(2), jac=True, bounds=bounds,
                    options={"solver_selection": "auto"})
    assert _routed_to(auto) == "qp-ipm"
    np.testing.assert_allclose(auto.x, [1.0, 2.0], atol=1e-6)


def test_jac_true_lp_routes_to_lp_selector():
    # The same spelling on an LP: min −x0 + x1 on [−5, 5]² → x* = (5, −5).
    c = np.array([-1.0, 1.0])
    fg = lambda x: (float(c @ x), c.copy())
    res = minimize(fg, np.zeros(2), jac=True, bounds=[(-5.0, 5.0)] * 2,
                   options={"solver_selection": "lp-ipm"})

    assert _routed_to(res) == "lp-ipm"
    np.testing.assert_allclose(res.x, [5.0, -5.0], atol=1e-6)


def test_jac_true_pair_is_evaluated_once_per_probe_point():
    # The pair is cached per point (and shared with the SOCP router on `auto`),
    # so splitting `(f, grad)` must not double the user's forward passes.
    P, c = np.eye(2), np.array([-1.0, -2.0])
    calls = {"n": 0, "points": set()}

    def fg(x):
        calls["n"] += 1
        calls["points"].add(np.asarray(x, dtype=float).tobytes())
        return float(0.5 * x @ P @ x + c @ x), P @ x + c

    minimize(fg, np.zeros(2), jac=True, bounds=[(-5.0, 5.0)] * 2,
             options={"solver_selection": "qp-ipm"})

    assert calls["n"] == len(calls["points"])


def test_jac_true_gradient_buffer_reuse_does_not_poison_probes():
    # A `fun` that returns one reused gradient buffer must not corrupt cached
    # probe data (the hazard `_point_cache` guards against, on the pair cache).
    P, c = np.eye(2), np.array([-1.0, -2.0])
    buf = np.empty(2)

    def fg(x):
        buf[:] = P @ x + c
        return float(0.5 * x @ P @ x + c @ x), buf

    res = minimize(fg, np.zeros(2), jac=True, bounds=[(-5.0, 5.0)] * 2,
                   options={"solver_selection": "qp-ipm"})

    assert _routed_to(res) == "qp-ipm"
    np.testing.assert_allclose(res.x, [1.0, 2.0], atol=1e-6)


def test_jac_true_with_args_routes():
    # `args` binding and the `(f, grad)` split have to compose.
    fun = lambda x, k: ((x[0] - k) ** 2 + x[1] ** 2,
                        np.array([2 * (x[0] - k), 2 * x[1]]))
    res = minimize(fun, [0.0, 0.0], args=(3.0,), jac=True,
                   bounds=[(-10, 10), (-10, 10)],
                   options={"solver_selection": "qp-ipm"})

    assert _routed_to(res) == "qp-ipm"
    np.testing.assert_allclose(res.x, [3.0, 0.0], atol=1e-6)


def test_jac_false_routes_like_an_omitted_jac():
    # scipy's explicit "no gradient" spelling. `False is not None`, so it too
    # reached the routers as a non-callable and failed every probe. (#750)
    a = np.array([0.3, 0.7])
    fun = lambda x: float((x[0] - a[0]) ** 2 + (x[1] - a[1]) ** 2)
    res = minimize(fun, [0.0, 0.0], jac=False, bounds=[(0, 1), (0, 1)],
                   options={"solver_selection": "qp-ipm"})

    assert _routed_to(res) == "qp-ipm"
    np.testing.assert_allclose(res.x, a, atol=1e-5)


def test_jac_true_still_rejected_when_genuinely_not_convex():
    # The fix must not weaken detection: a quartic objective spelled `jac=True`
    # is still refused by a forced convex selector.
    fg = lambda x: (x[0] ** 4 + x[1] ** 2, np.array([4 * x[0] ** 3, 2 * x[1]]))
    with pytest.raises(ValueError):
        minimize(fg, [1.0, 1.0], jac=True, options={"solver_selection": "qp-ipm"})
