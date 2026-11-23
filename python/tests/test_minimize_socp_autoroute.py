"""Auto-routing of ``pounce.minimize`` to the conic (SOCP) solver for convex
QCQPs.

A convex *quadratically-constrained* QP has a quadratic constraint that the
LP/QP router rejects (it only accepts linear constraints). The QCQP router
(``pounce._route.classify_and_extract_socp``) probes each constraint's Hessian,
confirms the feasible set is convex (a scipy ``ineq`` ``g(x) ≥ 0`` qualifies
only when ``g`` is concave), and reformulates every convex-quadratic constraint
to one second-order cone — the same reformulation the CLI's
``extract_socp_with_map`` uses. These tests pin that convex QCQPs route to the
conic solver and agree with the NLP path, while nonconvex / nonlinear problems
stay on NLP.
"""

import numpy as np
import pytest

from pounce import minimize


def _routed_to(res):
    """The convex selector a result was routed through, or ``None`` for NLP."""
    return res.info.get("solver")


def _ball_constraint():
    # x0² + x1² ≤ 1  ⇔  g(x) = 1 − x0² − x1² ≥ 0 (concave g → convex set).
    return {
        "type": "ineq",
        "fun": lambda x: 1.0 - x[0] ** 2 - x[1] ** 2,
        "jac": lambda x: np.array([-2 * x[0], -2 * x[1]]),
    }


def test_convex_qcqp_routes_to_socp():
    # min −x0 − x1  s.t.  x0² + x1² ≤ 1  → x*=(1/√2, 1/√2), f*=−√2.
    fun = lambda x: -x[0] - x[1]
    jac = lambda x: np.array([-1.0, -1.0])
    res = minimize(fun, [0.1, 0.1], jac=jac, constraints=[_ball_constraint()],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "socp"
    assert res.info["problem_class"] == "convex_qcqp"
    assert res.success
    np.testing.assert_allclose(res.x, [1 / np.sqrt(2)] * 2, atol=1e-5)
    assert res.fun == pytest.approx(-np.sqrt(2.0), abs=1e-5)


def test_qcqp_with_quadratic_objective_and_constraint():
    # min (x0−2)² + (x1−2)²  s.t.  x0² + x1² ≤ 1.  Unconstrained min is (2,2),
    # which violates the ball, so the optimum sits on the boundary at the point
    # nearest (2,2): the radial point (1/√2, 1/√2).
    fun = lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    jac = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])
    res = minimize(fun, [0.0, 0.0], jac=jac, constraints=[_ball_constraint()],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "socp"
    np.testing.assert_allclose(res.x, [1 / np.sqrt(2)] * 2, atol=1e-5)


def test_routed_qcqp_matches_nlp_solve():
    # Transparency: the conic route must agree with the NLP solve.
    fun = lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    jac = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])
    kw = dict(jac=jac, constraints=[_ball_constraint()])

    auto = minimize(fun, [0.0, 0.0], options={"solver_selection": "auto"}, **kw)
    nlp = minimize(fun, [0.0, 0.0], options={"solver_selection": "nlp"}, **kw)

    assert _routed_to(auto) == "socp"
    assert _routed_to(nlp) is None
    np.testing.assert_allclose(auto.x, nlp.x, atol=1e-5)
    assert auto.fun == pytest.approx(nlp.fun, abs=1e-5)


def test_qcqp_folds_constraint_constant():
    # min x0  s.t.  (x0−3)² ≤ 1  → x0 ∈ [2, 4], optimum x0 = 2. The constraint's
    # linear (−6x0) and constant (+9) terms must be folded so the cone encodes
    # x0² − 6x0 + 8 ≤ 0, not x0² ≤ 1.
    fun = lambda x: x[0]
    jac = lambda x: np.array([1.0])
    con = {
        "type": "ineq",
        "fun": lambda x: 1.0 - (x[0] - 3.0) ** 2,
        "jac": lambda x: np.array([-2.0 * (x[0] - 3.0)]),
    }
    res = minimize(fun, [3.0], jac=jac, constraints=[con],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "socp"
    assert res.x[0] == pytest.approx(2.0, abs=1e-5)


def test_qcqp_with_linear_and_quadratic_constraints():
    # Mix a linear inequality and bounds with the quadratic constraint to
    # exercise the leading nonnegative block plus the SOC block.
    # min −x1  s.t.  x0² + x1² ≤ 1,  x0 ≥ 0,  0 ≤ x ≤ 1  → x*=(0, 1).
    fun = lambda x: -x[1]
    jac = lambda x: np.array([0.0, -1.0])
    lin = {"type": "ineq", "fun": lambda x: x[0], "jac": lambda x: np.array([1.0, 0.0])}
    res = minimize(
        fun, [0.1, 0.1], jac=jac,
        bounds=[(0, 1), (0, 1)], constraints=[_ball_constraint(), lin],
        options={"solver_selection": "auto"},
    )

    assert _routed_to(res) == "socp"
    # What the problem determines at the default tol=1e-8: the objective and
    # x1 to tol, but x0 only to sqrt(2·tol). On the top of the ball
    # x0 = sqrt(2·(1 − x1)), so a point whose objective is within ε of −1 can
    # sit at x0 ≈ sqrt(2ε) — feasible and optimal to tol. gh#1243 measured
    # it: folding the bound rows onto the KKT diagonal leaves the iterates the
    # same to ~9 digits, but the solve stops at iteration 20 at x0 = 1.4e-5
    # (objective gap 1.4e-10, x0² + x1² = 1 − 9e-11) instead of iteration 30
    # at x0 = 3e-8. The difference is the noise in this model's
    # primal-residual reading, not the point. The former `atol=1e-5` on x0
    # was tighter than the model allows.
    tol = 1e-8
    assert res.fun == pytest.approx(-1.0, abs=tol)
    assert res.x[1] == pytest.approx(1.0, abs=tol)
    assert -tol <= res.x[0] <= np.sqrt(2.0 * tol)
    assert res.x[0] ** 2 + res.x[1] ** 2 <= 1.0 + tol


def test_nonconvex_quadratic_constraint_stays_on_nlp():
    # g(x) = x0² + x1² − 1 ≥ 0: feasible region is OUTSIDE the ball, a nonconvex
    # set. The conic solver would be wrong, so the router must fall back to NLP.
    fun = lambda x: x[0] + x[1]
    jac = lambda x: np.array([1.0, 1.0])
    con = {
        "type": "ineq",
        "fun": lambda x: x[0] ** 2 + x[1] ** 2 - 1.0,
        "jac": lambda x: np.array([2 * x[0], 2 * x[1]]),
    }
    res = minimize(fun, [2.0, 2.0], jac=jac, constraints=[con],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) is None


def test_quadratic_equality_stays_on_nlp():
    # A quadratic *equality* x0² + x1² = 1 is a nonconvex feasible set; it must
    # never route to the conic solver.
    fun = lambda x: x[0] + x[1]
    jac = lambda x: np.array([1.0, 1.0])
    con = {
        "type": "eq",
        "fun": lambda x: x[0] ** 2 + x[1] ** 2 - 1.0,
        "jac": lambda x: np.array([2 * x[0], 2 * x[1]]),
    }
    res = minimize(fun, [0.5, 0.5], jac=jac, constraints=[con],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) is None


def test_forced_socp_on_nonlinear_constraint_raises():
    fun = lambda x: x[0] ** 2
    jac = lambda x: np.array([2 * x[0]])
    con = {
        "type": "ineq",
        "fun": lambda x: np.cos(x[0]) - 0.5,
        "jac": lambda x: np.array([-np.sin(x[0])]),
    }
    with pytest.raises(ValueError):
        minimize(
            fun, [0.5], jac=jac, constraints=[con],
            options={"solver_selection": "socp"},
        )


def test_forced_socp_on_plain_qp_raises():
    # A box QP has no quadratic constraint, so it is not a QCQP — forcing socp
    # must raise rather than silently solving it as a (trivial) cone program.
    fun = lambda x: x[0] ** 2 + x[1] ** 2
    jac = lambda x: np.array([2 * x[0], 2 * x[1]])
    with pytest.raises(ValueError):
        minimize(
            fun, [0.5, 0.5], jac=jac, bounds=[(0, 1), (0, 1)],
            options={"solver_selection": "socp"},
        )


def test_plain_qp_still_routes_to_qp_not_socp():
    # A convex QP with only linear constraints stays on the cheaper QP path even
    # though the QCQP router exists; auto must not divert it to the conic solver.
    fun = lambda x: x[0] ** 2 + x[1] ** 2 - 3 * x[0] - 4 * x[1]
    jac = lambda x: np.array([2 * x[0] - 3, 2 * x[1] - 4])
    res = minimize(fun, [0.5, 0.5], jac=jac, bounds=[(0, 1), (0, 1)],
                   options={"solver_selection": "auto"})

    assert _routed_to(res) == "qp-ipm"


def test_jac_true_qcqp_routes_to_socp():
    # scipy's `jac=True` spelling reached the SOCP router as a tuple-returning
    # `fun` plus the bare bool `True`, so every probe failed and a convex QCQP
    # was rejected by `solver_selection="socp"` / skipped by `auto`. (#750)
    # min −x0 + 2x1  s.t.  x0² + x1² ≤ 1  →  x* = (1, −2)/√5.
    c = np.array([-1.0, 2.0])
    fg = lambda x: (float(c @ x), c.copy())
    expected = -c / np.linalg.norm(c)

    forced = minimize(fg, [0.1, 0.1], jac=True, constraints=[_ball_constraint()],
                      options={"solver_selection": "socp"})
    assert _routed_to(forced) == "socp"
    np.testing.assert_allclose(forced.x, expected, atol=1e-5)

    auto = minimize(fg, [0.1, 0.1], jac=True, constraints=[_ball_constraint()],
                    options={"solver_selection": "auto"})
    assert _routed_to(auto) == "socp"
    np.testing.assert_allclose(auto.x, expected, atol=1e-5)
