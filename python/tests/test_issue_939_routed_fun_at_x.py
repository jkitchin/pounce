"""gh #939 — on the convex routes, ``minimize`` reported the *model's*
objective as ``fun``, not ``fun(x)``.

``solver_selection="auto"`` (and the explicit ``"qp-ipm"`` / ``"lp-ipm"`` /
``"socp"``) does not hand the user's ``fun`` to the solver. ``pounce._route``
**finite-differences** the opaque callable into a quadratic model and the
convex solver optimizes *that*, so the value it reports back
(``res.obj + ex.obj_const``) is the model's value at ``x``, not the user's
objective at ``x``.

Reported: an 11-variable convex QP with bounds and a linear equality came back
with ``fun`` 8.8e-5 (1.1e-5 relative) **below** the problem's proven global
optimum — a value no feasible point attains — carrying ``success=True`` and
"Optimization terminated successfully.". ``fun(x)`` at that same returned ``x``
was right to 3e-9. scipy's contract, which this facade mirrors, is that ``fun``
is the objective evaluated at ``x``.

The fix evaluates the user's objective at the returned point, one extra call,
and keeps the model's own value as ``info["obj_model"]`` so the extraction
error stays measurable rather than being reported *as* the answer.

Mutation table (measured, not asserted from reading):

* make ``_objective_at_solution`` return ``(model_val, 0)`` unconditionally —
  the pre-fix behaviour — and 7 of the 10 go red: the two QP end-to-end tests,
  the SOCP arm, the ``args`` binding, both fallback arms, and the unit-level
  statement. The three that survive are the LP arm (its model is exact; see
  its docstring), the no-objective case, and the import guard.
* drop ``objective=route_fun`` at the ``_solve_via_convex`` call site only —
  3 red, all of them QP, SOCP green.
* drop it at the two ``_solve_via_socp`` call sites only — 1 red, the SOCP
  arm, QP green. That asymmetry is the point: neither arm's tests cover the
  other's wiring.

What this file is **not** evidence about:

* the returned ``x``. The FD model also moves the solution — 2.7e-5 from the
  optimum against the NLP route's 1.9e-8 on the issue's QP — and that is the
  cost of probing an opaque callable, untouched here. The tests compare
  ``fun`` against ``f(x)``, never assert that ``x`` is as good as the NLP
  route's.
* whether a given problem routes at all. ``_issue_qp`` says why it passes an
  exact ``jac``; the routing decision itself belongs to
  ``test_minimize_autoroute.py``.
* the CLI. It classifies from the ``.nl`` expression tree, so its convex model
  is symbolic and exact — this defect is specific to the callable facade.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

import pounce
from pounce import minimize


def _ball_constraint():
    # x0² + x1² ≤ 1  ⇔  g(x) = 1 − x0² − x1² ≥ 0 (concave g → convex set).
    return {
        "type": "ineq",
        "fun": lambda x: 1.0 - x[0] ** 2 - x[1] ** 2,
        "jac": lambda x: np.array([-2 * x[0], -2 * x[1]]),
    }


def _issue_qp():
    """The issue's own model: an 11-variable convex QP (cond(H) ≈ 33) with a
    box and one linear equality, built from the reported seed.

    The report's ``minimize`` call passes no ``jac``, which leaves *both* the
    gradient and the Hessian to finite differences — and whether that
    double-FD model clears the router's constant-Hessian check is a coin toss
    decided by the platform's floating-point noise (measured: it routes on
    aarch64-apple-darwin, and misses the check by 0.0035 against a 0.00255
    tolerance on x86-64 linux). An exact ``jac`` routes the same problem on
    every platform while leaving the *objective* model finite-differenced,
    which is the part this bug is about — so the test pins the defect instead
    of pinning one platform's FD noise.
    """
    rng = np.random.default_rng(2149425415)
    n = int(rng.integers(3, 12))
    M = rng.standard_normal((n, n))
    H = M @ M.T + np.diag(10 ** rng.uniform(-2, 2, n))
    g = rng.standard_normal(n) * 5
    lb = np.empty(n)
    ub = np.empty(n)
    for i in range(n):
        lb[i], ub[i] = -rng.uniform(0.1, 1), rng.uniform(0.1, 1)
    a = rng.standard_normal(n)
    b = rng.uniform(-0.2, 0.2)

    fun = lambda x: 0.5 * float(x @ H @ x) + float(g @ x)  # noqa: E731
    jac = lambda x: H @ x + g  # noqa: E731
    x0 = np.clip(a * b / (a @ a), lb, ub)
    kw = dict(
        jac=jac,
        bounds=list(zip(lb, ub)),
        constraints=[LinearConstraint(a[None, :], b, b)],
    )
    return fun, x0, kw


def test_issue_939_routed_qp_reports_the_objective_at_x():
    """The reported issue, end to end: the routed result's ``fun`` must be the
    user's objective at the returned ``x``, and must agree with the NLP route's
    optimum rather than sitting below it."""
    fun, x0, kw = _issue_qp()

    routed = minimize(fun, x0, solver_selection="qp-ipm", **kw)
    nlp = minimize(fun, x0, solver_selection="nlp", **kw)

    assert routed.info["solver"] == "qp-ipm"
    assert routed.success
    # The contract: `fun` IS the objective at `x`, to the last bit — not a
    # number the solver computed from its own model of the objective.
    assert routed.fun == float(fun(routed.x))
    # ...and so it can no longer sit below the optimum the NLP route proves.
    assert routed.fun == pytest.approx(nlp.fun, abs=1e-6)
    assert routed.fun >= nlp.fun - 1e-6


def test_routed_fun_is_the_objective_at_x_when_the_model_is_visibly_off():
    """The same defect with the extraction error made *deterministic* — a
    platform-independent measurement of what the pre-fix ``fun`` was.

    The router accepts a model that matches the objective to ``rtol`` (1e-5
    relative) at the held-out probes, so a barely-non-quadratic objective is
    routed **by design**, and the model it dispatches is then off by a term
    exact arithmetic can predict rather than by FD round-off nobody can pin.
    Here that term is 7.8e-5 at the solution — the same order as the 8.8e-5 the
    issue measured, and, like it, *below* the true objective.

    Pre-fix that number was returned as ``fun``. Post-fix it is still visible,
    as ``info["obj_model"]``, which is what keeps this test non-vacuous: the
    gap assertion proves the model really did disagree, so the equality above
    it is a repair and not a tautology.
    """
    scale = 1e6
    P = np.array([[4.0, 1.0], [1.0, 3.0]]) * scale
    c = np.array([-1.0, -2.0]) * scale
    eps = 1e-7 * scale  # within the router's 1e-5 relative model tolerance

    fun = lambda x: 0.5 * float(x @ P @ x) + float(c @ x) + eps * x[0] ** 3  # noqa: E731
    jac = lambda x: P @ x + c + np.array([3 * eps * x[0] ** 2, 0.0])  # noqa: E731

    res = minimize(
        fun,
        [0.0, 0.0],
        jac=jac,
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        solver_selection="auto",
    )

    assert res.info["solver"] == "qp-ipm"
    assert res.fun == float(fun(res.x))
    assert res.info["obj_val"] == res.fun  # info mirrors the reported value
    # Non-vacuity: the model the solver optimized really was off, and low.
    gap = res.info["obj_model"] - float(fun(res.x))
    assert abs(gap) > 1e-6, "the model must visibly disagree, or this proves nothing"
    assert gap < 0, "the pre-fix `fun` under-reported the objective, as reported"
    # One objective call: the final evaluation at `x`. No gradient/Hessian
    # callback fires on this route.
    assert (res.nfev, res.njev, res.nhev) == (1, 0, 0)


def test_routed_lp_reports_the_objective_at_x():
    """The LP arm folds ``obj_const`` back in; ``fun`` must still be ``f(x)``
    rather than that reconstruction.

    Unlike the QP arms, this one passes on the pre-fix code too: a linear
    objective's model is recovered *exactly* (the gradient is constant, so
    ``c`` is read off directly and ``d`` from one ``fun`` evaluation), so the
    model value and ``f(x)`` agree. It is here to say the arm was checked, not
    as the regression pin — the mutation table below records which tests are
    that.
    """
    fun = lambda x: -x[0] - 2 * x[1] + 7.5  # noqa: E731
    con = {"type": "ineq", "fun": lambda x: 1.0 - x[0] - x[1]}
    res = minimize(
        fun,
        [0.1, 0.1],
        bounds=[(0, None), (0, None)],
        constraints=con,
        solver_selection="auto",
    )

    assert res.info["solver"] == "lp-ipm"
    assert res.fun == float(fun(res.x))
    assert res.fun == pytest.approx(5.5, abs=1e-6)


def test_routed_socp_reports_the_objective_at_x():
    """The conic arm reads its ``P``/``c``/``obj_const`` out of the *same* FD
    probe of the objective, so it carries the identical defect and takes the
    identical fix."""
    fun = lambda x: (x[0] - 2) ** 2 + (x[1] - 2) ** 2  # noqa: E731
    jac = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])  # noqa: E731
    res = minimize(
        fun,
        [0.0, 0.0],
        jac=jac,
        constraints=[_ball_constraint()],
        solver_selection="socp",
    )

    assert res.info["solver"] == "socp"
    assert res.fun == float(fun(res.x))
    assert "obj_model" in res.info
    assert (res.nfev, res.njev, res.nhev) == (1, 0, 0)


def test_the_final_objective_call_binds_args():
    """The final evaluation goes through the same ``args``-bound callable the
    routers probed (#196's binding). Calling the raw ``fun`` instead would
    raise ``TypeError`` and silently fall back to the model value."""
    P = np.array([[2.0, 0.0], [0.0, 2.0]])

    def fun(x, shift, const):
        return float((x - shift) @ P @ (x - shift)) + const

    def jac(x, shift, const):
        return 2 * P @ (x - shift)

    args = (np.array([0.25, -0.5]), 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the fallback warning must not fire
        res = minimize(
            fun,
            [0.0, 0.0],
            args=args,
            jac=jac,
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            solver_selection="qp-ipm",
        )

    assert res.info["solver"] == "qp-ipm"
    assert res.fun == float(fun(res.x, *args))
    assert res.fun == pytest.approx(3.0, abs=1e-8)


# --- the fallback: the evaluation can fail at a point the user never chose ---


class _FakeQpResult:
    """Enough of ``QpResult`` for ``_solve_via_convex`` to adapt."""

    def __init__(self, x, obj):
        self.x = np.asarray(x, dtype=float)
        self.obj = float(obj)
        self.status = "optimal"
        self.iters = 7
        self.residuals = {}


class _FakeExtract:
    kind = "convex_qp"
    P = np.eye(2)
    c = np.zeros(2)
    obj_const = 0.5
    A = b = G = h = None
    lb = ub = None


def _patch_solve_qp(monkeypatch, x, obj):
    import pounce.qp as qp

    monkeypatch.setattr(qp, "solve_qp", lambda **kw: _FakeQpResult(x, obj))


@pytest.mark.parametrize(
    "objective, why",
    [
        (lambda x: (_ for _ in ()).throw(ValueError("outside the domain")), "raises"),
        (lambda x: float("nan"), "returns NaN"),
    ],
    ids=["raises", "nan"],
)
def test_objective_failure_at_the_returned_point_warns_and_keeps_the_model_value(
    monkeypatch, objective, why
):
    """The extra call happens *after* the solve, at a point the user never
    chose — an interior-point solver can land a hair outside a bound, where a
    restricted-domain ``fun`` blows up. Losing an otherwise good result to that
    would be a worse bug than the one being fixed, so the model value stands —
    but not silently, because the returned ``fun`` is then not ``fun(x)``.
    """
    from pounce._minimize import _solve_via_convex

    _patch_solve_qp(monkeypatch, [1.0, 2.0], obj=1.25)

    with pytest.warns(UserWarning, match="could not evaluate the objective"):
        res = _solve_via_convex(_FakeExtract(), {}, objective=objective)

    assert res.success
    np.testing.assert_allclose(res.x, [1.0, 2.0])
    assert res.fun == 1.75  # 1.25 + obj_const, the model's own value
    assert res.info["obj_model"] == 1.75


def test_the_reported_fun_is_the_objective_not_the_model_value(monkeypatch):
    """The unit-level statement of the fix, with the two numbers forced apart:
    the adapter is handed a model value of 1.75 and an objective that reads
    -3.0 at the returned point. It must report -3.0."""
    from pounce._minimize import _solve_via_convex

    _patch_solve_qp(monkeypatch, [1.0, 2.0], obj=1.25)

    res = _solve_via_convex(_FakeExtract(), {}, objective=lambda x: -3.0)

    assert res.fun == -3.0
    assert res.info["obj_val"] == -3.0
    assert res.info["obj_model"] == 1.75
    assert res.nfev == 1


def test_no_objective_means_no_extra_call(monkeypatch):
    """A caller that hands in no objective (an internal driver, a test double)
    still gets the model value and an honest ``nfev`` of 0 — the adapter must
    not invent an evaluation it did not make."""
    from pounce._minimize import _solve_via_convex

    _patch_solve_qp(monkeypatch, [1.0, 2.0], obj=1.25)

    res = _solve_via_convex(_FakeExtract(), {})

    assert res.fun == 1.75
    assert res.info["obj_model"] == 1.75
    assert res.nfev == 0


def test_pounce_exports_minimize():
    """Guard the import surface this file leans on."""
    assert pounce.minimize is minimize
