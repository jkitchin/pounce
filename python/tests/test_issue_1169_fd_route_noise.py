"""gh #1169: convex LP/QP detection without ``jac`` must not depend on the
objective's constant term.

With no ``jac`` the router finite-differences ``fun`` for the gradient, and
that gradient again for the Hessian. At the old ``sqrt(eps)`` / ``eps^(1/3)``
steps the nested Hessian carried ``~2.5e-3·|f|`` of round-off, so a convex QP
with a constant offset of 1 failed the Hessian-constancy gate. ``auto`` then
fell back to the NLP solver, and a forced ``qp-ipm`` raised ``ValueError``. An
LP was misread as a QP at offset 100 and stopped routing at 1000.

A central difference of a quadratic has no truncation error at any step, so
the router now uses a large step (``_H_STRUCT``) kept inside the box, with a
one-sided stencil that is still exact on that class where a probe sits on a
bound. These tests cover:

* each stencil branch of ``_box_diff`` (central, one-sided 3-point, one-sided
  2-point, the narrow-box fallback), each exact on its class;
* the routing verdict across offsets, for a QP and an LP, at an anchor in the
  interior and on a bound, with and without a box;
* the dangerous direction, which must still stay on the NLP path:
  non-quadratic objectives, including ones with a large offset, and a
  non-finite objective.
"""

import numpy as np
import pytest

from pounce import minimize
from pounce._route import _H_GRAD, _H_HESS, _H_STRUCT, _box_diff

Q = np.diag([1.0, 2.0, 3.0, 4.0])
Q_ISSUE_C = np.array([1.2, -0.7, 2.4, -3.1])
Q_SWEEP_C = np.array([0.3, -0.2, 0.5, -1.0])
BOX = [(-1.0, 1.0)] * 4


def _routed_to(res):
    return res.info.get("solver")


def _closed_form(c):
    """Minimizer and value of ½xᵀQx + cᵀx on the box (Q is diagonal)."""
    x = np.clip(-c / np.diag(Q), -1.0, 1.0)
    return x, 0.5 * x @ Q @ x + c @ x


# --- _box_diff: one test per stencil branch ---------------------------------

_P = np.array([[3.0, 1.0], [1.0, 2.0]])
_C = np.array([0.5, -1.5])


def _quad(x):
    return 0.5 * x @ _P @ x + _C @ x + 1e3


def _grad(x):
    return _P @ x + _C


@pytest.mark.parametrize(
    "x, lb, ub, branch",
    [
        (np.array([0.2, -0.3]), None, None, "central (no box)"),
        (
            np.array([0.2, -0.3]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            "central",
        ),
        (
            np.array([1.0, -0.3]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            "one-sided down",
        ),
        (
            np.array([-1.0, -0.3]),
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
            "one-sided up",
        ),
    ],
)
def test_box_diff_gradient_is_exact_on_a_quadratic(x, lb, ub, branch):
    got = np.array(
        [
            _box_diff(_quad, x, i, _H_STRUCT, _H_GRAD, lb, ub, quadratic=True)
            for i in range(2)
        ]
    )
    # Round-off only: ~eps·|f|/h with |f| ≈ 1e3.
    np.testing.assert_allclose(got, _grad(x), rtol=0, atol=1e-10, err_msg=branch)


@pytest.mark.parametrize("on", [1.0, -1.0])
def test_box_diff_hessian_one_sided_is_exact_on_an_affine_gradient(on):
    x = np.array([on, 0.0])
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    col = _box_diff(_grad, x, 0, _H_STRUCT, _H_HESS, lb, ub, quadratic=False)
    np.testing.assert_allclose(col, _P[:, 0], rtol=0, atol=1e-12)


def test_box_diff_stays_inside_the_box():
    seen = []

    def f(x):
        seen.append(x.copy())
        return _quad(x)

    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    for x0 in ([1.0, 0.0], [-1.0, 0.0], [1.0 - 1e-3, 0.0]):
        _box_diff(f, np.array(x0), 0, _H_STRUCT, _H_GRAD, lb, ub, quadratic=True)
    pts = np.array(seen)
    assert np.all(pts >= lb) and np.all(pts <= ub)


def test_box_diff_narrow_box_falls_back_to_the_old_central_step():
    # Width far below the floor step: the stencil must be the pre-#1169
    # central one at `floor`, which leaves the box by that floor at most.
    seen = []

    def f(x):
        seen.append(x.copy())
        return _quad(x)

    x = np.array([0.0, 0.0])
    w = 1e-12
    _box_diff(
        f,
        x,
        0,
        _H_STRUCT,
        _H_GRAD,
        np.array([-w, -1.0]),
        np.array([w, 1.0]),
        quadratic=True,
    )
    steps = sorted(p[0] for p in seen)
    assert steps == pytest.approx([-_H_GRAD, _H_GRAD], rel=1e-12)


# --- routing verdict ---------------------------------------------------------


def test_issue_reproduction_forced_qp_ipm_without_jac():
    f = lambda x: 0.5 * x @ Q @ x + Q_ISSUE_C @ x
    res = minimize(f, np.zeros(4), bounds=BOX, solver_selection="qp-ipm")
    xs, fs = _closed_form(Q_ISSUE_C)
    assert _routed_to(res) == "qp-ipm"
    assert res.fun == pytest.approx(fs, abs=1e-7)
    np.testing.assert_allclose(res.x, xs, atol=1e-6)


@pytest.mark.parametrize("off", [0.0, 1.0, 10.0, 100.0, 1e3, 1e5])
@pytest.mark.parametrize("x0", [np.zeros(4), np.ones(4)], ids=["interior", "on-bound"])
def test_qp_without_jac_routes_whatever_the_offset(off, x0):
    f = lambda x: 0.5 * x @ Q @ x + Q_SWEEP_C @ x + off
    res = minimize(f, x0, bounds=BOX, solver_selection="auto")
    xs, fs = _closed_form(Q_SWEEP_C)
    assert _routed_to(res) == "qp-ipm"
    assert res.info["problem_class"] == "convex_qp"
    assert res.fun - off == pytest.approx(fs, abs=1e-6 * max(1.0, off))
    np.testing.assert_allclose(res.x, xs, atol=1e-6)


@pytest.mark.parametrize("off", [0.0, 100.0, 1e3, 1e5])
def test_lp_without_jac_routes_as_an_lp_whatever_the_offset(off):
    f = lambda x: Q_SWEEP_C @ x + off
    res = minimize(f, np.zeros(4), bounds=BOX, solver_selection="auto")
    assert _routed_to(res) == "lp-ipm"
    assert res.fun - off == pytest.approx(
        -np.abs(Q_SWEEP_C).sum(), abs=1e-6 * max(1.0, off)
    )


@pytest.mark.parametrize("off", [0.0, 1e3])
def test_unboxed_qp_with_linear_constraint_routes(off):
    # No bounds at all: every stencil is central at the full step.
    f = lambda x: 0.5 * x @ Q @ x + Q_SWEEP_C @ x + off
    cons = [
        {"type": "ineq", "fun": lambda x: 1.0 - x.sum(), "jac": lambda x: -np.ones(4)}
    ]
    res = minimize(f, np.zeros(4), constraints=cons, solver_selection="auto")
    assert _routed_to(res) == "qp-ipm"
    x = -Q_SWEEP_C / np.diag(Q)  # the constraint is inactive here (sum = -0.0417)
    assert res.fun - off == pytest.approx(0.5 * x @ Q @ x + Q_SWEEP_C @ x, abs=1e-6)


# --- the dangerous direction stays on the NLP path ---------------------------

_NONQUADRATIC = {
    "quartic": lambda x: np.sum(x**4) + Q_SWEEP_C @ x,
    "quartic+offset": lambda x: np.sum(x**4) + Q_SWEEP_C @ x + 1e3,
    "small cubic": lambda x: 0.5 * x @ Q @ x + 0.05 * x[0] ** 3,
    "small cubic+offset": lambda x: 0.5 * x @ Q @ x + 0.05 * x[0] ** 3 + 1e3,
    "exp": lambda x: np.sum(np.exp(x)),
    "abs": lambda x: np.sum(np.abs(x - 0.1)),
}


@pytest.mark.parametrize("name", list(_NONQUADRATIC))
def test_nonquadratic_objective_is_not_routed(name):
    f = _NONQUADRATIC[name]
    res = minimize(f, np.zeros(4), bounds=BOX, solver_selection="auto")
    assert _routed_to(res) is None, name
    with pytest.raises(ValueError, match="not detected as a convex"):
        minimize(f, np.zeros(4), bounds=BOX, solver_selection="qp-ipm")


def test_nonfinite_objective_is_not_routed():
    # NaN compares false against every gate, so a NaN probe must be caught
    # explicitly rather than read as "within tolerance". NaN on half the box
    # is certain to meet some probe or stencil point.
    f = lambda x: 0.5 * x @ Q @ x + (np.nan if x[0] > 0.0 else 0.0)
    with pytest.raises(ValueError, match="not detected as a convex"):
        minimize(f, np.zeros(4), bounds=BOX, solver_selection="qp-ipm")
