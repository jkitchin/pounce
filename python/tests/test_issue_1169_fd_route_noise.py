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
from pounce._route import (
    _EPS,
    _H_GRAD,
    _H_HESS,
    _H_STRUCT,
    _box_diff,
    _flat_along,
    classify_and_extract,
)

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


@pytest.mark.parametrize(
    "x, lb, ub, quadratic",
    [
        (np.array([0.2, -0.3]), None, None, True),
        (np.array([1.0, -0.3]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), True),
        (np.array([1.0, -0.3]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), False),
        (np.array([0.0, 0.0]), np.array([-1e-12, -1.0]), np.array([1e-12, 1.0]), True),
    ],
    ids=["central", "one-sided 3-point", "one-sided 2-point", "narrow fallback"],
)
@pytest.mark.parametrize("off", [0.0, 1e7, 1e12])
def test_box_diff_noise_bounds_its_own_error(x, lb, ub, quadratic, off):
    # The reported bound must cover the actual round-off, and must grow with
    # |f|: that is what lets the LP gate allow for it.
    f = lambda y: _C @ y + off  # affine: every branch is exact on it
    nz = []
    got = _box_diff(f, x, 0, _H_STRUCT, _H_GRAD, lb, ub, quadratic=quadratic, noise=nz)
    assert len(nz) == 1
    assert abs(got - _C[0]) <= nz[0]
    assert nz[0] >= _EPS * off / 2.0


def test_flat_along_affine_with_huge_offset():
    a, p = np.zeros(4), np.ones(4)
    assert _flat_along(lambda x: Q_SWEEP_C @ x + 1e12, a, p, 1e-7)


def test_flat_along_sees_curvature_the_gradient_allowance_cannot():
    # f(a)+f(p)-2f(m) = 2c here; at offset 1e9 it is ~7e-6 of rounding.
    a, p = np.zeros(4), np.ones(4)
    f = lambda x: 1e-4 * (x @ x) + Q_SWEEP_C @ x + 1e9
    assert not _flat_along(f, a, p, 1e-7)


def test_flat_along_rejects_nonfinite():
    from pounce._route import _NotConvex

    f = lambda x: np.nan if x[0] > 0.4 else 0.0
    with pytest.raises(_NotConvex):
        _flat_along(f, np.zeros(4), np.ones(4), 1e-7)


# --- routing verdict ---------------------------------------------------------


def test_issue_reproduction_forced_qp_ipm_without_jac():
    f = lambda x: 0.5 * x @ Q @ x + Q_ISSUE_C @ x
    res = minimize(f, np.zeros(4), bounds=BOX, solver_selection="qp-ipm")
    xs, fs = _closed_form(Q_ISSUE_C)
    assert _routed_to(res) == "qp-ipm"
    assert res.fun == pytest.approx(fs, abs=1e-7)
    np.testing.assert_allclose(res.x, xs, atol=1e-6)


@pytest.mark.parametrize("off", [0.0, 1.0, 10.0, 100.0, 1e3, 1e5, 1e7])
@pytest.mark.parametrize("x0", [np.zeros(4), np.ones(4)], ids=["interior", "on-bound"])
def test_qp_without_jac_routes_whatever_the_offset(off, x0):
    f = lambda x: 0.5 * x @ Q @ x + Q_SWEEP_C @ x + off
    res = minimize(f, x0, bounds=BOX, solver_selection="auto")
    xs, fs = _closed_form(Q_SWEEP_C)
    assert _routed_to(res) == "qp-ipm"
    assert res.info["problem_class"] == "convex_qp"
    assert res.fun - off == pytest.approx(fs, abs=1e-6 * max(1.0, off))
    np.testing.assert_allclose(res.x, xs, atol=1e-6)


@pytest.mark.parametrize("off", [0.0, 100.0, 1e3, 1e5, 1e7, 1e9])
@pytest.mark.parametrize("x0", [np.zeros(4), np.ones(4)], ids=["interior", "on-bound"])
def test_lp_without_jac_routes_as_an_lp_whatever_the_offset(off, x0):
    # 1e7 and up need the gate's round-off allowance: the differenced
    # gradient's noise there exceeds the fixed 1e-7 tolerance.
    f = lambda x: Q_SWEEP_C @ x + off
    res = minimize(f, x0, bounds=BOX, solver_selection="auto")
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


def _kind(f, jac=None, box=True):
    lb, ub = (-np.ones(4), np.ones(4)) if box else (None, None)
    e = classify_and_extract(
        fun=f,
        jac=jac,
        hess=None,
        lb=lb,
        ub=ub,
        m=0,
        g_combined=None,
        jac_combined=None,
        cl=None,
        cu=None,
        x0=np.zeros(4),
    )
    return None if e is None else e.kind


@pytest.mark.parametrize("c, off", [(1e-4, 1e9), (1e-2, 1e9), (1e-2, 1e12)], ids=str)
@pytest.mark.parametrize("box", [True, False], ids=["box", "unbounded"])
def test_small_curvature_at_large_offset_is_not_read_as_an_lp(c, off, box):
    # The round-off allowance is wide enough at these offsets to pass the
    # curvature in the gradient; the value check must catch it. (1e-2 at
    # 1e12 reads as an LP without the value check.)
    f = lambda x: c * (x @ x) + Q_SWEEP_C @ x + off
    assert _kind(f, box=box) != "lp"


def test_allowance_is_not_used_when_jac_is_exact():
    # Same curvature/offset an FD gradient cannot resolve: with `jac` the
    # gradient carries no |f| round-off, so the fixed gate sees the QP.
    c, off = 1e-7, 1e7
    f = lambda x: c * (x @ x) + Q_SWEEP_C @ x + off
    assert _kind(f, jac=lambda x: 2 * c * x + Q_SWEEP_C) == "convex_qp"


def test_nonfinite_objective_is_not_routed():
    # NaN compares false against every gate, so a NaN probe must be caught
    # explicitly rather than read as "within tolerance". NaN on half the box
    # is certain to meet some probe or stencil point.
    f = lambda x: 0.5 * x @ Q @ x + (np.nan if x[0] > 0.0 else 0.0)
    with pytest.raises(ValueError, match="not detected as a convex"):
        minimize(f, np.zeros(4), bounds=BOX, solver_selection="qp-ipm")


# --- the constraint half: rows with no `jac` --------------------------------
#
# A callable constraint with no `jac` reached the router through the NLP
# path's finite difference (eps^(1/3) step), whose round-off scales with the
# row's value. A linear row with a large constant term read as nonlinear
# (offset 1e5), and a convex-quadratic one read as a non-constant Hessian
# (offset 1e3). The router now differences those rows itself
# (`_constraint_jac`). Each test below reaches one branch: every row
# differenced, mixed rows, no differenced rows, and the dangerous direction.

from scipy.optimize import LinearConstraint  # noqa: E402

from pounce._minimize import _wrap_constraints  # noqa: E402
from pounce._route import _all_constraint_hessians, _constraint_jac  # noqa: E402

_LIN_C = np.array([0.3, -0.2, 0.5, -1.0])
_OFFSETS = [0.0, 1e3, 1e5, 1e7]


def _ball(off, r2=0.3):
    """x·x ≤ r2 as a scipy ``ineq`` whose value carries a constant ``off``."""
    return lambda x: (off + r2) - (x @ x + off)


def _ball_reference():
    res = minimize(
        lambda x: _LIN_C @ x,
        np.zeros(4),
        jac=lambda x: _LIN_C,
        bounds=BOX,
        constraints=[{"type": "ineq", "fun": _ball(0.0), "jac": lambda x: -2 * x}],
        solver_selection="auto",
    )
    assert _routed_to(res) == "socp"
    return res.fun


@pytest.mark.parametrize("off", _OFFSETS)
def test_linear_constraint_without_jac_routes_whatever_its_offset(off):
    f = lambda x: 0.5 * x @ Q @ x + Q_SWEEP_C @ x
    cons = [{"type": "ineq", "fun": lambda x: (off + 0.5) - (x.sum() + off)}]
    res = minimize(
        f,
        np.zeros(4),
        jac=lambda x: Q @ x + Q_SWEEP_C,
        bounds=BOX,
        constraints=cons,
        solver_selection="auto",
    )
    assert _routed_to(res) == "qp-ipm"
    ref = minimize(
        f,
        np.zeros(4),
        jac=lambda x: Q @ x + Q_SWEEP_C,
        bounds=BOX,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: 0.5 - x.sum(),
                "jac": lambda x: -np.ones(4),
            }
        ],
        solver_selection="auto",
    )
    assert res.fun == pytest.approx(ref.fun, abs=1e-7)


@pytest.mark.parametrize("off", _OFFSETS)
def test_quadratic_constraint_without_jac_routes_whatever_its_offset(off):
    res = minimize(
        lambda x: _LIN_C @ x,
        np.zeros(4),
        jac=lambda x: _LIN_C,
        bounds=BOX,
        constraints=[{"type": "ineq", "fun": _ball(off)}],
        solver_selection="auto",
    )
    assert _routed_to(res) == "socp"
    assert res.fun == pytest.approx(_ball_reference(), abs=1e-6)


@pytest.mark.parametrize("which_has_jac", ["linear", "quadratic"])
def test_mixed_rows_route(which_has_jac):
    # One row differenced by the router, the other read from its analytic jac.
    off = 1e5
    lin = {"type": "ineq", "fun": lambda x: (off + 0.5) - (x.sum() + off)}
    quad = {"type": "ineq", "fun": _ball(off)}
    if which_has_jac == "linear":
        lin["jac"] = lambda x: -np.ones(4)
    else:
        quad["jac"] = lambda x: -2 * x
    res = minimize(
        lambda x: _LIN_C @ x,
        np.zeros(4),
        jac=lambda x: _LIN_C,
        bounds=BOX,
        constraints=[lin, quad],
        solver_selection="auto",
    )
    assert _routed_to(res) == "socp"


def test_fd_rows_marks_only_callable_rows_without_jac():
    cons = [
        LinearConstraint(np.ones((2, 3)), -1.0, 1.0),
        {"type": "ineq", "fun": lambda x: x[:2], "jac": lambda x: np.eye(3)[:2]},
        {"type": "eq", "fun": lambda x: x.sum()},
    ]
    _, _, jac_values, *_ = _wrap_constraints(cons, 3, np.zeros(3))
    np.testing.assert_array_equal(
        jac_values.fd_rows, [False, False, False, False, True]
    )


def test_constraint_jac_is_unchanged_without_fd_rows():
    J = lambda x: np.eye(2)
    assert _constraint_jac(None, J, None, None, None) is J
    assert _constraint_jac(None, J, np.array([False, False]), None, None) is J


def test_exact_rows_keep_their_old_hessian_bit_for_bit():
    # Row 0 analytic, row 1 differenced. Row 0's Hessian must equal what the
    # unwrapped Jacobian gives, the pre-#1169 computation.
    def g(x):
        return np.array([x @ x + 3.0 * x[0], 1e5 - x @ x])

    def jac(x):  # row 1 stands in for the NLP path's FD: deliberately noisy
        return np.vstack([2 * x + np.array([3.0, 0.0]), -2 * x + 1e-3])

    x = np.array([0.3, -0.4])
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    Jw = _constraint_jac(g, jac, np.array([False, True]), lb, ub)
    new = _all_constraint_hessians(Jw, x, 2, 2)
    old = _all_constraint_hessians(jac, x, 2, 2)
    np.testing.assert_array_equal(new[0], old[0])
    np.testing.assert_allclose(new[1], -2 * np.eye(2), rtol=0, atol=1e-8)


_BAD_CONSTRAINTS = {
    # x·x ≥ 0.3: quadratic, but its feasible set is nonconvex.
    "nonconvex quadratic+offset": {
        "type": "ineq",
        "fun": lambda x: (x @ x + 1e3) - (1e3 + 0.3),
    },
    "cubic+offset": {
        "type": "ineq",
        "fun": lambda x: (1e3 + 0.5) - (x[0] ** 3 + x.sum() + 1e3),
    },
    "quadratic equality": {"type": "eq", "fun": lambda x: (1e3 + 0.3) - (x @ x + 1e3)},
    "nan": {"type": "ineq", "fun": lambda x: np.nan if x[0] > 0.0 else 1.0},
}


@pytest.mark.parametrize("name", list(_BAD_CONSTRAINTS))
def test_nonconvex_or_nonlinear_constraint_without_jac_is_not_routed(name):
    res = minimize(
        lambda x: _LIN_C @ x,
        np.zeros(4),
        jac=lambda x: _LIN_C,
        bounds=BOX,
        constraints=[_BAD_CONSTRAINTS[name]],
        solver_selection="auto",
    )
    assert _routed_to(res) not in ("qp-ipm", "lp-ipm", "socp"), name
