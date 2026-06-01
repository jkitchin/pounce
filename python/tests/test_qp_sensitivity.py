"""Post-optimal QP sensitivity (the sIPOPT analog) — pounce.qp.QpSensitivity.

The parametric step predicts how the optimum moves when an equality
constraint's right-hand side (the "pinned" parameter) changes, reusing one
active-set KKT factorization across queries. Each test cross-checks the
first-order predictor against an exact re-solve of the perturbed QP.
"""

import numpy as np
import pytest

import pounce
from pounce.qp import QpSensitivity, ReducedHessian, solve_qp


def test_top_level_export():
    assert pounce.QpSensitivity is QpSensitivity


def test_equality_rhs_matches_closed_form_and_resolve():
    # min ½‖x‖²  s.t.  x0 + x1 = b   → x* = (b/2, b/2), dx/db = (½, ½).
    s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0])
    np.testing.assert_allclose(s.x, [1.0, 1.0], atol=1e-7)
    dx = s.parametric_step([0], [1.0])
    np.testing.assert_allclose(dx, [0.5, 0.5], atol=1e-6)
    # Predictor lands on the exact re-solve at b = 3.
    exact = solve_qp(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[3.0])
    np.testing.assert_allclose(s.x + dx, exact.x, atol=1e-6)


def test_step_with_active_inequality():
    # min ½‖x‖²  s.t.  x0 + x1 = 1,  x0 ≥ 1.  The bound binds: x* = (1, 0).
    # Perturbing b slides along the active face: x = (1, b−1), dx/db = (0, 1).
    s = QpSensitivity(
        P=np.eye(2), c=[0.0, 0.0],
        A=[[1.0, 1.0]], b=[1.0],
        G=[[-1.0, 0.0]], h=[-1.0],  # −x0 ≤ −1  ⇔  x0 ≥ 1
    )
    np.testing.assert_allclose(s.x, [1.0, 0.0], atol=1e-6)
    dx = s.parametric_step([0], [0.5])
    np.testing.assert_allclose(dx, [0.0, 0.5], atol=1e-6)
    exact = solve_qp(
        P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[1.5],
        G=[[-1.0, 0.0]], h=[-1.0],
    )
    np.testing.assert_allclose(s.x + dx, exact.x, atol=1e-6)


def test_step_with_active_variable_bound():
    # min ½‖x‖²  s.t.  x0 + x1 = 1,  x0 ≥ 0.6 via a variable bound.
    # x* = (0.6, 0.4); perturbing b moves x1: dx/db = (0, 1).
    s = QpSensitivity(
        P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[1.0], lb=[0.6, -10.0]
    )
    np.testing.assert_allclose(s.x, [0.6, 0.4], atol=1e-6)
    dx = s.parametric_step([0], [0.2])
    np.testing.assert_allclose(dx, [0.0, 0.2], atol=1e-6)


def test_multiple_pins_and_factor_reuse():
    # Two equality constraints, both pinned; and repeated queries reuse the
    # factorization (build-once / solve-many).
    # min ½‖x‖²  s.t.  x0 = b0,  x1 = b1   → x* = (b0, b1), dx = Δb.
    s = QpSensitivity(
        P=np.eye(3), c=[0.0, 0.0, 0.0],
        A=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], b=[1.0, 2.0],
    )
    np.testing.assert_allclose(s.x[:2], [1.0, 2.0], atol=1e-6)
    d1 = s.parametric_step([0, 1], [0.3, -0.5])
    np.testing.assert_allclose(d1, [0.3, -0.5, 0.0], atol=1e-6)
    # A second, different query against the same cached factor.
    d2 = s.parametric_step([1], [1.0])
    np.testing.assert_allclose(d2, [0.0, 1.0, 0.0], atol=1e-6)


def test_unbounded_qp_raises():
    with pytest.raises(ValueError):
        QpSensitivity(c=[-1.0], G=[[-1.0]], h=[0.0])  # min −x, x ≥ 0


def test_mismatched_pin_and_delta_lengths_raise():
    s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0])
    with pytest.raises(ValueError):
        s.parametric_step([0], [1.0, 2.0])


def test_pin_index_out_of_range_raises():
    s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[2.0])
    with pytest.raises(ValueError):
        s.parametric_step([5], [1.0])  # only 1 equality constraint


def test_top_level_reduced_hessian_export():
    assert pounce.ReducedHessian is ReducedHessian


def test_reduced_hessian_unconstrained_equals_P():
    # No active constraints: the null space is all of ℝⁿ, so H_R = P and its
    # eigenvalues are P's diagonal {2, 3}.
    s = QpSensitivity(P=np.diag([2.0, 3.0]), c=[0.0, 0.0])
    rh = s.reduced_hessian()
    assert isinstance(rh, ReducedHessian)
    assert rh.n_dof == 2
    np.testing.assert_allclose(rh.eigenvalues, [2.0, 3.0], atol=1e-9)
    assert rh.is_positive_definite


def test_reduced_hessian_hand_value():
    # P = [[3,1],[1,2]], x0 + x1 = 0 ⇒ Z = (1,−1)/√2, zᵀPz = 3/2.
    s = QpSensitivity(P=[[3.0, 1.0], [1.0, 2.0]], c=[0.0, 0.0], A=[[1.0, 1.0]], b=[0.0])
    rh = s.reduced_hessian()
    assert rh.n_dof == 1
    np.testing.assert_allclose(rh.eigenvalues, [1.5], atol=1e-9)
    np.testing.assert_allclose(rh.matrix, [[1.5]], atol=1e-9)


def test_reduced_hessian_matches_numpy_nullspace():
    # Cross-check the eigenvalues against an independent null-space
    # projection computed with numpy (eigenvalues are basis-invariant).
    P = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    A = np.array([[1.0, 1.0, 1.0]])
    s = QpSensitivity(P=P, c=[0.0, 0.0, 0.0], A=A, b=[1.0])
    rh = s.reduced_hessian()
    assert rh.n_dof == 2

    # Orthonormal null-space basis of A from the SVD (rank(A) = 1).
    _, _, vt = np.linalg.svd(A)
    Z = vt[1:].T  # (3, 2), orthonormal columns spanning null(A)
    expected = np.linalg.eigvalsh(Z.T @ P @ Z)  # ascending
    np.testing.assert_allclose(rh.eigenvalues, expected, atol=1e-7)

    # H_R should reconstruct from its own eigendecomposition.
    recon = rh.eigenvectors @ np.diag(rh.eigenvalues) @ rh.eigenvectors.T
    np.testing.assert_allclose(recon, rh.matrix, atol=1e-9)


def test_reduced_hessian_full_rank_active_set_has_zero_dof():
    # Two independent active constraints in 2 variables pin the point
    # completely: zero degrees of freedom, so the reduced Hessian is 0×0.
    s = QpSensitivity(
        P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[1.0], lb=[0.6, -10.0]
    )
    rh = s.reduced_hessian()
    assert rh.n_dof == 0
    assert rh.matrix.shape == (0, 0)
    assert rh.is_positive_definite  # vacuously true


def test_reduced_hessian_with_active_bound():
    # min ½‖x‖² s.t. x0+x1+x2 = 1, x0 ≥ 0.9. The bound binds (x0 = 0.9),
    # leaving 1 DOF in the (x1, x2) plane along (0, 1, −1)/√2: H_R = 1.
    s = QpSensitivity(
        P=np.eye(3), c=[0.0, 0.0, 0.0],
        A=[[1.0, 1.0, 1.0]], b=[1.0], lb=[0.9, -10.0, -10.0],
    )
    np.testing.assert_allclose(s.x, [0.9, 0.05, 0.05], atol=1e-6)
    rh = s.reduced_hessian()
    assert rh.n_dof == 1
    np.testing.assert_allclose(rh.eigenvalues, [1.0], atol=1e-7)


def test_finite_difference_agreement():
    # The analytic step agrees with a central finite difference of the
    # re-solve, on a non-trivial QP with an active inequality.
    P = np.array([[2.0, 0.5], [0.5, 1.0]])
    A = [[1.0, 2.0]]
    G = [[1.0, 0.0]]
    base = dict(P=P, c=[-1.0, 0.5], A=A, b=[1.0], G=G, h=[0.4])
    s = QpSensitivity(**base)
    dx = s.parametric_step([0], [1.0])  # d x / d b0

    eps = 1e-5
    xp = solve_qp(**{**base, "b": [1.0 + eps]}).x
    xm = solve_qp(**{**base, "b": [1.0 - eps]}).x
    fd = (xp - xm) / (2 * eps)
    np.testing.assert_allclose(dx, fd, atol=1e-5)
