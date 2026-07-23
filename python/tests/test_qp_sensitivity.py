"""Post-optimal QP sensitivity (the sIPOPT analog) — pounce.qp.QpSensitivity.

The parametric step predicts how the optimum moves when an equality
constraint's right-hand side (the "pinned" parameter) changes, reusing one
active-set KKT factorization across queries. Each test cross-checks the
first-order predictor against an exact re-solve of the perturbed QP.
"""

import numpy as np
import pytest

import pounce
from pounce.qp import ActiveSet, QpSensitivity, ReducedHessian, solve_qp


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
        P=np.eye(2),
        c=[0.0, 0.0],
        A=[[1.0, 1.0]],
        b=[1.0],
        G=[[-1.0, 0.0]],
        h=[-1.0],  # −x0 ≤ −1  ⇔  x0 ≥ 1
    )
    np.testing.assert_allclose(s.x, [1.0, 0.0], atol=1e-6)
    dx = s.parametric_step([0], [0.5])
    np.testing.assert_allclose(dx, [0.0, 0.5], atol=1e-6)
    exact = solve_qp(
        P=np.eye(2),
        c=[0.0, 0.0],
        A=[[1.0, 1.0]],
        b=[1.5],
        G=[[-1.0, 0.0]],
        h=[-1.0],
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
        P=np.eye(3),
        c=[0.0, 0.0, 0.0],
        A=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        b=[1.0, 2.0],
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
        P=np.eye(3),
        c=[0.0, 0.0, 0.0],
        A=[[1.0, 1.0, 1.0]],
        b=[1.0],
        lb=[0.9, -10.0, -10.0],
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


# --------------------------------------------------------------------------
# issue M35 — QpSensitivity (like QpFactorization.solve and the NLP Solver)
# must release the GIL for the duration of the pure-Rust IPM solve, so
# concurrent solves on multiple Python threads run in parallel instead of
# being serialized. Pre-fix the solve held the GIL continuously, so N
# threaded solves took as long as N serial ones. (Code review M35.)
# --------------------------------------------------------------------------


def _big_convex_qp(n, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + n * np.eye(n)  # SPD → strictly convex, nontrivial to factor
    c = rng.standard_normal(n)
    return dict(P=P, c=c, lb=-np.ones(n), ub=np.ones(n))


def test_qp_solve_releases_the_gil():
    import os
    import threading
    import time

    if (os.cpu_count() or 1) < 4:
        pytest.skip("need ≥4 cores to observe parallel speedup")

    n, k = 180, 8
    args = [_big_convex_qp(n, s) for s in range(k)]

    def run_all():
        for a in args:
            QpSensitivity(**a)

    def run_all_threaded():
        ths = [threading.Thread(target=lambda a=a: QpSensitivity(**a)) for a in args]
        for t in ths:
            t.start()
        for t in ths:
            t.join()

    # Best-of-2 for each to damp scheduling noise.
    serial = min(_timed(run_all) for _ in range(2))
    threaded = min(_timed(run_all_threaded) for _ in range(2))

    # With the GIL released the k solves overlap across cores; pre-fix they
    # serialize (ratio ≈ 1). A generous 0.75 threshold separates the regimes
    # (measured ≈ 0.4 released vs ≈ 0.97 held) while tolerating a busy CI box.
    assert threaded < 0.75 * serial, (
        f"threaded solves did not overlap (threaded={threaded:.3f}s, "
        f"serial={serial:.3f}s, ratio={threaded / serial:.2f}); the GIL was "
        f"not released during the QP solve"
    )


def _timed(fn):
    import time

    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# --- weak activity / non-strict complementarity (gh #219) -------------------


def _degenerate_qp(h, **kw):
    """min ½‖x‖² s.t. x0 + x1 = 1, x0 − 2x1 ≤ h.

    At h = −½ the equality-only optimum (½, ½) hits the inequality exactly,
    so strict complementarity fails and dx/db is one-sided.
    """
    return QpSensitivity(
        P=np.eye(2),
        c=[0.0, 0.0],
        A=[[1.0, 1.0]],
        b=[1.0],
        G=[[1.0, -2.0]],
        h=[h],
        **kw,
    )


def test_active_set_exports():
    assert pounce.ActiveSet is ActiveSet


def test_active_indices_reports_membership_by_identity():
    # h = −0.9: the inequality binds strictly (multiplier ~8.9e-2).
    s = _degenerate_qp(-0.9)
    assert s.active_indices.inequalities == (0,)
    assert s.active_indices.bounds == ()
    # And the count still agrees with what kkt_dim encodes: n + m_eq + n_active.
    assert s.kkt_dim == 2 + 1 + len(s.active_indices)


def test_inactive_constraint_is_not_in_the_active_set():
    # h = 0.5: slack at the optimum.
    s = _degenerate_qp(0.5)
    assert s.active_indices.inequalities == ()
    assert not s.active_indices


def test_weakly_active_detected_regardless_of_tol():
    # The gh #219 gap. dx/db is two-valued here — (2/3, 1/3) vs (1/2, 1/2),
    # 33% apart — and which branch parametric_step reports depends on `tol`,
    # an unrelated setting. kkt_dim flips 4 -> 3 across the sweep; the weak
    # flag must stay on throughout, since the geometry never changed.
    seen_dims = set()
    for tol in (None, 1e-12, 1e-14):
        s = _degenerate_qp(-0.5, tol=tol)
        assert s.weakly_active_indices.inequalities == (0,), f"missed at tol={tol}"
        seen_dims.add(s.kkt_dim)
    # Guards the premise: if the sweep stopped straddling the active-set
    # boundary this test would pass while demonstrating nothing.
    assert seen_dims == {3, 4}, f"sweep no longer straddles the boundary: {seen_dims}"


@pytest.mark.parametrize("h", [-0.9, 0.5])
def test_strictly_complementary_is_not_flagged_weak(h):
    # False-positive guard: a screen that fired on every active constraint,
    # or on every small multiplier, would pass the test above and be useless.
    s = _degenerate_qp(h)
    assert s.weakly_active_indices.inequalities == ()
    assert not s.weakly_active_indices


def test_weakly_active_matches_the_one_sided_branches():
    # Ground the flag in the behaviour it warns about: the predictor really
    # does disagree with the two sides of a finite difference here.
    s = _degenerate_qp(-0.5, tol=1e-12)
    assert s.weakly_active_indices.inequalities == (0,)
    dx = s.parametric_step([0], [1.0])

    # A finite difference at a degenerate optimum needs care: too small a step
    # and the solve error swamps the perturbation, returning the *average* of
    # the two branches (~(0.583, 0.417)) — an artifact, not a third branch. Use
    # a tight solve tol and a step no smaller than 1e-3.
    delta = 1e-3

    def at(b):
        return solve_qp(
            P=np.eye(2), c=[0.0, 0.0], A=[[1.0, 1.0]], b=[b],
            G=[[1.0, -2.0]], h=[-0.5], tol=1e-12,
        ).x

    fwd = (at(1.0 + delta) - s.x) / delta
    bwd = (s.x - at(1.0 - delta)) / delta
    # The two one-sided derivatives are genuinely different: (½,½) vs (⅔,⅓).
    np.testing.assert_allclose(fwd, [0.5, 0.5], atol=5e-3)
    np.testing.assert_allclose(bwd, [2 / 3, 1 / 3], atol=5e-3)
    # The predictor matches exactly one of them — it is one-sided, and it is
    # not the average of the two.
    matches_fwd = np.allclose(dx, fwd, atol=5e-3)
    matches_bwd = np.allclose(dx, bwd, atol=5e-3)
    assert matches_fwd != matches_bwd, f"dx={dx} fwd={fwd} bwd={bwd}"


# ---- gh #284: near-LICQ conditioning diagnostic + refinement --------------


def _hilbert_qp_data():
    """The gh #284 family: P = D·H6·D (Hilbert, cond ≈ 7e15), badly scaled."""
    d = np.array([1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2])
    H = np.array([[1.0 / (i + j + 1) for j in range(6)] for i in range(6)])
    P = np.outer(d, d) * H
    c = np.array([1.0, -2.0, 3.0, -1.0, 0.5, -0.25])
    return P, c


def _near_parallel_A(eps):
    """Two nearly-parallel equality rows differing by ``eps`` (near-LICQ)."""
    return np.array([[1.0] * 6, [1.0] * 5 + [1.0 + eps]])


def _dxdb_lu_reference(P, A, pin):
    """dx/db = x-block of the true (unregularized) KKT solve with rhs [0; e_pin],
    by plain float64 LU — the reference the issue shows recovers the info."""
    n, m = P.shape[0], A.shape[0]
    K = np.zeros((n + m, n + m))
    K[:n, :n] = P
    K[:n, n:] = A.T
    K[n:, :n] = A
    rhs = np.zeros(n + m)
    rhs[n + pin] = 1.0
    return np.linalg.solve(K, rhs)[:n]


def test_near_licq_sensitivity_is_flagged_ill_conditioned():
    # The heart of gh #284: on a near-LICQ QP, dx/db silently over-damps to a
    # ~100%-wrong value while every existing signal looks ordinary. The new
    # diagnostic must let a caller DETECT that.
    P, c = _hilbert_qp_data()
    s = QpSensitivity(P=P, c=c, A=_near_parallel_A(1e-10), b=[1.0, 1.0])
    # The old signals stay silent.
    assert not s.weakly_active_indices
    # The new diagnostic fires.
    assert s.ill_conditioned
    assert s.kkt_cond_estimate > 1e14
    # And the per-step residual confirms the near-singular step was not solved.
    s.parametric_step([0], [1.0])
    assert s.last_step_residual > 1e-6


def test_well_conditioned_sensitivity_not_flagged_and_accurate():
    # The false-alarm guard: the badly-scaled but full-rank equality case
    # (cond(KKT) ≈ 5e9). The diagnostic must stay quiet and dx/db must match a
    # plain float64 LU reference to ~1e-7.
    P, c = _hilbert_qp_data()
    A = np.array([[1.0] * 6, [1e4, 1.0, 1.0, 1.0, 1.0, 1e-4]])
    s = QpSensitivity(P=P, c=c, A=A, b=[1.0, 2.0])
    assert not s.ill_conditioned
    assert s.kkt_cond_estimate < 1e12
    dx = s.parametric_step([0], [1.0])
    ref = _dxdb_lu_reference(P, A, 0)
    scale = max(1.0, np.abs(ref).max())
    assert np.abs(dx - ref).max() / scale < 1e-7
    assert s.last_step_residual < 1e-8


def test_refinement_recovers_dxdb_where_information_survives():
    # At eps = 1e-6 a single regularized back-solve over-damps dx/db to ~4e-5
    # rel err, but float64 LU recovers it. Internal refinement must close that
    # gap, and the flag must stay quiet (the step IS reliable here).
    P, c = _hilbert_qp_data()
    A = _near_parallel_A(1e-6)
    s = QpSensitivity(P=P, c=c, A=A, b=[1.0, 1.0])
    assert not s.ill_conditioned
    dx = s.parametric_step([0], [1.0])
    ref = _dxdb_lu_reference(P, A, 0)
    scale = np.abs(ref).max()
    # Comfortably better than the ~4e-5 an un-refined solve yields here.
    assert np.abs(dx - ref).max() / scale < 1e-6
    assert s.last_step_residual < 1e-6


def test_diagnostic_attributes_before_any_step():
    # kkt_cond_estimate / ill_conditioned are build-time (no step needed);
    # last_step_residual is None until a step is taken.
    P, c = _hilbert_qp_data()
    s = QpSensitivity(P=P, c=c, A=_near_parallel_A(1e-6), b=[1.0, 1.0])
    assert s.last_step_residual is None
    assert isinstance(s.kkt_cond_estimate, float)
    assert isinstance(s.ill_conditioned, bool)
    s.parametric_step([0], [1.0])
    assert s.last_step_residual is not None
