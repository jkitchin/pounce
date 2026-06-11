"""Sensitivity analysis API: `Problem.solve_with_sens`.

Reproduces the ParametricTNLP example from upstream sIPOPT
(`ref/Ipopt/contrib/sIPOPT/examples/parametric_cpp/`) in Python and
compares Δx for Δeta = (-0.5, 0) against the same golden values used by
the Rust integration tests.
"""

import numpy as np
import pytest

import pounce


class ParametricNLP:
    """Pythonic mirror of `ParametricTNLP` (5 vars, 4 constraints).

    Variables `x[3]`, `x[4]` are the parameters `eta1`, `eta2`, pinned
    by equality constraints `g[2] = eta1`, `g[3] = eta2`. Caller picks
    the nominal eta by supplying `cl/cu = eta_nominal` for rows 2, 3.
    """

    def objective(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def gradient(self, x):
        return np.array([2 * x[0], 2 * x[1], 2 * x[2], 0.0, 0.0])

    def constraints(self, x):
        x1, x2, x3, eta1, eta2 = x
        return np.array([
            6 * x1 + 3 * x2 + 2 * x3 - eta1,
            eta2 * x1 + x2 - x3 - 1.0,
            eta1,
            eta2,
        ])

    def jacobianstructure(self):
        rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 3], dtype=np.int64)
        cols = np.array([0, 1, 2, 3, 0, 1, 2, 4, 3, 4], dtype=np.int64)
        return rows, cols

    def jacobian(self, x):
        return np.array([
            6.0, 3.0, 2.0, -1.0,
            x[4], 1.0, -1.0, x[0],
            1.0,
            1.0,
        ])

    def hessianstructure(self):
        rows = np.array([0, 1, 2, 4], dtype=np.int64)
        cols = np.array([0, 1, 2, 0], dtype=np.int64)
        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        return np.array([
            2.0 * obj_factor,
            2.0 * obj_factor,
            2.0 * obj_factor,
            lagrange[1],
        ])


def _make(eta1=5.0, eta2=1.0):
    p = pounce.Problem(
        n=5, m=4, problem_obj=ParametricNLP(),
        lb=[0.0, 0.0, 0.0, -1e19, -1e19],
        ub=[1e19, 1e19, 1e19, 1e19, 1e19],
        cl=[0.0, 0.0, eta1, eta2],
        cu=[0.0, 0.0, eta1, eta2],
    )
    p.add_option("tol", 1e-10)
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    return p


# Upstream sIPOPT's reported Δx (without bound checking) for
# Δeta = (-0.5, 0) starting from nominal eta = (5, 1). Same numbers as
# `crates/pounce-sensitivity/tests/parametric_cpp.rs::UPSTREAM_*`.
UPSTREAM_X_NOMINAL = np.array([
    0.6326530575199982,
    0.3877551079680027,
    0.020408165488001078,
    5.0,
    1.0,
])
UPSTREAM_X_PERTURBED = np.array([
    0.5765306011683219,
    0.3775510381306848,
    -0.04591836070099331,
    4.5,
    1.0,
])
UPSTREAM_DX = UPSTREAM_X_PERTURBED - UPSTREAM_X_NOMINAL


def test_solve_with_sens_matches_upstream_dx():
    x0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
    x, info = _make().solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        deltas=[-0.5, 0.0],
    )
    assert info["status_msg"] == "Solve_Succeeded"
    dx = info["dx"]
    assert dx is not None
    # atol=5e-8: pounce's IPM converges to a slightly different
    # floating-point floor (~1e-9) than homebrew sIPOPT 3.14.19 used to
    # capture the golden numbers; the linear step inherits that noise.
    np.testing.assert_allclose(dx, UPSTREAM_DX, atol=5e-8)
    np.testing.assert_allclose(info["dx_full"][:5], UPSTREAM_DX, atol=5e-8)
    # We didn't ask for a reduced Hessian.
    assert info["reduced_hessian"] is None


def test_solve_with_sens_reduced_hessian_is_symmetric():
    x0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
    x, info = _make().solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        compute_reduced_hessian=True,
    )
    assert info["status_msg"] == "Solve_Succeeded"
    hr = info["reduced_hessian"]
    assert hr is not None
    assert hr.shape == (4,), "2x2 reduced Hessian flattened column-major"
    # H_R is symmetric by construction even when not positive definite.
    assert abs(hr[1] - hr[2]) < 1e-8


def test_solve_with_sens_requires_either_deltas_or_reduced_hessian():
    """Calling without specifying any output should raise."""
    with pytest.raises(ValueError, match="deltas"):
        _make().solve_with_sens(
            x0=np.array([0.15, 0.15, 0.0, 0.0, 0.0]),
            pin_constraint_indices=[2, 3],
        )


def test_solve_with_sens_validates_pin_indices():
    with pytest.raises(ValueError, match="out of range"):
        _make().solve_with_sens(
            x0=np.array([0.15, 0.15, 0.0, 0.0, 0.0]),
            pin_constraint_indices=[2, 99],
            deltas=[-0.5, 0.0],
        )


def test_solve_with_sens_rh_eigendecomp_diagonalizes_hr():
    """rh_eigendecomp=True returns ascending eigenvalues + eigenvectors
    that diagonalize the reduced Hessian.
    """
    x0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
    x, info = _make().solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        rh_eigendecomp=True,
    )
    assert info["status_msg"] == "Solve_Succeeded"
    hr = info["reduced_hessian"]
    eigvals = info["reduced_hessian_eigenvalues"]
    eigvecs = info["reduced_hessian_eigenvectors"]
    assert hr is not None and eigvals is not None and eigvecs is not None
    assert eigvals.shape == (2,)
    assert eigvecs.shape == (4,)  # column-major 2x2

    # Ascending order.
    assert eigvals[0] <= eigvals[1] + 1e-12

    # Reshape Hr and V to 2x2 (column-major); verify Hr V = V diag(eigvals).
    Hr = hr.reshape((2, 2), order="F")
    V = eigvecs.reshape((2, 2), order="F")
    np.testing.assert_allclose(Hr @ V, V * eigvals[np.newaxis, :], atol=1e-10)
    # Eigenvectors are orthonormal.
    np.testing.assert_allclose(V.T @ V, np.eye(2), atol=1e-10)


def test_solve_with_sens_boundcheck_clamps_violating_step():
    """sens_boundcheck=True projects dx so x_curr+dx stays in [lb, ub].

    With deltas=[-0.5, 0.0] the unconstrained linear step drives x[2] to
    ~-0.046, below its lower bound of 0. The clamp should zero that
    coordinate exactly while leaving non-violating slots untouched.
    """
    x0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])

    # Reference: unclamped solve, to confirm x[2]+dx[2] < 0.
    _, info_unclamped = _make().solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        deltas=[-0.5, 0.0],
    )
    x_nominal_2 = UPSTREAM_X_NOMINAL[2]
    dx_unclamped_2 = info_unclamped["dx"][2]
    assert x_nominal_2 + dx_unclamped_2 < -1e-6, "precondition: step violates bound"

    # Clamped solve.
    _, info = _make().solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        deltas=[-0.5, 0.0],
        sens_boundcheck=True,
    )
    assert info["status_msg"] == "Solve_Succeeded"
    dx = info["dx"]
    assert dx is not None

    # x[2] is clamped to lb=0 → dx[2] = 0 - x_nominal[2].
    # Use 5e-8 to match the convergence-floor tolerance used elsewhere in
    # this file (pounce's converged x[2] vs the upstream-captured golden).
    assert dx[2] == pytest.approx(-x_nominal_2, abs=5e-8)
    # Non-violating coordinates (x[0], x[1], pinned x[3], x[4]) unchanged.
    np.testing.assert_allclose(dx[0], UPSTREAM_DX[0], atol=5e-8)
    np.testing.assert_allclose(dx[1], UPSTREAM_DX[1], atol=5e-8)
    np.testing.assert_allclose(dx[3], UPSTREAM_DX[3], atol=5e-8)
    np.testing.assert_allclose(dx[4], UPSTREAM_DX[4], atol=5e-8)


class ScaledPinNLP:
    """pounce#128 regression fixture: a badly-scaled 2-variable
    least-squares NLP with a badly-scaled parameter pin.

        min  c0*(x - p)**2 + c1*(x - 1)**2
        s.t. SCALE*p = SCALE*p_hat

    The objective gradient at the start (~2*c1 = 1.2e5) and the pin
    row's Jacobian entry (SCALE = 1e4) both exceed
    nlp_scaling_max_gradient (100), so the default gradient-based NLP
    scaling fires on both the objective and the pin row. Analytically,
    f*(p) = c0*c1*(p-1)**2/(c0+c1), and w.r.t. the pin RHS r = SCALE*p,
    the reduced Hessian in pounce's pin-row sign convention is

        H = -d2f*/dr2 = -2*c0*c1 / ((c0+c1)*SCALE**2)

    Before #128 the returned value was off by exactly df/dc**2 (the
    objective / pin-row scaling factors).
    """

    C0 = 4.0e4
    C1 = 6.0e4
    SCALE = 1.0e4
    P_HAT = 0.7

    def objective(self, x):
        xx, p = x
        return self.C0 * (xx - p) ** 2 + self.C1 * (xx - 1.0) ** 2

    def gradient(self, x):
        xx, p = x
        return np.array([
            2 * self.C0 * (xx - p) + 2 * self.C1 * (xx - 1.0),
            -2 * self.C0 * (xx - p),
        ])

    def constraints(self, x):
        return np.array([self.SCALE * x[1]])

    def jacobianstructure(self):
        return np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)

    def jacobian(self, x):
        return np.array([self.SCALE])

    def hessianstructure(self):
        rows = np.array([0, 1, 1], dtype=np.int64)
        cols = np.array([0, 0, 1], dtype=np.int64)
        return rows, cols

    def hessian(self, x, lagrange, obj_factor):
        return obj_factor * np.array([
            2 * (self.C0 + self.C1),
            -2 * self.C0,
            2 * self.C0,
        ])


H_ANALYTIC_128 = (
    -2.0 * ScaledPinNLP.C0 * ScaledPinNLP.C1
    / ((ScaledPinNLP.C0 + ScaledPinNLP.C1) * ScaledPinNLP.SCALE ** 2)
)


def _make_scaled_pin(nlp_scaling_method=None):
    nlp = ScaledPinNLP()
    rhs = nlp.SCALE * nlp.P_HAT
    p = pounce.Problem(
        n=2, m=1, problem_obj=nlp,
        lb=[-1e19, -1e19], ub=[1e19, 1e19],
        cl=[rhs], cu=[rhs],
    )
    p.add_option("tol", 1e-9)
    p.add_option("print_level", 0)
    p.add_option("sb", "yes")
    if nlp_scaling_method is not None:
        p.add_option("nlp_scaling_method", nlp_scaling_method)
    return p


def test_reduced_hessian_is_unscaled_regardless_of_nlp_scaling_pounce_128():
    """The headline #128 fix: -inv(reduced_hessian) (the covariance
    recipe) must be the same with scaling off and with the default
    gradient-based scaling actively firing."""
    x0 = np.array([0.0, 0.0])
    results = {}
    for method in ("none", None):  # None = pounce default (gradient-based)
        _, info = _make_scaled_pin(method).solve_with_sens(
            x0=x0,
            pin_constraint_indices=[0],
            compute_reduced_hessian=True,
        )
        assert info["status_msg"] == "Solve_Succeeded"
        results[method] = info

    for method, info in results.items():
        h = info["reduced_hessian"][0]
        np.testing.assert_allclose(
            h, H_ANALYTIC_128, rtol=1e-6,
            err_msg=f"nlp_scaling_method={method}: natural-units H wrong",
        )

    # Scaling factors are reported. With scaling off they are trivial...
    info_off = results["none"]
    assert info_off["obj_scaling_factor"] == pytest.approx(1.0)
    np.testing.assert_allclose(info_off["pin_g_scaling"], [1.0])
    np.testing.assert_allclose(
        info_off["reduced_hessian_scaled"], info_off["reduced_hessian"]
    )

    # The clean quadratic fixture needs no inertia correction: the
    # all-zero perturbations certify the covariance reading is exact.
    for info in results.values():
        np.testing.assert_allclose(info["kkt_perturbations"], np.zeros(4))

    # ...and with gradient-based scaling both df and the pin dc fire,
    # and the scaled accessor reconstructs from the reported factors.
    info_on = results[None]
    df = info_on["obj_scaling_factor"]
    dc = info_on["pin_g_scaling"][0]
    assert df < 1.0  # max grad ~1.2e5 >> 100
    assert dc < 1.0  # pin row max = SCALE = 1e4 >> 100
    np.testing.assert_allclose(
        info_on["reduced_hessian_scaled"],
        info_on["reduced_hessian"] * df / dc ** 2,
        rtol=1e-12,
    )
    # The pre-#128 value really was wildly off for this fixture.
    ratio = info_on["reduced_hessian_scaled"][0] / info_on["reduced_hessian"][0]
    assert not ratio == pytest.approx(1.0, rel=0.5)


def test_solver_session_reduced_hessian_scaled_accessor_and_nlp_scaling():
    """Session API mirror: Solver.reduced_hessian is natural-units by
    default, scaled=True returns the solver-space value, and
    Solver.nlp_scaling exposes the factors."""
    problem = _make_scaled_pin()  # default gradient-based scaling
    solver = pounce.Solver(problem)
    _, info = solver.solve(x0=np.array([0.0, 0.0]))
    assert info["status_msg"] == "Solve_Succeeded"

    h = solver.reduced_hessian([0])
    np.testing.assert_allclose(h[0], H_ANALYTIC_128, rtol=1e-6)

    scaling = solver.nlp_scaling
    df = scaling["obj"]
    assert df < 1.0
    assert scaling["c_scale"] is not None and scaling["c_scale"][0] < 1.0
    assert scaling["d_scale"] is None  # no inequalities in this fixture

    h_scaled = solver.reduced_hessian([0], scaled=True)
    dc = scaling["c_scale"][0]
    np.testing.assert_allclose(h_scaled[0], h[0] * df / dc ** 2, rtol=1e-12)

    # Factor-regularization diagnostic mirrors the info-dict key.
    np.testing.assert_allclose(solver.kkt_perturbations, np.zeros(4))


def test_solve_with_sens_finite_difference_cross_check():
    """First-order sensitivity Δx_sens vs Δx from a fresh resolve."""
    x0 = np.array([0.15, 0.15, 0.0, 0.0, 0.0])
    step = 1e-2

    # Nominal solve + sensitivity step.
    x_nom, info = _make(eta1=5.0).solve_with_sens(
        x0=x0,
        pin_constraint_indices=[2, 3],
        deltas=[step, 0.0],
    )
    dx_sens = info["dx"]

    # Finite-difference reference: re-solve at eta1 + step.
    x_pert, _ = _make(eta1=5.0 + step).solve(x0=x0)
    dx_fd = x_pert - x_nom

    # Pinned slot reproduces the perturbation by construction.
    sign = np.sign(dx_sens[3] * dx_fd[3])
    assert sign > 0
    # Non-parameter primals (x[0..3]) match FD to first order.
    np.testing.assert_allclose(sign * dx_sens[:3], dx_fd[:3], atol=1e-6)
