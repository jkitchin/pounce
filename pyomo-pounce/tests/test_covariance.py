"""Tests for pyomo_pounce covariance: one-solve asymptotic parameter
covariance from the held KKT factorization.

The scaling convention is pinned empirically, not assumed: for a plain
sum-of-squares objective with the fitted parameters FREE in the solve,
the linear-regression fixture has the exact answer
cov = sigma^2 * inv(X^T X), and the parameter block of the inverse KKT
matrix must reproduce it through cov = 2 * sigma_sq * (K^-1)_pp.
"""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import covariance, declare_fitted, declare_residual

N_LIN = 25
SIGMA_LIN = 0.3


def linear_data():
    rng = np.random.default_rng(42)
    x = np.linspace(0.0, 4.0, N_LIN)
    y = 1.5 - 0.7 * x + SIGMA_LIN * rng.standard_normal(N_LIN)
    X = np.column_stack([np.ones(N_LIN), x])
    return x, y, X


def linear_model(x, y, declare=True):
    """y_i = a + b*x_i + eps_i as an estimation NLP: residual variables
    tied by equalities, objective = sum of squared residuals, (a, b)
    free."""
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(x) - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i]) - mm.a
        - mm.b * float(x[i]))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    if declare:
        declare_fitted(m.a)
        declare_fitted(m.b)
        declare_residual(m.r)
    return m


@pytest.fixture(scope="module")
def linear():
    x, y, X = linear_data()
    m = linear_model(x, y)
    pyo.SolverFactory("pounce").solve(m)
    return m, x, y, X


def test_one_solve_estimates_match_least_squares(linear):
    m, x, y, X = linear
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    assert pyo.value(m.a) == pytest.approx(beta[0], rel=1e-8)
    assert pyo.value(m.b) == pytest.approx(beta[1], rel=1e-8)


def test_known_sigma_matches_analytical_covariance(linear):
    m, x, y, X = linear
    cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    cov_true = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_true, rtol=1e-9)


def test_declared_residuals_estimate_sigma(linear):
    m, x, y, X = linear
    cov = covariance(m)                     # zero extra arguments
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    ssr = float(np.sum((y - X @ beta) ** 2))
    assert cov.sigma_sq == pytest.approx(ssr / (N_LIN - 2), rel=1e-9)
    cov_classical = cov.sigma_sq * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_classical, rtol=1e-9)


def test_keyed_access_and_eigen(linear):
    m, _, _, _ = linear
    cov = covariance(m)
    assert cov[m.a, m.b] == pytest.approx(cov[m.b, m.a])
    assert cov[m.a] == pytest.approx(cov.std_err[m.a] ** 2)
    assert abs(cov.correlation[m.a, m.b]) < 1.0
    evals, evecs = cov.eigen()
    rebuilt = evecs @ np.diag(evals) @ evecs.T
    np.testing.assert_allclose(rebuilt, cov.matrix, atol=1e-14)


def test_explicit_form_equals_declared(linear):
    m_decl, x, y, _ = linear
    cov_decl = covariance(m_decl)
    m = linear_model(x, y, declare=False)
    pyo.SolverFactory("pounce").solve(
        m, fitted=[m.a, m.b], residuals=[m.r])
    cov_expl = covariance(m)
    np.testing.assert_allclose(cov_expl.matrix, cov_decl.matrix, rtol=1e-9)


def test_n_data_fallback():
    """The n_data= branch (SSR taken from the objective, no declared
    residuals) must reproduce the classical sigma^2 (X^T X)^-1."""
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)             # fitted, but NO residuals
    pyo.SolverFactory("pounce").solve(m)
    cov = covariance(m, n_data=N_LIN)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    ssr = float(np.sum((y - X @ beta) ** 2))
    assert cov.sigma_sq == pytest.approx(ssr / (N_LIN - 2), rel=1e-9)
    cov_classical = cov.sigma_sq * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_classical, rtol=1e-9)


def test_n_data_ssr_is_the_solve_time_objective():
    """Writing into the model after the solve must not move the
    n_data= covariance: the SSR is the solve-time objective value, not
    an evaluation on the live model (gh #426)."""
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)
    pyo.SolverFactory("pounce").solve(m)
    cov_before = covariance(m, n_data=N_LIN)
    for i in m.I:
        m.r[i].set_value(999.0)          # the receding-horizon write
    m.a.set_value(0.0)
    cov_after = covariance(m, n_data=N_LIN)
    assert cov_after.sigma_sq == cov_before.sigma_sq
    np.testing.assert_allclose(cov_after.matrix, cov_before.matrix,
                               rtol=0, atol=0)


def test_n_data_rejects_an_unevaluated_objective():
    """A session whose solve evaluated no objective reports NaN, and
    n_data= must refuse it rather than hand back a NaN covariance. NaN
    is the sentinel the engine uses (0.0 is an ordinary objective
    value), so the guard tests isfinite, not None."""
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)
    pyo.SolverFactory("pounce").solve(m)
    session = m.__dict__["_pounce_sens"].session
    session.base_obj = float("nan")
    with pytest.raises(RuntimeError, match="no usable objective value"):
        covariance(m, n_data=N_LIN)


def test_n_data_ignored_when_residuals_declared_warns(linear):
    m, x, y, X = linear                     # fixture declares residuals
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, n_data=999)     # bogus count; must be ignored
    assert any("n_data is ignored" in str(wi.message) for wi in w)
    np.testing.assert_allclose(cov.matrix, covariance(m).matrix, rtol=1e-12)


def test_explicit_form_repeated_solve_is_stable():
    """The explicit (call-time) declarations are solve-local: re-solving
    the same model must not accumulate residuals and drift the variance."""
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    sf = pyo.SolverFactory("pounce")
    sf.solve(m, fitted=[m.a, m.b], residuals=[m.r])
    cov1 = covariance(m)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # no spurious mismatch warning
        sf.solve(m, fitted=[m.a, m.b], residuals=[m.r])
    cov2 = covariance(m)
    assert cov2.sigma_sq == pytest.approx(cov1.sigma_sq, rel=1e-12)
    np.testing.assert_allclose(cov2.matrix, cov1.matrix, rtol=1e-12)
    reg = m.__dict__["_pounce_sens"]
    assert reg.residuals == []              # registry left clean


def test_dict_sigma_without_groups_errors(linear):
    m, _, _, _ = linear                     # residuals declared ungrouped
    with pytest.raises(ValueError,
                       match="no named residual groups were declared"):
        covariance(m, sigma_sq={"lo": 0.1, "hi": 0.2})


def test_error_paths():
    x, y, _ = linear_data()
    m2 = linear_model(x, y, declare=False)
    declare_fitted(m2.a)
    declare_fitted(m2.b)             # no residuals declared
    pyo.SolverFactory("pounce").solve(m2)
    with pytest.raises(ValueError, match="noise variance is unknown"):
        covariance(m2)
    with pytest.raises(ValueError, match="must exceed"):
        covariance(m2, n_data=2)
    m3 = linear_model(x, y, declare=False)
    with pytest.raises(RuntimeError, match="no sensitivity session"):
        covariance(m3)


def test_nonlinear_against_fd_hessian():
    """Exponential decay: covariance vs a finite-difference Hessian of
    the reduced objective f*(A, k)."""
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 3.0, 20)
    y = 2.0 * np.exp(-1.3 * t) + 0.05 * rng.standard_normal(20)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 19)
    m.A = pyo.Var(initialize=1.5)
    m.k = pyo.Var(initialize=1.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i])
        - mm.A * pyo.exp(-mm.k * float(t[i])))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_fitted(m.A)
    declare_fitted(m.k)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    cov = covariance(m, sigma_sq=0.05**2)

    A0, k0 = pyo.value(m.A), pyo.value(m.k)

    def f(A, k):
        return float(np.sum((y - A * np.exp(-k * t)) ** 2))

    h = 1e-5
    H = np.zeros((2, 2))
    steps = [(h, 0.0), (0.0, h)]
    for i, (da, dk) in enumerate(steps):
        for j, (da2, dk2) in enumerate(steps):
            H[i, j] = (f(A0 + da + da2, k0 + dk + dk2)
                       - f(A0 + da - da2, k0 + dk - dk2)
                       - f(A0 - da + da2, k0 - dk + dk2)
                       + f(A0 - da - da2, k0 - dk - dk2)) / (4 * h * h)
    cov_fd = 2.0 * 0.05**2 * np.linalg.inv(H)
    np.testing.assert_allclose(cov.matrix, cov_fd, rtol=1e-4)


def test_two_group_sandwich_matches_closed_form():
    """Two response groups with different noise, unweighted fit: the
    sandwich covariance has the closed form
    (X^T X)^-1 (sum_g sigma_g^2 Xg^T Xg) (X^T X)^-1."""
    rng = np.random.default_rng(3)
    n1, n2 = 15, 15
    s1, s2 = 0.1, 0.6
    x1 = np.linspace(0.0, 2.0, n1)
    x2 = np.linspace(2.0, 4.0, n2)
    y1 = 1.0 + 0.5 * x1 + s1 * rng.standard_normal(n1)
    y2 = 1.0 + 0.5 * x2 + s2 * rng.standard_normal(n2)

    m = pyo.ConcreteModel()
    m.I1 = pyo.RangeSet(0, n1 - 1)
    m.I2 = pyo.RangeSet(0, n2 - 1)
    m.a = pyo.Var(initialize=0.0)
    m.b = pyo.Var(initialize=0.0)
    m.r1 = pyo.Var(m.I1, initialize=0.0)
    m.r2 = pyo.Var(m.I2, initialize=0.0)
    m.res1 = pyo.Constraint(
        m.I1, rule=lambda mm, i: mm.r1[i] == float(y1[i]) - mm.a
        - mm.b * float(x1[i]))
    m.res2 = pyo.Constraint(
        m.I2, rule=lambda mm, i: mm.r2[i] == float(y2[i]) - mm.a
        - mm.b * float(x2[i]))
    m.obj = pyo.Objective(
        expr=sum(m.r1[i] ** 2 for i in m.I1)
        + sum(m.r2[i] ** 2 for i in m.I2))
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r1, group="lo")
    declare_residual(m.r2, group="hi")
    pyo.SolverFactory("pounce").solve(m)

    cov = covariance(m, sigma_sq={"lo": s1**2, "hi": s2**2})

    X1 = np.column_stack([np.ones(n1), x1])
    X2 = np.column_stack([np.ones(n2), x2])
    A = np.linalg.inv(X1.T @ X1 + X2.T @ X2)
    # theta_hat = A_sum^-1 X^T y with A_sum = X^T X (stacked), so
    # cov = A_sum^-1 (sum_g sigma_g^2 Xg^T Xg) A_sum^-1.
    closed = A @ (s1**2 * X1.T @ X1 + s2**2 * X2.T @ X2) @ A
    np.testing.assert_allclose(cov.matrix, closed, rtol=1e-8)
    # per-group sigma estimation route also runs and reports both keys
    cov_est = covariance(m)
    assert set(cov_est.sigma_sq.keys()) == {"lo", "hi"}
    # linear model: the Gauss-Newton sandwich is the same closed form
    cov_gn = covariance(m, sigma_sq={"lo": s1**2, "hi": s2**2},
                        hessian="gauss-newton")
    np.testing.assert_allclose(cov_gn.matrix, closed, rtol=1e-8)


def test_gauss_newton_linear_equals_lagrangian(linear):
    """For a linear model the residual curvature is zero, so the two
    information forms must agree exactly."""
    m, _, _, X = linear
    cov_gn = covariance(m, hessian="gauss-newton")
    cov_obs = covariance(m)
    np.testing.assert_allclose(cov_gn.matrix, cov_obs.matrix, rtol=1e-9)


def test_gauss_newton_nonlinear_matches_analytic_jacobian():
    """Exponential decay: Gauss-Newton covariance vs sigma^2 inv(J^T J)
    with the analytic residual Jacobian at the solution."""
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 3.0, 20)
    y = 2.0 * np.exp(-1.3 * t) + 0.05 * rng.standard_normal(20)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, 19)
    m.A = pyo.Var(initialize=1.5)
    m.k = pyo.Var(initialize=1.0)
    m.r = pyo.Var(m.I, initialize=0.0)
    m.res = pyo.Constraint(
        m.I, rule=lambda mm, i: mm.r[i] == float(y[i])
        - mm.A * pyo.exp(-mm.k * float(t[i])))
    m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
    declare_fitted(m.A, m.k)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)

    cov = covariance(m, sigma_sq=0.05**2, hessian="gauss-newton")
    A0, k0 = pyo.value(m.A), pyo.value(m.k)
    # r_i = y_i - A exp(-k t_i): dr/dA = -exp(-k t), dr/dk = A t exp(-k t)
    J = np.column_stack([-np.exp(-k0 * t), A0 * t * np.exp(-k0 * t)])
    cov_true = 0.05**2 * np.linalg.inv(J.T @ J)
    np.testing.assert_allclose(cov.matrix, cov_true, rtol=1e-5)


def test_gauss_newton_error_paths():
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)                 # no residuals declared
    pyo.SolverFactory("pounce").solve(m)
    with pytest.raises(ValueError, match="gauss-newton"):
        covariance(m, n_data=N_LIN, hessian="gauss-newton")
    with pytest.raises(ValueError, match="hessian"):
        covariance(m, n_data=N_LIN, hessian="expected")


def test_bound_active_warns_and_projects():
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    m.a.setlb(2.0)                      # binds: true intercept ~1.44
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m)
    assert any("bound" in str(wi.message) for wi in w)
    # pinned direction: exactly zero variance, correlation defined as 0
    assert cov[m.a] == 0.0
    assert cov.std_err[m.a] == 0.0
    assert cov.correlation[m.a, m.b] == 0.0
    # free direction: covariance conditional on a = 2 is the
    # slope-only regression of (y - 2) on x
    var_b = cov.sigma_sq / float(np.sum(x**2))
    assert cov[m.b] == pytest.approx(var_b, rel=1e-9)
    # Gauss-Newton agrees on this linear model, projection included
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_gn = covariance(m, hessian="gauss-newton")
    np.testing.assert_allclose(cov_gn.matrix, cov.matrix, rtol=1e-9)


def test_residual_objective_mismatch_warns():
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    # regularized objective: no longer the plain SSR of the residuals
    m.obj.deactivate()
    m.obj2 = pyo.Objective(
        expr=sum(m.r[i] ** 2 for i in m.I) + 10.0 * m.b ** 2)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pyo.SolverFactory("pounce").solve(m)
    assert any("plain sum of squares" in str(wi.message) for wi in w)


def test_clone_keeps_declarations():
    x, y, _ = linear_data()
    m = linear_model(x, y)
    c = m.clone()
    pyo.SolverFactory("pounce").solve(c)
    cov = covariance(c)
    assert cov.std_err[c.a] > 0


def test_varargs_declarations_equal_singles():
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)              # varargs form
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    cov_va = covariance(m)

    m2 = linear_model(x, y)                  # single-call declarations
    pyo.SolverFactory("pounce").solve(m2)
    cov_single = covariance(m2)
    np.testing.assert_allclose(cov_va.matrix, cov_single.matrix, rtol=1e-9)


def test_weakly_active_bound_kept_with_true_variance():
    # data shifted so the unconstrained intercept optimum sits EXACTLY
    # on its bound: slack and multiplier vanish together (weakly
    # active). The classifier keeps the parameter and the value
    # correction removes the barrier diagonal, so the variance is the
    # full unconstrained one; the factor alone would report half of it
    # (W = H + Sigma with Sigma = q at weak activity), and the old
    # slack test would have deleted it outright.
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y2 = y + (2.0 - beta[0])
    m = linear_model(x, y2, declare=False)
    m.a.setlb(2.0)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    assert any("weakly active" in str(wi.message) for wi in w)
    cov_true = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_true, rtol=1e-5)


def test_inactive_bound_changes_nothing():
    # a far-away bound classifies inactive: no warning, and the
    # numbers match the boundless model to solver precision (the
    # Sigma subtraction removes an O(mu) drift, never adds one)
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    m.a.setlb(-50.0)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    assert not [x for x in w if "covariance:" in str(x.message)]
    cov_true = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_true, rtol=1e-7)


def test_binding_row_projects_the_combination():
    # a + b <= cap binding: zero variance along (1,1), finite along
    # the difference, correlation -1, and the matrix equals the
    # restricted-least-squares covariance of the equality-constrained
    # fit. Gauss-Newton agrees on this linear model (step-6 parity).
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    cap = float(beta[0] + beta[1]) - 0.5
    m = linear_model(x, y, declare=False)
    m.capcon = pyo.Constraint(expr=m.a + m.b <= cap)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    assert any("pins the fitted combination" in str(wi.message) for wi in w)
    C = cov.matrix
    u = np.array([1.0, 1.0])
    assert abs(u @ C @ u) < 1e-9 * max(1.0, float(np.abs(C).max()))
    assert cov.correlation[m.a, m.b] == pytest.approx(-1.0, abs=1e-6)
    # restricted least squares: C_r = C0 - C0 u (u' C0 u)^-1 u' C0
    C0 = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    Cr = C0 - np.outer(C0 @ u, u @ C0) / float(u @ C0 @ u)
    np.testing.assert_allclose(C, Cr, rtol=1e-5, atol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_gn = covariance(m, sigma_sq=SIGMA_LIN**2,
                            hessian="gauss-newton")
    np.testing.assert_allclose(cov_gn.matrix, C, rtol=1e-6, atol=1e-12)


def test_bound_and_row_spellings_agree():
    # jkitchin/pounce#362 at the matrix level: the same limit spelled
    # as a variable bound and as a constraint row returns identical
    # covariance matrices, not only identical classifications
    x, y, X = linear_data()
    mA = linear_model(x, y, declare=False)
    mA.a.setlb(2.0)
    declare_fitted(mA.a)
    declare_fitted(mA.b)
    declare_residual(mA.r)
    pyo.SolverFactory("pounce").solve(mA)
    mB = linear_model(x, y, declare=False)
    mB.bnd = pyo.Constraint(expr=mB.a >= 2.0)
    declare_fitted(mB.a)
    declare_fitted(mB.b)
    declare_residual(mB.r)
    pyo.SolverFactory("pounce").solve(mB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covA = covariance(mA, sigma_sq=SIGMA_LIN**2)
        covB = covariance(mB, sigma_sq=SIGMA_LIN**2)
    np.testing.assert_allclose(covB.matrix, covA.matrix,
                               rtol=1e-6, atol=1e-12)


def test_weak_row_and_bound_spellings_agree():
    # the weak regime through the row machinery: optimum exactly on the
    # limit spelled as a constraint row. Kept, warned, value-corrected
    # (the kept row's barrier weight is subtracted), and identical to
    # the bound spelling and to the unconstrained analytic covariance.
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y2 = y + (2.0 - beta[0])

    mA = linear_model(x, y2, declare=False)
    mA.a.setlb(2.0)
    declare_fitted(mA.a)
    declare_fitted(mA.b)
    declare_residual(mA.r)
    pyo.SolverFactory("pounce").solve(mA)

    mB = linear_model(x, y2, declare=False)
    mB.lim = pyo.Constraint(expr=mB.a >= 2.0)
    declare_fitted(mB.a)
    declare_fitted(mB.b)
    declare_residual(mB.r)
    pyo.SolverFactory("pounce").solve(mB)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        covA = covariance(mA, sigma_sq=SIGMA_LIN**2)
        covB = covariance(mB, sigma_sq=SIGMA_LIN**2)
    assert sum("weakly active" in str(wi.message) for wi in w) == 2
    cov_true = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(covA.matrix, cov_true, rtol=1e-4)
    np.testing.assert_allclose(covB.matrix, covA.matrix, rtol=1e-4)


def test_objective_scaling_does_not_move_the_correction():
    # data two orders larger, pushing the max gradient past
    # nlp_scaling_max_gradient so gradient-based scaling engages
    # (df != 1). The report's Sigma is scaled-space; unnormalized it
    # would miscorrect the weakly active kept variance by exactly df.
    scale = 400.0
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 4.0, N_LIN)
    y = scale * (1.5 - 0.7 * x) + (scale * SIGMA_LIN) * rng.standard_normal(N_LIN)
    X = np.column_stack([np.ones(N_LIN), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y2 = y + (2.0 - beta[0])
    m = linear_model(x, y2, declare=False)
    m.a.setlb(2.0)
    # gradient-based scaling reads the STARTING point; residuals
    # initialize to 0 there (gradient 2r = 0), so seed them large to
    # push the max objective gradient past nlp_scaling_max_gradient
    for i in m.I:
        m.r[i].set_value(400.0)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    session = m.__dict__["_pounce_sens"].session
    assert abs(float(session.solver.nlp_scaling["obj"]) - 1.0) > 1e-6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, sigma_sq=(scale * SIGMA_LIN) ** 2)
    assert any("weakly active" in str(wi.message) for wi in w)
    cov_true = (scale * SIGMA_LIN) ** 2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov.matrix, cov_true, rtol=1e-4)


def test_mixed_normal_binding_row_warns_not_projects():
    # a + r[0] <= cap: after eliminating r[0] = y0 - a - b*x0 the row
    # actually pins y0 - b*x0, a b-direction; the restricted normal
    # reads e_a and would project the wrong direction. The honest
    # v0.10 behavior: keep unprojected, warn explicitly.
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    k = 12                                   # x[k] != 0, so b survives
    rk = float(y[k] - beta[0] - beta[1] * x[k])
    cap = float(beta[0] + rk) - 0.5          # binds at the optimum
    m = linear_model(x, y, declare=False)
    m.capcon = pyo.Constraint(expr=m.a + m.r[k] <= cap)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cov = covariance(m, sigma_sq=SIGMA_LIN**2)
    assert any("involves non-fitted variables" in str(wi.message)
               for wi in w)
    # unprojected: full rank, no zero direction
    ev = np.linalg.eigvalsh(cov.matrix)
    assert ev[0] > 1e-12 * ev[-1]


def test_classify_ratio_covers_all_branches():
    from pyomo_pounce.sens import _classify_ratio
    assert _classify_ratio(1e-12, 1e-10) == "inactive"
    assert _classify_ratio(1.0, 1e-10) == "weakly_active"
    assert _classify_ratio(1e12, 1e-10) == "strongly_active"
    assert _classify_ratio(1e-4, 1e-10) == "ambiguous"     # gap low
    assert _classify_ratio(1e4, 1e-10) == "ambiguous"      # gap high
    assert _classify_ratio(0.05, 1e-2) == "inactive"       # mu branch
    assert _classify_ratio(1.0, 1e-2) == "ambiguous"
    assert _classify_ratio(50.0, 1e-2) == "strongly_active"


def test_issue_362_row_pinned_variable_projects():
    """The defect of gh #362, its own reproduction: five residuals pull
    A above a cap of 1.0. As a variable bound the old code projected
    (std_err 0.0, warned); as a Constraint it silently returned the
    barrier's leftover variance (7.9e-6 in the report) with no signal.
    Both spellings must project to exactly zero and warn."""
    def build(spelling):
        m = pyo.ConcreteModel()
        m.I = pyo.RangeSet(5)
        if spelling == "bound":
            m.A = pyo.Var(bounds=(None, 1.0), initialize=0.5)
        else:
            m.A = pyo.Var(initialize=0.5)
            m.cap = pyo.Constraint(expr=m.A <= 1.0)
        m.r = pyo.Var(m.I, initialize=0.0)
        m.res = pyo.Constraint(
            m.I, rule=lambda mm, i: mm.r[i] == 3.0 - mm.A)
        m.obj = pyo.Objective(expr=sum(m.r[i] ** 2 for i in m.I))
        declare_fitted(m.A)
        declare_residual(m.r)
        return m

    for spelling in ("bound", "row"):
        m = build(spelling)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pyo.SolverFactory("pounce").solve(m)
            cov = covariance(m, sigma_sq=0.05**2)
        assert cov.std_err[m.A] == 0.0, spelling
        assert any("strongly active" in str(x.message) for x in w), spelling


def test_explicit_bound_relax_refuses_covariance():
    """An explicit user bound_relax_factor wins over the sens solve's
    forced 0 (options land after defaults), and covariance() then
    refuses with the classifier's clean error instead of classifying
    slacks measured against relaxed bounds."""
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    declare_fitted(m.a, m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(
        m, options={"bound_relax_factor": 1e-8})
    with pytest.raises(ValueError, match="bound_relax_factor"):
        covariance(m, sigma_sq=SIGMA_LIN**2)


def test_weakly_active_bound_gauss_newton_matches():
    # the GN branch rebuilds from the exact recovered Jacobian
    # (Z_r @ inv(M) = J, the W-based factor cancels identically), so a
    # weakly active kept parameter needs no Sigma correction there and
    # must match the same unconstrained analytic covariance the
    # Lagrangian branch reaches via the correction
    x, y, X = linear_data()
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y2 = y + (2.0 - beta[0])
    m = linear_model(x, y2, declare=False)
    m.a.setlb(2.0)
    declare_fitted(m.a)
    declare_fitted(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_gn = covariance(m, sigma_sq=SIGMA_LIN**2,
                            hessian="gauss-newton")
    cov_true = SIGMA_LIN**2 * np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov_gn.matrix, cov_true, rtol=1e-4)


def test_inactive_row_spelling_agrees_to_o_mu():
    # inactive rows are skipped by design (their geometric weight is
    # O(mu), and fetching every normal costs an O(m*n) sweep on wide
    # models), so the two spellings of an INACTIVE limit agree to
    # O(mu) rather than exactly
    x, y, X = linear_data()
    mA = linear_model(x, y, declare=False)
    mA.a.setlb(-50.0)
    declare_fitted(mA.a)
    declare_fitted(mA.b)
    declare_residual(mA.r)
    pyo.SolverFactory("pounce").solve(mA)
    mB = linear_model(x, y, declare=False)
    mB.far = pyo.Constraint(expr=mB.a >= -50.0)
    declare_fitted(mB.a)
    declare_fitted(mB.b)
    declare_residual(mB.r)
    pyo.SolverFactory("pounce").solve(mB)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covA = covariance(mA, sigma_sq=SIGMA_LIN**2)
        covB = covariance(mB, sigma_sq=SIGMA_LIN**2)
    np.testing.assert_allclose(covB.matrix, covA.matrix, rtol=1e-7)



class _ScalarStudy:
    """min ½x² − p·x, the activity-classification study model.

    Moving the unconstrained minimizer `p` walks the three regimes the
    Rust and Python rules share: p > 0 puts it strictly inside the
    limit x ≥ 0 (inactive), p < 0 outside (strongly active), p = 0
    exactly on it (weakly active). Curvature is 1 by construction, so
    `q` never sinks under the identification floor and the Rust
    classifier returns a real status instead of `unidentified` --
    unlike a fitted parameter in the residual-variable idiom, whose
    raw Lagrangian diagonal is zero. `row` spells the same limit as a
    constraint row (x unbounded, row x ≥ 0) instead of a bound.
    """

    def __init__(self, p, row):
        self.p, self.row = p, row

    def objective(self, x):
        return 0.5 * x[0] ** 2 - self.p * x[0]

    def gradient(self, x):
        return np.array([x[0] - self.p])

    def constraints(self, x):
        return np.array([x[0]]) if self.row else np.array([])

    def jacobianstructure(self):
        if not self.row:
            e = np.array([], dtype=np.int64)
            return e, e
        return np.array([0]), np.array([0])

    def jacobian(self, x):
        return np.array([1.0]) if self.row else np.array([])

    def hessianstructure(self):
        z = np.array([0], dtype=np.int64)
        return z, z

    def hessian(self, x, lagrange, obj_factor):
        return np.array([obj_factor])


def _study_report(p, row):
    import pounce

    prob = pounce.Problem(
        n=1, m=1 if row else 0, problem_obj=_ScalarStudy(p, row),
        lb=[-1e19] if row else [0.0], ub=[1e19],
        cl=[0.0] if row else [], cu=[1e19] if row else [],
    )
    for k, v in (("tol", 1e-10), ("bound_relax_factor", 0.0),
                 ("print_level", 0), ("sb", "yes")):
        prob.add_option(k, v)
    solver = pounce.Solver(prob)
    _, info = solver.solve(x0=np.array([0.5]))
    assert info["status_msg"] == "Solve_Succeeded", (p, row, info)
    return solver.classify_activity()


def test_classify_ratio_agrees_with_the_rust_classifier():
    """Drift guard: `_classify_ratio` re-implements
    pounce_sensitivity::activity's rule in Python, and today nothing
    fails if only one of the two moves. Both are branch-tested in
    isolation (`test_classify_ratio_covers_all_branches` here, the
    `classify` tests in activity.rs), but neither pins them to EACH
    OTHER. This drives real solves through the Rust classifier and
    requires every classified entry to re-derive its own status from
    the report's own `(ratio, mu)` through the Python rule.

    `unidentified` is the one deliberate divergence and is exempt:
    Rust maps `q` below the identification floor there unconditionally
    and reports `Σ/floor` as a lower bound rather than the ratio, so
    the Python rule -- which only ever sees a ratio -- neither can nor
    should reproduce it.
    """
    from pyomo_pounce.sens import _classify_ratio

    seen = set()
    for p, expected in ((1.0, "inactive"), (0.0, "weakly_active"),
                        (-1.0, "strongly_active")):
        for row in (False, True):
            rep = _study_report(p, row)
            mu = float(rep["mu"])
            key = "row" if row else "var"
            st = rep[f"{key}_status"][0]
            r = float(rep[f"{key}_ratio"][0])
            assert st == expected, (p, row, st)      # fixture still valid
            assert np.isfinite(r)
            assert _classify_ratio(r, mu) == st, (
                f"{key} spelling, p={p}: Rust says {st}, the Python rule "
                f"says {_classify_ratio(r, mu)} for r={r:.6g}, mu={mu:.6g}")
            seen.add(st)

    assert seen == {"inactive", "weakly_active", "strongly_active"}, seen
