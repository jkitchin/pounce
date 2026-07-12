"""Tests for pyomo_pounce covariance: one-solve asymptotic parameter
covariance from the held KKT factorization.

The scaling convention is pinned empirically, not assumed: for a plain
sum-of-squares objective with the estimated parameters FREE in the solve,
the linear-regression fixture has the exact answer
cov = sigma^2 * inv(X^T X), and the parameter block of the inverse KKT
matrix must reproduce it through cov = 2 * sigma_sq * (K^-1)_pp.
"""
import warnings

import numpy as np
import pytest
import pyomo.environ as pyo

import pyomo_pounce  # noqa: F401  (registers 'pounce')
from pyomo_pounce import covariance, declare_estimated, declare_residual

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
        declare_estimated(m.a)
        declare_estimated(m.b)
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
        m, estimated=[m.a, m.b], residuals=[m.r])
    cov_expl = covariance(m)
    np.testing.assert_allclose(cov_expl.matrix, cov_decl.matrix, rtol=1e-9)


def test_n_data_fallback(linear):
    m, x, y, X = linear
    cov = covariance(m, n_data=N_LIN)
    cov_res = covariance(m)
    np.testing.assert_allclose(cov.matrix, cov_res.matrix, rtol=1e-9)


def test_error_paths():
    x, y, _ = linear_data()
    m2 = linear_model(x, y, declare=False)
    declare_estimated(m2.a)
    declare_estimated(m2.b)             # no residuals declared
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
    declare_estimated(m.A)
    declare_estimated(m.k)
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
    declare_estimated(m.a)
    declare_estimated(m.b)
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


def test_bound_active_warns():
    x, y, X = linear_data()
    m = linear_model(x, y, declare=False)
    m.a.setlb(2.0)                      # binds: true intercept ~1.44
    declare_estimated(m.a)
    declare_estimated(m.b)
    declare_residual(m.r)
    pyo.SolverFactory("pounce").solve(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        covariance(m)
    assert any("bound" in str(wi.message) for wi in w)


def test_residual_objective_mismatch_warns():
    x, y, _ = linear_data()
    m = linear_model(x, y, declare=False)
    # regularized objective: no longer the plain SSR of the residuals
    m.obj.deactivate()
    m.obj2 = pyo.Objective(
        expr=sum(m.r[i] ** 2 for i in m.I) + 10.0 * m.b ** 2)
    declare_estimated(m.a)
    declare_estimated(m.b)
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
