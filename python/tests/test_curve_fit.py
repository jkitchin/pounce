"""Tests for ``pounce.curve_fit``.

Covers parity with scipy / closed-form statistics, the covariance routes
(pounce-native reduced Hessian vs Jacobian), robust losses, parameter
constraints, the data-sensitivity (``dpopt/ddata``) feature, and the result
object's UX. Models are written with ``jax.numpy`` where the exact-derivative
(reduced-Hessian) path is being exercised.
"""

import numpy as np
import pytest

import pounce

scipy_optimize = pytest.importorskip("scipy.optimize")
jnp = pytest.importorskip("jax.numpy")


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def line(x, a, b):
    return a * x + b


def line_j(x, a, b):
    return a * x + b


def expdecay(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def expdecay_np(x, a, b, c):
    return a * np.exp(-b * x) + c


# --------------------------------------------------------------------------
# 1. OLS calibration: pins the reduced-Hessian -> covariance constant.
# --------------------------------------------------------------------------

def test_ols_matches_closed_form_and_scipy():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 25)
    y = 2.0 * x - 1.0 + rng.normal(0, 0.5, x.size)

    # closed-form OLS
    X = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    s2 = resid @ resid / (x.size - 2)
    cov_ols = s2 * np.linalg.inv(X.T @ X)

    sp_popt, sp_pcov = scipy_optimize.curve_fit(line, x, y, p0=[1, 0])

    r = pounce.curve_fit(line_j, x, y, p0=[1, 0])
    assert r.success
    assert r.cov_source == "reduced_hessian"  # pounce-native factor route
    np.testing.assert_allclose(r.popt, beta, rtol=1e-6)
    np.testing.assert_allclose(r.popt, sp_popt, rtol=1e-6)
    np.testing.assert_allclose(r.pcov, cov_ols, rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(r.pcov, sp_pcov, rtol=1e-5, atol=1e-10)


# --------------------------------------------------------------------------
# 2. Nonlinear parity with scipy, weighted, absolute_sigma both ways.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_nonlinear_parity_with_scipy(absolute_sigma):
    rng = np.random.default_rng(1)
    x = np.linspace(0.2, 4, 30)
    sigma = 0.03 + 0.01 * x
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, sigma)

    sp_popt, sp_pcov = scipy_optimize.curve_fit(
        expdecay_np, x, y, p0=[1, 1, 0], sigma=sigma, absolute_sigma=absolute_sigma
    )
    r = pounce.curve_fit(
        expdecay, x, y, p0=[1, 1, 0], sigma=sigma, absolute_sigma=absolute_sigma
    )
    np.testing.assert_allclose(r.popt, sp_popt, rtol=1e-5)
    np.testing.assert_allclose(r.perr, np.sqrt(np.diag(sp_pcov)), rtol=1e-4)
    np.testing.assert_allclose(r.pcov, sp_pcov, rtol=1e-4, atol=1e-10)


# --------------------------------------------------------------------------
# 3. reduced-Hessian covariance == Jacobian covariance (cross-check).
# --------------------------------------------------------------------------

def test_reduced_hessian_matches_jacobian_covariance():
    rng = np.random.default_rng(2)
    x = np.linspace(0.2, 4, 30)
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, 0.05, x.size)

    r_rh = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0])           # jax -> factor
    r_fd = pounce.curve_fit(expdecay_np, x, y, p0=[1, 1, 0], jac="fd")
    assert r_rh.cov_source == "reduced_hessian"
    assert r_fd.cov_source == "jacobian"
    np.testing.assert_allclose(r_rh.pcov, r_fd.pcov, rtol=1e-3, atol=1e-10)


# --------------------------------------------------------------------------
# 4. Robust losses resist outliers; covariance is the sandwich estimator.
# --------------------------------------------------------------------------

def test_robust_loss_resists_outliers():
    rng = np.random.default_rng(5)
    x = np.linspace(0, 10, 50)
    y = line(x, 2.0, -1.0) + rng.normal(0, 0.3, x.size)
    y[[7, 23, 41]] += np.array([9.0, -8.0, 10.0])  # outliers
    truth = np.array([2.0, -1.0])

    r_sse = pounce.curve_fit(line_j, x, y, p0=[1, 0], loss="sse")
    r_hub = pounce.curve_fit(line_j, x, y, p0=[1, 0], loss="huber", f_scale=1.0)
    assert r_hub.cov_source == "sandwich"
    assert np.linalg.norm(r_hub.popt - truth) < np.linalg.norm(r_sse.popt - truth)
    assert np.all(r_hub.perr > 0)


# --------------------------------------------------------------------------
# 5. Data sensitivity dpopt/ddata.
# --------------------------------------------------------------------------

def test_sensitivity_matches_pinv_and_finite_difference():
    rng = np.random.default_rng(7)
    x = np.linspace(0.2, 4, 20)
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, 0.05, x.size)

    r = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], sensitivity=True)
    assert r.dpopt_ddata is not None
    assert r.dpopt_ddata.shape == (3, x.size)

    # analytic identity: dpopt/dy = pinv(J)
    J = np.column_stack(
        [np.exp(-r.popt[1] * x), -r.popt[0] * x * np.exp(-r.popt[1] * x), np.ones_like(x)]
    )
    np.testing.assert_allclose(r.dpopt_ddata, np.linalg.pinv(J), rtol=1e-4, atol=1e-6)

    # finite-difference re-fit at a few points
    eps = 1e-4
    for i in (0, 7, 15):
        yp = y.copy(); yp[i] += eps
        ym = y.copy(); ym[i] -= eps
        rp = pounce.curve_fit(expdecay, x, yp, p0=[1, 1, 0])
        rm = pounce.curve_fit(expdecay, x, ym, p0=[1, 1, 0])
        fd = (rp.popt - rm.popt) / (2 * eps)
        # dpopt_ddata is the first-order (Gauss-Newton) influence; the FD
        # re-fit additionally feels residual curvature, so allow a few %.
        np.testing.assert_allclose(r.dpopt_ddata[:, i], fd, rtol=8e-2, atol=2e-2)


# --------------------------------------------------------------------------
# 6. Parameter constraints: positivity, range, general relation.
# --------------------------------------------------------------------------

def test_positivity_bound_active():
    rng = np.random.default_rng(9)
    x = np.linspace(0, 10, 40)
    y = line(x, 2.0, -3.0) + rng.normal(0, 0.5, x.size)  # true b < 0

    r = pounce.curve_fit(line_j, x, y, p0=[1, 0.1], bounds=[(None, None), (0, np.inf)])
    assert r.popt[1] >= -1e-7
    assert r.active_mask[1] and not r.active_mask[0]
    assert "projected" in r.cov_source
    # the active parameter carries ~no variance
    assert r.pcov[1, 1] < 1e-10


def test_range_bound_respected():
    rng = np.random.default_rng(10)
    x = np.linspace(0, 10, 40)
    y = line(x, 2.0, -1.0) + rng.normal(0, 0.5, x.size)
    r = pounce.curve_fit(line_j, x, y, p0=[1.5, 0], bounds=[(1.0, 1.8), (None, None)])
    assert 1.0 - 1e-7 <= r.popt[0] <= 1.8 + 1e-7


def test_general_parameter_constraint():
    rng = np.random.default_rng(11)
    x = np.linspace(0, 10, 40)
    y = line(x, 2.0, -1.0) + rng.normal(0, 0.5, x.size)
    # require a + b <= 0  ->  ineq g(p) = -(a+b) >= 0
    cons = [{"type": "ineq", "fun": lambda p: -(p[0] + p[1])}]
    r = pounce.curve_fit(line_j, x, y, p0=[0.1, -0.2], constraints=cons)
    assert r.popt[0] + r.popt[1] <= 1e-6


# --------------------------------------------------------------------------
# 7. Result object UX.
# --------------------------------------------------------------------------

def test_result_ux():
    rng = np.random.default_rng(13)
    x = np.linspace(0.2, 4, 30)
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, 0.05, x.size)
    r = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], alpha=0.05)

    assert r.param_names == ["a", "b", "c"]
    # predict
    yhat = r.predict(x)
    np.testing.assert_allclose(yhat, expdecay_np(x, *r.popt), rtol=1e-6)
    # confidence band brackets the prediction
    yb, lo, hi = r.confidence_band(x)
    assert np.all(hi >= yb) and np.all(yb >= lo)
    # prediction band is strictly wider (adds observation noise)
    _, plo, phi = r.confidence_band(x, kind="prediction")
    assert np.all(phi - plo > hi - lo)
    with pytest.raises(ValueError, match="confidence.*prediction|kind"):
        r.confidence_band(x, kind="bogus")
    # CI consistent with perr and t-quantile
    assert r.ci.shape == (3, 2)
    assert np.all(r.ci[:, 0] < r.popt) and np.all(r.popt < r.ci[:, 1])
    # correlation diagonal is 1
    np.testing.assert_allclose(np.diag(r.correlation), np.ones(3), atol=1e-8)
    # dict access + summary
    assert r["success"] is True
    assert "curve_fit summary" in r.summary()
    # goodness of fit
    assert 0.9 < r.r_squared <= 1.0
    assert r.dof == x.size - 3


# --------------------------------------------------------------------------
# 8. Derivative resolution: FD fallback warns; jac='jax' validates.
# --------------------------------------------------------------------------

def test_fd_fallback_warns_on_non_traceable_model():
    rng = np.random.default_rng(15)
    x = np.linspace(0.2, 4, 30)
    # np.exp on a JAX tracer is not traceable -> FD fallback with a warning
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, 0.05, x.size)
    with pytest.warns(UserWarning, match="finite-difference"):
        r = pounce.curve_fit(expdecay_np, x, y, p0=[1, 1, 0])
    assert r.success
    assert r.cov_source == "jacobian"


def test_jac_jax_raises_on_non_traceable_model():
    x = np.linspace(0.2, 4, 30)
    y = expdecay_np(x, 3.0, 0.8, 0.5)
    with pytest.raises(ValueError, match="not JAX-traceable"):
        pounce.curve_fit(expdecay_np, x, y, p0=[1, 1, 0], jac="jax")


def test_analytic_jac_is_exact_path():
    rng = np.random.default_rng(17)
    x = np.linspace(0, 10, 40)
    y = line(x, 2.0, -1.0) + rng.normal(0, 0.5, x.size)

    def jac(x, a, b):
        return np.column_stack([x, np.ones_like(x)])

    r = pounce.curve_fit(line, x, y, p0=[1, 0], jac=jac)
    assert r.cov_source == "reduced_hessian"
    sp_popt, sp_pcov = scipy_optimize.curve_fit(line, x, y, p0=[1, 0])
    np.testing.assert_allclose(r.pcov, sp_pcov, rtol=1e-5, atol=1e-10)


def test_p0_inferred_from_signature():
    rng = np.random.default_rng(19)
    x = np.linspace(0.2, 4, 30)
    y = expdecay_np(x, 3.0, 0.8, 0.5) + rng.normal(0, 0.05, x.size)
    r = pounce.curve_fit(expdecay, x, y)  # no p0 -> inferred (a, b, c) = ones
    assert r.n_params == 3
    assert r.success
