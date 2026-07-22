"""Tests for ``pounce.curve_fit``.

Covers parity with scipy / closed-form statistics, the covariance routes
(pounce-native reduced Hessian vs Jacobian), robust losses, parameter
constraints, the data-sensitivity (``dpopt/ddata``) feature, and the result
object's UX. Models are written with ``jax.numpy`` where the exact-derivative
(reduced-Hessian) path is being exercised.
"""

import warnings

import numpy as np
import pytest

import pounce
from pounce._curve_fit import _normalize_bound_arg

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


def expdecay_2p(x, a, b):
    return a * np.exp(-b * x)


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


def test_active_equality_constraint_projects_covariance():
    """An ACTIVE general equality between parameters must be projected out of
    ``pcov``: the constrained combination carries ~zero variance and the
    parameters pick up an induced anti-correlation -- neither holds for the
    unconstrained covariance. Pre-fix the projection ignored general
    constraints entirely and returned the unconstrained covariance while
    mislabeling it ``reduced_hessian(projected)``. (Code review M30.)"""
    rng = np.random.default_rng(31)
    x = np.linspace(0.0, 10.0, 40)
    y = line(x, 2.0, -1.0) + rng.normal(0.0, 0.5, x.size)  # unconstrained a+b != 1
    # require a + b = 1  (active equality g(p) = a + b - 1 = 0)
    cons = [{"type": "eq", "fun": lambda p: p[0] + p[1] - 1.0,
             "jac": lambda p: np.array([[1.0, 1.0]])}]
    r = pounce.curve_fit(line_j, x, y, p0=[1.0, 0.0], constraints=cons)

    assert abs(r.popt[0] + r.popt[1] - 1.0) <= 1e-6
    assert r.cov_source == "reduced_hessian(projected)"
    g = np.array([1.0, 1.0])                      # the active-constraint gradient
    # variance along the constrained direction is ~zero (the binding relation
    # is known exactly), and the parameters are perfectly anti-correlated.
    assert float(g @ r.pcov @ g) < 1e-9
    assert r.pcov[0, 1] < 0.0
    # cross-check against the closed-form projected covariance.
    Jw = np.column_stack([x, np.ones_like(x)])    # d line / d[a, b]
    M = Jw.T @ Jw
    A = np.array([[1.0, 1.0]])
    _, _, Vt = np.linalg.svd(A)
    Z = Vt[1:].T
    pcov_ref = r._s2 * Z @ np.linalg.pinv(Z.T @ M @ Z) @ Z.T
    np.testing.assert_allclose(r.pcov, pcov_ref, rtol=1e-5, atol=1e-10)


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
    r = pounce.curve_fit(expdecay, x, y)  # no p0 -> inferred (a, b, c)
    assert r.n_params == 3
    assert r.success
    np.testing.assert_allclose(r.popt, [3.0, 0.8, 0.5], atol=0.1)


def test_omitted_p0_data_driven_seed_handles_bad_scaling():
    # True parameters are ~1e6: a flat ``ones`` seed is poorly scaled, but the
    # data-driven default anchors candidates on the data magnitude.
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 60)
    y = 2.0e6 * x - 5.0e5 + rng.normal(0, 1e3, x.size)
    r = pounce.curve_fit(line_j, x, y)  # no p0
    # The data-driven seed lets the solve recover the (badly scaled) truth.
    np.testing.assert_allclose(r.popt, [2.0e6, -5.0e5], rtol=1e-3)


def test_omitted_p0_seed_respects_bounds():
    # With finite bounds and no p0, the seed must be inside the box and the fit
    # must still succeed.
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 5.0, 40)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.02, x.size)
    r = pounce.curve_fit(line_j, x, y, bounds=[(0.0, 5.0), (-1.0, 3.0)])
    assert r.success
    assert 0.0 <= r.popt[0] <= 5.0 and -1.0 <= r.popt[1] <= 3.0
    np.testing.assert_allclose(r.popt, [2.0, 1.0], atol=0.1)


def test_initial_guess_never_worse_than_ones():
    # ``ones`` is always one of the scored candidates, so the chosen seed's
    # objective can never exceed it.
    from pounce._curve_fit import _initial_guess, _loss_sse

    rng = np.random.default_rng(11)
    x = np.linspace(0.0, 10.0, 50)
    y = 1.0e6 * x + 2.0e6 + rng.normal(0, 1e2, x.size)

    def model(xx, p):
        return p[0] * xx + p[1]

    n = 2
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    w = np.ones(x.size)
    seed = _initial_guess(model, x, y, w, lo, hi, n, _loss_sse, 1.0)

    def sse(p):
        r = model(x, p) - y
        return float(r @ r)

    assert sse(seed) <= sse(np.ones(n))


# --------------------------------------------------------------------------
# curve_fit_minima: multiple parameter sets via find_minima
# --------------------------------------------------------------------------

def _gauss_np(x, a, mu, sig):
    return a * np.exp(-(x - mu) ** 2 / (2.0 * sig ** 2))


def test_curve_fit_minima_finds_multiple_parameter_sets():
    # A single Gaussian fit to a two-peak signal: with sigma constrained so no
    # one Gaussian can straddle both peaks, the LS surface has two minima --
    # "sit on the left peak" vs "sit on the right peak".
    rng = np.random.default_rng(0)
    x = np.linspace(-10, 10, 200)
    y = _gauss_np(x, 1.0, -4.0, 1.0) + _gauss_np(x, 0.7, 4.0, 1.5)
    y = y + rng.normal(0, 0.01, x.size)
    bounds = [(0.0, 3.0), (-10.0, 10.0), (0.1, 2.5)]

    fits = pounce.curve_fit_minima(
        _gauss_np, x, y, bounds=bounds, jac="fd",
        method="multistart", n_minima=4, seed=3,
    )
    # at least the two genuine basins
    assert len(fits) >= 2
    # every entry is a fully-formed result, ranked best-SSE-first
    assert all(isinstance(r, pounce.CurveFitResult) for r in fits)
    assert all(fits[i].sse <= fits[i + 1].sse for i in range(len(fits) - 1))
    # the recovered centers include both peaks
    centers = sorted(round(float(r.popt[1])) for r in fits)
    assert -4 in centers and 4 in centers
    # results carry the usual machinery
    assert fits[0].pcov.shape == (3, 3)
    assert fits[0].ci.shape == (3, 2)


def test_curve_fit_minima_single_basin_matches_curve_fit():
    # A line has one minimum; curve_fit_minima should return a single result
    # equal to plain curve_fit.
    rng = np.random.default_rng(5)
    x = np.linspace(0.0, 5.0, 40)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.02, x.size)
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    single = pounce.curve_fit(line_j, x, y, p0=[0.0, 0.0], bounds=bounds, jac="fd")
    multi = pounce.curve_fit_minima(
        line_j, x, y, bounds=bounds, jac="fd", method="multistart",
        n_minima=3, seed=1,
    )
    assert len(multi) == 1
    np.testing.assert_allclose(multi[0].popt, single.popt, atol=1e-4)


# --------------------------------------------------------------------------
# Regression: scipy-free Student-t quantile fallback is accurate.
# --------------------------------------------------------------------------

def test_t_quantile_fallback_matches_scipy_for_small_dof():
    """The scipy-free ``_t_ppf_fallback`` must match scipy across the whole
    ``dof >= 1`` range. The previous normal-plus-Cornish-Fisher fallback was
    ~66% too small at ``dof=1``, silently narrowing confidence intervals on a
    numpy-only install (scipy is an optional dependency)."""
    from scipy.stats import t as scit

    from pounce._curve_fit import _t_ppf_fallback

    for dof in (1, 2, 3, 4, 5, 8, 10, 30, 100, 1000):
        for q in (0.6, 0.9, 0.95, 0.975, 0.99, 0.995):
            approx = _t_ppf_fallback(q, dof)
            exact = float(scit.ppf(q, dof))
            assert abs(approx - exact) <= 1e-6 * abs(exact), (dof, q, approx, exact)
    # symmetry about the median
    assert _t_ppf_fallback(0.025, 7) == pytest.approx(-_t_ppf_fallback(0.975, 7))


def test_curve_fit_ci_is_scipy_free_accurate(monkeypatch):
    """End-to-end: hiding scipy must not change the reported confidence
    intervals (regression for the over-narrow scipy-free CIs)."""
    import builtins

    rng = np.random.default_rng(11)
    x = np.linspace(0, 2, 7)  # small sample -> dof=4, where the old bug bit
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.standard_normal(x.size)

    with_scipy = pounce.curve_fit(expdecay_np, x, y, p0=[2.0, 1.0, 0.0], jac="fd")

    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy hidden for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without_scipy = pounce.curve_fit(expdecay_np, x, y, p0=[2.0, 1.0, 0.0], jac="fd")

    np.testing.assert_allclose(without_scipy.ci, with_scipy.ci, rtol=1e-5)


# --------------------------------------------------------------------------
# Regression: degenerate degrees of freedom are reported honestly.
# --------------------------------------------------------------------------

def test_non_positive_dof_warns_and_reports_undefined_uncertainty():
    """An exactly- or under-determined fit (n_data <= n_params) must report the
    true (<= 0) dof, warn, and hand back non-finite covariance/CIs rather than
    clamp dof to 1 and fabricate finite uncertainties."""
    def f(x, a, b):
        return a * np.exp(-b * x)

    # n_data == n_params -> dof == 0
    x = np.array([0.0, 1.0])
    y = f(x, 2.5, 1.3)
    with pytest.warns(UserWarning, match="degrees of freedom"):
        r = pounce.curve_fit(f, x, y, p0=[2.0, 1.0], jac="fd")
    assert r.dof == 0
    assert not np.all(np.isfinite(r.perr))
    assert not np.all(np.isfinite(r.ci))


def test_huber_and_soft_l1_are_documented_aliases():
    """``huber`` and ``soft_l1`` are intentionally the same smooth (C2)
    pseudo-Huber loss — a true piecewise Huber is only C1, which the IPM can't
    use. Pin the alias so the duplication stays deliberate and documented."""
    from pounce._curve_fit import _LOSSES

    assert _LOSSES["huber"] is _LOSSES["soft_l1"]

    rng = np.random.default_rng(7)
    x = np.linspace(0, 2, 30)
    y = 2.5 * np.exp(-1.3 * x) + 0.03 * rng.standard_normal(x.size)

    def f(x, a, b):
        return a * np.exp(-b * x)

    rh = pounce.curve_fit(f, x, y, p0=[2.0, 1.0], loss="huber", jac="fd")
    rs = pounce.curve_fit(f, x, y, p0=[2.0, 1.0], loss="soft_l1", jac="fd")
    np.testing.assert_array_equal(rh.popt, rs.popt)


def test_curve_fit_rejects_wrong_length_bounds():
    """Both bounds forms are length-checked up front. The scipy 2-tuple form
    ``(lo, hi)`` with array sides must match the parameter count, and a
    per-parameter list of pairs must have one pair per parameter — otherwise a
    too-short list used to silently leave parameters unbounded."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 5, 40)
    y = 3.0 * np.exp(-0.9 * x) + 0.5 + 0.02 * rng.standard_normal(x.size)

    # scipy 2-tuple form with wrong-length array sides (3 params, length-2 arrays)
    with pytest.raises(ValueError, match="bounds (lower|upper) has length 2 but the problem has 3"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0],
                         bounds=([0, 0], [5, 5]))

    # per-parameter list of pairs with the wrong count (2 pairs, 3 params)
    with pytest.raises(ValueError, match="bounds has 2 entries but the problem has 3"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0],
                         bounds=[(0, 5), (0, 5)])

    # correct forms still work: scalar scipy tuple broadcasts, full list matches
    r1 = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], bounds=(-10, 10))
    r2 = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0],
                          bounds=[(-10, 10), (-10, 10), (-10, 10)])
    np.testing.assert_allclose(r1.popt, r2.popt, rtol=1e-6)


def test_scipy_tuple_bounds_two_params_not_misread_as_pair_list():
    """A length-2 ``(lower, upper)`` tuple follows scipy's convention even for a
    2-parameter model, where it is *also* structurally a list of two ``(lo, hi)``
    pairs (issue #260).

    Previously pounce resolved that ambiguity the pair-list way, silently
    applying the transposed box (param 0 in ``[l0, l1]``, param 1 in
    ``[u0, u1]``) and returning ``Solve_Succeeded`` with a badly wrong fit. The
    scipy oracle is the ground truth for the API pounce documents it mirrors.
    """
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x)  # noiseless: A=2, k=1; deterministic
    model = expdecay_2p

    # scipy convention: bounds = (lower_array, upper_array) -> A in [0,10],
    # k in [1.6,10]. k rides its lower bound, A re-fits to compensate.
    lower, upper = [0.0, 1.6], [10.0, 10.0]
    rp = pounce.curve_fit(model, x, y, p0=[1.0, 2.0], bounds=(lower, upper),
                          jac="fd")
    rs, _ = scipy_optimize.curve_fit(model, x, y, p0=[1.0, 2.0],
                                     bounds=(lower, upper))
    np.testing.assert_allclose(rp.popt, rs, rtol=1e-4)
    # the true constrained optimum, not the misread box [1.6, 10.0]
    np.testing.assert_allclose(rp.popt, [2.42445211, 1.6], rtol=1e-4)

    # The equivalent per-parameter pair *list* is pounce's own API and gives the
    # identical box (A in [0,10], k in [1.6,10]) -- the two forms agree here.
    rl = pounce.curve_fit(model, x, y, p0=[1.0, 2.0],
                          bounds=[(0.0, 10.0), (1.6, 10.0)], jac="fd")
    np.testing.assert_allclose(rl.popt, rs, rtol=1e-4)


def test_scipy_tuple_bounds_match_scipy_across_param_counts():
    """The scipy ``(lower, upper)`` 2-tuple maps to the same box as scipy for
    n=2 (the formerly-broken ambiguous case) and n=3, and the scalar form keeps
    working."""
    x = np.linspace(0.0, 3.0, 80)

    # n=2
    y2 = 2.0 * np.exp(-1.0 * x)
    b2 = ([0.0, 1.6], [10.0, 10.0])
    p2 = pounce.curve_fit(expdecay_2p, x, y2, p0=[1.0, 2.0], bounds=b2,
                          jac="fd").popt
    s2, _ = scipy_optimize.curve_fit(expdecay_2p, x, y2, p0=[1.0, 2.0], bounds=b2)
    np.testing.assert_allclose(p2, s2, rtol=1e-4)

    # n=3 (already matched scipy; guard against regression)
    y3 = expdecay_np(x, 3.0, 0.9, 0.5)
    b3 = ([0.0, 0.0, -5.0], [10.0, 5.0, 5.0])
    p3 = pounce.curve_fit(expdecay_np, x, y3, p0=[1.0, 1.0, 0.0], bounds=b3,
                          jac="fd").popt
    s3, _ = scipy_optimize.curve_fit(expdecay_np, x, y3, p0=[1.0, 1.0, 0.0],
                                     bounds=b3)
    np.testing.assert_allclose(p3, s3, rtol=1e-3)

    # scalar scipy form
    ps = pounce.curve_fit(expdecay_2p, x, y2, p0=[1.0, 2.0], bounds=(0, 10),
                          jac="fd").popt
    ss, _ = scipy_optimize.curve_fit(expdecay_2p, x, y2, p0=[1.0, 2.0],
                                     bounds=(0, 10))
    np.testing.assert_allclose(ps, ss, rtol=1e-4)


def test_scipy_tuple_bounds_reversed_looking_box_is_valid():
    """``bounds=([0.5, 0.1], [0.6, 0.2])`` is valid scipy semantics (param 0 in
    ``[0.5, 0.6]``, param 1 in ``[0.1, 0.2]``) and must not raise the
    reversed-bound error it produced when misread as pairs (issue #260)."""
    x = np.linspace(0, 3, 60)
    y = 0.55 * np.exp(-0.15 * x)
    r = pounce.curve_fit(expdecay_2p, x, y, p0=[0.55, 0.15],
                         bounds=([0.5, 0.1], [0.6, 0.2]), jac="fd")
    assert 0.5 <= r.popt[0] <= 0.6 + 1e-9
    assert 0.1 <= r.popt[1] <= 0.2 + 1e-9


# --------------------------------------------------------------------------
# issue #265: the mirror of #260 -- an n=2 tuple *of pairs* is ambiguous and
# must raise rather than silently transpose; NaN bounds are rejected.
# --------------------------------------------------------------------------

def test_normalize_bound_arg_unit_matrix():
    """Pin the disambiguation table directly on the bounds normaliser (#265):
    the ambiguous n=2 tuple-of-pairs raises, every unambiguous spelling maps to
    exactly one reading, and the n!=2 / scalar / None-side cases are unchanged.
    """
    inf = np.inf

    # ambiguous n=2 shapes -> raise (tuple-of-pairs, and None inside either side)
    with pytest.raises(ValueError, match="ambiguous for a 2-parameter model"):
        _normalize_bound_arg(((0.0, 10.0), (0.0, 10.0)), 2)
    with pytest.raises(ValueError, match="ambiguous for a 2-parameter model"):
        _normalize_bound_arg(((None, 10.0), (0.0, None)), 2)
    with pytest.raises(ValueError, match="ambiguous for a 2-parameter model"):
        _normalize_bound_arg(([None, 10.0], [0.0, None]), 2)   # None inside a list
    with pytest.raises(ValueError, match="ambiguous for a 2-parameter model"):
        _normalize_bound_arg(((0.0, 10.0), None), 2)           # element 0 pair-shaped

    # literal NaN in the scipy branch -> raise
    with pytest.raises(ValueError, match="NaN"):
        _normalize_bound_arg((float("nan"), 10.0), 2)

    # pair *lists* pass through untouched (the unambiguous pair-list spelling)
    assert _normalize_bound_arg([(0.0, 10.0), (0.0, 10.0)], 2) == [(0.0, 10.0), (0.0, 10.0)]
    assert _normalize_bound_arg([(None, 10.0), (0.0, None)], 2) == [(None, 10.0), (0.0, None)]

    # scipy tuple of lists/arrays -> per-parameter pairs (param i in [l_i, u_i])
    assert _normalize_bound_arg(([0.0, 1.6], [10.0, 10.0]), 2) == [(0.0, 10.0), (1.6, 10.0)]
    assert _normalize_bound_arg(
        (np.array([0.0, 1.6]), np.array([10.0, 10.0])), 2
    ) == [(0.0, 10.0), (1.6, 10.0)]
    assert _normalize_bound_arg(([0, 0, 0], [10, 10, 10]), 3) == [
        (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

    # scalar scipy form broadcasts at every n
    assert _normalize_bound_arg((0, 10), 1) == [(0.0, 10.0)]
    assert _normalize_bound_arg((0, 10), 2) == [(0.0, 10.0), (0.0, 10.0)]
    assert _normalize_bound_arg((0, 10), 3) == [(0.0, 10.0)] * 3

    # a bare None side takes the scipy reading, None -> ∓inf (no raise). The n=1
    # case is the regression guard for §3.2 (was NaN-as-unbounded-below before).
    assert _normalize_bound_arg((None, 10.0), 1) == [(-inf, 10.0)]
    assert _normalize_bound_arg((None, 10.0), 2) == [(-inf, 10.0), (-inf, 10.0)]
    assert _normalize_bound_arg((None, None), 2) == [(-inf, inf), (-inf, inf)]

    # n != 2 tuple pair list never enters the scipy branch (len != 2) -> unchanged
    assert _normalize_bound_arg(((0, 10), (0, 10), (0, 10)), 3) == ((0, 10), (0, 10), (0, 10))
    assert _normalize_bound_arg(((0, 10),), 1) == ((0, 10),)

    # None passes through as None
    assert _normalize_bound_arg(None, 2) is None


def test_curve_fit_ambiguous_tuple_bounds_raise():
    """#265: the n=2 tuple-of-pairs box now raises on the single-fit surface
    instead of silently transposing to a pinned box that still 'succeeds'."""
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x)  # noiseless: A=2, k=1
    with pytest.raises(ValueError, match="ambiguous"):
        pounce.curve_fit(expdecay_2p, x, y, p0=[1.0, 2.0],
                         bounds=((0.0, 10.0), (0.0, 10.0)), jac="fd")
    with pytest.raises(ValueError, match="ambiguous"):
        pounce.curve_fit(expdecay_2p, x, y, p0=[1.0, 2.0],
                         bounds=((None, 10.0), (0.0, None)), jac="fd")


def test_curve_fit_list_spelling_fits_and_matches_scipy():
    """The unambiguous pair-*list* spelling of the same box fits correctly and
    agrees with scipy's equivalent (lower, upper) call."""
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x)
    r = pounce.curve_fit(expdecay_2p, x, y, p0=[1.0, 2.0],
                         bounds=[(None, 10.0), (0.0, None)], jac="fd")
    np.testing.assert_allclose(r.popt, [2.0, 1.0], rtol=1e-4)
    rs, _ = scipy_optimize.curve_fit(
        expdecay_2p, x, y, p0=[1.0, 2.0],
        bounds=([-np.inf, 0.0], [10.0, np.inf]))
    np.testing.assert_allclose(r.popt, rs, rtol=1e-4)


def test_curve_fit_minima_ambiguous_tuple_raises_and_list_fits():
    """``curve_fit_minima`` inherits the raise (covers ``_minima_bounds``); the
    finite pair-*list* box enumerates the correct minimum near (2, 1)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x) + 0.01 * rng.standard_normal(x.size)

    with pytest.raises(ValueError, match="ambiguous"):
        pounce.curve_fit_minima(expdecay_2p, x, y, p0=[1.0, 2.0],
                                bounds=((0.0, 10.0), (0.0, 10.0)),
                                jac="fd", seed=0)

    fits = pounce.curve_fit_minima(expdecay_2p, x, y, p0=[1.0, 2.0],
                                   bounds=[(0.0, 10.0), (0.0, 10.0)],
                                   jac="fd", seed=0)
    assert fits, "expected at least one minimum"
    np.testing.assert_allclose(fits[0].popt, [2.0, 1.0], rtol=5e-2)
    # not the degenerate zero-width-box state: a real box gives a real perr
    assert np.all(np.isfinite(fits[0].perr))
    assert np.any(fits[0].perr > 0)


def test_curve_fit_minima_rejects_nan_pair_bound():
    """A NaN inside a pair-list entry is rejected (not laundered to 'unbounded'),
    consistent with the minimize path (§3.4)."""
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x)
    with pytest.raises(ValueError, match="NaN"):
        pounce.curve_fit_minima(expdecay_2p, x, y, p0=[1.0, 2.0],
                                bounds=[(float("nan"), 10.0), (0.0, 10.0)],
                                jac="fd", seed=0)


def test_curve_fit_streaming_ambiguous_tuple_raises_and_scipy_box_fits():
    """``curve_fit_streaming`` inherits the raise; the scipy tuple-of-arrays box
    still fits to (2, 1)."""
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.0 * x)
    with pytest.raises(ValueError, match="ambiguous"):
        pounce.curve_fit_streaming(expdecay_2p, _batched_source(x, y),
                                   p0=[1.0, 2.0],
                                   bounds=((0.0, 10.0), (0.0, 10.0)), jac="fd")
    r = pounce.curve_fit_streaming(expdecay_2p, _batched_source(x, y),
                                   p0=[1.0, 2.0],
                                   bounds=([0.0, 0.0], [10.0, 10.0]), jac="fd")
    np.testing.assert_allclose(r.popt, [2.0, 1.0], rtol=1e-4)


def test_curve_fit_validates_data_sigma_fscale_p0():
    """Imperfect-but-plausible inputs that used to fail cryptically (LinAlgError,
    ZeroDivisionError, broadcast errors, a silent garbage fit) now raise clear
    ValueErrors up front, like scipy's validation."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 5, 20)
    y = 3.0 * np.exp(-0.9 * x) + 0.5 + 0.02 * rng.standard_normal(x.size)

    # sigma must be positive and finite (was: SVD/LinAlgError on 0, silent on <0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], sigma=np.r_[0.0, np.ones(19)])
    with pytest.raises(ValueError, match="sigma must be positive"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], sigma=-np.ones(20))

    # f_scale must be positive and finite (was: LinAlgError on 0, silent on <0)
    with pytest.raises(ValueError, match="f_scale must be a positive"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], loss="huber", f_scale=0.0)
    with pytest.raises(ValueError, match="f_scale must be a positive"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0], loss="huber", f_scale=-1.0)

    # p0 length must match the model arity (was: TypeError deep in the solve)
    with pytest.raises(ValueError, match="p0 has 2 value.* but the model takes 3"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1])
    with pytest.raises(ValueError, match="p0 has 4 value.* but the model takes 3"):
        pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0, 0])

    # xdata / ydata length mismatch (was: cryptic broadcast ValueError)
    with pytest.raises(ValueError, match="length mismatch"):
        pounce.curve_fit(expdecay, x[:10], y, p0=[1, 1, 0])

    # empty data (was: ZeroDivisionError)
    with pytest.raises(ValueError, match="ydata is empty"):
        pounce.curve_fit(expdecay, np.array([]), np.array([]), p0=[1, 1, 0])

    # non-finite data (was: RuntimeError: back-solve failed)
    with pytest.raises(ValueError, match="ydata contains non-finite"):
        pounce.curve_fit(expdecay, x, np.r_[np.nan, y[1:]], p0=[1, 1, 0])
    with pytest.raises(ValueError, match="xdata contains non-finite"):
        pounce.curve_fit(expdecay, np.r_[np.inf, x[1:]], y, p0=[1, 1, 0])

    # the well-formed call still fits
    r = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0])
    np.testing.assert_allclose(r.popt, [3.0, 0.9, 0.5], atol=0.1)


def test_curve_fit_rejects_keyword_only_model():
    """A model with keyword-only parameters can't be called positionally as
    ``f(x, *params)`` (how curve_fit invokes it); fail with a clear message
    instead of a downstream ``TypeError: takes 1 positional argument``."""
    def kwonly(x, *, a, b):
        return a * np.exp(-b * x)

    x = np.linspace(0.2, 5, 20)
    y = 2.0 * np.exp(-0.5 * x)
    with pytest.raises(ValueError, match="keyword-only"):
        pounce.curve_fit(kwonly, x, y, p0=[1, 1])


def test_confidence_band_validates_shapes():
    """The delta-method band is a per-point 1-D quantity: a wrong-dimensional
    ``x`` used to raise a cryptic einsum error, and a wrong-length prediction
    ``sigma`` a cryptic broadcast error. Both now raise clear ValueErrors."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.2, 5, 20)
    y = 3.0 * np.exp(-0.9 * x) + 0.5 + 0.02 * rng.standard_normal(x.size)
    r = pounce.curve_fit(expdecay, x, y, p0=[1, 1, 0])

    # x with the wrong number of dimensions
    with pytest.raises(ValueError, match="same dimensionality"):
        r.confidence_band(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # prediction-band sigma that is neither scalar nor x-shaped
    with pytest.raises(ValueError, match="sigma must be a scalar or match x"):
        r.confidence_band(x[:5], kind="prediction", sigma=np.ones(3))

    # well-formed bands still work (scalar sigma, matching sigma, confidence)
    for band in (
        r.confidence_band(x[:5]),
        r.confidence_band(x[:5], kind="prediction", sigma=0.5),
        r.confidence_band(x[:5], kind="prediction", sigma=np.ones(5)),
    ):
        assert all(z.shape == (5,) for z in band)


# --------------------------------------------------------------------------
# Streaming / out-of-core fits (curve_fit_streaming)
#
# The solver only needs additive sums over data points, so streaming the data
# in mini-batches and accumulating those sums must reproduce the in-memory fit
# exactly -- only one batch is ever held in memory.
# --------------------------------------------------------------------------

def _batched_source(x, y, sigma=None, step=137):
    """Return a zero-arg factory yielding fresh (x, y[, sigma]) batches.

    ``step`` deliberately does not divide the data length, exercising a smaller
    final batch (and the JAX shape retrace it triggers).
    """
    def source():
        for i in range(0, len(y), step):
            if sigma is None:
                yield x[i : i + step], y[i : i + step]
            else:
                yield x[i : i + step], y[i : i + step], sigma[i : i + step]
    return source


@pytest.mark.parametrize("jac", ["jax", "fd"])
@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("use_sigma", [False, True])
def test_streaming_matches_full_memory(jac, absolute_sigma, use_sigma):
    """Exact parity: streamed fit == in-memory fit on the same data."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 1000)
    sigma = (0.04 + 0.02 * x) if use_sigma else None
    noise = rng.normal(0.0, 0.05, x.size) if sigma is None else rng.normal(0.0, sigma)
    y = expdecay_np(x, 2.0, 0.3, 0.5) + noise

    model = expdecay if jac == "jax" else expdecay_np
    kw = dict(p0=[1.0, 1.0, 0.0], jac=jac, absolute_sigma=absolute_sigma)

    full = pounce.curve_fit(model, x, y, sigma=sigma, **kw)
    streamed = pounce.curve_fit_streaming(
        model, _batched_source(x, y, sigma), **kw
    )

    np.testing.assert_allclose(streamed.popt, full.popt, rtol=1e-8)
    np.testing.assert_allclose(streamed.pcov, full.pcov, rtol=1e-6, atol=1e-12)
    assert streamed.dof == full.dof
    assert streamed.n_data == full.n_data == 1000
    assert streamed.cov_source == full.cov_source
    np.testing.assert_allclose(streamed.sse, full.sse, rtol=1e-9)
    np.testing.assert_allclose(streamed.chi_square, full.chi_square, rtol=1e-9)
    np.testing.assert_allclose(streamed.r_squared, full.r_squared, rtol=1e-9)
    np.testing.assert_allclose(streamed.mae, full.mae, rtol=1e-9)


def test_streaming_robust_sandwich_matches_full_memory():
    """Robust loss -> the sandwich covariance is accumulated over batches."""
    rng = np.random.default_rng(3)
    x = np.linspace(0.0, 10.0, 800)
    y = expdecay_np(x, 2.0, 0.3, 0.5) + rng.normal(0.0, 0.05, x.size)
    y[::50] += 2.0  # outliers

    full = pounce.curve_fit(expdecay, x, y, p0=[1.0, 1.0, 0.0], loss="huber")
    streamed = pounce.curve_fit_streaming(
        expdecay, _batched_source(x, y), p0=[1.0, 1.0, 0.0], loss="huber"
    )
    assert full.cov_source == streamed.cov_source == "sandwich"
    np.testing.assert_allclose(streamed.popt, full.popt, rtol=1e-7)
    np.testing.assert_allclose(streamed.pcov, full.pcov, rtol=1e-5, atol=1e-12)


def test_streaming_active_bound_projects_covariance():
    """An active bound projects the covariance onto the free set, same as the
    in-memory fit."""
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 5.0, 400)
    y = expdecay_np(x, 3.0, 0.9, 0.5) + rng.normal(0.0, 0.05, x.size)
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, 0.3)]  # true c=0.5

    full = pounce.curve_fit(expdecay, x, y, p0=[1.0, 1.0, 0.0], bounds=bounds)
    streamed = pounce.curve_fit_streaming(
        expdecay, _batched_source(x, y), p0=[1.0, 1.0, 0.0], bounds=bounds
    )
    np.testing.assert_array_equal(streamed.active_mask, full.active_mask)
    assert streamed.active_mask[2]
    assert streamed.cov_source == full.cov_source == "reduced_hessian(projected)"
    np.testing.assert_allclose(streamed.popt, full.popt, rtol=1e-6)
    np.testing.assert_allclose(streamed.pcov, full.pcov, rtol=1e-5, atol=1e-12)


def test_streaming_active_equality_projects_covariance():
    """The streaming covariance must project out an active general equality
    exactly like the in-memory fit (the streaming twin of the M30 bug)."""
    rng = np.random.default_rng(31)
    x = np.linspace(0.0, 10.0, 400)
    y = line(x, 2.0, -1.0) + rng.normal(0.0, 0.5, x.size)
    cons = [{"type": "eq", "fun": lambda p: p[0] + p[1] - 1.0,
             "jac": lambda p: np.array([[1.0, 1.0]])}]

    full = pounce.curve_fit(line_j, x, y, p0=[1.0, 0.0], constraints=cons)
    streamed = pounce.curve_fit_streaming(
        line_j, _batched_source(x, y), p0=[1.0, 0.0], constraints=cons
    )
    assert streamed.cov_source == full.cov_source == "reduced_hessian(projected)"
    g = np.array([1.0, 1.0])
    assert float(g @ streamed.pcov @ g) < 1e-9
    np.testing.assert_allclose(streamed.popt, full.popt, rtol=1e-6)
    np.testing.assert_allclose(streamed.pcov, full.pcov, rtol=1e-5, atol=1e-10)


def test_streaming_disables_residuals_and_sensitivity():
    """The O(n_data) outputs are not materialised; summary still renders."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 5.0, 300)
    y = expdecay_np(x, 3.0, 0.9, 0.5) + rng.normal(0.0, 0.05, x.size)
    res = pounce.curve_fit_streaming(expdecay, _batched_source(x, y), p0=[1.0, 1.0, 0.0])
    assert res.residuals is None
    assert res.dpopt_ddata is None
    assert isinstance(res.summary(), str)
    # confidence band on new x still works (homoscedastic fallback)
    band = res.confidence_band(x[:5])
    assert all(z.shape == (5,) for z in band)


def test_streaming_factory_is_reusable_and_never_materializes():
    """The source must be a *factory* (fresh iterator per pass), and the solve
    must never pull a slice wider than one batch."""
    rng = np.random.default_rng(0)
    n = 50_000
    x = np.linspace(0.0, 10.0, n)
    y = expdecay_np(x, 2.0, 0.3, 0.5) + rng.normal(0.0, 0.05, n)
    batch = 1000

    state = {"passes": 0, "max_batch": 0}

    def source():
        state["passes"] += 1
        for i in range(0, n, batch):
            xb = x[i : i + batch]
            state["max_batch"] = max(state["max_batch"], xb.size)
            yield xb, y[i : i + batch]

    # two independent iterators from one factory
    it1, it2 = source(), source()
    assert next(it1) is not None and next(it2) is not None

    res = pounce.curve_fit_streaming(expdecay, source, p0=[1.0, 1.0, 0.0])
    assert res.success
    assert state["passes"] > 1                 # re-read once per iteration
    assert state["max_batch"] == batch         # never a full-array read


def test_streaming_requires_factory_and_params():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 5.0, 100)
    y = expdecay_np(x, 3.0, 0.9, 0.5)

    # a one-shot iterator (not a factory) is rejected
    with pytest.raises(ValueError, match="zero-argument callable"):
        pounce.curve_fit_streaming(expdecay, iter([(x, y)]), p0=[1.0, 1.0, 0.0])

    # a model with *args params and no p0/n_params can't fix the count
    def variadic(x, *p):
        return p[0] * x + p[1]

    with pytest.raises(ValueError, match="number of parameters"):
        pounce.curve_fit_streaming(variadic, _batched_source(x, y))


def test_curve_fit_acceptable_level_reports_success():
    """gh #119 / #123 analog for curve_fit. A fit that stalls at the *acceptable*
    tolerance after the tight tolerance exits returns status 1
    (``Solved_To_Acceptable_Level``) with a fully populated ``popt``/``pcov`` —
    a converged solve. ``_solve_fit`` used to gate ``success`` on ``status == 0``
    alone, so it reported ``success=False`` at a verified optimum and callers
    gating on ``result.success`` silently discarded valid fits. It must now count
    the acceptable level (and an acceptable final KKT error) as success, matching
    ``minimize`` and the jax/torch paths.

    A very tight ``tol`` over the finite-difference path forces the acceptable
    stall deterministically.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 4.0, 60)
    y = expdecay_np(x, 2.5, 1.3, 0.5) + 0.01 * rng.standard_normal(x.size)

    with pytest.warns(UserWarning, match="finite-difference"):
        res = pounce.curve_fit(
            expdecay_np, x, y, p0=[1.0, 1.0, 0.0],
            options={"tol": 1e-12, "acceptable_tol": 1e-5, "print_level": 0},
        )

    # The tight tol forces the acceptable-level stall rather than a status-0 exit.
    assert res.status == 1, f"expected Solved_To_Acceptable_Level, got {res.message}"
    # ...which must now read as success, with a valid recovered fit.
    assert res.success is True
    np.testing.assert_allclose(res.popt, [2.5, 1.3, 0.5], atol=0.1)


def test_curve_fit_success_mapping_matches_nlp_minimize():
    """The curve_fit success rule reuses the NLP ``minimize`` status set, so the
    two entry points agree on what counts as a converged solve (no divergence to
    re-introduce the gh #119 class of bug)."""
    from pounce._minimize import _NLP_SUCCESS_STATUS

    assert 0 in _NLP_SUCCESS_STATUS      # Solve_Succeeded
    assert 1 in _NLP_SUCCESS_STATUS      # Solved_To_Acceptable_Level
    assert 2 not in _NLP_SUCCESS_STATUS  # Infeasible_Problem_Detected


def test_curve_fit_no_kkt_fallback_gate_matches_minimize():
    """L50 lockstep: curve_fit's KKT-error success fallback must exclude the
    same statuses as minimize's, so an external abort (``User_Requested_Stop``,
    code 5) can never be laundered into ``success=True`` by a coincidentally
    small final KKT error. curve_fit exposes no intermediate callback today, so
    the gate is latent — this pins both sites to the *same* constant so the two
    "judge success exactly as minimize" sites cannot drift."""
    from pounce._curve_fit import _NO_KKT_FALLBACK_STATUS as cf_gate
    from pounce._minimize import _NO_KKT_FALLBACK_STATUS as m_gate

    assert cf_gate is m_gate              # one shared source of truth
    assert 5 in cf_gate                   # User_Requested_Stop stays failed
    assert not cf_gate & {0, 1}           # success codes never gated


def test_curve_fit_constraint_probed_at_p0_not_origin():
    """L47 (curve_fit side): constraint sizing must probe at the starting seed
    ``p0``, not the origin, mirroring ``minimize``'s x0 probe — a constraint
    undefined at 0 (e.g. ``log``) but defined at p0 must not blow up before
    the solve begins. Covers both the in-memory and the streaming builder."""
    from pounce._curve_fit import _build_fit_problem, _build_streaming_fit_problem

    x = np.linspace(0.0, 10.0, 40)
    y = line(x, 2.0, 1.5)
    p0 = [1.5, 1.2]

    probes = []

    def con(p):
        probes.append(np.array(p, dtype=float))
        # Undefined (-inf) at the origin; finite at p0.
        return np.log(p)

    prob = _build_fit_problem(
        line, x, y, p0, None, (-np.inf, np.inf),
        [{"type": "ineq", "fun": con}], "sse", 1.0, "fd",
    )
    assert prob.m_con == 2
    np.testing.assert_allclose(probes[0], p0)  # probed at p0, not 0
    assert np.all(np.isfinite(prob.g_combined(np.asarray(p0, dtype=float))))

    s_probes = []

    def s_con(p):
        s_probes.append(np.array(p, dtype=float))
        return np.log(p)

    _build_streaming_fit_problem(
        line, lambda: iter([(x, y)]), p0, None, (-np.inf, np.inf),
        [{"type": "ineq", "fun": s_con}], "sse", 1.0, "fd", None,
    )
    np.testing.assert_allclose(s_probes[0], p0)  # streaming builder too


# --------------------------------------------------------------------------
# Degenerate-covariance warnings (#265): a zero-width bound or an all-active
# corner reports perr = 0 legitimately, but that must no longer be silent.
# --------------------------------------------------------------------------

def test_zero_width_bounds_warn_and_are_not_an_estimate():
    """A zero-width bound (lo == hi) pins a parameter; its perr of 0 is the
    constraint, and pounce warns while the free parameter keeps a real perr."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 3.0, 60)
    y = expdecay_2p(x, 2.0, 1.0) + 0.01 * rng.standard_normal(x.size)

    with pytest.warns(UserWarning, match="zero-width"):
        r = pounce.curve_fit(expdecay_2p, x, y, p0=[2.0, 1.0],
                             bounds=[(2.0, 2.0), (0.0, 10.0)], jac="fd")

    np.testing.assert_allclose(r.popt[0], 2.0, atol=1e-9)  # pinned exactly
    assert r.perr[0] == 0.0                                 # constraint, not estimate
    assert r.perr[1] > 0.0                                  # free param has real uncertainty


def test_all_params_on_active_bounds_warn_fully_degenerate():
    """A box that excludes the optimum pins every parameter on a bound; the
    covariance is then fully degenerate (pcov = 0) and pounce warns."""
    x = np.linspace(-1.0, 1.0, 41)  # sum(x) == 0 => SSE separable in a and b
    y = x.copy()                    # unconstrained optimum a=1, b=0
    # Box excludes the optimum; separability pins the corner (a=2, b=1).
    with pytest.warns(UserWarning, match="fully degenerate"):
        r = pounce.curve_fit(line, x, y, p0=[2.5, 1.5],
                             bounds=[(2.0, 3.0), (1.0, 2.0)], jac="fd")

    assert r.active_mask is not None and r.active_mask.all()
    np.testing.assert_array_equal(r.pcov, np.zeros_like(r.pcov))
    np.testing.assert_array_equal(r.perr, np.zeros_like(r.perr))


def test_ordinary_bounded_fit_does_not_warn_degenerate():
    """A normal fit with wide bounds and an interior optimum emits neither of
    the degenerate-covariance warnings."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 3.0, 60)
    y = expdecay_2p(x, 2.0, 1.0) + 0.02 * rng.standard_normal(x.size)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pounce.curve_fit(expdecay_2p, x, y, p0=[2.0, 1.0],
                         bounds=[(0.0, 10.0), (0.0, 10.0)], jac="fd")

    msgs = [str(w.message) for w in caught]
    assert not any("zero-width" in m for m in msgs), msgs
    assert not any("fully degenerate" in m for m in msgs), msgs


def test_curve_fit_minima_zero_width_box_warns():
    """The #265 scenario (minus the transposition): a fully zero-width box via
    curve_fit_minima still yields perr = 0 everywhere, but no longer silently."""
    x = np.linspace(0.0, 3.0, 60)
    y = 2.0 * np.exp(-1.0 * x)

    with pytest.warns(UserWarning, match="zero-width"):
        fits = pounce.curve_fit_minima(expdecay_2p, x, y, p0=[1.0, 2.0],
                                       bounds=[(0.0, 0.0), (10.0, 10.0)],
                                       jac="fd", seed=0)

    assert fits, "expected at least one result"
    for r in fits:
        np.testing.assert_array_equal(r.perr, np.zeros_like(r.perr))
