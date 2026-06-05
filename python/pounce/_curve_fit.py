"""scipy.optimize.curve_fit-style nonlinear fitting on top of pounce.

`curve_fit` fits a model ``f(x, *params)`` to data by minimising a sum of
(optionally robust) losses over the residuals with pounce's interior-point
solver, and returns a rich :class:`CurveFitResult` carrying the parameter
covariance, standard errors, confidence intervals, goodness-of-fit metrics,
and — uniquely for pounce — the parametric sensitivity ``dpopt/ddata``.

Why pounce rather than scipy: pounce keeps the converged KKT factorisation,
so the parameter covariance comes straight from the inverse-Hessian block of
``K`` (no separate Jacobian inversion) and is *correct under active bounds /
constraints* (the reduced Hessian is the projection onto the active-constraint
nullspace), and the data sensitivities are a batched back-solve against the
same factor.

Conventions match scipy / :func:`pycse.nlinfit`:

* residuals ``r_i = (f(x_i, p) - y_i) / sigma_i``; objective ``S = sum rho(r_i)``,
* covariance ``pcov = s^2 * (J_w^T J_w)^-1`` with ``s^2 = SSE/(m-n)`` (reduced
  chi-square) unless ``absolute_sigma=True`` (then ``s^2 = 1``),
* confidence intervals ``popt +/- t_{dof,1-alpha/2} * perr``.

Only **C2 (twice-differentiable)** losses are supported because the underlying
solver is an interior-point method; non-smooth L1/MAE is intentionally out of
scope (use a robust loss such as ``"huber"`` or ``"soft_l1"`` instead).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._pounce import Solver
from ._minimize import _normalize_bounds, _wrap_constraints

_EPS = float(np.finfo(np.float64).eps) ** 0.5


# --------------------------------------------------------------------------
# Loss registry. Each loss provides rho, rho', rho'' as functions of the
# squared residual z = r**2 (scipy's least_squares convention), so the
# objective is sum_i f_scale**2 * rho(r_i**2 / f_scale**2).
# We expose them through residual-space helpers below.
# --------------------------------------------------------------------------

def _loss_sse(z):
    # rho(z) = z  ->  sum r^2
    rho = z
    rho1 = np.ones_like(z)
    rho2 = np.zeros_like(z)
    return rho, rho1, rho2


def _loss_huber(z):
    # rho(z) = z if z<=1 else 2*sqrt(z)-1   (pseudo-Huber-ish, C1; smooth form below)
    t = np.sqrt(1.0 + z)
    rho = 2.0 * (t - 1.0)
    rho1 = 1.0 / t
    rho2 = -0.5 / (t ** 3)
    return rho, rho1, rho2


def _loss_cauchy(z):
    rho = np.log1p(z)
    rho1 = 1.0 / (1.0 + z)
    rho2 = -1.0 / (1.0 + z) ** 2
    return rho, rho1, rho2


def _loss_soft_l1(z):
    t = np.sqrt(1.0 + z)
    rho = 2.0 * (t - 1.0)
    rho1 = 1.0 / t
    rho2 = -0.5 / (t ** 3)
    return rho, rho1, rho2


# "huber" here uses the smooth pseudo-Huber so it stays C2 for the IPM.
_LOSSES: dict[str, Callable] = {
    "sse": _loss_sse,
    "linear": _loss_sse,
    "chi2": _loss_sse,
    "huber": _loss_huber,
    "cauchy": _loss_cauchy,
    "soft_l1": _loss_soft_l1,
}

_ROBUST = {"huber", "cauchy", "soft_l1"}


def _to_array(a, dtype=np.float64) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def _t_ppf(q: float, dof: int) -> float:
    """Student-t inverse CDF; uses scipy when available, else a normal
    approximation with a Cornish-Fisher-style small-sample correction."""
    try:
        from scipy.stats import t as _t

        return float(_t.ppf(q, dof))
    except Exception:
        # Normal quantile via inverse erf, plus a light dof correction.
        z = math.sqrt(2.0) * _erfinv(2.0 * q - 1.0)
        if dof > 0:
            g1 = (z ** 3 + z) / (4.0 * dof)
            return z + g1
        return z


def _erfinv(y: float) -> float:
    # Winitzki approximation; adequate for CI quantiles.
    a = 0.147
    ln = math.log(1.0 - y * y)
    term = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(term * term - ln / a) - term), y)


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------

@dataclass
class CurveFitResult:
    """Rich result of :func:`curve_fit`.

    Mirrors the information in :func:`scipy.optimize.curve_fit` (``popt``,
    ``pcov``) and :func:`pycse.nlinfit` (confidence intervals), and adds
    goodness-of-fit metrics plus pounce-only parametric sensitivities.
    """

    popt: np.ndarray
    pcov: np.ndarray
    perr: np.ndarray
    ci: np.ndarray  # (n, 2): lower/upper at level ``alpha``

    residuals: np.ndarray
    sse: float
    rmse: float
    mae: float
    r_squared: float
    adj_r_squared: float
    chi_square: float
    reduced_chi_square: float
    dof: int
    n_data: int
    n_params: int

    alpha: float
    loss: str
    success: bool
    status: int
    message: str
    nit: int
    cov_source: str
    optimize_result: Any = None

    param_names: list[str] | None = None
    active_mask: np.ndarray | None = None  # True where a param sits on a bound
    dpopt_ddata: np.ndarray | None = None  # (n_params, n_data)

    _model: Callable | None = field(default=None, repr=False)
    _model_jac: Callable | None = field(default=None, repr=False)
    _s2: float = field(default=1.0, repr=False)        # covariance scale (for prediction bands)
    _sigma: np.ndarray | None = field(default=None, repr=False)  # per-point noise weights
    _xdata: np.ndarray | None = field(default=None, repr=False)

    # dict-style access (parity with OptimizeResult)
    def __getitem__(self, k: str) -> Any:
        return getattr(self, k)

    @property
    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix (normalised covariance)."""
        d = np.sqrt(np.diag(self.pcov))
        denom = np.outer(d, d)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 0, self.pcov / denom, 0.0)
        return corr

    def predict(self, x) -> np.ndarray:
        """Model evaluated at ``x`` with the fitted parameters."""
        if self._model is None:
            raise RuntimeError("result has no bound model")
        return _to_array(self._model(_to_array(x), self.popt))

    def confidence_band(self, x, alpha: float | None = None, kind: str = "confidence", sigma=None):
        """Delta-method band on the model at ``x``. Returns ``(yhat, lo, hi)``.

        ``kind="confidence"`` (default) is the band on the *fitted curve* —
        uncertainty in the mean ``E[y|x]``, ``var = g^T pcov g``. It is narrow
        and most data points fall **outside** it; it shrinks toward zero as the
        data grows. This answers "where is the true curve?".

        ``kind="prediction"`` is the band for a *new observation* — it adds the
        observation noise, ``var = g^T pcov g + sigma^2``, and is what contains
        ~``1-alpha`` of the data. ``sigma`` is the per-point noise standard
        deviation at ``x`` (scalar or array). If omitted, the noise level from
        the fit is reused: the supplied ``sigma`` weights scaled by the fitted
        variance ``s^2`` (so a heteroscedastic fit yields a heteroscedastic
        band), or ``sqrt(s^2)`` when the fit was unweighted.
        """
        if kind not in ("confidence", "prediction"):
            raise ValueError("kind must be 'confidence' or 'prediction'")
        x = _to_array(x)
        a = self.alpha if alpha is None else alpha
        yhat = self.predict(x)
        G = self._model_jac(x, self.popt)  # (m, n)
        var = np.einsum("ij,jk,ik->i", G, self.pcov, G)
        if kind == "prediction":
            var = var + self._noise_var(x, sigma)
        se = np.sqrt(np.clip(var, 0.0, None))
        tval = _t_ppf(1.0 - a / 2.0, max(self.dof, 1))
        return yhat, yhat - tval * se, yhat + tval * se

    def _noise_var(self, x, sigma):
        """Per-point observation-noise variance at ``x`` for a prediction band.

        ``s^2 * sigma^2``, where ``sigma`` is provided, or (when the fit was
        weighted) the fit's own ``sigma`` if ``x`` matches the data, else the
        homoscedastic level ``s^2``.
        """
        s2 = self._s2
        if sigma is not None:
            return s2 * _to_array(sigma) ** 2 * np.ones(x.shape)
        if self._sigma is not None:
            if x.shape == self._sigma.shape and np.allclose(x, self._xdata):
                return s2 * self._sigma ** 2
            # unknown noise at new points: use the mean fitted noise level
            return s2 * float(np.mean(self._sigma ** 2)) * np.ones(x.shape)
        return s2 * np.ones(x.shape)

    def summary(self) -> str:
        names = self.param_names or [f"p{i}" for i in range(self.n_params)]
        tval = _t_ppf(1.0 - self.alpha / 2.0, max(self.dof, 1))
        lines = [
            f"curve_fit summary  (loss={self.loss}, cov={self.cov_source})",
            f"  status: {self.message}  |  n={self.n_data}  params={self.n_params}  dof={self.dof}",
            f"  SSE={self.sse:.6g}  RMSE={self.rmse:.6g}  R^2={self.r_squared:.6f}"
            f"  adjR^2={self.adj_r_squared:.6f}  reduced-chi^2={self.reduced_chi_square:.6g}",
            f"  {100*(1-self.alpha):.0f}% CIs (t_{{{self.dof}}}={tval:.4g}):",
        ]
        for i, nm in enumerate(names):
            lo, hi = self.ci[i]
            flag = "  *bound-active*" if (self.active_mask is not None and self.active_mask[i]) else ""
            lines.append(
                f"    {nm:>10s} = {self.popt[i]:+.6g} +/- {self.perr[i]:.4g}"
                f"   [{lo:+.6g}, {hi:+.6g}]{flag}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Core fitter
# --------------------------------------------------------------------------

def _infer_param_names(f: Callable) -> list[str] | None:
    try:
        params = list(inspect.signature(f).parameters.values())
    except (TypeError, ValueError):
        return None
    names = []
    for p in params[1:]:  # skip the x argument
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            return None
        names.append(p.name)
    return names or None


def _model_jac_fd(model: Callable, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Forward finite-difference df/dp -> (m, n)."""
    f0 = _to_array(model(x, p))
    m = f0.size
    n = p.size
    J = np.empty((m, n))
    for j in range(n):
        h = _EPS * max(1.0, abs(p[j]))
        pp = p.copy()
        pp[j] += h
        J[:, j] = (_to_array(model(x, pp)) - f0) / h
    return J


def curve_fit(
    f: Callable,
    xdata,
    ydata,
    p0=None,
    *,
    sigma=None,
    absolute_sigma: bool = False,
    bounds: Sequence | tuple = (-np.inf, np.inf),
    constraints: Sequence | Mapping | None = None,
    loss: str | Callable = "sse",
    f_scale: float = 1.0,
    jac: Callable | str | None = None,
    alpha: float = 0.05,
    sensitivity: bool = False,
    options: Mapping[str, Any] | None = None,
) -> CurveFitResult:
    """Fit ``f(x, *params)`` to ``(xdata, ydata)`` using pounce.

    Parameters mirror :func:`scipy.optimize.curve_fit` where they overlap.
    Extras: ``loss`` (smooth/robust loss family), ``constraints`` (general
    relations between parameters), ``sensitivity`` (compute ``dpopt/ddata``),
    and ``alpha`` (confidence level for the returned intervals).

    Returns
    -------
    CurveFitResult
    """
    xdata = _to_array(xdata)
    ydata = _to_array(ydata).ravel()
    m_data = ydata.size

    param_names = _infer_param_names(f)

    # --- initial guess / parameter count -------------------------------
    if p0 is None:
        if param_names is None:
            raise ValueError("cannot infer number of parameters; pass p0")
        p0 = np.ones(len(param_names))
    p0 = _to_array(p0).ravel()
    n = p0.size

    # --- weights -------------------------------------------------------
    if sigma is None:
        w = np.ones(m_data)
    else:
        sigma = _to_array(sigma).ravel()
        if sigma.shape != (m_data,):
            raise ValueError("sigma must be a 1-D array matching ydata")
        w = 1.0 / sigma  # residual weight 1/sigma

    # --- loss ----------------------------------------------------------
    if callable(loss):
        loss_fn = loss
        loss_name = getattr(loss, "__name__", "custom")
    else:
        if loss not in _LOSSES:
            raise ValueError(f"unknown loss {loss!r}; choose from {sorted(_LOSSES)}")
        loss_fn = _LOSSES[loss]
        loss_name = loss
    is_robust = (not callable(loss)) and loss in _ROBUST
    fs2 = float(f_scale) ** 2

    # --- model in parameter-vector form --------------------------------
    def model(x, p):
        return _to_array(f(x, *np.asarray(p))).ravel()

    model_jac, jac_exact, jac_kind = _resolve_model_jac(f, model, jac, xdata, p0)

    # weighted residual and its building blocks
    def residual(p):
        return (model(xdata, p) - ydata) * w  # r_i

    def loss_terms(p):
        # Transient overflow in intermediate iterates (large residuals during
        # the line search) is normal and benign; don't spam the user.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = residual(p)
            z = (r * r) / fs2
            rho, rho1, rho2 = loss_fn(z)
        return r, rho, rho1, rho2

    # --- objective / gradient / Gauss-Newton Hessian -------------------
    def objective(p):
        r, rho, _, _ = loss_terms(_to_array(p))
        return float(fs2 * np.sum(rho))

    def gradient(p):
        p = _to_array(p)
        r, _, rho1, _ = loss_terms(p)
        J = model_jac(xdata, p)  # (m, n), d model / dp
        # d/dp [fs2 * rho(z)] = fs2 * rho1 * (2 r / fs2) * dr/dp = 2 rho1 r * w * dmodel/dp
        return 2.0 * (J.T @ (rho1 * r * w))

    def gn_hessian(p):
        p = _to_array(p)
        r, _, rho1, rho2 = loss_terms(p)
        J = model_jac(xdata, p)
        # Gauss-Newton: keep the rho1 (first-order) curvature, drop d2model.
        # H ~ sum 2 * (rho1 + 2 z rho2) * (w dmodel/dp)(...)^T
        weight = 2.0 * (rho1 + 2.0 * (r * r) / fs2 * rho2)
        Jw = J * (w[:, None])
        return (Jw.T * weight) @ Jw

    # --- build pounce Problem object -----------------------------------
    lb, ub = _normalize_bounds(_normalize_bound_arg(bounds, n), n)
    m_con, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

    problem_obj = _make_problem_obj(
        objective, gradient, gn_hessian, n, m_con, g_combined, jac_combined
    )

    from ._pounce import Problem

    problem = Problem(n=n, m=m_con, problem_obj=problem_obj, lb=lb, ub=ub, cl=cl, cu=cu)
    problem.add_option("print_level", 0)
    problem.add_option("sb", "yes")
    # With exact derivatives (analytic/JAX) the IPM converges cleanly with no
    # NLP scaling, which keeps the converged factor's Hessian *unscaled* so the
    # pounce-native covariance/sensitivity back-solves are exact. With a
    # finite-difference fallback we leave scaling on for convergence robustness
    # (and read covariance from the scaling-independent Jacobian instead).
    user_opts = dict(options or {})
    if jac_exact and "nlp_scaling_method" not in user_opts:
        problem.add_option("nlp_scaling_method", "none")
    for k, v in user_opts.items():
        problem.add_option(k, v)
    scaling_off = jac_exact and user_opts.get("nlp_scaling_method", "none") == "none"

    solver = Solver(problem)
    popt, info = solver.solve(x0=p0)
    popt = np.asarray(popt)
    success = int(info["status"]) == 0
    # The converged factor is only trustworthy as an *unscaled* Hessian when
    # scaling was off and the solve actually held a factor.
    factor_ok = bool(scaling_off and solver.converged)

    # --- residual diagnostics (unweighted, for reporting) --------------
    yhat = model(xdata, popt)
    resid = ydata - yhat
    rw = residual(popt)  # weighted
    sse = float(resid @ resid)
    chi2 = float(rw @ rw)
    dof = max(m_data - n, 1)
    reduced_chi2 = chi2 / dof
    rmse = math.sqrt(sse / m_data)
    mae = float(np.mean(np.abs(resid)))
    ss_tot = float(np.sum((ydata - ydata.mean()) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (m_data - 1) / dof if ss_tot > 0 else float("nan")

    # scale factor s^2 (scipy's absolute_sigma rule)
    s2 = 1.0 if absolute_sigma else reduced_chi2

    # --- covariance ----------------------------------------------------
    active_mask = _active_bounds(popt, lb, ub, info)
    pcov, cov_source = _covariance(
        solver, popt, model_jac, xdata, w, s2, is_robust,
        residual, loss_fn, fs2, active_mask, n, m_con, factor_ok,
    )
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    tval = _t_ppf(1.0 - alpha / 2.0, dof)
    ci = np.column_stack([popt - tval * perr, popt + tval * perr])

    # --- sensitivity dpopt/ddata --------------------------------------
    dpopt = None
    if sensitivity:
        dpopt = _data_sensitivity(solver, model_jac, xdata, w, m_con, popt, factor_ok)

    return CurveFitResult(
        popt=popt,
        pcov=pcov,
        perr=perr,
        ci=ci,
        residuals=resid,
        sse=sse,
        rmse=rmse,
        mae=mae,
        r_squared=r2,
        adj_r_squared=adj_r2,
        chi_square=chi2,
        reduced_chi_square=reduced_chi2,
        dof=dof,
        n_data=m_data,
        n_params=n,
        alpha=alpha,
        loss=loss_name,
        success=success,
        status=int(info["status"]),
        message=str(info["status_msg"]),
        nit=int(info["iter_count"]),
        cov_source=cov_source,
        optimize_result=info,
        param_names=param_names,
        active_mask=active_mask,
        dpopt_ddata=dpopt,
        _model=model,
        _model_jac=model_jac,
        _s2=s2,
        _sigma=(None if sigma is None else _to_array(sigma).ravel()),
        _xdata=xdata,
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _normalize_bound_arg(bounds, n):
    """Accept scipy's ``(lo, hi)`` scalar/array form OR a per-parameter list
    of ``(lo, hi)`` pairs, and normalise to the per-parameter pair list that
    :func:`pounce._minimize._normalize_bounds` expects."""
    if bounds is None:
        return None
    # scipy form: a 2-tuple of (lower, upper), each scalar or length-n array
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and not _is_pair_list(bounds, n)
    ):
        lo, hi = bounds
        lo = np.broadcast_to(_to_array(lo), (n,))
        hi = np.broadcast_to(_to_array(hi), (n,))
        return list(zip(lo.tolist(), hi.tolist()))
    return bounds  # already a per-parameter list of pairs


def _is_pair_list(bounds, n):
    if len(bounds) != n:
        return False
    return all(
        (isinstance(b, (tuple, list)) and len(b) == 2) or b is None for b in bounds
    )


def _make_problem_obj(objective, gradient, hess, n, m, g, jac_g):
    members: dict[str, Any] = {
        "objective": lambda self, x: objective(x),
        "gradient": lambda self, x: _to_array(gradient(x)).ravel(),
    }
    if m > 0:
        members["constraints"] = lambda self, x: _to_array(g(x)).ravel()

        def jacobianstructure(self):
            rows = np.repeat(np.arange(m), n)
            cols = np.tile(np.arange(n), m)
            return rows, cols

        members["jacobianstructure"] = jacobianstructure
        members["jacobian"] = lambda self, x: _to_array(jac_g(x)).ravel()

    # objective Hessian always available (Gauss-Newton); pinned-constraint
    # rows are linear so they add nothing to the Lagrangian Hessian.
    def hessianstructure(self):
        r, c = np.tril_indices(n)
        return r, c

    def hessian(self, x, lam, obj_factor):
        H = obj_factor * _to_array(hess(x))
        r, c = np.tril_indices(n)
        return H[r, c]

    members["hessianstructure"] = hessianstructure
    members["hessian"] = hessian

    return type("_CurveFitProblem", (object,), members)()


def _active_bounds(popt, lb, ub, info, tol=1e-6):
    n = popt.size
    mask = np.zeros(n, dtype=bool)
    if lb is not None:
        mask |= np.isfinite(lb) & (popt - lb <= tol * np.maximum(1.0, np.abs(lb)))
    if ub is not None:
        mask |= np.isfinite(ub) & (ub - popt <= tol * np.maximum(1.0, np.abs(ub)))
    return mask


def _inv_hessian_from_factor(solver, n, m_con):
    """Columns of ``inv(H_S)`` via back-solves against the converged factor.

    For an interior optimum with no active general constraints, ``K = H_S``
    and ``kkt_solve(e_j in x-block)`` returns column ``j`` of ``inv(H_S)``.
    Returns ``None`` if the factor is unavailable or shape is unexpected.
    """
    dim = solver.kkt_dim
    if dim is None:
        return None
    cols = np.zeros((n, n))
    eye_rhs = np.zeros((n, dim))
    for j in range(n):
        eye_rhs[j, j] = 1.0
    lhs = solver.kkt_solve_many(eye_rhs.reshape(-1), n).reshape(n, dim)
    for j in range(n):
        cols[:, j] = lhs[j, :n]
    # symmetrise (numerical)
    return 0.5 * (cols + cols.T)


def _covariance(
    solver, popt, model_jac, xdata, w, s2, is_robust,
    residual, loss_fn, fs2, active_mask, n, m_con, factor_ok,
):
    """Parameter covariance.

    For least squares ``pcov = s^2 (J_w^T J_w)^-1`` (the inverse reduced
    Hessian of the sum-of-squares objective). When the converged factor is
    trustworthy (exact derivatives, scaling off) this is read pounce-natively
    from the inverse-Hessian block of ``K`` (``pcov = 2 s^2 inv(H_S)``, no
    explicit matrix inverse); otherwise it is formed from the model Jacobian,
    which is scaling-independent and gives the identical value. Robust losses
    use the sandwich estimator; active bounds/constraints project onto the
    free parameter set.
    """
    J = model_jac(xdata, popt)              # (m, n) dmodel/dp
    Jw = J * w[:, None]                      # weighted Jacobian
    M = Jw.T @ Jw                            # = H_S / 2 for squared loss

    if is_robust:
        # sandwich: A^-1 B A^-1, A = GN Hessian/2, B = score outer product.
        r = residual(popt)
        z = (r * r) / fs2
        _, rho1, rho2 = loss_fn(z)
        A = (Jw.T * (rho1 + 2.0 * z * rho2)) @ Jw
        score = (rho1 * r)[:, None] * Jw
        B = score.T @ score
        Ainv = np.linalg.pinv(A)
        return s2 * (Ainv @ B @ Ainv), "sandwich"

    # active bounds / general constraints: project onto the free set.
    if m_con > 0 or (active_mask is not None and active_mask.any()):
        free = ~active_mask if active_mask is not None else np.ones(n, bool)
        cov = np.zeros((n, n))
        cov[np.ix_(free, free)] = s2 * np.linalg.pinv(M[np.ix_(free, free)])
        return cov, "reduced_hessian(projected)"

    # interior least-squares optimum.
    if factor_ok:
        inv_hs = _inv_hessian_from_factor(solver, n, m_con)
        if inv_hs is not None:
            # inv(H_S) read directly from pounce's converged factor.
            return 2.0 * s2 * inv_hs, "reduced_hessian"
    return s2 * np.linalg.pinv(M), "jacobian"


def _data_sensitivity(solver, model_jac, xdata, w, m_con, popt, factor_ok):
    """``dpopt/ddata`` (n_params x n_data).

    For a (weighted) least-squares fit the implicit-function theorem gives
    ``dpopt/dy_i = 2 w_i^2 inv(H_S) g_i`` with ``g_i = dmodel/dp`` at point
    ``i`` (for unweighted data this is exactly ``pinv(J)``). When the factor is
    trustworthy each ``inv(H_S) g_i`` is a back-solve against pounce's
    converged factor, fanned out in one ``kkt_solve_many`` call; otherwise the
    same value is formed from the dense Gauss-Newton Hessian.
    """
    J = model_jac(xdata, popt)            # (m, n) dmodel/dp at the optimum
    n = popt.size
    m = J.shape[0]
    dim = solver.kkt_dim
    if not factor_ok or dim is None or m_con != 0:
        Jw = J * w[:, None]
        inv_hs = np.linalg.pinv(2.0 * (Jw.T @ Jw))
        return (2.0 * (w ** 2))[None, :] * (inv_hs @ J.T)

    rhs = np.zeros((m, dim))
    rhs[:, :n] = J                         # pack each g_i into the x-block
    lhs = solver.kkt_solve_many(rhs.reshape(-1), m).reshape(m, dim)
    inv_hs_g = lhs[:, :n].T                 # (n, m): column i = inv(H_S) g_i
    return (2.0 * (w ** 2))[None, :] * inv_hs_g


# --------------------------------------------------------------------------
# derivative resolution: analytic jac > JAX autodiff > finite differences
# --------------------------------------------------------------------------

def _resolve_model_jac(f, model, jac, xdata, p0):
    """Return ``(model_jac, exact, kind)``.

    ``model_jac(x, p) -> (len(x), len(p))`` is ``dmodel/dp``. Preference order:
    a user ``jac`` callable, then JAX autodiff (exact, default when the model
    is traceable), then a finite-difference fallback. ``exact`` is True for the
    first two and gates the scaling-off / pounce-native factor path.
    """
    if callable(jac):
        def mj(x, p):
            return np.atleast_2d(_to_array(jac(x, *np.asarray(p))))
        return mj, True, "analytic"

    def fd(x, p):
        return _model_jac_fd(model, x, np.asarray(p, dtype=float))

    if jac == "fd":
        return fd, False, "fd"

    if jac in (None, "jax", "auto"):
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            _jfn = jax.jit(
                jax.jacfwd(lambda p, x: jnp.atleast_1d(jnp.asarray(f(x, *p))), argnums=0)
            )

            def mj(x, p):
                out = _jfn(jnp.asarray(p, dtype=jnp.float64), jnp.asarray(x, dtype=jnp.float64))
                return np.atleast_2d(np.asarray(out))

            mj(xdata, p0)  # probe: trace once; raises if not JAX-traceable
            return mj, True, "jax"
        except Exception as exc:  # noqa: BLE001 - any trace failure => fall back
            if jac in ("jax",):
                raise ValueError(
                    f"jac='jax' requested but the model is not JAX-traceable: {exc}"
                ) from exc
            import warnings

            warnings.warn(
                "pounce.curve_fit: using a finite-difference Jacobian "
                "(model is not JAX-traceable and no analytic jac given). "
                "Pass jac=<callable> or write the model with jax.numpy for exact "
                "derivatives; covariance and sensitivity are most accurate then.",
                stacklevel=3,
            )
            return fd, False, "fd"

    raise ValueError(f"invalid jac={jac!r}; use a callable, 'jax', 'fd', or None")
