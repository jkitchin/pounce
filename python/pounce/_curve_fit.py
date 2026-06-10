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


def _loss_soft_l1(z):
    # Pseudo-Huber / Charbonnier loss: rho(z) = 2*(sqrt(1+z) - 1). This is the
    # canonical *smooth* (C2) Huber — quadratic for small residuals, linear in
    # |r| for large ones. scipy.least_squares calls it `soft_l1`; statistics
    # texts call it pseudo-Huber. `_LOSSES["huber"]` is a deliberate alias of
    # this function (see below), NOT a separate loss.
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


# `huber` and `soft_l1` map to the SAME function on purpose. scipy's true
# `huber` loss (rho = z for z<=1, 2*sqrt(z)-1 otherwise) is only C1 — its second
# derivative jumps at the knee — and this interior-point solver needs a
# continuous Hessian, so we serve the smooth pseudo-Huber under both names
# rather than ship a loss whose curvature is discontinuous. They are aliases;
# fitting with `loss="huber"` and `loss="soft_l1"` is identical by design.
_LOSSES: dict[str, Callable] = {
    "sse": _loss_sse,
    "linear": _loss_sse,
    "chi2": _loss_sse,
    "huber": _loss_soft_l1,
    "cauchy": _loss_cauchy,
    "soft_l1": _loss_soft_l1,
}

_ROBUST = {"huber", "cauchy", "soft_l1"}


def _to_array(a, dtype=np.float64) -> np.ndarray:
    return np.asarray(a, dtype=dtype)


def _t_ppf(q: float, dof: float) -> float:
    """Student-t inverse CDF (quantile). Uses scipy when available; otherwise an
    accurate scipy-free fallback built on the inverse regularized incomplete
    beta function (see :func:`_t_ppf_fallback`).

    The old fallback was a normal quantile plus a one-term Cornish-Fisher
    correction, which is badly inaccurate for small ``dof`` (the t-value at
    ``dof=1`` was ~3x too small), silently producing over-narrow confidence
    intervals on a numpy-only install (scipy is optional). The beta-inverse
    fallback matches scipy to ~1e-6 over the whole ``dof >= 1`` range.
    """
    try:
        from scipy.stats import t as _t

        return float(_t.ppf(q, dof))
    except Exception:
        return _t_ppf_fallback(q, dof)


def _t_ppf_fallback(q: float, dof: float) -> float:
    """Accurate scipy-free Student-t quantile via the inverse incomplete beta.

    For ``T ~ t_nu`` and ``t >= 0``, ``P(|T| > t) = I_x(nu/2, 1/2)`` with
    ``x = nu/(nu + t**2)``. Inverting that incomplete-beta relation gives the
    quantile exactly (up to the bisection tolerance), with no small-``dof``
    approximation error.
    """
    dof = float(dof)
    if q <= 0.0:
        return -math.inf
    if q >= 1.0:
        return math.inf
    if q == 0.5:
        return 0.0
    if not math.isfinite(dof) or dof <= 0.0:
        # No t-distribution to speak of; fall back to the normal quantile.
        return _norm_ppf(q)
    upper = q > 0.5
    p = q if upper else 1.0 - q
    tail = 2.0 * (1.0 - p)  # two-tailed P(|T| > t)
    x = _betaincinv(0.5 * dof, 0.5, tail)
    if x <= 0.0:
        t = math.inf
    else:
        t = math.sqrt(dof * (1.0 - x) / x)
    return t if upper else -t


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction core of the regularized incomplete beta (Lentz's
    method; Numerical Recipes ``betacf``)."""
    MAXIT = 300
    EPS = 1e-15
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` (Numerical Recipes ``betai``)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betaincinv(a: float, b: float, y: float) -> float:
    """Inverse of :func:`_betainc` in ``x``: find ``x`` with ``I_x(a, b) = y``.

    ``I_x`` is monotone increasing in ``x``, so plain bisection is robust and
    converges to machine precision; this is called only O(n_params) times per
    fit, so its cost is irrelevant next to the solve.
    """
    if y <= 0.0:
        return 0.0
    if y >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _norm_ppf(q: float) -> float:
    """Standard-normal quantile (used only for the degenerate ``dof <= 0``
    fallback, where no t-distribution is defined)."""
    return math.sqrt(2.0) * _erfinv(2.0 * q - 1.0)


def _erfinv(y: float) -> float:
    # Winitzki approximation; adequate for the degenerate normal fallback.
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
        if self._xdata is not None and x.ndim != self._xdata.ndim:
            raise ValueError(
                f"confidence_band expects x with the same dimensionality as the "
                f"fitted xdata (ndim {self._xdata.ndim}); got ndim {x.ndim}"
            )
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
            sig = _to_array(sigma)
            if sig.ndim > 0 and sig.shape != x.shape:
                raise ValueError(
                    f"sigma must be a scalar or match x (shape {x.shape}); got "
                    f"shape {sig.shape}"
                )
            return s2 * sig ** 2 * np.ones(x.shape)
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
            # ``f(x, *params)`` / ``f(x, **kw)``: count is unknown but the model
            # is still callable as ``f(x, *p)``; defer to the supplied ``p0``.
            return None
        if p.kind == p.KEYWORD_ONLY:
            # ``f(x, *, a, b)`` cannot be called positionally as ``f(x, *p)``,
            # which is how curve_fit invokes the model — fail clearly rather
            # than with a downstream "takes 1 positional argument" TypeError.
            raise ValueError(
                f"model parameter {p.name!r} is keyword-only; curve_fit calls "
                f"the model positionally as f(x, *params), so write the "
                f"parameters as positional arguments"
            )
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


def _initial_guess(model, xdata, ydata, w, lb, ub, n, loss_fn, fs2):
    """Data-driven starting point used when ``p0`` is omitted.

    ``curve_fit`` can't know what each parameter *means*, so instead of a flat
    vector of ones we score a small, model-agnostic set of candidate seeds by
    the actual (weighted, robust) objective and keep the best. Candidates are
    bound-aware — a parameter with two finite bounds is seeded at the box
    midpoint, a one-sided bound is offset safely into the feasible region, and
    the remaining free parameters sweep a handful of magnitudes anchored on the
    data scale. This gives a badly-scaled problem (true parameters ~1e6 or
    ~1e-6) a far better seed than ones while staying inside the bounds; the
    interior-point solver's own bound-push / barrier initialization takes it
    from there.
    """
    lo = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float)
    hi = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float)

    has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
    both = has_lo & has_hi
    only_lo = has_lo & ~has_hi
    only_hi = ~has_lo & has_hi
    free = ~has_lo & ~has_hi

    # Per-parameter anchor that respects whatever bounds exist.
    anchor = np.zeros(n)
    anchor[both] = 0.5 * (lo[both] + hi[both])
    anchor[only_lo] = lo[only_lo] + np.maximum(1.0, np.abs(lo[only_lo]))
    anchor[only_hi] = hi[only_hi] - np.maximum(1.0, np.abs(hi[only_hi]))

    # Magnitudes to try on the free parameters, anchored on the data scale (and
    # its reciprocal) so neither very large nor very small true values are
    # systematically out of reach. The same scalar is applied to all free
    # parameters per candidate, keeping the count linear rather than n-D.
    # Use only the finite data to set the scale; an all-NaN/empty ``ydata``
    # falls through to 1.0 without tripping numpy's "All-NaN slice" warning.
    finite_y = np.abs(ydata[np.isfinite(ydata)]) if ydata.size else ydata[:0]
    yscale = float(finite_y.max()) if finite_y.size else 1.0
    if not np.isfinite(yscale) or yscale == 0.0:
        yscale = 1.0
    mags = {1.0, 0.1, yscale, 1.0 / yscale}
    scales = sorted({sign * m for sign in (1.0, -1.0) for m in mags})

    def score(p):
        # Large/NaN residuals at a bad seed are expected; treat them as +inf so
        # the candidate is simply discarded.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r = (model(xdata, p) - ydata) * w
            s = float(fs2 * np.sum(loss_fn((r * r) / fs2)[0]))
        return s if np.isfinite(s) else np.inf

    candidates = [np.ones(n), anchor.copy()]
    for s in scales:
        cand = anchor.copy()
        cand[free] = s
        candidates.append(cand)

    best_p, best_s = None, np.inf
    for cand in candidates:
        cand = np.clip(cand, lo, hi)
        sc = score(cand)
        if sc < best_s:
            best_p, best_s = cand, sc
    # Every candidate overflowed/NaN'd (pathological model): fall back to ones
    # clipped into the box and let the solver sort it out.
    return best_p if best_p is not None else np.clip(np.ones(n), lo, hi)


@dataclass
class _FitProblem:
    """Everything needed to run (and re-run) a pounce curve fit.

    Built once by :func:`_build_fit_problem` and consumed by :func:`_solve_fit`,
    so :func:`curve_fit` and :func:`curve_fit_minima` share an *identical*
    objective, weighting, robust loss, and resolved Jacobian — the latter just
    drives the same setup from many starting points.
    """

    f: Callable
    model: Callable
    model_jac: Callable
    residual: Callable
    objective: Callable
    gradient: Callable
    gn_hessian: Callable
    loss_fn: Callable
    loss_name: str
    is_robust: bool
    fs2: float
    w: np.ndarray
    sigma: np.ndarray | None
    lb: np.ndarray | None
    ub: np.ndarray | None
    n: int
    param_names: list[str] | None
    m_con: int
    g_combined: Callable | None
    jac_combined: Callable | None
    cl: np.ndarray | None
    cu: np.ndarray | None
    jac_exact: bool
    xdata: np.ndarray
    ydata: np.ndarray
    m_data: int
    x0: np.ndarray
    bounds: Any
    constraints: Any

    # --- out-of-core / mini-batch streaming (see ``curve_fit_streaming``) ---
    # When ``streaming`` is True the full ``xdata``/``ydata`` are *not* held in
    # memory: the objective/gradient/Gauss-Newton-Hessian closures above are the
    # methods of ``closures`` (a :class:`_StreamingClosures`) which accumulate
    # those sums by re-reading the data from ``data_source`` one batch at a time.
    streaming: bool = False
    data_source: Callable | None = None
    closures: Any = None


def _build_fit_problem(
    f, xdata, ydata, p0, sigma, bounds, constraints, loss, f_scale, jac
) -> _FitProblem:
    """Assemble the model, weights, loss, Jacobian, objective/gradient/Hessian,
    bounds, and constraints for a curve fit — without solving. ``x0`` is the
    starting seed (the user's ``p0`` or the data-driven default)."""
    xdata = _to_array(xdata)
    ydata = _to_array(ydata).ravel()
    m_data = ydata.size

    # --- data sanity ---------------------------------------------------
    if m_data == 0:
        raise ValueError("ydata is empty; need at least one data point to fit")
    if xdata.ndim == 1 and xdata.size != m_data:
        raise ValueError(
            f"xdata and ydata length mismatch: len(xdata)={xdata.size}, "
            f"len(ydata)={m_data}"
        )
    if not np.all(np.isfinite(ydata)):
        raise ValueError("ydata contains non-finite values (nan/inf)")
    if not np.all(np.isfinite(xdata)):
        raise ValueError("xdata contains non-finite values (nan/inf)")

    param_names = _infer_param_names(f)

    # --- parameter count -----------------------------------------------
    # ``p0`` may be omitted. We only fix the parameter *count* here and defer
    # the seed itself until the model and bounds are in hand, so an omitted
    # guess gets a data-driven starting point rather than a flat vector of ones
    # (see ``_initial_guess``).
    if p0 is None:
        if param_names is None:
            raise ValueError("cannot infer number of parameters; pass p0")
        n = len(param_names)
        auto_p0 = True
    else:
        p0 = _to_array(p0).ravel()
        n = p0.size
        auto_p0 = False
        if param_names is not None and n != len(param_names):
            raise ValueError(
                f"p0 has {n} value(s) but the model takes {len(param_names)} "
                f"parameter(s) ({', '.join(param_names)}); pass one start per "
                f"parameter"
            )

    # --- weights -------------------------------------------------------
    if sigma is None:
        w = np.ones(m_data)
        sigma_arr = None
    else:
        sigma_arr = _to_array(sigma).ravel()
        if sigma_arr.shape != (m_data,):
            raise ValueError("sigma must be a 1-D array matching ydata")
        if not np.all(np.isfinite(sigma_arr)) or np.any(sigma_arr <= 0.0):
            raise ValueError(
                "sigma must be positive and finite (it is a per-point standard "
                "deviation, so 1/sigma is the weight)"
            )
        w = 1.0 / sigma_arr  # residual weight 1/sigma

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
    f_scale = float(f_scale)
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("f_scale must be a positive, finite scale")
    fs2 = f_scale ** 2

    # --- model in parameter-vector form --------------------------------
    def model(x, p):
        return _to_array(f(x, *np.asarray(p))).ravel()

    # Normalize bounds here (not only at solve time) so the data-driven seed
    # below can place each parameter inside the feasible box.
    lb, ub = _normalize_bounds(_normalize_bound_arg(bounds, n), n)
    if auto_p0:
        p0 = _initial_guess(model, xdata, ydata, w, lb, ub, n, loss_fn, fs2)

    model_jac, jac_exact, _ = _resolve_model_jac(f, model, jac, xdata, p0)

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

    m_con, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

    return _FitProblem(
        f=f, model=model, model_jac=model_jac, residual=residual,
        objective=objective, gradient=gradient, gn_hessian=gn_hessian,
        loss_fn=loss_fn, loss_name=loss_name, is_robust=is_robust, fs2=fs2,
        w=w, sigma=sigma_arr, lb=lb, ub=ub, n=n, param_names=param_names,
        m_con=m_con, g_combined=g_combined, jac_combined=jac_combined,
        cl=cl, cu=cu, jac_exact=jac_exact, xdata=xdata, ydata=ydata,
        m_data=m_data, x0=np.asarray(p0, dtype=float),
        bounds=bounds, constraints=constraints,
    )


def _solve_fit(
    prob: _FitProblem, p0, *, absolute_sigma, alpha, sensitivity, options
) -> CurveFitResult:
    """Run one pounce solve from ``p0`` against a built ``_FitProblem`` and
    assemble the full :class:`CurveFitResult` (covariance, CIs, sensitivity)."""
    pr = prob
    problem_obj = _make_problem_obj(
        pr.objective, pr.gradient, pr.gn_hessian, pr.n, pr.m_con,
        pr.g_combined, pr.jac_combined,
    )

    from ._pounce import Problem

    problem = Problem(
        n=pr.n, m=pr.m_con, problem_obj=problem_obj,
        lb=pr.lb, ub=pr.ub, cl=pr.cl, cu=pr.cu,
    )
    problem.add_option("print_level", 0)
    problem.add_option("sb", "yes")
    # With exact derivatives (analytic/JAX) the IPM converges cleanly with no
    # NLP scaling; keep it off so the converged factor matches the natural
    # problem exactly. Since pounce#128 this is a preference, not a
    # correctness requirement: held-factor back-solves (``Solver.kkt_solve``)
    # undo any active NLP scaling, so a user who explicitly turns scaling on
    # still gets natural-units covariance from the factor path.
    user_opts = dict(options or {})
    if pr.jac_exact and "nlp_scaling_method" not in user_opts:
        problem.add_option("nlp_scaling_method", "none")
    for k, v in user_opts.items():
        problem.add_option(k, v)

    solver = Solver(problem)
    popt, info = solver.solve(x0=np.asarray(p0, dtype=float))
    popt = np.asarray(popt)
    success = int(info["status"]) == 0
    # The converged factor is trustworthy when the derivatives that built it
    # were exact and the solve actually held a factor. (Scaling state no
    # longer enters: factor back-solves are scaling-corrected, pounce#128.)
    factor_ok = bool(pr.jac_exact and solver.converged)

    # --- residual diagnostics (unweighted, for reporting) --------------
    # Both paths produce the same scalar sums; streaming accumulates them over a
    # data pass (re-using the cached pass at ``popt``) rather than materialising
    # the residual vector, so ``resid`` is None and is not returned.
    if pr.streaming:
        sp = pr.closures._ensure(popt)
        n_data = sp.n_data
        sse = sp.sse
        chi2 = sp.chi2
        mae = sp.abs_sum / n_data
        # total sum of squares via the streaming variance identity
        ss_tot = sp.sum_y2 - (sp.sum_y * sp.sum_y) / n_data
        resid = None
    else:
        yhat = pr.model(pr.xdata, popt)
        resid = pr.ydata - yhat
        rw = pr.residual(popt)  # weighted
        sse = float(resid @ resid)
        chi2 = float(rw @ rw)
        mae = float(np.mean(np.abs(resid)))
        ss_tot = float(np.sum((pr.ydata - pr.ydata.mean()) ** 2))
        n_data = pr.m_data
    # Honest degrees of freedom. When the fit is exactly- or under-determined
    # (n_data <= n_params) there is no residual variance to estimate, so the
    # reduced chi-square, covariance scale, and confidence intervals are not
    # defined. Report the true (possibly <= 0) dof and let s^2 -> inf carry that
    # through to inf standard errors / CIs, rather than silently clamping dof to
    # 1 and handing back finite-but-meaningless uncertainties.
    dof = n_data - pr.n
    if dof <= 0:
        import warnings

        warnings.warn(
            f"pounce.curve_fit: non-positive degrees of freedom "
            f"(n_data={n_data} <= n_params={pr.n}); reduced chi-square, "
            f"covariance, standard errors, and confidence intervals are not "
            f"well-defined and should not be trusted (the relative-sigma "
            f"covariance is reported as inf).",
            stacklevel=2,
        )
        reduced_chi2 = float("inf")
    else:
        reduced_chi2 = chi2 / dof
    rmse = math.sqrt(sse / n_data)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = (
        1.0 - (1.0 - r2) * (n_data - 1) / dof
        if (ss_tot > 0 and dof > 0)
        else float("nan")
    )

    # scale factor s^2 (scipy's absolute_sigma rule). With dof <= 0 and relative
    # sigma this is inf, propagating "undefined" into pcov / perr / ci.
    s2 = 1.0 if absolute_sigma else reduced_chi2

    # --- covariance ----------------------------------------------------
    active_mask = _active_bounds(popt, pr.lb, pr.ub, info)
    if pr.streaming:
        pcov, cov_source = _stream_covariance(
            solver, popt, pr.data_source, pr.model, pr.model_jac, pr.loss_fn,
            pr.fs2, s2, pr.is_robust, active_mask, pr.n, pr.m_con, factor_ok,
        )
    else:
        pcov, cov_source = _covariance(
            solver, popt, pr.model_jac, pr.xdata, pr.w, s2, pr.is_robust,
            pr.residual, pr.loss_fn, pr.fs2, active_mask, pr.n, pr.m_con, factor_ok,
        )
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))
    # Use max(dof, 1) for the quantile so the t-value stays finite; when dof <= 0
    # the undefined-ness is carried by perr (inf), keeping the CI inf rather than
    # NaN (scipy's t.ppf returns NaN for dof <= 0).
    tval = _t_ppf(1.0 - alpha / 2.0, max(dof, 1))
    ci = np.column_stack([popt - tval * perr, popt + tval * perr])

    # --- sensitivity dpopt/ddata --------------------------------------
    # dpopt/ddata is (n_params x n_data) -- the size of the data -- so it is not
    # offered in streaming mode (see ``curve_fit_streaming``).
    dpopt = None
    if sensitivity and not pr.streaming:
        dpopt = _data_sensitivity(
            solver, pr.model_jac, pr.xdata, pr.w, pr.m_con, popt, factor_ok
        )

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
        n_data=n_data,
        n_params=pr.n,
        alpha=alpha,
        loss=pr.loss_name,
        success=success,
        status=int(info["status"]),
        message=str(info["status_msg"]),
        nit=int(info["iter_count"]),
        cov_source=cov_source,
        optimize_result=info,
        param_names=pr.param_names,
        active_mask=active_mask,
        dpopt_ddata=dpopt,
        _model=pr.model,
        _model_jac=pr.model_jac,
        _s2=s2,
        _sigma=pr.sigma,
        _xdata=pr.xdata,
    )


def _minima_bounds(bounds, n):
    """Translate curve_fit-style ``bounds`` into the per-parameter ``(lo, hi)``
    list :func:`find_minima` expects, mapping infinite limits to ``None``
    (unbounded) so a fully finite box is recognised as a sampling box."""
    pairs = _normalize_bound_arg(bounds, n)
    if pairs is None:
        return None
    out = []
    for bd in pairs:
        if bd is None:
            out.append(None)
            continue
        lo, hi = bd
        lo = None if (lo is None or not np.isfinite(lo)) else float(lo)
        hi = None if (hi is None or not np.isfinite(hi)) else float(hi)
        out.append((lo, hi))
    return out


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

    ``p0`` is optional: when omitted, the number of parameters is read from the
    signature of ``f`` and the starting point is chosen data-drivenly (a
    bound-aware, data-scale sweep scored by the objective; see
    :func:`_initial_guess`) rather than defaulting to ones.

    Returns
    -------
    CurveFitResult
    """
    prob = _build_fit_problem(
        f, xdata, ydata, p0, sigma, bounds, constraints, loss, f_scale, jac
    )
    return _solve_fit(
        prob, prob.x0,
        absolute_sigma=absolute_sigma, alpha=alpha,
        sensitivity=sensitivity, options=options,
    )


# --------------------------------------------------------------------------
# Out-of-core / mini-batch streaming
#
# The solver only ever asks the Python side for three things per iteration --
# objective, gradient, and the Gauss-Newton Hessian -- and all three are
# *additive sums over data points* (see the closures in ``_build_fit_problem``).
# So they can be computed by streaming mini-batches through and accumulating the
# sums, never materialising the full dataset. The result is exact (identical to
# the in-memory fit), at the cost of one pass over the data per solver iteration.
# --------------------------------------------------------------------------

def _unpack_batch(batch):
    """Normalise one streamed batch into ``(x_batch, y_batch, weight_batch)``.

    A batch is ``(x, y)`` or ``(x, y, sigma)``; ``weight = 1/sigma`` (ones when
    no sigma is given), matching the global-array weighting in
    :func:`_build_fit_problem`.
    """
    if len(batch) == 3:
        xb, yb, sb = batch
        xb = _to_array(xb)
        yb = _to_array(yb).ravel()
        sb = _to_array(sb).ravel()
        if sb.shape != yb.shape:
            raise ValueError("sigma_batch must be 1-D and match y_batch length")
        if not np.all(np.isfinite(sb)) or np.any(sb <= 0.0):
            raise ValueError(
                "sigma must be positive and finite (per-point standard deviation)"
            )
        wb = 1.0 / sb
    elif len(batch) == 2:
        xb, yb = batch
        xb = _to_array(xb)
        yb = _to_array(yb).ravel()
        wb = np.ones(yb.size)
    else:
        raise ValueError(
            "each batch must be a (x, y) or (x, y, sigma) tuple; "
            f"got a sequence of length {len(batch)}"
        )
    return xb, yb, wb


@dataclass
class _StreamPass:
    """Everything accumulated in a single streaming pass over the data at a
    fixed parameter vector: the solver quantities (objective/gradient/GN-Hessian)
    plus the running sums needed for the goodness-of-fit diagnostics."""

    obj: float
    grad: np.ndarray
    H: np.ndarray
    n_data: int
    sse: float        # sum of unweighted squared residuals
    chi2: float       # sum of weighted squared residuals
    abs_sum: float    # sum of |unweighted residual| (for MAE)
    sum_y: float      # sum y  (for R^2 total sum of squares)
    sum_y2: float     # sum y^2


def _stream_pass(p, data_source, model, model_jac, loss_fn, fs2, n) -> _StreamPass:
    """One full pass over ``data_source`` accumulating the objective, gradient,
    Gauss-Newton Hessian, and diagnostics sums at ``p``.

    Mirrors the in-memory closures of :func:`_build_fit_problem` term by term;
    summation over batches is just the associativity of the per-point sums, so
    the result equals the full-memory computation to floating-point round-off.
    The Jacobian is computed on every pass (even for a pure objective request)
    because re-deriving it would mean a second data read -- the data pass itself,
    not the Jacobian, is the cost out of core.
    """
    p = _to_array(p)
    obj = 0.0
    grad = np.zeros(n)
    H = np.zeros((n, n))
    n_data = 0
    sse = chi2 = abs_sum = sum_y = sum_y2 = 0.0
    any_batch = False
    # Transient overflow at line-search trial points is normal and benign.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for batch in data_source():
            xb, yb, wb = _unpack_batch(batch)
            if yb.size == 0:
                continue
            any_batch = True
            mb = model(xb, p)
            r = (mb - yb) * wb
            z = (r * r) / fs2
            rho, rho1, rho2 = loss_fn(z)
            obj += float(fs2 * np.sum(rho))
            J = model_jac(xb, p)                       # (batch, n)
            grad += 2.0 * (J.T @ (rho1 * r * wb))
            weight = 2.0 * (rho1 + 2.0 * z * rho2)
            Jw = J * wb[:, None]
            H += (Jw.T * weight) @ Jw
            # diagnostics (unweighted residual + weighted chi2 + y moments)
            ru = yb - mb
            n_data += yb.size
            sse += float(ru @ ru)
            chi2 += float(r @ r)
            abs_sum += float(np.sum(np.abs(ru)))
            sum_y += float(yb.sum())
            sum_y2 += float(yb @ yb)
    if not any_batch:
        raise ValueError(
            "data_source yielded no data; need at least one non-empty batch"
        )
    return _StreamPass(obj, grad, H, n_data, sse, chi2, abs_sum, sum_y, sum_y2)


class _StreamingClosures:
    """objective/gradient/gn_hessian over a streamed dataset.

    The interior-point solver evaluates objective, then gradient, then Hessian at
    the *same* parameter vector each iteration (the Rust bridge does not forward
    its ``new_x`` flag to Python). A one-slot cache keyed on the parameter bytes
    therefore collapses those three calls into a single data pass: the first call
    streams the data and accumulates all three quantities; the other two are
    cache hits.
    """

    def __init__(self, data_source, model, model_jac, loss_fn, fs2, n):
        self._src = data_source
        self._model = model
        self._model_jac = model_jac
        self._loss_fn = loss_fn
        self._fs2 = fs2
        self._n = n
        self._key = None
        self._pass: _StreamPass | None = None

    def _ensure(self, p) -> _StreamPass:
        p = _to_array(p)
        key = p.tobytes()
        if key != self._key or self._pass is None:
            self._pass = _stream_pass(
                p, self._src, self._model, self._model_jac,
                self._loss_fn, self._fs2, self._n,
            )
            self._key = key
        return self._pass

    def objective(self, p):
        return self._ensure(p).obj

    def gradient(self, p):
        return self._ensure(p).grad

    def gn_hessian(self, p):
        return self._ensure(p).H


def _stream_covariance(
    solver, popt, data_source, model, model_jac, loss_fn, fs2,
    s2, is_robust, active_mask, n, m_con, factor_ok,
):
    """Streaming counterpart of :func:`_covariance`.

    The interior least-squares optimum with exact derivatives reads the
    covariance straight from the converged KKT factor -- *data-free*. Otherwise
    the Gauss-Newton ``M = sum (w g)(w g)^T`` (and, for robust losses, the
    sandwich pieces ``A``/``B``) are accumulated over one data pass and fed
    through the identical algebra as :func:`_covariance`.
    """
    has_active = m_con > 0 or (active_mask is not None and active_mask.any())

    # Interior, exact-derivative least-squares optimum: no data needed.
    if not is_robust and not has_active and factor_ok:
        inv_hs = _inv_hessian_from_factor(solver, n, m_con)
        if inv_hs is not None:
            return 2.0 * s2 * inv_hs, "reduced_hessian"

    M = np.zeros((n, n))
    A = np.zeros((n, n)) if is_robust else None
    B = np.zeros((n, n)) if is_robust else None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for batch in data_source():
            xb, yb, wb = _unpack_batch(batch)
            if yb.size == 0:
                continue
            J = model_jac(xb, popt)
            Jw = J * wb[:, None]
            M += Jw.T @ Jw
            if is_robust:
                r = (model(xb, popt) - yb) * wb
                z = (r * r) / fs2
                _, rho1, rho2 = loss_fn(z)
                A += (Jw.T * (rho1 + 2.0 * z * rho2)) @ Jw
                score = (rho1 * r)[:, None] * Jw
                B += score.T @ score

    if is_robust:
        Ainv = np.linalg.pinv(A)
        return s2 * (Ainv @ B @ Ainv), "sandwich"
    if has_active:
        free = ~active_mask if active_mask is not None else np.ones(n, bool)
        cov = np.zeros((n, n))
        cov[np.ix_(free, free)] = s2 * np.linalg.pinv(M[np.ix_(free, free)])
        return cov, "reduced_hessian(projected)"
    return s2 * np.linalg.pinv(M), "jacobian"


def _build_streaming_fit_problem(
    f, data_source, p0, n_params, bounds, constraints,
    loss, f_scale, jac, param_names,
) -> _FitProblem:
    """Assemble a :class:`_FitProblem` whose objective/gradient/Hessian stream
    over ``data_source`` instead of capturing in-memory arrays. ``sigma`` is
    supplied per-batch inside the data tuples, not here."""
    if not callable(data_source):
        raise ValueError(
            "data_source must be a zero-argument callable returning a fresh "
            "iterator of (x_batch, y_batch[, sigma_batch]) tuples (it is "
            "re-read once per solver iteration)"
        )

    inferred = param_names if param_names is not None else _infer_param_names(f)

    # --- parameter count + starting point ------------------------------
    if p0 is None:
        if n_params is None and inferred is None:
            raise ValueError(
                "cannot infer the number of parameters; pass p0 (a full "
                "starting vector) or n_params"
            )
        n = int(n_params) if n_params is not None else len(inferred)
        auto_p0 = True
    else:
        p0 = _to_array(p0).ravel()
        n = p0.size
        auto_p0 = False
        if n_params is not None and n_params != n:
            raise ValueError(f"p0 has {n} value(s) but n_params={n_params}")
        if inferred is not None and n != len(inferred):
            raise ValueError(
                f"p0 has {n} value(s) but the model takes {len(inferred)} "
                f"parameter(s) ({', '.join(inferred)})"
            )

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
    f_scale = float(f_scale)
    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("f_scale must be a positive, finite scale")
    fs2 = f_scale ** 2

    def model(x, p):
        return _to_array(f(x, *np.asarray(p))).ravel()

    lb, ub = _normalize_bounds(_normalize_bound_arg(bounds, n), n)
    if auto_p0:
        # No full-data pass is available for the data-driven _initial_guess, so
        # fall back to ones clipped strictly inside the box. Passing p0 is
        # strongly recommended for a streaming fit.
        p0 = np.ones(n)
        if lb is not None:
            p0 = np.where(np.isfinite(lb), np.maximum(p0, lb), p0)
        if ub is not None:
            p0 = np.where(np.isfinite(ub), np.minimum(p0, ub), p0)

    # Probe one batch (a throwaway fresh iterator) to trace the Jacobian. The
    # JAX path retraces once if a later batch has a different shape, which is
    # correct -- uniform batch sizes simply avoid the extra trace.
    try:
        first = next(iter(data_source()))
    except StopIteration:
        raise ValueError("data_source yielded no batches") from None
    probe_x, _probe_y, _ = _unpack_batch(first)
    model_jac, jac_exact, _ = _resolve_model_jac(f, model, jac, probe_x, p0)

    closures = _StreamingClosures(data_source, model, model_jac, loss_fn, fs2, n)
    m_con, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

    return _FitProblem(
        f=f, model=model, model_jac=model_jac, residual=None,
        objective=closures.objective, gradient=closures.gradient,
        gn_hessian=closures.gn_hessian,
        loss_fn=loss_fn, loss_name=loss_name, is_robust=is_robust, fs2=fs2,
        w=None, sigma=None, lb=lb, ub=ub, n=n, param_names=inferred,
        m_con=m_con, g_combined=g_combined, jac_combined=jac_combined,
        cl=cl, cu=cu, jac_exact=jac_exact, xdata=None, ydata=None,
        m_data=0, x0=np.asarray(p0, dtype=float),
        bounds=bounds, constraints=constraints,
        streaming=True, data_source=data_source, closures=closures,
    )


def curve_fit_streaming(
    f: Callable,
    data_source: Callable[[], Any],
    p0=None,
    *,
    n_params: int | None = None,
    absolute_sigma: bool = False,
    bounds: Sequence | tuple = (-np.inf, np.inf),
    constraints: Sequence | Mapping | None = None,
    loss: str | Callable = "sse",
    f_scale: float = 1.0,
    jac: Callable | str | None = None,
    alpha: float = 0.05,
    options: Mapping[str, Any] | None = None,
    param_names: list[str] | None = None,
) -> CurveFitResult:
    """Out-of-core :func:`curve_fit` for datasets too large to hold in memory.

    Identical model and objective to :func:`curve_fit`, but the data is read in
    mini-batches from ``data_source`` instead of as in-memory arrays. Because the
    solver's objective, gradient, and Gauss-Newton Hessian are additive sums over
    data points, streaming and accumulating them yields the *exact* same fit as
    the in-memory call -- only one batch (plus an ``n_params x n_params`` matrix)
    is ever resident. The cost is one pass over the data per solver iteration
    (~10-50), so ``data_source`` must be re-readable.

    Parameters
    ----------
    f
        Model ``f(x, *params)`` (write it with ``jax.numpy`` for exact
        derivatives; otherwise pass an analytic ``jac`` or ``jac="fd"``).
    data_source
        A **zero-argument callable** returning a *fresh* iterator of
        ``(x_batch, y_batch)`` or ``(x_batch, y_batch, sigma_batch)`` tuples.
        It is called once per solver pass, so it must yield the full dataset
        each time (e.g. re-open an ``mmap``/HDF5 file and slice it). Uniform
        batch sizes avoid an extra JAX retrace on a smaller final batch.
    p0
        Starting parameter vector. **Strongly recommended** -- with only
        ``n_params`` the seed falls back to ones clipped into ``bounds`` (the
        data-driven seed used by :func:`curve_fit` needs a full in-memory pass).
    n_params
        Number of parameters, required if ``p0`` is omitted and the model
        signature does not name them.

    Notes
    -----
    ``residuals`` (length ``n_data``) and the parametric sensitivity
    ``dpopt_ddata`` (shape ``n_params x n_data``) are **not** returned -- they
    are the size of the data and would defeat the purpose; both fields are
    ``None``. All scalar diagnostics (SSE, chi-square, R^2, dof) and the
    covariance / standard errors / confidence intervals are computed and are
    identical to the in-memory fit. ``confidence_band`` works for new ``x`` but
    falls back to a homoscedastic noise level (the per-point ``sigma`` is not
    retained).

    Returns
    -------
    CurveFitResult
    """
    prob = _build_streaming_fit_problem(
        f, data_source, p0, n_params, bounds, constraints,
        loss, f_scale, jac, param_names,
    )
    return _solve_fit(
        prob, prob.x0,
        absolute_sigma=absolute_sigma, alpha=alpha,
        sensitivity=False, options=options,
    )


def curve_fit_minima(
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
    method: str = "multistart",
    n_minima: int = 10,
    max_solves: int | None = None,
    patience: int = 8,
    dedup: float = 1e-4,
    seed: int | None = None,
    find_minima_kw: Mapping[str, Any] | None = None,
) -> list[CurveFitResult]:
    """Find **multiple** parameter sets that fit ``f`` to ``(xdata, ydata)``.

    Nonlinear least squares is generally non-convex, so the objective
    :func:`curve_fit` minimizes can have several local minima — distinct
    parameter sets that each explain the data (peak-assignment ambiguity,
    frequency aliasing in sinusoids, amplitude/decay trade-offs in sums of
    exponentials, sign/label symmetry, …). This drives
    :func:`pounce.find_minima` over *exactly* that objective — the same
    weighting (``sigma``), robust ``loss``, ``f_scale``, ``constraints``, and
    resolved Jacobian as :func:`curve_fit` — to enumerate the distinct minima,
    then refines each into a full :class:`CurveFitResult` (covariance,
    confidence intervals, optional ``dpopt/ddata``).

    The model Jacobian is reused as the search **gradient** and the
    Gauss-Newton matrix as the search **Hessian**, which both sharpens the
    basin escapes and lets ``find_minima`` certify each point as a true minimum
    (rejecting saddles) before recording it.

    Finite ``bounds`` are strongly recommended: they define the box that
    ``find_minima`` samples / repels within. With the default unbounded box the
    search degrades to jittered restarts around the (data-driven) seed.

    Parameters
    ----------
    f, xdata, ydata, p0, sigma, absolute_sigma, bounds, constraints, loss, \
f_scale, jac, alpha, sensitivity, options
        As in :func:`curve_fit`. ``p0`` (or the data-driven default) is used as
        the search's initial point.
    method, n_minima, max_solves, patience, dedup, seed
        Forwarded to :func:`pounce.find_minima` — the enumeration strategy
        (``"multistart"`` | ``"deflation"`` | ``"flooding"`` | ``"tunneling"``
        | ``"mlsl"`` | ``"basinhopping"``), the target count, the solve budget,
        the give-up patience, the de-duplication distance, and the RNG seed.
    find_minima_kw
        Extra keyword arguments forwarded to :func:`find_minima` (e.g.
        ``strategy_kw`` for per-method knobs, ``psd_tol``, ``distance``).

    Returns
    -------
    list[CurveFitResult]
        One fully-populated result per distinct parameter set, ranked by SSE
        (best first). May be empty if no minimum was found, and may contain
        fewer than ``n_minima`` entries when the landscape has fewer minima.

    See Also
    --------
    curve_fit : single fit from one starting point.
    pounce.find_minima : the underlying multi-minimum search.
    """
    from ._minima import find_minima

    prob = _build_fit_problem(
        f, xdata, ydata, p0, sigma, bounds, constraints, loss, f_scale, jac
    )

    # ``find_minima_kw`` is for knobs without a dedicated parameter here
    # (``strategy_kw``, ``psd_tol``, ``distance``, …). Reject keys that collide
    # with the arguments we already forward, so the failure is a clear message
    # rather than a cryptic duplicate-keyword ``TypeError``.
    extra_kw = dict(find_minima_kw or {})
    forwarded = {
        "method", "jac", "hess", "bounds", "constraints", "n_minima",
        "max_solves", "patience", "dedup", "seed", "options",
    }
    clash = forwarded & extra_kw.keys()
    if clash:
        raise TypeError(
            "find_minima_kw must not contain "
            f"{sorted(clash)}; pass {'them' if len(clash) > 1 else 'it'} as the "
            "dedicated curve_fit_minima argument(s) instead."
        )

    # Keep the inner local solves quiet by default; the user's options win.
    search_options = {"print_level": 0, "sb": "yes"}
    search_options.update(options or {})

    res = find_minima(
        prob.objective, prob.x0,
        method=method,
        jac=prob.gradient,
        hess=prob.gn_hessian,
        bounds=_minima_bounds(prob.bounds, prob.n),
        constraints=prob.constraints,
        n_minima=n_minima,
        max_solves=max_solves,
        patience=patience,
        dedup=dedup,
        seed=seed,
        options=search_options,
        **extra_kw,
    )

    # Refine every distinct minimum into a full CurveFitResult, then rank by
    # goodness of fit so the best explanation comes first.
    fits = [
        _solve_fit(
            prob, p,
            absolute_sigma=absolute_sigma, alpha=alpha,
            sensitivity=sensitivity, options=options,
        )
        for p in res.minima
    ]
    fits.sort(key=lambda r: r.sse)
    return fits


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
        lo_a, hi_a = _to_array(lo), _to_array(hi)
        for side, arr in (("lower", lo_a), ("upper", hi_a)):
            if arr.ndim > 1 or (arr.ndim == 1 and arr.size not in (1, n)):
                raise ValueError(
                    f"bounds {side} has length {arr.size} but the problem has "
                    f"{n} parameter(s); pass a scalar or a length-{n} array"
                )
        lo = np.broadcast_to(lo_a, (n,))
        hi = np.broadcast_to(hi_a, (n,))
        return list(zip(lo.tolist(), hi.tolist()))
    return bounds  # already a per-parameter list of pairs (validated downstream)


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
    trustworthy (exact derivatives) this is read pounce-natively from the
    inverse-Hessian block of ``K`` (``pcov = 2 s^2 inv(H_S)``, no explicit
    matrix inverse; the back-solve is scaling-corrected, pounce#128);
    otherwise it is formed from the model Jacobian, which is
    scaling-independent and gives the identical value. Robust losses
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
