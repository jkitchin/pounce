"""Find *multiple* local minima — a global-search companion to ``minimize``.

``pounce.minimize`` finds one local minimum from a starting point.
``pounce.find_minima`` instead sweeps a landscape for *many* distinct
minima (or the global one among them) by driving the same local solver in a
loop. The loop is one engine; the way each iteration escapes an
already-found minimum is a pluggable ``method``:

Repulsion — transform the problem so the solver can't settle where it
already has, then re-solve:

* ``"flooding"``  — add a repulsive Gaussian bump to the objective at each
  found minimum (filled-function / metadynamics; Ge 1990, Laio &
  Parrinello 2002).
* ``"deflation"`` — add a singular ``1/‖x−x*‖^p`` pole penalty at each found
  minimum (deflation lineage; Brown & Gearhart 1971, Farrell et al. 2015).
* ``"tunneling"`` — between descents, seek an equal-height point away from
  known minima, then descend (Levy & Montalvo 1985).

Restart — leave the problem alone, just choose the next start cleverly:

* ``"multistart"`` — random / Sobol box sampling.
* ``"mlsl"`` — Multi-Level Single Linkage clustering (Rinnooy Kan &
  Timmer 1987): start a local solve from a sample only when no better
  sample / found minimum is nearby, so each basin is descended ~once.

Hopping — a Markov chain over minima:

* ``"basinhopping"`` — perturb the current minimum, locally minimize, accept
  by Metropolis on the objective (Li & Scheraga 1987, Wales & Doye 1997).

All methods reuse :func:`pounce.minimize`, so bounds and constraints carry
through, and the acceptance test (polish on the clean objective, reject
saddles via the Hessian) is shared. See ``docs/src/find-minima.md`` and the
notebooks ``python/notebooks/15–17`` for walkthroughs and method selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._minimize import minimize, OptimizeResult

__all__ = ["find_minima", "MinimaResult"]


# --------------------------------------------------------------------------
# Repulsion kernels (analytic value / gradient / Hessian).
# --------------------------------------------------------------------------
def _gauss_terms(x, centers, amplitude, sigma):
    """Σ A·exp(−½‖(x−c)/σ‖²) and its gradient / Hessian contributions.

    ``sigma`` is a per-dimension width vector (anisotropic Gaussian).
    ``amplitude`` may be a scalar (shared) or one value per center.
    """
    n = x.size
    val = 0.0
    grad = np.zeros(n)
    hess = np.zeros((n, n))
    m = 1.0 / (np.asarray(sigma, float) ** 2)   # per-dimension 1/σ²
    amps = np.atleast_1d(np.asarray(amplitude, float))
    if amps.size == 1:
        amps = np.full(len(centers), float(amps[0]))
    for c, A in zip(centers, amps):
        d = x - c
        g = A * np.exp(-0.5 * float(np.sum(m * d * d)))
        md = m * d
        val += g
        grad += -g * md
        hess += g * (np.outer(md, md) - np.diag(m))
    return val, grad, hess


def _auto_amplitude(hess, x, sigma, margin, floor=1e-12):
    """Curvature-based bump height for escaping the minimum at ``x``.

    The bump turns the minimum into a saddle once its height exceeds the
    smallest generalized eigenvalue ``μ_min`` of ``H v = μ·diag(1/σ²) v`` —
    i.e. of ``S = diag(σ)·H·diag(σ)``. Returns ``margin · μ_min`` (a few ×
    over threshold), or ``None`` when no Hessian is available.
    """
    if hess is None:
        return None
    H = np.asarray(hess(x), float)
    H = 0.5 * (H + H.T)
    s = np.asarray(sigma, float)
    S = s[:, None] * H * s[None, :]
    mu_min = float(np.linalg.eigvalsh(S)[0])
    return margin * max(mu_min, floor)


def _pole_terms(x, centers, eta, power, soft, length):
    """Σ η·(‖(x−c)/ℓ‖²+soft)^(−p/2) and its gradient / Hessian contributions.

    ``length`` is a per-dimension scale vector, so the pole is anisotropic.
    A softened singular pole: huge near a found minimum, decaying like 1/r^p
    away from it; ``soft`` keeps everything finite for the solver.
    """
    n = x.size
    q = power / 2.0
    val = 0.0
    grad = np.zeros(n)
    hess = np.zeros((n, n))
    m = 1.0 / (np.asarray(length, float) ** 2)   # per-dimension 1/ℓ²
    for c in centers:
        d = x - c
        r2 = float(np.sum(m * d * d)) + soft
        p = m * d
        val += eta * r2 ** (-q)
        coef1 = -2.0 * q * eta * r2 ** (-q - 1.0)
        grad += coef1 * p
        hess += (4.0 * q * (q + 1.0) * eta * r2 ** (-q - 2.0)) * np.outer(p, p) \
            - 2.0 * q * eta * r2 ** (-q - 1.0) * np.diag(m)
    return val, grad, hess


# --------------------------------------------------------------------------
# Per-dimension length scales (anisotropic widths + scaled dedup metric).
# --------------------------------------------------------------------------
def _scale_from_bounds(bounds, n):
    """Per-dimension length scale L (box width per variable) and whether a
    finite box was available."""
    if bounds is not None and all(
        b is not None and b[0] is not None and b[1] is not None for b in bounds
    ):
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
        L = hi - lo
        L[L <= 0] = 1.0
        return L, True
    return np.ones(n), False


def _resolve_lengths(spec, L, has_box, frac, fallback):
    """Turn a width spec into a per-dimension vector.

    ``"auto"`` -> ``frac * L`` (a fraction of each variable's range) when a
    box is known, else ``fallback``; a scalar -> isotropic; a vector -> used
    as given.
    """
    n = L.size
    if isinstance(spec, str) and spec == "auto":
        return frac * L if has_box else np.full(n, fallback)
    arr = np.asarray(spec, float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


# --------------------------------------------------------------------------
# Archive of accepted minima.
# --------------------------------------------------------------------------
class MinimaArchive:
    """Accepted minima plus dedup. ``distance`` is pluggable (PBC etc.)."""

    def __init__(self, dedup, distance=None):
        self.dedup = float(dedup)
        self.distance = distance or (lambda a, b: float(np.linalg.norm(a - b)))
        self.xs: list[np.ndarray] = []
        self.fs: list[float] = []
        self.results: list[OptimizeResult] = []

    @property
    def centers(self):
        return self.xs

    def is_known(self, x):
        return any(self.distance(x, m) <= self.dedup for m in self.xs)

    def near_any(self, x, radius):
        return any(self.distance(x, m) <= radius for m in self.xs)

    def add(self, x, f, result):
        self.xs.append(np.array(x, dtype=float))
        self.fs.append(float(f))
        self.results.append(result)

    def __len__(self):
        return len(self.xs)


# --------------------------------------------------------------------------
# Result object.
# --------------------------------------------------------------------------
@dataclass
class MinimaResult:
    """Outcome of :func:`find_minima`, sorted by objective (lowest first)."""

    minima: list[np.ndarray]
    values: list[float]
    results: list[OptimizeResult]
    status: str
    n_solves: int
    trace: list[dict] = field(default_factory=list)

    @property
    def x(self):
        return self.minima[0] if self.minima else None

    @property
    def fun(self):
        return self.values[0] if self.values else None

    def __len__(self):
        return len(self.minima)


# --------------------------------------------------------------------------
# Driver internals.
# --------------------------------------------------------------------------
class _Stop(Exception):
    def __init__(self, status):
        self.status = status


class _Context:
    """Holds the clean problem and the shared solve / polish / verify ops."""

    def __init__(self, fun, jac, hess, bounds, constraints, options, psd_tol):
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.bounds = bounds
        self.constraints = constraints
        self.options = options
        self.psd_tol = psd_tol
        self.n_solves = 0
        self.max_solves = None

    def solve(self, fun, x0, jac=None, hess=None):
        if self.max_solves is not None and self.n_solves >= self.max_solves:
            raise _Stop("budget_exhausted")
        self.n_solves += 1
        return minimize(
            fun, x0, jac=jac, hess=hess,
            bounds=self.bounds, constraints=self.constraints,
            **(self.options or {}),
        )

    def in_bounds(self, x):
        if self.bounds is None:
            return True
        for xi, bd in zip(x, self.bounds):
            if bd is None:
                continue
            lo, hi = bd
            if lo is not None and xi < lo - 1e-9:
                return False
            if hi is not None and xi > hi + 1e-9:
                return False
        return True

    def is_minimum(self, x):
        """Reject saddles/maxima via the clean Hessian (when available)."""
        if self.hess is None:
            return True
        H = np.asarray(self.hess(x), dtype=float)
        H = 0.5 * (H + H.T)
        return float(np.linalg.eigvalsh(H)[0]) >= -self.psd_tol


class _State:
    """Acceptance bookkeeping + termination."""

    def __init__(self, ctx, archive, n_minima, patience, callback):
        self.ctx = ctx
        self.archive = archive
        self.n_minima = n_minima
        self.patience = patience
        self.callback = callback
        self.stagnant = 0
        self.trace: list[dict] = []

    def consider(self, x, success, polish):
        ctx = self.ctx
        cand = np.asarray(x, dtype=float)
        if success and polish:
            res_p = ctx.solve(ctx.fun, cand, jac=ctx.jac, hess=ctx.hess)
            success = res_p.success
            cand = np.asarray(res_p.x, dtype=float)
        else:
            res_p = None

        fval = float(ctx.fun(cand))
        # Reject non-finite candidates/values: a NaN point would never be
        # "known" (NaN distance never compares <= dedup) and would flood the
        # archive with duplicates; NaN also breaks the Hessian eigendecomp.
        finite = bool(np.all(np.isfinite(cand)) and np.isfinite(fval))
        accepted = (
            success
            and finite
            and ctx.in_bounds(cand)
            and ctx.is_minimum(cand)
            and not self.archive.is_known(cand)
        )
        self.trace.append({
            "x": cand, "f": fval, "success": bool(success),
            "accepted": accepted, "n_found": len(self.archive),
        })

        if accepted:
            result = res_p if res_p is not None else _mk_result(cand, fval)
            self.archive.add(cand, fval, result)
            self.stagnant = 0
            if self.callback is not None:
                self.callback(cand, fval)
            if len(self.archive) >= self.n_minima:
                raise _Stop("target_reached")
        else:
            self.stagnant += 1
            if self.stagnant >= self.patience:
                raise _Stop("converged")
        return accepted


def _mk_result(x, f):
    return OptimizeResult(
        x=np.asarray(x), fun=float(f), success=True, status=0,
        message="recorded", nit=0, info={},
    )


# --------------------------------------------------------------------------
# Start-point helpers.
# --------------------------------------------------------------------------
def _box(bounds):
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return lo, hi


def _has_box(bounds):
    return bounds is not None and all(
        b is not None and b[0] is not None and b[1] is not None
        for b in bounds
    )


def _sample(bounds, x0, rng, jitter, sobol=None):
    """Draw a fresh start: Sobol/uniform in the box, else jitter around x0."""
    if _has_box(bounds):
        lo, hi = _box(bounds)
        if sobol is not None:
            u = sobol.random(1)[0]
        else:
            u = rng.random(x0.shape)
        return lo + (hi - lo) * u
    return x0 + jitter * rng.standard_normal(x0.shape)


def _make_sobol(n, seed, enabled):
    if not enabled:
        return None
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d=n, scramble=True, seed=seed)
    except Exception:
        return None


def _clip(x, bounds):
    if not _has_box(bounds):
        return x
    lo, hi = _box(bounds)
    return np.clip(x, lo, hi)


# --------------------------------------------------------------------------
# Strategy loops (each runs until a _Stop is raised).
# --------------------------------------------------------------------------
def _augment(ctx, kernel):
    """Build (fun, jac, hess) = clean + repulsion kernel at archive centers."""
    fun, jac, hess = ctx.fun, ctx.jac, ctx.hess

    def f2(x):
        return float(fun(x)) + kernel(x)[0]

    j2 = None
    if jac is not None:
        def j2(x):
            return np.asarray(jac(x), float).ravel() + kernel(x)[1]

    h2 = None
    if hess is not None:
        def h2(x):
            return np.asarray(hess(x), float) + kernel(x)[2]

    return f2, j2, h2


def _run_flooding(ctx, state, x0, rng, kw):
    L, has_box = kw["_L"], kw["_has_box"]
    sigma = _resolve_lengths(kw.get("sigma", "auto"), L, has_box,
                             frac=kw.get("sigma_frac", 0.1), fallback=0.5)
    amp_spec = kw.get("amplitude", "auto")
    margin = kw.get("amp_margin", 2.0)
    bump_factor = kw.get("amp_bump", 3.0)
    bump_cap = kw.get("amp_bump_cap", 1e3)
    fallback_amp = kw.get("amp_fallback", 2.0)
    jitter = kw.get("restart_jitter", 0.5)
    sobol = _make_sobol(x0.size, kw.get("seed"), kw.get("sobol", True))

    base_amp: list[float] = []   # per-center height
    mult: list[float] = []       # per-center adaptive multiplier
    start = x0.copy()
    last_center = None           # the basin we are trying to escape from
    fails = 0
    while True:
        centers = list(state.archive.centers)
        while len(base_amp) < len(centers):
            k = len(base_amp)
            if amp_spec == "auto":
                a = _auto_amplitude(ctx.hess, centers[k], sigma, margin)
                base_amp.append(fallback_amp if a is None else a)
            else:
                base_amp.append(float(amp_spec))
            mult.append(1.0)
        eff = [base_amp[k] * mult[k] for k in range(len(centers))]
        kernel = lambda x, _c=centers, _a=eff: _gauss_terms(x, _c, _a, sigma)
        f2, j2, h2 = _augment(ctx, kernel)
        res = ctx.solve(f2, start, j2, h2)
        if state.consider(res.x, res.success, polish=bool(centers)):
            start = state.archive.xs[-1].copy()
            last_center = len(state.archive.xs) - 1
            fails = 0
        elif last_center is not None and mult[last_center] < bump_cap and fails < 8:
            # Under-flooded the basin we started from: raise its bump and retry.
            mult[last_center] *= bump_factor
            start = centers[last_center] + 0.05 * sigma * rng.standard_normal(x0.shape)
            fails += 1
            continue
        else:
            start = _sample(ctx.bounds, x0, rng, jitter, sobol)
            last_center = None
            fails = 0


def _run_deflation(ctx, state, x0, rng, kw):
    L, has_box = kw["_L"], kw["_has_box"]
    eta = kw.get("eta", 1.0)
    power = kw.get("power", 2.0)
    soft = kw.get("soft", 1e-3)
    length = _resolve_lengths(kw.get("length", "auto"), L, has_box,
                              frac=kw.get("length_frac", 0.1), fallback=0.5)
    jitter = kw.get("restart_jitter", 0.5)
    sobol = _make_sobol(x0.size, kw.get("seed"), kw.get("sobol", True))
    start = x0.copy()
    while True:
        centers = list(state.archive.centers)
        kernel = lambda x: _pole_terms(x, centers, eta, power, soft, length)
        f2, j2, h2 = _augment(ctx, kernel)
        # Start a small step off the pole so the first gradient is finite.
        s = start
        if centers and state.archive.is_known(s):
            s = s + 0.1 * length * rng.standard_normal(s.shape)
        res = ctx.solve(f2, s, j2, h2)
        if state.consider(res.x, res.success, polish=bool(centers)):
            start = state.archive.xs[-1].copy()
        else:
            start = _sample(ctx.bounds, x0, rng, jitter, sobol)


def _run_tunneling(ctx, state, x0, rng, kw):
    L, has_box = kw["_L"], kw["_has_box"]
    eta = kw.get("eta", 1.0)
    power = kw.get("power", 2.0)
    soft = kw.get("soft", 1e-3)
    length = _resolve_lengths(kw.get("length", "auto"), L, has_box,
                              frac=kw.get("length_frac", 0.1), fallback=0.5)
    jitter = kw.get("restart_jitter", 0.75)
    # Seed: one clean descent.
    res = ctx.solve(ctx.fun, x0, ctx.jac, ctx.hess)
    state.consider(res.x, res.success, polish=False)
    while True:
        centers = list(state.archive.centers)
        # Tunnel at the height of the most-recently found minimum, away from
        # all known minima — the classic monotone-descending tunnel.
        f_ref = state.archive.fs[-1] if state.archive.fs else float(ctx.fun(x0))

        def T(x, _c=centers, _r=f_ref):
            pole = _pole_terms(x, _c, eta, power, soft, length)[0]
            return (float(ctx.fun(x)) - _r) ** 2 + pole

        # Start by stepping outward from the latest minimum.
        anchor = state.archive.xs[-1] if state.archive.xs else x0
        start = _clip(anchor + jitter * rng.standard_normal(anchor.shape),
                      ctx.bounds)
        res = ctx.solve(T, start)  # FD gradient; L-BFGS
        state.consider(res.x, res.success, polish=True)


def _run_multistart(ctx, state, x0, rng, kw):
    jitter = kw.get("restart_jitter", 1.0)
    sobol = _make_sobol(x0.size, kw.get("seed"), kw.get("sobol", True))
    res = ctx.solve(ctx.fun, x0, ctx.jac, ctx.hess)
    state.consider(res.x, res.success, polish=False)
    while True:
        start = _sample(ctx.bounds, x0, rng, jitter, sobol)
        res = ctx.solve(ctx.fun, start, ctx.jac, ctx.hess)
        state.consider(res.x, res.success, polish=False)


def _run_mlsl(ctx, state, x0, rng, kw):
    batch = int(kw.get("samples_per_round", 20))
    gamma = float(kw.get("gamma", 2.0))
    jitter = kw.get("restart_jitter", 1.0)
    sobol = _make_sobol(x0.size, kw.get("seed"), kw.get("sobol", True))
    n = x0.size
    pool_x: list[np.ndarray] = []
    pool_f: list[float] = []
    # Work in the per-dimension scaled metric: the box is unit width per
    # variable, so its diagonal is sqrt(n) and distances are scale-free.
    L = kw["_L"]
    sdist = lambda a, b: float(np.linalg.norm((a - b) / L))
    diag = float(np.sqrt(n))
    # Seed from x0.
    res = ctx.solve(ctx.fun, x0, ctx.jac, ctx.hess)
    state.consider(res.x, res.success, polish=False)
    while True:
        # Grow the sample pool.
        for _ in range(batch):
            s = _sample(ctx.bounds, x0, rng, jitter, sobol)
            pool_x.append(s)
            pool_f.append(float(ctx.fun(s)))
        N = len(pool_x)
        # Reduced radius shrinks as the pool grows (MLSL clustering rule).
        # Guard N>=2: log(1)=0 would give radius 0 (no clustering at all).
        Ne = max(N, 2)
        radius = gamma * diag * (np.log(Ne) / Ne) ** (1.0 / n)
        order = np.argsort(pool_f)
        for i in order:
            si, fi = pool_x[i], pool_f[i]
            # Single-linkage: skip if a *better* sample is within radius
            # (distances in the scaled metric).
            better_near = any(
                pool_f[j] < fi and sdist(si, pool_x[j]) < radius
                for j in range(N) if j != i
            )
            if better_near or state.archive.near_any(si, radius):
                continue
            res = ctx.solve(ctx.fun, si, ctx.jac, ctx.hess)
            state.consider(res.x, res.success, polish=False)


def _run_basinhopping(ctx, state, x0, rng, kw):
    step = float(kw.get("step", 0.5))
    temperature = float(kw.get("temperature", 1.0))
    res = ctx.solve(ctx.fun, x0, ctx.jac, ctx.hess)
    state.consider(res.x, res.success, polish=False)
    cur = np.asarray(res.x, float)
    cur_f = float(res.fun) if res.success else float(ctx.fun(cur))
    while True:
        trial = _clip(cur + step * rng.standard_normal(cur.shape), ctx.bounds)
        res = ctx.solve(ctx.fun, trial, ctx.jac, ctx.hess)
        if not res.success:
            state.consider(res.x, False, polish=False)
            continue
        state.consider(res.x, True, polish=False)
        new_f = float(res.fun)
        if new_f < cur_f or rng.random() < np.exp(-(new_f - cur_f) / temperature):
            cur, cur_f = np.asarray(res.x, float), new_f


_STRATEGIES = {
    "flooding": _run_flooding,
    "deflation": _run_deflation,
    "tunneling": _run_tunneling,
    "multistart": _run_multistart,
    "mlsl": _run_mlsl,
    "basinhopping": _run_basinhopping,
}


# --------------------------------------------------------------------------
# Public entry point.
# --------------------------------------------------------------------------
def find_minima(
    fun: Callable[[np.ndarray], float],
    x0,
    *,
    method: str = "deflation",
    jac: Callable | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    n_minima: int = 10,
    max_solves: int | None = None,
    patience: int = 8,
    dedup: float = 1e-4,
    psd_tol: float = 1e-6,
    options: Mapping[str, Any] | None = None,
    strategy_kw: Mapping[str, Any] | None = None,
    distance: Callable | None = None,
    seed: int | None = None,
    callback: Callable | None = None,
) -> MinimaResult:
    """Find multiple local minima of ``fun`` by driving ``minimize`` in a loop.

    Parameters mirror :func:`pounce.minimize` (``fun``, ``jac``, ``hess``,
    ``bounds``, ``constraints``, ``options``) plus search controls:

    method
        ``"deflation"`` | ``"flooding"`` | ``"tunneling"`` (repulsion);
        ``"multistart"`` | ``"mlsl"`` (restart); ``"basinhopping"``
        (hopping). See the module docstring for the references.
    n_minima
        Target: stop once this many distinct minima are found.
    max_solves
        Budget: hard cap on solver calls (default ``8 * n_minima``).
    patience
        Give-up: stop after this many solves in a row that find nothing new
        — the graceful exit when fewer minima exist than requested.
    dedup
        Two minima within this distance are treated as the same. With the
        default metric this is measured in the **per-dimension scaled** space
        (each variable divided by its bounds range), so ``dedup`` is
        scale-free.
    strategy_kw
        Method-specific knobs. Repulsion widths are **per-dimension and
        ``"auto"`` by default** — sized to a fraction (``sigma_frac`` /
        ``length_frac``, default 0.1) of each variable's bounds range, so
        disparate variable scales are handled automatically. Override with a
        scalar (isotropic) or a length-``n`` vector. The flooding
        ``amplitude`` is also ``"auto"`` by default — set per minimum from
        the local curvature (``amp_margin × μ_min``, the well-tempered
        escape height) and raised adaptively if the solver returns to a
        flooded basin — so no manual energy scale is needed (give a scalar
        to override). Examples: ``sigma``/``amplitude`` (flooding),
        ``eta``/``power``/``length`` (deflation), ``step``/``temperature``
        (basinhopping), ``samples_per_round``/``gamma`` (mlsl).
    distance
        Custom metric ``d(a, b) -> float`` for dedup (e.g. periodic boxes).
        Defaults to Euclidean distance in the per-dimension scaled space.
    psd_tol
        Smallest Hessian eigenvalue tolerated by the saddle-rejection check.
        Note this is an **absolute** tolerance (scale-sensitive), unlike the
        scale-free dedup metric; scale ``psd_tol`` with your objective's
        curvature if needed.

    Returns
    -------
    MinimaResult
        ``.minima`` / ``.values`` sorted by objective, ``.x`` the best,
        ``.status`` one of ``"target_reached" | "converged" |
        "budget_exhausted"``, plus ``.n_solves`` and ``.trace``.
    """
    if method not in _STRATEGIES:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_STRATEGIES)}"
        )
    x0 = np.asarray(x0, dtype=float)
    rng = np.random.default_rng(seed)
    if max_solves is None:
        # Generous default: repulsion methods spend a polish solve per
        # accepted minimum, so the effective attempt budget is ~half this.
        max_solves = 8 * n_minima
    kw = dict(strategy_kw or {})
    kw.setdefault("seed", seed)

    # Per-dimension length scale (box width per variable). Drives the
    # anisotropic bump widths and, unless the caller supplies its own
    # ``distance``, the dedup metric — so flooding and de-duplication agree
    # and ``dedup`` is in scale-free (fraction-of-range) units.
    L, has_box = _scale_from_bounds(bounds, x0.size)
    kw["_L"], kw["_has_box"] = L, has_box
    if distance is None:
        distance = lambda a, b: float(np.linalg.norm((a - b) / L))

    ctx = _Context(fun, jac, hess, bounds, constraints, options, psd_tol)
    ctx.max_solves = max_solves
    archive = MinimaArchive(dedup, distance)
    state = _State(ctx, archive, n_minima, patience, callback)

    status = "budget_exhausted"
    try:
        _STRATEGIES[method](ctx, state, x0, rng, kw)
    except _Stop as stop:
        status = stop.status

    order = list(np.argsort(archive.fs)) if archive.fs else []
    return MinimaResult(
        minima=[archive.xs[i] for i in order],
        values=[archive.fs[i] for i in order],
        results=[archive.results[i] for i in order],
        status=status,
        n_solves=ctx.n_solves,
        trace=state.trace,
    )
