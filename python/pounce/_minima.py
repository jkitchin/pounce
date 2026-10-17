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

The search itself runs in Rust — ``pounce_minima::find_minima``, the same
code ``pounce model.nl --minima <method>`` and ``pounce_rs::minima`` run
(gh#966). This module builds the problem exactly as :func:`pounce.minimize`
does (bounds, constraints, finite-difference warnings and option
translation included), hands it over, and wraps the answer; only the
model's own callbacks, and ``callback`` / ``distance``, call back into
Python. See ``docs/src/find-minima.md`` and the notebooks
``python/notebooks/15–17`` for walkthroughs and method selection.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._minimize import (
    OptimizeResult,
    _normalize_bounds,
    _prepare_nlp,
    _validate_bounds_length,
    _wrap_constraints,
)

__all__ = ["find_minima", "MinimaResult"]

_METHODS = ("basinhopping", "deflation", "flooding", "mlsl", "multistart", "tunneling")

# ``strategy_kw`` keys the search reads. An unknown key used to be ignored
# silently, so a typo (``"sigma_fac"``) ran the default; it now raises.
_STRATEGY_KEYS = frozenset({
    "sigma", "sigma_frac", "amplitude", "amp_margin", "amp_bump",
    "amp_bump_cap", "amp_fallback", "eta", "power", "soft", "length",
    "length_frac", "gamma", "samples_per_round", "max_samples", "step",
    "temperature", "restart_jitter", "sobol", "seed",
})


# --------------------------------------------------------------------------
# Per-dimension length scales (anisotropic widths + scaled dedup metric).
# --------------------------------------------------------------------------
def _scale_from_bounds(bounds, n):
    """Per-dimension length scale L (box width per variable) and whether a
    finite box was available. The search computes the same from the
    problem's bounds; :func:`pounce.find_critical_points` uses this one."""
    if bounds is not None and all(
        b is not None and b[0] is not None and b[1] is not None for b in bounds
    ):
        lo = np.array([b[0] for b in bounds], float)
        hi = np.array([b[1] for b in bounds], float)
        L = hi - lo
        L[L <= 0] = 1.0
        return L, True
    return np.ones(n), False


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
    #: gh#983 review: converged solves the ``kkt_tol`` filter rejected. When
    #: every converged candidate was rejected and nothing was accepted, a
    #: ``RuntimeWarning`` is raised as well, so an empty result is never silent.
    n_kkt_rejected: int = 0
    #: The seed the search ran with. With ``seed=None`` one is drawn at
    #: random and reported here, so any run can be replayed exactly.
    seed: int | None = None
    #: Base-problem constraint multipliers at each minimum (same order as
    #: ``minima``; empty arrays without constraints), recovered by a clean
    #: re-solve from the minimum.
    lambdas: list[np.ndarray] = field(default_factory=list)

    @property
    def x(self):
        return self.minima[0] if self.minima else None

    @property
    def fun(self):
        return self.values[0] if self.values else None

    def __len__(self):
        return len(self.minima)


def _recorded(x, f, lam):
    # The archive records points, not solves; ``info`` carries the minimum's
    # constraint multipliers when there are any. An empty ``info`` must still
    # repr cleanly (gh #340).
    return OptimizeResult(
        x=np.asarray(x), fun=float(f), success=True, status=0,
        message="recorded", nit=0,
        info={"mult_g": np.asarray(lam)} if np.size(lam) else {},
    )


def _width(spec, n, name):
    """``"auto"`` / scalar / length-``n`` vector -> what the search takes."""
    if spec is None or (isinstance(spec, str) and spec == "auto"):
        return None
    arr = np.asarray(spec, float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != (n,):
        raise ValueError(f"strategy_kw[{name!r}] must be a scalar or have length {n}")
    return arr


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
    kkt_tol: float | None = 1e-4,
    options: Mapping[str, Any] | None = None,
    strategy_kw: Mapping[str, Any] | None = None,
    distance: Callable | None = None,
    seed: int | None = None,
    callback: Callable | None = None,
) -> MinimaResult:
    """Find multiple local minima of ``fun`` by driving the local solver in a loop.

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
        escape height) and raised adaptively (``amp_bump``, up to
        ``amp_bump_cap``) if the solver returns to a flooded basin, with
        ``amp_fallback`` when there is no Hessian — so no manual energy
        scale is needed (give a scalar to override). Examples:
        ``sigma``/``amplitude`` (flooding), ``eta``/``power``/``length``
        (deflation), ``step``/``temperature`` (basinhopping),
        ``samples_per_round``/``gamma``/``max_samples`` (mlsl). An
        unrecognised key raises ``ValueError``.
    distance
        Custom metric ``d(a, b) -> float`` for dedup (e.g. periodic boxes).
        Defaults to Euclidean distance in the per-dimension scaled space.
    psd_tol
        Smallest Hessian eigenvalue tolerated by the saddle-rejection check
        (run when ``hess`` is given and ``n <= 256``). Note this is an
        **absolute** tolerance (scale-sensitive), unlike the scale-free dedup
        metric; scale ``psd_tol`` with your objective's curvature if needed.
    kkt_tol
        A converged local solve is rejected as a candidate when the solver's
        own **unscaled** dual infeasibility exceeds
        ``kkt_tol * max(1, g0, scale)``: ``g0`` is ``max |grad f|`` at ``x0``
        (the problem's gradient scale) and ``scale`` the magnitude of the
        terms the residual is assembled from at the returned point
        (``max |grad f|, |J^T lambda|, |z|``). Default ``1e-4``; ``None``
        disables. The solver's success verdict is judged in a scaled space,
        and this is the model-units check that applies whether or not
        ``hess=`` is given (gh#983). Being relative, it does not reject every
        candidate of an objective whose gradients legitimately live at a
        large scale. Rejections are counted in ``MinimaResult.n_kkt_rejected``,
        and a ``RuntimeWarning`` is raised when they emptied the result.
    seed
        Seeds the start sampler (scrambled Sobol' and a ChaCha8 PRNG — the
        CLI's, so the same seed runs the same search from ``pounce --minima
        --seed``). ``None`` draws one at random; ``MinimaResult.seed``
        reports it either way.
    callback
        ``callback(x, f)``, called on each accepted minimum. An exception
        raised in it ends the search and propagates.

    Returns
    -------
    MinimaResult
        ``.minima`` / ``.values`` sorted by objective, ``.x`` the best,
        ``.status`` one of ``"target_reached" | "converged" |
        "budget_exhausted"`` (``max_solves`` spent) ``|
        "sample_cap_reached"`` (MLSL ``max_samples`` hit), plus ``.n_solves``,
        ``.trace``, ``.seed`` and ``.lambdas``.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(_METHODS)}")
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = x0.size
    _validate_bounds_length(bounds, n)
    if n_minima < 1:
        raise ValueError(f"n_minima must be >= 1, got {n_minima}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if max_solves is not None and max_solves < 1:
        raise ValueError(f"max_solves must be >= 1, got {max_solves}")
    bad = np.where(~np.isfinite(x0))[0]
    if bad.size:
        raise ValueError(f"x0[{int(bad[0])}] is {x0[bad[0]]}; the starting point must be finite.")

    kw = dict(strategy_kw or {})
    unknown = sorted(set(kw) - _STRATEGY_KEYS)
    if unknown:
        raise ValueError(
            f"find_minima: unknown strategy_kw key(s) {unknown}; "
            f"expected some of {sorted(_STRATEGY_KEYS)}"
        )
    # `strategy_kw["seed"]` is the old spelling of `seed`; an explicit `seed=`
    # wins.
    if seed is None:
        seed = kw.pop("seed", None)
    else:
        kw.pop("seed", None)
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0])
    seed = int(seed)

    opts = dict(options or {})
    selection = str(opts.pop("solver_selection", "nlp")).lower()
    if selection != "nlp":
        raise ValueError(
            "find_minima runs every local solve on the NLP path (the repulsion "
            "methods change the objective, so the convex routes do not apply); "
            f"got solver_selection={selection!r}."
        )
    opts.pop("route_tol", None)
    disp = bool(opts.pop("disp", False))
    opts.setdefault("print_level", 5 if disp else 0)

    lb, ub = _normalize_bounds(bounds, n)
    m, g_comb, jac_comb, cl, cu, jrows, jcols = _wrap_constraints(constraints, n, x0)
    problem, _, _ = _prepare_nlp(
        fun=fun, n=n, m=m, args=(), jac=jac, hess=hess,
        constraints=constraints, g_combined=g_comb, jac_combined=jac_comb,
        jac_rows=jrows, jac_cols=jcols, lb=lb, ub=ub, cl=cl, cu=cu,
        callback=None, options=opts, facade="pounce.find_minima",
    )

    search = {
        "method": method, "n_minima": int(n_minima),
        "max_solves": None if max_solves is None else int(max_solves),
        "patience": int(patience), "dedup": float(dedup),
        "psd_tol": float(psd_tol),
        "kkt_tol": None if kkt_tol is None else float(kkt_tol),
        "seed": seed,
    }
    for key in ("sigma", "length"):
        if key in kw:
            search[key] = _width(kw.pop(key), n, key)
    if "amplitude" in kw:
        amp = kw.pop("amplitude")
        search["amplitude"] = None if (isinstance(amp, str) and amp == "auto") else float(amp)
    search.update(kw)

    from ._pounce import find_minima as _native_find_minima

    out = _native_find_minima(
        problem, x0, search, callback=callback,
        distance=None if distance is None else
        (lambda a, b, _d=distance: float(_d(a, b))),
    )

    if not out["minima"] and out["n_kkt_rejected"]:
        warnings.warn(
            f"find_minima: every converged candidate ({out['n_kkt_rejected']}) was "
            f"rejected by kkt_tol={kkt_tol!r} (unscaled dual infeasibility "
            f"above kkt_tol * max(1, dual scale)); no minima are returned. "
            f"Inspect the solves' info['final_unscaled_dual_inf'] or loosen "
            f"kkt_tol.",
            RuntimeWarning,
            stacklevel=2,
        )
    for w in out["warnings"]:
        warnings.warn(f"find_minima: {w}", RuntimeWarning, stacklevel=2)

    minima = [np.asarray(x) for x in out["minima"]]
    values = [float(v) for v in out["values"]]
    lambdas = [np.asarray(lam) for lam in out["lambdas"]]
    return MinimaResult(
        minima=minima,
        values=values,
        results=[_recorded(x, f, lam) for x, f, lam in zip(minima, values, lambdas)],
        status=out["status"],
        n_solves=int(out["n_solves"]),
        trace=list(out["trace"]),
        n_kkt_rejected=int(out["n_kkt_rejected"]),
        seed=seed,
        lambdas=lambdas,
    )
