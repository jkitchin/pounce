"""Starting-point generation and repair.

Three composable building blocks (see ``docs/src/initialization.md``):

* :func:`generate_starts` — draw N diverse starting points (Sobol /
  uniform / jitter / bounds midpoint). This is the sampler that powers
  :func:`pounce.find_minima`, exposed as a standalone primitive.
* :func:`project_to_feasible` — min-norm repair of a candidate point
  onto the linearized constraints and bounds (one convex QP).
* :func:`race_starts` — run a few solver iterations from each of N
  starts and rank them, so the full-effort solve continues only from
  the most promising one(s).

The sampling internals here are also imported by ``pounce._minima``;
keep the private helpers' signatures stable.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

__all__ = ["generate_starts", "project_to_feasible", "race_starts"]

# Bounds at or beyond this magnitude count as infinite (the solver's
# NLP_*_BOUND_INF sentinels).
_BOUND_INF = 1e19


# --------------------------------------------------------------------------
# Sampling primitives (shared with pounce._minima).
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


def _finite(b) -> bool:
    return b is not None and np.isfinite(b) and -_BOUND_INF < b < _BOUND_INF


def _midpoint(bounds, x0, n):
    """Bounds-aware deterministic start: midpoint of each finite box,
    one unit inside a one-sided bound, else the x0 component (or 0)."""
    base = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel()
    if bounds is None:
        return base.copy()
    out = base.copy()
    for j, b in enumerate(bounds):
        lo = b[0] if b is not None else None
        hi = b[1] if b is not None else None
        flo, fhi = _finite(lo), _finite(hi)
        if flo and fhi:
            out[j] = 0.5 * (lo + hi)
        elif flo:
            out[j] = max(out[j], lo + 1.0)
        elif fhi:
            out[j] = min(out[j], hi - 1.0)
    return out


# --------------------------------------------------------------------------
# Public API.
# --------------------------------------------------------------------------
def generate_starts(
    n_points: int,
    *,
    bounds=None,
    x0=None,
    strategy: str = "sobol",
    jitter: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate ``n_points`` starting points, shape ``(n_points, n)``.

    This is the sampler behind :func:`pounce.find_minima`, exposed as a
    composable primitive — feed the result to
    :func:`pounce.solve_nlp_batch`, :func:`race_starts`, or a loop of
    :func:`pounce.minimize` calls.

    Args:
        n_points: How many starts to generate.
        bounds: ``[(lo, hi), ...]`` box, scipy-style. Entries (or either
            side) may be ``None`` / ``±inf`` for unbounded.
        x0: Anchor point. Required for the ``jitter`` strategy and for
            any strategy when ``bounds`` has unbounded components.
        strategy: One of
            ``"sobol"`` — scrambled Sobol sequence in the box (falls
            back to uniform when SciPy is unavailable);
            ``"uniform"`` — i.i.d. uniform in the box;
            ``"jitter"`` — Gaussian ``x0 + jitter * N(0, I)`` samples;
            ``"midpoint"`` — the deterministic bounds midpoint first
            (the cold start the solver *doesn't* give you: zeros +
            clamp), then Sobol for the remainder.
        jitter: Scale for the ``jitter`` strategy (also used as the
            fallback when a box strategy meets unbounded components).
        seed: RNG seed for reproducibility.

    Returns:
        ``(n_points, n)`` array; every row is clipped into ``bounds``.
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float).ravel()
        n = x0.size
    elif bounds is not None:
        n = len(bounds)
    else:
        raise ValueError("generate_starts needs bounds or x0 to fix the dimension")
    if x0 is None:
        if not _has_box(bounds):
            raise ValueError(
                "generate_starts: with unbounded components, pass x0 as the anchor"
            )
        x0 = _midpoint(bounds, None, n)

    strategy = strategy.lower()
    if strategy not in ("sobol", "uniform", "jitter", "midpoint"):
        raise ValueError(f"unknown strategy {strategy!r}")

    rng = np.random.default_rng(seed)
    starts = np.empty((n_points, n), dtype=float)
    k = 0
    if strategy == "midpoint":
        starts[0] = _midpoint(bounds, x0, n)
        k = 1
    if strategy == "jitter":
        for i in range(k, n_points):
            starts[i] = x0 + jitter * rng.standard_normal(n)
    else:
        sobol = _make_sobol(n, seed, strategy in ("sobol", "midpoint"))
        for i in range(k, n_points):
            starts[i] = _sample(bounds, x0, rng, jitter, sobol)
    return np.array([_clip(s, bounds) for s in starts])


def project_to_feasible(
    problem_obj: Any,
    x0,
    *,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    tol: Optional[float] = None,
) -> np.ndarray:
    """Min-norm repair of ``x0`` onto the linearized constraints + bounds.

    Solves the convex QP ``min ½‖x − x0‖²`` subject to
    ``cl ≤ g(x0) + J(x0)(x − x0) ≤ cu`` and ``lb ≤ x ≤ ub`` — the
    standalone form of the solver's ``least_square_init_primal`` step
    (see ``docs/src/initialization.md``). For mostly-linear models this
    slashes iteration-0 infeasibility; for a strongly nonlinear ``g``
    the linearization is only a local repair (it may be worth a second
    call at the projected point).

    Parameters mirror :class:`pounce.Problem` /
    :func:`pounce.preflight`: a cyipopt-style ``problem_obj`` (only
    ``constraints`` and ``jacobian`` are used) and bound arrays.

    Returns the repaired point. With no constraints it is simply the
    clip of ``x0`` into the box. Raises ``RuntimeError`` when the
    projection QP fails (e.g. the linearized constraints are themselves
    inconsistent).
    """
    from .qp import solve_qp

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    x_l = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).ravel()
    x_u = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float).ravel()

    m = 0
    if cl is not None:
        m = np.asarray(cl, dtype=float).ravel().size
    if m == 0:
        return np.clip(x0, x_l, x_u)

    g_l = np.asarray(cl, dtype=float).ravel()
    g_u = np.full(m, np.inf) if cu is None else np.asarray(cu, dtype=float).ravel()

    g0 = np.asarray(problem_obj.constraints(x0), dtype=float).ravel()
    jv = np.asarray(problem_obj.jacobian(x0), dtype=float).ravel()
    J = np.zeros((m, n))
    if hasattr(problem_obj, "jacobianstructure"):
        rows, cols = problem_obj.jacobianstructure()
        J[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)] = jv
    else:
        J = jv.reshape(m, n)

    # Linearized rows: J x ∈ [cl − g0 + J x0, cu − g0 + J x0].
    shift = J @ x0 - g0
    row_lo = g_l + shift
    row_hi = g_u + shift

    A_rows, b_rows, G_rows, h_rows = [], [], [], []
    for i in range(m):
        lo_f = _finite(g_l[i])
        hi_f = _finite(g_u[i])
        eq = abs(g_u[i] - g_l[i]) <= 1e-12 if (lo_f and hi_f) else False
        if eq:
            A_rows.append(J[i])
            b_rows.append(row_lo[i])
            continue
        if hi_f:
            G_rows.append(J[i])
            h_rows.append(row_hi[i])
        if lo_f:
            G_rows.append(-J[i])
            h_rows.append(-row_lo[i])

    res = solve_qp(
        P=np.eye(n),
        c=-x0,
        A=np.array(A_rows) if A_rows else None,
        b=np.array(b_rows) if b_rows else None,
        G=np.array(G_rows) if G_rows else None,
        h=np.array(h_rows) if h_rows else None,
        lb=x_l,
        ub=x_u,
        tol=tol,
    )
    if not res.success:
        raise RuntimeError(
            f"project_to_feasible: projection QP ended with status "
            f"{res.status!r} — the linearized constraints may be inconsistent"
        )
    return np.asarray(res.x)


def race_starts(
    fun,
    starts,
    *,
    jac=None,
    bounds=None,
    constraints=None,
    iters: int = 10,
    top: int = 1,
    options: Optional[dict] = None,
) -> List[Any]:
    """Run ``iters`` solver iterations from each start and rank them.

    A cheap tournament: each candidate gets a short, truncated
    :func:`pounce.minimize` run (``max_iter=iters``), and the resulting
    iterates are ranked by (constraint violation beyond tolerance,
    objective value). Continue the real solve from the winner —
    typically with ``warm_start=pounce.WarmStart.from_info(res.x,
    res.info)``.

    Returns the ``top`` best :class:`OptimizeResult` objects, best
    first.
    """
    from ._minimize import minimize

    opts = dict(options or {})
    opts["max_iter"] = int(iters)
    results = []
    for s in np.atleast_2d(np.asarray(starts, dtype=float)):
        res = minimize(
            fun, s, jac=jac, bounds=bounds, constraints=constraints, **opts
        )
        viol = float(res.info.get("final_constr_viol", 0.0))
        if not np.isfinite(viol):
            viol = np.inf
        obj = res.fun if np.isfinite(res.fun) else np.inf
        results.append((max(viol - 1e-6, 0.0), obj, res))
    results.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in results[: max(1, int(top))]]
