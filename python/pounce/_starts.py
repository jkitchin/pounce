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

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

__all__ = [
    "generate_starts",
    "project_to_feasible",
    "ProjectionReport",
    "race_starts",
]

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


def _lower_present(b) -> bool:
    """Is this *lower* bound real, or the absent-bound sentinel? Directional,
    not a magnitude band (gh #403)."""
    return b is not None and np.isfinite(b) and b > -_BOUND_INF


def _upper_present(b) -> bool:
    """Is this *upper* bound real? See :func:`_lower_present`."""
    return b is not None and np.isfinite(b) and b < _BOUND_INF


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
        flo, fhi = _lower_present(lo), _upper_present(hi)
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


@dataclass
class ProjectionReport:
    """What :func:`project_to_feasible` did.

    ``violation_initial`` / ``violation_final`` are the nonlinear
    violation merit ``‖max(cl − g(x), g(x) − cu, 0)‖₂`` — the *true*
    constraint violation, evaluated at the returned point, not the
    linearized one. ``accepted`` is False when no trial improved it
    and the original point was returned unchanged.
    """

    violation_initial: float = 0.0
    violation_final: float = 0.0
    step_norm: float = 0.0
    #: Trust-region radius in effect when the last step was accepted.
    radius: float = 0.0
    #: Trial steps whose true violation failed the acceptance test.
    rejected_trials: int = 0
    #: Outer re-linearizations performed.
    iterations: int = 0
    n_constraint_evals: int = 0
    n_jacobian_evals: int = 0
    accepted: bool = False
    termination: str = ""
    #: Sum of the elastic variables at the last solve. Nonzero means the
    #: linearization was inconsistent and some rows were relaxed rather
    #: than the whole solve failing.
    elastic_total: float = 0.0


def _violation(g, g_l, g_u):
    """Nonlinear violation merit: ``‖max(cl − g, g − cu, 0)‖₂``."""
    if g.size == 0:
        return 0.0
    return float(np.linalg.norm(np.maximum(np.maximum(g_l - g, g - g_u), 0.0)))


def _jacobian_coo(problem_obj, x, m, n):
    """Jacobian at ``x`` as a scipy COO matrix, without ever forming a
    dense ``m × n``.

    Uses ``jacobianstructure()`` when the problem provides one (the
    cyipopt convention, and the only shape that carries sparsity).
    Without it the values are a dense row-major block and we have no
    choice but to reshape — but we still hand a sparse matrix onward.
    """
    import scipy.sparse as sp

    jv = np.asarray(problem_obj.jacobian(x), dtype=float).ravel()
    if hasattr(problem_obj, "jacobianstructure"):
        rows, cols = problem_obj.jacobianstructure()
        rows = np.asarray(rows, dtype=int).ravel()
        cols = np.asarray(cols, dtype=int).ravel()
        return sp.coo_matrix((jv, (rows, cols)), shape=(m, n)).tocsc()
    dense = jv.reshape(m, n)
    return sp.csc_matrix(dense)


def project_to_feasible(
    problem_obj: Any,
    x0,
    *,
    lb=None,
    ub=None,
    cl=None,
    cu=None,
    tol: Optional[float] = None,
    max_iter: int = 3,
    radius: Optional[float] = None,
    rho: float = 1e3,
    sigma: float = 1.0,
    margin: float = 0.0,
    accept_ratio: float = 1e-2,
    max_trials: int = 5,
    return_report: bool = False,
) -> np.ndarray:
    """Repair ``x0`` onto the constraints and bounds by a safeguarded,
    sparse elastic normal step.

    Each outer iteration linearizes ``g`` at the current point and
    solves the sparse convex QP

    .. code-block:: text

        min_{d,p,q}  σ/2 ‖D d‖² + ½ ‖W p‖² + ½ ‖W q‖² + ρ 1ᵀ(p + q)
        s.t.         cl − p ≤ g(x) + J d + 0 ≤ cu + q
                     max(lb − x + margin, −Δ/D) ≤ d ≤ min(ub − x − margin, Δ/D)
                     p, q ≥ 0

    where ``D`` is a diagonal variable scaling, ``W`` a diagonal row
    scaling, and ``Δ`` the trust-region radius (an ∞-norm/box region,
    which keeps the subproblem a QP rather than a QCQP and composes
    directly with the bound box).

    Three things this buys over a plain min-norm projection:

    * **Sparsity.** ``P`` is diagonal and ``J`` is kept in scipy-sparse
      form throughout, so nothing here allocates an ``n × n`` identity
      or a dense ``m × n`` Jacobian. On a chain-structured model with
      ``n = 3000`` that is the difference between ~226 MB and ~5 MB.
    * **Elasticity.** ``p``/``q`` relax rows the linearization cannot
      satisfy, so an inconsistent or rank-deficient linearization
      returns the least-violating step instead of failing outright.
    * **Safeguarding.** A linearized solution is a local model step,
      not automatically a better starting point. Every trial step is
      scored on the *true* nonlinear violation; a trial is accepted
      only when the actual reduction is at least ``accept_ratio``
      times the reduction the model predicted. Otherwise ``Δ`` is
      halved and the step retried, up to ``max_trials`` times. If
      nothing is accepted, ``x0`` is returned unchanged.

    The returned point therefore *never* has a worse nonlinear
    violation than ``x0`` — which the previous linearize-once-and-copy
    behaviour did not guarantee.

    Parameters mirror :class:`pounce.Problem` / :func:`pounce.preflight`:
    a cyipopt-style ``problem_obj`` (only ``constraints``, ``jacobian``
    and optionally ``jacobianstructure`` are used) and bound arrays.

    ``sigma`` defaults to ``1``, which makes the ``d``-term exactly the
    ``½‖x − x0‖²`` of a min-norm projection: when the linearization is
    consistent the elastics price themselves out (``rho`` is large) and
    the step is the same minimum-norm repair this function has always
    returned. Lower it only if you want the repair to travel further in
    exchange for a smaller residual.

    ``max_iter`` outer re-linearizations (default 3) let the repair
    follow a curved feasible set instead of stopping at the first
    tangent step. ``margin`` keeps the result strictly inside the box.
    ``return_report=True`` additionally returns a
    :class:`ProjectionReport` with initial/final violation, step norm,
    rejected-trial count and termination reason.

    Raises ``RuntimeError`` only when the projection QP itself fails
    for a reason elasticity cannot absorb (e.g. the solver errors).
    An inconsistent linearization is no longer an error — it is
    absorbed by the elastic variables and reported through
    ``ProjectionReport.elastic_total``.
    """
    import scipy.sparse as sp

    from .qp import solve_qp

    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    x_l = np.full(n, -np.inf) if lb is None else np.asarray(lb, dtype=float).ravel()
    x_u = np.full(n, np.inf) if ub is None else np.asarray(ub, dtype=float).ravel()
    x_l = np.where(x_l <= -_BOUND_INF, -np.inf, x_l)
    x_u = np.where(x_u >= _BOUND_INF, np.inf, x_u)

    report = ProjectionReport()

    m = 0
    if cl is not None:
        m = np.asarray(cl, dtype=float).ravel().size
    if m == 0:
        report.termination = "no constraints; box clip only"
        x = np.clip(x0, x_l, x_u)
        report.step_norm = float(np.linalg.norm(x - x0))
        return (x, report) if return_report else x

    g_l = np.asarray(cl, dtype=float).ravel()
    g_u = np.full(m, np.inf) if cu is None else np.asarray(cu, dtype=float).ravel()
    g_l = np.where(g_l <= -_BOUND_INF, -np.inf, g_l)
    g_u = np.where(g_u >= _BOUND_INF, np.inf, g_u)

    # Rows split by kind. Equalities go to A; one- or two-sided
    # inequalities to G. Free rows (no finite side) are dropped —
    # they constrain nothing and would only add elastic columns.
    eq_mask = np.isfinite(g_l) & np.isfinite(g_u) & (np.abs(g_u - g_l) <= 1e-12)
    lo_mask = np.isfinite(g_l) & ~eq_mask
    hi_mask = np.isfinite(g_u) & ~eq_mask

    def _clip_box(v):
        return np.clip(v, x_l, x_u)

    x = _clip_box(x0.copy())
    g = np.asarray(problem_obj.constraints(x), dtype=float).ravel()
    report.n_constraint_evals += 1
    theta = _violation(g, g_l, g_u)
    report.violation_initial = theta
    report.violation_final = theta
    best_x = x.copy()

    if theta == 0.0:
        report.termination = "x0 already feasible"
        report.accepted = True
        return (best_x, report) if return_report else best_x

    # Variable scaling D: unit for now, but kept explicit so the
    # trust region and the σ term are expressed in scaled units.
    d_scale = np.ones(n)
    # Row scaling W: damp rows whose Jacobian is large so one stiff row
    # does not dominate the least-squares residual.
    for _outer in range(max(1, int(max_iter))):
        report.iterations += 1
        J = _jacobian_coo(problem_obj, x, m, n)
        report.n_jacobian_evals += 1
        row_norm = np.sqrt(np.asarray(abs(J).power(2).sum(axis=1)).ravel())
        w = 1.0 / np.maximum(row_norm, 1.0)

        if radius is None:
            delta = max(1.0, float(np.linalg.norm(x, ord=np.inf)))
        else:
            delta = float(radius)

        # Elastic column count: one p and one q per constrained row.
        n_p = int(np.count_nonzero(eq_mask | lo_mask))
        n_q = int(np.count_nonzero(eq_mask | hi_mask))
        p_idx = np.full(m, -1, dtype=int)
        p_idx[eq_mask | lo_mask] = np.arange(n_p)
        q_idx = np.full(m, -1, dtype=int)
        q_idx[eq_mask | hi_mask] = np.arange(n_q)
        nz = n + n_p + n_q

        # Diagonal Hessian — never an n×n identity.
        w_p = w[eq_mask | lo_mask]
        w_q = w[eq_mask | hi_mask]
        P = sp.diags(
            np.concatenate([sigma * d_scale**2, w_p**2, w_q**2]),
            format="csc",
        )
        c_lin = np.concatenate([np.zeros(n), rho * np.ones(n_p + n_q)])

        Ep = sp.coo_matrix(
            (np.ones(n_p), (np.flatnonzero(eq_mask | lo_mask), np.arange(n_p))),
            shape=(m, n_p),
        ).tocsc()
        Eq = sp.coo_matrix(
            (np.ones(n_q), (np.flatnonzero(eq_mask | hi_mask), np.arange(n_q))),
            shape=(m, n_q),
        ).tocsc()
        # Row block: g + J d + p − q, in the elastic column layout.
        row_block = sp.hstack([J, Ep, -Eq], format="csc")

        A = b = G = h = None
        if eq_mask.any():
            A = row_block.tocsr()[np.flatnonzero(eq_mask)].tocsc()
            b = (g_l - g)[eq_mask]
        g_blocks, h_blocks = [], []
        if hi_mask.any():
            g_blocks.append(row_block.tocsr()[np.flatnonzero(hi_mask)].tocsc())
            h_blocks.append((g_u - g)[hi_mask])
        if lo_mask.any():
            g_blocks.append(-row_block.tocsr()[np.flatnonzero(lo_mask)].tocsc())
            h_blocks.append(-(g_l - g)[lo_mask])
        if g_blocks:
            G = sp.vstack(g_blocks, format="csc")
            h = np.concatenate(h_blocks)

        accepted_this_outer = False
        trial_delta = delta
        for _trial in range(max(1, int(max_trials))):
            tr = trial_delta / d_scale
            d_lo = np.maximum(x_l - x + margin, -tr)
            d_hi = np.minimum(x_u - x - margin, tr)
            # A margin wider than the box would invert it; a degenerate
            # box just pins d to 0 for that component.
            d_hi = np.maximum(d_hi, d_lo)
            z_lo = np.concatenate([d_lo, np.zeros(n_p + n_q)])
            z_hi = np.concatenate([d_hi, np.full(n_p + n_q, np.inf)])

            # `check_psd=False` is a fact here, not an optimism: `P`
            # is built above as a diagonal with entries `sigma*D**2`
            # and `w**2`, all non-negative, so it is PSD by
            # construction. Letting the default fire would run a dense
            # O(k^3) eigenvalue solve on the (n + n_p + n_q) block
            # whenever that stays under the solver's 1500 threshold —
            # which on a sparse model is the single largest allocation
            # in the whole routine.
            res = solve_qp(
                P=P,
                c=c_lin,
                A=A,
                b=b,
                G=G,
                h=h,
                lb=z_lo,
                ub=z_hi,
                tol=tol,
                check_psd=False,
            )
            if not res.success:
                report.rejected_trials += 1
                trial_delta *= 0.5
                continue

            z = np.asarray(res.x, dtype=float).ravel()
            d = z[:n]
            report.elastic_total = float(np.sum(np.abs(z[n:])))

            # Predicted violation at the linearized point.
            g_lin = g + J @ d
            theta_pred = _violation(g_lin, g_l, g_u)
            predicted = theta - theta_pred

            x_try = _clip_box(x + d)
            g_try = np.asarray(problem_obj.constraints(x_try), dtype=float).ravel()
            report.n_constraint_evals += 1
            theta_try = _violation(g_try, g_l, g_u)
            actual = theta - theta_try

            # Accept only on a real reduction in the TRUE violation,
            # and only when it is a defensible fraction of what the
            # model promised.
            if (
                np.isfinite(theta_try)
                and theta_try < theta
                and actual >= accept_ratio * max(predicted, 0.0)
            ):
                x, g, theta = x_try, g_try, theta_try
                best_x = x_try
                report.radius = trial_delta
                report.accepted = True
                accepted_this_outer = True
                break

            report.rejected_trials += 1
            trial_delta *= 0.5

        if not accepted_this_outer:
            report.termination = (
                "no trial improved the nonlinear violation"
                if not report.accepted
                else "converged (no further improvement available)"
            )
            break
        if theta <= (tol if tol is not None else 1e-10):
            report.termination = "violation below tolerance"
            break
    else:
        report.termination = "max_iter reached"

    report.violation_final = theta
    report.step_norm = float(np.linalg.norm(best_x - x0))
    return (best_x, report) if return_report else best_x


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
