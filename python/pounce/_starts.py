"""Starting-point generation and repair.

Three composable building blocks (see ``docs/src/initialization.md``):

* :func:`generate_starts` — draw N diverse starting points (Sobol /
  uniform / jitter / bounds midpoint). This is the sampler that powers
  :func:`pounce.find_minima`, exposed as a standalone primitive.
* :func:`project_to_feasible` — min-norm repair of a candidate point
  onto the linearized constraints and bounds (one convex QP).
* :func:`race_starts` — run a few solver iterations from each of N
  starts and rank them, so the full-effort solve continues only from
  the most promising one(s). Two policies: ``"fixed"`` (the default —
  every candidate gets the same truncated budget) and ``"halving"``
  (pounce#610, opt-in — an adaptive successive-halving ladder that
  pauses and resumes candidates instead of re-running them, much
  cheaper, but see the quality caveat in :func:`race_starts`).

The sampling internals here are also imported by ``pounce._minima``;
keep the private helpers' signatures stable.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "generate_starts",
    "project_to_feasible",
    "ProjectionReport",
    "race_starts",
    "RaceReport",
    "RaceRound",
    "RaceCandidate",
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


# --------------------------------------------------------------------------
# Racing (pounce#610).
# --------------------------------------------------------------------------

#: Statuses that mean the candidate has *finished* — no budget it is
#: given afterwards would be spent. Mirrors ``_minimize._NLP_SUCCESS_STATUS``,
#: including ``FeasiblePointFound`` (6) as of gh #815: a square problem's
#: feasible point is its solution, so the candidate is done, not broken. While
#: 6 was outside this set a square candidate that solved was scored as
#: ``solve failed`` and eliminated at its rung, which is the opposite of what
#: happened.
_DONE_STATUS = frozenset({0, 1, 6})

#: Statuses that mean the candidate stopped because its truncation budget
#: ran out, i.e. it is *paused* and can be resumed. Everything outside
#: ``_DONE_STATUS`` and this set is a genuine failure.
_PAUSED_STATUS = frozenset({-1, -4, -5})

#: Evaluations-per-iteration assumed before a candidate has run once, used
#: only to turn rung 0's evaluation budget into an iteration cap. Every
#: later rung uses the candidate's own measured ratio (see
#: :meth:`_Racer._advance`), which is the whole point of budgeting in
#: evaluations: a candidate whose iterations are expensive gets fewer of
#: them for the same resource.
_EVALS_PER_ITER_GUESS = 4.0

#: Violation below which a candidate counts as feasible, matching the
#: pre-#610 ranking's ``max(viol - 1e-6, 0)`` slack.
_FEAS_TOL = 1e-6

#: Default weights for the composite rung score. Feasibility carries the
#: most because an infeasible candidate's objective is not a number about
#: the problem being solved; see :func:`race_starts`.
DEFAULT_RACE_WEIGHTS: Dict[str, float] = {
    "violation": 3.0,
    "feasibility_progress": 1.0,
    "kkt": 1.5,
    "objective_progress": 1.0,
    "health": 1.0,
}


@dataclass
class RaceCandidate:
    """One racer's whole life, from its start to its exit.

    Attributes:
        index: Position in the ``starts`` array. Candidates are
            identified by this everywhere in the report.
        x0: The starting point.
        x: Current iterate.
        status / status_msg: Last solve's exit.
        obj / violation / kkt: Objective, constraint violation and the
            scaled KKT residual at the current iterate.
        obj0 / violation0: The same two at the *first* pause, which is
            what ``objective_progress`` / ``feasibility_progress`` are
            measured against.
        evals: Solver callback evaluations charged to this candidate,
            summed over every rung it ran — the resource the ladder
            budgets in.
        iters: Solver iterations, summed the same way.
        nfev / njev / nhev: Python-side callback counts, in
            :func:`pounce.minimize`'s vocabulary.
        resumes: Times this candidate was resumed from held state rather
            than started. ``rungs_run - 1`` for anything that survived a
            cut.
        restoration_calls: Restoration entries summed over its rungs —
            the numerical-health signal.
        eliminated_round: Rung at which it was cut, or ``None`` if it
            reached the end.
        reason: Why it left, or why it was kept.
        result: The candidate's :class:`~pounce.OptimizeResult`.
    """

    index: int
    x0: np.ndarray
    x: np.ndarray = field(default=None, repr=False)
    status: Optional[int] = None
    status_msg: str = ""
    obj: float = float("nan")
    violation: float = float("nan")
    kkt: float = float("nan")
    obj0: float = float("nan")
    violation0: float = float("nan")
    evals: int = 0
    iters: int = 0
    nfev: int = 0
    njev: int = 0
    nhev: int = 0
    rungs_run: int = 0
    resumes: int = 0
    restoration_calls: int = 0
    eliminated_round: Optional[int] = None
    reason: str = ""
    result: Any = field(default=None, repr=False)

    @property
    def feasible(self) -> bool:
        return np.isfinite(self.violation) and self.violation <= _FEAS_TOL


@dataclass
class RaceRound:
    """One rung of the ladder.

    ``eval_budget`` is the *cumulative* per-candidate evaluation budget
    this rung raises everyone to, so ``evals`` (what the rung actually
    cost, across its entrants) is generally well below
    ``eval_budget × len(entrants)``: a candidate already at the budget,
    or already converged, spends nothing.
    """

    index: int
    eval_budget: int
    iter_budget: int = 0
    entrants: List[int] = field(default_factory=list)
    survivors: List[int] = field(default_factory=list)
    eliminated: List[Tuple[int, str]] = field(default_factory=list)
    evals: int = 0
    iters: int = 0
    resumed: int = 0
    started: int = 0
    scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class RaceReport:
    """What the race did, rung by rung (pounce#610 acceptance item 3).

    Every number here is reproducible: the policy draws no random
    numbers, so two runs over the same starts on a deterministic backend
    produce identical rounds, scores and elimination reasons — which is
    what ``python/tests/test_starts_racing.py`` asserts, on the whole
    record rather than on the winner alone.
    """

    policy: str = "halving"
    rounds: List[RaceRound] = field(default_factory=list)
    candidates: List[RaceCandidate] = field(default_factory=list)
    eta: int = 3
    weights: Dict[str, float] = field(default_factory=dict)

    @property
    def total_evals(self) -> int:
        return sum(c.evals for c in self.candidates)

    @property
    def total_iters(self) -> int:
        return sum(c.iters for c in self.candidates)

    @property
    def n_resumes(self) -> int:
        return sum(c.resumes for c in self.candidates)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    def report(self) -> str:
        """One-line-per-rung summary, for a log or a notebook."""
        out = [
            f"race: policy={self.policy} eta={self.eta} "
            f"candidates={len(self.candidates)} rungs={self.n_rounds}"
        ]
        for r in self.rounds:
            out.append(
                f"  rung {r.index}: budget={r.eval_budget} evals "
                f"entrants={len(r.entrants)} -> survivors={len(r.survivors)} "
                f"spent={r.evals} evals / {r.iters} iters "
                f"({r.resumed} resumed, {r.started} started)"
            )
            for idx, why in r.eliminated:
                out.append(f"      - #{idx}: {why}")
        out.append(
            f"  total {self.total_evals} evals / {self.total_iters} iters, "
            f"{self.n_resumes} resumes"
        )
        return "\n".join(out)


def _rank01(values: Sequence[float]) -> List[float]:
    """Rank-transform to ``[0, 1]``, small is good, ties share a rank.

    Rank-transforming each signal before combining them is what lets a
    violation in units of "mol/s" and a dimensionless KKT residual sit in
    one weighted sum without an invented scale factor. Non-finite entries
    sort last, deterministically.
    """
    n = len(values)
    if n <= 1:
        return [0.0] * n
    keyed = sorted(
        range(n),
        key=lambda i: (not np.isfinite(values[i]), values[i]
                       if np.isfinite(values[i]) else 0.0, i),
    )
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Tied values share the mean of the ranks they span, so a cohort
        # where nothing separates on one signal contributes nothing to
        # the ordering rather than an arbitrary permutation.
        while (j + 1 < n and np.isfinite(values[keyed[j + 1]])
               and np.isfinite(values[keyed[i]])
               and values[keyed[j + 1]] == values[keyed[i]]):
            j += 1
        share = 0.5 * (i + j) / (n - 1)
        for k in range(i, j + 1):
            out[keyed[k]] = share
        i = j + 1
    return out


def _log_scale(v: float) -> float:
    """``log10`` of a residual, with non-finite mapped to ``+inf``."""
    if not np.isfinite(v):
        return float("inf")
    return math.log10(max(v, 1e-300))


class _Racer:
    """The adaptive ladder. One instance per :func:`race_starts` call."""

    def __init__(self, fun, starts, *, jac, hess, args, bounds, constraints,
                 iters, top, options, eta, rungs, eval_budget, min_survivors,
                 explore, cluster_tol, weights, min_rung_iters, feas_band):
        from ._minimize import (
            _normalize_bounds,
            _prepare_nlp,
            _wrap_constraints,
        )

        self._starts = np.atleast_2d(np.asarray(starts, dtype=float))
        self._n_starts = self._starts.shape[0]
        n = self._starts.shape[1]
        self._top = max(1, int(top))
        self._eta = max(2, int(eta))
        self._explore = max(0, int(explore))
        self._min_survivors = max(1, int(min_survivors))
        self._cluster_tol = float(cluster_tol)
        self._feas_band = float(feas_band)
        self._weights = dict(DEFAULT_RACE_WEIGHTS)
        self._weights.update(weights or {})

        opts = dict(options or {})
        selection = str(opts.pop("solver_selection", "nlp")).lower()
        if selection != "nlp":
            # The ladder holds one `pounce.Solver` session per candidate,
            # which is the NLP path's object. Silently routing to the
            # convex solver instead would return answers with no session
            # to pause, so the ladder would degrade to the cold-restart
            # loop this issue exists to remove — without saying so.
            raise ValueError(
                "race_starts(policy='halving') runs on the NLP path: it holds "
                "a pounce.Solver session per candidate so a rung can be paused "
                "and resumed, and the convex/conic routes have no such "
                f"session. Got solver_selection={selection!r}. Use "
                "policy='fixed' to race through pounce.minimize's router."
            )
        opts.pop("route_tol", None)
        disp = bool(opts.pop("disp", False))
        opts.setdefault("print_level", 5 if disp else 0)
        self._options = opts

        self._lb, self._ub = _normalize_bounds(bounds, n)
        m, g_comb, jac_comb, cl, cu, jrows, jcols = _wrap_constraints(
            constraints, n, self._starts[0]
        )
        self._m, self._cl, self._cu = m, cl, cu
        self._prepare = lambda: _prepare_nlp(
            fun=fun, n=n, m=m, args=args, jac=jac, hess=hess,
            constraints=constraints, g_combined=g_comb, jac_combined=jac_comb,
            jac_rows=jrows, jac_cols=jcols, lb=self._lb, ub=self._ub,
            cl=cl, cu=cu, callback=None, options=self._options,
            facade="pounce.race_starts",
        )

        # Ladder geometry. `iters` remains what it always was — the
        # budget the *winner* ends up with — so a halving race and a
        # fixed race give the eventual winner the same effort and differ
        # only in what the losers cost.
        #
        # Two things bound the rung count. The field has to reach `top`
        # (that is the span term), and rung 0 has to be long enough to
        # rank on: a rung of one iteration ranks candidates on the
        # solver's first step, which on a multi-basin problem is noise,
        # and eliminating the eventual winner there is exactly the
        # failure a race is supposed to prevent. `min_rung_iters` is the
        # floor, and it wins — a shorter ladder is better than a
        # meaningless bottom rung.
        self._iters = max(1, int(iters))
        floor_iters = max(1, int(min_rung_iters))
        if rungs is None:
            span = max(1.0, self._n_starts / float(self._top))
            rungs = 1 + int(math.ceil(math.log(span) / math.log(self._eta)))
        room = 1 + int(math.floor(
            math.log(max(1.0, self._iters / floor_iters)) / math.log(self._eta)
        ))
        self._rungs = max(1, min(int(rungs), room))
        self._e0 = None if eval_budget is None else max(1.0, float(eval_budget))
        self._scale = self._variable_scale(n)

    def _iter_budget(self, rung):
        """Cumulative iteration ceiling at `rung`.

        Geometric, landing exactly on `iters` at the top, so the winner
        of a halving race gets the effort a fixed race would have given
        it. This is a *ceiling*: the evaluation budget below is the
        resource that normally binds, and binds sooner for a candidate
        whose iterations are expensive.
        """
        return max(1, int(round(
            self._iters / self._eta ** (self._rungs - 1 - rung))))

    # -- geometry ------------------------------------------------------

    def _variable_scale(self, n):
        """Per-variable scale for the diversity metric: the box width
        where the box is finite, else 1. Clustering has to be in scaled
        units or a variable that ranges over 1e6 decides every distance."""
        s = np.ones(n)
        if self._lb is None or self._ub is None:
            return s
        lo = np.asarray(self._lb, float).ravel()
        hi = np.asarray(self._ub, float).ravel()
        # `_normalize_bounds` leaves an absent side at ±inf; the solver's
        # own ±1e19 sentinels arrive that way too, and neither gives a
        # width worth scaling by.
        lo = np.where(lo <= -_BOUND_INF, -np.inf, lo)
        hi = np.where(hi >= _BOUND_INF, np.inf, hi)
        w = hi - lo
        ok = np.isfinite(w) & (w > 0)
        s[ok] = w[ok]
        return s

    def _survivor_count(self, rung, n_alive):
        """How many candidates rung `rung` leaves standing."""
        want = int(math.ceil(self._n_starts / self._eta ** (rung + 1)))
        return max(self._top, self._min_survivors, min(want, n_alive))

    # -- the starting point --------------------------------------------

    def _measure(self, state, x0):
        """``(objective, violation)`` at `x0`, without solving.

        Goes through the same ``problem_obj`` the solver will call, so
        the counters see it and the numbers are in the same units the
        rungs will report.
        """
        obj = float(state["problem_obj"].objective(np.asarray(x0, float)))
        if not self._m:
            return obj, 0.0
        g = np.asarray(
            state["problem_obj"].constraints(np.asarray(x0, float)), float
        ).ravel()
        cl = np.asarray(self._cl, float).ravel()
        cu = np.asarray(self._cu, float).ravel()
        return obj, _violation(g, cl, cu)

    # -- one candidate, one rung ---------------------------------------

    def _advance(self, cand, eval_budget, iter_budget, state):
        """Spend `cand` up to cumulative `eval_budget` / `iter_budget`.

        This is the pause/resume step. `state` holds the candidate's live
        ``Problem``/``Solver`` session and the :class:`~pounce.WarmStart`
        captured when it was last paused — its primal iterate, all three
        multiplier blocks, and the barrier parameter μ it had reached.
        Resuming replays that whole state through the #607 warm-start
        path (μ included, so #606's recentering measures the point it is
        actually handed); it does not re-derive a start from the
        candidate's answer.

        `eval_budget` is ``None`` on rung 0, which has no evaluation
        budget to spend against because nothing has yet been measured —
        rung 0 *is* the calibration, and the cohort's rung-0 spend is
        what sets the ladder's unit (see :meth:`run`).

        Returns ``(spent_evals, spent_iters, resumed)``.
        """
        from ._minimize import _result_from_info
        from ._warm_start import WarmStart
        from ._warm_start_schema import WarmStartOrderingUnverifiedWarning

        problem, solver, ws = state["problem"], state["solver"], state["ws"]
        max_iter = iter_budget - cand.iters
        if eval_budget is not None:
            want = eval_budget - cand.evals
            if want <= 0:
                return 0, 0, False
            # Evaluations -> iterations, through *this* candidate's own
            # measured cost. This is what makes evaluations rather than
            # iterations the comparable resource: a candidate whose
            # iterations are expensive (a dozen line-search trials, a
            # restoration excursion) is granted fewer of them for the
            # same budget than a cheap one, instead of the two being
            # charged the same for wildly different work.
            epi = ((cand.evals / cand.iters) if cand.iters > 0
                   else _EVALS_PER_ITER_GUESS)
            max_iter = min(max_iter, int(want / max(epi, 1.0)))
        if max_iter <= 0 or cand.status in _DONE_STATUS:
            return 0, 0, False
        max_iter = max(1, int(max_iter))

        resumed = ws is not None
        snapshot = problem.options_snapshot()
        try:
            problem.add_option("max_iter", int(max_iter))
            if resumed:
                # The warm-start overlay belongs to this rung, not to the
                # Problem: `add_option` is append-only, so without the
                # snapshot the seven `warm_start_*` / `mu_init` options
                # outlive the rung and the next candidate's first,
                # nominally cold, solve is not (pounce#607).
                # gh#660's note does not apply here and would fire on
                # every rung of every race. The state a few lines below
                # is captured with `probe=False` *because* it is
                # captured from, and resumed on, this same `Problem`
                # object a moment later — there is no reordering for the
                # comparison to have missed. The check still runs; only
                # the "could not have seen a reordering" caveat is
                # dropped, and only where it is provably vacuous.
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore", WarmStartOrderingUnverifiedWarning)
                    ws.check_compatible(problem)
                for key, val in ws.overlay_options(
                        problem.options_snapshot()).items():
                    problem.add_option(key, val)
                kw = ws.solve_kwargs()
                kw.pop("working_set", None)  # not a Solver.solve keyword
                x, info = solver.solve(x0=ws.x, **kw)
            else:
                x, info = solver.solve(x0=cand.x0)
        finally:
            problem.restore_options(snapshot)

        spent = _info_evals(info)
        used = int(info.get("iter_count", 0))
        cand.evals += spent
        cand.iters += used
        cand.rungs_run += 1
        cand.resumes += int(resumed)
        cand.restoration_calls += int(info.get("restoration_calls", 0))
        cand.status = int(info.get("status", -99))
        cand.status_msg = str(info.get("status_msg", ""))
        cand.x = np.asarray(x, float).copy()
        cand.obj = float(info.get("obj_val", float("nan")))
        v = float(info.get("final_constr_viol", 0.0))
        cand.violation = v if np.isfinite(v) else float("inf")
        cand.kkt = float(info.get("final_kkt_error", float("nan")))
        counters = state["counters"]
        cand.nfev = int(counters["nfev"])
        cand.njev = int(counters["njev"])
        cand.nhev = int(counters["nhev"])
        cand.result = _result_from_info(x, info, counters, self._options)

        # Capture the pause. Signed against this candidate's own Problem,
        # so a resume is checked rather than merely attempted.
        #
        # `probe=False` deliberately: the pounce#621 model probe exists to
        # catch a *reordering* between capture and replay, and this state
        # is captured from, and resumed on, the same `Problem` object in
        # the same process a moment later — there is no reordering to
        # catch. It is not free either. The probe evaluates the model, and
        # on this path the model is the caller's counted callable, so
        # probing once per rung would spend four of the user's evaluations
        # per resume on a facet that cannot fire: the racing suite went
        # from 2592 to 2920 evaluations for the same three answers, which
        # is what test_the_ladder_beats_the_pre_610_cost caught.
        # Declining it at capture also makes the resume-side check free,
        # because a target is only probed when the artifact carries a
        # probe to compare it against.
        state["ws"] = WarmStart.from_info(x, info, problem=problem, probe=False)
        return spent, used, resumed

    # -- scoring -------------------------------------------------------

    def _score(self, cohort):
        """Composite rung score for `cohort`; smaller is better.

        The five signals pounce#610 names, each rank-transformed within
        the cohort and then weighted:

        1. **violation** — how infeasible the iterate is now;
        2. **feasibility progress** — how much of its initial violation
           it has removed;
        3. **KKT** — the scaled first-order residual, in log units;
        4. **objective progress** — objective removed per evaluation
           spent, so a candidate that bought its descent expensively
           does not outrank a cheap one;
        5. **health** — restoration share, plus a penalty for a
           non-finite objective or a failed (not merely paused) exit.

        A feasible candidate is never outranked by an infeasible one on
        objective alone: the violation term is weighted highest, and an
        infeasible candidate's objective progress is damped below.
        """
        viol, dviol, kkt, dobj, health = [], [], [], [], []
        for c in cohort:
            viol.append(max(c.violation - _FEAS_TOL, 0.0)
                        if np.isfinite(c.violation) else float("inf"))
            v0 = c.violation0 if np.isfinite(c.violation0) else float("inf")
            if not np.isfinite(v0) or v0 <= _FEAS_TOL:
                # Nothing to reduce: neither credit nor blame.
                dviol.append(0.0)
            else:
                cur = c.violation if np.isfinite(c.violation) else v0
                dviol.append(-(v0 - cur) / v0)
            kkt.append(_log_scale(c.kkt))
            if np.isfinite(c.obj) and np.isfinite(c.obj0) and c.evals > 0:
                prog = (c.obj0 - c.obj) / c.evals
                # Objective progress made while running away from the
                # feasible set is not progress on this problem. Damping
                # it here (rather than dropping it) is what makes the
                # adversarial case in the tests come out right without
                # blinding the score on feasible cohorts.
                if not c.feasible:
                    prog *= 0.25
                dobj.append(-prog)
            else:
                dobj.append(float("inf"))
            bad = 0.0
            if c.iters > 0:
                bad += float(c.restoration_calls) / c.iters
            if not np.isfinite(c.obj):
                bad += 10.0
            if c.status is not None and c.status not in _DONE_STATUS \
                    and c.status not in _PAUSED_STATUS:
                bad += 10.0
            health.append(bad)

        parts = {
            "violation": _rank01(viol),
            "feasibility_progress": _rank01(dviol),
            "kkt": _rank01(kkt),
            "objective_progress": _rank01(dobj),
            "health": _rank01(health),
        }
        return [
            sum(self._weights.get(k, 0.0) * parts[k][i] for k in parts)
            for i in range(len(cohort))
        ]

    # -- diversity -----------------------------------------------------

    def _distance(self, a, b):
        return float(np.linalg.norm((np.asarray(a, float) - np.asarray(b, float))
                                    / self._scale))

    def _dedupe(self, ordered):
        """Collapse near-identical candidates, keeping the best of each.

        Two starts that have already fallen into the same basin are one
        candidate wearing two hats; letting both occupy survivor slots
        spends the next rung's budget twice on the same answer and
        squeezes out the basin nobody has looked at yet.
        """
        kept, dropped = [], []
        for c in ordered:
            twin = next(
                (k for k in kept
                 if self._distance(c.x, k.x) <= self._cluster_tol), None)
            if twin is None:
                kept.append(c)
            else:
                dropped.append(
                    (c, f"duplicate of candidate {twin.index} "
                        f"(scaled distance {self._distance(c.x, twin.x):.3g} "
                        f"<= {self._cluster_tol:g})")
                )
        return kept, dropped

    def _explore_pick(self, kept, pool, quota):
        """Retain up to `quota` of `pool`, farthest-first from `kept`.

        The exploration quota is the ladder's admission that its own
        score is a guess made on a fraction of the budget. Choosing the
        *most distant* rather than the next-best-scoring is deliberate:
        the next-best is usually in a basin already represented.
        """
        chosen = []
        remaining = list(pool)
        anchor = list(kept)
        while remaining and len(chosen) < quota:
            best_i, best_d = None, None
            for i, c in enumerate(remaining):
                d = min((self._distance(c.x, a.x) for a in anchor),
                        default=float("inf"))
                if best_d is None or d > best_d:
                    best_i, best_d = i, d
            pick = remaining.pop(best_i)
            chosen.append(pick)
            anchor.append(pick)
        return chosen

    # -- the ladder ----------------------------------------------------

    def run(self):
        cands = [RaceCandidate(index=i, x0=self._starts[i].copy(), x=self._starts[i].copy())
                 for i in range(self._n_starts)]
        # One Problem + one Solver per candidate: the session is the
        # thing being paused, so it cannot be shared.
        states = []
        for i in range(len(cands)):
            if i == 0:
                problem, pobj, counters = self._prepare()
            else:
                # Every candidate builds the same model from the same
                # arguments, so `_prepare_nlp`'s finite-difference and
                # dropped-`hess` warnings are the same warning N times.
                # Emit it once, from the first build.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    problem, pobj, counters = self._prepare()
            states.append({"problem": problem, "problem_obj": pobj,
                           "solver": _Solver(problem), "counters": counters,
                           "ws": None})

        # Measure each start *before* racing. Without this, `obj0` and
        # `violation0` would first be set at the end of rung 0, making
        # "objective progress" and "feasibility reduction" identically
        # zero for everyone on the one rung where the largest number of
        # candidates is eliminated — two of the five signals the issue
        # asks the ranking to use, silently contributing nothing. Two
        # callbacks per candidate, charged to its budget like any other.
        for c, st in zip(cands, states):
            obj0, viol0 = self._measure(st, c.x0)
            c.obj0, c.violation0 = obj0, viol0
            c.obj, c.violation = obj0, viol0
            c.evals += 1 + (1 if self._m else 0)

        report = RaceReport(policy="halving", eta=self._eta,
                            weights=dict(self._weights), candidates=cands)
        alive = list(cands)
        e0 = self._e0

        for rung in range(self._rungs):
            # Rung 0 never carries an evaluation budget, even a pinned
            # one. Two reasons, and the second is the load-bearing one:
            # nothing has been measured yet, so any number would be a
            # guess about how expensive a solve of *this* model is; and a
            # budget small enough to be already spent would leave a
            # candidate never run at all, which is not a ranking, it is a
            # coin toss. Rung 0 runs on the iteration ceiling and
            # *becomes* the measurement — `e0` is then the cohort's worst
            # rung-0 spend and every later rung is a multiple of it.
            # Self-calibrating, and deterministic, being a max over a
            # fixed set.
            budget = None if rung == 0 else int(round(e0 * self._eta ** rung))
            iter_budget = self._iter_budget(rung)
            rec = RaceRound(index=rung, eval_budget=int(budget or 0),
                            iter_budget=iter_budget,
                            entrants=[c.index for c in alive])
            for c in alive:
                spent, used, resumed = self._advance(
                    c, budget, iter_budget, states[c.index])
                rec.evals += spent
                rec.iters += used
                if spent or used:
                    rec.resumed += int(resumed)
                    rec.started += int(not resumed)
            if rung == 0:
                if e0 is None:
                    e0 = float(max((c.evals for c in alive), default=1.0))
                rec.eval_budget = int(round(max(e0, 1.0)))

            # A candidate whose solve failed outright (not merely ran out
            # of budget) is out regardless of rank: there is no state to
            # resume and nothing the next rung would do with it.
            broken = [c for c in alive
                      if c.status is not None and c.status not in _DONE_STATUS
                      and c.status not in _PAUSED_STATUS]
            for c in broken:
                c.eliminated_round = rung
                c.reason = f"solve failed: {c.status_msg or c.status}"
                rec.eliminated.append((c.index, c.reason))
            # Membership by `index`, never by `==`: RaceCandidate is a
            # dataclass holding numpy arrays, so its generated __eq__
            # would compare them element-wise and `bool()` an array.
            gone = {c.index for c in broken}
            alive = [c for c in alive if c.index not in gone]
            if not alive:
                rec.survivors = []
                report.rounds.append(rec)
                break

            scores = self._score(alive)
            for c, s in zip(alive, scores):
                rec.scores[c.index] = float(s)
            # Ties break on the start's own index, so the ordering — and
            # therefore every elimination below it — is a function of the
            # inputs alone.
            ordered = [c for _, _, c in sorted(
                ((s, c.index, c) for c, s in zip(alive, scores)),
                key=lambda t: (t[0], t[1]))]

            if rung == self._rungs - 1:
                rec.survivors = [c.index for c in ordered]
                report.rounds.append(rec)
                alive = ordered
                break

            unique, dupes = self._dedupe(ordered)
            if len(unique) < self._top:
                # Dedup must not cost the caller results it asked for.
                # `ordered` is best-first and `_dedupe` preserves that
                # order in `dupes`, so re-admitting from the front takes
                # back the best of what was collapsed.
                take = self._top - len(unique)
                unique = unique + [c for c, _ in dupes[:take]]
                dupes = dupes[take:]
            for c, why in dupes:
                c.eliminated_round = rung
                c.reason = why
                rec.eliminated.append((c.index, why))

            k = self._survivor_count(rung, len(unique))
            kept, cut = unique[:k], unique[k:]
            if cut and self._explore:
                picked = self._explore_pick(kept, cut, self._explore)
                for c in picked:
                    c.reason = "retained: exploration quota"
                taken = {c.index for c in picked}
                cut = [c for c in cut if c.index not in taken]
                kept = kept + picked
            for pos, c in enumerate(cut):
                c.eliminated_round = rung
                c.reason = (f"below halving cut (rank {k + pos + 1} of "
                            f"{len(unique)}, keep {k})")
                rec.eliminated.append((c.index, c.reason))

            rec.survivors = [c.index for c in kept]
            report.rounds.append(rec)
            alive = kept

        for c in alive:
            if not c.reason:
                c.reason = ("converged" if c.status in _DONE_STATUS
                            else "survived to the final rung")
        if not alive:
            # Every candidate's solve failed. Returning an empty list
            # would make a total failure look like a successful race
            # whose caller forgot to check — and the pre-#610 policy
            # always handed back `top` results whatever their status.
            # Rank the wreckage and return it; the caller reads
            # `.status` / `.success`, as it would have before.
            alive = [c for c in cands if c.result is not None]
        alive.sort(key=self._final_key)
        return [c.result for c in alive[: self._top]], report

    def _final_key(self, c):
        """Sort key for the survivors that are actually returned.

        The pre-#610 policy ordered on raw ``(violation, objective)``.
        That is right for a *converged* pair and wrong for a truncated
        one, and it is wrong in a way that costs the race its answer: a
        candidate that has driven its violation from 15 down to 2e-4 in
        the wrong basin outranks one that reached 2e-2 in the right one,
        purely on the first key, and the caller then continues the
        loser at full effort. (Measured, on HS71 from a six-point hand
        picked field — the fixed policy finds 17.014 and this sort,
        applied to the ladder's survivors, returned 27.146.)

        So: a candidate that has removed all but ``feas_band`` of the
        violation it started with is *on track to feasible*, and among
        those the objective decides. Only candidates that are not on
        track are ordered by violation first. Ties break on the start's
        own index, so the returned order is a function of the inputs.
        """
        viol = c.violation if np.isfinite(c.violation) else float("inf")
        obj = c.obj if np.isfinite(c.obj) else float("inf")
        v0 = c.violation0 if np.isfinite(c.violation0) else 0.0
        band = max(_FEAS_TOL, self._feas_band * v0)
        if viol <= band:
            return (0, 0.0, obj, c.index)
        return (1, viol, obj, c.index)


def _info_evals(info) -> int:
    """Solver-side callback evaluations charged to one solve.

    The solver's own tallies (pounce#610 added them to ``info``), not a
    Python-side wrapper's: they are the frontend-neutral measure, so a
    race driven from ``pounce.jax`` or an ``.nl`` model is budgeted in
    the same currency as one driven from plain callables.
    """
    return sum(int(info.get(k, 0)) for k in (
        "n_obj_evals", "n_grad_evals", "n_constr_evals", "n_jac_evals",
        "n_hess_evals"))


def _Solver(problem):
    """`pounce.Solver` bound late, mirroring ``_continuation._Solver``."""
    from . import _pounce

    return _pounce.Solver(problem)


def _race_fixed(fun, starts, *, jac, bounds, constraints, iters, top, options):
    """The pre-#610 policy, unchanged.

    Kept verbatim — same loop, same truncation, same two-key sort, same
    tie behaviour — so ``policy="fixed"`` reproduces results recorded
    against any earlier release exactly.
    ``python/tests/test_starts_racing.py`` pins that against a
    hand-written transcription of the 0.10.0 body.

    Returns ``(results, rows)``, where `rows` is the per-start
    ``(sort_key_violation, objective, result)`` triple **in start
    order** — recorded as the loop runs, so building
    :func:`_fixed_report` costs no extra solve and changes no behaviour.
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
    rows = list(results)
    results.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in results[: max(1, int(top))]], rows


def _fixed_report(starts, rows, returned) -> RaceReport:
    """A one-rung :class:`RaceReport` for the fixed policy.

    The baseline spends the same budget on every candidate and keeps no
    state between candidates, so its report has exactly one rung, no
    resumes, and one elimination reason. Reporting it through the same
    object as the ladder is what makes the two policies comparable —
    including on the number the issue cares about, total evaluations,
    which the pre-#610 function could not report at all because it threw
    every candidate outside ``top`` away.
    """
    grid = np.atleast_2d(np.asarray(starts, dtype=float))
    kept = {id(r) for r in returned}
    cands = []
    rec = RaceRound(index=0, eval_budget=0, entrants=list(range(len(rows))),
                    started=len(rows))
    for i, (_key, _obj, r) in enumerate(rows):
        c = RaceCandidate(index=i, x0=grid[i].copy(), x=np.asarray(r.x, float))
        c.result = r
        c.obj = c.obj0 = float(r.fun)
        v = float(r.info.get("final_constr_viol", 0.0))
        c.violation = c.violation0 = v if np.isfinite(v) else float("inf")
        c.kkt = float(r.info.get("final_kkt_error", float("nan")))
        c.status, c.status_msg = int(r.status), str(r.message)
        c.iters, c.evals = int(r.nit), _info_evals(r.info)
        c.nfev, c.njev, c.nhev = int(r.nfev), int(r.njev), int(r.nhev)
        c.restoration_calls = int(r.info.get("restoration_calls", 0))
        c.rungs_run = 1
        if id(r) in kept:
            c.reason = "returned by the fixed-budget policy"
            rec.survivors.append(i)
        else:
            c.eliminated_round = 0
            c.reason = "outranked on terminal (violation, objective)"
            rec.eliminated.append((i, c.reason))
        cands.append(c)
    rec.evals = sum(c.evals for c in cands)
    rec.iters = sum(c.iters for c in cands)
    return RaceReport(policy="fixed", rounds=[rec], candidates=cands, eta=1)


def race_starts(
    fun,
    starts,
    *,
    jac=None,
    hess=None,
    args: tuple = (),
    bounds=None,
    constraints=None,
    iters: int = 10,
    top: int = 1,
    options: Optional[dict] = None,
    policy: str = "fixed",
    eta: int = 3,
    rungs: Optional[int] = None,
    eval_budget: Optional[float] = None,
    min_survivors: int = 1,
    min_rung_iters: int = 3,
    feas_band: float = 1e-2,
    explore: int = 1,
    cluster_tol: float = 1e-3,
    weights: Optional[Dict[str, float]] = None,
    return_report: bool = False,
):
    """Race N starting points and return the ``top`` most promising.

    A cheap tournament run before the real solve, so the full-effort
    solve continues only from a start worth the effort — typically with
    ``warm_start=pounce.WarmStart.from_info(res.x, res.info)``.

    Two policies:

    ``"fixed"`` (default)
        Every candidate gets exactly ``max_iter=iters`` from a cold
        start, and the field is ranked once on terminal violation and
        objective. Every start is measured on the same budget, so the
        ranking is as good as that budget allows and no start is
        discarded on a partial one.

    ``"halving"`` (pounce#610, opt-in)
        An adaptive successive-halving ladder. Every candidate gets a
        small evaluation budget; the field is ranked and the weakest
        fraction eliminated; the survivors are **resumed from their held
        solver state** with a budget ``eta`` times larger, and so on.
        The winner ends with about the same effort a fixed race would
        have given it, and the losers cost a fraction of it — measured
        over ``benchmarks/scripts/race_starts_bench.py``, 17.9% fewer
        user-callable evaluations and 34.9% fewer solver iterations.

        **It is not the default, and the reason is measured.** The
        ladder's early rungs rank the field on a handful of iterations,
        and on a strongly multimodal model that ranking carries almost
        no information about which basin ends lowest — so rung 0
        discards the eventual winner. On 2-D Ackley from 27 Sobol
        starts (``iters=40``, so rung 0 is four iterations) the
        eventual global winner is cut at rung 0 in every seed tried,
        ranked 19th, 13th and 24th of 27; the fixed policy reaches
        4e-16 on all three and the ladder reaches 3.57 / 5.38 / 3.57.
        Across an independent five-model set the ladder was 30% cheaper
        and returned a worse answer in 13 of 45 configurations, and it
        got *worse* with more starts, because a larger field is culled
        harder on the same weak signal.

        Nor is it a tuning accident. ``explore`` does not help — it
        retains the *farthest* candidate, which is not the winner —
        and the only setting that recovered the answer,
        ``min_rung_iters=20`` (half the total budget, i.e. one cut),
        cost slightly *more* than the fixed policy on both models. On a
        genuinely multimodal problem the ladder's saving is the quality
        loss. Reach for it when a solver iteration is expensive and the
        basins are few or well separated, and check the answer against
        ``policy="fixed"`` before relying on it.

    **What "resumed" means here.** POUNCE has no API for suspending an
    IPM mid-iteration and re-entering the same algorithm object — each
    ``Solver.solve`` builds its application afresh (see
    ``crates/pounce-py/src/solver.rs``). What a pause *does* carry is the
    whole interior-point iterate: the primal point, all three multiplier
    blocks, and the barrier parameter μ, replayed through the pounce#607
    warm-start path so pounce#606's recentering measures the point it is
    handed. That is materially not a cold restart — measured on the
    ``rastrigin_eq`` fixture in ``python/tests/test_starts_racing.py``,
    eight candidates paused at 5 iterations reach the same answers in 17
    iterations when resumed and 43 when restarted from their own
    iterates; paused at 8, where they have already converged, the resume
    costs **0** iterations against the restart's 43 (see
    ``docs/src/initialization.md``). The size of that gap is
    model-dependent — on HS71 it is a wash. What is *not* carried is the filter
    history and the line-search state; carrying those needs a
    ``Solver.resolve()`` that does not exist yet.

    Args:
        fun, jac, hess, args, bounds, constraints: As
            :func:`pounce.minimize`.
        starts: ``(N, n)`` array of starting points, e.g. from
            :func:`generate_starts`.
        iters: Iteration budget the **winner** ends up with. Under
            ``"fixed"`` every candidate gets exactly this; under
            ``"halving"`` it sets the top of the ladder.
        top: How many results to return, best first.
        options: Solver options, as :func:`pounce.minimize` takes them.
            ``policy="halving"`` runs on the NLP path only and rejects a
            non-``"nlp"`` ``solver_selection`` rather than quietly
            losing the session it needs.
        policy: ``"fixed"`` (default) or ``"halving"``. Every argument
            below applies to ``"halving"`` only.
        eta: Elimination factor — each rung keeps about ``1/eta`` of the
            field and multiplies the survivors' budget by ``eta``.
        rungs: Number of ladder rungs. Default: just enough to walk
            ``N`` down to ``top`` at rate ``eta``.
        eval_budget: Pins the ladder's evaluation unit — rung ``r``'s
            cumulative per-candidate budget is ``eval_budget * eta**r``
            for ``r >= 1``. Default: calibrate it from what rung 0
            actually cost, which needs no guess about how expensive this
            model is. Rung 0 itself is never bounded by it, on either
            setting: a unit small enough to be already spent would leave
            a candidate never run, which is not a ranking.
        min_survivors: Floor on how many candidates a rung leaves.
        min_rung_iters: Iterations rung 0 must be worth. The rung count
            is shortened rather than let the bottom rung fall below
            this: ranking a multi-basin field on the solver's first
            step is noise, and the candidate it discards is as likely
            to be the winner as not.
        feas_band: Fraction of its *initial* violation a survivor may
            still carry and still be ordered on objective rather than
            on feasibility. See :meth:`_Racer._final_key`.
        explore: How many candidates outside the cut to retain anyway,
            chosen farthest-first from the retained set. ``0`` disables
            the exploration quota.
        cluster_tol: Scaled distance at which two survivors count as the
            same candidate and the worse of the pair is dropped.
        weights: Overrides for the composite score's weights; see
            :data:`DEFAULT_RACE_WEIGHTS`.
        return_report: Also return a :class:`RaceReport` with per-rung
            resource use and per-candidate elimination reasons.

    Returns:
        ``list[OptimizeResult]``, best first — or
        ``(list, RaceReport)`` when ``return_report=True``.
    """
    policy = str(policy).lower()
    if policy not in ("halving", "fixed"):
        raise ValueError(
            f"race_starts: unknown policy {policy!r}; expected 'halving' or "
            "'fixed'"
        )
    if policy == "fixed":
        if hess is not None or args:
            raise TypeError(
                "race_starts(policy='fixed'): hess= / args= are pounce#610 "
                "additions and are not part of the frozen baseline policy; "
                "pass them with policy='halving', or fold them into `fun`."
            )
        results, rows = _race_fixed(
            fun, starts, jac=jac, bounds=bounds, constraints=constraints,
            iters=iters, top=top, options=options,
        )
        if return_report:
            return results, _fixed_report(starts, rows, results)
        return results

    racer = _Racer(
        fun, starts, jac=jac, hess=hess, args=args, bounds=bounds,
        constraints=constraints, iters=iters, top=top, options=options,
        eta=eta, rungs=rungs, eval_budget=eval_budget,
        min_survivors=min_survivors, min_rung_iters=min_rung_iters,
        feas_band=feas_band, explore=explore, cluster_tol=cluster_tol,
        weights=weights,
    )
    results, report = racer.run()
    return (results, report) if return_report else results
