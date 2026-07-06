"""Find saddle points and other critical points — two routes.

A critical (stationary) point of ``f`` is any ``x`` with ``∇f(x) = 0``;
minima, saddles, and maxima are distinguished by the **Morse index**, the
number of negative Hessian eigenvalues (0 = minimum, 1 = transition state,
… , n = maximum). The same repulsion / restart ideas behind
:func:`pounce.find_minima` extend to *all* critical points in two ways:

* **Route A — :func:`find_critical_points`.** Stationary points are the
  roots of ``∇f(x) = 0``. Minimizing the gradient-norm merit
  ``½‖∇f(x)‖²`` drives the solver onto them (the merit is zero there), and
  :func:`find_minima`'s deflation / multistart enumerates the distinct ones.
  Each is then classified by its Morse index. This treats pounce as a
  root-finder and reuses the whole ``find_minima`` machine.

* **Route B — :func:`find_saddles`.** Reframe saddle-finding as
  minimization: at an index-``k`` saddle the energy is a *maximum* along the
  ``k`` softest Hessian modes and a minimum in the rest. Eigenvector
  following (Cerjan & Miller 1981; the dimer method of Henkelman & Jónsson
  1999 is the Hessian-free cousin) walks uphill along those modes and
  downhill in the others, landing on an index-``k`` saddle. Wrapping it in
  multistart enumerates several.

References: Cerjan & Miller, *J. Chem. Phys.* **75**, 2800 (1981);
Henkelman & Jónsson, *J. Chem. Phys.* **111**, 7010 (1999),
doi:10.1063/1.480097; E & Zhou, *Nonlinearity* **24**, 1831 (2011),
doi:10.1088/0951-7715/24/6/008; Farrell, Birkisson & Funke, *SIAM J. Sci.
Comput.* **37**, A2026 (2015), doi:10.1137/140984798.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._minima import find_minima, _scale_from_bounds
from ._minimize import _validate_bounds_length

__all__ = [
    "find_critical_points", "find_saddles", "reaction_network",
    "CriticalPoint", "CriticalPointResult", "Connection", "ReactionNetwork",
]


# --------------------------------------------------------------------------
@dataclass
class CriticalPoint:
    x: np.ndarray
    f: float
    index: int            # Morse index = # negative Hessian eigenvalues
    eigvalues: np.ndarray
    grad_norm: float
    degenerate: bool = False   # a Hessian eigenvalue is ~0 (non-Morse point)

    @property
    def kind(self) -> str:
        if self.degenerate:
            return f"degenerate(index>={self.index})"
        if self.index == 0:
            return "minimum"
        if self.index == self.x.size:
            return "maximum"
        return f"saddle(index={self.index})"


@dataclass
class CriticalPointResult:
    points: list[CriticalPoint]
    status: str
    n_solves: int
    trace: list[dict] = field(default_factory=list)

    def of_index(self, index: int) -> list[CriticalPoint]:
        return [p for p in self.points if p.index == index]

    @property
    def minima(self):
        return self.of_index(0)

    @property
    def saddles(self):
        return [p for p in self.points if 0 < p.index < p.x.size]

    @property
    def maxima(self):
        return [p for p in self.points if p.index == p.x.size]

    def __len__(self):
        return len(self.points)


def _morse_index(H, eig_tol):
    """Morse index, eigenvalues, and a degeneracy flag (an eigenvalue ~0,
    so the point is non-Morse and the min/saddle/max label is unreliable)."""
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    index = int(np.sum(eig < -eig_tol))
    degenerate = bool(np.any(np.abs(eig) <= eig_tol))
    return index, eig, degenerate


def _critical_point(fun, grad, hess, x, eig_tol):
    x = np.asarray(x, float)
    gn = float(np.linalg.norm(np.asarray(grad(x), float).ravel()))
    index, eig, degen = _morse_index(np.asarray(hess(x), float), eig_tol)
    return CriticalPoint(x, float(fun(x)), index, eig, gn, degen)


# --------------------------------------------------------------------------
# Route A: enumerate critical points as roots of grad f = 0.
# --------------------------------------------------------------------------
def find_critical_points(
    fun: Callable,
    x0,
    *,
    grad: Callable,
    hess: Callable,
    bounds: Sequence | None = None,
    method: str = "multistart",
    n_points: int = 20,
    max_solves: int | None = None,
    patience: int = 20,
    dedup: float = 1e-3,
    grad_tol: float = 1e-6,
    eig_tol: float = 1e-8,
    options: Mapping[str, Any] | None = None,
    strategy_kw: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> CriticalPointResult:
    """Enumerate critical points of ``fun`` and classify by Morse index.

    Minimizes the gradient-norm merit ``½‖∇f‖²`` (zero exactly at stationary
    points) with :func:`find_minima`, then keeps points where ``‖∇f‖`` is
    below ``grad_tol`` and labels each by the number of negative Hessian
    eigenvalues. ``method`` selects the enumeration strategy passed to
    ``find_minima`` (``"multistart"``, ``"deflation"``, ``"flooding"`` …).
    """
    def merit(x):
        g = np.asarray(grad(x), float).ravel()
        return 0.5 * float(g @ g)

    def merit_jac(x):
        H = np.asarray(hess(x), float)
        g = np.asarray(grad(x), float).ravel()
        return H @ g                      # ∇(½‖g‖²) = Hᵀg (H symmetric)

    def merit_hess(x):
        H = np.asarray(hess(x), float)
        return H @ H                      # Gauss-Newton (drops 3rd-derivative term)

    r = find_minima(
        merit, x0, method=method, jac=merit_jac, hess=merit_hess,
        bounds=bounds, n_minima=n_points, max_solves=max_solves,
        patience=patience, dedup=dedup, options=options,
        strategy_kw=strategy_kw, seed=seed,
    )

    points: list[CriticalPoint] = []
    for x in r.minima:
        x = np.asarray(x, float)
        g = np.asarray(grad(x), float).ravel()
        gn = float(np.linalg.norm(g))
        if not np.isfinite(gn) or gn > grad_tol:
            continue                      # non-finite or spurious (∇f ≠ 0)
        H = np.asarray(hess(x), float)
        if not np.all(np.isfinite(H)):
            continue                      # guard the eigendecomposition
        index, eig, degen = _morse_index(H, eig_tol)
        points.append(CriticalPoint(x, float(fun(x)), index, eig, gn, degen))

    points.sort(key=lambda p: (p.index, p.f))
    return CriticalPointResult(points, r.status, r.n_solves, r.trace)


# --------------------------------------------------------------------------
# Route B: eigenvector-following local search for index-k saddles.
# --------------------------------------------------------------------------
def _clip_to_bounds(x, bounds):
    if bounds is None:
        return x
    lo = np.array([(-np.inf if b is None or b[0] is None else b[0]) for b in bounds])
    hi = np.array([(np.inf if b is None or b[1] is None else b[1]) for b in bounds])
    return np.clip(x, lo, hi)


def _eigvec_follow(grad, hess, x0, k, max_iter, tol, max_step, bounds=None):
    """Walk to an index-k saddle: ascend the k softest modes, Newton-descend
    the rest (Cerjan-Miller eigenvector following). Steps are clipped to the
    bounds box so the walk stays feasible."""
    x = _clip_to_bounds(np.asarray(x0, float).copy(), bounds)
    for _ in range(max_iter):
        g = np.asarray(grad(x), float).ravel()
        if not np.all(np.isfinite(g)):
            break
        if np.linalg.norm(g) < tol:
            break
        H = 0.5 * (np.asarray(hess(x), float) + np.asarray(hess(x), float).T)
        if not np.all(np.isfinite(H)):
            break
        w, U = np.linalg.eigh(H)          # eigenvalues ascending
        gu = U.T @ g
        denom = np.maximum(np.abs(w), 1e-8)
        step = -gu / denom                # Newton: descend in every mode
        step[:k] = gu[:k] / denom[:k]     # flip the k softest → ascend
        dx = U @ step
        nrm = np.linalg.norm(dx)
        if nrm > max_step:
            dx *= max_step / nrm
        x = _clip_to_bounds(x + dx, bounds)
    return x, float(np.linalg.norm(np.asarray(grad(x), float).ravel()))


def find_saddles(
    fun: Callable,
    x0,
    *,
    grad: Callable,
    hess: Callable,
    bounds: Sequence | None = None,
    index: int = 1,
    n_saddles: int = 10,
    max_solves: int | None = None,
    patience: int = 30,
    dedup: float = 1e-3,
    grad_tol: float = 1e-6,
    eig_tol: float = 1e-8,
    max_step: float = 0.2,
    local_max_iter: int = 200,
    distance: Callable | None = None,
    seed: int | None = None,
) -> CriticalPointResult:
    """Find index-``index`` saddle points by multistart eigenvector following.

    Each start is driven to a saddle by :func:`_eigvec_follow`; results with
    the requested Morse index, a converged gradient, and within bounds are
    de-duplicated and collected. ``distance`` overrides the dedup metric
    (default: per-dimension scaled Euclidean, matching :func:`find_minima`);
    :func:`reaction_network` passes a mode-aware metric so a flat Hessian
    direction cannot spawn duplicate saddles (pounce#183).
    """
    x0 = np.atleast_1d(np.asarray(x0, float))
    _validate_bounds_length(bounds, x0.size)
    rng = np.random.default_rng(seed)
    n = x0.size
    if not (1 <= index <= n):
        raise ValueError(
            f"index must be between 1 and the dimension {n} for a saddle, "
            f"got {index}"
        )
    if n_saddles < 1:
        raise ValueError(f"n_saddles must be >= 1, got {n_saddles}")
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")
    if max_solves is not None and max_solves < 1:
        raise ValueError(f"max_solves must be >= 1, got {max_solves}")
    if max_solves is None:
        max_solves = 6 * n_saddles
    # Dedup in the same per-dimension scaled metric find_minima uses, so a
    # single `dedup` tolerance means the same thing across routes.
    L, _ = _scale_from_bounds(bounds, n)
    sdist = distance or (lambda a, b: float(np.linalg.norm((a - b) / L)))

    def in_bounds(x):
        if bounds is None:
            return True
        for xi, bd in zip(x, bounds):
            if bd is None:
                continue
            lo, hi = bd
            if (lo is not None and xi < lo - 1e-9) or (hi is not None and xi > hi + 1e-9):
                return False
        return True

    def sample():
        if bounds is not None and all(
            b is not None and b[0] is not None and b[1] is not None for b in bounds
        ):
            lo = np.array([b[0] for b in bounds], float)
            hi = np.array([b[1] for b in bounds], float)
            return lo + (hi - lo) * rng.random(n)
        return x0 + rng.standard_normal(n)

    found: list[CriticalPoint] = []
    trace: list[dict] = []
    n_solves = 0
    stagnant = 0
    start = x0.copy()
    while len(found) < n_saddles and n_solves < max_solves and stagnant < patience:
        x, gn = _eigvec_follow(grad, hess, start, index, local_max_iter,
                               grad_tol, max_step, bounds)
        n_solves += 1
        # Short-circuit on the gradient before any eigendecomposition, so a
        # diverged / non-finite iterate never reaches eigh.
        ok = bool(np.all(np.isfinite(x))) and gn <= grad_tol and in_bounds(x)
        idx = -1
        if ok:
            idx, eig, degen = _morse_index(np.asarray(hess(x), float), eig_tol)
            known = any(sdist(x, p.x) <= dedup for p in found)
            ok = idx == index and not known
        trace.append({"x": x, "grad_norm": gn, "index": idx, "accepted": ok})
        if ok:
            found.append(CriticalPoint(x, float(fun(x)), idx, eig, gn, degen))
            stagnant = 0
        else:
            stagnant += 1
        start = sample()

    if len(found) >= n_saddles:
        status = "target_reached"
    elif stagnant >= patience:
        status = "converged"
    else:
        status = "budget_exhausted"
    found.sort(key=lambda p: p.f)
    return CriticalPointResult(found, status, n_solves, trace)


# --------------------------------------------------------------------------
# Reaction network: minima (states) + index-1 saddles (transition states)
# + the connectivity / barrier table between them.
# --------------------------------------------------------------------------
@dataclass
class Connection:
    """A transition state and the two minima it joins."""

    ts: CriticalPoint
    minima: tuple[int, int]        # indices into ReactionNetwork.minima (-1 = unconnected)
    barrier: tuple[float, float]   # E(ts) - E(min) for each side, matching `minima`
    path: np.ndarray               # approximate min-energy path: min_i -> ts -> min_j


@dataclass
class ReactionNetwork:
    minima: list[CriticalPoint]            # stable states, sorted by energy
    transition_states: list[CriticalPoint]
    connections: list[Connection]
    status: str
    n_solves: int

    @property
    def edges(self):
        return [c.minima for c in self.connections]

    def neighbors(self, i: int):
        out = set()
        for c in self.connections:
            a, b = c.minima
            if a == i and b >= 0 and b != i:
                out.add(b)
            if b == i and a >= 0 and a != i:
                out.add(a)
        return sorted(out)

    def barrier(self, i: int, j: int) -> float:
        """Lowest single-step barrier from minimum ``i`` to minimum ``j``
        (``inf`` if no transition state directly connects them)."""
        best = np.inf
        for c in self.connections:
            a, b = c.minima
            if (a, b) == (i, j):
                best = min(best, c.barrier[0])
            elif (a, b) == (j, i):
                best = min(best, c.barrier[1])
        return best

    def path_between(self, i: int, j: int):
        """The connecting MEP oriented from minimum ``i`` to minimum ``j``."""
        for c in self.connections:
            a, b = c.minima
            if (a, b) == (i, j):
                return c.path
            if (a, b) == (j, i):
                return c.path[::-1]
        return None

    def summary(self) -> str:
        lines = [f"{len(self.minima)} states, "
                 f"{len(self.transition_states)} transition states "
                 f"({self.status}, {self.n_solves} solves)"]
        lines.append("states:")
        for k, m in enumerate(self.minima):
            tag = "  (global)" if k == 0 else ""
            lines.append(f"  {k}: ({m.x[0]:+.4f}, {m.x[1]:+.4f})  E={m.f:8.3f}{tag}"
                         if m.x.size == 2 else f"  {k}: E={m.f:8.3f}{tag}")
        lines.append("barriers:")
        for c in self.connections:
            i, j = c.minima
            lines.append(f"  state {i} <-> state {j}  via TS E={c.ts.f:8.3f}   "
                         f"{i}->{j}: {c.barrier[0]:7.3f}   {j}->{i}: {c.barrier[1]:7.3f}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.minima)


def _descend(grad, x_start, minima_x, ds, max_steps, reach, distance=None):
    """Normalized steepest-descent path; stop on reaching a known minimum.

    ``distance`` overrides the reach metric (default: Euclidean). The
    reaction-network caller passes a mode-aware metric so a descent that
    parks at a different point along a flat Hessian direction still matches
    the minimum in that basin instead of recording an unmapped endpoint
    (pounce#183).
    """
    dist = distance or (lambda a, b: float(np.linalg.norm(a - b)))
    x = np.asarray(x_start, float).copy()
    path = [x.copy()]
    for _ in range(max_steps):
        g = np.asarray(grad(x), float).ravel()
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        x = x - ds * g / ng
        path.append(x.copy())
        if minima_x:
            d = [dist(x, m) for m in minima_x]
            j = int(np.argmin(d))
            if d[j] < reach:
                path.append(np.asarray(minima_x[j], float))
                return np.array(path), j
    if minima_x:
        d = [dist(x, m) for m in minima_x]
        j = int(np.argmin(d))
        if d[j] < reach:
            return np.array(path), j
    return np.array(path), -1


def _mode_aware_distance(hess, eig_tol, L):
    """Dedup metric that ignores displacement along flat (near-null) modes.

    The default minima-dedup metric is full-coordinate Euclidean distance.
    When the PES has a genuine zero eigenmode — rigid translation/rotation
    of any molecule, or an intrinsically flat coordinate — a minimizer can
    halt anywhere along that direction, so two coordinates that are the
    *same* basin land arbitrarily far apart and dedup keeps them as
    distinct minima. On a fixed ``n_states`` budget those copies crowd out
    genuinely different basins (pounce#183).

    Measuring the displacement only in the **non-null subspace** of the
    Hessian at the already-accepted minimum ``b`` collapses zero-mode
    copies onto one representative while leaving stiff directions
    untouched. When every mode is stiff the projector is the identity and
    this reduces exactly to the ordinary per-dimension scaled metric, so
    it is a safe always-on refinement rather than a behavior change for
    well-conditioned surfaces.

    ``dedup`` (the caller's tolerance) is still interpreted in the same
    per-dimension scaled space ``(a - b) / L``; projection only removes
    components, so a retained-mode separation still reads the same
    magnitude.
    """
    L = np.asarray(L, float)
    # The archive only ever compares a candidate against a *stored*
    # minimum (the second argument), and there are at most ``n_states`` of
    # them, so caching the projector per accepted minimum keeps this to one
    # Hessian evaluation + eigendecomposition per basin.
    cache: dict[bytes, np.ndarray] = {}

    def projector(b):
        key = b.tobytes()
        P = cache.get(key)
        if P is None:
            H = np.asarray(hess(b), float)
            H = 0.5 * (H + H.T)
            w, U = np.linalg.eigh(H)
            keep = np.abs(w) > eig_tol
            Uk = U[:, keep]
            # Uk @ Ukᵀ projects onto the span of the stiff modes; identity
            # when `keep` is all-True (no null modes to quotient out).
            P = Uk @ Uk.T
            cache[key] = P
        return P

    def distance(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return float(np.linalg.norm(projector(b) @ ((a - b) / L)))

    return distance


def reaction_network(
    fun: Callable,
    x0,
    *,
    grad: Callable,
    hess: Callable,
    bounds: Sequence | None = None,
    n_states: int = 10,
    n_transition_states: int = 10,
    minima_method: str = "flooding",
    dedup: float = 1e-2,
    max_solves: int | None = None,
    patience: int = 40,
    grad_tol: float = 1e-6,
    eig_tol: float = 1e-8,
    descent_step: float = 0.01,
    descent_reach: float = 0.05,
    connect_offset: float = 0.02,
    options: Mapping[str, Any] | None = None,
    minima_kw: Mapping[str, Any] | None = None,
    saddle_kw: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> ReactionNetwork:
    """Map the reaction network of a potential energy surface ``fun``.

    Finds the stable states (minima) and the index-1 transition states
    (saddles) between them, then connects each transition state to the two
    minima it joins — by descending its unstable mode into each adjacent
    basin — and tabulates the barrier heights.

    Parameters mirror :func:`find_minima` / :func:`find_saddles`. Use
    ``minima_method`` and ``minima_kw`` to tune the state search, and
    ``saddle_kw`` for the transition-state search (e.g.
    ``{"max_step": 0.05, "grad_tol": 1e-5}``). The eigenvector-following
    saddle search rarely tightens below ~1e-5, so ``grad_tol`` for the
    transition-state phase is floored at ``1e-5`` unless ``saddle_kw``
    sets it explicitly. A transition state whose two descent directions land
    in the *same* basin is recorded with ``minima == (i, i)`` but is not
    reported as an edge by :meth:`ReactionNetwork.neighbors`.

    Returns
    -------
    ReactionNetwork
        ``.minima`` (states, sorted by energy), ``.transition_states``,
        ``.connections`` (each with the joined minima, both barriers, and the
        minimum-energy path), plus ``.barrier(i, j)``, ``.neighbors(i)``,
        ``.path_between(i, j)`` and ``.summary()``.
    """
    # Generous default budgets: locating every state/TS usually needs more
    # than find_minima's lean 4*n default.
    ms_min = max_solves if max_solves is not None else 40 * n_states
    ms_ts = max_solves if max_solves is not None else 40 * n_transition_states

    # Mode-aware dedup: quotient out flat (near-null) Hessian directions so
    # a zero eigenmode cannot manufacture arbitrarily many "distinct" minima
    # that exhaust the `n_states` budget before flooding reaches other basins
    # (pounce#183). Reduces to the ordinary scaled metric when no null modes
    # are present.
    n_vars = np.atleast_1d(np.asarray(x0, float)).size
    L, _ = _scale_from_bounds(bounds, n_vars)
    mode_distance = _mode_aware_distance(hess, eig_tol, L)

    mres = find_minima(
        fun, x0, method=minima_method, jac=grad, hess=hess, bounds=bounds,
        n_minima=n_states, max_solves=ms_min, patience=patience,
        dedup=dedup, options=options, strategy_kw=minima_kw, seed=seed,
        distance=mode_distance,
    )
    minima = [_critical_point(fun, grad, hess, x, eig_tol) for x in mres.minima]
    minima_x = [m.x for m in minima]

    skw = dict(saddle_kw or {})
    skw.setdefault("grad_tol", max(grad_tol, 1e-5))
    sres = find_saddles(
        fun, x0, grad=grad, hess=hess, bounds=bounds, index=1,
        n_saddles=n_transition_states, max_solves=ms_ts,
        patience=patience, dedup=dedup, eig_tol=eig_tol,
        distance=mode_distance, seed=seed, **skw,
    )

    connections: list[Connection] = []
    for p in sres.points:
        H = 0.5 * (np.asarray(hess(p.x), float) + np.asarray(hess(p.x), float).T)
        _, U = np.linalg.eigh(H)
        v = U[:, 0]                      # unstable (softest) eigenvector
        segs, ends = [], []
        for sgn in (+1.0, -1.0):
            seg, j = _descend(grad, p.x + connect_offset * sgn * v, minima_x,
                              descent_step, 3000, descent_reach,
                              distance=mode_distance)
            segs.append(seg)
            ends.append(j)
        i, j = ends
        bi = p.f - minima[i].f if i >= 0 else float("nan")
        bj = p.f - minima[j].f if j >= 0 else float("nan")
        path = np.vstack([segs[0][::-1], p.x[None, :], segs[1]])
        connections.append(Connection(p, (i, j), (bi, bj), path))

    status = f"minima:{mres.status}, ts:{sres.status}"
    return ReactionNetwork(
        minima, list(sres.points), connections, status,
        mres.n_solves + sres.n_solves,
    )
