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

from ._minima import find_minima

__all__ = ["find_critical_points", "find_saddles", "CriticalPoint", "CriticalPointResult"]


# --------------------------------------------------------------------------
@dataclass
class CriticalPoint:
    x: np.ndarray
    f: float
    index: int            # Morse index = # negative Hessian eigenvalues
    eigvalues: np.ndarray
    grad_norm: float

    @property
    def kind(self) -> str:
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
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    return int(np.sum(eig < -eig_tol)), eig


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
    grad = grad
    hess = hess

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
        gn = float(np.linalg.norm(np.asarray(grad(x), float).ravel()))
        if gn > grad_tol:
            continue                      # spurious merit minimum, ∇f ≠ 0
        index, eig = _morse_index(np.asarray(hess(x), float), eig_tol)
        points.append(CriticalPoint(x, float(fun(x)), index, eig, gn))

    points.sort(key=lambda p: (p.index, p.f))
    return CriticalPointResult(points, r.status, r.n_solves, r.trace)


# --------------------------------------------------------------------------
# Route B: eigenvector-following local search for index-k saddles.
# --------------------------------------------------------------------------
def _eigvec_follow(grad, hess, x0, k, max_iter, tol, max_step):
    """Walk to an index-k saddle: ascend the k softest modes, Newton-descend
    the rest (Cerjan-Miller eigenvector following)."""
    x = np.asarray(x0, float).copy()
    for _ in range(max_iter):
        g = np.asarray(grad(x), float).ravel()
        if np.linalg.norm(g) < tol:
            break
        H = 0.5 * (np.asarray(hess(x), float) + np.asarray(hess(x), float).T)
        w, U = np.linalg.eigh(H)          # eigenvalues ascending
        gu = U.T @ g
        denom = np.maximum(np.abs(w), 1e-8)
        step = -gu / denom                # Newton: descend in every mode
        step[:k] = gu[:k] / denom[:k]     # flip the k softest → ascend
        dx = U @ step
        nrm = np.linalg.norm(dx)
        if nrm > max_step:
            dx *= max_step / nrm
        x = x + dx
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
    seed: int | None = None,
) -> CriticalPointResult:
    """Find index-``index`` saddle points by multistart eigenvector following.

    Each start is driven to a saddle by :func:`_eigvec_follow`; results with
    the requested Morse index, a converged gradient, and within bounds are
    de-duplicated and collected.
    """
    x0 = np.asarray(x0, float)
    rng = np.random.default_rng(seed)
    n = x0.size
    if max_solves is None:
        max_solves = 6 * n_saddles

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
        x, gn = _eigvec_follow(grad, hess, start, index, local_max_iter, grad_tol, max_step)
        n_solves += 1
        idx, eig = _morse_index(np.asarray(hess(x), float), eig_tol)
        known = any(np.linalg.norm(x - p.x) <= dedup for p in found)
        ok = gn <= grad_tol and idx == index and in_bounds(x) and not known
        trace.append({"x": x, "grad_norm": gn, "index": idx, "accepted": ok})
        if ok:
            found.append(CriticalPoint(x, float(fun(x)), idx, eig, gn))
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
