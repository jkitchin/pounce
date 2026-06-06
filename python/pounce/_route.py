"""Structure detection + extraction to auto-route a scipy-style
:func:`pounce.minimize` problem to the specialized convex LP/QP solver.

The CLI classifies a problem by walking its symbolic ``.nl`` expression tree,
so its routing is *certain*. ``minimize`` takes opaque Python callables
(``fun``/``jac``/``hess`` and constraint functions), so we cannot read the
structure — we have to **probe** the callables at several points, fit a
linear/quadratic model, and then **validate** that model against the true
functions at held-out points before trusting it.

Detection is deliberately conservative. The two misclassification directions
are not symmetric:

* a convex LP/QP routed to the NLP solver is merely *slower* — the filter-IPM
  solves convex QPs correctly;
* a genuinely nonlinear problem routed to the QP solver returns a **silently
  wrong** answer.

So the held-out validation gates the dangerous direction: any probe that
raises, any model mismatch beyond tolerance, a non-constant Hessian/Jacobian,
or an indefinite Hessian (nonconvex QP) all fall back to ``None`` — meaning
"let the general NLP solver handle it."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

_EPS = float(np.finfo(np.float64).eps)
# Central-difference steps: ~eps^(1/2) for a first derivative (gradient) and
# ~eps^(1/3) for a second derivative (Hessian), the usual optimal balances of
# truncation vs. round-off error.
_H_GRAD = _EPS**0.5
_H_HESS = _EPS ** (1.0 / 3.0)


@dataclass
class QpExtract:
    """A convex LP/QP recovered from the callable problem.

    ``kind`` is ``"lp"`` (``P is None``) or ``"convex_qp"``. The objective is
    ``½ xᵀP x + cᵀx + obj_const``; ``obj_const`` is the degree-0 term that the
    QP solver does not see and must be added back to the reported value.
    Equality block is ``A x = b``, inequality block ``G x ≤ h``, with box
    ``lb ≤ x ≤ ub`` (either may be ``None``).
    """

    kind: str
    P: Optional[np.ndarray]
    c: np.ndarray
    obj_const: float
    A: Optional[np.ndarray]
    b: Optional[np.ndarray]
    G: Optional[np.ndarray]
    h: Optional[np.ndarray]
    lb: Optional[np.ndarray]
    ub: Optional[np.ndarray]


class _NotConvex(Exception):
    """Internal sentinel: the problem is not a confidently-convex LP/QP."""


def _grad_fn(fun: Callable, jac: Optional[Callable]) -> Callable:
    """Return a gradient callable: the user's ``jac`` if given, else a
    central finite-difference of ``fun`` (central, not forward, because the
    structure tests want the extra accuracy)."""
    if jac is not None:
        return lambda x: np.asarray(jac(x), dtype=np.float64).ravel()

    def g(x):
        out = np.empty(x.size)
        for i in range(x.size):
            step = _H_GRAD * max(1.0, abs(x[i]))
            xp = x.copy()
            xm = x.copy()
            xp[i] += step
            xm[i] -= step
            out[i] = (float(fun(xp)) - float(fun(xm))) / (2.0 * step)
        return out

    return g


def _hessian(grad: Callable, x: np.ndarray, hess: Optional[Callable]) -> np.ndarray:
    """Symmetric Hessian at ``x`` — the user's ``hess`` if given, else a
    central finite-difference of the gradient."""
    if hess is not None:
        return np.asarray(hess(x), dtype=np.float64).reshape(x.size, x.size)
    n = x.size
    H = np.empty((n, n))
    for j in range(n):
        step = _H_HESS * max(1.0, abs(x[j]))
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        H[:, j] = (grad(xp) - grad(xm)) / (2.0 * step)
    return 0.5 * (H + H.T)


def _probe_points(x0, lb, ub, rng, k=5):
    """``x0`` plus ``k`` random in-domain probe points.

    Steps are scaled to the box width (when finite) or to ``max(1, |x0|)``,
    and clipped back into ``[lb, ub]`` so we never evaluate the user's
    functions outside their declared domain (a log-barrier objective, say).
    The first point is the anchor used to read off coefficients; the rest are
    held out for validation.
    """
    n = x0.size
    if lb is not None and ub is not None:
        width = ub - lb
        finite = np.isfinite(width)
        span = np.where(finite, np.maximum(width, 1e-6) * 0.25,
                        np.maximum(np.abs(x0), 1.0))
    else:
        span = np.maximum(np.abs(x0), 1.0)
    pts = [x0.copy()]
    for _ in range(k):
        p = x0 + span * rng.standard_normal(n)
        if lb is not None:
            p = np.maximum(p, lb)
        if ub is not None:
            p = np.minimum(p, ub)
        pts.append(p)
    return pts


def _objective_model(fun, grad, hess, probes):
    """Fit the objective to ``c·x + d`` (LP) or ``½xᵀPx + c·x + d`` (QP).

    Returns ``(P_or_None, c, d)``; raises :class:`_NotConvex` if the gradient
    is not affine-consistent enough to be a quadratic with a *constant*
    Hessian. The quadratic vs. linear vs. nonlinear decision is finalized by
    the held-out validation in :func:`classify_and_extract`.
    """
    anchor = probes[0]
    grads = [grad(p) for p in probes]
    g0 = grads[0]
    gscale = max(1.0, float(np.max(np.abs(g0))))

    # Linear objective ⇔ the gradient is the same at every probe.
    gvar = max(float(np.max(np.abs(gi - g0))) for gi in grads[1:])
    if gvar <= 1e-7 * gscale:
        c = g0
        d = float(fun(anchor)) - float(c @ anchor)
        return None, c, d

    # Otherwise fit a quadratic. With finite differences, require the Hessian
    # to be constant across two probes (a true quadratic's is); with an exact
    # user ``hess`` one evaluation already pins it.
    P = _hessian(grad, anchor, hess)
    if hess is None:
        P2 = _hessian(grad, probes[1], hess)
        pscale = max(1.0, float(np.max(np.abs(P))))
        if float(np.max(np.abs(P - P2))) > 1e-4 * pscale:
            raise _NotConvex("Hessian is not constant — objective is not quadratic")
    # grad(x) = P x + c  ⇒  c = grad(anchor) − P·anchor.
    c = g0 - P @ anchor
    d = float(fun(anchor)) - (0.5 * float(anchor @ P @ anchor) + float(c @ anchor))
    return P, c, d


def _linear_constraints(g_combined, jac_combined, cl, cu, probes, m):
    """Recover ``A x = b`` / ``G x ≤ h`` from the coalesced constraint
    callable, or raise :class:`_NotConvex` if any constraint is nonlinear.

    ``cl``/``cu`` carry the scipy-style two-sided bounds that
    ``_wrap_constraints`` produced (``[0, 0]`` for an equality, ``[0, ∞]``
    for ``g(x) ≥ 0``). The constraint value model is ``g(x) = J x + g0``.
    """
    if m == 0:
        return None, None, None, None

    anchor = probes[0]
    J0 = np.atleast_2d(np.asarray(jac_combined(anchor), dtype=np.float64))
    g_anchor = np.asarray(g_combined(anchor), dtype=np.float64).ravel()
    g0 = g_anchor - J0 @ anchor  # the affine offset

    jscale = max(1.0, float(np.max(np.abs(J0))))
    for p in probes[1:]:
        gp = np.asarray(g_combined(p), dtype=np.float64).ravel()
        model = J0 @ p + g0
        if float(np.max(np.abs(gp - model))) > 1e-6 * (1.0 + float(np.max(np.abs(gp)))):
            raise _NotConvex("a constraint is nonlinear")
        Jp = np.atleast_2d(np.asarray(jac_combined(p), dtype=np.float64))
        if float(np.max(np.abs(Jp - J0))) > 1e-6 * jscale:
            raise _NotConvex("a constraint Jacobian is not constant")

    A_rows, b_vals, G_rows, h_vals = [], [], [], []
    for i in range(m):
        Ji, off = J0[i], g0[i]
        lo, hi = cl[i], cu[i]
        if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
            # Equality g = lo  ⇒  J x = lo − off.
            A_rows.append(Ji)
            b_vals.append(lo - off)
            continue
        if np.isfinite(hi):
            # g ≤ hi  ⇒  J x ≤ hi − off.
            G_rows.append(Ji)
            h_vals.append(hi - off)
        if np.isfinite(lo):
            # g ≥ lo  ⇒  −J x ≤ off − lo.
            G_rows.append(-Ji)
            h_vals.append(off - lo)

    A = np.array(A_rows, dtype=np.float64) if A_rows else None
    b = np.array(b_vals, dtype=np.float64) if b_vals else None
    G = np.array(G_rows, dtype=np.float64) if G_rows else None
    h = np.array(h_vals, dtype=np.float64) if h_vals else None
    return A, b, G, h


def _clean_bounds(lb, ub):
    """Drop an all-infinite bound vector to ``None`` (no box)."""
    if lb is not None and np.all(np.isinf(lb)):
        lb = None
    if ub is not None and np.all(np.isinf(ub)):
        ub = None
    return lb, ub


def classify_and_extract(
    *,
    fun,
    jac,
    hess,
    lb,
    ub,
    m,
    g_combined,
    jac_combined,
    cl,
    cu,
    x0,
    rtol: float = 1e-5,
    seed: int = 0,
) -> Optional[QpExtract]:
    """Detect a convex LP/QP behind the callable problem and extract its data.

    Returns a :class:`QpExtract` if the objective is linear or convex-quadratic
    *and* every constraint is linear (validated at held-out probe points),
    otherwise ``None`` (route to the NLP solver). Any evaluation error during
    probing — a domain error, a NaN, a shape surprise — also yields ``None``:
    we never let a probe failure turn into a wrong solver choice.
    """
    rng = np.random.default_rng(seed)
    grad = _grad_fn(fun, jac)
    try:
        probes = _probe_points(x0, lb, ub, rng)
        P, c, d = _objective_model(fun, grad, hess, probes)

        # Validate the fitted objective model at the held-out probes.
        for p in probes[1:]:
            quad = 0.5 * float(p @ P @ p) if P is not None else 0.0
            model = quad + float(c @ p) + d
            fv = float(fun(p))
            if abs(model - fv) > rtol * (1.0 + abs(fv)):
                raise _NotConvex("objective does not match its linear/quadratic model")

        # Convexity: a quadratic must have a positive-semidefinite Hessian.
        if P is not None:
            eig = np.linalg.eigvalsh(P)
            if float(eig.min()) < -1e-8 * max(1.0, abs(float(eig.max()))):
                raise _NotConvex("indefinite Hessian — nonconvex QP")

        A, b, G, h = _linear_constraints(g_combined, jac_combined, cl, cu, probes, m)
    except _NotConvex:
        return None
    except Exception:
        # Probing blew up (domain error, NaN, bad shape) — stay on the NLP path.
        return None

    lb_c, ub_c = _clean_bounds(lb, ub)
    return QpExtract(
        kind="lp" if P is None else "convex_qp",
        P=P,
        c=np.asarray(c, dtype=np.float64).ravel(),
        obj_const=float(d),
        A=A,
        b=b,
        G=G,
        h=h,
        lb=lb_c,
        ub=ub_c,
    )
