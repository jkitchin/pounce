"""SciPy-signature boundary value problem solver on top of pounce.

:func:`solve_bvp` matches the call signature and return shape of
:func:`scipy.integrate.solve_bvp`, but solves the Hermite--Simpson
collocation system (see :mod:`pounce.bvp._core`) as a **pounce feasibility
NLP** — ``min 0`` subject to the square collocation residual
``R(z) = 0`` — rather than SciPy's bespoke damped-Newton iteration.

Differences from SciPy worth knowing:

* **Mesh.** ``adaptive=False`` (default) solves on the mesh ``x`` as given —
  fast and predictable, and what the differentiable ``pounce.jax`` /
  ``pounce.torch`` layers rely on (a fixed mesh keeps ``theta -> y`` smooth).
  ``adaptive=True`` enables SciPy-style residual-driven refinement and
  reproduces SciPy's mesh sequence essentially node-for-node.
* **Solver (``method``).** ``"newton"`` (default) factors the exact sparse
  ``N x N`` collocation Jacobian with FERAL's unsymmetric LU — SciPy's
  algorithm, typically faster. ``"ipm"`` solves the collocation feasibility
  NLP with the interior-point method.
* **Singular term ``S``** is not yet supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .._pounce import Problem
from . import _core
from ._jac import CollocationJacobian


@dataclass
class BVPResult:
    """Result of :func:`solve_bvp`, mirroring SciPy's ``Bunch``.

    Attributes match :func:`scipy.integrate.solve_bvp` so existing code can
    consume the result unchanged: ``sol`` (callable cubic-Hermite
    interpolant returning shape ``(n, ...)``), ``x`` (mesh), ``y`` (states
    at the mesh, ``(n, m)``), ``yp`` (derivatives at the mesh), ``p``
    (converged unknown parameters or ``None``), ``rms_residuals``,
    ``niter``, ``status``, ``message``, ``success``.
    """

    sol: Callable
    p: Any
    x: np.ndarray
    y: np.ndarray
    yp: np.ndarray
    rms_residuals: np.ndarray
    niter: int
    status: int
    message: str
    success: bool
    info: dict = field(default_factory=dict, repr=False)


def _make_spline(x, y, yp):
    """Lazily-built cubic-Hermite interpolant ``sol(xq) -> (n, ...)`` from
    node states ``y`` ``(n, m)`` and node derivatives ``yp`` ``(n, m)``.

    Construction is deferred to the first call so callers that only read
    ``res.y`` / ``res.p`` don't pay for the spline build.
    """
    cache = {}

    def sol(xq):
        if "spline" not in cache:
            from scipy.interpolate import CubicHermiteSpline

            # CubicHermiteSpline interpolates along axis 0; feed (m, n) and
            # transpose the query result back to SciPy's (n, ...) layout.
            cache["spline"] = CubicHermiteSpline(x, y.T, yp.T)
        return cache["spline"](xq).T

    return sol


def _make_jac_adapters(fun_jac, bc_jac, uses_p, k):
    """Adapt SciPy-style ``fun_jac`` / ``bc_jac`` to the per-node block
    callables :class:`CollocationJacobian` expects, or return ``None`` to
    select the finite-difference path.

    ``fun_jac(x, y[, p])`` returns ``df_dy`` ``(n, n, mq)`` (and ``df_dp``
    ``(n, k, mq)`` when ``p`` is present); we transpose to the ``(mq, n,
    *)`` block layout. ``bc_jac(ya, yb[, p])`` returns
    ``(dbc_dya, dbc_dyb[, dbc_dp])``.
    """
    df_blocks = None
    if fun_jac is not None:
        def df_blocks(xq, Yq, p, fq):
            if uses_p:
                out = fun_jac(xq, Yq, p)
                df_dy, df_dp = out if isinstance(out, tuple) else (out, None)
            else:
                df_dy, df_dp = fun_jac(xq, Yq), None
            J = np.transpose(np.asarray(df_dy, dtype=np.float64), (2, 0, 1))
            if k > 0:
                K = np.transpose(np.asarray(df_dp, dtype=np.float64), (2, 0, 1))
            else:
                K = None
            return J, K

    dbc_block = None
    if bc_jac is not None:
        def dbc_block(ya, yb, p, b0):
            if uses_p:
                dya, dyb, dp = bc_jac(ya, yb, p)
            else:
                dya, dyb = bc_jac(ya, yb)
                dp = np.zeros((ya.shape[0], 0), dtype=np.float64)
            return (np.asarray(dya, dtype=np.float64),
                    np.asarray(dyb, dtype=np.float64),
                    np.asarray(dp, dtype=np.float64))

    return df_blocks, dbc_block


class _BvpNlp:
    """Cyipopt-shaped feasibility problem: ``min 0`` s.t. ``R(z) = 0``.

    The objective and its gradient are identically zero, so the
    interior-point method reduces to a Newton iteration on the square
    collocation residual. The constraint Jacobian is the exact **sparse**
    collocation Jacobian (:class:`CollocationJacobian`).

    The Lagrangian Hessian is supplied as **exactly zero**. This is not an
    approximation: for ``min 0`` s.t. ``R(z) = 0`` with a square,
    nonsingular ``J = dR/dz``, the KKT stationarity ``Jᵀλ = 0`` forces the
    optimal multipliers ``λ* = 0``, so the Lagrangian Hessian
    ``Σ_t λ_t ∇²R_t`` vanishes at the solution. With the zero-Hessian
    block the IPM step solves ``[[0, Jᵀ],[J, 0]] [dz; dλ] = [-Jᵀλ; -R]``,
    whose primal part is ``dz = -J⁻¹ R`` — precisely the Newton step on the
    collocation system that SciPy's ``solve_bvp`` takes (SciPy likewise
    uses only the residual Jacobian, never second derivatives of ``f``).
    Supplying it directly avoids the limited-memory quasi-Newton machinery
    and converges in one Newton step on linear problems.
    """

    def __init__(self, residual_fn, jac, n, m, k):
        self._r = residual_fn
        self._jac = jac
        self._n = n
        self._m = m
        self._N = n * m + k
        self._empty = np.zeros(0, dtype=np.float64)
        self._empty_idx = np.zeros(0, dtype=np.int64)

    def objective(self, z):
        return 0.0

    def gradient(self, z):
        return np.zeros(self._N, dtype=np.float64)

    def constraints(self, z):
        return np.asarray(self._r(z), dtype=np.float64)

    def jacobianstructure(self):
        return self._jac.structure()

    def jacobian(self, z):
        z = np.asarray(z, dtype=np.float64)
        Y = z[: self._n * self._m].reshape(self._n, self._m)
        p = z[self._n * self._m :]
        return self._jac.values(Y, p)

    def hessianstructure(self):
        return (self._empty_idx, self._empty_idx)

    def hessian(self, z, lagrange, obj_factor):
        return self._empty


def solve_bvp(
    fun,
    bc,
    x,
    y,
    p=None,
    S=None,
    fun_jac=None,
    bc_jac=None,
    tol=1e-3,
    max_nodes=1000,
    verbose=0,
    bc_tol=None,
    method="newton",
    adaptive=False,
):
    """Solve a boundary value problem on a fixed mesh with pounce.

    Drop-in for :func:`scipy.integrate.solve_bvp`. ``fun(x, y[, p])``
    returns the ``(n, m)`` RHS over the mesh; ``bc(ya, yb[, p])`` returns
    the ``n + k`` boundary residuals. See the module docstring for the
    (small) behavioural differences from SciPy.

    ``adaptive=False`` (default) solves on the mesh ``x`` as given — fast and
    predictable. ``adaptive=True`` turns on SciPy-style mesh refinement: it
    re-solves, estimates the relative RMS residual of the continuous solution
    between collocation points, inserts nodes where it exceeds ``tol``, and
    repeats up to ``max_nodes``. (The differentiable ``pounce.jax`` /
    ``pounce.torch`` paths are always fixed-mesh — a parameter-dependent mesh
    would make the solution nonsmooth.)

    ``method`` selects the forward solver:

    * ``"newton"`` (default) — damped Newton on the square collocation
      system, factoring the ``N x N`` Jacobian with FERAL's unsymmetric
      sparse LU. This is the fast path (matches SciPy's algorithm and
      beats it on speed); it bypasses the interior-point method.
    * ``"ipm"`` — pose the collocation system as a pounce feasibility NLP
      (``min 0`` s.t. ``R(z) = 0``) and solve with the interior-point
      method. Slower (it factors the ``2N`` saddle KKT each iteration),
      kept as a reference and for the constrained extension.

    Returns
    -------
    BVPResult
        SciPy-compatible result bunch.
    """
    if S is not None:
        raise NotImplementedError(
            "pounce.bvp.solve_bvp does not yet support the singular term S."
        )

    if adaptive:
        return _solve_bvp_adaptive(
            fun, bc, x, y, p=p, fun_jac=fun_jac, bc_jac=bc_jac,
            tol=tol, max_nodes=max_nodes, verbose=verbose, method=method,
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("`y` must be 2-D with shape (n, m).")
    n, m = y.shape
    if x.shape != (m,):
        raise ValueError(f"`x` must have shape ({m},) to match `y`.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("`x` must be strictly increasing.")

    uses_p = p is not None
    p0 = np.asarray(p, dtype=np.float64).ravel() if uses_p else np.zeros(0)
    k = p0.shape[0]

    nfun, nbc = _core._make_normalized(fun, bc, theta=None, uses_p=uses_p)

    # Sanity-check the boundary-residual count: collocation contributes
    # n*(m-1) equations, so bc must supply the remaining n + k.
    bc0 = np.asarray(nbc(y[:, 0], y[:, -1], p0), dtype=np.float64)
    if bc0.shape != (n + k,):
        raise ValueError(
            f"`bc` must return {n + k} residuals (n + k); got {bc0.shape}."
        )

    N = _core.num_unknowns(n, m, k)

    def residual_fn(z):
        return _core.residual_of_z(z, nfun, nbc, x, n, m, k, np.concatenate)

    df_blocks, dbc_block = _make_jac_adapters(fun_jac, bc_jac, uses_p, k)
    jac = CollocationJacobian(
        nfun, nbc, x, n, m, k, df_blocks=df_blocks, dbc_block=dbc_block,
    )
    z0 = _core.pack_z(y, p0, np.concatenate)

    if method == "newton":
        from ._newton import newton_solve

        # The collocation system is solved to (near) round-off, NOT to the
        # mesh ``tol``: the residual estimate that drives adaptive refinement
        # divides the collocation residual by the interval width, so a
        # loosely-solved system reads as a huge residual on fine meshes.
        # ``tol`` controls refinement, not the Newton stop.
        newton_tol = min(float(tol), 1e-10)
        z_star, niter, converged, _rn = newton_solve(
            residual_fn, jac, z0, n, m, k, tol=newton_tol,
        )
        info = {}
        status = 0 if converged else 1
        message = "converged" if converged else "Newton did not converge"
    elif method == "ipm":
        obj = _BvpNlp(residual_fn, jac, n, m, k)
        cl = np.zeros(N, dtype=np.float64)
        cu = np.zeros(N, dtype=np.float64)
        problem = Problem(n=N, m=N, problem_obj=obj, cl=cl, cu=cu)
        problem.add_option("tol", float(tol))
        # Collocation residuals are naturally well-scaled; skip the scaling
        # pass (its setup cost buys nothing here).
        problem.add_option("nlp_scaling_method", "none")
        problem.add_option("print_level", 5 if verbose >= 2 else 0)
        z_star, info = problem.solve(x0=z0)
        niter = int(info.get("iter_count", 0))
        status = 0 if info.get("status", 1) in (0, 1) else 1
        message = info.get("status_msg", "")
    else:
        raise ValueError(f"unknown method {method!r}; use 'newton' or 'ipm'.")

    z_star = np.asarray(z_star, dtype=np.float64)
    Y, p_star = _core.unpack_z(z_star, n, m)
    Y = np.array(Y)
    yp = np.asarray(nfun(x, Y, p_star), dtype=np.float64)

    # Per-interval RMS of the collocation residual (state-major block).
    r_star = residual_fn(z_star)
    col = r_star[: n * (m - 1)].reshape(n, m - 1)
    rms_residuals = np.sqrt(np.mean(col**2, axis=0))

    success = status == 0
    sol = _make_spline(x, Y, yp)

    return BVPResult(
        sol=sol,
        p=(p_star.copy() if uses_p else None),
        x=x,
        y=Y,
        yp=yp,
        rms_residuals=rms_residuals,
        niter=int(niter),
        status=status,
        message=message,
        success=success,
        info=info,
    )


# --------------------------------------------------------------------------
# Adaptive mesh refinement (opt-in, SciPy-style)
# --------------------------------------------------------------------------

def _estimate_rms_residuals(nfun, x, Y, yp, p):
    """Relative RMS collocation residual per interval (SciPy's estimator).

    Faithful port of ``scipy.integrate._bvp.estimate_rms_residuals``: build
    the C1 cubic-Hermite spline from node states ``Y`` and node derivatives
    ``yp = f(x, Y)``, then estimate, on each interval, the relative residual
    ``r = sol'(s) - f(s, sol(s))`` (normalised by ``1 + |f|``) with a
    **5-point Lobatto quadrature**. Crucially the residual is sampled at the
    *superconvergent* Gauss points ``x_mid ± 0.5 h √(3/7)`` and the interval
    midpoint (where ``r_mid = 1.5 · col_res / h``); at those points the
    cubic's residual reflects the true 4th-order solution error rather than
    its own interpolation error, which is what makes the refinement converge
    at scipy's rate instead of over-refining.
    """
    from scipy.interpolate import CubicHermiteSpline

    h = x[1:] - x[:-1]                                  # (m-1,)
    f = np.asarray(yp, dtype=np.float64)                # f(x, Y) at nodes, (n, m)
    x_mid = x[:-1] + 0.5 * h
    y_mid = 0.5 * (Y[:, 1:] + Y[:, :-1]) - 0.125 * h * (f[:, 1:] - f[:, :-1])
    f_mid = np.asarray(nfun(x_mid, y_mid, p), dtype=np.float64)   # (n, m-1)
    col_res = (Y[:, 1:] - Y[:, :-1]) - (h / 6.0) * (f[:, :-1] + 4 * f_mid + f[:, 1:])
    r_mid = 1.5 * col_res / h                           # midpoint residual (≠0)

    spline = CubicHermiteSpline(x, Y.T, f.T)
    dspline = spline.derivative()
    s = 0.5 * h * (3 / 7) ** 0.5                        # Gauss offset
    x1 = x_mid + s
    x2 = x_mid - s
    f1 = np.asarray(nfun(x1, spline(x1).T, p), dtype=np.float64)
    f2 = np.asarray(nfun(x2, spline(x2).T, p), dtype=np.float64)
    r1 = dspline(x1).T - f1
    r2 = dspline(x2).T - f2

    r_mid = r_mid / (1.0 + np.abs(f_mid))
    r1 = r1 / (1.0 + np.abs(f1))
    r2 = r2 / (1.0 + np.abs(f2))
    r_mid = np.sum(r_mid**2, axis=0)
    r1 = np.sum(r1**2, axis=0)
    r2 = np.sum(r2**2, axis=0)
    return (0.5 * (32 / 45 * r_mid + 49 / 90 * (r1 + r2))) ** 0.5


def _refine_mesh(x, rms, tol, max_nodes):
    """Insert nodes in intervals whose residual exceeds ``tol``.

    One node (the midpoint) where ``tol < rms <= 100*tol``; two (the
    thirds) where ``rms > 100*tol`` — SciPy's rule. Capped at ``max_nodes``.
    Returns the new mesh (strictly increasing), or the old one if nothing
    can be inserted within the node budget.
    """
    pieces = [x[:1]]
    budget = max_nodes - x.size
    for i in range(x.size - 1):
        a, b = x[i], x[i + 1]
        if rms[i] > tol and budget > 0:
            if rms[i] > 100 * tol and budget >= 2:
                pieces.append(np.array([a + (b - a) / 3, a + 2 * (b - a) / 3]))
                budget -= 2
            else:
                pieces.append(np.array([0.5 * (a + b)]))
                budget -= 1
        pieces.append(np.array([b]))
    return np.concatenate(pieces)


def _solve_bvp_adaptive(
    fun, bc, x, y, p=None, fun_jac=None, bc_jac=None,
    tol=1e-3, max_nodes=1000, verbose=0, method="newton", max_rounds=30,
):
    """SciPy-style refine loop around the fixed-mesh :func:`solve_bvp`.

    Solve → estimate per-interval residual → insert nodes where it exceeds
    ``tol`` → re-solve (warm-started by interpolating the previous solution
    onto the new mesh), until every interval is under ``tol`` or the mesh
    reaches ``max_nodes``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    uses_p = p is not None
    p_cur = (np.asarray(p, dtype=np.float64).ravel() if uses_p else None)

    res = None
    for _ in range(max_rounds):
        res = solve_bvp(
            fun, bc, x, y, p=(p_cur if uses_p else None),
            fun_jac=fun_jac, bc_jac=bc_jac, tol=tol, max_nodes=max_nodes,
            verbose=verbose, method=method, adaptive=False,
        )
        nfun, _ = _core._make_normalized(
            fun, bc, theta=None, uses_p=uses_p,
        )
        p_eval = res.p if uses_p else np.zeros(0)
        rms = _estimate_rms_residuals(nfun, res.x, res.y, res.yp, p_eval)
        res.rms_residuals = rms
        if not res.success or rms.max() < tol or res.x.size >= max_nodes:
            break
        x_new = _refine_mesh(res.x, rms, tol, max_nodes)
        if x_new.size == res.x.size:
            break  # node budget exhausted; return best effort
        y = res.sol(x_new)
        x = x_new
        if uses_p:
            p_cur = res.p
    if res is not None and res.rms_residuals.max() >= tol:
        res.status = max(res.status, 1)
        res.success = res.status == 0
        if res.message in ("converged", ""):
            res.message = "adaptive refinement hit max_nodes before reaching tol"
    return res
