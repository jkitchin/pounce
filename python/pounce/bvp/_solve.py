"""SciPy-signature boundary value problem solver on top of pounce.

:func:`solve_bvp` matches the call signature and return shape of
:func:`scipy.integrate.solve_bvp`, but solves the Hermite--Simpson
collocation system (see :mod:`pounce.bvp._core`) as a **pounce feasibility
NLP** — ``min 0`` subject to the square collocation residual
``R(z) = 0`` — rather than SciPy's bespoke damped-Newton iteration.

Differences from SciPy worth knowing:

* **Fixed mesh.** The mesh ``x`` you pass is used as-is; there is no
  adaptive refinement. This is deliberate: a fixed mesh makes the
  solution map ``theta -> y`` smooth, which is what the differentiable
  ``pounce.jax`` / ``pounce.torch`` layers exploit. Refine by passing a
  denser ``x``. ``max_nodes`` is accepted for signature compatibility.
* **Derivatives.** The collocation Jacobian handed to the interior-point
  solver is formed by forward finite differences of the residual; the
  Hessian uses pounce's limited-memory quasi-Newton approximation. The
  ``fun_jac`` / ``bc_jac`` arguments are accepted for signature
  compatibility (a future revision can assemble the exact sparse
  collocation Jacobian from them).
* **Singular term ``S``** is not yet supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .._pounce import Problem
from . import _core


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
    """Cubic-Hermite interpolant ``sol(xq) -> (n, ...)`` from node states
    ``y`` ``(n, m)`` and node derivatives ``yp`` ``(n, m)``."""
    from scipy.interpolate import CubicHermiteSpline

    # CubicHermiteSpline interpolates along axis 0; feed (m, n) and
    # transpose the query result back to SciPy's (n, ...) convention.
    spline = CubicHermiteSpline(x, y.T, yp.T)

    def sol(xq):
        return spline(xq).T

    return sol


def _numerical_residual_jac(residual_fn, z, r0):
    """Dense forward-difference Jacobian ``dR/dz`` at ``z`` (``r0 = R(z)``)."""
    N = z.shape[0]
    J = np.empty((N, N), dtype=np.float64)
    eps = np.sqrt(np.finfo(np.float64).eps)
    for j in range(N):
        step = eps * max(1.0, abs(z[j]))
        zj = z.copy()
        zj[j] += step
        J[:, j] = (residual_fn(zj) - r0) / step
    return J


class _BvpNlp:
    """Cyipopt-shaped feasibility problem: ``min 0`` s.t. ``R(z) = 0``.

    The objective and its gradient are identically zero, so the
    interior-point method reduces to a Newton iteration on the square
    collocation residual. The Jacobian is dense forward-difference; the
    Hessian is omitted so pounce falls back to its limited-memory
    quasi-Newton approximation.
    """

    def __init__(self, residual_fn, N):
        self._r = residual_fn
        self._N = N
        idx = np.arange(N)
        self._rows = np.repeat(idx, N)
        self._cols = np.tile(idx, N)

    def objective(self, z):
        return 0.0

    def gradient(self, z):
        return np.zeros(self._N, dtype=np.float64)

    def constraints(self, z):
        return np.asarray(self._r(z), dtype=np.float64)

    def jacobianstructure(self):
        return (self._rows, self._cols)

    def jacobian(self, z):
        z = np.asarray(z, dtype=np.float64)
        r0 = np.asarray(self._r(z), dtype=np.float64)
        return _numerical_residual_jac(self._r, z, r0).reshape(-1)


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
):
    """Solve a boundary value problem on a fixed mesh with pounce.

    Drop-in for :func:`scipy.integrate.solve_bvp`. ``fun(x, y[, p])``
    returns the ``(n, m)`` RHS over the mesh; ``bc(ya, yb[, p])`` returns
    the ``n + k`` boundary residuals. See the module docstring for the
    (small) behavioural differences from SciPy.

    Returns
    -------
    BVPResult
        SciPy-compatible result bunch.
    """
    if S is not None:
        raise NotImplementedError(
            "pounce.bvp.solve_bvp does not yet support the singular term S."
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

    obj = _BvpNlp(residual_fn, N)
    cl = np.zeros(N, dtype=np.float64)
    cu = np.zeros(N, dtype=np.float64)
    problem = Problem(n=N, m=N, problem_obj=obj, cl=cl, cu=cu)
    problem.add_option("tol", float(tol))
    problem.add_option("hessian_approximation", "limited-memory")
    problem.add_option("print_level", 5 if verbose >= 2 else 0)

    z0 = _core.pack_z(y, p0, np.concatenate)
    z_star, info = problem.solve(x0=z0)
    z_star = np.asarray(z_star, dtype=np.float64)

    Y, p_star = _core.unpack_z(z_star, n, m)
    Y = np.array(Y)
    yp = np.asarray(nfun(x, Y, p_star), dtype=np.float64)

    # Per-interval RMS of the collocation residual (state-major block).
    r_star = residual_fn(z_star)
    col = r_star[: n * (m - 1)].reshape(n, m - 1)
    rms_residuals = np.sqrt(np.mean(col**2, axis=0))

    status = 0 if info.get("status", 1) in (0, 1) else 1
    success = status == 0
    message = info.get("status_msg", "")
    sol = _make_spline(x, Y, yp)

    return BVPResult(
        sol=sol,
        p=(p_star.copy() if uses_p else None),
        x=x,
        y=Y,
        yp=yp,
        rms_residuals=rms_residuals,
        niter=int(info.get("iter_count", 0)),
        status=status,
        message=message,
        success=success,
        info=info,
    )
