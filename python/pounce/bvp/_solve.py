"""SciPy-signature boundary value problem solver on top of pounce.

:func:`solve_bvp` matches the call signature and return shape of
:func:`scipy.integrate.solve_bvp`, but solves the Hermite--Simpson
collocation system (see :mod:`pounce.bvp._core`) as a **pounce feasibility
NLP** — ``min 0`` subject to the square collocation residual
``R(z) = 0`` — rather than SciPy's bespoke damped-Newton iteration.

Differences from SciPy worth knowing:

* **Mesh.** ``adaptive=True`` (default, like SciPy) refines the mesh to meet
  ``tol`` and reproduces SciPy's mesh sequence essentially node-for-node.
  ``adaptive=False`` solves the given mesh as-is — fast and predictable, and
  the mode the differentiable ``pounce.jax`` / ``pounce.torch`` layers use
  internally (a fixed mesh keeps ``theta -> y`` smooth).
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

from .._pounce import Problem, SparseLU
from .._result import ResultMixin
from . import _core
from ._jac import CollocationJacobian

# Result status codes. 0/1/2/3 follow SciPy's solve_bvp (converged / max
# nodes / singular Jacobian / bc_tol unmet); 4 and 5 are pounce-specific so
# they can't be mistaken for SciPy's meanings.
_TERMINATION_MESSAGES = {
    0: "The algorithm converged to the desired tolerance.",
    1: "The maximum number of mesh nodes is exceeded.",
    2: "A singular Jacobian was encountered.",
    3: "The boundary condition residuals do not satisfy bc_tol.",
    4: "The Newton iteration did not converge.",
    5: "The interior-point solver reached only its acceptable tolerance.",
}


def _ipm_status(ipm_code):
    """Map a pounce IPM return code to a BVP result status.

    The IPM's code 1 is ``SolvedToAcceptableLevel`` — the *looser*
    ``acceptable_tol``, which is not a full-accuracy solve, so it is surfaced
    as status 5 (``success=False``) rather than silently reported as a clean
    convergence.
    """
    if ipm_code == 0:
        return 0
    if ipm_code == 1:
        return 5
    return 4


# --- verbose reporting (mirrors SciPy's solve_bvp output) ------------------

def _print_progress_header():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}".format(
        "Iteration", "Max residual", "Max BC residual", "Total nodes",
        "Nodes added"))


def _print_progress(iteration, residual, bc_residual, total_nodes, nodes_added):
    print("{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}".format(
        iteration, residual, bc_residual, total_nodes, str(nodes_added)))


def _print_termination(status, niter, n_nodes, max_rms, max_bc):
    print(f"{_TERMINATION_MESSAGES.get(status, 'Unknown status.')}")
    print(f"Number of iterations {niter}, number of nodes {n_nodes}.")
    print(f"Maximum relative residual: {max_rms:.2e}")
    print(f"Maximum boundary residual: {max_bc:.2e}")


@dataclass
class BVPResult(ResultMixin):
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


def ift_solve_transpose(nfun, nbc, x_np, n, m, k, z_star, v):
    """Implicit-function-theorem back-solve, shared by the JAX/Torch VJPs.

    Solves ``R_zᵀ u = v`` at the converged ``z*`` by factoring the sparse
    collocation Jacobian with FERAL's LU. Both differentiable frontends route
    their (framework-native) ``custom_vjp`` / ``autograd.Function`` backward
    through this one host-side numpy routine, so the linear-algebra half of
    the IFT can't drift between them.

    ``v`` may be a single cotangent ``(N,)`` or a batch ``(B, N)`` (e.g.
    ``jax.jacobian``); the batch case uses one factorization and a multi-RHS
    ``solve_transpose_many``. The factor is taken at ``z*`` (not reused from
    the forward Newton, whose frozen-Jacobian factor is generally at an
    earlier iterate) so the sensitivity is exact.
    """
    v = np.asarray(v, dtype=np.float64)
    z_star = np.asarray(z_star, dtype=np.float64)
    N = n * m + k
    jac = CollocationJacobian(nfun, nbc, x_np, n, m, k)
    rows, cols = jac.structure()
    lu = SparseLU(N, np.asarray(rows, np.int64), np.asarray(cols, np.int64))
    Y = z_star[: n * m].reshape(n, m)
    pp = z_star[n * m :]
    lu.factor(jac.values(Y, pp))
    if v.ndim == 2:
        B = v.shape[0]
        return np.asarray(lu.solve_transpose_many(v.reshape(-1), B), np.float64).reshape(B, N)
    return np.asarray(lu.solve_transpose(v), np.float64)


def _to_numpy(a):
    """Concrete NumPy view of a NumPy / JAX / detached-Torch array."""
    try:
        return np.asarray(a, dtype=np.float64)
    except (TypeError, RuntimeError):  # torch tensor requiring grad
        return np.asarray(a.detach().cpu().numpy(), dtype=np.float64)


def _make_spline(x, y, yp):
    """Lazily-built cubic-Hermite interpolant ``sol(xq) -> (n, ...)`` from
    node states ``y`` ``(n, m)`` and node derivatives ``yp`` ``(n, m)``.

    Shared by the NumPy, JAX, and Torch frontends. Construction is deferred
    to the first call so callers that only read ``res.y`` / ``res.p`` don't
    pay for the spline build, and so a solve inside a JAX trace (where ``y``
    is a tracer) doesn't eagerly force concrete values.
    """
    cache = {}

    def sol(xq):
        if "spline" not in cache:
            from scipy.interpolate import CubicHermiteSpline

            # CubicHermiteSpline interpolates along axis 0; feed (m, n) and
            # transpose the query result back to SciPy's (n, ...) layout.
            # It requires strictly increasing x, so a decreasing mesh (e.g.
            # backward ODE integration) is reversed before the build.
            xn, yn, ypn = _to_numpy(x), _to_numpy(y), _to_numpy(yp)
            if xn.size > 1 and xn[0] > xn[-1]:
                xn, yn, ypn = xn[::-1], yn[:, ::-1], ypn[:, ::-1]
            cache["spline"] = CubicHermiteSpline(xn, yn.T, ypn.T)
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
    adaptive=True,
    args=None,
):
    """Solve a boundary value problem with pounce (SciPy-compatible).

    Drop-in for :func:`scipy.integrate.solve_bvp`. ``fun(x, y[, p])``
    returns the ``(n, m)`` RHS over the mesh; ``bc(ya, yb[, p])`` returns
    the ``n + k`` boundary residuals. See the module docstring for the
    (small) behavioural differences from SciPy.

    ``adaptive=True`` (default, like SciPy) refines the mesh: solve, estimate
    the relative RMS residual of the continuous solution between collocation
    points, insert nodes where it exceeds ``tol``, and repeat up to
    ``max_nodes``. ``adaptive=False`` solves once on the mesh ``x`` as given —
    fast and predictable, and the mode the differentiable ``pounce.jax`` /
    ``pounce.torch`` paths use internally (a parameter-dependent mesh would
    make the solution nonsmooth).

    ``verbose`` mirrors SciPy: ``1`` prints a one-line termination report,
    ``2`` additionally prints per-iteration mesh-refinement progress.

    ``method`` selects the forward solver:

    * ``"newton"`` (default) — damped Newton on the square collocation
      system, factoring the ``N x N`` Jacobian with FERAL's unsymmetric
      sparse LU. This is the fast path (matches SciPy's algorithm and
      beats it on speed); it bypasses the interior-point method.
    * ``"ipm"`` — pose the collocation system as a pounce feasibility NLP
      (``min 0`` s.t. ``R(z) = 0``) and solve with the interior-point
      method. Slower (it factors the ``2N`` saddle KKT each iteration),
      kept as a reference and for the constrained extension.

    ``bc_tol`` (default ``tol``) is the absolute tolerance on the boundary
    residuals: an otherwise-converged solve whose ``max|bc|`` exceeds it is
    reported as ``status=3`` (matching SciPy), ``success=False``.

    ``args`` (tuple) supplies extra fixed parameters for parameterized runs,
    appended to the solver's positional arguments — ``fun(x, y[, p], *args)``,
    ``bc(ya, yb[, p], *args)``, and likewise ``fun_jac`` / ``bc_jac`` — so a
    sweep over a fixed parameter needs no closure. (A pounce convenience;
    ``scipy.integrate.solve_bvp`` has no ``args``.)

    Returns
    -------
    BVPResult
        SciPy-compatible result bunch. ``status`` codes: ``0`` converged;
        ``1`` max mesh nodes exceeded (adaptive); ``2`` singular Jacobian;
        ``3`` boundary residuals exceed ``bc_tol``; ``4`` Newton did not
        converge; ``5`` (``method="ipm"``) the IPM reached only its
        *acceptable* tolerance, not full accuracy. ``success`` is
        ``status == 0``.
    """
    if S is not None:
        raise NotImplementedError(
            "pounce.bvp.solve_bvp does not yet support the singular term S."
        )

    # Extra fixed parameters for parameterized runs: appended after the
    # solver's positional arguments, so `fun(x, y[, p], *args)` etc.
    if args is not None:
        if not isinstance(args, tuple):
            args = (args,)
        fun = (lambda _f: (lambda *a: _f(*a, *args)))(fun)
        bc = (lambda _b: (lambda *a: _b(*a, *args)))(bc)
        if fun_jac is not None:
            fun_jac = (lambda _fj: (lambda *a: _fj(*a, *args)))(fun_jac)
        if bc_jac is not None:
            bc_jac = (lambda _bj: (lambda *a: _bj(*a, *args)))(bc_jac)

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
        z_star, niter, status, _rn = newton_solve(
            residual_fn, jac, z0, n, m, k, tol=newton_tol,
        )  # status: 0 converged / 2 singular / 4 not converged
        info = {}
    elif method == "ipm":
        obj = _BvpNlp(residual_fn, jac, n, m, k)
        cl = np.zeros(N, dtype=np.float64)
        cu = np.zeros(N, dtype=np.float64)
        problem = Problem(n=N, m=N, problem_obj=obj, cl=cl, cu=cu)
        problem.add_option("tol", float(tol))
        # Collocation residuals are naturally well-scaled; skip the scaling
        # pass (its setup cost buys nothing here).
        problem.add_option("nlp_scaling_method", "none")
        # `verbose` drives our SciPy-style report, not the IPM's internal log.
        problem.add_option("print_level", 0)
        z_star, info = problem.solve(x0=z0)
        niter = int(info.get("iter_count", 0))
        status = _ipm_status(info.get("status", -1))
    else:
        raise ValueError(f"unknown method {method!r}; use 'newton' or 'ipm'.")

    z_star = np.asarray(z_star, dtype=np.float64)
    Y, p_star = _core.unpack_z(z_star, n, m)
    Y = np.array(Y)
    yp = np.asarray(nfun(x, Y, p_star), dtype=np.float64)

    # Per-interval RMS of the collocation residual (matches SciPy's
    # `rms_residuals`, which is the collocation estimate, not the boundary
    # residual — the boundary conditions are checked separately below).
    r_star = residual_fn(z_star)
    col = r_star[: n * (m - 1)].reshape(n, m - 1)
    rms_residuals = np.sqrt(np.mean(col**2, axis=0))

    max_bc = float(np.max(np.abs(np.asarray(nbc(Y[:, 0], Y[:, -1], p_star)))))
    # Boundary-condition tolerance (SciPy status 3): only downgrade an
    # otherwise-converged solve.
    if status == 0 and max_bc > (float(tol) if bc_tol is None else float(bc_tol)):
        status = 3

    message = _TERMINATION_MESSAGES.get(status, "unknown status")
    success = status == 0

    if verbose >= 2:  # single fixed-mesh solve = one mesh iteration
        _print_progress_header()
        _print_progress(niter, float(rms_residuals.max()), max_bc, m, 0)
    if verbose >= 1:
        _print_termination(status, niter, m, float(rms_residuals.max()), max_bc)

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
    nfun, nbc = _core._make_normalized(fun, bc, theta=None, uses_p=uses_p)

    if verbose >= 2:
        _print_progress_header()

    res = None
    for iteration in range(1, max_rounds + 1):
        # Inner solve is silent (verbose=0); this loop owns the reporting.
        res = solve_bvp(
            fun, bc, x, y, p=(p_cur if uses_p else None),
            fun_jac=fun_jac, bc_jac=bc_jac, tol=tol, max_nodes=max_nodes,
            verbose=0, method=method, adaptive=False,
        )
        p_eval = res.p if uses_p else np.zeros(0)
        rms = _estimate_rms_residuals(nfun, res.x, res.y, res.yp, p_eval)
        res.rms_residuals = rms
        done = (not res.success) or rms.max() < tol or res.x.size >= max_nodes
        if not done:
            x_new = _refine_mesh(res.x, rms, tol, max_nodes)
            nodes_added = x_new.size - res.x.size
        else:
            nodes_added = 0
        if verbose >= 2:
            max_bc = float(np.max(np.abs(np.asarray(
                nbc(res.y[:, 0], res.y[:, -1], p_eval)))))
            _print_progress(iteration, float(rms.max()), max_bc,
                            res.x.size, nodes_added)
        if done:
            break
        if x_new.size == res.x.size:
            break  # node budget exhausted; return best effort
        y = res.sol(x_new)
        x = x_new
        if uses_p:
            p_cur = res.p
    if res is not None:
        # SciPy reports the mesh-refinement iteration count, not the inner
        # Newton count.
        res.niter = iteration
    if res is not None and res.success and res.rms_residuals.max() >= tol:
        # The inner solve converged but the mesh couldn't be refined enough
        # to bring the estimated residual under tol (SciPy status 1).
        res.status = 1
        res.success = False
        res.message = _TERMINATION_MESSAGES[1]
    if verbose >= 1 and res is not None:
        p_eval = res.p if uses_p else np.zeros(0)
        max_bc = float(np.max(np.abs(np.asarray(
            nbc(res.y[:, 0], res.y[:, -1], p_eval)))))
        _print_termination(res.status, res.niter, res.x.size,
                           float(res.rms_residuals.max()), max_bc)
    return res
