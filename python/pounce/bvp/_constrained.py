"""Constrained boundary value problems — the pounce-unique capability.

:func:`solve_bvp_constrained` solves a collocation BVP **subject to bounds
on the states / parameters and inequality path constraints**, optionally
minimising an objective:

    dy/dx = f(x, y, p),         a <= x <= b
    bc(y(a), y(b), p) = 0
    ylo <= y(x) <= yhi          (state bounds, at every node)
    clo <= c(x, y, p) <= chi    (inequality path constraints, at every node)
    minimise  J(Y, p)           (optional)

This is a genuine nonlinear program — not a square root-find — so it is
solved with pounce's interior-point method, not the Newton path. SciPy's
``solve_bvp`` cannot express bounds, path constraints, or an objective;
this is where routing a collocation discretisation through pounce pays off.

The collocation equality block (residual + boundary conditions) reuses the
exact sparse Jacobian from :mod:`pounce.bvp._jac`; the inequality path block
is assembled here (block-diagonal per mesh node). The Lagrangian Hessian is
left to pounce's limited-memory quasi-Newton approximation, since with
nonlinear path constraints it is no longer identically zero.
"""

from __future__ import annotations

import numpy as np

from .._pounce import Problem
from . import _core
from ._jac import CollocationJacobian, _FD_EPS
from ._solve import BVPResult, _make_spline, _ipm_status, _TERMINATION_MESSAGES


def _broadcast_bounds(b, length):
    """Broadcast a bound spec to a flat length-``length`` vector."""
    if b is None:
        return None
    return np.broadcast_to(np.asarray(b, dtype=np.float64), (length,)).copy()


def _state_bounds_flat(bound, n, m):
    """Per-state bound ``(n,)`` or ``(n, m)`` -> flat state-major ``(n*m,)``."""
    if bound is None:
        return None
    arr = np.asarray(bound, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full((n, m), float(arr))
    elif arr.shape == (n,):
        arr = np.broadcast_to(arr[:, None], (n, m))
    elif arr.shape == (n, m):
        pass
    else:
        raise ValueError(f"state bound must be scalar, (n,), or (n, m); got {arr.shape}")
    return np.array(arr).reshape(-1)


class _PathJacobian:
    """Sparse Jacobian of the node-wise path constraints ``c(x, Y, p)``.

    ``c`` has shape ``(q, m)``; constraint row ``j*m + i`` (component ``j``
    at node ``i``) couples only ``y_i`` (cols ``b*m + i``) and ``p`` (cols
    ``n*m + l``). Values come from a vectorised finite difference of the
    path function (``O(n)`` evaluations).
    """

    def __init__(self, npath, x, n, m, k, q, row_offset):
        self._npath = npath
        self._x = x
        self._n, self._m, self._k, self._q = n, m, k, q
        self._row_offset = row_offset
        self._build_structure()

    def _build_structure(self):
        n, m, k, q = self._n, self._m, self._k, self._q
        off = self._row_offset
        rows, cols = [], []
        # d c_j(node i) / d y_b  ->  row off + j*m + i, col b*m + i
        J, I, B = np.meshgrid(np.arange(q), np.arange(m), np.arange(n), indexing="ij")
        rows.append((off + J * m + I).ravel())
        cols.append((B * m + I).ravel())
        if k > 0:
            Jp, Ip, L = np.meshgrid(
                np.arange(q), np.arange(m), np.arange(k), indexing="ij"
            )
            rows.append((off + Jp * m + Ip).ravel())
            cols.append((n * m + L).ravel())
        self._rows = np.concatenate(rows).astype(np.int64)
        self._cols = np.concatenate(cols).astype(np.int64)

    def structure(self):
        return self._rows, self._cols

    def values(self, Y, p):
        n, m, k, q = self._n, self._m, self._k, self._q
        x = self._x
        c0 = np.asarray(self._npath(x, Y, p), dtype=np.float64)  # (q, m)
        # d c / d y_b at every node: (q, m) per perturbed state b.
        dcdy = np.empty((q, m, n), dtype=np.float64)
        for b in range(n):
            step = _FD_EPS * np.maximum(1.0, np.abs(Y[b]))
            Yb = Y.copy()
            Yb[b] += step
            cb = np.asarray(self._npath(x, Yb, p), dtype=np.float64)
            dcdy[:, :, b] = (cb - c0) / step
        # Structure order is (j, i, b): for each component j, node i, state b.
        vals = [dcdy.reshape(-1)]
        if k > 0:
            dcdp = np.empty((q, m, k), dtype=np.float64)
            for l in range(k):
                step = _FD_EPS * max(1.0, abs(p[l]))
                pl = p.copy()
                pl[l] += step
                cl = np.asarray(self._npath(x, Y, pl), dtype=np.float64)
                dcdp[:, :, l] = (cl - c0) / step
            vals.append(dcdp.reshape(-1))
        return np.concatenate(vals)


class _ConstrainedBvpNlp:
    """Cyipopt-shaped constrained collocation NLP.

    Variables ``z = [vec(Y); p]``. Equality constraints: collocation
    residual + boundary conditions (the ``N``-row block, exact sparse
    Jacobian). Inequality constraints: node-wise path constraints
    (``q*m`` rows). Objective: ``J(Y, p)`` if supplied, else ``0``
    (feasibility); its gradient is finite-differenced.
    """

    def __init__(self, residual_fn, eq_jac, path_jac, npath, objective, n, m, k, q):
        self._r = residual_fn
        self._eq_jac = eq_jac
        self._path_jac = path_jac
        self._npath = npath
        self._objective = objective
        self._n, self._m, self._k, self._q = n, m, k, q
        self._N = n * m + k
        er, ec = eq_jac.structure()
        if path_jac is not None:
            pr, pc = path_jac.structure()
            self._rows = np.concatenate([er, pr])
            self._cols = np.concatenate([ec, pc])
        else:
            self._rows, self._cols = er, ec

    def _unpack(self, z):
        z = np.asarray(z, dtype=np.float64)
        return z[: self._n * self._m].reshape(self._n, self._m), z[self._n * self._m :]

    def objective(self, z):
        if self._objective is None:
            return 0.0
        Y, p = self._unpack(z)
        return float(self._objective(Y, p))

    def gradient(self, z):
        if self._objective is None:
            return np.zeros(self._N, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        g = np.empty(self._N, dtype=np.float64)
        f0 = self.objective(z)
        for j in range(self._N):
            step = _FD_EPS * max(1.0, abs(z[j]))
            zj = z.copy()
            zj[j] += step
            g[j] = (self.objective(zj) - f0) / step
        return g

    def constraints(self, z):
        eq = np.asarray(self._r(z), dtype=np.float64)
        if self._path_jac is None:
            return eq
        Y, p = self._unpack(z)
        c = np.asarray(self._npath(self._path_jac._x, Y, p), dtype=np.float64)
        return np.concatenate([eq, c.reshape(-1)])

    def jacobianstructure(self):
        return (self._rows, self._cols)

    def jacobian(self, z):
        Y, p = self._unpack(z)
        ev = self._eq_jac.values(Y, p)
        if self._path_jac is None:
            return ev
        return np.concatenate([ev, self._path_jac.values(Y, p)])


def solve_bvp_constrained(
    fun,
    bc,
    x,
    y,
    p=None,
    *,
    y_bounds=None,
    p_bounds=None,
    path=None,
    path_bounds=None,
    objective=None,
    tol=1e-8,
    max_iter=3000,
    verbose=0,
    args=None,
):
    """Solve a bounded / path-constrained BVP with pounce's IPM.

    Parameters
    ----------
    fun, bc, x, y, p
        As in :func:`pounce.bvp.solve_bvp` (``bc`` returns ``n + k``
        residuals).
    y_bounds : (lower, upper) or None
        State bounds, each scalar, ``(n,)`` (per state) or ``(n, m)``
        (per state and node). ``None`` entries / sides default to
        ``±inf``. Enforced at every mesh node.
    p_bounds : (lower, upper) or None
        Bounds on the unknown parameters, each ``(k,)``.
    path : callable or None
        Inequality path constraint ``path(x, Y, p) -> (q, m)`` evaluated
        over the mesh (SciPy-style vectorised). Enforced at every node.
    path_bounds : (clo, chi)
        Bounds for the path constraints, each ``(q,)`` (broadcast over
        nodes). Required when ``path`` is given.
    objective : callable or None
        Optional objective ``objective(Y, p) -> float`` to minimise
        (gradient is finite-differenced). ``None`` solves a feasibility
        problem (any solution satisfying the ODE, BCs, bounds, and path
        constraints).

    args : tuple or None
        Extra fixed parameters appended to ``fun`` / ``bc`` / ``path``
        (``fun(x, y[, p], *args)`` …) for parameterized runs.

    Returns
    -------
    BVPResult
    """
    if args is not None:
        if not isinstance(args, tuple):
            args = (args,)
        fun = (lambda _f: (lambda *a: _f(*a, *args)))(fun)
        bc = (lambda _b: (lambda *a: _b(*a, *args)))(bc)
        if path is not None:
            path = (lambda _p: (lambda *a: _p(*a, *args)))(path)

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
    N = _core.num_unknowns(n, m, k)

    nfun, nbc = _core._make_normalized(fun, bc, theta=None, uses_p=uses_p)
    bc0 = np.asarray(nbc(y[:, 0], y[:, -1], p0), dtype=np.float64)
    if bc0.ndim != 1:
        raise ValueError(f"`bc` must return a 1-D residual; got shape {bc0.shape}.")
    bc_len = bc0.shape[0]
    if bc_len > n + k:
        raise ValueError(
            f"`bc` returns {bc_len} residuals but the system has only n + k = "
            f"{n + k} degrees of freedom; an over-determined BVP is infeasible."
        )
    # Number of equality rows (collocation + boundary). When bc_len < n + k
    # the system is under-determined — the remaining freedom is resolved by
    # the objective / constraints (an optimal-control collocation).
    n_eq = n * (m - 1) + bc_len

    def residual_fn(z):
        return _core.residual_of_z(z, nfun, nbc, x, n, m, k, np.concatenate)

    eq_jac = CollocationJacobian(nfun, nbc, x, n, m, k, bc_len=bc_len)

    # Path constraints.
    npath = None
    path_jac = None
    q = 0
    if path is not None:
        if path_bounds is None:
            raise ValueError("`path_bounds` is required when `path` is given.")
        if uses_p:
            npath = lambda xx, YY, pp: path(xx, YY, pp)
        else:
            npath = lambda xx, YY, pp: path(xx, YY)
        c0 = np.asarray(npath(x, y, p0), dtype=np.float64)
        if c0.ndim != 2 or c0.shape[1] != m:
            raise ValueError(f"`path` must return shape (q, {m}); got {c0.shape}.")
        q = c0.shape[0]
        path_jac = _PathJacobian(npath, x, n, m, k, q, row_offset=n_eq)

    obj = _ConstrainedBvpNlp(residual_fn, eq_jac, path_jac, npath, objective, n, m, k, q)

    # Variable bounds.
    lb = np.full(N, -2e19)
    ub = np.full(N, 2e19)
    if y_bounds is not None:
        ylo, yhi = y_bounds
        lo = _state_bounds_flat(ylo, n, m)
        hi = _state_bounds_flat(yhi, n, m)
        if lo is not None:
            lb[: n * m] = lo
        if hi is not None:
            ub[: n * m] = hi
    if uses_p and p_bounds is not None:
        plo, phi = p_bounds
        if plo is not None:
            lb[n * m :] = np.broadcast_to(np.asarray(plo, np.float64), (k,))
        if phi is not None:
            ub[n * m :] = np.broadcast_to(np.asarray(phi, np.float64), (k,))

    # Constraint bounds: equality block (n_eq rows) = 0; path block = [clo, chi].
    m_con = n_eq + q * m
    cl = np.zeros(m_con)
    cu = np.zeros(m_con)
    if q > 0:
        clo, chi = path_bounds
        clo = np.broadcast_to(np.asarray(clo, np.float64)[:, None], (q, m)).reshape(-1)
        chi = np.broadcast_to(np.asarray(chi, np.float64)[:, None], (q, m)).reshape(-1)
        cl[n_eq:] = clo
        cu[n_eq:] = chi

    problem = Problem(n=N, m=m_con, problem_obj=obj, lb=lb, ub=ub, cl=cl, cu=cu)
    problem.add_option("tol", float(tol))
    problem.add_option("max_iter", int(max_iter))
    problem.add_option("print_level", 5 if verbose >= 2 else 0)

    z0 = _core.pack_z(y, p0, np.concatenate)
    z_star, info = problem.solve(x0=z0)
    z_star = np.asarray(z_star, dtype=np.float64)

    Y, p_star = _core.unpack_z(z_star, n, m)
    Y = np.array(Y)
    yp = np.asarray(nfun(x, Y, p_star), dtype=np.float64)
    r_star = residual_fn(z_star)
    col = r_star[: n * (m - 1)].reshape(n, m - 1)
    rms_residuals = np.sqrt(np.mean(col**2, axis=0))

    # IPM code 1 ("acceptable level") is the looser tolerance, not a full
    # solve — surface it as status 5 (success=False) rather than reporting
    # it as a clean convergence (constraint violations show up here too,
    # since the IPM status reflects the full KKT, bounds and path rows
    # included).
    status = _ipm_status(info.get("status", -1))
    return BVPResult(
        sol=_make_spline(x, Y, yp),
        p=(p_star.copy() if uses_p else None),
        x=x,
        y=Y,
        yp=yp,
        rms_residuals=rms_residuals,
        niter=int(info.get("iter_count", 0)),
        status=status,
        message=_TERMINATION_MESSAGES.get(status, info.get("status_msg", "")),
        success=status == 0,
        info=info,
    )
