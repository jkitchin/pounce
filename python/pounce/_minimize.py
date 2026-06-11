"""scipy.optimize.minimize-style facade over pounce.Problem.

Thin wrapper that adapts SciPy conventions (functional ``fun``, ``jac``,
``hess``; bound list of ``(lo, hi)`` pairs; constraint dicts with
``'type': 'eq'|'ineq'``) into a cyipopt/pounce-style Problem.

Notes
-----
* When ``jac`` is omitted (or ``False``) we fall back to **central** finite
  differences (step ``eps**(1/3)``) and emit a one-time ``UserWarning`` naming
  the remedies. Central differences have an ``O(h^2)`` truncation error whose
  noise floor sits well below the tight default tolerance, so the solve
  converges cleanly instead of stalling just short of it (gh #123).
  Production callers should still provide an analytic Jacobian (or use the
  autodiff frontends ``pounce.jax`` / ``pounce.torch``).
* When ``jac=True``, ``fun(x, *args)`` must return ``(f, grad)``; the pair is
  cached so each Ipopt iterate triggers only one forward pass.
* When ``hess`` is omitted, or when constraints are present, the solver
  is driven with ``hessian_approximation = limited-memory``.
* Equality / inequality dicts are concatenated into a single ``g(x)``
  with bound vectors ``cl`` / ``cu``. Constraint Jacobian is dense by
  design — sparse Jacobians belong on the :class:`Problem` API.
* ``callback`` accepts both scipy signatures (chosen by parameter-name
  introspection): ``callback(intermediate_result=OptimizeResult)`` or
  ``callback(xk)``. Raise ``StopIteration`` to terminate early.
  ``intermediate_result.x`` is read from a cache populated by the
  objective evaluation that precedes Ipopt's intermediate hook.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult

from ._pounce import Problem
from ._route import classify_and_extract, classify_and_extract_socp

# Central-difference step. The optimal step for a central difference is
# ``~eps**(1/3)`` (≈6.06e-6), balancing the ``O(h^2)`` truncation error against
# the ``O(eps/h)`` round-off error. Its noise floor (~1e-10) sits well below the
# tight default ``tol=1e-8``, so the gradient is accurate enough for the IPM to
# drive the dual infeasibility under tolerance instead of plateauing on FD noise
# and tripping the tiny-step exit at the true optimum (gh #119 / #123).
_CDIFF_STEP = float(np.finfo(np.float64).eps) ** (1.0 / 3.0)

# Ipopt's default ``acceptable_tol``: the NLP error below which a stalled solve
# is still considered converged "to an acceptable level". Used by the E-path
# success heuristic when the exit status is not itself a success code.
_DEFAULT_ACCEPTABLE_TOL = 1e-6

# Convex-solver status string → scipy-style integer status (0 == success),
# matching the NLP path's convention.
_QP_STATUS_CODE = {
    "optimal": 0,
    "primal_infeasible": 2,
    "dual_infeasible": 3,
    "iteration_limit": 1,
    "numerical_failure": 4,
}

# NLP ``ApplicationReturnStatus`` codes that count as a successful solve for the
# scipy-style ``success`` flag. ``SolveSucceeded`` (0) is the obvious one;
# ``SolvedToAcceptableLevel`` (1) means the iterate met the *acceptable*
# tolerance after the tight tolerance stalled — Ipopt/cyipopt and scipy both
# treat that as a success, and pounce's own differentiable path already does
# (``jax/_path.py`` / ``torch/_path.py`` ``_OK_STATUS``). Excluding it (gh #119)
# made HS071 and similar problems report ``success=False`` at a verified
# optimum. Codes 2..6 (infeasible, tiny step, diverging, …) stay failures.
_NLP_SUCCESS_STATUS = frozenset({0, 1})


# Mapping of scipy-canonical option names to their Ipopt equivalents. Multiple
# scipy tolerance options (``gtol`` / ``ftol`` / ``xtol``) all collapse onto
# Ipopt's single ``tol`` knob — last-write-wins if the caller sets more than
# one. Used by :func:`_translate_option`.
_SCIPY_TO_IPOPT_OPTION_NAMES = {
    "maxiter": "max_iter",
    "gtol": "tol",
    "ftol": "tol",
    "xtol": "tol",
    "iprint": "print_level",
    "maxcor": "limited_memory_max_history",
}


def _translate_option(k: str, v: Any) -> tuple[str, Any]:
    """Translate a scipy-canonical ``(name, value)`` pair to its Ipopt form.

    Handles both name aliases (``maxiter`` → ``max_iter``) and value coercions
    where the types differ between scipy and Ipopt (``disp`` is bool in scipy
    but Ipopt's ``print_level`` is an int 0–12).
    """
    if k == "disp":
        # scipy bool/int → Ipopt print_level (0 quiet, 5 standard).
        if isinstance(v, bool):
            return "print_level", 5 if v else 0
        return "print_level", int(v)
    if k == "iprint":
        return "print_level", int(v)
    if k == "maxcor":
        return "limited_memory_max_history", int(v)
    return _SCIPY_TO_IPOPT_OPTION_NAMES.get(k, k), v


def _to_array(x, dtype=np.float64) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _finite_diff_grad(fun: Callable, x: np.ndarray) -> np.ndarray:
    """Central-difference gradient of a scalar ``fun`` at ``x``.

    Central (two-sided) differences have an ``O(h^2)`` truncation error, two
    orders better than a one-sided difference, at the cost of a second function
    evaluation per coordinate. See ``_CDIFF_STEP`` for why this matters for the
    tight-tolerance solve (gh #123).
    """
    g = np.empty_like(x)
    for i in range(x.size):
        h = _CDIFF_STEP * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        g[i] = (float(fun(xp)) - float(fun(xm))) / (2.0 * h)
    return g


class _FunAndGradCache:
    """Memoize ``(f, g) = fun(x, *args)`` on the most recent ``x``.

    Ipopt evaluates ``objective`` and ``gradient`` as separate calls
    (often at the same point). When ``jac=True``, the user-supplied
    ``fun`` returns both in one forward pass — caching here preserves
    that single-pass guarantee across the two Ipopt callbacks.
    """

    def __init__(self, fun: Callable, args: tuple):
        self._fun = fun
        self._args = args
        self._x: np.ndarray | None = None
        self._f: float | None = None
        self._g: np.ndarray | None = None

    def _ensure(self, x: np.ndarray) -> None:
        if (
            self._x is None
            or self._x.shape != x.shape
            or not np.array_equal(self._x, x)
        ):
            f, g = self._fun(x, *self._args)
            self._x = x.copy()
            self._f = float(f)
            self._g = _to_array(g).ravel()

    def f(self, x: np.ndarray) -> float:
        self._ensure(x)
        assert self._f is not None
        return self._f

    def g(self, x: np.ndarray) -> np.ndarray:
        self._ensure(x)
        assert self._g is not None
        return self._g


class _LastXCache:
    """Stash the most recent ``x`` seen by ``objective`` for the callback shim.

    Pounce's Rust ``intermediate`` hook doesn't pass the primal iterate to
    python. Ipopt evaluates the objective at the accepted iterate just
    before firing ``intermediate``, so the latest cached ``x`` is the right
    point to surface as ``OptimizeResult.x`` to a scipy-style callback.
    """

    def __init__(self) -> None:
        self.x: np.ndarray | None = None

    def remember(self, x: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float64).copy()


def _wrap_callback(callback: Callable | None) -> Callable | None:
    """Normalize a scipy-style callback to ``(OptimizeResult) -> bool``.

    Returned shim returns ``True`` to continue, ``False`` to stop. Raising
    ``StopIteration`` inside the user callback also stops the solve.
    Mirrors ``scipy.optimize._minimize._wrap_callback`` introspection: a
    single parameter literally named ``intermediate_result`` selects the
    new-style signature; anything else gets the old-style ``xk``.
    """
    if callback is None:
        return None
    try:
        sig = inspect.signature(callback)
        use_new = set(sig.parameters) == {"intermediate_result"}
    except (TypeError, ValueError):
        use_new = False

    def wrapped(res: OptimizeResult) -> bool:
        try:
            if use_new:
                callback(intermediate_result=res)
            else:
                callback(np.copy(res.x))
        except StopIteration:
            return False
        return True

    return wrapped


def _finite_diff_jac(g_fun: Callable, x: np.ndarray, m: int) -> np.ndarray:
    """Central-difference Jacobian of a vector ``g_fun`` at ``x``."""
    J = np.empty((m, x.size))
    for i in range(x.size):
        h = _CDIFF_STEP * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        xm = x.copy()
        xm[i] -= h
        J[:, i] = (_to_array(g_fun(xp)) - _to_array(g_fun(xm))) / (2.0 * h)
    return J


def _validate_bounds_length(bounds, n: int) -> None:
    """Reject a per-variable ``bounds`` sequence that doesn't have exactly ``n``
    entries (one ``(lo, hi)`` pair per variable).

    Without this, a too-short list silently leaves trailing variables unbounded
    — and in the sampling-based searches it can *broadcast* one variable's box
    across several — while a too-long list trips a cryptic ``IndexError`` deep in
    the solve setup. scipy validates bounds length; so do we, up front.
    """
    if bounds is None:
        return
    try:
        length = len(bounds)
    except TypeError:
        raise ValueError(
            "bounds must be a sequence of (lo, hi) pairs, one per variable"
        ) from None
    if length != n:
        raise ValueError(
            f"bounds has {length} entr{'y' if length == 1 else 'ies'} but the "
            f"problem has {n} variable(s); pass one (lo, hi) pair per variable"
        )


def _normalize_bounds(bounds, n: int):
    """Accept ``None``, a list of ``(lo, hi)`` pairs, or a ``scipy.optimize.Bounds``.

    ``scipy.optimize.Bounds.keep_feasible`` is silently ignored — Ipopt's
    barrier method keeps the iterate strictly inside the box for the entire
    solve, which is at least as strong as ``keep_feasible=True``.
    """
    if bounds is None:
        return None, None
    if isinstance(bounds, Bounds):
        lb = np.broadcast_to(np.asarray(bounds.lb, dtype=np.float64), (n,)).copy()
        ub = np.broadcast_to(np.asarray(bounds.ub, dtype=np.float64), (n,)).copy()
        return lb, ub
    # Legacy: iterable of (lo, hi) pairs, one per dimension.
    _validate_bounds_length(bounds, n)
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    for i, bd in enumerate(bounds):
        if bd is None:
            continue
        lo, hi = bd
        if lo is not None:
            lb[i] = lo
        if hi is not None:
            ub[i] = hi
    bad = np.where(lb > ub)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"bounds[{i}] is reversed: lower {lb[i]} > upper {ub[i]}; "
            f"each bound must be (low, high) with low <= high"
        )
    return lb, ub


@dataclass
class _ConstraintBlock:
    """One contiguous block of constraint rows in the unified representation.

    Linear blocks (from :class:`scipy.optimize.LinearConstraint`) carry their
    sparse COO triplet directly in ``constant_vals``; the Jacobian value
    callback simply slices these into its output. Legacy dict blocks fall back
    to a fully-dense per-row pattern with ``constant_vals=None``; their
    ``fun`` / ``jac`` get evaluated at solve time.
    """

    rows: np.ndarray  # 0-indexed within block, length nnz
    cols: np.ndarray  # absolute column indices into x, length nnz
    constant_vals: np.ndarray | None  # None → dynamic (dict); else COO values
    fun: Callable | None  # dict: returns the constraint value vector
    jac: Callable | None  # dict: optional explicit jacobian
    args: tuple
    lb: np.ndarray  # length n_rows
    ub: np.ndarray  # length n_rows
    n_rows: int


def _empty_constraints():
    return 0, None, None, None, None, None, None


def _block_from_linear_constraint(lc: LinearConstraint, n: int) -> _ConstraintBlock:
    A = lc.A
    if not sparse.issparse(A):
        A = sparse.coo_array(np.atleast_2d(np.asarray(A, dtype=np.float64)))
    elif not isinstance(A, sparse.coo_array):
        # csr/csc/etc. — coalesce to COO so we can read row/col directly.
        A = A.tocoo()
    if A.shape[1] != n:
        raise ValueError(f"LinearConstraint.A has {A.shape[1]} columns; expected {n}")
    m_rows = int(A.shape[0])
    lb = np.broadcast_to(np.asarray(lc.lb, dtype=np.float64), (m_rows,)).copy()
    ub = np.broadcast_to(np.asarray(lc.ub, dtype=np.float64), (m_rows,)).copy()
    return _ConstraintBlock(
        rows=np.asarray(A.row, dtype=np.int64),
        cols=np.asarray(A.col, dtype=np.int64),
        constant_vals=np.asarray(A.data, dtype=np.float64),
        fun=None,
        jac=None,
        args=(),
        lb=lb,
        ub=ub,
        n_rows=m_rows,
    )


def _block_from_dict(c: dict, n: int) -> _ConstraintBlock:
    # Mirror the validation main added to the dict path: clear ValueErrors
    # instead of bare KeyError / cryptic TypeError surfacing later.
    if "type" not in c or "fun" not in c:
        missing = sorted({"type", "fun"} - set(c))
        raise ValueError(
            f"constraint dict is missing required key(s) {missing}; a "
            f"constraint needs {{'type': 'eq'|'ineq', 'fun': callable}}"
        )
    kind = c["type"]
    if kind not in ("eq", "ineq"):
        raise ValueError(f"unknown constraint type {kind!r}; use 'eq' or 'ineq'")
    fun = c["fun"]
    if not callable(fun):
        raise ValueError("constraint 'fun' must be callable")
    ca = tuple(c.get("args", ()))
    probe = np.zeros(n)
    m_rows = int(_to_array(fun(probe, *ca)).size)
    # Dense sparsity pattern: every row may touch every column.
    rows = np.repeat(np.arange(m_rows, dtype=np.int64), n)
    cols = np.tile(np.arange(n, dtype=np.int64), m_rows)
    if kind == "eq":
        lb = np.zeros(m_rows)
        ub = np.zeros(m_rows)
    else:
        lb = np.zeros(m_rows)
        ub = np.full(m_rows, np.inf)
    return _ConstraintBlock(
        rows=rows,
        cols=cols,
        constant_vals=None,
        fun=fun,
        jac=c.get("jac"),
        args=ca,
        lb=lb,
        ub=ub,
        n_rows=m_rows,
    )


def _wrap_constraints(constraints, n: int):
    """Build a unified Ipopt-shaped constraint representation.

    Accepts heterogeneous input:
      - ``None`` or empty sequence: no constraints
      - a single :class:`scipy.optimize.LinearConstraint` (dense or sparse ``A``)
      - a single legacy dict ``{"type": "eq"|"ineq", "fun": ..., "jac": ..., "args": ...}``
      - a list mixing both forms

    Returns ``(m_total, g_combined, jac_values, cl, cu, jac_rows, jac_cols)``.
    ``(jac_rows, jac_cols)`` declare Ipopt's ``jacobianstructure``; ``jac_values(x)``
    produces values in matching order. LinearConstraint blocks contribute their
    constant COO triplet; dict blocks fall back to a fully-dense per-row pattern
    and evaluate jac on demand.
    """
    if constraints is None:
        return _empty_constraints()
    if isinstance(constraints, (dict, LinearConstraint)):
        constraints = [constraints]
    elif not constraints:
        return _empty_constraints()

    blocks: list[_ConstraintBlock] = []
    for c in constraints:
        if isinstance(c, LinearConstraint):
            blocks.append(_block_from_linear_constraint(c, n))
        elif isinstance(c, dict):
            blocks.append(_block_from_dict(c, n))
        else:
            raise ValueError(
                f"each constraint must be a dict with 'type' and 'fun', a "
                f"scipy.optimize.LinearConstraint, or a list mixing those — "
                f"got {type(c).__name__}"
            )

    if not blocks:
        return _empty_constraints()

    row_offset = 0
    nnz_start = 0
    row_parts, col_parts, lb_parts, ub_parts = [], [], [], []
    block_nnz_spans: list[tuple[int, int]] = []
    for blk in blocks:
        row_parts.append(blk.rows + row_offset)
        col_parts.append(blk.cols)
        lb_parts.append(blk.lb)
        ub_parts.append(blk.ub)
        nnz_end = nnz_start + int(blk.rows.size)
        block_nnz_spans.append((nnz_start, nnz_end))
        nnz_start = nnz_end
        row_offset += blk.n_rows

    m_total = row_offset
    nnz_total = nnz_start
    jac_rows = np.concatenate(row_parts)
    jac_cols = np.concatenate(col_parts)
    cl = np.concatenate(lb_parts)
    cu = np.concatenate(ub_parts)

    def g_combined(x):
        x = np.asarray(x, dtype=np.float64)
        parts = []
        for blk in blocks:
            if blk.constant_vals is not None:
                A_blk = sparse.coo_array(
                    (blk.constant_vals, (blk.rows, blk.cols)),
                    shape=(blk.n_rows, x.size),
                )
                parts.append(np.asarray(A_blk @ x).ravel())
            else:
                parts.append(_to_array(blk.fun(x, *blk.args)).ravel())
        return np.concatenate(parts)

    def jac_values(x):
        out = np.empty(nnz_total)
        for (start, end), blk in zip(block_nnz_spans, blocks):
            if blk.constant_vals is not None:
                out[start:end] = blk.constant_vals
            elif blk.jac is not None:
                J = np.atleast_2d(_to_array(blk.jac(x, *blk.args)))
                out[start:end] = J.ravel()
            else:
                J = _finite_diff_jac(
                    lambda xx, fn=blk.fun, ca=blk.args: fn(xx, *ca),
                    x,
                    blk.n_rows,
                )
                out[start:end] = J.ravel()
        return out

    return m_total, g_combined, jac_values, cl, cu, jac_rows, jac_cols


def _build_problem_obj(
    *,
    fun: Callable,
    n: int,
    m: int,
    args: tuple,
    jac: Callable | bool | None,
    hess: Callable | None,
    g: Callable | None,
    jac_g: Callable | None,
    jac_rows: np.ndarray | None,
    jac_cols: np.ndarray | None,
    callback: Callable | None,
    eval_counters: dict,
):
    """Build a problem-object-with-methods on the fly. Only attaches
    ``hessian`` / ``hessianstructure`` when ``hess`` is provided so
    Problem's ``hasattr`` probe correctly falls back to L-BFGS. Likewise,
    ``intermediate`` is only attached when ``callback`` is provided so the
    no-callback case has zero per-iter Python overhead."""

    members: dict[str, Any] = {}
    xcache = _LastXCache()
    counters = eval_counters  # alias the caller's dict so we can mutate it

    if jac is True:
        cache = _FunAndGradCache(fun, args)

        def objective(self, x, _c=cache, _xc=xcache, _ctr=counters):
            _xc.remember(x)
            _ctr["nfev"] += 1
            return _c.f(x)

        def gradient(self, x, _c=cache, _ctr=counters):
            _ctr["njev"] += 1
            return _c.g(x)
    else:

        def objective(self, x, _xc=xcache, _ctr=counters):
            _xc.remember(x)
            _ctr["nfev"] += 1
            return float(fun(x, *args))

        def gradient(self, x, _ctr=counters):
            _ctr["njev"] += 1
            if jac is None or jac is False:
                return _finite_diff_grad(lambda x: fun(x, *args), x)
            return _to_array(jac(x, *args)).ravel()

    members["objective"] = objective
    members["gradient"] = gradient

    if callback is not None:
        wrapped_cb = _wrap_callback(callback)
        assert wrapped_cb is not None

        def intermediate(
            self,
            *,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials,
            _cb=wrapped_cb,
            _xc=xcache,
            _n=n,
        ):
            x = _xc.x if _xc.x is not None else np.full(_n, np.nan)
            res = OptimizeResult(
                x=x,
                fun=float(obj_value),
                success=False,
                status=0,
                message="intermediate",
                nit=int(iter_count),
                info={
                    "alg_mod": int(alg_mod),
                    "inf_pr": float(inf_pr),
                    "inf_du": float(inf_du),
                    "mu": float(mu),
                    "d_norm": float(d_norm),
                    "regularization_size": float(regularization_size),
                    "alpha_du": float(alpha_du),
                    "alpha_pr": float(alpha_pr),
                    "ls_trials": int(ls_trials),
                },
            )
            return _cb(res)

        members["intermediate"] = intermediate

    if m > 0:
        assert jac_rows is not None and jac_cols is not None

        def constraints(self, x):
            return _to_array(g(x)).ravel()

        def jacobianstructure(self, _r=jac_rows, _c=jac_cols):
            return (_r, _c)

        def jacobian(self, x):
            return _to_array(jac_g(x)).ravel()

        members["constraints"] = constraints
        members["jacobianstructure"] = jacobianstructure
        members["jacobian"] = jacobian

    if hess is not None and m == 0:

        def hessianstructure(self):
            r, c = np.tril_indices(n)
            return (r, c)

        def hessian(self, x, lam, obj_factor):
            H = obj_factor * _to_array(hess(x, *args))
            r, c = np.tril_indices(n)
            return H[r, c]

        members["hessianstructure"] = hessianstructure
        members["hessian"] = hessian

    cls = type("_MinimizeProblem", (object,), members)
    return cls()


def _solve_via_convex(ex, opts: dict) -> OptimizeResult:
    """Adapt a routed convex LP/QP solve back into an :class:`OptimizeResult`.

    The convex solver minimizes ``½xᵀPx + cᵀx`` and never sees the objective's
    degree-0 term, so we add ``ex.obj_const`` back to the reported value (the
    same constant the CLI threads through ``run_convex_qp``). The result shape
    is identical to the NLP path so the router is transparent to callers.
    """
    from .qp import solve_qp

    res = solve_qp(
        P=ex.P, c=ex.c, A=ex.A, b=ex.b, G=ex.G, h=ex.h, lb=ex.lb, ub=ex.ub,
        tol=opts.get("tol"), max_iter=opts.get("max_iter"),
    )
    fun_val = float(res.obj) + ex.obj_const
    success = res.status == "optimal"
    selector = "lp-ipm" if ex.kind == "lp" else "qp-ipm"
    return OptimizeResult(
        x=np.asarray(res.x),
        fun=fun_val,
        success=success,
        status=_QP_STATUS_CODE.get(res.status, 1),
        message=res.status,
        nit=int(res.iters),
        info={
            "solver": selector,
            "problem_class": ex.kind,
            "obj_val": fun_val,
            "obj_constant": ex.obj_const,
            "status": res.status,
            "status_msg": res.status,
            "iter_count": int(res.iters),
            "residuals": res.residuals,
        },
    )


def _solve_via_socp(ex, opts: dict) -> OptimizeResult:
    """Adapt a routed convex-QCQP solve (reformulated to a SOCP) back into an
    :class:`OptimizeResult`.

    Mirrors :func:`_solve_via_convex`: the conic solver minimizes
    ``½xᵀPx + cᵀx`` over the cone constraints and never sees the objective's
    degree-0 term, so ``ex.obj_const`` is added back to the reported value (the
    same constant the CLI threads through ``run_convex_socp``). The result shape
    matches the NLP path so the router stays transparent to callers.
    """
    from .qp import solve_socp

    res = solve_socp(
        P=ex.P, c=ex.c, A=ex.A, b=ex.b, G=ex.G, h=ex.h, cones=ex.cones,
        tol=opts.get("tol"), max_iter=opts.get("max_iter"),
    )
    fun_val = float(res.obj) + ex.obj_const
    success = res.status == "optimal"
    return OptimizeResult(
        x=np.asarray(res.x),
        fun=fun_val,
        success=success,
        status=_QP_STATUS_CODE.get(res.status, 1),
        message=res.status,
        nit=int(res.iters),
        info={
            "solver": "socp",
            "problem_class": ex.kind,
            "obj_val": fun_val,
            "obj_constant": ex.obj_const,
            "status": res.status,
            "status_msg": res.status,
            "iter_count": int(res.iters),
            "residuals": res.residuals,
        },
    )


def _any_constraint_without_jac(constraints) -> bool:
    """True if any scipy-style constraint dict omits ``'jac'`` (so its Jacobian
    is finite-differenced). Used to decide whether to warn (gh #123, D)."""
    if not constraints:
        return False
    if isinstance(constraints, (dict, LinearConstraint)):
        constraints = [constraints]
    return any(
        isinstance(c, dict) and c.get("jac") is None for c in constraints
    )


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | bool | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | LinearConstraint | dict | None = None,
    callback: Callable | None = None,
    **options: Any,
) -> OptimizeResult:
    """scipy.optimize.minimize-style facade over pounce.

    The signature mirrors scipy's ``_custom`` callable-method contract: ``hessp``
    and unknown options arrive via ``**options`` and are filtered out before
    being forwarded to Ipopt. This makes ``pounce.minimize`` a drop-in target
    for ``scipy.optimize.minimize(method=pounce.minimize, ...)``.

    Solver routing mirrors the CLI's ``solver_selection`` but is **opt-in**:
    the default is the NLP backend, with no structure probing overhead. Pass
    ``solver_selection="auto"`` (or one of the explicit selectors) to enable
    routing. A linear or convex-quadratic objective with only linear constraints
    can be dispatched to the specialized convex LP/QP interior-point solver
    (``pounce.solve_qp``), a convex-quadratic objective/constraints problem (a
    convex QCQP) to the conic solver (``pounce.solve_socp``), and everything
    else falls through to the general NLP filter-IPM. Detection is conservative
    and validated against the true callables at held-out points, so a nonlinear
    problem is never silently sent to the convex solver.

    * ``"nlp"`` (default) — always use the NLP solver, skipping the probe;
    * ``"auto"`` — route LP/convex-QP to the convex QP solver, a convex QCQP to
      the conic solver, else NLP;
    * ``"lp-ipm"`` / ``"qp-ipm"`` — force the convex QP solver, raising
      ``ValueError`` if the problem is not detected as an LP / convex QP;
    * ``"socp"`` — force the conic solver, raising ``ValueError`` if the
      problem is not detected as a convex QCQP.

    Like :func:`scipy.optimize.minimize`, this facade is **silent by default**.
    Pass ``disp=True`` for a concise log or an explicit ``print_level=N``
    (0–12) to control the NLP backend's IPM iteration table directly.
    """
    # Accept both calling conventions: scipy-style ``options={...}`` (one dict
    # argument) and the splatted ``**options`` form (kwargs absorbed by the
    # signature, as scipy's ``_custom`` dispatch sends them). Explicit kwargs
    # win over the legacy dict if both are supplied.
    legacy_options = options.pop("options", None)
    if isinstance(legacy_options, Mapping):
        options = {**dict(legacy_options), **options}

    # Promote a scalar / 0-d x0 to 1-D, matching scipy.optimize.minimize, so a
    # single-variable problem can be written ``minimize(f, 1.5)``.
    x0 = np.atleast_1d(_to_array(x0))
    n = x0.size
    lb, ub = _normalize_bounds(bounds, n)
    m, g_combined, jac_combined, cl, cu, jac_rows, jac_cols = _wrap_constraints(
        constraints, n
    )

    # Solver routing (mirrors the CLI's `solver_selection`). Pop routing keys
    # so the remainder of `options` flows to the NLP backend. The default is
    # `"nlp"` (no probe) — opt in to `"auto"` to enable structure detection.
    selection = str(options.pop("solver_selection", "nlp")).lower()
    route_tol = float(options.pop("route_tol", 1e-5))
    # scipy.optimize.minimize is silent unless `disp=True`; match that. pounce's
    # NLP backend otherwise prints a full IPM iteration table by default (and
    # the log is written from Rust to fd 1, so Python stdout redirection can't
    # catch it). Default print_level to 0 (silent) unless the caller passes an
    # explicit print_level or scipy-style disp=True. (#115)
    disp = bool(options.pop("disp", False))
    options.setdefault("print_level", 5 if disp else 0)

    if selection != "nlp" and m > 0:
        # The router's `_linear_constraints` expects a dense `m × n` Jacobian
        # from `jac_combined(x)`. Our `_wrap_constraints` returns a flat
        # nnz-length value vector instead. Materialize a dense view here only
        # when routing is on, so the no-route path pays nothing.
        def _jac_combined_dense(x, _vals=jac_combined, _r=jac_rows, _c=jac_cols, _m=m, _n=n):
            out = np.zeros((_m, _n), dtype=np.float64)
            out[_r, _c] = _vals(x)
            return out
        router_jac_combined = _jac_combined_dense
    else:
        router_jac_combined = jac_combined  # value-vec; never read when nlp
    route_kw = dict(
        fun=fun, jac=jac, hess=hess, lb=lb, ub=ub, m=m,
        g_combined=g_combined, jac_combined=router_jac_combined,
        cl=cl, cu=cu, x0=x0, rtol=route_tol,
    )
    if selection in ("auto", "lp-ipm", "qp-ipm"):
        extract = classify_and_extract(**route_kw)
        if selection == "lp-ipm" and (extract is None or extract.kind != "lp"):
            raise ValueError(
                "solver_selection='lp-ipm' but the problem was not detected as "
                "a linear program (linear objective + linear constraints)"
            )
        if selection == "qp-ipm" and extract is None:
            raise ValueError(
                "solver_selection='qp-ipm' but the problem was not detected as "
                "a convex LP/QP (convex-quadratic objective + linear constraints)"
            )
        if extract is not None:
            return _solve_via_convex(extract, options)
        # Auto: an LP/QP wasn't found — try a convex QCQP before giving up to
        # the NLP solver (a quadratic *constraint* lands here, not above).
        if selection == "auto":
            socp = classify_and_extract_socp(**route_kw)
            if socp is not None:
                return _solve_via_socp(socp, options)
    elif selection == "socp":
        socp = classify_and_extract_socp(**route_kw)
        if socp is None:
            raise ValueError(
                "solver_selection='socp' but the problem was not detected as a "
                "convex QCQP (convex-quadratic objective and/or constraints, all "
                "convex, with only linear equalities)"
            )
        return _solve_via_socp(socp, options)

    eval_counters: dict[str, int] = {"nfev": 0, "njev": 0}

    # (D, gh #123) The NLP path finite-differences any derivative the caller
    # did not supply. FD derivatives are slower and less accurate, and on a
    # tight solve can stall just short of the tolerance and report
    # ``success=False`` at the true optimum. Warn once — naming the remedies —
    # rather than degrading silently. (scipy.optimize.minimize is silent here;
    # pounce deliberately is not, because this is the #1 source of confusing
    # "failed at the right answer" reports.)
    fd_targets = []
    if jac is None:
        fd_targets.append("the objective gradient (pass jac=...)")
    if m > 0 and _any_constraint_without_jac(constraints):
        fd_targets.append("constraint Jacobian(s) (pass 'jac' in each "
                          "constraint dict)")
    if fd_targets:
        warnings.warn(
            "pounce.minimize is approximating " + " and ".join(fd_targets)
            + " by finite differences. This is slower and less accurate than "
            "analytic derivatives. For a faster, more robust solve supply them "
            "directly, or use the autodiff frontends pounce.jax / pounce.torch.",
            stacklevel=2,
        )

    problem_obj = _build_problem_obj(
        fun=fun,
        n=n,
        m=m,
        args=args,
        jac=jac,
        hess=hess,
        g=g_combined,
        jac_g=jac_combined,
        jac_rows=jac_rows,
        jac_cols=jac_cols,
        callback=callback,
        eval_counters=eval_counters,
    )

    problem = Problem(
        n=n,
        m=m,
        problem_obj=problem_obj,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    # ``options`` was already drained of routing keys (`solver_selection`,
    # `route_tol`) and ``disp`` above, so only genuine solver options reach
    # the NLP backend. Drop None-valued kwargs (scipy's ``_custom`` dispatch
    # always sends ``hessp=None``, ``bounds=None``, etc.; absorbed here when
    # undeclared on the signature). Real Ipopt option misses still surface as
    # ``RuntimeError`` from ``problem.solve()`` — by design.
    for k, v in options.items():
        if v is None:
            continue
        ipopt_k, ipopt_v = _translate_option(k, v)
        problem.add_option(ipopt_k, ipopt_v)

    x, info = problem.solve(x0=x0)
    # (E, gh #119 / #123) Judge success on the final KKT error, not the exit
    # status enum alone. Ipopt-family solvers report a non-success status (e.g.
    # ``Search_Direction_Becomes_Too_Small``, code 3) when progress stalls — but
    # a stall at a point whose overall NLP error is already at the acceptable
    # tolerance is a converged solve, not a failure. cyipopt/scipy treat such a
    # point as a success; so do we. ``final_kkt_error`` is the unscaled overall
    # NLP error at the final iterate (exposed from the Rust SolveStatistics); it
    # is NaN on paths that never computed it, which ``np.isfinite`` filters out.
    status_code = int(info["status"])
    acceptable_tol = float(options.get("acceptable_tol", _DEFAULT_ACCEPTABLE_TOL))
    kkt_error = float(info.get("final_kkt_error", float("nan")))
    success = status_code in _NLP_SUCCESS_STATUS or (
        np.isfinite(kkt_error) and kkt_error <= acceptable_tol
    )
    return OptimizeResult(
        x=np.asarray(x),
        fun=float(info["obj_val"]),
        success=success,
        status=status_code,
        message=str(info["status_msg"]),
        nit=int(info["iter_count"]),
        nfev=int(eval_counters["nfev"]),
        njev=int(eval_counters["njev"]),
        info=dict(info),
    )
