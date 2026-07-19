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
  with bound vectors ``cl`` / ``cu``. A dict constraint whose ``jac`` returns
  a scipy-sparse matrix declares that matrix's (fixed) COO structure, so a
  genuinely sparse constraint Jacobian stays sparse through to the
  :class:`Problem` API (mirroring cyipopt's detect-by-return-type); a dense or
  absent ``jac`` keeps a fully-dense per-row pattern.
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
from scipy.optimize import Bounds, LinearConstraint
from scipy.optimize import OptimizeResult as _ScipyOptimizeResult

from ._pounce import Problem
from ._route import _point_cache, classify_and_extract, classify_and_extract_socp


class OptimizeResult(_ScipyOptimizeResult):
    """SciPy ``OptimizeResult`` with a pounce back-compat shim.

    pounce nests its solver-specific extras (``solver``, ``problem_class``,
    ``status_msg``, ``residuals``, …) under a single ``info`` mapping rather
    than as top-level keys. Before #97 the result was a bespoke dataclass whose
    ``__getitem__`` fell back to ``info`` so ``res["solver"]`` worked; switching
    to scipy's plain-dict ``OptimizeResult`` would silently break that subscript
    (it would ``KeyError``). This subclass restores the fallback: a key absent at
    the top level is looked up in ``info`` before raising. New code should prefer
    ``res.info[...]`` explicitly; this only preserves the old access pattern.
    """

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            info = super().get("info")
            if isinstance(info, Mapping) and key in info:
                return info[key]
            raise


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
    "optimal_inaccurate": 1,
    "primal_infeasible": 2,
    "dual_infeasible": 3,
    "iteration_limit": 1,
    "numerical_failure": 4,
}

# Human-readable message for each convex-solver status. `pounce.minimize`
# always *minimizes*, so for the routed LP/QP the HSDE certificates map to the
# usual primal verdicts: a dual-infeasibility certificate means the objective
# is unbounded below over the feasible region (primal unbounded), and a
# primal-infeasibility certificate means the feasible set is empty. The raw
# certificate string stays available in ``info["status"]`` for callers that
# key on it; this clearer text is surfaced as the result ``message`` so an
# unbounded / infeasible solve is not mistaken for a generic iteration limit
# (gh #160).
_QP_STATUS_MESSAGE = {
    "optimal": "Optimization terminated successfully.",
    # Reduced-accuracy convergence (ECOS/Clarabel ``*_INACC`` analogue): the
    # KKT residual reached the inaccurate band but never the tight ``tol``. We
    # report ``success=False`` (the residual is in ``info["residuals"]`` for a
    # caller that wants to judge for itself), with a message that says so
    # rather than leaking the raw ``optimal_inaccurate`` token (pounce#173).
    "optimal_inaccurate": "Solved to acceptable level (reduced accuracy); the "
    "KKT residual did not reach the requested tolerance.",
    "primal_infeasible": "The problem appears infeasible (the convex solver "
    "returned a primal-infeasibility certificate).",
    "dual_infeasible": "The problem appears unbounded — the objective is "
    "unbounded below over the feasible region (the convex solver returned a "
    "dual-infeasibility certificate).",
    "iteration_limit": "Maximum number of iterations reached.",
    "numerical_failure": "Numerical difficulties encountered.",
}

# NLP ``ApplicationReturnStatus`` codes that count as a successful solve for the
# scipy-style ``success`` flag. ``SolveSucceeded`` (0) is the obvious one;
# ``SolvedToAcceptableLevel`` (1) means the iterate met the *acceptable*
# tolerance after the tight tolerance stalled — Ipopt/cyipopt and scipy both
# treat that as a success, and pounce's own differentiable path already does
# (``jax/_path.py`` / ``torch/_path.py`` ``_OK_STATUS``). Excluding it (gh #119)
# made HS071 and similar problems report ``success=False`` at a verified
# optimum. Codes 2..6 (infeasible, tiny step, diverging, …) stay failures.
# The `solver_selection` values the Rust option registry accepts
# (`upstream_options.rs`, `add_string_option("solver_selection", ...)`). Kept in
# sync by `test_solver_selection_values_match_rust`, which parses that file --
# a hardcoded list here would drift the moment a selector is added.
#
# Behaviour differs by surface, and `pounce.minimize` is a *library* consumer
# (no `.nl` problem-structure extraction), so the split is:
#   auto / lp-ipm / qp-ipm / socp  -- handled here by Python-side extraction
#   nlp                            -- the default; straight to the NLP backend
#   qp-active-set                  -- forwarded to the backend, which selects the
#                                     active-set SQP engine
_SOLVER_SELECTION_VALUES = frozenset(
    {"auto", "nlp", "lp-ipm", "qp-ipm", "qp-active-set", "socp"}
)

_NLP_SUCCESS_STATUS = frozenset({0, 1})

# Statuses for which the KKT-error fallback below must NOT upgrade the solve to
# ``success=True``. ``User_Requested_Stop`` (5) means the solve was aborted by
# the user's ``intermediate`` callback — or, via M32, by a callback that raised
# (which the bridge maps to this same status). That is an external abort, not a
# numerical stall at an acceptable point, so judging it "successful" because the
# last computed KKT error happened to be small is wrong and can mask a crashing
# callback. (L50)
_NO_KKT_FALLBACK_STATUS = frozenset({5})


# Pure-rename map: scipy option name → Ipopt option name with no value
# coercion. Multiple scipy tolerance options (``gtol`` / ``ftol`` / ``xtol``)
# all collapse onto Ipopt's single ``tol`` knob — last-write-wins if the
# caller sets more than one. Options that also need a *value* transform
# (``disp``, ``iprint``, ``maxcor``) are handled by branches in
# :func:`_translate_option` and intentionally do NOT appear here.
_SCIPY_TO_IPOPT_OPTION_NAMES = {
    "maxiter": "max_iter",
    "gtol": "tol",
    "ftol": "tol",
    "xtol": "tol",
}


def _translate_option(k: str, v: Any) -> tuple[str, Any]:
    """Translate a scipy-canonical ``(name, value)`` pair to its Ipopt form.

    Handles both name aliases (``maxiter`` → ``max_iter``) and value coercions
    where the types differ between scipy and Ipopt (``disp`` is bool in scipy
    but Ipopt's ``print_level`` is an int 0–12). Value-coercing entries live
    in the branches below; pure renames live in ``_SCIPY_TO_IPOPT_OPTION_NAMES``.
    """
    if k == "disp":
        # scipy bool/int → Ipopt print_level (0 quiet, 5 standard).
        if isinstance(v, bool):
            return "print_level", 5 if v else 0
        return "print_level", int(v)
    if k == "iprint":
        # scipy may pass float; Ipopt's print_level is strictly int.
        return "print_level", int(v)
    if k == "maxcor":
        # scipy passes integer L-BFGS history; Ipopt option is int-typed,
        # so coerce defensively for callers that send floats.
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
    else:
        # Legacy: iterable of (lo, hi) pairs, one per dimension. Entries may
        # be ``None`` (no bound on that dim) or carry partial-None for a
        # one-sided bound; missing sides stay at ±inf.
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
    # Both paths share the reversed-bound check (the ``Bounds`` path silently
    # produced an infeasible box before).
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
    # Dict block whose ``jac`` returns a *scipy-sparse* matrix: ``(rows, cols)``
    # above are that matrix's COO structure (canonicalized) and ``jac_values``
    # streams the per-iteration COO data instead of a dense ``ravel()``. Lets a
    # nonlinear (or linear) dict constraint declare a sparse Jacobian, mirroring
    # cyipopt's detect-by-return-type behavior.
    sparse_jac: bool = False
    # Pre-assembled ``sparse.coo_array`` for linear blocks. ``g_combined``
    # would otherwise rebuild this from ``(rows, cols, constant_vals)`` on
    # every Ipopt iteration (per-iter waste in the inner loop). ``None`` for
    # dict blocks, which don't have a constant Jacobian.
    sparse_A: Any = None


def _empty_constraints():
    return 0, None, None, None, None, None, None


def _all_linear_constraints(constraints) -> bool:
    """True iff every constraint is an explicit ``scipy.optimize.LinearConstraint``.

    Used to decide whether a user-supplied ``hess`` can be honored even with
    constraints present: for *linear* constraints the constraint-curvature term
    ``Σᵢ λᵢ ∇²gᵢ`` of the Lagrangian Hessian is zero, so the objective Hessian
    *is* the Lagrangian Hessian. We rely solely on the declared type — a dict
    constraint (even if secretly affine) counts as nonlinear; there is no
    probing.
    """
    if isinstance(constraints, LinearConstraint):
        return True
    if constraints is None or isinstance(constraints, dict):
        return False
    try:
        items = list(constraints)
    except TypeError:
        return False
    return len(items) > 0 and all(isinstance(c, LinearConstraint) for c in items)


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
    rows_i = np.asarray(A.row, dtype=np.int64)
    cols_i = np.asarray(A.col, dtype=np.int64)
    vals_i = np.asarray(A.data, dtype=np.float64)
    # Cache the assembled sparse matrix once; ``g_combined`` reuses it
    # every iteration instead of rebuilding from the triplets.
    sparse_A = sparse.coo_array((vals_i, (rows_i, cols_i)), shape=(m_rows, n))
    return _ConstraintBlock(
        rows=rows_i,
        cols=cols_i,
        constant_vals=vals_i,
        fun=None,
        jac=None,
        args=(),
        lb=lb,
        ub=ub,
        n_rows=m_rows,
        sparse_A=sparse_A,
    )


def _block_from_dict(c: dict, n: int, x0=None) -> _ConstraintBlock:
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
    # Probe at the user's x0 when available, not the origin, so a constraint
    # undefined at 0 but defined at a feasible start doesn't fail at build
    # time (L47, mirrors the dict path on main).
    probe = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel()
    m_rows = int(_to_array(fun(probe, *ca)).size)
    jac = c.get("jac")

    # If an explicit jac is supplied and returns a *scipy-sparse* matrix, declare
    # a sparse Jacobian structure from its COO triplet (canonicalized) instead of
    # the dense grid — so nonlinear (and linear) dict constraints can be sparse,
    # mirroring how cyipopt detects sparsity from the jac's return type. Any
    # sparse format is accepted; ``.tocsr().tocoo()`` gives a deterministic
    # row-major order so build- and solve-time orderings align (Ipopt requires
    # a fixed pattern). A dense return (or no jac → finite differences) keeps the
    # original fully-dense pattern.
    sparse_jac = False
    if callable(jac):
        probe_jac = jac(probe, *ca)
        if sparse.issparse(probe_jac):
            Jc = probe_jac.tocsr().tocoo()
            if Jc.shape != (m_rows, n):
                raise ValueError(
                    f"constraint Jacobian has shape {Jc.shape}; expected "
                    f"{(m_rows, n)} (m_rows from 'fun', n from x)"
                )
            rows = np.asarray(Jc.row, dtype=np.int64)
            cols = np.asarray(Jc.col, dtype=np.int64)
            sparse_jac = True
    if not sparse_jac:
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
        jac=jac,
        args=ca,
        lb=lb,
        ub=ub,
        n_rows=m_rows,
        sparse_jac=sparse_jac,
    )


def _wrap_constraints(constraints, n: int, x0=None):
    """Build a unified Ipopt-shaped constraint representation.

    Accepts heterogeneous input:
      - ``None`` or empty sequence: no constraints
      - a single :class:`scipy.optimize.LinearConstraint` (dense or sparse ``A``)
      - a single legacy dict ``{"type": "eq"|"ineq", "fun": ..., "jac": ..., "args": ...}``
      - a list mixing both forms

    Returns ``(m_total, g_combined, jac_values, cl, cu, jac_rows, jac_cols)``.
    ``(jac_rows, jac_cols)`` declare Ipopt's ``jacobianstructure``; ``jac_values(x)``
    produces values in matching order. LinearConstraint blocks contribute their
    constant COO triplet; a dict block whose ``jac`` returns a scipy-sparse matrix
    declares that matrix's COO structure (sparse — for linear or nonlinear
    constraints alike); dict blocks with a dense (or absent) jac fall back to a
    fully-dense per-row pattern and evaluate jac on demand. Dict constraints are
    probed for their output size — and their jac sparsity — at ``x0`` when
    supplied, not the origin (L47).
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
            blocks.append(_block_from_dict(c, n, x0))
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
            if blk.sparse_A is not None:
                # Linear block: reuse the COO matrix assembled once at setup.
                parts.append(np.asarray(blk.sparse_A @ x).ravel())
            else:
                parts.append(_to_array(blk.fun(x, *blk.args)).ravel())
        return np.concatenate(parts)

    def jac_values(x):
        out = np.empty(nnz_total)
        for (start, end), blk in zip(block_nnz_spans, blocks):
            if blk.constant_vals is not None:
                out[start:end] = blk.constant_vals
            elif blk.sparse_jac:
                # Sparse dict Jacobian: stream COO data in the same canonical
                # order declared at build time (``.tocsr().tocoo()``). Verify
                # the *full* structure (row/col positions), not just the nnz
                # count: Ipopt requires a fixed sparsity pattern, and a Jacobian
                # that keeps the same number of nonzeros but moves one to a
                # different position would otherwise misalign its values against
                # the declared structure and be fed to Ipopt as silently-wrong
                # derivatives.
                J = blk.jac(x, *blk.args).tocsr().tocoo()
                if (
                    J.data.size != (end - start)
                    or not np.array_equal(J.row, blk.rows)
                    or not np.array_equal(J.col, blk.cols)
                ):
                    raise ValueError(
                        "constraint Jacobian sparsity pattern changed between "
                        "probe and solve; the pattern (nonzero row/col positions) "
                        "must be fixed across iterations "
                        f"(declared {end - start} nonzeros at the build-time "
                        f"pattern, got {J.data.size})"
                    )
                out[start:end] = J.data
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
    constraints_all_linear: bool = False,
):
    """Build a problem-object-with-methods on the fly. Only attaches
    ``hessian`` / ``hessianstructure`` when ``hess`` is provided AND it can be
    used as the Lagrangian Hessian — i.e. unconstrained (``m == 0``) or all
    constraints linear (zero constraint curvature) — so Problem's ``hasattr``
    probe correctly falls back to L-BFGS otherwise. Likewise, ``intermediate``
    is only attached when ``callback`` is provided so the no-callback case has
    zero per-iter Python overhead."""

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

    # Attach an exact Hessian when ``hess`` is supplied AND it equals the
    # Lagrangian Hessian: unconstrained, or all constraints linear (then the
    # constraint-curvature term ``Σᵢ λᵢ ∇²gᵢ`` is zero, so ``lam`` is unused).
    if hess is not None and (m == 0 or constraints_all_linear):

        def hessianstructure(self):
            r, c = np.tril_indices(n)
            return (r, c)

        def hessian(self, x, lam, obj_factor, _ctr=counters):
            _ctr["nhev"] += 1
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
        P=ex.P,
        c=ex.c,
        A=ex.A,
        b=ex.b,
        G=ex.G,
        h=ex.h,
        lb=ex.lb,
        ub=ex.ub,
        tol=opts.get("tol"),
        max_iter=opts.get("max_iter"),
    )
    fun_val = float(res.obj) + ex.obj_const
    success = res.status == "optimal"
    selector = "lp-ipm" if ex.kind == "lp" else "qp-ipm"
    message = _QP_STATUS_MESSAGE.get(res.status, res.status)
    return OptimizeResult(
        x=np.asarray(res.x),
        fun=fun_val,
        success=success,
        status=_QP_STATUS_CODE.get(res.status, 1),
        message=message,
        nit=int(res.iters),
        # The convex solver consumes the extracted quadratic form, not the
        # python callables, so no objective/gradient/Hessian callbacks fire
        # during the solve. Report 0 (rather than omitting the keys) so the
        # scipy-standard counters are present on every result regardless of
        # which backend ran.
        nfev=0,
        njev=0,
        nhev=0,
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
        P=ex.P,
        c=ex.c,
        A=ex.A,
        b=ex.b,
        G=ex.G,
        h=ex.h,
        cones=ex.cones,
        tol=opts.get("tol"),
        max_iter=opts.get("max_iter"),
    )
    fun_val = float(res.obj) + ex.obj_const
    success = res.status == "optimal"
    message = _QP_STATUS_MESSAGE.get(res.status, res.status)
    return OptimizeResult(
        x=np.asarray(res.x),
        fun=fun_val,
        success=success,
        status=_QP_STATUS_CODE.get(res.status, 1),
        message=message,
        nit=int(res.iters),
        # See ``_solve_via_convex``: the conic solver works on the extracted
        # cone program, so the python objective callbacks never fire — report
        # 0 for the scipy-standard eval counters.
        nfev=0,
        njev=0,
        nhev=0,
        info={
            "solver": "socp",
            "problem_class": ex.kind,
            "obj_val": fun_val,
            "obj_constant": ex.obj_const,
            "status": res.status,
            "status_msg": message,
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
    return any(isinstance(c, dict) and c.get("jac") is None for c in constraints)


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | bool | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | LinearConstraint | dict | None = None,
    callback: Callable | None = None,
    warm_start=None,
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

    Auto-routing has a cost: detection probes the opaque callables and
    finite-differences a quadratic model of the objective (and any
    constraints), which is ``O(n²)`` extra ``fun`` evaluations. That is why
    routing is opt-in — the default ``"nlp"`` skips the probe entirely, so a
    general NLP (or an expensive ``fun``) pays nothing. Pass
    ``solver_selection="auto"`` only when you want the convex/conic fast paths
    and accept the detection overhead on problems that fall through to NLP.

    Like :func:`scipy.optimize.minimize`, this facade is **silent by default**.
    Pass ``disp=True`` for a concise log or an explicit ``print_level=N``
    (0–12) to control the NLP backend's IPM iteration table directly.

    Re-solves of a related problem can pass ``warm_start=`` a
    :class:`pounce.WarmStart` captured from a previous result
    (``WarmStart.from_info(res.x, res.info)``): it seeds the primal and
    dual iterates and sets the enabling warm-start options (see
    the initialization chapter of the docs). A warm start always runs on
    the NLP path — convex routing is skipped.
    """
    # Accept both calling conventions: scipy-style ``options={...}`` (one dict
    # argument) and the splatted ``**options`` form (kwargs absorbed by the
    # signature, as scipy's ``_custom`` dispatch sends them). Explicit kwargs
    # win over the legacy dict if both are supplied.
    legacy_options = options.pop("options", None)
    if legacy_options is not None and not isinstance(legacy_options, Mapping):
        # Silently discarding it would drop EVERY option the caller passed --
        # tolerances, iteration limits, solver_selection -- and still return a
        # plausible answer from the defaults. That is the gh #213 failure mode
        # one level up: the mistake is invisible because the solve succeeds.
        raise TypeError(
            f"options= must be a mapping, got {type(legacy_options).__name__}; "
            "all options would otherwise be silently ignored"
        )
    if isinstance(legacy_options, Mapping):
        options = {**dict(legacy_options), **options}

    # Promote a scalar / 0-d x0 to 1-D, matching scipy.optimize.minimize, so a
    # single-variable problem can be written ``minimize(f, 1.5)``.
    x0 = np.atleast_1d(_to_array(x0))
    n = x0.size
    lb, ub = _normalize_bounds(bounds, n)
    m, g_combined, jac_combined, cl, cu, jac_rows, jac_cols = _wrap_constraints(
        constraints, n, x0
    )

    # Capture the option keys the *user* actually passed, before we pop routing
    # keys or inject `disp`/`print_level` defaults below. The convex routers
    # honor only a subset, so this lets us warn about the rest (L48) instead of
    # silently discarding them.
    requested_opt_keys = set(options)

    # Solver routing (mirrors the CLI's `solver_selection`). Pop routing keys
    # so the remainder of `options` flows to the NLP backend. The default is
    # `"nlp"` (no probe) — opt in to `"auto"` to enable structure detection.
    selection = str(options.pop("solver_selection", "nlp")).lower()
    # Validate against the registry rather than letting an unrecognized value
    # fall through to the NLP path. Silently substituting a different engine
    # than the caller named is the worst failure mode here: it still returns a
    # correct answer on easy problems, so a typo (`qp_ipm`) or a selector this
    # facade does not implement is invisible until someone benchmarks or ships
    # the wrong solver. The CLI rejects these with OPTION_INVALID; match it.
    # (gh #213)
    if selection not in _SOLVER_SELECTION_VALUES:
        raise ValueError(
            f"solver_selection={selection!r} is not a valid selector; "
            f"expected one of {sorted(_SOLVER_SELECTION_VALUES)}"
        )
    route_tol = float(options.pop("route_tol", 1e-5))
    if warm_start is not None and selection != "nlp":
        warnings.warn(
            "pounce.minimize: warm_start= runs on the NLP path; "
            f"solver_selection={selection!r} is ignored for this solve.",
            stacklevel=2,
        )
        selection = "nlp"
    if selection == "qp-active-set":
        # Not a Python-side route: hand it back to the backend, whose
        # `is_sqp_algorithm_selected` treats it as equivalent to
        # `algorithm=active-set-sqp`. Note the library path does *not*
        # class-validate this selector (the CLI restricts it to LP/convex QP);
        # it simply runs the SQP engine on whatever problem it is given.
        options["solver_selection"] = "qp-active-set"
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
        def _jac_combined_dense(
            x, _vals=jac_combined, _r=jac_rows, _c=jac_cols, _m=m, _n=n
        ):
            out = np.zeros((_m, _n), dtype=np.float64)
            out[_r, _c] = _vals(x)
            return out

        router_jac_combined = _jac_combined_dense
    else:
        router_jac_combined = jac_combined  # value-vec; never read when nlp

    # Bind the objective's extra `args` into `fun`/`jac`/`hess` before handing
    # them to the routers, which probe them as bare `f(x)`. The NLP path below
    # applies `args` itself (`fun(x, *args)`), but the router copies would
    # otherwise call the raw callables without `args` — so a parameterized
    # convex objective either never routes (auto falls back to NLP) or is
    # wrongly rejected as "not convex" under a forced solver_selection. (#196
    # sibling: a user input silently not honored on the convex route.)
    def _bind_args(f):
        if f is None or not args:
            return f
        return lambda x, _f=f, _a=args: _f(x, *_a)

    # Wrap router callables in one shared point-cache (M34): the LP/QP and SOCP
    # routers probe an identical point set (same seed), so caching makes the
    # second router's probes cache hits instead of re-evaluating the objective.
    # Only these router copies are cached; the NLP fallback below still calls
    # the original `fun`/`jac`/… so the actual solve is unaffected.
    route_kw = dict(
        fun=_point_cache(_bind_args(fun)),
        jac=_point_cache(_bind_args(jac)),
        hess=_point_cache(_bind_args(hess)),
        lb=lb,
        ub=ub,
        m=m,
        g_combined=_point_cache(g_combined),
        jac_combined=_point_cache(router_jac_combined),
        cl=cl,
        cu=cu,
        x0=x0,
        rtol=route_tol,
    )
    # Options the dedicated convex (LP/QP/SOCP) routers actually honor; the
    # routing key and tolerance are consumed here, not "ignored". Everything
    # else the user passed (e.g. `print_level`, `disp`, `acceptable_tol`) has
    # no effect on the convex path, so warn rather than drop silently (L48).
    # `disp` is deliberately NOT in this set: `_solve_via_convex` /
    # `_solve_via_socp` read only `tol` / `max_iter`, so `disp=True` is dropped
    # on the convex routes and must trigger the warning below.
    _CONVEX_HONORED = {"solver_selection", "route_tol", "tol", "max_iter"}

    def _warn_convex_dropped_opts(route_name: str) -> None:
        ignored = sorted(requested_opt_keys - _CONVEX_HONORED)
        if hess is not None:
            ignored.append("hess (argument)")
        # The convex/SOCP routers consume the extracted quadratic form directly
        # and never call back into Python, so a user `callback` would never fire
        # on this route. It is a named argument (not in `options`), so the set
        # diff above cannot catch it — surface it explicitly rather than let it
        # be silently dropped (issue #196, related). Pass solver_selection='nlp'
        # to keep the callback firing.
        if callback is not None:
            ignored.append("callback (argument)")
        if ignored:
            warnings.warn(
                f"pounce.minimize routed this problem to the dedicated {route_name} "
                "solver, which honors only 'tol' and 'max_iter'. These were ignored: "
                f"{ignored}. Pass solver_selection='nlp' to force the general NLP "
                "solver, which honors them.",
                stacklevel=2,
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
            _warn_convex_dropped_opts("convex LP/QP")
            return _solve_via_convex(extract, options)
        # Auto: an LP/QP wasn't found — try a convex QCQP before giving up to
        # the NLP solver (a quadratic *constraint* lands here, not above).
        if selection == "auto":
            socp = classify_and_extract_socp(**route_kw)
            if socp is not None:
                _warn_convex_dropped_opts("convex SOCP")
                return _solve_via_socp(socp, options)
    elif selection == "socp":
        socp = classify_and_extract_socp(**route_kw)
        if socp is None:
            raise ValueError(
                "solver_selection='socp' but the problem was not detected as a "
                "convex QCQP (convex-quadratic objective and/or constraints, all "
                "convex, with only linear equalities)"
            )
        _warn_convex_dropped_opts("convex SOCP")
        return _solve_via_socp(socp, options)

    eval_counters: dict[str, int] = {"nfev": 0, "njev": 0, "nhev": 0}

    # A user-supplied ``hess`` equals the Lagrangian Hessian (so the IPM can use
    # it directly) when unconstrained or when *every* constraint is linear — then
    # the constraint-curvature term ``Σ λᵢ ∇²gᵢ`` vanishes. We rely only on the
    # declared ``LinearConstraint`` type (no probing).
    constraints_all_linear = _all_linear_constraints(constraints)

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
        fd_targets.append("constraint Jacobian(s) (pass 'jac' in each constraint dict)")
    if fd_targets:
        warnings.warn(
            "pounce.minimize is approximating "
            + " and ".join(fd_targets)
            + " by finite differences. This is slower and less accurate than "
            "analytic derivatives. For a faster, more robust solve supply them "
            "directly, or use the autodiff frontends pounce.jax / pounce.torch.",
            stacklevel=2,
        )

    # (L48) The NLP wrapper can only supply the *objective* Hessian; for
    # *nonlinear* constraints it has no way to assemble the constraint-curvature
    # term ``Σ λᵢ ∇²gᵢ`` the IPM needs for the full Lagrangian Hessian, so a
    # supplied ``hess`` is dropped (L-BFGS fallback). With all-linear constraints
    # that term is zero, so ``hess`` IS the Lagrangian Hessian and is used —
    # no warning. Warn only for the genuinely-nonlinear case.
    if hess is not None and m > 0 and not constraints_all_linear:
        warnings.warn(
            "pounce.minimize ignores the supplied 'hess' when nonlinear "
            "constraints are present: the wrapper cannot form the "
            "constraint-curvature term of the Lagrangian Hessian, so the solver "
            "uses an L-BFGS approximation. The objective Hessian is used for "
            "unconstrained problems and problems with only linear constraints.",
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
        constraints_all_linear=constraints_all_linear,
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

    # Pass warm_start only when set: test doubles (and any cyipopt-style
    # stand-in) may not accept the keyword.
    if warm_start is not None:
        x, info = problem.solve(x0=x0, warm_start=warm_start)
    else:
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
        status_code not in _NO_KKT_FALLBACK_STATUS
        and np.isfinite(kkt_error)
        and kkt_error <= acceptable_tol
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
        nhev=int(eval_counters["nhev"]),
        # Solve wall-clock total and its per-subsystem breakdown (seconds),
        # surfaced top-level for discoverability (pounce#180 item 3). The
        # full breakdown dict — objective / gradient / constraint /
        # Jacobian / Lagrangian-Hessian eval time and the linear-algebra
        # factorization-vs-back-solve split — is also in ``info["timing"]``.
        wall_time=float(info.get("wall_time", float("nan"))),
        timing=dict(info.get("timing", {})),
        info=dict(info),
    )
