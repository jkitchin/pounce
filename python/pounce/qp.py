"""Convex LP/QP solver — Pythonic wrapper over the ``pounce-convex`` IPM.

Solves the standard-form convex quadratic program

.. code-block:: text

    minimize    ½ xᵀP x + cᵀx
    subject to  A x = b
                G x ≤ h
                lb ≤ x ≤ ub

with a specialized interior-point method (Mehrotra predictor-corrector),
presolve, and verified infeasibility / unboundedness detection. ``P = 0``
gives an LP.

This module is the friendly surface over the compiled ``_pounce``
bindings: it accepts dense vectors and (optionally) scipy-sparse or dense
matrices, and returns a small :class:`QpResult`. For differentiable QP
layers (JAX), see :mod:`pounce.jax` (``solve_qp`` / ``QpLayer``).

For problems with more than ~1000 variables, pass ``P`` **and** the
constraint matrices ``A``/``G`` as **scipy-sparse** matrices (e.g.
``scipy.sparse.csc_matrix``): the dense path is 60-80x slower and far
heavier on memory at that size, and a large dense matrix triggers a
one-time :class:`PounceSparsityWarning`.

Example
-------
>>> import numpy as np
>>> from pounce.qp import solve_qp
>>> # min ½‖x‖²·2 − 3x0 − 4x1  s.t.  0 ≤ x ≤ 1
>>> r = solve_qp(P=np.diag([2.0, 2.0]), c=[-3.0, -4.0],
...              lb=[0, 0], ub=[1, 1])
>>> r.status, r.x
('optimal', array([1., 1.]))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from . import _pounce

__all__ = [
    "ActiveSet",
    "QpResult",
    "QpFactorization",
    "QpSensitivity",
    "ReducedHessian",
    "PounceSparsityWarning",
    "solve_qp",
    "solve_socp",
    "solve_qp_batch",
    "solve_qp_multi_rhs",
]


class PounceSparsityWarning(UserWarning):
    """A large *dense* matrix was passed to a convex solver where a scipy-sparse
    matrix would be dramatically faster and smaller (issue #116). Silence with
    ``warnings.filterwarnings("ignore", category=pounce.qp.PounceSparsityWarning)``."""


# Dense matrices at/above this element count put the convex solver on its dense
# path, which at a few thousand variables is 60-80x slower and far heavier than
# the scipy-sparse path (issue #116). ~1e6 ≈ a 1000x1000 dense matrix.
_DENSE_WARN_ELEMS = 1_000_000
_dense_input_warned = False


def _warn_large_dense(what: str, shape) -> None:
    """Emit a one-time :class:`PounceSparsityWarning` for a large dense input."""
    global _dense_input_warned
    if _dense_input_warned:
        return
    _dense_input_warned = True
    warnings.warn(
        f"a large dense `{what}` ({shape[0]}x{shape[1]}) was passed to the "
        "convex solver. At this size the dense path can be 60-80x slower and use "
        "far more memory than scipy-sparse inputs; if the matrix is sparse, pass "
        "a scipy.sparse matrix (e.g. scipy.sparse.csc_matrix(M)) for both `P` and "
        "the constraint blocks. This warning is emitted once per process.",
        PounceSparsityWarning,
        stacklevel=4,
    )


@dataclass
class QpResult:
    """Solution of a convex QP.

    Attributes
    ----------
    status:
        One of ``"optimal"``, ``"primal_infeasible"``,
        ``"dual_infeasible"`` (unbounded), ``"iteration_limit"``,
        ``"numerical_failure"``.
    x:
        Primal solution, shape ``(n,)``.
    y:
        Equality multipliers, shape ``(m_eq,)``.
    z:
        Inequality multipliers ``≥ 0``, shape ``(m_ineq,)``.
    z_lb, z_ub:
        Bound multipliers ``≥ 0``, shape ``(n,)``.
    obj:
        Objective value ``½ xᵀP x + cᵀx``.
    iters:
        Interior-point iterations taken.
    residuals:
        Final KKT residuals as a dict with keys
        ``primal_infeasibility``, ``dual_infeasibility``,
        ``complementarity``, and ``kkt_error`` (the max of the three).
        For a conic (:func:`solve_socp`) solve these are measured against
        the solve's own cones — cone-membership violation for the primal
        residual and the per-block inner product for complementarity —
        not the orthant's per-row test, which is meaningless there.
    iterates:
        Per-iteration convergence trace — a list of dicts with keys
        ``iter``, ``objective``, ``primal_infeasibility``,
        ``dual_infeasibility``, ``mu``, ``alpha_primal``, ``alpha_dual``.
        Empty unless the solve was called with ``collect_iterates=True``.
    scaling_warning:
        ``None`` on a cleanly-solved, well-scaled problem. Otherwise a
        human-readable warning that the objective curvature ``‖P‖`` is tiny
        relative to the problem data and the (non-``optimal``) result may be
        inaccurate — set only when the solve did not converge cleanly *and* the
        problem is in that ill-scaled regime, with an actionable remedy
        (rescale the objective, or cross-check with a reference solver).
    """

    status: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    z_lb: np.ndarray
    z_ub: np.ndarray
    obj: float
    iters: int
    residuals: Optional[dict] = None
    iterates: list = field(default_factory=list)
    scaling_warning: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == "optimal"

    @property
    def kkt_error(self) -> Optional[float]:
        """Overall KKT error (the max residual), or ``None`` if unavailable."""
        return None if self.residuals is None else self.residuals["kkt_error"]


@dataclass
class ReducedHessian:
    """Reduced Hessian of a QP on its active manifold, with eigendecomposition.

    Attributes
    ----------
    n_dof:
        Degrees of freedom — the dimension of every array here. Equals
        ``n`` minus the rank of the active-constraint Jacobian.
    matrix:
        The reduced Hessian ``H_R = Zᵀ P Z``, shape ``(n_dof, n_dof)``.
    eigenvalues:
        Eigenvalues of ``H_R`` in ascending order, shape ``(n_dof,)``. All
        positive ⟺ a strict second-order minimizer; the smallest gives the
        weakest curvature, and the spread is the conditioning on the active
        manifold.
    eigenvectors:
        Eigenvectors as columns, shape ``(n_dof, n_dof)``; column ``j``
        pairs with ``eigenvalues[j]``.
    """

    n_dof: int
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def is_positive_definite(self) -> bool:
        """Whether every eigenvalue is positive (strict second-order min)."""
        return self.n_dof == 0 or bool(self.eigenvalues[0] > 0.0)


@dataclass(frozen=True)
class ActiveSet:
    """Which constraints of a QP are active at the optimum.

    The two index spaces are distinct and are kept separate deliberately:
    ``inequalities`` indexes rows of ``G``, ``bounds`` indexes *variables*.

    Attributes
    ----------
    inequalities:
        Row indices of ``G`` whose constraint is active.
    bounds:
        Indices of variables whose lower or upper bound is active.
    """

    inequalities: Tuple[int, ...]
    bounds: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.inequalities) + len(self.bounds)

    def __bool__(self) -> bool:
        return len(self) > 0


def _coo(mat, n_cols: int, what: str):
    """Return ``(rows, cols, vals)`` int/int/float lists for a matrix
    given as a scipy-sparse matrix, a dense array, or ``None``."""
    if mat is None:
        return [], [], []
    # scipy sparse (any format) → COO.
    if hasattr(mat, "tocoo"):
        coo = mat.tocoo()
        return (
            coo.row.astype(np.int64).tolist(),
            coo.col.astype(np.int64).tolist(),
            coo.data.astype(np.float64).tolist(),
        )
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D matrix, got shape {arr.shape}")
    if arr.size >= _DENSE_WARN_ELEMS:
        _warn_large_dense(what, arr.shape)
    rows, cols = np.nonzero(arr)
    return (
        rows.astype(np.int64).tolist(),
        cols.astype(np.int64).tolist(),
        arr[rows, cols].tolist(),
    )


def _lower_triangle_coo(P, n: int):
    """COO of the lower triangle of the symmetric Hessian ``P``.

    Accepts a scipy-sparse or dense ``P`` (assumed symmetric) and keeps
    only entries with ``row >= col``; ``None`` → empty (an LP)."""
    r, c, v = _coo(P, n, "P")
    out_r, out_c, out_v = [], [], []
    for ri, ci, vi in zip(r, c, v):
        if ri >= ci:
            out_r.append(ri)
            out_c.append(ci)
            out_v.append(vi)
    return out_r, out_c, out_v


# Largest n for which the default (auto) PSD check runs a dense O(n³)
# eigenvalue solve. Above this the check is skipped unless ``check_psd=True``
# is passed explicitly, so a large sparse QP is not silently slowed down by an
# O(n³) densify-and-eig (see the dense-input scaling concern, issue #116).
_PSD_CHECK_AUTO_MAX_N = 1500


def _min_eig_lower_coo(pr, pc, pv, n: int) -> float:
    """Smallest eigenvalue of the symmetric Hessian reconstructed from its
    lower-triangle COO — i.e. exactly the matrix the solver sees.

    Duplicate ``(row, col)`` entries **accumulate**, matching both the COO
    convention and what the solver does with them. Assigning instead of
    accumulating (last-duplicate-wins) made this guard validate a different
    matrix than the one being solved: ``coo_matrix(([2, 2, 1.5, 1.5],
    ([0, 1, 1, 1], [0, 1, 0, 0])))`` is indefinite once summed
    (eigenvalues ``[-1, 5]``) but positive definite under overwrite
    (``[0.5, 3.5]``), so an indefinite ``P`` sailed past ``check_psd`` and
    ``solve_qp`` returned ``status="optimal"`` at a saddle point. See gh #279.

    The diagonal is written once per entry — accumulating into both
    ``(ri, ci)`` and ``(ci, ri)`` would double it when ``ri == ci``.
    """
    M = np.zeros((n, n), dtype=np.float64)
    for ri, ci, vi in zip(pr, pc, pv):
        M[ri, ci] += vi
        if ri != ci:
            M[ci, ri] += vi
    return float(np.linalg.eigvalsh(M)[0]) if n else 0.0


def _check_psd(pr, pc, pv, n: int) -> None:
    """Raise ``ValueError`` if the Hessian ``P`` is not positive semidefinite.

    The convex IPM and its unboundedness detection assume a PSD ``P``; an
    indefinite ``P`` otherwise returns a silently-wrong ``status="optimal"``
    (issue #112). The tolerance is relative to the spectral scale so genuine
    PSD matrices with round-off-level negative eigenvalues pass."""
    if not pr:  # no Hessian entries → LP, trivially PSD
        return
    lam_min = _min_eig_lower_coo(pr, pc, pv, n)
    scale = max(abs(v) for v in pv)
    if lam_min < -1e-8 * max(scale, 1.0):
        raise ValueError(
            f"P is not positive semidefinite (min eigenvalue {lam_min:.3e}); "
            "the convex QP solver requires a PSD Hessian. A nonconvex QP is "
            "unbounded below in the indefinite directions and has no convex "
            "optimum. Pass check_psd=False to skip this check (e.g. if you "
            "know P is PSD and want to avoid the O(n^3) eigenvalue cost)."
        )


def _maybe_check_psd(P, c, check_psd) -> None:
    """Run the issue-#112 PSD guard on ``P`` unless explicitly disabled.

    Shared by *every* QP entry point so an indefinite (nonconvex) Hessian is
    rejected uniformly — not only by :func:`solve_qp`. ``check_psd=False``
    skips it; ``True`` forces it; ``None`` (the default) runs it only when
    ``n <= _PSD_CHECK_AUTO_MAX_N`` so a large QP is not slowed by the O(n^3)
    eigenvalue solve. ``c`` fixes ``n``."""
    if check_psd is False:
        return
    n = np.asarray(c, dtype=np.float64).ravel().shape[0]
    if check_psd or n <= _PSD_CHECK_AUTO_MAX_N:
        _check_psd(*_lower_triangle_coo(P, n), n)


def _mat_shape(mat):
    """``(n_rows, n_cols)`` of a sparse-or-dense matrix, or ``None`` for a
    ``None`` matrix or a dense array that is not 2-D (``_coo`` raises a clear
    error for the latter)."""
    if mat is None:
        return None
    if hasattr(mat, "tocoo") and hasattr(mat, "shape"):  # scipy sparse
        return tuple(mat.shape)
    sh = np.asarray(mat).shape
    return sh if len(sh) == 2 else None


def _validate(P, c, A, b, G, h, lb, ub, n: int) -> None:
    """Reject malformed inputs up front with a precise ``ValueError`` instead
    of a misleading solver status (issue #113): a shape mismatch otherwise
    surfaces as ``primal_infeasible`` and a NaN/Inf as ``iteration_limit``."""

    def _finite(name, arr, allow_inf=False):
        if arr is None:
            return
        data = np.asarray(
            arr.tocoo().data if hasattr(arr, "tocoo") else arr, dtype=np.float64
        )
        if not data.size:
            return
        # ±inf bounds are the idiomatic "no bound"; only NaN is malformed there.
        bad = np.isnan(data) if allow_inf else ~np.isfinite(data)
        if bad.any():
            what = "NaN" if allow_inf else "NaN or Inf"
            raise ValueError(f"solve_qp: `{name}` contains {what}")

    for name, arr in (("P", P), ("c", c), ("A", A), ("b", b), ("G", G), ("h", h)):
        _finite(name, arr)
    _finite("lb", lb, allow_inf=True)
    _finite("ub", ub, allow_inf=True)

    psh = _mat_shape(P)
    if psh is not None and psh != (n, n):
        raise ValueError(f"solve_qp: `P` has shape {psh} but must be ({n}, {n})")

    for mname, mat, vname, vec in (("A", A, "b", b), ("G", G, "h", h)):
        sh = _mat_shape(mat)
        if sh is None:
            continue
        rows, cols = sh
        if cols != n:
            raise ValueError(
                f"solve_qp: `{mname}` has {cols} columns but n={n} (from `c`)"
            )
        vlen = 0 if vec is None else np.asarray(vec).ravel().shape[0]
        if vlen != rows:
            raise ValueError(
                f"solve_qp: `{mname}` has {rows} rows but `{vname}` has length {vlen}"
            )

    for name, vec in (("lb", lb), ("ub", ub)):
        if vec is not None:
            vlen = np.asarray(vec).ravel().shape[0]
            if vlen != n:
                raise ValueError(
                    f"solve_qp: `{name}` has length {vlen} but n={n} (from `c`)"
                )

    # ±inf marks an *absent* bound (lower = -inf, upper = +inf). The opposite
    # signs are not "absent" — they are constraints no finite value can meet.
    # The solver's presence test (`lb > -BOUND_INF`, `ub < BOUND_INF`) is
    # sign-agnostic, so `lb = +inf` / `ub = -inf` were dropped as if unbounded
    # and the solve returned `status="optimal"` at a point violating the stated
    # bound by an infinite margin. Note the *finite* analogue (`lb=1 > ub=0`)
    # is correctly reported `primal_infeasible` by the solver, so this rejects
    # only the degenerate spellings that silently produced a wrong answer.
    # See gh #275.
    for name, vec, bad_val, cmp_txt in (
        ("lb", lb, np.inf, ">= +inf"),
        ("ub", ub, -np.inf, "<= -inf"),
    ):
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float64).ravel()
        bad = np.where(arr == bad_val)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"solve_qp: `{name}[{i}]` is {arr[i]}, which no finite value can "
                f"satisfy (it requires x[{i}] {cmp_txt}). Use "
                f"{'-inf' if name == 'lb' else '+inf'} to leave that side "
                f"unbounded, or a finite value for a real bound"
            )


def _build(
    P,
    c: Sequence[float],
    A,
    b: Optional[Sequence[float]],
    G,
    h: Optional[Sequence[float]],
    lb: Optional[Sequence[float]],
    ub: Optional[Sequence[float]],
) -> "_pounce.QpProblem":
    c = np.asarray(c, dtype=np.float64).ravel()
    n = c.shape[0]
    _validate(P, c, A, b, G, h, lb, ub, n)
    pr, pc, pv = _lower_triangle_coo(P, n)
    ar, ac, av = _coo(A, n, "A")
    gr, gc, gv = _coo(G, n, "G")
    return _pounce.QpProblem(
        n=n,
        c=c.tolist(),
        p_rows=pr,
        p_cols=pc,
        p_vals=pv,
        a_rows=ar,
        a_cols=ac,
        a_vals=av,
        b=[] if b is None else np.asarray(b, dtype=np.float64).ravel().tolist(),
        g_rows=gr,
        g_cols=gc,
        g_vals=gv,
        h=[] if h is None else np.asarray(h, dtype=np.float64).ravel().tolist(),
        lb=[] if lb is None else np.asarray(lb, dtype=np.float64).ravel().tolist(),
        ub=[] if ub is None else np.asarray(ub, dtype=np.float64).ravel().tolist(),
    )


def _to_result(d: dict) -> QpResult:
    return QpResult(
        status=d["status"],
        x=np.asarray(d["x"]),
        y=np.asarray(d["y"]),
        z=np.asarray(d["z"]),
        z_lb=np.asarray(d["z_lb"]),
        z_ub=np.asarray(d["z_ub"]),
        obj=float(d["obj"]),
        iters=int(d["iters"]),
        residuals=d.get("residuals"),
        iterates=list(d.get("iterates", [])),
        scaling_warning=d.get("scaling_warning"),
    )


# A convergence tolerance here bounds the max KKT residual / duality measure,
# and the convex IPM tests it at *every* iterate — including the interior-point
# self-dual starting point. Unlike the NLP line search (which makes progress and
# still returns the right answer for a loose tol), the convex solver therefore
# *short-circuits* at a non-stationary point whenever `tol` is loose enough to
# admit the starting iterate: with `tol >= 1` the O(1) KKT residual at the start
# already "passes", so the solve returns after 0 iterations at a wildly wrong
# point still labeled ``status="optimal"`` (gh #277). A meaningful KKT tolerance
# is well below 1, so reject `tol >= 1`; this guarantees that an accepted `tol`
# with an ``"optimal"`` result carries `kkt_error <= tol < 1` — a genuinely
# near-stationary point, never the 0-iteration wrong point. The unsatisfiable
# `tol <= 0` / non-finite values are rejected the same way every other pounce
# surface already does (NLP ``minimize``, the CLI, and ``sos_minimize`` all
# raise ``OPTION_INVALID``).
_TOL_MAX = 1.0


def _validate_solver_opts(tol, max_iter, func: str) -> None:
    """Validate the shared ``tol`` / ``max_iter`` options for every convex
    entry point, matching the NLP / CLI / ``sos_minimize`` surfaces (which all
    reject ``tol <= 0`` and non-finite ``tol`` and a non-positive iteration
    count with ``OPTION_INVALID``).

    Both are optional (``None`` keeps the solver default). ``max_iter`` is
    checked here — *before* it reaches the PyO3 ``usize`` binding — so a
    negative value raises a clear named error instead of leaking a raw
    ``OverflowError: can't convert negative int to unsigned`` (gh #277)."""
    if tol is not None:
        t = float(tol)
        if not np.isfinite(t) or t <= 0.0 or t >= _TOL_MAX:
            raise ValueError(
                f"{func}: `tol` must be a finite positive number below "
                f"{_TOL_MAX} (it bounds the KKT-residual convergence measure); "
                f"got {tol!r}. A value <= 0, NaN, or Inf is unsatisfiable, and "
                f"tol >= {_TOL_MAX} would accept the non-stationary starting "
                f"iterate and return a wrong point labeled 'optimal'."
            )
    if max_iter is not None:
        # bool is an int subclass; treat True/False as a type error, not 1/0.
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise ValueError(
                f"{func}: `max_iter` must be a positive integer, got {max_iter!r}"
            )
        if max_iter < 1:
            raise ValueError(
                f"{func}: `max_iter` must be a positive integer (at least 1), "
                f"got {max_iter}"
            )


def _warm_dict(warm):
    """Coerce a warm start (a :class:`QpResult` or a mapping) into the
    ``{x, y, z, z_lb, z_ub}`` dict the binding expects, or ``None``."""
    if warm is None:
        return None
    if isinstance(warm, QpResult):
        src = {
            "x": warm.x,
            "y": warm.y,
            "z": warm.z,
            "z_lb": warm.z_lb,
            "z_ub": warm.z_ub,
        }
    else:
        src = warm
    out = {}
    for k in ("x", "y", "z", "z_lb", "z_ub"):
        v = src.get(k) if hasattr(src, "get") else src[k]
        if v is not None:
            out[k] = np.asarray(v, dtype=np.float64).ravel().tolist()
    return out


def solve_qp(
    P=None,
    c=None,
    A=None,
    b=None,
    G=None,
    h=None,
    lb=None,
    ub=None,
    *,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_start=None,
    collect_iterates: bool = False,
    check_psd: Optional[bool] = None,
) -> QpResult:
    """Solve one convex QP. See the module docstring for the form.

    ``P`` (lower triangle is used; assumed symmetric) and ``A``/``G`` may
    be scipy-sparse or dense; ``None`` matrices are empty. ``c`` is
    required and sets ``n``.

    ``warm_start`` (optional) is a previous :class:`QpResult` (or a mapping
    with ``x``/``y``/``z``/``z_lb``/``z_ub``) for a *nearby* problem. It
    seeds the interior-point iteration to reduce the iteration count; it
    does not change the solution, and a dimension mismatch is ignored.

    ``check_psd`` guards against an indefinite (nonconvex) ``P``, which the
    convex solver would otherwise accept and report a silently-wrong
    ``"optimal"`` for (issue #112). ``None`` (the default) runs the check
    only when ``n <= 1500`` so a large sparse QP is not slowed by the
    O(n^3) eigenvalue solve; pass ``True`` to always check or ``False`` to
    never check.

    The returned :class:`QpResult` carries the final KKT ``residuals``;
    pass ``collect_iterates=True`` to also capture the per-iteration
    convergence trace in ``result.iterates``.
    """
    if c is None:
        raise ValueError("solve_qp: `c` is required")
    _validate_solver_opts(tol, max_iter, "solve_qp")
    _maybe_check_psd(P, c, check_psd)
    prob = _build(P, c, A, b, G, h, lb, ub)
    return _to_result(
        _pounce.solve_qp(
            prob,
            tol=tol,
            max_iter=max_iter,
            warm_start=_warm_dict(warm_start),
            collect_iterates=collect_iterates,
        )
    )


def _normalize_cones(cones):
    """Coerce a cone partition into the binding's ``[(kind, dim), …]``.

    Accepts ``("soc", 3)`` / ``("nonneg", 2)`` / ``("exp", 3)`` /
    ``("pow", 0.5)`` / ``("psd", 3)`` tuples, or the shorthand ``3`` (a
    second-order cone of that dim). Kind strings are case-insensitive
    (``"soc"``/``"q"``, ``"nonneg"``/``"nn"``/``"+"``,
    ``"exp"``/``"exponential"``, ``"pow"``/``"power"``, ``"psd"``/``"sdp"``).
    The second element is the dimension for ``soc``/``nonneg``, the exponent
    ``α`` for ``pow``, and the **matrix size n** for ``psd`` (spanning
    ``n(n+1)/2`` svec rows)."""
    out = []
    for spec in cones:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            # Pass the value through as a float; the binding interprets it as a
            # dimension (soc/nonneg) or an exponent (pow).
            out.append((str(spec[0]), float(spec[1])))
        elif isinstance(spec, int):
            out.append(("soc", float(spec)))
        else:
            raise ValueError(f"bad cone spec {spec!r}; use (kind, dim) or an int")
    return out


def solve_socp(
    P=None,
    c=None,
    A=None,
    b=None,
    G=None,
    h=None,
    *,
    cones,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    collect_iterates: bool = False,
    check_psd: Optional[bool] = None,
) -> QpResult:
    """Solve a standard-form conic program (LP/QP + second-order and/or
    exponential cones).

    Same form as :func:`solve_qp` minus variable bounds, but the inequality
    block ``Gx ≤ h`` is partitioned by ``cones`` — a sequence of
    ``(kind, dim)`` specs covering the rows of ``G`` in order. Each slack
    block ``s = h − Gx`` must lie in its cone:

    - ``("nonneg", d)`` — the nonnegative orthant ``s ≥ 0``;
    - ``("soc", d)`` — the second-order cone ``{ (t, x) : t ≥ ‖x‖₂ }``
      (an int ``d`` is shorthand for this);
    - ``("exp", 3)`` — the 3-D exponential cone
      ``{ (x, y, z) : y·exp(x/y) ≤ z, y > 0 }``, which routes to the
      non-symmetric HSDE solver and unlocks geometric programming, entropy,
      log-sum-exp, and logistic models;
    - ``("pow", α)`` — the 3-D power cone
      ``{ (x, y, z) : |x| ≤ y^α z^{1−α}, y,z ≥ 0 }`` with ``α ∈ (0, 1)``
      (the second tuple element is the **exponent**, not a dimension); the
      building block for ``p``-norm and general geometric constraints.
    - ``("psd", n)`` — the positive-semidefinite cone over symmetric
      ``n×n`` matrices (small dense SDPs). Its slack block is the
      **symmetric vectorization** ``svec(X)`` (length ``n(n+1)/2``; lower
      triangle, column by column, off-diagonals scaled by ``√2`` so that
      ``⟨X,Y⟩ = svec(X)·svec(Y)``), and ``smat(s) ⪰ 0`` is enforced.

    A second-order cone may be freely mixed with an exp/power cone (the
    non-symmetric driver handles both). The PSD cone is self-scaled and runs
    on the symmetric driver, so it **cannot** be combined with exp/power
    cones in one problem (a clear error is raised if you try).

    Examples
    --------
    >>> # min t  s.t.  (t, x − x*) ∈ SOC   (minimize ‖x − x*‖)
    >>> r = solve_socp(c=[1, 0, 0], G=-np.eye(3), h=[0, -2, 1],
    ...                cones=[("soc", 3)])

    >>> # Geometric program  min x + 1/x = min_u e^u + e^{-u}  (optimum 2).
    >>> # Variables (u, t1, t2); (u,1,t1)∈Kexp, (-u,1,t2)∈Kexp.
    >>> import numpy as np
    >>> G = np.zeros((6, 3))
    >>> G[0, 0] = -1.0   # s0 = u
    >>> G[2, 1] = -1.0   # s2 = t1
    >>> G[3, 0] = 1.0    # s3 = -u
    >>> G[5, 2] = -1.0   # s5 = t2
    >>> r = solve_socp(c=[0, 1, 1], G=G, h=[0, 1, 0, 0, 1, 0],
    ...                cones=[("exp", 3), ("exp", 3)])
    >>> round(r.obj, 6)
    2.0
    """
    if c is None:
        raise ValueError("solve_socp: `c` is required")
    _validate_solver_opts(tol, max_iter, "solve_socp")
    _maybe_check_psd(P, c, check_psd)
    prob = _build(P, c, A, b, G, h, None, None)
    specs = _normalize_cones(cones)
    return _to_result(
        _pounce.solve_socp(
            prob, specs, tol=tol, max_iter=max_iter, collect_iterates=collect_iterates
        )
    )


def solve_qp_batch(
    problems: Sequence[dict],
    *,
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    warm_starts: Optional[Sequence] = None,
    check_psd: Optional[bool] = None,
) -> list[QpResult]:
    """Solve a batch of convex QPs in parallel (across instances).

    ``problems`` is a sequence of kwarg dicts, each accepted by
    :func:`solve_qp` (keys ``P, c, A, b, G, h, lb, ub``). Returns one
    :class:`QpResult` per input, in order.

    ``warm_starts`` (optional) is a sequence — one per problem — of prior
    :class:`QpResult`\\ s or mappings (for a sequence of nearby batches).
    Each seeds its instance's iteration; mismatched entries are ignored.

    ``check_psd`` guards each problem's Hessian against indefiniteness
    (issue #112), with the same ``None``/``True``/``False`` semantics as
    :func:`solve_qp`; an offending problem raises ``ValueError`` before any
    solve runs.
    """
    _validate_solver_opts(tol, max_iter, "solve_qp_batch")
    for pr in problems:
        _maybe_check_psd(pr.get("P"), pr["c"], check_psd)
    built = [
        _build(
            pr.get("P"),
            pr["c"],
            pr.get("A"),
            pr.get("b"),
            pr.get("G"),
            pr.get("h"),
            pr.get("lb"),
            pr.get("ub"),
        )
        for pr in problems
    ]
    ws = None
    if warm_starts is not None:
        if len(warm_starts) != len(built):
            raise ValueError(
                f"warm_starts has length {len(warm_starts)}, expected {len(built)}"
            )
        ws = [_warm_dict(w) or {} for w in warm_starts]
    dicts = _pounce.solve_qp_batch(built, tol=tol, max_iter=max_iter, warm_starts=ws)
    return [_to_result(d) for d in dicts]


def solve_qp_multi_rhs(
    P=None,
    c=None,
    A=None,
    b=None,
    G=None,
    h=None,
    lb=None,
    ub=None,
    *,
    cs: Sequence[Sequence[float]],
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
    check_psd: Optional[bool] = None,
) -> list[QpResult]:
    """Solve one QP *structure* against many linear objectives, in parallel.

    All of ``P``/``A``/``b``/``G``/``h``/``lb``/``ub`` are shared; only the
    linear term varies, given as ``cs`` — a sequence of length-``n`` vectors
    (one objective per solve). Returns one :class:`QpResult` per entry of
    ``cs``, in order. The ``c`` argument here is only a placeholder for
    shape; the per-solve objectives come from ``cs``.

    This is the multiple-right-hand-side analog of :func:`solve_qp_batch`:
    use it when the constraint geometry is fixed and you are sweeping the
    objective (e.g. a family of cost vectors, a parametric linear term, or
    the inner objective of a bilevel sweep).
    """
    if cs is None or len(cs) == 0:
        raise ValueError("solve_qp_multi_rhs: `cs` must be a non-empty sequence")
    _validate_solver_opts(tol, max_iter, "solve_qp_multi_rhs")
    n = len(np.asarray(cs[0], dtype=np.float64).ravel())
    # `c` only fixes `n` for the base structure; the real objectives are `cs`.
    base_c = c if c is not None else np.zeros(n)
    # `P` is shared across every right-hand side, so one check covers the batch.
    _maybe_check_psd(P, base_c, check_psd)
    base = _build(P, base_c, A, b, G, h, lb, ub)
    cs_list = [np.asarray(ci, dtype=np.float64).ravel().tolist() for ci in cs]
    dicts = _pounce.solve_qp_multi_rhs(base, cs_list, tol=tol, max_iter=max_iter)
    return [_to_result(d) for d in dicts]


class QpFactorization:
    """Build-once / solve-many handle for a fixed QP *structure*.

    Builds the KKT symbolic factor once; each :meth:`solve` reuses it for
    a problem that shares the structure (same sparsity and set of finite
    bounds, varying only ``c``/``b``/``h``/bound *values*). A mismatched
    problem returns a result with status ``"numerical_failure"``.
    """

    def __init__(
        self,
        P=None,
        c=None,
        A=None,
        b=None,
        G=None,
        h=None,
        lb=None,
        ub=None,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        check_psd: Optional[bool] = None,
    ):
        if c is None:
            raise ValueError(
                "QpFactorization: `c` is required (representative problem)"
            )
        _validate_solver_opts(tol, max_iter, "QpFactorization")
        # `P` is fixed for the lifetime of the handle, so one check at build
        # time covers every same-structure `solve` (issue #112).
        _maybe_check_psd(P, c, check_psd)
        base = _build(P, c, A, b, G, h, lb, ub)
        self._inner = _pounce.QpFactorization(base, tol=tol, max_iter=max_iter)

    def solve(
        self,
        P=None,
        c=None,
        A=None,
        b=None,
        G=None,
        h=None,
        lb=None,
        ub=None,
        *,
        warm_start=None,
    ) -> QpResult:
        """Solve a same-structure instance, reusing the symbolic factor.

        Pass ``warm_start`` (a previous :class:`QpResult` for a nearby
        problem) to also seed the iteration — combining symbolic-factor
        reuse with warm starting.
        """
        if c is None:
            raise ValueError("QpFactorization.solve: `c` is required")
        prob = _build(P, c, A, b, G, h, lb, ub)
        return _to_result(self._inner.solve(prob, warm_start=_warm_dict(warm_start)))


class QpSensitivity:
    """Post-optimal sensitivity for a convex QP — the sIPOPT analog.

    Solves the QP on construction and holds the active-set KKT
    factorization, so each :meth:`parametric_step` is a single
    back-substitution (build-once / solve-many). This mirrors the NLP
    :class:`pounce.Solver` session — which caches the converged factor for
    ``parametric_step`` / ``reduced_hessian`` — specialized to a QP, where
    the Lagrangian Hessian is the constant ``P``.

    The standard use is a *parametric* QP: designate one or more equality
    constraints as parameters (their right-hand side ``b`` is the
    parameter), then predict how the optimum moves as those values change.
    ``sensitivity.x + sensitivity.parametric_step(pins, deltas)`` is the
    first-order predictor of the perturbed solution — exact while the
    active set is unchanged.

    On a *near*-LICQ problem (active-constraint gradients nearly, but not
    exactly, rank-deficient) the sensitivity KKT is near-singular and
    ``parametric_step`` can silently over-damp ``dx/db`` (issue #284). Two
    guards address this: the solve is internally refined against the
    unregularized KKT to recover ``dx/db`` wherever the information survives in
    double precision, and :attr:`ill_conditioned` / :attr:`kkt_cond_estimate`
    (build-time) and :attr:`last_step_residual` (per-step) let a caller
    *detect* when a step is untrustworthy.

    Example
    -------
    >>> import numpy as np
    >>> from pounce.qp import QpSensitivity
    >>> # min ½‖x‖²  s.t.  x0 + x1 = 2   → x* = (1, 1), dx/db = (½, ½)
    >>> s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0],
    ...                   A=[[1.0, 1.0]], b=[2.0])
    >>> dx = s.parametric_step([0], [1.0])     # perturb b0 by +1
    >>> np.round(s.x + dx, 6)
    array([1.5, 1.5])
    """

    def __init__(
        self,
        P=None,
        c=None,
        A=None,
        b=None,
        G=None,
        h=None,
        lb=None,
        ub=None,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
        active_tol: float = 1e-7,
        check_psd: Optional[bool] = None,
    ):
        if c is None:
            raise ValueError("QpSensitivity: `c` is required")
        _validate_solver_opts(tol, max_iter, "QpSensitivity")
        _maybe_check_psd(P, c, check_psd)
        prob = _build(P, c, A, b, G, h, lb, ub)
        self._inner = _pounce.QpSensitivity(
            prob, tol=tol, max_iter=max_iter, active_tol=active_tol
        )

    @property
    def x(self) -> np.ndarray:
        """The optimal primal solution ``x*``."""
        return np.asarray(self._inner.x)

    @property
    def obj(self) -> float:
        """The optimal objective value."""
        return float(self._inner.obj)

    @property
    def kkt_dim(self) -> int:
        """Active-set KKT dimension ``n + m_eq + n_active``.

        Since ``n_active = kkt_dim − n − m_eq``, a change in this value across
        a parameter sweep signals that the active set changed — and so that the
        :meth:`parametric_step` predictor's precondition was crossed somewhere
        in the sweep. :attr:`active_indices` reports the same thing by identity
        rather than by count, and :attr:`weakly_active_indices` catches the
        harder case where the count does not move at all.
        """
        return int(self._inner.kkt_dim)

    @property
    def kkt_cond_estimate(self) -> float:
        """Estimated condition number ``κ₁`` of the active-set KKT system.

        A cheap Hager 1-norm estimate of the conditioning of the (factored)
        sensitivity system. It is the quantitative early-warning that
        :attr:`kkt_dim` and :attr:`weakly_active_indices` cannot give: on a
        *near*-LICQ problem — where the active-constraint gradients are nearly
        (not exactly) rank-deficient — the KKT is near-singular and
        :meth:`parametric_step` can silently over-damp ``dx/db`` toward a
        smooth but badly wrong value (issue #284). A large estimate flags that
        risk.

        Well-conditioned sensitivities report a modest value (a few ``×10⁹``
        even on badly-scaled data); a numerically singular one saturates near
        ``1e16``. See :attr:`ill_conditioned` for the thresholded boolean and
        :attr:`last_step_residual` for the achieved per-step residual.
        """
        return float(self._inner.kkt_cond_estimate)

    @property
    def ill_conditioned(self) -> bool:
        """Whether ``dx/db`` may be unreliable because the KKT is near-singular.

        ``True`` when :attr:`kkt_cond_estimate` exceeds an internal threshold
        (``1e14``) — the regime where the sensitivity system is so near-LICQ
        that even the internal iterative refinement cannot recover ``dx/db``
        from double precision (issue #284). It stays ``False`` on
        well-conditioned problems, including the badly-scaled equality-only and
        active-set cases, so it does not false-alarm.

        Use it as a guard: if ``ill_conditioned`` is ``True``, treat the
        :meth:`parametric_step` result as untrustworthy (or cross-check it),
        rather than consuming the silently-damped value.
        """
        return bool(self._inner.ill_conditioned)

    @property
    def last_step_residual(self) -> Optional[float]:
        """Relative KKT residual of the most recent :meth:`parametric_step`.

        ``‖rhs − K·step‖∞ / (1 + ‖rhs‖∞)`` measured against the *unregularized*
        KKT, or ``None`` before any step has been taken. It reports how well the
        returned step actually satisfies the true sensitivity system (issue
        #284): a round-off-level value means the step is trustworthy; a large
        one means the refinement could not solve the near-singular system and
        the step is unreliable. This is the per-query companion to the
        build-time :attr:`ill_conditioned` / :attr:`kkt_cond_estimate`.
        """
        r = self._inner.last_step_residual
        return None if r is None else float(r)

    @property
    def active_indices(self) -> ActiveSet:
        """Which constraints are in the active set at the optimum.

        The active set is read from the dual certificate, using the
        ``active_tol`` passed to the constructor.
        """
        return ActiveSet(
            inequalities=tuple(self._inner.active_inequalities),
            bounds=tuple(self._inner.active_bounds),
        )

    @property
    def weakly_active_indices(self) -> ActiveSet:
        """Constraints at which **strict complementarity fails**.

        A weakly active constraint is binding in the primal while carrying a
        negligible multiplier. Classical post-optimal sensitivity (Fiacco)
        assumes this does not happen; where it does, the perturbation changes
        the active set and :meth:`parametric_step` returns a genuine *one-sided*
        derivative — the other direction has a different, equally correct value.

        Nothing returned by :meth:`parametric_step` is wrong when this is
        non-empty; both branches are real derivatives. What it means is that the
        predictor should not be assumed to extrapolate in both directions. Probe
        the direction you actually care about.

        This is the check :attr:`kkt_dim` cannot provide. On the QP below the
        two branches of ``dx/db`` differ by 33%, and which one is reported turns
        on the solver's ``tol`` — an unrelated setting. ``kkt_dim`` flips 4 → 3
        across that change while this flag stays on throughout.

        Example
        -------
        >>> import numpy as np
        >>> from pounce.qp import QpSensitivity
        >>> # min ½‖x‖² s.t. x0 + x1 = 1, x0 − 2x1 ≤ −½.
        >>> # The equality-only optimum (½, ½) hits the inequality exactly.
        >>> s = QpSensitivity(P=np.eye(2), c=[0.0, 0.0],
        ...                   A=[[1.0, 1.0]], b=[1.0],
        ...                   G=[[1.0, -2.0]], h=[-0.5])
        >>> s.weakly_active_indices.inequalities
        (0,)
        """
        return ActiveSet(
            inequalities=tuple(self._inner.weakly_active_inequalities),
            bounds=tuple(self._inner.weakly_active_bounds),
        )

    def parametric_step(self, pin_constraint_indices, deltas) -> np.ndarray:
        """First-order primal step ``dx ≈ x*(b + Δb) − x*(b)``.

        Equality constraint ``pin_constraint_indices[k]`` (an index into
        ``b``) is perturbed by ``deltas[k]``; all other data is held fixed.
        Returns the length-``n`` sensitivity, so ``self.x + dx`` predicts
        the perturbed solution (exact to first order while the active set is
        unchanged). The factorization is reused, so a continuation sweep
        costs one back-substitution per query.

        The "while the active set is unchanged" precondition is checkable:
        :attr:`weakly_active_indices` reports where it fails at this optimum,
        which is where ``dx`` is a one-sided derivative rather than the
        derivative. See that property for what to do about it.
        """
        pins = [int(i) for i in pin_constraint_indices]
        ds = [float(d) for d in deltas]
        return np.asarray(self._inner.parametric_step(pins, ds))

    def reduced_hessian(self, rank_tol: float = 1e-9) -> ReducedHessian:
        """Reduced Hessian ``Zᵀ P Z`` on the active manifold + eigendecomp.

        Projects the objective Hessian ``P`` onto the null space of the
        active constraints (equalities, active inequalities, and active
        variable bounds), then eigendecomposes it. The eigenvalues are the
        objective's curvatures along feasible directions — all positive
        confirms a strict (well-conditioned) minimizer. Mirrors the NLP
        ``solve_with_sens(compute_reduced_hessian=True, rh_eigendecomp=True)``.

        ``rank_tol`` is the relative threshold used to determine the rank of
        the active Jacobian (hence the degrees of freedom). The computation
        densifies ``P``, so it is meant for QPs with a modest variable count.
        """
        d = self._inner.reduced_hessian(rank_tol)
        n = int(d["n_dof"])
        # The Rust side returns column-major flat arrays.
        matrix = np.asarray(d["matrix"]).reshape((n, n), order="F")
        eigvecs = np.asarray(d["eigenvectors"]).reshape((n, n), order="F")
        return ReducedHessian(
            n_dof=n,
            matrix=matrix,
            eigenvalues=np.asarray(d["eigenvalues"]),
            eigenvectors=eigvecs,
        )
