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

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import _pounce

__all__ = [
    "QpResult",
    "QpFactorization",
    "QpSensitivity",
    "ReducedHessian",
    "solve_qp",
    "solve_socp",
    "solve_qp_batch",
    "solve_qp_multi_rhs",
]


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
        ``None`` for conic (:func:`solve_socp`) solves, where the slack
        lives in a non-orthant cone and these orthant residuals do not
        apply.
    iterates:
        Per-iteration convergence trace — a list of dicts with keys
        ``iter``, ``objective``, ``primal_infeasibility``,
        ``dual_infeasibility``, ``mu``, ``alpha_primal``, ``alpha_dual``.
        Empty unless the solve was called with ``collect_iterates=True``.
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

    @property
    def success(self) -> bool:
        return self.status == "optimal"

    @property
    def kkt_error(self) -> Optional[float]:
        """Overall KKT error (max residual), or ``None`` for conic solves."""
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
) -> QpResult:
    """Solve one convex QP. See the module docstring for the form.

    ``P`` (lower triangle is used; assumed symmetric) and ``A``/``G`` may
    be scipy-sparse or dense; ``None`` matrices are empty. ``c`` is
    required and sets ``n``.

    ``warm_start`` (optional) is a previous :class:`QpResult` (or a mapping
    with ``x``/``y``/``z``/``z_lb``/``z_ub``) for a *nearby* problem. It
    seeds the interior-point iteration to reduce the iteration count; it
    does not change the solution, and a dimension mismatch is ignored.

    The returned :class:`QpResult` carries the final KKT ``residuals``;
    pass ``collect_iterates=True`` to also capture the per-iteration
    convergence trace in ``result.iterates``.
    """
    if c is None:
        raise ValueError("solve_qp: `c` is required")
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
    ``("pow", 0.5)`` tuples, or the shorthand ``3`` (a second-order cone of
    that dim). Kind strings are case-insensitive (``"soc"``/``"q"``,
    ``"nonneg"``/``"nn"``/``"+"``, ``"exp"``/``"exponential"``,
    ``"pow"``/``"power"``). The second element is the dimension for
    ``soc``/``nonneg`` and the exponent ``α`` for ``pow``."""
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

    A second-order cone may be freely mixed with an exp/power cone (the
    non-symmetric driver handles both).

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
) -> list[QpResult]:
    """Solve a batch of convex QPs in parallel (across instances).

    ``problems`` is a sequence of kwarg dicts, each accepted by
    :func:`solve_qp` (keys ``P, c, A, b, G, h, lb, ub``). Returns one
    :class:`QpResult` per input, in order.

    ``warm_starts`` (optional) is a sequence — one per problem — of prior
    :class:`QpResult`\\ s or mappings (for a sequence of nearby batches).
    Each seeds its instance's iteration; mismatched entries are ignored.
    """
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
    n = len(np.asarray(cs[0], dtype=np.float64).ravel())
    # `c` only fixes `n` for the base structure; the real objectives are `cs`.
    base_c = c if c is not None else np.zeros(n)
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
    ):
        if c is None:
            raise ValueError("QpFactorization: `c` is required (representative problem)")
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
    ):
        if c is None:
            raise ValueError("QpSensitivity: `c` is required")
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
        """Active-set KKT dimension ``n + m_eq + n_active``."""
        return int(self._inner.kkt_dim)

    def parametric_step(self, pin_constraint_indices, deltas) -> np.ndarray:
        """First-order primal step ``dx ≈ x*(b + Δb) − x*(b)``.

        Equality constraint ``pin_constraint_indices[k]`` (an index into
        ``b``) is perturbed by ``deltas[k]``; all other data is held fixed.
        Returns the length-``n`` sensitivity, so ``self.x + dx`` predicts
        the perturbed solution (exact to first order while the active set is
        unchanged). The factorization is reused, so a continuation sweep
        costs one back-substitution per query.
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
