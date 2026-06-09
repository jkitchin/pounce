"""scipy.optimize.minimize-style facade over pounce.Problem.

Thin wrapper that adapts SciPy conventions (functional ``fun``, ``jac``,
``hess``; bound list of ``(lo, hi)`` pairs; constraint dicts with
``'type': 'eq'|'ineq'``) into a cyipopt/pounce-style Problem.

Notes
-----
* When ``jac`` is omitted we fall back to **central** finite differences
  (step ``eps**(1/3)``) and emit a one-time ``UserWarning`` naming the
  remedies. Central differences have an ``O(h^2)`` truncation error whose
  noise floor sits well below the tight default tolerance, so the solve
  converges cleanly instead of stalling just short of it (gh #123).
  Production callers should still provide an analytic Jacobian (or use the
  autodiff frontends ``pounce.jax`` / ``pounce.torch``).
* When ``hess`` is omitted, or when constraints are present, the solver
  is driven with ``hessian_approximation = limited-memory``.
* Equality / inequality dicts are concatenated into a single ``g(x)``
  with bound vectors ``cl`` / ``cu``. Constraint Jacobian is dense by
  design — sparse Jacobians belong on the :class:`Problem` API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

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


@dataclass
class OptimizeResult:
    """SciPy-OptimizeResult-shaped solve result."""

    x: np.ndarray
    fun: float
    success: bool
    status: int
    message: str
    nit: int
    info: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, k: str) -> Any:
        if hasattr(self, k):
            return getattr(self, k)
        return self.info[k]


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
    if bounds is None:
        return None, None
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


def _wrap_constraints(constraints, n: int):
    """Coalesce scipy-style constraint dict(s) into one g(x) + (cl, cu)."""
    if not constraints:
        return 0, None, None, None, None
    if isinstance(constraints, dict):
        constraints = [constraints]

    funs, jacs = [], []
    for c in constraints:
        if not isinstance(c, dict):
            raise ValueError(
                f"each constraint must be a dict with 'type' and 'fun', got "
                f"{type(c).__name__}"
            )
        if "type" not in c or "fun" not in c:
            missing = sorted({"type", "fun"} - set(c))
            raise ValueError(
                f"constraint dict is missing required key(s) {missing}; a "
                f"constraint needs {{'type': 'eq'|'ineq', 'fun': callable}}"
            )
        kind = c["type"]
        if kind not in ("eq", "ineq"):
            raise ValueError(f"unknown constraint type {kind!r}; use 'eq' or 'ineq'")
        if not callable(c["fun"]):
            raise ValueError("constraint 'fun' must be callable")
        funs.append(c["fun"])
        jacs.append(c.get("jac"))

    probe = np.zeros(n)
    sizes = [int(_to_array(fn(probe)).size) for fn in funs]
    m_total = int(sum(sizes))

    def g_combined(x):
        return np.concatenate([_to_array(fn(x)).ravel() for fn in funs])

    def jac_combined(x):
        rows = []
        for fn, jc in zip(funs, jacs):
            if jc is not None:
                rows.append(np.atleast_2d(_to_array(jc(x))))
            else:
                m_i = _to_array(fn(x)).size
                rows.append(_finite_diff_jac(fn, x, m_i))
        return np.vstack(rows)

    cl = np.empty(m_total)
    cu = np.empty(m_total)
    off = 0
    for sz, c in zip(sizes, constraints):
        if c["type"] == "eq":
            cl[off : off + sz] = 0.0
            cu[off : off + sz] = 0.0
        else:  # ineq: g(x) >= 0
            cl[off : off + sz] = 0.0
            cu[off : off + sz] = float(np.inf)
        off += sz

    return m_total, g_combined, jac_combined, cl, cu


def _build_problem_obj(
    *,
    fun: Callable,
    n: int,
    m: int,
    jac: Callable | None,
    hess: Callable | None,
    g: Callable | None,
    jac_g: Callable | None,
):
    """Build a problem-object-with-methods on the fly. Only attaches
    ``hessian`` / ``hessianstructure`` when ``hess`` is provided so
    Problem's ``hasattr`` probe correctly falls back to L-BFGS."""

    members: dict[str, Any] = {}

    def objective(self, x):
        return float(fun(x))

    def gradient(self, x):
        if jac is None:
            return _finite_diff_grad(fun, x)
        return _to_array(jac(x)).ravel()

    members["objective"] = objective
    members["gradient"] = gradient

    if m > 0:

        def constraints(self, x):
            return _to_array(g(x)).ravel()

        def jacobianstructure(self):
            rows = np.repeat(np.arange(m), n)
            cols = np.tile(np.arange(n), m)
            return (rows, cols)

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
            H = obj_factor * _to_array(hess(x))
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
    if isinstance(constraints, dict):
        constraints = [constraints]
    return any(
        isinstance(c, dict) and c.get("jac") is None for c in constraints
    )


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    jac: Callable | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """scipy.optimize.minimize-style facade over pounce.

    Solver routing mirrors the CLI's ``solver_selection``. By default
    (``options={"solver_selection": "auto"}``) the problem is probed for
    structure: a linear or convex-quadratic objective with only linear
    constraints is dispatched to the specialized convex LP/QP interior-point
    solver (``pounce.solve_qp``), a convex-quadratic objective/constraints
    problem (a convex QCQP) is reformulated to a second-order cone program and
    dispatched to the conic interior-point solver (``pounce.solve_socp``), and
    everything else falls through to the general NLP filter-IPM. Detection is
    conservative and validated against the true callables at held-out points,
    so a nonlinear problem is never silently sent to the convex solver.
    Override with ``"solver_selection"``:

    * ``"auto"`` (default) — route LP/convex-QP to the convex QP solver, a
      convex QCQP to the conic solver, else NLP;
    * ``"nlp"`` — always use the NLP solver (the pre-routing behavior);
    * ``"lp-ipm"`` / ``"qp-ipm"`` — force the convex QP solver, raising
      ``ValueError`` if the problem is not detected as an LP / convex QP;
    * ``"socp"`` — force the conic solver, raising ``ValueError`` if the
      problem is not detected as a convex QCQP.

    Like :func:`scipy.optimize.minimize`, this facade is **silent by default**.
    Pass ``options={"disp": True}`` for a concise log or an explicit
    ``options={"print_level": N}`` (0–12) to control the NLP backend's IPM
    iteration table directly.
    """
    # Promote a scalar / 0-d x0 to 1-D, matching scipy.optimize.minimize, so a
    # single-variable problem can be written ``minimize(f, 1.5)``.
    x0 = np.atleast_1d(_to_array(x0))
    n = x0.size
    lb, ub = _normalize_bounds(bounds, n)
    m, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

    # Solver routing (mirrors the CLI's `solver_selection`). Pop the routing
    # keys so the remainder of `options` still flows to the NLP solver.
    opts = dict(options) if options else {}
    selection = str(opts.pop("solver_selection", "auto")).lower()
    route_tol = float(opts.pop("route_tol", 1e-5))
    # scipy.optimize.minimize is silent unless `disp=True`; match that. pounce's
    # NLP backend otherwise prints a full IPM iteration table by default (and the
    # log is written from Rust to fd 1, so Python stdout redirection can't catch
    # it). Default print_level to 0 (silent) unless the caller passes an explicit
    # print_level or scipy-style disp=True. (#115)
    disp = bool(opts.pop("disp", False))
    opts.setdefault("print_level", 5 if disp else 0)
    route_kw = dict(
        fun=fun, jac=jac, hess=hess, lb=lb, ub=ub, m=m,
        g_combined=g_combined, jac_combined=jac_combined,
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
            return _solve_via_convex(extract, opts)
        # Auto: an LP/QP wasn't found — try a convex QCQP before giving up to
        # the NLP solver (a quadratic *constraint* lands here, not above).
        if selection == "auto":
            socp = classify_and_extract_socp(**route_kw)
            if socp is not None:
                return _solve_via_socp(socp, opts)
    elif selection == "socp":
        socp = classify_and_extract_socp(**route_kw)
        if socp is None:
            raise ValueError(
                "solver_selection='socp' but the problem was not detected as a "
                "convex QCQP (convex-quadratic objective and/or constraints, all "
                "convex, with only linear equalities)"
            )
        return _solve_via_socp(socp, opts)

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
        jac=jac,
        hess=hess,
        g=g_combined,
        jac_g=jac_combined,
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
    # `opts` is `options` minus the routing keys (`solver_selection`,
    # `route_tol`), so only genuine solver options reach the NLP backend.
    for k, v in opts.items():
        problem.add_option(k, v)

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
    acceptable_tol = float(opts.get("acceptable_tol", _DEFAULT_ACCEPTABLE_TOL))
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
        info=dict(info),
    )
