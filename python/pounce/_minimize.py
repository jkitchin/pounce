"""scipy.optimize.minimize-style facade over pounce.Problem.

Thin wrapper that adapts SciPy conventions (functional ``fun``, ``jac``,
``hess``; bound list of ``(lo, hi)`` pairs; constraint dicts with
``'type': 'eq'|'ineq'``) into a cyipopt/pounce-style Problem.

Notes
-----
* When ``jac`` is omitted we fall back to forward finite differences
  (step ``sqrt(eps)``). Production callers should provide a Jacobian.
* When ``hess`` is omitted, or when constraints are present, the solver
  is driven with ``hessian_approximation = limited-memory``.
* Equality / inequality dicts are concatenated into a single ``g(x)``
  with bound vectors ``cl`` / ``cu``. Constraint Jacobian is dense by
  design — sparse Jacobians belong on the :class:`Problem` API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._pounce import Problem

_EPS = float(np.finfo(np.float64).eps) ** 0.5


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
    f0 = float(fun(x))
    g = np.empty_like(x)
    for i in range(x.size):
        h = _EPS * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        g[i] = (float(fun(xp)) - f0) / h
    return g


def _finite_diff_jac(g_fun: Callable, x: np.ndarray, m: int) -> np.ndarray:
    g0 = _to_array(g_fun(x))
    J = np.empty((m, x.size))
    for i in range(x.size):
        h = _EPS * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        J[:, i] = (_to_array(g_fun(xp)) - g0) / h
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


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    jac: Callable | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """scipy.optimize.minimize-style facade over pounce."""
    # Promote a scalar / 0-d x0 to 1-D, matching scipy.optimize.minimize, so a
    # single-variable problem can be written ``minimize(f, 1.5)``.
    x0 = np.atleast_1d(_to_array(x0))
    n = x0.size
    lb, ub = _normalize_bounds(bounds, n)
    m, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

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
    if options:
        for k, v in options.items():
            problem.add_option(k, v)

    x, info = problem.solve(x0=x0)
    return OptimizeResult(
        x=np.asarray(x),
        fun=float(info["obj_val"]),
        success=int(info["status"]) == 0,
        status=int(info["status"]),
        message=str(info["status_msg"]),
        nit=int(info["iter_count"]),
        info=dict(info),
    )
