"""scipy.optimize.minimize-style facade over pounce.Problem.

Thin wrapper that adapts SciPy conventions (functional ``fun``, ``jac``,
``hess``; bound list of ``(lo, hi)`` pairs; constraint dicts with
``'type': 'eq'|'ineq'``) into a cyipopt/pounce-style Problem.

Notes
-----
* When ``jac`` is omitted (or ``False``) we fall back to forward finite
  differences (step ``sqrt(eps)``). Production callers should provide
  a Jacobian.
* When ``jac=True``, ``fun(x, *args)`` must return
  ``(f, grad)``; the pair is cached so each Ipopt iterate triggers only
  one forward pass.
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
    g0 = _to_array(g_fun(x))
    J = np.empty((m, x.size))
    for i in range(x.size):
        h = _EPS * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += h
        J[:, i] = (_to_array(g_fun(xp)) - g0) / h
    return J


def _normalize_bounds(bounds, n: int):
    if bounds is None:
        return None, None
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
    return lb, ub


def _wrap_constraints(constraints, n: int):
    """Coalesce scipy-style constraint dict(s) into one g(x) + (cl, cu)."""
    if not constraints:
        return 0, None, None, None, None
    if isinstance(constraints, dict):
        constraints = [constraints]

    funs, jacs, cargs = [], [], []
    for c in constraints:
        kind = c["type"]
        if kind not in ("eq", "ineq"):
            raise ValueError(f"unknown constraint type {kind!r}")
        funs.append(c["fun"])
        jacs.append(c.get("jac"))
        cargs.append(tuple(c.get("args", ())))

    probe = np.zeros(n)
    sizes = [int(_to_array(fn(probe, *ca)).size) for fn, ca in zip(funs, cargs)]
    m_total = int(sum(sizes))

    def g_combined(x):
        return np.concatenate(
            [_to_array(fn(x, *ca)).ravel() for fn, ca in zip(funs, cargs)]
        )

    def jac_combined(x):
        rows = []
        for fn, jc, ca in zip(funs, jacs, cargs):
            if jc is not None:
                rows.append(np.atleast_2d(_to_array(jc(x, *ca))))
            else:
                m_i = _to_array(fn(x, *ca)).size
                rows.append(
                    _finite_diff_jac(lambda xx, fn=fn, ca=ca: fn(xx, *ca), x, m_i)
                )
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
    args: tuple,
    jac: Callable | bool | None,
    hess: Callable | None,
    g: Callable | None,
    jac_g: Callable | None,
    callback: Callable | None,
):
    """Build a problem-object-with-methods on the fly. Only attaches
    ``hessian`` / ``hessianstructure`` when ``hess`` is provided so
    Problem's ``hasattr`` probe correctly falls back to L-BFGS. Likewise,
    ``intermediate`` is only attached when ``callback`` is provided so the
    no-callback case has zero per-iter Python overhead."""

    members: dict[str, Any] = {}
    xcache = _LastXCache()

    if jac is True:
        cache = _FunAndGradCache(fun, args)

        def objective(self, x, _c=cache, _xc=xcache):
            _xc.remember(x)
            return _c.f(x)

        def gradient(self, x, _c=cache):
            return _c.g(x)
    else:

        def objective(self, x, _xc=xcache):
            _xc.remember(x)
            return float(fun(x, *args))

        def gradient(self, x):
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
            H = obj_factor * _to_array(hess(x, *args))
            r, c = np.tril_indices(n)
            return H[r, c]

        members["hessianstructure"] = hessianstructure
        members["hessian"] = hessian

    cls = type("_MinimizeProblem", (object,), members)
    return cls()


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | bool | None = None,
    hess: Callable | None = None,
    bounds: Sequence | None = None,
    constraints: Sequence | dict | None = None,
    callback: Callable | None = None,
    options: Mapping[str, Any] | None = None,
) -> OptimizeResult:
    """scipy.optimize.minimize-style facade over pounce."""
    x0 = _to_array(x0)
    n = x0.size
    lb, ub = _normalize_bounds(bounds, n)
    m, g_combined, jac_combined, cl, cu = _wrap_constraints(constraints, n)

    problem_obj = _build_problem_obj(
        fun=fun,
        n=n,
        m=m,
        args=args,
        jac=jac,
        hess=hess,
        g=g_combined,
        jac_g=jac_combined,
        callback=callback,
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
