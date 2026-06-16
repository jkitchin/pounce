"""Build a POUNCE problem object from a GAMS Modeling Object (GMO) view.

POUNCE is a *local* NLP solver (an Ipopt-style interior-point method), so it
only ever needs function / gradient / Hessian **values** at a point — never the
symbolic expression DAG.  GMO evaluates those values directly
(``gmoEvalFuncObj`` / ``gmoEvalGradObj`` / ``gmoEvalFunc`` / ``gmoEvalGrad`` /
``gmoHessLagValue``), so the translation here is far simpler than a global
solver's: it wires GMO's numerical evaluators straight into POUNCE's
cyipopt-compatible :class:`pounce.Problem` callbacks.

Everything is read through the small :class:`GmoView` protocol below.  Keeping
the interface this thin lets the whole translate -> build -> solve path be
unit-tested with an in-memory fake (no GAMS license required); the only
GAMS-library-specific code lives in the adapter in :mod:`pounce.gams.link`.

The sign / objective conventions mirror the C link ``gams/gams_pounce.c``
exactly:

* ``obj_sign = -1`` for a GAMS *maximize* model, ``+1`` otherwise.  POUNCE
  always minimizes, so the objective and its gradient are scaled by ``obj_sign``
  (maximize -> minimize ``-f``).
* The Hessian of the Lagrangian is obtained from a single ``gmoHessLagValue``
  call with ``objweight = obj_sign * obj_factor`` and ``conweight = -1.0``.  The
  ``conweight = -1`` is equivalent to negating the multipliers (GAMS's ``pi`` is
  ``-lambda``), so POUNCE's standard cyipopt ``lagrange`` vector is passed
  through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

# POUNCE / GAMS "infinity": bounds at or beyond GAMS's +/- inf are mapped to
# +/- this value, matching gams_pounce.c.
POUNCE_INF = 1e19


class GmoView(Protocol):
    """Minimal numerical read interface over a GAMS Modeling Object.

    All indexing is 0-based (the adapter sets ``gmoIndexBaseSet(0)``).  The
    evaluators work in the model's *native* objective sense; the sign flip for
    maximization is applied by :func:`problem_from_gmo`, not by the view.
    """

    # --- dimensions -------------------------------------------------------
    def name(self) -> str: ...
    def num_vars(self) -> int: ...
    def num_cons(self) -> int: ...

    # True for a GAMS ``maximizing`` model, False for ``minimizing``.
    def maximize(self) -> bool: ...

    # True if GMO can supply an analytical Hessian of the Lagrangian.
    def has_hessian(self) -> bool: ...

    # --- bounds / initial point ------------------------------------------
    # Each returns a length-``num_vars`` (or ``num_cons``) sequence with GAMS
    # infinities already mapped to +/- POUNCE_INF.
    def var_lower(self) -> Sequence[float]: ...
    def var_upper(self) -> Sequence[float]: ...
    def var_init(self) -> Sequence[float]: ...
    def con_lower(self) -> Sequence[float]: ...
    def con_upper(self) -> Sequence[float]: ...

    # --- sparsity structure (0-based COO) --------------------------------
    # Constraint Jacobian nonzeros, in the order ``eval_jac`` returns values.
    def jac_structure(self) -> tuple[Sequence[int], Sequence[int]]: ...
    # Lower-triangle Hessian-of-Lagrangian nonzeros (only when has_hessian()).
    def hess_structure(self) -> tuple[Sequence[int], Sequence[int]]: ...

    # --- numerical evaluators (native sense) -----------------------------
    def eval_obj(self, x: np.ndarray) -> float: ...
    def eval_grad_obj(self, x: np.ndarray) -> Sequence[float]: ...
    def eval_cons(self, x: np.ndarray) -> Sequence[float]: ...
    def eval_jac(self, x: np.ndarray) -> Sequence[float]: ...

    def hess_lag_value(
        self,
        x: np.ndarray,
        lam: np.ndarray,
        obj_weight: float,
        con_weight: float,
    ) -> Sequence[float]: ...


@dataclass
class GmoProblem:
    """Everything needed to construct and solve a :class:`pounce.Problem`.

    ``problem_obj`` carries the cyipopt callbacks; the remaining fields are the
    constructor arguments and a couple of conveniences for the link.
    """

    problem_obj: "_GmoProblemObj"
    n: int
    m: int
    lb: list[float]
    ub: list[float]
    cl: list[float]
    cu: list[float]
    x0: np.ndarray
    has_hessian: bool
    obj_sign: float  # +1 minimize, -1 maximize


class _GmoProblemObj:
    """cyipopt-style problem object delegating to a :class:`GmoView`.

    Mirrors the callback set POUNCE expects (see ``pounce.Problem``):
    ``objective``, ``gradient``, ``constraints``, ``jacobianstructure`` and
    ``jacobian``.  When an analytical Hessian is available the
    :class:`_GmoProblemObjHess` subclass adds ``hessianstructure`` /
    ``hessian``; POUNCE detects the Hessian callbacks by their presence
    (``hasattr``), so the no-Hessian object must simply not define them, which
    selects limited-memory (L-BFGS) mode exactly as the C link does.  The
    objective-sense and Hessian-weight conventions match ``gams_pounce.c``.
    """

    def __init__(self, view: GmoView, obj_sign: float):
        self._v = view
        self._sign = obj_sign
        jr, jc = view.jac_structure()
        self._jac_rows = np.asarray(jr, dtype=np.int64)
        self._jac_cols = np.asarray(jc, dtype=np.int64)

    # --- objective --------------------------------------------------------
    def objective(self, x):
        return self._sign * float(self._v.eval_obj(np.asarray(x, dtype=float)))

    def gradient(self, x):
        g = np.asarray(self._v.eval_grad_obj(np.asarray(x, dtype=float)), dtype=float)
        return self._sign * g

    # --- constraints ------------------------------------------------------
    def constraints(self, x):
        return np.asarray(self._v.eval_cons(np.asarray(x, dtype=float)), dtype=float)

    def jacobianstructure(self):
        return self._jac_rows, self._jac_cols

    def jacobian(self, x):
        return np.asarray(self._v.eval_jac(np.asarray(x, dtype=float)), dtype=float)


class _GmoProblemObjHess(_GmoProblemObj):
    """:class:`_GmoProblemObj` plus the analytical Hessian of the Lagrangian."""

    def __init__(self, view: GmoView, obj_sign: float):
        super().__init__(view, obj_sign)
        hr, hc = view.hess_structure()
        self._hess_rows = np.asarray(hr, dtype=np.int64)
        self._hess_cols = np.asarray(hc, dtype=np.int64)

    def hessianstructure(self):
        return self._hess_rows, self._hess_cols

    def hessian(self, x, lagrange, obj_factor):
        # gmoHessLagValue computes  objweight * d2f + conweight * sum_i pi_i d2c_i.
        # objweight = obj_sign*obj_factor (negate objective Hessian for max);
        # conweight = -1.0 absorbs the GAMS pi = -lambda sign flip, so the
        # standard cyipopt `lagrange` is passed straight through.
        vals = self._v.hess_lag_value(
            np.asarray(x, dtype=float),
            np.asarray(lagrange, dtype=float),
            self._sign * float(obj_factor),
            -1.0,
        )
        return np.asarray(vals, dtype=float)


def problem_from_gmo(view: GmoView) -> GmoProblem:
    """Translate a :class:`GmoView` into a :class:`GmoProblem`.

    The returned ``problem_obj`` exposes ``hessianstructure`` / ``hessian`` only
    when ``view.has_hessian()`` is true; otherwise those callbacks are absent so
    POUNCE falls back to a limited-memory (L-BFGS) Hessian approximation,
    exactly as the C link does.
    """
    n = int(view.num_vars())
    m = int(view.num_cons())
    obj_sign = -1.0 if view.maximize() else 1.0

    lb = [float(v) for v in view.var_lower()]
    ub = [float(v) for v in view.var_upper()]
    cl = [float(v) for v in view.con_lower()] if m else []
    cu = [float(v) for v in view.con_upper()] if m else []
    x0 = np.asarray(view.var_init(), dtype=float)

    has_hess = bool(view.has_hessian())
    obj = (
        _GmoProblemObjHess(view, obj_sign)
        if has_hess
        else _GmoProblemObj(view, obj_sign)
    )

    return GmoProblem(
        problem_obj=obj,
        n=n,
        m=m,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
        x0=x0,
        has_hessian=has_hess,
        obj_sign=obj_sign,
    )
