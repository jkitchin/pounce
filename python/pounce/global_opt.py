"""Deterministic global optimization of factorable nonconvex problems.

Minimize a nonconvex objective subject to nonconvex constraints over a box, to a
*certified* global optimum (a feasible point plus a proven optimality gap), via
spatial branch-and-bound. Unlike :func:`pounce.minimize` (which finds a local
optimum) this returns the global one for the supported function class:
``+ - * /``, integer powers, ``sqrt``, ``exp``, ``log``, ``abs``, ``sin``,
``cos``.

Build expressions with :func:`var` and the usual Python operators / methods::

    from pounce.global_opt import var, minimize_global, ge

    # Six-hump camel — six local minima, global value ≈ −1.0316.
    x, y = var(0), var(1)
    f = (4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
    r = minimize_global(f, lo=[-2, -1.5], hi=[2, 1.5])
    r.objective        # ≈ −1.0316  (certified global minimum)
    r.x                # the global minimizer

    # min x + y  s.t.  x·y ≥ 4 on [1,5]²  → 4 at (2, 2)
    g = var(0) * var(1)
    r = minimize_global(var(0) + var(1), constraints=[ge(g, 4.0)], lo=[1, 1], hi=[5, 5])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from . import _pounce

__all__ = ["Expr", "var", "con", "le", "ge", "eq", "minimize_global", "GlobalResult"]

# Op tags — must match crates/pounce-py/src/global_opt.rs.
_CONST, _VAR, _ADD, _SUB, _MUL, _DIV, _POW = 0, 1, 2, 3, 4, 5, 6
_NEG, _SQRT, _EXP, _LN, _ABS, _SIN, _COS = 7, 8, 9, 10, 11, 12, 13


class Expr:
    """A factorable expression over the problem variables, built with operator
    overloading and compiled to a flat op tape for the solver."""

    __slots__ = ("tag", "a", "b", "c")

    def __init__(self, tag: int, a=None, b=None, c: float = 0.0):
        self.tag = tag
        self.a = a  # child Expr, or (for Var) the int index
        self.b = b  # child Expr for binary ops
        self.c = c  # constant value / integer exponent

    # -- construction helpers --
    def powi(self, n: int) -> "Expr":
        return self.__pow__(n)

    def exp(self) -> "Expr":
        return Expr(_EXP, self)

    def log(self) -> "Expr":
        return Expr(_LN, self)

    def sqrt(self) -> "Expr":
        return Expr(_SQRT, self)

    def sin(self) -> "Expr":
        return Expr(_SIN, self)

    def cos(self) -> "Expr":
        return Expr(_COS, self)

    # -- operators --
    def __add__(self, o):
        return Expr(_ADD, self, _coerce(o))

    def __radd__(self, o):
        return Expr(_ADD, _coerce(o), self)

    def __sub__(self, o):
        return Expr(_SUB, self, _coerce(o))

    def __rsub__(self, o):
        return Expr(_SUB, _coerce(o), self)

    def __mul__(self, o):
        return Expr(_MUL, self, _coerce(o))

    def __rmul__(self, o):
        return Expr(_MUL, _coerce(o), self)

    def __truediv__(self, o):
        return Expr(_DIV, self, _coerce(o))

    def __rtruediv__(self, o):
        return Expr(_DIV, _coerce(o), self)

    def __neg__(self):
        return Expr(_NEG, self)

    def __abs__(self):
        return Expr(_ABS, self)

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("only non-negative integer powers are supported")
        return Expr(_POW, self, c=float(n))

    def to_ops(self) -> list:
        """Compile to a flat ``[(tag, a, b, c), …]`` tape (operands are indices
        into earlier entries)."""
        ops: list = []
        self._emit(ops)
        return ops

    def _emit(self, ops: list) -> int:
        if self.tag == _CONST:
            ops.append((_CONST, -1, -1, float(self.c)))
        elif self.tag == _VAR:
            ops.append((_VAR, int(self.a), -1, 0.0))
        elif self.tag in (_ADD, _SUB, _MUL, _DIV):
            ai = self.a._emit(ops)
            bi = self.b._emit(ops)
            ops.append((self.tag, ai, bi, 0.0))
        elif self.tag == _POW:
            ai = self.a._emit(ops)
            ops.append((_POW, ai, -1, float(self.c)))
        else:  # unary 7..13
            ai = self.a._emit(ops)
            ops.append((self.tag, ai, -1, 0.0))
        return len(ops) - 1


def _coerce(v) -> Expr:
    return v if isinstance(v, Expr) else Expr(_CONST, c=float(v))


def var(i: int) -> Expr:
    """The expression for problem variable ``i`` (0-based)."""
    return Expr(_VAR, a=int(i))


def con(v: float) -> Expr:
    """A constant expression."""
    return Expr(_CONST, c=float(v))


# Constraint builders → (expr, lo, hi).
def le(g: Expr, ub: float) -> Tuple[Expr, float, float]:
    """`g(x) ≤ ub`."""
    return (g, -math.inf, float(ub))


def ge(g: Expr, lb: float) -> Tuple[Expr, float, float]:
    """`g(x) ≥ lb`."""
    return (g, float(lb), math.inf)


def eq(g: Expr, rhs: float) -> Tuple[Expr, float, float]:
    """`g(x) = rhs`."""
    return (g, float(rhs), float(rhs))


@dataclass
class GlobalResult:
    """Result of a global solve.

    Attributes
    ----------
    status:
        ``"optimal"`` (gap within tolerance), ``"infeasible"`` (feasible set
        proven empty), or ``"node_limit"`` (budget exhausted — ``x`` is the best
        found and ``[lower_bound, objective]`` brackets the global optimum).
    x:
        Best feasible point found (the global minimizer when ``optimal``).
    objective:
        Objective at ``x`` — a valid global **upper** bound.
    lower_bound:
        Certified global **lower** bound.
    gap:
        ``objective − lower_bound``.
    nodes:
        Branch-and-bound nodes processed.
    """

    status: str
    x: np.ndarray
    objective: float
    lower_bound: float
    gap: float
    nodes: int

    @property
    def success(self) -> bool:
        return self.status == "optimal"


def minimize_global(
    objective: Expr,
    *,
    constraints: Iterable[Tuple[Expr, float, float]] = (),
    lo: Sequence[float],
    hi: Sequence[float],
    abs_gap: float = 1e-6,
    rel_gap: float = 1e-6,
    feas_tol: float = 1e-6,
    box_tol: float = 1e-7,
    max_nodes: int = 5000,
    local_solve_iters: int = 50,
    sandwich_rounds: int = 4,
    obbt_passes: int = 2,
    alphabb_cuts: int = 1,
    rlt: bool = True,
    multilinear: bool = True,
    threads: int = 1,
) -> GlobalResult:
    """Globally minimize ``objective`` over the box ``[lo, hi]`` subject to the
    given ``constraints`` (each an ``(expr, lo, hi)`` tuple — see :func:`le`,
    :func:`ge`, :func:`eq`).

    The keyword options mirror the Rust ``GlobalOptions``: gap tolerances, the
    per-node relaxation knobs (``obbt_passes``, ``sandwich_rounds``,
    ``alphabb_cuts``, ``rlt``, ``multilinear``), the local-NLP upper-bound
    iteration cap, the node budget, and ``threads`` (``> 1`` runs the parallel
    node pool — faster but non-deterministic in node order; the certified
    optimum is unchanged). Returns a :class:`GlobalResult`.
    """
    lo = [float(v) for v in lo]
    hi = [float(v) for v in hi]
    if len(lo) != len(hi):
        raise ValueError("lo and hi must have the same length")
    n_vars = len(lo)
    cons = [(g.to_ops(), float(clo), float(chi)) for (g, clo, chi) in constraints]

    d = _pounce.solve_global(
        n_vars,
        lo,
        hi,
        objective.to_ops(),
        cons,
        abs_gap=abs_gap,
        rel_gap=rel_gap,
        feas_tol=feas_tol,
        box_tol=box_tol,
        max_nodes=max_nodes,
        local_solve_iters=local_solve_iters,
        sandwich_rounds=sandwich_rounds,
        obbt_passes=obbt_passes,
        alphabb_cuts=alphabb_cuts,
        rlt=rlt,
        multilinear=multilinear,
        threads=threads,
    )
    return GlobalResult(
        status=d["status"],
        x=np.asarray(d["x"]),
        objective=float(d["objective"]),
        lower_bound=float(d["lower_bound"]),
        gap=float(d["gap"]),
        nodes=int(d["nodes"]),
    )
