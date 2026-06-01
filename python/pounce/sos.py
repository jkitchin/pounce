"""Polynomial global optimization via sum-of-squares (SOS / Lasserre).

Globally minimize a polynomial — optionally subject to polynomial
inequality/equality constraints — over the SDP solver. Returns a certified
global lower bound and, when the relaxation is exact (the moment matrix is
flat), the global minimizer(s).

Polynomials are written as dicts mapping an **exponent tuple** to its
coefficient. Over variables ``(x, y)`` the term ``3·x²y`` is ``(2, 1): 3.0``;
a constant is the all-zeros key. For example ``x⁴ − 2x² + 3`` over one
variable is ``{(4,): 1.0, (2,): -2.0, (0,): 3.0}``.

Example
-------
>>> from pounce.sos import sos_minimize
>>> r = sos_minimize({(4,): 1.0, (2,): -2.0, (0,): 3.0})  # x⁴ − 2x² + 3
>>> round(r.lower_bound, 6)
2.0
>>> r.is_exact, r.num_minimizers          # two global minimizers, x = ±1
(True, 2)
>>> # min −x  s.t.  1 − x² ≥ 0   (x ∈ [−1, 1])  →  −1 at x = 1
>>> r = sos_minimize({(1,): -1.0}, inequalities=[{(0,): 1.0, (2,): -1.0}])
>>> round(r.lower_bound, 6)
-1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import _pounce

__all__ = ["sos_minimize", "SosResult"]


@dataclass
class SosResult:
    """Result of an SOS/Lasserre solve.

    Attributes
    ----------
    lower_bound:
        Certified global lower bound ``γ* ≤ min p`` (the global minimum when
        ``is_exact``).
    status:
        Underlying SDP solve status (``"optimal"`` on success).
    is_exact:
        ``True`` when the moment matrix is flat — a *sufficient* certificate
        that ``lower_bound`` is the global minimum. Non-unique optima (which an
        interior-point solver would otherwise return at inflated rank) are
        handled by a facial-reduction re-solve, so all global minimizers are
        recovered in that case too. It can still be ``False`` — e.g. when the
        relaxation order is too low for flatness, or the relaxation is not
        exact — but ``lower_bound`` is a valid lower bound either way.
    num_minimizers:
        Number of global minimizers detected (the flat moment-matrix rank).
    minimizers:
        The extracted global minimizers, each a length-``n_vars`` array.
        Populated when ``is_exact``.
    """

    lower_bound: float
    status: str
    is_exact: bool
    num_minimizers: int
    minimizers: list

    @property
    def success(self) -> bool:
        return self.status == "optimal"


def _terms(poly, n_vars: int, what: str):
    """Normalize a polynomial (dict ``{exp_tuple: coeff}`` or an iterable of
    ``(exp_tuple, coeff)``) into the binding's ``[(list[int], float), …]``."""
    items = poly.items() if hasattr(poly, "items") else poly
    out = []
    for exps, coef in items:
        exps = tuple(int(e) for e in exps)
        if len(exps) != n_vars:
            raise ValueError(
                f"{what}: exponent {exps} has length {len(exps)}, "
                f"expected n_vars = {n_vars}"
            )
        out.append((list(exps), float(coef)))
    return out


def _infer_n_vars(*polys) -> int:
    for p in polys:
        keys = p.keys() if hasattr(p, "keys") else (e for e, _ in p)
        for k in keys:
            return len(tuple(k))
    raise ValueError("cannot infer n_vars from empty polynomials; pass n_vars=")


def sos_minimize(
    objective,
    *,
    inequalities: Sequence = (),
    equalities: Sequence = (),
    n_vars: Optional[int] = None,
    order: Optional[int] = None,
) -> SosResult:
    """Globally minimize ``objective`` subject to ``gᵢ ≥ 0`` (``inequalities``)
    and ``hⱼ = 0`` (``equalities``) via the SOS/Lasserre relaxation.

    Each polynomial is a dict ``{exponent_tuple: coefficient}`` (see the module
    docstring). ``n_vars`` is inferred from the exponent tuples if omitted.
    ``order`` raises the relaxation order above the minimum to tighten the
    bound (the Lasserre hierarchy). Returns an :class:`SosResult`.
    """
    polys = [objective, *inequalities, *equalities]
    if n_vars is None:
        n_vars = _infer_n_vars(*polys)
    obj = _terms(objective, n_vars, "objective")
    ineq = [_terms(g, n_vars, "inequality") for g in inequalities]
    eq = [_terms(h, n_vars, "equality") for h in equalities]
    d = _pounce.sos_minimize(n_vars, obj, ineq, eq, order=order)
    return SosResult(
        lower_bound=float(d["lower_bound"]),
        status=d["status"],
        is_exact=bool(d["is_exact"]),
        num_minimizers=int(d["num_minimizers"]),
        minimizers=[np.asarray(m) for m in d["minimizers"]],
    )
