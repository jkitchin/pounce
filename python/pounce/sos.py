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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import _pounce

__all__ = ["sos_minimize", "SosResult"]


def _check_poly(poly, what: str) -> None:
    """Reject a non-polynomial-dict ``objective``/constraint with a clear
    message instead of a cryptic ``TypeError`` from deep inside the term
    normalizer (issue #117). A polynomial is a dict ``{exp_tuple: coeff}`` or an
    iterable of ``(exp_tuple, coeff)`` pairs."""
    if isinstance(poly, Mapping):
        return
    # A SymPy expression is the natural first attempt for a *polynomial*
    # optimizer — give a targeted conversion hint rather than a generic error.
    if hasattr(poly, "free_symbols") or type(poly).__module__.split(".")[0] == "sympy":
        raise TypeError(
            f"{what} must be a dict {{exponent_tuple: coefficient}}, not a SymPy "
            "expression. Convert it first, e.g. "
            "`{m: float(c) for m, c in sympy.Poly(expr, *syms).terms()}` "
            "(Poly.terms() yields (exponent_tuple, coefficient) pairs)."
        )
    # Otherwise it must be a sequence of (exponent_tuple, coefficient) pairs.
    try:
        ok = all(len(item) == 2 for item in poly)
    except TypeError:
        ok = False
    if not ok:
        raise TypeError(
            f"{what} must be a dict {{exponent_tuple: coefficient}} or a sequence "
            f"of (exponent_tuple, coefficient) pairs; got {type(poly).__name__}. "
            "See the pounce.sos module docstring for the polynomial format."
        )


@dataclass
class SosResult:
    """Result of an SOS/Lasserre solve.

    Attributes
    ----------
    lower_bound:
        Certified global lower bound ``γ* ≤ min p`` (the global minimum when
        ``is_exact``). ``nan`` when ``status`` is not ``"optimal"`` (a failed
        relaxation has no valid bound).
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
    certified:
        Whether ``lower_bound`` is **rigorous** — proved to be a true lower
        bound, not merely accurate to the solver's tolerance. An uncertified
        bound is the raw value the SDP reported; it is normally correct to
        several digits but can land slightly *above* the true minimum, which
        would make it not a lower bound at all. A certified bound has the
        measured miss in the SOS identity subtracted, so it is valid however
        the solve went.

        Certification needs the feasible set to lie in a box readable off the
        constraints (``x >= l`` / ``x <= u`` pairs, or the ``c - a*x**2 >= 0``
        idiom). It is ``False`` on an unbounded feasible set — an
        unconstrained problem, say — where no finite correction exists. Adding
        explicit box constraints you know contain the minimizer upgrades such
        a problem to a certified bound.
    order:
        The relaxation order that produced ``lower_bound``. Normally the order
        requested, but *lower* when that order failed to converge and a coarser
        one did — a coarser bound is still a valid bound, so it is reported
        rather than discarded. Check this before reading a converged result as
        a statement about the order you asked for.
    """

    lower_bound: float
    status: str
    is_exact: bool
    num_minimizers: int
    minimizers: list
    certified: bool
    order: int

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
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
) -> SosResult:
    """Globally minimize ``objective`` subject to ``gᵢ ≥ 0`` (``inequalities``)
    and ``hⱼ = 0`` (``equalities``) via the SOS/Lasserre relaxation.

    Each polynomial is a dict ``{exponent_tuple: coefficient}`` (see the module
    docstring). ``n_vars`` is inferred from the exponent tuples if omitted.
    ``order`` raises the relaxation order above the minimum to tighten the
    bound (the Lasserre hierarchy). Returns an :class:`SosResult`.

    ``tol`` and ``max_iter`` override the underlying SDP solver's convergence
    tolerance (default ``1e-8``) and iteration cap (default ``200``). They are
    an escape hatch for a relaxation that will not converge: loosening ``tol``
    trades certificate strength for convergence. Loosening it cannot make the
    answer *unsound* when ``certified`` is set, because the certification
    subtracts the identity's measured miss rather than trusting the solve.
    """
    _check_poly(objective, "objective")
    for g in inequalities:
        _check_poly(g, "inequality")
    for h in equalities:
        _check_poly(h, "equality")

    polys = [objective, *inequalities, *equalities]
    if n_vars is None:
        n_vars = _infer_n_vars(*polys)
    obj = _terms(objective, n_vars, "objective")
    ineq = [_terms(g, n_vars, "inequality") for g in inequalities]
    eq = [_terms(h, n_vars, "equality") for h in equalities]
    d = _pounce.sos_minimize(
        n_vars, obj, ineq, eq, order=order, tol=tol, max_iter=max_iter
    )
    status = d["status"]
    # On a failed/non-optimal relaxation the raw bound is meaningless (it can be
    # ~5e9 for a problem whose minimum is 1.0); report NaN so a garbage bound
    # can't be mistaken for a real certificate (issue #117, F7).
    lower_bound = float(d["lower_bound"]) if status == "optimal" else float("nan")
    return SosResult(
        lower_bound=lower_bound,
        status=status,
        is_exact=bool(d["is_exact"]),
        num_minimizers=int(d["num_minimizers"]),
        minimizers=[np.asarray(m) for m in d["minimizers"]],
        certified=bool(d["certified"]),
        order=int(d["order"]),
    )
