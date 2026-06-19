"""Boundary value problems solved with pounce.

``pounce.bvp.solve_bvp`` is a drop-in for
:func:`scipy.integrate.solve_bvp`: it discretises the BVP with 4th-order
Hermite--Simpson collocation on a **fixed** mesh and solves the resulting
square root-find as a pounce feasibility NLP.

The differentiable counterparts live next to the autodiff frontends —
``pounce.jax.solve_bvp`` and ``pounce.torch.solve_bvp`` — which expose a
``theta`` knob (any parameter inside ``fun`` / ``bc``, the unknown
parameters ``p``, or boundary values) and return a solution that is
differentiable w.r.t. ``theta`` via the implicit-function theorem on the
collocation KKT system. Those imports are deferred to their backends so
``pounce.bvp`` itself needs only NumPy + SciPy.
"""

from ._solve import solve_bvp, BVPResult
from ._constrained import solve_bvp_constrained

__all__ = ["solve_bvp", "BVPResult", "solve_bvp_constrained"]
