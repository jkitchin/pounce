"""Stiff ODE / DAE initial value problems solved with pounce.

``pounce.ode.solve_ivp`` is a drop-in for :func:`scipy.integrate.solve_ivp`
with the implicit ``Radau`` method (3-stage Radau IIA, order 5, L-stable):
it integrates **stiff** systems, and — given a mass matrix ``M`` (``M y' =
f``) — index-1 **DAEs**, which SciPy's ``solve_ivp`` cannot. Each step's
collocation stage system is solved by Newton with FERAL's sparse LU.

The differentiable counterpart (gradients of the trajectory w.r.t.
parameters / initial conditions over a *fixed* step sequence, via the
implicit-function theorem on each step's stage solve) lives in
``pounce.jax.odeint`` and is imported on demand.
"""

from ._solve import solve_ivp, OdeResult

__all__ = ["solve_ivp", "OdeResult"]
