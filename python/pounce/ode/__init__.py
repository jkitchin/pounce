"""Stiff ODE / DAE initial value problems solved with pounce.

``pounce.ode.solve_ivp`` is a drop-in for :func:`scipy.integrate.solve_ivp`
with the implicit ``Radau`` method (3-stage Radau IIA, order 5, L-stable):
it integrates **stiff** systems, and — given a mass matrix ``M`` (``M y' =
f``) — index-1 **DAEs**.

Why this reimplements SciPy's ``Radau`` rather than calling it
--------------------------------------------------------------
The core integrator (:mod:`._radau`) is deliberately the *same* method as
SciPy's ``Radau``, and much of the tableau/error-control logic is
transcribed from it. We do not simply dispatch to
``scipy.integrate.solve_ivp`` because it cannot express — or is closed to —
the three things this module exists to provide:

1. **Mass matrices / index-1 DAEs.** SciPy's ``solve_ivp`` has *no*
   mass-matrix argument: it only integrates the explicit form
   ``y' = f(t, y)``. The Robertson problem and other index-1 DAEs need the
   *singular*-``M`` form ``M y' = f`` (e.g. ``M = diag(1, 1, 0)``), which
   SciPy has no way to represent. ``solve_dae`` / the ``mass=`` argument are
   the reason this module is here at all.

2. **Differentiability.** The trajectory must be differentiable w.r.t.
   parameters and initial conditions (via the implicit-function theorem on
   each step's stage solve, in ``pounce.jax.odeint``). That requires access
   to the internal stage system, which a black-box SciPy call does not
   expose.

3. **A pounce-native numerical stack.** Stage/error operators are factored
   with pounce's own dense LU (:class:`pounce._pounce.DenseLU`, faer-backed)
   rather than SciPy/LAPACK, and — crucially — the stage Jacobian defaults to
   **exact JAX autodiff** when the RHS is traceable (falling back to a
   central finite difference otherwise; see :mod:`._jacobian`). SciPy's
   ``Radau`` uses a finite-difference Jacobian only.

So the duplication is intentional: it is the price of a differentiable,
mass-matrix-capable, JAX-native stiff solver. For a *plain* explicit ODE
with no mass matrix and no gradient requirement, SciPy's ``Radau`` is
equivalent (and this implementation is validated to match its step counts).

The differentiable counterpart (gradients over a *fixed* step sequence via
the implicit-function theorem on each step's stage solve) lives in
``pounce.jax.odeint`` and is imported on demand.
"""

from ._solve import solve_ivp, solve_dae, OdeResult

__all__ = ["solve_ivp", "solve_dae", "OdeResult"]
