# Covariance and information: v0.10 roadmap

**Status: roadmap proposal for pyomo-pounce, targeting v0.10.** This note
scopes the post-solve second-order surface in pyomo-pounce: `covariance()`,
shipping in v0.9, and the additions v0.10 should make around it. Everything
here is additive to the v0.9 `covariance()` surface; nothing changes an
existing signature. Companion to the active-set sensitivity roadmap
(`sensitivity-roadmap.md`), which extends `estimate()`; this note extends
`covariance()`.

## Where we are

v0.9 ships `covariance()` (in `pyomo-pounce/pyomo_pounce/sens.py`). You
`declare_fitted` a set of free variables, solve, and `covariance(model)`
returns their asymptotic covariance: the scaled parameter block of the
inverse KKT matrix, `2 * sigma_sq * (K^-1)_pp`. Under the hood that block
is `inv(d2f*/dp2)`, the inverse reduced Hessian of the eliminated problem,
obtained by one backsolve per declared variable against the held
factorization. `hessian=` selects the Lagrangian (observed) or Gauss-Newton
(expected) form.

Two limits matter for what comes next:

1. **There is no un-inverted accessor.** `covariance()` returns the inverse
   reduced Hessian. A caller who wants the reduced Hessian itself, the
   information matrix, has to invert the covariance back.
2. **The reduce-onto block is fixed at declaration time.** `covariance()`
   reports over the whole `declare_fitted` set. Asking about a different
   block means re-declaring and re-solving.

## Why v0.10 needs more

The motivating consumer is moving horizon estimation. The information-form
arrival cost is `Gamma(x0) = 0.5 (x0 - xhat)^T Pi^{-1} (x0 - xhat)`, and the
weighting `Pi^{-1}` is the reduced Hessian marginalized onto the arrival
state, the Lagrangian information, not the covariance. Two things the v0.9
surface makes awkward:

- Getting `Pi^{-1}` through `covariance()` means `inv(covariance(...))`.
  Since the covariance was already `inv(reduced Hessian)` from a backsolve,
  that is invert-then-reinvert. The reduced Hessian was available before the
  inversion; the round trip loses conditioning exactly where MHE lives, the
  weakly-observable direction where the covariance entry is large and
  near-singular to invert back.
- MHE wants two blocks off one solve: the information about the arrival
  state (for the arrival cost) and, when parameters are estimated, their
  covariance. The fixed-declaration surface cannot select two different
  blocks from one factorization.

## Roadmap

**1. `information()`, the un-inverted sibling of `covariance()`.** Returns
the information matrix over the block: the scaled reduced Hessian, formed
directly from the held factor (the Schur complement onto the block's rows)
rather than by inverting the covariance. Numerically it equals
`inv(covariance(...))` on a well-conditioned block, but it skips the round
trip and stays well-scaled in the poorly-identified directions. Same
`hessian=` selector: Lagrangian (default, the exact reduced Hessian, what
the information-form arrival cost wants) and Gauss-Newton (PSD). Same
bound-projection and scaling conventions as `covariance()`.

**2. `wrt=` block selection on both.** `covariance(model, wrt=block)` and
`information(model, wrt=block)` reduce onto the given block, any free
variables, off the held factor, post-solve. The factor captured at the
solution covers every free variable, so the block is a call argument, not a
fixed declaration. `declare_fitted` becomes the default block when `wrt=` is
omitted, which keeps `covariance(model)` behaving exactly as in v0.9. Each
call reduces onto its own argument, so one solve serves as many blocks as
you ask about.

**3. Sensitivity as a solve-time switch, decoupled from `declare_fitted`.**
Retaining the factorization for post-solve use is currently triggered by the
presence of a declaration. Item 2 makes the block a call argument, so a
caller may want no declaration at all. Add an explicit switch to enable
sensitivity (retain the factor) at solve time. `declare_fitted` still
enables it as before and now also serves as the default block; the switch
lets a declaration-free `information(model, wrt=...)` flow work.

**4. Block declaration for time slices.** The MHE arrival block is a state
at one time point, `[m.x[t, c] for c in comp]`, a slice of an indexed Var.
Accept the slice or a `(Var, time)` pair directly, so "the state block at t"
is one call, not a VarData enumeration.

## Marginal versus conditional: the one semantic to get right

Each call reduces onto its argument, and that yields the block's
**marginal**: everything not in the block, other states, parameters, is
integrated out through the KKT reduction. This is what the arrival cost
wants.

The trap is the asymmetry between the two objects. Slicing a covariance
gives a marginal; slicing an information matrix gives a **conditional**. So
`information(wrt=T)` must re-reduce onto `T`, not slice a joint reduction
over some larger set. If it sliced, the arrival-state block of a joint
`{state, params}` information would be the information conditional on the
parameters held fixed, not the marginal that carries their uncertainty.
Making `wrt=` mean "reduce onto this" gives the marginal directly, and the
answer does not depend on what else was declared.

A useful consequence: `information(wrt=arrival_block)` marginalizes the
parameters for free, because they are simply not in the block. You reach for
the conditional only by deliberately putting the parameters in the block and
slicing.

## MHE in one solve

Per window: solve the MHE NLP once with sensitivity on. Then

- `information(model, wrt=arrival_block)` is `Pi^{-1}` for the next window,
  Lagrangian. It comes out as the posterior, not just the data: the current
  arrival-cost term is in the objective, so it enters the reduced Hessian,
  and marginalizing the old arrival state onto the next one carries the prior
  forward, giving prior plus window data, the recursion. Feed it into the
  next arrival cost.
- `covariance(model, wrt=param_block)` is the parameter covariance, marginal
  over the states.

The arrival block is the components of the state that becomes the next
window's start, one time slice, not the whole window. Interior states are
undeclared and marginalized automatically. Whether the arrival cost carries
parameter uncertainty (marginal) or treats parameters as known (conditional)
is a modeling choice, set by whether the parameters are in the arrival block.

## Scope and compatibility

pyomo-pounce only. All four items are additive to v0.9: `information()` is a
new function, `wrt=` is a new optional keyword, the sensitivity switch and
the slice declaration are new surface. Nothing changes an existing
signature. v0.9 `covariance(model)` with no `wrt=` reduces onto the declared
set, which is exactly the v0.10 no-argument default, so the v0.9 surface is
a forward-compatible subset. Nothing here needs to be rushed into v0.9.

## Validation

- `information(...)` against `inv(covariance(...))` to tolerance on a
  well-conditioned block; the conditioning advantage on a deliberately
  ill-identified one.
- The marginal identity: `inv(state block of covariance(wrt={state,
  params}))` against `information(wrt=state)`, both the parameter-marginal
  state information.
- The conditional identity: the state block of `information(wrt={state,
  params})` against `information(wrt=state)` computed with the parameters
  fixed.
- Lagrangian versus Gauss-Newton agree on a linear model and in the
  small-residual limit; the Lagrangian can go indefinite where Gauss-Newton
  stays PSD.
- MHE recursion sanity: the posterior information equals prior plus data on a
  linear-Gaussian window.
