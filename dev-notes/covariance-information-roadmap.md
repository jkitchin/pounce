# Covariance and information: v0.10 roadmap

**Status: roadmap proposal for pyomo-pounce, targeting v0.10.** This note
scopes the post-solve second-order surface in pyomo-pounce: `covariance()`,
shipping in v0.9, and the additions v0.10 should make around it. Everything
here is additive to the v0.9 `covariance()` surface; nothing changes an
existing signature. Companion to the active-set sensitivity roadmap
(`sensitivity-roadmap.md`), which extends `estimate()`; this note extends
`covariance()`.

## State of the art

Parameter covariance from the reduced Hessian is standard: at an estimation
optimum the covariance is the scaled inverse of the reduced Hessian of the
Lagrangian. sIPOPT computes it (its section 4), k_aug computes it, and
scipy's `curve_fit` reports it in the Gauss-Newton form.

pounce already ships the object in several of its own interfaces. The core
`Problem.solve_with_sens` returns the reduced Hessian in natural (unscaled)
units, so `-inv(reduced_hessian)` is directly the covariance, with an
eigendecomposition (pounce#128, mirroring sIPOPT's `rh_eigendecomp`).
`QpSensitivity.reduced_hessian` mirrors it on the convex-QP side, and
`pounce.curve_fit` is the scipy-style covariance frontend for callable
models, of which `covariance()` is the Pyomo-model sibling.

The pyomo-pounce interface is the exception: it exposes only `covariance()`,
the inverse, over a fixed declared set, with no reduced-Hessian accessor and
no per-call block.

## Benefit hypothesis

The contribution is not the reduced Hessian or the covariance recipe. Both
are established, and pounce already ships them in its core, QP, and
`curve_fit` interfaces (see State of the art). It is two things the
pyomo-pounce interface lacks and that no pounce interface offers together:

- an `information()` accessor consistent with `covariance()` and the core's
  natural-units reduced Hessian, so a Pyomo model gets the un-inverted
  object without the invert-then-reinvert round trip; and
- post-solve `wrt=` block selection off one retained factor, reducing onto
  arbitrary free-variable blocks from a single solve, the
  one-solve-two-blocks flow the MHE arrival cost needs, with `retain_kkt()`
  as the declaration-free enabler.

So this is an interface and ergonomics contribution on the pyomo side,
layered on the core's existing reduced Hessian, not new numerics.

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
   information matrix, has to invert the covariance back. This is a
   pyomo-interface gap, not a pounce one: the core and QP interfaces already
   expose the reduced Hessian directly (see State of the art).
2. **The reduce-onto block is fixed at declaration time.** `covariance()`
   reports over the whole `declare_fitted` set. Asking about a different
   block means re-declaring and re-solving.

## The MHE arrival cost

The motivating consumer is moving horizon estimation. Its information-form
arrival cost is `Gamma(x0) = 0.5 (x0 - xhat)^T Pi^{-1} (x0 - xhat)`, where the
weighting `Pi^{-1}` is the reduced Hessian marginalized onto the arrival
state, the Lagrangian information, not the covariance. That un-inverted,
per-block object is what the roadmap below adds; the concrete per-window loop
is in the MHE section.

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
you ask about. The block, whether declared or passed to `wrt=`, accepts a
slice (`m.x[t, :]`) or a `(Var, time)` pair, not just a hand-listed VarData
set, so an MHE arrival state at one time point is one call rather than an
enumeration.

**3. `retain_kkt()`, a factor-retention switch decoupled from
declarations.** The solve factors the KKT to solve the NLP; the only
question is whether that factor is kept for post-solve queries. Today it is
kept whenever a declaration is present (`declare_sens_param`,
`declare_fitted`, or `declare_residual`). Item 2 lets the block move to a
call argument, so a caller may want no declaration at all. `retain_kkt()`
keeps the factor without committing to a block, param, or residual. It
defaults off, so a solve with no sensitivity pays nothing.

`retain_kkt()` is not specific to this surface. Keeping the factor is the
substrate every sensitivity feature rests on, and `gradient()` and
`estimate()` already get it as a side effect of the `declare_sens_param`
they require, so they never call `retain_kkt()` and it has no user-facing
effect on them. The only flow that needs a declaration-free retain is
`covariance` / `information` driven purely by `wrt=`.

`wrt=` itself is not gated by `retain_kkt()`. It needs the factor kept, and
`declare_fitted` already keeps it, so `wrt=` works with `declare_fitted`
alone. `retain_kkt()` earns its place only when you want the factor kept
with no default block at all: the declaration-free MHE case, where the
arrival state and the parameters are each queried by `wrt=` and neither is a
default.

| setup | factor kept | `covariance(model)` | `covariance(model, wrt=T)` |
|---|---|---|---|
| nothing | no | error | error |
| `declare_fitted(S)` | yes | over S | over T |
| `retain_kkt()` only | yes | error, no default | over T |
| `retain_kkt()` + `declare_fitted(S)` | yes | over S | over T |

The columns show `covariance()` for concreteness; `information()` follows the
same rows, since factor retention and the default block are accessor-agnostic.

Any declaration keeps the factor, not only `declare_fitted`, so `retain_kkt()`
is needed only when nothing at all is declared. In particular
`declare_sens_param` alone (no `declare_fitted`, no `retain_kkt()`) does
support `covariance(model, wrt=T)` and `information(model, wrt=T)` off the same
solve. It just carries no default block, so a bare `covariance(model)` errors,
exactly the `retain_kkt()`-only row. The block `T` then comes out conditional
on the pinned parameter, since fixing an input conditions rather than
marginalizes.

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

Per window: solve the MHE NLP once with `retain_kkt()` set, so the factor is
kept with no default block. Then

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

pyomo-pounce only. All three items are additive to v0.9: `information()` is
a new function, `wrt=` (with its slice and `(Var, time)` block forms) is a
new optional keyword, and `retain_kkt()` is new surface. Nothing
changes an existing signature. v0.9 `covariance(model)` with no `wrt=` reduces onto the declared
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
