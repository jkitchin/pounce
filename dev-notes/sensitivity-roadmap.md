# Active-set-aware parametric sensitivity — roadmap proposal

**Status: proposal, for discussion.** This note scopes extending
`estimate()` to handle active-set changes, reaching parity with sIPOPT and
then past it, on a clean mechanism/policy boundary. Nothing here is
implemented yet; the intent is to agree the shape before any PR.

## Where we are

`estimate()` (`pyomo-pounce/pyomo_pounce/sens.py:598`) computes the
first-order parametric step and returns the updated solution:

```
dx    = session.solver.parametric_step(pin_idx, deltas)   # Schur backsolve
x_new = session.base_x + dx                               # sens.py:623
...
x_new = np.clip(x_new, lo, hi)                            # sens.py:636 (warns)
```

The step itself is one Schur-complement backsolve against the held KKT
factorization (`crates/pounce-sensitivity/src/solver.rs:360`,
`parametric_step`, via `IndexSchurData`). Two properties matter for this
roadmap:

1. **On a bound crossing it clamps.** When the linear step leaves the
   variable bounds, `estimate` projects back onto them and warns. That is
   the crude treatment: the active set has changed and the first-order
   prediction is no longer valid there.
2. **The predictor is `base_x + dx`, with no barrier-parameter
   correction.** It is evaluated against the factorization at the final
   `mu` but carries no explicit `mu`-correction term.

## The reference: what sIPOPT actually implements

From Pirnay, López-Negrete & Biegler, *Optimal sensitivity based on
IPOPT*, Math. Program. Comput. 4 (2012) 307–331
([DOI](https://doi.org/10.1007/s12532-012-0043-2)), read against the
paper's §2.3–2.4:

- **Base linear predictor** — the same Schur-complement step pounce does.
- **Single-crossing fix-relax.** When a perturbed variable would violate
  its bound, sIPOPT augments the KKT system with a row pinning that
  variable to the bound and relaxes the matching complementarity condition
  with a new multiplier, solved through the Schur complement on the
  already-held factorization. This is a genuine active-set correction, not
  a clamp, done as a low-rank update rather than a refactor.
- **Evaluated at the final `mu` with an explicit `mu`-correction** (the
  paper's eq. 10).

Explicitly **not** implemented in sIPOPT, per the paper: the QP/LCP
directional method (eq. 14, "the current version of sIPOPT does not
include an implementation of (14)"), multi-step path-following, and any
predictor-corrector loop.

So today pounce is *behind* sIPOPT by one item: sIPOPT fix-relaxes the
first crossing where pounce clamps.

## Two failure modes (they want different treatments)

1. **A finite perturbation crosses a bound.** The base point is fine but
   the step activates or deactivates a constraint partway. Fixed by
   fix-relax (one crossing) or path-following (many).
2. **Degeneracy at the base point.** A constraint is weakly active
   (at its bound with a near-zero multiplier), so strict complementarity
   fails where we linearize. The linear step gives a two-sided derivative
   that is wrong on at least one side; the correct object is a directional
   derivative from a small QP.

## Roadmap

Staged by dependency and by "parity vs past it".

**0. Diagnostics foundation.** Breakpoint detection (the ratio test to the
first crossing) and a report the estimate returns: which variables
crossed, the residual, the `mu` used. Cheap, needed by everything below,
and useful on its own — it turns the current silent clamp into "here is
what happened". *Beyond sIPOPT (it exposes no such report).*

**1. Single-crossing fix-relax → sIPOPT parity.** Augment the held
factorization to pin the first crossing variable at its bound and relax its
complementarity, via the existing `IndexSchurData` path
(`solver.rs:360`,`:384`). Low-rank Schur update, no refactor. This is the
one capability sIPOPT has and pounce lacks, so it is the parity milestone,
and it has a reference to validate against. *Reaches parity on the
active-set axis.*

**2. Multi-crossing path-following → past sIPOPT (crossing axis).** Iterate
the fix-relax across successive breakpoints toward the target
perturbation. This is the stepwise continuation the 2012 paper derived
(§2.3) but left unimplemented. Needs breakpoint ordering, constraint add
*and* drop, and anti-cycling. Cost scales with active-set churn, up to a
re-solve.

**3. QP directional → past sIPOPT (degeneracy axis).** For a weakly-active
base point, solve the small QP (the paper's eq. 14) over the weakly-active
set for the correct one-sided derivative, through the QP already in
`pounce-convex` (`crates/pounce-convex/src/ipm.rs`). Independent of 1–2;
auto-triggered when the solve detects weak activity, not a user knob.

**+ Corrector-step primitive.** One Newton/primal-dual iteration reusing
the held factorization, returning the residual. Small and general; it is
what an advanced-step controller's corrector loop calls. Composes with
path-following (path-following gets the active set, the corrector polishes
the point).

**+ (minor) `mu`-correction term.** The remaining small gap to sIPOPT's
predictor (eq. 10). Negligible at tight convergence tolerance, so listed
separately from the parity milestone rather than bundled into it.

## API surface

A single ordered `mode` on `estimate`, each level a correctness-superset of
the one below:

- `linear` — pure predictor, clamp, warn (today's behaviour).
- `fix_relax` — single-crossing correction.
- `path` — multi-step path-following.
- `exact` — re-solve (ground truth, and the honest cap when path-following
  would cost as much anyway).

The report is always returned; base-point degeneracy is handled
automatically rather than as a mode. One knob, ordered, obvious default —
not a matrix of independent flags.

## Scope boundary: mechanism in pounce, policy in the caller

Everything above is a **mechanism**: a stateless operation on the held
factorization (a step, a corrector iteration, a fix-relax solve, a
breakpoint test) plus the diagnostics to decide. The **policy** — how many
correctors, when to stop, which mode per cycle, the advanced-step
solve-ahead-then-update orchestration — belongs to the application layer
(e.g. an advanced-step NMPC controller such as
[drto](https://github.com/devin-griff/drto)), because those decisions
depend on a real-time budget the solver has no business knowing.

Keeping the split means the primitives serve estimation, RTO, and control
alike, and pounce stays a general solver rather than absorbing a
controller. Concretely: path-following runs to completion inside pounce
(its size is fixed by the problem), while a corrector *loop* lives in the
caller (its size is fixed by an external deadline), calling the corrector
primitive repeatedly.

## Open decisions (for discussion)

1. **First PR scope.** Diagnostics alone (a clean standalone contribution),
   or diagnostics + fix-relax as one "active-set-aware estimate" PR?
2. **Default mode.** `linear` keeps today's behaviour and suits hot loops
   but is silently wrong across a crossing; `fix_relax` is correct-by-
   default at a small cost. This is the main philosophical call.
3. **`mu`-correction.** Fold into the predictor, or keep it a separate
   small item, or expose the barrier smoothing as a knob at all?
4. **Sequencing the novel milestones.** Path-following and QP directional
   have no reference to point at, unlike fix-relax; worth a short design
   note each before code?

## Validation

- **fix-relax** against sIPOPT's own worked example (the paper's §2.8
  parametric QP with a documented active-set change) and against a full
  re-solve.
- **path-following** against re-solve across several crossings.
- **QP directional** against finite differences and a constructed
  weakly-active case.
- End-to-end: the [cstr-sensitivity](https://cstr-sensitivity.griffith-pse.com)
  demo, where the estimate visibly flattens against a bound today and
  should bend correctly under `fix_relax`.

## References

- Pirnay, López-Negrete, Biegler, *Optimal sensitivity based on IPOPT*,
  Math. Program. Comput. 4 (2012) 307–331.
  [DOI](https://doi.org/10.1007/s12532-012-0043-2)
- Zavala, Biegler, *The advanced-step NMPC controller*, Automatica 45
  (2009) 86–93. [DOI](https://doi.org/10.1016/j.automatica.2008.06.011)
- Fiacco, *Introduction to Sensitivity and Stability Analysis in Nonlinear
  Programming*, Academic Press, 1983 (the regularity conditions and the
  directional-derivative QP).
