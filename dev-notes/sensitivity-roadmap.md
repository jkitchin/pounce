# Active-set-aware parametric sensitivity: v0.10 roadmap

**Status: roadmap proposal for pyomo-pounce, targeting v0.10.** This note
scopes extending pyomo-pounce's `estimate()` to handle active-set changes,
reaching parity with sIPOPT and then past it, on a clean mechanism/policy
boundary. That is the pyomo / held-factorization sensitivity path
specifically; the jax/torch and convex-QP frontends have their own
sensitivity surfaces (see Related work). Item 0 is implemented and
merged. The rest is a proposal, and the intent is to agree the shape
before any PR.

## State of the art

Two paradigms address parametric NLP sensitivity. The
**held-factorization** family computes `dx*/dp` from the converged KKT
factor and updates from there (sIPOPT, k_aug, WORHP Zen, CasADi's
sensitivity); this is the paradigm here. The **SQP/QP real-time** family
re-solves a QP subproblem each step (acados' real-time iteration; the
warm-starting of SNOPT, KNITRO). The individual techniques the roadmap
uses are all in the literature: Fiacco's stability theory and the
directional-derivative QP; the Büskens/Maurer active-set-change
sensitivity for real-time control; and the sIPOPT paper itself, which
derives multi-step path-following and the eq. 14 QP but leaves them
unimplemented. Where the boxes are checked (base predictor, fix-relax,
path-following, degeneracy QP, corrector), for the two whose coverage is
established from primary sources:

| solver / module | base | fix-relax | path | degen | corr |
|---|:--:|:--:|:--:|:--:|:--:|
| IPOPT + sIPOPT / k_aug (open) | ✓ | ✓ | ✗ | ✗ | ✗ |
| CasADi sensitivity | ✓ | ✗ | ✗ | ✗ | ✗ |

sIPOPT is the open reference and checks two: the paper states outright that
the eq. 14 QP is not implemented, and neither multi-step path-following nor
a corrector loop appears. WORHP Zen documents parametric sensitivity for a
closed solver (Kuhlmann et al., below); acados works the corrector in the
SQP paradigm rather than this one. Their per-technique coverage is not
established here, so the claim this roadmap rests on is the one about the
open reference, which is checkable.

## Benefit hypothesis

Every method here is established. The value is assembling the known menu
into one open implementation: an explicit ordered-mode plus diagnostics API,
automatic degeneracy handling, and the mechanism/policy split. No open
package offers that today, and the closest commercial work is closed. The
payoff is that advanced-step NMPC, RTO, and estimation get an auditable open
stack instead of a clamp.

## Where we are

In pyomo-pounce, the held-factor sensitivity is just the linear predictor
today. `estimate()` (in `pyomo-pounce/pyomo_pounce/sens.py`) computes the
first-order parametric step and returns the updated solution:

```
dx    = session.scatter_x(                                # Schur backsolve;
    session.solver.parametric_step(pin_idx, deltas))      # factor rows to full-x
x_new = session.base_x + dx                               # linear predictor
...
x_new = np.clip(x_new, lo, hi)                            # clamp; warns first
```

The step itself is one Schur-complement backsolve against the held KKT
factorization (`parametric_step` in
`crates/pounce-sensitivity/src/solver.rs`, via `IndexSchurData`). The
sIPOPT port itself is done
(pounce#7 and its spun-out pieces #16, #17 are closed): the predictor runs
on real problems. Two properties matter for this roadmap:

1. **On a bound crossing it does a single-pass clamp.** The offending
   variable is clipped to its bound and the other coordinates keep their
   linear-predictor values, frozen. This holds at both layers: the
   pyomo-pounce `np.clip` above and the Rust
   `crates/pounce-sensitivity/src/boundcheck.rs`, which is gated by the
   `sens_boundcheck` option and is itself only a single-pass clamp, "a
   simpler single-pass clamp rather than upstream's iterative
   Schur-refinement loop" in its own words.
2. **The predictor is `base_x + dx`, with no barrier-parameter
   correction.** It is evaluated against the factorization at the final
   `μ` but carries no explicit `μ`-correction term.

## The reference: what sIPOPT implements, and the gap

From Pirnay, López-Negrete & Biegler, *Optimal sensitivity based on
IPOPT*, Math. Program. Comput. 4 (2012) 307–331
([DOI](https://doi.org/10.1007/s12532-012-0043-2)), §2.3–2.4:

- **Base linear predictor** — the same Schur-complement step pounce does.
- **Fix-relax.** When a perturbed variable would violate its bound, sIPOPT
  augments the KKT system with a row pinning that variable to the bound and
  relaxes the matching complementarity condition with a new multiplier,
  solved through the Schur complement on the already-held factorization. A
  low-rank update rather than a refactor.
- **Evaluated at the final `μ` with an explicit `μ`-correction** (the
  paper's eq. 10).

Explicitly **not** implemented in sIPOPT, per the paper: the QP/LCP
directional method (eq. 14, "the current version of sIPOPT does not
include an implementation of (14)"), multi-step path-following, and any
predictor-corrector loop.

Fix-relax is an option there too, `sens_boundcheck`, defaulting **off**
(pounce mirrors this: `sens_app.rs` registers it `false`, "Mirrors upstream
... SensApplication.cpp:63"). The gap is what the option does when on.
sIPOPT re-solves: each violation adds a row to the augmented system pinning
that variable, so the non-violating coordinates **shift** to stay consistent
with the implicit-function-theorem relations under the pin. pounce pins the
offender and **freezes** the rest, which is what matters on a deep
violation, where the estimate flattens against a bound. pounce also omits
the eq. 10 `μ`-correction. Item 1 closes both.

The `boundcheck.rs` pointer to pounce#7 for "the full refinement" is stale
(that umbrella issue is closed); no open issue currently tracks this.

## Related sensitivity work in other pounce frontends

pyomo-pounce is not the only pounce frontend with sensitivity machinery,
and this roadmap should reuse the others' vocabulary and shared core
primitives rather than reinvent them.

- The jax and torch frontends ship a predictor-corrector path follower
  (`PathFollower`, pounce#90): a held-factor predictor, an
  active-set-margin monitor (pounce#89), and a warm re-solve corrector
  built on the barrier-`μ` warm start (pounce#86). It traces problems
  defined through those autodiff frontends and crosses active-set changes
  by re-solving, not by fix-relax.
- The convex-QP frontend ships `QpSensitivity` (`python/pounce/qp.py`,
  backed by `crates/pounce-convex/src/sensitivity.rs`), described in its
  own docstring as the sIPOPT analog. It already exposes `parametric_step`,
  weak-activity detection (`weakly_active_indices`), and active-set-change
  diagnostics (`kkt_dim`, `active_indices`). Its predictor is still linear,
  so it detects degeneracy but does not correct for it.

Neither lives on the held-factorization / pyomo path that `estimate()`
exposes to Pyomo models, so neither delivers the pyomo-pounce capability
below, and neither supplies the classifier item 0 needs: `QpSensitivity`
screens on a solver with no `μ`, so the barrier ratio that separates the
regimes does not exist there. The pyomo path should share the same core Schur
backsolve the jax follower already leans on.

## Two failure modes (they want different treatments)

1. **A finite perturbation crosses a bound.** The base point is fine but
   the step activates or deactivates a constraint partway. Fixed by
   fix-relax (one crossing) or path-following (many).
2. **Degeneracy at the base point.** A constraint is weakly active
   (at its bound with a near-zero multiplier), so strict complementarity
   fails where we linearize. The linear step gives a two-sided derivative
   that is wrong on at least one side; the correct object is a directional
   derivative from a small QP. The step also sits strictly between the two
   one-sided values at every `μ`, so tightening the solver tolerance does
   not move it toward either.

## Roadmap

Staged by dependency: the diagnostics foundation (item 0), parity (item 1),
then items 2–4. Everything except item 1 reaches past sIPOPT; item 1 is
the parity step.

**0. Diagnostics foundation → past sIPOPT.** Breakpoint detection (the
ratio test to the first crossing) and a report the estimate returns:
which variables crossed, the constraint violation at the predicted
point, the `μ` used, and the activity classification, so a caller can
tell a derivative from one element of a set. Useful on its own — it
turns the current silent clamp into "here is what happened" (sIPOPT
exposes no such report).

The violation is the primal half of the residual, and it is the half
this item can carry. The NL evaluators supply it directly, whereas the
dual half needs the multipliers at the perturbed point. Only the primal
block and the equality multipliers have a user-space mapping from the
compound vector, so assembling the dual half here would mean exposing
the inequality and bound multiplier steps as well. Item 4 computes the
full KKT residual instead, inside the corrector that already holds the
updated multipliers.

The classification is which regime each bounded variable is in, inactive,
weakly active or strongly active, read off the ratio of its barrier curvature
to the objective's own curvature there. That ratio is `O(μ)`, `O(1)` and
`O(1/μ)` across the three, so it separates them at any `μ`. That
classifier shipped with the covariance/information work:
`Solver.classify_activity()` in the core, with the user-space report and
the natural-units exports (`var_sigma`, `row_sigma`, `row_normal(j)`,
`hessian_vec(v)`, `primal_rows`), recorded in
`covariance-information-design.md`. It is the same classifier rather
than a second one, and the exposure gate this item and item 3 once
waited on is cleared. The breakpoint half and the report assembly are
now implemented as `estimate_report()`.

The report also carries the provenance of the number it returns: the `μ` it
was evaluated at, whether the solver relaxed the bounds, and whether
`kkt_perturbations` is non-zero. Those are the three things that separate the
predictor from the exact active-set value, they are all cheap, and without
them a caller comparing against a re-solve cannot tell which one explains the
gap. All three are already exposed: the merged classifier checks
`bound_relax_factor` and the central path on every call, and the
covariance/information accessors read (and warn on) `kkt_perturbations`,
so the report only passes them through.

**1. Fix-relax + `μ`-correction → sIPOPT parity.** Two changes together
constitute full parity. **(a) Fix-relax** (the substantial one): upgrade
`boundcheck.rs` from the single-pass clamp to the Schur-refinement loop —
augment the held factorization with a row pinning the first crossing
variable, re-solve so the non-violating coordinates absorb the pin, via
the `IndexSchurData` path that already does the augmented backsolve
(`parametric_step` in `solver.rs`). The `sens_boundcheck` option and the
module already exist, so this is one module and a low-rank Schur update.
Validate against upstream `SensStdStepCalc.cpp`. Since pounce#7 no longer
covers it, it should get its own issue. **(b) `μ`-correction** (minor):
apply the eq. 10 term that
corrects the predictor for the factorization sitting at `μ` > 0 rather
than `μ` = 0. Automatic, inside the predictor, and negligible at tight
tolerance under strict complementarity. It is not the fix for failure mode
2: eq. 10 is derived under the paper's Property 1, whose third condition is
strict complementarity, so where that fails the correction does not apply
and item 3 is what handles it. Fix-relax carries the active set, the
`μ`-correction finishes the predictor, and the two are full sIPOPT parity.

**2. Multi-crossing path-following → past sIPOPT (crossing axis).** Iterate
the fix-relax across successive breakpoints toward the target
perturbation. The 2012 paper takes the same stepwise shape for its QP
(§2.3, solve at the first active-set change, then again from there to the
target) and credits stepwise application of the base sensitivity step to
earlier work; neither is in sIPOPT. Needs breakpoint ordering, constraint add
*and* drop, and anti-cycling. Cost scales with active-set churn, up to a
re-solve.

**3. QP directional → past sIPOPT (degeneracy axis).** For a weakly-active
base point, solve the small QP (the paper's eq. 14) over the weakly-active
set for the correct one-sided derivative. Those constraints are already
rows in the held factorization, so this is an active-set search over them
on that factor, the same held-factor Schur primitive fix-relax uses
(`IndexSchurData`), not a fresh QP solve. Detection is item 0's classifier;
the correction is what pyomo-pounce adds. Independent of items 1 and 2,
auto-triggered on a weakly active base point, not a user knob. Cost is
conditional: the classification is a threshold scan, always paid; the QP
fires only on a degenerate base point, over the weakly-active set, at roughly
a backsolve per weakly-active constraint.

There is no side to choose for `estimate()`. The call names a perturbation, so
the direction is given, and eq. 14 takes `Δp` as an input. It returns the
directional value for the direction asked, with no signature change and no
refusal, so a loop stepping a saturated control keeps running.

`gradient()` is where the two-valuedness is real, since `dx/dp` with no
direction has two answers at a kink. It keeps returning a float, warns that
the base point is degenerate and the value one-sided, and reports which side,
so a caller who needs the other one asks through `estimate()`. Refusing would
break a call that always answers today over a condition most users will not
recognize.

**4. Corrector-step primitive → past sIPOPT.** One Newton/primal-dual
iteration reusing the held factorization, returning the full KKT
residual, both the constraint violation and stationarity. The
stationarity half belongs here rather than in item 0 because the
iteration updates the multipliers and so already holds them, where
item 0 would have to reach for multiplier steps that have no user-space
mapping. Small and general; it composes with path-following
(path-following gets the active set, the corrector polishes the point).
Expose two surfaces: the
raw single step, for callers that drive their own loop (e.g. a
deadline-bounded one in an advanced-step controller), and a convenience
wrapper that loops to a residual tolerance with an iteration cap and a
stagnation guard. The residual tolerance is a numerical stopping criterion
the solver owns; a budget or deadline stop stays with the caller and uses
the raw step. Cost is ~1 backsolve per iteration.

### Measured: corrector refactorization policies on the double column

A study (2026-08-18) settles the refactorization question for item 4.
Model: the double column DAE stack, N=25 Radau, 62,167 variables,
61,967 equalities, base solve 93 iterations, final `μ` 2.5e-9, 792
weakly active bounds. The corrector iterated the primal-dual barrier
system at the held final `μ`: residuals from the NL evaluators at the
moving iterate, backsolves through the held factorization, fraction to
the boundary at 0.9995. The perturbation is a fixed random direction
over all 246 initial-state parameters (the advanced-step scenario),
each entry scaled by its distance to the nearer bound, one knob
scaling the vector. Knob 1 to 2 percent corresponds to realistic
estimator mismatch. The predictor is the linear sensitivity step. Truth
per knob is a warm re-solve at tol 1e-10. The floor is the offset
between the barrier solution at the held `μ` and that truth, 7.2e-7 to
7.6e-7 on rows away from the weakly active set. It is the base solve's
own solution quality, and it appears only against the tighter truth:
the corrector's residual itself converges to 1e-13.

The floor statement describes the corrector's converged fixed point,
the perturbed problem's central-path point at the held `μ`. The
barrier system carries no active-set choice, so a predictor that
misjudged the active set does not move that fixed point. What it
affects is convergence to it: larger drift, slower contraction, and in
the worst measured case the residual cycle noted below. The converged
residual is the certificate that the fixed point was reached. Without
it the point can still be good (the cycling case below sits at the
floor) but nothing certifies it, and on a nonconvex problem a far
enough start could in principle converge to a different local branch.

Iterations to the floor (error is the inf norm over rows away from the
weakly active set):

| knob | predictor error | chord on held factors | factor once at the predictor, then chord | refactor every iteration |
|------|-----------------|-----------------------|------------------------------------------|--------------------------|
| 1%   | 3.2e-3          | 1 (0.09 s)            | 1 (0.53 s)                               | 1 (0.59 s)               |
| 2%   | 1.3e-2          | 2 (0.18 s)            | 1 (0.57 s)                               | 1 (0.72 s)               |
| 5%   | 8.1e-2          | 3 (0.33 s)            | 2 (0.71 s)                               | 2 (1.30 s)               |
| 10%  | 3.2e-1          | 3 (0.30 s)            | 2 (0.72 s)                               | 2 (1.38 s)               |
| 20%  | 1.3e0           | 5 (0.50 s)            | 2 (0.70 s)                               | 2 (1.30 s)               |

Per-iteration costs on the study machine: 0.07 to 0.095 s for a
held-factor backsolve plus residual evaluation, 0.6 to 0.7 s for an
iteration containing a factorization, 3 to 4.3 s for a full warm
re-solve.

Findings:

- Chord on the held factors wins every cell. Refactorizing every
  iteration is dominated everywhere: the matrix barely changes between
  corrector iterates, so the extra factorizations buy residual polish
  below the floor and nothing else.
- One factorization at the predictor point gives contraction near
  1/200 per iteration independent of step size, because the chord
  drift is then the predictor error, which is quadratic in the step.
  It is the robust variant but its first iteration alone costs more
  than chord's whole path at every knob here.
- A second direction, a persistent feed-composition parameter swept +1
  to +20 percent, reproduces all of the above through +10 percent. At
  +20 percent chord needs 9 backsolves to the floor and then enters a
  bounded residual two-cycle driven by the fraction-to-the-boundary
  damping (error stays at the floor, residual oscillates between
  5.9e-8 and 2.6e-6). One refactorization at the current iterate
  removes the cycle. Non-monotone residual is the stagnation signal.
- The fix_relax and path predictors produced corrected trajectories
  identical to the linear predictor's in the state direction, at 37 to
  72 s per estimate call (dominated by the directional QP exhausting
  its trial budget over the 792 weakly active bounds). With a
  corrector, the linear predictor dominates the expensive modes.

Design consequence: the corrector is chord on the held factorization
with an iteration cap and a residual-tolerance early exit at the base
solve's own tolerance. Refactorization is not a policy knob. The one
escalation worth building is a single refactorization at the current
iterate when the residual stops decreasing monotonically, which
reproduces the factor-once behavior exactly where it is needed and
nowhere else.

### Item 3 findings from the double column (2026-08-19)

The same model exercised the shipped `degeneracy="directional"` search and
an exact external solve of its eq. 14 QP. Results, each measured:

- The shipped search spends 32 s exhausting its 16-trial budget over the
  792 weakly active bounds and falls back to one-sided on every call. The
  cost is one released refactorization per trial, and the trial count a
  correct enumeration needs grows combinatorially with the weak set, so
  no budget fixes it.
- A candidate working set that holds a weak bound whose variable an
  equality row already pins is singular by construction, and one singular
  trial ends the whole search. The column has about twenty such rows, the
  t=0 states under their initial conditions, and dynamic models generally
  carry this class.
- The weak set splits into 444 trace-composition lower bounds, present at
  any operating point of this model family, and 348 tray-holdup bounds
  from the transient level policy. The composition bounds also sit inside
  equality-pinned initial conditions at t=0.
- The exact QP, solved in the weak rows' reduced space against the held
  factor: single-violator pivoting converges deterministically (1248
  pivots, holds 84 of 792). Undamped and damped block pivoting both
  cycle. Equality-pinned weak rows must start released and never pivot.
  The cost floor for any factor-reuse implementation is one released
  refactorization plus one released-factor column per row that enters
  the working set, which on this model is several seconds even with
  batched backsolves, above the 3.5 s warm re-solve.
- An earlier two-way disagreement between pivoting variants was traced to
  two sign-test defects in the harness (the multiplier baseline carried
  the factor's upper-row sign, and the tolerance was scaled by the
  multiplier block). Nonconvexity of the QP was not established: the one
  tested difference direction had positive curvature. GSSOSC remains
  unverified on this model class.
- The mode machinery carries its own cost layers at this scale,
  independent of any degeneracy handling: fix_relax's refinement is about
  3 s per pin round (47.7 s per call here) and path about 1 s per walk
  segment (15.0 s per call at its 16-segment cap). Supplying the QP
  decision through the existing bordered machinery adds a column per
  released row (105 s) or a walk segment per forced release (18 minutes,
  capped, and the capped predictor is useless). Every mode that manages
  bound activity costs more than the 3.5 s warm re-solve on this model.
- Corrected results are predictor-independent: all four predictors
  (fix_relax and path, one-sided and QP-decided) reach the 7.6e-7 floor
  within three chord iterations at +1 percent, including the capped path
  predictor whose own error was 0.13. A better predictor buys at most one
  backsolve.
- The predictor is the corrector's first iteration: chord started from
  the base point evaluates the residual at the shifted parameters, and
  its first backsolve reproduces the linear predictor up to the barrier
  correction term. A predictor-only call stays meaningful as the variant
  that needs no model evaluations, and as `corrector_iter=0` continuity.

Design consequence for item 3: replace the enumeration's interior with
single-violator pivoting on one in-place released refactorization, keep
equality-pinned weak rows released, and add a regime guard: when the
all-released solve reports many sign violations, skip the search and fall
back immediately, because the decision then costs more than a re-solve
and correction makes it unnecessary. The directional mode's documented
scope is small weakly active sets. The recommended path on large models
is `mode="linear"` with corrector iterations.

## API surface

Three **modes** of `estimate()`, an ordered ladder on a single `mode`
argument, each a correctness-superset of the one below, plus the
**corrector step**, a separate primitive the caller drives in a loop. Item 3
is not on the surface: it applies automatically inside `estimate()` when the
base point is degenerate. Costs are in **backsolves** against the held
factorization; a **re-solve** is the expensive bound.

| element | choose when | cost | type |
|---------|-------------|------|------|
| `linear` (default) | small perturbations that stay interior, or hot loops where a clamp at a bound is acceptable | 1 backsolve | mode |
| `fix_relax` | the perturbation crosses a bound and you want the whole solution to bend around the pin rather than truncate one coordinate | ~2 backsolves (predictor + one Schur-augmented solve); low-rank update, no refactor | mode |
| `path` | large perturbations that cross several bounds, when the estimate must track the exact re-solve | `k` crossings × (backsolve + Schur update), bounded above by one re-solve | mode |
| corrector step | the caller polishes the estimate toward feasibility / optimality, in a loop | ~1 backsolve per iteration | primitive |

The report is always returned; the modes are one ordered knob, not a
matrix of independent flags. The `μ`-correction folded into item 1 is
always applied inside the predictor, so every mode is `μ`-corrected. The
default is `linear`, which matches today's active-set semantics up to that
negligible `μ`-correction, and matches the reference: sIPOPT ships with
`sens_boundcheck` off, i.e. it defaults to the plain predictor and makes
the active-set correction opt-in.

## Scope boundary: mechanism in pounce, policy in the caller

A loop whose size is fixed by the problem or by numerical convergence runs to
completion inside pounce, so path-following across its crossings and a
corrector loop to a residual tolerance both live here. A loop whose size is
fixed by an external budget lives in the caller and drives the raw single
step, because only the caller knows the deadline.

## Validation

- **diagnostics** — breakpoint detection (which variable crosses first,
  and the step fraction) against a brute-force ratio-test scan; the report
  fields (crossed set, residual, `μ`, classification) against ground truth.
  The classifier is already validated by its merged suites.
- **fix-relax** against sIPOPT's own worked example (the paper's §2.8
  parametric QP with a documented active-set change) and against a full
  re-solve.
- **path-following** against re-solve across several crossings.
- **QP directional** against finite differences and a constructed
  weakly-active case.
- **corrector** — the residual decreases monotonically per iteration and
  the corrected point converges to the exact re-solve; the convenience
  loop terminates correctly on the residual tolerance, the iteration cap,
  and a stagnation case.
- End-to-end: a constrained optimal-control example (e.g. a CSTR whose
  controls hit their bounds), where the estimate visibly flattens against
  a bound today and should bend correctly under `fix_relax`.

## References

- Pirnay, López-Negrete, Biegler, *Optimal sensitivity based on IPOPT*,
  Math. Program. Comput. 4 (2012) 307–331.
  [DOI](https://doi.org/10.1007/s12532-012-0043-2)
- Zavala, Biegler, *The advanced-step NMPC controller*, Automatica 45
  (2009) 86–93. [DOI](https://doi.org/10.1016/j.automatica.2008.06.011)
- Fiacco, *Introduction to Sensitivity and Stability Analysis in Nonlinear
  Programming*, Academic Press, 1983 (the regularity conditions and the
  directional-derivative QP).
- Büskens, Maurer, *Sensitivity analysis and real-time optimization of
  parametric nonlinear programming problems*, in Online Optimization of
  Large Scale Systems, Springer, 2001 (active-set-change sensitivity for
  real-time control; the WORHP Zen lineage).
- Kuhlmann, Geffken, Büskens, *WORHP Zen: Parametric Sensitivity Analysis
  for the Nonlinear Programming Solver WORHP*, Operations Research
  Proceedings 2017, Springer, 2018.
  [link](https://link.springer.com/chapter/10.1007/978-3-319-89920-6_86)
- Gros, Zanon, Quirynen, Bemporad, Diehl, *From linear to nonlinear MPC:
  bridging the gap via the real-time iteration*, Int. J. Control 93 (2020)
  (the RTI / SQP real-time paradigm behind acados).
- Andersson et al., *CasADi: a software framework for nonlinear
  optimization and optimal control*, Math. Program. Comput. 11 (2019).
