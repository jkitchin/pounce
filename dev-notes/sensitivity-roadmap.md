# Active-set-aware parametric sensitivity — roadmap proposal

**Status: roadmap proposal, targeting v0.10.** This note scopes extending
`estimate()` to handle active-set changes, reaching parity with sIPOPT and
then past it, on a clean mechanism/policy boundary. Nothing here is
implemented yet; the intent is to agree the shape before any PR.

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
unimplemented. A light survey of where the boxes are checked
(base predictor, fix-relax, path-following, degeneracy QP, corrector):

| solver / module | base | fix-relax | path | degen | corr |
|---|:--:|:--:|:--:|:--:|:--:|
| IPOPT + sIPOPT / k_aug (open) | ✓ | ✓ | ✗ | ✗ | ✗ |
| WORHP Zen (commercial) | ✓ | ~ | ~ | ? | ~ |
| KNITRO | ~ | ✗ | ✗ | ✗ | ✗ |
| SNOPT | ✗ | ✗ | ✗ | ✗ | ✗ |
| acados RTI (SQP paradigm) | ✗ | ~ | ~ | ✗ | ✓ |
| CasADi sensitivity | ✓ | ✗ | ✗ | ✗ | ✗ |

No single solver checks the full menu in the held-factorization paradigm.
sIPOPT, the open reference, checks two boxes; WORHP Zen is the strongest
but is closed and its full coverage is unconfirmed; acados checks the
corrector box in a different paradigm. The techniques exist scattered
across the literature and these tools, but the full integrated set in one
open solver does not.

## Benefit hypothesis

The contribution is not new sensitivity mathematics: every method here is
established (see State of the art above). Its value is assembling the
known menu into one coherent, open, cleanly-layered implementation — an
explicit ordered-mode plus diagnostics API, automatic degeneracy handling,
and the mechanism/policy split that keeps the primitives in pounce and the
control policy in the caller — which no open package offers today, and
which the one commercial package that might (WORHP Zen) keeps closed. The
payoff is that pounce becomes the open reference for full active-set-aware
parametric sensitivity, so advanced-step NMPC, RTO, and estimation can be
built on an auditable open stack instead of half-measures (a clamp) or a
closed one.

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
`parametric_step`, via `IndexSchurData`). The sIPOPT port itself is done
(pounce#7 and its spun-out pieces #16, #17 are closed): the predictor runs
on real problems. Two properties matter for this roadmap:

1. **On a bound crossing it does a single-pass clamp.** The offending
   variable is clipped to its bound and the other coordinates keep their
   linear-predictor values, frozen. This holds at both layers: the
   pyomo-pounce `np.clip` above (`sens.py:636`) and the Rust
   `crates/pounce-sensitivity/src/boundcheck.rs`, which is gated by the
   `sens_boundcheck` option and is itself only a single-pass clamp, "a
   simpler single-pass clamp rather than upstream's iterative
   Schur-refinement loop" in its own words.
2. **The predictor is `base_x + dx`, with no barrier-parameter
   correction.** It is evaluated against the factorization at the final
   `mu` but carries no explicit `mu`-correction term.

### The precise gap versus sIPOPT

sIPOPT's bound handling is also an option, `sens_boundcheck`, and it
defaults **off** (pounce mirrors this: `sens_app.rs` registers
`sens_boundcheck` default `false`, "Mirrors upstream ...
SensApplication.cpp:63"). The difference is what that option *does* when
on. sIPOPT runs the **iterative Schur-refinement fix-relax**: each
violation adds a row to the augmented system pinning that variable, then
re-solves, so the non-violating coordinates **shift** to stay consistent
with the implicit-function-theorem relations under the pin. pounce pins
the offender and **freezes** the rest.

So the gap is not "no bound handling" and not "no scaffolding". It is the
re-solve: pounce clamps one coordinate, sIPOPT lets the whole solution
bend to absorb the pin. That is exactly what matters on a deep violation,
where the estimate flattens against a bound. The `boundcheck.rs` pointer
to pounce#7 for "the full refinement" is stale (that umbrella issue is
closed); no open issue currently tracks this.

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

So today pounce is *behind* sIPOPT by exactly one step, detailed under
"Where we are" above: sIPOPT's `sens_boundcheck` re-solves so the
solution bends around the pinned variable, where pounce's clamp freezes
the rest.

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

**0. Diagnostics foundation → past sIPOPT.** Breakpoint detection (the
ratio test to the first crossing) and a report the estimate returns:
which variables crossed, the residual, the `mu` used. Cheap, needed by
everything below, and useful on its own — it turns the current silent
clamp into "here is what happened" (sIPOPT exposes no such report).

**1. Fix-relax + `mu`-correction → sIPOPT parity.** Two changes together
constitute full parity. **(a) Fix-relax** (the substantial one): upgrade
`boundcheck.rs` from the single-pass clamp to the Schur-refinement loop —
augment the held factorization with a row pinning the first crossing
variable, re-solve so the non-violating coordinates absorb the pin, via
the `IndexSchurData` path that already does the augmented backsolve
(`solver.rs:360`,`:384`). The `sens_boundcheck` option and the module
already exist, so this is a scoped change to one module, not greenfield;
low-rank Schur update, no refactor; validate against upstream
`SensStdStepCalc.cpp`. Since pounce#7 no longer covers it, it should get
its own issue. **(b) `mu`-correction** (minor): apply the eq. 10 term that
corrects the predictor for the factorization sitting at `mu` > 0 rather
than `mu` = 0. Automatic, inside the predictor, negligible at tight
tolerance — the small remaining formal gap. Fix-relax carries the active
set, the `mu`-correction finishes the predictor, and the two are full
sIPOPT parity.

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
auto-triggered when the solve detects weak activity, not a user knob. Cost
is conditional: the detection is a negligible threshold scan over the
converged multipliers, always paid; the QP fires only on a degenerate base
point, and when it does it is small — over the weakly-active set
(dimension = the number of weakly-active constraints, usually a handful),
solved against the held factorization at roughly a backsolve per
weakly-active constraint, no refactor and well short of a re-solve.

**4. Corrector-step primitive → past sIPOPT.** One Newton/primal-dual
iteration reusing the held factorization, returning the residual. Small
and general; it composes with path-following (path-following gets the
active set, the corrector polishes the point). Expose two surfaces: the
raw single step, for callers that drive their own loop (e.g. a
deadline-bounded one in an advanced-step controller), and a convenience
wrapper that loops to a residual tolerance with an iteration cap and a
stagnation guard. The residual tolerance is a numerical stopping criterion
the solver owns; a budget or deadline stop stays with the caller and uses
the raw step. Cost is ~1 backsolve per iteration.

## API surface

The sensitivity API surface is four elements of two types. Three are
**modes** of `estimate()`, an ordered ladder on a single `mode` argument
(`linear` the baseline, `fix_relax` item 1, `path` item 2), each a
correctness-superset of the one below. The fourth is the **corrector
step** (item 4), a separate primitive the caller drives in a loop. Item 3
(QP directional) is not part of the surface: it is applied automatically
inside `estimate()` when the base point is degenerate, so the caller
neither selects nor calls it. The cost unit is a **backsolve** against the
held factorization (microseconds; the cheap operation the feature exists
to exploit); a **re-solve** is the expensive bound.

| element | choose when | cost | type |
|---------|-------------|------|------|
| `linear` (default) | small perturbations that stay interior, or hot loops where a clamp at a bound is acceptable | 1 backsolve | mode |
| `fix_relax` | the perturbation crosses a bound and you want the whole solution to bend around the pin rather than truncate one coordinate | ~2 backsolves (predictor + one Schur-augmented solve); low-rank update, no refactor | mode |
| `path` | large perturbations that cross several bounds, when the estimate must track the exact re-solve | `k` crossings × (backsolve + Schur update), bounded above by one re-solve | mode |
| corrector step | the caller polishes the estimate toward feasibility / optimality, in a loop | ~1 backsolve per iteration | primitive |

The report is always returned; the modes are one ordered knob, not a
matrix of independent flags. The `mu`-correction folded into item 1 is
always applied inside the predictor, so every mode is `mu`-corrected. The
default is `linear`, matching today's behaviour and the reference: sIPOPT
ships with `sens_boundcheck` off, i.e. it defaults to the plain predictor
and makes the active-set correction opt-in.

## Scope boundary: mechanism in pounce, policy in the caller

Everything above is a **mechanism**: a stateless operation on the held
factorization (a step, a corrector iteration, a fix-relax solve, a
breakpoint test) plus the diagnostics to decide. The **policy** — how many
correctors, when to stop, which mode per cycle, whether to abandon the
estimate and call a full `solve()` instead, the advanced-step
solve-ahead-then-update orchestration — belongs to the application layer
(e.g. an advanced-step NMPC controller such as
[drto](https://github.com/devin-griff/drto)), because those decisions
depend on a real-time budget the solver has no business knowing.

Keeping the split means the primitives serve estimation, RTO, and control
alike, and pounce stays a general solver rather than absorbing a
controller. Concretely: a loop whose size is fixed by the problem or by
numerical convergence runs to completion inside pounce — path-following
across its crossings, or a corrector loop to a residual tolerance. A loop
whose size is fixed by an external budget — a deadline-bounded corrector
loop — lives in the caller and drives the raw single step, because only
the caller knows the deadline.

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
- Büskens, Maurer, *Sensitivity analysis and real-time optimization of
  parametric nonlinear programming problems*, in Online Optimization of
  Large Scale Systems, Springer, 2001 (active-set-change sensitivity for
  real-time control; the WORHP Zen lineage).
- Nikolayzik, Büskens et al., *WORHP Zen: Parametric Sensitivity Analysis
  for the NLP Solver WORHP*, 2018.
  [link](https://link.springer.com/chapter/10.1007/978-3-319-89920-6_86)
- Gros, Zanon, Quirynen, Bemporad, Diehl, *From linear to nonlinear MPC:
  bridging the gap via the real-time iteration*, Int. J. Control 93 (2020)
  (the RTI / SQP real-time paradigm behind acados).
- Andersson et al., *CasADi: a software framework for nonlinear
  optimization and optimal control*, Math. Program. Comput. 11 (2019).
