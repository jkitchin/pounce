<!-- gh#990 item 16. Paste-ready replacement for the jkitchin/pounce.wiki page
"Recovering from a bad start" (a separate repository this checkout cannot edit).
Changes against the live page: the ladder is four rungs, not two; rung 3
(start_point_perturbation=1e-2) DOES move the start, so "never the starting
point" is gone; the pin-the-trajectory advice names all four *_retry options;
the 0.10.0 transcripts are labelled as such and the one that cannot be
re-measured here (the flash model needs Pyomo) is not re-claimed. Delete this
comment when pasting. -->
# Recovering from a bad start

You are here because a solve failed, stalled, or returned something you
do not believe, and you suspect the point it started from. This page
runs in that direction: from the failure, back to the point, to the
move that fixes it.

For *how* initialization works — the interior push, the dual defaults,
warm starts — see [Initialization and Warm Starts](https://jkitchin.github.io/pounce/initialization.html).
For option recipes indexed by symptom, see
[Troubleshooting Recipes](https://jkitchin.github.io/pounce/troubleshooting.html). This page is the bridge
between them.

## First: is the starting point actually the problem?

The control that settles it costs one solve. Re-run from a point you
trust — a converged solution of a nearby model, a coarser model's
answer, the midpoint of the bounds, anything defensible — and see what
happens. If it still fails, the start was never the cause, and
[Troubleshooting Recipes](https://jkitchin.github.io/pounce/troubleshooting.html) is the right page.

Before that, the preflight evaluates the model once at its starting
point and reports what iteration 0 will see, without solving:

```sh
pounce check-x0 model.nl              # text report; --json for tools
pounce check-x0 model.nl --x0-file candidate.txt
```

```python
report = pounce.preflight(problem_obj, x0, lb=lb, ub=ub, cl=cl, cu=cu)
report = pyomo_pounce.preflight(model)          # Pyomo models
```

Exit code 0 means the model evaluates cleanly (warnings allowed); 21
means a solve from this point would abort.

> **`check-x0` evaluates; it does not factor.** It sees NaN/inf, bound
> violations, how far the interior clamp will move you, the initial
> constraint violation, the derivative scale spread, and the factors
> automatic scaling will pick here. It cannot see the *rank* of the
> Jacobian at your point. For that you need `presolve=yes` (structural,
> see [LICQ below](#the-jacobian-is-rank-deficient-where-you-started))
> or the debugger's `print rank` (numerical) — neither of which is a
> no-solve preflight.

### What POUNCE already tried before it gave up

Several recovery paths are on by default, so by the time you see a
failure the solver has usually re-solved two or three times. A local
infeasibility verdict, for example, is re-tried along up to four
different trajectories — the **second-opinion ladder** — before it is
believed. The CLI narrates it (format as printed by 0.12):

```
pounce: local infeasibility — re-solving along 3 different trajectories before
  believing it (second-opinion ladder: feral_scaling=mc64, mu_strategy=adaptive,
  start_point_perturbation=1e-2).
pounce: second opinion — re-solving with feral_scaling=mc64…
pounce: feral_scaling=mc64 re-solve did not recover (InfeasibleProblemDetected).
pounce: second opinion — re-solving with mu_strategy=adaptive…
pounce: mu_strategy=adaptive re-solve did not recover (InfeasibleProblemDetected).
pounce: second opinion — re-solving with start_point_perturbation=1e-2…
pounce: start_point_perturbation=1e-2 re-solve did not recover (InfeasibleProblemDetected).
pounce: keeping the original Infeasible_Problem_Detected verdict; it survived
  3 independent re-solve(s) (feral_scaling=mc64, mu_strategy=adaptive,
  start_point_perturbation=1e-2).
```

The four rungs:

| rung | re-solve with | varies |
|---|---|---|
| 1 | `feral_scaling=mc64` | the linear algebra |
| 2 | `mu_strategy=adaptive` | the barrier trajectory |
| 3 | `start_point_perturbation=1e-2` | **where the trajectory starts** |
| 4 | `feral_increase_quality=no` | whether a stalled factorization may reroute the trajectory (opens only when the failed solve actually escalated its factorization) |

Which rungs open depends on the status: `Infeasible_Problem_Detected` can
reach all four, `Invalid_Number_Detected` only rung 3 (a NaN out of your
callbacks is a statement about the callbacks at a point), `Restoration_Failed`
rungs 3 and 4, `Maximum_Iterations_Exceeded` rung 4 alone. Rung 3 is
`infeasibility_perturbed_start_retry`; to switch the whole ladder off you
must name all four `*_retry` options. Full detail, including what each
option's own text claims, in
[Troubleshooting Recipes](https://jkitchin.github.io/pounce/troubleshooting.html).
`mu_strategy_fallback` and the ℓ₁ exact-penalty wrapper have the same
shape on other statuses.

> **Rungs 1, 2 and 4 vary the algorithm, never the starting point; rung 3
> varies the starting point, but only by a small deterministic
> displacement.** The older version of this page said every ladder varies
> "the algorithm, never the starting point" — that was true of the
> two-rung ladder it was written against and is not true now.
> `start_point_perturbation=1e-2` moves each variable by
> `1e-2·(1 + |xᵢ|)·uᵢ`, `uᵢ` drawn deterministically from `[-1, 1)` and
> clipped back inside its bounds. That restores rank at a *structurally*
> degenerate point — a squared slack sitting at zero, the origin on a
> homogeneous quadratic, where LICQ fails and the filter line search has no
> descent direction to find. It does not repair a start that is merely
> poor: far from the feasible region, on the wrong side of a nonconvexity,
> or with the scaling poisoned. So a verdict that "survived 3 independent
> re-solves" is stronger evidence than one that survived two, and still
> not evidence the model is hard: it says several trajectories agree about
> what they can see from a neighbourhood of where you put them. The
> [worked example](#a-feasible-model-reported-infeasible) below is a
> feasible model reported infeasible from the origin.

A second consequence: these re-solves make the report ambiguous. If you
are measuring anything, pin the trajectory first —
`mu_strategy=monotone mu_strategy_fallback=no` **and** all four ladder
options (`feral_infeasibility_scaling_retry=no`,
`infeasibility_mu_strategy_retry=no`, `infeasibility_perturbed_start_retry=no`,
`feral_increase_quality_retry=no`; naming a subset leaves the other rungs
live) — and confirm one solve ran.

## Three kinds of bad start

They are worth separating because the fixes are disjoint.

**(A) The point is structurally degenerate.** The origin; every
component equal; everything sitting exactly on a bound; a symmetric
point on a symmetric model. The pathology is a property of `x0` alone,
and the fix is a *different point*.

**(B) The point makes the algorithm's first step degenerate.** The
Jacobian is rank-deficient where you started but not at the solution;
the least-square dual estimate blows past `constr_mult_init_max` and is
discarded; the gradient-based scaler takes its sample at a nonsense
point; inertia correction fires at iteration 0 and never stops. The
point may be perfectly reasonable — the fix is *different treatment of
the same point*.

**(C) The point is merely bad.** Far from feasible, wrong order of
magnitude, in a different basin, or outside a function's domain. The
fix is often a *different model* — usually bounds.

## The atlas

| Bad start | What you see | First move |
|---|---|---|
| The origin, on a model with `log`, `sqrt` or a division | `Invalid_Number_Detected`, zero iterations | [add bounds](#invalid_number_detected-at-iteration-0) so the clamp lands in the domain |
| The origin, on a model whose objective gradient blows up there | `Solve_Succeeded` at a slightly wrong answer | [check the scaling preview](#a-start-that-poisons-the-scaling) |
| The origin, via an unset `Var.value` or `x.L` | `Infeasible_Problem_Detected` on a feasible model | [fill the values](#a-feasible-model-reported-infeasible) |
| `x'Qx ≤ b` written about the origin | nothing at all; a slow or wrong solve later | [Scaling](https://jkitchin.github.io/pounce/scaling.html), §"Quadratic rows the sampler cannot see" |
| Everything exactly on a bound | the clamp moves every component | [read the clamp report](#the-point-you-gave-is-not-the-point-it-starts-from) |
| A previous solution used as a cold start | the active set does not survive | [warm start instead](https://jkitchin.github.io/pounce/initialization.html#warm-starting-the-interior-point-path) |
| All components equal, on a symmetric model | regularization from iteration 0 | [break the symmetry](#symmetric-and-all-equal-starts) *(unmeasured)* |
| LICQ fails at `x0` but not at `x*` | `δ_c` applied from the first factorization | [`presolve=yes`](#the-jacobian-is-rank-deficient-where-you-started) |
| Far from feasible, large `θ(x0)` | restoration at iteration 1; `Restoration_Failed` | [repair the point](#restoration-at-iteration-1-and-restoration-that-never-leaves) |
| Multipliers of the wrong magnitude | `inf_du` large at iteration 0 and never falling | `constr_mult_init_max`, `bound_mult_init_val` *(unmeasured)* |
| A different basin | `Solve_Succeeded`, wrong answer, nothing says so | [get an oracle](#success-at-the-wrong-answer) |
| Non-differentiable at `x0` (`abs`, `min`, `max`) | tiny steps; `derivative_test` disagreement | perturb off the kink *(unmeasured)* |

> **Rows marked *(unmeasured)*** say what should happen and why, but
> have not been shown to recover a named problem here. See
> [Troubleshooting Recipes](https://jkitchin.github.io/pounce/troubleshooting.html) for the difference that
> makes.

## `Invalid_Number_Detected` at iteration 0

The most common bad start is the one nobody chose. A Pyomo `Var` whose
`.value` was never set is written as `0` into the `.nl` file; GAMS
levels default to `0` unless you assign `x.L`; a `.nl` variable with no
initial-guess entry is `0`. A model initialized "nowhere" is
initialized at the origin, which for most process models is outside
every meaningful range and a domain error for `log`, `/` and friends.

Entropy minimisation on the simplex — `min Σ xᵢ log xᵢ + cᵀx` subject
to `Σ xᵢ = 1`, four variables, `c = (0, 1, 2, 3)`, which has the closed
form `x*ᵢ ∝ exp(-cᵢ)` and `f* = -log Σ exp(-cᵢ)` to check against —
started at the origin:

| variables | status | iterations |
|---|---|---:|
| free (no bounds) | `Invalid_Number_Detected` | 0 |
| bounded `0 ≤ x ≤ 10` | `Solve_Succeeded` | 16 |

Identical model, identical starting point. The difference is that the
interior clamp moves a bounded variable off `0` to `0.01` *before the
first evaluation*, so `log` never sees a zero.

> **The clamp repairs bound violations, not domain errors.** A free
> variable is left exactly where you put it. Adding a bound that the
> clamp can push you off is the cheapest robustness lever there is, and
> it is usually more honest than the alternative — you generally do
> know that a mole fraction is non-negative.

Note the second row solves but is not out of trouble; see
[the scaling section](#a-start-that-poisons-the-scaling).

## The point you gave is not the point it starts from

Per component, the initializer clamps `x` into
`[lo + p_l, hi - p_u]` with
`p_l = min(bound_push · max(|lo|, 1), bound_frac · (hi - lo))`. With
the defaults (`bound_push = bound_frac = 1e-2`), a variable sitting
exactly on its lower bound `1.0` starts at `1.01`. `check-x0` shows
exactly what will move:

```
  x0 vs bounds:
    violations: 0  on-bound components: 20
    interior clamp moves 20 component(s), max move 1.000e-2
        x[0]: 1.000000e0 -> 1.010000e0  (moved 1.000e-2)
        x[1]: 1.000000e0 -> 1.010000e0  (moved 1.000e-2)
        ...
```

So a deliberately-designed active set does not survive a cold start,
and neither does a previous solution handed back as `x0`. That is what
[warm starts](https://jkitchin.github.io/pounce/initialization.html#warm-starting-the-interior-point-path)
are for: `warm_start_init_point=yes` uses a separate set of push
constants an order of magnitude tighter than the cold path's —
`warm_start_bound_push` and `warm_start_bound_frac` default to `1e-3`
against `1e-2` — plus the slack and multiplier equivalents.

**Tightening `bound_push` is not the fix.** On the problem above — 20
variables whose optimum is exactly on the lower bound, started exactly
there — sweeping it changes almost nothing:

| `bound_push` | `1e-1` | `1e-2` | `1e-4` | `1e-6` | `1e-8` |
|---|---:|---:|---:|---:|---:|
| iterations | 7 | 6 | 7 | 7 | 7 |

The likely reason is that the barrier parameter starts at
`mu_init = 0.1` regardless of how good your point is, so the early
iterations are dominated by centering rather than by where inside the
box you began — but that is the explanation, not something measured
here. What *is* measured is the null: four orders of `bound_push`, one
iteration of spread. If you want the active set honored, warm-start;
do not shrink the push.

> **That sweep needs `solver_selection=nlp` to mean anything.** The
> problem above is a convex QP, so it routes by default to
> `pounce-convex`, which has its own cold start and ignores every
> option on this page. Check the banner:
> `Problem class: convex QP. Selected solver: convex QP interior-point
> (pounce-convex) [solver_selection=auto].` See
> [LP / QP Solver Routing](https://jkitchin.github.io/pounce/lp-qp-routing.html).

## A start that poisons the scaling

`nlp_scaling_method=gradient-based`, the default, is a *point sample*:
it takes the gradients once, at `x0`, and derives the factors the whole
solve then runs under. A starting point that is bad for evaluation is
therefore also bad for scaling, and that damage outlives the point.

The entropy model again, bounded so it evaluates, against its analytic
optimum `f* = -0.4401896985611953`. From `check-x0`:

```
# started at the origin
  automatic scaling at x0 (nlp_scaling_method=gradient-based, nlp_scaling_max_gradient=100):
    objective: ||grad f|| inf -> factor 1.000e-8

# started at x = 0.25
  automatic scaling at x0 (nlp_scaling_method=gradient-based, nlp_scaling_max_gradient=100):
    objective: ||grad f|| 2.614e0 -> factor 1.000e0  (below the cutoff: unscaled)
```

The gradient is *infinite* at the origin, so the scaler falls to
`nlp_scaling_min_value` (default `1e-8`) and the model is solved eight
orders of magnitude down-scaled. Both runs report `Solve_Succeeded`:

| run | iterations | `\|f - f*\|` |
|---|---:|---:|
| `x0 = 0`, defaults | 16 | 1.3e-09 |
| `x0 = 0`, `nlp_scaling_method=none` | 10 | **5.6e-17** |
| `x0 = 0`, `least_square_init_primal=yes` | 15 | 1.3e-09 |
| `x0 = 0`, `bound_push=0.25` | 16 | 1.3e-09 |
| `x0 = 0`, `mu_init=1.0` | 17 | 1.3e-09 |
| `x0 = 0.25` (a sane start) | 9 | **5.6e-17** |

Two things to take from this.

**Only the scaling lever recovers the accuracy, and it recovers it
exactly.** Turning the point-sampled scaling off from the bad start
reaches the same answer, to the last bit, as starting from a sane
point. The three initialization knobs move the iteration count by at
most one and do not touch the answer at all.

**The failure is invisible in the status.** Both runs "succeeded". The
convergence test is applied to the *scaled* problem, so a solve
running at `obj_scale = 1e-8` can satisfy it eight orders early. The
unscaled column of the final report is where this shows:

```
                                   (scaled)                 (unscaled)
Objective...............:   -4.4018969722021155e-09    -4.4018969722021151e-01
```

Watch for a scaled/unscaled pair that disagree by many orders.

POUNCE does notice, which is why the error is `1e-9` rather than
something far worse — it refuses a certificate earned at an extreme
objective scale and keeps going:

```
INFO pounce_algorithm::conv_check::opt_error: refusing a termination
certificate masked by an extreme objective scale; continuing toward the true
minimum (obj_scale_certificate_threshold=0 disables) obj_scale=1e-8
unscaled_kkt_error=0.000331204481957356 scaled_nlp_error=3.31204481957356e-12
threshold=0.0001
```

That guard bounds the damage; it does not undo it. Run with
`RUST_LOG=info` to see it fire — it is a reliable signal that your
starting point set the scaling. See
[Scaling](https://jkitchin.github.io/pounce/scaling.html) for the full picture, including the related trap
where a quadratic row written about the origin has a zero Jacobian
there and is left unscaled entirely.

## Symmetric and all-equal starts

A symmetric model started at a symmetric point produces a KKT system
that inherits the symmetry, so the step direction the solver needs to
break the tie is exactly the one the linear algebra cannot supply.
Regularization fires from iteration 0 and the trajectory crawls.

The fix is to break the symmetry before the solver has to:

```python
starts = pounce.generate_starts(4, x0=x0, strategy="jitter",
                                jitter=0.1, seed=0)
```

*(unmeasured — how large the jitter must be is not established here,
and is a question worth measuring on your model rather than
guessing.)*

## The Jacobian is rank-deficient where you started

Dependent active constraint gradients put a singular block in the KKT
matrix, and the perturbation handler answers with `δ_c` from the very
first factorization. When that is a property of the *model*, presolve
can say so structurally without a solve:

```sh
pounce model.nl presolve=yes presolve_print_level=3
```

`presolve_licq_check` is on by default. Its verdict distinguishes an
empty equality row, more equalities than variables, and a structural
rank shortfall found by bipartite matching.
`presolve_licq_action=auto_l1` switches on the ℓ₁ exact-penalty wrapper
when it fires. Note that presolve frequently *removes* the problem
instead of reporting it — aggregating dependent linear equalities away
— which is a fix, not a miss.

When the dependency is numerical rather than structural, the debugger
is the tool:

```
pounce-dbg> print rank
equality Jacobian J_c: 7 row(s) × 7 column(s)
numerical rank = 7 / 7  (deficiency 0)
σ_max = 3.056e0   σ_min = 8.077e-2   cond = 3.784e1   (rank tol τ = 4.751e-15)
J_c has full row rank at this iterate.
```

Reading a *clean* rank report is as useful as reading a dirty one: it
rules the model out and sends you back to the starting point. The
transcript above is from the failing solve in the
[worked example](#a-feasible-model-reported-infeasible) below.

## Restoration at iteration 1, and restoration that never leaves

A large initial constraint violation `θ(x0)` sends the solve into the
feasibility restoration phase immediately. `check-x0` flags it before
you get there:

```
  initial constraint violation:
    rows violated: 1  max violation: 9.997e11

  warnings:
    - very large initial infeasibility (max constraint violation 9.997e11);
      consider a better starting point or least_square_init_primal=yes
```

Restoration that cannot escape is the failure mode to recognise. The
algorithm carries three detectors for it: a static-cycle test that
exits immediately when two restoration entries are within a relative
`1e-10` of each other, a recovery-cycle test with a ten-strike limit,
and a near-feasible re-entry counter. What you see is
`Restoration_Failed`, a large `restoration_calls`, and an iteration
count that stopped meaning anything.

```sh
RUST_LOG=pounce::restoration=debug pounce model.nl
POUNCE_DBG_RESTO_CYCLE=1 pounce model.nl
```

`restoration_calls`, `restoration_outer_iters` and
`restoration_inner_iters` are in the JSON report's `statistics` block
and in Python's `info` dict.

> **A failed restoration does not always look like a failure.** Before
> returning `Restoration_Failed`, the algorithm tries to fall back to a
> stored acceptable point — so a start bad enough to break restoration
> can surface as `Solved_To_Acceptable_Level` instead.

The moves, cheapest first: `pounce.project_to_feasible` to repair the
point onto the constraints;
[`least_square_init_primal=yes`](https://jkitchin.github.io/pounce/initialization.html) and its
safeguard;
then `start_with_resto=yes`; then the
[ℓ₁ exact-penalty wrapper](https://jkitchin.github.io/pounce/troubleshooting.html).

### A repaired start is not automatically a better one

`least_square_init_primal` replaces `x0` with the min-norm solution of
the *linearized* constraints, and its safeguard only accepts a step
that actually reduces the true nonlinear violation. On
`crates/pounce-cli/tests/fixtures/hs13_bigstart.nl` — the fixture whose
`check-x0` output is quoted above — it does exactly what it promises:

```sh
RUST_LOG=pounce::algorithm=debug pounce hs13_bigstart.nl \
    least_square_init_primal=yes
# DEBUG pounce::algorithm: pounce: least_square_init_primal safeguard decision
#   violation_initial=999700039999.0 violation_final=0.0 alpha=1.0
#   step_norm=16666.133424793596 rejected_trials=0 termination="accepted"
```

A violation of `1e12` driven to **exactly zero**, at full step, with no
rejected trials. And the solve gets *harder*:

| configuration | solves | iterations | status |
|---|---:|---|---|
| defaults | 1 | 29 | `Solve_Succeeded` |
| `least_square_init_primal=yes` | 3 | 4, 4, 20 | `Solve_Succeeded` |

From the repaired point the first two trajectories (the base solve and the
`feral_scaling=mc64` re-solve) converge to local infeasibility in four
iterations each; only the `mu_strategy=adaptive` rung of the second-opinion
ladder recovers it (this transcript is from 0.10.0, whose ladder had two
rungs; later versions also have `start_point_perturbation=1e-2` after it):

```
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
pounce: second opinion — re-solving with feral_scaling=mc64…
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
pounce: second opinion — re-solving with mu_strategy=adaptive…
EXIT: Optimal Solution Found.
pounce: mu_strategy=adaptive re-solve recovered the problem — promoting.
```

Feasibility at iteration 0 is not the objective; convergence is. A
point sitting exactly on a nonlinear constraint manifold can be a worse
place to start than one comfortably off it, and `mu_strategy_fallback=no`
does not suppress this ladder — it has its own trigger. **Count the
`Number of Iterations` lines before you read one: there is exactly one
per solve, and the final `EXIT:` verdict is echoed, so it is not a
reliable count.**

## Success at the wrong answer

A bad start can land you in a different basin, and nothing in the
report will say so. `pounce verify` certifies *feasibility*, not
optimality, so it passes such a point too.

This is why recovery is two-dimensional: a solve that converges from a
repaired start to a *different* local minimum has not been recovered.
Every claim on this page is checked against an oracle — an analytic
optimum, a known objective, or a second solver — and yours should be
too. [Finding Multiple Minima](https://jkitchin.github.io/pounce/find-minima.html) is the systematic
version.

## The recovery ladder

### Tier 0 — find out whether the point is the problem

`pounce check-x0` (no solve) · `derivative_test=first-order`, because
wrong derivatives look exactly like a bad start · `presolve=yes` with
`presolve_licq_check` · `pounce model.nl --debug-on-error` to land in
the debugger at the moment of failure, then `diagnose`:

```
── pounce-dbg ── TERMINATED (LocalInfeasibility)  iter 25  obj=-1.088834e0
[warning] primal_infeasible: Primal infeasibility 2.45e-3; worst constraint
          residual is c[bal[3]] = +2.455e-3.
[warning] dual_infeasible: Dual infeasibility 1.00e0; largest stationarity
          residual is grad_x_L[x[1]] = -1.000e0.
[warning] tiny_step: Accepted primal step α_pr=1.01e-9 is tiny — the line
          search is barely moving.
[   info] bounds_pinned: 1 variable bound(s) are active (slack < 1e-6).
```

Its other codes read as a degeneracy checklist: `structural_singularity`
(a Dulmage–Mendelsohn decomposition that names the dependent
equations), `rank_deficient_jacobian`, `inertia_wrong`,
`heavy_regularization`, `in_restoration`, `mu_stalled`. Alongside it,
`print rank` and `print kkt`, and `sweep` / `multistart` for
[initialization sensitivity](https://jkitchin.github.io/pounce/debugger.html#multi-start-and-initialization-sensitivity).
Over a finished run instead of a live one, `pounce-studio` has a
post-mortem `diagnose` with `restoration_used`, `restoration_loop`,
`mu_stuck`, `heavy_line_search` and `convergence_stall`.

### Tier 1 — repair the point

```python
x0, rep = pounce.project_to_feasible(problem_obj, x0, lb=lb, ub=ub,
                                     cl=cl, cu=cu, return_report=True)
starts  = pounce.generate_starts(16, bounds=bounds, seed=0)
```

`project_to_feasible` is a safeguarded elastic repair that never
returns a point whose true nonlinear violation is worse than the one
you gave it. For Pyomo models, `pyomo_pounce` carries a full
initialization pipeline: `initialize_missing_values` for the unset-value
trap, and `block_analyze` / `block_initialize` / `block_repair_plan` /
`initialize` for a Dulmage–Mendelsohn square-part decomposition solved
block by block in topological order, restoring seeds on failure.

### Tier 2 — change how the solver treats the point

`bound_push`, `bound_frac`, `slack_bound_push`, `slack_bound_frac` ·
`bound_mult_init_val`, `constr_mult_init_max` · `mu_init` ·
`least_square_init_primal` · `start_with_resto` · `nlp_scaling_method`
· `start_point_perturbation` (a deterministic relative displacement of the
start; the ladder's rung 3 applies `1e-2` for you after an infeasibility /
invalid-number / restoration failure)
· `warm_start_init_point` and its push constants ·
`presolve_licq_action`.

> **Two of these look like levers and are not.**
> `least_square_init_duals=yes` and `bound_mult_init_method=mu-based`
> parse — so an `ipopt.opt` written for Ipopt still loads — and are
> then *refused* with an explanation rather than silently downgraded to
> something else. The run stops; remove the option to proceed.

### Tier 3 — stop relying on one point

`pounce.generate_starts` feeding `solve_nlp_batch`, or
`pounce.race_starts` with `policy="halving"` so hopeless candidates are
dropped after a few iterations instead of paying full freight ·
[`find_minima`](https://jkitchin.github.io/pounce/find-minima.html) · homotopy from a relaxation you *can*
solve, via [Continuation](https://jkitchin.github.io/pounce/continuation.html) or
[Sessions](https://jkitchin.github.io/pounce/sessions.html).

### Tier 4 — change the model, not the point

Add bounds — it keeps the interior clamp inside the function's domain
and costs nothing when they are honest. Reformulate the term that
cannot evaluate. Supply scaling by hand
([`user-scaling`](https://jkitchin.github.io/pounce/scaling.html#user-scaling)) when the point sample
cannot be trusted.

## A feasible model reported infeasible

A three-component isothermal flash: equilibrium `yᵢ = Kᵢxᵢ`, component
balances `zᵢ = V·yᵢ + (1-V)·xᵢ`, `Σxᵢ = 1`, entropy-of-mixing
objective. Seven variables, seven constraints, written the way models
are usually written — no `Var.value` set anywhere.

```python
import pyomo.environ as pyo, pyomo_pounce

K = {1: 2.5, 2: 1.0, 3: 0.35}
z = {1: 0.3, 2: 0.4, 3: 0.3}

m = pyo.ConcreteModel()
m.i   = pyo.Set(initialize=[1, 2, 3])
m.x   = pyo.Var(m.i, domain=pyo.NonNegativeReals)      # liquid fractions
m.y   = pyo.Var(m.i, domain=pyo.NonNegativeReals)      # vapour fractions
m.V   = pyo.Var(bounds=(0.01, 0.99))                   # vapour fraction
m.eq  = pyo.Constraint(m.i, rule=lambda m, i: m.y[i] == K[i] * m.x[i])
m.bal = pyo.Constraint(m.i, rule=lambda m, i: z[i] == m.V*m.y[i]
                                                    + (1 - m.V)*m.x[i])
m.sum = pyo.Constraint(expr=sum(m.x[i] for i in m.i) == 1)
m.obj = pyo.Objective(expr=sum(m.x[i]*pyo.log(m.x[i]) for i in m.i))
```

```
>>> print(pyomo_pounce.preflight(m))
pyomo-pounce preflight — starting-point check
  model      : 7 vars, 7 constraints
  unset vars : 7 (become 0 in the .nl file)  e.g. x[1], x[2], x[3], y[1], y[2]
  objective  : NOT EVALUABLE
  bounds     : 1 violated, 6 on-bound (the solver's interior clamp moves these)
      V: value 0.0 outside [0.01, 0.99] by 1.000e-02
  constraints: 4 violated at the start (max 1.000e+00), 0 not evaluable
  warning: 7 variable(s) have no value and will be written as 0 in the .nl file;
           set Var values (or run initialize_missing_values / block_initialize)
           before solving
  warning: the model does not evaluate at its starting point (as written,
           unset values = 0); a POUNCE solve would abort with
           Invalid_Number_Detected
  VERDICT: FATAL
```

Solved as written, and then after one line of initialization:

| start | solves run | iterations | status | objective |
|---|---:|---|---|---:|
| as written (the origin) | 3 | 25, 25, 22 | `Infeasible_Problem_Detected` | — |
| `initialize_missing_values(m)` | 1 | 4 | `Solve_Succeeded` | -1.0407048079 |

Seventy-two iterations across three solves spent proving a feasible
model infeasible, against four iterations to the answer. The
initialization that fixes it is not clever — it sets every unset mole
fraction to `1.0` and the vapour fraction to `0.5`, neither of which is
feasible or even sensible. It is merely *not the origin*.

And the second-opinion ladder this page opens with is this model: on
0.10.0, three trajectories (the base solve and the two rungs it had), three
agreeing infeasibility verdicts, one false conclusion, because all three
started in the same place. Later versions add rung 3,
`start_point_perturbation=1e-2`, which displaces the origin by a relative
`1e-2` before re-solving. That is exactly the kind of start it was built
for (the origin is structurally degenerate here), but **the numbers in the
table above were not re-measured with it** — they are the 0.10.0 two-rung
run — so whether rung 3 alone would have rescued this model is not claimed
here. Fixing the start (`initialize_missing_values`) remains the one-line
answer and it costs four iterations.

To reproduce the CLI transcripts, write the model out:

```python
m.write("flash.nl", io_options={"symbolic_solver_labels": True})
```

```sh
pounce check-x0 flash.nl          # VERDICT: FATAL
pounce flash.nl --debug-on-error  # then type `diagnose`, `print rank`
```

## What is measured here, and what is not

> **Much of this page is mechanism, not evidence.** The numbers in the
> [scaling](#a-start-that-poisons-the-scaling),
> [bound-push](#the-point-you-gave-is-not-the-point-it-starts-from) and
> [flash](#a-feasible-model-reported-infeasible) sections were measured
> on POUNCE 0.10.0 on one host, cross-checked between the CLI and the
> Python frontend, and against an analytic optimum where one exists.
> Rows in the atlas marked *(unmeasured)* were not, and are marked so
> because saying "this should help" and "this was shown to help" in the
> same voice is how a page like this goes stale.

Two standing cautions, both of which have cost real measurement time
here:

**Pin the trajectory before you measure anything.** With the
second-opinion ladders on by default, an option comparison on
problems that struggle is a comparison of two configurations reported
as one.

**Get a correctness oracle.** Status is not one. Both entropy runs
above reported `Solve_Succeeded`, and one of them was eight orders of
magnitude down-scaled.

## See also

- [Initialization and Warm Starts](https://jkitchin.github.io/pounce/initialization.html) — the mechanism
- [Troubleshooting Recipes](https://jkitchin.github.io/pounce/troubleshooting.html) — recipes by symptom
- [Scaling](https://jkitchin.github.io/pounce/scaling.html) — why a point sample can mislead
- [Interactive Debugger](https://jkitchin.github.io/pounce/debugger.html) — `diagnose`, `print rank`, `multistart`
- [Finding Multiple Minima](https://jkitchin.github.io/pounce/find-minima.html) — when one start is not enough
- [Continuation](https://jkitchin.github.io/pounce/continuation.html) — homotopy from a problem you can solve
- [Pyomo](https://jkitchin.github.io/pounce/pyomo.html) and [Python API](https://jkitchin.github.io/pounce/python.html) — the frontends
