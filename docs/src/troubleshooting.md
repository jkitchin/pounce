# Troubleshooting Recipes

When a pounce solve fails, stalls, or settles for "acceptable" instead
of "optimal", the default options aren't always the best fit. This
page collects concrete, reproducible recipes that turn failures into
successes (or improve already-successful solves) on real problems.

Each entry follows the same shape:

- **When to try it** — symptoms in the iter table or the final report
  that point to this knob.
- **The knob** — exact option(s) and CLI invocation.
- **Worked example** — before/after table on a named problem so you
  can verify the recipe reproduces on your machine.

A recipe earns a place on this page when there's a *named problem
where it demonstrably helps*. "Should help in theory" entries belong
in the reference pages
([Scaling](scaling.md), [FBBT](fbbt.md), [Options](options.md)), not
here. If you find a new win, the contribution guide
([CONTRIBUTING.md](https://github.com/jkitchin/pounce/blob/main/CONTRIBUTING.md))
walks through adding it.

## Quick lookup by symptom

| Symptom | Recipe |
|---|---|
| Exit "Solved To Acceptable Level" but you need strict optimality | [Ruiz linear-system scaling](#ruiz-scaling-on-the-augmented-kkt-system) |
| Hundreds of small steps, slow convergence on a problem with loose bounds | [FBBT on nonlinear constraints](#fbbt-feasibility-based-bound-tightening) |
| `Search Direction is becoming Too Small` early in the iter table | [Ruiz linear-system scaling](#ruiz-scaling-on-the-augmented-kkt-system), then [μ-strategy switch](#monotone-vs-adaptive) |
| Restoration phase fires repeatedly | [ℓ₁ exact-penalty wrapper](#restoration--ℓ₁-exact-penalty-wrapper) |
| Iterates wander on an LP-like / linearly constrained problem | [`mehrotra_algorithm=yes`](#mehrotra-predictor-corrector) |
| Hundreds of iterations, monotone μ stair-steps slowly toward optimal | [`mu_strategy=adaptive`](#monotone-vs-adaptive) |
| Collocation, optimal-control or PDE-in-time model that is slow per iteration | [`feral_ordering=metis`](#feral-ordering-when-the-adaptive-dispatcher-guesses-wrong) |
| Iter count looks fine but seconds-per-iter is dominated by the linear solve on a hard QCQP / banded problem | [`feral_ordering=auto_race`](#feral-ordering-when-the-adaptive-dispatcher-guesses-wrong) |
| `alpha_pr` halves toward `1/128` while `\|\|d\|\|` grows and the dual residual stalls | [`feral_singular_pivot_floor`](#feral_singular_pivot_floor-a-reduced-hessian-that-collapses-to-singular) |
| `Infeasible_Problem_Detected` on a model you believe is feasible | [the second-opinion ladder](#the-second-opinion-ladder-what-those-extra-solves-in-your-log-are), then [what POUNCE says about the start](#what-pounce-says-when-it-stops-from-a-degenerate-point) |
| `Invalid_Number_Detected` with no indication of which number | [what POUNCE says about the start](#what-pounce-says-when-it-stops-from-a-degenerate-point) |
| Fails from the bundled start, solves from a hand-picked one | [conditioning the starting point](initialization.md#conditioning-the-starting-point) |

---

## Presolve: bound-tightening and row drops

### `presolve=yes` (start here)

The pounce presolve pipeline drops fixed variables, propagates bounds
from linear rows, detects empty / redundant constraints, and warm-starts
bound multipliers. It is **off by default** to match upstream Ipopt's
no-surprises behavior; turn it on for any non-trivial NLP.

```
pounce problem.nl presolve=yes
```

Cheap, almost always helpful, and a prerequisite for FBBT.

### FBBT (feasibility-based bound tightening)

Interval propagation through the nonlinear constraint DAG to discover
variable bounds the user did not write down (`x² + y² ≤ 1` ⇒
`x ∈ [-1, 1]`, `exp(x) ≤ 10` ⇒ `x ≤ ln 10`, etc.). Full reference
in [Feasibility-Based Bound Tightening](fbbt.md).

**When to try it.** Hundreds of small steps in the iter table, the
primal infeasibility stuck against a bound, or a problem that's
clearly under-constrained from the modeler's side. Requires a
structural-expression representation, which today means an `.nl`
input.

**The knob.**

```
pounce problem.nl presolve=yes presolve_fbbt=yes
```

**Worked example — `clnlbeam`** (Mittelmann):

|                       | `presolve=yes` | `+ presolve_fbbt=yes` |
|---                    |---             |---                    |
| Exit status           | Optimal Solution Found | Optimal Solution Found |
| Iterations            | 552            | **65**                |
| Wall time             | 41.4 s         | **8.2 s**             |

FBBT discovers tight nonlinear bounds the linear sweep missed; the
IPM then has a much smaller feasibility gap to close and converges
in roughly one-eighth the iterations.

Not every problem benefits. On `corkscrw` and `arki0003` FBBT
produces no measurable change or a slight regression — the
infrastructure is cheap (one pass per constraint per outer sweep,
capped at `fbbt_max_iter=10`), so the worst case is a few percent of
extra presolve time.

## Scaling

Full reference in [Scaling](scaling.md). The two layers are
independent.

### Ruiz scaling on the augmented KKT system

**When to try it.** Exit status is "Solved To Acceptable Level" with
small step sizes near the end, or `dual_inf` plateaus several orders
above `tol` while primal feasibility is already at machine epsilon.
That pattern signals a poorly-conditioned KKT augmented matrix — the
back-solve loses the last few fractional digits the convergence check
needs.

**The knob.**

```
pounce problem.nl presolve=yes linear_system_scaling=ruiz \
       linear_scaling_on_demand=no
```

`linear_scaling_on_demand=no` forces always-on Ruiz; the default
(`yes`) defers scaling until the linear solver flags an iterate as
poorly scaled. For diagnostic runs, force it on.

**Worked example — `nql180`** (Mittelmann):

|                          | default | `+ linear_system_scaling=ruiz` |
|---                       |---      |---                             |
| Exit status              | Solved To Acceptable Level | **Optimal Solution Found** |
| Iterations               | 41      | 50                             |
| Primal infeasibility     | 4.0e-11 | **1.2e-15**                    |
| Dual infeasibility       | 1.0e-5  | 3.1e-4                         |
| Complementarity          | 1.2e-9  | 9.9e-10                        |
| Overall NLP error        | 2.4e-7  | **9.9e-10**                    |

Symmetric ∞-norm equilibration improves primal feasibility by four
orders of magnitude and overall NLP error by ~3 orders, letting the
solver clear the strict `tol` gate. The extra nine iterations are
well spent. Resolves [issue #25](https://github.com/jkitchin/pounce/issues/25).

**Worked example — `WM_CFy`** (Mittelmann ampl-nlp, n=8709, m=12850):

|                       | default | `+ linear_system_scaling=ruiz` |
|---                    |---      |---                             |
| Exit status           | Optimal Solution Found | Optimal Solution Found |
| Iterations            | 605     | **241**                        |
| Wall time             | ~2300 s | **~543 s**                     |
| Overall NLP error     | 3.4e-9  | 2.6e-9                         |

A 4× wall-time speedup on a problem that previously sat in the "hard
W-B" bucket: every Ipopt + linear-solver combination tried in
[issue #29](https://github.com/jkitchin/pounce/issues/29) had failed
to converge within a 600 s budget. Ruiz wasn't just an iteration-count
win — at 605 iters / 2300 s default-pounce was the only configuration
that even *finished*; Ruiz cuts that to under ten minutes. Same
underlying mechanism as `nql180`: the augmented KKT system is
ill-conditioned enough that the back-solve burns iterations chasing
residuals symmetric ∞-norm equilibration fixes in one preconditioning
pass.

Pairing `mu_strategy=adaptive` with Ruiz on this problem solves to a
~50× tighter NLP error (5e-11) but takes twice as long (491 iters,
1100 s). For a tighter solution at any cost, use both; for a fast
solve, Ruiz alone wins.

### NLP-level scaling: when the default hurts

The gradient-based default at the NLP level is computed *once* at
`x_0` and is sometimes the wrong fingerprint of the problem — for
instance when the starting point lives near a flat region of the
objective. If the IPM stalls with no clear infeasibility and the
unscaled gradients in the report look reasonable, try turning NLP
scaling off:

```
pounce problem.nl nlp_scaling_method=none
```

Or, if you know the natural units of your problem better than the
solver does, supply `user-scaling` (see [Scaling](scaling.md) for the
end-to-end recipe).

## μ-strategy

### Monotone vs. adaptive

**Monotone** (the default) decreases the barrier parameter μ in
geometric steps; **adaptive** uses a quality-function oracle to pick
each new μ based on the current iterate's complementarity. Adaptive
is more aggressive in well-conditioned regions and more conservative
near degeneracy.

**When to try it.** Convex or nearly-convex problems where the
monotone schedule wastes iterations stair-stepping toward a μ that
the iterate clearly accepts; alternately, ill-conditioned problems
where monotone overshoots and triggers restoration.

**The knob.**

```
pounce problem.nl mu_strategy=adaptive
```

Pair with `mu_oracle=quality-function` (the default) or
`mu_oracle=probing` for the Mehrotra-style affine probe.

**Worked example — `arki0009`** (Mittelmann):

|                       | `mu_strategy=monotone` (default) | `mu_strategy=adaptive` |
|---                    |---                               |---                     |
| Exit status           | Optimal Solution Found           | Optimal Solution Found |
| Iterations            | 358                              | **108**                |

A 70 % iteration-count reduction with no quality regression. The
quality-function oracle picks larger μ-decrements when the
complementarity gap is well-balanced, skipping the slow stair-step
that monotone is forced into on this instance.

`nql180` is also rescued by `mu_strategy=adaptive` alone
(Acceptable → Optimal in 61 iters) — so for that problem you have a
choice between the Ruiz recipe (above) and the adaptive-μ recipe.
Ruiz gives a numerically cleaner solution (primal infeasibility
1.2e-15 vs ~5e-12); adaptive μ is one knob instead of two and has no
linear-system overhead.

### Mehrotra predictor-corrector

For problems that are LP-like (linear or mildly nonlinear constraints,
quadratic objective), the Mehrotra predictor-corrector mode
short-circuits the filter line search and accepts every trial step:

```
pounce problem.nl mehrotra_algorithm=yes
```

This sets a Mehrotra-canonical configuration (`adaptive_mu_globalization=never-monotone-mode`,
`accept_every_trial_step=yes`, `alpha_for_y=bound-mult`, larger
`bound_push` and `bound_mult_init_val`). On well-conditioned LP-like
problems it routinely cuts iteration counts in half. On nonconvex
NLPs it can destabilize — see
[issue #58](https://github.com/jkitchin/pounce/issues/58) for the
trade-off discussion.

## Restoration & ℓ₁ exact-penalty wrapper

When restoration fires repeatedly, the standard IPM is stuck on an
infeasible subproblem the filter cannot accept. The ℓ₁ exact-penalty
wrapper rephrases the constraints as an additive penalty term and
solves a sequence of bound-constrained subproblems instead:

```
pounce problem.nl l1_exact_penalty_barrier=yes
```

Or, only invoke the wrapper as a fallback when standard restoration
fails:

```
pounce problem.nl l1_fallback_on_restoration_failure=yes
```

This is the recipe for problems with rank-deficient constraints,
ill-defined bounds at the starting point, or pathological LICQ
violations — anywhere the filter's history rules out feasibility
restoration paths the wrapper can still find.

### Worked example: certifying genuine infeasibility

The built-in `infeasible-eq` problem is the smallest fixture that
exercises the fallback end-to-end:

```text
min  x0^2 + x1^2
s.t. x0 + x1 = 1     (g0)
     x0 + x1 = 2     (g1)
```

The two equalities are mutually contradictory, so no `x` exists with
`||g(x)||_∞ = 0`. The standard solve diagnoses this without the
wrapper:

```
$ pounce --problem infeasible-eq
...
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
```

That message is the filter giving up: it found an iterate where the
constraint gradients are linearly dependent and no admissible step
reduces infeasibility further. The output does not tell you whether
the problem is *genuinely* infeasible or whether the filter rejected
a feasible neighborhood that another method could reach. Re-run with
the wrapper to find out:

```
$ pounce --problem infeasible-eq l1_fallback_on_restoration_failure=yes
iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) ...
   0  0.0000000e+00 2.00e+00 0.00e+00   -1.0 0.00e+00     -  ...
   1  1.1250000e+00 5.00e-01 4.22e-09   -1.0 7.50e-01     -  ...
   2r 1.1250000e+00 5.00e-01 9.99e+02   -0.3 0.00e+00     -  ...   ← restoration
...
iter      objective   inf_pr   inf_du lg(mu)    ||d|| lg(rg) ...   ← second inner solve
   0  3.0202000e+00 9.90e-03 0.00e+00   -1.0 0.00e+00     -  ...
...
   6  1.5000000e+00 2.22e-16 2.53e-14   -8.6 1.88e-06     -  ...   ← wrapper converges
                                                                     in the slacked
                                                                     problem
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
```

Read this trace carefully. The wrapper's inner solve **converges** to
KKT tolerance on the *slacked* problem — `inf_pr` falls to 1e-16 in
six iterations because the added slack variables `s+, s-` absorb the
inconsistency `g0 ≠ g1`. But pounce reports the overall verdict on
the *original* constraints, so the final `Constraint violation = 0.5`
is unchanged: that's the irreducible gap `(g1 − g0)/2`. Two
independent solvers (filter IPM and ℓ₁-penalty barrier) landing on
the same least-infeasible iterate, from different starting strategies,
is what makes this an *infeasibility certificate* rather than a
diagnosis of solver fragility.

The recipe in plain English:

- **Standard solve says "local infeasibility"** → may or may not be a
  real obstruction; could be filter history, LICQ degeneracy, or a
  bad starting point.
- **Wrapper agrees on the same least-infeasible iterate** → trust the
  certificate; reformulate the model.
- **Wrapper promotes to `Solve_Succeeded`** → the standard filter was
  rejecting a feasible neighborhood it could not reach; the model
  itself is fine.

> **Implementation note** — running this case used to panic with
> `restoration factory invoked more than once` because the CLI wired
> a one-shot restoration factory into the application. The fix
> ([pounce#24](https://github.com/jkitchin/pounce/issues/24)) routes
> through a multi-pass *provider* so the wrapper can mint a fresh
> restoration phase per inner solve. The regression test that guards
> it (`crates/pounce-cli/tests/l1_fallback_no_panic.rs`) uses this
> same `infeasible-eq` builtin.

### The second-opinion ladder (what those extra solves in your log are)

Before shipping a local-infeasibility verdict POUNCE re-solves the
problem along up to four *different* trajectories and only keeps the
verdict if they agree. This is not a CLI feature — see
[The ladder is not a CLI feature](#the-ladder-is-not-a-cli-feature)
below — but the CLI is where you see it narrated:

```
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
pounce: local infeasibility — re-solving along 3 different trajectories before
        believing it (second-opinion ladder: feral_scaling=mc64,
        mu_strategy=adaptive, start_point_perturbation=1e-2).
pounce: second opinion — re-solving with feral_scaling=mc64…
pounce: feral_scaling=mc64 re-solve did not recover (InfeasibleProblemDetected).
pounce: second opinion — re-solving with mu_strategy=adaptive…
pounce: mu_strategy=adaptive re-solve recovered the problem — promoting (SolveSucceeded).
Status: Solve_Succeeded
```

An `Invalid_Number_Detected` opens the ladder too, but reaches only the third
rung. A NaN out of your model is a statement about the *callbacks* at a point;
re-running the same callbacks at the same point under a different
linear-solver scaling or a different barrier strategy evaluates the same
non-finite quantity again, so those two rungs are not evidence about it and
would only burn solves.

`Restoration_Failed` opens it as well, and reaches rungs 3 and 4
([pounce#815](https://github.com/jkitchin/pounce/issues/815),
[pounce#857](https://github.com/jkitchin/pounce/issues/857)). Restoration
failing is a report about the *path*: the iterate reached somewhere the
restoration sub-problem could not work from. Rungs 1 and 2 vary the path from
the same starting point and can arrive somewhere just as bad; rung 3 moves the
point, which makes it a different sub-problem. Note that this is **not** a
budget exit — a restoration failure typically stops far short of `max_iter`,
so "give it more iterations" is not the available answer.

`Maximum_Iterations_Exceeded` reaches **rung 4 only**, and only when the solve
actually escalated its factorization. A budget exit is normally not evidence of
anything except a small budget, and the honest answer to it is a bigger budget,
which is why no other rung opens on it. The exception is narrow and measured:
`feral_increase_quality` reroutes the trajectory when the linear solver's
refinement stalls, and where that reroute is what walked the solve into the
wall, a bigger budget just re-runs the same wall.
`square_flowsheet_resto` under `hessian_approximation=limited-memory` is the
case — 3000 iterations at the cap with the escalation, 178 and `Optimal`
without it. The gate is the `quality_escalations` statistic: a solve that never
escalated is provably not a candidate and opens no rung at all, so this is not
a blanket extra solve on every capped run. Turn it off with
`feral_increase_quality_retry=no`, which holds a capped run to exactly the
budget it was given.

Rung 4 also opens on `Infeasible_Problem_Detected`, under the same escalation
gate, and for the same reason read one step further: the reroute can produce a
*false* infeasibility verdict, which is worse than a budget exit because it is
a wrong answer on a feasible model reported as a verdict rather than as a
failure. The same `square_flowsheet_resto` limited-memory leg exits that way on
linux/x86_64 — identical iteration count and identical escalation count,
different verdict — and rungs 1–3 all fail to rescue it. On a model that really
is infeasible the rung cannot recover anything and the extra solve only
confirms the verdict, which is a real cost paid on purpose; the escalation gate
is what limits it to the runs where the escalation is a candidate explanation.

Note the trailing `Status:` line. Each rung prints its own `EXIT:` banner,
so a laddered run has several and only the last one is the verdict that
shipped — if you are parsing pounce's output, read `Status:` and ignore the
banners. It carries the upstream IPOPT enumerator spelling
(`Infeasible_Problem_Detected`, `Maximum_Iterations_Exceeded`, …).

The specialized convex engines (LP / QP interior-point, the parametric
active-set QP engine, and the conic QCQP engine) print the same `EXIT:` block
and the same `Status:` line, in the same spelling, so a parser needs no
convex-specific case. If you are reading the JSON report rather than the log,
compare against `solution.status_upstream`, which carries that spelling;
`solution.status` is the Rust enum-variant name (`Solve_Succeeded` vs
`SolveSucceeded`) and does not match IPOPT's tables.

The four rungs probe different things, and the distinction matters when
you are reading a log:

| rung | option | varies |
|---|---|---|
| `feral_scaling=mc64` | `feral_infeasibility_scaling_retry` | the linear algebra |
| `mu_strategy=adaptive` | `infeasibility_mu_strategy_retry` | the barrier trajectory |
| `start_point_perturbation=1e-2` | `infeasibility_perturbed_start_retry` | where the trajectory starts |
| `feral_increase_quality=no` | `feral_increase_quality_retry` | whether a stalled factorization was allowed to reroute the trajectory |

Rung 4 is the odd one out in a second way: every other rung's gate is a
property of the options the failing solve ran under, so the ladder can be
assembled before the solve. Rung 4's gate is a *measurement of the solve that
just failed* — it needs `quality_escalations >= 1`, and an escalation leaves no
other trace, not in the status, the objective, the iteration count or the
engine.

**To switch the ladder off entirely you must name all four options** — there
is no single master switch, and each option's own text says "set to no to keep
behaviour bit-for-bit faithful to upstream IPOPT", which is true of that rung
and not of the solver:

```
feral_infeasibility_scaling_retry no
infeasibility_mu_strategy_retry   no
infeasibility_perturbed_start_retry no
feral_increase_quality_retry      no
```

Naming a subset leaves the remaining rungs live. This bit four of POUNCE's
own regression tests when rung 4 was added — three Rust ones that had used
`infeasibility_perturbed_start_retry=no` as shorthand for "no ladder", and
the Python `test_turning_the_whole_ladder_off_restores_upstream_behaviour`,
whose "whole ladder" was a dict of the other three.

The first rung is evidence only when the trajectory is
hypersensitive — two equally backward-stable scalings staying
bit-identical for many iterations, then diverging by ~1 ULP into
different basins (`discs.nl` is the canonical case). When it is not,
MC64 retraces the same iterates and agrees for the same reason the
first solve was wrong, so **the scaling rung agreeing is not by itself
a reason to believe the verdict**. That is why the barrier rung exists
([pounce#524](https://github.com/jkitchin/pounce/issues/524): CUTE
`cresc4` is feasible, Ipopt solves it in 71 iterations, and the MC64
re-solve reproduced the failing trajectory bit-identically).

The third rung changes neither of those — it changes the point the
trajectory starts from, by displacing each variable by a relative `1e-2`
and clipping back into its bounds. It is last because it is the biggest
change, and it exists because measurement said it is by far the most
effective. Over a 244-problem corpus taken from the
[KRONOS](acknowledgments.md#starting-point-conditioning-kronos) benchmark set, fifteen models failed from
their bundled start; ten of them are models an independent solver proves
feasible to `2.4e-7` or better, so the verdict was wrong. Of those fifteen:

| what was tried | recovered |
|---|---|
| nothing (the default) | 0 / 15 |
| `start_with_resto` | 0 / 15 |
| `expect_infeasible_problem` | 0 / 15 |
| `mu_strategy=adaptive` | 4 / 15 |
| a displaced start | **13 / 15** |
| a displaced start + restoration | 14 / 15 |

That ordering is the diagnosis. The iterate does not need to be *better*,
it needs to be *non-degenerate*. The common failure is a start at which
the constraint Jacobian is structurally rank-deficient — a squared slack
sitting at zero, or an origin start on a homogeneous quadratic — where
LICQ fails and the filter line search has no descent direction to find,
whatever you hand it. Displacing the point restores rank, and the solve
that follows is an ordinary one.

The displacement is deterministic: it is drawn from a SplitMix64 stream
seeded by `start_point_perturbation_seed` and nothing else — no clock, no
address, no thread identity — so a promoted retry reproduces and a failed
one is reportable. Non-finite entries in the starting vector are replaced
with a finite in-bounds value first, because NaN plus noise is NaN.

You can apply the same displacement yourself, without waiting for a
failure, with `start_point_perturbation 1e-2`; vary
`start_point_perturbation_seed` to drive a multistart by hand.

Things worth knowing:

- A rung is promoted only if it returns `Solve_Succeeded` /
  `Solved_To_Acceptable_Level`, so an overturned verdict always comes
  with a point that passed the ordinary convergence check.
- Rungs are applied to your baseline options, not stacked on each
  other, and a rung that would change nothing (you already set
  `mu_strategy=adaptive`) is skipped.
- The extra solves are spent only on runs that would otherwise report
  failure. Nothing changes on a successful solve.
- All three rungs are on by default; set them to `no` for upstream
  IPOPT's behaviour of shipping the first verdict.
- If a rung recovers the problem, that is a signal about your model as
  well as about the solver: the verdict was trajectory-dependent, so
  the starting point or the scaling of the formulation is worth a look.

#### The ladder is not a CLI feature

Every ordinary single-solve entry point runs it, on by default and with
the same three options: the CLI, the Python `Problem.solve`, the C
`IpoptSolve`, and the `pounce-rs` builder. If you drive POUNCE from a
modelling layer you are, if anything, the caller who needs it most — an
uninitialized decision variable reaches the solver as a zero, and the
origin is where a squared slack or a homogeneous quadratic loses rank.

Three entry points deliberately do **not** run it, and it is worth
knowing which, because on a model the ladder would have recovered they
report the failure that `Problem.solve` does not:

| Entry point | Why not |
|---|---|
| `solve_nlp_batch` | A failed start is routine in a multi-start; up to three extra solves per failed start multiplies the search cost for no benefit. |
| the CLI's `minima` global search | Same reason. |
| `Problem.solve_with_sens` | Sensitivity is taken *about a particular solution*. The third rung displaces the starting point, which on a multi-modal model can converge somewhere else entirely — and your `pin_constraint_indices` and `deltas` are posed against the solution you expected, so silently answering about a different local optimum is worse than reporting the failure. |

So `problem.solve(x0)` and `problem.solve_with_sens(x0, ...)` can
disagree about whether a model is solvable, and on a degenerate start
they will. If you want the ladder's starting point *and* sensitivity
about it, run `solve` first, then pass the `x` it returns back into
`solve_with_sens` as `x0` — that makes the choice of base point explicit
and reproducible, which is what sensitivity analysis wants anyway.
`info["second_opinion"]` is always `None` from `solve_with_sens`.

From Python, what the ladder did comes back in the info dict:

```python
x, info = problem.solve(x0)
so = info["second_opinion"]          # None if the ladder never ran
if so:
    print(so["tried"])               # e.g. ['feral_scaling=mc64', 'mu_strategy=adaptive']
    print(so["promoted_by"])         # the rung that was adopted, or None
    print("\n".join(so["log"]))     # the narration the CLI prints to stderr
```

`info["second_opinion"]` is `None` on the overwhelmingly common path —
the solve did not fail in a way the ladder second-guesses, and nothing
extra was spent. When it is not `None` and `promoted_by` is `None`, the
original verdict survived every rung, which is a much stronger statement
about your model than a single failed solve.

The narration is collected rather than printed for the library callers,
so an embedded solve does not write to someone else's stderr; the C
interface prints it, matching where the solver's own banners already go.

One place deliberately does **not** run it: the multi-start paths
(`solve_nlp_batch`, the CLI's `minima` global search). A failed start is
routine there, and up to three extra solves per failed start multiplies
the cost of a search for no benefit.

### What POUNCE says when it stops from a degenerate point

If the ladder runs out and the failure verdict stands, POUNCE audits
the starting point once — one evaluation of each callback, spent only
on a run that has already failed — and prints what it finds before the
machine-readable `Status:` line. There are two findings.

**The model is not finite where it starts.** All four
`Invalid_Number_Detected` cases in the corpus above were correct stops
reported unhelpfully: the solver said a number was invalid without
saying which one. Now it names it.

```
$ pounce nanstart.nl
pounce: invalid number — re-solving along 1 different trajectory before believing it
        (second-opinion ladder: start_point_perturbation=1e-2).
pounce: second opinion — re-solving with start_point_perturbation=1e-2…
pounce: start_point_perturbation=1e-2 re-solve did not recover (InvalidNumberDetected).
pounce: keeping the original Invalid_Number_Detected verdict; it survived
        1 independent re-solve(s) (start_point_perturbation=1e-2).
pounce: the model is not finite at its own starting point: objective f(x) = NaN.
Status: Invalid_Number_Detected
```

The audit covers `x` itself, `f`, `grad f`, `g` and the Jacobian, names
the offending index (and column, for a Jacobian entry), and reports the
value's sign for an infinity. It caps the list and counts the rest, so
a model that is non-finite everywhere prints a line, not a wall. One
corpus model, `hong`, ships a starting point that is literally
`[nan, nan, nan, nan, 0, …]` — worth knowing before you go looking for
a bug in the objective.

**The constraint Jacobian is rank-deficient there.** This is the one
that changes an answer rather than a message. A local-infeasibility
verdict reached from a point where LICQ fails is not evidence about the
problem:

```
$ pounce degen.nl
pounce: keeping the original Infeasible_Problem_Detected verdict; it survived
        3 independent re-solve(s) (feral_scaling=mc64, mu_strategy=adaptive,
        start_point_perturbation=1e-2).
pounce: the constraint Jacobian is rank-deficient there: 2 of 2 constraint rows
        have an identically zero gradient here (rows 0, 1); 2 of 2 variable
        columns are identically zero here (variables 0, 1).
pounce: LICQ fails at a point like that, so a local-infeasibility verdict reached
        from it is as much a statement about the starting point as about the
        problem. Try a different starting point, or `start_point_perturbation 1e-2`.
Status: Infeasible_Problem_Detected
```

Two caveats on how to read it. POUNCE reports *identically zero* rows
and columns, not a rank estimate — an SVD is not affordable to run
speculatively on every failed solve, and a zero row is the degeneracy
that actually shows up in practice (a squared slack at zero, an origin
start on a homogeneous quadratic). A full-rank-looking Jacobian can
still be numerically rank-deficient, so the absence of this line is not
a clean bill of health. And **structural** absence is never reported:
a column the model never declared is not a finding, only a column the
model declared and then evaluated to zero.

The audit runs on your own model — before presolve, elimination and
scaling — so the indices it prints are the ones in your file. That is
deliberate: a wrapper renumbers variables, and naming `x[3]` of a
presolved model would point at a neighbouring variable's answer.

### When the residual is small but the verdict still says infeasible

Some models cannot reach a small *absolute* residual no matter how well
they are solved. An ill-conditioned change of variables — a moving-boundary
PDE on a Landau coordinate, say — can leave a row carrying a coefficient
of `1e9`, so a residual of `1e-3` is eleven relative digits: the equation
is satisfied about as well as double precision allows, and no absolute
tolerance will ever be met. That is exactly the regime the acceptable-level
fallback exists for, and the exit you want is
`Solved_To_Acceptable_Level`.

Set `acceptable_tol` to a level you can actually reach, and read the
result there:

```
$ pounce model.nl -AMPL tol=1e-6 acceptable_tol=1e-3
```

Three things are worth knowing about how that interacts with the
infeasibility detector:

- `acceptable_constr_viol_tol` (default `1e-2`) is the feasibility band
  the acceptable-level exit uses, and it is **separate** from
  `constr_viol_tol`. Widening the latter does not widen the former.
- Tightening `constr_viol_tol` does **not** make POUNCE readier to call a
  model infeasible. The rapid-infeasibility detector's violation floor is
  clamped so it never convicts a point whose violation sits inside the
  band the defaults call acceptable
  ([pounce#519](https://github.com/jkitchin/pounce/issues/519)). If you
  are still seeing `Infeasible_Problem_Detected`, the point is outside
  the band you declared: compare the reported `Overall NLP error` against
  your `acceptable_tol`, and the `Constraint violation` against
  `acceptable_constr_viol_tol`.
- If the solve did pass through an acceptable iterate before giving up,
  that point is returned rather than discarded, whichever internal route
  reached the verdict
  ([pounce#505](https://github.com/jkitchin/pounce/issues/505)).

If the residual is large relative to its own row — not just in absolute
terms — the verdict is the honest one, and the ℓ₁ wrapper above is the
way to corroborate it.

## Linear solver choice

`linear_solver=ma57` (when built with HSL):

```
pounce problem.nl linear_solver=ma57
```

For problems that go many hundreds of iterations, the round-off chain
of the inner sparse factorization matters — MUMPS, FERAL/SSIDS, and
MA57 do not produce bitwise-identical iterates, and on the worst-case
instances the difference can be the difference between convergence
and a μ-reset spiral
([issue #58](https://github.com/jkitchin/pounce/issues/58),
[issue #64](https://github.com/jkitchin/pounce/issues/64)).

Consider pairing with `ma57_automatic_scaling=yes` and leaving
`linear_system_scaling=none` — MA57's internal scaling and a
pounce-level Ruiz pass should not be stacked. Note that
`ma57_automatic_scaling` defaults to `no`, matching upstream Ipopt;
turning it on is a deliberate step. (This page previously called it
"default in HSL builds", which was never true — and until the fix for
[issue #825](https://github.com/jkitchin/pounce/issues/825) setting it
either way had no effect, because no `ma57_*` option reached the
backend.)

### FERAL ordering: when the adaptive dispatcher guesses wrong

When `linear_solver=feral` (the default) and per-iter wall time is
dominated by the linear solve — typical on dense / quadratically-
coupled KKT systems where iteration counts look reasonable but
seconds-per-iter are high — the fill-reducing ordering choice often
matters more than any other knob. By default, `feral_ordering=auto`
picks AMD for very large, very sparse matrices and AMF for everything
else — it never picks nested dissection. That is right in the common
case and misses badly on one whole class.

**If the model is a collocation, optimal-control or PDE-in-time
transcription, set `metis`:**

```
pounce problem.nl feral_ordering=metis
```

Its KKT matrix is a space-by-time mesh, which is the shape nested
dissection exists for; with feral 0.18 this runs GasLib-40 transient
control 2.1× faster at 56k variables and 3.3× at 112k, and the
`laptime` collocation benchmark 2×. It is slower on other shapes, so it
is a per-model choice — the measurements, and where it loses, are under
[collocation models](options.md#collocation-optimal-control-and-pde-in-time-models-set-metis).

For any other hard model, *measure* rather than guess:

```
pounce problem.nl feral_ordering=auto_race
```

This runs symbolic factorization on AMD and METIS (feral ≥ 0.18)
and keeps the one with the smallest `factor_nnz`. Smallest fill is not
always fastest, so confirm its pick against a pinned run. Costs ~2× a
single symbolic pass — paid once per problem because symbolic factorization
is cached across numeric refactorizations with the same pattern, so
the overhead is invisible to the per-iter cost on anything but a
one-iter problem.

`feral_ordering=amd` (concrete pin) is the right escalation when the
race itself is showing AMD winning consistently — pinning skips the
race entirely on subsequent runs. See the full
[`feral_ordering` table](options.md#feral_ordering-variants) for the
other variants.

### `feral_singular_pivot_floor`: a reduced Hessian that collapses to singular

#### When to try it

`alpha_pr` walks down `1/2, 1/4, … 1/128` with a matching `ls` count,
`||d||` *grows* instead of shrinking, and the run exits with `dual_inf`
parked a couple of orders of magnitude above `tol` — or reaches `tol`
only after a long tail of tiny steps. Feasibility is usually already at
machine precision, and the objective is right to many digits; only the
dual residual will not come down. The `lg(rg)` column in that tail is
typically *churning* — small values re-escalating iteration after
iteration — rather than settling.

That combination means the reduced Hessian `Zᵀ W Z` has become
numerically singular, so the Newton step runs off along a direction
whose curvature is at the noise floor and the line search has no choice
but to cut the step to nothing. It shows up on problems whose solution
set is a manifold rather than a point — degenerate eigenvalue models
are the classic case — and it is not something the exit criteria can
fix, because the iterate handed to them is the problem.

Since [#544](https://github.com/jkitchin/pounce/pull/544) pounce already
handles the sharpest form of this automatically: when the KKT is
singular to working precision its inertia count is meaningless, and
`feral_inertia_pivot_floor` (default `n · eps` since
[#592](https://github.com/jkitchin/pounce/issues/592), where `n` is the
order of the factored KKT) routes that case to `δ_c` rather than
answering an unmeasurable test with `δ_w`. The recipe below
is for what remains — it attacks the same degeneracy higher up, capping
the null-direction step outright, and on some models that is still
markedly faster.

To confirm before reaching for the knob, dump the KKT systems and look
at the smallest pivot:

```
pounce problem.nl --dump kkt:all --dump-dir /tmp/dump-problem
```

#### The knob

```
pounce problem.nl feral_singular_pivot_floor=1e-8
```

FERAL force-accepts a pivot at the working-precision floor and still
reports a clean factorization with the right inertia. This option is
pounce's analog of MA57's `CNTL(2)`: after a successful factor the
smallest accepted D-block pivot is compared against the floor, and a
factor below it is reported singular so the perturbation handler
escalates `δ_w`. The default `1e-20` almost never fires — deliberately,
because on a *bounded* problem a tiny pivot usually comes from the
barrier blocks (`Σ_x = z/x` as a bound activates) and is both expected
and harmless. Raising it is a per-problem call, not a global default:
`airport`, `jit1` and `pooling_rt2stp` all converge to `Optimal` with
smallest pivots between `1e-12` and `1e-21`, and a `1e-8` floor would
flag every one of them.

Start at `1e-8` and back off toward `1e-10`/`1e-12` if the extra
factorizations cost more than they save.

#### Worked example: `eigenb2` (Vanderbei)

110 variables, 55 equality constraints, no bounds at all. `Zᵀ W Z`'s
smallest eigenvalue falls from `1.4e+02` at iteration 2 to `1.4e-11` by
iteration 36, against `‖W‖ ≈ 1.3e+02`. The KKT is singular to working
precision down that tail, so its negative-eigenvalue count stops being
measurable — FERAL reports anywhere from 43 to 64 against an expected
55.

Since #544 the default solve certifies `Optimal` (before it, this
exited `Solved To Acceptable Level` in 67 iterations). **Since #693 the
default is also the fastest route on this model, and the knob is no
longer worth reaching for here:**

| options | iterations | dual inf | exit |
|---|---|---|---|
| *(defaults)* | 21 | 2.71e-09 | Optimal Solution Found |
| `feral_singular_pivot_floor=1e-8` | 72 | 2.39e-08 | Solved To Acceptable Level |
| `feral_singular_pivot_floor=1e-8 mu_strategy=adaptive` | 86 | 1.77e-08 | Solved To Acceptable Level |
| `mu_strategy=adaptive` | 21 | 2.71e-09 | Optimal Solution Found |

For the record, on 0.10.0 the same four rows read 67 / 39 / 30 / 63
iterations, all `Optimal Solution Found`. #693 removed a Tikhonov
perturbation from the equality-multiplier initializer; `eigenb2`'s
default trajectory got three times shorter and the knob's inverted from
a speedup into a cost that also loses the certificate.

**So do not read this section as "try `feral_singular_pivot_floor=1e-8`
on a model like `eigenb2`".** More generally, do not read it as a
recommendation at all. It is a *gamble worth taking when you are already
stuck*, and the odds have now been measured rather than guessed.

#### What the knob is actually worth, across the corpus

The 110 hardest problems in the benchmark corpus — every one that either
exits non-`Optimal` with `dual_inf` above `tol`, or takes 100+ iterations
to certify — run with and without `feral_singular_pivot_floor=1e-8`:

| outcome | count |
|---|---|
| unchanged | 89 |
| rescues a failed or acceptable-level solve | 5 |
| ≥20% faster, both `Optimal` | 5 |
| **costs the certificate or the solve** | **7** |
| ≥25% slower, both `Optimal` | 4 |

Ten better, eleven worse. In aggregate the knob is a coin flip — but the
individual effects are large in *both* directions, which is what makes it
worth trying and worth measuring:

| the best cases | | the worst cases | |
|---|---|---|---|
| `britgas` | `Restoration Failed` @2748 → `Optimal` @54 | `twirism1` | `Optimal` @178 → `Optimal` @1679 |
| `ex9_1_1` | `Error In Step Computation` @99 → `Optimal` @27 | `palmer7e` | `Optimal` @1677 → hits the 3000 cap |
| `ssebnln` | `Error In Step Computation` @215 → `Optimal` @101 | `ncvxqp6` | `Optimal` @301 → `Error In Step Computation` @505 |
| `deconvu` | `Optimal` @321 → `Optimal` @95 | `scosine` | `Optimal` @129 → `Acceptable` @326 |

(38 further problems hit a wall-clock cap in one arm or the other and are
excluded rather than counted — they were measured 8-way parallel and the
cap says more about the machine than about the solver.)

Two things follow, and they are the practical advice:

1. **The characteristic failure mode is losing the certificate, not
   losing the answer.** Five of the seven regressions above are
   `Optimal → Solved To Acceptable Level`: the point is still right, the
   dual residual just parks an order of magnitude above `tol`. That is
   the same thing `eigenb2` now does. So after setting this knob,
   **check `dual_inf` against `tol` in the exit block** — a run that
   still looks fine may have quietly stopped certifying.
2. **It only pays when you are already losing.** All five rescues start
   from a failed or acceptable-level solve. Nothing in the corpus shows
   it turning a healthy `Optimal` run into a better one often enough to
   justify reaching for it speculatively — it made four healthy runs
   substantially slower over the same sample.

So: reach for it when the symptom at the top of this section is what you
are looking at, back it off from `1e-8` toward `1e-10`/`1e-12` if it does
not pay immediately, and check the certificate before you trust the
result. Do not carry it into a options file as a default.

The fixture is committed, so this reproduces without a benchmark corpus:

```
pounce crates/pounce-cli/tests/fixtures/eigenb2.nl \
       feral_singular_pivot_floor=1e-8
```

Full diagnosis in
`dev-notes/issue-541-eigenb2-degenerate-reduced-hessian.md`
([issue #541](https://github.com/jkitchin/pounce/issues/541)).

## Diagnosing before you reach for a knob

Before trying recipes, dump the per-iter diagnostic categories that
pounce supports:

```
pounce problem.nl --dump kkt --dump iterate \
       --dump-dir /tmp/dump-problem
```

The dumps land as JSONL under `/tmp/dump-problem/`. Two categories
have wired dump sites today:

- `--dump kkt` — KKT residuals and condition-number proxy; large
  values motivate [Ruiz scaling](#ruiz-scaling-on-the-augmented-kkt-system).
- `--dump iterate` — primal/dual values; needed to spot whether a
  small step is bound-snapping or infeasibility-driven.

> The `--dump mu` and `--dump resto` categories are accepted by the CLI
> but not yet wired to a dump site, so they currently emit no data. For
> the μ trajectory and restoration entries/exits, use the Studio queries
> below (which read the iteration stream from the solve report).

The Studio MCP (`pounce-studio`) wraps these dumps in higher-level
diagnostic queries (`diagnose`, `find_stalls`, `restoration_windows`),
which is the recommended workflow when iterating on options.

## Logs, colors, and machine-readable output

POUNCE routes diagnostics through [`tracing`](https://docs.rs/tracing).
The knobs are environment variables (see
[Options › Logging and colored output](options.md#logging-and-colored-output)),
not solver options.

### When to try it
- You want more detail than the iteration table shows (which phase fired,
  why restoration triggered, linear-solver fallbacks).
- A downstream tool (Studio, CI) needs to parse per-iteration data.
- Color is garbling a log file, or you want color forced through a pipe.

### The knobs

| Goal | Invocation |
|---|---|
| Verbose, everything | `RUST_LOG=debug pounce problem.nl` |
| Just the restoration phase | `RUST_LOG=pounce::restoration=debug pounce problem.nl` |
| Separate logs from results | `pounce problem.nl > result.txt 2> solve.log` |
| Plain text (no color) | `NO_COLOR=1 pounce problem.nl` |
| Force color through a pipe | `CLICOLOR_FORCE=1 pounce problem.nl | less -R` |
| Line-delimited JSON iterations | `POUNCE_LOG_FORMAT=json pounce problem.nl 2> iters.jsonl` |

Logs go to **stderr**; the iteration table, final summary, and `--dump`
output are program output on **stdout**. The colored table uses a
tiger/rust theme — restoration lines get a kind-dependent background and
the row text reddens as the step length `alpha` shrinks, so a stalling or
restoration-heavy solve is visible at a glance. When stdout is not a
terminal (or `NO_COLOR` is set) the table is emitted as plain text with
the same column layout.

### Subsystem debug gates

For output finer than `RUST_LOG=<target>=debug` gives on its own, several
subsystems have a `POUNCE_DBG_*` gate that switches on extra per-iteration
diagnostics (adaptive-μ oracle decisions, the quality-function σ sweep,
inertia-perturbation choices, restoration internals, KKT-matrix dumps, …).
Most emit at debug level, so pair the gate with the matching `RUST_LOG`
target. The full table — including which gate takes a value and which
prints straight to stderr — is in
[Options › Environment overrides](options.md#environment-overrides-feral-and-debug-gates).

## Contributing a new recipe

A recipe earns a place here when:

1. There is a **named, reproducible problem** where the recipe
   demonstrably helps. Mittelmann benchmark (`benchmarks/mittelmann/nl/`)
   is preferred but any committed `.nl` works.
2. The before/after numbers are captured at `print_level=3` or higher
   and pasted into the worked-example table.
3. The recipe is not a special case of an existing one. (If your
   problem needs three knobs together, write one entry; if your
   problem benefits from a knob already documented here, file a PR to
   add a second worked example under that entry.)

Open a PR adding to this file with the table populated. The
maintainer-side review checks that the numbers reproduce against the
current `main` and that the recipe really is a recipe — not a
problem-specific accident.
