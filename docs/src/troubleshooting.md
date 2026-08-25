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
| `Search Direction is becoming Too Small` early in the iter table | [Ruiz linear-system scaling](#ruiz-scaling-on-the-augmented-kkt-system), then [μ-strategy switch](#mu-strategy-monotone-vs-adaptive) |
| Restoration phase fires repeatedly | [ℓ₁ exact-penalty wrapper](#l1-exact-penalty-barrier-wrapper) |
| Iterates wander on an LP-like / linearly constrained problem | [`mehrotra_algorithm=yes`](#mehrotra-predictor-corrector) |
| Hundreds of iterations, monotone μ stair-steps slowly toward optimal | [`mu_strategy=adaptive`](#monotone-vs-adaptive) |
| Iter count looks fine but seconds-per-iter is dominated by the linear solve on a hard QCQP / banded problem | [sweep `feral_ordering`](#feral-ordering-when-the-adaptive-dispatcher-guesses-wrong) (check the factorization share first — the wins are rarer than the symptom) |
| `alpha_pr` halves toward `1/128` while `\|\|d\|\|` grows and the dual residual stalls | [`feral_singular_pivot_floor`](#feral_singular_pivot_floor-a-reduced-hessian-that-collapses-to-singular) |

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
`accept_every_trial_step=yes`, `alpha_for_y=bound_mult`, larger
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

Before shipping a local-infeasibility verdict the CLI re-solves the
problem along up to two *different* trajectories and only keeps the
verdict if they agree. You will see this in the log:

```
EXIT: Converged to a point of local infeasibility. Problem may be infeasible.
pounce: local infeasibility — re-solving along 2 different trajectories before
        believing it (second-opinion ladder: feral_scaling=mc64,
        mu_strategy=adaptive).
pounce: second opinion — re-solving with feral_scaling=mc64…
pounce: feral_scaling=mc64 re-solve did not recover (InfeasibleProblemDetected).
pounce: second opinion — re-solving with mu_strategy=adaptive…
pounce: mu_strategy=adaptive re-solve recovered the problem — promoting (SolveSucceeded).
Status: Solve_Succeeded
```

Note the trailing `Status:` line. Each rung prints its own `EXIT:` banner,
so a laddered run has several and only the last one is the verdict that
shipped — if you are parsing pounce's output, read `Status:` and ignore the
banners. It carries the upstream IPOPT enumerator spelling
(`Infeasible_Problem_Detected`, `Maximum_Iterations_Exceeded`, …).

The two rungs probe different things, and the distinction matters when
you are reading a log:

| rung | option | varies |
|---|---|---|
| `feral_scaling=mc64` | `feral_infeasibility_scaling_retry` | the linear algebra |
| `mu_strategy=adaptive` | `infeasibility_mu_strategy_retry` | the barrier trajectory |

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

Things worth knowing:

- A rung is promoted only if it returns `Solve_Succeeded` /
  `Solved_To_Acceptable_Level`, so an overturned verdict always comes
  with a point that passed the ordinary convergence check.
- Rungs are applied to your baseline options, not stacked on each
  other, and a rung that would change nothing (you already set
  `mu_strategy=adaptive`) is skipped.
- The extra solves are spent only on runs that would otherwise report
  failure. Nothing changes on a successful solve.
- Both rungs are on by default; set them to `no` for upstream IPOPT's
  behaviour of shipping the first verdict.
- If a rung recovers the problem, that is a signal about your model as
  well as about the solver: the verdict was trajectory-dependent, so
  the starting point or the scaling of the formulation is worth a look.

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

Pair with `ma57_automatic_scaling=yes` (default in HSL builds) and
leave `linear_system_scaling=none` — MA57's internal scaling and a
pounce-level Ruiz pass should not be stacked.

### FERAL ordering: when the adaptive dispatcher guesses wrong

When `linear_solver=feral` (the default) and per-iter wall time is
dominated by the linear solve — typical on dense / quadratically-
coupled KKT systems where iteration counts look reasonable but
seconds-per-iter are high — the fill-reducing ordering choice can
matter more than any other knob. By default, `feral_ordering=auto`
picks AMD / AMF / METIS from cheap pattern features. This is right
in the common case but can miss badly on a single hard problem.

**"The linear solve dominates" is not by itself the trigger.** On the
Mittelmann NLP suite `LinearSystemFactorization` is 44–98 % of
`OverallAlgorithm` on 36 of 47 instances, and on almost all of them
every ordering is within the noise of `auto`
([gh#768](https://github.com/jkitchin/pounce/issues/768)). The wins are
concentrated where the share is *extreme* and the fill is *bad*, so
spend two cheap checks before the sweep:

```
pounce problem.nl print_timing_statistics=yes --json-output run.json
```

1. **Factorization share**, from the timing block:
   `LinearSystemFactorization` against `OverallAlgorithm`. Merely
   dominant (say 50–80 %) rarely pays; the instance that gave 3.08×
   below sits at 96 %.
2. **How bad `auto`'s ordering actually is**, from
   `linear_solver.max_fill_ratio` and `linear_solver.last_nnz_l` in
   `run.json`. A fill ratio near 2 means there is very little to win —
   the same instance that gave 3.08× has a ratio of 33.9 and 63.8 M
   nonzeros in `L`.

If both point the right way, sweep the **concrete** variants and pin the
winner — a capped `max_iter` is enough to rank them, because the
ordering is chosen once per pattern:

```
for o in auto amd amf metis scotch kahip; do
  pounce problem.nl max_iter=50 feral_ordering=$o \
    print_timing_statistics=yes --no-sol
done
```

On `qssp180` (96 % factorization, fill ratio 33.9) that ranking holds up
on the full solve: 45.06 s for the default against 14.63 s for
`feral_ordering=amd` — 3.08× — and 23.6 s for `metis`. Confirm the
winner with one uncapped run before pinning it in a script.

**Why the sweep and not `feral_ordering=auto_race`.** The race runs
symbolic factorization on AMD, METIS, SCOTCH and KaHIP and keeps the
smallest `factor_nnz`, for ~4× a single symbolic pass. That reads like
the safe way to measure, and it is not: *fill is a proxy for wall clock
that fails on exactly the structured problems where the ordering
matters*. On `robot_a`, METIS and KaHIP carry ~1.9 % more nonzeros in
`L` and factor 25–31 % faster, so the race keeps AMD — the loser — and
spends 4.06 s of symbolic — against AMD's 0.56 s — to get there. On `qssp180` it lands at 1.57×
where the `amd` pin gives 3.08×. And on the whole Mittelmann suite it is
the worst of the ten options screened: 27 of 42 instances slower, 1
faster, median 0.82×. It is useful once, as a diagnostic that reports
how much the four methods disagree about fill — not as a setting to
leave on.

See the full [`feral_ordering` table](options.md#feral_ordering-variants)
for the per-variant regimes.

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
