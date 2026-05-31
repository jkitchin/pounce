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
| Iter count looks fine but seconds-per-iter is dominated by the linear solve on a hard QCQP / banded problem | [`feral_ordering=auto_race`](#feral-ordering-when-the-adaptive-dispatcher-guesses-wrong) |

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
seconds-per-iter are high — the fill-reducing ordering choice often
matters more than any other knob. By default, `feral_ordering=auto`
picks AMD / AMF / METIS from cheap pattern features. This is right
in the common case but can miss badly on a single hard problem.

The safe recipe is to *measure* the right ordering rather than guess:

```
pounce problem.nl feral_ordering=auto_race
```

This runs symbolic factorization on AMD, METIS, SCOTCH and KaHIP and
keeps the one with the smallest `factor_nnz`. Costs ~4× a single
symbolic pass — paid once per problem because symbolic factorization
is cached across numeric refactorizations with the same pattern, so
the overhead is invisible to the per-iter cost on anything but a
one-iter problem.

`feral_ordering=amd` (concrete pin) is the right escalation when the
race itself is showing AMD winning consistently — pinning skips the
race entirely on subsequent runs. See the full
[`feral_ordering` table](options.md#feral_ordering-variants) for the
other variants.

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
