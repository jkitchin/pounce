# `robot_a` / `robot_b` / `robot_c`: where the time actually goes

Investigation for [pounce#476](https://github.com/jkitchin/pounce/issues/476).
On the Mittelmann `ampl-nlp` run of 3 Aug 2026 these three instances were the
single largest contributor to POUNCE's scaled shifted geometric mean: ~2140 s /
2122 s / 2298 s against IPOPT's ~690 s and KNITRO's 1 s. They share the shape
**m ≫ n** — 52013 constraints on 1001 variables, ~52:1 — which `robot_1600`
(n ≫ m, 5 s) does not.

The issue asked to separate two candidate causes, iteration *count* and *cost
per iteration*, before optimizing either. They separate cleanly, and they belong
to two different comparisons:

- **Against IPOPT (3.1x) the whole gap is per-iteration cost.** The iteration
  count is not merely similar, it is *identical* — POUNCE reproduces Ipopt's
  iterates digit for digit (§1). The dominant component was an evaluation
  strategy that re-evaluates shared subexpressions once per reference (§3);
  fixing it took 1.28x, and condensing the KKT is worth perhaps the rest (§5).
- **Against KNITRO (2000x) essentially none of it is per-iteration cost.** Both
  POUNCE and IPOPT *stall*: the barrier parameter moves one level in 3000
  iterations, on 1e-4 steps with 10–15 backtracks (§4b). Converging costs order
  5000–10000 iterations. No amount of making an iteration cheaper closes that,
  and POUNCE's own active-set SQP does not currently help either (§4c).

§4b is the important one and it was not on the issue's checklist.

## Reproducing without AMPL

`benchmarks/mittelmann/gen_robot_nl.py` writes `robot_a.nl` directly from the
`.mod` source, emitting `V` segments for the model's `SUM`/`SUM1`/`SUM2` defined
variables. That fidelity is the whole point: inlining them (which a generic
`.nl` writer would do) changes the tape the solver builds and would have hidden
the finding below.

It reproduces the published sizes exactly — `n = 1001`, `m = 52013`,
`nzJ = 196781` — and IPOPT's published failure mode: 3000 iterations, no
convergence. (Matching `m` requires one AMPL presolve step by hand: the six
linear families `gx1/gx2/gx7/gx8/gx13/gx14` are all lower bounds on the *same*
linear form `SUM[i]`, so they collapse to one row per collocation point,
18·4001 → 13·4001 = 52013.)

All numbers below: Apple M-series, single instance at a time, POUNCE 0.9.0 with
FERAL, `ipopt` 3.14.19 with MUMPS 5.6.2, both reading the same `robot_a.nl`.

## 1. Not a basin artifact — the iterates are Ipopt's

`dev-notes/research/narx-cfy-iteration-count.md` closed a superficially similar
symptom as a basin-of-attraction artifact, so that was ruled out first by diffing
the two iteration logs column by column.

| iterations | what matches |
| --- | --- |
| 0–71 | **every printed column**: objective, `inf_du`, `lg(mu)`, `‖d‖`, `alpha_du`, `alpha_pr`, `ls` — all 8 digits |
| 72–112 | objective differs in the 7th–8th digit; every step-control column still exact |
| 113+ | step-control columns begin to differ; by iteration 176 the objectives are 41 % apart |

That is roundoff (FERAL vs MUMPS) amplified by a chaotic non-converging
trajectory, not a different algorithm. The same sensitivity shows up *within*
Ipopt: MUMPS reaches 8.526 at iteration 3000 here where the checked-in MA57
reference reaches 8.173.

**Conclusion:** POUNCE takes the same steps Ipopt takes, so it needs the same
number of them. There is no iteration-count win available by making POUNCE
*more* like Ipopt — the count is a property of the filter line-search method on
this model, and §4b shows it is the filter's acceptance test specifically.

## 2. Splitting the wall clock

With the iteration count settled, the ratio is per-iteration cost, measured over
200 iterations on the same file:

| | s / iteration | vs IPOPT |
| --- | --- | --- |
| IPOPT 3.14.19 + MUMPS | 0.138 | 1.0× |
| POUNCE 0.9.0 (before) | 0.270 | 1.96× |
| POUNCE (after this change) | 0.221 | 1.60× |

Main-thread profile (`sample`, 30 s window, before):

| share | what |
| --- | --- |
| 44 % | line search — **30 % constraint evaluation** (`eval_g`), 12 % second-order correction |
| 33 % | search direction (KKT solve) |
| 17 % | exact Hessian (`eval_h`) |
| 4 % | constraint Jacobian (`eval_jac_g`) |

`print_timing_statistics yes` after the change, same 200 iterations (42.2 s
total) — instrumented, and the number to plan against rather than the sampled
percentages above:

| seconds | share | |
| --- | --- | --- |
| 13.006 | 30.8 % | `LinearSystemFactorization` |
| 8.559 | 20.3 % | `LinearSystemBackSolve` |
| 9.342 | 22.1 % | `LagrangianHessianEvaluations` |
| 4.294 | 10.2 % | `ConstraintEvaluations` |
| 2.452 | 5.8 % | `ConstraintJacobianEvaluations` |

So the linear solver is 51 % of the run split roughly 60/40 between factorizing
and solving, and AD is 38 %. `eval_g` is the inner loop:
the line search averages ~10 trial points per iteration on this model (Ipopt's
own counters agree: 35398 constraint evaluations over 3000 iterations).

## 3. The finding: shared subexpressions evaluated once per *reference*

`NlTnlp` splits each constraint into summands and builds an independent flat
`Tape` per summand. `Tape::build` deduplicates CSE bodies by `Arc` identity
*within one tape*, so a `.nl` defined variable referenced from many summands is
re-emitted — and re-evaluated — once per reference. The construction site said
as much: "bodies shared across summands are duplicated, which we accept as a
simplicity tradeoff".

That tradeoff is invisible on most models and brutal on this shape. `robot_a`
has 12003 defined variables (`SUM[i]`, `SUM1[i]`, `SUM2[i]` — the cubic-spline
evaluations), each feeding 13 constraint rows, and `SUM[i]` alone is referenced
~30 times per sweep across those rows' summands:

```
flat per-summand tapes   3 606 095 ops per eval_g
shared-CSE prelude         893 790 ops per eval_g     (4.03x fewer)
```

`HybridTape` — per-summand local ops plus a prelude holding every CSE body
referenced by ≥2 summands — already existed in `nl_tape.rs`, fully written with
forward / gradient / Hessian entry points, but had **zero callers** outside its
own tests (flagged as dead code by the 2026-06 review, item L34). It is exactly
the fix, and it builds in 87 ms on this model.

### What was changed

- `eval_g` (and `eval_f`) evaluate through a problem-wide `HybridTape`: one
  `forward_prelude` for the whole constraint block, then one local sweep per
  summand. Gated on `hybrid_supported` — the hybrid builder *panics* on
  comparisons, AND/OR/NOT, if-then-else, min/max lists and external funcalls, so
  models using those keep the flat path — and on the prelude coming out
  non-empty, so models with nothing to share are untouched.
- `eval_g`/`eval_f` also stopped allocating: they called `Tape::eval`, which
  allocates a `Vec<f64>` per summand — ~148k allocations per call here, ~20 % of
  `eval_g`. New `Tape::eval_into` reuses the existing `vals_scratch` arena, the
  same fix `eval_jac_g`/`eval_grad_f` already had (M18).

Result on `robot_a`, 200 iterations, median of 3 interleaved runs:
**56.9 s → 44.2 s (−22 %)**, with a bit-identical iteration trajectory.
`eval_g` fell from 30 % of the run to 9.5 %.

### Why `eval_jac_g` / `eval_h` were left on the flat path

The prelude is only shared in the *forward value* sweep, where one pass serves
all 148037 summands. `gradient_summand` / `hessian_summand` each walk their own
summand's `prelude_reach` for their adjoint pass, so nothing amortizes across
summands — measured op counts come out roughly even with the flat tapes, for a
much larger change (`hessian_summand` seeds per variable and would displace the
Hessian-coloring machinery). Not worth it on this evidence.

## 4. The linear algebra is not the problem

Worth recording, because the issue suspected "a bad path in factorization or in
how we assemble/scale the augmented system" on the 52:1 shape. It is not:

- KKT is 105027 × 105027 with 356818 nonzeros; the factor has 565826. Max fill
  ratio 1.586 — the ordering is doing its job on this block structure.
  (Dimension is `n_x + n_s + n_c + n_d` = 1001 + 52013 + 0 + 52013: Ipopt's
  formulation carries one slack *and* one multiplier per inequality row. The
  nonzero count confirms the layout exactly — 3998 `W` + 52013 `(2,2)` +
  196781 `J_d` + 52013 `-I` + 52013 `(4,4)` = 356818.)
- 372 factorizations over 200 iterations (1.86/iteration, i.e. inertia
  correction is retrying about once per iteration).
- Inside FERAL the *refinement* outweighs the factorization (4820 vs 3231
  worker-pool samples). If the KKT share is attacked next, iterative refinement
  is where the time is, not the numeric factorization.

Inequality/bound handling is not the driver either: all 52013 rows are one-sided
inequalities, there are no equalities and no variable bounds, and POUNCE's steps
match Ipopt's exactly (§1). The control instance is consistent with the CSE
story rather than with an aspect-ratio story per se — `robot_1600` also has
defined variables (`I_the`, `I_phi`), but each is referenced by only 2
constraints, so its redundancy factor is ~2 rather than ~13.

## 4b. The barrier parameter barely moves — this is a *stall*, not slowness

Counting `lg(mu)` across the iteration logs reframes everything above:

| solver | iterations | distinct `mu` values reached |
| --- | --- | --- |
| POUNCE | 201 | `1e-1` only — **`mu` never decreased once** |
| IPOPT + MUMPS | 3000 | `1e-1` for 370 iterations, then `~2e-2` for 2631 |

An interior-point method has to walk `mu` down to ~`1e-9` to hit `tol = 1e-8`,
i.e. nine or ten barrier levels. IPOPT managed **one** in 3000 iterations. Both
solvers sit at tiny steps with 10–15 backtracks per iteration (`alpha_pr` of
1e-4 to 1e-5) for the entire run.

So `robot_a` is not a model where the filter line-search IPM is *slow*; it is
one where it **stalls on the barrier subproblem**, and the published times
(IPOPT 688 s, POUNCE 2140 s) are what it costs to grind through the stall —
order 5000 and 10000 iterations respectively at the per-iteration costs measured
in §2.

That resets the priority order. Per-iteration cost is worth maybe 1.5–1.7x more
(§3 already took 1.28x of it, condensing might take the rest). The stall is
worth two to three orders of magnitude, and it is shared with IPOPT — which
means it is a property of the algorithm on this model, and the place to look is
why the filter admits only 1e-4 steps here, not how fast each one is computed.
That deserves its own issue, and it is the honest answer to "what would close
the gap to KNITRO".

### The stall is the filter's acceptance test — and it costs ~80x

`accept_every_trial_step=yes` (accept the first trial point; no filter, no
Armijo test) turns the instance from "does not converge in 3000 iterations" into
a converged, independently verified solution:

| | iterations | wall clock | objective | status |
| --- | --- | --- | --- | --- |
| POUNCE default | 3000+ | 2140 s (Mittelmann) | — | grinds |
| **POUNCE `accept_every_trial_step=yes`** | **119** | **13.8 s** | 1.0431952 | Optimal |
| IPOPT default | 3000+ | 688 s (Mittelmann) | — | grinds |
| **IPOPT `accept_every_trial_step=yes`** | **116** | **8.7 s** | 1.0432009 | Optimal |

`pounce verify` on the POUNCE result, independently of the solver that produced
it: max constraint violation `0.000e0`, KKT stationarity residual `2.297e-11`,
complementarity `1.022e-9` — **VERIFIED**. And the objective is far *better*
than what either solver reaches by grinding (IPOPT sits at 8.17 with MA57 /
8.53 with MUMPS at its 3000-iteration cap).

Both solvers agree to 6 digits, so this is a property of the filter
line-search method, not a POUNCE defect — the same conclusion §1 reached from
the other direction. It also demystifies KNITRO: on this machine a solver that
simply does not fall into this stall finishes in ~10 s. No decomposition or
exotic algorithm is needed to explain 1 s on a faster box with MA57.

**This is a diagnostic, not a fix.** `accept_every_trial_step` discards the
global convergence safeguard and must not become a default. What it establishes
is *where* the 80x lives: the acceptance test rejects a path that is not merely
viable but better.

### None of the safe knobs reproduce it

All of Ipopt's anti-stall machinery is present (`mu_strategy`, `mu_oracle`,
`corrector_type`, `max_soc`, watchdog, restoration, `nlp_scaling_method`).
Tried, 200 iterations unless noted, `lg(mu)` reached being the progress metric —
the barrier has to walk from -1 to about -9:

| variant | `lg(mu)` reached | levels | note |
| --- | --- | --- | --- |
| default | **-1.0** | 1 | never moves |
| `nlp_scaling_method=none` | -1.7 | 2 | |
| `mu_strategy=adaptive` | -1.8 | 4 | |
| `+ corrector_type=affine` | -1.8 | 4 | identical to plain adaptive |
| `+ mu_oracle=probing` | -3.5 | 5 | best safe result; still short at 2950 iters |
| `max_soc=0` (3000 iters) | -8.6 | 5 | μ moves, but lands on obj 22.1 vs 1.04 |
| `accept_every_trial_step=yes` | **-9.0** | 6 | converges, 119 iterations |

So this is a research question, not an option-tuning exercise.

**Mechanism: characterized, not yet diagnosed.** An earlier draft of this note
proposed that with no equality constraints the filter's `theta` is structurally
~0, degenerating the filter into a pure Armijo test. **That is wrong** —
`POUNCE_DBG_LS=1 RUST_LOG=pounce::linesearch=debug` dumps the acceptor's own
`theta`/`phi` per accepted step, and `theta` is ~1e4, not 0:

```
iter=14 mu=1.0e-1 alpha=2.062e-5 mode=f theta=9.820150e3 phi=-1.447839e4 n_steps=6
iter=17 mu=1.0e-1 alpha=4.883e-4 mode=f theta=9.935307e3 phi=-1.464936e4 n_steps=11
iter=25 mu=1.0e-1 alpha=1.562e-2 mode=f theta=9.716279e3 phi=-1.574878e4 n_steps=6
```

What is actually established:

- `theta` climbs from 0 to ~9.9e3 in the first 14 iterations and then **pins
  there** — the iterates are not converging to feasibility at all.
- `phi` decreases steadily and monotonically the whole time.
- `alpha` is 1e-5 to 1e-2 with 5–11 backtracks per iteration.
- Every accepted step in POUNCE's first 200 iterations is **f-type** (`f`=155,
  `w`=36, `F`=9) — Armijo-governed, filter not consulted. POUNCE and IPOPT have
  *identical* accept-character sequences over those 200, another confirmation of
  §1.
- Over IPOPT's full 3000 the profile inverts: `h`=2014, `w`=679, `f`=294. So the
  governing test **changes character** across the run — Armijo early,
  filter-dominated late, with 2014 entries accumulating in the filter.

So the honest state is: we know the barrier subproblem at `mu = 1e-1` is never
solved, that `theta` stalls at ~1e4 while `phi` falls, and that bypassing the
acceptance test entirely reaches a verified solution. We do **not** yet know
which of the two regimes above is causal. Nobody should design a fix before
running the `POUNCE_DBG_LS` trace out to several hundred iterations and through
the f-type-to-h-type transition — that trace is cheap and is the obvious next
step.

**A caution on the "better objective".** `robot_a` is nonconvex (the constraint
bodies carry `S^3` and `S^5`), so `accept_every_trial_step` landing at 1.0432
against 8.17 may simply be a different basin rather than evidence the filter
steers wrong. The 80x — converges at all, versus does not — is solid. "Finds a
better optimum" is weaker evidence than it first looks.

**Sizing a safe fix.** Three tiers, in increasing risk:

1. *Detect and report* (low risk, useful regardless). `mu` failing to move for N
   iterations while steps are still being accepted is a crisp, cheap signal.
   POUNCE already has the machinery — `DiagnosticsState`, and a `find_stalls`
   tool in the studio MCP server. Turning a silent 2140 s grind into "barrier
   stalled at mu=1e-1 since iteration 14" is worth doing on its own.
2. *Escalate on detection* (medium). There is in-repo precedent: the
   `mu_strategy_fallback` option is a POUNCE addition with no Ipopt counterpart,
   so "strategy X stalled, switch to Y" is an established pattern here. Safety
   depends entirely on the escalation being bounded and revertible.
3. *Change the acceptance test* (high — a research project, not a patch). It
   touches every model, it forfeits the digit-for-digit Ipopt equivalence that
   made this whole investigation tractable (§1), and it needs its own global
   convergence argument. Gate on the full benchmark suite, not on `robot_a`.

## 4c. POUNCE's own active-set SQP is much worse here, not better

`algorithm=active-set-sqp` is the obvious thing to try given §5's hypothesis
that an active-set route is what makes this model cheap. It does not work on
this instance:

| `sqp_qp_max_iter` | wall clock | SQP major iterations completed |
| --- | --- | --- |
| 5 | 1.5 s | 0 |
| 20 | 4.8 s | 0 |
| 50 | 11.1 s | 0 |
| 200 (default) | 42.8 s | 0 |

Time is exactly linear in the QP pivot cap at ~0.22 s per pivot, and the **first
QP subproblem never completes** — the solver evaluates the model once and then
exhausts its pivot budget. A cold-start working set for a 1001-variable QP needs
on the order of 1001 pivots, i.e. ≳ 220 s for the first subproblem alone, before
any SQP major iteration happens.

0.22 s per pivot is itself anomalous for a 1001-variable QP — pricing 52013
rows is a 196781-nonzero matvec, microseconds. Unverified hypothesis: the QP
inner solver is refactorizing a KKT that still carries all 52013 constraint
rows rather than just the working set (`sqp_qp_max_schur_updates_before_refactor`
would set the cadence). Worth checking before anyone concludes "active-set does
not suit this model" — the current evidence only says *our* active-set path does
not, and it may be for an incidental reason.

## 5. What is left, and the KNITRO question

After this change the profile is KKT-dominated: search direction 43 %, exact
Hessian 22 %, line search 26 % (of which `eval_g` is 9.5 %), Jacobian 5 %.

The remaining structural idea is **condensing the augmented system**. With only
inequalities, blocks 2 and 4 of the KKT layout in `std_aug_system_solver.rs`
have diagonal (2,2) and (4,4) and a `-I` at (4,2), so `Δs` and `Δv_d` can be
eliminated analytically, leaving

```text
  (W + D_x + δ_x + J_dᵀ Σ J_d) Δx = rhs,   Σ = (D_s+δ_s)[I + (D_d+δ_d)(D_s+δ_s)]⁻¹
```

with `Σ` diagonal. On `robot_a` that replaces a **105027 × 105027** indefinite
factorization (565826 factor nonzeros, ~1.86 of them per iteration) with a
**1001 × 1001** one. And `J_dᵀ Σ J_d` costs nothing here: every row of `J_d`
touches 4 *consecutive* variables (the cubic-spline support), so the product is
banded with half-bandwidth 3 and 3998 lower-triangle nonzeros — bit-for-bit the
same sparsity as `W` itself. A banded Cholesky on that is microseconds.

Three things make it real work rather than a rewrite of `solve_once`:

- **Inertia.** The full-space `LDLᵀ` hands the algorithm its inertia, which is
  what drives `δ_x` regularization (the `lg(rg)` column, and the 1.86
  factorizations per iteration). Condensed, the eliminated block has known sign,
  so Sylvester's law fixes its inertia a priori and a *failed Cholesky* stands
  in for "wrong inertia" — the same argument `schur_aug_system_solver.rs`
  already makes for its partition.
- **Conditioning.** `Σ ~ z/s` spans many orders of magnitude near convergence,
  so `JᵀΣJ` squares an already-bad condition number. This is exactly why Ipopt
  defaults to the full space. Refinement has to run against the *unreduced*
  residual — which POUNCE already does, and where its linear-solver time
  already goes.
- **A gate.** `JᵀΣJ` is only sparse when `J`'s rows have small overlapping
  support; one dense-ish row makes it dense. Needs a symbolic `nnz(JᵀJ)`
  estimate up front and a permanent, transparent fallback — the discipline
  `SchurAugSystemSolver` already implements against `max_schur_frac`.

The seam exists: `AugSystemSolver` is a trait, and `SchurAugSystemSolver` is
already a second implementation that wraps `StdAugSystemSolver`, reuses its
assembly and RHS packing, routes the factorization elsewhere, and falls back
permanently on a cheap structural test. A condensed solver is a third
implementation of the same shape, not surgery on the algorithm.

**Expected value, and why it should be spiked before it is scheduled.** Against
the instrumented split in §2: the 13.0 s of factorization (31 %) is what
collapses — 105027 indefinite → 1001 banded. The 8.6 s of back-solve (20 %)
mostly does *not*. The triangular solves shrink with the factor, but the
expansion back to `Δs` / `Δv_d` is still O(m) per solve, and iterative
refinement has to run against the unreduced residual to be trustworthy, which
means a matvec with the full 356818-nonzero KKT on every refinement step. Best
case is therefore roughly 42 s → 25 s, ~1.7x, putting POUNCE at ~0.12 s per
iteration against IPOPT/MUMPS's 0.138 — ahead, for the first time on this shape.

Two things could eat that, and one of them is serious:

- **Conditioning (the real risk).** The reported pivot range on the full-space
  system is already `min_abs_pivot = 6.1e-08` against `max_abs_pivot = 1.6e+07`
  — fourteen orders of magnitude before condensing. `Σ ~ z/s` grows without
  bound as the barrier parameter falls, and `JᵀΣJ` *squares* the condition
  number. If that forces more refinement steps, the 20 % back-solve grows and
  can swallow the 31 % that was saved. This is not a hypothetical concern; it is
  the documented reason Ipopt keeps the full space by default.
- **Forming the product.** `JᵀΣJ` is rebuilt whenever `Σ` changes: 52013 rows ×
  a 4×4 outer product ≈ 0.8M FMAs plus scatter, per assembly, ~372 assemblies
  per 200 iterations. Small against 13 s, but not zero.

One thing gets *better*: at 372 factorizations per 200 iterations the algorithm
is retrying inertia correction ~1.86 times per iteration, and in condensed form
each retry is a failed Cholesky rather than a fresh indefinite factorization.

So the cheap experiment first: dump `J_d`, `Σ` and `W` from a *late* iterate
(small `mu`, where conditioning is worst), form the condensed matrix offline,
and check that Cholesky succeeds and the refined solution matches the
full-space one. That costs an afternoon and either de-risks the whole feature
or kills it before anyone writes a `CondensedAugSystemSolver`.

**And it is a per-iteration play only** — see §4b, which is the bigger term by
two or three orders of magnitude. Scope condensing as "beat IPOPT on m ≫ n",
not "match KNITRO".

On KNITRO, being careful about what is measured and what is inferred. Nothing
here was run against KNITRO — the 1 s is Mittelmann's published wall clock, and
wall clock to a verified solution is the only cross-solver quantity that is
apples-to-apples. *Iteration* counts are not: KNITRO is a different algorithm
family, and an SLQP major iteration (an LP subproblem plus its simplex minors)
is not the same unit of work as a filter-IPM iteration. So "KNITRO takes fewer
iterations" is not a claim this note can make.

What it can bound: one constraint sweep of this model costs 2.1 ms
(4.294 s / 2056 evaluations), which is a floor per iteration for anything doing
comparable AD over 52013 rows. Even with free linear algebra and no line search
that is at most ~500 sweeps inside 1 s, against the 3000+ iterations IPOPT
burns here *without converging*. So whatever KNITRO does, it is doing an order
of magnitude less total work — not the same work faster.

The plausible mechanism is the one the issue proposed. `robot_a` is a
discretized semi-infinite program: one continuous constraint family sampled at
4001 collocation points × 18 families. At a nondegenerate solution at most
n = 1001 of the 52013 rows can be active, so ~98 % are slack — and an
active-set method only ever works with the small active set, while an IPM
carries every row in the KKT system at every iteration and assigns all of them
a nonzero multiplier. Our own run shows exactly that: at iteration 200, 99.8 %
of the 52013 multipliers are above 1e-8 of the largest. That is a different
project from condensing, and the more likely route to KNITRO's number.

## 6. Loose end: `inf_pr` means something different from Ipopt's

Not a performance issue, but it surfaced while diffing the logs and is worth its
own look. On this pure-inequality model Ipopt reports `inf_pr = 0.00e+00` for
almost every iteration and a final "Constraint violation: 0.0"; POUNCE reports
1.25e+00 scaled / 2.79e+04 unscaled at the same iterate. The trajectories are
identical, so the filter is not being driven by different values — the two are
reporting different quantities: Ipopt's is the residual ‖c(x) − s‖ of its own
slack reformulation (structurally zero while the slacks track), POUNCE's is the
violation of the original row bounds. POUNCE's number is arguably the more
useful one, but for a solver that advertises drop-in Ipopt compatibility a
headline metric that reads "infeasible" where Ipopt reads "feasible" is a
compatibility wart.
