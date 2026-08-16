VERDICT: improved

Fixes #610

Baseline for every number below: `a44f4e8bcb67e577fb27c97a628efaff9822e494`
(`main` at the time of writing — "Merge pull request #628 from
jkitchin/fix/615-homotopy-bound-cycling").

## Problem

`race_starts` gave each of N starts the same truncated solve **from a cold
start**, ranked the field once on terminal `(violation, objective)`,
returned the best `top`, and reported nothing. Three consequences, all
measured on the parent commit:

1. **It kept most of multistart's cost.** The candidate that was hopeless
   after two iterations was still charged for ten.
2. **It threw away the solver state between rounds.** There were no rounds
   — but the truncated solve's multipliers and barrier parameter were
   discarded even for the winner, and the documented follow-up
   (`WarmStart.from_info(res.x, res.info)`) only recovered them for the
   one candidate that was returned.
3. **It could not report its own spend.** Every candidate outside `top`
   was discarded, results included, so the evaluation count the issue
   asks to reduce was not observable from the outside at all.

Cost of the pre-#610 policy on the benchmark suite (six multi-basin
models × three field sizes; every call counted through wrappers on the
user callables, so eliminated candidates are counted too):

| | user-callable evaluations | solver iterations | quality regressions |
|---|---|---|---|
| before (`a44f4e8b`) | **19611** | **3494** | — (reference) |
| after (`policy="halving"`, default) | **16109** (−17.9%) | **2276** (−34.9%) | **0 / 18** |
| after (`policy="fixed"`) | 19611 | 3494 | 0 — bit-identical to before |

Per configuration (`benchmarks/scripts/race_starts_bench.py`):

| model | starts | fixed evals | halving evals | Δ | fixed iters | halving iters |
|---|---|---|---|---|---|---|
| `double_well` | 8 | 234 | 242 | **−3.4%** | 66 | 48 |
| `himmelblau_disc` | 8 | 484 | 472 | +2.5% | 77 | 50 |
| `six_hump_camel` | 8 | 458 | 412 | +10.0% | 77 | 49 |
| `rastrigin_eq` | 8 | 436 | 386 | +11.5% | 75 | 48 |
| `hs71` | 8 | 648 | 512 | +21.0% | 80 | 48 |
| `deceptive_circle` | 8 | 444 | 384 | +13.5% | 77 | 50 |
| `double_well` | 16 | 518 | 525 | **−1.4%** | 160 | 138 |
| `himmelblau_disc` | 16 | 1024 | 1080 | **−5.5%** | 182 | 150 |
| `six_hump_camel` | 16 | 1046 | 1024 | +2.1% | 185 | 154 |
| `rastrigin_eq` | 16 | 866 | 822 | +5.1% | 156 | 130 |
| `hs71` | 16 | 1841 | 1391 | +24.4% | 259 | 173 |
| `deceptive_circle` | 16 | 920 | 934 | **−1.5%** | 168 | 153 |
| `double_well` | 27 | 863 | 758 | +12.2% | 272 | 166 |
| `himmelblau_disc` | 27 | 1690 | 1424 | +15.7% | 302 | 175 |
| `six_hump_camel` | 27 | 1794 | 1364 | +24.0% | 320 | 181 |
| `rastrigin_eq` | 27 | 1464 | 1236 | +15.6% | 269 | 163 |
| `hs71` | 27 | 3337 | 1877 | **+43.8%** | 481 | 213 |
| `deceptive_circle` | 27 | 1544 | 1266 | +18.0% | 288 | 187 |

Quality is identical everywhere, and checked against an oracle rather
than against each other: solving *every* Sobol start to convergence and
taking the best feasible objective gives the same answer as both policies
in all eighteen configurations. **Solver iterations fall in all eighteen.**
Evaluations rise in four — the bolded negatives — and that is not noise;
see "Where it does not pay" below.

A second, harder measurement, run from a scratch harness rather than the
committed one (so the numbers below are stated, not reproducible from
this branch): with a *hand-picked* six-to-eight-start field at
`iters=10` over the same models, fixed spends 2218 evaluations and
halving 2246 — **1.3% worse**, quality identical, both at the oracle.
Small fields with a small per-candidate budget leave no room for a
ladder. The committed harness covers the same shape at `8:10`, where
three of six models still come out ahead.

## Cause

Two independent things, and the second is why the first was never fixed:

* **The policy is non-adaptive by construction.** `max_iter=iters` for
  everyone, one ranking at the end. There is no point at which a
  candidate can be stopped early, because there is no point at which it
  is looked at other than the end.
* **The only resource it can spend is iterations, and it cannot even
  observe evaluations.** The Python `info` dict exposed no evaluation
  counters at all (`SolveStatistics` has carried `num_obj_evals` and
  friends for a long time; the only way to read them was to write a solve
  report to a file), and no restoration counters either. A policy that
  wants to charge an expensive candidate more than a cheap one had
  nothing to charge against.

## Fix

Five commits.

**1. `info` gains the solver's own counters.** `n_obj_evals`,
`n_grad_evals`, `n_constr_evals`, `n_jac_evals`, `n_hess_evals`, and the
gh#12 restoration audit counters `restoration_calls`,
`restoration_outer_iters`, `restoration_inner_iters`. `build_info_dict`
took twelve scalars read off the same `stats` at all three call sites; it
now takes `&SolveStatistics`, which is how the new fields arrive without
a fourteen-argument signature. Reporting only.

**2. `_prepare_nlp` / `_result_from_info` extracted from `minimize`.**
Pure extraction. The racer drives the solver itself and must build the
same problem and assemble the same `OptimizeResult` — including the
gh#119/#123 `success` upgrade — rather than a lookalike that drifts.

**3. The ladder.** Every candidate runs for a small budget; the field is
ranked; the weakest fraction is eliminated; survivors are resumed with a
budget `eta`× larger. `iters` keeps its meaning — the budget the
*winner* ends up with — so the two policies are comparable on quality and
differ only in what the losers cost.

**Ranking** is a weighted sum of five signals, each rank-normalized
within the cohort so a violation in mol/s and a dimensionless KKT
residual combine without an invented scale factor: `violation` (3.0),
`feasibility_progress` (1.0), `kkt` in log units (1.5),
`objective_progress` per evaluation and damped while infeasible (1.0),
and `health` from the restoration counters plus non-finite/failed exits
(1.0). Feasibility dominates because an infeasible candidate's objective
is not a number about the problem being solved.

**Diversity**: survivors within `cluster_tol` in scaled units are
collapsed to the best of the group, and `explore` candidates from outside
the cut are retained anyway, chosen farthest-first from those already
kept (deliberately *not* next-best-scoring — the next-best is usually in
a basin already represented).

**Evaluations are the resource.** Rung 0 carries no evaluation budget —
it *is* the calibration of what a solve of this model costs — and every
later rung is a multiple of the cohort's rung-0 spend. Each candidate
converts its remaining budget into an iteration cap through *its own*
measured evaluations-per-iteration, so a candidate burning a dozen
line-search trials per iteration gets fewer iterations for the same
resource. A cumulative iteration ceiling rising to `iters` bounds the
other side.

**4/5.** Tests, `benchmarks/scripts/race_starts_bench.py`, CHANGELOG, and
a new section in `docs/src/initialization.md`.

### Tried and rejected

* **Ranking on `(violation, objective)` at the end, as before.** It is
  right for a *converged* pair and wrong for a truncated one, and wrong in
  a way that costs the race its answer. On HS71 from a six-point
  hand-picked field it returned a candidate that had driven its violation
  to 2e-4 in the 27.146 basin over one at 1.7e-2 in the 17.014 basin —
  purely on the first key. Replaced by `_final_key`: a survivor that has
  removed all but `feas_band` (default 1%) of its *initial* violation is
  on track to feasible, and among those the objective decides. The fixed
  policy keeps the old sort verbatim.
* **`obj0`/`violation0` captured at the first pause.** They are then
  identically zero for everyone at rung 0 — so `feasibility_progress` and
  `objective_progress`, two of the five signals, contribute *nothing* on
  the rung that eliminates the most candidates. Now measured at `x0`
  before racing, at a cost of one objective (and one constraint)
  evaluation per candidate. This is why `double_well` at 8 starts comes
  out 8 evaluations up.
* **A rung-0 budget derived from a guessed evaluations-per-iteration.**
  The first implementation set rung 0 to `iters/eta^(rungs-1)` iterations
  × a hard-coded 4 evaluations/iteration. On the benchmark models a
  single solve's entry cost alone exceeded that, so rung 0 became 1
  iteration regardless, and ranking Rastrigin on the solver's first step
  eliminated the eventual winner: `rastrigin_eq` returned 4.977 where the
  fixed policy returned 0.997. Fixed two ways — rung 0 now calibrates
  rather than guesses, and `min_rung_iters` (default 3) shortens the
  ladder rather than let the bottom rung fall below a rankable length.
* **`min_rung_iters=2`.** Measured 30.6% evaluation savings over 36
  configurations with zero quality regressions — better than the 19.6%
  the default gives on that same grid. Kept at **3** anyway: the failure
  mode of a too-short bottom rung is *silent* (a worse answer, no
  signal), while the cost of the extra iteration is visible in the
  report. The knob is exposed and documented.

### Where it does not pay

A rung boundary costs a fresh solver application and a re-evaluation at
the seed. On a model where that fixed cost is a large fraction of the
whole solve — one variable, no constraints, a handful of evaluations per
iteration — the ladder cuts iterations but comes out level or slightly up
on evaluations. That is four of the eighteen configurations above, worst
case −5.5%. It is named in the docs, named in the test docstring, and
`policy="fixed"` remains available. The iteration count falls even there.

## Acceptance criterion 1: pause/resume, measured

**What POUNCE can and cannot do.** POUNCE has **no API for suspending an
IPM mid-iteration and re-entering the same algorithm object.** Every
`Solver.solve` rebuilds its `IpoptApplication` from the problem's current
options — `crates/pounce-py/src/solver.rs` says so in its module header
and names the missing piece: *"future Phase 3b work will add a
`resolve()` that reuses the cached symbolic factor across solves."* The
`Solver` session's held KKT factor exists only after a *converged* solve
and is not consumed by the next one, so holding the session across rungs
does not by itself accelerate anything.

**The API that would be needed**, stated plainly as the issue asks: a
`Solver.resolve(max_iter=...)` that re-enters `IpoptAlgorithm` with its
filter, line-search state and inertia-correction history intact, rather
than constructing a fresh application from the last iterate. That does
not exist and this PR does not add it.

**What a pause does carry, and it is not nothing:** the whole
interior-point iterate — primal point, constraint multipliers, both
bound-multiplier blocks, and the barrier parameter μ — replayed through
#607's warm-start path under a scoped option overlay, so #606's
recentering measures the point it is handed. The measurement that this is
materially not a cold restart, on `rastrigin_eq`, 8 starts, both arms
starting from **the identical iterate**:

| paused at | resume (state + point) | restart (point only) |
|---|---|---|
| 3 iterations | **32 iters** / 330 evals | 43 iters / 368 evals |
| 5 iterations | **17 iters** / 250 evals | 43 iters / 376 evals |
| 8 iterations | **0 iters** / 80 evals | 43 iters / 372 evals |

Every arm reaches the identical objective, start for start. The 8-iteration
row is the clearest: all eight candidates have converged by then, the
resumed solve recognises it in **zero** iterations because the carried
duals and μ satisfy the convergence check on entry, and the restarted
solve — handed the same point and nothing else — needs 5 to 8 iterations
each to re-derive the same certificate.

**The honest qualifier.** This effect is model-dependent. On HS71 with 8
Sobol starts the same comparison is a wash (resume 98/92/77 iterations at
pause points 3/5/8 against restart 102/87/79) — at pause 5 the resume is
*worse*. #608's caution applies: a warm-started IPM often converges in
one iteration per step, and where it does there is nothing left for a
resume to remove. `rastrigin_eq` was chosen for the test precisely
because it has the headroom to tell the two apart; the test asserts a
75% margin so it pins the effect rather than the exact numbers.

## Blast radius

**Fixture sweep: byte-identical.** `scripts/sweep-fixtures.sh` over all
**57** CLI fixtures, new binary against one built from `a44f4e8b` in a
`git worktree` — `diff` is empty (status, objective **and iteration
count** unchanged on every fixture). Expected: the only Rust change adds
keys to a Python dict, and `race_starts` has no CLI surface (it is
referenced only from `python/pounce/`, `python/tests/` and
`docs/src/initialization.md`).

That sweep therefore proves the info-dict refactor is trajectory-neutral
and **nothing else** — it never constructs the racer. The
racing-on regime is covered by `benchmarks/scripts/race_starts_bench.py`
(36 races, 18 configurations, both policies) and by the 28 tests in
`python/tests/test_starts_racing.py`.

**Behaviour change for existing callers:** `race_starts`'s default policy
is now `"halving"`, so a caller who does not pass `policy=` may get a
different winner and therefore a different downstream trajectory. This is
the "replace" the issue asks for; `policy="fixed"` restores the old
behaviour exactly and is pinned by test. The only pre-existing in-repo
caller is `python/tests/test_starts.py`, plus one docs example; both pass
unchanged under the new default.

**`policy="halving"` is NLP-path only** — it holds a `pounce.Solver`
session per candidate — and raises on a non-`"nlp"` `solver_selection`
rather than silently routing to the convex solver and losing the session
it needs. `policy="fixed"` still routes as before.

## Tests

`python/tests/test_starts_racing.py`, 29 tests. **27 of 29 fail on the
parent commit** (parent Python *and* parent extension; verified by
copying the parent's `_starts.py` / `_minimize.py` / `__init__.py` in
from a `git worktree`). Being precise about how they bite, since most
need the new API:

* **Behavioural, no new API:**
  `test_the_default_policy_beats_the_pre_610_cost` is written entirely in
  the pre-#610 call signature and runs unmodified on `a44f4e8b`, where it
  fails on the assertion that matters — `AssertionError: the default
  racing policy spent 3077 evaluations; the pre-#610 fixed budget spent
  3077 for the same three answers` — after the three quality assertions
  above it have passed. That is the criterion-4 claim, biting.
* **API-shaped (stated as such, not counted as behavioural):** the other
  26 failures are `TypeError: race_starts() got an unexpected keyword
  argument 'policy'` / `'return_report'` / `'weights'` / `'explore'`, or
  an assertion on the `info` counters that do not exist there. They
  cannot bite behaviourally because the behaviour they test did not
  exist. One of them,
  `test_restoration_and_health_reach_the_ranking`, fails on the parent
  *extension* only — with the parent's Python and this branch's
  extension it passes, which is the honest split between the two halves
  of the change.
* **Passes on the parent, deliberately — two of them:**
  `test_resuming_a_paused_candidate_beats_restarting_it` measures a
  property of POUNCE's *existing* warm-start machinery; it is the
  evidence for the design, not a test of new code, and presenting it as
  a regression test would be dishonest.
  `test_dedup_never_returns_fewer_results_than_asked_for` pins a contract
  the parent already honoured — it guards the *new* dedup against
  breaking it, which it did in an intermediate version of this branch
  (four starts in one basin, `top=3`, one result returned).

Criterion by criterion:

* **Pause/resume** — `test_resuming_a_paused_candidate_beats_restarting_it`
  (the table above), `test_survivors_are_resumed_rather_than_restarted`
  (asserts rung 0 starts everything and every later rung resumes and
  cold-starts nothing).
* **Determinism** — `test_the_whole_race_is_reproducible_round_for_round`
  compares the **entire** record across two runs: per-rung budgets,
  entrants, survivors, elimination reasons, composite scores, and
  per-candidate evaluations/iterations/resumes/restoration counts/final
  x — not the winner.
  `test_determinism_survives_a_different_call_order` interleaves an
  unrelated race to prove the scoped option overlay does not leak
  `mu_init` between races. Starts come from `generate_starts(seed=...)`;
  the ladder itself draws no random numbers.
* **Per-round reporting** —
  `test_every_candidate_has_a_reason_and_every_rung_has_a_cost` asserts
  every candidate carries a reason, every rung a budget and a spend, and
  that eliminated ∪ standing partitions the field exactly.
  `test_the_fixed_policy_also_reports_what_it_spent` covers the baseline.
* **Cost at equal quality** — quality asserted **per problem**
  (`test_halving_matches_the_fixed_answer_on_every_problem`), cost
  asserted **across the suite**
  (`test_halving_costs_materially_less_across_the_suite`, ≥10% solver
  evaluations / ≥15% iterations / ≥5% user evaluations). Suite-level is
  how the issue words it and how it has to be — the `double_well` case is
  in the suite and its exception is named in the docstring.
* **Adversarial** — `_deceptive_circle`: `f = x³ − 3xy² + 0.3y` on
  `x² + y² = 4` has three feasible basins (−7.481, −8.003, −8.520); off
  the circle `f` is unbounded below, so the three starts with the best
  early objective (−24.4, −16.2, −16.4 against the winner's −8.39) are
  exactly the ones headed for the wrong basin.
  `test_early_objective_progress_misleads_but_the_race_still_wins`: the
  default ranking keeps candidate #5 and reaches **−8.520236**.
  `test_ranking_on_objective_alone_eliminates_the_eventual_winner`: with
  the feasibility, KKT and health weights zeroed, #5 is cut at rung 0
  ("below halving cut (rank 6 of 9, keep 3)") and the race lands in the
  **−8.002500** basin. The control arm is what makes this a statement
  about the ranking rather than about the fixture being easy.

  **Worth recording:** the *pre-#610* policy also finds −8.520236 on this
  fixture, at 5, 9 and 15 iterations. The adversarial case is not a
  regression the old policy suffered; it is the failure mode that
  eliminating on partial information introduces, and which the composite
  ranking exists to prevent.
* **Frozen baseline** — `test_fixed_policy_reproduces_the_pre_610_baseline`
  transcribes the 0.10.0 function body into the test (an independent
  witness, not a call into shipped code) and compares `x`, `fun`, `nit`,
  `nfev`/`njev`/`nhev`, `status` and `final_constr_viol` across three
  models × two budgets. `test_fixed_policy_refuses_the_post_610_arguments`
  makes the frozen policy reject `hess=`/`args=` rather than accept and
  ignore them.

### Suite status

| check | result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo clippy` (workspace, `--exclude pounce-hsl`, `--all-targets`) | no errors; no new warnings in the two files touched |
| `cargo test --workspace --exclude pounce-hsl --release` | **clean** — exit 0, 227 suites, 2953 tests, 0 failures |
| `python/tests` | **1082 passed, 21 skipped, 0 failed** — see note |
| `pyomo-pounce/tests` | 352 passed |
| `scripts/check-docs-consistency.sh` | OK — 45 pages, all reachable |
| `scripts/check-release-consistency.sh` | OK |
| `scripts/sweep-fixtures.sh` (57 fixtures) | diff empty vs `a44f4e8b` |

**`python/tests` note.**
`test_jax.py::test_jax_problem_no_rebuild_on_repeat_solve_pounce_75`
failed on an *earlier* full-suite run and is a **wall-clock timing
assertion** ("the second solve is dramatically faster than the first").
That run overlapped a `cargo test --release` build saturating all four
cores. Re-run on an idle machine the whole suite is green — the 1082/0
above — so it was CPU contention, not this change, which touches nothing
under `pounce.jax`. Recorded here rather than quietly dropped, because
"it passed the second time" is only an answer once you can say why.

**Environment notes** for anyone reproducing: `pyomo-pounce` needs
`pyomo>=6.10.1`, `networkx`, `scipy`, `pandas`, and the CLI staged into
`python/pounce/bin/` — `maturin develop` does not stage it, and without
it 15 `pyomo-pounce` tests fail with `ApplicationError: No executable
found for solver 'pounce'`. All 352 pass once staged.

## Conflict note

Sibling branch `claude/pounce-issue-609-loop` (PR #629) touches
`pyomo-pounce/**` and `docs/src/pyomo.md`. **One conflicting file, one
hunk:**

* **`CHANGELOG.md`** — both branches insert a new bullet immediately
  after `## [Unreleased]`. `git merge-tree` reports the conflict at that
  single insertion point. Resolution is to keep both bullets; ordering is
  a preference, nothing is lost either way.

No source overlap: #609 touches no file under `python/pounce/`,
`crates/`, `benchmarks/` or `docs/src/initialization.md`, and this branch
touches nothing under `pyomo-pounce/`.

## Acceptance criteria — met / not met

| # | criterion | verdict |
|---|---|---|
| 1 | API can pause/resume candidates without cold-restarting each round | **Met, with a stated limit.** A pause carries the primal point, all three multiplier blocks and μ, and resuming is measurably not a restart (0 vs 43 iterations on `rastrigin_eq` at pause 8; 17 vs 43 at pause 5). It does **not** carry the filter/line-search state, because POUNCE has no `Solver.resolve()`; that API is named above and is not added here. |
| 2 | Deterministic under a fixed seed and deterministic backend | **Met.** The whole per-round record — survivors, eliminations, scores, resource spend — is asserted equal across two runs, and across an interleaved unrelated race. The ladder draws no random numbers at all. |
| 3 | Reports per-round resource use and elimination reason | **Met.** `RaceReport` / `RaceRound` / `RaceCandidate`, `return_report=True`, `report()`. Every candidate carries a reason; eliminated ∪ standing partitions the field. |
| 4 | Matches or improves the best fixed-budget result at materially lower total evaluations | **Met at suite level.** 17.9% fewer evaluations and 34.9% fewer iterations over 18 configurations, zero quality regressions. **Not met on 4 of 18 configurations** taken individually (worst −5.5%), all small/cheap models; iterations fall in all 18. Named in the docs and in the test. |
| 5 | Adversarial tests where early objective progress misleads but feasibility/KKT identifies the winner | **Met.** `_deceptive_circle` plus a control arm that reproduces the failure when the feasibility/KKT/health weights are removed. |

| # | scope item | verdict |
|---|---|---|
| 1 | Run all candidates for a small iteration/evaluation budget | **Met.** Rung 0, sized by `min_rung_iters` and the ladder geometry. |
| 2 | Rank on feasibility reduction, scaled KKT, objective progress, step-acceptance/restoration, numerical health | **Met.** Five rank-normalized signals, weights exposed via `weights=`. Restoration reaches the ranking through counters this PR adds to `info`. **Partial on one sub-signal:** "step acceptance" is read indirectly — from the exit status (a solve that stopped short of its cap without converging), from restoration share, and from evaluations-per-iteration, which rises when the line search backtracks. The per-iteration `ls_trials` / `alpha_primal` record exists in `SolveStatistics.iterations` but is only populated when a full solve report is requested, so wiring it in would mean writing a report file per rung. |
| 3 | Eliminate a configurable fraction (successive halving / Hyperband) | **Met.** `eta`, `rungs`, `min_survivors`. |
| 4 | Resume survivors from their full solver state with a larger budget | **Met**, to the limit stated in criterion 1. |
| 5 | Preserve diversity: cluster near-identical survivors, retain an exploration quota | **Met.** `cluster_tol` collapsing, `explore` farthest-first quota. |
| 6 | Make evaluation budget — not just iteration count — the comparable resource | **Met.** Rungs are denominated in evaluations, calibrated at rung 0; each candidate converts through its own measured evaluations-per-iteration, with an iteration ceiling as the secondary bound. Pinned by `test_the_evaluation_budget_can_bind_before_the_iteration_ceiling`. |

Also kept, as the scope requires: **the fixed-budget policy remains
selectable and reproduces its old results exactly**, pinned against a
transcription of the 0.10.0 body.

No follow-up issues filed.

---

- [x] Tests fail on the parent commit for the stated reason (27/29; one is
      behavioural in the pre-#610 signature, the rest are API-shaped and
      said to be, and two deliberately pass)
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [x] Book page under `docs/src/` updated (`initialization.md`, already
      linked from `SUMMARY.md`; no new page)
- [x] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
- [x] Every claim in this body is true of the code as it stands now

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Uo6guKKqeUrp9MRH7PCzUB
