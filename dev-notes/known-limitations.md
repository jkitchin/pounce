# Known limitations

Measured defects and accepted regressions that nobody is scheduled to work on.
Each one used to be an open issue; it lives here instead so the issue tracker
holds only planned work, while the measurements stay where the next person
can find them.

An entry records what is wrong, how big it is, what has already been tried
and failed, and how to tell whether a fix works. If you pick one up, open a
fresh issue that links back to the entry, and delete the entry when the fix
merges.

---

## NLP: L-BFGS fixtures slower after restoration matched upstream (gh#1146)

*Measured on `claude/charming-carson-cz5psg` against `main` at `358ecbb`,
`scripts/sweep-fixtures.sh`, L-BFGS leg, on **two architectures**: x86_64
(Linux) by the PR, arm64 (Apple Silicon, macOS) in the PR #1148 review.
Recorded 2026-11-07.*

**The change.** Under `hessian_approximation=limited-memory`, restoration's
quasi-Newton update ran over the whole restoration primal
`[x | n_c | p_c | n_d | p_d]`. Upstream (`update_for_resto_`) confines it to
the `x` block and leaves the linear slack blocks at zero curvature. Pounce
now does the same (`LimMemQuasiNewtonUpdater::leading_block`). On the gh#1146
model under `nlp_scaling_method=none` that turns `Restoration_Failed` at
`x = 0.0429` into Ipopt's own trajectory and answer (`x = 0.44927`,
14 iterations, iterate for iterate).

### What is worse

No status moved on either architecture, and the exact leg is untouched on
both. L-BFGS lines, `main` → branch:

| fixture | x86_64 | arm64 |
|---|---|---|
| `square_flowsheet_resto` | 2834 → **314** | total **3402 → 6210**: first attempt now at the cap, rescued later by `square_system_lm` |
| `dallass` | 301 → 272 | **218 → 279** |
| `circle_parabola_t1p5` | 66 → 154 (206 → 553 with ladder) | same |
| `eigena2` | 239 → 263 | same |
| `infeasible_square_scaled_1em4` | 24 → 26 (90 → 115) | same |
| `cc:design_cent_3` (lowered-solve total) | 4724 → 10967 | 4724 → 8210 |
| `cc:infeasible_pair` | 36 → 29 it | 36 → 29 it, driver total 3233 → 3672 |

Better on both: `cresc4` solves on the first attempt (was the ladder,
3200 total), `circle_parabola_t0p5` and `discs` no longer need the ladder,
`eigmaxa` 19 → 16, `issue981_water_main` 25 → 21 (100 → 67),
`issue_508_infeasible_gap_1em4` 79 → 73 (550 → 369),
`degenerate_start_infeasible` total 69 → 43.

### What is understood, and what is not

**`circle_parabola_t1p5` and `eigena2` (both architectures).** Both had
already left Ipopt's path in the **main loop, before any restoration call**,
on `main` and on the branch alike. Measured against Ipopt 3.14.20
(`amplpy` `ampl_module_ipopt`), same options, x86_64:

- `circle_parabola_t1p5`: Ipopt takes 31 iterations. Pounce leaves its
  path at main-loop iteration 2 (`‖d‖ = 7.66` against Ipopt's `25.7`).
- `eigena2`: Ipopt takes 167 iterations and first enters restoration at
  iteration 132. Pounce leaves its path at main-loop iteration 6.

So restoration is entered from points Ipopt never visits, and its
iteration count there measures that earlier gap, not this change.

**`square_flowsheet_resto` and `dallass` (opposite directions on the two
architectures): rounding-chaotic, so a single run is not a measurement.**

- *It is not FERAL's NEON kernels.* Cross-compiled for
  `aarch64-unknown-linux-gnu` and run under `qemu-aarch64`, both binaries
  reproduce the **x86_64** numbers exactly: on `square_flowsheet_resto`,
  `main` 2834 and the branch 314 and 197. Thread count (`RAYON_NUM_THREADS`
  1–16) changes nothing either. What differs on the reviewer's machine is
  therefore macOS itself, most likely libm's `exp`/`log` in the last ulp.
- *`square_flowsheet_resto` has a singular start.* `cond(J(x0)) = 1.5e19`,
  measured by central differences on the residual in
  `crates/pounce-cli/tests/square_system_recovery.rs`. Every solver's first
  step is a regularized guess. Ipopt 3.14.20 takes `‖dx‖ = 7.6e7` and pounce
  `1.1e7`, where the exact `J⁻¹c` is `5.6e16`; neither satisfies `J·dx = −c`
  to better than 40·‖c‖. Pounce leaves Ipopt's path at iteration 1, on `main`
  and on the branch alike, before any restoration.
- *Measured as distributions.* Perturbing `bound_push` to
  `1e-2·(1 + k·1e-12)`, k = 0…23 (x86_64, L-BFGS):

  | fixture, options | `main` | branch |
  |---|---|---|
  | `square_flowsheet_resto`, default ladder, `square_system_recovery=no`, `feral_increase_quality_retry=no` | 10 succeed, 5 false verdicts (`Infeasible`/`Restoration_Failed`), 9 at the cap | 6 succeed, 1 false verdict, 17 at the cap |
  | `dallass`, defaults | 10 succeed (7 Optimal, 3 Acceptable), 14 `ErrorInStepComputation`; median 239.5 it | 8 succeed (7 + 1), 16 `ErrorInStepComputation`; median 243.5 it |

  On `dallass` the two distributions are the same within sampling noise.
  On `square_flowsheet_resto` the branch trades false infeasibility verdicts
  for budget exits. That is the right direction for the verdict, but it also
  succeeds less often over these 24 draws. Ipopt's own L-BFGS is *not*
  chaotic on this model: all 12 perturbations end "Converged to a point of
  local infeasibility" at 42 iterations, a false verdict, since the model is
  feasible by construction.
- *Consequence for tests.* No test pins this fixture's L-BFGS outcome.
  `issue850_increase_quality_regression.rs` reports both runs.
  `square_system_recovery.rs` and `issue857_escalation_gated_quality_rung.rs`
  cap `max_iter` at 150: across the same 24 perturbations, with their
  options, no branch base solve succeeds before 197 iterations (277 with
  `square_system_recovery.rs`'s options), so the budget exit those tests
  need is stable.

**The same trade beyond the fixture corpus: Mittelmann's `qcqp…nc`.**
Measured in the PR #1148 review on arm64 (macOS), `hessian_approximation=limited-memory`,
120 s per problem, both binaries run concurrently:

| problem | `main` | branch |
|---|---|---|
| `qcqp1000-1nc` | `Restoration_Failed`, 611 it | `MaxIter`, 3000 |
| `qcqp500-3nc` | `Restoration_Failed`, 428 | `MaxIter`, 3000 |
| `qcqp750-2nc` | `Restoration_Failed`, 1026 | `MaxIter`, 3000 |
| `qcqp750-2c` | `Restoration_Failed`, 290 | **`Solve_Succeeded`**, 2987 |

- **Same shape as `square_flowsheet_resto`:** restoration no longer gives
  up early, and the outer loop then runs to the iteration cap. Each `nc` row
  is a failure before and after, but the new failure spends the whole budget.
- **Objective:** in all three, the objective at the cap is lower than at
  the old restoration exit.
- **Suite totals:** 31 solved before and after over the 47 problems. On the
  30 both solve, 8,996 → 9,020 iterations.
- **Not traced:** these models were not reproduced for this PR. The `.mod`
  sources need AMPL to translate, and Mittelmann's server could not be
  reached from the PR's environment. Whether the extra iterations buy a
  better point or only spend the budget is open.
- **How to tell:** if a later change makes these exit early again, check
  that it does so without losing `qcqp750-2c`.

**`cc:` totals (both architectures).** `cc:design_cent_3` and
`cc:infeasible_pair` go through the complementarity driver's lowered solves
and have no direct Ipopt comparison. Not traced.

### How to tell whether a fix works

- `circle_parabola_t1p5` and `eigena2` under
  `hessian_approximation=limited-memory` should match Ipopt's iteration log
  through their first restoration call.
- For `square_flowsheet_resto` and `dallass`, compare the perturbation
  distributions above, not single runs. A fix for `square_flowsheet_resto`
  belongs at its singular first step (pounce leaves Ipopt at iteration 1),
  not in restoration.
- Do not restore the whole-compound update to win any of these back. It is
  what made the gh#1146 model fail, and
  `crates/pounce-cinterface/tests/issue_1146_lbfgs_feasible_not_infeasible.rs`
  (`unscaled_lbfgs_restoration_recovers_as_upstream_does`) pins it.

---

## MPCC: four MacMPEC answers stay complementary only to ~7e-5

*Measured on `mpcc/cost-sweep-report` (after `df3fbd12`), MacMPEC through
the CLI. Recorded 2026-11-04.*

**Scope.** Answers whose branch re-solve (the "branch polish", see
`docs/src/complementarity.md`, "Complementary to `tol`, not to `sqrt(tol)`")
is refused, so the reported point keeps a nearly biactive pair at
`sqrt(τ)`. The report says so: `complementarity.natural_residual` and
`complementarity.polish.reason`, and `pounce-studio diagnose` warns
(`mpcc_loose_pairs`).

### What is wrong

| model | `natural_residual` | class | why the re-solve was refused |
|---|---|---|---|
| `pack-comp1-8` | 6.9e-5 | S | the branch is `Infeasible_Problem_Detected` |
| `pack-comp1p-8` | 6.9e-5 | S | the same |
| `pack-comp1p-32` | 7.0e-5 | W | the same |
| `incid-set2-32` | 6.4e-7 | S | the re-solved point certifies only W and a branch descends from it |

The objectives are at the best known to about 1e-8 relative
(`pack-comp1-8`: 0.6000000082 against 0.6), so the cost is the residual, not
the value.

### Already tried, did not work

On `pack-comp1-8`, 33 of its 49 pairs have `min(|G|, |H|)` above `10·tol`.
About 30 of them are strictly complementary, with sides of about 1e-6 and
1e-3 to 1e-2, so their product is just `τ`. Four are nearly biactive, with
sides within a factor of 30: pairs 9, 13, 17 and 46. Holding each pair's
other side, retried once from the same warm start when the first branch
fails:

- **Every loose pair.** This is wrong by construction: it pins the 1e-3
  sides of strictly complementary pairs. It is infeasible on all three
  `pack-comp1` models. On `opf_vv_lower_knee_held` it flips only that
  model's one loose pair, and the other branch is infeasible there too:
  solved directly, `s[1] = 0` with `r[1] >= 5e-5` reports local
  infeasibility.
- **Only the pairs whose sides are within 100× of each other.** Infeasible
  on all three.
- **Each of the four nearly biactive pairs alone.** Infeasible every time.

So the orientation of the nearly biactive pairs is not what makes the branch
infeasible. The retry was removed: it cost a solve on every refused polish
and gained nothing that was measured. Holding both sides of a pair is no
alternative, because that set lies inside both one-sided branches.

Untested: whether the branch NLP is truly infeasible or only locally so from
the warm start. A cold solve of the same branch, or a feasibility problem
with slacks on the held sides, would tell.

### How to tell whether a fix works

`make -C benchmarks macmpec-run` with `grep -l "keeping it"` over the logs.
The four models above should move to "re-solved on its branch", and nothing
else should move from at to worse. The `cc:` lines of
`scripts/sweep-fixtures.sh` must stay as they are except where the fix
intends them to change. `opf_vv_lower_knee_held.nl` (one branch infeasible)
and a both-branches-infeasible variant are the fixtures to test it against.
In the variant, `s[1] >= 5e-5` is added as well; the generator is in the
data root, `mpcc_opf.model.build` plus one `Constraint`.

---

## MPCC: what the coupled continuation costs where it falls back

*Measured on `mpcc/fast-continuation` (`5180695f` against `ecc2e4ff`, which
`complementarity_continuation=schedule` reproduces exactly). Recorded
2026-11-06. Tracked with the rest of pounce's MPCC cost in #1136.*

**Scope.** The default continuation (`complementarity_continuation =
coupled`, `docs/src/complementarity.md`) solves `τ = 1`, then one solve with
`τ` tied to `μ`, then the exact finish from three starts. Where its answer
does not pass the schedule's test, or the coupled solve reaches its budget,
the schedule resumes at `τ = 0.1` from the `τ = 1` solve, and what the coupled
attempt spent is the cost. Every case below falls back or loses an answer
the schedule found; none changes a status.

### What is worse

| case | schedule | coupled | why |
|---|---|---|---|
| fixture `ralph1`, lbfgs leg | 321 iterations | 584 | the coupled solve reaches the floor; all three finish attempts (100 each) miss; the schedule runs |
| fixture `opf_vv_lower_knee`, lbfgs leg | 554 | 804 | the same |
| OPF transmission (59), `mpcc-compare` | 240 s | 270 s (+12%), answers identical | on the IEEE 300 cases the coupled solve crawls toward the floor (about 3100 iterations unbudgeted, against about 900 for the whole schedule); its budget, `max(5 × first solve, 200)`, hands over |
| CALIPSO bunny-hop seed 1 | 462.605 | 463.195 (+0.13%) | both cheap routes reach the same exact answer, so the cold exact product, which finds 462.605 from this start, is skipped (`skips_cold_solve`); under the schedule the continuation's finish fails and the cold solve runs |

Elsewhere it is ahead: MacMPEC 158 at / 0 unsolved against 157 / 1 and
1149 s → 1000 s, `relu_nn` −23% wall, OPF and capacity −27% iterations,
CALIPSO ball-in-cup seed 1 better.

### Already tried, did not work

- **Each rung to `tol = κ·τ`** instead of coupling: −3% on `relu_nn`, and on
  MacMPEC 35 outcomes changed and `tap-15` turned `Infeasible_Problem_Detected`.
- **Coupling from `τ = 1`**: MacMPEC `bilevel1` 0 → 5 (the full `τ = 1` solve
  is what reaches the basin); net 154 at / 9 worse / 1 FAIL.
- **`τ = 1` first, the coupled solve warm at its `μ`**: crawls (`digits_128x3`
  251 iterations against 66 with `μ` restarted at `τ`).
- **A cleanup rung at `τ = 1e-8` before the finish**: converges in one
  iteration; the finish still fails. A monotone `μ` for the finish: no effect.
- **`κ < 1`** (`τ` reaches its floor earlier, so `μ ≪ τ` at the end): the
  finish still fails from the same endpoints.

The finish failures are the multipliers, not the point: the coupled and
scheduled endpoints agree to 1.4e-7 in `x`, but at degenerate pairs the
product-row multipliers differ (0.50 against 0.18 on rows at `G·H ≈ τ/2`).

### How to tell whether a fix works

Sweep against a baseline binary (`scripts/sweep-fixtures.sh`) and check the
two lbfgs `cc:` lines above; run `make -C benchmarks mpcc-compare` (the
transmission suite's wall time against `complementarity_continuation=schedule`)
and CALIPSO declared (`benchmarks/calipso`, the `declared` route, bunny-hop
seed 1). A fix should move those toward the schedule's numbers without
giving back MacMPEC's 158 / 0 or `relu_nn`'s time. The console line "the
coupled continuation did not reach an exact answer; resuming the schedule"
marks every fallback.

## MPCC: the exact-product stage is slow on dense degenerate QPECs

*Measured in the PR review of declared complementarity (jkitchin/pounce#1103),
on MacMPEC through the CLI. Recorded 2026-10-31.*

**Scope.** Models with declared complementarity pairs whose complementarity
rows are dense, under the default `complementarity_strategy=both` or `=exact`.
The qpecgen family in MacMPEC is the measured case. Models without pairs are
unaffected.

### What is wrong

The exact-product stage lowers each pair to `G·H = 0`, which has no strict
interior. On `qpec-100-2` (110 variables), run with
`complementarity_strategy=exact` to isolate the stage, it takes **561
iterations and 8.3 s**:

- the multipliers grow to about 1e12;
- it exits through the scale-relative dual-infeasibility floor (`dual_inf`
  612 against a dual scale of 1.6e12), not a converged KKT error;
- 5.0 s of the 8.3 s is Lagrangian Hessian evaluation, because qpecgen's
  complementarity rows are dense (11 020 Jacobian nonzeros on 110 variables).

Under the default `both`, the stage runs for every solve. `qpec-200-2` takes
**434 s** there, but that total also includes about 4 200 continuation
iterations, and the exact stage was not timed alone on it; how much of the
434 s is the exact stage is not measured. Only the `qpec-100-2` numbers
above isolate it.

### What has been tried

`complementarity_strategy=continuation` skips the stage and is the
workaround for a model known to behave this way. The stage is kept in the
default because on the Gate 0 corpus and the CALIPSO bunny-hop it finds
answers the continuation misses (see `docs/src/complementarity.md`).

**A relaxed exact stage (jkitchin/pounce#1121, 2026-11-01).** Solving
`G·H <= 1e-6` from the start, then a warm step to `1e-8` and the exact
finish, instead of `G·H = 0` cold. On the 28 MacMPEC models where the stage
costs or matters most (`complementarity_strategy=exact`) it takes 1,879
iterations against 25,712, converges on 26 against 20, and lands at or below
the published value on 23 against 19. `G·H <= 0` gains almost nothing (23,439
iterations): no interior either. A thinner `1e-8` from cold does worse than
`1e-6` (3,311 iterations, 20 at). Two things did not work:

- *Replacing* the cold solve: MacMPEC time −21%, but `water-net` moves from
  927 to 953 (the relaxed path settles on another branch) and two CALIPSO
  bunny-hop starts swap between 462.605 and 463.195. The two solves find
  different minima and neither dominates.
- Pinning each pair's smaller side at the relaxed point and solving the
  branch NLP (#1109's machinery): it does not recover `water-net`'s 927, and
  costs 266 extra iterations on `pack-comp1-32`.

So #1121 runs both and keeps the better certified answer. That is the
accepted cost recorded here: on the 167 public MacMPEC stubs, wall time
**+19%** (3,177 s to 3,767 s) and iterations **+37%** (78,431 to 107,563),
measured in clean sequential runs against `main` at `29ebd48`; CALIPSO's
declared route 170 s to 209 s. In exchange no answer is lost and three
improve (`dempe` 28.25 and `ex9.2.5` 5, both below their published values;
`pack-comp2p-8` reaches its published value). `pack-comp1p-32` still times
out at 300 s: the relaxed path alone solves it (0.704 against a published
0.653), and the cold solve's budget is what pushes the run over.

**Later (2026-11-02), and what remains.** Three rules cut MacMPEC's time
from 3,663 s to 2,528 s with no answer changed (see
`docs/src/complementarity.md`): a finish no longer replaces a better relaxed
point, the cold solve is skipped when the continuation and the relaxed path
agree, and a failed relaxation step is retried warm. Measured while finding
them: POUNCE's per-iteration cost is at parity with stock Ipopt + MUMPS on
the same lowered NLP (pack-comp1-32's `τ = 1` rung: 64 against 73 ms per
iteration), and stock Ipopt stalls on the cold `G·H = 0` solve the same way,
so the remaining time is in how many exact-product iterations the schedule
spends, not in the linear algebra. Three models timed out at 300 s when
this was written (the run at `5882e01`):

- `siouxfls`, `siouxfls1`: the first relaxation (`τ = 1`) alone takes about
  1,000 iterations and is solved twice (~110 s), and the `1e-7`/`1e-8` rungs
  struggle in restoration; the published value is already reached at
  `τ = 1e-6` (`siouxfls`) and `1e-7` (`siouxfls1`). With
  `complementarity_strategy=continuation max_cpu_time=900`, `siouxfls`
  reaches 208.258. An early stop when successive rungs stop moving the
  objective would trade product tightness for time.
- `pack-comp1p-32`: the continuation reaches 0.65304 (published 0.652979)
  in ~50 s; the exact stage's relaxed path and the cold solve take the rest.

The rerun after #1126 and gh#1128 (`benchmarks/macmpec/README.md`, Reference
result) moves this list: `pack-comp1p-32` and `siouxfls1` now finish inside
300 s, `siouxfls` still does not (676 s), and `qpec-200-2` joins it (335 s).

### How to tell whether a fix works

`qpec-100-2` under `complementarity_strategy=exact` should take a small
fraction of 561 iterations without changing its objective (time `qpec-200-2`
per stage first if it is to be the second witness), while the Gate 0 corpus
(`benchmarks/mpcc/declared.py`) stays at 60/60 cells at `f*`. (Neither model
is among the 167 public MacMPEC stubs #1121 measured; they are in the data
root's compiled set.) A cheaper schedule than #1121's two solves works if,
on the public MacMPEC stubs, it keeps 136 at / 12 better / 3 worse with
`water-net` at 927 and `dempe`, `ex9.2.5` below their published values, in
less than 3,767 s, and changes no CALIPSO declared-route answer. Integrated
approaches that tie the relaxation to the barrier parameter (relaxing the
products by an amount proportional to μ inside one solve) avoid the empty
interior and are the obvious direction. The MacMPEC suite and its harness are
in the benchmark data root (`macmpec/`).

---

## NLP: qcqp1500-1c hits the CPU limit at the default

*Not tracked. This entry is what remains of gh#1098, which was closed on
2026-11-01 when #1117 fixed `qcqp1500-1nc`. `qcqp1500-1c` has no known
path to a fix, so per the tracker's planned-work-only rule it has no
issue; its signature also differs from the crawl #1098 described. (#1098
also listed `waterx` reaching a different local minimum than Ipopt; that
is not a limitation for a local solver, so it is not recorded.) The 1nc crawl came
in with 49ab28aa (#1077), which put pounce on Ipopt's trajectory; Ipopt
crawls the same way. Recorded 2026-10-28, revised 2026-10-31 and
2026-11-01.*

**Scope.** The default exact-Hessian NLP path. The L-BFGS leg solves both
`qcqp1500` models.

### What was fixed

Mittelmann `qcqp1500-1nc` sat at μ ≈ 1.8e-6 for about 1200 iterations and
hit the 300 s CPU limit. The diagnosis is in
`dev-notes/qcqp1500-crawl-slack-perturbation.md`: inertia correction puts
`δ_s = δ_x` on every slack row, and on the thousands of inactive rows that
term is the whole curvature contribution, so successive regularized
directions disagree. Since 2026-11-01 the default is a row budget,
`δ_s = perturb_slack_fact · min(1, perturb_slack_max_rows / m_d) · δ_x`
with `perturb_slack_max_rows = 1000`, and `qcqp1500-1nc` solves in
128 iterations. Models with at most 1000 inequality rows are
untouched; `perturb_slack_max_rows=0` is upstream's rule exactly. The
Mittelmann before/after table is in the dev-note.
The budget's cost, Vanderbei `cresc132`, is the next entry.

### What is still wrong

- **`qcqp1500-1c`** still runs out the 300 s CPU limit, at the default and
  under every rule measured for 1nc. Its signature is different: the
  inertia correction never fires in the stretch it reaches (main and the
  budgeted default are identical through every row), so the slack
  perturbation has nothing to act on; the dual infeasibility stays near
  1e6–1e7 while `inf_pr` falls, with line searches of up to 17 trials, and
  it is around iteration 100–125 when the clock runs out.
  `ftb_centrality_floor=0.01` with
  `perturb_slack_fact=0.01` solves it (148 iterations) but turns the
  `discs` fixture into a false `InfeasibleProblemDetected`, so neither is
  on by default.

### What has been tried

- A fixed `perturb_slack_fact` (0.01–0.3): solves 1nc at 0.03–0.1, never
  1c; breaks `cresc4` / `discs` at 0.01 and 0.1; moves 22 CLI tests at 0.03.
- `(n/m_d)^p`, norm-based and per-row `Σ_s`-scaled rules for `δ_s`: each
  measured and rejected in the dev-note (wrong basins on `discs`, retries
  on `cresc4`, 2556 iterations on `pooling_rt2stp`).
- `ftb_centrality_floor` (γ > 0): solves 1c with a small factor, false
  infeasibility on `discs`; opt-in, default 0.

### How to tell whether a fix works

1. `qcqp1500-1c` solves within 300 s at the default, and `qcqp1500-1nc`
   stays at or near 128 iterations.
2. `discs` and `cresc4` keep their verdicts on the first attempt (no
   second-opinion rescue, `2nd=-` in the sweep).
3. `scripts/sweep-fixtures.sh` on both legs, diffing `2nd=` and `mu=`, and
   `benchmarks/grid` (15 of 22 models above the row cap) not worse than
   909 iterations.

### When to revisit

- The inertia-correction or `δ_s` handling changes.
- A user reports a large QCQP stalling at a fixed μ with `inf_pr` near 1e-3
  on a model with fewer than 1000 inequality rows — the budget does not
  reach it, and `perturb_slack_fact` is the knob.

---

## NLP: the slack row budget loses `cresc132` (and slows `laptime`)

*gh#1127. Measured on Apple silicon (14 cores) at #1126's head `6b669294`,
default options except the cap. Recorded 2026-11-01.*

**Scope.** The default exact-Hessian NLP path on models with more than
`perturb_slack_max_rows = 1000` inequality rows, which the gh#1098 budget
reaches. Complementarity models are not affected: the MPCC driver sets the
cap to 0 (#1126).

### What is wrong

Vanderbei `cresc132` (n = 6, m_d = 2654, factor 0.38) no longer solves
within 300 s at the default. With the cap off it solves as v0.14.0 does
(610 iterations including the adaptive re-solve, f = 0.85771). So in the
suites the budget gains `qcqp1500-1nc` (Mittelmann) and loses `cresc132`
(Vanderbei), one each.

The cap-off and cap-1000 iteration tables agree through iteration 21,
which includes the first restoration call. They diverge at 22, the first
iterate where regularization fires. From there the cap-1000 run falls back
into restoration and stays there.

### It is not a rule that is wrong, it is a model that is chaotic in `δ_s`

`cresc132` alone, by cap (factor = cap/2654):

| `perturb_slack_max_rows` | factor | result |
|---|---|---|
| 0 (upstream) | 1 | Solve_Succeeded, 610 it, only after the first attempt fails and the adaptive re-solve rescues it |
| 2500 | 0.94 | Solve_Succeeded, 247 it, 25 s |
| 2000 | 0.75 | **Infeasible_Problem_Detected**, 591 it (false) |
| 1000 (default) | 0.38 | CPU limit, stuck in restoration |
| 500 | 0.19 | **Infeasible_Problem_Detected**, 153 it (false) |

The response is not monotone in the factor. Even upstream's own rule solves
this model only on its second attempt.

`large_scale/laptime` (n = 58 014, m_d = 6000, factor 0.17) shows the same
pattern without a status change. Its iteration counts are 327 at cap 0, 331
at 500, **756** at 1000 and 284 at 2000. Cap 1000 lands in a different,
slightly lower local minimum, f = 65.29277 (the others reach 65.2947, as does
Ipopt): re-solved 2026-11-02 at `f1787277` it is a genuine KKT point (scaled
constraint violation 1.1e-10, dual infeasibility 7.1e-11), not a loose stop.
At cap 1000 it took 310 s of wall-clock time. That is past 300 s, but the
sweep's limit is on CPU time (`max_cpu_time`), so the run still ended
`SolveSucceeded`; a harness with a 300 s wall-clock limit would lose it.
The CPU limit is no wider a margin: the 0.15.0 sweep finished it at 756
iterations, while a standalone re-solve of the same binary under
`max_cpu_time=300` ran out at iteration 729 (244 s wall). Expect it to flip
to `Maximum_CpuTime_Exceeded` between otherwise identical runs.

### What the cap does on every model it reaches

All 87 non-LP/QP/MPCC models in the benchmark data with m_d > 1000 were run at
cap 1000 against cap 0 (300 s CPU limit, 5 at a time):

- **Better under the cap (iterations):** `qcqp1500-1nc` (timeout → 128),
  `pglib_case1888_rte` (grid 501 → 183, ECF 212 → 166),
  `pglib_case10000_goc` (polar 131 → 85, grid 99 → 87), the other 2000/500/179
  grid cases, `arki0003` 448 → 335, `qcqp1000-2nc` 289 → 184, `robot_b`
  271 → 180, `robot_c` 222 → 116, `svanberg` (both suites), `nql180`,
  `optmass`, `oet2` 63 → 34, `oet7` 297 → 189, and `NARX_CFy`, which lands in
  a different local minimum.
- **Worse:** `cresc132` (lost), `laptime` (327 → 756), `gausselm`
  (871 → 1311, a different local value −17.848 against −17.835), `robot_a`
  145 → 193, `arki0009` 328 → 332, and `ngone` 40 → 46 (it also reaches a
  lower local value, −0.63597 against −0.63328).
- Everything else is identical, including the 25 that route to a convex
  engine.

### What has been tried

- **Gating the budget on `n` or `m_d/n`** (the issue's suggestion). The
  losers sit at n = 6, 1495 and 58 014 and m_d/n = 442, 1.8 and 0.10.
  The winners sit in between (`qcqp1500-1nc` 1500 and 6.7, the grid cases
  0.4–2.4, `robot_b` 52, `oet2`/`oet7` n ≤ 7 and m_d/n ≈ 140–330). No threshold on
  either separates them. And `cresc132`'s table above shows that no factor
  rule can "keep" it, because neighbouring factors fail in both directions.
- **Other cap values.** `dev-notes/qcqp1500-crawl-slack-perturbation.md`
  measured 500/1000/2000 on the 19 Mittelmann models above the cap: every
  value solves `qcqp1500-1nc`, and 1000 has the lowest iteration sum. Both
  losers here fail or slow down at 1000 and recover at a nearby value, so
  moving the constant would trade them for other models' luck.

### How to tell whether a fix works

1. `cresc132` solves within 300 s at the default, *and* it keeps solving at
   caps either side of the default (to show it isn't another lucky value).
2. `qcqp1500-1nc` still solves, and the 87-model comparison above shows no
   new status losses. The model list and runner are reproducible from the
   description: `n`, `m`, `m_d` read from each `.nl` header, m_d > 1000.
3. The fixture sweep cannot see this (no fixture has more than 1100 rows,
   and only `manyineq_concave_qp` is above the cap), so use the Vanderbei
   and Mittelmann suites.

### Workaround

`perturb_slack_max_rows=0` restores upstream's rule for one model.

---

## Active-set QP: the per-pass prune cap is a tuned number (QSCFXM3, QSCSD6 slowdowns)

*Was gh#1081. Introduced by #1078 (gh#1076 item 3). Recorded 2026-10-27.*

**Scope.** Only `solver_selection=qp-active-set`. The default `auto` sends
convex QPs to the interior-point engine (`cvx-qp`), which never runs this code.

### What is wrong

`pounce-qp`'s Schur loop prunes an over-full active set to an independent
subset before it factors it. At a degenerate vertex the row the prune drops
is often a genuine blocker. The next step moves into it, it re-enters at
`α = 0`, and the prune drops it again. Left alone, that cycle runs to
`max_iter`. The fix in #1078 lets the counting prune run at most
`COUNT_PRUNE_CAP = 32` times per pass (`crates/pounce-qp/src/solver.rs`,
`schur_reset_rank_repaired`). The cap is measured, not derived, and the
outcome is not monotone in it:

- QGFRDXPN fails below 24.
- QFORPLAN solves at 16, 20, 32 and 40. It cycles to `MaxIter` at 24, 28,
  48, 64 and with no cap.

Both models are pinned by
`crates/pounce-cli/tests/issue_1076_qgfrdxpn_active_set_prune_cap.rs`, so a
change that pushes either across the window fails CI rather than shipping.

### How big

Small. The cap cost two Maros-Meszaros problems more than 2× in time; both
still solve. Measured on arm64, serial, base `ab59784a` vs #1078:

| problem | base | #1078 | ratio |
|---|---|---|---|
| QSCFXM3 | 229 s | 494 s | 2.16× |
| QSCSD6 | 19 s | 47 s | 2.41× |

QSHELL (1.45×) and QGROW22 (1.41×) also slowed. That comes from #1078's
phase-1 row-feasibility check, not the cap. Across the 39 problems both
versions solve, total wall time fell from 4087 s to 3415 s, and QGFRDXPN went
from failing to solving. The trade is a net gain.

**Wall-time exposure.** The convex arm has no deadline unless
`max_wall_time` is passed. On a 2.8 GHz Xeon cloud container (x86_64, idle,
default options) QSCFXM3 took **1119 s** (`SolveSucceeded`, 18 092
iterations). It would fail any `max_wall_time` below that. That container
runs about 3× slower than the arm64 machine above (QGFRDXPN: 26 s vs 6–9 s),
so read the number as "much slower hardware tips it over", not as an x86
penalty.

### Why each problem is slow

- **QSCFXM3:** its first phase-1 pass reaches the cap, stands down, and then
  runs unpruned for 21 310 working-set changes. Only the third attempt
  solves it; on base the second attempt did.
- **QSCSD6:** its first pass enters an exact prune cycle. The cycle guard
  ends it, and the solve then pays for the recovery path. Under the old rule
  that pass finished directly.

### Already tried, did not work

From #1078: per-slot limits, objective-moved conditions, ending the pass on
the first repeat, and caps from 8 to 64. QFORPLAN also fails at caps 24, 28,
48 and 64.

**Re-entry precedence** (2026-10-27, x86_64, iteration counts). When the
counting prune runs, slots it already dropped in this pass get
`PRECEDENCE_WEIGHT` in `independent_active_subset_sparse_weighted`, so the
probe drops a different member of the dependent set:

| problem | cap 32 (shipped) | old rule, no cap | re-entry, no cap | re-entry + cap 32 |
|---|---|---|---|---|
| QGFRDXPN | optimal, 3553 | — | optimal, 4151 | optimal, 4151 |
| QFORPLAN | optimal, 12465 | `MaxIter`, 12963 | optimal, 13445 | optimal, 13016 |
| QSCSD6 | optimal, 5874 | — | **`MaxWallTime` at 1800 s**, 7573 | optimal, 5874 |
| QSCFXM3 | optimal, 18092 | — | not run | optimal, 17605 |

Without the cap, the rule solves both pinned models, which no rule tried in
#1078 did. It loses QSCSD6: the cycle stops repeating exactly but does not
end. With the cap, it changes almost nothing. So protecting a re-entering
row is part of the answer but not all of it, and QSCSD6 is the model a
cap-free rule has to answer for.

### What a fix would look like

The cap stands in for a missing anti-cycling rule. The textbook candidates:

- a lexicographic or Bland-style choice of entering and dropped row;
- a stronger EXPAND (the loop already carries GMSW `expand_tol`);
- a small perturbation of the bounds at the vertex.

Any of these should end cycling whether it repeats exactly or not, which is
what QSCSD6 needs.

**Suggested first step:** trace QSCSD6 under re-entry precedence with no cap
(`POUNCE_QP_TRACE=1`) to see what the uncapped run spends its iterations on.

**Acceptance:**

1. QGFRDXPN, QFORPLAN, QSCSD6 and QSCFXM3 all solve with no cap.
   `benchmarks/qp/generate_nl.py <NAME>` builds them; run each with
   `solver_selection=qp-active-set`.
2. Then the full 138-problem `benchmarks/qp` run under
   `solver_selection=qp-active-set`, compared against the current tree for
   status and wall time.
3. Then `scripts/sweep-fixtures.sh` (both legs). This is a trajectory change
   to a core loop, and every line that moves needs an explanation.

### Also unguarded

The stationarity-prune cycle guard added in #1078 (fix 1) has unit tests for
its decision logic. No test fails if its call is removed from the loop,
because no small QP could be built that drives a real pass into that cycle.

### When to revisit

- The active-set QP becomes a default route for some class of model.
- The gh#1076 pins start failing after a FERAL, probe-pivoting or phase-1
  change.
- A user reports QSCFXM3-like timeouts under `qp-active-set`.

---

## Convex LP/QP: Farkas certificates are a tolerance at the far-feasible boundary

*Recorded 2026-11-01 with the fix that introduced `FARKAS_MARGIN_MIN`
(`crates/pounce-convex/src/ipm.rs`, `FarkasMargin`). Not tracked: no rule
can close it (see "What is wrong").*

**Scope.** The ratio Farkas test in `detect_infeasibility_with` (every
HSDE and direct-driver iteration of the convex QP/QCQP/conic arm) and the
free-column residual waiver in `active_set::certifies_primal_infeasible`.
The exact box proof (`farkas_box_proof`) and presolve verdicts are not
affected.

### What is wrong

An approximate ray `(y, z)` with `q = Aᵀy + Gᵀz ≠ 0` and
`β = bᵀy + hᵀz < 0` proves only that no feasible `x` has
`‖x‖₁ < |β|/‖q‖∞`. With `s = (|b|ᵀ|y| + |h|ᵀ|z|)/‖|A|ᵀ|y| + |G|ᵀ|z|‖∞`,
the data's natural scale, that radius equals `(ρ_h/ρ_g)·s`. The test
accepts at `ρ_h/ρ_g ≥ 1e4`. Two consequences follow, and no finite
threshold removes either:

1. **False certificates beyond `1e4·s`.** Take `g·(x0 + x1) ≤ −1` and
   `−g·(x0 + (1−ε)·x1) ≤ 0`, with no bounds. It is feasible at
   `x1 ≤ −1/(gε)`, its ray `z = (1, 1)` has radius `2/ε`, and it is
   certified infeasible by both engines when `ε ≤ 2e-4` and the absolute
   tests also pass (`gε ≤ 1e-10`). The unit test
   `margin_threshold_borderline` pins this boundary at `ε = 1e-4`
   (accepted) and `ε = 4e-4` (rejected).
2. **Lost certificates below `1e4·s`.** Integer `A` with an exact null
   vector, `Σh = −δ`, data near `1e6` and `δ = 1e-6` (a relative margin
   near `1e-13`): the IPM certifies 20 of 39 and active-set 18 of 39.
   The rest end `dual_infeasible`, `numerical_failure`,
   `iteration_limit`, `optimal_inaccurate` or, on active-set only,
   `optimal` (3 of 39). The feasible twins (`Σh = +δ`) show the same
   distribution, and the old test certified 14 of those 39 as infeasible,
   so at that margin the model's verdict lies below its data's rounding.

### Measurements (default options, before → after)

| corpus | before | after |
|---|---|---|
| netlib-infeas (29) | 29 infeasible | 29, same iteration counts |
| MIPLIB 2017 Infeas, nnz ≤ 150k (49) | 32 infeasible, 17 unbounded | same; `lr1dr04vc05v17a-t360_if` 36 → 37 iterations (first ray radius 2.0e3) |
| `benchmarks/qp` (138) | 137 optimal, BOYD2 timeout | identical, iteration counts included |
| fixture sweep (134 × 2 legs) | — | only `farkas_far_feasible_no_box` moves (Infeasible → Diverging) |
| synthetic, feasible (180 + 360) | 15 certified infeasible | 0 |
| synthetic, infeasible (360) | 360 / 360 (both engines) | IPM 341, active-set 339; every loss at `δ = 1e-6`, data `1e6` |

Ratio-path certificates on netlib-infeas have radius `2.4e8` (`pang`)
to `1.5e12`. Every false certificate measured had radius `0.16`–`2.4`.
`1e4` is the log-midpoint of that gap and the order of Clarabel's reduced
`AlmostPrimalInfeasible` tolerance (`5e-5`).

### What has been tried

- **K = 10…1e8.** Every `K ≥ 10` removes all 15 measured false
  certificates. On the infeasible family the IPM keeps 360/355/341/341/305/292
  at `K = 10, 1e2, 1e3, 1e4, 1e6, 1e8`, and netlib is unchanged up to
  `1e8`.
- **A componentwise `ρ_g = maxⱼ |qⱼ|/magⱼ`.** Rejected genuine netlib
  certificates (`ex72a`, `ex73a`, `ω ≈ 0.59`) because some columns have
  tiny magnitudes. The max-norm version is used instead.

### How to tell whether a fix works

A fix would need a status between certified and not, like Clarabel's
`AlmostPrimalInfeasible`, or an exact (rational or interval) check of the
ray. It works if `farkas_far_feasible_no_box.nl` stays uncertified, the
`two_row` borderline in `ipm.rs` with `ε = 1e-4` stops being certified,
and the netlib/MIPLIB counts above hold.

## Convex LP: a near-constant infeasible row needs equilibration (gh#1143)

`ε·x₀ ≤ −1` beside `x₀ ≥ 0` is infeasible for every `ε > 0`. Two paths no
longer reach a verdict on it at the smallest `ε`. Before gh#1143 they reported
it, wrongly, as unbounded (or rightly by accident, below); they now end
`NumericalFailure`:

| path | ε | before gh#1143 | now |
|---|---|---|---|
| `solve_qp_ipm`, default | 1e-14 | `DualInfeasible` (wrong) | `PrimalInfeasible`, 24 it |
| `solve_qp_ipm`, `equilibrate = false`, HSDE | 1e-14 | `DualInfeasible` (wrong) | `NumericalFailure`, 159 it |
| `solve_qp_ipm`, `equilibrate = false`, direct | 1e-14 … 1e-10 | `PrimalInfeasible`, 1 it (by accident) | `NumericalFailure`, ~100 it |
| `solve_socp_ipm` (`solver_selection=socp`), default, orthant or orthant + SOC | 1e-14 | `DualInfeasible` (wrong) | `NumericalFailure`, 155–159 it |
| `solve_socp_ipm`, same | 1e-10 | `DualInfeasible` (wrong) | `PrimalInfeasible`, 13 it |

What reaches the verdict through `solve_qp_ipm` is its gh#293 equilibrated
retry after a failed solve. The conic entry has no such retry, and Ruiz
equilibration is only sound per row on orthant rows, so adding one there is a
separate change to that arm's trajectory, not part of this fix.

Before gh#1143 the direct unequilibrated driver answered `PrimalInfeasible`
at iteration **1** — not by finding a Farkas ray, but by accepting the
non-diverging iterate `x ≈ (8.2, 0.06)` as a recession direction (row 0's
`(Gd)₀ = 8e-14` passed the absolute `1e-7·‖d‖` slack), issuing a false
`DualInfeasible`, and having the objective-free twin relabel it. The answer
was right by accident; the fix (a per-row recession slack, `min(1, ‖Gᵢ‖∞)`)
removes the false certificate and with it the accident.

### How to tell whether a fix works

In `crates/pounce-convex/tests/issue1143_tiny_row_false_unbounded.rs`,
`conic_entry_near_constant_infeasible_row_is_not_unbounded` can assert
`PrimalInfeasible` at `ε = 1e-14` too (it asserts only "not
`DualInfeasible`" there today), `near_constant_infeasible_row_is_primal_infeasible`
passes with `equilibrate: false` added to its option sets, and both
`*_satisfied_row_stays_unbounded` tests keep `DualInfeasible`.
