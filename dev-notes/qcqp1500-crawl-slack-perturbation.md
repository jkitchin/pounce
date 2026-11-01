# qcqp1500-1nc crawl: slack-block perturbation and the centrality floor (gh#1098)

## The symptom

On the default exact-Hessian path, Mittelmann `qcqp1500-1nc` (1500
variables, 10 508 constraints) sat at μ = 1.8e-6 for about 1200 iterations
and hit the 300 s CPU limit. During that stretch `inf_pr` oscillated between
1e-3 and 1e-2, the step fractions were tiny, and `δ_w` was non-zero almost
every iteration. Ipopt 3.14 crawls the same way: it was still unsolved at
1800 s, after 2827 iterations. #1077 put pounce on Ipopt's trajectory, so
it inherited the crawl.

## What the crawl is

Instrumenting the crawl iterates gave this picture:

- It is a walk from vertex to vertex of a nonconvex polytope, at about 15
  iterations per net change of active set.
- **Fraction to the boundary.** When a blocking pair (slack, multiplier)
  sits on the central path, `s·v ≈ μ`, the τ rule
  (`τ = max(τ_min, 1 − μ)`) cuts it to about `μ²`. Recentering it then
  costs 5–10 iterations as soon as the next direction reverses.
- **Inertia correction.** It adds `δ_x` to the Hessian block and,
  following Ipopt, the same `δ_s = δ_x` to the slack block.
  - The slack block already carries `Σ_s = v/s`. That is small on inactive
    inequalities, so the added `δ_s` mostly stiffens the inactive slacks.
  - Across thousands of inactive rows, that term dominates the curvature
    along a direction.
  - Successive regularized directions then disagree, and the sawtooth in
    `δ_w` follows.

**Refuted.** A first explanation was that the crawl was `δ_w` on the slacks
alone, removable by a smaller `δ_s` applied only once a stall is detected.
A stall-triggered prototype did not get out of a crawl that had already
started. A small `δ_s` has to be in effect from the start, before the
trajectory reaches the slow basin.

## The two remedies

1. **`perturb_slack_fact`**, so that `δ_s = fact · δ_x`. Upstream is 1.
   Wired through `PDPerturbationHandler::delta_s_fact`. It changes only
   the slack block. `δ_x` stays on the Hessian, where the inertia defect
   is, and the inertia test still decides whether a perturbation is
   accepted.
2. **`ftb_centrality_floor` (γ)**: Mehrotra's step-length heuristic
   (Mehrotra 1992 §5; Wright, *Primal-Dual Interior-Point Methods*,
   eq. 10.9).
   - It keeps a `1 − γ` floor on the fraction, and steps beyond it only
     while the blocking pair's product stays `≥ γμ`.
   - Only pairs that are currently centered get the floor.
   - The floored step is capped at the τ rule's step, so it can only
     shorten.
   - It is not applied in restoration.
   - See `crates/pounce-algorithm/src/ftb_floor.rs`.

## Why the floor is capped and why it is off

The first version was not capped.
`issue_390_nonlinear_equality_scale` caught the problem.

- At iteration 0 (μ = 0.1) the blocking pair had `s·v = 9`.
- "Keep `s·v ≥ γμ`" then allowed a longer dual step than τ.
- That put the multiplier at `γμ/s` instead of about `μ/s`: less central
  than upstream leaves it, which is the opposite of what the rule is for.
- At row scale 1e-8 the infeasibility certificate became
  `Restoration_Failed`.

Capping fixed that test. Three others still change verdict with γ = 0.01:

- `issue_884_biactive_dual_divergence`: `ErrorInStepComputation`, where the
  fixture is supposed to reach a certificate without the retry.
- `masked_certificate_fuzz`: with the certificate veto off and
  `mu_strategy=adaptive`, it reports `SolveSucceeded` at f = 0.125 against
  a bound optimum of 0.1. Under the default veto the answer is still
  correct.
- `line_search_options_wiring`: the fixture's trajectory no longer reacts
  to `filter_max_margin`.

In the fixture sweep, γ = 0.01 also turns `discs` into a final
`InfeasibleProblemDetected`. So γ stays opt-in, at default 0.

## Measurements

Baseline: main `26991b7b`.

| config | γ | slack fact |
|---|---|---|
| main | 0 | 1 |
| A | 0 | 0.01 |
| B | 0.01 (capped) | 0.01 |
| C (best opt-in) | 0 | 0.03 |

The three lanes ran at the same time, so the times are approximate.

**Mittelmann:**

| config | solved | iterations | time |
|---|---|---|---|
| main | 44/47 | 5182 | 2216 s |
| A | 45/47 | 4324 | 1818 s |
| B | 46/47 | 4575 | 1814 s |
| C | 45/47 | 4355 | 1674 s |

| model | main | A | B | C |
|---|---|---|---|---|
| qcqp1500-1nc | CPU limit | 199 | 143 | 228 |
| qcqp1500-1c | CPU limit | CPU limit | 148 | CPU limit |
| henon120 | 267 | 67 | 69 | 80 |
| lane_emden120 | 84 | 21 | 21 | 25 |
| qcqp500-3c | 330 | 138 | 140 | 177 |
| qcqp750-2c | 75 | 218 | 307 | 198 |
| dirichlet120 | 56 | 66 | 154 | 79 |

**Water:** all configs solve 6/6.

| config | total iterations | waterx objective |
|---|---|---|
| main | 1401 | 1019.7 |
| A | 1248 | 927.1 |
| B | 1336 | 963.9 |
| C | 1331 | 1151.5 |

**Fixture sweep, 252 lines.** All lines that move are on the exact leg.

| config | lines moved | status changes and retries |
|---|---|---|
| A | 14 | `cresc4` and `discs`: false `Infeasible_Problem_Detected` on the first attempt, rescued by the second-opinion ladder (total 824 and 484, against 69 and 177 on main) |
| B | 47 | `discs`: false infeasible, final status |
| C | 14 | no status loss and no new retry; `ssebnln` acceptable → optimal |

Config C, every line that moves:

- **Different minimum** (nonconvex models):
  - `autocorr_bern55-06`: better, −2368 against −2304.
  - `hs13_bigstart`: 0.98168 against 0.98493.
- **Same answer, fewer iterations:**
  - `pooling_rt2stp` 201 → 130
  - `discs` 177 → 166
  - `haverly_multistart_t3_brf0` 37 → 31
- **Same answer, more iterations:**
  - `csfi2` 21 → 37
  - `deb7` 180 → 195, still acceptable
  - `mpcc_qpec_small_biactive` 29 → 31
  - `nonconvex_qcqp` 59 → 61
- **Last digit of the objective only:**
  - `cresc4`
  - `nonconvex_qp_ineq`
- **Unbounded models**, same status, diverging at a different rate:
  - `unbounded_cubic` 290 → 65
  - `unbounded_exp` 23 → 31
- **`ssebnln`**: acceptable → optimal. It still uses the `mu_strategy`
  retry.

**The value does not respond smoothly.** On `cresc4`, `discs` and
`qcqp1500-1nc`:

- 0.01: false first-attempt infeasibility on two fixtures.
- 0.03: clean.
- 0.1: `discs` locally infeasible.
- 0.3: `cresc4` needs a retry, and `qcqp1500-1nc` takes 446 iterations /
  275 s.

0.03 was picked from those runs and then confirmed on the full sweep,
Water and Mittelmann. It is a measured choice, not an optimum.

## Why both options ship opt-in

With 0.03 as the default, the fixture sweep was clean, but 22 `pounce-cli`
regression tests that pass on main failed (`cargo test --workspace`).

- **Most are premise tests.** `csfi2` now converges on the first attempt,
  so the retry, fallback and one-verdict tests built on it (gh#748, #757,
  #1042, #616, `one_verdict_per_run`) no longer reach their branch. Two
  witness tests (#534, #592) also lose their premise.
- **Some assert an answer:**
  - `hs13` from the remote start lands at 0.98168, where Ipopt gets 0.98493.
  - `issue_250` reads `autocorr_bern55-06` at −2368 as a worse basin.
  - `mpcc_qpec_small_biactive` fails under the options the #884 and #981
    tests use.

The sweep runs every fixture at stock options. These tests run a few
fixtures at non-default options and check specific branches, so they see
what the sweep cannot.

Both options therefore default to upstream's behaviour (`1` and `0`), and
the workspace tests pass unchanged. What is still needed to make 0.03 the
default is recorded in `dev-notes/known-limitations.md`.

## Lesson: diff the `2nd=` column too

My first reading of config A's sweep compared only status and `it=`, and
reported "nothing that converged stopped converging". That was true of the
final status. But two fixtures had started being declared infeasible on the
first attempt, and the second-opinion ladder had hidden it. A first attempt
that fails and is rescued only shows in the `2nd=` column, so diff it along
with the other columns.

## Making it the default: a row budget instead of a constant (2026-11-01)

The constant `0.03` could not be the default because it acts on every
model, and the models it hurts are the small ones the regression suite is
built on. The mechanism it addresses, on the other hand, is a function of
how many *inactive inequality rows* the perturbation lands on: along a
direction `z` the slack rows contribute `Σ_i (Σ_s,i + δ_s) (J_d,i z)²`, and
on an inactive row `Σ_s,i` is tiny, so there `δ_s` is the whole term and the
sum grows with the row count. So the rule that ships is

```text
δ_s = perturb_slack_fact · min(1, perturb_slack_max_rows / m_d) · δ_x
```

with `perturb_slack_max_rows = 1000` and `perturb_slack_fact = 1`. It
holds `δ_s · m_d` — the total the perturbation adds to the slack block's
trace — at the level a 1000-row model would see. Below the cap it is
upstream's rule exactly; on `qcqp1500-1nc` (10 005 rows) the factor is
0.1. `perturb_slack_max_rows=0` disables the cap, and with
`perturb_slack_fact=1` that reproduces upstream Ipopt. Implementation:
`PdPerturbationHandler::delta_s_fact_effective`, fed the slack-row count
by `PdFullSpaceSolver::solve_once` from `Σ_s`'s dimension (so restoration,
whose NLP has the same `d` rows, sees the same factor).

### Rules measured and rejected first

Every candidate was run on a key set — `cresc4`, `discs`, `csfi2`,
`hs13_bigstart`, `autocorr_bern55-06`, `mpcc_qpec_small_biactive`,
`pooling_rt2stp`, `nonconvex_qcqp`, `unbounded_cubic`, `ssebnln`,
`deb7` — chosen because they are the fixtures `0.03` moves.

- **`(n/m_d)^p`.** At `p=1` (0.75 on `cresc4` and `discs`) `cresc4`
  needs the μ fallback after a first attempt that runs to `max_iter`
  (217 iterations in the end) and `discs` converges to a wrong minimum
  (14.4495, not 12.0001). At `p=2` `discs` is a false infeasibility
  rescued by the second-opinion ladder (`tot=733`). The ratio does not
  separate the models that need the reduction from the ones it hurts.
- **Norm-based factors** `min(1, p/‖J_d‖_F²)`, `min(1, p/(‖J_d‖_1‖J_d‖_∞))`.
  `cresc4` 96–107 iterations or a retry, `discs` infeasible at
  `p=1e3`, `pooling_rt2stp` 2556 iterations, `autocorr` at −2336 and
  `pooling` at −4391 at `p=1e3` on the second form. The norms also swing
  by orders of magnitude between iterations (`hs13`: `‖J_d‖_F²` from 1e4
  to 1e-13), so the factor is not even stable within a solve.
- **Per-row `δ_s,i` scaled by `Σ_s,i`.** Refuted from diagnostics before
  it was built: on `discs` and `cresc4`, during the regularized phases
  (`δ_x` of 1e3–1e5) most rows have `Σ_s,i < 0.1·δ_x` — the same
  signature as `qcqp1500-1nc` — so a per-row rule reduces to a near-zero
  factor on exactly the models a small constant breaks.
- **The cap's value.** `min(1, 300/m_d)` moves `deb7` (351 rows, factor
  0.85) from `Solved_To_Acceptable_Level` in 180 iterations to
  `Solve_Succeeded` in 110 — through the μ fallback, `tot=441`. A 15%
  change on a 351-row model is visible, so the cap has to sit above every
  fixture: `min(1, 1000/m_d)` leaves the key set identical to main. At a
  fixed factor on `qcqp1500-1nc` itself: 0.03 → 228 iterations, 0.05 →
  231, 0.1 → 128; the budget lands on 0.1.

### What the budget changes, measured

Candidate at defaults against main `29ebd488`, same host, several lanes at
once, so wall-clock is indicative only; iteration counts are exact.

- **Fixture sweep** (`scripts/sweep-fixtures.sh`, both legs): identical,
  252 of 252 lines — *by construction*, since the largest fixture has
  351 inequality rows. That is a statement about the corpus, not
  evidence about the branch; `manyineq_concave_qp` (1100 rows,
  `scripts/gen-manyineq-nl.py`, `issue_1098_slack_row_budget.rs`) was
  added so the sweep has a line on the budgeted branch (29 → 28 on the
  exact leg when added; L-BFGS never raises `δ_x`, unchanged at 20).
- **Water** (6 models, at most 84 rows): identical.
- **`benchmarks/grid`** (AC-OPF, 15 of 22 above the cap): 1261 → 909
  iterations, 22/22 solved both. `pglib_case1888_rte` 501 → 183,
  `pglib_case2000_goc` 61 → 46, `pglib_case10000_goc` 99 → 87,
  `pglib_case500_goc` 49 → 44, `pglib_case179_goc` 44 → 42; nothing
  worse.
- **`benchmarks/grid_ecf`** (15 of 20 above the cap): 1031 → 981.
  `pglib_case1888_rte` 212 → 166, `pglib_case2000_goc` 51 → 45;
  `pglib_case10000_goc` 98 → 99 and `pglib_case162_ieee_dtc` 49 → 50.
- **Mittelmann** — see the table below.

#### Mittelmann, 47 models, candidate at defaults against main `29ebd488`

Status and iteration count are the per-model JSON report's (the promoted
solve's count, where the gh#983 scale audit re-solved); a model with no
report (timed out) takes the driver's scraped status and the last row of
its iteration table. Wall time is the driver's measurement of the whole
process, and both runs shared the host with other sessions' builds (load
average 80–100 on 14 cores), so it is indicative only — and it is why two
lines below are timeouts that are not trajectory changes. Bold lines moved.

| model | main | candidate | wall s (main → cand) | note |
|---|---|---|---|---|
| **NARX_CFy** | Optimal, 378 it | Optimal, 305 it | 209 → 180 | objective 0.008589722 → 0.008525684 |
| **WM_CFy** | CPU limit, 85 it | CPU limit, 171 it | 300 → 300 |  |
| **arki0003** | Optimal, 448 it | Optimal, 335 it | 12 → 4 |  |
| **arki0009** | Optimal, 328 it | Optimal, 332 it | 16 → 22 |  |
| bearing_400 | Optimal, 16 it | Optimal, 16 it | 20 → 27 |  |
| camshape_6400 | Optimal, 88 it | Optimal, 88 it | 14 → 18 | scale audit: first solve 88 → 88 it; re-solve kept → kept |
| **clnlbeam** | Optimal, 20 it | CPU limit, 2295 it | 156 → 300 | scale audit: first solve 2465 → None it; re-solve promoted → none |
| cont5_1_l | Optimal, 12 it | Optimal, 12 it | 15 → 18 | scale audit: first solve 16 → 16 it; re-solve promoted → promoted |
| cont5_2_1_l | Optimal, 42 it | Optimal, 42 it | 130 → 188 | scale audit: first solve 42 → 42 it; re-solve kept → kept |
| cont5_2_2_l | Optimal, 48 it | Optimal, 48 it | 37 → 25 | scale audit: first solve 48 → 48 it; re-solve kept → kept |
| cont5_2_3_l | Optimal, 55 it | Optimal, 55 it | 30 → 28 | scale audit: first solve 55 → 55 it; re-solve promoted → promoted |
| cont5_2_4_l | Optimal, 10 it | Optimal, 10 it | 12 → 20 | scale audit: first solve 14 → 14 it; re-solve promoted → promoted |
| corkscrw | Optimal, 19 it | Optimal, 19 it | 18 → 17 | scale audit: first solve 273 → 273 it; re-solve promoted → promoted |
| dirichlet120 | Optimal, 56 it | Optimal, 56 it | 95 → 116 |  |
| dtoc1nd | Optimal, 8 it | Optimal, 8 it | 1 → 1 |  |
| dtoc2 | Optimal, 5 it | Optimal, 5 it | 8 → 7 |  |
| elec_400 | Optimal, 126 it | Optimal, 126 it | 87 → 52 |  |
| ex1_160 | Optimal, 15 it | Optimal, 15 it | 4 → 4 | scale audit: first solve 10 → 10 it; re-solve promoted → promoted |
| ex1_320 | Optimal, 15 it | Optimal, 15 it | 21 → 17 | scale audit: first solve 9 → 9 it; re-solve promoted → promoted |
| ex4_2_160 | Optimal, 21 it | Optimal, 21 it | 10 → 6 | scale audit: first solve 21 → 21 it; re-solve promoted → promoted |
| ex4_2_320 | Optimal, 25 it | Optimal, 25 it | 32 → 56 | scale audit: first solve 17 → 17 it; re-solve promoted → promoted |
| ex8_2_2 | Optimal, 67 it | Optimal, 67 it | 2 → 3 |  |
| ex8_2_3 | Optimal, 71 it | Optimal, 71 it | 1 → 4 |  |
| gasoil_3200 | Optimal, 14 it | Optimal, 14 it | 3 → 4 |  |
| henon120 | Optimal, 267 it | Optimal, 267 it | 97 → 135 |  |
| lane_emden120 | Optimal, 84 it | Optimal, 84 it | 70 → 175 |  |
| marine_1600 | Optimal, 13 it | Optimal, 13 it | 2 → 3 |  |
| **nql180** | Optimal, 37 it | Optimal, 33 it | 79 → 71 |  |
| **optmass** | Optimal, 32 it | Optimal, 31 it | 2 → 2 |  |
| pinene_3200 | Optimal, 13 it | Optimal, 13 it | 3 → 3 |  |
| qcqp1000-1nc | Optimal, 177 it | Optimal, 177 it | 17 → 18 |  |
| qcqp1000-2c | Optimal, 163 it | Optimal, 163 it | 97 → 144 |  |
| **qcqp1000-2nc** | Optimal, 289 it | Optimal, 184 it | 48 → 33 |  |
| **qcqp1500-1c** | CPU limit, 125 it | CPU limit, 100 it | 300 → 300 |  |
| **qcqp1500-1nc** | CPU limit, 362 it | Optimal, 128 it | 300 → 48 |  |
| qcqp500-3c | Optimal, 330 it | Optimal, 330 it | 21 → 14 |  |
| qcqp500-3nc | Optimal, 115 it | Optimal, 115 it | 6 → 4 |  |
| qcqp750-2c | Optimal, 75 it | Optimal, 75 it | 16 → 10 |  |
| qcqp750-2nc | Optimal, 98 it | Optimal, 98 it | 20 → 13 |  |
| qssp180 | Optimal, 27 it | Optimal, 27 it | 71 → 51 |  |
| robot_1600 | Optimal, 35 it | Optimal, 35 it | 1 → 1 |  |
| **robot_a** | Optimal, 145 it | Optimal, 193 it | 53 → 45 | scale audit: first solve 189 → 166 it; re-solve promoted → promoted |
| **robot_b** | Optimal, 271 it | Optimal, 180 it | 159 → 71 | scale audit: first solve 271 → 284 it; re-solve kept → promoted |
| **robot_c** | Optimal, 222 it | Optimal, 116 it | 62 → 63 | scale audit: first solve 222 → 215 it; re-solve kept → promoted |
| rocket_12800 | Optimal, 30 it | Optimal, 30 it | 4 → 4 |  |
| steering_12800 | Optimal, 23 it | Optimal, 23 it | 5 → 4 |  |
| **svanberg** | Optimal, 34 it | Optimal, 27 it | 12 → 8 |  |
| **total** | 44 solved, 4939 it | 44 solved, 6593 it | 2675 → 2634 | 14 lines moved |

Every moved line, explained:

- **`qcqp1500-1nc`**: the crawl. CPU limit at 362 iterations (`μ`
  stalled at 1.8e-6) → Optimal in 128, 47 s. 10 005 inequality rows,
  factor 0.1.
- **`clnlbeam`** is *not* a trajectory change: it has **0 inequality
  rows**, so the candidate's `δ_s` rule is upstream's there, and the two
  iteration tables are identical through every one of the candidate's
  2296 rows. Main's line is 20 iterations because its first solve ran 2465
  iterations (156 s) and the scale audit's re-solve, 20 iterations, was
  promoted; under the load the candidate run had reached row 2295 when
  the 300 s wall clock ran out. Re-run alone at the default:
  Optimal, 2465 + 20 iterations, the re-solve promoted, 198 s, iteration
  table identical to main's through all 2487 rows.
- **`WM_CFy`** and **`qcqp1500-1c`** time out on both sides and the
  counts are how far each got in 300 s of wall clock under different
  load. `qcqp1500-1c` (10 008 rows) is identical through all 101 candidate
  rows — its inertia correction never fires in that stretch, so the budget
  has nothing to act on, and it is still the open half of gh#1098 (dual
  infeasibility 1e6–1e7 at the limit, both sides). `WM_CFy` (4536 rows,
  factor 0.22) diverges at row 1 at the 1e-6 level and at row 85, main's
  last, the candidate is at `inf_du` 6.8e-1 against main's 3.8e-1 and
  `inf_pr` 5.4e-5 against 7.0e-6 — neither is close, and the candidate
  went on to row 171 at 6.0e-2.
- **`NARX_CFy`** (4536 rows, factor 0.22): a basin change. The tables
  diverge at row 2 (the first with `δ_x > 0`), and the candidate
  converges in 305 iterations to f = 8.5257e-3 against main's 378
  iterations to 8.5897e-3 — lower, so not a loss, but a different local
  minimum of a nonconvex model.
- **`arki0003`** 448 → 335, **`qcqp1000-2nc`** 289 → 184, **`nql180`**
  37 → 33, **`optmass`** 32 → 31, **`svanberg`** 34 → 27: same answer,
  fewer iterations. **`arki0009`** 328 → 332: the one line that is
  longer by the budget alone (+4).
- **`robot_a`**, **`robot_b`**, **`robot_c`** (52 013 rows, factor 0.02)
  all run the gh#983 scale audit, which re-solves from the returned point
  and reports the re-solve's count when it is promoted. Counting the
  work actually done (first solve + re-solve): `robot_a` 189+145 = 334 →
  166+193 = 359, the first solve shorter and the promoted re-solve
  longer; `robot_b` 271+813 = 1084 (re-solve rejected) → 284+180 = 464
  (promoted), wall 159 → 71 s; `robot_c` 222+145 = 367 → 215+116 = 331.
  `robot_b`'s objective moves from 2.3330990 to 2.3330959 because the
  promoted re-solve is the tighter point (unscaled KKT error 2.5e-5 →
  1.5e-6 on `robot_a`, the audit's own criterion).

Net: solved 44 → 44 in the suite run (45 with `clnlbeam` read as the
timeout artifact it is); over the 43 lines that end in a verdict on
both sides, 4347 → 3899 iterations (−10 %), with `arki0009` (+4) and
`robot_a`'s promoted re-solve (+48, −23 on the first solve) the only
lines longer. Setting `perturb_slack_max_rows=0` on the candidate binary
reproduces main's line on each of those: `robot_a` 189+145 iterations, `arki0009` 328, `NARX_CFy` 378 at f = 8.5897e-3, each with an iteration table identical to main's row for row (338, 323 and 379 rows).

#### Robustness of the cap: 500, 1000, 2000 on the 19 models above 1000 rows

The cap is a constant, so the question is whether the result depends on
its exact value. The 19 Mittelmann models above 1000 rows, run at
`perturb_slack_max_rows` = 500, 1000 and 2000 (candidate binary; main is
the cap-off column). Iteration counts are exact; the four timeout cells
per row are wall-clock timeouts under load and show the row reached.

| model | m_d | main | cap 500 | cap 1000 (default) | cap 2000 |
|---|---|---|---|---|---|
| NARX_CFy | 4536 | 378 | 331 (f=0.008594632) | 305 (f=0.008525684) | 413 (f=0.008657439) |
| WM_CFy | 4536 | timeout @85 | timeout @132 | timeout @165 | timeout @190 |
| arki0003 | 1691 | 448 | 469 | 335 | 448 |
| arki0009 | 3304 | 328 | 328 | 332 | 348 |
| camshape_6400 | 19197 | 88 | 88 | 88 | 88 |
| corkscrw | 5000 | 19 | 19 | 19 | 19 |
| ex8_2_2 | 1943 | 67 | 67 | 67 | 67 |
| ex8_2_3 | 3155 | 71 | 71 | 71 | 71 |
| nql180 | 32400 | 37 | 35 | 33 | 43 |
| optmass | 10001 | 32 | 31 | 31 | 31 |
| qcqp1000-2c | 5007 | 163 | timeout @127 | 163 | 163 |
| qcqp1000-2nc | 5005 | 289 | 134 | 184 | 278 |
| qcqp1500-1c | 10008 | timeout @125 | timeout @95 | timeout @126 | timeout @114 |
| qcqp1500-1nc | 10005 | timeout @362 | 187 | 128 | 158 |
| qssp180 | 65341 | 27 | 27 | 27 | 27 |
| robot_a | 52013 | 145 | 184 | 193 | 127 |
| robot_b | 52013 | 271 | 173 | 180 | 179 |
| robot_c | 52013 | 222 | 128 | 116 | 121 |
| svanberg | 50000 | 34 | 25 | 27 | 27 |
| **sum over the 15 models with a verdict in every column** | | 2456 | 2110 | 2008 | 2287 |

What it says:

- **Every cap solves `qcqp1500-1nc`** (187 / 128 / 158 iterations at
  factor 0.05 / 0.1 / 0.2) and every cap leaves the nine models whose
  inertia correction never fires — `camshape_6400`, `corkscrw`,
  `ex8_2_2`, `ex8_2_3`, `qssp180`, `qcqp1000-2c`, `qcqp1500-1c`, and
  the `optmass` / `svanberg` pair within a few iterations — where main
  has them. The `qcqp1000-2c` miss at 500 is a load timeout on a
  trajectory identical to main's through all 128 rows it reached (no
  regularization ever fires on that model).
- **It is not a sharp optimum, and it is not monotone.** 1000 is the
  best of the three on the 15-model sum (2008 against 2110 and 2287,
  main 2456), but `arki0003` is 469 at 500 and 448 at 2000 against 335
  at 1000, `qcqp1000-2nc` is best at 500 (134), and `robot_a` best at
  2000 (127). A model's count moves by tens of percent in either
  direction between neighbouring caps; the sum improves at every cap.
- **`NARX_CFy` is a different local minimum at every cap**: f = 8.590e-3
  (main), 8.595e-3 (500), 8.526e-3 (1000), 8.657e-3 (2000). That the
  default lands on the lowest of the four is luck, not design — the model
  has several minima within 2 % of each other and any change to the
  regularized directions picks a different one. Reported so that nobody
  reads the 1000-column objective as a property of the rule.
- `WM_CFy` and `qcqp1500-1c` time out at every cap.

So 1000 is defensible as "above every fixture in the corpus (351) by a
margin, and the best of the three values measured on the suite", and no
stronger claim is made for it. A user with a many-inequality model that
got slower can set `perturb_slack_max_rows=0` (upstream exactly) or a
different cap; the knob and the measured response are both documented.
