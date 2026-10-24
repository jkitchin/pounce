# gh#1042 — recovering the 0.13.0 accepted trajectory costs

Plan for the agent loop on branch `claude/pounce-regressions-recovery-tz8uh2`.
One work item per iteration. This file is also the status board: update the
**Status** column of the inventory and the per-item log at the end of every
iteration, in the same commit as the code.

Written against `0a299151` (main after 0.13.0, 2026-10-22), on **x86_64**
(4 CPUs, AVX2+FMA, FERAL 0.21.0). Every number marked *measured* below was
taken on that machine with a release `pounce` built from `0a299151`
(`p-head`) and from tag `v0.12.0` = `a31dab9c` (`p-base`), built in a
separate worktree with its own `CARGO_TARGET_DIR`. Numbers marked *issue*
are copied from gh#1042 / PR #1038 and were **not** reproduced here (arm64
numbers cannot be).

## 1. Summary and inventory

gh#1042 is the tracking record CLAUDE.md requires for the trajectory costs
0.13.0 shipped: the FERAL 0.20 → 0.21.0 bump (#1038, commits `97bc0f15`,
`5fc8c797`, `2599c2be`), the gh#1000 convex-routing change (#1002,
`f6a4e4c2`) and the MLSL `gamma` default change (#993, `91b6dcda`). An issue
comment (id 6344966966, moved from jkitchin/feral#224) adds two benchmark
diagnoses: the qcqp1000-2c slowdown is pounce reusing the multiplier-init
matrix's symbolic ordering, and brainpc0/1's slowdown is 811 restoration
iterations after a now-*correct* inertia report.

### What the measurements here add or correct

| claim in the issue | measured here (x86_64) | verdict |
|---|---|---|
| exact `pooling_rt2stp` 162 → 492 (x86) | `p-base` 162 it / q=4 / 3 resto calls; `p-head` 492 it / q=3 / **24 resto calls, 106 resto it**, **40 iterations with lg(rg) ≥ 19 and 171 with lg(rg) ≥ 10** | confirmed; the ladder-at-cap diagnosis is consistent with the log but still unmeasured at the handler level |
| exact `dallass` 41 → 123 (x86) | `p-base` 41 it / q=6 / **2 resto calls**; `p-head` 123 it / **q=19 / 12 resto calls (53 resto it)** | confirmed; the cost is 10 extra non-finite-Hessian restoration entries, not the strike exemption |
| exact `deb7` Acceptable/180 **arm64 only**, x86 improves 143 → 114 | `p-base` 143, `p-head` 114, both `SolveSucceeded` | confirmed as an x86 improvement; the arm64 regression is unverifiable here |
| lbfgs `dallass` Optimal/224 → ErrorInStep/308 **arm64 only** | `p-base` **ErrorInStep/213**, `p-head` ErrorInStep/146 (ladder tot 292) | **discrepancy**: on x86 the lbfgs leg fails on 0.12.0 too, so there is no x86 regression to recover and the "arm64 only" is about the *verdict* on arm64, not the leg being healthy elsewhere |
| lbfgs `pooling_rt2stp` worse optimum **arm64 only**; x86 failing → solved | `p-base` ErrorInStep/716, `p-head` SolveSucceeded/82 at −3273.95 | confirmed x86 gain |
| `scaled_feasible_a` nlp 22/19 → cvx-qp 200/200 | `p-base` nlp 22 / 19; `p-head` cvx-qp 200 / 200, obj 0, final kkt 5e-19 | confirmed, **mechanism found** (item R1) |
| minima `circle_parabola_t0p5` mlsl 2 → 1 | `p-base` (gamma 2) found 2 in 11 solves; `p-head` (gamma 0.5) found 1 in 9 solves, `converged` = stagnation at `patience 8` | confirmed; gamma grid measured (item R4) |
| not in the issue | lbfgs `deb7`: `p-base` ErrorInStep/1010 → `p-head` RestorationFailed/1206 (ladder tot 1738). Both fail; the sweep diff line exists but is not a lost solve | recorded, out of scope |

### Inventory

| id | fixture / benchmark (leg) | v0.12.0 → 0.13.0 | suspected cause | recoverable? | Status |
|---|---|---|---|---|---|
| **R1** | `scaled_feasible_a` (exact and lbfgs) | nlp 22 / 19 it → cvx-qp **200 / 200** it, right answer | #1002 `f6a4e4c2`: HSDE endgame certifies the gh#293 equilibrated retry, whose loop never meets `pres < tol` (plateau 5.0e-6 → 5.7e-7) while `mu` runs to 1e-121; `Optimal` is stamped only by the post-loop `true_kkt_error` upgrade | **yes** — a stall-gated in-loop certificate, or a primal-residual floor that reads the stalled `pres` as the regularization floor it is | **not recoverable this way** (2026-10-22: hypothesis refuted — `m_eq = 0`, the plateau is a rounding floor, and the iterate is not optimal until it 165; see log + §5). Alternate routes: **R1b–R1e** below |
| **R1b** | `scaled_feasible_a` (sibling of R1) | retry it 20 → 120: `mu` 5.0e-19 → 3.2e-117 while `true_comp` 2.6e5 → 9.8e3 | HSDE centering: the `σμ` target is driven on an internal `s` that has decoupled from `h − Gx̂`, so `mu` collapses ~8×/iteration while the true complementarity crawls linearly | **candidate** — floor `σμ` at the true-slack complementarity (in homogenized units); convex-arm-wide trajectory change | todo (`needs benchmarks/qp before merge` at best) |
| **R1c** | `scaled_feasible_a` (sibling of R1) | both HSDE solves run at `τ ≈ 1e-10` to represent `‖x̂‖ = 5e9` from the unit self-dual start | cold start not sized to the data; the direct driver's gh#689 Mehrotra recentering took the same model 175 → 27 | **candidate** — seed the embedding from the recentered cold seed at `τ = 1`; the largest blast radius of the R1 family | todo (`needs benchmarks/qp before merge` at best) |
| **R1d** | `scaled_feasible_a` (sibling of R1) | first, unequilibrated HSDE solve burns 200 it to `true_kkt 0.149` before the gh#293 retry: 400 HSDE it for a reported 200 | no early-failure detection: `τ` creeps linearly 1.6e-9 → 7.6e-8 over it 19–199 with `mu ~1e-37` | **candidate** — hand the retry the budget early; independent value (halves the work) even if R1b/c land, reported count unchanged | todo (`needs benchmarks/qp before merge` at best) |
| **R1e** | `scaled_feasible_a` (sibling of R1) | retry it 165 → 200 is certifiable (`true_kkt 8.6e-9`) but the loop waits on `pres < tol`, unreachable at 4 ulp of `5e9` | stopping test has no exit for an iterate whose only unmet term is pinned at its rounding floor | **partial** — ≤ 17% of this model's count; R1 declined it as insufficient alone; kept as a sibling because it composes with R1b/c and may have corpus-wide value | todo (lowest priority in the family) |
| **R2** | `pooling_rt2stp` (exact, x86) | 162 → **492** it (arm64: 109 → 242 total, *issue*) | FERAL 0.21 trajectory; `2599c2be` removed the latch (904 → 492); the remaining cost is the `δ_x` ladder climbing to `max_hessian_perturbation` with `δ_c` on (*issue*, unmeasured) | **partial** — the "stop the ladder once inertia is guaranteed" follow-up the issue names; must be measured first | **not recoverable this way** (2026-10-22: hypothesis refuted — 0 of 146 `WrongInertia` factors with `δ_c > 0` had a PD (1,1) block; the climb answers a genuinely indefinite `W`, λ_min down to −1.2e19 vs v0.12.0's −2.2e9; trigger found at iteration 4, see log + §5). Alternate routes: **R2b–R2e** below. ~~Family target met on x86 by R2b~~ **Correction (R2e iteration): the target is not met; R2b reverted in C1** — pooling exact is back to 0.13.0's single-attempt 492. R2b's reported 109 is the `mu_strategy_fallback` re-solve; its base solve is **494 it, `Solved_To_Acceptable_Level`, −4336.99 (not an optimum), 22 resto calls**, against 0.13.0's single `Solve_Succeeded` 492 — total 603 iterations / 25 resto calls, and neither the JSON report nor the sweep shows the first attempt (R2e log). Every member is now terminal (§4 rule 6: R2/R2c/R2d/R2e not recoverable, R2b swept), so the family is closed **by status with its target unmet**; R2b needs the owner's (or a correction iteration's) re-measurement before anything is built on it |
| **R2b** | `pooling_rt2stp` (exact, x86; sibling of R2) | 492 it; blanket counterfactual 492 → 109 with 19 collateral lines | one `augsys_improved` flag per matrix is shared by the inertia ladder's too-few-negatives branch and iterative refinement; at iteration 4 the former spends it on a 71-vs-72 count and refinement keeps a 6.4e16 step | **candidate** — one escalation budget per consumer per matrix; must avoid the counterfactual's collateral (lbfgs `circle_parabola_t0p5` 8.155 → 10.428, lbfgs `scale_audit_retry_meyer3` 112 → 152, 15 lines escalating to cap) | **reverted — first solve regressed** (C1, 2026-10-22: with the fallback visible, R2b turned 0.13.0's single `Solve_Succeeded`/492 (24 resto) into a first attempt `Solved_To_Acceptable_Level`/494 at −4336.99 (unscaled dual inf 5.3e6, 22 resto) rescued by `mu_strategy_fallback`'s 109 — total 603 it / 25 resto. No narrowing exists inside the lever: its census population is one matrix, and the `refuted` / `dc0` / `refuted_dc0` variants were byte-identical to the split, so every variant either takes that decision (494 Acceptable) or is the revert (492 Optimal). Code `git revert`ed; its §3 text and §6 log are kept as the record. As first recorded: "swept — needs vanderbei before merge", exact 492 → 109 — that 109 was the fallback's) |
| **R2c** | `pooling_rt2stp` (exact, x86; sibling of R2) | iteration-4 refinement stalls at ratio 1.6e-8 and takes the `S` exit with a dual part of 6.4e16 | `compute_residual_ratio` caps its normalizer at `1e6·‖rhs‖`, so the `S` test (ratio < `residual_ratio_singular` = 1e-5) is blind to a step many orders above the RHS | **candidate** — refuse `S` (take `s` → `perturb_for_singular`) when `‖res‖∞ > κ·‖rhs‖∞`; the narrowest of the family | **not recoverable this way (measured)** (2026-10-22: a step dwarfing the RHS is the *ordinary* `S` exit, not the tail — 135 of the corpus's 188 `S` exits are above `κ = 1e6`, 90 above `1e12`, and the pooling iteration-4 exit (2.2e14) ranks 62nd; keeping the rule under the item's 10% stop needs `κ ≥ 4.7e20`. Counterfactual at `κ = 1e6` / `1e12`: pooling's base solve is lost (`ErrorInStep` 795 / 173, rescued by the `mu_strategy=adaptive` rung at 140), exact `circle_parabola_t1p5` Infeasible → MaxIter 3000, and the `s` path never reinstates `δ_c` (six uncertified `Singular`s climb `δ_x` to 8.7 at `δ_c = 0`); see log). Next: **R2b** |
| **R2d** | `pooling_rt2stp` iteration 4 (sibling of R2) | FERAL 0.21 reports 71 negatives, 72 required; FERAL 0.20 reported 73 on v0.12.0's matrix | either an inertia-count defect in FERAL (crates.io dependency `feral = 0.21.0`, upstream jkitchin/feral — not vendored) or a genuinely ambiguous near-singular matrix | **measure first** — exact count vs both FERAL counts on the dumped matrix; fix upstream, or work around in `pounce-feral` (classify as noise → `Singular`, no escalation spent) | **not recoverable here — not a count defect (measured)** (2026-10-22: the 0.13.0 iteration-4 matrix is singular to working precision — two scaled eigenvalues 1.6e-18 / 1.1e-15, next 7.7e-4 — and its exact count is 71, which FERAL 0.21, 0.20 and 0.18 all report on the dump; v0.12.0's iteration-4 matrix is a *different*, nonsingular one (min \|λ\| 2.4e-9, exact 73, all three FERALs 73); the noise rule is gated off by `min_piv` 5.95e-14 vs floor 3.33e-14; three pounce-feral workarounds measured, none admissible; see log). Next: **R2c** |
| **R2e** | `pooling_rt2stp` iteration 4 (sibling of R2) | both versions accept a `δ_c`-free factor with min \|eig\| 2e-17 after the gh#592 walk-back | the walk-back accepts any factor FERAL calls `Success`; a numerically singular one is the source of the 6.4e16 step | **candidate** — refuse the withdrawal when the factor's `min_pivot_magnitude` is under the inertia trust floor; keeps `δ_c` instead of repairing the step downstream | **not recoverable this way (measured)** (2026-10-22 on `3e186634`. *Re-baselined after R2b* first: the regression still reproduces — base solve 494 it `Solved_To_Acceptable_Level`, 22 resto, rescued by `mu_strategy_fallback` (109) — so not `dropped`. Refuted by the item's own clause: the iteration-4 walk-back's `Success` has `min_piv` **3.98e-14, above** the `150·ε` = 3.33e-14 floor. The rule's corpus population, both legs, is **8** first-`δ_c`-free `Success` factors, all on exact pooling (iterations 5–39 + restoration); firing on them takes the base solve 494 Acceptable → **3000 `MaximumIterationsExceeded`** and the reported verdict `SolveSucceeded` → MaxIter. The widened form (certify sub-floor uncertified `Singular`s too) is gh#981's rejected "reinstate on any `Singular`": base 141 Acceptable at −3261.36, reported 491, gh#884's reproducer dual 1.3e-10 → 7.0e-5; see log) |
| **R3** | `dallass` (exact, x86) | 41 → **123** it (arm64 41 → 65, *issue*) | FERAL 0.21 path lands on `x = 0` of `|x|²·|x|^0.85` terms 12 times instead of 2; each is a non-finite Hessian → restoration entry; `5fc8c797` only stops the third from ending the solve | **partial** — (A) the Hessian is an evaluator artefact (`0·∞` from the product rule; the true second derivative of `|x|^2.85` at 0 is 0), fixable in `pounce-nl`; (B) R2 may move it; re-baseline after R2 | **done** (2026-10-22, option A: `.nl` reader merges `|e|^p·|e|^q`; exact 123 → **28** it, 12 → **0** restoration calls; lbfgs ErrorInStep/146 → SolveSucceeded/301; see log) |
| **R4** | minima `circle_parabola_t0p5` mlsl | found 2 of 2 → **1** | #993 `91b6dcda`: `MLSL_GAMMA` 2 → 0.5; at 0.5 the extra launches re-find the first minimum and the stagnation counter (`patience 8`) stops the search | **partial** — no single gamma wins at seed 0 (table in R4); needs a 10-seed measurement; the structural stagnation rule is now its own item, **R4b** | **not recoverable this way (measured)** (2026-10-24, 10 seeds × 8 gammas, x86_64: no gamma meets step 3(a). circle_parabola finds 2 on **5/10** seeds at every gamma 0.4–0.75 (4/10 at 0.6) and **7/10** at 1.0 and at 0.12.0's 2.0 — the seed-0 "2 → 1" is 7/10 → 5/10 across seeds, mean found 1.7 → 1.5 — and the gammas that reach 7/10 cost camel 5.8 → 4.0 / 2.0 and haverly 2.0 → 1.5 / 1.0 (wavy 6 → 1 at 2.0). Gamma stays 0.5; nothing changed. Table: `dev-notes/issue-1042-r4-mlsl-gamma-grid.tsv`; see log. The structural lever is R4b's — refuted 2026-10-24 by its trace count; the open route is **R4c**. Family closed 2026-10-24 by **R4c** (done): circle 5/10 → 8/10 seeds across 10 seeds, above 0.12.0's 7/10; the seed-0 line stays found=1 (re-find-limited)) |
| **R4b** | minima `circle_parabola_t0p5` mlsl (sibling of R4) | at gamma 0.5 the search ends `converged` after 9 solves: 8 consecutive launches re-find minimum 1 and trip `patience 8` with 21 of 30 solves unused | the stagnation counter treats an MLSL launch that converges to an archived minimum as a failed independent draw; it is basin information | **candidate** — grow that minimum's exclusion radius and do not count the re-find as a full strike; gamma stays 0.5 | **not recoverable this way (measured)** (2026-10-24, direct trace count, 10 seeds, gamma 0.5, x86_64: on the five circle seeds that stop at found 1 the eight `patience` strikes are **6, 1, 1, 1, 3 re-finds** — the rest are launches ending `Converged to a point of local infeasibility`, every one of them at the same point (0, −1) — so re-finds dominate the stop on **1 of 5** failing seeds (seed 0 only). Where re-finds *do* dominate is where the search is already complete: haverly's stop is 8 of 8 re-finds of its 2 minima on all 10 seeds, 28 of 30 solves at unbounded patience. Replaying the traces, half-weight re-finds take circle 5 → 7/10 but haverly mean solves 10.2 → **18.2 (+78%)**, outside the item's 20%. Nothing changed; table `dev-notes/issue-1042-r4b-mlsl-strike-trace.tsv`; see log. The data's route is **R4c**) |
| **R4c** | minima `circle_parabola_t0p5` mlsl (sibling of R4/R4b) | at gamma 0.5, 5/10 seeds stop at found 1 / 9 solves; of their 40 strikes, **28 are launches ending locally infeasible, all at one point (0, −1)** (5 first hits + 23 repeats) and 12 re-finds; the second minimum is reached at solve 10–17 on all five when patience does not stop the search (budget 30) | `reject` counts a launch that converges to an already-seen locally infeasible point as a full strike: the infeasible attractor drains the lower half of the box, and on a constrained model eight hits of the *same* infeasible point are not eight independent failures | **candidate** — record locally infeasible landing points (not as minima); a repeat hit is not a full strike. Trace replay at gamma 0.5: weight 0 → circle **8/10**, mean solves 10.0 → 19.5; weight 0.5 → 6/10, 13.7; camel / haverly / wavy / wavy_1e12 / the saddle and `kkt_rej` lines have **zero** infeasible launches, so a counting-only rule leaves them byte-identical | **done** (2026-10-24, x86_64: a failed MLSL launch whose landing point violates the base constraints by more than 1e-4 at a point an earlier launch already ended at costs no `patience` strike, once a minimum is archived. Classifier (driver-side, from the point; no API change) flags exactly the 229 locally infeasible launches of 1848 solves and none of the 19 feasible failures. 10 seeds: circle **5/10 → 8/10** find both, mean solves 10.0 → 19.5, reproducing the replay per seed; camel / haverly / wavy / wavy_1e12 rows byte-identical; seed-0 sweep line stays `found=1` (9 → 10 solves), as predicted; `sweep-fixtures.sh` empty. See log) |
| **B1** | benchmark `qcqp1000-2c` (mittelmann) | 99 it / 28 s → 163 it / 112 s (*comment*) | pounce reuses the symbolic ordering FERAL chose for the least-squares multiplier-init matrix (W = 0, D = I) for every later KKT; `invalidate_symbolic_cache()` after the init factor halves factor time (*comment*) | **yes** for the code change; **measurement blocked** here (no benchmark corpus on this machine); a branch `perf/feral-symbolic-after-init` is said to be in progress — check `origin` first | **skipped — owner's work in progress** (2026-10-22: the branch is on `origin` at `c9182194` and is draft PR **#1046**, open; the owner's follow-up comment 6346290866 records it as model-dependent — wins qcqp1000-2c 106 → 76 s wall, NARX_CFy, clnlbeam, pooling; losses qcqp1000-2nc, qcqp1500-1c 206 → 586 it, nql180, dallasl, x86 deb7 — and moves the principled fix into FERAL, jkitchin/feral#227. Not duplicated here; see §6 B1 log) |
| B2 | benchmark `brainpc0/1` (vanderbei) | 58 it / 1.1 s → 1073 + 52 it / 75 s; 185 + 24 → 2117 + 24 (*comment*) | after a now-correct inertia at one iterate, pounce's restoration takes 811 of 1073 iterations (*comment*) | **unknown** — investigate only after B1 lands (B1 changes the trajectory) and only with the corpus | blocked (and owner's work in progress: open PR **#1047**, branch `fix/brainpc-restoration-path`, plus the restoration outer-progress item in comment 6350601458; B1 log) |
| B3 | benchmark `qcqp1500-1c` | Optimal → `Maximum_CpuTime_Exceeded` at 300 s (*issue*) | presumably B1 (same family); **unverified** | follows B1 | blocked |
| B4 | benchmark `cresc132` Optimal → Infeasible (Ipopt agrees); `ssebnln` Optimal → Acceptable | (*issue*) | FERAL 0.21 trajectory | **no / out of scope** (§5) | n/a |
| X1 | exact `deb7` arm64 Optimal/147 → Acceptable/180 | (*issue*) | FERAL 0.21 NEON trajectory | **not recoverable here** (x86 improves 143 → 114) (§5) | n/a |
| X2 | lbfgs `dallass` arm64 Optimal/224 → ErrorInStep/308 | (*issue*); x86 fails on both versions | FERAL 0.21 trajectory | **not recoverable here**; the x86 lbfgs failure predates 0.13.0 (§5) | n/a |
| X3 | lbfgs `pooling_rt2stp` arm64 −4391.83 → −3273.95 | (*issue*); x86 failing → solved | bistable nonconvex model | **no** — both are certified optima (§5) | n/a |
| F1 | `issue_592` pin `< 500` sits 8 above x86's 492 | — | R2 | resolved by whichever of R2b–R2e lands; **do not raise the pin** | **margin 8 on a single attempt** (C1, after R2b's revert: pooling exact is one `Solve_Succeeded` attempt at 492, `mu=-`, so `< 500` is met by the first solve, not a fallback. On the R2b tip it had been met by the fallback's 109 with the first attempt at 494 — margin 6, Acceptable). The `< 500` pin is untouched; it gets no fallback assertion here because pooling's arm64 trajectory (issue: "109 → 242 total") is unmeasured and may legitimately take a retry there |
| **C1** | the measurement instrument (correction iteration, after R2e) | — | a promoted `mu_strategy_fallback` (pounce#138/#748) re-solve replaces `iteration_count`, `restoration_calls` and the status in the JSON report, and the sweep had no column for it, so a first attempt that got *worse* read as a speed-up (R2b's 492 → 109) | **yes** — report the fallback, re-audit every item with it | **done** (2026-10-22: `statistics.mu_strategy_fallback` in the JSON report — first attempt's status / iterations / restoration calls / objective, re-solve's strategy / status / iterations, `promoted` — and a `mu=` sweep column, report-only (patched sweeps byte-identical to unpatched on four binaries). Re-audit at v0.12.0 / 0.13.0 / pre-R2b / tip: the only fallback line the branch moved is R2b's pooling exact; `csfi2` exact is a pre-existing promotion on all four; R3 stands. R2b reverted on the numbers; R3's pin asserts no fallback; see log) |
| **M1** | re-measurement pass after every landed R2x item | — | R2b/c/d/e each move the NLP arm's step for every exact-Hessian solve; R3 landed *before* them (dallass 28), so the original "R3 after R2" re-baseline is inverted | rule, not a code item: see §3 M1 and §4 "Sibling items" | standing (runs inside each R2x iteration) |

## 2. Ordering

Original order (first pass, 2026-10-22): R1 → R2 → R3 → B1 → R4 → B2/B3. R1
and R2 were refuted as planned (docs-only commits `463eeeae`, `015a8fe4`); R3
landed (`250857b0`, dallass 123 → 28). The second pass below replaces the
refuted items with the alternate routes their measurements suggested (§3,
"Second-pass items"), ordered cheapest and least risky first. The two
sibling families share one defect each; §4 "Sibling items" says what happens
to the rest of a family once one member recovers the target.

1. **R2d** FERAL 71-vs-72 at pooling iteration 4 — a *measurement* first
   (dump one matrix, three counts), no pounce code unless the workaround
   route is taken; its answer tells R2b/R2c whether the count they are
   working around is a defect or rounding. **Closed 2026-10-22: not a
   count defect** — the matrix is singular to working precision (nullity
   2), so for R2b/R2c the 71 is rounding, and the escalation the
   too-few-negatives branch spends on it is spent on a matrix no pivot
   rung can repair (§6 R2d log).
2. **R2c** refuse refinement's `S` exit on a step that dwarfs the RHS — one
   comparison in one branch of `pd_full_space_solver.rs`; narrowest of the
   R2 family. **Closed 2026-10-22: not recoverable this way** — a step
   6–37 orders above the RHS is what most `S` exits in the corpus look
   like (135 of 188 above `1e6`), so the rule is a blanket change; and on
   pooling the `s` path it routes to climbs `δ_x` with `δ_c` withdrawn
   instead of reinstating `δ_c`, which is R2e's lever (§6 R2c log).
3. **R2b** separate escalation budgets for the inertia branch and
   refinement — the shape the R2 log named first; wider than R2c (the
   counterfactual's 19-line collateral is the thing to avoid). **Landed
   2026-10-22, `swept — needs vanderbei before merge`:** pooling exact
   492 → 109; the split's corpus population is one matrix and the sweep
   moves one line (§6 R2b log). **Reverted 2026-10-22 in C1:** the 109 was
   `mu_strategy_fallback`'s re-solve; the first attempt went Optimal/492 →
   Acceptable/494, total 603 (§6 C1 log).
4. **R2e** walk-back refuses a numerically singular `δ_c`-free factor —
   upstream of both R2b and R2c in the iteration-4 chain, but touches
   gh#592/gh#981 territory, so last in the family. **Closed 2026-10-22:
   not recoverable this way** — the iteration-4 factor it aimed at is
   above the floor (3.98e-14 vs 3.33e-14); its only corpus population is
   8 later factors on pooling, and firing there sends the base solve to
   the 3000 cap. The same iteration found that **R2b's 109 is the
   `mu_strategy_fallback` re-solve** (base solve 494 Acceptable), so the
   R2 family closes by status with its target unmet (§6 R2e log).
   — **M1** runs inside each R2x iteration: re-measure the inventory on the
   new tip before the next sibling starts (§3 M1).
   — **C1** (correction iteration, inserted after R2e, 2026-10-22): make
   `mu_strategy_fallback` re-solves visible to the JSON report and the
   sweep, re-audit R3, R2b and the M1 table with first-solve numbers, and
   decide R2b on them. Runs before B1, because B1's re-baselines would
   otherwise be taken with the same blind spot (§6 C1 log).
5. **B1** symbolic-ordering reuse — after the R2 family, not between its
   members: it changes *which factorization* FERAL performs and through
   pivoting can move trajectories, which would confound the sibling
   re-baselines. Code change implementable and unit-testable without the
   corpus; the benchmark claim can only be *verified* with it.
   **Skipped 2026-10-22:** the owner's `perf/feral-symbolic-after-init` is on
   `origin` as draft PR #1046, and the owner has since routed the
   principled fix to FERAL (jkitchin/feral#227) — not duplicated (§6 B1
   log). B2's restoration path is likewise in the owner's open PR #1047.
6. **R4** MLSL gamma — independent, own sweep (`sweep-minima.sh`); its
   10-seed table is also R4b's measure-first step, so R4 runs first.
   **Closed 2026-10-24: not recoverable this way** — no gamma recovers
   circle_parabola on more than 7 of 10 seeds (0.12.0's own 2.0 included),
   and every value that reaches 7 loses camel / haverly minima; the table
   is `dev-notes/issue-1042-r4-mlsl-gamma-grid.tsv`, regenerated by the
   `.sh` beside it (§6 R4 log).
7. **R4b** structural MLSL-aware stagnation rule — sibling of R4; drop or
   keep per §4 once R4's decision is in. **Closed 2026-10-24: not
   recoverable this way** — a direct trace count shows re-finds dominate
   the stop on 1 of the 5 failing circle seeds; the rest are repeat hits
   of one locally infeasible point, and the re-find-dominated stops are
   haverly's, where the search is already complete (§6 R4b log,
   `dev-notes/issue-1042-r4b-mlsl-strike-trace.tsv`).
   — **R4c** (inserted 2026-10-24 from R4b's trace count): a launch that
   lands on an already-seen locally infeasible point is not a full
   `patience` strike. Counting-only, so by construction it cannot move an
   unconstrained or saddle line; runs before the R1 family because, like
   R4/R4b, it can reach `done` on this machine.
   **Done 2026-10-24:** circle 5/10 → 8/10 seeds at gamma 0.5 (mean solves
   10.0 → 19.5), every other grid row and every other sweep line unchanged;
   MLSL only — multistart was measured with the rule and gained nothing
   (§6 R4c log).
8. **R1d** cut the first, unequilibrated HSDE solve short — convex arm, but
   by construction it only touches solves that end `IterationLimit`
   anyway; reported counts unchanged, so the fixture sweep must diff
   **empty**. Independent value even if R1b/c land.
9. **R1e** the post-165 tail — stopping-test exit for a certifiable iterate
   whose only unmet term is pinned at its rounding floor; lowest value in
   the family (≤ 17% of this model's count), kept because it composes with
   R1b/c.
10. **R1b** HSDE centering safeguard — convex-arm-wide trajectory change;
    the measure-first step says on how many corpus lines it would bite.
11. **R1c** data-sized HSDE cold start — the largest blast radius on the
    branch (every HSDE solve starts elsewhere); last of the R1 family, and
    dropped if R1b already meets R1's ≤ 60.
12. **B2 / B3** — only if `POUNCE_BENCH_DATA` resolves to a corpus
    (`. scripts/bench_data_root.sh && bench_data_root`); otherwise record as
    blocked and stop.

Items 8–11 cannot be marked `done` on this machine: each changes the convex
arm's trajectory and `benchmarks/qp` is not available (`POUNCE_BENCH_DATA`
unset, no `~/projects/pounce-bench-data`). They may be implemented, tested
and swept, and are then logged `swept — needs benchmarks/qp before merge`
(§4). Items 2–4 (and 1 if it changes `pounce-feral`) change the NLP arm's
step for every exact-Hessian solve and are likewise logged `swept — needs
vanderbei before merge`; B1 the same with `mittelmann` added. R4/R4b/R4c and
R2d-as-measurement are the only second-pass items that can reach `done`
here.

Items that interact: R2b/c/d/e (one defect, four levers — §4 sibling
rule), R1b/c/d/e (one defect, four levers), R4 ↔ R4b ↔ R4c, R2x → R3 (R3 is
already in; an R2x item that moves `dallass` off 28 is a reportable line,
not a re-baseline), B1 → B2/B3 (re-baseline). The R1 family touches
neither arm the R2 family touches, R4/R4b/R4c touch only `pounce-minima`, but
every item still diffs the full fixture sweep against the **branch tip
before the item** (`p-prev`), so that each commit's own blast radius is
accounted for separately.

## 3. Work items

Common to every item: build `p-prev` from the branch tip before you start
(`cargo build --release -p pounce-cli && cp target/release/pounce
/tmp/p-prev`), keep `p-base` (v0.12.0) around for the regressed fixtures'
"before" numbers, and sweep **both legs** with `scripts/sweep-fixtures.sh`.
Fixture paths are `crates/pounce-cli/tests/fixtures/<name>.nl`.

### R1 — `scaled_feasible_a` runs the convex IPM to `max_iter` before reporting Optimal

**Goal.** The convex arm certifies `scaled_feasible_a` `Optimal` in a number
of iterations comparable to a healthy HSDE solve (≤ 60 on both legs; the
pre-#1002 NLP route took 22 / 19), without changing any verdict.

**Files / symbols.**
- `crates/pounce-convex/src/hsde.rs`: the main loop (`for it in 0..opts.max_iter`,
  ~line 631), the stopping test (`abs_ok`, `rel_in_floor`, `comp_ok`,
  `converged`, ~lines 898–1000), the relative-candidate stash
  (`rel_candidate`, `restore_candidate`, ~1010–1050), the adaptive static
  regularization `base_reg` (~1094–1110), the post-loop `true_kkt_error`
  upgrade (~1620–1640).
- `crates/pounce-convex/src/ipm.rs`: `solve_qp_ipm_core` (the gh#293
  equilibrated retry, ~527–700), `equilibrated_retry_budget` (~807),
  `optimum_is_genuine` (~1104), `verify_or_repair_optimum`.
- `crates/pounce-cli/src/main.rs` ~2941 and ~4902: the gh#535 convex → NLP
  decline rule, whose comment still says this model "needs 3596 iterations".
- Tests to read first: `crates/pounce-cli/tests/issue_689_direct_driver_scaled_feasible.rs`
  (`ITER_BUDGET = 60` on the `qp_hsde=no` route; its header explains the
  `obj_constant` normalizer), `declared_optimum_sentinel.rs` (~163–220,
  routing at default and at `bound_relax_factor=1e-8`),
  `issue_1000_*`, `crates/pounce-convex/tests/issue984_stopping_units.rs`,
  `issue_689_direct_cold_start_scale.rs`, `issue414_cost_normalized_false_optimal.rs`,
  `illconditioned_huge_scale.rs`; `dev-notes/issue-689-direct-driver-cold-start.md`.

**Measured mechanism (x86, `p-head`, `--json-detail full`).** With
`qp_equilibrate=no` the convex arm does *not* certify and the model falls to
the NLP arm (22 it), so the returned 200-iteration run is the **gh#293
equilibrated retry** (full `max_iter` budget because `P ≠ 0`). Its trace:
`inf_pr` falls to 3.9e-5 at iteration 13, then sits at **5.000e-6** from
iteration 20 to ~125 while `inf_du` drifts 5e-5 → 5e-9 and `mu` collapses
from 1e-19 to 1e-121; around 125–150 the iterate is thrown away (`inf_pr`
back to 4.8e9, `alpha 0`), recovers by 155, and then `inf_pr` oscillates
5.7e-7 ↔ 2.8e-6 with `inf_du` ~1e-14 until the cap. `pres < opts.tol` (1e-8)
is never met, so neither `abs_ok` nor `rel_in_floor` (which also requires
`pres <= cap_p`) is, and `converged` never fires. After the loop the
un-homogenized point's `true_kkt_error` is 5e-19, so the verdict is upgraded
to `Optimal`. The 5e-6 plateau is the static regularization floor
`δ·‖ŷ‖` on the equality block in the equilibrated metric (the commit message
of `f6a4e4c2` records that lowering the equality-block floor `delta_c` was
tried and dropped because it cost QFORPLAN 33 → 139).

**Correction (R1 iteration, 2026-10-22).** The last sentence above is wrong:
`scaled_feasible_a` has no equality rows (`m_eq = 0`), every bit of `pres` is
the inequality block, and `5.0e-6` is the rounding floor of `Gx + s − hτ` at
`‖x̂‖ = 5e9` (≈ 4 ulp, as `ipm.rs`'s own gh#689 comment says). Through the
plateau the iterate is not optimal either (`true_kkt_error` ≥ 23 until it
150). The hypothesis below is refuted; see the R1 item log and §5. The
alternate routes that measurement suggested are items **R1b–R1e** under
"Second-pass items" below.

**Hypothesis.** The loop lacks a stop for an iterate that is optimal in the
caller's coordinates but whose *homogeneous, equilibrated* primal residual is
floored by regularization. The post-loop certificate already encodes the
right test; it is just never consulted while iterating.

**Steps.**
1. Reproduce: `pounce scaled_feasible_a.nl out.sol --json-output a.json
   --json-detail full` → `statistics.iteration_count == 200`,
   `solution.engine == "cvx-qp"`. Confirm the plateau in `iterations[]`.
   Also run `hessian_approximation=limited-memory` (same numbers expected —
   the convex arm ignores it).
2. Instrument (temporarily) which of `pres`, `dres`, `gap`, `comp_max`,
   `true_comp` fails at iterations 20–120 and what `base_reg`/`delta_c`
   floor is in force. Record the numbers in the item log.
3. Implement the narrowest fix that the measurement supports. Preferred
   shape, in the loop after `converged` is computed: when the iterate has
   **stalled** — `pres` within a factor of 2 of its value `k` iterations ago
   (k = 5 is a starting point), `mu` below `REL_FLOOR_KAPPA·ε·tol` or similar
   so no polishing is happening, and `dres`/`gap`/`comp_ok` already satisfied
   — compute `true_kkt_error(prob, cone, x/τ, y/τ, z/τ, 1.0)` and stop
   `Optimal` if it is `< opts.tol`. This honours the existing comment in
   `ipm.rs` ("deliberately runs *after* the loop … breaking early on the
   certificate would hand back a less-polished iterate"): a stalled loop is
   not polishing anything. Alternative shape, if the measurement says the
   plateau is exactly the regularization floor: let `cap_p` include the
   regularization floor `δ·‖ŷ‖`, as the comment on `QpOptions::reg` already
   describes it. Do **not** change `equilibrated_retry_budget` to hide the
   cost, and do not change the gh#535 decline rule.
4. Update the stale "3596 iterations" comments in `main.rs` only if the
   number you measure is different; cite the measurement.
5. Add a regression test in `crates/pounce-convex/tests/` (or extend
   `issue_689_direct_driver_scaled_feasible.rs`) that pins the default
   route: engine `cvx-qp`, `Optimal`, `iteration_count <= 60`, objective
   `|f| <= 1e-3`. Mutation-check it: with the fix reverted it must fail on
   the iteration count.

**Acceptance.** `scaled_feasible_a`: `cvx-qp`, `SolveSucceeded`, obj
`|f| ≤ 1e-3`, `iteration_count ≤ 60` on **both** legs (measured on this
machine, 200 → target). `scaled_feasible_b` unchanged or better.
`pounce verify` on the `.sol` still reports 0 violation.

**Validation.**
- `scripts/sweep-fixtures.sh` both legs, `p-prev` vs new. Only
  `scaled_feasible_a` lines (and possibly other convex lines that *improve*)
  may move; any convex line whose iteration count **rises** or whose
  engine/status changes is a stop-and-explain. Recount the engine split in
  the item log.
- `cargo test -p pounce-convex -p pounce-cli` (the `issue_689_*`,
  `declared_optimum_sentinel`, `issue_760_*`, `issue_1000_*`,
  `issue984_stopping_units`, `issue414_*`, `illconditioned_huge_scale`
  files are the guards that must not regress).
- If `bench_data_root` resolves: `benchmarks/qp` A/B total iterations
  (0.13.0 release: 137 problems optimal, total 2998 at #1002); a rise of more
  than a few percent, or any status flip, is a stop.
- `python/tests/test_qp*.py` if the Python extension builds here
  (`maturin develop` in `python/`); otherwise record "not run".

**Must not regress.** The gh#414 false-optimum guards (`optimum_is_genuine`,
`verify_or_repair_optimum`), the gh#984 stash rule ("every stash satisfies
`comp_ok`"), gh#1000 I.D/I.8 numbers (battery QP optimal at tol 1e-6/1e-8;
blend LP 14–23 iterations across scales — both pinned in `issue984_*` /
`issue_1000_*` tests), the gh#689 `ITER_BUDGET` on the direct driver.

### R2 — `pooling_rt2stp` exact: stop the `δ_x` ladder once the inertia is guaranteed

**Goal.** Bring x86 `pooling_rt2stp` (exact) from 492 iterations back toward
0.12.0's 162 without reintroducing the latch `2599c2be` removed, and restore
the margin under the `issue_592` `< 500` pin by lowering the count, not the
pin.

**Files / symbols.**
- `crates/pounce-common/src/pd_perturbation.rs`: `perturb_for_wrong_inertia`,
  `get_deltas_for_wrong_inertia` (the ladder: `delta_xs_init`,
  `delta_xs_first_inc_fact`, `delta_xs_inc_fact`, `delta_xs_max`, the
  overflow branch at ~687 with the `2599c2be` reset), `maybe_withdraw_delta_c`
  (gh#592 walk-back), `perturb_for_singular_with` (gh#981 reinstatement),
  the unit tests at the bottom (`an_overflowed_ladder_does_not_poison_the_next_system`,
  the gh#592 walk-back tests).
- `crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs` ~1640–1760: the
  retry loop that reads `ESymSolverStatus` and calls the handler
  (`Singular` → `perturb_for_singular_with`; `WrongInertia` with too few
  negatives → `escalate_aug_quality` then `perturb_for_singular`; other
  `WrongInertia`/`Singular` → `perturb_for_wrong_inertia`).
- `crates/pounce-algorithm/src/kkt/aug_system_solver.rs`: `AugSysCoeffs`
  (`w: Option<&dyn SymMatrix>`, `w_factor`, `d_x`, `delta_x`, `d_s`,
  `delta_s`, `delta_c`, `delta_d`) — everything a Gershgorin bound on the
  (1,1) block needs is in this struct.
- `crates/pounce-algorithm/src/kkt/std_aug_system_solver.rs` ~200–320:
  downcasts `W` to `SymTMatrix` (exact Hessian) or `DiagMatrix`; the L-BFGS
  arm uses `LowRankUpdateSymMatrix` through
  `low_rank_aug_system_solver.rs`, for which a row-max is not implemented
  (the issue says so) — the lbfgs leg keeps the current behaviour.
- `crates/pounce-feral/src/lib.rs` ~1350–1440: where `WrongInertia` /
  `Singular` are decided (`inertia_trust_floor`, `sub_floor_count_is_noise`,
  gh#1003).
- Guards: `crates/pounce-cli/tests/issue_592_delta_c_walkback.rs`
  (`< 500` pin, both optima accepted, walk-back inert), `issue_250_dual_guard_never_worse.rs`
  (`pooling_rt2stp_solves_at_default_settings`), `crates/pounce-wasm/src/lib.rs`
  ~942–965 (pooling shim), `issue857_*` (quality-escalation counts on
  `deb7`/`circle_parabola_t0p5`), `issue_884_biactive_dual_divergence.rs`,
  `pd_perturbation` unit tests; `dev-notes/issue-592-restart-non-idempotence.md`,
  `dev-notes/issue-981-delta-c-walkback.md`, `dev-notes/issue-540-eigena2-inertia-noise.md`.
- Debug aid: `POUNCE_DBG_PERT=1` makes the handler emit `[PERT]` lines
  through the sink's `debug` (tracing at debug level; check how the CLI
  enables it, e.g. `RUST_LOG`/`--log-level`).

**Hypothesis (the issue's, unmeasured).** With `δ_c > 0` the KKT matrix is
quasi-definite as soon as the (1,1) block `W + Σ_x + δ_x I` (and the slack
block `Σ_s + δ_s I`) is positive definite, and then its inertia is exactly
`(n + m_s, m_c + m_d, 0)` by Sylvester. When FERAL's count disagrees on such
a matrix the count is rounding, and the ladder should accept the factor
instead of climbing (×8 per rung) to `max_hessian_perturbation`. The log is
consistent: 40 of 492 iterations carry `lg(rg) ≥ 19`, 171 carry
`lg(rg) ≥ 10`, and the over-damped steps feed 24 restoration entries
(0.12.0: 3).

**Steps.**
1. Reproduce: `pounce pooling_rt2stp.nl o.sol --json-output p.json` →
   492 it, `-3273.955`, no second opinion; count `lg(rg) ≥ 19` lines.
2. **Measure before coding.** With `POUNCE_DBG_PERT=1` (and the tracing
   level that shows it) record per ladder: `δ_c` on/off, rungs climbed,
   whether it overflowed, and — with a temporary Gershgorin probe on
   `AugSysCoeffs` — at which rung the (1,1) block became diagonally dominant.
   The item is only worth implementing if a material share of the 171
   high-`δ_x` iterations had `δ_c > 0` and a provably PD block at a rung well
   below where the ladder stopped. Write the histogram into the item log.
   If the hypothesis is false, record that, mark R2 "not recoverable this
   way", and propose the alternative the measurement suggests (e.g. the
   `delta_c` reinstatement/withdrawal interplay of gh#592/gh#981) as a
   follow-up issue instead of forcing a change.
3. Implement, if supported: in `pd_full_space_solver.rs`, when the status is
   `WrongInertia` (not `Singular`), `d.delta_c > 0` **or** `m_c == 0`, and a
   helper (new `AugSystemSolver` method, default `false`; implemented in
   `StdAugSystemSolver` for `SymTMatrix`/`DiagMatrix` W only) proves
   `W·w_factor + D_x + δ_x I` strictly diagonally dominant with positive
   diagonal and `D_s + δ_s > 0`, treat the factor as `Success` and record an
   info-string letter (e.g. `g`) so the iteration table shows it. Note the
   gh#1003 reasoning in `pounce-feral`: a *surplus* negative on a clean factor
   stays `WrongInertia` — this item accepts exactly that case when the matrix
   proves the count wrong. Do not touch the `Singular` arm (that is the
   gh#981 reinstatement's territory) and do not alter the overflow reset.
4. Unit test in `pounce-algorithm` (a 2×2 or 3×3 KKT whose (1,1) block is
   diagonally dominant with `δ_c > 0` and a fake linear solver reporting a
   wrong count) and a CLI pin: `pooling_rt2stp` exact `SolveSucceeded`, a
   known optimum, `iteration_count` below a bound set from the measurement
   (leave the `< 500` pin where it is).

**Acceptance.** `pooling_rt2stp` exact (x86): `SolveSucceeded`, objective one
of −3273.955 / −4391.826, `iteration_count ≤ 300` (0.12.0: 162; anything
≤ 300 restores the `issue_592` margin; report the actual number). Fewer
restoration calls than 24. No new `max_hessian_perturbation` overflows on
systems where the block was provably PD.

**Validation.**
- `scripts/sweep-fixtures.sh` both legs, `p-prev` vs new, **and** the same
  lines against `p-base` for the regressed fixtures. Every moved line is
  explained: an exact-leg line whose `q=` or `it=` moves is expected on
  nonconvex models; any status loss, any objective change outside a known
  bistable pair, any `nlp → square-lm` move, and any lbfgs line moving at
  all (the helper must be a no-op there) are stops.
- `cargo test -p pounce-common -p pounce-algorithm -p pounce-cli -p pounce-wasm`.
  Guards named above; in particular `the_walkback_is_inert_on_this_fixture_since_gh693`
  (`< 300` with the walk-back off) and `issue857_*`.
- Python `python/tests/test_mehrotra_feasible_gh986.py` if the extension
  builds; otherwise record.
- If the corpus is available: `benchmarks/vanderbei` status A/B (0.13.0
  release: 707 of 733) — `steenbrd/e/f/g`, `eigena2/b2` and `dallas*` are
  the models the inertia path has historically moved.

**Must not regress.** `an_overflowed_ladder_does_not_poison_the_next_system`
(the `2599c2be` fix); gh#592's walk-back tests; gh#981's reinstatement test
(`issue981_cstr_dup_row`); gh#540/gh#1003 `pounce-feral` unit tests; the
`issue_250` hang guard; `deb7` must stay `SolveSucceeded` on the exact leg.

**Refuted (R2 iteration, 2026-10-22).** Measured with a temporary dense
probe on every ladder factorization (log below): no `WrongInertia` factor
with `δ_c > 0` has a positive-definite (1,1) block, so the planned
"accept when Sylvester guarantees the inertia" arm would never fire on this
fixture. The ladder stops within ~1 rung of the (1,1) PD threshold
(median `δ_x / −λ_min(W+Σ_x)` = 5.7); it climbs high because `W` itself is
indefinite by 1e13–1e19 after iteration 5, which v0.12.0 never is. The
trigger is an iteration-4 step from a `δ_c`-free, numerically singular
factor that iterative refinement accepts because the matrix's single
quality escalation was already spent by the ladder's too-few-negatives
branch. Follow-up in §5; the four levers that chain names are items
**R2b–R2e** under "Second-pass items" below.

### R3 — `dallass` exact: the `|x|²·|x|^0.85` kink

**Goal.** x86 `dallass` exact back toward 41 iterations (0.12.0) with the
strike exemption kept.

**Files / symbols.**
- `crates/pounce-cli/tests/fixtures/dallass.nl` lines ~300–340: the objective
  terms are `o2 (o5 (o15 vK) n2) (o5 (o15 vK) n0.85)`, i.e. `|x|² · |x|^0.85`.
  The product rule gives `u''v + 2u'v' + uv''` with `v' = 0.85|x|^-0.15` and
  `v'' = -0.1275|x|^-1.15`, so at `x = 0` two terms are `0·∞ = NaN` while the
  true second derivative of `|x|^2.85`, `2.85·1.85·|x|^0.85`, is **0**.
- `crates/pounce-nl/src/nl_tape.rs`: `TapeOp::Pow` / `TapeOp::Abs` forward,
  gradient and Hessian arms (~373, 511, 694, 951, 1130, 1470, 1838); the
  `hessian_pow_*_matches_fd` tests (~5131–5150).
- `crates/pounce-algorithm/src/ipopt_alg.rs` ~3175–3195 (`resto_entry_nonfinite_w`)
  and ~4120 (the strike exemption); `crates/pounce-cli/tests/nonfinite_hessian_restoration_strike.rs`.

**Measured.** `p-base`: 41 it, 2 restoration calls (14 resto it), q=6.
`p-head`: 123 it, **12 restoration calls (53 resto it)**, q=19; the `R`
entries sit at iterations 31, 39, 45, 53, … — the trajectory keeps returning
a flow to exactly 0. Hessian evaluations = iterations + 1 on both, so the
cost is the restoration entries and the post-restoration re-convergence, not
extra evaluations.

**Hypotheses.** (A) The NaN is an evaluator artefact: merging the two powers
of the same `Abs` operand (`|x|^a · |x|^b → |x|^(a+b)`, valid because the
base is non-negative), or evaluating the product-rule Hessian of
`pow(abs(x), p)` nodes so that an exact-zero factor annihilates an infinite
co-factor **only when the merged exponent proves the limit is finite**, gives
a finite Hessian at the kink and removes the restoration entries entirely.
(B) R2 changes the path and the kink is met less often — re-measure first.

**Steps.**
1. After R2 lands, re-run `dallass` both legs; record. If exact ≤ ~60 with
   ≤ 3 restoration calls, mark R3 "recovered by R2" and stop.
2. Otherwise prototype (A) as a tape-level peephole at `.nl` load time
   (`nl_reader.rs` / `nl_tape.rs`): a product whose two factors are
   `Pow(Abs(v), c1)` and `Pow(Abs(v), c2)` with constant exponents becomes
   `Pow(Abs(v), c1 + c2)`. Values are unchanged up to rounding; first
   derivatives at `x ≠ 0` unchanged; at `x = 0` the Hessian becomes the
   mathematically correct finite value. Add a unit test that the Hessian of
   `|x|²·|x|^0.85` at 0 is finite and matches a finite difference at
   `x = ±1e-3`, and that a model without the pattern builds an identical
   tape.
3. Measure `dallass` again: expect **no** non-finite-Hessian restoration
   entries (`resto_entry_nonfinite_w` never set; add a counter to the
   statistics or read the `R` lines).

**Acceptance.** `dallass` exact (x86): `SolveSucceeded` at −32393.2257,
`iteration_count ≤ 60`, restoration calls ≤ 3. The lbfgs leg is not an
acceptance criterion (it fails on 0.12.0 too on x86) but must not get
worse (`ErrorInStepComputation`/146 today).

**Validation.** Full fixture sweep both legs (the peephole changes the
Hessian of any model with that pattern — grep the fixture corpus for
`o15` under `o5` under `o2` and list the affected fixtures beforehand);
`cargo test -p pounce-nl -p pounce-cli`; `nonfinite_hessian_restoration_strike.rs`
must still pass — note its header says the exemption is what makes it pass,
so if (A) removes the kink entirely that test no longer exercises the
exemption: keep the exemption, and give the test a second model that still
reaches a non-finite Hessian (e.g. `sqrt(x)` at `x = 0` with
`x ≥ 0`), mutation-checked with the exemption reverted.

**Must not regress.** The `5fc8c797` exemption and its test; `dallasl`,
`dallasm` on the corpus (same term structure) if available.

### B1 — pounce reuses the multiplier-init matrix's symbolic ordering

**Goal.** The first IPM KKT factorization gets its own symbolic analysis
instead of inheriting the ordering FERAL's value-dependent `LdltCompress`
preprocessing chose for the least-squares multiplier-init system
(W = 0, D_x = D_s = I). Comment 6344966966: replaying qcqp1000-2c, 98.98 s
→ 56.39 s factor time, delays 2.95M → 1.61M, identical inertia on all 165
systems.

**Before starting:** `git fetch origin && git branch -r | grep -i symbolic`
and search open PRs for `perf/feral-symbolic-after-init`. If that branch or
PR exists, do not duplicate it — record its state in the item log and skip
B1 (it is the owner's in-progress work). It was **not** on `origin` when
this plan was written; it **was** by the B1 iteration (2026-10-22,
`c9182194`, draft PR #1046), so B1 is skipped — see the §6 B1 log.

**Files / symbols.**
- `crates/pounce-feral/src/lib.rs`: the per-pattern symbolic cache ("pattern-
  fingerprint cache likewise reuses the symbolic factorization", header
  ~line 17), `FeralSolverInterface::factor` (~1307), the invalidation block
  (~1604); FERAL 0.21's `Solver::invalidate_symbolic_cache` (registry source
  `feral-0.21.0/src/numeric/solver.rs` ~2812).
- `crates/pounce-algorithm/src/eq_mult/least_square.rs` (~49–200): the
  multiplier-init solve goes through the same `AugSystemSolver`
  (`StdAugSystemSolver` "pins its triplet structure" there).
- Other first-factorization sites the comment lists: restoration's first
  factorization (`AugRestoSystemSolver`), the second-opinion ladder and
  gh#983 audit re-solves, warm starts.
- `crates/pounce-linalg` `SparseSymLinearSolverInterface` trait, for where a
  "next factor is a different regime, re-analyse" hint belongs.

**Steps.**
1. Trace which `FeralSolverInterface` instance the init solve and the first
   IPM KKT share, and confirm (with a counter or `RUST_LOG`) that the second
   factor does not re-run symbolic analysis. Record it.
2. Add an interface method (e.g. `invalidate_symbolic_cache`, default no-op)
   and call it once after the multiplier-init factor (and at the other
   regime boundaries only if step 1 shows they share a cache). Keep the
   per-pattern cache otherwise: re-analysing every factor is the slowdown
   the cache exists to prevent.
3. Unit test in `pounce-feral`: two factors with the same pattern but
   different values re-use the analysis by default and re-analyse after the
   invalidation; an `pounce-algorithm` test that the init → first-KKT
   transition triggers exactly one re-analysis.

**Acceptance.** Without the corpus: the tests above, and the fixture sweep
explained. With the corpus: qcqp1000-2c wall time materially down from 112 s
toward the comment's replay (56 s on FERAL 0.21; the "99 it" of 0.12.0 is a
different trajectory and not a target), `LinearSystemFactorization` time
reported in the JSON report, inertia unchanged.

**Validation.** Full fixture sweep both legs — a different ordering can
change pivots, so lines may move; a line moving only in `q=` or `it=` on a
nonconvex model is a trajectory change to explain, any status loss is a
stop. `cargo test -p pounce-feral -p pounce-algorithm -p pounce-cli`.
`benchmarks/mittelmann` and `vanderbei` A/B if the corpus is available.

**Must not regress.** `pounce-feral`'s factor-stats tests (#955), gh#857
quality-escalation accounting (escalations never step back down), the
`issue_983_scale_audit_retry_budget` test.

### R4 — MLSL default `gamma` and `circle_parabola_t0p5`

**Goal.** `circle_parabola_t0p5` under `--minima mlsl --seed 0 --n-minima 6
--max-solves 30 --patience 8` finds both minima again without losing the
gains #993 bought on camel / wavy / wavy_1e12.

**Files / symbols.** `crates/pounce-minima/src/search.rs`: `MLSL_GAMMA`
(~line 75), `run_mlsl` (~647–690: `radius = gamma·√n·(ln N/N)^(1/n)`,
`better_sample_within`, `near_any` against the archive), `consider` /
`reject` / `patience` (~530–605); `options.rs` (`gamma: Option<f64>`);
`crates/pounce-cli/src/cli.rs` ~328 (help text says 0.5);
`python/pounce/_minima.py`; `docs/src/find-minima-cli.md` ~84,
`docs/src/find-minima.md` ~174. Guards:
`crates/pounce-cli/tests/issue989_mlsl_cli.rs`,
`python/tests/test_minima.py::test_mlsl_default_gamma_launches_solves_on_a_2d_box`
and `::test_mlsl_sample_cap_status_names_the_cap`,
`python/tests/test_minima_cli_parity.py`; `scripts/sweep-minima.sh`.

**Measured (x86, `p-head`, seed 0, the sweep's budget `--n-minima 6
--max-solves 30 --patience 8`; "found" = minima, status in brackets):**

| gamma | camel | circle_parabola_t0p5 | haverly_t3_brf0 | wavy | wavy_1e12 |
|---|---|---|---|---|---|
| 0.5 (default) | 4 (converged, 12 solves) | **1** (converged, 9) | 2 (10) | 6 (target) | 6 (target) |
| 0.75 | 5 (sample cap, 5) | 2 (17) | 2 (10) | 6 | 6 |
| 1.0 | 4 (sample cap, 4) | 2 (15) | **1** (sample cap, 1) | 6 | 6 |
| 1.5 | 2 (sample cap) | 2 (12) | 1 (sample cap) | **1** (sample cap) | 1 |
| 2.0 (0.12.0) | 2 (sample cap) | 2 (11) | 1 (sample cap) | 1 (sample cap) | 1 |

(`nonconvex_qcqp`, `nonconvex_two_escapes`, `saddle_axis_first`,
`wavy_accept_anything` read `found=0` at every gamma, as the sweep expects.)
`p-base` at its default 2 reproduces found=2 / 11 solves; `p-base
--gamma 0.5` reproduces found=1 / 9. So the loss is the gamma value, not the
gh#966 extraction, and at gamma 0.5 the search ends by **stagnation** (8
consecutive rejected solves re-finding minimum 1), not by budget. No single
value dominates at one seed: 0.75 is the only row that loses nothing in this
table, and it is one seed.

**Steps.**
1. Reproduce the row above with `p-prev`.
2. Measure over **seeds 0–9** the five fixtures above at gamma ∈ {0.5, 0.6,
   0.75, 1.0} (CLI loop; ~25 runs per cell are sub-second). Score per fixture:
   mean minima found and mean solves. This is the measurement #993 did not
   record for the constrained fixtures.
3. Decide from the measurement: (a) a new default (only if one value is
   never worse on mean found than 0.5 on camel/wavy and recovers
   circle_parabola on most seeds); or (b) a structural change that does not
   move gamma — e.g. in `run_mlsl`, a solve launched from a sample that
   converges to an already-archived minimum extends the archive's
   `near_any` radius for that minimum (so the clustering filter learns the
   basin) instead of counting as a stagnation strike. (b) is the better
   shape — the stagnation rule was written for multistart, where every solve
   is an independent draw, and MLSL's launches are not independent — but it
   is a search change and needs the same 10-seed table. Pick one; do not do
   both in one iteration.
4. Update the help text, the two docs pages and the Python docstring if the
   default moves; update `issue989_mlsl_cli.rs` and the Python gamma tests
   only to the extent their *stated* contracts (all six camel minima at
   `--max-solves 60 --patience 15`; `sample_cap_reached` at gamma 100) need
   it — those contracts must still hold.

**Acceptance.** `scripts/sweep-minima.sh` `p-prev` vs new: the
`circle_parabola_t0p5 mlsl` line reads `found=2`; no other `mlsl` line reads
fewer minima than before; non-mlsl lines (the other five methods) **diff
empty**. The 10-seed table is in the item log and in the commit message.

**Validation.** `scripts/sweep-minima.sh` (54 lines) `p-prev` vs new;
`scripts/sweep-fixtures.sh` must diff **empty** (the single-solve path is
untouched); `cargo test -p pounce-minima -p pounce-cli` (`issue989_mlsl_cli`,
`minima_*`); `python/tests/test_minima.py` and `test_minima_cli_parity.py`
if the extension builds — the parity test is the one that catches a CLI /
Python default drifting apart, so if the default moves and Python cannot be
run here, say so loudly in the log and keep the Python default in
`python/pounce/_minima.py` in lockstep by inspection.

**Must not regress.** `mlsl_default_gamma_finds_all_six_camel_minima`,
`the_sample_cap_is_named_as_such`, the gh#966 parity contract, the
`sweep-minima.sh` `found=0` saddle lines and the `kkt_rej` line.

### B2 / B3 — brainpc0/1 restoration path, qcqp1500-1c (corpus required)

Only with `bench_data_root` resolving, and only after B1 has landed and the
vanderbei/mittelmann A/B has been re-taken on top of it. Goal for B2: find
why restoration spends 811 of 1073 iterations on brainpc1 after a *correct*
inertia at iteration 41 (the comment's diagnosis), with the restoration
cycle detectors in `ipopt_alg.rs` (`resto_no_outer_progress_count`,
`resto_near_feasible_count`, the gh#1000 I.20 inner budget inheriting
`max_iter`) as the first suspects. No acceptance numbers can be set without
the corpus; if it is absent, record "blocked: no benchmark corpus on this
machine" and stop. B3 is presumed to follow B1 (same family); verify, do not
assume.

### Second-pass items (added 2026-10-22 after R1/R2 were refuted)

Every number below is from the R1 and R2 item logs (§6) unless marked
otherwise; nothing new was measured when these items were written. Each item
restates where it sits in its family and what the §4 sibling rule does to it
once another member lands. The common preamble of §3 (build `p-prev`, keep
`p-base`, sweep both legs) applies unchanged.

The iteration-4 chain on `pooling_rt2stp` exact, which all of R2b–R2e act
on, in order:

1. the ladder's first factor reports **71 negatives, 72 required** (FERAL
   0.20 on v0.12.0's matrix: 73) → `pd_full_space_solver.rs` ~1687
   too-few-negatives branch → `escalate_aug_quality` succeeds
   (`augsys_improved = true`, info `q`) **(R2d, R2b)**;
2. `Singular`, then `WrongInertia(73)` ×3 with `δ_c` on → the gh#592
   walk-back withdraws `δ_c` → `Success` at `δ_x = 1e-4`, `δ_c = 0`, min
   |eig| 2.0e-17 **(R2e)**;
3. the solve's dual part is 6.4e16; refinement ratios 9.1e-6, 2.2e-8,
   1.2e-8, 1.6e-8 → stall; `augsys_improved` already set → `pretend_singular`
   → ratio 1.6e-8 < `residual_ratio_singular` 1e-5 → **`S` exit, step kept**
   **(R2b, R2c)**;
4. λ_min(W+Σ_x) at iteration 5 is −4.1e13 (v0.12.0, whose refinement
   escalated and refactored to a 3.0e6 dual part: −7.4e2), and the rest of
   the run answers an indefinite `W` with high `δ_x`, 24 restoration calls
   and 492 iterations.

Blanket counterfactual (R2 log): letting refinement escalate regardless of
the flag gives 492 → **109** (−4391.826, 3 restoration calls) but moves 19
sweep lines: escalations to the cap (q = 12 / 24) on 15, lbfgs
`circle_parabola_t0p5` 8.155 → 10.428 with a different second-opinion rung,
lbfgs `scale_audit_retry_meyer3` 112 → 152. Those three facts are the
must-not-regress guards for every R2x item.

### R2d — FERAL 0.21 reports 71 of 72 negatives at `pooling_rt2stp` iteration 4

**Goal.** Decide, with numbers, whether the 71-vs-72 count that starts the
chain is an inertia-count defect in FERAL 0.21 or a legitimately ambiguous
count on a near-singular matrix — and route the fix accordingly: upstream
if a defect, a `pounce-feral` classification change if ambiguity, or
nothing here (R2b/R2c absorb it) if neither is clean.

**Where FERAL lives.** Not vendored. `Cargo.toml` pins the crates.io
release (`feral = "0.21.0"`; the comment at ~line 93 says why a git rev is
rejected: `scripts/check-release-consistency.sh` refuses git dependencies
for the crates.io publish). Source: `~/.cargo/registry/src/index.crates.io-*/feral-0.21.0/`
(0.18.0 is also in the registry; 0.20.0, the version v0.12.0 built against,
is reachable through the `p-base` worktree's lockfile). Upstream:
`https://github.com/jkitchin/feral` (the issue's benchmark comment was moved
from jkitchin/feral#224). A FERAL fix therefore cannot ship from this branch
alone — it needs a FERAL release and a version bump, which is itself a
trajectory change (the 0.20 → 0.21 bump is what gh#1042 is about).

**Files / symbols.**
- FERAL: `src/numeric/solver.rs` ~2546–2600 (`num_negative_eigenvalues`,
  `inertia()`, `min_diagonal`, `min_pivot_magnitude`, `max_pivot_magnitude`);
  `src/dense/factor.rs` ~555–600 — the **band-pivot sign fallback**
  (2026-05-17): a pivot in `(zero_tol, null_pivot_tol]` is counted *by sign*,
  not as zero, with `needs_refinement = true` (the rationale cites
  `FBRAIN3LS_0839`, a trailing pivot of +2.47e-16 that MUMPS/SSIDS count
  positive); the 2×2 pivot classification (`pivot_inertia`, `zero_tol_2x2`,
  the `fallback_2x2_*` / `scalar_2x2_delay_*` counters ~135–165);
  `src/inertia.rs`. FERAL's own `dev/research/f01-rankdef-underreporting.md`
  and `near-singularity-signal.md` are referenced from those comments.
- pounce: `crates/pounce-feral/src/lib.rs` ~1337–1420 — the status decision:
  `inertia()` → `(neg, zero)`; `zero > 0` → `Singular` (certified); count
  mismatch → gh#540/gh#1003 noise rule **only if** `min_piv <
  inertia_trust_floor` (`n·ε` unless `feral_inertia_pivot_floor` pins it;
  for the 46-row pooling KKT ≈ 1.0e-14) and `sub_floor_count_is_noise(got,
  expected, flagged, below, below_100x)` → `Singular`, else `WrongInertia`;
  then the `singular_pivot_floor` (default **1e-20**, ~712) near-singularity
  flag. Tests ~2102 `check_neg_evals_mismatch_returns_wrong_inertia`, ~2370
  `well_conditioned_inertia_mismatch_still_reports_wrong_inertia`, ~2358
  `sub_floor_count_is_noise_branches`.
- The R2 probe (dense eigenvalues of the assembled KKT in
  `StdAugSystemSolver::solve`) is the measurement tool; it is not committed,
  re-create it from the R2 log's description.

**Hypotheses (exclusive).** (a) The unperturbed iteration-4 matrix has at
least one eigenvalue within `κ·ε·‖K‖` of zero, so 71/72/73 are all within
the count's honest uncertainty; FERAL is not wrong, pounce's noise rule
should have classified the count as untrustworthy (→ `Singular`, which does
**not** spend the escalation) and did not, because `min_piv ≥ floor` or
`sub_floor_count_is_noise` returned false. (b) The exact count is 72, the
smallest |λ| is far above rounding, and FERAL 0.21's count is wrong by one —
a sign-fallback band pivot or a 2×2 classification; a FERAL defect.

**Reproduce.** `p-prev` exact 492 it. `RUST_LOG=pounce::linsol=debug` (or
the CLI's log-level flag — check how the sink enables tracing) and capture
the iteration-4 `"inertia mismatch"` line: `got_neg`, `expected`, `min_piv`,
`floor`. Confirm it is the first factor of the iteration-4 ladder (`δ_x`,
`δ_c` both from the previous system's state) and that the info string
carries `q`.

**Measure first (the whole item, if (a)).** Dump that matrix (triplets +
`δ`s) from the probe, then record: exact inertia by dense symmetric
eigendecomposition (faer, as in R2); the number of eigenvalues with
`|λ| < 46·ε·‖K‖∞` and with `|λ| < 1e-10·‖K‖∞`; FERAL 0.21's count on the
dump at its default pivot threshold and at 0.1 / 0.5, Identity and
InfNorm scaling; FERAL 0.20's count on the same dump (`p-base` worktree);
`min_pivot_magnitude` / `max_pivot_magnitude` from both; `last_factor_stats`
(`n_tiny`, `n_threshold_fail_accepts`). Three numbers go in the log: exact
count, 0.21 count, 0.20 count, plus the smallest |λ| relative to `‖K‖`.

**Steps.**
1. If (a): the fix is in `pounce-feral`'s classification. Find which guard
   stopped the noise rule (`min_piv` above `n·ε`? `below`/`below_100x`
   shape?) and whether widening it is justified by the eigenvalue evidence.
   The alternative at the consumer — do not spend the escalation on a
   too-few count whose factor has a sub-floor pivot — is R2b's territory;
   if that is the cleaner change, fold R2d into R2b and record it.
   Mutation-check any rule change with the gh#540 eigena2 dumps and the
   gh#1003 tests, which pin the opposite verdicts.
2. If (b): file upstream (jkitchin/feral) with the dumped matrix in a format
   FERAL's `io` module reads, the three counts, and the pivot sequence; in
   this file record the issue number and that the pounce-side mitigation is
   R2b (or R2c, whichever lands). Do not patch the registry copy and do not
   add a git dependency.
3. If neither — the matrix is ambiguous *and* the noise rule is correctly
   silent — record "not a count defect" and move to R2c.

**Acceptance.** The question answered with the three counts and the
smallest-|λ| number in the item log. If route (a) changes code:
`pooling_rt2stp` exact ≤ 300 it, restoration calls < 24, objective one of
−3273.955 / −4391.826, and R2's `issue_592` margin restored.

**Validation (only if code changes).** `cargo test -p pounce-feral -p
pounce-algorithm -p pounce-cli`; full fixture sweep both legs vs `p-prev`,
every moved line explained, with the three counterfactual guards above
checked by name; status `swept — needs vanderbei before merge` (the gh#540 /
gh#1003 models `eigena2`, `eigenb2`, `steenbr*` are the ones a
classification change has historically moved).

**Must not regress.** gh#540 (`eigena2` 64/58/62-for-55 is noise) and
gh#1003 (a surplus negative on a clean factor stays `WrongInertia`) unit
tests; `issue981_cstr_dup_row`; `the_walkback_is_inert_on_this_fixture_since_gh693`.

**Stop.** Two iterations; or the measurement lands in case 3 (then R2c is
next, and R2d is `not a defect (measured)`); or the fix needs FERAL's
sign-fallback band rule weakened (that rule exists to agree with
MUMPS/SSIDS on `FBRAIN3LS_0839` — it is FERAL's own correctness contract,
not ours to move).

**Correction (R2d iteration, 2026-10-22).** Three facts above are wrong
and the item log has the measured values: (1) v0.12.0 builds against
**FERAL 0.18.0** (its `Cargo.lock`; 0.19 and 0.20 came in between, the
0.20 → 0.21 bump is `97bc0f15`), so "FERAL 0.20 on v0.12.0's matrix" is
0.18 on a matrix 0.13.0 never factors; (2) the pooling KKT is
**150**-dimensional (`n_x + n_s = 78`, 72 constraint rows), not 46, so the
trust floor is `150·ε = 3.33e-14`; (3) the "73 on v0.12.0" and "71 on
0.13.0" are counts of **different matrices** — the two runs factor a
bit-identical KKT only twice (the first two factors), and diverge at the
first back-solve. Hypothesis (b) is refuted; (a) holds for the matrix
but its consequence ("widen the noise rule") does not survive
measurement. Verdict: case 3, `not a count defect (measured)`.

### R2c — refuse refinement's `S` exit when the step dwarfs the right-hand side

**Goal.** At iteration 4 the stalled refinement must not keep a step whose
dual part is 6.4e16 by declaring it "not singular"; the `s` path
(`pretend_singular` → `perturb_for_singular` → `δ_c` back on) should take it,
without touching the escalation flag at all.

**Files / symbols.**
- `crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs` ~886–915: the
  `quit_refinement` block. After a stall with `augsys_improved` set,
  `pretend_singular = true`; then `if residual_ratio < residual_ratio_singular
  { pretend_singular = false; "S" } else { "s" }`. `residual_ratio_singular`
  defaults to **1e-5** (~141). `pretend_singular` is consumed by the next
  `solve_once` (~830) as a `Singular` status.
- Same file ~1918–1933, `compute_residual_ratio`: `nrm_resid / (min(nrm_res,
  max_cond·nrm_rhs) + nrm_rhs)` with `max_cond = 1e6` — Ipopt's
  `ComputeResidualRatio`. The `min` **caps the normalizer**: once `‖res‖∞ >
  1e6·‖rhs‖∞` the ratio no longer grows with the step, so a residual that
  is tiny relative to a 6.4e16 step reads as a tiny ratio. That is the
  blind spot this item closes.
- `crates/pounce-common/src/pd_perturbation.rs` `perturb_for_singular`
  (~306) / `perturb_for_singular_with` (~327, gh#981): what the `s` path
  reaches; confirm it *reinstates* `δ_c` after a walk-back rather than
  overflowing.
- Guards: `issue_592_delta_c_walkback.rs`, `issue_250_dual_guard_never_worse.rs`,
  `issue857_*`, `issue_884_biactive_dual_divergence.rs`, `issue981_cstr_dup_row`,
  `pounce-wasm` pooling shim.

**Hypothesis.** A refined step with `‖res‖∞ > κ·‖rhs‖∞` for κ at or above
the normalizer's own `max_cond` is outside the regime the `S` test was
designed for; keeping it on the strength of a capped ratio is what let one
step blow `W` up by 1e13. Refusing `S` there (taking `s` instead) repairs
exactly the iteration-4 decision and nothing else, if the measure-first step
shows such exits are rare.

**Reproduce.** `p-prev` exact 492 it; the R2 probe's iteration-4 refinement
trace (ratios 9.1e-6 → 1.6e-8, then `S`).

**Measure first.** Temporary counter at every `S` and `s` exit across the
full sweep corpus (both legs, `p-prev` build + probe): per exit record
`nrm_rhs`, `nrm_res`, `nrm_resid`, `nrm_res / nrm_rhs`, the iteration, and
whether that fixture-leg ends solved. Histogram `nrm_res / nrm_rhs` at `S`
exits. The item is worth implementing only if the pooling iteration-4 exit
sits far in the tail (≥ 1e6, expected ~1e12) **and** the number of other
`S` exits above the chosen κ is small (tens, not hundreds) — otherwise the
rule is a blanket change in disguise, and the counterfactual already shows
what that costs. Record how many `S` exits the baseline corpus has in all.

**Steps.**
1. Add a field (constant first; promote to an option only if the
   measurement wants tuning) `refine_step_rhs_ratio_max`, default from the
   measurement (candidates: `1e6` = `max_cond`, or one decade above the
   largest healthy exit seen).
2. In the `pretend_singular` branch: `S` only if `residual_ratio <
   residual_ratio_singular` **and** `nrm_res <= refine_step_rhs_ratio_max ·
   nrm_rhs`; otherwise `s`. Append a distinct info letter (e.g. `S!` → pick
   one character; `b` for "big step") so the iteration table shows the new
   exit. Do not touch the escalation flag, `residual_ratio_singular`, or
   `compute_residual_ratio`.
3. Unit test in `pounce-algorithm` with a fake `AugSystemSolver` whose
   solution is `1e12 ×` the RHS with a residual that passes the ratio test:
   the solver must report `pretend_singular` (status `Singular` on the next
   call) instead of `Success`. Mutation: with the new condition removed it
   reports `Success`.
4. CLI pin: `pooling_rt2stp` exact `SolveSucceeded`, objective one of the
   two optima, `iteration_count` ≤ a bound set from the measurement (leave
   `issue_592`'s `< 500` alone; tighten only per §4).

**Acceptance.** `pooling_rt2stp` exact (x86): `SolveSucceeded`, one of
−3273.955 / −4391.826, `iteration_count ≤ 300` (counterfactual 109; report
the number), restoration calls < 24, no new `max_hessian_perturbation`
overflow. The three counterfactual guards hold: lbfgs `circle_parabola_t0p5`
it 77 / q 4 / obj 8.155059653 / second-opinion rung `feral_scaling=mc64`,
lbfgs `scale_audit_retry_meyer3` it 112 / q 5, and no line's `q` rising to
12 or 24 that was not there before.

**Validation.** Full fixture sweep both legs `p-prev` vs new, and the
regressed fixtures vs `p-base`; every moved line explained; any status
loss, any objective change outside a known bistable pair, any `nlp →
square-lm`, is a stop. `cargo test -p pounce-common -p pounce-algorithm -p
pounce-cli -p pounce-wasm`. `python/tests/test_mehrotra_feasible_gh986.py`
if the extension builds. M1 re-measurement. Status at best `swept — needs
vanderbei before merge` (0.13.0 release: 707 of 733; `steenbr*`, `eigen*`,
`dallas*` are the sensitive models).

**Must not regress.** `an_overflowed_ladder_does_not_poison_the_next_system`;
gh#592 walk-back tests and `the_walkback_is_inert_on_this_fixture_since_gh693`
(`< 300` with the walk-back off); `issue981_cstr_dup_row`; `issue_250`;
`issue857_*` escalation accounting; `deb7` exact `SolveSucceeded` (114).

**Stop.** The rule fires on more than ~10% of the baseline corpus's `S`
exits; more than one status loss in the sweep; the `s` path overflows
`δ_c` on pooling instead of reinstating it (then this is R2e's problem);
two iterations.

### R2b — one quality-escalation budget per consumer per matrix

> **Reverted in C1 (2026-10-22).** The result below was scored on the
> reported count, which was `mu_strategy_fallback`'s re-solve; the first
> attempt regressed (Optimal/492 → Acceptable/494, total 603). Kept as the
> record of the lever and its census (§6 R2b and C1 logs).

**Goal.** Refinement keeps its one escalation on a matrix even when the
inertia ladder's too-few-negatives branch has already spent one, so the
iteration-4 refinement refactors (as v0.12.0's did, 6.4e16 → 3.0e6) — and
the counterfactual's collateral does not come with it.

**Files / symbols.**
- `pd_full_space_solver.rs`: `augsys_improved` (~69–71, "Mirrors
  `augsys_improved_`"), `escalate_aug_quality` (~169, the single
  `increase_quality()` call site; bumps `quality_escalations`), the resets
  (~592 `solve`, ~794 cache miss / `consider_new_system`), the two consumers
  (~1690 inertia branch, ~894 refinement). Ipopt's
  `IpPDFullSpaceSolver.cpp` shares one `augsys_improved_` between the same
  two sites — this item departs from upstream on purpose, so say so in the
  doc comment.
- `crates/pounce-feral/src/lib.rs` ~1677 `increase_quality`: rung ladder
  (scaling Identity → InfNorm, then `pivot_threshold^0.75`), gated by
  `feral_increase_quality` (gh#850: "a *lateral* move in trajectory terms");
  `feral_increase_quality_retry`. A second escalation on one matrix is a
  **second rung**, which is where the counterfactual's `q → 12/24` lines
  come from.
- gh#857 accounting (`quality_escalations`, "escalations never step back
  down", `issue857_*` on `deb7` / `circle_parabola_t0p5`),
  `issue_983_scale_audit_retry_budget`.

**Hypothesis.** The counterfactual let refinement escalate *without bound*
per matrix (every stall → another rung), and that, not the per-consumer
split, is what ran 15 lines to the cap. One escalation per consumer per
matrix — two flags, `augsys_improved_inertia` and `augsys_improved_refine`,
each set once — gives the iteration-4 refinement its single rung and no
line more than two rungs on one matrix. If the measure-first step shows the
collateral lines took exactly one refinement escalation each (i.e. the
split alone reproduces them), the item must narrow further: allow the
refinement escalation only when the inertia-branch escalation was spent on
a count the next factor refuted (came back `Singular` or at the expected
count), or only when `δ_c = 0` on the matrix.

**Reproduce.** `p-prev` exact 492; iteration-4 info string shows `q` from
the inertia branch and `S` from refinement.

**Measure first.** Re-run the counterfactual (env-gated probe, as in R2) with
a per-matrix counter of *refinement* escalations, over the 19 moved lines:
how many matrices escalated once vs more than once from refinement, and on
the three named collateral lines specifically. Also, on the baseline, count
matrices where the inertia branch spent the flag and refinement later
stalled (the population this item changes) — on the whole corpus, both legs.

**Steps.**
1. Split the flag; `escalate_aug_quality` takes which consumer is asking;
   the `q` info string and `quality_escalations` tally are unchanged in
   meaning (count every successful escalation).
2. The refinement branch's `if !self.augsys_improved` becomes `if
   !self.augsys_improved_refine`; the inertia branch's likewise. Both reset
   where the old flag resets.
3. If the measurement demanded the narrowing, implement that condition and
   name the population it was measured on in the doc comment.
4. Unit test with a fake aug solver: inertia branch escalates, refinement
   stall still gets one escalation, a second refinement stall does **not**
   (mutation: the old single flag fails the first assertion, an unbounded
   rule fails the second). CLI pin as in R2c.

**Acceptance.** As R2c's (pooling ≤ 300, resto < 24, an optimum), plus: of
the counterfactual's 19 lines, at most the `pooling_rt2stp` line and lines
that *improve* in `it=` without changing `q` to the cap may move; lbfgs
`circle_parabola_t0p5` and lbfgs `scale_audit_retry_meyer3` unchanged.

**Validation.** As R2c (sweep both legs, crate tests, Python test if it
builds, M1), plus `issue857_*` and `issue_983_scale_audit_retry_budget`
named explicitly. Status at best `swept — needs vanderbei before merge`.

**Must not regress.** Everything in R2c's list; gh#850's
`feral_increase_quality=no` contract (the sweep line `cresc4` lbfgs carries
that second-opinion rung — it must still appear and still fail the same
way).

**Stop.** More than three of the counterfactual's 19 lines move the same
way they did under the counterfactual; any `q` running to 12/24 that was
not there before; two iterations.

### R2e — the gh#592 walk-back refuses a numerically singular `δ_c`-free factor

**Goal.** Keep `δ_c` at iteration 4 instead of accepting a factor with
min |eig| 2.0e-17, so the 6.4e16 step is never produced and neither R2b
nor R2c has anything to repair.

**Files / symbols.**
- `crates/pounce-common/src/pd_perturbation.rs` `maybe_withdraw_delta_c`
  (~631, gh#592 walk-back), `perturb_for_singular_with` (~327, gh#981
  reinstatement with the `certified` argument), the gh#592 and gh#981 unit
  tests; `dev-notes/issue-592-restart-non-idempotence.md`,
  `dev-notes/issue-981-delta-c-walkback.md`.
- `crates/pounce-feral/src/lib.rs`: `min_pivot_magnitude` (FERAL
  `solver.rs` ~2583), `singular_pivot_floor` default **1e-20** (~712 —
  effectively off), `inertia_trust_floor` = `n·ε` (~200),
  `singularity_certified` (~1721). `crates/pounce-linalg`
  `SparseSymLinearSolverInterface` / `AugSystemSolver` for where a
  `min_pivot_magnitude()` accessor (default `None`) belongs.
- `pd_full_space_solver.rs` ~1640–1680, where a `Success` is accepted; the
  handler knows whether the system it just factored is the one whose `δ_c`
  the walk-back withdrew.

**Hypothesis.** Both versions accept the withdrawn-`δ_c` factor because
FERAL says `Success` (72 negatives, correct) — the count is right and the
matrix is still singular to working precision. v0.12.0 survives it only by
luck downstream. A factor whose `min_pivot_magnitude` is below the inertia
trust floor (`n·ε ≈ 1e-14` here; the eigenvalue is 2e-17, the LDLᵀ pivot
may be larger — **measure**) should be treated as `Singular`-certified,
which sends the handler to `perturb_for_singular_with(certified = true)`
and reinstates `δ_c` (gh#981's path), with no change to refinement or
escalation.

**Reproduce.** `p-prev` exact 492; the R2 probe's iteration-4 line
(`Success` at `δ_x = 1e-4`, `δ_c = 0`, min |eig| 2.0e-17).

**Measure first.** Expose `min_pivot_magnitude` through a tracing line in
`pounce-feral` (or the probe) and, over the full sweep corpus both legs,
record for every **accepted** factor that follows a walk-back: `min_piv`,
`max_piv`, `n·ε`, and whether the fixture-leg ends solved. Histogram. The
item is refuted if the pooling iteration-4 pivot is **not** below the
floor (then the eigenvalue is small but the factor does not know it, and
the lever is R2c's); it is a blanket change if many healthy walk-backs sit
below the floor (then `the_walkback_is_inert_on_this_fixture_since_gh693`
and the gh#592 fixtures will tell you quickly).

**Steps.**
1. `min_pivot_magnitude()` on the aug-solver trait, `None` by default,
   implemented for the FERAL backend.
2. In `pd_full_space_solver.rs`, on `Success` for a system whose `δ_c` the
   walk-back just withdrew: if `min_piv < inertia_trust_floor`, treat as
   `Singular` with `singularity_certified` true → `perturb_for_singular_with`
   reinstates `δ_c`. Info letter (e.g. `W`) so the table shows it. Do not
   touch `maybe_withdraw_delta_c`'s decision to *try* the withdrawal — the
   change is to what counts as the withdrawal succeeding.
3. Unit test with a fake backend reporting `Success` and a sub-floor min
   pivot after a walk-back → `δ_c` reinstated; mutation: without the check
   `δ_c` stays 0. CLI pin as in R2c.

**Acceptance.** As R2c's (pooling ≤ 300, resto < 24, an optimum). In
addition the gh#592 fixtures (`issue_592_delta_c_walkback.rs`,
`issue981_cstr_dup_row`) keep their counts: the walk-back must still
withdraw `δ_c` wherever it does today unless the factor is sub-floor.

**Validation.** As R2c. Status at best `swept — needs vanderbei before
merge`; `steenbr*` and `dallas*` are where `δ_c` reinstatement has
historically moved counts.

**Must not regress.** `the_walkback_is_inert_on_this_fixture_since_gh693`;
every gh#592 walk-back test; gh#981's reinstatement test; `issue_250`;
`deb7` exact `SolveSucceeded`.

**Stop.** The pooling pivot is above the floor (refuted — record); the rule
fires on more than a handful of healthy walk-backs in the corpus; the fix
needs the gh#592 walk-back disabled on a fixture it is inert on today (the
206 → 812 history in CLAUDE.md is the one thing this family must not
reopen); two iterations.

### M1 — re-measure the inventory after every landed R2x item

Not a code item; a rule that runs inside each R2b/c/d/e iteration (and B1's)
before the next sibling starts, because R3 landed **before** the R2 family
and the original "R3 is re-baselined after R2" order is inverted.

**After each landed R2x item**, on the new tip, record in the item log and
in the inventory Status cells:

| fixture (leg) | what to record | why |
|---|---|---|
| `pooling_rt2stp` exact + lbfgs | status, it, q, objective (which optimum), restoration calls | the target; the lbfgs line is the collateral check (82 it, −3273.954992 today) |
| `dallass` exact + lbfgs | status, it, restoration calls | R3's 28 / 301 is the baseline **now**; an R2x item that moves `dallass` is a reportable line (it has no kink left, so any move is pure trajectory) |
| `deb7` exact + lbfgs | status, it | 114 `SolveSucceeded` / `RestorationFailed` 1206 — the guard the plan names for every NLP-arm item |
| `scaled_feasible_a` both | engine, it | must **not** move (convex arm); if it does, the item leaked into routing |
| lbfgs `circle_parabola_t0p5`, lbfgs `scale_audit_retry_meyer3` | it, q, 2nd, objective | the counterfactual's collateral lines |
| F1 margin | 500 − pooling exact it | `issue_592` pin; tighten only per §4 |

**Ordering consequences.** The first R2x item's `p-prev` is the tip that
contains R3 (`250857b0` or later). B1 waits for the R2 family to finish
(it changes pivoting → trajectories → would confound the sibling
re-baselines). R4/R4b are unaffected by any of this but their
`sweep-fixtures.sh` diff must still be empty wherever they land. R1b–R1e
touch only the convex arm and can be interleaved anywhere after B1 without
affecting M1's table (their own re-baseline table is in §4's sibling rule).

### R1d — cut the first, unequilibrated HSDE solve short

**Goal.** `scaled_feasible_a` stops costing 400 HSDE iterations for a
reported 200: the first solve, which ends `IterationLimit` at `true_kkt
0.149` anyway, is recognised as hopeless early and the gh#293 equilibrated
retry gets the budget. Verdict, engine and the *reported* count are
unchanged by construction; wall time roughly halves on this model. This is
the one R1 route with independent value whatever R1b/c do, because the
detector applies to any HSDE solve that would run to the cap and lose.

**Files / symbols.**
- `crates/pounce-convex/src/hsde.rs`: `status = QpStatus::IterationLimit`
  (~617) and the main loop (~631); `rel_candidate` / `restore_candidate`
  (~307, ~1039, ~1576); the post-loop `true_kkt_error` upgrade (~1620–1640);
  the gh#989 proximal-stall guard (~774–795, `PROX_STALL_ITERS`) as the
  precedent for an in-loop stall detector; the trace (`QpIterate`).
- `crates/pounce-convex/src/ipm.rs` ~601–690: `retry_on =
  NumericalFailure | IterationLimit | OptimalInaccurate`; the retry is
  accepted only when genuine (gh#712); `equilibrated_retry_budget` (~807):
  full `max_iter` when `P ≠ 0`, capped on a pure LP. `hsde_scalars::tracked`
  for measuring both solves' counts.
- Tests: `issue_689_direct_driver_scaled_feasible.rs`, `issue984_stopping_units.rs`,
  `issue_1000_*`, `issue414_*`, `illconditioned_huge_scale.rs`.

**Measured (R1 log).** First solve: τ 1.6e-9 → 7.6e-8 **linearly** over
it 19–199, `mu` ~1e-37, `true_comp` ∝ 1/τ (7.2 → 0.149), ends
`IterationLimit` at `true_kkt_error 0.149`. Retry: 200 more, `Optimal` by
the post-loop certificate. `max_iter = 400` converges at 230.

**Hypothesis.** A solve in which τ has collapsed (≪ 1) and is growing only
linearly while `mu` is already far below anything the stopping test can
use is not going to certify within the budget; a conservative pattern
detector — e.g. over a window of `k = 20` iterations, τ < 1e-6, `mu <
tol²`, and `true_kkt_error` falling by less than 2× — returns
`IterationLimit` at the window's end instead of at the cap, and the
existing retry logic does the rest. It changes nothing on a solve that
converges, and on a solve that would have ended `IterationLimit` it only
changes *when*.

**Reproduce.** `p-prev` exact: engine `cvx-qp`, 200 it reported; with
`hsde_scalars::tracked` (or a temporary counter) confirm 200 + 200 HSDE
iterations in the call.

**Measure first.** Per-iteration τ, `mu`, `true_kkt_error`, α on **every**
HSDE solve of the sweep corpus (92 `cvx-qp` + 14 `cvx-qcqp` lines, both
legs — same binary, so one leg suffices — and the retries, which the sweep
does not see): for each solve, its final status and the first iteration at
which the candidate pattern holds. The detector is admissible only if it
fires on **no** solve that ends `Optimal` (genuine) and fires on
`scaled_feasible_a`'s first solve well before 200 (target ≤ 60). Report the
table: solves examined, solves that end `IterationLimit`, solves the
detector would cut, iteration saved.

**Steps.**
1. Implement the detector in the main loop as a status reason distinct
   from the cap (e.g. `IterationLimit` with a trace/stat flag `early_exit`),
   so JSON and the sweep can tell them apart if the reason is ever
   surfaced; `retry_on` needs no change.
2. `pounce-convex` test on `scaled_feasible_a`'s data through the public
   solve: the first solve returns before iteration 80, the overall answer is
   still `Optimal` with `|f| ≤ 1e-3` and `true_kkt < tol`. Mutation: with
   the detector disabled, the first solve runs 200.
3. If the sweep's retries show any *accepted* retry whose first solve the
   detector would have cut earlier, record it — that is the win beyond
   this fixture.

**Acceptance.** `scaled_feasible_a`: total HSDE iterations across both
solves 400 → ≤ 260; reported engine / status / objective / `iteration_count`
unchanged (200, until R1b/c change it); wall time down; `scaled_feasible_b`
36 unchanged; the detector fires on zero solves that end `Optimal` in the
corpus.

**Validation.** `scripts/sweep-fixtures.sh` both legs must diff **empty**
(the sweep reads the reported count, which is the retry's) — a moved
convex line means the detector fired on a returned solve and is a stop.
`cargo test -p pounce-convex -p pounce-cli`; `python/tests/test_qp*.py` if
the extension builds. `benchmarks/qp` status + total iterations + wall A/B
(0.13.0: 137 optimal, total 2998) — **not available here**, so status at
best `swept — needs benchmarks/qp before merge`.

**Must not regress.** The gh#293 retry acceptance rule and its LP budget;
gh#414 / gh#984 guards; gh#1000 I.D/I.8 pins; gh#689 `ITER_BUDGET`; the
gh#989 proximal-stall guard's own test (do not merge the two detectors).

**Stop.** The detector fires on any corpus solve that ends `Optimal`; it
cannot fire on `scaled_feasible_a` before ~150 without doing so; two
iterations.

### R1e — the post-165 tail: a certifiable iterate waiting on an unreachable `pres`

**Goal.** From retry iteration 165 `comp_ok` holds and `true_kkt_error`
is 8.6e-9 (< tol), yet the loop runs to the cap because `pres < tol` is
unreachable (`pres` is 4 ulp of a `5e9` cancellation) and the relative arm's
`gap_rel` is unmet. Stop there with the certificate.

**Files / symbols.** `hsde.rs` ~898–1000: `abs_ok`, `rel_in_floor`
(needs `pres ≤ cap_p` — already true, `cap_p = 7.1e-5`), `comp_ok`,
`converged`, `REL_FLOOR_KAPPA`, `rel_candidate`; the post-loop certificate
(~1620–1640) and `ipm.rs`'s comment that it "deliberately runs *after* the
loop … breaking early on the certificate would hand back a less-polished
iterate". `issue984_stopping_units.rs`, `issue_1000_*` (I.D/I.8), gh#846.

**Measured (R1 log).** it 165: `pres` 1.1e-6, `dres` 1.3e-10, `gap` 1.3e-4,
`true_kkt` 8.6e-9; it 199: `pres` 8.5e-6, `dres` 7.8e-15, `gap` 3.2e-5,
`true_kkt` 9.1e-15. So the tail **does** polish — six orders in the
certificate — which is the honest counter-argument. R1 declined this
shape as insufficient alone (200 → ~166 at best).

**Hypothesis.** When `true_kkt_error(x/τ, y/τ, z/τ) < tol` **and** the only
unmet stopping term is `pres`, **and** `pres` has not decreased over `k`
iterations while sitting below `κ·ε·scale_p` (its rounding floor), the
loop is waiting on a term it provably cannot meet; stopping is not
"handing back a less-polished iterate" in any sense the stopping test can
express. Expected gain here: ≤ 35 iterations (17%). The item earns its
place only if the measure-first step shows the pattern on other corpus
lines.

**Reproduce.** `p-prev` retry trace, it 165–199 as above.

**Measure first.** On every convex sweep solve: at the final iteration,
which stopping terms were unmet, and for how many consecutive iterations
`pres` had been pinned (no decrease, below `κ·ε·scale_p`) while
`true_kkt < tol`. Count the solves where that pattern holds for ≥ 10
iterations. Zero or one → record and drop (R1b/c subsume the model's
count); several → implement.

**Steps.** Add the exit after `converged` is computed, gated exactly as
the hypothesis says (all three conditions, `k ≥ 10`); emit a distinct trace
flag; test on `scaled_feasible_a` data that the retry stops ≤ 180 with the
same verdict, mutation-checked; keep the post-loop upgrade as is.

**Acceptance.** `scaled_feasible_a` retry ≤ 180 it (alone) with the same
verdict and `|f| ≤ 1e-3`; no convex line in the sweep loses a status or
rises in count. Composition with R1b/c is recorded, not assumed.

**Validation.** Sweep both legs (only convex lines that *fall* may move);
`cargo test -p pounce-convex -p pounce-cli`; Python `test_qp*` if it
builds; `benchmarks/qp` — status at best `swept — needs benchmarks/qp
before merge`.

**Must not regress.** The gh#1000 stopping-test corrections (this adds an
exit, it must not loosen any term), gh#984's "every stash satisfies
`comp_ok`", gh#846, gh#414.

**Stop.** The measure-first count is ≤ 1 corpus solve; any convex line's
objective changes beyond `tol`; two iterations.

### R1b — HSDE centering safeguard: the `σμ` target must not fall below the true-slack complementarity

**Goal.** The retry's it 20 → 125 crawl — `mu` dividing by ~8 every full
step (5.0e-19 → 3.2e-117) while the true complementarity falls linearly
(2.6e5 → 9.8e3) — is replaced by a trajectory that keeps the barrier at the
scale the un-homogenized point actually needs; `scaled_feasible_a` certifies
in ≤ 60 (R1's target) on both legs.

**Files / symbols.**
- `hsde.rs` ~1284–1292: `mu_aff`, `sigma = (mu_aff / mu)³`, `sigma_mu`; the
  gh#218 centering ladder (~145–175, `CENTERING_*`, **step-length**
  triggered — it never fires here because α = 1 throughout); ~931
  `row_slack[i] = h[i] − (Gx)[i]/τ`; ~233 `true_slack_complementarity`
  and ~991 `true_comp` (orthant only); `tol_true_comp`; `correctors.rs`
  (Gondzio, orthant only).
- `dev-notes/issue-689-direct-driver-cold-start.md` (the *direct* driver's
  slack decoupling and how it was re-coupled); `ipm.rs` ~4303 (the
  "≈ 4 ulp" comment on this model).
- Tests: `issue984_stopping_units.rs`, `issue_1000_*`,
  `issue414_cost_normalized_false_optimal.rs`, `illconditioned_huge_scale.rs`,
  `issue_689_*`, the gh#218 centering test, `declared_optimum_sentinel.rs`.

**Hypothesis.** The internal complementarity `s·z` is driven to 1e-37 …
1e-101 on a slack `s` that has decoupled from the model's `h − Gx̂` (the
gh#689 decoupling, now on the HSDE side), so Mehrotra's σ sees a nearly
centered, nearly converged iterate and asks for almost no centering while
the true gap is 1e5. Units: internal `mu` is in homogenized units (`s =
τŝ`, `z = τẑ`), so the comparable true quantity is `τ²·true_comp/(degree+1)`
— at it 20 that is ≈ 2.6e5·(9.7e-11)² ≈ 2.4e-15 against `mu` 5.0e-19, four
orders apart, and ~1e-100 apart by it 120. A floor `σμ ≥ θ·τ²·true_comp/
(degree+1)` (θ to be measured; start 0.1) keeps the target from running
away from the true point. Alternative, if the floor does not bite: re-couple
`s` to the true slack (`s ← τ·(h − Gx̂)` projected into the cone) when the
two disagree by more than a factor κ — the HSDE analogue of the gh#689 fix.
Prefer the floor (one expression in the σ computation, orthant-only like
`true_comp`); measure the re-coupling only if the floor is inert.

**Reproduce.** `p-prev` exact and lbfgs: `cvx-qp` 200 / 200; the R1 log's
retry table.

**Measure first.** Per iteration on the retry: `mu`, `τ²·true_comp/(deg+1)`,
their ratio, σ, α. Then the same on every convex solve in the sweep corpus
(one leg): the **maximum** ratio each solve reaches and at which iteration.
The floor is admissible as a narrow change only if healthy solves keep
the ratio within ~1e2 (so a θ of 0.1 never fires on them); report the
distribution. If healthy solves routinely exceed it, the floor is a
convex-arm-wide retune and the item is "re-coupling or nothing".

**Steps.**
1. Implement the floor behind a constant (`CENTERING_TRUE_COMP_FLOOR`), with
   the units derivation in the doc comment and the measured θ; emit a trace
   flag when it binds.
2. `pounce-convex` test on `scaled_feasible_a` data: ≤ 60 iterations on the
   returned solve, `Optimal`, genuine; mutation: floor removed → 200.
3. If the floor is inert on the retry (ratio never exceeds θ⁻¹ because
   `true_comp` itself is what the stopping test is failing on), switch to
   the re-coupling shape and repeat the measurement — do not do both in one
   iteration.

**Acceptance.** `scaled_feasible_a`: `cvx-qp`, `SolveSucceeded`, `|f| ≤
1e-3`, `iteration_count ≤ 60` on both legs (200 → target; report the
number and whether the first solve now certifies, which also retires the
retry on this model). `scaled_feasible_b` ≤ 36. The floor binds on no
convex sweep solve that ends `Optimal` today, or on the ones it binds on
the count does not rise.

**Validation.** Full sweep both legs `p-prev` vs new: convex lines may move
only downward in count; any rise > 10%, any status or engine change, any
objective change beyond `tol` is a stop; recount the engine split. `cargo
test -p pounce-convex -p pounce-cli`; Python `test_qp*` if it builds.
`benchmarks/qp` total iterations + statuses (137 optimal / 2998 at #1002) —
**not available here**: status at best `swept — needs benchmarks/qp before
merge`. QFORPLAN (33 it at `f6a4e4c2`) is the equality-heavy model the
commit message of #1002 names as sensitive; it is in that corpus.

**Must not regress.** gh#218's centering ladder and test; gh#984's stash
rule; gh#414 `optimum_is_genuine` / `verify_or_repair_optimum`; gh#1000
I.D/I.8; gh#689 `ITER_BUDGET`; gh#846; `cvx-qcqp` lines (14) must not move
at all — `true_comp` is orthant-only and so is the floor.

**Stop.** The floor binds on > 10% of healthy corpus solves; the retry's
ratio never exceeds θ⁻¹ and re-coupling does not reach ≤ 100 either;
two iterations.

### R1c — data-sized HSDE cold start

**Goal.** Start the embedding at the data's scale so τ does not have to
collapse to ~1e-10 to represent `‖x̂‖ = 5e9`, removing the regime in which
every homogenized quantity is 1e-20 of its true size and the R1b crawl
occurs; `scaled_feasible_a` ≤ 60 on both legs, ideally on the **first**
solve so the retry never runs.

**Files / symbols.**
- `hsde.rs` ~533–540: the self-dual start `x = y = 0`, `s = z = e` (cone
  identity), `τ = κ = 1`.
- `ipm.rs` ~4485–4610: the `run_ipm` starting-iterate builder — cold seed
  (`x = 0`, identity duals) through the same Mehrotra recentering as a
  warm seed (gh#689: "what sizes the cold start to the problem's own data";
  the direct solve of this model 175 → 27). The orthant positivity shift and
  SOC λ_min lift are there.
- `issue_689_direct_cold_start_scale.rs`,
  `issue_689_direct_driver_scaled_feasible.rs` (`ITER_BUDGET = 60`, direct
  route), `issue_689_*`; the gh#293 comment in `solve_qp_ipm_core` on why
  HSDE "must not get" Ruiz (a warning about composing preconditioning on
  this arm, relevant here).

**Hypothesis.** The unit start is scale-free by design (Clarabel/ECOS use
it for that reason) but on data with `‖x̂‖ ~ 5e9` it forces τ to carry the
scale; seeding `(x, y, s, z)` from the recentered cold seed at `τ = κ = 1`
starts at the right scale and the embedding's dynamics are unchanged
(interior `(s, z, τ, κ)` is all the theory needs). Infeasibility and
unboundedness certificates still arise from τ → 0 / κ → 0; what moves is
how many iterations each solve takes to get anywhere — i.e. **every** HSDE
trajectory.

**Reproduce.** `p-prev` both legs 200 / 200; the R1 log's note that the
plateau is at `‖x̂‖ = 5e9` in the equilibrated retry too (so equilibration
does not remove the scale).

**Measure first.** (1) On every convex sweep solve, τ at the first
stopping-test pass and at exit: how many solves ever run at τ < 1e-6 (only
those can move materially; the rest move by rounding). (2) In a probe
build, apply the seed to `scaled_feasible_a` only (both solves) and record
counts; if the first solve certifies, the item retires the retry on this
model — a gain R1b/R1d cannot give. (3) The gh#293 LP-budget reasoning
("every accepted LP retry converged by iteration 78") is a trajectory
statistic that will move; note it.

**Steps.**
1. Implement the seed behind an option (`qp_hsde_start = identity |
   recentered`), **default `identity`** until the corpus A/B — this is the
   only way the code can merge from a machine without `benchmarks/qp`; the
   default flip is a separate, corpus-gated commit. Reuse the `run_ipm`
   recentering, do not reimplement it.
2. `pounce-convex` test: `scaled_feasible_a` with `recentered` ≤ 60 both
   solves, `Optimal`, genuine; identity path unchanged (its existing pins).
3. Sweep with the option forced on (temporarily, via the options file or
   an env var the sweep honours) to measure the default flip's blast radius
   now, even though the flip itself waits.

**Acceptance.** With `recentered`: `scaled_feasible_a` `cvx-qp`,
`SolveSucceeded`, `|f| ≤ 1e-3`, ≤ 60 on both legs; `scaled_feasible_b` ≤
36; every `issue_689_*` / `issue_1000_*` / `issue984_*` / `issue414_*` /
`illconditioned_huge_scale` test passes under both settings (run the
convex test suite twice, once per default, via the option).

**Validation.** Sweep both legs at the default (must diff **empty** — the
default did not move) and with the option forced (every convex line is a
candidate; each explained; any status/engine/objective change is a stop;
the 14 `cvx-qcqp` lines included — the SOC lift is part of the seed).
`cargo test -p pounce-convex -p pounce-cli`; Python `test_qp*` if it
builds. Status: the opt-in commit can be `done`; the default flip is
`swept — needs benchmarks/qp before merge` and is listed as such in the
inventory.

**Must not regress.** Everything in R1b's list; infeasibility /
unboundedness certification tests in `pounce-convex` (the seed must not
delay a certificate: pin the iteration at which the corpus's infeasible
models certify, before and after).

**Stop.** Any certificate test flips or certifies > 2× later; the forced
sweep moves a convex status; R1b has already met ≤ 60 on both legs **and**
the first solve does not certify with the seed (no independent value);
two iterations.

### R4b — a structural, MLSL-aware stagnation rule (separate from R4's gamma grid)

**Goal.** `circle_parabola_t0p5` under `--minima mlsl --seed 0 --n-minima 6
--max-solves 30 --patience 8` finds both minima at the current default
`gamma 0.5`, by making the stagnation rule understand what an MLSL launch
that re-finds an archived minimum means — without moving gamma and without
losing the #993 gains on camel / wavy / wavy_1e12.

**Files / symbols.** `crates/pounce-minima/src/search.rs`: `reject` (~599:
`stagnant += 1; if stagnant >= patience → Stop::Converged`), `consider`
(~530: `accepted` requires `!is_known(x)`; a known re-find reaches
`reject`), `near_any` (~592, against the archive at `radius`), `run_mlsl`
(~647–690: pool, `radius = gamma·√n·(ln N/N)^(1/n)`, single-linkage filter
`better_near[i] || near_any(pool_x[i], radius)`), the archive type;
`options.rs` (`patience`, `gamma`); `crates/pounce-cli/src/cli.rs` ~328;
`python/pounce/_minima.py`; `docs/src/find-minima*.md`. Guards:
`issue989_mlsl_cli.rs`, `python/tests/test_minima.py` (camel six minima at
`--max-solves 60 --patience 15`; `sample_cap_reached` at gamma 100),
`python/tests/test_minima_cli_parity.py`, `scripts/sweep-minima.sh` (54
lines; the saddle `found=0` lines and the `kkt_rej` line).

**Measured (R4 table, seed 0).** gamma 0.5: found 1 in **9** solves, status
`converged` — eight consecutive launches re-find minimum 1 and trip
`patience 8` with 21 of 30 solves unused; gamma 2 (0.12.0): found 2 in 11,
because the larger radius filters those launches before they run.
`p-base --gamma 0.5` reproduces found 1 / 9, so it is the rule, not gh#966.

**Hypothesis.** The patience counter was written for multistart, where
every solve is an independent draw and eight failures in a row is
evidence the landscape is exhausted. Under MLSL a launch whose sample
passed the clustering filter and then converged to an archived minimum is
evidence that the **filter's radius under-covers that basin**, not that
the search is done. Rule: on such a re-find (i) grow that minimum's
exclusion radius in the archive to the distance from the launching sample
to the minimum (so `near_any` filters that basin from the next round on),
and (ii) count it as a strike only when the round produced no launch at
all, or at half weight — the stagnation that remains is "rounds without a
new minimum", which is what patience is for.

**Reproduce.** `p-prev` row above (found 1 / 9 / `converged`); count the
re-finds in the trace (`TraceEntry.accepted == false` with `success`).

**Measure first.** *(R4 ran, 2026-10-24: the table is
`dev-notes/issue-1042-r4-mlsl-gamma-grid.tsv`, re-runnable on any binary with
`dev-notes/issue-1042-r4-mlsl-gamma-grid.sh <pounce> <out.tsv> 0.5`; its
`exit_*` columns give the re-find / infeasible split per run — read the R4
log's strike table before implementing, it bears directly on this item's
gate.)* The 10-seed table R4 step 2 produces (seeds 0–9, camel /
circle_parabola_t0p5 / haverly_t3_brf0 / wavy / wavy_1e12, gamma 0.5) —
reuse it if R4 ran; otherwise run it here, adding per run: launches that
re-found an archived minimum, and the stop reason. The item is worth it
only if re-finds dominate the stop on circle_parabola on most seeds and
are rare where found already equals the target (camel 4–5 of 6, wavy 6).

**Steps.**
1. Per-minimum radius in the archive (initialised to the dedup radius,
   grown on re-finds); `near_any` reads it.
2. `reject` takes a reason; a re-find under MLSL does not count a full
   strike (pick (ii)'s exact weighting from the table; document it).
3. `pounce-minima` unit test: a 1-D two-basin function sampled so that
   every early sample lands in basin 1 — without the rule the search stops
   at `patience`; with it, basin 2 is reached within the sample budget.
   Mutation-checked both ways.
4. Python: the crate is shared, so `find_minima` inherits the rule; run
   `test_minima_cli_parity.py` if the extension builds, else say so and
   leave `_minima.py` untouched (no default moved).

**Acceptance.** `scripts/sweep-minima.sh` `p-prev` vs new:
`circle_parabola_t0p5 mlsl` reads `found=2`; no `mlsl` line reads fewer
minima; non-`mlsl` lines diff **empty**. 10-seed table: mean minima on
camel / wavy / wavy_1e12 / haverly not below gamma-0.5's; mean solves not
up by more than 20%. `scripts/sweep-fixtures.sh` diffs empty.

**Validation.** `sweep-minima.sh`, `sweep-fixtures.sh` (empty), `cargo test
-p pounce-minima -p pounce-cli`, the Python tests if buildable; the
10-seed table in the log and the commit message.

**Must not regress.** `mlsl_default_gamma_finds_all_six_camel_minima`,
`the_sample_cap_is_named_as_such`, gh#966 parity, the saddle `found=0`
lines, the `kkt_rej` line, the gh#103 sample budget (the rule must not
make a round that launches nothing loop forever — `note_sample` still
bounds it).

**Sibling rule with R4.** If R4 moves gamma and recovers found=2 on ≥ 8 of
10 seeds, R4b is re-baselined: kept only if it recovers seeds the gamma
change misses or lowers mean solves, with the table; otherwise `dropped —
recovered by R4`. If R4b lands first and recovers, R4 is `dropped` (gamma
stays 0.5, help text and docs unchanged). Never both in one iteration.

**Stop.** Re-finds do not dominate the stop on circle_parabola across
seeds; any `mlsl` line loses a minimum; two iterations.

### R4c — a repeat hit of a locally infeasible point is not a full `patience` strike (sibling of R4 / R4b)

*Added 2026-10-24 by the R4b iteration, from its direct trace count (§6
R4b log, `dev-notes/issue-1042-r4b-mlsl-strike-trace.tsv`). Not
implemented there.*

**Goal.** `circle_parabola_t0p5` under `--minima mlsl --n-minima 6
--max-solves 30 --patience 8` at the default `gamma 0.5` finds both minima
on **at least 8 of 10 seeds** (0.12.0's gamma 2 reaches 7/10; 0.13.0 5/10),
without moving gamma or patience and without moving any line of a model
that has no locally infeasible launches.

**Measured (R4b trace count, x86_64, `p-prev` = `e84acb08` plus a
print-only patch).** On the five gamma-0.5 seeds that stop at found 1
(0, 1, 2, 4, 6) the 40 `patience` strikes are 12 re-finds of minimum 1 and
**28 launches that end `Converged to a point of local infeasibility`** — on
every seed, every such launch lands on the **same** point, (0, −1) to
1e-3: the closest point of the circle `x1² + x2² = 1` to the parabola, an
infeasible stationary point of the constraint violation that drains the
lower half of the box. Per seed the first hit is one strike and the repeats
are 1, 6, 6, 6, 4. With patience unbounded (budget 30) the second minimum
is accepted at solve **11, 10, 17, 16, 14** on those seeds — within the
budget every time; the search is stopped, not exhausted. camel, haverly,
wavy, wavy_1e12 and the sweep's saddle (`found=0`) and `kkt_rej` lines have
**zero** locally infeasible launches.

**Hypothesis.** `reject` was written for draws that are independent
failures. A launch that converges to a locally infeasible point the search
has *already* seen is evidence about one infeasible attractor's basin, not
about whether feasible minima remain; eight hits of one point are one
fact. Rule: keep the landing points of locally infeasible launches in a
side list (never archived, never reported, never a candidate); a launch
that lands within `dedup` of a listed point counts weight `w` < 1 against
`patience`; the first hit of a new infeasible point stays a full strike, so
a model whose launches go infeasible in many different places still
converges.

**Replayed counterfactual (trace replay, not a build).** The rule only
changes the count, not the samples or the solves, so the unbounded-patience
traces replay it exactly; the replay reproduces the measured gamma-0.5 row
of R4's grid to the digit (5.8 / 8.2, 1.5 / 10.0, 2.0 / 10.2, 6.0 / 6.1,
6.0 / 6.0). Cell = mean found / mean solves, seeds 0–9:

| rule | camel | circle_parabola_t0p5 | haverly | wavy | wavy_1e12 |
|---|---|---|---|---|---|
| current | 5.8 / 8.2 | 1.5 / 10.0 (5/10) | 2.0 / 10.2 | 6.0 / 6.1 | 6.0 / 6.0 |
| repeat-infeasible `w = 0.5` | 5.8 / 8.2 | 1.6 / 13.7 (6/10) | 2.0 / 10.2 | 6.0 / 6.1 | 6.0 / 6.0 |
| repeat-infeasible `w = 0` | 5.8 / 8.2 | 1.8 / 19.5 (**8/10**) | 2.0 / 10.2 | 6.0 / 6.1 | 6.0 / 6.0 |
| every infeasible `w = 0.5` | 5.8 / 8.2 | 1.6 / 13.8 (6/10) | 2.0 / 10.2 | 6.0 / 6.1 | 6.0 / 6.0 |
| (R4b's (ii), re-find `w = 0.5`) | 6.0 / 8.9 | 1.7 / 14.5 (7/10) | 2.0 / **18.2** | 6.0 / 6.1 | 6.0 / 6.0 |
| (both at `w = 0.5`) | 6.0 / 8.9 | 1.9 / 22.2 (9/10) | 2.0 / **18.2** | 6.0 / 6.1 | 6.0 / 6.0 |

At `w = 0` the two seeds still lost are 0 and 2, both re-find-limited
(seed 0: seven re-finds plus the first infeasible hit before the second
minimum at solve 11; seed 2: the same, second minimum at solve 17). **Seed 0 — the sweep's seed — is not recovered by this rule**:
its line stays `found=1`, and the acceptance below says so up front rather
than letting the sweep line decide the item. The circle solve count rises
because finding the second minimum costs solves 10–17; that is the price of
the result, not collateral, and it stays under the budget.

**Files / symbols.** `crates/pounce-minima/src/search.rs`: `reject`,
`consider`, `run_solve` / `SolveOut` (carries only `success` and `x`),
`run_mlsl`; `crates/pounce-minima/src/local_solve.rs`: `LocalOutcome`
(`converged`, `x`, `quality` — **no termination status**, so "locally
infeasible" is not visible to the driver today). Two ways to see it: (a) a
new `LocalOutcome` field — public API, and `crates/pounce-py/src/minima.rs`
implements `LocalSolve` too, so both frontends must fill it or the gh#966
parity test is the thing that catches the drift; (b) driver-side, from the
point alone: a failed launch whose landing `x` violates the base problem's
constraints by more than `constr_viol_tol`-scale (one `eval_g`), which
needs no API change and is frontend-independent. Prefer (b) unless the
measure-first step shows it misclassifies.

**Measure first.**
1. Reproduce: `p-prev` sweep line `circle_parabola_t0p5 mlsl converged
   solves=9 found=1`, and the R4b trace table's circle rows (re-run the
   count with a print-only patch on `consider`, as the R4b log shows).
2. Classifier check, before any counting change: over the 10-seed traces of
   all five grid fixtures plus the sweep's 54 lines, does (b) flag exactly
   the launches whose `EXIT:` banner reads `Converged to a point of local
   infeasibility` — no feasible-but-failed solve (`MaxIter`, `ErrorInStep`)
   and no converged one? Record the confusion counts. A failed launch that
   is feasible must keep its full strike.
3. Distinct-point check: across the same runs, how many distinct infeasible
   landing points per run (R4b measured 1 on all 10 circle seeds). If any
   fixture shows many distinct points, the first-hit-is-full-strike clause
   is what bounds it; say how many strikes that run ends on.
4. Pick `w` from the replay table above (0 meets the goal; 0.5 does not) and
   re-run the replay on the built binary — the built rule must reproduce
   the replay's per-seed numbers, or the difference is explained.

**Steps.** Side list of infeasible landing points on `Driver` (reset
nowhere: it is search-global like the archive); `reject` takes a reason;
the repeat-infeasible reason adds `w`; `stagnant` stays an integer count
of full strikes plus a separate counter if `w` is fractional (no float
equality on the patience test). Multistart and the other four methods go
through the same `reject`: decide per method whether the rule applies
(it is about independent draws, so multistart is the other candidate) and
measure; if a non-mlsl line moves, that is the item's own blast radius and
is explained line by line, not waved through. `pounce-minima` unit test: a
fake `LocalSolve` whose launches from half the box land on one fixed
infeasible point and from the other half on a second minimum — without the
rule the search stops at `patience` with one minimum, with it the second
is found inside `max_solves`; mutation-checked both ways (rule off → red;
first-hit also exempted → a second test with many distinct infeasible
points must go red, because the search must still converge there).

**Acceptance.** 10-seed table at gamma 0.5 (R4's grid script, or the R4b
trace count): circle_parabola_t0p5 finds 2 on **≥ 8/10** seeds; camel /
haverly / wavy / wavy_1e12 rows **identical** to the R4 grid's gamma-0.5
rows (they have no infeasible launches, so any change is a defect in the
classifier); circle mean solves reported, ≤ 30 by construction.
`scripts/sweep-minima.sh` `p-prev` vs new: every line diffs empty except
lines on models with locally infeasible launches, each explained; the
seed-0 `circle_parabola_t0p5 mlsl` line is **expected to stay `found=1`**
(re-find-limited, above) and is not a failure of this item. The `mlsl`
lines never read fewer minima. `scripts/sweep-fixtures.sh` both legs diff
empty (single-solve path untouched).

**Validation.** As R4b: `sweep-minima.sh`, `sweep-fixtures.sh` (empty),
`cargo test -p pounce-minima -p pounce-cli` (incl. `issue989_mlsl_cli`),
the new unit test and its mutation check, fmt, clippy (§4 command),
`python/tests/test_minima_cli_parity.py` if the extension builds — if (a)
is chosen it is mandatory, not optional.

**Must not regress.** As R4b, plus: a model whose launches go infeasible
at many distinct points still stops by `patience` (the first-hit clause);
the gh#103 sample budget and `max_solves` still bound every loop.

**Sibling rule.** R4c moves a different population from R4b's (infeasible
repeats, not re-finds), so it is independent value under §4 rule 2. If a
later iteration revisits R4b's re-find weighting it is re-baselined against
the R4c tip and still has to answer haverly's +78% solves.

**Stop.** The classifier cannot separate local infeasibility from other
failures without a public API change *and* the Python frontend cannot be
built to run the parity test; any non-infeasible line moves; circle stays
below 8/10 on the built binary; two iterations.

## 4. Loop protocol

**Pick the next item** in the §2 order whose Status is `todo` and whose
dependencies are `done`, `skipped`, `not recoverable` or `dropped` (B1 waits
for the R2 family; R4b waits for R4's table, R4c for R4b's trace count; B2/B3 wait for B1 and the
corpus). One item per iteration. Never start a second item in the same
iteration, even if the first was quick — its sweep is the evidence for its
own commit.

**Status vocabulary.** `todo`; `done`; `not recoverable (reason)` (hypothesis
refuted, docs-only commit); `dropped — recovered by <id>` (a sibling's
reproduce step no longer reproduces, numbers recorded); `re-baselined after
<id>: <numbers>` (a sibling that still has something to recover, with its
acceptance restated against the new tip); `swept — needs <corpus> before
merge` (code, tests and the fixture sweep are in and explained, but the
item changes an arm's trajectory corpus-wide and the corpus that would
measure the blast radius is not on this machine — `benchmarks/qp` for the
convex arm, `vanderbei` for the NLP arm, `mittelmann` added for B1). A
`swept — needs …` item is **not** `done`: it is left for the owner to A/B
on a machine with the corpus, and its commit message says so in the
first line after the numbers. `POUNCE_BENCH_DATA` is unset here and
`bench_data_root` resolves nothing; re-check once per iteration, not more.

**Sibling items** (R2b/c/d/e; R1b/c/d/e; R4/R4b/R4c). Each family shares one
defect; the members are different levers on the same iteration-4 chain or
the same τ-collapse. Rules:

1. Before starting a sibling, run its own **reproduce** step on the current
   tip (which may contain an earlier sibling). If the regression no longer
   reproduces to within the item's acceptance, mark it `dropped — recovered
   by <id>` with the before/after numbers and do not implement it. A
   docs-only commit records the drop.
2. If it still reproduces partially, restate the acceptance against the new
   numbers in the item log (`re-baselined after <id>: …`) and proceed only if
   the sibling's hypothesis names a mechanism the landed item does not
   touch (R1d's first-solve cut after R1b has fixed the retry; R2e's
   walk-back after R2c has fixed the exit). "The same fixture moves again"
   is not independent value; "a different population moves" or "the same
   fixture's *hidden* cost (first-solve iterations, restoration calls,
   wall time) moves" is. Say which, with numbers.
3. Never land two siblings whose only evidence is the same fixture line.
   The second must move something the first did not, and its sweep must be
   taken against the tip containing the first.
4. A landed sibling is not reverted because a later one would have been
   narrower. Record the comparison (both counts, both collateral lists) in
   §5 and leave the choice to the owner.
5. M1's table is re-recorded after every landed R2x item; the R1 family
   re-records `scaled_feasible_a` / `_b` both legs, the total HSDE
   iterations of the call (both solves), and the convex engine split.
6. A family is closed when its target is met or every member is `done`,
   `dropped`, `swept — needs …` or `not recoverable`; the inventory row of
   the parent (R1, R2, R4) gets a one-line summary pointing at the member
   that did it.

**First attempt, not only the reported result (C1).** Every before/after
number an item records — reproduce step, acceptance, M1 table, sweep
explanation — names the **first attempt** and any retry that replaced it,
not only the final reported result. Concretely: read the sweep's `mu=` and
`2nd=` columns (or the JSON's `statistics.mu_strategy_fallback` and
`second_opinion`) for every fixture the item claims; a line whose final
result improved while `mu=` or `2nd=` changed from `-` to a promotion is a
**lost first attempt rescued by a retry**, not a recovery, and its
before/after is reported as `first attempt status/it → status/it, total
it`. A pin on an item's target asserts the first attempt (no promoted
fallback, no second opinion) before it reads any count. The sweep diff is
taken on all columns; when comparing against a sweep taken before a column
existed, drop it first (`sed -E 's/ mu=[^ ]+ +/ /'`) so the real moves stay
visible.

**Per-iteration checklist.**
1. `git status` clean, on `claude/pounce-regressions-recovery-tz8uh2`,
   `git pull --ff-only origin claude/pounce-regressions-recovery-tz8uh2`.
2. Build `p-prev` from the current tip (release, `pounce-cli`), and if
   missing, `p-base` from `v0.12.0` in a worktree with its own
   `CARGO_TARGET_DIR`. Keep both outside the tree.
3. **Reproduce** the regression with `p-prev` exactly as the item's
   "reproduce" step says, and write the numbers into the item log *before*
   changing code. If the regression does not reproduce, stop and record
   that (it may have been recovered by an earlier item — that is a result).
4. Measure what the item says to measure before implementing (R2 step 2, R4
   step 2). If the measurement refutes the hypothesis, mark the item
   `not recoverable (reason)` with the data, suggest a follow-up issue
   title in the log, and end the iteration with a docs-only commit of this
   file.
5. Implement the narrowest change; add the regression test; mutation-check
   it (revert the fix, the test fails; restore).
6. Validate per the item: `cargo fmt --all -- --check`, `cargo clippy
   --workspace --exclude pounce-hsl --all-targets -- -D clippy::correctness
   -D clippy::suspicious`, the crate tests named, then the sweep(s) named,
   `p-prev` vs new (and vs `p-base` for the item's own fixtures). Paste the
   full `diff` of the sweep into the item log, with one line of explanation
   per moved line. An unexplained moved line is a stop, not a footnote.
7. Update this file: the inventory Status cell, the item log (numbers
   before/after, sweep diff, what was not run and why).
8. Commit code + this file together. Message format:

   ```
   <area>: <what changed> (#1042 <id>)

   <2–6 lines: before → after numbers on this machine, root cause in one
   sentence, what the sweep moved and why, what was not run.>

   Refs #1042
   <the attribution trailers your session's reminder specifies>
   ```

   Use `Refs`, not `Fixes`/`Closes`: #1042 stays open until the owner
   closes it. Name the architecture every number was taken on.
9. `git push origin claude/pounce-regressions-recovery-tz8uh2`.
10. If CI on the branch goes red on something this iteration touched, the
    next iteration fixes it first (docs-only commits excepted).

**Stop / skip conditions.** Mark an item `not recoverable` (with the
measurement) rather than forcing it when any of these holds: the
measurement in step 4 refutes the hypothesis; the fix needs the fix that
caused the regression to be weakened (the `2599c2be` reset, the `5fc8c797`
exemption, the gh#1000 stopping-test corrections, the gh#414/gh#984
guards); the sweep moves a status, an engine, or an objective outside a
known bistable pair and you cannot explain it from the change; the item
needs arm64 or the benchmark corpus and neither is available; or two
iterations have been spent on it. In every such case write the suggested
follow-up issue (title + the three numbers that justify it) into §5 of this
file and move on.

**Keeping the branch green.** Every commit carries its own passing tests
and formatted code; never commit a half-implemented item (stash it or
revert). Do not re-point an existing pin to make a number fit — the
`issue_592` `< 500` bound in particular is left alone unless R2 lowers the
count well below it, in which case tighten it to the measured margin, never
raise it. Docs-only iterations (skips) still run `cargo fmt --check`.

## 5. Judged unrecoverable or out of scope

- **X1 exact `deb7` arm64 Optimal/147 → Acceptable/180.** Same objective to
  7 digits; on x86 the same change is an improvement (143 → 114, measured).
  A NEON-vs-AVX2 rounding difference on a nonconvex model is not something
  this branch can measure or should steer. Leave recorded in #1042; if the
  owner wants it chased it needs an arm64 machine and the R2 ladder
  measurement there.
- **X2 lbfgs `dallass` arm64 Optimal/224 → ErrorInStep/308.** On x86 the
  lbfgs leg fails on **both** 0.12.0 (ErrorInStep/213) and 0.13.0
  (ErrorInStep/146 + ladder, tot 292), so the issue's "arm64 only" names
  where the *verdict* changed, not a leg that is otherwise healthy. The
  lbfgs failure on this model is older than 0.13.0 and is a separate
  defect (suggested follow-up: "dallass under limited-memory ends
  Error_In_Step_Computation on x86 on 0.12.0 and 0.13.0").
- **X3 lbfgs `pooling_rt2stp` arm64 lands on −3273.95 instead of −4391.83.**
  Both are certified optima of a bistable nonconvex model; the tests already
  accept either; on x86 the leg went from failing to solved. Nothing to
  recover.
- **lbfgs `deb7` x86 ErrorInStep/1010 → RestorationFailed/1206 (tot 1738).**
  Not in the issue; both fail; recorded here so the next sweep reader does
  not mistake it for a new loss.
- **B4 `cresc132` Optimal → Infeasible** (Ipopt's verdict too) and
  **`ssebnln` Optimal → Acceptable** (same objective): corpus-only,
  FERAL-trajectory, no mechanism named by anyone, and `cresc132` was already
  in the report's "genuine misses" list. Out of scope unless B1/B2 work
  happens to move them; record if it does.
- **R1 `scaled_feasible_a` cvx-qp 200 / 200 it (both legs).** Not
  recoverable with the planned stall-gated certificate or `cap_p` floor: the
  instrumented loop (log below) shows the model has **no equality rows**
  (`m_eq = 0`), the `pres = 5.0e-6` plateau is the rounding floor of
  inequality rows at `‖x̂‖ = 5e9` (`ε·5e9 = 1.1e-6`; `cap_p = 7.1e-5`, which
  `pres` already satisfies), and through the plateau the un-homogenized point
  is **not optimal** (the candidate shapes at the end of this bullet are now
  items R1b–R1e in §3) (`true_kkt_error` 2.6e5 at it 20, 9.8e3 at it 120, 23 at
  it 125). The earliest any in-loop certificate could stop is it 165, after a
  blow-up (`inf_pr` 4.8e9 at it 140) and a 15-iteration recovery — 200 → ~166
  at best, and dependent on that blow-up. Suggested follow-up issue: **"HSDE:
  μ collapses ~8×/iteration while true complementarity crawls linearly once
  τ ≈ 1e-10 (`scaled_feasible_a`)"** — numbers: retry it 20 → 120 `mu`
  5.0e-19 → 3.2e-117 with `alpha = 1` every step while `true_comp` only goes
  2.6e5 → 9.8e3 and `dres` falls by a constant 5.0e-7/iteration; the first
  (unequilibrated) HSDE solve also ends at the 200 cap with
  `true_kkt_error 0.149` (τ creeping linearly 1.6e-9 → 7.6e-8, `true_comp`
  ∝ 1/τ), so the model costs 400 HSDE iterations in all; `max_iter=400`
  converges at 230. Candidate shapes (both trajectory changes across the
  whole convex arm, so both need `benchmarks/qp`, absent here): a centering
  safeguard that keeps the `σμ` target from falling below the
  *true*-slack complementarity when the internal `s` has decoupled from
  `h − Gx̂` (the gh#689 decoupling), or an HSDE cold start sized to the
  data the way the direct driver's cold seed is (gh#689: Mehrotra
  recentering of the seed in the `run_ipm` starting-iterate builder,
  `crates/pounce-convex/src/ipm.rs` ~4485, which took the direct solve of
  this model 175 → 27 iterations). The `ipm.rs` doc comment on the direct
  driver's scale-relative arm (~4303) already names this plateau as `≈ 4 ulp`
  of a `5e9` cancellation — the plan's "regularization floor" reading was
  wrong.
- **R2 `pooling_rt2stp` exact 492 it (v0.12.0: 162).** (The candidate
  shapes at the end of this bullet are now items R2b–R2e in §3, plus R2d for
  the 71-vs-72 count itself.) Not recoverable
  with the planned Gershgorin / quasi-definite acceptance: of 1310 ladder
  factorizations, the 146 `WrongInertia` ones with `δ_c > 0` **all** have an
  indefinite (1,1) block (0 PD, 0 Gershgorin-dominant), so the rule is
  inert here. The high `δ_x` is the correct answer to a `W` whose λ_min is
  −1e13…−1.2e19 (151 of 516 systems below −1e10; v0.12.0: worst −2.2e9,
  none below −1e10). The `W` blow-up starts at iteration 4: the gh#592
  walk-back withdraws `δ_c` (both versions do this) and the `δ_c`-free
  factor at `δ_x = 1e-4` is numerically singular (min |eig| 2e-17, FERAL
  reports `Success`, 72 negatives); its solve has a dual part of **6.4e16**.
  On v0.12.0 the same decision gives 3.5e16, but iterative refinement stalls
  (7.5e-9), escalates pivot quality, refactors and gets **3.0e6**. On
  0.13.0 the ladder's first factor at iteration 4 reported 71 < 72
  negatives, which takes `pd_full_space_solver.rs`'s too-few-negatives
  branch and spends the matrix's one quality escalation there
  (`augsys_improved = true`); refinement then stalls at 1.6e-8, cannot
  escalate, takes the `S` exit (ratio < `residual_ratio_singular`) and keeps
  the 6.4e16 step; λ_min(W) is −4.1e13 at iteration 5. Counterfactual
  (probe build only, env-gated, not committed): letting refinement escalate
  even when `augsys_improved` is set takes `pooling_rt2stp` exact
  **492 → 109** (−4391.826, the other certified optimum; 3 restoration
  calls), `dallass` 123 and `deb7` 114 unchanged — but as a blanket rule it
  moves 19 sweep lines (escalations run to the cap of 12/24 on 15 of them;
  lbfgs `circle_parabola_t0p5` changes second-opinion rung and objective
  8.155 → 10.428; lbfgs `scale_audit_retry_meyer3` 112 → 152). Suggested
  follow-up issue: **"PD solver: a quality escalation spent by the inertia
  ladder's too-few-negatives branch leaves iterative refinement none for
  the same matrix, so a numerically singular `δ_c`-free factor's step is
  kept (`pooling_rt2stp`, 492 vs 109 it)"** — numbers: iteration-4 dual
  step 6.4e16 kept vs 3.0e6 after escalation; λ_min(W) −2.2e9 (v0.12.0) vs
  −1.2e19 (0.13.0); 492 → 109 under the blanket counterfactual with 19
  collateral sweep lines. Candidate shapes for that issue: track the two
  escalation reasons separately (one for the inertia branch, one for
  refinement, per matrix), or refuse the `S` exit when the step's dual part
  is many orders above the right-hand side's. Both are trajectory changes
  across the NLP arm; they need the full sweep and the vanderbei A/B.
- **R2d `pooling_rt2stp` iteration-4 count 71 of 72.** Not a FERAL count
  defect and not recoverable in `pounce-feral`. The matrix FERAL 0.21
  factors there (δ_x = δ_c = 0) has exactly two eigenvalues at working
  precision (unscaled: faer +2.2e-17 / −3.8e-16, LAPACK −1.6e-15 /
  +1.9e-15 — the two solvers disagree on their signs; in FERAL's
  in-run scaling 1.6e-18 / 1.1e-15 against a next eigenvalue of 7.7e-4),
  so 70, 71 and 72 negatives are all honest readings of it and the
  remedy is `δ_c`, not a pivot rung. FERAL 0.21, 0.20 and 0.18 all report
  71 on the dump at default settings and 70–72 across pivot thresholds,
  scalings and orderings; v0.12.0's iteration-4 matrix is a different,
  nonsingular one (min |λ| 2.4e-9; exact 73; all three versions 73). The
  noise rule (`sub_floor_count_is_noise` would say "noise": deficit and
  `n_threshold_fail_accepts = 1`) is never consulted because the in-run
  `min_piv` 5.95e-14 is 1.8× the `150·ε = 3.33e-14` floor; replaying the
  in-run ordering and scaling on the dump, FERAL 0.20/0.18 put that pivot
  at 6.0e-15 (under the floor → `Singular`, no escalation). Three
  workarounds measured on `p-prev` + probe and rejected (log): pinning the
  floor (6e-14 → `SolvedToAcceptableLevel`/191 at −3580.0, not an optimum;
  1e-13 → 140; 1e-12 → 495: chaotic, and the `n·ε` floor *is* gh#592's
  fix); a floor relative to `max_piv` (pooling 471, `deb7` exact
  114 → 606, lbfgs `dallass` solved → `ErrorInStep`); reporting every
  deficit `Singular` regardless of the floor (pooling **162**, −4391.826,
  3 restoration calls — but lbfgs `dallass` `SolveSucceeded`/301 →
  `ErrorInStepComputation`/269). No upstream issue is warranted. Data and
  the analysis program: `dev-notes/issue-1042-r2d/`. What it means for
  the siblings: R2b's narrowing condition ("the inertia-branch escalation
  was spent on a count the next factor refuted") is met at this exact
  factor — the refactor after the escalation came back 70 and was routed
  `Singular` by the noise rule.
- **R2c refuse refinement's `S` exit when `‖res‖∞ > κ·‖rhs‖∞`.** Not
  recoverable as a narrow rule: the measure-first histogram (log) puts
  the pooling iteration-4 exit (`‖res‖/‖rhs‖` = 2.2e14) 62nd of the
  corpus's 188 `S` exits; 135 sit above `κ = 1e6` (the normalizer's own
  `max_cond`), 90 above `1e12`, and the item's "> 10% of `S` exits" stop
  is cleared only at `κ ≥ 4.7e20`, six orders above the target. The
  `S` exit's capped ratio is blind to step size *by design* on every
  restoration-heavy model in the corpus, not on pooling alone. The
  env-gated counterfactual at `κ = 1e6` moves 15 sweep lines (at `1e12`,
  11) — pooling's base solve fails (`ErrorInStepComputation` 795; 173 at
  `1e12`) and is promoted from the `mu_strategy=adaptive` rung at 140; exact
  `circle_parabola_t1p5` `InfeasibleProblemDetected`/603 →
  `MaximumIterationsExceeded`/3000; lbfgs `scale_audit_retry_meyer3`
  `SolveSucceeded`/112 → `SolvedToAcceptableLevel`/286 and lbfgs
  `circle_parabola_t0p5` 8.155 → 10.428 (two of R2's three counterfactual
  guards, at `1e6`). The mechanism behind the pooling loss: the `s` path
  re-enters the ladder, `δ_c` goes on for three `WrongInertia(73)` rungs,
  is withdrawn again by the gh#592 walk-back, and six `δ_c`-free factors
  that the gh#540 noise rule reports as *uncertified* `Singular` (70–71
  negatives, sub-floor pivots) climb `δ_x` 2.7e-4 → 8.7 — gh#981
  reinstates `δ_c` only on a *certified* `Singular`, so it never comes
  back. That is the plan's own stop ("the `s` path overflows `δ_c` on
  pooling instead of reinstating it — then this is R2e's problem").
  Suggested follow-up issue: **"PD solver: refinement's `S` exit keeps a
  step 1e6–1e37 × the RHS on most restoration-heavy solves; the capped
  residual ratio cannot tell it from a good one"** — numbers: 135 of 188
  corpus `S` exits above `1e6·‖rhs‖`; pooling iteration 4 at 2.2e14; the
  blanket refusal loses a base solve and a status (15 lines). One
  by-product worth the owner's look: lbfgs `square_flowsheet_resto`'s
  hidden cost falls 6008 → 51 total iterations at `κ = 1e12` (same
  `square-lm` verdict; its base solve ends `RestorationFailed`/22 instead
  of running to the 3000 cap).
- **R2e certify a sub-floor `δ_c`-free factor after the gh#592
  walk-back.** Not recoverable this way (measured on `3e186634`). The
  iteration-4 factor the item was written against has `min_piv` 3.98e-14
  against the `150·ε` = 3.33e-14 floor (1.19× above) — the item's own
  refutation clause. Corpus, both legs: 63 walk-backs on 8 fixture-legs
  (39 on exact pooling); first `δ_c`-free factor `WrongInertia` 21 (all
  above the floor), `Success` 14 (8 sub-floor, all exact pooling),
  certified `Singular` 9, uncertified `Singular` 19 (all sub-floor — the
  gh#540 noise rule only speaks below it). As specified the rule fires
  only on pooling and sends its base solve 494 Acceptable → 3000
  `MaximumIterationsExceeded` (reported `SolveSucceeded`/109 →
  MaxIter/3000, tot 4406); widened it is gh#981's rejected candidate
  (pooling base 141 Acceptable −3261.36, reported 491; gh#884's
  reproducer dual 1.3e-10 → 7.0e-5). No follow-up issue for the lever:
  17 of pooling's 24 accepted `δ_c`-free factors are sub-floor, so a
  sub-floor pivot is not the marker of the bad one.
- **R2b's result is a `mu_strategy_fallback` rescue, and the reporting
  hides it** (found in the R2e iteration). On `3e186634` pooling exact's
  default run is two attempts: base 494 it, `Solved_To_Acceptable_Level`
  at −4336.990888 (unscaled dual inf 5.3e6), 22 resto calls; then the
  pounce#748 fallback re-solves and its `Solve_Succeeded`/109/−4391.826
  is promoted. The JSON report (`iteration_count`, `restoration_calls`,
  no fallback field), hence the sweep line, `issue_592`'s `< 500` pin and
  R2b's own `≤ 300` pin all read the fallback's numbers. 0.13.0 (`p-head`,
  `0a299151`) and v0.12.0 run pooling as one attempt (492 / 162; checked on
  the console), and the pre-R2b tip's sweep line is `p-head`'s. With
  `mu_strategy_fallback=no` the tip reports `SolvedToAcceptableLevel` 494.
  Suggested follow-up issues: **"pooling_rt2stp: R2b's split escalation
  budget turns the base solve Optimal/492 into Acceptable/494; the 109
  is mu_strategy_fallback's"** (numbers: 492 → 494, Optimal → Acceptable,
  total 603 iterations / 25 resto calls vs 492 / 24) — R2b's census of the
  one changed decision stands, its outcome reading does not; and
  **"solve report / sweep: record a promoted mu_strategy_fallback
  attempt"** (the sweep's 2nd column covers the second-opinion ladder
  only; a promoted fallback leaves no trace, which is how this was
  misread, the gh#850 lesson one mechanism over).
  **Resolved in C1 (2026-10-22):** the second issue is implemented
  (`statistics.mu_strategy_fallback`, the sweep's `mu=` column); the first
  is moot — R2b's code is reverted, pooling exact is 0.13.0's single
  `Solve_Succeeded` attempt at 492 again. The R2 family's target (pooling
  ≤ 300 on the first attempt) remains unmet with every member terminal;
  a further lever would need a new item, measured on the first attempt.
- **R4 MLSL default `gamma` (circle_parabola_t0p5 mlsl).** No default
  recovers it without undoing #993: over seeds 0–9 circle_parabola finds
  both minima on 5/10 seeds at gamma 0.4–0.75 and 7/10 at 1.0 and 2.0,
  while camel's mean found falls 5.8 (0.5) → 4.0 (1.0) → 2.0 (2.0) and
  haverly's 2.0 → 1.5 → 1.0 — the #993/gh#989 failure (sample cap before
  the solves launch) coming back. Even v0.12.0's gamma 2 misses circle's
  second minimum on 3 of 10 seeds, so the seed-0 line overstated the loss
  (7/10 → 5/10, not 2 → 1). On every failing seed the stop is `patience 8`
  (status `converged`), and the strikes are mostly **locally infeasible
  launches**, not re-finds of minimum 1: at gamma 0.5, seeds 1/2/4 are
  7 infeasible + 1 re-find, seed 6 is 5 + 3, only seed 0 is 2 + 6. Suggested
  follow-up issue: **"find-minima mlsl: circle_parabola_t0p5 stops at
  `patience 8` on locally infeasible launches, finding 1 of 2 minima on
  5/10 seeds with 21 of 30 solves unused"** — whether an infeasible
  launch should be a full strike on a constrained model is the question,
  not a known defect. Numbers: 5/10 seeds stop at
  found 1, 9 solves; 7 of 8 strikes infeasible on 3 of those 5; 0 of 10
  stops are budget or sample cap.
- **R4b MLSL-aware re-find weighting (circle_parabola_t0p5 mlsl).** Its
  gate is unmet on a direct trace count: re-finds are 6, 1, 1, 1, 3 of the
  eight strikes on the five seeds that stop at found 1; the other 28 are
  launches ending locally infeasible, all at (0, −1). The stops re-finds do
  dominate are haverly's (8 of 8 on all 10 seeds, with its 2 minima
  already found), so a re-find discount spends budget exactly where the
  search is complete: replayed at half weight, haverly mean solves 10.2 →
  18.2. The route the data points to is **R4c** (§3). Suggested follow-up
  issue if R4c also fails: **"find-minima mlsl: eight hits of one locally
  infeasible point count as eight patience strikes — circle_parabola_t0p5
  finds 1 of 2 minima on 5/10 seeds at gamma 0.5"** — numbers: 28 of 40
  strikes on those seeds are that one point; the second minimum is reached
  at solve 10–17 of 30 on all five when patience does not stop the search;
  0 of the 40 non-circle grid runs have an infeasible launch.
- **Anything that requires the benchmark corpus** (B1's benefit, B2, B3) is
  blocked on this machine: `POUNCE_BENCH_DATA` is unset and no
  `~/projects/pounce-bench-data` exists. The code half of B1 is still in
  scope.

## 6. Item logs

Append one subsection per iteration: `### <id> — <date> — <status>` with
the before/after table, the sweep diff and its explanations, tests run,
tests not run, and the commit SHA.

### Baseline — 2026-10-22 — measured on x86_64 at `0a299151` vs `v0.12.0`

| fixture (leg) | `p-base` v0.12.0 | `p-head` 0a299151 |
|---|---|---|
| pooling_rt2stp (exact) | nlp SolveSucceeded 162 it q=4 −3273.954992, 3 resto calls | nlp SolveSucceeded 492 it q=3 −3273.95499, 24 resto calls / 106 resto it |
| pooling_rt2stp (lbfgs) | nlp ErrorInStepComputation 716 it q=12 −4391.82 | nlp SolveSucceeded 82 it q=0 −3273.954992 |
| dallass (exact) | nlp SolveSucceeded 41 it q=6 −32393.22574, 2 resto calls | nlp SolveSucceeded 123 it q=19, 12 resto calls / 53 resto it |
| dallass (lbfgs) | nlp ErrorInStepComputation 213 it q=7 | nlp ErrorInStepComputation 146 it q=4 (ladder kept, tot 292) |
| deb7 (exact) | nlp SolveSucceeded 143 it 97.55993795 | nlp SolveSucceeded 114 it 97.55993795 |
| deb7 (lbfgs) | nlp ErrorInStepComputation 1010 it | nlp RestorationFailed 1206 it (ladder kept, tot 1738) |
| scaled_feasible_a (exact / lbfgs) | nlp SolveSucceeded 22 / 19 it, obj 1.1e-8 / 2.8e-7 | cvx-qp SolveSucceeded 200 / 200 it, obj 0, kkt 5e-19 |
| circle_parabola_t0p5 mlsl seed 0 | found 2 in 11 solves (gamma 2) | found 1 in 9 solves (gamma 0.5), status converged |

### R1 — 2026-10-22 — not recoverable this way (hypothesis refuted)

Measured on x86_64 (4 CPUs, AVX2+FMA) with `p-prev` = release `pounce` built
from `75e410d2` in a separate worktree with its own `CARGO_TARGET_DIR`.

**Reproduced (before any change).**

| run (`p-prev`) | engine | status | it | objective | final kkt |
|---|---|---|---|---|---|
| exact (default) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| lbfgs (`hessian_approximation=limited-memory`) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| `bound_relax_factor=0` (and `+ solver_selection=qp-ipm`) | cvx-qp | SolveSucceeded | 200 | 0.0 | 5.0e-19 |
| `max_iter=400` | cvx-qp | SolveSucceeded | 230 | 0.0 | 0.0 |
| `qp_equilibrate=no` / `qp_hsde=no` | nlp | SolveSucceeded | 22 | 1.1e-8 | 9.0e-9 |
| `scaled_feasible_b` default | cvx-qp | SolveSucceeded | 36 | 0.0 | — |

The `iterations[]` trace matches the plan (inf_pr 3.9e-5 at it 13, 5.000e-6
from it 20 to ~125, alpha 1.0 throughout, blow-up to 4.8e9 around 130–150,
recovery by 153, then 5.7e-7 ↔ 2.8e-6 to the cap).

**Instrumented (temporary `eprintln!` in `hsde.rs`, reverted; not
committed).** Per iteration: `pres` split into `ρ_y`/`ρ_z`, `dres`, `gap`,
`comp_max`, `true_comp`, the stop flags, `base_reg`/`reg_eff`/`δ_c`/`δ_w`,
and `true_kkt_error(prob, cone, x, y, z, τ)`. Both HSDE solves of the call
were traced (the first, unequilibrated one also runs the full 200).

| retry it | pres | dres | gap | comp_max | true_comp | `true_kkt_error` | mu | τ |
|---|---|---|---|---|---|---|---|---|
| 13 | 3.9e-5 | 2.9e-2 | 7.0e8 | 8.8e7 | 8.8e7 | 8.8e7 | 4.2e-13 | 1.0e-10 |
| 20 | 5.0e-6 | 5.4e-5 | 6.6e2 | 1.0e2 | 2.6e5 | 2.6e5 | 5.0e-19 | 9.7e-11 |
| 60 | 5.0e-6 | 3.2e-5 | 1.7e-2 | 8.3e-37 | 1.6e5 | 1.6e5 | 7.9e-57 | 9.8e-11 |
| 120 | 5.0e-6 | 2.0e-6 | 1.5e-2 | 3.5e-97 | 9.8e3 | 9.8e3 | 3.2e-117 | 9.8e-11 |
| 125 | 5.0e-6 | 4.6e-9 | 5.1e-4 | 7.2e-101 | 2.3e1 | 2.3e1 | 4.6e-121 | 9.8e-11 |
| 140 | 4.8e9 | 1.2e-7 | 5.6e-2 | 5.6e-2 | 8.3 | 8.3 | 5.0e-23 | 9.8e-11 |
| 160 | 5.7e-7 | 1.7e-8 | 5.8e-4 | 1.5e-4 | 0 | 1.5e-4 | 7.9e-25 | 9.8e-11 |
| 165 | 1.1e-6 | 1.3e-10 | 1.3e-4 | 8.6e-9 | 0 | **8.6e-9** (first < tol) | 3.5e-29 | 9.8e-11 |
| 199 | 8.5e-6 | 7.8e-15 | 3.2e-5 | 9.1e-21 | 0 | 9.1e-15 | 8.5e-42 | 9.8e-11 |

- `m_eq = 0`, `m_ineq = 10`, `n = 4`: `ρ_y` is empty, so there is no
  equality-block `δ·‖ŷ‖` floor; `δ_c` sits at 1e-10, `δ_w` at 1e-10,
  `base_reg = reg_eff = 1e-14`, zero inertia retries, on every iteration.
- `cap_p = 7.1e-5` (`64·ε·scale_p`, `scale_p ≈ 5e9`): `pres = 5e-6` already
  satisfies it. What blocks a stop through it 20–125 is `comp_ok`
  (`true_comp` 2.6e5 → 23 against `tol_true_comp = 1e-6`) — the point is
  genuinely not optimal, with internal `s·z` at 1e-37…1e-101 while the true
  row slack times `ẑ` is 1e5. `true_kkt_error` tracks `true_comp`.
- After the blow-up the loop recovers and converges in ~15 iterations
  (151 → 165); from 165 on `comp_ok` holds and `true_kkt_error < tol`, and
  the stop is blocked only by `pres < tol` (unreachable, ~4 ulp) and by
  `gap_rel` (the relative arm has no gap excuse).
- First, unequilibrated HSDE solve: τ creeps linearly 1.6e-9 → 7.6e-8 over
  it 19–199, `true_comp` falls ∝ 1/τ (7.2 → 0.149), `mu` ~1e-37; ends
  `IterationLimit` at `true_kkt_error 0.149`, so the gh#293 retry runs.

**Verdict.** Both planned shapes are refuted by the measurement: (1) a
stall-gated in-loop `true_kkt_error` certificate cannot fire before it 165,
because the plateau iterate is not optimal; it would buy 200 → ~166 at best,
only by virtue of the chaotic blow-up-and-recovery, and against `ipm.rs`'s
reasoning for judging only after the loop; (2) widening `cap_p` changes
nothing, since `pres ≤ cap_p` already holds. The acceptance target (≤ 60 on
both legs) is out of reach for either. The real cost is the 20–125 crawl —
`mu` dividing by ~8 per full step while the true complementarity falls
linearly — which is an HSDE centering / cold-start problem whose fix would
move trajectories across the whole convex arm and needs `benchmarks/qp`
(not on this machine). Follow-up issue and numbers in §5. Not touched, per
the plan: `equilibrated_retry_budget`, the gh#535 decline rule, the gh#414 /
gh#984 guards. The "3596 iterations" comments in `main.rs` were not
re-measured and are left as they are.

**Tests / sweep.** Docs-only commit: no code changed, so no regression test,
no mutation check, no fixture sweep (nothing can move) and no `cargo test`;
`cargo fmt --all -- --check` run (clean). Not run: `benchmarks/qp` (no
corpus: `POUNCE_BENCH_DATA` unset, `bench_data_root` finds none), Python
tests (no code change).


### R2 — 2026-10-22 — not recoverable this way (hypothesis refuted)

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` built
from the clean tip `463eeeae` (code-identical to `75e410d2`) with its own
`CARGO_TARGET_DIR`; the cached binary from the R1 iteration was not reused
because a rebuild from the clean tree was not byte-identical (R1's
reverted instrumentation had touched `pounce-convex`). `p-base` = v0.12.0.

**Reproduced (before any change).**

| run | engine | status | it | objective | resto calls / it | q | lg(rg) ≥ 19 / ≥ 10 |
|---|---|---|---|---|---|---|---|
| `p-prev` exact | nlp | SolveSucceeded | 492 | −3273.95499 | 24 / 106 | 3 | 40 / 171 |
| `p-prev` lbfgs | nlp | SolveSucceeded | 82 | −3273.954992 | — | 0 | — |
| `p-base` exact | nlp | SolveSucceeded | 162 | −3273.954992 | 3 / 9 | 4 | 0 / 0 |
| `p-prev` exact, `perturb_delta_c_max_rungs=0` | nlp | SolveSucceeded | 115 (stalls, dual inf 1.8e8) + re-solve 205 | −3273.95500 | 0 | — | 2 / 50 |
| `p-base` exact, `perturb_delta_c_max_rungs=0` | nlp | SolveSucceeded | 116 | — | 2 / 7 | — | 0 / 0 |

No second opinion on the default exact run. `p-base`'s largest `δ_x` is
3.6e9 (40 iterations ≥ 1e4); `p-prev`'s is 9.7e19 (183 iteration records
≥ 1e10, restoration included).

**Instrumented (temporary probe, reverted; not committed).** In
`StdAugSystemSolver::solve`, after every factorization: the assembled KKT
densified and its eigenvalues taken (faer), the (1,1) block `W + Σ_x + δ_x I`
— its λ_min and Gershgorin margin — the slack diagonal minimum, `δ_x`,
`δ_c`, FERAL's reported negative count against the required 72, and the
max-norm of the primal and dual parts of the solution; in
`PdFullSpaceSolver` the iteration at each `consider_new_system`, the
accepted deltas, and the iterative-refinement residual ratios. The same
probe was applied to a v0.12.0 worktree (built with its own target dir) so
both versions were measured identically. Restoration ladders run in the
same 46-variable space (`AugRestoSystemSolver` condenses), so the counts
below include them. The probe does not perturb the run: 492 / 162
reproduced under it.

All 1310 ladder factorizations on `p-prev` (re-solves after a quality
escalation excluded), by status, `δ_c`, and whether the (1,1) block is PD
(λ_min > 0) / Gershgorin-dominant:

| status | `δ_c > 0` | (1,1) PD | Gershgorin | count |
|---|---|---|---|---|
| WrongInertia | yes | no | no | **146** |
| WrongInertia | yes | yes | — | **0** |
| WrongInertia | no | yes | 7 of 9 | 9 |
| WrongInertia | no | no | no | 138 |
| Singular | yes | yes | yes | 5 |
| Singular | yes | no | no | 202 |
| Singular | no | yes | 55 of 57 | 57 |
| Singular | no | no | no | 237 |
| Success | (all) | | | 516 |

- The planned arm (`WrongInertia`, `δ_c > 0` or `m_c = 0`, block provably
  PD) has **zero** candidates: `m_c = 40`, and every `WrongInertia` with
  `δ_c > 0` has an indefinite (1,1) block. The 9 PD `WrongInertia` factors
  have `δ_c = 0`, where Sylvester proves nothing without a rank argument for
  `J`. The 5 provable cases are `Singular` (excluded by the plan: gh#981
  territory).
- Of 516 ladders, 170 end at `δ_x ≥ 1e10` (1 overflows). In those, the
  first-rung λ_min(W+Σ_x) is negative in 165 (median −1.5e14, min −1.2e19).
  At acceptance `δ_x / −λ_min` has quantiles 0.00016 / 1.4 / 2.4 / **5.7** /
  2.1e5 / 3.7e8 (min/10/25/50/75/90 %): the median ladder stops about one
  rung past the (1,1) PD threshold. 88 of 169 accept at their first
  non-zero rung (the start is `δ_x_last/3`), and the over-damped tail
  (41 + 14 ladders > 100× the threshold) comes from that inherited start,
  not from climbing.
- `W` itself: λ_min(W+Σ_x) at each system, quantiles min/5/10/25/50 %:
  `p-prev` −1.2e19 / −1.2e19 / −3.4e15 / −6.0e12 / −1.1e6 (151 of 516 below
  −1e10, 347 below −1e4); `p-base` −2.2e9 / −2.2e9 / −2.2e9 / −8.2e5 /
  −6.4e5 (0 of 162 below −1e10, 132 below −1e4).

**Where `W` blows up (iteration 4 → 5).**

| | `p-base` v0.12.0 | `p-prev` 0.13.0 |
|---|---|---|
| it-4 ladder | WI(73), WI, WI with `δ_c` → walk-back `w` → Success at `δ_x` 3.3e-3, `δ_c` 0 | WI(**71** < 72) → too-few-negatives branch **escalates quality** (`q`) → Singular → WI(73) ×3 with `δ_c` → walk-back → Success at `δ_x` 1e-4, `δ_c` 0 |
| min eig of the accepted factor | 1.6e-17 | 2.0e-17 |
| dual part of the solve | 3.5e16 | 6.4e16 |
| refinement ratios | 9.0e-7, 7.5e-9, 7.5e-9 → stall, `augsys_improved` false → escalate, refactor | 9.1e-6, 2.2e-8, 1.2e-8, 1.6e-8 → stall, `augsys_improved` already true → `S` exit (< `residual_ratio_singular`), step kept |
| dual part after refinement | **3.0e6** (ratio 1e-16) | **6.4e16** |
| λ_min(W+Σ_x) at iteration 5 | −7.4e2 | **−4.1e13** |

So both versions withdraw `δ_c` and accept a numerically singular factor
at iteration 4; v0.12.0 is rescued by the refinement's quality escalation,
0.13.0 is not, because FERAL 0.21's count at that iteration's unperturbed
factor (71 instead of 73) spent the escalation in the inertia branch. The
`2599c2be` reset is not involved.

**Counterfactual (probe build only, env-gated, not committed).** Allowing
the refinement branch to escalate even when `augsys_improved` is set:
`pooling_rt2stp` exact 492 → **109**, −4391.82601 (the other certified
optimum, accepted by `issue_592`), 3 restoration calls, q = 4; `dallass`
exact 123 (unchanged), `deb7` exact 114 (unchanged). Full sweep, `p-prev`
vs counterfactual, 19 lines move:

```
exact  infeasible_square_scaled_1em4  q=3 -> q=12
exact  pooling_rt2stp                 it=492 q=3 obj=-3273.95499 -> it=109 q=4 obj=-4391.82601
exact  square_flowsheet_resto         it=95 q=4 -> it=76 q=13
lbfgs  circle_parabola_t0p5           it=77 q=4 2nd=feral_scaling=mc64@.../48,tot=125 obj=8.155059653
                                   -> it=51 q=24 2nd=start_point_perturbation=1e-2@.../48,tot=227 obj=10.42794559
lbfgs  circle_parabola_t1p5           q=10 -> q=12
lbfgs  cresc4                         q=0 2nd=feral_increase_quality=no@.../3000,tot=3200 -> q=24 2nd=-
lbfgs  degenerate_start_infeasible    tot=74 -> tot=77
lbfgs  eigena2                        it=239 q=3 tot=478 -> it=164 q=11 tot=328 (ErrorInStep both)
lbfgs  eigmaxa                        q=2 -> q=12
lbfgs  infeasible_square_scaled_1em4  q=1 -> q=12
lbfgs  issue880_sigma_uncertified     it=67 q=8 -> it=59 q=12
lbfgs  issue981_water_main            q=1 -> q=12
lbfgs  issue_508_infeasible_gap_1em2  q=5 tot=284 -> q=12 tot=290
lbfgs  issue_508_infeasible_gap_1em4  q=1 tot=346 -> q=12 tot=349
lbfgs  scale_audit_retry_meyer3       it=112 q=5 -> it=152 q=12
```

No status or engine moves, but the blanket rule runs escalations to the
cap and changes an lbfgs objective, so it is a measurement of where the
lever is, not a candidate fix. The narrower shapes are in §5.

**Verdict.** The plan's hypothesis — FERAL miscounting the inertia of a
provably quasi-definite matrix, so that the ladder climbs past a rung where
Sylvester already guarantees it — is false on this fixture: the arm it
describes has zero candidates in 1310 factorizations. The climb is the
correct response to a Hessian of the Lagrangian that is indefinite by up to
1.2e19, and that comes from multipliers blown up by one kept step at
iteration 4. Per §4 step 4 no code change was made; the follow-up issue
(title + numbers) is in §5. The `issue_592` `< 500` pin is untouched (F1
stays open with a margin of 8). The unit tests named in the plan were not
needed — nothing they guard changed.

**Effect on R3.** R2 changed no code, so R3's re-baseline is the baseline:
`dallass` exact 123 it / 12 restoration calls / q = 19, lbfgs
ErrorInStepComputation 146 (tot 292), both re-measured on `p-prev` in this
iteration's sweep. R3's dependency is resolved (R2 closed as not
recoverable); R3 proceeds to its step 2 (the `pounce-nl` peephole).

**Tests / sweep.** Docs-only commit: no code changed, so no regression
test, no mutation check, no `cargo test`; `cargo fmt --all -- --check` run
(clean). `scripts/sweep-fixtures.sh` was run on `p-prev` (234 fixture-legs;
engine split nlp 127, cvx-qp 92, cvx-qcqp 14, square-lm 1) only as the
baseline for the counterfactual above; there is no new binary to diff
against. Not run: `benchmarks/vanderbei` (no corpus: `POUNCE_BENCH_DATA`
unset), Python tests (no code change), arm64 (not available).

### R3 — 2026-10-22 — done (option A)

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` rebuilt
from the tip `015a8fe4` in a separate worktree with its own
`CARGO_TARGET_DIR` (the cached binary was not reused).

**Reproduced (before any change).** R2 changed no code, so this is the
baseline, and it reproduces exactly:

| run (`p-prev`) | engine | status | it | objective | resto calls / it | q |
|---|---|---|---|---|---|---|
| exact | nlp | SolveSucceeded | 123 | −32393.225736 | 12 / 53 | 19 |
| lbfgs | nlp | ErrorInStepComputation | 146 (2nd opinion `mu_strategy=adaptive`, tot 292) | −32388.508 | 2 / 8 | 4 |

A temporary probe (an `eprintln!` at every `invoke_restoration_debugged`
and at the step-failure entry that sets `resto_entry_nonfinite_w`; reverted)
classifies the entries: on the exact leg **11 of 12** restoration entries
are non-finite-`W` entries (iterations 4, 19, 29, 36, 42, 47, 58, 77, 83,
99, 111); the 12th (iteration 106) is an ordinary line-search entry. The
lbfgs leg has no non-finite-`W` entry (it never evaluates a Hessian); its
two calls are at 136 and 140 and the base run fails at 146.

`dallass.nl` is the only fixture in `crates/pounce-cli/tests/fixtures` that
uses `abs` (`o15`) at all: 72 occurrences, every one directly under a
`pow` (`o5`), written as `o2 (o2 k (o5 (o15 vK) n2)) (o5 (o15 vK) n0.85)`
— i.e. `(k·|x|^2)·|x|^0.85`, so the two powers are **not** siblings: the
first sits one `Mul` level down beside the coefficient. A peephole that
matches only `Mul(Pow(Abs v, a), Pow(Abs v, b))` would not fire on this
model at all.

**Implementation (option A).** `merge_abs_pow_product` in
`crates/pounce-nl/src/nl_reader.rs`, called from the `o2` arm of
`parse_opcode`: when a factor `|e|^p` of the left operand meets a factor
`|e|^q` of the right one — inspecting each operand and at most one `Mul`
level inside it, which is what `(k·|x|^2)·|x|^0.85` needs — the pair becomes
`|e|^(p+q)` and the other factors are kept, left-associated. Guards, each
because without it the rewrite would change a *value*, not only a
derivative: the base must be `abs(·)` (for a bare `e < 0`, `e^0.5·e^0.5` is
NaN but `e^1` is not); both exponents must be finite literals `> 0` (`|0|^-1·|0|^2`
is NaN, `|0|^1` is 0); `p + q` finite; the operands must be the same
expression (`same_operand`: same `Var`, same `Cse` by pointer, bit-identical
`Const`, or the same unary/binary operator over identical operands; anything
else declines). With no match the node is the `Mul(a, b)` it always was, so a
model without the pattern gets an identical tree. The rewrite sits on the
`.nl` path only; in-memory `Expr` builders (Python, `from_expressions`) are
untouched. No other code changed; the `5fc8c797` exemption is intact.

**Value / derivative preservation (measured, temporary probe in
`pounce-nl/tests/`, deleted).** `dallass.nl` vs an unmerged copy
(`tests/data/dallass_kink_unmerged.nl`, below) through `NlTnlp` at 200
perturbed points, 21 of them with some flows exactly 0: objective max
relative difference 9.4e-16; gradient 3.5e-15 relative at `x ≠ 0` and
bit-identical at `x = 0` (the unmerged tape's gradient already read 0 there);
Hessian 3.2e-15 relative at every point where the unmerged one is finite, with
an identical sparsity structure; the unmerged Hessian is non-finite at all 21
zero-flow points, the merged one at none. So the only non-rounding change is
`NaN → 0` in the Hessian at the kink.

**After.**

| run | engine | status | it | objective | resto calls / it | q |
|---|---|---|---|---|---|---|
| `p-base` v0.12.0 exact (baseline table) | nlp | SolveSucceeded | 41 | −32393.22574 | 2 / 14 | 6 |
| `p-prev` exact | nlp | SolveSucceeded | 123 | −32393.225736 | 12 / 53 | 19 |
| **new exact** | nlp | SolveSucceeded | **28** | −32393.225736 | **0** | 0 |
| `p-base` lbfgs | nlp | ErrorInStepComputation | 213 | — | — | 7 |
| `p-prev` lbfgs | nlp | ErrorInStepComputation | 146 (tot 292) | −32388.508 | 2 / 8 | 4 |
| **new lbfgs** | nlp | **SolveSucceeded** | 301 (no 2nd opinion) | −32393.225738 | 1 / 3 | 1 |

Exact: 29 Hessian evaluations (iterations + 1); the probe build shows no
restoration entry of any kind. The trajectory is identical to `p-prev`'s
through iteration 3 and departs at iteration 4, `p-prev`'s first
non-finite-`W` entry. Acceptance (`≤ 60` it, `≤ 3` restoration calls, the
optimum) met with margin, and below v0.12.0's 41.

Lbfgs: never evaluates a Hessian, so the merge reaches it only through
rounding-level differences in `f` and `∇f` (≤ 3.5e-15 relative, above). The
iteration tables agree through iteration 1 and first differ at iteration 2
(`inf_pr` 3.1e-13 vs 9.3e-7 after the same full step, objective in the 8th
digit) — rounding amplified by the step solve on that iterate — after which
the run solves instead of failing. Not an acceptance criterion; "must not get
worse" holds (failing → solved, 292 → 301 total iterations).

**Fixture sweep** (`scripts/sweep-fixtures.sh`, 234 fixture-legs, both legs,
`p-prev` vs new), the full diff:

```
19c19
< exact  dallass  nlp  SolveSucceeded          it=123  q=19  2nd=-                obj=-32393.22574
---
> exact  dallass  nlp  SolveSucceeded          it=28   q=0   2nd=-                obj=-32393.22574
136c136
< lbfgs  dallass  nlp  ErrorInStepComputation  it=146  q=4   2nd=kept(1),tot=292  obj=-32388.50847
---
> lbfgs  dallass  nlp  SolveSucceeded          it=301  q=1   2nd=-                obj=-32393.22574
```

- exact `dallass`: the 11 non-finite-`W` restoration entries are gone (the
  fix's purpose); `q` 19 → 0 because those entries' escalations went with them.
- lbfgs `dallass`: rounding-level change in `f`/`∇f` from the merged form
  (above), trajectory departs at iteration 2; failing → solved at the optimum.
- Every other line is unchanged, as it must be: `dallass.nl` is the only
  fixture that contains `abs` (`o15`) at all, so no other tree can match.
  Engine split unchanged both sides: nlp 127, cvx-qp 92, cvx-qcqp 14,
  square-lm 1. No status loss, no engine move.

**Tests.** New: three `pounce-nl` unit tests in `nl_reader.rs`
(`abs_pow_product_is_merged_and_its_derivatives_are_finite_at_the_kink` —
tree shape, `f`/`∇f`/`∇²f` against `3·|x|^2.85` at eight points incl.
negative ones and `0`, exactly `(0, 0, 0)` at the kink, Hessian vs a central
difference of the gradient at `±1e-3`;
`abs_pow_near_misses_build_the_unmerged_tree` — different operands,
negative and zero powers, bare base, variable exponent all build the exact
`Mul(a, b)` tree; `abs_pow_merge_keeps_the_other_factors` — a pair one `Mul`
level down on either side, an integer merged power, values unchanged), and
`crates/pounce-cli/tests/issue_1042_dallass_abs_pow_merge.rs` (default exact
run: solved, optimum, no second opinion, `≤ 60` it, `≤ 3` restoration calls).

`nonfinite_hessian_restoration_strike.rs` no longer exercised the
exemption on `dallass.nl` (no kink left), so per the plan it gets a model that
still reaches a non-finite Hessian: `tests/data/dallass_kink_unmerged.nl`,
`dallass.nl` with each `|x|^0.85` operand written `x + 0` (36 sites), which
evaluates identically but declines the merge. It reproduces `p-prev`'s run
bit for bit (123 it, 12 calls / 53 it, `q` 19, objective
−32393.225736201894). It is in `tests/data/`, not `tests/fixtures/`, so the
sweep does not double-count it. A `sqrt(x)`-at-0 model was not needed.

Mutation checks: (1) `o2` back to a plain `Mul` — the first and third unit
tests fail on the tree shape, and with the shape assertion removed the first
fails numerically (`x=0: 0 0 NaN`, i.e. Hessian NaN); the R3 CLI pin fails
(`dallass took 123 iterations`); (2) the `5fc8c797` exemption reverted
(`if true || !entry_nonfinite_w`) — the strike test fails (`dallass did not
solve; … counted as a gh#274 strike`) on the unmerged copy, while the R3 pin
still passes (the merged model never enters that branch). Both restored.

Run: `cargo fmt --all -- --check` (clean); `cargo clippy --workspace
--exclude pounce-hsl --all-targets -- -D clippy::correctness -D
clippy::suspicious` (clean; one pre-existing unused-import warning in
`pounce-restoration`, untouched); `cargo test -p pounce-nl -p pounce-cli
--no-fail-fast` (170 test binaries, 1160 passed, 0 failed, 1 ignored); the sweep above. Not run: `dallasl` /
`dallasm` and `benchmarks/vanderbei` (no corpus on this machine:
`POUNCE_BENCH_DATA` unset); arm64 (not available — the issue's arm64
41 → 65 is not re-measured); Python tests (the Python frontend builds `Expr`
in memory and never reaches the `.nl` parser this changes).

**Effect on other items.** None of R4 / B1 depend on R3. F1 (the
`issue_592` margin) is untouched: `pooling_rt2stp` has no `abs`, and its
sweep line did not move.

### R2d — 2026-10-22 — not recoverable here: not a count defect (measured)

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` rebuilt
from the tip `d3476a97` in a separate worktree with its own
`CARGO_TARGET_DIR` (no cached binary reused). The probe builds (below) are
`d3476a97` and `v0.12.0` = `a31dab9c` with a temporary env-gated dump in
`FeralSolverInterface::factor`, each in its own worktree and target dir;
neither is committed. `POUNCE_BENCH_DATA` unset, `bench_data_root`
resolves nothing (checked once).

**Reproduced (before anything else).** `p-prev`, both legs, the M1
fixtures — identical to R3's after-table, so the baseline holds:

| fixture (leg) | engine | status | it | q | resto calls / it | objective |
|---|---|---|---|---|---|---|
| pooling_rt2stp (exact) | nlp | SolveSucceeded | **492** | 3 | 24 / 106 | −3273.95499 |
| pooling_rt2stp (lbfgs) | nlp | SolveSucceeded | 82 | 0 | 0 | −3273.954992 |
| dallass (exact) | nlp | SolveSucceeded | 28 | 0 | 0 | −32393.225736 |
| dallass (lbfgs) | nlp | SolveSucceeded | 301 | 1 | 1 / 3 | −32393.225738 |
| deb7 (exact) | nlp | SolveSucceeded | 114 | 0 | 1 / 2 | 97.55993795 |
| deb7 (lbfgs) | nlp | RestorationFailed | 1206 | 0 | 4 / 343 | 99.629 |
| scaled_feasible_a (exact / lbfgs) | cvx-qp | SolveSucceeded | 200 / 200 | 0 | 0 | 0 |
| circle_parabola_t0p5 (lbfgs) | nlp | SolveSucceeded | 77 | 4 | 0 | 8.155059653 |
| scale_audit_retry_meyer3 (lbfgs) | nlp | SolveSucceeded | 112 | 5 | 0 | 3.8908e9 |

F1 margin: 500 − 492 = 8 (unchanged).

**The iteration-4 factor.** The probe (`POUNCE_R2D_DUMP=<dir>`) writes
every factored KKT as Matrix Market plus FERAL's symbolic permutation and
the scaling vector of the factor, and logs inertia, `min_piv`, `max_piv`,
`n_tiny`, `n_threshold_fail_accepts`, pivot threshold and quality level;
with `RUST_LOG=pounce::linsol=debug POUNCE_DBG_PERT=1` the `pounce-feral`
verdicts and `[PERT] iter=` lines interleave. The run is unperturbed by it
(492). The iteration-4 ladder is factors 10–15 of the run:

| factor | δ | FERAL 0.21 inertia (+, −, 0) | `min_piv` | pounce-feral verdict |
|---|---|---|---|---|
| 10 | δ_x = δ_c = 0 | (79, **71**, 0) | **5.95e-14** | `WrongInertia` ("inertia mismatch": `min_piv` ≥ floor 3.33e-14, so the noise rule is not consulted) → too-few branch escalates (pivtol 1e-6 → 3.16e-5, the `q`) |
| 11 | same matrix, refactored | (80, 70, 0) | 6.5e-15 | `Singular` (noise rule: deficit, flagged) → `δ_c` on |
| 12–14 | `δ_c` on, `δ_x` 0 / 1e-4 / 1e-2 | (77, 73, 0) ×3 | 1.8e-3 | `WrongInertia` (surplus, clean factor — gh#1003) |
| 15 | walk-back: `δ_x` 1e-4, `δ_c` 0 | (78, 72, 0) | 4.0e-14 | `Success` — R2's 6.4e16 step |

On `v0.12.0` the iteration-4 ladder's first factor (its factor 14, also at
`δ = 0`) reads (77, **73**, 0) with `min_piv` 1.9e-3 — a clean surplus.

**Three counts, both matrices** (`dev-notes/issue-1042-r2d/`: the two
dumps, the in-run permutation and scaling, and the analysis program — a
standalone crate, `cargo run --release --features dense -- <mtx> [perm]
[pivtol] [scal]`; edit its `feral = "=0.21.0"` to 0.20.0 / 0.18.0 for the
other versions). Exact inertia from dense symmetric eigendecomposition
(faer 0.24, cross-checked with LAPACK through numpy):

| | 0.13.0 iteration-4 matrix | v0.12.0 iteration-4 matrix |
|---|---|---|
| order / ‖K‖∞ | 150 / 5.30e7 | 150 / 5.36e7 |
| exact inertia (faer; LAPACK) | (79, **71**, 0); (79, 71, 0) | (77, **73**, 0); (77, 73, 0) |
| smallest \|λ\|, unscaled | faer **+2.2e-17, −3.8e-16**; LAPACK −1.6e-15, +1.9e-15 (signs disagree); next 1.9e-9 | 2.4e-9, −3.8e-9 |
| smallest \|λ\| of the in-run scaled matrix `DKD` | **1.6e-18, 1.1e-15**, next **7.7e-4** | — |
| # \|λ\| < 150·ε·‖K‖∞ (1.8e-6) / < 1e-10·‖K‖∞ | 16 / 24 | 16 / 24 |
| FERAL 0.21 / 0.20 / 0.18, fresh analysis, default (Auto, pivtol 1e-8) | 71 / 71 / 71 (min piv 3.0e-16) | 73 / 73 / 73 (min piv 7.8e-7) |
| same, pivtol 1e-6, 1e-2, 0.1 · 0.5 | 0.21: 71, 71, 71 · **72**; 0.20 and 0.18: 71 throughout | 73 throughout, all versions |
| same, Identity scaling (any pivtol) | (78, 71, **1**) all three versions | 73 |
| same, InfNorm (pivtol ≤ 0.1 · 0.5) | 71 · 0.21 (79, 70, **1**), 0.20 and 0.18 71 | 73 |
| **in-run ordering + scaling replayed**, pivtol 1e-6 | 0.21 (79, **71**, 0), `min_piv` **5.95e-14** — the in-run factor, exactly; 0.20 and 0.18 (80, **70**, 0), `min_piv` 6.0e-15 | — |
| in-run replay, smallest pivots | 0.21: 5.95e-14, 1.06e-11, then 8.5e-6; 0.20/0.18: 6.0e-15, 3.2e-14, then 8.5e-6 | — |

Readings:
- **Hypothesis (b) is refuted.** FERAL 0.21's 71 is the exact count of
  the matrix it was given; 0.20 and 0.18 agree at every setting but one.
  The "73 on v0.12.0" is a different matrix: the two runs factor a
  bit-identical KKT only for their first two factors (the dumps are
  byte-equal), the first back-solve differs at rounding level (FERAL
  0.18 vs 0.21), and the third factor already differs in 36 of 555
  entries (max relative 1.1e-10). v0.12.0 also built against **0.18.0**,
  not 0.20.
- **(a) holds for the matrix:** it has a numerical null space of
  dimension exactly 2 (two scaled eigenvalues at 1.6e-18 / 1.1e-15, then a
  11-order gap), so 70, 71, 72 are all rounding — even the two dense
  eigensolvers disagree on the signs. 72 is reached only if both null
  directions round negative. Such a matrix is `Singular`'s case (`δ_c`),
  and any quality escalation spent on it is wasted: no pivot rung changes
  the rank.
- **The guard that stopped the noise rule** is `min_piv < floor`:
  `sub_floor_count_is_noise` would have returned `true` twice over
  (deficit; `n_threshold_fail_accepts = 1`), but `min_piv` 5.95e-14 is
  1.8× the `150·ε` floor. The LDLᵀ pivot is ~4 orders above the null
  eigenvalue it carries, and 0.21 places it 10× higher than 0.20/0.18 do
  on the same ordering and scaling (6.0e-15, which *is* under the floor).
  Relative to the factor's own `max_piv` 2.7e4 both tiny pivots are at
  ε (2.2e-18, 4.0e-16). That is a pivot-magnitude difference in a
  rank-2-deficient factor — not a count error and not something FERAL's
  contract promises to keep fixed — so no upstream issue is filed.

**Workarounds in `pounce-feral`, measured (probe build, env-gated, not
committed)** — the plan allows one only if it is justified:

| variant | pooling exact | pooling lbfgs | dallass exact / lbfgs | deb7 exact / lbfgs |
|---|---|---|---|---|
| `p-prev` | 492, q 3, 24 resto | 82 | 28 / 301 solved | 114 / RestorationFailed 1206 |
| `feral_inertia_pivot_floor=6e-14` (just above this pivot) | **SolvedToAcceptableLevel 191, −3580.0075** (not an optimum), 9 resto | 82 | — | — |
| `feral_inertia_pivot_floor=1e-13` | 140, −3273.955, 4 resto | 82 | — | — |
| `feral_inertia_pivot_floor=1e-12` (the pre-gh#592 value) | 495, 19 resto | 82 | — | — |
| floor `150·ε·max(1, max_piv)` (scale-relative) | 471, −4391.826, 19 resto | 82 | 28 / **ErrorInStep 269** | **606** / 1206 |
| every deficit → `Singular`, floor ignored | **162**, −4391.826, 3 resto, q 1 | 82 | 28 / **ErrorInStep 269** | 114 / 1206 |

None is admissible. A fixed floor is chaotic in its value (a pin just
above this pivot, 6e-14, loses the verdict; 1e-12 is gh#592's own rejected value,
and widening it is the "weaken the fix" stop of §4). The scale-relative
floor does not help pooling and costs `deb7` 5×. The deficit rule — the
noise rule's own doc comment already argues "δ_c is the right remedy for
a deficit whether or not the count is noise" — recovers pooling to
v0.12.0's 162, but loses the lbfgs `dallass` solve that R3 won. It is a
measurement of where the lever is, and it says the same as R2's
counterfactual: the cheap lever is *not spending the escalation* on this
factor, which is R2b's consumer-side shape (its narrowing condition — the
refactor after the inertia-branch escalation came back `Singular` — is
met here, factor 11), or repairing the kept step afterwards (R2c).

**Verdict.** Case 3 of the item: the matrix is genuinely ambiguous, the
count is not a FERAL defect, and the noise rule's silence is the
consequence of a pivot-magnitude gate the measurement gives no principled
way to move. No code changed. R2c is next per §2; R2b inherits the
deficit-rule numbers above as an extra collateral guard (lbfgs `dallass`
must stay `SolveSucceeded`/301).

**Tests / sweep.** Docs-only commit (the plan, plus
`dev-notes/issue-1042-r2d/`: two 150×150 dumps, the in-run permutation and
scaling, and the standalone analysis crate, which is not a workspace
member). `cargo fmt --all -- --check` clean. No regression test, mutation
check, clippy run or fixture sweep: nothing compiled into pounce changed,
so nothing can move, and M1 is not triggered (no landed R2x code). Not
run: `benchmarks/vanderbei` (no corpus), arm64 (not available), Python
tests (no code change).

### R2c — 2026-10-22 — not recoverable this way (measured)

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` rebuilt
from the tip `9a9723f9` in a separate worktree with its own
`CARGO_TARGET_DIR`. The probe build is the same tip with a temporary,
env-gated `eprintln!` at the `quit_refinement` decision in
`PdFullSpaceSolver::solve` (`POUNCE_R2C_PROBE`: exit letter, iteration,
`‖rhs‖∞`, `‖res‖∞`, `‖resid‖∞`, ratio) and, for the counterfactual, the
item's rule behind `POUNCE_R2C_KAPPA=<κ>` (refuse `S` when `‖res‖∞ >
κ·‖rhs‖∞`, take `s`, info letter `b`); neither is committed. With neither
variable set the probe build's sweep diffs **empty** against `p-prev`.
`POUNCE_BENCH_DATA` unset, `bench_data_root` resolves nothing (checked
once).

**Reproduced (before anything else).** `p-prev`, the M1 fixtures, both
legs — identical to the R2d table, so the baseline holds:

| fixture (leg) | engine | status | it | q | resto calls / it | objective |
|---|---|---|---|---|---|---|
| pooling_rt2stp (exact) | nlp | SolveSucceeded | **492** | 3 | 24 / 106 | −3273.95499 |
| pooling_rt2stp (lbfgs) | nlp | SolveSucceeded | 82 | 0 | 0 | −3273.954992 |
| dallass (exact) | nlp | SolveSucceeded | 28 | 0 | 0 | −32393.22574 |
| dallass (lbfgs) | nlp | SolveSucceeded | 301 | 1 | 1 / 3 | −32393.22574 |
| deb7 (exact) | nlp | SolveSucceeded | 114 | 0 | 1 / 2 | 97.55993795 |
| deb7 (lbfgs) | nlp | RestorationFailed | 1206 (kept, tot 1738) | 0 | 4 / 343 | 99.62936037 |
| scaled_feasible_a (exact / lbfgs) | cvx-qp | SolveSucceeded | 200 / 200 | 0 | 0 | 0 |
| scaled_feasible_b (exact / lbfgs) | cvx-qp | SolveSucceeded | 36 / 36 | 0 | 0 | 0 |
| circle_parabola_t0p5 (lbfgs) | nlp | SolveSucceeded | 77 (`feral_scaling=mc64`, tot 125) | 4 | 0 | 8.155059653 |
| scale_audit_retry_meyer3 (lbfgs) | nlp | SolveSucceeded | 112 | 5 | 0 | 3890764353 |

F1 margin: 500 − 492 = 8 (unchanged). The iteration-4 exit reproduces as
the R2 log describes: `exit=S iter=4 ‖rhs‖ 2.908e2 ‖res‖ 6.380e16
‖resid‖ 4.634 ratio 1.593e-8`, i.e. `‖res‖/‖rhs‖` = **2.19e14**.

**Measure first: every refinement exit in the corpus** (234 fixture-legs,
both legs, second-opinion rungs and restoration sub-solves included —
they share the code path). 1024 exits: 480 `s`, **188 `S`** (on 20
fixture-legs, 115 of them on legs that end `SolveSucceeded`), 178 `q`, 178
second stalls on the same matrix (`pretend_singular_last_time`, no
letter). `‖res‖/‖rhs‖` at the 188 `S` exits, by decade:

| decade | ≤1e-1 | 1e0–1e4 | 1e5 | 1e6–1e9 | 1e10–1e13 | 1e14–1e17 | 1e18–1e21 | ≥ 1e22 |
|---|---|---|---|---|---|---|---|---|
| `S` exits | 24 | 24 | 5 | 33 | 38 | 27 | 25 | 12 |

| κ | `S` exits the rule would refuse | share | fixture-legs |
|---|---|---|---|
| 1e6 (`max_cond`) | 135 | 72% | 17 |
| 1e8 | 104 | 55% | 14 |
| 1e12 | 90 | 48% | 13 |
| 1e14 | 64 | 34% | — |
| 2.19e14 (just under pooling it 4) | 62 | 33% | — |
| 4.7e20 (the 90th percentile) | 19 | 10% | — |

`S` exits per fixture-leg: lbfgs `square_flowsheet_resto` 56, lbfgs
`issue880_sigma_uncertified` 17, exact `pooling_rt2stp` 16, lbfgs
`scale_audit_retry_meyer3` 15, exact `circle_parabola_t1p5` 13, lbfgs
`infeasible_square_scaled_1em4` 13, lbfgs `degenerate_start_infeasible`
10, lbfgs `eigena2` 8, exact `infeasible_square_scaled_1em4` 8, exact
`issue_508_infeasible_gap_1em4` 6, lbfgs `cresc4` 5, exact
`issue_508_infeasible_gap_1em2` 4, lbfgs `issue981_water_main` 4, lbfgs
`circle_parabola_t1p5` 3, lbfgs `issue_508_infeasible_gap_1em2` 3, exact /
lbfgs `circle_parabola_t0p5` 2 / 2, exact `square_flowsheet_resto`, lbfgs
`eigmaxa`, lbfgs `issue_508_infeasible_gap_1em4` 1 each. Pooling alone has
14 `S` exits above `1e12` besides iteration 4's.

So the hypothesis' premise — "such exits are rare", the pooling exit "far
in the tail" — is false. A step 1e6–1e37 times the RHS is what an `S`
exit *usually* looks like on the corpus's restoration-heavy and
infeasible models: once `‖res‖ > 1e6·‖rhs‖` the capped ratio no longer
reads step size at all, on every model, not only at pooling iteration 4.
The pooling exit ranks 62nd of 188. The item's stop — "the rule fires on
more than ~10% of the baseline corpus's `S` exits" — holds at every κ
that could reach the target exit; κ would have to be ≥ 4.7e20.

**Counterfactual (env-gated probe, not committed).** The rule as specified,
at the two candidate κ:

| run | pooling exact | it-4 exit | pooling lbfgs |
|---|---|---|---|
| `p-prev` | SolveSucceeded 492, q 3, 24 resto, −3273.95499 | `S` | 82 |
| κ = 1e6 | **base `ErrorInStepComputation`/795**, promoted by `mu_strategy=adaptive` 140, tot 935, −3273.954989 | `b` | 82 |
| κ = 1e12 | **base `ErrorInStepComputation`/173**, promoted by `mu_strategy=adaptive` 140, tot 313 | `b` | 82 |
| κ = 1e14 | SolveSucceeded 232, q 4, 16 resto, −3273.954992 | `b` | — |
| κ = 3e14 (does **not** fire at it 4) | SolveSucceeded 109, q 4, 3 resto, −4391.826 | `S` | — |

The last two rows show the pooling count is chaotic in κ and is not
governed by the iteration-4 decision: the best result comes from a κ that
leaves iteration 4 alone and refuses only later, larger exits.

What the `s` path does at iteration 4 (`POUNCE_DBG_PERT=1
RUST_LOG=debug`, κ = 1e6): the pretend-singular re-solve re-enters the
ladder, `δ_c` goes back on for three rungs (`WrongInertia`, 73 negatives,
`min_piv` 1.8e-3, `δ_x` 0 / 3.3e-5 / 2.7e-4), the gh#592 walk-back
withdraws it again, and six `δ_c`-free factors — each 70–71 negatives with
a sub-floor pivot, reported **uncertified** `Singular` by the gh#540 noise
rule — climb `δ_x` 2.7e-4 → **8.7** (×8 per rung). gh#981 reinstates `δ_c`
only on a *certified* `Singular`, so `δ_c` stays 0; iteration 5's step is
taken at `δ_x = 8.7`, and the run goes on to blow up anyway (`lg(rg)` 3.2 at
iteration 6, restoration at 13). The plan's stop — "the `s` path overflows
`δ_c` on pooling instead of reinstating it (then this is R2e's problem)" —
holds in the form "never reinstates it".

Full sweep, `p-prev` vs counterfactual. κ = 1e6, 15 lines:

```
< exact circle_parabola_t1p5 nlp InfeasibleProblemDetected it=603 q=0 2nd=kept(3),tot=4190 obj=10
> exact circle_parabola_t1p5 nlp MaximumIterationsExceeded it=3000 q=12 2nd=kept(1),tot=6000 obj=10.556119
< exact infeasible_square_scaled_1em4 nlp InfeasibleProblemDetected it=17 q=3 2nd=kept(4),tot=74 obj=0.1250000024
> exact infeasible_square_scaled_1em4 nlp InfeasibleProblemDetected it=14 q=1 2nd=kept(4),tot=65 obj=0.1250000024
< exact issue_508_infeasible_gap_1em2 nlp InfeasibleProblemDetected it=114 q=0 2nd=kept(3),tot=290 obj=25
> exact issue_508_infeasible_gap_1em2 nlp InfeasibleProblemDetected it=640 q=0 2nd=kept(3),tot=1652 obj=25
< exact issue_508_infeasible_gap_1em4 nlp InfeasibleProblemDetected it=441 q=0 2nd=kept(3),tot=982 obj=25
> exact issue_508_infeasible_gap_1em4 nlp InfeasibleProblemDetected it=384 q=0 2nd=kept(3),tot=1136 obj=25
< exact pooling_rt2stp nlp SolveSucceeded it=492 q=3 2nd=- obj=-3273.95499
> exact pooling_rt2stp nlp SolveSucceeded it=140 q=3 2nd=mu_strategy=adaptive@Error_In_Step_Computation/795,tot=935 obj=-3273.954989
< exact square_flowsheet_resto nlp SolveSucceeded it=95 q=4 2nd=- obj=0
> exact square_flowsheet_resto nlp SolveSucceeded it=57 q=1 2nd=- obj=0
< lbfgs circle_parabola_t0p5 nlp SolveSucceeded it=77 q=4 2nd=feral_scaling=mc64@Infeasible_Problem_Detected/48,tot=125 obj=8.155059653
> lbfgs circle_parabola_t0p5 nlp SolveSucceeded it=106 q=9 2nd=feral_scaling=mc64@Infeasible_Problem_Detected/48,tot=154 obj=10.42794559
< lbfgs circle_parabola_t1p5 nlp InfeasibleProblemDetected it=66 q=10 2nd=kept(4),tot=272 obj=1.018446456
> lbfgs circle_parabola_t1p5 nlp InfeasibleProblemDetected it=66 q=10 2nd=kept(4),tot=330 obj=1.018446456
< lbfgs eigena2 nlp ErrorInStepComputation it=239 q=3 2nd=kept(1),tot=478 obj=82.50000002
> lbfgs eigena2 nlp ErrorInStepComputation it=236 q=5 2nd=kept(1),tot=472 obj=82.5
< lbfgs infeasible_square_scaled_1em4 nlp InfeasibleProblemDetected it=24 q=1 2nd=kept(4),tot=114 obj=0.1250000027
> lbfgs infeasible_square_scaled_1em4 nlp InfeasibleProblemDetected it=19 q=1 2nd=kept(4),tot=95 obj=0.1679925196
< lbfgs issue880_sigma_uncertified nlp SolveSucceeded it=67 q=8 2nd=- obj=-2.205e+12
> lbfgs issue880_sigma_uncertified nlp SolveSucceeded it=32 q=3 2nd=- obj=-2.205e+12
< lbfgs issue_508_infeasible_gap_1em2 nlp InfeasibleProblemDetected it=65 q=5 2nd=kept(4),tot=284 obj=25
> lbfgs issue_508_infeasible_gap_1em2 nlp InfeasibleProblemDetected it=65 q=5 2nd=kept(4),tot=302 obj=25
< lbfgs issue_508_infeasible_gap_1em4 nlp InfeasibleProblemDetected it=79 q=1 2nd=kept(4),tot=346 obj=25
> lbfgs issue_508_infeasible_gap_1em4 nlp InfeasibleProblemDetected it=79 q=1 2nd=kept(4),tot=362 obj=25
< lbfgs scale_audit_retry_meyer3 nlp SolveSucceeded it=112 q=5 2nd=- obj=3890764353
> lbfgs scale_audit_retry_meyer3 nlp SolvedToAcceptableLevel it=286 q=12 2nd=- obj=3890764351
< lbfgs square_flowsheet_resto square-lm SolveSucceeded it=8 q=0 2nd=square_system_lm@Maximum_Iterations_Exceeded/3000,tot=6008 obj=0
> lbfgs square_flowsheet_resto square-lm SolveSucceeded it=8 q=0 2nd=square_system_lm@Restoration_Failed/22,tot=3039 obj=0
```

κ = 1e12, 11 lines: the same exact `circle_parabola_t1p5` (→ MaxIter
3000), exact `infeasible_square_scaled_1em4`, both exact and both lbfgs
`issue_508_*`, lbfgs `circle_parabola_t1p5` and lbfgs
`infeasible_square_scaled_1em4` lines as above, byte for byte; plus

```
> exact pooling_rt2stp nlp SolveSucceeded it=140 q=3 2nd=mu_strategy=adaptive@Error_In_Step_Computation/173,tot=313 obj=-3273.954989
> lbfgs eigena2 nlp ErrorInStepComputation it=159 q=4 2nd=kept(1),tot=318 obj=82.5
> lbfgs square_flowsheet_resto square-lm SolveSucceeded it=8 q=0 2nd=square_system_lm@Restoration_Failed/22,tot=51 obj=0
```

(lbfgs `circle_parabola_t0p5`, lbfgs `scale_audit_retry_meyer3`, lbfgs
`issue880_sigma_uncertified` and exact `square_flowsheet_resto` do not move
at `1e12`.) Every moved line is a fixture-leg with `S` exits above κ in
the histogram, so each move is the rule firing, not noise. Read against
the item's acceptance and stops: pooling's base solve is **lost** at both
κ (the reported 140 is a second-opinion rung's); exact
`circle_parabola_t1p5` loses its `InfeasibleProblemDetected` verdict at
both (more than one status loss at `1e6`, with lbfgs `meyer3`
`SolveSucceeded` → `SolvedToAcceptableLevel`); at `1e6` two of R2's three
counterfactual guards break (lbfgs `circle_parabola_t0p5` 8.155 → 10.428,
lbfgs `meyer3` 112 → 286 at `q = 12`) and exact `circle_parabola_t1p5`
runs `q` to the cap of 12. No `nlp → square-lm` move; engine split
unchanged (nlp 127, cvx-qp 92, cvx-qcqp 14, square-lm 1); `dallass`,
`deb7` and `scaled_feasible_*` lines do not move at either κ. The two
lines that improve — exact `square_flowsheet_resto` 95 → 57 and lbfgs
`issue880_sigma_uncertified` 67 → 32 at `1e6`; lbfgs
`square_flowsheet_resto` tot 6008 → 51 at `1e12` — are recorded in §5 for
the owner and do not change the verdict.

**Verdict.** Refuted at the measure-first step and confirmed by the
counterfactual: three of the item's stops hold (fires on 33–72% of `S`
exits at any κ that reaches the target; more than one status loss; the
`s` path does not reinstate `δ_c`). No code changed; the follow-up issue
(title + numbers) is in §5. R2b is next per §2.

**What it means for the siblings.** (1) R2b: the refinement-side
escalation it would grant fires at the same 188 `S` / 480 `s` stalls
measured here, so its measure-first count of "matrices where the inertia
branch spent the flag and refinement later stalled" can be read off the
same probe (`exit=S`/`s` whose matrix carried an inertia-branch `q`); the
pooling iteration-4 stall is one of them. (2) R2e: the trace above is
direct evidence for its premise — at iteration 4 every `δ_c`-free factor
after the walk-back is sub-floor (`min_piv` 9.4e-16 … 3.0e-14 against
`150·ε = 3.33e-14`), and the only thing that keeps `δ_c` off is that the
gh#540 noise rule reports them *uncertified*. (3) lbfgs `dallass`
`SolveSucceeded`/301 (R2d's added guard) is unmoved by this rule at
either κ.

**Tests / sweep.** Docs-only commit: no code changed, so no regression
test, no mutation check, no clippy run, no `cargo test`, and M1 is not
triggered (no landed R2x code; the M1 fixtures were re-measured above as
the reproduce step). `cargo fmt --all -- --check` clean.
`scripts/sweep-fixtures.sh` was run four times (234 fixture-legs each,
both legs): `p-prev`, the probe build with no variable set (diff empty),
and κ = 1e6 / 1e12 (diffs above). Not run: `benchmarks/vanderbei` (no
corpus), arm64 (not available), Python tests (no code change).

### R2b — 2026-10-22 — swept — needs vanderbei before merge

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` rebuilt
from the tip `85d08bc6` in a separate worktree with its own
`CARGO_TARGET_DIR`; its sweep is byte-identical to R2c's `p-prev` sweep.
The probe build is the same tip in a second worktree and target dir, with a
temporary, env-gated instrument in `pd_full_space_solver.rs` (not
committed): `POUNCE_R2B_PROBE=1` logs every successful/declined escalation
with its consumer, the next factor's verdict after an inertia-branch
escalation, and every refinement stall (which consumer had spent the flag,
`δ_c`, exit letter); `POUNCE_R2B_MODE=blanket|split|refuted|dc0|refuted_dc0`
selects the refinement rule. With neither set its sweep diffs **empty**
against `p-prev`. `POUNCE_BENCH_DATA` unset, `bench_data_root` resolves
nothing (checked once).

**Reproduced (before any change).** `p-prev`, the M1 fixtures — identical
to the R2c table: pooling exact SolveSucceeded **492**, q 3, 24 / 106
restoration, −3273.95499; lbfgs 82; dallass 28 / 301; deb7 114 /
RestorationFailed 1206; scaled_feasible_a 200 / 200 cvx-qp; lbfgs
circle_parabola_t0p5 77 / q 4 / `feral_scaling=mc64` tot 125 / 8.155059653;
lbfgs scale_audit_retry_meyer3 112 / q 5. The iteration-4 chain reproduces
in the probe: `esc who=inertia ok=true iter=4` (q 2), next factor
`Singular` (70 of 72 negatives — the count it was spent on is refuted), then
a refinement stall with the flag spent by the inertia branch, `δ_c = 0`,
exit `S`, ratio 1.59e-8, `‖res‖/‖rhs‖` 2.19e14.

**Measure first.**

(1) *The population the split changes* — matrices where the inertia branch
spent the flag and refinement later stalled wanting to escalate — over the
whole corpus, both legs (234 fixture-legs, second-opinion rungs and
restoration sub-solves included): refinement stalls 1024 (480 `s`, 188 `S`,
178 `q`, 178 second stalls — the R2c census exactly); successful
inertia-branch escalations **3** in all (exact pooling it 4, exact
square_flowsheet_resto it 2, lbfgs dallass it 36); refinement stalls on a
matrix whose flag the inertia branch had spent: **1** — pooling exact
iteration 4. The other two inertia-branch escalations are never followed by
a refinement stall on the same matrix. So the split changes exactly one
decision in the corpus before trajectories diverge.

(2) *Where the counterfactual's collateral came from.* Re-running R2's
blanket rule (refinement escalates on every stall) in the probe reproduces
R2's diff byte for byte — **15** moved lines, not the 19 the §3 text says
(the R2 log's own listing has 15). Successful refinement escalations per
matrix:

| rule | matrices escalated once | twice | 10–12 times |
|---|---|---|---|
| `p-prev` (shared flag) | 178 | 0 | 0 |
| blanket (R2 counterfactual) | 5 | 1 | **43** (38 at 12, the backend cap) |
| split (this item) | 178 | 0 | 0 |

Every collateral line under the blanket rule has matrices escalated 10–12
times from refinement (lbfgs circle_parabola_t0p5 6 matrices at 12, lbfgs
scale_audit_retry_meyer3 1 at 12, lbfgs cresc4 2 at 12, …). The hypothesis
holds: the collateral is the *unbounded* per-matrix budget, not the
per-consumer split, so the plan's narrowing (refuted-count / `δ_c = 0`) is
not needed. Measured anyway: `refuted`, `dc0` and `refuted_dc0` produce a
sweep byte-identical to `split` (the one matrix in the population is
`δ_c = 0` and refuted, so every variant takes the same decision there).

**Implementation.** `pd_full_space_solver.rs`: the `augsys_improved` flag is
replaced by `EscalationBudget { inertia_spent, refinement_spent }`, consulted
by consumer (`EscalationConsumer::InertiaDeficit` at the too-few-negatives
branch, `::Refinement` at the refinement stall); `escalate_aug_quality`
takes the consumer and records a success only (a declined escalation
spends nothing, as before); both resets (cache miss, `invalidate_aug_cache`)
clear both. The `q` letter and the gh#857 `quality_escalations` tally are
unchanged in meaning (every successful escalation, either consumer). The doc
comment on `EscalationBudget` records that this departs from Ipopt's shared
`augsys_improved_` on purpose and names the measured population above. Not
touched: `residual_ratio_singular`, `compute_residual_ratio`, the `S`/`s`
exit, the gh#592 walk-back, the gh#981 reinstatement, the `2599c2be` reset,
`pounce-feral`.

**After.**

| fixture (leg) | `p-base` v0.12.0 | `p-prev` 85d08bc6 | new |
|---|---|---|---|
| pooling_rt2stp (exact) | 162 it, q 4, 3 resto, −3273.954992 | 492, q 3, 24 / 106, −3273.95499 | **109**, q 4, **3 / 66**, −4391.82601 (global optimum; accepted by `issue_592`) |
| pooling_rt2stp (lbfgs) | ErrorInStep 716 | 82, −3273.954992 | 82 (unchanged) |

In the probe the iteration-4 matrix now carries two escalations (inertia,
then refinement); refinement refactors and does not stall again on that
matrix. Acceptance met: SolveSucceeded, an optimum, ≤ 300 (109), restoration
calls < 24 (3), no new `max_hessian_perturbation` overflow (the sweep has
no new status), the three counterfactual guards unmoved, no `q` to 12/24.

**Fixture sweep** (`scripts/sweep-fixtures.sh`, 234 fixture-legs, both legs,
`p-prev` vs new), the full diff:

```
78c78
< exact  pooling_rt2stp  nlp  SolveSucceeded  it=492  q=3  2nd=-  obj=-3273.95499
---
> exact  pooling_rt2stp  nlp  SolveSucceeded  it=109  q=4  2nd=-  obj=-4391.82601
```

- exact `pooling_rt2stp`: the one matrix in the measured population;
  refinement's escalation is the extra `q` (3 → 4), the step is no longer
  kept, and the run lands on the other certified optimum of the bistable
  pair (both are accepted by `issue_592`). Expected and the item's target.
- Nothing else moves, as the census predicts: no other matrix in the corpus
  sees a different decision. The new binary's sweep is byte-identical to the
  probe's `split` mode. Against R2's counterfactual list: of its 15 lines only
  pooling moves (the stop allows ≤ 3); lbfgs `circle_parabola_t0p5` (77, q 4,
  `feral_scaling=mc64`, 8.155059653), lbfgs `scale_audit_retry_meyer3` (112,
  q 5) and lbfgs `cresc4` (q 0, `feral_increase_quality=no@…/3000`, the
  gh#850 rung) are unchanged; lbfgs `dallass` SolveSucceeded/301 (R2d's
  guard) unchanged. Engine split unchanged: nlp 127, cvx-qp 92, cvx-qcqp
  14, square-lm 1. No status, engine or `nlp → square-lm` move.

**M1 (new tip).**

| fixture (leg) | engine | status | it | q | resto calls / it | objective / 2nd |
|---|---|---|---|---|---|---|
| pooling_rt2stp (exact) | nlp | SolveSucceeded | **109** (was 492) | 4 | 3 / 66 | −4391.82601 |
| pooling_rt2stp (lbfgs) | nlp | SolveSucceeded | 82 | 0 | 0 | −3273.954992 |
| dallass (exact) | nlp | SolveSucceeded | 28 | 0 | 0 | −32393.22574 |
| dallass (lbfgs) | nlp | SolveSucceeded | 301 | 1 | 1 / 3 | −32393.22574 |
| deb7 (exact) | nlp | SolveSucceeded | 114 | 0 | 1 / 2 | 97.55993795 |
| deb7 (lbfgs) | nlp | RestorationFailed | 1206 (kept, tot 1738) | 0 | 4 / 343 | 99.62936037 |
| scaled_feasible_a (exact / lbfgs) | cvx-qp | SolveSucceeded | 200 / 200 | 0 | 0 | 0 |
| scaled_feasible_b (exact / lbfgs) | cvx-qp | SolveSucceeded | 36 / 36 | 0 | 0 | 0 |
| circle_parabola_t0p5 (lbfgs) | nlp | SolveSucceeded | 77 | 4 | 0 | 8.155059653 / `feral_scaling=mc64`, tot 125 |
| scale_audit_retry_meyer3 (lbfgs) | nlp | SolveSucceeded | 112 | 5 | 0 | 3890764353 |
| F1 margin | | | 500 − 109 = **391** | | | |

Only the pooling exact row moved. `dallass` is not moved (R3's 28 stands).

**Comparison with the sibling counterfactuals (§4 rule 4, recorded for the
owner).** R2d's "every deficit → `Singular`" rule reached pooling **162**
(v0.12.0's count) but lost lbfgs `dallass` (SolveSucceeded/301 →
ErrorInStep/269); R2c's `S` refusal lost pooling's base solve at every κ
that reached iteration 4. R2b reaches 109 with no other line moving.

**Tests.** New: `pd_full_space_solver.rs` unit tests
`refinement_keeps_one_escalation_after_the_inertia_branch_spends_its_own`
(the iteration-4 order: inertia spends, refinement still may, then may not
again) and `each_consumer_budget_is_independent_and_per_matrix` (the other
order; a reset restores both); CLI pin
`crates/pounce-cli/tests/issue_1042_pooling_escalation_budget.rs` (default
exact run: solved, no second opinion, either optimum, ≤ 300 it, < 24
restoration calls). Mutation checks: (1) `may_escalate` returning the old
shared flag (`!(inertia_spent || refinement_spent)`) — both unit tests fail,
the first at "an inertia-branch escalation must not spend refinement's",
and the CLI pin fails (`pooling_rt2stp took 492 iterations`); (2)
refinement unbounded (`Refinement => true`) — both unit tests fail, the
first at "refinement escalates at most once per matrix". Restored. The
`issue_592` `< 500` pin is untouched (it guards gh#544's 812 régime; R2b's
pin carries the ≤ 300 bound; tightening `issue_592` waits for the
vanderbei A/B).

Run: `cargo fmt --all -- --check` (clean); `cargo clippy --workspace
--exclude pounce-hsl --all-targets -- -D clippy::correctness -D
clippy::suspicious` (clean; the new CLI pin carries the same six
`expect_used`-style warnings as R3's pin, nothing denied); `cargo test -p
pounce-common -p pounce-algorithm -p pounce-feral -p pounce-cli -p
pounce-wasm --no-fail-fast` (250 test binaries, **2084 passed, 0 failed**, 1
ignored) — the named guards among them: `issue_592_delta_c_walkback` (all
three, incl. `the_walkback_is_inert_on_this_fixture_since_gh693`),
`issue_250_dual_guard_never_worse`, `issue857_*` (both files),
`issue_983_scale_audit_retry_budget`, `issue_884_biactive_dual_divergence`,
gh#981's `a_duplicated_row_keeps_its_delta_c` (±adaptive μ), the
`pd_perturbation` unit tests (`an_overflowed_ladder_does_not_poison_the_next_system`,
the gh#592 walk-back and gh#981 reinstatement tests), the `pounce-feral`
gh#540/gh#1003 tests, the `pounce-wasm` pooling shim, and R3's
`issue_1042_dallass_abs_pow_merge`. (The suite was run with
`CARGO_INCREMENTAL=0 CARGO_PROFILE_{DEV,TEST}_DEBUG=0`: with debuginfo the
debug test binaries filled the disk on this machine.) Sweep and M1 above.
Not run: `benchmarks/vanderbei` (no corpus; `steenbr*`, `eigen*`, `dallas*`
are the models to watch in the owner's A/B) — hence `swept — needs
vanderbei before merge`, not `done`; arm64 (not available); Python
`test_mehrotra_feasible_gh986.py` (no `maturin` and no built extension on
this machine).

**Next.** Per §2, **R2e** — first its §4 sibling-rule-1 reproduce step on
this tip: the iteration-4 walk-back still accepts the same `δ_c`-free
factor (R2b repairs the step downstream, it does not stop the factor being
accepted), so whether R2e is `dropped — recovered by R2b` or has a
*different population* to move (rule 2) is that iteration's measurement.

### R2e — 2026-10-22 — not recoverable this way (measured); R2b's 109 is a fallback rescue

Measured on x86_64 (4 CPUs, AVX2+FMA). `p-prev` = release `pounce` rebuilt
from the tip `3e186634` (R2b) in a separate worktree with its own
`CARGO_TARGET_DIR`; its sweep (234 fixture-legs, both legs; engine split
nlp 127, cvx-qp 92, cvx-qcqp 14, square-lm 1) matches R2b's after-sweep
line for line. `p-head` (`0a299151`, 0.13.0) and `p-base` (v0.12.0) are the
binaries the baseline was taken with. The probe build is the same tip in
the same worktree (not committed): an env-gated line after every KKT
factorization in `PdFullSpaceSolver`'s retry loop (`POUNCE_R2E_PROBE`:
iteration, status, `singularity_certified`, negatives vs required, the
handler's `delta_c_abandoned` / `withdrawal_untested` /
`delta_c_reinstated`, `δ_x`, `δ_c`, and the factor's `min_piv`, `max_piv`
and `inertia_trust_floor`, from a thread-local set in
`FeralSolverInterface::factor`), plus two counterfactual rules behind
`POUNCE_R2E_CF` / `POUNCE_R2E_CF2` (below). With the probe on and no rule
set its 234 result lines are identical to `p-prev`'s sweep.
`POUNCE_BENCH_DATA` unset, `bench_data_root` resolves nothing (checked
once).

**Reproduce (§4 sibling rule 1) — and a correction to R2b.** `p-prev`, the
M1 fixtures, read off the JSON report as every earlier iteration did:
identical to R2b's M1 table (pooling exact SolveSucceeded **109**, q 4,
3 / 66 restoration, −4391.82601; lbfgs 82; dallass 28 / 301; deb7 114 /
RestorationFailed 1206 tot 1738; scaled_feasible_a 200 / 200;
scaled_feasible_b 36 / 36; lbfgs circle_parabola_t0p5 77 q 4
`feral_scaling=mc64` tot 125 8.155059653; lbfgs meyer3 112 q 5). Read off
the **console**, the pooling exact run is two attempts:

| binary | attempt 1 (base solve) | attempt 2 | JSON / sweep reports |
|---|---|---|---|
| `p-base` v0.12.0 | 162, `Solve_Succeeded`, 3 resto | — | 162 |
| `p-head` 0.13.0 | 492, `Solve_Succeeded`, −3273.95499, 24 resto | — | 492 |
| `p-prev` R2b tip | **494, `Solved_To_Acceptable_Level`, −4336.990888, unscaled dual inf 5.3e6, 22 resto** | `mu_strategy_fallback` (pounce#748) re-solve: 109, `Solve_Succeeded`, −4391.82601, 3 resto — promoted | 109, 3 resto |
| `p-prev`, `mu_strategy_fallback=no` | 494, Acceptable (the gh#884 dual-divergence retry then runs 294, Acceptable, not promoted) | — | `SolvedToAcceptableLevel` 494 |

The JSON report carries no field for a promoted fallback, so its
`iteration_count` and `restoration_calls` are attempt 2's alone; the
sweep's `2nd=` column covers the second-opinion ladder only (it reads
`-`); and both pooling pins — `issue_592`'s `< 500` and R2b's
`issue_1042_pooling_escalation_budget.rs` `≤ 300` / `< 24` resto / "no
second opinion" — read those fields. So R2b's "492 → 109, 24 → 3
restoration calls" is the fallback's result: R2b's one changed decision
(its census stands) moved the base solve from `Solve_Succeeded`/492 to
`Solved_To_Acceptable_Level`/494 at a non-optimal point, and the total
cost of the run rose to 603 iterations / 25 restoration calls. That is
recorded in the inventory (R2, R2b, F1 rows) and §5 with two suggested
issues; R2b's code is not touched here (§4 rule 4, one item per
iteration). Of the other M1 rows only pooling exact reports a promoted
attempt that `p-head` did not need (dallass, deb7 exact, pooling lbfgs,
scaled_feasible_* are single attempts; the multi-attempt rows —
circle_parabola_t0p5 lbfgs, deb7 lbfgs, meyer3 — are second-opinion or
scale-audit attempts already in the sweep's columns or pre-existing).

Against R2e's acceptance (as R2c's: pooling ≤ 300, resto < 24, an
optimum) the regression **does reproduce** on the base solve (494, 22,
not an optimum), so R2e is not `dropped — recovered by R2b`;
re-baselined after R2b, its target is the base solve: `Solve_Succeeded`
≤ 300 at an optimum with < 24 restoration calls and no fallback attempt.

**The item's target factor.** Pooling iteration 4 on this tip runs the
same ladder as before (R2b acts in refinement, after it): `WrongInertia`
71 (`min_piv` 5.95e-14) → escalation → `Singular` 70 (6.5e-15) →
`WrongInertia` 73 ×3 with `δ_c` (1.8e-3) → walk-back → `Success` at
`δ_x` 1e-4, `δ_c` 0, 72 negatives, **`min_piv` 3.977e-14**, `max_piv`
1.48e4, floor `150·ε` = **3.331e-14**. The pivot is 1.19× *above* the
floor — the item's own refutation clause ("refuted if the pooling
iteration-4 pivot is not below the floor"; R2d measured 4.0e-14 on the
pre-R2b tip). The rule could never have stopped the 6.4e16 step. R2c's
note that every `δ_c`-free factor after the iteration-4 walk-back is
sub-floor described the `s` path its own counterfactual opened.

**Population census** (whole corpus, both legs, second-opinion rungs,
fallback attempts and restoration sub-solves included; 51 360
factorizations). Walk-backs (first `δ_c`-free factor after a
withdrawal): **63 on 8 fixture-legs** — exact pooling_rt2stp 39, exact
unbounded_exp 9, lbfgs issue_508_infeasible_gap_1em2 8, exact
mu_fallback_point_floor 2, exact square_flowsheet_resto 2, exact
circle_parabola_t1p5 / infeasible_square_scaled_1em4 / issue981_cstr_dup_row
1 each.

| first `δ_c`-free factor | above floor | below floor |
|---|---|---|
| `WrongInertia` (gh#592's full-rank case) | 21 | 0 |
| `Success` | 6 (pooling 5, lbfgs issue_508_1em2 1) | **8 (all exact pooling)** |
| certified `Singular` (gh#981 already reinstates) | — | 9 |
| uncertified `Singular` (gh#540 noise rule) | 0 | 19 (pooling 16, mu_fallback_point_floor 2, square_flowsheet_resto 1) |

The 8 sub-floor `Success`es sit at pooling iterations 5, 7, 8, 10, 30, 39
(`min_piv` 2.5e-14 … 2.8e-16) and two in later attempts / restoration
(7.3e-19, 1.4e-14); `min_piv/floor` over all 14 first-factor `Success`es
spans 1e-5 … 1e15 with no gap at the floor. Over every accepted
`δ_c`-free factor (any rung) 20 of 38 are sub-floor — pooling 17 of 24,
mu_fallback_point_floor 2 of 2, square_flowsheet_resto 1 of 1. The rule as
specified has no population outside pooling (§4 rule 2), and on pooling
it does not reach iteration 4.

**Counterfactual 1 — the rule as specified** (`POUNCE_R2E_CF`: a
`Success` while the withdrawal is untested and not reinstated, with
`min_piv < floor`, is reported certified `Singular`, so gh#981 reinstates
`δ_c`). Run on the 8 walk-back fixture-legs; no other leg has a
walk-back, and trajectories are identical up to the first firing, so no
other line can move. Attempts read off the console:

| fixture-leg | `p-prev` attempts → reported | CF 1 attempts → reported | fires |
|---|---|---|---|
| exact pooling_rt2stp | 494 Acceptable + 109 → SolveSucceeded 109 | **3000 MaxIter** (25 resto, −3183.18) + `feral_increase_quality=no` rung 1406 kept → **MaximumIterationsExceeded 3000**, tot 4406 | 3 base (it 5, 105, 141) + 1 rung |
| the other 7 | — | identical | 0 |

First firing at iteration 5 (`min_piv` 2.48e-14): the reinstated `δ_c`
meets the surplus count (74 of 72) the walk-back had answered, `δ_x`
climbs to 0.14 with `δ_c` on, and the run sits at `lg(rg)` 17–20 to the
cap; the fallback is not eligible (MaxIter) and the rung does not rescue
it. A reported status lost.

**Counterfactual 2 — the widened reading** (`POUNCE_R2E_CF2`: also treat an
uncertified `Singular` with `min_piv < floor` from the first `δ_c`-free
factor as certified). The gh#540 noise rule reports `Singular` only below
the floor, so this is "reinstate on any first `Singular`" — the gh#981
candidate `dev-notes/issue-981-delta-c-walkback.md` measured and rejected:

| fixture-leg | `p-prev` | CF 2 | fires |
|---|---|---|---|
| exact pooling_rt2stp | 494 Acc + 109 → SolveSucceeded 109 | base **141 Acceptable at −3261.358** (8 resto) + fallback 491 → SolveSucceeded **491**, −3273.954992; tot 632 | 11 |
| exact square_flowsheet_resto | SolveSucceeded 95 | SolveSucceeded 107 | 1 |
| exact mu_fallback_point_floor | 55 Acc + 31 → Acceptable 55 | 49 Acc + 46 → Acceptable 49 | 2 |
| `mpcc_qpec_small_biactive`, `bound_relax_factor=0 mu_strategy_fallback=no` (gh#884's reproducer; not a sweep line) | SolveSucceeded 97, unscaled dual inf 1.3e-10 | SolveSucceeded 53, unscaled dual inf **7.0e-5** | 1 |
| the other 5 walk-back legs | — | identical | 0 |

Neither base solve is optimal; the widened rule's reported count is
0.13.0's (491 vs 492), and it reopens gh#884's failure class — the
certified/uncertified split is gh#981's guard, so this is the "weaken an
original fix" stop of §4.

**Verdict.** Not `dropped`: on the base solve the regression still
reproduces (494, Acceptable). **Not recoverable this way**: the item's
own refutation clause holds (iteration-4 pivot above the floor), its
only population is 8 later pooling factors where firing loses the solve
(reported `SolveSucceeded` → `MaximumIterationsExceeded` 3000), and the
widened form is gh#981's rejected candidate (gh#884 dual 1.3e-10 →
7.0e-5). No follow-up issue for this lever (§5); two for what the
reproduce step found about R2b (§5). Every R2 member is now terminal, so
the family closes by status (§4 rule 6) **with its target unmet** on the
base solve. **Next per §2: B1** — but B1 waits for the R2 family *so that
its re-baselines are not confounded*, and the R2 baseline it would be
taken against is the corrected one above; the owner may prefer a
correction iteration on R2b first (re-measure its census outcome with the
fallback visible, and make its pin read the base solve), which this plan
does not yet list as an item.

**Tests / sweep.** Docs-only commit: no code changed, so no regression
test, mutation check, clippy run or `cargo test`, and M1 is not triggered
(no landed R2x code; the M1 fixtures were re-measured above as the
reproduce step: JSON numbers unchanged from R2b's table, F1 margin 391 on
the reported count and **6** on the base solve). `cargo fmt --all --
--check` clean. `scripts/sweep-fixtures.sh` was run once on `p-prev` (the
baseline the probe census was checked against, identical); with no new
binary there is nothing to diff. Not run: `benchmarks/vanderbei` (no
corpus), arm64 (not available), Python tests (no code change). The probe
and both counterfactuals lived only in the scratch worktree, which was
removed.

### C1 — 2026-10-22 — done; R2b reverted (first solve regressed)

Measured on x86_64 (4 CPUs, AVX2+FMA). Four binaries, each carrying the C1
report patch (below) so that all four columns are read the same way, each
built in its own worktree with its own `CARGO_TARGET_DIR`:
`v0.12.0` = `a31dab9c` (`pounce 0.12.0`; the patch applied with a
hand-resolved 3-way merge, since v0.12.0's fallback driver has no
`RetryOptionScope`), `0.13.0` = `0a299151`, `pre-R2b` = `85d08bc6` (contains
R3), `tip` = `36fefb32` (contains R2b). `POUNCE_BENCH_DATA` unset,
`bench_data_root` resolves nothing (checked once).

**How a fallback was reported before C1.** `run_with_mu_strategy_fallback`
(`pounce-algorithm/src/application.rs`) runs the first attempt; on
`Solved_To_Acceptable_Level` (stock termination options) or
`Maximum_Iterations_Exceeded` (unless the gh#857 decline applies) it flips
`mu_strategy` and re-solves, promoting the re-solve only on
`Solve_Succeeded`. On a promotion the statistics are the re-solve's
(`iteration_count`, `restoration_calls`, residuals); on a loss the
pounce#870 floor restores the first attempt's. Nothing recorded that the
retry ran: no JSON field (the second-opinion ladder's `second_opinion`
covers that ladder only), no sweep column, and only `RUST_LOG`-level
tracing on a loss. The console shows two iteration tables, which is how R2e
found it; nothing machine-readable did.

**The instrument.**
- `pounce_nlp::MuStrategyFallbackRecord` (`solve_statistics.rs`):
  `base_status`, `base_iteration_count`, `base_restoration_calls`,
  `base_objective`, `retry_mu_strategy`, `retry_status`,
  `retry_iteration_count`, `promoted`, plus `total_iteration_count()`.
  `SolveStatistics::mu_strategy_fallback: Option<…>` (`serde(default)`).
- `IpoptApplication`: a `mu_strategy_fallback_record` cell, reset per solve
  in `optimize_tnlp_dispatch` beside the gh#884 flags, filled by the
  fallback driver before and after the re-solve, copied into every later
  attempt's statistics (sticky, like `dual_divergence_signature`) and
  written into the returned statistics on both the promote and the floor
  path. No decision reads it.
- Solve report: `statistics.mu_strategy_fallback`, omitted when no retry
  ran (additive to `pounce.solve-report/v1`).
- `scripts/sweep-fixtures.sh`: a `mu=` column between `2nd=` and `obj=`:
  `-`; `<strategy>@<first status>/<first it>,tot=N` (promoted); or
  `kept(<strategy>:<retry status>/<retry it>),tot=N` (lost). Documented in
  the script header and in CLAUDE.md's sweep section (the line format
  changed).

**The patch is report-only (measured).** With the `mu=` column dropped
(`sed -E 's/ mu=[^ ]+ +/ /'`), each patched binary's sweep is byte-identical
to the unpatched one: v0.12.0 against `build/p-base`'s sweep (taken this
iteration); 0.13.0 against R2's `p-prev` sweep (`463eeeae`, code-identical
to `0a299151`); pre-R2b against R2c's `p-prev` sweep (`85d08bc6`); tip
against R2c's `p-prev` with the one pooling line R2b's log records. The
unpatched v0.12.0 binary reads `mu=-` on all 234 lines, including `csfi2`,
which the patched one shows promoted — the blind spot, in one line.

**Every line where a fallback ran** (234 fixture-legs per binary, both legs):

| fixture (leg) | v0.12.0 | 0.13.0 | pre-R2b | tip |
|---|---|---|---|---|
| csfi2 (exact) | `adaptive@Acceptable/35,tot=56`, it 21 | same | same | same |
| mu_fallback_point_floor (exact) | `kept(adaptive:Infeasible/31),tot=86`, it 31¹ | `kept(…/31),tot=86`, it 55 | same as 0.13.0 | same |
| eigenb2 (lbfgs) | `kept(monotone:Acceptable/41),tot=88`, it 41 | `kept(…/42),tot=89`, it 47 | same as 0.13.0 | same |
| mu_fallback_point_floor (lbfgs) | `kept(monotone:Acceptable/26),tot=46`, it 26¹ | `kept(…/26),tot=46`, it 20 | same as 0.13.0 | same |
| **pooling_rt2stp (exact)** | `-` (162, one attempt) | `-` (492, one attempt) | `-` (492) | **`adaptive@Solved_To_Acceptable_Level/494,tot=603`**, it 109 |

¹ v0.12.0 reports the losing retry's `it=` beside the first attempt's
status (pre-gh#983-item-5: the floor did not yet restore
`iteration_count`); the record's `base_iteration_count` (55 / 20) is the
first attempt's. Not a regression, a reporting fix that shipped in 0.13.0.

So there is exactly **one hidden-fallback line** introduced anywhere on the
branch: R2b's exact `pooling_rt2stp`. `csfi2` exact is a promoted fallback
on all four binaries with identical numbers — pre-existing, unmoved, and
until now invisible; it is not a #1042 regression. The three `kept` lines
were already scored on the first attempt (`it=` is the first attempt's on
0.13.0 and later). No line combines a fallback with a second-opinion
ladder, so no line's `2nd=` base count hides a fallback either.

**Audit table** (sweep lines; first attempt / final; `resto` from the JSON):

| fixture (leg) | v0.12.0 | 0.13.0 | pre-R2b (R3) | tip (R2b) |
|---|---|---|---|---|
| pooling_rt2stp (exact) | Optimal 162, one attempt | Optimal 492 (24 resto), one attempt | Optimal 492, one attempt | **first Acceptable 494 (−4336.99, 22 resto) → fallback Optimal 109 (3 resto); total 603** |
| pooling_rt2stp (lbfgs) | ErrorInStep 716, one attempt | Optimal 82, one attempt | 82 | 82 |
| dallass (exact) | Optimal 41, one attempt | Optimal 123, one attempt | **Optimal 28, one attempt** | 28 |
| dallass (lbfgs) | ErrorInStep 213, one attempt | ErrorInStep 146, ladder `kept(1)` tot 292 | **Optimal 301, one attempt** | 301 |
| deb7 (exact) | Optimal 143 | Optimal 114 | 114 | 114 |
| deb7 (lbfgs) | ErrorInStep 1010 | RestorationFailed 1206, `kept(1)` tot 1738 | same | same |
| scaled_feasible_a (exact / lbfgs) | nlp 22 / 19 | cvx-qp 200 / 200 | same | same |
| scaled_feasible_b (exact / lbfgs) | cvx-qp 47 / 47 | cvx-qp 36 / 36 | same | same |
| circle_parabola_t0p5 (lbfgs) | `feral_scaling=mc64@Infeasible/48`, 77, tot 125 | same | same | same |
| scale_audit_retry_meyer3 (lbfgs) | 112, q 5 | same | same | same |

No M1 row other than pooling exact carries a fallback on any binary.

- **R3 stands as recorded.** `dallass` is a single attempt on both legs at
  pre-R2b and tip (`mu=-`, `2nd=-`): exact 123 → 28, lbfgs ErrorInStep
  146 (+ ladder, 292) → Optimal 301. R3's pin checked "no second opinion"
  but not the fallback; it now checks both (below).
- **R2b was mis-scored.** Its first attempt is *worse* than 0.13.0's —
  Optimal 492 → Acceptable 494 at a non-optimal point (unscaled dual inf
  5.3e6) — and the run's total is 603 iterations / 25 restoration calls
  against 492 / 24. Decision below.
- **The M1 rows of R2c/R2d/R2e's logs stand** (those iterations changed no
  code, and their M1 rows are the pre-R2b / tip columns above). Their
  *counterfactual* numbers are a different matter: R2e's were read off the
  console attempt by attempt, but R2c's κ table and R2d's "every deficit →
  `Singular`" result (pooling 162) were read off the JSON report before C1
  and **may hide a fallback**. Neither lever landed, so nothing is built on
  them; they are marked *unverified for fallback* here rather than re-run
  (the probes lived in removed scratch worktrees).
- **R1's numbers stand**: the convex arm has no μ fallback;
  `scaled_feasible_*` lines are single attempts on every binary.

**Tests (instrument).** New
`crates/pounce-cli/tests/issue_1042_mu_fallback_is_reported.rs`: a promoted
fallback (`csfi2` exact: record present, first attempt
`Solved_To_Acceptable_Level`, `retry_iteration_count == iteration_count`,
total = sum), a losing one (`mu_fallback_point_floor` exact: not promoted,
`base_iteration_count == iteration_count`), and none under
`mu_strategy_fallback=no`. Mutation checks: record never written — the
first two fail; `promoted` hard-wired `false` — the first fails;
`retry_iteration_count` read before the re-solve — the first fails.
Restored. R3's pin (`issue_1042_dallass_abs_pow_merge.rs`) now asserts
`statistics.mu_strategy_fallback` is absent before it reads any count;
mutation check: pointed at `pooling_rt2stp.nl` on the R2b tip, every
earlier assertion passes (Optimal, no second opinion) and the new one fails
with the record (`494 … Solved_To_Acceptable_Level … promoted`). Restored.

**R2b decision: revert.** The criterion (this iteration's brief): the first
attempt must not get worse than 0.13.0's Optimal/492 and total work must
not exceed it, unless a net win elsewhere is shown; a fallback rescue is not
a recovery.

| pooling_rt2stp (exact) | first attempt | retry | reported | total it / resto |
|---|---|---|---|---|
| 0.13.0 (`0a299151`) | `Solve_Succeeded` 492, −3273.95499, 24 resto | — | 492 | 492 / 24 |
| pre-R2b (`85d08bc6`) | same | — | 492 | 492 / 24 |
| R2b (`36fefb32`) | **`Solved_To_Acceptable_Level` 494, −4336.990888, unscaled dual inf 5.3e6, 22 resto** | adaptive, `Solve_Succeeded` 109, −4391.82601, 3 resto — promoted | 109 | **603 / 25** |
| after revert (this commit) | `Solve_Succeeded` 492, −3273.95499, 24 resto, q 3 | — | 492 | 492 / 24 |

Both halves of the criterion fail (first attempt Optimal → Acceptable at a
non-optimal point; total 492 → 603), and there is no win elsewhere to set
against it: the sweep moved no other line (R2b's census — one changed
decision in the whole corpus — is confirmed by C1's tip-vs-pre-R2b diff,
which is that one line on every column including `mu=`). Lbfgs pooling 82
either way. **No narrowing can help inside this lever:** the population is
one matrix, and R2b's own measurement found the `refuted`, `dc0` and
`refuted_dc0` narrowings byte-identical to the split, so any variant that
still takes the iteration-4 decision reproduces the Acceptable/494 first
attempt and any that does not is the revert. So `git revert 3e186634` for
the code (`pd_full_space_solver.rs` back to `85d08bc6`'s byte for byte, and
R2b's CLI pin `issue_1042_pooling_escalation_budget.rs` removed with it —
it passed on the fallback's count, which is the defect C1 names); R2b's §3
text and §6 log stay as the record, marked reverted. §4 rule 4 ("a landed
sibling is not reverted because a later one would have been narrower") does
not apply: this is not a narrower-sibling comparison but a landed item
whose own target was never met on the first attempt.

**Fixture sweep** (`scripts/sweep-fixtures.sh`, 234 fixture-legs, both
legs). The `mu=` column moves every line's layout, so the diff is read two
ways. (1) Old columns only (`sed -E 's/ mu=[^ ]+ +/ /'`), tip vs the final
binary (the tip's old columns are R2c's unpatched `85d08bc6` sweep plus
R2b's recorded pooling line; whitespace squeezed below):

```
< exact  pooling_rt2stp  nlp  SolveSucceeded  it=109  q=4  2nd=-  obj=-4391.82601
> exact  pooling_rt2stp  nlp  SolveSucceeded  it=492  q=3  2nd=-  obj=-3273.95499
```

(2) All columns, whitespace squeezed, patched tip vs final:

```
< exact pooling_rt2stp nlp SolveSucceeded it=109 q=4 2nd=- mu=adaptive@Solved_To_Acceptable_Level/494,tot=603 obj=-4391.82601
> exact pooling_rt2stp nlp SolveSucceeded it=492 q=3 2nd=- mu=- obj=-3273.95499
```

- exact `pooling_rt2stp`: R2b's one decision undone; the run is again
  0.13.0's single attempt (first-attempt status unchanged from 0.13.0,
  `mu=` promotion gone). Expected and the purpose of the revert.
- Nothing else moves. The final sweep equals the patched pre-R2b sweep
  line for line (all columns), and with `mu=` dropped it equals R2c's
  unpatched `85d08bc6` sweep. Engine split unchanged: nlp 127, cvx-qp 92,
  cvx-qcqp 14, square-lm 1. The four other `mu=` lines (csfi2 exact
  promoted; mu_fallback_point_floor both legs, eigenb2 lbfgs kept) are
  identical on all four binaries.

**M1 (new tip)** — pre-R2b's column of the audit table, first attempts
named: pooling exact Optimal 492 single attempt (24 / 106 resto, q 3,
−3273.95499); pooling lbfgs 82 single; dallass 28 / 301, single attempts
both legs; deb7 114 / RestorationFailed 1206 (`kept(1)`, tot 1738);
scaled_feasible_a 200 / 200 cvx-qp; scaled_feasible_b 36 / 36; lbfgs
circle_parabola_t0p5 77 q 4 `feral_scaling=mc64@Infeasible/48` tot 125;
lbfgs meyer3 112 q 5. No M1 row carries a fallback.

**F1.** The `issue_592` `< 500` pin is met by a single attempt at 492
(margin 8), not by a fallback. On the R2b tip it was met by the fallback's
109 while the first attempt sat at 494 (margin 6, Acceptable). The pin is
untouched — not raised, and no fallback assertion added, because pooling's
arm64 run (issue: "109 → 242 total") is unmeasured and might take a retry
there legitimately; that is for whoever can run arm64.

**Pins.** R3's pin asserts `statistics.mu_strategy_fallback` is absent
before it reads a count (mutation check above). R2b's pin is removed with
its code. `nonfinite_hessian_restoration_strike.rs` (the `5fc8c797`
exemption) unchanged and run.

**Run.** `cargo fmt --all -- --check` (clean); `cargo clippy --workspace
--exclude pounce-hsl --all-targets -- -D clippy::correctness -D
clippy::suspicious` (clean, nothing denied); `cargo test -p pounce-nlp -p
pounce-solve-report -p pounce-algorithm -p pounce-cli --no-fail-fast` on the
final tree (`CARGO_INCREMENTAL=0 CARGO_PROFILE_{DEV,TEST}_DEBUG=0`; 249 test
binaries, **2083 passed, 0 failed**, 1 ignored) — among them the guards
this brief names: `issue_592_delta_c_walkback` (all three, incl.
`the_walkback_is_inert_on_this_fixture_since_gh693`), R3's
`issue_1042_dallass_abs_pow_merge` (with the new assertion),
`nonfinite_hessian_restoration_strike` (the `5fc8c797` exemption), gh#981's
`a_duplicated_row_keeps_its_delta_c` (± adaptive μ), and
`issue_870_mu_fallback_solution_floor`, `issue857_*`,
`issue_983_scale_audit_retry_budget`, `issue_250_dual_guard_never_worse`,
`issue_884_*`; the new `issue_1042_mu_fallback_is_reported`. The four
sweeps and the final sweep above. Not run: `benchmarks/vanderbei` (no
corpus) — moot for R2b now that it is reverted, and C1's instrument changes
no decision; arm64 (not available); Python tests (no `maturin` / built
extension here; the Python report reads `SolveStatistics` through
`pounce-py`, which compiles under the clippy run, and gains the field
additively). The scratch worktrees and their target dirs were removed.

**Next.** Per §2, **B1** — its re-baselines can now be taken with the
fallback visible. The R2 family stays closed with its target unmet
(pooling exact 492 on the first attempt); any new lever on it is a new
item, scored on the first attempt per §4's C1 rule.

### B1 — 2026-10-22 — skipped (owner's work in progress: draft PR #1046)

Checked on `c5e5984d` (branch tip after C1), x86_64. `POUNCE_BENCH_DATA`
unset, `bench_data_root` resolves nothing, no `~/projects/pounce-bench-data`
(checked once).

**The before-starting check fired.** `git fetch origin` brought in
`origin/perf/feral-symbolic-after-init` at `c9182194` ("Give the IPM's first
KKT factorization its own FERAL analysis (jkitchin/feral#224)", one commit on
`c344c954`, 14 files, +752/−103), and it is open **draft PR #1046** in
jkitchin/pounce (created 2026-10-22 16:04Z). Per §3 B1 ("If that branch or
PR exists, do not duplicate it — record its state in the item log and skip
B1"), no code was written, no `p-prev` built and no sweep taken in this
iteration. Recorded from the PR and the issue thread, **not measured here**:

- **What it does.** `analysis_count()` / `forget_analysis()` on
  `SparseSymLinearSolverInterface` and `AugSystemSolver` (no-op defaults;
  FERAL: `symbolic_call_count` / `invalidate_symbolic_cache`; block, Schur
  and restoration composites forward; `LowRankAugSystemSolver` forwards to
  the main solver only, so the gh#730 init bypass keeps its `W = 0`
  analysis). `IpoptAlgorithm::optimize_inner` brackets
  `set_initial_iterates` with `analysis_count()` and forgets only an
  analysis the initializer *created* — which is exactly the semantics
  question this plan left open (step 3's "re-use by default, re-analyse
  after the invalidation"): a pooled backend or second `optimize` keeps its
  reuse. Tests: `crates/pounce-algorithm/tests/feral_analysis_after_multiplier_init.rs`
  (dim-220 fixture where MC64 provably picks different orderings for the init
  and first-KKT matrices; mutation-checked both ways) and a `pounce-feral`
  unit test. It **weakens** `issue_540_eigena2_superlinear_tail.rs`'s 16×
  dual-residual margin assertion (said so in the PR body) — for the owner to
  judge; this branch would not be allowed to (§4 "never weaken original
  fixes").
- **Its numbers (arm64, 14 cores, PR body).** qcqp1000-2c LSF 103.2 →
  73.9 s, wall 106.1 → 76.3 s (it 163 → 182); NARX_CFy wall 150 → 112 s;
  clnlbeam base solve 2465 → 514 it. Losses: qcqp1000-2nc wall +37% at equal
  iterations (nnz(L) 1.48M → 2.06M); **qcqp1500-1c 206 → 586 it** (B3's
  model); nql180 +21% wall; brainpc3 +1.1 s; dallasl 45 → 135 + 24 (a clean
  solve now rescued). Net over 26 models: wall 1229 → 1196 s, factor
  929 → 939 s. brainpc0/1 bit-identical (B2 untouched).
- **Its sweep (both legs, vs `c344c954`, both arches).** Two exact lines
  move; lbfgs identical. x86_64: `pooling_rt2stp` 492 q=3 → **111** q=1
  (−3273.954992); `deb7` `SolveSucceeded`/114 `2nd=-` → `SolveSucceeded`/225
  with `2nd=mu_strategy=adaptive@Error_In_Step_Computation/396,tot=621` —
  by §4's C1 rule a **lost first attempt rescued by a retry** (114 → base
  ErrorInStep 396, total 621), which the PR itself calls a real x86_64
  regression. **That sweep predates the `mu=` column (C1, `423f8568`)**, so
  its `pooling_rt2stp` 111 is *unverified for a promoted
  `mu_strategy_fallback`* — the exact blind spot that mis-scored R2b's 109.
  The PR reports pooling's base solve "stops failing", so it is probably a
  first attempt, but whoever re-sweeps #1046 should read the `mu=` column
  (or `statistics.mu_strategy_fallback`) before counting it toward the R2
  family's target (pooling exact ≤ 300 on the first attempt) or F1's margin.
- **The owner's direction since.** Issue comment 6346290866 (16:52Z): the
  host-side forget is model-dependent with total factor time flat; "the
  principled fix is a staleness-triggered re-analysis inside FERAL …
  jkitchin/feral#227", after which pounce bumps FERAL, exposes the policy
  option, and sweeps/benchmarks both arches. So B1's pounce-side lever as
  written in §3 is superseded by an upstream change this branch cannot ship
  (crates.io dependency, no git deps — §3 R2d "Where FERAL lives").

**B2 / B3.** Also owner's work now: open PR **#1047**
(`origin/fix/brainpc-restoration-path`, `4d2cde1a`, "Restoration: end the
cycle on an unreleased feasible hand-back the outer cannot leave"), and
comment 6350601458 records a further restoration item (the outer-progress
callback is passed max-norm violation and plain `f` where the filter holds
1-norm and the barrier objective). Both stay blocked here on the corpus
regardless; nothing to do on this branch.

**Not done, and why.** No implementation (would duplicate #1046); no
`p-prev` / sweep / wall-clock / factor-count measurement (nothing on this
branch changed — docs-only commit; measuring #1046's binary is the owner's
A/B, and the one number this machine could add to it is the `mu=` read on
its x86 pooling line, suggested above rather than taken, to stay inside
"do not duplicate"). `cargo fmt --all -- --check` run (clean).

**Next.** Per §2, **R4** (MLSL default `gamma`, its own
`sweep-minima.sh`; `sweep-fixtures.sh` must diff empty) — independent of
B1 and of FERAL.

### Loop paused — 2026-10-22, at `e4d87a51`

Paused for budget before R4 started; R4 made no changes. To resume, pick up
§2 at **R4** (then R4b, R1d, R1e, R1b, R1c), following §4 including the C1
first-attempt rule. The `p-base` (v0.12.0) binary and sweep outputs lived in
the session scratchpad and are gone; rebuild them.

### Re-validated on main 68230fab — 2026-10-24

Branch tip `59fec629` = merge of `origin/main` `68230fab` (gh#1054 SQP at
scale / IPM→SQP handover, gh#1056 cone balancing, gh#1050 SOC corrector,
gh#1055 external-iterate sensitivity, gh#1049) into the previous head
`eab6e8fc`. Measured on x86_64, release `pounce-cli` built per worktree for
`68230fab` (`p-main`) and `59fec629` (`p-pr`). Both sweeps used the
branch's `scripts/sweep-fixtures.sh` (main's has no `mu=` column; `p-main`
reads `-` there regardless), and the column was dropped with
`sed -E 's/ mu=[^ ]+ +/ /'` before diffing. 242 fixture-legs each (121
fixtures × 2), no `NO_JSON`; engine split identical either side (exact
46 cvx-qp / 12 cvx-qcqp / 63 nlp; lbfgs 46 / 12 / 62 + 1 `square-lm`,
`square_flowsheet_resto`, unchanged from main).

```
19c19
< exact  dallass  nlp  SolveSucceeded          it=123  q=19  2nd=-                obj=-32393.22574
---
> exact  dallass  nlp  SolveSucceeded          it=28   q=0   2nd=-                obj=-32393.22574
140c140
< lbfgs  dallass  nlp  ErrorInStepComputation  it=146  q=4   2nd=kept(1),tot=292  obj=-32388.50847
---
> lbfgs  dallass  nlp  SolveSucceeded          it=301  q=1   2nd=-                obj=-32393.22574
```

Exactly R3's two lines, with R3's numbers unchanged; nothing main brought in
interacts with the `.nl` merge. Lines with `mu=` other than `-` on `p-pr`
are the same four as at C1, none new from main:

- exact `csfi2` — `adaptive@Solved_To_Acceptable_Level/35,tot=56` (promoted; `it=21`)
- exact `mu_fallback_point_floor` — `kept(adaptive:Infeasible_Problem_Detected/31),tot=86`
- lbfgs `mu_fallback_point_floor` — `kept(monotone:Solved_To_Acceptable_Level/26),tot=46`
- lbfgs `eigenb2` — `kept(monotone:Solved_To_Acceptable_Level/42),tot=89`

`cargo fmt --all -- --check` clean; `cargo clippy --workspace --exclude
pounce-hsl --all-targets -- -D clippy::correctness -D clippy::suspicious`
exit 0 (warnings only, none denied); `cargo test -p pounce-nl -p pounce-nlp
-p pounce-solve-report -p pounce-algorithm -p pounce-cli --no-fail-fast`
2345 passed, 0 failed, 1 ignored. Not run: `vanderbei`/`benchmarks/qp`
(`POUNCE_BENCH_DATA` unset), `scripts/sweep-minima.sh` (no minima change).

### R4 — 2026-10-24 — not recoverable this way (measured)

Measured on x86_64 (4 CPUs, AVX2+FMA) with `p-prev` = release `pounce`
built from `59fec629` (PR #1048's head after merging main; this file and
the `mu=` column are in it) in a separate worktree and `CARGO_TARGET_DIR`,
on branch `claude/pounce-1042-loop-r4` (the loop's own branch is being
validated by another agent in parallel and was not touched).
`POUNCE_BENCH_DATA` unset, no `~/projects/pounce-bench-data` (checked once;
R4 does not need it). `p-base` (v0.12.0) was **not** rebuilt: the
Baseline log already records `p-base` at its default 2 = found 2 / 11 and
`p-base --gamma 0.5` = found 1 / 9 at seed 0, i.e. the loss is the
gamma value, and `p-prev --gamma 2` reproduces 2 / 11 exactly, so the
grid's `2.0` row stands in for 0.12.0's default on this binary.

**#993's rationale, checked (PR #993 item (f), issue #989 item 4).**
gamma 2 on a 2-D box gave a critical radius that left "a better sample
nearby" for almost every sample, so MLSL hit the sample cap after 2
solves on six-hump camel (2 of 6 minima at `max_solves 60, patience 15`);
0.5 finds all 6, and the CLI was aligned to the Python default, which
reads the same crate constant (`python/pounce/_minima.py` passes
`strategy_kw["gamma"]` through; the default lives only in
`MLSL_GAMMA`). Any recovery has to keep that: the camel / wavy columns
below are that contract measured across seeds.

**Reproduce (`p-prev`, sweep-minima.sh, seed 0).** `circle_parabola_t0p5
mlsl converged solves=9 found=1` — reproduces the inventory row. The other
mlsl lines: camel converged 12 / 4, haverly converged 10 / 2, wavy and
wavy_1e12 target_reached 6 / 6, the three saddle fixtures converged 8 / 0,
wavy_accept_anything converged 8 / 0 kkt_rej=8 (54 lines total).

**Measured: seeds 0–9 × gamma ∈ {0.4, 0.45, 0.5, 0.55, 0.6, 0.75, 1.0,
2.0}** (step 2's grid plus 0.4 / 0.45 / 0.55 to rule out a nearby winner
and 2.0 as the 0.12.0 reference), sweep budget `--n-minima 6
--max-solves 30 --patience 8`. Cell = mean minima found / mean solves over
10 seeds. All 400 runs: `dev-notes/issue-1042-r4-mlsl-gamma-grid.tsv`,
regenerated byte-identically by `dev-notes/issue-1042-r4-mlsl-gamma-grid.sh
<pounce> <out.tsv> [gammas]` (it reads the embedded models out of
`scripts/sweep-minima.sh`, so the two cannot drift); deterministic — two
independent runs of the grid were identical.

| gamma | camel | circle_parabola_t0p5 | haverly_t3_brf0 | wavy | wavy_1e12 |
|---|---|---|---|---|---|
| 0.4 | 6.0 / 7.4 | 1.5 / 10.0 (**5/10** find 2) | 2.0 / 10.2 | 6.0 / 6.0 | 6.0 / 6.1 |
| 0.45 | 5.9 / 8.0 | 1.5 / 10.0 (**5/10** find 2) | 2.0 / 10.4 | 6.0 / 6.1 | 6.0 / 6.1 |
| **0.5** (default) | 5.8 / 8.2 | 1.5 / 10.0 (**5/10** find 2) | 2.0 / 10.2 | 6.0 / 6.1 | 6.0 / 6.0 |
| 0.55 | 5.7 / 6.6 | 1.5 / 10.0 (**5/10** find 2) | 2.0 / 10.2 | 6.0 / 6.0 | 6.0 / 6.0 |
| 0.6 | 5.5 / 5.9 | 1.4 / 9.8 (**4/10** find 2) | 2.0 / 10.3 | 6.0 / 6.0 | 6.0 / 6.0 |
| 0.75 | 4.6 / 4.6 | 1.5 / 10.6 (**5/10** find 2) | 2.0 / 8.8 | 6.0 / 6.0 | 6.0 / 6.0 |
| 1.0 | 4.0 / 4.0 | 1.7 / 11.5 (**7/10** find 2) | 1.5 / 2.0 | 6.0 / 6.0 | 6.0 / 6.0 |
| 2.0 (0.12.0) | 2.0 / 2.0 | 1.7 / 10.6 (**7/10** find 2) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |

Stop status: at 0.5 camel is `target_reached` 9/10 (seed 0 `converged`
at 4), wavy / wavy_1e12 `target_reached` 10/10, circle and haverly
`converged` 10/10; at 1.0 camel and haverly are `sample_cap_reached`
10/10; at 2.0 every fixture except circle is `sample_cap_reached` 10/10 —
the gh#989 failure. circle_parabola is `converged` (stagnation at
`patience 8`) on all 80 runs: never budget, never sample cap.

**Decision (step 3).** (a) needs a value that is never worse than 0.5 on
camel / wavy mean found *and* recovers circle on most seeds. No value
does: circle is flat at 5/10 from 0.4 to 0.75 (4/10 at 0.6), and the only
values that lift it (7/10 at 1.0 and 2.0) take camel 5.8 → 4.0 / 2.0 and
haverly 2.0 → 1.5 / 1.0 — reintroducing exactly what #993 fixed. 0.12.0's
own default fails circle on 3 of 10 seeds, so the seed-0 line overstated
the regression: across seeds it is **7/10 → 5/10 seeds, mean found 1.7 →
1.5**, not "2 → 1". (b) is the structural stagnation change, which §3
since made its own item, **R4b**; §4 forbids both in one iteration, so R4
closes here. **gamma stays 0.5; no code, help text, docs, CHANGELOG or
Python default changed.**

**For R4b — what the strikes are.** The grid's `exit_*` columns count
per-solve `EXIT:` banners (they include one dual-recovery re-solve per
accepted minimum, m > 0; the identity `solves = exit_optimal +
exit_infeasible + exit_other − found` holds on all 80 circle and 80
haverly rows). "rej-opt" below is `exit_optimal − 2·found`: launches that
converged Optimal and were rejected — re-finds of an archived minimum, or
the saddle / bounds vetoes (`kkt_rej` is 0 throughout), so it is an
*upper* bound on re-finds. circle_parabola_t0p5, found / solves (rej-opt,
locally infeasible):

| seed | γ 0.5 | γ 1.0 | γ 2.0 |
|---|---|---|---|
| 0 | 1 / 9 (6, 2) | 2 / 15 (7, 6) | 2 / 11 (3, 6) |
| 1 | 1 / 9 (1, 7) | 2 / 14 (6, 6) | 1 / 9 (2, 6) |
| 2 | 1 / 9 (1, 7) | 1 / 9 (3, 5) | 2 / 11 (4, 5) |
| 3 | 2 / 10 (3, 5) | 2 / 10 (4, 4) | 2 / 10 (2, 6) |
| 4 | 1 / 9 (1, 7) | 1 / 9 (1, 7) | 2 / 15 (5, 8) |
| 5 | 2 / 11 (3, 6) | 2 / 11 (5, 4) | 2 / 11 (4, 5) |
| 6 | 1 / 9 (3, 5) | 1 / 9 (2, 6) | 1 / 9 (1, 7) |
| 7 | 2 / 10 (5, 3) | 2 / 10 (6, 2) | 2 / 10 (3, 5) |
| 8 | 2 / 11 (2, 7) | 2 / 16 (2, 12) | 1 / 9 (1, 7) |
| 9 | 2 / 13 (5, 6) | 2 / 12 (6, 4) | 2 / 11 (4, 5) |

On the five gamma-0.5 seeds that stop at found 1, the eight strikes are
6+2, 1+7, 1+7, 1+7, 3+5 (rej-opt + infeasible): re-finds dominate **only
seed 0**, the one seed every earlier table used. On seeds 1/2/4 seven of
eight strikes are launches that converged to a point of local
infeasibility (objective 2.0, the circle's infeasible stationary point).
R4b's measure-first gate ("re-finds dominate the stop on circle_parabola
on most seeds") reads as **unmet on this data** — R4b should confirm with
a direct trace count (`TraceEntry.accepted == false` with `success`)
before implementing, and its stop condition is the first thing to check.
Not decided here (R4b is its own iteration).

**Sweeps.** `scripts/sweep-minima.sh` was taken on `p-prev` (the numbers
above). No code changed, so the new binary *is* `p-prev`: the minima
sweep and `scripts/sweep-fixtures.sh` (both legs) diff empty by
construction and were not re-run.

**Tests.** `cargo fmt --all -- --check` (clean). Not run, and why: the
pounce-minima / pounce-cli tests and `issue989_mlsl_cli` (nothing they
exercise changed); the Python tests — the extension is not built in this
environment (no `pounce` module, no `maturin`, no `pytest`), and nothing
on the Python side moved (the default is the crate's, read by both
frontends).

**Next.** R4b (§2 item 7), starting from this table and the strike split
above. Sibling rule: R4 did not move gamma, so R4b is not re-baselined —
its reproduce step is this log's seed-0 row (found 1 / 9, `converged`).

### R4b — 2026-10-24 — not recoverable this way (measured)

Measured on x86_64 (4 CPUs, AVX2+FMA), branch `claude/pounce-1042-loop-r4`
at `e84acb08`. `p-prev` = release `pounce` built from `e84acb08` in its own
worktree and `CARGO_TARGET_DIR`; the trace binary is `p-prev` plus a
**print-only** patch (never committed) that, when `R4B_TRACE` is set, prints
one line per `consider` call — `success`, `accepted`, `f`, `in_bounds`,
`is_minimum`, `is_known`, `archive.len()`, `stagnant`, `x` — and one per
MLSL launch (pool size, radius, sample). That is the content of
`TraceEntry` plus the reason a candidate was rejected; the CLI does not
expose `MinimaResult::trace`, so the patch is the direct count the R4 log
asked for. Each `consider` line was classed by the `EXIT:` banner of the
solve just before it: **A** accepted; **R** re-find (`success`,
`is_minimum`, in bounds, `is_known`); **I** / **i** a launch ending
`Converged to a point of local infeasibility`, at a new / an already-seen
landing point (within 1e-3); `o` / `f` any other rejection / failure (none
occurred). `POUNCE_BENCH_DATA` unset (checked once; not needed).

**Reproduce (`p-prev`, seed 0).** `circle_parabola_t0p5 --minima mlsl
--n-minima 6 --max-solves 30 --patience 8 --seed 0 --gamma 0.5` →
`converged`, 9 solves, found 1 — the inventory row. The trace binary gives
the same status / solves / found on all 50 grid runs as R4's gamma-0.5
rows.

**Trace count — the stop (gamma 0.5, budget as above).** Strikes since the
last acceptance, i.e. the eight that tripped `patience`:

| seed | found / solves | re-finds | infeasible, first hit | infeasible, repeat | other |
|---|---|---|---|---|---|
| 0 | 1 / 9 | **6** | 1 | 1 | 0 |
| 1 | 1 / 9 | 1 | 1 | **6** | 0 |
| 2 | 1 / 9 | 1 | 1 | **6** | 0 |
| 3 | 2 / 10 | 3 | 1 | 4 | 0 |
| 4 | 1 / 9 | 1 | 1 | **6** | 0 |
| 5 | 2 / 11 | 3 | 0 | 5 | 0 |
| 6 | 1 / 9 | 3 | 1 | 4 | 0 |
| 7 | 2 / 10 | 5 | 1 | 2 | 0 |
| 8 | 2 / 11 | 2 | 0 | 6 | 0 |
| 9 | 2 / 13 | 5 | 0 | 3 | 0 |

This is the R4 log's EXIT-line split (6+2, 1+7, 1+7, 1+7, 3+5 on the
failing seeds) confirmed exactly: the EXIT upper bound was tight, every
rejected-Optimal launch is a re-find (no saddle or bound veto fired). New
here: every locally infeasible launch on every seed lands on **one** point,
(0, −1) — one distinct infeasible landing point per run, 10/10 seeds.
Elsewhere, with found at the target or the model's full set: camel
re-finds 0–5 per run on the nine `target_reached` seeds and 8 of 8 on seed
0 (`converged` at 4 of 6); haverly **8 of 8 re-finds on all 10 seeds**
(found 2 every time; at unbounded patience 28 of 30 solves re-find and
found stays 2); wavy / wavy_1e12 0–1. No grid run outside circle has an
infeasible launch, and the sweep's three saddle fixtures (seed 0) reject
8 of 8 on `is_minimum`, none infeasible.

**Gate.** "Re-finds dominate the stop on circle_parabola on most seeds and
are rare where found already equals the target": re-finds dominate on
**1 of the 5** failing seeds (seed 0), and they *are* the whole stop where
the search is already complete (haverly, 10/10). Both clauses fail. **R4b
is not recoverable this way**; nothing implemented, gamma 0.5.

**Replay (what the counting half would have done).** Changing how a
rejection is counted does not change samples or solves, so the traces at
unbounded patience (`--patience 1000`, budget 30) replay any weighting
exactly; the unweighted replay reproduces R4's gamma-0.5 row digit for
digit. R4b's (ii) at half weight: circle 5 → 7/10 (mean solves 10.0 →
14.5), camel 5.8 → 6.0 (seed 0's two hidden minima), **haverly 10.2 →
18.2 mean solves (+78%)**, over the item's 20% — the re-find discount
spends budget exactly where the search is done. At weight 0, haverly runs
to the 30-solve budget on every seed. R4b's (i), the radius growth, changes
the trajectory and cannot be replayed; it was not built, because the gate
is the precondition and it does not speak to the 28 infeasible strikes at
all (they never reach an archived minimum, so no radius grows). Full
replay table in §3 R4c. Per-run data, including each run's full
unbounded-patience event string (`A R I i`):
`dev-notes/issue-1042-r4b-mlsl-strike-trace.tsv`.

**Where the data points: R4c (added to §1, §2, §3).** 28 of the 40
strikes on the failing seeds are hits of one infeasible point (5 first
hits, 23 repeats); with patience unbounded the second minimum is accepted
at solve 11, 10, 17, 16, 14 on seeds 0, 1, 2, 4, 6 — inside the budget.
Replayed, not counting repeat hits of a seen infeasible point (`w = 0`)
takes circle to **8/10** and leaves camel / haverly / wavy / wavy_1e12
identical (they have no infeasible launches); `w = 0.5` reaches only 6/10.
Seed 0 stays `found=1` under it (re-find-limited), so R4c's acceptance is
the 10-seed table, not the seed-0 sweep line. One API finding for R4c:
`LocalOutcome` carries no termination status, so the driver cannot see
"locally infeasible" today — §3 R4c names the two ways to get it.

**Sweeps.** No code changed: the new binary is `p-prev`, so
`scripts/sweep-minima.sh` and `scripts/sweep-fixtures.sh` (both legs) diff
empty by construction and were not re-run; the seed-0 minima numbers above
are `p-prev`'s.

**Tests.** `cargo fmt --all -- --check` (clean). Not run, and why:
pounce-minima / pounce-cli / `issue989_mlsl_cli` and clippy (no code
changed); the Python tests (extension not built here — no `pounce` module,
`maturin` or `pytest`; nothing on the Python side moved).

**Next.** R4c (§2 item 7, after R4b), starting from its measure-first
step 2 (the classifier check). Sibling rule: nothing landed, so R4c is not
re-baselined; its reproduce step is the seed-0 row above and the circle
rows of the trace table.

### R4c — 2026-10-24 — done

Measured on x86_64 (4 CPUs, AVX2+FMA), branch `claude/pounce-1042-loop-r4`
at `a0ed089a`. `p-prev` = release `pounce` built from `a0ed089a` in its own
worktree and `CARGO_TARGET_DIR`; `p-new` = this commit, built the same way.
`POUNCE_BENCH_DATA` unset (checked once; not needed — R4c touches only
`pounce-minima`). Sibling rule: nothing in the family had landed, so R4c is
not re-baselined.

**Reproduce (`p-prev`).** `scripts/sweep-minima.sh` line
`circle_parabola_t0p5 mlsl converged solves=9 found=1`, and
`dev-notes/issue-1042-r4-mlsl-gamma-grid.sh p-prev … 0.5` reproduces the
committed grid's 50 gamma-0.5 rows **byte-identically** (circle 5/10, mean
1.5 found / 10.0 solves).

**Measure first — the classifier (step 2).** Route (b), driver-side from
the point: a failed launch whose landing `x` violates the base problem's
constraint bounds by more than `1e-4` (Ipopt's default `constr_viol_tol`;
one `eval_g`) is "infeasible". Checked with a print-only patch on
`run_solve` (never committed) over the 10-seed grid of all five fixtures at
`--patience 8` **and** `--patience 1000`, plus every one of the
`sweep-minima.sh` runs (9 fixtures × 6 methods, seed 0) — 1848 solves, each
paired with the `EXIT:` banner of its own solve:

| exit banner | converged | n | landing violation | flagged by (b) |
|---|---|---|---|---|
| Converged to a point of local infeasibility | no | 229 | 1.5 (all) | **229** |
| Maximum Number of Iterations Exceeded (haverly, deflation) | no | 1 | 5.3e-5 | 0 |
| Error in step computation (haverly, tunneling) | no | 1 | 1.6e-10 | 0 |
| Search Direction is becoming Too Small (wavy_1e12, `m = 0`) | no | 17 | 0 | 0 |
| any (Optimal, Acceptable, …) | yes | 1600 | ≤ 2.0e-6 | 0 (not a failure) |

Confusion: 229 true positives, 0 false positives, 0 false negatives. The
nearest feasible failure (5.3e-5) is under the threshold by 1.9×; it is a
deflation solve, a method the rule does not apply to. Route (a) (a status
field on `LocalOutcome`) was therefore not needed: no public API change, and
`crates/pounce-py`'s `PyLocalSolve` delegates to `IpoptLocalSolve` and calls
the same `find_minima_with`, so the Python `find_minima` gets the rule from
the crate with nothing to fill.

**Distinct-point check (step 3).** Every run with an infeasible launch has
**one** distinct landing point — (0, −1), spread ≤ 6e-18 within a run — on
all 20 grid circle runs (both patiences) and the four sweep circle lines
that have one (multistart 2, mlsl 2, deflation 4, tunneling 1). No fixture
exercises the first-hit clause; the unit test below does.

**Implemented.** `Driver::is_repeat_infeasible_landing`
(`crates/pounce-minima/src/search.rs`): under `Method::Mlsl`, a rejected
failed launch that is infeasible by the test above and lands within `dedup`
(the dedup metric, `Hooks::distance` when given) of a recorded infeasible
landing costs **no** strike (`w = 0`, from the replay table: 0.5 reaches
only 6/10), once a minimum is archived. A first hit is recorded and is a
full strike; before any minimum is archived every hit is a strike, so a
model infeasible from everywhere stops at `patience` rather than spending
`max_solves` (default `8 × n_minima`). `stagnant` stays an integer count;
no float. Infeasible landings are never archived, reported or candidates.

**Per method (step "decide per method").** Multistart was measured with the
rule on (`circle_parabola_t0p5 --minima multistart`, same budget, seeds
0–9): found both on 7/10 → 7/10, mean solves 10.8 → 14.5 (+34%), sweep
line 10 → 13 solves found=2 unchanged — cost, no gain, so the rule is
**MLSL only**. The other four methods' next start depends on the last
result and their failed solves are on an augmented problem; not applied.

**10-seed table (gamma 0.5, `--n-minima 6 --max-solves 30 --patience 8`;
R4's grid script).** camel / haverly / wavy / wavy_1e12: all 40 rows
**byte-identical** to `p-prev` and to the committed grid. circle, per seed
(status solves found):

| seed | `p-prev` | `p-new` | replay (R4b trace, `w = 0`) |
|---|---|---|---|
| 0 | converged 9 1 | converged 10 1 | converged 10 1 |
| 1 | converged 9 1 | budget_exhausted 30 **2** | budget_exhausted 30 2 |
| 2 | converged 9 1 | converged 16 1 | converged 16 1 |
| 3 | converged 10 2 | converged 15 2 | converged 15 2 |
| 4 | converged 9 1 | budget_exhausted 30 **2** | budget_exhausted 30 2 |
| 5 | converged 11 2 | converged 17 2 | converged 17 2 |
| 6 | converged 9 1 | converged 30 **2** | converged 30 2 |
| 7 | converged 10 2 | converged 12 2 | converged 12 2 |
| 8 | converged 11 2 | converged 19 2 | converged 19 2 |
| 9 | converged 13 2 | converged 16 2 | converged 16 2 |

| fixture | `p-prev` mean found / solves | `p-new` |
|---|---|---|
| camel | 5.8 / 8.2 | 5.8 / 8.2 |
| circle_parabola_t0p5 | 1.5 / 10.0 (**5/10**) | 1.8 / 19.5 (**8/10**) |
| haverly_multistart_t3_brf0 | 2.0 / 10.2 | 2.0 / 10.2 |
| wavy | 6.0 / 6.1 | 6.0 / 6.1 |
| wavy_1e12 | 6.0 / 6.0 | 6.0 / 6.0 |

The built rule reproduces the R4b-trace replay on **all 50 runs** (the
replay of the committed unbounded-patience event strings, re-run here
against the built binary's grid). Seeds 0 and 2 stay at found 1:
re-find-limited, as R4c's acceptance said up front. Two seeds now end
`budget_exhausted` at 30 solves after finding both minima — once both are
archived, the remaining strikes are re-finds and the repeat infeasible
hits no longer stop the search; that is the price of the recovery the
item named (circle mean solves ≤ 30 by construction), not collateral.
Against 0.12.0's gamma 2 (7/10, 1.7 / 10.6): above it on seeds found, at
about twice the solves, with camel / haverly / wavy kept at #993's levels.

**`scripts/sweep-minima.sh` `p-prev` vs `p-new` (54 lines):**

```
10c10
< circle_parabola_t0p5         mlsl         converged          solves=9 found=1 kkt_rej=0 values=[8.155059653] x=[0.5682214846,0.8228756555] lambda=[-1.846686857,-2.606563995] zL=[] zU=[]
---
> circle_parabola_t0p5         mlsl         converged          solves=10 found=1 kkt_rej=0 values=[8.155059653] x=[0.5682214846,0.8228756555] lambda=[-1.846686857,-2.606563995] zL=[] zU=[]
```

One line moves, on the one model with infeasible launches: seed 0's event
string is `A I R R i R R R R R …`; the `i` at launch 5 (the second hit of
(0, −1)) no longer counts, so the eighth strike lands at solve 10 instead
of 9. found, values, x and duals unchanged. Every other line (the other 53,
including the circle lines of the five other methods, the three saddle
`found=0` lines and the `kkt_rej=8` line) diffs empty.

**`scripts/sweep-fixtures.sh` `p-prev` vs `p-new`:** 242 lines (121 exact,
121 lbfgs), **empty** — the single-solve path is untouched.

**Tests.** New in `crates/pounce-minima/tests/search.rs`, over a scripted
`LocalSolve` on a constant-objective box with one row (MLSL launches every
pooled sample): `mlsl_repeat_infeasible_landings_do_not_exhaust_patience`
(one infeasible attractor, second minimum first drawn at launch 40 — MLSL
reaches `target_reached`, multistart stops `converged` at 1),
`mlsl_distinct_infeasible_landings_still_exhaust_patience` (first-hit
clause: stops at `1 + patience`), `mlsl_repeat_feasible_failures_still_exhaust_patience`
(a repeated *feasible* failure keeps its strike),
`mlsl_with_nothing_found_counts_every_infeasible_landing` (empty-archive
guard: stops at `patience`). Mutation-checked, each mutation turning
exactly one test red: rule off → the first; first hit also exempted → the
second; violation test dropped (any failure "infeasible") → the third;
empty-archive guard dropped → the fourth; multistart included → the first
(its multistart half). `cargo test -p pounce-minima -p pounce-cli`: 171
test binaries, 966 passed, 0 failed (incl. `issue989_mlsl_cli`,
`minima_sol_duals`, `minima_bound_multiplier_suffixes`). `cargo fmt --all
-- --check` clean; clippy (§4 command) passes (the new test file's
`unwrap_used` warnings are the file's existing style). `cargo check -p
pounce-py` passes (not touched). **Not run:** the Python tests,
`python/tests/test_minima.py` and `test_minima_cli_parity.py` — the
extension is not built here (no `maturin`; `import pounce` fails; `pytest`
is present). Parity holds by construction (same crate entry point, same
`LocalSolve` adapter), but it was not executed.

**Docs.** `docs/src/find-minima.md` (Termination), `docs/src/find-minima-cli.md`,
CHANGELOG `[Unreleased] / Changed`. No default moved (gamma 0.5, patience 8),
so the CLI help and `python/pounce/_minima.py` are unchanged.

**Next.** §2 item 8, **R1d** (cut the first, unequilibrated HSDE solve
short) — convex arm, so it can at most reach `swept — needs benchmarks/qp
before merge` here. The R4 family is closed by R4c.

### Vanderbei A/B: `dallasl` / `dallasm` — 2026-10-24 — x86_64

Models supplied by the owner; binaries: v0.12.0, main `68230fab`, PR
`2864a85a` (the later merge of R4c touches only `pounce-minima`). Every PR
run is a single attempt (`mu_strategy_fallback` null, no second opinion).

| model | leg | v0.12.0 | main | PR |
|---|---|---|---|---|
| dallasl | exact | Acceptable/49, mu retry lost (tot 77) | Optimal 145, 10 resto | **Optimal 28, 0 resto** |
| dallasl | lbfgs | ErrorInStep 495 | Optimal 51 (−202577.38) | Optimal 146 (−202603.62) |
| dallasm | exact | ErrorInStep 32 | Optimal 314, 26 resto | **Optimal 22, 0 resto** |
| dallasm | lbfgs | ErrorInStep 377 | ErrorInStep 332 + ladder (tot 664) | **Optimal 22** |

The merge fires on every term (872 in dallasl, 182 in dallasm, 0 on main);
non-finite Hessian evaluations go 9 → 0 and 26 → 0. Attribution controls are
bit-identical: the PR binary on an unmerged copy reproduces main's tables,
and main on a hand-merged copy reproduces the PR's. The one slower line,
dallasl lbfgs 51 → 146, splits from main at iteration 4 on a rounding-level
difference; perturbing one start value by 1e-14 moves that leg 51–345 on main
and 22–380 on the PR, while the exact leg stays 22–30 with no restoration on
the PR (main 40–314, always restoring). Not measured: arm64.
