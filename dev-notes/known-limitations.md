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

Nothing beyond `complementarity_strategy=continuation`, which skips the stage
and is the workaround for a model known to behave this way. The stage is kept
in the default because on the Gate 0 corpus and the CALIPSO bunny-hop it finds
answers the continuation misses (see `docs/src/complementarity.md`).

### How to tell whether a fix works

`qpec-100-2` under `complementarity_strategy=exact` should take a small
fraction of 561 iterations without changing its objective (time `qpec-200-2`
per stage first if it is to be the second witness), while the Gate 0 corpus
(`benchmarks/mpcc/declared.py`) stays at 60/60 cells at `f*`. Integrated
approaches that tie the relaxation to the barrier parameter (relaxing the
products by an amount proportional to μ inside one solve) avoid the empty
interior and are the obvious direction. The MacMPEC suite and its harness are
in the benchmark data root (`macmpec/`).

---

## NLP: qcqp1500-1nc crawls at the default; the fix is opt-in (`perturb_slack_fact=0.03`)

*Tracked in gh#1098. The crawl came in with 49ab28aa (#1077), which put
pounce on Ipopt's trajectory; Ipopt crawls the same way. Recorded
2026-10-28, revised 2026-10-31.*

**Scope.** The default exact-Hessian NLP path. The L-BFGS leg solves
`qcqp1500-1nc`.

### What is wrong

At the default, Mittelmann `qcqp1500-1nc` sits at μ ≈ 1.8e-6 for about
1200 iterations and hits the 300 s CPU limit. `qcqp1500-1c` behaves the
same. `waterx` reaches f = 1019.7, where Ipopt reaches 944.6. The diagnosis
is in `dev-notes/qcqp1500-crawl-slack-perturbation.md`: inertia correction
puts `δ_s = δ_x` on the slack block, and the fraction-to-the-boundary rule
pushes centered blocking pairs to `s·v ≈ μ²`.

### How big, and the opt-in

`perturb_slack_fact=0.03` scales the slack-block perturbation to
`0.03·δ_x`. Measured against main `26991b7b`, three lanes at once, so the
timings are approximate:

| | main | `perturb_slack_fact=0.03` |
|---|---|---|
| Mittelmann solved | 44/47 | 45/47 |
| Mittelmann total iterations | 5182 | 4355 |
| qcqp1500-1nc | CPU limit | Optimal, 228 it |
| qcqp1500-1c | CPU limit | CPU limit |
| henon120 / lane_emden120 | 267 / 84 | 80 / 25 |
| qcqp750-2c / dirichlet120 | 75 / 56 | 198 / 79 |
| water4 / watersbp | 137 / 164 | 213 / 233 |
| waterx | f = 1019.7 | f = 1151.5 |
| fixture sweep | — | 14/252 lines move, no status loss, no new retry |

`ftb_centrality_floor=0.01` together with `perturb_slack_fact=0.01` also
solves `qcqp1500-1c` (148 it), but it turns the `discs` fixture into a false
`InfeasibleProblemDetected`.

### Why it is not the default

At 0.03, 22 `pounce-cli` regression tests that pass on main fail:

- **About 10 are premise tests.** Their fixture, mostly `csfi2`, now
  converges on the first attempt, so the retry or fallback they pin
  (gh#748, #757, #1042, #616, one-verdict-per-run, #534, #592) never fires.
- **About 6 assert an answer:**
  - `false_local_infeasibility::hs13_from_remote_start_reaches_the_optimum`
    gets 0.98168 where Ipopt gets 0.98493.
  - `issue_250_dual_guard_never_worse` reads `autocorr_bern55-06` at −2368
    as a worse basin than −2304, and its control fixture becomes
    infeasible.
  - `mpcc_qpec_small_biactive` fails in the #884 and #981 tests.
  - `issue_1000` and `issue857` also move.

### What a fix would look like

Making 0.03 the default means triaging those 22 tests:

- Re-point each premise test to a fixture that still reaches its branch.
- For each answer test, decide whether the new answer is a regression.

A per-row `δ_s` scaled by `Σ_s`, which regularizes only the slacks that
need it, might move fewer of them, and might also solve `qcqp1500-1c`.

Acceptance:

1. `qcqp1500-1nc` solves within 300 s at the default.
2. The workspace tests pass, with any re-pointed fixture explained.
3. `scripts/sweep-fixtures.sh` on both legs, diffing the `2nd=` column as
   well.

### When to revisit

- The inertia-correction or `δ_s` handling changes.
- A user reports a large QCQP stalling at a fixed μ with `inf_pr` near 1e-3.

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
