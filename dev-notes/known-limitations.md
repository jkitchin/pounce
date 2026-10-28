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

## NLP: qcqp1500-1nc stalls at μ = 2e-6; waterx lands in a worse minimum

*Tracked in gh#1098. Introduced by 49ab28aa (#1077, gh#1076 §2). Recorded
2026-10-28 while preparing 0.14.0.*

**Scope.** Two benchmark models on the default exact-Hessian NLP path. The
L-BFGS leg solves `qcqp1500-1nc` (106 iterations, 32 s). Neither model is in
the fixture corpus.

### What is wrong

#1077 made restoration's release test compare the outer filter's own
quantities (1-norm `θ`, scaled barrier objective) where it had compared the
max-norm `θ` and the plain `f`. That brought pounce onto Ipopt's trajectory,
and on these two models Ipopt's trajectory is worse than the one the
mismatched test produced by accident:

- **qcqp1500-1nc** (Mittelmann, 1500 variables, 10 508 constraints). 0.13.0
  released restoration at iteration 2 and converged in 77 iterations. 0.14.0
  stays in restoration to iteration 8, as Ipopt does, and matches Ipopt's
  iterates to about five digits from there through at least iteration 40.
  From about iteration 140 it sits at μ = 2e-6 for 1200 iterations: `inf_pr`
  oscillates between 1e-3 and 1e-2 and never meets the barrier tolerance,
  steps are long with tiny `alpha_pr` and `alpha_du`, and `δ_w` is non-zero
  almost every iteration. Ipopt crawls the same way.
- **waterx** (Water suite). Same 249 iterations, objective 927.1 → 1019.7
  (Ipopt 944.6). At iteration 146 restoration no longer hands back at once.

### How big

| model | 0.13.0 (`9958a24e`) | 0.14.0 (`c082fe76`) | Ipopt 3.14.20 MA57 |
|---|---|---|---|
| qcqp1500-1nc | Optimal, 77 it, 39 s | CPU limit at 300 s; uncapped Optimal, 1346 it, 951 s, f 0.019% higher | CPU limit at 1800 s, 2827 it |
| waterx | Optimal, f = 927.1 | Optimal, f = 1019.7 | Optimal, f = 944.6 |

Arm64 timings. Uncapped, `qcqp1500-1nc` ran alongside other work, so its 951 s
is high. The Mittelmann suite drops from 45/47 to 44/47; nothing else in the
1345-model benchmark lost a solve.

### Already tried, did not work

On `qcqp1500-1nc`, 0.14.0 binary, 400 s limit:

- **Turning SOC off is luck, not a fix.** `max_soc=0` solves in 148
  iterations (Ipopt with `max_soc=0`: 116). But it matches the default run
  exactly up to iteration 77, where the default accepts one second-order
  correction (`H`) that leads into the slow basin. During the crawl, SOC is
  accepted on only 4 of about 1200 iterations, so a rule that disables SOC
  when the solve stalls would not fire where it matters.
- **Lowering μ sooner.** `barrier_tol_factor=100` and `=1000` reach
  μ = 2e-6 earlier, then stall there for 430 and 375 iterations.
- **Other options.** `mu_strategy=adaptive` (with or without
  `mu_oracle=quality-function`), `mu_init=1e-4` and
  `watchdog_shortened_iter_trigger=0` were all still unsolved at 400 s.

Reverting #1077's test is not on the table: it restores a comparison of
mismatched quantities, and #1077 measured what that costs elsewhere.

### What a fix would look like

The suspected cause is untested. With 10 508 constraints on 1500 variables,
more constraints may be active near the solution than there are variables, so
the active Jacobian would be rank-deficient (LICQ fails). That would explain
steps that cannot reduce `inf_pr`, tiny step fractions and persistent
regularization.

**Suggested first step:** at a crawl iterate, around iteration 300, count the
active constraints, measure the numerical rank of their Jacobian, and check
whether the KKT solve ever applies `δ_c` or only `δ_w`. If the Jacobian is
degenerate, the fix is in how a rank-deficient constraint block is regularized
at fixed μ. That would make pounce better than Ipopt here, not just equal to
it.

**Acceptance:**

1. `qcqp1500-1nc` solves within 300 s on the exact-Hessian path, without
   bringing back #1077's mismatched release test.
2. The Mittelmann suite is otherwise unchanged
   (`make -C benchmarks mittelmann-rerun`), and `waterx` is re-measured.
3. `scripts/sweep-fixtures.sh`, both legs, with every moved line explained.

### When to revisit

- A user reports a large QCQP stalling at a fixed μ with `inf_pr` near 1e-3.
- The restoration release test or the inertia and regularization handling
  changes. Re-run both models.

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
