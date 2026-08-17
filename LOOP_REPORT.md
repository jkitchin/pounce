VERDICT: improved

Fixes #617
Fixes #618

Two #606 follow-ups in the same function region of
`crates/pounce-algorithm/src/init/warm_start.rs`, measured on the same
corpus in overlapping seed modes. Dispatching them separately would have
guaranteed a conflict and a re-measure, so they are one branch and one PR.

All numbers in this body were taken on the parent commit
**`b4c4d32e`** (`Merge pull request #633`) and on this branch's head
`d58ffe98`, on the same machine, in the same session. `#630` and `#633`
both merged to `main` while this work was in flight; the branch was
merged forward onto each and every measurement re-taken afterwards.

## Problem

Both issues' numbers are stale — they were filed against
`70bf53ded3d893cfa2da6ead5195fda5ac096f68`, and `main` has since taken
#613, #619, #614, #623, #620, #628, #627, #629, #626, #630 and #633.
Both were reproduced on current `main` before anything was designed,
using the attribution control both issues name: `warm_start_recentering=none`
restores pre-#606 behaviour exactly, so #606 is isolated at a single
commit rather than across `main`'s drift.

Neither regression has gone away. Both are **larger** than filed.

`benchmarks/warmstart`, 14 default-tier families, `large` scale (the 4x
path both issues specify), summed iterations over the corpus:

| seed | pre-#606 (`none`) | parent `b4c4d32e` | this branch |
|---|--:|--:|--:|
| exact restart, full duals | 13 | 9 | **9** |
| exact restart, `lagrange` only | 24 | 11 | **11** |
| exact restart, primal only | 24 | 24 | **24** |
| **corrupted duals, full** | 33 | **164** | **25** |
| **corrupted duals, `lagrange` only** | 27 | **115** | **20** |
| corrupted duals, primal only | 24 | 24 | **24** |
| stale seed, full | 344 | 351 | **354** |
| **stale seed, `lagrange` only** | 273 | **339** | **279** |
| stale seed, primal only | 255 | 255 | **255** |
| cold (no seed) | 166 | 166 | 166 |
| **total, warm cells** | **1017** | **1292** | **1001** |

### #617 as filed vs. as reproduced

| | filed (9 families, `70bf53d`) | reproduced (14 families, `b4c4d32e`) |
|---|--:|--:|
| before #606 | 18 | 33 |
| after #606 | 71 | **164** |
| cold (no seed) | 81 | 166 |

Worst single family as filed was `hanging_chain` 3 → 13; reproduced it is
`hanging_chain` 4 → 17, with `simplex_proj` 2 → 19 and `mpc_horizon_80`
3 → 23 worse still. Status is `SolveSucceeded` throughout, at the same
objective — a slowdown, not a correctness failure, exactly as filed.

### #618 as filed vs. as reproduced

| | filed (9 families, `70bf53d`) | reproduced (14 families, `b4c4d32e`) |
|---|--:|--:|
| before #606 | 118 | 344 |
| after #606 | 127 | **351** |

**The per-family regressions #618 names have all gone.** On current
`main`, `redundant_rows` is 4 → 4, `degenerate_corner` 9 → 9,
`moving_bound_qp` 29 → 29, and `hanging_chain` 18 → **14** — it now
*improves*. The +7 that remains is a different set of families, and it is
dominated by one: `degenerate_vertex` 2 → 12.

The larger instance of #618's phenomenon is in a cell #618 did not
measure — a `lagrange`-only stale seed, +66 (273 → 339), concentrated in
`mpc_horizon_40` (45 → 81) and `mpc_horizon_80` (53 → 77). This is
#620's single-block fix reaching models with one constraint block, the
same mechanism as the `nmpc_vanderpol @ large` 195 → 209 datapoint added
to #618 in comment. That family is in the corpus above and its stale
`lagrange`-only seed moves 42 → 36 on the parent (a #606 *win*).

## Cause

### #617 — the escalation, not the reconstruction

#617 attributes the regression to `v_L`/`v_U` being reconstructed off a
corrupted `y`. The diagnostics say otherwise, and say it unanimously:
**every** corrupted regression in the corpus ends at `mu_out = 0.1`.

The corrupted multipliers read back as an average complementarity of
`2e-1` to `4.4e+2`. `final_mu`'s escalation takes that at face value and
moves μ eight orders — from the converged barrier to `MU_CEILING` —
while `x` stays at the exact solution. That is the most off-centre a
point can be for the barrier it has just been given: the slacks are
sized for `μ ≈ 2.5e-9` and the barrier now claims `0.1`. Hence the
result costing about what a cold solve costs, and on `simplex_proj`
more (19 against a cold 11).

On families whose only bound multipliers are seedable variable bounds
(`mpc_horizon_*`, `nmpc_vanderpol` at a full seed) *nothing at all* is
reconstructed — `bound_duals: accepted`, `bound_duals_reconstructed: 0`
— and they still regress 3 → 23. The reconstruction is not the
mechanism there; the barrier move is the whole of it.

### #618 — the reconstruction reads the seed's active set

Both halves of the bound reconstruction treat a small slack as "this
bound is active": `μ̂ / slack` sizes the multiplier from it, and the
stationarity split decides which bound carries the residual from it. On a
seed carried across a parameter move that is a claim about a bound that
may since have released. On `mpc_horizon_40`, `nrec = 244` — every
variable bound — with `mu_in = mu_out = 2.51e-9`: μ never moves, and the
entire 45 → 81 is the reconstruction rerouting the trajectory.

That is #618's diagnosis exactly ("stale enough that dual reconstruction
fires and reroutes the trajectory, but not stale enough to trip the
escalation that is supposed to pay for it") — but located in the
reconstruction rather than in μ.

## Fix

Three guards for #617 and one for #618, all at the same 10x conservatism
`MU_ESCALATION_TRIGGER` already uses, because refusing a seed reroutes
the trajectory exactly as much as trusting a bad one does.

1. **A seeded bound-multiplier block is refused when its implied
   complementarity cannot belong to its primal point.** Measured on
   `|z_i| · s_i` over the entries the caller actually seeded — the
   quantity the barrier *is*, and the machinery #617 correctly says
   already exists — and refused when that reads ten times above **both**
   `mu_init` and the point's own `inf_pr`. `inf_pr` is in that max
   because a point that misses feasibility by that much is *allowed*
   complementarity of that order; it is what keeps the strongly-stale
   case out of the test. A refused block takes the pre-#606 constant
   fill and stops being an input to anything, which drops `avrg_compl`
   back to the barrier and takes the escalation with it — no separate μ
   rule was needed. `|z_i|` and not `z_i`: a corrupted block is roughly
   half negative and a signed average cancels to nothing.
2. **A seeded `y` that does not belong to the point stops feeding the
   split.** Refused when its stationarity residual dwarfs `∇f` and the
   bound multipliers *the caller supplied*. The block itself is left
   where the caller put it — that is what the pre-#606 path does with a
   supplied `y`, and there is no constant to fall back to — but nothing
   is derived from it.
3. **The split may not raise a multiplier by orders.** It may raise it
   above the complementarity floor — the slack push's inflation put
   #606's reconstructed HS071 multiplier 5.5x low, so the margin is real
   — but it is now capped at ten times that floor.
4. **(#618) A slack the point's own primal residual swamps is not a
   measurement.** Those entries keep the pre-#606 constant. Per-entry,
   not a per-point gate, so a partly-stale seed keeps its reconstruction
   wherever the slacks still outrun the residual — the reconstruction's
   reach scales down with the measurement instead of switching off at a
   threshold, which is the second remedy #618 offers.

### What was tried and rejected

* **#618's first proposed remedy — a gentler μ escalation band below the
  10x trigger — is inert, and was measured rather than assumed.** Every
  stale regression remaining on current `main` has a measured
  complementarity within **1.2x to 3x** of `mu_init`
  (`degenerate_vertex` 2.986e-9 against 2.506e-9 — a 1.19x overshoot;
  `mpc_horizon_40` 7.54e-9 against 2.51e-9). A band starting at 1x has
  nothing to raise μ *to*.
  Separately, it cannot be built without moving `cresc4`, which sits at
  5x — the band would have to exclude the very ratio that motivates it.
* **Scaling the complementarity fill by `μ̂ / max(slack, inf_pr)`**
  instead of falling back to the constant: measured, and worth −1 on
  stale/full and +11 on stale/`lagrange`-only. Rejected; the constant
  fallback is both simpler and better.
* **Comparing slacks against `inf_pr` unguarded**: fires on *converged*
  points, whose `inf_pr` sits at solver tolerance and routinely exceeds
  the slacks `warm_start_slack_bound_push` has shoved to 1e-9. It called
  an exact restart's active bounds swamped and cost the exact partial
  restart 11 → 15. Guarded by requiring `inf_pr` to clear the barrier
  tenfold first.
* **Scaling the `y` test by all four bound-multiplier blocks** including
  the ones this same pass reconstructed: a `μ̂ / slack` fill at a tight
  bound is large by construction, and on `nmpc_vanderpol` it was
  vouching for the very seed it had been derived from — 12 iterations
  against a cold solve's 12. The scale now uses only caller-supplied
  multipliers.
* **A sign test** (`z_i < 0` proves corruption) separates this corpus
  perfectly — 0.00 negative fraction on every exact and stale seed,
  0.48–0.88 on every corrupted one. Rejected as fitting the fixture:
  the corruption model here is symmetric Gaussian, and an all-positive
  corruption would defeat it while leaving the magnitude problem intact.

### A caveat stated rather than hidden

Guard 4 compares a slack against `inf_pr`. For the slack-bound blocks
those are the same units. For *variable*-bound blocks they are not — the
link from constraint-space error to variable-space position error goes
through the Jacobian, so this is an order-of-magnitude argument with
`‖J‖ ~ 1` assumed, in the same spirit as the rest of the module. It is
the case guard 4 helps most (`mpc_horizon_*`), and it is worth knowing
that the reasoning there is scale-approximate.

## Results

### #617 — every family, at or below pre-#606

Corrupted seed, full duals, per family (pre-#606 / parent / branch):

| family | pre-#606 | parent | branch |
|---|--:|--:|--:|
| `simplex_proj` | 2 | 19 | **2** |
| `moving_bound_qp` | 3 | 12 | **3** |
| `degenerate_corner` | 2 | 12 | **2** |
| `redundant_rows` | 3 | 7 | **2** |
| `degenerate_vertex` | 1 | 5 | **1** |
| `hanging_chain` | 4 | 17 | **3** |
| `rosenbrock_ring` | 2 | 5 | **1** |
| `rosenbrock_ring_cycle` | 2 | 5 | **1** |
| `double_well_chain` | 0 | 0 | 0 |
| `nmpc_vanderpol` | 3 | 14 | **2** |
| `mpc_horizon_10` | 2 | 14 | **2** |
| `mpc_horizon_20` | 3 | 18 | **2** |
| `mpc_horizon_40` | 3 | 13 | **2** |
| `mpc_horizon_80` | 3 | 23 | **2** |
| **total** | **33** | **164** | **25** |

Not one family is left above its pre-#606 number, and eight are below
it — the refused seed still leaves the exact primal point, which the
pre-#606 constants also had but paired with a less useful `z`.

On HS071 with the same corruption: **9 → 3** iterations (cold 8,
pre-#606 3), `mu_out` back at `mu_in`. With `lagrange` corrupted and the
bound multipliers left out: **7 → 3**.

### #618 — the reconstruction cost recovered

Stale seed, `lagrange` only, the families that move:

| family | pre-#606 | parent | branch |
|---|--:|--:|--:|
| `mpc_horizon_40` | 45 | 81 | **45** |
| `mpc_horizon_80` | 53 | 77 | **53** |
| `mpc_horizon_20` | 23 | 30 | **23** |
| `mpc_horizon_10` | 17 | 24 | **17** |
| `degenerate_corner` | 10 | 12 | **6** |
| `degenerate_vertex` | 2 | 12 | 12 |
| `rosenbrock_ring` | 17 | 18 | **17** |
| `simplex_proj` | 11 | 10 | 11 |
| `moving_bound_qp` | 22 | 12 | 22 |
| `hanging_chain` | 18 | 14 | 18 |
| `nmpc_vanderpol` | 42 | 36 | 42 |
| **total** | **273** | **339** | **279** |

The +66 regression becomes +6. `degenerate_corner` lands 4 *below*
pre-#606. The cost is #606's stale-seed wins on `moving_bound_qp`,
`hanging_chain` and `nmpc_vanderpol` (−10, −4, −6), given back to buy
−74 on the `mpc_horizon` column.

`stale`/full moves 351 → 354, i.e. from +7 to +10 over pre-#606: the
same `hanging_chain` win (14 → 18) is given back there too, and the
`rosenbrock_ring` +1 is recovered.

### What is not fixed — `degenerate_vertex`

One family, 2 → 12, in both stale seed modes, unchanged by this branch.
Its diagnostics:

```
inf_pr = 0.0        avrg_compl = 2.986e-09  mu_in = mu_out = 2.506e-09
inf_du = 5.76       bound_duals = reconstructed (12)   split = true
```

θ enters this family through the objective, so a stale `x` is still
exactly feasible: `inf_pr` is **0**, not small. It is centred at exactly
the barrier it asked for, μ never moves, and only its `y` has moved.
Guard 1 cannot see it (complementarity is *right*); guard 4 cannot see
it (`inf_pr` is zero, so no slack can be swamped); guard 2 does not fire
(the stationarity residual is a small multiple of the scale it is
measured against, nowhere near 10x); a gentler μ band cannot see it
(1.19x overshoot). It is #618's
"hole in the middle" in its purest form. No measurement available to the
initializer separates it from a good seed, and lowering any trigger far
enough to catch it is precisely the un-conservative move both issues
warn against — #617 says "similarly conservative", #618 says `cresc4`
"has to be left bit-identical". This is stated here rather than spun out
as a new issue, per the standing instruction.

### Cross-check at a second scale

Same corpus at `small` (1.0x rather than 4.0x), against the pre-#606
control: exact 7/8 (control 10/22), corrupted 25/18 (control 32/28),
stale 204/193 (control 201/189). Same shape — corrupted below pre-#606,
exact well below, stale within a few.

## `cresc4` — the canary

**Bit-identical, 85 iterations, `obj = 0.87189753`,** on parent and
branch, in the tightened-push regime where #606 measured it. No
justification needed because nothing moved.

For the record, the attribution control at the parent shows what #606
itself still moves on the CLI corpus in that regime — and this branch
leaves all three exactly where #606 put them:

| fixture | `recentering=none` | `recentering=residual` (parent = branch) |
|---|--:|--:|
| `hs13_bigstart` | 30 | 29 |
| `jit1_boxed` | 16 | 17 |
| `jit1_node` | 19 | 21 |
| `cresc4` | 85 | 85 |

## Blast radius

`scripts/sweep-fixtures.sh`, 57 fixtures, against a release binary built
from the parent commit `b4c4d32e` in a `git worktree`, in all three
regimes #606 established reach this code:

| regime | result |
|---|---|
| default options (`warm_start_init_point` defaults to `no`) | **bit-identical**, 57/57 |
| `warm_start_init_point=yes`, default pushes | **bit-identical**, 57/57 |
| tightened recipe (`mu_init=1e-7`, all five `warm_start_*_push`/`_frac` at `1e-9`) | **bit-identical**, 57/57 |

Zero moving lines, so there is nothing to explain. The middle regime is
load-bearing evidence that the initializer actually runs: it differs from
the cold sweep on **17 of 57** fixtures, so a bit-identical result there
is a real negative, not a no-op.

The reason the corpus does not move is that the guards fire only on
seeds it does not contain. The `.nl` fixtures carry either no dual
segment or a coherent one, and none is a stale replay.

`warm_start_recentering=none` — the kill switch — is **bit-identical to
the parent across the entire seed-mode corpus**, every family, every
staleness, every content mode. Nothing here escapes the switch.

Beyond the solver: `info["warm_start"]` gains `bound_duals_rejected`
(int) and `eq_duals_rejected` (bool), `bound_duals` gains a `rejected`
verdict string, and the `print_level=5` iteration line gains `wz!` and
`wy!`. All additive.

## Tests

`crates/pounce-algorithm/tests/warm_start_recentering.rs`, +6 tests.

**Behavioural, and verified failing on the parent for the right reason.**
Both were lifted into a `git worktree` at `b4c4d32e` and run there:

* `a_corrupted_dual_seed_does_not_escalate_mu_to_the_ceiling` — parent
  output: `residual=9 (SolveSucceeded) none=3 cold=8 mu 2.5059e-9 -> 1e-1`.
  Fails on `mu_out <= mu_in`. The warm solve costs *more than a cold
  one* there (9 against 8). Branch: `residual=3 ... mu 2.5059e-9 ->
  2.5059e-9`.
* `a_corrupted_lagrange_seed_is_not_split_into_bound_multipliers` —
  parent output: `residual=7 none=3 mu 2.5059e-9 -> 1e-1 compl=9.13e+1
  split=true`. Same failure. Branch: `residual=3 ... split=false`.

**API-shaped, and stated as such.** These reference `BlockVerdict::Rejected`
and the two new counters, so they cannot be compiled against the parent
at all. They pin the diagnostics, not a behaviour, and are not counted
as a behavioural bite:

* `a_refused_dual_seed_says_so_in_the_diagnostics` — also pins that the
  verdicts are per block: HS071 refuses the two seeded variable-bound
  blocks *and* still reconstructs the slack-bound blocks no frontend can
  seed, and both counters stay honest about it.
* `an_exact_restart_is_unchanged_by_the_seed_guards` — nothing refused,
  reconstruction and split both still run, still beats the constants.
* `a_strongly_stale_point_still_gets_its_looser_barrier` — the case #618
  cites as "both ends of the range are fine". Still escalates, still is
  not mistaken for a corrupted seed. Live output: `residual=37
  (SolveSucceeded) none=36 (RestorationFailed)` — the case that *fails*
  before #606 still passes.

Full suites, on this branch's head:

* `cargo fmt --all -- --check` — clean.
* `cargo clippy --workspace --exclude pounce-hsl --all-targets` — no new
  warning. `init/warm_start.rs` emits the same 7 warnings (6
  `unwrap_used`, 1 collapsible `if`) on this branch as on the parent,
  verified by running clippy in both trees and diffing. The
  `too_many_arguments` this work first introduced was fixed by grouping
  the three measurements into `SeedMeasurement`, not by an `allow`.
* `cargo test --workspace --exclude pounce-hsl --release` — **231 test
  binaries, all ok, 0 failures**.
* `python/tests` — **1082 passed, 23 skipped**. One failure before `jax`
  was installed (`test_step_controller_is_what_the_jax_follower_uses`,
  a bare `ImportError` from `pounce.jax` with no skip guard) — an
  environment gap, unrelated, and it passes with `jax` present. `torch`
  is not installed and its modules skip.
* `pyomo-pounce/tests` — **382 passed** (`pyomo 6.10.1`, `networkx`,
  `scipy`, `pandas`, CLI staged into `python/pounce/bin/`).
* `scripts/check-docs-consistency.sh` — OK, 46 pages.
* `scripts/check-release-consistency.sh` — OK.

## Conflict note

Both siblings **merged into `main` during this work**, so these are
resolved conflicts, not predictions.

* **#630** (`claude/pounce-issue-610-loop`) merged as `7d406cfd`. Two
  overlaps, both auto-merged cleanly:
  - `CHANGELOG.md`, top of `[Unreleased]` — its `race_starts` entry
    versus this branch's warm-start entry. Resolved with this entry
    first and #630's existing order (`Pyomo` → `race_starts` → `Bool`)
    preserved beneath, unchanged.
  - `crates/pounce-py/src/problem.rs`, `build_info_dict` — #630 replaced
    twelve scalar parameters with `stats: &SolveStatistics` and dropped
    the `#[allow(clippy::too_many_arguments)]`; this branch adds two
    `set_item` calls to the `warm_start` sub-dict and one arm to
    `verdict_name`. Different hunks of the same function; no textual
    conflict. Verified by rebuilding and re-running the full seed-mode
    corpus after the merge — identical in every cell.
  - `docs/src/initialization.md` — #630 added `### Racing starts: the
    successive-halving ladder` at the end of the page; this branch adds
    `### A seed the solver will not believe` inside the warm-start
    section. No overlap.
* **#633** (`claude/pounce-issue-616-loop`) merged as `b4c4d32e`. Also
  auto-merged cleanly:
  - `CHANGELOG.md` — its least-square-init entry slotted below, ordering
    preserved.
  - `docs/src/initialization.md` — #633 added `### What the safeguard
    costs, and why it is not tuned away` and `### A declined step is not
    the same as never asking` to the **cold-start** half (lines 152 and
    242); this branch's section is at line 385 in the **warm-start**
    half. Disjoint.
  - `crates/pounce-algorithm/src/init/default.rs` — #633's only solver
    edit, and it is a pure refactor (`accepts_trial` extracted with
    identical logic) plus a `tracing::debug!` line. No overlap with
    `init/warm_start.rs`. Confirmed non-interacting by re-running the
    three-regime fixture sweep against a baseline built at `b4c4d32e`
    after the merge: still bit-identical.

No conflict required manual resolution. All measurements in this body
post-date both merges.

## The decisions, and what each cost

**#617: refuse the seed.** Take the remedy the issue asks for, but
locate it where the measurement says the damage is — the μ escalation
reading a corrupted `z`, not the reconstruction reading a corrupted `y`
— and add the split cap the second reading exposed. **Cost: nothing
measurable.** Exact restarts are bit-identical to #606's, the fixture
corpus is bit-identical, the kill switch is bit-identical, and every
corrupted family lands at or below pre-#606. The only thing spent is a
trigger constant that a future retuner has to keep conservative.

**#618: scale the reconstruction, don't touch μ.** Take the issue's
*second* proposed remedy after measuring the first and finding it inert.
**Cost: three of #606's stale-seed wins** — `moving_bound_qp` 12 → 22,
`nmpc_vanderpol` 36 → 42, `hanging_chain` 14 → 18, and the same
`hanging_chain` win again in the full-seed column. That is 20 iterations of #606's gains
given back in the `lagrange`-only column, and 4 more in the full column,
traded for 74 of its losses — plus a `degenerate_corner` improvement of
4 that neither issue predicted. `stale`/full ends 3 worse
than the parent (351 → 354) while `stale`/`lagrange`-only ends 60
better (339 → 279).

**Left undone, deliberately:** `degenerate_vertex`, 2 → 12. Nothing
available to the initializer distinguishes it from a good seed, and
inventing a trigger that catches it is the move both issues were filed
to prevent.
