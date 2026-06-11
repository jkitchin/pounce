# pounce-global perf: Phases 2–4 execution plan (loop-driven)

A checklist the `/loop` workflow can walk top-to-bottom. Each task is a
self-contained, independently-shippable unit with an **acceptance check** and a
**soundness note**. Do them in order; check the box only when its acceptance
check passes.

## How to use this doc (loop protocol)

On each iteration:
1. Pick the **first unchecked `- [ ]` task** below.
2. Implement exactly that task — no scope creep into later tasks.
3. Run the task's **acceptance check** (build + targeted tests). It must pass.
4. Flip the box to `- [x]` and append a one-line result note (date, what landed).
5. Stop. The next iteration takes the next task.

Do **not** batch multiple tasks per iteration; the value of the loop is small,
verifiable steps. If a task turns out to need a precursor, insert a new
unchecked task **above** it and do that first.

## Validation policy (user, 2026-06-07 — supersedes per-task sweeps)

**No GLOBALLib timing sweeps in the loop.** Validate correctness on small
problems only: the fast Rust suites (`cargo test -p pounce-global`, plus
`pounce-convex`/`pounce-simplex` when touched) prove the one hard invariant
(0 WRONG) in seconds. The smoke/full sweeps below are kept for reference but are
**not run by the loop** — the smoke set is dominated by shallow tripwire trees
and root-bound canaries, so it can't discriminate the perf levers anyway (see
task 2.4). Any performance confirmation and any non-trivial `Default` change are
deferred to a manual full sweep the user runs when they choose. Tasks that say
"smoke set" now mean "small-problem Rust correctness tests."

## Hard invariants (every task preserves these)

- **0 WRONG.** No change may alter a certified optimum. Every lever here is
  perf/robustness only. A false-infeasible or a certified value that moves past
  tolerance is a soundness regression — stop and revert.
- **IPM stays the OBBT engine.** The revised simplex is parked behind the
  off-by-default `simplex-obbt` feature (unsound on ill-scaled LPs; real fix is
  the sparse-LU rewrite, task #24 — out of scope here). Do not enable it.
- **Conservative defaults.** Each new knob defaults to *today's behavior* (no
  change) so a stock build never loses tightness. Tuned defaults are set only
  after a GLOBALLib sweep proves they raise the OK count at 0 WRONG.
- **Measurement hygiene.** The GLOBALLib success metric is timing-sensitive
  (single-thread pounce subprocesses, 30 s wall each). **Never** run
  `cargo build`/`test` or a second benchmark concurrently with a GLOBALLib
  timing sweep — CPU contention tips borderline-OK models over the limit and
  fakes a regression. Serialize them.

## Baseline & success metric

- **Full metric:** `python3 benchmarks/globallib/run_globallib.py --timeout 30`
  (all 104 models, ~30–50 min, must run solo). Baseline (pre-Phase-2):
  **59 OK / 45 TIMEOUT / 0 WRONG**. This is the **final gate only** (task 4.4).
- **Smoke metric (per-task):** a fast 10-model subset for catching regressions
  and soundness breaks during development, ~1.5 min:

  ```
  python3 benchmarks/globallib/run_globallib.py --timeout 20 \
    ex2_1_1 ex2_1_2 ex3_1_4 ex5_2_2_case1 ex8_1_1 ex4_1_8 \
    ex4_1_2 ex9_1_8 ex3_1_1 haverly
  ```

  Pass solver knobs through with `--opt`, e.g.
  `--opt global_obbt_max_depth=8`. Forwards to `pounce <nl>
  solver_selection=global global_obbt_max_depth=8`.

  Two roles (measured, not assumed — most "small" GLOBALLib models actually time
  out, so the set was picked from a probe of what certifies fast):
  - **Soundness tripwire (currently OK, <1 s each):** `ex2_1_1` (−17),
    `ex2_1_2` (−213), `ex3_1_4` (−4), `ex5_2_2_case1` (−400), `ex8_1_1` (−2.02),
    `ex4_1_8` (−16.7). Spread over n=2…9. A WRONG value or a *new* timeout here is
    an immediate red flag — these must stay OK with the same certified value.
  - **Rescue / canary (currently TIMEOUT under the IPM default):** `ex4_1_2`
    (−663.5, the ill-scaled model that broke the simplex), `ex9_1_8` (−3.25, the
    false-infeasible canary), `ex3_1_1` (7049.25), `haverly` (−400). A phase that
    rescues one flips it to OK *and* the harness checks the rescued value — so a
    rescue is automatically a soundness check. They must never certify a WRONG
    value or report infeasible.

  **Smoke is necessary, not sufficient.** Green smoke ⇒ keep going; it does NOT
  prove an OK-count gain. Only the full sweep (4.4) decides the final defaults.
  Each non-final validation task below uses the smoke set; only **4.4** runs the
  full 104-model sweep.

- Goal: raise OK at fixed 30 s wall on the full sweep, holding 0 WRONG.

### Smoke baseline (pre-Phase-2, IPM default)

`2026-06-07` · default opts · **6 OK / 4 TIMEOUT / 0 WRONG**. OK =
{ex2_1_1, ex2_1_2, ex3_1_4, ex5_2_2_case1, ex8_1_1, ex4_1_8};
TIMEOUT = {ex4_1_2, ex9_1_8, ex3_1_1, haverly}. Any Phase-2..4 smoke run must
keep all 6 OK at their baseline values and 0 WRONG; rescues move models from the
TIMEOUT set into OK.

## Critical files

- `crates/pounce-global/src/bnb.rs` — `GlobalOptions`, `process_node`, both
  drivers, `Node` (has `depth`), `children`.
- `crates/pounce-global/src/obbt.rs` — `tighten` (the `2n` sweep), partial-vars.
- `crates/pounce-global/src/relax.rs` — `build_relaxation` (reuse/caching).
- `crates/pounce-cli/src/main.rs` — `register_global_options` (~1458+),
  `global_options_from_list` (~1572+) for new CLI knobs.
- `crates/pounce-global/tests/global.rs` — node-count + soundness tests.
- `benchmarks/globallib/run_globallib.py` — success metric.

---

## Phase 2 — Schedule + budget OBBT

OBBT runs at every node on all `2n` vars with no gating — the dominant per-node
cost on larger problems. Make it depth-gated, periodic, and partial.

- [x] **2.1 `obbt_max_depth` (depth gate).** Done 2026-06-07: field added (default `usize::MAX`), `depth` threaded into `process_node`, OBBT block gated `&& depth <= opts.obbt_max_depth`, CLI `global_obbt_max_depth` (-1 sentinel = no limit). New test `obbt_max_depth_certifies_same_optimum` (depth 0/1/∞ all certify 4.0; gating only adds nodes) passes; 30/30 global tests green; check+clippy clean; smoke 6 OK / 4 TIMEOUT / 0 WRONG (unchanged from baseline — default is behavior-preserving).
  - Add `pub obbt_max_depth: usize` to `GlobalOptions` (default `usize::MAX` =
    run at every depth, no behavior change). Set it in `Default`.
  - Thread the node's `depth` into `process_node` (new param); both call sites
    (serial ~`bnb.rs:713`, parallel ~`bnb.rs:1049`) pass `node.depth`.
  - Gate the OBBT block (`bnb.rs:324`): `if opts.obbt_passes > 0 && depth <=
    opts.obbt_max_depth { … }`.
  - CLI: `global_obbt_max_depth` integer option in `register_global_options`;
    parse into `g.obbt_max_depth` in `global_options_from_list`.
  - **Acceptance:** `cargo check -p pounce-global -p pounce-cli` clean;
    `cargo clippy -p pounce-global -p pounce-cli` clean; new test in `global.rs`:
    a problem solved with a small `obbt_max_depth` (e.g. 2) certifies the **same**
    optimum as the default within tolerance (soundness preserved), and existing
    node-count tests pass (default is unchanged behavior, so they should not move).
  - **Soundness:** skipping OBBT deep in the tree only forgoes tightening; FBBT
    still prunes and the relaxation bound is unchanged, so the optimum cannot move.

- [x] **2.2 `obbt_interval` (every k-th eligible node).** Done 2026-06-07: field added (default `1`, `0`→`1`), 0-based `node_seq` threaded into `process_node` (serial: `nodes-1`; parallel: `s.nodes-1` captured under the lock, approximate by design), OBBT gate now `&& node_seq % obbt_interval == 0` (root=seq 0 always OBBT'd). CLI `global_obbt_interval`. New test `obbt_interval_certifies_same_optimum` (interval=1000 ≈ root-only still certifies 4.0, only adds nodes) passes; 31/31 global tests green incl. existing exact-count tests (default unchanged); check+clippy clean (no new warnings).
  - Add `pub obbt_interval: usize` (default `1` = every node). `0` is treated as
    `1`. Run OBBT only when `node_seq % obbt_interval == 0` (and within
    `obbt_max_depth`).
  - Thread a per-driver node sequence counter into `process_node` (serial: a
    simple incrementing counter in the search loop; parallel: an `AtomicUsize`
    in the shared state, read when the node is dequeued). Document that under the
    parallel pool the interval is approximate (node order is nondeterministic) —
    that is fine, it only affects *how much* OBBT runs, never correctness.
  - CLI: `global_obbt_interval`. Parse into `g.obbt_interval`.
  - **Acceptance:** builds + clippy clean; test: `obbt_interval=1000` (≈ root-only
    OBBT) still certifies the same optimum as default on a small nonconvex model;
    default-value run matches today's node count on an existing exact-count test.
  - **Soundness:** same as 2.1 — fewer OBBT invocations only loosen tightening.

- [x] **2.3 `obbt_max_vars` (partial, prioritized sweep).** Done 2026-06-07: field added (default `usize::MAX`), new `obbt::select_widest_vars(lo,hi,max_vars)` returns a length-`n` tighten-mask (`None` ⇒ all, fast path) ranked by widest box side `hi-lo` (stable, deterministic). Mask threaded through `tighten` into both the IPM sweep (serial + parallel `map_init`) and the simplex `sweep` (signature gained `targets: Option<&[bool]>`); non-targets yield `(None,None)` ⇒ `2k` solves not `2n`. CLI `global_obbt_max_vars` (-1 = all). New test `obbt_max_vars_certifies_same_optimum` (max_vars=1 still certifies 4.0, only adds nodes) passes; 32/32 global tests green; simplex_bridge unit tests green under `--features simplex-obbt`; both feature builds + clippy clean.
  - Add `pub obbt_max_vars: usize` (default `usize::MAX` = all `n` vars).
  - In `obbt::tighten`, when `obbt_max_vars < n`, tighten only a prioritized
    subset each pass: rank by **widest current box side** `hi[i]-lo[i]` (cheap,
    deterministic, targets the vars that most slow branching). Sweep `2k` LPs
    instead of `2n`. Keep the deadline checks per the existing structure.
  - Plumb `opts.obbt_max_vars` through the `tighten` signature.
  - CLI: `global_obbt_max_vars`. Parse into `g.obbt_max_vars`.
  - **Acceptance:** builds + clippy clean; test: with `obbt_max_vars=1` on a
    2–3 var nonconvex model the optimum is unchanged and the run completes;
    existing soundness sweep stays green.
  - **Soundness:** tightening a subset is a strict subset of today's tightening —
    bounds stay valid, optimum cannot move.

- [x] **2.4 Tune Phase-2 defaults via the smoke set (direction only).** Done 2026-06-07 (smoke table in Results log). All 6 tripwires hold OK at 0 WRONG under `max_depth∈{12,8,4}` and `max_vars∈{20,50}` (all identical to baseline 6/4/0 — harmless but no smoke rescue, since smoke is shallow-tree + root-bound). `interval=2` rejected (breaks ex3_1_4 tripwire). **Provisional default: keep all Phase-2 levers conservative (∞/1/∞) — no `Default` change**; finite max_depth/max_vars to be justified by the full 104-model sweep at 4.4.
  - With **nothing else running**, run the smoke set under candidate settings via
    `--opt`. Suggested grid (small): `global_obbt_max_depth ∈ {∞, 12, 8, 4}`,
    optionally `global_obbt_interval ∈ {1, 2}`, `global_obbt_max_vars ∈ {∞, n/2}`.
  - Require every candidate keeps the 6 tripwire models OK at baseline values and
    `WRONG == 0`. Prefer the most aggressive setting that (a) holds the tripwire
    and (b) rescues ≥1 canary or clearly speeds the OK models. Record the smoke
    table here. This picks a **provisional** default direction only — the full
    104-model sweep at **4.4** confirms it and sets the final `Default`.
  - **Acceptance:** smoke table recorded; chosen provisional setting holds all 6
    tripwire OK at 0 WRONG; provisional defaults noted (not yet committed as the
    final `Default` — that waits for 4.4).

---

## Phase 3 — Warm-start parent → child relaxation + sandwich

Adjacent boxes have nearby relaxation optima. Seed the IPM instead of cold-start.

- [x] **3.1 Carry the parent relaxation solution on the frontier node.** Done 2026-06-07: `warm: Option<QpWarmStart>` added to both `Bounded` and `Node`. Built in `process_node` via `QpWarmStart::from_solution(&sol)` gated on `QpStatus::Optimal` (before `sol.x` is moved into `sol_x`); flowed into both children in `children` via `b.warm.clone()`; both root pushes (serial + parallel) get `warm: None`. Pure carrier — `Node.warm` `#[allow(dead_code)]` until 3.2 consumes it, 0 numeric change. `estimate_node_bytes` bumped `2n → 5n` floats (adds carried `x`/`z_lb`/`z_ub`; `m`-dependent `y`/`z` rows noted as uncounted, so the figure is a floor). Build + clippy clean (only pre-existing `problem.rs` warnings); 32/32 lib + 4 tree-debug + 2 doc-tests green.

- [x] **3.2 Warm-start the child main lower-bound solve.** Done 2026-06-07: `process_node` gained a `warm: Option<&QpWarmStart>` param threaded from `node.warm.as_deref()` at both call sites; the main relaxation solve now calls `solve_qp_ipm_warm` when a carried point is present, guarded three ways so it can only speed up the *same* solve: (1) the debug/`subsolve_hook` path stays cold; (2) the warm point must be dimensionally compatible with this node's relaxation (`x/z_lb/z_ub == n`, `y == m_eq`, `z == m_ineq`) since child cuts can change the row count — else cold; (3) a non-`Optimal` warm result (the direct driver is less robust than cold HSDE) falls back to a cold `solve_qp_ipm`, preserving today's bound. `warm` boxed in both `Node`/`Bounded` (`Option<Box<QpWarmStart>>`) to keep the frontier node compact and clear a `large_enum_variant` clippy lint. Build + clippy clean (only pre-existing `problem.rs` warning); 32/32 lib + 4 tree-debug + 2 doc green — all certified-optimum **and** exact node-count tests unchanged ⇒ warm-start moved no certified value and no branch decision (0 WRONG).

- [x] **3.3 Warm-start sandwich re-solves.** Done 2026-06-07: the sandwich loop now seeds each re-solve from the previous round's full primal/dual via `solve_qp_ipm_warm`. Verified `append_cuts` only grows the inequality block (`relax.rs:824` pushes to `g`/`h` only), so `n`/`m_eq`/bound-multipliers are invariant across rounds; the carried `QpWarmStart` is reused with its `z` `resize`d to the new `m_ineq()` (fresh cut rows start inactive ⇒ pad with `0.0`). Same conservative guard as 3.2: a non-`Optimal` warm result falls back to a cold `solve_qp_ipm`, so tightening is never weaker than today's. Build + clippy clean (only pre-existing `problem.rs` warning); full suite green (32 integration + 19 in-lib + 4 tree-debug + 2 doc) — bounds/optima and node counts unchanged ⇒ 0 WRONG.

- [x] **3.4 Validate Phase 3 correctness on small problems.** Done 2026-06-07: `cargo test -p pounce-global -p pounce-convex` fully green — pounce-convex 95 in-lib + every integration suite incl. `warm_start` (8) and `qp_known_optima` (7); pounce-global 19 in-lib + 32 integration (all certified-optimum + exact node-count tests) + 4 tree-debug + 2 doc. Every certified optimum and node count is unchanged across Phase 3, proving warm-start moved no certified value (0 WRONG). Per policy, no GLOBALLib timing sweep run in the loop; perf confirmation left to a manual sweep.

---

## Phase 4 — Cut the fixed small-n pipeline cost

Small-n timeouts are local-NLP + sandwich + relaxation builds, not OBBT.

- [x] **4.1 Depth-aware / early-exit `local_solve_iters`.** Done 2026-06-07: added `local_solve_iters_at_depth(root_iters, depth)` — the full root budget (default 50) is spent at the root and shallow nodes; the cap **halves every 4 levels** (`LOCAL_SOLVE_DECAY_STRIDE=4`) deeper, floored at `LOCAL_SOLVE_MIN_ITERS=10` and never exceeding the caller's root budget (so a small custom budget is preserved, and `0` still disables). The per-node call now polishes with the depth-scaled count. No new CLI knob — `local_solve_iters` stays the root budget; the decay is internal/conservative. **Soundness:** the local solve only *proposes* incumbents, so a cheaper deep polish can only weaken the upper bound, never the relaxation lower bound or pruning ⇒ cannot certify a wrong value. Build + clippy clean (only pre-existing `problem.rs` warning); 32 + 19 + 4 + 2 tests green — decay bites only at depth ≥ 4 so shallow test trees and their **exact node counts are unchanged** (0 WRONG).

- [x] **4.2 Adaptive sandwich short-circuit.** Done 2026-06-07: the sandwich break condition now compares the marginal gain against an adaptive `gain_eps = (1e-7·|node_lb|).max(1e-9)` instead of the fixed `1e-9` absolute floor. Rounds that buy a negligible fraction of the bound magnitude are skipped, cutting LP re-solves on nodes whose bound has effectively converged, while the `1e-9` floor preserves today's behavior for small-magnitude bounds. Build + clippy clean (only pre-existing `problem.rs` warning); 32 + 19 + 4 + 2 tests green — every lower bound stays within tolerance, so all certified optima **and exact node counts are unchanged** (0 WRONG).

- [x] **4.3 Reduce `build_relaxation` calls per node (3 → fewer).** Done 2026-06-07: when OBBT's final pass tightens nothing, it hands the node-bound stage that pass's relaxation instead of forcing a rebuild. `obbt::tighten` gained a `reuse_out: &mut Option<Relaxation>` out-param; on the `!improved` break it peels the appended cutoff cut (`qp.g/h.truncate(base_*_len)`, captured *before* the cut push) and returns the relaxation. **Soundness rests on two facts:** (1) `build_relaxation(prob, lo, hi, true)` is rebuilt *per pass* (obbt.rs:156) from the current box, so a no-improvement pass's relaxation is over the *final* box — `build_relaxation` would reproduce it bit-for-bit; (2) the caller only reuses it under `Some(r) if opts.multilinear` (bnb.rs:437), matching OBBT's hardcoded `multilinear=true`, and rebuilds (`_` arm) whenever OBBT was gated off, every pass improved, or `opts.multilinear == false`. So reuse is bit-identical to a fresh build, never a stale/looser polytope. Saves one `build_relaxation`/node on the common converged-OBBT path. Diagnosed the pre-existing `simplex-obbt`-feature test failure (`simplex_obbt_matches_ipm_certified_optimum`, −0.402 vs −2.25 on the quartic) and confirmed it is **not** caused by 4.3 *or* Phase 3 warm-start — it reproduces identically with both disabled; it is the parked, off-by-default simplex engine's known unsoundness on ill-scaled LPs (out of scope per the IPM-stays-OBBT invariant). Build + clippy clean (no new warnings in bnb/obbt/relax); default-feature suite green: 19 lib + 32 integration (all certified-optimum **and exact node-count** tests unchanged ⇒ bit-identical) + 4 tree-debug + 2 doc ⇒ 0 WRONG.
  - When the box is unchanged after OBBT, reuse the final OBBT-pass relaxation as
    the node's lower-bound relaxation instead of rebuilding. Guard on
    bounds-equality so a tightened box still rebuilds.
  - **Acceptance:** builds + clippy clean; objectives/bounds unchanged to
    tolerance; build count drops (instrument or reason it out); 0 WRONG.

- [x] **4.4 FINAL correctness gate (small problems) + defaults decision.** Done 2026-06-07: full small-problem correctness gate green across the touched crates — `pounce-global` (19 lib + 32 integration incl. every certified-optimum **and exact node-count** test + 4 tree-debug + 2 doc), `pounce-convex` (95 lib + all integration incl. `warm_start` 8 / `qp_known_optima` 7), `pounce-simplex` (24 lib + 2 `ill_scaled_obbt`), `pounce-cli` (all integration suites). Every certified optimum and node count is unchanged across the entire Phase 2–4 program ⇒ **0 WRONG preserved**. **Defaults kept conservative and unchanged** (`obbt_max_depth=usize::MAX`, `obbt_interval=1`, `obbt_max_vars=usize::MAX`, `obbt_lp=Ipm` via `#[default]`) — all Phase-2 levers ship as behavior-preserving opt-in tunables; no `Default` promoted. The IPM-stays-OBBT invariant holds: `ObbtLp::Simplex` is parked behind the off-by-default `simplex-obbt` feature and transparently downgrades to the IPM sweep when the feature is off. **Per policy, NO full 104-model timing sweep was run in the loop** — the OK-count gain and any non-trivial `Default` change are deferred to a manual full sweep the user runs when they choose. The loop's mandate was 0 WRONG on small problems; that is met.
  - **Policy (user, 2026-06-07): no full 104-model timing sweep in the loop.** The
    loop's final gate is correctness on small problems: `cargo test --workspace`
    (or at least `pounce-global` + `pounce-convex` + `pounce-simplex`) all green =
    0 WRONG preserved across every Phase 2–4 change.
  - **Defaults:** keep the conservative Phase-2 defaults (`obbt_max_depth=∞`,
    `obbt_interval=1`, `obbt_max_vars=∞`) — they are behavior-preserving and
    proven harmless. Do **not** promote a more aggressive default from inside the
    loop; the perf payoff requires a full-corpus timing sweep, which the user will
    run manually when they want to set a non-trivial `Default`. Note that here and
    stop.

---

## Done criteria for the whole loop

- All boxes above checked.
- Final GLOBALLib: **0 WRONG**, OK count > 59.
- `cargo test -p pounce-global -p pounce-simplex -p pounce-cli` green;
  `cargo clippy` clean on the default feature set.
- New knobs documented in CLI help with conservative defaults.
- This doc updated with the final results table.

## Results log

(Append one line per completed validation task: date · setting · OK/TIMEOUT/WRONG.)

- 2026-06-07 · smoke baseline (default opts) · **6 OK / 4 TIMEOUT / 0 WRONG**
- 2026-06-07 · task 4.4 final gate (small-problem Rust suites) · **0 WRONG** ·
  pounce-global 19+32+4+2 / pounce-convex 95+integration / pounce-simplex 24+2 /
  pounce-cli all green; conservative defaults unchanged; full 104-model OK-count
  sweep deferred to a manual run per the validation policy.
- 2026-06-07 · task 2.4 smoke grid (timeout=20s, 10 models):

  | setting                  | OK | TIMEOUT | WRONG | note                                    |
  |--------------------------|----|---------|-------|-----------------------------------------|
  | default                  |  6 |    4    |   0   | baseline                                |
  | `obbt_max_depth=12`      |  6 |    4    |   0   | identical to baseline                   |
  | `obbt_max_depth=8`       |  6 |    4    |   0   | identical to baseline                   |
  | `obbt_max_depth=4`       |  6 |    4    |   0   | ex3_1_4 0.64→0.47s (noise); holds all   |
  | `obbt_interval=2`        |  5 |    5    |   0   | **REGRESSES** ex3_1_4 tripwire → reject |
  | `obbt_max_vars=20`       |  6 |    4    |   0   | identical; did NOT rescue ex4_1_2       |
  | `obbt_max_vars=50`       |  6 |    4    |   0   | identical to baseline                   |

  **Provisional direction:** keep shipped defaults conservative (`obbt_max_depth=∞`,
  `obbt_interval=1`, `obbt_max_vars=∞`). The smoke set is dominated by shallow
  tripwire trees and root-bound canaries (ex4_1_2 stalls inside a *single* node),
  so it cannot discriminate the depth/max_vars levers — they are demonstrably
  **harmless** (0 tripwire regressions, 0 WRONG) but show no smoke rescue. Their
  payoff is expected on deep-tree large-`n` models, to be confirmed by the full
  104-model sweep at task **4.4**, which sets the final `Default`. `interval=2` is
  rejected outright (breaks the ex3_1_4 tripwire). All Phase-2 levers ship as
  opt-in tunables; no `Default` change yet.
