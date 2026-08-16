VERDICT: improved

Fixes #606

## Problem

Two defects in the interior-point warm start, both invisible from the
outside, both measured on the parent commit `70bf53de` before anything
was changed.

**1. You cannot seed every multiplier block, and the block you cannot
seed was filled with a constant.** `TNLP::get_starting_point` — what
`lagrange` / `zl` / `zu` reach on Python, the C ABI and the `.nl` path
alike — carries the constraint multipliers and the *variable*-bound
multipliers. The IPM also needs a multiplier for each inequality row's
slack (`v_L` / `v_U` internally), and no frontend has a field for one.
On every warm start POUNCE has ever run, that block arrived as zero and
was floored at `warm_start_mult_bound_push` — a number chosen with no
reference to the slacks it is paired against. The "warm" point handed to
the solver was therefore not a stationary point of anything.

On HS071, restarting from its own converged solution with every
multiplier the TNLP surface can carry:

| | iters | measured `‖∇ₓL‖∞` after init |
|---|---|---|
| before | 2 | — (not measured) |
| after | 1 | 3.5e-10 |
| cold, same model | 8 | |

**2. `mu_init` was taken on trust.** A restart at the optimum and a
stale point from a different parameter both started on exactly the
barrier the caller named. On HS071 warm-started from the far corner of
its box with the converged duals attached — the documented recipe from
`docs/src/initialization.md`, applied to a point that has moved:

| | status | iters |
|---|---|---|
| before | **`RestorationFailed`** | 36 |
| after | `SolveSucceeded` | 43 |
| cold, same model | `SolveSucceeded` | 8 |

That is not a slowdown. The pre-#606 path *fails* on it.

**3. Two advertised options did nothing.** `warm_start_same_structure`
and `warm_start_entire_iterate` parsed, set a field on
`WarmStartOptions`, and were never read. Verified bit-for-bit on
`70bf53de` before touching anything — a warm re-solve of `simplex_proj`
with each flag, both, or neither:

| `70bf53de` | status | iters | objective | KKT error |
|---|---|---|---|---|
| neither set | 0 | 2 | 0.590398708046706 | 2.5059430435806254e-09 |
| `same_structure=yes` | 0 | 2 | 0.590398708046706 | 2.5059430435806254e-09 |
| `entire_iterate=yes` | 0 | 2 | 0.590398708046706 | 2.5059430435806254e-09 |
| both `yes` | 0 | 2 | 0.590398708046706 | 2.5059430435806254e-09 |

Identical to the last digit. The issue's claim that their "promised
reuse semantics are incomplete" is generous: they were complete no-ops.

## Cause

The warm initializer (`crates/pounce-algorithm/src/init/warm_start.rs`)
received `cq` and `aug_solver` and used neither — both were bound as
`_cq` and `_aug_solver`. Every decision it made was a function of the
options rather than of the iterate it was handed: fixed pushes, a fixed
multiplier floor, a fixed `mu`. There was no measurement anywhere in it,
so there was nothing for it to adapt to.

The two dead flags name Ipopt's `TNLP::GetWarmStartIterate` surface,
which POUNCE does not expose at all.

## Fix

`warm_start_recentering` (new, `residual` | `none`, default `residual`)
turns the initializer into a pass over the supplied point.

1. **Measure.** `inf_pr` first — it is the one residual that does not
   depend on the duals, so it is meaningful even when every multiplier
   block is missing.
2. **Reconstruct the bound multipliers.** An entry that arrives as
   exactly `0` or `NaN` is not a legal barrier multiplier (the barrier
   needs `z > 0`), so it was never a seed — and it is already
   special-cased today, being floored at
   `warm_start_mult_bound_push`. What changes is the value it takes:
   the stationarity identity `P_L z_L − P_U z_U = ∇f + J_c^T y_c +
   J_d^T y_d`, and its slack-block twin `P_L v_L − P_U v_U = −y_d`,
   split by sign and floored at `μ / slack` so an inactive bound still
   gets what complementarity implies.
3. **Reconstruct an identically-zero `y` block** through the same
   regularized least-squares augmented solve the cold path uses
   (`LeastSquareMults`), now with real bound multipliers in its
   right-hand side.
4. **Raise `μ`** to the point's measured average complementarity when
   that overshoots `mu_init` by more than 10×.

`warm_start_target_mu` still pins `μ` outright.
`warm_start_recentering=none` restores pre-#606 behaviour exactly.

### Four things tried and rejected, each on a number

A future maintainer retuning this will otherwise repeat these
experiments. Each is documented at its definition in the source.

**Rejected: `μ ≥ κ·max(inf_pr, inf_du)`.** This is the obvious reading
of the issue's "choose `mu` … from those residuals", and it is wrong.
It cost **715 → 1129 iterations** across the 27 parametric paths of
`benchmarks/warmstart`. On `simplex_proj/tiny`, a re-solve that needed
one iteration measured `inf_du = 2e-3`, took `μ = 2e-3`, and needed
five. A warm point at a moved parameter carries a primal and a dual
residual of order Δθ *by construction* — that is the premise of warm
starting — and raising `μ` to meet them throws the warm start away to
pay for a Newton step that was about to happen anyway. Of the three
KKT residuals, complementarity is the one `μ` *is*; the other two are
what the Newton step is for. `avrg_compl` on that same point read
`2.6e-9`, the converged barrier, correctly recognised.

**Rejected: filling `z = μ̂/slack` with a residual-inflated `μ̂`.** The
slacks reaching the initializer have already been pushed into the bound
interior by `warm_start_slack_bound_push`, so on a converged point —
where the true slack sits at `μ/z` and the push dominates it — `μ̂/slack`
is off by exactly the push's inflation. On HS071 an 8×-inflated `μ̂` put
the reconstructed slack multiplier 8× high and took an exact restart
from 1 iteration to 5. The fill barrier is `mu_init`; the stationarity
split, which never looks at a slack, is immune to it entirely.

**Rejected: overriding `μ` whenever the measurement disagrees.** Moving
`μ` reroutes the whole trajectory, so it has to be earned. On `cresc4`
(nonconvex, enters restoration on iteration 2, where perturbations
compound) a measurement of `5e-7` against a `mu_init` of `1e-7` moved
`μ` by half an order and the solve from **85 iterations to 206** — same
status, same objective, 2.4× the work. This is the gh#544 shape exactly.
Hence the 10× trigger; with it that fixture is bit-identical again, and
the cases this exists for (stale seeds miss by three to six orders)
still fire.

**Rejected: reconstructing from a seed with no duals at all.**
Completing a *partial* warm start is well-posed — the supplied blocks
pin the missing ones through stationarity. Manufacturing a whole one
from a primal point is not: what comes out is the cold path's estimate
wearing the warm path's barrier, and it cost **1102 → 1211 iterations**
on the same corpus (`degenerate_corner` 17 → 38 at every step size).
Such a seed is now reported `unseeded` and left exactly as before. The
same reasoning capped the least-squares `y` estimate at
`constr_mult_init_max` (1e3, the cold path's cap) rather than
`warm_start_mult_init_max` (1e6): on `redundant_rows`, whose duplicated
equality rows make the system singular, the looser cap let an arbitrary
estimate through and cost 7 → 25 iterations.

### Scope: what was and was not done

The issue lists six items. Items 1, 2, 3, 5 and 6 are implemented.

- **Item 4** — "a genuinely complete iterate transfer (… and reusable
  structure/factorization where valid)" — is **not** done, and is the
  same missing capability as item 5's refused half: it needs the
  `GetWarmStartIterate` TNLP surface POUNCE does not expose, plus a
  factorization-reuse path that does not exist. Building it is a
  feature, not a line; refusing the two options that advertise it is
  the honest half, and is what shipped.
- **Item 2's "bound/slack pushes"** — `warm_start_bound_push` and
  friends are still fixed constants. Only the multiplier *floors* and
  `μ` became residual-derived. Making the primal push adaptive moves
  the primal point itself, which is a much larger trajectory risk than
  anything here, and nothing in the measurements pointed at it.
- The **GAMS native C link** and the **C ABI** were not touched. They
  reach the same core initializer, so they get the behaviour change;
  they get no new frontend surface for the diagnostics.

## Blast radius

Baseline for every number below: **`70bf53ded3d893cfa2da6ead5195fda5ac096f68`**
(`main`, the parent of this branch). Same machine, same binaries, both
directions measured with the identical script.

### Fixture sweep — three regimes

`scripts/sweep-fixtures.sh` with default options never reaches this
code: `warm_start_init_point` defaults to `no`, so the warm initializer
is not even constructed. That sweep is bit-identical, and it proves
almost nothing. It was run anyway, and then twice more in regimes that
*do* exercise the change:

| sweep | options | result |
|---|---|---|
| cold (the standard one) | defaults | **bit-identical across all 57 fixtures** |
| warm, default pushes | `warm_start_init_point=yes` | **bit-identical across all 57 fixtures** |
| warm, tightened recipe | `warm_start_init_point=yes mu_init=1e-7` + all five `warm_start_*_push`/`_frac` at `1e-9` | **3 of 57 move** |

The middle sweep is the load-bearing one: it differs from the cold sweep
on 17 of 57 fixtures, so the warm initializer demonstrably runs, and the
change is still a no-op there. The mechanism is exact — with no dual
seeds in the `.nl` files, `any_dual_seeded` is false, so no
reconstruction runs; and with the default `warm_start_mult_bound_push`
of `1e-3` against `O(1)` slacks the measured complementarity is `~1e-3`,
below the `mu_init` default of `0.1`, so `μ` is untouched.

The third sweep is the documented tight-push recipe, i.e. the regime the
change is *for*. Every line that moves:

| fixture | before | after | status / objective |
|---|---|---|---|
| `hs13_bigstart` | it=30, obj=0.9891769904 | it=29, obj=0.9883552912 | `SolveSucceeded` both; improvement |
| `jit1_boxed` | it=16 | it=17 | `SolveSucceeded` both, obj=173345.3768 to 10 digits |
| `jit1_node` | it=19 | it=21 | `SolveSucceeded` both, obj=173345.3768 to 10 digits |

Each was attributed by running it back with
`warm_start_recentering=none`, which reproduces the "before" column
exactly on all three — so the mechanism is this change and nothing else,
and the kill switch works. `hs13` is the known MFCQ-failure model, whose
objective is tolerance-legal anywhere near 0.988–0.989; both runs sit in
that band. `cresc4` moved 85 → 206 in an earlier revision and is
bit-identical now (see "Rejected" above).

### Repeated-solve benchmark

`benchmarks/warmstart`, 9 families × 3 step scales = 27 parametric
paths, 7 warm re-solves per path (189 warm solves per mode). Every mode
sees the *same* per-step seed, so the only variable is how much of the
dual state is handed back.

| seed | | iters | obj evals | jac evals |
|---|---|---|---|---|
| **full** (x, λ, z_L, z_U, μ) | before | 715 | 1207 | 994 |
| | **after** | **677** | **1068** | **950** |
| | | −5.3% | −11.5% | −4.4% |
| **partial** (bound mults kept, `lagrange` dropped) | before | 718 | 1210 | 997 |
| | **after** | **709** | **1167** | **988** |
| | | −1.3% | −3.6% | −0.9% |
| **primal only** (x and μ) | before | 1102 | 2421 | 1492 |
| | **after** | 1102 | 2421 | 1492 |
| | | *bit-identical by construction* | | |
| **cold** (reference, no seed) | before/after | 2036 | 2619 | 2199 |

`partial` is the issue's own acceptance criterion — "partial warm starts
outperform zero-filled duals on a repeated-solve benchmark" — and it is
met, though modestly. `full` is what `pounce.WarmStart` produces and
what the docs recommend, and is where the change pays.

Only 8 of the 108 (path × mode) cells move at all. Every one:

| path | mode | before | after |
|---|---|---|---|
| `hanging_chain/tiny` | full | 49 | **14** |
| `hanging_chain/small` | full | 65 | **57** |
| `hanging_chain/large` | full | 68 | **47** |
| `rosenbrock_ring/small` | partial | 41 | **35** |
| `rosenbrock_ring/large` | partial | 40 | **37** |
| `rosenbrock_ring/tiny` | full | 28 | 37 |
| `rosenbrock_ring/small` | full | 42 | 45 |
| `rosenbrock_ring/large` | full | 36 | 50 |

`hanging_chain` is the flipping-contact family — 15 inequality rows
whose slack multipliers were exactly the block nobody could seed, so
it is where reconstructing them pays most. `rosenbrock_ring` is
nonconvex with a single activation switch crossed along the path; the
reconstructed slack multipliers reroute it, better in `partial` and
worse in `full`. Net over the family: `full` +23, `partial` −9.

### Exact same-model restarts

Solve, then re-solve the identical model from its own answer. Summed
over the 9 families:

| seed | before | after |
|---|---|---|
| full | 6 | **2** |
| partial | 9 | **7** |
| primal only | 17 | 17 |

No family regresses. `hanging_chain` and `rosenbrock_ring` go from 2
iterations to **0** on a full restart — the supplied point is now
recognised as the KKT point it is.

### Regressions, named

Two, both on deliberately-bad seeds, both accepted here as the cost of
the fix. **Both need their own issue and an owner** (per `CLAUDE.md`:
a measured regression recorded only in a commit message is
indistinguishable from noise to the next reader). This PR does not
close them; they are listed here so they get filed, not buried.

**(a) Corrupted dual seeds cost more than they used to.** Seeding the
exact primal solution together with multipliers corrupted by `N(0,1e4)`
noise, summed over 9 families: **18 → 71 iterations**. Per family the
worst is `hanging_chain` 3 → 13; every case still converges with status
`SolveSucceeded` to the same objective, and lands within ~1.4× of the
cold solve (cold sum 81). Mechanism: the reconstructed slack
multipliers are derived from the seeded `y`, so they inherit its
corruption, where before they were discarded in favour of a constant.
The pre-#606 "2 iterations" was not skill — it was throwing the bad
duals away. Follow-up: reject a seeded dual block whose implied
complementarity is orders away from the rest of the point, rather than
reconstructing off it.

**(b) Mildly stale seeds regress slightly.** Seeding each family from
the far end of its own 4×-scale path: **118 → 127** iterations summed
over 9 families with a full seed (`redundant_rows` 3 → 7 is the worst,
`degenerate_corner` 9 → 11, `hanging_chain` 8 → 10, `moving_bound_qp`
11 → 12; the rest unchanged). These points are stale enough for the
reconstruction to fire but not stale enough to trip the 10× `μ`
escalation, so they get the reconstruction's trajectory change without
the recentering that is supposed to pay for it. The *strongly* stale
case is the opposite — HS071 from the far corner goes from
`RestorationFailed` to `SolveSucceeded` — so the gap is in the middle of
the staleness range. Follow-up: a second, gentler escalation band.

## Tests

`crates/pounce-algorithm/tests/warm_start_recentering.rs`, 7 tests over
an HS071 whose starting point *and* dual seeds are both settable.

Four of them fail on the parent commit `70bf53de` for behavioural
reasons. Verified in a `git worktree` at that commit with the
assertions that need new API (`BlockVerdict`,
`warm_start_diagnostics()`, the `warm_start_recentering` option itself)
stripped, so the failures are behavioural and not compile errors.
Exactly what they printed:

```
exact restart: none=2 residual=2 cold=8
  panicked: residual=2 vs none=2                 (assert residual < none)
partial: none=3 residual=3 cold=8
  panicked: residual=3 vs none=3                 (assert residual < none)
stale: status=RestorationFailed iters=36 cold=8
  panicked                                       (assert SolveSucceeded)
warm_start_same_structure=yes -> refusal None
  panicked                                       (assert is_some)
```

On this branch the same four read `1 vs 2`, `2 vs 3`,
`SolveSucceeded in 43`, and a refusal naming `GetWarmStartIterate` and
issue 606.

The remaining three pin the kill switch (`recentering=none` leaves every
block alone and does not move `μ`), the deliberate non-behaviour on a
primal-only seed, and a registry invariant.

**Deliberately not a regression test:**
`every_registered_warm_start_option_is_consumed_or_refused` — it mirrors
gh#604's `Initialization` invariant one category over, but on the parent
commit the two dead flags *are* read (into fields nothing consumes), so
it would have passed there. It guards the next warm-start option, not
this one.

**Deliberately not pinned:** the corpus-level iteration counts above.
They come from `benchmarks/warmstart`, which is a benchmark suite rather
than a test, needs the Python extension built, and takes minutes. The
numbers in this PR are the record; nothing in CI will notice if they
drift. That gap is the same one the fixture sweep exists to cover for
the cold path, and the third sweep regime above is the closest
CI-shaped proxy for it.

Also run clean: `cargo fmt --all -- --check`, `cargo clippy` (no new
warning categories in the changed files — 6 `unwrap_used` warnings
against the baseline's 4, both new ones in `scatter`, matching the
type-tested idiom already in `resolve_nan_seeds` beside it),
`cargo test --workspace --exclude pounce-hsl`, the Python suite
(777 passed, 34 skipped), `scripts/check-release-consistency.sh`,
`scripts/check-docs-consistency.sh`.

---

- [x] Tests fail on the parent commit for the stated reason
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [x] Book page under `docs/src/` updated (`initialization.md`; already
      in `SUMMARY.md`)
- [x] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
- [x] Every claim in this PR body and in the code comments is true of
      the code as it stands now
