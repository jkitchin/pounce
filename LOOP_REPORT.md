VERDICT: no-improvement

Fixes #616

## Problem

gh#605 gave the least-square primal initializer a backtracking
safeguard: the min-norm solution of the *linearized* constraints is
accepted only when it actually reduces the true nonlinear violation.
Under `least_square_init_primal=yes` — off by default — that costs two
tolerance downgrades, and #616 was filed so the cost had an owner
instead of sitting in a PR body.

**The issue's numbers were measured against `70bf53de`. They reproduce
exactly on current `main` (`a44f4e8b`), after #613, #619, #614, #623,
#620, #628 and #627.** Measured by building a binary with the safeguard
bypassed (pre-#605 semantics) from the same commit, so the comparison
isolates #605 rather than main's drift:

| fixture | issue's number (70bf53d) | reproduced: unsafeguarded on `a44f4e8b` | reproduced: `main` today |
|---|---|---|---|
| `csfi2` | `SolveSucceeded` → `SolvedToAcceptableLevel`, 53 → 35 it | `SolveSucceeded`, 53 it, obj 55.01760453 | `SolvedToAcceptableLevel`, 35 it, obj 55.01760453 |
| `eigenb2` | `SolveSucceeded` → `SolvedToAcceptableLevel`, 1.6 → 1.599999991 | `SolveSucceeded`, 55 it, obj 1.6 | `SolvedToAcceptableLevel`, 57 it, obj 1.599999991 |
| `pooling_rt2stp` | obj −4391.83 → −3273.95, 134 → 81 it | 134 it, obj −4391.826001 | 81 it, obj −3273.954992 |
| `deb7` | 479 → 202 it, 249.75 → 97.56 | 479 it, obj 249.7459752 | 202 it, obj 97.55993789 |
| `unbounded_cubic` | 91 → 290 it, `DivergingIterates` both | `DivergingIterates`, 91 it | `DivergingIterates`, 290 it |
| `SolveSucceeded` count | 46 → 44 | 46 | 44 |
| solved-or-acceptable | unchanged at 46 | 46 | 46 |

Every row reproduces. Fourteen fixtures move in total.

The one figure that does **not** reproduce is the total-iteration
headline: the issue reports 2067 → 1687 (−18%), and on `a44f4e8b` the
same sweep reads **5426 → 5245 (−3.3%)**. That is corpus drift, not a
changed conclusion — `issue_508_infeasible_gap_1em4` (3000 iterations)
and `cresc4` (742) dominate the current total and are identical on both
sides. The set of fixtures that move is exactly the set the issue
describes.

## Cause

The issue asked, first, whether the `csfi2`/`eigenb2` downgrades share
a mechanism with `unbounded_cubic`'s 91 → 290. **They do not — and the
two downgrades do not share one with each other either.** Attributing
all fourteen moving fixtures through `LeastSquareInitReport` puts them
in three different arms of the safeguard:

| arm | fixtures | what happens |
|---|---|---|
| `θ₀ = 0`, short-circuit | `unbounded_cubic`, `unbounded_exp`, `boxed_qp_fixed_var` | already feasible; no direction is ever computed |
| declined | `csfi2`, `deb7`, `pooling_rt2stp`, `linear_eq_aggregation`, `linear_eq_aggregation_row_constant`, `issue_372_infeasible_bounds` | all four trials worse than `θ₀`; the user's `x` is kept |
| backtracked accept | `eigena2`, `eigenb2`, `hs71_obj1e8`, `user_scaling_suffix`, `user_scaling_var_suffix` | accepted at `α < 1` |

`csfi2` is in the *declined* group (`θ₀ = 1508.554`, 4 rejected
trials). `eigenb2` is in the *accepted* group (`θ₀ = 1.0 → θ = 0.25` at
`α = 0.5`). `unbounded_cubic` never reaches either test. Three arms,
three mechanisms, independent.

That answer settles the design question, because it is what makes both
of the issue's proposed remedies unreachable:

- **`csfi2` cannot be recovered by a tighter test.** The safeguard
  already declines there. Its old `SolveSucceeded` came from taking a
  step that *raises* the true violation above 1508.554 — the one thing
  gh#605 exists to refuse. A stricter criterion still declines. Only
  removing the safeguard gets it back.
- **`eigenb2` cannot be separated from a win.** `eigena2` hands the
  safeguard **bit-identical** numbers — `θ₀ = 1.0`, `θ =
  0.2500000062500001`, `α = 0.5`, one rejected trial, step norm
  3.2596011939729705 — and improves (78 → 65 iterations, same
  objective) where `eigenb2` drops a band. Any criterion computed from
  the safeguard's own inputs treats the two alike.

## Fix

Three candidate criteria were measured, including both the issue named.
All three fail:

| candidate | result | why it fails |
|---|---|---|
| Retune `least_square_init_accept_ratio` | rejected | Acceptance is `θ₀ − θ ≥ η·α·θ₀`, so `eigenb2`'s trial survives every `η ≤ 1.5`; `η > 1` is meaningless (it demands a negative violation at `α = 1`). No reachable setting rejects it. |
| Acceptance band preferring the untouched point when the improvement is marginal | rejected | `eigenb2`'s step is not marginal — it cuts the violation 4×, the **median** of the sixteen accepted steps in the corpus, the same ratio as `airport` (103.6 → 25.96), `cresc4` (1715.3 → 437.2) and both `issue_508_infeasible_gap_*`, all wins. A band tight enough to exclude it excludes those too. |
| Require the accepted point not to degrade the dual residual | rejected | Measured: iteration-0 `inf_du` *improves* on both, 100 → 13.9 on `eigena2` and 100 → 47.7 on `eigenb2`. The gate accepts the step. |

So the decision is: **the downgrades stay.** They are tolerance-legal
answers at the right objective, on a route that is off by default, and
recovering them means giving back `deb7` 479 → 202 and
`pooling_rt2stp` 134 → 81 — which, per the issue's own framing, is not
a fix. The PR is documentation plus tests that pin current behaviour.

Two supporting changes were made, neither touching the solve:

1. **The safeguard's decision is emitted through `tracing`**, once per
   solve at `debug` on `pounce::algorithm`. `least_square_init_report()`
   is unreachable from the CLI, and the fixture sweep runs the CLI, so
   taking this measurement meant patching `init/default.rs` to print the
   report and rebuilding the workspace — once per hypothesis. The next
   investigator runs `RUST_LOG=pounce::algorithm=debug`.
2. **The accept test is a named predicate**,
   `DefaultIterateInitializer::accepts_trial`, so the argument above is
   pinned as executable properties rather than prose.

### Also found and documented: a declined step is not a no-op

Not in the issue, found while attributing it, and it corrects what the
docs implied. Declining restores the user's `x` exactly, but not the
solver's state: `calculate_least_square_primals` has by then driven the
first factorization through the augmented-system solver, on the `W = 0`
least-square matrix rather than the first real KKT matrix.

Isolated by forcing a decline on either side of that call — declining
*before* it is bit-identical to `least_square_init_primal=no` on every
fixture, declining *after* it is bit-identical to the real safeguard.
So the carrier is that one solve; the staging and the trial evaluations
are free.

It shows on two of the eight declining fixtures — `pooling_rt2stp` (298
iterations with the option off, 81 declined, same objective and status)
and `deb7` (154 against 202). The other six agree with `=no` exactly.
**Not changed**: making the decline a true no-op needs a separate
augmented-system solver for the initializer and costs `pooling_rt2stp`
81 → 298 iterations to buy a tidier contract on an off-by-default
option. That is a trajectory regression of exactly the shape CLAUDE.md
warns about, so it is described rather than done.

## Blast radius

Baseline: parent commit **`a44f4e8bcb67e577fb27c97a628efaff9822e494`**,
release binary built from a `git worktree` of that commit, kept outside
the repo. `scripts/sweep-fixtures.sh`, all 57 fixtures, three regimes:

| regime | result |
|---|---|
| default options | **bit-identical** to the parent |
| `least_square_init_primal=yes` (the regime that reaches the code) | **bit-identical** to the parent |
| `mehrotra_algorithm=yes` | **bit-identical** to the parent |

Nothing moves, which is the intended result: this branch adds a
`tracing::debug!` line, extracts a predicate without changing it, and
writes documentation and tests.

For completeness, the **`mehrotra_algorithm=yes` neutrality claim in the
issue was re-checked** against the unsafeguarded binary, since it was
measured at `70bf53d`. It still holds in the sense that matters — same
27 `SolveSucceeded`, same objectives — though the "292 iterations on
both sides" figure is stale and now reads 2475 → 2463. Twelve fixtures
move there; ten fail on both sides, so only the failure label and its
meaningless objective change, and the other two are `eigena2` and
`eigenb2`, which solve to the same objectives either way and differ by
one iteration.

Across the 57 fixtures the safeguard engages on 29 (16 accept, 8
decline, 5 short-circuit). It is inert on the other 28: 26 are LP or
convex-QP models the CLI dispatches to `pounce-convex`, which does not
run this initializer, and 2 have no constraints for the step to act on.

## Tests

Two new files, 12 tests.

`crates/pounce-cli/tests/issue_616_ls_init_downgrades.rs` (7) pins the
corpus measurement, so the decision is visible to the suite rather than
only to a PR body — the failure mode CLAUDE.md names, and the one that
let gh#544's 206 → 812 ship. Status and objective throughout, plus
three iteration-count *relations* that each carry a mechanism claim; no
absolute iteration count is asserted.

`crates/pounce-algorithm/tests/issue_616_ls_init_accept_test.rs` (5)
pins the algebra that makes the corpus result general rather than a
coincidence of two models.

**How they bite, stated precisely:**

- `the_safeguard_decision_is_attributable_from_the_cli` **fails on the
  parent commit for the right reason** — the debug line does not exist
  there, and the assertion says so: *"RUST_LOG=pounce::algorithm=debug
  did not surface the safeguard's decision"*. This is the one
  behavioural change on the branch, and this test is its regression
  test.
- **The other six CLI tests pass on the parent, and are meant to.**
  They pin behaviour this branch deliberately leaves alone. That is
  inherent to `VERDICT: no-improvement` and is not dressed up as
  anything else.
- **The five algorithm tests do not compile on the parent**, because
  `accepts_trial` is new there. That is a missing-API failure, not a
  behavioural one, and it is not offered as evidence that they bite.

Verified by copying both files into a `git worktree` at `a44f4e8b` and
running them.

**Full suite** (this branch): `cargo fmt --all -- --check` clean;
`cargo clippy --workspace --exclude pounce-hsl --all-targets` no errors
(the two `expect_used` warnings on the new CLI test match the
surrounding test files); `cargo test --workspace --exclude pounce-hsl`
229 test binaries, 0 failures; `python/tests` 1114 passed, 7 skipped
(plotly, ipopt, gamsapi, two missing `.nl` fixtures — all environmental
and all reproduce on the parent); `pyomo-pounce/tests` 352 passed;
`scripts/check-docs-consistency.sh` OK; `scripts/check-release-consistency.sh`
OK.

## Conflict note

Test-merged both open sibling branches into this one:

| branch | PR | conflict |
|---|---|---|
| `claude/pounce-issue-609-loop` | #629 | **`CHANGELOG.md`** — both add at the top of `[Unreleased]`. Nothing else. |
| `claude/pounce-issue-610-loop` | #630 | **`CHANGELOG.md`** — same. Nothing else. |

One collision each, textual, at the top of `[Unreleased]`; resolution is
to keep both entries.

Worth flagging: **#630 also edits `docs/src/initialization.md`**, which
the issue brief did not list. It does **not** conflict — #630 adds at
line ~515 (the "No good starting point at all?" section), this branch
adds at line ~138 (inside "Safeguarding the least-square start"). The
merge is clean, but a reviewer merging both should read the resulting
page once, since the two additions describe neighbouring machinery
(#630's start racing, this branch's safeguard).

Neither sibling touches `crates/pounce-algorithm/src/init/default.rs`.

---

- [x] Tests fail on the parent commit for the stated reason — one of
      them does, behaviourally; the rest are pins, and the section above
      says which is which rather than blurring it
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [x] Book page under `docs/src/` updated (`initialization.md`,
      existing section extended; already in `SUMMARY.md`)
- [x] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
- [x] Every claim in this body and in the code comments is true of the
      code as it stands now
