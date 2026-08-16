VERDICT: improved

Harden `WarmStart` persistence and keep warm-start options call-scoped

Fixes #607

## Problem

Two independent silences in `python/pounce/_warm_start.py`. Both are the
gh#544 shape: the answer is fine, everything else is not.

All numbers below are measured on the parent commit
**`70bf53ded3d893cfa2da6ead5195fda5ac096f68`** ("Register and validate the
cold-start initialization options (#604) (#613)") and on this branch, with
the same harness, the same HS071 fixture, and release builds of both.

**1. Warm-start options leaked into every later solve.** Applying a warm
start called `add_option` on the *persistent* `Problem`, and `add_option`
is append-only, so all seven enabling options stayed set for good.

| HS071, ordinary cold solve from `x0 = [4.9]*4` | status | objective | iters |
|---|---|---|---|
| pristine `Problem` (both commits) | `Solve_Succeeded` | 17.0140171452 | **17** |
| after one warm solve — parent | `Solve_Succeeded` | 17.0140171452 | **24** |
| after one warm solve — this branch | `Solve_Succeeded` | 17.0140171452 | **17** |
| after a warm solve that *raised* — parent | `Solve_Succeeded` | 17.0140171452 | **24** |
| after a warm solve that *raised* — this branch | `Solve_Succeeded` | 17.0140171452 | **17** |

41% more iterations, identical objective to ten digits, nothing anywhere
saying why. The harness confirms the mechanism rather than assuming it:
hand-applying `ws.options()` to a pristine `Problem` reproduces the 24
exactly on the parent, and no longer matches on this branch.

**2. A persisted `WarmStart` could not tell what model it belonged to.**
It recorded arrays and a four-wide float row — no dimensions, ordering,
sparsity, bounds, scaling, algorithm or model fingerprint — so an
incompatible replay was simply attempted. The true optimum is
**17.0140171452**.

| Replay of a saved artifact | parent commit | this branch (signed artifact) |
|---|---|---|
| same problem (control) | 2 iters, 17.0140171449 | 2 iters, 17.0140171449 |
| **variables reordered** | solved, `Error_In_Step_Computation`, obj **16.3801074993**, `x` off by **0.257**, 44 iters | refused — `var_ids: identifiers differ` (see limitation below) |
| **upper bound 5 → 4** | solved, `Restoration_Failed`, obj **15.8435942123**, 7 iters | refused — `bounds:` facet |
| **declared sparsity changed** | solved silently, `Solve_Succeeded`, 2 iters | refused — `sparsity:` facet |
| **different model, same n/m/box/sparsity** | solved silently, `Solve_Succeeded`, 9 iters | refused — `bounds:` facet |
| **n 4 → 5** | `ValueError: x0: expected length 5, got 4` from solve preparation | refused before the solver — `n: captured 4, target 5` |
| **n 4 → 5, explicit right-length `x0`** | `ValueError: zl: expected length 5, got 4` | same refusal, same place |
| **`nlp_scaling_method` changed** | solved silently, `Solve_Succeeded`, 2 iters | refused — `scaling:` facet |
| **`algorithm` changed to SQP** | solved silently, `Solve_Succeeded`, 2 iters | refused — `algorithm:` facet |

Control, both commits: a compatible round trip is lossless — `x`,
`lagrange`, `zl`, `zu` compare bit-identical after `save`/`load`, and the
reloaded state still takes the re-solve from 11 iterations cold to 3.

## Cause

`add_option` pushes onto three `Vec`s that `PyProblem::prepare` replays at
every `solve()`. There was no removal operation, so the wrapper could not
have scoped its options even if it had wanted to; the docstring
acknowledged the leak ("note they persist on this Problem like any
`add_option`") rather than fixing it. And nothing in the wrapper ran
inside a `try`, so a raising solve left the overlay behind too.

For persistence, `save()` wrote `x`, the optional dual arrays, an optional
working set, and `_meta = [mu, bound_push, mu_init, mu_init_fallback]`.
Every array in a warm start is indexed by a *model* — variables in one
order, constraints in another — and none of that indexing was recorded, so
the only thing a replay could check was array length, and only
incidentally, deep inside solve preparation.

## Fix

**Rust (`crates/pounce-py/src/problem.rs`), four additive accessors.**
`options_snapshot()` / `restore_options()` are the snapshot-restore pair
that makes a scoped overlay expressible — restore undoes intervening
`add_option` calls including ones that introduced a name that had never
been set, which no sequence of `add_option` can express. `get_bounds()`,
`get_problem_scaling()` and the `problem_obj` getter are read-only views of
state the `Problem` already holds, following the `set_`/`clear_`/`get_`
convention of `set_ordering` / `set_kkt_schur_block` beside them.

**Scoping.** The wrapper now snapshots, applies, and restores in a
`finally`. It never enumerates option names — whatever `ws.options()`
returns is what gets scoped — so a key added to that recipe later is
covered by construction. This is not a hypothetical: it is exactly what
makes #620's `warm_start_recentering` scope correctly with no change on
either side (verified, see the conflict note).

**Schema.** Archives are versioned (v2) and carry a `ProblemSignature`:
dimensions, optional stable variable/constraint IDs, and digests of the
bound signature, the declared jacobian/Hessian structure, the scaling
convention, the algorithm/backend, and the model-defining options.
Fingerprinting rules live in a separate module, `_warm_start_schema.py`:
they change for reasons that have nothing to do with the warm-start
object's own fields.

`compat` is `"strict"` (raise) / `"warn"` (report and proceed) /
`"unsafe"` (skip), settable on the object, on `load()`, or per call.
Replay is labelled `exact` or `mapped`; `transfer(prob, mapper)` is the
hook for a horizon shift or a changed discretization, `reindex(prob,
var_ids=…)` writes the mapper when both sides carry stable IDs, and a
mapped artifact is still refused on a third problem. Unmatched multiplier
entries are seeded `NaN` — the native "unseeded" contract — rather than
fabricated.

**The compatibility decision, stated deliberately.** Strict is the
default. Strict on a *signed* artifact means any differing or unverifiable
facet raises. Strict on a **legacy (v1) artifact** means: load it, replay
it, check the one facet its own arrays witness (the dimensions), and warn
once with the migration path. Refusing v1 outright would break every
archive written before this change *while it still replays correctly on
the model it came from* — a migration cost with no safety return, which is
not what #518 and #613 refuse things for. Those refuse operations that
silently do nothing; a legacy warm start silently does the right thing on
the right model. The migration is `ws.migrate(prob)` (re-sign as-is) or
re-capture with `problem=` and re-save. An unsigned *in-memory*
`from_info(x, info)` — the pre-#607 call — is silent and unchanged.

**Rejected:** maintaining a Python-side shadow of the option list instead
of the Rust pair. It cannot remove an option that was not previously set,
so a warm solve on a `Problem` that had never seen `mu_init` would still
leak it — the exact measured case above.

**Rejected:** deriving the fingerprint inside `solve` so capture is
automatic. That is the same region of `problem.rs` #620 rewrites, it costs
every solve, and it makes signing non-optional. `from_info(x, info,
problem=prob)` keeps it opt-in and keeps the diff off #620's hunks.

## Blast radius

**Fixture sweep: bit-identical across all 57 models** — status, objective
*and* iteration count — `scripts/sweep-fixtures.sh` run against a release
binary built from the untouched parent `70bf53de` in a separate worktree,
diffed against a release binary from this branch. Expected: the change is
Python-side plus four accessors the solver never calls, and no solver step
moved. Reported because "it cannot produce a wrong answer" is not the
relevant safety property here.

Behaviour changes a user can see:

- A warm solve no longer leaves options on the `Problem`. Code that
  *relied* on the leak — one warm solve to configure a series — now gets
  the cold defaults. That is the bug being fixed, but it is a behaviour
  change, and the CHANGELOG says so with the numbers.
- Replaying a v1 archive emits one `WarmStartLegacyWarning`. Under `-W
  error` that is a new failure; `compat="unsafe"`, `migrate()`, or
  re-capturing clears it.
- Signed artifacts are new, so nothing existing can be refused by the
  strict check.

## Known limitations

Stated because one of them is a case the issue names by name.

- **A pure reordering is invisible to the digests.** Permuting a model
  with a uniform box and a dense jacobian leaves the bound and sparsity
  digests bit-identical — measured, not assumed: `ws.check_compatible()`
  returns no mismatch for the reordered HS071 above. Ordering is knowledge
  only the caller has, so it is caught only when `var_ids` is supplied on
  both sides. Alternatives were worse: making a missing target `var_ids`
  an unverifiable-facet mismatch would mean signing an artifact *with* IDs
  is strictly worse than signing it without, since a live `Problem` has no
  idea what its variables are called.
- **A mapped interior-point warm start is a correctness mechanism, not a
  speed-up.** On the slew-limited receding-horizon fixture in the test
  file the transferred point costs 12 iterations against 7 for a cold
  solve; on a longer sinusoidal track the gap widens with the horizon —
  12 vs 9 at horizon 5, 15 vs 11 at 10, 22 vs 9 at 20, 30 vs 10 at 40.
  Dropping the carried `mu`, or loosening it, or loosening `bound_push`,
  changes none of that (11–13 iterations across six variants on the test
  fixture). This is the barrier/active-set limit
  `docs/src/initialization.md` already describes, and it is what #620
  attacks on the Rust side. The horizon test therefore asserts the answer
  and the labelling, and explicitly does not assert an iteration win — the
  numbers are in the test comment so the next reader does not re-run the
  experiment.
- The `model` fingerprint is an explicit allowlist of options that
  redefine the problem, not a digest of everything. A warm start stays
  valid across a changed `max_iter` (verified) and must not across a
  changed `fixed_variable_treatment` (verified). A future model-defining
  option has to be added to the list.

## Tests

`python/tests/test_warm_start_schema.py`, 34 tests, covering every case
the issue names: reordered variables, changed bounds, changed sparsity,
horizon transfer, the exception path, and legacy artifact migration —
plus strict/warn/unsafe, mapped-artifact provenance, mapper length
checking, and v1↔v2 format compatibility in both directions.

**They fail on the parent commit, and behaviourally.** Three of them
reduce to the pre-existing API only, so the failure is "the parent behaves
wrongly", not "the parent will not import". Run against a release build of
`70bf53de`:

```
FAILED test_warm_options_do_not_leak_into_the_next_ordinary_solve
    assert after["iter_count"] == pristine["iter_count"]
E   assert 24 == 17
FAILED test_warm_options_are_restored_even_when_the_solve_raises
    assert after["iter_count"] == pristine["iter_count"]
E   assert 24 == 17
FAILED test_legacy_artifact_replay_says_it_is_unverified
    with pytest.warns(UserWarning):
E   Failed: DID NOT WARN. No warnings of type (<class 'UserWarning'>,)
    were emitted. Emitted warnings: [].
3 failed in 0.75s
```

The same three pass on this branch. The compatibility half is a *new
capability* with no old-API expression, so its tests cannot be reduced
that way — on the parent they fail at collection with `ImportError: cannot
import name 'WarmStartCompatibilityError' from 'pounce'`. That gap is
stated rather than papered over.

Green on this branch: `cargo fmt --all -- --check` (clean), `cargo clippy
--workspace --exclude pounce-hsl --all-targets` (no new warnings; the
pre-existing `pounce-restoration` ones are untouched), `cargo test
--workspace --exclude pounce-hsl` (2926 passed, 0 failed), the Python
suite (807 passed, 38 skipped — 773 before, 34 new),
`scripts/check-release-consistency.sh` and
`scripts/check-docs-consistency.sh` (both OK).

## Conflict note — PR #620 (issue #606), branch `claude/pounce-issue-606-loop`

#620 is open and unmerged and edits `python/pounce/_warm_start.py`. This
branch was developed against `main` so #607 is measured in isolation, and
#620's branch was **not** merged in and is not depended on. A trial merge
was run to characterize the conflict; the resolution below was then
applied, built, and tested for real.

**`git merge` of #620 into this branch: 4 conflicts, all pure
"both-added", none semantic.** `crates/pounce-py/src/problem.rs` and
`docs/src/initialization.md` auto-merge — #620's `problem.rs` edits are in
`solve` / `solve_with_sens` / `build_info_dict`, and this branch's are a
new `#[pymethods]` block next to `get_kkt_schur_block`.

| file | conflict | resolution |
|---|---|---|
| `_warm_start.py` | dataclass docstring: #620 adds `recentering:`, this branch adds `signature:` / `compat:` / `replay:` / `source_signature:` / `schema_version:` / `origin:` at the same anchor | keep both, `recentering` first to match field order |
| `_warm_start.py` | field list: same anchor, same reason | keep both, `recentering` first |
| `_warm_start.py` | `load()`: #620 adds `recentering=recentering,` as the last kwarg, this branch adds `**cls._schema_overrides(data, compat),` | keep both lines |
| `CHANGELOG.md` | both add a bullet at the top of `[Unreleased]` | keep both bullets |

Everything else merges clean *by construction*, and that was a design
constraint, not luck:

- `WarmStart.options()` is **untouched** here, so #620's
  `warm_start_recentering` entry merges without a conflict — and, more
  importantly, is scoped correctly with no change on either side, because
  the overlay snapshots the option list rather than naming keys.
- The v2 persistence keys are added through one line in `save()`
  (`payload.update(self._schema_payload())`, after the working-set block)
  and one in `load()`, away from #620's `_recentering` hunks.
- The bulk of the new code is in a new file, `_warm_start_schema.py`, and
  a new test file, neither of which #620 touches.

**The resolution was verified, not assumed.** With the four conflicts
resolved as above, the merged tree builds and the full Python suite passes
— 807 passed, 38 skipped, the union of both branches' tests. And the
interaction that mattered was checked directly on the merged tree:

```
#606 options() includes: ['warm_start_recentering']
option list restored exactly: True
warm_start_recentering left behind: False
pristine iters=17   after-warm iters=17
```

Suggested merge order: either way round works. If #620 lands first, this
branch rebases onto it with the four hunks above; if this lands first,
#620 does the same.

---

- [x] Tests fail on the parent commit for the stated reason
- [x] `CHANGELOG.md` `[Unreleased]` entry, in the user's terms
- [x] Book page under `docs/src/` updated (`initialization.md`; no new page,
      so no `SUMMARY.md` change — `check-docs-consistency.sh` is OK)
- [x] `cargo fmt --all -- --check`, `cargo clippy`, `cargo test` clean
- [x] Every claim in this body and in the code comments is true of the code
      as it stands now. The before/after numbers are all from
      `70bf53ded3d893cfa2da6ead5195fda5ac096f68` and this branch's head,
      taken with the same harness in the same session; the sweep was run
      against a release binary built from that commit in a separate
      worktree.
