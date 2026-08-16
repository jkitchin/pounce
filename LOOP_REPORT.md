VERDICT: improved

Fixes #609

Baseline for every measurement below: **`cfc11218d75b1751d4bb2730be30bf35814a4016`**
(tip of `main` when this branch was cut). No Rust changed.

## Problem

The Pyomo initialization stack had four measurable gaps. Each was
reproduced on the baseline before anything was written; the harness that
produced these numbers is the same script run against both trees, with
`PYTHONPATH` selecting which `pyomo_pounce` is imported and the same
`pounce` binary (built from `cfc11218`) serving both.

### 1. Options stopped at the projection

`initialize(model, options={"max_iter": 137, "tol": 1e-9})`, with the
solver's `solve` wrapped to record what each stage was handed:

| stage | baseline `cfc11218` | after |
|---|---|---|
| projection | `{max_iter: 137, tol: 1e-9}` | `{max_iter: 137, tol: 1e-9, nlp_scaling_method: user-scaling}` |
| block subsystem solve | `{}` | `{max_iter: 137, tol: 1e-9}` |
| stages dropping the option | `['block_subsystem']` | `[]` |

`block_initialize` had no `options` argument at all, so the stage that
produces the starting point you actually get ran on solver defaults.

### 2. A failed block abandoned independent branches

Two branches sharing no variable: a 2x2 whose solution lies outside
`p`'s bounds, and an unrelated `q -> r` chain.

| | baseline | after |
|---|---|---|
| `q` (exact 7.0) | **0.0** (seed) | 7.0 |
| `r` (exact 8.0) | **0.0** (seed) | 8.0 |
| `n_vars_initialized` | 0 | 2 |
| failed block's own seeds | restored | restored |

### 3. A near-singular square block returned wrong values, silently

`u + v == 2`, `u + (1+eps)v == 2 + eps` (Jacobian condition number
~`4/eps`), feeding `w == u + 2v`. Exact answer `u = v = 1`, `w = 3`.

| eps | baseline values | baseline error | after error | diagnostic |
|---|---|---|---|---|
| 1e-10 | `u=2, v=0, w=2` | **1.0** | 7.54e-09 | rcond 2.50e-11, routed to fallback |
| 1e-14 | `u=2, v=0, w=2` | **1.0** | 7.54e-09 | rcond 2.49e-15, routed to fallback |

The baseline reported `report.ok == True` and said nothing.

### 4. The projection merit was unscaled

**Rows.** A 1e6-unit energy balance beside a 1e-6-unit trace balance,
both true at `a = b = 1`:

| | baseline | after |
|---|---|---|
| relative residual, energy row | 1.48e-16 | 0.0 |
| relative residual, trace row | **1.18e-08** | 0.0 |
| error vs exact | 7.87e-09 | **0.0** |

**Variables.** A pressure at 1e6 Pa and a mole fraction at 1e-4,
coupled so each contributes 1.0 in its own units, with a 10% deficit:

| | baseline | after |
|---|---|---|
| relative move, pressure | 1.75e-12 | 0.100 |
| relative move, mole fraction | 0.200 | 0.100 |
| ratio | **1.14e+11** | **1.000** |

**Row-rescale invariance** (acceptance criterion 1) was *already met* on
the baseline and is reported honestly as such — see Blast radius.

## Cause

1. `repair.initialize` called `block_initialize(model, solver=..., tee=...,
   repair="off", igraph=...)` — no `options`, because the parameter did
   not exist. There was also nowhere to put a knob that is not a solver
   option, so the scaling/conditioning/recovery policies did not exist.
2. The block loop `break`s on the first failed block. The comment said
   stopping was "the conservative choice" for blocks that "typically
   feed on the failed one" — but the loop had no representation of which
   blocks actually did, so it could not distinguish them.
3. The Dulmage–Mendelsohn partition establishes *structural*
   solvability. Nothing checked numerical rank, so a square block with a
   near-null direction went to `calculate_variable_from_constraint` /
   the subsystem solve and landed wherever that direction took it.
4. The merit was `sum((v - v0)**2)` — an absolute distance, so across
   mixed units a repair goes to whichever variable the constraint
   gradient favours regardless of physical meaning. And POUNCE's default
   gradient-based row scaling follows Ipopt's `min(1, g_max/||grad c||)`,
   which scales a large row **down** and leaves a small row alone; an
   absolute convergence test then enforces a 1e-6-unit row eight orders
   of magnitude more loosely than a 1e6-unit one.

## Fix

**`pyomo_pounce/init_options.py`** — `InitOptions`, frozen, validated,
carrying `solver_options`, `tee`, `scaling`, `conditioning`, `cond_tol`,
`fallback`, `regularization`, `on_block_failure`, `max_list`. One is
built per `initialize()` call and threaded through the projection, every
block solve, and every fallback solve. `InitOptions.coerce` keeps a bare
mapping meaning solver options and **never** reinterprets it as policy —
POUNCE has a solver option called `scaling`, so guessing would silently
change a solve.

**`pyomo_pounce/init_scaling.py`** — two-sided row normalisation
(`1/||grad c||_inf`) and per-anchor variable weights (`w = 1/|v0|`, with
a floor so an anchor at zero stays free to move and a cap on the
weight spread). Row factors are delivered through the model's own
`scaling_factor` Suffix plus `nlp_scaling_method=user-scaling` — the
route the NL writer already emits and `pyomo_pounce.scaling` already
reads — so the solver learns nothing new, a user's own entry simply
wins, and the Suffix is restored entry-by-entry afterwards (a model that
declared none does not acquire one).

**`block_init.py`** — `BlockAnalysisReport.block_dependencies` is the
block DAG, read off the incidence graph already in hand; a failure skips
its transitive descendants via reverse reachability and other branches
continue. Each block is rank-checked before solving, on a *scaled*
Jacobian, and a weak one is routed to a regularized least-squares
fallback. `BlockOutcome` records index/size/constraint/status/rcond/
depends_on per block.

**Rejected along the way**, so nobody repeats them:

* *Rounding row factors to powers of two* to make the scaling exact in
  binary. Implemented, measured, removed: it did not move the residual
  1.2e-9 row-rescale spread at all (1.2496e-9 → 1.2200e-9), which
  disproved the hypothesis that the spread came from factor rounding.
  It is a solver stopping-point effect — tightening `tol` to 1e-12
  collapses the spread to 2.2e-16.
* *Column-normalising the Jacobian* for the conditioning check. It
  drives every 1x1 block to exactly rcond 1.0, retiring the check on
  most of a typical calculation order. Columns are scaled by their
  variable's magnitude instead, which keeps 1x1 blocks informative.
* *Ridging the whole merged group* in the coupled fallback. Measured: it
  picked `u = 2, v = 0` — precisely the arbitrary end of the near-null
  direction the fallback exists to avoid, because the downstream block's
  pull toward its own seed decided the weak block's answer. The ridge
  now covers only the weak block's own variables.
* *Raising the default `regularization`.* On the regularized path the
  error **is** the ridge bias and falls linearly with it (7.5e-9 at
  1e-8, 7.5e-5 at 1e-4), so the default stays small. The coupled path is
  limited by the ridge gradient against the solver tolerance instead and
  wants a larger value (7.4e-3 at 1e-8, 7.6e-5 at 1e-6); that is
  documented rather than forced into one default.

## Blast radius

**Fixture sweep: empty diff, 57/57 fixtures bit-identical.**
`scripts/sweep-fixtures.sh` run against a release binary built from the
untouched parent `cfc11218` in a separate worktree, and against a
release binary built from this branch; `diff` of the two outputs is
empty. This is the expected result — the change is entirely Python-side
under `pyomo-pounce/` and no Rust file, `Cargo.toml`, or `Cargo.lock`
is touched (`git diff --stat -- '*.rs' Cargo.toml Cargo.lock crates/`
is empty). The sweep was run anyway because a scaled projection merit
changes the point later solves start from, which is exactly the
trajectory class `CLAUDE.md` warns about; the corpus sweep drives the
CLI on `.nl` fixtures and does not exercise `pyomo-pounce`, so it bounds
the Rust side only. The Python-side trajectory change is bounded by the
before/after tables above instead, which is where it is visible.

**Row-rescale invariance is preserved, and this is the one number that
moved the wrong way.** Multiplying a row by a constant does not change
the feasible set, so the projection must land in the same place:

| rescale factor k | baseline max deviation | after (default tol) | after (`tol=1e-12`) |
|---|---|---|---|
| 1e6 | 0.0 | 1.25e-09 | 5.6e-17 |
| 1e-6 | 0.0 | 0.0 | 0.0 |
| 1e10 | 0.0 | 1.25e-09 | 2.2e-16 |
| 1e-10 | 0.0 | 2.2e-16 | 2.2e-16 |
| nonlinear rows, worst over k ∈ {1e±6, 1e10} | 2.2e-16 | 2.2e-16 | — |

The baseline reached exact invariance; this branch reaches 1.25e-9 at
the default tolerance. It is a *tolerance* effect, not a scaling defect,
and the evidence is that tightening `tol` to 1e-12 collapses it to
2.2e-16 — with normalised rows the solver stops as soon as the scaled
residual is inside `tol`, and where that happens depends on the iterate
path. It also is not a quality regression: on the same model the
baseline's "invariant" answer sits 2.8e-11 from the true projection
while this branch's is exact to machine precision at k=1. The criterion
says "invariant within tolerance", and 1.25e-9 is inside the default
1e-8. Recorded here rather than left for the next reader to rediscover.

**Behaviour changes for existing callers**, all opt-out:

* The projection now installs a temporary `scaling_factor` Suffix and
  turns on `nlp_scaling_method=user-scaling`, unless the caller set
  `nlp_scaling_method` themselves, the model carries a *non*-export
  `scaling_factor` Suffix (left strictly alone — flipping it to EXPORT
  would ship values the user deliberately kept local), or there is
  nothing to scale. `scaling="none"` restores the old merit exactly.
* The merit is weighted, so the projected point moves on any model whose
  anchors span magnitudes. On models whose anchors are all the same
  magnitude the weights are equal and the result is unchanged — the
  existing `test_project_repairs_inconsistent_fill` still gets 1/3 each.
* A failed block no longer stops the traversal. `on_block_failure="stop"`
  restores it. Dependent blocks are still skipped, so
  `test_failure_stops_before_downstream_blocks` is unaffected.
* Weak blocks are rerouted. `conditioning="off"` skips the check;
  `fallback="off"` diagnoses without rerouting.

**Conflict note.** `claude/pounce-issue-608-loop` (PR #625) conflicts
with this branch in **two files**, both trivially:

* `pyomo-pounce/pyomo_pounce/__init__.py` — one hunk, adjacent import
  lines directly after the `block_init` import block. #608 adds
  `from pyomo_pounce.continuation import continuation, shift_map`; this
  branch adds `from pyomo_pounce.init_options import InitOptions`.
  Resolution is to keep both lines. The `__all__` additions
  (`continuation`/`shift_map` vs `BlockOutcome`/`InitOptions`) land in
  different parts of the list and merge cleanly.
* `CHANGELOG.md` — both add an entry at the top of `[Unreleased]`.
  Resolution is to keep both entries.

`repair.py` and `block_init.py` are untouched by #608, and
`continuation.py` is untouched by this branch, so there is no semantic
overlap — only the two textual hunks above. Verified with
`git merge-tree --write-tree HEAD origin/claude/pounce-issue-608-loop`.

## Tests

`pyomo-pounce/tests/test_issue_609_init_hardening.py`, 29 tests. Every
number in their comments is a measurement on `cfc11218`.

**How they bite.** As written the module does not import on the parent
at all — `InitOptions` does not exist — so collection fails there and
that alone is not evidence. Re-run against the parent with the
`InitOptions` import and the new-attribute assertions stripped, so only
parent-era API is used, **nine fail behaviourally and for the right
reason**:

| test | parent failure |
|---|---|
| `small_magnitude_row_is_enforced_like_a_large_one` | `0.9999999921 != 1.0 ± 1e-11` |
| `merit_shares_repair_by_relative_magnitude` | pressure moved `1.75e-12`, expected `0.2` |
| `user_scaling_suffix_entries_win_over_the_automatic_ones` | `0.9999999921 != 1.0 ± 1e-11` |
| `options_reach_every_stage` | `('block_subsystem', {})` — option dropped |
| `independent_branch_survives_a_failure` | `q = 0.0`, expected `7.0` |
| `near_singular_block_...[1e-10]`, `[1e-14]` | `u = 2.0`, expected `1.0` |
| `healthy_blocks_are_not_diagnosed` | `AttributeError: no attribute 'diagnostics'` |
| `block_initialize_takes_options_directly` | `TypeError: unexpected keyword argument 'options'` |

The last two are **API absence, not a behavioural bite**, and are listed
as such rather than dressed up: there is no parent-era way to ask "was a
diagnostic emitted" when the field does not exist.

**Two tests pass on the parent by design** — they are preservation
guards, not regression tests:

* `test_projection_leaves_no_scaling_suffix_behind` holds the Suffix
  restoration.
* `test_still_one_incidence_walk_per_initialize` holds gh #444's single
  whole-model incidence walk, which is an acceptance criterion here.

Both bite when broken. Making `_block_dependencies` construct an
`IncidenceGraphInterface` instead of reading the one already in hand
fails **four** cache guards: the two in this file and
`test_shared_graph_matches_fresh_analysis` /
`test_shared_graph_zero_decision_falls_back_to_fresh` in
`test_repair.py`. That was run and reverted, not assumed.

**Suite results** (Pyomo 6.10.1, so the `pyomo_pounce.v2` floor is met;
the CLI binary was built and staged into `python/pounce/bin/` so the
plugin resolves this checkout's build rather than falling back to PATH):

| suite | result |
|---|---|
| `pyomo-pounce/tests` | **372 passed, 0 skipped** (343 before this branch + 29 new) |
| `python/tests` | 815 passed, 36 skipped |
| `cargo fmt --all -- --check` | clean |
| `cargo clippy --workspace --exclude pounce-hsl --all-targets` | exit 0 (pre-existing warnings only; no Rust touched) |
| `cargo test --workspace --exclude pounce-hsl` | exit 0, 226 suites green |
| `scripts/check-release-consistency.sh` | OK |
| `scripts/check-docs-consistency.sh` | OK — 44 pages, no dead links |

**Not pinned, deliberately.** The row-rescale invariance test asserts at
`tol=1e-12` rather than the default, because at the default tolerance the
quantity it measures is the solver's stopping behaviour and not the
scaling. Pinning 1.25e-9 at the default would pin solver noise.

---

## Acceptance criteria

| # | criterion | verdict |
|---|---|---|
| 1 | Initialization is invariant within tolerance under equivalent row rescaling | **met** — 2.2e-16 at `tol=1e-12`; 1.25e-9 at the default, inside it. Already held on the baseline; kept, and now pinned by `test_projection_is_invariant_under_row_rescaling`. The deviation at default tolerance is recorded above rather than hidden. |
| 2 | The same options reach every selected initialization stage | **met** — stages dropping the option: `['block_subsystem']` → `[]` |
| 3 | A failure in one branch does not prevent initialization of independent branches | **met** — `n_vars_initialized` 0 → 2; `q, r` reach 7.0, 8.0 |
| 4 | Near-singular structurally square blocks produce a diagnostic/fallback rather than unstable values | **met** — error 1.0 → 7.5e-9, with `rcond` and a `diagnostics` entry |
| 5 | Existing incidence-plan caching from #444 is preserved | **met** — one walk per call, on the success *and* the failure path; four guards fail if broken |

## Scope items

| # | item | verdict |
|---|---|---|
| 1 | Row/variable scaling in the projection merit, user scaling suffix support, robust automatic defaults | **met** — `init_scaling.py`; suffix entries win; floor + spread cap as the robustness guards |
| 2 | One typed initialization-options object, propagated through repair, block solve, projection, and fallback | **met** — `InitOptions`, built once per call and threaded through all four |
| 3 | Track the block graph explicitly; on failure skip dependent descendants but continue independent branches | **met** — `BlockAnalysisReport.block_dependencies` + reverse reachability |
| 4 | Numerical rank/conditioning checks after structural matching; route weak blocks to regularized least squares or a coupled fallback | **met** — scaled-Jacobian rcond vs `cond_tol`; both `fallback="regularized"` and `fallback="coupled"` implemented and tested |
| 5 | Structured report of initialized, skipped, failed, and fallback blocks | **met** — `BlockOutcome` + `report.blocks` and the four bucket properties |

Nothing deferred; no follow-up issues filed.
