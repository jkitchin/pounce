# Issue #15 — Tier A, Wave 1: option wiring notes

**Status: wired.** All ten options below now drive solver behavior.

Scope: wire the lowest-effort / highest-impact stub options identified in
issue #15. Three groups:

1. `max_cpu_time`, `max_wall_time` — time-budget termination
2. `dual_inf_tol`, `compl_inf_tol` — fill out the convergence-tolerance triplet
3. `acceptable_*` (6 options) — relaxed-convergence acceptance

This file is a working reference: where each option is registered, where
the comparable live option is consumed today, what to edit, and what to
test. Citations are `file:line` against pounce HEAD at the time of writing.

## Summary of what landed

- `crates/pounce-common/src/timing.rs` — added
  `TimedTask::live_wallclock_time` / `live_cpu_time` so the conv check
  can read elapsed time mid-solve.
- `crates/pounce-algorithm/src/ipopt_cq.rs` — added
  `curr_complementarity_max` (max-norm of unbarriered complementarity
  blocks) to support the per-component compl gate.
- `crates/pounce-algorithm/src/conv_check/trait.rs` — extended
  `ConvergenceStatus` with `CpuTimeExceeded` / `WallTimeExceeded`;
  added `current_is_acceptable_with_state` and
  `set_curr_acceptable_obj` trait methods.
- `crates/pounce-algorithm/src/conv_check/opt_error.rs` — new fields
  (`dual_inf_tol`, `constr_viol_tol`, `compl_inf_tol`, plus the four
  `acceptable_*` per-component tolerances, plus
  `last_acceptable_obj`); `check_convergence_with_state` now enforces
  the upstream per-component gate + time budgets; the
  `passes_component_tols` / `passes_acceptable_tols` pure helpers are
  unit-tested.
- `crates/pounce-algorithm/src/alg_builder.rs` — new
  `ConvCheckOptions` struct on `AlgorithmBuilder`; `build_inner`
  bakes its fields into the assembled `OptErrorConvCheck`.
- `crates/pounce-algorithm/src/application.rs` — reads each option
  off `OptionsList` and pushes it into the builder's
  `ConvCheckOptions`.
- `crates/pounce-algorithm/src/ipopt_alg.rs` — main loop now maps the
  two new convergence statuses; `current_is_acceptable_with_state`
  replaces the scalar call; the obj at the stashed iterate is
  recorded via `set_curr_acceptable_obj`.
- Tests: 4 new unit tests in `opt_error.rs`; 4 new integration tests
  in `tests/optimize_hs71.rs` covering the dual/compl gate, both time
  budgets, and the acceptable-streak success path.

## 1. Where the options are registered

All ten options are already registered in
`crates/pounce-algorithm/src/upstream_options.rs` under the "Termination"
category (line 213):

| Line | Option                       | Type   | Default |
|-----:|------------------------------|--------|--------:|
| 215  | `max_wall_time`              | number | `1e20`  |
| 216  | `max_cpu_time`               | number | `1e20`  |
| 217  | `dual_inf_tol`               | number | `1.0`   |
| 219  | `compl_inf_tol`              | number | `1e-4`  |
| 220  | `acceptable_tol`             | number | `1e-6`  |
| 221  | `acceptable_iter`            | int    | `15`    |
| 222  | `acceptable_dual_inf_tol`    | number | `1e10`  |
| 223  | `acceptable_constr_viol_tol` | number | `1e-2`  |
| 224  | `acceptable_compl_inf_tol`   | number | `1e-2`  |
| 225  | `acceptable_obj_change_tol`  | number | `1e20`  |

Defaults mirror upstream Ipopt 3.14. No changes needed in `upstream_options.rs`.

## 2. How a live option is consumed today (reference pattern)

`tol` (live):
- read at `crates/pounce-algorithm/src/application.rs:431` via
  `get_numeric_value("tol", "")` (default `1e-8`)
- assigned into `data.borrow_mut().tol` at line 435

`constr_viol_tol` (live):
- read at `crates/pounce-algorithm/src/application.rs:323`
  (default `1e-4`)
- consumed by `orig_nlp.relax_bounds(bound_relax_factor, constr_viol_tol)`
  at line 327

Pattern: read options in `IpoptApplication::optimize_tnlp`, push the
values into the struct(s) used at iterate time. The wave-1 work follows
this pattern.

## 3. Existing convergence-check infrastructure

`OptErrorConvCheck` (`crates/pounce-algorithm/src/conv_check/opt_error.rs:14-22`)
already carries the fields we need:

```text
tol, acceptable_tol, acceptable_iter, max_iter,
max_cpu_time, max_wall_time, acceptable_count
```

`max_cpu_time` / `max_wall_time` fields exist but the main loop never
checks them. `acceptable_tol` and `acceptable_iter` exist but are
populated from hard-coded defaults, not from `OptionsList`.

Acceptance probe (`current_is_acceptable`,
`crates/pounce-algorithm/src/conv_check/opt_error.rs:68-78`) currently
gates only on `nlp_err <= acceptable_tol`. Upstream's
`OptimalityErrorConvergenceCheck::CurrentIsAcceptable` also checks the
per-criterion tolerances (`acceptable_dual_inf_tol`,
`acceptable_constr_viol_tol`, `acceptable_compl_inf_tol`) and the
objective-change tolerance (`acceptable_obj_change_tol`). Wave 1 expands
the probe to match.

Call site for the acceptance probe:
`crates/pounce-algorithm/src/ipopt_alg.rs:317` — decides whether to call
`store_acceptable_point()`.

Return code plumbing already exists:
- `pounce-nlp/src/return_codes.rs:15` — `SolvedToAcceptableLevel`
- `pounce-algorithm/src/application.rs:638` — maps to
  `StopAtAcceptablePoint`

## 4. Where the main loop lives

`IpoptAlgorithm::optimize` main loop:
`crates/pounce-algorithm/src/ipopt_alg.rs:1071-1100`

Convergence call inside `iterate`:
`crates/pounce-algorithm/src/ipopt_alg.rs:290-295`
```rust
self.bundle.conv_check.check_convergence_with_state(
    nlp_err, iter_count, &self.data, &self.cq)
```

Termination match: `ipopt_alg.rs:296-310`
(`Continue | Converged | MaxIterExceeded | Failed`).

`MaxIter` is currently checked inside `check_convergence_with_state`
against the hard-coded value at line 1087. We'll need a parallel
`MaxCpuTimeExceeded` / `MaxWallTimeExceeded` branch (or fold into
`MaxIterExceeded`-style "stop now" status with the right return code).

## 5. Time tracking

No elapsed-since-start field on `IpoptData` today. Timing infrastructure
exists:

- `IpoptData.timing: Rc<TimingStatistics>`
  (`crates/pounce-algorithm/src/ipopt_data.rs:33-103`, field at 102)
- `TimingStatistics`
  (`crates/pounce-common/src/timing.rs:129-153`)
- `overall_alg: TimedTask` is started in
  `crates/pounce-algorithm/src/application.rs:281`
- `TimedTask::total_wallclock_time()` / `total_cpu_time()` already give
  elapsed seconds (`pounce-common/src/timing.rs:14-94`)

Wave 1 plan: read `overall_alg.total_wallclock_time()` and
`.total_cpu_time()` from inside `check_convergence_with_state`. No new
state needed.

Upstream return codes:
- `MaximumCpuTimeExceeded`
- `MaximumWallTimeExceeded`

Confirm both exist in `pounce-nlp/src/return_codes.rs` before wiring; add
if missing.

## 6. Tests

Integration tests: `crates/pounce-algorithm/tests/*.rs`
Examples to mirror:
- `optimize_hs71.rs:144` — `hs071_solves_via_application`
- `optimize_hs71.rs:195` — `hs071_solves_with_penalty_line_search`

Unit tests live alongside implementation, e.g.
`crates/pounce-algorithm/src/conv_check/opt_error.rs:81-105`.

Wave 1 should add at minimum:

- `max_cpu_time=1e-9` → solve returns `MaximumCpuTimeExceeded` (or wall
  equivalent) immediately
- Non-default `acceptable_tol` triggers `SolvedToAcceptableLevel` on a
  problem that doesn't reach `tol`
- `dual_inf_tol` / `compl_inf_tol` set tight enough to *prevent*
  convergence on a problem that otherwise converges; loose enough to
  allow it.

## 7. Order of work

1. `dual_inf_tol` / `compl_inf_tol` — smallest change; modifies the
   convergence struct's per-criterion gate.
2. `max_cpu_time` / `max_wall_time` — adds two new termination branches;
   needs return-code additions if missing.
3. `acceptable_*` — wire the six options through application → conv
   check, expand `current_is_acceptable` to match upstream.

Each step is one commit with its test.

## Out of scope (wave 1)

- Output / logging options (wave 2)
- `mu_init` / `mu_max` / `mu_min` / `mu_target` (wave 2)
- Watchdog options (wave 2)
- Anything in tier B / C / D
