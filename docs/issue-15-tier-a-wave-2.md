# Issue #15 — Tier A, Wave 2: option wiring notes

**Status: wired.** Fifteen more options now drive solver behavior.

Scope: the next tier of stub options identified in issue #15, grouped
into three areas:

1. **Barrier (μ) options** — `mu_init`, `mu_max`, `mu_max_fact`,
   `mu_min`, `mu_target`, `mu_linear_decrease_factor`,
   `mu_superlinear_decrease_power`, `mu_allow_fast_monotone_decrease`
2. **Watchdog options** — `watchdog_shortened_iter_trigger`,
   `watchdog_trial_iter_max`
3. **Output / logging** — `print_frequency_iter`,
   `print_frequency_time`, `output_file`, `file_print_level`,
   `file_append` (plus `print_timing_statistics`, already wired in
   wave 1, now also fans the report out to the journalist)

## Summary of what landed

- `crates/pounce-algorithm/src/alg_builder.rs` — three new option
  groups on `AlgorithmBuilder`: `MuOptions`, `LineSearchOptions`,
  `OutputOptions`. `build_inner` bakes each into the assembled
  strategy:
  - Monotone μ-update: all eight `mu_*` knobs.
  - Adaptive μ-update: `mu_init`, `mu_max`, `mu_max_fact`, `mu_min`,
    `mu_linear_decrease_factor`, `mu_superlinear_decrease_power`
    (`mu_target` / `mu_allow_fast_monotone_decrease` are
    monotone-only upstream).
  - `BacktrackingLineSearch`: the two watchdog counters.
  - `OrigIterationOutput`: `print_frequency_iter`,
    `print_frequency_time`.
- `crates/pounce-algorithm/src/mu/monotone.rs` — added `mu_max` field
  (default `1e5`, clamps `mu_init` in `initialize`) and
  `mu_allow_fast_monotone_decrease` (default `true`, caps the
  reduction loop at a single μ step when off).
- `crates/pounce-algorithm/src/application.rs` —
  `algorithm_builder_from_options` now reads every wave-2 option off
  the `OptionsList` and pushes it into the builder. Added
  `open_output_file_journal`, called from both
  `initialize_with_options_str` / `initialize_with_options_file` —
  attaches a `FileJournal` honoring `file_print_level` /
  `file_append`. The timing report is now also routed through the
  journalist (gated on `print_timing_statistics yes`) so the file
  journal picks it up.
- `crates/pounce-common/src/reg_options.rs` — `is_valid_string` now
  treats a registered `"*"` entry as a wildcard (matches upstream's
  behavior for free-form options like `output_file`). Without this
  fix, any `set_string_value("output_file", "...")` was rejected as
  invalid.
- Tests:
  - `tests/optimize_hs71.rs` — three new integration tests:
    `hs071_solves_with_nondefault_mu_init`,
    `hs071_solves_with_sparse_iter_output`,
    `hs071_output_file_captures_timing_report` (covers the full
    `initialize_with_options_str → open_output_file_journal →
    journalist.print` chain).

## Known limitation: iteration rows still bypass the journalist

Pounce's per-iteration output (`OrigIterationOutput::format_row`) is
emitted to stdout via `println!` rather than through the journalist.
With wave 2 the file journal is attached and the end-of-solve timing
report is fanned out to it, but per-iter rows do not yet land in
`output_file`. Routing the iter-output path through the journalist is
its own piece of work — kept out of scope here.

## Out of scope (wave 2)

- Anything else from tier B / C / D.
- Re-routing iteration rows through the journalist (see note above).
