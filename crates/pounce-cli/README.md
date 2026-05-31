# pounce-cli

The `pounce` command-line driver. Solves built-in TNLPs and AMPL `.nl`
files. Console output is structured to mirror upstream `ipopt`'s
banner + per-iteration table + final summary, so anyone used to
reading `ipopt` logs can drop in `pounce` without relearning where
the numbers live.

## Install

From the workspace root:

```sh
make install                          # → $HOME/.local/bin/pounce
sudo make install PREFIX=/usr/local   # system-wide
```

Or build only the CLI:

```sh
cargo build -p pounce-cli --release
```

For the HSL MA57 backend (requires `libcoinhsl` on the link path):

```sh
cargo build -p pounce-cli --release --features ma57
```

## Usage

```sh
pounce problem.nl
pounce problem.nl print_level=8 max_iter=500 tol=1e-10
pounce problem.nl linear_solver=ma57            # with --features ma57
pounce problem.nl --options-file ipopt.opt      # upstream-format options file
```

Trailing `KEY=VALUE` pairs follow the same syntax and semantics as the
upstream Ipopt CLI; they override values loaded from `--options-file`.

### Degenerate / MPCC NLPs — ℓ₁-exact penalty-barrier wrapper

For problems where the standard IPM thrashes in restoration because
LICQ fails at the iterate (degenerate equalities, MPCC-like
complementarity), enable the Thierry-Biegler ℓ₁-exact penalty-barrier
wrapper:

```sh
pounce problem.nl l1_exact_penalty_barrier=yes
```

The wrapper turns every equality row `c_i(x) = g_i` into a slack-relaxed
`c_i(x) − p_i + n_i = g_i` with `(p_i, n_i) ≥ 0`, augments the
objective by `ρ · Σ(p + n)`, and runs a Byrd-Nocedal-Waltz outer loop
that escalates `ρ` until the slacks collapse (constraints satisfied)
or saturate (locally infeasible problem detected). The user-visible
`(x*, λ*)` are reported in the original variable space.

For everyday use, the simpler form is auto-fallback:

```sh
pounce problem.nl l1_fallback_on_restoration_failure=yes
```

Pounce first runs the standard solve. If it terminates in
`Restoration_Failed`, `Infeasible_Problem_Detected`,
`Solved_To_Acceptable_Level`, `Maximum_Iterations_Exceeded`, or
`Not_Enough_Degrees_Of_Freedom`, the wrapper is invoked transparently
and the result is promoted to `Solve_Succeeded` only if the retry
succeeds. Otherwise the original status is preserved.

Tuning knobs (all default-tuned; rarely need overriding):
`l1_penalty_init` (1.0), `l1_penalty_max` (1e6),
`l1_penalty_increase_factor` (8.0), `l1_penalty_max_outer_iter` (8),
`l1_slack_tol` (1e-6), `l1_steering_factor` (10.0). See
[`pounce-l1penalty`](../pounce-l1penalty/README.md) for the
algorithmic background.

### Machine-readable JSON output

```sh
pounce problem.nl --json-output result.json
pounce problem.nl --json-output result.json --json-detail full
```

The JSON report carries the same data the AMPL `.sol` file does
(status, primal `x`, dual `lambda`, suffixes) plus FAIR-aligned
provenance metadata (solver identity, schema version, timestamps).
See [`docs/src/schema/solve-report-v1.md`](../../docs/src/schema/solve-report-v1.md)
for the field reference.

`--json-detail` knobs:

| Level | Emits |
|---|---|
| `summary` (default) | FAIR metadata, problem dims, final solution, aggregate statistics |
| `full` | Above + per-iteration trajectory + sensitivity / suffix blocks |

Choose `summary` for production logs; `full` for debugging (parallel
to upstream's `print_level=8`).

### Sensitivity analysis

For AMPL `.nl` inputs declaring sIPOPT-style suffixes (`sens_state_1`,
`sens_state_value_1`, `sens_init_constr`), `pounce` auto-detects the
sensitivity request and runs a post-optimal sensitivity step after the
IPM solve — no separate binary or flag needed:

```sh
pounce problem.nl                    # writes problem.sol
pounce problem.nl out.sol --json-output result.json --json-detail full
```

Output: an AMPL `.sol` file with the perturbed primal in a
`sens_sol_state_1` suffix, optionally a structured JSON report
mirroring everything in the `.sol` plus FAIR provenance + per-iter
history. Matches upstream sIPOPT's golden output to ~6e-9 per
component on `parametric_cpp` (see
[`tests/pounce_sens_end_to_end.rs`](tests/pounce_sens_end_to_end.rs)).

Related flags:

* `--sens-boundcheck` / `--sens-bound-eps EPS` — clamp the perturbed
  primal `x* + Δx` onto the declared `[x_l, x_u]` box.
* `--compute-red-hessian` / `--rh-eigendecomp` — compute the reduced
  Hessian (and its eigendecomposition) over the variables tagged by
  the `red_hessian` integer var-suffix.

The `pounce_sens` binary is retained as a thin backward-compatibility
alias — `pounce_sens in.nl out.sol` is identical to `pounce in.nl
out.sol` — so existing AMPL/solver scripts keep working unchanged.

### Diagnostic dumps

```sh
pounce problem.nl --dump iterates:summary
pounce problem.nl --dump kkt:5-10+L --dump-dir ./my-dump
pounce problem.nl --dump kkt:5-10+L+Lvals
```

Per-iteration diagnostic captures land in `<dump-dir>` (default
`./pounce-dump-<ts>/`) as JSONL streams. Categories:

- `iterates:{summary,full}` — outer/restoration trajectory (issue #68).
  Consumed by [`pounce-studio`](../pounce-studio-core).
- `kkt[:spec][+L][+Lvals]` — augmented-system snapshots; `+L` adds the
  LDLᵀ factor pattern, `+Lvals` adds factor values (issue #69).

See `pounce --help` for the full grammar (`:5`, `:2-10`, `:5-:full`, …).

### Built-in problems

```sh
pounce --list-problems
pounce --problem quadratic
pounce --problem rosenbrock
```

`builtin.rs` ships several TNLPs that exercise the full pipeline
without parsing `.nl` (`--list-problems` is authoritative):

- `quadratic` — `min (x[0]-3)² + (x[1]-4)²` (unconstrained, exact
  Hessian, optimum `(3, 4)`).
- `rosenbrock` — `min 100·(x[1]-x[0]²)² + (1-x[0])²` (unconstrained,
  exact Hessian, optimum `(1, 1)`).
- `bounded-quadratic` — `quadratic` with box bounds `0 ≤ x ≤ 2`.
- `eq-quadratic` — `min x[0]² + x[1]²` s.t. `x[0] + x[1] = 1`.
- `circle` — `min x[0]` s.t. `x[0]² + x[1]² = 1`.
- `infeasible-eq` — two contradictory equalities (infeasibility path).

### AMPL / Pyomo solver mode

AMPL drivers — and Pyomo's ASL interface — invoke a solver as
`solver problem.nl -AMPL`. Pass `-AMPL` to run pounce that way:

```sh
pounce problem.nl -AMPL
```

It changes nothing about the solve itself; it switches the process to
the AMPL exit-code contract (see [Exit codes](#exit-codes)), so the
driver reads the termination from the `.sol` file rather than the exit
status. The [`pyomo-pounce`](../../pyomo-pounce) package registers
pounce as a Pyomo `SolverFactory` solver on top of this.

### Help

```sh
pounce --help
pounce --version          # also -v, -V
pounce --about            # solver, license, FERAL/HSL backend, build target
```

### Flag reference

The recipes above cover the common paths; this is the full flag set for
look-up. Trailing `KEY=VALUE` pairs are option overrides, not flags.

| Flag | Argument | Notes |
|------|----------|-------|
| `--problem` | `NAME` | Run a built-in TNLP. See `--list-problems`. |
| `--nl-file` | `PATH` | Explicit form of the positional `.nl` argument; useful when scripting alongside other flags. |
| `--options-file` | `PATH` | Upstream-format options file; trailing `KEY=VALUE` pairs override it. |
| `--sol-output` | `PATH` | Override the default `<input>.sol` output path. |
| `--no-sol` | — | Suppress `.sol` writing entirely (used by harnesses that only consume `--json-output`). |
| `--json-output` | `PATH` | Emit a `pounce.solve-report/v1` JSON report. |
| `--json-detail` | `summary` \| `full` | Detail level for the JSON report (default `summary`). |
| `--dump` | `cat[:spec][+L][+Lvals]` | Per-iteration diagnostic capture. May be repeated. |
| `--dump-dir` | `PATH` | Override the default `./pounce-dump-<ts>/` location. |
| `--dump-format` | `jsonl` | Reserved; `jsonl` is the only format today. |
| `--sens-boundcheck` | — | Clamp the sensitivity-perturbed primal onto `[x_l, x_u]`. |
| `--sens-bound-eps` | `EPS` | Tolerance for the boundcheck clamp. |
| `--compute-red-hessian` | — | Compute the reduced Hessian over the `red_hessian` var-suffix set. |
| `--rh-eigendecomp` | — | Also return the reduced-Hessian eigendecomposition. |
| `-AMPL` | — | AMPL solver-protocol mode (exit-code contract; see [Exit codes](#exit-codes)). |
| `--list-problems` | — | List the built-in TNLP names. |
| `--about` | — | Print solver identity, license, linked linear backends, build target. |
| `-h` / `--help` | — | Show the full grammar. |
| `-v` / `-V` / `--version` | — | Crate version. |

## Exit codes

- `0` — `Solve_Succeeded` (or `Solved_To_Acceptable_Level`).
- non-zero — any other `ApplicationReturnStatus`.

In AMPL solver mode (`-AMPL`) the exit code instead follows the AMPL
contract: `0` for any solve that ran and produced a `.sol` file —
limit-reached, infeasible, even a failed solve — since the termination
is carried by the file's `solve_result_num`. Genuine startup failures
(unreadable `.nl`, bad option) still exit non-zero.

## What's exposed as a library

`pounce_cli` also re-exports its argv parser and built-in problems
so the integration tests can drive the same code without invoking
`main`. The `nl_reader` / `nl_tape` modules implement the AMPL `.nl`
parser used by the CLI.

## License

EPL-2.0.
