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
See [`docs/schema/solve-report-v1.md`](../../docs/schema/solve-report-v1.md)
for the field reference.

`--json-detail` knobs:

| Level | Emits |
|---|---|
| `summary` (default) | FAIR metadata, problem dims, final solution, aggregate statistics |
| `full` | Above + per-iteration trajectory + sensitivity / suffix blocks |

Choose `summary` for production logs; `full` for debugging (parallel
to upstream's `print_level=8`).

### Sensitivity analysis: `pounce_sens`

For AMPL `.nl` inputs declaring sIPOPT-style suffixes (`sens_state_1`,
`sens_state_value_1`, `sens_init_constr`), the `pounce_sens` binary
runs an IPM solve followed by a post-optimal sensitivity step:

```sh
pounce_sens problem.nl                    # writes problem.sol
pounce_sens problem.nl out.sol --json-output result.json --json-detail full
```

Output: an AMPL `.sol` file with the perturbed primal in a
`sens_sol_state_1` suffix, optionally a structured JSON report
mirroring everything in the `.sol` plus FAIR provenance + per-iter
history. Matches upstream sIPOPT's golden output to ~6e-9 per
component on `parametric_cpp` (see
[`tests/pounce_sens_end_to_end.rs`](tests/pounce_sens_end_to_end.rs)).

### Built-in problems

```sh
pounce --list-problems
pounce --problem quadratic
pounce --problem rosenbrock
```

`builtin.rs` ships two TNLPs that exercise the full pipeline without
parsing `.nl`:

- `quadratic` — `min (x[0]-3)² + (x[1]-4)²` (unconstrained, exact
  Hessian, optimum `(3, 4)`).
- `rosenbrock` — `min 100·(x[1]-x[0]²)² + (1-x[0])²` (unconstrained,
  exact Hessian, optimum `(1, 1)`).

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
```

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
