# Running Solves

The `pounce` command-line driver solves built-in TNLPs and AMPL `.nl`
files. Its console output mirrors upstream `ipopt`'s banner,
per-iteration table, and final summary, so anyone used to reading
`ipopt` logs can read `pounce` logs unchanged.

## Basic usage

```sh
pounce problem.nl
pounce problem.nl print_level=8 max_iter=500 tol=1e-10
pounce problem.nl linear_solver=ma57            # with --features ma57
pounce problem.nl --options-file ipopt.opt      # upstream-format options file
```

Trailing `KEY=VALUE` pairs follow the same syntax and semantics as the
upstream Ipopt CLI; they override values loaded from `--options-file`.
See [Solver Options](options.md).

## Built-in problems

```sh
pounce --list-problems
pounce --problem quadratic
pounce --problem rosenbrock
```

- `quadratic` — `min (x[0]-3)² + (x[1]-4)²` (unconstrained, optimum
  `(3, 4)`).
- `rosenbrock` — `min 100·(x[1]-x[0]²)² + (1-x[0])²` (unconstrained,
  optimum `(1, 1)`).
- `bounded-quadratic` — `quadratic` with box bounds `0 ≤ x ≤ 2`
  (optimum at the upper corner `(2, 2)`).
- `eq-quadratic` — `min x[0]² + x[1]²` s.t. `x[0] + x[1] = 1`
  (a single equality).
- `circle` — `min x[0]` s.t. `x[0]² + x[1]² = 1` (a nonlinear equality).
- `infeasible-eq` — two contradictory equalities (`x[0]+x[1]=1` and
  `=2`); exercises the infeasibility-detection path.

Run `pounce --list-problems` for the authoritative list.

Built-in problems have no `.nl` stub, so they only write a `.sol` file
when `--sol-output` is given explicitly.

## Degenerate / MPCC NLPs — the ℓ₁-exact penalty-barrier wrapper

For problems where the standard IPM thrashes in restoration because
LICQ fails at the iterate (degenerate equalities, MPCC-like
complementarity), enable the Thierry–Biegler ℓ₁-exact penalty-barrier
wrapper:

```sh
pounce problem.nl l1_exact_penalty_barrier=yes
```

The wrapper turns every equality row `c_i(x) = g_i` into a
slack-relaxed `c_i(x) − p_i + n_i = g_i` with `(p_i, n_i) ≥ 0`,
augments the objective by `ρ · Σ(p + n)`, and runs a
Byrd–Nocedal–Waltz outer loop that escalates `ρ` until the slacks
collapse (constraints satisfied) or saturate (locally infeasible
problem detected). The user-visible `(x*, λ*)` are reported in the
original variable space.

For everyday use, the simpler form is an auto-fallback:

```sh
pounce problem.nl l1_fallback_on_restoration_failure=yes
```

POUNCE first runs the standard solve. If it terminates in
`Restoration_Failed`, `Infeasible_Problem_Detected`,
`Solved_To_Acceptable_Level`, `Maximum_Iterations_Exceeded`, or
`Not_Enough_Degrees_Of_Freedom`, the wrapper is invoked transparently
and the result is promoted to `Solve_Succeeded` only if the retry
succeeds. Otherwise the original status is preserved.

The tuning knobs are listed under [Solver Options](options.md).

## AMPL / Pyomo solver mode

AMPL drivers — and Pyomo's ASL interface — invoke a solver as
`solver problem.nl -AMPL`. Pass `-AMPL` to run `pounce` that way:

```sh
pounce problem.nl -AMPL
```

It changes nothing about the solve itself; it switches the process to
the AMPL exit-code contract (see below), so the driver reads the
termination from the `.sol` file rather than the exit status. The
[`pyomo-pounce`](pyomo.md) package builds on top of this mode.

## Exit codes

- `0` — `Solve_Succeeded` (or `Solved_To_Acceptable_Level`).
- non-zero — any other `ApplicationReturnStatus`.

In AMPL solver mode (`-AMPL`) the exit code instead follows the AMPL
contract: `0` for any solve that ran and produced a `.sol` file —
limit-reached, infeasible, even a failed solve — since the termination
is carried by the file's `solve_result_num`. Genuine startup failures
(unreadable `.nl`, bad option) still exit non-zero.

## Diagnostics & introspection

```sh
pounce --about                                   # version, build info, features, backends
pounce problem.nl --dump kkt:5-10 --dump iterate # dump per-iteration diagnostics
pounce problem.nl --dump kkt --dump-dir /tmp/d   # override the dump root
```

- `--about` — print version, build info, enabled features, and linear-solver
  backends, then exit.
- `--dump <cat>[:<spec>]` — write the diagnostic category to per-iteration
  files (JSONL). Wired categories are `kkt` and `iterate`; an optional
  `:<spec>` selects iterations (e.g. `kkt:5`, `kkt:2-10`, `iterate:all`).
- `--dump-dir <path>` — override the dump root (default
  `./pounce-dump-<timestamp>`).
- `--dump-format <fmt>` — dump format (default `jsonl`).

## Help

```sh
pounce --help
pounce --version          # also -v, -V
```
