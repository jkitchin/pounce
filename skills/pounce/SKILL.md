---
name: pounce
description: Install pounce, and inspect and diagnose pounce solve reports via the pounce-studio CLI. Use when the user asks how to install or set up pounce or pounce-studio, or asks to analyze a pounce.solve-report JSON, diagnose a stuck or non-converging solve, compare two runs, look up a per-iteration column or finding code, pre-flight an AMPL .nl or GAMS .gms file before solving, check or fix a starting point (check-x0), set up a warm start, or solve or interpret a model with complementarity constraints (AMPL complements, Pyomo mpec, MPCC).
---

# pounce — solver post-mortem and pre-flight

This skill wraps the `pounce-studio` CLI for AI-assisted post-mortem of
pounce solve reports plus pre-flight inspection of `.nl` and `.gms`
input files. It is the CLI-and-skill counterpart to the
`pounce-studio-mcp` MCP server (in `studio/mcp/`); both backends
ultimately call the same Rust analysis core, so findings agree.

Use this skill when the user asks any of:
- "diagnose this solve report"
- "did pounce converge?"
- "compare these two runs"
- "what does `inf_pr` mean?"
- "is this .nl file going to be a problem?"
- "what happened in iteration 47?"
- "how do I install pounce?"
- "my model has complementarity constraints / is an MPCC"

## Installing pounce

Two binaries, installed separately:

- **`pounce`** — the solver. Needed to run solves.
- **`pounce-studio`** — the analysis CLI every recipe below calls.
  Needed for everything except a bare solve.

Check first: `pounce -v` and `pounce-studio --help`. Install only what is
missing.

**`pounce`** — pick by what the user already has:

```sh
pip install pounce-solver        # Python 3.9+, prebuilt wheels for Linux,
                                 # macOS, Windows; no Rust needed. Puts the
                                 # `pounce` CLI on PATH and installs the
                                 # `pounce` Python package.
pip install pyomo-pounce         # add this for Pyomo models
cargo install pounce-cli --bin pounce --locked   # Rust toolchain, no Python
docker pull ghcr.io/jkitchin/pounce              # nothing installed on the host
```

**`pounce-studio`** is not in the pip wheel and has no prebuilt binary; it
needs a Rust toolchain (<https://rustup.rs>):

```sh
cargo install pounce-studio-core --bin pounce-studio --locked
```

`cargo install` puts binaries in `~/.cargo/bin/`, which must be on `PATH`.
In a clone of the repo, `make install-skill` builds and installs both
binaries into `~/.local/bin/` instead. Full options, including extras
(`pounce-solver[jax]`, `[torch]`, `[gams]`) and the GAMS link:
`docs/src/installation.md`, or
<https://jkitchin.github.io/pounce/installation.html>.

## Tools at your disposal

All output is pretty-printed JSON on stdout — pipe through `jq` to
slice. If `pounce-studio` is not on `PATH`, invoke it via its absolute
path (e.g. `~/.cargo/bin/pounce-studio` or `~/.local/bin/pounce-studio`,
depending on how it was installed).

### Post-mortem on a JSON solve report

```sh
pounce-studio summary <report>                # headline outcome
pounce-studio diagnose <report>               # failure-mode heuristics
pounce-studio find-stalls <report>            # stalled-progress windows
    [--min-window N] [--max-progress P]
pounce-studio convergence-trace <report>      # per-iter trajectory
    [--columns iter,inf_pr,inf_du,mu,d_norm,...]
pounce-studio get-iterate <report> <k>        # full iter k record
pounce-studio restoration-windows <report>    # restoration entry→exit cycles
pounce-studio compare <r1> <r2> ...           # side-by-side
    [--labels A,B,C]
pounce-studio linear-solver-summary <report>  # FERAL post-mortem
pounce-studio inspect <report> [--json]       # Markdown summary by default;
                                              # --json dumps the whole report
```

The report must be at JSON detail `full` for the iter-level tools
(`convergence-trace`, `get-iterate`, `find-stalls`, `restoration-windows`).
The `summary` and `diagnose` tools work on either detail level.

### Glossary

```sh
pounce-studio explain <term>                  # column or finding code
pounce-studio citations [--topic T] [--key K] # curated papers
```

Known column names: `iter`, `objective`, `inf_pr`, `inf_du`, `mu`,
`d_norm`, `regularization`, `alpha_dual`, `alpha_primal`,
`alpha_primal_char`, `ls_trials`, `log10_*`, plus the linear-solver
summary fields (`n_factors`, `n_pattern_reuse`, `max_fill_ratio`, …).

Known finding codes: `converged`, `max_iter_exceeded`,
`restoration_used`, `restoration_loop`, `mu_stuck`,
`heavy_line_search`, `hessian_regularized`, `convergence_stall`.

### Pre-flight on an input file

```sh
pounce-studio analyze-nl <path.nl>            # AMPL .nl header
pounce-studio analyze-nl --builtin <name>     # builtin problem metadata
pounce-studio analyze-gms <path.gms>          # GAMS .gms header + suggestions
pounce-studio parse-gams-listing <path.lst>   # GAMS .lst SOLVE SUMMARY
pounce-studio list-gams [--suite S]           # bundled .gms instances
pounce-studio list-builtins                   # builtin --problem options
```

The suggestion blocks are **advisory** — they are never auto-applied.
Decide based on the user's intent whether to pass any of them to a
fresh solve.

### Starting-point pre-flight (`check-x0`, on the `pounce` binary)

```sh
pounce check-x0 <path.nl> [--json]            # evaluate the model at x0
pounce check-x0 <path.nl> --x0-file cand.txt  # ... at a candidate point
pounce check-x0 --builtin <name> --json
```

Evaluates the model once at its starting point (no solve) and reports
NaN/inf evaluations (exit 21 — a solve WOULD abort with
`Invalid_Number_Detected`), bound violations, how far the `bound_push`
interior clamp will move the point, initial constraint violation per
row, and derivative scale spread. Run it before the first solve of a
new model, after any `Invalid_Number_Detected`, and whenever a warm
start saved no iterations.

## Running a fresh solve

`pounce-studio` does the analysis; the `pounce` CLI does the solve.
The skill's job is to chain them.

### From a builtin or `.nl` file

```sh
pounce --problem rosenbrock \
    --json-output /tmp/report.json --json-detail full \
    max_iter=500 mu_strategy=adaptive

# or
pounce path/to/model.nl \
    --json-output /tmp/report.json --json-detail full \
    tol=1e-7
```

CLI option key=value pairs go at the end (the same format as upstream
ipopt). `--json-detail full` is required for per-iter tools.

After the solve, immediately diagnose:

```sh
pounce-studio summary /tmp/report.json
pounce-studio diagnose /tmp/report.json
```

If `diagnose` reports `convergence_stall` or `mu_stuck`, drill in:

```sh
pounce-studio find-stalls /tmp/report.json
# pick a midpoint iter K from the window
pounce-studio get-iterate /tmp/report.json <K>
```

### Through GAMS

When the user has a `.gms` file and the `gams` CLI installed (`gams`
must be on PATH or set `GAMS_BIN`), drive it like so:

```sh
cd /a/working/dir
cp /path/to/model.gms .

cat > pounce.opt <<EOF
print_level 0
mu_strategy adaptive
tol 1e-6
json_output $(pwd)/model.report.json
json_detail full
EOF

gams model.gms NLP=POUNCE optfile=1 lo=2
```

GAMS writes a listing file `model.lst` and a log `model.log`. To
extract the SOLVE SUMMARY block:

```sh
pounce-studio parse-gams-listing model.lst
pounce-studio summary model.report.json
pounce-studio diagnose model.report.json
```

Pre-flight the `.gms` first if the user wants option suggestions:

```sh
pounce-studio analyze-gms model.gms
```

## Workflow recipes

### "Did this solve work?"

```sh
pounce-studio summary /path/to/report.json
```

The `status` field (`SolveSucceeded`, `MaximumIterationsExceeded`, …)
and `final_kkt_error` tell you. If the status is not
`SolveSucceeded`, follow up with `diagnose`.

### "Why is this solve stuck?"

```sh
pounce-studio diagnose /path/to/report.json
```

The findings carry codes the user can ask you to explain via
`pounce-studio explain <code>`. Walk through them in severity order
(`error` → `warning` → `info`).

### "Compare these runs"

```sh
pounce-studio compare --labels baseline,adaptive a.json b.json
```

Returns one row per report with status, iter count, objective, KKT
error, restoration calls, elapsed seconds.

### "What does this column mean?"

```sh
pounce-studio explain inf_pr
pounce-studio explain alpha_primal_char
```

Returns a definition, typical numeric range, what abnormal values
mean, and `see_also` citation keys.

### "This model won't start / initialize" — the initialization playbook

Reference: the *Initialization and Warm Starts* chapter of the docs
(`docs/src/initialization.md`).

1. **Preflight.** `pounce check-x0 model.nl --json`. `"fatal": true`
   means the model does not evaluate at its start — fix that first
   (in-domain values, or bounds that keep the clamp in the domain).
   Not fatal: read the warnings in order.
2. **Interpret.**
   - *all-zeros x0* → the model supplies no start. Set values
     (Pyomo `Var.value` / GAMS `x.L`), or generate them:
     `pounce.generate_starts(...)` (Python),
     `pyomo_pounce.initialize_missing_values(model)` /
     `pyomo_pounce.block_initialize(model)` (Pyomo).
   - *on-bound components + clamp moves* on a re-solve → the previous
     solution is being discarded; use the warm-start recipe:
     `ws = pounce.WarmStart.from_info(x, info); prob.solve(warm_start=ws)`
     from Python, or the `warm_start_init_point=yes mu_init=1e-7
     warm_start_*=1e-9` option set on the CLI.
   - *very large initial infeasibility* →
     `least_square_init_primal=yes`, or repair the point:
     `pounce.project_to_feasible(...)`.
   - *large derivative scale spread* → a scaling problem wearing an
     initialization costume; go to the scaling recipes.
3. **Re-solve with `--json-output ... --json-detail full` and
   `pounce-studio diagnose`.** `restoration_used` findings that
   disappear after re-initialization confirm the diagnosis; a clean
   preflight plus a failing solve means the start is NOT the problem —
   switch to the troubleshooting recipes.

The MCP twin (`suggest_initialization` in `pounce-studio-mcp`) runs
step 1 and emits these suggestions mechanically; treat its output as
advisory, exactly like the `analyze-*` suggestion blocks.

### "This model has complementarity constraints" — the MPCC recipe

Reference: the *Complementarity Constraints (MPCC)* chapter
(`docs/src/complementarity.md`); worked examples in
`python/notebooks/47_declared_complementarity.ipynb`.

1. **Recognize it.** `pounce-studio analyze-nl model.nl` reports the
   class as `…, N complementarity rows (MPCC)` and says what changes (the
   header count includes rows whose variable is free, which are read as
   equations; two-sided "mixed" pairs are solved as mixed complementarity).
   From Python, `pounce.read_nl(path).complementarity` lists the pairs.
2. **Declare, don't lower.** Pass the pairs as data and set nothing:
   AMPL `complements` and Pyomo `mpec` (`SolverFactory("pounce")`, no
   transformation) carry them in the `.nl`; from Python,
   `pounce.Problem(..., complementarity=[(row, var), ...])`. Do not
   hand-write `G·H = 0`, a Scholtes schedule or a penalty — the driver
   runs the continuation and the exact product and keeps the better
   certified answer. A pair is one-sided (a row with one finite side; the
   variable's matching bound is in the pair, a second one an ordinary
   limit) or mixed (`l <= x <= u ⟂ g` with the row free); a boxed row is
   refused with the reason. Cone complementarity is not supported.
3. **Read the verdict.** The JSON report does **not** carry the
   certificate yet, so `diagnose` cannot see it. Read the console line
   (`pounce: complementarity: answered by …; stationarity: S`) or Python
   `info["complementarity"]`:
   - `S` — strongly stationary, a local-minimum candidate. Done.
   - `W` (`fallback` set, `accepted_by` None) — weakly stationary only:
     a descent direction may switch which side of a pair is zero. Try
     other starts.
   - `none` with status `Solved_To_Acceptable_Level` — converged and
     feasible but not stationary; the solve stalled. Treat as a failure.
   - `not_searched` — not determined (the multipliers form a set too
     large to search). Accepted like S; say so when reporting.
4. **Cost.** Running both formulations costs about the iterations of
   both. When one is known to work for the model,
   `complementarity_strategy=exact` (or `=continuation`) runs only it.
5. **What refuses pairs.** `--minima`, sensitivity / reduced-Hessian
   requests, the debugger, `pounce verify`, `check-x0`, the WASM build,
   `pounce.Solver`, `find_minima`, `solve_nlp_batch` and
   `solve_with_sens` refuse a model with pairs rather than solve it
   without them; presolve is declined. Use `Problem.solve()` or the
   plain CLI.

### "Cite this for me"

```sh
pounce-studio citations --topic interior_point
pounce-studio citations --key wachter2006
pounce-studio citations                       # list all topics + keys
```

## Notes on JSON detail levels

- `--json-detail summary` (default for `pounce`): writes outcome,
  statistics, no per-iter history. Compatible with `summary`,
  `diagnose`, `linear-solver-summary`, and `compare`. The iter-level
  tools (`convergence-trace`, `get-iterate`, `find-stalls`,
  `restoration-windows`) return empty arrays / errors.
- `--json-detail full`: adds per-iter history and suffix blocks.
  Required for the iter-level tools.

When kicking off a fresh solve to feed into the post-mortem tools,
always pass `--json-detail full`.

## Troubleshooting

**`pounce-studio: command not found`**
Not installed, or installed but not on PATH. Install it with `cargo
install pounce-studio-core --bin pounce-studio --locked` (see *Installing
pounce*). If it is installed, `~/.cargo/bin/` (or `~/.local/bin/` after
`make install-skill`) is missing from `PATH`: add it, or call the binary
by its absolute path, e.g. `~/.cargo/bin/pounce-studio summary
report.json`.

**`pounce: command not found`** (when running fresh solves)
The solver is missing. `pip install pounce-solver` is the quickest fix;
`cargo install pounce-cli --bin pounce --locked` works without Python. If
it was pip-installed, the `pounce` script lives in the active Python
environment's `bin/` (`Scripts\` on Windows): activate that environment.

**`unexpected schema "..."`**
The JSON file is not a `pounce.solve-report/v1` document — perhaps an
AMPL `.sol`, a different solver's output, or a hand-written file.
Re-run the solve with `--json-output <path>`.

**`summary` works but `convergence-trace` returns empty arrays**
The report was written at `--json-detail summary`. Re-run the solve
with `--json-detail full`.

**`gams: Could not spawn solver`**
The GAMS POUNCE link is not installed. See `gams/Makefile` for
build/install instructions.
