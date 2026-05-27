# pounce-studio-mcp

MCP server exposing pounce solve reports as Claude-callable tools. Lets any
MCP client (Claude Code, Claude Desktop, Cursor, Zed, ...) load a
`pounce.solve-report/v1` JSON file and ask questions about convergence,
restoration, stalls, and per-iteration state.

## Status

Phase 0. Post-mortem analysis of solve-report JSON and POUNCEIT v1
binary `.iterdump` traces. Backed by `pounce-studio-core` via PyO3 —
the Rust core is the single source of truth, this package marshals
across the FFI.

## Install

A Rust toolchain (1.75+) and Python 3.10+ are required.

**One-liner — build, install, and register with Claude Code:**

```bash
cd studio/mcp
./install.sh --register --user
```

This:
1. Builds the `pounce` CLI (`cargo build --release -p pounce-cli`).
2. Creates a venv at `studio/mcp/.venv` (override with `POUNCE_VENV=...`).
3. Installs `maturin`, `mcp`, and the native extension (`maturin develop --release`).
4. Runs the test suite to confirm the build is healthy.
5. With `--register`: invokes `claude mcp add pounce-studio ...` so Claude
   Code picks it up immediately. `--user` registers it at user scope so it's
   available in every project. Drop `--user` for project-local scope.

After install, `claude mcp list` should show `pounce-studio: ✓ Connected`.

**Build without registering** (manual config or non-Claude-Code clients):

```bash
./install.sh
```

The script prints the absolute path to the installed binary and a copy-pasteable
JSON snippet for Claude Desktop / Cursor / Zed at the end.

**Manual install** (if you prefer step-by-step):

```bash
cd studio/mcp
python3 -m venv .venv
source .venv/bin/activate
pip install maturin mcp
maturin develop --release
```

Either path exposes a `pounce-studio-mcp` console script that speaks MCP over
stdio.

## Tools

| Tool                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `analyze_problem`     | Inspect a builtin or `.nl` file: dimensions, class, option suggestions  |
| `run_problem`         | Shell out to the `pounce` CLI; return parsed report + embedded analysis |
| `explain`             | Glossary lookup for per-iter columns and diagnose finding codes         |
| `citations`           | Curated paper references by subsystem topic or bib key                  |
| `load_solve_report`   | Validate a JSON report and return a headline summary                    |
| `convergence_trace`   | Per-iteration trajectory as column-oriented arrays (optionally filtered)|
| `get_iterate`         | Full per-iteration record for iter `k`, with derived log10 fields       |
| `find_stalls`         | Windows where log-residual barely moved (configurable threshold)        |
| `restoration_windows` | Contiguous runs of iters tagged as restoration                          |
| `diagnose`            | Common Ipopt-failure heuristics with severity-tagged findings           |
| `compare_runs`        | Side-by-side comparison of multiple reports                             |
| `linear_solver_summary` | Aggregate FERAL backend post-mortem: factor counts, fill ratio, extremal pivots, final inertia |
| `list_gams_examples`  | Enumerate bundled .gms instances (globallib, princetonlib, mittelmann, powerflow, examples, smoke) |
| `analyze_gams_problem` | Inspect a .gms file: dims (vars/eqs/NL nnz), `Solve` directive, model class, option suggestions |
| `parse_gams_listing`  | Parse a .lst SOLVE SUMMARY block + the embedded `--- POUNCE` solver status block |
| `run_gams_problem`    | Run a .gms through `gams NLP=POUNCE` with JSON report capture; returns parsed lst + report summary |

### `run_problem` notes

`run_problem` locates the `pounce` binary via `POUNCE_BIN`, then by
walking up from the installed package looking for `target/release/pounce`,
then via `$PATH`. Set `POUNCE_BIN` explicitly if your MCP client runs
the server with a stripped environment.

By default, `analyze=True` runs `analyze_problem` first and embeds the
result under `analysis`. Suggestions there are **advisory** — they are
never auto-applied; the agent decides whether to re-run with them
forwarded via the `options` arg.

## Wire it into an MCP client

The same server (`pounce-studio-mcp`) speaks stdio MCP and works with any
client that follows the spec. The config object is identical across
clients; only the file location differs.

### Claude Code

The fastest path is `./install.sh --register --user` (see [Install](#install)).
That runs the equivalent of:

```bash
claude mcp add pounce-studio --scope user \
    --env "POUNCE_BIN=$PWD/../../target/release/pounce" \
    -- "$PWD/.venv/bin/pounce-studio-mcp"
```

Drop `--scope user` for project-local scope (writes `.mcp.json` in the cwd).

### Claude Desktop

Add to `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux** (community builds): `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pounce-studio": {
      "command": "pounce-studio-mcp"
    }
  }
}
```

If `pounce-studio-mcp` is installed inside a venv that the desktop app
can't see on `$PATH`, give the absolute path instead:

```json
{
  "mcpServers": {
    "pounce-studio": {
      "command": "/abs/path/to/studio/mcp/.venv/bin/pounce-studio-mcp"
    }
  }
}
```

Restart the app after editing the config.

### Other MCP clients

Cursor, Zed, and Continue all consume the same `mcpServers` object —
check each client's docs for the file path. The `command` value is
always `pounce-studio-mcp` (or an absolute path).

### Example prompts

Once wired up:

- "Analyze the rosenbrock builtin and then run it; tell me if any suggested options would help."
- "Run my `mymodel.nl` with max_iter=500 and diagnose the result."
- "Load `studio/mcp/fixtures/rosenbrock-stalled.json` and diagnose what went wrong."
- "Compare `rosenbrock.json` against `rosenbrock-stalled.json` and tell me what changed."
- "Show the inf_du trajectory and identify where the search direction quality degraded."

## Generate fresh solve reports

The bundled `fixtures/` were generated from the CLI:

```bash
# from the repo root
make build
./target/release/pounce --problem rosenbrock \
    --json-output studio/mcp/fixtures/rosenbrock.json \
    --json-detail full
```

`--json-detail full` is required for per-iteration history. `--json-detail
summary` still works with `load_solve_report` and `diagnose`, but
`convergence_trace`, `get_iterate`, `find_stalls`, and `restoration_windows`
return empty.

## Tests

```bash
cd studio/mcp
source .venv/bin/activate
python -m pytest tests/ -v
```

The Rust side has its own tests covering the same logic at the source —
run with `cargo test -p pounce-studio-core` from the repo root.

## Troubleshooting

**`ImportError: cannot import name '_native' from 'pounce_studio_mcp'`**
The native extension hasn't been built. Activate the venv and run
`maturin develop --release` inside `studio/mcp/`. If pytest still
fails with this even after a successful build, you're probably running
Python from a working directory that puts `studio/mcp/` on `sys.path`
and shadows the installed wheel — activate the venv (`source
.venv/bin/activate`) so the editable install wins.

**`maturin: Couldn't find a virtualenv or conda environment`**
`maturin develop` needs an active venv to know where to install. Create
one with `python3 -m venv .venv && source .venv/bin/activate` and rerun.

**Wheel install fails with `Cannot uninstall <pkg>, RECORD file not found`**
A system-Python install is blocking pip's upgrade path. Always work
inside a venv for this package; never `pip install` it against the
system interpreter.

**Rust toolchain too old**
PyO3 0.22 requires rustc 1.75+. Update via `rustup update stable`.

**`Error: unexpected schema "..." (expected "pounce.solve-report/v1")`**
The JSON file isn't a pounce solve report — maybe an AMPL `.sol`, a
different solver's output, or a hand-written file. Regenerate from the
pounce CLI with `--json-output <path>`.

**`load_solve_report` works but `convergence_trace` returns empty arrays**
The report was written at `--json-detail summary` (the default), which
omits per-iteration history. Rerun the solve with `--json-detail full`.

**`pounce-studio-mcp: command not found` from an MCP client**
Either the venv isn't activated for the client's environment, or the
client doesn't inherit your shell `$PATH`. Use the absolute path to the
venv-installed script (see the Claude Desktop snippet above).

## See also

- `crates/pounce-studio-core/README.md` — the Rust core that does the
  actual analysis, plus the `pounce-studio inspect` CLI which renders
  the same Markdown summary for shell / CI use without an MCP client.
- `tools/iter-dump/FORMAT.md` — POUNCEIT v1 binary spec. The MCP tools
  currently only consume JSON solve reports; the binary `.iterdump`
  format is handled by `pounce-studio dump-summary <trace.bin>` on the
  CLI side, and exposed to Python via `_native.IterDump.from_path`.
