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

A Rust toolchain (1.75+) is required because the package builds a
native extension via maturin.

```bash
cd studio/mcp
python3 -m venv .venv
source .venv/bin/activate
pip install maturin mcp
maturin develop --release   # builds _native.*.so and installs editable
```

This exposes a `pounce-studio-mcp` console script that speaks MCP over
stdio.

## Tools

| Tool                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `load_solve_report`   | Validate a JSON report and return a headline summary                    |
| `convergence_trace`   | Per-iteration trajectory as column-oriented arrays (optionally filtered)|
| `get_iterate`         | Full per-iteration record for iter `k`, with derived log10 fields       |
| `find_stalls`         | Windows where log-residual barely moved (configurable threshold)        |
| `restoration_windows` | Contiguous runs of iters tagged as restoration                          |
| `diagnose`            | Common Ipopt-failure heuristics with severity-tagged findings           |
| `compare_runs`        | Side-by-side comparison of multiple reports                             |

## Wire it into an MCP client

The same server (`pounce-studio-mcp`) speaks stdio MCP and works with any
client that follows the spec. The config object is identical across
clients; only the file location differs.

### Claude Code

Add to `~/.claude/settings.json` (user-wide) or
`.claude/settings.json` in a project:

```json
{
  "mcpServers": {
    "pounce-studio": {
      "command": "pounce-studio-mcp"
    }
  }
}
```

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
