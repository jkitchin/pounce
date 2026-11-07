# pounce-studio-mcp

> **Where the code lives:** the server is the `pounce.studio` package of
> `pounce-solver` (`python/pounce/studio/`), shipped as the `[mcp]` extra.
> Its tests are in `python/tests/studio/`; the Claude Desktop extension is
> `studio/mcpb/`. This directory keeps the report fixtures (which
> `pounce-studio-core`'s tests also read) and `examples/`.

MCP server exposing pounce solve reports as Claude-callable tools. Lets any
MCP client (Claude Code, Claude Desktop, Cursor, Zed, ...) load a
`pounce.solve-report/v1` JSON file and ask questions about convergence,
restoration, stalls, and per-iteration state.

> **Lighter-weight alternative:** [`skills/pounce/`](../../skills/pounce/) ships the
> same analysis surface as a Claude skill that drives the `pounce-studio`
> CLI directly — no Python venv, no MCP registration, one-line install
> via `make install-skill`. Use the MCP server if you want structured
> tool-call UI or are driving a non-Claude MCP client; use the skill if
> you just want it to work everywhere with the minimum setup.

## Status

Phase 0. Post-mortem analysis of solve-report JSON and POUNCEIT v1
binary `.iterdump` traces. Backed by `pounce-studio-core` via PyO3 —
the Rust core is the single source of truth, this package marshals
across the FFI. The bindings are compiled into `pounce-solver`'s own
extension and reached as `pounce._pounce.studio`.

## Install

Python 3.10+. The server is the `[mcp]` extra of `pounce-solver`, whose
wheel already carries the `pounce` binary that `run_problem` and the live
debugger spawn — no Rust toolchain and no separate build:

```bash
pip install "pounce-solver[mcp]"
pounce-studio-mcp              # stdio MCP server
```

or, with nothing installed beforehand:

```bash
uvx --from "pounce-solver[mcp]" pounce-studio-mcp
```

(`python -m pounce.studio` is the same entry point.) The extra needs a
`pounce-solver` release newer than 0.16.0; 0.16.0 and earlier do not
contain `pounce.studio`.

**From a source checkout** — build the extension into a venv and register
it with Claude Code:

```bash
make install-mcp                   # user scope
make install-mcp MCP_SCOPE=local   # this project only
```

The GAMS example tools (`list_examples`, ...) read suites from a pounce
source checkout. Run from inside one, or set `POUNCE_REPO=/path/to/pounce`.

Environment variables:

| Variable         | Effect                                                              |
|------------------|---------------------------------------------------------------------|
| `POUNCE_WORKDIR` | Directory relative model paths resolve against (Desktop starts servers in `/`) |
| `POUNCE_BIN`     | `pounce` binary to spawn instead of the bundled one                 |
| `POUNCE_REPO`    | pounce source checkout, for the GAMS example suites                 |

## Tools

| Tool                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `analyze_problem`     | Inspect a builtin or `.nl` file: dimensions, class, option suggestions  |
| `run_problem`         | Shell out to the `pounce` CLI; return parsed report + embedded analysis |
| `verify_solution`     | Independently check a `.sol` against a canonical `.nl` (feasibility + optional KKT); returns a content-addressed, optionally HMAC-signed receipt — see note below |
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

### `verify_solution` notes — proof of concept

`verify_solution` lets an agent *request* an independent feasibility check it
cannot fake: the verdict comes from the `pounce verify` binary recomputing
`g(x*)` against the canonical `.nl`, not from the agent's claim. The
**feasibility check itself is solid**.

The **trust layer** around it is a **proof of concept**, not hardened:

* It signs the receipt (HMAC-SHA256) only when this server process holds
  `POUNCE_VERIFY_KEY`. That key only stays out of the agent's reach if the
  server runs in a **separate trust boundary** (different user / container /
  host). Over plain stdio on the same user as the agent, the signature adds
  nothing — the agent can read the env. Prefer the consumer simply
  re-running `pounce verify`, which needs no key.
* Tools take file **paths**, so remote use assumes a shared filesystem.
* Running this server remotely (`mcp.run("streamable-http")`) turns it into a
  network service that needs TLS, authn, sandboxing, and rate limits.

See [`docs/src/verify.md`](../../docs/src/verify.md) — "Status and
hardening" — for the full trust model and the checklist of what production
hardening would require, and `examples/signer_service.py` for a standalone
reference signer.

## Wire it into an MCP client

The same server speaks stdio MCP and works with any client that follows
the spec.

### Claude Desktop — one-click extension

Download `pounce-studio-<version>.mcpb` from the GitHub release (or build it
with `make mcpb`, which writes `dist/`) and open it: Claude Desktop installs
`pounce-solver[mcp]` with uv and asks for a **Workspace** folder, which
relative model paths resolve against. Source: [`studio/mcpb/`](../mcpb/).

### Claude Desktop — manual config

Add to `claude_desktop_config.json`
(macOS `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "pounce-studio": {
      "command": "uvx",
      "args": ["--from", "pounce-solver[mcp]", "pounce-studio-mcp"],
      "env": { "POUNCE_WORKDIR": "/abs/path/to/your/models" }
    }
  }
}
```

Desktop does not inherit your shell `PATH`; if `uvx` is not found, give its
absolute path (`which uvx`). Restart the app after editing the config.

### Claude Code

```bash
claude mcp add pounce-studio --scope user -- uvx --from "pounce-solver[mcp]" pounce-studio-mcp
```

or `make install-mcp` from a checkout (see [Install](#install)).

### Other MCP clients

Cursor, Zed, and Continue all consume the same `mcpServers` object —
check each client's docs for the file path.

### Any agent, via knowhow

[`mcp/pounce-studio.json`](../../mcp/pounce-studio.json) declares the
server in the portable `.mcp.json` format, so
[knowhow](https://github.com/jkitchin/knowhow) can write it into each
agent's own config (Claude Code, VS Code, Cursor, Codex, Gemini CLI,
OpenCode, ...):

```bash
uvx knowhow add jkitchin/pounce --mcp pounce-studio -a claude-code -a vscode
```

It launches the server through `uvx`, so nothing needs to be installed
first. The file lives in `mcp/` rather than a root `.mcp.json` on purpose:
Claude Code loads a root `.mcp.json` as this repository's own project
config, which would offer every contributor a server they did not ask for.

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
pip install -e "python[dev]"     # or: maturin develop -m python/pyproject.toml -E dev
python -m pytest python/tests/studio -v
```

The directory skips when `mcp` is not installed. The debugger tests spawn a
real `pounce`, and skip when none is found (bundled, `POUNCE_BIN`,
`target/release`, or `PATH`). The Rust side has its own tests covering the
same logic at the source — `cargo test -p pounce-studio-core`.

## Troubleshooting

**`pounce-studio-mcp needs the mcp package`**
Install the extra: `pip install "pounce-solver[mcp]"` (Python 3.10+).

**`No module named 'pounce.studio'`**
The installed `pounce-solver` predates the `[mcp]` extra (0.16.0 or older).
Upgrade it, or build from source.

**A relative model path is not found from Claude Desktop**
Desktop starts servers with `/` as the working directory. Set
`POUNCE_WORKDIR` (the extension's Workspace setting), or use absolute paths.

**`Error: unexpected schema "..." (expected "pounce.solve-report/v1")`**
The JSON file isn't a pounce solve report — maybe an AMPL `.sol`, a
different solver's output, or a hand-written file. Regenerate from the
pounce CLI with `--json-output <path>`.

**`load_solve_report` works but `convergence_trace` returns empty arrays**
The report was written at `--json-detail summary` (the default), which
omits per-iteration history. Rerun the solve with `--json-detail full`.

## See also

- `crates/pounce-studio-core/README.md` — the Rust core that does the
  actual analysis, plus the `pounce-studio inspect` CLI which renders
  the same Markdown summary for shell / CI use without an MCP client.
- `tools/iter-dump/FORMAT.md` — POUNCEIT v1 binary spec. The MCP tools
  currently only consume JSON solve reports; the binary `.iterdump`
  format is handled by `pounce-studio dump-summary <trace.bin>` on the
  CLI side, and exposed to Python via `pounce._pounce.studio.IterDump.from_path`.
