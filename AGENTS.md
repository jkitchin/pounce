# Agent guide for POUNCE

POUNCE is an interior-point NLP solver (a Rust port/reimagining of Ipopt
3.14). This file is the entry point for **LLM agents and automated tools**:
it points at the machine-driveable interfaces so you don't have to
rediscover them from source.

## Driving the solver programmatically

| You want to… | Use | Discover it via |
|---|---|---|
| Step/inspect/mutate a **live** solve | `pounce <model> --debug-json` | the `hello` handshake (self-describing) |
| Post-mortem a **finished** solve | pounce-studio MCP server, or `--json-output` | `studio/mcp/`, `docs/src/schema/` |
| Solve and get machine-readable output | `pounce <model> --json-output r.json --json-detail full` | `docs/src/schema/solve-report-v1.md` |

### Interactive debugger — `--debug-json` (live)

A *pdb for the interior-point loop*. Launch `pounce <model.nl> --debug-json`
(or `--problem <name>` for a built-in) with stdin/stdout piped. The **first
line is a `hello` event** that enumerates everything you can do —
`commands`, `events`, `checkpoints`, `metrics`, `blocks`, and a
`capabilities` map. Feature-detect off those lists, not the version string.
Then send `{"cmd":"…","id":N}` lines and read `pause` / `progress` /
`terminated` events; every event carries the scalar metrics under the names
listed in `hello.metrics` (`objective`, `mu`, `inf_pr`, `inf_du`,
`nlp_error`, `complementarity`, `iter`). Stop with `{"cmd":"continue"}`
(run to completion) or `{"cmd":"quit"}`.

Full contract and a worked transcript: **`docs/src/debugger.md`**
(see "For an LLM agent: the whole contract"). Human REPL variants:
`--debug`, `--debug-on-error`, `--debug-on-interrupt`,
`--debug-script <file>`.

### pounce-studio MCP server

`studio/mcp/` is a FastMCP server with two tool families:

- **Post-mortem** (`diagnose`, `find_stalls`, `restoration_windows`,
  `convergence_trace`, `compare_runs`, `run_problem`, …) — analyze a
  **finished** `pounce.solve-report/v1` JSON.
- **Live debug sessions** (`debug_start`, `debug_command`, `debug_state`,
  `debug_sessions`, `debug_close`) — a stateful proxy over `--debug-json`.
  `debug_start` spawns and parks a solver child; `debug_command` steps it
  one command at a time. This drives the live debugger over MCP without
  the agent managing the child process or the wire framing.

`debug_session_guide` documents the underlying protocol (for callers
driving `--debug-json` directly instead of through the proxy).

## Repo conventions

- Build: `cargo build --release` (CLI binary at `target/release/pounce`).
- Test: `cargo test` (workspace) or `cargo test -p <crate>`.
- Docs: `make book` renders `docs/src/` (mdbook) to `docs/book/`.
- The user guide lives in `docs/src/`; `docs/src/SUMMARY.md` is its TOC.
- `gams/nlpbench/` and `benchmarks/` hold benchmark suites; the former is
  gitignored.
