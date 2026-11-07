# Agent guide for POUNCE

POUNCE is an interior-point NLP solver (a Rust port/reimagining of Ipopt
3.14). This file is the entry point for **LLM agents and automated tools**:
it points at the machine-driveable interfaces so you don't have to
rediscover them from source.

## Driving the solver programmatically

| You want to… | Use | Discover it via |
|---|---|---|
| Step/inspect/mutate a **live** solve | `pounce <model> --debug-json` | the `hello` handshake (self-describing) |
| Post-mortem a **finished** solve | pounce-studio MCP server, or `--json-output` | `python/pounce/studio/`, `docs/src/schema/` |
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

`python/pounce/studio/` (`pip install "pounce-solver[mcp]"`, then
`pounce-studio-mcp`; a Claude Desktop extension is built from
`studio/mcpb/`) is a FastMCP server with two tool families:

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

## Debugging discipline

Three rules, each earned on a solver bug that took far longer than it should
have (gh #505 is the worked example; see
`dev-notes/termination-status-invariants.md`).

- **Run the kill switch first.** Most POUNCE-only heuristics can be disabled by
  an option — `infeas_stationarity_tol=0`, `presolve=no`, `acceptable_iter=0`,
  `acceptable_progress_kappa=0`, `obj_scale_certificate_threshold=0`,
  `primal_noise_floor_kappa=0`, `dual_divergence_retry=no`,
  `nlp_scaling_method=none`. If one of them flips the outcome, that heuristic is
  the mechanism, established in a single run with no oracle and no code reading.
  On gh #505 this control was decisive and was not run until roughly eighteen
  hours and fifteen comments into the investigation.

- **A fixture built from a hypothesis is not evidence for that hypothesis.** A
  reduced model constructed to sit in the regime a theory predicts will
  reproduce the symptom whether or not the theory is right — and it will feel
  like confirmation. On gh #505 that fixture did reproduce *a* real defect,
  which was then assumed to be *the* defect; the reporter's own measurement
  falsified the route a day later, after a PR had already been opened on it.
  The fixture was eventually dropped altogether — once the actual arming
  condition was found and fixed (gh #519), it no longer reproduced anything.
  Reproduce on the reporter's artifact before writing the fix, and treat a
  purpose-built fixture as a regression test, never as a diagnosis.

- **Stamp every measurement with the commit it was taken on, when you take it.**
  Numbers from two build sets get quoted across each other otherwise. That
  happened twice on gh #505 — once publicly, caught by a reviewer who noticed a
  "bit-identical" claim between two numbers that differed in the fourth digit.
  When a correction lands, propagate it to *every* surface carrying the bad
  number: issue comment, PR body, commit message, code comment. The PR body is
  what survives as the merge commit.

## Repo conventions

- Build: `cargo build --release` (CLI binary at `target/release/pounce`).
- Test: `cargo test` (workspace) or `cargo test -p <crate>`.
- Docs: `make book` renders `docs/src/` (mdbook) to `docs/book/`.
- The user guide lives in `docs/src/`; `docs/src/SUMMARY.md` is its TOC.
- `gams/nlpbench/` and `benchmarks/` hold benchmark suites; the former is
  gitignored. Most suites' problems live outside the repo in
  `$POUNCE_BENCH_DATA` (see `benchmarks/README.md`). Its `SOURCES.md`
  records where every suite comes from; a new suite needs a row there, and
  a `SOURCE.md` with its authors, pinned version, licence and citation,
  before its results are reported.
- Complementarity (MPCC) answers are judged by recomputing `min(G, H)` and
  whatever the pairs encode from the returned point, never by solver status:
  an NLP solver given `G·H ≤ 0` reports success at complementarity-
  infeasible points. `make -C benchmarks mpcc-compare` runs pounce against
  Ipopt (MA57) and CCOpt this way.
