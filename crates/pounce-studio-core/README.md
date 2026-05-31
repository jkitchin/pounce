# pounce-studio-core

Pure-Rust parsers and analysis helpers for pounce solve reports and
POUNCEIT iter-dumps. Foundation crate for the `pounce-studio` GUI work
(VS Code extension + desktop app + MCP server).

Phase 0 of the studio roadmap.

## What it does

- Parses `pounce.solve-report/v1` JSON (writer:
  `crates/pounce-cli/src/solve_report.rs`) into typed Rust structs.
- Parses `POUNCEIT v1` binary iter-dumps (writer:
  `crates/pounce-algorithm/src/iter_dump.rs`; format spec:
  `tools/iter-dump/FORMAT.md`) into typed records.
- Computes derived series: convergence-stall windows, restoration-phase
  windows, per-iteration column trace.
- Runs failure-mode diagnostics (max-iter, mu-stuck, line-search
  collapse, restoration loops, regularization growth).
- Renders a Markdown inspection report.

Output mirrors the Python MCP server (`studio/mcp/`) one-for-one so the
desktop / VS Code shells and Claude tools agree on every notion.

## WASM-readiness

The library is intentionally I/O-free: it takes `&[u8]` slices and
returns owned data. No `std::fs`, no env reads, no panics on bad input
(everything goes through `Result<T, Error>`). The bundled `pounce-studio`
binary is the one piece that touches the filesystem, and that target is
built only for native platforms.

This means the same crate will compile to `wasm32-unknown-unknown` for
the VS Code webview when we get there.

## Use

```bash
# inspect a JSON solve report as Markdown
cargo run -p pounce-studio-core --bin pounce-studio -- \
    inspect studio/mcp/fixtures/rosenbrock.json

# inspect a POUNCEIT v1 binary trace
cargo run -p pounce-studio-core --bin pounce-studio -- \
    dump-summary trace.iterdump
```

Programmatic:

```rust
use pounce_studio_core::{SolveReport, analysis};

let bytes = std::fs::read("report.json")?;
let report = SolveReport::from_json_slice(&bytes)?;
let summary = analysis::summarize(&report);
let findings = analysis::diagnose(&report);
```

## Tests

```bash
cargo test -p pounce-studio-core
```

43 tests (31 unit + 12 integration) covering JSON parsing, schema
validation, binary parsing, stall detection, diagnose heuristics, and
Markdown rendering against real solver-produced fixtures.
