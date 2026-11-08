# pounce-studio-pyo3

PyO3 bindings for [`pounce-studio-core`](../pounce-studio-core). Linked
into `pounce-solver`'s own extension (`crates/pounce-py`) and mounted as
the submodule `pounce._pounce.studio` via [`register`]; the crate's own
`_native` module entry point is kept for standalone builds.

The Python MCP server in `python/pounce/studio/` (the `[mcp]` extra of
`pounce-solver`) delegates all
analysis through these bindings, so the Rust core stays the single
source of truth for derived series and diagnostics.

## FFI strategy

The bindings return **JSON strings** rather than Python objects; the
Python wrapper does the `json.loads`. This keeps the code trivial on
both sides, avoids a `pythonize` dependency, and is plenty fast at our
data sizes (a Full-detail solve report is a few hundred KB).

Method names intentionally omit a `_json` suffix: the Python wrapper
hides the marshalling, so `Report.summarize()` returns a parsed dict
even though the Rust side returns a string.

## Surface

- `Report` (`pyclass`) — a parsed `pounce.solve-report/v1` document.
  Construct via `from_bytes` / `from_path`. Methods: `summarize`,
  `convergence_trace`, `find_stalls`, `restoration_windows`,
  `diagnose`, `get_iterate`, `render_markdown`, `linear_solver_summary`.
  Parameter-less results are memoised per-instance for cheap repeat
  MCP-tool calls.
- `IterDump` (`pyclass`) — a parsed iter-dump trace. Construct via
  `from_bytes` / `from_path`. Methods: `header`, `records`,
  `record_count`.
- `compare_reports(pairs)` — compare a set of named reports.
- `register(module)` — adds all of the above to a module; `pounce-py`
  calls it on `pounce._pounce.studio`.
- `_native` (`pymodule`) — a standalone module built from `register`.

## Build

`maturin` sets the `extension-module` feature when building the wheel;
plain `cargo build` leaves it off so workspace CI stays
Python-toolchain-free. The crate is `publish = false` — it ships only
inside the `pounce-solver` wheel.

```sh
make install-mcp      # or: maturin develop -m python/pyproject.toml -E mcp
```

## License

EPL-2.0.
