# pounce-studio-mcp

MCP server exposing pounce solve reports as Claude-callable tools. Lets any
MCP client (Claude Code, Claude Desktop, Cursor, Zed, ...) load a
`pounce.solve-report/v1` JSON file and ask questions about convergence,
restoration, stalls, and per-iteration state.

## Status

Phase 0 spike. Post-mortem analysis of solve-report JSON only — no live
streaming, no binary `.iterdump` parser yet.

## Install

```bash
cd studio/mcp
pip install -e .
```

This exposes a `pounce-studio-mcp` console script that speaks MCP over stdio.

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

## Wire it into Claude Code

Add to `~/.claude/settings.json` (or the project's `.claude/settings.json`):

```json
{
  "mcpServers": {
    "pounce-studio": {
      "command": "pounce-studio-mcp"
    }
  }
}
```

Then in a session you can ask things like:

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
python -m pytest tests/ -v
```
