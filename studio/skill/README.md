# pounce-studio skill

Claude-skill alternative to the `pounce-studio-mcp` MCP server. Same
analysis surface, but the tool is invoked by Claude via the
`pounce-studio` CLI rather than via the Model Context Protocol — no
MCP server, no Python venv, no `claude mcp add` registration, no
restart.

## Install

The full one-shot recipe (build both binaries + install the skill):

```sh
make install-skill
```

This:
1. Builds `pounce-studio` and `pounce` in `target/release/`.
2. Copies both binaries into `~/.local/bin/` (override with `PREFIX=...`).
3. Drops this directory at `~/.claude/skills/pounce/` (override with
   `SKILL_DIR=...`).

After install, in a fresh Claude Code session, ask "diagnose
`/path/to/report.json`" — Claude will load this skill, invoke the CLI,
and answer.

Manual install — for users who don't want to use `make`:

```sh
cargo install --path crates/pounce-studio-core --bin pounce-studio --locked
cargo install --path crates/pounce-cli --bin pounce --locked
mkdir -p ~/.claude/skills
cp -r studio/skill ~/.claude/skills/pounce
```

Both installs build straight from crates.io — the committed `Cargo.toml`
pins [`feral`][feral] to a published release (`feral = "0.10.0"`), so no
sibling checkout or `[patch.crates-io]` override is required. (Solver-core
development against feral's unreleased HEAD uses a *local, uncommitted*
`[patch.crates-io]` git-rev override; that is a maintainer workflow and
does not affect installing the skill.)

If you'd rather not build `pounce` at all, skip its install and reuse a
`pounce` binary you have elsewhere on `PATH` — the skill only invokes
`pounce-studio` directly; `pounce` is needed only when you want to drive a
fresh solve from within the skill.

The skill works in every Claude Code session — desktop, web, IDE
extension, the CLI — because skills travel with the configuration
rather than running as a separate process.

[feral]: https://github.com/jkitchin/feral

## What the skill exposes

See [`SKILL.md`](SKILL.md). All `pounce-studio-mcp` tools have a
corresponding `pounce-studio` subcommand:

| MCP tool                   | CLI subcommand                              |
|----------------------------|---------------------------------------------|
| `load_solve_report`        | `pounce-studio summary <report>`            |
| `diagnose`                 | `pounce-studio diagnose <report>`           |
| `find_stalls`              | `pounce-studio find-stalls <report>`        |
| `convergence_trace`        | `pounce-studio convergence-trace <report>`  |
| `get_iterate`              | `pounce-studio get-iterate <report> <k>`    |
| `restoration_windows`      | `pounce-studio restoration-windows <report>`|
| `compare_runs`             | `pounce-studio compare <r1> <r2> ...`       |
| `linear_solver_summary`    | `pounce-studio linear-solver-summary <r>`   |
| `explain`                  | `pounce-studio explain <term>`              |
| `citations`                | `pounce-studio citations [--topic | --key]` |
| `analyze_problem` (nl)     | `pounce-studio analyze-nl <path \| --builtin>` |
| `analyze_gams_problem`     | `pounce-studio analyze-gms <path>`          |
| `parse_gams_listing`       | `pounce-studio parse-gams-listing <lst>`    |
| `list_gams_examples`       | `pounce-studio list-gams [--suite ...]`     |
| `run_problem`              | encoded as a skill recipe — chain `pounce` + `pounce-studio summary` |
| `run_gams_problem`         | encoded as a skill recipe — chain `gams ... NLP=POUNCE` + `pounce-studio parse-gams-listing` |

## When to use which backend

- **Skill (this)** — preferred default. Single CLI on PATH, no
  background server, works in every Claude session including
  remote/web. Setup is one `make` target.
- **MCP server** (`studio/mcp/`) — keep if you prefer structured
  tool-call args in the Claude UI, or if you want to drive it from a
  non-Claude MCP client (Cursor, Zed, Continue).

Both paths can coexist on the same machine. They share the Rust
analysis core (`crates/pounce-studio-core`) so findings agree.
