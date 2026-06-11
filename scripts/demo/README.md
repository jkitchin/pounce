# pounce debugger screencasts

Reproducible [asciinema](https://asciinema.org) screencasts of the interactive
solver debugger (`pounce --debug` — a pdb for the interior-point method). The
recordings are driven programmatically so they can be regenerated whenever the
debugger UI changes, instead of drifting out of date.

## What's here

| File | Purpose |
| --- | --- |
| `drive_debug.py` | pexpect driver — replays a scenario against `pounce --debug`, typing each command character-by-character so the recording looks hand-driven. Honors `$POUNCE_BIN`. |
| `_rec_pty.py` | Runs `asciinema` inside a wide pty so the iteration table isn't re-wrapped at the default 80 columns. |
| `record.sh` | Records every scenario with asciinema and (if `agg` is installed) renders a GIF. Outputs to `docs/demo/`. |
| `scenarios/*.dbg` | One scenario each: a short list of debugger commands plus `# problem:` / `# title:` directives. |

`record.sh` prefers this repo's freshly built `target/release/pounce` (falling
back to `target/debug/pounce`, then `pounce` on `PATH`) so the recorded banner
shows the workspace version — build first with `cargo build --release` (or
`make build`). Theme is `github-dark`; override with `POUNCE_DEMO_THEME` (run
`agg --help` for the list).

Generated assets live in [`docs/demo/`](../../docs/demo): `*.cast` (the
original asciicast — crisp, copy-pasteable, embeddable on asciinema.org) and
`*.gif` (drop-in for a README).

## Scenarios

1. **`01-rosenbrock`** — the happy path: single-step the IPM, print the iterate
   (`p x`), inspect the KKT summary (`i`), then `continue` to the optimum.
2. **`02-circle-mutation`** — it's a real debugger, not a trace viewer: break at
   iteration 3, read the reduced-Hessian inertia (`p kkt`), **overwrite a primal
   variable mid-solve** (`set x[0] 1.2`), `diff` the change, then watch the
   solver recover from the bad point.
3. **`03-infeasible-eq`** — the failure case: an event breakpoint on restoration
   entry (`break on resto_entered`) fires, inspect the state, then run to the
   "converged to a point of local infeasibility" exit.

## Regenerate

```sh
make screencast              # all scenarios -> docs/demo/*.{cast,gif}
scripts/demo/record.sh circle   # just the scenario(s) matching "circle"
```

Requirements: `asciinema`, `python3` + `pexpect`, and `pounce` on `PATH`. `agg`
is optional and only needed for the GIF conversion.

## Add a scenario

Drop a new `scenarios/NN-name.dbg` file:

```
# problem: rosenbrock          # which built-in to solve (required)
# title:   What this shows     # banner printed before the session
s                              # one debugger command per line
@1.6 set x[0] 1.2              # optional @<seconds> prefix overrides the pause
# a plain comment line is ignored
```

Built-in problems: `quadratic`, `rosenbrock`, `bounded-quadratic`,
`eq-quadratic`, `circle`, `infeasible-eq`. Type `help` at the `pounce-dbg>`
prompt for the full command set. Avoid `viz` in scenarios — it opens an
external viewer that won't render in a terminal recording; use the textual
`print` / `diff` / `i` commands instead.
