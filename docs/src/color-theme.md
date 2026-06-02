# Color Theme

POUNCE's terminal output uses one **tiger / rust / warm** palette across
every colored surface — the iteration table, the branded wordmark, and
the interactive debugger. This page is the single reference for *what the
colors mean*; the palette itself lives in
[`pounce-common::style`](https://github.com/jkitchin/pounce/blob/main/crates/pounce-common/src/style.rs)
(a pure, unit-tested module — no I/O, no globals).

For the environment variables that turn color on/off (`NO_COLOR`,
`CLICOLOR_FORCE`, `RUST_LOG`, `POUNCE_LOG_FORMAT`) see
[Solver Options → Logging and colored output](options.md#logging-and-colored-output).

## The palette

| Name | Hex | Role |
|---|---|---|
| `ALPHA_COOL` | `#000000` | iteration-row text at α = 1 (full Newton step) |
| `ALPHA_HOT` | `#cc2200` | iteration-row text at α → 0 (stalling); molten-claw base |
| `TAN` | `#8a6d3b` | restoration **soft-stay** row background (`s`) |
| `AMBER` | `#b56a12` | restoration **soft-exit** row background (`S`) |
| `RUST_DEEP` | `#6e260e` | restoration **hard** row background (`R` / resto-phase rows) |
| `CREAM` | `#f5e6c8` | restoration-row text at α = 1 |
| `BRIGHT_YEL` | `#ffe03a` | restoration-row text at α → 0; molten-claw top |
| `TIGER_ORANGE` | `#e87a1e` | `WARN` logs, banner accents, molten-claw mid |

Two further surfaces reuse these or a small extension:

| Name | Hex | Role |
|---|---|---|
| steel-hi → steel-lo | `#d2d6dc` → `#5c6068` | wordmark letter sheen, top row → bottom row |
| gold | `#ffb000` | debugger banner highlight (`interior-point debugger`, `help`) |
| dim | `#7a7e88` | debugger banner gloss text |

## Where the colors appear

### The iteration table

Two orthogonal channels encode solver state on each row:

- **Background = restoration kind**, keyed off the row's
  `alpha_primal_char` tag:
  - `s` soft-stay → **tan**, `S` soft-exit → **amber**, `R` hard (and the
    dedicated restoration phase's `r`-suffixed rows) → **deep rust**.
  - Normal (non-restoration) rows have no background.
  - Tiny-step tags (`t`/`T`) deliberately get *no* background — that
    stall is shown by the foreground instead.
- **Foreground = a smooth gradient on the primal step length α ∈ [0, 1]**
  (a visual stalling cue):
  - Normal rows: **black** (α = 1, full step) → **hot red** (α → 0).
  - Restoration rows: **cream** (α = 1) → **bright yellow** (α → 0), so
    the text stays legible on the dark background.
  - α is clamped to `[0, 1]`; a non-finite α is treated as a full step
    (no false stalling alarm).

So at a glance: a **dark row** means restoration (its shade tells you
which kind), and **redder / yellower text** means a shorter step (the
solver is struggling to move).

### The branded wordmark (`pounce` logo)

Printed atop a normal solve and at the top of the debugger REPL. The
`POUNCE` block letters carry a top-lit **steel sheen** (light silver
`#d2d6dc` at the top row fading to dark steel `#5c6068` at the bottom),
and three diagonal **molten claw slashes** rake across them, glowing
**bright yellow → tiger-orange → deep red** top-to-bottom — the project
logo's forged-metal-with-lava look.

### The interactive debugger

The REPL open banner (`--debug`) reuses the same wordmark, then a command
cheat-sheet whose shortcut keys are **tiger-orange**, the
`interior-point debugger` line and the `help` hint are **gold**, and the
descriptive gloss is **dim grey**. Pause banners and command output are
otherwise uncolored. (`--debug-json` emits no color — its stdout is a
pure JSON channel.)

`viz kkt` / `viz L` open in the external Plotly viewer
([`pounce-dbg-viz`](debugger.md#interactive-figures-pounce-dbg-viz)),
which is a *separate* visual language: the sparse-matrix heatmaps use a
diverging **red–blue** scale keyed on entry *value* (sign + magnitude),
not the terminal palette.

### Logs

`WARN`-level log lines (on stderr) take the **tiger-orange** accent;
other levels use the subscriber's defaults.

## Terminal support & downgrade

- **Truecolor** (24-bit) is used when the terminal advertises it via
  `COLORTERM` — every color above is emitted as exact RGB.
- **256-color** terminals get a graceful fallback: each RGB color snaps
  to the nearest xterm 6×6×6 cube color (`downgrade` /
  `nearest_ansi256`). The theme still reads correctly, just quantized.

## When color is emitted

Color is opt-out and TTY-aware:

- The **iteration table** is colored only when **stdout** is a terminal
  (via `anstream::AutoStream`, which strips escapes from redirected
  output while keeping identical column alignment).
- The **debugger banner** is colored only when **stderr** is a terminal.
- `NO_COLOR` (any value) disables color everywhere; `CLICOLOR_FORCE`
  forces it even into a non-terminal sink. See
  [Solver Options](options.md#logging-and-colored-output).

Because the policy is consistent, redirected logs/output are always plain
text — safe to diff, grep, and ingest.

## For contributors

Add or change colors in `pounce-common::style`, never with inline ANSI:
the constants, the α-gradient (`alpha_gradient_rgb`), the restoration
mapping (`resto_background_rgb`), the composed `iteration_row_style`, and
the truecolor `downgrade` all live there and are unit-tested without a
TTY. Print sites style through `anstyle` + `anstream` (or, for the
debugger banner on stderr, gate on `stderr().is_terminal()` and
`NO_COLOR`). Keep the two iteration-table channels — background =
*restoration kind*, foreground = *step length* — orthogonal.
