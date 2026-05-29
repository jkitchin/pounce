# Interactive Solver Debugger

POUNCE ships an interactive debugger for the interior-point loop — a
*pdb for the IPM*. You can pause the solve at well-defined points,
inspect and **mutate** the live mathematical state (the iterate,
multipliers, the barrier parameter μ), set breakpoints (by iteration, on
a numeric condition, or on a solver *event*), step through an iteration's
internal phases, rewind to an earlier iterate, re-solve from a saved
point with new options, and drop in automatically when a solve fails.

It has two front ends sharing one command engine:

- a **human REPL** (`--debug`) with history, Ctrl-R search, and Tab
  completion, and
- a **newline-delimited JSON protocol** (`--debug-json`) that an LLM
  agent, a script, or a visual debugger (e.g. a VS Code Debug Adapter)
  can drive programmatically.

No production NLP solver ships anything like this; if you have used
`ipopt` you have had `print_level` and a log. This is a live debugger.

> The debugger has **zero effect on the solve when it is not attached**.
> The checkpoint fire-sites short-circuit when no debugger is installed,
> so the standard regression suite is bit-for-bit identical with and
> without the feature compiled in.

---

## Quick start

```sh
pounce problem.nl --debug          # human REPL, pauses at iteration 0
pounce problem.nl --debug-json     # JSON protocol on stdin/stdout
pounce problem.nl --debug-on-error # run freely; drop in only if it fails
pounce problem.nl --debug-on-interrupt   # run; Ctrl-C drops you in
```

A 30-second session (human REPL):

```text
$ pounce --problem rosenbrock --debug

── pounce-dbg ── iter 0 @iter_start  mu=1.000e-1  obj=2.420000e1  inf_pr=0.00e0  inf_du=1.00e2
pounce-dbg> info
iter      = 0
mu        = 1.000000e-1
objective = 2.42000000e1
...
pounce-dbg> print x
x = [-1.200000e0, 1.000000e0]
pounce-dbg> break if inf_du<1e-6
conditional breakpoint: inf_du<1e-6
pounce-dbg> continue
... solver runs ...
── pounce-dbg ── iter 21 @iter_start  mu=...  inf_du=8.7e-7
   ↳ inf_du<1e-6
pounce-dbg> quit
```

The prompt is on **stderr**; the solver's own iteration table stays on
stdout, so a redirected log is unaffected.

---

## The two front ends

| | `--debug` (REPL) | `--debug-json` |
|---|---|---|
| Audience | human at a terminal | agent / script / GUI |
| Channel | prompt + output on stderr | pure JSON on stdout |
| Line editing | rustyline: history (`~/.pounce_dbg_history`), Ctrl-R, **Tab** completion | n/a (caller supplies UI) |
| Solver table | shown on stdout | suppressed (`print_level 0`) |
| Commands | bare strings | bare strings *or* `{"cmd":…,"args":[…],"id":…}` |

On a non-TTY stdin (a pipe), the REPL falls back to a plain line reader
(no history/Tab) but otherwise behaves identically — handy for scripted
tests.

The JSON protocol is documented in full [below](#the-json-protocol).

---

## Pausing and flow control

### Checkpoints

The loop fires the debugger at these points (a `pause` reports which one
via its `checkpoint` field):

| Checkpoint | Fires | What's fresh |
|---|---|---|
| `iter_start` | top of each outer iteration | the accepted iterate from the previous step |
| `after_mu` | μ updated for this iteration | the new barrier parameter |
| `after_search_dir` | Newton step `δ` solved | the step (`dx` …), regularization, **KKT inertia** |
| `after_step` | trial accepted | the step lengths α, the new iterate |
| `pre_restoration_entry` | just before restoration | the iterate that tripped restoration |
| `post_restoration_exit` | restoration returned | what restoration produced |
| `terminated` | once, before the solve returns | the final / failing iterate + status |

By default the debugger only *stops* at `iter_start` (and `terminated`).
The sub-iteration checkpoints fire every iteration but resume immediately
unless you ask to stop at them.

### Stepping

| Command | Effect |
|---|---|
| `step` / `s` / `n` | run to the next `iter_start` |
| `step sub` / `stepi` / `si` | run to the next checkpoint of *any* kind (walk an iteration's phases) |
| `continue` / `c` | run to the next breakpoint (or to completion) |
| `run N` / `r N` | run until iteration `N` |
| `stop-at <cp>` | always pause at checkpoint `<cp>` |
| `detach` | stop pausing; run to completion |
| `quit` / `q` | stop the solve now |

`stop-at` takes a checkpoint name or a friendly alias:

```text
stop-at after_search_dir     # or:  stop-at kkt
stop-at pre_restoration_entry   # or:  stop-at resto
stop-at                      # list active stop-at checkpoints
stop-at clear
```

Aliases: `mu` → `after_mu`, `kkt`/`search_dir` → `after_search_dir`,
`step` → `after_step`, `resto` → `pre_restoration_entry`, `resto_exit` →
`post_restoration_exit`.

---

## Breakpoints

Three kinds, all reported in `break` and surfaced as the pause `reason`.

### By iteration

```text
break 12            # pause at iteration 12   (alias: b 12)
break               # list all breakpoints
break del 12        # remove
break clear         # remove everything (iters + conditions + events)
```

### Conditional (with compound predicates)

```text
break if inf_pr<1e-6
break if mu<1e-4 && inf_pr>1e-3
break if iter>10 && (inf_du>1e-2 || obj<0)
break clear cond
```

- **Metrics:** `mu`, `inf_pr`, `inf_du`, `obj`, `err` (overall NLP
  error), `iter`.
- **Operators:** `<`, `<=`, `>`, `>=`, `==` (`==` is float-tolerant).
- **Compound:** `&&` and `||`, evaluated strictly **left-to-right with no
  precedence**; parentheses are accepted but stripped (they don't group).
  For real grouping, register several conditions — any one that holds
  fires.

Conditions are evaluated at `iter_start`.

### On a solver event

```text
break on regularized
break on resto_entered
break clear events
```

| Event | Fires when |
|---|---|
| `resto_entered` | the algorithm enters restoration |
| `resto_exited` | restoration returns |
| `regularized` | the KKT system needed regularization (δ_w > 0 — inertia correction) |
| `tiny_step` | the primal step is numerically negligible (‖dx‖∞ < 1e-10) |
| `ls_rejected` | the line search tried more than one trial point |
| `nan` | the NLP error or objective became non-finite |

Events fire at whatever checkpoint makes them observable (e.g.
`regularized` at `after_search_dir`), and pause with
`reason: "event: <name>"`.

---

## Inspecting state

```text
info                 # one-line summary: iter, mu, obj, inf_pr, inf_du, nlp_error, dims
print x              # a primal/dual block (alias: p x)
print dx             # a search-direction block (d + block name)
print mu             # a scalar: mu|obj|inf_pr|inf_du|err|iter
print kkt            # KKT inertia + regularization (see below)
```

**Blocks** (the eight components of the primal-dual iterate):

| Name | Meaning |
|---|---|
| `x` | primal variables |
| `s` | inequality slacks |
| `y_c` | equality-constraint multipliers |
| `y_d` | inequality-constraint multipliers |
| `z_l`, `z_u` | bound multipliers on `x` |
| `v_l`, `v_u` | bound multipliers on `s` |

Prefix any block with `d` (`dx`, `dz_l`, …) to print the corresponding
block of the most recent Newton step.

### `print kkt` — inertia and regularization

Available at/after `after_search_dir` (use `stop-at kkt`). This is the
view a solver expert reaches for when a step looks wrong:

```text
pounce-dbg> stop-at kkt
pounce-dbg> continue
── pounce-dbg ── iter 3 @after_search_dir ...
pounce-dbg> print kkt
dim       = 3
inertia   = n+=2 n-=1 (expected n-=1) → correct
delta_w   = 0.000000e0   (primal regularization)
delta_c   = 0.000000e0   (dual regularization)
status    = Success
```

The augmented (KKT) system has expected inertia `(n₊ = n, n₋ = m,
n₀ = 0)` where `m` is the number of equality + inequality multipliers.
A mismatch — or a nonzero `delta_w`/`delta_c` — is the classic signal
that the step is being stabilized (the solver added regularization to
fix the inertia). The full KKT-matrix and L-factor *heatmap* aren't
captured here; see [Limitations](#limitations).

---

## Mutating state

Mutations feed straight back into the solve.

```text
set mu 0.5           # overwrite the barrier parameter
set x[2] 1.5         # overwrite one component of a block
set x 1.0,2.0,3.0    # overwrite a whole block (comma-separated)
```

Setting any block works (`set z_l[0] 1e-3`, …). Iterate edits rebuild the
iterate with a fresh change-tag, so the cached derived quantities
(`curr_f`, slacks, σ, …) invalidate correctly and the next step is
computed from the new point — exactly as if the line search had produced
it.

Staging a solver option (validated against the registry):

```text
set opt mu_strategy adaptive
set opt linear_solver ma57
```

Staged options are **not** applied to the strategies already built for
the running solve (they don't re-read options mid-iteration). They take
effect on a [`resolve`](#re-solve-from-a-saved-point) or the next solve.

---

## Discovering options

```text
opt                  # list every registered option
opt mu               # filter by name/category substring
complete pri         # completion candidates for a prefix
```

`opt <exact-name>` also prints the long description. In the REPL, **Tab**
completes command verbs, block names, metric names (after `break if`),
checkpoint names (after `stop-at`), event names (after `break on`), and
option names (after `set opt` / `opt`).

---

## Time travel

### Rewind (`goto` / `restart`)

The debugger snapshots the primal-dual state (`x`, `s`, multipliers, μ,
τ) every iteration. `goto` rewinds to a captured iteration and stays
paused so you can re-tune before resuming:

```text
goto 3               # rewind to the start of iteration 3
restart              # rewind to the earliest snapshot
```

> **Caveat — this is a *soft* rewind.** Only the primal-dual state is
> restored; strategy *history* (the filter, the adaptive-μ oracle, the
> quasi-Newton memory) is **not** rolled back. So continuing from a
> rewound point is "resume from here," not a bit-exact replay of the
> original run.

### Re-solve from a saved point

`resolve` re-runs the solve from the **current** `x` with any
`set opt` edits applied — a primal warm start with new options. Use it
for "what if I change `mu_strategy` from here?":

```text
pounce-dbg> set opt mu_strategy adaptive
pounce-dbg> resolve
re-solving from current x with 1 staged option override(s)…
── pounce-dbg ── iter 0 @iter_start ...   # fresh solve, seeded from the captured x
```

Because each solve rebuilds its strategies from the options, the changes
*do* take effect on the re-solve. The seed is dropped (falling back to
the problem's own start) if presolve / fixed-variable elimination changed
the coordinate count.

---

## Saving and visualizing artifacts

```text
save                 # write the current iterate + residuals to a temp JSON
save /tmp/iter3.json # explicit path
viz x                # write a block and open it in an external viewer
viz dx               # a search-direction block
viz kkt              # the KKT inertia/regularization report
```

`save` writes every non-empty block, the search-direction blocks, and the
residual scalars (`iter`, `mu`, `objective`, `inf_pr`, `inf_du`,
`nlp_error`) — a self-contained artifact for external analysis.

`viz` opens the artifact with `$POUNCE_DBG_VIEWER` (a command template
where `{}` is replaced by the file path) if set, otherwise the platform
opener (`xdg-open` / `open`). For a plot, point it at a script:

```sh
export POUNCE_DBG_VIEWER='python my_plot.py {}'
```

---

## Ask Claude about the state

`ask [question]` packages the current paused state — checkpoint,
residuals, step lengths, dimensions, and the KKT inertia/regularization —
into a prompt and runs it through **Claude Code** (`claude -p`, headless
print mode), printing the reply inline. It's AI-assisted debugging
without leaving the loop:

```text
pounce-dbg> stop-at kkt
pounce-dbg> continue
pounce-dbg> ask why is the dual infeasibility stalling?
# → Claude's analysis of the state + suggested options to try
```

With no question it defaults to "explain the current state and suggest
what to try next." The command is configurable via `$POUNCE_DBG_LLM`
(default `claude -p`); the prompt is fed on the tool's stdin, or
substituted into a `{}` placeholder if the template has one:

```sh
export POUNCE_DBG_LLM='claude -p'          # default
export POUNCE_DBG_LLM='llm -m claude-opus' # any prompt-on-stdin CLI
export POUNCE_DBG_LLM='mytool --ask {}'    # prompt as an argument
```

In JSON mode the reply comes back in the `result` event's `data.reply`.

---

## Attaching to a run

You don't have to single-step from iteration 0.

- **Drop in on failure** — `--debug-on-error` runs the solve freely and
  pauses at the `terminated` checkpoint **only if the solve did not
  succeed**, leaving you at the failing iterate for a post-mortem. (Plain
  `--debug` also pauses at `terminated` for a final-point inspect.)
- **Attach with Ctrl-C** — `--debug-on-interrupt` runs normally but
  installs a SIGINT handler; a first Ctrl-C drops you in at the next
  iteration (`reason: "interrupt (Ctrl-C)"`), a second Ctrl-C aborts.
  Ctrl-C also breaks into any other debug mode mid-`continue`. (At a
  rustyline prompt, Ctrl-C is an ordinary line-cancel, not a signal.)

---

## Exit model

| Path | Result |
|---|---|
| `quit` | stops now → `UserRequestedStop` |
| `continue` / `detach` | run to natural completion |
| stdin EOF, REPL (Ctrl-D) | detach and finish (pdb convention) |
| stdin EOF, JSON (pipe closed) | abort — the controlling client is gone |
| external SIGKILL | process dies (no `terminated` event) |

Every non-kill path ends with a `terminated` event in JSON mode.

---

## Command reference

| Command (aliases) | Summary |
|---|---|
| `help` (`h`, `?`) | list commands |
| `info` (`i`) | current-iterate summary |
| `print <what>` (`p`) | block, `d`-block, scalar, or `kkt` |
| `step` (`s`, `n`) | run to next `iter_start` |
| `step sub` / `stepi` (`si`) | run to next checkpoint of any kind |
| `continue` (`c`) | run to next breakpoint |
| `run N` (`r`) | run until iteration N |
| `break …` (`b`) | iteration / `if` / `on` breakpoints; list; `clear`; `del N` |
| `stop-at <cp>` | always pause at a checkpoint |
| `set mu/x/<block>/opt …` | mutate μ, the iterate, or stage an option |
| `opt [filter]` | list/search registered options |
| `complete <prefix>` | completion candidates |
| `viz <target>` | open an artifact in a viewer |
| `save [path]` | dump the iterate to JSON |
| `goto N` / `restart` | soft-rewind to a captured iteration |
| `resolve` | re-solve from current x with staged options |
| `ask [question]` | ask Claude Code (`claude -p` / `$POUNCE_DBG_LLM`) about the state |
| `progress [on/off]` | toggle JSON progress events |
| `detach` | stop pausing; run to completion |
| `quit` (`q`, `exit`) | stop the solve |

---

## The JSON protocol

`--debug-json` makes **stdout a pure stream of newline-delimited JSON
objects** (the banner, problem stats, and final summary are routed to
stderr, and `print_level` is forced to 0). A program reads one JSON
object per line.

### Session lifecycle

1. `hello` — emitted once, up front. The handshake.
2. `pause` — at each stop.
3. `result` — one per command, echoing the client's `request_id`.
4. `progress` — one per iteration while running between pauses.
5. `terminated` — once, after the solve.

### Commands

Write one per line to stdin, either a bare string or an object:

```json
{"cmd": "print", "args": ["x"], "id": 7}
{"cmd": "break if inf_pr<1e-6", "id": 8}
"continue"
```

`id` (any JSON value) is echoed back as `request_id` on the matching
`result`, for async correlation.

### `hello`

```json
{"event":"hello","protocol":"pounce-dbg/1","pounce_version":"0.2.0",
 "capabilities":{"inspect":true,"mutate_iterate":true,"mutate_mu":true,
   "conditional_breakpoints":"compound","request_ids":true,
   "viz":["block","delta"],"save":true,"kkt_inspect":true,
   "rewind":"primal_dual","resolve":true,"terminal_checkpoint":true,
   "interruptible":true,"progress_events":true,"async_pause":"checkpoint"},
 "checkpoints":["iter_start","after_mu","after_search_dir","after_step",
                "pre_restoration_entry","post_restoration_exit","terminated"],
 "events":["resto_entered","resto_exited","regularized","tiny_step",
           "ls_rejected","nan"],
 "commands":[…],"blocks":[…],"metrics":[…]}
```

A client should **feature-detect off `capabilities` / `checkpoints` /
`events`** rather than the protocol string — those lists are additive as
the debugger grows.

### `pause`

```json
{"event":"pause","checkpoint":"iter_start","status":null,
 "iter":3,"mu":2.0e-2,"objective":5.05,"inf_pr":0.0,"inf_du":2.7e-14,
 "nlp_error":0.0237,"dims":{"x":2,"s":0,"y_c":0,"y_d":0,"z_l":2,"z_u":2,
 "v_l":0,"v_u":0},"breakpoints":[],"conditions":[],"reason":"mu<0.05"}
```

`status` is non-null only at the `terminated` checkpoint. `reason`
carries the firing breakpoint / condition / event / interrupt.

### `result`

```json
{"event":"result","request_id":7,"command":"print x","ok":true,
 "output":["x = [-1.18e0, 1.38e0]"],"data":{"name":"x","values":[-1.18,1.38]}}
```

`output` is human-readable lines; `data` is the structured payload
(present for inspection commands).

### `progress`

```json
{"event":"progress","iter":42,"mu":1.0e-5,"inf_pr":3.2e-7,"inf_du":1.1e-6,"obj":12.34}
```

Emitted once per outer iteration during a `continue`, so a UI can show
live progress instead of a hang. Default on; toggle with the `progress`
command.

### `terminated`

```json
{"event":"terminated","status":"SolveSucceeded",
 "status_message":"Optimal Solution Found.","iterations":6,
 "objective":4.9999999,"evals":{"obj":7,"obj_grad":7,"constr":1,
 "constr_jac":12,"hess":6}}
```

### Async pause

A running `continue` is interrupted by sending **SIGINT** to the process
(`process.kill(pid, "SIGINT")`); the debugger pauses at the next
checkpoint with `reason: "interrupt (Ctrl-C)"`. This is what a Debug
Adapter's pause button maps to. (`hello.capabilities.async_pause` is
`"checkpoint"`.)

---

## Tutorials

### 1. Why did this problem go to restoration?

```text
$ pounce hard.nl --debug-json
{"cmd":"break on resto_entered"}
{"cmd":"continue"}
# → pause at checkpoint "pre_restoration_entry", reason "event: resto_entered"
{"cmd":"info"}                # how infeasible is the iterate?
{"cmd":"print kkt"}           # was the KKT singular / heavily regularized?
{"cmd":"print x"}
```

### 2. Catch a step that gets regularized

```text
break on regularized
continue
# → pause at after_search_dir when delta_w > 0
print kkt        # inertia n- vs expected; delta_w / delta_c
print dx         # the (stabilized) Newton step
```

### 3. What-if: try a different μ strategy from here

```text
break 5
continue                 # stop at iteration 5
set opt mu_strategy adaptive
resolve                  # re-solve from the iter-5 point with adaptive μ
```

### 4. Post-mortem on a failure

```sh
pounce maybe-infeasible.nl --debug-on-error
```

Runs unattended; if the solve returns anything but success you land at
the final iterate:

```text
── pounce-dbg ── TERMINATED (LocalInfeasibility)  iter 11  obj=1.13e0  inf_pr=5.0e-1  inf_du=1.2e-8
pounce-dbg> print x
pounce-dbg> print kkt
```

### 5. Drive it from a program / agent

```python
import subprocess, json
p = subprocess.Popen(["pounce", "hs071.nl", "--debug-json"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def send(cmd, **kw): p.stdin.write(json.dumps({"cmd": cmd, **kw}) + "\n"); p.stdin.flush()
def recv():          return json.loads(p.stdout.readline())

hello = recv()                       # capabilities / vocabulary
print(recv())                        # initial pause
send("break if inf_du<1e-6", id=1)
print(recv())                        # result, request_id=1
send("continue")
for line in p.stdout:                # progress … pause … terminated
    ev = json.loads(line)
    if ev["event"] == "terminated": break
```

---

## Limitations

- **No live KKT-matrix / L-factor heatmap.** `print kkt` / `viz kkt`
  give the inertia and regularization *scalars*; the full sparse matrix
  and its `LDLᵀ` factor pattern are not captured at the checkpoint. Use
  the offline dump for those: `--dump kkt:<iter>+L+Lvals --dump-dir DIR`
  (see [Running Solves](cli.md)).
- **Soft rewind only.** `goto`/`restart` restore the primal-dual state,
  not strategy history (see the caveat above).
- **Async pause is SIGINT-based.** An in-band `{"cmd":"pause"}` (for
  platforms without convenient signals) is not yet wired.
- **`set opt` is staged, not hot-applied** to a running solve; it takes
  effect on `resolve` / the next solve.
