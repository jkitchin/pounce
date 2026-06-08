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

The same debugger spans **every** POUNCE solver — the NLP filter-IPM and the
convex / conic interior-point solver share one command engine and one REPL.
See [Beyond the interior-point loop](#beyond-the-interior-point-loop) for the
small set of commands whose availability is backend-conditional.

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
| `step_rejected` | line search gave up (tiny step / all backtracks failed), before restoration | the search direction `δ` and the un-accepted iterate |
| `pre_restoration_entry` | just before restoration | the iterate that tripped restoration |
| `post_restoration_exit` | restoration returned | what restoration produced |
| `terminated` | once, before the solve returns | the final / failing iterate + status |

By default the debugger only *stops* at `iter_start` (and `terminated`).
The sub-iteration checkpoints fire every iteration but resume immediately
unless you ask to stop at them.

**Stepping into restoration.** The same debugger drives the restoration
*inner* IPM: when the solve enters restoration, the inner solve's
checkpoints fire too. A `step`/`stepi` that lands on an inner iteration
pauses there with `in_restoration: true` (REPL banner shows
`[restoration]`), and `print x` shows the restoration sub-NLP iterate.
`stop-at resto` (`pre_restoration_entry`) is the easy way to catch the
hand-off and then step inward.

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
tbreak 12           # one-shot: pause at 12, then delete itself (alias: tb)
break               # list all breakpoints
break del 12        # remove
break clear         # remove everything (iters + conditions + events)
```

### Watchpoints (data breakpoints)

```text
watchpoint x[3]        # pause when x[3] changes (alias: wp)
watchpoint x 1e-3      # pause when any x component moves by > 1e-3
watchpoint             # list; watchpoint del x[3]; watchpoint clear
```

Distinct from `watch` (which only *displays*): a watchpoint *pauses* the
solve when the watched value changes by more than its threshold (default
0 = any change) between iterations. Useful for a component expected to
stay put (e.g. a variable pinned at a bound).

### Breakpoint command lists

Attach commands to a breakpoint that run automatically when it hits —
semicolon-separated, ending with a flow command to auto-resume:

```text
break 5
commands 5 print kkt ; set mu 0.1 ; continue   # at iter 5: inspect, tweak μ, go
commands 5 clear                               # remove
commands                                       # list all
```

When iteration 5 is reached, the debugger emits the `pause`, runs the
attached commands (each `result` is reported), and if one of them
resumes/stops, honors it without dropping to the prompt — otherwise it
falls through to the interactive prompt as usual.

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
| `mu_stalled` | μ held (to tolerance) for 3 consecutive iterations |
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
print mu             # a scalar: mu|obj|inf_pr|inf_du|err|compl|iter
print kkt            # KKT inertia + regularization (see below)
print rank           # SVD numerical rank of the equality Jacobian J_c (see below)
print active         # which bound categories are near-active (small slack)
watch mu             # auto-print a target at every pause (alias: display)
watch                # list watches; watch del mu; watch clear
```

`watch <target>` registers any `print` target (block, `dx`, scalar,
`kkt`) to be shown automatically at every subsequent pause — the
debugger's equivalent of gdb's `display`. In JSON mode the values arrive
in the pause event's `watches` array.

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

### Model names (`.col` / `.row`)

A solver-internal diagnostic that says *"variable 132 in equation 3 looks
singular"* is far less actionable than one that says *"`T_reactor` in
`energy_balance`"*. Lee et al. (2024) identify this gap — between
detecting an issue numerically and tracing it back to a *named* equation
in the modeling environment — as a central roadblock for debugging
equation-oriented models.[^lee2024]

AMPL `.nl` files carry no names, but AMPL emits two optional sibling
files when the modeler sets `option auxfiles rc;`:

| File | Contents |
|---|---|
| `stub.col` | one **variable** name per line, in column order |
| `stub.row` | one **constraint** name per line, in row order |

When these sit next to the `.nl`, pounce captures them
(`NlProblem::var_names` / `con_names`) and exposes them through the
`ExpressionProvider::variable_name` / `constraint_name` seam. Missing or
malformed name files are non-fatal — names are a diagnostic aid, never
load-blocking, so the debugger simply falls back to index labels.

`print residuals` uses these names directly. Residual values live in the
solver's *split* space (equalities and inequalities separated, fixed
variables removed), so a name only labels the right row if it is carried
through the same permutations. The TNLP publishes its `.col`/`.row` names
under the conventional `idx_names` metadata key, and `OrigIpoptNlp`
projects them into split space (`x_not_fixed_map` for variables, `c_map`
for equalities, `d_map` for inequalities) — the debugger reads the result
via `DebugCtx::split_names`. So a near-singular equality residual prints as

```
c[energy_balance] = +3.142e-04   |3.142e-04|
```

instead of `c[3]`. The same `idx_names` pool labels `grad_x_L[...]`
(variable names) and `grad_s_L[...]` / `d-s[...]` (inequality names). The
JSON payload keeps the numeric `index` and adds a `name` field.

> **Status.** Capture, exposure, and `print residuals` labeling are live
> on the AMPL `.nl` path with names projected through the bound /
> c-d-split permutations. **Presolve** renumbers rows, so `PresolveTnlp`
> declines `idx_names` rather than risk mislabeling a permuted row — under
> presolve the debugger safely falls back to index labels. Carrying names
> through the presolve map and decorating `print active` are the next
> steps built on this foundation.

[^lee2024]: A. Lee, R. B. Parker, S. Poon, D. Gunter, A. W. Dowling, and
    B. Nicholson, "Model Diagnostics for Equation-Oriented Models:
    Roadblocks and the Path Forward," *Systems and Control Transactions*
    3:966–974 (2024). <https://doi.org/10.69997/sct.147875>

### `print equation` — the algebra of a named constraint

Naming a culprit row is only half the story; the next question is always
*what does that equation actually say?* Lee et al. (2024) make this the
core of actionable equation-oriented diagnostics — a debugger should
surface the **named equation**, not just a row index.[^lee2024] `print
equation` closes that loop: once `print residuals` points at, say,
`c[energy_balance]`, you read the constraint's source algebra directly.

```text
(dbg) print equation energy_balance
energy_balance:  T_reactor*flow - 300*flow - Q = 0

(dbg) print equation 14          # by original .nl row index
c[14]:  x[3]^2 + x[7]^2 <= 1
```

A constraint is addressable by its **model name** (preferred, and robust
to row reordering) or its **original `.nl` row index**. With no argument,
`print equation` reports how many equations are available. The renderer
works from the faithful `Expr` DAG the `.nl` parser built — not the lossy
evaluation tape — so common-subexpressions, imported functions, and
piecewise/conditional forms render as written. The affine part is printed
with tidy signs (`a - 2*b`, not `a + -2*b`), zero-coefficient Jacobian
placeholders are suppressed, and bounds render in their natural relation
(`= rhs`, `lo <= body <= hi`, `>= lo`, `<= hi`). The JSON payload carries
`{index, name, equation}`.

Equations are *static model data* in original `.nl` row order, so unlike
residuals they need no split-space projection — `print equation` works
regardless of presolve. It is available whenever a model was loaded from
an `.nl` file; the JSON `name` field is present only when a `.row` auxfile
supplied one.

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
fix the inertia).

For the matrix and factor themselves:

```text
viz kkt     # the assembled augmented-system matrix (triplets) + inertia
viz L       # the LDLᵀ factor (strict-lower triplets + values)
```

`viz kkt` writes the KKT matrix as 1-based lower-triangle triplets
(`dim`, `irn`, `jcn`, `vals`) alongside the inertia summary — point
`$POUNCE_DBG_VIEWER` at a heatmap script. `viz L` writes the `LDLᵀ`
factor (`n`, fill-reducing `perm`, strict-lower `l_irn`/`l_jcn`/`l_vals`
in permuted coordinates), read out of the factor the solver actually
computed.

Both are read-only and always show the **most recent factorization**:
the current iteration's system at an `after_search_dir` stop, or the
**previous** iteration's at the default `iter_start` pause (the step
that produced where you're standing). The matrix and factor are captured
every iteration while the debugger is *stepping*; once you `detach` (run
free) the capture is dropped — so on a large problem a free run doesn't
pay the O(nnz) assembly. If you `viz kkt`/`viz L` right after a free run,
`step` once to re-capture.

### `print rank` — numerical rank of the equality Jacobian

`print kkt` tells you *that* the dual system needed regularization
(`delta_c > 0`) or that the inertia was wrong; the `structural_singularity`
finding names equations that are dependent *by sparsity pattern*. `print
rank` closes the last gap: a **rank-revealing SVD of the equality Jacobian
`J_c` at the current iterate**. It factors the matrix the solver actually
sees (constraint scaling already applied), so it localizes the dependency
to specific equations — *including dependencies that are numerical only*
(values that cancel over a full sparsity pattern), which the structural
Dulmage–Mendelsohn pass cannot detect.

It doesn't just *name* the culprit equations — it **prints them**. When a
`.nl` model is loaded, each implicated row's source algebra is rendered
directly beneath it (the same DAG-faithful text `print equation` shows), so
you read the dependency without a second command:

```text
pounce-dbg> print rank
equality Jacobian J_c: 3 row(s) × 4 column(s)
numerical rank = 2 / 3  (deficiency 1)
σ_max = 3.162e0   σ_min = 0.000e0   cond = inf (σ_min = 0)   (rank tol τ = 1.40e-15)
singular values: [3.162e0, 1.414e0, 0.000e0]
rank-deficient: 1 equation(s) lie in the near-null space (linearly dependent / redundant) — the source of δ_c regularization:
  c[mass_balance]       (participation 0.50)
      x[0] + x[1] - 10 = 0
  c[mass_balance_dup]   (participation 0.50)
      x[0] + x[1] - 10 = 0
```

The two equations print identically — that *is* the redundancy, now visible
on its face.

For the SVD `J_c = U Σ Vᵀ`, the left singular vectors `u_k` whose singular
value `σ_k ≈ 0` span the left null space — the row combinations `u_kᵀ J_c ≈
0` that vanish. Each row's **participation** `w_i = Σ_{k : σ_k ≤ τ} u[i,k]²
∈ [0, 1]` localizes the dependency: a redundancy shared between two
equations splits ≈ 0.5/0.5, while `w_i = 1` means row `i` lies entirely in
the null space. The numerical-rank threshold is the standard LAPACK/NumPy
`τ = σ_max · max(m, n) · ε`; the implicated rows are resolved to model
names through the same `.row` plumbing as `print residuals` / `print
equation`.

The inline algebra is resolved **by model name**, so it appears for named
rows. The rank report's row index is the *split* equality position, not the
original `.nl` row the equation source keys on, so an unnamed row can't be
mapped — there `print rank` falls back to a `print equation <name>` hint
instead of guessing. When `J_c` has **full row rank**, that is reported as a
positive signal (`J_c has full row rank at this iterate.`) with the
σ_min/cond witnessing how far it is from degenerate — silence would be
ambiguous. The command is available whenever the iterate has an equality
block; a problem with no equality constraints returns a short explanatory
error. The JSON payload is `{iter, n_rows, n_cols, rank, deficiency,
rank_deficient, sigma_max, sigma_min, cond, tol, singular_values, culprits:
[{row, kind, index, name, label, weight, equation}]}` (`equation` is the
rendered source or `null` when unresolved; `cond` is `null` when `σ_min =
0`, since JSON has no infinity).

### `diagnose` — a live, named health report

`info`, `print residuals`, and `print kkt` each expose *one* facet of the
current iterate. `diagnose` (alias `diag`) runs a panel of heuristics over
all of them at once and returns a ranked list of findings — and, crucially,
**names the culprit equation or variable** behind each numerical symptom.
That last step is the actionable-diagnostics path of Lee et al.
(2024):[^lee2024] a report that says *"`mass_balance` is the worst
constraint residual"* is worth far more than *"row 13 is infeasible."*

```text
pounce-dbg> diagnose
[  error] primal_infeasible: Primal infeasibility 1.70e+02; worst constraint
         residual is c[mass_balance] = +1.701e+02. Inspect this equation's
         feasibility and scaling (`print equation mass_balance`).
[warning] dual_infeasible: Dual infeasibility 9.84e-01; largest stationarity
         residual is grad_x_L[T_reactor] = -9.838e-01.
[warning] inertia_wrong: KKT inertia is wrong (n-=2 vs expected 1): the system
         was indefinite/singular and the step had to be stabilized.
[   info] bounds_pinned: 3 variable bound(s) are active (slack < 1e-6).
```

This is the **live** counterpart to the `pounce-studio` `diagnose` tool,
which runs *temporal* heuristics over a finished solve report. The two
share a `{severity, code, message}` shape so
a client can treat them uniformly, but the live command sees what a saved
report cannot: the **current KKT inertia and regularization**, and the
**named** primal/dual residuals at this exact point. Findings are sorted
`error` → `warning` → `info`; a clean iterate yields a single `healthy`
finding. The checks:

| code | severity | fires when |
|---|---|---|
| `primal_infeasible` | error/warning | `inf_pr` above tol → names the worst constraint residual |
| `dual_infeasible` | warning | `inf_du` above tol → names the worst stationarity residual |
| `inertia_wrong` | warning | KKT inertia ≠ expected (rank-deficient Jacobian / indefinite Hessian) |
| `heavy_regularization` | info | primal δ_w applied (Hessian indefinite) |
| `dual_regularization` | warning | dual δ_c applied (linearly dependent / redundant equalities) |
| `structural_singularity` | warning | a subset of equalities is over-determined → **names the dependent equations** |
| `rank_deficient_jacobian` | warning | SVD of `J_c` is numerically rank-deficient → **names the equations in the near-null space** (catches value-only dependencies too) |
| `large_multipliers` | warning | a multiplier exceeds 1e8 (constraint-qualification / scaling) |
| `bounds_pinned` | info | variables pressed against their bounds |
| `tiny_step` | warning | accepted α_pr collapsed |
| `heavy_line_search` | warning | ≥10 backtracking trials for the accepted step |
| `in_restoration` | warning | currently inside feasibility restoration |
| `mu_stalled` | warning | μ flat for ≥3 consecutive iterations |

KKT-derived findings (`inertia_wrong`, `*_regularization`) need a computed
search direction, so they appear at/after `after_search_dir`. Names follow
the same rule as `print residuals`: present on the `.nl` path with
`.col`/`.row` files, index labels (`c[13]`) under presolve. The JSON payload
is `{iter, findings: [{severity, code, message}], n_findings}`.

#### Structural rank: naming the dependent equations

`inertia_wrong` and `dual_regularization` *detect* a rank-deficient
Jacobian, but only as a scalar — they tell you a redundancy exists, not
*which* equations are redundant. `structural_singularity` closes that gap
with a **Dulmage–Mendelsohn decomposition** of the equality Jacobian's
sparsity pattern (the same structural check at the heart of IDAES's
`DiagnosticsToolbox`). A maximum bipartite matching between equality rows
and variables partitions the system; any **over-determined block** — more
equations than the variables they jointly touch — forces at least one of
those equations to be redundant or mutually inconsistent (LICQ fails). The
finding lists those equations *by model name*, e.g.:

```text
pounce-dbg> diagnose
[warning] structural_singularity: Constraint Jacobian is structurally singular
         (Dulmage–Mendelsohn): 2 equation(s) over-determine the 1 variable(s)
         they jointly touch (flow_rate), so ≥1 of them must be redundant or
         mutually inconsistent (LICQ fails on this block). Candidate dependent
         equations: mass_balance, mass_balance_dup. Inspect them with
         `print equation <name>`; this names the rows behind any δ_c
         dual-regularization / wrong-inertia signal.
```

This is the **named-culprit** payoff of Lee et al. (2024):[^lee2024]
reporting *"`mass_balance` and `mass_balance_dup` are linearly dependent"*
rather than *"the Jacobian is singular."* The check is iterate-independent
(it reads only the sparsity pattern), so unlike the KKT-derived findings it
fires from iteration 0 — it can flag a structurally broken model before the
solver ever stalls on it. It is suppressed for well-posed problems: an NLP
with more variables than equality constraints is the normal case (the spare
degrees of freedom are pinned by the objective, bounds, and inequalities),
so only the over-determined side is reported, never the under-determined
one. Available on the `.nl` path; names fall back to `c[i]`/`x[i]` when no
`.col`/`.row` auxiliary files were emitted.

#### Numerical rank: the value-dependency the structure can't see

`structural_singularity` reads only the *sparsity pattern*, so it is blind
to a redundancy that lives in the **values** — three equations whose every
entry is nonzero (a structurally full-rank pattern) but whose rows satisfy
`row₂ = row₀ + row₁` numerically. `rank_deficient_jacobian` is the
numerical complement: it runs the same SVD as [`print
rank`](#print-rank--numerical-rank-of-the-equality-jacobian) over `J_c` at
the current iterate and, when the numerical rank falls short, names the
equations in the near-null space:

```text
pounce-dbg> diagnose
[warning] rank_deficient_jacobian: Equality Jacobian J_c is numerically
         rank-deficient at this iterate: rank 2/3 (deficiency 1),
         σ_min=0.00e0, cond=inf (σ_min = 0). Linearly dependent or redundant
         equality constraints — the root cause behind δ_c regularization /
         wrong inertia. Implicated equations: c[mass_balance],
         c[mass_balance_dup].
```

Unlike the structural check, this one is **iterate-dependent** — it factors
`J_c` at the current `x`, so it reflects the matrix the solver is actually
regularizing and catches dependencies that only appear at certain points.
The two checks are deliberately layered: `structural_singularity` fires
from iteration 0 on the pattern alone; `rank_deficient_jacobian` confirms
it numerically and, more importantly, surfaces the value-only dependencies
the structural pass provably cannot. See [`print
rank`](#print-rank--numerical-rank-of-the-equality-jacobian) for the SVD
math and the per-equation participation weights.

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

The read-side counterpart is `get opt <name>`, which reports an option's
current (or staged) value and its registry metadata — so you can confirm
what a `set opt` actually staged before you `resolve`:

```text
get opt mu_strategy        # → mu_strategy = adaptive  (staged)
```

---

## Discovering options

```text
opt                  # list every registered option
opt mu               # filter by name/category substring
complete pri         # completion candidates for a prefix
```

`opt <exact-name>` also prints the long description. In the REPL, **Tab**
completes command verbs, block names, metric names (after `break if`),
checkpoint names (after `stop-at`), event names (after `break on`),
option names (after `set opt` / `opt`), and **filesystem paths** (after
`load` / `sweep` / `save` / `source` — directories get a trailing `/`).
The same contexts are available programmatically via the `complete
<prefix…>` command (JSON `complete`), so an agent or GUI can offer the
same completions.

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

### `load` — the inverse of `save`

Typing a start point by hand is fine for a 2-variable toy and miserable for
anything real. `load` reads a block straight into the live iterate, so you
generate the point once (a prior solve, a surrogate, a sampler) and pull it
in:

```text
load /tmp/it0.json       # a `save` artifact: every block it contains is loaded
load start.csv           # a plain numeric file → x (comma/space/newline sep)
load start.csv s         # … into a named block instead of x
```

Two input shapes are accepted:

- **A `save` artifact** (JSON). Blocks are read from the top level or from
  an `iterate` object; every block present (`x`, `s`, multipliers, …) is
  written, each validated against the current dimension. So
  `save`→`load` round-trips a full point, and you can lift just the part
  that fits if dimensions changed.
- **A plain numeric file** — values separated by commas, whitespace, or
  newlines — written into the named block (default `x`). This is the
  many-variable escape hatch: `numpy.savetxt("start.csv", x0)` then
  `load start.csv`.

A loaded `x` becomes the seed for the next step (or for `resolve` — a
warm start from an externally-computed point with no typing).

### Interactive figures (`pounce-dbg-viz`)

`viz` writes a JSON artifact and hands it to a viewer. The Python package
ships an interactive **Plotly** viewer that renders these properly —
a spy/heatmap for `viz kkt` (the augmented matrix, colored by value, with
the inertia/regularization in the title) and `viz L` (the LDLᵀ factor),
and a bar chart for vector blocks (`viz x`, `viz dx`):

```sh
pip install 'pounce-solver[viz]'    # installs the `pounce-dbg-viz` script
```

When `pounce-dbg-viz` is on `PATH`, `viz` uses it automatically (opening
an interactive figure in your browser). The launch order is:

1. `$POUNCE_DBG_VIEWER` — a command template (`{}` ← the artifact path),
   if set;
2. `pounce-dbg-viz` — the bundled Plotly viewer, if installed;
3. the OS opener (`xdg-open` / `open`) on the raw JSON.

So `export POUNCE_DBG_VIEWER='python my_plot.py {}'` overrides with your
own plotter, and with nothing set + the `viz` extra installed it just
works. The same `pounce-dbg-viz <file.json>` also renders a `save`
artifact (the full iterate).

---

## Multi-start and initialization sensitivity

Interior-point methods find a *local* solution, and which one depends on
where you start. Two commands turn the debugger into an
initialization-sensitivity probe: they run many full solves — each from a
different start — and tabulate where each one ends up. Both build on the
same re-solve machinery as [`resolve`](#re-solve-from-a-saved-point) (so
they need the restart cell the CLI wires by default; they error in
contexts without it), and both leave you at a normal prompt on the final
solve afterward.

### `sweep <file>` — explicit starts

Run one solve per start point listed in a file (one start per line,
comma/whitespace-separated; `#`/`//` comments and blank lines skipped):

```text
pounce-dbg> sweep starts.txt
   sweep 1/4: Success                iters=21   obj=3.743990e-21 inf_pr=0.00e0
   sweep 2/4: Success                iters=15   obj=1.233088e-28 inf_pr=0.00e0
   sweep 3/4: Success                iters=14   obj=1.328861e-28 inf_pr=0.00e0
   sweep 4/4: Success                iters=29   obj=2.982346e-18 inf_pr=0.00e0

── sweep complete ── 4 solves, 4 succeeded, 1 distinct minima
     #  status                 iters       objective     inf_pr
     0  Success                   21    3.743990e-21     0.00e0
     1  Success                   15    1.233088e-28     0.00e0
     2  Success                   14    1.328861e-28     0.00e0
     3  Success                   29    2.982346e-18     0.00e0
   best: solve #2  obj=1.32886077e-28
```

Each start must have the same length as `x` (mismatches are reported with
the line number). The summary clusters successful objectives to a relative
`1e-6` to count **distinct minima** and flags the best (lowest-objective)
solve. This is the "is this solve fragile to its start, and to which basins
does it fall?" diagnostic — and unlike a black-box global search it leaves
every solve's *trajectory* observable: set a `break on resto_entered` or a
`stop-at kkt` first and the sweep will pause inside whichever solve trips
it.

### `multistart <N> [rel]` — sampled restarts

When you don't have a file of starts, `multistart` generates `N` of them:

```text
pounce-dbg> multistart 8          # 8 starts
pounce-dbg> multistart 8 0.3      # wider jitter on any unbounded vars
```

Each variable that has a **finite box** `[x_Lᵢ, x_Uᵢ]` is sampled
**uniformly inside it** — a genuine box multistart. Variables that are
unbounded on either side fall back to a relative jitter `±rel·(|xᵢ|+1)`
around the current point (`rel` default 0.1, with a floor so components at
zero still move). The command reports the split, e.g.
`multistart 8 (box 5/7 vars; 2 unbounded → jitter rel=0.1)`.

Start 0 is always the *unperturbed* current `x` (so the run includes where
you already are), and the sampler is a fixed-seed PRNG, so a `multistart`
run **reproduces exactly**.

The bounds are the ones the *algorithm* sees — full-length, post-scaling,
after any `bound_relax_factor` — so every sampled start is a valid seed.
For a problem with no finite bounds (a pure unconstrained NLP) `multistart`
degrades to jitter around `x`; `sweep` an external sample if you want a
specific spread there.

### Driving a sweep from a file with `load`

The pieces compose. To seed a sweep from points computed elsewhere, write
them with `numpy.savetxt` and `sweep` the file directly — or, for a single
externally-computed warm start, `load` it and `resolve`:

```python
import numpy as np
np.savetxt("starts.txt", sampler(n=32), delimiter=",")   # 32 starts, one per row
```

```text
pounce-dbg> sweep starts.txt
```

### sweep vs. `find_minima`

`sweep`/`multistart` are *diagnostics*: they show you how a handful of
starts behave, with full visibility into each solve's path. For an
automated global search — Sobol sampling, deduplication, minimum
certification (PSD Hessian), redundant-descent avoidance — reach for the
Python [`pounce.find_minima`](./find-minima.md), whose `multistart` and
`mlsl` methods are the production tools. Rule of thumb: **debugger sweep**
when you're asking *why* a solve is start-sensitive; **`find_minima`** when
you want the minima themselves.

---

## Ask an LLM about the state

`ask [question]` packages the current paused state — checkpoint,
residuals, step lengths, dimensions, and the KKT inertia/regularization —
into a prompt and runs it through an LLM CLI (by default **Claude Code**,
`claude -p` headless print mode), printing the reply inline. It's
AI-assisted debugging without leaving the loop:

```text
pounce-dbg> stop-at kkt
pounce-dbg> continue
pounce-dbg> ask why is the dual infeasibility stalling?
# → the model's analysis of the state + suggested options to try
```

With no question it defaults to "explain the current state and suggest
what to try next."

### Choosing the LLM

Set `$POUNCE_DBG_LLM` to pick the backend. It accepts either a **bare
provider keyword** — which expands to that CLI's correct non-interactive
invocation — or a **full command template**:

```sh
export POUNCE_DBG_LLM=claude     # Claude Code      → claude -p   (default)
export POUNCE_DBG_LLM=codex      # OpenAI Codex CLI → codex exec <prompt>
export POUNCE_DBG_LLM=gemini     # Google Gemini    → gemini -p <prompt>
export POUNCE_DBG_LLM=llm        # simonw's llm     → llm <prompt>
# …or a full template:
export POUNCE_DBG_LLM='llm -m claude-opus'  # any prompt-on-stdin CLI
export POUNCE_DBG_LLM='mytool --ask {}'     # prompt substituted into {}
```

For a template, the prompt is fed on the tool's stdin unless it contains a
`{}` placeholder, in which case it is substituted as an argument. A bare
word that isn't a known provider is treated as a program name with the
prompt on stdin.

**Graceful when the CLI is absent.** If the selected tool isn't installed
or on `PATH`, `ask` returns an error naming the tool and listing the
provider keywords — the rest of the debugger (and the solve) is
unaffected. `ask` is the only command that shells out; nothing else
depends on an LLM being present.

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
  Ctrl-C also breaks into any other debug mode mid-`continue`.

**Ctrl-C at the prompt.** At a rustyline prompt Ctrl-C arrives as input,
not a signal, so it has its own analogous double-tap: the **first** Ctrl-C
cancels the current input line (readline convention), a **second** in a row
**stops the solve** (a clean `UserRequestedStop`, same as `quit`). So
whether you are running or sitting at the prompt, two Ctrl-Cs always get you
out; `quit`/`q` and Ctrl-D (EOF, which *detaches* and finishes) remain the
explicit exits.

---

## Scripting

Run a sequence of debugger commands from a file — one per line, `#` and
`//` comments and blank lines skipped:

```text
# warmup.pdbg
break if inf_pr<1e-6
watch mu
stop-at after_search_dir
continue
```

```sh
pounce problem.nl --debug-script warmup.pdbg        # run at the first pause
```

```text
pounce-dbg> source warmup.pdbg                       # or interactively
```

A script runs top-to-bottom and stops early if a command resumes or
stops the solve (so ending with `continue` hands control back at the
first breakpoint). `--debug-script` implies `--debug` when no `--debug*`
mode is given, and runs once at the first pause (not on a `resolve`).

### Example: a scripted initialization-sensitivity run

Because `load`, `sweep`, and `set opt` are ordinary commands, a whole
diagnostic fits in a script file. This one watches each solve's path and
sweeps a set of externally-generated starts:

```text
# sensitivity.pdbg — generate starts.txt first (e.g. numpy.savetxt)
break on resto_entered      # surface any start that falls into restoration
sweep starts.txt            # one solve per row; tabulated at the end
```

```sh
pounce model.nl --debug-script sensitivity.pdbg
```

Or compare a baseline against a what-if on the same starts by staging an
option before the sweep:

```text
# adaptive-vs-monotone.pdbg
set opt mu_strategy adaptive
multistart 16 0.2           # 16 sampled restarts, all under adaptive μ
```

### Example: drive a multistart from a program (JSON protocol)

For many variables and many starts, hold the `x0`s as arrays in a driver
program and let it assemble the commands — no point is ever typed. The
`--debug-json` protocol emits a `sweep_result` per solve and a final
`sweep_summary`:

```python
import subprocess, json, numpy as np

p = subprocess.Popen(["pounce", "big.nl", "--debug-json"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
send = lambda c, **k: (p.stdin.write(json.dumps({"cmd": c, **k}) + "\n"), p.stdin.flush())

recv = lambda: json.loads(p.stdout.readline())
recv()                                   # hello
recv()                                   # initial pause

# Option A — let the debugger sample: N restarts (uniform in finite boxes).
send("multistart", args=["32", "0.25"])

# Option B — supply your own starts via a file and sweep it:
# np.savetxt("starts.txt", my_sampler(n=32), delimiter=",")
# send("sweep", args=["starts.txt"])

results = []
for line in p.stdout:
    ev = json.loads(line)
    if ev.get("event") == "sweep_result":
        results.append((ev["status"], ev["objective"]))
    elif ev.get("event") == "sweep_summary":
        print(f"{ev['succeeded']}/{ev['solves']} ok, "
              f"{ev['distinct_minima']} distinct minima, "
              f"best obj {ev['best_objective']:.6e}")
        break
```

Each `sweep_result` carries `index`, `status`, `iters`, `objective`,
`inf_pr`, and the `seed` it started from; the `sweep_summary` adds
`distinct_minima`, `best_index`, and `best_objective`. A client can
feature-detect support via `hello.capabilities.sweep`.

## Exit model

| Path | Result |
|---|---|
| `quit` | stops now → `UserRequestedStop` |
| Ctrl-C ×2 at the prompt | cancel line, then stop → `UserRequestedStop` |
| Ctrl-C ×2 mid-`continue` | break in, then abort (exit 130) |
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
| `print <what>` (`p`) | block, `d`-block, scalar, `kkt`, or `residuals` |
| `print equation <name\|row>` | source algebra of a constraint, by model name or `.nl` row |
| `step` (`s`, `n`) | run to next `iter_start` |
| `step sub` / `stepi` (`si`) | run to next checkpoint of any kind |
| `continue` (`c`) | run to next breakpoint |
| `run N` (`r`) | run until iteration N |
| `break …` (`b`) | iteration / `if` / `on` breakpoints; list; `clear`; `del N` |
| `stop-at <cp>` | always pause at a checkpoint |
| `set mu/x/<block>/opt …` | mutate μ, the iterate, or stage an option |
| `get opt <name>` (`get <name>`) | report an option's current/staged value, source, and default |
| `opt [filter]` | list/search registered options |
| `complete <prefix>` | completion candidates |
| `viz <target>` | open an artifact in a viewer |
| `save [path]` | dump the iterate to JSON |
| `load <file> [block]` | read a block (default `x`) from a save artifact / numeric file |
| `sweep <file>` | one solve per start in `<file>`; tabulate outcomes |
| `multistart <N> [rel]` | `N` restarts (uniform in each finite box; jitter elsewhere); tabulate |
| `watch <target>` (`display`) | auto-print a target at every pause |
| `tbreak N` (`tb`) | one-shot iteration breakpoint |
| `commands N <c>;<c>…` | auto-run commands when iteration N's breakpoint hits (`commands N clear` removes) |
| `watchpoint <blk>[<i>] [τ]` (`wp`) | pause when a value changes by > τ |
| `diff` | what changed in the iterate since the last iteration |
| `diagnose` (`diag`) | live health report: named culprit residuals, KKT inertia, stalls |
| `source <file>` | run debugger commands from a file |
| `goto N` / `restart` | soft-rewind to a captured iteration |
| `resolve` | re-solve from current x with staged options |
| `ask [question]` | ask an LLM about the state (default Claude Code; `$POUNCE_DBG_LLM`=`claude`/`codex`/`gemini`/`llm` or a template) |
| `progress [on/off]` | toggle JSON progress events |
| `detach` | stop pausing; run to completion |
| `quit` (`q`, `exit`) | stop the solve |

---

## The JSON protocol

`--debug-json` makes **stdout a pure stream of newline-delimited JSON
objects** (the banner, problem stats, and final summary are routed to
stderr, and `print_level` is forced to 0). A program reads one JSON
object per line.

### For an LLM agent: the whole contract

You do **not** need this page to drive the debugger — the protocol is
self-describing. The contract is five lines:

1. **Launch** `pounce <model> --debug-json` (or `--problem <name>`), with
   the child's stdin and stdout piped.
2. **Read the first line — `hello`.** It enumerates everything you can do:
   `commands` (the verbs), `events` (breakpoint triggers), `checkpoints`
   (where you can pause), `metrics` (the scalar field names), `blocks`
   (the inspectable vectors), and a `capabilities` map. **Feature-detect
   off these lists**, never off the version string.
3. **Send commands**, one JSON object (or bare string) per line, e.g.
   `{"cmd":"break if inf_pr<1e-6","id":1}` then `{"cmd":"continue","id":2}`.
   Set `id` to correlate the matching `result`.
4. **Read events** until you see the one you want. Every `pause` /
   `progress` / `terminated` event carries the same scalar metric fields,
   under the exact names listed in `hello.metrics`
   (`objective`, `mu`, `inf_pr`, `inf_du`, `nlp_error`,
   `complementarity`, `iter`) — so you can index them directly.
5. **Finish** with `{"cmd":"continue"}` to run to completion (then read
   `terminated`), or `{"cmd":"quit"}` to stop early.

A complete minimal transcript (→ sent, ← received), eliding long lines:

```text
←  {"event":"hello","protocol":"pounce-dbg/1","commands":[…],"metrics":[…],…}
←  {"event":"pause","checkpoint":"iter_start","iter":0,"objective":24.2,…}
→  {"cmd":"break if inf_du<1e-6","id":1}
←  {"event":"result","request_id":1,"command":"break","ok":true,…}
→  {"cmd":"continue","id":2}
←  {"event":"progress","iter":1,"objective":4.7,"inf_du":2.1e1,…}
   … more progress events …
←  {"event":"pause","checkpoint":"iter_start","iter":21,"inf_du":8.7e-7,"reason":"inf_du<1e-6"}
→  {"cmd":"continue","id":3}
←  {"event":"terminated","status":"SolveSucceeded","iterations":21,…}
```

If you are wired in through the **pounce-studio MCP server**, you don't
even spawn the CLI yourself: call `debug_start` to open a live session and
`debug_command` to step it (`debug_state` / `debug_sessions` /
`debug_close` round it out) — the server owns the child process and the
framing, and `debug_start` hands you the same `hello` handshake. Call
`debug_session_guide` for the contract and a launch snippet if you'd
rather drive `--debug-json` directly. The MCP analysis tools (`diagnose`,
`find_stalls`, …) are *post-mortem* over a finished report; the
`debug_*` tools and `--debug-json` are the *live* loop.

### Session lifecycle

1. `hello` — emitted once, up front. The handshake.
2. `pause` — at each stop.
3. `result` — one per command, echoing the client's `request_id`.
4. `progress` — one per iteration while running between pauses.
5. `sweep_result` / `sweep_summary` — during a `sweep`/`multistart`: one
   `sweep_result` per completed solve, then a `sweep_summary` at the end.
6. `terminated` — once, after the solve.

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
{"event":"hello","protocol":"pounce-dbg/1","pounce_version":"0.4.0",
 "capabilities":{"inspect":true,"mutate_iterate":true,"mutate_mu":true,
   "conditional_breakpoints":"compound","request_ids":true,
   "viz":["block","delta","kkt","L"],"save":true,"load":true,"sweep":true,
   "kkt_inspect":true,"diagnose":true,"llm_assist":true,
   "pause_command":true,"equations":false,"structural_diagnose":false,
   "rewind":"primal_dual","resolve":true,"terminal_checkpoint":true,
   "interruptible":true,"progress_events":true,"async_pause":"checkpoint"},
 "checkpoints":["iter_start","after_mu","after_search_dir","after_step",
                "step_rejected","pre_restoration_entry",
                "post_restoration_exit","terminated"],
 "events":["resto_entered","resto_exited","regularized","tiny_step",
           "ls_rejected","mu_stalled","nan"],
 "commands":[…],"blocks":[…],"metrics":[…]}
```

A client should **feature-detect off `capabilities` / `checkpoints` /
`events`** rather than the protocol string — those lists are additive as
the debugger grows. A few capabilities are model-conditional: `equations`
and `structural_diagnose` are `true` only when the solve came from an
`.nl` file (which carries the source algebra and structural metadata) and
`false` for a built-in problem, as shown above.

### `pause`

```json
{"event":"pause","checkpoint":"iter_start","status":null,
 "iter":3,"mu":2.0e-2,"objective":5.05,"inf_pr":0.0,"inf_du":2.7e-14,
 "nlp_error":0.0237,"complementarity":1.9e-2,"dims":{"x":2,"s":0,"y_c":0,
 "y_d":0,"z_l":2,"z_u":2,"v_l":0,"v_u":0},"breakpoints":[],"conditions":[],
 "reason":"mu<0.05"}
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
{"event":"progress","iter":42,"mu":1.0e-5,"inf_pr":3.2e-7,"inf_du":1.1e-6,
 "objective":12.34,"nlp_error":1.1e-6,"complementarity":9.0e-6}
```

Emitted once per outer iteration during a `continue`, so a UI can show
live progress instead of a hang. Carries the same scalar fields, under
the same names, as `pause` — so `hello.metrics` names index directly off
either event. Default on; toggle with the `progress` command.

### `terminated`

```json
{"event":"terminated","status":"SolveSucceeded",
 "status_message":"Optimal Solution Found.","iterations":6,
 "objective":4.9999999,"evals":{"obj":7,"obj_grad":7,"constr":1,
 "constr_jac":12,"hess":6}}
```

### Async pause

A running `continue` can be interrupted two ways, both pausing at the
next checkpoint with a `reason`:

- **SIGINT** — `process.kill(pid, "SIGINT")` (or Ctrl-C). This is what a
  Debug Adapter's pause button maps to. Reason: `"interrupt (Ctrl-C)"`.
- **In-band command** — send `{"cmd":"pause"}` on stdin while the solve
  is running (JSON mode). No signals, so it works on Windows. Reason:
  `"pause (requested)"`.

`hello.capabilities.async_pause` is `"checkpoint"`, and
`pause_command` is `true`.

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

### 6. Is this solve sensitive to its start?

```text
break on resto_entered       # flag any start that falls into restoration
multistart 16                # 16 restarts (uniform in each finite box)
# → per-solve lines, then a table: succeeded / distinct minima / best
```

Swap `multistart 16` for `sweep starts.txt` to run your own start
points (`numpy.savetxt("starts.txt", X0, delimiter=",")`). See
[Multi-start and initialization sensitivity](#multi-start-and-initialization-sensitivity).

---

## Beyond the interior-point loop

Everything above is the NLP filter-IPM. The same debugger — same command
engine, same REPL — drives the other solvers too.

### Convex and conic solves

The convex LP/QP interior-point solver and the HSDE conic drivers (SOCP,
the exponential / power cones, and small PSD cones) expose the **same**
checkpoints and commands as the NLP loop. The iterate blocks follow the QP
standard form — `x` (variables), `s` (cone slacks), `y` (equality
multipliers), `z` (inequality / cone multipliers) — and the HSDE drivers
additionally expose the homogenizing scalars `tau` / `kappa` as 1-element
blocks (`print tau`). `set <block>` and `goto` work as on the NLP path;
`set mu` is rejected, because the convex μ is *derived* from `⟨s, z⟩`
(edit `s`/`z` to move it).

```sh
pounce model.nl --debug                 # LP / convex-QP (auto-routed) — IPM REPL
pounce_cblib model.cbf --debug          # SOCP / exp / power / PSD (conic) — IPM REPL
pounce_cblib model.cbf --debug-script s.pdbg
```

### Capability matrix

The flow-control core — checkpoints, stepping, breakpoints, watchpoints,
block/scalar inspection, `diff`, `goto`/`restart`, `save`, `ask`, and the
JSON protocol — works identically on **every** backend. The table below is
just the commands whose availability is **backend- or model-conditional**;
anything not listed is universal. A command that isn't available on the
current backend returns an explicit *"not available for this solver"* error
(it never silently no-ops), and a JSON client should feature-detect off
`hello.capabilities` rather than this table.

| Command | NLP filter-IPM | Convex / conic IPM | Notes |
|---|:---:|:---:|---|
| `print kkt` | ✅ | ➖ | convex IPM exposes no augmented-system inertia |
| `print rank` | ✅ | ➖ | SVD rank of the equality Jacobian — NLP only |
| `print residuals` | ✅ | ➖ | per-component primal/dual residuals — NLP only |
| `print active` / `inactive` | ✅ | ➖ | needs a bound-slack notion |
| `print equation <name\|row>` | ⚠️ | ⚠️ | needs a source `.nl` model (`capabilities.equations`) |
| `viz kkt` / `viz L` | ✅ | ➖ | depends on a captured KKT matrix / factor |
| `diagnose` | ✅ | ➖ | live health report — NLP only |
| `resolve` | ✅ | ➖ | warm re-solve from the current iterate — NLP only |
| `sweep` / `multistart` / `load` | ✅ | ➖ | initialization-sensitivity tools — NLP only |
| `set opt <name> <val>` | ✅ | ➖ | staged option edits — NLP only |
| `set mu` | ✅ | ❌ | rejected on convex: μ is *derived* from `⟨s, z⟩` (edit `s`/`z`) |
| `set <block>` / `goto` / `restart` | ✅ | ✅ | snapshots are supported on both |

✅ available · ⚠️ model-conditional · ➖ reports "not available for this
solver" · ❌ explicitly rejected with an explanation

The streamed scalar **metric vocabulary** (`iter`, `mu`, `objective`,
`inf_pr`, `inf_du`, `nlp_error`, `complementarity`) is the same on every
backend — see `hello.metrics`. Each backend maps its native quantities onto
these NLP-centric names; the convex IPM, for instance, reports
`nlp_error = max(pinf, dinf, μ)`. A backend that has no value for a metric
reports it as JSON `null` (never a dropped field), and a test pins the
emitted set to that single advertised vocabulary so it can't drift.

> A third backend — an interactive **branch-and-bound tree debugger** for a
> spatial global solver — is not part of this release.

---

## Limitations

- **Soft rewind only.** `goto`/`restart` restore the primal-dual state,
  not strategy history (see the caveat above).
- **`set opt` is staged, not hot-applied** to a running solve; it takes
  effect on `resolve` / the next solve.

<!-- Stuck on a hard solve at 2am? There may be a `coffee` waiting at the prompt. -->
