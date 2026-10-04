# LP / QP Solver Routing

POUNCE can route **linear programs (LP)**, **convex quadratic
programs (QP)**, and **convex quadratically-constrained QPs (QCQP)** to a
specialized interior-point solver (`pounce-convex`) instead of the general
nonlinear (NLP) filter-IPM. The specialized path uses Mehrotra
predictor-corrector and reaches the solution in materially fewer iterations
on these problem classes — typically 30–50% fewer than the general NLP path
on bound- or inequality-constrained convex QPs.

Routing is **automatic and transparent**: you do not change how you
call POUNCE. The same `pounce problem.nl`, the same
`SolverFactory('pounce')` in Pyomo, and the same AMPL `solve` all work
unchanged — POUNCE inspects the problem and picks the solver.

## How routing works

When POUNCE loads a problem it classifies it into one of:

| Class            | Routed to                                  |
|------------------|--------------------------------------------|
| **LP**           | convex IPM (`pounce-convex`)               |
| **convex QP**    | convex IPM (`pounce-convex`)               |
| **convex QCQP**  | conic IPM (`pounce-convex`, SOCP)          |
| **nonconvex QP** | NLP filter-IPM (finds a local minimum)     |
| **NLP**          | NLP filter-IPM                             |

The classifier is **conservative**: a problem is sent to the convex
solver only when POUNCE can *prove* it is convex — an LP or convex QP
(degree-≤2 objective with a positive-semidefinite Hessian, linear
constraints), or a convex QCQP (additionally allowing convex-quadratic
inequality constraints, each with a positive-semidefinite Hessian and a
one-sided `≤` bound, which are reformulated to second-order cones).
Anything it cannot prove convex — transcendental terms, an indefinite
objective Hessian, a quadratic *equality*, or a quadratic inequality whose
feasible set is nonconvex — falls back to the general NLP solver, which
always produces a correct (locally optimal) answer. You never get a wrong
"optimum" from a misclassification.

The **nonconvex QP** class is narrower than "any quadratic that is not
convex": it means an indefinite objective Hessian over **linear** rows. A
model that is also curved in its *constraints* is a nonconvex QCQP and
classifies **NLP**, because the QP extractor behind the convex path keeps only
the linear part of each row — calling such a model a QP would hand an engine a
problem with its curved constraints deleted. `POUNCE_DBG_CLASSIFY=1` names
which of the two a model landed in.

> **Note on QP detection.** The AMPL `.nl` format has no dedicated
> quadratic section: a QP's quadratic terms are written into the
> nonlinear expression tree. POUNCE walks that tree to recover the
> Hessian and test convexity, the same way QP-capable AMPL solvers do.

> **Note on row constants.** A `.nl` writer may leave a constant on the
> left of a constraint — `x0 + x1 + 3 <= 6` rather than `x0 + x1 <= 3` —
> and it too lands in the nonlinear expression tree. The reader folds such
> a constant into the row's bounds when the file is read, so a model that
> is otherwise an LP still classifies as one. The shift is exact: body and
> bound move together, so the solution and every multiplier are the same
> as for the hand-folded model.

## Choosing the solver explicitly

The `solver_selection` option overrides the automatic choice. It is a
normal POUNCE option, so it works on the command line, in an options
file, or through Pyomo's `solver.options`.

| Value           | Behavior                                                                        |
|-----------------|---------------------------------------------------------------------------------|
| `auto`          | **Default.** Route by detected class (table above).                             |
| `nlp`           | Always use the NLP filter-IPM, regardless of class.                             |
| `lp-ipm`        | Force the convex IPM; **errors** if the problem is not an LP.                   |
| `qp-ipm`        | Force the convex IPM; **errors** if the problem is not LP/convex-QP.            |
| `socp`          | Force the conic IPM; **errors** if the problem is not a convex QCQP.            |
| `qp-active-set` | Force the active-set QP engine; accepts an LP or a QP with linear constraints, **convex or indefinite**; **errors** on anything else. |

```sh
# Let POUNCE decide (default):
pounce model.nl

# Force the NLP path even on a convex QP (e.g. to compare):
pounce model.nl solver_selection=nlp

# Insist the problem is a convex QP — fail loudly if it is not:
pounce model.nl solver_selection=qp-ipm

# Solve that same QP with the active-set engine instead of the IPM:
pounce model.nl solver_selection=qp-active-set
```

A forced value that does not match the detected class is rejected with
a clear message rather than silently ignored:

```text
pounce: problem class NLP does not match forced solver qp-ipm
        (expected an LP or convex QP)
```

`qp-active-set` hands the QP directly to `pounce-qp`'s
`ParametricActiveSetSolver`, through the same convex driver the IPM uses —
so it inherits presolve, postsolve, dual recovery, `.sol` writing, timing
and the convex status vocabulary. It is **not** the same route as
`algorithm=active-set-sqp`, which wraps the QP in the full SQP outer loop;
that option still exists and is the right one for a genuine NLP.

**It is the one forced selection that takes a nonconvex problem.**
`pounce-qp` handles an indefinite Hessian by construction — inertia control
shifts the H block until the reduced KKT factor has the right inertia — so
`solver_selection=qp-active-set` on a **nonconvex QP** solves it rather than
refusing it. What comes back is then a *local* optimum, exactly as
`Optimal Solution Found` means locally optimal on the NLP path, and the status
line names the class so the reader can tell:

```text
Problem class: nonconvex QP. Selected solver: active-set QP (pounce-qp) [solver_selection=qp-active-set].
POUNCE (nonconvex QP active-set, pounce-qp): Optimal Solution Found.  obj=...
```

`auto` still sends that class to the NLP filter-IPM. The class is POUNCE's
inference, and for a nonconvex model the general path is the safer default —
so the active-set route is reachable only by naming the engine. A nonconvex
**QCQP** is refused on this route for the reason given under *How routing
works* above: it does not classify as a QP at all.

**Choose it deliberately.** For a *cold, one-shot* convex QP the
interior-point path (`qp-ipm`, and what `auto` selects) is materially more
robust: on the 138-problem Maros-Mészáros set the IPM solves 137 while the
active-set engine solves substantially fewer, mostly by exhausting its
iteration budget on large degenerate instances. That is the expected
character of a cold active-set method rather than a defect — its iteration
count is combinatorial in the size of the active set, where an
interior-point count is nearly independent of problem size. The active-set
engine earns its keep on **warm-started sequences** — MPC steps,
branch-and-bound nodes, continuation — where consecutive QPs differ little
and the working set carries over; see `solve_parametric`.

What it will not do is lie: it reports `Maximum_Iterations_Exceeded` rather
than a wrong answer, and every claimed optimum is re-verified against the
original problem's KKT conditions before being reported.

### From Pyomo

```python
solver = SolverFactory('pounce')
solver.options['solver_selection'] = 'qp-ipm'   # or 'auto', 'nlp', ...
solver.solve(model)
```

## What you get back

Before solving, POUNCE prints a one-line **routing banner** naming the
detected class, the solver it selected, and the effective
`solver_selection` — so it is always clear which of POUNCE's solvers ran
and why:

```text
Problem class: LP. Selected solver: convex QP interior-point (pounce-convex) [solver_selection=auto].
```

(The banner is suppressed alongside the startup banner — `sb yes` or
JSON-debug protocol mode — to keep stdout clean for machine consumers.)

The convex IPM then reports the same way as the NLP path: an
optimal-status line, the objective value (in your original sense — a
`maximize` objective and any constant term are reported correctly), and a
`.sol` file with the primal solution when one is requested.

```text
POUNCE (LP IPM, pounce-convex): Optimal Solution Found.
        obj=2.00000000  iters=2
```

> **Driver.** The convex path uses the **homogeneous self-dual embedding
> (HSDE)** interior-point driver — the same self-dual formulation
> Clarabel/ECOS use. It is self-starting, returns verified
> infeasibility/unboundedness certificates, and conditions the KKT system
> internally through its per-cone scaling, so it solves even badly-scaled
> LPs (e.g. NETLIB `nl`, `‖c‖ ~ 1e6`) without external pre-scaling.

## Presolve

Before the convex interior-point solve, POUNCE runs a **presolve** pass
that shrinks the problem and can detect trivial infeasibility or
unboundedness without solving. It removes empty, duplicate, and
activity-redundant rows; fixes and substitutes structural columns
(singleton-row fixings, free columns, free column singletons); **folds
away two-variable equality rows** (below); and recovers both the primal
and dual of the eliminated pieces so the reported solution is for your
original problem. When it reduces the model, it logs a one-line summary:

```text
Presolve: 40 → 24 vars, 12 → 4 rows (fixed 3, free-fixed 2, substituted 3, aggregated 8, ...)
```

### Two-variable equality rows (aggregation)

A row `a₁·x + a₂·y = b` linking two variables says one of them *is* the
other, up to a scale and a shift — an arc equality between two units, a
`Reference` alias, a unit conversion. Neither variable is determined by
it, so nothing in the older catalog could act on it, and on a flowsheet
these rows are most of the model. POUNCE now substitutes one variable
for the other and drops the row, iterating to a fixed point so *chains*
of aliases collapse to a single column. Any bound on the eliminated
variable is carried across onto the one that survives, so the reduced
problem describes exactly the same feasible set.

Two things this deliberately does **not** do:

- It never calls your model infeasible. A contradictory alias system —
  `x = y` and `x = y + 1` — makes the pass stand down and hand the model
  over untouched, for the rest of presolve or the solver itself to judge.
- It does not run on the conic path (SOCP, exponential/power cones, SDP,
  SOS). Those rows are structurally coupled in fixed-size blocks that a
  substitution would rewrite.

The aggregation shares its planner with the NLP path's Phase 6, so the
two agree on what can be eliminated (see [NLP Presolve](./options.md)).

### Infeasibility verdicts are re-derived before they are reported

A presolve infeasibility comes back in milliseconds with no iteration
behind it, so when it is wrong it is the most expensive answer the solver
can give. Two reductions — forcing constraints and dominated columns —
*fix a variable* at a value they choose from a tolerance judgment, and a
fixing that is wrong is substituted into every row that variable appears
in until some row reads as contradictory: a false infeasibility, reported
against a row nowhere near the reduction that caused it.

So presolve does not report an infeasibility on the strength of the pass
that found it. It re-derives the verdict from your original model with
those two reductions switched off, and reports `Infeasible_Problem_Detected`
only if that pass reaches the same conclusion on its own. If it does not,
the model is solved normally and presolve says so:

```text
Presolve: discarded an unconfirmed infeasibility claim — <screen> (<detail>); solving normally
```

A confirmed verdict now names the screen that proved it and the row,
column, or bound it tripped on, rather than exiting silently:

```text
Presolve: proved primal infeasible — empty equality row (equality row 7 is `0 = 3e0`)
```

Nothing that only *reports* is withheld from the re-derivation — empty
rows, activity ranges, parallel rows, and emptied-row residuals all still
apply — so no infeasibility presolve could detect before goes undetected
now. What the guard costs, in the rare case it fires, is a handful of
eliminations.

### When the reduction is truncated

The reductions are iterated to a **fixpoint** — each one can expose work
for the next, so presolve keeps going until nothing fires. It also carries
a cap on how many layers that may take, and on a model with a long
bound-propagation chain the cap is what stops it. When that happens the
summary line says so:

```text
Presolve: 315 → 128 vars, 233 → 77 rows (fixed 61, ..., tightened 158, cap-truncated after 32 layers)
```

**This is common and it is not a problem.** Measured across the LP and QP
suites, the cap binds on 46% of LP models and 25% of QP models — and on
every one of the 394 models that presolve at all, it changed *only how
tightly variable boxes were narrowed*, never the structural reduction: same
variables, same rows, same fixings, aggregations, forcing rows and
dominated columns as running the iteration to convergence. Bound
propagation is the one reduction that can keep going indefinitely, so it is
what the cap ends up trimming.

What you get is still a *correct* problem — every reduction applied is a
sound transform with its own dual recovery, and your solution is postsolved
back to the original either way. The suffix is there so a reduction that
came out of a truncated loop is distinguishable from one that converged,
which matters when you are comparing two runs or reporting a bug against
presolve. There is no option to turn it up.

Presolve is on by default. Turn it off with `qp_presolve=no` (e.g. to
compare timings or isolate a solver issue):

```sh
pounce model.nl qp_presolve=no
```

### Presolve on a convex QCQP

The switch applies to the conic driver too — the one that solves convex
QCQPs. Two things about it differ from the LP/QP path, and both follow from
the same fact: a quadratic constraint is reformulated into a second-order
cone *block*, and a cone block's rows are coupled to one another.

**Only the ordinary linear rows are reduced.** Every row of a cone block is
protected: it is never dropped, never merged with another row, never used to
tighten a variable bound, and the variables it couples are excluded from the
dominated-column reduction. Dropping any single row of a block would change
which constraint the block encodes, with nothing to signal it — the answer
would simply come back wrong. So on a model that is a variable box plus
quadratic constraints and nothing else, presolve has nothing to act on and
prints no summary line. That is the expected result, not a failure.

**The loop runs once, not to a fixpoint.** The reduced cone partition has to
be readable off the surviving rows, which holds for a single pass. There is
no `cap-truncated` suffix on this path for the same reason.

Where a QCQP *does* carry ordinary linear inequalities — which is the common
shape — those are reduced exactly as on the LP/QP path, and the summary line
looks the same:

```text
Presolve: 4 → 4 vars, 9 → 7 rows (fixed 0, free-fixed 0, substituted 0, forcing 0, dominated 0, tightened 0)
```

## Tuning the convex IPM

Beyond the shared `tol` and `max_iter`, the convex engine takes these:

| Option | Default | Meaning |
|---|---|---|
| `qp_presolve` | `yes` | Presolve before the solve (above). Applies to the conic driver as well, with the cone rows protected — see [Presolve on a convex QCQP](#presolve-on-a-convex-qcqp). |
| `qp_tau` | `0.95` | Fraction-to-boundary τ ∈ (0,1): the floor of the adaptive rule, and the flat value on the predictor step and on second-order / PSD cone blocks. |
| `qp_tau_max` | `1 − 1e-12` | Ceiling of the adaptive (Mehrotra-tail) τ on orthant blocks. Set equal to `qp_tau` to pin τ flat. |
| `qp_reg` | `1e-10` | Static KKT regularization δ ≥ 0, for a stable LDLᵀ inertia. |
| `qp_infeas_tol` | `1e-7` | Relative tolerance on the value and cone-membership parts of an infeasibility / unboundedness certificate. |
| `qp_hsde` | `yes` | Homogeneous self-dual embedding (self-starting, native certificates) vs. the infeasible-start primal–dual method. |
| `qp_equilibrate` | `yes` | Ruiz-equilibrate the data first. Only when `qp_hsde=no`; HSDE conditions internally. |
| `qp_crossover` | `no` | Pure LPs only: purify the interior iterate to an exact vertex. Opt-in; slow on large degenerate LPs (#133). |
| `qp_gondzio_corr` | `3` | Maximum Gondzio multiple centrality correctors per iteration, on nonnegative-orthant blocks only. Each is one extra back-solve through the factorization already in hand, kept only if it lengthens the step. `0` disables. Both drivers honour it. |

`qp_gondzio_corr` is worth a sentence on where it does and does not
apply. The correctors box-project the complementarity products `sᵢzᵢ`
back into `[0.1·μ, 10·μ]`, which needs the product to be *elementwise* —
so the loop is gated on the cone being a pure nonnegative orthant and a
solve carrying a single second-order or PSD block never enters it. That
includes convex QCQP on the conic route, whose whole point is the SOC
reformulation. `POUNCE_DBG_GONDZIO=1` prints one line per convex solve —
iterations, correctors attempted, correctors accepted and the mean step
gain — which is the direct way to check whether the scheme is doing
anything on your model before tuning the number.

These reach the engine through the `pounce` CLI, which is the one entry
point that classifies a `.nl` model and routes it. **A library solve
refuses a non-default value** rather than accepting one it would drop —
`IpoptApplication` has no structure extraction, so it cannot route to the
convex engines at all (the same reason `solver_selection=lp-ipm` errors
there). From Python, `pounce.solve_qp` / `pounce.solve_cone` drive the
engine directly and take these knobs as typed arguments.

## Scope and limitations

- **Convex problems only.** Nonconvex (indefinite-Hessian) QPs, quadratic
  equalities, and quadratic inequalities whose feasible set is nonconvex are
  solved by the NLP path to a *local* minimum; POUNCE does not do global
  optimization.
- **Convex QCQP** (convex-quadratic constraints) routes to the conic IPM:
  each convex-quadratic inequality `½xᵀQx + aᵀx + b ≤ 0` (with `Q ⪰ 0`) is
  reformulated to one second-order cone (`Q = FᵀF`, so `‖Fx‖² = xᵀQx`) and
  solved alongside the QP objective and linear constraints.

Both the primal solution and the constraint duals are written to the
`.sol` file, in the same sign convention as POUNCE's NLP path (so Pyomo
and AMPL read them identically regardless of which solver ran).

### Requests the convex path does not implement

The convex solvers are a specialized fast path, not a drop-in for every
option the NLP path honors. Where a request would be *dropped* rather
than merely unused, routing gives way rather than answering a different
question:

| Request | Under `auto` | Under an explicit `solver_selection` |
|---|---|---|
| `obj_scaling_factor < 0` (maximize) | re-routes to the NLP path | **refused** (exit 2) — running would report the minimizer |
| `nlp_scaling_method=user-scaling` with `scaling_factor` suffixes | re-routes to the NLP path | warns; the scaling is skipped |
| `--compute-red-hessian` (or `compute_red_hessian=yes`) | re-routes to the NLP path | warns; the step is skipped |
| sIPOPT `sens_*` suffixes whose pin is **not** a unit equality row | re-routes to the NLP path | warns; the step is skipped |

A *positive* `obj_scaling_factor` is not in this table: it only rescales
conditioning, and the convex path reports natural units either way, so
both paths give the same answer.

### Sensitivity is served here, not routed away

A plain **parametric sensitivity request** — the sIPOPT `sens_*` suffixes,
without a reduced-Hessian request — is no longer in the table above. On an LP
or convex QP the convex path computes it directly, through
`pounce_convex::QpSensitivity`, and writes the same `sens_sol_state_1` block
the NLP path writes; a `.sol` consumer cannot tell which engine answered, and
the banner names the one that did.

This was a reroute until the convex arm grew a parametric step of its own.
It is worth knowing which way it goes, because the two engines have very
different costs on a large LP and the answer is the same either way.

Three things keep it in the table:

- **A reduced-Hessian request.** `QpSensitivity` has one, but it is a
  *different computation* behind the same word — a null-space projection where
  the CLI's sIPOPT path takes the Schur route. Serving it here would silently
  change which number `--compute-red-hessian` returns.
- **A conic model.** Every cone family has a face decomposition now (see
  [The convex/conic solver](convex-solver.md#cones-the-face-not-the-rows)), but
  the CLI's conic dispatch extracts through its own provenance map and mapping
  pins through that is unwritten.
- **A pin that is not `x_p = p₀` with a unit coefficient.** The convex step
  perturbs the equality right-hand side `b`; an inequality pin lives in `h`,
  which is a different perturbation. Rather than answer a different question,
  the model goes to the path that has always handled it.

**Presolve is switched off for a run that serves a sensitivity request**, on
this path exactly as on the NLP one — but not for the reason it looks like.
The convex driver postsolves back to the extracted-QP space before anything
downstream runs, so the pins stay valid with presolve on and the step is still
within `1e-6` of the NLP path's. What presolve costs is accuracy: it can fix
the very parameter the pin parametrizes and drop its row, leaving the
sensitivity to read a postsolve reconstruction rather than the KKT the solve
converged — four orders on the fixture that exercises it. Whether that can also
move the *active set* the sensitivity infers, which would be a wrong derivative
rather than a less accurate one, is not yet measured.

### When the convex path cannot certify an LP

Routing gives way one more time, and this one is decided *after* the solve
rather than before it. Under `auto`, an **LP** whose convex solve finishes
without a KKT certificate — `Solved to acceptable level (reduced accuracy)`
or `Maximum iterations exceeded` — is re-solved on the general NLP
interior-point path, which owns the whole verdict. Nothing from the
declined convex solve is printed or written, so a rerouted run still
reports exactly one status.

The case this exists for is the NETLIB `gen` / `gen1` family. They are
highly degenerate and rank-deficient, strict complementarity fails, and a
pure interior-point method cannot certify the optimal vertex: the convex
IPM spends its whole 200-iteration budget (190.8 s) and stops at a primal
residual of `1.4e-7` against `tol = 1e-8`. The NLP filter-IPM — the same
binary, the default for every other class — solves the same model in 19
iterations and 0.98 s to a strict certificate, matching Ipopt-3.14.20/MA57
to four figures. Rerouting is also the *faster* answer here: a second solve
of one second is nothing against the three minutes the first one costs.

The fallback is narrow by construction, and does not fire when:

| | why |
|---|---|
| the class is not LP (`P ≠ 0`) | a stalling convex QP is a different, unmeasured population |
| the solve certified (`Optimal Solution Found`) | there is nothing to improve, and a second solve would double the cost of every LP |
| the status is infeasible or unbounded | those verdicts carry a *verified* certificate (see below); a second solve must not overwrite a proof |
| `solver_selection` names an engine | a named engine keeps its verdict — that is what makes the stall observable |
| `max_iter` was set explicitly | a user-set budget is the question being asked; `max_iter=0` in particular must stop without a solve |
| the interactive debugger is attached | you are stepping *this* engine |

A tightened `tol` is deliberately **not** in that list: that is an accuracy
request, so trying the engine that can meet it is the right response.

An explicitly set `max_wall_time` is forwarded to every automatic convex LP,
QP, active-set QP, and SOCP route. Time spent extracting and presolving the
convex model is charged to the same budget as all engine retries. Expiration is
reported as `MaximumWallTimeExceeded`, AMPL `solve_result_num = 400`, with the
message “Maximum wallclock time exceeded.” A timed-out convex solve is final:
automatic routing never starts a fresh convex attempt or falls back to the NLP
engine. `max_cpu_time` remains an NLP-side option and is not forwarded.

### Infeasible and unbounded problems

The convex solver detects infeasibility and unboundedness directly,
reporting a clean status instead of exhausting the iteration budget:

- **Primal infeasible** — no point satisfies the constraints. Reported
  with AMPL `solve_result_num` 200.
- **Unbounded** (dual infeasible) — the objective decreases without
  bound along a feasible direction. Reported with `solve_result_num`
  300.

Each verdict is backed by a *verified* certificate (a Farkas
infeasibility proof or an unbounded recession direction that is checked,
not merely inferred), so these statuses are never reported in error; a
problem the solver cannot certify simply runs to the iteration limit —
and, if it is an LP under `auto`, is then handed to the NLP path (above).

`solver_selection=qp-active-set` follows the same contract. Its inner QP
certifies the recession ray of the *linearization*, which on a nonlinear
model is not yet a statement about the problem, so the ray is re-tested
against the true objective and constraints before the 300 is reported;
a ray that does not survive yields
`Search_Direction_Becomes_Too_Small`, never an unboundedness claim.

#### The certificate vectors are unnormalized (scale 1/τ)

The HSDE iterate is homogeneous: it carries a scalar `τ` that tends to a
positive number at an optimum and to `0` at an infeasible or unbounded
problem. At an optimum `x/τ`, `y/τ`, `z/τ` are the answer, and that
division is already done before you see it. At an infeasibility or
unboundedness verdict `τ → 0` and there is nothing to normalize *by*: the
returned `y`, `z`, `z_lb`, `z_ub` (infeasible) or `x` (unbounded) are the
Farkas / recession ray, a **direction** whose length is arbitrary and grows
like `1/τ`. On a two-variable infeasible LP the multipliers read `z ≈ 4.5e10`
(gh#990). Do not read the magnitude as a bound on anything; rescale the ray
yourself if you need a unit-length certificate, and test it with the sign
and residual conditions the verification step already applied. Python's
`solve_qp` / `solve_socp` result carries what is needed: `tau` and `kappa`
(the homogeneous scalars of the last HSDE run in the solve; `None` when the
answer came from a driver that has none, e.g. the direct driver or the
active-set engine), and `certificate_scale`, the inf-norm of the returned ray
(`(y, z, z_lb, z_ub)` for `primal_infeasible`, `x` for `dual_infeasible`,
`None` on every other status) - divide by it for a unit-norm certificate.
`kappa / tau` says how decisive the verdict is: `tau ≈ 4e-11`, `kappa ≈ 1.06`
on the two-variable example above. In Rust, `pounce_convex::hsde_scalars`
(`clear()` before the solve, `take()` after, same thread) carries the same
pair.

#### Degenerate LP duals are not unique

At a degenerate vertex the dual solution is not unique: the optimal duals
form a face. An interior-point method converges to the **relative interior**
of that face (the analytic-center-like point), so its duals can move with
data that is *inactive at the vertex* — raising an inactive upper bound from
20 to 100 changed `z` from `[0, 108.76, 0, 26.29, 91.24]` to
`[0, 112.76, 0, 38.29, 87.24]` with `x` identical (gh#990). Both are valid
optimal duals. If you need the **vertex** dual (the one a simplex code
reports, and the one a sensitivity or shadow-price reading usually wants),
set `qp_crossover=yes` (pure LPs): it purifies the interior iterate to an
exact vertex and reads the duals from that basis. The CLI prints one line
about it at the default print level, right under the `POUNCE (...)` result
line: `Crossover: simplex engine, vertex accepted; 20 superbasic(s) pushed,
22 pivot(s) (push 20, phase 1 0, phase 2 2), 1 bound flip(s); KKT error
1.29e-09 -> 2.84e-14` (engine `simplex` or the `active-set` bridge it falls
back to, whether the vertex replaced the interior iterate, the structurals
the interior point left strictly inside their bounds, basis changes by stage,
and the KKT error before and after). In Rust the same record is
`pounce_convex::crossover::take_report()`. Still not unique when the
vertex is degenerate, but a vertex of the dual face, hence stable under
changes to inactive data.

The design and roadmap live in
[`dev-notes/lp-qp-routing.md`](https://github.com/jkitchin/pounce/blob/main/dev-notes/lp-qp-routing.md).
