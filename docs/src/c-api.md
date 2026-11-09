# C API

`libpounce_cinterface` is POUNCE's C ABI. It is a drop-in replacement for
Ipopt 3.14's `IpStdCInterface.h`: every upstream function name, argument list
and return code is kept, so a program (or a wrapper such as cyipopt, a GAMS or
AMPL driver, or the CasADi plugin) that links against `libipopt` can link
against `libpounce_cinterface` instead without source changes. POUNCE's own
extensions carry a `Pounce` or `Ipopt…` name that upstream does not have.

The header, `crates/pounce-cinterface/include/pounce.h`, is the reference:
each prototype is documented there. This page is the map. Every exported
symbol is named on it, and a test (`tests/export_surface.rs` in the crate)
fails if one is not, or if the header and the library disagree on any
function's name, arity or argument widths.

The `IpoptProblem` handle reaches the NLP path: the interior-point method and,
with `algorithm=active-set-sqp`, the active-set SQP loop. It does not route to
the CLI's convex QP/QCQP engines. A matrix-form convex QP or conic program goes
through the separate `PounceQp` handle instead
([below](#matrix-form-convex-qp-and-conic-programs)).

## Build and link

```sh
cargo build --release -p pounce-cinterface
cc app.c -I crates/pounce-cinterface/include \
   -L target/release -lpounce_cinterface \
   -Wl,-rpath,target/release -o app
```

`make install` copies `libpounce_cinterface.{so,dylib}` into `$PREFIX/lib`
(default `$HOME/.local/lib`). The header needs `IpoptReturnCodes.h` from the
same directory. Scalars match upstream: `ipnumber` is `double`, `ipindex` is
`int`, and `Bool` is C99 `bool` (one byte).

## Lifecycle and options

- `CreateIpoptProblem` builds a handle from the dimensions, the bounds, the
  Jacobian and Hessian nonzero counts, the index style (0 = C, 1 = Fortran)
  and the five evaluation callbacks. It returns `NULL` on an invalid
  definition. `FreeIpoptProblem` releases it (`NULL` is a no-op).
- `AddIpoptStrOption`, `AddIpoptNumOption`, `AddIpoptIntOption` set an option
  and return `FALSE` for an unknown keyword or a value of the wrong type.
  `GetPounceOptionType` says which of the three a keyword expects (it works
  with a `NULL` handle too), for callers that receive options as untyped
  text. The full list is in [Solver Options](options.md).
- `OpenIpoptOutputFile` attaches a log file at its own print level.
- `SetIpoptProblemScaling` supplies objective, variable and constraint
  scaling factors; they take effect with `nlp_scaling_method=user-scaling`
  (see [NLP and Linear-System Scaling](scaling.md)).
- `SetIntermediateCallback` installs the per-iteration callback. Returning
  `false` from it stops the solve with `User_Requested_Stop`.
- `IpoptSetNonlinearVariables` / `IpoptClearNonlinearVariables` declare
  which variables enter nonlinearly, which restricts the limited-memory
  Hessian approximation to them. Indices are in the problem's index style;
  a bad declaration is refused whole.
- `AddPounceComplementarityPair`, `SetPounceComplementarityPairs`,
  `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`,
  `GetPounceComplementarityPairs` and `AddPounceComplementsConstraint`
  declare complementarity pairs; see [Complementarity](#complementarity).
- `SetPounceBounds` / `GetPounceBounds` change and read bounds on an
  existing handle; see [Updating bounds](#updating-bounds).

## Solving

- `IpoptSolve` runs a solve from the starting point in `x` and writes the
  solution, constraint values, objective and multipliers into the caller's
  buffers (any output but `x` may be `NULL`). It returns an
  `ApplicationReturnStatus` from `IpoptReturnCodes.h`.
- Inside the intermediate callback, `GetIpoptCurrentIterate` reads the
  current `x`, bound multipliers, `g` and constraint multipliers, and
  `GetIpoptCurrentViolations` reads the bound violations,
  complementarity and Lagrangian gradient, in scaled or unscaled space, as in
  Ipopt 3.14. Both return `FALSE` outside the callback or when the dimensions
  passed do not match the problem's.
- With complementarity pairs declared, `IpoptSolve` runs the
  complementarity driver instead; see [Complementarity](#complementarity).

## Post-solve statistics

All of these read the most recent solve on the handle and answer "nothing"
(zero, `-1` or `FALSE`, as documented in the header) before the first one.

- `GetIpoptIterCount`, `GetIpoptSolveTime`, `GetIpoptPrimalInf`,
  `GetIpoptDualInf`, `GetIpoptComplInf`: iteration count, wall time, and the
  final primal infeasibility, dual infeasibility and complementarity error.
- `GetPounceRestorationStats`: restoration calls, inner and outer
  iterations, and wall time.
- `GetPounceFdHessianStats`: the finite-difference Hessian census (pattern
  used, nonzeros, colour groups); `pattern_used = -1` when the solve did not
  use `hessian_approximation=finite-difference`.
- `GetPounceLinearSolverStats` fills a `PounceLinearSolverStats` struct: the
  KKT backend that ran, factorization counts, pattern reuse, fill, pivot
  range and final inertia, with `NaN` / `-1` for anything not reported.
- `GetIpoptVersion` writes the library's major, minor and patch version.

## Working-set warm start

On the active-set SQP path, the converged working set (which bounds and
constraints were active) can be read back and fed to the next solve of a
nearby problem; see the [active-set SQP warm-start design note](active-set-sqp-warm-start.md).

- `IpoptGetWorkingSet` reads the last solve's working set as one status code
  per variable and per constraint (`POUNCE_WS_*` in the header).
- `IpoptSetWarmStartWorkingSet` stages one for the next solve (validated
  before it is stored); `IpoptClearWarmStartWorkingSet` drops it.
- `IpoptSolveWarmStart` is set + solve + get in one call.
- None of this combines with complementarity pairs: a staged working set
  plus declared pairs makes `IpoptSolve` (and `IpoptSolveWarmStart`) return
  `Invalid_Problem_Definition` without solving, and `IpoptGetWorkingSet`
  returns `FALSE` after a solve with pairs.

## Updating bounds

```c
int SetPounceBounds(IpoptProblem, ipindex nx, const ipindex *x_idx,
                    const ipnumber *x_L, const ipnumber *x_U,
                    ipindex ng, const ipindex *g_idx,
                    const ipnumber *g_L, const ipnumber *g_U,
                    ipindex *bad_index);
int GetPounceBounds(IpoptProblem, ipindex nx, const ipindex *x_idx,
                    ipnumber *x_L, ipnumber *x_U,
                    ipindex ng, const ipindex *g_idx,
                    ipnumber *g_L, ipnumber *g_U,
                    int view, ipindex *bad_index);
```

`SetPounceBounds` changes bounds on a handle without rebuilding it, so
everything else on the handle stays: options, scaling, the
nonlinear-variables hint, declared pairs and `Complements` declarations. It
saves that bookkeeping, not solver work: each `IpoptSolve` builds its solver
afresh either way. Warm starts stay explicit; pass the previous solution and
multipliers to `IpoptSolve` with `warm_start_init_point=yes`.

The update is **sparse**: `nx` variable entries `(x_idx[k], x_L[k], x_U[k])`
and `ng` row entries, indices in the problem's index style, at a cost that
depends on the entries and the pairs they touch, not on `n`, `m` or the
number of pairs. A NULL index array with the full count (`n` or `m`) is
dense; a NULL value array leaves that side of the named entries as it was.

It is **all or nothing**: the batch is checked against the bounds it would
install, and a refusal changes nothing. In order:

| check | code |
|---|---|
| NULL handle | `POUNCE_BOUNDS_ERR_NULL` (-1) |
| a negative count, a setter count above `n` (or `m`), or a NULL index array with a count other than 0 or the full size | `_ERR_COUNT` (-2) |
| per entry, variables first: index out of range | `_ERR_OUT_OF_RANGE` (-3) |
| the same index twice on one side | `_ERR_DUPLICATE` (-4) |
| a NaN bound | `_ERR_NAN` (-5) |
| a `Complements` variable would move to another row of the MOI table | `_ERR_COMPLEMENTS_CLASS` (-7) |
| a `Complements` row given a finite bound | `_ERR_COMPLEMENTS_ROW` (-8) |
| a declared pair would pair a different variable bound or row side | `_ERR_PAIR_CHANGED` (-6) |
| a staged SQP working set would claim something false | `_ERR_WORKING_SET` (-9) |
| inconsistent handle, internal error | `_ERR_INTERNAL` (-11) |

The range, duplicate and NaN checks run over every entry before any of the
rest. The `Complements` and pair checks then run entry by entry in batch
order (a variable's `Complements` check before its pair's), so the first
offending entry is the one reported; the working-set check comes last.
`bad_index` receives the position of the refused entry in the batch,
`0..nx-1` for variables and `nx..nx+ng-1` for rows, or `-1`. A pair is
checked once, against everything the batch does to it, so a
`_PAIR_CHANGED` refusal names the batch's first entry touching the pair
(variables come first), which need not be the entry that changed it.

**Declared pairs.** A pair's meaning is read off bounds, so an update may
move the paired bounds' values, and the variable may gain or lose the bound
that is not in the pair, but the pair must keep pairing the same variable
bound (lower or upper) with the same row side (or stay mixed). The rule is
on what is paired, not on the shape's name: `x >= 0 ⟂ g <= 0` is a legal
one-sided pair, and if `x` gains a finite upper bound the bounds classify as
one-sided-plus-bound pairing `g <= 0` with `x <= x_U` — every state valid, a
different complementarity condition. That update is refused, while the same
upper bound on `x >= 0 ⟂ g >= 0` is accepted.

**`Complements` declarations.** The handle keeps two views of the bounds.
The *user* view is what `CreateIpoptProblem` and `SetPounceBounds` wrote;
the *effective* view, which the solve and every other function see, adds
the row edits `AddPounceComplementsConstraint` made. `SetPounceBounds`
writes the user view: a `Complements` row must stay free there (the
`F >= 0` the helper wrote is not an input), and a `Complements` variable
must keep its row of the table (lower-only, upper-only, boxed, fixed, free),
since a change of row can create or remove a pair. To change either, build a
new problem. `GetPounceBounds` reads `POUNCE_BOUNDS_VIEW_USER` (0) or
`POUNCE_BOUNDS_VIEW_EFFECTIVE` (1), takes the same index arguments
(duplicates allowed), skips a NULL output array and writes nothing on a
refusal (`_ERR_VIEW` (-10) for an unknown view).

Crossed bounds (`l > u`) are not refused here; the solve reports them, as for
`CreateIpoptProblem`. The getters for the most recent solve stay readable
after an update and describe that solve. A session (`IpoptCreateSolver`)
owns its problem, so it takes no updates.

## Sessions

`IpoptCreateSolver` consumes a problem handle (and sets the caller's copy to
`NULL`) and returns an `IpoptSolver` that keeps the converged KKT factor
between operations. `IpoptSolverSolve` solves; `IpoptSolverGetKktDim`,
`IpoptSolverKktSolve` (natural units) and `IpoptSolverKktSolveScaled` (the
solver's internal scaling) back-solve against the factor;
`IpoptSolverParametricStep` and `IpoptSolverReducedHessian` answer
first-order sensitivity questions; `IpoptFreeSolver` releases it. A session
solve does not run the second-opinion ladder that `IpoptSolve` does. The
walkthrough is [Sessions](sessions.md). A problem with complementarity pairs
is refused: `IpoptCreateSolver` returns `NULL` and leaves the handle with
the caller, because the session would hold the lowered model's factor
(see [Complementarity](#complementarity)).

## Complementarity

The C API solves models with declared complementarity pairs through the
same driver as the CLI and Python, described in [Complementarity
Constraints](complementarity.md). `examples/complementarity.c` in the crate
exercises every function in this section; `make c-api-examples` builds and
runs it.

### Declaring pairs

A pair `(row, var)` pairs constraint row `g_row` with variable `x_var`, each
measured from its finite bound, in the three native shapes Python's
`Problem(complementarity=)` and the `.nl` reader accept (one-sided,
one-sided plus a plain bound, mixed; the table is under [Declaring
pairs](complementarity.md#declaring-pairs)):

```c
int     AddPounceComplementarityPair(IpoptProblem, ipindex row, ipindex var);
int     SetPounceComplementarityPairs(IpoptProblem, ipindex n_pairs,
                                      const ipindex *rows, const ipindex *vars,
                                      ipindex *bad_index /* may be NULL */);
Bool    ClearPounceComplementarityPairs(IpoptProblem);
ipindex GetPounceComplementarityPairCount(IpoptProblem);   /* -1 on NULL */
Bool    GetPounceComplementarityPairs(IpoptProblem, ipindex *rows, ipindex *vars);
```

`Add` appends; `Set` replaces the set with a batch, all or nothing, and
writes the batch position of the first refused pair to `bad_index` (`-1` on
success); `n_pairs = 0` clears. Indices are in the problem's index style,
and are read back in it. Each declaration is checked at once, against the
handle's bounds and the pairs already declared, by the classifier the
solve's lowering runs, so a pair Python or the `.nl` reader would refuse is
refused here, and a refused call changes nothing:

| code | value | meaning |
|---|---|---|
| `POUNCE_CC_OK` | 0 | declared |
| `POUNCE_CC_ERR_NULL` | -1 | NULL handle, or a NULL array with `n_pairs > 0` |
| `POUNCE_CC_ERR_OUT_OF_RANGE` | -2 | row or variable outside the problem (in its index style) |
| `POUNCE_CC_ERR_VAR_ALREADY` | -3 | the variable is already in a pair, or twice in the batch |
| `POUNCE_CC_ERR_ROW_ALREADY` | -4 | the row is already in a pair, or twice in the batch |
| `POUNCE_CC_ERR_SHAPE` | -5 | not a native shape (boxed or equality row, both sides free, a fixed variable with a free row, ...) |
| `POUNCE_CC_ERR_COUNT` | -6 | `n_pairs < 0` |
| `POUNCE_CC_ERR_INTERNAL` | -7 | an internal error caught at the boundary; not a diagnosis of the input |

Pairs persist across solves until cleared or replaced.

### MOI `[F, x] ∈ Complements`

```c
int AddPounceComplementsConstraint(IpoptProblem, ipindex row, ipindex var);
```

MathOptInterface reads the sign of `F` off `x`'s bounds; the native pairs
read both signs off bounds. `AddPounceComplementsConstraint` takes MOI's
semantics on a row with **no finite bound** and transcribes them by writing
into the handle's copy of that row's bounds, then declaring the native
pair:

| `x` bounds | edit on the handle | pair declared | returns |
|---|---|---|---|
| `l < u`, both finite | none | native mixed | `POUNCE_COMPLEMENTS_PAIR_MIXED` (0) |
| `x >= l` only | `g_L[row] = 0` (`F >= 0`) | one-sided | `POUNCE_COMPLEMENTS_PAIR_LOWER` (1) |
| `x <= u` only | `g_U[row] = 0` (`F <= 0`) | one-sided | `POUNCE_COMPLEMENTS_PAIR_UPPER` (2) |
| free | `g_L[row] = g_U[row] = 0` (`F = 0`) | none | `POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR` (3) |
| `l == u` | none (`F` free) | none | `POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR` (4) |

A negative return is a refusal that changed nothing, checked in this order:
`POUNCE_COMPLEMENTS_ERR_NULL` (-1); `_ERR_OUT_OF_RANGE` (-2); `_ERR_ROW_NOT_FREE`
(-5), for any row with a finite bound, **including one whose bound agrees
with the sign MOI implies** (`g_L = 0` with `x >= 0`: that is a native
declaration, use `AddPounceComplementarityPair`) and a row this function
already wrote; `_ERR_VAR_ALREADY_PAIRED` (-3) and `_ERR_ROW_ALREADY_PAIRED`
(-4), for a variable or row in a declared pair or in **any earlier accepted
call** of this function, pair or not; `_ERR_VAR_BOUNDS` (-6), NaN or crossed
variable bounds; and `_ERR_INTERNAL` (-7). The edits and that ledger are
permanent on the handle: `ClearPounceComplementarityPairs` and
`SetPounceComplementarityPairs` drop pairs, not edits, and do not free a row
or variable for another call. So after a clear the handle solves a plain
NLP with the edited row bounds, which is neither the original model nor the
complementarity one; the only way back to the original bounds is to free the
problem and create it again. `GetPounceComplementarityPairs` lists the
pairs declared here like any other.

### Solving

With pairs declared, `IpoptSolve` runs the complementarity driver instead
of one solve and the second-opinion ladder, honours the
`complementarity_strategy`, `complementarity_continuation` and
`complementarity_lift` options, and writes the reported point, its `g`,
objective and multipliers into the caller's buffers (the multipliers are the declared
model's; see [Multipliers](complementarity.md#multipliers)). `GetIpoptIterCount`
and the other post-solve accessors describe that reported solve, not the
whole run. With no pairs declared nothing changes. Three things refuse
pairs, because each is defined over the caller's rows while the driver
solves a lowered model with more (a product row per pair, possibly lifted
variables):

- `IpoptCreateSolver` returns `NULL` and does not consume the handle.
- `IpoptSolve` (and `IpoptSolveWarmStart`) with a staged SQP working set
  returns `Invalid_Problem_Definition` before any callback runs, and
  consumes neither the working set nor the pairs; `IpoptSolveWarmStart`
  then writes nothing to its output working-set buffers.
  `IpoptGetWorkingSet` returns `FALSE`, writing nothing, while pairs are
  declared, after a driver solve, and after any `IpoptSolve` that did not
  complete, so a set left over from an earlier SQP solve on the handle is
  never read as the current one.
- The intermediate callback fires from every lowered solve (each restarts
  at iteration 0) and returning `FALSE` ends the run with
  `User_Requested_Stop`, but `GetIpoptCurrentIterate` and
  `GetIpoptCurrentViolations` return `FALSE` throughout, for every request
  shape. This is per solve, not per thread: a pair-free `IpoptSolve` of
  another problem started from inside that callback has working inspectors
  for its own duration.

`SetIpoptProblemScaling` applies to the caller's rows and variables in
every lowered solve, and a nonlinear-variable declaration is forwarded with
the product variables added.

### What the driver did

```c
Bool    GetPounceComplementarityStats(IpoptProblem, PounceComplementarityStats *);
Bool    GetPounceComplementaritySolve(IpoptProblem, ipindex k,
                                      PounceComplementaritySolve *);
ipindex GetPounceComplementarityText(IpoptProblem, int which,
                                     char *buf, ipindex buflen);
```

Both structs are [extensible](#extensible-output-structs): set
`struct_size = sizeof` before each call, or the getter returns `FALSE`.
`GetPounceComplementarityStats` fills a `PounceComplementarityStats`: the
strategy, what was reported (`POUNCE_CC_REPORTED_ANSWER`, `_WEAK_FALLBACK`,
`_FEASIBLE_UNCERTIFIED`, `_VERDICT`) and by which formulation, the number of
lowered solves and their total iterations, the certificate of the reported
point (`POUNCE_CC_CLASS_S`, `_W`, `_NONE`, `_NOT_SEARCHED`, or `_ABSENT` when
the model could not be evaluated there), the B-stationarity check
(`POUNCE_CC_B_VERDICT_*`), escapes, lifted pairs, the natural residual and
the polish. `GetPounceComplementaritySolve` reads lowered solve `k`
(`0 <= k < n_solves`: stage, `tau`, status, iterations, violation, seconds)
into a `PounceComplementaritySolve`. `GetPounceComplementarityText` copies
the summary line (`POUNCE_CC_TEXT_SUMMARY`), the polish's reason for not
taking its point (`POUNCE_CC_TEXT_POLISH_REASON`) or the B-check's reason
for not deciding (`POUNCE_CC_TEXT_B_REASON`) with `snprintf`'s contract:
`buf = NULL, buflen = 0` returns the length without the NUL.

These are the fields of the solve report's `complementarity` block, under
the same encodings. Where the block has `null`, a real reads `NaN`; where a
whole record is absent (no certificate, no B-check), its counts and 0/1
flags read `-1`; `polish_taken` reads `0` when the polish did not run.

The getters describe the **most recent solve**. They answer `FALSE` (`-1`
for the text) before any solve, after a solve without pairs, and after a
solve that was refused (a refused solve discards the previous record).
Clearing the pairs does **not** erase the record of the solve that used
them; the next solve replaces it.

### Matching the CLI

A frontend that builds callbacks for a model it also has as a `.nl` file
(or compares against `pounce model.nl`) should expect the same **answer**
but, in two cases, a different **path** through the lowered solves. Both
were measured over the declared-pair fixtures
(`crates/pounce-cli/tests/cinterface_mpcc_parity.rs`); with them handled the
C API reproduces the CLI bit for bit, report block included.

- **Where a row's constant lives.** AMPL writes a row's constant into its
  bound: the `.nl` reader hands the solver `2 y1 - x >= 1`, where a
  hand-written model says `2 y1 - 1 - x >= 0`. The feasible set is the
  same, but `g` and the products the driver forms from it differ in the
  last bits, and on a degenerate model that moves the iteration count
  (`qpec_small`: 191 iterations in all from the `.nl` form, 139 from the
  hand-written one, same answer). To reproduce the CLI, pass the rows as
  the `.nl` file writes them. Neither form is more correct.
- **The nonlinear variables, under `hessian_approximation=limited-memory`.**
  The `.nl` header says which variables enter nonlinearly and the CLI
  restricts the L-BFGS approximation to them; a callback model is all
  nonlinear unless told. Declare the same list with
  `IpoptSetNonlinearVariables` (the product variables the lowering adds are
  included automatically). The exact Hessian ignores it.

A maximize objective is a third, cosmetic, difference: the `.nl` reader
negates it, so the CLI reports `f*` where a callback that minimizes `-f`
reports `-f*`.

## Solve report

`IpoptEnableIterHistory` (before the solve) turns on per-iteration capture,
and `IpoptWriteSolveReport` writes the last solve as a
`pounce.solve-report/v1` JSON file, at `"summary"` or `"full"` detail — the
same schema as the CLI's `--json-output` (see
[JSON Solve Report](json-output.md)). It returns `FALSE` for a `NULL`
handle, no prior solve, an unknown detail level or an I/O error.

After a solve that ran the complementarity driver, the report carries the
`complementarity` block at either detail level, written by the conversion
the CLI and Python use, so the three frontends write the same block for the
same driver outcome. Its fields are the ones the
`GetPounceComplementarity*` getters return, with `null` or an absent key
where a getter reads `NaN`, `-1` or an empty label. A solve without pairs
writes no block. A refused solve leaves no report: the write returns `FALSE`.

After a solve without pairs on which the second-opinion ladder ran a rung,
the report carries the `second_opinion` block at either detail level, from
the conversion the CLI and Python use: which rungs ran, which one was
promoted, and the base solve's status and iteration count. On a promotion
`solution.status` and `statistics.iteration_count` are the promoted rung's,
so the block is the only record in the report that the base solve failed
and of what the run cost (`total_iteration_count`). A solve on which no rung
ran (the converged one) writes no block, and neither does a solve with pairs
or a session solve, neither of which runs the ladder. Like the
`complementarity` block it describes the most recent solve and is replaced
by the next.

## Matrix-form convex QP and conic programs

Separate from the NLP handle, a `PounceQp` handle takes a convex problem in
matrix form and solves it with the convex engines, through the pipeline
Python's `solve_qp` and `solve_socp` run (`pounce_convex::pipeline`; see
[Convex Solver](convex-solver.md)). It is not an Ipopt API, so every name
carries the `Pounce` prefix. `examples/qp_socp.c` in the crate runs a QP, an
SOCP, an infeasibility certificate, the report and three refusals.

```c
PounceQp qp = PounceCreateQp(n, m_eq, m_ineq, 0);   /* 0-based triplets */
PounceQpSetObjective(qp, nnz_p, p_row, p_col, p_val, c);   /* P: lower triangle */
PounceQpSetEqualities(qp, nnz_a, a_row, a_col, a_val, b);
PounceQpSetInequalities(qp, nnz_g, g_row, g_col, g_val, h);
PounceQpSetBounds(qp, lb, NULL);                    /* NULL: no upper bounds */
PounceQpAddCone(qp, POUNCE_CONE_SOC, 3, 0.0);       /* optional */
int status = PounceQpSolve(qp);                     /* POUNCE_QP_STATUS_* or < 0 */
PounceQpGetSolution(qp, x, y, z, z_lb, z_ub, &obj, &iters);
PounceFreeQp(qp);
```

### Standard form and indexing

```text
minimize    ½ xᵀP x + cᵀx
subject to  A x = b,   G x ≤ h   (G x ⪯_K h with cones),   lb ≤ x ≤ ub
```

- `PounceCreateQp(n, m_eq, m_ineq, index_style)` fixes the dimensions: `n ≥ 1`
  variables, `m_eq ≥ 0` rows of `A`, `m_ineq ≥ 0` rows of `G`. It returns
  `NULL` on any other argument. `PounceFreeQp` releases the handle (`NULL` is
  a no-op).
- Every matrix is a list of `(row, col, value)` triplets, **0-based** when the
  handle was created with `index_style = 0` and **1-based** with
  `index_style = 1` (the `IpoptProblem` convention). Duplicate triplets are
  summed.
- `P` is given as its **lower triangle** (`row ≥ col`); an entry above the
  diagonal is refused. `P` may be empty (`nnz_p = 0`): an LP. `A` and `G` are
  given in full.
- `PounceQpSetObjective` takes `P` and `c` (length `n`, required);
  `PounceQpSetEqualities` takes `A` and `b` (length `m_eq`);
  `PounceQpSetInequalities` takes `G` and `h` (length `m_ineq`). Each must be
  called before a solve unless its row count is 0. A NULL array is accepted
  only where its length is 0.
- `PounceQpSetBounds(lb, ub)` sets the box. Either side may be `NULL` (no bound
  on that side); an entry `≤ -1e19` in `lb` or `≥ 1e19` in `ub` (infinities
  included) is no bound, as in Ipopt. Refused, storing nothing: `NaN`, and a
  box no point satisfies — `lb[i] ≥ 1e19` (`+∞`), `ub[i] ≤ -1e19` (`−∞`) or
  `lb[i] > ub[i]`. `lb[i] == ub[i]` fixes the variable. The default is no box.
  Bounds are accepted with cones too (they join the cone product as a
  trailing nonnegative block), which Python's `solve_socp` does not offer.
- Each builder validates all of its input before writing any of it and
  returns `FALSE`, changing nothing, on a `NULL` handle or array, an index out
  of range, an upper-triangle entry of `P` or a non-finite value. A later call
  replaces the earlier one. A builder that succeeds discards the previous
  solve's result (below).

### Cones

Without cones every `G` row is `Gx ≤ h`. With cones, `PounceQpAddCone(qp,
kind, dim, alpha)` appends one block to a partition of the `G` rows, in row
order, and returns its 0-based position. Each block's slack `s = h − Gx` (its
rows of it) must lie in its cone:

| kind | `dim` | rows | cone |
|---|---|---|---|
| `POUNCE_CONE_NONNEG` (0) | `≥ 0` | `dim` | `s ≥ 0` |
| `POUNCE_CONE_SOC` (1) | `≥ 1` | `dim` | `s₀ ≥ ‖(s₁, …, s_{dim−1})‖₂` |
| `POUNCE_CONE_EXP` (2) | `3` | 3 | `(s₀, s₁, s₂)` with `s₁·exp(s₀/s₁) ≤ s₂`, `s₁ > 0` (closure) |
| `POUNCE_CONE_POW` (3) | `3` | 3 | `\|s₀\| ≤ s₁^α · s₂^(1−α)`, `s₁, s₂ ≥ 0`, `α = alpha ∈ (0, 1)` |
| `POUNCE_CONE_PSD` (4) | matrix order `k ≥ 1` | `k(k+1)/2` | `smat(s) ⪰ 0` |

- **PSD layout.** A PSD block's rows are `svec(S)` of the symmetric `k × k`
  slack matrix: the lower triangle **column by column** — `(0,0), (1,0), …,
  (k−1,0), (1,1), (2,1), …, (k−1,k−1)` — with every **off-diagonal entry
  scaled by `√2`**, so that `⟨X, Y⟩ = svec(X)·svec(Y)`. The `G` rows and `h`
  entries of an off-diagonal position carry that `√2`, and so does the dual
  `z` read back for it.
- `alpha` is read for `POUNCE_CONE_POW` only and ignored otherwise.
- `AddCone` returns `POUNCE_QP_ERR_NULL` for a `NULL` handle and
  `POUNCE_QP_ERR_ARGUMENT` for an unknown kind, a `dim` out of range or an
  `alpha` outside `(0, 1)` (`NaN` and `±∞` included); nothing is added.
- At the solve, the blocks must cover exactly `m_ineq` rows, and a PSD block
  cannot be mixed with an exponential or power one (they run on different
  drivers); either is `POUNCE_QP_ERR_CONES`. SOC mixes with either family.

### Options

`PounceQpSetNumOption` and `PounceQpSetStrOption` return `FALSE`, changing
nothing, on a `NULL` handle, key or value, an unknown key, or a value out of
range. The names and meanings are the CLI's and Python's.

| key | values | meaning |
|---|---|---|
| `tol` | `(0, 1)` | convergence tolerance |
| `max_iter` | whole number `≥ 1` | iteration cap (`POUNCE_QP_STATUS_ITERATION_LIMIT`) |
| `time_limit` | seconds `≥ 0`, finite | wall-clock cap for this solve (`POUNCE_QP_STATUS_TIME_LIMIT`) |
| `tau`, `tau_max` | `(0, 1)` | fraction-to-boundary floor and ceiling; a bare `tau` raises the ceiling to it, an explicit `tau_max` wins (Python's rule) |
| `qp_reg` | finite `≥ 0` | KKT regularization of the IPM |
| `method` | `ipm` (default), `active-set` | the convex interior-point engine, or the `pounce-qp` parametric active-set engine |
| `hessian_inertia` | `auto` (default), `psd`, `indefinite`, `unknown` | where the PSD verdict on `P` comes from (below) |
| `qp_presolve` | `yes`, `no` | presolve → engine → postsolve (and, on the IPM, the unverified-optimal demotion) |
| `qp_hsde` | `yes`, `no` | homogeneous self-dual embedding on the IPM (`no`: the direct driver) |
| `qp_equilibrate` | `yes`, `no` | Ruiz equilibration |
| `qp_crossover` | `yes`, `no` | crossover after the IPM |

An option the selected engine would drop is refused at the solve
(`POUNCE_QP_ERR_OPTIONS`), not ignored:

- under `method=active-set` (no cones), setting `qp_reg`, `qp_hsde` or
  `qp_crossover` at all, to either value;
- **with any cone**, `method=active-set`, `tau`, `tau_max`, every `qp_*`
  switch (even `qp_hsde=yes`, the conic default) and any `hessian_inertia`
  but `auto` (`psd`, `indefinite`, `unknown`): Python's `solve_socp` takes
  none of them, and always runs the PSD screen, so a claim would skip a check
  Python never skips. The conic solve reads `tol`, `max_iter`, `time_limit`
  and `method=ipm`.

### PSD screen

Under `method=ipm`, with or without cones, a `P` that is not certified positive
semidefinite is refused with `POUNCE_QP_ERR_INDEFINITE` (gh#112): the convex
IPM would return a silently wrong "optimal" at a saddle point.

- With `hessian_inertia=auto` (the default) an empty `P` passes, and any other
  `P` is asked of `pounce_convex::certify_psd_lower_triangle`, as the CLI's
  dispatch does, at the CLI's band `t = 1e-9 · min(max|P_ij|, 1)`
  (`pounce_convex::psd_band`, the one definition both call; the largest
  entry after summing duplicates): `P` passes when `λ_min(P) ≥ −t`, decided by
  a sign scan for a diagonal `P` and by the inertia of an `LDLᵀ` factorization
  of `P + t·I` otherwise. A factorization that cannot decide counts as **not
  certified**. The verdict is kept on the handle: a re-solve after changing
  `b`, `h`, the bounds, the cones or an option does not factor `P` again;
  `PounceQpSetObjective` and any `hessian_inertia` setting drop it.
- This is **not** Python's check. Python decides the same question in numpy
  (`python/pounce/qp.py::_psd_verdict`: relative tolerance `1e-8 · max|P_ij|`,
  dense eigenvalues for `n ≤ 1500`, an inertia count above, an undecided
  factorization reported as unknown with a warning). On a borderline `P` —
  `λ_min` within a few tolerances of zero, or a badly scaled rank-deficient
  `P` — the C entry can refuse a `P` that Python's numpy check accepts. It
  errs toward refusing.
- Without cones, `hessian_inertia=psd`, `indefinite` or `unknown` replaces
  the verdict with the caller's claim, used as given (with cones a claim is
  refused, above): `psd` and `unknown` pass the screen,
  `indefinite` is refused under the IPM. `auto` restores the computed
  verdict. Use `psd` to accept a `P` you know is PSD and the screen refuses.
- Under `method=active-set` the verdict is not a refusal: it is handed to the
  engine, which solves an indefinite `P` to a first-order point (Python's
  `method="active-set"`; see [Convex Solver](convex-solver.md)).
- A `P` whose summed duplicate entries overflow to infinity is
  `POUNCE_QP_ERR_ARGUMENT` at the solve.

### Solving, statuses and errors

`PounceQpSolve` returns a status (`≥ 0`) and keeps the result, or a negative
error before any engine runs. It discards the previous result first, either
way. Without cones it is Python's `solve_qp` pipeline (the IPM, or presolve →
IPM → postsolve → unverified-optimal demotion with `qp_presolve=yes`; or the
active-set engine, optionally presolved); with any cone it is `solve_socp`'s
conic IPM (the symmetric driver, or the non-symmetric one when an exponential
or power cone is present).

| status | value | meaning | report `solution.status` |
|---|---|---|---|
| `POUNCE_QP_STATUS_OPTIMAL` | 0 | converged to `tol` | `Solve_Succeeded` |
| `POUNCE_QP_STATUS_OPTIMAL_INACCURATE` | 1 | reduced accuracy: usable point, residual above `tol` | `Solved_To_Acceptable_Level` |
| `POUNCE_QP_STATUS_PRIMAL_INFEASIBLE` | 2 | infeasible, with a Farkas certificate | `Infeasible_Problem_Detected` |
| `POUNCE_QP_STATUS_DUAL_INFEASIBLE` | 3 | unbounded, with a recession certificate | `Diverging_Iterates` |
| `POUNCE_QP_STATUS_ITERATION_LIMIT` | 4 | `max_iter` reached | `Maximum_Iterations_Exceeded` |
| `POUNCE_QP_STATUS_TIME_LIMIT` | 5 | `time_limit` reached | `Maximum_WallTime_Exceeded` |
| `POUNCE_QP_STATUS_NUMERICAL_FAILURE` | 6 | the KKT factorization failed | `Error_In_Step_Computation` |

| error | value | returned when |
|---|---|---|
| `POUNCE_QP_ERR_NULL` | −1 | `NULL` handle (`Solve`, `AddCone`) |
| `POUNCE_QP_ERR_ARGUMENT` | −2 | `AddCone`: bad kind, `dim` or `alpha`; `Solve`: `P` entries whose sum overflows |
| `POUNCE_QP_ERR_INCOMPLETE` | −3 | no objective, or `m_eq > 0` / `m_ineq > 0` rows never set |
| `POUNCE_QP_ERR_CONES` | −4 | the cones do not cover exactly `m_ineq` rows, or PSD mixed with exp/power |
| `POUNCE_QP_ERR_OPTIONS` | −5 | an option the engine would drop (above) |
| `POUNCE_QP_ERR_INDEFINITE` | −6 | `method=ipm` and `P` not certified PSD |
| `POUNCE_QP_ERR_INTERNAL` | −7 | an internal error caught at the boundary |

The report column is the CLI's convex mapping, so the AMPL `solve_result_num`
equals the CLI's for the same status (0, 1, 200, 300, 400, 400, 500).

### Reading the result

The getters describe the most recent `PounceQpSolve` that returned a status.
Before one, and after a solve that returned an error, they return `FALSE` and
write nothing. A builder call that succeeds (`SetObjective`,
`SetEqualities`, `SetInequalities`, `SetBounds`, `AddCone`) erases the result,
so the getters and the report never describe a problem the handle no longer
holds; they return `FALSE` until the next solve. An option call does **not**
erase it: the result is still a solve of the problem the handle holds, under
the options it ran with (as an `IpoptProblem`'s last solve survives
`AddIpoptNumOption`).

- `PounceQpGetSolution(qp, x, y, z, z_lb, z_ub, obj, iters)` copies `x`,
  `z_lb`, `z_ub` (length `n`), `y` (`m_eq`), `z` (`m_ineq`), the objective and
  the engine's iteration count; any output may be `NULL`. It returns `FALSE`,
  writing nothing, if a result vector is not of its stated length (an
  internal inconsistency, never copied past your buffer). Signs:
  `Px + c + Aᵀy + Gᵀz − z_lb + z_ub = 0` with `z, z_lb, z_ub ≥ 0` (`z` in the
  dual cone with cones; in `svec` coordinates on a PSD block).
- `PounceQpGetResiduals` and `PounceQpGetCertificate` fill
  [extensible](#extensible-output-structs) structs: set `struct_size =
  sizeof` first, or the getter returns `FALSE`.
- `PounceQpGetResiduals` fills a `PounceQpResiduals` (Python's
  `QpResult.residuals`): `primal_infeasibility`, `dual_infeasibility`,
  `complementarity`, `kkt_error` and their four `_raw` twins. Without cones the
  first four are read above their finite-precision floor, in the objective's
  units (the measurement the `Optimal` verdict is judged on), and the `_raw`
  four are the plain absolute values; with cones both sets are the plain conic
  residuals.
- `PounceQpGetCertificate` fills a `PounceQpCertificate` and is `TRUE` after
  every solve that returned a status. `kind` is
  `POUNCE_QP_CERT_PRIMAL_INFEASIBLE` (1) on a primal-infeasible status — the
  Farkas ray is in the `y`, `z`, `z_lb`, `z_ub` that `GetSolution` returns —,
  `POUNCE_QP_CERT_DUAL_INFEASIBLE` (2) on a dual-infeasible one — the recession
  ray is in `x` —, and `POUNCE_QP_CERT_NONE` (0) otherwise. `scale` is the
  ray's inf-norm (`NaN` with no certificate); the ray carries the embedding's
  `1/τ` scale, so divide by `scale` for a unit ray. `tau` and `kappa` are the
  homogeneous embedding's scalars at termination, `NaN` when the answer came
  from a driver that records none: the direct IPM (`qp_hsde=no`), the
  active-set engine and the exponential/power-cone driver.

### JSON report

`PounceQpWriteSolveReport(qp, path, detail)` writes the most recent result as
a `pounce.solve-report/v1` JSON file ([JSON Solve Report](json-output.md)) at
`detail` `"summary"` (or `NULL`) or `"full"`. It returns `FALSE` for a `NULL`
handle or path, no result, an unknown detail or an I/O error.

- `solution.engine` is `cvx-qp` (the convex IPM), `qp-active-set` (the
  active-set engine) or **`cvx-qcqp` for any solve with a cone**, exponential,
  power and PSD included: the conic IPM the CLI's QCQP arm also runs. There is
  no separate engine string for a matrix-form conic solve.
- The input descriptor is `TnlpDirect`; `problem.n_constraints` is
  `m_eq + m_ineq`; `solution.lambda` is `y` followed by `z`;
  `solution.status` is the mapping in the status table; the statistics'
  final residuals are the first four `PounceQpResiduals` fields.

### Python parity

The handle runs the functions Python's `solve_qp` / `solve_socp` run, on the
same `QpProblem`. The crate's `tests/qp_python_parity.rs` feeds 25 models
(LP, QP and every cone kind; both engines, presolve, crossover, the direct
driver and the `tau` / `qp_reg` options; infeasible, unbounded and capped) through the handle and compares
them with Python's results: **bit-exact on Linux x86_64** on status, objective,
iterations, `x`, `y`, `z`, `z_lb`, `z_ub`, all eight residuals, `tau`,
`kappa` and the certificate. Elsewhere the test asks only for Python's status
and, on a solution, `x` and the objective to `1e-6`, since the reference was generated on Linux x86_64
and floating-point results may differ by platform. The one deliberate
difference is the PSD screen above.

## Fortran shim

The same library exports the eight Fortran 77 entry points of Ipopt's
`IpStdFInterface.c`, with gfortran's trailing-underscore mangling:
`ipcreate_`, `ipfree_`, `ipsolve_`, `ipaddstroption_`, `ipaddnumoption_`,
`ipaddintoption_`, `ipsetcallback_` and `ipunsetcallback_`. Arguments are
passed by reference, strings as a buffer plus a trailing length, and the
option setters return 0 on success and 1 on failure. None of POUNCE's C
extensions are mirrored there: they are C-only.

## ABI and versioning

- `GetIpoptVersion` reports the version the library was built as, and the
  header's `POUNCE_VERSION_MAJOR`, `POUNCE_VERSION_MINOR`,
  `POUNCE_VERSION_PATCH` and `POUNCE_VERSION` match it (a test and the
  release guard both check this). Compare the two to detect a header/library
  mismatch at run time.
- Changes to the C ABI are **additions only**: no existing prototype,
  `#[repr(C)]` struct layout (`PounceLinearSolverStats`,
  `PounceComplementarityStats`, `PounceComplementaritySolve`,
  `PounceQpResiduals`, `PounceQpCertificate`) or `#define`
  value changes within the 0.x line, except that the four extensible structs
  below may gain fields at their end. New capabilities arrive as new
  symbols. A test compiles a C program against the header and holds every
  struct's size and every field's offset and width to the Rust side.
- POUNCE is pre-1.0, so wrappers gate on the minor version. Pounce.jl
  accepts a fixed range of native minors and checks its required symbols at
  load; widening that range is an ABI review (see
  [Julia and JuMP](julia.md)).

### Extensible output structs

`PounceComplementarityStats`, `PounceComplementaritySolve`,
`PounceQpResiduals` and `PounceQpCertificate` are caller-allocated and begin
with `ipindex struct_size`, which the caller sets to `sizeof` the struct
before every call:

```c
PounceQpResiduals r;
r.struct_size = sizeof r;
if (PounceQpGetResiduals(qp, &r)) { /* ... */ }
```

- A `struct_size` smaller than this release's `sizeof` — `0` from `= {0}`, or
  an uninitialized value — is refused: the getter returns `FALSE` and writes
  nothing.
- Otherwise the getter writes this release's fields only. It never writes
  `struct_size`, and never writes a byte past this release's `sizeof`,
  however large `struct_size` is.
- **Growth rule:** a later release only appends fields at the end (nothing is
  removed, reordered or retyped) and writes `min(struct_size, its sizeof)`
  bytes. A program built against an older header keeps working with a newer
  library, which honours its smaller `struct_size`; a program built against a
  newer header gets the older library's fields, with the rest of its struct
  left as it was.

This release's sizes (x86_64 Linux, 32-bit `ipindex`) are 224, 72, 72 and 32
bytes; a mirror in another language must reproduce the header's layout,
padding included, and the C layout test holds it to the Rust side.
`PounceLinearSolverStats` has no `struct_size`: it shipped before this
convention, and adding one would move every field under existing callers.
Its layout is frozen.
