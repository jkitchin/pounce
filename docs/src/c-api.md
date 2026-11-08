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

The C API reaches the NLP path: the interior-point method and, with
`algorithm=active-set-sqp`, the active-set SQP loop. It does not route to the
CLI's convex QP/QCQP engines.

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
objective and multipliers into the caller's buffers. `GetIpoptIterCount`
and the other post-solve accessors describe that reported solve, not the
whole run. With no pairs declared nothing changes. Three things refuse
pairs, because each is defined over the caller's rows while the driver
solves a lowered model with more (a product row per pair, possibly lifted
variables):

- `IpoptCreateSolver` returns `NULL` and does not consume the handle.
- `IpoptSolve` (and `IpoptSolveWarmStart`) with a staged SQP working set
  returns `Invalid_Problem_Definition` before any callback runs, and
  consumes neither the working set nor the pairs. `IpoptGetWorkingSet`
  returns `FALSE` after a driver solve.
- The intermediate callback fires from every lowered solve (each restarts
  at iteration 0) and returning `FALSE` ends the run with
  `User_Requested_Stop`, but `GetIpoptCurrentIterate` and
  `GetIpoptCurrentViolations` return `FALSE` throughout, for every request
  shape.

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
matrix form — `min ½xᵀPx + cᵀx` subject to `Ax = b`, `Gx ≤ h` (or `Gx ⪯_K h`
with cones) and `lb ≤ x ≤ ub` — and solves it with the convex engines, through
the same pipeline as Python's `solve_qp` and `solve_socp`
([Convex Solver](convex-solver.md)).

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

- **Lifecycle.** `PounceCreateQp` returns `NULL` unless `n ≥ 1`, `m_eq ≥ 0`,
  `m_ineq ≥ 0` and the index style (for every triplet) is 0 or 1;
  `PounceFreeQp` releases the handle (`NULL` is a no-op).
- **Builders.** `PounceQpSetObjective`, `PounceQpSetEqualities`,
  `PounceQpSetInequalities` and `PounceQpSetBounds` validate everything
  before changing anything and return `FALSE` on a `NULL` handle or array,
  an index out of range, an upper-triangle entry of `P` or a non-finite
  value. `PounceQpAddCone` appends a `POUNCE_CONE_*` block to the partition
  of the `G` rows and returns its position, or a negative `POUNCE_QP_ERR_*`.
- **Options.** `PounceQpSetNumOption` takes `tol`, `max_iter`,
  `time_limit`, `tau`, `tau_max` and `qp_reg`; `PounceQpSetStrOption`
  takes `method` (`ipm` or `active-set`), `hessian_inertia` and the
  `qp_presolve`, `qp_hsde`, `qp_equilibrate` and `qp_crossover` switches
  (`yes` or `no`), with the CLI's meanings.
- **Solve.** `PounceQpSolve` returns a status, or a negative error before
  any engine runs: an incomplete problem, cones that do not partition `G`,
  a switch the selected engine would drop, or, under `method=ipm`, a `P`
  that is not certified positive semidefinite (gh#112). The PSD verdict is
  `pounce_convex::certify_psd_lower_triangle` at the CLI's tolerance, not
  Python's numpy check, so the two can disagree on a borderline `P`.
- **Getters.** `PounceQpGetSolution`, `PounceQpGetResiduals` (a
  `PounceQpResiduals`) and `PounceQpGetCertificate` (a
  `PounceQpCertificate`: infeasibility certificate and HSDE scalars)
  describe the most recent solve that returned a status. They return
  `FALSE` before one and after a refused solve.
- **Report.** `PounceQpWriteSolveReport` writes a `pounce.solve-report/v1`
  JSON file, with `solution.engine` set to `cvx-qp`, `qp-active-set` or
  `cvx-qcqp`.

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
  `PounceComplementarityStats`, `PounceComplementaritySolve`) or `#define`
  value changes within the 0.x line. New capabilities arrive as new
  symbols. A test compiles a C program against the header and holds every
  struct's size and every field's offset and width to the Rust side.
- POUNCE is pre-1.0, so wrappers gate on the minor version. Pounce.jl
  accepts a fixed range of native minors and checks its required symbols at
  load; widening that range is an ABI review (see
  [Julia and JuMP](julia.md)).
