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
- `AddPounceComplementarityPair` declares one complementarity pair (a
  constraint row and a variable); `SetPounceComplementarityPairs` replaces
  the declared set with a batch, all or nothing, and reports the position of
  the first refused pair; `ClearPounceComplementarityPairs` drops them;
  `GetPounceComplementarityPairCount` and `GetPounceComplementarityPairs`
  read them back. Indices are in the problem's index style. Pairs are in
  the native shapes of [Complementarity Constraints](complementarity.md#c-api)
  and are checked when declared, by the same classifier the solve's
  lowering runs; a refusal returns a `POUNCE_CC_ERR_*` code (out of range,
  variable or row already paired, unsupported shape, NULL, negative count)
  and leaves the declared set unchanged. *In development: `IpoptSolve` does
  not solve with declared pairs yet.*

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

## Sessions

`IpoptCreateSolver` consumes a problem handle (and sets the caller's copy to
`NULL`) and returns an `IpoptSolver` that keeps the converged KKT factor
between operations. `IpoptSolverSolve` solves; `IpoptSolverGetKktDim`,
`IpoptSolverKktSolve` (natural units) and `IpoptSolverKktSolveScaled` (the
solver's internal scaling) back-solve against the factor;
`IpoptSolverParametricStep` and `IpoptSolverReducedHessian` answer
first-order sensitivity questions; `IpoptFreeSolver` releases it. A session
solve does not run the second-opinion ladder that `IpoptSolve` does. The
walkthrough is [Sessions](sessions.md).

## Solve report

`IpoptEnableIterHistory` (before the solve) turns on per-iteration capture,
and `IpoptWriteSolveReport` writes the last solve as a
`pounce.solve-report/v1` JSON file, at `"summary"` or `"full"` detail — the
same schema as the CLI's `--json-output` (see
[JSON Solve Report](json-output.md)). It returns `FALSE` for a `NULL`
handle, no prior solve, an unknown detail level or an I/O error.

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
  `#[repr(C)]` struct layout (`PounceLinearSolverStats`) or `#define` value
  changes within the 0.x line. New capabilities arrive as new symbols.
- POUNCE is pre-1.0, so wrappers gate on the minor version. Pounce.jl
  accepts a fixed range of native minors and checks its required symbols at
  load; widening that range is an ABI review (see
  [Julia and JuMP](julia.md)).
