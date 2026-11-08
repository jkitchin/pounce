# pounce-cinterface

C ABI for POUNCE. Port of Ipopt's `Interfaces/IpStdCInterface.{h,cpp}`.

For Julia and JuMP, use the source-checkout
[`Pounce.jl` package](../../Pounce.jl/README.md), which provides bindings for
this ABI and a tested MathOptInterface optimizer.

Provides the `CreateIpoptProblem` / `IpoptSolve` / `FreeIpoptProblem`
entry points so existing PyIpopt, cyipopt, JuMP, and AMPL wrappers can link
against POUNCE without source changes. Function names and signatures
match upstream exactly: consumers swap `libipopt.{dylib,so}` for
`libpounce_cinterface.{dylib,so}`.

## Crate type

```toml
[lib]
crate-type = ["lib", "cdylib"]
```

The `cdylib` is what wrappers link against. The `make install` target
in the workspace root drops `libpounce_cinterface.{dylib,so}` into
`$PREFIX/lib` (default `$HOME/.local/lib`).

### Examples

`examples/*.c` are C programs against `include/pounce.h`. From the workspace
root, build and run all of them with

```sh
make c-api-examples            # release cdylib; PROFILE=debug, CC=clang also work
```

which compiles each with `-std=c99 -Wall -Wextra -Werror -pedantic`, links
`-lpounce_cinterface -lm` with an rpath to the cdylib's directory, and runs
it. `cargo test -p pounce-cinterface` does the same through
`tests/c_examples_build_and_run.rs` and also checks the line each example
prints on success; a new example needs an entry in that test's
`EXPECTED_STDOUT`. The test fails when no C compiler (`$CC`, else `cc`) can
be run; set `POUNCE_C_EXAMPLES_SKIP=1` to skip it knowingly.

- `examples/sqp_warm_start.c` — working-set warm starts across a sequence
  of perturbed QPs on the active-set SQP path.
- `examples/complementarity.c` — declared complementarity on QpecSmall: one
  pair declared natively, one through `AddPounceComplementsConstraint`, the
  refusal codes, the session refusal, the solve, the three outcome getters
  and the JSON report's `complementarity` block.
- `examples/qp_socp.c` — the matrix-form `PounceQp*` entry: a convex QP
  (solution, multipliers, residuals, report engine `cvx-qp`), an SOCP (report
  engine `cvx-qcqp`), a primal-infeasible LP and its Farkas certificate, and
  three refusal codes (`POUNCE_QP_ERR_INDEFINITE`, `_CONES`, `_ARGUMENT`).

## Surface

Every exported symbol is listed here and in the book's
[C API page](../../docs/src/c-api.md); `include/pounce.h` documents each
prototype. `tests/export_surface.rs` fails if an export is missing from the
header, either doc, or the tests, or if the header and the Rust signature
disagree.

Upstream Ipopt 3.14 entry points (same names, arguments and return codes):

- `CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
  index_style, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)` →
  `IpoptProblem` handle; `FreeIpoptProblem(problem)`.
- `AddIpoptStrOption` / `AddIpoptNumOption` / `AddIpoptIntOption` — forward
  to the application's `OptionsList`.
- `OpenIpoptOutputFile`, `SetIpoptProblemScaling`, `SetIntermediateCallback`.
- `IpoptSolve(problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U,
  user_data)` → `ApplicationReturnStatus`.
- In-callback inspectors: `GetIpoptCurrentIterate`,
  `GetIpoptCurrentViolations`.
- Post-solve: `GetIpoptIterCount`, `GetIpoptSolveTime`, `GetIpoptPrimalInf`,
  `GetIpoptDualInf`, `GetIpoptComplInf`; `GetIpoptVersion`.

POUNCE extensions:

- Post-solve diagnostics: `GetPounceRestorationStats`,
  `GetPounceFdHessianStats`, `GetPounceLinearSolverStats` (fills the
  `#[repr(C)]` `PounceLinearSolverStats`).
- `GetPounceOptionType` — which `AddIpopt*Option` setter a keyword expects
  (works with a NULL handle).
- `IpoptSetNonlinearVariables` / `IpoptClearNonlinearVariables` — restrict
  the limited-memory Hessian to the variables that enter nonlinearly.
- Declared complementarity pairs (gh#1147): `AddPounceComplementarityPair`,
  `SetPounceComplementarityPairs` (replace, all or nothing),
  `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`,
  `GetPounceComplementarityPairs`. Indices in the problem's index style;
  each declaration is validated at once by the classifier the lowering runs
  and refused with a `POUNCE_CC_ERR_*` code (`POUNCE_CC_ERR_INTERNAL` is an
  internal error, not a diagnosis). With pairs declared,
  `IpoptSolve` solves through the complementarity driver (as the CLI and
  Python do; the `complementarity_*` options apply); `IpoptCreateSolver`
  and a staged SQP working set refuse them, and the live inspectors report
  no data during such a solve.
- MOI / JuMP `[F, x] ∈ Complements` (gh#1147): `AddPounceComplementsConstraint`
  takes MOI's semantics (`F` a free row, its sign read off `x`'s bounds) and
  maps them onto the native pairs by **editing the handle's copy of the row
  bounds** — `F >= 0` for `x >= l`, `F <= 0` for `x <= u`, `F = 0` for a free
  `x`, nothing for a boxed (mixed pair) or fixed (no pair) `x`. The row must
  be free — a row that already has a side, even one agreeing with MOI's
  sign, is `POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE` and is a native
  declaration; the `POUNCE_COMPLEMENTS_*` return code says which row of the
  table applied, or why the call was refused (nothing changes then). The
  edits, and the record of which rows and variables the helper has used,
  survive `ClearPounceComplementarityPairs`; to get the original row bounds
  back, free the problem and create it again.
- Complementarity outcome (gh#1147): `GetPounceComplementarityStats`
  (`#[repr(C)]`, extensible `PounceComplementarityStats`: strategy, what was reported
  and by which formulation, total iterations, the stationarity certificate,
  the B-stationarity check, the polish), `GetPounceComplementaritySolve`
  (one lowered solve, `PounceComplementaritySolve`) and
  `GetPounceComplementarityText` (summary, polish reason, B-check reason;
  `snprintf` contract). The fields and encodings of the solve report's
  `complementarity` block (`NaN` for its `null`, `-1` for an absent
  record). They describe the most recent solve: all three answer `FALSE` /
  `-1` before any solve and after one without pairs or a refused one;
  clearing the pairs keeps the record until the next solve.
- `IpoptEnableIterHistory` + `IpoptWriteSolveReport` — emits the structured
  `pounce.solve-report/v1` JSON (see
  [`pounce-solve-report`](../pounce-solve-report)) from the most
  recent solve. Schema-compatible with the CLI's `--json-output`; after a
  complementarity-driver solve it carries the same `complementarity` block
  the CLI and Python write for the same driver outcome, and after a solve
  the second-opinion ladder ran a rung on, the same `second_opinion` block
  (gh#850: which rung was promoted, and the base solve's status and cost).
- Working-set warm start (active-set SQP): `IpoptGetWorkingSet`,
  `IpoptSetWarmStartWorkingSet`, `IpoptClearWarmStartWorkingSet`,
  `IpoptSolveWarmStart`. Example: `examples/sqp_warm_start.c`.
  Not combined with complementarity pairs.
- Factor-once / solve-many session: `IpoptCreateSolver`,
  `IpoptSolverSolve`, `IpoptSolverGetKktDim`, `IpoptSolverKktSolve`,
  `IpoptSolverKktSolveScaled`, `IpoptSolverParametricStep`,
  `IpoptSolverReducedHessian`, `IpoptFreeSolver`. Reuses the converged KKT
  factor across parametric sweeps, reduced-Hessian queries, and raw
  back-solves. The classic `IpoptSolve` API is unchanged and unaffected. See
  [`docs/src/sessions.md`](../../docs/src/sessions.md) for the
  walkthrough.

- Matrix-form convex QP / conic entry (gh#1147 Stage 3; `src/qp.rs`): an
  opaque `PounceQp` handle — `PounceCreateQp`, `PounceFreeQp`; builders
  `PounceQpSetObjective` (`P` lower triangle + `c`),
  `PounceQpSetEqualities`, `PounceQpSetInequalities`, `PounceQpSetBounds`,
  `PounceQpAddCone` (`POUNCE_CONE_*`: nonneg, SOC, exp, power, PSD);
  options `PounceQpSetNumOption`, `PounceQpSetStrOption`; `PounceQpSolve`
  (a `POUNCE_QP_STATUS_*`, or a negative `POUNCE_QP_ERR_*` before any
  engine runs); getters `PounceQpGetSolution`, `PounceQpGetResiduals`
  (`#[repr(C)]`, extensible `PounceQpResiduals`), `PounceQpGetCertificate`
  (`PounceQpCertificate`, likewise); and `PounceQpWriteSolveReport`. It runs
  `pounce_convex::pipeline`, the code Python's `solve_qp` / `solve_socp`
  run, and matches Python's results bit for bit on Linux x86_64
  (`tests/qp_python_parity.rs`). Standard form `min ½xᵀPx + cᵀx`,
  `Ax = b`, `Gx ≤ h` (`⪯_K` with cones), `lb ≤ x ≤ ub`; triplets 0- or
  1-based by the handle's index style, `P` lower triangle, a PSD cone's rows
  in `svec` order (lower triangle by columns, off-diagonals `×√2`). With a
  cone, `method=active-set`, `tau`, `tau_max`, the `qp_*` switches and any
  `hessian_inertia` but `auto` are refused (`POUNCE_QP_ERR_OPTIONS`), and the report's `solution.engine` is
  `cvx-qcqp` for every cone kind. The IPM's PSD screen comes from
  `pounce_convex::certify_psd_lower_triangle` at the CLI's band, not
  Python's numpy check, so it can refuse a borderline `P` that Python
  accepts; `hessian_inertia=psd` overrides it. See `pounce.h` and the book's
  C API page for the cone table, every option, status and error code.

All entry points are `extern "C"` and `#[no_mangle]`. Pointers are raw
and the caller owns lifetimes; the `IpoptProblem` handle is opaque
(`void*` from C).

### Fortran shim

`src/fortran.rs` ports `IpStdFInterface.c`: the eight gfortran-mangled
symbols `ipcreate_`, `ipfree_`, `ipsolve_`, `ipaddstroption_`,
`ipaddnumoption_`, `ipaddintoption_`, `ipsetcallback_` and
`ipunsetcallback_`. Arguments are by reference, strings are
`(buffer, length)` pairs with the lengths trailing, and the option setters
return 0 (`OKRetVal`) or 1. POUNCE's C extensions are not mirrored here.

### Versioning

`GetIpoptVersion` reports the crate version, and the header's
`POUNCE_VERSION_*` defines must equal it (`tests/header_abi.rs` and
`scripts/check-release-consistency.sh` both check). The ABI changes by
addition only: no existing prototype, struct layout or `#define` value
changes (the extensible structs below may gain fields at their end). The
`#[repr(C)]` structs (`PounceLinearSolverStats`,
`PounceComplementarityStats`, `PounceComplementaritySolve`,
`PounceQpResiduals`, `PounceQpCertificate`) are held to the
header field by field (size, offset, width) by a C program in
`tests/c_examples_build_and_run.rs`.

The four gh#1147 output structs (`PounceComplementarityStats`,
`PounceComplementaritySolve`, `PounceQpResiduals`, `PounceQpCertificate`)
are *extensible*: their first field is `ipindex struct_size`, which the
caller sets to `sizeof` the struct. A smaller value (a forgotten `= {0}`) is
refused — `FALSE`, nothing written — and the getter otherwise writes this
release's fields only, never `struct_size` and never a byte past this
release's `sizeof` (`write_extensible` in `src/lib.rs`). Growth rule: later
releases only append fields and write `min(struct_size, their sizeof)`, so
old callers and new libraries, and new callers and old libraries, both
work. `PounceLinearSolverStats` shipped before the convention and stays
frozen, with no `struct_size`. Tests: `tests/common/struct_size.rs`, run by
`complementarity_getters.rs` and `qp_cones_and_errors.rs`.

## Status

Feature-complete for the upstream Ipopt C ABI surface. End-to-end
solves run via `IpoptSolve` through the regular algorithm path, or through
the complementarity driver when pairs are declared; option forwarding,
intermediate callback, post-solve stats, and the JSON solve-report writer
are all wired. A callback model can take a different path through the
driver's solves than the CLI on the `.nl` file of the same model (a row's
constant in `g` or in its bound; the `.nl` nonlinear-variable list under
L-BFGS): see "Matching the CLI" in [`docs/src/c-api.md`](../../docs/src/c-api.md). The C ABI is the path PyIpopt /
cyipopt / JuMP / AMPL drivers take when they link
`libpounce_cinterface` in place of `libipopt`.

## License

EPL-2.0.
