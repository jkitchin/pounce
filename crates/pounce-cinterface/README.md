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
  and refused with a `POUNCE_CC_ERR_*` code. (In development: `IpoptSolve`
  does not solve with them yet.)
- `IpoptEnableIterHistory` + `IpoptWriteSolveReport` — emits the structured
  `pounce.solve-report/v1` JSON (see
  [`pounce-solve-report`](../pounce-solve-report)) from the most
  recent solve. Schema-compatible with the CLI's `--json-output`.
- Working-set warm start (active-set SQP): `IpoptGetWorkingSet`,
  `IpoptSetWarmStartWorkingSet`, `IpoptClearWarmStartWorkingSet`,
  `IpoptSolveWarmStart`. Example: `examples/sqp_warm_start.c`.
- Factor-once / solve-many session: `IpoptCreateSolver`,
  `IpoptSolverSolve`, `IpoptSolverGetKktDim`, `IpoptSolverKktSolve`,
  `IpoptSolverKktSolveScaled`, `IpoptSolverParametricStep`,
  `IpoptSolverReducedHessian`, `IpoptFreeSolver`. Reuses the converged KKT
  factor across parametric sweeps, reduced-Hessian queries, and raw
  back-solves. The classic `IpoptSolve` API is unchanged and unaffected. See
  [`docs/src/sessions.md`](../../docs/src/sessions.md) for the
  walkthrough.

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
changes.

## Status

Feature-complete for the upstream Ipopt C ABI surface. End-to-end
solves run via `IpoptSolve` through the regular algorithm path; option
forwarding, intermediate callback, post-solve stats, and the JSON
solve-report writer are all wired. The C ABI is the path PyIpopt /
cyipopt / JuMP / AMPL drivers take when they link
`libpounce_cinterface` in place of `libipopt`.

## License

EPL-2.0.
