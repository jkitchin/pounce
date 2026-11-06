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

## Surface

- `CreateIpoptProblem(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
  index_style, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)` →
  `IpoptProblem` handle.
- `IpoptSolve(problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U,
  user_data)` → `ApplicationReturnStatus`.
- `FreeIpoptProblem(problem)`.
- `AddIpoptStrOption` / `AddIpoptIntOption` / `AddIpoptNumOption` —
  forward to the application's `OptionsList`.
- `SetIntermediateCallback`.
- Post-solve accessors: `GetIpoptIterCount`, `GetIpoptSolveTime`,
  termination-detail getters.
- `IpoptWriteSolveReport` — emits the structured
  `pounce.solve-report/v1` JSON (see
  [`pounce-solve-report`](../pounce-solve-report)) from the most
  recent solve. Schema-compatible with the CLI's `--json-output`.
- Working-set warm start: `IpoptGetWorkingSet`,
  `IpoptSetWarmStartWorkingSet`, `IpoptClearWarmStartWorkingSet`,
  `IpoptSolveWarmStart`.
- Factor-once / solve-many session: `IpoptCreateSolver`,
  `IpoptSolverSolve`, `IpoptSolverGetKktDim`, `IpoptSolverKktSolve`,
  `IpoptSolverParametricStep`, `IpoptSolverReducedHessian`,
  `IpoptFreeSolver`. Reuses the converged KKT factor across
  parametric sweeps, reduced-Hessian queries, and raw back-solves.
  The classic `IpoptSolve` API is unchanged and unaffected. See
  [`docs/src/sessions.md`](../../docs/src/sessions.md) for the
  walkthrough.

All entry points are `extern "C"` and `#[no_mangle]`. Pointers are raw
and the caller owns lifetimes; the `IpoptProblem` handle is opaque
(`void*` from C).

## Status

Feature-complete for the upstream Ipopt C ABI surface. End-to-end
solves run via `IpoptSolve` through the regular algorithm path; option
forwarding, intermediate callback, post-solve stats, and the JSON
solve-report writer are all wired. The C ABI is the path PyIpopt /
cyipopt / JuMP / AMPL drivers take when they link
`libpounce_cinterface` in place of `libipopt`.

## License

EPL-2.0.
