# pounce-cinterface

C ABI for POUNCE. Port of Ipopt's `Interfaces/IpStdCInterface.{h,cpp}`.

Provides the `IpoptCreate` / `IpoptSolve` / `IpoptFreeProblem` entry
points so existing PyIpopt, cyipopt, JuMP, and AMPL wrappers can link
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

- `IpoptCreate(n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess,
  index_style, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h)` →
  `IpoptProblem` handle.
- `IpoptSolve(problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U,
  user_data)` → `ApplicationReturnStatus`.
- `IpoptFreeProblem(problem)`.
- `AddIpoptStrOption` / `AddIpoptIntOption` / `AddIpoptNumOption` —
  forward to the application's `OptionsList`.
- `SetIntermediateCallback`.

All entry points are `extern "C"` and `#[no_mangle]`. Pointers are raw
and the caller owns lifetimes; the `IpoptProblem` handle is opaque
(`void*` from C).

## Status

Phase 11 of the port. The FFI surface, option setters, and
`FreeIpoptProblem` / `SetIntermediateCallback` are working. `IpoptSolve`
currently returns `InternalError`; end-to-end solve through the C ABI
lands when the algorithm-side optimizer is fully driving real TNLPs
through the option-table path (it already solves real problems through
the native Rust `optimize_tnlp` entry point — see
[`pounce-nlp`](../pounce-nlp)).

## License

EPL-2.0.
