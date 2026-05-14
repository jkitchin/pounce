# pounce-hsl

HSL MA57 backend for POUNCE. Port of Ipopt's
`IpMa57TSolverInterface.{hpp,cpp}`. Implements
[`SparseSymLinearSolverInterface`](../pounce-linsol) by linking against
`libcoinhsl` at runtime.

Off by default. Enable with `--features ma57` on `pounce-cli` or
`pounce-algorithm`; the option `linear_solver = ma57` then resolves to
this backend instead of falling back to FERAL.

## Prerequisites

1. **`libcoinhsl`** discoverable by the linker. Build from
   [HSL for IPOPT](https://www.hsl.rl.ac.uk/ipopt/) or install via
   your package manager.
2. On macOS, ensure `libcoinhsl.dylib` is on `DYLD_LIBRARY_PATH`
   or installed under `/usr/local/lib` (or `$HOME/.local/lib`).
3. Linux: `LD_LIBRARY_PATH` or `ldconfig`-known path.

`build.rs` checks for the library via `pkg-config` and falls back to
the linker default. The `links = "coinhsl"` declaration in `Cargo.toml`
prevents accidental double-link.

## Why MA57?

MA57 is the canonical sparse symmetric indefinite Bunch-Kaufman
factorization used by Ipopt for its KKT solves. It handles the
indefiniteness inherent to the augmented system, reports inertia, and
supports the `increase_quality` / `pivtol` escalation that the IPM
needs when the system is nearly singular. FERAL provides the same
contract in pure Rust; MA57 is generally faster on large problems.

## License

EPL-2.0 for the wrapper. The HSL routines themselves are governed by
their own [HSL license](https://www.hsl.rl.ac.uk/licencing.html);
this crate does not bundle them.
