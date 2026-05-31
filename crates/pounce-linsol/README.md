# pounce-linsol

Symmetric linear-solver trait layer for POUNCE. Port of Ipopt's
`Algorithm/LinearSolvers/`. Defines the contracts that concrete
backends (FERAL, MA57, future MUMPS) implement; contains no FFI of its
own.

## The two-level trait stack

| Trait                                  | Operates on                | Mirrors                                  |
|----------------------------------------|----------------------------|------------------------------------------|
| [`SymLinearSolver`]                    | `SymMatrix` (compound)     | `IpSymLinearSolver.hpp`                  |
| [`SparseSymLinearSolverInterface`]     | triplet/CSC slices         | `IpSparseSymLinearSolverInterface.hpp`   |

The wrapper [`TSymLinearSolver`] (port of `IpTSymLinearSolver.{hpp,cpp}`)
adapts the second to the first, handling the triplet-to-CSR conversion
and scaling pipeline so backends only have to implement the low-level
trait.

Return states live in [`ESymSolverStatus`] (`Success`, `Singular`,
`WrongInertia`, `CallAgain`, `FatalError`). The `CallAgain` /
`pivtol_changed` protocol is preserved verbatim from MA57 because the
IPM's [`PdFullSpaceSolver`] depends on it.

## Where backends live

- [`pounce-feral`](../pounce-feral) — pure-Rust FERAL backend
  (default). No external dependencies.
- [`pounce-hsl`](../pounce-hsl) — MA57 via `libcoinhsl` (optional,
  behind the `ma57` feature on `pounce-cli` / `pounce-algorithm`).
- MUMPS — slotted for v1.1.

## Scaling

`TSymScalingMethod` is the scaling-strategy trait (port of
`IpTSymScalingMethod.hpp`). Three implementations ship:

- `IdentityScalingMethod` — null scaling, the default.
- `RuizTSymScalingMethod` (this crate) — symmetric Ruiz ∞-norm
  equilibration. Pure Rust. Selected by `linear_system_scaling = ruiz`.
- `Mc19TSymScalingMethod` (in [`pounce-hsl`](../pounce-hsl)) — HSL
  MC19 (Curtis-Reid) via `libcoinhsl`. Selected by
  `linear_system_scaling = mc19`.

## License

EPL-2.0.

[`SymLinearSolver`]: src/sym_solver.rs
[`SparseSymLinearSolverInterface`]: src/sparse_sym_iface.rs
[`TSymLinearSolver`]: src/t_sym_solver.rs
[`ESymSolverStatus`]: src/status.rs
[`PdFullSpaceSolver`]: ../pounce-algorithm/src/kkt/
