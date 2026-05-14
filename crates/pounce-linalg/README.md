# pounce-linalg

Linear algebra primitives for POUNCE. Port of Ipopt's `src/LinAlg/`.

Internal crate. Like upstream, POUNCE uses an object-oriented vector /
matrix layer over concrete dense and structured types, so the
algorithm-side code can talk to compound iterates without knowing the
underlying storage.

## What's in it

- **Traits** — [`Vector`], [`Matrix`], [`SymMatrix`]: the abstract
  surface every concrete type implements (`IpVector.hpp`,
  `IpMatrix.hpp`, `IpSymMatrix.hpp`).
- **Dense** — `DenseVector`, `DenseGenMatrix`, `DenseSymMatrix`
  with companion `*Space` factories.
- **Compound** — `CompoundVector`, `CompoundMatrix`, `CompoundSymMatrix`:
  block-structured iterates, used heavily by the IPM (x, s, λ, ν, z_L,
  z_U live in a single `CompoundVector`).
- **Structured** — `DiagMatrix`, `ExpansionMatrix`, `IdentityMatrix`,
  `ZeroMatrix`, `SumMatrix`, `ScaledMatrix`, `TransposeMatrix`,
  `LowRankUpdateSymMatrix`, `MultiVectorMatrix`.
- **Sparse triplets** — `GenTMatrix`, `SymTMatrix` (COO storage),
  `TripletToCsrConverter` for handing matrices to sparse linear
  solvers.
- **BLAS-1** — `blas1` module with hand-rolled inner loops; no system
  BLAS dependency.

## Why an OO layer?

The IPM never instantiates a raw 𝐱 ∈ ℝⁿ — it operates on a compound
iterate with substructure that downstream operators (e.g. the augmented
KKT system, line-search expansion, scaling) need to know about. The
`Vector` / `Matrix` traits, with `VectorCache` / `MatrixCache` keyed on
the upstream `Tag` machinery (see [`pounce-common`](../pounce-common)),
let strategies operate on iterates without touching storage details.

## License

EPL-2.0.

[`Vector`]: src/vector.rs
[`Matrix`]: src/matrix.rs
[`SymMatrix`]: src/matrix.rs
