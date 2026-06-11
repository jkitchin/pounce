# Simplex Phase 6.2 — sparse LU basis engine (faer + an in-house update layer)

Design + record for replacing the hand-rolled dense LU basis engine
(`pounce-simplex/src/{lu,basis}.rs`) with a sparse factorization.

**Status: IMPLEMENTED (PFI-on-faer).** `FaerBasis` is the production basis
engine; the dense engine is retained under `cfg(test)` as `DenseBasis`, the
lockstep oracle. The previously-parked HiGHS ill-scaled regression
(`tests/ill_scaled_obbt.rs`, GLOBALLib `ex4_1_2`) now passes and is a live
guard. Forrest–Tomlin remains the future optimization (see below). The
architecture below is what shipped.

## The seam already exists

The simplex driver never touches `B⁻¹`. It speaks to the basis through exactly
six entry points (`simplex.rs:264,308,339,354,490,529,538`):

| Method | Contract |
|---|---|
| `identity(m)` | start basis `B = I` |
| `ftran(col, out)` | `out = B⁻¹ · col`, `col` a **sparse** column `&[(usize,f64)]` |
| `btran(row, out)` | `out = rowᵀ · B⁻¹`, `row` **dense** length `m` (forms `y = c_Bᵀ B⁻¹`) |
| `update(r, alpha)` | rank-1 product-form step; `alpha = B⁻¹ A_q` already FTRAN'd |
| `refactor(cols)` | rebuild from the sparse basic columns; `false` if singular |
| `updates_since_refactor()` | drives the `REFACTOR_INTERVAL = 50` cadence |

So Phase 6.2 is a **backend swap behind a stable interface**, not an algorithm
change. Concretely: promote `Basis` to a trait `BasisEngine` with these six
methods, keep the current dense struct as `DenseBasis` (now a *test oracle*, not
the production path), and add `FaerBasis`.

## What commercial / serious solvers actually do (the crux)

Short answer to "do they roll their own to get the rank-1 update?": **yes, every
serious simplex does — because the update *is* the simplex, and no general LU
library provides it.**

- **CPLEX, Gurobi, Xpress** (commercial) and **HiGHS, CLP/COIN** (open source)
  all maintain their *own* basis factorization-and-update machinery. They do
  **not** call a general-purpose LU (LAPACK, SuiteSparse, faer) for the per-pivot
  work.
- The factorization itself is a sparse LU with **threshold (Markowitz) pivoting**
  — trading fill against stability — i.e. the same *kind* of routine faer's
  sparse LU is, but tuned for simplex bases.
- The per-pivot **update** is the in-house part: **Forrest–Tomlin** (and the
  Suhl–Suhl refinement HiGHS uses), **Bartels–Golub**, or the older
  **product-form of the inverse (PFI)**. HiGHS additionally exploits
  *hyper-sparsity* in FTRAN/BTRAN (Huangfu & Hall).
- A general LU library gives you the **one-shot factorization** of a fixed
  matrix. It does **not** give you "replace column `q` of an already-factored
  basis cheaply." That gap is exactly what every simplex fills itself.

**Implication for us:** the right division of labor is
**factorization = faer, update = ours.** faer replaces only the periodic
`refactor` (the hard, numerically-delicate sparse-LU-with-pivoting part — the
part it is *worth* not re-deriving, the same lesson as feral). The simplex update
layer on top stays in-house because it has to. We are not choosing between "faer"
and "roll our own" — a real simplex is *both*.

## Architecture: faer factorization + PFI eta file

`FaerBasis` holds the LU of the **base basis `B₀`** (as of the last refactor)
plus a list of **eta vectors**, one per pivot since:

```
B⁻¹ = E_t · … · E_1 · B₀⁻¹          (t = updates_since_refactor)
```

- **`refactor(cols)`** — assemble a faer `SparseColMat` from the basic columns
  (each `(i, v)` in column `r` → triplet `(i, r, v)`), then
  `factorize_symbolic_lu` → numeric factorization. Store the factors, **clear the
  eta file**, reset the counter. faer returning a singular/zero-pivot error maps
  to `false` — strictly more principled than today's absolute `best <= 1e-12`
  threshold (`lu.rs:44`).
- **`ftran(col, out)`** — scatter the sparse `col` into a dense RHS, solve
  `B₀ x = col` with faer, then apply `E_1 … E_t` forward.
- **`btran(row, out)`** — apply the etas in reverse as transposes, then a faer
  **transpose** solve `B₀ᵀ y = …`.
- **`update(r, alpha)`** — push one eta `(r, alpha)`. Storage bounded by
  `REFACTOR_INTERVAL`, exactly as today; the existing driver cadence
  (`simplex.rs:538`) already caps the eta chain at 50 and refactors.

This is a **faithful drop-in**: same eta semantics, same refactor cadence, same
`NumericalFailure` path — only the dense `B⁻¹` multiply and the scalar dense LU
are replaced by faer sparse solves + a sparse base factorization.

### Why PFI first, Forrest–Tomlin later

PFI is the *minimal* change that matches the current code's behavior 1:1, so it
isolates the variable under test (the factorization) from the update scheme. It
reuses the driver's `REFACTOR_INTERVAL` logic verbatim. **Forrest–Tomlin** (which
updates `U` directly with far better fill control, and is what HiGHS/CLP use) is
the right *next* step — but it's a bigger build and belongs after PFI is green
and benchmarked. Sequence: PFI-on-faer (6.2) → FT update + hyper-sparse
FTRAN/BTRAN (a later phase) if profiling says the refactors dominate.

## Robustness & performance deltas vs. the current dense engine

**Robustness (better):** faer does real sparse threshold pivoting and reports
singularity from the factorization rather than a fixed `1e-12` magnitude cutoff;
we keep the *factors* and back-solve instead of forming an explicit dense `B⁻¹`
(killing the known inverse-formation anti-pattern in `basis.rs`). Upstream
geometric equilibration still helps the ill-scaled `ex4_1_2` case — faer's
pivoting is additive to it, not a replacement.

**Performance (better at scale, watch small `m`):** sparse fill-reducing ordering
+ supernodal blocked kernels (faer's `pulp` SIMD) replace the scalar triple-loop
`O(m³)` factor and the `O(m²)` dense `B⁻¹` apply. For sparse OBBT bases (typical)
this is an asymptotic win. For *very small* dense bases faer carries more
overhead — if profiling shows it, keep `DenseBasis` for `m` below some threshold.

## API facts (resolved against faer 0.24 source)

- Feature: the sparse solvers need `faer/sparse-linalg`; enabled additively in
  `pounce-simplex/Cargo.toml` only (the workspace dep stays `["std"]`, and we do
  **not** pull `rayon`, so the factorization is serial/deterministic).
- Factor: `SparseColMat::<usize,f64>::try_new_from_triplets(m, m, &[Triplet])`
  (sums duplicate `(row,col)` like the dense `+=`), then `.as_ref().sp_lu()
  -> Result<Lu, LuError>`. Type path: `faer::sparse::linalg::solvers::Lu`.
- Solve: the `faer::prelude::Solve` trait (blanket-impl'd for `SolveCore`) gives
  `solve_in_place` (FTRAN base) and **`solve_transpose_in_place`** (BTRAN base)
  on a `MatMut::from_column_major_slice_mut(&mut work, m, 1)`. The transpose
  solve exists, so BTRAN needs no manual `Uᵀ`/`Lᵀ` decomposition.

### One robustness gap found and closed

faer's `sp_lu` flags only **structural** singularity (an empty basic column); a
structurally-full but **numerically** singular basis (e.g. two equal columns)
factors without error, leaving a zero pivot in `U`. The dense engine caught this
via its absolute pivot threshold. `FaerBasis::refactor` closes the gap with a
cheap **probe solve** after factoring: a zero `U` pivot makes the back-solve
divide by zero, so a non-finite result ⇒ the basis is unusable ⇒ `refactor`
returns `false` (the `NumericalFailure` path). Merely *ill-conditioned* (not
exactly singular) bases are left to upstream equilibration + periodic refactor,
as production simplex codes do.

## Validation plan (the payoff of keeping `DenseBasis`)

The crate doc already promises the dense engine is "the correctness baseline it
will be validated against." Make that literal:

1. **Lockstep oracle test** — a `#[cfg(test)]` `BasisEngine` wrapper that runs
   `DenseBasis` and `FaerBasis` side by side on every FTRAN/BTRAN and asserts
   agreement to tolerance, over randomized pivot sequences.
2. **Existing regressions must stay green** — `ill_scaled_obbt.rs` (warm sweep +
   cold, HiGHS reference) and the `basis.rs` unit tests, now run against
   `FaerBasis`.
3. **Solver-level parity** — the full `pounce-simplex` and `pounce-global` OBBT
   suites unchanged; spot-check objectives against HiGHS on a few GLOBALLib LPs.

## Step order

1. Add `faer` to `pounce-simplex/Cargo.toml` (first dependency — accepted: the
   factorization is worth not re-deriving).
2. Extract `trait BasisEngine`; make the driver generic over it (or enum-dispatch
   `Dense | Faer`); current `Basis` becomes `DenseBasis`, unchanged.
3. Implement `FaerBasis` (refactor → ftran → btran → update), verifying the solve
   API fact above first.
4. Land the lockstep oracle test; run the OBBT suites.
5. Default the driver to `FaerBasis`; keep `DenseBasis` behind `#[cfg(test)]` as
   the permanent oracle.
