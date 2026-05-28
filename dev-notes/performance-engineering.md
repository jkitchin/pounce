# Performance engineering — design note

**Status: design only.** No code changes yet. This note is deliberately
*cross-cutting*: it applies to `pounce-feral`, the existing IPM-NLP, the
planned `pounce-convex` LP/QP/conic solvers, and every future
`pounce-linsol` consumer. It exists because
[`lp-qp-routing.md`](lp-qp-routing.md) specifies performance *targets*
(competitive with HiGHS/Clarabel) and *functional correctness*
(objective/primal/dual to 1e-6) but not the engineering methodology for
*achieving and maintaining* high performance — vectorization,
parallelism, profiling — nor any performance *gate* in CI. Today
`.github/workflows/ci.yml` gates `fmt`, `clippy -D correctness -D
suspicious`, `build`, `test`, and wheel smoke, but **no performance
regression can fail the build**, and there is no SIMD/parallel strategy
written down. This note fills both gaps.

## 1. The reproducibility-vs-performance fork — decide this first

Everything downstream depends on it.

**Current stance.** `crates/pounce-linalg/src/blas1.rs` deliberately
uses plain scalar loops with *no SIMD intrinsics and no `mul_add`*, to
stay **bit-equivalent with the netlib reference Fortran BLAS** that
upstream Ipopt builds against. This is a real asset for the **NLP port**:
bit-equivalence lets us validate `pounce-algorithm` against Ipopt
iteration-for-iteration.

**Why it does not bind `pounce-convex`.** The convex LP/QP/conic solver
is *greenfield* — there is no upstream Ipopt convex solver to match
bit-for-bit. So the bit-equivalence constraint that justifies scalar
BLAS in the NLP path has no analogue here; the convex solver is free to
vectorize, *if* we decide what level of determinism we actually require.

**Three determinism tiers** (pick a target per crate, not globally):

1. **Bit-identical to upstream Ipopt** — scalar reference BLAS, no FMA.
   *Keep for `pounce-algorithm` / `pounce-linalg` only*, where it is a
   validation asset. Do **not** impose it on the convex solver.
2. **Run-to-run + cross-platform reproducible** — a fixed binary on
   fixed inputs gives bit-identical output every run: deterministic
   reduction order, FMA used consistently (not conditionally),
   deterministic parallel reductions (fixed chunking). Allows SIMD. Does
   *not* promise equality with reference BLAS. Two sub-levels:
   - **2a — same machine, run-to-run identical.** Cheap: mainly "use
     fixed chunk sizes, don't let parallel reductions split adaptively."
   - **2b — cross-platform / cross-SIMD-width identical.** Harder:
     different lane widths (AVX2 4-wide vs AVX-512 8-wide vs NEON) force
     different reduction trees, so 2b needs a canonical accumulation
     scheme independent of hardware width, at some cost to speed.
3. **Best-effort fast** — SIMD + FMA + nondeterministic parallel
   reductions; results vary in the last few ULPs run-to-run. Gated only
   by the solution-tolerance check (§5).

**Decision.** `pounce-convex` and feral's performance-critical paths
target **tier 2**: it unlocks SIMD/FMA/parallelism while keeping
debugging and CI sane (a failing solve reproduces). Specifically, **2a
(same-machine run-to-run identity) is the firm requirement** — enforced
by the reproducibility test in §5 — and **2b (cross-platform identity)
is aspirational**, pursued where it's cheap but not allowed to block
performance. **Tier 1** stays in `pounce-algorithm`/`pounce-linalg` for
the Ipopt-validation story. **Tier 3** is allowed only behind an opt-in
feature for users who want maximum throughput and accept ULP-level
nondeterminism. In all tiers, **correctness is gated on the solution
tolerance (§5), never on bit-identity** — an optimizer's answer is
"correct" if it satisfies the KKT/feasibility tolerances, regardless of
last-bit differences.

A Rust-specific point makes tier 2 cheaper than it would be in C/Fortran:
**Rust does not auto-contract to FMA** (no `-ffp-contract=fast`
equivalent on by default). FMA happens only where code explicitly calls
`.mul_add()`, so FMA-determinism is controlled directly rather than
fought out of the optimizer.

## 2. Vectorization (SIMD)

**Landscape (2025).**

- `std::simd` (portable SIMD) — fastest portable abstraction, but
  **nightly-only**, pins the toolchain. Off the table while POUNCE
  targets stable.
- `wide` — stable, near-drop-in, slightly slower, but **build-time
  feature detection only** (no runtime CPU dispatch / `multiversion`).
- **`pulp`** — stable, portable SIMD *with runtime CPU dispatch*; this
  is what **faer** uses. Best fit for POUNCE's pure-Rust + stable
  constraints.
- `multiversion` — runtime CPU dispatch around autovectorized scalar
  code; good where hand-vectorization isn't worth it.

**Recommendation.** Use **`pulp`** for hand-vectorized hot kernels
(stable, runtime dispatch, proven in faer), and `multiversion` +
autovectorization for the simpler loops. This keeps a single binary that
dispatches AVX2/AVX-512/NEON at runtime — important for distribution
(one wheel, many CPUs) and consistent with the pure-Rust guarantee.

**Hot kernels to target** (profile first, §4):

- augmented-system / KKT assembly and the diagonal barrier updates;
- cone scaling updates (Nesterov–Todd scaling on SOC/PSD blocks);
- the large vector ops in the IPM step (`axpy`/`dot`/`nrm2` over the
  full variable vector) — but in `pounce-convex`'s own tier-2 copies,
  not by SIMD-izing the tier-1 `pounce-linalg` reference BLAS.

**faer as reference (and possible backend).** [faer](https://github.com/sarah-quinones/faer-rs)
is pure Rust, explicitly SIMD-optimized (x86-64 + Aarch64 NEON via
pulp), rayon-parallel, with sparse LLT/LDLT/Bunch-Kaufman. Because it is
*pure Rust*, it does not violate the no-C/C++ constraint that rules out
wrapping PaPILO — so faer is both the architectural reference for feral's
vectorization *and* a credible alternative backend behind
`SparseSymLinearSolverInterface` if feral's own kernels lag. Worth an
explicit build-vs-adopt evaluation for the factorization (§3).

## 3. Parallelization

**The factorization is the bottleneck — address it first.** In an IPM,
the per-iteration sparse symmetric factorization dominates wall-clock at
scale. Parallelism elsewhere is secondary. Options:

- make feral's LDLᵀ supernodal/multifrontal with task parallelism, or
- evaluate faer's sparse Cholesky/LDLT (pure Rust, rayon-parallel) as a
  `pounce-linsol` backend.

Either way, this is the highest-leverage parallel work and is *not*
LP/QP-specific — it benefits the NLP path equally.

**rayon elsewhere** (the idiomatic Rust data-parallel crate; not yet a
workspace dependency):

- presolve routines (already planned in the routing note: probing,
  dominated-column detection, constraint sparsification);
- independent per-cone work (barrier / gradient / Hessian / scaling
  updates across cone blocks are embarrassingly parallel);
- matrix assembly and multi-RHS back-solves.

**Per-call parallelism control (faer-style).** Expose parallelism as a
per-solve option, not a global that grabs every core. This matters for
(a) embedded/MPC where the caller controls the thread budget, and
(b) future B&B over `pounce-convex`, where the *outer* search is already
parallel and nested rayon pools must not oversubscribe.

## 4. Profiling & tooling

- **Sampling profiles:** `samply` or `cargo flamegraph` for "where does
  wall-clock go" on real benchmark instances.
- **Deterministic counts:** `iai-callgrind` (Cachegrind/Callgrind) for
  instruction/cache-miss counts that are stable in noisy CI (§6).
- **Discipline in hot loops:** no allocation (reuse scratch buffers
  across IPM iterations — the matrices are constant for LP/QP, §
  routing-note "constant P/A extraction"), cache-friendly CSC/CSR
  layouts, `#[inline]` on the small kernels.

## 5. Correctness checks (the invariant every perf change must preserve)

- **Solution-tolerance gate.** Across the benchmark suites
  (Mittelmann LP, Maros-Mészáros QP), every problem must still solve to
  the agreed tolerance (objective + primal + dual to 1e-6). This is the
  invariant a vectorization/parallelization change is allowed to touch
  *nothing* in — it is the definition of "still correct."
- **Cross-solver oracle.** Objective values cross-checked against
  Clarabel/HiGHS (LP/QP) and Ipopt (NLP), as the routing note's
  verification section already specifies.
- **Reproducibility test (tier 2a).** Same binary + same input ⇒
  bit-identical output, asserted in CI; catches an accidental
  nondeterministic reduction sneaking into a tier-2 path. (2b
  cross-platform identity is aspirational and not asserted.)
- **`clippy -D correctness`** stays as the existing static gate.

## 6. Gate checking (CI) — currently absent

`ci.yml` has no performance gate; a regression ships silently today.
Propose a **two-tier** scheme:

- **PR gate — instruction counts (deterministic).** Hot-kernel
  microbenchmarks under **`iai-callgrind`**, which counts instructions
  via Cachegrind and is *stable inside GitHub Actions VMs*. Wall-clock
  criterion benchmarks are too noisy to gate a PR on a cloud runner —
  use iai-callgrind for the pass/fail gate, with a small tolerance band
  to absorb codegen jitter.
- **Nightly / pre-release gate — wall-clock SGM.** Run the full
  Mittelmann/Maros-Mészáros suites and track the **shifted geometric
  mean (SGM)** of solve time across versions; fail if SGM regresses past
  a threshold. The `benchmarks/mittelmann/` harness already emits SGM
  reports — wire a regression threshold rather than only producing a
  report. `critcmp` / a continuous-benchmarking service can track the
  baseline.
- **Numerical-tolerance gate** (§5) runs in the *same* job as the
  wall-clock suite, so a "faster" change that breaks the 1e-6 tolerance
  fails even if it improves SGM.

`benchmarks/large_scale/` already contains a `sparse_qp` problem, a
ready hook for convex-QP perf benchmarking once `pounce-convex` lands.

## 7. Mapping onto the LP/QP phases

- **Phase 2** (bare IPM-QP + equilibration): stand up the tier-2
  determinism decision and the iai-callgrind PR gate on the first hot
  kernels; reuse-vs-vectorize feral here.
- **Phase 3** (Mehrotra/HSDE): vectorize the cone scaling/step kernels
  with pulp; add the wall-clock SGM nightly gate.
- **Phase 3.5** (presolve): rayon parallelism per the routing note.
- **Phases 4–6** (conic): per-cone parallelism; the cone kernels are the
  new hot paths each phase adds.
- **Factorization parallelism / faer evaluation** is cross-cutting and
  can land independently — it speeds up the NLP path too.

## References

- S. El Kazdadi et al., *faer: A linear algebra library for the Rust
  programming language*, JOSS (2024).
  <https://github.com/sarah-quinones/faer-rs> — pure-Rust SIMD (pulp) +
  rayon, sparse LLT/LDLT; reference and possible backend.
- S. Davidoff, *The state of SIMD in Rust in 2025*.
  <https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d>
  — std::simd vs wide vs pulp/macerator vs multiversion.
- `pulp`, `std::simd`, `wide`, `multiversion` crate docs.
- `iai-callgrind` (formerly iai) — deterministic instruction-count
  benchmarking for CI. <https://github.com/iai-callgrind/iai-callgrind>
- `criterion` + `critcmp` — wall-clock benchmarking and cross-run
  comparison.
- J. Demmel & H. D. Nguyen, *ReproBLAS / reproducible summation* — on FP
  non-associativity, FMA, and reproducible reductions (the basis for the
  tier-2 determinism argument).
