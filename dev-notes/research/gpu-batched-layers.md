# Design note — GPU acceleration for batched differentiable-optimization layers

**Status: research → plan. Not yet implemented.** This note proposes the
first GPU beachhead for pounce. It is scoped by two decisions taken up
front (§1): the GPU backend stays **strictly pure-Rust** (Rust toolchain
only — `wgpu`/`CubeCL`/`rust-gpu`, no C/CUDA link), and the first target
workload is **batched differentiable-optimization layers**, not
large-scale single solves. It sits beside the existing roadmap notes;
where `interior-cg-matrix-free.md` (C5) is the GPU play for *one huge*
problem, this is the GPU play for *many small identical* problems.

## 1. The two decisions that scope everything

A GPU roadmap for an interior-point NLP solver forks on two questions,
and the answers here are deliberate.

**Decision 1 — "pure Rust" means Rust-toolchain-only.** No linking
cuDSS / cuSOLVER / cuBLAS (that would be the `ma57`-feature pattern — an
optional non-pure backend — and is explicitly *not* taken here). The
GPU layer is authored in Rust and runs through a portable compute API:
`wgpu` (Vulkan / Metal / DX12 / WebGPU) with WGSL compute shaders, or a
Rust-kernel layer (`CubeCL`, `rust-gpu`) on top of it. The payoff is
cross-vendor (NVIDIA **and** AMD **and** Apple) and a single static
binary. The price is the **f64 wall**: WGSL has no portable `f64`
(Metal has none at all; Vulkan's `SHADER_F64` is a native-only
extension). A double-precision direct solver cannot live on this stack.

**Decision 2 — the first workload is batched differentiable layers.**
This is what makes Decision 1 survivable. A differentiable-optimization
layer is an ML inner loop: the gradient feeding backprop is consumed at
training tolerance, so an **f32 forward solve** (with optional cheap
refinement) is acceptable where an f64 production solve would not be.
The f64 wall that sinks a portable *direct solver* is walked around, not
hit. This is the rare pounce workload where strict-pure-Rust GPU is not
a contradiction.

The combination is coherent: the constraint and the target were chosen
to fit each other.

## 2. Why this workload is the right GPU shape

pounce already solves batches of related NLPs — `vmap_solve_parallel`
runs `B` independent IPMs across CPU worker threads
(`python/pounce/jax/_diff.py`), and the JAX layer wraps a single solve
in `jax.custom_vjp` with an implicit-function-theorem backward
(`python/pounce/jax/__init__.py` §2). The workloads that drive it —
training a model with an optimization layer, parameter sweeps, scenario
ensembles — share three properties that are individually rare and
jointly a GPU's ideal case:

1. **Identical structure.** Every batch element is the same `f`, `g`,
   the same `(n, m)`, the same sparsity pattern — only the parameter
   `p` (hence the numerical values) differs. The symbolic factorization
   is shared across the *entire batch* on top of the reuse pounce
   already gets across IPM iterations (FERAL's pattern-fingerprint
   cache, `crates/pounce-feral/src/lib.rs`).
2. **Many small problems, not one big one.** ML layers are typically
   tens to low-thousands of variables, dense or lightly sparse. This is
   precisely the regime FERAL's CPU parallelism *cannot* help — its
   tree parallelism is gated at ~10⁸ flops (`FERAL_PARALLEL`,
   `min_parallel_flops`), which a small problem never reaches. The GPU
   path is therefore **additive**, not competing with existing
   parallelism: it serves the regime CPU multicore leaves on the floor.
3. **f32-tolerant.** See Decision 2. The one regime where the portable
   GPU stack's precision ceiling is not disqualifying.

"Many tiny irregular problems a GPU hates" becomes "one big regular
**batched** workload a GPU loves." That transform is the whole idea, and
it is the same one OptNet/qpth (Amos & Kolter 2017) used to put
differentiable QP layers on the GPU.

## 3. What gets built — a batched IPM core, not a GPU-ified solver

The decisive architectural call: this is **not** GPU-ifying the existing
`SparseSymLinearSolverInterface` / `AugSystemSolver` per element. Those
traits are scalar — one problem, one factor, one solve — and pounce's
own BLAS-1/2 are deliberately scalar, no-SIMD, column-major for
bit-reproducibility (`crates/pounce-linalg/src/blas1.rs`,
`dense_sym_matrix.rs`). Threading a GPU under a per-element scalar trait
buys nothing: the win is *across* the batch dimension, not inside one
solve.

Instead, build a **batched IPM core** specialized for the layer
workload: a new crate (`pounce-gpu` / `pounce-batch-gpu`) that runs `B`
IPMs **in lockstep**, vectorized over the batch axis, with a **batched
dense/condensed KKT solve** as the inner kernel. This reuses the
*algorithmic logic* (μ-update, fraction-to-boundary, filter test,
convergence check) — re-expressed to operate on `B`-wide arrays — but
not the scalar linear-algebra path. It is narrow by construction: dense
or lightly-sparse KKT, `n` up to a few thousand, identical structure,
f32. Outside that envelope it does not apply; fall back to the CPU IPM.

The inner kernel, concretely:

- **Condense the KKT per element.** Eliminate slacks and bound-duals
  into a smaller SPD condensed (or dense reduced) system — the same
  reformulation that makes GPU IPM work elsewhere (MadNLP's condensed
  KKT). For the small-`n` layer regime the condensed system is often
  fully dense.
- **Batched dense factor + solve.** One workgroup (or thread tile) per
  batch element doing dense Cholesky / LDLᵀ + triangular solves — the
  cuSOLVER `*potrfBatched` pattern, but authored in WGSL/CubeCL. This
  is the kernel that scales: `B` independent dense factorizations are
  embarrassingly parallel across the batch.
- **Keep factors GPU-resident for the backward.** The implicit-function
  backward is one more solve against the *already-computed* factor at
  `x*`. Resident factors make it nearly free — forward solve + backward
  solve with no host round-trip. The active-set handling the existing
  VJP already specifies (`python/notebooks/03_implicit_differentiation`)
  carries over: pinned bounds and inactive inequality rows drop out of
  the batched KKT block exactly as they do on CPU.

## 4. What pounce already has to reuse

| Need | Existing component | Location |
|---|---|---|
| Batched-solve entry point + manual batching rule | `vmap_solve`, `vmap_solve_parallel` | `python/pounce/jax/_diff.py` |
| Implicit-function-theorem backward (KKT at `x*`) | `solve` `custom_vjp` | `python/pounce/jax/__init__.py` §2 |
| Active-set rule for the backward KKT block | differentiable-layer notebook | `python/notebooks/03_implicit_differentiation.ipynb` |
| IPM control logic to vectorize (μ, frac-to-bndry, filter) | the scalar IPM | `crates/pounce-algorithm/src/` |
| Condensed/reduced KKT precedent | sensitivity / reduced-Hessian | `crates/pounce-sensitivity/` |
| Shared symbolic structure across a batch | FERAL pattern-fingerprint cache (concept) | `crates/pounce-feral/src/lib.rs` |
| Warm-start duals across a batched step | `solve_with_warm`, batched warm-start nb | `python/notebooks/11_batched_warm_start.ipynb` |

The reuse is real but bounded: the *algorithm* and the *interfaces* are
reused; the scalar *linear algebra* is not — it is replaced by a batched
GPU kernel. This is a Tier-3-flavored effort (a new iteration skeleton)
but a *narrow* one — dense, batched, f32, layer-shaped problems only,
opt-in, never the default.

## 5. Proposed phasing

- **Phase 0 — spike (1–2 wks).** A `wgpu` batched dense Cholesky +
  triangular-solve kernel in f32, standalone, benchmarked against the
  CPU loop on `B` small identical SPD systems. Goal: measure the batch
  size `B` and dimension `n` at which the GPU crosses over the
  `vmap_solve_parallel` CPU-thread baseline, and confirm the f32
  accuracy envelope. Throwaway-friendly; pure decision-gathering.
  **The numerical half of Phase 0 is already done as a CPU proxy — see
  §10; it resolves the f32 risk in §8 with data. What remains for Phase
  0 is the GPU throughput crossover, which needs hardware.**
- **Phase 1 — batched condensed-KKT solve, fixed active set.** Wire the
  Phase-0 kernel as the inner solve of a batched IPM that assumes a
  fixed active set per element (valid for the convex-QP-layer case).
  Forward solve only. Shippable as a fast path for QP-shaped layers.
- **Phase 2 — full batched IPM, lockstep.** Vectorize the IPM control
  (μ-update, fraction-to-boundary, filter) over the batch; run all `B`
  to the slowest element's convergence with per-element masking (or
  periodic compaction). This is the general NLP layer.
- **Phase 3 — resident-factor backward.** Implement the
  implicit-function backward as a batched solve against the resident
  forward factor, returning the layer gradient without a host
  round-trip. Closes the train-a-model-with-a-pounce-layer loop on GPU.
- **Phase 4 — precision hardening.** Mixed-precision: f32 forward to a
  warm point, optional short f64-on-CPU refinement where the *forward
  solution* (not just the gradient) must be accurate. Documents the
  accuracy contract per use case.

Phases 0–1 are independently measurable; Phases 2–3 are the actual
differentiable-layer payoff; Phase 4 is the honesty pass.

## 6. Problem classes this unlocks

- **Differentiable optimization layers in ML models** — a pounce QP/NLP
  layer trained with batched forward+backward solves on any GPU. The
  headline class.
- **Large parameter sweeps / sensitivity ensembles** — `B` solves over
  a parameter grid, already the `vmap_solve` use case, moved to GPU.
- **Scenario-based / sample-average problems** — stochastic programming
  with many identical-structure scenario subproblems.
- **MINLP node relaxations / homotopy batches** — where many related
  NLPs share structure (overlaps C1 warm-starting, but here the win is
  batch width, not warm-start carryover).

What it does **not** unlock: a *single* large problem (that is
Interior/CG, `interior-cg-matrix-free.md`); anything needing production
f64 accuracy in the forward solution without the Phase-4 refinement;
problems whose structure differs across the batch (no shared symbolic).

## 7. Competitive read

The differentiable-optimization-layer field is real but vendor- and
language-locked:

| Prior work | Limitation pounce's path removes |
|---|---|
| OptNet / qpth (Amos & Kolter 2017) | CUDA-only; dense **QP** only |
| cvxpylayers / diffcp (Agrawal+ 2019) | CPU cone solver per element; not batched on GPU |
| JAXopt (Blondel+ 2022) | XLA-bound; mostly unconstrained / projected |
| Theseus (Pineda+ 2022) | CUDA; nonlinear *least squares*, not general NLP |

None is a **pure-Rust, cross-vendor, batched general-NLP** layer. That
is the defensible niche: not out-flopping cuDSS on one big factorization
(a race Julia/NVIDIA is winning), but being the differentiable NLP layer
that runs on **any** GPU and ships as one static binary.

## 8. Open questions for review

- **The f32 wall is the central risk — now measured on a CPU proxy
  (§10).** The JAX path today forces `jax_enable_x64` precisely because
  "Newton convergence stalls in float32" (`python/pounce/jax/__init__.py:36`).
  The §10 study confirms the stall is real (a plain f32 IPM cannot reach
  1e-8 and breaks down near the solution, worse as `n` grows) **but
  bounded**: f32 reaches 1e-6 on 100% of instances at every size, and a
  short f64 refinement (median 3 iterations) recovers 1e-8 on 100%. So
  the value proposition is **not** "GPU warm-starts rather than solves"
  — it is "GPU f32 solves to moderate accuracy, CPU f64 finishes
  cheaply," which keeps almost all the work on the GPU. The open part:
  the proxy is convex QP; pounce's nonconvex filter-line-search IPM (and
  worse conditioning) must still be measured for real.
- **Lockstep vs compaction.** Running `B` IPMs to the slowest element's
  iteration count wastes work on early-converging elements. Mask-only
  is simplest; periodic compaction recovers throughput at the cost of
  scatter/gather. Which, and at what `B`?
- **wgpu f32 vs CubeCL.** WGSL hand-written kernels (maximum
  portability, more code) vs CubeCL Rust-authored kernels (less code,
  multi-backend, newer). Decide at Phase 0.
- **Crate boundary.** New `pounce-gpu` crate vs an extension inside
  `pounce-py`'s JAX bridge. Recommendation: a standalone Rust crate
  with the batched core, exposed through the JAX layer — so it is
  usable from the Rust API too, not only from Python.
- **Default and gating.** Opt-in, never default; structure-gated
  (identical sparsity) and size-gated (dense-ish, small `n`). Auto-
  select when the layer detects a homogeneous batch above a width
  threshold, or only on explicit request?

## 9. Phase-0 preliminary evidence — f32 convergence (CPU proxy)

The numerical go/no-go question — *does an interior-point method
converge in f32, and to what accuracy?* — is independent of the GPU: the
GPU runs the same arithmetic, only elsewhere. It can therefore be
answered on CPU before any GPU code exists. The experiment
(`experiments/f32_qp_ipm_study.rs`, a dependency-free `rustc -O` program;
run with `rustc -O f32_qp_ipm_study.rs && ./f32_qp_ipm_study`) implements
a Mehrotra predictor-corrector solver for the canonical layer QP
`min ½xᵀQx + qᵀx s.t. Gx ≤ h`, generic over the float type, with the
condensed SPD Cholesky inner solve this note proposes for the GPU
kernel (§3). Each problem is solved in f64 and f32; accuracy is always
the **f64-recomputed** duality gap of the iterate, so we measure the
true accuracy of the f32 point, not f32's rounded self-assessment.
60 instances per size, well-conditioned `Q = (1/n)MᵀM + I`.

**Achievable accuracy (fraction of instances reaching a gap threshold):**

| size | prec | <1e-2 | <1e-4 | <1e-6 | <1e-8 | med best gap |
|---|---|---|---|---|---|---|
| 8×8 | f64 | 100% | 100% | 100% | 100% | 1.1e-9 |
| 8×8 | **f32** | 100% | 100% | **100%** | 65% | 6.1e-9 |
| 32×32 | f64 | 100% | 100% | 100% | 100% | 1.6e-9 |
| 32×32 | **f32** | 100% | 100% | **100%** | 17% | 2.4e-8 |
| 128×128 | f64 | 100% | 100% | 100% | 100% | 2.8e-9 |
| 128×128 | **f32** | 100% | 100% | **100%** | 0% | 1.1e-7 |

f32 reaches **1e-6 on 100% of instances at every size**, but the last
mile to 1e-8 collapses as `n` grows (65% → 0%) because the condensed
matrix's condition number scales like 1/μ and exceeds f32 epsilon near
the solution (the Cholesky loses positive-definiteness — a breakdown).

**Mixed-precision recovery (f32 forward → short f64 refinement):**

| size | f32 alone <1e-8 | f32→f64 hybrid <1e-8 | med f64 refine iters |
|---|---|---|---|
| 8×8 | 65% | **100%** | 3 |
| 32×32 | 17% | **100%** | 3 |
| 128×128 | 0% | **100%** | 3 |

Warm-starting an f64 pass from the f32 solution recovers full 1e-8
accuracy on **100% of instances at every size, in a median of 3 f64
iterations.** This is the §5 Phase-4 design validated: the GPU does the
bulk f32 work; a cheap CPU f64 tail cleans up.

**Caveats.** Convex QP, well-conditioned `Q`, plain Cholesky, no
inertia correction, no in-solve refinement. pounce's real IPM is a
nonconvex filter-line-search method where the condensed system is not
guaranteed SPD (needs inertia handling, likely harder in f32), and
worse conditioning lowers the f32 floor. The *qualitative* finding —
f32 to ≈1e-6, mixed-precision to recover — is robust and matches the
mixed-precision IPM literature; the f32 floor on pounce proper still
needs measuring on representative nonconvex layer problems.

## 10. References

- Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in
  Neural Networks," *ICML* 2017 — batched dense QP layer on GPU (qpth);
  the template for this note.
- Agrawal, Amos, Barratt, Boyd, Diamond & Kolter, "Differentiable
  Convex Optimization Layers," *NeurIPS* 2019 — cvxpylayers / diffcp.
- Blondel, Berthet, Cuturi, Frostig, Hoyer, Llinares-López, Pedregosa &
  Vert, "Efficient and Modular Implicit Differentiation," *NeurIPS*
  2022 — JAXopt.
- Pineda et al., "Theseus: A Library for Differentiable Nonlinear
  Optimization," *NeurIPS* 2022 — batched GPU nonlinear least squares.
- Shin, Pacaud & Anitescu, "Accelerating Optimal Power Flow with
  GPUs: SIMD abstraction of nonlinear programs and condensed-space
  interior-point methods," 2023 — MadNLP/ExaModels; the condensed-KKT
  GPU IPM reformulation reused in §3.
- `wgpu` (WebGPU implementation in Rust), WGSL compute; `CubeCL`
  (Tracel/Burn) Rust GPU-kernel language — the strict-pure-Rust stack
  options from Decision 1.

In-tree references: `python/pounce/jax/` (the layer + batching this
note extends); `dev-notes/research/interior-cg-matrix-free.md` (the
*other* GPU play — single large problem, the complement of this one).
