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
  **The numerical half of Phase 0 is already done on CPU — see §9 (a
  true-f32 proxy and a real-pounce f32-inner-solve run); it resolves
  the f32 risk in §8 with data. The harnesses themselves are built and
  runnable in `benchmarks/gpu_spike` (standalone crate, `wgpu` behind a
  `gpu` feature, runtime `--device cpu|gpu|both` A/B toggle): `baseline`
  (CPU throughput bar), `microbench` (GPU↔CPU crossover), `accuracy`
  (on-device f32 + run-to-run determinism + f64 tail). The CPU sides
  run anywhere; what remains is to run the GPU sides on an actual
  M-series (or other GPU) box to read the throughput crossover and the
  on-device f32/determinism numbers.**
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

- **The f32 wall is the central risk — now measured on CPU (§9).** The
  JAX path today forces `jax_enable_x64` precisely because "Newton
  convergence stalls in float32" (`python/pounce/jax/__init__.py:36`).
  The §9.1 true-f32 proxy confirms the stall is real (a plain f32 IPM
  cannot reach 1e-8 and breaks down near the solution, worse as `n`
  grows) **but bounded**: f32 reaches 1e-6 on 100% of instances at every
  size, and a short f64 refinement (median 3 iterations) recovers 1e-8
  on 100%. The §9.2 real-pounce run adds that the *algorithm* (inertia
  correction, scaling, filter) does not amplify f32 inexactness — the
  nonconvex case converges with ~±3% iteration noise. So the value
  proposition is **not** "GPU warm-starts rather than solves" — it is
  "GPU f32 solves to moderate accuracy, CPU f64 finishes cheaply." The
  open part (§9.2): the untested quadrant — many iterations *and* active
  inequality constraints driving μ→0 — still needs a real-pounce
  measurement (the `electrolyte` family is the natural probe).
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

## 9. Phase-0 preliminary evidence — f32 convergence

The numerical go/no-go question — *does an interior-point method
converge in f32, and to what accuracy?* — is independent of the GPU: the
GPU runs the same arithmetic, only elsewhere. It can therefore be
answered on CPU before any GPU code exists. Two experiments bracket the
answer: a standalone proxy that runs **true f32 arithmetic** on the
worst-case regime (§9.1), and real pounce with an f32-precision inner
solve on its actual nonconvex IPM (§9.2). The synthesis is in §9.3.

### 9.1 Standalone proxy — true f32 arithmetic, inequality QP

The experiment
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

This proxy is a bare Mehrotra solver (no inertia correction, no
scaling, no filter) on the *worst* regime for f32: inequality
constraints whose barrier drives `z/s → ∞`, pushing the condensed
matrix's condition number to ~1/μ. It isolates the f32 conditioning
wall; it says nothing about whether pounce's *algorithm* tolerates an
inexact step. §9.2 tests that.

### 9.2 Real pounce — f32-precision inner solve, nonconvex IPM

To test the actual solver, the FERAL adapter gained an env-gated f32
emulation (`POUNCE_FERAL_EMULATE_F32`, `crates/pounce-feral/src/lib.rs`,
default off): the KKT values are rounded to f32 before factoring (so
FERAL factors the f32-representable matrix and reads **inertia from f32
pivots**), the RHS is rounded to f32, f64 iterative refinement is
disabled, and the solution is rounded back to f32. The full nonconvex
filter-line-search IPM — inertia correction, scaling, filter,
restoration — then runs on top. Measured on the `large_scale` suite
(`tol=1e-8`), f64 baseline vs f32-emulated inner solve:

| problem | constraints | f64 iters | f32 iters | both reach 1e-8? |
|---|---|---|---|---|
| ChainedRosenbrock n=500 | unconstrained | 387 | 390 | yes |
| ChainedRosenbrock n=1000 | unconstrained | 765 | 734 | yes |
| ChainedRosenbrock n=2000 | unconstrained | 1484 | 1528 | yes |
| BratuProblem n=2000 | 1998 eq | 2 | 2 | yes |
| OptimalControl n=4001 | 2001 eq | 1 | 1 | yes |
| SparseQP n=3000 | 3000 ineq | 7 | 7 | yes |

Every problem still converges to the full 1e-8 tolerance; the nonconvex
case (Rosenbrock, 1500+ iterations of active inertia correction)
absorbs the f32 perturbation with only ~±3% iteration noise and an
identical objective. **pounce's robustness machinery does not amplify
f32-level inexactness in the step** — a real and reassuring finding the
bare proxy could not give.

Two honest limits keep this from being the whole story:

1. **Optimistic on arithmetic.** FERAL's factorization still runs in
   f64; only the *data* (matrix, RHS, solution) is f32. A real GPU f32
   kernel does f32 *arithmetic*, accumulating the error §9.1 exhibits.
   This bounds the good side; §9.1 bounds the binding side.
2. **Misses the worst regime.** The only long-running case here
   (Rosenbrock) is *unconstrained* — inertia correction keeps its KKT
   well-conditioned, so f32-data rounding is benign. The
   constraint-heavy problems converge in 1–7 iterations, never driving
   μ small enough to reach the §9.1 wall. The untested quadrant —
   **many iterations *and* active inequality constraints driving
   μ→0** — is exactly where f32 should bite hardest (e.g. the
   ill-conditioned `electrolyte` family).

### 9.3 Synthesis

The truth for a GPU f32 batched solve of a constrained layer NLP sits
between the two: it does true f32 arithmetic (§9.1's wall is real and
size-dependent) but benefits from pounce's inertia/scaling/filter
machinery (§9.2 shows that machinery is not fragile to inexact steps).
The binding constraint is §9.1's conditioning wall, and its mitigation
— **f32 forward to ≈1e-6, short f64 refinement to recover 1e-8** — is
validated. So the Phase-4 mixed-precision design is not a hedge; it is
the load-bearing element, and the §5 phasing should treat the f64
refinement tail as mandatory, not optional. The remaining measurement
(the §9.2 untested quadrant) is the next concrete experiment.

## 10. Determinism — the contract

A reviewer will ask whether f32/GPU makes solves nondeterministic —
"sometimes it solves, sometimes not." The answer hinges on a
distinction between two things that both get called nondeterminism, of
which only one is GPU-specific.

1. **Deterministic f32 inaccuracy.** f32 has fewer bits, so it computes
   a less accurate answer — but for a *fixed* computation it is the
   *same* less-accurate answer every run. This is what every §9
   experiment measured (`rustc -O` f32 and the f32-emulated FERAL path
   are both fully deterministic). It produces an accuracy *ceiling*
   (≈1e-6, §9.1), not run-to-run variation. It does **not** mean
   "sometimes solves, sometimes not."
2. **Run-to-run GPU reduction reordering.** Parallel reductions (dot
   products, matrix products) and atomic accumulation sum in an order
   set by GPU scheduling, and f32 addition is non-associative — so the
   *same* problem on the *same* device can give bitwise-different
   results across runs. This is the only source that can make a solve
   nondeterministic.

**Can (2) flip the outcome? Mechanically yes, but bounded.** The IPM
has discrete branches — the inertia count drives the perturbation
handler, the filter accepts/rejects, fraction-to-boundary and
active-set membership gate the step. Near a decision boundary a
rounding-level run-to-run wobble can flip a branch and send two runs
down different iteration paths, so in principle one run could converge
where another stalls or invokes restoration. Three facts bound this:

- **The machinery self-corrects.** §9.2 showed pounce absorbs
  f32-magnitude perturbation with ±3% iteration noise and no
  convergence failures: a flipped inertia decision just bumps `δ_w` and
  refactors; a rejected filter step backtracks. Noise changes the
  *path and iteration count*, not the *outcome*. (That test perturbs
  deterministically, so it bounds robustness to the *magnitude*; the
  re-correction works regardless of the noise's direction.)
- **The delivered result is f64-refined, hence deterministic.** The
  Phase-4 f64 tail converges to the same f64 solution no matter which
  f32 trajectory reached its neighbourhood. The nondeterminism lives
  entirely in the throwaway f32 warm-start; the **output is
  reproducible to f64 tolerance.** Path-nondeterministic,
  result-deterministic — that is the contract.
- **It can be removed at the source.** Deterministic reduction kernels
  (fixed reduction trees, no `atomicAdd` for sums) give bitwise-
  reproducible GPU results at some throughput cost. Nondeterminism is a
  *choice* here, not forced.

**Where it is a genuine concern.** The ill-conditioned / μ→0 /
active-inequality quadrant (the §9.2 untested one) has genuinely
near-degenerate decision boundaries — near-active constraints,
near-singular inertia — so run-to-run f32 noise is most likely to flip
outcomes *there specifically*. That is also where f32 is weakest, and
the reason the GPU path is conditioning-gated and leans on f64
refinement.

**Cultural note.** pounce deliberately keeps its CPU BLAS scalar and
SIMD-free *for bit-reproducibility* (§3). A GPU f32 path runs against
that grain, so it is explicitly a separate, opt-in, non-bit-reproducible
path whose guarantee is "deterministic *result* via f64 refinement," not
"deterministic *trajectory*." It must never be the default, and the
returned solution must always be f64-refined.

## 11. Portability & runtime selection

The GPU path is one codebase with one f32 kernel; it **falls back to
plain CPU where no usable, eligible GPU exists, and uses the GPU when
one does** — from a single binary. Selection happens in three
independent layers.

1. **Compile-time (cargo feature).** wgpu lives behind a `gpu` feature.
   The default `cargo build` is the existing pure-Rust CPU solver with
   **no wgpu dependency** — the pure-Rust story is untouched.
   `--features gpu` compiles the backend in. Whether the binary contains
   GPU code at all is opt-in.
2. **Runtime adapter detection (probe-and-verify).** Even with the
   feature on, wgpu enumerates adapters at startup; none — or init
   failure — routes to CPU. It must *run a tiny known kernel and fall
   back on error*, not merely check that an adapter is listed: some
   Linux/Vulkan setups enumerate a flaky adapter. "Use a GPU if it
   exists" means "if it exists *and* actually works."
3. **Per-call workload eligibility.** Even with a working GPU, route to
   it only for a **batch** of structure-identical, size/conditioning-
   eligible problems (§2–§3). A single small solve always stays on CPU.
   It is "GPU if present *and* the workload suits it," not "GPU for
   everything."

**One kernel, many backends.** wgpu runs the same WGSL on Metal
(macOS/iOS), Vulkan (Linux/Windows/Android), DX12 (Windows), and
WebGPU. Because f32 is WGSL's native type there is **no per-platform
kernel fork** (an f64 kernel would have fractured by backend). So
*correctness* is broadly portable; *performance* is not uniform —
backend maturity and limits (max workgroup size, shared-memory budget)
differ, so the batched-Cholesky tiling may need per-backend tuning.
"Runs everywhere" ≠ "fast everywhere"; that is a measure-and-tune item,
not a correctness risk.

**Cross-platform result consistency comes from the f64 tail.** Different
vendors/backends reduce and contract (FMA) differently, so the f32
forward solve is platform-dependent at the bit level. The f64 CPU
refinement tail (§5 Phase 4) makes the *final* answer agree to f64
tolerance regardless of which GPU/vendor/backend produced the
warm-start — so §10's determinism contract is also the *cross-platform*
reproducibility mechanism. Without the tail the same problem would
answer subtly differently on a Mac vs a Linux+NVIDIA box; with it they
agree. One more reason the tail is mandatory.

**Consequences for testing and shipping.** The GPU path stays opt-in
and non-default, so any platform where it is untested or flaky simply
leaves it off and nothing regresses. The CPU path is always the
reference oracle: GPU results are validated against it within tolerance
where hardware exists, and GPU-less CI still exercises the CPU path
fully. One feature-on binary detects and adapts at runtime; no
per-platform builds.

## 12. References

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
