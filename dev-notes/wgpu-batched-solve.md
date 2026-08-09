# A wgpu module for POUNCE — throughput, not FLOPs

**Status: assessment / not started.** Written to answer one question: does a
`wgpu` backend earn its keep in POUNCE if the goal is *many concurrent small
solves* rather than *one fast dense factorization*? Short answer: yes, but
only for a narrow, well-defined slice — a fixed-structure, branch-free,
**fp32** convex QP/SOCP solver batched across problem instances — and the
go/no-go should be decided by a measurement we have not yet taken.

## 1. The right framing

The instinct to reject "POUNCE on GPU" is correct *for the usual reason people
propose it*. POUNCE's hot path is a sparse symmetric indefinite LDLᵀ (feral)
with AMD ordering, symbolic analysis, supernodal factorization, dynamic
regularization, and inertia correction. That is irregular, data-dependent,
branch-heavy work. It is the worst possible fit for a compute shader, and the
GPU literature on sparse-direct linear algebra exists precisely because the
problem is hard. Nothing in a `wgpu` module is going to beat feral at feral's
job.

The question here is different. If each individual problem is *small* — an MPC
horizon, a CBF safety filter, an IK step — then the parallelism does not live
inside the factorization at all. It lives in the **batch dimension**. One
workgroup owns one problem; the arithmetic per problem is tiny; the win comes
from having 4096 of them in flight. This is the same reason GPUs are good at
particle systems and bad at graph traversal.

That regime is real, and it is where control lives.

## 2. Two literature anchors

**Arrizabalaga, Tracy & Manchester, *A Differentiable Interior-Point Method in
Single Precision*** ([arXiv:2605.17913](https://arxiv.org/abs/2605.17913),
code: [qpax-solver/qpax](https://github.com/qpax-solver/qpax)).

The observation: the standard primal-dual complementarity treatment makes the
Newton system's conditioning blow up like `1/μ` as the iterates approach the
solution. In fp64 (eps ≈ 2.2e-16) that is survivable down to `μ ≈ 1e-10`. In
fp32 (eps ≈ 1.2e-7) it is fatal well before you reach a useful tolerance — you
get arithmetic exceptions and, worse, *silently wrong gradients*, because the
implicit-function-theorem backsolve reuses the same near-singular matrix. The
paper replaces the complementarity representation with one whose linear systems
stay **spectrally bounded** all the way to the solution, which makes both the
solve and the derivative reliable in fp32.

This is not adjacent literature. It is the **enabling prerequisite** — see §3.

**Viljoen, Haffner, Tomizuka & Mehr, *Scaling Nonlinear Optimization: Many
Problems One GPU*** ([arXiv:2606.26341](https://arxiv.org/abs/2606.26341)).

Introduces `jaxipm`, a GPU-batched NLP solver built on the Ipopt algorithm in
JAX. The stated motivation is exactly the framing in §1: robotics NLPs (traj-opt,
IK, contact-rich planning) are individually small, and IPOPT is CPU-bound and
one-problem-at-a-time, so it cannot sit inside a GPU-batched learning pipeline.
Two contributions matter to us:

- **Heterogeneous iteration fusion** — removing control flow so that a batch of
  problems with *different* iteration counts and *different* line-search
  decisions can execute as one branch-free kernel.
- **Iteration-level batching** — minimizing GPU idle time from the long tail of
  slow-converging instances.

Reported: up to **32.85×** throughput over IPOPT on quadrotor NMPC. Their
linear algebra goes through `cuDSS` via
[`spineax`](https://github.com/johnviljoen/spineax), which is NVIDIA Turing+,
CUDA 13, Linux x86-64 only.

Note what those two contributions *are*: both are about the fact that an IPM's
control flow is data-dependent and a GPU hates that. That is the actual
engineering content of GPU-batched IPM, and it is not optional.

## 3. What choosing `wgpu` specifically forces on the design

`wgpu` is the right call for POUNCE's identity — pure Rust, no vendor SDK, no
system deps, and it runs on Metal / Vulkan / DX12 *and* WebGPU. But it is not a
free swap for CUDA. Three constraints are load-bearing:

**(a) There is no f64. At all.** WGSL has `f32` and (optionally) `f16`; the
`AbstractFloat` binary64 type exists only for shader-creation-time constant
evaluation and cannot be spelled in source. `f64` is
[a live spec issue](https://github.com/gpuweb/gpuweb/issues/2805) with only an
unofficial `naga_ext_f64` extension behind it, and Metal has no fp64 in the
language *or* the hardware — so on Apple silicon it is not emulatable at
acceptable cost, ever.

To be precise about what this is and is not: "GPUs are fp32" is **false** as a
hardware claim. The fp64:fp32 ratio is market segmentation, not physics —
NVIDIA A100/H100 are 1:2, AMD MI250X is 1:1 (vector), MI300X is 1:2 vector /
1:1 matrix. Exascale HPC runs fp64 on GPUs routinely. What is true is narrower:
consumer NVIDIA is deliberately crippled to 1:64, Apple silicon has no fp64 at
all, and WebGPU spec'd it out because the spec must hold on the weakest
conforming target (mobile Mali/Adreno, Apple).

So the real trade is **fp64 or portability, not both** — and it is a genuine
fork, not a constraint handed to us. Targeting CUDA on a datacenter part would
give fp64 at half rate and make Gate 1 (§8) unnecessary entirely; that is
effectively what `jaxipm` does, which is why it has no single-precision research
problem and can lean on cuDSS. Choosing `wgpu` buys Metal + AMD + Intel + the
browser and pays for it with an open numerical problem.

Two things argue for fp32 even on hardware that has fast fp64. First, the
binding constraint in this design is **shared memory, not FLOPs** (see (c)
below): 16 KB is 4096 `f32` but only 2048 `f64`, so fp64 halves the problem size
per workgroup and halves occupancy — for batched-tiny-problems that costs more
than the 2× arithmetic rate buys. Second, the trend runs the other way: Blackwell
deliberately trades FP64 silicon for FP4/FP8/BF16 tensor throughput, and B300's
FP64 is *lower than H100's*. Betting a solver on GPU fp64 throughput is betting
against the roadmap — which is exactly the Arrizabalaga framing, that IPMs cannot
exploit "the accelerated hardware that underpins modern machine learning" because
that hardware is moving to lower precision, not higher.

Consequence: **a wgpu module is an fp32 module, unconditionally.** Which makes
Arrizabalaga et al. a hard dependency rather than a nice citation. Without that
reformulation the kernel will converge to ~1e-3/1e-4 on easy problems, stall or
NaN on the rest, and produce sensitivities that are quietly garbage. Any plan
that treats "port to fp32" as a mechanical type substitution is a plan that
fails at the end, expensively.

**(a′) The fork may be a type parameter, not an architecture decision.** Rust
has three ordinary routes to CUDA — `cudarc` (safe wrappers over the driver API
plus cuBLAS/cuSOLVER/cuSPARSE/NVRTC; mature, used by `candle` and `dfdx`;
**no cuDSS bindings**), `rustc_codegen_nvvm`
([Rust CUDA](https://rust-gpu.github.io/rust-cuda/), rebooted 2025, kernels in
Rust but pinned to an exact nightly because it hooks rustc internals), and
plain nvcc/NVRTC FFI. None of them resolve the portability trade.

[CubeCL](https://github.com/tracel-ai/cubecl) does. It is a Rust compute DSL
(`#[cube]`) that JIT-compiles one kernel to CUDA, ROCm/HIP, Metal, Vulkan,
WebGPU, *and* CPU-SIMD, and it is the compute backend under Burn. `f64` is
supported on its CUDA backend and structurally cannot be on its wgpu backends
(WGSL has none). So a kernel written **generic over the float type** gives
`f64` on CUDA/ROCm and `f32` on Metal/Vulkan/WebGPU from one source.

That reorders the risk in §8 substantially: land the **CUDA fp64 path first**
— no numerical research risk at all — prove the batching design beats the rayon
baseline, and only then take on Gate 1's single-precision reformulation for the
portable path. It also yields an fp64 reference implementation in the same
language to validate the fp32 kernel against, and lets Gate 0's CPU baseline
share source with the kernel via the CPU-SIMD backend.

Costs: CubeCL is **alpha** (its README warns of breaking changes between minor
versions and recommends pinning); unsupported instructions fail at *runtime*,
not compile time, so per-backend CI coverage is mandatory; and it is a young
dependency for a core numerical component versus `wgpu`, which ships in Firefox.
Recommendation: prototype Gate 2 in CubeCL. If the alpha status bites, dropping
to raw `wgpu` loses only the DSL layer.

**(b) There is no `cuDSS`, and there will not be one.** `jaxipm` gets to call a
vendor sparse-direct solver. Under `wgpu` the options are: write a supernodal
sparse LDLᵀ in WGSL (nobody sane does this), or **do not use a sparse solver**.
The second option is fine, and is in fact the correct design — see §4. Note this
cuts both ways and mostly in our favour: `cudarc` has no cuDSS bindings either,
but because the §4 kernel needs no vendor sparse solver at all, that gap is
irrelevant to us. Avoiding vendor libraries is precisely what lets one kernel
run on all six CubeCL targets.

**(c) The portable resource limits are tight.** The defaults we must design
against, not the ones a desktop Vulkan backend happens to report:

| limit | default | what it buys |
|---|---|---|
| `maxComputeWorkgroupStorageSize` | 16384 B | 4096 `f32` of shared memory per problem |
| `maxComputeInvocationsPerWorkgroup` | 256 | ≤256 threads cooperating on one problem |
| `maxComputeWorkgroupsPerDimension` | 65535 | batch size per dispatch dimension |

Plus: WGSL atomics are `i32`/`u32` only — [no float atomics](https://github.com/gpuweb/gpuweb/issues/4894) — so
reductions are tree reductions in shared memory, not atomic accumulation.
Subgroups are available but optional, so any subgroup fast path needs a
workgroup-only fallback.

4096 `f32` of shared memory is the number that shapes everything. A dense
symmetric KKT of dimension `N` stored as a full square needs `N²` floats:
**N ≤ 64**. Packing the triangle gets you to ~90. Beyond that you spill to
global memory and lose most of the point.

## 4. The design that survives §3

Drop the sparse solver. For the target regime the structure is known at compile
time and the right factorization is **dense-in-shared-memory, or block-banded
via Riccati** — which is what HPIPM does on CPU for exactly these problems, for
exactly this reason.

Concretely, for linear-quadratic MPC with state dim `n_x`, input dim `n_u`,
horizon `H`:

- **Condensed** (eliminate states, decision var is the input sequence): KKT
  dimension `H·n_u`. A quadrotor at `n_u=4, H=20` gives `N=80` → 6400 floats →
  25.6 KB. Over the portable limit; workable on desktop backends, not in a
  browser.
- **Block-tridiagonal / Riccati** (keep the stage structure): the live working
  set is a handful of `(n_x+n_u)²` blocks. A quadrotor at `n_x=12, n_u=4` gives
  16×16 = 256 floats per block. Fits with two orders of magnitude to spare, and
  the cost is `O(H·(n_x+n_u)³)` rather than `O((H·n_u)³)`.

The Riccati route is the one to build. It fits the portable limit, it scales
linearly in horizon, and `pounce-qp/src/schur.rs` is already the CPU-side home
for structure-exploiting KKT work.

The rest of the kernel follows the `jaxipm` playbook, because it has to: fixed
iteration count (or fused heterogeneous iterations), no filter, no restoration
phase, no dynamic inertia correction, static regularization only. Every one of
those omissions is a piece of POUNCE's robustness we are choosing to give up in
exchange for branch-free execution. That is the actual trade, stated plainly.

**Scope estimate:** ~1500–2500 lines of WGSL plus Rust host code, in a new
`pounce-wgpu` crate. It reuses essentially none of `pounce-convex`'s IPM. It is
a **second numerical core**, not a backend behind the existing
`SparseSymLinearSolverInterface` trait. That is the honest cost line.

## 5. The baseline that matters (and why the 32.85× does not transfer)

This is the most important section in the note.

`jaxipm`'s headline is 32.85× **over IPOPT** — that is, over a single-threaded,
one-problem-at-a-time CPU solver. POUNCE is not that baseline. POUNCE already
has:

- `pounce_convex::batch::solve_qp_batch_parallel` — rayon across instances with
  an inner-**serial** feral backend, the outer-parallel/inner-serial model the
  module docs already work out in detail
  (`crates/pounce-convex/src/batch.rs:20-45`);
- `QpFactorization` — build the KKT symbolic factor once, reuse across a
  fixed-structure batch, skipping repeated AMD ordering;
- warm-started batch entry points for receding-horizon sequences;
- `python/pounce/jax/_problem.py` — `vmap_solve`, `vmap_solve_parallel`,
  `batched_solve`, `batched_solve_with_warm`, `custom_vjp`, and a batched KKT
  backsolve for the implicit-function-theorem gradient.

On a 32- or 64-core machine that existing path is *already* getting a large
multiple over serial IPOPT on small QPs. So the number a `pounce-wgpu` module
has to beat is **not** 32.85×; it is the ratio of GPU throughput to POUNCE's own
parallel CPU batch throughput, and that ratio is plausibly 2–5×, not 30×. It
could also be below 1× for small batches, where kernel launch and PCIe transfer
dominate.

**We have never measured our own batch throughput on representative MPC QPs.**
Until we do, every estimate in this note including that one is speculation. That
measurement is the gate.

There is one place the CPU baseline genuinely cannot follow, and it is not about
FLOPs: `python/pounce/jax/_diff.py:_solve_batch_threadpool` reaches the CPU via
`jax.pure_callback` into a `ThreadPoolExecutor`. Inside a JAX training loop that
is a device→host→device round trip and a hard barrier — it breaks `jit`,
breaks device residency, and serializes against the rest of the pipeline. A
`wgpu` kernel writing into a buffer the ML framework already owns removes the
round trip entirely. **That, not raw throughput, is the defensible argument**,
and it is precisely the gap `jaxipm` was written to close.

## 6. Where POUNCE is genuinely differentiated

Two things are true and worth stating, because they are the reasons this is not
merely reimplementing `jaxipm`:

1. **Portability.** `jaxipm`/`spineax` is NVIDIA Turing+, CUDA 13, Linux
   x86-64. A `wgpu` batched differentiable QP solver runs on Apple silicon, AMD,
   Intel, and NVIDIA, with no vendor SDK and no Fortran — consistent with the
   pure-Rust promise. There is no portable batched NLP/QP solver today. That is
   a real gap.
2. **The browser.** POUNCE already compiles to WebAssembly and ships a live
   demo. `wgpu` targets WebGPU. A batched GPU QP solver running client-side in a
   tab is something no other solver in this space can do, and it makes the
   existing demo qualitatively more interesting rather than incrementally
   faster.

Against that: fp32 caps achievable tolerance at roughly 1e-4–1e-6. That is
comfortably enough for MPC, IK, and safety filters. It is **not** enough for
process optimization, parameter estimation, or the CUTEst/Mittelmann suites that
constitute most of POUNCE's current audience and all of its benchmark identity.
The module serves a new audience; it does not upgrade the existing one.

CI is a solvable problem, not a blocker: `wgpu` on Vulkan via `lavapipe`
(software rasterizer) runs on GitHub Actions runners, so the kernel is testable
without GPU hardware in CI, with correctness cross-checked against the fp64 CPU
solver.

## 7. The other application: POUNCE as a training-data generator

A second use case surfaced while writing this, and it may be the higher-value
one. Instead of putting the solver *inside* the learning loop, use POUNCE to
manufacture the dataset the model trains on.

**The analogy is exact, and it is the DFT one.** An MLIP is not trained on
energies alone; it is trained on energies *and forces*. Forces are why MLIPs
work at the sample efficiency they do: one DFT calculation yields `1 + 3N`
labels instead of `1`, and the extra `3N` come nearly free because of
Hellmann–Feynman — once you have the converged wavefunction, the gradient is a
cheap contraction, not a second calculation.

POUNCE has the same structure:

| DFT / MLIP | POUNCE / learned optimizer |
|---|---|
| configuration `R` | problem parameter `p` (state, reference, cost weights, plant params) |
| energy `E(R)` | optimal value `V(p) = f(x*(p))`, or solution `x*(p)` |
| forces `−∂E/∂R` | sensitivity `∂x*/∂p`, `∂V/∂p` |
| Hellmann–Feynman: forces ≈ free given converged SCF | sIPOPT: sensitivity reuses the **converged KKT factorization** |
| Sobolev/force training → huge sample-efficiency gain | same theorem, same gain |

That third row is not a loose analogy — `docs/src/sensitivity.md` says it
outright: the sensitivity step *"reuses the KKT factorization from the converged
solve."* One solve buys `n × n_p` derivative labels for the cost of a backsolve.
And `∂V/∂p` is cheaper still: by the envelope theorem it is just the Lagrangian's
partial derivative at the solution, no linear system at all — the *exact*
Hellmann–Feynman situation.

The theoretical backing is Czarnecki et al., *Sobolev Training for Neural
Networks* (NeurIPS 2017): matching derivatives as well as values gives markedly
better generalization per sample. The MLIP community rediscovered this
empirically and it is now non-negotiable practice there. Nobody has
systematically applied it to amortized optimization, and POUNCE is unusually
well-equipped to — `pounce-sensitivity` is a validated sIPOPT port (matched to
upstream to 1e-8), and `python/pounce/qp.py` and `jax/_problem.py` already
expose sensitivities and reduced Hessians from Python.

### 7.1 Candidate applications

Roughly in order of how well the pieces already fit.

**(a) Learned value functions / terminal costs for short-horizon MPC.** The
cleanest instance of the analogy. Solve the long-horizon problem offline over
sampled initial states `p`; label with `V(p)` *and* `∂V/∂p`; fit `V̂`; deploy it
as the terminal cost of a much shorter online horizon. This is the standard
technique for making MPC real-time-feasible, and `∂V/∂p` is exactly the quantity
that determines closed-loop behavior — so training on it, rather than on `V`
alone, is directly aimed at the thing you care about. Envelope theorem makes the
labels nearly free.

**(b) Amortized / learned explicit MPC.** Fit `π̂(p) ≈ x*(p)` and train
`∂π̂/∂p` against `∂x*/∂p`. This is the learned successor to classical explicit
MPC (multi-parametric programming), which is exact but blows up combinatorially
in the number of critical regions. Sensitivity labels are informative here in a
specific way: `∂x*/∂p` is piecewise-constant-ish within an active set and jumps
at active-set changes, so the derivative labels effectively teach the network
where the critical-region boundaries are — information the value labels convey
only weakly.

**(c) Warm-start and active-set prediction.** Lower-ambition, higher-certainty.
Learn `p ↦` (good initial iterate, predicted active set) and hand it to POUNCE's
existing warm-start machinery, which is already built out
(`docs/src/active-set-sqp-warm-start.md`). The model does not have to be right,
only close — POUNCE still certifies. This is the variant with no correctness
risk, and it composes with (a)/(b) rather than competing.

**(d) Thermodynamic equilibrium surrogates.** *(expanded in
`dev-notes/sobolev-surrogates.md`, along with the training methodology for all
of §7.)* Constrained Gibbs free-energy
minimization gives equilibrium composition `x*(T, P, z)`; the sensitivities
`∂x*/∂T`, `∂x*/∂P`, `∂x*/∂z` are physically meaningful (they connect to reaction
enthalpy and heat capacity), which means the "forces" are quantities the domain
already understands and can validate against. Flash calculations sit in the
inner loop of every process simulation and are re-solved millions of times, so
the amortization payoff is direct. Closest to the DFT analogy in both spirit and
subject matter, and it exercises the NLP path rather than just the QP path.

**(e) Real-time optimization / process control surrogates.** Same shape as (b)
one level up: a plant-wide RTO problem solved on a slow cycle, surrogated so it
can advise on a fast one, with sensitivities giving the local gain matrix the
downstream controller wants anyway.

**(f) Optimal power flow surrogates.** Large and active literature on learning
OPF solutions from load profiles. The dual sensitivities are locational marginal
prices — the labels have direct economic meaning. Well-defined public
benchmarks, so it is the easiest to publish against; correspondingly the most
crowded.

**(g) Learned safety filters.** CBF-QP safety filters run at kHz rates and are
structurally tiny — the ideal amortization target, and the ideal batch-generation
target. Also the case where being wrong is most consequential, which argues for
the (c) posture: predict, then verify with a real solve.

### 7.2 Why this reframes the wgpu question

Dataset generation is **offline, embarrassingly parallel, throughput-bound, and
latency-insensitive**. Compare that to §5's requirements: no need for device
residency, no need to be inside a `jit` trace, no need to differentiate *through*
the solver, and a long tail of slow instances costs wall-clock but breaks
nothing. It is a strictly easier target that exercises the same kernel.

But it also *weakens* the GPU argument, and the reason is precision. Offline
generation is where you want fp64 sensitivities, because label noise in the
"forces" is exactly what Sobolev training is most sensitive to — and there is no
latency pressure forcing you to accept fp32. Which suggests a clean split:

- **Offline data generation → CPU, fp64**, via the existing
  `solve_qp_batch_parallel` + `pounce-sensitivity`. Available today.
- **Online in-the-loop batched solving → GPU, fp32**, with the Arrizabalaga
  reformulation. The thing that needs building.

The strategic consequence: **the entire training-data research direction can be
started now, on the CPU, with zero new numerics.** Build the dataset tooling, the
Sobolev-training recipe, and one convincing application end-to-end. If it works,
generation throughput becomes the measured bottleneck and the GPU case argues
itself with real numbers. If it does not work, we learn that for the cost of some
Python instead of the cost of a second numerical core.

## 8. Verdict and staged plan

A `wgpu` module has real value, for the throughput/portability/browser reasons in
§6 — but it is a new solver serving a new audience, not an accelerator for the
existing one, and the case for it is currently unmeasured. Sequence the cheap
disqualifying experiments first.

**Gate 0 — measure the baseline (days).** Benchmark
`solve_qp_batch_parallel` on representative MPC QPs (quadrotor: `n_x=12, n_u=4,
H=20`; batches of 64 / 1k / 16k) with symbolic-factor reuse and warm starts on.
This produces the number a GPU has to beat. *Kill if* CPU throughput already
covers the plausible application demand.

**Ordering note.** If Gate 2 is built in CubeCL (§3 a′), Gates 1 and 2 can swap:
a CUDA **fp64** kernel carries no numerical research risk, so it can validate the
whole batching design against Gate 0's baseline *before* any single-precision
work is funded. Gate 1 then becomes the portability step rather than the
feasibility step. Prefer this ordering unless the browser/Apple targets are the
primary motivation, in which case Gate 1 remains the gate.

**Gate 1 — fp32 on the CPU (1–2 weeks).** Implement the Arrizabalaga
complementarity reformulation in `pounce-convex` and run the IPM in fp32 *on the
CPU*, checking both solution and sensitivity accuracy against the fp64 path.
Prerequisite: `pounce-convex` uses raw `f64` ~1229× against
`pounce_common::types::Number` 33×, so introducing a crate-local scalar alias is
step zero. **This is the highest-value experiment in the whole plan.** It is
pure Rust, needs no GPU, and if POUNCE's IPM cannot hold fp32 on the CPU for
these problems, the GPU port is dead before a line of WGSL is written. It is
also independently publishable.

**Gate 2 — one kernel, one shape (4–6 weeks).** `pounce-wgpu` with a
Riccati/block-tridiagonal fp32 QP kernel for one fixed MPC shape, fixed iteration
count, correctness checked against the fp64 CPU solver, CI on `lavapipe`. Compare
against Gate 0's number. *Kill if* under ~3× on realistic batch sizes.

**Gate 3 — productionize.** Heterogeneous iteration fusion, the fp32 sensitivity
backsolve on-device, wiring under the existing `batched_solve` / `custom_vjp`
API so it is a backend swap rather than a new user-facing surface, and the WebGPU
demo.

Run §7's Gate 0′ — CPU-side Sobolev-training dataset generation for one
application — **in parallel with Gate 0/1**. It is independent, cheap, uses only
shipped functionality, and its outcome informs whether Gates 2–3 are worth
funding.

## 9. Packaging: in-tree crate, or a downstream application?

Both use cases above are better built as a **separate repo depending on
published pounce crates** than as a 21st workspace member. The reasoning is
specific, not stylistic.

**The dependency is clean here, unlike discopt.**
`dev-notes/discopt-pounce-integration.md` argues the opposite case for spatial
B&B: a generic plugin boundary makes discopt "a dispatcher," and the leverage
comes from warm state, certificates, and μ flowing *through the tree*. That
argument does not transfer. B&B wants state to flow node-to-node; a batch is
independent by construction. What this application needs from pounce is problem
data structures, `pounce-sensitivity`, and the fp64 solver as a correctness
oracle — an API boundary, not a lossy serialization boundary.

**Release lockstep.** Per `CLAUDE.md`, pounce publishes 20 crates in topological
order behind a guard that fails unless all versions agree, gating two PyPI
packages as well. Putting **CubeCL — self-described alpha, breaking changes
between minor versions** — inside that lockstep taxes every future release for
the sake of an experiment. `publish = false` and the long-lived
`feature/global` branch are the existing escape hatches; neither is pleasant.

**Maturity contract.** The README promises "production-ready for the core IPM
workflow." An fp32 GPU kernel is research-grade and serves a different audience
(control/RL, not process optimization). Keeping it out of the workspace keeps
both promises honest.

**Frame the app around §7, not around the kernel.** A repo whose only claim on
pounce is "we share a `QpProblem`" is thin — §4's kernel reuses none of
`pounce-convex`'s IPM. But the amortized-optimization layer of §7 depends on
pounce deeply and legitimately (batch solving, sIPOPT sensitivities, warm
starts, fp64 ground truth), needs **zero new numerics**, and is buildable
against pounce 0.9 today. Under that framing the GPU is a throughput
optimization *inside* the application, adopted iff Gate 0 says generation is the
bottleneck — so the app has a reason to exist whether or not the GPU bet pays
off. Sketch:

```
pounce-amortize/          (separate repo, deps from crates.io)
  ├── dataset/   generate (p, x*, ∂x*/∂p, V, ∂V/∂p) — CPU fp64, today
  ├── train/     Sobolev training recipes (JAX / PyTorch)
  ├── verify/    predict-then-certify: model proposes, pounce checks
  └── gpu/       CubeCL kernel — added later, gated on Gate 0/2
```

**The payoff is an API-surface test.** `pounce-rs` is the curated facade —
"re-exports only… pins a single curated public surface, so downstream code is
insulated from churn in the internal crate layout" — but it covers
`pounce-common`/`-nlp`/`-algorithm`/`-observability`, i.e. **the NLP path only**.
There is no `pounce-convex`, no `pounce-sensitivity`, no `pounce-qp`. A
downstream batched-QP-plus-sensitivity consumer must therefore reach past the
facade into precisely the internal crates it exists to hide.

That gap is the finding. Building out-of-tree forces the convex/QP + sensitivity
+ batch facade to be designed deliberately, by a real consumer, instead of being
assumed. Whatever exports turn out to be missing are a genuine improvement to
pounce; and if the application *cannot* be built at arm's length, that is a
cheap, early result about how modular the workspace actually is.

## References

- Arrizabalaga, J., Tracy, K., Manchester, Z. *A Differentiable Interior-Point
  Method in Single Precision.* [arXiv:2605.17913](https://arxiv.org/abs/2605.17913).
  Code: [qpax-solver/qpax](https://github.com/qpax-solver/qpax).
- Viljoen, J., Haffner, J., Tomizuka, M., Mehr, N. *Scaling Nonlinear
  Optimization: Many Problems One GPU.*
  [arXiv:2606.26341](https://arxiv.org/abs/2606.26341). Sparse-solver layer:
  [johnviljoen/spineax](https://github.com/johnviljoen/spineax).
- Czarnecki, W. M., et al. *Sobolev Training for Neural Networks.* NeurIPS 2017.
- Pirnay, H., López-Negrete, R., Biegler, L. T. *Optimal sensitivity based on
  IPOPT.* Math. Prog. Comp. 4(4), 307–331 (2012).
  [DOI: 10.1007/s12532-012-0043-2](https://doi.org/10.1007/s12532-012-0043-2).
  (The basis of `pounce-sensitivity`.)
- Frison, G., Diehl, M. *HPIPM: a high-performance quadratic programming
  framework for model predictive control.* IFAC 2020. (Riccati-structured
  KKT; the CPU precedent for §4.)
- WebGPU f64: [gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805).
  WGSL float atomics: [gpuweb#4894](https://github.com/gpuweb/gpuweb/issues/4894).
- Rust GPU tooling: [CubeCL](https://github.com/tracel-ai/cubecl) (multi-target
  `#[cube]` DSL; Burn's compute backend),
  [`cudarc`](https://crates.io/crates/cudarc) (driver + cuBLAS/cuSOLVER/cuSPARSE;
  no cuDSS), [Rust CUDA](https://rust-gpu.github.io/rust-cuda/)
  (`rustc_codegen_nvvm`, kernels in Rust, nightly-pinned).

## Related notes

- `dev-notes/socp-extension.md` — the cone abstraction a GPU kernel would have
  to mirror if it ever goes past the orthant.
- `dev-notes/performance-engineering.md` — CPU-side performance methodology.
- `docs/src/differentiable-solves.md`, `docs/src/sensitivity.md` — the existing
  differentiable/sensitivity surfaces both use cases build on.
