# Design note — Interior/CG: matrix-free interior point for large-scale NLP

**Status: design / proposed. Not yet implemented.** This note is the
research → plan half of a research → plan → implement workflow; it is
written for review before any code lands. It is the third design note
under `future-work-roadmap.md`, covering challenge area **C5**.

## 1. What this is

Interior/CG computes its step by an *iterative* (Krylov) solve of the
primal-dual system instead of a *direct* sparse factorization. It is
KNITRO's `algorithm=interior/cg` and the lineage of Ipopt's inexact
algorithm (`ref/Ipopt/src/Algorithm/Inexact/`). The target: problems
where the KKT matrix is too large to factorize, has dense rows that
wreck sparsity, or is only available as operators — PDE-constrained
optimization, large-scale OPF, inverse problems.

It is **not a standalone algorithm.** It is the composite-step method
(`composite-step-byrd-omojokun.md`) with two changes:

1. the normal and tangential subproblems are solved iteratively by
   CG, not by an `AugSystemSolver` factorization;
2. the CG iterations are stopped *early* (inexactly), and
   inexact-Newton termination tests certify that the inexact step
   still yields sufficient progress.

So Interior/CG **depends on the composite-step note's Phases 2–3
landing first.** This note covers only the delta: matrix-free linear
algebra plus inexact termination. In one line — composite-step gives
the *step decomposition*; Interior/CG gives the means to compute it
*without a factorization*.

## 2. Why a separate path (the architectural fact)

pounce solves the augmented KKT system by direct sparse factorization
— `AugSystemSolver` with FERAL / MA57 backends, plus
`PerturbationHandler` inertia correction. Direct factorization is
robust (exact step, exact inertia) — it is why pounce ≈ KNITRO
Interior/Direct — but its cost is `O(fill-in)` in memory and time. On
a 3-D PDE-constrained KKT, fill-in is superlinear; one dense row (a
global coupling constraint — total mass, a budget) destroys sparsity
entirely; and the matrix may simply not fit.

Interior/CG removes the factorization. The cost: a factorization
gives an *exact* solve and *exact inertia*; CG gives neither for free.
Both must be re-earned — exactness by inexact-Newton tests (accept an
inexact solve, but bound its residual), inertia by negative-curvature
detection inside CG (Steihaug truncation) instead of by the
perturbation handler. That is the whole content of this note.

## 3. The three things Interior/CG needs

### 3.1 An iterative augmented-system solver

The `AugSystemSolver` trait boundary is already the right seam (the
composite-step note relies on the same one). Two levels:

- **Level A — iterative on the assembled matrix.** pounce already
  assembles `W` and `J` as explicit sparse matrices. The first useful
  version keeps the assembly and swaps only the *solve*: a Krylov
  method on the assembled sparse saddle-point system. This already
  removes the factorization-fill bottleneck (dense rows, 3-D fill)
  without touching the NLP layer. Lowest risk.
- **Level B — truly matrix-free.** Never assemble `W`; use
  Hessian-vector products from the NLP layer. This needs the `.nl`
  reader and CUTEst FFI to expose `Hv` products — a real
  evaluation-layer addition. Reserve for the largest problems where
  even assembling `W` is too much.

Level A is the Phase-1 deliverable; Level B is a later, separable
extension.

### 3.2 Preconditioning — the make-or-break

A saddle-point KKT system is symmetric indefinite and, near the
boundary, viciously ill-conditioned (the barrier drives diagonal
entries to 0 and ∞). Unpreconditioned CG will not converge in
acceptable iteration counts. **This is the single biggest risk in the
note.** Two standard routes:

- **Projected (reduced) CG** — run CG in the null space of `J`, where
  the operator is the reduced Hessian `ZᵀWZ`, SPD whenever the reduced
  Hessian is positive definite (exactly the regular case). The
  projection onto `null(J)` is itself a solve with the constraint
  block. This is KNITRO's approach and the natural fit: the composite
  step *already* defines the tangential step in `null(J)`, so the
  projection is not extra conceptual machinery — it is the tangential
  subproblem.
- **Constraint preconditioner** — for a full-space MINRES/GMRES solve,
  a block preconditioner with an approximate (1,1) block and an exact
  constraint block. More general, more code.

Recommendation: projected CG, for the reuse argument above.

### 3.3 Inexact-Newton termination tests

Because CG stops early, the step is inexact — its residual is nonzero.
Naively accepting it can stall or diverge. The inexact-Newton
framework (Curtis–Schenk–Wächter; the Ipopt inexact algorithm)
supplies the tests:

- a **residual test** — stop CG when the relative residual is below a
  forcing sequence `ηₖ → 0` that tightens as the iterate approaches a
  solution (fast local convergence is retained iff `ηₖ → 0`);
- a **step-decomposition test** — decide whether the current inexact
  step is *tangential-dominated* (makes optimality progress) or must
  be *normal-dominated* (the linear solve is too inaccurate to trust
  for optimality — fall back to a feasibility-improving step);
- **negative-curvature handling** — Steihaug truncation: if CG
  generates a direction with `dᵀAd ≤ 0`, stop at the trust-region
  boundary. This *replaces* `PerturbForWrongInertia` on this path — no
  factorization means no inertia to read, so non-convexity is detected
  in the Krylov iteration instead.

These tests are the genuine algorithmic content, and they are shared
with the composite-step note's tangential Steihaug-CG — which is why
composite-step must land first.

## 4. What pounce already has to reuse

| Need | Existing component | Location |
|---|---|---|
| Iterative-solver trait seam | `AugSystemSolver` | `kkt/aug_system_solver.rs` |
| Assembled sparse `W`, `J` | computed each iteration | `ipopt_cq.rs` |
| Normal / tangential step decomposition | composite-step note Phase 2 | `composite-step-byrd-omojokun.md` |
| Trust-region radius + ratio test | composite-step note Phase 3 | (same) |
| Negative-curvature → boundary stop | composite-step Steihaug-CG | (same) |
| Constraint-violation / barrier-obj quantities | `curr_constraint_violation`, `curr_barrier_obj` | `ipopt_cq.rs` |
| Sparse matrix–vector products | FERAL / pounce sparse linear algebra | — |

The decisive reuse: **composite-step is the prerequisite, not a
parallel effort.** If composite-step lands, Interior/CG is "swap the
inner solver and add the forcing sequence." If it does not, Interior/CG
would have to build the entire step decomposition itself — at which
point it is a rewrite, not an extension.

## 5. Proposed phasing

- **Phase 0 — depends on composite-step.** Do not start Interior/CG
  until `composite-step-byrd-omojokun.md` Phases 2–3 are implemented
  and validated.
- **Phase 1 — iterative `AugSystemSolver`, Level A, quasi-exact.** A
  Krylov backend (projected CG / MINRES) on the *assembled* sparse
  KKT, solved to *tight* tolerance so the step is effectively exact.
  This is the roadmap's Tier-1 C5 linear-algebra item: a drop-in
  `AugSystemSolver` impl, numerically identical to the direct solver,
  that on its own removes the factorization-fill failure mode.
  Shippable and measurable alone.
- **Phase 2 — preconditioned projected CG.** Add the projection onto
  `null(J)` and a preconditioner. Still tight tolerance. This is where
  iteration counts become acceptable on genuinely large problems.
- **Phase 3 — inexact termination.** Introduce the forcing sequence
  `ηₖ` and the step-decomposition test; let CG stop early. This is the
  step from "iterative linear algebra" to "Interior/CG the
  algorithm," and it reuses composite-step's Steihaug machinery.
- **Phase 4 — truly matrix-free (Level B).** Expose Hessian-vector
  products from the `.nl` / CUTEst layers; drop the explicit `W`
  assembly. For the largest problems only.

Phases 1–2 are Tier 1–2 and independently valuable (faster, more
robust linear algebra for problems that already converge). Phases 3–4
are the genuine Interior/CG and Tier 2–3.

## 6. Problem classes this unlocks

Interior/CG is strictly a **large-scale** play. It unlocks:

- **PDE-constrained optimization** — optimal control and inverse
  problems with a discretized PDE as the constraint. The KKT matrix is
  the discretized PDE operator; 3-D meshes make direct-factorization
  fill-in superlinear. *The* canonical Interior/CG class.
- **Large-scale AC OPF** — transmission-grid optimal power flow at
  realistic size (10⁴–10⁶ buses). Directly relevant to this repo's
  `benchmarks/grid` family scaled up; direct factorization is the
  current ceiling.
- **Dense-row / globally-coupled problems** — a single constraint
  touching all variables (total mass, total budget, an integral)
  destroys sparsity for a direct solver; an iterative solver is
  indifferent to it.
- **Block-structured network / multiperiod problems** — multiperiod
  OPF, supply chains: an iterative solve with a block preconditioner
  exploits structure a generic factorization cannot.
- **The CUTEst large-n tail** — the large problems that currently run
  long or TIMEOUT in the cutest-full sweep (e.g. COSHFUN, n=6001).
  Once cutest-full finishes, the subset of timeouts whose runtime is
  dominated by factorization is the concrete target set; a per-problem
  breakdown is needed to separate those from iteration-count timeouts.

What it does **not** unlock: small and medium problems. There, direct
factorization wins — it is exact and carries no iteration-count risk.
Interior/CG should never be the default; it is opt-in (or
size-gated) for the large regime only.

## 7. Open questions for review

- **Hard dependency on composite-step.** This note assumes
  composite-step Phases 2–3 land first. Confirm that ordering — or
  Phases 3–4 here roughly double in effort.
- **Level A vs Level B.** Is "iterative on the assembled matrix"
  (Level A) enough for the target problems, or is truly matrix-free
  (Level B, `Hv` products through the `.nl` / CUTEst layers) required?
  Level A is far less code. Recommendation: ship Level A, measure,
  do Level B only if assembly itself becomes the bottleneck.
- **Preconditioner scope.** A good generic preconditioner for the
  reduced Hessian is hard. Start with a diagonal/Jacobi preconditioner
  and accept mediocre iteration counts, or invest in a constraint
  preconditioner up front?
- **Default and size-gating.** Opt-in, never default. Should pounce
  auto-select Interior/CG above a size threshold — and if so, on `n`,
  on estimated fill-in, or only on explicit request?
- **Inertia.** Confirm that negative-curvature detection in CG fully
  replaces `PerturbationHandler` on this path, or whether a hybrid is
  needed when CG curvature information is ambiguous.

## 8. References

Citation keys in brackets refer to sources ingested in `.crucible/`.

- Byrd, Hribar & Nocedal, "An interior point algorithm for
  large-scale nonlinear programming," *SIAM J. Optim.* 9 (1999)
  [byrd1999] — the composite-step + projected-CG large-scale
  algorithm; the foundation of KNITRO Interior/CG.
- Waltz, Morales, Nocedal & Orban, "An interior algorithm for
  nonlinear optimization that combines line search and trust region
  steps," *Math. Prog.* 107 (2006) [waltz2006] — the KNITRO interior
  algorithm; CG steps with a line-search fallback.
- Byrd, Gilbert & Nocedal, "A trust region method based on interior
  point techniques for nonlinear programming," *Math. Prog.* 89
  (2000) [byrd2000] — the trust-region interior-point foundation.
- Curtis, Schenk & Wächter, "An interior-point algorithm for
  large-scale nonlinear optimization with inexact step computations,"
  *SIAM J. Sci. Comput.* 32 (2010) [curtis2010] — Ipopt's inexact
  algorithm; the inexact-Newton termination tests.
- Curtis, Nocedal & Wächter, "A matrix-free algorithm for equality
  constrained optimization problems with rank-deficient Jacobians,"
  *SIAM J. Optim.* 20 (2009) — the matrix-free lineage.
- `ref/Ipopt/src/Algorithm/Inexact/` — Ipopt's in-tree inexact
  composite-step implementation; the closest reference.
