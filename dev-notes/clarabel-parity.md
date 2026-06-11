# Clarabel cone parity for the convex IPM — design note

**Status: scoping.** POUNCE's `pounce-convex` solves LP/QP/SOCP over a
product of nonnegative orthants and second-order cones (see
`socp-extension.md`). This note scopes closing the remaining cone gap
versus [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs): the
**exponential**, **power**, and **positive-semidefinite (PSD)** cones.
Together with what we have, that is the full Clarabel cone set and covers
geometric programming, entropy/logistic/softmax models, robust/relative-
entropy programs, and semidefinite programming.

## Where we are

The IPM is a Mehrotra predictor–corrector over the
[`Cone`](../crates/pounce-convex/src/cones/mod.rs) trait, dispatched
block-wise by [`CompositeCone`]. Every cone supplies `mu`, a `kkt_block`
(the `(z,z)` scaling), `comp_residual{,_corrector}`, `recover_ds`,
`rhs_comp_term`, `max_step`, `recenter_warm`. The driver, residuals,
factor reuse, presolve postsolve, batch, and warm start are all
cone-agnostic and reused.

The crucial property the current driver **assumes**: the cone is
**symmetric** (self-scaled). Concretely it bakes in

1. a Jordan product `s∘z` and centrality `μ = ⟨s,z⟩/degree`,
2. a Nesterov–Todd scaling point `W` with `W² z = s` (the `kkt_block`),
3. the Mehrotra corrector second-order term `ds_aff ∘ dz_aff`.

Nonneg and SOC are symmetric, so they fit. **PSD is symmetric too.**
**Exp and power are not.**

## Two machinery tracks

### Track S — PSD (symmetric, extends what we have)

The PSD cone `S₊ᵏ = { X = Xᵀ : X ⪰ 0 }` is self-scaled, so it slots into
the existing predictor–corrector with the *matrix* analogues of the SOC
algebra:

- **Vectorization.** Slack/dual are symmetric `k×k` matrices stored in
  `svec` (scaled lower triangle, off-diagonals ×√2 so `⟨svec a, svec b⟩ =
  ⟨A,B⟩`). A PSD block spans `k(k+1)/2` rows.
- **Jordan product / centrality.** `A∘B = ½(AB+BA)`, identity `I`,
  `μ = ⟨S,Z⟩/k`, degree `k` per block.
- **NT scaling.** `W` from `R` with `RᵀZR = I`, `RᵀSR⁻¹... ` — in practice
  `W = Z^{-1/2}(Z^{1/2}SZ^{1/2})^{1/2}Z^{-1/2}` (one symmetric
  eigendecomposition of `Z^{1/2}SZ^{1/2}` per iteration per block). The
  `kkt_block` is the dense `W⊗ₛW` operator on `svec` (a new
  `ConeBlock::Dense`/operator form — *not* diagonal-plus-rank-1).
- **Step to boundary.** `max_step` = largest `α` keeping `V + αdV ⪰ 0`,
  i.e. `1/λ_max(-V^{-1/2} dV V^{-1/2})` (a generalized-eigenvalue / Cholesky
  line search), the matrix analogue of SOC's boundary root.

**Lift:** an eigendecomposition (or two) per PSD block per iteration, the
`svec`/`smat` plumbing, and a genuinely **dense** `(z,z)` block (the SOC
diagonal-plus-rank-1 trick does not apply). For large/sparse SDPs,
competitiveness needs **chordal decomposition** (Clarabel's `clique`
merging) — split a sparse PSD constraint into many small coupled PSD
blocks. That is a sizable sub-project on its own and can be a later phase
(small dense SDPs first, chordal later).

**Risk:** medium-high but *contained to the existing loop* — no new IPM.
The risk is matrix-algebra correctness (NT matrix scaling, the dense KKT
operator, the eigen line search), validated the usual way (known SDP
optima: min/max eigenvalue, Lyapunov, a small SDP relaxation; plus a
randomized KKT-residual check).

### Track N — Exponential & power (non-symmetric, new IPM components)

`K_exp = cl{ (x,y,z) : y>0, y·e^{x/y} ≤ z }` and the power cone
`K_pow^α = { (x,y,z) : x^α y^{1-α} ≥ |z|, x,y≥0 }` are **not** self-scaled:
there is no `W` with `W²z = s`, no `s∘z`, no symmetric `μ`. They need the
non-symmetric path-following machinery (Nesterov–Todd 1997; Skajaa–Ye
2015; Dahl–Andersen 2021 — the MOSEK exp-cone algorithm; the approach
Clarabel and Hypatia use):

- **Barrier oracles.** Each cone supplies its logarithmically-homogeneous
  self-concordant barrier `f`, gradient `g=∇f`, and Hessian `H=∇²f`
  (exp-cone barrier `−log(y log(z/y) − x) − log y − log z`, degree 3). The
  trait grows `barrier_grad`/`barrier_hess` (symmetric cones can supply
  closed forms too, unifying the code).
- **Scaling.** Replace the NT point with a **dual-aware primal–dual
  scaling** built from *both* cone iterates — the Tunçel scaling (Tunçel
  2001; Myklebust–Tunçel 2014), specialized to 3-D and computed by a BFGS
  update as in Dahl–Andersen 2021. The `kkt_block` becomes that dense, small
  (3×3 for exp/power) `WᵀW`. The cheaper primal-only Hessian scaling was
  tried and **stalls** (the dual races to the boundary); see the worked
  construction and prototype findings in `hsde.md` (§"The dual-aware scaling
  (item #1)").
- **Centrality & step.** `μ = ⟨s,z⟩/Σdegree` still defines the target, but
  the corrector uses a **third-order** correction term (not `ds∘dz`) —
  Dahl–Andersen's Mehrotra-like nonsymmetric corrector — and the step
  length needs a **neighborhood / line search on the barrier** (stay where
  `f` is finite and inside the wider neighborhood), since there is no
  closed-form boundary root.
- **Robustness ⇒ HSDE (decision point).** Non-symmetric cones are far more
  robustly handled inside a **homogeneous self-dual embedding** (Clarabel,
  SCS, ECOS-exp all do). Our solver currently uses a direct primal–dual
  method with explicit Farkas/recession certificates. Adding exp/power
  *without* HSDE is possible (Mosek-style) but more fragile and complicates
  infeasibility detection; adding HSDE first is a foundational investment
  that also cleans up certificates and gives a single uniform driver for
  all cones. **This is the biggest architectural decision in the program.**

**Lift:** new IPM components (barrier oracles, non-symmetric scaling,
higher-order corrector, neighborhood line search) and, recommended, the
HSDE reformulation of the driver. The cones themselves are tiny (3-D), so
once the machinery exists, **power cone is incremental over exp cone**
(same framework, different barrier).

**Risk:** high — this is effectively a second IPM. Validate against known
optima (GP: posynomial min; entropy max; logistic regression NLL; the
exp-cone "softplus" epigraph) and randomized KKT residuals.

## Trait / driver changes (both tracks)

- `ConeBlock` gains a **dense operator** form for PSD (`W⊗ₛW` apply) and a
  small-dense form for exp/power (3×3); the KKT assembly already has a
  dense-lower path from SOC Tier-A — generalize it.
- `Cone` gains `barrier_grad`/`barrier_hess` (Track N), and PSD needs an
  `svec` working buffer + eigendecomposition (Track S). A small dense
  symmetric eig (Jacobi or tridiagonal QL) lands in the crate — **pure
  Rust, no LAPACK** (the project's standing constraint).
- Cold start: PSD at `I` (in svec), exp/power at the cones' analytic
  central ray.
- Presolve: gate `≤`-row reductions off PSD/exp/power blocks exactly as
  `presolve_conic` already does for SOC (coupled rows).
- Differentiable layer (last, per cone): the OptNet backward needs each
  cone's complementarity differential — the symmetrized matrix product for
  PSD, the barrier-Hessian form for exp/power — added and FD-validated as a
  distinct follow-up, exactly as SOC was.

## Recommended ordering (for discussion)

Three coherent ways to sequence; the choice is a genuine trade of
value-first vs risk-first and is the open question:

1. **Exp cone first (value-first).** Unlocks the largest *new application
   surface* (GP, logistic, entropy, softmax, relative entropy — the
   ML/stats workhorses) and builds the non-symmetric machinery that power
   cone then reuses almost for free. Highest value, highest risk; likely
   wants HSDE underneath.
2. **PSD cone first (fits-our-framework).** Stays inside the symmetric
   predictor–corrector we trust; marquee SDP capability; the linear-algebra
   lift (eig, svec, dense block, later chordal) is heavy but the *algorithm*
   is familiar. Lower algorithmic risk, no HSDE needed.
3. **HSDE foundation first.** Reformulate the driver into a homogeneous
   self-dual embedding, then drop exp → power → PSD onto it uniformly
   (Clarabel's structure). Slowest to first visible win, but the cleanest
   end state and the most robust non-symmetric handling.

| Track | Cone | Machinery | Value | Risk |
|---|---|---|---|---|
| S | PSD | extends NT; eig + dense svec block; chordal later | SDP | med-high (contained) |
| N | Exp | non-symmetric IPM; barrier oracles; +HSDE | GP/ML/entropy | high |
| N | Power | exp machinery + new barrier | robust/`p`-norm | low *after* exp |

Each cone follows the SOCP playbook: land forward/solve with intrinsic
validation (known optima + randomized KKT residual), gate presolve, add
warm-start recentering, then a cone-aware differentiable backward as a
separate FD-validated follow-up. The orthant/SOC paths stay byte-identical
throughout.
