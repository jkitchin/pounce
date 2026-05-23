# Design note — Active-set SQP for warm-started NLP sequences

**Status: design / proposed. Not yet implemented.** Research → plan
half of the research → plan → implement workflow; written for review
before any solver code lands. This note operationalizes the C1
active-set SQP entry of
[`future-work-roadmap.md`](future-work-roadmap.md) (§3.2, §5 Phase 5)
and pins each algorithmic choice to the literature so the
implementation phase has no remaining design discretion.

The target is a **state-of-the-art sparse active-set SQP solver** that
(a) reuses pounce's NLP / derivative / sparse-linalg foundation, (b)
warm-starts on the working set across solves (not just primal-dual
seeds), and (c) integrates symmetrically across the Rust API, C ABI,
Python bindings, and GAMS link.

## 1. What this is

A **sequential quadratic programming** algorithm with a sparse
parametric active-set QP subproblem — a second solver inside pounce
sharing the model / derivative / linalg foundation but with its own
iteration skeleton — designed for **warm-started sequences of related
NLPs**:

- **Model predictive control (MPC):** re-solve a similar NLP every
  control step. The horizon shifts by one stage; the active set
  rarely changes.
- **MINLP branch-and-bound:** thousands of node relaxations differing
  by a few bound changes. Bounds-only active-set updates dominate.
- **Parametric homotopy / continuation:** trace the solution along a
  parameter path. Predictor (sensitivity) + corrector (SQP step from
  the predicted point) reuses the working set across path steps.

The motivation is documented in `future-work-roadmap.md:185-206`:
interior-point methods warm-start badly because the barrier pushes
iterates to the interior, so a near-optimal point from a previous
solve sits near the bound boundary and cannot be exploited. Active-
set methods, by contrast, carry the **working set** across solves; if
the optimal active set is unchanged, the next solve converges in O(1)
QP iterations. This is the documented reason qpOASES, SNOPT, and
filterSQP dominate in MPC.

## 2. The architectural mismatch (read this first)

`IpoptData` / `IpoptCalculatedQuantities` are shaped around primal-
dual interior-point variables — slacks `s`, barrier `μ`, bound
multipliers `z_l`/`z_u`, complementarity quantities. Active-set SQP
has none of these: it carries `(x, λ, 𝒲)` where `𝒲` is the **working
set** — the indices of currently active inequalities and bounds — and
globalizes on a merit function or filter without a barrier at all.

This is therefore **a new `AlgorithmStrategy` end to end** — Tier 3 in
the roadmap's tier ladder (`future-work-roadmap.md:290-300`) — and
not an edit to the existing loop. The existing IPM
(`IpoptAlgorithm::optimize` in
`crates/pounce-algorithm/src/ipopt_alg.rs`) is left untouched and
remains the default solver. Active-set SQP is opt-in via a new
top-level `algorithm` option (§7.1), parallel to the existing
`linear_solver` (Ma57/Feral) and `mu_strategy` (Monotone/Adaptive)
choices in `alg_builder.rs:54-63`.

The dual-skeleton commitment is the cost; the warm-start strength is
the payoff.

## 3. What pounce already has that SQP can reuse

| Need | Existing component | Location |
|---|---|---|
| NLP model trait (`f`, `g`, `∇f`, `J`, `∇²ℒ`) | `IpoptNlp` / `TNLP` | `crates/pounce-algorithm/src/ipopt_nlp.rs`, `crates/pounce-nlp/` |
| `.nl` and CUTEst frontends | `pounce-cli`, `benchmarks/cutest` | unchanged |
| Sparse storage (triplet + CSC) | `SymTMatrix`, triplet→CSR converter | `crates/pounce-linalg/src/triplet.rs:374-405`, `triplet_convert.rs:40` |
| Sparse symmetric LDLᵀ with inertia | `SparseSymLinearSolverInterface` (FERAL, MA57) | `crates/pounce-linsol/src/sparse_sym_iface.rs:42-84` |
| Multi-RHS solve sharing one factor | `t_sym_solver.rs::multi_solve` | `crates/pounce-linsol/src/t_sym_solver.rs:174` |
| Inertia reporting (eigenvalue counts) | `SparseSymLinearSolverInterface::provides_inertia` | `crates/pounce-linsol/src/sparse_sym_iface.rs:84` |
| Limited-memory BFGS / SR1 | `hess/quasi_newton.rs` | reused for SQP Hessian approximation |
| Filter acceptor | `line_search/filter_ls_acceptor.rs` | dominance test reusable for SQP filter |
| Convergence-check trait | `conv_check::trait::ConvCheck` | reused; KKT-error formula is identical |
| Option / journalist / iteration-output | `pounce-common` + `output/` | reused; new fields for working-set events |
| Warm-start primal/dual seeds from `TNLP` | `init/warm_start.rs:60-100` | extended (§6) with working-set state |
| Parametric sensitivity (sIPOPT port) | `pounce-sensitivity` | provides predictor for parametric-homotopy use case |

The interfaces below pounce-nlp are stable enough that SQP inherits
the full derivative and linalg layer unchanged. Everything new lives
at the algorithm / solver level.

## 4. The algorithm — fully pinned

This section pins each algorithmic choice to literature. There is no
remaining "decide during implementation" discretion at the level of
algorithm class; only tuning constants are open.

### 4.1 Outer SQP loop — filter line search with Maratos correction

The outer loop is the **filter SQP** of Fletcher-Leyffer-Toint, with
the Wächter-Biegler second-order correction (Maratos effect) and
watchdog mechanism already implemented in `line_search/`. Filter
because:

- It avoids the penalty-parameter tuning of l1-merit (Han-Powell).
- It reuses pounce's existing `FilterLsAcceptor`
  (`line_search/filter_ls_acceptor.rs`) without modification — the
  dominance test on `(‖c‖, f)` is identical.
- It is the globalization in **filterSQP** (Fletcher-Leyffer) and
  **WORHP**, the two open-source SQP solvers that compete with SNOPT
  on CUTEst, and the documented choice in Nocedal-Wright §18.10.

**Alternative offered as opt-in:** l1-elastic merit (the SNOPT
choice), via a `sqp_globalization` option. l1 is simpler to reason
about under MPCC-like degeneracies; filter is faster on smooth
nonconvex NLPs in published benchmarks (Fletcher-Leyffer-Toint 2002
§6; Wächter-Biegler 2006 Tab. 3-5).

**References:**
- Fletcher, Leyffer, "Nonlinear programming without a penalty
  function", *Math. Prog.* **91** (2002), 239–269.
- Fletcher, Leyffer, Toint, "On the global convergence of a filter-
  SQP algorithm", *SIAM J. Optim.* **13** (2002), 44–59.
- Wächter, Biegler, "Line search filter methods for nonlinear
  programming: Motivation and global convergence", *SIAM J. Optim.*
  **16** (2005), 1–31.
- Wächter, Biegler, "On the implementation of an interior-point
  filter line-search algorithm for large-scale nonlinear
  programming", *Math. Prog.* **106** (2006) — pounce's existing
  filter implementation.

### 4.2 QP subproblem — sparse Schur-complement parametric active-set

The QP subproblem solver is a **sparse parametric active-set method
with Schur-complement basis updates**, the lineage of qpOASES
extended to sparse Hessian and Jacobian. This is the SOTA choice for
SQP subproblems in industrial MPC and for parametric / homotopy use:
it is the **only** active-set QP family in the literature with
proven cross-solve warm-start performance in the sparse regime.

**Why this family** (vs alternatives):

| Family | Sparse? | Indefinite H? | Parametric WS warm-start? | Reference |
|---|---|---|---|---|
| Goldfarb-Idnani (1983) | no (dense) | no (convex only) | partial | Goldfarb-Idnani 1983 |
| Range-space (SQOPT) | partial | yes | partial | Gill-Murray-Saunders 2008 |
| Null-space (Gould-Hribar-Nocedal) | partial | yes | partial | Gould-Hribar-Nocedal 2001 |
| qpOASES (online active set) | **no** (dense) | yes | **yes** (homotopy) | Ferreau et al. 2014 |
| **Sparse Schur-complement parametric** | **yes** | **yes** | **yes** | Kirches 2011, Janka 2017 |
| OSQP (ADMM) | yes | no (convex only) | seed only | Stellato et al. 2020 |
| PIQP / HPIPM (interior-point) | yes | yes | seed only | Schwan 2023, Frison-Diehl 2020 |

Only the sparse Schur-complement parametric method covers all three
columns. It is what is needed.

**Algorithm sketch.** At any iterate the QP solver maintains a
factorization of the **base KKT matrix** for some "base" working set
𝒲_base:

```
        ┌ H   Aᵀ_𝒲 ┐
K_𝒲 =  │            │,    LDLᵀ via pounce-linsol (FERAL/MA57)
        └ A_𝒲  0   ┘
```

When the working set changes (a constraint is added or dropped during
the homotopy), the new system is **not refactorized**. Instead, the
change is absorbed by a **Schur-complement update**: the modified
system has the form `K_𝒲 + UVᵀ` (low-rank correction), and solves
against the modified factor are obtained by the Schur-complement
formula

```
(K + UVᵀ)⁻¹ b = K⁻¹b − K⁻¹U (I + Vᵀ K⁻¹ U)⁻¹ Vᵀ K⁻¹ b
```

so each active-set change costs one rank-1 update of the dense Schur
complement `S = I + Vᵀ K⁻¹ U` plus one back-solve against the cached
sparse factor. This is the **Bartels-Golub-Reid** principle from
sparse simplex adapted to symmetric QP. When `S` grows too large
(default: 50 updates) or its condition number degrades, a fresh
sparse refactorization of `K_𝒲` resets the cycle.

The homotopy itself follows qpOASES: between two QPs `(H₀, g₀, A₀,
b₀)` and `(H₁, g₁, A₁, b₁)`, the solver traces the parametric path
`(H_t, g_t, A_t, b_t) = (1-t)·QP₀ + t·QP₁` for `t ∈ [0, 1]`, jumping
the working set at each `t` where a multiplier hits zero or a
constraint hits its bound. If the active set is identical at the two
endpoints (the warm-start sweet spot), the homotopy completes with
zero working-set changes.

**Why Schur-complement, not direct LDLᵀ update?** Direct sparse LDLᵀ
factor updates (the symbolic+numeric reanalysis required when a
constraint row is added or dropped) are known to be unstable under
many updates because fill-in is not bounded (Davis 2006 §11). The
Schur-complement / Bartels-Golub-Reid approach bounds the
asymptotic update cost and is the technique that production sparse
simplex (CPLEX, Gurobi, HiGHS) and SOTA sparse parametric QP
(Kirches's `qpDUNES`, the Janka `parOSQP` lineage) use.

**References:**
- Ferreau, Kirches, Potschka, Bock, Diehl, "qpOASES: a parametric
  active-set algorithm for quadratic programming", *Math. Prog.
  Comp.* **6** (2014), 327–363 — the dense reference algorithm.
- Kirches, *Fast Numerical Methods for Mixed-Integer Nonlinear
  Model-Predictive Control*, Vieweg+Teubner (2011), Ch. 5–7 — the
  sparse Schur-complement extension; the canonical reference.
- Janka, Kirches, Sager, Schlöder, "An SR1/BFGS SQP algorithm for
  nonconvex nonlinear programs with block-diagonal Hessian matrix",
  *Math. Prog. Comp.* **8** (2016), 435–459 — block-sparse extension.
- Kirches, Potschka, Bock, Sager, "A parametric active set method for
  quadratic programs with vanishing constraints", *Pacific J. Optim.*
  **9** (2013) — MPCC structure handling, relevant to C4 reuse.
- Bartels, "A stabilization of the simplex method", *Numer. Math.*
  **16** (1971); Reid, "A sparsity-exploiting variant of the
  Bartels-Golub decomposition", *Math. Prog.* **24** (1982) — the
  Schur-complement basis-update lineage.
- Eldersveld, Saunders, "A block-LU update for large-scale linear
  programming", *SIAM J. Matrix Anal. Appl.* **13** (1992).
- Gill, Murray, Saunders, "SNOPT: An SQP algorithm for large-scale
  constrained optimization", *SIAM Rev.* **47** (2005) — the
  range-space active-set used inside SNOPT; competing family.
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM (2006) —
  fill-in and refactor cost analysis.

### 4.3 Phase-1 / initial feasibility — l1 elastic mode

Active-set QP requires a feasible starting working set. The
**l1-elastic mode** (Gill-Murray-Saunders, SQOPT) reformulates the
infeasibility problem inside the *same* QP: each constraint gets a
nonnegative elastic slack with a large linear cost γ, the working
set starts empty, and elastic slacks are driven to zero as the
homotopy proceeds. If the original QP is feasible the elastic slacks
vanish at the solution; if infeasible the residual elastic slacks
certify the minimal infeasibility.

This is preferred over the Big-M approach used in dense qpOASES
because it preserves sparsity (the cost vector grows by `m` entries,
the Jacobian by `m` columns; no large constants in `H`).

**References:**
- Gill, Murray, Saunders, *User's Guide for SQOPT 7.7*, Stanford SOL
  Report (2008) — elastic-mode reference implementation.
- Friedlander, Saunders, "A globally convergent linearly constrained
  Lagrangian method for nonlinear optimization", *SIAM J. Optim.*
  **15** (2005) — elastic mode as feasibility restoration.

### 4.4 Anti-cycling — EXPAND

Degeneracy in the working set (multiple constraints active with
linearly dependent rows, or zero step lengths) can cause cycling in
naive active-set methods. The SOTA anti-cycling rule is **EXPAND**
(Gill-Murray-Saunders-Wright 1989): a small primal perturbation is
introduced and grown over iterations so that the step length is
always strictly positive, with periodic resets.

Bland's rule (1977) and Wolfe's rule (1963) are alternatives, but
EXPAND is faster in practice and is the rule used by SNOPT, MINOS,
LANCELOT, and qpOASES.

**References:**
- Gill, Murray, Saunders, Wright, "A practical anti-cycling
  procedure for linearly constrained optimization", *Math. Prog.*
  **45** (1989), 437–474.

### 4.5 Indefinite reduced Hessian — inertia control + projected modified Cholesky

For nonconvex NLP subproblems the Hessian of the Lagrangian is
indefinite. The QP must still be solved to a meaningful descent
direction. Two-layer scheme, both standard:

1. **Detect** via inertia of the LDLᵀ factor of `K_𝒲`. pounce-linsol
   already exposes inertia via `provides_inertia()` /
   `number_of_neg_evals` (`sparse_sym_iface.rs:84`). The correct
   inertia for an SQP subproblem with `m` working constraints is
   `(n − m, m, 0)`; any deviation flags reduced-Hessian indefiniteness.
2. **Correct** via projected modified Cholesky on the reduced
   Hessian: when wrong inertia is detected, shift `H ← H + δI` with δ
   chosen by the same inertia-correction logic pounce already uses
   in `kkt/perturbation_handler.rs:141-356`. This restores correct
   inertia at minimal modification.

**References:**
- Gould, "On modified factorizations for large-scale linearly
  constrained optimization", *SIAM J. Optim.* **9** (1999),
  1041–1063.
- Gould, Hribar, Nocedal, "On the solution of equality constrained
  quadratic programming problems arising in optimization", *SIAM J.
  Sci. Comput.* **23** (2001), 1376–1395 — the inertia-correction
  prescription for SQP subproblems.
- Forsgren, "Inertia-controlling factorizations for optimization
  algorithms", *Appl. Num. Math.* **43** (2002), 91–107.

### 4.6 Hessian approximation — exact, damped BFGS, L-BFGS

The SQP outer loop accepts three Hessian sources via the existing
`HessianUpdater` trait (`hess/r#trait.rs`):

- **Exact `∇²ℒ`** from the NLP (default when available). Indefinite
  on nonconvex problems; handled by §4.5.
- **Damped BFGS** (Powell 1978): full dense BFGS with Powell's
  damping rule, guaranteed PSD. Default fallback when exact Hessian
  is unavailable, for problems where `n` is small.
- **Limited-memory BFGS / SR1** (Liu-Nocedal 1989, Byrd-Nocedal-Schnabel
  1994): the existing pounce L-BFGS implementation. Default for
  large `n`. SR1 is the indefinite-Hessian variant preferred in
  Janka 2016 for nonconvex SQP block-sparse problems.

The QP subproblem absorbs whichever Hessian is supplied; only the
indefinite-handling path (§4.5) differs.

**References:**
- Powell, "A fast algorithm for nonlinearly constrained optimization
  calculations", in *Numerical Analysis Dundee 1977* (1978) — damped
  BFGS for SQP.
- Liu, Nocedal, "On the limited memory BFGS method for large scale
  optimization", *Math. Prog.* **45** (1989), 503–528.
- Byrd, Nocedal, Schnabel, "Representations of quasi-Newton
  matrices and their use in limited memory methods", *Math. Prog.*
  **63** (1994), 129–156.

### 4.7 Iterative refinement

Single iteration of fixed-precision iterative refinement on every QP
solve, using the cached factorization. Standard practice; pounce-feral
and MA57 backends already implement it (`t_sym_solver.rs::multi_solve`
applies refinement when configured).

**References:**
- Wilkinson, *The Algebraic Eigenvalue Problem*, OUP (1965) — original.
- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.,
  SIAM 2002), §12.

## 5. New crate `pounce-qp` — concrete types

Standalone crate. Depends on `pounce-linalg` and `pounce-linsol`;
depended on by `pounce-algorithm` (for SQP), `pounce-sensitivity`
(for the parametric corrector in Phase 5c+), optionally
`pounce-presolve` (for tighter feasibility checks in future work).

### 5.1 Types

All types are sparse from the start, using the existing
`pounce-linalg` storage conventions (`SymTMatrix` triplet → CSC for
the symmetric Hessian; `GenTMatrix` for the Jacobian).

```rust
// crates/pounce-qp/src/problem.rs

use pounce_linalg::triplet::{SymTMatrix, GenTMatrix};

/// A convex-or-nonconvex sparse QP:
///     min  ½ xᵀ H x + gᵀ x
///     s.t. bl ≤ A x ≤ bu
///          xl ≤   x ≤ xu
/// Two-sided general bounds; H is symmetric (upper triangle stored)
/// and may be indefinite (caller sets `hessian_inertia`).
pub struct QpProblem<'a> {
    pub n: usize,
    pub m: usize,
    pub h: &'a SymTMatrix,          // symmetric, upper triangle, may be indefinite
    pub g: &'a [f64],
    pub a: &'a GenTMatrix,           // m × n, sparse
    pub bl: &'a [f64], pub bu: &'a [f64],
    pub xl: &'a [f64], pub xu: &'a [f64],
    pub hessian_inertia: HessianInertia,  // PSD | Indefinite | Unknown
}

/// Discrete state per primal-and-constraint index. Carried across
/// solves to implement working-set warm start.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BoundStatus { Inactive, AtLower, AtUpper, Fixed }

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConsStatus { Inactive, AtLower, AtUpper, Equality }

pub struct WorkingSet {
    pub bounds:      Vec<BoundStatus>,   // length n
    pub constraints: Vec<ConsStatus>,    // length m
}

pub struct QpWarmStart {
    pub x:       Vec<f64>,
    pub lambda_g: Vec<f64>,              // length m
    pub lambda_x: Vec<f64>,              // length n (z_l − z_u, signed)
    pub working:  WorkingSet,
}

pub struct QpSolution {
    pub x:        Vec<f64>,
    pub lambda_g: Vec<f64>,
    pub lambda_x: Vec<f64>,
    pub working:  WorkingSet,
    pub obj:      f64,
    pub status:   QpStatus,              // Optimal | Infeasible | Unbounded | MaxIter | …
    pub stats:    QpStats,               // n_active_set_changes, n_refactor, time …
}
```

### 5.2 Trait surface

```rust
// crates/pounce-qp/src/solver.rs

use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;

pub trait QpSolver {
    /// Solve a single QP. `ws` is `None` for cold start.
    fn solve(
        &mut self,
        qp: &QpProblem,
        ws: Option<&QpWarmStart>,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;

    /// Parametric solve: trace the homotopy from a previous QP+solution
    /// to a new QP. Falls back to `solve` if the previous solution is
    /// `None`. This is the entry point SQP uses across outer iterations
    /// to reuse the cached factorization across consecutive QPs.
    fn solve_parametric(
        &mut self,
        qp_prev: &QpProblem,
        sol_prev: &QpSolution,
        qp_new:  &QpProblem,
        opts: &QpOptions,
    ) -> Result<QpSolution, QpError>;
}

pub struct QpOptions {
    pub algorithm: QpAlgorithm,          // ParametricActiveSet | …
    pub linear_solver_factory: …,        // injected from pounce-algorithm
    pub max_iter: usize,
    pub feas_tol: f64,
    pub opt_tol:  f64,
    pub max_schur_updates_before_refactor: usize,  // default 50, ref §4.2
    pub anti_cycling: AntiCyclingChoice, // Expand (default), Bland, None
    pub elastic_gamma: f64,              // §4.3 penalty for elastic mode
    pub print_level: i32,
}
```

The `linear_solver_factory` injection mirrors
`alg_builder.rs::LinearBackendFactory` (line 50) so `pounce-qp`
remains backend-agnostic: FERAL by default, MA57 when built with the
`ma57` feature.

### 5.3 Internal structure

```
crates/pounce-qp/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── problem.rs           — types from §5.1
    ├── working_set.rs       — WorkingSet ops: add, drop, validate
    ├── kkt.rs               — KKT assembly from QP + 𝒲
    ├── factor.rs            — sparse LDLᵀ wrapper + Schur-complement state
    ├── schur.rs             — block-LU update (Eldersveld-Saunders 1992)
    ├── homotopy.rs          — parametric step engine (§4.2 t ∈ [0,1])
    ├── elastic.rs           — phase-1 elastic mode (§4.3)
    ├── expand.rs            — EXPAND anti-cycling (§4.4)
    ├── inertia.rs           — indefinite handling (§4.5)
    ├── refine.rs            — iterative refinement (§4.7)
    ├── solver.rs            — QpSolver impl
    └── options.rs           — QpOptions, defaults
```

## 6. SQP iterate state and working-set warm-start contract

```rust
// crates/pounce-algorithm/src/sqp/iterates.rs

pub struct SqpIterates {
    pub x:        Rc<DenseVector>,
    pub lambda_g: Rc<DenseVector>,
    pub lambda_x: Rc<DenseVector>,
    pub working:  WorkingSet,            // §5.1
    pub h_approx: HessianStore,          // exact | DampedBfgs | LBfgs (existing)
    pub merit:    Option<f64>,           // l1-elastic mode or filter pair cache
}
```

The warm-start contract carried across calls to
`SqpAlgorithm::optimize` is the tuple `(x, λ_g, λ_x, 𝒲, H)`:

1. **`(x, λ_g, λ_x)`** — already supported by the existing
   `init/warm_start.rs` machinery; reuse the seed-from-NLP path
   (`warm_start.rs:60-100`).
2. **`𝒲` (working set)** — new. Encoded as `(Vec<BoundStatus>,
   Vec<ConsStatus>)`. Transmitted via:
   - Rust: a new `SqpWarmStartIterateInitializer` parallel to the
     IPM one, populated by an extended `TNLP::get_warm_start_working_set`
     hook (Rust trait default: returns `None` ⇒ cold-start the
     working set via §4.3 elastic mode).
   - C/Python/GAMS: §7.
3. **`H` (Hessian)** — already supported via the existing L-BFGS
   carry-forward path; reuse unchanged.

**Cold-warm bootstrap** (no prior `𝒲`): elastic-mode QP §4.3 with
empty initial working set. The first QP infers `𝒲₀` from which
elastic slacks vanish at its solution.

**Validation:** before consuming a user-supplied `𝒲_prev`, run a
linear feasibility check against the new bounds. If a previously
active bound is now infeasible, drop it (degrades to a cheaper warm
start, never to incorrectness). This is the same defensive check
qpOASES does on `set_warm_start_x`.

## 7. Integration with pounce — symmetric across interfaces

Each interface today is documented in the survey above. The
integration plan below adds the same five-point contract
(algorithm choice + suboptions + warm-start input + warm-start
output + working-set typed-or-string surface) to each, without
disturbing existing IPM users.

### 7.1 Rust / `alg_builder.rs` — the source of truth

New enum following the established `LinearSolverChoice` /
`MuStrategyChoice` pattern at `alg_builder.rs:54-63`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmChoice {
    InteriorPoint,   // default; existing IpoptAlgorithm
    ActiveSetSqp,    // new SqpAlgorithm
}
```

`AlgorithmBuilder` gains an `algorithm: AlgorithmChoice` field with
default `InteriorPoint`. `build_inner` branches on it, returning
either the existing `AlgorithmBundle` (IPM) or a new
`SqpAlgorithmBundle`. The two bundles share `init`, `conv_check`,
`hess`, `iter_output`; differ in main-loop driver.

New options registered in `upstream_options.rs` (the registry
pattern at lines 510-703 for the existing warm-start knobs):

| Option | Type | Default | Meaning |
|---|---|---|---|
| `algorithm` | enum | `interior-point` | `interior-point` ‖ `active-set-sqp` |
| `sqp_qp_solver` | enum | `parametric-active-set` | placeholder for future QP backends |
| `sqp_globalization` | enum | `filter` | `filter` ‖ `l1-elastic` |
| `sqp_hessian` | enum | `exact` | `exact` ‖ `damped-bfgs` ‖ `lbfgs` |
| `sqp_warm_start_working_set` | bool | `no` | accept caller-supplied 𝒲 |
| `sqp_max_qp_iter` | int | 200 | per-QP iteration cap |
| `sqp_qp_feas_tol` | num | 1e-9 | QP feasibility tolerance |
| `sqp_elastic_gamma` | num | 1e6 | elastic-mode penalty (§4.3) |
| `sqp_max_schur_updates` | int | 50 | refactor frequency (§4.2) |

### 7.2 C API (`crates/pounce-cinterface/`)

Three additions, all backward-compatible (existing IPM users see no
change).

**(a) Option exposure.** No new C entry point — `AddIpoptStrOption`
already accepts arbitrary option names. Setting `algorithm` via
`AddIpoptStrOption(problem, "algorithm", "active-set-sqp")` selects
the SQP path. This is identical to how `linear_solver` is selected
today.

**(b) Working-set transfer.** Three new C entry points in
`include/pounce.h`, ABI-stable (no change to existing structs):

```c
/* Length-n status vectors. 0=Inactive, 1=AtLower, 2=AtUpper, 3=Fixed/Equality. */
typedef int IpoptBoundStatus;
typedef int IpoptConsStatus;

/* Retrieve the working set from the last solve. Returns 0 on success.
 * Buffers must be sized n and m respectively. NULL buffer ⇒ skip that side. */
int IpoptGetWorkingSet(
    IpoptProblem problem,
    IpoptBoundStatus *bound_status_out,   /* length n, or NULL */
    IpoptConsStatus  *cons_status_out     /* length m, or NULL */
);

/* Supply a warm-start working set for the next solve. Buffers may
 * be NULL ⇒ that side is cold-started. Caller-owned; copied. */
int IpoptSetWarmStartWorkingSet(
    IpoptProblem problem,
    const IpoptBoundStatus *bound_status_in,   /* length n, or NULL */
    const IpoptConsStatus  *cons_status_in     /* length m, or NULL */
);

/* One-shot solve with warm-start state. Equivalent to IpoptSolve
 * preceded by IpoptSetWarmStartWorkingSet. Returns working set in
 * the supplied output buffers if non-NULL. */
int IpoptSolveWarmStart(
    IpoptProblem problem,
    Number *x, Number *g, Number *obj_val,
    Number *mult_g, Number *mult_x_L, Number *mult_x_U,
    const IpoptBoundStatus *bound_status_in,   /* in, or NULL */
    const IpoptConsStatus  *cons_status_in,    /* in, or NULL */
    IpoptBoundStatus *bound_status_out,        /* out, or NULL */
    IpoptConsStatus  *cons_status_out,         /* out, or NULL */
    UserDataPtr user_data
);
```

`IpoptProblem` (`lib.rs:67`) gains an internal `Option<WorkingSet>`
slot; the existing `IpoptSolve` signature is unchanged. The C ABI
adds three symbols; existing cyipopt / JuMP / AMPL clients are
unaffected.

**(c) Suboption strings.** Already covered by §7.1's option registry
via the existing `AddIpoptStrOption` / `AddIpoptIntOption` /
`AddIpoptNumOption` setters; no new C signatures required.

### 7.3 Python (`crates/pounce-py/`)

PyO3 bindings extend symmetrically. `pounce.Problem.add_option`
already accepts `algorithm` and the suboption strings from §7.1; no
binding change.

**New methods on `PyProblem`** (`crates/pounce-py/src/problem.rs`):

```python
class Problem:
    # existing ────────────────────────────────────────
    def add_option(self, name: str, value): ...
    def solve(self, x0,
              lagrange=None, zl=None, zu=None,
              # NEW kwargs, default None ⇒ cold:
              working_set: Optional[WorkingSet] = None
              ) -> SolveResult: ...

    # NEW ─────────────────────────────────────────────
    def get_working_set(self) -> WorkingSet: ...

@dataclass
class WorkingSet:
    bounds:      np.ndarray  # dtype=int8, length n
    constraints: np.ndarray  # dtype=int8, length m

@dataclass
class SolveResult:
    x: np.ndarray
    obj_val: float
    mult_g: np.ndarray
    mult_x_L: np.ndarray
    mult_x_U: np.ndarray
    working_set: Optional[WorkingSet]  # populated when algorithm == "active-set-sqp"
    info: dict
```

The MPC / parametric-continuation Python idiom becomes:

```python
prob = pounce.Problem(...)
prob.add_option("algorithm", "active-set-sqp")
prob.add_option("sqp_warm_start_working_set", True)

ws = None
for step in range(horizon):
    res = prob.solve(x0=x_prev, working_set=ws)
    ws  = res.working_set        # carry across solves
    x_prev = shift(res.x)
```

This is the **same** ergonomics as qpOASES's Python binding, deliberately.

### 7.4 GAMS (`gams/gams_pounce.c`)

GAMS is the hardest case because the link is single-shot per `solve`
statement and there is no in-process persistence between solves. Two
mechanisms cover the use cases:

**(a) Algorithm and suboption selection** via the existing
`pounce.opt` option file (`gams_pounce.c:220-273`). No code change —
the option file already forwards unknown keys to the C API via
`AddIpoptStrOption` etc. Adding the keys from §7.1 to the documented
GAMS options list is the only deliverable here:

```
* pounce.opt
algorithm active-set-sqp
sqp_globalization filter
sqp_hessian exact
sqp_warm_start_working_set yes
```

**(b) Working-set transfer across solves.** GAMS has no native
discrete-multiplier carry. Two mechanisms, both standard in GAMS
solver links:

1. **Marginal-based reconstruction** (the GAMS-native idiom). After
   a solve, GAMS variable `.m` (marginal) holds the bound multiplier
   and equation `.m` holds the constraint multiplier. The next
   solve's link reads these and reconstructs an approximate working
   set by sign + tolerance test:
   `bound_status[i] = AtLower if x.m[i] > tol else (AtUpper if x.m[i] < -tol else Inactive)`.
   This is **lossy** (degenerate cases ambiguous) but matches what
   CONOPT, IPOPT, and KNITRO already do under GAMS. Implemented in
   `gams_pounce.c::pouCallSolver` (`:437`) prior to building the
   problem.
2. **Persistent state file** (the precise idiom). The solver writes
   a per-model state file (e.g. `.<modelname>.pou-ws`) at the end of
   each solve and reads it at the start of the next. The state file
   holds `(bound_status, cons_status)` as a small binary blob,
   keyed by the model's GMO checksum so a structural change
   invalidates it cleanly. The GAMS option `sqp_state_file` controls
   the path; absence means cold-start.

Both mechanisms ship in Phase 5c; mechanism 1 is the default (no
configuration required), mechanism 2 is opt-in for users who care
about precision in degenerate cases. Documented limitation: full
fidelity requires a GUSS-style scenario sweep within a single GAMS
session.

### 7.5 Interface summary

| Layer | Algorithm switch | Working-set in | Working-set out | Bridge |
|---|---|---|---|---|
| Rust | `AlgorithmChoice::ActiveSetSqp` | `SqpWarmStartIterateInitializer` | `SqpSolution.working` | direct |
| C ABI | `AddIpoptStrOption("algorithm", …)` | `IpoptSetWarmStartWorkingSet` | `IpoptGetWorkingSet` | thin shim |
| Python | `add_option("algorithm", …)` | `solve(…, working_set=ws)` | `res.working_set` | PyO3 over C ABI |
| GAMS | `pounce.opt` | marginals ‖ state file | marginals ‖ state file | C code in `pouCallSolver` |

## 8. Test harness

### 8.1 QP-level testing (Phase 5a)

- **Maros-Mészáros QP test set** (138 problems, the canonical sparse
  convex-and-indefinite QP benchmark). Format: `.qps` (a QP-extended
  MPS). New reader in `pounce-qp/src/maros.rs`; the format is
  closely related to the MPS reader already needed by pounce-cli for
  CUTEst, so most code is shared.
- **Reference oracle:** **qpOASES** for dense problems (Ferreau 2014;
  exposes a C API we can FFI), **OSQP** for sparse convex problems
  (Stellato 2020; widely available, sparse, Python bindings).
  Tolerance: 1e-6 relative on objective, 1e-7 on KKT residual.
- **Synthetic warm-start sequence:**
  - 50 QPs from a randomized MPC quadrotor model (Frison-Diehl
    2020 §6 spec, n=20, m=40, horizon=20).
  - 50 QPs from a parametric Wachter-Biegler test (Wachter-Biegler
    2006 problem set §5, perturbation of bounds).
  - **Exit criterion:** ≥3× iteration-count reduction vs cold,
    matching the qpOASES paper's reported speedup (Ferreau 2014
    Tab. 3-4 reports 5–50× on similar MPC sequences).

### 8.2 NLP-level testing (Phase 5b–5c)

- **CUTEst small/medium NLP subset** (the existing pounce CUTEst
  benchmark harness in `benchmarks/cutest/`). Compare iteration
  counts and timing against filterSQP and SNOPT (published numbers
  in Wachter-Biegler 2006 Tab. 5).
- **Warm-start regression suite:**
  - **MPC:** quadrotor / autonomous-vehicle test sequences from
    `acados` example set (Verschueren et al. 2022). 100-step
    closed-loop.
  - **Parametric:** the Wachter-Biegler 2006 §5 parametric NLP
    sequence.
  - **MINLP:** the `minlplib` instances tagged "QP relaxation
    sequence" (Bussieck et al. 2003).

Each test suite committed alongside the phase it gates.

### 8.3 Unit tests

For each module in `pounce-qp/src/`:
- **`schur.rs`:** Hadamard update validated against full refactor.
- **`expand.rs`:** anti-cycling on Beale's cycling LP example.
- **`elastic.rs`:** feasibility detection on infeasible Maros-
  Mészáros subset.
- **`working_set.rs`:** add/drop sequences validated against full
  KKT solves.

## 9. Per-workload notes

### 9.1 MPC

- Block-shift working-set carry: `𝒲_{k+1}[i] = 𝒲_k[i+1]` with new
  terminal stage seeded cold. Modeling-layer convention; the solver
  only needs the warm-start API to be cheap.
- The qpOASES paper (Ferreau 2014 Tab. 3) reports the homotopy
  completing in 1–3 working-set changes per shift in the well-warm-
  started regime. This is the headline benchmark for Phase 5c.

### 9.2 MINLP branch-and-bound

- Sibling/child relaxations differ in one bound. The previous solve's
  𝒲 is feasible for the child unless the bound change invalidates
  it; then one active-set update fixes it. Documented in Pirnay-Lopez-
  Negrellos-Wachter (2012) §4 for IPM warm start; the active-set
  numbers are categorically better.

### 9.3 Parametric homotopy / continuation

- Step in parameter `t`: `min f(x; t) s.t. g(x; t) ≤ 0`.
- Predictor: `pounce-sensitivity` computes `dx/dt`, `dλ/dt` from the
  reduced Hessian at the previous solution. Reuse unchanged.
- Corrector: one SQP solve from `(x + Δt·dx/dt, λ + Δt·dλ/dt,
  𝒲_prev)`. If 𝒲_prev is still optimal, one QP iteration.
- This is the workload where SQP outperforms a well-warm-started IPM
  most clearly. Cleanest demo target.

## 10. Phasing

The roadmap places this whole effort at Phase 5
(`future-work-roadmap.md:398-401`). With the literature pinned, the
phasing tightens to four shippable milestones:

- **Phase 5a — `pounce-qp` standalone (4–6 weeks).** New crate with
  the §4.2 sparse parametric active-set algorithm, §4.3 elastic
  phase-1, §4.4 EXPAND, §4.5 inertia control, §4.7 iterative
  refinement. Maros-Mészáros reader and oracle harness from §8.1.
  **Exit:** match qpOASES (dense) and OSQP (sparse convex) on
  Maros-Mészáros to 1e-6; ≥3× cold-vs-warm iteration speedup on the
  synthetic MPC and parametric sequences.
- **Phase 5b — SQP NLP driver, cold (3–4 weeks).** `SqpAlgorithm`
  wired into `alg_builder.rs` via §7.1 `AlgorithmChoice`. Filter
  globalization (§4.1) reusing existing `FilterLsAcceptor`. Exact
  Hessian (§4.6); damped BFGS and L-BFGS opt-in. Cold-start only.
  **Exit:** convergence on the CUTEst small/medium NLP subset; total
  iteration counts within 30 % of filterSQP / SNOPT published numbers
  on the same problems.
- **Phase 5c — Working-set warm start + integration (2–3 weeks).**
  - SQP warm-start API per §6.
  - C API additions per §7.2.
  - Python additions per §7.3.
  - GAMS additions per §7.4 (both mechanisms).
  - Parametric integration with `pounce-sensitivity`: SQP corrector
    after the sensitivity predictor.
  - **Exit:** measured ≥5× iteration-count drop on the §8.2 MPC and
    parametric regression suites; functioning end-to-end demos in
    `python/examples/` and `gams/examples/`.
- **Phase 5d — l1-elastic globalization alternative (1–2 weeks,
  optional).** SNOPT-style globalization as an opt-in alongside the
  default filter, behind `sqp_globalization=l1-elastic`. Comparison
  benchmarks committed.

Total: 10–15 weeks of focused work, gated phase-by-phase. Phases 5a
and 5b each have standalone value (sparse QP solver; cold SQP NLP);
5c is where the warm-start payoff lands; 5d is comparison work.

## 11. Risk

- **Maintenance.** Two solver paths is a permanent maintenance
  liability. Mitigation: SQP shares the `IpoptNlp`, derivative,
  scaling, options, journalist, conv-check, and Hessian layers
  unchanged; only the iteration skeleton + QP subproblem are net new.
- **Indefinite-Hessian failure modes.** Reduced-Hessian indefiniteness
  with bad scaling can defeat §4.5 inertia control. Mitigation: SR1
  fallback (§4.6) and the same `kappa_d` damping pounce already
  applies in `mu/adaptive.rs`.
- **Schur-complement growth.** If the working set changes O(n) times
  before a refactor, the dense Schur block becomes a cost concern.
  Mitigation: refactor cap `sqp_max_schur_updates` (default 50, §7.1);
  Davis 2006 §11 and Eldersveld-Saunders 1992 give empirical guidance.
- **GAMS state-transfer ambiguity.** Mechanism §7.4(a) is lossy on
  degenerate active sets. Mitigation: §7.4(b) state file as opt-in;
  documentation calling out the limitation.
- **Benchmark target completeness.** No MPC, MINLP, or parametric
  workload sits in `benchmarks/` today. Phase 5c ships with at least
  one each (§8.2) committed alongside.

## 12. Open questions for review

All algorithmic choices are now pinned. The remaining open questions
are scope and policy:

- **Hessian default for cold SQP.** Should Phase 5b default to exact
  Hessian or damped BFGS? Exact is fastest when reliable; damped BFGS
  is robust on hard nonconvex problems. Default proposal: exact, with
  damped-BFGS auto-fallback on three consecutive QP failures.
- **GAMS state-file format.** Binary (compact, fast) or text (greppable,
  diffable). Default proposal: binary with a versioned magic header
  and a JSON manifest sibling for diagnostics.
- **C API entry-point granularity.** Single `IpoptSolveWarmStart` with
  many parameters vs the three-call sequence `IpoptSet… / IpoptSolve
  / IpoptGet…`. Default proposal: ship both; the three-call sequence
  is the primitive, the one-shot is convenience.
- **`pounce-sensitivity` integration timing.** Phase 5a or 5c? Default
  proposal: 5c, so the parametric workload becomes a real end-to-end
  test rather than a unit-test stub.
- **Crate placement.** `pounce-qp` at the workspace root vs inside
  `crates/`. Default proposal: `crates/pounce-qp/` matching the
  existing convention.

## 13. References

### Algorithm — outer SQP

- Fletcher, Leyffer (2002), *Math. Prog.* **91**, 239–269 — filter SQP.
- Fletcher, Leyffer, Toint (2002), *SIAM J. Optim.* **13**, 44–59 —
  convergence of filter SQP.
- Wächter, Biegler (2005), *SIAM J. Optim.* **16**, 1–31 — filter
  line search.
- Wächter, Biegler (2006), *Math. Prog.* **106** — IPOPT reference.
- Nocedal, Wright, *Numerical Optimization* (2nd ed., Springer 2006),
  Ch. 16 (QP), Ch. 18 (SQP).

### Algorithm — QP subproblem

- Ferreau, Kirches, Potschka, Bock, Diehl (2014), *Math. Prog. Comp.*
  **6**, 327–363 — qpOASES, dense parametric active set.
- Kirches (2011), *Fast Numerical Methods for Mixed-Integer Nonlinear
  Model-Predictive Control*, Vieweg+Teubner — sparse Schur-complement
  extension; the canonical reference for §4.2.
- Janka, Kirches, Sager, Schlöder (2016), *Math. Prog. Comp.* **8**,
  435–459 — block-sparse SR1/BFGS SQP.
- Goldfarb, Idnani (1983), *Math. Prog.* **27** — dual active-set for
  convex QP (competing family).
- Gill, Murray, Saunders (2005), *SIAM Rev.* **47**, 99–131 — SNOPT.
- Gould, Hribar, Nocedal (2001), *SIAM J. Sci. Comput.* **23**,
  1376–1395 — null-space, indefinite Hessian (§4.5).
- Stellato, Banjac, Goulart, Bemporad, Boyd (2020), *Math. Prog.
  Comp.* **12** — OSQP (operator-splitting alternative).

### Algorithm — sparse linear algebra and updates

- Bartels (1971), *Numer. Math.* **16** — basis-update lineage.
- Reid (1982), *Math. Prog.* **24** — Bartels-Golub-Reid sparse
  variant.
- Eldersveld, Saunders (1992), *SIAM J. Matrix Anal. Appl.* **13** —
  block-LU update used for the Schur complement.
- Davis, *Direct Methods for Sparse Linear Systems* (SIAM 2006) —
  fill-in analysis.

### Algorithm — anti-cycling, elastic mode, inertia

- Gill, Murray, Saunders, Wright (1989), *Math. Prog.* **45**,
  437–474 — EXPAND.
- Gill, Murray, Saunders (2008), *User's Guide for SQOPT 7.7* —
  l1-elastic mode.
- Friedlander, Saunders (2005), *SIAM J. Optim.* **15** — elastic
  globalization.
- Gould (1999), *SIAM J. Optim.* **9**, 1041–1063 — modified
  factorizations.
- Forsgren (2002), *Appl. Num. Math.* **43** — inertia control.

### Algorithm — Hessian approximation

- Powell (1978), in *Numerical Analysis Dundee 1977* — damped BFGS
  for SQP.
- Liu, Nocedal (1989), *Math. Prog.* **45**, 503–528 — L-BFGS.
- Byrd, Nocedal, Schnabel (1994), *Math. Prog.* **63** — compact
  representations.

### Test harness and benchmarks

- Maros, Mészáros (1999), *Optim. Methods Softw.* **11/12** —
  Maros-Mészáros QP test set.
- Verschueren et al. (2022), *Math. Prog. Comp.* **14** — `acados`
  MPC benchmark suite.
- Bussieck, Drud, Meeraus (2003), *INFORMS J. Comp.* **15** —
  MINLPLib.
- Pirnay, López-Negrete, Wächter (2012), *Math. Prog. Comp.* **4** —
  warm-start within IPM (comparison baseline).

### Roadmap context

- `docs/research/future-work-roadmap.md:185-206` — the C1 entry this
  note operationalizes.
- `docs/research/composite-step-byrd-omojokun.md` — sister design note
  for C3 (the trust-region globalization track).
- `docs/research/interior-cg-matrix-free.md` — sister design note for
  C5 (the Krylov-KKT track).
