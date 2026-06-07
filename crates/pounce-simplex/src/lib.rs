//! `pounce-simplex` — a bounded-variable revised simplex LP solver in pure Rust.
//!
//! POUNCE's spatial branch-and-bound optimizer spends most of its per-node time
//! in optimization-based bound tightening (OBBT): `2n` LPs per node that share
//! one polytope and differ only in the objective (minimize then maximize each
//! variable). An interior-point method re-walks the central path from scratch
//! for every one of them. A simplex method instead warm-starts from the previous
//! optimal basis and reaches the next optimum in a handful of pivots — the
//! genuine warm-start lever for that inner loop, and the same lever for the
//! small bound changes between a parent box and its child.
//!
//! ```text
//! minimize    cᵀ x
//! subject to  A x = b
//!             l ≤ x ≤ u        (bounds may be ±∞; pass inequalities as slacks)
//! ```
//!
//! Two-phase bounded-variable revised primal simplex. The basis engine is
//! isolated behind the `basis::BasisEngine` trait: the production path
//! (Phase 6.2) keeps a faer sparse LU of the base basis and applies per-pivot
//! product-form (eta) updates on top of it, refactoring periodically. The
//! original Phase 6.1 explicit-dense-inverse engine is retained under
//! `cfg(test)` as the correctness oracle.
//!
//! Two warm-start levers re-use that basis across the OBBT inner loop without a
//! cold Phase I/II. [`Simplex::solve_objective`] handles an objective flip
//! within one node (the basis stays *primal* feasible, so only Phase II reruns).
//! [`Simplex::solve_bounds`] (Phase 6.3) handles a parent→child *bound* change:
//! tightening bounds leaves the basis *dual* feasible, so a bounded-variable
//! dual simplex restores primal feasibility in a few pivots. The OBBT wiring
//! (Phase 6.4) builds on these.

mod basis;
mod simplex;

// The dense LU is now only the factorization kernel behind the `DenseBasis`
// test oracle (see `basis`); the production path uses faer's sparse LU.
#[cfg(test)]
mod lu;

pub use simplex::Simplex;

/// A sparse matrix entry, `A[row][col] = val`. Duplicate `(row, col)` entries
/// are summed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

impl Triplet {
    pub fn new(row: usize, col: usize, val: f64) -> Self {
        Triplet { row, col, val }
    }
}

/// Magnitude at or beyond which a bound is treated as infinite. Mirrors the
/// `pounce-convex` `BOUND_INF` convention so callers can hand bounds straight
/// through without re-mapping their sentinels.
pub const BOUND_INF: f64 = 1e20;

/// A linear program in equality-plus-bounds standard form:
/// `min cᵀx  s.t.  A x = b,  lb ≤ x ≤ ub`. Inequality constraints must be
/// converted to equalities with explicit slack columns before solving.
#[derive(Clone, Debug)]
pub struct LpProblem {
    /// Number of structural variables.
    pub n: usize,
    /// Number of equality rows.
    pub m: usize,
    /// Objective coefficients (length `n`).
    pub c: Vec<f64>,
    /// Constraint matrix in triplet form.
    pub a: Vec<Triplet>,
    /// Right-hand side (length `m`).
    pub b: Vec<f64>,
    /// Lower bounds (length `n`; `≤ -BOUND_INF` means `-∞`).
    pub lb: Vec<f64>,
    /// Upper bounds (length `n`; `≥ BOUND_INF` means `+∞`).
    pub ub: Vec<f64>,
}

/// Terminal status of an LP solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LpStatus {
    /// An optimal vertex was found; `x`/`obj` are valid.
    Optimal,
    /// The constraints and bounds admit no feasible point.
    Infeasible,
    /// The objective decreases without bound over the feasible region.
    Unbounded,
    /// The pivot cap was hit before convergence (should not happen on
    /// well-posed inputs; report and fall back).
    IterationLimit,
    /// The basis went singular to working precision; the result is unusable.
    NumericalFailure,
}

/// The result of an LP solve. `x` is the structural solution (length `n`); on a
/// non-`Optimal` status it holds the last iterate and `obj` is `NaN`.
#[derive(Clone, Debug)]
pub struct LpSolution {
    pub status: LpStatus,
    pub x: Vec<f64>,
    pub obj: f64,
}

/// Solve an [`LpProblem`] with the bounded-variable revised simplex (cold start).
pub fn solve_lp(prob: &LpProblem) -> LpSolution {
    simplex::solve(prob)
}
