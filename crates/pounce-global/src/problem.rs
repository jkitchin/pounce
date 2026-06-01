//! The global-optimization problem: a factorable objective and constraints
//! over a variable box.
//!
//! ```text
//!   minimize    f(x)
//!   subject to  cl_j ≤ g_j(x) ≤ cu_j        (j = 0 … m−1)
//!               x_lo ≤ x ≤ x_hi
//! ```
//!
//! `f` and each `g_j` are [`FbbtTape`]s (build them with [`crate::expr`]).
//! Equality constraints are the `cl_j = cu_j` case.

use crate::expr::Expr;
use pounce_nlp::{ExpressionProvider, FbbtTape};

/// A constraint `lo ≤ g(x) ≤ hi`. Use `lo == hi` for an equality.
#[derive(Clone, Debug)]
pub struct Constraint {
    pub tape: FbbtTape,
    pub lo: f64,
    pub hi: f64,
}

/// A factorable global-optimization problem over a variable box.
#[derive(Clone, Debug)]
pub struct GlobalProblem {
    pub n_vars: usize,
    /// Variable lower/upper bounds (length `n_vars`). A bounded box is required
    /// for the relaxation — use a large finite value for "unbounded".
    pub x_lo: Vec<f64>,
    pub x_hi: Vec<f64>,
    pub objective: FbbtTape,
    pub constraints: Vec<Constraint>,
}

impl GlobalProblem {
    /// New box-constrained problem minimizing `objective`; add constraints with
    /// [`GlobalProblem::subject_to`] / [`GlobalProblem::equality`].
    pub fn new(x_lo: Vec<f64>, x_hi: Vec<f64>, objective: &Expr) -> Self {
        assert_eq!(x_lo.len(), x_hi.len(), "x_lo / x_hi length mismatch");
        GlobalProblem {
            n_vars: x_lo.len(),
            x_lo,
            x_hi,
            objective: objective.to_tape(),
            constraints: Vec::new(),
        }
    }

    /// Add `lo ≤ g(x) ≤ hi`.
    pub fn subject_to(mut self, g: &Expr, lo: f64, hi: f64) -> Self {
        self.constraints.push(Constraint {
            tape: g.to_tape(),
            lo,
            hi,
        });
        self
    }

    /// Add `g(x) ≤ ub`.
    pub fn le(self, g: &Expr, ub: f64) -> Self {
        self.subject_to(g, f64::NEG_INFINITY, ub)
    }

    /// Add `g(x) ≥ lb`.
    pub fn ge(self, g: &Expr, lb: f64) -> Self {
        self.subject_to(g, lb, f64::INFINITY)
    }

    /// Add `g(x) = rhs`.
    pub fn equality(self, g: &Expr, rhs: f64) -> Self {
        self.subject_to(g, rhs, rhs)
    }

    /// Per-constraint bounds as the parallel `(g_lo, g_hi)` vectors `run_fbbt`
    /// expects.
    pub(crate) fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        self.constraints.iter().map(|c| (c.lo, c.hi)).unzip()
    }

    /// Worst-case estimate, **before** solving, of the peak frontier memory in
    /// bytes for the given options. Each processed node pushes at most two
    /// children and pops one, so the best-first frontier holds at most
    /// `max_nodes + 1` open nodes; multiplied by the per-node footprint
    /// ([`crate::estimate_node_bytes`]) this bounds the search's resident
    /// memory. The actual peak (usually far smaller, since most nodes are
    /// pruned) is reported back in [`crate::GlobalSolution::peak_memory_bytes`].
    pub fn estimated_peak_memory_bytes(&self, opts: &crate::GlobalOptions) -> usize {
        opts.max_nodes
            .saturating_add(1)
            .saturating_mul(crate::estimate_node_bytes(self.n_vars))
    }

    /// The maximum constraint violation of a point `x` — `0` for a feasible
    /// point, else the largest `max(lo − g, g − hi)` over the constraints. The
    /// branch-and-bound analog of a primal-infeasibility residual: it measures
    /// how feasible the returned incumbent actually is.
    pub fn max_violation(&self, x: &[f64]) -> f64 {
        self.constraints.iter().fold(0.0_f64, |worst, c| {
            let g = *crate::ad::forward_vals(&c.tape, x)
                .last()
                .expect("a constraint tape has at least one op");
            let lo_viol = if c.lo.is_finite() { c.lo - g } else { 0.0 };
            let hi_viol = if c.hi.is_finite() { g - c.hi } else { 0.0 };
            worst.max(lo_viol).max(hi_viol)
        })
    }
}

/// Exposes a problem's constraint tapes to FBBT. `run_fbbt` only reads
/// constraint expressions (the objective is bounded by the relaxation, not
/// FBBT), so the objective tape is intentionally not surfaced here.
pub(crate) struct ConstraintProvider<'a> {
    constraints: &'a [Constraint],
}

impl<'a> ConstraintProvider<'a> {
    pub(crate) fn new(constraints: &'a [Constraint]) -> Self {
        ConstraintProvider { constraints }
    }
}

impl ExpressionProvider for ConstraintProvider<'_> {
    fn constraint_expression(&self, i: usize) -> Option<FbbtTape> {
        self.constraints.get(i).map(|c| c.tape.clone())
    }
}
