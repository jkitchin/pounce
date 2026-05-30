//! Convex QP problem data in standard form.
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equality,   m_eq rows)
//!             G x ≤ h          (inequality, m_ineq rows)
//! ```
//!
//! `x` is free; variable bounds are expressed as rows of `G`. `P` must
//! be symmetric positive semidefinite (convexity); it is supplied as its
//! **lower triangle** in triplet form. `A` and `G` are general sparse
//! triplets. This is the form the IPM in [`crate::ipm`] consumes, and
//! the form the `.nl` → QP extraction (Phase 2 dispatch) will target.

/// A sparse matrix entry `(row, col, val)`, 0-based.
#[derive(Debug, Clone, Copy)]
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

/// Convex QP in the standard form documented at the module level.
#[derive(Debug, Clone)]
pub struct QpProblem {
    /// Number of decision variables.
    pub n: usize,
    /// Lower triangle (row ≥ col) of the symmetric PSD Hessian `P`.
    pub p_lower: Vec<Triplet>,
    /// Linear objective coefficient `c` (length `n`).
    pub c: Vec<f64>,
    /// Equality matrix `A` (m_eq × n), full triplets.
    pub a: Vec<Triplet>,
    /// Equality right-hand side `b` (length m_eq).
    pub b: Vec<f64>,
    /// Inequality matrix `G` (m_ineq × n), full triplets.
    pub g: Vec<Triplet>,
    /// Inequality right-hand side `h` (length m_ineq).
    pub h: Vec<f64>,
}

impl QpProblem {
    pub fn m_eq(&self) -> usize {
        self.b.len()
    }

    pub fn m_ineq(&self) -> usize {
        self.h.len()
    }

    /// Public `y += P x` (full symmetric product from the stored lower
    /// triangle). Exposed so external callers — e.g. a TNLP adapter
    /// reusing the same problem data — can evaluate the objective
    /// gradient consistently with the solver.
    pub fn p_mul_add_pub(&self, x: &[f64], y: &mut [f64]) {
        self.p_mul_add(x, y);
    }

    /// Public `y += A x`.
    pub fn a_mul_add_pub(&self, x: &[f64], y: &mut [f64]) {
        self.a_mul_add(x, y);
    }

    /// `y += P x` using the stored lower triangle (mirrors the implicit
    /// upper triangle for off-diagonal entries).
    pub(crate) fn p_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.p_lower {
            y[t.row] += t.val * x[t.col];
            if t.row != t.col {
                y[t.col] += t.val * x[t.row];
            }
        }
    }

    /// `y += A x`.
    pub(crate) fn a_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.a {
            y[t.row] += t.val * x[t.col];
        }
    }

    /// `y += Aᵀ v`.
    pub(crate) fn at_mul_add(&self, v: &[f64], y: &mut [f64]) {
        for t in &self.a {
            y[t.col] += t.val * v[t.row];
        }
    }

    /// `y += G x`.
    pub(crate) fn g_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.g {
            y[t.row] += t.val * x[t.col];
        }
    }

    /// `y += Gᵀ v`.
    pub(crate) fn gt_mul_add(&self, v: &[f64], y: &mut [f64]) {
        for t in &self.g {
            y[t.col] += t.val * v[t.row];
        }
    }

    /// Public `y += A x` (alias of [`Self::a_mul_add`]).
    pub fn a_mul(&self, x: &[f64], y: &mut [f64]) {
        self.a_mul_add(x, y);
    }

    /// Public `y += G x` (alias of [`Self::g_mul_add`]).
    pub fn g_mul(&self, x: &[f64], y: &mut [f64]) {
        self.g_mul_add(x, y);
    }

    /// Public `y += Aᵀ v` (alias of [`Self::at_mul_add`]).
    pub fn at_mul(&self, v: &[f64], y: &mut [f64]) {
        self.at_mul_add(v, y);
    }

    /// Public `y += Gᵀ v` (alias of [`Self::gt_mul_add`]).
    pub fn gt_mul(&self, v: &[f64], y: &mut [f64]) {
        self.gt_mul_add(v, y);
    }

    /// Public `y += P x` (alias of [`Self::p_mul_add`]).
    pub fn p_mul(&self, x: &[f64], y: &mut [f64]) {
        self.p_mul_add(x, y);
    }
}

/// Termination status of an IPM solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// Converged: KKT residuals and duality gap below tolerance.
    Optimal,
    /// Primal infeasible: no `x` satisfies `Ax = b, Gx ≤ h`. A Farkas
    /// certificate `(y, z ≥ 0)` with `Aᵀy + Gᵀz ≈ 0` and `bᵀy + hᵀz < 0`
    /// was detected and verified.
    PrimalInfeasible,
    /// Dual infeasible / unbounded below: a recession direction `d` with
    /// `Pd ≈ 0, Ad = 0, Gd ≤ 0, cᵀd < 0` was detected and verified.
    DualInfeasible,
    /// Iteration limit reached before convergence.
    IterationLimit,
    /// The KKT factorization failed (e.g. structurally singular system).
    NumericalFailure,
}

/// Result of an IPM solve: the primal/dual solution and status.
#[derive(Debug, Clone)]
pub struct QpSolution {
    pub status: QpStatus,
    /// Primal solution `x` (length `n`).
    pub x: Vec<f64>,
    /// Equality multipliers `y` (length m_eq).
    pub y: Vec<f64>,
    /// Inequality multipliers `z ≥ 0` (length m_ineq).
    pub z: Vec<f64>,
    /// Objective value `½ xᵀP x + cᵀx`.
    pub obj: f64,
    /// Iterations taken.
    pub iters: usize,
}
