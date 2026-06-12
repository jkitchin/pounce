//! l1-elastic mode (§4.3 of the design note).
//!
//! Each general-constraint row gets two non-negative elastic
//! slacks `v_l, v_u ≥ 0` and a penalty `γ·(v_l + v_u)` in the
//! objective. The augmented row reads
//!
//! ```text
//!     bl_i ≤ a_iᵀ x + v_l_i − v_u_i ≤ bu_i
//! ```
//!
//! so the augmented QP is always feasible (any infeasibility in
//! the original is absorbed into the slacks). At the augmented
//! optimum:
//!
//! * all slacks zero ⇒ original QP is feasible; the recovered
//!   `(x, λ_g, λ_x)` is the original solution to within the
//!   penalty's distortion (`O(1/γ)`).
//! * some slack non-zero ⇒ original QP is infeasible; the slack
//!   values certify the minimal l1 violation per row.
//!
//! The augmented Hessian carries the original `H` on the `x`
//! block and **zero on the slack diagonals**. The saddle-point
//! inertia theorem (Wright; Nocedal-Wright §16.5) still gives
//! inertia `(n_aug, k_active, 0)` provided the reduced Hessian
//! on `null(A_combined)` is positive definite — which the cold-
//! init below arranges by activating either the constraint or
//! the corresponding slack lower bound for every row.
//!
//! Reference: Gill-Murray-Saunders, *User's Guide for SQOPT
//! 7.7*, §4 (elastic-mode formulation).

use crate::problem::HessianInertia;
use crate::working_set::{BoundStatus, ConsStatus, WorkingSet};
use crate::QpProblem;
use pounce_common::types::{Index, Number, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use std::rc::Rc;

/// Owned augmented QP data for an l1-elastic reformulation. Holds
/// its own sparse matrices and bound vectors because the augmented
/// problem has a different sparsity pattern than the original; the
/// caller passes the augmented problem back through
/// `QpSolver::solve` as borrows into these fields.
pub struct ElasticReformulation {
    pub n_orig: usize,
    pub m_orig: usize,
    pub n_aug: usize,
    pub m_aug: usize,

    pub h_aug: SymTMatrix,
    pub g_aug: Vec<Number>,
    pub a_aug: GenTMatrix,
    pub bl_aug: Vec<Number>,
    pub bu_aug: Vec<Number>,
    pub xl_aug: Vec<Number>,
    pub xu_aug: Vec<Number>,

    pub gamma: Number,

    /// Inertia hint of the *original* `H`, captured at `build` time.
    /// The augmented Hessian is block-diag(`H_orig`, 0), so it shares
    /// `H_orig`'s definiteness category (adding zero slack diagonals
    /// never introduces negative curvature). `as_qp` propagates this
    /// to the augmented problem so an indefinite original is solved
    /// through the inertia-control path instead of being silently
    /// treated as PSD (L15).
    orig_inertia: HessianInertia,
}

impl ElasticReformulation {
    /// Build the augmented problem from `qp`. Each of the `m_orig`
    /// rows gains one lower-side slack `v_l_i` at index `n_orig + i`
    /// and one upper-side slack `v_u_i` at index `n_orig + m_orig
    /// + i`.
    pub fn build(qp: &QpProblem<'_>, gamma: Number) -> Self {
        let n = qp.n;
        let m = qp.m;
        let n_aug = n + 2 * m;
        let m_aug = m;

        // ---- H_aug = block-diag(H_orig, 0_{2m}) ----
        // Reuse H_orig's sparsity pattern unchanged on the (1..n,
        // 1..n) block. Slack diagonals stay implicit zeros — no
        // entries needed (FERAL treats missing diagonal entries as
        // zero).
        let h_irows: Vec<Index> = qp.h.irows().to_vec();
        let h_jcols: Vec<Index> = qp.h.jcols().to_vec();
        let h_vals: Vec<Number> = qp.h.values().to_vec();
        let h_space = SymTMatrixSpace::new(n_aug as Index, h_irows, h_jcols);
        let mut h_aug = SymTMatrix::new(Rc::clone(&h_space));
        h_aug.set_values(&h_vals);

        // ---- A_aug: original A with two extra entries per row ----
        let na_orig = qp.a.nonzeros() as usize;
        let na_aug = na_orig + 2 * m;
        let mut a_irows = Vec::with_capacity(na_aug);
        let mut a_jcols = Vec::with_capacity(na_aug);
        let mut a_vals = Vec::with_capacity(na_aug);
        // Copy original entries as-is (1-based row in [1, m],
        // column in [1, n]).
        a_irows.extend_from_slice(qp.a.irows());
        a_jcols.extend_from_slice(qp.a.jcols());
        a_vals.extend_from_slice(qp.a.values());
        // Slack entries: row i (1-based) gains an entry at column
        // (n + i) with value +1 (lower slack) and one at column
        // (n + m + i) with value -1 (upper slack).
        let n_i = n as Index;
        let m_i = m as Index;
        for i in 1..=m_i {
            a_irows.push(i);
            a_jcols.push(n_i + i);
            a_vals.push(1.0);

            a_irows.push(i);
            a_jcols.push(n_i + m_i + i);
            a_vals.push(-1.0);
        }
        let a_space = GenTMatrixSpace::new(m_aug as Index, n_aug as Index, a_irows, a_jcols);
        let mut a_aug = GenTMatrix::new(Rc::clone(&a_space));
        a_aug.set_values(&a_vals);

        // ---- g_aug = (g_orig, γ·1, γ·1) ----
        let mut g_aug = vec![0.0; n_aug];
        g_aug[..n].copy_from_slice(qp.g);
        g_aug[n..n + m].fill(gamma);
        g_aug[n + m..].fill(gamma);

        // ---- Constraint bounds unchanged ----
        let bl_aug = qp.bl.to_vec();
        let bu_aug = qp.bu.to_vec();

        // ---- Variable bounds: x bounds unchanged; slack bounds [0, +∞) ----
        let mut xl_aug = vec![0.0; n_aug];
        let mut xu_aug = vec![NLP_UPPER_BOUND_INF; n_aug];
        xl_aug[..n].copy_from_slice(qp.xl);
        xu_aug[..n].copy_from_slice(qp.xu);
        // Slack lower bounds = 0 by the vec! init; upper = +∞ by init.

        Self {
            n_orig: n,
            m_orig: m,
            n_aug,
            m_aug,
            h_aug,
            g_aug,
            a_aug,
            bl_aug,
            bu_aug,
            xl_aug,
            xu_aug,
            gamma,
            orig_inertia: qp.hessian_inertia,
        }
    }

    /// Build a borrowed [`QpProblem`] over the owned augmented data
    /// so the standard solver path can consume it.
    pub fn as_qp(&self) -> QpProblem<'_> {
        QpProblem {
            n: self.n_aug,
            m: self.m_aug,
            h: &self.h_aug,
            g: &self.g_aug,
            a: &self.a_aug,
            bl: &self.bl_aug,
            bu: &self.bu_aug,
            xl: &self.xl_aug,
            xu: &self.xu_aug,
            // The augmented Hessian is PSD with explicit zero
            // entries on the slack diagonals; mark accordingly so
            // the inertia check skips its strict-PD assumption.
            hessian_inertia: match self.original_inertia() {
                HessianInertia::Psd | HessianInertia::Unknown => HessianInertia::Psd,
                HessianInertia::Indefinite => HessianInertia::Indefinite,
            },
        }
    }

    fn original_inertia(&self) -> HessianInertia {
        // The inertia hint of the original `H`, captured at `build`
        // time from `qp.hessian_inertia`. `as_qp` maps it onto the
        // augmented problem: `Psd`/`Unknown` collapse to `Psd` (the
        // augmented Hessian is PSD with explicit zero slack diagonals),
        // while `Indefinite` propagates so the augmented solve takes
        // the inertia-control path rather than assuming PSD.
        self.orig_inertia
    }

    /// Compute an initial primal-dual seed for the augmented
    /// problem that makes every augmented constraint feasible.
    /// Given a candidate `x_orig` (typically the projection of 0
    /// into the original variable box), compute the slack values
    /// that satisfy `bl ≤ a·x + v_l − v_u ≤ bu`, and build the
    /// matching working set: slack variables at their lower bound
    /// when they are zero, original variable bounds snapped to
    /// active where applicable, constraint rows marked active on
    /// the side that is currently binding (which is always one
    /// side once slacks absorb the violation).
    pub fn initial_seed(
        &self,
        qp: &QpProblem<'_>,
        x_orig: &[Number],
        feas_tol: Number,
    ) -> (Vec<Number>, WorkingSet) {
        let n = self.n_orig;
        let m = self.m_orig;
        let mut x_aug = vec![0.0; self.n_aug];
        x_aug[..n].copy_from_slice(x_orig);

        let mut working = WorkingSet::cold(self.n_aug, self.m_aug);

        // Compute A·x_orig once.
        let ax = crate::kkt::a_times_x(qp.a, x_orig, m);

        for i in 0..m {
            let l = qp.bl[i];
            let u = qp.bu[i];

            let mut v_l = 0.0;
            let mut v_u = 0.0;
            if l > NLP_LOWER_BOUND_INF && ax[i] < l {
                v_l = l - ax[i];
            }
            if u < NLP_UPPER_BOUND_INF && ax[i] > u {
                v_u = ax[i] - u;
            }
            x_aug[n + i] = v_l;
            x_aug[n + m + i] = v_u;

            // The augmented row value:
            //     lhs = a·x + v_l − v_u
            let lhs = ax[i] + v_l - v_u;

            // Choose the constraint-row working status:
            //   * if lhs == bl ⇒ AtLower (the slack v_l absorbed
            //     the violation and pushed the row to its lower
            //     bound, or it was already there);
            //   * if lhs == bu ⇒ AtUpper;
            //   * if bl == bu ⇒ Equality (always);
            //   * else Inactive.
            working.constraints[i] = if l == u {
                ConsStatus::Equality
            } else if l > NLP_LOWER_BOUND_INF && (lhs - l).abs() <= feas_tol {
                ConsStatus::AtLower
            } else if u < NLP_UPPER_BOUND_INF && (lhs - u).abs() <= feas_tol {
                ConsStatus::AtUpper
            } else {
                ConsStatus::Inactive
            };

            // Slack working statuses: a slack at zero sits at its
            // lower bound; a positive slack is interior.
            working.bounds[n + i] = if v_l == 0.0 {
                BoundStatus::AtLower
            } else {
                BoundStatus::Inactive
            };
            working.bounds[n + m + i] = if v_u == 0.0 {
                BoundStatus::AtLower
            } else {
                BoundStatus::Inactive
            };
        }

        // Original-variable bounds: snap to whichever bound x_orig
        // is exactly at (same convention as `solve_box_constrained`).
        for (i, &xi) in x_orig.iter().enumerate() {
            let l = qp.xl[i];
            let u = qp.xu[i];
            let l_finite = l > NLP_LOWER_BOUND_INF;
            let u_finite = u < NLP_UPPER_BOUND_INF;
            if l_finite && u_finite && (l - u).abs() <= feas_tol {
                working.bounds[i] = BoundStatus::Fixed;
                x_aug[i] = l;
            } else if l_finite && (xi - l).abs() <= feas_tol {
                working.bounds[i] = BoundStatus::AtLower;
                x_aug[i] = l;
            } else if u_finite && (xi - u).abs() <= feas_tol {
                working.bounds[i] = BoundStatus::AtUpper;
                x_aug[i] = u;
            }
        }

        (x_aug, working)
    }

    /// True if all elastic slacks at the augmented solution are
    /// within `feas_tol` of zero. False ⇒ the original QP is
    /// infeasible and the slack vector certifies the minimal
    /// per-row violation.
    pub fn is_feasible(&self, x_aug: &[Number], feas_tol: Number) -> bool {
        let n = self.n_orig;
        let m = self.m_orig;
        for i in 0..m {
            if x_aug[n + i] > feas_tol || x_aug[n + m + i] > feas_tol {
                return false;
            }
        }
        true
    }
}
