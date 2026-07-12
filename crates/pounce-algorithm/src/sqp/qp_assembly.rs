//! Build a [`pounce_qp::QpProblem`] from the NLP linearization at
//! the current SQP iterate `(x, λ_g)`.
//!
//! Standard SQP QP subproblem (Nocedal-Wright §18.1):
//!
//! ```text
//!     min  ½ pᵀ ∇²L(x, λ) p + ∇f(x)ᵀ p
//!     s.t.   bl_c ≤ c(x) + ∇c(x) p ≤ bu_c
//!            xl − x ≤ p ≤ xu − x
//! ```
//!
//! The QP's general bounds are shifted RHSs: `bl_qp = bl_c − c(x)`
//! and `bu_qp = bu_c − c(x)` (treating equalities as
//! `bl_c = bu_c = 0`). The QP's variable bounds are `xl − x` and
//! `xu − x`, so the QP primal `p` directly equals the SQP step.
//!
//! `SqpQpData` owns the sparse storage and exposes a borrowed
//! `QpProblem` view; this is the analog of
//! `pounce_qp::ElasticReformulation::as_qp`.

use pounce_common::types::{Index, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_qp::{HessianInertia, QpProblem};
use std::rc::Rc;

/// Owned linearization data for a single SQP iteration.
pub struct SqpQpData {
    pub n: usize,
    pub m: usize,

    pub h: SymTMatrix,
    pub g: Vec<Number>,
    pub a: GenTMatrix,
    pub bl: Vec<Number>,
    pub bu: Vec<Number>,
    pub xl: Vec<Number>,
    pub xu: Vec<Number>,
    pub hessian_inertia: HessianInertia,
}

/// Sparse-triplet view of a derivative matrix. Indices are
/// 1-based per the pounce-linalg convention; values are owned.
pub struct Triplet {
    pub n_rows: usize,
    pub n_cols: usize,
    pub irow: Vec<Index>,
    pub jcol: Vec<Index>,
    pub vals: Vec<Number>,
}

impl SqpQpData {
    /// Assemble from concrete linearization arrays at iterate `x`.
    ///
    /// * `grad_f` — `∇f(x)`, length `n`.
    /// * `c_vals` — `c(x)`, length `m` (may contain inequality
    ///   slack values too; convention is `bl_c[i] ≤ c[i] ≤ bu_c[i]`).
    /// * `bl_c`, `bu_c` — original NLP constraint bounds.
    /// * `xl_orig`, `xu_orig` — original NLP variable bounds.
    /// * `jac_c` — `∇c(x)` triplet, `m × n`.
    /// * `hess_lag` — `∇²L(x, λ_g)` triplet, `n × n` symmetric.
    pub fn build(
        x: &[Number],
        grad_f: &[Number],
        c_vals: &[Number],
        bl_c: &[Number],
        bu_c: &[Number],
        xl_orig: &[Number],
        xu_orig: &[Number],
        jac_c: Triplet,
        hess_lag: Triplet,
        hessian_inertia: HessianInertia,
    ) -> Self {
        let n = grad_f.len();
        let m = c_vals.len();
        assert_eq!(x.len(), n);
        assert_eq!(bl_c.len(), m);
        assert_eq!(bu_c.len(), m);
        assert_eq!(xl_orig.len(), n);
        assert_eq!(xu_orig.len(), n);
        assert_eq!(jac_c.n_rows, m);
        assert_eq!(jac_c.n_cols, n);
        assert_eq!(hess_lag.n_rows, n);
        assert_eq!(hess_lag.n_cols, n);

        let h_space = SymTMatrixSpace::new(n as Index, hess_lag.irow, hess_lag.jcol);
        let mut h = SymTMatrix::new(Rc::clone(&h_space));
        h.set_values(&hess_lag.vals);

        let a_space = GenTMatrixSpace::new(m as Index, n as Index, jac_c.irow, jac_c.jcol);
        let mut a = GenTMatrix::new(Rc::clone(&a_space));
        a.set_values(&jac_c.vals);

        // QP general bounds: bl_c − c(x), bu_c − c(x) — but
        // preserve ±∞ markers so the QP solver's one-sided
        // ratio test still treats them as unbounded.
        let mut bl = Vec::with_capacity(m);
        let mut bu = Vec::with_capacity(m);
        for i in 0..m {
            bl.push(shift_bound(bl_c[i], c_vals[i], true));
            bu.push(shift_bound(bu_c[i], c_vals[i], false));
        }

        // QP step bounds: xl_orig − x, xu_orig − x.
        let mut xl = Vec::with_capacity(n);
        let mut xu = Vec::with_capacity(n);
        for i in 0..n {
            xl.push(shift_bound(xl_orig[i], x[i], true));
            xu.push(shift_bound(xu_orig[i], x[i], false));
        }

        Self {
            n,
            m,
            h,
            g: grad_f.to_vec(),
            a,
            bl,
            bu,
            xl,
            xu,
            hessian_inertia,
        }
    }

    /// Borrowed `QpProblem` view ready for
    /// `pounce_qp::QpSolver::solve`.
    pub fn as_qp(&self) -> QpProblem<'_> {
        QpProblem {
            n: self.n,
            m: self.m,
            h: &self.h,
            g: &self.g,
            a: &self.a,
            bl: &self.bl,
            bu: &self.bu,
            xl: &self.xl,
            xu: &self.xu,
            hessian_inertia: self.hessian_inertia,
        }
    }
}

/// Shift a bound by subtracting the current value. Preserves
/// `NLP_*_BOUND_INF` sentinels so the QP solver's `is_finite`
/// checks still trigger correctly.
fn shift_bound(bound: Number, current: Number, is_lower: bool) -> Number {
    if is_lower {
        if bound <= NLP_LOWER_BOUND_INF {
            NLP_LOWER_BOUND_INF
        } else {
            bound - current
        }
    } else if bound >= NLP_UPPER_BOUND_INF {
        NLP_UPPER_BOUND_INF
    } else {
        bound - current
    }
}
