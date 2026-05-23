//! KKT-system assembly from a [`QpProblem`].
//!
//! For an equality-constrained QP `min ½xᵀHx + gᵀx s.t. Ax = b`, the
//! KKT system is the symmetric saddle-point matrix
//!
//! ```text
//!     ┌ H   Aᵀ ┐ ┌ x ┐   ┌ -g ┐
//!     │        │ │   │ = │    │
//!     └ A    0 ┘ └ λ ┘   └  b ┘
//! ```
//!
//! with Lagrangian sign convention `L = ½xᵀHx + gᵀx + λᵀ(Ax − b)` (so
//! `∇_x L = Hx + g + Aᵀλ`).
//!
//! The assembly emits triplets in the format
//! [`EMatrixFormat::TripletFormat`](pounce_linsol::EMatrixFormat::TripletFormat)
//! that FERAL / MA57 / MUMPS consume: **lower triangle, 1-based
//! indices**. Pounce-linalg's `SymTMatrix` stores one of each
//! symmetric pair (upper *or* lower; the convention is not pinned),
//! so we normalize each H entry to `irow ≥ jcol` defensively. A
//! entries land at rows `(n + i)`, which is automatically below the
//! diagonal of the augmented matrix and therefore lower-triangular
//! without further work.
//!
//! Phase 5a commit 2 covers the equality-only path. The inequality-
//! handling (bounds + general one-sided constraints + working-set
//! adds/drops) lands with the §4.2 homotopy machinery.

use crate::problem::QpProblem;
use pounce_common::{Index, Number};

/// A KKT system in FERAL-compatible triplet form.
///
/// `dim` is the dimension of the full symmetric matrix
/// (`n + n_active`, where `n_active` is the number of active rows of
/// `A` — currently equal to `m` since only equality QPs are
/// supported). `irn`, `jcn`, `vals` are parallel arrays describing
/// the lower-triangle nonzeros in 1-based indexing.
#[derive(Debug, Clone)]
pub struct KktTriplet {
    pub dim: usize,
    pub irn: Vec<Index>,
    pub jcn: Vec<Index>,
    pub vals: Vec<Number>,
}

impl KktTriplet {
    /// Assemble the equality-only KKT matrix `[H Aᵀ; A 0]` for the
    /// QP. Caller must have validated the QP (see
    /// [`QpProblem::validate`]) and verified
    /// `is_pure_equality_no_bounds(qp)`.
    pub fn assemble_equality_only(qp: &QpProblem) -> Self {
        let n = qp.n;
        let m = qp.m;
        let dim = n + m;

        let nh = qp.h.nonzeros() as usize;
        let na = qp.a.nonzeros() as usize;
        let cap = nh + na;

        let mut irn = Vec::with_capacity(cap);
        let mut jcn = Vec::with_capacity(cap);
        let mut vals = Vec::with_capacity(cap);

        // ---- H block (top-left), lower-triangle normalized ----
        let h_irows = qp.h.irows();
        let h_jcols = qp.h.jcols();
        let h_vals = qp.h.values();
        for k in 0..nh {
            let i = h_irows[k];
            let j = h_jcols[k];
            let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
            irn.push(hi);
            jcn.push(lo);
            vals.push(h_vals[k]);
        }

        // ---- A block (bottom-left), rows shifted by n ----
        let a_irows = qp.a.irows();
        let a_jcols = qp.a.jcols();
        let a_vals = qp.a.values();
        let n_i = n as Index;
        for k in 0..na {
            irn.push(n_i + a_irows[k]);
            jcn.push(a_jcols[k]);
            vals.push(a_vals[k]);
        }

        // ---- (2,2) zero block is implicit ----

        Self {
            dim,
            irn,
            jcn,
            vals,
        }
    }
}

/// Right-hand side `[-g; b]` for the equality-only KKT.
///
/// For each general-constraint row the target value `b` is taken
/// from `bl` (caller has already verified `bl[i] == bu[i]`, which is
/// the equality-only contract).
pub fn rhs_equality_only(qp: &QpProblem) -> Vec<Number> {
    let mut rhs = Vec::with_capacity(qp.n + qp.m);
    rhs.extend(qp.g.iter().map(|&gi| -gi));
    rhs.extend_from_slice(qp.bl);
    rhs
}

/// Is this QP in the equality-only / no-variable-bounds subset that
/// the commit-2 fast path can solve directly? Caller routes to it
/// when the predicate holds.
///
/// Concretely:
/// * every general-constraint row is an equality (`bl[i] == bu[i]`);
/// * every variable is free (`xl[i] ≤ -1e19` and `xu[i] ≥ +1e19`,
///   matching the `NLP_*_BOUND_INF` convention pounce uses).
pub fn is_pure_equality_no_bounds(qp: &QpProblem) -> bool {
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
    for (&l, &u) in qp.bl.iter().zip(qp.bu.iter()) {
        if l != u {
            return false;
        }
    }
    for (&l, &u) in qp.xl.iter().zip(qp.xu.iter()) {
        if l > NLP_LOWER_BOUND_INF || u < NLP_UPPER_BOUND_INF {
            return false;
        }
    }
    true
}

/// Is this QP a pure-box problem (no general constraints)? Caller
/// routes to the box-constrained active-set path when this returns
/// true and the pure-equality predicate did not.
pub fn is_pure_box(qp: &QpProblem) -> bool {
    qp.m == 0
}

/// Are *all* general constraints equalities (`bl == bu`
/// element-wise)? Caller routes to the equality+bounds active-set
/// path when this returns true and the cheaper predicates above
/// did not.
pub fn is_all_equality_constraints(qp: &QpProblem) -> bool {
    qp.bl.iter().zip(qp.bu.iter()).all(|(&l, &u)| l == u)
}

/// Assemble `[H Eᵀ_W; E_W 0]` for a box-constrained QP where `E_W`
/// is the selection matrix for the currently-active bounds. Each
/// active bound contributes one unit row at the corresponding
/// variable's column. `active_bounds` lists the variable indices in
/// ascending order; the order determines the saddle-row order.
pub fn assemble_box_with_active(qp: &QpProblem, active_bounds: &[usize]) -> KktTriplet {
    let n = qp.n;
    let k = active_bounds.len();
    let dim = n + k;

    let nh = qp.h.nonzeros() as usize;
    let mut irn = Vec::with_capacity(nh + k);
    let mut jcn = Vec::with_capacity(nh + k);
    let mut vals = Vec::with_capacity(nh + k);

    // ---- H block ----
    let h_irows = qp.h.irows();
    let h_jcols = qp.h.jcols();
    let h_vals = qp.h.values();
    for k_h in 0..nh {
        let i = h_irows[k_h];
        let j = h_jcols[k_h];
        let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
        irn.push(hi);
        jcn.push(lo);
        vals.push(h_vals[k_h]);
    }

    // ---- E_W block: row (n+j+1), col (var+1), value 1 ----
    let n_i = n as Index;
    for (j, &var) in active_bounds.iter().enumerate() {
        irn.push(n_i + (j as Index) + 1);
        jcn.push((var as Index) + 1);
        vals.push(1.0);
    }

    KktTriplet {
        dim,
        irn,
        jcn,
        vals,
    }
}

/// `H · x` for a symmetric Hessian stored with one of each pair
/// (upper *or* lower triangle, never both — the pounce-linalg
/// convention).
pub fn h_times_x(h: &pounce_linalg::triplet::SymTMatrix, x: &[Number]) -> Vec<Number> {
    let n = h.space().dim() as usize;
    let mut out = vec![0.0; n];
    let irows = h.irows();
    let jcols = h.jcols();
    let vals = h.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        let v = vals[k];
        if i == j {
            out[i] += v * x[i];
        } else {
            out[i] += v * x[j];
            out[j] += v * x[i];
        }
    }
    out
}

/// `A · x` for a sparse general Jacobian.
pub fn a_times_x(a: &pounce_linalg::triplet::GenTMatrix, x: &[Number], m: usize) -> Vec<Number> {
    let mut out = vec![0.0; m];
    let irows = a.irows();
    let jcols = a.jcols();
    let vals = a.values();
    for k in 0..irows.len() {
        let i = (irows[k] - 1) as usize;
        let j = (jcols[k] - 1) as usize;
        out[i] += vals[k] * x[j];
    }
    out
}

/// Assemble `[H Aᵀ_W Eᵀ_W; A_W 0 0; E_W 0 0]` for an arbitrary
/// active set — both general constraints (`active_cons`, listing
/// the row indices of `qp.a` currently in the working set) and
/// variable bounds (`active_bounds`, listing the column indices
/// currently active). Both lists must be in ascending index order;
/// the order determines saddle-row layout.
///
/// This generalizes [`assemble_equality_plus_bounds`]: when every
/// constraint is an equality `active_cons` lists all `m` rows; for
/// general inequality QPs `active_cons` is a strict subset chosen
/// by the active-set inner loop.
pub fn assemble_active_set_kkt(
    qp: &QpProblem,
    active_cons: &[usize],
    active_bounds: &[usize],
) -> KktTriplet {
    let n = qp.n;
    let m = qp.m;
    let k_c = active_cons.len();
    let k_b = active_bounds.len();
    let dim = n + k_c + k_b;

    let nh = qp.h.nonzeros() as usize;
    let na = qp.a.nonzeros() as usize;
    let cap = nh + na + k_b;

    let mut irn = Vec::with_capacity(cap);
    let mut jcn = Vec::with_capacity(cap);
    let mut vals = Vec::with_capacity(cap);

    // ---- H block ----
    let h_irows = qp.h.irows();
    let h_jcols = qp.h.jcols();
    let h_vals = qp.h.values();
    for kk in 0..nh {
        let i = h_irows[kk];
        let j = h_jcols[kk];
        let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
        irn.push(hi);
        jcn.push(lo);
        vals.push(h_vals[kk]);
    }

    // ---- A_W block: only rows whose 0-based index appears in active_cons ----
    // Build a row-map: for each problem-row, what's its saddle-row offset?
    // (None ⇒ row not in working set, skip its entries entirely.)
    let mut row_offset: Vec<Option<Index>> = vec![None; m];
    let n_i = n as Index;
    for (j, &row) in active_cons.iter().enumerate() {
        row_offset[row] = Some(n_i + (j as Index) + 1);
    }
    let a_irows = qp.a.irows();
    let a_jcols = qp.a.jcols();
    let a_vals = qp.a.values();
    for kk in 0..na {
        let a_row = (a_irows[kk] - 1) as usize;
        if let Some(saddle_row) = row_offset[a_row] {
            irn.push(saddle_row);
            jcn.push(a_jcols[kk]);
            vals.push(a_vals[kk]);
        }
    }

    // ---- E_W block: selection rows for active bounds ----
    let nm_i = (n + k_c) as Index;
    for (j, &var) in active_bounds.iter().enumerate() {
        irn.push(nm_i + (j as Index) + 1);
        jcn.push((var as Index) + 1);
        vals.push(1.0);
    }

    KktTriplet {
        dim,
        irn,
        jcn,
        vals,
    }
}

/// Assemble `[H Aᵀ_eq Eᵀ_W; A_eq 0 0; E_W 0 0]` for a QP whose
/// general constraints are all equalities and whose currently
/// active variable-bound working set is `active_bounds`
/// (ascending). Layout in K-row order:
///
/// * rows `1..=n`         — H rows
/// * rows `n+1..=n+m`     — `A_eq` rows
/// * rows `n+m+1..=n+m+k` — selection rows for active bounds
///
/// The off-diagonal blocks land at K-rows below the H block, so
/// each entry is lower-triangular by construction.
pub fn assemble_equality_plus_bounds(qp: &QpProblem, active_bounds: &[usize]) -> KktTriplet {
    let n = qp.n;
    let m = qp.m;
    let k = active_bounds.len();
    let dim = n + m + k;

    let nh = qp.h.nonzeros() as usize;
    let na = qp.a.nonzeros() as usize;
    let cap = nh + na + k;

    let mut irn = Vec::with_capacity(cap);
    let mut jcn = Vec::with_capacity(cap);
    let mut vals = Vec::with_capacity(cap);

    // ---- H block ----
    let h_irows = qp.h.irows();
    let h_jcols = qp.h.jcols();
    let h_vals = qp.h.values();
    for kk in 0..nh {
        let i = h_irows[kk];
        let j = h_jcols[kk];
        let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
        irn.push(hi);
        jcn.push(lo);
        vals.push(h_vals[kk]);
    }

    // ---- A_eq block at rows (n+1)..(n+m) ----
    let n_i = n as Index;
    let a_irows = qp.a.irows();
    let a_jcols = qp.a.jcols();
    let a_vals = qp.a.values();
    for kk in 0..na {
        irn.push(n_i + a_irows[kk]);
        jcn.push(a_jcols[kk]);
        vals.push(a_vals[kk]);
    }

    // ---- E_W block at rows (n+m+1)..(n+m+k) ----
    let nm_i = (n + m) as Index;
    for (j, &var) in active_bounds.iter().enumerate() {
        irn.push(nm_i + (j as Index) + 1);
        jcn.push((var as Index) + 1);
        vals.push(1.0);
    }

    KktTriplet {
        dim,
        irn,
        jcn,
        vals,
    }
}
