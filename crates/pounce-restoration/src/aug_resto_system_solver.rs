//! Restoration-aug system solver — port of
//! `Algorithm/IpAugRestoSystemSolver.{hpp,cpp}`.
//!
//! The 8-block restoration KKT system is reduced via Schur complement
//! to the original-NLP 4-block aug system, which is then handed off to
//! a wrapped inner [`pounce_algorithm::kkt::AugSystemSolver`]
//! (typically a `StdAugSystemSolver` driving MA57/MUMPS).
//!
//! Pounce represents the resto KKT in *flat* form (matching what
//! `RestoIpoptNlp` emits in v0.1):
//!
//! * `W` is a flat [`SymTMatrix`] at dim `n_total = n_orig + 2·m_eq +
//!   2·m_ineq`. All triplets live in `1..=n_orig` (orig Hessian +
//!   proximity diagonal `obj_factor·η(μ)·D_R²`).
//! * `J_c` is a flat [`GenTMatrix`] of shape `m_eq × n_total` with
//!   triplets `[orig_J_c | +I_{m_eq} | −I_{m_eq} | 0 | 0]`.
//! * `J_d` is a flat [`GenTMatrix`] of shape `m_ineq × n_total` with
//!   triplets `[orig_J_d | 0 | 0 | +I_{m_ineq} | −I_{m_ineq}]`.
//! * `D_x` is a 5-block [`CompoundVector`] `[σ_orig | σ_n_c | σ_p_c |
//!   σ_n_d | σ_p_d]`.
//! * `rhs_x` follows the same 5-block compound layout.
//! * `rhs_s`, `rhs_c`, `rhs_d`, and the `D_s`/`D_c`/`D_d` weights are
//!   flat dense vectors.
//! * `sol_x` is the same 5-block compound; `sol_s`/`sol_c`/`sol_d` are
//!   dense.
//!
//! Reduction (mirroring `IpAugRestoSystemSolver.cpp:60-307`):
//!
//! 1. `σ̃_{n_c}⁻¹ = 1 / (σ_{n_c} + δ_x)`, similarly for `p_c`, `n_d`,
//!    `p_d`.
//! 2. `D_cR = +σ̃_{n_c}⁻¹ + σ̃_{p_c}⁻¹ + D_c` (pounce sign convention,
//!    see note below).
//! 3. `D_dR = +σ̃_{n_d}⁻¹ + σ̃_{p_d}⁻¹ + D_d` (same).
//!
//! **Sign convention note.** Pounce's [`StdAugSystemSolver`] assembles
//! the (3,3) block as `−(D_c + δ_c)·I`, whereas upstream Ipopt's
//! `IpStdAugSystemSolver` assembles it as `D_c − δ_c·I`. So `D_c` in
//! pounce has the **opposite sign** of `D_c` in upstream. The Schur
//! correction added to (3,3) is `+σ̃⁻¹_n + σ̃⁻¹_p` (positive scalar),
//! and to *subtract* that from the effective (3,3) using pounce's
//! convention `effective = −(D_cR + δ_c)`, we need
//! `D_cR = +σ̃⁻¹_n + σ̃⁻¹_p`. Upstream's `Neg_Omega_c_plus_D_c` returns
//! the negation because its convention is `effective = D_cR − δ_c`.
//! 4. `rhs_xR = rhs_x.comp(0)` (orig block of the compound rhs).
//! 5. `rhs_cR = rhs_c − σ̃_{n_c}⁻¹ · rhs_{n_c} + σ̃_{p_c}⁻¹ · rhs_{p_c}`.
//! 6. `rhs_dR = rhs_d − σ̃_{n_d}⁻¹ · rhs_{n_d} + σ̃_{p_d}⁻¹ · rhs_{p_d}`.
//! 7. Hand the reduced 4-block system to the inner aug solver.
//! 8. Back-substitute the slack solutions:
//!      sol_n_c = σ̃_{n_c}⁻¹ · (rhs_{n_c} − sol_{y_c})
//!      sol_p_c = σ̃_{p_c}⁻¹ · (rhs_{p_c} + sol_{y_c})
//!      sol_n_d = σ̃_{n_d}⁻¹ · (rhs_{n_d} − sol_{y_d})
//!      sol_p_d = σ̃_{p_d}⁻¹ · (rhs_{p_d} + sol_{y_d})

use pounce_algorithm::kkt::aug_system_solver::{
    AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver,
};
use pounce_common::types::{Index, Number};
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linalg::{Matrix, Vector};
use pounce_linsol::ESymSolverStatus;
use std::rc::Rc;

/// Resto-side wrapper around an inner [`AugSystemSolver`].
pub struct AugRestoSystemSolver {
    inner: Box<dyn AugSystemSolver>,

    /// Pinned on the first solve so the inner solver's structure cache
    /// stays valid across calls.
    initialized: bool,
    n_orig: Index,
    m_eq: Index,
    m_ineq: Index,
    /// Number of orig-only triplets in the flat `J_c` (the prefix of
    /// `J_c.values()` that belongs to the orig Jacobian, before the
    /// `±I` slack columns).
    nz_jc_orig: usize,
    /// Same, for `J_d`.
    nz_jd_orig: usize,

    /// Reduced (orig-only) Hessian: dim `n_orig`, same triplet pattern
    /// as the flat resto `W` (which contains only rows/cols in
    /// `1..=n_orig`).
    h_orig: Option<SymTMatrix>,
    j_c_orig: Option<GenTMatrix>,
    j_d_orig: Option<GenTMatrix>,

    /// Cached spaces for the dense intermediates.
    space_m_eq: Option<Rc<DenseVectorSpace>>,
    space_m_ineq: Option<Rc<DenseVectorSpace>>,
}

impl std::fmt::Debug for AugRestoSystemSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AugRestoSystemSolver")
            .field("initialized", &self.initialized)
            .field("n_orig", &self.n_orig)
            .field("m_eq", &self.m_eq)
            .field("m_ineq", &self.m_ineq)
            .finish_non_exhaustive()
    }
}

impl AugRestoSystemSolver {
    pub fn new(inner: Box<dyn AugSystemSolver>) -> Self {
        Self {
            inner,
            initialized: false,
            n_orig: 0,
            m_eq: 0,
            m_ineq: 0,
            nz_jc_orig: 0,
            nz_jd_orig: 0,
            h_orig: None,
            j_c_orig: None,
            j_d_orig: None,
            space_m_eq: None,
            space_m_ineq: None,
        }
    }

    fn build_structure(&mut self, w: &SymTMatrix, j_c: &GenTMatrix, j_d: &GenTMatrix) {
        let m_eq = j_c.n_rows();
        let m_ineq = j_d.n_rows();
        let n_total = j_c.n_cols();
        let n_orig = n_total - 2 * m_eq - 2 * m_ineq;

        // Orig Hessian: every triplet of W has row/col in 1..=n_orig
        // (eval_h emits the orig Hessian + diagonal proximity term;
        // slack rows/cols are zero), so we can reuse the same
        // (irows, jcols) at dim n_orig.
        let h_space = SymTMatrixSpace::new(n_orig, w.irows().to_vec(), w.jcols().to_vec());
        self.h_orig = Some(SymTMatrix::new(h_space));

        // Orig J_c: take the leading `nz_jc_orig` triplets (columns
        // 1..=n_orig). The trailing 2·m_eq triplets are the ±I slack
        // columns and don't belong in the reduced matrix.
        let nz_jc_orig = (j_c.nonzeros() as usize).saturating_sub(2 * m_eq as usize);
        let jc_space = GenTMatrixSpace::new(
            m_eq,
            n_orig,
            j_c.irows()[..nz_jc_orig].to_vec(),
            j_c.jcols()[..nz_jc_orig].to_vec(),
        );
        self.j_c_orig = Some(GenTMatrix::new(jc_space));

        let nz_jd_orig = (j_d.nonzeros() as usize).saturating_sub(2 * m_ineq as usize);
        let jd_space = GenTMatrixSpace::new(
            m_ineq,
            n_orig,
            j_d.irows()[..nz_jd_orig].to_vec(),
            j_d.jcols()[..nz_jd_orig].to_vec(),
        );
        self.j_d_orig = Some(GenTMatrix::new(jd_space));

        self.space_m_eq = Some(DenseVectorSpace::new(m_eq));
        self.space_m_ineq = Some(DenseVectorSpace::new(m_ineq));
        self.n_orig = n_orig;
        self.m_eq = m_eq;
        self.m_ineq = m_ineq;
        self.nz_jc_orig = nz_jc_orig;
        self.nz_jd_orig = nz_jd_orig;
        self.initialized = true;
    }

    fn refill_values(&mut self, w: &SymTMatrix, j_c: &GenTMatrix, j_d: &GenTMatrix) {
        // Hessian: same triplet count as W (slack triplets are absent).
        let h_dst = self.h_orig.as_mut().unwrap().values_mut();
        h_dst.copy_from_slice(w.values());
        // J_c / J_d: copy only the orig prefix.
        let jc_dst = self.j_c_orig.as_mut().unwrap().values_mut();
        jc_dst.copy_from_slice(&j_c.values()[..self.nz_jc_orig]);
        let jd_dst = self.j_d_orig.as_mut().unwrap().values_mut();
        jd_dst.copy_from_slice(&j_d.values()[..self.nz_jd_orig]);
    }
}

impl AugSystemSolver for AugRestoSystemSolver {
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }

    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        self.inner.last_solve_status()
    }

    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        // ---- Downcast the flat resto matrices. ----
        let w = coeffs
            .w
            .expect("AugRestoSystemSolver: W must be present (resto Hessian)")
            .as_any()
            .downcast_ref::<SymTMatrix>()
            .expect("AugRestoSystemSolver: W must be a SymTMatrix");
        let j_c = coeffs
            .j_c
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("AugRestoSystemSolver: J_c must be a GenTMatrix");
        let j_d = coeffs
            .j_d
            .as_any()
            .downcast_ref::<GenTMatrix>()
            .expect("AugRestoSystemSolver: J_d must be a GenTMatrix");

        if !self.initialized {
            self.build_structure(w, j_c, j_d);
        }
        self.refill_values(w, j_c, j_d);

        let m_eq = self.m_eq as usize;
        let m_ineq = self.m_ineq as usize;

        let dbg = std::env::var("POUNCE_RESTO_DBG").is_ok();
        if dbg {
            tracing::debug!(target: "pounce::restoration",
                "[resto-aug] n_orig={} m_eq={} m_ineq={} W.nz={} J_c.nz={} J_d.nz={} delta_x={:.3e} delta_c={:.3e} delta_d={:.3e}",
                self.n_orig, self.m_eq, self.m_ineq,
                w.nonzeros(), j_c.nonzeros(), j_d.nonzeros(),
                coeffs.delta_x, coeffs.delta_c, coeffs.delta_d,
            );
        }

        // ---- σ vectors from D_x compound. ----
        let dx_compound = coeffs
            .d_x
            .expect("AugRestoSystemSolver: D_x must be present (5-block compound)")
            .as_any()
            .downcast_ref::<CompoundVector>()
            .expect("AugRestoSystemSolver: D_x must be a CompoundVector");
        debug_assert_eq!(dx_compound.n_comps(), 5);
        let sigma_orig_dyn = dx_compound.comp(0); // &dyn Vector, n_orig dim
        let sigma_n_c = dense_values(dx_compound.comp(1));
        let sigma_p_c = dense_values(dx_compound.comp(2));
        let sigma_n_d = dense_values(dx_compound.comp(3));
        let sigma_p_d = dense_values(dx_compound.comp(4));

        // ---- σ̃⁻¹ vectors. ----
        let dx = coeffs.delta_x;
        let sig_tilde_n_c_inv: Vec<Option<Number>> = sigma_n_c
            .iter()
            .map(|&s| sigma_tilde_inv_elem(Some(s), dx))
            .collect();
        let sig_tilde_p_c_inv: Vec<Option<Number>> = sigma_p_c
            .iter()
            .map(|&s| sigma_tilde_inv_elem(Some(s), dx))
            .collect();
        let sig_tilde_n_d_inv: Vec<Option<Number>> = sigma_n_d
            .iter()
            .map(|&s| sigma_tilde_inv_elem(Some(s), dx))
            .collect();
        let sig_tilde_p_d_inv: Vec<Option<Number>> = sigma_p_d
            .iter()
            .map(|&s| sigma_tilde_inv_elem(Some(s), dx))
            .collect();

        // ---- Reduced D_cR, D_dR. ----
        // Pounce convention: effective (3,3) block = −(D_cR + δ_c).
        // Schur correction adds +σ̃⁻¹_n + σ̃⁻¹_p to the matrix; in
        // pounce's encoding that means D_cR = +σ̃⁻¹_n + σ̃⁻¹_p (+ D_c
        // if upstream-side scaling is present, which is the same sign
        // since D_c has been negated relative to upstream).
        let d_c_vals: Option<Vec<Number>> = coeffs.d_c.map(dense_values);
        let mut d_c_r = vec![0.0; m_eq];
        for i in 0..m_eq {
            let n_term = sig_tilde_n_c_inv[i].unwrap_or(0.0);
            let p_term = sig_tilde_p_c_inv[i].unwrap_or(0.0);
            let d_term = d_c_vals.as_ref().map(|v| v[i]).unwrap_or(0.0);
            d_c_r[i] = n_term + p_term + d_term;
        }
        let mut d_c_r_dense = self.space_m_eq.as_ref().unwrap().make_new_dense();
        d_c_r_dense.set_values(&d_c_r);

        // D_d typically None for resto; same pounce-sign rule.
        let d_d_vals: Option<Vec<Number>> = coeffs.d_d.map(dense_values);
        let mut d_d_r = vec![0.0; m_ineq];
        for i in 0..m_ineq {
            let n_term = sig_tilde_n_d_inv[i].unwrap_or(0.0);
            let p_term = sig_tilde_p_d_inv[i].unwrap_or(0.0);
            let d_term = d_d_vals.as_ref().map(|v| v[i]).unwrap_or(0.0);
            d_d_r[i] = n_term + p_term + d_term;
        }
        let mut d_d_r_dense = self.space_m_ineq.as_ref().unwrap().make_new_dense();
        d_d_r_dense.set_values(&d_d_r);

        // ---- Reduced rhs_xR, rhs_cR, rhs_dR. ----
        let rhs_x_compound = rhs
            .rhs_x
            .as_any()
            .downcast_ref::<CompoundVector>()
            .expect("AugRestoSystemSolver: rhs_x must be a CompoundVector");
        debug_assert_eq!(rhs_x_compound.n_comps(), 5);
        let rhs_x_r_dyn = rhs_x_compound.comp(0);
        let rhs_n_c = dense_values(rhs_x_compound.comp(1));
        let rhs_p_c = dense_values(rhs_x_compound.comp(2));
        let rhs_n_d = dense_values(rhs_x_compound.comp(3));
        let rhs_p_d = dense_values(rhs_x_compound.comp(4));

        let rhs_c_vals = dense_values(rhs.rhs_c);
        let rhs_d_vals = dense_values(rhs.rhs_d);

        let mut rhs_c_r = vec![0.0; m_eq];
        for i in 0..m_eq {
            rhs_c_r[i] = rhs_cr_elem(
                rhs_c_vals[i],
                sig_tilde_n_c_inv[i],
                rhs_n_c[i],
                sig_tilde_p_c_inv[i],
                rhs_p_c[i],
            );
        }
        let mut rhs_c_r_dense = self.space_m_eq.as_ref().unwrap().make_new_dense();
        rhs_c_r_dense.set_values(&rhs_c_r);

        let mut rhs_d_r = vec![0.0; m_ineq];
        for i in 0..m_ineq {
            // rhs_dR = rhs_d − σ̃_{n_d}⁻¹ · rhs_{n_d} + σ̃_{p_d}⁻¹ · rhs_{p_d}
            // (Pd_L = +I, −Pd_U = −I in pounce's flat resto).
            let n_contrib = sig_tilde_n_d_inv[i].map(|s| s * rhs_n_d[i]).unwrap_or(0.0);
            let p_contrib = sig_tilde_p_d_inv[i].map(|s| s * rhs_p_d[i]).unwrap_or(0.0);
            rhs_d_r[i] = rhs_d_vals[i] - n_contrib + p_contrib;
        }
        let mut rhs_d_r_dense = self.space_m_ineq.as_ref().unwrap().make_new_dense();
        rhs_d_r_dense.set_values(&rhs_d_r);

        // ---- Reduced sol scratch. ----
        // sol_x_R lands in `sol.sol_x.comp(0)` directly — we hand
        // it as `&mut dyn Vector` and let the inner solver write to it.
        // sol_s lives in `sol.sol_s` (slack `s` is shared between
        // resto and orig — same dim m_ineq — so we route the inner
        // solver's sol_s straight into it). sol_c / sol_d need scratch
        // copies because we use them post-solve for the slack
        // back-substitution.
        let mut sol_y_c_dense = self.space_m_eq.as_ref().unwrap().make_new_dense();
        let mut sol_y_d_dense = self.space_m_ineq.as_ref().unwrap().make_new_dense();

        // Borrow `sol.sol_x` as compound, then split off comp(0) as
        // mutable for the inner solve, leaving comp(1..4) for the
        // back-substitution stage below.
        let sol_x_compound = sol
            .sol_x
            .as_any_mut()
            .downcast_mut::<CompoundVector>()
            .expect("AugRestoSystemSolver: sol_x must be a CompoundVector");
        debug_assert_eq!(sol_x_compound.n_comps(), 5);

        let status = {
            let sol_x_r = sol_x_compound.comp_mut(0);
            let inner_coeffs = AugSysCoeffs {
                w: Some(self.h_orig.as_ref().unwrap()),
                w_factor: coeffs.w_factor,
                d_x: Some(sigma_orig_dyn),
                delta_x: coeffs.delta_x,
                d_s: coeffs.d_s,
                delta_s: coeffs.delta_s,
                j_c: self.j_c_orig.as_ref().unwrap(),
                d_c: Some(&d_c_r_dense),
                delta_c: coeffs.delta_c,
                j_d: self.j_d_orig.as_ref().unwrap(),
                d_d: Some(&d_d_r_dense),
                delta_d: coeffs.delta_d,
            };
            let inner_rhs = AugSysRhs {
                rhs_x: rhs_x_r_dyn,
                rhs_s: rhs.rhs_s,
                rhs_c: &rhs_c_r_dense,
                rhs_d: &rhs_d_r_dense,
            };
            let mut inner_sol = AugSysSol {
                sol_x: sol_x_r,
                sol_s: sol.sol_s,
                sol_c: &mut sol_y_c_dense,
                sol_d: &mut sol_y_d_dense,
            };
            self.inner.solve(
                &inner_coeffs,
                &inner_rhs,
                &mut inner_sol,
                check_neg_evals,
                num_neg_evals,
            )
        };

        if status != ESymSolverStatus::Success {
            return status;
        }

        // ---- Write y_c / y_d into the caller-provided sol. ----
        let sol_y_c_vals = sol_y_c_dense.expanded_values();
        let sol_y_d_vals = sol_y_d_dense.expanded_values();

        if dbg {
            let sigma_orig_vals = dense_values(sigma_orig_dyn);
            let rhs_x_orig_vals = dense_values(rhs_x_r_dyn);
            let inf_norm = |v: &[f64]| v.iter().fold(0.0_f64, |a, &x| a.max(x.abs()));
            let sol_x_r = sol_x_compound.comp(0);
            let sol_x_orig_vals = dense_values(sol_x_r);
            tracing::debug!(target: "pounce::restoration",
                "[resto-aug]   ||sigma_orig||={:.3e} ||sigma_n_c||={:.3e} ||sigma_p_c||={:.3e} ||sigma_n_d||={:.3e} ||sigma_p_d||={:.3e}",
                inf_norm(&sigma_orig_vals),
                inf_norm(&sigma_n_c), inf_norm(&sigma_p_c), inf_norm(&sigma_n_d), inf_norm(&sigma_p_d),
            );
            tracing::debug!(target: "pounce::restoration",
                "[resto-aug]   ||rhs_x_orig||={:.3e} ||rhs_n_c||={:.3e} ||rhs_p_c||={:.3e} ||rhs_n_d||={:.3e} ||rhs_p_d||={:.3e} ||rhs_c||={:.3e} ||rhs_d||={:.3e}",
                inf_norm(&rhs_x_orig_vals), inf_norm(&rhs_n_c), inf_norm(&rhs_p_c),
                inf_norm(&rhs_n_d), inf_norm(&rhs_p_d), inf_norm(&rhs_c_vals), inf_norm(&rhs_d_vals),
            );
            tracing::debug!(target: "pounce::restoration",
                "[resto-aug]   ||rhs_cR||={:.3e} ||rhs_dR||={:.3e} ||D_cR||={:.3e} ||D_dR||={:.3e} ||sol_x_orig||={:.3e} ||sol_y_c||={:.3e} ||sol_y_d||={:.3e}",
                inf_norm(&rhs_c_r), inf_norm(&rhs_d_r),
                inf_norm(&d_c_r), inf_norm(&d_d_r),
                inf_norm(&sol_x_orig_vals),
                inf_norm(&sol_y_c_vals), inf_norm(&sol_y_d_vals),
            );
        }
        downcast_dense_mut(sol.sol_c).set_values(&sol_y_c_vals);
        downcast_dense_mut(sol.sol_d).set_values(&sol_y_d_vals);

        // ---- Back-substitute slack solutions. ----
        let mut sol_n_c_vals = vec![0.0; m_eq];
        let mut sol_p_c_vals = vec![0.0; m_eq];
        for i in 0..m_eq {
            sol_n_c_vals[i] =
                expand_sol_n_c_elem(rhs_n_c[i], sol_y_c_vals[i], sig_tilde_n_c_inv[i]);
            sol_p_c_vals[i] =
                expand_sol_p_c_elem(rhs_p_c[i], sol_y_c_vals[i], sig_tilde_p_c_inv[i]);
        }
        let mut sol_n_d_vals = vec![0.0; m_ineq];
        let mut sol_p_d_vals = vec![0.0; m_ineq];
        for i in 0..m_ineq {
            // Pd_L = I → sol_n_d = σ̃_{n_d}⁻¹ · (rhs_{n_d} − sol_{y_d})
            sol_n_d_vals[i] =
                expand_sol_n_c_elem(rhs_n_d[i], sol_y_d_vals[i], sig_tilde_n_d_inv[i]);
            // −Pd_U = −I → sol_p_d = σ̃_{p_d}⁻¹ · (rhs_{p_d} + sol_{y_d})
            sol_p_d_vals[i] =
                expand_sol_p_c_elem(rhs_p_d[i], sol_y_d_vals[i], sig_tilde_p_d_inv[i]);
        }
        downcast_dense_mut(sol_x_compound.comp_mut(1)).set_values(&sol_n_c_vals);
        downcast_dense_mut(sol_x_compound.comp_mut(2)).set_values(&sol_p_c_vals);
        downcast_dense_mut(sol_x_compound.comp_mut(3)).set_values(&sol_n_d_vals);
        downcast_dense_mut(sol_x_compound.comp_mut(4)).set_values(&sol_p_d_vals);

        ESymSolverStatus::Success
    }
}

// ---------- Helpers ----------

fn dense_values(v: &dyn Vector) -> Vec<Number> {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .expect("AugRestoSystemSolver: expected DenseVector argument")
        .expanded_values()
}

fn downcast_dense_mut(v: &mut dyn Vector) -> &mut DenseVector {
    v.as_any_mut()
        .downcast_mut::<DenseVector>()
        .expect("AugRestoSystemSolver: expected DenseVector argument")
}

// ---------- Scalar reduction kernels ----------

/// Elementwise `σ̃⁻¹ = 1 / (σ + Δ_x)` per `IpAugRestoSystemSolver.cpp:407-449`.
///
/// Mirrors the three branches in upstream:
/// * both `σ` and `Δ_x` present → `1 / (σ + Δ_x)`,
/// * only `σ` present (`Δ_x == 0`) → `1 / σ`,
/// * only `Δ_x` present (`σ` absent) → `1 / Δ_x`.
///
/// The "neither present" case is handled by the caller (returns `None`
/// so the entire block can be skipped, matching the cache short-circuit
/// at line 415).
pub fn sigma_tilde_inv_elem(sigma: Option<f64>, delta_x: f64) -> Option<f64> {
    match (sigma, delta_x) {
        (Some(s), 0.0) => Some(1.0 / s),
        (Some(s), d) => Some(1.0 / (s + d)),
        (None, 0.0) => None,
        (None, d) => Some(1.0 / d),
    }
}

/// Elementwise `−Ω_c + D_c` per `IpAugRestoSystemSolver.cpp:309-356`.
///
/// `Ω_c = σ̃⁻¹_{n_c} + σ̃⁻¹_{p_c}`; the result is `−Ω_c + D_c` if any
/// component is present, else `None`.
pub fn neg_omega_plus_d_elem(
    sigma_tilde_n_inv: Option<f64>,
    sigma_tilde_p_inv: Option<f64>,
    d_c: Option<f64>,
) -> Option<f64> {
    if sigma_tilde_n_inv.is_none() && sigma_tilde_p_inv.is_none() && d_c.is_none() {
        return None;
    }
    let n_term = sigma_tilde_n_inv.unwrap_or(0.0);
    let p_term = sigma_tilde_p_inv.unwrap_or(0.0);
    let d_term = d_c.unwrap_or(0.0);
    Some(-n_term - p_term + d_term)
}

/// Elementwise reduction of the equality-block RHS for the resto Schur
/// complement. Mirrors `IpAugRestoSystemSolver.cpp:633-672` (`Rhs_cR`):
/// ```text
///   rhs_cR = rhs_c − σ̃_{n_c}⁻¹ · rhs_{n_c} + σ̃_{p_c}⁻¹ · rhs_{p_c}
/// ```
/// Either `σ̃` may be `None`, in which case its term drops out.
pub fn rhs_cr_elem(
    rhs_c: f64,
    sigma_tilde_n_inv: Option<f64>,
    rhs_n_c: f64,
    sigma_tilde_p_inv: Option<f64>,
    rhs_p_c: f64,
) -> f64 {
    let n_contrib = sigma_tilde_n_inv.map(|s| s * rhs_n_c).unwrap_or(0.0);
    let p_contrib = sigma_tilde_p_inv.map(|s| s * rhs_p_c).unwrap_or(0.0);
    rhs_c - n_contrib + p_contrib
}

/// Post-solve expansion for the `n_c` block. Mirrors
/// `IpAugRestoSystemSolver.cpp:267-273`:
/// ```text
///   sol_{n_c} = σ̃_{n_c}⁻¹ · (rhs_{n_c} − sol_{y_c})
/// ```
/// Returns `0.0` when `σ̃_{n_c}⁻¹` is absent (block contributes
/// nothing — slack pair was inactive).
pub fn expand_sol_n_c_elem(rhs_n_c: f64, sol_y_c: f64, sigma_tilde_n_inv: Option<f64>) -> f64 {
    sigma_tilde_n_inv
        .map(|s| s * (rhs_n_c - sol_y_c))
        .unwrap_or(0.0)
}

/// Post-solve expansion for the `p_c` block. Mirrors
/// `IpAugRestoSystemSolver.cpp:275-284`:
/// ```text
///   sol_{p_c} = σ̃_{p_c}⁻¹ · (rhs_{p_c} + sol_{y_c})
/// ```
/// (sign on `sol_yc` flipped vs. the `n_c` case — slack-pair sign).
pub fn expand_sol_p_c_elem(rhs_p_c: f64, sol_y_c: f64, sigma_tilde_p_inv: Option<f64>) -> f64 {
    sigma_tilde_p_inv
        .map(|s| s * (rhs_p_c + sol_y_c))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_tilde_inv_combines_sigma_and_delta() {
        assert_eq!(sigma_tilde_inv_elem(Some(0.25), 0.75), Some(1.0));
    }

    #[test]
    fn sigma_tilde_inv_pure_sigma_path() {
        assert_eq!(sigma_tilde_inv_elem(Some(0.5), 0.0), Some(2.0));
    }

    #[test]
    fn sigma_tilde_inv_pure_delta_path() {
        assert_eq!(sigma_tilde_inv_elem(None, 0.5), Some(2.0));
    }

    #[test]
    fn sigma_tilde_inv_skips_when_both_absent() {
        assert_eq!(sigma_tilde_inv_elem(None, 0.0), None);
    }

    #[test]
    fn neg_omega_returns_none_when_all_absent() {
        assert_eq!(neg_omega_plus_d_elem(None, None, None), None);
    }

    #[test]
    fn neg_omega_sums_negated_inverses() {
        let r = neg_omega_plus_d_elem(Some(2.0), Some(3.0), Some(0.5));
        assert_eq!(r, Some(-2.0 - 3.0 + 0.5));
    }

    #[test]
    fn neg_omega_propagates_d_alone() {
        assert_eq!(neg_omega_plus_d_elem(None, None, Some(0.7)), Some(0.7));
    }

    #[test]
    fn rhs_cr_combines_three_terms() {
        let r = rhs_cr_elem(1.0, Some(0.5), 2.0, Some(0.25), 4.0);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn rhs_cr_drops_terms_when_sigma_absent() {
        let r = rhs_cr_elem(2.0, None, 3.0, Some(0.5), 6.0);
        assert_eq!(r, 2.0 + 0.5 * 6.0);
        let r = rhs_cr_elem(2.0, None, 3.0, None, 6.0);
        assert_eq!(r, 2.0);
    }

    #[test]
    fn expand_sol_n_c_zero_when_sigma_absent() {
        assert_eq!(expand_sol_n_c_elem(1.0, 2.0, None), 0.0);
    }

    #[test]
    fn expand_sol_n_c_signs() {
        assert_eq!(expand_sol_n_c_elem(5.0, 1.0, Some(0.5)), 2.0);
        assert_eq!(expand_sol_n_c_elem(1.0, 5.0, Some(0.5)), -2.0);
    }

    #[test]
    fn expand_sol_p_c_signs() {
        assert_eq!(expand_sol_p_c_elem(5.0, 1.0, Some(0.5)), 3.0);
        assert_eq!(expand_sol_p_c_elem(1.0, 5.0, Some(0.5)), 3.0);
    }

    #[test]
    fn expand_sol_p_c_zero_when_sigma_absent() {
        assert_eq!(expand_sol_p_c_elem(1.0, 2.0, None), 0.0);
    }
}
