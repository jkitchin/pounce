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
use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrixSpace;
use pounce_linalg::multi_vector_matrix::MultiVectorMatrixSpace;
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linalg::{LowRankUpdateSymMatrix, Matrix, MultiVectorMatrix, Vector};
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
    /// The orig block in factored form, when `W` arrived low-rank and
    /// the inner solver can apply it that way (#684). Mutually exclusive
    /// with `h_orig` being meaningful on that solve.
    w_lowrank_orig: Option<LowRankUpdateSymMatrix>,
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
            w_lowrank_orig: None,
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

    /// `w` is `None` when the orig block is being carried in factored
    /// form (#684) — there are no triplets to pin, and `h_orig` stays
    /// unset. Everything else about the structure is unchanged.
    fn build_structure(&mut self, w: Option<&SymTMatrix>, j_c: &GenTMatrix, j_d: &GenTMatrix) {
        let m_eq = j_c.n_rows();
        let m_ineq = j_d.n_rows();
        let n_total = j_c.n_cols();
        let n_orig = n_total - 2 * m_eq - 2 * m_ineq;

        // Orig Hessian: every triplet of W has row/col in 1..=n_orig
        // (eval_h emits the orig Hessian + diagonal proximity term;
        // slack rows/cols are zero), so we can reuse the same
        // (irows, jcols) at dim n_orig.
        self.h_orig = w.map(|w| {
            let h_space = SymTMatrixSpace::new(n_orig, w.irows().to_vec(), w.jcols().to_vec());
            SymTMatrix::new(h_space)
        });

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
        self.refill_jacobians(j_c, j_d);
    }

    /// The Jacobian half of [`Self::refill_values`], for the factored
    /// path where there is no `h_orig` to fill (#684).
    fn refill_jacobians(&mut self, j_c: &GenTMatrix, j_d: &GenTMatrix) {
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
        let w_dyn = coeffs
            .w
            .expect("AugRestoSystemSolver: W must be present (resto Hessian)");
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

        // The flat Schur reduction reads `W`'s triplets directly. The
        // exact-Hessian path publishes `W` as a [`SymTMatrix`] (orig
        // Hessian + proximity diagonal, all triplets in `1..=n_orig`),
        // which we use as-is. The limited-memory path publishes `W` as a
        // [`LowRankUpdateSymMatrix`] (`B0 + VVᵀ − UUᵀ`, no triplets), so
        // we materialize its `n_orig` orig block into a dense-lower-
        // triangle `SymTMatrix` via matrix–vector products. Restoration
        // is a heavyweight fallback and `n_orig` is the orig variable
        // count, so the `O(n_orig²)` densification is negligible. Without
        // this, constrained limited-memory solves (every constrained
        // solve through the Python `Problem` API) panic the moment
        // restoration triggers (pounce#102).
        let m_eq = j_c.n_rows();
        let m_ineq = j_d.n_rows();
        let n_orig = j_c.n_cols() - 2 * m_eq - 2 * m_ineq;

        // ---- D_x compound (also the matvec-probe template for a low-rank W). ----
        let dx_compound = coeffs
            .d_x
            .expect("AugRestoSystemSolver: D_x must be present (5-block compound)")
            .as_any()
            .downcast_ref::<CompoundVector>()
            .expect("AugRestoSystemSolver: D_x must be a CompoundVector");
        debug_assert_eq!(dx_compound.n_comps(), 5);

        // Three ways to get the orig block to the inner solver, in
        // order of preference.
        //
        // 1. `W` is already a `SymTMatrix` (the exact-Hessian path) —
        //    use its triplets as they stand.
        // 2. `W` is low-rank and the inner solver applies low-rank `W`
        //    by Sherman-Morrison-Woodbury — hand it over factored. This
        //    is `O(n·rank)`.
        // 3. `W` is low-rank and the inner solver cannot — densify, as
        //    restoration always did. `O(n²)`, and the reason #684
        //    aborted a 60k-variable solve, so it is now the last resort
        //    rather than the only option.
        self.w_lowrank_orig = None;
        let w_owned;
        let w: Option<&SymTMatrix> = match w_dyn.as_any().downcast_ref::<SymTMatrix>() {
            Some(w) => Some(w),
            None => {
                if self.inner.handles_low_rank_w() {
                    self.w_lowrank_orig = low_rank_orig_block(w_dyn, n_orig);
                }
                if self.w_lowrank_orig.is_some() {
                    tracing::debug!(target: "pounce::restoration",
                        "[resto-aug] orig block carried in factored low-rank form (n_orig={})",
                        n_orig,
                    );
                    None
                } else {
                    tracing::debug!(target: "pounce::restoration",
                        "[resto-aug] densifying orig block, {} entries (n_orig={})",
                        (n_orig as u128) * (n_orig as u128 + 1) / 2, n_orig,
                    );
                    w_owned = materialize_orig_block(w_dyn, n_orig);
                    Some(&w_owned)
                }
            }
        };

        if !self.initialized {
            self.build_structure(w, j_c, j_d);
        }
        if let Some(w) = w {
            self.refill_values(w, j_c, j_d);
        } else {
            self.refill_jacobians(j_c, j_d);
        }

        let m_eq = self.m_eq as usize;
        let m_ineq = self.m_ineq as usize;

        // Restoration debug gate. The canonical spelling is
        // `POUNCE_DBG_RESTO` — the `POUNCE_DBG_*` convention shared by
        // every other debug gate, and already used by the restoration
        // entry trace in `pounce-algorithm/src/ipopt_alg.rs`. The
        // historical `POUNCE_RESTO_DBG` (this crate only) is retained as
        // a deprecated alias so existing invocations keep working; either
        // spelling enables this output. Reconciles pounce#235's two-
        // spellings trap onto one guessable name.
        let dbg =
            std::env::var("POUNCE_DBG_RESTO").is_ok() || std::env::var("POUNCE_RESTO_DBG").is_ok();
        if dbg {
            tracing::debug!(target: "pounce::restoration",
                "[resto-aug] n_orig={} m_eq={} m_ineq={} W={} J_c.nz={} J_d.nz={} delta_x={:.3e} delta_c={:.3e} delta_d={:.3e}",
                self.n_orig, self.m_eq, self.m_ineq,
                match w {
                    Some(w) => format!("nz={}", w.nonzeros()),
                    None => "low-rank (factored)".to_string(),
                },
                j_c.nonzeros(), j_d.nonzeros(),
                coeffs.delta_x, coeffs.delta_c, coeffs.delta_d,
            );
        }

        // ---- σ vectors from D_x compound. ----
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
                w: match self.w_lowrank_orig.as_ref() {
                    Some(lr) => Some(lr as &dyn pounce_linalg::SymMatrix),
                    None => Some(self.h_orig.as_ref().unwrap()),
                },
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
                s_blocks: None,
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

/// Densify the `n_orig` orig (top-left) block of a flat resto Hessian
/// `W` into a dense lower-triangle [`SymTMatrix`].
///
/// The exact-Hessian path publishes `W` as a [`SymTMatrix`] with triplet
/// storage; the limited-memory path publishes it as a
/// [`LowRankUpdateSymMatrix`] (`D + V Vᵀ − U Uᵀ`) with no triplets, so the
/// Schur reduction — which reads `W`'s entries directly — needs an
/// explicit form. Restoration only ever wraps the *plain* limited-memory
/// Hessian (full-space diagonal, identity `P`, no `reduced_diag`), so the
/// `(i, j)` entry of the orig block is, in closed form,
/// `Wᵢⱼ = D[i]·δᵢⱼ + Σ_k V[i,k]·V[j,k] − Σ_k U[i,k]·U[j,k]` for
/// `i, j < n_orig`. We read `D`, `V`, `U` directly through the matrix's
/// accessors rather than probing with `W·eⱼ`: the resto low-rank `W`
/// stores `D` as a flat [`DenseVector`] but its `V`/`U` columns as 5-block
/// resto [`CompoundVector`]s, so no single probe-vector type threads
/// through `mult_vector`. Cost is `O(rank·n_orig²)`; restoration is a rare
/// heavyweight fallback and `n_orig` is small, so this is negligible.
///
/// Panics with a clear message if `W` is neither a `SymTMatrix` (handled
/// by the caller) nor a plain-configuration `LowRankUpdateSymMatrix` —
/// i.e. if a future code path hands restoration a low-rank `W` with a
/// `p_lowrank` expansion or `reduced_diag`, which this closed form does
/// not cover (pounce#102).
/// The orig block of a low-rank resto `W`, kept in **factored** form.
///
/// `materialize_orig_block` below builds the same operator as a dense
/// lower triangle, which is `O(n²)` and was the only thing restoration
/// could do with a limited-memory `W`. On a 59,939-variable collocation
/// model that reserved 14 GB in a single allocation and aborted the
/// process the moment restoration was entered (#684). The dense form is
/// also wasteful on its own terms: `B = σI + VVᵀ − UUᵀ` carries
/// `O(n·rank)` of information with `rank ≤ 2·limited_memory_max_history`
/// (12 by default), and squaring it up throws none of that away — it
/// just spends `n²` to store it.
///
/// So when the inner solver can consume a low-rank `W` directly — the
/// Sherman-Morrison-Woodbury path that the main iteration has used all
/// along — hand it one. Reuses the same `orig_rows` /
/// `multi_vector_orig_cols` extraction the dense path already performed,
/// so the cost is the extraction alone.
///
/// Returns `None` when `W` is not a plain-configuration
/// `LowRankUpdateSymMatrix`, leaving the caller on the dense path rather
/// than guessing at a form this does not cover (pounce#102).
fn low_rank_orig_block(w: &dyn Matrix, n_orig: Index) -> Option<LowRankUpdateSymMatrix> {
    let n = n_orig as usize;
    let lr = w.as_any().downcast_ref::<LowRankUpdateSymMatrix>()?;
    if lr.p_lowrank().is_some() || lr.reduced_diag() {
        return None;
    }

    let space = LowRankUpdateSymMatrixSpace::new(n_orig, None, false);
    let mut out = LowRankUpdateSymMatrix::new(space);

    let diag = lr
        .get_diag()
        .map(|d| orig_rows(d.as_ref(), n))
        .unwrap_or_else(|| vec![0.0; n]);
    let dspace = DenseVectorSpace::new(n_orig);
    let mut dvec = dspace.make_new_dense();
    dvec.set_values(&diag);
    out.set_diag(Rc::new(dvec) as Rc<dyn Vector>);

    // `V` and `U` are restricted to their orig rows and re-hung in a
    // flat dense space: the resto `W` stores them as 5-block resto
    // `CompoundVector`s, which the inner solver's SMW path cannot
    // multiply against an orig-sized iterate.
    let pack = |cols: Vec<Vec<Number>>| -> Option<Rc<MultiVectorMatrix>> {
        if cols.is_empty() {
            return None;
        }
        let mv_space = MultiVectorMatrixSpace::new(cols.len() as Index, dspace.clone());
        let mut mv = MultiVectorMatrix::new(mv_space);
        for (k, c) in cols.iter().enumerate() {
            let mut col = dspace.make_new_dense();
            col.set_values(c);
            mv.set_vector(k as Index, Rc::new(col) as Rc<dyn Vector>);
        }
        Some(Rc::new(mv))
    };
    if let Some(v) = pack(multi_vector_orig_cols(lr.get_v(), n)) {
        out.set_v(v);
    }
    if let Some(u) = pack(multi_vector_orig_cols(lr.get_u(), n)) {
        out.set_u(u);
    }
    Some(out)
}

fn materialize_orig_block(w: &dyn Matrix, n_orig: Index) -> SymTMatrix {
    let n = n_orig as usize;
    let lr = w
        .as_any()
        .downcast_ref::<LowRankUpdateSymMatrix>()
        .expect("AugRestoSystemSolver: resto W must be a SymTMatrix or LowRankUpdateSymMatrix");
    assert!(
        lr.p_lowrank().is_none() && !lr.reduced_diag(),
        "AugRestoSystemSolver: resto W has a p_lowrank/reduced_diag low-rank form \
         that the orig-block densification does not cover (pounce#102)"
    );

    // Dense lower-triangle sparsity (1-based, row-major).
    //
    // `n(n+1)/2` entries in each of three arrays. At n = 59,956 that is
    // 1.8e9 entries and a 14 GB `vals` allocation, which aborts the
    // process rather than failing the solve (#684). The caller prefers
    // the factored path and only lands here when the inner solver cannot
    // take one, so refuse loudly instead of asking the allocator for
    // something no machine will give.
    //
    // The cap is on the entry count, not on bytes: 2^31 entries already
    // exceeds `Index`'s range for the triplet indices, so beyond it the
    // dense form is not merely large but unrepresentable.
    let tri = (n as u128) * (n as u128 + 1) / 2;
    assert!(
        tri < i32::MAX as u128,
        "AugRestoSystemSolver: restoration cannot densify a limited-memory Hessian \
         at n_orig={n} — the dense lower triangle needs {tri} entries, past what the \
         triplet index type can address. This solve needs an inner solver that \
         applies a low-rank W directly (see #684); the low-rank path is taken \
         automatically when one is installed."
    );
    let tri = tri as usize;
    let mut irows = Vec::with_capacity(tri);
    let mut jcols = Vec::with_capacity(tri);
    for i in 1..=n_orig {
        for j in 1..=i {
            irows.push(i);
            jcols.push(j);
        }
    }
    let space = SymTMatrixSpace::new(n_orig, irows, jcols);
    let mut sym = SymTMatrix::new(space);

    // Pull the orig (first `n_orig`) rows of D and of every V/U column.
    let diag = lr
        .get_diag()
        .map(|d| orig_rows(d.as_ref(), n))
        .unwrap_or_else(|| vec![0.0; n]);
    let v_cols = multi_vector_orig_cols(lr.get_v(), n);
    let u_cols = multi_vector_orig_cols(lr.get_u(), n);

    let vals = sym.values_mut();
    for ii in 0..n {
        for jj in 0..=ii {
            // Row-major triplet index of the 1-based entry (ii+1, jj+1).
            let idx = (ii + 1) * ii / 2 + jj;
            let mut acc = if ii == jj { diag[ii] } else { 0.0 };
            for col in &v_cols {
                acc += col[ii] * col[jj];
            }
            for col in &u_cols {
                acc -= col[ii] * col[jj];
            }
            vals[idx] = acc;
        }
    }
    sym
}

/// The first `n` rows of a resto primal vector, which is either a flat
/// [`DenseVector`] or a 5-block resto [`CompoundVector`] whose component 0
/// (the `n_orig` orig block) is a `DenseVector`.
fn orig_rows(v: &dyn Vector, n: usize) -> Vec<Number> {
    // `expanded_values` (not `values`) so a homogeneously-stored block —
    // e.g. the σ·I diagonal published by the limited-memory updater — is
    // materialized rather than tripping the dense-vector value assert.
    if let Some(c) = v.as_any().downcast_ref::<CompoundVector>() {
        let orig = c
            .comp(0)
            .as_any()
            .downcast_ref::<DenseVector>()
            .expect("AugRestoSystemSolver: resto W orig block must be a DenseVector");
        orig.expanded_values()[..n].to_vec()
    } else if let Some(d) = v.as_any().downcast_ref::<DenseVector>() {
        d.expanded_values()[..n].to_vec()
    } else {
        panic!("AugRestoSystemSolver: resto W component must be Dense or Compound");
    }
}

/// The orig (first `n`) rows of every column of an optional curvature
/// [`MultiVectorMatrix`] (`V` or `U`); empty when the factor is absent.
fn multi_vector_orig_cols(m: Option<&Rc<MultiVectorMatrix>>, n: usize) -> Vec<Vec<Number>> {
    match m {
        None => Vec::new(),
        Some(mv) => (0..mv.space().n_cols())
            .map(|k| orig_rows(mv.get_vector(k).as_ref(), n))
            .collect(),
    }
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

    /// The factored and densified orig blocks must be the *same
    /// operator* (#684).
    ///
    /// This is the correctness claim the fix rests on: restoration now
    /// hands the inner solver `B = D + VVᵀ − UUᵀ` in factored form
    /// instead of a dense lower triangle, and the two must act
    /// identically on every vector. Comparing solver outcomes on
    /// fixtures cannot establish that — a fixture that fails either way
    /// tells you nothing, and one that solves both ways would still
    /// agree if the factored form were subtly wrong in a direction the
    /// step never explored. So compare the operators directly.
    #[test]
    fn factored_and_densified_orig_blocks_are_the_same_operator() {
        use pounce_linalg::SymMatrix;

        let n: Index = 7;
        let nu = n as usize;
        let space = LowRankUpdateSymMatrixSpace::new(n, None, false);
        let mut w = LowRankUpdateSymMatrix::new(space);

        let dspace = DenseVectorSpace::new(n);
        let mut d = dspace.make_new_dense();
        d.set_values(&[2.0, 3.0, 0.5, 1.25, 4.0, 0.75, 1.5]);
        w.set_diag(Rc::new(d) as Rc<dyn Vector>);

        // Two positive columns and one negative, so both the V and the U
        // branch are exercised and the result is genuinely indefinite —
        // a test with V only would pass on a sign error in U.
        let mk = |cols: &[&[Number]]| -> Rc<MultiVectorMatrix> {
            let mv_space = MultiVectorMatrixSpace::new(cols.len() as Index, dspace.clone());
            let mut mv = MultiVectorMatrix::new(mv_space);
            for (k, c) in cols.iter().enumerate() {
                let mut col = dspace.make_new_dense();
                col.set_values(c);
                mv.set_vector(k as Index, Rc::new(col) as Rc<dyn Vector>);
            }
            Rc::new(mv)
        };
        w.set_v(mk(&[
            &[1.0, -0.5, 0.25, 0.0, 2.0, -1.0, 0.5],
            &[0.3, 0.7, -1.2, 0.9, 0.1, 0.4, -0.6],
        ]));
        w.set_u(mk(&[&[-0.8, 0.2, 0.6, -0.4, 1.1, 0.05, 0.9]]));

        let dense = materialize_orig_block(&w, n);
        let factored = low_rank_orig_block(&w, n).expect("plain low-rank W must convert");

        // Probe every basis direction: agreement on all of them is
        // agreement on the whole operator, since both are linear.
        for j in 0..nu {
            let mut x = dspace.make_new_dense();
            let mut xv = vec![0.0; nu];
            xv[j] = 1.0;
            x.set_values(&xv);

            let mut y_dense = dspace.make_new_dense();
            let mut y_fact = dspace.make_new_dense();
            dense.mult_vector(1.0, &x, 0.0, &mut y_dense);
            factored.mult_vector(1.0, &x, 0.0, &mut y_fact);

            let a = y_dense.expanded_values();
            let b = y_fact.expanded_values();
            for i in 0..nu {
                assert!(
                    (a[i] - b[i]).abs() < 1e-12,
                    "column {j}, row {i}: dense {} vs factored {}",
                    a[i],
                    b[i],
                );
            }
        }
    }

    /// A `W` the closed form does not cover must decline, not guess.
    /// The dense path has the same restriction and asserts on it; the
    /// factored path returns `None` so the caller falls back rather than
    /// silently producing a different operator.
    #[test]
    fn factored_orig_block_declines_a_shape_it_does_not_cover() {
        let n: Index = 3;
        let space = LowRankUpdateSymMatrixSpace::new(n, None, true);
        let w = LowRankUpdateSymMatrix::new(space);
        assert!(
            low_rank_orig_block(&w, n).is_none(),
            "reduced_diag form must decline",
        );
    }

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
