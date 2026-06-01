//! Low-rank augmented system solver — port of
//! `Algorithm/IpLowRankAugSystemSolver.{hpp,cpp}`.
//!
//! Wraps another [`AugSystemSolver`] and exploits a [`LowRankUpdateSymMatrix`]
//! Hessian via the Sherman-Morrison-Woodbury identity. The wrapped
//! solver factorizes the diagonal part `B0`; this solver applies the
//! rank-`(nV + nU)` correction using cached
//! `Vtilde1 = K⁻¹ V` and `Utilde2 = K⁻¹ U − Vtilde1·(J1^{-T}J1^{-1}·Vtilde1ᵀU)`
//! plus their dense Cholesky factors `J1 = chol(I + Vtilde1ᵀ V)` and
//! `J2 = chol(I − Utilde2ᵀ U)`.
//!
//! The augmented-system solution comes from upstream's recipe
//! (`IpLowRankAugSystemSolver.cpp:179-228`):
//!
//! 1. inner solver factors `K` (the aug system with `Wdiag` in place
//!    of `W`) and back-substitutes for `csol_diag = K⁻¹ rhs`.
//! 2. If `Utilde2_` is set, apply  `csol += Utilde2 · J2⁻¹ J2⁻ᵀ · Utilde2ᵀ rhs`.
//! 3. If `Vtilde1_` is set, apply  `csol −= Vtilde1 · J1⁻¹ J1⁻ᵀ · Vtilde1ᵀ rhs`.
//!
//! `Vtilde1` and `Utilde2` are stored as four separate per-block
//! [`MultiVectorMatrix`]es (x, s, c, d) — the same data that upstream
//! packs into a 4-component `CompoundVector` of dense columns. This
//! keeps the SMW arithmetic in dense linalg without needing a
//! compound-vector storage class.

use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use pounce_common::tagged::Tag;
use pounce_common::timing::TimingStatistics;
use pounce_common::types::{Index, Number};
use pounce_linalg::dense_gen_matrix::{DenseGenMatrix, DenseGenMatrixSpace};
use pounce_linalg::dense_sym_matrix::DenseSymMatrixSpace;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::diag_matrix::DiagMatrix;
use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrix;
use pounce_linalg::multi_vector_matrix::{MultiVectorMatrix, MultiVectorMatrixSpace};
use pounce_linalg::{Matrix, SymMatrix, Vector};
use pounce_linsol::ESymSolverStatus;
use std::rc::Rc;


pub struct LowRankAugSystemSolver {
    /// Inner solver that owns the diagonal factorization.
    inner: Box<dyn AugSystemSolver>,
    /// Whether `solve` has been called yet.
    first_call: bool,
    /// Cached negative-eigenvalue count.
    num_neg_evals: Index,
    /// Tag/scalar cache mirroring upstream's per-coefficient state.
    cache: AugSysCache,
    /// SMW factorization state (cleared on each rebuild).
    factor: Factorization,
}

#[derive(Debug, Clone)]
pub struct AugSysCache {
    pub w_tag: Tag,
    pub w_factor: Number,
    pub d_x_tag: Tag,
    pub delta_x: Number,
    pub d_s_tag: Tag,
    pub delta_s: Number,
    pub j_c_tag: Tag,
    pub d_c_tag: Tag,
    pub delta_c: Number,
    pub j_d_tag: Tag,
    pub d_d_tag: Tag,
    pub delta_d: Number,
}

impl Default for AugSysCache {
    fn default() -> Self {
        Self {
            w_tag: Tag::NONE,
            w_factor: 0.0,
            d_x_tag: Tag::NONE,
            delta_x: 0.0,
            d_s_tag: Tag::NONE,
            delta_s: 0.0,
            j_c_tag: Tag::NONE,
            d_c_tag: Tag::NONE,
            delta_c: 0.0,
            j_d_tag: Tag::NONE,
            d_d_tag: Tag::NONE,
            delta_d: 0.0,
        }
    }
}

#[derive(Default)]
struct Factorization {
    /// `Wdiag` substituted for `W` in every inner-solver call. Mirrors
    /// upstream `Wdiag_`. Held mutably so we can call
    /// [`DiagMatrix::set_diag`] on rebuild.
    wdiag: Option<Box<DiagMatrix>>,
    /// Dense Cholesky `J1 = chol(I + Vtilde1ᵀ · V)`. None when V is empty.
    j1: Option<DenseGenMatrix>,
    /// Dense Cholesky `J2 = chol(I − Utilde2ᵀ · U)`. None when U is empty.
    j2: Option<DenseGenMatrix>,
    /// Per-block `Vtilde1` storage (rank `nV`).
    vtilde1_x: Option<MultiVectorMatrix>,
    vtilde1_s: Option<MultiVectorMatrix>,
    vtilde1_c: Option<MultiVectorMatrix>,
    vtilde1_d: Option<MultiVectorMatrix>,
    /// Per-block `Utilde2` storage (rank `nU`).
    utilde2_x: Option<MultiVectorMatrix>,
    utilde2_s: Option<MultiVectorMatrix>,
    utilde2_c: Option<MultiVectorMatrix>,
    utilde2_d: Option<MultiVectorMatrix>,
}

impl LowRankAugSystemSolver {
    pub fn new(inner: Box<dyn AugSystemSolver>) -> Self {
        Self {
            inner,
            first_call: true,
            num_neg_evals: 0,
            cache: AugSysCache::default(),
            factor: Factorization::default(),
        }
    }

    /// Pure tag/scalar comparison — port of upstream
    /// `AugmentedSystemRequiresChange` (`IpLowRankAugSystemSolver.cpp:531-599`).
    pub fn augmented_system_requires_change(&self, coeffs: &AugSysCoeffs<'_>) -> bool {
        let cache = &self.cache;
        let zero_tag: Tag = Tag::NONE;

        let w_changed = match coeffs.w {
            Some(w) => w.as_tagged().get_tag() != cache.w_tag,
            None => cache.w_tag != zero_tag,
        };
        if w_changed || coeffs.w_factor != cache.w_factor {
            return true;
        }
        let dx_changed = match coeffs.d_x {
            Some(d) => d.as_tagged().get_tag() != cache.d_x_tag,
            None => cache.d_x_tag != zero_tag,
        };
        if dx_changed || coeffs.delta_x != cache.delta_x {
            return true;
        }
        let ds_changed = match coeffs.d_s {
            Some(d) => d.as_tagged().get_tag() != cache.d_s_tag,
            None => cache.d_s_tag != zero_tag,
        };
        if ds_changed || coeffs.delta_s != cache.delta_s {
            return true;
        }
        if coeffs.j_c.as_tagged().get_tag() != cache.j_c_tag {
            return true;
        }
        let dc_changed = match coeffs.d_c {
            Some(d) => d.as_tagged().get_tag() != cache.d_c_tag,
            None => cache.d_c_tag != zero_tag,
        };
        if dc_changed || coeffs.delta_c != cache.delta_c {
            return true;
        }
        if coeffs.j_d.as_tagged().get_tag() != cache.j_d_tag {
            return true;
        }
        let dd_changed = match coeffs.d_d {
            Some(d) => d.as_tagged().get_tag() != cache.d_d_tag,
            None => cache.d_d_tag != zero_tag,
        };
        if dd_changed || coeffs.delta_d != cache.delta_d {
            return true;
        }
        false
    }

    fn store_cache(&mut self, coeffs: &AugSysCoeffs<'_>) {
        let zero_tag = Tag::NONE;
        self.cache.w_tag = coeffs
            .w
            .map(|w| w.as_tagged().get_tag())
            .unwrap_or(zero_tag);
        self.cache.w_factor = coeffs.w_factor;
        self.cache.d_x_tag = coeffs
            .d_x
            .map(|d| d.as_tagged().get_tag())
            .unwrap_or(zero_tag);
        self.cache.delta_x = coeffs.delta_x;
        self.cache.d_s_tag = coeffs
            .d_s
            .map(|d| d.as_tagged().get_tag())
            .unwrap_or(zero_tag);
        self.cache.delta_s = coeffs.delta_s;
        self.cache.j_c_tag = coeffs.j_c.as_tagged().get_tag();
        self.cache.d_c_tag = coeffs
            .d_c
            .map(|d| d.as_tagged().get_tag())
            .unwrap_or(zero_tag);
        self.cache.delta_c = coeffs.delta_c;
        self.cache.j_d_tag = coeffs.j_d.as_tagged().get_tag();
        self.cache.d_d_tag = coeffs
            .d_d
            .map(|d| d.as_tagged().get_tag())
            .unwrap_or(zero_tag);
        self.cache.delta_d = coeffs.delta_d;
    }

    pub fn first_call(&self) -> bool {
        self.first_call
    }

    pub fn cache(&self) -> &AugSysCache {
        &self.cache
    }

    /// Rebuild `Wdiag`, `Vtilde1`, `Utilde2`, `J1`, `J2` from a fresh
    /// LR Hessian. Matches `IpLowRankAugSystemSolver.cpp::UpdateFactorization`
    /// (lines 233-404). Returns the inner-solver's status — on
    /// `WrongInertia` from a Cholesky failure, increments
    /// `num_neg_evals` so the upper layer (PerturbationHandler) sees a
    /// distinct retry target.
    fn update_factorization(
        &mut self,
        lr_w: &LowRankUpdateSymMatrix,
        coeffs: &AugSysCoeffs<'_>,
        proto: &AugSysRhs<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        let proto_x = downcast_dense(proto.rhs_x);
        let proto_s = downcast_dense(proto.rhs_s);
        let proto_c = downcast_dense(proto.rhs_c);
        let proto_d = downcast_dense(proto.rhs_d);
        let space_x = Rc::clone(proto_x.space());
        let space_s = Rc::clone(proto_s.space());
        let space_c = Rc::clone(proto_c.space());
        let space_d = Rc::clone(proto_d.space());

        // 1. Build Wdiag from B0 (with optional P_LM expansion when
        //    `reduced_diag` is set). When w_factor != 1.0, B0 is treated
        //    as zero per upstream `IpLowRankAugSystemSolver.cpp:268-272`.
        let b0_dense: DenseVector = if coeffs.w_factor == 1.0 {
            match lr_w.get_diag() {
                Some(d) => clone_dense(downcast_dense(d.as_ref())),
                None => zero_x_for(&space_x, lr_w),
            }
        } else {
            zero_x_for(&space_x, lr_w)
        };

        let wdiag_diag: Rc<dyn Vector> = match (lr_w.p_lowrank(), lr_w.reduced_diag()) {
            (Some(p_lm), true) => {
                // fullx = P_LM · B0
                let mut fullx = space_x.make_new_dense();
                p_lm.mult_vector(1.0, &b0_dense, 0.0, &mut fullx);
                Rc::new(fullx) as Rc<dyn Vector>
            }
            _ => Rc::new(clone_dense(&b0_dense)) as Rc<dyn Vector>,
        };
        let mut wdiag = Box::new(DiagMatrix::new(space_x.dim()));
        wdiag.set_diag(wdiag_diag);
        self.factor.wdiag = Some(wdiag);

        // 2. SolveMultiVector for V → Vtilde1 = K⁻¹ V (per-block).
        if coeffs.w_factor == 1.0 && lr_w.get_v().is_some() {
            let v = Rc::clone(lr_w.get_v().unwrap());
            let n_v = v.n_cols();

            // Build V_x: each column is either V[:,k] directly (no P_LM)
            // or P_LM · V[:,k]. We need V_x for the M1 update; we keep
            // it on the stack here.
            let v_x_space = MultiVectorMatrixSpace::new(n_v, Rc::clone(&space_x));
            let mut v_x = v_x_space.make_new_multi_vector();
            for k in 0..n_v {
                let vk = Rc::clone(v.get_vector(k));
                let rhs_x_k: Rc<dyn Vector> = match lr_w.p_lowrank() {
                    Some(p_lm) => {
                        let mut fullx = space_x.make_new_dense();
                        p_lm.mult_vector(1.0, vk.as_ref(), 0.0, &mut fullx);
                        Rc::new(fullx) as Rc<dyn Vector>
                    }
                    None => vk,
                };
                v_x.set_vector(k, rhs_x_k);
            }

            let (vt_x, vt_s, vt_c, vt_d) = self.multi_solve_block(
                &v_x,
                coeffs,
                &space_x,
                &space_s,
                &space_c,
                &space_d,
                check_neg_evals,
                num_neg_evals,
            );

            let vt_x = match vt_x {
                Ok(x) => x,
                Err(status) => return status,
            };

            // 3. M1 = I + Vtilde1_x^T · V_x; J1 = chol(M1).
            let m1_space = DenseSymMatrixSpace::new(n_v);
            let mut m1 = m1_space.make_new_dense_sym();
            m1.fill_identity(1.0);
            m1.high_rank_update_transpose(1.0, &vt_x, &v_x, 1.0);
            let j1_space = DenseGenMatrixSpace::new(n_v, n_v);
            let mut j1 = j1_space.make_new_dense_gen();
            if !j1.compute_cholesky_factor(&m1) {
                self.num_neg_evals += 1;
                return ESymSolverStatus::WrongInertia;
            }
            self.factor.vtilde1_x = Some(vt_x);
            self.factor.vtilde1_s = Some(vt_s);
            self.factor.vtilde1_c = Some(vt_c);
            self.factor.vtilde1_d = Some(vt_d);
            self.factor.j1 = Some(j1);
        } else {
            self.factor.vtilde1_x = None;
            self.factor.vtilde1_s = None;
            self.factor.vtilde1_c = None;
            self.factor.vtilde1_d = None;
            self.factor.j1 = None;
        }

        // 4. SolveMultiVector for U → Utilde1 = K⁻¹ U; orthogonalize
        //    against Vtilde1 (if present) to get Utilde2.
        if coeffs.w_factor == 1.0 && lr_w.get_u().is_some() {
            let u = Rc::clone(lr_w.get_u().unwrap());
            let n_u = u.n_cols();

            let u_x_space = MultiVectorMatrixSpace::new(n_u, Rc::clone(&space_x));
            let mut u_x = u_x_space.make_new_multi_vector();
            for k in 0..n_u {
                let uk = Rc::clone(u.get_vector(k));
                let rhs_x_k: Rc<dyn Vector> = match lr_w.p_lowrank() {
                    Some(p_lm) => {
                        let mut fullx = space_x.make_new_dense();
                        p_lm.mult_vector(1.0, uk.as_ref(), 0.0, &mut fullx);
                        Rc::new(fullx) as Rc<dyn Vector>
                    }
                    None => uk,
                };
                u_x.set_vector(k, rhs_x_k);
            }

            let (mut ut_x, mut ut_s, mut ut_c, mut ut_d) = match self.multi_solve_block(
                &u_x,
                coeffs,
                &space_x,
                &space_s,
                &space_c,
                &space_d,
                check_neg_evals,
                num_neg_evals,
            ) {
                (Ok(x), s, c, d) => (x, s, c, d),
                (Err(status), _, _, _) => return status,
            };

            // 5. If Vtilde1 is present: Utilde2 = Utilde1 − Vtilde1 · (J1⁻¹J1⁻ᵀ · Vtilde1ᵀU).
            if self.factor.vtilde1_x.is_some() {
                let vt1_x = self.factor.vtilde1_x.as_ref().unwrap();
                let vt1_s = self.factor.vtilde1_s.as_ref().unwrap();
                let vt1_c = self.factor.vtilde1_c.as_ref().unwrap();
                let vt1_d = self.factor.vtilde1_d.as_ref().unwrap();
                let n_v = vt1_x.n_cols();
                // C = Vtilde1_x^T · U_x  (n_v × n_u; HighRankUpdateTranspose's
                // generic-matrix variant — we synthesize via column dot products
                // since DenseGenMatrix doesn't expose a high_rank_update_transpose).
                let c_space = DenseGenMatrixSpace::new(n_v, n_u);
                let mut c_mat = c_space.make_new_dense_gen();
                {
                    let cv = c_mat.values_mut();
                    for j in 0..n_u as usize {
                        let uj = u_x.get_vector(j as Index).as_ref();
                        for i in 0..n_v as usize {
                            let vi = vt1_x.get_vector(i as Index).as_ref();
                            cv[i + j * n_v as usize] = vi.dot(uj);
                        }
                    }
                }
                self.factor
                    .j1
                    .as_ref()
                    .unwrap()
                    .cholesky_solve_matrix(&mut c_mat);
                ut_x.add_right_mult_matrix(-1.0, vt1_x, &c_mat, 1.0);
                ut_s.add_right_mult_matrix(-1.0, vt1_s, &c_mat, 1.0);
                ut_c.add_right_mult_matrix(-1.0, vt1_c, &c_mat, 1.0);
                ut_d.add_right_mult_matrix(-1.0, vt1_d, &c_mat, 1.0);
            }

            // 6. M2 = I − Utilde2_x^T · U_x; J2 = chol(M2). A non-positive
            //    pivot means the `−UUᵀ` correction drove the reduced
            //    Hessian indefinite: a genuine wrong-inertia signal that
            //    the perturbation handler should act on.
            let m2_space = DenseSymMatrixSpace::new(n_u);
            let mut m2 = m2_space.make_new_dense_sym();
            m2.fill_identity(1.0);
            m2.high_rank_update_transpose(-1.0, &ut_x, &u_x, 1.0);
            let j2_space = DenseGenMatrixSpace::new(n_u, n_u);
            let mut j2 = j2_space.make_new_dense_gen();
            if !j2.compute_cholesky_factor(&m2) {
                self.num_neg_evals += 1;
                return ESymSolverStatus::WrongInertia;
            }
            self.factor.utilde2_x = Some(ut_x);
            self.factor.utilde2_s = Some(ut_s);
            self.factor.utilde2_c = Some(ut_c);
            self.factor.utilde2_d = Some(ut_d);
            self.factor.j2 = Some(j2);
        } else {
            self.factor.utilde2_x = None;
            self.factor.utilde2_s = None;
            self.factor.utilde2_c = None;
            self.factor.utilde2_d = None;
            self.factor.j2 = None;
        }

        ESymSolverStatus::Success
    }

    /// Solve `K · Vtilde = [V_x; 0; 0; 0]` for one block of right-hand
    /// sides packed in `v_x` (dense column-by-column). Returns the four
    /// per-block columns of `Vtilde`. Mirrors the inner loop of
    /// upstream `SolveMultiVector` (`IpLowRankAugSystemSolver.cpp:406-528`).
    #[allow(clippy::too_many_arguments)]
    fn multi_solve_block(
        &mut self,
        v_x: &MultiVectorMatrix,
        coeffs: &AugSysCoeffs<'_>,
        space_x: &Rc<DenseVectorSpace>,
        space_s: &Rc<DenseVectorSpace>,
        space_c: &Rc<DenseVectorSpace>,
        space_d: &Rc<DenseVectorSpace>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> (
        Result<MultiVectorMatrix, ESymSolverStatus>,
        MultiVectorMatrix,
        MultiVectorMatrix,
        MultiVectorMatrix,
    ) {
        let n_cols = v_x.n_cols();

        // Allocate four per-block result MVMs.
        let mut out_x =
            MultiVectorMatrixSpace::new(n_cols, Rc::clone(space_x)).make_new_multi_vector();
        let mut out_s =
            MultiVectorMatrixSpace::new(n_cols, Rc::clone(space_s)).make_new_multi_vector();
        let mut out_c =
            MultiVectorMatrixSpace::new(n_cols, Rc::clone(space_c)).make_new_multi_vector();
        let mut out_d =
            MultiVectorMatrixSpace::new(n_cols, Rc::clone(space_d)).make_new_multi_vector();
        out_x.fill_with_new_vectors();
        out_s.fill_with_new_vectors();
        out_c.fill_with_new_vectors();
        out_d.fill_with_new_vectors();

        // Allocate zero RHS slots once; the four columns are reused
        // because we re-zero per call.
        let mut rhs_s = space_s.make_new_dense();
        rhs_s.set(0.0);
        let mut rhs_c = space_c.make_new_dense();
        rhs_c.set(0.0);
        let mut rhs_d = space_d.make_new_dense();
        rhs_d.set(0.0);

        for k in 0..n_cols {
            let rhs_x_dyn: &dyn Vector = v_x.get_vector(k).as_ref();
            let inner_rhs = AugSysRhs {
                rhs_x: rhs_x_dyn,
                rhs_s: rhs_s.as_dyn_vector(),
                rhs_c: rhs_c.as_dyn_vector(),
                rhs_d: rhs_d.as_dyn_vector(),
            };
            // Build solution slots (fresh each iteration).
            let mut sol_x = space_x.make_new_dense();
            let mut sol_s = space_s.make_new_dense();
            let mut sol_c = space_c.make_new_dense();
            let mut sol_d = space_d.make_new_dense();
            sol_x.set(0.0);
            sol_s.set(0.0);
            sol_c.set(0.0);
            sol_d.set(0.0);
            let inner_coeffs = inner_coeffs(&self.factor, coeffs);
            let status = {
                let mut sol = AugSysSol {
                    sol_x: &mut sol_x,
                    sol_s: &mut sol_s,
                    sol_c: &mut sol_c,
                    sol_d: &mut sol_d,
                };
                self.inner.solve(
                    &inner_coeffs,
                    &inner_rhs,
                    &mut sol,
                    check_neg_evals,
                    num_neg_evals,
                )
            };
            if self.inner.provides_inertia() {
                self.num_neg_evals = self.inner.number_of_neg_evals();
            }
            if status != ESymSolverStatus::Success {
                return (Err(status), out_s, out_c, out_d);
            }
            out_x.set_vector(k, Rc::new(sol_x) as Rc<dyn Vector>);
            out_s.set_vector(k, Rc::new(sol_s) as Rc<dyn Vector>);
            out_c.set_vector(k, Rc::new(sol_c) as Rc<dyn Vector>);
            out_d.set_vector(k, Rc::new(sol_d) as Rc<dyn Vector>);
        }
        (Ok(out_x), out_s, out_c, out_d)
    }
}

/// Build inner-solver coefficients that substitute `Wdiag` for `W`.
/// Free function (rather than method on `LowRankAugSystemSolver`) so
/// the borrow is on `&Factorization` only — leaving `self.inner`
/// available for `&mut`.
fn inner_coeffs<'b>(factor: &'b Factorization, coeffs: &AugSysCoeffs<'b>) -> AugSysCoeffs<'b> {
    let wdiag: &DiagMatrix = factor.wdiag.as_ref().expect("Wdiag unset").as_ref();
    AugSysCoeffs {
        w: Some(wdiag as &dyn SymMatrix),
        w_factor: 1.0,
        d_x: coeffs.d_x,
        delta_x: coeffs.delta_x,
        d_s: coeffs.d_s,
        delta_s: coeffs.delta_s,
        j_c: coeffs.j_c,
        d_c: coeffs.d_c,
        delta_c: coeffs.delta_c,
        j_d: coeffs.j_d,
        d_d: coeffs.d_d,
        delta_d: coeffs.delta_d,
    }
}

fn downcast_dense(v: &dyn Vector) -> &DenseVector {
    v.as_any()
        .downcast_ref::<DenseVector>()
        .expect("LowRankAugSystemSolver currently requires DenseVector RHS/solutions")
}

/// `DenseVector` doesn't implement `Clone`; this builds a fresh dense
/// vector in the same space populated with the same expanded values.
/// Cheap when the source is homogeneous.
fn clone_dense(src: &DenseVector) -> DenseVector {
    let mut out = src.space().make_new_dense();
    out.set_values(&src.expanded_values());
    out
}

fn zero_x_for(space_x: &Rc<DenseVectorSpace>, lr_w: &LowRankUpdateSymMatrix) -> DenseVector {
    // `MakeNew` either from the LR vector space (when reduced_diag is
    // active) or from the proto x-space. We don't have the LR vector
    // space surfaced directly, but B0 lives in either space; passing
    // None always means "no diag" so we just return a zero in space_x.
    let _ = lr_w;
    let mut z = space_x.make_new_dense();
    z.set(0.0);
    z
}

impl AugSystemSolver for LowRankAugSystemSolver {
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }

    fn number_of_neg_evals(&self) -> Index {
        if self.inner.provides_inertia() {
            self.inner.number_of_neg_evals()
        } else {
            self.num_neg_evals
        }
    }

    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        self.inner.last_solve_status()
    }

    fn set_timing_stats(&mut self, timing: Rc<TimingStatistics>) {
        self.inner.set_timing_stats(timing);
    }

    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        // Skip inertia checks when the inner solver doesn't provide
        // them — mirrors `IpLowRankAugSystemSolver.cpp:102-105`.
        let mut check_neg_evals = check_neg_evals;
        if !self.inner.provides_inertia() {
            check_neg_evals = false;
        }

        // Hessian-free / non-low-rank W: the least-square-multiplier
        // initialization (`init`) and the equality-multiplier estimates
        // (`eq_mult`) drive this same solver with their own zero W block
        // and `w_factor = 0` — there is no low-rank update to apply, so
        // bypass the SMW machinery and solve directly through the inner
        // augmented-system solver with the original coefficients.
        // `data.w` only carries a `LowRankUpdateSymMatrix` for the main
        // primal-dual solves (always `w_factor = 1`).
        let lr_w_opt = coeffs
            .w
            .and_then(|w| w.as_any().downcast_ref::<LowRankUpdateSymMatrix>());
        let Some(lr_w) = lr_w_opt else {
            let status = self
                .inner
                .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals);
            if self.inner.provides_inertia() {
                self.num_neg_evals = self.inner.number_of_neg_evals();
            }
            return status;
        };

        let needs_rebuild = self.first_call || self.augmented_system_requires_change(coeffs);
        if needs_rebuild {
            let status =
                self.update_factorization(lr_w, coeffs, rhs, check_neg_evals, num_neg_evals);
            if status != ESymSolverStatus::Success {
                return status;
            }
            self.store_cache(coeffs);
            self.first_call = false;
        }

        // 1. Diagonal solve through the inner aug-system solver.
        let ic = inner_coeffs(&self.factor, coeffs);
        let status = self
            .inner
            .solve(&ic, rhs, sol, check_neg_evals, num_neg_evals);
        if self.inner.provides_inertia() {
            self.num_neg_evals = self.inner.number_of_neg_evals();
        }
        if status != ESymSolverStatus::Success {
            return status;
        }

        // 2. SMW correction terms — mirror upstream's order:
        //    apply Utilde2 first, then Vtilde1 (cpp:210-227).
        if self.factor.utilde2_x.is_some() {
            self.apply_smw(/*sign=*/ 1.0, /*use_u=*/ true, rhs, sol);
        }
        if self.factor.vtilde1_x.is_some() {
            self.apply_smw(/*sign=*/ -1.0, /*use_u=*/ false, rhs, sol);
        }

        ESymSolverStatus::Success
    }
}

impl LowRankAugSystemSolver {
    /// Apply one SMW correction step:
    ///   `b = U_or_Vᵀ · rhs;  J⁻¹J⁻ᵀ b;  sol += sign · U_or_V · b`
    ///
    /// `use_u = true` selects `(Utilde2, J2, +1)`; `false` selects
    /// `(Vtilde1, J1, −1)` (sign passed in by caller).
    fn apply_smw(&self, sign: Number, use_u: bool, rhs: &AugSysRhs<'_>, sol: &mut AugSysSol<'_>) {
        let (mvx, mvs, mvc, mvd, j) = if use_u {
            (
                self.factor.utilde2_x.as_ref().unwrap(),
                self.factor.utilde2_s.as_ref().unwrap(),
                self.factor.utilde2_c.as_ref().unwrap(),
                self.factor.utilde2_d.as_ref().unwrap(),
                self.factor.j2.as_ref().unwrap(),
            )
        } else {
            (
                self.factor.vtilde1_x.as_ref().unwrap(),
                self.factor.vtilde1_s.as_ref().unwrap(),
                self.factor.vtilde1_c.as_ref().unwrap(),
                self.factor.vtilde1_d.as_ref().unwrap(),
                self.factor.j1.as_ref().unwrap(),
            )
        };
        let n = mvx.n_cols();
        // Build `b = M^T · crhs` from the four blocks. Reduction order
        // matches upstream's CompoundVector dot, which iterates blocks
        // in the order x, s, c, d (`IpCompoundVector.cpp::Dot`).
        let mut b_vec: Vec<Number> = Vec::with_capacity(n as usize);
        for k in 0..n {
            let dot = mvx.get_vector(k).dot(rhs.rhs_x)
                + mvs.get_vector(k).dot(rhs.rhs_s)
                + mvc.get_vector(k).dot(rhs.rhs_c)
                + mvd.get_vector(k).dot(rhs.rhs_d);
            b_vec.push(dot);
        }
        let space_b = DenseVectorSpace::new(n);
        let mut b = space_b.make_new_dense();
        b.set_values(&b_vec);
        // Apply J⁻¹ J⁻ᵀ in-place.
        j.cholesky_solve_vector(&mut b);
        // sol += sign · M · b  per block.
        mvx.mult_vector(sign, &b, 1.0, sol.sol_x);
        mvs.mult_vector(sign, &b, 1.0, sol.sol_s);
        mvc.mult_vector(sign, &b, 1.0, sol.sol_c);
        mvd.mult_vector(sign, &b, 1.0, sol.sol_d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrixSpace;
    use std::cell::Cell;

    /// Diagonal-solve stub: pretends the augmented system is just
    /// `(W + δ_x I) · sol_x = rhs_x` with `m_c = m_d = n_s = 0`. Reads
    /// `coeffs.w` as a `DiagMatrix` (i.e. the wdiag we built) and does
    /// a per-element divide. Plenty for the SMW test fixture.
    struct DiagInner {
        calls: Cell<usize>,
    }
    impl AugSystemSolver for DiagInner {
        fn provides_inertia(&self) -> bool {
            true
        }
        fn number_of_neg_evals(&self) -> Index {
            0
        }
        fn increase_quality(&mut self) -> bool {
            true
        }
        fn last_solve_status(&self) -> ESymSolverStatus {
            ESymSolverStatus::Success
        }
        fn solve(
            &mut self,
            coeffs: &AugSysCoeffs<'_>,
            rhs: &AugSysRhs<'_>,
            sol: &mut AugSysSol<'_>,
            _check_neg_evals: bool,
            _num_neg_evals: Index,
        ) -> ESymSolverStatus {
            self.calls.set(self.calls.get() + 1);
            let wdiag = coeffs
                .w
                .expect("DiagInner requires W")
                .as_any()
                .downcast_ref::<DiagMatrix>()
                .expect("DiagInner requires W to be a DiagMatrix");
            let diag_rc = wdiag.get_diag().expect("Wdiag has no diag set").clone();
            let diag = downcast_dense(diag_rc.as_ref()).expanded_values();
            let rhs_x = downcast_dense(rhs.rhs_x).expanded_values();
            let dx_vals: Option<Vec<Number>> =
                coeffs.d_x.map(|d| downcast_dense(d).expanded_values());
            let mut out = vec![0.0; rhs_x.len()];
            for i in 0..rhs_x.len() {
                let dx_i = match &dx_vals {
                    Some(v) => v[i],
                    None => 0.0,
                };
                let denom = diag[i] + dx_i + coeffs.delta_x;
                out[i] = rhs_x[i] / denom;
            }
            let sol_x_dv = sol
                .sol_x
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .unwrap();
            sol_x_dv.set_values(&out);
            // Other blocks stay zero — fixture has m_c = m_d = n_s = 0.
            ESymSolverStatus::Success
        }
    }

    fn dvec(space: &Rc<DenseVectorSpace>, vals: &[Number]) -> DenseVector {
        let mut v = space.make_new_dense();
        v.set_values(vals);
        v
    }

    fn dvec_rc(space: &Rc<DenseVectorSpace>, vals: &[Number]) -> Rc<DenseVector> {
        Rc::new(dvec(space, vals))
    }

    #[test]
    fn smw_recovers_low_rank_inverse() {
        // 1×1 system: W = b0 + v² (v ≠ 0); δ_x = 0.
        // Direct: sol = rhs / (b0 + v²).
        // SMW:    inner solves with diag b0 → sol_diag = rhs/b0;
        //         correction recovers rhs/(b0 + v²).
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        let b0_rc: Rc<dyn Vector> = dvec_rc(&space_x, &[2.0]);
        lr.set_diag(b0_rc);
        let v_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut v_mvm = v_space.make_new_multi_vector();
        v_mvm.set_vector(0, dvec_rc(&space_x, &[3.0]) as Rc<dyn Vector>);
        lr.set_v(Rc::new(v_mvm));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);

        let mut solver = LowRankAugSystemSolver::new(Box::new(DiagInner {
            calls: Cell::new(0),
        }));

        // Empty Jacobians.
        let j_c_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_d_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_c = j_c_space.make_new_dense_gen();
        let j_d = j_d_space.make_new_dense_gen();

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.0,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };

        let rhs_x = dvec(&space_x, &[5.0]);
        let rhs_s = dvec(&space_zero, &[]);
        let rhs_c = dvec(&space_zero, &[]);
        let rhs_d = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_s,
            rhs_c: &rhs_c,
            rhs_d: &rhs_d,
        };
        let mut sol_x = dvec(&space_x, &[0.0]);
        let mut sol_s = dvec(&space_zero, &[]);
        let mut sol_c = dvec(&space_zero, &[]);
        let mut sol_d = dvec(&space_zero, &[]);
        let mut sol = AugSysSol {
            sol_x: &mut sol_x,
            sol_s: &mut sol_s,
            sol_c: &mut sol_c,
            sol_d: &mut sol_d,
        };
        let status = solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        assert_eq!(status, ESymSolverStatus::Success);
        // Expected: 5 / (2 + 9) = 5/11.
        let got = sol_x.expanded_values()[0];
        let want = 5.0 / 11.0;
        assert!((got - want).abs() < 1e-12, "got {} want {}", got, want);
    }

    #[test]
    fn smw_with_u_only_applies_positive_correction() {
        // 1×1 system: W = b0 − u² (low-rank *negative* update).
        // Direct: sol = rhs / (b0 − u²).
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[5.0]));
        let u_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut u_mvm = u_space.make_new_multi_vector();
        u_mvm.set_vector(0, dvec_rc(&space_x, &[1.5]) as Rc<dyn Vector>);
        lr.set_u(Rc::new(u_mvm));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);

        let mut solver = LowRankAugSystemSolver::new(Box::new(DiagInner {
            calls: Cell::new(0),
        }));

        let j_c_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_d_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_c = j_c_space.make_new_dense_gen();
        let j_d = j_d_space.make_new_dense_gen();

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.0,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };

        let rhs_x = dvec(&space_x, &[7.0]);
        let rhs_s = dvec(&space_zero, &[]);
        let rhs_c = dvec(&space_zero, &[]);
        let rhs_d = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_s,
            rhs_c: &rhs_c,
            rhs_d: &rhs_d,
        };
        let mut sol_x = dvec(&space_x, &[0.0]);
        let mut sol_s = dvec(&space_zero, &[]);
        let mut sol_c = dvec(&space_zero, &[]);
        let mut sol_d = dvec(&space_zero, &[]);
        let mut sol = AugSysSol {
            sol_x: &mut sol_x,
            sol_s: &mut sol_s,
            sol_c: &mut sol_c,
            sol_d: &mut sol_d,
        };
        let status = solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        assert_eq!(status, ESymSolverStatus::Success);
        // Expected: 7 / (5 − 2.25) = 7 / 2.75.
        let got = sol_x.expanded_values()[0];
        let want = 7.0 / 2.75;
        assert!((got - want).abs() < 1e-12, "got {} want {}", got, want);
    }

    #[test]
    fn smw_reports_wrong_inertia_on_indefinite_negative_update() {
        // 1×1 system: W = b0 − u² with u² > b0, so B = 2 − 4 = −2 is
        // genuinely indefinite — the SR1 negative-curvature regime. The
        // SMW middle matrix M2 = 1 − Utilde2ᵀU = 1 − u²/b0 = −1 is then
        // not positive definite, so its Cholesky must fail and the solver
        // must report `WrongInertia` — the signal the perturbation handler
        // keys on to correct the step — rather than silently returning a
        // garbage solve. (`number_of_neg_evals` is not asserted here: with
        // a real inertia-providing inner solver it delegates to the inner;
        // the mock reports 0.)
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[2.0]));
        let u_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut u_mvm = u_space.make_new_multi_vector();
        u_mvm.set_vector(0, dvec_rc(&space_x, &[2.0]) as Rc<dyn Vector>);
        lr.set_u(Rc::new(u_mvm));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);

        let mut solver = LowRankAugSystemSolver::new(Box::new(DiagInner {
            calls: Cell::new(0),
        }));

        let j_c_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_d_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_c = j_c_space.make_new_dense_gen();
        let j_d = j_d_space.make_new_dense_gen();

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.0,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };

        let rhs_x = dvec(&space_x, &[1.0]);
        let rhs_s = dvec(&space_zero, &[]);
        let rhs_c = dvec(&space_zero, &[]);
        let rhs_d = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_s,
            rhs_c: &rhs_c,
            rhs_d: &rhs_d,
        };
        let mut sol_x = dvec(&space_x, &[0.0]);
        let mut sol_s = dvec(&space_zero, &[]);
        let mut sol_c = dvec(&space_zero, &[]);
        let mut sol_d = dvec(&space_zero, &[]);
        let mut sol = AugSysSol {
            sol_x: &mut sol_x,
            sol_s: &mut sol_s,
            sol_c: &mut sol_c,
            sol_d: &mut sol_d,
        };
        let status = solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        assert_eq!(status, ESymSolverStatus::WrongInertia);
    }

    #[test]
    fn smw_with_v_and_u_combines_corrections() {
        // 1×1 system: W = b0 + v² − u² (rank-2 update). Solve checks
        // both correction passes compose correctly.
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[10.0]));
        let v_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut v_mvm = v_space.make_new_multi_vector();
        v_mvm.set_vector(0, dvec_rc(&space_x, &[2.0]) as Rc<dyn Vector>);
        lr.set_v(Rc::new(v_mvm));
        let u_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut u_mvm = u_space.make_new_multi_vector();
        u_mvm.set_vector(0, dvec_rc(&space_x, &[1.0]) as Rc<dyn Vector>);
        lr.set_u(Rc::new(u_mvm));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);

        let mut solver = LowRankAugSystemSolver::new(Box::new(DiagInner {
            calls: Cell::new(0),
        }));

        let j_c_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_d_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_c = j_c_space.make_new_dense_gen();
        let j_d = j_d_space.make_new_dense_gen();

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.0,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };

        let rhs_x = dvec(&space_x, &[1.0]);
        let rhs_s = dvec(&space_zero, &[]);
        let rhs_c = dvec(&space_zero, &[]);
        let rhs_d = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_s,
            rhs_c: &rhs_c,
            rhs_d: &rhs_d,
        };
        let mut sol_x = dvec(&space_x, &[0.0]);
        let mut sol_s = dvec(&space_zero, &[]);
        let mut sol_c = dvec(&space_zero, &[]);
        let mut sol_d = dvec(&space_zero, &[]);
        let mut sol = AugSysSol {
            sol_x: &mut sol_x,
            sol_s: &mut sol_s,
            sol_c: &mut sol_c,
            sol_d: &mut sol_d,
        };
        let status = solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        assert_eq!(status, ESymSolverStatus::Success);
        // Expected: 1 / (10 + 4 − 1) = 1/13.
        let got = sol_x.expanded_values()[0];
        let want = 1.0 / 13.0;
        assert!((got - want).abs() < 1e-12, "got {} want {}", got, want);
    }

    #[test]
    fn unchanged_coeffs_skip_rebuild_after_first_call() {
        let mut lr_solver = LowRankAugSystemSolver::new(Box::new(DiagInner {
            calls: Cell::new(0),
        }));
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[2.0]));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);
        let j_c_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_d_space = pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, 1);
        let j_c = j_c_space.make_new_dense_gen();
        let j_d = j_d_space.make_new_dense_gen();
        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x: 0.001,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };
        let rhs_x = dvec(&space_x, &[1.0]);
        let rhs_zero = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_zero,
            rhs_c: &rhs_zero,
            rhs_d: &rhs_zero,
        };
        let mut sol_x = dvec(&space_x, &[0.0]);
        let mut sol_z1 = dvec(&space_zero, &[]);
        let mut sol_z2 = dvec(&space_zero, &[]);
        let mut sol_z3 = dvec(&space_zero, &[]);
        {
            let mut sol = AugSysSol {
                sol_x: &mut sol_x,
                sol_s: &mut sol_z1,
                sol_c: &mut sol_z2,
                sol_d: &mut sol_z3,
            };
            lr_solver.solve(&coeffs, &rhs, &mut sol, false, 0);
        }
        // Same coeffs → cache reports no change.
        assert!(!lr_solver.augmented_system_requires_change(&coeffs));
    }
}
