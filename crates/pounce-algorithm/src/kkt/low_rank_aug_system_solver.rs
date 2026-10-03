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
    /// Whether `inner` currently holds a numeric factorization of the
    /// matrix `inner_coeffs(&self.factor, coeffs)` describes — i.e. the
    /// `Wdiag`-substituted augmented system for the coefficients in
    /// `self.cache`.
    ///
    /// Set **only** immediately after an `inner.solve` returns `Success`
    /// against exactly that matrix, never as a side effect of a rebuild:
    /// `update_factorization` can legitimately perform zero inner solves
    /// (empty L-BFGS history — `get_v()` and `get_u()` both `None`, which
    /// happens on the first iteration and again whenever
    /// `limited_memory_max_skipping` clears the history mid-solve), and
    /// in that case the inner solver is still holding the *previous*
    /// iterate's factor of a different `Wdiag`.
    inner_has_factor: bool,
    /// The `num_neg_evals` target the cached factor was validated
    /// against, or `None` if it was produced without an inertia check.
    /// A fast path that skips re-factorizing must not also skip an
    /// inertia check the caller asked for against a different target.
    inner_factor_neg_evals: Option<Index>,
    /// Separate inner solver dedicated to the Hessian-free solves that
    /// take the bypass in [`LowRankAugSystemSolver::solve`], so that
    /// neither solver ever sees more than one W sparsity (gh#730).
    ///
    /// `None` restores the single-solver arrangement, in which both
    /// shapes share one inner solver and each alternation re-runs the
    /// backend's symbolic factorization. Unit tests that drive a stub
    /// inner solver directly use that arrangement; the builder always
    /// supplies a bypass solver on the limited-memory path.
    bypass: Option<Box<dyn AugSystemSolver>>,
    /// Whether the most recent `solve` took the Hessian-free bypass, so
    /// `last_solve_status` reports the solver that actually ran.
    last_solve_took_bypass: bool,
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
            inner_has_factor: false,
            inner_factor_neg_evals: None,
            bypass: None,
            last_solve_took_bypass: false,
        }
    }

    /// Same as [`Self::new`] but routes the Hessian-free solves through
    /// their own inner solver.
    ///
    /// Under limited-memory this solver drives its inner solver with two
    /// different (1,1) shapes: an empty W for the least-square multiplier
    /// initialization and the equality-multiplier estimates, and an
    /// `n`-diagonal `B0` for the main primal-dual solves.
    /// `StdAugSystemSolver` keys its structure signature on W's nonzero
    /// count, so the two alternate and every alternation re-runs the
    /// backend's symbolic factorization — for MA57 with
    /// `ma57_pivot_order = 5` a full MeTiS nested dissection of the whole
    /// KKT system, ~0.4 s per call on a 118 276-row model (gh#730).
    ///
    /// Ipopt never pays this because its low-rank layer owns those
    /// solves, so its inner solver sees one shape for its whole life —
    /// `IpTSymLinearSolver.cpp:182` asserts exactly that. Giving the
    /// bypass its own solver reproduces that invariant on both sides
    /// instead of on neither.
    ///
    /// Deliberately **not** done by making the two shapes agree — e.g.
    /// substituting a zeroed `n`-diagonal for the empty W, which is what
    /// the shape proposed in gh#730 does. That is numerically exact, but
    /// it changes the pattern the backend orders, and a different
    /// ordering rounds differently: swept, it takes `pooling_rt2stp`
    /// under MA57 from `obj = -4391.83` to `-3273.95` (both "Optimal",
    /// 25% apart) and `cresc4` from 110 to 267 iterations. Two solvers
    /// hand each path exactly the matrix and the ordering it already
    /// had, so the win is free of trajectory movement rather than paid
    /// for with it.
    ///
    /// The cost is one additional numeric factorization resident at
    /// once. That is the deliberate trade: memory is measurable and
    /// boundable, and a trajectory regression is what has repeatedly
    /// shipped here undetected (`dev-notes/trajectory-regressions-and-\
    /// the-fixture-sweep.md`).
    pub fn with_bypass_solver(
        inner: Box<dyn AugSystemSolver>,
        bypass: Box<dyn AugSystemSolver>,
    ) -> Self {
        Self {
            bypass: Some(bypass),
            ..Self::new(inner)
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
        // `Wdiag` is about to be replaced, so whatever the inner solver
        // holds is a factor of the *old* matrix from here on. Clearing
        // first (rather than setting the flag at the end) is what makes
        // the zero-column rebuild safe: when the L-BFGS history is empty
        // both `get_v()` and `get_u()` are `None`, this function performs
        // no inner solve at all, and the flag must stay false so the
        // diagonal solve in `solve` factorizes instead of back-solving
        // against the previous iterate's factor.
        self.inner_has_factor = false;
        self.inner_factor_neg_evals = None;

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
        let n_cols_us = n_cols as usize;

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

        // Every column here shares one matrix, so only the first one
        // needs a factorization; the rest are back-substitutions
        // against it. Batching is the only way `nrhs > 1` reaches the
        // backend at all — both FERAL (`solve_many_into`) and MA57
        // (`ma57cd_` with `nrhs`) block the triangular solves, and
        // neither can do so one column at a time (gh#729).
        //
        // Three paths, in preference order:
        //
        //  1. cold inner solver, backend affirms bit-identity at
        //     `n_cols`: one `try_solve_many_flat` — factorize and
        //     substitute every column together, which is upstream's
        //     single `MultiSolve` (`IpLowRankAugSystemSolver.cpp:487`);
        //  2. otherwise: factorize on column 0 through the single-RHS
        //     path, then batch the remaining columns. Correct, but it
        //     streams the factor twice;
        //  3. backend declines the packed path entirely: one
        //     single-RHS solve per column.
        //
        // Who answers the bit-identity gate matters as much as where it
        // is asked. FERAL answers from a measured width ceiling. MA57
        // blocks at every width and so declines by default; it takes
        // paths 1 and 2 only when `ma57_batched_backsolve` is on, which
        // is a permission the user grants and not a measurement anyone
        // has made — see `dev-notes/ma57-batched-backsolve.md`.
        let n_x = space_x.dim() as usize;
        let n_s = space_s.dim() as usize;
        let n_c = space_c.dim() as usize;
        let n_d = space_d.dim() as usize;
        let dim = n_x + n_s + n_c + n_d;

        // Cold inner solver: factorize and back-substitute every column
        // in ONE backend call, which is what upstream's single
        // `MultiSolve` does (`IpLowRankAugSystemSolver.cpp:487`).
        // Paying the factorization through the single-RHS path and then
        // batching the rest streams the factor twice — a sparse
        // triangular solve costs `F + nrhs*W` with `F` several times
        // `W` on a KKT this size, so the split throws away one `F` per
        // SMW update. The same bit-identity gate as the warm batch
        // below applies, at the wider `n_cols_us`.
        //
        // Note there is deliberately no `dim == inner.system_dim()`
        // precondition here, unlike the warm batch below. Cold, the
        // inner solver has not assembled yet, so its `system_dim()` is
        // still 0 and that check would reject every first factorization
        // — silently, leaving the merged path unexercised by any mock
        // whose `system_dim()` is 0 when cold. `try_solve_many_flat`
        // assembles first and then declines if the packed length
        // disagrees, which is the same guard applied where the answer
        // is actually known.
        if !self.inner_has_factor
            && n_cols_us > 1
            && dim > 0
            && self.inner.multi_solve_matches_single_solve(n_cols_us)
        {
            let mut packed = vec![0.0; dim * n_cols_us];
            let mut packed_ok = true;
            for k in 0..n_cols_us {
                let col = &mut packed[k * dim..k * dim + n_x];
                if !copy_dense_into(v_x.get_vector(k as Index).as_ref(), col) {
                    packed_ok = false;
                    break;
                }
                // s/c/d blocks are zero by construction; `packed`
                // starts zeroed.
            }
            if packed_ok {
                let ic = inner_coeffs(&self.factor, coeffs);
                if let Some(status) = self.inner.try_solve_many_flat(
                    &ic,
                    &mut packed,
                    n_cols_us,
                    check_neg_evals,
                    num_neg_evals,
                ) {
                    if self.inner.provides_inertia() {
                        self.num_neg_evals = self.inner.number_of_neg_evals();
                    }
                    if status != ESymSolverStatus::Success {
                        self.inner_has_factor = false;
                        self.inner_factor_neg_evals = None;
                        return (Err(status), out_s, out_c, out_d);
                    }
                    self.inner_has_factor = true;
                    self.inner_factor_neg_evals = check_neg_evals.then_some(num_neg_evals);
                    for k in 0..n_cols_us {
                        let col = &packed[k * dim..(k + 1) * dim];
                        let mut sol_x = space_x.make_new_dense();
                        let mut sol_s = space_s.make_new_dense();
                        let mut sol_c = space_c.make_new_dense();
                        let mut sol_d = space_d.make_new_dense();
                        sol_x.set_values(&col[..n_x]);
                        sol_s.set_values(&col[n_x..n_x + n_s]);
                        sol_c.set_values(&col[n_x + n_s..n_x + n_s + n_c]);
                        sol_d.set_values(&col[n_x + n_s + n_c..]);
                        out_x.set_vector(k as Index, Rc::new(sol_x) as Rc<dyn Vector>);
                        out_s.set_vector(k as Index, Rc::new(sol_s) as Rc<dyn Vector>);
                        out_c.set_vector(k as Index, Rc::new(sol_c) as Rc<dyn Vector>);
                        out_d.set_vector(k as Index, Rc::new(sol_d) as Rc<dyn Vector>);
                    }
                    return (Ok(out_x), out_s, out_c, out_d);
                }
            }
        }

        let mut k0 = 0usize;
        if !self.inner_has_factor && n_cols_us > 0 {
            let rhs_x_dyn: &dyn Vector = v_x.get_vector(0).as_ref();
            match self.solve_one_column(
                rhs_x_dyn,
                &rhs_s,
                &rhs_c,
                &rhs_d,
                coeffs,
                space_x,
                space_s,
                space_c,
                space_d,
                check_neg_evals,
                num_neg_evals,
            ) {
                Ok((sol_x, sol_s, sol_c, sol_d)) => {
                    out_x.set_vector(0, Rc::new(sol_x) as Rc<dyn Vector>);
                    out_s.set_vector(0, Rc::new(sol_s) as Rc<dyn Vector>);
                    out_c.set_vector(0, Rc::new(sol_c) as Rc<dyn Vector>);
                    out_d.set_vector(0, Rc::new(sol_d) as Rc<dyn Vector>);
                }
                Err(status) => return (Err(status), out_s, out_c, out_d),
            }
            k0 = 1;
        }

        // Batched back-substitution for columns `k0..n_cols`. Declines
        // (leaving `k0` untouched for the loop below) when the inner
        // solver does not expose the packed path, does not report a
        // dimension, or hands us a column we cannot read as a dense
        // slice.
        if k0 < n_cols_us {
            let nrhs = n_cols_us - k0;
            // The batch is a pure time optimization: these columns feed
            // the SMW correction of an iterate whose trajectory must not
            // move. A backend whose blocked substitution reassociates
            // returns a tolerance-equal but different answer, and on a
            // nonconvex problem that is enough to select a different local
            // optimum — MA57 takes `pooling_rt2stp` to an objective 25%
            // worse while still reporting `Optimal Solution Found` (gh#729).
            // So the backend has to affirm bit-identity at this width, and
            // the default answer is no.
            if nrhs > 1
                && dim > 0
                && dim == self.inner.system_dim() as usize
                && self.inner.multi_solve_matches_single_solve(nrhs)
            {
                let mut packed = vec![0.0; dim * nrhs];
                let mut packed_ok = true;
                for (j, k) in (k0..n_cols_us).enumerate() {
                    let col = &mut packed[j * dim..j * dim + n_x];
                    if !copy_dense_into(v_x.get_vector(k as Index).as_ref(), col) {
                        packed_ok = false;
                        break;
                    }
                    // The s/c/d blocks of the RHS are zero by
                    // construction, and `packed` starts zeroed.
                }
                if packed_ok {
                    let ic = inner_coeffs(&self.factor, coeffs);
                    if let Some(status) = self.inner.try_resolve_many_flat(&ic, &mut packed, nrhs) {
                        if self.inner.provides_inertia() {
                            self.num_neg_evals = self.inner.number_of_neg_evals();
                        }
                        if status != ESymSolverStatus::Success {
                            self.inner_has_factor = false;
                            self.inner_factor_neg_evals = None;
                            return (Err(status), out_s, out_c, out_d);
                        }
                        for (j, k) in (k0..n_cols_us).enumerate() {
                            let col = &packed[j * dim..(j + 1) * dim];
                            let mut sol_x = space_x.make_new_dense();
                            let mut sol_s = space_s.make_new_dense();
                            let mut sol_c = space_c.make_new_dense();
                            let mut sol_d = space_d.make_new_dense();
                            sol_x.set_values(&col[..n_x]);
                            sol_s.set_values(&col[n_x..n_x + n_s]);
                            sol_c.set_values(&col[n_x + n_s..n_x + n_s + n_c]);
                            sol_d.set_values(&col[n_x + n_s + n_c..]);
                            out_x.set_vector(k as Index, Rc::new(sol_x) as Rc<dyn Vector>);
                            out_s.set_vector(k as Index, Rc::new(sol_s) as Rc<dyn Vector>);
                            out_c.set_vector(k as Index, Rc::new(sol_c) as Rc<dyn Vector>);
                            out_d.set_vector(k as Index, Rc::new(sol_d) as Rc<dyn Vector>);
                        }
                        return (Ok(out_x), out_s, out_c, out_d);
                    }
                }
            }
        }

        // Fallback: one single-RHS back-substitution per column.
        for k in k0..n_cols_us {
            let rhs_x_dyn: &dyn Vector = v_x.get_vector(k as Index).as_ref();
            match self.solve_one_column(
                rhs_x_dyn,
                &rhs_s,
                &rhs_c,
                &rhs_d,
                coeffs,
                space_x,
                space_s,
                space_c,
                space_d,
                check_neg_evals,
                num_neg_evals,
            ) {
                Ok((sol_x, sol_s, sol_c, sol_d)) => {
                    out_x.set_vector(k as Index, Rc::new(sol_x) as Rc<dyn Vector>);
                    out_s.set_vector(k as Index, Rc::new(sol_s) as Rc<dyn Vector>);
                    out_c.set_vector(k as Index, Rc::new(sol_c) as Rc<dyn Vector>);
                    out_d.set_vector(k as Index, Rc::new(sol_d) as Rc<dyn Vector>);
                }
                Err(status) => return (Err(status), out_s, out_c, out_d),
            }
        }
        (Ok(out_x), out_s, out_c, out_d)
    }

    /// One column of [`Self::multi_solve_block`] through the inner
    /// solver's single-RHS path: `solve` (factorize) when the inner
    /// solver is cold, `resolve` (back-substitute) when it is not.
    /// Carries the inertia and factor bookkeeping either way, so the
    /// batched path and the fallback loop agree on solver state.
    #[allow(clippy::too_many_arguments)]
    fn solve_one_column(
        &mut self,
        rhs_x: &dyn Vector,
        rhs_s: &DenseVector,
        rhs_c: &DenseVector,
        rhs_d: &DenseVector,
        coeffs: &AugSysCoeffs<'_>,
        space_x: &Rc<DenseVectorSpace>,
        space_s: &Rc<DenseVectorSpace>,
        space_c: &Rc<DenseVectorSpace>,
        space_d: &Rc<DenseVectorSpace>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> Result<(DenseVector, DenseVector, DenseVector, DenseVector), ESymSolverStatus> {
        let inner_rhs = AugSysRhs {
            rhs_x,
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
        let ic = inner_coeffs(&self.factor, coeffs);
        let reuse = self.inner_has_factor;
        let status = {
            let mut sol = AugSysSol {
                sol_x: &mut sol_x,
                sol_s: &mut sol_s,
                sol_c: &mut sol_c,
                sol_d: &mut sol_d,
            };
            if reuse {
                self.inner.resolve(&ic, &inner_rhs, &mut sol)
            } else {
                self.inner
                    .solve(&ic, &inner_rhs, &mut sol, check_neg_evals, num_neg_evals)
            }
        };
        if self.inner.provides_inertia() {
            self.num_neg_evals = self.inner.number_of_neg_evals();
        }
        if status != ESymSolverStatus::Success {
            self.inner_has_factor = false;
            self.inner_factor_neg_evals = None;
            return Err(status);
        }
        if !reuse {
            self.inner_has_factor = true;
            self.inner_factor_neg_evals = check_neg_evals.then_some(num_neg_evals);
        }
        Ok((sol_x, sol_s, sol_c, sol_d))
    }
}

/// Copy a `dyn Vector` block into `dst`, expanding the homogeneous
/// (single-scalar) representation. Returns `false` — leaving `dst`
/// untouched — when the block is not a [`DenseVector`] or its length
/// disagrees, which the batched path in [`multi_solve_block`] treats as
/// "decline and take the per-column loop" rather than panicking.
///
/// [`multi_solve_block`]: LowRankAugSystemSolver::multi_solve_block
fn copy_dense_into(src: &dyn Vector, dst: &mut [Number]) -> bool {
    if dst.is_empty() {
        return true;
    }
    let Some(dv) = src.as_any().downcast_ref::<DenseVector>() else {
        return false;
    };
    if dv.dim() as usize != dst.len() {
        return false;
    }
    if dv.is_homogeneous() {
        let v = dv.scalar();
        dst.iter_mut().for_each(|x| *x = v);
    } else {
        dst.copy_from_slice(dv.values());
    }
    true
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

    fn singularity_certified(&self) -> bool {
        self.inner.singularity_certified()
    }

    fn number_of_neg_evals(&self) -> Index {
        if self.inner.provides_inertia() {
            self.inner.number_of_neg_evals()
        } else {
            self.num_neg_evals
        }
    }

    fn increase_quality(&mut self) -> bool {
        // The inner solver drops its cached factor here (it re-pivots at
        // a tighter tolerance), so ours is stale too.
        self.inner_has_factor = false;
        self.inner_factor_neg_evals = None;
        // Escalate both: a caller asking for tighter pivoting wants it on
        // the next solve whichever path that takes, and the two solvers
        // hold independent backend state.
        let inner = self.inner.increase_quality();
        let bypass = self
            .bypass
            .as_deref_mut()
            .map(|b| b.increase_quality())
            .unwrap_or(false);
        inner || bypass
    }

    fn last_solve_status(&self) -> ESymSolverStatus {
        // Whichever solver actually ran last — otherwise a bypass solve
        // reports the stale status of the previous main solve.
        match (self.last_solve_took_bypass, self.bypass.as_deref()) {
            (true, Some(bypass)) => bypass.last_solve_status(),
            _ => self.inner.last_solve_status(),
        }
    }

    fn set_timing_stats(&mut self, timing: Rc<TimingStatistics>) {
        // Both, or the bypass solver's symbolic and factorization time
        // lands in no phase at all — which is precisely the row gh#730
        // is about.
        if let Some(bypass) = self.bypass.as_deref_mut() {
            bypass.set_timing_stats(Rc::clone(&timing));
        }
        self.inner.set_timing_stats(timing);
    }

    fn set_slack_scaling(&mut self, nx: Index, s_scale: &[Number]) {
        // Both: the bypass solver assembles the same (2,2) slack block
        // and would otherwise scale it differently from the main path.
        if let Some(bypass) = self.bypass.as_deref_mut() {
            bypass.set_slack_scaling(nx, s_scale);
        }
        self.inner.set_slack_scaling(nx, s_scale);
    }

    fn handles_low_rank_w(&self) -> bool {
        true
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
            // The inner solver is about to factor a *different* matrix
            // (the caller's own W, not our `Wdiag` substitution), so any
            // cached SMW factor state no longer describes what it holds.
            // Only invalidate the shared-solver cache when the solve is
            // actually about to land on `self.inner`. With a dedicated
            // bypass solver `self.inner` keeps holding a valid factor of
            // the `Wdiag`-substituted system, and dropping it here would
            // force a needless refactorization on the next main solve.
            self.last_solve_took_bypass = true;
            let target = match self.bypass.as_deref_mut() {
                Some(bypass) => bypass,
                None => {
                    // The inner solver is about to factor a *different*
                    // matrix (the caller's own W, not our `Wdiag`
                    // substitution), so any cached SMW factor state no
                    // longer describes what it holds.
                    self.inner_has_factor = false;
                    self.inner_factor_neg_evals = None;
                    self.inner.as_mut()
                }
            };
            let status = target.solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals);
            if target.provides_inertia() {
                self.num_neg_evals = target.number_of_neg_evals();
            }
            return status;
        };

        self.last_solve_took_bypass = false;

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

        // 1. Diagonal solve through the inner aug-system solver. When we
        //    already hold a factor of this exact matrix — the rebuild
        //    above just produced one while solving for the SMW columns,
        //    or nothing has changed since the last call — this is a
        //    back-substitution rather than a fresh factorization.
        //
        //    Skipping the factorization also skips the inertia check that
        //    goes with it, so only take the fast path when the cached
        //    factor was already validated against the same target. In
        //    practice it always has been: a rebuild runs its first column
        //    with this call's own `check_neg_evals`/`num_neg_evals` and
        //    bails on `WrongInertia` before reaching here.
        let reuse = self.inner_has_factor
            && (!check_neg_evals || self.inner_factor_neg_evals == Some(num_neg_evals));
        let ic = inner_coeffs(&self.factor, coeffs);
        let status = if reuse {
            self.inner.resolve(&ic, rhs, sol)
        } else {
            self.inner
                .solve(&ic, rhs, sol, check_neg_evals, num_neg_evals)
        };
        if self.inner.provides_inertia() {
            self.num_neg_evals = self.inner.number_of_neg_evals();
        }
        if status != ESymSolverStatus::Success {
            self.inner_has_factor = false;
            self.inner_factor_neg_evals = None;
            return status;
        }
        if !reuse {
            self.inner_has_factor = true;
            self.inner_factor_neg_evals = check_neg_evals.then_some(num_neg_evals);
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

    /// Back-substitution against the cached factor, plus the same SMW
    /// corrections `solve` applies.
    ///
    /// Without this override the trait default falls through to `solve`,
    /// which re-factorizes — and because `PdFullSpaceSolver`'s iterative
    /// refinement and its same-matrix fast path both come in through
    /// `resolve`, that made the majority of the augmented-system solves
    /// on the limited-memory path re-factorize a matrix that had not
    /// changed. It also left `LinearSystemBackSolve` reading 0.000 s for
    /// a whole run, since the only back-solve timer guard lives on the
    /// path nothing reached (gh#698).
    ///
    /// Every condition that cannot be served falls back to `solve`, so
    /// this can lose an optimization but cannot change an answer — the
    /// same defensive shape as `StdAugSystemSolver::resolve`'s own
    /// `have_factor` fallback.
    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        // The fast path is only valid for a low-rank W whose SMW
        // factorization we hold and whose coefficients have not moved
        // since we built it. `first_call` additionally guarantees
        // `self.factor.wdiag` is populated for `inner_coeffs`.
        //
        // The non-low-rank bypass deliberately falls through to `solve`
        // rather than forwarding to `inner.resolve`: on that path the
        // inner solver's factor is of the caller's own W, which our
        // tag cache does not track, so we cannot certify it here.
        let is_low_rank = coeffs
            .w
            .and_then(|w| w.as_any().downcast_ref::<LowRankUpdateSymMatrix>())
            .is_some();
        if !is_low_rank
            || self.first_call
            || !self.inner_has_factor
            || self.augmented_system_requires_change(coeffs)
        {
            return self.solve(coeffs, rhs, sol, false, 0);
        }

        let ic = inner_coeffs(&self.factor, coeffs);
        let status = self.inner.resolve(&ic, rhs, sol);
        if status != ESymSolverStatus::Success {
            self.inner_has_factor = false;
            self.inner_factor_neg_evals = None;
            return status;
        }

        // Same correction order as `solve` (cpp:210-227).
        if self.factor.utilde2_x.is_some() {
            self.apply_smw(/*sign=*/ 1.0, /*use_u=*/ true, rhs, sol);
        }
        if self.factor.vtilde1_x.is_some() {
            self.apply_smw(/*sign=*/ -1.0, /*use_u=*/ false, rhs, sol);
        }

        ESymSolverStatus::Success
    }

    // `try_resolve_many_flat` is deliberately **not** overridden here.
    //
    // It hands back a raw `K⁻¹`-applied result that the caller
    // (`PdFullSpaceSolver::solve_many_cached`) unpacks and uses directly,
    // with no hook to apply the SMW correction afterwards. Our operator
    // is `(K + low-rank)⁻¹`, so forwarding the flat path to the inner
    // solver would silently drop the correction on every column and
    // return a plausible, wrong solution under a `Success` status.
    //
    // The trait default returns `None`, which the caller documents as
    // "fast path not taken, fall back to looping `solve`" — correct, and
    // the only safe answer for this wrapper unless the packed path grows
    // a way to post-process each column.
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
    use std::cell::{Cell, RefCell};

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

    /// Inner mock that models `StdAugSystemSolver`'s factor / back-solve
    /// split, which `DiagInner` above does not: `solve` "factorizes" by
    /// capturing the diagonal it was handed, and `resolve` back-solves
    /// against whatever was captured last.
    ///
    /// `resolve` deliberately ignores the coefficients it is passed and
    /// uses the cached ones. That is what a real cached factorization
    /// does, and it means a wrapper that back-solves against a stale
    /// factor produces a visibly *wrong answer* here, not merely a wrong
    /// count.
    #[derive(Default)]
    struct InnerStats {
        factorizations: Cell<usize>,
        backsolves: Cell<usize>,
        batched_calls: Cell<usize>,
        batched_cols: Cell<usize>,
        /// Calls to the *factorizing* multi-RHS path, and the columns
        /// they carried. Separate from `batched_calls` so a test can
        /// tell "one merged call" from "a factorization plus a batch".
        factor_batched_calls: Cell<usize>,
        factor_batched_cols: Cell<usize>,
        factored_diag: RefCell<Vec<Number>>,
        factored_delta_x: Cell<Number>,
    }

    struct CountingInner {
        stats: Rc<InnerStats>,
        /// When set, the mock exposes `system_dim` / `try_resolve_many_flat`
        /// the way `StdAugSystemSolver` does, so the batched arm of
        /// `multi_solve_block` is reachable. Off by default: the trait
        /// default `system_dim() == 0` is what a mock without the packed
        /// path looks like, and that is the arm the other tests exercise.
        packed: bool,
    }

    impl CountingInner {
        /// Returns the mock and a handle on its counters, so the test can
        /// read them after the solver has taken ownership of the box.
        fn new() -> (Self, Rc<InnerStats>) {
            let stats = Rc::new(InnerStats::default());
            (
                Self {
                    stats: Rc::clone(&stats),
                    packed: false,
                },
                stats,
            )
        }

        /// Same mock, but advertising the packed multi-RHS back-solve.
        fn with_packed_path() -> (Self, Rc<InnerStats>) {
            let stats = Rc::new(InnerStats::default());
            (
                Self {
                    stats: Rc::clone(&stats),
                    packed: true,
                },
                stats,
            )
        }

        /// `sol_x = rhs_x / (diag + delta_x)`, from the captured factor.
        fn apply_cached(&self, rhs: &AugSysRhs<'_>, sol: &mut AugSysSol<'_>) {
            let rhs_x = downcast_dense(rhs.rhs_x).expanded_values();
            let diag = self.stats.factored_diag.borrow();
            let delta_x = self.stats.factored_delta_x.get();
            let out: Vec<Number> = (0..rhs_x.len())
                .map(|i| rhs_x[i] / (diag[i] + delta_x))
                .collect();
            sol.sol_x
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .unwrap()
                .set_values(&out);
        }
    }

    impl AugSystemSolver for CountingInner {
        fn provides_inertia(&self) -> bool {
            false
        }
        fn number_of_neg_evals(&self) -> Index {
            0
        }
        fn increase_quality(&mut self) -> bool {
            false
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
            self.stats
                .factorizations
                .set(self.stats.factorizations.get() + 1);
            let wdiag = coeffs
                .w
                .expect("CountingInner requires W")
                .as_any()
                .downcast_ref::<DiagMatrix>()
                .expect("CountingInner requires W to be a DiagMatrix");
            let diag_rc = wdiag.get_diag().expect("Wdiag has no diag set").clone();
            *self.stats.factored_diag.borrow_mut() =
                downcast_dense(diag_rc.as_ref()).expanded_values();
            self.stats.factored_delta_x.set(coeffs.delta_x);
            self.apply_cached(rhs, sol);
            ESymSolverStatus::Success
        }
        fn resolve(
            &mut self,
            _coeffs: &AugSysCoeffs<'_>,
            rhs: &AugSysRhs<'_>,
            sol: &mut AugSysSol<'_>,
        ) -> ESymSolverStatus {
            assert!(
                !self.stats.factored_diag.borrow().is_empty(),
                "resolve reached with no cached factor"
            );
            self.stats.backsolves.set(self.stats.backsolves.get() + 1);
            self.apply_cached(rhs, sol);
            ESymSolverStatus::Success
        }
        fn system_dim(&self) -> Index {
            if self.packed {
                self.stats.factored_diag.borrow().len() as Index
            } else {
                0
            }
        }
        /// Mirrors `StdAugSystemSolver`: declines when cold, otherwise
        /// applies the cached factor to every packed column in place.
        /// Like the real one it ignores the coefficients it is handed and
        /// uses the cached ones, so back-solving against a stale factor
        /// shows up as a wrong answer rather than a wrong count.
        fn multi_solve_matches_single_solve(&self, _nrhs: usize) -> bool {
            self.packed
        }

        /// The factorizing counterpart: captures the factor like
        /// `solve` does, then applies it to every packed column. One
        /// call, one factorization, all columns — which is the whole
        /// point of the path.
        fn try_solve_many_flat(
            &mut self,
            coeffs: &AugSysCoeffs<'_>,
            packed_rhs: &mut [Number],
            nrhs: usize,
            _check_neg_evals: bool,
            _num_neg_evals: Index,
        ) -> Option<ESymSolverStatus> {
            if !self.packed {
                return None;
            }
            let wdiag = coeffs
                .w
                .expect("CountingInner requires W")
                .as_any()
                .downcast_ref::<DiagMatrix>()
                .expect("CountingInner requires W to be a DiagMatrix");
            let diag_rc = wdiag.get_diag().expect("Wdiag has no diag set").clone();
            let diag = downcast_dense(diag_rc.as_ref()).expanded_values();
            let dim = diag.len();
            if packed_rhs.len() != dim * nrhs {
                return None;
            }
            *self.stats.factored_diag.borrow_mut() = diag.clone();
            self.stats.factored_delta_x.set(coeffs.delta_x);
            self.stats
                .factorizations
                .set(self.stats.factorizations.get() + 1);
            self.stats
                .factor_batched_calls
                .set(self.stats.factor_batched_calls.get() + 1);
            self.stats
                .factor_batched_cols
                .set(self.stats.factor_batched_cols.get() + nrhs);
            let delta_x = coeffs.delta_x;
            for col in packed_rhs.chunks_mut(dim) {
                for (i, x) in col.iter_mut().enumerate() {
                    *x /= diag[i] + delta_x;
                }
            }
            Some(ESymSolverStatus::Success)
        }

        fn try_resolve_many_flat(
            &mut self,
            _coeffs: &AugSysCoeffs<'_>,
            packed_rhs: &mut [Number],
            nrhs: usize,
        ) -> Option<ESymSolverStatus> {
            if !self.packed {
                return None;
            }
            let diag = self.stats.factored_diag.borrow();
            if diag.is_empty() {
                return None;
            }
            let dim = diag.len();
            if packed_rhs.len() != dim * nrhs {
                return Some(ESymSolverStatus::FatalError);
            }
            self.stats
                .batched_calls
                .set(self.stats.batched_calls.get() + 1);
            self.stats
                .batched_cols
                .set(self.stats.batched_cols.get() + nrhs);
            let delta_x = self.stats.factored_delta_x.get();
            for col in packed_rhs.chunks_mut(dim) {
                for (i, x) in col.iter_mut().enumerate() {
                    *x /= diag[i] + delta_x;
                }
            }
            Some(ESymSolverStatus::Success)
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

    // ---- gh#698: factorization reuse ----

    /// The empty `m_c = m_d = 0` Jacobians every fixture in this section
    /// uses, over `n_x` variables.
    fn empty_jacobians(
        n_x: Index,
    ) -> (
        pounce_linalg::dense_gen_matrix::DenseGenMatrix,
        pounce_linalg::dense_gen_matrix::DenseGenMatrix,
    ) {
        (
            pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, n_x).make_new_dense_gen(),
            pounce_linalg::dense_gen_matrix::DenseGenMatrixSpace::new(0, n_x).make_new_dense_gen(),
        )
    }

    #[test]
    fn smw_columns_share_one_factorization() {
        // W = diag(2) + v vᵀ with a 3-column V. Upstream issues a single
        // `MultiSolve` for all columns and factorizes once; we must do
        // the same: 1 factorization, then back-solves for the remaining
        // two columns and for the diagonal solve.
        let space_x = DenseVectorSpace::new(3);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(3, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[2.0, 3.0, 4.0]));
        let v_space = MultiVectorMatrixSpace::new(3, Rc::clone(&space_x));
        let mut v = v_space.make_new_multi_vector();
        v.set_vector(0, dvec_rc(&space_x, &[0.5, 0.0, 0.0]));
        v.set_vector(1, dvec_rc(&space_x, &[0.0, 0.5, 0.0]));
        v.set_vector(2, dvec_rc(&space_x, &[0.0, 0.0, 0.5]));
        lr.set_v(Rc::new(v));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);
        let (j_c, j_d) = empty_jacobians(3);
        let delta_x = 0.0;

        let (inner, stats) = CountingInner::new();
        let mut solver = LowRankAugSystemSolver::new(Box::new(inner));

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x,
            d_s: None,
            delta_s: 0.0,
            j_c: &j_c as &dyn Matrix,
            d_c: None,
            delta_c: 0.0,
            j_d: &j_d as &dyn Matrix,
            d_d: None,
            delta_d: 0.0,
        };
        let rhs_x = dvec(&space_x, &[1.0, 1.0, 1.0]);
        let rhs_zero = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_zero,
            rhs_c: &rhs_zero,
            rhs_d: &rhs_zero,
        };
        let (mut sx, mut z1, mut z2, mut z3) = (
            dvec(&space_x, &[0.0, 0.0, 0.0]),
            dvec(&space_zero, &[]),
            dvec(&space_zero, &[]),
            dvec(&space_zero, &[]),
        );
        {
            let mut sol = AugSysSol {
                sol_x: &mut sx,
                sol_s: &mut z1,
                sol_c: &mut z2,
                sol_d: &mut z3,
            };
            assert_eq!(
                solver.solve(&coeffs, &rhs, &mut sol, false, 0),
                ESymSolverStatus::Success
            );
        }
        assert_eq!(
            stats.factorizations.get(),
            1,
            "three SMW columns plus the diagonal solve must share one factorization"
        );
        assert_eq!(
            stats.backsolves.get(),
            3,
            "2 remaining V columns + diagonal solve"
        );
    }

    #[test]
    fn batched_smw_columns_match_the_per_column_path() {
        // The batched arm of `multi_solve_block` is unreachable from a
        // mock that leaves `system_dim()` at its 0 default, so every
        // other test in this file exercises the per-column fallback and
        // a green suite says nothing about the packed path (gh#729).
        // Drive the identical problem down both arms and require the
        // solutions to agree bit-for-bit: the batched call must be work
        // removed, not work re-associated.
        fn run(packed: bool) -> (Vec<Number>, Rc<InnerStats>) {
            let space_x = DenseVectorSpace::new(3);
            let space_zero = DenseVectorSpace::new(0);
            let lr_space = LowRankUpdateSymMatrixSpace::new(3, None, false);
            let mut lr = lr_space.make_new_low_rank();
            lr.set_diag(dvec_rc(&space_x, &[2.0, 3.0, 4.0]));
            let v_space = MultiVectorMatrixSpace::new(3, Rc::clone(&space_x));
            let mut v = v_space.make_new_multi_vector();
            v.set_vector(0, dvec_rc(&space_x, &[0.5, 0.25, 0.0]));
            v.set_vector(1, dvec_rc(&space_x, &[0.0, 0.5, 0.125]));
            v.set_vector(2, dvec_rc(&space_x, &[0.25, 0.0, 0.5]));
            lr.set_v(Rc::new(v));
            let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);
            let (j_c, j_d) = empty_jacobians(3);

            let (inner, stats) = if packed {
                CountingInner::with_packed_path()
            } else {
                CountingInner::new()
            };
            let mut solver = LowRankAugSystemSolver::new(Box::new(inner));
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
            let rhs_x = dvec(&space_x, &[1.0, -2.0, 3.5]);
            let rhs_zero = dvec(&space_zero, &[]);
            let rhs = AugSysRhs {
                rhs_x: &rhs_x,
                rhs_s: &rhs_zero,
                rhs_c: &rhs_zero,
                rhs_d: &rhs_zero,
            };
            let (mut sx, mut z1, mut z2, mut z3) = (
                dvec(&space_x, &[0.0, 0.0, 0.0]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
            );
            {
                let mut sol = AugSysSol {
                    sol_x: &mut sx,
                    sol_s: &mut z1,
                    sol_c: &mut z2,
                    sol_d: &mut z3,
                };
                assert_eq!(
                    solver.solve(&coeffs, &rhs, &mut sol, false, 0),
                    ESymSolverStatus::Success
                );
            }
            (sx.expanded_values(), stats)
        }

        let (sol_loop, stats_loop) = run(false);
        let (sol_batch, stats_batch) = run(true);

        assert_eq!(
            stats_loop.batched_calls.get(),
            0,
            "a mock without the packed path must take the per-column arm"
        );
        assert!(
            stats_batch.factor_batched_calls.get() >= 1,
            "the packed mock must actually reach the merged arm — a mock \
             whose `system_dim()` is 0 when cold silently would not, and \
             then this test would be green about code it never ran"
        );
        assert_eq!(
            stats_batch.factorizations.get(),
            1,
            "batching must not cost an extra factorization"
        );
        assert_eq!(
            sol_loop, sol_batch,
            "batched and per-column arms must agree bit-for-bit"
        );

        // All three V columns ride along with the factorization in a
        // single backend call. Before the merge this read
        // `factor_batched_calls == 0`, `batched_cols == 2` — column 0
        // through the single-RHS `solve`, then a second call carrying
        // the other two. That second call streams the whole factor
        // again, which is the cost this removes.
        assert_eq!(
            stats_batch.factor_batched_calls.get(),
            1,
            "the cold SMW block must be ONE factorizing multi-RHS call"
        );
        assert_eq!(
            stats_batch.factor_batched_cols.get(),
            3,
            "all 3 V columns must ride along with the factorization"
        );
        assert_eq!(
            stats_batch.batched_calls.get(),
            0,
            "no separate back-solve pass may remain over the same factor"
        );
        // One `resolve` remains, and must: it is the actual RHS being
        // solved against the same factor after the V columns have built
        // the SMW correction. Upstream pays it too. What the merge
        // removes is the second pass over the factor that used to carry
        // the V columns.
        assert_eq!(
            stats_batch.backsolves.get(),
            1,
            "only the main RHS solve may remain over the cached factor"
        );
    }

    #[test]
    fn resolve_reuses_the_factor_instead_of_refactorizing() {
        // `PdFullSpaceSolver`'s refinement loop and its same-matrix fast
        // path both come in through `resolve`. Against an unchanged
        // matrix that must be a pure back-substitution.
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let lr_space = LowRankUpdateSymMatrixSpace::new(1, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(dvec_rc(&space_x, &[2.0]));
        let lr_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr);
        let (j_c, j_d) = empty_jacobians(1);
        let delta_x = 0.001;

        let (inner, stats) = CountingInner::new();
        let mut solver = LowRankAugSystemSolver::new(Box::new(inner));

        let coeffs = AugSysCoeffs {
            w: Some(lr_rc.as_ref() as &dyn SymMatrix),
            w_factor: 1.0,
            d_x: None,
            delta_x,
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
        let expected = 1.0 / (2.0 + delta_x);

        for round in 0..4 {
            let (mut sx, mut z1, mut z2, mut z3) = (
                dvec(&space_x, &[0.0]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
            );
            // Scoped so the mutable borrow of `sx` ends before it is read,
            // matching the other tests here.
            {
                let mut sol = AugSysSol {
                    sol_x: &mut sx,
                    sol_s: &mut z1,
                    sol_c: &mut z2,
                    sol_d: &mut z3,
                };
                let status = if round == 0 {
                    solver.solve(&coeffs, &rhs, &mut sol, false, 0)
                } else {
                    solver.resolve(&coeffs, &rhs, &mut sol)
                };
                assert_eq!(status, ESymSolverStatus::Success);
            }
            assert!(
                (sx.expanded_values()[0] - expected).abs() < 1e-12,
                "round {round} gave {:?}",
                sx.expanded_values()
            );
        }
        assert_eq!(
            stats.factorizations.get(),
            1,
            "three refinement re-solves must not re-factorize"
        );
    }

    #[test]
    fn rebuild_with_empty_history_still_factorizes() {
        // Regression guard for the sharp edge in this optimization.
        //
        // `update_factorization` performs *zero* inner solves when the
        // L-BFGS history is empty (`get_v()` and `get_u()` both `None`).
        // That happens on the first iteration and again whenever
        // `limited_memory_max_skipping` clears the history mid-solve
        // (gh#686, the `Wr` info string) — so it is reachable with the
        // inner solver warm, holding the *previous* iterate's factor.
        //
        // If `inner_has_factor` were set as a consequence of the rebuild
        // rather than of an actual inner solve, the diagonal solve would
        // back-substitute against that stale factor: wrong direction, no
        // error, and `StdAugSystemSolver::have_factor` would not catch it
        // because it is not cold. `CountingInner::resolve` reproduces
        // exactly that by answering from its cached diagonal, so this
        // asserts on the value as well as the count.
        let space_x = DenseVectorSpace::new(1);
        let space_zero = DenseVectorSpace::new(0);
        let space_lr = LowRankUpdateSymMatrixSpace::new(1, None, false);

        // Round 1: history present, σ = 2.
        let mut lr1 = space_lr.make_new_low_rank();
        lr1.set_diag(dvec_rc(&space_x, &[2.0]));
        let v_space = MultiVectorMatrixSpace::new(1, Rc::clone(&space_x));
        let mut v = v_space.make_new_multi_vector();
        v.set_vector(0, dvec_rc(&space_x, &[0.5]));
        lr1.set_v(Rc::new(v));
        let lr1_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr1);

        // Round 2: history cleared, σ moved to 7. No V, no U.
        let mut lr2 = LowRankUpdateSymMatrixSpace::new(1, None, false).make_new_low_rank();
        lr2.set_diag(dvec_rc(&space_x, &[7.0]));
        let lr2_rc: Rc<LowRankUpdateSymMatrix> = Rc::new(lr2);

        let (j_c, j_d) = empty_jacobians(1);

        let (inner, stats) = CountingInner::new();
        let mut solver = LowRankAugSystemSolver::new(Box::new(inner));

        let rhs_x = dvec(&space_x, &[1.0]);
        let rhs_zero = dvec(&space_zero, &[]);
        let rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_zero,
            rhs_c: &rhs_zero,
            rhs_d: &rhs_zero,
        };

        let run = |lr: &Rc<LowRankUpdateSymMatrix>, solver: &mut LowRankAugSystemSolver| {
            let coeffs = AugSysCoeffs {
                w: Some(lr.as_ref() as &dyn SymMatrix),
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
            let (mut sx, mut z1, mut z2, mut z3) = (
                dvec(&space_x, &[0.0]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
                dvec(&space_zero, &[]),
            );
            {
                let mut sol = AugSysSol {
                    sol_x: &mut sx,
                    sol_s: &mut z1,
                    sol_c: &mut z2,
                    sol_d: &mut z3,
                };
                assert_eq!(
                    solver.solve(&coeffs, &rhs, &mut sol, false, 0),
                    ESymSolverStatus::Success
                );
            }
            sx.expanded_values()[0]
        };

        run(&lr1_rc, &mut solver);
        let after_first = stats.factorizations.get();

        // Preconditions, asserted rather than assumed. Without these the
        // test still passes if the fixtures drift out from under it, but
        // stops testing the empty-history path -- it would be checking an
        // ordinary rebuild and reporting a pass. (The same vacuity trap
        // feral#179 hit building its full-budget refinement oracle: nothing
        // merely ill-conditioned reaches the budget, so an oracle that is
        // not pinned proves nothing.)
        //
        // 1. Round 1 must have left the hazard live: a factor cached and
        //    `inner_has_factor` set. That flag is private, so assert its
        //    observable consequence -- an identical re-solve reuses it.
        let repeat = run(&lr1_rc, &mut solver);
        assert_eq!(
            stats.factorizations.get(),
            after_first,
            "round 1 must leave a reusable factor, else round 2 has no \
             stale factor to wrongly reuse and this test is vacuous"
        );
        // 1/(2 + 0.5^2): the inner solve answers 1/2 from the cached
        // Wdiag factor and the SMW correction for V then applies on top.
        // Round 2 has no V, which is why its expected value below is the
        // bare 1/7 and why a stale factor there shows up as 1/2.
        assert!(
            (repeat - 4.0 / 9.0).abs() < 1e-12,
            "round 1 re-solve should answer 4/9 from the cached factor, \
             got {repeat}"
        );

        // 2. Round 2's matrix must genuinely carry no history, which is what
        //    makes `update_factorization` perform zero inner solves.
        assert!(
            lr2_rc.get_v().is_none() && lr2_rc.get_u().is_none(),
            "round 2 fixture must have empty L-BFGS history"
        );

        let got = run(&lr2_rc, &mut solver);

        assert_eq!(
            stats.factorizations.get(),
            after_first + 1,
            "a rebuild that performs no inner solve of its own must leave \
             the diagonal solve to factorize the new Wdiag"
        );
        // No V and no U, so the SMW correction is empty and the answer is
        // just the diagonal solve against the *new* σ. Back-solving
        // against round 1's factor would give 1/2, not 1/7.
        assert!(
            (got - 1.0 / 7.0).abs() < 1e-12,
            "expected the new Wdiag (1/7), got {got} — stale factor reused"
        );
    }
}
