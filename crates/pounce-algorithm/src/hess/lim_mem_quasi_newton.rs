//! Limited-memory quasi-Newton (L-BFGS / SR1) — port of
//! `Algorithm/IpLimMemQuasiNewtonUpdater.{hpp,cpp}`. **Phase 8.**
//!
//! Update strategy is selected by the `limited_memory_update_type`
//! option (`bfgs` or `sr1`) per `MAIN_LOOP.md`.
//!
//! Phase 8 publishes the limited-memory Hessian as `data.w` via the
//! **low-rank** assembler, for every problem size. At each
//! `update_hessian` call we walk the curvature-pair history (oldest to
//! newest) applying the rank-2 BFGS / rank-1 SR1 formulas to build the
//! compact factors of `B = σ I + V Vᵀ − U Uᵀ`, then publish a
//! [`pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrix`]
//! as `data.w`. No dense `n×n` buffer is ever formed: the walk is
//! `O(n · m)` per pair and `O(n · m²)` total (with `m = max_history`),
//! and storage is `O(n · m)`, so the limited-memory path scales to
//! arbitrarily large `n`. [`crate::kkt::low_rank_aug_system_solver`]
//! applies the Hessian's inverse action via the Sherman-Morrison-Woodbury
//! identity, factorizing only the diagonal `B0`. This removes the
//! `eval_h` requirement (the user no longer needs to declare a Hessian
//! sparsity pattern) and the former `O(n²)` memory cliff.
//!
//! `LowRankAugSystemSolver` wraps the standard augmented-system solver and
//! forwards the Hessian-free init / equality-multiplier solves (which
//! carry a non-low-rank `W`) straight through, so a single solver
//! instance serves the whole iteration.
//!
//! Update kernels:
//!   - [`initial_hessian_scalar`] (sigma per `LIM_MEM_INIT`)
//!   - [`powell_damping_theta`] (modified-y damping for BFGS)
//!   - [`bfgs_curvature_pair_ok`] (skip-criterion for L-BFGS)
//!   - [`sr1_denominator_ok`] (skip-criterion for SR1)

use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrixSpace;
use pounce_linalg::multi_vector_matrix::{MultiVectorMatrix, MultiVectorMatrixSpace};
use std::rc::Rc;

/// One curvature pair `(s, y)` plus the cached `||s||`, `||y||`, `s·y`
/// scalars the BFGS / SR1 update kernels need on every history walk.
#[derive(Debug, Clone)]
pub struct CurvaturePair {
    pub s: Rc<dyn Vector>,
    pub y: Rc<dyn Vector>,
    pub s_dot_y: Number,
    pub s_norm: Number,
    pub y_norm: Number,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateType {
    Bfgs,
    Sr1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialApprox {
    Identity,
    Scalar1,
    Scalar2,
}

pub struct LimMemQuasiNewtonUpdater {
    pub update_type: UpdateType,
    pub initial_approx: InitialApprox,
    pub max_history: i32,
    /// Powell-damping threshold. Default per upstream
    /// `IpLimMemQuasiNewtonUpdater.cpp:RegisterOptions`:
    /// `limited_memory_init_val_max=1e8` (clamp on initial sigma);
    /// the damping coefficient is hard-coded at 0.2 in the BFGS path.
    pub init_val_max: Number,
    pub init_val_min: Number,
    /// Rolling FIFO of curvature pairs, oldest at index 0. Capped at
    /// `max_history`; insertion drops the front.
    pub history: Vec<CurvaturePair>,
    /// `x` from the previous `update_hessian` call. None on the first
    /// iteration.
    pub last_x: Option<Rc<dyn Vector>>,
    /// `∇f(x_prev)` cached for the upstream y-difference formula
    /// (`IpLimMemQuasiNewtonUpdater.cpp:284`).
    pub last_grad_f: Option<Rc<dyn Vector>>,
    /// `J_c(x_prev)` cached for `J_c_prev^T · y_c_curr` in the
    /// y-difference. Stored as the trait object so we can call
    /// `trans_mult_vector` against the *current* multipliers — the
    /// upstream formula evaluates both Jacobians against `y_c_curr`
    /// (NOT `y_c_prev`).
    pub last_jac_c: Option<Rc<dyn pounce_linalg::matrix::Matrix>>,
    pub last_jac_d: Option<Rc<dyn pounce_linalg::matrix::Matrix>>,
}

impl Default for LimMemQuasiNewtonUpdater {
    fn default() -> Self {
        Self {
            update_type: UpdateType::Bfgs,
            initial_approx: InitialApprox::Scalar2,
            max_history: 6,
            init_val_max: 1e8,
            init_val_min: 1e-8,
            history: Vec::new(),
            last_x: None,
            last_grad_f: None,
            last_jac_c: None,
            last_jac_d: None,
        }
    }
}

impl LimMemQuasiNewtonUpdater {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to absorb a new curvature pair. Returns `true` when the
    /// pair was accepted (and pushed to history), `false` when the
    /// skip-criterion rejected it. The caller owns `s` and `y` as
    /// `Rc<dyn Vector>` so the history can retain them cheaply.
    ///
    /// This matches the per-iteration path in
    /// `IpLimMemQuasiNewtonUpdater.cpp:Update` after the `(x, ∇L)`
    /// difference has been formed: skip-or-keep, then push, then
    /// trim the history to `max_history`.
    pub fn ingest_pair(&mut self, s: Rc<dyn Vector>, y: Rc<dyn Vector>) -> bool {
        let s_dot_y = s.dot(&*y);
        let s_norm = s.nrm2();
        let y_norm = y.nrm2();
        let accept = match self.update_type {
            UpdateType::Bfgs => bfgs_curvature_pair_ok(s_dot_y, s_norm, y_norm),
            UpdateType::Sr1 => {
                // SR1's skip-criterion is `(y - Bs)^T s` not `s^T y`;
                // without `B` available here we use the upstream
                // fallback of `s^T y` magnitude as the gating heuristic
                // (a more accurate test lands once the low-rank matrix
                // is wired in).
                sr1_denominator_ok(s_dot_y, s_norm, y_norm)
            }
        };
        if !accept {
            return false;
        }
        self.history.push(CurvaturePair {
            s,
            y,
            s_dot_y,
            s_norm,
            y_norm,
        });
        // Drop oldest pairs to honor the memory budget.
        while self.history.len() > self.max_history.max(0) as usize {
            self.history.remove(0);
        }
        true
    }
}

impl HessianUpdater for LimMemQuasiNewtonUpdater {
    /// Snapshot the current `(x, ∇_x L)` pair, build `s = x − x_prev`
    /// and `y = ∇L − ∇L_prev`, ingest into history (skip per the
    /// BFGS / SR1 acceptance criterion), then build the low-rank factors
    /// of `B = σ I + V Vᵀ − U Uᵀ` from the rolling history and publish a
    /// [`pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrix`]
    /// as `data.w`. Mirrors `IpLimMemQuasiNewtonUpdater::Update`.
    fn update_hessian(&mut self, data: &IpoptDataHandle, cq: &IpoptCqHandle) -> bool {
        let (curr_x, curr_y_c, curr_y_d) = match data.borrow().curr.as_ref() {
            Some(c) => (c.x.clone(), c.y_c.clone(), c.y_d.clone()),
            None => return true,
        };
        let curr_grad_f = cq.borrow().curr_grad_f();
        let curr_jac_c = cq.borrow().curr_jac_c();
        let curr_jac_d = cq.borrow().curr_jac_d();

        // Upstream y formula (`IpLimMemQuasiNewtonUpdater.cpp:284-308`):
        //   y = (∇f_curr − ∇f_last)
        //     + (J_c_curr^T − J_c_last^T) · y_c_curr
        //     + (J_d_curr^T − J_d_last^T) · y_d_curr
        // i.e. the change in the *NLP* Lagrangian gradient (no bound
        // multipliers) where BOTH Jacobians are dotted against the
        // CURRENT y_c/y_d. Using `curr_grad_lag_x` here would inject
        // the bound-multiplier delta into y, which collapses spuriously
        // when μ drops and corrupts the BFGS update.
        if let (Some(prev_x), Some(prev_grad_f), Some(prev_jac_c), Some(prev_jac_d)) = (
            self.last_x.clone(),
            self.last_grad_f.clone(),
            self.last_jac_c.clone(),
            self.last_jac_d.clone(),
        ) {
            let mut s = curr_x.make_new();
            s.add_two_vectors(1.0, &*curr_x, -1.0, &*prev_x, 0.0);

            let mut y = curr_x.make_new();
            // y = ∇f_curr − ∇f_last
            y.add_two_vectors(1.0, &*curr_grad_f, -1.0, &*prev_grad_f, 0.0);
            // y += J_c_curr^T y_c_curr  −  J_c_last^T y_c_curr
            curr_jac_c.trans_mult_vector(1.0, &*curr_y_c, 1.0, &mut *y);
            prev_jac_c.trans_mult_vector(-1.0, &*curr_y_c, 1.0, &mut *y);
            // y += J_d_curr^T y_d_curr  −  J_d_last^T y_d_curr
            curr_jac_d.trans_mult_vector(1.0, &*curr_y_d, 1.0, &mut *y);
            prev_jac_d.trans_mult_vector(-1.0, &*curr_y_d, 1.0, &mut *y);

            self.ingest_pair(Rc::from(s), Rc::from(y));
        }
        self.last_x = Some(Rc::clone(&curr_x));
        self.last_grad_f = Some(Rc::clone(&curr_grad_f));
        self.last_jac_c = Some(Rc::clone(&curr_jac_c));
        self.last_jac_d = Some(Rc::clone(&curr_jac_d));

        let n_idx = curr_x.dim();
        let nu = n_idx as usize;
        let sigma = match self.update_type {
            UpdateType::Bfgs => self.compute_sigma_bfgs(),
            // SR1 uses the same `LIM_MEM_INIT` sigma source as BFGS for
            // the diagonal `B0`; the rank-1 corrections carry the sign.
            UpdateType::Sr1 => self.compute_sigma_bfgs(),
        };

        // Build the compact factors of  B = σ I + V Vᵀ − U Uᵀ  by walking
        // the curvature history. No dense `n×n` is ever formed: the walk
        // is `O(n · history_len)` per pair. Publishing a
        // `LowRankUpdateSymMatrix` lets `LowRankAugSystemSolver` apply the
        // Hessian via Sherman-Morrison-Woodbury with `O(n · m)` storage
        // for arbitrarily large `n`.
        let (v_cols, u_cols) = self.build_low_rank(sigma, nu);

        // Build `D`, `V`, `U` in `curr_x`'s *native* vector space rather
        // than a fabricated flat `DenseVectorSpace`. For an ordinary
        // (dense-primal) solve this is the same dense space as before; in
        // the feasibility-restoration sub-IPM the primal is a 5-block
        // `CompoundVector` `[orig | n_c | p_c | n_d | p_d]`, and a flat
        // dense `W` cannot be multiplied against those compound iterates —
        // `LowRankUpdateSymMatrix::mult_vector` panics in
        // `element_wise_multiply`/`lr_mult_vector` the moment restoration
        // runs (pounce#102). Cloning `curr_x` keeps `W` type-consistent
        // with the space it operates on.
        let col_space = DenseVectorSpace::new(n_idx);
        let mut diag = curr_x.make_new();
        diag.set(sigma);

        let lr_space = LowRankUpdateSymMatrixSpace::new(n_idx, None, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(Rc::from(diag));
        if let Some(mvm) = build_multi_vector(&col_space, curr_x.as_ref(), &v_cols) {
            lr.set_v(Rc::new(mvm));
        }
        if let Some(mvm) = build_multi_vector(&col_space, curr_x.as_ref(), &u_cols) {
            lr.set_u(Rc::new(mvm));
        }

        data.borrow_mut().w = Some(Rc::new(lr));
        true
    }
}

impl LimMemQuasiNewtonUpdater {
    fn compute_sigma_bfgs(&self) -> Number {
        if self.history.is_empty() {
            return 1.0;
        }
        let last = self.history.last().unwrap();
        let s_dot_s = last.s_norm * last.s_norm;
        let y_dot_y = last.y_norm * last.y_norm;
        initial_hessian_scalar(
            self.initial_approx,
            s_dot_s,
            last.s_dot_y,
            y_dot_y,
            self.init_val_min,
            self.init_val_max,
        )
    }

    /// Walk the curvature-pair history oldest→newest, applying the BFGS
    /// rank-2 / SR1 rank-1 recurrences against the running approximation
    /// `B = σ I + V Vᵀ − U Uᵀ` to grow the dense column lists `V` and
    /// `U`. Returns `(v_cols, u_cols)` in full primal space.
    ///
    /// For BFGS, each accepted pair `(s, y)` appends one positive column
    /// `r/√(sᵀr)` (the `r rᵀ/(sᵀr)` term, `r = θ y + (1−θ) Bs` after
    /// Powell damping) and one negative column `Bs/√(sᵀBs)` (the
    /// `−(Bs)(Bs)ᵀ/(sᵀBs)` term). For SR1 each pair appends a single
    /// column `(y−Bs)/√|denom|` to `V` (denom > 0) or `U` (denom < 0).
    /// This reproduces, column for column, the action of the former
    /// dense rebuild while never materializing an `n×n` buffer.
    fn build_low_rank(&self, sigma: Number, n: usize) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
        let mut v_cols: Vec<Vec<Number>> = Vec::new();
        let mut u_cols: Vec<Vec<Number>> = Vec::new();
        if n == 0 {
            return (v_cols, u_cols);
        }
        for pair in &self.history {
            let s = dense_from_vec(pair.s.as_ref(), n);
            let y = dense_from_vec(pair.y.as_ref(), n);

            // bs = B s = σ s + Σ_v (vᵀs) v − Σ_u (uᵀs) u.
            let mut bs: Vec<Number> = s.iter().map(|&si| sigma * si).collect();
            for v in &v_cols {
                let c: Number = (0..n).map(|i| v[i] * s[i]).sum();
                for i in 0..n {
                    bs[i] += c * v[i];
                }
            }
            for u in &u_cols {
                let c: Number = (0..n).map(|i| u[i] * s[i]).sum();
                for i in 0..n {
                    bs[i] -= c * u[i];
                }
            }

            match self.update_type {
                UpdateType::Bfgs => {
                    let s_bs: Number = (0..n).map(|i| s[i] * bs[i]).sum();
                    if s_bs <= 0.0 {
                        continue;
                    }
                    let sy = pair.s_dot_y;
                    let theta = powell_damping_theta(sy, s_bs);
                    let sr = theta * sy + (1.0 - theta) * s_bs;
                    if sr <= 0.0 {
                        continue;
                    }
                    let r_scale = 1.0 / sr.sqrt();
                    let bs_scale = 1.0 / s_bs.sqrt();
                    // r rᵀ / sr  →  positive column r/√sr.
                    v_cols.push(
                        (0..n)
                            .map(|i| (theta * y[i] + (1.0 - theta) * bs[i]) * r_scale)
                            .collect(),
                    );
                    // −(Bs)(Bs)ᵀ / s_bs  →  negative column Bs/√s_bs.
                    u_cols.push(bs.iter().map(|&bi| bi * bs_scale).collect());
                }
                UpdateType::Sr1 => {
                    let yms: Vec<Number> = (0..n).map(|i| y[i] - bs[i]).collect();
                    let denom: Number = (0..n).map(|i| yms[i] * s[i]).sum();
                    let yms_norm: Number = yms.iter().map(|&w| w * w).sum::<Number>().sqrt();
                    if !sr1_denominator_ok(denom, pair.s_norm, yms_norm) {
                        continue;
                    }
                    let scale = 1.0 / denom.abs().sqrt();
                    let col: Vec<Number> = yms.iter().map(|&w| w * scale).collect();
                    if denom > 0.0 {
                        v_cols.push(col);
                    } else {
                        u_cols.push(col);
                    }
                }
            }
        }
        (v_cols, u_cols)
    }
}

/// Pack flat column data into a [`MultiVectorMatrix`] whose columns are
/// allocated in `template`'s native vector space (so the resulting
/// low-rank `W` is type-consistent with the primal iterates — dense for
/// an ordinary solve, a 5-block resto `CompoundVector` under restoration;
/// see pounce#102). Returns `None` when there are no columns, so the
/// caller leaves the corresponding V/U slot unset.
fn build_multi_vector(
    col_space: &Rc<DenseVectorSpace>,
    template: &dyn Vector,
    cols: &[Vec<Number>],
) -> Option<MultiVectorMatrix> {
    if cols.is_empty() {
        return None;
    }
    let space = MultiVectorMatrixSpace::new(cols.len() as Index, Rc::clone(col_space));
    let mut mvm = space.make_new_multi_vector();
    for (k, col) in cols.iter().enumerate() {
        let mut cv = template.make_new();
        set_expanded(cv.as_mut(), col);
        mvm.set_vector(k as Index, Rc::from(cv));
    }
    Some(mvm)
}

/// Flatten a primal vector to its dense expanded values, handling both a
/// plain [`DenseVector`] and a (possibly nested) restoration
/// [`CompoundVector`].
fn expanded_of(v: &dyn Vector) -> Vec<Number> {
    if let Some(dv) = v.as_any().downcast_ref::<DenseVector>() {
        return dv.expanded_values();
    }
    if let Some(cv) = v.as_any().downcast_ref::<CompoundVector>() {
        let mut out = Vec::with_capacity(cv.dim() as usize);
        for i in 0..cv.n_comps() {
            out.extend(expanded_of(cv.comp(i)));
        }
        return out;
    }
    panic!("LimMemQuasiNewtonUpdater: unsupported primal vector type for expansion");
}

/// Inverse of [`expanded_of`]: scatter a flat slice back into a primal
/// vector of the same structure (dense or compound).
fn set_expanded(dst: &mut dyn Vector, flat: &[Number]) {
    if let Some(dv) = dst.as_any_mut().downcast_mut::<DenseVector>() {
        dv.set_values(flat);
        return;
    }
    if let Some(cv) = dst.as_any_mut().downcast_mut::<CompoundVector>() {
        let n = cv.n_comps();
        let dims: Vec<usize> = (0..n).map(|i| cv.comp(i).dim() as usize).collect();
        let mut off = 0usize;
        for (i, &d) in dims.iter().enumerate() {
            set_expanded(cv.comp_mut(i as Index), &flat[off..off + d]);
            off += d;
        }
        return;
    }
    panic!("LimMemQuasiNewtonUpdater: unsupported primal vector type for set_expanded");
}

fn dense_from_vec(v: &dyn Vector, n: usize) -> Vec<Number> {
    let ev = expanded_of(v);
    debug_assert_eq!(ev.len(), n);
    ev
}

/// Initial Hessian scalar used as the diagonal of `B_0` before the
/// rank-2 updates are applied. Mirrors upstream's three options
/// (`limited_memory_initialization` in
/// `IpLimMemQuasiNewtonUpdater.cpp`):
///
/// * `Identity` → `1.0`
/// * `Scalar1` → `(s^T y) / (s^T s)`
/// * `Scalar2` → `(y^T y) / (s^T y)`
///
/// Result is clamped to `[min_val, max_val]` per upstream's
/// `limited_memory_init_val_{min,max}` defaults.
pub fn initial_hessian_scalar(
    init: InitialApprox,
    s_dot_s: Number,
    s_dot_y: Number,
    y_dot_y: Number,
    min_val: Number,
    max_val: Number,
) -> Number {
    let raw = match init {
        InitialApprox::Identity => 1.0,
        InitialApprox::Scalar1 => {
            if s_dot_s > 0.0 {
                s_dot_y / s_dot_s
            } else {
                1.0
            }
        }
        InitialApprox::Scalar2 => {
            if s_dot_y > 0.0 {
                y_dot_y / s_dot_y
            } else {
                1.0
            }
        }
    };
    raw.clamp(min_val, max_val)
}

/// Powell damping coefficient `theta` for the modified-y BFGS update.
/// When the curvature pair `(s, y)` violates `s^T y >= 0.2 * s^T B s`,
/// we replace `y` by `y_bar = theta * y + (1 - theta) * B s` so that
/// the resulting update is positive-definite.
///
/// ```text
///   if s^T y >= 0.2 * s^T B s:  theta = 1
///   else:                        theta = (0.8 * s^T B s) / (s^T B s - s^T y)
/// ```
///
/// Mirrors upstream's `IpLimMemQuasiNewtonUpdater.cpp:PowellDamping`.
pub fn powell_damping_theta(s_dot_y: Number, s_dot_b_s: Number) -> Number {
    if s_dot_y >= 0.2 * s_dot_b_s {
        1.0
    } else {
        let denom = s_dot_b_s - s_dot_y;
        if denom > 0.0 {
            0.8 * s_dot_b_s / denom
        } else {
            1.0
        }
    }
}

/// L-BFGS curvature-pair acceptance: include `(s, y)` in history iff
/// `s^T y > eps * ||s|| ||y||`. Mirrors upstream's skip-criterion
/// (`IpLimMemQuasiNewtonUpdater.cpp` ~line 750: `eps = 1e-8`).
pub fn bfgs_curvature_pair_ok(s_dot_y: Number, s_norm: Number, y_norm: Number) -> bool {
    let eps = 1e-8_f64;
    s_dot_y > eps * s_norm * y_norm
}

/// SR1 acceptance: the SR1 update divides by `(y - Bs)^T s`, so we
/// need `|(y - Bs)^T s| > eps * ||s|| ||y - Bs||`. Mirrors upstream's
/// `IpLimMemQuasiNewtonUpdater.cpp` SR1 skip-criterion.
pub fn sr1_denominator_ok(yms_dot_s: Number, s_norm: Number, yms_norm: Number) -> bool {
    let eps = 1e-8_f64;
    yms_dot_s.abs() > eps * s_norm * yms_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_init_returns_one() {
        assert_eq!(
            initial_hessian_scalar(InitialApprox::Identity, 1.0, 1.0, 1.0, 1e-8, 1e8),
            1.0
        );
    }

    #[test]
    fn scalar1_init_is_sy_over_ss() {
        // s_dot_s=4, s_dot_y=2 → 2/4 = 0.5.
        let v = initial_hessian_scalar(InitialApprox::Scalar1, 4.0, 2.0, 0.0, 1e-8, 1e8);
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn scalar2_init_is_yy_over_sy() {
        // y_dot_y=8, s_dot_y=2 → 4.
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 2.0, 8.0, 1e-8, 1e8);
        assert!((v - 4.0).abs() < 1e-15);
    }

    #[test]
    fn init_clamped_to_max() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e-20, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e8);
    }

    #[test]
    fn init_clamped_to_min() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e20, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e-8);
    }

    #[test]
    fn powell_no_damping_when_curvature_ok() {
        // s^T y = 1, s^T B s = 1; 1 >= 0.2 * 1 → theta = 1.
        assert_eq!(powell_damping_theta(1.0, 1.0), 1.0);
    }

    #[test]
    fn powell_damps_when_curvature_violated() {
        // s^T y = 0.1, s^T B s = 1; 0.1 < 0.2 → theta = 0.8/(1-0.1) = 8/9.
        let theta = powell_damping_theta(0.1, 1.0);
        assert!((theta - 8.0 / 9.0).abs() < 1e-15);
    }

    #[test]
    fn bfgs_skip_criterion() {
        // s_dot_y = 1, ||s|| = 1, ||y|| = 1 → 1 > 1e-8: ok.
        assert!(bfgs_curvature_pair_ok(1.0, 1.0, 1.0));
        // s_dot_y = 1e-10, ||s|| = 1, ||y|| = 1 → 1e-10 < 1e-8: skip.
        assert!(!bfgs_curvature_pair_ok(1e-10, 1.0, 1.0));
    }

    #[test]
    fn sr1_skip_criterion_uses_absolute_value() {
        // Negative numerator is fine for SR1 (rank-1 update can have either sign).
        assert!(sr1_denominator_ok(-1.0, 1.0, 1.0));
        assert!(!sr1_denominator_ok(1e-10, 1.0, 1.0));
    }

    fn rcv(values: &[Number]) -> Rc<dyn Vector> {
        let mut v = pounce_linalg::dense_vector::DenseVectorSpace::new(values.len() as i32)
            .make_new_dense();
        v.set(0.0);
        v.values_mut().copy_from_slice(values);
        Rc::new(v)
    }

    #[test]
    fn ingest_pair_accepts_well_curved_pair() {
        let mut updater = LimMemQuasiNewtonUpdater::new();
        // s = (1, 0), y = (1, 0); s·y = 1 > 1e-8.
        let accepted = updater.ingest_pair(rcv(&[1.0, 0.0]), rcv(&[1.0, 0.0]));
        assert!(accepted);
        assert_eq!(updater.history.len(), 1);
        let pair = &updater.history[0];
        assert!((pair.s_dot_y - 1.0).abs() < 1e-15);
        assert!((pair.s_norm - 1.0).abs() < 1e-15);
        assert!((pair.y_norm - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ingest_pair_skips_zero_curvature() {
        let mut updater = LimMemQuasiNewtonUpdater::new();
        // s · y = 0 ⇒ skip per BFGS criterion (eps · ||s|| · ||y||).
        let accepted = updater.ingest_pair(rcv(&[1.0]), rcv(&[0.0]));
        assert!(!accepted);
        assert!(updater.history.is_empty());
    }

    #[test]
    fn history_caps_at_max_history() {
        let mut updater = LimMemQuasiNewtonUpdater {
            max_history: 2,
            ..LimMemQuasiNewtonUpdater::default()
        };
        for _ in 0..5 {
            updater.ingest_pair(rcv(&[1.0]), rcv(&[1.0]));
        }
        assert_eq!(updater.history.len(), 2);
    }

    #[test]
    fn sr1_path_routes_through_sr1_skip() {
        let mut updater = LimMemQuasiNewtonUpdater {
            update_type: UpdateType::Sr1,
            ..LimMemQuasiNewtonUpdater::default()
        };
        // SR1's heuristic accepts negative s·y (rank-1 sign-indefinite).
        assert!(updater.ingest_pair(rcv(&[1.0]), rcv(&[-1.0])));
    }

    fn pair(s: &[Number], y: &[Number]) -> CurvaturePair {
        let s_rc = rcv(s);
        let y_rc = rcv(y);
        let s_dot_y = s_rc.dot(&*y_rc);
        let s_norm = s_rc.nrm2();
        let y_norm = y_rc.nrm2();
        CurvaturePair {
            s: s_rc,
            y: y_rc,
            s_dot_y,
            s_norm,
            y_norm,
        }
    }

    /// Reconstruct the dense `B = σ I + V Vᵀ − U Uᵀ` from the low-rank
    /// factors so we can check the Hessian *action* the SMW solver sees.
    fn reconstruct_b(n: usize, sigma: Number, v: &[Vec<Number>], u: &[Vec<Number>]) -> Vec<Number> {
        let mut b = vec![0.0_f64; n * n];
        for i in 0..n {
            b[i * n + i] = sigma;
        }
        for col in v {
            for i in 0..n {
                for j in 0..n {
                    b[i * n + j] += col[i] * col[j];
                }
            }
        }
        for col in u {
            for i in 0..n {
                for j in 0..n {
                    b[i * n + j] -= col[i] * col[j];
                }
            }
        }
        b
    }

    fn mat_vec(b: &[Number], n: usize, x: &[Number]) -> Vec<Number> {
        (0..n)
            .map(|i| (0..n).map(|j| b[i * n + j] * x[j]).sum())
            .collect()
    }

    #[test]
    fn bfgs_low_rank_recovers_hessian_action() {
        // For a strictly-convex quadratic f(x) = ½ xᵀ A x with A SPD,
        // a single BFGS update from B₀ = I along a curvature pair
        // (s, y = A s) reproduces A on the s-direction:  B₁ s = y = A s.
        // Use A = diag(2, 5), s = (1, 1), so y = (2, 5).
        let mut up = LimMemQuasiNewtonUpdater::new();
        up.history.push(pair(&[1.0, 1.0], &[2.0, 5.0]));
        let (v, u) = up.build_low_rank(1.0, 2);
        let b = reconstruct_b(2, 1.0, &v, &u);
        let bs = mat_vec(&b, 2, &[1.0, 1.0]);
        assert!((bs[0] - 2.0).abs() < 1e-12, "Bs[0]={}", bs[0]);
        assert!((bs[1] - 5.0).abs() < 1e-12, "Bs[1]={}", bs[1]);
    }

    #[test]
    fn bfgs_low_rank_keeps_symmetry() {
        let mut up = LimMemQuasiNewtonUpdater::new();
        up.history.push(pair(&[1.0, 0.5], &[2.0, 1.0]));
        up.history.push(pair(&[0.7, 1.2], &[1.0, 2.5]));
        let (v, u) = up.build_low_rank(3.0, 2);
        let b = reconstruct_b(2, 3.0, &v, &u);
        // VVᵀ and UUᵀ are symmetric by construction, so B must be too.
        assert!((b[1] - b[2]).abs() < 1e-12);
    }

    #[test]
    fn sr1_low_rank_recovers_hessian_action() {
        // SR1 update with B₀ = I, s = (1, 1), y = (2, 5):
        // y - B s = (1, 4); denom = (1, 4)·(1, 1) = 5 > 0 → one V column.
        // ΔB = (1, 4)(1, 4)ᵀ / 5; B₁ s = (2.0, 5.0) = y. ✓
        let mut up = LimMemQuasiNewtonUpdater {
            update_type: UpdateType::Sr1,
            ..LimMemQuasiNewtonUpdater::default()
        };
        up.history.push(pair(&[1.0, 1.0], &[2.0, 5.0]));
        let (v, u) = up.build_low_rank(1.0, 2);
        assert_eq!(v.len(), 1, "positive denom routes to V");
        assert!(u.is_empty());
        let b = reconstruct_b(2, 1.0, &v, &u);
        let bs = mat_vec(&b, 2, &[1.0, 1.0]);
        assert!((bs[0] - 2.0).abs() < 1e-12);
        assert!((bs[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn empty_history_yields_no_columns() {
        let up = LimMemQuasiNewtonUpdater::new();
        let (v, u) = up.build_low_rank(1.0, 4);
        assert!(v.is_empty() && u.is_empty());
    }
}
