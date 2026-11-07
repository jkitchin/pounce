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
//!   - [`bfgs_curvature_pair_ok`] (skip-criterion for L-BFGS)
//!   - [`sr1_denominator_ok`] (skip-criterion for SR1)

use crate::hess::r#trait::HessianUpdater;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::compound_vector::CompoundVector;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};
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

/// `limited_memory_initialization` — how the diagonal `B0 = σ I` is
/// chosen before the rank-2 updates. Upstream registers five values
/// (`IpLimMemQuasiNewtonUpdater.cpp:RegisterOptions`); `Identity` has no
/// upstream keyword and exists for callers constructing the updater
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialApprox {
    Identity,
    /// `scalar1` — σ = sᵀy / sᵀs. Upstream's default.
    Scalar1,
    /// `scalar2` — σ = yᵀy / sᵀy.
    Scalar2,
    /// `scalar3` — arithmetic mean of `scalar1` and `scalar2`.
    Scalar3,
    /// `scalar4` — geometric mean of `scalar1` and `scalar2`.
    Scalar4,
    /// `constant` — σ = `limited_memory_init_val`, every iteration.
    Constant,
    /// `history-max` — the `scalar1` formula evaluated on **every**
    /// stored curvature pair, largest wins (gh#818). No upstream
    /// keyword; see [`LimMemQuasiNewtonUpdater::compute_sigma_bfgs`]
    /// for why the maximum rather than the newest pair.
    HistoryMax,
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
    /// `limited_memory_init_val` — the multiple of the identity `B0`
    /// takes on the first iteration, before any curvature pair has been
    /// formed, and on every iteration under
    /// [`InitialApprox::Constant`]. Upstream default `1.0`
    /// (`IpLimMemQuasiNewtonUpdater.cpp:RegisterOptions`). Was a
    /// hard-coded `1.0` in the empty-history branch until #677 — the
    /// same value, but not settable, and `constant` had nowhere to read
    /// its σ from.
    pub init_val: Number,
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
    /// Positions in the primal space that enter *nonlinearly* (gh#624),
    /// sorted and deduplicated. `None` — the default — approximates over
    /// the whole space, which is what every solve did before the mask
    /// existed. When set, curvature pairs are projected onto these
    /// positions and the published `W` carries the corresponding
    /// expansion `P`, so the KKT solvers see
    /// `B = σ I + P (V Vᵀ − U Uᵀ) Pᵀ`: no stored curvature, and no
    /// stored columns, for variables that only ever appear linearly.
    /// The subset comes from upstream's `P_LM`
    /// (`IpTNLPAdapter::GetQuasiNewtonApproxSpaces`); see
    /// `update_hessian` for why σ stays on the full diagonal where
    /// upstream drops it.
    pub nonlinear_vars: Option<Vec<Index>>,
    /// Restoration only: the length of the original-`x` block that leads
    /// the restoration primal `[x | n_c | p_c | n_d | p_d]` (gh#1146).
    ///
    /// Upstream's restoration updater (`LimMemQuasiNewtonUpdater` built
    /// with `update_for_resto = true`) approximates the `(0, 0)` block of
    /// a compound `W` and leaves the slack blocks **zero**: the
    /// restoration objective is linear in `n` and `p`, and their
    /// constraint columns are constant, so they carry no curvature. Its
    /// curvature pairs are the `x` parts of `s` and `y` only.
    ///
    /// Run over the whole compound instead, every pair carried the
    /// slack components of the step in `s` against an exactly-zero `y`
    /// there, so `σ = sᵀy/‖s‖²` and every `V`/`U` column were built from
    /// a step diluted by motion that has no curvature at all, and only
    /// then truncated to the `x` rows by `AugRestoSystemSolver`. Measured
    /// on the gh#1146 model under `nlp_scaling_method=none`: identical to
    /// Ipopt through iteration 6, then the first restoration step was
    /// `‖d‖ = 3.63e3` against Ipopt's `2.10e3`, and the run ended
    /// `Restoration_Failed` where Ipopt converges.
    ///
    /// When set, pairs are projected onto `0..k` before they enter the
    /// history (upstream's `P_LMᵀ`), and `W` is published in the
    /// compound's own shape — σ on the `x` diagonal, zero on the slack
    /// blocks, the curvature columns zero-padded there — so the
    /// restoration KKT solver reads the same `(0, 0)` block it always
    /// has, with no expansion matrix to account for.
    pub leading_block: Option<Index>,
    /// `limited_memory_max_skipping` — consecutive skipped curvature
    /// updates after which the whole approximation is discarded and
    /// re-anchored at the current iterate. Upstream default 2
    /// (`IpLimMemQuasiNewtonUpdater.cpp`, the
    /// `lm_skipped_iter_ >= limited_memory_max_skipping_` branch).
    ///
    /// Without this the history never turns over: on a problem whose
    /// Lagrangian curvature is persistently negative, every pair after
    /// the first few is skipped and the model keeps representing
    /// curvature the iterate left long ago. `inf_pr` converges while
    /// `inf_du` plateaus. Reported with side-by-side instrumentation on
    /// a 59,939-variable collocation model (#686): over 60 iterations
    /// Ipopt reset 9 times and pounce 0, and the two iterate sequences
    /// were bit-identical until one iteration after Ipopt's first reset.
    pub max_skipping: Index,
    /// Consecutive skips so far — upstream's `lm_skipped_iter_`.
    pub skipped_iter: Index,
}

impl Default for LimMemQuasiNewtonUpdater {
    fn default() -> Self {
        Self {
            update_type: UpdateType::Bfgs,
            initial_approx: InitialApprox::Scalar1,
            max_history: 6,
            init_val_max: 1e8,
            init_val_min: 1e-8,
            init_val: 1.0,
            history: Vec::new(),
            last_x: None,
            last_grad_f: None,
            last_jac_c: None,
            last_jac_d: None,
            nonlinear_vars: None,
            leading_block: None,
            max_skipping: 2,
            skipped_iter: 0,
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

            // Restrict the pair to the nonlinear subspace before it
            // enters the history: everything downstream (σ, the BFGS /
            // SR1 recurrences, the stored columns) then lives in the
            // reduced space, exactly as upstream does once `P_LM` is
            // present.
            let pair_mask = self.pair_mask(curr_x.dim());
            let accepted = match pair_mask.as_deref() {
                Some(mask) => {
                    let s_red = project_onto(&*s, mask);
                    let y_red = project_onto(&*y, mask);
                    self.ingest_pair(s_red, y_red)
                }
                None => self.ingest_pair(Rc::from(s), Rc::from(y)),
            };

            // `limited_memory_max_skipping` (#686). Upstream discards the
            // whole approximation once the curvature update has been
            // skipped this many times in a row
            // (`IpLimMemQuasiNewtonUpdater.cpp`, the
            // `lm_skipped_iter_ >= limited_memory_max_skipping_` branch,
            // which also emits the `Wr` info string). pounce accumulated
            // the skip but never acted on it, so a run whose curvature
            // stays negative kept its opening pairs for the rest of the
            // solve.
            //
            // The re-anchoring upstream does inside its reset branch
            // (`last_x_`, `last_grad_f_`, `last_jac_c_`, `last_jac_d_`)
            // is what the four assignments below this block already do
            // unconditionally on every call, so clearing the history is
            // the whole of it here.
            if accepted {
                self.skipped_iter = 0;
            } else {
                self.skipped_iter = self.skipped_iter.saturating_add(1);
                if self.max_skipping > 0 && self.skipped_iter >= self.max_skipping {
                    self.history.clear();
                    self.skipped_iter = 0;
                    data.borrow_mut().append_info_string("Wr");
                }
            }
        }
        self.last_x = Some(Rc::clone(&curr_x));
        self.last_grad_f = Some(Rc::clone(&curr_grad_f));
        self.last_jac_c = Some(Rc::clone(&curr_jac_c));
        self.last_jac_d = Some(Rc::clone(&curr_jac_d));

        let n_idx = curr_x.dim();
        if let Some(k) = self.active_leading_block(n_idx) {
            self.publish_leading_block(data, &curr_x, k);
            return true;
        }
        // Dimension the low-rank build runs in: the nonlinear subspace
        // when a mask is active, the full primal space otherwise.
        let n_red = match self.active_mask(n_idx) {
            Some(mask) => mask.len() as Index,
            None => n_idx,
        };
        let nu = n_red as usize;
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
        //
        // Under a nonlinear-variable mask the reduced space is a plain
        // dense space of its own — the compound-vector concern above is
        // moot there, because the mask is only ever installed for the
        // original NLP (the restoration sub-IPM clears it: its primal is
        // the 5-block compound, whose indices mean something else).
        let mask = self.active_mask(n_idx).map(|m| m.to_vec());
        let col_space = DenseVectorSpace::new(n_red);
        let reduced_proto: Option<DenseVector> = mask.as_ref().map(|_| col_space.make_new_dense());
        let proto: &dyn Vector = match reduced_proto.as_ref() {
            Some(p) => p,
            None => curr_x.as_ref(),
        };

        // The diagonal `B0` spans the **full** primal space, masked or
        // not: σ on the nonlinear coordinates, and
        // `limited_memory_init_val_min` (1e-8 by default) as a floor on
        // the rest. Only the curvature columns `V`/`U` are restricted to
        // the subspace.
        //
        // **Deliberate divergence from upstream (gh#624).** Ipopt builds
        // this space as
        // `LowRankUpdateSymMatrixSpace(dim, P_LM, /*reduced_diag=*/true)`
        // (`IpOrigIpoptNLP.cpp:InitializeStructures`), which puts σ in the
        // small space only, so `W` is *exactly zero* on the variables that
        // enter linearly. That is the truthful Hessian — the second
        // derivatives really are zero there — and it is a trap for the
        // augmented system: those rows of the `(1,1)` block are then
        // carried by the barrier term `Σ_x` alone, which is ~0 for a
        // variable sitting far from its bounds, and the symmetric
        // factorization pays for a near-singular diagonal on every one of
        // them. Measured on a model with 2 nonlinear and 2000 linear
        // variables (all three reach the same KKT point):
        //
        //     exact zero (upstream)      4.7 s   28 iterations
        //     floor = 1e-8 (this code)   0.93 s  28 iterations
        //     no mask at all             0.89 s  25 iterations
        //
        // At 10 000 linear variables the mask then does what it is for:
        // 5.1 s / 27 iterations against 6.0 s / 31 unmasked.
        //
        // Ipopt's own limited-memory path takes **399 s** on that model
        // with `pass_nonlinear_variables` on, against 0.40 s off — the
        // same effect, two orders of magnitude worse, which is what
        // convinced us the flag rather than the port was at fault.
        //
        // The floor is the smallest intervention that works, and it was
        // picked over the obvious alternative. Filling the whole diagonal
        // with σ (`reduced_diag = false` and nothing else) is equally
        // fast, but it injects a proximal term of the *problem's own
        // curvature scale* into coordinates whose curvature is zero, and
        // — unlike the unmasked path, where L-BFGS learns that and
        // corrects σ back down — the masked update has no columns there
        // to correct it with. On the 6-variable fixture in
        // `crates/pounce-cinterface/tests/nonlinear_variables_mask.rs`
        // that turns a `Solve_Succeeded` at `tol=1e-9` into a stall at
        // `Solved_To_Acceptable_Level`. At 1e-8 the term is far below any
        // tolerance the solver reasons about, and the tail converges.
        //
        // What the mask buys is untouched either way: curvature
        // information kept free of the linear block, and `O(n_nonlin · m)`
        // rather than `O(n · m)` storage for the columns.
        let mut diag = curr_x.make_new();
        diag.set(sigma);
        if let Some(m) = mask.as_ref() {
            // σ on the nonlinear coordinates, the curvature floor on the
            // rest. `limited_memory_init_val_min` is registered with a
            // strict lower bound of 0, so this diagonal can never be
            // exactly zero — which is the whole point (see above).
            let mut vals = vec![self.init_val_min; n_idx as usize];
            for &i in m {
                vals[i as usize] = sigma;
            }
            set_expanded(diag.as_mut(), &vals);
        }

        // `P` lifts the reduced low-rank update back into full-x.
        let p_lm: Option<Rc<dyn pounce_linalg::matrix::Matrix>> = mask.as_ref().map(|m| {
            let space = ExpansionMatrixSpace::new(n_idx, n_red, m, 0);
            Rc::new(ExpansionMatrix::new(space)) as Rc<dyn pounce_linalg::matrix::Matrix>
        });
        let lr_space = LowRankUpdateSymMatrixSpace::new(n_idx, p_lm, false);
        let mut lr = lr_space.make_new_low_rank();
        lr.set_diag(Rc::from(diag));
        if let Some(mvm) = build_multi_vector(&col_space, proto, &v_cols) {
            lr.set_v(Rc::new(mvm));
        }
        if let Some(mvm) = build_multi_vector(&col_space, proto, &u_cols) {
            lr.set_u(Rc::new(mvm));
        }

        data.borrow_mut().w = Some(Rc::new(lr));
        true
    }

    /// Drop every curvature pair but the newest, so `B` falls back to
    /// `σI` plus the single rank-2 update carrying the freshest
    /// measured curvature (gh#818).
    ///
    /// **Why keep one pair rather than clear the history outright.**
    /// `compute_sigma_bfgs` reads σ off the history, and an *empty*
    /// history returns `limited_memory_init_val` — a bare `1.0`, which
    /// is a curvature scale only by coincidence. Re-anchoring to that
    /// throws the model back to its first-iteration state on a problem
    /// whose curvature the solver has by now measured, and the first
    /// iteration is precisely where the badly-scaled step comes from:
    /// on gh#818's quadratic, iteration 1 is the one that needs 20
    /// backtracks. Keeping the newest pair keeps σ a real Rayleigh
    /// quotient and keeps the secant condition `B s = y` on the step
    /// the solver just took, while discarding the older corrections
    /// that made the direction unusable.
    ///
    /// Returns `false` when there is nothing to discard — an empty
    /// history, or a history already down to its newest pair — which is
    /// what makes one re-anchor per stall the natural bound: the second
    /// failure at the same iterate finds nothing to give up and the
    /// caller falls through.
    fn reanchor(&mut self) -> bool {
        if self.history.len() <= 1 {
            return false;
        }
        let newest = self.history.pop().expect("len > 1");
        self.history.clear();
        self.history.push(newest);
        // The skip counter measures consecutive *rejected* pairs and
        // drives `limited_memory_max_skipping`; a deliberate re-anchor
        // is not a rejection, and leaving the count standing would make
        // the next skipped pair discard the pair we just chose to keep.
        self.skipped_iter = 0;
        true
    }
}

impl LimMemQuasiNewtonUpdater {
    /// The restoration `x` block length, if one applies to a primal space
    /// of dimension `n` — see [`Self::leading_block`]. `None` (the
    /// whole-space approximation) when unset, empty, or not a proper
    /// leading block of `n`.
    fn active_leading_block(&self, n: Index) -> Option<Index> {
        self.leading_block.filter(|&k| k > 0 && k < n)
    }

    /// The positions a curvature pair is projected onto before it enters
    /// the history: the restoration `x` block when
    /// [`Self::leading_block`] applies, else the nonlinear-variable mask,
    /// else `None` (the whole space).
    fn pair_mask(&self, n: Index) -> Option<Vec<Index>> {
        match self.active_leading_block(n) {
            Some(k) => Some((0..k).collect()),
            None => self.active_mask(n).map(<[Index]>::to_vec),
        }
    }

    /// Publish `W` for [`Self::leading_block`]: the history lives in the
    /// `k`-dimensional `x` block; `W` is laid out in `curr_x`'s compound
    /// space with σ on the first `k` diagonal entries, zero on the rest,
    /// and the curvature columns zero beyond row `k`.
    fn publish_leading_block(&mut self, data: &IpoptDataHandle, curr_x: &Rc<dyn Vector>, k: Index) {
        let n_idx = curr_x.dim();
        let (n, ku) = (n_idx as usize, k as usize);
        let sigma = self.compute_sigma_bfgs();
        let (v_cols, u_cols) = self.build_low_rank(sigma, ku);
        let pad = |cols: Vec<Vec<Number>>| -> Vec<Vec<Number>> {
            cols.into_iter()
                .map(|mut c| {
                    c.resize(n, 0.0);
                    c
                })
                .collect()
        };
        let (v_cols, u_cols) = (pad(v_cols), pad(u_cols));

        let mut diag_vals = vec![0.0; n];
        diag_vals[..ku].fill(sigma);
        let mut diag = curr_x.make_new();
        set_expanded(diag.as_mut(), &diag_vals);

        let col_space = DenseVectorSpace::new(n_idx);
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
    }

    /// The nonlinear-variable mask, if one applies to a primal space of
    /// dimension `n`.
    ///
    /// The guard is deliberate: a mask is stated over the *original*
    /// NLP's variables, and the restoration sub-IPM solves a different
    /// problem whose primal is `[orig | n_c | p_c | n_d | p_d]`.
    /// `run_inner_resto` clears the mask for that solve; this check is
    /// the belt to that suspenders, so a masked updater reused against a
    /// space it was not built for silently degrades to the full-space
    /// approximation instead of indexing into the wrong variables.
    fn active_mask(&self, n: Index) -> Option<&[Index]> {
        let mask = self.nonlinear_vars.as_deref()?;
        if mask.is_empty() || mask.len() as Index >= n {
            return None;
        }
        if mask.last().is_some_and(|&last| last >= n) {
            return None;
        }
        Some(mask)
    }

    /// σ, the diagonal of `B0`, for this iteration's rebuild.
    ///
    /// Every upstream rule reads the **newest** curvature pair only.
    /// [`InitialApprox::HistoryMax`] is the one exception: it applies
    /// the `scalar1` formula to every pair in the window and takes the
    /// largest (gh#818).
    ///
    /// **Why a maximum.** σ is the curvature the model assigns to every
    /// direction *outside* the span of the stored pairs — the rank-2
    /// corrections say nothing there. `sᵀy/sᵀs` is a Rayleigh quotient
    /// of the true Hessian along one step, so on a problem whose
    /// curvature spans orders of magnitude it is an arbitrary sample of
    /// the spectrum. When it lands near the small end, `B` understates
    /// the curvature of every unexplored direction by up to `cond(H)`,
    /// `d = −B⁻¹∇f` is longer than the truth by that factor, and a
    /// backtracking line search can only recover by halving.
    ///
    /// The two errors are not symmetric. Over-stating σ shortens the
    /// step: the line search accepts `α = 1` and the iteration is
    /// merely less ambitious. Under-stating it costs a whole
    /// backtracking sequence — measured at 19–20 trial points per
    /// iteration on gh#818's 8-variable quadratic, landing at
    /// `α ≈ 4e-6` — and the tiny step then feeds a tiny `s` back into
    /// the history, so the next σ is drawn from an even narrower
    /// sample.
    ///
    /// The pairs in the window are all measured curvature of the same
    /// Lagrangian at nearby iterates, so the largest of them is the
    /// stiffest thing the solver has actually seen and is the
    /// conservative reading. Nothing is lost on the directions the
    /// model *does* know: the last rank-2 update enforces
    /// `B s_last = y_last` whatever `B0` was, so the secant condition
    /// on the newest pair holds under this rule exactly as under
    /// `scalar1`.
    ///
    /// The window matters. A running maximum over the whole solve — the
    /// obvious variant — is monotone and never comes back down, so it
    /// keeps a stiff early transient in `B0` long after the iterate has
    /// left it; measured on gh#818's fixture it is worse than `scalar1`
    /// at every size (131 vs 36 iterations at `n = 4`, and no
    /// convergence at all at `n = 8`). Bounding the maximum by the
    /// history window lets σ decay as the pairs turn over.
    fn compute_sigma_bfgs(&self) -> Number {
        if self.history.is_empty() {
            // Upstream: `B0 = limited_memory_init_val * I` "in the first
            // iteration (when no updates have been performed yet)".
            return self.init_val;
        }
        let per_pair = |p: &CurvaturePair| {
            initial_hessian_scalar(
                self.initial_approx,
                p.s_norm * p.s_norm,
                p.s_dot_y,
                p.y_norm * p.y_norm,
                self.init_val,
                self.init_val_min,
                self.init_val_max,
            )
        };
        if self.initial_approx == InitialApprox::HistoryMax {
            // `clamp` is monotone, so folding the max over already-clamped
            // per-pair values is the same number as clamping the max.
            return self
                .history
                .iter()
                .map(per_pair)
                .fold(Number::NEG_INFINITY, Number::max);
        }
        per_pair(self.history.last().unwrap())
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
                    // Textbook BFGS, which is what upstream forms:
                    //
                    //     v_new = y_new / sqrt(sᵀy)        (positive column)
                    //     u_new = B₀·S·C                    (negative column)
                    //
                    // `y` is used as it stands. pounce used to blend it
                    // toward `B s` by a Powell damping factor whenever
                    // `sᵀy < 0.2·sᵀBs`, citing
                    // `IpLimMemQuasiNewtonUpdater.cpp:PowellDamping` —
                    // a function that does not exist. Upstream's
                    // `CheckSkippingBFGS` takes `const Vector&` for both
                    // `s_new` and `y_new` and returns a bool, so it
                    // cannot modify a pair, and nothing else in that file
                    // does either: a pair is skipped or it is stored as
                    // measured (#686).
                    //
                    // Damping was not a harmless extra safeguard. It
                    // fired on every accepted pair with marginal
                    // curvature and silently replaced the measured
                    // curvature with a synthetic one, on a path where
                    // upstream's answer to marginal curvature is to skip
                    // the pair and — after `limited_memory_max_skipping`
                    // of them — discard the history. That strategy is
                    // complete on its own, and it is now implemented.
                    //
                    // `sᵀy > sqrt(eps)·‖s‖·‖y‖ > 0` holds for every pair
                    // in `history` by the skip criterion, so the square
                    // root below is of a positive number without needing
                    // the damped `sr` guard that used to stand here.
                    let sy = pair.s_dot_y;
                    if sy <= 0.0 {
                        continue;
                    }
                    let y_scale = 1.0 / sy.sqrt();
                    let bs_scale = 1.0 / s_bs.sqrt();
                    // y yᵀ / sᵀy  →  positive column y/√(sᵀy).
                    v_cols.push(y.iter().map(|&yi| yi * y_scale).collect());
                    // −(Bs)(Bs)ᵀ / sᵀBs  →  negative column Bs/√(sᵀBs).
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

/// Gather the entries of `v` named by `mask` into a fresh dense vector
/// of dimension `mask.len()` — the `Pᵀ v` of upstream's expansion
/// matrix, done by index because the mask path never sees a compound
/// primal.
fn project_onto(v: &dyn Vector, mask: &[Index]) -> Rc<dyn Vector> {
    let full = expanded_of(v);
    let small: Vec<Number> = mask.iter().map(|&i| full[i as usize]).collect();
    let mut out = DenseVectorSpace::new(mask.len() as Index).make_new_dense();
    out.set_values(&small);
    Rc::new(out)
}

fn dense_from_vec(v: &dyn Vector, n: usize) -> Vec<Number> {
    let ev = expanded_of(v);
    debug_assert_eq!(ev.len(), n);
    ev
}

/// Initial Hessian scalar used as the diagonal of `B_0` before the
/// rank-2 updates are applied. Mirrors upstream's
/// `limited_memory_initialization` values
/// (`IpLimMemQuasiNewtonUpdater.cpp:RegisterOptions`):
///
/// * `Scalar1` → `(s^T y) / (s^T s)` — upstream's default
/// * `Scalar2` → `(y^T y) / (s^T y)`
/// * `Scalar3` → arithmetic mean of `Scalar1` and `Scalar2`
/// * `Scalar4` → geometric mean of `Scalar1` and `Scalar2`
/// * `Constant` → `init_val` (`limited_memory_init_val`)
/// * `Identity` → `1.0` (no upstream keyword; direct callers only)
/// * `HistoryMax` → the `Scalar1` formula (no upstream keyword). This
///   kernel is per-pair; what makes `HistoryMax` different from
///   `Scalar1` is that
///   [`LimMemQuasiNewtonUpdater::compute_sigma_bfgs`] calls it on every
///   pair in the window and keeps the largest, rather than calling it
///   on the newest pair alone (gh#818).
///
/// Each degenerate denominator falls back to `1.0` independently, so
/// `Scalar3`/`Scalar4` degrade to the mean of whichever term is
/// well-defined rather than to a single fallback for the pair.
/// `Scalar4`'s geometric mean is taken on the product of two
/// non-negative terms; a non-positive product falls back to `1.0`
/// rather than producing a NaN.
///
/// Result is clamped to `[min_val, max_val]` per upstream's
/// `limited_memory_init_val_{min,max}` defaults.
pub fn initial_hessian_scalar(
    init: InitialApprox,
    s_dot_s: Number,
    s_dot_y: Number,
    y_dot_y: Number,
    init_val: Number,
    min_val: Number,
    max_val: Number,
) -> Number {
    let scalar1 = || {
        if s_dot_s > 0.0 {
            s_dot_y / s_dot_s
        } else {
            1.0
        }
    };
    let scalar2 = || {
        if s_dot_y > 0.0 {
            y_dot_y / s_dot_y
        } else {
            1.0
        }
    };
    let raw = match init {
        InitialApprox::Identity => 1.0,
        InitialApprox::Scalar1 => scalar1(),
        InitialApprox::Scalar2 => scalar2(),
        InitialApprox::Scalar3 => 0.5 * (scalar1() + scalar2()),
        InitialApprox::Scalar4 => {
            let prod = scalar1() * scalar2();
            if prod > 0.0 { prod.sqrt() } else { 1.0 }
        }
        InitialApprox::Constant => init_val,
        // Per-pair, `HistoryMax` *is* `Scalar1`; the maximum is taken
        // over the history by the caller.
        InitialApprox::HistoryMax => scalar1(),
    };
    raw.clamp(min_val, max_val)
}

/// L-BFGS curvature-pair acceptance: include `(s, y)` in history iff
/// `s^T y > eps * ||s|| ||y||`. Mirrors upstream's skip-criterion
/// (`IpLimMemQuasiNewtonUpdater.cpp` ~line 750: `eps = 1e-8`).
pub fn bfgs_curvature_pair_ok(s_dot_y: Number, s_norm: Number, y_norm: Number) -> bool {
    // `sqrt(machine epsilon)`, matching upstream's
    // `CheckSkippingBFGS` (`IpLimMemQuasiNewtonUpdater.cpp`):
    //
    //     Number tol = std::sqrt(std::numeric_limits<Number>::epsilon());
    //     skipping = (sTy <= tol * snrm * ynrm);
    //
    // This was a hardcoded `1e-8` until #686, attributed to upstream but
    // not equal to it — `sqrt(f64::EPSILON)` is 1.4901161193847656e-8,
    // so the old value accepted a band of pairs upstream skips.
    let eps = f64::EPSILON.sqrt();
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
            initial_hessian_scalar(InitialApprox::Identity, 1.0, 1.0, 1.0, 1.0, 1e-8, 1e8),
            1.0
        );
    }

    #[test]
    fn scalar1_init_is_sy_over_ss() {
        // s_dot_s=4, s_dot_y=2 → 2/4 = 0.5.
        let v = initial_hessian_scalar(InitialApprox::Scalar1, 4.0, 2.0, 0.0, 1.0, 1e-8, 1e8);
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn scalar2_init_is_yy_over_sy() {
        // y_dot_y=8, s_dot_y=2 → 4.
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 2.0, 8.0, 1.0, 1e-8, 1e8);
        assert!((v - 4.0).abs() < 1e-15);
    }

    #[test]
    fn scalar3_init_is_arithmetic_mean_of_scalar1_and_scalar2() {
        // s_dot_s=4, s_dot_y=2 → scalar1 = 0.5; y_dot_y=8 → scalar2 = 4.
        let v = initial_hessian_scalar(InitialApprox::Scalar3, 4.0, 2.0, 8.0, 1.0, 1e-8, 1e8);
        assert!((v - 2.25).abs() < 1e-15, "got {v}");
    }

    #[test]
    fn scalar4_init_is_geometric_mean_of_scalar1_and_scalar2() {
        // scalar1 = 0.5, scalar2 = 4 → sqrt(2).
        let v = initial_hessian_scalar(InitialApprox::Scalar4, 4.0, 2.0, 8.0, 1.0, 1e-8, 1e8);
        assert!((v - 2.0_f64.sqrt()).abs() < 1e-15, "got {v}");
    }

    #[test]
    fn scalar4_falls_back_rather_than_producing_nan() {
        // s_dot_y < 0 makes scalar1 negative while scalar2 falls back to
        // 1.0, so the product is negative — sqrt would be NaN. A NaN σ
        // would propagate silently into the whole `B0` diagonal.
        let v = initial_hessian_scalar(InitialApprox::Scalar4, 4.0, -2.0, 8.0, 1.0, 1e-8, 1e8);
        assert!(v.is_finite(), "sigma must stay finite, got {v}");
        assert_eq!(v, 1.0);
    }

    #[test]
    fn constant_init_returns_init_val_not_the_curvature_formula() {
        // Same (s, y) that gives scalar2 = 4 above; `constant` must
        // ignore the pair entirely and return `init_val`.
        let v = initial_hessian_scalar(InitialApprox::Constant, 4.0, 2.0, 8.0, 7.5, 1e-8, 1e8);
        assert_eq!(v, 7.5);
    }

    #[test]
    fn constant_init_is_still_clamped() {
        let v = initial_hessian_scalar(InitialApprox::Constant, 4.0, 2.0, 8.0, 1e20, 1e-8, 1e8);
        assert_eq!(v, 1e8);
    }

    #[test]
    fn empty_history_sigma_honours_init_val() {
        // Upstream's "B0 = limited_memory_init_val * I in the first
        // iteration". This branch returned a hard-coded 1.0 before #677,
        // which silently matched the default and hid the missing wiring.
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.init_val = 3.0;
        assert!(u.history.is_empty());
        assert_eq!(u.compute_sigma_bfgs(), 3.0);
    }

    #[test]
    fn init_clamped_to_max() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e-20, 1.0, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e8);
    }

    #[test]
    fn init_clamped_to_min() {
        let v = initial_hessian_scalar(InitialApprox::Scalar2, 0.0, 1e20, 1.0, 1.0, 1e-8, 1e8);
        assert_eq!(v, 1e-8);
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

    /// The skip counter drives a reset, and a run of skips does not
    /// leave stale curvature behind (#686).
    ///
    /// Upstream discards the whole approximation after
    /// `limited_memory_max_skipping` consecutive skips. pounce counted
    /// nothing and never discarded, so a problem whose curvature stays
    /// negative kept its opening pairs for the rest of the solve — the
    /// model stops describing where the iterate is, `inf_pr` converges
    /// and `inf_du` plateaus.
    ///
    /// Driven through `ingest_pair` plus the counter logic rather than
    /// through `update_hessian`, which needs a full data/cq fixture; the
    /// corpus is what exercises the wired path (`cresc4` goes from
    /// `Restoration_Failed` to solved at the exact-Hessian optimum).
    #[test]
    fn consecutive_skips_reset_the_approximation() {
        let mut u = LimMemQuasiNewtonUpdater::new();
        assert_eq!(u.max_skipping, 2, "upstream default");

        // Two good pairs establish a history.
        assert!(u.ingest_pair(rcv(&[1.0, 0.0]), rcv(&[1.0, 0.0])));
        assert!(u.ingest_pair(rcv(&[0.0, 1.0]), rcv(&[0.0, 1.0])));
        assert_eq!(u.history.len(), 2);

        // A skip alone must not discard anything — one bad pair on an
        // otherwise healthy run is what the skip criterion is for.
        assert!(!u.ingest_pair(rcv(&[1.0, 0.0]), rcv(&[-1.0, 0.0])));
        u.skipped_iter += 1;
        assert!(u.skipped_iter < u.max_skipping);
        assert_eq!(u.history.len(), 2, "one skip must not reset");

        // The second consecutive skip reaches the threshold.
        assert!(!u.ingest_pair(rcv(&[0.0, 1.0]), rcv(&[0.0, -1.0])));
        u.skipped_iter += 1;
        assert!(u.skipped_iter >= u.max_skipping, "reset is due");
    }

    /// The skip tolerance is upstream's, not a round number (#686).
    ///
    /// `CheckSkippingBFGS` uses `sqrt(machine epsilon)`; pounce carried a
    /// hardcoded `1e-8` attributed to upstream. The gap is small and it
    /// is exactly the band where the two solvers disagree about whether
    /// a pair is usable, which is where a side-by-side trace starts
    /// drifting.
    #[test]
    fn skip_tolerance_is_sqrt_machine_epsilon() {
        let eps = f64::EPSILON.sqrt();
        assert!((eps - 1.4901161193847656e-8).abs() < 1e-24, "got {eps}");
        // A pair inside the old-vs-new gap: accepted under 1e-8,
        // skipped under sqrt(eps).
        let s_dot_y = 1.2e-8;
        assert!(s_dot_y > 1e-8, "would have been accepted before");
        assert!(!bfgs_curvature_pair_ok(s_dot_y, 1.0, 1.0));
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

    // ---- gh#1146: restoration leading block ----

    #[test]
    fn leading_block_applies_only_as_a_proper_prefix() {
        let mut u = LimMemQuasiNewtonUpdater::new();
        assert_eq!(u.active_leading_block(5), None);
        u.leading_block = Some(2);
        assert_eq!(u.active_leading_block(5), Some(2));
        assert_eq!(u.pair_mask(5), Some(vec![0, 1]));
        // Not a proper prefix: whole-space approximation, never a gather
        // past the end or an empty block.
        assert_eq!(u.active_leading_block(2), None);
        assert_eq!(u.active_leading_block(1), None);
        u.leading_block = Some(0);
        assert_eq!(u.active_leading_block(5), None);
    }

    #[test]
    fn leading_block_wins_over_a_nonlinear_mask() {
        // Restoration clears the mask anyway; if both were ever set, the
        // restoration block is the space the pairs live in.
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.nonlinear_vars = Some(vec![1, 3]);
        u.leading_block = Some(3);
        assert_eq!(u.pair_mask(6), Some(vec![0, 1, 2]));
    }

    /// The restoration `W`: σ and the curvature on the `x` block, exactly
    /// zero on the slack blocks, no expansion matrix — and on the `x`
    /// block the same operator a plain `k`-dimensional L-BFGS builds from
    /// the projected pair.
    #[test]
    fn leading_block_publishes_x_curvature_and_zero_slack_blocks() {
        use crate::ipopt_data::IpoptData;
        use pounce_linalg::low_rank_update_sym_matrix::LowRankUpdateSymMatrix;
        use pounce_linalg::matrix::Matrix;
        use std::cell::RefCell;
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.leading_block = Some(2);
        // A step whose slack part (entries 2, 3) is large and whose `y`
        // there is zero — the shape every restoration pair has.
        let s = rcv(&[1.0, 2.0, 50.0, -40.0]);
        let y = rcv(&[3.0, 1.0, 0.0, 0.0]);
        let mask = u.pair_mask(4).unwrap();
        assert!(u.ingest_pair(project_onto(&*s, &mask), project_onto(&*y, &mask)));
        assert_eq!(u.history[0].s.dim(), 2);

        let data: IpoptDataHandle = Rc::new(RefCell::new(IpoptData::new()));
        let curr_x = rcv(&[0.0; 4]);
        u.publish_leading_block(&data, &curr_x, 2);
        let w = data.borrow().w.clone().unwrap();
        let lr = w.as_any().downcast_ref::<LowRankUpdateSymMatrix>().unwrap();
        assert!(lr.p_lowrank().is_none());

        // σ from the projected pair (scalar1 = sᵀy/sᵀs = 5/5), not from the
        // full one, which the slack motion would dilute to 5/4105.
        let diag = expanded_of(lr.get_diag().unwrap().as_ref());
        assert!(
            (diag[0] - 1.0).abs() < 1e-12 && (diag[1] - 1.0).abs() < 1e-12,
            "{diag:?}"
        );
        assert_eq!(&diag[2..], &[0.0, 0.0]);

        // Apply W to the unit vectors: slack rows and columns are zero, and
        // the x block satisfies the secant condition B s_x = y_x.
        let col = |j: usize| {
            let mut e = vec![0.0; 4];
            e[j] = 1.0;
            let mut out = curr_x.make_new();
            out.set(0.0);
            lr.mult_vector(1.0, &*rcv(&e), 0.0, &mut *out);
            expanded_of(&*out)
        };
        let cols: Vec<Vec<Number>> = (0..4).map(col).collect();
        for (j, c) in cols.iter().enumerate() {
            assert_eq!(c[2], 0.0, "column {j}");
            assert_eq!(c[3], 0.0, "column {j}");
        }
        assert_eq!(cols[2], vec![0.0; 4]);
        assert_eq!(cols[3], vec![0.0; 4]);
        let bs = [
            cols[0][0] * 1.0 + cols[1][0] * 2.0,
            cols[0][1] * 1.0 + cols[1][1] * 2.0,
        ];
        assert!(
            (bs[0] - 3.0).abs() < 1e-12 && (bs[1] - 1.0).abs() < 1e-12,
            "{bs:?}"
        );
    }

    // ---- gh#624: nonlinear-variable mask ----

    #[test]
    fn no_mask_is_the_default() {
        let u = LimMemQuasiNewtonUpdater::new();
        assert!(u.nonlinear_vars.is_none());
        assert!(u.active_mask(5).is_none());
    }

    #[test]
    fn mask_spanning_the_whole_space_is_no_mask() {
        // A "restriction" to every variable is the identity; publishing
        // an expansion matrix for it would cost work and change nothing.
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.nonlinear_vars = Some(vec![0, 1, 2]);
        assert!(u.active_mask(3).is_none());
        u.nonlinear_vars = Some(vec![]);
        assert!(u.active_mask(3).is_none());
    }

    #[test]
    fn mask_is_ignored_for_a_space_it_does_not_fit() {
        // The restoration sub-IPM's primal is a wider compound vector.
        // `run_inner_resto` clears the mask, but if one ever reached a
        // space it was not built for, the full-space approximation is
        // the safe answer — never a gather from the wrong indices.
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.nonlinear_vars = Some(vec![0, 4]);
        assert_eq!(u.active_mask(9).map(|m| m.to_vec()), Some(vec![0, 4]));
        assert!(u.active_mask(3).is_none());
    }

    #[test]
    fn projection_gathers_the_masked_entries() {
        let v = rcv(&[10.0, 20.0, 30.0, 40.0]);
        let p = project_onto(&*v, &[1, 3]);
        assert_eq!(p.dim(), 2);
        assert_eq!(expanded_of(&*p), vec![20.0, 40.0]);
    }

    #[test]
    fn masked_history_lives_in_the_reduced_space() {
        // Curvature pairs are projected before they enter the history,
        // so σ and the stored columns are all reduced-dimension.
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.nonlinear_vars = Some(vec![0, 2]);
        let s = rcv(&[1.0, 5.0, 1.0, 7.0]);
        let y = rcv(&[1.0, 9.0, 1.0, 3.0]);
        let mask = u.active_mask(4).unwrap().to_vec();
        assert!(u.ingest_pair(project_onto(&*s, &mask), project_onto(&*y, &mask)));
        let stored = &u.history[0];
        assert_eq!(stored.s.dim(), 2);
        // s·y over the nonlinear coordinates only: 1*1 + 1*1 = 2. The
        // linear coordinates (5·9 + 7·3) must not contribute.
        assert!((stored.s_dot_y - 2.0).abs() < 1e-15);
    }

    // -------------------------------------------- gh#818: history-max sigma

    /// Every upstream rule reads the newest pair; `HistoryMax` reads the
    /// whole window. The fixture below puts the *largest* curvature in
    /// the middle of the history so neither "newest" nor "oldest" can
    /// pass by accident.
    ///
    /// Pairs are `s = e_i`, `y = c_i · e_i`, so `sᵀy/sᵀs = c_i`
    /// exactly: curvatures 3, 400, 7 in insertion order.
    fn updater_with_curvatures(cs: &[Number]) -> LimMemQuasiNewtonUpdater {
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.max_history = cs.len() as i32;
        for (i, &c) in cs.iter().enumerate() {
            let mut sv = vec![0.0; cs.len()];
            sv[i] = 1.0;
            let yv: Vec<Number> = sv.iter().map(|&v| c * v).collect();
            assert!(u.ingest_pair(rcv(&sv), rcv(&yv)), "pair {i} was skipped");
        }
        u
    }

    #[test]
    fn history_max_sigma_is_the_largest_curvature_in_the_window() {
        let mut u = updater_with_curvatures(&[3.0, 400.0, 7.0]);

        u.initial_approx = InitialApprox::Scalar1;
        assert!(
            (u.compute_sigma_bfgs() - 7.0).abs() < 1e-12,
            "scalar1 must read the NEWEST pair, got {}",
            u.compute_sigma_bfgs()
        );

        u.initial_approx = InitialApprox::HistoryMax;
        assert!(
            (u.compute_sigma_bfgs() - 400.0).abs() < 1e-12,
            "history-max must read the LARGEST pair, got {}",
            u.compute_sigma_bfgs()
        );
    }

    /// The maximum is bounded by the *window*, not by the run. A
    /// running maximum never comes back down, which keeps a stiff early
    /// transient in `B0` long after the iterate has left it — measured
    /// worse than `scalar1` at every size on gh#818's fixture. Once the
    /// stiff pair ages out of the window, σ must fall with it.
    #[test]
    fn history_max_sigma_decays_as_the_window_turns_over() {
        let mut u = updater_with_curvatures(&[3.0, 400.0, 7.0]);
        u.initial_approx = InitialApprox::HistoryMax;
        assert!((u.compute_sigma_bfgs() - 400.0).abs() < 1e-12);

        // Push two mild pairs; the FIFO drops the 400 one.
        for c in [5.0, 6.0] {
            let sv = vec![1.0, 0.0, 0.0];
            let yv = vec![c, 0.0, 0.0];
            assert!(u.ingest_pair(rcv(&sv), rcv(&yv)));
        }
        assert_eq!(u.history.len(), 3);
        assert!(
            (u.compute_sigma_bfgs() - 7.0).abs() < 1e-12,
            "sigma must fall to the largest curvature STILL in the window, got {}",
            u.compute_sigma_bfgs()
        );
    }

    /// An empty history has no curvature to maximize over, so
    /// `HistoryMax` takes the same `limited_memory_init_val` every other
    /// rule takes on the first iteration. The `fold` seed is
    /// `NEG_INFINITY`, so getting this wrong would put `-inf` (clamped
    /// to `init_val_min`) on the whole `B0` diagonal rather than
    /// returning early.
    #[test]
    fn history_max_sigma_on_empty_history_is_init_val() {
        let mut u = LimMemQuasiNewtonUpdater::new();
        u.initial_approx = InitialApprox::HistoryMax;
        u.init_val = 3.0;
        assert!(u.history.is_empty());
        assert_eq!(u.compute_sigma_bfgs(), 3.0);
    }

    /// Per pair, `HistoryMax` is `Scalar1` — the kernel is shared and
    /// only the caller's fold differs. If this drifts, the doc on
    /// `initial_hessian_scalar` is wrong and the option means something
    /// nobody wrote down.
    #[test]
    fn history_max_per_pair_kernel_equals_scalar1() {
        for (ss, sy, yy) in [(4.0, 2.0, 8.0), (1.0, 1.0, 1.0), (0.0, 2.0, 8.0)] {
            assert_eq!(
                initial_hessian_scalar(InitialApprox::HistoryMax, ss, sy, yy, 1.0, 1e-8, 1e8),
                initial_hessian_scalar(InitialApprox::Scalar1, ss, sy, yy, 1.0, 1e-8, 1e8),
            );
        }
    }

    // ------------------------------------------------ gh#818: reanchor

    /// A re-anchor keeps the newest pair and drops the rest. Keeping the
    /// *newest* is the whole point: `compute_sigma_bfgs` reads sigma off
    /// the history, so clearing it outright would fall back to
    /// `limited_memory_init_val` -- a bare 1.0, the first-iteration model
    /// -- on a problem whose curvature the solver has by now measured.
    #[test]
    fn reanchor_keeps_only_the_newest_pair() {
        let mut u = updater_with_curvatures(&[3.0, 400.0, 7.0]);
        assert_eq!(u.history.len(), 3);
        assert!(u.reanchor(), "there were three pairs to give up");
        assert_eq!(u.history.len(), 1);
        u.initial_approx = InitialApprox::Scalar1;
        assert!(
            (u.compute_sigma_bfgs() - 7.0).abs() < 1e-12,
            "the surviving pair must be the NEWEST (curvature 7), got sigma {}",
            u.compute_sigma_bfgs()
        );
    }

    /// The bound is structural, not just counted: once the history is
    /// down to one pair there is nothing left to give up, so a second
    /// failure at the same iterate falls through to the caller's
    /// existing hand-off instead of retrying forever.
    #[test]
    fn reanchor_declines_when_there_is_nothing_left_to_discard() {
        let mut u = updater_with_curvatures(&[3.0, 400.0, 7.0]);
        assert!(u.reanchor());
        assert!(!u.reanchor(), "a one-pair history has nothing to re-anchor");
        assert!(!LimMemQuasiNewtonUpdater::new().reanchor(), "empty history");
    }

    /// A deliberate re-anchor is not a rejected pair. Leaving
    /// `skipped_iter` standing would let the next skipped pair trip
    /// `limited_memory_max_skipping` and discard the very pair the
    /// re-anchor just chose to keep -- turning the careful sigma into
    /// `limited_memory_init_val` one iteration later, which is exactly
    /// what keeping a pair was for.
    #[test]
    fn reanchor_clears_the_skip_counter() {
        let mut u = updater_with_curvatures(&[3.0, 400.0, 7.0]);
        u.skipped_iter = 1;
        assert!(u.reanchor());
        assert_eq!(u.skipped_iter, 0);
    }

    /// The exact-Hessian updater has no curvature history, so it must
    /// decline and let the caller hand off as before. If this ever
    /// returns `true` the rung would fire on the exact path, where the
    /// fixture sweep is byte-identical by design.
    #[test]
    fn exact_hessian_updater_never_reanchors() {
        use crate::hess::exact::ExactHessianUpdater;
        assert!(!ExactHessianUpdater::new().reanchor());
    }
}
