//! Quality-function mu oracle — port of
//! `IpQualityFunctionMuOracle.{hpp,cpp}`. Phase 10.
//!
//! The oracle picks `μ_new = σ * avrg_compl` by minimizing a 1-D
//! quality function `q(σ)` over `σ ∈ [σ_lo, σ_up]` via golden section.
//! The full vector-valued evaluator (which builds the trial slack /
//! multiplier vectors at a candidate σ and reduces them to a scalar
//! norm) is split into two pieces:
//!
//! * `evaluate_quality_function` — a *pure-scalar* reducer that takes
//!   already-computed `‖·‖` aggregates and combines them per the
//!   `(norm, centrality, balancing)` triple per
//!   `IpQualityFunctionMuOracle.cpp:566-658`. The vector→aggregate
//!   step is the caller's responsibility.
//! * `pick_sigma` — orchestrator that mirrors
//!   `IpQualityFunctionMuOracle.cpp::CalculateMu` lines 329-385: picks
//!   the σ-bracket, evaluates `q(1)` and `q(1−ε)` to decide whether
//!   to search above or below 1, then drives `golden_section`.
//!
//! Wiring `pick_sigma` to a fully populated `q(σ)` evaluator —
//! including the centering predictor solve — is the remaining scope.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::mu::oracle::r#trait::MuOracle;
use pounce_common::types::Number;
use pounce_linalg::Vector;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    OneNorm,
    /// Squared 2-norm — upstream `NM_NORM_2_SQUARED` (default).
    /// Aggregates are `||·||²` (no sqrt) and `(1−α)²` weighting.
    TwoNormSquared,
    TwoNorm,
    MaxNorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentralityType {
    None,
    LogCenter,
    ReciprocalCenter,
    CubedReciprocalCenter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalancingTermType {
    None,
    CubicTerm,
}

pub struct QualityFunctionMuOracle {
    pub norm_type: NormType,
    pub centrality_type: CentralityType,
    pub balancing_term: BalancingTermType,
    pub max_section_steps: i32,
    pub section_sigma_tol: Number,
    pub section_qf_tol: Number,
    pub sigma_max: Number,
    pub sigma_min: Number,
    pub mu_min: Number,
    pub mu_max: Number,
}

impl Default for QualityFunctionMuOracle {
    fn default() -> Self {
        // Defaults from `IpQualityFunctionMuOracle.cpp:RegisterOptions`.
        Self {
            norm_type: NormType::TwoNormSquared,
            centrality_type: CentralityType::None,
            balancing_term: BalancingTermType::None,
            max_section_steps: 8,
            section_sigma_tol: 1e-2,
            section_qf_tol: 0.0,
            sigma_max: 100.0,
            // Upstream `IpQualityFunctionMuOracle.cpp:62-69`
            // `RegisterOptions` default is 1e-6, not 1e-9. Setting it
            // too low lets golden-section collapse σ all the way to
            // the floor on outer iterations where q(σ) is nearly
            // flat over the bracket — which then drives μ to ~1e-11
            // in a single step and triggers a kappa_sigma blow-up
            // that pushes the algorithm into restoration. (HS1NE
            // and ~50 other CUTEst problems exhibited this.)
            sigma_min: 1e-6,
            mu_min: 1e-11,
            mu_max: 1e5,
        }
    }
}

impl QualityFunctionMuOracle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drive the predictor + centring solves through `pd_search_dir`,
    /// project the results onto the four bound-mask subspaces, then
    /// run [`pick_sigma`] over a `q(σ)` closure that builds the σ-step
    /// and reduces it to [`QualityFunctionAggregates`] before invoking
    /// [`evaluate_quality_function`]. Mirrors upstream
    /// `IpQualityFunctionMuOracle.cpp::CalculateMu` lines 188-485.
    ///
    /// Returns `None` if either linear solve fails (caller falls back
    /// to LOQO, matching upstream's
    /// `IpAdaptiveMuUpdate.cpp::CalculateMuFromOracle:330-340`).
    #[allow(clippy::too_many_lines)]
    pub fn calculate_mu_with_predictor_centering(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd_search_dir: &mut PdSearchDirCalc,
    ) -> Option<Number> {
        if !pd_search_dir.compute_affine_step(data, cq, nlp) {
            return None;
        }
        if !pd_search_dir.compute_centering_step(data, cq, nlp) {
            return None;
        }

        let delta_aff: IteratesVector = data.borrow().delta_aff.clone()?;
        let delta_cen: IteratesVector = data.borrow().delta_cen.clone()?;

        // Project step.x onto the bound subspaces. Each block matches
        // the `step_aff_x_L = P_L^T·δ_aff_x` setup in
        // `IpQualityFunctionMuOracle.cpp:308-323`.
        let nlp_ref = nlp.borrow();
        let cq_ref = cq.borrow();
        let curr_iv = cq_ref.curr_iv();
        let curr_slack_x_l = cq_ref.curr_slack_x_l();
        let curr_slack_x_u = cq_ref.curr_slack_x_u();
        let curr_slack_s_l = cq_ref.curr_slack_s_l();
        let curr_slack_s_u = cq_ref.curr_slack_s_u();
        let avrg_compl = cq_ref.curr_avrg_compl();

        let project = |sign_l_x: Number,
                       sign_u_x: Number,
                       step_x: &dyn Vector,
                       step_s: &dyn Vector|
         -> [Rc<dyn Vector>; 4] {
            let mut x_l = curr_slack_x_l.make_new();
            nlp_ref
                .px_l()
                .trans_mult_vector(sign_l_x, step_x, 0.0, &mut *x_l);
            let mut x_u = curr_slack_x_u.make_new();
            nlp_ref
                .px_u()
                .trans_mult_vector(sign_u_x, step_x, 0.0, &mut *x_u);
            let mut s_l = curr_slack_s_l.make_new();
            nlp_ref
                .pd_l()
                .trans_mult_vector(sign_l_x, step_s, 0.0, &mut *s_l);
            let mut s_u = curr_slack_s_u.make_new();
            nlp_ref
                .pd_u()
                .trans_mult_vector(sign_u_x, step_s, 0.0, &mut *s_u);
            [Rc::from(x_l), Rc::from(x_u), Rc::from(s_l), Rc::from(s_u)]
        };

        let [step_aff_x_l, step_aff_x_u, step_aff_s_l, step_aff_s_u] =
            project(1.0, -1.0, &*delta_aff.x, &*delta_aff.s);
        let [step_cen_x_l, step_cen_x_u, step_cen_s_l, step_cen_s_u] =
            project(1.0, -1.0, &*delta_cen.x, &*delta_cen.s);

        // The z/v step blocks are stored directly on the iterate — no
        // projection needed (upstream lines 318-323 use the raw blocks).
        let step_aff_z_l = delta_aff.z_l.clone();
        let step_aff_z_u = delta_aff.z_u.clone();
        let step_aff_v_l = delta_aff.v_l.clone();
        let step_aff_v_u = delta_aff.v_u.clone();
        let step_cen_z_l = delta_cen.z_l.clone();
        let step_cen_z_u = delta_cen.z_u.clone();
        let step_cen_v_l = delta_cen.v_l.clone();
        let step_cen_v_u = delta_cen.v_u.clone();

        // Drop the immutable nlp borrow before invoking CQ accessors
        // that may take a `nlp.borrow_mut()` (e.g. `curr_grad_lag_x` →
        // `curr_grad_f` → `nlp.eval_grad_f`).
        drop(nlp_ref);

        // Constant-in-σ aggregates: `dual_aggr` from ‖∇L_x‖, ‖∇L_s‖;
        // `primal_aggr` from ‖c‖, ‖d−s‖. Norm choice driven by
        // `self.norm_type`. Upstream `cpp:283-303`.
        let grad_lag_x = cq_ref.curr_grad_lag_x();
        let grad_lag_s = cq_ref.curr_grad_lag_s();
        let c = cq_ref.curr_c();
        let d_minus_s = cq_ref.curr_d_minus_s();
        let dual_aggr = match self.norm_type {
            NormType::OneNorm => grad_lag_x.asum() + grad_lag_s.asum(),
            NormType::TwoNormSquared => {
                let nx = grad_lag_x.nrm2();
                let ns = grad_lag_s.nrm2();
                nx * nx + ns * ns
            }
            NormType::TwoNorm => {
                let nx = grad_lag_x.nrm2();
                let ns = grad_lag_s.nrm2();
                (nx * nx + ns * ns).sqrt()
            }
            NormType::MaxNorm => grad_lag_x.amax().max(grad_lag_s.amax()),
        };
        let primal_aggr = match self.norm_type {
            NormType::OneNorm => c.asum() + d_minus_s.asum(),
            NormType::TwoNormSquared => {
                let nc = c.nrm2();
                let nd = d_minus_s.nrm2();
                nc * nc + nd * nd
            }
            NormType::TwoNorm => {
                let nc = c.nrm2();
                let nd = d_minus_s.nrm2();
                (nc * nc + nd * nd).sqrt()
            }
            NormType::MaxNorm => c.amax().max(d_minus_s.amax()),
        };

        let n_dual = curr_iv.x.dim() + curr_iv.s.dim();
        let n_pri = curr_iv.y_c.dim() + curr_iv.y_d.dim();
        let n_comp = curr_iv.z_l.dim() + curr_iv.z_u.dim() + curr_iv.v_l.dim() + curr_iv.v_u.dim();
        let tau = data.borrow().curr_tau;

        let curr_z_l = curr_iv.z_l.clone();
        let curr_z_u = curr_iv.z_u.clone();
        let curr_v_l = curr_iv.v_l.clone();
        let curr_v_u = curr_iv.v_u.clone();

        drop(cq_ref);

        let norm_type = self.norm_type;
        let centrality = self.centrality_type;
        let balancing = self.balancing_term;

        // q(σ) closure. Captures the eight aff/cen step projections,
        // the four current slacks, the four current bound multipliers,
        // and the constant aggregates; pure scalar work per call.
        let mut eval_q = |sigma: Number| -> Number {
            // step_σ = step_aff + σ · step_cen, projected blocks.
            let combine = |aff: &Rc<dyn Vector>, cen: &Rc<dyn Vector>| -> Box<dyn Vector> {
                let mut out = aff.make_new();
                out.set(0.0);
                out.add_two_vectors(1.0, &**aff, sigma, &**cen, 0.0);
                out
            };
            let stp_x_l = combine(&step_aff_x_l, &step_cen_x_l);
            let stp_x_u = combine(&step_aff_x_u, &step_cen_x_u);
            let stp_s_l = combine(&step_aff_s_l, &step_cen_s_l);
            let stp_s_u = combine(&step_aff_s_u, &step_cen_s_u);
            let stp_z_l = combine(&step_aff_z_l, &step_cen_z_l);
            let stp_z_u = combine(&step_aff_z_u, &step_cen_z_u);
            let stp_v_l = combine(&step_aff_v_l, &step_cen_v_l);
            let stp_v_u = combine(&step_aff_v_u, &step_cen_v_u);

            // α_pri = min over slacks of frac_to_bound(curr_slack, step, τ).
            let alpha_pri = curr_slack_x_l
                .frac_to_bound(&*stp_x_l, tau)
                .min(curr_slack_x_u.frac_to_bound(&*stp_x_u, tau))
                .min(curr_slack_s_l.frac_to_bound(&*stp_s_l, tau))
                .min(curr_slack_s_u.frac_to_bound(&*stp_s_u, tau));
            let alpha_du = curr_z_l
                .frac_to_bound(&*stp_z_l, tau)
                .min(curr_z_u.frac_to_bound(&*stp_z_u, tau))
                .min(curr_v_l.frac_to_bound(&*stp_v_l, tau))
                .min(curr_v_u.frac_to_bound(&*stp_v_u, tau));

            // Build σ-step trial slacks/duals: trial = curr + α·step.
            let mut trial_s_x_l = curr_slack_x_l.make_new();
            trial_s_x_l.set(0.0);
            trial_s_x_l.add_two_vectors(1.0, &*curr_slack_x_l, alpha_pri, &*stp_x_l, 0.0);
            let mut trial_s_x_u = curr_slack_x_u.make_new();
            trial_s_x_u.set(0.0);
            trial_s_x_u.add_two_vectors(1.0, &*curr_slack_x_u, alpha_pri, &*stp_x_u, 0.0);
            let mut trial_s_s_l = curr_slack_s_l.make_new();
            trial_s_s_l.set(0.0);
            trial_s_s_l.add_two_vectors(1.0, &*curr_slack_s_l, alpha_pri, &*stp_s_l, 0.0);
            let mut trial_s_s_u = curr_slack_s_u.make_new();
            trial_s_s_u.set(0.0);
            trial_s_s_u.add_two_vectors(1.0, &*curr_slack_s_u, alpha_pri, &*stp_s_u, 0.0);

            let mut trial_z_l = curr_z_l.make_new();
            trial_z_l.set(0.0);
            trial_z_l.add_two_vectors(1.0, &*curr_z_l, alpha_du, &*stp_z_l, 0.0);
            let mut trial_z_u = curr_z_u.make_new();
            trial_z_u.set(0.0);
            trial_z_u.add_two_vectors(1.0, &*curr_z_u, alpha_du, &*stp_z_u, 0.0);
            let mut trial_v_l = curr_v_l.make_new();
            trial_v_l.set(0.0);
            trial_v_l.add_two_vectors(1.0, &*curr_v_l, alpha_du, &*stp_v_l, 0.0);
            let mut trial_v_u = curr_v_u.make_new();
            trial_v_u.set(0.0);
            trial_v_u.add_two_vectors(1.0, &*curr_v_u, alpha_du, &*stp_v_u, 0.0);

            // Complementarity products at the σ-trial point.
            trial_s_x_l.element_wise_multiply(&*trial_z_l);
            trial_s_x_u.element_wise_multiply(&*trial_z_u);
            trial_s_s_l.element_wise_multiply(&*trial_v_l);
            trial_s_s_u.element_wise_multiply(&*trial_v_u);

            let compl_aggr = match norm_type {
                NormType::OneNorm => {
                    trial_s_x_l.asum()
                        + trial_s_x_u.asum()
                        + trial_s_s_l.asum()
                        + trial_s_s_u.asum()
                }
                NormType::TwoNormSquared => {
                    let a = trial_s_x_l.nrm2();
                    let b = trial_s_x_u.nrm2();
                    let c = trial_s_s_l.nrm2();
                    let d = trial_s_s_u.nrm2();
                    a * a + b * b + c * c + d * d
                }
                NormType::TwoNorm => {
                    let a = trial_s_x_l.nrm2();
                    let b = trial_s_x_u.nrm2();
                    let c = trial_s_s_l.nrm2();
                    let d = trial_s_s_u.nrm2();
                    (a * a + b * b + c * c + d * d).sqrt()
                }
                NormType::MaxNorm => trial_s_x_l
                    .amax()
                    .max(trial_s_x_u.amax())
                    .max(trial_s_s_l.amax())
                    .max(trial_s_s_u.amax()),
            };

            let xi = if matches!(centrality, CentralityType::None) {
                1.0
            } else {
                // Centrality: min(s_i z_i) / avg(s_i z_i). Cheap proxy
                // when centrality != None — upstream computes the same
                // ratio at line 612 onward.
                let total = trial_s_x_l.asum()
                    + trial_s_x_u.asum()
                    + trial_s_s_l.asum()
                    + trial_s_s_u.asum();
                let avg = if n_comp > 0 {
                    total / n_comp as Number
                } else {
                    1.0
                };
                let mn = trial_s_x_l
                    .min()
                    .min(trial_s_x_u.min())
                    .min(trial_s_s_l.min())
                    .min(trial_s_s_u.min());
                if avg > 0.0 { mn / avg } else { 1.0 }
            };

            let aggr = QualityFunctionAggregates {
                dual_aggr,
                primal_aggr,
                compl_aggr,
                n_dual,
                n_pri,
                n_comp,
            };

            if std::env::var("POUNCE_DBG_QF_AGGR").is_ok() {
                tracing::debug!(target: "pounce::mu",
                    "[QF_AGGR] σ={:.6e} α_pri={:.6e} α_du={:.6e} xi={:.6e} dual_aggr={:.6e} primal_aggr={:.6e} compl_aggr={:.6e} n_dual={} n_pri={} n_comp={}",
                    sigma, alpha_pri, alpha_du, xi,
                    dual_aggr, primal_aggr, compl_aggr,
                    n_dual, n_pri, n_comp
                );
            }

            evaluate_quality_function(
                norm_type, centrality, balancing, alpha_pri, alpha_du, xi, aggr,
            )
        };

        // One-shot σ-sweep dump for iter==N: emits q(σ) at 21 σ values
        // spanning [σ_min, σ_max] log-uniform. Enable with
        // `POUNCE_DBG_QF_SWEEP=<iter>` (matches `data.iter_count`).
        if let Ok(s) = std::env::var("POUNCE_DBG_QF_SWEEP") {
            if let Ok(target_iter) = s.parse::<i32>() {
                if data.borrow().iter_count == target_iter {
                    let lo = self.sigma_min.max(self.mu_min / avrg_compl);
                    let hi = self.sigma_max.min(self.mu_max / avrg_compl).max(lo * 10.0);
                    let log_lo = lo.ln();
                    let log_hi = hi.ln();
                    tracing::debug!(target: "pounce::mu", "[QF_SWEEP] iter={} avrg_compl={:.6e} σ_range=[{:.3e},{:.3e}] sigma_min={:.3e} sigma_max={:.3e} mu_min={:.3e} mu_max={:.3e}",
                        target_iter, avrg_compl, lo, hi,
                        self.sigma_min, self.sigma_max, self.mu_min, self.mu_max);
                    let n = 21;
                    for i in 0..n {
                        let frac = i as f64 / (n - 1) as f64;
                        let sig = (log_lo + frac * (log_hi - log_lo)).exp();
                        let q = eval_q(sig);
                        tracing::debug!(target: "pounce::mu", "[QF_SWEEP] σ={:.6e} q={:.10e}", sig, q);
                    }
                    let q1 = eval_q(1.0);
                    let s1m = 1.0 - self.section_sigma_tol.max(1e-4);
                    let q1m = eval_q(s1m);
                    tracing::debug!(target: "pounce::mu",
                        "[QF_SWEEP] σ=1.0 q={:.10e}  σ={:.6e} q={:.10e}  (q_1minus>q_1: {})",
                        q1,
                        s1m,
                        q1m,
                        q1m > q1
                    );
                }
            }
        }

        let sigma = pick_sigma(
            self.sigma_min,
            self.sigma_max,
            self.mu_min,
            self.mu_max,
            avrg_compl,
            self.section_sigma_tol,
            self.section_qf_tol,
            self.max_section_steps,
            &mut eval_q,
        );

        let mu_new = sigma * avrg_compl;
        let mu_clamped = mu_new.clamp(self.mu_min, self.mu_max);
        if std::env::var("POUNCE_DBG_QF").is_ok() {
            let iter_count = data.borrow().iter_count;
            let curr_mu = data.borrow().curr_mu;
            let sigma_floor = self.sigma_min.max(self.mu_min / avrg_compl);
            let sigma_up_dn = sigma_floor
                .max(1.0 - self.section_sigma_tol.max(1e-4))
                .min(self.mu_max / avrg_compl);
            tracing::debug!(target: "pounce::mu",
                "[QF] iter={} curr_mu={:.3e} avrg_compl={:.3e} sigma={:.3e} mu_new={:.3e} mu_clamped={:.3e} | sigma_min={:.3e} mu_min={:.3e} sigma_lo_dn={:.3e} sigma_up_dn={:.3e} mu_min/avrg={:.3e}",
                iter_count, curr_mu, avrg_compl, sigma, mu_new, mu_clamped,
                self.sigma_min, self.mu_min, sigma_floor, sigma_up_dn,
                self.mu_min / avrg_compl,
            );
        }
        Some(mu_clamped)
    }
}

impl MuOracle for QualityFunctionMuOracle {
    fn calculate_mu(&mut self) -> Option<Number> {
        // The full oracle needs the affine and centering steps; until
        // the iterate plumbing is finalized, return None so the
        // adaptive μ update falls through to the LOQO fallback as
        // upstream does at `IpAdaptiveMuUpdate.cpp:CheckSufficientProgress`.
        None
    }
}

/// Pure-scalar golden-section minimizer used by
/// `QualityFunctionMuOracle::PerformGoldenSection`
/// (`IpQualityFunctionMuOracle.cpp:668-790`).
///
/// Searches for `argmin_{σ ∈ [σ_lo, σ_up]} q(σ)` via golden-section.
/// Stops when *either*:
/// * `(σ_up − σ_lo) < σ_tol · σ_up` (relative width), or
/// * `1 − min(q_corners) / max(q_corners) < qf_tol` (function flat),
/// * `nsections ≥ max_steps`.
///
/// `q_lo` / `q_up` are the function values at the bracket endpoints,
/// as in upstream where they're often pre-evaluated and a sentinel
/// `-100.0` is passed when the value isn't yet known.
pub fn golden_section(
    sigma_lo_in: Number,
    sigma_up_in: Number,
    q_lo_in: Number,
    q_up_in: Number,
    sigma_tol: Number,
    qf_tol: Number,
    max_steps: i32,
    mut q: impl FnMut(Number) -> Number,
) -> Number {
    let mut sigma_lo = sigma_lo_in;
    let mut sigma_up = sigma_up_in;
    let mut q_lo = q_lo_in;
    let mut q_up = q_up_in;

    let gfac = (3.0 - 5.0_f64.sqrt()) / 2.0;
    let mut sigma_mid1 = sigma_lo + gfac * (sigma_up - sigma_lo);
    let mut sigma_mid2 = sigma_lo + (1.0 - gfac) * (sigma_up - sigma_lo);
    let mut qmid1 = q(sigma_mid1);
    let mut qmid2 = q(sigma_mid2);

    let mut nsections = 0;
    let mut width_ok;
    let mut qf_ok;
    loop {
        width_ok = (sigma_up - sigma_lo) >= sigma_tol * sigma_up;
        let qmin = q_lo.min(q_up).min(qmid1).min(qmid2);
        let qmax = q_lo.max(q_up).max(qmid1).max(qmid2);
        qf_ok = qmax > 0.0 && (1.0 - qmin / qmax) >= qf_tol;
        if !(width_ok && qf_ok && nsections < max_steps) {
            break;
        }
        nsections += 1;
        if qmid1 > qmid2 {
            sigma_lo = sigma_mid1;
            q_lo = qmid1;
            sigma_mid1 = sigma_mid2;
            qmid1 = qmid2;
            sigma_mid2 = sigma_lo + (1.0 - gfac) * (sigma_up - sigma_lo);
            qmid2 = q(sigma_mid2);
        } else {
            sigma_up = sigma_mid2;
            q_up = qmid2;
            sigma_mid2 = sigma_mid1;
            qmid2 = qmid1;
            sigma_mid1 = sigma_lo + gfac * (sigma_up - sigma_lo);
            qmid1 = q(sigma_mid1);
        }
    }

    // Post-loop selection — mirrors `IpQualityFunctionMuOracle.cpp:749-826`.
    //
    // Two distinct cases:
    //  * **qf_tol stop** (`width_ok && !qf_ok`): the four sampled values
    //    have converged to within `qf_tol`. Pick whichever of the four
    //    has the smallest q. Upstream reaches this branch only with real
    //    values — its loop condition `(1 - qmin/qmax) >= qf_tol` keeps a
    //    sentinel state alive (sentinel `-100.0` yields a large positive
    //    ratio) until the slot is overwritten, so `DBG_ASSERT(qf_min > -100.)`
    //    holds. pounce, however, adds a `qmax > 0.0` guard to `qf_ok`
    //    (line 499) to avoid a divide-by-zero when every sample is ≤ 0; that
    //    guard can force `qf_ok = false` while an endpoint still holds the
    //    sentinel, routing it here. So this branch must re-evaluate an unmoved
    //    sentinel endpoint first (below), exactly like the else-branch (L4).
    //  * **Else** (`!width_ok || nsections == max_steps`): pick min of
    //    the two midpoints, then check whether either endpoint *never
    //    moved during the loop*. If an unmoved endpoint was passed in
    //    with the `-100.0` sentinel, it has not been evaluated yet —
    //    compute its q now and compare. Without this, callers that
    //    pass a sentinel endpoint (every `pick_sigma` call does — one
    //    of `q_lo`/`q_up` is always `-100.0`) can have the routine
    //    return that *unevaluated* endpoint as the minimum, which is
    //    how DECONVBNE used to land on `sigma = sigma_min`.
    if width_ok && !qf_ok {
        // Re-evaluate any endpoint that *never moved during the loop* and is
        // still carrying the `-100.0` sentinel, before selecting the minimum.
        // Upstream only reaches this branch with real values (its loop keeps a
        // sentinel state alive because it lacks the `qmax > 0.0` guard); the
        // guard pounce adds at line 499 can route a sentinel-containing state
        // here, so we must mirror the else-branch / upstream re-evaluation or
        // we would return an unevaluated endpoint as the spurious minimum (L4).
        if sigma_lo == sigma_lo_in && q_lo < 0.0 {
            q_lo = q(sigma_lo);
        }
        if sigma_up == sigma_up_in && q_up < 0.0 {
            q_up = q(sigma_up);
        }
        let mut best_s = sigma_lo;
        let mut best_q = q_lo;
        if q_up < best_q {
            best_s = sigma_up;
            best_q = q_up;
        }
        if qmid1 < best_q {
            best_s = sigma_mid1;
            best_q = qmid1;
        }
        if qmid2 < best_q {
            best_s = sigma_mid2;
        }
        return best_s;
    }
    let (mut sigma, mut qval) = if qmid1 < qmid2 {
        (sigma_mid1, qmid1)
    } else {
        (sigma_mid2, qmid2)
    };
    if sigma_up == sigma_up_in {
        let qtmp = if q_up < 0.0 { q(sigma_up) } else { q_up };
        if qtmp < qval {
            sigma = sigma_up;
            qval = qtmp;
        }
    } else if sigma_lo == sigma_lo_in {
        let qtmp = if q_lo < 0.0 { q(sigma_lo) } else { q_lo };
        if qtmp < qval {
            sigma = sigma_lo;
        }
    }
    let _ = qval;
    sigma
}

/// Per-norm aggregates feeding [`evaluate_quality_function`].
///
/// All four arrays of pre-reduced complementarity infeasibilities are
/// caller-provided so the evaluator stays pure-scalar:
///
/// * `dual_aggr` — norm of `(grad_lag_x, grad_lag_s)` *before* scaling
///   by `(1 − α_du)`.
/// * `primal_aggr` — norm of `(c, d − s)` before `(1 − α_pri)` scaling.
/// * `compl_aggr` — norm of the four trial-complementarity products
///   `(s_L · z_L, s_U · z_U, σ_L · v_L, σ_U · v_U)` after the σ-step
///   has been applied.
/// * `n_dual`, `n_pri`, `n_comp` — block dimensions used by the
///   `1`-norm and `2`-norm averaging (the `2_squared` and `max`
///   variants do not divide).
#[derive(Debug, Clone, Copy)]
pub struct QualityFunctionAggregates {
    pub dual_aggr: Number,
    pub primal_aggr: Number,
    pub compl_aggr: Number,
    pub n_dual: i32,
    pub n_pri: i32,
    pub n_comp: i32,
}

/// Pure-scalar reducer corresponding to
/// `IpQualityFunctionMuOracle.cpp::CalculateQualityFunction`
/// lines 566-658 minus the vector→aggregate reduction. Combines the
/// caller-provided norm aggregates per the configured `(norm,
/// centrality, balancing)` triple.
///
/// `xi` is the centrality measure of the trial complementarity
/// products; ignored when `centrality == None`.
pub fn evaluate_quality_function(
    norm: NormType,
    centrality: CentralityType,
    balancing: BalancingTermType,
    alpha_primal: Number,
    alpha_dual: Number,
    xi: Number,
    aggr: QualityFunctionAggregates,
) -> Number {
    let (mut dual_inf, mut primal_inf, mut compl_inf) = match norm {
        NormType::OneNorm => {
            let mut d = (1.0 - alpha_dual) * aggr.dual_aggr;
            let mut p = (1.0 - alpha_primal) * aggr.primal_aggr;
            let mut c = aggr.compl_aggr;
            d /= aggr.n_dual as Number;
            if aggr.n_pri > 0 {
                p /= aggr.n_pri as Number;
            }
            debug_assert!(aggr.n_comp > 0);
            c /= aggr.n_comp as Number;
            (d, p, c)
        }
        NormType::TwoNormSquared => {
            // Upstream `IpQualityFunctionMuOracle.cpp:584-595`. The
            // (1−α)² weight and per-n averaging differ from the plain
            // 2-norm branch — and this is the upstream default.
            let mut d = (1.0 - alpha_dual).powi(2) * aggr.dual_aggr;
            let mut p = (1.0 - alpha_primal).powi(2) * aggr.primal_aggr;
            let mut c = aggr.compl_aggr;
            d /= aggr.n_dual as Number;
            if aggr.n_pri > 0 {
                p /= aggr.n_pri as Number;
            }
            debug_assert!(aggr.n_comp > 0);
            c /= aggr.n_comp as Number;
            (d, p, c)
        }
        NormType::MaxNorm => (
            (1.0 - alpha_dual) * aggr.dual_aggr,
            (1.0 - alpha_primal) * aggr.primal_aggr,
            aggr.compl_aggr,
        ),
        NormType::TwoNorm => {
            let mut d = (1.0 - alpha_dual) * aggr.dual_aggr;
            let mut p = (1.0 - alpha_primal) * aggr.primal_aggr;
            let mut c = aggr.compl_aggr;
            d /= (aggr.n_dual as Number).sqrt();
            if aggr.n_pri > 0 {
                p /= (aggr.n_pri as Number).sqrt();
            }
            debug_assert!(aggr.n_comp > 0);
            c /= (aggr.n_comp as Number).sqrt();
            (d, p, c)
        }
    };

    // Repair fp damage from the divisions when the input was already 0.
    if dual_inf.is_nan() {
        dual_inf = 0.0;
    }
    if primal_inf.is_nan() {
        primal_inf = 0.0;
    }
    if compl_inf.is_nan() {
        compl_inf = 0.0;
    }

    let mut q = dual_inf + primal_inf + compl_inf;

    match centrality {
        CentralityType::None => {}
        CentralityType::LogCenter => q -= compl_inf * xi.ln(),
        CentralityType::ReciprocalCenter => q += compl_inf / xi,
        CentralityType::CubedReciprocalCenter => q += compl_inf / xi.powi(3),
    }

    match balancing {
        BalancingTermType::None => {}
        BalancingTermType::CubicTerm => {
            let dom = dual_inf.max(primal_inf) - compl_inf;
            q += dom.max(0.0).powi(3);
        }
    }

    q
}

/// Sigma-bracket selection + golden-section orchestrator. Mirrors
/// `IpQualityFunctionMuOracle.cpp::CalculateMu` lines 329-385.
///
/// `q` is a black-box `q(σ)` evaluator (typically constructed by
/// composing the affine + σ·centering step into a trial point and
/// calling [`evaluate_quality_function`]).
///
/// Returns the σ that approximately minimizes `q` on the picked
/// bracket; the caller then sets `μ_new = σ · avrg_compl` and clamps
/// to `[mu_min, mu_max]`.
#[allow(clippy::too_many_arguments)]
pub fn pick_sigma(
    sigma_min: Number,
    sigma_max: Number,
    mu_min: Number,
    mu_max: Number,
    avrg_compl: Number,
    sigma_tol: Number,
    qf_tol: Number,
    max_steps: i32,
    mut q: impl FnMut(Number) -> Number,
) -> Number {
    let qf_1 = q(1.0);
    let sigma_1minus = 1.0 - sigma_tol.max(1e-4);
    let qf_1minus = q(sigma_1minus);

    if qf_1minus > qf_1 {
        // q decreases for σ > 1 — search up.
        let sigma_up = sigma_max.min(mu_max / avrg_compl);
        let sigma_lo = 1.0;
        if sigma_lo >= sigma_up {
            sigma_up
        } else {
            golden_section(
                sigma_lo, sigma_up, qf_1, -100.0, sigma_tol, qf_tol, max_steps, q,
            )
        }
    } else {
        // q decreases for σ < 1 — search down.
        let sigma_lo = sigma_min.max(mu_min / avrg_compl);
        let sigma_up = sigma_lo.max(sigma_1minus).min(mu_max / avrg_compl);
        if sigma_lo >= sigma_up {
            sigma_lo
        } else {
            golden_section(
                sigma_lo, sigma_up, -100.0, qf_1minus, sigma_tol, qf_tol, max_steps, q,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_section_minimizes_parabola() {
        // q(σ) = (σ − 0.3)²; minimum at σ = 0.3.
        let f = |s: f64| (s - 0.3).powi(2);
        let s = golden_section(0.0, 1.0, f(0.0), f(1.0), 1e-6, 0.0, 50, f);
        assert!((s - 0.3).abs() < 1e-3);
    }

    #[test]
    fn golden_section_respects_max_steps() {
        // Heavy max-step cap should still produce a reasonable σ.
        let f = |s: f64| (s - 0.5).powi(2);
        let s = golden_section(0.0, 1.0, f(0.0), f(1.0), 1e-12, 0.0, 5, f);
        assert!((s - 0.5).abs() < 0.2);
    }

    #[test]
    fn golden_section_handles_monotone() {
        // q monotone increasing → minimum at lo end.
        let f = |s: f64| s;
        let s = golden_section(0.1, 2.0, 0.1, 2.0, 1e-6, 0.0, 50, f);
        assert!(s < 0.2, "got s = {}", s);
    }

    #[test]
    fn golden_section_never_returns_unevaluated_sentinel() {
        // Regression for L4. `pick_sigma` always passes one endpoint with the
        // `-100.0` sentinel as its q-value (search-up → q_up = -100,
        // search-down → q_lo = -100). When every *evaluated* sample is ≤ 0,
        // pounce's added `qmax > 0.0` guard forces `qf_ok = false` on the
        // first pass and drops into the `width_ok && !qf_ok` branch. Before
        // the fix that branch compared the raw q values — including the
        // unevaluated `-100.0` — and returned the sentinel endpoint as the
        // spurious minimum, even though its true quality value is the *worst*
        // of the bracket. The fix re-evaluates any unmoved sentinel endpoint
        // first, mirroring the else-branch and upstream's `if( q_up < 0. )`.
        let sigma_lo = 1.0_f64;
        let sigma_up = 3.0_f64;
        // Negative on the interior/lo points (so qmax ≤ 0) but large and
        // positive exactly at the upper endpoint — the worst place to land.
        let q = move |s: f64| if s == sigma_up { 50.0 } else { -s };
        // search-up style: the upper endpoint carries the -100 sentinel.
        let s = golden_section(sigma_lo, sigma_up, q(sigma_lo), -100.0, 1e-3, 0.0, 50, q);
        assert!(
            s < sigma_up,
            "golden_section returned the unevaluated sentinel endpoint σ = {} \
             (true q there = {}, the bracket maximum); it must re-evaluate the \
             sentinel before selecting a minimum",
            s,
            q(s)
        );
    }

    #[test]
    fn calculate_mu_returns_none_until_plumbed() {
        let mut o = QualityFunctionMuOracle::new();
        assert!(o.calculate_mu().is_none());
    }

    fn aggr(
        d: Number,
        p: Number,
        c: Number,
        nd: i32,
        np: i32,
        nc: i32,
    ) -> QualityFunctionAggregates {
        QualityFunctionAggregates {
            dual_aggr: d,
            primal_aggr: p,
            compl_aggr: c,
            n_dual: nd,
            n_pri: np,
            n_comp: nc,
        }
    }

    #[test]
    fn evaluate_one_norm_averages_by_n() {
        // (1−α_du)*d/n_d + (1−α_pri)*p/n_p + c/n_c.
        let q = evaluate_quality_function(
            NormType::OneNorm,
            CentralityType::None,
            BalancingTermType::None,
            0.5,  // α_pri
            0.25, // α_du
            1.0,
            aggr(8.0, 4.0, 6.0, 4, 2, 3),
        );
        // d = 0.75 * 8 / 4 = 1.5; p = 0.5 * 4 / 2 = 1.0; c = 6/3 = 2.0; total = 4.5
        assert!((q - 4.5).abs() < 1e-12, "got {}", q);
    }

    #[test]
    fn evaluate_max_norm_does_not_divide() {
        let q = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::None,
            BalancingTermType::None,
            0.0,
            0.0,
            1.0,
            aggr(2.0, 3.0, 5.0, 10, 10, 10),
        );
        assert!((q - 10.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_two_norm_divides_by_sqrt_n() {
        let q = evaluate_quality_function(
            NormType::TwoNorm,
            CentralityType::None,
            BalancingTermType::None,
            0.0,
            0.0,
            1.0,
            aggr(2.0, 0.0, 4.0, 4, 0, 16),
        );
        // d = 2/2 = 1.0; p stays 0 (n_pri = 0 → no divide); c = 4/4 = 1.0
        assert!((q - 2.0).abs() < 1e-12, "got {}", q);
    }

    #[test]
    fn evaluate_one_norm_handles_zero_pri_dim() {
        // n_pri = 0 ⇒ primal must not be divided.
        let q = evaluate_quality_function(
            NormType::OneNorm,
            CentralityType::None,
            BalancingTermType::None,
            0.0,
            0.0,
            1.0,
            aggr(0.0, 0.0, 1.0, 1, 0, 1),
        );
        assert!(q.is_finite() && (q - 1.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_log_centrality_subtracts_compl_log_xi() {
        let base = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::None,
            BalancingTermType::None,
            0.0,
            0.0,
            std::f64::consts::E,
            aggr(0.0, 0.0, 4.0, 1, 1, 1),
        );
        let logc = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::LogCenter,
            BalancingTermType::None,
            0.0,
            0.0,
            std::f64::consts::E,
            aggr(0.0, 0.0, 4.0, 1, 1, 1),
        );
        // Difference is −compl_inf · ln(xi) = −4 · 1 = −4.
        assert!((base - logc - 4.0).abs() < 1e-12, "base={base} logc={logc}");
    }

    #[test]
    fn evaluate_reciprocal_centrality_adds_c_over_xi() {
        let q = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::ReciprocalCenter,
            BalancingTermType::None,
            0.0,
            0.0,
            0.5,
            aggr(0.0, 0.0, 1.0, 1, 1, 1),
        );
        // 1.0 + 1.0/0.5 = 3.0.
        assert!((q - 3.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_cubed_reciprocal_centrality_adds_c_over_xi3() {
        let q = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::CubedReciprocalCenter,
            BalancingTermType::None,
            0.0,
            0.0,
            0.5,
            aggr(0.0, 0.0, 1.0, 1, 1, 1),
        );
        // 1.0 + 1.0/0.125 = 9.0.
        assert!((q - 9.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_cubic_balancing_adds_when_dual_dominates() {
        let q = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::None,
            BalancingTermType::CubicTerm,
            0.0,
            0.0,
            1.0,
            aggr(5.0, 1.0, 2.0, 1, 1, 1),
        );
        // base = 5+1+2 = 8; dom = max(5,1) − 2 = 3; +27 → 35.
        assert!((q - 35.0).abs() < 1e-12, "got {}", q);
    }

    #[test]
    fn evaluate_cubic_balancing_zero_when_compl_dominates() {
        let q = evaluate_quality_function(
            NormType::MaxNorm,
            CentralityType::None,
            BalancingTermType::CubicTerm,
            0.0,
            0.0,
            1.0,
            aggr(1.0, 1.0, 5.0, 1, 1, 1),
        );
        // dom = max(1,1) − 5 = −4 → clamped to 0; total = 7.
        assert!((q - 7.0).abs() < 1e-12);
    }

    #[test]
    fn pick_sigma_searches_below_one_for_decreasing_q() {
        // Parabola minimum at σ = 0.4 (well below 1).
        let f = |s: f64| (s - 0.4).powi(2);
        let s = pick_sigma(1e-9, 100.0, 1e-11, 1e5, 1.0, 1e-6, 0.0, 50, f);
        assert!((s - 0.4).abs() < 1e-2, "got s = {}", s);
    }

    #[test]
    fn pick_sigma_searches_above_one_for_q_decreasing_in_sigma() {
        // q decreases as σ grows ⇒ minimum at top of bracket.
        let f = |s: f64| -s;
        let s = pick_sigma(1e-9, 10.0, 1e-11, 1e5, 1.0, 1e-6, 0.0, 50, f);
        // bracket up-end is min(sigma_max=10, mu_max/avrg=1e5) = 10.
        assert!(s > 5.0, "got s = {}", s);
    }

    #[test]
    fn pick_sigma_clamps_to_mu_max_over_avrg_in_up_search() {
        // mu_max/avrg = 2.0 should cap σ_up below sigma_max = 100.
        let f = |s: f64| -s;
        let s = pick_sigma(1e-9, 100.0, 1e-11, 2.0, 1.0, 1e-6, 0.0, 50, f);
        assert!(s <= 2.0 + 1e-9 && s >= 1.0, "got s = {}", s);
    }

    #[test]
    fn pick_sigma_clamps_to_mu_min_over_avrg_in_down_search() {
        // mu_min/avrg = 0.5 must dominate σ_min = 1e-9.
        // q monotone-decreasing toward 0 → search picks low end of bracket.
        let f = |s: f64| s;
        let s = pick_sigma(1e-9, 100.0, 0.5, 1e5, 1.0, 1e-6, 0.0, 50, f);
        assert!(s >= 0.5 - 1e-9 && s <= 1.0, "got s = {}", s);
    }
}
