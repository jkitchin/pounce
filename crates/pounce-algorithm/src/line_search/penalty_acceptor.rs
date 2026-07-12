//! Penalty line-search acceptor — port of `IpPenaltyLSAcceptor.{hpp,cpp}`.
//!
//! Phase 10. Backs `line_search_method = penalty`. Maintains a penalty
//! parameter `ν` that's bumped up whenever the predicted reduction
//! would otherwise be non-monotone:
//!
//! ```text
//!   ν⁺ = (∇φᵀ δ + ½ δᵀ W δ) / ((1 − ρ) · θ)
//!   if ν < ν⁺ then ν ← ν⁺ + ν_inc
//! ```
//!
//! Acceptance test (Armijo on the penalty merit `M = φ + ν · θ`,
//! upstream `IpPenaltyLSAcceptor.cpp:CheckAcceptabilityOfTrialPoint`):
//!
//! ```text
//!   pred(α) = − α · ∇φᵀδ − ½ α² · δᵀWδ + ν · (θ − θ₂(α))
//!   ared(α) = (φ_ref + ν · θ_ref) − (φ_trial + ν · θ_trial)
//!   accept iff Compare_le(η · pred, ared, |φ_ref + ν · θ_ref|)
//! ```
//!
//! where `θ₂(α)` is the 1-norm of the *linearised* constraint
//! infeasibility at the predicted step:
//!
//! ```text
//!   θ₂(α) = ‖c(x) + α · J_c · δx‖₁ + ‖d(x) − s + α · (J_d · δx − δs)‖₁
//! ```
//!
//! `init_this_line_search` (driven by the backtracking driver before
//! the α-loop) snapshots the reference state and the
//! linearisation vectors, then runs `update_nu`. `check_trial_point`
//! reads the snapshot to compute pred/ared per α — matching upstream
//! lines 188-247.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::iterates_vector::IteratesVector;
use crate::line_search::filter_acceptor::AcceptDecision;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use pounce_common::types::Number;
use pounce_common::utils::compare_le;
use pounce_linalg::Vector;
use std::rc::Rc;

pub struct PenaltyLsAcceptor {
    /// Convex-combination weight ρ in upstream's update rule.
    /// Default `0.1` per `IpPenaltyLSAcceptor.cpp:RegisterOptions`.
    pub rho: Number,
    /// Increment added when ν is bumped.
    pub nu_inc: Number,
    /// Initial value of ν.
    pub nu_init: Number,
    /// Max ν before declaring failure.
    pub nu_max: Number,
    /// Sufficient-decrease parameter η.
    pub eta_penalty: Number,
    nu: Number,
    last_nu: Number,
    /// Cached reference state — set by `init_this_line_search`,
    /// consumed by `check_trial_point`.
    cache: Option<RefCache>,
}

/// Reference-iterate snapshot needed by the pred/ared test.
struct RefCache {
    theta_ref: Number,
    barr_ref: Number,
    grad_barr_t_delta: Number,
    dwd: Number,
    /// `c(x)` at the reference iterate.
    c_ref: Rc<dyn Vector>,
    /// `d(x) − s` at the reference iterate.
    d_minus_s_ref: Rc<dyn Vector>,
    /// `J_c · δx`.
    jac_c_delta: Rc<dyn Vector>,
    /// `J_d · δx − δs`.
    jac_d_delta_minus_ds: Rc<dyn Vector>,
}

impl Default for PenaltyLsAcceptor {
    fn default() -> Self {
        Self {
            rho: 0.1,
            nu_inc: 1e-4,
            nu_init: 1e-6,
            nu_max: 1e40,
            eta_penalty: 1e-8,
            nu: 1e-6,
            last_nu: 1e-6,
            cache: None,
        }
    }
}

impl PenaltyLsAcceptor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn nu(&self) -> Number {
        self.nu
    }

    pub fn last_nu(&self) -> Number {
        self.last_nu
    }

    /// Reset to the initial ν. Called when the line search activates a
    /// new outer iteration.
    pub fn reset(&mut self) {
        self.nu = self.nu_init;
        self.last_nu = self.nu_init;
        self.cache = None;
    }

    /// Scalar core of `IpPenaltyLSAcceptor.cpp:148-157`:
    /// ```text
    ///   if reference_theta > 0:
    ///       ν⁺ = (gradBarrᵀδ + ½ δᵀWδ) / ((1 − ρ) · θ)
    ///       if ν < ν⁺ then ν ← ν⁺ + ν_inc
    /// ```
    /// `last_nu` snapshots `ν` *before* the bump, matching upstream's
    /// `last_nu_ = nu_`.
    pub fn update_nu(
        &mut self,
        grad_barr_t_delta: Number,
        delta_w_delta: Number,
        reference_theta: Number,
    ) {
        self.last_nu = self.nu;
        if reference_theta > 0.0 {
            let nu_plus =
                (grad_barr_t_delta + 0.5 * delta_w_delta) / ((1.0 - self.rho) * reference_theta);
            if self.nu < nu_plus {
                self.nu = nu_plus + self.nu_inc;
            }
        }
    }

    /// `pred(α)` from cached reference state. Upstream
    /// `IpPenaltyLSAcceptor.cpp:CalcPred` lines 169-198. Returns 0 if
    /// the closed-form value is negative.
    fn calc_pred(&self, alpha: Number) -> Number {
        let cache = self
            .cache
            .as_ref()
            .expect("calc_pred called before init_this_line_search");
        // theta_2(α) = ‖c + α·J_c·δx‖₁ + ‖d−s + α·(J_d·δx − δs)‖₁.
        let mut tmp_c = cache.c_ref.make_new();
        tmp_c.set(0.0);
        tmp_c.add_two_vectors(1.0, &*cache.c_ref, alpha, &*cache.jac_c_delta, 0.0);
        let mut tmp_d = cache.d_minus_s_ref.make_new();
        tmp_d.set(0.0);
        tmp_d.add_two_vectors(
            1.0,
            &*cache.d_minus_s_ref,
            alpha,
            &*cache.jac_d_delta_minus_ds,
            0.0,
        );
        let theta_2 = tmp_c.asum() + tmp_d.asum();

        let pred = -alpha * cache.grad_barr_t_delta - 0.5 * alpha * alpha * cache.dwd
            + self.nu * (cache.theta_ref - theta_2);
        if pred < 0.0 { 0.0 } else { pred }
    }
}

impl BacktrackingLsAcceptor for PenaltyLsAcceptor {
    fn reset(&mut self) {
        PenaltyLsAcceptor::reset(self);
    }

    /// Snapshot reference state and bump ν once per outer iteration.
    /// Mirrors upstream `IpPenaltyLSAcceptor.cpp:InitThisLineSearch`
    /// lines 87-167 (non-watchdog branch).
    fn init_this_line_search(
        &mut self,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
    ) {
        let cqr = cq.borrow();
        let theta_ref = cqr.curr_constraint_violation();
        let barr_ref = cqr.curr_barrier_obj();
        let grad_barr_t_delta = cqr.curr_grad_barr_t_delta(&*delta.x, &*delta.s);
        let dwd = cqr.curr_dwd(&*delta.x, &*delta.s);

        // Linearisation vectors.
        let c_ref = cqr.curr_c();
        let d_minus_s_ref = cqr.curr_d_minus_s();
        let jac_c_delta = cqr.curr_jac_c_times_vec(&*delta.x);
        // jac_d_delta_minus_ds = J_d · δx − δs.
        let jac_d_delta = cqr.curr_jac_d_times_vec(&*delta.x);
        let mut tmp = jac_d_delta.make_new();
        tmp.set(0.0);
        tmp.add_two_vectors(1.0, &*jac_d_delta, -1.0, &*delta.s, 0.0);
        let jac_d_delta_minus_ds: Rc<dyn Vector> = Rc::from(tmp);
        drop(cqr);

        self.cache = Some(RefCache {
            theta_ref,
            barr_ref,
            grad_barr_t_delta,
            dwd,
            c_ref,
            d_minus_s_ref,
            jac_c_delta,
            jac_d_delta_minus_ds,
        });

        // ν bump per `IpPenaltyLSAcceptor.cpp:148-157`.
        self.update_nu(grad_barr_t_delta, dwd, theta_ref);
    }

    /// Sufficient-decrease test on the penalty merit
    /// `M(x; ν) = φ + ν · θ`. Port of
    /// `IpPenaltyLSAcceptor.cpp:CheckAcceptabilityOfTrialPoint` lines
    /// 188-247:
    ///
    /// ```text
    ///   pred = −α·∇φᵀδ − ½ α²·δᵀWδ + ν·(θ_ref − θ₂(α))
    ///   ared = (φ_ref + ν·θ_ref) − (φ_trial + ν·θ_trial)
    ///   accept iff Compare_le(η·pred, ared, |φ_ref + ν·θ_ref|)
    /// ```
    ///
    /// The relaxed `≤` mirrors upstream's `Compare_le` ε-tolerance.
    /// `Reject` falls through to the driver's α-reduction step.
    fn check_trial_point(
        &mut self,
        alpha_primal: Number,
        _theta: Number,
        _phi: Number,
        _d_phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> AcceptDecision {
        // Without a fresh `init_this_line_search` snapshot we degenerate
        // to "always accept" — the driver's reset path triggers this on
        // the very first iteration before the acceptor has been wired.
        let cache = match &self.cache {
            Some(c) => c,
            None => return AcceptDecision::Accept,
        };

        let pred = self.calc_pred(alpha_primal);
        let ref_merit = cache.barr_ref + self.nu * cache.theta_ref;
        let ared = ref_merit - (phi_trial + self.nu * theta_trial);

        if compare_le(self.eta_penalty * pred, ared, ref_merit.abs()) {
            AcceptDecision::Accept
        } else {
            AcceptDecision::Reject
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_bump_when_theta_zero() {
        let mut a = PenaltyLsAcceptor::new();
        let nu0 = a.nu();
        a.update_nu(10.0, 5.0, 0.0);
        assert_eq!(a.nu(), nu0);
        assert_eq!(a.last_nu(), nu0);
    }

    #[test]
    fn bump_when_nu_plus_exceeds_current() {
        let mut a = PenaltyLsAcceptor {
            rho: 0.1,
            nu_inc: 1e-4,
            nu: 0.0,
            last_nu: 0.0,
            ..Default::default()
        };
        // grad·δ + 0.5·δWδ = 1 + 0 = 1
        // θ = 1; (1 − 0.1)·θ = 0.9 → ν⁺ ≈ 1.111…
        a.update_nu(1.0, 0.0, 1.0);
        assert!(a.last_nu() == 0.0);
        let expected = 1.0 / 0.9 + 1e-4;
        assert!((a.nu() - expected).abs() < 1e-12);
    }

    #[test]
    fn no_bump_when_already_above_nu_plus() {
        let mut a = PenaltyLsAcceptor {
            rho: 0.1,
            nu_inc: 1e-4,
            nu: 1e6,
            last_nu: 1e6,
            ..Default::default()
        };
        a.update_nu(1.0, 0.0, 1.0);
        assert_eq!(a.nu(), 1e6);
    }

    #[test]
    fn reset_restores_init() {
        let mut a = PenaltyLsAcceptor::new();
        a.update_nu(10.0, 0.0, 1.0); // bumps ν
        let bumped = a.nu();
        assert!(bumped > a.nu_init);
        PenaltyLsAcceptor::reset(&mut a);
        assert_eq!(a.nu(), a.nu_init);
    }

    #[test]
    fn check_trial_point_without_cache_accepts() {
        // Driver-init path not yet exercised → fall-through accept.
        let mut a = PenaltyLsAcceptor::new();
        assert_eq!(
            a.check_trial_point(1.0, 1.0, 10.0, -1.0, 0.5, 8.0),
            AcceptDecision::Accept
        );
    }

    /// Hand-built cache lets us exercise `calc_pred` and the
    /// pred/ared decision without spinning up an IpoptCq.
    fn cache_for_test(
        theta_ref: Number,
        barr_ref: Number,
        grad_barr_t_delta: Number,
        dwd: Number,
        c_ref: Vec<Number>,
        d_minus_s_ref: Vec<Number>,
        jac_c_delta: Vec<Number>,
        jac_d_delta_minus_ds: Vec<Number>,
    ) -> RefCache {
        use pounce_linalg::Vector;
        use pounce_linalg::dense_vector::DenseVectorSpace;
        let mkr = |v: Vec<Number>| -> Rc<dyn Vector> {
            let mut x = DenseVectorSpace::new(v.len() as i32).make_new_dense();
            x.values_mut().copy_from_slice(&v);
            Rc::new(x)
        };
        RefCache {
            theta_ref,
            barr_ref,
            grad_barr_t_delta,
            dwd,
            c_ref: mkr(c_ref),
            d_minus_s_ref: mkr(d_minus_s_ref),
            jac_c_delta: mkr(jac_c_delta),
            jac_d_delta_minus_ds: mkr(jac_d_delta_minus_ds),
        }
    }

    #[test]
    fn calc_pred_matches_closed_form() {
        // gradBarrᵀδ = 2; dWd = 4; ν = 0.5; θ_ref = 3.
        // c = (1, 2); J_c·δx = (-1, -1) → c+α·J_c·δ at α=0.5 = (0.5, 1.5); ‖·‖₁ = 2.0.
        // d−s = (4); J_d·δx − δs = (-2) → at α=0.5 = (3); ‖·‖₁ = 3.0.
        // θ₂(0.5) = 5.0.
        // pred = −0.5·2 − 0.5·0.25·4 + 0.5·(3 − 5) = −1 − 0.5 − 1 = −2.5 → clamps to 0.
        let mut a = PenaltyLsAcceptor::new();
        a.nu = 0.5;
        a.cache = Some(cache_for_test(
            3.0,
            0.0,
            2.0,
            4.0,
            vec![1.0, 2.0],
            vec![4.0],
            vec![-1.0, -1.0],
            vec![-2.0],
        ));
        assert!((a.calc_pred(0.5) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn calc_pred_positive_when_directions_align() {
        // gradBarrᵀδ = -2 (descent); dWd = 0; ν = 1; θ_ref = 3.
        // J_c·δx = -c → at α=1, c+J_c·δ = 0 ⇒ θ₂ = 0.
        // pred = −1·(−2) − 0 + 1·(3 − 0) = 2 + 3 = 5.
        let mut a = PenaltyLsAcceptor::new();
        a.nu = 1.0;
        a.cache = Some(cache_for_test(
            3.0,
            0.0,
            -2.0,
            0.0,
            vec![1.0, 2.0],
            vec![0.0],
            vec![-1.0, -2.0],
            vec![0.0],
        ));
        assert!((a.calc_pred(1.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn check_trial_point_accepts_when_ared_meets_pred() {
        // Reuse the descent setup: pred(1) = 5, η = 0.5 ⇒ η·pred = 2.5.
        // φ_ref = 0; ν·θ_ref = 3 → ref_merit = 3.
        // φ_trial = -3; ν·θ_trial = 0 → ared = 3 − (−3) = 6 ≥ 2.5 ⇒ Accept.
        let mut a = PenaltyLsAcceptor::new();
        a.nu = 1.0;
        a.eta_penalty = 0.5;
        a.cache = Some(cache_for_test(
            3.0,
            0.0,
            -2.0,
            0.0,
            vec![1.0, 2.0],
            vec![0.0],
            vec![-1.0, -2.0],
            vec![0.0],
        ));
        assert_eq!(
            a.check_trial_point(1.0, 3.0, 0.0, -2.0, 0.0, -3.0),
            AcceptDecision::Accept
        );
    }

    #[test]
    fn check_trial_point_rejects_insufficient_decrease() {
        // Same descent setup, but trial has barely any improvement.
        // ref_merit = 3; φ_trial + ν·θ_trial = 2.999 ⇒ ared ≈ 0.001 < η·pred = 2.5.
        let mut a = PenaltyLsAcceptor::new();
        a.nu = 1.0;
        a.eta_penalty = 0.5;
        a.cache = Some(cache_for_test(
            3.0,
            0.0,
            -2.0,
            0.0,
            vec![1.0, 2.0],
            vec![0.0],
            vec![-1.0, -2.0],
            vec![0.0],
        ));
        assert_eq!(
            a.check_trial_point(1.0, 3.0, 0.0, -2.0, 2.999, 0.0),
            AcceptDecision::Reject
        );
    }
}
