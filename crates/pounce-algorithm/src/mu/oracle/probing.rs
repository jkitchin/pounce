//! Probing (Mehrotra) mu oracle — port of
//! `IpProbingMuOracle.{hpp,cpp}`. Phase 10.
//!
//! Given the affine-step complementarity `mu_aff` and current `mu`,
//! the probing rule chooses
//!
//! ```text
//!   sigma = min((mu_aff / mu)^3, sigma_max)
//!   mu_new = sigma * mu
//! ```
//!
//! per `IpProbingMuOracle.cpp:117-121` and the standard Mehrotra
//! predictor-corrector formula.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::mu::oracle::r#trait::MuOracle;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::rc::Rc;

pub struct ProbingMuOracle {
    pub sigma_max: Number,
    pub mu_min: Number,
    pub mu_max: Number,
    pub mu_curr: Number,
    pub mu_aff: Number,
}

impl Default for ProbingMuOracle {
    fn default() -> Self {
        Self {
            // Default `sigma_max = 100` per `RegisterOptions`.
            sigma_max: 100.0,
            mu_min: 1e-11,
            mu_max: 1e5,
            mu_curr: 1.0,
            mu_aff: 1.0,
        }
    }
}

impl ProbingMuOracle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pure-arithmetic probing formula from `IpProbingMuOracle.cpp:117`.
    pub fn probing_mu(mu_curr: Number, mu_aff: Number, sigma_max: Number) -> Number {
        let sigma = (mu_aff / mu_curr).powi(3).min(sigma_max);
        sigma * mu_curr
    }

    /// Drive the affine (predictor) step through `pd_search_dir`,
    /// snapshot `μ_aff` from the projected complementarity, then return
    /// the probed `μ_new ∈ [mu_min, mu_max]`. Returns `None` if the
    /// affine linear solve fails (caller falls back to LOQO).
    ///
    /// `tau` is the fraction-to-the-boundary parameter (typically
    /// `1.0` for the affine probe per upstream's
    /// `IpProbingMuOracle.cpp:107`).
    pub fn calculate_mu_with_affine_step(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        pd_search_dir: &mut PdSearchDirCalc,
        tau: Number,
    ) -> Option<Number> {
        if !pd_search_dir.compute_affine_step(data, cq, nlp) {
            return None;
        }
        let delta_aff: IteratesVector = data.borrow().delta_aff.clone()?;
        let cq_ref = cq.borrow();
        // Upstream `IpProbingMuOracle.cpp:111` uses `curr_avrg_compl()`
        // (the actual average s·z at the current iterate), NOT the
        // stored barrier parameter `data.curr_mu`. Mehrotra's rule
        // sigma = (mu_aff / mu_curr)^3 only makes sense when mu_curr
        // is the present complementarity — using the barrier value
        // produces sigma values orders of magnitude off and causes
        // huge over-correction steps on iter 1 of LP-shaped problems.
        let mu_curr = cq_ref.curr_avrg_compl();
        let alpha_pri = cq_ref.aff_step_alpha_primal_max(&delta_aff, tau);
        let alpha_du = cq_ref.aff_step_alpha_dual_max(&delta_aff, tau);
        let mu_aff = cq_ref.aff_step_compl_avrg(&delta_aff, alpha_pri, alpha_du);
        self.mu_curr = mu_curr;
        self.mu_aff = mu_aff;
        let raw = Self::probing_mu(mu_curr, mu_aff, self.sigma_max);
        Some(raw.clamp(self.mu_min, self.mu_max))
    }
}

impl MuOracle for ProbingMuOracle {
    fn calculate_mu(&mut self) -> Option<Number> {
        let raw = Self::probing_mu(self.mu_curr, self.mu_aff, self.sigma_max);
        Some(raw.clamp(self.mu_min, self.mu_max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probing_aff_equals_curr_keeps_mu() {
        // ratio=1 → sigma=1 → mu_new=mu_curr.
        assert_eq!(ProbingMuOracle::probing_mu(1.0, 1.0, 100.0), 1.0);
    }

    #[test]
    fn probing_aff_half_curr() {
        // ratio=0.5 → sigma=0.125 → mu_new=0.125 * 1.0.
        let m = ProbingMuOracle::probing_mu(1.0, 0.5, 100.0);
        assert!((m - 0.125).abs() < 1e-15);
    }

    #[test]
    fn probing_caps_at_sigma_max() {
        // ratio=10 → sigma=1000, capped at 100. mu_new=100.
        let m = ProbingMuOracle::probing_mu(1.0, 10.0, 100.0);
        assert!((m - 100.0).abs() < 1e-13);
    }

    #[test]
    fn calculate_mu_via_trait_clamped() {
        let mut o = ProbingMuOracle {
            sigma_max: 100.0,
            mu_min: 0.5,
            mu_max: 10.0,
            mu_curr: 1.0,
            mu_aff: 0.001, // sigma ≈ 1e-9, mu_new ≈ 1e-9, clamps to 0.5.
        };
        assert_eq!(o.calculate_mu(), Some(0.5));
    }
}
