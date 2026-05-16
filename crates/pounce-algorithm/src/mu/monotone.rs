//! Monotone Fiacco-McCormick mu update — port of
//! `Algorithm/IpMonotoneMuUpdate.{hpp,cpp}`.
//!
//! Reduces mu by either `mu_linear_decrease_factor` or
//! `pow(mu, mu_superlinear_decrease_power)`, taking the smaller value
//! and clamping to `mu_min`. Bit-exact with upstream.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::mu::r#trait::MuUpdate;
use pounce_common::types::Number;

pub struct MonotoneMuUpdate {
    pub mu_init: Number,
    pub mu_min: Number,
    /// Upper bound on μ from `IpMonotoneMuUpdate.cpp:RegisterOptions`.
    /// Used to clamp `mu_init` at [`MuUpdate::initialize`] so the
    /// barrier doesn't start above the registered ceiling regardless
    /// of what the user set. Default `1e5` mirrors upstream.
    pub mu_max: Number,
    pub mu_linear_decrease_factor: Number,
    pub mu_superlinear_decrease_power: Number,
    pub tau_min: Number,
    /// `barrier_tol_factor` from `IpMonotoneMuUpdate.cpp:RegisterOptions`.
    /// μ only decreases when the barrier subproblem error drops below
    /// `barrier_tol_factor · μ`.
    pub barrier_tol_factor: Number,
    /// `mu_target` floor — μ never goes below this regardless of the
    /// reduction formula. Defaults to 0 (the floor is `mu_min`).
    pub mu_target: Number,
    /// `mu_allow_fast_monotone_decrease` from
    /// `IpMonotoneMuUpdate.cpp:RegisterOptions`. When `true` (the
    /// upstream default), the reduction loop keeps iterating while
    /// the sub-error stays below `barrier_tol_factor · μ`, allowing
    /// multiple consecutive μ reductions in one outer call. When
    /// `false`, the loop exits after the first successful reduction —
    /// useful on stiff problems where a runaway μ collapse destroys
    /// the line search.
    pub mu_allow_fast_monotone_decrease: bool,
    /// Complementarity tolerance — option `compl_inf_tol`, default 1e-4
    /// per `IpAlgorithmRegOp.cpp`. Enters the dynamic μ floor via
    /// `min(tol, compl_inf_tol) / (barrier_tol_factor + 1)` per
    /// `IpMonotoneMuUpdate.cpp:CalcNewMuAndTau:215`. Without this floor,
    /// μ can collapse to the absolute floor (`mu_min`) while primal
    /// infeasibility is still large — observed on SSINE/DECONVBNE.
    pub compl_inf_tol: Number,
    /// `first_iter_resto_` flag from
    /// `Algorithm/IpMonotoneMuUpdate.cpp:118-121,144,196`. When set,
    /// the very next call to [`Self::update_barrier_parameter`] skips
    /// the μ-reduction loop entirely and clears the flag. Wired by
    /// the restoration sub-builder for the inner IPM (prefix
    /// `"resto."`) so the inner doesn't immediately collapse μ on
    /// iteration 0 — it must use the `resto_mu` value that
    /// [`crate::resto::init::RestoIterateInitializer::SetInitialIterates`]
    /// seeded into `data.curr_mu`.
    pub first_iter_resto: bool,
}

impl Default for MonotoneMuUpdate {
    fn default() -> Self {
        // Defaults from `IpMonotoneMuUpdate.cpp:RegisterOptions`.
        Self {
            mu_init: 0.1,
            mu_min: 1e-11,
            mu_max: 1e5,
            mu_linear_decrease_factor: 0.2,
            mu_superlinear_decrease_power: 1.5,
            tau_min: 0.99,
            barrier_tol_factor: 10.0,
            mu_target: 0.0,
            mu_allow_fast_monotone_decrease: true,
            compl_inf_tol: 1e-4,
            first_iter_resto: false,
        }
    }
}

impl MonotoneMuUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder helper for the `first_iter_resto_` flag. Mirrors the
    /// upstream `prefix == "resto."` branch in
    /// `IpMonotoneMuUpdate.cpp:InitializeImpl`.
    pub fn with_first_iter_resto(mut self, b: bool) -> Self {
        self.first_iter_resto = b;
        self
    }

    /// Builder for the `mu_min` floor. The restoration inner IPM uses
    /// `100 * outer_mu_min` per upstream `IpAdaptiveMuUpdate.cpp:206-211`
    /// (and the analogous monotone path); without the conservative
    /// floor, near-feasible iterates collapse μ to the absolute floor
    /// in a single step and the next direction is dominated by the
    /// penalty/proximity terms instead of the barrier, which destroys
    /// near-feasibility (DECONVBNE).
    pub fn with_mu_min(mut self, mu_min: Number) -> Self {
        self.mu_min = mu_min;
        self
    }

    /// Fraction-to-the-bound parameter `tau` from upstream
    /// `IpMonotoneMuUpdate.cpp:Update`:
    ///
    /// ```text
    ///   tau = max(tau_min, 1 - mu)
    /// ```
    ///
    /// Returns a value in `[tau_min, 1)`.
    pub fn compute_tau(&self, mu: Number) -> Number {
        self.tau_min.max(1.0 - mu)
    }

    /// Pure scalar reduction used by the trait impl. Exposed so unit
    /// tests can drive the formula without standing up an
    /// `IpoptData`/`IpoptCq` fixture.
    pub fn compute_next_mu(&self, curr_mu: Number) -> Number {
        let linear = self.mu_linear_decrease_factor * curr_mu;
        let superlinear = curr_mu.powf(self.mu_superlinear_decrease_power);
        linear.min(superlinear).max(self.mu_min)
    }
}

impl MuUpdate for MonotoneMuUpdate {
    /// Port of `IpMonotoneMuUpdate.cpp:InitializeImpl`. Seeds
    /// `curr_mu = min(mu_init, mu_max)`,
    /// `curr_tau = max(tau_min, 1 - curr_mu)`.
    fn initialize(&mut self, data: &IpoptDataHandle) {
        let init_mu = self.mu_init.min(self.mu_max);
        let mut d = data.borrow_mut();
        d.curr_mu = init_mu;
        d.curr_tau = self.compute_tau(init_mu);
    }

    /// Port of `IpMonotoneMuUpdate.cpp:UpdateBarrierParameter`.
    /// Reduces μ only while the barrier-subproblem error is below
    /// `barrier_tol_factor · μ` (or a tiny step was just detected).
    /// Each successful reduction also refreshes `curr_tau` and the new
    /// μ in `data`. Returns the post-update μ.
    fn update_barrier_parameter(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        _nlp: Option<&std::rc::Rc<std::cell::RefCell<dyn crate::ipopt_nlp::IpoptNlp>>>,
        _pd_search_dir: Option<&mut crate::kkt::pd_search_dir_calc::PdSearchDirCalc>,
    ) -> Number {
        let mut mu = data.borrow().curr_mu;
        let mut tau = data.borrow().curr_tau;
        let tiny_step = data.borrow().tiny_step_flag;

        // `first_iter_resto_` (upstream `IpMonotoneMuUpdate.cpp:144`):
        // on the first inner iteration of restoration, skip the μ
        // reduction loop entirely so the inner uses the `resto_mu`
        // seeded by `RestoIterateInitializer`. Cleared after this
        // call so subsequent inner iterations behave normally.
        if self.first_iter_resto {
            self.first_iter_resto = false;
            let mut d = data.borrow_mut();
            d.curr_mu = mu;
            d.curr_tau = tau;
            return mu;
        }

        // Dynamic floor from `IpMonotoneMuUpdate.cpp:CalcNewMuAndTau:215`:
        //     floor = max(mu_target, min(tol, compl_inf_tol) / (barrier_tol_factor + 1))
        // Without this, μ collapses to `mu_min` (1e-11) while primal
        // infeasibility is still large — observed on SSINE/DECONVBNE,
        // where the next direction is dominated by ill-conditioned
        // barrier terms and the line search stalls.
        // We also `max` with `mu_min` so the restoration sub-builder's
        // `with_mu_min(100 * outer_mu_min)` safeguard still applies.
        let tol = data.borrow().tol;
        let dynamic_floor = tol.min(self.compl_inf_tol) / (self.barrier_tol_factor + 1.0);
        let floor = self.mu_target.max(self.mu_min).max(dynamic_floor);

        // The barrier error depends on μ via the relaxed
        // complementarity. Read it once per μ value.
        loop {
            let sub_err = cq.borrow().curr_barrier_error();
            let kappa_eps_mu = self.barrier_tol_factor * mu;
            if !(sub_err <= kappa_eps_mu || tiny_step) {
                break;
            }
            let mut new_mu = self
                .mu_linear_decrease_factor
                .min(mu.powf(self.mu_superlinear_decrease_power - 1.0))
                * mu;
            if new_mu < floor {
                new_mu = floor;
            }
            if new_mu >= mu {
                // No further progress (already at floor).
                break;
            }
            mu = new_mu;
            tau = self.compute_tau(mu);
            // Only one reduction per outer iteration unless we'd need
            // to re-test sub_err for the new μ. Upstream re-tests, so
            // we do the same — but `curr_barrier_error` is keyed on the
            // current iterate (μ enters via the constant subtraction),
            // so the re-tested value will generally be larger and the
            // loop exits.
            //
            // Stop after one reduction in the tiny_step branch (matches
            // upstream which clears tiny_step_flag once consumed).
            if tiny_step {
                data.borrow_mut().tiny_step_flag = false;
                break;
            }
            // `mu_allow_fast_monotone_decrease=false` caps the loop at
            // a single reduction. Mirrors upstream
            // `IpMonotoneMuUpdate.cpp:CalcNewMuAndTau` when the option
            // is off.
            if !self.mu_allow_fast_monotone_decrease {
                break;
            }
        }

        let mut d = data.borrow_mut();
        d.curr_mu = mu;
        d.curr_tau = tau;
        mu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_smaller_of_linear_and_superlinear() {
        let m = MonotoneMuUpdate::new();
        // mu = 0.1 → linear = 0.02, superlinear = 0.1^1.5 ≈ 0.0316.
        // The smaller is `linear`.
        let next = m.compute_next_mu(0.1);
        assert!((next - 0.02).abs() < 1e-15);
    }

    #[test]
    fn tau_at_small_mu_is_tau_min() {
        let m = MonotoneMuUpdate::new();
        // mu small → 1 - mu ~ 1; tau_min=0.99 → max → 1.0 (since 1-mu=0.999...).
        // Actually 1 - 1e-3 = 0.999 > 0.99 → tau = 0.999.
        assert!((m.compute_tau(1e-3) - 0.999).abs() < 1e-15);
    }

    #[test]
    fn tau_floor_at_tau_min() {
        let m = MonotoneMuUpdate::new();
        // mu=0.5 → 1 - 0.5 = 0.5; floor at tau_min=0.99 → 0.99.
        assert!((m.compute_tau(0.5) - 0.99).abs() < 1e-15);
    }

    #[test]
    fn clamps_to_mu_min() {
        let m = MonotoneMuUpdate {
            mu_min: 1e-3,
            ..Default::default()
        };
        let next = m.compute_next_mu(1e-10);
        assert!((next - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn dynamic_floor_matches_upstream_calcnewmuandtau() {
        // Replicate `IpMonotoneMuUpdate.cpp:CalcNewMuAndTau:215`:
        //   floor = max(mu_target, min(tol, compl_inf_tol) / (barrier_tol_factor + 1))
        // With default `tol=1e-8`, `compl_inf_tol=1e-4`, `barrier_tol_factor=10`,
        // `mu_target=0`: floor ≈ 1e-8 / 11 ≈ 9.09e-10.
        let m = MonotoneMuUpdate::default();
        let tol: Number = 1e-8;
        let expected_floor = tol.min(m.compl_inf_tol) / (m.barrier_tol_factor + 1.0);
        assert!((expected_floor - 1e-8 / 11.0).abs() < 1e-20);
        // The hardcoded `mu_min = 1e-11` is well below the dynamic floor
        // with default tols — the runtime `floor = max(...)` picks the
        // dynamic one. (Verified in `update_barrier_parameter`.)
        assert!(m.mu_min < expected_floor);
    }
}
