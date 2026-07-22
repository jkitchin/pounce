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

    /// `compl_inf_tol` expressed in the **internally scaled** space that μ
    /// lives in (pounce#257).
    ///
    /// The dynamic μ floor exists so the barrier stops just below the accuracy
    /// the convergence test demands, and its two terms are enforced in
    /// *different spaces*. `tol` is compared against the scaled NLP error, so
    /// it needs no conversion. `compl_inf_tol` is compared against the
    /// **unscaled** complementarity (`IpOptErrorConvCheck.cpp`; pounce#173),
    /// which is the scaled complementarity divided by the objective scaling
    /// factor — so `compl_inf_tol` in *scaled* units is
    /// `compl_inf_tol · |obj_scaling_factor|`.
    ///
    /// Taking the raw value put the floor `1/|df|` too high whenever the
    /// objective was scaled down. On jit1's branch-and-bound node subproblems
    /// (`df = 1e-5`, `tol = 1e-7`) μ bottomed out at `9.09e-9`, leaving an
    /// unscaled complementarity of `9.09e-4` — a hard 9× over `compl_inf_tol`
    /// that no further iteration could clear, since μ was already at its floor.
    /// The iterate sat *at* the optimum with a scaled NLP error 10× under
    /// `tol`, yet the strict certificate was unreachable; μ-at-floor plus the
    /// vanishing step then exited `STOP_AT_TINY_STEP`
    /// (`Search_Direction_Becomes_Too_Small`), which callers read as
    /// unboundedness. Converting the tolerance into μ's own space lets the
    /// barrier descend far enough for the certificate to be issued.
    ///
    /// The factor is signed (`obj_scaling_factor = -1` poses a maximization),
    /// so take its magnitude, and fall back to the unconverted tolerance when
    /// it is absent or degenerate — a floor that is too low only costs
    /// iterations, whereas one that is too high costs the certificate.
    pub fn scaled_compl_inf_tol(&self, obj_scaling_factor: Number) -> Number {
        let df = obj_scaling_factor.abs();
        if df.is_finite() && df > 0.0 {
            self.compl_inf_tol * df
        } else {
            self.compl_inf_tol
        }
    }

    /// `mu_min` capped so it can never block the termination certificate
    /// (pounce#266) — the companion of [`Self::scaled_compl_inf_tol`].
    ///
    /// #258 converted the *dynamic* term of the barrier floor into μ's scaled
    /// space, but the floor has a second, independent term: `mu_min`, a raw
    /// absolute constant (default `1e-11`) that also lives in scaled space.
    /// Once `compl_inf_tol·|df|/(barrier_tol_factor+1) < mu_min` — i.e.
    /// `|df|` below `≈ mu_min·(barrier_tol_factor+1)/compl_inf_tol` — the
    /// converted term stops mattering, μ bottoms out at `mu_min`, and the
    /// unscaled complementarity is pinned at `mu_min/|df| > compl_inf_tol`:
    /// the certificate is unreachable no matter how long the solve runs, and
    /// μ-at-floor plus the vanishing step exits `STOP_AT_TINY_STEP` on an
    /// iterate that is *at* the optimum (HS71 × 1e8, `df = 8.3e-8`).
    ///
    /// Upstream's monotone floor (`IpMonotoneMuUpdate.cpp:CalcNewMuAndTau`)
    /// has no `mu_min` term at all — pounce added it so the restoration
    /// sub-builder's `with_mu_min(100 * outer_mu_min)` safeguard applies —
    /// which is why Ipopt certifies these files even with `mu_min=1e-11`
    /// forced. Capping at `scaled_compl_inf_tol / (barrier_tol_factor + 1)`
    /// keeps `mu_min` inert exactly when it would cost the certificate, with
    /// the same headroom the dynamic floor reserves (μ then bottoms out where
    /// Ipopt's does: `7.58e-13` on HS71 × 1e8). A floor that is too low only
    /// costs iterations; one that is too high costs the certificate.
    ///
    /// The restoration safeguard is unaffected: `RestoIpoptNlp` does not
    /// override `obj_scaling_factor`, so the inner IPM sees `df = 1` and the
    /// cap (`compl_inf_tol/(barrier_tol_factor+1) ≈ 9e-6` at defaults) sits
    /// far above `100 · mu_min`.
    pub fn certificate_safe_mu_min(&self, obj_scaling_factor: Number) -> Number {
        self.mu_min
            .min(self.scaled_compl_inf_tol(obj_scaling_factor) / (self.barrier_tol_factor + 1.0))
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
    /// Monotone μ throws `TINY_STEP_DETECTED` when a tiny step is
    /// flagged and μ is already at its floor — see
    /// `IpMonotoneMuUpdate.cpp`. The main loop realises that throw as a
    /// `STOP_AT_TINY_STEP` termination.
    fn terminates_on_tiny_step(&self) -> bool {
        true
    }

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
        // `with_mu_min(100 * outer_mu_min)` safeguard still applies —
        // but capped by `certificate_safe_mu_min` (pounce#266): both
        // floor terms live in μ's scaled space, and an uncapped
        // absolute `mu_min` re-creates exactly the unreachable
        // certificate that `scaled_compl_inf_tol` (pounce#257) removed
        // from the dynamic term, once |df| drops below
        // `mu_min·(barrier_tol_factor+1)/compl_inf_tol ≈ 1e-7`.
        let tol = data.borrow().tol;
        let df = cq.borrow().obj_scaling_factor();
        let dynamic_floor =
            tol.min(self.scaled_compl_inf_tol(df)) / (self.barrier_tol_factor + 1.0);
        let floor = self
            .mu_target
            .max(self.certificate_safe_mu_min(df))
            .max(dynamic_floor);

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
            // Mirror upstream `IpData().Set_mu(mu)` *inside* the loop
            // (`IpMonotoneMuUpdate.cpp:CalcNewMuAndTau`): the next
            // `curr_barrier_error()` must see the reduced μ. The relaxed
            // complementarity `s⊙z − μ` is keyed on `data.curr_mu`
            // (see `ipopt_cq.rs::curr_relaxed_compl_*`), so without this
            // write the re-tested `sub_err` stays pinned to the *old* μ
            // while `kappa_eps_mu = barrier_tol_factor·mu` shrinks, and
            // the loop over-drops μ in a single outer iteration. Writing
            // it here makes the residual grow as μ falls (toward the
            // current `s⊙z`), so the loop exits after one effective
            // reduction — matching IPOPT.
            data.borrow_mut().curr_mu = mu;
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

    /// pounce#257: `compl_inf_tol` is enforced on the *unscaled*
    /// complementarity, so the floor must convert it into μ's scaled space.
    #[test]
    fn dynamic_floor_converts_compl_inf_tol_into_scaled_space() {
        let m = MonotoneMuUpdate::default();
        // Unscaled problem: nothing to convert.
        assert_eq!(m.scaled_compl_inf_tol(1.0), m.compl_inf_tol);
        // jit1's B&B node: df = 1e-5 deflates the objective, so a `1e-4`
        // unscaled tolerance is `1e-9` in the space μ lives in. Taking the raw
        // value left the floor at 9.09e-10 — an unscaled complementarity of
        // 9.09e-5 at best, and 9.09e-4 at the `tol=1e-7` the driver requested.
        let floor = |ct: Number| (1e-7 as Number).min(ct) / (m.barrier_tol_factor + 1.0);
        assert!((m.scaled_compl_inf_tol(1e-5) - 1e-9).abs() < 1e-24);
        assert!(floor(m.scaled_compl_inf_tol(1e-5)) < floor(m.compl_inf_tol));
        // Signed factor: `obj_scaling_factor = -1` poses a maximization, and
        // magnitude is what the unscaling means. A negative floor would sail
        // under every comparison.
        assert_eq!(m.scaled_compl_inf_tol(-1e-5), m.scaled_compl_inf_tol(1e-5));
        // Degenerate factors fall back to the unconverted tolerance rather
        // than producing a zero or NaN floor.
        for df in [0.0, Number::NAN, Number::INFINITY] {
            assert_eq!(m.scaled_compl_inf_tol(df), m.compl_inf_tol);
        }
    }

    /// pounce#266: `mu_min` is the *other* floor term in scaled space, and
    /// unconverted it blocks the certificate below `df ≈ 1e-7` exactly as the
    /// raw `compl_inf_tol` did in #257.
    #[test]
    fn mu_min_is_capped_so_certificate_stays_reachable() {
        let m = MonotoneMuUpdate::default();
        // The cap engages once `compl_inf_tol·df/(barrier_tol_factor+1)`
        // drops under `mu_min`, i.e. below
        // `df* = mu_min·(barrier_tol_factor+1)/compl_inf_tol = 1.1e-6`.
        // Unscaled and mildly scaled problems are untouched.
        let df_star = m.mu_min * (m.barrier_tol_factor + 1.0) / m.compl_inf_tol;
        assert!((df_star - 1.1e-6).abs() < 1e-21);
        for df in [1.0, -1.0, 1e-3, 1e-5, df_star] {
            assert_eq!(m.certificate_safe_mu_min(df), m.mu_min);
        }
        // HS71 × 1e8 computes df = 8.3e-8. The certificate needs
        // μ ≤ compl_inf_tol·df ≈ 8.3e-12 < mu_min, so the cap must engage —
        // with the dynamic floor's own headroom, landing at ≈ 7.55e-13
        // (which is where Ipopt's μ bottoms out on the same file).
        let df = 8.3e-8;
        let capped = m.certificate_safe_mu_min(df);
        assert!(capped < m.mu_min);
        assert!(
            capped <= m.scaled_compl_inf_tol(df),
            "floor {capped} still exceeds the scaled certificate bound",
        );
        assert!((capped - 1e-4 * 8.3e-8 / 11.0).abs() < 1e-27);
        // Signed factor, same as `scaled_compl_inf_tol`.
        assert_eq!(m.certificate_safe_mu_min(-df), capped);
        // Degenerate factors fall back to the unconverted tolerance inside
        // `scaled_compl_inf_tol`, whose cap (9.09e-6) leaves mu_min alone.
        for df in [0.0, Number::NAN, Number::INFINITY] {
            assert_eq!(m.certificate_safe_mu_min(df), m.mu_min);
        }
    }

    /// The restoration sub-builder raises the floor to `100 · outer_mu_min`
    /// (DECONVBNE safeguard). `RestoIpoptNlp` does not override
    /// `obj_scaling_factor`, so the resto inner IPM sees `df = 1` — the cap
    /// must leave that safeguard fully intact.
    #[test]
    fn resto_mu_min_safeguard_survives_the_cap() {
        let outer = MonotoneMuUpdate::default();
        let resto = MonotoneMuUpdate::new().with_mu_min(100.0 * outer.mu_min);
        assert_eq!(resto.certificate_safe_mu_min(1.0), 100.0 * outer.mu_min);
    }
}
