//! Primal-dual perturbation handler — port of
//! `Algorithm/IpPDPerturbationHandler.{hpp,cpp}`.
//!
//! Owns the four perturbations `(δ_x, δ_s, δ_c, δ_d)` that the
//! `PDFullSpaceSolver` adds to the augmented system to recover correct
//! inertia / non-singularity. Implements upstream's full state
//! machine:
//!
//! * [`Self::consider_new_system`] — first call per new aug-system.
//!   Finalizes the previous trial's degeneracy probe, decides whether
//!   to start a new degeneracy test, and seeds `δ_c` / `δ_d` if the
//!   Jacobian is already known to be degenerate (or `perturb_always_cd`
//!   is on).
//! * [`Self::perturb_for_singular`] — escalation step taken when MA57
//!   reports `Singular`.
//! * [`Self::perturb_for_wrong_inertia`] — escalation step taken when
//!   the factor's negative-eigenvalue count disagrees with what the
//!   KKT structure requires.
//! * [`Self::current_perturbation`] — read the most recently committed
//!   `(δ_x, δ_s, δ_c, δ_d)`.
//!
//! Returns `false` when no further escalation is possible (caller must
//! enter the restoration phase). The `info_string`-mutation calls in
//! upstream are emitted via the `IpoptData` handle the caller passes
//! in; if `None` is passed, the strings are simply dropped.

use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

/// Trial state — port of upstream `TrialStatus` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialStatus {
    NoTest,
    DcEq0DxEq0,
    DcGt0DxEq0,
    DcEq0DxGt0,
    DcGt0DxGt0,
}

/// Degeneracy state — port of `DegenType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegenType {
    NotDegenerate,
    Degenerate,
    NotYetDetermined,
}

/// State + algorithmic parameters. Defaults mirror
/// `IpPDPerturbationHandler.cpp::RegisterOptions`.
#[derive(Debug, Clone)]
pub struct PdPerturbationHandler {
    // ---- algorithmic parameters (read from options) ----
    pub delta_xs_max: Number,
    pub delta_xs_min: Number,
    pub delta_xs_first_inc_fact: Number,
    pub delta_xs_inc_fact: Number,
    pub delta_xs_dec_fact: Number,
    pub delta_xs_init: Number,
    pub delta_cd_val: Number,
    pub delta_cd_exp: Number,
    pub perturb_always_cd: bool,
    pub reset_last: bool,
    pub degen_iters_max: Index,

    // ---- live state ----
    pub delta_x_curr: Number,
    pub delta_s_curr: Number,
    pub delta_c_curr: Number,
    pub delta_d_curr: Number,
    pub delta_x_last: Number,
    pub delta_s_last: Number,
    pub delta_c_last: Number,
    pub delta_d_last: Number,
    pub get_deltas_for_wrong_inertia_called: bool,
    pub hess_degenerate: DegenType,
    pub jac_degenerate: DegenType,
    pub degen_iters: Index,
    pub test_status: TrialStatus,
}

impl Default for PdPerturbationHandler {
    fn default() -> Self {
        Self {
            delta_xs_max: 1e20,
            delta_xs_min: 1e-20,
            delta_xs_first_inc_fact: 100.0,
            delta_xs_inc_fact: 8.0,
            delta_xs_dec_fact: 1.0 / 3.0,
            delta_xs_init: 1e-4,
            delta_cd_val: 1e-8,
            delta_cd_exp: 0.25,
            perturb_always_cd: false,
            reset_last: false,
            degen_iters_max: 3,
            delta_x_curr: 0.0,
            delta_s_curr: 0.0,
            delta_c_curr: 0.0,
            delta_d_curr: 0.0,
            delta_x_last: 0.0,
            delta_s_last: 0.0,
            delta_c_last: 0.0,
            delta_d_last: 0.0,
            get_deltas_for_wrong_inertia_called: false,
            hess_degenerate: DegenType::NotYetDetermined,
            jac_degenerate: DegenType::NotYetDetermined,
            degen_iters: 0,
            test_status: TrialStatus::NoTest,
        }
    }
}

/// Snapshot of the four perturbations after a state-machine call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deltas {
    pub delta_x: Number,
    pub delta_s: Number,
    pub delta_c: Number,
    pub delta_d: Number,
}

impl PdPerturbationHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure `perturb_always_cd_` and rebuild the initial `jac`
    /// state. Mirrors upstream's `InitializeImpl`.
    pub fn set_perturb_always_cd(&mut self, on: bool) {
        self.perturb_always_cd = on;
        self.jac_degenerate = if on {
            DegenType::NotDegenerate
        } else {
            DegenType::NotYetDetermined
        };
    }

    /// First call when starting a new aug-system. `mu` is the current
    /// barrier parameter (used by the `δ_cd` formula).
    /// Returns `None` if no suitable starting perturbation could be
    /// found (the caller bails).
    pub fn consider_new_system(
        &mut self,
        mu: Number,
        ip_data: Option<&IpoptDataHandle>,
    ) -> Option<Deltas> {
        self.finalize_test(ip_data);

        // Bookkeeping: roll the previous trial's `_curr` values into
        // `_last` (matches upstream cpp:158-183).
        if self.reset_last {
            self.delta_x_last = self.delta_x_curr;
            self.delta_s_last = self.delta_s_curr;
            self.delta_c_last = self.delta_c_curr;
            self.delta_d_last = self.delta_d_curr;
        } else {
            if self.delta_x_curr > 0.0 {
                self.delta_x_last = self.delta_x_curr;
            }
            if self.delta_s_curr > 0.0 {
                self.delta_s_last = self.delta_s_curr;
            }
            if self.delta_c_curr > 0.0 {
                self.delta_c_last = self.delta_c_curr;
            }
            if self.delta_d_curr > 0.0 {
                self.delta_d_last = self.delta_d_curr;
            }
        }

        let undet = matches!(self.hess_degenerate, DegenType::NotYetDetermined)
            || matches!(self.jac_degenerate, DegenType::NotYetDetermined);
        self.test_status = if undet {
            if self.perturb_always_cd {
                TrialStatus::DcGt0DxEq0
            } else {
                TrialStatus::DcEq0DxEq0
            }
        } else {
            TrialStatus::NoTest
        };

        let mut delta_c = if matches!(self.jac_degenerate, DegenType::Degenerate) {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            append_info(ip_data, "l");
            v
        } else if self.perturb_always_cd {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            v
        } else {
            self.delta_c_curr = 0.0;
            0.0
        };
        let mut delta_d = delta_c;
        self.delta_d_curr = delta_d;

        let mut delta_x = 0.0;
        let mut delta_s = 0.0;

        if matches!(self.hess_degenerate, DegenType::Degenerate) {
            self.delta_x_curr = 0.0;
            self.delta_s_curr = 0.0;
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
        }

        self.delta_x_curr = delta_x;
        self.delta_s_curr = delta_s;
        self.delta_c_curr = delta_c;
        self.delta_d_curr = delta_d;
        set_info_regu_x(ip_data, delta_x);
        self.get_deltas_for_wrong_inertia_called = false;

        Some(Deltas {
            delta_x,
            delta_s,
            delta_c,
            delta_d,
        })
    }

    /// Escalation after `Singular` factorization status. Mirrors
    /// `PerturbForSingularity` (cpp:245-364).
    pub fn perturb_for_singular(
        &mut self,
        mu: Number,
        ip_data: Option<&IpoptDataHandle>,
    ) -> Option<Deltas> {
        let mut delta_x = 0.0;
        let mut delta_s = 0.0;
        let mut delta_c = 0.0;
        let mut delta_d = 0.0;

        let undet = matches!(self.hess_degenerate, DegenType::NotYetDetermined)
            || matches!(self.jac_degenerate, DegenType::NotYetDetermined);
        if undet {
            match self.test_status {
                TrialStatus::DcEq0DxEq0 => {
                    debug_assert!(self.delta_x_curr == 0.0 && self.delta_c_curr == 0.0);
                    if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                        let v = self.delta_cd(mu);
                        self.delta_c_curr = v;
                        self.delta_d_curr = v;
                        self.test_status = TrialStatus::DcGt0DxEq0;
                    } else {
                        debug_assert!(matches!(self.hess_degenerate, DegenType::NotYetDetermined));
                        if !self.get_deltas_for_wrong_inertia(
                            &mut delta_x,
                            &mut delta_s,
                            &mut delta_c,
                            &mut delta_d,
                            ip_data,
                        ) {
                            return None;
                        }
                        self.test_status = TrialStatus::DcEq0DxGt0;
                    }
                }
                TrialStatus::DcGt0DxEq0 => {
                    debug_assert!(self.delta_x_curr == 0.0 && self.delta_c_curr > 0.0);
                    debug_assert!(matches!(self.jac_degenerate, DegenType::NotYetDetermined));
                    if !self.perturb_always_cd {
                        self.delta_c_curr = 0.0;
                        self.delta_d_curr = 0.0;
                        if !self.get_deltas_for_wrong_inertia(
                            &mut delta_x,
                            &mut delta_s,
                            &mut delta_c,
                            &mut delta_d,
                            ip_data,
                        ) {
                            return None;
                        }
                        self.test_status = TrialStatus::DcEq0DxGt0;
                    } else if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    } else {
                        self.test_status = TrialStatus::DcGt0DxGt0;
                    }
                }
                TrialStatus::DcEq0DxGt0 => {
                    debug_assert!(self.delta_x_curr > 0.0 && self.delta_c_curr == 0.0);
                    let v = self.delta_cd(mu);
                    self.delta_c_curr = v;
                    self.delta_d_curr = v;
                    if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    }
                    self.test_status = TrialStatus::DcGt0DxGt0;
                }
                TrialStatus::DcGt0DxGt0 => {
                    if !self.get_deltas_for_wrong_inertia(
                        &mut delta_x,
                        &mut delta_s,
                        &mut delta_c,
                        &mut delta_d,
                        ip_data,
                    ) {
                        return None;
                    }
                }
                TrialStatus::NoTest => {
                    debug_assert!(false, "perturb_for_singular: NoTest in undetermined branch");
                }
            }
        } else if self.delta_c_curr > 0.0 {
            // Already perturbed C; treat as wrong-inertia.
            if !self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            ) {
                return None;
            }
        } else {
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            self.delta_d_curr = v;
            append_info(ip_data, "L");
        }

        let out = Deltas {
            delta_x: self.delta_x_curr,
            delta_s: self.delta_s_curr,
            delta_c: self.delta_c_curr,
            delta_d: self.delta_d_curr,
        };
        set_info_regu_x(ip_data, out.delta_x);
        Some(out)
    }

    /// Escalation after `WrongInertia` factorization status. Mirrors
    /// `PerturbForWrongInertia` (cpp:419-450).
    pub fn perturb_for_wrong_inertia(
        &mut self,
        mu: Number,
        ip_data: Option<&IpoptDataHandle>,
    ) -> Option<Deltas> {
        if std::env::var_os("POUNCE_DBG_PERT").is_some() {
            let it = ip_data.map(|d| d.borrow().iter_count).unwrap_or(-1);
            tracing::debug!(target: "pounce::linsol",
                "[PERT] iter={} WRONG_INERTIA mu={:.2e} dx_last={:.2e} dx_curr={:.2e}",
                it, mu, self.delta_x_last, self.delta_x_curr
            );
        }
        self.finalize_test(ip_data);

        let mut delta_x = 0.0;
        let mut delta_s = 0.0;
        let mut delta_c = 0.0;
        let mut delta_d = 0.0;
        let mut ok = self.get_deltas_for_wrong_inertia(
            &mut delta_x,
            &mut delta_s,
            &mut delta_c,
            &mut delta_d,
            ip_data,
        );
        // Upstream "no progress on δ_x but δ_c == 0" recovery: bring
        // up the C/D perturbation, reset Hessian degeneracy, retry.
        // Upstream peeks at the OUT-parameter `delta_c`, but
        // `get_deltas_for_wrong_inertia` only writes that on success;
        // we look at the handler's own δ_c_curr instead, which
        // matches the algorithmic intent unambiguously.
        if !ok && self.delta_c_curr == 0.0 {
            debug_assert_eq!(self.delta_d_curr, 0.0);
            let v = self.delta_cd(mu);
            self.delta_c_curr = v;
            self.delta_d_curr = v;
            self.delta_x_curr = 0.0;
            self.delta_s_curr = 0.0;
            self.test_status = TrialStatus::NoTest;
            if matches!(self.hess_degenerate, DegenType::Degenerate) {
                self.hess_degenerate = DegenType::NotYetDetermined;
            }
            ok = self.get_deltas_for_wrong_inertia(
                &mut delta_x,
                &mut delta_s,
                &mut delta_c,
                &mut delta_d,
                ip_data,
            );
        }
        if !ok {
            return None;
        }
        Some(Deltas {
            delta_x,
            delta_s,
            delta_c,
            delta_d,
        })
    }

    /// Read the most recently committed perturbations.
    pub fn current_perturbation(&self) -> Deltas {
        Deltas {
            delta_x: self.delta_x_curr,
            delta_s: self.delta_s_curr,
            delta_c: self.delta_c_curr,
            delta_d: self.delta_d_curr,
        }
    }

    /// Internal — pure escalation of `δ_x` / `δ_s`. Returns `false` if
    /// `δ_x` would exceed `delta_xs_max`. Mirrors
    /// `get_deltas_for_wrong_inertia`.
    fn get_deltas_for_wrong_inertia(
        &mut self,
        delta_x: &mut Number,
        delta_s: &mut Number,
        delta_c: &mut Number,
        delta_d: &mut Number,
        ip_data: Option<&IpoptDataHandle>,
    ) -> bool {
        if self.delta_x_curr == 0.0 {
            self.delta_x_curr = if self.delta_x_last == 0.0 {
                self.delta_xs_init
            } else {
                self.delta_xs_min
                    .max(self.delta_x_last * self.delta_xs_dec_fact)
            };
        } else if self.delta_x_last == 0.0 || 1e5 * self.delta_x_last < self.delta_x_curr {
            self.delta_x_curr *= self.delta_xs_first_inc_fact;
        } else {
            self.delta_x_curr *= self.delta_xs_inc_fact;
        }
        if self.delta_x_curr > self.delta_xs_max {
            self.delta_x_last = 0.0;
            self.delta_s_last = 0.0;
            append_info(ip_data, "dx");
            return false;
        }
        self.delta_s_curr = self.delta_x_curr;

        *delta_x = self.delta_x_curr;
        *delta_s = self.delta_s_curr;
        *delta_c = self.delta_c_curr;
        *delta_d = self.delta_d_curr;
        set_info_regu_x(ip_data, *delta_x);
        self.get_deltas_for_wrong_inertia_called = true;
        true
    }

    fn delta_cd(&self, mu: Number) -> Number {
        self.delta_cd_val * mu.powf(self.delta_cd_exp)
    }

    /// Read the test outcome from the just-completed (non-singular)
    /// factor and update degeneracy flags. Mirrors `finalize_test`
    /// (cpp:470-538).
    fn finalize_test(&mut self, ip_data: Option<&IpoptDataHandle>) {
        match self.test_status {
            TrialStatus::NoTest => (),
            TrialStatus::DcEq0DxEq0 => {
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined)
                    && matches!(self.jac_degenerate, DegenType::NotYetDetermined)
                {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nhj ");
                } else if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nh ");
                } else if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nj ");
                }
            }
            TrialStatus::DcGt0DxEq0 => {
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.hess_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nh ");
                }
                if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.degen_iters += 1;
                    if self.degen_iters >= self.degen_iters_max {
                        self.jac_degenerate = DegenType::Degenerate;
                        append_info(ip_data, "Dj ");
                    }
                    append_info(ip_data, "L");
                }
            }
            TrialStatus::DcEq0DxGt0 => {
                if matches!(self.jac_degenerate, DegenType::NotYetDetermined) {
                    self.jac_degenerate = DegenType::NotDegenerate;
                    append_info(ip_data, "Nj ");
                }
                if matches!(self.hess_degenerate, DegenType::NotYetDetermined) {
                    self.degen_iters += 1;
                    if self.degen_iters >= self.degen_iters_max {
                        self.hess_degenerate = DegenType::Degenerate;
                        append_info(ip_data, "Dh ");
                    }
                }
            }
            TrialStatus::DcGt0DxGt0 => {
                self.degen_iters += 1;
                if self.degen_iters >= self.degen_iters_max {
                    self.hess_degenerate = DegenType::Degenerate;
                    self.jac_degenerate = DegenType::Degenerate;
                    append_info(ip_data, "Dhj ");
                }
                append_info(ip_data, "L");
            }
        }
    }
}

fn append_info(ip_data: Option<&IpoptDataHandle>, s: &str) {
    if let Some(h) = ip_data {
        h.borrow_mut().append_info_string(s);
    }
}

fn set_info_regu_x(ip_data: Option<&IpoptDataHandle>, v: Number) {
    if let Some(h) = ip_data {
        h.borrow_mut().info_regu_x = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_wrong_inertia_perturbation_is_delta_xs_init() {
        let mut h = PdPerturbationHandler::new();
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        // delta_xs_init = first_hessian_perturbation = 1e-4
        assert!((d.delta_x - 1e-4).abs() < 1e-20);
        assert_eq!(d.delta_x, d.delta_s);
        assert_eq!(d.delta_c, 0.0);
        assert_eq!(d.delta_d, 0.0);
    }

    #[test]
    fn second_perturbation_uses_first_inc_fact() {
        // After the *first* nonzero δ_x, with δ_x_last == 0, the
        // doubling uses `delta_xs_first_inc_fact = 100` per upstream
        // (cpp:386-389: "if delta_x_last_ == 0 ...").
        let mut h = PdPerturbationHandler::new();
        let d1 = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        let d2 = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!((d2.delta_x - d1.delta_x * 100.0).abs() < 1e-15);
    }

    #[test]
    fn third_perturbation_uses_inc_fact() {
        // After delta_x_last has been set (via consider_new_system or
        // first inc), continued growth uses `delta_xs_inc_fact = 8`.
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = 1e-2;
        h.delta_x_last = 1e-2;
        let d = h.perturb_for_wrong_inertia(0.1, None).unwrap();
        assert!((d.delta_x - 1e-2 * 8.0).abs() < 1e-15);
    }

    #[test]
    fn perturbation_caps_at_max_when_dcd_already_active() {
        // When δ_c is already > 0 (e.g., perturb_always_cd, or after a
        // singular-recovery), the fallback path inside
        // `perturb_for_wrong_inertia` is skipped and the
        // δ_x-overflow surfaces as `None`.
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = h.delta_xs_max;
        h.delta_c_curr = 1e-4;
        h.delta_d_curr = 1e-4;
        assert!(h.perturb_for_wrong_inertia(0.1, None).is_none());
    }

    #[test]
    fn consider_new_system_with_perturb_always_cd_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        h.set_perturb_always_cd(true);
        let mu = 0.1;
        let d = h.consider_new_system(mu, None).unwrap();
        let expected = h.delta_cd_val * mu.powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
        assert!((d.delta_d - expected).abs() < 1e-15);
        assert_eq!(d.delta_x, 0.0);
        assert_eq!(d.delta_s, 0.0);
    }

    #[test]
    fn consider_new_system_default_zeros_dcd() {
        let mut h = PdPerturbationHandler::new();
        let d = h.consider_new_system(0.1, None).unwrap();
        assert_eq!(
            d,
            Deltas {
                delta_x: 0.0,
                delta_s: 0.0,
                delta_c: 0.0,
                delta_d: 0.0
            }
        );
    }

    #[test]
    fn singular_in_test_dc_eq0_dx_eq0_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        let _ = h.consider_new_system(0.1, None).unwrap();
        // After consider_new_system on a fresh handler, test_status
        // should be DcEq0DxEq0 (since both flags are NotYetDetermined,
        // and perturb_always_cd is false).
        assert_eq!(h.test_status, TrialStatus::DcEq0DxEq0);
        let d = h.perturb_for_singular(0.1, None).unwrap();
        let expected = h.delta_cd_val * (0.1_f64).powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
        assert!((d.delta_d - expected).abs() < 1e-15);
        assert_eq!(d.delta_x, 0.0);
        assert_eq!(h.test_status, TrialStatus::DcGt0DxEq0);
    }

    #[test]
    fn singular_when_determined_with_dc_zero_seeds_dcd() {
        let mut h = PdPerturbationHandler::new();
        h.hess_degenerate = DegenType::NotDegenerate;
        h.jac_degenerate = DegenType::NotDegenerate;
        h.test_status = TrialStatus::NoTest;
        let d = h.perturb_for_singular(0.1, None).unwrap();
        let expected = h.delta_cd_val * (0.1_f64).powf(h.delta_cd_exp);
        assert!((d.delta_c - expected).abs() < 1e-15);
    }

    #[test]
    fn finalize_test_sets_not_degenerate_after_dc_eq0_dx_eq0_pass() {
        let mut h = PdPerturbationHandler::new();
        let _ = h.consider_new_system(0.1, None).unwrap();
        // Simulate the next call recognizing that the previous trial
        // factor was non-singular: a fresh consider_new_system runs
        // finalize_test first.
        let _ = h.consider_new_system(0.1, None).unwrap();
        assert_eq!(h.hess_degenerate, DegenType::NotDegenerate);
        assert_eq!(h.jac_degenerate, DegenType::NotDegenerate);
    }

    #[test]
    fn current_perturbation_returns_committed_values() {
        let mut h = PdPerturbationHandler::new();
        h.delta_x_curr = 1.0;
        h.delta_s_curr = 2.0;
        h.delta_c_curr = 3.0;
        h.delta_d_curr = 4.0;
        let d = h.current_perturbation();
        assert_eq!(d.delta_x, 1.0);
        assert_eq!(d.delta_s, 2.0);
        assert_eq!(d.delta_c, 3.0);
        assert_eq!(d.delta_d, 4.0);
    }
}
