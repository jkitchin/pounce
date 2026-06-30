//! Optimal-error convergence check — port of
//! `Algorithm/IpOptErrorConvCheck.{hpp,cpp}`.
//!
//! Tolerance state machine over `(nlp_err, iter_count)` plus
//! per-component infeasibilities pulled directly from
//! [`IpoptCalculatedQuantities`]. The scalar
//! [`Self::check_convergence`] entry point only gates on
//! `nlp_err <= tol` (matching upstream when the per-component
//! tolerances are at their `+∞` sentinels); the state-aware
//! [`Self::check_convergence_with_state`] adds the
//! `dual_inf_tol` / `constr_viol_tol` / `compl_inf_tol` gates that
//! mirror upstream `OptimalityErrorConvergenceCheck::CheckConvergence`.

use crate::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

pub struct OptErrorConvCheck {
    pub tol: Number,
    pub dual_inf_tol: Number,
    pub constr_viol_tol: Number,
    pub compl_inf_tol: Number,
    pub acceptable_tol: Number,
    pub acceptable_dual_inf_tol: Number,
    pub acceptable_constr_viol_tol: Number,
    pub acceptable_compl_inf_tol: Number,
    pub acceptable_obj_change_tol: Number,
    pub acceptable_iter: Index,
    pub max_iter: Index,
    pub max_cpu_time: Number,
    pub max_wall_time: Number,
    pub acceptable_count: Index,
    /// Objective value at the last iterate the main loop stashed via
    /// `set_curr_acceptable_obj`. Used by the
    /// `acceptable_obj_change_tol` cross-check. `None` until an
    /// acceptable point has been recorded.
    pub last_acceptable_obj: Option<Number>,
    /// Tolerance on the scaled infeasibility stationarity
    /// `‖Jᵀc‖/max(1,‖c‖)`. An iterate counts toward the infeasibility
    /// streak when this ratio is at or below this value while the
    /// constraint violation stays bounded away from zero. Rapid
    /// infeasibility detection is disabled when this is non-positive.
    pub infeas_stationarity_tol: Number,
    /// Multiple of `constr_viol_tol` the constraint violation must
    /// exceed before an iterate can count as infeasible-stationary —
    /// keeps detection from firing on nearly-feasible flat spots.
    pub infeas_viol_kappa: Number,
    /// Consecutive infeasible-stationary iterations required before
    /// terminating with `LocallyInfeasible`. Non-positive disables
    /// rapid infeasibility detection.
    pub infeas_max_streak: Index,
    /// Running count of consecutive infeasible-stationary iterations.
    pub infeas_streak: Index,
}

impl Default for OptErrorConvCheck {
    fn default() -> Self {
        // Defaults from `IpOptErrorConvCheck.cpp:RegisterOptions`.
        Self {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1e10,
            acceptable_constr_viol_tol: 1e-2,
            acceptable_compl_inf_tol: 1e-2,
            acceptable_obj_change_tol: 1e20,
            acceptable_iter: 15,
            max_iter: 3000,
            max_cpu_time: 1e6,
            max_wall_time: 1e6,
            acceptable_count: 0,
            last_acceptable_obj: None,
            infeas_stationarity_tol: 1e-8,
            infeas_viol_kappa: 1e2,
            infeas_max_streak: 5,
            infeas_streak: 0,
        }
    }
}

impl OptErrorConvCheck {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pure helper for the per-component upstream gate. Returns `true`
    /// iff every supplied residual sits at or below its tolerance.
    /// Factored out so tests can exercise the gating logic without
    /// constructing a full `IpoptCq`.
    fn passes_component_tols(
        &self,
        overall: Number,
        dual_inf: Number,
        constr_viol: Number,
        compl_inf: Number,
    ) -> bool {
        overall <= self.tol
            && dual_inf <= self.dual_inf_tol
            && constr_viol <= self.constr_viol_tol
            && compl_inf <= self.compl_inf_tol
    }

    /// Pure helper mirroring upstream
    /// `OptimalityErrorConvergenceCheck::CurrentIsAcceptable`. Tests
    /// the per-component `acceptable_*_tol` triplet plus the optional
    /// `acceptable_obj_change_tol` stability cross-check.
    fn passes_acceptable_tols(
        &self,
        overall: Number,
        dual_inf: Number,
        constr_viol: Number,
        compl_inf: Number,
        curr_f: Number,
    ) -> bool {
        // A point is never acceptable if the scaled error metric or the
        // objective itself is non-finite. Without the `curr_f` guard a NaN/Inf
        // objective with otherwise-small infeasibility (e.g. CUTE `himmelbj`,
        // where f evaluates to NaN at a near-feasible point) would be recorded
        // as the acceptable rollback point and reported under
        // `Solved_To_Acceptable_Level` with a `nan` objective.
        if !overall.is_finite() || !curr_f.is_finite() {
            return false;
        }
        let component_ok = overall <= self.acceptable_tol
            && dual_inf <= self.acceptable_dual_inf_tol
            && constr_viol <= self.acceptable_constr_viol_tol
            && compl_inf <= self.acceptable_compl_inf_tol;
        if !component_ok {
            return false;
        }
        // Upstream `IpOptErrorConvCheck.cpp:CurrentIsAcceptable` — when
        // an acceptable point has already been recorded and the user
        // tightened `acceptable_obj_change_tol` below the 1e20
        // sentinel, the iterate is only re-acceptable if `f` has moved
        // by less than `tol * max(1, |f|)` relative to the recorded
        // value. Skipped when no prior point exists or the cross-check
        // is disabled.
        if self.acceptable_obj_change_tol < 1e20 {
            if let Some(prev) = self.last_acceptable_obj {
                let denom = curr_f.abs().max(1.0);
                if (prev - curr_f).abs() >= self.acceptable_obj_change_tol * denom {
                    return false;
                }
            }
        }
        true
    }

    /// Pure predicate for a single infeasible-stationary iterate: the
    /// constraint violation is bounded away from zero
    /// (`constr_viol > infeas_viol_kappa · constr_viol_tol`) and the
    /// scaled infeasibility gradient `‖Jᵀc‖/max(1,‖c‖)` is at or below
    /// `infeas_stationarity_tol`. Returns `false` when rapid
    /// infeasibility detection is disabled (either knob non-positive).
    fn is_infeasible_stationary(&self, constr_viol: Number, stationarity: Number) -> bool {
        if self.infeas_stationarity_tol <= 0.0 || self.infeas_max_streak <= 0 {
            return false;
        }
        constr_viol > self.infeas_viol_kappa * self.constr_viol_tol
            && stationarity <= self.infeas_stationarity_tol
    }

    /// Advance the rapid-infeasibility-detection streak by one
    /// iteration. An infeasible-stationary iterate (see
    /// [`Self::is_infeasible_stationary`]) increments the streak; any
    /// other iterate resets it to zero. Returns `true` once the streak
    /// reaches `infeas_max_streak`, signalling the caller to terminate
    /// with `ConvergenceStatus::LocallyInfeasible`. The streak guards
    /// against firing on a transient flat spot.
    fn note_infeasible_stationary(&mut self, constr_viol: Number, stationarity: Number) -> bool {
        if self.is_infeasible_stationary(constr_viol, stationarity) {
            self.infeas_streak += 1;
            self.infeas_streak >= self.infeas_max_streak
        } else {
            self.infeas_streak = 0;
            false
        }
    }
}

impl ConvCheck for OptErrorConvCheck {
    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus {
        if nlp_err <= self.tol {
            return ConvergenceStatus::Converged;
        }
        // `acceptable_iter == 0` disables acceptable-level termination,
        // mirroring upstream `IpOptErrorConvCheck.cpp:241`
        // (`if( acceptable_iter_ > 0 && CurrentIsAcceptable() )`). Without
        // the `> 0` guard, a zero would make `acceptable_count >= 0` fire on
        // the first acceptable iterate — the opposite of "disabled".
        if self.acceptable_iter > 0 && nlp_err <= self.acceptable_tol {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::ConvergedToAcceptable;
            }
        } else {
            self.acceptable_count = 0;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
        }
        ConvergenceStatus::Continue
    }

    fn check_convergence_with_state(
        &mut self,
        nlp_err: Number,
        iter_count: Index,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> ConvergenceStatus {
        // Mirror upstream `IpOptErrorConvCheck.cpp::CheckConvergence`:
        // the scaled scalar `nlp_err` must drop below `tol` AND each
        // per-component value must sit under its own tolerance. The
        // component tolerances (`dual_inf_tol`/`constr_viol_tol`/
        // `compl_inf_tol`) are defined on the *unscaled* (user-original)
        // residuals — both upstream and per pounce's own option help text
        // — so we gate on the unscaled accessors. This resolves the former
        // M1 deviation (gating on internally-scaled residuals), which let
        // an ill-conditioned, nlp_scaling-deflated solve report
        // `Solve_Succeeded` while the user-space duals had drifted
        // (pounce#173). When no scaling is active the unscaled accessors
        // return the scaled values unchanged, so behaviour is identical on
        // the common path.
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        let curr_f = cq_ref.curr_f();
        drop(cq_ref);

        if self.passes_component_tols(nlp_err, dual_inf, constr_viol, compl_inf) {
            return ConvergenceStatus::Converged;
        }
        // `acceptable_iter == 0` disables acceptable-level termination
        // (upstream `IpOptErrorConvCheck.cpp:241`). See `check_convergence`.
        if self.acceptable_iter > 0
            && self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f)
        {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::ConvergedToAcceptable;
            }
        } else {
            self.acceptable_count = 0;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
        }
        // Rapid infeasibility detection — recognise an iterate
        // converging to a stationary point of the constraint
        // violation with the violation bounded away from zero, and
        // exit with `LocallyInfeasible` instead of grinding to
        // `max_iter` or thrashing restoration. Gated behind an
        // `infeas_max_streak`-iteration streak to avoid firing on a
        // transient flat spot. The outer guard skips the two
        // transpose-products when detection is disabled.
        if self.infeas_stationarity_tol > 0.0 && self.infeas_max_streak > 0 {
            let stationarity = cq.borrow().curr_infeasibility_stationarity();
            if self.note_infeasible_stationary(constr_viol, stationarity) {
                return ConvergenceStatus::LocallyInfeasible;
            }
        }
        // Time-budget gates. Upstream
        // `IpOptErrorConvCheck.cpp::CheckConvergence` reads the
        // application-level start time; pounce piggybacks on
        // `data.timing.overall_alg`, which `IpoptApplication` starts
        // at the top of `optimize_constrained`. `live_*` returns the
        // running elapsed without forcing a `start/end` cycle.
        let timing = &data.borrow().timing;
        if timing.overall_alg.live_cpu_time() >= self.max_cpu_time {
            return ConvergenceStatus::CpuTimeExceeded;
        }
        if timing.overall_alg.live_wallclock_time() >= self.max_wall_time {
            return ConvergenceStatus::WallTimeExceeded;
        }
        ConvergenceStatus::Continue
    }

    fn tol_or_default(&self) -> Number {
        self.tol
    }

    fn set_tolerance(&mut self, name: &str, value: Number) -> bool {
        match name {
            "tol" => self.tol = value,
            "dual_inf_tol" => self.dual_inf_tol = value,
            "constr_viol_tol" => self.constr_viol_tol = value,
            "compl_inf_tol" => self.compl_inf_tol = value,
            "acceptable_tol" => self.acceptable_tol = value,
            "acceptable_dual_inf_tol" => self.acceptable_dual_inf_tol = value,
            "acceptable_constr_viol_tol" => self.acceptable_constr_viol_tol = value,
            "acceptable_compl_inf_tol" => self.acceptable_compl_inf_tol = value,
            "acceptable_obj_change_tol" => self.acceptable_obj_change_tol = value,
            _ => return false,
        }
        true
    }

    fn current_is_acceptable(&self, nlp_err: Number) -> bool {
        // Scalar fallback used when the caller has no `IpoptCq` handle
        // (e.g. unit tests). The state-aware variant
        // [`Self::current_is_acceptable_with_state`] mirrors upstream
        // more faithfully by gating on the per-component
        // `acceptable_*_tol` triplet plus the obj-change cross-check.
        nlp_err.is_finite() && nlp_err <= self.acceptable_tol
    }

    fn current_is_acceptable_with_state(
        &self,
        nlp_err: Number,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> bool {
        let cq_ref = cq.borrow();
        // Unscaled per-component residuals — see `check_convergence_with_state`
        // (the `acceptable_*_tol` triplet is likewise defined on the
        // user-original residuals).
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        let curr_f = cq_ref.curr_f();
        drop(cq_ref);
        self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f)
    }

    fn set_curr_acceptable_obj(&mut self, obj: Number) {
        self.last_acceptable_obj = Some(obj);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_at_tol() {
        let mut c = OptErrorConvCheck::new();
        assert_eq!(c.check_convergence(1e-9, 0), ConvergenceStatus::Converged);
    }

    #[test]
    fn acceptable_iter_count_threshold() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        // nlp_err between tol (1e-8) and acceptable (1e-6).
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 1), ConvergenceStatus::Continue);
        assert_eq!(
            c.check_convergence(1e-7, 2),
            ConvergenceStatus::ConvergedToAcceptable
        );
    }

    #[test]
    fn acceptable_iter_zero_disables_acceptable_termination() {
        // Upstream `IpOptErrorConvCheck.cpp:241` gates the acceptable
        // counter on `acceptable_iter_ > 0`, so a zero disables the
        // acceptable-level exit entirely. Before the guard, `>= 0` made
        // pounce fire on the FIRST acceptable iterate (the opposite).
        let mut c = OptErrorConvCheck {
            acceptable_iter: 0,
            ..Default::default()
        };
        // Many iterates parked between tol (1e-8) and acceptable (1e-6)
        // must never trigger ConvergedToAcceptable; the run continues
        // until tol or max_iter.
        for k in 0..50 {
            assert_eq!(
                c.check_convergence(1e-7, k),
                ConvergenceStatus::Continue,
                "acceptable_iter=0 must not stop at the acceptable level (iter {k})"
            );
        }
        // tol is still honored regardless.
        assert_eq!(c.check_convergence(1e-9, 51), ConvergenceStatus::Converged);
    }

    #[test]
    fn streak_resets_when_above_acceptable() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        // Above acceptable resets the counter.
        assert_eq!(c.check_convergence(1e-3, 1), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 2), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 3), ConvergenceStatus::Continue);
        assert_eq!(
            c.check_convergence(1e-7, 4),
            ConvergenceStatus::ConvergedToAcceptable
        );
    }

    #[test]
    fn passes_acceptable_tols_gates_on_per_component_triplet() {
        let c = OptErrorConvCheck {
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1e-3,
            acceptable_constr_viol_tol: 1e-3,
            acceptable_compl_inf_tol: 1e-3,
            ..Default::default()
        };
        assert!(c.passes_acceptable_tols(1e-7, 1e-4, 1e-4, 1e-4, 0.0));
        // dual_inf above its acceptable threshold blocks.
        assert!(!c.passes_acceptable_tols(1e-7, 1.0, 1e-4, 1e-4, 0.0));
        // overall above acceptable_tol blocks.
        assert!(!c.passes_acceptable_tols(1e-5, 1e-4, 1e-4, 1e-4, 0.0));
    }

    #[test]
    fn passes_acceptable_tols_honors_obj_change_tol() {
        let mut c = OptErrorConvCheck {
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1.0,
            acceptable_constr_viol_tol: 1.0,
            acceptable_compl_inf_tol: 1.0,
            acceptable_obj_change_tol: 0.1,
            ..Default::default()
        };
        // First call always acceptable (no prior obj).
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 10.0));
        c.set_curr_acceptable_obj(10.0);
        // Same f → change well under threshold → still acceptable.
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 10.0));
        // f moved by 2.0 with threshold 0.1 * max(1, |11.0|) = 1.1 →
        // absolute change 1.0 < 1.1: acceptable.
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 11.0));
        // f moved by 5.0 — absolute change 5.0 > 1.5 = 0.1 * 15 →
        // rejected (the stability cross-check fires).
        assert!(!c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 15.0));
    }

    use crate::conv_check::r#trait::ConvCheck;

    #[test]
    fn set_curr_acceptable_obj_records_for_cross_check() {
        let mut c = OptErrorConvCheck::new();
        assert!(c.last_acceptable_obj.is_none());
        ConvCheck::set_curr_acceptable_obj(&mut c, 4.2);
        assert_eq!(c.last_acceptable_obj, Some(4.2));
    }

    #[test]
    fn passes_component_tols_requires_all_under_threshold() {
        let c = OptErrorConvCheck {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // All under threshold → converged.
        assert!(c.passes_component_tols(1e-9, 0.5, 1e-5, 1e-5));
        // dual_inf above its tolerance blocks even when nlp_err is tiny.
        assert!(!c.passes_component_tols(1e-12, 2.0, 1e-5, 1e-5));
        // compl_inf above its tolerance blocks.
        assert!(!c.passes_component_tols(1e-12, 0.0, 0.0, 1e-2));
        // constr_viol above its tolerance blocks.
        assert!(!c.passes_component_tols(1e-12, 0.0, 1e-2, 0.0));
    }

    #[test]
    fn infeasible_stationary_requires_violation_and_flat_gradient() {
        let c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2, // violation threshold = 1e-2
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 5,
            ..Default::default()
        };
        // Violation well above 1e-2 and the infeasibility gradient
        // essentially zero → counts as infeasible-stationary.
        assert!(c.is_infeasible_stationary(1e-1, 1e-9));
        // Violation above threshold but the gradient is not flat →
        // still making feasibility progress, does not count.
        assert!(!c.is_infeasible_stationary(1e-1, 1e-3));
        // Gradient flat but violation below threshold → nearly
        // feasible, does not count.
        assert!(!c.is_infeasible_stationary(1e-3, 1e-9));
    }

    #[test]
    fn infeasible_stationary_disabled_by_nonpositive_knobs() {
        let off_tol = OptErrorConvCheck {
            infeas_stationarity_tol: 0.0,
            infeas_max_streak: 5,
            ..Default::default()
        };
        assert!(!off_tol.is_infeasible_stationary(1e9, 0.0));
        let off_streak = OptErrorConvCheck {
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 0,
            ..Default::default()
        };
        assert!(!off_streak.is_infeasible_stationary(1e9, 0.0));
    }

    #[test]
    fn infeasible_stationary_streak_fires_only_after_max_streak() {
        let mut c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2, // violation threshold = 1e-2
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 3,
            ..Default::default()
        };
        // Infeasible-stationary iterate: violation 1e-1 > 1e-2, flat
        // gradient. Streak accrues but does not fire until the third.
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        assert!(c.note_infeasible_stationary(1e-1, 1e-9));
    }

    #[test]
    fn infeasible_stationary_streak_resets_on_feasibility_progress() {
        let mut c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2,
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 3,
            ..Default::default()
        };
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        // A non-stationary iterate (gradient not flat) resets the streak.
        assert!(!c.note_infeasible_stationary(1e-1, 1e-3));
        assert_eq!(c.infeas_streak, 0);
        // The streak must rebuild from scratch — no carry-over credit.
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        assert!(!c.note_infeasible_stationary(1e-1, 1e-9));
        assert!(c.note_infeasible_stationary(1e-1, 1e-9));
    }

    #[test]
    fn infeasible_stationary_streak_never_fires_when_disabled() {
        let mut c = OptErrorConvCheck {
            infeas_stationarity_tol: 0.0,
            infeas_max_streak: 5,
            ..Default::default()
        };
        for _ in 0..20 {
            assert!(!c.note_infeasible_stationary(1e9, 0.0));
        }
        assert_eq!(c.infeas_streak, 0);
    }

    #[test]
    fn max_iter_exceeded() {
        let mut c = OptErrorConvCheck {
            max_iter: 5,
            ..Default::default()
        };
        assert_eq!(
            c.check_convergence(1.0, 5),
            ConvergenceStatus::MaxIterExceeded
        );
    }
}
