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
        if !overall.is_finite() {
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
}

impl ConvCheck for OptErrorConvCheck {
    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus {
        if nlp_err <= self.tol {
            return ConvergenceStatus::Converged;
        }
        if nlp_err <= self.acceptable_tol {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::Converged;
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
        // unscaled component must sit under its own tolerance. The
        // `acceptable_*` machinery still runs off the scalar; that
        // expansion lands with the `acceptable_*` option wiring.
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_complementarity_max();
        let curr_f = cq_ref.curr_f();
        drop(cq_ref);

        if self.passes_component_tols(nlp_err, dual_inf, constr_viol, compl_inf) {
            return ConvergenceStatus::Converged;
        }
        if self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f) {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::Converged;
            }
        } else {
            self.acceptable_count = 0;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
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
        let dual_inf = cq_ref.curr_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_complementarity_max();
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
        assert_eq!(c.check_convergence(1e-7, 2), ConvergenceStatus::Converged);
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
        assert_eq!(c.check_convergence(1e-7, 4), ConvergenceStatus::Converged);
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
