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
    /// Objective-scale floor below which a strict certificate is refused
    /// while the *unscaled* KKT error is still above `acceptable_tol`
    /// (gh #200). See [`certificate_masked`]. `0` disables the mechanism
    /// entirely, restoring bit-for-bit upstream-Ipopt behaviour.
    pub obj_scale_certificate_threshold: Number,
    /// Whether a masked **strict** certificate was ever refused this solve.
    pub veto_fired: bool,
    /// Whether a masked **acceptable-level** termination was ever refused.
    ///
    /// Tracked separately because the two refusals must be undone differently:
    /// a refused strict certificate restores as `Success`, a refused
    /// acceptable-level one as `StopAtAcceptablePoint`. Conflating them would
    /// either over-claim a status or, as originally written, leave the
    /// acceptable-level refusal with no safety net at all.
    pub acceptable_veto_fired: bool,
    /// Iterations spent since the veto first refused a certificate.
    ///
    /// The veto is a bet that continuing reaches a better point. Some problems
    /// never let it pay off — an unscaled error pinned above `acceptable_tol`
    /// by an unbounded direction keeps the veto engaged until `max_iter`,
    /// turning a 40-iteration solve into a 300-iteration one for nothing. Past
    /// [`VETO_MAX_EXTRA_ITERS`] the bet is called off and the run is allowed to
    /// terminate normally; correctness does not depend on the cap, because the
    /// refused certificate is restored either way.
    pub veto_extra_iters: Index,
}

/// How many iterations the veto may spend before its bet is called off.
///
/// Generous relative to what a successful rescue costs — the reported quartics
/// reach the true minimum in 11-15 extra iterations — but bounded, so a veto
/// that can never lift (an unscaled error pinned above `acceptable_tol` by an
/// unbounded direction) cannot run to `max_iter`. Correctness does not rest on
/// this number: whatever happens after the budget is spent, the refused
/// certificate is still restored if the run ends without a better one.
const VETO_MAX_EXTRA_ITERS: Index = 60;

/// Is a passing strict certificate *masked* by an extreme objective scale
/// (gh #200)?
///
/// Gradient-based scaling picks `df = nlp_scaling_max_gradient / max‖∇f‖`,
/// floored at `nlp_scaling_min_value = 1e-8`. On a flat quartic the initial
/// gradient is enormous (`quartc`: ~4e12 → `df` pinned at the floor), and the
/// strict test then runs on the *scaled* aggregate. Because a quartic's
/// gradient vanishes cubically toward its minimum while `df` stays fixed at its
/// initial value, the scaled error crosses `tol` roughly 30% of the way in: the
/// solver certifies optimality at `quartc` objective 248.88 when the true
/// minimum is ~0, with an unscaled dual infeasibility of 0.84.
///
/// This predicate deliberately does **not** try to decide whether the stop is
/// genuinely false — it only asks whether the conditions that make a false stop
/// *possible* are present. Distinguishing a masked certificate from an honest
/// one at a small scale cannot be done from the residual magnitude: `meyer3`
/// sits at the same 1e-8 scale floor as `quartc` while being genuinely
/// converged, and the unscaled error is a *dimensional* quantity, so any
/// absolute cutoff separating them would move if the objective were rescaled —
/// precisely the sensitivity this bug is about. An earlier revision of this
/// work did exactly that (a 5e-2 bar fitted to the gap in one benchmark suite);
/// it is not defensible and was removed.
///
/// Instead the caller *tests* the hypothesis: it refuses to stop, continues,
/// and sees whether the iterates actually go anywhere. If they do, the stop was
/// false. If they do not, the certificate is honoured unchanged — so the
/// mechanism is never worse than not having it (see `terminate_vetoed_or`).
pub fn certificate_masked(
    obj_scale: Number,
    unscaled_err: Number,
    threshold: Number,
    acceptable_tol: Number,
) -> bool {
    // A non-positive threshold is the documented opt-out; NaN is treated the
    // same way rather than silently enabling the mechanism.
    if threshold.is_nan() || threshold <= 0.0 {
        return false;
    }
    // Magnitude, not signed value: a negative `obj_scaling_factor` (the
    // documented way to maximize) is trivially below any positive threshold,
    // which would arm this on every maximization regardless of scale.
    obj_scale.abs() < threshold && unscaled_err > acceptable_tol
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
            // 1e-4 separates the falsely-certified problems (objective scale
            // pinned at the 1e-8 floor) from every recorded collateral case
            // (`hs1`/`hs38` at ~4e-2, the 19-problem list at ~1e-2). See
            // [`certificate_masked`].
            obj_scale_certificate_threshold: 1e-4,
            veto_fired: false,
            acceptable_veto_fired: false,
            veto_extra_iters: 0,
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

    /// Advance the acceptable-level streak, returning whether the run should
    /// terminate with `ConvergedToAcceptable`.
    ///
    /// Acceptable-level termination is **count-based**: it needs
    /// `acceptable_iter` *consecutive* qualifying iterates. The masked-scale
    /// veto (gh #200) suppresses that termination, so the count has to keep
    /// running underneath the suppression — otherwise the mechanism cannot know
    /// where the unvetoed run would have stopped.
    ///
    /// The subtle part, and an earlier bug: `masked` is **not constant over a
    /// run**. `obj_scale` is fixed, but the veto's other condition is
    /// `unscaled_err > acceptable_tol`, and that quantity crosses the bar
    /// during the endgame — the crossing *is* the veto lifting. A streak can
    /// therefore straddle the boundary. Keeping two disjoint counters (a real
    /// one and a shadow), each reset by the other's phase, silently discarded a
    /// streak the unvetoed run would have kept: fourteen unmasked qualifying
    /// iterates followed by one masked qualifying iterate left the real count at
    /// zero, where the baseline would have reached fifteen and stopped. The run
    /// then fell through to `max_iter` — with no snapshot armed, because the
    /// shadow had only just started — and returned a bare failure where the
    /// baseline returned `Solved_To_Acceptable_Level`. That is precisely the
    /// "never worse" guarantee failing.
    ///
    /// So there is **one** counter, advanced on `acceptable_now` regardless of
    /// `masked`. `masked` decides only what happens when it crosses the
    /// threshold: terminate, or record that a termination was refused here —
    /// which is exactly the iterate the unvetoed run would have returned.
    fn note_acceptable(&mut self, acceptable_now: bool, masked: bool) -> bool {
        if !acceptable_now {
            self.acceptable_count = 0;
            return false;
        }
        self.acceptable_count += 1;
        if self.acceptable_count < self.acceptable_iter {
            return false;
        }
        if masked {
            self.acceptable_veto_fired = true;
            false
        } else {
            true
        }
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
    fn certificate_vetoed(&self) -> bool {
        self.veto_fired
    }

    fn acceptable_certificate_vetoed(&self) -> bool {
        self.acceptable_veto_fired
    }

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
        let unscaled_err = cq_ref.curr_unscaled_nlp_error();
        // The gate asks whether *our* scaling clamped, not how the user chose
        // to scale their objective — see `certificate_masked`.
        let obj_scale = cq_ref.computed_obj_scaling_factor();
        drop(cq_ref);

        // gh #200: refuse a certificate the objective scaling has masked, and
        // keep iterating. A constant objective scale cancels out of the Newton
        // step and every line-search test is scale-invariant, so the continued
        // run follows exactly the trajectory an unscaled run would and reaches
        // the true minimum — at which point the unscaled error falls under
        // `acceptable_tol`, the veto lifts, and an honest strict certificate is
        // issued. Refusing to stop early is the whole intervention; the strict
        // tolerance in scaled space is untouched.
        if self.veto_fired || self.acceptable_veto_fired {
            self.veto_extra_iters += 1;
        }
        // Call the bet off once it has plainly not paid off, so a veto that can
        // never lift cannot cost an unbounded number of iterations. The refused
        // certificate is restored regardless, so this bounds cost, not
        // correctness.
        let budget_spent = self.veto_extra_iters > VETO_MAX_EXTRA_ITERS;
        // A non-finite objective disqualifies the veto outright. `passes_component_tols`
        // never inspects `f`, so a strict certificate can pass at an iterate whose
        // objective is NaN while its residuals are finite and tiny — and the unvetoed
        // run returns exactly that, NaN objective and all. Refusing it would arm a
        // snapshot the restore then declines (`honour_refused_certificate` requires a
        // finite objective), surfacing a failure where the baseline reported success.
        // Declining to engage keeps that case bit-identical to the baseline instead.
        // The acceptable-level side already had this property: finite `f` is a
        // precondition of qualifying there.
        let masked = curr_f.is_finite()
            && !budget_spent
            && certificate_masked(
                obj_scale,
                unscaled_err,
                self.obj_scale_certificate_threshold,
                self.acceptable_tol,
            );
        // Record a refusal only when a strict certificate was genuinely on the
        // table. `masked` alone is far broader — it holds on ordinary iterates
        // long before convergence — and using it would arm the fallback (and
        // snapshot an arbitrary mid-solve iterate) on runs that were never
        // about to stop.
        let refusing_strict =
            masked && self.passes_component_tols(nlp_err, dual_inf, constr_viol, compl_inf);
        if refusing_strict && !self.veto_fired {
            self.veto_fired = true;
            tracing::info!(
                obj_scale,
                unscaled_kkt_error = unscaled_err,
                scaled_nlp_error = nlp_err,
                threshold = self.obj_scale_certificate_threshold,
                "refusing a termination certificate masked by an extreme objective scale; \
                 continuing toward the true minimum (obj_scale_certificate_threshold=0 disables)"
            );
        }

        if !masked && self.passes_component_tols(nlp_err, dual_inf, constr_viol, compl_inf) {
            return ConvergenceStatus::Converged;
        }
        // `acceptable_iter == 0` disables acceptable-level termination
        // (upstream `IpOptErrorConvCheck.cpp:241`). See `check_convergence`.
        // The veto covers this branch too, so a refused strict certificate is
        // not merely swapped for an acceptable-level one at the same wrong
        // point. Acceptable-point *storage* is deliberately left un-vetoed —
        // that stashed point is the rollback target if the run later stalls.
        let acceptable_now = self.acceptable_iter > 0
            && self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f);
        if self.note_acceptable(acceptable_now, masked) {
            return ConvergenceStatus::ConvergedToAcceptable;
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
            // The surrogate here is a cheap PRE-FILTER, not the verdict. It is
            // a threshold on `||J^T c|| / max(1, ||c||)`, which is not
            // scale-invariant: under a row scaling `dc` the numerator carries
            // `dc^2` while the denominator clamps at 1, so an aggressive scaling
            // drives it to zero regardless of where the iterate is. That is how
            // HS13 from x0 = (1e4, 1e4) reached `5e-14` at a point whose
            // constraint violation was 0.51, and got reported infeasible.
            //
            // Retuning does not fix it. Measured over 800 corpus models, every
            // tolerance that fires on genuinely infeasible problems also
            // introduces new false infeasibility (>= 3 models at the smallest
            // viable value), and measuring the surrogate unscaled or
            // scale-invariantly does not separate the cases either. So the
            // surrogate stays as-is, and the claim the status actually makes --
            // that no local move reduces the violation -- is confirmed directly
            // before the verdict is issued.
            let stationarity = cq.borrow().curr_infeasibility_stationarity();
            if self.note_infeasible_stationary(constr_viol, stationarity) {
                if cq.borrow().infeasibility_descent_available() {
                    // Descent exists: not a stationary point of the violation,
                    // so the surrogate was wrong here. Drop the streak and keep
                    // solving.
                    self.infeas_streak = 0;
                } else {
                    return ConvergenceStatus::LocallyInfeasible;
                }
            }
        }
        // Time-budget gates. When the application installed a shared
        // [`Deadline`] (pounce#242) it is authoritative: it measures
        // global elapsed time from a fixed start instant, so it fires
        // correctly even inside the restoration inner IPM, whose fresh
        // `timing.overall_alg` is never started. Absent a deadline (the
        // direct-driver / unit-test path), fall back to the `overall_alg`
        // timer, which `IpoptApplication` starts at the top of
        // `optimize_constrained`; `live_*` returns the running elapsed
        // without forcing a `start/end` cycle. Upstream
        // `IpOptErrorConvCheck.cpp::CheckConvergence` reads the
        // application-level start time similarly.
        let d = data.borrow();
        if let Some(deadline) = d.deadline.as_ref() {
            match deadline.exceeded() {
                Some(pounce_common::timing::DeadlineKind::Cpu) => {
                    return ConvergenceStatus::CpuTimeExceeded;
                }
                Some(pounce_common::timing::DeadlineKind::Wall) => {
                    return ConvergenceStatus::WallTimeExceeded;
                }
                None => {}
            }
        } else {
            let timing = &d.timing;
            if timing.overall_alg.live_cpu_time() >= self.max_cpu_time {
                return ConvergenceStatus::CpuTimeExceeded;
            }
            if timing.overall_alg.live_wallclock_time() >= self.max_wall_time {
                return ConvergenceStatus::WallTimeExceeded;
            }
        }
        ConvergenceStatus::Continue
    }

    fn current_passes_strict(
        &self,
        nlp_err: Number,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> bool {
        // The strict per-component gate of `check_convergence_with_state`, minus
        // the masking veto — see the trait doc. Unscaled per-component residuals,
        // matching that method (the `*_tol` triplet is defined on the
        // user-original residuals).
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        drop(cq_ref);
        self.passes_component_tols(nlp_err, dual_inf, constr_viol, compl_inf)
    }

    fn tol_or_default(&self) -> Number {
        self.tol
    }

    fn acceptable_constr_viol_tol_or_default(&self) -> Number {
        self.acceptable_constr_viol_tol
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
    fn a_non_finite_objective_disqualifies_the_veto() {
        // `passes_component_tols` never inspects `f`, so a strict certificate can
        // pass at an iterate whose objective is NaN while its residuals are finite
        // and tiny — and the unvetoed run returns exactly that. Refusing it would
        // arm a snapshot that the restore then declines (it requires a finite
        // objective), surfacing a failure where the baseline reported success:
        // a never-worse violation, on the one path where the objective is not
        // usable as a tiebreak.
        let c = OptErrorConvCheck {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // The residuals alone say "converged"; the objective says nothing usable.
        assert!(c.passes_component_tols(1e-12, 1e-9, 0.0, 0.0));
        // The masked predicate itself is unchanged — the finiteness gate lives at
        // the call site, where `curr_f` is in hand.
        assert!(certificate_masked(
            1e-8,
            8.4e-1,
            c.obj_scale_certificate_threshold,
            c.acceptable_tol
        ));
        // Both the guard's inputs behave as the call site composes them.
        for bad in [Number::NAN, Number::INFINITY, Number::NEG_INFINITY] {
            assert!(!bad.is_finite(), "{bad} should disqualify the veto");
        }
        assert!((1.0_f64).is_finite());
    }

    #[test]
    fn acceptable_streak_survives_a_masked_boundary_mid_streak() {
        // gh #200. `masked` is not constant over a run: it also depends on the
        // unscaled error crossing `acceptable_tol`, and that crossing is exactly
        // what happens during the endgame. So an acceptable-level streak can
        // straddle the boundary.
        //
        // The earlier implementation kept two disjoint counters, each reset by
        // the other's phase. Fourteen unmasked qualifying iterates followed by
        // one masked qualifying iterate left the real count at 0 while the
        // unvetoed run would have reached 15 and stopped — so the run fell
        // through to `max_iter` and returned a bare failure where the baseline
        // returned `Solved_To_Acceptable_Level`, with no snapshot armed to roll
        // back to. Never-worse, violated.
        let mut c = OptErrorConvCheck {
            acceptable_iter: 15,
            ..Default::default()
        };
        // 14 qualifying iterates while unmasked: no termination yet.
        for i in 0..14 {
            assert!(!c.note_acceptable(true, false), "terminated early at {i}");
        }
        // The 15th qualifies too, but the veto is now engaged. The streak must
        // be honoured — recorded as a refused termination, not discarded.
        assert!(
            !c.note_acceptable(true, true),
            "a masked iterate must not terminate the run"
        );
        assert!(
            c.acceptable_veto_fired,
            "the streak crossed `acceptable_iter` while masked, so a termination was \
             refused here and must be recorded — otherwise the fallback has nothing to \
             restore and the run returns a bare failure"
        );

        // The mirror direction: a streak that begins masked and finishes
        // unmasked must terminate on the same iterate the baseline would.
        let mut c = OptErrorConvCheck {
            acceptable_iter: 15,
            ..Default::default()
        };
        for _ in 0..14 {
            assert!(!c.note_acceptable(true, true));
        }
        assert!(
            c.note_acceptable(true, false),
            "the veto lifted with the streak already at 14; the 15th qualifying iterate \
             must terminate exactly as it would without the mechanism"
        );

        // And a non-qualifying iterate still breaks the streak, in either phase.
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        assert!(!c.note_acceptable(true, false));
        assert!(!c.note_acceptable(false, true));
        assert_eq!(
            c.acceptable_count, 0,
            "a non-qualifying iterate resets the streak"
        );
        assert!(!c.note_acceptable(true, false));
        assert!(!c.note_acceptable(true, false));
        assert!(
            c.note_acceptable(true, false),
            "3 consecutive qualifying iterates terminate"
        );
    }

    #[test]
    fn certificate_masked_needs_both_an_extreme_scale_and_a_non_stationary_point() {
        // gh #200. Both conditions are load-bearing, and each was independently
        // shown to be insufficient on the benchmark suite.
        let (th, atol) = (1e-4, 1e-6);

        // The reported failure: scale pinned at the 1e-8 floor, unscaled error
        // 0.84 — the strict test passed in scaled space at `quartc` obj 248.88.
        assert!(certificate_masked(1e-8, 8.4e-1, th, atol));

        // An ordinary objective scale is never second-guessed, however large
        // the unscaled error. Keying on the error alone effectively tightens
        // `tol` by `1/df` and regressed hs1/hs38 (scale ~4e-2).
        assert!(!certificate_masked(4e-2, 8.4e-1, th, atol));
        assert!(!certificate_masked(1.0, 1e3, th, atol));

        // An extreme scale at a point that really is stationary is fine — this
        // is what lifts the veto once the continued run reaches the minimum.
        assert!(!certificate_masked(1e-8, 1e-9, th, atol));

        // Boundaries: strictly below the scale threshold, strictly above the
        // error tolerance.
        assert!(!certificate_masked(th, 1.0, th, atol));
        assert!(!certificate_masked(1e-8, atol, th, atol));

        // `0` disables the mechanism outright (the documented opt-out) — the
        // most extreme possible inputs must not trip it.
        assert!(!certificate_masked(1e-30, 1e30, 0.0, atol));
        // A negative threshold is treated as disabled rather than as "always".
        assert!(!certificate_masked(1e-30, 1e30, -1.0, atol));
    }

    #[test]
    fn veto_blocks_both_strict_and_acceptable_termination() {
        // A refused strict certificate must not simply reappear as an
        // acceptable-level one at the same wrong point, so the veto covers both
        // branches. Exercised through the pure predicates the two branches
        // share, since a full `check_convergence_with_state` needs a live cq.
        let c = OptErrorConvCheck {
            tol: 1e-8,
            acceptable_tol: 1e-6,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // The gh #200 iterate: passes the strict test in scaled space...
        assert!(c.passes_component_tols(1e-9, 8.4e-1, 0.0, 0.0));
        // ...and the veto is what withholds it.
        assert!(certificate_masked(
            1e-8,
            8.4e-1,
            c.obj_scale_certificate_threshold,
            c.acceptable_tol
        ));
        // Default threshold is the documented 1e-4, and the veto starts clear.
        assert_eq!(c.obj_scale_certificate_threshold, 1e-4);
        assert!(!c.veto_fired);
        assert!(!ConvCheck::certificate_vetoed(&c));
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
