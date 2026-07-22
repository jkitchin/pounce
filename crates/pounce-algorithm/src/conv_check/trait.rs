//! `ConvCheck` trait — port of `IpConvCheck.hpp`.
//!
//! Upstream `ConvergenceCheck::CheckConvergence` reads the NLP error
//! and iter count off `IpData()`/`IpCq()`. The default trait method
//! pushes that read to the caller (the main loop in `ipopt_alg.rs`)
//! so simple convergence policies stay pure scalar state machines
//! over `(nlp_err, iter_count)`. The richer
//! [`ConvCheck::check_convergence_with_state`] entry point exposes
//! the live `(IpoptData, IpoptCq)` so policies that need iterate
//! components — notably the restoration-side
//! `RestoFilterConvergenceCheck::TestOrigProgress` — can read them.
//! Default impl just delegates to the scalar method, preserving
//! backwards compatibility for every existing impl.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    Continue,
    Converged,
    /// Converged to the looser `acceptable_*` tolerance band rather
    /// than the tight `tol` — upstream `CONVERGED_TO_ACCEPTABLE_POINT`.
    /// Maps to `SolverReturn::StopAtAcceptablePoint` →
    /// `ApplicationReturnStatus::SolvedToAcceptableLevel`.
    ConvergedToAcceptable,
    MaxIterExceeded,
    /// `max_cpu_time` budget reached. Maps to
    /// `SolverReturn::CpuTimeExceeded` → `MaximumCpuTimeExceeded`.
    CpuTimeExceeded,
    /// `max_wall_time` budget reached. Maps to
    /// `SolverReturn::WallTimeExceeded` → `MaximumWallTimeExceeded`.
    WallTimeExceeded,
    /// Rapid infeasibility detection fired — the iterate is
    /// converging to a stationary point of the constraint violation
    /// with the violation bounded away from zero. Maps to
    /// `SolverReturn::LocalInfeasibility`.
    LocallyInfeasible,
    Failed,
}

pub trait ConvCheck {
    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus;

    /// State-aware convergence check. The main loop calls this on
    /// every iteration so policies that need access to the iterate
    /// (e.g. `RestoConvCheckAdapter`'s orig-NLP `inf_pr` evaluation
    /// for the kappa-reduction early-exit) can read `data.curr` and
    /// the cq layer. Default impl delegates to
    /// [`Self::check_convergence`], so scalar-only policies don't
    /// need to override.
    /// Whether this policy ever refused a termination certificate it judged
    /// masked by an extreme objective scale (gh #200).
    ///
    /// The main loop reads it on the failure exits: a run that was held back
    /// from stopping must not end up *worse off* than it would have been, so
    /// when it later stalls the stored acceptable point is restored rather
    /// than a bare failure surfaced. Policies that never veto keep the
    /// default `false` and the failure paths behave exactly as before.
    fn certificate_vetoed(&self) -> bool {
        false
    }

    /// Whether this policy ever refused an *acceptable-level* termination it
    /// judged masked (gh #200). Undone differently from a strict refusal — see
    /// `OptErrorConvCheck::acceptable_veto_fired`.
    fn acceptable_certificate_vetoed(&self) -> bool {
        false
    }

    fn check_convergence_with_state(
        &mut self,
        nlp_err: Number,
        iter_count: Index,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
    ) -> ConvergenceStatus {
        self.check_convergence(nlp_err, iter_count)
    }

    /// Whether the supplied `nlp_err` is at or below the acceptable
    /// tolerance — port of upstream
    /// `OptimalityErrorConvergenceCheck::CurrentIsAcceptable`. Used by
    /// the main loop to gate `StoreAcceptablePoint` /
    /// `RestoreAcceptablePoint`. Default returns `false` so policies
    /// that don't track an acceptable level (e.g. resto-of-resto inner
    /// adapters) silently skip the rollback machinery.
    fn current_is_acceptable(&self, _nlp_err: Number) -> bool {
        false
    }

    /// State-aware acceptance check. Mirrors upstream
    /// `OptimalityErrorConvergenceCheck::CurrentIsAcceptable` which
    /// reads the per-component residuals and current `f` to gate the
    /// `acceptable_dual_inf_tol` / `acceptable_constr_viol_tol` /
    /// `acceptable_compl_inf_tol` / `acceptable_obj_change_tol`
    /// triplet. Default delegates to the scalar [`Self::current_is_acceptable`].
    fn current_is_acceptable_with_state(
        &self,
        nlp_err: Number,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
    ) -> bool {
        self.current_is_acceptable(nlp_err)
    }

    /// Record the current objective at the iterate the main loop just
    /// stashed as the latest "acceptable point" — mirrors upstream
    /// `OptimalityErrorConvergenceCheck::SetCurrAcceptableF`. The
    /// recorded value feeds the `acceptable_obj_change_tol` stability
    /// cross-check on subsequent iterates. Default no-op for policies
    /// that don't track acceptable points.
    fn set_curr_acceptable_obj(&mut self, _obj: Number) {}

    /// Outer NLP convergence tolerance, as used by the main loop's
    /// almost-feasible bypass guard (port of
    /// `IpBacktrackingLineSearch.cpp:580`). Default `1e-8` matches
    /// upstream's default `tol`.
    fn tol_or_default(&self) -> Number {
        1e-8
    }

    /// Acceptable-level primal-feasibility band `acceptable_constr_viol_tol`, in
    /// the unscaled max-norm space `curr_unscaled_primal_infeasibility_max`
    /// reports. Read by the best-acceptable fallback's feasibility-aware ranking
    /// (gh #267), which caps it at the upstream default so a user-widened band
    /// cannot let the fallback spend feasibility to buy objective. Default
    /// `1e-2` matches upstream's default `acceptable_constr_viol_tol`; policies
    /// that track no such tolerance keep it.
    fn acceptable_constr_viol_tol_or_default(&self) -> Number {
        1e-2
    }

    /// Live-update a named convergence tolerance mid-solve, for the
    /// debugger's in-place option hot-swap. Returns `true` if `name`
    /// matched a tolerance this policy owns (so the caller can report
    /// whether it took). Default: this policy exposes no live
    /// tolerances → `false`.
    fn set_tolerance(&mut self, _name: &str, _value: Number) -> bool {
        false
    }
}
