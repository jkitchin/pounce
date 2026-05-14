//! `RestorationPhase` trait — port of `IpRestoPhase.hpp`.
//!
//! Defined here in `pounce-algorithm` (rather than `pounce-restoration`)
//! so that [`crate::ipopt_alg::IpoptAlgorithm`] can call into it without
//! creating a circular crate dependency. Concrete impls (the default
//! `MinC1NormRestoration`, the rare `RestoRestorationPhase`) live in
//! `pounce-restoration` and `impl RestorationPhase for ...`.
//!
//! Called by the main loop when the line search exhausts its alpha
//! reductions without acceptance (or by the iterate initializer when
//! `start_with_resto = true`). On success the impl writes a recovered
//! iterate to `data.trial` and the main loop accepts it; on failure the
//! main loop surfaces `SolverReturn::RestorationFailure`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::aug_system_solver::AugSystemSolver;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::rc::Rc;

/// Callback that the inner restoration IPM consults at every iteration
/// to decide whether the recovered iterate is acceptable to the *outer*
/// algorithm's filter and reference iterate. Mirrors upstream
/// `IpRestoFilterConvCheck::TestOrigProgress`
/// (`IpRestoFilterConvCheck.cpp:53-80`): given `(orig_trial_barr,
/// orig_trial_theta)` evaluated at the inner iterate's `(x_orig, s)`
/// slice, returns `true` iff
///
/// 1. the pair is acceptable to the outer filter, AND
/// 2. the pair is acceptable to the outer reference iterate (with the
///    rapid-barrier-increase guard disabled — `force_armijo=true` /
///    `called_from_restoration=true`).
///
/// Constructed by [`crate::line_search::ls_acceptor::BacktrackingLsAcceptor::make_orig_progress_check`]
/// at restoration entry, with the outer filter cloned and the outer
/// reference `(theta, barr)` snapshotted in the closure.
pub type OrigProgressCallback = Box<dyn Fn(Number, Number) -> bool>;

/// Outcome of a restoration attempt. Mirrors upstream's `bool` return
/// from `RestorationPhase::PerformRestoration` plus the in-band
/// `info_skip_output` / `iter_count` side-effects that the impl writes
/// to `data` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestorationOutcome {
    /// Resto succeeded; outer loop should `accept_trial_point` and
    /// continue. The impl has written the recovered iterate into
    /// `data.trial`, set `info_skip_output = true`, and updated the
    /// info counters.
    Recovered,
    /// Resto failed. Outer loop maps this to
    /// `SolverReturn::RestorationFailure`.
    Failed,
    /// The inner sub-IPM converged its KKT system but the orig-NLP
    /// constraint violation at the converged point is still well above
    /// `tol`. Mirrors the `LOCALLY_INFEASIBLE` exception thrown from
    /// `IpRestoConvCheck.cpp:240`. Outer loop maps this to
    /// `SolverReturn::LocalInfeasibility`.
    LocallyInfeasible,
}

pub trait RestorationPhase {
    /// Inner-IPM iteration count from the most recent
    /// `perform_restoration` call. Read by `IpoptAlgorithm` for the
    /// pounce#12 audit counters in `SolveStatistics`. Default 0; the
    /// concrete `MinC1NormRestoration` impl stashes
    /// `RestoSolveResult::iter_count` and returns it here.
    fn last_inner_iter_count(&self) -> pounce_common::types::Index {
        0
    }

    /// Drive a feasibility-restoration sub-solve. The impl reads the
    /// outer iterate from `data.curr`, the original NLP from `nlp`,
    /// uses `aug_solver` for any post-success multiplier-recomputation
    /// least-square solve, and on success writes the recovered iterate
    /// into `data.trial`. Default returns
    /// [`RestorationOutcome::Failed`] — the trait surface is uniform
    /// for `AlgBuilder` even when no concrete restoration is wired.
    fn perform_restoration(
        &mut self,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        _nlp: &Rc<RefCell<dyn IpoptNlp>>,
        _aug_solver: &mut dyn AugSystemSolver,
    ) -> RestorationOutcome {
        RestorationOutcome::Failed
    }

    /// Inject the orig-progress callback the inner IPM should consult at
    /// every iteration. Mirrors upstream
    /// `IpRestoFilterConvCheck::SetOrigLSAcceptor` (the outer line
    /// search hands its acceptor to the resto conv check at restoration
    /// entry). Default no-op so non-filter-aware drivers compose.
    fn set_orig_progress_check(&mut self, _cb: Option<OrigProgressCallback>) {}
}
