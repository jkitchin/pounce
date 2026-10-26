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

/// Restoration's release test as the restoration phase consumes it: given a
/// restoration iterate's `(x, s)` in the **outer** problem's space, may the
/// outer phase resume from it?
///
/// Built by the outer algorithm, which owns the quantities the outer filter
/// holds. It evaluates the outer problem's own barrier objective and 1-norm
/// constraint violation at `(x, s)` (`trial_barrier_obj`,
/// `trial_constraint_violation`, at the outer `mu`), exactly as upstream's
/// `IpRestoFilterConvCheck::TestOrigProgress` does by setting the original
/// problem's trial point, and hands that pair to the
/// [`OrigProgressCallback`]. The restoration crate cannot compute the pair
/// itself without re-deriving the barrier terms, damping included, that the
/// filter's entries were made with; it used to pass the plain objective and
/// the max-norm violation instead, which the filter does not hold
/// (gh#1042).
pub type OrigProgressCheck =
    Box<dyn Fn(&dyn pounce_linalg::Vector, &dyn pounce_linalg::Vector) -> bool>;

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
    /// The user's intermediate callback returned `false` from a
    /// restoration-phase fire (gh#645). Outer loop maps this to
    /// `SolverReturn::UserRequestedStop` **without** promoting the
    /// staged trial point, so the solve hands back the last iterate
    /// accepted for the *original* NLP rather than a point of the
    /// restoration subproblem — the same discipline the pounce#244
    /// deadline exit already follows.
    UserRequestedStop,
    /// The original NLP is square and the restoration phase reached a
    /// point feasible for it to `constr_viol_tol`. Port of the
    /// `FEASIBILITY_PROBLEM_SOLVED` throw at `IpRestoMinC_1Nrm.cpp:269`;
    /// the outer loop maps this to `SolverReturn::FeasiblePointFound`
    /// (`IpIpoptAlg.cpp:542`) after recomputing the multipliers of the
    /// feasibility problem. The impl has already promoted the recovered
    /// point to `data.curr`.
    FeasiblePointFound,
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

    /// Whether the most recent `perform_restoration` call returned
    /// [`RestorationOutcome::Recovered`] with a point its own convergence
    /// check never released (gh#1042): the restoration sub-problem
    /// converged at a point feasible for the original NLP, but that point
    /// either failed the outer filter / reference-iterate test or did not
    /// cut the violation by `kappa_resto`. Upstream fails restoration there
    /// (`RESTORATION_CONVERGED_TO_FEASIBLE_POINT`,
    /// `IpRestoConvCheck.cpp:227-234`); pounce hands the point back, and
    /// the outer loop uses this to tell when that hand-back did not pay
    /// off. Default `false`.
    fn last_recovery_unreleased(&self) -> bool {
        false
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

    /// Forward the outer interactive debugger onto the restoration inner
    /// IPM so the same debugger can step the sub-solve. Default no-op.
    fn set_debug_hook(
        &mut self,
        _hook: Option<std::rc::Rc<std::cell::RefCell<dyn crate::debug::DebugHook>>>,
    ) {
    }

    /// Forward the user's TNLP onto the restoration inner IPM so its
    /// `intermediate_callback` fires from the sub-solve too (gh#645).
    /// Default no-op, like [`Self::set_debug_hook`] above, which this
    /// deliberately mirrors — the debugger took the same route first.
    ///
    /// Without this the callback fires only from the outer loop, so a
    /// caller is blind for the whole of restoration. That is the phase
    /// most likely to overrun a control period, which makes it the
    /// phase a real-time caller most needs to be able to abort in.
    ///
    /// The inner IPM iterates on the min-C1-norm feasibility
    /// subproblem, so the stats those fires carry describe *that*
    /// problem, not the user's NLP. `AlgorithmMode::RestorationPhaseMode`
    /// on each such fire is what makes them interpretable, and
    /// [`crate::ipopt_alg::IpoptAlgorithm::fires_as_restoration`] is
    /// what sets it — see the note there on why the live-inspector
    /// context is deliberately *not* installed for these fires.
    fn set_intermediate_tnlp(
        &mut self,
        _tnlp: Option<std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::tnlp::TNLP>>>,
    ) {
    }

    /// Inject the orig-progress callback the inner IPM should consult at
    /// every iteration. Mirrors upstream
    /// `IpRestoFilterConvCheck::SetOrigLSAcceptor` (the outer line
    /// search hands its acceptor to the resto conv check at restoration
    /// entry). Default no-op so non-filter-aware drivers compose.
    fn set_orig_progress_check(&mut self, _cb: Option<OrigProgressCheck>) {}

    /// Propagate the outer algorithm's per-iteration print gate to the
    /// restoration driver so the nested restoration IPM honors
    /// `print_level == 0` instead of leaking its `r`-suffixed iteration
    /// table to stdout. Default no-op for drivers without a nested IPM.
    fn set_print_iter_output(&mut self, _enabled: bool) {}
}
