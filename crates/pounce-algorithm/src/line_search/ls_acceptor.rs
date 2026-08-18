//! Line-search acceptor trait — port of `IpBacktrackingLSAcceptor.hpp`
//! and `IpLineSearch.hpp`.

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::iterates_vector::IteratesVector;
use crate::line_search::filter_acceptor::AcceptDecision;
use crate::restoration::OrigProgressCallback;
use pounce_common::types::Number;

/// Acceptor side of the backtracking line search. Concrete impls:
/// [`super::filter_acceptor::FilterLsAcceptor`] (Phase 7),
/// `PenaltyLsAcceptor` (Phase 10), `CGPenaltyLsAcceptor` (Phase 10).
///
/// The driver calls `check_trial_point` on each backtracking step.
/// Acceptors that need the trial-iterate components (rather than
/// scalar `(theta, phi)`) can extend this surface in later phases —
/// the filter acceptor only needs the four scalars upstream feeds in
/// at line `IpFilterLSAcceptor.cpp:CheckAcceptabilityOfTrialPoint`.
pub trait BacktrackingLsAcceptor {
    /// Reset acceptor state for a new outer iteration.
    fn reset(&mut self);

    /// Hook called once per outer iteration, after the search direction
    /// `delta` has been computed and before the α-loop. Mirrors
    /// `IpPenaltyLSAcceptor.cpp:InitThisLineSearch` — the penalty
    /// acceptor uses it to snapshot reference (θ, φ, ∇φᵀδ, δᵀWδ) and to
    /// bump the penalty parameter ν. Default: no-op (filter acceptor
    /// has nothing to cache between α-loop iterations).
    fn init_this_line_search(
        &mut self,
        _data: &IpoptDataHandle,
        _cq: &IpoptCqHandle,
        _delta: &IteratesVector,
    ) {
    }

    /// The penalty acceptor's registered constants, in the order
    /// `(nu_init, nu_inc, rho, eta_penalty)`. `None` for acceptors that
    /// have no penalty parameter (the filter acceptor).
    ///
    /// All four are registered options whose only consumer lives inside
    /// the acceptor, and until #551 none of them had a read site. With
    /// the acceptor reachable only as `dyn BacktrackingLsAcceptor`, the
    /// furthest a test could follow such an option was the builder
    /// struct — one hop short of the object that uses the value, which
    /// is exactly the gap that let `limited_memory_initialization` look
    /// wired while it was not (#677). This closes that hop.
    fn penalty_parameters(&self) -> Option<(Number, Number, Number, Number)> {
        None
    }

    /// Compute the minimum primal step length below which the
    /// driver should declare a tiny step / hand off to restoration.
    /// Mirrors `IpFilterLSAcceptor.cpp:CalculateAlphaMin` — the value
    /// depends on the current `(theta, d_phi)` pair (the directional
    /// derivative of the barrier objective along the search step) and
    /// on the acceptor's lazily-initialised `theta_min`. Default impl
    /// returns 0.0 so non-filter acceptors degenerate to the driver's
    /// own absolute `alpha_min` floor.
    fn calc_alpha_min(&mut self, _d_phi: Number, _theta: Number) -> Number {
        0.0
    }

    /// Decide whether the trial `(theta_trial, phi_trial)` at primal
    /// step `alpha_primal` is acceptable, given the current iterate's
    /// `(theta, phi)` and the directional derivative `d_phi`.
    /// Default: always accept (lets stub acceptors compose without
    /// interfering with the driver's α-loop).
    ///
    /// Mutable receiver so concrete acceptors (notably
    /// [`super::filter_acceptor::FilterLsAcceptor`]) can record per-trial
    /// state used by the filter-reset heuristic
    /// (`IpFilterLSAcceptor.cpp:407-433`).
    fn check_trial_point(
        &mut self,
        _alpha_primal: Number,
        _theta: Number,
        _phi: Number,
        _d_phi: Number,
        _theta_trial: Number,
        _phi_trial: Number,
    ) -> AcceptDecision {
        AcceptDecision::Accept
    }

    /// Post-accept hook — port of
    /// `IpFilterLSAcceptor::UpdateForNextIteration`. Both decides the
    /// `info_alpha_primal_char` tag *and* augments the filter when
    /// upstream would. Returns:
    ///
    /// * `'f'` — F-type Armijo step (`IsFtype && ArmijoHolds`); filter
    ///   is **not** augmented.
    /// * `'h'` — anything else (`!IsFtype || !ArmijoHolds`); filter
    ///   **is** augmented with `(theta_add, phi_add) = ((1 - γ_θ)·θ_ref,
    ///   φ_ref - γ_φ·θ_ref)`.
    ///
    /// The driver calls this once per accepted step, after
    /// `check_trial_point` returns Accept and before
    /// `accept_trial_point` promotes `trial → curr`. Default impl
    /// returns `'h'` (no filter), so non-filter acceptors remain valid.
    fn update_for_next_iteration(
        &mut self,
        _alpha_primal: Number,
        _theta: Number,
        _phi: Number,
        _d_phi: Number,
        _phi_trial: Number,
    ) -> char {
        'h'
    }

    /// Build the orig-progress callback the inner restoration IPM
    /// should consult to decide whether the recovered iterate is
    /// acceptable to *this* (outer) acceptor's filter and reference
    /// iterate. Mirrors upstream
    /// `IpRestoFilterConvCheck::SetOrigLSAcceptor` /
    /// `TestOrigProgress`. Default returns `None` — penalty / CG-penalty
    /// acceptors do not gate restoration on a filter, so they fall
    /// through to the kappa-reduction-only guard.
    ///
    /// `reference_theta` and `reference_barr` are the outer iterate's
    /// `(curr_constraint_violation, curr_barrier_obj)` at restoration
    /// entry; `obj_max_inc` is the upstream `obj_max_inc` option
    /// (default 5.0).
    fn make_orig_progress_check(
        &self,
        _reference_theta: Number,
        _reference_barr: Number,
        _obj_max_inc: Number,
    ) -> Option<OrigProgressCallback> {
        None
    }

    /// Hook called by the algorithm immediately before invoking the
    /// restoration phase — port of
    /// `IpFilterLSAcceptor::PrepareRestoPhaseStart` →
    /// `AugmentFilter` (`IpFilterLSAcceptor.cpp:898-901`, called from
    /// `IpBacktrackingLineSearch.cpp:566`). The filter acceptor
    /// augments the filter with the resto-entry iterate's shrunk
    /// envelope `((1 - γ_θ)·θ_ref, φ_ref - γ_φ·θ_ref)`. After
    /// restoration recovers, the outer's Newton step is then forced
    /// by the filter to make real progress vs the entry point —
    /// without this, the outer can accept null-progress 'h' steps
    /// and re-enter restoration (observed on DECONVBNE: 323 R-accepts
    /// vs ipopt's 21). Default: no-op for non-filter acceptors.
    fn prepare_resto_phase_start(&mut self, _reference_theta: Number, _reference_barr: Number) {}

    /// Override the filter acceptor's `theta_max_fact` (default 1e4).
    /// Used by the resto sub-IPM wiring to bump the gate to 1e8, which
    /// mirrors upstream `IpRestoMinC_1Nrm.cpp:91`
    /// (`resto.theta_max_fact = 1e8`). Without this override the inner
    /// IPM's first line search caps `theta_max = 1e4` (since reference
    /// θ ≈ 0 after slack init), and the first non-trivial trial whose
    /// resto-NLP θ_trial exceeds 1e4 is rejected at the `theta_max`
    /// gate before reaching f-type/Armijo dispatch. Default: no-op for
    /// non-filter acceptors.
    fn set_theta_max_fact(&mut self, _theta_max_fact: Number) {}

    /// Tell the acceptor how many constraint rows back `theta`'s 1-norm
    /// (`dim(c) + dim(d - s)`). The filter acceptor floors its
    /// `theta_max` reference at `theta_max_row_scale_kappa * rows` so
    /// the ceiling does not degenerate to the bare constant `1e4` on a
    /// large-`m` model with a feasible starting point. Called once per
    /// solve, before the first `theta_max` lock. Default: no-op.
    fn set_theta_rows(&mut self, _rows: Number) {}

    /// Override the filter acceptor's `theta_max_row_scale_kappa`.
    /// Used by the resto sub-IPM wiring to set it to `0`, i.e. to opt
    /// the inner IPM out of the row-count floor: upstream already
    /// covers the resto phase's version of the same degeneracy with
    /// its hard-coded `theta_max_fact = 1e8`
    /// (`IpRestoMinC_1Nrm.cpp:91`), and stacking the row floor on top
    /// of that would push the inner ceiling to `1e8 · m` — effectively
    /// removing it. Default: no-op for non-filter acceptors.
    fn set_theta_max_row_scale_kappa(&mut self, _kappa: Number) {}

    /// Override the filter acceptor's `theta_max_adaptive_trigger`.
    /// Used by the resto sub-IPM wiring to set it to `0` for the same
    /// reason as [`Self::set_theta_max_row_scale_kappa`]: the inner IPM
    /// already runs at `theta_max_fact = 1e8`, which is upstream's own
    /// hard-coded version of this rescue, so letting the adaptive rule
    /// ratchet on top of that would raise an already-enormous ceiling
    /// further. Default: no-op for non-filter acceptors.
    fn set_theta_max_adaptive_trigger(&mut self, _trigger: u32) {}
}
