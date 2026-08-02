//! Restoration-phase convergence checks.
//!
//! Three flavours mirror upstream:
//!
//! * `RestoConvCheck` (`IpRestoConvCheck.{hpp,cpp}`) — base.
//! * `RestoFilterConvCheck` (`IpRestoFilterConvCheck.{hpp,cpp}`) —
//!   used when the outer phase uses the filter line search.
//! * `RestoPenaltyConvCheck` (`IpRestoPenaltyConvCheck.{hpp,cpp}`) —
//!   used when the outer phase uses the penalty line search.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoConvergenceStatus {
    Continue,
    Converged,
    /// Restoration reached the sub-problem's *acceptable* level and the
    /// recovered point is feasible for the original NLP. Upstream's
    /// `CONVERGED_TO_ACCEPTABLE_POINT` on the square-problem arm of
    /// `IpRestoConvCheck.cpp:225`.
    ConvergedToAcceptable,
    /// The restoration sub-problem converged at a point whose original-NLP
    /// constraint violation is still bounded away from zero — upstream's
    /// `LOCALLY_INFEASIBLE` throw (`IpRestoConvCheck.cpp:240`).
    LocallyInfeasible,
    MaxIterExceeded,
    UserStop,
}

/// The verdict upstream renders once the restoration sub-problem has
/// converged in its own right — layer 2 of `IpRestoConvCheck::
/// CheckConvergence` (`IpRestoConvCheck.cpp:200-240`).
///
/// Layer 1 (the `kappa_resto` reduction guard, the square-problem fast
/// path, and the outer-filter acceptance test) answers *can the trial
/// point leave restoration?*. Layer 2 answers the complementary question
/// — *has restoration provably done everything it can?* — and it is the
/// only thing that bounds a restoration which can never satisfy layer 1.
/// Without it, a sub-problem sitting at its own KKT point is
/// indistinguishable from one still making progress, and restoration
/// grinds to an iteration cap that reports the wrong status (pounce#438).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoOrigVerdict {
    /// The original NLP is nearly feasible here and the sub-problem
    /// tolerance can still be tightened: shrink it by `1e-2` and keep
    /// iterating (`IpRestoConvCheck.cpp:212-217`). Bounded by
    /// construction — see [`resto_orig_verdict`].
    TightenAndContinue,
    /// Square problem, feasible within `constr_viol_tol`
    /// (`IpRestoConvCheck.cpp:222-226`).
    ConvergedToAcceptable,
    /// Restoration converged to a feasible point of the original NLP that
    /// the outer phase nevertheless would not accept
    /// (`IpRestoConvCheck.cpp:227-234`).
    ConvergedToFeasiblePoint,
    /// The sub-problem is at a stationary point of the constraint
    /// violation whose residual is bounded away from zero
    /// (`IpRestoConvCheck.cpp:235-241`).
    LocallyInfeasible,
}

/// Pure core of upstream's layer-2 four-way verdict
/// (`IpRestoConvCheck.cpp:200-240`), evaluated once the restoration
/// sub-problem's own convergence check has reported
/// `CONVERGED` / `CONVERGED_TO_ACCEPTABLE_POINT`.
///
/// * `orig_trial_inf_pr` — original-NLP constraint violation at the
///   converged restoration iterate, in **unscaled** (user) units,
///   because every threshold below is an absolute user-facing magnitude.
/// * `tol` — the *restoration sub-problem's* current tolerance, which
///   [`RestoOrigVerdict::TightenAndContinue`] shrinks.
/// * `orig_tol` — the original NLP's `tol`, the floor the tightening arm
///   is bounded against.
/// * `orig_constr_viol_tol` — the original NLP's `constr_viol_tol`,
///   consulted only on the square-problem arm.
///
/// The tightening arm terminates by construction: it fires only while
/// `tol > 1e-1 · orig_tol` and shrinks `tol` by `1e-2` each time, so from
/// `tol == orig_tol` it fires exactly once (`1e-2 · orig_tol` is no
/// longer `> 1e-1 · orig_tol`). Without that guard the arm is an infinite
/// loop, which is why it is part of the port rather than an optimisation.
pub fn resto_orig_verdict(
    orig_trial_inf_pr: f64,
    orig_trial_rel_inf_pr: f64,
    tol: f64,
    orig_tol: f64,
    orig_constr_viol_tol: f64,
    rel_viol_threshold: f64,
    is_square_problem: bool,
) -> RestoOrigVerdict {
    // Every arm but the last asks "is this nearly feasible for the orig NLP?"
    // and upstream asks it with an absolute threshold alone. That is not
    // scale-invariant: `s*g(x) >= s*b` is the same constraint for every s > 0,
    // but its residual carries s. Measured on `inf_372` (0 <= x <= 0.6 with
    // s*x >= s*0.7, infeasible by 14% of the row at every s): the violation is
    // 1.0e-1 at s = 1 and 1.0e-9 at s = 1e-8, so the same model clears
    // `1e2 * tol` at small s and the arms read a decisively infeasible point as
    // nearly feasible. gh #390 (residual of #387) settled this for the outer
    // check; the same invariant is needed the moment layer 2 judges
    // feasibility. Rows with no declared magnitude contribute 0 and fall back
    // to the absolute test, which is already invariant for them.
    let nearly_feasible =
        orig_trial_inf_pr <= 1e2 * tol && orig_trial_rel_inf_pr <= rel_viol_threshold;
    // `IpRestoConvCheck.cpp:212` — nearly feasible for the orig NLP, and
    // there is tolerance budget left to spend chasing the rest.
    //
    // Upstream states the arm's premise in its own comment: it tightens
    // "in case the problem is only very slightly infeasible". *In*feasible
    // is the operative word — the arm spends the tolerance budget chasing
    // a residual violation down to zero. Once the recovered point is
    // feasible to the original NLP's own `tol` there is no residual left
    // to chase, and tightening only drives the sub-solve past the point
    // the outer asked for: eigmaxa/eigmina reach `inf_pr = 7.5e-15`, get
    // tightened anyway, and run on into a tiny-step restoration failure
    // where handing the point straight back solves the problem. dallasm
    // is the same story at `1.5e-10`.
    //
    // Upstream never has to make this distinction, because it cannot
    // arrive here at a feasible point: `IpBacktrackingLineSearch.cpp:578`
    // refuses to enter restoration once the violation is under
    // `1e-2 · tol`, and layer 1's reduction target is floored at
    // `min(tol, constr_viol_tol)` so a feasible trial is released before
    // layer 2 is consulted. Pounce reaches it by both routes, so the
    // premise has to be checked rather than assumed. A point at or under
    // `orig_tol` falls through to the feasible-point arm below, which is
    // the verdict that describes it.
    let slightly_infeasible = orig_trial_inf_pr > orig_tol;
    if nearly_feasible && slightly_infeasible && tol > 1e-1 * orig_tol {
        return RestoOrigVerdict::TightenAndContinue;
    }
    // `IpRestoConvCheck.cpp:222` — a square problem has nothing to
    // optimise, so any point feasible to `constr_viol_tol` is the answer.
    if is_square_problem
        && orig_trial_inf_pr <= orig_constr_viol_tol
        && orig_trial_rel_inf_pr <= rel_viol_threshold
    {
        return RestoOrigVerdict::ConvergedToAcceptable;
    }
    // `IpRestoConvCheck.cpp:227`.
    if nearly_feasible {
        return RestoOrigVerdict::ConvergedToFeasiblePoint;
    }
    // `IpRestoConvCheck.cpp:240`.
    RestoOrigVerdict::LocallyInfeasible
}

/// Scalar core of `IpRestoConvCheck::CheckConvergence` (lines 132-240).
/// The full upstream implementation needs the restoration NLP and the
/// outer filter; this struct holds only the per-call mutable state and
/// exposes a pure `check_convergence` taking the relevant scalars and a
/// closure that decides whether the trial point would be accepted by
/// the outer-phase filter / penalty acceptor.
pub struct RestoConvCheck {
    pub kappa_resto: f64,
    pub maximum_iters: i32,
    pub maximum_resto_iters: i32,
    /// `constr_viol_tol` from the restoration sub-options; used in the
    /// square-problem fast path.
    pub orig_constr_viol_tol: f64,
    /// Fraction of a row's own magnitude the violation must fall under before
    /// layer 2 will call the orig NLP nearly feasible. `1e-2` matches the band
    /// `OptErrorConvCheck::relative_viol_threshold` uses on the outer side.
    pub rel_viol_threshold: f64,
    /// The restoration sub-problem's own tolerance (`IpData().tol()` of
    /// the resto algorithm). Layer 2's tightening arm shrinks it; see
    /// [`resto_orig_verdict`].
    pub tol: f64,
    first_resto_iter: bool,
    successive_resto_iter: i32,
}

impl Default for RestoConvCheck {
    fn default() -> Self {
        // Defaults from `IpRestoConvCheck.cpp:RegisterOptions`.
        Self {
            kappa_resto: 0.9,
            maximum_iters: 3000,
            maximum_resto_iters: 3000,
            orig_constr_viol_tol: 1e-4,
            rel_viol_threshold: 1e-2,
            tol: 1e-8,
            first_resto_iter: true,
            successive_resto_iter: 0,
        }
    }
}

impl RestoConvCheck {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset per-restoration-entry state. Called by the line search
    /// when restoration is (re)activated.
    pub fn reset(&mut self) {
        self.first_resto_iter = true;
        self.successive_resto_iter = 0;
    }

    /// Port of `IpRestoConvCheck.cpp:132-240` (both layers) excluding the
    /// bits that
    /// require the live `IpoptData` / outer filter — those are passed
    /// in as scalars and a closure.
    ///
    /// * `iter_count`           — `IpData().iter_count()` of the *outer* algorithm.
    /// * `is_square_problem`    — `IpCq().IsSquareProblem()`.
    /// * `orig_curr_inf_pr`     — primal infeasibility of the outer algorithm at the
    ///   restoration start (current iterate).
    /// * `orig_trial_inf_pr`    — primal infeasibility of the outer algorithm at the
    ///   *trial* iterate produced by the current restoration iterate.
    /// * `orig_tol`             — the outer `tol` option.
    /// * `acceptable_to_outer`  — closure returning whether the restoration trial
    ///   would be accepted by the outer filter / penalty acceptor.
    /// * `sub_problem_converged` — closure returning whether the restoration
    ///   *sub-problem* has converged in its own right, i.e. upstream's
    ///   `OptimalityErrorConvergenceCheck::CheckConvergence(false)` at
    ///   `IpRestoConvCheck.cpp:202`. This is the layer-2 query; without it a
    ///   restoration that has provably done everything it can is
    ///   indistinguishable from one still making progress (pounce#438).
    pub fn check_convergence(
        &mut self,
        iter_count: i32,
        is_square_problem: bool,
        orig_curr_inf_pr: f64,
        orig_trial_inf_pr: f64,
        orig_trial_rel_inf_pr: f64,
        orig_tol: f64,
        acceptable_to_outer: impl FnOnce() -> bool,
        sub_problem_converged: impl FnOnce() -> bool,
    ) -> RestoConvergenceStatus {
        // Outer iter cap (line 137).
        if iter_count > self.maximum_iters {
            return RestoConvergenceStatus::MaxIterExceeded;
        }

        // Successive-restoration-iter cap (line 144).
        if self.successive_resto_iter > self.maximum_resto_iters {
            return RestoConvergenceStatus::MaxIterExceeded;
        }
        self.successive_resto_iter += 1;

        // Skip the reduction / acceptance test on the very first
        // restoration iteration — no prior `orig_curr_inf_pr` to
        // compare against (line 152).
        if self.first_resto_iter {
            self.first_resto_iter = false;
            return RestoConvergenceStatus::Continue;
        }

        // Square-problem fast path: any feasible trial is the answer
        // (line 162).
        if is_square_problem {
            let target = orig_tol.min(self.orig_constr_viol_tol);
            if orig_trial_inf_pr <= target {
                return RestoConvergenceStatus::Converged;
            }
        }

        // kappa_resto reduction guard (line 175). When kappa_resto == 0
        // upstream disables this guard entirely.
        //
        // NOTE: upstream floors this target at `min(tol, constr_viol_tol)`
        // (`IpRestoConvCheck.cpp:162`); pounce does not, so a restoration
        // entered near-feasible can never satisfy the guard and exits via
        // the sub-problem's own stationarity instead. Adding the floor is
        // upstream-faithful but measurably worse here — it releases the
        // sub-solve earlier, at points far from its own KKT point, and
        // regresses dallasl (Optimal → Error_In_Step_Computation). Left
        // as-is deliberately; it is a separate question from #438.
        let reduction_sufficient =
            self.kappa_resto <= 0.0 || orig_trial_inf_pr <= self.kappa_resto * orig_curr_inf_pr;

        // Reduction was sufficient — defer to the outer-phase filter /
        // penalty acceptor (line 198).
        if reduction_sufficient && acceptable_to_outer() {
            return RestoConvergenceStatus::Converged;
        }

        // ---- Layer 2 (lines 200-240). ---------------------------------
        //
        // Everything above answers "can the trial point leave
        // restoration?" and, when the answer is no, upstream's `status`
        // is still `CONTINUE`. Only then does it ask the second, entirely
        // separate question: has the restoration *sub-problem* itself
        // converged? A sub-problem at its own KKT point has provably done
        // all it can, so "continue" is no longer an available answer —
        // one of four verdicts is (pounce#438).
        if sub_problem_converged() {
            match resto_orig_verdict(
                orig_trial_inf_pr,
                orig_trial_rel_inf_pr,
                self.tol,
                orig_tol,
                self.orig_constr_viol_tol,
                self.rel_viol_threshold,
                is_square_problem,
            ) {
                RestoOrigVerdict::TightenAndContinue => {
                    self.tol *= 1e-2;
                    return RestoConvergenceStatus::Continue;
                }
                RestoOrigVerdict::ConvergedToAcceptable => {
                    return RestoConvergenceStatus::ConvergedToAcceptable;
                }
                RestoOrigVerdict::ConvergedToFeasiblePoint => {
                    return RestoConvergenceStatus::Converged;
                }
                RestoOrigVerdict::LocallyInfeasible => {
                    return RestoConvergenceStatus::LocallyInfeasible;
                }
            }
        }

        RestoConvergenceStatus::Continue
    }
}

/// Restoration-phase convergence check used when the *outer* algorithm
/// runs the filter line search. Wraps `RestoConvCheck` and adds the
/// upstream `TestOrigProgress` predicate from
/// `IpRestoFilterConvCheck.cpp:53-80`, which is what the resto sub-
/// solver consults each iteration to decide whether the recovered
/// iterate is admissible to the outer filter.
pub struct RestoFilterConvCheck {
    pub base: RestoConvCheck,
    /// `obj_max_inc` option, default 5.0. Forwarded to the outer
    /// filter acceptor's rapid-barrier-increase guard.
    pub obj_max_inc: f64,
}
impl RestoFilterConvCheck {
    pub fn new() -> Self {
        Self {
            base: RestoConvCheck::new(),
            obj_max_inc: 5.0,
        }
    }

    /// Mirrors `RestoFilterConvergenceCheck::TestOrigProgress`.
    /// Returns `Converged` only when the trial pair is acceptable to
    /// both the outer filter *and* the outer reference iterate (the
    /// latter with `force_armijo=true` per upstream line 66).
    pub fn test_orig_progress(
        &self,
        outer: &pounce_algorithm::line_search::filter_acceptor::FilterLsAcceptor,
        orig_trial_barr: f64,
        orig_trial_theta: f64,
        reference_barr: f64,
        reference_theta: f64,
    ) -> RestoConvergenceStatus {
        if !outer.is_acceptable_to_current_filter(orig_trial_barr, orig_trial_theta) {
            return RestoConvergenceStatus::Continue;
        }
        if !outer.is_acceptable_to_current_iterate(
            orig_trial_barr,
            orig_trial_theta,
            reference_barr,
            reference_theta,
            self.obj_max_inc,
            true, // called_from_restoration
        ) {
            return RestoConvergenceStatus::Continue;
        }
        RestoConvergenceStatus::Converged
    }
}
impl Default for RestoFilterConvCheck {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RestoPenaltyConvCheck {
    pub base: RestoConvCheck,
}

/// Adapter wiring [`RestoConvCheck`] into the algorithm-side
/// [`pounce_algorithm::conv_check::r#trait::ConvCheck`] trait so the
/// nested IPM in [`crate::resto_inner_solver`] can swap out the
/// regular-phase [`pounce_algorithm::conv_check::opt_error::OptErrorConvCheck`].
///
/// This is the live path — `run_inner_resto` installs it as the nested
/// IPM's `conv_check` — and it carries both layers of
/// `IpRestoConvCheck::CheckConvergence`:
///
/// * the resto-side iteration-cap state machine (`maximum_iters` /
///   `maximum_resto_iters` from `IpRestoConvCheck.cpp:RegisterOptions`);
/// * **layer 1**, *can the trial point leave restoration?* — the
///   kappa-reduction guard (`orig_trial_inf_pr <= kappa_resto ·
///   orig_curr_inf_pr`, wired by [`Self::with_orig_progress_guard`])
///   followed by the outer filter / iterate acceptance test (wired by
///   [`Self::with_orig_progress_callback`]);
/// * the sub-problem's own stationarity check, delegated to a wrapped
///   `OptErrorConvCheck`;
/// * **layer 2**, *has the sub-problem provably done everything it can?*
///   — the four-way verdict of [`resto_orig_verdict`], wired by
///   [`Self::with_orig_convergence_verdict`] (pounce#438).
///
/// Only layer 2 can bound a restoration whose kappa target is out of
/// reach: layer 1 answers `Continue` at every such iterate, forever, so
/// without a verdict at sub-problem convergence the only remaining exits
/// are the iteration caps — which report "the solve ran out of
/// iterations" for what is really "restoration cannot succeed here".
///
/// Snapshots the outer scalars at construction (option (b) from the task
/// design note) so the trait surface stays narrow.
pub struct RestoConvCheckAdapter {
    inner: pounce_algorithm::conv_check::opt_error::OptErrorConvCheck,
    /// `IpRestoConvCheck.cpp:137` outer-iter cap.
    maximum_iters: i32,
    /// `IpRestoConvCheck.cpp:144` successive-restoration cap.
    maximum_resto_iters: i32,
    /// Bumped on every `check_convergence` call after the first; once
    /// it reaches `maximum_resto_iters` the adapter forces
    /// `MaxIterExceeded`.
    successive_resto_iter: i32,
    /// Orig NLP for the kappa-reduction early-exit
    /// (`IpRestoConvCheck.cpp:175`). When wired alongside
    /// [`Self::orig_curr_inf_pr`], the adapter evaluates the orig-NLP
    /// `max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` at every inner
    /// iterate's `(x_orig, s)` slice and reports `Converged` once the
    /// reduction satisfies `orig_trial_inf_pr ≤ kappa_resto · orig_curr_inf_pr`.
    /// Without it, the adapter falls back to inner-stationarity only.
    orig_nlp: Option<std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>>,
    /// Snapshot of the outer-iterate's orig-NLP `inf_pr` at restoration
    /// entry; used as the reference for the kappa-reduction guard.
    orig_curr_inf_pr: f64,
    /// `kappa_resto` from `IpRestoConvCheck.cpp:RegisterOptions`
    /// (default 0.9). When `0.0`, the kappa guard is disabled (matches
    /// upstream's "kappa_resto == 0 disables this guard entirely"
    /// branch on line 175).
    kappa_resto: f64,
    /// Orig-progress callback supplied by the outer line search at
    /// restoration entry (mirrors upstream
    /// `IpRestoFilterConvCheck::SetOrigLSAcceptor`). When wired, the
    /// adapter reports `Converged` only after the kappa-reduction guard
    /// passes *and* the callback returns `true` for
    /// `(orig_trial_barr=f(x_orig), orig_trial_theta=inf_pr)`. When
    /// `None`, the kappa guard alone gates `Converged` (matches the
    /// `RestoConvCheck`-base behavior — the filter-aware variant only
    /// fires when the outer phase's acceptor is `FilterLsAcceptor`).
    orig_progress_callback: Option<pounce_algorithm::restoration::OrigProgressCallback>,
    /// The *original* NLP's `tol`. Layer 2's tightening arm is bounded
    /// against it (`IpRestoConvCheck.cpp:213`'s
    /// `IpData().tol() > 1e-1 * orig_ip_data->tol()`); the sub-problem's
    /// own, tightenable tolerance lives on [`Self::inner`].
    orig_tol: f64,
    /// The original NLP's `constr_viol_tol`
    /// (`IpRestoConvCheck::InitializeImpl` reads it with the *original*
    /// prefix). Consulted only on layer 2's square-problem arm.
    orig_constr_viol_tol: f64,
    /// See [`RestoConvCheck::rel_viol_threshold`].
    rel_viol_threshold: f64,
    /// Whether the original NLP is square (`IpCq().IsSquareProblem()`).
    is_square_problem: bool,
    /// Times layer 2's tightening arm has fired this restoration entry.
    /// Bounded by construction (see [`resto_orig_verdict`]); tracked for
    /// the debug trace only.
    resto_tol_tightenings: i32,
}

impl RestoConvCheckAdapter {
    /// Build an adapter from a [`RestoConvCheck`] template. `tol` /
    /// `acceptable_tol` come from the resto sub-options (typically
    /// the "resto." prefixed knobs); `max_iter` is the per-call cap on
    /// inner IPM iterations.
    pub fn new(
        tol: f64,
        acceptable_tol: f64,
        acceptable_iter: i32,
        max_iter: i32,
        maximum_resto_iters: i32,
    ) -> Self {
        let mut inner = pounce_algorithm::conv_check::opt_error::OptErrorConvCheck::new();
        inner.tol = tol;
        inner.acceptable_tol = acceptable_tol;
        inner.acceptable_iter = acceptable_iter;
        inner.max_iter = max_iter;
        Self {
            inner,
            maximum_iters: max_iter,
            maximum_resto_iters,
            successive_resto_iter: 0,
            orig_nlp: None,
            orig_curr_inf_pr: f64::INFINITY,
            kappa_resto: 0.9,
            orig_progress_callback: None,
            // Defaults keep layer 2 inert-but-consistent for callers that
            // don't wire it: `orig_tol == tol` means the tightening arm
            // fires exactly once, and a non-square problem takes the
            // upstream arms unchanged.
            orig_tol: tol,
            orig_constr_viol_tol: 1e-4,
            rel_viol_threshold: 1e-2,
            is_square_problem: false,
            resto_tol_tightenings: 0,
        }
    }

    /// Inner-IPM stationarity tolerance (`OptErrorConvCheck::tol`) this
    /// adapter was built with. Read-only accessor used by the resto
    /// factory tests to confirm the user's outer `tol` is threaded
    /// through instead of a hardcoded default.
    pub fn inner_tol(&self) -> f64 {
        self.inner.tol
    }

    /// Inner-IPM acceptable tolerance (`OptErrorConvCheck::acceptable_tol`).
    pub fn inner_acceptable_tol(&self) -> f64 {
        self.inner.acceptable_tol
    }

    /// Inner-IPM acceptable-iteration count (`OptErrorConvCheck::acceptable_iter`).
    pub fn inner_acceptable_iter(&self) -> i32 {
        self.inner.acceptable_iter
    }

    /// Wire the orig-progress callback the outer line search built at
    /// restoration entry. Adds a second gate to the `Converged`
    /// decision: after the kappa-reduction passes, the recovered
    /// iterate must also be acceptable to the outer filter and
    /// reference iterate (mirrors upstream
    /// `IpRestoFilterConvCheck::TestOrigProgress`). When the callback
    /// is wired but the orig NLP is not (no
    /// [`Self::with_orig_progress_guard`]), the adapter is unable to
    /// evaluate `orig_trial_barr`/`orig_trial_theta`, so the gate is
    /// skipped — same behavior as upstream's
    /// `RestoConvCheck`-without-filter case.
    pub fn with_orig_progress_callback(
        mut self,
        cb: pounce_algorithm::restoration::OrigProgressCallback,
    ) -> Self {
        self.orig_progress_callback = Some(cb);
        self
    }

    /// Wire the orig NLP and the outer-curr orig-NLP `inf_pr` so the
    /// adapter can run upstream's kappa-reduction early-exit guard
    /// (`IpRestoConvCheck.cpp:175`) on every inner iteration.
    /// `kappa_resto` defaults to upstream's 0.9; pass `0.0` to disable
    /// the guard while keeping the orig-NLP plumbing live (e.g. for
    /// instrumentation-only runs).
    pub fn with_orig_progress_guard(
        mut self,
        orig: std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
        orig_curr_inf_pr: f64,
        kappa_resto: f64,
    ) -> Self {
        self.orig_nlp = Some(orig);
        self.orig_curr_inf_pr = orig_curr_inf_pr;
        self.kappa_resto = kappa_resto;
        self
    }

    /// Wire the original-NLP scalars layer 2 needs
    /// (`IpRestoConvCheck.cpp:200-240`, pounce#438): the original `tol`
    /// that bounds the tolerance-tightening arm, the original
    /// `constr_viol_tol` the square-problem arm compares against, and
    /// whether the original NLP is square.
    ///
    /// Layer 2 additionally needs the orig NLP itself to evaluate the
    /// recovered point's constraint violation, so it only becomes live
    /// once [`Self::with_orig_progress_guard`] has been called too;
    /// without it the adapter keeps its pre-#438 behaviour of reporting
    /// whatever the sub-problem's own convergence check said.
    pub fn with_orig_convergence_verdict(
        mut self,
        orig_tol: f64,
        orig_constr_viol_tol: f64,
        is_square_problem: bool,
    ) -> Self {
        self.orig_tol = orig_tol;
        self.orig_constr_viol_tol = orig_constr_viol_tol;
        self.is_square_problem = is_square_problem;
        self
    }

    /// Layer 2 of `IpRestoConvCheck::CheckConvergence`
    /// (`IpRestoConvCheck.cpp:200-240`), evaluated once the restoration
    /// sub-problem's own convergence check has fired.
    ///
    /// Returns `Some(status)` when the verdict overrides what the
    /// sub-problem reported, `None` when it does not (the caller then
    /// keeps the sub-problem's own status).
    fn orig_convergence_verdict(
        &mut self,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    ) -> Option<pounce_algorithm::conv_check::r#trait::ConvergenceStatus> {
        use pounce_algorithm::conv_check::r#trait::ConvergenceStatus;

        let orig_rc = self.orig_nlp.clone()?;
        let m = eval_orig_trial_measures(data, &orig_rc)?;
        let tol = self.inner.tol;
        let verdict = resto_orig_verdict(
            m.inf_pr_unscaled,
            m.inf_pr_relative,
            tol,
            self.orig_tol,
            self.orig_constr_viol_tol,
            self.rel_viol_threshold,
            self.is_square_problem,
        );

        if std::env::var_os("POUNCE_DBG_RESTO_LAYER2").is_some() {
            tracing::debug!(target: "pounce::restoration",
                "[PN_RESTO_LAYER2] verdict={:?} orig_inf_pr={:.6e} orig_inf_pr_scaled={:.6e} orig_inf_pr_rel={:.6e} tol={:.6e} orig_tol={:.6e} orig_constr_viol_tol={:.6e} square={} tightenings={}",
                verdict,
                m.inf_pr_unscaled,
                m.inf_pr_scaled,
                m.inf_pr_relative,
                tol,
                self.orig_tol,
                self.orig_constr_viol_tol,
                self.is_square_problem,
                self.resto_tol_tightenings,
            );
        }

        match verdict {
            RestoOrigVerdict::TightenAndContinue => {
                self.inner.tol = 1e-2 * tol;
                self.resto_tol_tightenings += 1;
                Some(ConvergenceStatus::Continue)
            }
            RestoOrigVerdict::ConvergedToAcceptable => {
                Some(ConvergenceStatus::ConvergedToAcceptable)
            }
            // Upstream throws `RESTORATION_CONVERGED_TO_FEASIBLE_POINT`
            // here, which surfaces at the outer level as a restoration
            // *failure*. Pounce instead hands the recovered — and by this
            // arm's own test, feasible — point back to the outer phase,
            // which is what it already did before #438 and is strictly
            // more useful than discarding it. Deliberate deviation, scoped
            // so that #438 changes only the arm that was missing a verdict
            // entirely.
            RestoOrigVerdict::ConvergedToFeasiblePoint => None,
            RestoOrigVerdict::LocallyInfeasible => {
                // Square problems are exempt. `MinC_1NrmRestorationPhase`
                // returns the recovered point to the outer unconditionally
                // on a square problem (`IpRestoMinC_1Nrm.cpp:357-371`), and
                // pounce found short-circuiting that path regresses
                // PFIT3/PFIT4 — the outer needs the extra shots. Same
                // carve-out the post-hoc `strict_locally_infeasible` gate
                // in `resto_inner_solver` makes, for the same reason.
                if self.is_square_problem {
                    return None;
                }
                // Never claim infeasibility at a point the solver's own
                // convergence test would call feasible. `inf_pr_unscaled`
                // is measured in user units (correctly — the thresholds
                // above are absolute user-facing magnitudes) while `tol`
                // is applied in the scaled space, so on a model whose rows
                // are scaled down the two disagree. Mirrors the guard
                // `resto_inner_solver` applies to its post-hoc gates.
                if m.inf_pr_scaled <= self.orig_tol {
                    return None;
                }
                Some(ConvergenceStatus::LocallyInfeasible)
            }
        }
    }

    /// Construct from a base [`RestoConvCheck`] using the resto-side
    /// option defaults plus an explicit inner-stationarity tolerance.
    pub fn from_base(base: &RestoConvCheck, tol: f64, acceptable_tol: f64) -> Self {
        Self::new(
            tol,
            acceptable_tol,
            15, // OptErrorConvCheck::default acceptable_iter
            base.maximum_iters,
            base.maximum_resto_iters,
        )
    }
}

impl pounce_algorithm::conv_check::r#trait::ConvCheck for RestoConvCheckAdapter {
    fn check_convergence(
        &mut self,
        nlp_err: pounce_common::types::Number,
        iter_count: pounce_common::types::Index,
    ) -> pounce_algorithm::conv_check::r#trait::ConvergenceStatus {
        use pounce_algorithm::conv_check::r#trait::ConvergenceStatus;
        if iter_count >= self.maximum_iters
            || self.successive_resto_iter >= self.maximum_resto_iters
        {
            return ConvergenceStatus::MaxIterExceeded;
        }
        self.successive_resto_iter += 1;
        self.inner.check_convergence(nlp_err, iter_count)
    }

    fn check_convergence_with_state(
        &mut self,
        nlp_err: pounce_common::types::Number,
        iter_count: pounce_common::types::Index,
        data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
        _cq: &pounce_algorithm::ipopt_cq::IpoptCqHandle,
    ) -> pounce_algorithm::conv_check::r#trait::ConvergenceStatus {
        use pounce_algorithm::conv_check::r#trait::ConvergenceStatus;
        // 1. Iter-cap checks (pre-bump). Match the scalar branch.
        if iter_count >= self.maximum_iters
            || self.successive_resto_iter >= self.maximum_resto_iters
        {
            return ConvergenceStatus::MaxIterExceeded;
        }
        self.successive_resto_iter += 1;

        // 1b. Time-budget gate at each inner restoration iteration
        //     (pounce#246). The stationarity fallback below delegates to
        //     the wrapped *scalar* `OptErrorConvCheck::check_convergence`,
        //     which cannot see `data` and so never consulted the shared
        //     [`Deadline`] the regular-phase
        //     `OptErrorConvCheck::check_convergence_with_state` honors.
        //     Without this the restoration inner IPM's convergence check
        //     ignored the wall/CPU budget — a bad-start restoration grind
        //     only tripped it one iteration later at `IpoptAlgorithm::
        //     iterate`'s post-`compute_search_direction` gate, and the
        //     per-iteration orig-NLP evaluation in the kappa-reduction
        //     step below ran needlessly while over budget. Checked before
        //     that evaluation, and ordered CPU-before-wall to match the
        //     regular-phase check so an inner solve reports the identical
        //     time-limit status.
        if let Some(deadline) = data.borrow().deadline.as_ref() {
            match deadline.exceeded() {
                Some(pounce_common::timing::DeadlineKind::Cpu) => {
                    return ConvergenceStatus::CpuTimeExceeded;
                }
                Some(pounce_common::timing::DeadlineKind::Wall) => {
                    return ConvergenceStatus::WallTimeExceeded;
                }
                None => {}
            }
        }

        // 2. Kappa-reduction early-exit on orig-NLP `inf_pr`. Mirrors
        //    upstream `IpRestoConvCheck.cpp:175` — when the inner
        //    iterate's orig `(theta_trial)` is below
        //    `kappa_resto · orig_curr_inf_pr`, restoration has done
        //    enough. (Upstream floors that target — see the note in
        //    `RestoConvCheck::check_convergence`; pounce deliberately
        //    does not.) Upstream then runs `TestOrigProgress` (filter +
        //    iterate acceptance) before declaring `Converged`; we mirror
        //    that gate via [`Self::orig_progress_callback`]. The first
        //    inner iter is skipped (no prior reduction reference yet)
        //    by checking `iter_count > 0`; that matches upstream's
        //    `first_resto_iter` freebie at line 152.
        if iter_count > 0 && self.kappa_resto > 0.0 {
            if let Some(orig_rc) = self.orig_nlp.clone() {
                if let Some(OrigTrialMeasures {
                    inf_pr_scaled: orig_trial_inf_pr,
                    f: orig_trial_f,
                    ..
                }) = eval_orig_trial_measures(data, &orig_rc)
                {
                    if std::env::var_os("POUNCE_DBG_RESTO_KAPPA").is_some() {
                        tracing::debug!(target: "pounce::restoration",
                            "[PN_RESTO_KAPPA] iter={} orig_trial_inf_pr={:.6e} orig_curr_inf_pr={:.6e} kappa_resto={:.3e} threshold={:.6e} guard_passes={}",
                            iter_count,
                            orig_trial_inf_pr,
                            self.orig_curr_inf_pr,
                            self.kappa_resto,
                            self.kappa_resto * self.orig_curr_inf_pr,
                            orig_trial_inf_pr <= self.kappa_resto * self.orig_curr_inf_pr
                        );
                    }
                    if orig_trial_inf_pr <= self.kappa_resto * self.orig_curr_inf_pr {
                        // Kappa reduction satisfied. Now consult the
                        // outer-filter / iterate-acceptance callback if
                        // wired (mirrors `TestOrigProgress`). When the
                        // callback is absent, kappa alone gates the
                        // exit (matches `RestoConvCheck`-base).
                        let outer_accept = match &self.orig_progress_callback {
                            Some(cb) => cb(orig_trial_f, orig_trial_inf_pr),
                            None => true,
                        };
                        if outer_accept {
                            return ConvergenceStatus::Converged;
                        }
                    }
                }
            }
        }

        // 3. Inner-stationarity check (resto NLP's own KKT residual) —
        //    upstream's `OptimalityErrorConvergenceCheck::CheckConvergence`
        //    call at `IpRestoConvCheck.cpp:202`.
        let inner_status = self.inner.check_convergence(nlp_err, iter_count);

        // 4. Layer 2 (`IpRestoConvCheck.cpp:200-240`, pounce#438). Steps
        //    1-3 above only ever answer "can the trial point leave
        //    restoration?"; when the sub-problem has converged in its own
        //    right, that question is no longer the operative one. A
        //    restoration whose sub-NLP is at its KKT point has provably
        //    done everything it can, so it must render a verdict instead
        //    of reporting the sub-problem's status and letting the outer
        //    re-enter — or, when the kappa target is out of reach,
        //    grinding to an iteration cap that reports the wrong status.
        if !matches!(
            inner_status,
            ConvergenceStatus::Converged | ConvergenceStatus::ConvergedToAcceptable
        ) {
            return inner_status;
        }
        // `None` means the verdict declines to override — the caller then
        // keeps whatever the sub-problem's own check said.
        self.orig_convergence_verdict(data).unwrap_or(inner_status)
    }
}

/// Original-NLP measurements at the restoration sub-solve's current
/// iterate, in both unit systems.
struct OrigTrialMeasures {
    /// `max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` in the solver's internal
    /// (row-scaled) space. This is the units the kappa-reduction guard
    /// wants, because it compares against `orig_curr_inf_pr` — itself a
    /// scaled quantity from `IpoptCalculatedQuantities` — and because a
    /// ratio test cancels the scaling anyway.
    inf_pr_scaled: f64,
    /// The same quantity in unscaled (user) units. This is the units
    /// layer 2 wants: its thresholds (`1e2 · tol`, `constr_viol_tol`) are
    /// absolute user-facing magnitudes, so comparing them against a
    /// row-scaled residual mixes two unit systems — the same mismatch
    /// documented at `resto_inner_solver::eval_orig_inf_pr_at_inner_curr`.
    inf_pr_unscaled: f64,
    /// The orig NLP's constraint violation as a fraction of the violated
    /// row's own magnitude, `max_i viol_i / max(|d_l_i|, |d_u_i|)`.
    ///
    /// Neither absolute measure above can be scale-invariant: a user may write
    /// any row as `s*g(x) >= s*b` for `s > 0` without changing the feasible
    /// set, and the residual then carries `s`. gh #390 (residual of #387)
    /// settled this for the outer check; layer 2 needs the same invariant the
    /// moment it makes a feasibility judgement of its own.
    ///
    /// Measured against the row's **bounds**, not against the restoration
    /// slack: it is the orig row that has a declared magnitude. Rows with no
    /// finite bound, or a declared magnitude of zero, contribute nothing — the
    /// absolute measures are already invariant there.
    inf_pr_relative: f64,
    /// Unscaled `f(x_orig)`.
    f: f64,
}

/// Evaluate the orig-NLP measurements at the inner iterate's
/// `(x_orig, s)` slice:
///
/// * `inf_pr = max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` — used by the
///   kappa-reduction guard, by layer 2's verdict, and as the
///   orig-trial-theta passed to the outer-progress callback. Returned in
///   both the scaled and unscaled unit systems; see [`OrigTrialMeasures`]
///   for which consumer wants which.
/// * `f = unscaled f(x_orig)` — used as the orig-trial-barr proxy for
///   the outer-progress callback. v0.1 simplification: the upstream
///   `TestOrigProgress` consults `trial_barrier_obj()`, which folds in
///   `-mu * sum log(slacks)` over all bound slacks. The simplified
///   `f`-only proxy is sound because the iterate-acceptance branch
///   used in restoration runs with `force_armijo = true`
///   (`called_from_restoration = true`), which disables the
///   rapid-barrier-increase guard — leaving only the
///   `theta`-progress / `barr`-progress disjunction, where the theta
///   branch is the dominant exit path during feasibility recovery.
///
/// Returns `None` on any downcast / dim failure (caller falls back to
/// the scalar inner-stationarity path).
fn eval_orig_trial_measures(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    orig_rc: &std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
) -> Option<OrigTrialMeasures> {
    use pounce_algorithm::ipopt_cq::unscaled_block_amax;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::{CompoundVector, Vector};

    let curr = data.borrow().curr.clone()?;
    let xc = curr.x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(crate::resto_nlp::BLOCK_X);
    let s_inner = &*curr.s;

    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();
    let (dc, dd) = (orig.c_scale_vec(), orig.d_scale_vec());

    // Each pair is `(unscaled, scaled)`.
    let c_amax = if m_eq > 0 {
        let mut c_buf = DenseVectorSpace::new(m_eq).make_new_dense();
        orig.eval_c(x_orig, &mut c_buf);
        (unscaled_block_amax(&c_buf, dc.as_deref()), c_buf.amax())
    } else {
        (0.0, 0.0)
    };

    let d_minus_s_amax = if m_ineq > 0 {
        let mut d_buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut d_buf);
        d_buf.axpy(-1.0, s_inner);
        (unscaled_block_amax(&d_buf, dd.as_deref()), d_buf.amax())
    } else {
        (0.0, 0.0)
    };

    // Scale-free companion, mirroring `IpoptCq::relative_d_infeasibility_max`.
    // The bound vectors are **compressed** — one entry per row that has that
    // side — so they must be expanded through `pd_l`/`pd_u` before they can be
    // indexed against a row. Projecting an all-ones vector the same way gives
    // the presence masks, without which a projected `0` is ambiguous between
    // "no bound this side" and "a declared bound of zero".
    let inf_pr_relative = if m_ineq > 0 {
        let mut d_row = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut d_row);

        // Prefer the declared bounds: on the live vectors a relaxed zero bound
        // reads as ~1e-8 and would fabricate a magnitude for a row with none.
        let (mut cl, mut cu) = (orig.d_l().make_new(), orig.d_u().make_new());
        cl.copy(orig.d_l());
        cu.copy(orig.d_u());
        if let Some((dl, du)) = orig.declared_d_bounds() {
            if let (Some(cld), Some(cud)) = (
                cl.as_any_mut()
                    .downcast_mut::<pounce_linalg::dense_vector::DenseVector>(),
                cu.as_any_mut()
                    .downcast_mut::<pounce_linalg::dense_vector::DenseVector>(),
            ) {
                cld.set_values(&dl);
                cud.set_values(&du);
            }
        }

        let mut lo = d_row.make_new();
        lo.set(0.0);
        orig.pd_l().mult_vector(1.0, &*cl, 0.0, &mut *lo);
        let mut hi = d_row.make_new();
        hi.set(0.0);
        orig.pd_u().mult_vector(1.0, &*cu, 0.0, &mut *hi);

        let mut ones_l = orig.d_l().make_new();
        ones_l.set(1.0);
        let mut mask_l = d_row.make_new();
        mask_l.set(0.0);
        orig.pd_l().mult_vector(1.0, &*ones_l, 0.0, &mut *mask_l);
        let mut ones_u = orig.d_u().make_new();
        ones_u.set(1.0);
        let mut mask_u = d_row.make_new();
        mask_u.set(0.0);
        orig.pd_u().mult_vector(1.0, &*ones_u, 0.0, &mut *mask_u);

        use pounce_linalg::dense_vector::DenseVector;
        match (
            d_row.as_any().downcast_ref::<DenseVector>(),
            (&*lo).as_any().downcast_ref::<DenseVector>(),
            (&*hi).as_any().downcast_ref::<DenseVector>(),
            (&*mask_l).as_any().downcast_ref::<DenseVector>(),
            (&*mask_u).as_any().downcast_ref::<DenseVector>(),
        ) {
            (Some(d), Some(lo), Some(hi), Some(ml), Some(mu)) => {
                let (dv, lov, hiv, mlv, muv) = (
                    d.expanded_values(),
                    lo.expanded_values(),
                    hi.expanded_values(),
                    ml.expanded_values(),
                    mu.expanded_values(),
                );
                let mut worst = 0.0_f64;
                for i in 0..dv.len() {
                    let (has_l, has_u) = (mlv[i] > 0.5, muv[i] > 0.5);
                    let (mut viol, mut mag) = (0.0_f64, 0.0_f64);
                    if has_l {
                        viol = viol.max(lov[i] - dv[i]);
                        mag = mag.max(lov[i].abs());
                    }
                    if has_u {
                        viol = viol.max(dv[i] - hiv[i]);
                        mag = mag.max(hiv[i].abs());
                    }
                    if mag > 0.0 && mag.is_finite() && viol.is_finite() && viol > 0.0 {
                        worst = worst.max(viol / mag);
                    }
                }
                worst
            }
            _ => 0.0,
        }
    } else {
        0.0
    };

    let f = orig.eval_f(x_orig);

    Some(OrigTrialMeasures {
        inf_pr_scaled: c_amax.1.max(d_minus_s_amax.1),
        inf_pr_unscaled: c_amax.0.max(d_minus_s_amax.0),
        inf_pr_relative,
        f,
    })
}

impl RestoPenaltyConvCheck {
    pub fn new() -> Self {
        Self {
            base: RestoConvCheck::new(),
        }
    }
}
impl Default for RestoPenaltyConvCheck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_iteration_always_continues() {
        let mut cc = RestoConvCheck::new();
        let s = cc.check_convergence(0, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn outer_iter_cap_triggers_max() {
        let mut cc = RestoConvCheck::new();
        cc.maximum_iters = 5;
        let s = cc.check_convergence(6, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        assert_eq!(s, RestoConvergenceStatus::MaxIterExceeded);
    }

    #[test]
    fn successive_resto_cap_triggers_max() {
        let mut cc = RestoConvCheck::new();
        cc.maximum_resto_iters = 2;
        // Burn through 3 calls — fourth should hit the cap.
        cc.check_convergence(0, false, 1.0, 0.9, 0.0, 1e-8, || false, || false);
        cc.check_convergence(1, false, 1.0, 0.8, 0.0, 1e-8, || false, || false);
        cc.check_convergence(2, false, 1.0, 0.7, 0.0, 1e-8, || false, || false);
        let s = cc.check_convergence(3, false, 1.0, 0.6, 0.0, 1e-8, || false, || false);
        assert_eq!(s, RestoConvergenceStatus::MaxIterExceeded);
    }

    #[test]
    fn square_problem_fast_path_converges() {
        let mut cc = RestoConvCheck::new();
        cc.orig_constr_viol_tol = 1e-4;
        // Burn the first-iter freebie.
        cc.check_convergence(0, true, 1.0, 0.5, 0.0, 1e-8, || false, || false);
        // Now feed a feasible trial.
        let s = cc.check_convergence(1, true, 0.5, 1e-10, 0.0, 1e-8, || false, || false);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn insufficient_reduction_keeps_going() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.95, 0.0, 1e-8, || true, || false);
        // trial_inf_pr (0.95) > 0.9 * curr_inf_pr (1.0) — not enough.
        let s = cc.check_convergence(1, false, 1.0, 0.95, 0.0, 1e-8, || true, || false);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn sufficient_reduction_plus_filter_accept_converges() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        let s = cc.check_convergence(1, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn sufficient_reduction_but_filter_rejects_continues() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.5, 0.0, 1e-8, || false, || false);
        let s = cc.check_convergence(1, false, 1.0, 0.5, 0.0, 1e-8, || false, || false);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn kappa_zero_disables_reduction_guard() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.0;
        cc.check_convergence(0, false, 1.0, 1.5, 0.0, 1e-8, || true, || false);
        // Even with trial > curr, the guard is bypassed and we go to
        // the outer-filter check, which accepts.
        let s = cc.check_convergence(1, false, 1.0, 1.5, 0.0, 1e-8, || true, || false);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn reset_clears_state() {
        let mut cc = RestoConvCheck::new();
        cc.check_convergence(0, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        cc.check_convergence(1, false, 1.0, 0.5, 0.0, 1e-8, || true, || false);
        cc.reset();
        assert!(cc.first_resto_iter);
        assert_eq!(cc.successive_resto_iter, 0);
    }

    // ---- Layer 2 (`IpRestoConvCheck.cpp:200-240`, pounce#438) ---------

    /// The tightening arm fires while there is tolerance budget left, and
    /// stops on its own — the guard `tol > 1e-1 · orig_tol` is the only
    /// thing that makes it terminate, so it is the property worth pinning.
    #[test]
    fn layer2_tightening_arm_is_bounded_by_construction() {
        let orig_tol = 1e-8;
        let mut tol = orig_tol;
        let mut fired = 0;
        // Slightly *infeasible* — above `orig_tol`, so the arm's premise
        // holds — and inside `1e2 · tol` so it is near-feasible too.
        let inf_pr = 1e-7;
        while resto_orig_verdict(inf_pr, 0.0, tol, orig_tol, 1e-4, 1e-2, false)
            == RestoOrigVerdict::TightenAndContinue
        {
            tol *= 1e-2;
            fired += 1;
            assert!(fired < 10, "tightening arm failed to terminate");
        }
        assert_eq!(fired, 1, "one round from tol == orig_tol");
        assert_eq!(tol, 1e-2 * orig_tol);
    }

    #[test]
    fn layer2_square_problem_feasible_within_constr_viol_tol_is_acceptable() {
        // Square, violation under `constr_viol_tol` but above `1e2 · tol`
        // so the tightening arm cannot claim it first.
        let v = resto_orig_verdict(1e-5, 0.0, 1e-10, 1e-8, 1e-4, 1e-2, true);
        assert_eq!(v, RestoOrigVerdict::ConvergedToAcceptable);
        // The same numbers on a non-square problem are a local infeasibility.
        let v = resto_orig_verdict(1e-5, 0.0, 1e-10, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
    }

    #[test]
    fn layer2_feasible_point_when_tolerance_budget_is_spent() {
        // `tol` already tightened past the guard, so the tightening arm is
        // closed; the violation is still inside `1e2 · tol`.
        let v = resto_orig_verdict(1e-11, 0.0, 1e-10, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::ConvergedToFeasiblePoint);
    }

    /// The #438 case: the sub-problem has converged at a point whose
    /// original-NLP violation is orders of magnitude above anything the
    /// tolerances would accept. Before layer 2 this returned no verdict at
    /// all and restoration ran to an iteration cap.
    #[test]
    fn layer2_locally_infeasible_when_violation_is_bounded_away_from_zero() {
        // qcqp1000-1nc's numbers: orig violation pinned at ~5e-3 with the
        // sub-problem at its own KKT point, default tolerances.
        let v = resto_orig_verdict(5.05e-3, 0.0, 1e-8, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
        // Square problems reach the same verdict once the violation clears
        // `constr_viol_tol`; the caller applies pounce's square carve-out.
        let v = resto_orig_verdict(5.05e-3, 0.0, 1e-8, 1e-8, 1e-4, 1e-2, true);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
    }

    /// The property the whole relative measure exists for: multiplying a row
    /// by a positive constant cannot change the verdict, because it cannot
    /// change the feasible set.
    ///
    /// `inf_372` (`0 <= x <= 0.6` with `s*x >= s*0.7`) is infeasible by 14% of
    /// the row at every `s`, but its *absolute* violation is `1.0e-1` at
    /// `s = 1` and `1.0e-9` at `s = 1e-8`. On the absolute test alone the
    /// second reads as nearly feasible and the arms mis-fire — the regression
    /// `pyomo-pounce/tests/test_scale_invariance.py` caught.
    #[test]
    fn layer2_verdict_is_invariant_to_row_scaling() {
        for (abs_viol, rel_viol) in [
            (1.0e-1, 1.428571e-1),      // s = 1
            (1.005193e-9, 1.432384e-1), // s = 1e-8, same model
        ] {
            let v = resto_orig_verdict(abs_viol, rel_viol, 1e-8, 1e-8, 1e-4, 1e-2, false);
            assert_eq!(
                v,
                RestoOrigVerdict::LocallyInfeasible,
                "abs={abs_viol:e} rel={rel_viol:e} must reach the same verdict at every row scaling"
            );
        }
    }

    /// The relative measure only ever *withholds* near-feasibility; it can
    /// never manufacture it. A point tiny on both measures still tightens.
    #[test]
    fn layer2_relative_gate_blocks_but_never_creates_near_feasibility() {
        // Small absolutely, but 14% of the row: not nearly feasible.
        // (Above `orig_tol`, so only the relative measure can block it —
        // which is what this test isolates.)
        let v = resto_orig_verdict(1e-7, 1.4e-1, 1e-8, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
        // Small on both: unchanged from the absolute-only behaviour.
        let v = resto_orig_verdict(1e-7, 1e-6, 1e-8, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::TightenAndContinue);
        // Large absolutely but small relatively is still not nearly feasible:
        // both tests must pass, so neither can override the other.
        let v = resto_orig_verdict(5.05e-3, 1e-6, 1e-8, 1e-8, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
    }

    /// A row with no declared magnitude — homogeneous, or an NLP that does not
    /// track one — contributes `0` to the relative measure and must fall back
    /// to the absolute test, which is already scale-invariant there
    /// (`s*g(x) == 0` is the same row at every `s`).
    ///
    /// This also pins the meaning of the `0.0` the other unit tests pass: it
    /// is the abstaining case, not a way of switching the gate off.
    #[test]
    fn layer2_abstains_when_the_row_has_no_declared_magnitude() {
        // Same three inputs as the arms' own tests, with an abstaining
        // relative measure: each must land exactly where it did before.
        assert_eq!(
            resto_orig_verdict(1e-11, 0.0, 1e-10, 1e-8, 1e-4, 1e-2, false),
            RestoOrigVerdict::ConvergedToFeasiblePoint
        );
        assert_eq!(
            resto_orig_verdict(1e-5, 0.0, 1e-10, 1e-8, 1e-4, 1e-2, true),
            RestoOrigVerdict::ConvergedToAcceptable
        );
        assert_eq!(
            resto_orig_verdict(5.05e-3, 0.0, 1e-8, 1e-8, 1e-4, 1e-2, false),
            RestoOrigVerdict::LocallyInfeasible
        );
    }

    /// The square-problem arm is gated too. It is a feasibility judgement like
    /// the others (`orig_trial_inf_pr <= constr_viol_tol`), so leaving it on
    /// the absolute test alone would reopen the same hole one arm along.
    #[test]
    fn layer2_square_arm_is_gated_on_the_relative_measure_too() {
        // Absolutely inside `constr_viol_tol`, but 14% of the row.
        let v = resto_orig_verdict(1e-5, 1.4e-1, 1e-10, 1e-8, 1e-4, 1e-2, true);
        assert_ne!(v, RestoOrigVerdict::ConvergedToAcceptable);
        assert_eq!(v, RestoOrigVerdict::LocallyInfeasible);
    }

    /// Layer 2 runs when layer 1 says `Continue`, which includes the case
    /// the issue is about: a `kappa_resto` target the restoration can never
    /// reach. An insufficient reduction must no longer swallow the verdict.
    #[test]
    fn insufficient_reduction_still_renders_a_layer2_verdict() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.tol = 1e-8;
        // Burn the first-iter freebie.
        cc.check_convergence(0, false, 5.96e-8, 5.05e-3, 0.0, 1e-8, || false, || false);
        // Reduction is five orders out of reach, so layer 1 is `Continue`
        // forever — but the sub-problem has converged.
        let s = cc.check_convergence(1, false, 5.96e-8, 5.05e-3, 0.0, 1e-8, || false, || true);
        assert_eq!(s, RestoConvergenceStatus::LocallyInfeasible);
    }

    /// ... and while the sub-problem has *not* converged, the answer is
    /// still `Continue` — layer 2 must not fire on a restoration that is
    /// simply still working.
    #[test]
    fn unconverged_sub_problem_still_continues() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 5.96e-8, 5.05e-3, 0.0, 1e-8, || false, || false);
        let s = cc.check_convergence(1, false, 5.96e-8, 5.05e-3, 0.0, 1e-8, || false, || false);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    /// Layer 1 keeps precedence: when the trial point can leave
    /// restoration, that is the answer regardless of the sub-problem's own
    /// state.
    #[test]
    fn layer1_acceptance_outranks_layer2() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.5, 0.0, 1e-8, || true, || true);
        let s = cc.check_convergence(1, false, 1.0, 0.5, 0.0, 1e-8, || true, || true);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn layer2_tightening_arm_shrinks_the_sub_problem_tolerance() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.tol = 1e-8;
        // Slightly infeasible (above `orig_tol = 1e-8`), so the arm applies.
        cc.check_convergence(0, false, 1.0, 1e-7, 0.0, 1e-8, || false, || true);
        let s = cc.check_convergence(1, false, 1.0, 1e-7, 0.0, 1e-8, || false, || true);
        assert_eq!(s, RestoConvergenceStatus::Continue);
        assert_eq!(cc.tol, 1e-10);
        // Second visit: budget spent, so the verdict is now terminal. At
        // `tol = 1e-10` the point no longer clears `1e2 · tol`, so the
        // terminal verdict for this still-infeasible point is a local
        // infeasibility. (The feasible-point counterpart is covered by
        // `layer2_feasible_point_when_tolerance_budget_is_spent`.)
        let s = cc.check_convergence(2, false, 1.0, 1e-7, 0.0, 1e-8, || false, || true);
        assert_eq!(s, RestoConvergenceStatus::LocallyInfeasible);
        assert_eq!(cc.tol, 1e-10, "no further tightening");
    }

    #[test]
    fn test_orig_progress_converges_when_filter_and_iterate_accept() {
        use pounce_algorithm::line_search::filter_acceptor::FilterLsAcceptor;
        let outer = FilterLsAcceptor::new();
        let cc = RestoFilterConvCheck::new();
        // Empty filter ⇒ filter-acceptable; trial dominates reference
        // ⇒ iterate-acceptable.
        let s = cc.test_orig_progress(&outer, 0.5, 0.1, 1.0, 1.0);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn test_orig_progress_continues_when_filter_dominates() {
        use pounce_algorithm::line_search::filter_acceptor::FilterLsAcceptor;
        let mut outer = FilterLsAcceptor::new();
        // Plant a filter entry that dominates the trial.
        outer.filter.add(0.05, 0.4, 0);
        let cc = RestoFilterConvCheck::new();
        // (theta_trial=0.1, barr_trial=0.5) is dominated by (0.05,0.4).
        let s = cc.test_orig_progress(&outer, 0.5, 0.1, 1.0, 1.0);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn test_orig_progress_continues_when_iterate_rejects() {
        use pounce_algorithm::line_search::filter_acceptor::FilterLsAcceptor;
        let outer = FilterLsAcceptor::new();
        let cc = RestoFilterConvCheck::new();
        // trial_theta == reference_theta (no theta progress); trial_barr
        // > reference_barr (no phi progress) ⇒ iterate-acceptance fails.
        let s = cc.test_orig_progress(&outer, 2.0, 1.0, 1.0, 1.0);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn adapter_converges_at_inner_stationarity_tol() {
        use pounce_algorithm::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
        let mut a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000);
        // nlp_err well below tol ⇒ converged on iter 0.
        assert_eq!(a.check_convergence(1e-12, 0), ConvergenceStatus::Converged);
    }

    #[test]
    fn adapter_caps_at_maximum_resto_iters() {
        use pounce_algorithm::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
        let mut a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 1000, 2);
        // First two calls bump the resto counter; third call sees
        // successive_resto_iter == 2 == maximum_resto_iters and trips
        // the cap before bumping further.
        assert_eq!(a.check_convergence(1.0, 0), ConvergenceStatus::Continue);
        assert_eq!(a.check_convergence(1.0, 1), ConvergenceStatus::Continue);
        assert_eq!(
            a.check_convergence(1.0, 2),
            ConvergenceStatus::MaxIterExceeded
        );
    }

    #[test]
    fn adapter_caps_at_outer_max_iter() {
        use pounce_algorithm::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
        let mut a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 5, 3000);
        assert_eq!(
            a.check_convergence(1.0, 5),
            ConvergenceStatus::MaxIterExceeded
        );
    }

    #[test]
    fn with_orig_progress_callback_records_callback() {
        // Wiring smoke test: builder records the callback and the
        // adapter is constructible with both guards live. The full
        // gate (kappa-reduction AND callback-accept ⇒ Converged) is
        // exercised by the `restoration_triggers` integration test
        // through the nested IPM.
        let cb: pounce_algorithm::restoration::OrigProgressCallback =
            Box::new(|_barr: f64, _theta: f64| true);
        let a =
            RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000).with_orig_progress_callback(cb);
        assert!(a.orig_progress_callback.is_some());
    }

    /// Layer 2 is inert until the orig NLP is wired, because the verdict
    /// cannot be rendered without evaluating the recovered point. An
    /// adapter carrying only the scalars must keep reporting whatever the
    /// sub-problem's own check said.
    #[test]
    fn layer2_is_inert_without_the_orig_nlp() {
        use pounce_algorithm::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
        let mut a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000)
            .with_orig_convergence_verdict(1e-8, 1e-4, false);
        assert!(a.orig_nlp.is_none());
        assert_eq!(a.check_convergence(1e-12, 0), ConvergenceStatus::Converged);
    }

    #[test]
    fn with_orig_convergence_verdict_records_the_orig_scalars() {
        // The wired-in verdict needs a live `(data, cq)` pair and an orig
        // NLP; the four arms themselves are covered by the
        // `resto_orig_verdict` tests above, and the end-to-end path by the
        // `restoration_triggers` integration test.
        let a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000)
            .with_orig_convergence_verdict(1e-7, 1e-3, true);
        assert_eq!(a.orig_tol, 1e-7);
        assert_eq!(a.orig_constr_viol_tol, 1e-3);
        assert!(a.is_square_problem);
        assert_eq!(a.resto_tol_tightenings, 0);
    }

    /// The tightening arm chases a *residual violation*. Once the
    /// recovered point is feasible to the orig NLP's own `tol` there is
    /// nothing left to chase, and tightening drives the sub-solve past
    /// the point the outer wanted: eigmaxa/eigmina (7.5e-15) regressed
    /// Optimal → Restoration_Failed, dallasm (1.5e-10) Optimal →
    /// Error_In_Step_Computation. Such a point is the feasible-point arm's.
    #[test]
    fn layer2_does_not_tighten_at_an_already_feasible_point() {
        let orig_tol = 1e-8;
        // eigmaxa's number: feasible by seven orders. Not the tightening
        // arm's business — hand it back.
        let v = resto_orig_verdict(7.5e-15, 0.0, 1e-8, orig_tol, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::ConvergedToFeasiblePoint);
        // dallasm's number, same verdict.
        let v = resto_orig_verdict(1.53e-10, 0.0, 1e-8, orig_tol, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::ConvergedToFeasiblePoint);
        // Exactly at `orig_tol` is feasible — the premise is a strict `>`.
        let v = resto_orig_verdict(orig_tol, 0.0, 1e-8, orig_tol, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::ConvergedToFeasiblePoint);
        // Genuinely *slightly infeasible* — above `tol`, inside `1e2 · tol`
        // — is what the arm was ported for, and it still fires there.
        let v = resto_orig_verdict(1e-7, 0.0, 1e-8, orig_tol, 1e-4, 1e-2, false);
        assert_eq!(v, RestoOrigVerdict::TightenAndContinue);
    }

    /// `new` must leave the verdict's scalars self-consistent for callers
    /// that never wire them: `orig_tol == tol` is what makes the tightening
    /// arm's bound meaningful rather than accidental.
    #[test]
    fn adapter_defaults_orig_tol_to_the_sub_problem_tol() {
        let a = RestoConvCheckAdapter::new(1e-5, 1e-4, 15, 3000, 3000);
        assert_eq!(a.orig_tol, 1e-5);
        assert_eq!(a.inner_tol(), 1e-5);
        assert!(!a.is_square_problem);
    }

    #[test]
    fn with_orig_progress_guard_stores_reference_and_kappa() {
        // Construction-only check: the builder records the orig-NLP
        // handle, the outer-curr inf_pr snapshot, and kappa. The
        // wired-in early-exit behavior is exercised by the
        // `restoration_triggers` integration test (which now drives
        // the inner IPM through the full guard).
        let a = RestoConvCheckAdapter::new(1e-8, 1e-6, 15, 3000, 3000);
        assert!(a.orig_nlp.is_none());
        assert!(a.orig_curr_inf_pr.is_infinite());
        assert_eq!(a.kappa_resto, 0.9);
    }
}
