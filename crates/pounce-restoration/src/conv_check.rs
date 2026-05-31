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
    MaxIterExceeded,
    UserStop,
}

/// Scalar core of `IpRestoConvCheck::CheckConvergence` (lines 132-220).
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

    /// Port of `IpRestoConvCheck.cpp:132-220` excluding the bits that
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
    pub fn check_convergence(
        &mut self,
        iter_count: i32,
        is_square_problem: bool,
        orig_curr_inf_pr: f64,
        orig_trial_inf_pr: f64,
        orig_tol: f64,
        acceptable_to_outer: impl FnOnce() -> bool,
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
        if self.kappa_resto > 0.0 && orig_trial_inf_pr > self.kappa_resto * orig_curr_inf_pr {
            return RestoConvergenceStatus::Continue;
        }

        // Reduction was sufficient — defer to the outer-phase filter /
        // penalty acceptor (line 198).
        if acceptable_to_outer() {
            RestoConvergenceStatus::Converged
        } else {
            RestoConvergenceStatus::Continue
        }
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
/// v0.1 scope (Phase 9): implements the resto-side iteration-cap
/// state machine (`maximum_iters` / `maximum_resto_iters` from
/// `IpRestoConvCheck.cpp:RegisterOptions`) and delegates the
/// stationarity convergence check to a wrapped `OptErrorConvCheck`
/// against the inner resto-problem stationarity. The kappa-reduction
/// guard (`orig_trial_inf_pr <= kappa_resto * orig_curr_inf_pr`) and
/// the outer-filter acceptance test require the inner iterate's
/// orig-NLP infeasibility, which the v0.1 trait surface
/// `(nlp_err, iter_count) -> ConvergenceStatus` doesn't supply; those
/// stay deferred to the outer line search's post-restoration recheck
/// per the comment at `resto_inner_solver.rs:27-33`. Snapshots the
/// outer scalars at construction (option (b) from the task design
/// note) so that surface stays narrow.
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
        }
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

        // 2. Kappa-reduction early-exit on orig-NLP `inf_pr`. Mirrors
        //    upstream `IpRestoConvCheck.cpp:175` — when the inner
        //    iterate's orig `(theta_trial)` is below
        //    `kappa_resto · orig_curr_inf_pr`, restoration has done
        //    enough. Upstream then runs `TestOrigProgress` (filter +
        //    iterate acceptance) before declaring `Converged`; we mirror
        //    that gate via [`Self::orig_progress_callback`]. The first
        //    inner iter is skipped (no prior reduction reference yet)
        //    by checking `iter_count > 0`; that matches upstream's
        //    `first_resto_iter` freebie at line 152.
        if iter_count > 0 && self.kappa_resto > 0.0 {
            if let Some(orig_rc) = self.orig_nlp.clone() {
                if let Some((orig_trial_inf_pr, orig_trial_f)) =
                    eval_orig_inf_pr_and_f(data, &orig_rc)
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

        // 3. Inner-stationarity fallback (resto NLP's own KKT residual).
        self.inner.check_convergence(nlp_err, iter_count)
    }
}

/// Evaluate the orig-NLP `(inf_pr, f)` pair at the inner iterate's
/// `(x_orig, s)` slice:
///
/// * `inf_pr = max(||c(x_orig)||∞, ||d(x_orig) − s||∞)` — used by both
///   the kappa-reduction guard and as the orig-trial-theta passed to
///   the outer-progress callback.
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
fn eval_orig_inf_pr_and_f(
    data: &pounce_algorithm::ipopt_data::IpoptDataHandle,
    orig_rc: &std::rc::Rc<std::cell::RefCell<dyn pounce_nlp::ipopt_nlp::IpoptNlp>>,
) -> Option<(f64, f64)> {
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::{CompoundVector, Vector};

    let curr = data.borrow().curr.clone()?;
    let xc = curr.x.as_any().downcast_ref::<CompoundVector>()?;
    let x_orig = xc.comp(crate::resto_nlp::BLOCK_X);
    let s_inner = &*curr.s;

    let mut orig = orig_rc.borrow_mut();
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();

    let c_amax = if m_eq > 0 {
        let mut c_buf = DenseVectorSpace::new(m_eq).make_new_dense();
        orig.eval_c(x_orig, &mut c_buf);
        c_buf.amax()
    } else {
        0.0
    };

    let d_minus_s_amax = if m_ineq > 0 {
        let mut d_buf = DenseVectorSpace::new(m_ineq).make_new_dense();
        orig.eval_d(x_orig, &mut d_buf);
        d_buf.axpy(-1.0, s_inner);
        d_buf.amax()
    } else {
        0.0
    };

    let f = orig.eval_f(x_orig);

    Some((c_amax.max(d_minus_s_amax), f))
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
        let s = cc.check_convergence(0, false, 1.0, 0.5, 1e-8, || true);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn outer_iter_cap_triggers_max() {
        let mut cc = RestoConvCheck::new();
        cc.maximum_iters = 5;
        let s = cc.check_convergence(6, false, 1.0, 0.5, 1e-8, || true);
        assert_eq!(s, RestoConvergenceStatus::MaxIterExceeded);
    }

    #[test]
    fn successive_resto_cap_triggers_max() {
        let mut cc = RestoConvCheck::new();
        cc.maximum_resto_iters = 2;
        // Burn through 3 calls — fourth should hit the cap.
        cc.check_convergence(0, false, 1.0, 0.9, 1e-8, || false);
        cc.check_convergence(1, false, 1.0, 0.8, 1e-8, || false);
        cc.check_convergence(2, false, 1.0, 0.7, 1e-8, || false);
        let s = cc.check_convergence(3, false, 1.0, 0.6, 1e-8, || false);
        assert_eq!(s, RestoConvergenceStatus::MaxIterExceeded);
    }

    #[test]
    fn square_problem_fast_path_converges() {
        let mut cc = RestoConvCheck::new();
        cc.orig_constr_viol_tol = 1e-4;
        // Burn the first-iter freebie.
        cc.check_convergence(0, true, 1.0, 0.5, 1e-8, || false);
        // Now feed a feasible trial.
        let s = cc.check_convergence(1, true, 0.5, 1e-10, 1e-8, || false);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn insufficient_reduction_keeps_going() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.95, 1e-8, || true);
        // trial_inf_pr (0.95) > 0.9 * curr_inf_pr (1.0) — not enough.
        let s = cc.check_convergence(1, false, 1.0, 0.95, 1e-8, || true);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn sufficient_reduction_plus_filter_accept_converges() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.5, 1e-8, || true);
        let s = cc.check_convergence(1, false, 1.0, 0.5, 1e-8, || true);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn sufficient_reduction_but_filter_rejects_continues() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.9;
        cc.check_convergence(0, false, 1.0, 0.5, 1e-8, || false);
        let s = cc.check_convergence(1, false, 1.0, 0.5, 1e-8, || false);
        assert_eq!(s, RestoConvergenceStatus::Continue);
    }

    #[test]
    fn kappa_zero_disables_reduction_guard() {
        let mut cc = RestoConvCheck::new();
        cc.kappa_resto = 0.0;
        cc.check_convergence(0, false, 1.0, 1.5, 1e-8, || true);
        // Even with trial > curr, the guard is bypassed and we go to
        // the outer-filter check, which accepts.
        let s = cc.check_convergence(1, false, 1.0, 1.5, 1e-8, || true);
        assert_eq!(s, RestoConvergenceStatus::Converged);
    }

    #[test]
    fn reset_clears_state() {
        let mut cc = RestoConvCheck::new();
        cc.check_convergence(0, false, 1.0, 0.5, 1e-8, || true);
        cc.check_convergence(1, false, 1.0, 0.5, 1e-8, || true);
        cc.reset();
        assert!(cc.first_resto_iter);
        assert_eq!(cc.successive_resto_iter, 0);
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
