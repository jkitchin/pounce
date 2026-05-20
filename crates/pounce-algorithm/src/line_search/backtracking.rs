//! Backtracking line-search driver — port of
//! `Algorithm/IpBacktrackingLineSearch.{hpp,cpp}`.
//!
//! Owns the alpha-reduction loop, max-soc / second-order-correction
//! slot, watchdog mechanism, and the fallback to restoration. Phase 7
//! ships the alpha-loop for the filter line search; SOC and watchdog
//! land alongside the restoration phase (Phase 9).
//!
//! The contract with the acceptor is the trio
//! `(theta, phi, d_phi)` at the current iterate plus the trial
//! `(theta_trial, phi_trial)` per backtracking step. Trial-point
//! construction is `x_trial = x + α·dx`, `s_trial = s + α·ds`; the dual
//! step uses the same α for the filter acceptor (upstream
//! `IpBacktrackingLineSearch.cpp:702-728` — primal-dual share α
//! when no fraction-to-the-boundary truncation differs).
//!
//! `find_acceptable_trial_point` returns `Outcome::Accepted` on a
//! successful trial, `Outcome::TinyStep` when α drops below
//! `alpha_min`, and `Outcome::Failed` when the alpha loop exhausts
//! without acceptance (which the main loop maps to a restoration
//! attempt).

use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::pd_search_dir_calc::PdSearchDirCalc;
use crate::line_search::filter_acceptor::AcceptDecision;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use pounce_common::types::Number;
use std::cell::RefCell;
use std::rc::Rc;

/// Outcome of the backtracking line search. Mirrors the booleans
/// upstream returns through `accept_` plus the `tiny_step_flag` on
/// `IpoptData`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Trial point accepted at the recorded `alpha`.
    Accepted,
    /// `alpha` fell below `alpha_min_frac` × current α₀ ⇒ tiny step.
    /// Caller maps to `STEP_BECOMES_TINY` in upstream's exception flow.
    TinyStep,
    /// All α reductions rejected; the caller hands off to restoration.
    Failed,
}

pub struct BacktrackingLineSearch {
    pub acceptor: Box<dyn BacktrackingLsAcceptor>,
    pub alpha_red_factor: Number,
    pub max_soc: i32,
    /// Threshold for the SOC outer-loop convergence test
    /// `theta_trial <= kappa_soc * theta_soc_old`. Mirrors upstream's
    /// `kappa_soc` (default 0.99).
    pub kappa_soc: Number,
    /// SOC RHS variant. `0` = upstream default ("old"), `1` = scaled
    /// gradient-block variant. Both correspond to upstream's
    /// `soc_method` option.
    pub soc_method: i32,
    /// Number of consecutive shortened iterations before the watchdog
    /// procedure activates. Disabled when `<= 0`. Mirrors upstream's
    /// `watchdog_shortened_iter_trigger` (default 10).
    pub watchdog_shortened_iter_trigger: i32,
    /// Maximum number of outer iterations the watchdog will accept
    /// non-decreasing trial points before reverting to the snapshot.
    /// Mirrors upstream's `watchdog_trial_iter_max` (default 3).
    pub watchdog_trial_iter_max: i32,
    /// Lower bound on α; below this we declare a tiny step (mirrors
    /// `alpha_min_frac` flow, `IpBacktrackingLineSearch.cpp:CalculateAlphaMin`).
    pub alpha_min: Number,
    /// Maximum trial-iteration cap before declaring failure.
    pub max_trials: i32,

    // ---- Watchdog state (port of `IpBacktrackingLineSearch.{hpp,cpp}`'s
    //      `in_watchdog_`, `watchdog_iterate_`, `watchdog_delta_`,
    //      `watchdog_alpha_primal_test_`, `watchdog_trial_iter_`,
    //      `watchdog_shortened_iter_`, `last_mu_`).
    //
    // Watchdog mechanism: after `watchdog_shortened_iter_trigger`
    // consecutive shortened (n_steps > 0) accepts, we snapshot the
    // current iterate `(curr, delta, theta, phi, d_phi)` and enter
    // watchdog mode. While in watchdog: the acceptor's reference
    // values are FROZEN to the snapshot for up to
    // `watchdog_trial_iter_max` outer iterations. Each iteration's
    // alpha-loop runs against the frozen reference; if it accepts,
    // watchdog terminates with success ("W"). If it rejects, we
    // accept the last trial anyway (info char 'w') and let the next
    // outer iteration try again. If `watchdog_trial_iter_max` outer
    // iterations all reject, we revert to the snapshot and re-run
    // the alpha-loop on the saved `delta` with `skip_first=true`.
    /// True iff currently inside a watchdog window.
    in_watchdog: bool,
    /// Snapshot of the iterate at watchdog activation.
    watchdog_iterate: Option<IteratesVector>,
    /// Snapshot of the search direction at watchdog activation.
    watchdog_delta: Option<IteratesVector>,
    /// Snapshot of `primal_frac_to_the_bound(τ, δ)` at watchdog
    /// activation. Currently unused inside the alpha loop (pounce's
    /// driver passes `alpha_init` directly), but stored for parity
    /// with upstream's iter-by-iter trace.
    #[allow(dead_code)]
    watchdog_alpha_primal_test: Number,
    /// Number of outer iterations elapsed since watchdog activation.
    watchdog_trial_iter: i32,
    /// Number of consecutive shortened (n_steps > 0) accepts.
    /// Reset on a full step (n_steps == 0), on mu change, on watchdog
    /// success, and on watchdog stop-with-revert.
    watchdog_shortened_iter: i32,
    /// `mu` at the previous outer iteration. A change clears the
    /// watchdog state (`IpBacktrackingLineSearch.cpp:259-270`).
    last_mu: Number,
    /// Frozen reference theta at watchdog activation.
    watchdog_theta: Number,
    /// Frozen reference phi at watchdog activation.
    watchdog_phi: Number,
    /// Frozen reference d_phi at watchdog activation.
    watchdog_d_phi: Number,

    // ---- Soft restoration phase (port of `IpBacktrackingLineSearch`'s
    //      `in_soft_resto_phase_`, `soft_resto_counter_`).
    //
    // When the regular filter line search fails, before handing off to
    // the full (sub-NLP) restoration phase, the driver tries a single
    // damped primal-dual step along the *same* search direction. The
    // step is damped only by the fraction-to-the-boundary rule and is
    // accepted if it either satisfies the original filter criterion
    // ('S' — leave soft resto) or merely reduces the primal-dual KKT
    // system error by `soft_resto_pderror_reduction_factor` ('s' —
    // stay in soft resto). Subsequent outer iterations keep taking
    // soft-resto steps until the original criterion is met, the step
    // is rejected, or `max_soft_resto_iters` consecutive iterations
    // elapse — any of which drops through to full restoration.
    /// Required relative reduction in the primal-dual system error for
    /// a soft-resto step to be accepted. `0` disables soft restoration.
    /// Mirrors upstream `soft_resto_pderror_reduction_factor`
    /// (default `1 - 1e-4`).
    pub soft_resto_pderror_reduction_factor: Number,
    /// Cap on consecutive soft-resto iterations before full
    /// restoration is forced. Mirrors upstream `max_soft_resto_iters`
    /// (default 10).
    pub max_soft_resto_iters: i32,
    /// True iff the driver is currently inside the soft-resto phase.
    in_soft_resto_phase: bool,
    /// Count of consecutive soft-resto iterations taken so far.
    soft_resto_counter: i32,
}

/// Internal alpha-loop outcome. The watchdog wrapper translates this
/// into the public [`Outcome`] after applying its state machine.
enum AlphaResult {
    /// Trial accepted at `alpha_used` after `n_steps` reductions.
    Accepted { n_steps: i32 },
    /// α dropped below `alpha_min_eff` ⇒ tiny step. `last_alpha` is
    /// the smallest α actually evaluated; `n_steps` is the number of
    /// reductions performed.
    TinyStep { n_steps: i32, last_alpha: Number },
    /// `max_trials` exhausted without acceptance. The last attempted
    /// trial iterate is left in `data.trial` so the watchdog
    /// "accept-anyway" path can promote it.
    ///
    /// `evaluation_error` flags that the last attempted trial produced
    /// a non-finite `theta_trial`/`phi_trial` — mirrors upstream's
    /// `evaluation_error` tracked from `IpoptNLP::Eval_Error`
    /// (`IpBacktrackingLineSearch.cpp:776-784`). The watchdog handler
    /// must treat this as a forced StopWatchDog
    /// (`IpBacktrackingLineSearch.cpp:493`) — accepting a non-finite
    /// iterate via the 'w' branch propagates NaN/Inf into the next
    /// outer iter (observed on PFIT3 iter 53: inf_pr=7.87e305 from a
    /// 'w'-accepted trial; on PFIT4 iter 31: inf_pr=1.01e11).
    Failed {
        n_steps: i32,
        last_alpha: Number,
        evaluation_error: bool,
    },
}

impl BacktrackingLineSearch {
    pub fn new(acceptor: Box<dyn BacktrackingLsAcceptor>) -> Self {
        Self {
            acceptor,
            alpha_red_factor: 0.5,
            max_soc: 4,
            kappa_soc: 0.99,
            soc_method: 0,
            watchdog_shortened_iter_trigger: 10,
            watchdog_trial_iter_max: 3,
            alpha_min: 1e-12,
            max_trials: 50,
            in_watchdog: false,
            watchdog_iterate: None,
            watchdog_delta: None,
            watchdog_alpha_primal_test: 0.0,
            watchdog_trial_iter: 0,
            watchdog_shortened_iter: 0,
            last_mu: -1.0,
            watchdog_theta: 0.0,
            watchdog_phi: 0.0,
            watchdog_d_phi: 0.0,
            soft_resto_pderror_reduction_factor: 1.0 - 1e-4,
            max_soft_resto_iters: 10,
            in_soft_resto_phase: false,
            soft_resto_counter: 0,
        }
    }

    /// Test-only accessor for the watchdog active flag.
    #[cfg(test)]
    pub(crate) fn in_watchdog(&self) -> bool {
        self.in_watchdog
    }

    /// Test-only accessor for the shortened-iter counter.
    #[cfg(test)]
    pub(crate) fn watchdog_shortened_iter(&self) -> i32 {
        self.watchdog_shortened_iter
    }

    pub fn acceptor(&self) -> &dyn BacktrackingLsAcceptor {
        &*self.acceptor
    }

    pub fn acceptor_mut(&mut self) -> &mut dyn BacktrackingLsAcceptor {
        &mut *self.acceptor
    }

    /// Reset the acceptor state at the start of a new outer iteration.
    pub fn reset(&mut self) {
        self.acceptor.reset();
    }

    /// Public line-search entry point. Wraps the regular filter line
    /// search ([`Self::run_filter_line_search`]) with the soft
    /// restoration phase — port of the `in_soft_resto_phase_` state
    /// machine in `IpBacktrackingLineSearch::FindAcceptableTrialPoint`
    /// (`IpBacktrackingLineSearch.cpp:439-465` for the in-phase
    /// continuation, `:528-556` for entering the phase).
    ///
    /// Outcomes:
    /// - `Accepted`: a trial point is in `data.trial` — either a
    ///   regular filter/watchdog step or a soft-resto step (info char
    ///   's' = stay in soft resto, 'S' = step also satisfies the
    ///   original filter so soft resto is left).
    /// - `TinyStep` / `Failed`: neither the regular line search nor a
    ///   soft-resto step could make progress; the caller hands off to
    ///   the full restoration phase.
    #[allow(clippy::too_many_arguments)]
    pub fn find_acceptable_trial_point(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
        alpha_init: Number,
        alpha_dual: Number,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        search_dir: Option<&mut PdSearchDirCalc>,
    ) -> Outcome {
        // ---- Soft-resto continuation. Already inside the phase: bump
        // the counter, bail to full restoration once it exceeds
        // `max_soft_resto_iters`, otherwise take another damped
        // primal-dual step along the caller's `delta`
        // (`IpBacktrackingLineSearch.cpp:439-465`).
        if self.in_soft_resto_phase {
            self.soft_resto_counter += 1;
            if self.soft_resto_counter > self.max_soft_resto_iters {
                self.in_soft_resto_phase = false;
                self.soft_resto_counter = 0;
                return self.fail_to_restoration(data);
            }
            // Per-outer-iteration acceptor hook (no-op for the filter
            // acceptor; the penalty acceptor caches its reference here).
            self.acceptor.init_this_line_search(data, cq, delta);
            return match self.try_soft_resto_step(data, cq, delta) {
                Some(satisfies_original) => {
                    if satisfies_original {
                        self.in_soft_resto_phase = false;
                        self.soft_resto_counter = 0;
                        data.borrow_mut().info_alpha_primal_char = 'S';
                    } else {
                        data.borrow_mut().info_alpha_primal_char = 's';
                    }
                    Outcome::Accepted
                }
                None => {
                    self.in_soft_resto_phase = false;
                    self.soft_resto_counter = 0;
                    self.fail_to_restoration(data)
                }
            };
        }

        // ---- Regular filter line search (watchdog + alpha loop).
        let outcome =
            self.run_filter_line_search(data, cq, delta, alpha_init, alpha_dual, nlp, search_dir);
        if outcome == Outcome::Accepted {
            return Outcome::Accepted;
        }

        // ---- Regular line search failed. Before the (expensive) full
        // restoration sub-NLP, try to *enter* the soft restoration
        // phase with one damped primal-dual step
        // (`IpBacktrackingLineSearch.cpp:528-556`). `prepare_resto_phase_start`
        // augments the outer filter with the entry envelope — mirrors
        // upstream's `acceptor_->PrepareRestoPhaseStart()` at line 537.
        let reference_theta = cq.borrow().curr_constraint_violation();
        let reference_barr = cq.borrow().curr_barrier_obj();
        self.acceptor
            .prepare_resto_phase_start(reference_theta, reference_barr);
        match self.try_soft_resto_step(data, cq, delta) {
            Some(satisfies_original) => {
                if satisfies_original {
                    data.borrow_mut().info_alpha_primal_char = 'S';
                } else {
                    self.in_soft_resto_phase = true;
                    self.soft_resto_counter = 0;
                    data.borrow_mut().info_alpha_primal_char = 's';
                }
                Outcome::Accepted
            }
            // Soft resto could not help — fall through to full
            // restoration with the original failure outcome. The
            // caller's `invoke_restoration` re-runs
            // `prepare_resto_phase_start`; the duplicate filter
            // augmentation is idempotent (same envelope).
            None => outcome,
        }
    }

    /// Stamp the info fields for a hand-off to the full restoration
    /// phase and return `Outcome::Failed`. Used when the soft
    /// restoration phase exhausts its iteration budget or its step is
    /// rejected mid-phase.
    fn fail_to_restoration(&self, data: &IpoptDataHandle) -> Outcome {
        let mut d = data.borrow_mut();
        d.trial = None;
        d.info_alpha_primal = 0.0;
        d.info_alpha_dual = 0.0;
        d.info_alpha_primal_char = 'R';
        d.info_ls_count = 0;
        Outcome::Failed
    }

    /// Attempt a single damped primal-dual step for the soft
    /// restoration phase — port of
    /// `BacktrackingLineSearch::TrySoftRestoStep`
    /// (`IpBacktrackingLineSearch.cpp:1112-1217`). The step along
    /// `delta` is damped only by the fraction-to-the-boundary rule,
    /// with an identical step length for primal and dual variables.
    ///
    /// Returns:
    /// - `Some(true)`  — trial accepted *and* it satisfies the
    ///   original filter criterion ⇒ caller leaves soft resto ('S').
    /// - `Some(false)` — trial accepted only on the primal-dual error
    ///   reduction test ⇒ caller stays in soft resto ('s').
    /// - `None`        — trial rejected (or soft resto disabled / a
    ///   non-finite evaluation) ⇒ caller falls through to the full
    ///   restoration phase.
    ///
    /// On a `Some(_)` return the accepted trial is left in `data.trial`
    /// and the numeric `info_*` fields are stamped; the caller stamps
    /// `info_alpha_primal_char`.
    fn try_soft_resto_step(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
    ) -> Option<bool> {
        // Soft restoration is disabled when the reduction factor is
        // zero (`IpBacktrackingLineSearch.cpp:1124`).
        if self.soft_resto_pderror_reduction_factor == 0.0 {
            return None;
        }
        let curr = data.borrow().curr.clone()?;
        let tau = data.borrow().curr_tau;

        // Identical step length for primal and dual variables, damped
        // only by the fraction-to-the-boundary rule
        // (`IpBacktrackingLineSearch.cpp:1135-1140`).
        let alpha = {
            let cq_ref = cq.borrow();
            cq_ref
                .aff_step_alpha_primal_max(delta, tau)
                .min(cq_ref.aff_step_alpha_dual_max(delta, tau))
        };

        let trial_iv = scaled_step(&curr, delta, alpha, alpha);
        data.borrow_mut().set_trial(trial_iv);

        let theta_trial = cq.borrow().trial_constraint_violation();
        let phi_trial = cq.borrow().trial_barrier_obj();
        if !theta_trial.is_finite() || !phi_trial.is_finite() {
            // Upstream retries up to three times on `Eval_Error`; the
            // step length is fixed, so a non-finite eval here is
            // deterministic — treat it as a rejection.
            return None;
        }

        let theta = cq.borrow().curr_constraint_violation();
        let phi = cq.borrow().curr_barrier_obj();
        let d_phi = self.compute_d_phi(cq, delta);

        // First test: is the trial acceptable to the *original*
        // backtracking globalization? Upstream
        // `acceptor_->CheckAcceptabilityOfTrialPoint(0.)`.
        if self
            .acceptor
            .check_trial_point(0.0, theta, phi, d_phi, theta_trial, phi_trial)
            == AcceptDecision::Accept
        {
            let mut d = data.borrow_mut();
            d.info_alpha_primal = alpha;
            d.info_alpha_dual = alpha;
            d.info_ls_count = 1;
            return Some(true);
        }

        // Second test: sufficient reduction in the primal-dual KKT
        // system error (`IpBacktrackingLineSearch.cpp:1184-1211`).
        let mu = data.borrow().curr_mu;
        let curr_pderror = cq.borrow().curr_primal_dual_system_error(mu);
        let trial_pderror = cq.borrow().trial_primal_dual_system_error(mu);
        if !trial_pderror.is_finite() {
            return None;
        }
        if trial_pderror <= self.soft_resto_pderror_reduction_factor * curr_pderror {
            let mut d = data.borrow_mut();
            d.info_alpha_primal = alpha;
            d.info_alpha_dual = alpha;
            d.info_ls_count = 1;
            return Some(false);
        }
        None
    }

    /// Drive the watchdog state machine + alpha-reduction loop.
    /// Port of `IpBacktrackingLineSearch::FindAcceptableTrialPoint`
    /// (`IpBacktrackingLineSearch.cpp:252-677`) restricted to the
    /// regular (non-soft-resto) filter-acceptor, exact-Hessian path.
    /// The soft restoration phase is layered on top by
    /// [`Self::find_acceptable_trial_point`].
    ///
    /// Outcomes:
    /// - `Accepted`: a trial point is in `data.trial`, info fields are
    ///   stamped. The watchdog state has been advanced (success → "W",
    ///   `accept-anyway` → 'w').
    /// - `TinyStep`: α dropped below the dynamic alpha-min before any
    ///   trial was accepted. Caller hands off to restoration.
    /// - `Failed`: alpha-loop exhausted AND watchdog could not rescue.
    ///   Caller hands off to restoration.
    #[allow(clippy::too_many_arguments)]
    fn run_filter_line_search(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
        alpha_init: Number,
        alpha_dual: Number,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        search_dir: Option<&mut PdSearchDirCalc>,
    ) -> Outcome {
        // ---- Watchdog: detect mu change → reset state.
        // Mirrors `IpBacktrackingLineSearch.cpp:259-270`.
        let curr_mu = data.borrow().curr_mu;
        if self.last_mu < 0.0 || self.last_mu != curr_mu {
            self.in_watchdog = false;
            self.watchdog_iterate = None;
            self.watchdog_delta = None;
            self.watchdog_shortened_iter = 0;
            self.last_mu = curr_mu;
        }

        // ---- Watchdog: maybe wake up.
        // Mirrors `IpBacktrackingLineSearch.cpp:376-380`.
        if !self.in_watchdog
            && self.watchdog_shortened_iter_trigger > 0
            && self.watchdog_shortened_iter >= self.watchdog_shortened_iter_trigger
        {
            self.start_watchdog(data, cq, delta);
        }

        // Per-outer-iteration acceptor hook.
        self.acceptor.init_this_line_search(data, cq, delta);

        // Decide reference (theta, phi, d_phi). Mirrors upstream's
        // `FilterLSAcceptor::InitThisLineSearch(in_watchdog)` choice
        // between `curr_*` and the saved `watchdog_*` snapshot.
        let (theta, phi, d_phi) = if self.in_watchdog {
            (self.watchdog_theta, self.watchdog_phi, self.watchdog_d_phi)
        } else {
            let theta = cq.borrow().curr_constraint_violation();
            let phi = cq.borrow().curr_barrier_obj();
            let d_phi = self.compute_d_phi(cq, delta);
            (theta, phi, d_phi)
        };

        // Run the alpha-loop on the caller's `delta`.
        let result = self.run_alpha_loop(
            data, cq, delta, alpha_init, alpha_dual, nlp, search_dir, theta, phi, d_phi,
            /*skip_first*/ false,
        );

        match result {
            AlphaResult::Accepted { n_steps } => {
                // Update the shortened-iter counter
                // (`IpBacktrackingLineSearch.cpp:644-655`).
                if n_steps == 0 {
                    self.watchdog_shortened_iter = 0;
                } else {
                    self.watchdog_shortened_iter += 1;
                }
                if self.in_watchdog {
                    // Watchdog success — clear state, info char already
                    // stamped by the alpha loop's
                    // `update_for_next_iteration` call. Upstream also
                    // appends "W" to the info string here; pounce
                    // doesn't track an info string yet.
                    self.in_watchdog = false;
                    self.watchdog_iterate = None;
                    self.watchdog_delta = None;
                    self.watchdog_shortened_iter = 0;
                }
                Outcome::Accepted
            }
            AlphaResult::TinyStep {
                n_steps,
                last_alpha,
            } => {
                let mut d = data.borrow_mut();
                d.trial = None;
                d.info_alpha_primal = last_alpha;
                d.info_alpha_dual = 0.0;
                d.info_alpha_primal_char = 'R';
                d.info_ls_count = n_steps + 1;
                Outcome::TinyStep
            }
            AlphaResult::Failed {
                n_steps,
                last_alpha,
                evaluation_error,
            } => {
                if self.in_watchdog {
                    self.handle_watchdog_failure(
                        data,
                        cq,
                        alpha_init,
                        alpha_dual,
                        nlp,
                        n_steps,
                        last_alpha,
                        evaluation_error,
                    )
                } else {
                    // Genuine failure → restoration.
                    let mut d = data.borrow_mut();
                    d.trial = None;
                    d.info_alpha_primal = last_alpha;
                    d.info_alpha_dual = 0.0;
                    d.info_alpha_primal_char = 'R';
                    d.info_ls_count = n_steps + 1;
                    Outcome::Failed
                }
            }
        }
    }

    /// Snapshot the current `(curr, delta, theta, phi, d_phi)` and
    /// activate the watchdog. Mirrors upstream
    /// `IpBacktrackingLineSearch::StartWatchDog`
    /// (`IpBacktrackingLineSearch.cpp:855-869`) plus
    /// `IpFilterLSAcceptor::StartWatchDog`
    /// (`IpFilterLSAcceptor.cpp:506-513`) — pounce stores the
    /// frozen reference values directly on the driver because the
    /// acceptor is stateless w.r.t. reference values (the driver
    /// passes them per call).
    fn start_watchdog(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
    ) {
        let curr = data.borrow().curr.clone();
        let Some(curr) = curr else {
            return;
        };
        self.in_watchdog = true;
        self.watchdog_iterate = Some(curr);
        self.watchdog_delta = Some(delta.clone());
        self.watchdog_trial_iter = 0;
        let tau = data.borrow().curr_tau;
        self.watchdog_alpha_primal_test = cq.borrow().aff_step_alpha_primal_max(delta, tau);
        self.watchdog_theta = cq.borrow().curr_constraint_violation();
        self.watchdog_phi = cq.borrow().curr_barrier_obj();
        self.watchdog_d_phi = self.compute_d_phi(cq, delta);
    }

    /// Handle alpha-loop failure while in watchdog mode. Bumps
    /// `watchdog_trial_iter`; if the cap is exceeded, reverts to the
    /// snapshot (StopWatchDog) and re-runs the alpha-loop on the
    /// saved `delta` with `skip_first=true`. Otherwise accepts the
    /// current trial as 'w' and returns. Mirrors
    /// `IpBacktrackingLineSearch.cpp:480-503` together with
    /// `IpBacktrackingLineSearch.cpp:871-908`'s `StopWatchDog`.
    fn handle_watchdog_failure(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        alpha_init: Number,
        alpha_dual: Number,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        n_steps: i32,
        last_alpha: Number,
        evaluation_error: bool,
    ) -> Outcome {
        self.watchdog_trial_iter += 1;
        // Mirror upstream `IpBacktrackingLineSearch.cpp:493`:
        // `if (evaluation_error || watchdog_trial_iter > max)` →
        // StopWatchDog. A non-finite trial must NOT be promoted via
        // the 'w' accept-anyway path; doing so propagates NaN/Inf
        // into the next outer iter and the iterate is unrecoverable
        // (observed on PFIT3, PFIT4).
        if evaluation_error || self.watchdog_trial_iter > self.watchdog_trial_iter_max {
            // StopWatchDog: revert curr to the snapshot, re-run on
            // saved delta with `skip_first=true` (alpha starts at
            // `alpha_init * alpha_red_factor`).
            let snapshot_iter = self.watchdog_iterate.take();
            let snapshot_delta = self.watchdog_delta.take();
            self.in_watchdog = false;
            self.watchdog_shortened_iter = 0;
            let (Some(snap), Some(snap_delta)) = (snapshot_iter, snapshot_delta) else {
                // Defensive — this should not happen if start_watchdog
                // ran successfully. Fall through to genuine failure.
                let mut d = data.borrow_mut();
                d.trial = None;
                d.info_alpha_primal = last_alpha;
                d.info_alpha_dual = 0.0;
                d.info_alpha_primal_char = 'R';
                d.info_ls_count = n_steps + 1;
                return Outcome::Failed;
            };
            {
                let mut d = data.borrow_mut();
                d.set_curr(snap);
            }
            let theta = cq.borrow().curr_constraint_violation();
            let phi = cq.borrow().curr_barrier_obj();
            let d_phi = self.compute_d_phi(cq, &snap_delta);
            // SOC is disabled on the StopWatchDog retry. The original
            // `search_dir` was consumed by the first alpha-loop call
            // and we want a plain backtracking pass over the saved
            // delta; mirrors upstream's behavior of not running the
            // soc_method on the recovered search.
            let result2 = self.run_alpha_loop(
                data,
                cq,
                &snap_delta,
                alpha_init,
                alpha_dual,
                nlp,
                None,
                theta,
                phi,
                d_phi,
                /*skip_first*/ true,
            );
            match result2 {
                AlphaResult::Accepted { n_steps: ns2 } => {
                    if ns2 == 0 {
                        self.watchdog_shortened_iter = 0;
                    } else {
                        self.watchdog_shortened_iter += 1;
                    }
                    Outcome::Accepted
                }
                AlphaResult::TinyStep {
                    n_steps: ns2,
                    last_alpha: la2,
                } => {
                    let mut d = data.borrow_mut();
                    d.trial = None;
                    d.info_alpha_primal = la2;
                    d.info_alpha_dual = 0.0;
                    d.info_alpha_primal_char = 'R';
                    d.info_ls_count = ns2 + 1;
                    Outcome::TinyStep
                }
                AlphaResult::Failed {
                    n_steps: ns2,
                    last_alpha: la2,
                    evaluation_error: _,
                } => {
                    let mut d = data.borrow_mut();
                    d.trial = None;
                    d.info_alpha_primal = la2;
                    d.info_alpha_dual = 0.0;
                    d.info_alpha_primal_char = 'R';
                    d.info_ls_count = ns2 + 1;
                    Outcome::Failed
                }
            }
        } else {
            // Accept the last attempted trial despite filter rejection
            // — `accept-anyway` watchdog branch
            // (`IpBacktrackingLineSearch.cpp:498-503`). The trial
            // iterate from the final α attempt is already in
            // `data.trial`. Crucially, we do NOT call
            // `update_for_next_iteration`, so the filter is NOT
            // augmented (matching upstream's char='w' branch at
            // line 833-836 which skips `UpdateForNextIteration`).
            let mut d = data.borrow_mut();
            d.info_alpha_primal = last_alpha;
            d.info_alpha_dual = alpha_dual;
            d.info_alpha_primal_char = 'w';
            d.info_ls_count = n_steps + 1;
            Outcome::Accepted
        }
    }

    /// Inner alpha-reduction loop. Tries
    /// `alpha = alpha_init * alpha_red_factor^k` (or
    /// `alpha_red_factor^(k+1)` when `skip_first=true`) and consults
    /// the acceptor against the supplied reference `(theta, phi, d_phi)`.
    /// On accept stamps the info fields and calls
    /// `update_for_next_iteration`. On reject leaves the LAST trial in
    /// `data.trial` so the watchdog `accept-anyway` path can promote
    /// it.
    #[allow(clippy::too_many_arguments)]
    fn run_alpha_loop(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        delta: &IteratesVector,
        alpha_init: Number,
        alpha_dual: Number,
        nlp: Option<&Rc<RefCell<dyn IpoptNlp>>>,
        search_dir: Option<&mut PdSearchDirCalc>,
        theta: Number,
        phi: Number,
        d_phi: Number,
        skip_first: bool,
    ) -> AlphaResult {
        let curr = match data.borrow().curr.clone() {
            Some(c) => c,
            None => {
                return AlphaResult::Failed {
                    n_steps: 0,
                    last_alpha: 0.0,
                    evaluation_error: false,
                }
            }
        };

        let mut evaluation_error = false;

        let mut soc_search_dir = search_dir;
        let (mut c_soc_buf, mut dms_soc_buf) =
            if soc_search_dir.is_some() && nlp.is_some() && self.max_soc > 0 && !skip_first {
                let cq_ref = cq.borrow();
                let curr_c = cq_ref.curr_c();
                let curr_dms = cq_ref.curr_d_minus_s();
                let mut c_soc = curr_c.make_new();
                c_soc.copy(&*curr_c);
                let mut dms_soc = curr_dms.make_new();
                dms_soc.copy(&*curr_dms);
                (Some(c_soc), Some(dms_soc))
            } else {
                (None, None)
            };

        let mut alpha = if skip_first {
            alpha_init * self.alpha_red_factor
        } else {
            alpha_init
        };
        let mut last_alpha = alpha;
        let mut n_steps: i32 = 0;
        let acceptor_alpha_min = self.acceptor.calc_alpha_min(d_phi, theta);
        let alpha_min_eff = self.alpha_min.max(acceptor_alpha_min);

        for trial in 0..self.max_trials {
            if alpha < alpha_min_eff {
                return AlphaResult::TinyStep {
                    n_steps,
                    last_alpha,
                };
            }
            last_alpha = alpha;
            n_steps = trial;

            let trial_iv = scaled_step(&curr, delta, alpha, alpha_dual);
            data.borrow_mut().set_trial(trial_iv);

            let theta_trial = cq.borrow().trial_constraint_violation();
            let phi_trial = cq.borrow().trial_barrier_obj();
            if !theta_trial.is_finite() || !phi_trial.is_finite() {
                // Mirror upstream `IpBacktrackingLineSearch.cpp:776-784`:
                // a non-finite eval is treated as `Eval_Error`, sets the
                // `evaluation_error` flag, and the alpha-loop continues
                // to backtrack. Under watchdog, upstream breaks out
                // immediately (line 791-794) so the watchdog handler
                // can force StopWatchDog via line 493.
                evaluation_error = true;
                if self.in_watchdog {
                    return AlphaResult::Failed {
                        n_steps: trial,
                        last_alpha: alpha,
                        evaluation_error: true,
                    };
                }
                alpha *= self.alpha_red_factor;
                continue;
            }

            let decision =
                self.acceptor
                    .check_trial_point(alpha, theta, phi, d_phi, theta_trial, phi_trial);
            if decision == AcceptDecision::Accept {
                let mode = self
                    .acceptor
                    .update_for_next_iteration(alpha, theta, phi, d_phi, phi_trial);
                if std::env::var_os("POUNCE_DBG_LS").is_some() {
                    let d = data.borrow();
                    eprintln!(
                        "[PN_LS] iter={} mu={:.3e} alpha={:.3e} alpha_d={:.3e} mode={} theta={:.6e} theta_trial={:.6e} phi={:.6e} phi_trial={:.6e} n_steps={}",
                        d.iter_count, d.curr_mu, alpha, alpha_dual, mode, theta, theta_trial, phi, phi_trial, trial
                    );
                }
                let mut d = data.borrow_mut();
                d.info_alpha_primal = alpha;
                d.info_alpha_dual = alpha_dual;
                d.info_ls_count = trial + 1;
                d.info_alpha_primal_char = mode;
                return AlphaResult::Accepted { n_steps: trial };
            }

            // Watchdog: under upstream `IpBacktrackingLineSearch.cpp:791-794`,
            // a failed trial inside the watchdog window breaks out of the
            // alpha-loop immediately — alpha is NOT reduced. The trial just
            // attempted (at the full `alpha_init`) is left in `data.trial`
            // so `handle_watchdog_failure` can promote it via the 'w'
            // accept-anyway branch. Without this break, pounce kept
            // reducing alpha under watchdog and accepted the same tiny
            // step that triggered watchdog activation in the first place,
            // leaving the iterate stalled (observed on HATFLDFLNE: iter 11
            // accepted α=1.22e-4 'h' instead of α=1.00 'w').
            if self.in_watchdog {
                return AlphaResult::Failed {
                    n_steps: trial,
                    last_alpha: alpha,
                    evaluation_error,
                };
            }

            // SOC: only on the first non-skipped trial when constraint
            // violation grew. Disabled when `skip_first=true` (no SOC
            // buffers were allocated). Also disabled under watchdog (the
            // `in_watchdog` break above pre-empts SOC, matching upstream
            // which gates SOC after the in_watchdog break).
            if trial == 0
                && !skip_first
                && self.max_soc > 0
                && theta <= theta_trial
                && c_soc_buf.is_some()
                && dms_soc_buf.is_some()
            {
                let alpha_test = alpha;
                let mut count_soc: i32 = 0;
                let mut theta_soc_old: Number = 0.0;
                let mut theta_trial_local = theta_trial;
                let mut alpha_primal_soc = alpha;
                let mut soc_accepted = false;
                while count_soc < self.max_soc
                    && !soc_accepted
                    && (count_soc == 0 || theta_trial_local <= self.kappa_soc * theta_soc_old)
                {
                    theta_soc_old = theta_trial_local;
                    {
                        let cq_ref = cq.borrow();
                        let trial_c = cq_ref.trial_c();
                        let trial_dms = cq_ref.trial_d_minus_s();
                        if let Some(c_soc) = c_soc_buf.as_mut() {
                            c_soc.scal(alpha_primal_soc);
                            c_soc.axpy(1.0, &*trial_c);
                        }
                        if let Some(dms_soc) = dms_soc_buf.as_mut() {
                            dms_soc.scal(alpha_primal_soc);
                            dms_soc.axpy(1.0, &*trial_dms);
                        }
                    }
                    let delta_soc_opt = {
                        let sd = soc_search_dir
                            .as_deref_mut()
                            .expect("SOC: search_dir is gated above");
                        let nlp_ref = nlp.expect("SOC: nlp is gated above");
                        let c_soc = c_soc_buf.as_deref().expect("SOC: c_soc_buf is gated above");
                        let dms_soc = dms_soc_buf
                            .as_deref()
                            .expect("SOC: dms_soc_buf is gated above");
                        sd.compute_soc_step(
                            data,
                            cq,
                            nlp_ref,
                            c_soc,
                            dms_soc,
                            alpha_primal_soc,
                            self.soc_method,
                        )
                    };
                    let Some(delta_soc) = delta_soc_opt else {
                        break;
                    };
                    let tau = data.borrow().curr_tau;
                    alpha_primal_soc = cq.borrow().aff_step_alpha_primal_max(&delta_soc, tau);
                    let mut trial_iv = curr.deep_copy();
                    trial_iv.x.axpy(alpha_primal_soc, &*delta_soc.x);
                    trial_iv.s.axpy(alpha_primal_soc, &*delta_soc.s);
                    trial_iv.y_c.axpy(alpha, &*delta.y_c);
                    trial_iv.y_d.axpy(alpha, &*delta.y_d);
                    trial_iv.z_l.axpy(alpha_dual, &*delta.z_l);
                    trial_iv.z_u.axpy(alpha_dual, &*delta.z_u);
                    trial_iv.v_l.axpy(alpha_dual, &*delta.v_l);
                    trial_iv.v_u.axpy(alpha_dual, &*delta.v_u);
                    let trial_iv = trial_iv.freeze();
                    data.borrow_mut().set_trial(trial_iv);
                    let theta_soc = cq.borrow().trial_constraint_violation();
                    let phi_soc = cq.borrow().trial_barrier_obj();
                    if !theta_soc.is_finite() || !phi_soc.is_finite() {
                        break;
                    }
                    let dec = self
                        .acceptor
                        .check_trial_point(alpha_test, theta, phi, d_phi, theta_soc, phi_soc);
                    if dec == AcceptDecision::Accept {
                        let mode = self
                            .acceptor
                            .update_for_next_iteration(alpha_test, theta, phi, d_phi, phi_soc);
                        let mut d = data.borrow_mut();
                        d.info_alpha_primal = alpha_primal_soc;
                        d.info_alpha_dual = alpha_dual;
                        d.info_ls_count = trial + 1;
                        d.info_alpha_primal_char = mode.to_ascii_uppercase();
                        return AlphaResult::Accepted { n_steps: trial };
                    }
                    count_soc += 1;
                    theta_trial_local = theta_soc;
                    soc_accepted = false;
                }
            }

            alpha *= self.alpha_red_factor;
        }

        AlphaResult::Failed {
            n_steps,
            last_alpha,
            evaluation_error,
        }
    }

    /// Directional derivative of the barrier objective along the step
    /// `delta`: `d_phi = ∇_x φ · dx + ∇_s φ · ds`.
    fn compute_d_phi(&self, cq: &IpoptCqHandle, delta: &IteratesVector) -> Number {
        let cq_ref = cq.borrow();
        let g_x = cq_ref.curr_grad_barrier_obj_x();
        let g_s = cq_ref.curr_grad_barrier_obj_s();
        g_x.dot(&*delta.x) + g_s.dot(&*delta.s)
    }
}

/// `out = curr + alpha * delta` for all eight components, returned as a
/// fresh `IteratesVector` with `Rc<dyn Vector>` slots. Mirrors
/// `IpoptData::SetTrialBoundMultipliersFromStep` + the primal step
/// path in upstream — both share the same scalar α here because
/// fraction-to-the-boundary truncation has already been folded into
/// `alpha_init` upstream.
fn scaled_step(
    curr: &IteratesVector,
    delta: &IteratesVector,
    alpha_primal: Number,
    alpha_dual: Number,
) -> IteratesVector {
    let mut out = curr.make_new_zeroed();
    out.add_one_vector(1.0, curr, 0.0); // out = curr
    out.x.axpy(alpha_primal, &*delta.x);
    out.s.axpy(alpha_primal, &*delta.s);
    out.y_c.axpy(alpha_primal, &*delta.y_c);
    out.y_d.axpy(alpha_primal, &*delta.y_d);
    out.z_l.axpy(alpha_dual, &*delta.z_l);
    out.z_u.axpy(alpha_dual, &*delta.z_u);
    out.v_l.axpy(alpha_dual, &*delta.v_l);
    out.v_u.axpy(alpha_dual, &*delta.v_u);
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterates_vector::IteratesVector;
    use crate::line_search::filter_acceptor::FilterLsAcceptor;
    use pounce_linalg::dense_vector::DenseVectorSpace;
    use pounce_linalg::Vector;
    use std::rc::Rc;

    fn dense(n: i32, vals: &[Number]) -> Rc<dyn Vector> {
        let mut v = DenseVectorSpace::new(n).make_new_dense();
        v.set(0.0);
        if !vals.is_empty() {
            v.values_mut().copy_from_slice(vals);
        }
        Rc::new(v)
    }

    fn iv_from(x: &[Number], s: &[Number]) -> IteratesVector {
        IteratesVector::new(
            dense(x.len() as i32, x),
            dense(s.len() as i32, s),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
            dense(0, &[]),
        )
    }

    #[test]
    fn driver_constructs_with_defaults() {
        let bls = BacktrackingLineSearch::new(Box::new(FilterLsAcceptor::new()));
        assert_eq!(bls.alpha_red_factor, 0.5);
        assert_eq!(bls.max_soc, 4);
    }

    #[test]
    fn scaled_step_writes_curr_plus_alpha_delta() {
        // curr.x = (0,0), delta.x = (1,1) → at alpha=0.5, trial.x = (0.5, 0.5).
        let curr = iv_from(&[0.0, 0.0], &[0.0]);
        let delta = iv_from(&[1.0, 1.0], &[2.0]);
        let trial = scaled_step(&curr, &delta, 0.5, 0.5);
        let xv = trial
            .x
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(xv, vec![0.5, 0.5]);
        let sv = trial
            .s
            .as_any()
            .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(sv, vec![1.0]); // 0.0 + 0.5 * 2.0
    }

    #[test]
    fn outcome_variants_are_distinct() {
        assert_ne!(Outcome::Accepted, Outcome::Failed);
        assert_ne!(Outcome::Accepted, Outcome::TinyStep);
        assert_ne!(Outcome::Failed, Outcome::TinyStep);
    }

    #[test]
    fn watchdog_state_starts_inactive() {
        // Mirror upstream `IpBacktrackingLineSearch::InitializeImpl`
        // (`IpBacktrackingLineSearch.cpp:240-249`): the watchdog is
        // inactive at construction and `last_mu_` is initialised to
        // a sentinel `-1` so the first iteration's mu always
        // triggers the reset branch (which is harmless when the
        // watchdog was never armed).
        let bls = BacktrackingLineSearch::new(Box::new(FilterLsAcceptor::new()));
        assert!(!bls.in_watchdog());
        assert_eq!(bls.watchdog_shortened_iter(), 0);
        assert!(bls.last_mu < 0.0);
        assert_eq!(bls.watchdog_shortened_iter_trigger, 10);
        assert_eq!(bls.watchdog_trial_iter_max, 3);
    }

    #[test]
    fn alpha_result_failed_carries_n_steps_and_last_alpha() {
        // Sanity check on the internal AlphaResult enum: the watchdog
        // wrapper relies on `Failed { n_steps, last_alpha }` to stamp
        // the info-* fields when handing off to restoration.
        let r = AlphaResult::Failed {
            n_steps: 7,
            last_alpha: 1e-6,
            evaluation_error: false,
        };
        match r {
            AlphaResult::Failed {
                n_steps,
                last_alpha,
                evaluation_error,
            } => {
                assert_eq!(n_steps, 7);
                assert!((last_alpha - 1e-6).abs() < 1e-20);
                assert!(!evaluation_error);
            }
            _ => unreachable!(),
        }
    }
}
