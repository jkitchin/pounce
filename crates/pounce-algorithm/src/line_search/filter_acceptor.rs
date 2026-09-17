//! Filter line-search acceptor — port of
//! `Algorithm/IpFilterLSAcceptor.{hpp,cpp}`.
//!
//! Combines an [`super::filter::Filter`] with Fletcher-Leyffer's
//! filter logic: at each backtracking step it asks whether the
//! `(theta_trial, phi_trial)` pair is acceptable to the filter and
//! to a sufficient-decrease test. The decisions are:
//!
//! * **Switching condition**: `d_phi < 0 ∧
//!   alpha * (-d_phi)^s_phi > delta * theta^s_theta` — when true, we
//!   require an Armijo decrease in `phi`; otherwise we relax to the
//!   filter test.
//! * **Armijo decrease**:
//!   `phi_trial - phi <= eta_phi * alpha * d_phi`, compared via
//!   `Compare_le` (a round-off-tolerant `<=`) — see [`FilterLsAcceptor::armijo_holds`].
//! * **Filter acceptance**: `phi_trial < phi - gamma_phi * theta` OR
//!   `theta_trial < (1 - gamma_theta) * theta`.
//!
//! See `ref/Ipopt/AGENT_REFERENCE/LINE_SEARCH.md` §"Acceptance" for
//! the full statement; constants below default to upstream's
//! `RegisterOptions` values.

use crate::line_search::filter::Filter;
use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
use pounce_common::types::Number;

/// `POUNCE_DBG_LS=1` toggle, cached so per-trial overhead is one
/// atomic load instead of a syscall. Used by [`FilterLsAcceptor::check_acceptability`]
/// to emit per-trial `(α, θ, φ, d_phi, θ_trial, φ_trial, rapid_inc_ok,
/// suff_progress_ok)` for the pounce#21 W-B parity investigation.
fn dbg_ls_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("POUNCE_DBG_LS").as_deref() == Ok("1"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptDecision {
    /// `(theta_trial, phi_trial)` is acceptable to the filter and
    /// passes the decrease test for the current `alpha`.
    Accept,
    /// Trial point fails the filter or decrease check.
    Reject,
}

pub struct FilterLsAcceptor {
    pub filter: Filter,
    pub eta_phi: Number,
    pub delta_armijo: Number,
    pub theta_min_fact: Number,
    pub theta_max_fact: Number,
    pub gamma_phi: Number,
    pub gamma_theta: Number,
    pub s_phi: Number,
    pub s_theta: Number,
    /// `obj_max_inc` option (default 5.0) — the rapid-barrier-increase
    /// guard's log-scale cap. Both the live `check_acceptability` path and
    /// the [`Self::is_acceptable_to_current_iterate`] helper read this, so
    /// the regular-phase line search and the restoration progress test stay
    /// in lockstep instead of one path hard-coding 5.0.
    pub obj_max_inc: Number,
    pub max_soc: i32,
    /// `alpha_min_frac` from `IpFilterLSAcceptor.cpp:RegisterOptions` —
    /// safety factor applied to the dynamic alpha-min before declaring
    /// a tiny step / handing off to restoration. Default 0.05.
    pub alpha_min_frac: Number,
    /// Lazily initialised `theta_min_` (upstream
    /// `IpFilterLSAcceptor.cpp:333-339`). `None` until the first call to
    /// [`Self::calc_alpha_min`] sees a reference theta.
    theta_min: Option<Number>,
    /// Lazily initialised `theta_max_` (upstream
    /// `IpFilterLSAcceptor.cpp:325-331`). Locked on first encounter to
    /// `theta_max_fact * max(1, reference_theta)`. Any trial iterate
    /// with `theta_trial > theta_max` is rejected outright by the
    /// filter — this guards against the line search accepting a step
    /// that catastrophically inflates constraint violation (e.g. a
    /// Newton step from a poorly-scaled iterate landing far outside
    /// the feasible basin).
    theta_max: Option<Number>,
    /// `theta_max_row_scale_kappa` — multiplier on the row count used as
    /// the floor of the `theta_max` reference. **Opt-in: default `0`,
    /// which is upstream's bare `1.0` floor bit-for-bit.**
    ///
    /// Upstream locks `theta_max = theta_max_fact * max(1, theta_0)`.
    /// That `1` is dimensionally wrong for POUNCE, because `theta` is a
    /// **1-norm over constraint rows** (`ipopt_cq.rs`
    /// `curr_constraint_violation` = `c.asum() + dms.asum()`), so it
    /// grows with `m` while the floor does not. On a model with a
    /// feasible starting point (`theta_0 = 0`, so the `max` collapses to
    /// the floor) and large `m`, the ceiling becomes the bare constant
    /// `1e4` no matter how many rows are being summed — on `robot_a`
    /// (`m = 52013`) that is a mean per-row residual of just `0.19`, and
    /// the route to the optimum passes through `theta ~ 9.4e7`, so the
    /// solve can never get there at any iteration budget.
    ///
    /// The fix is to floor the reference at `kappa * rows` instead of at
    /// `1`, which keeps the ceiling's meaning ("a mean per-row violation
    /// of `theta_max_fact`") fixed as `m` varies. Upstream needs the
    /// same correction and does not have it; it papers over the one case
    /// it hit by hard-coding `resto.theta_max_fact = 1e8` for the
    /// restoration sub-IPM (`IpRestoMinC_1Nrm.cpp:91`), whose slack init
    /// produces exactly this `theta_0 ~ 0` degeneracy.
    ///
    /// **Why this is not on by default.** Raising the ceiling is not
    /// free: it relaxes a global-convergence safeguard, and a model that
    /// was not being blocked by it can wander instead. Measured on the
    /// Vanderbei corpus, `brainpc1/3/5/7` (`m = 6900`, `theta_0 = 1e-2`)
    /// regress — `brainpc1` from `Optimal` in 64 iterations to divergent
    /// (objective `3.7e3` against `4.4e-04`). A kappa scan showed the
    /// damage is a **step function, not a gradient**: `brainpc3` and
    /// `brainpc7` land on the identical worse answer at every kappa in
    /// `{0.01, 0.05, 0.2, 1.0}`, while `robot_a` improves monotonically
    /// (287 → 153 → 127 → 112 iterations). So no single multiplier
    /// separates the two families, and a *static* floor cannot: the real
    /// question is whether a model's route to the optimum needs the
    /// headroom, which the row count does not answer. Until an adaptive
    /// rule exists (pounce#476), this stays a documented opt-in for
    /// large models that stall from a feasible start.
    pub theta_max_row_scale_kappa: Number,
    /// Number of constraint rows backing `theta`'s 1-norm, i.e. `dim(c)
    /// + dim(d - s)`. Set once per solve by the line search from the
    /// calculated quantities; `1.0` until then, which reproduces the
    /// upstream floor.
    theta_rows: Number,
    /// `theta_max_adaptive_trigger` — number of *consecutive* line
    /// searches in which **every** trial was rejected at the Eqn.-21
    /// `theta_max` gate before the ceiling is raised. `0` disables the
    /// rule, leaving `theta_max` locked exactly as upstream locks it.
    ///
    /// This is the adaptive answer to the problem
    /// [`Self::theta_max_row_scale_kappa`] documents and cannot solve
    /// (pounce#476, #546). The static floor asks "does this model have
    /// many rows?", which is a *proxy* for needing headroom, and the
    /// kappa scan showed it is the wrong proxy: `robot_a` needs the
    /// headroom and `brainpc1/3/5/7` are damaged by it, at every kappa,
    /// with no separating value. The question that actually decides it
    /// is whether the model's route to the optimum passes *through* a
    /// `theta` above the ceiling — which is directly observable. A trial
    /// rejected because `theta_trial > theta_max` takes a distinct early
    /// return in [`Self::check_acceptability`], so counting those and
    /// comparing against the number of trials attempted says whether the
    /// gate, rather than the filter or the Armijo test, is what refused
    /// the line search.
    ///
    /// The rule only fires when the gate refused **all** of a line
    /// search's trials, for this many consecutive line searches. That
    /// makes the failure mode it targets — every step toward the
    /// solution refused at the gate, backtracking to `alpha_min`, hand
    /// off to restoration, repeat — the only thing it responds to.
    /// `brainpc` never trips it, because `brainpc` is not being refused
    /// at the gate; it converges at the upstream ceiling and is only
    /// hurt when the ceiling is loosened. That model family is therefore
    /// bit-for-bit upstream **by construction** rather than by choosing
    /// a lucky constant, which is the property the static floor could
    /// not have.
    ///
    /// Requiring a *streak* rather than a single line search is
    /// deliberate: one Newton direction that overshoots into a huge
    /// `theta` can legitimately have all of its trials refused, and
    /// backtracking is the correct response to that. Only a model that
    /// cannot get past the gate repeatedly is one whose route needs the
    /// headroom.
    pub theta_max_adaptive_trigger: u32,
    /// Factor by which `theta_max` is multiplied each time the adaptive
    /// rule fires. The ladder is geometric so that a model needing many
    /// orders of headroom reaches it in a few raises without any single
    /// raise being large enough to discard the safeguard outright.
    pub theta_max_adaptive_factor: Number,
    /// Hard cap on how many times `theta_max` may be raised in one
    /// solve. Wächter–Biegler's global-convergence argument (Thm. 2)
    /// needs `theta_max` to be *finite*, not fixed; a bounded number of
    /// bounded increases keeps it finite, so the ceiling still exists
    /// and the solve cannot ratchet it away one line search at a time.
    pub theta_max_adaptive_max_raises: u32,
    /// Trials seen in the line search currently in progress.
    trials_this_ls: u32,
    /// How many of [`Self::trials_this_ls`] were refused at the
    /// `theta_max` gate specifically.
    theta_max_gate_rejections_this_ls: u32,
    /// Consecutive completed line searches in which every trial was
    /// refused at the gate. Reset by any line search that was not
    /// entirely gate-refused.
    gate_blocked_ls_streak: u32,
    /// Raises used so far this solve, against
    /// [`Self::theta_max_adaptive_max_raises`].
    n_theta_max_raises: u32,
    /// Maximum number of filter resets allowed per solve (upstream
    /// option `max_filter_resets`, default 5). Set to `0` to disable
    /// the heuristic entirely.
    pub max_filter_resets: i32,
    /// Number of consecutive filter-rejected accepts that triggers a
    /// reset (upstream option `filter_reset_trigger`, default 5).
    pub filter_reset_trigger: i32,
    /// Resets used so far this solve. Bumped each time the heuristic
    /// fires; not cleared by [`Self::reset`] — only re-initialising
    /// the acceptor (a fresh `Default::default()`) zeroes it. Mirrors
    /// upstream `n_filter_resets_`, which is cleared only in
    /// `InitializeImpl` (per-solve), never in `Reset()`.
    n_filter_resets: i32,
    /// `true` when the most recent trial was rejected because of the
    /// filter (as opposed to the iterate-acceptability test). Mirrors
    /// upstream `last_rejection_due_to_filter_`. Read on the accept
    /// path to decide whether to bump `count_successive_filter_rejections`.
    last_rejection_due_to_filter: bool,
    /// Number of consecutive accepted trials whose immediately-preceding
    /// rejection was due to the filter. Reset whenever an accept
    /// follows a non-filter rejection (or the very first accept of the
    /// solve). Mirrors `count_successive_filter_rejections_`.
    count_successive_filter_rejections: i32,
}

impl Default for FilterLsAcceptor {
    fn default() -> Self {
        // Defaults from `IpFilterLSAcceptor.cpp:RegisterOptions`.
        Self {
            filter: Filter::new(),
            eta_phi: 1e-8,
            delta_armijo: 1.0,
            theta_min_fact: 1e-4,
            theta_max_fact: 1e4,
            gamma_phi: 1e-8,
            gamma_theta: 1e-5,
            s_phi: 2.3,
            s_theta: 1.1,
            obj_max_inc: 5.0,
            max_soc: 4,
            alpha_min_frac: 0.05,
            theta_min: None,
            theta_max: None,
            theta_max_row_scale_kappa: 0.0,
            theta_rows: 1.0,
            theta_max_adaptive_trigger: 3,
            theta_max_adaptive_factor: 100.0,
            theta_max_adaptive_max_raises: 4,
            trials_this_ls: 0,
            theta_max_gate_rejections_this_ls: 0,
            gate_blocked_ls_streak: 0,
            n_theta_max_raises: 0,
            max_filter_resets: 5,
            filter_reset_trigger: 5,
            n_filter_resets: 0,
            last_rejection_due_to_filter: false,
            count_successive_filter_rejections: 0,
        }
    }
}

impl FilterLsAcceptor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Switching condition — true ⇒ require Armijo decrease in phi
    /// rather than the filter check. Mirrors
    /// `IpFilterLSAcceptor.cpp:IsSwitchingCondition` (line ~590).
    pub fn is_switching_condition(
        &self,
        alpha_primal: Number,
        d_phi: Number,
        theta: Number,
    ) -> bool {
        if d_phi >= 0.0 {
            return false;
        }
        let lhs = alpha_primal * (-d_phi).powf(self.s_phi);
        let rhs = self.delta_armijo * theta.powf(self.s_theta);
        lhs > rhs
    }

    /// Armijo sufficient decrease in `phi`.
    /// Mirrors `IpFilterLSAcceptor.cpp:ArmijoHolds`, which compares with
    /// `Compare_le` — a `<=` carrying a `10·eps·|phi|` round-off slack —
    /// not a bare `<=`. The slack is essential near a solution: when the
    /// barrier objective is flat, `phi_trial - phi` is dominated by
    /// floating-point summation noise (a tiny *positive* value even on a
    /// genuine descent step), while `eta_phi·alpha·d_phi` is a tiny
    /// *negative* number. A bare `<=` can then never hold, so the line
    /// search backtracks to `alpha_min` and falls into restoration —
    /// mislabelling a converged iterate `Error_In_Step_Computation`
    /// (PALMER1/2, VESUVIALS, HIELOW, MGH10LS, ... — all reach the
    /// optimum, then stall here).
    pub fn armijo_holds(
        &self,
        alpha_primal: Number,
        d_phi: Number,
        phi: Number,
        phi_trial: Number,
    ) -> bool {
        pounce_common::utils::compare_le(phi_trial - phi, self.eta_phi * alpha_primal * d_phi, phi)
    }

    /// Sufficient progress check (used when *not* in switching mode).
    /// Mirrors the OR-test in `IpFilterLSAcceptor.cpp:IsAcceptableToCurrentIterate`.
    ///
    /// The two comparisons use `compare_le` — a `<=` carrying a
    /// `10·eps·max(1, |basval|)` round-off slack (the floor is a
    /// deliberate deviation from upstream; see `compare_le`'s doc for
    /// the Mittelmann evidence) — exactly like the live
    /// [`Self::check_acceptability`] path and
    /// [`Self::is_acceptable_to_current_iterate`]. An earlier version used
    /// bare `<` here, so this (then-dead) helper silently disagreed with the
    /// live path on the round-off boundary: near a solution `phi_trial - phi`
    /// is dominated by summation noise (a tiny *positive* value on a genuine
    /// descent step) while `-gamma_phi·theta` is a tiny *negative* one, and a
    /// bare `<` rejects the step that `compare_le` accepts (the same
    /// flat-objective failure mode documented on [`Self::armijo_holds`]).
    /// This is now the single source of truth for the OR-test (L6).
    pub fn is_sufficient_progress(
        &self,
        theta: Number,
        phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> bool {
        pounce_common::utils::compare_le(theta_trial, (1.0 - self.gamma_theta) * theta, theta)
            || pounce_common::utils::compare_le(phi_trial - phi, -self.gamma_phi * theta, phi)
    }

    /// Mirrors `FilterLSAcceptor::IsAcceptableToCurrentFilter`
    /// (`IpFilterLSAcceptor.cpp:501-504`). True iff `(theta, barr)` is
    /// not dominated by any entry in the filter.
    pub fn is_acceptable_to_current_filter(&self, trial_barr: Number, trial_theta: Number) -> bool {
        !self.filter.dominated_by_any(trial_theta, trial_barr)
    }

    /// Mirrors `FilterLSAcceptor::IsAcceptableToCurrentIterate`
    /// (`IpFilterLSAcceptor.cpp:471-499`).
    ///
    /// `reference_*` is the iterate-pair the line search is comparing
    /// against (typically the current iterate's `(barr, theta)` at the
    /// start of the line search). `obj_max_inc` is the upstream
    /// `obj_max_inc` option (default 5.0). `called_from_restoration`
    /// disables the rapid-barrier-increase guard — used by
    /// [`crate::line_search::filter_acceptor`] consumers in the
    /// restoration phase, since the resto sub-solver's barrier value
    /// has no direct comparability to the outer one.
    pub fn is_acceptable_to_current_iterate(
        &self,
        trial_barr: Number,
        trial_theta: Number,
        reference_barr: Number,
        reference_theta: Number,
        obj_max_inc: Number,
        called_from_restoration: bool,
    ) -> bool {
        if !called_from_restoration && trial_barr > reference_barr {
            // Rapid-increase guard: log-scale jump cap.
            let basval = if reference_barr.abs() > 10.0 {
                reference_barr.abs().log10()
            } else {
                1.0
            };
            if (trial_barr - reference_barr).log10() > obj_max_inc + basval {
                return false;
            }
        }
        // Filter-style sufficient-progress test (line 497-498) — delegated to
        // the canonical [`Self::is_sufficient_progress`] so this helper and
        // the live path share one implementation (L6).
        self.is_sufficient_progress(reference_theta, reference_barr, trial_theta, trial_barr)
    }

    /// Single-trial accept decision. Caller has already computed the
    /// trial `(theta, phi)` pair and the directional derivative
    /// `d_phi`. Mirrors the body of
    /// `IpFilterLSAcceptor::CheckAcceptabilityOfTrialPoint` (lines
    /// 311-437): the iterate test runs first (Armijo when F-type AND
    /// `theta <= theta_min`, otherwise `IsAcceptableToCurrentIterate`
    /// with the `obj_max_inc` rapid-increase guard); only on iterate
    /// acceptance do we then consult the filter.
    ///
    /// The `&mut self` receiver lets the method record the rejection
    /// reason in `last_rejection_due_to_filter` and run the filter-reset
    /// heuristic on the accept path
    /// (`IpFilterLSAcceptor.cpp:407-433`).
    pub fn check_acceptability(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> AcceptDecision {
        // theta_min / theta_max may not yet have been initialised if
        // the caller skipped `calc_alpha_min` for some reason; fall
        // back to the same lazy formula upstream uses on first
        // encounter (`IpFilterLSAcceptor.cpp:325-339`).
        let theta_min = self
            .theta_min
            .unwrap_or_else(|| self.theta_min_fact * theta.max(1.0));
        let theta_max = self
            .theta_max
            .unwrap_or_else(|| self.theta_max_fact * theta.max(self.theta_max_reference_floor()));

        // `IpFilterLSAcceptor.cpp:341-348`: any trial iterate above
        // `theta_max` is rejected outright. Without this guard the
        // line search may accept a step that inflates constraint
        // violation by many orders of magnitude (POLAK6, ROSENMMX,
        // ACOPR14: theta jumps from 8 to 1e12 on iter 1).
        // Adaptive-ceiling bookkeeping (pounce#546). Counted here, at
        // the top, so that `trials_this_ls` is the number of trials the
        // gate had the opportunity to refuse — the denominator the
        // "every trial was refused at the gate" predicate needs.
        self.trials_this_ls = self.trials_this_ls.saturating_add(1);

        if theta_trial > theta_max {
            self.theta_max_gate_rejections_this_ls =
                self.theta_max_gate_rejections_this_ls.saturating_add(1);
            self.last_rejection_due_to_filter = false;
            return AcceptDecision::Reject;
        }

        let f_type = alpha_primal > 0.0 && self.is_switching_condition(alpha_primal, d_phi, theta);
        let take_armijo = f_type && theta <= theta_min;

        let iterate_ok = if take_armijo {
            self.armijo_holds(alpha_primal, d_phi, phi, phi_trial)
        } else {
            // `IsAcceptableToCurrentIterate` with `called_from_restoration=false`.
            // Rapid-barrier-increase guard, capped by the `obj_max_inc` field
            // (default 5.0) — the same value the restoration progress test
            // reads through `is_acceptable_to_current_iterate`, so the two
            // paths no longer diverge on a hard-coded constant (L6).
            let rapid_increase_ok = if phi_trial > phi {
                let basval = if phi.abs() > 10.0 {
                    phi.abs().log10()
                } else {
                    1.0
                };
                (phi_trial - phi).log10() <= self.obj_max_inc + basval
            } else {
                true
            };
            // Single source of truth for the sufficient-progress OR-test.
            let suff_progress_ok = self.is_sufficient_progress(theta, phi, theta_trial, phi_trial);
            // pounce#21 diagnostic — env-gated. Emits one line per
            // trial when POUNCE_DBG_LS=1 so the divergence-vs-Ipopt
            // investigation can correlate which branch (rapid-increase
            // guard vs sufficient-progress) was the rejection cause.
            // Env lookup cached in a `OnceLock` so the disabled case
            // costs one atomic load per trial.
            if dbg_ls_enabled() {
                tracing::debug!(target: "pounce::linesearch",
                    "DBG_LS alpha={:.3e} theta={:.3e} phi={:.3e} d_phi={:.3e} theta_trial={:.3e} phi_trial={:.3e} theta_max={:.3e} rapid_inc_ok={} suff_progress_ok={}",
                    alpha_primal,
                    theta,
                    phi,
                    d_phi,
                    theta_trial,
                    phi_trial,
                    theta_max,
                    rapid_increase_ok,
                    suff_progress_ok,
                );
            }
            rapid_increase_ok && suff_progress_ok
        };

        if !iterate_ok {
            // Iterate-acceptability rejection (the LS-test branch in
            // upstream's `CheckAcceptabilityOfTrialPoint`, lines 363-381).
            // Upstream sets `last_rejection_due_to_filter_ = false` on
            // both the unfortunate-Armijo-failure and the
            // sufficient-progress-failure paths.
            self.last_rejection_due_to_filter = false;
            return AcceptDecision::Reject;
        }

        if self.filter.dominated_by_any(theta_trial, phi_trial) {
            // Iterate test passed but filter dominates → mark this
            // rejection as filter-due (line 397).
            self.last_rejection_due_to_filter = true;
            return AcceptDecision::Reject;
        }

        // Trial accepted. Run the filter-reset heuristic
        // (`IpFilterLSAcceptor.cpp:407-433`).
        if self.max_filter_resets > 0 && self.n_filter_resets < self.max_filter_resets {
            if self.last_rejection_due_to_filter {
                self.count_successive_filter_rejections += 1;
                if self.count_successive_filter_rejections >= self.filter_reset_trigger {
                    self.filter.clear();
                    self.n_filter_resets += 1;
                    self.count_successive_filter_rejections = 0;
                }
            } else {
                self.count_successive_filter_rejections = 0;
            }
        }
        // Clear for the next outer iteration's α-loop (line 434).
        self.last_rejection_due_to_filter = false;

        AcceptDecision::Accept
    }
}

impl FilterLsAcceptor {
    /// Lazy initialiser for `theta_min` matching upstream
    /// `IpFilterLSAcceptor.cpp:333-339`: on first invocation (and never
    /// again — upstream resets only `theta_min_ = -1` in
    /// `InitializeImpl`, not in `Reset()`), set
    /// `theta_min = theta_min_fact * max(1, reference_theta)`.
    fn ensure_theta_min(&mut self, reference_theta: Number) -> Number {
        *self
            .theta_min
            .get_or_insert_with(|| self.theta_min_fact * reference_theta.max(1.0))
    }

    /// Floor for the `theta_max` reference. Upstream
    /// (`IpFilterLSAcceptor.cpp:325-331`) uses the bare constant `1.0`;
    /// we use `max(1, kappa * rows)` so the ceiling keeps its meaning as
    /// `m` varies — see [`Self::theta_max_row_scale_kappa`]. With
    /// `kappa = 0`, or before the row count is known, this is exactly
    /// upstream's `1.0`.
    fn theta_max_reference_floor(&self) -> Number {
        if self.theta_max_row_scale_kappa <= 0.0 {
            return 1.0;
        }
        (self.theta_max_row_scale_kappa * self.theta_rows).max(1.0)
    }

    /// Lazy initialiser for `theta_max_` matching upstream
    /// `IpFilterLSAcceptor.cpp:325-331`, with the row-scaled floor:
    /// locked once on first encounter to
    /// `theta_max_fact * max(floor, reference_theta)`.
    fn ensure_theta_max(&mut self, reference_theta: Number) -> Number {
        let floor = self.theta_max_reference_floor();
        let fact = self.theta_max_fact;
        *self
            .theta_max
            .get_or_insert_with(|| fact * reference_theta.max(floor))
    }

    /// Close the books on the line search that just finished and, if it
    /// completes a long enough streak of gate-blocked line searches,
    /// raise `theta_max` (pounce#546).
    ///
    /// Called from [`BacktrackingLsAcceptor::init_this_line_search`],
    /// which the driver invokes once per outer iteration *before* the
    /// α-loop — so on entry the counters describe the previous line
    /// search, complete. A call with no trials recorded is a no-op,
    /// which is what makes it safe against the driver reaching this
    /// hook twice without an intervening α-loop.
    fn end_of_line_search(&mut self) {
        if self.trials_this_ls == 0 {
            return;
        }
        // "Every trial refused at the gate" — not "some", because a line
        // search that got past the gate and was then refused by the
        // filter or the Armijo test is being refused by the safeguard
        // that is supposed to refuse it.
        let gate_blocked = self.theta_max_gate_rejections_this_ls == self.trials_this_ls;
        self.trials_this_ls = 0;
        self.theta_max_gate_rejections_this_ls = 0;

        if !gate_blocked {
            self.gate_blocked_ls_streak = 0;
            return;
        }
        self.gate_blocked_ls_streak = self.gate_blocked_ls_streak.saturating_add(1);

        if self.theta_max_adaptive_trigger == 0
            || self.gate_blocked_ls_streak < self.theta_max_adaptive_trigger
            || self.n_theta_max_raises >= self.theta_max_adaptive_max_raises
            || self.theta_max_adaptive_factor <= 1.0
        {
            return;
        }
        // Nothing to raise until the ceiling has actually locked.
        let Some(old) = self.theta_max else {
            return;
        };
        let new = old * self.theta_max_adaptive_factor;
        if !new.is_finite() {
            return;
        }
        self.theta_max = Some(new);
        self.n_theta_max_raises += 1;
        self.gate_blocked_ls_streak = 0;
        tracing::info!(target: "pounce::linesearch",
            "theta_max raised {old:.3e} -> {new:.3e} (raise {} of {}): {} consecutive line searches \
             had every trial rejected at the theta_max gate (pounce#546)",
            self.n_theta_max_raises,
            self.theta_max_adaptive_max_raises,
            self.theta_max_adaptive_trigger,
        );
    }
}

impl BacktrackingLsAcceptor for FilterLsAcceptor {
    fn reset(&mut self) {
        // Mirrors upstream `FilterLSAcceptor::Reset`
        // (`IpFilterLSAcceptor.cpp:524-532`): clears the filter and the
        // per-LS rejection-tracking state, but **does not** clear
        // `n_filter_resets` (that's only cleared in `InitializeImpl`,
        // i.e. via constructing a fresh acceptor).
        self.filter.clear();
        self.last_rejection_due_to_filter = false;
        self.count_successive_filter_rejections = 0;
    }

    /// Upstream's filter acceptor has nothing to cache per line search.
    /// pounce uses the hook as the line-search *boundary*: it is called
    /// once per outer iteration before the α-loop, so on entry the trial
    /// counters describe the line search that just finished, complete.
    /// That is what [`Self::end_of_line_search`] needs to decide whether
    /// the `theta_max` gate — rather than the filter — is what refused
    /// it (pounce#546).
    fn init_this_line_search(
        &mut self,
        _data: &crate::ipopt_data::IpoptDataHandle,
        _cq: &crate::ipopt_cq::IpoptCqHandle,
        _delta: &crate::iterates_vector::IteratesVector,
    ) {
        self.end_of_line_search();
    }

    /// Forward the gh#945 round-off floor to the filter. The driver sets
    /// it for the duration of one retry pass and clears it again; see
    /// [`super::filter::Filter::dominated_by_any`].
    fn set_theta_roundoff_floor(&mut self, floor: Number) {
        self.filter.set_theta_roundoff_floor(floor);
    }

    /// Port of `IpFilterLSAcceptor.cpp:CalculateAlphaMin` (lines
    /// 450-469). Returns `alpha_min_frac * alpha_min` where
    /// `alpha_min` is `gamma_theta` by default, tightened to
    /// `gamma_phi * theta / (-d_phi)` when `d_phi < 0`, and further
    /// tightened to `delta * theta^s_theta / (-d_phi)^s_phi` when
    /// `theta <= theta_min`.
    fn calc_alpha_min(&mut self, d_phi: Number, theta: Number) -> Number {
        let theta_min = self.ensure_theta_min(theta);
        let _ = self.ensure_theta_max(theta);
        let mut alpha_min = self.gamma_theta;
        if d_phi < 0.0 {
            alpha_min = alpha_min.min(self.gamma_phi * theta / (-d_phi));
            if theta <= theta_min {
                alpha_min = alpha_min
                    .min(self.delta_armijo * theta.powf(self.s_theta) / (-d_phi).powf(self.s_phi));
            }
        }
        self.alpha_min_frac * alpha_min
    }

    fn check_trial_point(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        theta_trial: Number,
        phi_trial: Number,
    ) -> AcceptDecision {
        self.check_acceptability(alpha_primal, theta, phi, d_phi, theta_trial, phi_trial)
    }

    /// Port of `IpFilterLSAcceptor::UpdateForNextIteration` (lines
    /// 881-895). Returns `'f'` when both the switching condition fires
    /// AND Armijo holds (in which case the filter is **not** augmented);
    /// otherwise returns `'h'` and augments the filter with the
    /// pre-shrunken envelope `(theta_add, phi_add) = ((1 - γ_θ)·θ_ref,
    /// φ_ref - γ_φ·θ_ref)`.
    fn update_for_next_iteration(
        &mut self,
        alpha_primal: Number,
        theta: Number,
        phi: Number,
        d_phi: Number,
        phi_trial: Number,
    ) -> char {
        let is_ftype = self.is_switching_condition(alpha_primal, d_phi, theta);
        let armijo = self.armijo_holds(alpha_primal, d_phi, phi, phi_trial);
        if !is_ftype || !armijo {
            let phi_add = phi - self.gamma_phi * theta;
            let theta_add = (1.0 - self.gamma_theta) * theta;
            self.filter.add(theta_add, phi_add, 0);
            'h'
        } else {
            'f'
        }
    }

    /// Build the orig-progress callback for the inner restoration IPM.
    /// Clones the current filter and the iterate-acceptance constants
    /// into the closure so it can be evaluated repeatedly without
    /// holding a borrow on the acceptor.
    fn make_orig_progress_check(
        &self,
        reference_theta: Number,
        reference_barr: Number,
        // `called_from_restoration=true` in the closure below disables
        // the rapid-barrier-increase guard, which is the only place
        // upstream consumes `obj_max_inc`. Param kept on the trait
        // surface for parity with upstream's signature.
        _obj_max_inc: Number,
    ) -> Option<crate::restoration::OrigProgressCallback> {
        let filter_snapshot = self.filter.clone();
        let gamma_theta = self.gamma_theta;
        let gamma_phi = self.gamma_phi;
        Some(Box::new(move |trial_barr: Number, trial_theta: Number| {
            // 1. Filter acceptance — `IsAcceptableToCurrentFilter`.
            if filter_snapshot.dominated_by_any(trial_theta, trial_barr) {
                return false;
            }
            // 2. Iterate acceptance with `called_from_restoration=true`
            //    — disables the rapid-barrier-increase guard and runs
            //    only the sufficient-progress branch
            //    (`IpFilterLSAcceptor.cpp:495-498`).
            pounce_common::utils::compare_le(
                trial_theta,
                (1.0 - gamma_theta) * reference_theta,
                reference_theta,
            ) || pounce_common::utils::compare_le(
                trial_barr - reference_barr,
                -gamma_phi * reference_theta,
                reference_barr,
            )
        }))
    }

    /// Port of `IpFilterLSAcceptor::PrepareRestoPhaseStart` →
    /// `AugmentFilter` (`IpFilterLSAcceptor.cpp:297-308, 898-901`).
    /// Called by the algorithm immediately before restoration; adds
    /// the resto-entry envelope to the filter so that after recovery,
    /// the outer's Newton step is forced to make real progress vs
    /// the entry point. Without this, pounce on DECONVBNE enters
    /// restoration 323× (vs ipopt 21×): the outer accepts null-progress
    /// 'h' steps and immediately re-enters restoration.
    fn prepare_resto_phase_start(&mut self, reference_theta: Number, reference_barr: Number) {
        let phi_add = reference_barr - self.gamma_phi * reference_theta;
        let theta_add = (1.0 - self.gamma_theta) * reference_theta;
        self.filter.add(theta_add, phi_add, 0);
    }

    fn set_theta_max_fact(&mut self, theta_max_fact: Number) {
        self.theta_max_fact = theta_max_fact;
        self.theta_max = None;
    }

    fn set_theta_rows(&mut self, rows: Number) {
        // Only meaningful before `theta_max` locks; once locked, leave it
        // alone so the ceiling stays fixed for the rest of the solve as
        // upstream requires.
        if self.theta_max.is_none() && rows.is_finite() && rows >= 1.0 {
            self.theta_rows = rows;
        }
    }

    fn set_theta_max_row_scale_kappa(&mut self, kappa: Number) {
        self.theta_max_row_scale_kappa = kappa;
        self.theta_max = None;
    }

    fn set_theta_max_adaptive_trigger(&mut self, trigger: u32) {
        self.theta_max_adaptive_trigger = trigger;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#476: **the row floor is opt-in.** Its default is `0`, so a
    /// stock POUNCE reproduces upstream's `theta_max` exactly on every
    /// model — no row count, no size dependence. This is the invariant
    /// the corpus evidence demanded: at every nonzero kappa tried
    /// (`0.01 … 1.0`) `brainpc1/3/5/7` regressed identically, so the
    /// floor cannot be a default until it is gated on something better
    /// than the row count. If this test fails, a default change has been
    /// made and it needs a full-corpus sweep behind it.
    #[test]
    fn the_row_floor_is_off_by_default() {
        let a = FilterLsAcceptor::new();
        assert_eq!(a.theta_max_row_scale_kappa, 0.0);

        // ...and off means off: the floor is upstream's bare `1.0` even
        // once a large row count has been reported.
        let mut a = FilterLsAcceptor::new();
        a.set_theta_rows(52_013.0);
        assert_eq!(a.theta_max_reference_floor(), 1.0);
        assert_eq!(a.ensure_theta_max(0.0), 1e4);
    }

    /// gh#546: the adaptive rule's whole claim is that it responds to
    /// *evidence* rather than to problem size. Build a line search in
    /// which every trial is refused at the `theta_max` gate, repeat it
    /// for the trigger count, and the ceiling goes up.
    #[test]
    fn a_streak_of_gate_blocked_line_searches_raises_theta_max() {
        let mut a = FilterLsAcceptor::new();
        a.set_theta_rows(52_013.0);
        let locked = a.ensure_theta_max(0.0);
        assert_eq!(locked, 1e4, "upstream's collapsed ceiling");

        // Two full line searches short of the trigger: still locked.
        for _ in 0..(a.theta_max_adaptive_trigger - 1) {
            gate_blocked_line_search(&mut a, locked);
        }
        assert_eq!(a.theta_max, Some(locked), "raised before the trigger");

        gate_blocked_line_search(&mut a, locked);
        assert_eq!(
            a.theta_max,
            Some(locked * a.theta_max_adaptive_factor),
            "the trigger was reached and the ceiling did not move",
        );
        assert_eq!(a.n_theta_max_raises, 1);
    }

    /// The property the static row floor could not have: a model that is
    /// *converging* never trips this, because converging means trials are
    /// getting past the gate. `brainpc1/3/5/7` regressed at every kappa
    /// under the static floor; under this rule they are untouched by
    /// construction, and this test is what pins "by construction".
    #[test]
    fn a_line_search_the_filter_refuses_does_not_raise_theta_max() {
        let mut a = FilterLsAcceptor::new();
        let locked = a.ensure_theta_max(1.0);

        // Many line searches, all refused — but refused *below* the
        // ceiling, i.e. by the filter / iterate tests that are supposed
        // to refuse them.
        for _ in 0..(a.theta_max_adaptive_trigger * 10) {
            for _ in 0..5 {
                // theta_trial under the ceiling, but no progress on
                // either measure ⇒ rejected, and not at the gate.
                assert_eq!(
                    a.check_acceptability(1.0, 1.0, 1.0, -1.0, locked / 2.0, 1e3),
                    AcceptDecision::Reject,
                );
            }
            a.end_of_line_search();
        }
        assert_eq!(a.theta_max, Some(locked), "the ceiling must not move");
        assert_eq!(a.n_theta_max_raises, 0);
    }

    /// A line search in which the gate refused *some* trials is not
    /// evidence the gate is binding — the ones it let through were
    /// judged on their merits. Only "every trial" counts.
    #[test]
    fn a_partially_gate_blocked_line_search_breaks_the_streak() {
        let mut a = FilterLsAcceptor::new();
        let locked = a.ensure_theta_max(1.0);

        for _ in 0..(a.theta_max_adaptive_trigger - 1) {
            gate_blocked_line_search(&mut a, locked);
        }
        // One line search where the first trial cleared the gate.
        let _ = a.check_acceptability(1.0, 1.0, 1.0, -1.0, locked / 2.0, 1e3);
        let _ = a.check_acceptability(0.5, 1.0, 1.0, -1.0, locked * 10.0, 1e3);
        a.end_of_line_search();
        assert_eq!(a.gate_blocked_ls_streak, 0, "the streak must reset");

        // ...and the streak now has to be rebuilt from scratch.
        for _ in 0..(a.theta_max_adaptive_trigger - 1) {
            gate_blocked_line_search(&mut a, locked);
        }
        assert_eq!(a.theta_max, Some(locked));
    }

    /// Wächter–Biegler Thm. 2 needs `theta_max` finite, not fixed. The
    /// cap is what keeps it finite: a solve cannot ratchet the ceiling
    /// away one line search at a time.
    #[test]
    fn the_ceiling_stops_rising_at_the_raise_cap() {
        let mut a = FilterLsAcceptor::new();
        let locked = a.ensure_theta_max(1.0);
        let cap = a.theta_max_adaptive_max_raises;

        for _ in 0..(a.theta_max_adaptive_trigger * (cap + 5)) {
            let ceiling = a.theta_max.unwrap();
            gate_blocked_line_search(&mut a, ceiling);
        }
        assert_eq!(a.n_theta_max_raises, cap);
        let expected = locked * a.theta_max_adaptive_factor.powi(cap as i32);
        assert!(
            (a.theta_max.unwrap() - expected).abs() < 1e-6 * expected,
            "ceiling {:?} is not exactly {cap} raises above {locked}",
            a.theta_max,
        );
    }

    /// `theta_max_adaptive_trigger = 0` is the escape hatch back to
    /// upstream's fixed ceiling. It has to be a real off switch, not a
    /// smaller threshold — the restoration sub-IPM wiring relies on it
    /// (`resto_inner_solver.rs`), since upstream already corrects the
    /// resto phase's instance of this with `theta_max_fact = 1e8`.
    #[test]
    fn trigger_zero_leaves_the_ceiling_exactly_where_upstream_locks_it() {
        let mut a = FilterLsAcceptor::new();
        a.set_theta_max_adaptive_trigger(0);
        let locked = a.ensure_theta_max(1.0);

        for _ in 0..100 {
            gate_blocked_line_search(&mut a, locked);
        }
        assert_eq!(a.theta_max, Some(locked));
        assert_eq!(a.n_theta_max_raises, 0);
    }

    /// The boundary hook is driven by the line-search driver, which
    /// reaches it once per outer iteration — but the soft-restoration
    /// path reaches it without running an α-loop. A boundary call with
    /// no trials behind it must not be counted as a gate-blocked line
    /// search, or the rule would fire on evidence that does not exist.
    #[test]
    fn a_boundary_with_no_trials_is_not_evidence() {
        let mut a = FilterLsAcceptor::new();
        let locked = a.ensure_theta_max(1.0);
        for _ in 0..100 {
            a.end_of_line_search();
        }
        assert_eq!(a.gate_blocked_ls_streak, 0);
        assert_eq!(a.theta_max, Some(locked));
    }

    /// Run one line search in which every trial is refused at the gate.
    /// `ceiling` is the currently locked `theta_max`; trials are placed
    /// an order of magnitude above it.
    fn gate_blocked_line_search(a: &mut FilterLsAcceptor, ceiling: Number) {
        for k in 0..4 {
            let alpha = 0.5_f64.powi(k);
            assert_eq!(
                a.check_acceptability(alpha, 1.0, 1.0, -1.0, ceiling * 10.0, 1.0),
                AcceptDecision::Reject,
                "trial must be refused at the gate for this helper to mean anything",
            );
        }
        a.end_of_line_search();
    }

    /// gh#476: the degeneracy the row floor exists to fix. `theta` is a
    /// 1-norm over constraint rows, so a model started at a FEASIBLE
    /// point (`theta_0 = 0`) collapses upstream's `max(1, theta_0)` to
    /// the bare constant `1` and locks `theta_max = theta_max_fact`
    /// regardless of how many rows there are. On `robot_a`
    /// (`m = 52013`) that is a mean per-row allowance of `1e4/52013 =
    /// 0.19`, while the path to the optimum passes through
    /// `theta ~ 9.4e7` — so every step toward the solution is refused
    /// at the gate and the solve grinds to `max_iter`.
    #[test]
    fn feasible_start_on_a_large_model_does_not_collapse_theta_max() {
        let rows = 52_013.0; // robot_a
        let theta_zero = 0.0; // starts feasible

        // Upstream's floor: the row count is invisible, so the ceiling is
        // the bare `theta_max_fact` and the converged path is unreachable.
        let mut upstream = FilterLsAcceptor::new();
        upstream.theta_max_row_scale_kappa = 0.0;
        upstream.set_theta_rows(rows);
        let upstream_max = upstream.ensure_theta_max(theta_zero);
        assert_eq!(upstream_max, 1e4);
        assert!(
            9.4e7 > upstream_max,
            "the trial theta on robot_a's route to the optimum must be \
             above the upstream ceiling — otherwise this test pins nothing",
        );

        // With the row floor opted into, the ceiling scales with the
        // model and the same route is admissible.
        let mut scaled = FilterLsAcceptor::new();
        scaled.theta_max_row_scale_kappa = 1.0;
        scaled.set_theta_rows(rows);
        let scaled_max = scaled.ensure_theta_max(theta_zero);
        assert_eq!(scaled_max, 1e4 * rows);
        assert!(9.4e7 < scaled_max);
    }

    /// The floor keeps the ceiling's *meaning* — "a mean per-row
    /// violation of `theta_max_fact`" — invariant in `m`, which is the
    /// whole justification for scaling by the row count rather than by
    /// some other quantity.
    #[test]
    fn row_floor_holds_the_per_row_allowance_constant_across_model_sizes() {
        for rows in [1.0, 10.0, 5_000.0, 52_013.0, 1e6] {
            let mut a = FilterLsAcceptor::new();
            a.theta_max_row_scale_kappa = 1.0;
            a.set_theta_rows(rows);
            let per_row = a.ensure_theta_max(0.0) / rows;
            assert!(
                (per_row - a.theta_max_fact).abs() < 1e-9 * a.theta_max_fact,
                "per-row allowance drifted at rows = {rows}: {per_row}",
            );
        }
    }

    /// The floor only ever *raises* the reference: a model whose own
    /// `theta_0` already exceeds `kappa * rows` is untouched, so the
    /// change cannot tighten the filter on any problem.
    #[test]
    fn row_floor_never_lowers_the_reference() {
        let rows = 100.0;
        let big_theta_zero = 1e6; // >> kappa * rows
        let mut a = FilterLsAcceptor::new();
        a.theta_max_row_scale_kappa = 1.0;
        a.set_theta_rows(rows);
        assert_eq!(a.ensure_theta_max(big_theta_zero), 1e4 * big_theta_zero);

        // ...and that matches what upstream would have locked, exactly.
        let mut upstream = FilterLsAcceptor::new();
        upstream.theta_max_row_scale_kappa = 0.0;
        assert_eq!(
            upstream.ensure_theta_max(big_theta_zero),
            a.ensure_theta_max(big_theta_zero),
        );
    }

    /// A single-row problem reduces to upstream exactly, for any
    /// `theta_0` — the floor is `max(kappa * 1, 1) = 1`, which is
    /// upstream's constant.
    #[test]
    fn single_row_problem_is_bit_for_bit_upstream() {
        for theta_zero in [0.0, 1e-12, 0.5, 1.0, 7.25, 1e9] {
            let mut a = FilterLsAcceptor::new();
            a.theta_max_row_scale_kappa = 1.0;
            a.set_theta_rows(1.0);
            let mut upstream = FilterLsAcceptor::new();
            upstream.theta_max_row_scale_kappa = 0.0;
            assert_eq!(
                a.ensure_theta_max(theta_zero),
                upstream.ensure_theta_max(theta_zero),
                "diverged from upstream at theta_0 = {theta_zero}",
            );
        }
    }

    /// `theta_max` locks once per solve. A later row count — the resto
    /// sub-IPM's, say, or a second line search — must not move a
    /// ceiling the filter has already been judging trials against.
    #[test]
    fn theta_rows_is_ignored_once_theta_max_has_locked() {
        let mut a = FilterLsAcceptor::new();
        a.theta_max_row_scale_kappa = 1.0;
        a.set_theta_rows(10.0);
        let locked = a.ensure_theta_max(0.0);
        assert_eq!(locked, 1e4 * 10.0);

        a.set_theta_rows(52_013.0);
        assert_eq!(
            a.ensure_theta_max(0.0),
            locked,
            "a locked theta_max moved under a later row count",
        );
    }

    #[test]
    fn switching_condition_requires_descent() {
        let a = FilterLsAcceptor::new();
        // d_phi >= 0 always returns false.
        assert!(!a.is_switching_condition(1.0, 0.0, 1.0));
        assert!(!a.is_switching_condition(1.0, 0.5, 1.0));
    }

    #[test]
    fn switching_condition_holds_when_descent_dominates_theta() {
        let a = FilterLsAcceptor::new();
        // alpha * (-d_phi)^s_phi vs delta * theta^s_theta.
        // alpha=1, d_phi=-1 → lhs=1; theta=1e-3, s_theta=1.1
        //   → rhs ≈ 1.0 * (1e-3)^1.1 ≈ 5e-4. lhs > rhs.
        assert!(a.is_switching_condition(1.0, -1.0, 1e-3));
    }

    #[test]
    fn armijo_strict_decrease() {
        let a = FilterLsAcceptor::new();
        // phi - phi_trial >= -eta_phi * alpha * d_phi  (with d_phi<0).
        // alpha=1, d_phi=-1, eta_phi=1e-8 → req: phi_trial - phi <= -1e-8.
        assert!(a.armijo_holds(1.0, -1.0, 0.0, -1e-7));
        assert!(!a.armijo_holds(1.0, -1.0, 0.0, 1e-7));
    }

    #[test]
    fn accept_when_filter_clear_and_progress_in_phi() {
        let mut a = FilterLsAcceptor::new();
        // Not in switching mode (small descent, small alpha).
        // Sufficient progress: phi_trial = -1 < phi=0 - gamma_phi*theta=0.
        let d = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 1.0, -1.0);
        assert_eq!(d, AcceptDecision::Accept);
    }

    #[test]
    fn is_sufficient_progress_accepts_round_off_boundary_like_live_path() {
        // L6: `is_sufficient_progress` must carry the same `compare_le`
        // round-off slack as the live `check_acceptability` path. Construct
        // the φ-branch boundary exactly: `phi_trial - phi == -gamma_phi*theta`
        // with the θ-branch firmly false. A bare `<` (the old, divergent
        // implementation) rejects this equality; `compare_le`'s 10·eps·|phi|
        // slack accepts it.
        let a = FilterLsAcceptor::new();
        let theta = 1.0;
        let theta_trial = 1.0; // not < (1-gamma_theta)*theta → θ-branch false
        let phi = 0.0;
        let phi_trial = -a.gamma_phi * theta; // φ-branch equality boundary
        assert!(
            a.is_sufficient_progress(theta, phi, theta_trial, phi_trial),
            "is_sufficient_progress must mirror the live compare_le path and \
             accept the boundary phi_trial - phi == -gamma_phi*theta"
        );
    }

    #[test]
    fn check_acceptability_honors_obj_max_inc_field() {
        // L6: the rapid-barrier-increase guard must read the `obj_max_inc`
        // field, not a hard-coded 5.0. A ~1e7 jump in phi (log10 ≈ 7) is
        // rejected at the default cap 5.0 (basval 1.0 → threshold 6) but
        // accepted once the field is raised to 10.0 (threshold 11). d_phi=0
        // keeps us out of the switching/Armijo branch; theta_trial stays
        // under theta_max; the θ-branch satisfies sufficient progress so the
        // decision turns purely on the rapid-increase guard.
        let args = (1.0, 1.0, 1.0, 0.0, 0.5, 1.0 + 1e7);

        let mut default_cap = FilterLsAcceptor::new();
        assert_eq!(
            default_cap.check_acceptability(args.0, args.1, args.2, args.3, args.4, args.5),
            AcceptDecision::Reject,
            "default obj_max_inc=5.0 should reject a 1e7 barrier jump"
        );

        let mut relaxed_cap = FilterLsAcceptor::new();
        relaxed_cap.obj_max_inc = 10.0;
        assert_eq!(
            relaxed_cap.check_acceptability(args.0, args.1, args.2, args.3, args.4, args.5),
            AcceptDecision::Accept,
            "raising obj_max_inc to 10.0 should accept the same jump"
        );
    }

    #[test]
    fn reject_when_filter_dominates_trial() {
        let mut a = FilterLsAcceptor::new();
        a.filter.add(0.5, -0.5, 0);
        // Trial (1.0, 1.0) is dominated by (0.5, -0.5).
        let d = a.check_acceptability(1.0, 1.0, 0.0, -1.0, 1.0, 1.0);
        assert_eq!(d, AcceptDecision::Reject);
    }

    #[test]
    fn reject_when_switching_but_armijo_fails() {
        let mut a = FilterLsAcceptor::new();
        // Switching mode active, but phi_trial is *worse* than phi.
        let d = a.check_acceptability(1.0, 1e-3, 0.0, -1.0, 1e-3, 1.0);
        assert_eq!(d, AcceptDecision::Reject);
    }

    #[test]
    fn accept_when_switching_and_armijo_holds() {
        let mut a = FilterLsAcceptor::new();
        // Switching mode and phi_trial is much smaller than phi.
        let d = a.check_acceptability(1.0, 1e-3, 0.0, -1.0, 1e-3, -1.0);
        assert_eq!(d, AcceptDecision::Accept);
    }

    #[test]
    fn calc_alpha_min_floor_at_alpha_min_frac_times_gamma_theta_when_no_descent() {
        let mut a = FilterLsAcceptor::new();
        // d_phi >= 0 → upstream skips both descent-based tightenings;
        // alpha_min = alpha_min_frac * gamma_theta.
        let v = a.calc_alpha_min(0.0, 1.0);
        assert!((v - 0.05 * 1e-5).abs() < 1e-20);
    }

    #[test]
    fn calc_alpha_min_uses_descent_term_when_d_phi_negative() {
        let mut a = FilterLsAcceptor::new();
        // theta=1 (large), so the second tightening (theta <= theta_min)
        // does NOT fire on first call; theta_min lazy-inits to
        // theta_min_fact*max(1,theta) = 1e-4. theta=1 > 1e-4.
        // alpha_min = min(gamma_theta, gamma_phi*theta/(-d_phi))
        //           = min(1e-5, 1e-8*1/1)  = 1e-8
        // returned = alpha_min_frac * 1e-8 = 5e-10.
        let v = a.calc_alpha_min(-1.0, 1.0);
        assert!((v - 0.05 * 1e-8).abs() < 1e-25);
    }

    #[test]
    fn calc_alpha_min_lazy_inits_theta_min_from_first_reference() {
        let mut a = FilterLsAcceptor::new();
        let _ = a.calc_alpha_min(0.0, 0.5);
        // theta_min should now be 1e-4 * max(1, 0.5) = 1e-4.
        assert!((a.theta_min.unwrap() - 1e-4).abs() < 1e-15);
        // Subsequent calls keep the original theta_min (matches
        // upstream where theta_min_ stays set across iterations).
        let _ = a.calc_alpha_min(0.0, 100.0);
        assert!((a.theta_min.unwrap() - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn make_orig_progress_check_accepts_when_filter_clear_and_theta_drops() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let a = FilterLsAcceptor::new();
        // Empty filter, reference (theta=1.0, barr=0.0), trial below
        // (1-gamma_theta)*reference_theta ⇒ accepted via theta branch.
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        assert!(cb(2.0, 0.5)); // trial_barr=2.0 worse, but theta=0.5<<1.0
    }

    #[test]
    fn make_orig_progress_check_rejects_when_filter_dominates() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let mut a = FilterLsAcceptor::new();
        // Plant a filter entry that dominates the trial.
        a.filter.add(0.05, 0.0, 0);
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        // (theta_trial=0.1, barr_trial=0.5) is dominated by (0.05, 0.0).
        assert!(!cb(0.5, 0.1));
    }

    #[test]
    fn make_orig_progress_check_rejects_when_no_progress() {
        use crate::line_search::ls_acceptor::BacktrackingLsAcceptor;
        let a = FilterLsAcceptor::new();
        // Reference (theta=1.0, barr=0.0). Trial (theta=1.0, barr=2.0)
        // — no theta progress and barr increases. force_armijo skips
        // the rapid-increase guard, but the sufficient-progress
        // disjunction still fails on both branches.
        let cb = a
            .make_orig_progress_check(1.0, 0.0, 5.0)
            .expect("FilterLsAcceptor returns Some");
        assert!(!cb(2.0, 1.0));
    }

    #[test]
    fn filter_reset_heuristic_clears_after_trigger_consecutive_filter_rejected_accepts() {
        let mut a = FilterLsAcceptor::new();
        a.filter_reset_trigger = 2;
        a.max_filter_resets = 5;
        // Plant a filter entry that dominates the planned filter-reject
        // trial. Accept iterates between two trial sets:
        //   - theta_trial=1.0, phi_trial=10.0 (dominated by (0.5, 9.0))
        //   - theta_trial=0.4, phi_trial=8.0 (passes filter, accepted)
        // For the heuristic to fire we need the LS to record a
        // filter-due rejection just before each accept.
        a.filter.add(0.5, 9.0, 0);

        // Backtrack 1: filter-reject (sets last_rejection_due_to_filter).
        let r1 = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        assert_eq!(r1, AcceptDecision::Reject);
        assert!(a.last_rejection_due_to_filter);
        // Backtrack 2: accept on a smaller alpha → bumps count to 1.
        let r2 = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(r2, AcceptDecision::Accept);
        assert_eq!(a.count_successive_filter_rejections, 1);
        assert_eq!(a.n_filter_resets, 0);
        assert!(!a.filter.entries().is_empty());

        // Re-plant filter and repeat for the second accept → count
        // reaches the trigger=2, filter is reset.
        a.filter.add(0.5, 9.0, 0);
        let r3 = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        assert_eq!(r3, AcceptDecision::Reject);
        let r4 = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(r4, AcceptDecision::Accept);
        assert_eq!(a.n_filter_resets, 1);
        assert_eq!(a.count_successive_filter_rejections, 0);
        assert!(a.filter.entries().is_empty());
    }

    #[test]
    fn filter_reset_count_clears_when_last_rejection_was_iterate_test() {
        let mut a = FilterLsAcceptor::new();
        a.filter_reset_trigger = 2;
        a.filter.add(0.5, 9.0, 0);

        // First sequence: filter-rejection + accept → count = 1.
        let _ = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
        let _ = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        assert_eq!(a.count_successive_filter_rejections, 1);

        // Second sequence: iterate-reject (theta and phi both fail
        // sufficient progress AND switching/Armijo) + accept → count
        // resets to 0, no filter reset.
        a.filter.clear();
        // Reject: not switching (small alpha), no theta progress
        // (theta_trial == theta), and phi is essentially unchanged
        // (no sufficient barrier decrease).
        let r1 = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 1.0, 0.0);
        assert_eq!(r1, AcceptDecision::Reject);
        assert!(!a.last_rejection_due_to_filter);
        // Accept on a trial with theta progress.
        let r2 = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 0.5, 0.0);
        assert_eq!(r2, AcceptDecision::Accept);
        assert_eq!(a.count_successive_filter_rejections, 0);
        assert_eq!(a.n_filter_resets, 0);
    }

    #[test]
    fn filter_reset_disabled_when_max_filter_resets_zero() {
        let mut a = FilterLsAcceptor::new();
        a.max_filter_resets = 0;
        a.filter_reset_trigger = 1;
        a.filter.add(0.5, 9.0, 0);

        // Any number of filter-reject + accept cycles must not reset.
        for _ in 0..5 {
            a.filter.add(0.5, 9.0, 0);
            let _ = a.check_acceptability(1.0, 1.0, 10.0, -1.0, 1.0, 9.5);
            let _ = a.check_acceptability(0.5, 1.0, 10.0, -1.0, 0.4, 8.0);
        }
        assert_eq!(a.n_filter_resets, 0);
        assert!(!a.filter.entries().is_empty());
    }

    #[test]
    fn accept_via_theta_progress_when_phi_unchanged() {
        let mut a = FilterLsAcceptor::new();
        // Not in switching mode; phi stays put but theta drops by
        // more than gamma_theta. Sufficient progress in theta.
        // theta=1, theta_trial=0.5 < (1-1e-5)*1 = 0.99999.
        let d = a.check_acceptability(1e-12, 1.0, 0.0, -1e-12, 0.5, 0.0);
        assert_eq!(d, AcceptDecision::Accept);
    }
}
