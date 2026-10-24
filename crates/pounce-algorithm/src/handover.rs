//! Interior-point → active-set handover inside one solve (gh#1054).
//!
//! # Why
//!
//! On large collocation models the interior-point method gets close fast and
//! then spends a long tail meeting `tol`: the reporter of gh#1054 measured the
//! objective within 0.1 % of final after ~40 % of the iterations, the rest
//! going to a tail where the dual infeasibility oscillates and the filter
//! backtracks. Near the solution the active set is already identified, and an
//! SQP step on it converges in a few iterations. Running the two as separate
//! solves needs a working set between them, and getting one at that size was
//! the blocker: post-convergence crossover ran after the tail it was meant to
//! avoid.
//!
//! # What it does
//!
//! With `ipm_sqp_handover=yes`, the interior-point loop offers each iterate
//! that has not converged to [`HandoverTrigger::should_try`]. When the
//! trigger fires, the iterate goes through the same bridge as post-convergence
//! crossover ([`crate::crossover::run`]): identify the active set from the
//! multipliers against the slacks, take one QP step over it, and fall back to
//! the full active-set SQP from that working set. The point that comes back
//! replaces the interior iterate only if it meets the stopping tolerances,
//! and the solve then ends `Solve_Succeeded`. Otherwise the interior
//! iteration continues from exactly where it was: nothing it carries is
//! touched by a declined attempt, so the fallback the request asked for is
//! the interior method itself, not a restart.
//!
//! # When it fires
//!
//! Either test, whichever comes first:
//!
//! - **tolerance**: the overall NLP error is at or below
//!   `ipm_sqp_handover_tol` (a loosened `tol`);
//! - **stall**: `ipm_sqp_handover_stall_iter > 0` and the best error seen has
//!   not halved in that many iterations.
//!
//! A declined attempt re-arms only after the error has fallen another order
//! of magnitude below the error at that attempt, or a full stall window has
//! passed without one, and at most `ipm_sqp_handover_max_attempts` attempts
//! are made. Without the back-off, a model whose active set is not yet
//! identifiable would pay an SQP attempt on every remaining iteration.
//!
//! # Identification
//!
//! At an interior iterate a bound's slack `s` and multiplier `z` satisfy
//! `s·z ≈ μ`, so an active bound has `s ≈ μ/z` and an inactive one
//! `z ≈ μ/s`; `√μ` separates the two. The primal tolerance handed to the
//! classifier is therefore `max(crossover_primal_tol, √μ)`: the crossover
//! default (`1e-6`) is sized for a converged iterate, and at a loosened
//! tolerance it would call every active bound inactive. A misclassified row
//! costs pivots in the QP, not correctness.

use pounce_common::types::{Index, Number};

/// Tuning, read from the `ipm_sqp_handover*` options.
#[derive(Debug, Clone)]
pub struct HandoverOptions {
    /// `ipm_sqp_handover`. Default off.
    pub enabled: bool,
    /// `ipm_sqp_handover_tol`: try once the overall NLP error is this small.
    pub tol: Number,
    /// `ipm_sqp_handover_stall_iter`: try once the best error has not halved
    /// in this many iterations. `0` disables the stall test.
    pub stall_iter: Index,
    /// `ipm_sqp_handover_max_attempts`.
    pub max_attempts: u32,
}

impl Default for HandoverOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            tol: 1e-4,
            stall_iter: 0,
            max_attempts: 3,
        }
    }
}

/// Decides, iteration by iteration, whether to attempt a handover. See the
/// module docs for the rules.
#[derive(Debug, Clone)]
pub struct HandoverTrigger {
    opts: HandoverOptions,
    attempts: u32,
    /// Error and iteration at the last declined attempt.
    last_attempt: Option<(Number, Index)>,
    /// Best error seen, and the iteration it was first reached at.
    best: Option<(Number, Index)>,
}

impl HandoverTrigger {
    pub fn new(opts: HandoverOptions) -> Self {
        Self {
            opts,
            attempts: 0,
            last_attempt: None,
            best: None,
        }
    }

    /// Whether to attempt a handover at iteration `iter` with overall NLP
    /// error `err`. Counts the attempt when it says yes.
    pub fn should_try(&mut self, iter: Index, err: Number) -> bool {
        if !self.opts.enabled || !err.is_finite() || self.attempts >= self.opts.max_attempts {
            return false;
        }
        // "Best" moves only on a halving, so the stall window measures the
        // time since the last real progress, not since the last new digit.
        match self.best {
            Some((b, _)) if err > 0.5 * b => {}
            _ => self.best = Some((err, iter)),
        }
        let stall = self.opts.stall_iter;
        let stalled = stall > 0 && self.best.is_some_and(|(_, at)| iter - at >= stall);
        let fire = err <= self.opts.tol || stalled;
        if !fire {
            return false;
        }
        // Back-off after a declined attempt.
        if let Some((e0, i0)) = self.last_attempt {
            let improved = err <= 0.1 * e0;
            let waited = stall > 0 && iter - i0 >= stall;
            if !(improved || waited) {
                return false;
            }
        }
        self.attempts += 1;
        self.last_attempt = Some((err, iter));
        // A fresh stall window starts at the attempt.
        self.best = Some((err, iter));
        true
    }

    /// Attempts made so far.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

/// What the handover did on the most recent solve. Retrieved from
/// [`crate::application::IpoptApplication::handover_report`].
#[derive(Debug, Clone, Default)]
pub struct HandoverReport {
    /// Attempts made.
    pub attempts: u32,
    /// Interior-point iteration at which an attempt was accepted, if one was.
    pub accepted_at_iter: Option<Index>,
    /// The crossover report of the last attempt (the accepted one, if any).
    pub last: Option<crate::crossover::CrossoverReport>,
}

/// The primal tolerance for classifying the active set at an interior
/// iterate whose barrier parameter is `mu`. See the module docs.
pub fn identification_primal_tol(crossover_primal_tol: Number, mu: Number) -> Number {
    let sep = if mu.is_finite() && mu > 0.0 {
        mu.sqrt()
    } else {
        0.0
    };
    crossover_primal_tol.max(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trig(tol: Number, stall: Index, max: u32) -> HandoverTrigger {
        HandoverTrigger::new(HandoverOptions {
            enabled: true,
            tol,
            stall_iter: stall,
            max_attempts: max,
        })
    }

    #[test]
    fn off_never_fires() {
        let mut t = HandoverTrigger::new(HandoverOptions::default());
        assert!(!t.should_try(10, 0.0));
    }

    #[test]
    fn tolerance_fires_once_reached() {
        let mut t = trig(1e-4, 0, 3);
        assert!(!t.should_try(1, 1.0));
        assert!(!t.should_try(2, 1e-3));
        assert!(t.should_try(3, 5e-5));
        assert_eq!(t.attempts(), 1);
    }

    #[test]
    fn a_declined_attempt_backs_off_until_an_order_of_magnitude() {
        let mut t = trig(1e-4, 0, 3);
        assert!(t.should_try(3, 5e-5));
        assert!(!t.should_try(4, 3e-5), "not yet 10x below the last attempt");
        assert!(!t.should_try(5, 6e-6));
        assert!(t.should_try(6, 4e-6));
        assert_eq!(t.attempts(), 2);
    }

    #[test]
    fn stall_fires_without_reaching_the_tolerance() {
        let mut t = trig(1e-8, 5, 3);
        assert!(!t.should_try(0, 1e-2));
        // Oscillating around 1e-2, never halving: the gh#1054 tail.
        for (k, e) in [(1, 1.2e-2), (2, 0.9e-2), (3, 1.1e-2), (4, 0.8e-2)] {
            assert!(!t.should_try(k, e));
        }
        assert!(t.should_try(5, 1.0e-2));
        // A fresh window: no second attempt for another 5 iterations.
        assert!(!t.should_try(8, 1.0e-2));
        assert!(t.should_try(10, 1.0e-2));
    }

    #[test]
    fn progress_resets_the_stall_window() {
        let mut t = trig(1e-8, 3, 3);
        assert!(!t.should_try(0, 1.0));
        assert!(!t.should_try(2, 0.4)); // halved: window restarts at 2
        assert!(!t.should_try(4, 0.35));
        assert!(t.should_try(5, 0.35));
    }

    #[test]
    fn attempts_are_capped() {
        let mut t = trig(1.0, 1, 2);
        assert!(t.should_try(0, 0.5));
        assert!(t.should_try(1, 0.5));
        assert!(!t.should_try(2, 0.5));
        assert!(!t.should_try(3, 1e-9));
    }

    #[test]
    fn identification_tolerance_follows_the_barrier_width() {
        assert_eq!(identification_primal_tol(1e-6, 1e-4), 1e-2);
        assert_eq!(identification_primal_tol(1e-6, 1e-14), 1e-6);
        assert_eq!(identification_primal_tol(1e-6, Number::NAN), 1e-6);
    }
}
