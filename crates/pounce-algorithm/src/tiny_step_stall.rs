//! When the tiny-step stop may stop waiting for the multiplier step.
//!
//! Upstream stops at a tiny step only after the multiplier step is also
//! below `tiny_step_y_tol` (`IpBacktrackingLineSearch.cpp`), so that a
//! primal iterate which has stopped moving is not abandoned while its
//! multipliers are still converging. At a point whose multipliers are not
//! unique they need not converge: they run away, the multiplier step stays
//! large, and the solve repeats the same iterate until `max_iter`.
//!
//! [`TinyStepDualStall`] bounds that wait by dual progress. Fed one
//! observation per tiny primal step, it releases the stop once
//! `max_iters` consecutive tiny steps have gone by without the scaled dual
//! infeasibility falling below `0.99` times its best value in the run.
//! Multipliers that are converging reduce the dual infeasibility and reset
//! the count, so they are never cut short; a settled multiplier step or a
//! non-tiny step ends the run.
//!
//! The case: the phase-change flash example
//! (`pounce.examples.flash_mpcc.declare`) at 332 K, started from 331 K's
//! all-vapor answer, sat for 3 000 iterations at a primal step of `5.6e-16`
//! with the dual infeasibility constant at `1.3e29`, and the adaptive-mu
//! re-solve then finished it in 14. With the default bound the base attempt
//! ends after 43 and the same re-solve takes it from there
//! (`python/tests/test_flash_mpcc.py`).

use pounce_common::types::Number;

/// Relative decrease of the dual infeasibility that counts as progress.
const PROGRESS: Number = 0.99;

#[derive(Debug, Clone)]
pub struct TinyStepDualStall {
    /// `tiny_step_dual_stall_iters`; `0` never releases (upstream's rule).
    pub max_iters: usize,
    count: usize,
    best: Number,
    released: bool,
}

impl TinyStepDualStall {
    pub fn new(max_iters: usize) -> Self {
        Self {
            max_iters,
            count: 0,
            best: Number::INFINITY,
            released: false,
        }
    }

    /// A step that was not tiny ends the run of tiny steps.
    pub fn not_tiny(&mut self) {
        self.count = 0;
        self.best = Number::INFINITY;
    }

    /// One tiny primal step. `y_settled`: the multiplier step is already
    /// below `tiny_step_y_tol`, so upstream's own rule applies and this
    /// one has nothing to add. `acceptable`: the iterate passes the
    /// acceptable-level test, so it has an exit of its own
    /// (`acceptable_iter`) and is left to it -- cutting it short would turn
    /// `Solved_To_Acceptable_Level` into a tiny-step stop (gh#532's
    /// `dual_inf_scale_kappa=0` case, where the dual sits on a round-off
    /// floor at an acceptable point). `dual_inf`: the scaled dual
    /// infeasibility at the iterate. Returns whether the stop may proceed
    /// as if the multiplier step had settled.
    pub fn observe(&mut self, y_settled: bool, acceptable: bool, dual_inf: Number) -> bool {
        if y_settled || acceptable || self.max_iters == 0 {
            self.not_tiny();
            return false;
        }
        if dual_inf.is_finite() && dual_inf < PROGRESS * self.best {
            self.best = dual_inf;
            self.count = 0;
        } else {
            self.count += 1;
        }
        let stalled = self.count >= self.max_iters;
        self.released |= stalled;
        stalled
    }

    /// Whether the rule released the stop at least once.
    pub fn released(&self) -> bool {
        self.released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flat or growing dual infeasibility over tiny steps is released after
    /// exactly `max_iters` steps without progress, the first observation
    /// setting the baseline.
    #[test]
    fn a_dual_that_stops_improving_is_released_after_max_iters() {
        for duals in [
            vec![1.3e29; 20],
            (0..20).map(|k| 1e21 * 10f64.powi(k)).collect(),
        ] {
            let mut s = TinyStepDualStall::new(10);
            let fired: Vec<bool> = duals.iter().map(|&d| s.observe(false, false, d)).collect();
            assert_eq!(fired.iter().position(|&f| f), Some(10), "{duals:?}");
            assert!(s.released());
        }
    }

    /// Multipliers that are converging improve the dual every step and are
    /// never cut short, however long the run of tiny steps.
    #[test]
    fn converging_multipliers_are_never_released() {
        let mut s = TinyStepDualStall::new(10);
        for k in 0..200 {
            assert!(!s.observe(false, false, 1e3 * 0.9f64.powi(k)), "step {k}");
        }
        assert!(!s.released());
    }

    /// Progress anywhere in the run restarts the count.
    #[test]
    fn progress_restarts_the_count() {
        let mut s = TinyStepDualStall::new(10);
        for _ in 0..9 {
            assert!(!s.observe(false, false, 5.0));
        }
        assert!(!s.observe(false, false, 1.0)); // progress: count back to 0
        for _ in 0..9 {
            assert!(!s.observe(false, false, 1.0));
        }
        assert!(s.observe(false, false, 1.0));
    }

    /// A settled multiplier step is upstream's case, and a non-tiny step
    /// ends the run; both reset.
    #[test]
    fn a_settled_multiplier_step_or_a_real_step_resets() {
        let mut s = TinyStepDualStall::new(3);
        s.observe(false, false, 1.0);
        s.observe(false, false, 1.0);
        assert!(!s.observe(true, false, 1.0));
        assert!(!s.observe(false, false, 1.0)); // baseline again
        assert!(!s.observe(false, false, 1.0));
        s.not_tiny();
        assert!(!s.observe(false, false, 1.0));
        assert!(!s.observe(false, false, 1.0));
        assert!(!s.observe(false, false, 1.0));
        assert!(s.observe(false, false, 1.0));
    }

    /// An acceptable iterate is left to the acceptable-level exit, and
    /// resets the run like a settled multiplier step.
    #[test]
    fn an_acceptable_iterate_is_left_to_its_own_exit() {
        let mut s = TinyStepDualStall::new(3);
        for _ in 0..100 {
            assert!(!s.observe(false, true, 1.3e29));
        }
        assert!(!s.released());
        assert!(!s.observe(false, false, 1.0));
        assert!(!s.observe(false, false, 1.0));
        assert!(!s.observe(false, true, 1.0));
        assert!(!s.observe(false, false, 1.0));
    }

    /// `0` is upstream's rule: never released.
    #[test]
    fn zero_is_upstreams_rule() {
        let mut s = TinyStepDualStall::new(0);
        for _ in 0..5000 {
            assert!(!s.observe(false, false, 1.3e29));
        }
        assert!(!s.released());
    }

    /// A non-finite dual is no progress, and sets no baseline: it counts
    /// from the first observation.
    #[test]
    fn a_non_finite_dual_is_not_progress() {
        let mut s = TinyStepDualStall::new(2);
        assert!(!s.observe(false, false, Number::NAN));
        assert!(s.observe(false, false, Number::INFINITY));
    }
}
