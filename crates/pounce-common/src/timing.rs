//! Per-task timing accumulator.
//!
//! Mirrors `Common/IpTimedTask.hpp` (`Common/IpDebug.{hpp,cpp}` is
//! omitted — debug tracing is replaced by the journalist).

use crate::types::Number;
use crate::utils::{cpu_time, sys_time, wallclock_time};
use std::cell::Cell;

/// Equivalent to `Ipopt::TimedTask`. Use [`TimedTask::start`] /
/// [`TimedTask::end`] around a section to accumulate cpu/system/wall
/// time. [`TimedTask::end_if_started`] is the exception-safe variant.
#[derive(Debug)]
pub struct TimedTask {
    enabled: Cell<bool>,
    start_called: Cell<bool>,
    end_called: Cell<bool>,
    start_cpu: Cell<Number>,
    start_sys: Cell<Number>,
    start_wall: Cell<Number>,
    total_cpu: Cell<Number>,
    total_sys: Cell<Number>,
    total_wall: Cell<Number>,
}

impl Default for TimedTask {
    fn default() -> Self {
        Self {
            enabled: Cell::new(true),
            start_called: Cell::new(false),
            end_called: Cell::new(true),
            start_cpu: Cell::new(0.0),
            start_sys: Cell::new(0.0),
            start_wall: Cell::new(0.0),
            total_cpu: Cell::new(0.0),
            total_sys: Cell::new(0.0),
            total_wall: Cell::new(0.0),
        }
    }
}

impl TimedTask {
    pub fn new() -> Self { Self::default() }

    pub fn enable(&self) { self.enabled.set(true); }
    pub fn disable(&self) { self.enabled.set(false); }
    pub fn is_enabled(&self) -> bool { self.enabled.get() }
    pub fn is_started(&self) -> bool { self.start_called.get() }

    pub fn reset(&self) {
        self.total_cpu.set(0.0);
        self.total_sys.set(0.0);
        self.total_wall.set(0.0);
        self.start_called.set(false);
        self.end_called.set(true);
    }

    pub fn start(&self) {
        if !self.enabled.get() { return; }
        self.end_called.set(false);
        self.start_called.set(true);
        self.start_cpu.set(cpu_time());
        self.start_sys.set(sys_time());
        self.start_wall.set(wallclock_time());
    }

    pub fn end(&self) {
        if !self.enabled.get() { return; }
        self.end_called.set(true);
        self.start_called.set(false);
        self.total_cpu.set(self.total_cpu.get() + cpu_time() - self.start_cpu.get());
        self.total_sys.set(self.total_sys.get() + sys_time() - self.start_sys.get());
        self.total_wall.set(self.total_wall.get() + wallclock_time() - self.start_wall.get());
    }

    pub fn end_if_started(&self) {
        if !self.enabled.get() { return; }
        if self.start_called.get() {
            self.end();
        }
    }

    pub fn total_cpu_time(&self) -> Number { self.total_cpu.get() }
    pub fn total_sys_time(&self) -> Number { self.total_sys.get() }
    pub fn total_wallclock_time(&self) -> Number { self.total_wall.get() }

    /// Running wallclock seconds since `start()` plus accumulated total
    /// from prior start/end cycles. When the task is not currently
    /// started this is the same as [`Self::total_wallclock_time`].
    /// Used by `OptErrorConvCheck` to gate `max_wall_time` mid-solve
    /// without forcing a `start()`/`end()` round-trip every iter.
    pub fn live_wallclock_time(&self) -> Number {
        if self.enabled.get() && self.start_called.get() {
            self.total_wall.get() + wallclock_time() - self.start_wall.get()
        } else {
            self.total_wall.get()
        }
    }

    /// Live counterpart of [`Self::total_cpu_time`]; see
    /// [`Self::live_wallclock_time`] for the contract.
    pub fn live_cpu_time(&self) -> Number {
        if self.enabled.get() && self.start_called.get() {
            self.total_cpu.get() + cpu_time() - self.start_cpu.get()
        } else {
            self.total_cpu.get()
        }
    }

    /// RAII-style guard: start the timer immediately, end it when the
    /// returned value is dropped (or when [`TimedGuard::stop`] is
    /// called). Survives early returns / `?` in the caller scope.
    pub fn guard(&self) -> TimedGuard<'_> {
        self.start();
        TimedGuard { task: Some(self) }
    }
}

/// Drop-on-end guard returned by [`TimedTask::guard`]. Calls
/// [`TimedTask::end_if_started`] in its destructor so a function with
/// many exit paths can wrap a section with a single line.
#[must_use = "the guard ends the timer when dropped; bind it to a variable"]
pub struct TimedGuard<'a> {
    task: Option<&'a TimedTask>,
}

impl<'a> TimedGuard<'a> {
    /// End the timer immediately. Useful when you want to stop timing
    /// before the natural scope exit (e.g. before a long-running
    /// follow-up that should not be attributed to this section).
    pub fn stop(mut self) {
        if let Some(t) = self.task.take() {
            t.end_if_started();
        }
    }
}

impl<'a> Drop for TimedGuard<'a> {
    fn drop(&mut self) {
        if let Some(t) = self.task.take() {
            t.end_if_started();
        }
    }
}

/// Aggregate of per-subsystem [`TimedTask`] counters. Mirrors
/// `Algorithm/IpTimingStatistics.{hpp,cpp}`. Owned by `IpoptApplication`
/// and shared (via `Rc`) with the algorithm, NLP, and KKT solver so each
/// subsystem can bump its own field. Reported at the end of a solve
/// when `print_timing_statistics yes`.
#[derive(Debug, Default)]
pub struct TimingStatistics {
    pub overall_alg: TimedTask,
    pub print_problem_statistics: TimedTask,
    pub initialize_iterates: TimedTask,
    pub update_hessian: TimedTask,
    pub output_iteration: TimedTask,
    pub update_barrier_parameter: TimedTask,
    pub compute_search_direction: TimedTask,
    pub compute_acceptable_trial_point: TimedTask,
    pub accept_trial_point: TimedTask,
    pub check_convergence: TimedTask,

    pub linear_system_factorization: TimedTask,
    pub linear_system_back_solve: TimedTask,
    pub linear_system_structure_converter: TimedTask,
    pub linear_system_structure_converter_init: TimedTask,
    pub quality_function_search: TimedTask,
    pub total_callback_time: TimedTask,
    pub total_function_evaluation_time: TimedTask,
    pub eval_obj: TimedTask,
    pub eval_grad_obj: TimedTask,
    pub eval_constr: TimedTask,
    pub eval_constr_jac: TimedTask,
    pub eval_lag_hess: TimedTask,
}

impl TimingStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Format a per-subsystem timing report (wall-clock seconds, mirroring
    /// upstream `IpoptApplication`'s end-of-run "Timing Statistics" block
    /// but with sys/cpu columns omitted — pounce only tracks wall time).
    /// Lines are indented to reflect the upstream visual nesting
    /// (OverallAlgorithm → its phases; TotalFunctionEvaluations → its
    /// per-callback breakdown). Returns a multi-line string ending in a
    /// trailing newline so callers can `print!` it directly.
    pub fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let row = |s: &mut String, label: &str, t: &TimedTask| {
            let _ = writeln!(
                s,
                "{label:<42} {wall:>10.3}s",
                wall = t.total_wallclock_time()
            );
        };
        s.push_str("\nTiming Statistics:\n");
        row(&mut s, "OverallAlgorithm....................:", &self.overall_alg);
        row(&mut s, " InitializeIterates.................:", &self.initialize_iterates);
        row(&mut s, " UpdateHessian......................:", &self.update_hessian);
        row(&mut s, " OutputIteration....................:", &self.output_iteration);
        row(&mut s, " UpdateBarrierParameter.............:", &self.update_barrier_parameter);
        row(&mut s, " ComputeSearchDirection.............:", &self.compute_search_direction);
        row(&mut s, " ComputeAcceptableTrialPoint........:", &self.compute_acceptable_trial_point);
        row(&mut s, " AcceptTrialPoint...................:", &self.accept_trial_point);
        row(&mut s, " CheckConvergence...................:", &self.check_convergence);
        row(&mut s, "LinearSystemFactorization...........:", &self.linear_system_factorization);
        row(&mut s, "LinearSystemBackSolve...............:", &self.linear_system_back_solve);
        row(&mut s, "QualityFunctionSearch...............:", &self.quality_function_search);
        row(&mut s, "TotalFunctionEvaluations............:", &self.total_function_evaluation_time);
        row(&mut s, " ObjectiveFunctionEvaluations.......:", &self.eval_obj);
        row(&mut s, " ObjectiveGradientEvaluations.......:", &self.eval_grad_obj);
        row(&mut s, " ConstraintEvaluations..............:", &self.eval_constr);
        row(&mut s, " ConstraintJacobianEvaluations......:", &self.eval_constr_jac);
        row(&mut s, " LagrangianHessianEvaluations.......:", &self.eval_lag_hess);
        s
    }

    /// Reset all counters. Mirrors upstream `ResetTimes()`.
    pub fn reset(&self) {
        self.overall_alg.reset();
        self.print_problem_statistics.reset();
        self.initialize_iterates.reset();
        self.update_hessian.reset();
        self.output_iteration.reset();
        self.update_barrier_parameter.reset();
        self.compute_search_direction.reset();
        self.compute_acceptable_trial_point.reset();
        self.accept_trial_point.reset();
        self.check_convergence.reset();
        self.linear_system_factorization.reset();
        self.linear_system_back_solve.reset();
        self.linear_system_structure_converter.reset();
        self.linear_system_structure_converter_init.reset();
        self.quality_function_search.reset();
        self.total_callback_time.reset();
        self.total_function_evaluation_time.reset();
        self.eval_obj.reset();
        self.eval_grad_obj.reset();
        self.eval_constr.reset();
        self.eval_constr_jac.reset();
        self.eval_lag_hess.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_end_accumulates_nonneg() {
        let t = TimedTask::new();
        t.start();
        for _ in 0..1000 { std::hint::black_box(0u64); }
        t.end();
        assert!(t.total_wallclock_time() >= 0.0);
    }

    #[test]
    fn disabled_is_noop() {
        let t = TimedTask::new();
        t.disable();
        t.start();
        t.end();
        assert_eq!(t.total_wallclock_time(), 0.0);
    }

    #[test]
    fn end_if_started_handles_unstarted() {
        let t = TimedTask::new();
        t.end_if_started();
        assert_eq!(t.total_wallclock_time(), 0.0);
    }
}
