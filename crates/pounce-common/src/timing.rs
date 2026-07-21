//! Per-task timing accumulator.
//!
//! Mirrors `Common/IpTimedTask.hpp` (`Common/IpDebug.{hpp,cpp}` is
//! omitted — debug tracing is replaced by the journalist).

use crate::types::Number;
use crate::utils::{cpu_time, sys_time, wallclock_time};
use std::cell::Cell;
use std::rc::Rc;

/// Which time budget a [`Deadline`] check found crossed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineKind {
    /// Wall-clock budget (`max_wall_time`) exceeded.
    Wall,
    /// CPU-time budget (`max_cpu_time`) exceeded.
    Cpu,
}

/// A monotonic wall/CPU-clock deadline for a single solve (pounce#242).
///
/// Cheaply clonable (`Rc`-backed) so the outer loop, the KKT solver, the
/// line search, and the *restoration inner IPM* can all check the same
/// global budget — not just the outer-iteration convergence check. The
/// motivating bug: `max_wall_time` was only tested between outer
/// iterations (in `OptErrorConvCheck`), so a solve whose per-iteration
/// cost is dominated by a single expensive step — a slow KKT
/// factorization, or a restoration sub-solve that runs an entire nested
/// IPM under one outer "iteration" — overshot the requested budget by up
/// to a full iteration (~7x on the reported 1611-variable NLP). Checking
/// this deadline at the granularity of the expensive inner steps bounds
/// the overshoot to roughly one such step.
///
/// The elapsed time is measured from the instant the `Deadline` is
/// constructed, using the same process clocks the timing subsystem uses.
/// Unlike [`TimedTask::live_wallclock_time`] it does **not** depend on a
/// `start()`/`end()` cycle, so it works inside the nested restoration
/// solve — whose fresh [`TimingStatistics`] has an `overall_alg` timer
/// that is never started, which is exactly why the inner loop used to run
/// unbounded by wall time.
#[derive(Debug, Clone)]
pub struct Deadline {
    inner: Rc<DeadlineInner>,
}

#[derive(Debug)]
struct DeadlineInner {
    wall_start: Number,
    cpu_start: Number,
    max_wall: Number,
    max_cpu: Number,
}

impl Deadline {
    /// Create a deadline that fires once `max_wall` wall seconds or
    /// `max_cpu` CPU seconds have elapsed from *now*. The pounce defaults
    /// for both budgets are `1e6`, i.e. effectively unbounded; a caller
    /// that passes those gets a deadline that never trips in practice.
    pub fn new(max_wall: Number, max_cpu: Number) -> Self {
        Self {
            inner: Rc::new(DeadlineInner {
                wall_start: wallclock_time(),
                cpu_start: cpu_time(),
                max_wall,
                max_cpu,
            }),
        }
    }

    /// Return `Some(kind)` if either budget has been crossed, else
    /// `None`. CPU is tested before wall to match the branch order of
    /// upstream `OptimalityErrorConvergenceCheck::CheckConvergence` (and
    /// pounce's `OptErrorConvCheck`), so a solve that trips both in the
    /// same check reports `MaximumCpuTimeExceeded` identically to the
    /// coarse path.
    pub fn exceeded(&self) -> Option<DeadlineKind> {
        if cpu_time() - self.inner.cpu_start >= self.inner.max_cpu {
            return Some(DeadlineKind::Cpu);
        }
        if wallclock_time() - self.inner.wall_start >= self.inner.max_wall {
            return Some(DeadlineKind::Wall);
        }
        None
    }

    /// The wall-clock budget this deadline was built with.
    pub fn max_wall(&self) -> Number {
        self.inner.max_wall
    }

    /// The CPU-time budget this deadline was built with.
    pub fn max_cpu(&self) -> Number {
        self.inner.max_cpu
    }
}

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enable(&self) {
        self.enabled.set(true);
    }
    pub fn disable(&self) {
        self.enabled.set(false);
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled.get()
    }
    pub fn is_started(&self) -> bool {
        self.start_called.get()
    }

    pub fn reset(&self) {
        self.total_cpu.set(0.0);
        self.total_sys.set(0.0);
        self.total_wall.set(0.0);
        self.start_called.set(false);
        self.end_called.set(true);
    }

    pub fn start(&self) {
        if !self.enabled.get() {
            return;
        }
        self.end_called.set(false);
        self.start_called.set(true);
        self.start_cpu.set(cpu_time());
        self.start_sys.set(sys_time());
        self.start_wall.set(wallclock_time());
    }

    pub fn end(&self) {
        if !self.enabled.get() {
            return;
        }
        self.end_called.set(true);
        self.start_called.set(false);
        self.total_cpu
            .set(self.total_cpu.get() + cpu_time() - self.start_cpu.get());
        self.total_sys
            .set(self.total_sys.get() + sys_time() - self.start_sys.get());
        self.total_wall
            .set(self.total_wall.get() + wallclock_time() - self.start_wall.get());
    }

    pub fn end_if_started(&self) {
        if !self.enabled.get() {
            return;
        }
        if self.start_called.get() {
            self.end();
        }
    }

    pub fn total_cpu_time(&self) -> Number {
        self.total_cpu.get()
    }
    pub fn total_sys_time(&self) -> Number {
        self.total_sys.get()
    }
    pub fn total_wallclock_time(&self) -> Number {
        self.total_wall.get()
    }

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
        row(
            &mut s,
            "OverallAlgorithm....................:",
            &self.overall_alg,
        );
        row(
            &mut s,
            " InitializeIterates.................:",
            &self.initialize_iterates,
        );
        row(
            &mut s,
            " UpdateHessian......................:",
            &self.update_hessian,
        );
        row(
            &mut s,
            " OutputIteration....................:",
            &self.output_iteration,
        );
        row(
            &mut s,
            " UpdateBarrierParameter.............:",
            &self.update_barrier_parameter,
        );
        row(
            &mut s,
            " ComputeSearchDirection.............:",
            &self.compute_search_direction,
        );
        row(
            &mut s,
            " ComputeAcceptableTrialPoint........:",
            &self.compute_acceptable_trial_point,
        );
        row(
            &mut s,
            " AcceptTrialPoint...................:",
            &self.accept_trial_point,
        );
        row(
            &mut s,
            " CheckConvergence...................:",
            &self.check_convergence,
        );
        row(
            &mut s,
            "LinearSystemFactorization...........:",
            &self.linear_system_factorization,
        );
        row(
            &mut s,
            "LinearSystemBackSolve...............:",
            &self.linear_system_back_solve,
        );
        row(
            &mut s,
            "QualityFunctionSearch...............:",
            &self.quality_function_search,
        );
        row(
            &mut s,
            "TotalFunctionEvaluations............:",
            &self.total_function_evaluation_time,
        );
        row(
            &mut s,
            " ObjectiveFunctionEvaluations.......:",
            &self.eval_obj,
        );
        row(
            &mut s,
            " ObjectiveGradientEvaluations.......:",
            &self.eval_grad_obj,
        );
        row(
            &mut s,
            " ConstraintEvaluations..............:",
            &self.eval_constr,
        );
        row(
            &mut s,
            " ConstraintJacobianEvaluations......:",
            &self.eval_constr_jac,
        );
        row(
            &mut s,
            " LagrangianHessianEvaluations.......:",
            &self.eval_lag_hess,
        );
        s
    }

    /// Structured wall-clock breakdown (seconds) of the major solve
    /// subsystems, as ordered `(label, seconds)` pairs. Same numbers
    /// [`Self::report`] prints, but as data rather than formatted text,
    /// so a programmatic consumer (e.g. the Python `Problem.solve` `info`
    /// dict) can attribute a solve's runtime without scraping the report
    /// or patching the solver.
    ///
    /// Ordered coarse→fine: the overall algorithm total; the
    /// linear-algebra split (`linear_system_total` = factorization +
    /// back-solve, with factorization broken out); and the per-callback
    /// function-evaluation split (objective / gradient / constraints /
    /// Jacobian / Lagrangian Hessian). This is exactly the func /
    /// Jacobian / Hessian time split issue #180 needs to reproduce a
    /// Table-6-style "where did the time go" analysis for a
    /// reduced-space / variable-aggregation solve.
    pub fn wall_time_breakdown(&self) -> Vec<(&'static str, Number)> {
        let factorization = self.linear_system_factorization.total_wallclock_time();
        let back_solve = self.linear_system_back_solve.total_wallclock_time();
        vec![
            ("overall_alg", self.overall_alg.total_wallclock_time()),
            ("update_hessian", self.update_hessian.total_wallclock_time()),
            (
                "compute_search_direction",
                self.compute_search_direction.total_wallclock_time(),
            ),
            ("linear_system_total", factorization + back_solve),
            ("linear_system_factorization", factorization),
            ("linear_system_back_solve", back_solve),
            (
                "function_evaluations_total",
                self.total_function_evaluation_time.total_wallclock_time(),
            ),
            ("eval_objective", self.eval_obj.total_wallclock_time()),
            ("eval_gradient", self.eval_grad_obj.total_wallclock_time()),
            ("eval_constraints", self.eval_constr.total_wallclock_time()),
            (
                "eval_constraint_jacobian",
                self.eval_constr_jac.total_wallclock_time(),
            ),
            (
                "eval_lagrangian_hessian",
                self.eval_lag_hess.total_wallclock_time(),
            ),
            (
                "total_callback",
                self.total_callback_time.total_wallclock_time(),
            ),
        ]
    }

    /// Enable or disable the *detailed* per-subsystem timers, mirroring
    /// upstream Ipopt's `timing_statistics` gating (`IpoptApplication`
    /// only measures the detailed function/phase timers when
    /// `timing_statistics=yes`). When `on` is `false` every `start()` /
    /// `end()` on these tasks becomes a no-op, so a fast-objective solve
    /// stops paying two `getrusage` syscalls per timed section (issue
    /// #190).
    ///
    /// [`Self::overall_alg`] is deliberately left untouched: its
    /// `live_cpu_time()` feeds the `max_cpu_time` convergence check and
    /// its total is reported regardless of the option — upstream's help
    /// text is explicit that "the overall algorithm time is unaffected by
    /// this option". Callers that need the detailed
    /// [`Self::wall_time_breakdown`] populated (the Python `info["timing"]`
    /// dict, the CLI `timing.json`) must therefore set `timing_statistics`
    /// (or `print_timing_statistics`, which implies it) to `yes`.
    pub fn set_detailed_enabled(&self, on: bool) {
        let set = |t: &TimedTask| {
            if on {
                t.enable();
            } else {
                t.disable();
            }
        };
        // Every field except `overall_alg`.
        set(&self.print_problem_statistics);
        set(&self.initialize_iterates);
        set(&self.update_hessian);
        set(&self.output_iteration);
        set(&self.update_barrier_parameter);
        set(&self.compute_search_direction);
        set(&self.compute_acceptable_trial_point);
        set(&self.accept_trial_point);
        set(&self.check_convergence);
        set(&self.linear_system_factorization);
        set(&self.linear_system_back_solve);
        set(&self.linear_system_structure_converter);
        set(&self.linear_system_structure_converter_init);
        set(&self.quality_function_search);
        set(&self.total_callback_time);
        set(&self.total_function_evaluation_time);
        set(&self.eval_obj);
        set(&self.eval_grad_obj);
        set(&self.eval_constr);
        set(&self.eval_constr_jac);
        set(&self.eval_lag_hess);
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
    fn deadline_unbounded_never_trips() {
        // The pounce "no budget" defaults (1e6 seconds each) must never
        // fire in any realistic test runtime.
        let d = Deadline::new(1e6, 1e6);
        assert!(d.exceeded().is_none());
        assert_eq!(d.max_wall(), 1e6);
        assert_eq!(d.max_cpu(), 1e6);
    }

    #[test]
    fn deadline_zero_wall_trips_wall() {
        // Zero wall budget, effectively-unbounded CPU budget: after any
        // wall time elapses the check reports `Wall` (CPU is tested first
        // but its budget is not crossed).
        let d = Deadline::new(0.0, 1e6);
        // Busy-spin until the monotonic wall clock has advanced past the
        // start instant, so the assertion is not racing a zero-duration
        // `elapsed()` on a coarse clock.
        for _ in 0..10_000 {
            if d.exceeded().is_some() {
                break;
            }
            std::hint::black_box(0u64);
        }
        assert_eq!(d.exceeded(), Some(DeadlineKind::Wall));
    }

    #[test]
    fn deadline_zero_cpu_takes_priority() {
        // Both budgets at zero: CPU is checked first, so a solve that
        // crosses both in the same check reports `Cpu` — matching the
        // convergence check's branch order.
        let d = Deadline::new(0.0, 0.0);
        for _ in 0..10_000 {
            if d.exceeded().is_some() {
                break;
            }
            std::hint::black_box(0u64);
        }
        assert_eq!(d.exceeded(), Some(DeadlineKind::Cpu));
    }

    #[test]
    fn deadline_is_cheaply_clonable_and_shares_start() {
        // Cloning shares the same start instant / budgets (the restoration
        // inner IPM relies on this to be bounded by the outer solve's
        // elapsed time, not its own).
        let d = Deadline::new(1e6, 1e6);
        let d2 = d.clone();
        assert_eq!(d2.max_wall(), d.max_wall());
        assert_eq!(d2.max_cpu(), d.max_cpu());
        assert!(d2.exceeded().is_none());
    }

    #[test]
    fn start_end_accumulates_nonneg() {
        let t = TimedTask::new();
        t.start();
        for _ in 0..1000 {
            std::hint::black_box(0u64);
        }
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
    fn set_detailed_enabled_gates_all_but_overall_alg() {
        let stats = TimingStatistics::new();
        // Default: every timer enabled.
        assert!(stats.overall_alg.is_enabled());
        assert!(stats.eval_obj.is_enabled());
        assert!(stats.check_convergence.is_enabled());

        // Disabling the detail timers (issue #190: `timing_statistics=no`)
        // must leave `overall_alg` alive — it feeds the `max_cpu_time`
        // check and is always reported — while every other timer becomes
        // a no-op that skips the `getrusage` syscalls.
        stats.set_detailed_enabled(false);
        assert!(stats.overall_alg.is_enabled(), "overall_alg must stay live");
        assert!(!stats.eval_obj.is_enabled());
        assert!(!stats.check_convergence.is_enabled());
        assert!(!stats.total_function_evaluation_time.is_enabled());
        assert!(!stats.linear_system_factorization.is_enabled());

        // A disabled detail timer accumulates nothing even across start/end.
        stats.eval_obj.start();
        stats.eval_obj.end();
        assert_eq!(stats.eval_obj.total_wallclock_time(), 0.0);

        // Re-enabling restores them.
        stats.set_detailed_enabled(true);
        assert!(stats.eval_obj.is_enabled());
        assert!(stats.check_convergence.is_enabled());
    }

    #[test]
    fn end_if_started_handles_unstarted() {
        let t = TimedTask::new();
        t.end_if_started();
        assert_eq!(t.total_wallclock_time(), 0.0);
    }

    #[test]
    fn wall_time_breakdown_reports_subsystems() {
        let stats = TimingStatistics::new();
        // Accumulate into two distinct subsystems so the breakdown is
        // not trivially all-zero and the linear-algebra total is the
        // sum of its two parts.
        stats.linear_system_factorization.start();
        stats.linear_system_factorization.end();
        stats.eval_lag_hess.start();
        stats.eval_lag_hess.end();

        let bd = stats.wall_time_breakdown();
        let get = |k: &str| bd.iter().find(|(label, _)| *label == k).map(|(_, v)| *v);

        // Every advertised key is present and non-negative.
        for key in [
            "overall_alg",
            "linear_system_total",
            "linear_system_factorization",
            "linear_system_back_solve",
            "function_evaluations_total",
            "eval_objective",
            "eval_gradient",
            "eval_constraints",
            "eval_constraint_jacobian",
            "eval_lagrangian_hessian",
        ] {
            assert!(get(key).is_some(), "missing breakdown key {key}");
            assert!(get(key).unwrap() >= 0.0, "negative time for {key}");
        }

        // linear_system_total == factorization + back_solve, exactly.
        let total = get("linear_system_total").unwrap();
        let fact = get("linear_system_factorization").unwrap();
        let back = get("linear_system_back_solve").unwrap();
        assert_eq!(total, fact + back);
    }
}
