//! gh#1176: `max_wall_time` / `max_cpu_time` bound the whole run, second-
//! opinion ladder included.
//!
//! Every solve builds its own deadline from those options, so before this
//! each ladder rung -- and the `mu_strategy_fallback` re-solve inside each --
//! got a fresh full allowance. On the issue's 2594-variable distillation model
//! `max_wall_time=40` ran 112 s: 26 s base, then 21 + 40 + 0 + 25 s of rungs.
//!
//! The rule branches on whether the caller set a limit and, if so, on whether
//! any of it is left when a rung is about to start; one test per branch.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_restoration::second_opinion_driver::{SecondOpinionOutcome, run_second_opinion_ladder};

/// `min (x0-1)² + (x1-1)²  s.t.  x0² + x1² = -1`: infeasible everywhere, so
/// the base solve fails (`Restoration_Failed` today) and the ladder opens. `delay` makes every constraint evaluation sleep, so a solve
/// costs measurable wall time.
struct Infeasible {
    delay: Duration,
}

impl TNLP for Infeasible {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-1e19);
        b.x_u.fill(1e19);
        b.g_l[0] = -1.0;
        b.g_u[0] = -1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.7]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        std::thread::sleep(self.delay);
        g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values need x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let l = lambda.map_or(0.0, |l| l[0]);
                values.fill(2.0 * obj + 2.0 * l);
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

fn quiet_app(limits: &[(&str, Number)]) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    for (k, v) in limits {
        app.options_mut()
            .set_numeric_value(k, *v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

/// Base solve, then `pause`, then the ladder. Returns the outcome, the
/// ladder's console lines and the wall time from the base solve's start.
fn run(
    app: &mut IpoptApplication,
    delay: Duration,
    pause: Duration,
) -> (SecondOpinionOutcome, Vec<String>, f64) {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Infeasible { delay }));
    let t0 = Instant::now();
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::InfeasibleProblemDetected
                | ApplicationReturnStatus::RestorationFailed
        ),
        "the fixture no longer fails the way that opens the ladder: {status:?}"
    );
    let stats = app.statistics();
    std::thread::sleep(pause);
    let mut lines = Vec::new();
    let outcome =
        run_second_opinion_ladder(app, tnlp, status, stats, &mut |l| lines.push(l.to_string()));
    (outcome, lines, t0.elapsed().as_secs_f64())
}

fn caller_set(app: &IpoptApplication, name: &str) -> Option<Number> {
    app.options()
        .get_numeric_value(name, "")
        .ok()
        .and_then(|(v, found)| found.then_some(v))
}

#[test]
fn with_no_limit_set_every_rung_runs_and_the_limits_stay_unset() {
    let mut app = quiet_app(&[]);
    let (outcome, _, _) = run(&mut app, Duration::ZERO, Duration::ZERO);
    assert!(outcome.ran());
    assert!(
        outcome.skipped_for_time.is_empty(),
        "{:?}",
        outcome.skipped_for_time
    );
    assert_eq!(outcome.rung_wallclock_secs.len(), outcome.tried.len());
    // Budgeting a limit the caller never set would write the default back
    // as caller-set, which code that treats an explicit `max_cpu_time`
    // specially would then misread.
    assert_eq!(caller_set(&app, "max_wall_time"), None);
    assert_eq!(caller_set(&app, "max_cpu_time"), None);
}

#[test]
fn a_spent_limit_runs_no_rung_and_says_so() {
    // The base solve takes milliseconds; the pause spends the rest.
    let mut app = quiet_app(&[("max_wall_time", 0.3)]);
    let (outcome, lines, _) = run(&mut app, Duration::ZERO, Duration::from_millis(400));
    assert!(outcome.tried.is_empty(), "ran {:?}", outcome.tried);
    assert!(!outcome.skipped_for_time.is_empty());
    assert_eq!(
        outcome.status, outcome.base_status,
        "nothing ran, nothing changed"
    );
    assert!(
        lines.iter().any(|l| l.contains("time limit is spent")),
        "{lines:#?}"
    );
    // Put back exactly as the caller wrote it.
    assert_eq!(caller_set(&app, "max_wall_time"), Some(0.3));
}

#[test]
fn rungs_spend_what_is_left_of_the_run_not_a_fresh_allowance() {
    // Each constraint evaluation sleeps, so a solve takes over a second. The
    // base solve runs with no limit at all, and the limit is set only once it
    // returns: `base` seconds already spent plus `LEFT` for the ladder. So the
    // base solve's own duration cannot make the test miss -- the limit is
    // sized off *this* run, not a separate measurement of a similar one -- and
    // the run clock, which started at the base solve, is what charges the
    // rungs. Charged, the run ends near the limit; each rung on a fresh
    // allowance -- the defect -- ran a whole `~base`-second solve past it.
    //
    // The sleep is calibrated so `base` is large against `LEFT`: `sleep(2 ms)`
    // overshoots by a platform-dependent amount (a base solve measured 2.6 s
    // on macOS and 0.95 s on a Linux runner).
    const LEFT: f64 = 0.3;
    let measure = |delay: Duration| {
        let mut probe = quiet_app(&[]);
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Infeasible { delay }));
        let t0 = Instant::now();
        probe.optimize_tnlp(t);
        t0.elapsed().as_secs_f64()
    };
    let mut delay = Duration::from_millis(2);
    let probe = measure(delay);
    if probe < 1.2 {
        delay = delay.mul_f64((1.5 / probe.max(0.05)).min(20.0));
    }

    let mut app = quiet_app(&[]);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Infeasible { delay }));
    let t0 = Instant::now();
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    let base = t0.elapsed().as_secs_f64();
    assert!(
        base > 4.0 * LEFT,
        "the fixture no longer takes the time this test is sized for: {base:.3} s \
         at a {delay:?} sleep per evaluation"
    );
    let limit = base + LEFT;
    app.options_mut()
        .set_numeric_value("max_wall_time", limit, true, false)
        .unwrap();
    let stats = app.statistics();
    let outcome = run_second_opinion_ladder(&mut app, tnlp, status, stats, &mut |_| {});
    let elapsed = t0.elapsed().as_secs_f64();
    assert!(outcome.ran(), "no rung ran; {LEFT} s should have been left");
    // The solver checks its deadline between steps, so a rung overshoots the
    // limit by a step's worth; an uncharged rung overshoots by a whole solve.
    assert!(
        elapsed < limit + 0.5 * base,
        "the run took {elapsed:.3} s against max_wall_time={limit:.3} \
         (rungs {:?} took {:?} s)",
        outcome.tried,
        outcome.rung_wallclock_secs,
    );
    let total = outcome.total_wallclock_secs();
    assert!(
        (total - elapsed).abs() < 0.25_f64.max(0.1 * elapsed),
        "reported total {total:.3} s, measured {elapsed:.3} s"
    );
}

#[test]
fn every_solve_measures_its_limit_from_the_run_clock() {
    // gh#1176 review item 1: the interior point's deadline starts at the run
    // clock, not at "now", so a solve the application makes below
    // `optimize_tnlp` -- the mu fallback, the gh#884 retry, the gh#983 audit,
    // none of which resets the clock -- is charged to the run. Driven here
    // through the public hold, the one way to put a solve inside a run that
    // started earlier. Both branches: a run with time left solves as usual,
    // a run whose limit is spent stops at once.
    let run_in = |already_spent: f64| {
        let mut app = quiet_app(&[("max_wall_time", 5.0)]);
        let now = (
            pounce_common::utils::wallclock_time(),
            pounce_common::utils::cpu_time(),
        );
        app.hold_run_clock(true);
        app.set_run_clock((now.0 - already_spent, now.1));
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SlowRosenbrock {
            delay: Duration::ZERO,
            pause_at: None,
        }));
        let status = app.optimize_tnlp(t);
        app.hold_run_clock(false);
        (status, app.statistics().iteration_count)
    };
    let (fresh, _) = run_in(0.0);
    assert_eq!(fresh, ApplicationReturnStatus::SolveSucceeded);
    let (spent, iters) = run_in(10.0);
    assert_eq!(spent, ApplicationReturnStatus::MaximumWallTimeExceeded);
    assert!(iters <= 1, "a spent run still iterated {iters} times");
}

/// Unconstrained Rosenbrock with a sleeping objective: a run capped by
/// `max_iter` ends `Maximum_Iterations_Exceeded`, which is what opens the
/// `mu_strategy_fallback` re-solve.
struct SlowRosenbrock {
    delay: Duration,
    /// Sleep this long, once, in the intermediate callback at this iteration.
    pause_at: Option<(Index, Duration)>,
}

impl TNLP for SlowRosenbrock {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-1e19);
        b.x_u.fill(1e19);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-1.2, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        std::thread::sleep(self.delay);
        Some(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values need x");
                values[0] = obj * (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0);
                values[1] = obj * (-400.0 * x[0]);
                values[2] = obj * 200.0;
            }
        }
        true
    }
    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        _ip_data: &IpoptData,
        _ip_cq: &IpoptCq,
    ) -> bool {
        if let Some((at, pause)) = self.pause_at
            && stats.iter == at
        {
            self.pause_at = None;
            std::thread::sleep(pause);
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

#[test]
fn a_mu_strategy_fallback_skipped_for_time_is_recorded() {
    // gh#1176 review item 4. The first attempt stops at its iteration cap, so
    // the fallback opens; a pause in the callback after its last iteration spends the limit, so
    // the re-solve is skipped. The record says so instead of reading as a
    // fallback that never opened.
    let mut app = quiet_app(&[("max_wall_time", 0.3)]);
    app.options_mut()
        .set_integer_value("max_iter", 12, true, false)
        .unwrap();
    let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SlowRosenbrock {
        delay: Duration::ZERO,
        pause_at: Some((12, Duration::from_millis(400))),
    }));
    let status = app.optimize_tnlp(t);
    assert_eq!(status, ApplicationReturnStatus::MaximumIterationsExceeded);
    let record = app
        .statistics()
        .mu_strategy_fallback
        .expect("a skipped fallback leaves no record");
    assert!(record.skipped_for_time, "{record:?}");
    assert_eq!(record.retry_iteration_count, 0, "{record:?}");
}

#[test]
fn the_mu_strategy_fallback_re_solve_spends_what_is_left_of_the_run() {
    // Sized off a measured capped run: the limit leaves the re-solve ~0.2 s
    // of the ~`first` seconds it would take uncharged.
    let delay = Duration::from_millis(40);
    let opts = [("max_iter", 12.0)];
    let solve = |limits: &[(&str, Number)]| {
        let mut app = quiet_app(&[]);
        for (k, v) in limits {
            if *k == "max_iter" {
                app.options_mut()
                    .set_integer_value(k, *v as Index, true, false)
                    .unwrap();
            } else {
                app.options_mut()
                    .set_numeric_value(k, *v, true, false)
                    .unwrap();
            }
        }
        let t: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SlowRosenbrock {
            delay,
            pause_at: None,
        }));
        let t0 = Instant::now();
        let status = app.optimize_tnlp(t);
        (status, app.statistics(), t0.elapsed().as_secs_f64())
    };
    let (status, stats, both) = solve(&opts);
    assert_eq!(status, ApplicationReturnStatus::MaximumIterationsExceeded);
    let record = stats
        .mu_strategy_fallback
        .expect("the fixture no longer opens the mu_strategy fallback");
    assert_eq!(record.retry_iteration_count, 12, "{record:?}");
    // Both attempts ran to the cap; the first is about half.
    let first = both / 2.0;
    assert!(
        first > 0.4,
        "the fixture is too fast to size against: {first:.3} s"
    );

    let limit = first + 0.2;
    let (_, stats, elapsed) = solve(&[opts[0], ("max_wall_time", limit)]);
    if let Some(r) = &stats.mu_strategy_fallback {
        assert!(
            r.retry_iteration_count < 12,
            "the re-solve ran its full cap: {r:?}"
        );
    }
    assert!(
        elapsed < limit + 0.5 * first,
        "the run took {elapsed:.3} s against max_wall_time={limit:.3}; \
         uncharged it takes {both:.3} s"
    );
}
