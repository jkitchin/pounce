//! Post-optimal sensitivity across a solve whose answer was **replayed from a
//! floor** describes the attempt that was reported, not the one that lost.
//!
//! `on_converged` fires once per converged attempt. When a retry converges but
//! is declined -- the `mu_strategy_fallback` re-solve here, the gh#884
//! dual-divergence retry and the gh#983 scale audit are the same shape -- the
//! application replays the first attempt's `finalize_solution` payload, but the
//! hook has already run for the retry. Before this was fixed every consumer that
//! kept "the last firing" (`Solver`, `SensSolve`, the CLI) handed back the
//! losing retry's point, multipliers and factor alongside the winner's status:
//! a parametric step taken about a point the solver had just decided not to
//! report.
//!
//! The trigger is deterministic. `tol` is set below what any attempt can reach,
//! so every attempt exits at `Solved_To_Acceptable_Level`; an explicit
//! `mu_strategy_fallback=yes` makes that status retry-worthy, the flipped
//! schedule converges to acceptable as well, and a retry promotes only on
//! `Solve_Succeeded`, so the first attempt's answer is floored back. The
//! reference is the same solve with `mu_strategy_fallback=no`, which never
//! spends the retry and reports that same first attempt.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::{SensSolve, Solver};

const N: usize = 3;
const P0: Number = 0.7;

/// `min Σᵢ sᵢ(xᵢ − 1)⁴ + ½ sᵢ(xᵢ − 1)² + p·x₀·x₁` over `0.2 ≤ xᵢ ≤ 5`, with
/// the parameter `p = x₂` pinned by row 0. Curvature spanning orders of
/// magnitude and live bounds make the two barrier schedules walk visibly
/// different paths, so the two acceptable points differ.
#[derive(Default)]
struct Recording {
    /// Every `finalize_solution` payload's `x`, in order.
    payloads: Vec<Vec<Number>>,
}

fn scale(i: usize) -> Number {
    10f64.powi(2 * i as i32 - 2)
}

impl TNLP for Recording {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as Index,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..2 {
            b.x_l[i] = 0.2;
            b.x_u[i] = 5.0;
        }
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = P0;
        b.g_u[0] = P0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[4.0, 3.5, P0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = x[2] * x[0] * x[1];
        for i in 0..2 {
            let s = scale(i);
            f += s * (x[i] - 1.0).powi(4) + 0.5 * s * (x[i] - 1.0).powi(2);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..2 {
            let s = scale(i);
            g[i] = 4.0 * s * (x[i] - 1.0).powi(3) + s * (x[i] - 1.0);
        }
        g[0] += x[2] * x[1];
        g[1] += x[2] * x[0];
        g[2] = x[0] * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // Lower triangle: (0,0), (1,1), (1,0), (2,0), (2,1).
        const RC: [(Index, Index); 5] = [(0, 0), (1, 1), (1, 0), (2, 0), (2, 1)];
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in RC.iter().enumerate() {
                    irow[k] = r;
                    jcol[k] = c;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let d = |i: usize| scale(i) * (12.0 * (x[i] - 1.0).powi(2) + 1.0);
                values[0] = obj_factor * d(0);
                values[1] = obj_factor * d(1);
                values[2] = obj_factor * x[2];
                values[3] = obj_factor * x[1];
                values[4] = obj_factor * x[0];
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.payloads.push(sol.x.to_vec());
    }
}

fn app(fallback: bool) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("sb", "yes", true, false).unwrap();
    o.set_numeric_value("tol", 1e-30, true, false).unwrap();
    o.set_numeric_value("acceptable_tol", 1e-2, true, false)
        .unwrap();
    o.set_integer_value("acceptable_iter", 2, true, false)
        .unwrap();
    o.set_string_value(
        "mu_strategy_fallback",
        if fallback { "yes" } else { "no" },
        true,
        false,
    )
    .unwrap();
    app.initialize().unwrap();
    app
}

struct Run {
    status: ApplicationReturnStatus,
    payloads: Vec<Vec<Number>>,
    held_x: Vec<Number>,
    dx: Vec<Number>,
    restored: bool,
}

fn run(fallback: bool) -> Run {
    let concrete = Rc::new(RefCell::new(Recording::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let mut solver = Solver::new(app(fallback), tnlp);
    let status = solver.solve();
    let restored = solver.app().answer_restored_from_floor();
    let held_x = solver
        .converged()
        .expect("a converged state is held after an acceptable exit")
        .x
        .clone();
    let dx = solver
        .parametric_step(&[0], &[1e-2])
        .expect("parametric_step ok");
    let payloads = concrete.borrow().payloads.clone();
    Run {
        status,
        payloads,
        held_x,
        dx,
        restored,
    }
}

fn max_diff(a: &[Number], b: &[Number]) -> Number {
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, Number::max)
}

/// The premise: the retry ran, converged to a *different* point, and was
/// floored back. Without all three the other tests prove nothing.
#[test]
fn premise_the_retry_converges_elsewhere_and_is_floored_back() {
    let r = run(true);
    assert_eq!(r.status, ApplicationReturnStatus::SolvedToAcceptableLevel);
    assert!(r.restored, "the retry was promoted or never ran");
    assert!(
        r.payloads.len() >= 3,
        "expected base, retry and replay payloads, got {}",
        r.payloads.len()
    );
    let (base, retry) = (&r.payloads[0], &r.payloads[1]);
    assert!(
        max_diff(base, retry) > 1e-7,
        "the two attempts reached the same point ({base:?} vs {retry:?}), so a \
         stale capture would be invisible"
    );
    assert_eq!(r.payloads.last().unwrap(), base, "the replay is the base");
}

#[test]
fn the_held_point_is_the_reported_point() {
    let r = run(true);
    let reported = r.payloads.last().unwrap();
    assert!(
        max_diff(&r.held_x, reported) <= 1e-12,
        "Solver holds {:?} but the solve reported {reported:?} -- the losing \
         retry's state survived the floor replay",
        r.held_x
    );
}

#[test]
fn the_step_across_a_floored_solve_matches_a_solve_that_never_retried() {
    let floored = run(true);
    let reference = run(false);
    assert!(!reference.restored);
    assert_eq!(
        reference.payloads.len(),
        1,
        "mu_strategy_fallback=no must not retry"
    );
    assert!(
        max_diff(floored.payloads.last().unwrap(), &reference.payloads[0]) <= 1e-12,
        "premise: both runs report the first attempt's point"
    );
    assert!(
        max_diff(&floored.dx, &reference.dx) <= 1e-12,
        "parametric step after the floor {:?} differs from the never-retried \
         run's {:?}: it was taken about the losing retry's point",
        floored.dx,
        reference.dx
    );
}

/// `SensSolve` keeps its own capture, separately from `Solver`.
#[test]
fn sens_solve_reports_the_floored_attempts_step() {
    let go = |fallback: bool| {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Recording::default()));
        let mut a = app(fallback);
        SensSolve::new(vec![0])
            .with_deltas(vec![1e-2])
            .run(&mut a, tnlp)
    };
    let floored = go(true);
    let reference = go(false);
    let (fx, rx) = (floored.x.expect("x"), reference.x.expect("x"));
    let (fd, rd) = (floored.dx.expect("dx"), reference.dx.expect("dx"));
    assert!(max_diff(&fx, &rx) <= 1e-12, "x {fx:?} vs {rx:?}");
    assert!(max_diff(&fd, &rd) <= 1e-12, "dx {fd:?} vs {rd:?}");
}
