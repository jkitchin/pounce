//! The corrector's operator is assembled at the predicted point.
//!
//! A chord iteration converges at the rate the distance between its
//! operator and the true Jacobian sets. On a problem whose Hessian
//! changes along the step, an operator held at the base point buys a
//! contraction of `|1 - W*/W0|` per iteration, while one assembled at
//! the predicted point starts next to `W*` and contracts an order
//! faster. The fixture makes that gap decisive within the iteration
//! budget: `W = e^x` doubles over the step, the base-operator rate is
//! about one half, and the predicted-operator rate is under a tenth,
//! so only the predicted operator reaches the tolerance below.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::Solver;

mod common;
use common::compare::{Frame, assert_same_correction};
use common::outside_engine::{
    RecordingTnlp, Relaxation, correct_outside, correct_outside_holding, outside_engine,
};

/// ```text
/// min  exp(x) - p x
/// s.t. p = 1
/// ```
///
/// Stationarity gives `x*(p) = ln p`, so the base solution is `x = 0`
/// and the exact solution at `p = 1 + delta` is `ln(1 + delta)`. The
/// predictor is the tangent `dx = delta`, and everything the corrector
/// closes is genuine curvature: at `delta = 0.5` the predictor is off
/// by `0.5 - ln 1.5 = 9.45e-2`.
struct ExpCurvature {
    /// Exact-Hessian evaluations served, for the limited-memory test
    /// below: a `limited-memory` solve and its correction must never
    /// ask for one.
    h_evals: Rc<Cell<usize>>,
}

impl TNLP for ExpCurvature {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.2;
        sp.x[1] = 1.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0].exp() - x[1] * x[0])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0].exp() - x[1];
        g[1] = -x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[1];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 1;
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (x,x), (p,x)
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 0]);
            }
            SparsityRequest::Values { values } => {
                self.h_evals.set(self.h_evals.get() + 1);
                let xv = x.expect("hessian point")[0];
                values[0] = obj_factor * xv.exp();
                values[1] = -obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved_with(limited_memory: bool) -> (Solver, Rc<Cell<usize>>) {
    let app = app(limited_memory);
    let h_evals = Rc::new(Cell::new(0_usize));
    let tnlp = ExpCurvature {
        h_evals: Rc::clone(&h_evals),
    };
    let mut solver = Solver::new(app, Rc::new(RefCell::new(tnlp)));
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed (limited_memory={limited_memory}): {status:?}",
    );
    (solver, h_evals)
}

/// The options every solve of this fixture runs with.
fn app(limited_memory: bool) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    if limited_memory {
        app.options_mut()
            .set_string_value("hessian_approximation", "limited-memory", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

/// [`solved_with`] with the model wrapped to record what the solve returns.
fn solved_recorded(limited_memory: bool) -> (Solver, Rc<RefCell<RecordingTnlp<ExpCurvature>>>) {
    let model = Rc::new(RefCell::new(RecordingTnlp::new(ExpCurvature {
        h_evals: Rc::new(Cell::new(0_usize)),
    })));
    let tnlp: Rc<RefCell<dyn TNLP>> = model.clone();
    let mut solver = Solver::new(app(limited_memory), tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed (limited_memory={limited_memory}): {status:?}",
    );
    (solver, model)
}

fn solved() -> Solver {
    solved_with(false).0
}

/// Six iterations must close the curvature gap. The base-point
/// operator's contraction rate here is `|1 - e^{ln 1.5}| = 0.5`, which
/// leaves about `1.5e-3` of the predictor's `9.45e-2` error after six
/// iterations, three orders short of the tolerance. The predicted
/// point's operator contracts at under a tenth per iteration and
/// clears it.
#[test]
fn the_corrector_closes_curvature_the_base_operator_cannot() {
    let solver = solved();
    let base_x = solver.converged().expect("converged").x[0];
    assert!(base_x.abs() < 1e-8, "base x is ln 1 = 0, got {base_x}");

    let delta = 0.5;
    let exact = (1.0_f64 + delta).ln();

    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("full step");
    let predictor_err = (base_x + step[0] - exact).abs();
    assert!(
        (predictor_err - 9.45e-2).abs() < 1e-3,
        "the predictor's curvature error is the quantity under test, \
         got {predictor_err:e}",
    );

    let (refined, report) = solver
        .correct_step(&[0], &[delta], &step, 6)
        .expect("corrector");
    assert!(report.improved(), "the corrector must report improvement");
    let err = (base_x + refined[0] - exact).abs();
    assert!(
        err < 1e-6,
        "six iterations on the predicted point's operator close the \
         curvature gap: |corrected - ln(1+delta)| = {err:e}, predictor \
         error {predictor_err:e}",
    );
}

/// A `limited-memory` solve keeps its quasi-Newton matrix through the
/// correction: `ConvergedState` captures the option, the corrector
/// skips the exact-Hessian re-evaluation, and the exact Hessian is
/// never requested, which the fixture's own counter proves. The
/// correction still acts, against the L-BFGS matrix held while the
/// Jacobians and the barrier diagonal move to the predicted point: on
/// this fixture the quasi-Newton matrix sits near `e^0 = 1` against a
/// true `W* = 1.5`, so the contraction is near one half per iteration
/// rather than the exact-Hessian operator's tenth, and eight
/// iterations take the predictor's `9.45e-2` curvature error to
/// `4.0e-4`.
#[test]
fn a_limited_memory_solve_corrects_without_an_exact_hessian() {
    let (solver, h_evals) = solved_with(true);
    assert_eq!(
        h_evals.get(),
        0,
        "a limited-memory solve never asks for the exact Hessian",
    );
    let base_x = solver.converged().expect("converged").x[0];
    let delta = 0.5;
    let exact = (1.0_f64 + delta).ln();
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("full step");
    let predictor_err = (base_x + step[0] - exact).abs();
    let (refined, report) = solver
        .correct_step(&[0], &[delta], &step, 8)
        .expect("corrector");
    assert_eq!(
        h_evals.get(),
        0,
        "the correction must not evaluate one either: the quasi-Newton          matrix is all there is",
    );
    assert!(report.improved(), "the corrector must report improvement");
    let err = (base_x + refined[0] - exact).abs();
    assert!(
        err < predictor_err / 100.0,
        "eight iterations against the held quasi-Newton matrix still          close two orders of the curvature gap: {err:e} from a          predictor error of {predictor_err:e}",
    );
}

// ---------------------------------------------------------------------------
// The same corrections over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and hands the same step to the corrector over an engine built
// from that report.

#[test]
fn the_corrector_closes_curvature_the_base_operator_cannot_over_an_outside_engine() {
    let (solver, model) = solved_recorded(false);
    let delta = 0.5;
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("full step");
    let pounce = solver
        .correct_step(&[0], &[delta], &step, 6)
        .expect("pounce's corrector");
    let eng = outside_engine(&model, Relaxation::None);
    let outside = correct_outside(&eng, &[0], &[delta], &step, 6).expect("outside corrector");
    assert_same_correction("exact Hessian", &pounce, &outside, Frame::Same);
}

/// The operator of a limited-memory solve keeps the solve's quasi-Newton
/// matrix, and a report carries no such matrix, so an engine outside pounce
/// cannot rebuild that operator. What it holds instead is its own: here the
/// exact Hessian at the converged point, kept rather than re-evaluated, which
/// is what `exact_hessian = false` asks of an engine. The two corrections
/// iterate against different operators and are not the same point. Both
/// have to close the curvature gap as the test above requires.
#[test]
fn a_limited_memory_solve_corrects_without_an_exact_hessian_over_an_outside_engine() {
    let (solver, model) = solved_recorded(true);
    let base_x = solver.converged().expect("converged").x[0];
    let delta = 0.5;
    let exact = (1.0_f64 + delta).ln();
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("full step");
    let predictor_err = (base_x + step[0] - exact).abs();
    let (pounce, _) = solver
        .correct_step(&[0], &[delta], &step, 8)
        .expect("pounce's corrector");
    let eng = outside_engine(&model, Relaxation::None);
    let (outside, report) =
        correct_outside_holding(&eng, &[0], &[delta], &step, 8, false).expect("outside corrector");
    assert!(
        report.improved(),
        "the outside corrector must report improvement"
    );
    for (who, refined) in [("pounce", &pounce), ("outside", &outside)] {
        let err = (base_x + refined[0] - exact).abs();
        assert!(
            err < predictor_err / 100.0,
            "{who}: {err:e} from a predictor error of {predictor_err:e}",
        );
    }
}
