//! A predicted point outside the model's own domain must not read as a
//! perfect correction (gh#845).
//!
//! ```text
//! min (y - 1/2)^2 + 0.1 (x - 2)^2
//! s.t.  y - log(x) = 0
//!       x - 4      = 0      <- the pin, whose right-hand side moves
//! ```
//!
//! `x` is declared with no bounds, so nothing in `correct_step`'s
//! clamp keeps it positive: the clamp puts a coordinate back inside its
//! *declared* bounds, and a variable held in a function's domain by a
//! **constraint** has none to be put back inside. The system is square,
//! so the parametric step is exact and linear -- `dx = dp`, `dy = dp/x`
//! -- and a `dp = -5` lands the predicted point at `x = -1`, where
//! `log(x)` is NaN and the whole barrier residual goes with it.
//!
//! The defect was that `corrector::run` normed that residual with a
//! fold over `f64::max`, which returns the *other* operand when one is
//! NaN. An all-NaN residual normed to `0.0` -- the smallest value the
//! stopping rule can see -- so the NaN iterate was accepted as the best
//! point yet and reported with `residual = 0.0`, all three split
//! residuals `0.0`, `converged = true` and `improved() = true`, while
//! the step handed back was all NaN.
//!
//! The two things this file pins are that a non-finite residual is a
//! failure rather than a perfect score, and that no NaN reaches the
//! caller in a step.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::Solver;

/// The pinned row's index in `g`, and the base point's `x`.
const PIN: Index = 1;
const BASE_X: Number = 4.0;

struct LogDomain;

impl TNLP for LogDomain {
    fn get_scaling_parameters(&mut self, _req: ScalingRequest<'_>) -> bool {
        false
    }

    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 3,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // No bounds on either variable: the point of the fixture.
        for i in 0..2 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
            b.g_l[i] = 0.0;
            b.g_u[i] = 0.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = BASE_X;
        sp.x[1] = 1.3;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[1] - 0.5).powi(2) + 0.1 * (x[0] - 2.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 0.2 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 0.5);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[1] - x[0].ln();
        g[1] = x[0] - BASE_X;
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1]);
                jcol.copy_from_slice(&[0, 1, 0]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("jacobian values need x");
                values[0] = -1.0 / x[0];
                values[1] = 1.0;
                values[2] = 1.0;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("hessian values need x");
                let l0 = lambda.map_or(0.0, |l| l[0]);
                // d2/dx2 of -lambda0 * log(x) is +lambda0 / x^2.
                values[0] = obj_factor * 0.2 + l0 / (x[0] * x[0]);
                values[1] = obj_factor * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved() -> Solver {
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
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(LogDomain));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

/// The perturbation that puts the predicted `x` at `-1`, outside
/// `log`'s domain. The step is exact here, so this is arithmetic and
/// not a guess: `dx = dp`.
const DP: Number = -5.0;

#[test]
fn the_step_really_does_leave_the_domain() {
    // The premise of the whole file, asserted rather than assumed: the
    // uncorrected step is finite and lands where `log` is not defined.
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    assert!(
        (base[0] - BASE_X).abs() < 1e-8,
        "base x should be {BASE_X}, got {}",
        base[0]
    );
    let step = solver
        .parametric_step_full(&[PIN], &[DP])
        .expect("parametric step");
    assert!(
        step[..2].iter().all(|v| v.is_finite()),
        "the uncorrected step is finite: {:?}",
        &step[..2]
    );
    assert!(
        base[0] + step[0] < 0.0,
        "the predicted x must leave log's domain: {} + {} = {}",
        base[0],
        step[0],
        base[0] + step[0]
    );
}

#[test]
fn a_non_finite_residual_is_a_failure_not_a_perfect_score() {
    let solver = solved();
    let step = solver
        .parametric_step_full(&[PIN], &[DP])
        .expect("parametric step");
    match solver.correct_step(&[PIN], &[DP], &step, 4) {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("finite"),
                "the error should name the non-finite residual, got {msg}"
            );
        }
        Ok((out, report)) => panic!(
            "the corrector reported success on a point outside the model's \
             domain: residual={:.3e} initial={:.3e} converged={} \
             stationarity={:.3e} feasibility={:.3e} complementarity={:.3e} \
             improved={} step={:?}",
            report.residual,
            report.initial_residual,
            report.converged,
            report.stationarity,
            report.feasibility,
            report.complementarity,
            report.improved(),
            &out[..2],
        ),
    }
}

#[test]
fn a_correction_never_hands_back_a_non_finite_step() {
    // Whatever the corrector returns for this model, it is not NaN.
    // Separate from the test above because "reports failure" and
    // "returns no NaN" are two different promises, and a fix that keeps
    // only the first is still shipping NaN into a caller's estimate.
    let solver = solved();
    let step = solver
        .parametric_step_full(&[PIN], &[DP])
        .expect("parametric step");
    if let Ok((out, report)) = solver.correct_step(&[PIN], &[DP], &step, 4) {
        assert!(
            out.iter().all(|v| v.is_finite()),
            "corrector returned a non-finite step: {:?}",
            &out[..2]
        );
        assert!(
            report.residual.is_finite()
                && report.stationarity.is_finite()
                && report.feasibility.is_finite()
                && report.complementarity.is_finite(),
            "corrector reported a non-finite residual as a number: {report:?}"
        );
    }
}

/// A perturbation that stays inside the domain must be unaffected by
/// the screen: the corrector still runs, still improves, and still
/// reports finite numbers. Without this the fix could be "always fail"
/// and the tests above would not notice.
#[test]
fn a_perturbation_inside_the_domain_is_untouched() {
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    let dp = 0.5;
    let step = solver
        .parametric_step_full(&[PIN], &[dp])
        .expect("parametric step");
    let (out, report) = solver
        .correct_step(&[PIN], &[dp], &step, 8)
        .expect("an in-domain perturbation still corrects");
    assert!(
        out.iter().all(|v| v.is_finite()) && report.residual.is_finite(),
        "in-domain correction went non-finite: {report:?}"
    );
    // The system is square and the step exact in `x`, so the corrected
    // x lands on 4 + dp whatever the multipliers do.
    assert!(
        (base[0] + out[0] - (BASE_X + dp)).abs() < 1e-6,
        "corrected x should be {}, got {}",
        BASE_X + dp,
        base[0] + out[0]
    );
}
