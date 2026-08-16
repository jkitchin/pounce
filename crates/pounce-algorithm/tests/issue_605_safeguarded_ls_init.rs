//! gh#605 — the least-square primal initializer must not hand the
//! algorithm a *worse* starting point than the user supplied.
//!
//! `least_square_init_primal` replaces `x0` with the minimum-norm
//! solution of the **linearized** constraints. Where the Jacobian is
//! small relative to the residual, that linearization asks for a huge
//! correction and the true nonlinear violation at the far end is far
//! worse than where it started.
//!
//! This model is the smallest case that shows it:
//!
//! ```text
//! min  (x0 - 3)^2 + (x1 - 3)^2
//! s.t. x0^2 + x1^2 = 1          started at (0.05, 0.05)
//! ```
//!
//! At `x0 = (0.05, 0.05)` the constraint residual is `-0.995` and the
//! Jacobian is `(0.1, 0.1)`. The min-norm linearized correction is
//! `d = (4.975, 4.975)`, landing at `(5.025, 5.025)` where the true
//! violation is `2*5.025^2 - 1 ≈ 49.5` — **50x worse** than the
//! `0.995` it started with.
//!
//! The safeguard added in gh#605 scores every trial step on the true
//! nonlinear violation and backtracks (or declines outright) when it
//! does not improve, so the algorithm never starts from the far point.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct PoorLinearization;

impl TNLP for PoorLinearization {
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
        b.x_l.copy_from_slice(&[-10.0, -10.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.05, 0.05]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
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
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = obj_factor * 2.0 + lam[0] * 2.0;
                values[1] = obj_factor * 2.0 + lam[0] * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve_with_ls_init() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    // The Mehrotra cascade is the route that turns
    // `least_square_init_primal` on.
    app.options_mut()
        .set_string_value("mehrotra_algorithm", "yes", true, false)
        .unwrap();
    // The initializer runs once, before iteration 1, so the report is
    // fully populated no matter where the solve goes afterwards. Cap
    // the iterations and silence the log: this test is about the
    // starting point, not about what Mehrotra does with it.
    app.options_mut()
        .set_integer_value("max_iter", 5, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PoorLinearization));
    let _ = app.optimize_tnlp(tnlp);
    app
}

/// The unsafeguarded step lands at `(5.025, 5.025)`, whose true
/// violation is `2*5.025^2 - 1 = 49.50125`. Anything at or above this
/// means the raw linearized point was taken.
const UNSAFEGUARDED_VIOLATION: Number = 49.50125;

#[test]
fn least_square_init_never_worsens_the_nonlinear_violation() {
    let app = solve_with_ls_init();
    let report = app
        .least_square_init_report()
        .expect("least_square_init_primal ran, so it must report");
    eprintln!("gh605: {report:?}");

    assert!(
        (report.violation_initial - 0.995).abs() < 1e-6,
        "expected theta0 = 0.995 at (0.05, 0.05), got {}",
        report.violation_initial,
    );
    // The contract: the point handed to the algorithm is never worse
    // than the one the user gave.
    assert!(
        report.violation_final <= report.violation_initial,
        "initializer made the starting point WORSE: {} -> {}",
        report.violation_initial,
        report.violation_final,
    );
    // And specifically, nowhere near the unsafeguarded landing point.
    assert!(
        report.violation_final < UNSAFEGUARDED_VIOLATION * 0.5,
        "initializer accepted (or nearly accepted) the raw linearized \
         step: violation_final = {} (unsafeguarded lands at {})",
        report.violation_final,
        UNSAFEGUARDED_VIOLATION,
    );
}

#[test]
fn poor_linearization_backtracks_before_accepting() {
    let app = solve_with_ls_init();
    let report = app
        .least_square_init_report()
        .expect("least_square_init_primal ran, so it must report");
    // alpha = 1 lands at violation ~49.5, alpha = 1/2 at ~11.8,
    // alpha = 1/4 at ~2.4 -- all worse than 0.995. The first trial
    // that improves is alpha = 1/8.
    assert!(
        report.rejected_trials > 0,
        "expected the full-length step to be rejected, but {} trials \
         were rejected (report: {report:?})",
        report.rejected_trials,
    );
    assert!(
        report.alpha < 1.0,
        "expected a backtracked step, got alpha = {}",
        report.alpha,
    );
    assert_eq!(report.termination, "accepted");
    assert!(
        report.step_norm > 0.0,
        "an accepted step must have a positive norm",
    );
}
