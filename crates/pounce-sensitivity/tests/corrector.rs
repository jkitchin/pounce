//! The corrector on a QP whose exact answer is known.
//!
//! ```text
//! min 0.5 x1^2 + 0.5 x2^2 + G x1 x2 - a(p) x1 - b(p) x2
//! s.t. x3 = p,   a(p) = 0.18 + 1.10 x3,   b(p) = -0.29 + 0.11 x3
//!      0 <= x1 <= 1,   0 <= x2 <= 10
//! ```
//!
//! The objective is quadratic and the constraint linear, so the
//! solution is exactly linear in `p` while the active set holds, and
//! the parametric step is already exact. That makes the fixture a
//! check that the corrector does no harm where there is nothing to
//! correct: the residual it reports must be at the barrier's own
//! floor, and the step must come back unchanged.
//!
//! Moving `p` changes the active set in either direction, and the
//! fixture reaches both. The base solve holds `x2` on its lower bound,
//! since `b(0)` is negative. Past `p = 0.573` the true `x2` is
//! positive, so that bound leaves the active set, and below `p = 0.728`
//! the true `x1` is still inside its upper bound, so a `p` between the
//! two releases one bound and touches nothing else. At `p = 1` the
//! opposite happens: `x1` reaches its upper bound, which the base
//! point's factor holds nothing against.
//!
//! Both cases have a closed form, since the solution solves a 2x2
//! linear system while the active set holds, so `exact_at` gives the
//! answer the corrector is aiming at without solving again.

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

const G: Number = -0.28;
const A0: Number = 0.18;
const A1: Number = 1.10;
const B0: Number = -0.29;
const B1: Number = 0.11;

/// `x_scaling` reports per-variable factors through
/// `get_scaling_parameters`, so the same QP can be solved in the
/// model's own units or in scaled ones. The corrector reads the
/// algorithm's iterate, which is scaled, and is handed a step and
/// bounds in the model's units, so the two frames have to be
/// reconciled and only a non-unit factor shows whether they are.
struct ParamQp {
    x_scaling: Option<[Number; 3]>,
    /// Row factor for the single constraint, the pinned equality. The
    /// residual the corrector measures sits in the algorithm's scaled
    /// equality block, so the perturbation has to carry the same
    /// factor, and only a non-unit one shows whether it does.
    g_scaling: Option<[Number; 1]>,
    /// A factor written into the constraint itself, `row_scale * x3 = 0`
    /// rather than `x3 = 0`, so `nlp_scaling_method=gradient-based` --
    /// the default -- derives the row factor from the Jacobian instead
    /// of being handed one. Same `dc` in the end, reached the other way,
    /// and the two routes are far enough apart that exercising only the
    /// supplied one leaves the default method uncovered.
    row_scale: Number,
}

impl TNLP for ParamQp {
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        if self.x_scaling.is_none() && self.g_scaling.is_none() {
            return false;
        }
        *req.obj_scaling = 1.0;
        match self.x_scaling {
            Some(d) => {
                *req.use_x_scaling = true;
                req.x_scaling.copy_from_slice(&d);
            }
            None => *req.use_x_scaling = false,
        }
        match self.g_scaling {
            Some(c) => {
                *req.use_g_scaling = true;
                req.g_scaling.copy_from_slice(&c);
            }
            None => *req.use_g_scaling = false,
        }
        true
    }

    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 10.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.3;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        let a = A0 + A1 * p;
        let b = B0 + B1 * p;
        Some(0.5 * x1 * x1 + 0.5 * x2 * x2 + G * x1 * x2 - a * x1 - b * x2)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        g[0] = x1 + G * x2 - (A0 + A1 * p);
        g[1] = x2 + G * x1 - (B0 + B1 * p);
        g[2] = -A1 * x1 - B1 * x2;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.row_scale * x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = self.row_scale,
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 1, 0, 0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor * G;
                values[3] = -obj_factor * A1;
                values[4] = -obj_factor * B1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved() -> Solver {
    solved_scaled(None)
}

fn solved_scaled(x_scaling: Option<[Number; 3]>) -> Solver {
    solved_with(x_scaling, None)
}

/// The QP with its constraint written `row_scale * x3 = 0`, solved
/// under the default `gradient-based` method so the row factor is
/// derived rather than supplied. `nlp_scaling_max_gradient` is 100, so
/// a row_scale above it puts `dc` at `100 / row_scale`.
fn solved_gradient_based(row_scale: Number) -> Solver {
    solved_inner(None, None, row_scale, Some("gradient-based"))
}

fn solved_with(x_scaling: Option<[Number; 3]>, g_scaling: Option<[Number; 1]>) -> Solver {
    // `None` leaves the option unset, the way this fixture always ran.
    let method = (x_scaling.is_some() || g_scaling.is_some()).then_some("user-scaling");
    solved_inner(x_scaling, g_scaling, 1.0, method)
}

fn solved_inner(
    x_scaling: Option<[Number; 3]>,
    g_scaling: Option<[Number; 1]>,
    row_scale: Number,
    method: Option<&str>,
) -> Solver {
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
    if let Some(m) = method {
        app.options_mut()
            .set_string_value("nlp_scaling_method", m, true, false)
            .unwrap();
        // set rather than assumed: the derived row factor is
        // `nlp_scaling_max_gradient / row_scale`, so a changed default
        // would move the number this fixture reasons about without
        // failing.
        app.options_mut()
            .set_numeric_value("nlp_scaling_max_gradient", 100.0, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParamQp {
        x_scaling,
        g_scaling,
        row_scale,
    }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "base solve failed: {status:?}",
    );
    solver
}

/// The solution with both bounds inactive, from the 2x2 system the
/// first-order conditions reduce to.
fn exact_at(p: Number) -> (Number, Number) {
    let (a, b) = (A0 + A1 * p, B0 + B1 * p);
    let det = 1.0 - G * G;
    ((a - G * b) / det, (b - G * a) / det)
}

/// The step `pyomo-pounce` hands the corrector under `mode="fix_relax"`:
/// that mode's primal block carrying the plain step's multipliers.
///
/// It matters which one the corrector gets. A bound the solve held
/// contributes a barrier diagonal the plain step cannot move against,
/// so the plain step leaves the variable on the bound and the endpoint
/// shows no release. `fix_relax` decides the release itself, and its
/// endpoint is where the corrector reads it from.
fn fix_relax_step(solver: &Solver, dp: Number) -> Vec<Number> {
    let mut full = solver
        .parametric_step_full(&[0], &[dp])
        .expect("parametric step");
    let (primal, _, _) = solver
        .parametric_step_bounded(&[0], &[dp], 16, None)
        .expect("fix_relax step");
    full[..primal.len()].copy_from_slice(&primal);
    full
}

/// `p` between the two thresholds in the module header, where the only
/// active-set change is `x2` leaving its lower bound.
const RELEASE_DP: Number = 0.65;

#[test]
fn the_corrector_releases_a_bound_the_solve_held() {
    // The base solve holds x2 down, and at this p the true x2 is off
    // the bound. The corrector takes that bound out of the operator
    // once, and the iterations then reach the exact answer.
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    let step = fix_relax_step(&solver, RELEASE_DP);
    let (out, report) = solver
        .correct_step(&[0], &[RELEASE_DP], &step, 12)
        .expect("corrector");
    assert_eq!(
        (report.released, report.pinned),
        (1, 0),
        "this p should release one bound and pin none, got {report:?}",
    );
    assert!(
        report.residual < report.initial_residual * 1e-6,
        "releasing the bound should let the iterations converge: {:.3e} -> {:.3e}",
        report.initial_residual,
        report.residual,
    );
    assert!(
        report.converged,
        "the loop should stop on its own: {report:?}"
    );
    let (x1, x2) = exact_at(RELEASE_DP);
    assert!(
        (base[0] + out[0] - x1).abs() < 1e-7 && (base[1] + out[1] - x2).abs() < 1e-7,
        "corrected to ({}, {}), exact is ({x1}, {x2})",
        base[0] + out[0],
        base[1] + out[1],
    );
}

#[test]
fn a_plain_step_leaves_a_held_bound_where_it_was() {
    // The same p through the plain step. Its endpoint keeps x2 on the
    // bound, so there is no release to read and the corrector iterates
    // against the base point's own barrier term, which holds x2 down.
    // Nothing moves, and that is what the residual reports.
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    let step = solver
        .parametric_step_full(&[0], &[RELEASE_DP])
        .expect("parametric step");
    let (_, report) = solver
        .correct_step(&[0], &[RELEASE_DP], &step, 12)
        .expect("corrector");
    assert_eq!(
        (report.released, report.pinned),
        (0, 0),
        "the plain step's endpoint shows no active-set change: {report:?}",
    );
    let (_, x2) = exact_at(RELEASE_DP);
    assert!(
        x2 > 1e-3,
        "this p should want x2 off its bound, exact x2 = {x2}"
    );
    assert!(
        base[1] + step[1] < 1e-6,
        "the plain step should leave x2 on the bound, at {}",
        base[1] + step[1],
    );
    assert!(
        report.residual > report.initial_residual * 0.99,
        "with the bound still in the operator there is nothing to gain:          {:.3e} -> {:.3e}",
        report.initial_residual,
        report.residual,
    );
}

#[test]
fn the_corrector_leaves_an_exact_step_alone() {
    // A small move keeps the active set, where the quadratic model
    // makes the parametric step exact. There is nothing to correct, so
    // the residual must start at the barrier floor and the returned
    // step must match the one handed in.
    let solver = solved();
    let step = solver
        .parametric_step_full(&[0], &[0.05])
        .expect("parametric step");
    let (out, report) = solver
        .correct_step(&[0], &[0.05], &step, 8)
        .expect("corrector");
    assert_eq!(out.len(), step.len());
    assert_eq!(
        (report.released, report.pinned),
        (0, 0),
        "a step this small changes no bound's status: {report:?}",
    );
    assert!(
        report.initial_residual < 1e-6,
        "an exact step should start at the barrier floor, got {}",
        report.initial_residual,
    );
    assert!(
        report.residual <= report.initial_residual,
        "the corrector must not make the residual worse: {} -> {}",
        report.initial_residual,
        report.residual,
    );
    let moved = out
        .iter()
        .zip(&step)
        .fold(0.0_f64, |a, (&o, &s)| a.max((o - s).abs()));
    assert!(
        moved < 1e-6,
        "an exact step should come back unchanged, moved by {moved}",
    );
}

#[test]
fn the_corrector_reduces_the_residual_where_the_step_is_wrong() {
    // p = 1 carries x1 to its upper bound, so the linear step is well
    // off and leaves a residual the held factor can work on.
    let solver = solved();
    let step = solver
        .parametric_step_full(&[0], &[1.0])
        .expect("parametric step");
    let (out, report) = solver
        .correct_step(&[0], &[1.0], &step, 12)
        .expect("corrector");
    assert!(
        report.initial_residual > 1e-6,
        "this step should leave a residual to work on, got {}",
        report.initial_residual,
    );
    assert_eq!(
        (report.released, report.pinned),
        (0, 1),
        "this p should pin one bound and release none: {report:?}",
    );
    // A magnitude, not just `improved()`. Without the pin applied the
    // iterations move the residual by about 1e-10 of itself, which a
    // strict inequality accepts and this does not.
    assert!(
        report.residual < report.initial_residual * 0.95,
        "the corrector should reduce the residual by more than roundoff:          {:.17e} -> {:.17e} in {} iteration(s)",
        report.initial_residual,
        report.residual,
        report.iterations,
    );
    assert!(report.iterations >= 1);
    // every corrected point stays inside the variable bounds
    assert!(
        out[0] + solver.converged().expect("converged").x[0] <= 1.0 + 1e-9,
        "x1 left its upper bound",
    );
}

#[test]
fn a_zero_budget_measures_the_step_without_iterating() {
    // Zero iterations still puts the point inside the bounds, since
    // that is what the returned point guarantees and what the residual
    // is defined at. So this costs one evaluation, no back-solve, and
    // reports how far the step is from satisfying the barrier system.
    let solver = solved();
    let base = solver.converged().expect("converged").x.clone();
    let step = solver
        .parametric_step_full(&[0], &[1.0])
        .expect("parametric step");
    assert!(
        base[0] + step[0] > 1.0,
        "this step should carry x1 past its upper bound, to {}",
        base[0] + step[0],
    );
    let (out, report) = solver
        .correct_step(&[0], &[1.0], &step, 0)
        .expect("corrector");
    assert_eq!(report.iterations, 0);
    assert_eq!(
        (report.released, report.pinned),
        (0, 1),
        "the active set is decided before the first iteration: {report:?}",
    );
    assert_eq!(report.residual, report.initial_residual);
    assert!(!report.improved());
    assert!(
        report.residual > 0.0,
        "a zero budget should still report the step's residual",
    );
    assert!(
        base[0] + out[0] <= 1.0,
        "the returned point must satisfy the bounds, x1 at {}",
        base[0] + out[0],
    );
}

#[test]
fn the_corrector_rejects_a_step_of_the_wrong_length() {
    let solver = solved();
    let err = solver
        .correct_step(&[0], &[1.0], &[0.0; 3], 4)
        .expect_err("a short step should be refused");
    let msg = format!("{err:?}");
    assert!(msg.contains("step"), "unhelpful error: {msg}");
}

#[test]
fn the_correction_is_the_same_under_variable_scaling() {
    // The corrector adds the algorithm's iterate, which the solve
    // keeps scaled, to a step and bounds that arrive in the model's own
    // units. With unit factors the two frames coincide and any mix-up
    // is invisible, so this runs the release case at three factors and
    // asks for the same answer from each (gh#733 review).
    let (x1, x2) = exact_at(RELEASE_DP);
    for d in [1.0, 2.0, 10.0] {
        let scaling = (d != 1.0).then_some([d, d, d]);
        let solver = solved_scaled(scaling);
        let base = solver.converged().expect("converged").x.clone();
        let step = fix_relax_step(&solver, RELEASE_DP);
        let (out, report) = solver
            .correct_step(&[0], &[RELEASE_DP], &step, 12)
            .expect("corrector");
        let (e1, e2) = (base[0] + out[0] - x1, base[1] + out[1] - x2);
        assert!(
            e1.abs() < 1e-7 && e2.abs() < 1e-7,
            "d = {d}: corrected to ({}, {}), exact is ({x1}, {x2}); {report:?}",
            base[0] + out[0],
            base[1] + out[1],
        );
        // and it still has to land inside the declared bounds
        assert!(
            base[0] + out[0] >= -1e-9 && base[0] + out[0] <= 1.0 + 1e-9,
            "d = {d}: x1 left its bounds at {}",
            base[0] + out[0],
        );
        // The answer alone does not say the frames agree. The step this
        // fixture hands over is already exact, so a direction that is
        // merely damped still lands on it, and only the residual says
        // whether the iterations did their work. `E` applied twice --
        // the scaled residual passed to a `solve` that wants natural
        // units -- left this at 4.1e-4 for d = 10 against 3.7e-11 for
        // d = 1, all of it in stationarity (gh#733 review).
        assert!(
            report.residual < report.initial_residual * 1e-6,
            "d = {d}: the iterations should drive the residual down as far \
             as they do unscaled, got {:.3e} -> {:.3e} (stationarity {:.3e})",
            report.initial_residual,
            report.residual,
            report.stationarity,
        );
    }
}

#[test]
fn the_correction_is_the_same_under_constraint_scaling() {
    // The other half of the frame. The perturbation is a move in the
    // pinned equality's right-hand side, and the residual it is
    // measured against lives in the algorithm's scaled equality block,
    // so the delta has to carry that row's factor. A unit factor hides
    // whether it does (gh#733 review, reproducer B).
    let (x1, x2) = exact_at(RELEASE_DP);
    for c in [1.0, 1.0e3, 1.0e4] {
        let scaling = (c != 1.0).then_some([c]);
        let solver = solved_with(None, scaling);
        let base = solver.converged().expect("converged").x.clone();
        let step = fix_relax_step(&solver, RELEASE_DP);
        let (out, report) = solver
            .correct_step(&[0], &[RELEASE_DP], &step, 12)
            .expect("corrector");
        let (e1, e2) = (base[0] + out[0] - x1, base[1] + out[1] - x2);
        assert!(
            e1.abs() < 1e-7 && e2.abs() < 1e-7,
            "row scale {c}: corrected to ({}, {}), exact is ({x1}, {x2}); {report:?}",
            base[0] + out[0],
            base[1] + out[1],
        );
        assert!(
            report.residual < report.initial_residual * 1e-6,
            "row scale {c}: the residual has to fall by orders, {:.4e} -> {:.4e};              {report:?}",
            report.initial_residual,
            report.residual,
        );
    }
}

/// The derived route to a row factor: `gradient-based`, the default
/// method, computing `dc` from the Jacobian instead of being handed it.
///
/// [`the_correction_is_the_same_under_constraint_scaling`] supplies
/// `g_scaling` under `user-scaling`. Both end at a non-unit `dc`, but
/// they reach it by different code, and the default method is the one
/// almost every solve runs under. Exercising only the supplied route is
/// the same shape of gap as the three this fixture already covers: a
/// dimension no fixture varies (gh#733 review, reproducer B).
///
/// The perturbation is a move in the constraint's right-hand side, so
/// it carries the row factor: pinning `row_scale * x3 = 0` and asking
/// for `RELEASE_DP * row_scale` moves `x3` by `RELEASE_DP`, the same
/// perturbation the unscaled case takes.
#[test]
fn the_correction_is_the_same_under_derived_constraint_scaling() {
    let (x1, x2) = exact_at(RELEASE_DP);
    for row_scale in [1.0, 1.0e3, 1.0e4] {
        let solver = solved_gradient_based(row_scale);
        // Without this the test can go vacuous in silence: if the
        // derived factor came back 1.0 the fixture would be the
        // unscaled one under a different name, which is the failure
        // mode that left `g_scaling` unexercised in the first place.
        let dc = solver.pin_g_scaling(&[0]).expect("pin scaling")[0];
        let expected = if row_scale > 100.0 {
            100.0 / row_scale
        } else {
            1.0
        };
        assert!(
            (dc - expected).abs() < 1e-9 * expected,
            "row scale {row_scale}: gradient-based should derive dc = \
             {expected}, got {dc}",
        );
        let base = solver.converged().expect("converged").x.clone();
        let delta = RELEASE_DP * row_scale;
        let mut step = solver
            .parametric_step_full(&[0], &[delta])
            .expect("parametric step");
        let (primal, _, _) = solver
            .parametric_step_bounded(&[0], &[delta], 16, None)
            .expect("fix_relax step");
        step[..primal.len()].copy_from_slice(&primal);
        let (out, report) = solver
            .correct_step(&[0], &[delta], &step, 12)
            .expect("corrector");
        assert!(
            (base[0] + out[0] - x1).abs() < 1e-7 && (base[1] + out[1] - x2).abs() < 1e-7,
            "row scale {row_scale}: corrected to ({}, {}), exact is ({x1}, {x2}); {report:?}",
            base[0] + out[0],
            base[1] + out[1],
        );
        // The answer alone is not evidence the direction was right, the
        // same reason the other two scaling tests assert this.
        assert!(
            report.residual < report.initial_residual * 1e-6,
            "row scale {row_scale}: the residual has to fall by orders, \
             {:.4e} -> {:.4e}; {report:?}",
            report.initial_residual,
            report.residual,
        );
    }
}
