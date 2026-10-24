//! `reduced_activity` over the four regimes one bounded coordinate can
//! be in, on a fixture whose reduced curvature is known by
//! construction (gh#763).
//!
//! `classify_activity` normalizes a variable's `Sigma` by the Hessian
//! DIAGONAL; `reduced_activity` normalizes it by the curvature
//! **reduced** along that coordinate, which is what generates the
//! multiplier. The two agree exactly when the coordinate is decoupled,
//! and this fixture is diagonal on purpose so that agreement is the
//! assertion: a refinement that moved a decoupled answer would be
//! wrong, and the interesting disagreement -- a coupled kink -- lives
//! in `sens_invariance_legs.rs`, on the fixture that carries one.
//!
//! The regimes, all four in one solve:
//!
//! | var | regime | reduced curvature |
//! |-----|--------|-------------------|
//! | `x0` | strongly active at its upper bound (`Sigma = O(1/mu)`) | 1 |
//! | `x1` | interior, bound inactive (`Sigma = O(mu)`) | 1 |
//! | `x2` | no finite bound | 1 |
//! | `x3` | bounded, but pinned by an equality | infinite |
//!
//! `x0` is where the subtraction `1/(K^-1)_ii - Sigma_i` cancels
//! hardest: `Sigma` there is `6e9` and the answer is `1`, so the
//! assertion below is also the cancellation check.

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
use pounce_sensitivity::activity::{INACTIVE, STRONGLY_ACTIVE, UNBOUNDED};

mod common;
use common::outside_engine::{
    Relaxation, outside_engine, outside_engine_minimizing, recorded_solve, same_classification,
    same_classification_of, same_reduced_activity,
};
use pounce_sensitivity::{Solver, SolverError};

/// ```text
/// min 0.5(x0-5)^2 + 0.5(x1-0.5)^2 + 0.5 x2^2 + 0.5(x3-9)^2
/// s.t. x3 == 2,  0 <= x0 <= 1,  0 <= x1 <= 1,  x2 free,  0 <= x3 <= 10
/// ```
struct RegimeTnlp;

impl TNLP for RegimeTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 1.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.x_l[3] = 0.0;
        b.x_u[3] = 10.0;
        b.g_l[0] = 2.0;
        b.g_u[0] = 2.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        sp.x[2] = 0.0;
        sp.x[3] = 2.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            0.5 * (x[0] - 5.0).powi(2)
                + 0.5 * (x[1] - 0.5).powi(2)
                + 0.5 * x[2] * x[2]
                + 0.5 * (x[3] - 9.0).powi(2),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 5.0;
        g[1] = x[1] - 0.5;
        g[2] = x[2];
        g[3] = x[3] - 9.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 3;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1, 2, 3]);
                jcol.copy_from_slice(&[0 as Index, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = obj;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// [`RegimeTnlp`] with the objective negated, so that under
/// `obj_scaling_factor = -1` the IPM re-negates it and the INTERNAL
/// problem — iterate, `Sigma`, factor — is identical to the `df = +1`
/// solve. Anything that moves between the two is the objective scale
/// being mishandled, not a different model.
struct NegatedRegimeTnlp;

impl TNLP for NegatedRegimeTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 1.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.x_l[3] = 0.0;
        b.x_u[3] = 10.0;
        b.g_l[0] = 2.0;
        b.g_u[0] = 2.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        sp.x[2] = 0.0;
        sp.x[3] = 2.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            -(0.5 * (x[0] - 5.0).powi(2)
                + 0.5 * (x[1] - 0.5).powi(2)
                + 0.5 * x[2] * x[2]
                + 0.5 * (x[3] - 9.0).powi(2)),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = -(x[0] - 5.0);
        g[1] = -(x[1] - 0.5);
        g[2] = -x[2];
        g[3] = -(x[3] - 9.0);
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 3;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1, 2, 3]);
                jcol.copy_from_slice(&[0 as Index, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = -obj;
                }
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
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(RegimeTnlp));
    let mut s = Solver::new(app, tnlp);
    let st = s.solve();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{st:?}"
    );
    s
}

/// Relative agreement, so the `Sigma = 6e9` row is held to the same
/// standard as the `Sigma = 2e-8` one.
fn assert_rel(what: &str, got: Number, want: Number, rtol: Number) {
    assert!(
        (got - want).abs() <= rtol * want.abs().max(1.0),
        "{what}: got {got:e}, want {want:e} (rtol {rtol:e})"
    );
}

/// The Hessian is the identity, so every free coordinate's reduced
/// curvature is exactly its diagonal and the refinement must not move
/// any verdict. Two regimes at once: `x0` is held hard by its bound
/// (`Sigma = O(1/mu)`) and `x1` is not (`Sigma = O(mu)`), and the
/// classifier's own edges are decades apart, so an agreement here is
/// not an artifact of a wide band.
#[test]
fn the_reduced_normalizer_agrees_with_the_diagonal_on_a_decoupled_model() {
    let s = solved();
    let rep = s.classify_activity().expect("activity report");
    let red = s.reduced_activity(&[0, 1]).expect("reduced activity");

    assert_eq!(
        rep.var_status[0], STRONGLY_ACTIVE,
        "precondition: x0 is held at its upper bound"
    );
    assert_eq!(rep.var_status[1], INACTIVE, "precondition: x1 is interior");
    for (k, i) in [0usize, 1].iter().enumerate() {
        // `H = I` on this model, so the reduced curvature is 1 whatever
        // the barrier is doing: the subtraction has to give back the
        // model's own diagonal.
        assert_rel(
            &format!("x{i} reduced curvature"),
            red.q_reduced[k],
            1.0,
            1e-5,
        );
        assert_eq!(red.q_sign[k], 1, "x{i} curvature sign");
        assert_rel(
            &format!("x{i} reduced ratio against the diagonal ratio"),
            red.ratio[k],
            rep.var_ratio[*i],
            1e-5,
        );
        assert_eq!(
            red.status[k], rep.var_status[*i],
            "x{i}: a decoupled coordinate must not change class under the \
             reduced normalizer (diagonal ratio {:e}, reduced {:e})",
            rep.var_ratio[*i], red.ratio[k]
        );
        assert_eq!(
            red.sigma[k], rep.var_sigma[*i],
            "x{i} Sigma is the report's"
        );
        assert_eq!(
            red.var[k], *i,
            "x{i} entry answers about the index asked for"
        );
    }
    assert_eq!(red.mu, rep.mu, "same converged iterate, same mu");
}

/// A variable with no finite bound has a reduced curvature like any
/// other -- the back-solve was paid for either way -- but no activity
/// question, so the status and ratio say so rather than reporting the
/// `Sigma = 0` as "inactive".
#[test]
fn a_variable_with_no_finite_bound_carries_curvature_but_no_verdict() {
    let s = solved();
    let red = s.reduced_activity(&[2]).expect("reduced activity");

    assert_eq!(red.status[0], UNBOUNDED, "x2 has no finite bound");
    assert!(
        red.ratio[0].is_nan(),
        "no bound, no ratio: {:e}",
        red.ratio[0]
    );
    assert_rel("x2 reduced curvature", red.q_reduced[0], 1.0, 1e-9);
}

/// `x3` is bounded `[0, 10]` and pinned to `2` by an equality, so the
/// solve leaves it no free direction: `(K^-1)_33` is `0` and the
/// reduced curvature is infinite. The ratio divides to `0`, which is
/// the classifier's INACTIVE -- and it is the right word here.
/// Whatever holds `x3`, it is not its bound.
#[test]
fn a_coordinate_an_equality_determines_has_no_direction_left_to_reduce_along() {
    let s = solved();
    let rep = s.classify_activity().expect("activity report");
    let red = s.reduced_activity(&[3]).expect("reduced activity");

    assert!(
        red.q_reduced[0].is_infinite() || red.q_reduced[0] > 1e12,
        "x3 is determined by its equality: curvature {:e}",
        red.q_reduced[0]
    );
    assert_eq!(
        red.ratio[0], 0.0,
        "an infinite normalizer divides Sigma to 0"
    );
    assert_eq!(red.status[0], INACTIVE);
    assert_eq!(
        rep.var_status[3], INACTIVE,
        "and the diagonal normalizer agrees, by a different route: \
         Sigma is O(mu) there"
    );
}

/// The accessor takes user-space indices, so it owes a user-space
/// range check rather than a panic out of a raw row index.
#[test]
fn an_index_past_the_users_variable_count_is_an_error() {
    let s = solved();
    let err = s
        .reduced_activity(&[4])
        .expect_err("index 4 of 4 is out of range");
    match err {
        SolverError::BadShape {
            what,
            got,
            expected,
        } => {
            assert_eq!(what, "reduced_activity variable index");
            assert_eq!((got, expected), (4, 4));
        }
        other => panic!("wrong error: {other:?}"),
    }
}

/// An empty request is an empty answer, not a back-solve.
#[test]
fn no_indices_is_an_empty_report() {
    let s = solved();
    let red = s.reduced_activity(&[]).expect("reduced activity");
    assert!(red.status.is_empty() && red.q_reduced.is_empty() && red.var.is_empty());
    assert!(red.mu > 0.0, "mu is still the converged one: {:e}", red.mu);
}

/// `max -[...]` via `obj_scaling_factor = -1`, the documented way to
/// maximize (see `scaling_invariance.rs`).
fn solved_maximizing() -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("obj_scaling_factor", -1.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(NegatedRegimeTnlp));
    let mut s = Solver::new(app, tnlp);
    let st = s.solve();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{st:?}"
    );
    s
}

/// The classification must not depend on the sign of the objective
/// scale, and the reported quantities must keep the sign the
/// natural-units contract gives them.
///
/// `compute` runs the rule on the df-in `Sigma` — the internal `z/s`,
/// which is non-negative whatever `df` is — and divides the objective
/// scale out only on export. A refinement that classified the
/// *natural* `Sigma` instead would hand the rule a negative ratio
/// under `obj_scaling_factor < 0` and read `x0`, pinned hard at its
/// upper bound, as INACTIVE.
///
/// This is the decoupled fixture, so as in
/// [`the_reduced_normalizer_agrees_with_the_diagonal_on_a_decoupled_model`]
/// the refinement must agree with the report — and here it must agree
/// under a negative `df` too, against a `+1` solve whose internal
/// problem is the same one.
#[test]
fn the_classification_is_unmoved_by_a_negative_objective_scale() {
    let plus = solved();
    let minus = solved_maximizing();

    let rep_p = plus.classify_activity().expect("activity report");
    let rep_m = minus.classify_activity().expect("activity report");
    let red_p = plus.reduced_activity(&[0, 1]).expect("reduced activity");
    let red_m = minus.reduced_activity(&[0, 1]).expect("reduced activity");

    for (k, i) in [0usize, 1].into_iter().enumerate() {
        // the report is the reference: it already classifies in the
        // objective-scale-positive frame, so a negative df moves
        // nothing in it
        assert_eq!(
            rep_p.var_status[i], rep_m.var_status[i],
            "report status for x{i} moved with the objective sign",
        );

        // the refinement must reach the same verdict as the report,
        // at BOTH signs -- this is the decoupled model
        assert_eq!(
            red_m.status[k], rep_m.var_status[i],
            "x{i}: reduced status {} disagrees with the report's {} under \
             obj_scaling_factor = -1",
            red_m.status[k], rep_m.var_status[i],
        );
        assert_eq!(
            red_m.status[k], red_p.status[k],
            "x{i}: reduced status moved with the objective sign",
        );
        assert_rel(
            &format!("x{i} reduced ratio under negative df"),
            red_m.ratio[k],
            red_p.ratio[k],
            1e-6,
        );

        // ...while the REPORTED quantities keep the natural-units
        // sign, as `var_sigma` does. Asserting this pins the fix to
        // the classification input and stops it being "take an abs
        // somewhere", which would corrupt the curvature the caller
        // reads.
        assert!(
            red_m.sigma[k] < 0.0 && red_p.sigma[k] > 0.0,
            "x{i}: sigma must carry the natural-units sign (got {} at df=-1, {} at df=+1)",
            red_m.sigma[k],
            red_p.sigma[k],
        );
        assert_rel(
            &format!("x{i} sigma magnitude under negative df"),
            red_m.sigma[k].abs(),
            red_p.sigma[k].abs(),
            1e-6,
        );
        assert_rel(
            &format!("x{i} q_reduced under negative df"),
            red_m.q_reduced[k],
            -red_p.q_reduced[k],
            1e-6,
        );
        assert_eq!(
            red_m.q_sign[k], -red_p.q_sign[k],
            "x{i}: q_sign must flip with the declared problem's curvature",
        );
    }
}

// ---------------------------------------------------------------------------
// The same classifications over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and asserts that the classification and its refinement by the
// reduced curvature come out the same over an engine built from that report.
// The refinement back-solves against each engine's own factorization, here
// the outside engine's dense LU of the eight-block matrix.

/// The options [`solved`] runs with.
fn regime_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

#[test]
fn the_reduced_normalizer_agrees_with_the_diagonal_on_a_decoupled_model_over_an_outside_engine() {
    let (s, model, _) = recorded_solve(regime_app(), RegimeTnlp);
    same_classification("regime", &s, &model, Relaxation::None);
    let eng = outside_engine(&model, Relaxation::None);
    same_reduced_activity("regime, x0 and x1", &s, &eng, &[0, 1]);
}

#[test]
fn a_variable_with_no_finite_bound_carries_curvature_but_no_verdict_over_an_outside_engine() {
    let (s, model, _) = recorded_solve(regime_app(), RegimeTnlp);
    let eng = outside_engine(&model, Relaxation::None);
    same_reduced_activity("regime, unbounded x2", &s, &eng, &[2]);
}

#[test]
fn a_coordinate_an_equality_determines_has_no_direction_left_to_reduce_along_over_an_outside_engine()
 {
    let (s, model, _) = recorded_solve(regime_app(), RegimeTnlp);
    same_classification("regime", &s, &model, Relaxation::None);
    let eng = outside_engine(&model, Relaxation::None);
    same_reduced_activity("regime, x3", &s, &eng, &[3]);
}

#[test]
fn an_index_past_the_users_variable_count_is_an_error_over_an_outside_engine() {
    let (_s, model, _) = recorded_solve(regime_app(), RegimeTnlp);
    let eng = outside_engine(&model, Relaxation::None);
    let err = pounce_sens_core::activity::reduced_activity(&eng, &[4])
        .expect_err("index 4 of 4 is out of range");
    assert_eq!(
        err,
        pounce_sens_core::activity::ReducedActivityError::OutOfRange {
            got: 4,
            n_full_x: 4
        },
    );
}

#[test]
fn no_indices_is_an_empty_report_over_an_outside_engine() {
    let (s, model, _) = recorded_solve(regime_app(), RegimeTnlp);
    let eng = outside_engine(&model, Relaxation::None);
    let red = pounce_sens_core::activity::reduced_activity(&eng, &[]).expect("reduced activity");
    assert!(red.status.is_empty() && red.q_reduced.is_empty() && red.var.is_empty());
    let pounce = s.reduced_activity(&[]).expect("pounce's reduced activity");
    assert_eq!(red.mu, pounce.mu, "the same converged barrier level");
}

/// The maximizing arm is solved as `min −f` with `obj_scaling_factor = −1`,
/// so its outside engine is told it minimized `−f`. pounce reports the
/// multipliers in the model's own sense, negated from the ones it iterated
/// with, and the outside engine negates them back.
#[test]
fn the_classification_is_unmoved_by_a_negative_objective_scale_over_an_outside_engine() {
    let (plus, model_p, _) = recorded_solve(regime_app(), RegimeTnlp);
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("obj_scaling_factor", -1.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (minus, model_m, _) = recorded_solve(app, NegatedRegimeTnlp);
    let eng_p = outside_engine(&model_p, Relaxation::None);
    let eng_m = outside_engine_minimizing(&model_m, Relaxation::None, -1.0);
    same_classification_of("plus", &plus, eng_p.clone());
    same_classification_of("minus", &minus, eng_m.clone());
    same_reduced_activity("plus", &plus, &eng_p, &[0, 1]);
    same_reduced_activity("minus", &minus, &eng_m, &[0, 1]);
}
