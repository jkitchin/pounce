//! `reduced_row_activity` over the regimes one bounded constraint row
//! can be in, on fixtures whose reduced curvature is known by
//! construction (gh#804).
//!
//! `classify_activity` normalizes a row's `Sigma` by the curvature
//! along the row's own gradient, `|grad_d^T H grad_d| / ||grad_d||^2`.
//! That is a genuine directional curvature -- strictly better than the
//! variable path's bare `H_ii`, which is why gh#763 fixed the
//! variables first -- but it is not a *reduced* curvature: the other
//! free coordinates still re-optimize, and what is left after they do
//! is what generates the row's multiplier. So a row's ratio there is
//! `reduced/directional`, and `reduced_row_activity` divides by the
//! reduced one.
//!
//! # Two fixtures, because the rule branches
//!
//! Per the `CLAUDE.md` branch rule, a leg is only evidence about the
//! branch its fixture reaches, so both live here:
//!
//! * [`DecoupledRowTnlp`] -- a diagonal model whose rows point along
//!   eigen-directions, so the two normalizers must AGREE. A refinement
//!   that moved a decoupled answer would be wrong.
//! * [`CoupledKinkRowTnlp`] -- a genuine row kink whose direction is
//!   coupled to a free partner, so the two must DISAGREE, by exactly
//!   the coupling. That is the test; the first is not a duplicate of
//!   it.
//!
//! Both carry an EQUALITY row ahead of the inequalities, so a full-g
//! index read as an inequality position picks a NEIGHBORING row --
//! the gh#450 hazard, one block over -- and both carry rows whose
//! gradients are not unit length, so the `||grad_d||^2` conversions
//! between the row's own units and the unit normal have somewhere to
//! go wrong.

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
use pounce_sensitivity::activity::{AMBIGUOUS, EQUALITY, INACTIVE, STRONGLY_ACTIVE, WEAKLY_ACTIVE};
use pounce_sensitivity::{Solver, SolverError};

/// Gradient scale of the bounded rows. Not 1, so that every
/// `||grad_d||^2` conversion -- the geometric weight in the numerator,
/// the unit-normal curvature in the denominator -- has a factor to get
/// right rather than an identity to pass through.
const G0_SCALE: Number = 2.0;
const G1_SCALE: Number = 3.0;

/// ```text
/// min 0.5(x0-5)^2 + 0.5(x1-0.5)^2 + 0.5 x2^2
/// s.t. g0:      x2 == 0            (equality, ahead of the d block)
///      g1: 2 * x0 <= 2             (strongly active: Sigma = O(1/mu))
///      g2: 3 * x1 >= -30           (inactive:        Sigma = O(mu))
/// ```
///
/// The Hessian is the identity, so each row's gradient is an
/// eigen-direction and the reduced curvature along it is exactly the
/// directional one: `1` along either unit normal, at both regimes.
/// The equality ahead of the inequalities makes full-g and the d block
/// diverge from `g1` on.
struct DecoupledRowTnlp {
    /// Negate the objective, so that under `obj_scaling_factor = -1`
    /// the IPM re-negates it and the INTERNAL problem is identical to
    /// the `df = +1` solve. Anything that moves between the two is the
    /// objective scale being mishandled, not a different model.
    negated: bool,
}

impl DecoupledRowTnlp {
    fn sgn(&self) -> Number {
        if self.negated { -1.0 } else { 1.0 }
    }
}

impl TNLP for DecoupledRowTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 3,
            nnz_jac_g: 3,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, _r: ScalingRequest<'_>) -> bool {
        false
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..3 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = -1.0e19;
        b.g_u[1] = 2.0;
        b.g_l[2] = -30.0;
        b.g_u[2] = 1.0e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        sp.x[2] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(
            self.sgn()
                * (0.5 * (x[0] - 5.0).powi(2) + 0.5 * (x[1] - 0.5).powi(2) + 0.5 * x[2] * x[2]),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let s = self.sgn();
        g[0] = s * (x[0] - 5.0);
        g[1] = s * (x[1] - 0.5);
        g[2] = s * x[2];
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = G0_SCALE * x[0];
        g[2] = G1_SCALE * x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1, 2]);
                jcol.copy_from_slice(&[2 as Index, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = G0_SCALE;
                values[2] = G1_SCALE;
            }
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
                irow.copy_from_slice(&[0 as Index, 1, 2]);
                jcol.copy_from_slice(&[0 as Index, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                // the rows are linear, so the Lagrangian Hessian is the
                // objective's alone
                for v in values.iter_mut() {
                    *v = obj * self.sgn();
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn app_with_defaults() -> IpoptApplication {
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
    app
}

fn run(mut app: IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>) -> Solver {
    app.initialize().unwrap();
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

fn solved_decoupled() -> Solver {
    run(
        app_with_defaults(),
        Rc::new(RefCell::new(DecoupledRowTnlp { negated: false })),
    )
}

/// `max -[...]` via `obj_scaling_factor = -1`, the documented way to
/// maximize (see `scaling_invariance.rs`).
fn solved_decoupled_maximizing() -> Solver {
    let mut app = app_with_defaults();
    app.options_mut()
        .set_numeric_value("obj_scaling_factor", -1.0, true, false)
        .unwrap();
    run(
        app,
        Rc::new(RefCell::new(DecoupledRowTnlp { negated: true })),
    )
}

/// Relative agreement, so the `Sigma = O(1/mu)` row is held to the
/// same standard as the `O(mu)` one.
fn assert_rel(what: &str, got: Number, want: Number, rtol: Number) {
    assert!(
        (got - want).abs() <= rtol * want.abs().max(1.0),
        "{what}: got {got:e}, want {want:e} (rtol {rtol:e})"
    );
}

/// The Hessian is the identity and both rows point along
/// eigen-directions, so each row's reduced curvature is exactly the
/// directional one and the refinement must not move any verdict. Two
/// regimes at once -- `g1` is held hard against its bound and `g2` is
/// not -- and the classifier's own edges are decades apart, so an
/// agreement here is not an artifact of a wide band.
///
/// `g1` is also where the subtraction `1/(K^-1)_ss - Sigma` cancels
/// hardest: `Sigma` there is `O(1/mu)` and the answer is `1`, so this
/// is the cancellation check too.
#[test]
fn the_reduced_normalizer_agrees_with_the_directional_one_on_a_decoupled_model() {
    let s = solved_decoupled();
    let rep = s.classify_activity().expect("activity report");
    let red = s.reduced_row_activity(&[1, 2]).expect("reduced rows");

    assert_eq!(
        rep.row_status[1], STRONGLY_ACTIVE,
        "precondition: g1 holds x0 away from 5 (ratio {:e})",
        rep.row_ratio[1]
    );
    assert_eq!(
        rep.row_status[2], INACTIVE,
        "precondition: g2 is slack (ratio {:e})",
        rep.row_ratio[2]
    );

    for (k, j) in [1usize, 2].iter().enumerate() {
        // `H = I` on this model and both rows are eigen-directions, so
        // the curvature along either unit normal is 1 whatever the
        // barrier is doing: the subtraction has to give the model's
        // own curvature back.
        assert_rel(
            &format!("g{j} reduced curvature along the unit normal"),
            red.q_reduced[k],
            1.0,
            1e-5,
        );
        assert_eq!(red.q_sign[k], 1, "g{j} curvature sign");
        assert_rel(
            &format!("g{j} reduced ratio against the directional ratio"),
            red.ratio[k],
            rep.row_ratio[*j],
            1e-5,
        );
        assert_eq!(
            red.status[k], rep.row_status[*j],
            "g{j}: a decoupled row must not change class under the reduced \
             normalizer (directional ratio {:e}, reduced {:e})",
            rep.row_ratio[*j], red.ratio[k]
        );
        assert_eq!(
            red.sigma[k], rep.row_sigma[*j],
            "g{j} Sigma is the report's, raw"
        );
        assert_eq!(
            red.row[k], *j,
            "g{j} entry answers about the index asked for"
        );
    }
    assert_eq!(red.mu, rep.mu, "same converged iterate, same mu");
}

/// The accessor is handed full-g indices while the factor's slack rows
/// are the d block, and an equality ahead of them makes the two
/// diverge. Reading `g1` as d-row 1 returns `g2`'s answer -- plausible
/// and wrong, the gh#450 hazard one block over -- so this asserts the
/// two rows are genuinely distinguishable first, and then that each
/// index gets its own answer.
#[test]
fn a_full_g_index_is_not_read_as_an_inequality_position() {
    let s = solved_decoupled();
    let rep = s.classify_activity().expect("activity report");
    let red = s.reduced_row_activity(&[1, 2]).expect("reduced rows");

    assert_ne!(
        rep.row_status[1], rep.row_status[2],
        "fixture drifted: the two inequality rows must be in different \
         regimes or a swapped index would be invisible",
    );
    assert_eq!(red.status[0], rep.row_status[1], "g1 answered as g1");
    assert_eq!(red.status[1], rep.row_status[2], "g2 answered as g2");
    // and the ratios differ by decades, so this is not a coincidence
    // of two classes that happen to collide
    assert!(
        red.ratio[0] / red.ratio[1] > 1e6,
        "g1 {:e} and g2 {:e} should be decades apart",
        red.ratio[0],
        red.ratio[1]
    );
}

/// An equality row has no slack and no barrier multiplier pair, so
/// there is no activity question and nothing to reduce: the report
/// says EQUALITY and so does this, rather than erroring or answering
/// about the row next door.
#[test]
fn an_equality_row_reports_the_reports_placeholder() {
    let s = solved_decoupled();
    let rep = s.classify_activity().expect("activity report");
    let red = s.reduced_row_activity(&[0]).expect("reduced rows");

    assert_eq!(
        rep.row_status[0], EQUALITY,
        "precondition: g0 is the equality"
    );
    assert_eq!(red.status[0], EQUALITY);
    assert!(red.ratio[0].is_nan(), "no ratio: {:e}", red.ratio[0]);
    assert!(
        red.q_reduced[0].is_nan(),
        "no curvature: {:e}",
        red.q_reduced[0]
    );
    assert_eq!(red.q_sign[0], 0);
    assert_eq!(red.sigma[0], 0.0);
}

/// The accessor takes user-space constraint indices, so it owes a
/// user-space range check rather than a panic out of a raw row index.
#[test]
fn an_index_past_the_users_constraint_count_is_an_error() {
    let s = solved_decoupled();
    let err = s
        .reduced_row_activity(&[3])
        .expect_err("index 3 of 3 is out of range");
    match err {
        SolverError::BadShape {
            what,
            got,
            expected,
        } => {
            assert_eq!(what, "reduced_row_activity constraint index");
            assert_eq!((got, expected), (3, 3));
        }
        other => panic!("wrong error: {other:?}"),
    }
}

/// An empty request is an empty answer, not a back-solve.
#[test]
fn no_indices_is_an_empty_row_report() {
    let s = solved_decoupled();
    let red = s.reduced_row_activity(&[]).expect("reduced rows");
    assert!(red.status.is_empty() && red.q_reduced.is_empty() && red.row.is_empty());
    assert!(red.mu > 0.0, "mu is still the converged one: {:e}", red.mu);
}

/// The classification must not depend on the sign of the objective
/// scale, and the reported quantities must keep the sign the
/// natural-units contract gives them -- the gh#763 defect
/// `the_classification_is_unmoved_by_a_negative_objective_scale`
/// pinned for variables, on the row path.
///
/// `compute` runs the rule on the df-in `Sigma` -- the internal `v/s`,
/// which is non-negative whatever `df` is -- and divides the objective
/// scale out only on export. A refinement that classified the
/// *natural* `Sigma` instead would hand the rule a negative ratio
/// under `obj_scaling_factor < 0` and read `g1`, holding `x0` hard
/// against its bound, as INACTIVE.
#[test]
fn the_row_classification_is_unmoved_by_a_negative_objective_scale() {
    let plus = solved_decoupled();
    let minus = solved_decoupled_maximizing();

    let rep_p = plus.classify_activity().expect("activity report");
    let rep_m = minus.classify_activity().expect("activity report");
    let red_p = plus.reduced_row_activity(&[1, 2]).expect("reduced rows");
    let red_m = minus.reduced_row_activity(&[1, 2]).expect("reduced rows");

    for (k, j) in [1usize, 2].into_iter().enumerate() {
        assert_eq!(
            rep_p.row_status[j], rep_m.row_status[j],
            "report status for g{j} moved with the objective sign",
        );
        assert_eq!(
            red_m.status[k], rep_m.row_status[j],
            "g{j}: reduced status {} disagrees with the report's {} under \
             obj_scaling_factor = -1",
            red_m.status[k], rep_m.row_status[j],
        );
        assert_eq!(
            red_m.status[k], red_p.status[k],
            "g{j}: reduced status moved with the objective sign",
        );
        assert_rel(
            &format!("g{j} reduced ratio under negative df"),
            red_m.ratio[k],
            red_p.ratio[k],
            1e-6,
        );

        // ...while the REPORTED quantities keep the natural-units
        // sign, as `row_sigma` does. Asserting this pins the fix to
        // the classification input and stops it being "take an abs
        // somewhere", which would corrupt the curvature the caller
        // reads.
        assert!(
            red_m.sigma[k] < 0.0 && red_p.sigma[k] > 0.0,
            "g{j}: sigma must carry the natural-units sign (got {} at df=-1, {} at df=+1)",
            red_m.sigma[k],
            red_p.sigma[k],
        );
        assert_rel(
            &format!("g{j} sigma magnitude under negative df"),
            red_m.sigma[k].abs(),
            red_p.sigma[k].abs(),
            1e-6,
        );
        assert_rel(
            &format!("g{j} q_reduced under negative df"),
            red_m.q_reduced[k],
            -red_p.q_reduced[k],
            1e-6,
        );
        assert_eq!(
            red_m.q_sign[k], -red_p.q_sign[k],
            "g{j}: q_sign must flip with the declared problem's curvature",
        );
    }
}

// ---------------------------------------------------------------------
// The coupled row: the branch the decoupled fixture cannot reach
// ---------------------------------------------------------------------

/// Diagonal curvature of both coordinates in [`CoupledKinkRowTnlp`].
const H_DIAG: Number = 1.0;
const M_DIAG: Number = 1.0;
/// Reduced curvature along the kink row's direction, chosen so the
/// report's ratio is `1e-2`: a decade inside the ambiguous class.
const RHO: Number = 1.0e-2;
/// Gradient scale of the kink row. Not 1, again so the unit-normal
/// conversions cannot pass by being identities.
const K_SCALE: Number = 2.0;
/// Coupling of the pin into the kink coordinate.
const A: Number = 0.25;

/// Cross term giving reduced curvature `rho`: `h - c^2/m = rho`.
fn cross(rho: Number) -> Number {
    (M_DIAG * (H_DIAG - rho)).sqrt()
}

/// ```text
/// min  0.5*h*k^2 + c*k*y + 0.5*m*y^2 - A*p*k
/// s.t. g0: p == 0                     (equality, ahead of the d block)
///      g1: K_SCALE * k >= 0           (the kink row)
///      k, y, p unbounded
/// ```
///
/// The row analogue of `sens_invariance_legs.rs`'s `CoupledKinkTnlp`:
/// at `p = 0` the reduced gradient along the row's direction is zero
/// and its multiplier vanishes with `mu`, so `g1` is a genuine kink --
/// but the direction is coupled to the free partner `y`.
///
/// Eliminating `y` from `[[h, c], [c, m]]` leaves `h - c^2/m = rho`,
/// which is what generates the multiplier, while the row's own
/// directional curvature stays `h`. So the report's ratio is
/// `rho/h` -- AMBIGUOUS at `RHO` -- and the reduced one is `1`.
/// `rho = H_DIAG` is decoupled, and driving it toward `0` drives
/// `c^2/(h*m)` toward `1`.
struct CoupledKinkRowTnlp {
    rho: Number,
    /// `g_scaling[1]` under `nlp_scaling_method=user-scaling`; `None`
    /// declines scaling entirely.
    g_scale: Option<Number>,
}

impl TNLP for CoupledKinkRowTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(dg) = self.g_scale else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = false;
        *req.use_g_scaling = true;
        req.g_scaling[0] = 1.0;
        req.g_scaling[1] = dg;
        true
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..3 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 1.0e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.0;
        sp.x[2] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let (k, y, p) = (x[0], x[1], x[2]);
        Some(0.5 * H_DIAG * k * k + cross(self.rho) * k * y + 0.5 * M_DIAG * y * y - A * p * k)
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let (k, y, p) = (x[0], x[1], x[2]);
        g[0] = H_DIAG * k + cross(self.rho) * y - A * p;
        g[1] = cross(self.rho) * k + M_DIAG * y;
        g[2] = -A * k;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = K_SCALE * x[0];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1]);
                jcol.copy_from_slice(&[2 as Index, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = K_SCALE;
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (k,k), (y,k), (y,y), (p,k)
                irow.copy_from_slice(&[0 as Index, 1, 1, 2]);
                jcol.copy_from_slice(&[0 as Index, 0, 1, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * H_DIAG;
                values[1] = obj_factor * cross(self.rho);
                values[2] = obj_factor * M_DIAG;
                values[3] = -obj_factor * A;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved_coupled_row(rho: Number, g_scale: Option<Number>) -> Solver {
    let mut app = app_with_defaults();
    if g_scale.is_some() {
        app.options_mut()
            .set_string_value("nlp_scaling_method", "user-scaling", true, false)
            .unwrap();
    }
    run(
        app,
        Rc::new(RefCell::new(CoupledKinkRowTnlp { rho, g_scale })),
    )
}

/// The precondition the disagreement rests on, asserted rather than
/// assumed: the coupled fixture's row really is a kink, and the
/// report really does land it in the AMBIGUOUS class rather than the
/// certified one. If a classifier change ever moves it, this fails
/// rather than letting the test below pass vacuously against the
/// wrong branch.
///
/// AMBIGUOUS here is the DIRECTIONAL normalizer's verdict, and it is a
/// mislabeling of a genuine kink (gh#804) -- pinned, not endorsed.
#[test]
fn the_coupled_row_fixture_carries_an_ambiguous_kink() {
    let s = solved_coupled_row(RHO, None);
    let rep = s.classify_activity().expect("activity report");

    assert_eq!(
        rep.row_status[1], AMBIGUOUS,
        "the coupled row kink must land in the AMBIGUOUS class (got {}, \
         WEAKLY_ACTIVE is {}); the ratio is {:e}",
        rep.row_status[1], WEAKLY_ACTIVE, rep.row_ratio[1]
    );
    // The ratio is `reduced/directional` up to the barrier's own
    // finite `mu`; the point is the decade it sits in, not the last
    // digit.
    assert!(
        (rep.row_ratio[1] - RHO / H_DIAG).abs() < 1e-3 * RHO,
        "the ratio is reduced/directional = {:e}, got {:e}",
        RHO / H_DIAG,
        rep.row_ratio[1]
    );
}

/// The headline of gh#804, the row half of gh#763's table: four solves
/// that are the SAME row kink, differing only in how strongly the
/// row's direction is coupled to its free partner.
///
/// `classify_activity` divides the geometric weight by the curvature
/// along the row's own gradient, so its ratio is
/// `reduced/directional` and tracks the coupling: the bottom two rows
/// fall out of the `[1e-1, 1e1]` band and read AMBIGUOUS. No tolerance
/// recovers them -- the ratio is `mu`-independent, so a tighter solve
/// reports the same thing. `reduced_row_activity` divides by the
/// curvature that generates the multiplier, so its ratio is `1` at
/// every coupling and all four certify.
#[test]
fn the_reduced_normalizer_certifies_a_coupled_row_kink_at_every_coupling() {
    for rho in [1.0, 1.0e-1, 1.0e-2, 1.0e-3] {
        let s = solved_coupled_row(rho, None);
        let rep = s.classify_activity().expect("activity report");
        let red = s.reduced_row_activity(&[1]).expect("reduced rows");

        // What the directional normalizer says: the ratio IS the
        // coupling, and it is blind to `K_SCALE`.
        assert!(
            (rep.row_ratio[1] - rho / H_DIAG).abs() < 1e-3 * rho,
            "rho {rho:e}: the directional ratio is reduced/directional, got {:e}",
            rep.row_ratio[1]
        );
        let directional_class = if rho >= 1.0e-1 {
            WEAKLY_ACTIVE
        } else {
            AMBIGUOUS
        };
        assert_eq!(
            rep.row_status[1], directional_class,
            "rho {rho:e}: the report's class should follow the coupling",
        );

        // What the reduced normalizer says: the same kink, every time.
        assert_rel(
            &format!("rho {rho:e}: reduced curvature along the unit normal"),
            red.q_reduced[0],
            rho,
            1e-3,
        );
        assert_rel(
            &format!("rho {rho:e}: reduced ratio"),
            red.ratio[0],
            1.0,
            1e-3,
        );
        assert_eq!(
            red.status[0], WEAKLY_ACTIVE,
            "rho {rho:e}: a genuine row kink must certify at every coupling \
             (ratio {:e}, report said {})",
            red.ratio[0], rep.row_status[1],
        );
    }
}

/// Row-scaling leg (gh#804). The change of variables is not the only
/// scaling axis a `user-scaling` solve moves: the constraint rows
/// carry their own factors, and unlike the variable accessor -- where
/// `leg_scaling_the_reduced_curvature_is_unmoved_by_a_row_scaling`
/// pins an invariance that holds by construction, the natural-units
/// conjugation carrying no `dg` into the `x` block at all -- the row
/// accessor has real arithmetic to get right. Three separate `dg`
/// factors meet in one ratio: `Sigma` is exported as
/// `sigma * dg^2 / df`, the back-solved `(K^-1)_ss` carries `dg^-2`
/// through the conjugation, and `||grad_d||^2` is gathered in the
/// frame that still has `dg` in it. Drop or double any one of the
/// three and a `dg` of `1e3` moves the answer by six orders.
///
/// The leg sweeps six decades of `dg` and splits its tolerances by
/// what each quantity actually depends on, measured rather than
/// assumed:
///
/// * The statuses and the curvature sign are EXACT across the sweep.
/// * `q_reduced` is the model's own geometry, independent of where
///   the barrier stopped, and is bit-stable to ~1e-16.
/// * `Sigma = v/s` and every ratio built on it track the converged
///   iterate, and the solver's convergence test lives in SCALED
///   space, so how many digits of the natural-units `Sigma` are
///   pinned degrades in proportion to `dg`: ~6e-9 relative at
///   `dg = 1e1`, ~6e-7 at `dg = 1e3`, and tightening `tol` by three
///   decades moves that by only 4x. That is the solve, not the
///   accessor -- and it is seven decades below the `1e3` a single
///   stray factor of `dg` would move the same number.
#[test]
fn leg_scaling_the_reduced_row_curvature_is_unmoved_by_a_row_scaling() {
    /// Four decades above the ~1e-16 the geometry actually reproduces.
    const Q_RTOL: Number = 1.0e-12;
    /// Two decades above the ~6e-7 the scaled-space convergence test
    /// leaves on `Sigma` at `dg = 1e3`, and seven below a one-factor
    /// leak.
    const SIGMA_RTOL: Number = 1.0e-4;

    let base = solved_coupled_row(RHO, Some(1.0));
    let rb = base.reduced_row_activity(&[1]).expect("reduced rows");
    let pb = base.classify_activity().expect("activity report");

    // the kink really is the coupled one this fixture is about, so the
    // leg cannot pass by classifying nothing
    assert_eq!(
        rb.status[0], WEAKLY_ACTIVE,
        "fixture drifted: g1 should certify as a kink on the reduced normalizer",
    );
    assert_eq!(
        pb.row_status[1], AMBIGUOUS,
        "fixture drifted: the report should still read the coupled row kink as ambiguous",
    );

    for dg in [1.0e-3, 1.0e-1, 1.0e1, 1.0e3] {
        let scaled = solved_coupled_row(RHO, Some(dg));
        let rs = scaled.reduced_row_activity(&[1]).expect("reduced rows");
        let ps = scaled.classify_activity().expect("activity report");

        assert_eq!(
            rb.status[0], rs.status[0],
            "g1: reduced status moved under a row scaling of {dg:e}",
        );
        assert_eq!(
            rb.q_sign[0], rs.q_sign[0],
            "g1: reduced curvature sign moved under a row scaling of {dg:e}",
        );
        assert_eq!(
            pb.row_status[1], ps.row_status[1],
            "g1: report status moved under a row scaling of {dg:e}",
        );
        assert_rel_strict(
            &format!("g1 reduced curvature at dg={dg:e}"),
            rs.q_reduced[0],
            rb.q_reduced[0],
            Q_RTOL,
        );
        assert_rel_strict(
            &format!("g1 reduced ratio at dg={dg:e}"),
            rs.ratio[0],
            rb.ratio[0],
            SIGMA_RTOL,
        );
        assert_rel_strict(
            &format!("g1 sigma at dg={dg:e}"),
            rs.sigma[0],
            rb.sigma[0],
            SIGMA_RTOL,
        );
        assert_rel_strict(
            &format!("g1 report ratio at dg={dg:e}"),
            ps.row_ratio[1],
            pb.row_ratio[1],
            SIGMA_RTOL,
        );
    }
}

/// Truly relative agreement, unlike [`assert_rel`]'s `max(1.0)` floor:
/// the leg's quantities span `2.5e-3` to `1e0`, and a floor would
/// hold the small ones to an absolute bound and quietly weaken the
/// leg by the same factor.
fn assert_rel_strict(what: &str, got: Number, want: Number, rtol: Number) {
    let err = (got - want).abs() / want.abs();
    assert!(
        err <= rtol,
        "{what}: got {got:.17e}, want {want:.17e} (rel {err:e} > {rtol:e})"
    );
}

/// A relaxed solve shifts the slacks every one of these quantities is
/// read from, so the accessor refuses it rather than answering off a
/// perturbed bound -- the same guard `classify_activity` and
/// `reduced_activity` carry.
#[test]
fn a_relaxed_solve_is_refused() {
    let mut app = app_with_defaults();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 1e-8, true, false)
        .unwrap();
    let s = run(
        app,
        Rc::new(RefCell::new(DecoupledRowTnlp { negated: false })),
    );
    match s.reduced_row_activity(&[1]) {
        Err(SolverError::BadOptions(m)) => {
            assert!(
                m.contains("reduced_row_activity requires bound_relax_factor=0"),
                "{m}"
            );
        }
        other => panic!("a relaxed solve must be refused, got {other:?}"),
    }
}

/// The identity the implementation rests on, checked against the
/// alternative the issue proposed rather than left as prose.
///
/// gh#804 suggested driving the `x` block with the row's gradient:
/// `grad_d^T K^-1 grad_d`. The implementation drives the `s` block
/// with a unit vector instead, because the row's own value IS that
/// coordinate -- the slack the barrier acts on, tied to the model by
/// `d_j(x) = s_j` -- which needs no gradient assembled into the
/// right-hand side. The two are the same number for every `j`, since
/// `grad_d` reaches the system only through the row it defines: the
/// `s` row's elimination sends `M - M*Sigma_s*(I + M*Sigma_s)^-1*M` to
/// `(I + M*Sigma_s)^-1*M` either way, with `M = A W A^T`.
///
/// This fixture is where that has content: the row's gradient is
/// `K_SCALE * e_k`, the coordinate is coupled to a free partner, and
/// there is an equality in the system, so the two routes agree only if
/// the algebra actually holds rather than by both reducing to `H^-1`.
#[test]
fn the_slack_unit_vector_and_the_row_gradient_give_the_same_back_solve() {
    let s = solved_coupled_row(RHO, None);
    let dim = s.kkt_dim().expect("converged factor");
    let dims = s.block_dims().expect("converged factor");
    let (n_x, s_off) = (dims[0], dims[0]);
    // g1 is the second user row and the first (only) inequality
    let d_pos = 0usize;

    // route A: unit right-hand side in the `s` block
    let mut rhs = vec![0.0; dim];
    rhs[s_off + d_pos] = 1.0;
    let mut lhs = vec![0.0; dim];
    s.kkt_solve(&rhs, &mut lhs).expect("back-solve");
    let by_slack = lhs[s_off + d_pos];

    // route B: the row's own gradient in the `x` block, in natural
    // units and mapped full-x -> var-x like every other factor row
    let grad = s.row_normal(1).expect("row normal");
    let cols: Vec<Index> = (0..grad.len() as Index).collect();
    let var_rows = s.x_primal_rows(&cols).expect("primal rows");
    let mut rhs = vec![0.0; dim];
    for (full, row) in var_rows.iter().enumerate() {
        if let Some(r) = row {
            assert!((*r as usize) < n_x, "var-x row inside the x block");
            rhs[*r as usize] = grad[full];
        }
    }
    let mut lhs = vec![0.0; dim];
    s.kkt_solve(&rhs, &mut lhs).expect("back-solve");
    let by_gradient: Number = var_rows
        .iter()
        .enumerate()
        .filter_map(|(full, row)| row.map(|r| grad[full] * lhs[r as usize]))
        .sum();

    // the fixture must not make this trivially true
    assert!(
        by_slack.abs() > 1e-6,
        "fixture drifted: the back-solve should be a real number, got {by_slack:e}"
    );
    assert_rel_strict(
        "grad_d^T K^-1 grad_d against (K^-1)_ss",
        by_gradient,
        by_slack,
        1e-9,
    );
}
