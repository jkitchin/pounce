//! gh#852: the walk has to be able to take a weakly active bound BACK.
//!
//! A weakly active bound is in the factorization with a sigma of order
//! ONE, not order `1/mu`. That is the whole content of the kink: the
//! bound is carried as a finite penalty, so it bends the direction
//! without enforcing anything, and a direction that presses into it
//! walks the variable straight out of its box. `step_along_path` used
//! to bar every base-active bound from its reach scan, on the correct
//! reasoning that a strongly held bound cannot be reached — its
//! variable is already on it and cannot move — and the weak rows went
//! out with it. On the holding side of the coupled kink below the walk
//! then found no breakpoint at all: zero segments, `x` outside its
//! lower bound, and a caller left to clamp. A clamp moves the crossing
//! coordinate only, so `y`, which the equality ties to it, kept the
//! one-sided value 2/7 against the re-solve's 1.
//!
//! The fix narrows the exclusion to bounds the classifier certifies as
//! strongly active, so this file has to reach BOTH sides of that
//! branch — the rule `sens_invariance_legs.rs` states and gh#756 paid
//! for. Three tests, one per branch the reach scan can now take:
//!
//! 1. weak bound, direction presses IN: the walk re-holds it and the
//!    coupled neighbour follows (the defect);
//! 2. weak bound, direction presses OUT: the walk still releases it,
//!    and the new reach event does not pre-empt that release;
//! 3. strongly active bound, direction presses IN: the exclusion is
//!    still in force, nothing is re-held, and the answer is unmoved.
//!
//! A re-held weak row also leaves the factorization, and that half of
//! the fix is invisible on the three above: while the hold stands, the
//! row's order-one sigma multiplies a coordinate the hold has already
//! taken to zero, so releasing it changes nothing anyone can measure.
//! It becomes measurable one breakpoint later, where the hold DROPS
//! and the coordinate moves again — a stale sigma damps it. Test 4 is
//! a second model built to reach that fraction, and it is the only
//! thing in the crate that does.
//!
//! Mutation table:
//!
//! | change | red |
//! |---|---|
//! | drop `weak_rows` from `factor_holds` (restore the old exclusion) | 1 |
//! | drop the base-activity test from `factor_holds` entirely | 3 |
//! | skip the release of the re-held row in the reach handler | 4 |
//! | release on `base_active_row` instead of on `weak_rows` | 4 |
//!
//! Every number here is checked against a re-solve at the perturbed
//! parameter, which for these QPs is the exact answer.
//!
//! What this file is NOT evidence about:
//!
//! * **Scaling.** Both models run unit-scaled. The branch the reach
//!   scan takes is a set membership plus the sign of `d[i]`, and
//!   `sens_invariance_legs.rs` leg 1 already pins that the weak set
//!   itself is unmoved by a change of variables; the Python side
//!   carries a `user-scaling` arm on the same coupled kink.
//! * **The magnitude of test 1's answer.** At an exact kink the
//!   converged base point sits `O(sqrt(mu))` off the solution, which
//!   here is 8.5e-6 — inside that test's 1e-5 tolerance. So a
//!   predictor that returned the base point unchanged would clear the
//!   two `x`/`y` bounds there. It would not clear the assertion that
//!   `y` moved off 2/7, and it would not clear the record assertion,
//!   which is what carries that test. Tests 2 and 4 have no such
//!   margin: their answers are 0.71 and 0.26 against a base of ~1e-5.
//! * **Index spaces.** Both models keep the parameter in the x block,
//!   so var-x rows and model columns coincide. `cd_split_pin_mapping.rs`
//!   and `sens_invariance_legs.rs` leg 3 own that dimension. (The one
//!   new cross-space read, `WeakBound::row` against a bound-multiplier
//!   row, is checked by mutation: swapping it for `var_row` in either
//!   caller turns tests 1 and 4 red.)

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::Solver;

/// The kink, coupled to an interior variable through an equality, with
/// the parameter carried as a third variable pinned by one:
///
/// ```text
/// min  (x - p)^2 + 0.1 (y - 1)^2
/// s.t. g0 = p            (the pin)
///      g1: y - 2x - 1 = 0
///      0 <= x <= 10,  -50 <= y <= 50
/// ```
///
/// Along the equality the objective is `(x - p)^2 + 0.4 x^2`, so the
/// unconstrained minimizer is `x = p / 1.4` and the solution is
///
/// ```text
/// p <= 0:  x = 0,        y = 1
/// p >= 0:  x = p / 1.4,  y = 2 p / 1.4 + 1
/// ```
///
/// At `p = 0` those two branches meet: `x` sits on its lower bound
/// with a vanishing multiplier, the canonical kink, and the coupling
/// is what makes the defect visible — a clamp can put `x` back on its
/// bound, but only a re-optimization moves `y` with it.
struct CoupledKink {
    p_nominal: Number,
}

impl TNLP for CoupledKink {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 3,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = -50.0;
        b.x_u[1] = 50.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = self.p_nominal;
        b.g_u[0] = self.p_nominal;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 2.0;
        sp.x[2] = self.p_nominal;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (xx, y, p) = (x[0], x[1], x[2]);
        Some((xx - p) * (xx - p) + 0.1 * (y - 1.0) * (y - 1.0))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (xx, y, p) = (x[0], x[1], x[2]);
        g[0] = 2.0 * (xx - p);
        g[1] = 0.2 * (y - 1.0);
        g[2] = -2.0 * (xx - p);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = x[1] - 2.0 * x[0] - 1.0;
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 3] = [0, 1, 1];
                let cs: [Index; 3] = [2, 0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = -2.0;
                values[2] = 1.0;
            }
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
        // Lower triangle. Both constraints are linear, so lambda
        // contributes nothing and only the objective appears.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 4] = [0, 1, 2, 2];
                let cs: [Index; 4] = [0, 1, 0, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 0.2 * obj_factor;
                values[2] = -2.0 * obj_factor;
                values[3] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn configured() -> IpoptApplication {
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
    // The activity classifier reads slacks, and a relaxed solve shifts
    // them, which makes `weakly_active_bounds` return nothing and the
    // fix under test inert. The corresponding Python surface takes the
    // same care.
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn solved_at(p: Number) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(CoupledKink { p_nominal: p })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at p={p} failed: {status:?}",
    );
    solver
}

/// `(x, y)` at the exact solution for `p`.
fn resolve_at(p: Number) -> [Number; 2] {
    let solver = solved_at(p);
    let x = solver.converged().expect("converged state").x.clone();
    [x[0], x[1]]
}

/// Solve at `p0`, then walk and repair to `p0 + dp`. Returns the
/// walk's `(x, y)`, the base-point repair's `(x, y)`, and the walk's
/// breakpoint record.
fn walk_and_repair(
    p0: Number,
    dp: Number,
) -> (
    [Number; 2],
    [Number; 2],
    Vec<pounce_sensitivity::boundcheck::PathSegment>,
) {
    let solver = solved_at(p0);
    let base = solver.converged().expect("converged state").x.clone();
    let (fixed, _, _) = solver
        .parametric_step_bounded(&[0], &[dp], 8, None)
        .expect("parametric_step_bounded");
    let (walked, segs) = solver
        .parametric_step_path(&[0], &[dp], 8)
        .expect("parametric_step_path");
    (
        [base[0] + walked[0], base[1] + walked[1]],
        [base[0] + fixed[0], base[1] + fixed[1]],
        segs,
    )
}

/// The var-x row of `x`, whose bound is the one in question. The pin
/// variable is not removed from the x block here, so var-x rows and
/// model columns coincide; the file asserts on row 0 only, which is
/// `x` under either reading.
const X_ROW: usize = 0;

#[test]
fn the_walk_reholds_a_weak_bound_the_perturbation_presses_into() {
    // The defect. dp = -1 from the kink: the true solution keeps x on
    // its bound and leaves y at 1. Before the fix the walk reported no
    // breakpoint, left x at -5/14, and y at 2/7.
    let truth = resolve_at(-1.0);
    let (walked, fixed, segs) = walk_and_repair(0.0, -1.0);

    assert!(
        (truth[0]).abs() < 1e-6 && (truth[1] - 1.0).abs() < 1e-6,
        "the re-solve should hold the bound: {truth:?}",
    );
    let err = (walked[0] - truth[0])
        .abs()
        .max((walked[1] - truth[1]).abs());
    assert!(
        err < 1e-5,
        "the walk should reproduce the re-solve, off by {err}: \
         walk {walked:?} against {truth:?} (segments {segs:?})",
    );
    // What made the wrong answer wrong: y, not x. A clamp gets x right
    // on its own, so an assertion on x alone passes over the defect.
    assert!(
        (walked[1] - 1.0).abs() < 1e-5,
        "the coupled neighbour has to follow the re-held bound, got y = {} \
         (the pre-fix value was 2/7 = {})",
        walked[1],
        2.0 / 7.0,
    );
    // The repair mode was always right here; the walk now agrees with it.
    let gap = (walked[0] - fixed[0])
        .abs()
        .max((walked[1] - fixed[1]).abs());
    assert!(
        gap < 1e-5,
        "path and fix_relax should agree on this model, apart by {gap}: \
         walk {walked:?}, repair {fixed:?}",
    );
    // The re-hold is a breakpoint, and the record has to name it: the
    // working set gained a hold it did not have at the base point.
    let reheld: Vec<_> = segs
        .iter()
        .filter(|s| s.var_row == X_ROW && s.pinned && s.lower)
        .collect();
    assert_eq!(
        reheld.len(),
        1,
        "one re-hold of x's lower bound expected, record {segs:?}",
    );
    assert!(
        reheld[0].at < 1e-3,
        "at an exact kink the bound is taken essentially at once, got {}",
        reheld[0].at,
    );
}

#[test]
fn the_walk_still_releases_a_weak_bound_the_perturbation_leaves() {
    // The other side of the same kink, and the branch the new reach
    // event must not pre-empt: dp = +1 frees x, and the record has to
    // read "leaves", not "reaches".
    let truth = resolve_at(1.0);
    let (walked, _, segs) = walk_and_repair(0.0, 1.0);

    assert!(
        (truth[0] - 1.0 / 1.4).abs() < 1e-6,
        "the re-solve should leave the bound: {truth:?}",
    );
    let err = (walked[0] - truth[0])
        .abs()
        .max((walked[1] - truth[1]).abs());
    assert!(
        err < 1e-5,
        "the walk should reproduce the re-solve, off by {err}: \
         walk {walked:?} against {truth:?} (segments {segs:?})",
    );
    let departures: Vec<_> = segs.iter().filter(|s| !s.pinned).collect();
    assert_eq!(
        departures.len(),
        1,
        "one departure expected, record {segs:?}",
    );
    assert_eq!(departures[0].var_row, X_ROW, "record {segs:?}");
    assert!(
        !segs.iter().any(|s| s.pinned),
        "nothing should be re-held on the releasing side, record {segs:?}",
    );
}

#[test]
fn a_strongly_active_bound_is_still_barred_from_the_reach_scan() {
    // The branch the exclusion still covers. Held at p = -0.5 the same
    // bound is strongly active, multiplier 1.0 against a slack of order
    // mu, and pressing further in changes nothing: the factorization's
    // order-1/mu sigma is what holds the variable, and a Schur hold on
    // top of it would enforce the bound twice through a near-singular
    // complement. Narrowing the exclusion to weak rows has to leave
    // this alone.
    let truth = resolve_at(-1.0);
    let (walked, _, segs) = walk_and_repair(-0.5, -0.5);

    assert!(
        walked[0].abs() < 1e-6 && (walked[1] - 1.0).abs() < 1e-6,
        "the held solution is unmoved by pressing further into the bound, \
         got {walked:?} against {truth:?}",
    );
    assert!(
        segs.is_empty(),
        "a strongly active bound is not a breakpoint: nothing enters and \
         nothing leaves, record {segs:?}",
    );
}

/// The second model: a weak bound the walk re-holds and then DROPS,
/// which is the only place the release half of the fix is visible.
///
/// ```text
/// min  0.5 x^2 + 0.5 w^2 + 0.8 x w - p (0.5 x + w)
/// s.t. g0 = p             (the pin)
///      0 <= x <= 10,  -50 <= w <= 0.3
/// ```
///
/// The Hessian `[[1, 0.8], [0.8, 1]]` is positive definite, so at
/// `p = 0` the minimizer is the origin: `x` on its lower bound with a
/// vanishing multiplier — the same kink as above — and `w` strictly
/// inside its box.
///
/// Toward `p = 1` the walk crosses three breakpoints, each of them
/// hand-computable:
///
/// * fraction 0: the unconstrained direction is
///   `(0.5 - 0.8) / (1 - 0.64) < 0` in `x`, so the perturbation
///   presses into the weak bound and the walk re-holds it.
/// * fraction 0.3: with `x` held, `w` tracks `p` one for one and
///   reaches its upper bound `0.3`.
/// * fraction 0.48: with `w` held at `0.3`, stationarity in `x` wants
///   `x = 0.5 p - 0.8 (0.3)`, which reaches zero at `p = 0.48`. The
///   hold's multiplier crosses zero there and the bound is dropped.
///
/// Past that last fraction `x` is free and rises at `dx/dp = 0.5` to
/// `0.26` at `p = 1`. That rate is the assertion: with the row
/// released it is `0.5`; with the row's weak sigma left in the factor
/// it is `0.5 / (1 + sigma)`, and the walk lands at 0.191 instead —
/// an answer that is wrong by 26% while every breakpoint in the
/// record is still at the right fraction.
struct DroppedKink {
    p_nominal: Number,
}

const XW_COUPLING: Number = 0.8;
const X_PULL: Number = 0.5;
const W_UPPER: Number = 0.3;

impl TNLP for DroppedKink {
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
        b.x_u[0] = 10.0;
        b.x_l[1] = -50.0;
        b.x_u[1] = W_UPPER;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = self.p_nominal;
        b.g_u[0] = self.p_nominal;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.1;
        sp.x[1] = 0.1;
        sp.x[2] = self.p_nominal;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (xx, w, p) = (x[0], x[1], x[2]);
        Some(0.5 * xx * xx + 0.5 * w * w + XW_COUPLING * xx * w - p * (X_PULL * xx + w))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (xx, w, p) = (x[0], x[1], x[2]);
        g[0] = xx + XW_COUPLING * w - p * X_PULL;
        g[1] = w + XW_COUPLING * xx - p;
        g[2] = -(X_PULL * xx + w);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
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
                values[2] = obj_factor * XW_COUPLING;
                values[3] = -obj_factor * X_PULL;
                values[4] = -obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn dropped_kink_solver(p: Number) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(DroppedKink { p_nominal: p })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at p={p} failed: {status:?}",
    );
    solver
}

#[test]
fn a_re_held_weak_row_leaves_the_factor_so_the_drop_moves_at_the_right_rate() {
    let truth = {
        let s = dropped_kink_solver(1.0);
        let x = s.converged().expect("converged state").x.clone();
        [x[0], x[1]]
    };
    assert!(
        (truth[0] - 0.26).abs() < 1e-6 && (truth[1] - W_UPPER).abs() < 1e-6,
        "the re-solve should be (0.26, 0.3): {truth:?}",
    );

    let solver = dropped_kink_solver(0.0);
    let base = solver.converged().expect("converged state").x.clone();
    let (walked, segs) = solver
        .parametric_step_path(&[0], &[1.0], 8)
        .expect("parametric_step_path");
    let end = [base[0] + walked[0], base[1] + walked[1]];

    // The record first, so a failure says which breakpoint moved.
    let record: Vec<(Number, usize, bool)> =
        segs.iter().map(|s| (s.at, s.var_row, s.pinned)).collect();
    assert_eq!(
        segs.len(),
        3,
        "re-hold, w's bound, then the drop: three breakpoints, got {record:?}",
    );
    assert!(
        segs[0].var_row == X_ROW && segs[0].pinned && segs[0].at < 1e-3,
        "first: the weak bound is re-held essentially at once, {record:?}",
    );
    assert!(
        segs[1].var_row == 1 && segs[1].pinned && (segs[1].at - 0.3).abs() < 1e-3,
        "second: w reaches its upper bound at 0.3, {record:?}",
    );
    assert!(
        !segs[2].pinned && segs[2].var_row == X_ROW && (segs[2].at - 0.48).abs() < 1e-3,
        "third: the hold on x drops at 0.48, {record:?}",
    );

    // And the number the release buys. Leaving the weak row in the
    // factor damps x over the last 0.52 of the path and lands it at
    // 0.191; the assertion is tight enough to separate the two.
    let err = (end[0] - truth[0]).abs().max((end[1] - truth[1]).abs());
    assert!(
        err < 1e-4,
        "the walk should reproduce the re-solve past the drop, off by {err}: \
         walk {end:?} against {truth:?} (the un-released value is x = 0.191)",
    );
}

#[test]
fn the_fixtures_take_different_branches() {
    // The three reach-scan branches are told apart by the activity
    // classifier and by nothing else, so this asserts the split rather
    // than assuming it. A fixture that drifted into its partner's
    // class would take the evidence with it and leave every test in
    // this file green — which is the failure `sens_invariance_legs.rs`
    // describes and gh#756 shipped.
    let kink = solved_at(0.0);
    let weak = kink.weakly_active_bounds().expect("classify at the kink");
    assert_eq!(
        weak.len(),
        1,
        "the kink has exactly one weakly active bound: {weak:?}",
    );
    assert!(
        weak[0].var_row == X_ROW && weak[0].lower,
        "and it is x's lower bound: {weak:?}",
    );

    let held = solved_at(-0.5);
    let strong = held.weakly_active_bounds().expect("classify off the kink");
    assert!(
        strong.is_empty(),
        "at p = -0.5 the same bound is strongly active, which is the only \
         thing that makes the third test a test: {strong:?}",
    );

    let dropped = dropped_kink_solver(0.0);
    let weak_dropped = dropped
        .weakly_active_bounds()
        .expect("classify the drop model");
    assert!(
        weak_dropped.iter().any(|w| w.var_row == X_ROW && w.lower),
        "the drop model's x bound has to be weak too, or the fourth test \
         never reaches the reach scan's new branch: {weak_dropped:?}",
    );
}
