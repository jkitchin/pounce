//! gh#928: the walk has to notice when its own answer leaves the box.
//!
//! gh#852 taught `step_along_path` to keep watching a bound whose
//! sigma is order ONE rather than order `1/mu` — one the factorization
//! carries as a finite penalty, so it bends the direction and enforces
//! nothing. The walk learns which those are from `weak_rows`, which
//! `Solver::weakly_active_bounds` fills from the activity classifier.
//!
//! That leaves the door the classifier cannot see through. `classify`
//! needs a curvature to divide by, so where the Hessian diagonal falls
//! under the identification floor every bound comes back
//! `UNIDENTIFIED`, `weakly_active_bounds` returns an empty list, and
//! the walk is told nothing. Meanwhile its own base-activity table
//! marks the bound enforced on a test that is exactly `Sigma > 1`, and
//! the reach scan skips it. The bound is then neither held by the
//! factorization nor watched by the path: the direction carries the
//! variable straight through it, zero breakpoints are recorded, and
//! the caller is handed a point outside the box with no warning.
//!
//! An LP is the whole of that class — a zero Hessian is as
//! unidentifiable as a diagonal gets — and so is any model whose cost
//! is linear in the coordinate that reaches the bound, which is what
//! made this reachable from a production AC-OPF: at linear generation
//! cost the walk put a generator 33 MW past its nameplate rating, and
//! adding a quadratic term worth 0.1% of the linear one fixed it by
//! lifting the diagonal over the floor and nothing else.
//!
//! The fix does not extend the classification. It cannot: the
//! quantity that would discriminate inside `UNIDENTIFIED` is the
//! curvature that is by hypothesis unmeasurable here, and admitting
//! the class wholesale also admits `y`'s genuinely enforced bound
//! below (`UNIDENTIFIED` at `Sigma = 1.1e11`) and makes the augmented
//! system singular. Instead the walk checks its own answer against the
//! box and treats a base-active bound the answer CROSSED as measured
//! proof that the factorization did not enforce it.
//!
//! So this file's job is to reach the branch that proof is made on,
//! and the assertions say in so many words which branch that is:
//! every test asserts the classifier returned `UNIDENTIFIED` and an
//! empty weak list, so nothing here is riding on the gh#852 path.
//! `var_sigma` is what separates the tests, because it is the quantity
//! the base-activity test actually compares.
//!
//! Five tests, one per branch:
//!
//! 1. `Sigma > 1`, upper bound, direction presses IN: the exclusion is
//!    in force, nothing told the walk, and the answer must still land
//!    on the bound (the defect);
//! 2. the same on a LOWER bound, which is a different entry of the
//!    base-activity table and a different side of the repair's own
//!    side-selection;
//! 3. `Sigma > 1`, direction presses OUT: the bound genuinely leaves,
//!    the answer is interior, and no repair may fire;
//! 4. `Sigma < 1` on the same model and the same classifier verdict:
//!    the walk never marked the bound active, the reach scan always
//!    watched it, and the answer was always right. The control that
//!    isolates `Sigma > 1` as the trigger rather than `UNIDENTIFIED`.
//! 5. the same model with the walk CAPPED, which is the guard on the
//!    post-loop box check rather than on the repair. `max_iter = 0`
//!    asks for the plain linear step, which on this model is the
//!    defect's own answer — requested deliberately, and legal before
//!    the repair existed, so it still answers rather than failing.
//!
//! Mutation table:
//!
//! | change | red |
//! |---|---|
//! | return `walk_once`'s answer directly (drop the repair loop) | 1, 2 |
//! | read the crossed side off the base point's slack, not the answer | 1, 2 |
//! | hard-code the crossed side to `upper` | 2 |
//! | seed the watch list empty instead of from `weak_rows` | nothing here; gh#852 test 4 |
//! | drop the `max_iter` gate on the post-loop check | 5 only |
//!
//! That last row is the interesting one, and it was measured rather
//! than assumed. Dropping the `weak_rows` seed leaves four of gh#852's
//! five tests green, because the box check rediscovers those bounds by
//! observation — the seed is nearly redundant for them. It is not
//! redundant for their test 4, which is a *rate* error: a stale sigma
//! damps a coordinate at a later breakpoint without ever pushing it
//! out of its box, so there is no violation for a box check to see.
//! The seed catches what the classifier can name; the box check
//! catches what it cannot; neither covers the other, and this file is
//! evidence only about the second.
//!
//! Every answer is checked against a re-solve at the perturbed
//! parameter, which for an LP is exact.
//!
//! What this file is NOT evidence about:
//!
//! * **The failed-re-walk arm.** When a repair pass errors the wrapper
//!   propagates it rather than handing back the answer in hand — that
//!   answer is out of the box, which is why there was a second walk,
//!   and returning it silently is the defect itself. No fixture here
//!   reaches that arm: the repair only ever admits a bound the answer
//!   crossed, and releasing one of those keeps this model
//!   nonsingular. It is a guard, and it is unexercised, so this file
//!   is not evidence that the message it produces is reachable.
//! * **The `!grew` arm and the exhausted cap**, where the answer is
//!   still outside a bound but no base-active row explains it, or the
//!   repair spent `PATH_MAX_BOX_REPAIRS` passes. Both now report
//!   instead of returning the point, because returning it is gh#928's
//!   own signature — a violation with nothing in the record naming
//!   it. Test 5 pins the one case that must NOT report, a walk the
//!   caller capped; no fixture reaches the arms that do.
//! * **How small a violation still fires the repair.** The check
//!   inherits `ctx.eps` — the solve's `bound_relax_factor`, floored at
//!   `1e-9` — which is the crate's existing definition of "outside a
//!   bound" and the one `parametric_step_bounded` already uses. It is
//!   an *absolute length in the model's units*, so on a model whose
//!   variables are scaled to `1e-9` or below the check is vacuous.
//!   The failure mode there is the pre-fix behaviour rather than a new
//!   one, and the alternative — a relative margin — would be a second,
//!   inconsistent definition of the same word. The violations that
//!   motivated the fix are nowhere near it: `1e-2` on a unit box here,
//!   and 33 MW on a 170 MW rating in the AC-OPF.
//! * **The cap.** `PATH_MAX_BOX_REPAIRS` bounds the number of walks,
//!   not the termination, which comes from the watch list growing.
//!   Both models here repair in one pass, so nothing measures a second.
//! * **Constraint rows.** This is a *variable bound*, and the repair
//!   reads `lo` / `hi` over the x block, so it says nothing about a
//!   limit written as a row instead. Measured on this same LP with
//!   `x <= 1` moved from the bound to a constraint: the step lands
//!   `1e-2` past the row with zero segments, and `linear`, `bounded`,
//!   `directional`, `release_all` and `path` all miss it identically.
//!   That is a scope limit of the whole active-set layer rather than
//!   of this repair — path segments are declared in var-x — and it is
//!   filed separately. Do not read a green file here as covering it.
//! * **Index spaces.** The parameter stays in the x block, so var-x
//!   rows and model columns coincide. `cd_split_pin_mapping.rs` and
//!   `sens_invariance_legs.rs` leg 3 own that dimension.
//! * **Scaling.** Both arms run unit-scaled. `sens_invariance_legs.rs`
//!   leg 1 owns it.
//! * **Magnitude.** `dp = 1e-2` against a base slack of `1e-6`, so the
//!   answer is four orders off the base point in `y` and the assertions
//!   have no room to pass on a predictor that returned the base point.

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
use pounce_sensitivity::activity::UNIDENTIFIED;

/// The smallest model that reaches the defect: a linear program with
/// the parameter carried as a third variable pinned by an equality.
///
/// ```text
/// min  s x + 2 y
/// s.t. g0: s x + y - lam = 0
///      g1: lam = lam0            (the pin)
///      x in [0, 1] for s = +1, in [-1, 0] for s = -1;  y >= 0
/// ```
///
/// With `u = s x` in `[0, 1]` this is `min u + 2 y` subject to
/// `u + y = lam`, so `u` is the cheap unit and the solution is
///
/// ```text
/// lam <= 1:  u = lam,  y = 0
/// lam >= 1:  u = 1,    y = lam - 1
/// ```
///
/// The two branches meet at `lam = 1`, where `u` sits on its bound
/// with a vanishing multiplier. `s` chooses which of `x`'s two bounds
/// that is, so the same kink is reachable from either side of the
/// base-activity table.
///
/// The Hessian is identically zero. That is the point: there is no
/// curvature for `classify` to divide by, so every bound in this model
/// is `UNIDENTIFIED` no matter how firmly it is held.
struct PinnedLp {
    /// `+1.0` puts the kink on `x`'s upper bound, `-1.0` on its lower.
    s: Number,
    lam: Number,
}

impl TNLP for PinnedLp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // The kinked bound is x's upper for s = +1 and its lower for
        // s = -1; the other side of the box is slack either way.
        b.x_l[0] = if self.s > 0.0 { 0.0 } else { -1.0 };
        b.x_u[0] = if self.s > 0.0 { 1.0 } else { 0.0 };
        b.x_l[1] = 0.0;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = self.lam;
        b.g_u[1] = self.lam;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5 * self.s;
        sp.x[1] = 0.5;
        sp.x[2] = self.lam;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(self.s * x[0] + 2.0 * x[1])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.s;
        g[1] = 2.0;
        g[2] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.s * x[0] + x[1] - x[2];
        g[1] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 4] = [0, 0, 0, 1];
                let cs: [Index; 4] = [0, 1, 2, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = self.s;
                values[1] = 1.0;
                values[2] = -1.0;
                values[3] = 1.0;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // Everything is linear, so the Lagrangian Hessian is zero. One
        // structural entry, carrying that zero, is what puts every
        // bound in this model below the identification floor.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 0.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn configured() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    for (k, v) in [("print_level", 0)] {
        app.options_mut()
            .set_integer_value(k, v, true, false)
            .unwrap();
    }
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    for (k, v) in [
        ("tol", 1e-10),
        ("constr_viol_tol", 1e-10),
        ("compl_inf_tol", 1e-10),
        ("dual_inf_tol", 1e-8),
        // Mandatory for every activity accessor: the classifier reads
        // slacks, and a relaxed solve shifts them. It also sets the
        // walk's own `eps`, the margin a violation has to clear before
        // the repair fires.
        ("bound_relax_factor", 0.0),
    ] {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

fn solved_at(s: Number, lam: Number) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(PinnedLp { s, lam })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at s={s} lam={lam} failed: {status:?}",
    );
    solver
}

/// `(x, y)` at the exact solution for `lam`.
fn resolve_at(s: Number, lam: Number) -> [Number; 2] {
    let x = solved_at(s, lam)
        .converged()
        .expect("converged state")
        .x
        .clone();
    [x[0], x[1]]
}

/// What the walk actually did, plus the two facts that say which
/// branch it did it on: `x`'s activity class and its `Sigma`.
struct Walked {
    xy: [Number; 2],
    segments: usize,
    status: i8,
    sigma: Number,
    weak: usize,
}

/// Solve at `lam0`, classify, then walk to `lam0 + dp`. The pin is
/// constraint 1.
fn walk(s: Number, lam0: Number, dp: Number) -> Walked {
    let solver = solved_at(s, lam0);
    let base = solver.converged().expect("converged state").x.clone();
    let report = solver.classify_activity().expect("classify_activity");
    let weak = solver.weakly_active_bounds().expect("weakly_active_bounds");
    let (dx, segs) = solver
        .parametric_step_path(&[1], &[dp], 64)
        .expect("parametric_step_path");
    Walked {
        xy: [base[0] + dx[0], base[1] + dx[1]],
        segments: segs.len(),
        status: report.var_status[X_ROW],
        sigma: report.var_sigma[X_ROW],
        weak: weak.len(),
    }
}

/// The var-x row of `x`, whose bound is the kinked one. The pin
/// variable is not removed from the x block, so var-x rows and model
/// columns coincide here.
const X_ROW: usize = 0;

/// Both sides of the kink are checked against a re-solve, which for an
/// LP is exact, so the tolerance is numerical rather than modelling.
const TOL: Number = 1e-9;

/// Every test asserts the classifier told the walk nothing. Without
/// this the file would pass on the gh#852 path and prove nothing about
/// gh#928.
fn assert_the_classifier_is_blind(w: &Walked) {
    assert_eq!(
        w.status, UNIDENTIFIED,
        "the fixture must reach the unclassifiable branch, got status {} \
         (a nonzero Hessian diagonal would take the gh#852 path instead)",
        w.status,
    );
    assert_eq!(
        w.weak, 0,
        "`weakly_active_bounds` must be empty here, or the walk is being \
         told about the bound and this file tests gh#852, not gh#928",
    );
}

#[test]
fn the_walk_stays_inside_a_box_the_classifier_cannot_certify() {
    // The defect. Base one slack-unit below the kink, with the slack
    // chosen so Sigma clears 1 and the base-activity table marks the
    // bound enforced; then press into it. Truth holds x on the bound
    // and puts the whole step into y.
    let (lam0, dp) = (1.0 - 1e-6, 1e-2);
    let truth = resolve_at(1.0, lam0 + dp);
    let w = walk(1.0, lam0, dp);

    assert_the_classifier_is_blind(&w);
    assert!(
        w.sigma > 1.0,
        "the fixture must sit above the walk's base-activity test \
         (Sigma > 1), got {} -- below it the reach scan already \
         watches the bound and there is no defect to fix",
        w.sigma,
    );
    assert!(
        (truth[0] - 1.0).abs() < TOL && (truth[1] - (dp - 1e-6)).abs() < TOL,
        "the re-solve should hold the bound and spend the step on y: {truth:?}",
    );
    // The headline: pre-fix this returned x = 1.009999, a hundredth
    // outside a box whose width is 1, and reported no breakpoint at all.
    assert!(
        w.xy[0] <= 1.0 + TOL,
        "the walk returned a point outside x's box: x = {} against ub 1 \
         (segments {})",
        w.xy[0],
        w.segments,
    );
    let err = (w.xy[0] - truth[0]).abs().max((w.xy[1] - truth[1]).abs());
    assert!(
        err < TOL,
        "the walk should reproduce the re-solve, off by {err}: {:?} against {truth:?}",
        w.xy,
    );
    // Reaching the bound is an active-set change and the record has to
    // say so. Zero segments was half the defect: a caller with no
    // breakpoint has nothing to tell it the answer needs checking.
    assert!(
        w.segments > 0,
        "reaching the bound is a breakpoint and must be recorded",
    );
}

#[test]
fn the_lower_side_is_watched_too() {
    // The same kink on x's LOWER bound: a different entry of the
    // base-activity table, and the other side of the repair's own
    // choice of which bound the answer crossed.
    let (lam0, dp) = (1.0 - 1e-6, 1e-2);
    let truth = resolve_at(-1.0, lam0 + dp);
    let w = walk(-1.0, lam0, dp);

    assert_the_classifier_is_blind(&w);
    assert!(w.sigma > 1.0, "Sigma = {} must clear 1", w.sigma);
    assert!(
        (truth[0] + 1.0).abs() < TOL,
        "the re-solve should hold the lower bound: {truth:?}",
    );
    assert!(
        w.xy[0] >= -1.0 - TOL,
        "the walk returned a point outside x's box: x = {} against lb -1 \
         (segments {})",
        w.xy[0],
        w.segments,
    );
    let err = (w.xy[0] - truth[0]).abs().max((w.xy[1] - truth[1]).abs());
    assert!(
        err < TOL,
        "the walk should reproduce the re-solve, off by {err}: {:?} against {truth:?}",
        w.xy,
    );
    assert!(w.segments > 0, "reaching the bound is a breakpoint");
}

#[test]
fn a_capped_walk_keeps_the_old_contract() {
    // The repair promises a point inside the box, and reports rather
    // than returning one outside it -- but only when the walk was
    // allowed to finish. `max_iter` caps segments, so `max_iter = 0`
    // asks for the plain linear step, which on this model IS outside
    // the box: that is the whole defect, requested deliberately. It
    // was a legal thing to ask for before the repair existed, so it
    // still answers rather than failing.
    //
    // This is the guard on the post-loop check, and it is the only
    // arm of that check any fixture reaches. Remove the `max_iter`
    // gate and this test goes red while every other test in the file
    // stays green -- the cost of leaving it out is a caller who
    // capped the walk being told the repair failed.
    let (lam0, dp) = (1.0 - 1e-6, 1e-2);
    let solver = solved_at(1.0, lam0);
    let base = solver.converged().expect("converged state").x.clone();

    let (dx, segs) = solver
        .parametric_step_path(&[1], &[dp], 0)
        .expect("a capped walk answers rather than reporting");
    assert_eq!(segs.len(), 0, "a zero cap takes no segments");
    assert!(
        base[0] + dx[0] > 1.0 + TOL,
        "the capped answer is the linear step, which is past the bound: {}",
        base[0] + dx[0],
    );

    // One segment is enough to reach the bound, so the cap stops
    // binding and the answer comes back inside.
    let (dx, segs) = solver
        .parametric_step_path(&[1], &[dp], 1)
        .expect("parametric_step_path");
    assert_eq!(segs.len(), 1);
    assert!(
        base[0] + dx[0] <= 1.0 + TOL,
        "one segment is enough: {}",
        base[0] + dx[0],
    );
}

#[test]
fn a_direction_that_leaves_the_bound_is_not_repaired() {
    // The other branch of the reach scan, from the same base point:
    // press AWAY from the kink and the bound genuinely leaves. The
    // answer is interior, so the box check finds nothing and the
    // repair is a no-op -- which is what keeps it inert on every model
    // that was already right.
    let (lam0, dp) = (1.0 - 1e-6, -1e-2);
    let truth = resolve_at(1.0, lam0 + dp);
    let w = walk(1.0, lam0, dp);

    assert_the_classifier_is_blind(&w);
    assert!(w.sigma > 1.0, "Sigma = {} must clear 1", w.sigma);
    assert!(
        truth[0] < 1.0 - 1e-3 && truth[1].abs() < TOL,
        "the re-solve should come off the bound and leave y at zero: {truth:?}",
    );
    let err = (w.xy[0] - truth[0]).abs().max((w.xy[1] - truth[1]).abs());
    assert!(
        err < TOL,
        "the walk should reproduce the re-solve, off by {err}: {:?} against {truth:?}",
        w.xy,
    );
}

#[test]
fn below_the_activity_threshold_the_scan_already_watched_it() {
    // The control. Same model, same perturbation, same `UNIDENTIFIED`
    // verdict and same empty weak list -- only Sigma differs, and it
    // falls below 1. The walk never marked the bound enforced, the
    // reach scan watched it all along, and this case was correct
    // before the fix. It is here so the file cannot be read as
    // "`UNIDENTIFIED` is the trigger": Sigma is.
    let (lam0, dp) = (1.0 - 1e-5, 1e-2);
    let truth = resolve_at(1.0, lam0 + dp);
    let w = walk(1.0, lam0, dp);

    assert_the_classifier_is_blind(&w);
    assert!(
        w.sigma < 1.0,
        "the control must sit BELOW the base-activity test, got Sigma = {} \
         -- if the solve drifted above 1 this stopped being a control",
        w.sigma,
    );
    let err = (w.xy[0] - truth[0]).abs().max((w.xy[1] - truth[1]).abs());
    assert!(
        err < TOL,
        "the walk should reproduce the re-solve, off by {err}: {:?} against {truth:?}",
        w.xy,
    );
}
