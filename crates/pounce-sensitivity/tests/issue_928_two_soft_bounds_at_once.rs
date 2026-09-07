//! gh#928, the multi-bound arm: two unclassifiable bounds on one walk.
//!
//! `issue_928_path_leaves_an_unclassifiable_box.rs` is a one-bound
//! model. That leaves the corpus uniform in exactly the dimension the
//! repair loop acts on -- it adds rows to a watch list and re-walks,
//! so a model with one candidate row cannot distinguish "adds the
//! right row" from "adds the only row". This file is the second
//! fixture the branch rule asks for, not a duplicate of the first.
//!
//! It also reaches a branch the one-bound file cannot. The rejected
//! alternative fix -- admit every `UNIDENTIFIED` bound to the weak
//! list and let the factorization sort it out -- does not merely give
//! a worse answer here, it *fails to solve*:
//!
//! ```text
//! step_along_path: augmented solve failed (holds [1, 0], released [6, 7])
//! ```
//!
//! because it releases a bound that is genuinely enforced (`Sigma`
//! around `1e11`) alongside one that is not. The model below carries
//! both kinds at once on purpose, so that alternative stays rejected
//! by a test rather than by a memory.
//!
//! ```text
//! min  x + 2 y + 3 w
//! s.t. g0: x + y + w - lam = 0
//!      g1: lam = lam0            (the pin)
//!      x in [0, 1],  y in [0, 1],  w >= 0
//! ```
//!
//! Cheapest-first: `x` fills to 1, then `y` to 1, then `w` runs
//! unbounded. So
//!
//! ```text
//! lam <= 1:      x = lam,  y = 0,        w = 0
//! 1 <= lam <= 2: x = 1,    y = lam - 1,  w = 0
//! lam >= 2:      x = 1,    y = 1,        w = lam - 2
//! ```
//!
//! with kinks at `lam = 1` and `lam = 2`. Just below `lam = 2` the two
//! upper bounds are in different states in the same solve: `x`'s is
//! firmly held (its reduced cost is `2 - 1 = 1` against a vanishing
//! slack) while `y`'s is at the kink. Just above, both are held and
//! `w`'s lower bound is the kink. The Lagrangian Hessian is
//! identically zero, so none of them can be classified.

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

/// Column order is `[x, y, w, lam]`; the pin is constraint 1.
const N: usize = 4;
const PIN: Index = 1;
const COST: [Number; N] = [1.0, 2.0, 3.0, 0.0];
const LO: [Number; N] = [0.0, 0.0, 0.0, -1.0e19];
const HI: [Number; N] = [1.0, 1.0, 1.0e19, 1.0e19];

struct TwoSoftBounds {
    lam: Number,
    /// Per-variable curvature on `[x, y, w]`. Zero is the LP this file
    /// is about; a nonzero entry is used by one test to demonstrate
    /// *which* variables' curvature the walk's released system needs.
    curv: [Number; 3],
}

impl TNLP for TwoSoftBounds {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as Index,
            m: 2,
            nnz_jac_g: 5,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&LO);
        b.x_u.copy_from_slice(&HI);
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = self.lam;
        b.g_u[1] = self.lam;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        sp.x[2] = 0.5;
        sp.x[3] = self.lam;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let lin: Number = COST.iter().zip(x).map(|(c, xi)| c * xi).sum();
        let quad: Number = self
            .curv
            .iter()
            .zip(x)
            .map(|(c, xi)| 0.5 * c * xi * xi)
            .sum();
        Some(lin + quad)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&COST);
        for (i, c) in self.curv.iter().enumerate() {
            g[i] += c * x[i];
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[2] - x[3];
        g[1] = x[3];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 0, 0, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 3]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, -1.0, 1.0]);
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
        // Structurally present so the classifier has a diagonal to
        // divide by, and (for the LP this file is about) carrying zero,
        // so it finds it below the identification floor.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                for (v, c) in values.iter_mut().zip(self.curv) {
                    *v = obj_factor * c;
                }
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
    for (k, v) in [
        ("tol", 1e-10),
        ("constr_viol_tol", 1e-10),
        ("compl_inf_tol", 1e-10),
        ("dual_inf_tol", 1e-8),
        // Mandatory for every activity accessor, and it is also the
        // walk's `eps` -- the margin a violation clears before the
        // repair fires.
        ("bound_relax_factor", 0.0),
    ] {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

fn solved_at(lam: Number) -> Solver {
    solved_curved_at(lam, [0.0; 3])
}

fn solved_curved_at(lam: Number, curv: [Number; 3]) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(TwoSoftBounds { lam, curv })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at lam={lam} curv={curv:?} failed: {status:?}",
    );
    solver
}

/// The closed form above, which is what the re-solve reproduces. Used
/// to check the re-solve itself, so a wrong oracle cannot quietly
/// certify a wrong walk.
fn closed_form(lam: Number) -> [Number; 3] {
    [
        lam.clamp(0.0, 1.0),
        (lam - 1.0).clamp(0.0, 1.0),
        (lam - 2.0).max(0.0),
    ]
}

fn resolve_at(lam: Number) -> [Number; 3] {
    let x = solved_at(lam).converged().expect("converged").x.clone();
    let got = [x[0], x[1], x[2]];
    let want = closed_form(lam);
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() < 1e-7,
            "the re-solve disagrees with the closed form at lam={lam}: \
             {got:?} vs {want:?}",
        );
    }
    got
}

struct Walked {
    xyw: [Number; 3],
    segments: usize,
    status: [i8; 3],
    weak: usize,
}

fn walk(lam0: Number, dp: Number) -> Walked {
    let solver = solved_at(lam0);
    let base = solver.converged().expect("converged").x.clone();
    let report = solver.classify_activity().expect("classify_activity");
    let weak = solver.weakly_active_bounds().expect("weakly_active_bounds");
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[dp], 64)
        .expect("parametric_step_path");
    Walked {
        xyw: [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]],
        segments: segs.len(),
        status: [
            report.var_status[0],
            report.var_status[1],
            report.var_status[2],
        ],
        weak: weak.len(),
    }
}

/// Nothing in this model is classifiable, so the walk is on its own.
/// Without this the file could pass on the gh#852 path.
fn assert_the_classifier_is_blind(w: &Walked) {
    assert_eq!(
        w.status, [UNIDENTIFIED; 3],
        "every bound here must be unclassifiable, got {:?}",
        w.status,
    );
    assert_eq!(
        w.weak, 0,
        "`weakly_active_bounds` must be empty, or this file is testing \
         gh#852 rather than gh#928",
    );
}

/// Box violation, worst first, in the units of the model.
fn outside(xyw: &[Number; 3]) -> Vec<(usize, Number)> {
    let mut v: Vec<(usize, Number)> = (0..3)
        .filter_map(|i| {
            let past = (LO[i] - xyw[i]).max(xyw[i] - HI[i]);
            (past > 1e-9).then_some((i, past))
        })
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

/// The grid: both sides of both kinks, over five orders of slack, and
/// both directions at two magnitudes. Pre-fix this returns points
/// outside the box on 48 of the 192 combinations it generates in the
/// Python probe this file is the Rust form of; the count here is
/// smaller only because the Rust fixture drops the redundant
/// magnitudes.
fn grid() -> Vec<(Number, Number)> {
    let mut out = Vec::new();
    for kink in [1.0, 2.0] {
        for sgn in [-1.0, 1.0] {
            for d in [1e-8, 1e-6, 1e-4, 1e-3] {
                for dp in [-1e-1, -1e-2, 1e-2, 1e-1] {
                    out.push((kink + sgn * d, dp));
                }
            }
        }
    }
    out
}

#[test]
fn no_walk_on_the_grid_ends_outside_the_box() {
    // The defect, over the whole grid rather than at one point: a
    // model carrying two unclassifiable upper bounds returns a point
    // outside one of them. Pre-fix, a quarter of these do.
    let mut blind = 0usize;
    for (lam0, dp) in grid() {
        let w = walk(lam0, dp);
        if w.status == [UNIDENTIFIED; 3] && w.weak == 0 {
            blind += 1;
        }
        let bad = outside(&w.xyw);
        assert!(
            bad.is_empty(),
            "lam0={lam0} dp={dp}: the walk ended outside the box \
             (variable {}, past by {:e}); x={:?}, segments {}",
            bad[0].0,
            bad[0].1,
            w.xyw,
            w.segments,
        );
    }
    assert_eq!(
        blind,
        grid().len(),
        "every point on this grid must reach the unclassifiable branch",
    );
}

#[test]
fn the_walk_reproduces_the_resolve_on_the_whole_grid() {
    // Staying inside the box is necessary, not sufficient: a walk that
    // stopped at the bound and refused to spend the rest of the step
    // is also in the box and also wrong. The model is piecewise linear
    // in `lam`, so a correct walk matches the re-solve exactly, and
    // the tolerance below is numerical rather than modelling.
    for (lam0, dp) in grid() {
        let truth = resolve_at(lam0 + dp);
        let w = walk(lam0, dp);
        assert_the_classifier_is_blind(&w);
        let err = (0..3)
            .map(|i| (w.xyw[i] - truth[i]).abs())
            .fold(0.0, Number::max);
        assert!(
            err < 1e-7,
            "lam0={lam0} dp={dp}: walk {:?} vs re-solve {truth:?} \
             (worst coordinate off by {err:e}, segments {})",
            w.xyw,
            w.segments,
        );
    }
}

#[test]
fn crossing_two_kinks_is_exact_when_one_bound_is_released() {
    // A step long enough to cross both kinks. Starting just ABOVE the
    // first one, `x` is already on its bound and only `y`'s lower has
    // to leave the barrier, so the walk's released system stays
    // nonsingular and the answer is exact over three segments. This is
    // the two-kink case the repair does handle; the next test is the
    // one it does not.
    let (lam0, dp) = (1.0 + 1e-6, 1.2);
    let truth = resolve_at(lam0 + dp);
    assert!(
        truth[0] > 0.99 && truth[1] > 0.99 && truth[2] > 0.0,
        "the fixture must cross both kinks: {truth:?}",
    );
    let w = walk(lam0, dp);
    assert_the_classifier_is_blind(&w);
    assert!(
        outside(&w.xyw).is_empty(),
        "the walk left the box crossing two kinks: {:?}",
        w.xyw,
    );
    let err = (0..3)
        .map(|i| (w.xyw[i] - truth[i]).abs())
        .fold(0.0, Number::max);
    assert!(
        err < 1e-7,
        "walk {:?} vs re-solve {truth:?}, off by {err:e} over {} segments",
        w.xyw,
        w.segments,
    );
    assert!(
        w.segments >= 2,
        "crossing two kinks should record at least two breakpoints, got {}",
        w.segments,
    );
}

/// The walk's released system, and why two releases can kill it.
///
/// Holding a bound the path reached is a Schur pin applied to the
/// factored released system `K`, so `K` itself has to be invertible
/// *before* the pins go on. Releasing a bound takes that bound's
/// `Sigma` out of `K`'s diagonal. On a model with no curvature the
/// diagonal is then exactly zero, and the stationarity rows of two
/// such released variables that share a constraint become linearly
/// dependent -- for the model here, both read `y_c0 = rhs`. `K` is
/// singular, the Schur solve cannot run, and the walk reports.
///
/// That is a limit of the hold-and-release architecture (gh#852,
/// tracked open as gh#930), not
/// of the box repair: pre-fix this same step returned a point outside
/// the box in silence, which is worse. The repair turned a wrong
/// answer into a refusal, and the refusal is what this test pins --
/// together with the measurement that says the cause is what the
/// paragraph above claims.
#[test]
fn two_releases_without_curvature_are_refused_not_answered() {
    // Below the first kink, stepping past the second. The walk has to
    // release BOTH x's upper (reached and held) and y's lower (its
    // multiplier hits zero), and neither variable has curvature.
    let (lam0, dp) = (1.0 - 1e-6, 1.2);
    let solver = solved_at(lam0);
    let err = solver.parametric_step_path(&[PIN], &[dp], 64).expect_err(
        "two curvature-free releases must be refused; returning an \
             answer here means either the architecture changed -- in \
             which case assert the answer instead -- or the box check \
             stopped firing, which is gh#928 back again",
    );
    let err = format!("{err:?}");
    assert!(
        err.contains("step_along_path"),
        "the refusal must name the walk that produced it: {err}",
    );

    // The cause, measured rather than argued: curvature on exactly the
    // two RELEASED variables makes the same step exact, and the same
    // amount of curvature on the third variable does not. If only the
    // first of these held, "the released system is singular" would be
    // indistinguishable from "any regularization helps".
    let truth = resolve_at(lam0 + dp);
    for (label, curv, want_ok) in [
        ("on the two released variables", [1e-3, 1e-3, 0.0], true),
        ("on the third variable only", [0.0, 0.0, 1e-3], false),
    ] {
        let s = solved_curved_at(lam0, curv);
        let base = s.converged().expect("converged").x.clone();
        let got = s.parametric_step_path(&[PIN], &[dp], 64);
        match (got, want_ok) {
            (Ok((dx, segs)), true) => {
                let xyw = [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]];
                let e = (0..3)
                    .map(|i| (xyw[i] - truth[i]).abs())
                    .fold(0.0, Number::max);
                assert!(
                    e < 1e-5,
                    "curvature {label} should make the step exact, got \
                     {xyw:?} vs {truth:?} (off by {e:e}, {} segments)",
                    segs.len(),
                );
            }
            (Err(_), false) => {}
            (Ok(_), false) => panic!(
                "curvature {label} should NOT rescue the step -- if it \
                 does, the released system's singularity is not what \
                 refuses this walk and the doc comment above is wrong",
            ),
            (Err(e), true) => panic!("curvature {label} should rescue the step: {e:?}"),
        }
    }
}

#[test]
fn a_firmly_held_bound_is_not_released_alongside_a_soft_one() {
    // Why the repair watches what the answer crossed rather than
    // admitting every unclassifiable bound. Just below lam = 2, x's
    // upper bound is genuinely enforced -- reduced cost 1 against a
    // vanishing slack, so `Sigma` is enormous -- while y's is at the
    // kink. Both read UNIDENTIFIED. The rejected fix released both and
    // the augmented solve failed outright; this one steps down, which
    // holds x and moves y, and gets the re-solve.
    let (lam0, dp) = (2.0 - 1e-6, -1e-1);
    let solver = solved_at(lam0);
    let report = solver.classify_activity().expect("classify_activity");
    assert_eq!(
        [report.var_status[0], report.var_status[1]],
        [UNIDENTIFIED, UNIDENTIFIED],
        "both upper bounds must be unclassifiable for the point to stand",
    );
    assert!(
        report.var_sigma[0] > 1e6 * report.var_sigma[1].max(1.0),
        "x's bound must be the firmly held one and y's the soft one: \
         Sigma = {:e} vs {:e}",
        report.var_sigma[0],
        report.var_sigma[1],
    );

    let truth = resolve_at(lam0 + dp);
    let base = solver.converged().expect("converged").x.clone();
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[dp], 64)
        .expect("the walk must solve, not fail in the augmented system");
    let got = [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]];
    let err = (0..3)
        .map(|i| (got[i] - truth[i]).abs())
        .fold(0.0, Number::max);
    assert!(
        err < 1e-7,
        "walk {got:?} vs re-solve {truth:?}, off by {err:e} over {} segments",
        segs.len(),
    );
    assert!(
        (got[0] - 1.0).abs() < 1e-7,
        "x must stay on its firmly held bound, got {}",
        got[0],
    );
}

// # Mutation table
//
// Every row was run. A test that goes red for a change is evidence
// about that change; a row with no red test is a change this file
// cannot see, and is listed as such rather than omitted.
//
// | change                                                        | red                        |
// |---------------------------------------------------------------|----------------------------|
// | budget the repair loop at zero passes                          | grid (both)                |
// | seed the watch list empty instead of from `weak_rows`          | nothing here; gh#852 test 4 |
// | read the crossed side off the base point, not the answer       | grid (both)                |
// | let the singular released system return its garbage answer     | `two_releases_...`         |
// | regularize the released system on every walk                   | `two_releases_...` (the third-variable arm passes, so the "on the two released variables" arm is what pins the cause) |
// | drop the closed-form check inside `resolve_at`                 | nothing; it guards the oracle, not the walk |
//
// `crossing_two_kinks_is_exact_when_one_bound_is_released` stays
// green when the repair is disabled entirely. That is deliberate and
// worth naming: it is a test of the hold-and-release architecture
// crossing two kinks, not of the box repair, and it is here as the
// control for the test that follows it.
//
// # What this file is not evidence about
//
// * **The repair budget.** Measured, a budget of one pass suffices for
//   every fixture here and in
//   `issue_928_path_leaves_an_unclassifiable_box.rs`; the shipped
//   budget is the base-activity table's length, which is slack. No
//   fixture reaches a second pass, so the second pass is untested.
// * **Constraint rows.** Every limit here is written as a variable
//   bound, so nothing here exercises the `v_l` / `v_u` half of the
//   bound rows or the `s`-block release diagonal.
//   `issue_928_a_limit_written_as_a_row.rs` owns that arm.
// * **The refusal's blast radius.** `two_releases_...` pins that the
//   walk refuses rather than answers, not that refusing is the best
//   possible behaviour; gh#930 tracks answering it. Pre-fix this step returned a point outside the
//   box in silence; a later change that makes the walk *answer* it
//   correctly should replace that test with an equality assertion, not
//   delete it.
// * **Scaling.** Everything here runs unit-scaled. Leg 1 of
//   `sens_invariance_legs.rs` owns that dimension.
// * **Magnitude.** Four variables. The largest convex fixture in the
//   corpus is 534 columns and `benchmarks/qp` reaches 93 263; nothing
//   here bounds what happens there.
