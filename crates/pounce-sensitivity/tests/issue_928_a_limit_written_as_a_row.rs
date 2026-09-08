//! gh#928, the constraint-row arm: the same limit, written the other
//! way, was watched by nothing.
//!
//! The companion files put the limit on a **variable** -- `x <= 1`
//! declared in `x_u`. This one writes the identical limit as a
//! **constraint row**, `g(x) <= cap`, which is how a capacity, a ramp,
//! a nameplate, or any limit on a computed quantity is normally
//! stated. The two describe the same feasible set, and before this
//! branch they were not the same to the sensitivity layer at all:
//!
//! * a variable bound's multiplier lives in `z_l` / `z_u`, and the
//!   quantity it constrains is a row of the `x` block;
//! * a row's limit bounds the **slack**, so its multiplier lives in
//!   `v_l` / `v_u`, and the quantity it constrains is a row of the `s`
//!   block.
//!
//! `bound_variable_rows` emitted only the `z` half, and
//! `bound_context`'s box covered only the `x` block. A row limit was
//! therefore in no bound row and in no box: the reach scan never saw
//! it, no breakpoint could name it, the base-activity table had no
//! entry for it, and gh#928's repair loop had nothing to add to its
//! watch list. This is not the identification floor the companion
//! files are about -- it is a whole class of limit that the walk could
//! not represent, at any curvature.
//!
//! ```text
//! min  0.5 (x - lam)^2
//! s.t. g0: lam = lam0        (the pin)
//!      g1: x <= CAP          (the limit, as a ROW)
//!      x free
//! ```
//!
//! `x` carries no bound of its own, so the cap is reachable only
//! through `g1`. The optimum is `x = min(lam, CAP)`, with a kink at
//! `lam = CAP`, and the objective is strictly convex in `x`, so what
//! the released system does with the cap's barrier term is visible in
//! the answer rather than masked by an equality.
//!
//! # Measured, on this model, at HEAD before the fix
//!
//! ```text
//!  lam0   dp     branch          x         truth    segments
//!  0.8   +0.5    reach       1.300000000    1.0        []
//!  0.8   +0.3    reach       1.100000000    1.0        []
//!  1.3   -0.5    release     1.000000000    0.8        []
//!  1.3   -0.4    release     1.000000000    0.9        []
//!  1.0   +0.01   reach       1.004992176    1.0        []
//!  1.0   -0.01   release     0.994992176    0.99       []
//!  1.3   +0.2    (neither)   1.000000000    1.0        []
//!  0.8   -0.2    (neither)   0.600000000    0.6        []
//! ```
//!
//! Three tenths outside a stated cap, with an empty record. That empty
//! record is the point: a caller cannot even tell the answer is
//! suspect, which is gh#928's signature one block over. After the fix
//! every row above is exact to about 1e-10 and carries a segment on
//! the cap's own slack row. The last two rows touch the cap in neither
//! direction and are exact both before and after -- the control that
//! says the machinery is not simply clamping.
//!
//! # Both branches, deliberately
//!
//! A leg is only evidence about the branch its fixture reaches, and
//! the walk has two ways to touch a row limit:
//!
//! * **reach** -- the base point is below the cap and the step presses
//!   into it. Needs the row in the scan, in the box, and a Schur hold
//!   on its `s` row. No backsolver arithmetic: `path_direction` pins
//!   any KKT row, which is what makes the `(x, s)` prefix a single box
//!   rather than a second index space.
//! * **release** -- the base point holds the cap with a multiplier of
//!   order one and the step drives that multiplier to zero. Needs the
//!   barrier's **`s`-block diagonal** rebuilt with the released
//!   entry's `v / s` taken out, which is arithmetic the release half
//!   did not have. Pre-fix this arm did not merely lose the record: at
//!   `lam0 = 1.3`, `dp = -0.5` the answer stayed pinned at the cap,
//!   1.0 against a truth of 0.8, because nothing took the cap's
//!   stiffness out of the operator.
//!
//! # Mutation table
//!
//! Every row below was applied to this tree and run; the "what goes
//! red" column is the observed result, not the intended one. Two of
//! them came out differently from the way they were first written
//! down, and both corrections are load-bearing -- see the notes.
//!
//! | change | what goes red |
//! |---|---|
//! | drop the `pd_l` / `pd_u` groups from `bound_variable_rows` | release, kink, **and the control** -- but *not* reach (note 1) |
//! | narrow `bound_context`'s box back to the `x` block | reach, release, kink; the control stays green |
//! | route an `s`-block release into `sigma_x` | release, kink-release, and `releasing_the_row_limit_leaves_the_variable_bound_alone` (note 2) |
//! | leave `sigma_s` at its base value on a release | the same three, with a different message on the third (note 2) |
//! | drop the `s`-block offset in `Solver::d_slack_rows` | `the_cap_is_a_row_and_its_primal_row_is_n_x` |
//! | skip the `s` block's `natural_units_factor` conversion in `bound_context` | `leg_scaling_a_row_limit_is_reached_at_the_same_place_under_a_row_scaling`, and *only* that one (note 4) |
//! | keep `slacks_and_directions`' `[..n_x]` truncation | **nothing here** -- `cd_split_pin_mapping.rs` catches it (note 3) |
//!
//! **Note 1 -- the two halves of the fix are separable, and the
//! control is the discriminator.** The reach scan reads the box
//! directly (`for i in 0..n_p` in `walk_once`), so widening
//! `bound_context`'s box is by itself enough to reach a row limit;
//! the bound rows are not on that path.
//! `a_row_limit_the_step_presses_into_is_reached_and_held` therefore
//! stays green with the `v` groups suppressed. What goes red instead
//! is `a_step_that_does_not_touch_the_cap_records_nothing`: with no
//! bound row for the cap, the strongly-held base point has no
//! base-activity entry, the walk reads the slack as free, and
//! `lam0 = 1.3, dp = +0.2` -- a step that never approaches the cap --
//! records a spurious reach at fraction 0.600. So the control is not
//! a formality. It is the only test in the file that fails when the
//! base-activity table loses the row, and it is what separates "the
//! walk can see the cap" from "the walk knows the cap is already
//! held".
//!
//! **Note 2 -- the first model cannot separate mutations 3 and 4, so
//! there is a second one.** On the model above the two mutations are
//! bit-identical (`x` 0.9999999999373727 against a re-solve of
//! 0.7999999999545443, the segment still recorded at fraction 0.600),
//! because no variable in it carries a bound of its own: `sigma_x` is
//! identically zero, so misrouting the release into it is
//! observationally the same as dropping it. That is the gh#450 hazard
//! in its quiet form -- the wrong write lands somewhere harmless *on
//! this model*, and a corrector that got the `s` half right while
//! also corrupting a neighbouring `x` entry would run green.
//!
//! No other fixture closes it either:
//! `issue_928_two_soft_bounds_at_once.rs` is all variable bounds and
//! has no row limit at all. So the second model below carries both at
//! once -- `w >= 0` strongly held with multiplier 3 at `x`-block index
//! 0, and the cap's slack at index `n_x`, which is exactly the index a
//! misrouted release maps back onto. Measured, the two mutations now
//! fail `releasing_the_row_limit_leaves_the_variable_bound_alone`
//! with *different* messages:
//!
//! ```text
//! mutation 3:  `w`'s bound must not move;
//!              segments [(0.600, 3, false, false), (0.600, 0, true, true)]
//! mutation 4:  x walked to 0.999999999937373 against a closed form of 0.8
//! ```
//!
//! Under 3 the answer for `x` is *right* and only `w` is wrong, which
//! is the whole reason the model exists: every other test in this file
//! reads `x`.
//!
//! **Note 3 -- the corrector is not on this file's path.**
//! `parametric_step_path` does not run `corrector::run`, so restoring
//! the `[..n_x]` truncation leaves all six tests green. Across the
//! whole crate exactly one test catches it, and it catches it loudly:
//! `cd_split_pin_mapping::correct_step_translates_pin_index_through_cd_split`
//! panics at `corrector.rs:255` with "the len is 3 but the index is
//! 3". Recorded here because the obvious reading -- that the file
//! covering the `s`-block bound rows also covers every consumer of
//! them -- is wrong.
//!
//! **Note 4 -- the scaling legs are not vacuous, but only one of
//! them bites.** Dropping the `s`-block frame conversion reproduces
//! gh#928's original symptom on the scaled arm exactly:
//!
//! ```text
//! scaled: the walk left the box at 1.2999999998863827, cap 1
//! ```
//!
//! -- the same 1.2999999998 the pre-fix table records, from a
//! *correct* box that was compared in the wrong frame. The release
//! leg stays green under that mutation: the release decision is made
//! on multipliers, and the base slack's frame only reaches the box
//! comparison, which the reach arm is what exercises. So the release
//! leg is evidence that the release arithmetic survives a row
//! scaling, and not evidence about the conversion itself.
//!
//! # Not evidence about
//!
//! Scaling: this model is unit-scaled, and
//! `sens_invariance_legs.rs` leg 1 owns that dimension. Index spaces
//! beyond the one block boundary the file is about:
//! `cd_split_pin_mapping.rs` owns g-index versus KKT row, and this
//! model has `m = 2` with one inequality, so the two coincide.
//! Magnitude at benchmark scale. The convex arm, which has its own
//! gh#928 file. A row limit that is *also* a variable bound on the
//! same quantity: both would be watched, and which one the walk picks
//! is not decided here.

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

/// Column order is `[x, lam]`; the pin is constraint 0 and the cap is
/// constraint 1.
const N: usize = 2;
const PIN: Index = 0;
const CAP: Number = 1.0;
/// The cap's full-g row index: the pin is constraint 0, the cap is
/// constraint 1.
const CAP_ROW: Index = 1;
/// The slack row: the `s` block sits immediately after the `x` block,
/// and there is exactly one inequality, so the cap's primal KKT row is
/// `n_x`. Asserted in `the_cap_is_a_row_and_its_primal_row_is_n_x`
/// rather than trusted.
const SLACK_ROW: usize = N;
/// How far past a bound counts as outside. `bound_relax_factor` is
/// zero, so the box is the declared one and this is roundoff headroom.
const OUT: Number = 1e-9;

struct RowLimit {
    lam: Number,
    /// Per-variable and per-row factors for the scaling leg. `None`
    /// means "hand nothing back", which under `user-scaling` is the
    /// identity -- the same option, the same objective factor, and the
    /// only difference between the leg's two arms.
    scaling: Option<(Vec<Number>, Vec<Number>)>,
}

impl TNLP for RowLimit {
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some((d, g)) = self.scaling.as_ref() else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(d);
        *req.use_g_scaling = true;
        req.g_scaling.copy_from_slice(g);
        true
    }

    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N as Index,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // `x` carries NO bound of its own. The cap is reachable only
        // through the row, which is the whole point of the file.
        b.x_l.copy_from_slice(&[-1.0e19; N]);
        b.x_u.copy_from_slice(&[1.0e19; N]);
        b.g_l[0] = self.lam;
        b.g_u[0] = self.lam;
        // One-sided on purpose: a two-sided row would give the `s`
        // block a lower bound too and blur which side was reached.
        b.g_l[1] = -1.0e19;
        b.g_u[1] = CAP;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.lam;
        sp.x[1] = self.lam;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let r = x[0] - x[1];
        Some(0.5 * r * r)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let r = x[0] - x[1];
        g[0] = r;
        g[1] = -r;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[1];
        g[1] = x[0];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[1, 0]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0]);
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
        // Lower triangle of [[1, -1], [-1, 1]].
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[obj_factor, -obj_factor, obj_factor]);
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
        // walk's `eps`.
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
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(RowLimit { lam, scaling: None })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at lam={lam} failed: {status:?}",
    );
    solver
}

/// The closed form. The re-solve is checked against it so a wrong
/// oracle cannot quietly certify a wrong walk.
fn closed_form(lam: Number) -> Number {
    lam.min(CAP)
}

fn resolve_at(lam: Number) -> Number {
    let got = solved_at(lam).converged().expect("converged").x[0];
    let want = closed_form(lam);
    assert!(
        (got - want).abs() < 1e-7,
        "the re-solve disagrees with the closed form at lam={lam}: \
         {got} vs {want}",
    );
    got
}

struct Walked {
    /// `x` at the walk's answer, which is also the `g1` row's value --
    /// the quantity the cap actually bounds.
    x: Number,
    segments: Vec<(Number, usize, bool, bool)>,
    n_x: usize,
}

impl Walked {
    /// The events the walk recorded on the cap's own slack row.
    fn on_the_cap(&self) -> Vec<(Number, bool, bool)> {
        self.segments
            .iter()
            .filter(|&&(_, row, _, _)| row >= self.n_x)
            .map(|&(at, _, lower, pinned)| (at, lower, pinned))
            .collect()
    }
}

fn walk(lam0: Number, dp: Number) -> Walked {
    walk_solver(&solved_at(lam0), dp)
}

fn walk_solver(solver: &Solver, dp: Number) -> Walked {
    let base = solver.converged().expect("converged").x[0];
    let n_x = solver.block_dims().expect("block_dims")[0];
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[dp], 64)
        .expect("parametric_step_path");
    Walked {
        x: base + dx[0],
        segments: segs
            .iter()
            .map(|s| (s.at, s.var_row, s.lower, s.pinned))
            .collect(),
        n_x,
    }
}

/// The premise, in one place: the cap really is a row, the `s` block
/// really is one long, and the cap's primal KKT row really is `n_x`.
/// Every index in this file is read against that.
#[test]
fn the_cap_is_a_row_and_its_primal_row_is_n_x() {
    let solver = solved_at(0.8);
    let dims = solver.block_dims().expect("block_dims");
    assert_eq!(
        dims[0], N,
        "no variable is removed here, so the `x` block is the model's own \
         column count",
    );
    assert_eq!(
        dims[1], 1,
        "the model must have exactly one inequality row, so the `s` block \
         is one long and the cap's primal row is `n_x`",
    );
    assert_eq!(SLACK_ROW, dims[0], "the file's `SLACK_ROW` is stale");
    // The cap is not a variable bound wearing a row's clothes: `x` has
    // no declared bound at all, so `z_l` and `z_u` are empty.
    assert_eq!(
        (dims[4], dims[5]),
        (0, 0),
        "`x` must carry no bound of its own, or the walk could reach the \
         cap through the `z` half and the file would prove nothing",
    );
    assert!(
        dims[7] >= 1,
        "the cap's multiplier must be in `v_u`, got block dims {dims:?}",
    );
    // The accessor the record's consumer resolves an `s`-block row
    // with. A segment naming row `SLACK_ROW` is only translatable back
    // to "the `g1` row" because this map says so, and the map is the
    // `x` block's width plus the row's position in the `d` block --
    // dropping the offset returns `0`, a real row in the `x` block,
    // which is the gh#450 hazard one block over.
    let slack = solver.d_slack_rows(&[PIN, CAP_ROW]).expect("d_slack_rows");
    assert_eq!(
        slack,
        vec![None, Some(SLACK_ROW as Index)],
        "the pin is an equality and owns no slack; the cap is the only \
         inequality and its slack is the first row of the `s` block",
    );
}

/// The three base points reach three different activity states on the
/// same row -- checked, not assumed, because each test below is only
/// evidence about the branch its base point takes.
#[test]
fn the_three_base_points_reach_three_different_states() {
    use pounce_sensitivity::activity::{INACTIVE, STRONGLY_ACTIVE, WEAKLY_ACTIVE};
    for (lam, want, sigma_range) in [
        (0.8, INACTIVE, (0.0, 1e-6)),
        (1.0, WEAKLY_ACTIVE, (1e-3, 1e3)),
        (1.3, STRONGLY_ACTIVE, (1e6, Number::INFINITY)),
    ] {
        let report = solved_at(lam)
            .classify_activity()
            .expect("classify_activity");
        assert_eq!(
            report.row_status[1], want,
            "at lam={lam} the cap's row status is {} and should be {want}",
            report.row_status[1],
        );
        let sigma = report.row_sigma[1];
        assert!(
            sigma > sigma_range.0 && sigma < sigma_range.1,
            "at lam={lam} the cap's row Sigma is {sigma:e}, outside \
             {sigma_range:?}; the three fixtures are not reaching three \
             different branches",
        );
    }
}

/// The reach branch: the base point is below the cap and the step
/// presses into it. Pre-fix this returned 1.3 against a truth of 1.0
/// with an empty segment list.
#[test]
fn a_row_limit_the_step_presses_into_is_reached_and_held() {
    for &(dp, want_at) in &[(0.5, 0.4), (0.3, 2.0 / 3.0), (0.25, 0.8)] {
        let w = walk(0.8, dp);
        assert!(
            w.x <= CAP + OUT,
            "dp={dp}: the walk put the `g1` row at {} against a cap of \
             {CAP}, over by {:e}; segments {:?}",
            w.x,
            w.x - CAP,
            w.segments,
        );
        let truth = resolve_at(0.8 + dp);
        assert!(
            (w.x - truth).abs() < 1e-7,
            "dp={dp}: x walked to {} against a re-solve of {truth}",
            w.x,
        );
        // The record has to name what happened. The answer above was
        // reachable pre-fix only with an empty segment list, which is
        // the half of gh#928 that makes it silent.
        let events = w.on_the_cap();
        assert_eq!(
            events.len(),
            1,
            "dp={dp}: exactly one event on the cap's slack row is expected, \
             got {:?} (all segments {:?}, n_x = {})",
            events,
            w.segments,
            w.n_x,
        );
        let (at, lower, pinned) = events[0];
        assert!(
            !lower && pinned,
            "dp={dp}: the cap is an UPPER limit the walk holds, so the \
             segment must read lower=false pinned=true; got \
             lower={lower} pinned={pinned}",
        );
        // The cap is hit where `lam` crosses it, which the closed form
        // fixes exactly: a wrong fraction would still land the endpoint
        // on the cap and pass every check above.
        assert!(
            (at - want_at).abs() < 1e-6,
            "dp={dp}: the cap is reached at fraction {at}, and `lam` \
             crosses it at {want_at}",
        );
    }
}

/// The release branch: the base point holds the cap with a multiplier
/// of order one and the step drives it to zero. This is the arm that
/// needs the `s`-block barrier diagonal rebuilt -- pre-fix the answer
/// stayed pinned at 1.0 against a truth of 0.8, because nothing took
/// the cap's stiffness out of the operator.
#[test]
fn a_row_limit_the_step_leaves_is_released() {
    for &(dp, want_at) in &[(-0.5, 0.6), (-0.4, 0.75), (-0.6, 0.5)] {
        let w = walk(1.3, dp);
        let truth = resolve_at(1.3 + dp);
        assert!(
            (w.x - truth).abs() < 1e-7,
            "dp={dp}: x walked to {} against a re-solve of {truth}; \
             segments {:?}",
            w.x,
            w.segments,
        );
        // Held at 1.0 is the pre-fix answer, and it is inside the box,
        // so the box check alone would not catch it. Said separately so
        // a regression reads as what it is.
        assert!(
            w.x < CAP - 1e-3,
            "dp={dp}: x is still at the cap ({}), so the release did not \
             reach the operator",
            w.x,
        );
        let events = w.on_the_cap();
        assert_eq!(
            events.len(),
            1,
            "dp={dp}: leaving the cap is an active-set change and must be \
             in the record; got {:?} (all segments {:?})",
            events,
            w.segments,
        );
        let (at, lower, pinned) = events[0];
        assert!(
            !lower && !pinned,
            "dp={dp}: the cap is an UPPER limit the walk releases, so the \
             segment must read lower=false pinned=false; got \
             lower={lower} pinned={pinned}",
        );
        assert!(
            (at - want_at).abs() < 1e-6,
            "dp={dp}: the cap is released at fraction {at}, and its \
             multiplier reaches zero at {want_at}",
        );
    }
}

/// The kink, both directions. The cap's `Sigma` is order one here, so
/// the factorization neither enforces it nor ignores it, and the walk
/// is what decides. Pre-fix both directions were wrong by 5.0e-3 --
/// half the perturbation -- which is the two-sided average a
/// degenerate base point produces when nothing watches the bound.
#[test]
fn at_the_kink_both_directions_are_decided_by_the_walk() {
    for &(dp, want_pinned) in &[(0.01, true), (-0.01, false), (0.1, true), (-0.1, false)] {
        let w = walk(CAP, dp);
        let truth = resolve_at(CAP + dp);
        assert!(
            (w.x - truth).abs() < 1e-7,
            "dp={dp}: x walked to {} against a re-solve of {truth}; \
             segments {:?}",
            w.x,
            w.segments,
        );
        assert!(w.x <= CAP + OUT, "dp={dp}: {} is past the cap", w.x,);
        let events = w.on_the_cap();
        assert_eq!(
            events.len(),
            1,
            "dp={dp}: got {:?} (all segments {:?})",
            events,
            w.segments,
        );
        let (_, lower, pinned) = events[0];
        assert!(
            !lower && pinned == want_pinned,
            "dp={dp}: stepping {} the cap must {} it; got lower={lower} \
             pinned={pinned}",
            if dp > 0.0 { "into" } else { "away from" },
            if want_pinned { "hold" } else { "release" },
        );
    }
}

/// The control: perturbations that touch the cap in neither direction.
/// These were exact before the fix too, so a test file that only
/// checked answers could pass on them while the two branches above
/// were broken.
#[test]
fn a_step_that_does_not_touch_the_cap_records_nothing() {
    for &(lam0, dp) in &[(1.3, 0.2), (0.8, -0.2), (1.3, 0.5), (0.8, -0.5)] {
        let w = walk(lam0, dp);
        let truth = resolve_at(lam0 + dp);
        assert!(
            (w.x - truth).abs() < 1e-7,
            "lam0={lam0} dp={dp}: x walked to {} against a re-solve of \
             {truth}",
            w.x,
        );
        assert!(
            w.on_the_cap().is_empty(),
            "lam0={lam0} dp={dp}: nothing happens to the cap here, so the \
             record must not name it; got {:?}",
            w.segments,
        );
    }
}

// ---------------------------------------------------------------
// The second model: a row limit and a live variable bound at once.
// ---------------------------------------------------------------
//
// The model above cannot tell a *misrouted* `s`-block release from a
// *dropped* one, because it has no variable bound and its `sigma_x`
// is identically zero -- writing the release into the `x` diagonal
// lands somewhere inert. That is the gh#450 hazard in its quiet form,
// and note 2 in the header says so. This model closes it:
//
// ```text
// min  0.5 (x - lam)^2 + 0.5 w^2 + 3 w
// s.t. g0: lam = lam0        (the pin)
//      g1: x <= CAP          (the limit, as a row)
//      w >= 0                (a bound of its own, strongly held)
//      x, lam free
// ```
//
// Columns are `[w, x, lam]`, so `w` sits at `x`-block index 0 and the
// cap's slack sits at index `n_x`. `w`'s unconstrained minimum is
// `-3`, so its lower bound is held with multiplier 3 and a vanishing
// slack: `sigma_x[0]` is large and *live*. Releasing the cap must not
// touch it. A release written to `sigma_x[var_row - n_x]` --
// `sigma_x[0]` -- takes `w`'s own stiffness out of the operator, and
// `w` walks off its bound toward `-3`. The answer for `x` is
// unchanged either way, which is what makes this worth a separate
// model: the wrong write is invisible in the quantity the file's
// other tests look at.

const N2: usize = 3;
/// `w`'s column, and the `x`-block index a misrouted release would
/// land on.
const W2: usize = 0;
/// `x`'s column in the second model.
const X2: usize = 1;
/// `w`'s pull: its unconstrained minimum is `-W_PULL`.
const W_PULL: Number = 3.0;

struct RowLimitPlusBound {
    lam: Number,
}

impl TNLP for RowLimitPlusBound {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: N2 as Index,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, -1.0e19, -1.0e19]);
        b.x_u.copy_from_slice(&[1.0e19; N2]);
        b.g_l[0] = self.lam;
        b.g_u[0] = self.lam;
        b.g_l[1] = -1.0e19;
        b.g_u[1] = CAP;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[W2] = 1.0;
        sp.x[X2] = self.lam;
        sp.x[2] = self.lam;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let r = x[X2] - x[2];
        Some(0.5 * r * r + 0.5 * x[W2] * x[W2] + W_PULL * x[W2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let r = x[X2] - x[2];
        g[W2] = x[W2] + W_PULL;
        g[X2] = r;
        g[2] = -r;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        g[1] = x[X2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[2, X2 as Index]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0]);
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
        // Lower triangle of diag-plus-coupling on `[w, x, lam]`:
        // (w,w)=1, (x,x)=1, (lam,x)=-1, (lam,lam)=1.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[obj_factor, obj_factor, -obj_factor, obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved_at2(lam: Number) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(RowLimitPlusBound { lam })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve at lam={lam} failed: {status:?}",
    );
    solver
}

#[test]
fn the_second_model_holds_a_variable_bound_and_a_row_limit_at_once() {
    let solver = solved_at2(1.3);
    let dims = solver.block_dims().expect("block_dims");
    assert_eq!(dims[0], N2, "no variable is removed");
    assert_eq!(dims[1], 1, "one inequality, so one slack");
    assert_eq!(
        dims[4], 1,
        "`w` must carry a lower bound of its own, so `z_l` is one long \
         and `sigma_x` has a live entry a misrouted release could \
         corrupt; got block dims {dims:?}",
    );
    assert!(dims[7] >= 1, "the cap's multiplier must be in `v_u`");
    assert_eq!(
        solver.d_slack_rows(&[CAP_ROW]).expect("d_slack_rows"),
        vec![Some(N2 as Index)],
        "the cap's slack must sit at `n_x`, which is the index a \
         misrouted release maps to `sigma_x[0]` -- `w`'s own entry",
    );
    let conv = solver.converged().expect("converged");
    assert!(
        (conv.x[W2] - 0.0).abs() < 1e-7,
        "`w` must sit on its lower bound, got {}",
        conv.x[W2],
    );
    assert!(
        (conv.x[X2] - CAP).abs() < 1e-7,
        "`x` must sit on the cap, got {}",
        conv.x[X2],
    );
}

#[test]
fn releasing_the_row_limit_leaves_the_variable_bound_alone() {
    // The discriminating case. `dp = -0.5` releases the cap; `w`'s
    // bound is untouched by that and must stay held. Routing the
    // release into `sigma_x[0]` zeroes `w`'s barrier stiffness, and
    // `w` walks off zero toward `-W_PULL` -- while `x` still lands on
    // the right answer, so nothing else in this file notices.
    let solver = solved_at2(1.3);
    let base = solver.converged().expect("converged").x.clone();
    let dp = -0.5;
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[dp], 64)
        .expect("parametric_step_path");
    let w = base[W2] + dx[W2];
    let x = base[X2] + dx[X2];

    // The cap is released, so the record must say so exactly once.
    let cap_segs: Vec<_> = segs
        .iter()
        .filter(|s| s.var_row == N2 && !s.pinned)
        .collect();
    assert_eq!(
        cap_segs.len(),
        1,
        "the cap must be released exactly once; segments {:?}",
        segs.iter()
            .map(|s| (s.at, s.var_row, s.lower, s.pinned))
            .collect::<Vec<_>>(),
    );
    // `w`'s own bound is not part of this step and must never appear.
    assert!(
        segs.iter().all(|s| s.var_row != W2),
        "`w`'s bound must not move; segments {:?}",
        segs.iter()
            .map(|s| (s.at, s.var_row, s.lower, s.pinned))
            .collect::<Vec<_>>(),
    );
    assert!(
        w.abs() < 1e-6,
        "`w` must stay on its lower bound across the cap's release, \
         got {w} -- a release written into `sigma_x[{W2}]` takes `w`'s \
         own stiffness out of the operator and it walks toward \
         {}",
        -W_PULL,
    );
    // And the answer for `x` is still right, so the assertion above is
    // the only thing standing between a misrouted write and a green
    // run.
    let truth = closed_form(1.3 + dp);
    assert!(
        (x - truth).abs() < 1e-6,
        "x walked to {x} against a closed form of {truth}",
    );
}

// ---------------------------------------------------------------
// Leg 1's dimension, for the `s` block: does a row scaling move the
// answer?
// ---------------------------------------------------------------
//
// `bound_context` converts the `s` block's box and base point out of
// the algorithm's frame with `natural_units_factor()`, exactly as the
// `x` block is converted with `variable_scaling()`. Under unit scaling
// both conversions are the identity, so every other test in this file
// -- and the pre-fix table in the header -- is blind to them. That is
// the corpus-uniform-in-the-dimension-the-change-acts-on shape, and
// `sens_invariance_legs.rs` leg 1 exists because `205bb67` shipped
// through exactly it: the corrector "added the scaled iterate to
// bounds in the model's units, which coincide only at unit scaling,
// which is every fixture it had."
//
// Both arms below run under `user-scaling` with the same option and
// the same objective factor. The only difference is whether the TNLP
// hands factors back, per leg 1's discipline. The row factor is the
// load-bearing one: it is what makes `natural_units_factor` non-unit
// on the `s` rows.

/// Per-variable factors on `[x, lam]`, spread over decades.
const D_ROW: [Number; N] = [4.0, 1.0e-2];
/// Per-row factors on `[g0 (the pin), g1 (the cap)]`. `g1`'s is the
/// one this leg is about.
const G_ROW: [Number; 2] = [3.0, 2.5e2];

fn configured_user_scaling() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    for (k, v) in [
        ("tol", 1e-10),
        ("constr_viol_tol", 1e-10),
        ("compl_inf_tol", 1e-10),
        ("dual_inf_tol", 1e-8),
        ("bound_relax_factor", 0.0),
    ] {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

fn solved_scaled(lam: Number, scaling: Option<(Vec<Number>, Vec<Number>)>) -> Solver {
    let mut solver = Solver::new(
        configured_user_scaling(),
        Rc::new(RefCell::new(RowLimit { lam, scaling })) as Rc<RefCell<dyn TNLP>>,
    );
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "scaled solve at lam={lam} failed: {status:?}",
    );
    solver
}

fn scaled_pair(lam: Number) -> (Solver, Solver) {
    (
        solved_scaled(lam, None),
        solved_scaled(lam, Some((D_ROW.to_vec(), G_ROW.to_vec()))),
    )
}

#[test]
fn leg_scaling_the_arms_really_do_run_under_different_factors() {
    // Assert the split rather than assume it -- an arm that silently
    // stopped applying its factors would take the whole leg with it,
    // which is `the_fixtures_take_different_branches`' lesson in
    // `sens_resolve_oracle.rs`.
    let (plain, scaled) = scaled_pair(1.3);
    let dims = scaled.block_dims().expect("block_dims");
    assert_eq!(dims[1], 1, "the scaled arm must still have one slack");
    assert!(
        plain.converged().is_some() && scaled.converged().is_some(),
        "both arms must converge",
    );
    // Same answer in the model's own units, from two different
    // internal frames.
    for s in [&plain, &scaled] {
        let x = s.converged().expect("converged").x[0];
        assert!(
            (x - CAP).abs() < 1e-7,
            "both arms must sit on the cap in natural units, got {x}",
        );
    }
}

#[test]
fn leg_scaling_a_row_limit_is_reached_at_the_same_place_under_a_row_scaling() {
    let (plain, scaled) = scaled_pair(0.8);
    let a = walk_solver(&plain, 0.5);
    let b = walk_solver(&scaled, 0.5);
    let truth = closed_form(0.8 + 0.5);
    for (name, w) in [("plain", &a), ("scaled", &b)] {
        assert!(
            w.x <= CAP + OUT,
            "{name}: the walk left the box at {}, cap {CAP}",
            w.x,
        );
        assert!(
            (w.x - truth).abs() < 1e-6,
            "{name}: walked to {} against a closed form of {truth}",
            w.x,
        );
    }
    let (pa, pb) = (a.on_the_cap(), b.on_the_cap());
    assert_eq!(
        (pa.len(), pb.len()),
        (1, 1),
        "both arms must record exactly one cap event; plain {pa:?}, scaled {pb:?}",
    );
    assert_eq!(
        (pa[0].1, pa[0].2),
        (pb[0].1, pb[0].2),
        "the side and the direction moved under the change of variables",
    );
    assert!(
        (pa[0].0 - pb[0].0).abs() < 1e-6,
        "the breakpoint moved under the change of variables: plain {}, scaled {}",
        pa[0].0,
        pb[0].0,
    );
}

#[test]
fn leg_scaling_a_row_limit_is_released_at_the_same_place_under_a_row_scaling() {
    // The release arm is the one that reads the `s` diagonal, so it is
    // the one a frame error in `natural_units_factor` reaches.
    let (plain, scaled) = scaled_pair(1.3);
    let a = walk_solver(&plain, -0.5);
    let b = walk_solver(&scaled, -0.5);
    let truth = closed_form(1.3 - 0.5);
    for (name, w) in [("plain", &a), ("scaled", &b)] {
        assert!(
            (w.x - truth).abs() < 1e-6,
            "{name}: walked to {} against a closed form of {truth}",
            w.x,
        );
    }
    let (pa, pb) = (a.on_the_cap(), b.on_the_cap());
    assert_eq!(
        (pa.len(), pb.len()),
        (1, 1),
        "both arms must record exactly one cap release; plain {pa:?}, scaled {pb:?}",
    );
    assert_eq!((pa[0].1, pa[0].2), (pb[0].1, pb[0].2));
    assert!(
        (pa[0].0 - pb[0].0).abs() < 1e-6,
        "the release breakpoint moved under the change of variables: \
         plain {}, scaled {}",
        pa[0].0,
        pb[0].0,
    );
}
