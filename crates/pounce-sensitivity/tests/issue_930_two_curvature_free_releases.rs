//! gh#930: two curvature-free releases, and the operator the pin rides on.
//!
//! # The obstruction
//!
//! The path walk holds a bound it reached by applying a Schur row to
//! the *factored released system*: solve `K w - E du = r` subject to
//! `Eᵀ w = 0`, where `E` picks the held primal rows. The Schur
//! complement it builds is `-Eᵀ K⁻¹ E`, so `K⁻¹` has to exist before
//! a single hold goes on.
//!
//! Releasing a bound takes that bound's `Sigma` off `K`'s diagonal.
//! On a model with no curvature the diagonal is then exactly zero, and
//! two released variables that share a constraint row are left with
//! *linearly dependent* stationarity rows. `K` is singular, and the
//! walk reported:
//!
//! ```text
//! step_along_path: augmented solve failed (holds [0, 1], released [9, 7])
//! ```
//!
//! That refusal was gh#928's improvement on the silence that preceded
//! it -- the same step used to return a point outside the box without
//! comment. It was still a refusal.
//!
//! # The fix, and why it costs nothing in accuracy
//!
//! The pin is not what changed. What changed is the operator it is
//! applied to: on a failure the walk retries against
//! `K_reg = K_released + Σ_pin E Eᵀ`, whose pinned diagonals are
//! raised until it is invertible. The *same* Schur row then goes on
//! top, and `Eᵀ w = 0` says every pinned coordinate of `w` is zero --
//! which annihilates the term that was added. The system actually
//! solved is
//!
//! ```text
//! K_released w - E du = r,     Eᵀ w = 0
//! ```
//!
//! the released one, exactly, with `du` in the identical frame and
//! units. Nothing is converted, and a walk that takes the plain
//! operator on one segment and the regularized one on the next is
//! still accumulating a single `h.mult` per hold.
//!
//! `Σ_pin` is the gh#737 ceiling (`sigma_pin_cap` of the row's largest
//! constraint coefficient): the stiffest diagonal the row's own
//! couplings still survive. Stiff and reachable, not infinite -- the
//! held coordinate creeps by roundoff, which is what
//! `the_pin_is_stiff_not_infinite` bounds.
//!
//! # What is measured here
//!
//! * the previously-refused walk against a **re-solve at the
//!   perturbed parameter** (`/sens-review` entry 5: the only guard
//!   that reads an outside number);
//! * that the plain operator really is singular on this walk, so the
//!   fallback is reached rather than decorative;
//! * that the two operators **agree** on a working set where both run
//!   -- in `d` and in `du`, which is the claim that lets the walk
//!   switch between them mid-path;
//! * both branches of *how* the plain operator fails. Rank deficiency
//!   two and it refuses; deficiency one and the factorization absorbs
//!   it, the solve returns `Ok`, and a held row comes back holding a
//!   fifth of the step. The second is why the fallback is triggered on
//!   the pinned rows' residual and not on the error.
//!
//! # The fixture
//!
//! ```text
//! min  x + 2 y + 3 w
//! s.t. g0: x + y + w - lam = 0
//!      g1: lam = lam0            (the pin)
//!      x in [0, 1],  y in [0, 1],  w >= 0
//! ```
//!
//! Cheapest-first, so `x` fills to 1, then `y` to 1, then `w` runs:
//!
//! ```text
//! lam <= 1:      x = lam,  y = 0,        w = 0
//! 1 <= lam <= 2: x = 1,    y = lam - 1,  w = 0
//! lam >= 2:      x = 1,    y = 1,        w = lam - 2
//! ```
//!
//! The Lagrangian Hessian is identically zero, so no bound here can be
//! classified and the walk is on its own. Basing just below `lam = 1`
//! and stepping past `lam = 2` makes it release `x`'s upper bound
//! (reached and then held) *and* `y`'s lower bound (its multiplier
//! hits zero) -- two curvature-free releases sharing `g0`.
//!
//! One test perturbs it: a little curvature on `w` **only**. `x` and
//! `y`, whose bounds are the released ones, still have none, so their
//! rows are still dependent -- but `w`'s is no longer dependent on
//! them, and that one rank is the whole difference between a refusal
//! and a silently wrong answer.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::activity::UNIDENTIFIED;
use pounce_sensitivity::{PathOperator, Solver};

/// Column order is `[x, y, w, lam]`; the pin is constraint 1.
const N: usize = 4;
const PIN: Index = 1;
const COST: [Number; N] = [1.0, 2.0, 3.0, 0.0];
const LO: [Number; N] = [0.0, 0.0, 0.0, -1.0e19];
const HI: [Number; N] = [1.0, 1.0, 1.0e19, 1.0e19];

struct TwoReleases {
    lam: Number,
    /// Per-variable curvature on `[x, y, w]`. All zero is the model
    /// the file is about; `the_plain_operator_can_also_fail_silently`
    /// puts a little on `w` alone, which leaves the two released rows
    /// dependent but drops the deficiency from two to one.
    curv: [Number; 3],
}

impl TNLP for TwoReleases {
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
        // divide by, and carrying zero, so it finds it below the
        // identification floor. That zero is the whole subject here.
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
    solved_curved_at(lam, [0.0; 3])
}

fn solved_curved_at(lam: Number, curv: [Number; 3]) -> Solver {
    let mut solver = Solver::new(
        configured(),
        Rc::new(RefCell::new(TwoReleases { lam, curv })) as Rc<RefCell<dyn TNLP>>,
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

/// The closed form above. The re-solve is checked against it so a
/// wrong oracle cannot quietly certify a wrong walk.
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

/// The bound-multiplier rows this file names, derived from
/// `block_dims` rather than written down, and asserted to be the shape
/// the prose describes.
///
/// The KKT blocks are `[x, s, y_c, y_d, z_L, z_U, v_L, v_U]`. Every
/// variable here carries a lower bound so `z_L` is `[x, y, w]`; only
/// `x` and `y` carry an upper one so `z_U` is `[x, y]`. Reading one
/// block's `k`th row as another's is gh#450, which is why this is
/// computed and checked instead of hard-coded.
struct Rows {
    y_lo: usize,
    x_hi: usize,
}

fn rows_of(solver: &Solver) -> Rows {
    let d = solver.block_dims().expect("block_dims");
    assert_eq!(
        d,
        [4, 0, 2, 0, 3, 2, 0, 0],
        "the fixture's KKT shape changed; every row index below is derived \
         from it and the prose names which variable each one is",
    );
    let z_l = d[0] + d[1] + d[2] + d[3];
    let z_u = z_l + d[4];
    // `z_L` is `[x, y, w]`, so `x`'s lower bound is `z_l` itself and
    // `y`'s is the row after it. Only `x`'s lower bound goes unnamed
    // below, because nothing here releases it.
    Rows {
        y_lo: z_l + 1,
        x_hi: z_u,
    }
}

/// Just below the first kink, stepping past the second: the walk must
/// release `x`'s upper bound and `y`'s lower one, and neither has
/// curvature.
const LAM0: Number = 1.0 - 1e-6;
const DP: Number = 1.2;

#[test]
fn the_classifier_is_blind_here() {
    // Without this the file could be passing on the gh#852 weak-row
    // path rather than the one it describes.
    let solver = solved_at(LAM0);
    let report = solver.classify_activity().expect("classify_activity");
    assert_eq!(
        [
            report.var_status[0],
            report.var_status[1],
            report.var_status[2]
        ],
        [UNIDENTIFIED; 3],
        "every bound here must be unclassifiable",
    );
    assert!(
        solver
            .weakly_active_bounds()
            .expect("weakly_active_bounds")
            .is_empty(),
        "`weakly_active_bounds` must be empty, or the walk is not on its own",
    );
}

/// The plain released operator is genuinely singular on this working
/// set, so the fallback is reached rather than decorative.
///
/// This is the test that keeps the fix honest. Every other test here
/// would stay green if `path_direction` had simply started succeeding
/// for some unrelated reason; this one names the operator that fails
/// and the operator that does not.
#[test]
fn the_plain_operator_is_singular_and_the_regularized_one_is_not() {
    let solver = solved_at(LAM0);
    let r = rows_of(&solver);
    let released = [r.x_hi as Index, r.y_lo as Index];
    let held = [0 as Index, 1 as Index];

    let plain = solver.path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Plain);
    let reg =
        solver.path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Regularized);

    // `path_direction` itself falls back internally, so the plain
    // operator has to be asked for by name to see it fail. It is not
    // reachable through `parametric_step_path` any more, by design.
    assert!(
        plain.is_err(),
        "releasing two curvature-free variables that share g0 must leave \
         the plain operator singular -- if it does not, this fixture has \
         stopped reaching the branch the file is about",
    );
    let msg = format!("{:?}", plain.unwrap_err());
    assert!(
        msg.contains("augmented solve failed"),
        "the failure must be the Schur solve, not something upstream: {msg}",
    );
    let (d, du) = reg.expect("the regularized operator must answer");
    assert_eq!(du.len(), held.len(), "one force per hold");
    for &h in &held {
        let h = h as usize;
        assert!(
            d[h].abs() < 1e-9,
            "the Schur row must hold primal row {h} at zero, got {}",
            d[h],
        );
    }
}

/// The two operators are the same answer where both run.
///
/// This is the claim that lets a walk take one on one segment and the
/// other on the next while accumulating a single multiplier per hold:
/// `Eᵀ w = 0` annihilates the diagonal the regularized operator adds,
/// so the system solved is the released one either way.
#[test]
fn the_two_operators_agree_where_both_run() {
    let solver = solved_at(LAM0);
    let r = rows_of(&solver);
    // Working sets the *plain* operator survives: at most one
    // curvature-free release, so the stationarity rows stay
    // independent.
    let cases: [(&str, Vec<Index>, Vec<Index>); 4] = [
        ("nothing released, x held", vec![], vec![0]),
        ("nothing released, x and y held", vec![], vec![0, 1]),
        ("x's upper released, y held", vec![r.x_hi as Index], vec![1]),
        ("y's lower released, x held", vec![r.y_lo as Index], vec![0]),
    ];
    for (label, released, held) in cases {
        let (d0, u0) = solver
            .path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Plain)
            .unwrap_or_else(|e| panic!("{label}: the plain operator must run here: {e:?}"));
        let (d1, u1) = solver
            .path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Regularized)
            .unwrap_or_else(|e| panic!("{label}: the regularized operator must run here: {e:?}"));
        let scale = d0.iter().fold(1.0 as Number, |a, v| a.max(v.abs()));
        let ed = (0..d0.len())
            .map(|i| (d0[i] - d1[i]).abs())
            .fold(0.0, Number::max);
        assert!(
            ed < 1e-9 * scale,
            "{label}: the two operators must give the same direction, off \
             by {ed:e} (scale {scale:e})",
        );
        let uscale = u0.iter().fold(1.0 as Number, |a, v| a.max(v.abs()));
        let eu = (0..u0.len())
            .map(|i| (u0[i] - u1[i]).abs())
            .fold(0.0, Number::max);
        assert!(
            eu < 1e-9 * uscale,
            "{label}: the two operators must report the same hold force, \
             off by {eu:e} (scale {uscale:e}) -- {u0:?} vs {u1:?}",
        );
    }
    // And the pin is not vacuous on these sets: at least one of them
    // has to carry a force, or "they agree" is agreement about zero.
    let (_, u) = solver
        .path_direction_decided(&[PIN], &[DP], &[], &[0], PathOperator::Preferred)
        .expect("plain");
    assert!(
        u.iter().any(|v| v.abs() > 1e-6),
        "the held row must actually be carrying a force: {u:?}",
    );
}

/// The step gh#930 refused, against a re-solve at the perturbed
/// parameter.
///
/// `/sens-review` entry 5: this is the only number here that the
/// sensitivity layer did not produce. Internal consistency cannot
/// catch a step that is self-consistently wrong, and a regularized
/// operator is exactly the shape of change that could be.
#[test]
fn the_refused_walk_reproduces_the_resolve() {
    let solver = solved_at(LAM0);
    let base = solver.converged().expect("converged").x.clone();
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[DP], 64)
        .expect("the walk must answer, not refuse");
    let got = [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]];
    let truth = resolve_at(LAM0 + DP);
    let err = (0..3)
        .map(|i| (got[i] - truth[i]).abs())
        .fold(0.0, Number::max);
    assert!(
        err < 1e-9,
        "walk {got:?} vs re-solve {truth:?}, off by {err:e} over {} segments",
        segs.len(),
    );
    assert!(
        segs.len() >= 2,
        "crossing two kinks must record breakpoints, got {}",
        segs.len(),
    );
}

/// The same, swept: both kinks, both directions, several slacks.
///
/// One `(lam0, dp)` is a fixture; a sweep is the claim that the
/// fallback is not tuned to one working set. Every combination that
/// crosses at least one kink is included, including the ones that
/// never reach the fallback at all -- a fix that broke the ordinary
/// path to rescue this one would show up here.
#[test]
fn the_walk_reproduces_the_resolve_across_the_grid() {
    let mut worst: (Number, Number, Number) = (0.0, 0.0, 0.0);
    let mut answered = 0usize;
    for slack in [1e-2, 1e-4, 1e-6, 1e-8] {
        for kink in [1.0, 2.0] {
            for side in [-1.0, 1.0] {
                let lam0 = kink + side * slack;
                for dp in [-1.5, -0.7, -0.2, 0.2, 0.7, 1.5] {
                    if lam0 + dp < 0.05 {
                        continue;
                    }
                    let solver = solved_at(lam0);
                    let base = solver.converged().expect("converged").x.clone();
                    let (dx, _) = solver
                        .parametric_step_path(&[PIN], &[dp], 64)
                        .unwrap_or_else(|e| {
                            panic!("lam0={lam0} dp={dp}: the walk must answer: {e:?}")
                        });
                    let got = [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]];
                    let truth = resolve_at(lam0 + dp);
                    let err = (0..3)
                        .map(|i| (got[i] - truth[i]).abs())
                        .fold(0.0, Number::max);
                    answered += 1;
                    if err > worst.0 {
                        worst = (err, lam0, dp);
                    }
                }
            }
        }
    }
    assert!(
        answered >= 40,
        "the grid must actually run: {answered} combinations",
    );
    assert!(
        worst.0 < 1e-7,
        "worst disagreement with the re-solve {:e} at lam0={} dp={} over \
         {answered} combinations",
        worst.0,
        worst.1,
        worst.2,
    );
}

/// The **silent** half of the same defect, and the branch the choice
/// between the operators is actually made on.
///
/// How loudly the plain operator fails is a property of the model, not
/// of the defect. With every variable curvature-free the two released
/// stationarity rows coincide *and* `w`'s is dependent on them too:
/// deficiency two, the augmented solve refuses, and
/// `the_plain_operator_is_singular_and_the_regularized_one_is_not`
/// pins that. Put a little curvature on `w` alone and `x`'s and `y`'s
/// rows still coincide -- deficiency one -- and the factorization
/// absorbs it: the solve returns `Ok`, and the held row it was asked
/// to keep at zero comes back holding a fifth of the step.
///
/// So `path_direction` cannot decide between the operators on whether
/// the solve errored. It decides on the pinned rows' **residual**,
/// which is `pin_residual` in `pounce-sens-core`, referenced to the
/// motion the pin was asked to remove and not to the compound
/// vector's norm -- the multiplier rows here run at `3e11`, and a
/// residual divided by that reads `3e-12` on a pin that missed by
/// 200%.
///
/// Without this test the file would be evidence about one branch of a
/// two-branch rule, which is the shape this repo has shipped defects
/// through more than once.
#[test]
fn the_plain_operator_can_also_fail_silently() {
    // Curvature on `w` only: `x` and `y`, the two variables whose
    // bounds are released, still have none.
    let solver = solved_curved_at(LAM0, [0.0, 0.0, 1e-3]);
    let r = rows_of(&solver);
    let released = [r.x_hi as Index, r.y_lo as Index];
    let held = [0 as Index, 1 as Index];

    let (d_plain, _) = solver
        .path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Plain)
        .expect(
            "this is the SILENT branch: the plain operator must return an              answer here, or the fixture has drifted onto the loud one and              the residual test below proves nothing",
        );
    let moved = d_plain[0].abs().max(d_plain[1].abs());
    assert!(
        moved > 1e-3,
        "the plain operator must leave a held row visibly off zero for          this test to have a subject, got {:e} and {:e}",
        d_plain[0],
        d_plain[1],
    );

    // What the walk actually runs holds them, having noticed.
    let (d, du) = solver
        .path_direction_decided(&[PIN], &[DP], &released, &held, PathOperator::Preferred)
        .expect("the preferred operator must answer");
    assert_eq!(du.len(), held.len(), "one force per hold");
    for &h in &held {
        let h = h as usize;
        assert!(
            d[h].abs() < 1e-12 * moved,
            "primal row {h} must be held at zero, got {:e} against the              plain operator's {:e}",
            d[h],
            moved,
        );
    }

    // And the walk end to end, which is what a user sees. Before the
    // residual test this returned a point 1.0e-5 outside `y`'s upper
    // bound, and gh#928's repair turned that into a refusal rather
    // than an answer.
    let base = solver.converged().expect("converged").x.clone();
    let truth = resolve_at(LAM0 + DP);
    let (dx, segs) = solver
        .parametric_step_path(&[PIN], &[DP], 64)
        .expect("the walk must answer on the silent branch too");
    let got = [base[0] + dx[0], base[1] + dx[1], base[2] + dx[2]];
    let err = (0..3)
        .map(|i| (got[i] - truth[i]).abs())
        .fold(0.0, Number::max);
    assert!(
        err < 1e-9,
        "walk {got:?} vs re-solve {truth:?}, off by {err:e} over {} segments",
        segs.len(),
    );
}

/// The pin is stiff, not infinite, and the creep it allows stays
/// inside the box.
///
/// `Σ_pin` is the gh#737 ceiling, not `INFINITY` -- an infinite
/// diagonal is a `NaN` in the factorization, not a stiffer pin. So a
/// held coordinate moves by roundoff under the force it carries, and
/// the property gh#928 exists to defend is that the roundoff points
/// *inward* far enough that the answer never leaves the box by more
/// than the walk's own `eps`.
#[test]
fn the_pin_is_stiff_not_infinite() {
    let solver = solved_at(LAM0);
    let base = solver.converged().expect("converged").x.clone();
    let (dx, _) = solver
        .parametric_step_path(&[PIN], &[DP], 64)
        .expect("the walk must answer");
    for i in 0..2 {
        let v = base[i] + dx[i];
        let past = (LO[i] - v).max(v - HI[i]);
        assert!(
            past < 1e-11,
            "coordinate {i} left the box by {past:e} (at {v})",
        );
    }
    assert!(
        base[2] + dx[2] >= -1e-11,
        "w left its lower bound at {}",
        base[2] + dx[2],
    );
}

// # Mutation table
//
// Every row was run against this file. A row with no red test is a
// change this file cannot see, and is listed as such rather than
// omitted.
//
// | change                                                          | red |
// |-----------------------------------------------------------------|-----|
// | `solve_released_pinned` returns `false` (no fallback)            | `the_plain_operator_...`, `the_refused_walk_...`, `the_pin_is_stiff_...` |
// | drop the Schur row and keep only the raised diagonal (penalty pin) | `the_two_operators_agree_...` (`du` leaks ~1%, `d` on the held rows reads 3.8e-3 instead of 0) |
// | `path_pin_add` returns `INFINITY` instead of the gh#737 ceiling  | `the_plain_operator_...` (`NaN` out of the factorization) |
// | `path_pin_add` returns 1.0 (a reachable but limp diagonal)       | `the_two_operators_agree_...` |
// | regularize on the *first* attempt instead of on failure          | nothing here; it is a cost regression, not a wrong answer -- the factorization cache keys on the sigma tag, so every back-solve in the segment re-factors |
// | fall back only on `Err`, not on the pin residual                 | `the_plain_operator_can_also_fail_silently` (the walk refuses with `y` 1.0e-5 outside its bound) |
// | `PIN_TAKE_RTOL` loosened to `1e-8`                               | `the_plain_operator_can_also_fail_silently` -- the residual it has to catch is `8.1e-9`, and this was the draft threshold |
// | `pin_residual` divides by `max |d|` instead of by the pin's own rhs | `the_plain_operator_can_also_fail_silently`; the multiplier rows run at `3e11`, so a 200% miss reads `3e-12` |
// | return `plain` whenever it is `Ok`, instead of the smaller residual | `the_plain_operator_can_also_fail_silently` |
// | accept multiplier rows as pins (drop the `r >= n_p` guard)       | nothing here; every pin this fixture makes is primal. `issue_928_a_limit_written_as_a_row.rs` owns the `s` block |
// | swap `x_hi` and `y_lo` in `rows_of`                              | `rows_of`'s own `block_dims` assertion does not catch it; `the_plain_operator_...` does, because the swapped set is not singular |
//
// # What this file is not evidence about
//
// * **How singular an operator the ceiling can lift.** One dependent
//   pair, one shared row. A model whose released system is singular
//   in a direction the pinned rows do not span is not rescued by this
//   fallback at all, and nothing here says otherwise -- the walk would
//   report the same failure it used to.
// * **Scaling.** Everything here runs unit-scaled. `Σ_pin` is built
//   from the constraint coefficients in the frame the factor lives
//   in, so a scaled model reaches a different ceiling; leg 1 of
//   `sens_invariance_legs.rs` owns that dimension.
// * **Cost.** The regularized operator rebuilds its diagonal per
//   solve and therefore misses the factorization cache. That it is
//   the *second* choice is asserted by nothing here.
// * **The convex arm.** `pounce-convex` reaches `step_along_path`
//   through its own backsolver, which does not implement
//   `solve_released_pinned`, so on a convex model with this shape the
//   preferred operator has only the plain one to offer.
//   `crates/pounce-convex/tests/` owns that arm.
// * **Where the two residual populations separate on a large model.**
//   `PIN_TAKE_RTOL` sits between `1.9e-16` and `8.1e-9` as measured
//   on four-variable models. It is a *shortcut* threshold -- above it
//   both operators run and the better-pinned answer is returned -- so
//   a large model that lands between them pays a refactorization and
//   not an answer. That is a claim about the code path, not a
//   measurement, and nothing here measures it.
// * **Magnitude.** Four variables and one shared row.
