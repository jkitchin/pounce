//! The corrector on a base solve the gh#737 sigma ceiling touched, or
//! that crossed over into the gh#654 declared frame.
//!
//! In both of those cases the sensitivity layer stores a base-point
//! diagonal for the back-solves through the held factor, and
//! `barrier_sigma_x()` returns the stored copy. The corrector must
//! not use it: it factors its own operator at the predicted point,
//! so `corrector_sigma` rebuilds both diagonal blocks there, with
//! the frame rule and the ceiling re-derived at that point, and this
//! file pins that a correction on such solves acts instead of
//! stalling.
//!
//! The fixture is gh#737's cap trigger (a variable held between a
//! binding equality and its own bound, the multiplier split not
//! unique) welded to a curvature workload on a second parameter: the
//! perturbation moves `y` through `exp(y)` curvature, which the plain
//! step misses at second order and a correction closes.
//!
//! # Measured (2026-08-28), error against a tol 1e-10 re-solve
//!
//! ```text
//!                                  plain err   c1        c8
//!  ceiling engaged, frozen copy    1.48e-1     1.48e-1   1.48e-1   improved: false
//!  ceiling engaged, rebuilt        1.48e-1     1.05e-2   1.05e-2   improved: true
//!  bound removed (control)         1.48e-1     1.04e-2   5.0e-8    improved: true
//! ```
//!
//! The frozen row is the pre-`corrector_sigma` behaviour, the double
//! column's no-progress signature: the first iteration fails to
//! reduce the residual against the mixed operator and the
//! no-improvement rule keeps the handed step at every budget. The
//! rebuilt row's first iteration matches the control's, and the
//! plateau past it belongs to the degenerate rows themselves: the
//! trial residual at the non-unique multiplier split stops falling,
//! measured identically under a forced raw uncapped diagonal
//! (`6.9e27` on this geometry), so no diagonal choice moves it. The
//! control, with nothing degenerate, continues to the floor.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

mod common;
use common::compare::{Frame, assert_same_correction, assert_same_correction_to};
use common::outside_engine::{
    RecordingTnlp, Relaxation, correct_outside, outside_engine, same_classification_of,
};

/// Where the parameter row holds `pv`, and equally `x`'s upper bound.
const PIN: Number = 1.0;
/// The second parameter row's base value.
const Q0: Number = 1.0;
/// The perturbation asked of the second parameter row.
const DELTA: Number = -0.45;

/// ```text
/// min  ½(x − t)² + ½(w − 1)² + exp(y) − q·y + ½(b − ½ − (q − Q0))²
/// s.t. g₀:  x − pv = 0
///      g₁:      pv = PIN      ← holds x on its own upper bound
///      g₂:  w − x  = 0
///      g₃:      q  = q0       ← the perturbed parameter row
///      0 ≤ x ≤ PIN,  0 ≤ b,  pv, w, q, y free
/// ```
///
/// The `x`/`pv`/`w` block is gh#737's cap trigger verbatim and is not
/// perturbed. The `q` block is the correction workload: at the base,
/// `y* = ln(Q0) = 0` and `b* = ½`; at `q0 + DELTA` the re-solve has
/// `y* = ln(0.55)` while the linear step reaches `ln'(1)·DELTA`, and
/// `b* = 0.05`, one tenth of its base slack.
struct CeilingWithWorkload {
    /// Objective target for `x`, above `PIN`. `2.0` trips the
    /// ceiling; `1e3` converges through a different multiplier split
    /// and lands two hundred times short of it (measured on gh#737's
    /// fixture, unchanged here).
    t: Number,
    /// The second parameter row's right-hand side.
    q0: Number,
    /// Whether `x` carries its upper bound. `false` removes only the
    /// bound, so the equality still holds `x` at `PIN` but nothing is
    /// degenerate and no ceiling can engage.
    bounded: bool,
    /// Per-variable `user-scaling` factors, or `None` to decline
    /// scaling. The declared frame is read at the iterate, so it has to
    /// be in the iterate's own units; every other fixture here is
    /// unit-scaled, which is the configuration `205bb67` was invisible
    /// in.
    x_scaling: Option<[Number; 6]>,
    /// Starting point override, for warm-starting the truth solve.
    start: Option<Vec<Number>>,
}

impl TNLP for CeilingWithWorkload {
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(d) = self.x_scaling else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(&d);
        *req.use_g_scaling = false;
        true
    }

    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 6,
            m: 4,
            nnz_jac_g: 6,
            nnz_h_lag: 7,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // x, pv, w, q, y, b
        b.x_l[0] = 0.0;
        b.x_u[0] = if self.bounded { PIN } else { 1.0e19 };
        for i in 1..5 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        b.x_l[5] = 0.0;
        b.x_u[5] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = PIN;
        b.g_u[1] = PIN;
        b.g_l[2] = 0.0;
        b.g_u[2] = 0.0;
        b.g_l[3] = self.q0;
        b.g_u[3] = self.q0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        match &self.start {
            Some(x0) => sp.x.copy_from_slice(x0),
            None => {
                sp.x[..3].fill(0.5);
                sp.x[3] = Q0;
                sp.x[4] = 0.0;
                sp.x[5] = 0.5;
            }
        }
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let d = x[5] - 0.5 - (x[3] - Q0);
        Some(
            0.5 * (x[0] - self.t).powi(2) + 0.5 * (x[2] - 1.0).powi(2) + x[4].exp() - x[3] * x[4]
                + 0.5 * d * d,
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let d = x[5] - 0.5 - (x[3] - Q0);
        g[0] = x[0] - self.t;
        g[1] = 0.0;
        g[2] = x[2] - 1.0;
        g[3] = -x[4] - d;
        g[4] = x[4].exp() - x[3];
        g[5] = d;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - x[1];
        g[1] = x[1];
        g[2] = x[2] - x[0];
        g[3] = x[3];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0, 0), (0, 1), (1, 1), (2, 2), (2, 0), (3, 3)]
                    .iter()
                    .enumerate()
                {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, -1.0, 1.0, 1.0, -1.0, 1.0]);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // lower triangle: (0,0) (2,2) (3,3) (4,4) (5,5) (4,3) (5,3)
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0, 0), (2, 2), (3, 3), (4, 4), (5, 5), (4, 3), (5, 3)]
                    .iter()
                    .enumerate()
                {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let y = x.map_or(0.0, |x| x[4]);
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor;
                values[3] = obj_factor * y.exp();
                values[4] = obj_factor;
                values[5] = -obj_factor;
                values[6] = -obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

fn solved(
    t: Number,
    q0: Number,
    bounded: bool,
    crossover: bool,
    start: Option<Vec<Number>>,
    tol: Option<Number>,
) -> Solver {
    solved_in_frame(t, q0, bounded, crossover, start, tol, 0.0, None)
}

/// `solved`, with the two knobs the declared-frame recompute is
/// sensitive to and every other fixture here pins at one value:
/// `relax` (`bound_relax_factor`) and `x_scaling`.
#[allow(clippy::too_many_arguments)]
fn solved_in_frame(
    t: Number,
    q0: Number,
    bounded: bool,
    crossover: bool,
    start: Option<Vec<Number>>,
    tol: Option<Number>,
    relax: Number,
    x_scaling: Option<[Number; 6]>,
) -> Solver {
    let app = app(crossover, tol, relax, x_scaling.is_some());
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CeilingWithWorkload {
        t,
        q0,
        bounded,
        x_scaling,
        start,
    }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must converge (t={t:e}, q0={q0}); got {status:?}",
    );
    solver
}

/// The options every solve of this fixture runs with.
fn app(
    crossover: bool,
    tol: Option<Number>,
    relax: Number,
    user_scaling: bool,
) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_string_value("sb", "yes", true, false).unwrap();
        o.set_numeric_value("bound_relax_factor", relax, true, false)
            .unwrap();
        if user_scaling {
            o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
                .unwrap();
        }
        o.set_string_value(
            "crossover",
            if crossover { "yes" } else { "no" },
            true,
            false,
        )
        .unwrap();
        if let Some(tol) = tol {
            o.set_numeric_value("tol", tol, true, false).unwrap();
        }
    }
    app.initialize().unwrap();
    app
}

/// [`solved_in_frame`] at `t = 2`, `q0 = Q0` from the default start, with
/// the model wrapped to record what the solve returns.
fn solved_recorded(
    bounded: bool,
    crossover: bool,
    relax: Number,
    x_scaling: Option<[Number; 6]>,
) -> (Solver, Rc<RefCell<RecordingTnlp<CeilingWithWorkload>>>) {
    let model = Rc::new(RefCell::new(RecordingTnlp::new(CeilingWithWorkload {
        t: 2.0,
        q0: Q0,
        bounded,
        x_scaling,
        start: None,
    })));
    let tnlp: Rc<RefCell<dyn TNLP>> = model.clone();
    let mut solver = Solver::new(app(crossover, None, relax, x_scaling.is_some()), tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must converge; got {status:?}",
    );
    (solver, model)
}

fn truth(t: Number, bounded: bool, base_x: &[Number]) -> Vec<Number> {
    let s = solved(
        t,
        Q0 + DELTA,
        bounded,
        false,
        Some(base_x.to_vec()),
        Some(1e-10),
    );
    s.converged().expect("truth converged").x.clone()
}

fn dist(a: &[Number], b: &[Number]) -> Number {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

fn add(base: &[Number], step: &[Number]) -> Vec<Number> {
    base.iter().zip(step).map(|(&b, &s)| b + s).collect()
}

/// A base solve the ceiling touched corrects like any other: the
/// diagonal is rebuilt at the predicted point with the ceiling
/// re-derived there, never the stored base-point copy.
///
/// The first iteration takes the error down an order and the residual
/// fifteenfold, matching the no-ceiling control's first iteration.
/// The iterations then stop: the trial residual at the degenerate
/// rows, where the multiplier split between the bound and the
/// equality is not unique, does not fall further, and the measured
/// plateau is identical under a forced raw uncapped diagonal, so it
/// is the degeneracy's own property and no diagonal choice moves it.
/// The budget-8 error equals the budget-1 error here, pinned as a
/// measured fact rather than a target.
#[test]
fn a_ceiling_solve_corrects_at_the_predicted_point() {
    let solver = solved(2.0, Q0, true, false, None, None);
    let report = solver.classify_activity().expect("activity report");
    assert!(
        report.var_sigma[0] > 1e10 && report.var_sigma[0] < 1e20,
        "the fixture only says anything with the ceiling engaged: \
         sigma on x is {:e}",
        report.var_sigma[0],
    );
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, true, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    assert!(
        plain > 1e-1,
        "the workload must leave the plain step something to close: {plain:e}",
    );
    for budget in [1_usize, 8] {
        let (out, rep) = solver
            .correct_step(&[3], &[DELTA], &step, budget)
            .expect("corrector");
        let corrected = dist(&add(&base, &out[..n]), &want);
        assert!(
            rep.improved() && corrected < plain / 10.0,
            "budget {budget}: the predicted-point diagonal must let the \
             correction act: {plain:e} -> {corrected:e} (residual {:e} \
             -> {:e})",
            rep.initial_residual,
            rep.residual,
        );
    }
}

/// The identical workload with only the degenerate bound removed has
/// no ceiling and nothing degenerate, and converges to the floor: the
/// control separating the workload from the degenerate rows' plateau
/// above.
#[test]
fn the_same_workload_without_the_ceiling_corrects() {
    let solver = solved(2.0, Q0, false, false, None, None);
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, false, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    let (out, rep) = solver
        .correct_step(&[3], &[DELTA], &step, 8)
        .expect("corrector");
    let corrected = dist(&add(&base, &out[..n]), &want);
    assert!(
        rep.improved() && corrected < 1e-6 && corrected < plain * 1e-5,
        "the live operator closes the curvature: {plain:e} -> {corrected:e} \
         (residual {:e} -> {:e})",
        rep.initial_residual,
        rep.residual,
    );
}

/// The other way the stored diagonal freezes: a solve that crossed
/// over into the declared frame. `corrector_sigma` re-derives that
/// frame at the predicted point (the declared bounds are constants,
/// so the frame follows the iterate), and the correction acts the
/// same way it does on the interior ceiling solve above.
#[test]
fn a_crossover_solve_corrects_at_the_predicted_point() {
    let solver = solved(2.0, Q0, true, true, None, None);
    let base = solver.converged().expect("converged").x.clone();
    let want = truth(2.0, true, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let n = base.len();
    let plain = dist(&add(&base, &step[..n]), &want);
    assert!(
        plain > 1e-1,
        "the workload must leave the plain step something to close: {plain:e}",
    );
    let (out, rep) = solver
        .correct_step(&[3], &[DELTA], &step, 8)
        .expect("corrector");
    let corrected = dist(&add(&base, &out[..n]), &want);
    assert!(
        rep.improved() && corrected < plain / 10.0,
        "the declared frame must follow the iterate: {plain:e} ->          {corrected:e} (residual {:e} -> {:e})",
        rep.initial_residual,
        rep.residual,
    );
}

/// The declared-frame arm under a **non-unit change of variables**.
///
/// `corrector_sigma` forms `s = Pᵀx − b_l` from `curr.x` and
/// `declared_x_bounds()`. Those coincide at unit scaling, which is what
/// every other fixture in this file runs, and `205bb67` is the corrector
/// frame defect that was invisible for exactly that reason — in its own
/// words, the two "coincide only at unit scaling, which is every fixture
/// it had."
///
/// The reason this holds is worth stating, because it is not obvious
/// from `corrector_sigma`: variable scaling is applied by a TNLP
/// *wrapper* (`pounce_nlp::scaling_tnlp::wrap_with_scaling`) that sits
/// **outside** `OrigIpoptNlp`, so the box the core snapshots as
/// `declared_x_l` and the iterate it carries are both already in the
/// scaled frame. There is nothing to reapply, which is what
/// `declared_x_bounds`'s own comment says. This arm is what makes that
/// a checked fact rather than a read.
///
/// Asserted as **parity with the unscaled arm**, not against a
/// hand-derived value: the correction must close the same fraction of
/// the same gap, since a change of variables does not move the KKT
/// point. That is the criterion `variable_scaling_sensitivity.rs` uses.
#[test]
fn the_declared_frame_follows_the_iterate_under_a_change_of_variables() {
    // Deliberately spread over four orders, and non-unit on the pinned
    // variable and the workload variable both.
    const D: [Number; 6] = [1.0e2, 1.0e-2, 5.0, 1.0e-1, 2.0, 1.0e1];

    let unscaled = solved_in_frame(2.0, Q0, true, true, None, None, 0.0, None);
    let scaled = solved_in_frame(2.0, Q0, true, true, None, None, 0.0, Some(D));

    let closed = |solver: &Solver| -> (Number, Number) {
        let base = solver.converged().expect("converged").x.clone();
        let want = truth(2.0, true, &base);
        let step = solver
            .parametric_step_full(&[3], &[DELTA])
            .expect("full step");
        let n = base.len();
        let plain = dist(&add(&base, &step[..n]), &want);
        let (out, rep) = solver
            .correct_step(&[3], &[DELTA], &step, 8)
            .expect("corrector");
        assert!(rep.improved(), "the correction must act");
        (plain, dist(&add(&base, &out[..n]), &want))
    };

    let (plain_u, corr_u) = closed(&unscaled);
    let (plain_s, corr_s) = closed(&scaled);

    // The workload is the same problem either way, so the plain step
    // leaves the same gap. If this fails the two arms are not comparable
    // and the parity below would be meaningless.
    assert!(
        (plain_s - plain_u).abs() <= 1e-6 * plain_u.max(1.0),
        "the two arms must pose the same problem: {plain_u:e} vs {plain_s:e}",
    );
    assert!(
        corr_u < plain_u / 10.0 && corr_s < plain_s / 10.0,
        "both arms must correct: {plain_u:e} -> {corr_u:e}, {plain_s:e} -> {corr_s:e}",
    );
    // The invariant. A frame mix would show here as the scaled arm
    // closing a different fraction, because `s` would carry `d` while
    // the bound did not.
    let ratio = (corr_s / corr_u).abs();
    assert!(
        (0.1..=10.0).contains(&ratio),
        "the corrected error must not depend on the change of variables: \
         unscaled {corr_u:e}, scaled {corr_s:e} (ratio {ratio:e})",
    );
}

/// The declared-frame arm where the predicted point lands **outside the
/// declared box**.
///
/// `bound_context` builds the corrector's box from the **live relaxed**
/// bounds, so at a non-zero `bound_relax_factor` the predicted point may
/// sit up to `δ` outside a declared bound and `s = x − b_l` goes
/// negative. `declared_slack_floor` is what keeps that a
/// large-but-finite `Σ` rather than a negative diagonal or a NaN. Every
/// other fixture here sets `bound_relax_factor = 0.0`, so declared and
/// relaxed bounds coincide and that floor is never load-bearing.
///
/// **Reaching the branch needed a wider perturbation than the file's
/// `DELTA`, and that is the point of the constant below.** At -0.45 the
/// predicted point still lands inside the declared box — the lower
/// block's minimum raw slack is exactly `0`, the pinned variable on its
/// bound — so this test passed while testing nothing. At -0.9 the raw
/// slack reaches `-9.85e-9`, and it tracks `bound_relax_factor`
/// (`-9.99e-5` at `1e-2`), which is what identifies it as the relaxation
/// margin rather than as noise.
///
/// **What is asserted is the floor's guarantee, not convergence.** At
/// this perturbation the corrector declines — `improved()` is false and
/// the handed step comes back unchanged — and that is correct rather
/// than a defect: the same `-0.9` step at `bound_relax_factor = 0`,
/// where no slack goes negative, *does* report `improved()` and lands
/// **1.28e0 from the truth against the plain step's 4.51e-1**, i.e. it
/// "improves" a residual while moving the answer three times further
/// away. That is the gh#764 property the re-solve oracle exists for —
/// `improved()` plus a falling residual does not imply the answer is
/// close — and declining is the better of the two behaviours. Asserting
/// "the correction must act" here would have pinned the worse one.
#[test]
fn the_declared_frame_survives_a_point_outside_the_declared_box() {
    // The registered default, i.e. what a user actually gets.
    const RELAX: Number = 1e-8;
    // Wide enough that the predicted point leaves the declared box; see
    // the doc comment for the measurement.
    const WIDE: Number = -0.9;

    let solver = solved_in_frame(2.0, Q0, true, true, None, None, RELAX, None);
    let base = solver.converged().expect("converged").x.clone();
    let step = solver
        .parametric_step_full(&[3], &[WIDE])
        .expect("full step");
    let n = base.len();

    let (out, rep) = solver
        .correct_step(&[3], &[WIDE], &step, 8)
        .expect("corrector");

    // The floor's guarantee: a finite operator, hence a finite step. A
    // negative diagonal or a `z/0` would surface here as a NaN or an
    // infinity rather than as a bad number.
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the corrected step must be finite at a point outside the declared box",
    );
    assert!(
        rep.residual.is_finite() && rep.initial_residual.is_finite(),
        "residuals must be finite: {:e} -> {:e}",
        rep.initial_residual,
        rep.residual,
    );
    // Declining returns the caller's own step, put back into the box.
    // Pinned so that a corrector which learns to act here fails
    // deliberately rather than silently — at which point the assertion
    // to add is against a re-solve, not against `improved()`.
    //
    // "Declines" is a residual that does not *materially* fall, not the
    // strict `improved()`: the box-projected step's residual sits within
    // round-off of the handed step's, so which side of `<` it lands on is
    // the base point's last digits. gh#974 moved the crossover's base point
    // by ~5e-9 (a permanent row the rank probe used to drop is now kept),
    // and `improved()` flipped on a residual fall of 1.5e-11 in 0.4 — with
    // the returned step bit-identical to before, 0.4500 from the truth
    // against the plain step's 0.4514.
    let fall = (rep.initial_residual - rep.residual) / rep.initial_residual;
    assert!(
        fall <= 1e-8,
        "measured: the corrector declines at this perturbation (residual \
         {:e} -> {:e}); if it now acts, check it against a re-solve before \
         loosening this",
        rep.initial_residual,
        rep.residual,
    );
    // The property the decline protects (gh#764): whatever comes back is
    // no further from the truth than the step it was handed.
    let want_wide = truth(2.0, true, &base);
    assert!(
        dist(&add(&base, &out[..n]), &want_wide) <= dist(&add(&base, &step[..n]), &want_wide),
        "the declined step must not move away from the truth",
    );

    // And the narrower perturbation still corrects at the same relaxed
    // bounds, so the decline above is the step width, not the relaxation.
    let step_n = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let want = truth(2.0, true, &base);
    let plain = dist(&add(&base, &step_n[..n]), &want);
    let (out_n, rep_n) = solver
        .correct_step(&[3], &[DELTA], &step_n, 8)
        .expect("corrector");
    let corrected = dist(&add(&base, &out_n[..n]), &want);
    assert!(
        rep_n.improved() && corrected < plain / 10.0,
        "the declared frame must still correct at the default \
         bound_relax_factor: {plain:e} -> {corrected:e}",
    );
}

// ---------------------------------------------------------------------------
// The same corrections over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and asserts that the corrector, handed the same step, and the
// classification come out the same over an engine built from that report.

fn relaxation(relax: Number) -> Relaxation {
    if relax == 0.0 {
        Relaxation::None
    } else {
        Relaxation::Pounce {
            factor: relax,
            cap: 1e-4,
        }
    }
}

fn both_corrections(
    solver: &Solver,
    model: &Rc<RefCell<RecordingTnlp<CeilingWithWorkload>>>,
    relax: Number,
    delta: Number,
    budget: usize,
    frame: Frame,
    label: &str,
) {
    let step = solver
        .parametric_step_full(&[3], &[delta])
        .expect("full step");
    let pounce = solver
        .correct_step(&[3], &[delta], &step, budget)
        .expect("pounce's corrector");
    let eng = outside_engine(model, relaxation(relax));
    let outside =
        correct_outside(&eng, &[3], &[delta], &step, budget).expect("the outside corrector");
    assert_same_correction(label, &pounce, &outside, frame);
}

/// pounce's classification and the outside engine's, the engine marked as
/// reading a crossover vertex when the solve ran crossover.
fn both_classifications(
    solver: &Solver,
    model: &Rc<RefCell<RecordingTnlp<CeilingWithWorkload>>>,
    relax: Number,
    crossover: bool,
    label: &str,
) {
    let eng = outside_engine(model, relaxation(relax));
    let eng = if crossover {
        eng.after_crossover()
    } else {
        eng
    };
    same_classification_of(label, solver, eng);
}

/// pounce caps `x0`'s barrier diagonal at its gh#737 ceiling, `7.0e13`,
/// where `z/s` is `6.9e27`, because its augmented factorization loses the
/// step above the ceiling. The outside engine factors the eight-block matrix,
/// which keeps `z` and `s` apart, and needs no cap.
///
/// After one iteration the two corrected steps agree to `1e-6`, and both
/// land `1.048e-2` from the re-solved truth, from a plain step `1.478e-1`
/// away. After that they part. pounce's second iteration fails to improve on
/// its first, so it stops there at `1.048e-2`. The outside engine goes on to
/// `1.356e-3` by its third iteration, the point the crossover solve below
/// reaches with no cap, and its residual falls to `5.1e-9` by the eighth.
/// The ceiling costs pounce a factor of `7.7` in the answer here.
///
/// `Frame::Scaled` leaves the residuals out of the budget-1 comparison,
/// since the cap moves them by a few parts in a hundred thousand.
#[test]
fn a_ceiling_solve_corrects_at_the_predicted_point_over_an_outside_engine() {
    let (solver, model) = solved_recorded(true, false, 0.0, None);
    both_classifications(&solver, &model, 0.0, false, "ceiling");
    let base = solver.converged().expect("converged").x.clone();
    let n = base.len();
    let want = truth(2.0, true, &base);
    let step = solver
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let eng = outside_engine(&model, Relaxation::None);
    let a = solver
        .correct_step(&[3], &[DELTA], &step, 1)
        .expect("pounce's corrector");
    let b = correct_outside(&eng, &[3], &[DELTA], &step, 1).expect("outside corrector");
    assert_same_correction_to("ceiling, budget 1", &a, &b, Frame::Scaled, 1e-4);

    let a = solver
        .correct_step(&[3], &[DELTA], &step, 8)
        .expect("pounce's corrector");
    let b = correct_outside(&eng, &[3], &[DELTA], &step, 8).expect("outside corrector");
    assert_eq!(
        (a.1.iterations, a.1.converged, b.1.iterations, b.1.converged),
        (2, true, 8, false),
        "the iterations end where the doc above measured them",
    );
    let (da, db) = (
        dist(&add(&base, &a.0[..n]), &want),
        dist(&add(&base, &b.0[..n]), &want),
    );
    assert!(
        da > 5.0 * db,
        "the ceiling no longer costs pounce the measured factor of 7.7: pounce lands {da:e} \
         from the truth and the outside engine {db:e}",
    );
}

#[test]
fn the_same_workload_without_the_ceiling_corrects_over_an_outside_engine() {
    let (solver, model) = solved_recorded(false, false, 0.0, None);
    both_classifications(&solver, &model, 0.0, false, "no ceiling");
    both_corrections(&solver, &model, 0.0, DELTA, 8, Frame::Same, "no ceiling");
}

/// Equal to roundoff at every budget.
///
/// `x0` is held at its upper bound by two equality rows, and crossover parks
/// it there with no relaxation, so the diagonal pounce factors is the
/// declared frame's (gh#654) and the ceiling does not bind. The second
/// iteration lands `x0` exactly on the bound, since `1 - 5e-17` rounds to
/// `1.0`. From there its slack is zero, and the fraction-to-boundary rule
/// cuts the whole primal step to zero whenever `x0`'s direction is positive.
/// That direction is zero in exact arithmetic and roundoff-sized in practice,
/// so the two engines continue alike only while their solves give it the
/// same sign. With iterative refinement in the outside engine's solve they
/// do, and both run all eight iterations to the same point, `1.356e-3` from
/// the re-solved truth.
#[test]
fn a_crossover_solve_corrects_at_the_predicted_point_over_an_outside_engine() {
    let (solver, model) = solved_recorded(true, true, 0.0, None);
    both_classifications(&solver, &model, 0.0, true, "crossover");
    for budget in [1_usize, 2, 3, 8] {
        let label = format!("crossover, budget {budget}");
        both_corrections(&solver, &model, 0.0, DELTA, budget, Frame::Same, &label);
    }
}

#[test]
fn the_declared_frame_follows_the_iterate_under_a_change_of_variables_over_an_outside_engine() {
    const D: [Number; 6] = [1.0e2, 1.0e-2, 5.0, 1.0e-1, 2.0, 1.0e1];
    // Budget 2, before the iterate lands exactly on `x0`'s bound: see
    // `a_crossover_solve_corrects_at_the_predicted_point_over_an_outside_engine`.
    //
    // The outside engine works in natural units, so it gives one answer for
    // both arms. pounce's two arms differ from each other at this point, which
    // violates LICQ, by `1.3e-5` in the corrected step, and the test above
    // allows that, asking only that their errors agree within a factor of
    // ten. So the outside engine is held to pounce's unscaled arm to
    // roundoff, and to the scaled one to the size of that difference.
    let (solver_u, model_u) = solved_recorded(true, true, 0.0, None);
    both_classifications(&solver_u, &model_u, 0.0, true, "crossover, unscaled");
    both_corrections(
        &solver_u,
        &model_u,
        0.0,
        DELTA,
        2,
        Frame::Same,
        "crossover, unscaled",
    );
    let (solver_s, model_s) = solved_recorded(true, true, 0.0, Some(D));
    both_classifications(&solver_s, &model_s, 0.0, true, "crossover, scaled");
    let step = solver_s
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let pounce_scaled = solver_s
        .correct_step(&[3], &[DELTA], &step, 2)
        .expect("pounce's corrector");
    let eng = outside_engine(&model_s, relaxation(0.0));
    let outside = correct_outside(&eng, &[3], &[DELTA], &step, 2).expect("outside corrector");
    assert_same_correction_to(
        "crossover, scaled",
        &pounce_scaled,
        &outside,
        Frame::Scaled,
        1e-4,
    );
    let step_u = solver_u
        .parametric_step_full(&[3], &[DELTA])
        .expect("full step");
    let pounce_unscaled = solver_u
        .correct_step(&[3], &[DELTA], &step_u, 2)
        .expect("pounce's corrector");
    assert_same_correction(
        "crossover, the scaled report against pounce's unscaled arm",
        &pounce_unscaled,
        &outside,
        Frame::Same,
    );
}

#[test]
fn the_declared_frame_survives_a_point_outside_the_declared_box_over_an_outside_engine() {
    const RELAX: Number = 1e-8;
    const WIDE: Number = -0.9;
    // No classification: it needs `bound_relax_factor = 0`, and pounce
    // refuses it here, as the test above never asks for it.
    let (solver, model) = solved_recorded(true, true, RELAX, None);
    both_corrections(
        &solver,
        &model,
        RELAX,
        WIDE,
        8,
        Frame::Same,
        "relaxed crossover, wide",
    );
    both_corrections(
        &solver,
        &model,
        RELAX,
        DELTA,
        8,
        Frame::Same,
        "relaxed crossover",
    );
}
