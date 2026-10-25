//! A square NLP is reported solved at a barrier parameter it never
//! reached, and the sensitivity layer must read the barrier level off
//! the point rather than off the driver.
//!
//! `IsSquareProblem` (`IpIpoptCalculatedQuantities.cpp:3732`) is a
//! purely dimensional test, `dim(x) == dim(y_c)`: no degrees of
//! freedom, so the objective is decorative and the answer is whatever
//! point satisfies the equalities. Ipopt exploits that in
//! `ComputeFeasibilityMultipliers` (`IpIpoptAlg.cpp:893`, ported in
//! gh#508) — it zeroes all four bound-multiplier blocks, solves for the
//! feasibility multipliers, and converges the check outright.
//!
//! The consequence for anything reading `IpoptData::curr_mu`: such a
//! problem terminates on iteration 1 with `mu` still at `mu_init` and
//! every complementarity product identically zero. `curr_mu` describes
//! the driver, not the iterate, and the two have parted company.
//!
//! Taking `curr_mu` at face value is not cosmetic. The equation-11
//! barrier correction adds `mu` to the complementarity rows to carry
//! the step from the barrier problem's solution toward the original
//! problem's; on a point that is already *at* the original problem's
//! solution that term is pure error, and it lands on the returned dual
//! step with the wrong sign.
//!
//! Fixture (square: n = m = 2, plus a bound so there are bound rows to
//! zero):
//!
//! ```text
//!   min (x0 - 5)² + (x1 - 5)²
//!   s.t.  g0:  x0 = p0
//!         g1:  x1 = p1
//!         0 <= x0 <= 10
//! ```
//!
//! Fully determined: `x = [p0, p1]`, both bounds inactive. Perturbing
//! `g0`'s right-hand side by Δ moves `x0` by Δ and nothing else, and
//! the multipliers move by exactly `-2Δ` — every quantity here is
//! available in closed form, so the assertions are on the answer, not
//! on a tolerance band around a previous run.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

mod common;
use common::compare::{Frame, assert_same_correction};
use common::outside_engine::{RecordingTnlp, Relaxation, correct_outside, outside_engine};

struct SquareTNLP {
    p0: Number,
    p1: Number,
}

impl TNLP for SquareTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // x0 is bounded on both sides — inactive at the solution, but it
        // is what puts rows in the z_l / z_u blocks so the zeroing has
        // something to zero.
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.g_l[0] = self.p0;
        b.g_u[0] = self.p0;
        b.g_l[1] = self.p1;
        b.g_u[1] = self.p1;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.p0;
        sp.x[1] = self.p1;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 5.0) * (x[0] - 5.0) + (x[1] - 5.0) * (x[1] - 5.0))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 5.0);
        g[1] = 2.0 * (x[1] - 5.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[1];
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
                let rs: [Index; 2] = [0, 1];
                let cs: [Index; 2] = [0, 1];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved() -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(SquareTNLP { p0: 1.0, p1: 2.0 }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}"
    );
    solver
}

#[test]
fn square_problem_step_carries_no_barrier_term() {
    let solver = solved();
    let dims = solver.converged().unwrap().block_dims();
    let (n_x, n_s, n_yc, n_yd) = (dims[0], dims[1], dims[2], dims[3]);
    assert_eq!(
        n_x, n_yc,
        "fixture must be square for this to test anything"
    );

    let delta = 0.25;
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("parametric_step_full ok");

    // dx = [Δ, 0]: g0 fixes x0, g1 fixes x1, nothing couples them.
    assert!(
        (step[0] - delta).abs() < 1e-9 && step[1].abs() < 1e-9,
        "primal step {:?}, expected [{delta}, 0]",
        &step[..n_x],
    );

    // dy = [-2Δ, 0]: stationarity is ∇f + Jᵀy = 0 with J = I, so
    // y = -∇f = [-2(x0-5), -2(x1-5)] and moving x0 by Δ moves y0 by -2Δ.
    // Not only the bound rows are wrong when the barrier level is read
    // off `curr_mu`: the spurious complementarity right-hand side feeds
    // back through the solve and lands here too, as -0.5889 against an
    // exact -0.5.
    let yc = &step[n_x + n_s..n_x + n_s + n_yc];
    assert!(
        (yc[0] + 2.0 * delta).abs() < 1e-9 && yc[1].abs() < 1e-9,
        "equality-multiplier step {yc:?}, expected [{}, 0]",
        -2.0 * delta,
    );

    // The point that matters. Both bounds are inactive and the reported
    // point carries no bound multipliers at all, so no barrier term is
    // owed here. Reading the barrier level off `curr_mu` instead —
    // which the square path leaves at `mu_init`, three iterations'
    // worth of reduction above where the point sits — puts an O(mu)
    // entry in each of these rows, with the sign of the correction
    // rather than of the answer.
    let duals = &step[n_x + n_s + n_yc + n_yd..];
    assert!(
        !duals.is_empty(),
        "fixture must have bound rows for this to test anything",
    );
    for (k, &v) in duals.iter().enumerate() {
        assert!(
            v.abs() < 1e-9,
            "bound-multiplier step row {k} = {v}, expected 0; \
             full step {step:?}",
        );
    }
}

#[test]
fn square_problem_exact_step_starts_at_the_barrier_floor() {
    let solver = solved();
    let delta = 0.25;
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("parametric_step_full ok");
    let (out, report) = solver
        .correct_step(&[0], &[delta], &step, 8)
        .expect("correct_step ok");

    assert!(
        (out[0] - delta).abs() < 1e-7 && out[1].abs() < 1e-7,
        "corrected to {:?}, expected [{delta}, 0]; {report:?}",
        &out[..2],
    );
    // The step is exact — the problem is linear in the pinned
    // right-hand side — so there is nothing for the corrector to do.
    // It measures the complementarity rows against the barrier level,
    // and against `mu_init` a point with zero bound multipliers reads
    // as a full `mu` off the floor no matter how exact the step was.
    assert!(
        report.initial_residual < 1e-6,
        "an exact step should start at the barrier floor: {report:?}",
    );
}

// ---------------------------------------------------------------------------
// The same corrections over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and hands the same step to the corrector over an engine built
// from that report. These solves run at pounce's default `bound_relax_factor`,
// so the outside engine measures its slacks against bounds relaxed the way
// pounce relaxes them.

#[test]
fn square_problem_exact_step_starts_at_the_barrier_floor_over_an_outside_engine() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    let model = Rc::new(RefCell::new(RecordingTnlp::new(SquareTNLP {
        p0: 1.0,
        p1: 2.0,
    })));
    let tnlp: Rc<RefCell<dyn TNLP>> = model.clone();
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}"
    );
    let delta = 0.25;
    let step = solver
        .parametric_step_full(&[0], &[delta])
        .expect("parametric_step_full ok");
    let pounce = solver
        .correct_step(&[0], &[delta], &step, 8)
        .expect("pounce's corrector");
    let eng = outside_engine(&model, Relaxation::POUNCE_DEFAULT);
    assert_eq!(
        eng.mu(),
        0.0,
        "every bound multiplier of this report is zero, so the point sits on no barrier path",
    );
    let outside = correct_outside(&eng, &[0], &[delta], &step, 8).expect("outside corrector");
    assert_same_correction("square problem", &pounce, &outside, Frame::Same);
}
