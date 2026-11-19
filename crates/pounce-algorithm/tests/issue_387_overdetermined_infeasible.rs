//! gh #387 — an over-determined system that is *provably* infeasible must
//! report `InfeasibleProblemDetected`, not `NotEnoughDegreesOfFreedom`.
//!
//! `x == 0.2` with `x == 0.8` over `x in [0, 1]` has an empty feasible set —
//! about as provable as infeasibility gets — yet it exited through the DOF
//! gate (1 variable < 2 equality rows) with the structural 5xx error before
//! anything could look at the constraints. The gate fires before a single
//! iteration runs, so no downstream mechanism ever had the chance to detect
//! the contradiction.
//!
//! The fix consults presolve's bound-propagation certification on the DOF
//! failure path (independent of the `presolve` master switch — nothing is
//! transformed, no solve runs through the probe). The certification's
//! fail-closed safety net is inherited wholesale:
//!
//! * a *consistent* over-determined system still reports the DOF error, and
//! * the witness-refutation rule still withdraws the proof whenever a point in
//!   the declared box genuinely satisfies every row.
//!
//! gh #391 closed the residual: the three smallest row scalings still reported
//! the DOF error, because the witness test's *clamped* accepting form
//! (`tol * max(scale, 1)`) reinstates an absolute floor once a row's magnitude
//! drops below 1, so at `s <= ~3e-8` every point of `[0, 1]` "satisfied" both
//! rows and withdrew a proof whose crossing is `0.6` at every `s`. On this path
//! the solve cannot run, so there is no `Solve_Succeeded` counterfactual for
//! the clamp to protect; the witness now measures against the row's *declared*
//! magnitude with no clamp (`pounce_presolve::WitnessRule`), and the verdict is
//! scale-invariant across the full range.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x^2  s.t.  s*a1*x == s*r1,  s*a2*x == s*r2,  x in [0, 1]`.
///
/// Every row is scaled by the same `s > 0`, which leaves the feasible set
/// exactly unchanged; the contradiction (or consistency) lives in
/// `(a1, r1, a2, r2)`.
struct TwoEqualities {
    scale: Number,
    a: [Number; 2],
    r: [Number; 2],
}

impl TwoEqualities {
    fn contradictory(scale: Number) -> Self {
        Self {
            scale,
            a: [1.0, 1.0],
            r: [0.2, 0.8],
        }
    }

    fn consistent(scale: Number) -> Self {
        Self {
            scale,
            a: [1.0, 2.0],
            r: [0.2, 0.4],
        }
    }
}

impl TNLP for TwoEqualities {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 1.0;
        for i in 0..2 {
            b.g_l[i] = self.scale * self.r[i];
            b.g_u[i] = self.scale * self.r[i];
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = self.scale * self.a[0] * x[0];
        g[1] = self.scale * self.a[1] * x[0];
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = self.scale * self.a[0];
                values[1] = self.scale * self.a[1];
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Solve with all-default options — in particular `presolve` stays at its
/// default "no", which is the configuration the issue was filed against.
fn solve(problem: TwoEqualities) -> ApplicationReturnStatus {
    solve_with(problem, false)
}

/// `solve`, optionally with `presolve=yes`. The scale sweeps run both legs:
/// with presolve off they never reach Phase 2's redundant-row removal, which
/// dropped both rows at `s = 1e-12` under an absolute tolerance and reported
/// this infeasible model solved (gh #1197).
fn solve_with(problem: TwoEqualities, presolve: bool) -> ApplicationReturnStatus {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    if presolve {
        app.options_mut()
            .set_string_value("presolve", "yes", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(problem));
    app.optimize_tnlp(tnlp)
}

#[test]
fn contradictory_equalities_are_infeasible_not_dof_error() {
    let status = solve(TwoEqualities::contradictory(1.0));
    assert_eq!(
        status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "x == 0.2 with x == 0.8 is provably infeasible; the structural DOF \
         error is the strictly weaker answer"
    );
}

/// Multiplying every row by `s > 0` leaves the feasible set unchanged, so the
/// verdict must not depend on `s` — over the *whole* range the scale-invariance
/// harness sweeps, not just the part above the old absolute floor (gh#391).
#[test]
fn verdict_is_scale_invariant_where_certifiable() {
    for presolve in [false, true] {
        for k in [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12] {
            let status = solve_with(TwoEqualities::contradictory(10.0_f64.powi(k)), presolve);
            assert_eq!(
                status,
                ApplicationReturnStatus::InfeasibleProblemDetected,
                "row scale 1e{k}, presolve={presolve}: same empty feasible set, \
                 different verdict"
            );
        }
    }
}

/// gh#391: the deliberate flip of what this test used to pin.
///
/// It formerly asserted the DOF error at `s = 1e-12`, on the reading that
/// claiming "proved infeasible" would contradict what the solver would report
/// if it could run. The premise does not hold on this path — *the solver
/// cannot run*, that is why the gate fired — so there was never a
/// `Solve_Succeeded` to contradict, only the structural error the proof
/// replaces. The witness now measures against the row's declared magnitude, and
/// the sub-tolerance scales are certified like every other.
#[test]
fn sub_tolerance_scales_are_certified_too() {
    for presolve in [false, true] {
        let status = solve_with(TwoEqualities::contradictory(1e-12), presolve);
        assert_eq!(
            status,
            ApplicationReturnStatus::InfeasibleProblemDetected,
            "presolve={presolve}: the crossing is 0.6 at every row scale; only \
             the witness's absolute floor made 1e-12 different"
        );
    }
}

/// The safety net the flip above must not have cut: a genuinely feasible model
/// at the same sub-tolerance row scale is still refuted by the witness, so the
/// DOF error stands. `x == 0.2` with `2x == 0.4` is consistent at `x = 0.2`.
#[test]
fn sub_tolerance_consistent_system_still_reports_dof_error() {
    for presolve in [false, true] {
        let status = solve_with(TwoEqualities::consistent(1e-12), presolve);
        assert!(
            !matches!(status, ApplicationReturnStatus::InfeasibleProblemDetected),
            "presolve={presolve}: `x == 0.2, 2x == 0.4` is consistent; got {status:?}"
        );
        if !presolve {
            assert_eq!(status, ApplicationReturnStatus::NotEnoughDegreesOfFreedom);
        }
    }
}

/// A consistent over-determined system (`x == 0.2`, `2x == 0.4`) is not
/// infeasible; the structural DOF error must survive the fix.
#[test]
fn consistent_equalities_still_report_dof_error() {
    let status = solve(TwoEqualities::consistent(1.0));
    assert_eq!(status, ApplicationReturnStatus::NotEnoughDegreesOfFreedom);
}
