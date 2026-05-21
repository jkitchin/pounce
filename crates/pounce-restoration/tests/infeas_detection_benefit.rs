//! Benefit check for rapid infeasibility detection (#5A).
//!
//! Runs genuinely infeasible NLPs through `IpoptApplication::optimize_tnlp`
//! with detection disabled vs enabled and compares the outcome. Detection
//! recognises an iterate converging to a stationary point of the
//! constraint violation (`‖Jᵀc‖/max(1,‖c‖)` small) with the violation
//! bounded away from zero, and exits early with `LocalInfeasibility`
//! instead of grinding to `max_iter` or thrashing restoration.

use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Smooth, genuinely infeasible NLP with no real solution:
///
/// ```text
/// min  1/2 (x1^2 + x2^2)
/// s.t. x1^2 + x2^2 = -1
/// ```
///
/// The constraint `x1^2 + x2^2 + 1 = 0` has no real root. The closest
/// point — the minimiser of `1/2‖c‖^2` — is the origin, where the
/// violation `θ = 1` and the infeasibility gradient `Jᵀc = 2x·(‖x‖^2+1)`
/// vanishes. Both `f` and `θ` pull the iterate to the origin, so the
/// line search keeps accepting steps: the IPM marches straight into an
/// infeasible stationary point. This is exactly the regime #5A targets.
struct InfeasibleCircle;

impl TNLP for InfeasibleCircle {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; 2]);
        b.x_u.copy_from_slice(&[2.0e19; 2]);
        // g(x) = x1^2 + x2^2, pinned to -1 → infeasible.
        b.g_l[0] = -1.0;
        b.g_u[0] = -1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.5, -1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(0.5 * (x[0] * x[0] + x[1] * x[1]))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                // H = obj_factor·I + lambda0·diag(2,2).
                values[0] = obj_factor + 2.0 * lam[0];
                values[1] = obj_factor + 2.0 * lam[0];
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Backend factory matching the application-side default: every
/// `LinearSolverChoice` returns a fresh FERAL instance.
fn feral_backend() -> LinearBackendFactory {
    Box::new(
        |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(pounce_feral::FeralSolverInterface::new())
        },
    )
}

/// Solve `InfeasibleCircle` with the given detection knobs. `max_streak`
/// of 0 disables detection entirely. Returns `(status, iteration_count)`.
fn solve(max_streak: i32, stationarity_tol: f64) -> (ApplicationReturnStatus, usize) {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_integer_value("max_iter", 3000, true, false).unwrap();
        o.set_integer_value("infeas_max_streak", max_streak, true, false)
            .unwrap();
        if stationarity_tol > 0.0 {
            o.set_numeric_value("infeas_stationarity_tol", stationarity_tol, true, false)
                .unwrap();
        }
    }
    app.initialize().unwrap();

    let bff: InnerBackendFactoryFactory = Box::new(feral_backend);
    let factory = make_default_restoration_factory(
        RestoAlgorithmBuilder::new(),
        AlgorithmBuilder::new(),
        bff,
    );
    app.set_restoration_factory(factory);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(InfeasibleCircle));
    let status = app.optimize_tnlp(tnlp);
    (status, app.statistics().iteration_count as usize)
}

#[test]
fn rapid_infeasibility_detection_benefit_check() {
    // Baseline: detection disabled. The main IPM dives to the infeasible
    // stationary point, then hands off to the restoration phase.
    let (off_status, baseline_iters) = solve(0, 1e-8);

    // Sweep (streak, stationarity_tol). A run that exits in *fewer*
    // iterations than the baseline has terminated via #5A before the
    // restoration hand-off.
    let configs: &[(i32, f64, &str)] = &[
        (5, 1e-8, "shipped default"),
        (5, 1e-6, ""),
        (5, 1e-4, ""),
        (5, 1e-2, ""),
        (3, 1e-2, ""),
        (1, 1e-2, ""),
        (1, 1e-4, ""),
    ];

    eprintln!("=== #5A benefit check: InfeasibleCircle ===");
    eprintln!("  detection OFF                       : {off_status:?}  iters={baseline_iters}");
    let mut fired_early = false;
    for &(streak, tol, note) in configs {
        let (status, iters) = solve(streak, tol);
        let tag = if iters < baseline_iters {
            fired_early = true;
            "  <-- #5A fired early"
        } else {
            ""
        };
        eprintln!("  streak={streak} tol={tol:>7.0e} {note:15} : {status:?}  iters={iters}{tag}",);
        // Detection must never turn an infeasible problem into a
        // claimed success, at any tuning.
        assert!(
            !matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
                    | ApplicationReturnStatus::FeasiblePointFound
            ),
            "streak={streak} tol={tol:e} claimed success on an infeasible NLP: {status:?}",
        );
    }

    assert!(
        !matches!(
            off_status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
                | ApplicationReturnStatus::FeasiblePointFound
        ),
        "baseline claimed success on an infeasible NLP: {off_status:?}",
    );
    // With loose-enough knobs the gate must be reachable end-to-end:
    // #5A terminates the solve before the restoration hand-off.
    assert!(
        fired_early,
        "#5A never fired early at any tuning -- gate may be unreachable",
    );
}
