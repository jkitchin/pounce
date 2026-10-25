//! Integration tests for the value-typed [`Solver`] session API.
//!
//! Confirms numerical equivalence between `Solver::solve` +
//! `Solver::parametric_step` and the existing `SensSolve` builder on
//! the same `ParametricTNLP` fixture used in
//! `tests/parametric_cpp.rs`. Also exercises the post-convergence
//! `kkt_solve` path against a hand-crafted RHS to confirm the held
//! factor is functional.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_sensitivity::{SensSolve, Solver};

mod common;
use common::outside_engine::{Relaxation, recorded_solve, same_classification};

/// Same NLP as `tests/parametric_cpp.rs::ParametricTNLP` /
/// `tests/convenience_api.rs::ParametricTNLP`. Replicated to keep
/// test binaries independent.
struct ParametricTNLP {
    nominal_eta1: Number,
    nominal_eta2: Number,
}

impl ParametricTNLP {
    fn new(eta1: Number, eta2: Number) -> Self {
        Self {
            nominal_eta1: eta1,
            nominal_eta2: eta2,
        }
    }
}

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 5,
            nnz_jac_g: 11,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.nominal_eta1;
        b.g_u[2] = self.nominal_eta1;
        b.g_l[3] = self.nominal_eta2;
        b.g_u[3] = self.nominal_eta2;
        // This duplicates x[0]'s lower bound. Generic presolve tightens the
        // bound and drops the redundant inequality, while sensitivity must
        // keep the original-space KKT dimensions.
        b.g_l[4] = 0.0;
        b.g_u[4] = 1.0e19;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.15;
        sp.x[1] = 0.15;
        sp.x[2] = 0.0;
        sp.x[3] = 0.0;
        sp.x[4] = 0.0;
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::NonLinear);
        types[4] = Linearity::Linear;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        g[3] = 0.0;
        g[4] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, eta1, eta2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - eta1;
        g[1] = eta2 * x1 + x2 - x3 - 1.0;
        g[2] = eta1;
        g[3] = eta2;
        g[4] = x1;
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
                let rs: [Index; 11] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4];
                let cs: [Index; 11] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 6.0;
                values[1] = 3.0;
                values[2] = 2.0;
                values[3] = -1.0;
                values[4] = x[4];
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0];
                values[8] = 1.0;
                values[9] = 1.0;
                values[10] = 1.0;
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
                let rs: [Index; 5] = [0, 1, 2, 4, 0];
                let cs: [Index; 5] = [0, 1, 2, 0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = lam[1];
                values[4] = 0.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn make_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

#[test]
fn solver_parametric_step_matches_sens_solve_builder() {
    // SensSolve baseline.
    let tnlp_a: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut app_a = make_app();
    let baseline = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_a, tnlp_a);
    assert!(matches!(
        baseline.status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    let dx_baseline = baseline.dx.expect("dx populated");

    // Solver session API.
    let tnlp_b: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp_b);
    let status = solver.solve();
    assert!(matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    assert!(solver.converged().is_some());

    let dx_session = solver
        .parametric_step(&[2, 3], &[-0.5, 0.0])
        .expect("parametric_step ok");
    assert_eq!(dx_session.len(), 5);

    for k in 0..5 {
        let err = (dx_session[k] - dx_baseline[k]).abs();
        assert!(
            err < 1e-10,
            "dx[{k}]: solver={}, sens_solve={}, |err|={err} not < 1e-10",
            dx_session[k],
            dx_baseline[k],
        );
    }

    // Independent operations against the same factor work too.
    let dx2 = solver
        .parametric_step(&[2, 3], &[0.0, 0.25])
        .expect("second parametric_step ok");
    assert_eq!(dx2.len(), 5);
}

#[test]
fn solver_keeps_sensitivity_kkt_in_original_space_when_presolve_is_enabled() {
    let mut app = make_app();
    app.options_mut()
        .set_string_value("presolve", "yes", true, false)
        .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(app, tnlp);

    let status = solver.solve();
    assert!(matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    let converged = solver.converged().expect("converged state captured");
    assert_eq!(
        converged.block_dims(),
        [5, 1, 4, 1, 3, 0, 1, 0],
        "sensitivity must retain the original five-row TNLP KKT layout"
    );
    drop(converged);
    assert!(
        solver
            .app()
            .options()
            .get_bool_value("presolve", "")
            .unwrap()
            .0,
        "the sensitivity guard must not overwrite the user's option"
    );
    assert!(solver.parametric_step(&[2, 3], &[-0.5, 0.0]).is_ok());
}

#[test]
fn solver_kkt_solve_against_zero_rhs_returns_zero() {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp);
    let status = solver.solve();
    assert!(matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    let dim = solver
        .kkt_dim()
        .expect("kkt_dim available post-convergence");

    let rhs = vec![0.0; dim];
    let mut lhs = vec![1.0; dim]; // pre-fill with garbage to confirm overwrite
    solver.kkt_solve(&rhs, &mut lhs).expect("kkt_solve ok");
    for (k, v) in lhs.iter().enumerate() {
        assert!(
            v.abs() < 1e-12,
            "K·0 = 0 expected, lhs[{k}] = {v} not near zero",
        );
    }
}

#[test]
fn solver_kkt_solve_shape_mismatch_errors() {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp);
    solver.solve();
    let dim = solver.kkt_dim().expect("converged");

    let rhs = vec![0.0; dim + 1];
    let mut lhs = vec![0.0; dim];
    let err = solver
        .kkt_solve(&rhs, &mut lhs)
        .expect_err("wrong rhs length must error");
    matches!(err, pounce_sensitivity::SolverError::BadShape { .. });
}

#[test]
fn solver_reduced_hessian_matches_sens_solve_builder() {
    let tnlp_a: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut app_a = make_app();
    let baseline = SensSolve::new(vec![2, 3])
        .with_reduced_hessian()
        .run(&mut app_a, tnlp_a);
    let hr_baseline = baseline.reduced_hessian.expect("reduced Hessian populated");

    let tnlp_b: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp_b);
    solver.solve();
    let hr_session = solver
        .compute_reduced_hessian(&[2, 3], 1.0)
        .expect("reduced_hessian ok");
    assert_eq!(hr_session.len(), 4);
    for k in 0..4 {
        let err = (hr_session[k] - hr_baseline[k]).abs();
        assert!(
            err < 1e-10,
            "Hr[{k}]: solver={}, sens_solve={}, |err|={err}",
            hr_session[k],
            hr_baseline[k],
        );
    }
}

/// The session API's eigendecomposition must be the one-shot builder's.
///
/// Before this the spectrum was reachable only through
/// `SensSolve::with_reduced_hessian_eigen`, so a caller holding a `Solver` —
/// which already has the factor and the reduced Hessian — had to **re-solve
/// the whole NLP** through the one-shot path to decompose a matrix it was
/// already holding. That is the parity gap; this asserts closing it did not
/// invent a second answer.
#[test]
fn solver_reduced_hessian_eigen_matches_sens_solve_builder() {
    let tnlp_a: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut app_a = make_app();
    let baseline = SensSolve::new(vec![2, 3])
        .with_reduced_hessian_eigen()
        .run(&mut app_a, tnlp_a);
    let vals_baseline = baseline
        .reduced_hessian_eigenvalues
        .expect("eigenvalues populated by the one-shot builder");
    let vecs_baseline = baseline
        .reduced_hessian_eigenvectors
        .expect("eigenvectors populated by the one-shot builder");

    let tnlp_b: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp_b);
    solver.solve();
    let (hr, vals, vecs) = solver
        .compute_reduced_hessian_eigen(&[2, 3], 1.0)
        .expect("session eigendecomposition ok");

    assert_eq!(hr.len(), 4);
    assert_eq!(vals.len(), 2);
    assert_eq!(vecs.len(), 4);
    // The matrix itself must still be `compute_reduced_hessian`'s, so the new
    // entry point is the same computation plus a decomposition rather than a
    // second route to the number.
    let hr_plain = solver
        .compute_reduced_hessian(&[2, 3], 1.0)
        .expect("reduced_hessian ok");
    assert_eq!(hr, hr_plain);

    for k in 0..2 {
        assert!(
            (vals[k] - vals_baseline[k]).abs() < 1e-10,
            "eigenvalue {k}: session={}, sens_solve={}",
            vals[k],
            vals_baseline[k]
        );
    }
    // Eigenvectors are sign-pinned by `symmetric_eigen`, so they must agree
    // component-wise rather than only up to a sign — that pinning is what
    // makes a column readable as a direction across builds.
    for k in 0..4 {
        assert!(
            (vecs[k] - vecs_baseline[k]).abs() < 1e-10,
            "eigenvector component {k}: session={}, sens_solve={}",
            vecs[k],
            vecs_baseline[k]
        );
    }
}

#[test]
fn solver_converged_none_before_solve() {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let solver = Solver::new(make_app(), tnlp);
    assert!(solver.converged().is_none());
    assert!(solver.kkt_dim().is_none());
}

/// `classify_activity` guards the `bound_relax_factor` the held solve
/// ran under, not the application's live options. Both directions
/// matter: the bounds were relaxed (or not) once, during the solve,
/// and no later `set_numeric_value` re-measures the slacks the
/// classifier reads.
#[test]
fn classify_activity_guards_the_solves_bound_relax_factor() {
    let mut app = make_app();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(app, tnlp);
    solver.solve();
    assert_eq!(
        solver
            .converged()
            .expect("converged state captured")
            .bound_relax_factor,
        0.0,
    );
    assert!(solver.classify_activity().is_ok());

    // Raising the option cannot invalidate a report the held state is
    // still perfectly good for: those bounds were never relaxed.
    solver
        .app_mut()
        .options_mut()
        .set_numeric_value("bound_relax_factor", 1e-8, true, false)
        .unwrap();
    assert!(
        solver.classify_activity().is_ok(),
        "the guard must read the solve's value, not the live option",
    );
}

#[test]
fn classify_activity_refuses_a_solve_that_relaxed_its_bounds() {
    // make_app leaves bound_relax_factor at the 1e-8 default.
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let mut solver = Solver::new(make_app(), tnlp);
    solver.solve();
    assert!(matches!(
        solver.classify_activity(),
        Err(pounce_sensitivity::SolverError::BadOptions(_)),
    ));

    // The option name is load-bearing in the message, not decoration:
    // pyomo-pounce's `estimate_report` tells this refusal apart from
    // every other BadOptions by matching that token, and reports the
    // solve as having relaxed its bounds instead of raising. Rewording
    // the message without updating that check turns the branch back
    // into a re-raise, silently.
    let msg = match solver.classify_activity() {
        Err(pounce_sensitivity::SolverError::BadOptions(m)) => m,
        Err(other) => panic!("expected BadOptions, got {other:?}"),
        Ok(_) => panic!("expected BadOptions, got a report"),
    };
    assert!(
        msg.contains("bound_relax_factor"),
        "the message must name the option: pyomo-pounce matches it to \
         report a relaxed solve rather than raise (got {msg:?})",
    );

    // Clearing the option afterwards must NOT unlock the stale state:
    // its slacks were measured against relaxed bounds and still are.
    solver
        .app_mut()
        .options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    assert!(
        matches!(
            solver.classify_activity(),
            Err(pounce_sensitivity::SolverError::BadOptions(_)),
        ),
        "clearing the option must not retroactively un-relax the held slacks",
    );
}

// ---------------------------------------------------------------------------
// The same classification over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and asserts that the classification over an engine built from
// that report is the same as pounce's.

/// The unrelaxed solve the guard accepts. Its companion,
/// `classify_activity_refuses_a_solve_that_relaxed_its_bounds`, has no
/// counterpart here: the refusal is `Solver`'s, made from the option the
/// solve ran with, and an engine outside pounce has no `Solver` to make it.
/// The precondition is stated on `pounce_sens_core::activity::compute` for
/// that engine to check.
#[test]
fn classify_activity_guards_the_solves_bound_relax_factor_over_an_outside_engine() {
    let mut app = make_app();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    let (solver, model, _) = recorded_solve(app, ParametricTNLP::new(5.0, 1.0));
    same_classification("unrelaxed", &solver, &model, Relaxation::None);
}
