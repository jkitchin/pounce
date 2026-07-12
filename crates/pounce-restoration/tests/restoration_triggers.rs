//! End-to-end wiring smoke test for Phase 9 restoration plumbing.
//!
//! Verifies that
//! [`pounce_algorithm::application::IpoptApplication::set_restoration_factory`]
//! routes a line-search failure into the user-supplied `RestorationPhase`.
//!
//! The fixture problem is structurally infeasible (two contradictory
//! equality constraints), so the outer IPM cannot make progress and
//! must call into the restoration hook. We swap the inner-solver hook
//! for a counter that always returns `None` (forcing
//! [`pounce_restoration::min_c_1nrm::MinC1NormRestoration`] to surface
//! [`pounce_algorithm::restoration::RestorationOutcome::Failed`]) and
//! assert the count is positive after the solve, plus the application
//! status is one of the failure-mode variants.

use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::{IpoptApplication, RestorationFactory};
use pounce_algorithm::restoration::RestorationPhase;
use pounce_common::types::Number;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::min_c_1nrm::{MinC1NormRestoration, RestoInnerSolver};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Trivially infeasible NLP (one equality + one inequality):
///   min  x^2
///   s.t. x = 1,        (equality)
///        x >= 3,       (inequality, contradicts the equality)
///        0 <= x <= 5.
struct InfeasibleScalar;

impl TNLP for InfeasibleScalar {
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
        b.x_u[0] = 5.0;
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        b.g_l[1] = 3.0;
        b.g_u[1] = 2.0e19;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 1.5;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        g[1] = x[0];
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
                values[0] = 1.0;
                values[1] = 1.0;
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
                irow.copy_from_slice(&[0]);
                jcol.copy_from_slice(&[0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn line_search_failure_invokes_user_supplied_restoration_phase() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    // A counter that the inner-solver hook bumps every time it's
    // invoked. `Rc<RefCell<u32>>` lets us read it after the solve.
    let invocation_count = Rc::new(RefCell::new(0u32));
    let count_for_factory = Rc::clone(&invocation_count);

    let factory: RestorationFactory = Box::new(move || {
        let count = Rc::clone(&count_for_factory);
        let hook: RestoInnerSolver = Box::new(move |_, _, _, _, _, _| {
            *count.borrow_mut() += 1;
            // Returning `None` makes `MinC1NormRestoration` surface
            // `RestorationOutcome::Failed`, so the outer algorithm
            // terminates with `RestorationFailure`. That's what we
            // want for a wiring smoke test.
            None
        });
        let driver = MinC1NormRestoration::new().with_inner_solver(hook);
        Box::new(driver) as Box<dyn RestorationPhase>
    });
    app.set_restoration_factory(factory);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(InfeasibleScalar));
    let status = app.optimize_tnlp(tnlp);

    let calls = *invocation_count.borrow();
    eprintln!("status = {status:?}, restoration inner-solver calls = {calls}");
    assert!(
        calls >= 1,
        "expected the line-search → restoration path to invoke the inner-solver hook at least once, got {calls} calls",
    );

    // With the inner solver returning `None`, the outermost driver
    // must surface a failure-mode status. We accept any of the
    // restoration / infeasibility / step-error variants since the
    // exact SolverReturn depends on which check fires first.
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::RestorationFailed
                | ApplicationReturnStatus::InfeasibleProblemDetected
                | ApplicationReturnStatus::ErrorInStepComputation
                | ApplicationReturnStatus::InternalError
                | ApplicationReturnStatus::MaximumIterationsExceeded
                | ApplicationReturnStatus::SearchDirectionBecomesTooSmall
        ),
        "unexpected status on infeasible problem: {status:?}",
    );
}

/// Backend factory matching the application-side default: every
/// `LinearSolverChoice` returns a fresh FERAL instance.
fn ma57_backend() -> LinearBackendFactory {
    Box::new(
        |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(pounce_feral::FeralSolverInterface::new())
        },
    )
}

/// End-to-end smoke test for the nested IPM produced by
/// [`pounce_restoration::resto_inner_solver::make_default_restoration_factory`].
/// The resto Hessian is emitted in flat `SymTMatrix` form (orig
/// triplets + n_orig diagonal entries for the proximity term), so
/// `StdAugSystemSolver` consumes it directly — bit-equivalence with
/// upstream's `CompoundSymMatrix(SumSymMatrix(orig_h, η·DR²))` shape
/// is a Phase-10 concern.
#[test]
fn make_default_restoration_factory_drives_nested_ipm_without_panicking() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    let resto_builder = RestoAlgorithmBuilder::new();
    let inner_alg_builder = AlgorithmBuilder::new();
    let bff: InnerBackendFactoryFactory = Box::new(ma57_backend);
    let factory = make_default_restoration_factory(resto_builder, inner_alg_builder, bff);
    app.set_restoration_factory(factory);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(InfeasibleScalar));
    let status = app.optimize_tnlp(tnlp);

    eprintln!("default-factory nested-IPM status = {status:?}");
    // The full end-to-end resto trace is a Phase 10 deliverable; for
    // now we only require that wiring through the real factory
    // terminates with *some* status (i.e. doesn't panic in the inner
    // IPM) and doesn't claim success on an infeasible problem.
    assert!(
        !matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
                | ApplicationReturnStatus::FeasiblePointFound
        ),
        "default factory claimed success on an infeasible NLP: {status:?}",
    );
}
