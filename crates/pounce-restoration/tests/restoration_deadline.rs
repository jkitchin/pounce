//! Restoration wall-time-deadline guard (pounce#246, complementary check).
//!
//! A restoration-heavy solve must honor the shared wall-time `Deadline`
//! promptly at the initialization / restoration-entry boundary — the window
//! #242/#244/#245 left un-checked (they bounded the outer-iteration and
//! between-KKT-factorization boundaries). This guards the two small deadline
//! checks added alongside the #246 fix: the pre-loop init check in
//! `IpoptAlgorithm::optimize_inner` and the per-inner-iteration check in
//! `RestoConvCheckAdapter`. (The primary #246 fix — the dual-divergence
//! guard that prevents the emfl050 warm-start factorization stall — is
//! exercised separately.)
//!
//! The fixture is per-variable infeasible (`x_i = 1` AND `x_i >= 3`), so the
//! outer IPM cannot make progress and repeatedly enters the nested
//! restoration IPM. With a generous budget the solve runs restoration to
//! completion (hundreds of inner iterations); with a tight budget it must be
//! cut short — far fewer inner iterations and a `MaximumWallTimeExceeded`
//! status.

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
    InnerBackendFactoryFactory, make_default_restoration_factory,
};
use std::cell::RefCell;
use std::rc::Rc;

/// `n` variables `x_i` in `[0, 5]`, start `x_i = 1.5`. For each `i` two
/// contradictory constraints (a vectorised `InfeasibleScalar`):
///   `g_{2i}   = x_i = 1`   (equality)
///   `g_{2i+1} = x_i >= 3`  (inequality)
/// infeasible per variable, so the outer IPM must enter restoration.
struct InfeasibleVec {
    n: usize,
}

impl TNLP for InfeasibleVec {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n;
        Some(NlpInfo {
            n: n as i32,
            m: (2 * n) as i32,
            nnz_jac_g: (2 * n) as i32,
            nnz_h_lag: n as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..self.n {
            b.x_l[i] = 0.0;
            b.x_u[i] = 5.0;
            b.g_l[2 * i] = 1.0;
            b.g_u[2 * i] = 1.0;
            b.g_l[2 * i + 1] = 3.0;
            b.g_u[2 * i + 1] = 2.0e19;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for i in 0..self.n {
            sp.x[i] = 1.5;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x.iter().map(|v| v * v).sum())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.n {
            g[i] = 2.0 * x[i];
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.n {
            g[2 * i] = x[i];
            g[2 * i + 1] = x[i];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..n {
                    irow[2 * i] = (2 * i) as i32;
                    jcol[2 * i] = i as i32;
                    irow[2 * i + 1] = (2 * i + 1) as i32;
                    jcol[2 * i + 1] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut().take(2 * n) {
                    *v = 1.0;
                }
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
                for i in 0..self.n {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut().take(self.n) {
                    *v = 2.0 * obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn feral_backend() -> LinearBackendFactory {
    Box::new(
        |_choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(pounce_feral::FeralSolverInterface::new())
        },
    )
}

/// Run the fixture with `max_wall_time = budget`; return
/// `(status, outer_iters, restoration_inner_iters)`.
fn solve_with_budget(budget: f64, n: usize) -> (ApplicationReturnStatus, i32, i32) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("max_wall_time", budget, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let bff: InnerBackendFactoryFactory = Box::new(feral_backend);
    let factory = make_default_restoration_factory(
        RestoAlgorithmBuilder::new(),
        AlgorithmBuilder::new(),
        bff,
    );
    app.set_restoration_factory(factory);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(InfeasibleVec { n }));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    (status, stats.iteration_count, stats.restoration_inner_iters)
}

#[test]
fn restoration_grind_honors_wall_deadline() {
    let n = 40;

    // Baseline: a generous budget lets restoration run to completion.
    let (base_status, _base_outer, base_inner) = solve_with_budget(60.0, n);
    // Sanity: the fixture is genuinely restoration-heavy (many inner iters)
    // and does not spuriously "succeed" on an infeasible problem.
    assert!(
        base_inner > 50,
        "fixture is not restoration-heavy enough to be a meaningful guard \
         (inner iters = {base_inner}); adjust `n`",
    );
    assert!(
        !matches!(
            base_status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
                | ApplicationReturnStatus::FeasiblePointFound
        ),
        "fixture claimed success on an infeasible NLP: {base_status:?}",
    );

    // Tight budget: the solve must stop on the wall-time limit having done
    // dramatically fewer restoration inner iterations. Comparing the inner
    // iteration count (rather than raw wall time) keeps the assertion robust
    // across machine speeds: the budget is short enough that no realistic
    // machine reaches even a third of the unbounded inner-iteration count.
    let (tight_status, _tight_outer, tight_inner) = solve_with_budget(0.05, n);
    assert!(
        matches!(
            tight_status,
            ApplicationReturnStatus::MaximumWallTimeExceeded
        ),
        "tight-budget solve did not terminate on the wall-time limit: {tight_status:?}",
    );
    assert!(
        tight_inner * 3 < base_inner,
        "restoration was not cut short by the deadline: tight inner iters \
         = {tight_inner}, baseline = {base_inner} (#246)",
    );
}
