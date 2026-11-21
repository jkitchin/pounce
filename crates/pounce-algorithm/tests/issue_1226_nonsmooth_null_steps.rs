//! gh#1226 — a nonsmooth objective must not turn the line search into
//! thousands of null steps.
//!
//! discopt hands pounce `min ‖v‖_∞` (and `max`/matrix-norm relatives) as an
//! NLP whose objective is nonsmooth at the iterate. This file's model is
//! that NLP, rebuilt from the callbacks discopt passes:
//!
//! ```text
//! min   max(|v0|, |v1|)          (gradient: the first maximizer's sign)
//! s.t.  1 - v0 - v1 <= 0,   v in [-2, 2]^2
//! ```
//!
//! with no Hessian entries. The iterate reaches the kink nearly feasible
//! (`θ ≈ 1e-15`) and the step grows each iteration (`‖d‖` to `1.8e11`).
//!
//! **Before gh#1217** the backtracking floor was `max(1e-12, acceptor's
//! CalculateAlphaMin)`: the sweep stopped at `1e-12`, the solve went to
//! restoration, and it failed fast. Those are discopt's expected outcomes:
//! it treats them as failed local solves and moves on (35, 52 and 54
//! iterations on the three starts below).
//!
//! **gh#1217 dropped the `1e-12` clamp** everywhere so that one step with
//! `‖d‖ = 3.7e17` at `θ = 4e3` could be taken at `α = 5.2e-23`. In the
//! switching region (`θ ≤ θ_min`) the acceptor's floor falls with `θ`
//! itself. Here that let every iteration backtrack 30–46 times and accept
//! an `α ≈ 1e-25` step that leaves the objective fixed to eight digits,
//! until `max_iter`. All three starts ran 3000 iterations, and discopt's
//! `max`/norm models became 5–37× slower.
//!
//! The clamp now holds in the switching region only. This file pins that
//! branch. `pounce-cli/tests/issue_1217_max_entropy.rs` pins the other
//! one: outside the region the deep floor is still reached.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

struct NormInf {
    x0: [Number; 2],
}

impl TNLP for NormInf {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-2.0);
        b.x_u.fill(2.0);
        b.g_l[0] = -1e20;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0].abs().max(x[1].abs()))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        // numpy's argmax: the first maximizer wins a tie.
        let i = if x[1].abs() > x[0].abs() { 1 } else { 0 };
        grad.fill(0.0);
        grad[i] = if x[i] < 0.0 { -1.0 } else { 1.0 };
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 1.0 - x[0] - x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[-1.0, -1.0]);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// discopt's options for these solves (`max_iter = 3000`, `tol = 1e-7`).
fn solve(x0: [Number; 2]) -> (SolveStatistics, ApplicationReturnStatus) {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_integer_value("max_iter", 3000, true, false)
        .unwrap();
    opts.set_numeric_value("tol", 1e-7, true, false).unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(NormInf { x0 }));
    let status = app.optimize_tnlp(tnlp);
    (app.statistics(), status)
}

/// The three starts discopt's branch-and-bound tries first. Each one ran
/// 3000 iterations under gh#1217's floor and runs 35–54 under this one.
/// The bound is loose on purpose: the claim is "ends without running to
/// `max_iter`", not a particular count.
#[test]
fn nonsmooth_kink_does_not_run_to_max_iter() {
    for x0 in [[0.5, 0.5], [-0.25, -0.25], [1.0, 0.3]] {
        let (stats, status) = solve(x0);
        assert_ne!(
            status,
            ApplicationReturnStatus::MaximumIterationsExceeded,
            "x0 = {x0:?}: ran to max_iter ({} iterations); the line search \
             is accepting null steps in the switching region (gh#1226)",
            stats.iteration_count
        );
        assert!(
            stats.iteration_count < 300,
            "x0 = {x0:?}: {} iterations (35–54 expected, 3000 under \
             gh#1217's floor)",
            stats.iteration_count
        );
    }
}
