//! End-to-end smoke test: solve HS014 through `IpoptApplication::optimize_tnlp`.
//!
//! HS014 (Hock & Schittkowski, 1981, problem 14):
//!
//! ```text
//! min  (x1 - 2)^2 + (x2 - 1)^2
//! s.t. x1 - 2*x2 + 1 = 0
//!      x1^2/4 + x2^2 - 1 <= 0
//! ```
//!
//! Known optimum: f* ≈ 1.3934649 at x* ≈ (0.8228756, 0.9114378).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const F_STAR: Number = 1.39346491;

#[derive(Default)]
struct Hs014;

impl TNLP for Hs014 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            // jac_g: dense 2x2.
            nnz_jac_g: 4,
            // h_lag: full lower 2x2 (3 entries) — but only the diagonal
            // is non-trivial; we still report 3 to allow off-diagonal
            // sparsity (set to zero).
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        // No primal bounds (treat ±∞).
        b.x_l.copy_from_slice(&[-2.0e19; 2]);
        b.x_u.copy_from_slice(&[2.0e19; 2]);
        // g0: linear equality x1 - 2*x2 + 1 = 0  → g_l = g_u = 0.
        // g1: nonlinear ineq x1^2/4 + x2^2 - 1 <= 0  → g_l = -∞, g_u = 0.
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = -2.0e19;
        b.g_u[1] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[2.0, 2.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 2.0);
        g[1] = 2.0 * (x[1] - 1.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 2.0 * x[1] + 1.0;
        g[1] = x[0] * x[0] / 4.0 + x[1] * x[1] - 1.0;
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
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 1.0;
                values[1] = -2.0;
                values[2] = 0.5 * x[0];
                values[3] = 2.0 * x[1];
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
        // Hessian of Lagrangian (lower triangle):
        //   H_obj = diag(2, 2); H_g0 = 0; H_g1 = diag(1/2, 2).
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = obj_factor * 2.0 + lam[1] * 0.5;
                values[1] = 0.0;
                values[2] = obj_factor * 2.0 + lam[1] * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn hs014_solves_via_application() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs014));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS14: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - F_STAR).abs() < 1e-4,
        "final_objective = {} (expected ~{F_STAR})",
        stats.final_objective,
    );
}

#[test]
fn hs014_solves_with_adaptive_mu_quality_function() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("mu_oracle", "quality-function", true, false)
        .unwrap();
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs014));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS14 adaptive+qf: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - F_STAR).abs() < 1e-4,
        "final_objective = {} (expected ~{F_STAR})",
        stats.final_objective,
    );
}

/// pounce#180 item 2: solving with a Schur KKT partition installed reaches the
/// same optimum as the standard full-space solve. HS14's augmented KKT has
/// dimension `n_x + n_s + n_c + n_d = 2 + 1 + 1 + 1 = 5` in `x, s, c, d` block
/// order; the constraint-dual block is indices `[3, 4]`. Choosing that as the
/// Schur block leaves the primal `(x, s)` block — positive definite in the
/// interior — as the eliminated `A_FF`, the classic range/null-space split.
#[test]
fn hs014_solves_with_schur_kkt_block() {
    let mut app = IpoptApplication::new();
    app.set_kkt_schur_block(vec![3, 4]);
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs014));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS14 schur[3,4]: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - F_STAR).abs() < 1e-4,
        "schur final_objective = {} (expected ~{F_STAR})",
        stats.final_objective,
    );
}

/// An oversized / unsuitable Schur block must not break the solve: the wrapper
/// falls back to the standard full-space solver transparently. Here the block
/// covers all-but-one index (well past the gate), so the Schur path is never
/// activated, yet the solve still converges to the optimum.
#[test]
fn hs014_schur_oversized_block_falls_back() {
    let mut app = IpoptApplication::new();
    app.set_kkt_schur_block(vec![0, 1, 2, 3]); // 4/5 of the KKT → gate rejects
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs014));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - F_STAR).abs() < 1e-4,
        "fallback final_objective = {} (expected ~{F_STAR})",
        stats.final_objective,
    );
}
