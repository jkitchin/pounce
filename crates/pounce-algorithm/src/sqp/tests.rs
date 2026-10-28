//! Scaffolding-level tests for the SQP module. Phase 5b commit 1
//! only verifies that the types compile, the defaults are sane,
//! and `AlgorithmChoice::ActiveSetSqp` is wired into the builder
//! enum. End-to-end optimize tests land in later commits.

use crate::alg_builder::AlgorithmChoice;
use crate::sqp::ipopt_adapter::IpoptNlpAdapter;
use crate::sqp::iterates::SqpIterates;
use crate::sqp::options::{SqpGlobalization, SqpHessianSource, SqpOptions};
use crate::sqp::problem::SqpProblemSpec;
use crate::sqp::qp_assembly::{SqpQpData, Triplet};
use crate::sqp::result::{SqpError, SqpStatus};
use crate::sqp::sqp_alg::SqpAlgorithm;
use pounce_common::types::{Index, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF, Number};
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use pounce_linalg::expansion_matrix::{ExpansionMatrix, ExpansionMatrixSpace};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
use pounce_linalg::{Matrix, SymMatrix, Vector};
use pounce_qp::{HessianInertia, ParametricActiveSetSolver, QpOptions, QpSolver, QpStatus};
use std::cell::RefCell;
use std::rc::Rc;

#[test]
fn algorithm_choice_default_is_interior_point() {
    assert_eq!(AlgorithmChoice::default(), AlgorithmChoice::InteriorPoint);
}

#[test]
fn builder_default_returns_none_for_sqp_dispatch() {
    use crate::alg_builder::AlgorithmBuilder;
    use pounce_linsol::SparseSymLinearSolverInterface;

    let builder = AlgorithmBuilder::default();
    let factory: crate::alg_builder::LinearBackendFactory =
        Box::new(|_choice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(pounce_feral::FeralSolverInterface::new())
        });
    // Default algorithm is InteriorPoint ⇒ SQP build returns None.
    assert!(builder.build_sqp_with_backend(factory).is_none());
}

#[test]
fn builder_active_set_sqp_constructs_sqp_algorithm() {
    use crate::alg_builder::{AlgorithmBuilder, AlgorithmChoice as AC};
    use pounce_linsol::SparseSymLinearSolverInterface;

    let mut builder = AlgorithmBuilder::default();
    builder.algorithm = AC::ActiveSetSqp;
    let factory: crate::alg_builder::LinearBackendFactory =
        Box::new(|_choice| -> Box<dyn SparseSymLinearSolverInterface> {
            Box::new(pounce_feral::FeralSolverInterface::new())
        });
    let alg = builder.build_sqp_with_backend(factory);
    assert!(alg.is_some());
    // Spot-check: the constructed SqpAlgorithm uses the builder's
    // SqpOptions (defaults).
    let alg = alg.unwrap();
    assert_eq!(alg.options().max_iter, 200);
}

#[test]
fn sqp_options_default_matches_design_note() {
    let opts = SqpOptions::default();
    assert_eq!(opts.globalization, SqpGlobalization::Filter);
    assert_eq!(opts.hessian, SqpHessianSource::Exact);
    assert_eq!(opts.max_iter, 200);
    assert!((opts.tol - 1e-8).abs() < f64::EPSILON);
    assert!((opts.l1_penalty - 1.0).abs() < f64::EPSILON);
}

#[test]
fn sqp_iterates_cold_has_expected_lengths_and_no_working_set() {
    let it = SqpIterates::cold(5, 3);
    assert_eq!(it.n(), 5);
    assert_eq!(it.m(), 3);
    assert_eq!(it.x, vec![0.0; 5]);
    assert_eq!(it.lambda_g, vec![0.0; 3]);
    assert_eq!(it.lambda_x, vec![0.0; 5]);
    assert!(it.working.is_none());
}

// ─────────────────────────────────────────────────────────────────
// QP-from-linearization end-to-end on a closed-form NLP:
//
//     min f(x) = ½(x₁² + x₂²) − x₁ − 2x₂
//     s.t. c(x) = x₁ + x₂ − 1 = 0,  no bounds.
//
// This is a *convex quadratic* NLP, so its SQP linearization at
// any iterate is also the original problem (∇²L = I, ∇f = x − g,
// J_c = (1, 1), c = x₁+x₂ − 1). One SQP iteration from x_0 =
// (0, 0) should produce a step p that lands at the true optimum.
//
// True optimum: minimize ½(x₁²+x₂²) − x₁ − 2x₂ s.t. x₁+x₂ = 1.
// Lagrangian: x₁ − 1 + λ = 0, x₂ − 2 + λ = 0, x₁+x₂ = 1.
// ⇒ x₁ = 1 − λ, x₂ = 2 − λ, sum: 3 − 2λ = 1 ⇒ λ = 1.
// ⇒ x* = (0, 1), λ_g = 1. From x_0 = (0, 0): p = (0, 1).
// ─────────────────────────────────────────────────────────────────
#[test]
fn qp_assembly_one_sqp_iter_solves_convex_eq_nlp() {
    let n = 2;
    let m = 1;
    let x = vec![0.0; n];

    // ∇f(x) = x − (1, 2)
    let grad_f = vec![x[0] - 1.0, x[1] - 2.0];

    // c(x) = x₁ + x₂ − 1 = −1 at x_0
    let c_vals = vec![x[0] + x[1] - 1.0];
    let bl_c = vec![0.0];
    let bu_c = vec![0.0];

    // No variable bounds.
    let xl_orig = vec![NLP_LOWER_BOUND_INF; n];
    let xu_orig = vec![NLP_UPPER_BOUND_INF; n];

    // J_c = [1, 1] — one row, two columns.
    let jac_c = Triplet {
        n_rows: m,
        n_cols: n,
        irow: vec![1, 1],
        jcol: vec![1, 2],
        vals: vec![1.0, 1.0],
    };

    // ∇²L = diag(1, 1) — Lagrangian Hessian of ½‖x‖² is I.
    let hess_lag = Triplet {
        n_rows: n,
        n_cols: n,
        irow: vec![1, 2],
        jcol: vec![1, 2],
        vals: vec![1.0, 1.0],
    };

    let qp_data = SqpQpData::build(
        &x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl_orig,
        &xu_orig,
        jac_c,
        hess_lag,
        HessianInertia::Psd,
    );

    // Spot-check the constructed QP bounds:
    //   bl_qp = bl_c − c_vals = 0 − (−1) = 1
    //   bu_qp = bu_c − c_vals = 0 − (−1) = 1
    assert!((qp_data.bl[0] - 1.0).abs() < 1e-12);
    assert!((qp_data.bu[0] - 1.0).abs() < 1e-12);

    // Solve the QP for the SQP step.
    let qp = qp_data.as_qp();
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let sol = solver.solve(&qp, None, &QpOptions::default()).unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);

    // Step p should land at (0, 1) from x_0 = (0, 0).
    assert!((sol.x[0] - 0.0).abs() < 1e-10, "p[0] = {}", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-10, "p[1] = {}", sol.x[1]);
    // QP multiplier on the equality matches the closed-form
    // Lagrange multiplier (sign per our convention):
    // Hx + Aᵀλ = −g ⇒ at x_0 + p = (0, 1): (0, 1) + (λ, λ) = (1, 2),
    // so λ = 1, hence sol.lambda_g[0] should be 1.0.
    assert!(
        (sol.lambda_g[0] - 1.0).abs() < 1e-10,
        "λ_g[0] = {}",
        sol.lambda_g[0]
    );
}

// ─────────────────────────────────────────────────────────────────
// Hand-coded NLP fixture used by the SqpAlgorithm tests below.
//
//     min ½(x₁² + x₂²) − x₁ − 2x₂  s.t. x₁ + x₂ = 1, no bounds.
//
// Closed form (Lagrangian): x* = (0, 1), λ_g = 1, obj* = −1.5.
// Same problem as `qp_assembly_one_sqp_iter_solves_convex_eq_nlp`
// but driven through the full SqpAlgorithm::optimize loop.
// ─────────────────────────────────────────────────────────────────
struct ConvexEqNlp;

impl SqpProblemSpec for ConvexEqNlp {
    fn n(&self) -> usize {
        2
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 2], vec![NLP_UPPER_BOUND_INF; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![0.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        0.5 * (x[0] * x[0] + x[1] * x[1]) - x[0] - 2.0 * x[1]
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] - 1.0, x[1] - 2.0]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] + x[1] - 1.0]
    }
    fn eval_jac_c(&mut self, _x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 2,
            irow: vec![1, 1],
            jcol: vec![1, 2],
            vals: vec![1.0, 1.0],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], _lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 2],
            jcol: vec![1, 2],
            vals: vec![1.0, 1.0],
        }
    }
}

#[test]
fn sqp_optimize_convex_eq_nlp_one_iter() {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg = SqpAlgorithm::new(qp_solver, SqpOptions::default());
    let mut nlp = ConvexEqNlp;

    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(res.status, SqpStatus::Optimal);

    // Closed form: x* = (0, 1), λ_g = 1, obj* = −1.5.
    assert!((res.x[0] - 0.0).abs() < 1e-9, "x[0] = {}", res.x[0]);
    assert!((res.x[1] - 1.0).abs() < 1e-9, "x[1] = {}", res.x[1]);
    assert!(
        (res.lambda_g[0] - 1.0).abs() < 1e-9,
        "λ_g[0] = {}",
        res.lambda_g[0]
    );
    assert!((res.obj - (-1.5)).abs() < 1e-9, "obj = {}", res.obj);

    // The QP is exact for this NLP (∇²L is constant; ∇f is
    // linear) — convergence should take exactly one full SQP
    // iteration: solve QP, take step, KKT-check on the new
    // iterate, declare optimal.
    assert_eq!(res.n_iter, 1);
    assert_eq!(res.n_qp_solves, 1);
}

// ─────────────────────────────────────────────────────────────────
// Ill-conditioned equality-constrained QP (gh #361):
//
//     min ½ xᵀPx + cᵀx   s.t.  x₁ + x₂ + x₃ = 3
//     P = diag(1, 100, 1000),  c = (−2, −101, −1001)
//
// Closed form: x* = (1, 1, 1), λ* = 1, f* = −553.5.
//
// Under a quasi-Newton Hessian this used to expose the
// inconsistent-multiplier curvature pair: `y` was differenced across
// *two different* multipliers, so for these (linear) constraints it
// carried a spurious `Aᵀ(λ_k − λ_{k−1})` term. The multiplier then
// oscillated and diverged (−13, 19, −69, 104, −145, 581, −1320, 3176, …)
// while `x` sat on the exact optimum, and the solve exited
// `MaxIter` *at the right answer* because the stationarity residual is
// formed from that multiplier. Hence the assertions below check the
// multiplier and the status, not just `x`.
// ─────────────────────────────────────────────────────────────────
struct IllConditionedEqQp;

impl SqpProblemSpec for IllConditionedEqQp {
    fn n(&self) -> usize {
        3
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.0, 0.0, 0.0]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 3], vec![NLP_UPPER_BOUND_INF; 3])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![0.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1] + 1000.0 * x[2] * x[2])
            - 2.0 * x[0]
            - 101.0 * x[1]
            - 1001.0 * x[2]
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] - 2.0, 100.0 * x[1] - 101.0, 1000.0 * x[2] - 1001.0]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] + x[1] + x[2] - 3.0]
    }
    fn eval_jac_c(&mut self, _x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 3,
            irow: vec![1, 1, 1],
            jcol: vec![1, 2, 3],
            vals: vec![1.0, 1.0, 1.0],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], _lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: 3,
            n_cols: 3,
            irow: vec![1, 2, 3],
            jcol: vec![1, 2, 3],
            vals: vec![1.0, 100.0, 1000.0],
        }
    }
}

#[test]
fn issue_361_ill_conditioned_eq_qp_converges_under_quasi_newton() {
    for hess in [
        SqpHessianSource::DampedBfgs,
        SqpHessianSource::Lbfgs,
        SqpHessianSource::Exact,
    ] {
        let qp_solver =
            ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
        let mut opts = SqpOptions::default();
        opts.hessian = hess;
        let mut alg = SqpAlgorithm::new(qp_solver, opts);
        let mut nlp = IllConditionedEqQp;
        let res = alg.optimize(&mut nlp).unwrap();

        assert_eq!(res.status, SqpStatus::Optimal, "hess={hess:?}: {res:?}");
        for (i, xi) in res.x.iter().enumerate() {
            assert!(
                (xi - 1.0).abs() < 1e-6,
                "hess={hess:?}: x[{i}] = {xi}, expected 1"
            );
        }
        // The multiplier is the quantity the old inconsistent-`y` update
        // destroyed, so assert it directly.
        assert!(
            (res.lambda_g[0] - 1.0).abs() < 1e-4,
            "hess={hess:?}: λ_g = {}, expected 1",
            res.lambda_g[0]
        );
        assert!(
            (res.obj - (-553.5)).abs() < 1e-5,
            "hess={hess:?}: obj = {}",
            res.obj
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Nonlinear NLP:
//
//     min ½(x − 3)² + ½(y − 2)²   s.t.  x² + y² = 4
//
// True optimum on the circle of radius 2 closest to (3, 2). The
// optimum is on the ray from origin to (3, 2), at distance 2:
// scale (3, 2) by 2/√13. x* = 6/√13 ≈ 1.6641, y* = 4/√13 ≈ 1.1094.
// ─────────────────────────────────────────────────────────────────
struct NonlinearEqNlp;

impl SqpProblemSpec for NonlinearEqNlp {
    fn n(&self) -> usize {
        2
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![1.0, 1.0] // on the feasible disk interior
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 2], vec![NLP_UPPER_BOUND_INF; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![0.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        0.5 * ((x[0] - 3.0).powi(2) + (x[1] - 2.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] - 3.0, x[1] - 2.0]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1] - 4.0]
    }
    fn eval_jac_c(&mut self, x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 2,
            irow: vec![1, 1],
            jcol: vec![1, 2],
            vals: vec![2.0 * x[0], 2.0 * x[1]],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], lambda_g: &[f64]) -> Triplet {
        // ∇²f = I; ∇²c = 2I. So ∇²L = I + λ_g · 2I = (1 + 2λ_g) I.
        let diag = 1.0 + 2.0 * lambda_g[0];
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 2],
            jcol: vec![1, 2],
            vals: vec![diag, diag],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Minimal `IpoptNlp` impl for the convex equality NLP. Exercises
// the IpoptNlpAdapter end-to-end through SqpAlgorithm::optimize.
//
// Same problem as ConvexEqNlp above (closed-form x* = (0, 1),
// λ_g = 1, obj = -1.5) but presented through the algorithm-side
// IpoptNlp trait so the adapter path is fully exercised.
// ─────────────────────────────────────────────────────────────────
struct ConvexEqIpoptNlp {
    x_l: DenseVector,
    x_u: DenseVector,
    d_l: DenseVector,
    d_u: DenseVector,
    jac_c_space: Rc<GenTMatrixSpace>,
    jac_d_space: Rc<GenTMatrixSpace>,
    hess_space: Rc<SymTMatrixSpace>,
    px_l_space: Rc<ExpansionMatrixSpace>,
    px_u_space: Rc<ExpansionMatrixSpace>,
    pd_l_space: Rc<ExpansionMatrixSpace>,
    pd_u_space: Rc<ExpansionMatrixSpace>,
}

impl ConvexEqIpoptNlp {
    fn new() -> Self {
        let n_var = 2;
        // All variables are unbounded → compressed `x_l`/`x_u` are
        // empty, and px_l/px_u are 2×0 ExpansionMatrices with no
        // small→large map entries (matching the OrigIpoptNlp
        // contract for an all-free-variable NLP).
        let bound_space = DenseVectorSpace::new(0);
        let d_space_inh = DenseVectorSpace::new(0);
        let x_l = bound_space.make_new_dense();
        let x_u = bound_space.make_new_dense();
        let d_l = d_space_inh.make_new_dense();
        let d_u = d_space_inh.make_new_dense();

        // Jacobian of c (1 row × 2 cols, nnz = 2 at columns 1, 2).
        let jac_c_space = GenTMatrixSpace::new(1, 2, vec![1, 1], vec![1, 2]);
        // Empty jac_d (0 rows × 2 cols).
        let jac_d_space = GenTMatrixSpace::new(0, 2, vec![], vec![]);
        // Hessian (2 × 2 diag).
        let hess_space = SymTMatrixSpace::new(2, vec![1, 2], vec![1, 2]);
        // Bound expansion matrices: n_large = n_var (or 0 for d), n_small = 0.
        let px_l_space = ExpansionMatrixSpace::new(n_var, 0, &[], 0);
        let px_u_space = ExpansionMatrixSpace::new(n_var, 0, &[], 0);
        let pd_l_space = ExpansionMatrixSpace::new(0, 0, &[], 0);
        let pd_u_space = ExpansionMatrixSpace::new(0, 0, &[], 0);

        Self {
            x_l,
            x_u,
            d_l,
            d_u,
            jac_c_space,
            jac_d_space,
            hess_space,
            px_l_space,
            px_u_space,
            pd_l_space,
            pd_u_space,
        }
    }
}

impl crate::ipopt_nlp::Nlp for ConvexEqIpoptNlp {
    fn n(&self) -> Index {
        2
    }
    fn m_eq(&self) -> Index {
        1
    }
    fn m_ineq(&self) -> Index {
        0
    }

    fn eval_f(&mut self, x: &dyn Vector) -> Number {
        let dx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let v = dx.expanded_values();
        0.5 * (v[0] * v[0] + v[1] * v[1]) - v[0] - 2.0 * v[1]
    }
    fn eval_grad_f(&mut self, x: &dyn Vector, g: &mut dyn Vector) {
        let dx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let v = dx.expanded_values();
        let dg = g.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        dg.set_values(&[v[0] - 1.0, v[1] - 2.0]);
    }
    fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
        let dx = x.as_any().downcast_ref::<DenseVector>().unwrap();
        let v = dx.expanded_values();
        let dc = c.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        dc.set_values(&[v[0] + v[1] - 1.0]);
    }
    fn eval_d(&mut self, _x: &dyn Vector, _d: &mut dyn Vector) {
        // m_ineq = 0; no work.
    }
    fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
        let mut jac = GenTMatrix::new(Rc::clone(&self.jac_c_space));
        jac.set_values(&[1.0, 1.0]);
        Rc::new(jac)
    }
    fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
        Rc::new(GenTMatrix::new(Rc::clone(&self.jac_d_space)))
    }
    fn eval_h(
        &mut self,
        _x: &dyn Vector,
        _obj_factor: Number,
        _y_c: &dyn Vector,
        _y_d: &dyn Vector,
    ) -> Rc<dyn SymMatrix> {
        let mut h = SymTMatrix::new(Rc::clone(&self.hess_space));
        h.set_values(&[1.0, 1.0]);
        Rc::new(h)
    }
}

impl crate::ipopt_nlp::IpoptNlp for ConvexEqIpoptNlp {
    fn x_l(&self) -> &dyn Vector {
        &self.x_l
    }
    fn x_u(&self) -> &dyn Vector {
        &self.x_u
    }
    fn d_l(&self) -> &dyn Vector {
        &self.d_l
    }
    fn d_u(&self) -> &dyn Vector {
        &self.d_u
    }
    fn px_l(&self) -> Rc<dyn Matrix> {
        Rc::new(ExpansionMatrix::new(Rc::clone(&self.px_l_space)))
    }
    fn px_u(&self) -> Rc<dyn Matrix> {
        Rc::new(ExpansionMatrix::new(Rc::clone(&self.px_u_space)))
    }
    fn pd_l(&self) -> Rc<dyn Matrix> {
        Rc::new(ExpansionMatrix::new(Rc::clone(&self.pd_l_space)))
    }
    fn pd_u(&self) -> Rc<dyn Matrix> {
        Rc::new(ExpansionMatrix::new(Rc::clone(&self.pd_u_space)))
    }
    fn get_starting_x(&mut self, x: &mut dyn Vector) -> bool {
        let dx = x.as_any_mut().downcast_mut::<DenseVector>().unwrap();
        dx.set_values(&[0.0, 0.0]);
        true
    }
}

#[test]
fn sqp_via_ipopt_adapter_solves_convex_eq_nlp() {
    let nlp: Rc<RefCell<dyn crate::ipopt_nlp::IpoptNlp>> =
        Rc::new(RefCell::new(ConvexEqIpoptNlp::new()));
    let mut adapter = IpoptNlpAdapter::new(nlp);

    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg = SqpAlgorithm::new(qp_solver, SqpOptions::default());

    let res = alg.optimize(&mut adapter).unwrap();
    assert_eq!(res.status, SqpStatus::Optimal);
    assert!((res.x[0] - 0.0).abs() < 1e-9, "x[0] = {}", res.x[0]);
    assert!((res.x[1] - 1.0).abs() < 1e-9, "x[1] = {}", res.x[1]);
    assert!(
        (res.lambda_g[0] - 1.0).abs() < 1e-9,
        "λ_g[0] = {}",
        res.lambda_g[0]
    );
    assert!((res.obj - (-1.5)).abs() < 1e-9, "obj = {}", res.obj);
    assert_eq!(res.n_iter, 1);
    assert_eq!(res.n_qp_solves, 1);
}

#[test]
fn sqp_lbfgs_converges_on_nonlinear_nlp() {
    // Same circle-projection NLP. With
    // `SqpHessianSource::Lbfgs` the QP Hessian is rebuilt at each
    // step from a circular buffer of (s, y) pairs.  Should
    // converge to the same closed-form optimum.
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut opts = SqpOptions::default();
    opts.hessian = SqpHessianSource::Lbfgs;
    opts.lbfgs_max_history = 6;
    opts.max_iter = 100;
    let mut alg = SqpAlgorithm::new(qp_solver, opts);
    let mut nlp = NonlinearEqNlp;

    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "L-BFGS SQP must converge; got {:?} after {} iters",
        res.status,
        res.n_iter
    );
    let scale = 2.0 / 13.0_f64.sqrt();
    let expected = [3.0 * scale, 2.0 * scale];
    for (i, (a, b)) in res.x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "L-BFGS SQP x[{i}] = {a}, expected {b} (diff {:.2e})",
            (a - b).abs(),
        );
    }
}

#[test]
fn sqp_damped_bfgs_converges_on_nonlinear_nlp() {
    // Same circle-projection NLP. With
    // `SqpHessianSource::DampedBfgs` the QP Hessian comes from
    // a Powell-damped BFGS iterate rather than `eval_hess_lag`.
    // Should converge to the same closed-form optimum;
    // iteration count may be different (BFGS lags exact ∇²L by
    // a few iterations).
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut opts = SqpOptions::default();
    opts.hessian = SqpHessianSource::DampedBfgs;
    opts.max_iter = 50; // generous budget for BFGS convergence
    let mut alg = SqpAlgorithm::new(qp_solver, opts);
    let mut nlp = NonlinearEqNlp;

    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "BFGS SQP must converge; got {:?} after {} iters",
        res.status,
        res.n_iter
    );
    let scale = 2.0 / 13.0_f64.sqrt();
    let expected = [3.0 * scale, 2.0 * scale];
    for (i, (a, b)) in res.x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "BFGS SQP x[{i}] = {a}, expected {b} (diff {:.2e})",
            (a - b).abs(),
        );
    }
}

#[test]
fn sqp_filter_globalization_matches_l1_on_nonlinear_nlp() {
    // Same circle-projection NLP. With SqpGlobalization::Filter,
    // the Fletcher-Leyffer acceptance criterion replaces the l1
    // Armijo backtracking. Both should converge to the same
    // closed-form optimum.
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut opts = SqpOptions::default();
    opts.globalization = SqpGlobalization::Filter;
    let mut alg = SqpAlgorithm::new(qp_solver, opts);
    let mut nlp = NonlinearEqNlp;

    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "filter SQP must converge; got {:?} after {} iters",
        res.status,
        res.n_iter
    );
    let scale = 2.0 / 13.0_f64.sqrt();
    let expected = [3.0 * scale, 2.0 * scale];
    for (i, (a, b)) in res.x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "filter SQP x[{i}] = {a}, expected {b} (diff {:.2e})",
            (a - b).abs(),
        );
    }
}

#[test]
fn sqp_warm_start_skips_qp_solve_when_already_optimal() {
    // Solve the convex equality NLP, then re-run starting from the
    // optimum with the result's `working_set` carried in. The
    // re-run must converge with zero QP solves (the very first
    // KKT check declares optimality before the QP loop body
    // executes).
    let qp_solver_a =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg_a = SqpAlgorithm::new(qp_solver_a, SqpOptions::default());

    let mut nlp = ConvexEqNlp;
    let res_a = alg_a.optimize(&mut nlp).unwrap();
    assert_eq!(res_a.status, SqpStatus::Optimal);
    let ws = res_a.working_set.expect("solve must produce a WS");

    // Build a SqpIterates that pins the iterate at the optimum
    // and carries the working set.
    let warm = SqpIterates {
        x: res_a.x.clone(),
        lambda_g: res_a.lambda_g.clone(),
        lambda_x: res_a.lambda_x.clone(),
        working: Some(ws),
    };

    let qp_solver_b =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg_b = SqpAlgorithm::new(qp_solver_b, SqpOptions::default());
    let res_b = alg_b
        .optimize_with_warm_start(&mut nlp, Some(warm))
        .unwrap();
    assert_eq!(res_b.status, SqpStatus::Optimal);
    assert_eq!(
        res_b.n_qp_solves, 0,
        "warm-started at optimum should solve no QP"
    );
    assert!(res_b.working_set.is_some());
}

#[test]
fn classify_working_set_reproduces_sqp_solver_output_on_convex_eq() {
    // Run a cold SQP solve, then ask `classify_working_set` to
    // reconstruct the active set from the converged primal/dual
    // and bounds. The classifier output must match the QP
    // solver's working set entry-by-entry — the cross-check that
    // the IPM→SQP-corrector handoff produces the right WS.
    use crate::sqp::classify_working_set;

    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg = SqpAlgorithm::new(qp_solver, SqpOptions::default());
    let mut nlp = ConvexEqNlp;
    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(res.status, SqpStatus::Optimal);

    // Use the SqpProblemSpec to fetch bounds + final g(x*).
    let (xl, xu) = nlp.variable_bounds();
    let (bl_c, bu_c) = nlp.constraint_bounds();
    let g_final = nlp.eval_c(&res.x);

    // `lambda_g` is m_eq + m_ineq stacked; for this fixture
    // m_eq = m, m_ineq = 0, so all rows are equalities.
    let ws_classified = classify_working_set(
        &res.lambda_x,
        &res.lambda_g,
        nlp.m(),
        &res.x,
        &xl,
        &xu,
        &g_final,
        &bl_c,
        &bu_c,
        1e-8,
        1e-6,
    );
    let ws_solver = res.working_set.as_ref().unwrap();
    // For this fixture both must classify the single equality
    // row as Equality and both variables as Inactive (no finite
    // bounds).
    assert_eq!(ws_classified.bounds, ws_solver.bounds);
    assert_eq!(ws_classified.constraints, ws_solver.constraints);
}

#[test]
fn sqp_warm_start_rejects_wrong_dimension() {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg = SqpAlgorithm::new(qp_solver, SqpOptions::default());
    let mut nlp = ConvexEqNlp;
    // n = 2, m = 1; deliberately wrong-sized iterate.
    let bogus = SqpIterates {
        x: vec![0.0, 0.0, 0.0],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0, 0.0],
        working: None,
    };
    let err = alg.optimize_with_warm_start(&mut nlp, Some(bogus));
    assert!(matches!(err, Err(SqpError::DimensionMismatch(_))));
}

#[test]
fn sqp_optimize_nonlinear_eq_nlp_converges() {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg = SqpAlgorithm::new(qp_solver, SqpOptions::default());
    let mut nlp = NonlinearEqNlp;

    let res = alg.optimize(&mut nlp).unwrap();
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "status = {:?}, n_iter = {}",
        res.status,
        res.n_iter
    );

    // x* = (6/√13, 4/√13) ≈ (1.6641, 1.1094).
    let scale = 2.0 / 13.0_f64.sqrt();
    let expected = [3.0 * scale, 2.0 * scale];
    for (i, (a, b)) in res.x.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "x[{i}] = {a}, expected {b}, diff = {}",
            (a - b).abs(),
        );
    }
    // Constraint violation should be < tol.
    let cx = res.x[0] * res.x[0] + res.x[1] * res.x[1] - 4.0;
    assert!(cx.abs() < 1e-6, "‖c(x*)‖ = {} but should be near zero", cx);
}

#[test]
fn qp_assembly_preserves_inf_bounds_in_shift() {
    // Variable bounds: xl = -inf, xu = +inf. After shifting by
    // x, they should remain at the sentinel ±inf values.
    let n = 1;
    let m = 0;
    let x = vec![3.7];
    let grad_f = vec![0.0];
    let c_vals: Vec<f64> = vec![];
    let bl_c: Vec<f64> = vec![];
    let bu_c: Vec<f64> = vec![];
    let xl_orig = vec![NLP_LOWER_BOUND_INF];
    let xu_orig = vec![NLP_UPPER_BOUND_INF];

    let jac_c = Triplet {
        n_rows: 0,
        n_cols: n,
        irow: vec![],
        jcol: vec![],
        vals: vec![],
    };
    let hess_lag = Triplet {
        n_rows: n,
        n_cols: n,
        irow: vec![1],
        jcol: vec![1],
        vals: vec![1.0],
    };
    let data = SqpQpData::build(
        &x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl_orig,
        &xu_orig,
        jac_c,
        hess_lag,
        HessianInertia::Psd,
    );

    // After shifting, ±inf is still ±inf — the sentinels survive.
    assert_eq!(data.xl[0], NLP_LOWER_BOUND_INF);
    assert_eq!(data.xu[0], NLP_UPPER_BOUND_INF);

    // Ignore unused-but-clippy-cared-about warning by touching m.
    let _ = m;
}

// ─────────────────────────────────────────────────────────────────
// Maratos-effect fixture (issue #349, FINDING B).
//
//     min 2(x₁² + x₂² − 1) − x₁   s.t.  x₁² + x₂² = 1,   x* = (1, 0).
// ─────────────────────────────────────────────────────────────────
struct MaratosNlp {
    x0: [f64; 2],
}
impl SqpProblemSpec for MaratosNlp {
    fn n(&self) -> usize {
        2
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        self.x0.to_vec()
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 2], vec![NLP_UPPER_BOUND_INF; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![0.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        2.0 * (x[0] * x[0] + x[1] * x[1] - 1.0) - x[0]
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![4.0 * x[0] - 1.0, 4.0 * x[1]]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0] + x[1] * x[1] - 1.0]
    }
    fn eval_jac_c(&mut self, x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 2,
            irow: vec![1, 1],
            jcol: vec![1, 2],
            vals: vec![2.0 * x[0], 2.0 * x[1]],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], lambda_g: &[f64]) -> Triplet {
        // ∇²f = 4I; ∇²c = 2I. ∇²L = (4 + 2λ) I.
        let diag = 4.0 + 2.0 * lambda_g[0];
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 2],
            jcol: vec![1, 2],
            vals: vec![diag, diag],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// HS6 fixture (issue #349, FINDING B).
//
//     min (1 − x₁)²   s.t.  10(x₂ − x₁²) = 0,   x* = (1, 1).
// ─────────────────────────────────────────────────────────────────
struct Hs6Nlp {
    x0: [f64; 2],
}
impl SqpProblemSpec for Hs6Nlp {
    fn n(&self) -> usize {
        2
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        self.x0.to_vec()
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 2], vec![NLP_UPPER_BOUND_INF; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![0.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2)
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![-2.0 * (1.0 - x[0]), 0.0]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![10.0 * (x[1] - x[0] * x[0])]
    }
    fn eval_jac_c(&mut self, x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 2,
            irow: vec![1, 1],
            jcol: vec![1, 2],
            vals: vec![-20.0 * x[0], 10.0],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], lambda_g: &[f64]) -> Triplet {
        // ∇²f = diag(2, 0); ∇²c = diag(−20, 0).
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 2],
            jcol: vec![1, 2],
            vals: vec![2.0 - 20.0 * lambda_g[0], 0.0],
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Issue #349 regression tests: the second-order correction (SOC)
// added to both globalizations must defeat the Maratos effect.
// Before SOC, the filter/l1 line searches stalled
// (`Search_Direction_Becomes_Too_Small`) or hit the iteration cap on
// these standard curved-constraint NLPs; Ipopt converges from every
// start. See `sqp/filter.rs` / `sqp/line_search.rs`.
// ─────────────────────────────────────────────────────────────────

/// The Maratos problem must converge from every starting angle on the
/// unit circle, under every Hessian source and both globalizations —
/// and quickly, since SOC restores the unit-step (superlinear) rate
/// that the Maratos effect otherwise destroys.
#[test]
fn issue_349_maratos_converges_from_all_starts() {
    for hess in [
        SqpHessianSource::Exact,
        SqpHessianSource::DampedBfgs,
        SqpHessianSource::Lbfgs,
    ] {
        for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
            for &t in &[0.05_f64, 0.10, 0.30, 0.50, 0.80] {
                let qp_solver = ParametricActiveSetSolver::new(Box::new(
                    pounce_feral::FeralSolverInterface::new(),
                ));
                let mut opts = SqpOptions::default();
                opts.hessian = hess;
                opts.globalization = glob;
                let mut alg = SqpAlgorithm::new(qp_solver, opts);
                let mut nlp = MaratosNlp {
                    x0: [t.cos(), t.sin()],
                };
                let res = alg.optimize(&mut nlp).unwrap();
                let xerr = ((res.x[0] - 1.0).powi(2) + res.x[1].powi(2)).sqrt();
                assert_eq!(
                    res.status,
                    SqpStatus::Optimal,
                    "Maratos t={t} hess={hess:?} glob={glob:?}: {:?} after {} iters (xerr={xerr:.2e})",
                    res.status,
                    res.n_iter,
                );
                assert!(
                    xerr < 1e-4,
                    "Maratos t={t} hess={hess:?} glob={glob:?}: xerr={xerr:.2e} too large",
                );
            }
        }
    }
}

/// HS6 (singular objective Hessian, strongly curved equality) must
/// converge from `(0.5, 0.5)` under *every* Hessian source and both
/// globalizations. This bites on the parent commit: pre-SOC, the
/// filter globalization with the L-BFGS Hessian stalled here
/// (`QpStepFailed`, `Search_Direction_Becomes_Too_Small`, xerr≈3e-4);
/// the SOC plus the cold-start QP fallback carry it to x* = (1,1).
/// The canonical `(-1.2, 1)` start is additionally checked under the
/// exact Hessian (Ipopt reaches x* in ~5 iters).
#[test]
fn issue_349_hs6_converges() {
    for hess in [
        SqpHessianSource::Exact,
        SqpHessianSource::DampedBfgs,
        SqpHessianSource::Lbfgs,
    ] {
        for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
            let qp_solver =
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
            let mut opts = SqpOptions::default();
            opts.hessian = hess;
            opts.globalization = glob;
            let mut alg = SqpAlgorithm::new(qp_solver, opts);
            let mut nlp = Hs6Nlp { x0: [0.5, 0.5] };
            let res = alg.optimize(&mut nlp).unwrap();
            let xerr = ((res.x[0] - 1.0).powi(2) + (res.x[1] - 1.0).powi(2)).sqrt();
            assert_eq!(
                res.status,
                SqpStatus::Optimal,
                "HS6 (0.5,0.5) hess={hess:?} glob={glob:?}: {:?} after {} iters (f={:.4}, xerr={xerr:.2e})",
                res.status,
                res.n_iter,
                res.obj,
            );
            assert!(
                xerr < 1e-4,
                "HS6 (0.5,0.5) hess={hess:?} glob={glob:?}: xerr={xerr:.2e} too large",
            );
        }
    }

    // Canonical HS6 start, exact Hessian, both globalizations.
    for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
        let qp_solver =
            ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
        let mut opts = SqpOptions::default();
        opts.hessian = SqpHessianSource::Exact;
        opts.globalization = glob;
        let mut alg = SqpAlgorithm::new(qp_solver, opts);
        let mut nlp = Hs6Nlp { x0: [-1.2, 1.0] };
        let res = alg.optimize(&mut nlp).unwrap();
        let xerr = ((res.x[0] - 1.0).powi(2) + (res.x[1] - 1.0).powi(2)).sqrt();
        assert_eq!(
            res.status,
            SqpStatus::Optimal,
            "HS6 (-1.2,1) exact glob={glob:?}: {:?} after {} iters (xerr={xerr:.2e})",
            res.status,
            res.n_iter,
        );
        assert!(
            xerr < 1e-5,
            "HS6 (-1.2,1) exact glob={glob:?}: xerr={xerr:.2e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Issue #416 fixture — extended Rosenbrock inside a ball.
//
//     min Σᵢ 100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²
//     s.t. ‖x‖² ≤ 10 (optional),  −5 ≤ x ≤ 5
//
// From the canonical `(−1.2, 1, −1.2, 1, …)` start, with
// `sqp_hessian = exact`, ∇²L turns indefinite a few iterations in. The
// step QP then hits §4.5 inertia control, and before the `model_step_cap`
// fix each of its "full" steps was a δ-sized crawl: the subproblem spent
// its entire 200-iteration budget making ZERO working-set changes and
// exited `MaxIter`, which the outer loop reported as `QpIterationLimit`
// after 4 iterations at f = 9.62 (against 3.99 from the interior-point
// path). The tell was that the smallest budget that converged was 250
// whether n was 10 or 40 — a fixed-count crawl, not active-set work.
// ─────────────────────────────────────────────────────────────────
struct RosenbrockBallNlp {
    n: usize,
    /// Include the `‖x‖² ≤ 10` row. The failure reproduces identically
    /// with and without it (issue #416: "constraints are not involved"),
    /// and the two shapes take different paths through the QP dispatcher
    /// — general-inequality vs. pure-box — so both are exercised.
    ball: bool,
}

impl SqpProblemSpec for RosenbrockBallNlp {
    fn n(&self) -> usize {
        self.n
    }
    fn m(&self) -> usize {
        usize::from(self.ball)
    }
    fn x_init(&self) -> Vec<Number> {
        (0..self.n)
            .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
            .collect()
    }
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (vec![-5.0; self.n], vec![5.0; self.n])
    }
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        if self.ball {
            (vec![NLP_LOWER_BOUND_INF], vec![10.0])
        } else {
            (Vec::new(), Vec::new())
        }
    }
    fn eval_f(&mut self, x: &[Number]) -> Number {
        (0..self.n - 1)
            .map(|i| {
                let a = x[i + 1] - x[i] * x[i];
                let b = 1.0 - x[i];
                100.0 * a * a + b * b
            })
            .sum()
    }
    fn eval_grad_f(&mut self, x: &[Number]) -> Vec<Number> {
        let mut g = vec![0.0; self.n];
        for i in 0..self.n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            g[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * a;
        }
        g
    }
    fn eval_c(&mut self, x: &[Number]) -> Vec<Number> {
        if self.ball {
            vec![x.iter().map(|v| v * v).sum()]
        } else {
            Vec::new()
        }
    }
    fn eval_jac_c(&mut self, x: &[Number]) -> Triplet {
        if self.ball {
            Triplet {
                n_rows: 1,
                n_cols: self.n,
                irow: vec![1; self.n],
                jcol: (1..=self.n as Index).collect(),
                vals: x.iter().map(|v| 2.0 * v).collect(),
            }
        } else {
            Triplet {
                n_rows: 0,
                n_cols: self.n,
                irow: Vec::new(),
                jcol: Vec::new(),
                vals: Vec::new(),
            }
        }
    }
    fn eval_hess_lag(&mut self, x: &[Number], lambda_g: &[Number]) -> Triplet {
        let n = self.n;
        // ∇²f: tridiagonal. ∇²c = 2I when the ball row is present.
        let mut diag = vec![0.0; n];
        let mut sub = vec![0.0; n - 1];
        for i in 0..n - 1 {
            let a = x[i + 1] - x[i] * x[i];
            diag[i] += -400.0 * a + 800.0 * x[i] * x[i] + 2.0;
            diag[i + 1] += 200.0;
            sub[i] = -400.0 * x[i];
        }
        if self.ball {
            for d in diag.iter_mut() {
                *d += 2.0 * lambda_g[0];
            }
        }
        let mut irow: Vec<Index> = (1..=n as Index).collect();
        let mut jcol: Vec<Index> = (1..=n as Index).collect();
        let mut vals = diag;
        for (i, &s) in sub.iter().enumerate() {
            irow.push(i as Index + 2);
            jcol.push(i as Index + 1);
            vals.push(s);
        }
        Triplet {
            n_rows: n,
            n_cols: n,
            irow,
            jcol,
            vals,
        }
    }
}

/// The issue's own repro: `n = 10`, exact Hessian, **default** QP budget.
/// Pre-fix this exits `QpIterationLimit` after 4 outer iterations; the
/// documented workaround was `sqp_qp_max_iter = 250`, which is why the
/// budget is left at its default here — raising it would hide the bug.
/// Both globalizations and both problem shapes are checked, and `n = 20`
/// bounds-only pins the dimension-independence: a budget that genuinely
/// ran out would need more room as `n` grows, and this one did not.
#[test]
fn issue_416_indefinite_rosenbrock_converges_at_the_default_qp_budget() {
    for (n, ball) in [(10, true), (10, false), (20, false)] {
        for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
            let qp_solver =
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
            let opts = SqpOptions {
                hessian: SqpHessianSource::Exact,
                globalization: glob,
                ..SqpOptions::default()
            };
            let mut alg = SqpAlgorithm::new(qp_solver, opts);
            let mut nlp = RosenbrockBallNlp { n, ball };
            let res = alg.optimize(&mut nlp).unwrap();

            assert_eq!(
                res.status,
                SqpStatus::Optimal,
                "n={n} ball={ball} glob={glob:?}: {:?} after {} iters (f={:.6}, \
                 stationarity={:.2e})",
                res.status,
                res.n_iter,
                res.obj,
                res.final_stationarity,
            );
            // The stall parked at f = 9.62 with a KKT error of 2.6; the
            // interior-point path reaches 3.9866 on the same problem and start.
            assert!(
                res.obj < 4.1,
                "n={n} ball={ball} glob={glob:?}: f={:.6} — the local minimum here is 3.9866",
                res.obj,
            );
            assert!(
                res.final_stationarity <= 1e-4,
                "n={n} ball={ball} glob={glob:?}: stationarity {:.2e}",
                res.final_stationarity,
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Issue #423 fixture — a chain of coupled double wells.
//
//     min Σᵢ (xᵢ² − cᵢ)² + K Σᵢ (xᵢ₊₁ − xᵢ)²
//
// `m = 0`, every variable bound infinite: the configuration with
// **nothing that can ever block a ratio test**. Started at
// `x = 0.35·1`, where `12xᵢ² − 4cᵢ < 0` for every i, so the very
// first ∇²L is indefinite. The objective is a sum of quartic wells
// and is bounded below, so the NLP is not unbounded.
//
// #419 made an unblocked non-positive-curvature direction a
// recession certificate. Here every indefinite iterate produces one,
// the driver's re-verification correctly says "the NLP is not
// unbounded", and before #423 there was no third branch: the solve
// died at outer iteration 1 with `QpStepFailed`
// (`Search_Direction_Becomes_Too_Small`) at f = 26.03, ~950× the
// f = 0.027424 the interior-point path and the damped-BFGS SQP both
// reach from the same start.
// ─────────────────────────────────────────────────────────────────
struct DoubleWellChainNlp {
    n: usize,
}

impl DoubleWellChainNlp {
    /// Coupling stiffness K.
    const K: Number = 0.5;
    /// Well centres `cᵢ`, spread over `[0.6, 2.4]`.
    fn c(&self, i: usize) -> Number {
        0.6 + 1.8 * (i as Number) / ((self.n - 1) as Number)
    }
}

impl SqpProblemSpec for DoubleWellChainNlp {
    fn n(&self) -> usize {
        self.n
    }
    fn m(&self) -> usize {
        0
    }
    fn x_init(&self) -> Vec<Number> {
        vec![0.35; self.n]
    }
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (
            vec![NLP_LOWER_BOUND_INF; self.n],
            vec![NLP_UPPER_BOUND_INF; self.n],
        )
    }
    fn constraint_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        (Vec::new(), Vec::new())
    }
    fn eval_f(&mut self, x: &[Number]) -> Number {
        let mut f = 0.0;
        for (i, &xi) in x.iter().enumerate() {
            let w = xi * xi - self.c(i);
            f += w * w;
        }
        for i in 0..self.n - 1 {
            let d = x[i + 1] - x[i];
            f += Self::K * d * d;
        }
        f
    }
    fn eval_grad_f(&mut self, x: &[Number]) -> Vec<Number> {
        let mut g: Vec<Number> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 4.0 * xi * (xi * xi - self.c(i)))
            .collect();
        for i in 0..self.n - 1 {
            let d = 2.0 * Self::K * (x[i + 1] - x[i]);
            g[i] -= d;
            g[i + 1] += d;
        }
        g
    }
    fn eval_c(&mut self, _x: &[Number]) -> Vec<Number> {
        Vec::new()
    }
    fn eval_jac_c(&mut self, _x: &[Number]) -> Triplet {
        Triplet {
            n_rows: 0,
            n_cols: self.n,
            irow: Vec::new(),
            jcol: Vec::new(),
            vals: Vec::new(),
        }
    }
    fn eval_hess_lag(&mut self, x: &[Number], _lambda_g: &[Number]) -> Triplet {
        let n = self.n;
        let mut diag: Vec<Number> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 12.0 * xi * xi - 4.0 * self.c(i))
            .collect();
        for d in diag.iter_mut().take(n - 1) {
            *d += 2.0 * Self::K;
        }
        for d in diag.iter_mut().skip(1) {
            *d += 2.0 * Self::K;
        }
        let mut irow: Vec<Index> = (1..=n as Index).collect();
        let mut jcol: Vec<Index> = (1..=n as Index).collect();
        let mut vals = diag;
        for i in 0..n - 1 {
            irow.push(i as Index + 2);
            jcol.push(i as Index + 1);
            vals.push(-2.0 * Self::K);
        }
        Triplet {
            n_rows: n,
            n_cols: n,
            irow,
            jcol,
            vals,
        }
    }
}

#[test]
fn issue_423_unconstrained_nonconvex_survives_an_unbounded_model() {
    // (n, local minimum reached from `0.35·1` — the value the
    // interior-point path and the pre-#419 SQP both return).
    for (n, f_opt) in [(12, 0.027424), (20, 0.016089)] {
        for hess in [SqpHessianSource::Exact, SqpHessianSource::DampedBfgs] {
            for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
                let qp_solver = ParametricActiveSetSolver::new(Box::new(
                    pounce_feral::FeralSolverInterface::new(),
                ));
                let opts = SqpOptions {
                    hessian: hess,
                    globalization: glob,
                    ..SqpOptions::default()
                };
                let mut alg = SqpAlgorithm::new(qp_solver, opts);
                let mut nlp = DoubleWellChainNlp { n };
                let res = alg.optimize(&mut nlp).unwrap();

                assert_eq!(
                    res.status,
                    SqpStatus::Optimal,
                    "n={n} hess={hess:?} glob={glob:?}: {:?} after {} iters \
                     (f={:.6}, stationarity={:.2e})",
                    res.status,
                    res.n_iter,
                    res.obj,
                    res.final_stationarity,
                );
                // The regression parked at f = 26.03 for n = 12 — ~950x
                // the optimum — after a single outer iteration.
                assert!(
                    (res.obj - f_opt).abs() < 1e-5,
                    "n={n} hess={hess:?} glob={glob:?}: f={:.6}, expected {f_opt:.6}",
                    res.obj,
                );
                assert!(
                    res.final_stationarity <= 1e-4,
                    "n={n} hess={hess:?} glob={glob:?}: stationarity {:.2e}",
                    res.final_stationarity,
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Simplex projection with bounds — the fixture the warm-start
// benchmark (`benchmarks/warmstart/`) is built around, shrunk to
// n = 4:
//
//     min ½‖x − p‖²   s.t.  Σx = 1,  x ≥ 0,   p = (0.9, 0.4, −0.2, 0.1)
//
// Two components clamp at zero, so a cold QP has to *find* that
// active set, while a warm-started one is handed it.
// ─────────────────────────────────────────────────────────────────
struct SimplexProjNlp;

const SIMPLEX_P: [f64; 4] = [0.9, 0.4, -0.2, 0.1];

impl SqpProblemSpec for SimplexProjNlp {
    fn n(&self) -> usize {
        4
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.25; 4]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0; 4], vec![NLP_UPPER_BOUND_INF; 4])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![1.0], vec![1.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        0.5 * x
            .iter()
            .zip(SIMPLEX_P.iter())
            .map(|(xi, pi)| (xi - pi) * (xi - pi))
            .sum::<f64>()
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(SIMPLEX_P.iter())
            .map(|(xi, pi)| xi - pi)
            .collect()
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x.iter().sum::<f64>()]
    }
    fn eval_jac_c(&mut self, _x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 4,
            irow: vec![1; 4],
            jcol: vec![1, 2, 3, 4],
            vals: vec![1.0; 4],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], _lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: 4,
            n_cols: 4,
            irow: vec![1, 2, 3, 4],
            jcol: vec![1, 2, 3, 4],
            vals: vec![1.0; 4],
        }
    }
}

#[test]
fn sqp_reports_qp_working_set_changes_and_warm_start_removes_them() {
    // `n_qp_working_set_changes` is the measurement a working-set
    // warm start is judged on, and the only one that moves on a
    // QP-shaped NLP: the outer loop terminates in one iteration
    // either way, so an outer-iteration comparison shows nothing.
    // Cold must pay for finding the active set; warm-started at the
    // optimum must pay nothing.
    let mut nlp = SimplexProjNlp;

    let qp_solver_a =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg_a = SqpAlgorithm::new(qp_solver_a, SqpOptions::default());
    let cold = alg_a.optimize(&mut nlp).unwrap();
    assert_eq!(cold.status, SqpStatus::Optimal);
    assert!(
        cold.n_qp_working_set_changes > 0,
        "a cold solve has to search for the active set, but reported \
         {} working-set changes over {} QP solves",
        cold.n_qp_working_set_changes,
        cold.n_qp_solves
    );

    let warm = SqpIterates {
        x: cold.x.clone(),
        lambda_g: cold.lambda_g.clone(),
        lambda_x: cold.lambda_x.clone(),
        working: Some(cold.working_set.clone().expect("cold solve produces a WS")),
    };
    let qp_solver_b =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let mut alg_b = SqpAlgorithm::new(qp_solver_b, SqpOptions::default());
    let hot = alg_b
        .optimize_with_warm_start(&mut nlp, Some(warm))
        .unwrap();
    assert_eq!(hot.status, SqpStatus::Optimal);
    assert_eq!(
        hot.n_qp_working_set_changes, 0,
        "warm-started at the optimum with the right working set, no \
         active-set change should be needed"
    );
    assert!(hot.n_qp_working_set_changes < cold.n_qp_working_set_changes);
}

// ─────────────────────────────────────────────────────────────────
// gh#1054: `max_wall_time` / `max_cpu_time` reach the SQP arm.
// ─────────────────────────────────────────────────────────────────

fn sqp_with_deadline(deadline: pounce_common::timing::Deadline) -> SqpAlgorithm {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = SqpOptions {
        max_iter: 100,
        deadline: Some(deadline),
        ..SqpOptions::default()
    };
    SqpAlgorithm::new(qp_solver, opts)
}

#[test]
fn issue_1054_spent_wall_budget_stops_the_sqp_before_its_first_qp() {
    let mut alg = sqp_with_deadline(pounce_common::timing::Deadline::new(0.0, 1e20));
    let res = alg.optimize(&mut NonlinearEqNlp).unwrap();
    assert_eq!(res.status, SqpStatus::WallTimeExceeded);
    assert_eq!(res.n_iter, 0);
    assert_eq!(
        res.n_qp_solves, 0,
        "the budget is checked before the step QP"
    );
}

#[test]
fn issue_1054_spent_cpu_budget_is_reported_as_cpu() {
    let mut alg = sqp_with_deadline(pounce_common::timing::Deadline::new(1e20, 0.0));
    let res = alg.optimize(&mut NonlinearEqNlp).unwrap();
    assert_eq!(res.status, SqpStatus::CpuTimeExceeded);
}

#[test]
fn issue_1054_unlimited_default_budget_changes_nothing() {
    // `1e20` is the registered default for both options and lies outside
    // `Duration`'s range: the conversion to a QP `time_limit` must clamp it,
    // not panic, and the solve must reach the same optimum as with no
    // deadline at all.
    let mut alg = sqp_with_deadline(pounce_common::timing::Deadline::new(1e20, 1e20));
    let res = alg.optimize(&mut NonlinearEqNlp).unwrap();
    assert_eq!(res.status, SqpStatus::Optimal);
    let scale = 2.0 / 13.0_f64.sqrt();
    assert!((res.x[0] - 3.0 * scale).abs() < 1e-6);
    assert!((res.x[1] - 2.0 * scale).abs() < 1e-6);
}

#[test]
fn issue_1054_deadline_remaining_picks_the_tighter_budget() {
    use crate::sqp::sqp_alg::deadline_remaining;
    use pounce_common::timing::{Deadline, DeadlineKind};
    let (left, kind) = deadline_remaining(&Deadline::new(0.0, 1e20));
    assert_eq!(
        (left, kind),
        (std::time::Duration::ZERO, DeadlineKind::Wall)
    );
    let (left, kind) = deadline_remaining(&Deadline::new(1e20, 0.0));
    assert_eq!((left, kind), (std::time::Duration::ZERO, DeadlineKind::Cpu));
    let (left, _) = deadline_remaining(&Deadline::new(1e20, 1e20));
    assert!(left.as_secs() > 1_000_000_000, "unlimited stays unlimited");
}

#[test]
fn issue_1054_crossover_declines_on_a_spent_budget_and_keeps_the_interior_point() {
    use crate::crossover::{CrossoverDecline, CrossoverOptions, CrossoverSeed};
    let sqp_opts = SqpOptions {
        deadline: Some(pounce_common::timing::Deadline::new(0.0, 1e20)),
        ..SqpOptions::default()
    };
    let seed = CrossoverSeed {
        x: vec![1.0, 1.0],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0, 0.0],
    };
    let mut nlp = NonlinearEqNlp;
    let (report, accepted) = crate::crossover::run(
        &mut nlp,
        &seed,
        &CrossoverOptions {
            enabled: true,
            ..CrossoverOptions::default()
        },
        &sqp_opts,
        &QpOptions::sqp_subproblem(),
        || Box::new(pounce_feral::FeralSolverInterface::new()),
        |_| panic!("no full active-set run once the budget is gone"),
    );
    assert!(accepted.is_none());
    assert_eq!(report.declined, Some(CrossoverDecline::TimeLimit));
    assert_eq!(report.n_qp_solves, 0);
}

/// A KKT backend whose every factorization is a hard failure, so the step-3
/// QP cannot end `Optimal`.
struct AlwaysFatal;

impl pounce_linsol::SparseSymLinearSolverInterface for AlwaysFatal {
    fn initialize_structure(
        &mut self,
        _dim: Index,
        _nonzeros: Index,
        _ia: &[Index],
        _ja: &[Index],
    ) -> pounce_linsol::ESymSolverStatus {
        pounce_linsol::ESymSolverStatus::FatalError
    }
    fn values_array_mut(&mut self) -> &mut [Number] {
        &mut []
    }
    fn multi_solve(
        &mut self,
        _new_matrix: bool,
        _ia: &[Index],
        _ja: &[Index],
        _nrhs: Index,
        _rhs_vals: &mut [Number],
        _check_neg_evals: bool,
        _number_of_neg_evals: Index,
    ) -> pounce_linsol::ESymSolverStatus {
        pounce_linsol::ESymSolverStatus::FatalError
    }
    fn number_of_neg_evals(&self) -> Index {
        0
    }
    fn increase_quality(&mut self) -> bool {
        false
    }
    fn provides_inertia(&self) -> bool {
        true
    }
    fn matrix_format(&self) -> pounce_linsol::EMatrixFormat {
        pounce_linsol::EMatrixFormat::TripletFormat
    }
}

/// gh#1061: `QpFailed` folds every non-optimal step-3 outcome together, so
/// the report carries the QP's own verdict and what it spent.
#[test]
fn issue_1061_crossover_reports_why_the_step3_qp_failed() {
    use crate::crossover::{CrossoverDecline, CrossoverOptions, CrossoverSeed};
    let seed = CrossoverSeed {
        x: vec![1.0, 1.0],
        lambda_g: vec![0.0],
        lambda_x: vec![0.0, 0.0],
    };
    let mut nlp = NonlinearEqNlp;
    let (report, accepted) = crate::crossover::run(
        &mut nlp,
        &seed,
        &CrossoverOptions {
            enabled: true,
            max_iter: 0,
            ..CrossoverOptions::default()
        },
        &SqpOptions::default(),
        &QpOptions::sqp_subproblem(),
        || Box::new(AlwaysFatal),
        |_| panic!("step 4 is disabled"),
    );
    assert!(accepted.is_none());
    assert_eq!(report.declined, Some(CrossoverDecline::QpFailed));
    let eqp = report.eqp.expect("the step-3 QP ran, so it is reported");
    assert!(
        matches!(eqp.outcome, Err(pounce_qp::QpError::LinearSolverFailure(_))),
        "the hard error must survive, not fold into QpFailed: {:?}",
        eqp.outcome
    );
}

/// The accepted path reports the step-3 QP too, so a slow crossover that
/// succeeded can still be attributed.
#[test]
fn issue_1061_an_accepted_crossover_reports_its_step3_qp() {
    use crate::crossover::{CrossoverOptions, CrossoverSeed};
    let mut nlp = NonlinearEqNlp;
    let solved = sqp_with_deadline(pounce_common::timing::Deadline::new(1e20, 1e20))
        .optimize(&mut nlp)
        .unwrap();
    assert_eq!(solved.status, SqpStatus::Optimal);
    let seed = CrossoverSeed {
        x: solved.x.clone(),
        lambda_g: solved.lambda_g.clone(),
        lambda_x: solved.lambda_x.clone(),
    };
    let (report, accepted) = crate::crossover::run(
        &mut nlp,
        &seed,
        &CrossoverOptions {
            enabled: true,
            ..CrossoverOptions::default()
        },
        &SqpOptions::default(),
        &QpOptions::sqp_subproblem(),
        || Box::new(pounce_feral::FeralSolverInterface::new()),
        |o| {
            Some(SqpAlgorithm::new(
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new())),
                o,
            ))
        },
    );
    assert!(accepted.is_some(), "{report:?}");
    let eqp = report.eqp.expect("the step-3 QP ran, so it is reported");
    assert_eq!(eqp.outcome, Ok(QpStatus::Optimal));
}

// ─────────────────────────────────────────────────────────────────
// gh#1054: L-BFGS in lifted low-rank form.
// ─────────────────────────────────────────────────────────────────

/// `min ½‖x − t‖² + ¼ Σ xᵢ⁴  s.t.  xᵢ − xᵢ₊₁ ≤ 0.5,  Σ xᵢ = n/2,  −1 ≤ xᵢ ≤ 0.8`
/// — sparse, bounded, with a dense-coupling equality and a chain of
/// inequalities, so both bounds and rows enter the working set.
struct ChainNlp {
    n: usize,
}

impl ChainNlp {
    fn target(i: usize) -> f64 {
        ((i * 7919) % 13) as f64 / 6.0 - 0.5
    }
}

impl SqpProblemSpec for ChainNlp {
    fn n(&self) -> usize {
        self.n
    }
    fn m(&self) -> usize {
        self.n
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.0; self.n]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-1.0; self.n], vec![0.8; self.n])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        // Row 0 is the equality; rows 1.. the chain.
        let mut bl = vec![NLP_LOWER_BOUND_INF; self.n];
        let mut bu = vec![0.5; self.n];
        bl[0] = self.n as f64 / 2.0 * 0.3;
        bu[0] = bl[0];
        (bl, bu)
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &v)| 0.5 * (v - Self::target(i)).powi(2) + 0.25 * v.powi(4))
            .sum()
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| v - Self::target(i) + v.powi(3))
            .collect()
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        let mut c = vec![x.iter().sum::<f64>()];
        c.extend((1..self.n).map(|i| x[i - 1] - x[i]));
        c
    }
    fn eval_jac_c(&mut self, _x: &[f64]) -> Triplet {
        let (mut irow, mut jcol, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..self.n {
            irow.push(1);
            jcol.push(j as Index + 1);
            vals.push(1.0);
        }
        for i in 1..self.n {
            irow.extend([i as Index + 1, i as Index + 1]);
            jcol.extend([i as Index, i as Index + 1]);
            vals.extend([1.0, -1.0]);
        }
        Triplet {
            n_rows: self.n,
            n_cols: self.n,
            irow,
            jcol,
            vals,
        }
    }
    fn eval_hess_lag(&mut self, x: &[f64], _lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: self.n,
            n_cols: self.n,
            irow: (1..=self.n as Index).collect(),
            jcol: (1..=self.n as Index).collect(),
            vals: x.iter().map(|v| 1.0 + 3.0 * v * v).collect(),
        }
    }
}

fn solve_chain(n: usize, hessian: SqpHessianSource, dense_max_n: usize) -> crate::sqp::SqpResult {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = SqpOptions {
        hessian,
        dense_hessian_max_n: dense_max_n,
        max_iter: 200,
        ..SqpOptions::default()
    };
    SqpAlgorithm::new(qp_solver, opts)
        .optimize(&mut ChainNlp { n })
        .unwrap()
}

#[test]
fn issue_1054_lifted_lbfgs_reaches_the_dense_lbfgs_answer() {
    // Same update, same pairs: the lifted QP computes the same step as the
    // dense one, so the two solves must land on the same point.
    let dense = solve_chain(40, SqpHessianSource::Lbfgs, usize::MAX);
    let lifted = solve_chain(40, SqpHessianSource::Lbfgs, 0);
    assert_eq!(dense.status, SqpStatus::Optimal, "dense: {dense:?}");
    assert_eq!(lifted.status, SqpStatus::Optimal, "lifted: {lifted:?}");
    let exact = solve_chain(40, SqpHessianSource::Exact, usize::MAX);
    assert_eq!(exact.status, SqpStatus::Optimal);
    for i in 0..40 {
        assert!(
            (lifted.x[i] - exact.x[i]).abs() < 1e-5,
            "x[{i}]: lifted {} vs exact {}",
            lifted.x[i],
            exact.x[i]
        );
        assert!((lifted.x[i] - dense.x[i]).abs() < 1e-5);
    }
    assert!((lifted.obj - exact.obj).abs() < 1e-8 * exact.obj.abs().max(1.0));
    // The working set comes back in the NLP's dimensions, lift stripped.
    let ws = lifted.working_set.expect("a QP was solved");
    assert_eq!((ws.bounds.len(), ws.constraints.len()), (40, 40));
}

#[test]
fn issue_1054_quasi_newton_auto_resolves_by_size() {
    use SqpHessianSource::*;
    assert_eq!(QuasiNewtonAuto.resolve(1000, 1000), DampedBfgs);
    assert_eq!(QuasiNewtonAuto.resolve(1001, 1000), Lbfgs);
    assert_eq!(
        DampedBfgs.resolve(1_000_000, 1000),
        DampedBfgs,
        "explicit choice stands"
    );
    assert_eq!(Exact.resolve(1_000_000, 1000), Exact);
}

/// `min Σ ½(xᵢ − tᵢ)² + ¼ xᵢ⁴  s.t.  Σ xᵢ = n/10,  −1 ≤ xᵢ ≤ 0.8`, where
/// only every 1000th target lies outside the box, started feasible: a few
/// dozen working-set changes at any `n`, so the cost measured is the QP's
/// size and not the length of a cold active-set walk.
struct FewActiveNlp {
    n: usize,
}

impl FewActiveNlp {
    fn target(i: usize) -> f64 {
        if i % 1000 == 7 {
            1.5
        } else {
            0.1 + 0.05 * ((i % 7) as f64 - 3.0)
        }
    }
}

impl SqpProblemSpec for FewActiveNlp {
    fn n(&self) -> usize {
        self.n
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.1; self.n]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![-1.0; self.n], vec![0.8; self.n])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let c = self.n as f64 / 10.0;
        (vec![c], vec![c])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &v)| 0.5 * (v - Self::target(i)).powi(2) + 0.25 * v.powi(4))
            .sum()
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| v - Self::target(i) + v.powi(3))
            .collect()
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x.iter().sum()]
    }
    fn eval_jac_c(&mut self, _x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: self.n,
            irow: vec![1; self.n],
            jcol: (1..=self.n as Index).collect(),
            vals: vec![1.0; self.n],
        }
    }
    fn eval_hess_lag(&mut self, x: &[f64], _lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: self.n,
            n_cols: self.n,
            irow: (1..=self.n as Index).collect(),
            jcol: (1..=self.n as Index).collect(),
            vals: x.iter().map(|v| 1.0 + 3.0 * v * v).collect(),
        }
    }
}

#[test]
fn issue_1054_quasi_newton_at_twenty_thousand_variables() {
    // A dense BFGS matrix here is 2·10⁸ stored entries (1.6 GB as a
    // triplet's values alone) and every QP factorization dense — before
    // gh#1054 this is what `hessian_approximation=limited-memory` handed
    // the QP at any size. `QuasiNewtonAuto` must route to the lifted form by
    // itself, and the solve must then cost what a sparse QP costs.
    let n = 20_000;
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = SqpOptions {
        hessian: SqpHessianSource::QuasiNewtonAuto,
        max_iter: 200,
        ..SqpOptions::default()
    };
    let t = std::time::Instant::now();
    let res = SqpAlgorithm::new(qp_solver, opts)
        .optimize(&mut FewActiveNlp { n })
        .unwrap();
    assert_eq!(
        res.status,
        SqpStatus::Optimal,
        "{:?} after {} iters",
        res.status,
        res.n_iter
    );
    assert!(
        t.elapsed() < std::time::Duration::from_secs(120),
        "took {:?}",
        t.elapsed()
    );
    assert!((res.x.iter().sum::<f64>() - n as f64 / 10.0).abs() < 1e-6);
    // The out-of-box targets end on their upper bound.
    for i in (7..n).step_by(1000) {
        assert!((res.x[i] - 0.8).abs() < 1e-6, "x[{i}] = {}", res.x[i]);
    }
}

// ─────────────────────────────────────────────────────────────────
// gh#1061 sweep: an infeasible linearization is not an infeasible NLP.
//
// `min x  s.t.  x² = 4,  0 ≤ x ≤ 3`, started at `x = 0.25`. The only
// feasible point is `x = 2`. At the start the linearized row asks for
// `0.0625 + 0.5 d = 4`, i.e. `x + d = 8.125`, past the bound at 3: the step
// QP is infeasible, and is certified so. The SQP used to stop there with
// `InfeasibleSubproblem` — `Infeasible_Problem_Detected` to every frontend —
// which is what `deb7` (813 variables, feasible, 97.56) got under
// `algorithm=active-set-sqp`. The QP's elastic step goes to the bound,
// cutting the linearized violation from 3.94 to 2.56, and from `x = 3` the
// linearization is consistent.
// ─────────────────────────────────────────────────────────────────

struct InfeasibleLinearizationNlp;

impl SqpProblemSpec for InfeasibleLinearizationNlp {
    fn n(&self) -> usize {
        1
    }
    fn m(&self) -> usize {
        1
    }
    fn x_init(&self) -> Vec<f64> {
        vec![0.25]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![0.0], vec![3.0])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![4.0], vec![4.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        x[0]
    }
    fn eval_grad_f(&mut self, _x: &[f64]) -> Vec<f64> {
        vec![1.0]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        vec![x[0] * x[0]]
    }
    fn eval_jac_c(&mut self, x: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 1,
            irow: vec![1],
            jcol: vec![1],
            vals: vec![2.0 * x[0]],
        }
    }
    fn eval_hess_lag(&mut self, _x: &[f64], lambda_g: &[f64]) -> Triplet {
        Triplet {
            n_rows: 1,
            n_cols: 1,
            irow: vec![1],
            jcol: vec![1],
            vals: vec![2.0 * lambda_g[0]],
        }
    }
}

#[test]
fn an_infeasible_step_qp_takes_its_elastic_step() {
    for glob in [SqpGlobalization::L1Elastic, SqpGlobalization::Filter] {
        let qp_solver =
            ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
        let opts = SqpOptions {
            globalization: glob,
            ..SqpOptions::default()
        };
        let mut alg = SqpAlgorithm::new(qp_solver, opts);
        let res = alg.optimize(&mut InfeasibleLinearizationNlp).unwrap();
        assert_eq!(
            res.status,
            SqpStatus::Optimal,
            "{glob:?}: status {:?} at x = {:?} after {} iterations",
            res.status,
            res.x,
            res.n_iter
        );
        assert!((res.x[0] - 2.0).abs() < 1e-6, "{glob:?}: x = {}", res.x[0]);
    }
}

// ─────────────────────────────────────────────────────────────────
// gh#1061: a large cold step QP is seeded by the convex interior point.
//
// The gate (`qp_ipm_cold_start_min_dim`, default 5000) keeps every model in
// this file on the old path, so these tests lower it to 0. On the reporter's
// first step QP (n = 14 952, 10 818 equalities, H = I) the seeded solve took
// 4.5 s with zero working-set changes, against a cold active-set start that
// had not finished after 30 minutes; that model is not small enough to live
// here, so these pin the contract on small ones instead.
// ─────────────────────────────────────────────────────────────────

/// A convex QP-shaped NLP: the interior point hands the active-set solve the
/// right working set, so the cold solve that `sqp_reports_qp_working_set_
/// changes_and_warm_start_removes_them` shows paying for its search pays
/// nothing, and lands on the same point.
#[test]
fn ipm_seeded_cold_step_qp_finds_the_working_set() {
    let solve = |gate: Option<usize>| {
        let qp_solver =
            ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
        let opts = SqpOptions {
            qp_ipm_cold_start_min_dim: gate,
            ..SqpOptions::default()
        };
        SqpAlgorithm::new(qp_solver, opts)
            .optimize(&mut SimplexProjNlp)
            .unwrap()
    };
    let cold = solve(None);
    let seeded = solve(Some(0));
    assert_eq!(seeded.status, SqpStatus::Optimal);
    assert!(cold.n_qp_working_set_changes > 0);
    assert_eq!(
        seeded.n_qp_working_set_changes, 0,
        "the interior point should have supplied the working set"
    );
    for (a, b) in seeded.x.iter().zip(&cold.x) {
        assert!(
            (a - b).abs() < 1e-8,
            "seeded {:?} vs cold {:?}",
            seeded.x,
            cold.x
        );
    }
}

/// A nonconvex NLP under the exact Hessian. Where the Hessian is not PSD the
/// certificate declines and the step QP is solved from cold exactly as
/// before, so the answer cannot change; where an iterate's Hessian happens to
/// be PSD, the seeded solve must reach the same local minimum.
#[test]
fn ipm_seeded_cold_step_qp_leaves_a_nonconvex_solve_alone() {
    for glob in [SqpGlobalization::Filter, SqpGlobalization::L1Elastic] {
        let solve = |gate: Option<usize>| {
            let qp_solver =
                ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
            let opts = SqpOptions {
                hessian: SqpHessianSource::Exact,
                globalization: glob,
                qp_ipm_cold_start_min_dim: gate,
                ..SqpOptions::default()
            };
            SqpAlgorithm::new(qp_solver, opts)
                .optimize(&mut DoubleWellChainNlp { n: 12 })
                .unwrap()
        };
        let cold = solve(None);
        let seeded = solve(Some(0));
        assert_eq!(seeded.status, cold.status, "{glob:?}");
        assert!(
            (seeded.obj - cold.obj).abs() < 1e-8,
            "{glob:?}: seeded f = {}, cold f = {}",
            seeded.obj,
            cold.obj
        );
    }
}

/// The other half of `an_infeasible_step_qp_takes_its_elastic_step`: on an
/// NLP that really is infeasible, elastic steps lead to a point where none
/// reduces the linearized violation, and the verdict stands. The CLI fixture
/// `infeasible_equalities` — `min x₀² + x₁²` with `x₀³ + x₁³` asked to be
/// both 1 and 2, started at `(10⁴, 10⁴)`. Taking elastic steps, its step QPs
/// reached a `γ = 1e6` elastic point whose next step stayed at
/// `‖p‖∞ = 1.9e-6` on every iteration, above `opt_tol`, and the refactor path,
/// which lacked the Schur path's null-iteration guard, ran its whole budget
/// on that no-op: the SQP ended `QpIterationLimit` where it had said
/// infeasible.
struct InfeasibleCubicsNlp;

impl SqpProblemSpec for InfeasibleCubicsNlp {
    fn n(&self) -> usize {
        2
    }
    fn m(&self) -> usize {
        2
    }
    fn x_init(&self) -> Vec<f64> {
        vec![1e4, 1e4]
    }
    fn variable_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![NLP_LOWER_BOUND_INF; 2], vec![NLP_UPPER_BOUND_INF; 2])
    }
    fn constraint_bounds(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![1.0, 2.0], vec![1.0, 2.0])
    }
    fn eval_f(&mut self, x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }
    fn eval_grad_f(&mut self, x: &[f64]) -> Vec<f64> {
        vec![2.0 * x[0], 2.0 * x[1]]
    }
    fn eval_c(&mut self, x: &[f64]) -> Vec<f64> {
        let s = x[0].powi(3) + x[1].powi(3);
        vec![s, s]
    }
    fn eval_jac_c(&mut self, x: &[f64]) -> Triplet {
        let (a, b) = (3.0 * x[0] * x[0], 3.0 * x[1] * x[1]);
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 1, 2, 2],
            jcol: vec![1, 2, 1, 2],
            vals: vec![a, b, a, b],
        }
    }
    fn eval_hess_lag(&mut self, x: &[f64], lambda_g: &[f64]) -> Triplet {
        let l = lambda_g[0] + lambda_g[1];
        Triplet {
            n_rows: 2,
            n_cols: 2,
            irow: vec![1, 2],
            jcol: vec![1, 2],
            vals: vec![2.0 + 6.0 * l * x[0], 2.0 + 6.0 * l * x[1]],
        }
    }
}

#[test]
fn an_infeasible_nlp_is_still_reported_infeasible() {
    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let opts = SqpOptions {
        hessian: SqpHessianSource::Exact,
        ..SqpOptions::default()
    };
    let res = SqpAlgorithm::new(qp_solver, opts)
        .optimize(&mut InfeasibleCubicsNlp)
        .unwrap();
    assert_eq!(
        res.status,
        SqpStatus::InfeasibleSubproblem,
        "status {:?} at x = {:?} after {} iterations",
        res.status,
        res.x,
        res.n_iter
    );
}

/// gh#1061: a warm start that pins feasibly but is far from its answer is
/// given a budget of `WARM_ITERS_PER_SEED_ITER` active-set iterations per
/// iteration the last interior-point seed took, and is seeded once it runs
/// over. On the reporter's model such warm step QPs took 68–350
/// working-set changes and up to 263 s each, where a seed takes ~20 s.
///
/// `min ½‖x‖² − cᵀx  s.t.  Σx ≤ 100,  0 ≤ x ≤ 1`, `0 < c < 1`, warm-started
/// with every lower bound active: `n` drops warm, none seeded.
#[test]
fn a_warm_step_qp_over_its_budget_is_seeded() {
    use crate::sqp::sqp_alg::{IpmSeed, solve_subproblem_qp};
    use pounce_qp::{BoundStatus, WorkingSet};
    let n = 20;
    let c: Vec<Number> = (0..n)
        .map(|i| 0.1 + 0.8 * i as Number / n as Number)
        .collect();
    let grad: Vec<Number> = c.iter().map(|v| -v).collect();
    let idx: Vec<Index> = (1..=n as Index).collect();
    let data = SqpQpData::build(
        &vec![0.0; n],
        &grad,
        &[0.0],
        &[NLP_LOWER_BOUND_INF],
        &[100.0],
        &vec![0.0; n],
        &vec![1.0; n],
        Triplet {
            n_rows: 1,
            n_cols: n,
            irow: vec![1; n],
            jcol: idx.clone(),
            vals: vec![1.0; n],
        },
        Triplet {
            n_rows: n,
            n_cols: n,
            irow: idx.clone(),
            jcol: idx,
            vals: vec![1.0; n],
        },
        HessianInertia::Indefinite,
    );
    let mut warm = WorkingSet::cold(n, 1);
    warm.bounds.fill(BoundStatus::AtLower);
    let opts = QpOptions {
        use_schur_updates: false,
        ..QpOptions::sqp_subproblem()
    };
    let solve = |last_seed: Option<usize>| {
        let mut solver =
            ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
        let seed = IpmSeed::default();
        seed.last_iters.set(last_seed);
        let sol = solve_subproblem_qp(
            &mut solver,
            &data,
            data.as_qp(),
            Some(&warm),
            &opts,
            Some(&seed),
            "step QP",
        )
        .unwrap();
        assert_eq!(sol.status, QpStatus::Optimal);
        for (xi, ci) in sol.x.iter().zip(&c) {
            assert!((xi - ci).abs() < 1e-7, "{:?}", sol.x);
        }
        sol
    };
    // No seed yet, so no budget: the warm solve drops every bound.
    assert_eq!(solve(None).stats.n_working_set_changes, n as u32);
    // A one-iteration seed allows four iterations: two drops, then the seed,
    // whose working set is already right.
    assert_eq!(solve(Some(1)).stats.n_working_set_changes, 0);
}

/// gh#1091: the lifted L-BFGS subproblem reaches the warm Schur rule as
/// `Indefinite` and so leaves updates off at size, as an exact Hessian's
/// does, while a dense PSD one keeps them. Built through `build_low_rank`,
/// the constructor the SQP loop uses, and judged by the driver's own
/// `warm_qp_opts`, so the test fails if either stops carrying the hint the QP
/// solver branches on (#1089 read the Hessian source instead, and kept
/// updates on for the lift).
#[test]
fn issue_1091_a_lifted_lbfgs_warm_step_qp_leaves_schur_updates_off() {
    use crate::sqp::lbfgs::LowRankHessian;
    let (n, m) = (3, 1);
    let x = vec![0.0; n];
    let grad_f = vec![1.0, -1.0, 0.5];
    let c_vals = vec![0.0];
    let (bl_c, bu_c) = (vec![0.0], vec![0.0]);
    let (xl, xu) = (vec![-1.0; n], vec![1.0; n]);
    let jac = || Triplet {
        n_rows: m,
        n_cols: n,
        irow: vec![1, 1, 1],
        jcol: vec![1, 2, 3],
        vals: vec![1.0, 1.0, 1.0],
    };
    // σI + aaᵀ − bbᵀ with σ large enough that B stays PD: the lift is
    // indefinite while the matrix it represents is not.
    let b = LowRankHessian {
        sigma: 2.0,
        vecs: vec![vec![1.0, 0.0, 0.5], vec![0.0, 0.5, 0.0]],
        signs: vec![1.0, -1.0],
    };
    let lifted = SqpQpData::build_low_rank(&x, &grad_f, &c_vals, &bl_c, &bu_c, &xl, &xu, jac(), &b);
    assert_eq!(lifted.hessian_inertia, HessianInertia::Indefinite);
    let dense = SqpQpData::build(
        &x,
        &grad_f,
        &c_vals,
        &bl_c,
        &bu_c,
        &xl,
        &xu,
        jac(),
        Triplet {
            n_rows: n,
            n_cols: n,
            irow: vec![1, 2, 3],
            jcol: vec![1, 2, 3],
            vals: vec![2.0; 3],
        },
        HessianInertia::Psd,
    );

    let qp_solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    let alg = SqpAlgorithm::new(
        qp_solver,
        SqpOptions {
            hessian: SqpHessianSource::Lbfgs,
            ..SqpOptions::default()
        },
    );
    let on = QpOptions {
        use_schur_updates: true,
        ..QpOptions::sqp_subproblem()
    };
    let dim = crate::sqp::options::DEFAULT_QP_SCHUR_INDEFINITE_MAX_DIM;
    let warm = alg.warm_qp_opts(&on, dim, lifted.hessian_inertia);
    assert!(
        warm.is_some_and(|o| !o.use_schur_updates),
        "a warm lifted step QP at size refactors"
    );
    assert!(
        alg.warm_qp_opts(&on, dim, dense.hessian_inertia).is_none(),
        "a warm PSD step QP keeps its updates"
    );
    assert!(
        alg.warm_qp_opts(&on, dim - 1, lifted.hessian_inertia)
            .is_none(),
        "below the ceiling the size rule stands"
    );
}
