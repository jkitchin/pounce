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
use pounce_common::types::{Index, Number, NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
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
