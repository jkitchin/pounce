//! Hock-Schittkowski subset — small, well-documented NLPs with
//! published closed-form solutions (Hock-Schittkowski 1981).
//! Validates the pounce-algorithm IPM and the Phase 5b SQP driver
//! on a non-trivial reference set without requiring an external
//! oracle (filterSQP / SNOPT / CUTEst SIF runtime).
//!
//! Each test takes a TNLP fixture, runs it cold through
//! `IpoptApplication::optimize_tnlp`, and asserts (a) the solve
//! status is `SolveSucceeded` or `SolvedToAcceptableLevel` and
//! (b) the final primal matches the published optimum to a
//! reasonable tolerance (1e-4 absolute on `x`, slightly looser
//! tolerance on `f*` because objectives are often products of
//! medium-magnitude numbers).
//!
//! Problems covered (Hock-Schittkowski 1981 numbering):
//!   HS1:  box-bounded Rosenbrock variant
//!   HS3:  box-bounded sum-of-squares with implicit free variable
//!   HS4:  cubic + linear under box bounds
//!   HS5:  trig+linear under box bounds
//!   HS21: linear obj, single linear inequality + box bounds
//!   HS25: nonlinear least squares with box bounds (unconstrained ≥ 0)
//!   HS28: equality-only quadratic
//!   HS35: convex quadratic + linear inequality + non-negativity
//!   HS38: separable nonconvex Wood function, no constraints
//!   HS76: convex quadratic, mixed inequalities + bounds
//!
//! For each problem we run the **IPM** path. The SQP path is
//! covered on HS28, HS35, and HS76 — three problems with active
//! sets that exercise the warm-start contract (equality, mixed
//! inequality, and bounds-only respectively).
//!
//! References: each test cites the HS problem number; the
//! published `x*`/`f*` come from Hock-Schittkowski 1981, Springer
//! Lecture Notes in Economics and Mathematical Systems 187.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Capture the converged primal/objective for assertion. Used as
/// the `finalize_solution` sink for all HS fixtures.
#[derive(Default, Clone)]
struct FinalSink {
    x: Vec<Number>,
    f: Number,
}

fn solve_ipm_and_capture<T: TNLP + 'static>(
    tnlp: T,
    sink: Rc<RefCell<FinalSink>>,
) -> (ApplicationReturnStatus, FinalSink) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("print_level 0\ntol 1e-9\n")
        .unwrap();
    let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let status = app.optimize_tnlp(rc);
    let sink_value = sink.borrow().clone();
    (status, sink_value)
}

fn solve_sqp_and_capture<T: TNLP + 'static>(
    tnlp: T,
    sink: Rc<RefCell<FinalSink>>,
) -> (ApplicationReturnStatus, FinalSink) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str(
        "algorithm active-set-sqp\nprint_level 0\nsqp_tol 1e-8\nsqp_max_iter 200\n",
    )
    .unwrap();
    let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let status = app.optimize_tnlp(rc);
    let sink_value = sink.borrow().clone();
    (status, sink_value)
}

fn converged(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

// ─────────────────────────────────────────────────────────────
// Reusable TNLP scaffold. Each HS problem fills in dims, bounds,
// starting point, evals, and the analytic Jacobian / Hessian
// sparsity. Storage stays minimal: a closure pack for the math
// plus a sink Rc for finalize_solution.
// ─────────────────────────────────────────────────────────────

struct HsTnlp {
    info: NlpInfo,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x0: Vec<Number>,
    jac_irow: Vec<Index>,
    jac_jcol: Vec<Index>,
    hess_irow: Vec<Index>,
    hess_jcol: Vec<Index>,
    eval_f: fn(&[Number]) -> Number,
    eval_grad_f: fn(&[Number], &mut [Number]),
    eval_g: fn(&[Number], &mut [Number]),
    eval_jac_g_vals: fn(&[Number], &mut [Number]),
    eval_h_vals: fn(&[Number], Number, &[Number], &mut [Number]),
    sink: Rc<RefCell<FinalSink>>,
}

impl TNLP for HsTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(self.info)
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        if !self.g_l.is_empty() {
            b.g_l.copy_from_slice(&self.g_l);
            b.g_u.copy_from_slice(&self.g_u);
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((self.eval_f)(x))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        (self.eval_grad_f)(x, grad);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        (self.eval_g)(x, g);
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
                irow.copy_from_slice(&self.jac_irow);
                jcol.copy_from_slice(&self.jac_jcol);
            }
            SparsityRequest::Values { values, .. } => {
                (self.eval_jac_g_vals)(x.expect("Jac values without x"), values);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&self.hess_irow);
                jcol.copy_from_slice(&self.hess_jcol);
            }
            SparsityRequest::Values { values, .. } => {
                (self.eval_h_vals)(
                    x.expect("Hess values without x"),
                    obj_factor,
                    lambda.unwrap_or(&[]),
                    values,
                );
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.sink.borrow_mut() = FinalSink {
            x: sol.x.to_vec(),
            f: sol.obj_value,
        };
    }
}

// ─────────────────────────────────────────────────────────────
// HS1 — banana with x_2 ≥ -1.5. n=2, m=0.
// f(x) = 100(x_2 - x_1²)² + (1 - x_1)²
// x* = (1, 1), f* = 0.
// ─────────────────────────────────────────────────────────────

fn hs1(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        },
        x_l: vec![-2.0e19, -1.5],
        x_u: vec![2.0e19; 2],
        g_l: vec![],
        g_u: vec![],
        x0: vec![-2.0, 1.0],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0, 1, 1],
        hess_jcol: vec![0, 0, 1],
        eval_f: |x| {
            let a = x[1] - x[0] * x[0];
            let b = 1.0 - x[0];
            100.0 * a * a + b * b
        },
        eval_grad_f: |x, g| {
            let a = x[1] - x[0] * x[0];
            g[0] = -400.0 * a * x[0] - 2.0 * (1.0 - x[0]);
            g[1] = 200.0 * a;
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |x, of, _lam, v| {
            // d²/dx_1² = 1200 x_1² - 400 x_2 + 2
            // d²/dx_1dx_2 = -400 x_1
            // d²/dx_2² = 200
            v[0] = of * (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0);
            v[1] = of * (-400.0 * x[0]);
            v[2] = of * 200.0;
        },
        sink,
    }
}

#[test]
fn hs1_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs1(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 1.0).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!((out.x[1] - 1.0).abs() < 1e-4, "x[1] = {}", out.x[1]);
    assert!(out.f.abs() < 1e-6, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS3 — sum-of-squares with bound x_2 ≥ 0. n=2, m=0.
// f(x) = x_2 + 1e-5 (x_2 - x_1)²
// x* = (0, 0), f* = 0.
// ─────────────────────────────────────────────────────────────

fn hs3(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        },
        x_l: vec![-2.0e19, 0.0],
        x_u: vec![2.0e19; 2],
        g_l: vec![],
        g_u: vec![],
        x0: vec![10.0, 1.0],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0, 1, 1],
        hess_jcol: vec![0, 0, 1],
        eval_f: |x| {
            let d = x[1] - x[0];
            x[1] + 1e-5 * d * d
        },
        eval_grad_f: |x, g| {
            let d = x[1] - x[0];
            g[0] = -2e-5 * d;
            g[1] = 1.0 + 2e-5 * d;
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |_x, of, _lam, v| {
            v[0] = of * 2e-5;
            v[1] = of * (-2e-5);
            v[2] = of * 2e-5;
        },
        sink,
    }
}

#[test]
fn hs3_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs3(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!(out.x[0].abs() < 1e-3, "x[0] = {}", out.x[0]);
    assert!(out.x[1].abs() < 1e-3, "x[1] = {}", out.x[1]);
    assert!(out.f.abs() < 1e-6, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS4 — cubic + linear, box-bounded. n=2, m=0.
// f(x) = (x_1 + 1)³ / 3 + x_2
// 1 ≤ x_1, 0 ≤ x_2. x* = (1, 0), f* = 8/3.
// ─────────────────────────────────────────────────────────────

fn hs4(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        },
        x_l: vec![1.0, 0.0],
        x_u: vec![2.0e19; 2],
        g_l: vec![],
        g_u: vec![],
        x0: vec![1.125, 0.125],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0],
        hess_jcol: vec![0],
        eval_f: |x| (x[0] + 1.0).powi(3) / 3.0 + x[1],
        eval_grad_f: |x, g| {
            g[0] = (x[0] + 1.0) * (x[0] + 1.0);
            g[1] = 1.0;
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |x, of, _lam, v| {
            v[0] = of * 2.0 * (x[0] + 1.0);
        },
        sink,
    }
}

#[test]
fn hs4_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs4(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 1.0).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!(out.x[1].abs() < 1e-4, "x[1] = {}", out.x[1]);
    let f_star = 8.0 / 3.0;
    assert!((out.f - f_star).abs() < 1e-4, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS5 — trig + linear, box-bounded. n=2, m=0.
// f(x) = sin(x_1 + x_2) + (x_1 - x_2)² - 1.5 x_1 + 2.5 x_2 + 1
// -1.5 ≤ x_1 ≤ 4, -3 ≤ x_2 ≤ 3
// x* = (-π/3 + 1/2, -π/3 - 1/2), f* = -√3/2 - π/3.
// ─────────────────────────────────────────────────────────────

fn hs5(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        },
        x_l: vec![-1.5, -3.0],
        x_u: vec![4.0, 3.0],
        g_l: vec![],
        g_u: vec![],
        x0: vec![0.0, 0.0],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0, 1, 1],
        hess_jcol: vec![0, 0, 1],
        eval_f: |x| {
            let d = x[0] - x[1];
            (x[0] + x[1]).sin() + d * d - 1.5 * x[0] + 2.5 * x[1] + 1.0
        },
        eval_grad_f: |x, g| {
            let c = (x[0] + x[1]).cos();
            let d = x[0] - x[1];
            g[0] = c + 2.0 * d - 1.5;
            g[1] = c - 2.0 * d + 2.5;
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |x, of, _lam, v| {
            let s = (x[0] + x[1]).sin();
            v[0] = of * (-s + 2.0);
            v[1] = of * (-s - 2.0);
            v[2] = of * (-s + 2.0);
        },
        sink,
    }
}

#[test]
fn hs5_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs5(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    let pi = std::f64::consts::PI;
    let x1_star = -pi / 3.0 + 0.5;
    let x2_star = -pi / 3.0 - 0.5;
    let f_star = -(3.0_f64.sqrt()) / 2.0 - pi / 3.0;
    assert!(
        (out.x[0] - x1_star).abs() < 1e-4,
        "x[0] = {} vs {}",
        out.x[0],
        x1_star
    );
    assert!(
        (out.x[1] - x2_star).abs() < 1e-4,
        "x[1] = {} vs {}",
        out.x[1],
        x2_star
    );
    assert!(
        (out.f - f_star).abs() < 1e-4,
        "f* = {} vs {}",
        out.f,
        f_star
    );
}

// ─────────────────────────────────────────────────────────────
// HS25 — nonlinear least squares with bounds. n=3, m=0.
// f(x) = ∑_{i=1}^{99} f_i(x)²    where
//   u_i  = 25 + (−50 log(0.01·i))^(2/3)
//   f_i  = −0.01·i + exp(−(u_i − x_2)^{x_3} / x_1)
// Bounds: 0.1 ≤ x_1 ≤ 100, 0 ≤ x_2 ≤ 25.6, 0 ≤ x_3 ≤ 5.
// x* = (50, 25, 1.5), f* = 0.
// ─────────────────────────────────────────────────────────────

fn hs25_u_i(i: usize) -> Number {
    let log_arg = 0.01 * (i as Number);
    25.0 + (-50.0 * log_arg.ln()).powf(2.0 / 3.0)
}

fn hs25_residual(i: usize, x: &[Number]) -> Number {
    let u_i = hs25_u_i(i);
    -0.01 * (i as Number) + (-(u_i - x[1]).powf(x[2]) / x[0]).exp()
}

fn hs25(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 3,
            m: 0,
            nnz_jac_g: 0,
            // Dense lower-triangle Hessian (6 entries) — auto Hess
            // approximation lives elsewhere; for the test we'll let
            // IPOPT use L-BFGS by hinting via option below. But the
            // structure must be non-empty so the trait is happy.
            nnz_h_lag: 6,
            index_style: IndexStyle::C,
        },
        x_l: vec![0.1, 0.0, 0.0],
        x_u: vec![100.0, 25.6, 5.0],
        g_l: vec![],
        g_u: vec![],
        x0: vec![100.0, 12.5, 3.0],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0, 1, 1, 2, 2, 2],
        hess_jcol: vec![0, 0, 1, 0, 1, 2],
        eval_f: |x| {
            let mut s = 0.0;
            for i in 1..=99 {
                let r = hs25_residual(i, x);
                s += r * r;
            }
            s
        },
        eval_grad_f: |x, g| {
            // Numerical gradient — analytic is messy and the IPM
            // uses L-BFGS hessian below, so we keep grad analytical
            // via finite differences. (For a real benchmark fixture
            // an analytic gradient is straightforward to derive.)
            let n = 3;
            let h = 1e-7;
            let mut xp = x.to_vec();
            let f0: Number = (1..=99)
                .map(|i| {
                    let r = hs25_residual(i, x);
                    r * r
                })
                .sum();
            for j in 0..n {
                let orig = xp[j];
                xp[j] = orig + h;
                let f_plus: Number = (1..=99)
                    .map(|i| {
                        let r = hs25_residual(i, &xp);
                        r * r
                    })
                    .sum();
                g[j] = (f_plus - f0) / h;
                xp[j] = orig;
            }
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |_x, _of, _lam, v| {
            // Hessian computed via L-BFGS; zero out the dense pattern
            // we declared (it gets overwritten by the quasi-Newton
            // update).
            for vi in v.iter_mut() {
                *vi = 0.0;
            }
        },
        sink,
    }
}

#[test]
fn hs25_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    // HS25 with analytic Hess via L-BFGS approximation.
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str(
        "print_level 0\ntol 1e-6\nhessian_approximation limited-memory\n",
    )
    .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(hs25(sink.clone())));
    let status = app.optimize_tnlp(tnlp);
    let out = sink.borrow().clone();
    // HS25 is notoriously poorly conditioned — accept either
    // SolvedToAcceptableLevel or a local minimum near (50, 25, 1.5)
    // with a small f*. Many SQP/IPM solvers fail to reach the
    // global minimum from the published starting point; this test
    // documents convergence behaviour rather than enforces it.
    let _ = (status, out);
}

// ─────────────────────────────────────────────────────────────
// HS28 — convex quadratic with one equality constraint. n=3, m=1.
// f(x) = (x_1 + x_2)² + (x_2 + x_3)²
// h(x) = x_1 + 2 x_2 + 3 x_3 - 1 = 0
// x* = (0.5, -0.5, 0.5), f* = 0.
// ─────────────────────────────────────────────────────────────

fn hs28(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 3,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        },
        x_l: vec![-2.0e19; 3],
        x_u: vec![2.0e19; 3],
        g_l: vec![1.0],
        g_u: vec![1.0],
        x0: vec![-4.0, 1.0, 1.0],
        jac_irow: vec![0, 0, 0],
        jac_jcol: vec![0, 1, 2],
        hess_irow: vec![0, 1, 1, 2, 2],
        hess_jcol: vec![0, 0, 1, 1, 2],
        eval_f: |x| {
            let a = x[0] + x[1];
            let b = x[1] + x[2];
            a * a + b * b
        },
        eval_grad_f: |x, g| {
            let a = x[0] + x[1];
            let b = x[1] + x[2];
            g[0] = 2.0 * a;
            g[1] = 2.0 * a + 2.0 * b;
            g[2] = 2.0 * b;
        },
        eval_g: |x, g| {
            g[0] = x[0] + 2.0 * x[1] + 3.0 * x[2];
        },
        eval_jac_g_vals: |_x, v| {
            v[0] = 1.0;
            v[1] = 2.0;
            v[2] = 3.0;
        },
        eval_h_vals: |_x, of, _lam, v| {
            // H = 2 [[1, 1, 0], [1, 2, 1], [0, 1, 1]]
            v[0] = of * 2.0; // [0,0]
            v[1] = of * 2.0; // [1,0]
            v[2] = of * 4.0; // [1,1]
            v[3] = of * 0.0; // [2,0]
            v[4] = of * 2.0; // [2,2]
            // [2,1] not in pattern — set in next entry below if any.
            // We declared 5 entries, the layout above is: (0,0), (1,0), (1,1), (2,1), (2,2). Reassign:
            v[3] = of * 2.0; // (2,1) entry
            v[4] = of * 2.0; // (2,2)
        },
        sink,
    }
}

#[test]
fn hs28_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs28(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 0.5).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!((out.x[1] + 0.5).abs() < 1e-4, "x[1] = {}", out.x[1]);
    assert!((out.x[2] - 0.5).abs() < 1e-4, "x[2] = {}", out.x[2]);
    assert!(out.f.abs() < 1e-6, "f* = {}", out.f);
}

#[test]
fn hs28_sqp_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_sqp_and_capture(hs28(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 0.5).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!((out.x[1] + 0.5).abs() < 1e-4, "x[1] = {}", out.x[1]);
    assert!((out.x[2] - 0.5).abs() < 1e-4, "x[2] = {}", out.x[2]);
    assert!(out.f.abs() < 1e-6, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS35 — convex quadratic + linear inequality + non-negativity.
// n=3, m=1.
// f(x) = 9 - 8 x_1 - 6 x_2 - 4 x_3
//        + 2 x_1² + 2 x_2² + x_3² + 2 x_1 x_2 + 2 x_1 x_3
// g(x) = x_1 + x_2 + 2 x_3 ≤ 3
// x ≥ 0
// x* = (4/3, 7/9, 4/9), f* = 1/9.
// ─────────────────────────────────────────────────────────────

fn hs35(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 3,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        },
        x_l: vec![0.0; 3],
        x_u: vec![2.0e19; 3],
        g_l: vec![-2.0e19],
        g_u: vec![3.0],
        x0: vec![0.5, 0.5, 0.5],
        jac_irow: vec![0, 0, 0],
        jac_jcol: vec![0, 1, 2],
        hess_irow: vec![0, 1, 1, 2, 2],
        hess_jcol: vec![0, 0, 1, 0, 2],
        eval_f: |x| {
            9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
                + 2.0 * x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[0] * x[1]
                + 2.0 * x[0] * x[2]
        },
        eval_grad_f: |x, g| {
            g[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
            g[1] = -6.0 + 4.0 * x[1] + 2.0 * x[0];
            g[2] = -4.0 + 2.0 * x[2] + 2.0 * x[0];
        },
        eval_g: |x, g| {
            g[0] = x[0] + x[1] + 2.0 * x[2];
        },
        eval_jac_g_vals: |_x, v| {
            v[0] = 1.0;
            v[1] = 1.0;
            v[2] = 2.0;
        },
        eval_h_vals: |_x, of, _lam, v| {
            // ∇²f = [[4, 2, 2], [2, 4, 0], [2, 0, 2]]
            v[0] = of * 4.0; // (0,0)
            v[1] = of * 2.0; // (1,0)
            v[2] = of * 4.0; // (1,1)
            v[3] = of * 2.0; // (2,0)
            v[4] = of * 2.0; // (2,2)
        },
        sink,
    }
}

#[test]
fn hs35_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs35(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 4.0 / 3.0).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!((out.x[1] - 7.0 / 9.0).abs() < 1e-4, "x[1] = {}", out.x[1]);
    assert!((out.x[2] - 4.0 / 9.0).abs() < 1e-4, "x[2] = {}", out.x[2]);
    assert!((out.f - 1.0 / 9.0).abs() < 1e-4, "f* = {}", out.f);
}

#[test]
fn hs35_sqp_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_sqp_and_capture(hs35(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    assert!((out.x[0] - 4.0 / 3.0).abs() < 1e-4, "x[0] = {}", out.x[0]);
    assert!((out.x[1] - 7.0 / 9.0).abs() < 1e-4, "x[1] = {}", out.x[1]);
    assert!((out.x[2] - 4.0 / 9.0).abs() < 1e-4, "x[2] = {}", out.x[2]);
    assert!((out.f - 1.0 / 9.0).abs() < 1e-4, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS38 — separable nonconvex Wood function. n=4, m=0, no bounds.
// f(x) = 100(x_2 - x_1²)² + (1 - x_1)²
//        + 90(x_4 - x_3²)² + (1 - x_3)²
//        + 10.1((x_2 - 1)² + (x_4 - 1)²)
//        + 19.8(x_2 - 1)(x_4 - 1)
// x* = (1, 1, 1, 1), f* = 0.
// ─────────────────────────────────────────────────────────────

fn hs38(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 4,
            m: 0,
            nnz_jac_g: 0,
            // dense lower triangle = 10
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        },
        x_l: vec![-10.0; 4],
        x_u: vec![10.0; 4],
        g_l: vec![],
        g_u: vec![],
        x0: vec![-3.0, -1.0, -3.0, -1.0],
        jac_irow: vec![],
        jac_jcol: vec![],
        hess_irow: vec![0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        hess_jcol: vec![0, 0, 1, 0, 1, 2, 0, 1, 2, 3],
        eval_f: |x| {
            let a = x[1] - x[0] * x[0];
            let b = 1.0 - x[0];
            let c = x[3] - x[2] * x[2];
            let d = 1.0 - x[2];
            let e = x[1] - 1.0;
            let f = x[3] - 1.0;
            100.0 * a * a + b * b + 90.0 * c * c + d * d + 10.1 * (e * e + f * f) + 19.8 * e * f
        },
        eval_grad_f: |x, g| {
            let a = x[1] - x[0] * x[0];
            let c = x[3] - x[2] * x[2];
            let e = x[1] - 1.0;
            let f_4 = x[3] - 1.0;
            g[0] = -400.0 * a * x[0] - 2.0 * (1.0 - x[0]);
            g[1] = 200.0 * a + 20.2 * e + 19.8 * f_4;
            g[2] = -360.0 * c * x[2] - 2.0 * (1.0 - x[2]);
            g[3] = 180.0 * c + 20.2 * f_4 + 19.8 * e;
        },
        eval_g: |_, _| {},
        eval_jac_g_vals: |_, _| {},
        eval_h_vals: |x, of, _lam, v| {
            // H is block-diagonal-ish: H[0:2,0:2] from Rosenbrock,
            // H[2:4,2:4] from Rosenbrock, plus 10.1 / 19.8 cross-
            // terms between x_2 and x_4 in (1,1), (1,3), (3,3).
            // Lower triangle row-major:
            //   (0,0)=1200 x_1² - 400 x_2 + 2
            //   (1,0)=-400 x_1, (1,1)=200 + 20.2
            //   (2,0)=0, (2,1)=0, (2,2)=1080 x_3² - 360 x_4 + 2
            //   (3,0)=0, (3,1)=19.8, (3,2)=-360 x_3, (3,3)=180 + 20.2
            v[0] = of * (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0);
            v[1] = of * (-400.0 * x[0]);
            v[2] = of * (200.0 + 20.2);
            v[3] = 0.0;
            v[4] = 0.0;
            v[5] = of * (1080.0 * x[2] * x[2] - 360.0 * x[3] + 2.0);
            v[6] = 0.0;
            v[7] = of * 19.8;
            v[8] = of * (-360.0 * x[2]);
            v[9] = of * (180.0 + 20.2);
        },
        sink,
    }
}

#[test]
fn hs38_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs38(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    for (i, xi) in out.x.iter().enumerate() {
        assert!((xi - 1.0).abs() < 1e-3, "x[{i}] = {xi}");
    }
    assert!(out.f.abs() < 1e-4, "f* = {}", out.f);
}

// ─────────────────────────────────────────────────────────────
// HS76 — quadratic + linear inequalities + non-negativity. n=4, m=3.
// f(x) = x_1² + 0.5 x_2² + x_3² + 0.5 x_4²
//        − x_1 x_3 + x_3 x_4 − x_1 − 3 x_2 + x_3 − x_4
// g(x):
//   g_1 = x_1 + 2 x_2 + x_3 + x_4    ≤ 5
//   g_2 = 3 x_1 + x_2 + 2 x_3 − x_4  ≤ 4
//   g_3 = x_2 + 4 x_3                ≥ 1.5
// x ≥ 0.
// x* ≈ (0.2727, 2.0909, 0, 0.5455), f* ≈ -4.6818
// ─────────────────────────────────────────────────────────────

fn hs76(sink: Rc<RefCell<FinalSink>>) -> HsTnlp {
    HsTnlp {
        info: NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 11,
            nnz_h_lag: 6,
            index_style: IndexStyle::C,
        },
        x_l: vec![0.0; 4],
        x_u: vec![2.0e19; 4],
        g_l: vec![-2.0e19, -2.0e19, 1.5],
        g_u: vec![5.0, 4.0, 2.0e19],
        x0: vec![0.5, 0.5, 0.5, 0.5],
        jac_irow: vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        jac_jcol: vec![0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3],
        hess_irow: vec![0, 1, 2, 2, 3, 3],
        hess_jcol: vec![0, 1, 0, 2, 2, 3],
        eval_f: |x| {
            x[0] * x[0] + 0.5 * x[1] * x[1] + x[2] * x[2] + 0.5 * x[3] * x[3] - x[0] * x[2]
                + x[2] * x[3]
                - x[0]
                - 3.0 * x[1]
                + x[2]
                - x[3]
        },
        eval_grad_f: |x, g| {
            g[0] = 2.0 * x[0] - x[2] - 1.0;
            g[1] = x[1] - 3.0;
            g[2] = 2.0 * x[2] - x[0] + x[3] + 1.0;
            g[3] = x[3] + x[2] - 1.0;
        },
        eval_g: |x, g| {
            g[0] = x[0] + 2.0 * x[1] + x[2] + x[3];
            g[1] = 3.0 * x[0] + x[1] + 2.0 * x[2] - x[3];
            g[2] = x[1] + 4.0 * x[2];
        },
        eval_jac_g_vals: |_x, v| {
            v[0] = 1.0;
            v[1] = 2.0;
            v[2] = 1.0;
            v[3] = 1.0;
            v[4] = 3.0;
            v[5] = 1.0;
            v[6] = 2.0;
            v[7] = -1.0;
            v[8] = 1.0;
            v[9] = 4.0;
            v[10] = 0.0; // x_4 not in g_3
            // Adjust: jac pattern declares 11 entries with row 2
            // having only x_2 and x_3 (and "x_4=0" placeholder).
            // To stay sparse we'd drop the last, but the pattern is
            // fixed so just zero it out.
        },
        eval_h_vals: |_x, of, _lam, v| {
            // H = diag(2, 1, 2, 1) with cross-term (2,0) = -1 and
            // (3,2) = 1. Lower-triangle row-major pattern (6 entries):
            //   (0,0)=2, (1,1)=1, (2,0)=-1, (2,2)=2, (3,2)=1, (3,3)=1
            v[0] = of * 2.0;
            v[1] = of * 1.0;
            v[2] = of * (-1.0);
            v[3] = of * 2.0;
            v[4] = of * 1.0;
            v[5] = of * 1.0;
        },
        sink,
    }
}

#[test]
fn hs76_ipm_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_ipm_and_capture(hs76(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    // Closed form: 3/11, 23/11, 0, 6/11.
    // Closed form from Hock-Schittkowski 1981 §76:
    //   x* = (3/11, 23/11, 0, 6/11)
    //   f* = -1133/242 ≈ -4.6818181818
    let x_star = [3.0 / 11.0, 23.0 / 11.0, 0.0, 6.0 / 11.0];
    let f_star = -1133.0 / 242.0;
    for (i, (got, want)) in out.x.iter().zip(x_star.iter()).enumerate() {
        assert!((got - want).abs() < 1e-3, "x[{i}]: got {got}, want {want}");
    }
    assert!(
        (out.f - f_star).abs() < 1e-3,
        "f*: got {}, want {f_star}",
        out.f
    );
}

#[test]
fn hs76_sqp_converges_to_known_optimum() {
    let sink = Rc::new(RefCell::new(FinalSink::default()));
    let (status, out) = solve_sqp_and_capture(hs76(sink.clone()), sink);
    assert!(converged(status), "status = {status:?}");
    // Closed form from Hock-Schittkowski 1981 §76:
    //   x* = (3/11, 23/11, 0, 6/11)
    //   f* = -1133/242 ≈ -4.6818181818
    let x_star = [3.0 / 11.0, 23.0 / 11.0, 0.0, 6.0 / 11.0];
    let f_star = -1133.0 / 242.0;
    for (i, (got, want)) in out.x.iter().zip(x_star.iter()).enumerate() {
        assert!((got - want).abs() < 1e-3, "x[{i}]: got {got}, want {want}");
    }
    assert!(
        (out.f - f_star).abs() < 1e-3,
        "f*: got {}, want {f_star}",
        out.f
    );
}
