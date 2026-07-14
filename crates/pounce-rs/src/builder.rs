//! Ergonomic builder API over the [`TNLP`](crate::TNLP) trait.
//!
//! The raw `TNLP` interface is a faithful port of Ipopt's C++ `TNLP` (nine
//! methods, sparsity bookkeeping, an `Rc<RefCell<dyn TNLP>>` driver) — full
//! control, but heavy for a simple problem. This module offers the
//! argmin-style alternative requested in
//! [#168](https://github.com/jkitchin/pounce/issues/168): implement the small
//! [`Problem`] trait (only `objective` is required), then configure and solve
//! with the [`Nlp`] builder. Anything you don't implement is finite-
//! differenced (gradient / constraint Jacobian) or approximated (the Hessian
//! defaults to limited-memory L-BFGS), so a basic problem stays small while the
//! full `TNLP` trait remains available for everything this doesn't expose.
//!
//! ```
//! use pounce_rs::builder::{Problem, Nlp};
//!
//! // min (x0-1)^2 + (x1-2)^2  s.t.  x0 + x1 == 3,  0 <= xi <= 5
//! struct P;
//! impl Problem for P {
//!     fn objective(&self, x: &[f64]) -> f64 {
//!         (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
//!     }
//!     fn n_constraints(&self) -> usize { 1 }
//!     fn constraints(&self, x: &[f64], g: &mut [f64]) { g[0] = x[0] + x[1]; }
//! }
//!
//! let sol = Nlp::new(P)                       // variable count inferred below
//!     .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
//!     .constraint_bounds(&[3.0], &[3.0])      // equality: lower == upper
//!     .x0(&[0.0, 0.0])
//!     .option_num("tol", 1e-10)
//!     .solve();
//!
//! assert!(sol.success);
//! assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::{
    ApplicationReturnStatus, BoundsInfo, IndexStyle, IpoptApplication, IpoptCq, IpoptData, NlpInfo,
    Solution as TnlpSolution, SparsityRequest, StartingPoint, TNLP,
};

const FD: f64 = 1.4901161193847656e-8; // sqrt(f64::EPSILON)
const INF: f64 = 2.0e19; // Ipopt's "infinity" bound sentinel

/// A nonlinear program. Implement `objective`; override the rest as needed.
///
/// `gradient` / `jacobian` return `false` (their default) to request a
/// finite-difference approximation. The Hessian is never required — the
/// builder uses a limited-memory (L-BFGS) approximation by default.
pub trait Problem {
    /// Objective `f(x)` to minimize.
    fn objective(&self, x: &[f64]) -> f64;

    /// Number of constraints `m` (default `0`, i.e. bound-constrained only).
    fn n_constraints(&self) -> usize {
        0
    }

    /// Constraint values `g(x)` into `out` (length `n_constraints`).
    fn constraints(&self, _x: &[f64], _out: &mut [f64]) {}

    /// Objective gradient `∇f(x)` into `grad`; return `false` for finite
    /// differences.
    fn gradient(&self, _x: &[f64], _grad: &mut [f64]) -> bool {
        false
    }

    /// Dense constraint Jacobian (row-major, `n_constraints × n`) into `jac`;
    /// return `false` for finite differences.
    fn jacobian(&self, _x: &[f64], _jac: &mut [f64]) -> bool {
        false
    }
}

/// The outcome of [`Nlp::solve`].
#[derive(Debug, Clone)]
pub struct Solution {
    /// Solver status; `success` is the convenient boolean.
    pub status: ApplicationReturnStatus,
    /// `true` for `SolveSucceeded` / `SolvedToAcceptableLevel`.
    pub success: bool,
    /// Optimal variables.
    pub x: Vec<f64>,
    /// Objective at the solution.
    pub objective: f64,
    /// Constraint multipliers `λ` (length `n_constraints`).
    pub multipliers: Vec<f64>,
}

/// Builder: `Nlp::new(problem)` then `.var_bounds(..)` / `.x0(..)` (which fix
/// the number of variables) and `.solve()`.
pub struct Nlp<P: Problem> {
    problem: P,
    n: Option<usize>, // inferred from var_bounds / x0 (must agree)
    x_l: Option<Vec<f64>>,
    x_u: Option<Vec<f64>>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Option<Vec<f64>>,
    num: Vec<(String, f64)>,
    int: Vec<(String, i32)>,
    string: Vec<(String, String)>,
}

impl<P: Problem + 'static> Nlp<P> {
    /// A new builder for `problem`. The number of variables is inferred from
    /// the first of [`var_bounds`](Self::var_bounds) / [`x0`](Self::x0) you set
    /// (they must agree); the number of constraints comes from
    /// `Problem::n_constraints`. Variable bounds default to `±∞`, constraint
    /// bounds to `0`, and `x0` to the origin.
    pub fn new(problem: P) -> Self {
        let m = problem.n_constraints();
        Nlp {
            problem,
            n: None,
            x_l: None,
            x_u: None,
            g_l: vec![0.0; m],
            g_u: vec![0.0; m],
            x0: None,
            num: Vec::new(),
            int: Vec::new(),
            string: Vec::new(),
        }
    }

    // Record (and cross-check) the variable count implied by a length-`len`
    // argument.
    fn set_n(&mut self, len: usize, what: &str) {
        match self.n {
            Some(n) if n != len => panic!(
                "pounce_rs::Nlp: {what} has length {len}, but the problem was \
                 already sized to {n} variables",
            ),
            _ => self.n = Some(len),
        }
    }

    /// Variable bounds `x_l ≤ x ≤ x_u` (use `±2e19` for ∞). Fixes the number of
    /// variables.
    pub fn var_bounds(mut self, lo: &[f64], hi: &[f64]) -> Self {
        assert_eq!(lo.len(), hi.len(), "var_bounds: lo and hi differ in length");
        self.set_n(lo.len(), "var_bounds");
        self.x_l = Some(lo.to_vec());
        self.x_u = Some(hi.to_vec());
        self
    }

    /// Constraint bounds `g_l ≤ g(x) ≤ g_u` (`g_l == g_u` is an equality).
    pub fn constraint_bounds(mut self, lo: &[f64], hi: &[f64]) -> Self {
        self.g_l = lo.to_vec();
        self.g_u = hi.to_vec();
        self
    }

    /// Initial guess. Fixes the number of variables.
    pub fn x0(mut self, x0: &[f64]) -> Self {
        self.set_n(x0.len(), "x0");
        self.x0 = Some(x0.to_vec());
        self
    }

    /// A numeric solver option (e.g. `("tol", 1e-8)`).
    pub fn option_num(mut self, tag: &str, value: f64) -> Self {
        self.num.push((tag.to_string(), value));
        self
    }

    /// An integer solver option (e.g. `("max_iter", 500)`).
    pub fn option_int(mut self, tag: &str, value: i32) -> Self {
        self.int.push((tag.to_string(), value));
        self
    }

    /// A string solver option (e.g. `("mu_strategy", "adaptive")`).
    pub fn option_str(mut self, tag: &str, value: &str) -> Self {
        self.string.push((tag.to_string(), value.to_string()));
        self
    }

    /// Build the `TNLP` adapter and run the interior-point solver.
    ///
    /// # Panics
    /// If the number of variables was never fixed (no `var_bounds` or `x0`).
    pub fn solve(self) -> Solution {
        let n = self.n.expect(
            "pounce_rs::Nlp: number of variables unknown — call .var_bounds(..) \
             or .x0(..) to set it",
        );
        let m = self.problem.n_constraints();
        let adapter = Rc::new(RefCell::new(Adapter {
            problem: self.problem,
            n,
            m,
            x_l: self.x_l.unwrap_or_else(|| vec![-INF; n]),
            x_u: self.x_u.unwrap_or_else(|| vec![INF; n]),
            g_l: self.g_l,
            g_u: self.g_u,
            x0: self.x0.unwrap_or_else(|| vec![0.0; n]),
            sol_x: Vec::new(),
            sol_obj: 0.0,
            sol_lambda: Vec::new(),
        }));

        let mut app = IpoptApplication::new();
        app.initialize().expect("IpoptApplication::initialize");
        // No analytic Hessian is required from `Problem`, so default to L-BFGS.
        let _ = app.options_mut().set_string_value(
            "hessian_approximation",
            "limited-memory",
            true,
            true,
        );
        // The active-set SQP engine (selected by `solver_selection=qp-active-set`
        // or `algorithm=active-set-sqp`) reads `sqp_hessian`, whose default
        // `exact` needs the analytic Hessian this builder never supplies. Default
        // it to limited-memory BFGS so the SQP route works Hessian-free too; a
        // user `.option_str("sqp_hessian", ...)` below overrides it.
        let _ = app
            .options_mut()
            .set_string_value("sqp_hessian", "lbfgs", true, true);
        for (k, v) in &self.string {
            let _ = app.options_mut().set_string_value(k, v, true, true);
        }
        for (k, v) in &self.num {
            let _ = app.options_mut().set_numeric_value(k, *v, true, true);
        }
        for (k, v) in &self.int {
            let _ = app.options_mut().set_integer_value(k, *v, true, true);
        }

        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&adapter) as _;
        let status = app.optimize_tnlp(tnlp);
        let a = adapter.borrow();
        Solution {
            status,
            success: matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            x: a.sol_x.clone(),
            objective: a.sol_obj,
            multipliers: a.sol_lambda.clone(),
        }
    }
}

/// Internal `TNLP` adapter: owns the user [`Problem`] and config, fills in
/// finite-difference gradient / Jacobian and a dense Jacobian sparsity.
struct Adapter<P: Problem> {
    problem: P,
    n: usize,
    m: usize,
    x_l: Vec<f64>,
    x_u: Vec<f64>,
    g_l: Vec<f64>,
    g_u: Vec<f64>,
    x0: Vec<f64>,
    sol_x: Vec<f64>,
    sol_obj: f64,
    sol_lambda: Vec<f64>,
}

impl<P: Problem> TNLP for Adapter<P> {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.n) as i32, // dense Jacobian
            nnz_h_lag: 0,                        // L-BFGS: no analytic Hessian
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }

    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some(self.problem.objective(x))
    }

    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad: &mut [f64]) -> bool {
        if self.problem.gradient(x, grad) {
            return true;
        }
        // forward-difference fallback
        let f0 = self.problem.objective(x);
        let mut xp = x.to_vec();
        for j in 0..self.n {
            let h = FD * x[j].abs().max(1.0);
            xp[j] = x[j] + h;
            grad[j] = (self.problem.objective(&xp) - f0) / h;
            xp[j] = x[j];
        }
        true
    }

    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        self.problem.constraints(x, g);
        true
    }

    fn eval_jac_g(&mut self, x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut k = 0;
                for i in 0..self.m {
                    for j in 0..self.n {
                        irow[k] = i as i32;
                        jcol[k] = j as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                if self.problem.jacobian(x, values) {
                    return true;
                }
                // forward-difference fallback (dense)
                let mut g0 = vec![0.0; self.m];
                self.problem.constraints(x, &mut g0);
                let mut xp = x.to_vec();
                let mut gp = vec![0.0; self.m];
                for j in 0..self.n {
                    let h = FD * x[j].abs().max(1.0);
                    xp[j] = x[j] + h;
                    self.problem.constraints(&xp, &mut gp);
                    for i in 0..self.m {
                        values[i * self.n + j] = (gp[i] - g0[i]) / h;
                    }
                    xp[j] = x[j];
                }
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[f64]>,
        _new_x: bool,
        _obj_factor: f64,
        _lambda: Option<&[f64]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false // never called: the builder uses limited-memory (L-BFGS)
    }

    fn finalize_solution(&mut self, sol: TnlpSolution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.sol_x = sol.x.to_vec();
        self.sol_obj = sol.obj_value;
        self.sol_lambda = sol.lambda.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quad; // min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 == 3
    impl Problem for Quad {
        fn objective(&self, x: &[f64]) -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) {
            g[0] = x[0] + x[1];
        }
    }

    #[test]
    fn infers_n_from_bounds_and_solves() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0]) // n inferred = 2
            .constraint_bounds(&[3.0], &[3.0])
            .option_num("tol", 1e-10)
            .solve();
        assert!(sol.success);
        assert!((sol.x[0] - 1.0).abs() < 1e-5 && (sol.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn infers_n_from_x0() {
        let sol = Nlp::new(Quad)
            .constraint_bounds(&[3.0], &[3.0])
            .x0(&[0.0, 0.0]) // n inferred = 2
            .solve();
        assert!(sol.success);
    }

    #[test]
    fn qp_active_set_selection_solves() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("solver_selection", "qp-active-set")
            .solve();
        assert!(sol.success, "status = {:?}", sol.status);
        assert!((sol.x[0] - 1.0).abs() < 1e-4 && (sol.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn forced_convex_selection_fails_in_builder() {
        let sol = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .option_str("solver_selection", "qp-ipm")
            .solve();
        assert!(
            !sol.success,
            "forced qp-ipm must not silently succeed via NLP"
        );
        assert_eq!(sol.status, ApplicationReturnStatus::InvalidOption);
    }

    #[test]
    #[should_panic(expected = "already sized to 2")]
    fn mismatched_sizes_panic() {
        let _ = Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .x0(&[0.0, 0.0, 0.0]) // length 3 != 2
            .solve();
    }

    #[test]
    #[should_panic(expected = "number of variables unknown")]
    fn missing_size_panics() {
        let _ = Nlp::new(Quad).constraint_bounds(&[3.0], &[3.0]).solve();
    }
}
