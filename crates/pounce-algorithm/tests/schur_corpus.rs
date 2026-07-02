//! pounce#180 item 2, Phase 3 — corpus correctness parity in CI.
//!
//! Solves a scalable, strictly-convex, box-and-equality-constrained NLP both
//! through the standard full-space solver and through the Schur KKT solver
//! (constraint-dual block as the Schur set, the range/null-space split), and
//! asserts the two reach the *same* optimum. This guards, at moderate scale and
//! for a nontrivial (~15-iteration) barrier trajectory, that the Schur path —
//! block factorization + Sylvester inertia + the per-iteration inertia check +
//! resolve — stays in lock-step with the monolithic path. The larger sweep and
//! the linear-algebra timing comparison live in
//! `python/benchmarks/schur_kkt_180.py`.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

/// min Σ 0.5 d_i (x_i − t_i)²  s.t.  A x = b,  −0.5 ≤ x ≤ 0.5.
/// Diagonal (PD) Hessian ⇒ the primal block stays positive definite, so the
/// dual block is a clean Schur set. Each equality couples a contiguous window
/// of variables (banded Jacobian).
struct ConvexQp {
    n: usize,
    m: usize,
    width: usize,
    t: Vec<Number>,
    d: Vec<Number>,
    // Jacobian triplet (constant): row i couples cols [starts[i], starts[i]+width).
    starts: Vec<usize>,
    coef: Vec<Number>, // length m*width
    b: Vec<Number>,
    captured_x: Rc<RefCell<Vec<Number>>>,
}

impl ConvexQp {
    fn new(n: usize, m: usize) -> Self {
        let width = (n / (2 * m)).max(4);
        // Deterministic pseudo-random data (no rng dependency).
        let h = |k: usize| ((k.wrapping_mul(2_654_435_761) >> 8) % 1000) as f64 / 1000.0;
        let t: Vec<Number> = (0..n).map(|i| 2.0 * h(i + 1) - 1.0).collect();
        let d: Vec<Number> = (0..n).map(|i| 0.5 + 1.5 * h(i + 7)).collect();
        let starts: Vec<usize> = (0..m)
            .map(|i| {
                if m > 1 {
                    (i * (n - width)) / (m - 1)
                } else {
                    0
                }
            })
            .collect();
        let coef: Vec<Number> = (0..m * width).map(|k| 2.0 * h(k + 101) - 1.0).collect();
        // Feasible RHS from a point strictly inside the box.
        let x_feas: Vec<Number> = (0..n).map(|i| 0.8 * (h(i + 31) - 0.5)).collect();
        let mut b = vec![0.0; m];
        for i in 0..m {
            let mut s = 0.0;
            for w in 0..width {
                s += coef[i * width + w] * x_feas[starts[i] + w];
            }
            b[i] = s;
        }
        Self {
            n,
            m,
            width,
            t,
            d,
            starts,
            coef,
            b,
            captured_x: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl TNLP for ConvexQp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as i32,
            m: self.m as i32,
            nnz_jac_g: (self.m * self.width) as i32,
            nnz_h_lag: self.n as i32, // diagonal
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -0.5);
        b.x_u.iter_mut().for_each(|v| *v = 0.5);
        // Equalities.
        for i in 0..self.m {
            b.g_l[i] = self.b[i];
            b.g_u[i] = self.b[i];
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 0.0;
        for i in 0..self.n {
            let dx = x[i] - self.t[i];
            f += 0.5 * self.d[i] * dx * dx;
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.n {
            g[i] = self.d[i] * (x[i] - self.t[i]);
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for i in 0..self.m {
            let mut s = 0.0;
            for w in 0..self.width {
                s += self.coef[i * self.width + w] * x[self.starts[i] + w];
            }
            g[i] = s;
        }
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
                let mut k = 0;
                for i in 0..self.m {
                    for w in 0..self.width {
                        irow[k] = i as i32;
                        jcol[k] = (self.starts[i] + w) as i32;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&self.coef);
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
        // Linear constraints ⇒ Hessian of Lagrangian = obj_factor · diag(d).
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..self.n {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                for i in 0..self.n {
                    values[i] = obj_factor * self.d[i];
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.captured_x.borrow_mut() = sol.x.to_vec();
    }
}

fn solve(n: usize, m: usize, schur: bool) -> (ApplicationReturnStatus, i32, Number, Vec<Number>) {
    let mut app = IpoptApplication::new();
    if schur {
        // Constraint-dual block: KKT dim = n + n_slack + n_eq + n_ineq. All
        // constraints are equalities ⇒ n_slack = n_ineq = 0, so dim = n + m and
        // the dual block is [n, n + m).
        app.set_kkt_schur_block((n..n + m).collect());
    }
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.initialize().unwrap();
    let prob = ConvexQp::new(n, m);
    let cap = Rc::clone(&prob.captured_x);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(prob));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    let x = cap.borrow().clone();
    (status, stats.iteration_count, stats.final_objective, x)
}

fn ok(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

fn parity_case(n: usize, m: usize) {
    let (s_std, it_std, f_std, x_std) = solve(n, m, false);
    let (s_sch, it_sch, f_sch, x_sch) = solve(n, m, true);
    assert!(ok(s_std), "std status {s_std:?} (n={n}, m={m})");
    assert!(ok(s_sch), "schur status {s_sch:?} (n={n}, m={m})");
    let dx = x_std
        .iter()
        .zip(&x_sch)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "schur corpus n={n} m={m}: iters std/schur={it_std}/{it_sch} f={f_std:.6} max|dx|={dx:e}"
    );
    assert!(
        (f_std - f_sch).abs() < 1e-6,
        "objective mismatch: std={f_std} schur={f_sch}"
    );
    assert!(dx < 1e-6, "solution mismatch: max|dx|={dx:e}");
}

#[test]
fn schur_matches_full_space_small() {
    parity_case(120, 6);
}

#[test]
fn schur_matches_full_space_medium() {
    parity_case(600, 10);
}
