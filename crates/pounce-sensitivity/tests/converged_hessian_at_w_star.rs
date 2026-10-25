//! The post-optimal KKT matrix is the Jacobian of the KKT residual **at the
//! converged point** `w*`, Hessian block included.
//!
//! The main loop checks convergence before it updates the Hessian, so when it
//! stops `data.w` still holds `W(w_k)` from the final Newton step, while the
//! Jacobians and multipliers the sensitivity hook factors are read at `w*`.
//! Factoring that as-is gives a hybrid of two iterates. The application now
//! re-evaluates `W` at `w*` before the hook fires.
//!
//! The check is exact rather than tolerance-bound against the true solution:
//! the reported `w*` sits `O(tol)` off the exact optimum, which would swamp an
//! `O(‖w* − w_k‖)` effect. Instead this test assembles
//!
//! ```text
//!     K(w*) = [ W(x*, λ*)  Jᵀ(x*) ]
//!             [ J(x*)      0      ]
//! ```
//!
//! from the reported `x*`, `λ*` and the model's own derivatives, and asserts
//! that `parametric_step` is `K(w*)⁻¹ · rhs` to near roundoff. The model has no
//! bounds and no inequalities, so there is no barrier diagonal and `K` is
//! exactly that matrix.
//!
//! The model is chosen so that `W` depends on both `x` and `λ`, **and** so
//! that the perturbed direction reaches the entries that move: the parameter
//! `p` enters `c₀` bilinearly as `p·a·b`, putting `λ₀·b` and `λ₀·a` in `W`'s
//! `p` column. The `parametric_cpp` fixture's finite-difference test is blind
//! to the stale `W` for exactly this reason: its λ-dependent entries sit in
//! the column of a parameter that test never perturbs.
//!
//! Measured when this was added: against the stale `W(w_k)` the step was off
//! `K(w*)⁻¹ · rhs` by 7.0e-6 relative; with the refresh, 4.3e-17. Reverting the
//! refresh in `IpoptApplication` turns the second test red and only that one.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

const P0: Number = 0.5;

/// `min (a−2)² + (b−1)² + a²b²/4`
/// `s.t. c₀ = a² + b² + p·a·b − 3 = 0`, `c₁ = p = P0`.
struct BilinearPin {
    /// `(x*, λ*)` as `finalize_solution` reports them, in user space.
    reported: Rc<RefCell<Option<(Vec<Number>, Vec<Number>)>>>,
}

fn grad_f(x: &[Number]) -> [Number; 3] {
    let (a, b) = (x[0], x[1]);
    [
        2.0 * (a - 2.0) + 0.5 * a * b * b,
        2.0 * (b - 1.0) + 0.5 * a * a * b,
        0.0,
    ]
}

/// Constraint Jacobian, dense, rows `c₀`, `c₁`.
fn jac(x: &[Number]) -> [[Number; 3]; 2] {
    let (a, b, p) = (x[0], x[1], x[2]);
    [[2.0 * a + p * b, 2.0 * b + p * a, a * b], [0.0, 0.0, 1.0]]
}

/// Lagrangian Hessian `∇²f + λ₀ ∇²c₀`, dense and symmetric.
fn hess(x: &[Number], obj_factor: Number, lam: &[Number]) -> [[Number; 3]; 3] {
    let (a, b, p) = (x[0], x[1], x[2]);
    let l = lam[0];
    let aa = obj_factor * (2.0 + 0.5 * b * b) + 2.0 * l;
    let bb = obj_factor * (2.0 + 0.5 * a * a) + 2.0 * l;
    let ab = obj_factor * (a * b) + p * l;
    let pa = b * l;
    let pb = a * l;
    [[aa, ab, pa], [ab, bb, pb], [pa, pb, 0.0]]
}

impl TNLP for BilinearPin {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-1.0e19);
        b.x_u.fill(1.0e19);
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = P0;
        b.g_u[1] = P0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 1.0, P0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (a, b) = (x[0], x[1]);
        Some((a - 2.0).powi(2) + (b - 1.0).powi(2) + 0.25 * a * a * b * b)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&grad_f(x));
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (a, b, p) = (x[0], x[1], x[2]);
        g[0] = a * a + b * b + p * a * b - 3.0;
        g[1] = p;
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
                irow.copy_from_slice(&[0, 0, 0, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 2]);
            }
            SparsityRequest::Values { values } => {
                let j = jac(x.expect("eval_jac_g(Values) without x"));
                values.copy_from_slice(&[j[0][0], j[0][1], j[0][2], j[1][2]]);
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
        // Lower triangle: (a,a), (b,a), (b,b), (p,a), (p,b).
        const RC: [(Index, Index); 5] = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)];
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in RC.iter().enumerate() {
                    irow[k] = r;
                    jcol[k] = c;
                }
            }
            SparsityRequest::Values { values } => {
                let h = hess(
                    x.expect("eval_h(Values) without x"),
                    obj_factor,
                    lambda.expect("eval_h(Values) without lambda"),
                );
                for (k, &(r, c)) in RC.iter().enumerate() {
                    values[k] = h[r as usize][c as usize];
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.reported.borrow_mut() = Some((sol.x.to_vec(), sol.lambda.to_vec()));
    }
}

/// Dense Gaussian elimination with partial pivoting.
fn dense_solve(mut a: Vec<Vec<Number>>, mut b: Vec<Number>) -> Vec<Number> {
    let n = b.len();
    for k in 0..n {
        let piv = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        a.swap(k, piv);
        b.swap(k, piv);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let s: Number = (k + 1..n).map(|j| a[k][j] * x[j]).sum();
        x[k] = (b[k] - s) / a[k][k];
    }
    x
}

fn solve() -> (Solver, Vec<Number>, Vec<Number>) {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    // Keep the algorithm's space and the user's the same, so the multipliers
    // `finalize_solution` reports are the ones the held factor was built from.
    opts.set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();
    app.initialize().unwrap();

    let reported = Rc::new(RefCell::new(None));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(BilinearPin {
        reported: Rc::clone(&reported),
    }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let (x, lam) = reported.borrow().clone().expect("finalize_solution ran");
    (solver, x, lam)
}

#[test]
fn the_reported_point_is_a_kkt_point_under_the_sign_convention_used_below() {
    // Guards the assembly below: if the multiplier sign convention were the
    // other one, `K(w*)` would carry `−λ` in `W` and the main test would be
    // comparing against the wrong matrix.
    let (_s, x, lam) = solve();
    let g = grad_f(&x);
    let j = jac(&x);
    for i in 0..3 {
        let r = g[i] + j[0][i] * lam[0] + j[1][i] * lam[1];
        assert!(r.abs() < 1e-7, "stationarity row {i}: {r:e}");
    }
    assert!(
        lam[0].abs() > 1e-2,
        "λ₀ = {} must be order one for W to depend on it",
        lam[0]
    );
}

#[test]
fn the_parametric_step_solves_the_kkt_matrix_at_the_converged_point() {
    let (mut solver, x, lam) = solve();

    let delta = 1.0e-2;
    let dx = solver
        .parametric_step(&[1], &[delta])
        .expect("parametric_step ok");
    assert_eq!(dx.len(), 3);

    // K(w*) from the reported point, rows (a, b, p, c₀, c₁).
    let w = hess(&x, 1.0, &lam);
    let j = jac(&x);
    let mut k = vec![vec![0.0; 5]; 5];
    for r in 0..3 {
        for c in 0..3 {
            k[r][c] = w[r][c];
        }
        for i in 0..2 {
            k[r][3 + i] = j[i][r];
            k[3 + i][r] = j[i][r];
        }
    }
    // Moving the pin's right-hand side by `delta`: `J dx = (0, delta)`.
    let want = dense_solve(k, vec![0.0, 0.0, 0.0, 0.0, delta]);

    assert!(
        (dx[2] - delta).abs() < 1e-14,
        "the pinned parameter moves by delta exactly, got {}",
        dx[2]
    );
    // The direction must reach the λ-dependent `p` column for this test to
    // say anything: both free coordinates move at order delta.
    assert!(want[0].abs() > 1e-4 && want[1].abs() > 1e-4, "{want:?}");

    let scale = want[..3].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for i in 0..3 {
        let rel = (dx[i] - want[i]).abs() / scale;
        // Against `W(w_k)` this read 7.0e-6; with the refresh, 4.3e-17.
        assert!(
            rel < 1e-13,
            "dx[{i}] = {:.17e}, K(w*)⁻¹·rhs = {:.17e}, rel {rel:e}",
            dx[i],
            want[i]
        );
    }
}
