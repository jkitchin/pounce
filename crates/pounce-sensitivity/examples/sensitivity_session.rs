//! Sensitivity workflow without callbacks.
//!
//! Solves the parametric NLP from
//! `tests/parametric_cpp.rs::ParametricTNLP` once, then issues several
//! cheap follow-up operations against the held KKT factor:
//!
//!   * `parametric_step` for two different parameter perturbations
//!   * `compute_reduced_hessian` over the pinned-row set
//!   * `kkt_solve` for a raw back-solve against a synthetic RHS
//!
//! Each follow-up reuses the symbolic + numeric factor cached inside
//! the linear-solver backend — no IPM iteration, no refactorization.
//!
//! Run with:
//!   `cargo run --example sensitivity_session -p pounce-sensitivity`

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

struct ParametricTNLP {
    eta1: Number,
    eta2: Number,
}

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 4,
            nnz_jac_g: 10,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.eta1;
        b.g_u[2] = self.eta1;
        b.g_l[3] = self.eta2;
        b.g_u[3] = self.eta2;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.15;
        sp.x[1] = 0.15;
        sp.x[2] = 0.0;
        sp.x[3] = 0.0;
        sp.x[4] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        g[3] = 0.0;
        g[4] = 0.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, e1, e2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - e1;
        g[1] = e2 * x1 + x2 - x3 - 1.0;
        g[2] = e1;
        g[3] = e2;
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
                let rs: [Index; 10] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3];
                let cs: [Index; 10] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("Values without x");
                values[0] = 6.0;
                values[1] = 3.0;
                values[2] = 2.0;
                values[3] = -1.0;
                values[4] = x[4];
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0];
                values[8] = 1.0;
                values[9] = 1.0;
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 2, 4, 0];
                let cs: [Index; 5] = [0, 1, 2, 0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("Values without lambda");
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = lam[1];
                values[4] = 0.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn make_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn main() {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP {
        eta1: 5.0,
        eta2: 1.0,
    }));

    let mut solver = Solver::new(make_app(), tnlp);
    let status = solver.solve();
    println!("solve status: {status:?}");
    assert!(solver.converged().is_some(), "solver did not converge");

    let pins = vec![2 as Index, 3];

    // Two cheap parametric steps against the same factor.
    for deltas in &[vec![-0.5, 0.0], vec![0.0, 0.2]] {
        let dx = solver
            .parametric_step(&pins, deltas)
            .expect("parametric_step ok");
        println!("parametric_step(deltas={deltas:?}) -> dx = {dx:?}");
    }

    // Reduced Hessian over the same pinned-row set.
    let hr = solver
        .compute_reduced_hessian(&pins, 1.0)
        .expect("reduced Hessian ok");
    println!("reduced Hessian (2x2, column-major) = {hr:?}");

    // Raw back-solve against a zero RHS — must come back zero.
    let dim = solver.kkt_dim().expect("kkt_dim available");
    let rhs = vec![0.0; dim];
    let mut lhs = vec![1.0; dim];
    solver.kkt_solve(&rhs, &mut lhs).expect("kkt_solve ok");
    let max_abs = lhs.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    println!("kkt_solve(0) max |lhs| = {max_abs:e}");
    assert!(max_abs < 1e-10);
}
