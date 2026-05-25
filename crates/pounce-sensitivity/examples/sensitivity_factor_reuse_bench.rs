//! Tracking-only timing comparison: factor-once + N sensitivity steps
//! vs N fresh cold-start solves. The point of `pounce_sensitivity::Solver`
//! is that the converged KKT factor survives between operations, so
//! each follow-up parametric step / reduced-Hessian / KKT-solve is a
//! back-solve rather than a full IPM re-run.
//!
//! Not a regression test — numbers vary across machines and builds.
//! Run with:
//!   `cargo run --release --example sensitivity_factor_reuse_bench -p pounce-sensitivity`

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::{SensSolve, Solver};

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

const N: usize = 20;

fn main() {
    let deltas: Vec<Vec<Number>> = (1..=N)
        .map(|k| vec![0.0, 0.01 * k as Number])
        .collect();
    let pins = vec![2 as Index, 3];

    // (1) Held-factor path: 1 solve + N parametric steps.
    let t = Instant::now();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP {
        eta1: 5.0,
        eta2: 1.0,
    }));
    let mut solver = Solver::new(make_app(), tnlp);
    solver.solve();
    let solve_dt = t.elapsed();
    let mut held_steps = Vec::with_capacity(N);
    let t = Instant::now();
    for d in &deltas {
        let dx = solver
            .parametric_step(&pins, d)
            .expect("parametric_step ok");
        held_steps.push(dx);
    }
    let held_step_dt = t.elapsed();
    let held_total = solve_dt + held_step_dt;

    // (2) Cold path: N fresh SensSolve runs (one full IPM each).
    let mut cold_steps = Vec::with_capacity(N);
    let t = Instant::now();
    for d in &deltas {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP {
            eta1: 5.0,
            eta2: 1.0,
        }));
        let mut app = make_app();
        let r = SensSolve::new(pins.clone())
            .with_deltas(d.clone())
            .run(&mut app, tnlp);
        cold_steps.push(r.dx.expect("dx"));
    }
    let cold_total = t.elapsed();

    // Cross-check: held vs cold dx should agree.
    let mut max_err = 0.0_f64;
    for (h, c) in held_steps.iter().zip(cold_steps.iter()) {
        for (a, b) in h.iter().zip(c.iter()) {
            max_err = max_err.max((a - b).abs());
        }
    }

    println!("Workload: 1 IPM solve + {N} parametric steps (Δeta2 sweep).");
    println!();
    println!(
        "Held-factor path : solve {:.2} ms + {N} steps {:.2} ms = {:.2} ms total",
        solve_dt.as_secs_f64() * 1e3,
        held_step_dt.as_secs_f64() * 1e3,
        held_total.as_secs_f64() * 1e3,
    );
    println!(
        "Cold-restart path:                          {N} fresh solves = {:.2} ms total",
        cold_total.as_secs_f64() * 1e3,
    );
    println!();
    println!(
        "Speedup of held-factor over cold-restart: {:.1}x",
        cold_total.as_secs_f64() / held_total.as_secs_f64()
    );
    println!("Numerical agreement (held vs cold dx): max |err| = {max_err:.2e}");
}
