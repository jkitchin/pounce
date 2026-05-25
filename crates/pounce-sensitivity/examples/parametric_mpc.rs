//! Sliding-horizon parametric step — what MPC does between full
//! re-solves.
//!
//! MPC typically alternates: every K steps, do a full IPM solve of the
//! current horizon; between those, use a first-order parametric step
//! `Δx ≈ ∂x*/∂p · Δp` to update the primal as the parameter slides.
//! The parametric step is the cheap part because it's a back-solve
//! against the cached factor; the IPM solve is the expensive part.
//!
//! This example demonstrates the **parametric-step half** of that
//! loop: one IPM solve, followed by 10 parametric-step queries against
//! the held factor as the parameter sweeps. Each query is a single
//! back-solve, so the cost is essentially flat after the initial
//! solve.
//!
//! What's deliberately *not* shown here: the periodic re-solve with
//! symbolic-factor reuse across IPM runs. That's tracked by Phase 3b's
//! `BackendPool` + `resolve()` work in
//! `dev-notes/backend-pool-resolve.md`.
//!
//! Run with:
//!   `cargo run --release --example parametric_mpc -p pounce-sensitivity`

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

/// Same parametric NLP as `examples/sensitivity_session.rs`: a small
/// 5-variable, 4-constraint problem with two pin-constraint parameters
/// (`eta1`, `eta2`) we can sweep.
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

    // One full IPM solve at the nominal parameter.
    let mut solver = Solver::new(make_app(), tnlp);
    let t0 = Instant::now();
    let status = solver.solve();
    let solve_dt = t0.elapsed();
    println!(
        "IPM solve at nominal (eta1=5.0, eta2=1.0): status={status:?}, {:.3} ms",
        solve_dt.as_secs_f64() * 1e3
    );
    assert!(solver.converged().is_some());

    // 10 parametric steps as eta2 sweeps from 1.0 to 1.5. Each one is
    // a back-solve against the held factor — no IPM iteration.
    let pins = vec![2 as Index, 3];
    let mut total_step_dt = std::time::Duration::ZERO;
    let mut steps = Vec::new();
    for k in 1..=10 {
        let d_eta2 = 0.05 * k as f64;
        let t = Instant::now();
        let dx = solver
            .parametric_step(&pins, &[0.0, d_eta2])
            .expect("parametric_step ok");
        total_step_dt += t.elapsed();
        steps.push((d_eta2, dx));
    }

    println!("\nparametric steps against the held factor:");
    for (d_eta2, dx) in &steps {
        println!("  Δeta2 = {d_eta2:+.2}  -> Δx_primal = [{:+.5}, {:+.5}, {:+.5}]",
            dx[0], dx[1], dx[2]);
    }
    let avg_step_us = total_step_dt.as_secs_f64() * 1e6 / steps.len() as f64;
    println!(
        "\n10 parametric steps: total {:.3} ms, mean {avg_step_us:.1} µs/step",
        total_step_dt.as_secs_f64() * 1e3
    );
    println!(
        "\nRatio: each parametric step is roughly {:.0}x cheaper than the IPM solve.",
        solve_dt.as_secs_f64() / (total_step_dt.as_secs_f64() / steps.len() as f64),
    );
}
