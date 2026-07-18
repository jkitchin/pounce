//! End-to-end iteration capture through `pounce_rs`

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::prelude::*;

/// min (x0-1)^2 + (x1-2)^2  s.t. x0 + x1 == 3
struct Quad;
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
fn builder_solve_inside_with_iter_capture_records_trajectory() {
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    assert!(!iters.is_empty(), "no iteration records captured");
    assert_eq!(iters[0].iter, 0, "trajectory must start at iteration 0");
    assert!(
        iters.windows(2).all(|w| w[0].iter < w[1].iter),
        "iteration counter must be strictly increasing"
    );
}

#[test]
fn builder_capture_iterations_flag_fills_stats() {
    let sol = Nlp::new(Quad)
        .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
        .constraint_bounds(&[3.0], &[3.0])
        .capture_iterations()
        .solve();
    assert!(sol.success, "status = {:?}", sol.status);
    assert!(!sol.stats.iterations.is_empty());
    assert!(sol.stats.iteration_count > 0);
    assert!(sol.stats.total_wallclock_time_secs > 0.0);
    assert_eq!(sol.g.len(), 1);
    assert_eq!(sol.z_l.len(), 2);
}

#[test]
fn with_iter_capture_around_capture_iterations_gets_records_too() {
    let (sol, iters) = with_iter_capture(|| {
        Nlp::new(Quad)
            .var_bounds(&[0.0, 0.0], &[5.0, 5.0])
            .constraint_bounds(&[3.0], &[3.0])
            .capture_iterations()
            .solve()
    });
    assert!(sol.success, "status = {:?}", sol.status);
    assert!(!sol.stats.iterations.is_empty());
    assert_eq!(
        iters.len(),
        sol.stats.iterations.len(),
        "outer capture must see the same trajectory the driver recorded"
    );
    assert_eq!(iters[0].iter, sol.stats.iterations[0].iter);
}

/// min (x0-1)^2 + (x1+2)^2, bounds only (m = 0), L-BFGS Hessian.
#[derive(Default)]
struct Bounded2 {
    solved: bool,
}
impl TNLP for Bounded2 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-5.0, -5.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }
    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some((x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, grad_f: &mut [f64]) -> bool {
        grad_f[0] = 2.0 * (x[0] - 1.0);
        grad_f[1] = 2.0 * (x[1] + 2.0);
        true
    }
    fn eval_g(&mut self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[f64]>, _new_x: bool, _mode: SparsityRequest<'_>) -> bool {
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.solved = true;
    }
}

#[test]
fn application_iter_history_fills_under_collector_scope() {
    let _scope = collector_scope();
    let mut app = IpoptApplication::new();
    let init = app.initialize();
    assert!(
        init.is_ok(),
        "IpoptApplication::initialize failed: {init:?}"
    );
    let _ =
        app.options_mut()
            .set_string_value("hessian_approximation", "limited-memory", true, true);
    app.enable_iter_history();

    let prob = Rc::new(RefCell::new(Bounded2::default()));
    let status = app.optimize_tnlp(Rc::clone(&prob) as Rc<RefCell<dyn TNLP>>);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(prob.borrow().solved);

    let stats = app.statistics();
    assert!(
        !stats.iterations.is_empty(),
        "enable_iter_history under collector_scope must fill statistics().iterations"
    );
    assert_eq!(stats.iterations[0].iter, 0);
}
