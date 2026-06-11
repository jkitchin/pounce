//! End-to-end smoke test: solve HS071 through `IpoptApplication::optimize_tnlp`.
//!
//! HS071 (Hock & Schittkowski, 1981, problem 71):
//!
//! ```text
//! min  x1*x4*(x1 + x2 + x3) + x3
//! s.t. x1*x2*x3*x4 >= 25
//!      x1^2 + x2^2 + x3^2 + x4^2 == 40
//!      1 <= xi <= 5
//! ```
//!
//! Known optimum: f* = 17.0140173 at x* = (1, 4.7429996, 3.8211499, 1.3794082).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Hs071 {
    final_x: Option<[Number; 4]>,
    final_obj: Option<Number>,
    final_lambda: Option<Vec<Number>>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
}

impl TNLP for Hs071 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 8,
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[1.0; 4]);
        b.x_u.copy_from_slice(&[5.0; 4]);
        // g0: x1*x2*x3*x4 >= 25  (inequality, finite lower only)
        // g1: sum xi^2 == 40     (equality)
        b.g_l.copy_from_slice(&[25.0, 40.0]);
        b.g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
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
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = x[1] * x[2] * x[3];
                values[1] = x[0] * x[2] * x[3];
                values[2] = x[0] * x[1] * x[3];
                values[3] = x[0] * x[1] * x[2];
                values[4] = 2.0 * x[0];
                values[5] = 2.0 * x[1];
                values[6] = 2.0 * x[2];
                values[7] = 2.0 * x[3];
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
                irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda");
                let of = obj_factor;
                let l0 = lam[0];
                let l1 = lam[1];
                values[0] = of * (2.0 * x[3]) + l1 * 2.0;
                values[1] = of * x[3] + l0 * (x[2] * x[3]);
                values[2] = l1 * 2.0;
                values[3] = of * x[3] + l0 * (x[1] * x[3]);
                values[4] = l0 * (x[0] * x[3]);
                values[5] = l1 * 2.0;
                values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]);
                values[7] = of * x[0] + l0 * (x[0] * x[2]);
                values[8] = of * x[0] + l0 * (x[0] * x[1]);
                values[9] = l1 * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        if sol.x.len() == 4 {
            self.final_x = Some([sol.x[0], sol.x[1], sol.x[2], sol.x[3]]);
        }
        self.final_obj = Some(sol.obj_value);
        self.final_lambda = Some(sol.lambda.to_vec());
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
    }
}

#[test]
fn hs071_solves_via_application() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);

    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let stats = app.statistics();
    eprintln!(
        "HS71: status={:?} iter={} obj={} wall_s={:.3}",
        status, stats.iteration_count, stats.final_objective, stats.total_wallclock_time_secs,
    );
    assert!(
        stats.iteration_count < 50,
        "iter_count = {} (expected < 50)",
        stats.iteration_count,
    );

    // Restoration audit counters (pounce#12) — HS071 is a well-behaved
    // problem that does not enter restoration. All counters must be 0.
    assert_eq!(
        stats.restoration_calls, 0,
        "HS071 unexpectedly entered restoration {} time(s)",
        stats.restoration_calls,
    );
    assert_eq!(stats.restoration_outer_iters, 0);
    assert_eq!(stats.restoration_inner_iters, 0);
    assert!(
        stats.restoration_wall_secs.abs() < 1e-9,
        "wall_secs should be 0.0, got {}",
        stats.restoration_wall_secs,
    );

    // Final objective at optimum is 17.0140173.
    let obj = stats.final_objective;
    assert!(
        (obj - 17.014017).abs() < 1e-4,
        "final_objective = {obj} (expected ~17.014017)",
    );

    // The user's TNLP::finalize_solution should also have been called
    // with the same objective value.
    let user = tnlp_concrete.borrow();
    assert!(user.final_obj.is_some(), "finalize_solution was not called");
    let f_user = user.final_obj.unwrap();
    assert!(
        (f_user - 17.014017).abs() < 1e-4,
        "user-side final_obj = {f_user} (expected ~17.014017)",
    );
}

#[test]
fn hs071_solves_with_penalty_line_search() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("line_search_method", "penalty", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 penalty: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-4,
        "final_objective = {} (expected ~17.014017)",
        stats.final_objective,
    );
}

#[test]
fn hs071_solves_with_adaptive_mu_loqo_oracle() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("mu_oracle", "loqo", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 adaptive+loqo: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
}

#[test]
fn hs071_solves_with_adaptive_mu_quality_function_oracle() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("mu_oracle", "quality-function", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let stats = app.statistics();
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-4,
        "final_objective = {} (expected ~17.014017)",
        stats.final_objective,
    );
}

/// Setting `dual_inf_tol` and `compl_inf_tol` impossibly small must
/// block the scaled-error convergence path. Baseline HS71 reports
/// `SolveSucceeded` in single-digit iters; with the per-component
/// gate set to 1e-20 the solver can't claim success on the same
/// iterate, so it exits through the tiny-step / max-iter side
/// instead. Demonstrates that the per-component gate from
/// `OptimalityErrorConvergenceCheck::CheckConvergence` is wired
/// through end-to-end.
#[test]
fn hs071_dual_inf_tol_blocks_convergence() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("dual_inf_tol", 1e-20, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("compl_inf_tol", 1e-20, true, false)
        .unwrap();
    // Disable the acceptable-streak fallback so the only way out is
    // SearchDirectionBecomesTooSmall / MaximumIterationsExceeded.
    app.options_mut()
        .set_integer_value("acceptable_iter", 9999, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 25, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 strict dual/compl: status={:?} iter={}",
        status, stats.iteration_count,
    );
    assert!(
        !matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "per-component gate ignored: status = {status:?}",
    );
}

/// Tightening `tol` past machine precision and pairing it with a
/// generous `acceptable_tol = 1e-4` plus `acceptable_iter = 1` must
/// route HS71 through `SolvedToAcceptableLevel`. Demonstrates that
/// the `acceptable_*` triplet is wired through `OptionsList`.
#[test]
fn hs071_acceptable_tol_triggers_acceptable_level_status() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("tol", 1e-30, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("acceptable_tol", 1e-4, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("acceptable_iter", 1, true, false)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 25, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 acceptable: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    // The streak fires inside the conv-check (returns Converged), so
    // the application maps it to SolveSucceeded — same return code
    // upstream uses when the acceptable streak supplies the
    // termination. The pounce-side `SolvedToAcceptableLevel` status
    // is reserved for the restore-acceptable-on-restoration-failure
    // path. Either way, the run must succeed without hitting
    // max_iter.
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        stats.iteration_count < 25,
        "iter_count = {} (expected < 25)",
        stats.iteration_count,
    );
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-3,
        "final_objective = {}",
        stats.final_objective,
    );
}

/// `max_wall_time` set to effectively zero must short-circuit the
/// solver before the first real iterate test. Maps to
/// `MaximumWallTimeExceeded` per upstream.
#[test]
fn hs071_max_wall_time_terminates() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("max_wall_time", 1e-12, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 wall budget: status={:?} iter={} wall_s={:.6}",
        status, stats.iteration_count, stats.total_wallclock_time_secs,
    );
    assert!(
        matches!(status, ApplicationReturnStatus::MaximumWallTimeExceeded),
        "unexpected status: {status:?}",
    );
}

/// `max_cpu_time` analogue.
#[test]
fn hs071_max_cpu_time_terminates() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("max_cpu_time", 1e-12, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71 cpu budget: status={:?} iter={}",
        status, stats.iteration_count,
    );
    assert!(
        matches!(status, ApplicationReturnStatus::MaximumCpuTimeExceeded),
        "unexpected status: {status:?}",
    );
}

#[test]
fn hs071_solves_with_adaptive_mu_probing_oracle() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("mu_oracle", "probing", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let stats = app.statistics();
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-4,
        "final_objective = {} (expected ~17.014017)",
        stats.final_objective,
    );
}

/// `mu_init` flowing through the builder reaches the monotone
/// updater. Solve still converges to the HS71 optimum.
#[test]
fn hs071_solves_with_nondefault_mu_init() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_numeric_value("mu_init", 1e-3, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let stats = app.statistics();
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-4,
        "final_objective = {} (expected ~17.014017)",
        stats.final_objective,
    );
}

/// Regression test for pounce#11. Before the fix,
/// `Solution.lambda` / `z_l` / `z_u` were forwarded to the user as
/// all-zero stubs (see the comment at `application.rs` ≈ line 640).
/// After the fix `OrigIpoptNlp::finalize_solution_{lambda,z_l,z_u}`
/// lift the algorithm-side compressed multipliers back to the user's
/// constraint-row / full-x indexing.
///
/// HS071 has known multipliers at `x* = (1, 4.7430, 3.8211, 1.3794)`,
/// `f* = 17.014`. x[0] sits on its lower bound; the other three are
/// interior. lambda[0] (inequality `x0·x1·x2·x3 ≥ 25`, active) and
/// lambda[1] (equality `Σxᵢ² = 40`) are both non-zero. Sign matches
/// upstream Ipopt's `min f + λ·g(x)`.
///
/// Exact reference values vary with options / linear backend / scaling.
/// We assert (a) multipliers are populated (non-zero, not the all-zero
/// stub from pre-#11), and (b) the KKT residual recomposed from the
/// user-visible quantities is small.
#[test]
fn hs071_reports_nonzero_multipliers_at_optimum() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let user = tnlp_concrete.borrow();
    let lambda = user.final_lambda.as_ref().expect("lambda populated");
    let z_l = user.final_z_l.as_ref().expect("z_l populated");
    let z_u = user.final_z_u.as_ref().expect("z_u populated");

    assert_eq!(lambda.len(), 2);
    assert_eq!(z_l.len(), 4);
    assert_eq!(z_u.len(), 4);

    let lam_max = lambda.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        lam_max > 1e-3,
        "lambda must be non-zero at HS071 optimum; got {:?}",
        lambda
    );

    // KKT stationarity recomposed from user-visible quantities:
    //   ∇f + J_g^T λ − z_l + z_u ≈ 0  at the optimum.
    let x = user.final_x.unwrap();
    let mut grad_f = [0.0_f64; 4];
    grad_f[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
    grad_f[1] = x[0] * x[3];
    grad_f[2] = x[0] * x[3] + 1.0;
    grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
    let l0 = lambda[0];
    let l1 = lambda[1];
    let mut kkt = [0.0_f64; 4];
    kkt[0] = grad_f[0] + l0 * (x[1] * x[2] * x[3]) + l1 * (2.0 * x[0]) - z_l[0] + z_u[0];
    kkt[1] = grad_f[1] + l0 * (x[0] * x[2] * x[3]) + l1 * (2.0 * x[1]) - z_l[1] + z_u[1];
    kkt[2] = grad_f[2] + l0 * (x[0] * x[1] * x[3]) + l1 * (2.0 * x[2]) - z_l[2] + z_u[2];
    kkt[3] = grad_f[3] + l0 * (x[0] * x[1] * x[2]) + l1 * (2.0 * x[3]) - z_l[3] + z_u[3];
    let kkt_inf = kkt.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    assert!(
        kkt_inf < 1e-3,
        "KKT residual with lifted multipliers = {:.2e} (lambda={:?}, z_l={:?}, z_u={:?})",
        kkt_inf,
        lambda,
        z_l,
        z_u,
    );

    eprintln!(
        "HS071 multipliers — lambda = {:?}, z_l = {:?}, z_u = {:?}",
        lambda, z_l, z_u,
    );
}

/// `print_info_string yes` + `inf_pr_output = internal` flow through
/// `algorithm_builder_from_options` into `OrigIterationOutput`. The
/// solve must still converge; this is a smoke test that the option
/// wiring doesn't break the iter-row format.
#[test]
fn hs071_solves_with_iter_diagnostic_options() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_bool_value("print_info_string", true, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("inf_pr_output", "internal", true, false)
        .unwrap();
    app.options_mut()
        .set_bool_value("print_user_options", true, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
}

/// `print_frequency_iter = 100` makes pounce skip per-iter output
/// rows but the solve itself must still converge cleanly. Smokes the
/// option wiring without trying to assert against captured stdout.
#[test]
fn hs071_solves_with_sparse_iter_output() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_frequency_iter", 100, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
}

/// `output_file` opens a `FileJournal` under the configured name and
/// the timing report — gated on `print_timing_statistics yes` — is
/// fanned out to it. Per-iter rows are not yet routed through the
/// journalist (pounce writes them straight to stdout), so the test
/// asserts only that the timing-statistics block landed on disk.
#[test]
fn hs071_output_file_captures_timing_report() {
    // Lowercase path under `/tmp` so the upstream-style filename
    // handling (no case folding) and the pid-suffixed uniqueness both
    // work on case-sensitive filesystems.
    let path = std::path::PathBuf::from(format!(
        "/tmp/pounce-test-output-{}.log",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let mut app = IpoptApplication::new();
    let opts = format!(
        "output_file {}\nfile_print_level 5\nprint_timing_statistics yes\n",
        path.display(),
    );
    app.initialize_with_options_str(&opts).unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let _ = app.optimize_tnlp(tnlp);
    app.journalist().flush_buffer();

    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("expected log at {}", path.display()));
    assert!(
        contents.contains("Timing Statistics"),
        "output_file at {} missing timing-statistics block; got:\n{}",
        path.display(),
        contents,
    );
    let _ = std::fs::remove_file(&path);
}

/// Wave-4 smoke: the eight `warm_start_*` knobs and the
/// `warm_start_init_point` toggle parse through
/// `algorithm_builder_from_options` without erroring. End-to-end
/// behavior (the warm-started solve itself) isn't exercised here —
/// it needs a populated `data.curr` from a prior solve, which the
/// re-optimize workflow that drives this initializer doesn't yet
/// surface. The unit tests in `init::warm_start` cover the clamp /
/// target-mu semantics in isolation.
#[test]
fn warm_start_options_flow_through_builder() {
    let mut app = IpoptApplication::new();
    let opts = "\
        warm_start_init_point yes\n\
        warm_start_same_structure yes\n\
        warm_start_bound_push 1e-2\n\
        warm_start_bound_frac 1e-2\n\
        warm_start_slack_bound_push 1e-2\n\
        warm_start_slack_bound_frac 1e-2\n\
        warm_start_mult_bound_push 1e-2\n\
        warm_start_mult_init_max 1e3\n\
        warm_start_target_mu 1e-2\n\
        warm_start_entire_iterate yes\n\
    ";
    app.initialize_with_options_str(opts).unwrap();
    // Every key the test set must be visible on the OptionsList; this
    // smoke-tests the registry surface (e.g. proves
    // `warm_start_init_point` registered correctly).
    for key in [
        "warm_start_init_point",
        "warm_start_same_structure",
        "warm_start_entire_iterate",
    ] {
        let (_, found) = app.options().get_string_value(key, "").unwrap();
        assert!(found, "{key} did not parse through the registry");
    }
}
