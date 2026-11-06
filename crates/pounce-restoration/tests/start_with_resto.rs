//! `start_with_resto` forces the feasibility restoration phase in the
//! first iteration.
//!
//! It is an **outer**-loop behaviour, and that is what went wrong. The
//! option was threaded from the `OptionsList` into
//! `AlgorithmBuilder::resto`, from there into `RestoAlgorithmBuilder`,
//! and from there into `MinC1NrmDriver` — a field on the *inner*
//! restoration solver, which has no first outer iteration to force. Every
//! layer set it and no layer read it, so `start_with_resto yes` was a
//! silent no-op.
//!
//! `unimplemented_options.rs`'s `the_restoration_switches_reach_the_builder`
//! could not see it: it asserts that the value *reaches the builder*,
//! which is exactly the "read site populating a field nobody consumes
//! would be a fresh silent no-op" that its own comment names as the
//! defect to avoid. Reaching a field is not being read.
//!
//! The fixture below is deliberately one that **never enters restoration
//! on its own**, which is what lets a non-zero call count be attributed
//! to the option and nothing else. `resto_fd_hessian.rs`'s fixture is the
//! opposite by construction and cannot be reused here.

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory, feral_config_from_options,
};
use pounce_common::types::Number;
use pounce_nlp::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::rc::Rc;

/// min Σ(xᵢ − 3)² + exp(x₀x₁)
/// s.t.  x₀² + x₁² = 1,  exp(x₂) + x₃³ = 0.5,  x₀x₁x₂ = 2
///
/// Well behaved from the start below: it converges without ever calling
/// the restoration phase, which is the property this file needs.
#[derive(Default)]
struct Smooth;

impl TNLP for Smooth {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 12,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e20; 4]);
        b.x_u.copy_from_slice(&[1e20; 4]);
        b.g_l.copy_from_slice(&[0.0; 3]);
        b.g_u.copy_from_slice(&[0.0; 3]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-5.0, 8.0, -4.0, 6.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let s: Number = x.iter().map(|v| (v - 3.0) * (v - 3.0)).sum();
        Some(s + (x[0] * x[1]).exp())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let e = (x[0] * x[1]).exp();
        for i in 0..4 {
            g[i] = 2.0 * (x[i] - 3.0);
        }
        g[0] += e * x[1];
        g[1] += e * x[0];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
        g[1] = x[2].exp() + x[3] * x[3] * x[3] - 0.5;
        g[2] = x[0] * x[1] * x[2] - 2.0;
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
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
                values[2] = 0.0;
                values[3] = 0.0;
                values[4] = 0.0;
                values[5] = 0.0;
                values[6] = x[2].exp();
                values[7] = 3.0 * x[3] * x[3];
                values[8] = x[1] * x[2];
                values[9] = x[0] * x[2];
                values[10] = x[0] * x[1];
                values[11] = 0.0;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        false
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve(start_with_resto: bool) -> (ApplicationReturnStatus, i32, i32) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 500, true, true)
        .unwrap();
    if start_with_resto {
        app.options_mut()
            .set_string_value("start_with_resto", "yes", true, true)
            .unwrap();
    }
    app.initialize().unwrap();

    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone(), Ma57Config::default()))
    };
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(resto_provider);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Smooth));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    (status, stats.restoration_calls, stats.iteration_count)
}

#[test]
fn start_with_resto_forces_the_first_iteration_into_restoration() {
    let (off_status, off_calls, off_iters) = solve(false);
    let (on_status, on_calls, on_iters) = solve(true);

    // The attribution. If this fixture ever starts entering restoration on
    // its own, the assertion below stops being about the option.
    assert_eq!(
        off_calls, 0,
        "the fixture reached restoration without the option, so a non-zero \
         count with it proves nothing"
    );

    assert!(
        on_calls > 0,
        "start_with_resto=yes did not enter restoration (calls = {on_calls}). \
         The option reached AlgorithmBuilder::resto and RestoAlgorithmBuilder \
         and MinC1NrmDriver, and no read site acted on it."
    );

    // Both still solve: forcing restoration changes the path, not the answer.
    assert_eq!(off_status, ApplicationReturnStatus::SolveSucceeded);
    assert_eq!(
        on_status,
        ApplicationReturnStatus::SolveSucceeded,
        "forcing restoration broke an otherwise solvable model"
    );
    assert_ne!(
        off_iters, on_iters,
        "the trajectories are identical, which means the forced restoration \
         did not actually change the path"
    );
}

/// The restoration sub-solve runs under the solve's own `max_iter`, not the
/// one the restoration provider was built with. The provider is built here
/// at `max_iter = 500` and the cap then lowered to 1, as the complementarity
/// driver budgets each stage on one application. Forced into restoration at
/// iteration 0, the solve stops at the cap: before the forwarding
/// (`RestorationPhase::set_solve_max_iter`), restoration ran on under the
/// provider's 500, and MacMPEC `ralph1`'s 100-iteration finish ran 3000.
#[test]
fn restoration_runs_under_the_solves_max_iter_not_the_providers() {
    let mut app = IpoptApplication::new();
    for (k, v) in [
        ("hessian_approximation", "limited-memory"),
        ("start_with_resto", "yes"),
    ] {
        app.options_mut()
            .set_string_value(k, v, true, true)
            .unwrap();
    }
    app.options_mut()
        .set_integer_value("print_level", 0, true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("max_iter", 500, true, true)
        .unwrap();
    app.initialize().unwrap();
    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone(), Ma57Config::default()))
    };
    app.set_restoration_factory_provider(make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    ));
    // Lowered after the provider took its snapshot.
    app.options_mut()
        .set_integer_value("max_iter", 1, true, true)
        .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Smooth));
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    assert!(
        stats.restoration_calls > 0,
        "the fixture must enter restoration"
    );
    assert!(
        stats.iteration_count <= 1 && stats.restoration_inner_iters <= 1,
        "restoration ran past the solve's max_iter: {} iterations, {} in restoration ({status:?})",
        stats.iteration_count,
        stats.restoration_inner_iters
    );
}
