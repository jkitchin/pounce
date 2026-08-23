//! The local-infeasibility second-opinion ladder runs on the library path.
//!
//! The ladder re-solves along a different trajectory before shipping an
//! `Infeasible_Problem_Detected` verdict, and both its rungs are on by
//! default. It used to live in `crates/pounce-cli/src/main.rs`, so the CLI and
//! every in-process frontend could return *different verdicts* for the same
//! model — and a false local infeasibility is the dangerous direction for a
//! branch-and-bound driver, which silently prunes a node that may contain the
//! optimum.
//!
//! What is pinned here is the mechanism on this path: the re-solves happen,
//! the caller's option table is handed back untouched, and a verdict that
//! survives the ladder is still reported. Whether a rung ever *promotes* is a
//! property of the model, and no fixture in this repo currently exercises it
//! (see `false_local_infeasibility.rs` — gh#693 removed the case that did);
//! `second_opinion_tests` in pounce-algorithm covers the promotion logic
//! directly.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::pounce_algorithm;
use pounce_rs::prelude::*;
use pounce_rs::{ApplicationReturnStatus, IpoptApplication};

/// `min (x − 5)²  s.t.  x² + δ = 0` with `δ = 0.1` — no real point satisfies
/// the equality, so the IPM can only reach a stationary point of the
/// constraint violation and reports local infeasibility. This is the shape
/// gh#508 measured its console/`.sol` disagreement on, and the CLI carries it
/// as the `issue_508_infeasible_gap_*` fixtures.
///
/// `solves` counts how many times the application started a solve.
struct Infeasible {
    solves: Rc<RefCell<usize>>,
}

impl TNLP for Infeasible {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -10.0;
        b.x_u[0] = 10.0;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // A proxy for how much solving happened, not an exact solve count:
        // the restoration phase re-reads the starting point too. Only ever
        // compared between two runs of the same model, never asserted
        // absolutely.
        *self.solves.borrow_mut() += 1;
        sp.x[0] = 1.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 5.0) * (x[0] - 5.0))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 5.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + 0.1;
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * x.map(|x| x[0]).unwrap_or(0.0);
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor + 2.0 * lambda.map(|l| l[0]).unwrap_or(0.0);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Install the restoration phase, the way the CLI, the C interface and the
/// Python extension all do.
///
/// `pounce-rs` does not depend on `pounce-restoration`, so a bare
/// `IpoptApplication` has no restoration provider and a solve that needs one
/// stops at `Restoration_Failed` instead of reaching a local-infeasibility
/// verdict — which is what this ladder is about. Injecting it here keeps these
/// tests about the ladder rather than about that separate gap.
fn install_restoration(app: &mut IpoptApplication) {
    use pounce_algorithm::application::{
        algorithm_builder_from_option_list, default_backend_factory, feral_config_from_options,
    };
    use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
    use pounce_restoration::resto_inner_solver::{
        InnerBackendFactoryFactory, make_default_restoration_factory_provider,
    };

    let mint = |options: &pounce_common::options_list::OptionsList| {
        let feral_cfg = feral_config_from_options(options);
        let bff_mint = move || -> InnerBackendFactoryFactory {
            let feral_cfg = feral_cfg.clone();
            Box::new(move || default_backend_factory(feral_cfg.clone()))
        };
        make_default_restoration_factory_provider(
            RestoAlgorithmBuilder::new(),
            algorithm_builder_from_option_list(options),
            bff_mint,
        )
    };
    let provider = mint(app.options());
    app.set_restoration_factory_provider(provider);
    app.set_restoration_provider_mint(std::rc::Rc::new(mint));
}

fn app(extra: &str) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    if !extra.is_empty() {
        app.options_mut().read_from_str(extra, true).unwrap();
    }
    install_restoration(&mut app);
    app
}

fn solve(extra: &str) -> (ApplicationReturnStatus, usize, IpoptApplication) {
    let solves = Rc::new(RefCell::new(0usize));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Infeasible {
        solves: Rc::clone(&solves),
    }));
    let mut a = app(extra);
    let status = a.optimize_tnlp(tnlp);
    let n = *solves.borrow();
    (status, n, a)
}

/// With the ladder on (the default), a local-infeasibility verdict costs more
/// than one solve on the library path — the re-solves actually run here now,
/// not only under the CLI.
#[test]
fn the_ladder_re_solves_on_the_library_path() {
    let (status, solves_on, _) = solve("");
    assert_eq!(
        status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "x² + 0.1 = 0 has no real solution"
    );
    assert!(
        solves_on > 1,
        "the ladder must re-solve before shipping the verdict, got {solves_on} solve(s)"
    );
}

/// The two gate options are read on this path. They were registered
/// core-side — so every frontend parsed and accepted them — and read only by
/// the CLI, which is exactly the shape of silent no-op this audit was chasing.
#[test]
fn both_rungs_can_be_disabled_from_the_library() {
    let (off_status, off_work, off_app) =
        solve("feral_infeasibility_scaling_retry no\ninfeasibility_mu_strategy_retry no\n");
    assert_eq!(
        off_status,
        ApplicationReturnStatus::InfeasibleProblemDetected
    );
    assert!(
        !off_app.last_second_opinion_unpromoted(),
        "with both rungs off no ladder may run at all"
    );

    let (on_status, on_work, on_app) = solve("");
    assert_eq!(
        on_status,
        ApplicationReturnStatus::InfeasibleProblemDetected
    );
    assert!(on_app.last_second_opinion_unpromoted(), "the ladder ran");
    assert!(
        on_work > off_work,
        "the default must do strictly more solving than the disabled ladder \
         (on={on_work}, off={off_work})"
    );
}

/// A verdict that survives the ladder is still reported as infeasible, and the
/// caller's option table is handed back exactly as they left it — the rungs
/// must not leak `feral_scaling=mc64` or `mu_strategy=adaptive` into the
/// options the caller reads afterwards.
#[test]
fn an_unpromoted_ladder_restores_the_callers_options() {
    let (status, _, a) = solve("");
    assert_eq!(status, ApplicationReturnStatus::InfeasibleProblemDetected);
    assert!(
        a.last_second_opinion_unpromoted(),
        "the ladder ran and did not promote"
    );

    let (mu, mu_set) = a.options().get_string_value("mu_strategy", "").unwrap();
    assert!(
        !(mu_set && mu == "adaptive"),
        "rung 2's mu_strategy=adaptive leaked into the caller's options"
    );
    let scaling = a.options().get_string_value("feral_scaling", "").ok();
    assert!(
        !matches!(scaling, Some((ref v, true)) if v == "mc64"),
        "rung 1's feral_scaling=mc64 leaked into the caller's options: {scaling:?}"
    );
}

/// A feasible problem never enters the ladder, so it costs exactly one solve.
#[test]
fn a_feasible_problem_never_enters_the_ladder() {
    struct Feasible;
    impl TNLP for Feasible {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1,
                m: 0,
                nnz_jac_g: 0,
                nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -10.0;
            b.x_u[0] = 10.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 3.0;
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            Some((x[0] - 1.0) * (x[0] - 1.0))
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * (x[0] - 1.0);
            true
        }
        fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
            true
        }
        fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
            true
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            obj_factor: Number,
            _l: Option<&[Number]>,
            _nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = 2.0 * obj_factor,
            }
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
    }

    let mut a = app("");
    let status = a.optimize_tnlp(Rc::new(RefCell::new(Feasible)));
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        !a.last_second_opinion_unpromoted(),
        "a successful solve must not report a failed ladder"
    );
}
