//! Line-search options that were registered but never read (#551, #677).
//!
//! Every option exercised here already had its behaviour implemented in
//! the algorithm — the α-loop's reduction factor and its
//! accept-anyway escape hatch, the filter switching rule's `delta`, the
//! penalty acceptor's ν update and Armijo relaxation, the adaptive-μ
//! obj-constr filter's margin — and no read site, so setting any of
//! them did exactly nothing and said nothing about it.
//!
//! The read site is not the deliverable. #551 is explicit: "a read site
//! that parses a value and discards it is the same silent no-op this
//! whole line of work exists to kill, and it is indistinguishable from a
//! real fix by inspection." So each option is tested twice — that it
//! reaches the object that consumes it, and that the thing that object
//! decides actually moves when it is set. A builder-field assertion on
//! its own would pass against a field nothing downstream reads.
//!
//! Every default asserted below equals the value `upstream_options.rs`
//! registers, which is what makes the wiring trajectory-neutral: only a
//! solve that sets one of these options sees any change.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

// ------------------------------------------------------------- test models

/// Rosenbrock under a quadratic constraint, started outside it:
///
/// ```text
/// min  100 (y - x^2)^2 + (1 - x)^2
/// s.t. x^2 + y^2 <= 2   (== 2 when `equality`),  -5 <= x, y <= 5
/// start (x, y) = (-1.2, 1), where g = 2.44 > 2
/// ```
///
/// HS071 — the crate's usual end-to-end model — cannot see a line-search
/// knob at all: it accepts the full Newton step every iteration, so
/// nothing ever backtracks, and it starts feasible, so the switching
/// rule and the filter margins never bind. This one backtracks and
/// starts infeasible, which is what makes the α-loop observable.
#[derive(Default)]
struct ConstrainedRosenbrock {
    equality: bool,
}

impl TNLP for ConstrainedRosenbrock {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-5.0, -5.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        b.g_l
            .copy_from_slice(&[if self.equality { 2.0 } else { -2.0e19 }]);
        b.g_u.copy_from_slice(&[2.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-1.2, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let t = x[1] - x[0] * x[0];
        Some(100.0 * t * t + (1.0 - x[0]) * (1.0 - x[0]))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let t = x[1] - x[0] * x[0];
        g[0] = -400.0 * x[0] * t - 2.0 * (1.0 - x[0]);
        g[1] = 200.0 * t;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
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
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda")[0];
                values[0] = obj_factor * (1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0) + 2.0 * lam;
                values[1] = obj_factor * (-400.0 * x[0]);
                values[2] = obj_factor * 200.0 + 2.0 * lam;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `min -(x + y)  s.t.  x^2 + y^2 = 1`, started at `(2, 2)`.
///
/// The start is infeasible (θ = 7) and the objective pulls *away* from
/// the feasible circle, so the step has `∇φᵀδ > 0`. That is the only
/// regime in which the penalty acceptor's ν update fires at all
/// (`ν ← ν⁺ + nu_inc` only when `ν < ν⁺ = (∇φᵀδ + ½δᵀWδ) / ((1-ρ)θ)`),
/// which is what makes `rho` observable.
#[derive(Default)]
struct PullAway;

impl TNLP for PullAway {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-5.0, -5.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        b.g_l.copy_from_slice(&[1.0]);
        b.g_u.copy_from_slice(&[1.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[2.0, 2.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(-(x[0] + x[1]))
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = -1.0;
        g[1] = -1.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda")[0];
                values[0] = 2.0 * lam;
                values[1] = 0.0;
                values[2] = 2.0 * lam;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

// ---------------------------------------------------------------- helpers

fn builder_from(
    setup: impl FnOnce(&mut IpoptApplication),
) -> pounce_algorithm::alg_builder::AlgorithmBuilder {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.algorithm_builder_from_options()
}

/// Solve `model` under `setup`; report the iteration count and check the
/// answer still stands. Trajectory tests compare the counts; the
/// objective assertion is what keeps "it moved" from meaning "it broke".
fn solve(
    label: &str,
    model: Rc<RefCell<dyn TNLP>>,
    expected_obj: Number,
    setup: impl FnOnce(&mut IpoptApplication),
) -> i32 {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    setup(&mut app);
    app.initialize().unwrap();
    let status = app.optimize_tnlp(model);
    let stats = app.statistics();
    eprintln!(
        "[{label}]: status={status:?} iter={} obj={}",
        stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "[{label}] did not solve: {status:?}",
    );
    assert!(
        (stats.final_objective - expected_obj).abs() < 1e-5,
        "[{label}]: objective {} drifted off {expected_obj}",
        stats.final_objective,
    );
    stats.iteration_count
}

fn rosen() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(ConstrainedRosenbrock::default())) as _
}

fn rosen_eq() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(ConstrainedRosenbrock { equality: true })) as _
}

fn pull_away() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(PullAway)) as _
}

/// Objective at the optimum of each model, to five decimals.
const ROSEN_OBJ: Number = 0.0;
const ROSEN_EQ_OBJ: Number = 3.99554730;
const PULL_OBJ: Number = -1.41421356;

// ----------------------------------------------------- reaching the builder

/// The registered defaults and the struct defaults must agree, or wiring
/// the option would silently move every solve that never mentions it.
/// The values on the right are the ones `upstream_options.rs` registers.
#[test]
fn line_search_defaults_match_the_registry() {
    let b = builder_from(|_| {});
    assert_eq!(b.line_search.alpha_red_factor, 0.5);
    assert_eq!(b.line_search.accept_after_max_steps, -1);
    assert_eq!(b.line_search.delta, 1.0);
    assert_eq!(b.line_search.nu_init, 1e-6);
    assert_eq!(b.line_search.nu_inc, 1e-4);
    assert_eq!(b.line_search.rho, 0.1);
    assert_eq!(b.line_search.eta_penalty, 1e-8);
    assert_eq!(b.mu.filter_margin_fact, 1e-5);
    assert_eq!(b.mu.filter_max_margin, 1.0);
}

#[test]
fn overrides_reach_the_builder() {
    let b = builder_from(|app| {
        let o = app.options_mut();
        for (k, v) in [
            ("alpha_red_factor", 0.25),
            ("delta", 3.5),
            ("nu_init", 2e-5),
            ("nu_inc", 7e-3),
            ("rho", 0.4),
            ("eta_penalty", 3e-7),
            ("filter_margin_fact", 2e-4),
            ("filter_max_margin", 0.75),
        ] {
            o.set_numeric_value(k, v, true, false).unwrap();
        }
        o.set_integer_value("accept_after_max_steps", 4, true, false)
            .unwrap();
    });
    assert_eq!(b.line_search.alpha_red_factor, 0.25);
    assert_eq!(b.line_search.accept_after_max_steps, 4);
    assert_eq!(b.line_search.delta, 3.5);
    assert_eq!(b.line_search.nu_init, 2e-5);
    assert_eq!(b.line_search.nu_inc, 7e-3);
    assert_eq!(b.line_search.rho, 0.4);
    assert_eq!(b.line_search.eta_penalty, 3e-7);
    assert_eq!(b.mu.filter_margin_fact, 2e-4);
    assert_eq!(b.mu.filter_max_margin, 0.75);
}

/// One hop past the builder: the values must be on the assembled
/// objects, not just on the options struct that feeds them.
#[test]
fn overrides_reach_the_assembled_line_search() {
    let bundle = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("alpha_red_factor", 0.25, true, false)
            .unwrap();
        app.options_mut()
            .set_integer_value("accept_after_max_steps", 4, true, false)
            .unwrap();
    })
    .build();
    assert_eq!(bundle.line_search.alpha_red_factor, 0.25);
    assert_eq!(bundle.line_search.accept_after_max_steps, 4);

    let default_bundle = builder_from(|_| {}).build();
    assert_eq!(default_bundle.line_search.alpha_red_factor, 0.5);
    assert_eq!(default_bundle.line_search.accept_after_max_steps, -1);
}

/// The four penalty constants, read back off the acceptor the builder
/// actually assembled. `nu_inc` has no other end-to-end witness: its
/// arithmetic is pinned by `penalty_acceptor.rs`'s `update_nu` unit
/// tests (ν ← ν⁺ + nu_inc), but on every model tried here the solve is
/// insensitive to the size of that increment, so this is as far as the
/// increment can be followed.
#[test]
fn the_assembled_penalty_acceptor_carries_its_four_constants() {
    let default_bundle = builder_from(|app| {
        app.options_mut()
            .set_string_value("line_search_method", "penalty", true, false)
            .unwrap();
    })
    .build();
    assert_eq!(
        default_bundle.line_search.acceptor().penalty_parameters(),
        Some((1e-6, 1e-4, 0.1, 1e-8)),
        "penalty acceptor defaults must equal the registered defaults",
    );

    let tuned = builder_from(|app| {
        app.options_mut()
            .set_string_value("line_search_method", "penalty", true, false)
            .unwrap();
        for (k, v) in [
            ("nu_init", 2e-5),
            ("nu_inc", 7e-3),
            ("rho", 0.4),
            ("eta_penalty", 3e-7),
        ] {
            app.options_mut()
                .set_numeric_value(k, v, true, false)
                .unwrap();
        }
    })
    .build();
    assert_eq!(
        tuned.line_search.acceptor().penalty_parameters(),
        Some((2e-5, 7e-3, 0.4, 3e-7)),
    );

    // The filter acceptor has no penalty parameter, so the accessor must
    // not quietly report the penalty defaults for it.
    assert_eq!(
        builder_from(|_| {})
            .build()
            .line_search
            .acceptor()
            .penalty_parameters(),
        None,
    );
}

// -------------------------------------------------------- moving a solve

#[test]
fn alpha_red_factor_changes_the_trajectory() {
    let base = solve("alpha_red_factor default", rosen(), ROSEN_OBJ, |_| {});
    let slow = solve("alpha_red_factor=0.1", rosen(), ROSEN_OBJ, |app| {
        app.options_mut()
            .set_numeric_value("alpha_red_factor", 0.1, true, false)
            .unwrap();
    });
    assert_ne!(
        base, slow,
        "backtracking with a 0.1 reduction factor took the same {base} iterations \
         as the default 0.5 — the option is parsed but not reaching the α-loop",
    );
}

/// `accept_after_max_steps` is the α-loop's escape hatch: after this
/// many backtracking steps the trial point is taken whatever the
/// acceptor says. `0` makes every first trial an accept.
#[test]
fn accept_after_max_steps_changes_the_trajectory() {
    let base = solve(
        "accept_after_max_steps default (-1)",
        rosen(),
        ROSEN_OBJ,
        |_| {},
    );
    let forced = solve("accept_after_max_steps=0", rosen(), ROSEN_OBJ, |app| {
        app.options_mut()
            .set_integer_value("accept_after_max_steps", 0, true, false)
            .unwrap();
    });
    assert_ne!(
        base, forced,
        "accepting the first trial of every line search took the same {base} \
         iterations as consulting the filter — the option is not reaching the α-loop",
    );
}

/// `delta` scales the constraint violation in the switching rule
/// (Eqn. (19)): `α (-d_φ)^s_φ > delta · θ^s_θ` decides whether a trial
/// is judged by the Armijo condition or by the filter's
/// sufficient-progress test. The two verdicts differ on the pair below,
/// so the option flips a real acceptance decision on the acceptor the
/// builder assembled.
///
/// This is a decision the corpus never puts on a knife edge — every
/// model tried solves identically for `delta` anywhere in
/// `[1e-12, 1e30]` — so the acceptor is driven directly rather than
/// through a solve.
#[test]
fn delta_flips_the_filter_acceptor_verdict() {
    use pounce_algorithm::line_search::filter_acceptor::AcceptDecision;

    // θ = 1e-6 ≤ θ_min (1e-4), d_φ = -1, α = 1, and a trial that halves
    // θ while leaving φ exactly where it was. Armijo needs
    // φ_trial ≤ φ - 1e-8, which this trial fails; the filter's
    // sufficient-progress test passes it on the θ decrease.
    let (alpha, theta, phi, d_phi) = (1.0, 1e-6, 10.0, -1.0);
    let (theta_trial, phi_trial) = (5e-7, 10.0);

    let mut default_bundle = builder_from(|_| {}).build();
    let with_default_delta = default_bundle.line_search.acceptor_mut().check_trial_point(
        alpha,
        theta,
        phi,
        d_phi,
        theta_trial,
        phi_trial,
    );

    let mut huge = builder_from(|app| {
        app.options_mut()
            .set_numeric_value("delta", 1e10, true, false)
            .unwrap();
    })
    .build();
    let with_huge_delta = huge.line_search.acceptor_mut().check_trial_point(
        alpha,
        theta,
        phi,
        d_phi,
        theta_trial,
        phi_trial,
    );

    assert_eq!(
        with_default_delta,
        AcceptDecision::Reject,
        "at delta = 1 the switching rule holds, so the trial owes Armijo a \
         decrease it does not have",
    );
    assert_eq!(
        with_huge_delta,
        AcceptDecision::Accept,
        "at delta = 1e10 the switching rule fails, so the same trial is judged \
         by sufficient progress and passes — if this still rejects, `delta` is \
         parsed and dropped before it reaches the acceptor",
    );
}

#[test]
fn nu_init_changes_the_penalty_trajectory() {
    let penalty = |app: &mut IpoptApplication| {
        app.options_mut()
            .set_string_value("line_search_method", "penalty", true, false)
            .unwrap();
    };
    let base = solve("penalty nu_init default", rosen(), ROSEN_OBJ, penalty);
    let big = solve("penalty nu_init=1e3", rosen(), ROSEN_OBJ, |app| {
        penalty(app);
        app.options_mut()
            .set_numeric_value("nu_init", 1e3, true, false)
            .unwrap();
    });
    assert_ne!(
        base, big,
        "starting the penalty parameter nine orders higher took the same {base} \
         iterations — `nu_init` is not reaching the penalty acceptor",
    );
}

#[test]
fn rho_changes_the_penalty_trajectory() {
    let penalty = |app: &mut IpoptApplication| {
        app.options_mut()
            .set_string_value("line_search_method", "penalty", true, false)
            .unwrap();
    };
    // `rho` only appears in the ν update, which only fires when the step
    // raises the barrier objective while θ > 0 — the pull-away regime.
    let base = solve("penalty rho default", pull_away(), PULL_OBJ, penalty);
    let small = solve("penalty rho=1e-3", pull_away(), PULL_OBJ, |app| {
        penalty(app);
        app.options_mut()
            .set_numeric_value("rho", 1e-3, true, false)
            .unwrap();
    });
    assert_ne!(
        base, small,
        "the ν update ran identically at rho = 0.1 and rho = 1e-3 ({base} \
         iterations either way) — `rho` is not reaching the penalty acceptor",
    );
}

#[test]
fn eta_penalty_changes_the_penalty_trajectory() {
    let penalty = |app: &mut IpoptApplication| {
        app.options_mut()
            .set_string_value("line_search_method", "penalty", true, false)
            .unwrap();
    };
    let base = solve("penalty eta_penalty default", rosen(), ROSEN_OBJ, penalty);
    let strict = solve("penalty eta_penalty=0.4", rosen(), ROSEN_OBJ, |app| {
        penalty(app);
        app.options_mut()
            .set_numeric_value("eta_penalty", 0.4, true, false)
            .unwrap();
    });
    assert_ne!(
        base, strict,
        "demanding 40% of the predicted reduction instead of 1e-8 of it took the \
         same {base} iterations — `eta_penalty` is not reaching the Armijo test",
    );
}

/// `filter_margin_fact` and `filter_max_margin` set the margin an entry
/// must clear in the adaptive-μ `obj-constr-filter` globalization test
/// (`margin = filter_margin_fact · min(filter_max_margin, err)`), so
/// both need `mu_strategy=adaptive` to bite.
#[test]
fn filter_margin_fact_changes_the_adaptive_mu_trajectory() {
    let adaptive = |app: &mut IpoptApplication| {
        app.options_mut()
            .set_string_value("mu_strategy", "adaptive", true, false)
            .unwrap();
    };
    let base = solve("adaptive margin default", rosen(), ROSEN_OBJ, adaptive);
    let wide = solve(
        "adaptive filter_margin_fact=0.9",
        rosen(),
        ROSEN_OBJ,
        |app| {
            adaptive(app);
            app.options_mut()
                .set_numeric_value("filter_margin_fact", 0.9, true, false)
                .unwrap();
        },
    );
    assert_ne!(
        base, wide,
        "a margin factor five orders wider left the adaptive-μ globalization \
         unchanged at {base} iterations — `filter_margin_fact` is not reaching \
         `AdaptiveMuUpdate`",
    );
}

#[test]
fn filter_max_margin_changes_the_adaptive_mu_trajectory() {
    // The cap only shows with a margin factor large enough for the cap
    // to be the binding half of the `min`.
    let wide_margin = |app: &mut IpoptApplication| {
        app.options_mut()
            .set_string_value("mu_strategy", "adaptive", true, false)
            .unwrap();
        app.options_mut()
            .set_numeric_value("filter_margin_fact", 0.9, true, false)
            .unwrap();
    };
    let uncapped = solve(
        "adaptive filter_max_margin default",
        rosen_eq(),
        ROSEN_EQ_OBJ,
        wide_margin,
    );
    let capped = solve(
        "adaptive filter_max_margin=1e-12",
        rosen_eq(),
        ROSEN_EQ_OBJ,
        |app| {
            wide_margin(app);
            app.options_mut()
                .set_numeric_value("filter_max_margin", 1e-12, true, false)
                .unwrap();
        },
    );
    assert_ne!(
        uncapped, capped,
        "capping the margin at 1e-12 left the trajectory at {uncapped} iterations \
         — `filter_max_margin` is not reaching `AdaptiveMuUpdate`",
    );
}
