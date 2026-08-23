//! Tests for the `solver_selection` and `qp_presolve` registered library options.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default, Clone)]
struct Sink {
    x: Vec<Number>,
}

fn converged(s: ApplicationReturnStatus) -> bool {
    matches!(
        s,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

// HS35
// f = 9 - 8x0 - 6x1 - 4x2 + 2x0^2 + 2x1^2 + x2^2 + 2x0x1 + 2x0x2,
// s.t. x0 + x1 + 2x2 <= 3, x >= 0.  x* = (4/3, 7/9, 4/9).
// A convex QP, so fit for `qp-active-set`.
struct Hs35 {
    sink: Rc<RefCell<Sink>>,
}
impl TNLP for Hs35 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 3,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0; 3]);
        b.x_u.copy_from_slice(&[2.0e19; 3]);
        b.g_l.copy_from_slice(&[-2.0e19]);
        b.g_u.copy_from_slice(&[3.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5, 0.5]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
                + 2.0 * x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[0] * x[1]
                + 2.0 * x[0] * x[2],
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
        g[1] = -6.0 + 4.0 * x[1] + 2.0 * x[0];
        g[2] = -4.0 + 2.0 * x[2] + 2.0 * x[0];
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + 2.0 * x[2];
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
                irow.copy_from_slice(&[0, 0, 0]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values, .. } => {
                values.copy_from_slice(&[1.0, 1.0, 2.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        of: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 0, 1, 0, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values, .. } => {
                values[0] = of * 4.0;
                values[1] = of * 2.0;
                values[2] = of * 4.0;
                values[3] = of * 2.0;
                values[4] = of * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.sink.borrow_mut() = Sink { x: sol.x.to_vec() };
    }
}

fn solve_hs35_with(options: &str) -> (IpoptApplication, ApplicationReturnStatus, Sink) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    if !options.is_empty() {
        app.initialize_with_options_str(options).unwrap();
    }
    let sink = Rc::new(RefCell::new(Sink::default()));
    let tnlp = Hs35 { sink: sink.clone() };
    let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let status = app.optimize_tnlp(rc);
    let out = sink.borrow().clone();
    (app, status, out)
}

#[test]
fn qp_active_set_selection_routes_to_sqp() {
    let (app, status, out) = solve_hs35_with("print_level 0\nsolver_selection qp-active-set\n");
    assert!(converged(status), "status = {status:?}");
    assert!(
        app.last_sqp_working_set().is_some(),
        "solver_selection=qp-active-set must run the SQP engine (a working set \
         should have been recorded); the interior-point path never sets one"
    );
    let x_star = [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0];
    for i in 0..3 {
        assert!(
            (out.x[i] - x_star[i]).abs() < 1e-3,
            "x[{i}] = {} (expected {})",
            out.x[i],
            x_star[i]
        );
    }
}

#[test]
fn default_selection_does_not_run_sqp() {
    let (app, status, _out) = solve_hs35_with("print_level 0\n");
    assert!(converged(status), "status = {status:?}");
    assert!(
        app.last_sqp_working_set().is_none(),
        "the default interior-point path must not record an SQP working set"
    );
}

#[test]
fn forced_convex_selection_is_rejected_in_library() {
    for forced in ["lp-ipm", "qp-ipm", "socp"] {
        let (_app, status, _out) =
            solve_hs35_with(&format!("print_level 0\nsolver_selection {forced}\n"));
        assert_eq!(
            status,
            ApplicationReturnStatus::InvalidOption,
            "solver_selection={forced} must error in a library solve, not fall back to NLP"
        );
    }
}

#[test]
fn solver_selection_and_qp_presolve_are_registered() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    assert!(
        opts.set_string_value("solver_selection", "qp-active-set", true, true)
            .is_ok(),
        "solver_selection should be a registered library option"
    );
    assert!(
        opts.set_string_value("qp_presolve", "no", true, true)
            .is_ok(),
        "qp_presolve should be a registered library option"
    );
    assert!(
        opts.set_string_value("solver_selection", "bogus", true, true)
            .is_err(),
        "an unregistered value must be rejected by the validating registry"
    );
}

// ---------------------------------------------------------------------
// The convex `qp_*` knobs on an entry point that cannot reach the engine
// that reads them (gh#604).
// ---------------------------------------------------------------------

/// All convex knobs live in the core registry, so every frontend
/// *parses* them. Until gh#604 the seven `qp_tau`-family names were
/// registered by `pounce-cli` at startup instead, which made setting one
/// from Python or the C interface fail with `Unknown option "qp_tau"` —
/// naming an option that exists — and made adding any `qp_*` name to the
/// core registry abort the CLI binary at startup.
#[test]
fn every_convex_knob_is_registered_core_side() {
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    for (name, value) in [
        ("qp_tau", 0.9),
        ("qp_tau_max", 0.99),
        ("qp_reg", 1e-9),
        ("qp_infeas_tol", 1e-6),
    ] {
        assert!(
            o.set_numeric_value(name, value, true, true).is_ok(),
            "`{name}` should be a registered library option"
        );
    }
    assert!(
        o.set_integer_value("qp_gondzio_corr", 2, true, true)
            .is_ok(),
        "`qp_gondzio_corr` should be a registered library option"
    );
    for name in ["qp_hsde", "qp_equilibrate", "qp_crossover", "qp_presolve"] {
        assert!(
            o.set_string_value(name, "no", true, true).is_ok(),
            "`{name}` should be a registered library option"
        );
    }
    // The registered ranges still bite: τ ∈ (0,1) open at both ends.
    assert!(o.set_numeric_value("qp_tau", 1.0, true, true).is_err());
    assert!(o.set_numeric_value("qp_reg", -1.0, true, true).is_err());
    assert!(
        o.set_integer_value("qp_gondzio_corr", 11, true, true)
            .is_err()
    );
    assert!(o.set_string_value("qp_hsde", "maybe", true, true).is_err());
}

/// Parsing is not honouring. A library solve cannot route to the convex
/// engines at all (`forced_convex_selection_is_rejected_in_library`
/// above), so a `qp_*` value set here would configure nothing — and is
/// refused with a message rather than accepted and dropped.
#[test]
fn a_convex_knob_is_refused_on_a_library_solve() {
    let (_app, status, _out) = solve_hs35_with("print_level 0\nqp_tau 0.9\n");
    assert_eq!(
        status,
        ApplicationReturnStatus::InvalidOption,
        "a convex-only knob must not be silently ignored on a library solve"
    );

    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("qp_crossover yes\n")
        .unwrap();
    let msg = app
        .unhonored_convex_option()
        .expect("a non-default convex knob must be refused");
    assert!(msg.contains("qp_crossover"), "{msg}");
    // The message used to point at gh#604, which is closed and was about
    // cold-start initialization options — it never covered this gap, and no
    // issue tracks it yet. What has to survive is the way out: the routes
    // that *do* reach the convex engine, and the nearest thing on this one.
    assert!(!msg.contains("604"), "stale gh#604 pointer is back: {msg}");
    assert!(
        msg.contains("solve_qp") && msg.contains("qp-active-set"),
        "message must name a route that works: {msg}"
    );

    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.options_mut()
        .set_integer_value("qp_gondzio_corr", 2, true, true)
        .unwrap();
    let msg = app
        .unhonored_convex_option()
        .expect("qp_gondzio_corr must be guarded like every convex-only knob");
    assert!(msg.contains("qp_gondzio_corr"), "{msg}");
}

/// The same default gate the rest of the refusals use: an `ipopt.opt`
/// that spells out defaults, or a round-tripped full option dump, asks
/// for nothing and must keep solving.
#[test]
fn a_convex_knob_at_its_default_asks_nothing_of_a_library_solve() {
    let (_app, status, _out) =
        solve_hs35_with("print_level 0\nqp_presolve yes\nqp_hsde yes\nqp_tau 0.95\n");
    assert!(
        converged(status),
        "explicitly-set defaults must not fail the solve; status = {status:?}"
    );
}

/// The `pounce` CLI *can* route to those engines, and says so. That is
/// also what keeps its NLP fallback working: a convex attempt that
/// returns no verified point hands the model to `optimize_tnlp`, having
/// genuinely used the `qp_*` values on the way.
#[test]
fn declaring_convex_routing_silences_the_guard() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("qp_tau 0.9\n").unwrap();
    assert!(app.unhonored_convex_option().is_some(), "library default");

    app.set_convex_routing_available(true);
    assert_eq!(
        app.unhonored_convex_option(),
        None,
        "a caller that can reach the convex engines honours the knob"
    );
}
