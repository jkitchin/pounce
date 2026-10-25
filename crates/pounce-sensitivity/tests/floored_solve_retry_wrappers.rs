//! Every retry wrapper's floor, and two of them in series, hand sensitivity the
//! reported attempt's KKT state -- and, nested, the reported attempt's *point*.
//!
//! `floored_solve_sensitivity.rs` owns the mechanism on the μ fallback alone.
//! This file covers the other two wrappers that replay a floor, and the case
//! where two declined retries run in series, each on a model from the CLI's
//! fixture corpus that reaches it:
//!
//! | case | model, options | `on_converged` firings |
//! |---|---|---|
//! | gh#884 retry declined | `mpcc_scholtes4_biactive`, `bound_relax_factor=0 tol=1e-8 mu_strategy_fallback=no` | base, retry (3.3e-5 away) |
//! | gh#983 audit declined | `scale_audit_bound_active`, `bound_relax_factor=1e-8` | base, re-solve (2.5e-6 away) |
//! | nested: μ, then gh#884 | scholtes4 as above, `mu_strategy_fallback=yes` | base, μ retry, gh#884 retry |
//!
//! scholtes4's base attempt exits `Solved_To_Acceptable_Level` with the gh#884
//! signature, which is what makes the nested case reachable: the explicit
//! μ fallback retries an acceptable exit and is declined (its retry is also
//! only acceptable), and then the gh#884 retry wrapped around it runs and is
//! declined too.
//!
//! The reference for each case is the same solve with the retry switched off,
//! which reports that same base attempt bit for bit.
//!
//! **The nested case is a wrong answer, not just a stale factor.** Before
//! `IpoptApplication::restore_floor` re-recorded a replayed floor in
//! `last_finalize`, the μ fallback's replay left its own discarded retry's
//! payload there, and the gh#884 wrapper took *that* as its floor: the point
//! delivered to the user's model was the declined μ retry's, beside the base
//! attempt's statistics, and `reported_on_converged_id` named the gh#884
//! retry's firing, so `Solver` held a factor 3.3e-5 away from either.
//!
//! MUTATION TABLE (measured):
//!
//! | change | what goes red |
//! |---|---|
//! | drop the `last_finalize` re-record in `restore_floor` | all ten |
//! | `Solver` keeps its last firing instead of the reported one | every `*_holds_*` and `*_kkt_solve_*` test (six) |
//!
//! The first row is all ten, not just the nested ones, because the re-record
//! is load-bearing for every floor: `reported_on_converged_id` reads
//! `last_finalize`, so without it even a single declined retry names its own
//! firing. Only `nested_delivers_the_reported_point` is about the delivered
//! point, and it is the one place this file sees the old wrong answer: the
//! user's model received the declined μ retry's `[1.68e-13, 1.68e-13,
//! -1.8175e-9]` against the base attempt's `[2.91e-13, 2.91e-13, -1.8170e-9]`.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_linalg::dense_vector::DenseVector;
use pounce_nl::nl_reader::{NlTnlp, read_nl_file};
use pounce_nlp::tnlp::TNLP;
use pounce_sensitivity::Solver;

struct Case {
    model: &'static str,
    with_retry: &'static [(&'static str, &'static str)],
    reference: &'static [(&'static str, &'static str)],
    firings: usize,
}

const GH884: Case = Case {
    model: "mpcc_scholtes4_biactive.nl",
    with_retry: &[
        ("bound_relax_factor", "0"),
        ("tol", "1e-8"),
        ("mu_strategy_fallback", "no"),
    ],
    reference: &[
        ("bound_relax_factor", "0"),
        ("tol", "1e-8"),
        ("mu_strategy_fallback", "no"),
        ("dual_divergence_retry", "no"),
    ],
    firings: 2,
};

const GH983: Case = Case {
    model: "scale_audit_bound_active.nl",
    with_retry: &[("bound_relax_factor", "1e-8")],
    reference: &[
        ("bound_relax_factor", "1e-8"),
        ("solve_quality_audit", "no"),
    ],
    firings: 2,
};

const NESTED: Case = Case {
    model: "mpcc_scholtes4_biactive.nl",
    with_retry: &[
        ("bound_relax_factor", "0"),
        ("tol", "1e-8"),
        ("mu_strategy_fallback", "yes"),
    ],
    reference: &[
        ("bound_relax_factor", "0"),
        ("tol", "1e-8"),
        ("mu_strategy_fallback", "no"),
        ("dual_divergence_retry", "no"),
    ],
    firings: 3,
};

fn model(name: &str) -> Rc<RefCell<NlTnlp>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../pounce-cli/tests/fixtures")
        .join(name);
    let prob = read_nl_file(&path).expect("read fixture");
    Rc::new(RefCell::new(NlTnlp::try_new(prob).expect("NlTnlp")))
}

fn app(opts: &[(&str, &str)]) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .expect("print_level");
    for (k, v) in opts {
        app.options_mut()
            .read_from_str(&format!("{k} {v}\n"), true)
            .expect("option");
    }
    app.initialize().expect("initialize");
    app
}

/// Every firing's primal iterate, and which one the solve reported.
struct Firings {
    x: Vec<Vec<Number>>,
    reported: Option<u64>,
    restored: bool,
}

fn firings(case: &Case) -> Firings {
    let mut app = app(case.with_retry);
    let seen: Rc<RefCell<Vec<Vec<Number>>>> = Rc::new(RefCell::new(Vec::new()));
    let sink = Rc::clone(&seen);
    app.set_on_converged(Box::new(move |data, _cq, _nlp, _pd| {
        let d = data.borrow();
        let x = d.curr.as_ref().expect("iterate").x.as_any();
        let x = x.downcast_ref::<DenseVector>().expect("dense");
        sink.borrow_mut().push(x.expanded_values());
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = model(case.model);
    let _ = app.optimize_tnlp(tnlp);
    let x = seen.borrow().clone();
    Firings {
        x,
        reported: app.reported_on_converged_id(),
        restored: app.answer_restored_from_floor(),
    }
}

/// What a sensitivity consumer sees after the solve.
struct Held {
    /// The point `Solver` holds a factor at.
    held_x: Vec<Number>,
    /// The point delivered to the user's model, i.e. the reported answer.
    delivered_x: Vec<Number>,
    /// One back-solve against the held factor.
    kkt: Vec<Number>,
}

fn held(opts: &[(&str, &str)], name: &str) -> Held {
    let concrete = model(name);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let mut solver = Solver::new(app(opts), tnlp);
    solver.solve();
    let state = solver.converged().expect("a converged attempt is reported");
    let held_x = state.x.clone();
    drop(state);
    let dim = solver.kkt_dim().expect("kkt_dim");
    let rhs: Vec<Number> = (0..dim).map(|i| 1.0 + 0.1 * i as Number).collect();
    let mut kkt = vec![0.0; dim];
    solver.kkt_solve(&rhs, &mut kkt).expect("kkt_solve");
    let delivered_x = concrete
        .borrow()
        .final_x()
        .expect("finalize_solution ran")
        .to_vec();
    Held {
        held_x,
        delivered_x,
        kkt,
    }
}

fn max_diff(a: &[Number], b: &[Number]) -> Number {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, Number::max)
}

/// The premise each case rests on: the retries fired, converged somewhere
/// else, and were floored back to the first firing.
fn premise(case: &Case) {
    let f = firings(case);
    assert_eq!(
        f.x.len(),
        case.firings,
        "{}: expected {} converged attempts",
        case.model,
        case.firings
    );
    assert!(f.restored, "{}: no floor was replayed", case.model);
    assert_eq!(
        f.reported,
        Some(1),
        "{}: the base attempt is reported",
        case.model
    );
    let last = f.x.last().expect("firings");
    assert!(
        max_diff(&f.x[0], last) > 1e-7,
        "{}: the last retry reached the base point, so a stale capture would be invisible",
        case.model
    );
}

fn holds_the_reported_point(case: &Case) {
    let got = held(case.with_retry, case.model);
    let want = held(case.reference, case.model);
    assert!(
        max_diff(&got.held_x, &want.held_x) == 0.0,
        "{}: Solver holds {:?}, the never-retried run {:?}",
        case.model,
        got.held_x,
        want.held_x
    );
}

fn kkt_solve_matches(case: &Case) {
    let got = held(case.with_retry, case.model);
    let want = held(case.reference, case.model);
    let scale = want.kkt.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    assert!(
        max_diff(&got.kkt, &want.kkt) <= 1e-12 * scale,
        "{}: the back-solve after the floor differs from the never-retried run's \
         by {:e}: the held factor is a discarded attempt's",
        case.model,
        max_diff(&got.kkt, &want.kkt)
    );
}

fn delivers_the_reported_point(case: &Case) {
    let got = held(case.with_retry, case.model);
    let want = held(case.reference, case.model);
    // Exact: both runs report the same, deterministic base attempt. The
    // nested defect delivered the declined μ retry's point, which on this
    // model sits only ~1e-13 from the base -- a tolerance would hide it.
    assert!(
        max_diff(&got.delivered_x, &want.delivered_x) == 0.0,
        "{}: the user's model received {:?}, the never-retried run {:?}",
        case.model,
        got.delivered_x,
        want.delivered_x
    );
}

#[test]
fn gh884_premise() {
    premise(&GH884);
}
#[test]
fn gh884_holds_the_reported_point() {
    holds_the_reported_point(&GH884);
}
#[test]
fn gh884_kkt_solve_matches_the_never_retried_run() {
    kkt_solve_matches(&GH884);
}

#[test]
fn gh983_premise() {
    premise(&GH983);
}
#[test]
fn gh983_holds_the_reported_point() {
    holds_the_reported_point(&GH983);
}
#[test]
fn gh983_kkt_solve_matches_the_never_retried_run() {
    kkt_solve_matches(&GH983);
}

#[test]
fn nested_premise() {
    premise(&NESTED);
}
#[test]
fn nested_holds_the_reported_point() {
    holds_the_reported_point(&NESTED);
}
#[test]
fn nested_kkt_solve_matches_the_never_retried_run() {
    kkt_solve_matches(&NESTED);
}
#[test]
fn nested_delivers_the_reported_point() {
    delivers_the_reported_point(&NESTED);
}
