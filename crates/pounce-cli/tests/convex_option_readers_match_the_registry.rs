//! The convex option readers must not be narrower than the registry.
//!
//! `pounce_convex::{QpOptions, ConvexPresolveOptions}` and
//! `pounce_qp::ActiveSetOverrides` re-check the bounds that
//! `pounce_algorithm::upstream_options` registers. That duplication is
//! deliberate — a caller may hand a reader an `OptionsList` with no registry
//! attached, and then the reader is the only thing standing between a bad
//! value and the solver — but it is duplication, and nothing else compares the
//! two copies.
//!
//! That is the shape of defect this repository keeps paying for. #677:
//! `limited_memory_initialization` was registered with one default and read
//! with another, and no layer of testing compared the registry against
//! behaviour. #551 caution 1: "re-derive the classification mechanically; do
//! not trust a hand-maintained list."
//!
//! So this test derives the registry's verdict at run time rather than
//! restating any bound. For every probe value it asks the *registry* whether
//! the value is legal, and where the answer is yes it requires the reader to
//! accept it too. A bound widened in `upstream_options.rs` and not widened in
//! the reader fails here, naming the option and the value.
//!
//! The converse — a reader more permissive than the registry — is not an
//! error and is not asserted. `max_wall_time` is the live example: the
//! registry bounds it strictly above zero, while the reader accepts `0.0` as
//! an immediate deadline for the no-registry caller. A reader can only be
//! reached with a registry-rejected value when there is no registry, and then
//! its own rule is the whole contract.
//!
//! This crate is the test's home because it is the only one that depends on
//! both `pounce-algorithm` (which owns the registry) and `pounce-convex`
//! (which owns the readers, and re-exports `pounce-qp`'s).

use pounce_algorithm::IpoptApplication;
use pounce_common::OptionsList;
// `ActiveSetOverrides` is defined in `pounce-qp` and re-exported here; reach
// it by the public path a CLI-side caller would.
use pounce_convex::{ActiveSetOverrides, ConvexPresolveOptions, QpOptions};

/// Run all three readers; `Err` from any of them is a rejection.
fn readers_accept(options: &OptionsList) -> Result<(), String> {
    QpOptions::try_from_options_list(options).map_err(|e| format!("QpOptions: {e}"))?;
    ConvexPresolveOptions::try_from_options_list(options)
        .map_err(|e| format!("ConvexPresolveOptions: {e}"))?;
    ActiveSetOverrides::try_from_options_list(options)
        .map_err(|e| format!("ActiveSetOverrides: {e}"))?;
    Ok(())
}

fn app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.initialize().expect("registry initializes");
    app
}

#[derive(Clone, Copy)]
enum Probe {
    Num(f64),
    Int(i32),
    Str(&'static str),
}

impl std::fmt::Display for Probe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Probe::Num(v) => write!(f, "{v}"),
            Probe::Int(v) => write!(f, "{v}"),
            Probe::Str(v) => write!(f, "{v}"),
        }
    }
}

/// Boundary and just-past-boundary values for every option the three readers
/// re-validate. Whether each is legal is *not* stated here — the registry is
/// asked at run time. The list only has to be wide enough to straddle each
/// registered bound.
const PROBES: &[(&str, &[Probe])] = &[
    (
        "qp_tau",
        &[
            Probe::Num(0.0),
            Probe::Num(1e-12),
            Probe::Num(0.5),
            Probe::Num(1.0 - 1e-12),
            Probe::Num(1.0),
        ],
    ),
    (
        "qp_tau_max",
        &[
            Probe::Num(0.0),
            Probe::Num(0.5),
            Probe::Num(1.0 - 1e-12),
            Probe::Num(1.0),
        ],
    ),
    (
        "qp_reg",
        &[Probe::Num(-1.0), Probe::Num(0.0), Probe::Num(1e-10)],
    ),
    (
        "qp_infeas_tol",
        &[Probe::Num(0.0), Probe::Num(1e-12), Probe::Num(1.0)],
    ),
    (
        "qp_gondzio_corr",
        &[
            Probe::Int(-1),
            Probe::Int(0),
            Probe::Int(3),
            Probe::Int(10),
            Probe::Int(11),
        ],
    ),
    (
        "tol",
        &[Probe::Num(0.0), Probe::Num(1e-14), Probe::Num(1.0)],
    ),
    (
        "max_iter",
        &[
            Probe::Int(-1),
            Probe::Int(0),
            Probe::Int(1),
            Probe::Int(3000),
        ],
    ),
    (
        "max_wall_time",
        &[Probe::Num(1e-9), Probe::Num(1.0), Probe::Num(1e20)],
    ),
    (
        "qp_hsde",
        &[Probe::Str("yes"), Probe::Str("no"), Probe::Str("maybe")],
    ),
    ("qp_equilibrate", &[Probe::Str("yes"), Probe::Str("no")]),
    ("qp_crossover", &[Probe::Str("yes"), Probe::Str("no")]),
    ("qp_presolve", &[Probe::Str("yes"), Probe::Str("no")]),
    ("presolve", &[Probe::Str("yes"), Probe::Str("no")]),
    (
        "sqp_qp_max_iter",
        &[
            Probe::Int(0),
            Probe::Int(1),
            Probe::Int(200),
            Probe::Int(5000),
        ],
    ),
    (
        "sqp_qp_max_schur_updates_before_refactor",
        &[Probe::Int(0), Probe::Int(1), Probe::Int(50)],
    ),
    (
        "sqp_qp_feas_tol",
        &[Probe::Num(0.0), Probe::Num(1e-14), Probe::Num(1.0)],
    ),
    (
        "sqp_qp_opt_tol",
        &[Probe::Num(0.0), Probe::Num(1e-14), Probe::Num(1.0)],
    ),
    (
        "sqp_qp_elastic_gamma",
        &[Probe::Num(0.0), Probe::Num(1e-14), Probe::Num(1e6)],
    ),
    (
        "sqp_qp_anti_cycling",
        &[
            Probe::Str("expand"),
            Probe::Str("bland"),
            Probe::Str("none"),
            Probe::Str("maybe"),
        ],
    ),
    (
        "sqp_qp_use_schur_updates",
        &[Probe::Str("yes"), Probe::Str("no")],
    ),
    (
        "sqp_qp_use_homotopy",
        &[Probe::Str("yes"), Probe::Str("no")],
    ),
    (
        "sqp_qp_certify_second_order",
        &[Probe::Str("yes"), Probe::Str("no")],
    ),
];

#[test]
fn a_value_the_registry_accepts_is_never_rejected_by_a_reader() {
    let mut app = app();
    let mut checked = 0usize;

    for (name, probes) in PROBES {
        for probe in *probes {
            let options = app.options_mut();
            options.unset_value(name);
            let registry_accepts = match probe {
                Probe::Num(v) => options.set_numeric_value(name, *v, true, true).is_ok(),
                Probe::Int(v) => options.set_integer_value(name, *v, true, true).is_ok(),
                Probe::Str(v) => options.set_string_value(name, v, true, true).is_ok(),
            };
            if !registry_accepts {
                continue;
            }
            checked += 1;
            if let Err(why) = readers_accept(app.options()) {
                panic!(
                    "the registry accepts `{name} = {probe}` but a convex option reader \
                     refuses it: {why}\n\n\
                     The reader's hardcoded bound has drifted from the one \
                     `pounce_algorithm::upstream_options` registers. Widen the reader to \
                     match, or tighten the registration — but do not leave a value that \
                     validates at set time and then fails when it is read."
                );
            }
            app.options_mut().unset_value(name);
        }
    }

    // A typo in `PROBES` that names no registered option would otherwise make
    // this test pass by checking nothing.
    assert!(
        checked >= PROBES.len(),
        "only {checked} probe values were accepted by the registry across \
         {} options — the probe table is not reaching the registry",
        PROBES.len()
    );
}

#[test]
fn an_untouched_registry_yields_exactly_the_rust_defaults() {
    // The other half of the drift guard, and the reason the readers gate every
    // field on the `explicitly_set` flag: a registered default must never be
    // read as a request. `max_iter` is the one that bites — Ipopt's default is
    // far larger than the convex driver's, and silently adopting it would turn
    // a tuned cap into no cap at all.
    let app = app();

    let qp = QpOptions::try_from_options_list(app.options()).expect("defaults must parse");
    let expected = QpOptions::default();
    assert_eq!(qp.max_iter, expected.max_iter);
    assert_eq!(qp.tol, expected.tol);
    assert_eq!(qp.time_limit, expected.time_limit);
    assert_eq!(qp.tau, expected.tau);
    assert_eq!(qp.tau_max, expected.tau_max);
    assert_eq!(qp.reg, expected.reg);
    assert_eq!(qp.gondzio_max_corr, expected.gondzio_max_corr);
    assert_eq!(qp.infeas_tol, expected.infeas_tol);
    assert_eq!(qp.use_hsde, expected.use_hsde);
    assert_eq!(qp.equilibrate, expected.equilibrate);
    assert_eq!(qp.crossover, expected.crossover);

    assert_eq!(
        ConvexPresolveOptions::try_from_options_list(app.options()).expect("defaults must parse"),
        ConvexPresolveOptions::default()
    );

    assert!(
        ActiveSetOverrides::try_from_options_list(app.options())
            .expect("defaults must parse")
            .is_empty(),
        "an untouched registry must produce no active-set overrides"
    );
}

/// The registered defaults are also the *values* the readers would apply if
/// the explicit-set gate ever broke — so pin the two that a reader would
/// otherwise silently disagree with.
#[test]
fn an_explicitly_set_default_still_reads_as_that_default() {
    let mut app = app();
    app.options_mut()
        .set_numeric_value("qp_tau", 0.95, true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("qp_gondzio_corr", 3, true, true)
        .unwrap();

    let qp = QpOptions::try_from_options_list(app.options()).unwrap();
    assert_eq!(qp.tau, 0.95);
    assert_eq!(qp.gondzio_max_corr, 3);
    // `qp_tau` lifts the ceiling with the floor, and 0.95 is below the
    // default ceiling, so an unset `qp_tau_max` must stay put.
    assert_eq!(qp.tau_max, QpOptions::default().tau_max);
}
