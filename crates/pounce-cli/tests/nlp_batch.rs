//! End-to-end batched NLP solve (pounce#126) on a real `.nl` model.
//!
//! Exercises the full native-Rust path the issue targets: parse
//! `tests/fixtures/parametric.nl` once, build per-instance [`NlTnlp`]
//! variants (multi-start x0, tightened bounds — the branch-and-bound
//! node shape), and solve them on rayon via
//! [`pounce_algorithm::solve_nlp_batch_parallel`] with the same
//! fully-equipped application recipe the CLI / Python bindings use
//! (serial FERAL backend per worker + restoration-factory provider).
//!
//! Checks the acceptance criteria from the issue:
//! - results come back in input order, one per instance;
//! - the batch result for the unmodified model matches a
//!   single-problem `optimize_tnlp` of the same instance bit-for-bit
//!   (same algorithm, same deterministic serial backend);
//! - a bound-tightened instance honors its own bounds while its
//!   siblings are unaffected.

use pounce_algorithm::application::{default_backend_factory, feral_config_from_options};
use pounce_algorithm::{install_serial_feral_backend, solve_nlp_batch_parallel, IpoptApplication};
use pounce_nl::nl_reader::{read_nl_file, NlTnlp, NlVariation};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

fn parametric_nl() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push("parametric.nl");
    p
}

/// The per-worker application recipe: quiet, inner-serial FERAL
/// backend, and the default restoration-phase provider — the same
/// wiring `pounce-cli`'s solve path and `pounce-py`'s
/// `Problem::prepare` install for a single solve.
fn configure(app: &mut IpoptApplication) {
    let _ = app
        .options_mut()
        .set_integer_value("print_level", 0, true, false);
    install_serial_feral_backend(app);
    let mut feral_cfg = feral_config_from_options(app.options());
    feral_cfg.parallel = Some(false);
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone()))
    };
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(resto_provider);
}

#[test]
fn nl_batch_matches_single_solve_and_honors_variants() {
    let prob = read_nl_file(&parametric_nl()).expect("parse parametric.nl");
    let base = NlTnlp::new(prob);
    let n = base.problem().n;

    // Reference: single-problem solve of the unmodified model.
    let mut app = IpoptApplication::new();
    configure(&mut app);
    let single = Rc::new(RefCell::new(base.clone()));
    let status = app.optimize_tnlp(Rc::clone(&single) as Rc<RefCell<dyn TNLP>>);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let single_x: Vec<f64> = single
        .borrow()
        .final_x()
        .expect("single solve final x")
        .to_vec();
    let single_iters = app.statistics().iteration_count;

    // Batch: [unmodified, multi-start x0, x0-tightened-bound node].
    let mut x0_shift = base.problem().x0.clone();
    for v in &mut x0_shift {
        *v += 0.1;
    }
    let mut xu_tight = base.problem().x_u.clone();
    xu_tight[0] = single_x[0] - 0.05; // force an active bound vs. instance 0
    let variants = base
        .variants(&[
            NlVariation::default(),
            NlVariation {
                x0: Some(x0_shift),
                ..Default::default()
            },
            NlVariation {
                x_u: Some(xu_tight.clone()),
                ..Default::default()
            },
        ])
        .expect("variants");

    let results = solve_nlp_batch_parallel(variants, configure);
    assert_eq!(results.len(), 3, "one result per instance, input order");
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.status,
            ApplicationReturnStatus::SolveSucceeded,
            "instance {i}"
        );
        let sol = r.solution.as_ref().expect("captured solution");
        assert_eq!(sol.x.len(), n);
    }

    // Instance 0 (unmodified) reproduces the single solve bit-for-bit.
    let batch0 = results[0].solution.as_ref().unwrap();
    assert_eq!(
        batch0.x, single_x,
        "batched solve of the unmodified instance must match the single solve"
    );
    assert_eq!(results[0].stats.iteration_count, single_iters);

    // Instance 1 (shifted multi-start) converges to the same optimum.
    let batch1 = results[1].solution.as_ref().unwrap();
    for j in 0..n {
        assert!(
            (batch1.x[j] - single_x[j]).abs() < 1e-5,
            "multi-start instance must reach the same optimum (x[{j}])"
        );
    }

    // Instance 2 (tightened upper bound on x0) honors its own bound
    // and is genuinely different from instance 0.
    let batch2 = results[2].solution.as_ref().unwrap();
    assert!(
        batch2.x[0] <= xu_tight[0] + 1e-8,
        "tightened bound must be honored: x[0] = {} > {}",
        batch2.x[0],
        xu_tight[0]
    );
    assert!(
        (batch2.x[0] - batch0.x[0]).abs() > 1e-6,
        "bound-tightened instance must differ from the unmodified one"
    );
}
