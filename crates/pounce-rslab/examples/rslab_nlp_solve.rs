//! Run a **whole NLP** through POUNCE's interior-point loop with RSLAB
//! underneath, and compare against the same solve on FERAL.
//!
//! Replaying captured KKT systems (`rslab_kkt_replay.rs`) measures one
//! factorization at a time against a fixed trajectory. It cannot answer the
//! question that actually decides whether a backend is usable, because the
//! trajectory is not fixed: a backend that reports a different inertia sends
//! the IPM to a different perturbation, which produces a different iterate,
//! which produces a different KKT. The only way to see that is to let the
//! backend drive.
//!
//! No change to POUNCE's core is needed for this —
//! `IpoptApplication::set_linear_backend_factory` is public API, and it is
//! what the solve below installs.
//!
//! ```sh
//! cargo run -p pounce-rslab --release --example rslab_nlp_solve
//! cargo run -p pounce-rslab --release --example rslab_nlp_solve -- model.nl
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use pounce_algorithm::alg_builder::LinearSolverChoice;
use pounce_algorithm::application::IpoptApplication;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_rslab::{RslabConfig, RslabSolverInterface};

struct Outcome {
    backend: &'static str,
    status: ApplicationReturnStatus,
    iters: i32,
    objective: f64,
    constr_viol: f64,
    dual_inf: f64,
    wall_s: f64,
}

fn solve(
    path: &Path,
    backend: &'static str,
    factory: impl FnMut(LinearSolverChoice) -> Box<dyn SparseSymLinearSolverInterface> + 'static,
) -> Result<Outcome, String> {
    let prob = pounce_nl::nl_reader::read_nl_file(path)?;
    let tnlp: Rc<RefCell<dyn pounce_nlp::tnlp::TNLP>> =
        Rc::new(RefCell::new(pounce_nl::nl_reader::NlTnlp::try_new(prob)?));
    let mut app = IpoptApplication::new();
    app.initialize().map_err(|e| format!("{e:?}"))?;
    app.initialize_with_options_str("print_level 0\nmax_iter 300\n")
        .map_err(|e| format!("{e:?}"))?;
    app.set_linear_backend_factory(Box::new(factory));
    let t = Instant::now();
    let status = app.optimize_tnlp(tnlp);
    let wall_s = t.elapsed().as_secs_f64();
    let s = app.statistics();
    Ok(Outcome {
        backend,
        status,
        iters: s.iteration_count,
        objective: s.final_objective,
        constr_viol: s.final_constr_viol,
        dual_inf: s.final_dual_inf,
        wall_s,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<PathBuf> = if args.is_empty() {
        default_models()
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    println!("# Whole-NLP solves, same model, four backend configurations");
    println!("# rslab    = exact mode (RSLAB's own SolverSettings::default)");
    println!("# rslab-sp = static pivoting at 1e-12*max|A|, the only mode that");
    println!("#            factors a POUNCE KKT at all");
    println!("# rslab-sp+t = rslab-sp, but a perturbed factorization's inertia is");
    println!("#            acted on instead of being routed to Singular\n");
    println!(
        "{:<22} {:<10} {:<26} {:>6} {:>16} {:>11} {:>11} {:>9}",
        "model", "backend", "status", "iters", "objective", "constr_viol", "dual_inf", "wall_s"
    );

    for model in &models {
        let name = model
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| model.display().to_string());
        if !model.is_file() {
            eprintln!("missing: {}", model.display());
            continue;
        }
        let runs = [
            solve(model, "feral", |_| {
                Box::new(pounce_feral::FeralSolverInterface::new())
            }),
            solve(model, "rslab", |_| Box::new(RslabSolverInterface::new())),
            solve(model, "rslab-sp", |_| {
                Box::new(RslabSolverInterface::with_config(
                    RslabConfig::static_pivoting(1e-12),
                ))
            }),
            // Identical to the arm above except that a perturbed
            // factorization's inertia is acted on rather than being routed to
            // `Singular`. The pair attributes any difference in outcome to the
            // reliability gate rather than to the factorization.
            solve(model, "rslab-sp+t", |_| {
                Box::new(RslabSolverInterface::with_config(RslabConfig {
                    trust_perturbed_inertia: true,
                    ..RslabConfig::static_pivoting(1e-12)
                }))
            }),
        ];
        for r in runs {
            match r {
                Ok(o) => println!(
                    "{:<22} {:<10} {:<26} {:>6} {:>16.8e} {:>11.3e} {:>11.3e} {:>9.3}",
                    name,
                    o.backend,
                    format!("{:?}", o.status),
                    o.iters,
                    o.objective,
                    o.constr_viol,
                    o.dual_inf,
                    o.wall_s
                ),
                Err(e) => println!("{name:<22} {:<10} ERROR {e}", "-"),
            }
        }
        println!();
    }
}

fn default_models() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("crates/pounce-cli/tests/fixtures");
    [
        "airport",
        "eigena2",
        "eigenb2",
        "deb7",
        "convex_qp_qscfxm1",
        "lp_degen2",
        "wyndor_min",
    ]
    .iter()
    .map(|m| root.join(format!("{m}.nl")))
    .collect()
}
