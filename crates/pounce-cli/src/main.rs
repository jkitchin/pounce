//! `pounce` — command-line driver for the POUNCE solver.
//!
//! Output is structured to mirror upstream `ipopt`'s console layout:
//! a banner, a problem-statistics block, the per-iteration table, and
//! a final residual / eval-count summary. The intent is that anyone
//! used to reading `ipopt` output can drop in `pounce` without
//! relearning where the numbers live.
//!
//! Exit status: 0 on `Solve_Succeeded`, non-zero otherwise.

use pounce_cli::builtin;
use pounce_cli::cli::{Args, ProblemSource};
use pounce_cli::counting_tnlp::CountingTnlp;
use pounce_cli::nl_reader;
use pounce_cli::print;
use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{make_default_restoration_factory, InnerBackendFactoryFactory};
use std::cell::RefCell;
use std::process::ExitCode;
use std::rc::Rc;

fn main() -> ExitCode {
    let args = match Args::parse_argv(std::env::args().collect()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("pounce: {msg}");
            eprintln!("{}", Args::usage());
            return ExitCode::from(2);
        }
    };

    if args.help {
        println!("{}", Args::usage());
        return ExitCode::SUCCESS;
    }
    if args.version {
        println!("pounce {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }

    let mut app = IpoptApplication::new();

    if let Some(path) = &args.options_file {
        if let Err(e) = app.initialize_with_options_file(path) {
            eprintln!("pounce: failed to load options file: {e}");
            return ExitCode::from(2);
        }
    } else if let Err(e) = app.initialize() {
        eprintln!("pounce: initialize failed: {e}");
        return ExitCode::from(2);
    }

    // Apply CLI `key=value` overrides after initialization, mirroring
    // how upstream's ipopt CLI lets command-line options override the
    // ipopt.opt file. Routed through `read_from_str` so the type
    // coercion (string / number / integer) matches the options-file
    // parser exactly.
    for (k, v) in &args.set_options {
        let line = format!("{k} {v}\n");
        if let Err(e) = app.options_mut().read_from_str(&line, true) {
            eprintln!("pounce: failed to set {k}={v}: {e}");
            return ExitCode::from(2);
        }
    }

    // Wire the restoration phase. Without this, any line-search failure
    // surfaces as `RestorationFailure` instead of falling back into the
    // ℓ1-feasibility sub-IPM. Mirrors what upstream's `IpAlgBuilder`
    // does unconditionally for every solve.
    let bff: InnerBackendFactoryFactory = Box::new(default_backend_factory);
    let resto_factory = make_default_restoration_factory(
        RestoAlgorithmBuilder::new(),
        AlgorithmBuilder::new(),
        bff,
    );
    app.set_restoration_factory(resto_factory);

    let inner_tnlp: Rc<RefCell<dyn TNLP>> = match args.problem {
        ProblemSource::Builtin(name) => match builtin::lookup(&name) {
            Some(t) => t,
            None => {
                eprintln!("pounce: unknown builtin problem '{name}'");
                eprintln!("available: {}", builtin::list().join(", "));
                return ExitCode::from(2);
            }
        },
        ProblemSource::NlFile(path) => {
            println!("Reading {}...", path.display());
            let t0 = std::time::Instant::now();
            match nl_reader::load_nl_as_tnlp(&path) {
                Ok(t) => {
                    let elapsed = t0.elapsed().as_secs_f64();
                    if let Some(info) = t.borrow_mut().get_nlp_info() {
                        println!(
                            "Parsed {} vars, {} cons, jac_nnz={}, h_nnz={} in {:.2}s",
                            info.n, info.m, info.nnz_jac_g, info.nnz_h_lag, elapsed
                        );
                    }
                    t
                }
                Err(e) => {
                    eprintln!("pounce: failed to read {}: {e}", path.display());
                    return ExitCode::from(2);
                }
            }
        }
    };

    // Optionally wrap with presolve before counting so eval-call
    // counts reflect what the solver actually issues.
    let presolve_opts =
        match pounce_presolve::PresolveOptions::from_options_list(app.options()) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("pounce: presolve setup failed: {e}");
                return ExitCode::from(2);
            }
        };
    let presolve_handle = if presolve_opts.enabled {
        let p = Rc::new(RefCell::new(pounce_presolve::PresolveTnlp::new(
            Rc::clone(&inner_tnlp),
            presolve_opts,
        )));
        // Force the lazy init now so we can print a one-line summary.
        let _ = p.borrow_mut().get_nlp_info();
        {
            let h = p.borrow();
            let tr = h.tighten_report();
            let dropped = h.n_dropped_rows();
            let licq = h
                .licq_verdict()
                .map(|v| format!("{v:?}"))
                .unwrap_or_else(|| "off".into());
            println!(
                "Presolve: tightened {} bounds ({} newly-finite), dropped {} redundant rows, LICQ={}",
                tr.n_tightened, tr.n_new_finite, dropped, licq
            );
        }
        Some(p)
    } else {
        None
    };
    let post_presolve: Rc<RefCell<dyn TNLP>> = match &presolve_handle {
        Some(p) => Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
        None => Rc::clone(&inner_tnlp),
    };

    // Wrap so we can pull eval-call counts out for the final summary.
    let counting = Rc::new(RefCell::new(CountingTnlp::new(Rc::clone(&post_presolve))));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&counting) as Rc<RefCell<dyn TNLP>>;

    // Banner + problem-statistics block, before the iteration table.
    let backend_tag = {
        let v = app
            .options()
            .get_string_value("linear_solver", "")
            .ok()
            .map(|(s, _)| s)
            .unwrap_or_else(|| "ma57".to_string());
        match v.as_str() {
            "ma57" => {
                #[cfg(feature = "ma57")]
                {
                    "MA57 (HSL)"
                }
                #[cfg(not(feature = "ma57"))]
                {
                    "FERAL (ma57 requested but not compiled)"
                }
            }
            _ => "FERAL",
        }
    };
    print::print_banner(backend_tag);
    if let Some(stats) = print::collect_stats(&tnlp) {
        print::print_problem_stats(&stats);
    }

    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    let solve_stats = app.statistics();
    let counters = counting.borrow();
    print::print_summary(status, &solve_stats, &counters);

    match status {
        ApplicationReturnStatus::SolveSucceeded
        | ApplicationReturnStatus::SolvedToAcceptableLevel => ExitCode::SUCCESS,
        _ => ExitCode::from(1),
    }
}

/// Default backend factory used by the restoration sub-IPM. Mirrors
/// the private `default_backend_factory` in `pounce-algorithm` (which
/// is not re-exported): FERAL is the shipping default, with MA57
/// available behind the `ma57` cargo feature.
fn default_backend_factory() -> LinearBackendFactory {
    Box::new(|choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
        match choice {
            LinearSolverChoice::Feral => Box::new(pounce_feral::FeralSolverInterface::new()),
            LinearSolverChoice::Ma57 => {
                #[cfg(feature = "ma57")]
                {
                    Box::new(pounce_hsl::Ma57SolverInterface::new())
                }
                #[cfg(not(feature = "ma57"))]
                {
                    Box::new(pounce_feral::FeralSolverInterface::new())
                }
            }
        }
    })
}
