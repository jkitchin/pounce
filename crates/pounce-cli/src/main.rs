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
    if args.about {
        print_about();
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
    //
    // Capture the feral config off the now-fully-loaded options so the
    // restoration sub-IPM honors the same `feral_*` overrides (e.g.
    // `feral_cascade_break yes` from an `--options-file`) as the main
    // IPM. Snapshot, not borrow: the BFF outlives the option-mutation
    // window we cleanly own here.
    let feral_cfg = pounce_algorithm::application::feral_config_from_options(app.options());
    let bff: InnerBackendFactoryFactory =
        Box::new(move || default_backend_factory(feral_cfg));
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
    // The registered default for `linear_solver` mirrors upstream IPOPT
    // (`"ma57"`), but pounce's actual default backend is FERAL. Only
    // treat `"ma57"` as user intent when the option was explicitly set
    // — otherwise the banner would always claim "ma57 requested" on
    // every default-config run.
    let backend_tag = {
        let (v, explicit) = app
            .options()
            .get_string_value("linear_solver", "")
            .unwrap_or_else(|_| ("feral".to_string(), false));
        match (v.as_str(), explicit) {
            ("ma57", true) => {
                #[cfg(feature = "ma57")]
                {
                    "MA57 (HSL)"
                }
                #[cfg(not(feature = "ma57"))]
                {
                    "FERAL (ma57 requested but not compiled)"
                }
            }
            ("ma57", false) => "FERAL",
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

/// `--about` output: version, build provenance, compiled-in features,
/// available linear-solver backends, and runtime paths. Intended for
/// bug reports — every field that distinguishes one build from another
/// should appear here.
fn print_about() {
    let pkg_ver = env!("CARGO_PKG_VERSION");
    let git = env!("POUNCE_BUILD_GIT");
    let when = env!("POUNCE_BUILD_TIME");
    let profile = env!("POUNCE_BUILD_PROFILE");
    let target = env!("POUNCE_BUILD_TARGET");
    let host = env!("POUNCE_BUILD_HOST");
    let rustc = env!("POUNCE_BUILD_RUSTC");

    println!("pounce {pkg_ver} (commit {git}, built {when})");
    println!();
    println!("Build:");
    println!("  profile:        {profile}");
    println!("  target:         {target}");
    if host != target {
        println!("  host:           {host}");
    }
    println!("  rustc:          {rustc}");
    println!();

    println!("Features:");
    #[cfg(feature = "ma57")]
    println!("  ma57:           enabled");
    #[cfg(not(feature = "ma57"))]
    println!("  ma57:           disabled (rebuild with --features ma57 to enable HSL MA57)");
    println!();

    println!("Linear solvers:");
    println!("  feral           FERAL pure-Rust sparse LDL^T  (always built-in)");
    #[cfg(feature = "ma57")]
    println!("  ma57            HSL MA57 via libcoinhsl       (compiled in)");
    #[cfg(not(feature = "ma57"))]
    println!("  ma57            HSL MA57 via libcoinhsl       (not compiled; resolves to FERAL at runtime)");
    println!();

    println!("Runtime paths:");
    match std::env::current_exe() {
        Ok(p) => println!("  executable:     {}", p.display()),
        Err(e) => println!("  executable:     <unknown: {e}>"),
    }
    match std::env::current_dir() {
        Ok(p) => println!("  cwd:            {}", p.display()),
        Err(e) => println!("  cwd:            <unknown: {e}>"),
    }
    println!();

    println!(
        "Report bugs at {}/issues",
        env!("CARGO_PKG_REPOSITORY")
    );
}

/// Default backend factory used by the restoration sub-IPM. Mirrors
/// the `default_backend_factory` in `pounce-algorithm`: FERAL is the
/// shipping default, with MA57 available behind the `ma57` cargo
/// feature. The `feral_cfg` argument carries the `feral_*` extension
/// options (cascade-break / FMA / iterative-refinement) captured from
/// the application's options list, so per-problem `.opt` overrides
/// flow into the resto sub-IPM as well.
fn default_backend_factory(feral_cfg: pounce_feral::FeralConfig) -> LinearBackendFactory {
    Box::new(move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
        match choice {
            LinearSolverChoice::Feral => {
                Box::new(pounce_feral::FeralSolverInterface::with_config(feral_cfg))
            }
            LinearSolverChoice::Ma57 => {
                #[cfg(feature = "ma57")]
                {
                    Box::new(pounce_hsl::Ma57SolverInterface::new())
                }
                #[cfg(not(feature = "ma57"))]
                {
                    Box::new(pounce_feral::FeralSolverInterface::with_config(feral_cfg))
                }
            }
        }
    })
}
