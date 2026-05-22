//! `pounce` — command-line driver for the POUNCE solver.
//!
//! Output is structured to mirror upstream `ipopt`'s console layout:
//! a banner, a problem-statistics block, the per-iteration table, and
//! a final residual / eval-count summary. The intent is that anyone
//! used to reading `ipopt` output can drop in `pounce` without
//! relearning where the numbers live.
//!
//! Exit status: 0 on `Solve_Succeeded`, non-zero otherwise. In AMPL
//! solver mode (`-AMPL`) the exit code instead follows the AMPL
//! contract — 0 for any solve that ran and produced a `.sol`, since
//! the termination is carried by the file's `solve_result_num`.

use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_cli::builtin;
use pounce_cli::cli::{Args, ProblemSource};
use pounce_cli::counting_tnlp::CountingTnlp;
use pounce_cli::nl_reader;
use pounce_cli::nl_writer;
use pounce_cli::print;
use pounce_cli::solve_report::{write_report_file, InputDescriptor, ReportBuilder, ReportDetail};
use pounce_common::diagnostics::{
    DiagCategory, DiagnosticsConfig, DiagnosticsState, DumpFormat, IterSpec,
};
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::path::PathBuf;
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

    // Opt into iter-history capture when the user asked for a JSON
    // report at Full detail — saves the per-iter alloc when they
    // didn't.
    if args.json_output.is_some() && matches!(args.json_detail, ReportDetail::Full) {
        app.enable_iter_history();
    }

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
    let bff: InnerBackendFactoryFactory = Box::new(move || default_backend_factory(feral_cfg));
    let resto_factory = make_default_restoration_factory(
        RestoAlgorithmBuilder::new(),
        AlgorithmBuilder::new(),
        bff,
    );
    app.set_restoration_factory(resto_factory);

    // Snapshot the problem source as a string — needed downstream by
    // the diagnostics manifest.
    let problem_desc: String = match &args.problem {
        ProblemSource::Builtin(s) => format!("builtin:{s}"),
        ProblemSource::NlFile(p) => format!("nl:{}", p.display()),
    };

    // Resolve where (if anywhere) to write an AMPL `.sol` solution
    // file. AMPL solver convention: a `.nl` input gets a sibling
    // `<stub>.sol` unless suppressed. Builtins have no stub on disk,
    // so they only produce a `.sol` when `--sol-output` is explicit.
    let sol_path: Option<PathBuf> = if args.no_sol {
        None
    } else if let Some(p) = &args.sol_output {
        Some(p.clone())
    } else {
        match &args.problem {
            ProblemSource::NlFile(p) => {
                let mut s = p.clone();
                s.set_extension("sol");
                Some(s)
            }
            ProblemSource::Builtin(_) => None,
        }
    };

    // Capture the converged primal / dual into `nominal_capture` so
    // the JSON report below can ship `solution.x` and
    // `solution.lambda` (mirrors what `pounce_sens` does).
    let nominal_capture: Rc<
        RefCell<
            Option<(
                Vec<pounce_common::types::Number>,
                Vec<pounce_common::types::Number>,
            )>,
        >,
    > = Rc::new(RefCell::new(None));
    if args.json_output.is_some() || sol_path.is_some() {
        let cap = Rc::clone(&nominal_capture);
        app.set_on_converged(Box::new(move |data, _cq, nlp, _pd| {
            let curr = match data.borrow().curr.clone() {
                Some(c) => c,
                None => return,
            };
            // Lift to full length so a fixed / eliminated variable
            // still occupies its slot — AMPL's `.sol` reader matches
            // the x block against the originating `.nl`'s var count.
            let x = nlp.borrow().lift_x_to_full(&*curr.x);
            let n_c = curr.y_c.dim() as usize;
            let n_d = curr.y_d.dim() as usize;
            let mut lambda = Vec::with_capacity(n_c + n_d);
            if let Some(dv) = curr
                .y_c
                .as_any()
                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            {
                lambda.extend_from_slice(&dv.expanded_values());
            } else {
                lambda.extend(std::iter::repeat(0.0).take(n_c));
            }
            if let Some(dv) = curr
                .y_d
                .as_any()
                .downcast_ref::<pounce_linalg::dense_vector::DenseVector>()
            {
                lambda.extend_from_slice(&dv.expanded_values());
            } else {
                lambda.extend(std::iter::repeat(0.0).take(n_d));
            }
            *cap.borrow_mut() = Some((x, lambda));
        }));
    }

    let inner_tnlp: Rc<RefCell<dyn TNLP>> = match &args.problem {
        ProblemSource::Builtin(name) => match builtin::lookup(name) {
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
            match nl_reader::load_nl_as_tnlp(path) {
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
    let presolve_opts = match pounce_presolve::PresolveOptions::from_options_list(app.options()) {
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
    // `sb yes` skips the copyright banner. Mirrors upstream
    // `IpoptApplication::Initialize` which also reads `sb` and gates
    // the banner. Problem-stats and iter rows are unaffected.
    let suppress_banner = app
        .options()
        .get_bool_value("sb", "")
        .ok()
        .and_then(|(v, f)| f.then_some(v))
        .unwrap_or(false);
    if !suppress_banner {
        print::print_banner(backend_tag);
    }
    if let Some(stats) = print::collect_stats(&tnlp) {
        print::print_problem_stats(&stats);
    }

    // Build diagnostics state from `--dump …` flags. None of these
    // flags is required, but `--dump-dir` / `--dump-format` on their
    // own (no `--dump <cat>`) yields an empty config and we skip
    // installation entirely — there's nothing to write.
    let diagnostics_handle = match build_diagnostics(
        &args.dump_specs,
        args.dump_dir.as_ref(),
        args.dump_format.as_deref(),
    ) {
        Ok(d) => d,
        Err(msg) => {
            eprintln!("pounce: {msg}");
            return ExitCode::from(2);
        }
    };
    if let Some(diag) = diagnostics_handle.as_ref() {
        println!(
            "Diagnostics: dumping to {} ({} categor{} configured)",
            diag.dump_dir().display(),
            diag.config.categories.len(),
            if diag.config.categories.len() == 1 {
                "y"
            } else {
                "ies"
            },
        );
        app.set_diagnostics(Rc::clone(diag));
    }

    // Snapshot NLP dimensions before the solve so we can use them in
    // both the console summary and the JSON report. Borrowing here is
    // safe because we hold no outstanding borrow on the counting
    // wrapper yet.
    let nlp_info_snapshot = tnlp.borrow_mut().get_nlp_info();

    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    let solve_stats = app.statistics();
    let counters = counting.borrow();
    print::print_summary(status, &solve_stats, &counters);
    drop(counters); // release before JSON block (which re-borrows the wrapped TNLP).

    // Emit the JSON solve report, when requested. Written AFTER the
    // console summary so a piped `pounce ... --json-output -` reader
    // could be wired up later without disturbing stdout (today we
    // write to a path; stdout-mode is a follow-up).
    if let Some(json_path) = &args.json_output {
        let input = match &args.problem {
            ProblemSource::Builtin(name) => InputDescriptor::Builtin { name: name.clone() },
            ProblemSource::NlFile(p) => InputDescriptor::NlFile {
                path: p.clone(),
                size_bytes: std::fs::metadata(p).ok().map(|m| m.len()),
            },
        };
        let mut builder = ReportBuilder::new(args.json_detail, input);
        if let Some(info) = nlp_info_snapshot {
            builder.problem.n_variables = info.n;
            builder.problem.n_constraints = info.m;
            builder.problem.n_objectives = 1; // pounce IPM uses obj 0; multi-obj is read but ignored
            builder.problem.nnz_jac_g = Some(info.nnz_jac_g);
            builder.problem.nnz_h_lag = Some(info.nnz_h_lag);
        }
        builder.solution.status = status;
        builder.solution.solve_result_num = status_to_solve_result_num(status);
        builder.solution.objective = solve_stats.final_objective;
        if let Some((x, lambda)) = nominal_capture.borrow().clone() {
            builder.solution.x = x;
            builder.solution.lambda = lambda;
        }
        builder.ingest_stats(&solve_stats);

        let report = builder.finish();
        if let Err(e) = write_report_file(json_path, &report) {
            eprintln!(
                "pounce: failed to write JSON report to {}: {e}",
                json_path.display()
            );
        } else {
            eprintln!("pounce: wrote {}", json_path.display());
        }
    }

    // Emit the AMPL `.sol` file. Written unconditionally once a target
    // path is resolved — even on a failed solve — so AMPL's reader
    // always sees a `solve_result_num`, matching `pounce_sens` and
    // upstream AMPL solver behaviour. When the solve never converged
    // the capture is empty; fall back to zero blocks sized from the
    // pre-solve NLP dimensions so the file still round-trips.
    if let Some(sol_path) = &sol_path {
        let (n, m) = nlp_info_snapshot
            .as_ref()
            .map(|i| (i.n as usize, i.m as usize))
            .unwrap_or((0, 0));
        let (x, lambda) = nominal_capture
            .borrow()
            .clone()
            .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m]));
        let message = format!("POUNCE {}: {status:?}", env!("CARGO_PKG_VERSION"));
        let payload = nl_writer::SolutionFile {
            message: &message,
            x: &x,
            lambda: &lambda,
            solve_result_num: status_to_solve_result_num(status),
            suffixes: &[],
        };
        match nl_writer::write_sol_file(sol_path, &payload) {
            Ok(_) => eprintln!("pounce: wrote {}", sol_path.display()),
            Err(e) => eprintln!("pounce: failed to write {}: {e}", sol_path.display()),
        }
    }

    // After the solve, drop a manifest + timing summary at the dump
    // root so consumers (and humans) can tell which run produced
    // which artifacts without reading the iter_NNN/ tree.
    if let Some(diag) = diagnostics_handle.as_ref() {
        write_diagnostics_manifest(diag, &problem_desc, status);
        write_diagnostics_timing(diag, &app);
    }

    match status {
        ApplicationReturnStatus::SolveSucceeded
        | ApplicationReturnStatus::SolvedToAcceptableLevel => ExitCode::SUCCESS,
        // AMPL solver-protocol mode: the process exit code is not the
        // status channel. AMPL and Pyomo's ASL interface read the
        // termination from the `.sol` file's `solve_result_num`, and
        // conventionally an AMPL solver exits 0 whenever it ran and
        // produced a `.sol` — limit reached, infeasible, even a failed
        // solve. A non-zero exit makes Pyomo raise `ApplicationError`
        // and never parse the `.sol`. Genuine startup failures (bad
        // `.nl`, bad option) already returned non-zero above, before
        // the solve, so reaching here in `-AMPL` mode means a `.sol`
        // was written and carries the verdict.
        _ if args.ampl => ExitCode::SUCCESS,
        _ => ExitCode::from(1),
    }
}

/// Translate the CLI's `--dump …` flags into a live `DiagnosticsState`.
/// Returns `Ok(None)` when no `--dump <cat>` was given (the `--dump-dir`
/// / `--dump-format` flags alone don't activate dumping).
fn build_diagnostics(
    dump_specs: &[(String, String)],
    dump_dir: Option<&std::path::PathBuf>,
    dump_format: Option<&str>,
) -> Result<Option<Rc<DiagnosticsState>>, String> {
    if dump_specs.is_empty() {
        if dump_dir.is_some() || dump_format.is_some() {
            return Err(
                "--dump-dir / --dump-format require at least one --dump <cat>[:spec]".to_string(),
            );
        }
        return Ok(None);
    }

    let dump_dir = dump_dir.cloned().unwrap_or_else(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::path::PathBuf::from(format!("pounce-dump-{secs}"))
    });

    let format = match dump_format {
        Some(f) => DumpFormat::parse(f)?,
        None => DumpFormat::Jsonl,
    };

    let mut config = DiagnosticsConfig::new(dump_dir);
    config.format = format;
    for (cat, spec) in dump_specs {
        let cat = DiagCategory::parse(cat)?;
        let spec = IterSpec::parse(spec)?;
        config = config.with_category(cat, spec);
    }

    let state = DiagnosticsState::new(config)
        .map_err(|e| format!("could not create dump directory: {e}"))?;
    Ok(Some(Rc::new(state)))
}

/// Drop a minimal JSON manifest summarising the run. Lets downstream
/// tools (and humans) join a dump directory back to its CLI args
/// without re-reading the per-iter files.
fn write_diagnostics_manifest(
    diag: &DiagnosticsState,
    problem_desc: &str,
    status: ApplicationReturnStatus,
) {
    let mut cats: Vec<String> = diag
        .config
        .categories
        .iter()
        .map(|(c, s)| format!("\"{}\":\"{:?}\"", c.as_str(), s))
        .collect();
    cats.sort();
    let manifest = format!(
        "{{\n  \"pounce_version\": \"{ver}\",\n  \"git\": \"{git}\",\n  \"problem\": \"{problem}\",\n  \"status\": \"{status:?}\",\n  \"format\": \"{fmt:?}\",\n  \"categories\": {{ {cats} }}\n}}\n",
        ver = env!("CARGO_PKG_VERSION"),
        git = env!("POUNCE_BUILD_GIT"),
        problem = problem_desc,
        fmt = diag.config.format,
        cats = cats.join(", "),
    );
    let _ = diag.write_top_level("manifest.json", &manifest);
}

/// Emit a sibling `timing.json` so dump consumers can correlate
/// per-iter files with the solve's wall-clock budget.
fn write_diagnostics_timing(diag: &DiagnosticsState, app: &IpoptApplication) {
    let t = app.timing_stats();
    let body = format!(
        "{{\n  \"overall_alg_secs\": {a:.6},\n  \"linear_system_factorization_secs\": {f:.6},\n  \"linear_system_back_solve_secs\": {b:.6}\n}}\n",
        a = t.overall_alg.total_wallclock_time(),
        f = t.linear_system_factorization.total_wallclock_time(),
        b = t.linear_system_back_solve.total_wallclock_time(),
    );
    let _ = diag.write_top_level("timing.json", &body);
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

    println!("Report bugs at {}/issues", env!("CARGO_PKG_REPOSITORY"));
}

/// Map a pounce `ApplicationReturnStatus` onto an AMPL-style
/// `solve_result_num` per Gay 2005 (Hooking Your Solver to AMPL §5,
/// p. 23 table): 0 = solved, 100s = warning, 200s = infeasible,
/// 400s = limit reached, 500s = failure.
fn status_to_solve_result_num(status: ApplicationReturnStatus) -> i32 {
    use ApplicationReturnStatus::*;
    match status {
        SolveSucceeded => 0,
        SolvedToAcceptableLevel => 100,
        FeasiblePointFound => 100,
        InfeasibleProblemDetected => 200,
        SearchDirectionBecomesTooSmall => 400,
        DivergingIterates => 401,
        MaximumIterationsExceeded => 400,
        MaximumCpuTimeExceeded => 400,
        MaximumWallTimeExceeded => 400,
        UserRequestedStop => 502,
        RestorationFailed => 500,
        ErrorInStepComputation => 500,
        InvalidNumberDetected => 500,
        InternalError => 500,
        UnrecoverableException => 500,
        NonIpoptExceptionThrown => 500,
        InsufficientMemory => 503,
        InvalidProblemDefinition => 504,
        InvalidOption => 504,
        NotEnoughDegreesOfFreedom => 504,
    }
}

/// Default backend factory used by the restoration sub-IPM. Mirrors
/// the `default_backend_factory` in `pounce-algorithm`: FERAL is the
/// shipping default, with MA57 available behind the `ma57` cargo
/// feature. The `feral_cfg` argument carries the `feral_*` extension
/// options (cascade-break / FMA / iterative-refinement) captured from
/// the application's options list, so per-problem `.opt` overrides
/// flow into the resto sub-IPM as well.
fn default_backend_factory(feral_cfg: pounce_feral::FeralConfig) -> LinearBackendFactory {
    Box::new(
        move |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
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
        },
    )
}
