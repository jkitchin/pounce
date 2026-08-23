//! `pounce` — command-line driver for the POUNCE solver.
//!
//! Output is structured to mirror upstream `ipopt`'s console layout:
//! a banner, a problem-statistics block, the per-iteration table, and
//! a final residual / eval-count summary. The intent is that anyone
//! used to reading `ipopt` output can drop in `pounce` without
//! relearning where the numbers live.
//!
//! Exit status: 0 on a successful solve — `Solve_Succeeded` or
//! `Solved_To_Acceptable_Level` (the reduced-accuracy convergence Ipopt
//! likewise treats as success) — and non-zero otherwise. In AMPL solver
//! mode (`-AMPL`) the exit code instead follows the AMPL contract — 0 for
//! any solve that ran and produced a `.sol`, since the termination is
//! carried by the file's `solve_result_num`.

use pounce_algorithm::alg_builder::{LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_cli::builtin;
use pounce_cli::cli::{Args, ProblemSource};
use pounce_cli::nl_reader;
use pounce_cli::nl_writer;
use pounce_cli::print;
use pounce_cli::sens;
use pounce_cli::solve_report::{
    InputDescriptor, ReportBuilder, ReportDetail, SolutionSuffix, status_to_solve_result_num,
    write_report_file,
};
use pounce_common::diagnostics::{
    DiagCategory, DiagnosticsConfig, DiagnosticsState, DumpFormat, IterSpec,
};
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::SolveStatistics;
use pounce_nlp::counting_tnlp::CountingTnlp;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::IterRecord;
use pounce_nlp::tnlp::{InfeasibilityProof, TNLP};
use pounce_restoration::install::install_default_restoration;
use std::cell::RefCell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;

/// The reported `(message, solve_result_num)` for a finished solve.
///
/// Single source of truth shared by the JSON report and the `.sol` writer — a
/// run reporting `201` in one and `200` in the other is a bug a caller has no
/// way to reconcile.
///
/// A presolve-certified infeasibility reports `201`; everything else uses the
/// standard status mapping. Both sit in AMPL's `200..299` infeasible band, so
/// band-reading consumers (Pyomo included) are unaffected either way.
fn presolve_verdict(
    certified: Option<InfeasibilityProof>,
    status: ApplicationReturnStatus,
) -> (String, i32) {
    // The certificate only *relabels* an infeasibility verdict — it never
    // manufactures one. The application short-circuits on a proof before
    // dispatch, so the two normally agree; but the SQP engine is dispatched
    // ahead of that check and reports its own status, and the CLI computes
    // `certified` from its own presolve handle. Without this guard a
    // disagreement would write "proved infeasible" with `201` on top of a
    // successful solve's `x` — a self-contradictory `.sol` that no caller
    // could reconcile. If they ever disagree, trust the engine that ran.
    match certified.filter(|_| status == ApplicationReturnStatus::InfeasibleProblemDetected) {
        Some(proof) => {
            let detail = match proof {
                InfeasibilityProof::BoundPropagation => "bound propagation".to_string(),
                InfeasibilityProof::IntervalArithmetic { witness } => {
                    format!("interval arithmetic, constraint {witness}")
                }
            };
            (
                format!(
                    "POUNCE {}: InfeasibleProblemDetected (detected by presolve: {detail})",
                    env!("CARGO_PKG_VERSION")
                ),
                201,
            )
        }
        None => (
            format!("POUNCE {}: {status:?}", env!("CARGO_PKG_VERSION")),
            status_to_solve_result_num(status),
        ),
    }
}

/// Whether the resolved options select `nlp_scaling_method=curvature-based`
/// (gh #703). Read off the `OptionsList` rather than the raw argv so the
/// option file, the `pounce_options` environment variable and the
/// command line are all honoured in the order they are applied.
fn curvature_scaling_requested(app: &pounce_algorithm::application::IpoptApplication) -> bool {
    app.options()
        .get_string_value("nlp_scaling_method", "")
        .ok()
        .and_then(|(v, f)| f.then_some(v))
        .is_some_and(|v| v == "curvature-based")
}

pub fn main() -> ExitCode {
    // Install the tracing subscriber first so even argument-parse
    // diagnostics and the iteration collector are active (pounce#71).
    // Honors RUST_LOG, NO_COLOR, and POUNCE_LOG_FORMAT.
    pounce_observability::init_subscriber();

    // `pounce verify <problem.nl> <claim.sol>` — an independent solution
    // checker that re-derives feasibility from the canonical problem. It is
    // a distinct subcommand (not a solve), so dispatch it before the normal
    // argv parser and solve path. See `pounce_cli::verify`.
    let raw_argv: Vec<String> = std::env::args().collect();
    if raw_argv.get(1).map(|s| s == "verify").unwrap_or(false) {
        return pounce_cli::verify::run_from_argv(&raw_argv[2..]);
    }

    // `pounce check-x0 <problem.nl>` — starting-point preflight: evaluate
    // the model once at x0 and report NaN/inf, bound/constraint violations,
    // interior-clamp displacement, and derivative scale spread before any
    // solve. See `pounce_cli::check_x0` and docs/src/initialization.md.
    if raw_argv.get(1).map(|s| s == "check-x0").unwrap_or(false) {
        return pounce_cli::check_x0::run_from_argv(&raw_argv[2..]);
    }

    let mut args = match Args::parse_argv(std::env::args().collect()) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("pounce: {msg}");
            eprintln!("{}", Args::usage());
            return ExitCode::from(2);
        }
    };

    // AMPL drivers pass solver directives via the `<solver>_options` env
    // var (`pounce_options`): a whitespace-separated list of `key=value`
    // tokens. Merge them ahead of the command-line `key=value` options so
    // an explicit CLI flag overrides the env var (set_options is applied
    // last-wins). Pyomo, which writes options as CLI args, is unaffected.
    if let Ok(env_opts) = std::env::var("pounce_options") {
        let mut merged = pounce_cli::cli::options_from_env(&env_opts);
        if !merged.is_empty() {
            merged.append(&mut args.set_options);
            args.set_options = merged;
        }
    }

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
    if args.cite {
        return run_cite(&args);
    }

    let mut app = IpoptApplication::new();

    // This binary is the one entry point that can route a model to the
    // convex LP/QP/SOCP engines (`solver_selection` + the structure
    // extraction that classifies the `.nl`), so the `qp_*` knobs those
    // engines read configure something here. Declaring that is what keeps
    // `IpoptApplication`'s guard from refusing them on the NLP fallback
    // path — a convex attempt that hands off to `optimize_tnlp` used them
    // for real (gh#604).
    app.set_convex_routing_available(true);

    // NOTE: the convex LP/QP knobs (`qp_tau`, `qp_tau_max`, `qp_reg`,
    // `qp_gondzio_corr`,
    // `qp_infeas_tol`, `qp_hsde`, `qp_equilibrate`, `qp_crossover`) and the
    // active-set SQP QP-subproblem knobs (`sqp_qp_feas_tol`, `sqp_qp_opt_tol`,
    // `sqp_qp_max_iter`, `sqp_qp_elastic_gamma`, `sqp_qp_anti_cycling`) used to
    // be registered here. They now live in the core registry
    // (`pounce_algorithm::upstream_options`) so the library and Python paths
    // see them too — registering them here as well would raise
    // OPTION_ALREADY_REGISTERED and abort the binary at startup (gh #360 for
    // the SQP block, gh #604 for the convex one).

    // Opt into iter-history capture when the user asked for a JSON
    // report at Full detail — saves the per-iter alloc when they
    // didn't.
    if args.json_output.is_some() && matches!(args.json_detail, ReportDetail::Full) {
        app.enable_iter_history();
    }

    // Load the options file before the `key=value` overrides below, so a
    // command-line option beats a file option and not the other way round
    // — which is also why `option_file_name` has to be read off argv here
    // rather than out of the option store (upstream reads it from the
    // store at this same point, before the store has the CLI's values).
    //
    // Until gh#518 the only way in was `--options-file`: `option_file_name`
    // was refused, and the implicit `pounce.opt` / `ipopt.opt` lookup did
    // not exist, so a run configured entirely through an option file ran
    // at stock defaults and still reported success.
    let option_file_choice = match args.option_file_choice() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("pounce: {msg}");
            return ExitCode::from(2);
        }
    };
    let mut option_file_read: Option<PathBuf> = None;
    match &option_file_choice {
        pounce_cli::cli::OptionFileChoice::Suppressed => {
            if let Err(e) = app.initialize() {
                eprintln!("pounce: initialize failed: {e}");
                return ExitCode::from(2);
            }
        }
        choice => {
            let explicit = match choice {
                pounce_cli::cli::OptionFileChoice::Named(p) => Some(p.as_path()),
                _ => None,
            };
            match app.initialize_with_option_file(explicit) {
                Ok(load) => {
                    for warning in &load.warnings {
                        eprintln!("pounce: warning: {warning}");
                    }
                    option_file_read = load.path;
                }
                Err(e) => {
                    // `e.message` rather than the full `Display`: this one
                    // is read by whoever wrote the options file, and the
                    // C++-style "in file … at line …" prefix names a
                    // pounce source location, not theirs.
                    eprintln!("pounce: failed to load options file: {}", e.message);
                    return ExitCode::from(2);
                }
            }
        }
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

    // Interactive solver debugger (`--debug` / `--debug-json`). Installs
    // a hook that pauses at each iteration. In JSON mode stdout becomes a
    // pure protocol channel: the per-iteration table, banner, problem
    // stats, and final summary are all silenced (the debugger and the
    // post-solve `terminated` event carry that information instead).
    let json_dbg = matches!(args.debug, Some(pounce_cli::cli::DebugMode::Json));
    // Shared slot the debugger's `resolve` command writes to; the
    // post-solve loop below reads it to re-run with new options.
    let restart_cell: pounce_cli::debug_repl::RestartCell = Rc::new(RefCell::new(None));
    // Held across `resolve` re-solves so the SAME debugger is reused rather
    // than rebuilt — keeps its single stdin-reader thread (no leak/contention),
    // its already-sent `hello`, and its breakpoints. The `--debug-script` is
    // consumed at the first pause, so reuse won't re-run it.
    let mut debug_hook: Option<Rc<RefCell<pounce_cli::debug_repl::SolverDebugger>>> = None;
    if let Some(mode) = args.debug {
        if json_dbg {
            let _ = app.options_mut().read_from_str("print_level 0\n", true);
        }
        let reg = Some(std::rc::Rc::clone(app.registered_options()));
        let hook = Rc::new(RefCell::new(build_debugger(
            mode,
            args.debug_on_error,
            args.debug_on_interrupt,
            args.debug_script.as_deref(),
            reg,
            restart_cell.clone(),
        )));
        app.set_debug_hook(hook.clone());
        // The ladder runs whole solves the user did not step into, so it stays
        // out of the way while the interactive debugger is driving.
        app.set_second_opinion_suppressed(true);
        debug_hook = Some(hook);
        // Install the Ctrl-C → break-into-debugger handler. All debug
        // modes are interruptible; this only changes Ctrl-C behavior
        // when a debugger is active.
        pounce_cli::debug_repl::interrupt::install();
        // Branded open banner (human REPL only).
        pounce_cli::debug_repl::print_open_banner(mode);
        let extra = if args.debug_on_error {
            ", on-error"
        } else if args.debug_on_interrupt {
            ", on-interrupt"
        } else {
            ""
        };
        eprintln!(
            "pounce: interactive debugger enabled ({}{}). Type `help` at the prompt; Ctrl-C breaks in.",
            match mode {
                pounce_cli::cli::DebugMode::Repl => "repl",
                pounce_cli::cli::DebugMode::Json => "json",
            },
            extra
        );
    }

    // Wire the restoration phase. Without this, any line-search failure
    // surfaces as `RestorationFailure` instead of falling back into the
    // ℓ1-feasibility sub-IPM. Mirrors what upstream's `IpAlgBuilder` does
    // unconditionally for every solve.
    //
    // The helper resolves the FERAL config off the now-fully-loaded options,
    // so the restoration sub-IPM honours the same `feral_*` overrides (e.g.
    // `feral_cascade_break yes` from an `--options-file`) as the main IPM; it
    // installs the multi-pass provider, so the ℓ₁ wrapper, the
    // ℓ₁-on-restoration-failure retry and the second-opinion ladder do not
    // hit "restoration factory invoked more than once" on their second inner
    // solve; and it installs the mint, so a ladder rung that changes
    // `feral_scaling` rebuilds the sub-IPM instead of leaving it on the
    // settings that just failed.
    install_default_restoration(&mut app);

    // gh#483 follow-up: refuse a `linear_solver` pounce does not
    // implement. Checked here — before the banner, and before the routing
    // that would send an LP/QP to `pounce-convex` without ever reaching
    // `optimize_tnlp`'s copy of this guard — so the verdict does not
    // depend on which engine the problem happens to classify into.
    if let Some(value) = app.unimplemented_linear_solver() {
        eprintln!(
            "{}",
            IpoptApplication::unimplemented_linear_solver_message(&value)
        );
        return ExitCode::from(2);
    }
    // Same treatment for every other option naming a feature pounce does
    // not implement, and the same reason for checking here: a model that
    // routes to `pounce-convex` never reaches `optimize_tnlp`, where the
    // library-side copy of this guard lives.
    if let Some(msg) = app
        .unimplemented_option_refusal()
        .or_else(|| app.unimplemented_option_value_refusal())
    {
        eprintln!("{msg}");
        return ExitCode::from(2);
    }
    // Knobs for a linear-solver backend pounce does not ship warn here
    // rather than refusing: an `ipopt.opt` that configures several
    // backends so one file runs everywhere is the compatibility the
    // registry exists to provide, and failing it over knobs this run
    // never touches would cost more than the silence did (gh#551).
    // `take_*`, not the plain getter: `optimize_tnlp` emits the same
    // warnings for every frontend that never passes through here, and a
    // CLI run reaches both sites. Printing the paragraph twice is how a
    // warning teaches its reader to skip warnings.
    let backend_warnings = app.take_unimplemented_backend_warnings();
    for warning in app
        .unexploited_hint_warnings()
        .into_iter()
        .chain(backend_warnings)
    {
        eprintln!("{warning}");
    }

    // gh#551 / gh#677: the sIPOPT keys (`run_sens`, `compute_red_hessian`,
    // `rh_eigendecomp`, `sens_boundcheck`, `sens_bound_eps`,
    // `sens_max_pdpert`) are registered so an sIPOPT `ipopt.opt` parses,
    // and until this read site existed setting one did nothing at all —
    // the same post-optimal work was reachable only through the `--*`
    // flags below. Read once, here, so the option and the flag agree
    // everywhere the request is consulted. Each option only ADDS to what
    // the flags asked for, except `run_sens=no` (upstream's spelling of
    // "do not take the step") — see `SensOptionOverrides` for why the
    // reader reports "explicitly set" rather than resolved defaults.
    let sens_options = pounce_sensitivity::SensOptionOverrides::from_options_list(app.options());

    // Branded logo + copyright banner, printed up-front — before the
    // problem is even read — so they head the output. `sb yes` suppresses
    // both (mirrors upstream `IpoptApplication::Initialize`).
    //
    // The registered default is `feral`, so the option's resolved value is
    // the whole story and the "was it explicitly set?" flag this used to
    // consult is no longer needed: `ma57` here always means someone asked
    // for it. (Under upstream's `ma57` default it did not, and the banner
    // would otherwise have claimed "ma57 requested" on every run.)
    let backend_tag = {
        let (v, _) = app
            .options()
            .get_string_value("linear_solver", "")
            .unwrap_or_else(|_| ("feral".to_string(), false));
        if v.eq_ignore_ascii_case("ma57") {
            #[cfg(feature = "ma57")]
            {
                "MA57 (HSL)"
            }
            #[cfg(not(feature = "ma57"))]
            {
                "FERAL (ma57 requested but not compiled)"
            }
        } else {
            "FERAL"
        }
    };
    let suppress_banner = app
        .options()
        .get_bool_value("sb", "")
        .ok()
        .and_then(|(v, f)| f.then_some(v))
        .unwrap_or(false);
    if !suppress_banner && !json_dbg {
        print::print_logo();
        print::print_banner(backend_tag);
    }
    // Which options file configured this run, on the same `sb` gate as the
    // banner (upstream prints its "Using option file" line here too). A
    // discovered file especially has to announce itself: nothing on the
    // command line hints that a `pounce.opt` sitting in the working
    // directory is steering the solve.
    if let Some(path) = &option_file_read {
        if !suppress_banner && !json_dbg {
            println!("Using option file \"{}\".\n", path.display());
        }
    }

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

    // Load the problem. For `.nl` inputs, keep the parsed suffixes and
    // dimensions around: the sIPOPT-style suffixes (`sens_state_1` …)
    // drive the post-optimal sensitivity step below, and they must be
    // read off `NlProblem` before `NlTnlp` consumes it.
    let mut nl_suffixes: Option<nl_reader::NlSuffixes> = None;
    let mut nl_dims: Option<(usize, usize)> = None;
    // The model's own AMPL option words, echoed back in the `.sol`
    // `Options` block the way an ASL solver does. Empty for problems
    // that did not come from a `.nl` header.
    let mut nl_ampl_options: Vec<i64> = Vec::new();
    // Problem class captured from the *first* `.nl` parse below, so the
    // LP/QP dispatch never has to re-read the file just to classify it
    // (re-parsing doubled parse time / peak memory on large models — code
    // review L24). `None` for builtins (treated as general NLP).
    let mut nl_class: Option<pounce_cli::dispatch::ProblemClass> = None;
    // gh #703: set alongside `nl_class` when curvature-based scaling is
    // switched on below, and only then. Records whether the model handed the
    // scheme any second-order coefficient to work with — see
    // `decline_convex_for_curvature_scaling` for why the answer decides
    // whether the request is worth the convex fast path.
    let mut nl_curvature_read_curvature = false;
    // `nl_expr_provider` shadows `inner_tnlp` for the `.nl`-file path:
    // both point at the same `NlTnlp`, but the second handle is typed
    // as `dyn ExpressionProvider` so the presolve wrapper can use it
    // for FBBT (issue #62). For built-in problems we leave it `None`.
    let mut nl_expr_provider: Option<
        Rc<RefCell<dyn pounce_nlp::expression_provider::ExpressionProvider>>,
    > = None;
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
            if !json_dbg {
                println!("Reading {}...", path.display());
            }
            let t0 = std::time::Instant::now();
            match nl_reader::read_nl_file(path) {
                Ok(prob) => {
                    nl_suffixes = Some(prob.suffixes.clone());
                    nl_dims = Some((prob.n, prob.m));
                    nl_ampl_options = prob.ampl_options.clone();
                    let elapsed = t0.elapsed().as_secs_f64();
                    // Render the source constraint equations and hand them to
                    // the debugger so `print equation <name|row>` can show a
                    // culprit constraint's algebra — the named-equation
                    // diagnostic of Lee et al. (2024,
                    // https://doi.org/10.69997/sct.147875). Built before
                    // `NlTnlp::new` moves `prob`.
                    if let Some(hook) = debug_hook.as_ref() {
                        let book = pounce_cli::debug_repl::EquationBook::new(
                            prob.con_names.clone(),
                            nl_reader::render_all_constraint_equations(&prob),
                        );
                        // Structural rank analysis of the equality Jacobian
                        // (Dulmage–Mendelsohn) so `diagnose` can name the
                        // dependent equations behind a singular system —
                        // Lee et al. (2024,
                        // https://doi.org/10.69997/sct.147875).
                        let (jac_irow, jac_jcol) = nl_reader::constraint_jacobian_sparsity(&prob);
                        let probe = pounce_presolve::incidence::ProbeView {
                            n_vars: prob.n,
                            m_rows: prob.m,
                            jac_irow: &jac_irow,
                            jac_jcol: &jac_jcol,
                            jac_values: None,
                            g_l: &prob.g_l,
                            g_u: &prob.g_u,
                            linearity: None,
                            one_based: false,
                            eq_tol: 1e-12,
                            excluded_vars: None,
                            excluded_rows: None,
                        };
                        let inc = pounce_presolve::incidence::EqualityIncidence::from_probe(&probe);
                        let structure = pounce_cli::debug_repl::StructureBook::new(
                            inc,
                            prob.con_names.clone(),
                            prob.var_names.clone(),
                        );
                        let mut h = hook.borrow_mut();
                        h.set_equation_book(book);
                        h.set_structure_book(structure);
                    }
                    // Classify now, while we still own `prob` (it's about to
                    // be moved into `NlTnlp`). Saves a second full parse in the
                    // LP/QP dispatch block below.
                    nl_class = Some(pounce_cli::dispatch::classify_problem(&prob));
                    let nl_rc = Rc::new(RefCell::new(nl_reader::NlTnlp::new(prob)));
                    // gh #703: `nlp_scaling_method=curvature-based` derives
                    // its factors from the model's coefficients, so it has
                    // to be switched on here — while the handle is still
                    // the concrete `NlTnlp` that owns them, and before any
                    // wrapper (presolve, penalty, the variable-scaling
                    // substitution itself) sits in front of it. The
                    // wrappers forward `get_scaling_parameters` and project
                    // the indices, so the factors reach the engine through
                    // the channel user factors already use.
                    if curvature_scaling_requested(&app)
                        && !nl_rc.borrow_mut().enable_curvature_scaling()
                    {
                        eprintln!(
                            "pounce: nlp_scaling_method=curvature-based needs \
                             every row and the objective to be degree <= 2 (it \
                             scales a model by its quadratic coefficients, and \
                             a genuine nonlinearity has none). This model has \
                             at least one row it cannot read that way. Use \
                             gradient-based, or user-scaling with your own \
                             scaling_factor suffixes."
                        );
                        return ExitCode::from(2);
                    }
                    nl_curvature_read_curvature = nl_rc.borrow().curvature_scaling_read_curvature();
                    nl_expr_provider = Some(Rc::clone(&nl_rc)
                        as Rc<RefCell<dyn pounce_nlp::expression_provider::ExpressionProvider>>);
                    let t: Rc<RefCell<dyn TNLP>> = nl_rc;
                    if let Some(info) = t.borrow_mut().get_nlp_info() {
                        if !json_dbg {
                            println!(
                                "Parsed {} vars, {} cons, jac_nnz={}, h_nnz={} in {:.2}s",
                                info.n, info.m, info.nnz_jac_g, info.nnz_h_lag, elapsed
                            );
                        }
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

    // issue #196 (and related): does the .nl / CLI request post-optimal work
    // that only the general NLP filter-IPM path provides — the sIPOPT
    // parametric sensitivity step (sens_* suffixes) or a reduced-Hessian
    // computation (--compute-red-hessian)? Neither the --minima multistart
    // driver nor the specialized convex solvers run it, so detect it up front
    // and make sure no path silently drops the request.
    let declares_sens_suffixes = nl_suffixes
        .as_ref()
        .map(sens::is_sensitivity_input)
        .unwrap_or(false);
    // `run_sens=no` is the one option that takes work away: it is how
    // upstream says "solve, but do not take the sensitivity step", and
    // without it a `.nl` carrying the suffixes has no off switch.
    let wants_sens = declares_sens_suffixes && !sens_options.suppresses_sens_step();
    // `compute_red_hessian=yes` reaches the same computation as
    // `--compute-red-hessian`; `rh_eigendecomp=yes` implies it, exactly
    // as `--rh-eigendecomp` does.
    let wants_red_hessian = args.compute_red_hessian || sens_options.wants_reduced_hessian();
    let wants_nlp_postopt = wants_sens || wants_red_hessian;
    // gh#483: does the run ask for user NLP scaling — `nlp_scaling_method=
    // user-scaling` together with at least one `scaling_factor` suffix in the
    // `.nl` for the solver to read? Only the general NLP path implements the
    // scaling callback, so this gates the same "reroute or warn" treatment
    // the post-optimal request gets, rather than the option quietly meaning
    // "no scaling" on a specialized path.
    let wants_user_scaling = app
        .options()
        .get_string_value("nlp_scaling_method", "")
        .ok()
        .and_then(|(v, set)| set.then_some(v))
        .is_some_and(|v| v == "user-scaling")
        && nl_suffixes.as_ref().is_some_and(|s| {
            s.obj_real.contains_key("scaling_factor")
                || s.con_real.contains_key("scaling_factor")
                || s.var_real.contains_key("scaling_factor")
        });
    // gh #703, gh#483 again: `nlp_scaling_method=curvature-based` reaches the
    // engine through exactly the same channel as `user-scaling` — the TNLP's
    // `get_scaling_parameters` callback — and the convex solvers do not call
    // it. They equilibrate internally, so routing a curvature-scaling request
    // there accepts the option and means "not this scaling".
    //
    // That is not a corner case for *this* option: the models it is defined
    // for are the models with quadratic rows, which is precisely the
    // population `classify_problem` sends to the convex path. Both fixtures
    // gh #703 added are convex QCQPs, and of the 47 corpus models the option
    // accepts, 38 classify convex. Without this gate the headline feature is
    // inert by default on the majority of the models it exists for, which is
    // the gh#483 failure verbatim.
    let wants_curvature_scaling = curvature_scaling_requested(&app);
    // gh#483 follow-up: `obj_scaling_factor` is an NLP-path knob — the convex
    // solvers run their own equilibration and never read it. A *negative*
    // factor is upstream's documented spelling for "maximize", so dropping it
    // does not merely leave the conditioning alone: the convex path minimizes
    // an objective the user asked to maximize and reports the wrong optimum
    // with no complaint. (`min (x−3)²` over `x ∈ [0,1]` with
    // `obj_scaling_factor=-1` returned `x = 1`, the minimizer, instead of
    // `x = 0`.) A *positive* factor is genuinely inert on that path — it
    // reports natural units already, so both paths give the same answer — and
    // is deliberately not treated as a conflict.
    //
    // There are *two* channels into the same sign flip, and the guard has to
    // watch both. The option is one. The other is the `.nl`'s objective
    // `scaling_factor` suffix under `nlp_scaling_method=user-scaling`:
    // `scale_user_supplied` installs it as `df` with no sign guard, so a
    // negative entry maximizes exactly as the option does. Watching only the
    // option left `scaling_factor[obj] = -1` plus a forced convex solver
    // returning the minimizer with an "the requested scaling will be skipped"
    // warning — which understates it, since what is skipped is the objective
    // sense, not conditioning. Found by adversarial testing of this guard.
    let negative_obj_scaling_option = app
        .options()
        .get_numeric_value("obj_scaling_factor", "")
        .ok()
        .and_then(|(v, set)| set.then_some(v))
        .is_some_and(|v| v < 0.0);
    let negative_obj_scaling_suffix = wants_user_scaling
        && nl_suffixes.as_ref().is_some_and(|s| {
            s.obj_real
                .get("scaling_factor")
                .and_then(|v| v.first())
                .is_some_and(|&f| f < 0.0)
        });
    let maximize_via_obj_scaling = negative_obj_scaling_option || negative_obj_scaling_suffix;
    // Human-readable description of the requested post-optimal work, reused in
    // the "not available on this path" messages below.
    let postopt_what = match (wants_sens, wants_red_hessian) {
        (true, true) => {
            "a parametric sensitivity step (sIPOPT sens_* suffixes) and a \
             reduced-Hessian computation"
        }
        (true, false) => "a parametric sensitivity step (sIPOPT sens_* suffixes)",
        _ => "a reduced-Hessian computation",
    };

    // Multistart / find-minima: when a `--minima` method is set, drive the
    // local solver in a loop over the *raw* problem TNLP (presolve / counting
    // wrappers are intentionally bypassed so coordinates match the original
    // problem and the clean objective is evaluated directly) and return.
    if let Some(mcfg) = &args.minima {
        // Related to #196: --minima is a multistart search, not a single
        // post-optimal solve, so it does not run the sIPOPT sensitivity /
        // reduced-Hessian step. Warn rather than silently drop the request
        // (sensitivity at a multistart optimum is ill-defined).
        if wants_nlp_postopt {
            eprintln!(
                "pounce: warning: the .nl requests {postopt_what}, but --minima \
                 runs a multistart search that does not compute it; the request \
                 will be skipped. Run without --minima to obtain it."
            );
        }
        app.set_presolve_already_applied(true);
        return pounce_cli::minima::run(&mut app, &inner_tnlp, mcfg, &args, sol_path.as_deref());
    }

    // LP/QP routing (Phase 1). Resolve the `solver_selection` option
    // against the detected problem class. For `.nl` inputs we classify
    // the parsed problem; for builtins we conservatively treat the class
    // as NLP (they are general nonlinear test problems). `auto`/`nlp`
    // both route to the existing solver — the only observable effect in
    // Phase 1 is that an explicit forcing value (e.g. `--solver=lp`)
    // that does not match the detected class is rejected with a clear
    // message, instead of being silently ignored.
    {
        use pounce_cli::dispatch::{ProblemClass, SolverChoice, SolverSelection, resolve_solver};
        let sel_str = app
            .options()
            .get_string_value("solver_selection", "")
            .map(|(v, _)| v)
            .unwrap_or_else(|_| "auto".to_string());
        let selection = match SolverSelection::parse(&sel_str) {
            Some(s) => s,
            None => {
                eprintln!(
                    "pounce: invalid solver_selection '{sel_str}'; valid values: {}",
                    SolverSelection::VALUES.join(", ")
                );
                return ExitCode::from(2);
            }
        };

        // Problem class. The `.nl` path was already classified during the
        // initial parse above (`nl_class`) — we do NOT re-read the file here
        // (re-parsing doubled parse time / peak memory on large models, and
        // its error arm silently fell back to NLP; code review L24). Builtins
        // are treated as general NLP.
        let class = match &args.problem {
            ProblemSource::NlFile(_) => nl_class.unwrap_or(ProblemClass::Nlp),
            ProblemSource::Builtin(_) => ProblemClass::Nlp,
        };

        let choice = match resolve_solver(class, selection) {
            Ok(c) => c,
            Err(msg) => {
                eprintln!("pounce: {msg}");
                return ExitCode::from(2);
            }
        };

        // issue #196: `wants_sens` / `wants_nlp_postopt` / `postopt_what` were
        // computed above (they also gate the --minima warning). Under `auto`,
        // decline the convex fast-path and fall through to the NLP filter-IPM
        // (which honors the request — correctness over the specialized path's
        // speed); under an explicit convex solver_selection, respect the forced
        // choice but warn (below) instead of silently skipping.
        let decline_convex_for_postopt =
            wants_nlp_postopt && matches!(selection, SolverSelection::Auto);

        // gh#483: same bargain for user NLP scaling. `nlp_scaling_method=
        // user-scaling` plus the `.nl`'s `scaling_factor` suffixes is honored
        // by the general NLP interior-point path only — the convex solvers run
        // their own internal equilibration and never see the TNLP's scaling
        // callback, so routing there would accept the option and mean "none".
        let decline_convex_for_user_scaling =
            wants_user_scaling && matches!(selection, SolverSelection::Auto);

        // gh #703: and again for curvature-based scaling, for the same reason
        // and through the same callback — but only when the model handed the
        // scheme some curvature to read.
        //
        // The `user-scaling` gate above already has this shape: it requires
        // the option *and* a `scaling_factor` suffix in the `.nl` for the
        // solver to read, because rerouting a model that carries no suffixes
        // buys a different engine and nothing else. The same test applies
        // here, and it is not cosmetic. Leaving it out was measured on the
        // corpus: `nlp_scaling_method=curvature-based` on `lp_israel`, a pure
        // LP, went from 29 iterations on the convex path to 296 on the
        // general one — 29 → 135 for the engine and 135 → 296 for a scaling
        // scheme whose defining input, `Qᵢ`, is empty in every row. On an LP
        // §8 degenerates to Ruiz equilibration of `[A b]`, which is what
        // `pounce-convex` already does internally, so the fast path is not
        // "quietly meaning none" here — it is doing the same kind of work by
        // its own route. That is a claim worth making out loud rather than
        // silently, so the no-curvature case still prints (below); what it
        // does not do is pay for an engine switch that cannot help.
        let decline_convex_for_curvature_scaling = wants_curvature_scaling
            && nl_curvature_read_curvature
            && matches!(selection, SolverSelection::Auto);

        // gh#483 follow-up: a negative `obj_scaling_factor` means maximize,
        // which the convex path cannot express. Unlike the two requests above
        // — where the fast path merely skips *extra* work — taking it here
        // returns the wrong optimum, so under an explicit `solver_selection`
        // this is refused outright below rather than warned about.
        let decline_convex_for_obj_scaling =
            maximize_via_obj_scaling && matches!(selection, SolverSelection::Auto);
        // Any of these declines the fast path; the messages below say which.
        let decline_convex = decline_convex_for_postopt
            || decline_convex_for_user_scaling
            || decline_convex_for_curvature_scaling
            || decline_convex_for_obj_scaling;

        // Same bargain for a conic solve that finishes without a verified KKT
        // point: under `auto` the class was our inference, not the user's
        // instruction, so fall through to the NLP filter-IPM (a convex QCQP is
        // also a valid NLP) rather than reporting a failure the general path
        // can solve. Under an explicit `solver_selection` the forced choice is
        // respected and the conic verdict stands — the user asked for that
        // engine and silently answering from a different one would hide it.
        let socp_nlp_fallback = matches!(selection, SolverSelection::Auto);

        // gh #535: and the same bargain again for an LP the convex IPM cannot
        // certify. `auto` for the same reason as above — the LP classification
        // was our inference, so a failure to certify it is ours to fix, while a
        // named engine keeps its verdict. Additionally suppressed when the user
        // set `max_iter` (their budget is the question being answered, and
        // `max_iter=0` must stop without a solve, pounce#186) and when the
        // interactive debugger is attached (the user is stepping *this* engine;
        // silently continuing into a different one would strand the session).
        // See `lp_declines_to_nlp` for the rest of the gating.
        let lp_nlp_fallback = matches!(selection, SolverSelection::Auto)
            && debug_hook.is_none()
            && !max_iter_explicitly_set(&app);

        // Banner-level routing line: report the detected problem class and
        // which of pounce's solvers was selected for it. Gated like the
        // banner (suppressed by `sb yes` and in JSON-debug protocol mode) so
        // stdout stays clean for machine consumers. When we decline the convex
        // fast-path for a post-optimal request (#196), report the NLP path that
        // actually runs, not the convex one `resolve_solver` picked.
        if !suppress_banner && !json_dbg {
            let described = if decline_convex {
                SolverChoice::Nlp.describe()
            } else {
                choice.describe()
            };
            println!(
                "Problem class: {}. Selected solver: {} [solver_selection={}].",
                class.name(),
                described,
                sel_str
            );
            println!();
        }

        // Dispatch to the specialized convex solvers when resolved.
        // `LpIpm`/`QpIpm` use the convex QP IPM (LP is P = 0); `SocpIpm`
        // reformulates a convex QCQP to second-order cones and uses the
        // conic IPM. Both live in `pounce-convex`.
        //
        // `QpActiveSet` joins them here rather than routing through the SQP
        // outer loop as it used to. The engine is different, but everything
        // wrapped around it — QP extraction, presolve, postsolve, `.sol`
        // writing, status vocabulary, timing — is shared with the IPM, and
        // that shared wrapper is the entire point: the active-set engine had
        // been running with no presolve and no scaling, which costs an
        // active-set method far more than it costs an IPM (its pivot count
        // grows with the problem, an IPM's essentially does not). See
        // `pounce_convex::active_set` for the full rationale.
        if matches!(
            choice,
            SolverChoice::LpIpm
                | SolverChoice::QpIpm
                | SolverChoice::SocpIpm
                | SolverChoice::QpActiveSet
        ) {
            // gh#483 follow-up: `derivative_test` is about the *model*,
            // not the engine, so on the convex route it is run here rather
            // than declined — this dispatch never reaches `optimize_tnlp`,
            // where the NLP path's copy lives. Checking the raw
            // `inner_tnlp` keeps the report in the user's own indices, and
            // running it here (not there) means it cannot fire twice.
            app.run_derivative_test(&inner_tnlp);
            // Same reason, same place: `install_constant_derivative_hints`
            // lives behind `optimize_tnlp`, so on this route the four
            // constant-derivative hints are read by nothing. gh #588 Q6
            // emptied the NLP path's unexploited-hint table — correctly,
            // it exploits them now — and that silenced the convex route's
            // warning too. Say it here, where the route is known.
            for warning in app.convex_unexploited_hint_warnings() {
                eprintln!("{warning}");
            }
            // gh#483 follow-up: a forced convex solver plus a negative
            // `obj_scaling_factor` has no honest outcome — the engine cannot
            // maximize, and running it anyway hands back the minimizer of the
            // problem the user asked to maximize. Refuse, the way a
            // class/solver mismatch is refused, instead of warning and
            // returning a wrong answer.
            if maximize_via_obj_scaling && !decline_convex_for_obj_scaling {
                eprintln!(
                    "pounce: the objective scaling is negative (maximize) — via \
                     obj_scaling_factor or the .nl's `scaling_factor` suffix — \
                     but solver_selection={sel_str} forces the convex solver \
                     (pounce-convex), which minimizes and does not read that \
                     option — it would report the minimizer of the objective \
                     you asked to maximize. Use solver_selection=nlp or auto \
                     (which routes here automatically), or negate the \
                     objective in the model and drop obj_scaling_factor."
                );
                return ExitCode::from(2);
            }
            // issue #196: if the .nl requested a sensitivity / reduced-Hessian
            // step, either reroute (auto) or warn (explicit convex force) so
            // the fast path never silently drops it.
            if wants_nlp_postopt {
                if decline_convex_for_postopt {
                    eprintln!(
                        "pounce: note: this problem classifies as {} but the .nl \
                         requests {postopt_what}, which the convex solver \
                         (pounce-convex) does not provide; routing to the general \
                         NLP interior-point path so the request is honored.",
                        class.name()
                    );
                } else {
                    eprintln!(
                        "pounce: warning: the .nl requests {postopt_what}, but \
                         solver_selection={sel_str} forces the convex solver \
                         (pounce-convex), which does not compute it; the request \
                         will be skipped. Use solver_selection=nlp or auto to \
                         obtain it."
                    );
                }
            }
            // gh#483: same treatment for `nlp_scaling_method=user-scaling`.
            if wants_user_scaling {
                if decline_convex_for_user_scaling {
                    eprintln!(
                        "pounce: note: this problem classifies as {} but \
                         nlp_scaling_method=user-scaling asks for the .nl's \
                         `scaling_factor` suffixes to be applied, which the \
                         convex solver (pounce-convex) does not do; routing to \
                         the general NLP interior-point path so the scaling is \
                         honored.",
                        class.name()
                    );
                } else {
                    eprintln!(
                        "pounce: warning: nlp_scaling_method=user-scaling asks \
                         for the .nl's `scaling_factor` suffixes to be applied, \
                         but solver_selection={sel_str} forces the convex solver \
                         (pounce-convex), which equilibrates internally and does \
                         not read them; the requested scaling will be skipped. \
                         Use solver_selection=nlp or auto to apply it."
                    );
                }
            }
            // gh #703: same treatment for `nlp_scaling_method=curvature-based`,
            // with a third case the other requests do not have — a model that
            // is degree <= 2 (so the option was accepted) but carries no
            // second-order coefficient at all.
            if wants_curvature_scaling {
                if decline_convex_for_curvature_scaling {
                    eprintln!(
                        "pounce: note: this problem classifies as {} but \
                         nlp_scaling_method=curvature-based asks for factors \
                         derived from the model's quadratic coefficients, which \
                         the convex solver (pounce-convex) does not read; \
                         routing to the general NLP interior-point path so the \
                         scaling is honored.",
                        class.name()
                    );
                } else if !nl_curvature_read_curvature {
                    eprintln!(
                        "pounce: note: nlp_scaling_method=curvature-based was \
                         accepted, but every quadratic coefficient in this \
                         model is zero, so the scheme reduces to Ruiz \
                         equilibration of the linear rows; the convex solver \
                         (pounce-convex) equilibrates internally and keeps the \
                         fast path. Use solver_selection=nlp to run the scheme \
                         on the general path anyway."
                    );
                } else {
                    eprintln!(
                        "pounce: warning: nlp_scaling_method=curvature-based \
                         asks for factors derived from the model's quadratic \
                         coefficients, but solver_selection={sel_str} forces \
                         the convex solver (pounce-convex), which equilibrates \
                         internally and does not read them; the requested \
                         scaling will be skipped. Use solver_selection=nlp or \
                         auto to apply it."
                    );
                }
            }
            // gh#483 follow-up: the auto-reroute half of the negative
            // `obj_scaling_factor` case (the forced half exited above).
            if decline_convex_for_obj_scaling {
                eprintln!(
                    "pounce: note: this problem classifies as {} but \
                     obj_scaling_factor is negative (maximize), which the \
                     convex solver (pounce-convex) cannot express; routing to \
                     the general NLP interior-point path so the objective \
                     sense is honored.",
                    class.name()
                );
            }
            // The convex solvers need the parsed `NlProblem`, but the initial
            // parse moved it into `NlTnlp`. Re-parse the file here — only on
            // the convex dispatch path (LP / convex-QP / SOCP), never for a
            // general NLP solve. Only `.nl` inputs ever classify as convex, so
            // the builtin arm falls through to NLP. A parse failure surfaces
            // and exits rather than silently mis-routing to NLP (L24).
            if decline_convex {
                // Declined for #196 / gh#483: fall through to the NLP solve
                // below, which runs the sensitivity / reduced-Hessian step in
                // `on_converged` (writing `sens_sol_state_1` to the `.sol`) and
                // reads the `scaling_factor` suffixes through the TNLP scaling
                // callback.
            } else if let ProblemSource::NlFile(path) = &args.problem {
                let prob = match nl_reader::read_nl_file(path) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!(
                            "pounce: failed to re-read {} for the convex solver: {e}",
                            path.display()
                        );
                        return ExitCode::from(2);
                    }
                };
                // JSON solve report, when requested — same schema as the NLP
                // path, so the benchmark harness can compare convex and NLP
                // solves.
                let json_cfg = args.json_output.as_deref().map(|p| {
                    let input = InputDescriptor::NlFile {
                        path: path.clone(),
                        size_bytes: std::fs::metadata(path).ok().map(|m| m.len()),
                    };
                    (p, args.json_detail, input)
                });
                // Materialize the convex controls in pounce-convex, which owns
                // their typed representation and precedence rules. In
                // particular, unset shared NLP options must not replace the
                // convex driver's independently tuned defaults.
                let convex_opts =
                    match pounce_convex::QpOptions::try_from_options_list(app.options()) {
                        Ok(options) => options,
                        Err(error) => {
                            eprintln!("pounce: convex option setup failed: {error}");
                            return ExitCode::from(2);
                        }
                    };
                // gh #744/#745: the same `bound_relax_factor` widening the NLP
                // path applies, so both arms solve one model.
                let bound_relax = convex_bound_relax(&app);
                // When the convex attempt declines (gh #535 / `socp_nlp_fallback`)
                // the NLP solve below opens its own `Deadline` from the *option
                // value*, which still names the full budget — so a run that spent
                // most of `max_wall_time` here would be granted it again there,
                // and the user's cap would buy up to twice the wall clock they
                // asked for. Charge the declined attempt against the budget; see
                // the deduction below the block.
                let convex_t0 = std::time::Instant::now();
                // Resolve the convex-path presolve switch (#139) once, above
                // the driver split: both convex drivers honour it.
                let presolve_on = match pounce_convex::ConvexPresolveOptions::try_from_options_list(
                    app.options(),
                ) {
                    Ok(options) => options.enabled,
                    Err(error) => {
                        eprintln!("pounce: convex presolve setup failed: {error}");
                        return ExitCode::from(2);
                    }
                };
                if matches!(choice, SolverChoice::SocpIpm) {
                    // `None` means the conic solve came back without a verified
                    // KKT point and declined the problem (only possible under
                    // `auto` — see `socp_nlp_fallback`). It printed and wrote
                    // nothing, so control falls out of this whole block to the
                    // NLP solve below, which produces the one and only verdict.
                    if let Some(code) = run_convex_socp(
                        &prob,
                        class,
                        sol_path.as_deref(),
                        json_cfg,
                        debug_hook.as_ref(),
                        args.ampl,
                        convex_opts,
                        bound_relax,
                        presolve_on,
                        socp_nlp_fallback,
                    ) {
                        return code;
                    }
                } else {
                    // The interactive debugger is a pdb-for-the-IPM: it pauses on
                    // barrier-IPM iterations (mu, search direction, fraction-to-
                    // the-boundary). The active-set engine is a different
                    // algorithm with no such hook, so a `--debug*` request would
                    // otherwise silently no-op. Say so explicitly.
                    // Forward the `sqp_qp_*` family to the inner engine. Only
                    // explicit settings materialize as overrides, leaving the
                    // direct driver's tuned defaults intact otherwise.
                    let engine_overrides =
                        match pounce_convex::ActiveSetOverrides::try_from_options_list(
                            app.options(),
                        ) {
                            Ok(options) => options,
                            Err(error) => {
                                eprintln!("pounce: active-set option setup failed: {error}");
                                return ExitCode::from(2);
                            }
                        };
                    if matches!(choice, SolverChoice::QpActiveSet) && debug_hook.is_some() {
                        eprintln!(
                            "pounce: note: the interactive debugger is IPM-only and does \
                             not engage on the active-set QP engine (solver_selection=\
                             qp-active-set); the solve runs without pausing. Use \
                             solver_selection=qp-ipm to debug a convex QP interactively."
                        );
                    }
                    // `None` means the convex solve finished an LP without a
                    // certificate and declined it (gh #535, `auto` only — see
                    // `lp_nlp_fallback`). It printed no verdict and wrote no
                    // `.sol`/JSON, so control falls out of this whole block to
                    // the NLP solve below, which owns the one verdict.
                    if let Some(code) = run_convex_qp(
                        &prob,
                        class,
                        sol_path.as_deref(),
                        presolve_on,
                        json_cfg,
                        debug_hook.as_ref(),
                        args.ampl,
                        convex_opts,
                        bound_relax,
                        matches!(choice, SolverChoice::QpActiveSet),
                        engine_overrides,
                        lp_nlp_fallback,
                    ) {
                        return code;
                    }
                }
                // Reaching here means the convex attempt declined and the NLP
                // path below owns the verdict; charge it for the time it spent.
                charge_wall_budget(app.options_mut(), convex_t0.elapsed());
            }
            // Builtins never classify as convex; fall through to NLP.
        }
        // `qp-active-set` no longer lands here: it is dispatched with the
        // other convex engines above, straight into `pounce-qp` via
        // `pounce_convex::active_set`, rather than being rewritten to
        // `algorithm=active-set-sqp` and run through the SQP outer loop.
        // Wrapping a QP in an SQP was never wrong — with an exact Hessian and
        // already-linear constraints the first subproblem *is* the original QP
        // — but it forfeited the convex path's presolve, scaling, timing, and
        // status vocabulary in exchange for machinery a QP has no use for.
        // The SQP route remains for genuine NLPs via `algorithm=active-set-sqp`,
        // where the outer loop is doing real work.
        //
        // `nlp` and any unmatched case fall through to the existing NLP
        // solve below unchanged.
        let _ = choice;
    }

    // Does the `.nl` ask for a parametric sensitivity step? When it
    // does, the post-optimal step runs inside `on_converged` below and
    // its result is written back as the `sens_sol_state_1` suffix.
    // `run_sens=no` turns that off (gh#551); `run_sens=yes` cannot turn
    // it *on* — the perturbation itself is declared by the suffixes and
    // there is nothing to step without them, so say so rather than
    // solve and silently report nothing.
    let sens_active = wants_sens;
    if sens_options.run_sens == Some(true) && !declares_sens_suffixes {
        eprintln!(
            "pounce: warning: `run_sens=yes` asks for a parametric sensitivity \
             step, but the input declares none of the sIPOPT suffixes \
             (sens_state_1, sens_state_value_1, sens_init_constr) that say \
             which parameter to perturb; no step will be computed."
        );
    }
    if sens_options.suppresses_sens_step() && declares_sens_suffixes {
        eprintln!(
            "pounce: `run_sens=no` — solving without the parametric \
             sensitivity step the input's sIPOPT suffixes ask for."
        );
    }

    // Capture the converged primal / dual into `nominal_capture` so the
    // JSON report and `.sol` below can ship `solution.x` /
    // `solution.lambda`. The same callback runs the suffix-driven
    // post-processing: the parametric sensitivity step
    // (`sens_sol_state_1`) and the reduced-Hessian computation.
    let nominal_capture: Rc<
        RefCell<
            Option<(
                Vec<pounce_common::types::Number>,
                Vec<pounce_common::types::Number>,
            )>,
        >,
    > = Rc::new(RefCell::new(None));
    let sens_capture: Rc<RefCell<Option<Vec<pounce_common::types::Number>>>> =
        Rc::new(RefCell::new(None));
    // Converged bound multipliers, lifted to full-x order and the user's
    // unscaled-Lagrangian convention (Ipopt `ipopt_zL_out`/`ipopt_zU_out`).
    // Both are `≥ 0` at an active bound; zero elsewhere. Written as `.sol`
    // suffix blocks so Pyomo's `model.ipopt_zL_out` / AMPL `.rc` are
    // populated for reduced-cost / sensitivity work (gh #296).
    let bound_mult_capture: Rc<
        RefCell<
            Option<(
                Vec<pounce_common::types::Number>,
                Vec<pounce_common::types::Number>,
            )>,
        >,
    > = Rc::new(RefCell::new(None));
    let red_hessian_capture: Rc<RefCell<Option<sens::RedHessianResult>>> =
        Rc::new(RefCell::new(None));
    if args.json_output.is_some() || sol_path.is_some() || sens_active || wants_red_hessian {
        let cap = Rc::clone(&nominal_capture);
        let sens_cap = Rc::clone(&sens_capture);
        let bmult_cap = Rc::clone(&bound_mult_capture);
        let rh_cap = Rc::clone(&red_hessian_capture);
        let suffixes_cb = nl_suffixes.clone();
        let dims_cb = nl_dims;
        let compute_rh = wants_red_hessian;
        let rh_eigen = args.rh_eigendecomp || sens_options.wants_eigendecomp();
        // `--sens-boundcheck` and `sens_boundcheck=yes` both switch the
        // refinement on. The margin: a `--sens-bound-eps` that was
        // actually typed (it carries its own value and implies the
        // flag) wins; otherwise `sens_bound_eps` from the options;
        // otherwise the registered default, which is the same 1e-3 the
        // flag defaults to — so reading the option changes nothing for
        // anyone who does not set it. `sens_bound_eps_explicit`, not a
        // comparison against the default: `--sens-bound-eps 1e-3` is a
        // real request and must still beat the options file.
        let boundcheck_eps = {
            let on = args.sens_boundcheck || sens_options.sens_boundcheck == Some(true);
            let eps = if args.sens_bound_eps_explicit {
                args.sens_bound_eps
            } else {
                sens_options
                    .sens_bound_eps
                    .unwrap_or(pounce_sensitivity::DEFAULT_SENS_BOUND_EPS)
            };
            on.then_some(eps)
        };
        // The refinement releases a bound whose multiplier the step
        // drives negative past the solve's own margin, not past
        // `sens_bound_eps`, which is a primal margin.
        let release_eps = pounce_sensitivity::release_floor_from_options(app.options());
        let sens_opts_cb = sens_options;
        app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
            let curr = match data.borrow().curr.clone() {
                Some(c) => c,
                None => return,
            };
            // Lift to full length so a fixed / eliminated variable
            // still occupies its slot — AMPL's `.sol` reader matches
            // the x block against the originating `.nl`'s var count.
            let x_iterate = nlp.borrow().lift_x_to_full(&*curr.x);
            // The `.sol` / JSON `solution.x` is the point the user is
            // *told* is the solution, so it goes through
            // `finalize_solution_x` — which adds the
            // `honor_original_bounds` projection undoing the
            // `bound_relax_factor` widening. Without it a bound-pinned
            // solution is reported just outside its own declared bounds
            // even with the option on, because this hook reads the raw
            // iterate rather than the `finalize_solution` payload.
            // `x_iterate` stays unprojected for the sensitivity /
            // reduced-Hessian steps below: those expand around the point
            // the KKT factorization was built at, and must not be handed
            // a base shifted out from under it.
            let x = nlp.borrow().finalize_solution_x(&*curr.x);
            // Reassemble the user-facing `lambda` (length `n_full_g`, in
            // original `.nl` g-row order) via `finalize_solution_lambda`, which
            // inverts the c/d split through `c_map`/`d_map`, unwinds the
            // `c_scale`/`d_scale` scaling, AND divides out `obj_scale_factor`
            // so the dual is in the user's unscaled-Lagrangian convention.
            // (`pack_lambda_for_user` omits the obj_scale division — it feeds
            // the scaled `eval_h` — so using it here left the duals scaled
            // whenever gradient-based scaling triggered: pounce#11 F1.)
            // Concatenating the raw `y_c` then `y_d` blocks here instead would
            // permute the duals on any `.nl` with interleaved eq/ineq rows and
            // leave them scaled — AMPL / Pyomo read the dual block positionally.
            let mut lambda = nlp
                .borrow()
                .finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
            if lambda.is_empty() {
                // Fallback for a non-`OrigIpoptNlp` whose `pack_lambda_for_user`
                // is the empty-vec default: emit the raw `y_c`-then-`y_d`
                // concatenation (no map/scale information available).
                let n_c = curr.y_c.dim() as usize;
                let n_d = curr.y_d.dim() as usize;
                lambda = Vec::with_capacity(n_c + n_d);
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
            }
            *cap.borrow_mut() = Some((x.clone(), lambda));

            // Lift the algorithm-side compressed bound multipliers to
            // full-x order in the user's convention. `finalize_solution_z_l`
            // /`_z_u` unwind the fixed-var/scaling maps and divide out
            // `obj_scale_factor`, yielding Ipopt-convention duals
            // (`z_l ≥ 0` at an active lower bound, `z_u ≥ 0` at an active
            // upper bound) — exactly Ipopt's `ipopt_zL_out`/`ipopt_zU_out`
            // (gh #296). A non-`OrigIpoptNlp` returns empty here and no
            // bound-multiplier suffixes are written.
            let z_l_full = nlp.borrow().finalize_solution_z_l(&*curr.z_l);
            let z_u_full = nlp.borrow().finalize_solution_z_u(&*curr.z_u);
            if !z_l_full.is_empty() || !z_u_full.is_empty() {
                *bmult_cap.borrow_mut() = Some((z_l_full, z_u_full));
            }

            // Suffix-driven post-processing on the converged KKT
            // system: the parametric sensitivity step and (on request)
            // the reduced Hessian.
            if let Some(suffixes) = &suffixes_cb {
                let (n_full, m_full) = dims_cb.unwrap_or((x.len(), 0));
                if sens_active {
                    if let Some(xp) = sens::compute_sens_perturbed_x(
                        data,
                        cq,
                        nlp,
                        Rc::clone(&pd),
                        suffixes,
                        n_full,
                        m_full,
                        &x_iterate,
                        boundcheck_eps,
                        release_eps,
                        &sens_opts_cb,
                    ) {
                        *sens_cap.borrow_mut() = Some(xp);
                    }
                }
                if compute_rh {
                    match sens::try_compute_red_hessian(
                        data,
                        cq,
                        nlp,
                        Rc::clone(&pd),
                        suffixes,
                        rh_eigen,
                        &sens_opts_cb,
                    ) {
                        Some(r) => *rh_cap.borrow_mut() = Some(r),
                        None => eprintln!(
                            "pounce: --compute-red-hessian requested but the `red_hessian` \
                             suffix is missing or empty in the input .nl"
                        ),
                    }
                }
            }
        }));
    }

    // Optionally wrap with presolve before counting so eval-call
    // counts reflect what the solver actually issues.
    let mut presolve_opts = match pounce_presolve::PresolveOptions::from_options_list(app.options())
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!("pounce: presolve setup failed: {e}");
            return ExitCode::from(2);
        }
    };
    // Sensitivity / reduced-Hessian post-processing reads the converged
    // KKT system and indexes it with suffixes defined against the
    // original `.nl`. Presolve tightens bounds and drops rows, which
    // would shift that indexing — so disable it when either is active.
    if (sens_active || wants_red_hessian) && presolve_opts.enabled {
        eprintln!(
            "pounce: disabling presolve — sensitivity / reduced-Hessian post-processing \
             operates on the original (un-presolved) KKT system"
        );
        presolve_opts.enabled = false;
    }
    let presolve_handle = if presolve_opts.enabled {
        let p = Rc::new(RefCell::new(match &nl_expr_provider {
            Some(ep) => pounce_presolve::PresolveTnlp::with_expression_provider(
                Rc::clone(&inner_tnlp),
                Rc::clone(ep),
                presolve_opts,
            ),
            None => pounce_presolve::PresolveTnlp::new(Rc::clone(&inner_tnlp), presolve_opts),
        }));
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
            if !json_dbg {
                println!(
                    "Presolve: tightened {} bounds ({} newly-finite), dropped {} redundant rows, LICQ={}",
                    tr.n_tightened, tr.n_new_finite, dropped, licq
                );
            }
            if let Some(fr) = h.fbbt_report() {
                if !json_dbg {
                    println!(
                        "Presolve FBBT: {} sweeps, {} variable tightenings (Σ|Δ|={:.3e})",
                        fr.iterations, fr.bound_updates, fr.total_tightening
                    );
                }
                if let Some(witness) = fr.infeasibility_witness {
                    eprintln!("pounce: FBBT detected infeasibility (witness constraint {witness})");
                }
            }
        }
        Some(p)
    } else {
        None
    };
    // Phase 6 (#487) stacks on top: it is the one pass that removes
    // *columns*, so it must be the outermost layer, and it is the layer the
    // `.sol` / JSON writers below read the full-space solution back out of.
    let elim_handle = match (&presolve_handle, presolve_opts.linear_eq_reduction) {
        (Some(p), true) => {
            let e = Rc::new(RefCell::new(pounce_presolve::LinearEqElimTnlp::new(
                Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
                presolve_opts,
            )));
            // Force the lazy init so the summary below reports real numbers.
            let _ = e.borrow_mut().get_nlp_info();
            if !json_dbg {
                let h = e.borrow();
                let r = h.report();
                println!(
                    "Presolve linear-equality reduction: eliminated {} columns \
                     ({} pinned, {} aggregated), dropped {} rows ({} redundant)",
                    h.n_eliminated_vars(),
                    r.n_constant_vars,
                    r.n_aggregated_vars,
                    h.n_eliminated_rows(),
                    r.n_redundant_rows,
                );
            }
            Some(e)
        }
        _ => None,
    };
    let post_presolve: Rc<RefCell<dyn TNLP>> = match (&elim_handle, &presolve_handle) {
        (Some(e), _) => Rc::clone(e) as Rc<RefCell<dyn TNLP>>,
        (None, Some(p)) => Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
        (None, None) => Rc::clone(&inner_tnlp),
    };

    // The CLI owns its explicit wrapper so `.nl` input can supply an
    // ExpressionProvider for FBBT and so it can report presolve diagnostics
    // before solving. This also keeps the sensitivity branch above un-presolved
    // when it disabled the local wrapper, without mutating the user's `presolve`
    // option.
    app.set_presolve_already_applied(true);

    // Wrap so we can pull eval-call counts out for the final summary.
    let counting = Rc::new(RefCell::new(CountingTnlp::new(Rc::clone(&post_presolve))));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&counting) as Rc<RefCell<dyn TNLP>>;

    // Problem statistics are emitted by the engine now (application.rs),
    // gated on print_level, so every frontend gets them identically (#206).
    // The branded logo + copyright banner still print up-front, before the
    // problem is read — see near the top of `run`.

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
        if !json_dbg {
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
        }
        app.set_diagnostics(Rc::clone(diag));
    }

    // Snapshot NLP dimensions before the solve so we can use them in
    // both the console summary and the JSON report. Borrowing here is
    // safe because we hold no outstanding borrow on the counting
    // wrapper yet.
    let nlp_info_snapshot = tnlp.borrow_mut().get_nlp_info();

    // Solve, with a re-solve loop: the debugger's `resolve` command stops
    // the current solve and leaves a `RestartRequest` in `restart_cell`.
    // We then apply the staged option overrides, seed the next solve from
    // the captured `x` (via `SeededTnlp`), re-install a fresh debugger,
    // and run again. Without `resolve`, this runs exactly once.
    let mut solve_tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp);
    let mut status = loop {
        let st = app.optimize_tnlp(Rc::clone(&solve_tnlp));
        let req = restart_cell.borrow_mut().take();
        let Some(req) = req else { break st };
        for (k, v) in &req.options {
            if let Err(e) = app.options_mut().read_from_str(&format!("{k} {v}\n"), true) {
                eprintln!("pounce: re-solve could not set {k}={v}: {e}");
            }
        }
        // Full primal-dual warm restart (`resolve`): install the captured
        // 8-vector iterate and turn on the warm-start initializer so the
        // duals carry over and the barrier resumes at the captured μ
        // instead of cold-restarting at `mu_init`. The primal-only path
        // (sweep / multistart, `warm == None`) leaves these off and just
        // seeds `x` through `SeededTnlp` below.
        if let Some(snap) = req.warm {
            let mu = snap.mu();
            app.set_warm_start_iterate(snap);
            let _ = app
                .options_mut()
                .read_from_str("warm_start_init_point yes\n", true);
            if mu.is_finite() && mu > 0.0 {
                let _ = app
                    .options_mut()
                    .read_from_str(&format!("warm_start_target_mu {mu}\n"), true);
            }
        }
        solve_tnlp = Rc::new(RefCell::new(pounce_nlp::seeded_tnlp::SeededTnlp::new(
            Rc::clone(&tnlp),
            req.seed_x,
        )));
        if let Some(hook) = debug_hook.as_ref() {
            // Re-arm the SAME debugger for the next solve (the hook is consumed
            // per `optimize_tnlp`). Reusing it — rather than building a fresh
            // one — preserves the stdin pump, the `hello` handshake, and any
            // breakpoints, and avoids leaking a second stdin-reader thread.
            app.set_debug_hook(hook.clone());
        }
        eprintln!(
            "pounce: re-solving from saved point with {} option override(s)…",
            req.options.len()
        );
    };

    // Snapshot the statistics from the solve whose verdict `status` currently
    // reflects. The MC64 scaling retry below runs a *second* solve into the
    // same `app`, which overwrites `app.statistics()`. On a non-promoting
    // retry we keep the original local-infeasibility verdict, so we must keep
    // the original stats too — otherwise the summary/JSON report would pair the
    // original verdict with the failed retry's iteration count / objective. We
    // adopt the retry's stats only when the retry is actually promoted (below).
    let mut solve_stats = app.statistics();

    // Local-infeasibility second-opinion ladder.
    //
    // The ladder itself now lives in `IpoptApplication::with_second_opinion`,
    // so every frontend gets the same verdict rather than only this one; see
    // that method for the two rungs and why each exists. What stays here is
    // the console reconciliation the CLI owns.
    //
    // A presolve-*certified* infeasibility needs no suppression flag:
    // `optimize_tnlp` short-circuits on the proof before it ever reaches
    // dispatch, so the ladder is never entered for one. The handle is still
    // read here for the verdict text below.
    //
    // gh #508: both solves print their own end-of-run banner, which is
    // expected and announced — but when the retry is not promoted the last
    // banner on the terminal is the retry's, while the `.sol`, the summary and
    // the JSON report all carry the original verdict. Two banners disagreeing
    // about one solve misleads a human reading the tail of the log and a
    // machine reading it the same way: `validation/p3_control.py` keeps the
    // last `EXIT:` line it sees and pairs it with the `.sol`, so it recorded a
    // status the `.sol` never held. Measured on `min (x-5)² s.t. x²+δ = 0` at
    // `tol=1e-4`: the console ended `Error in step computation.` (δ=1e-9) and
    // `Maximum Number of Iterations Exceeded.` (δ=1e-1) over a `.sol` that said
    // locally infeasible in both. Re-emitting the verdict that actually shipped
    // makes the terminal's final word the true one.
    //
    // Gated on `print_level >= 1` to match `Application::emit_end_summary`,
    // which is what printed the two banners this one arbitrates; at
    // `print_level 0` there are none to disagree.
    let presolve_certified = presolve_handle
        .as_ref()
        .and_then(|p| p.borrow().certified_infeasible());

    if app.last_second_opinion_unpromoted()
        && app
            .options()
            .get_integer_value("print_level", "")
            .map(|(v, _found)| v >= 1)
            .unwrap_or(true)
    {
        println!();
        println!("EXIT: {}", print::status_message(status));
        println!();
        println!(
            "POUNCE {}: {}",
            env!("CARGO_PKG_VERSION"),
            print::status_message(status)
        );
    }

    // The machine-readable verdict, printed exactly once per run, after every
    // path above has finished moving `status`.
    //
    // Free-form banners are not a usable status channel and the ladder is what
    // proved it. `Application::emit_end_summary` prints one `EXIT:` banner per
    // *solve*, so a laddered run prints one per rung — and a consumer that
    // scans the whole log for known phrases picks up whichever phrase it ranks
    // first, not whichever solve shipped. `benchmarks/scripts/run_nl_bench.sh`
    // ranks "Maximum Number of Iterations Exceeded" above "Converged to a point
    // of local infeasibility", so on `cresc100` — where the barrier rung hits
    // `max_iter` and the original infeasibility verdict then stands — it
    // recorded `Maximum_Iterations_Exceeded` for a run that shipped
    // `Infeasible_Problem_Detected`. Wrong status, no error, straight into
    // `BENCHMARK_REPORT.md`.
    //
    // That driver already prefers a `Status:` line and only falls back to
    // phrase-ranking because nothing ever emitted one. This is that line. It
    // carries the upstream enumerator spelling (`Infeasible_Problem_Detected`),
    // which is what CUTEst tables and the reference JSONs use, and being last
    // and unique it cannot be confused with a rung's banner.
    //
    // Gated like the banners it disambiguates: `print_level >= 1` (at 0 the
    // console is silent by request), and never under `--json-debug`, whose
    // stdout is a pure protocol channel.
    if !json_dbg
        && app
            .options()
            .get_integer_value("print_level", "")
            .map(|(v, _found)| v >= 1)
            .unwrap_or(true)
    {
        println!("Status: {}", status.upstream_name());
    }

    // `solve_stats` was snapshotted right after the solve loop and updated
    // above iff the MC64 retry was promoted, so it always matches `status`.
    let counters = counting.borrow();
    if json_dbg {
        // Pure protocol channel: emit a `terminated` lifecycle event in
        // place of the human summary, so a visual debugger gets a clean
        // end-of-session signal with the final status and stats.
        let ev = serde_json::json!({
            "event": "terminated",
            "status": format!("{status:?}"),
            "status_message": print::status_message(status),
            "iterations": solve_stats.iteration_count,
            "objective": solve_stats.final_objective,
            "evals": {
                "obj": counters.n_obj.get(),
                "obj_grad": counters.n_grad_f.get(),
                "constr": counters.n_g.get(),
                "constr_jac": counters.n_jac_g.get(),
                "hess": counters.n_h.get(),
            },
        });
        println!("{ev}");
    }
    // The console end-of-run summary is emitted by the engine now
    // (application.rs), gated on print_level; the CLI only prints the JSON
    // event variant above (#206).
    drop(counters); // release before JSON block (which re-borrows the wrapped TNLP).

    // Active-set SQP fallback: that solve path bypasses the IPM-only
    // `on_converged` hook the `.sol` / JSON writers read, so
    // `nominal_capture` is still empty even on a clean solve. Backfill it
    // from the solution `CountingTnlp` captured at `finalize_solution`
    // (original-problem space, the same `x` / `lambda` the IPM hook would
    // have recorded). Only fills when empty, so the IPM path is untouched.
    if nominal_capture.borrow().is_none() {
        if let Some(xl) = counting.borrow().captured_solution() {
            *nominal_capture.borrow_mut() = Some(xl);
        }
    }

    // Presolve row-dropping: both lambda sources above (`on_converged`
    // and the `CountingTnlp` fallback) sit *outside* presolve, so their
    // `lambda` is in the reduced kept-row space — length `m_out`, not the
    // original `.nl`'s `m`. AMPL / Pyomo read the `.sol` dual block
    // positionally against the originating `.nl`, so a short block
    // mis-aligns or is rejected. `PresolveTnlp::finalize_solution` already
    // lifted the duals back to the original row order *and* recovered
    // multipliers for the dropped rows; swap that full-length vector in.
    // Phase 6 (#487) removes columns too, so with it active every capture
    // taken outside the wrappers — `on_converged`, the `CountingTnlp`
    // fallback, the bound-multiplier suffixes — is in the reduced variable
    // space as well, and short in both directions.
    let elim_reduced = elim_handle
        .as_ref()
        .map(|e| {
            let h = e.borrow();
            h.n_eliminated_vars() > 0 || h.n_eliminated_rows() > 0
        })
        .unwrap_or(false);
    if let Some(p) = &presolve_handle {
        let lifted = if p.borrow().n_dropped_rows() > 0 || elim_reduced {
            p.borrow().finalized_full_solution()
        } else {
            None
        };
        if let Some((x_full, lam_full)) = lifted {
            if let Some((x, lambda)) = nominal_capture.borrow_mut().as_mut() {
                *lambda = lam_full;
                if elim_reduced {
                    *x = x_full;
                }
            }
        }
    }
    // Variable scaling (gh#486) is the same shape of problem as the
    // reductions above, one level further out: the `on_converged` hook
    // reads the algorithm's own iterate, and under a change of
    // variables that iterate is in scaled coordinates. `.sol` and the
    // JSON report must carry the model's own units, so undo the
    // substitution here. `finalize_solution` already did it for every
    // consumer that reads THAT payload; this fixes the ones that do not.
    //
    // The lengths are asserted rather than zipped: a `zip` against a
    // shorter factor vector would leave the tail in scaled coordinates
    // and report it as though it were in the model's units, which no
    // reader could detect. Both captures come from the same iterate the
    // factors were built against, so a mismatch is a wiring bug.
    if let Some(d) = app.variable_scaling() {
        if let Some((x, _lambda)) = nominal_capture.borrow_mut().as_mut() {
            assert_eq!(
                x.len(),
                d.len(),
                "scaling: captured {} variables but {} factors",
                x.len(),
                d.len()
            );
            for (xi, s) in x.iter_mut().zip(d.iter()) {
                *xi /= s;
            }
        }
        if let Some((z_l, z_u)) = bound_mult_capture.borrow_mut().as_mut() {
            assert_eq!(
                z_l.len(),
                d.len(),
                "scaling: captured {} bound multipliers but {} factors",
                z_l.len(),
                d.len()
            );
            assert_eq!(z_l.len(), z_u.len(), "z_L and z_U must be the same length");
            for ((l, u), s) in z_l.iter_mut().zip(z_u.iter_mut()).zip(d.iter()) {
                *l *= s;
                *u *= s;
            }
        }
    }

    // Bound multipliers are per *variable*, so only the column reduction can
    // shorten them. Swap in the wrapper's full-space pair when — and only
    // when — the captured one is the wrong length; leaving a correctly-sized
    // capture alone keeps the scaling path that produced it untouched.
    if elim_reduced {
        if let Some(full) = elim_handle
            .as_ref()
            .and_then(|e| e.borrow().finalized_full_solution().cloned())
        {
            if let Some((z_l, z_u)) = bound_mult_capture.borrow_mut().as_mut() {
                if z_l.len() != full.z_l.len() {
                    *z_l = full.z_l;
                    *z_u = full.z_u;
                }
            }
        }
    }

    // Reduced Hessian: print to stderr (informational), mirroring
    // upstream sIPOPT's RedHessian / Eigenvalues prints in
    // `SensReducedHessianCalculator.cpp`.
    if let Some(rh) = red_hessian_capture.borrow().as_ref() {
        sens::print_red_hessian_to_stderr(rh);
    } else if wants_red_hessian {
        eprintln!(
            "pounce: --compute-red-hessian requested but the reduced Hessian \
             was not produced (see warnings above)."
        );
    }

    // Assemble the AMPL `.sol` suffix blocks. The parametric
    // sensitivity step contributes `sens_sol_state_1` (the perturbed
    // primal) when the `.nl` declared the sIPOPT suffixes.
    let mut sol_suffixes: Vec<nl_writer::SolSuffix> = Vec::new();
    if let Some(xp) = sens_capture.borrow().clone() {
        sol_suffixes.push(nl_writer::SolSuffix {
            name: "sens_sol_state_1".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(xp),
        });
    }
    // Bound-multiplier suffixes (`ipopt_zL_out` / `ipopt_zU_out`): the
    // reduced costs / bound sensitivities. Pyomo maps these `.sol` suffix
    // blocks straight onto `model.ipopt_zL_out` / `model.ipopt_zU_out` and
    // AMPL onto variable `.rc` (gh #296).
    //
    // Sign convention — verified numerically against Ipopt 3.14 on
    // bound-active models (gh #296): Ipopt's AMPL `.sol` writes
    //   `ipopt_zL_out = +z_l`  (≥ 0 at an active lower bound), and
    //   `ipopt_zU_out = −z_u`  (≤ 0 at an active upper bound),
    // i.e. both equal the objective-gradient component at the bound
    // (`∂f/∂x_i`). `finalize_solution_z_l`/`_z_u` return the internal
    // multipliers with `z_l, z_u ≥ 0` (Ipopt's internal convention), so
    // the lower block is emitted as-is and the upper block is negated to
    // match Ipopt's output. (`min (x−3)² s.t. x≤1`: x*=1, ∂f/∂x=−4, so
    // Ipopt writes `ipopt_zU_out = −4`; pounce now matches.)
    if let Some((z_l_full, z_u_full)) = bound_mult_capture.borrow().clone() {
        let z_u_neg: Vec<pounce_common::types::Number> = z_u_full.iter().map(|&z| -z).collect();
        sol_suffixes.push(nl_writer::SolSuffix {
            name: "ipopt_zL_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_l_full),
        });
        sol_suffixes.push(nl_writer::SolSuffix {
            name: "ipopt_zU_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_u_neg),
        });
    }

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
            // `info.m` is the reduced kept-row count under presolve, but
            // the lifted `lambda` (and the `.sol`) carry the original
            // `.nl` constraint count — and `SolutionInfo::lambda` is
            // documented to have length `problem.n_constraints`. Report
            // the original `m` so that invariant holds.
            let n_dropped = presolve_handle
                .as_ref()
                .map(|p| p.borrow().n_dropped_rows())
                .unwrap_or(0);
            builder.problem.n_constraints = info.m + n_dropped;
            builder.problem.n_objectives = 1; // pounce IPM uses obj 0; multi-obj is read but ignored
            builder.problem.nnz_jac_g = Some(info.nnz_jac_g);
            builder.problem.nnz_h_lag = Some(info.nnz_h_lag);
        }
        builder.solution.status = status;
        // Same source of truth as the `.sol` writer below — a run must not
        // report 201 in one output and 200 in the other.
        builder.solution.solve_result_num = presolve_verdict(presolve_certified, status).1;
        builder.solution.objective = solve_stats.final_objective;
        if let Some((x, lambda)) = nominal_capture.borrow().clone() {
            builder.solution.x = x;
            builder.solution.lambda = lambda;
        }
        builder.ingest_stats(&solve_stats);
        if let Some(linsol) = app.linear_solver_summary() {
            builder.set_linear_solver_summary(linsol);
        }

        // `Full` detail carries the suffix blocks: the sensitivity
        // result and, when computed, the reduced Hessian (packed as
        // problem-level suffixes — see `pounce-cli`'s sens module).
        if matches!(args.json_detail, ReportDetail::Full) {
            for s in &sol_suffixes {
                builder
                    .solution
                    .suffixes
                    .push(sens::sol_suffix_to_report(s));
            }
            if let Some(rh) = red_hessian_capture.borrow().as_ref() {
                builder.solution.suffixes.push(SolutionSuffix {
                    name: "_red_hessian".to_string(),
                    target: "problem".to_string(),
                    kind: "real".to_string(),
                    values: rh.hr.clone(),
                    int_values: Vec::new(),
                });
                builder.solution.suffixes.push(SolutionSuffix {
                    name: "_red_hessian_vars".to_string(),
                    target: "problem".to_string(),
                    kind: "int".to_string(),
                    values: Vec::new(),
                    int_values: rh.var_indices.iter().map(|&v| v as i32).collect(),
                });
                if let Some(w) = &rh.eigenvalues {
                    builder.solution.suffixes.push(SolutionSuffix {
                        name: "_red_hessian_eigenvalues".to_string(),
                        target: "problem".to_string(),
                        kind: "real".to_string(),
                        values: w.clone(),
                        int_values: Vec::new(),
                    });
                }
                if let Some(v) = &rh.eigenvectors {
                    builder.solution.suffixes.push(SolutionSuffix {
                        name: "_red_hessian_eigenvectors".to_string(),
                        target: "problem".to_string(),
                        kind: "real".to_string(),
                        values: v.clone(),
                        int_values: Vec::new(),
                    });
                }
            }
        }

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
        let (n, m_out) = nlp_info_snapshot
            .as_ref()
            .map(|i| (i.n as usize, i.m as usize))
            .unwrap_or((0, 0));
        // `nlp_info_snapshot.m` is the reduced kept-row count when
        // presolve dropped rows; the zero-fallback block must be sized to
        // the original `.nl`'s `m` so a failed-solve `.sol` still aligns.
        let m = m_out
            + presolve_handle
                .as_ref()
                .map(|p| p.borrow().n_dropped_rows() as usize)
                .unwrap_or(0);
        let (x, lambda) = nominal_capture
            .borrow()
            .clone()
            .unwrap_or_else(|| (vec![0.0; n], vec![0.0; m]));
        // A presolve-certified infeasibility is reported as `201` rather than
        // the generic `200`. Both sit in AMPL's 200..299 "infeasible" band, so
        // every consumer that reads the band — Pyomo maps the whole range to
        // `TerminationCondition.infeasible` in both its readers — is unaffected,
        // while a caller reading `solve_result_num` directly can tell a *proof*
        // (bound propagation / interval arithmetic established the feasible
        // region is empty) from the numerical verdict `200` (converged to a
        // stationary point of the constraint violation, which on a nonconvex
        // problem does not preclude a feasible point elsewhere). Sub-coding
        // inside a band is the AMPL-native idiom — it is what Ipopt itself does
        // with 500/501/502 in the failure band.
        let (message, srn) = presolve_verdict(presolve_certified, status);
        let payload = nl_writer::SolutionFile {
            message: &message,
            x: &x,
            mult_g: &lambda,
            solve_result_num: srn,
            suffixes: &sol_suffixes,
        };
        match nl_writer::write_sol_file_with_options(sol_path, &payload, &nl_ampl_options) {
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

    nlp_exit_code(status, args.ampl)
}

/// Process exit code for the general NLP solve path.
///
/// A *successful* solve — `SolveSucceeded` **or** `SolvedToAcceptableLevel`
/// (the reduced-accuracy convergence Ipopt also treats as a success; see
/// `minimize()` parity, #119) — exits 0. Everything else exits 1, **except**
/// in AMPL solver mode.
///
/// In `-AMPL` mode the process exit code is not the status channel: AMPL and
/// Pyomo's ASL interface read the termination from the `.sol` file's
/// `solve_result_num`, and conventionally an AMPL solver exits 0 whenever it
/// ran and produced a `.sol` — limit reached, infeasible, even a failed solve.
/// A non-zero exit makes Pyomo raise `ApplicationError` and never parse the
/// `.sol`. Genuine startup failures (bad `.nl`, bad option) already returned
/// non-zero earlier, before the solve, so reaching here in `-AMPL` mode means a
/// `.sol` was written and carries the verdict. Mirrors [`convex_exit_code`].
fn nlp_exit_code(status: ApplicationReturnStatus, ampl: bool) -> ExitCode {
    if nlp_solve_succeeded(status) || ampl {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// Whether an NLP solve outcome counts as a "success" for the (non-AMPL) exit
/// code: `SolveSucceeded` or the reduced-accuracy `SolvedToAcceptableLevel`,
/// matching Ipopt and the `minimize()` success set (#119).
fn nlp_solve_succeeded(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// Build a `SolverDebugger` for the requested mode/flags, wired to the
/// shared restart cell. Used for the first install and each re-solve.
fn build_debugger(
    mode: pounce_cli::cli::DebugMode,
    on_error: bool,
    on_interrupt: bool,
    script: Option<&std::path::Path>,
    reg: Option<Rc<pounce_common::reg_options::RegisteredOptions>>,
    cell: pounce_cli::debug_repl::RestartCell,
) -> pounce_cli::debug_repl::SolverDebugger {
    use pounce_cli::debug_repl::SolverDebugger;
    let dbg = if on_error {
        SolverDebugger::on_error(mode, reg)
    } else if on_interrupt {
        SolverDebugger::on_interrupt(mode, reg)
    } else {
        SolverDebugger::new(mode, reg)
    }
    .with_restart(cell);
    match script {
        Some(p) => dbg.with_script(p.to_string_lossy().into_owned()),
        None => dbg,
    }
}

/// One rung of the local-infeasibility second-opinion ladder: a label for the
/// console plus the option assignments that define this re-solve's trajectory.
///
/// Assignments are applied on top of the *baseline* options, not on top of the
/// previous rung — see `second_opinion_rungs`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SecondOpinionRung {
    label: &'static str,
    assignments: Vec<String>,
}

/// What the baseline options already provide, so a rung that would be a no-op
/// can be dropped instead of burning a solve to re-derive the same answer.
#[derive(Debug, Clone, Copy)]
struct SecondOpinionAvailability {
    scaling_retry_enabled: bool,
    mu_retry_enabled: bool,
    already_mc64: bool,
    already_adaptive: bool,
    /// `feral_scaling` tag naming the baseline's *resolved* scaling strategy,
    /// which the barrier rung re-asserts so it varies exactly one knob.
    /// `None` when the resolved strategy has no tag to write back
    /// (`ScalingStrategy::External`), which drops the barrier rung rather than
    /// let it run under a scaling the baseline never used.
    baseline_scaling: Option<&'static str>,
}

/// Build the ladder of second-opinion re-solves for a local-infeasibility
/// verdict, in the order they should be tried.
///
/// Rung 1 (`feral_scaling=mc64`) perturbs the linear algebra only. Rung 2
/// (`mu_strategy=adaptive`) perturbs the barrier trajectory, and **restores the
/// baseline scaling first** so it varies exactly one knob from the original
/// solve. That reset is load-bearing, not tidiness: on gh #524's `cresc4`,
/// `mu_strategy=adaptive` recovers the optimum but `mu_strategy=adaptive` with
/// `feral_scaling=mc64` still reports local infeasibility, so a cumulative
/// ladder would have discarded the fix.
fn second_opinion_rungs(avail: SecondOpinionAvailability) -> Vec<SecondOpinionRung> {
    let mut rungs = Vec::new();
    if avail.scaling_retry_enabled && !avail.already_mc64 {
        rungs.push(SecondOpinionRung {
            label: "feral_scaling=mc64",
            assignments: vec!["feral_scaling mc64\n".to_string()],
        });
    }
    if let Some(baseline_scaling) = avail.baseline_scaling
        && avail.mu_retry_enabled
        && !avail.already_adaptive
    {
        rungs.push(SecondOpinionRung {
            label: "mu_strategy=adaptive",
            assignments: vec![
                format!("feral_scaling {baseline_scaling}\n"),
                "mu_strategy adaptive\n".to_string(),
            ],
        });
    }
    rungs
}

/// Did a second-opinion re-solve converge well enough to overturn the original
/// local-infeasibility verdict? Only a clean or acceptable-level solve
/// promotes; everything else (including a second infeasibility verdict) leaves
/// the original verdict standing.
fn scaling_retry_promoted(retry_status: ApplicationReturnStatus) -> bool {
    matches!(
        retry_status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// Resolve the final `(status, statistics)` after an MC64 hypersensitivity
/// re-solve (code review L23).
///
/// On promotion the retry is the authoritative solve, so its status **and** its
/// statistics are reported together. Otherwise the original local-infeasibility
/// verdict is kept — and so are the *original* solve's statistics, so the
/// summary / JSON report never pair the original verdict with the failed
/// retry's iteration count or objective. The pre-fix code reverted `status` to
/// `InfeasibleProblemDetected` but read `app.statistics()` *after* the retry,
/// leaking the retry solve's stats into a report labeled with the original
/// verdict.
fn resolve_scaling_retry_outcome(
    retry_status: ApplicationReturnStatus,
    original_stats: SolveStatistics,
    retry_stats: SolveStatistics,
) -> (ApplicationReturnStatus, SolveStatistics) {
    if scaling_retry_promoted(retry_status) {
        (retry_status, retry_stats)
    } else {
        (
            ApplicationReturnStatus::InfeasibleProblemDetected,
            original_stats,
        )
    }
}

/// Should an LP whose convex solve came back without a certificate be handed
/// to the general NLP interior-point path instead (gh #535)?
///
/// The NETLIB `gen`/`gen1` models are the case this exists for: `auto` routes
/// them to the convex IPM, which exhausts its 200-iteration budget in 191 s and
/// exits `OptimalInaccurate` with a primal residual of 1.4e-7 against
/// `tol = 1e-8`, while the general NLP filter-IPM — the same binary, the
/// default for every other class — solves the same model in 19 iterations and
/// 1.0 s to a strict certificate. The models are highly degenerate and
/// rank-deficient, strict complementarity fails, and a pure IPM cannot certify
/// the vertex (gh #133); crossover was built to close that gap and does not.
/// So the routing is the cheap lever: an LP is also a valid NLP, and the NLP
/// path is already in the binary.
///
/// The three gates, all necessary:
///
/// * **`allow_nlp_fallback`** — set by the caller only under `auto` (the class
///   was our inference, not the user's instruction), with no interactive
///   debugger attached, and only when the user did **not** set `max_iter`. A
///   user-set budget is a budget: `IterationLimit` is then the answer to the
///   question that was asked, and `max_iter=0` in particular must stop without
///   a solve (pounce#186), not launch a second one. An explicitly tightened
///   `tol` is deliberately *not* a suppressor — that is an accuracy request,
///   and trying the engine that can meet it is exactly the right response.
/// * **`ProblemClass::Lp`** — `P = 0`, per the issue. A convex QP that stalls
///   is a different (and unmeasured) population; leave it to the engine that
///   was chosen for it.
/// * **the status** — the three that mean "no certificate": a reduced-accuracy
///   exit, an exhausted budget, and a numerical failure. `Optimal` needs no
///   help, and `PrimalInfeasible` / `DualInfeasible` are verdicts the convex
///   solver *verified*, which a second solve must not be allowed to overwrite.
///   `NumericalFailure` was excluded until gh #724 on the grounds that it is
///   the post-solve verification refusing a point and no LP in the corpus
///   reached it. Both halves of that are the wrong test. It is the *strongest*
///   of the three "did not certify" signals — the point on offer missed even
///   the acceptable band — and it is the one status `run_convex_socp` reroutes
///   on for the conic path, so omitting it here made the LP and SOCP paths
///   disagree about what an unverified convex result means. An LP that reached
///   it was reported `InternalError` on a model the NLP path in the same
///   binary solves (gh #724 reproduces this on `lp_afiro` with `qp_tau=0.99`).
///   `TimeLimit` still does not reroute: it is a spent budget rather than a
///   stall, and rerouting it would answer "stop after `max_wall_time`" with a
///   second solve.
///
/// Note what this deliberately is **not**: the issue's "never-regress" variant,
/// which would keep whichever of the two results certifies at the lower KKT
/// error. That needs both verdicts in hand at reporting time, and the CLI's
/// standing rule — the one `run_convex_socp` follows and gh #508 re-litigated
/// for the NLP retry ladder — is that one solve prints one verdict. So the
/// decision is taken *before* any status line, `.sol` or JSON report is
/// emitted, and the NLP solve owns the whole report. The gates above are what
/// bound the downside: this only ever runs on an LP that already failed to
/// certify under a default budget.
fn lp_declines_to_nlp(
    class: pounce_cli::dispatch::ProblemClass,
    status: pounce_convex::QpStatus,
    allow_nlp_fallback: bool,
) -> bool {
    use pounce_convex::QpStatus;
    allow_nlp_fallback
        && class == pounce_cli::dispatch::ProblemClass::Lp
        && matches!(
            status,
            QpStatus::OptimalInaccurate | QpStatus::IterationLimit | QpStatus::NumericalFailure
        )
}

/// Did the user set `max_iter` explicitly? A user-set iteration budget
/// suppresses the gh #535 LP→NLP fallback — see [`lp_declines_to_nlp`].
fn max_iter_explicitly_set(app: &IpoptApplication) -> bool {
    matches!(
        app.options().get_integer_value("max_iter", ""),
        Ok((_, true))
    )
}

/// Solve a classified LP / convex-QP `.nl` problem through the
/// specialized `pounce-convex` interior-point method, write a `.sol`,
/// and return the process exit code. This is the LP/QP dispatch target
/// (see `dev-notes/lp-qp-routing.md`).
///
/// Writes the primal solution `x` and the constraint duals recovered
/// from the QP multipliers (`pounce_cli::qp_extract::recover_duals`).
/// The objective is reported in the user's original sense, including the
/// `.nl`'s constant term, which the standard-form QP drops.
/// Map the convex solver's status onto the NLP-side `ApplicationReturnStatus`
/// used by the JSON solve report, so QP and NLP reports share one status
/// vocabulary.
fn qp_status_to_ars(s: pounce_convex::QpStatus) -> ApplicationReturnStatus {
    use pounce_convex::QpStatus;
    match s {
        QpStatus::Optimal => ApplicationReturnStatus::SolveSucceeded,
        // Reduced-accuracy solve (residual above `tol` but usable) — Ipopt's
        // "Solved To Acceptable Level" is the matching NLP-side status.
        QpStatus::OptimalInaccurate => ApplicationReturnStatus::SolvedToAcceptableLevel,
        QpStatus::PrimalInfeasible => ApplicationReturnStatus::InfeasibleProblemDetected,
        QpStatus::DualInfeasible => ApplicationReturnStatus::DivergingIterates, // unbounded
        QpStatus::IterationLimit => ApplicationReturnStatus::MaximumIterationsExceeded,
        QpStatus::TimeLimit => ApplicationReturnStatus::MaximumWallTimeExceeded,
        QpStatus::NumericalFailure => ApplicationReturnStatus::InternalError,
    }
}

/// Map a convex-solver status onto the AMPL `.sol` terminal line: the message,
/// whether the solve is treated as a success (drives the exit code), and the
/// `solve_result_num`. AMPL convention: 0 solved, 100–199 solved with a
/// warning, 200–299 infeasible, 300–399 unbounded, 400–499 limit, 500–599
/// failure. Shared by the QP/LP and SOCP report paths so the two cannot drift.
///
/// `OptimalInaccurate` reports `1` — Ipopt's code for the same
/// reduced-accuracy convergence, and the same code the NLP path's
/// `SolvedToAcceptableLevel` reports (`status_to_solve_result_num`), which
/// this status maps onto in `qp_status_to_ars`. One status, one code: a model
/// must not change its `.sol` verdict band depending on which engine took it.
fn convex_status_report(s: pounce_convex::QpStatus) -> (&'static str, bool, i32) {
    use pounce_convex::QpStatus;
    match s {
        QpStatus::Optimal => ("Optimal Solution Found.", true, 0),
        QpStatus::OptimalInaccurate => ("Solved to acceptable level (reduced accuracy).", true, 1),
        QpStatus::PrimalInfeasible => ("Problem is primal infeasible.", false, 200),
        QpStatus::DualInfeasible => ("Problem is unbounded (dual infeasible).", false, 300),
        QpStatus::IterationLimit => ("Maximum iterations exceeded.", false, 400),
        QpStatus::TimeLimit => ("Maximum wallclock time exceeded.", false, 400),
        // Deliberately not "failure in KKT factorization": both convex engines
        // reach this status by failing the *post-solve* verification — the
        // returned point's true KKT error exceeded the acceptable band — which
        // a factorization breakdown is only one cause of. On the active-set
        // engine it is also where an uncertified infeasibility claim lands.
        QpStatus::NumericalFailure => ("Numerical failure (no verified KKT point).", false, 500),
    }
}

/// The `bound_relax_factor` widening the convex path must apply so that the
/// model it solves is the one the NLP path solves.
///
/// Reads the same two options `Application` feeds `OrigIpoptNlp::relax_bounds`
/// — `bound_relax_factor` (Ipopt default `1e-8`) and `constr_viol_tol`
/// (default `1e-4`) — and uses the same defaults when the user set neither,
/// so `pounce foo.nl` and `pounce foo.nl solver_selection=nlp` agree.
///
/// gh #744 / #745: before this, the convex arm ignored both and solved the
/// model exactly as declared. On the constraint-degenerate families
/// (`LISWET*`, `YAO`, `POWELL20`, the `pldd*`/`delf*`/`large*` LPs) that made
/// the two arms of the same binary disagree by up to 33% in the objective —
/// the convex arm reporting the exact optimum and the NLP arm the relaxed one.
/// See [`pounce_cli::qp_extract::BoundRelax`] for why so small a widening
/// moves the objective so far.
fn convex_bound_relax(app: &IpoptApplication) -> pounce_cli::qp_extract::BoundRelax {
    let opt = app.options();
    let num = |name: &str, default: f64| {
        opt.get_numeric_value(name, "")
            .ok()
            .and_then(|(v, set)| set.then_some(v))
            .unwrap_or(default)
    };
    pounce_cli::qp_extract::BoundRelax {
        factor: num("bound_relax_factor", 1e-8),
        cap: num("constr_viol_tol", 1e-4),
    }
}

fn convex_opts_with_remaining(
    mut opts: pounce_convex::QpOptions,
    started: std::time::Instant,
) -> pounce_convex::QpOptions {
    if let Some(limit) = opts.time_limit {
        opts.time_limit = Some(limit.saturating_sub(started.elapsed()));
    }
    opts
}

/// Charge `spent` against `max_wall_time`, so a solve that gets handed from one
/// engine to another spends **one** budget rather than one per engine.
///
/// The convex driver already deducts its own extraction and presolve from the
/// budget it passes down (`convex_opts_with_remaining`). The gap this closes is
/// one level up: when a convex attempt declines the problem (gh #535's LP→NLP
/// reroute, or the conic path's `socp_nlp_fallback`) the NLP solve that takes
/// over builds its `Deadline` from the *option value*, which still names the
/// whole budget. A run that spent 55 of its 60 seconds convex-side would then be
/// granted 60 more, and `max_wall_time` would buy nearly twice the wall clock it
/// promises.
///
/// Applies only when the user actually set the option. Unset it is `1e6` — the
/// effectively-unbounded default — and rewriting that as `1e6 - 3.2` states
/// nothing the default did not, while making the option read as explicitly
/// chosen to everything downstream that tests the `explicitly_set` flag.
///
/// A budget that is entirely gone floors at [`WALL_BUDGET_FLOOR`] rather than
/// zero: the option is registered with a *strict* lower bound of 0, so `0.0` is
/// rejected as invalid and the write would silently do nothing — leaving the
/// full budget in place, which is the failure this exists to prevent. The floor
/// is small enough that the NLP path's first deadline check trips on it.
fn charge_wall_budget(
    opts: &mut pounce_common::options_list::OptionsList,
    spent: std::time::Duration,
) {
    /// Smallest budget that can be *stored* — see [`charge_wall_budget`].
    const WALL_BUDGET_FLOOR: f64 = 1e-9;

    if let Ok((limit, true)) = opts.get_numeric_value("max_wall_time", "") {
        let left = (limit - spent.as_secs_f64()).max(WALL_BUDGET_FLOOR);
        let _ = opts.set_numeric_value("max_wall_time", left, true, false);
    }
}

/// Returns `None` when the LP→NLP fallback fires (gh #535): the problem is an
/// LP, the caller allowed the fallback, and the convex solve finished without a
/// certificate, so the caller falls through to the general NLP interior-point
/// path. No status line, `.sol` or JSON report has been emitted in that case —
/// the decision is taken above all three, so a rerouted solve produces exactly
/// one verdict. (A `Presolve:` reduction line may already have been printed;
/// it reports what presolve did, not what the solve concluded.) See
/// [`lp_declines_to_nlp`] for the gating.
fn run_convex_qp(
    prob: &nl_reader::NlProblem,
    class: pounce_cli::dispatch::ProblemClass,
    sol_path: Option<&std::path::Path>,
    presolve_on: bool,
    json_cfg: Option<(&std::path::Path, ReportDetail, InputDescriptor)>,
    debug_hook: Option<&Rc<RefCell<pounce_cli::debug_repl::SolverDebugger>>>,
    ampl: bool,
    convex_opts: pounce_convex::QpOptions,
    // gh #744/#745: the `bound_relax_factor` widening applied to the extracted
    // model, so this path solves what the NLP path solves.
    bound_relax: pounce_cli::qp_extract::BoundRelax,
    // Use the `pounce-qp` parametric active-set engine instead of the IPM
    // (`solver_selection=qp-active-set`). Everything else about this driver —
    // extraction, presolve, postsolve, reporting, `.sol` writing — is shared.
    use_active_set: bool,
    // Inner-engine overrides from the `sqp_qp_*` family; empty for the IPM.
    engine_overrides: pounce_convex::ActiveSetOverrides,
    // gh #535: may an uncertified LP solve be handed back to the NLP path?
    allow_nlp_fallback: bool,
) -> Option<ExitCode> {
    let t0 = std::time::Instant::now();
    use pounce_convex::active_set::solve_qp_active_set;
    use pounce_convex::presolve::{FixpointExit, PresolveOutcome, presolve};
    use pounce_convex::{QpOptions, QpStatus, solve_qp_ipm, solve_qp_ipm_debug};

    let (qp, con_map, obj_nl_const) =
        match pounce_cli::qp_extract::extract_qp_with_map(prob, bound_relax) {
            Some(q) => q,
            None => {
                eprintln!(
                    "pounce: internal error: {} not extractable as QP",
                    class.name()
                );
                return Some(ExitCode::from(2));
            }
        };

    // The reported objective must include *both* constant sources: the
    // `.nl` linear-section constant (`obj_constant`) and any degree-0 term
    // AMPL/Pyomo folded into the nonlinear objective tree (`obj_nl_const`,
    // recovered by `extract_qp_with_map`). Dropping the latter makes the
    // convex solve report an objective off by that constant versus the NLP
    // path (e.g. HS21 by −100, HS35 by +9). Both are in user sense.
    let obj_const = prob.obj_constant + obj_nl_const;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    let backend = || -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::new())
    };
    // With presolve on, reduce the problem (logging what was removed),
    // solve the reduced problem, then postsolve back to the extracted-QP
    // space — so the `con_map`-based dual recovery below still applies.
    // Trivial infeasibility / unboundedness is reported without solving.
    let trivial = |status| pounce_convex::QpSolution {
        status,
        x: vec![0.0; qp.n],
        y: vec![0.0; qp.m_eq()],
        z: vec![0.0; qp.m_ineq()],
        z_lb: vec![0.0; qp.n],
        z_ub: vec![0.0; qp.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    // Collect the per-iteration convergence trace only when a Full-detail
    // JSON report was requested (it carries the `iterations` array); the
    // default solve stays trace-free.
    let want_trace = matches!(&json_cfg, Some((_, ReportDetail::Full, _)));
    let qp_opts = QpOptions {
        collect_iterates: want_trace,
        // Tell the solver the objective constant it is *not* carrying (gh
        // #689). `QpProblem` holds the quadratic form only, so on a
        // least-squares-shaped objective the solver's `obj` is the reported one
        // displaced by `obj_const` — unbounded displacement, `5e11` on the
        // `scaled_feasible` pair — and the scale-relative stopping test then
        // normalizes the duality gap by a magnitude that belongs to the
        // constant rather than to the solution. In the solver's own (minimize)
        // sense the constant is `sign · obj_const`, the inverse of the
        // `reported_obj` line below. Convergence-test normalizer only: it
        // changes no residual, no dual, and not `sol.obj`.
        obj_constant: sign * obj_const,
        ..convex_opts
    };
    // The reduced problem presolve hands the solver differs from `qp` by
    // `ps.obj_offset()`, so the constant that makes *it* commensurate with the
    // user's objective carries that offset too.
    let solve_opts_offset = |offset: f64| {
        convex_opts_with_remaining(
            QpOptions {
                obj_constant: qp_opts.obj_constant + offset,
                ..qp_opts
            },
            t0,
        )
    };
    let solve_opts = || solve_opts_offset(0.0);
    // What presolve did, held back until we know this solve is the one that
    // reports (gh #535). These lines describe the reduction, not the verdict,
    // but they are the *only* stdout a declined convex attempt would otherwise
    // leave behind — and "the rerouted run prints nothing from the attempt it
    // discarded" is a cleaner contract than "nothing except one line". Flushed
    // below, immediately after the fallback check.
    let mut presolve_log: Vec<String> = Vec::new();
    let sol = if qp_opts.max_iter == 0 {
        // AMPL/Ipopt semantics: `max_iter=0` takes no iterations and so
        // cannot reach optimality. Presolve can otherwise solve a trivial
        // problem (e.g. an unconstrained quadratic) directly — or the IPM's
        // reduced/empty solve can report Optimal — regardless of the cap, so
        // enforce the zero-iteration stop here before any solve runs
        // (pounce#186). Mirrors the NLP path's MaximumIterationsExceeded.
        trivial(QpStatus::IterationLimit)
    } else if let Some(hook) = debug_hook.filter(|_| !use_active_set) {
        // Interactive debug: step the IPM on the extracted QP directly.
        // Presolve is skipped so the debugger's `x`/`s`/`y`/`z` blocks
        // correspond to the user's problem rather than a reduced one.
        //
        // Guarded on `!use_active_set`: the debugger hooks barrier-IPM
        // iterations and has no active-set analogue, so on that engine this
        // arm would quietly solve with a *different solver* than the one the
        // user selected. The caller has already printed the note explaining
        // the debugger does not engage; fall through and solve normally.
        let mut h = hook.borrow_mut();
        solve_qp_ipm_debug(&qp, &solve_opts(), &mut *h, backend)
    } else if presolve_on {
        match presolve(&qp) {
            PresolveOutcome::Reduced(ps) => {
                // A screen claimed infeasibility and the re-derivation without
                // the speculative fixings would not reproduce it, so presolve
                // solved on instead (gh #523). Say so: the guard turns a false
                // `Infeasible_Problem_Detected` into a normal solve, and this
                // line is the only trace of the reduction that misfired.
                if let Some(trigger) = ps.discarded_infeasibility() {
                    presolve_log.push(format!(
                        "Presolve: discarded an unconfirmed infeasibility claim — \
                         {trigger}; solving normally"
                    ));
                }
                let st = ps.stats();
                if st.reduced_anything() {
                    // Whether the fixpoint converged or the layer cap stopped
                    // it (gh #527), as a suffix rather than a line of its own.
                    // The corpus sweep on #530 measured the cap binding on 46%
                    // of LP and 25% of QP models — it is the common case, not
                    // an alarm, and it never changed the structural reduction
                    // on any of the 394 models that presolve at all. A second
                    // stdout line on half of all solves would read as a
                    // warning about something that is working as designed;
                    // what the reduction needs to carry is which of the two it
                    // came out of, and that fits here.
                    let exit = match st.exit {
                        FixpointExit::Fixpoint => String::new(),
                        FixpointExit::RoundCap => {
                            format!(", cap-truncated after {} layers", st.rounds)
                        }
                    };
                    presolve_log.push(format!(
                        "Presolve: {} → {} vars, {} → {} rows (fixed {}, \
                         free-fixed {}, substituted {}, aggregated {}, \
                         forcing {}, dominated {}, tightened {}{})",
                        st.orig_vars,
                        st.reduced_vars,
                        st.orig_rows,
                        st.reduced_rows,
                        st.fixed_vars,
                        st.free_cols_fixed,
                        st.free_col_singletons,
                        st.aggregated_vars,
                        st.forcing_rows,
                        st.dominated_cols,
                        st.tightened_bounds,
                        exit,
                    ));
                }
                let red = if use_active_set {
                    let mut mk = backend;
                    solve_qp_active_set(
                        &ps.reduced,
                        &solve_opts_offset(ps.obj_offset()),
                        &engine_overrides,
                        &mut mk,
                    )
                } else {
                    solve_qp_ipm(&ps.reduced, &solve_opts_offset(ps.obj_offset()), backend)
                };
                ps.postsolve(&red)
            }
            PresolveOutcome::Infeasible(trigger) => {
                // Name the screen and what it tripped on. A presolve
                // infeasibility arrives with no iteration behind it, so this
                // line is the whole record of *why* (gh #523).
                presolve_log.push(format!("Presolve: proved primal infeasible — {trigger}"));
                trivial(QpStatus::PrimalInfeasible)
            }
            PresolveOutcome::Unbounded => trivial(QpStatus::DualInfeasible),
        }
    } else if use_active_set {
        let mut mk = backend;
        solve_qp_active_set(&qp, &solve_opts(), &engine_overrides, &mut mk)
    } else {
        solve_qp_ipm(&qp, &solve_opts(), backend)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    // gh #535: the convex path finished an LP without a certificate. An LP is
    // also a valid NLP, and the general filter-IPM in this same binary solves
    // the degenerate rank-deficient ones the interior path cannot certify
    // (NETLIB `gen`/`gen1`: 199 iters / 191 s at reduced accuracy here, 19
    // iters / 1.0 s and a strict certificate there). Hand it over rather than
    // reporting the uncertified iterate as the last word.
    //
    // This sits above the status line, the `.sol` write and the JSON report on
    // purpose — everything below is the verdict, and the rerouted solve owns
    // it. See `lp_declines_to_nlp` for why each gate is there.
    if lp_declines_to_nlp(class, sol.status, allow_nlp_fallback) {
        let res = sol.kkt_residuals(&qp);
        eprintln!(
            "pounce: note: the convex ({}) solve did not certify a KKT point \
             after {} iterations in {elapsed:.3}s (KKT error {:.2e} against \
             tol {:.1e}); an LP is also a valid NLP, so it is being re-solved \
             on the general NLP interior-point path, which certifies the \
             degenerate, rank-deficient LPs the interior path stalls on (gh \
             #133). Use solver_selection=qp-ipm to see the convex result \
             instead.",
            class.name(),
            sol.iters,
            res.kkt_error(),
            qp_opts.tol,
        );
        return None;
    }

    // This solve is the one that reports, so what presolve did belongs on the
    // record after all.
    for line in &presolve_log {
        println!("{line}");
    }

    // Report the objective in the user's original sense, including the
    // dropped constant term: f_user = sign * (½xᵀPx + cᵀx) + const.
    let reported_obj = sign * sol.obj + obj_const;

    let (msg, ok, srn) = convex_status_report(sol.status);
    // Name the engine that actually ran — the two report different iteration
    // counts (barrier iterations vs active-set changes), so labelling an
    // active-set solve "IPM" would misread both the solver and the number.
    let engine = if use_active_set {
        "active-set, pounce-qp"
    } else {
        "IPM, pounce-convex"
    };
    println!(
        "POUNCE ({} {engine}): {msg}  obj={reported_obj:.8}  iters={}  ({elapsed:.3}s)",
        class.name(),
        sol.iters,
    );
    // gh #293 naive-caller guardrail: if the solve did not cleanly converge and
    // the objective curvature is tiny relative to the data, say so — the status
    // is honest but a naive caller might otherwise treat a truncated objective
    // as the optimum.
    if let Some(warn) = sol.scaling_diagnostic(&qp) {
        eprintln!("pounce: {warn}");
    }

    // Final KKT residuals from pounce-convex; reused for both the Ipopt-style
    // summary block and the JSON report below.
    let res = sol.kkt_residuals(&qp);
    // Ipopt-style summary so the objective/iteration count are scrapable by
    // consumers that parse Ipopt's end-of-run block (see print_convex_summary).
    print::print_convex_summary(
        sol.iters,
        reported_obj,
        res.primal_infeasibility,
        res.dual_infeasibility,
        res.complementarity,
        res.kkt_error(),
    );

    // Recover per-constraint duals once (mapped from the QP multipliers back
    // to per-`.nl`-constraint order); used by both the `.sol` and the JSON
    // report.
    let lambda = pounce_cli::qp_extract::recover_duals(prob, &con_map, &sol.y, &sol.z);

    // Bound multipliers (`ipopt_zL_out`/`ipopt_zU_out`). The QP extractor puts
    // the `.nl` variable bounds in the solver's explicit box, so these come
    // back directly in `sol.z_lb`/`z_ub`. They are for the *internal* minimize
    // form (`½xᵀPx + cᵀx`, a maximize objective negated), so `sign` restores
    // the user's objective sense — the same conversion `recover_duals` applies
    // to the constraint duals. The Ipopt output convention (verified
    // numerically, gh #296) is `ipopt_zL_out = +z_l`, `ipopt_zU_out = −z_u`,
    // both equal to the objective-gradient component at the bound. QP
    // variables are 1:1 with the `.nl` variables, so no remap is needed.
    let (z_lb_raw, z_ub_raw) = pounce_cli::qp_extract::recover_bound_mults(prob, &sol);
    let z_l_suffix: Vec<f64> = z_lb_raw.iter().map(|&z| sign * z).collect();
    let z_u_suffix: Vec<f64> = z_ub_raw.iter().map(|&z| -sign * z).collect();
    let qp_bound_suffixes = [
        nl_writer::SolSuffix {
            name: "ipopt_zL_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_l_suffix),
        },
        nl_writer::SolSuffix {
            name: "ipopt_zU_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_u_suffix),
        },
    ];

    // Write a `.sol` if requested: primal x and recovered constraint duals in
    // the AMPL `.sol` convention.
    if let Some(path) = sol_path {
        let payload = nl_writer::SolutionFile {
            message: &format!("POUNCE {} IPM (pounce-convex): {msg}", class.name()),
            x: &sol.x,
            mult_g: &lambda,
            solve_result_num: srn,
            suffixes: &qp_bound_suffixes,
        };
        // Log a `.sol` write failure but do not early-return a distinct exit
        // code: the NLP path (main.rs:1091-1093) only logs, and under `-AMPL`
        // the final exit must still follow the solve-outcome contract.
        if let Err(e) = nl_writer::write_sol_file_with_options(path, &payload, &prob.ampl_options) {
            eprintln!("pounce: failed to write {}: {e}", path.display());
        }
    }

    // Emit the JSON solve report, when requested — same `pounce.solve-report/v1`
    // schema as the NLP path, so the benchmark harness can compare QP and NLP
    // solves uniformly. (Per-iteration history is NLP-only for now; the convex
    // driver does not yet feed the iterate trace, so `iterations` stays empty
    // even at Full detail.)
    if let Some((json_path, detail, input)) = json_cfg {
        let mut builder = ReportBuilder::new(detail, input);
        builder.problem.n_variables = qp.n as _;
        builder.problem.n_constraints = lambda.len() as _;
        builder.problem.n_objectives = 1;
        builder.problem.minimize = prob.minimize;
        builder.solution.status = qp_status_to_ars(sol.status);
        builder.solution.solve_result_num = srn;
        builder.solution.objective = reported_obj;
        builder.solution.x = sol.x.clone();
        builder.solution.lambda = lambda.clone();
        builder.stats.iteration_count = sol.iters as _;
        builder.stats.final_objective = reported_obj;
        builder.stats.total_wallclock_time_secs = elapsed;
        // Real final KKT residuals (from pounce-convex, computed above), so the
        // harness sees genuine convergence numbers rather than zeros.
        builder.stats.final_constr_viol = res.primal_infeasibility;
        builder.stats.final_dual_inf = res.dual_infeasibility;
        builder.stats.final_compl = res.complementarity;
        builder.stats.final_kkt_error = res.kkt_error();
        // Per-iteration convergence trace at Full detail (the convex IPM's
        // iterate records map onto the report's IterRecord schema, shared with
        // the NLP path so the harness reads one format).
        if matches!(detail, ReportDetail::Full) {
            builder.iterations = sol
                .iterates
                .iter()
                .map(|it| IterRecord {
                    iter: it.iter as _,
                    objective: it.objective,
                    inf_pr: it.primal_infeasibility,
                    inf_du: it.dual_infeasibility,
                    mu: it.mu,
                    alpha_primal: it.alpha_primal,
                    alpha_dual: it.alpha_dual,
                    ..IterRecord::default()
                })
                .collect();
        }
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

    Some(convex_exit_code(ok, ampl))
}

/// Solve a classified **convex QCQP** by reformulating it to a second-order
/// cone program and running the conic IPM (`pounce-convex`). Mirrors
/// [`run_convex_qp`]: same objective-constant fold-back, `.sol`/JSON output,
/// and per-constraint dual recovery, but the constraints carry quadratic rows
/// that become SOC blocks (see `qp_extract::extract_socp_with_map`). Presolve
/// is skipped — it is the QP-path's nonnegative-orthant reducer and is not
/// cone-aware.
///
/// Returns `None` when `allow_nlp_fallback` is set and the conic solve came
/// back without a verified KKT point: the caller then falls through to the
/// general NLP interior-point path. Nothing has been printed or written to
/// the `.sol`/JSON in that case — the decision is taken before any output, so
/// the fallback solve owns the whole report and a user never sees two verdicts
/// for one solve. See the call site for why this exists.
fn run_convex_socp(
    prob: &nl_reader::NlProblem,
    class: pounce_cli::dispatch::ProblemClass,
    sol_path: Option<&std::path::Path>,
    json_cfg: Option<(&std::path::Path, ReportDetail, InputDescriptor)>,
    debug_hook: Option<&Rc<RefCell<pounce_cli::debug_repl::SolverDebugger>>>,
    ampl: bool,
    convex_opts: pounce_convex::QpOptions,
    // gh #744/#745: see the same parameter on `run_convex_qp`.
    bound_relax: pounce_cli::qp_extract::BoundRelax,
    // #139 / gh #588 (Q9b): the shared convex-path presolve switch. Q1 left
    // this driver with no presolve at all, so `qp_presolve` was silently
    // ignored on every convex QCQP; it is honoured here through the
    // *cone-aware* entry point, never the orthant one — see the call below.
    presolve_on: bool,
    // gh #535: may an unverified conic solve be handed back to the NLP path?
    allow_nlp_fallback: bool,
) -> Option<ExitCode> {
    let t0 = std::time::Instant::now();
    use pounce_convex::presolve::{PresolveOutcome, presolve_conic};
    use pounce_convex::{QpOptions, solve_socp_ipm, solve_socp_ipm_debug};

    let (qp, con_map, obj_nl_const, cones) =
        match pounce_cli::qp_extract::extract_socp_with_map(prob, bound_relax) {
            Some(q) => q,
            None => {
                eprintln!(
                    "pounce: internal error: {} not extractable as SOCP",
                    class.name()
                );
                return Some(ExitCode::from(2));
            }
        };

    // Reported objective includes both constant sources (the `.nl` linear
    // section and the degree-0 term folded into the nonlinear objective tree),
    // in the user's sense — identical to the QP path.
    let obj_const = prob.obj_constant + obj_nl_const;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    let backend = || -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::new())
    };
    let want_trace = matches!(&json_cfg, Some((_, ReportDetail::Full, _)));
    let qp_opts = QpOptions {
        collect_iterates: want_trace,
        // The objective constant the solver is not carrying, in its own
        // (minimize) sense — see the QP path for what it is for (gh #689).
        // This path does not presolve, so there is no reduction offset to add.
        obj_constant: sign * obj_const,
        ..convex_opts
    };
    let solve_opts = || convex_opts_with_remaining(qp_opts, t0);
    let trivial = |status| pounce_convex::QpSolution {
        status,
        x: vec![0.0; qp.n],
        y: vec![0.0; qp.m_eq()],
        z: vec![0.0; qp.m_ineq()],
        z_lb: vec![0.0; qp.n],
        z_ub: vec![0.0; qp.n],
        obj: 0.0,
        iters: 0,
        iterates: Vec::new(),
    };
    // Held back until we know this solve is the one that reports (gh #535),
    // exactly as `run_convex_qp` does: these lines describe the reduction,
    // not the verdict, and a declined conic attempt must leave no stdout.
    let mut presolve_log: Vec<String> = Vec::new();
    let sol = if qp_opts.max_iter == 0 {
        // `max_iter=0` cannot reach optimality — stop before any solve, the
        // same zero-iteration contract the QP path enforces (pounce#186).
        // Above the presolve arm for the same reason it is on the QP path:
        // presolve can otherwise settle a trivial problem without iterating.
        trivial(pounce_convex::QpStatus::IterationLimit)
    } else if let Some(hook) = debug_hook {
        // Interactive debug steps the conic IPM on the *extracted* problem,
        // so the debugger's blocks correspond to the user's rows rather than
        // a reduced set. Same carve-out as the QP path.
        let mut h = hook.borrow_mut();
        solve_socp_ipm_debug(&qp, &cones, &solve_opts(), &mut *h, backend)
    } else if presolve_on {
        // **`presolve_conic`, never `presolve`.** The orthant entry point
        // would hand `dedup_rows` an unprotected row list, and this driver's
        // rows are exactly the shape that breaks it: `extract_socp_with_map`
        // emits each quadratic row's linear part `aᵢ` *verbatim* as SOC rows
        // 0 and 1 of its block, so two quadratic constraints sharing a linear
        // part produce byte-identical rows in different cones.
        // `parallel_signature` hashes on the linear triplets alone, sees a
        // duplicate, and drops one — silently deleting half of one cone. See
        // `crates/pounce-convex/tests/presolve_conic_quadratic_rows.rs`.
        // `presolve_conic` protects every non-orthant row, which is what
        // makes this call safe; §7 of `dev-notes/quadratic-structure-
        // exploitation.md` is the validity table.
        match presolve_conic(&qp, &cones) {
            PresolveOutcome::Reduced(ps) => {
                if let Some(trigger) = ps.discarded_infeasibility() {
                    presolve_log.push(format!(
                        "Presolve: discarded an unconfirmed infeasibility claim — \
                         {trigger}; solving normally"
                    ));
                }
                let st = ps.stats();
                if st.reduced_anything() {
                    // No `cap-truncated` suffix here: `presolve_conic` is a
                    // single pass by construction (gh #527's round cap is a
                    // fixpoint notion), so `rounds` is always 1.
                    presolve_log.push(format!(
                        "Presolve: {} → {} vars, {} → {} rows (fixed {}, \
                         free-fixed {}, substituted {}, forcing {}, \
                         dominated {}, tightened {})",
                        st.orig_vars,
                        st.reduced_vars,
                        st.orig_rows,
                        st.reduced_rows,
                        st.fixed_vars,
                        st.free_cols_fixed,
                        st.free_col_singletons,
                        st.forcing_rows,
                        st.dominated_cols,
                        st.tightened_bounds,
                    ));
                }
                // The reduced cone partition: orthant blocks may shrink or
                // vanish, cone blocks pass through whole (that is what the
                // protection buys). `postsolve` restores `z` at the original
                // row indices, so `con_map`'s `z_row0`/`z_row1` — and hence
                // `recover_socp_duals` below — need no remapping.
                let red_cones = ps.reduced_cones(&cones);
                let red = solve_socp_ipm(&ps.reduced, &red_cones, &solve_opts(), backend);
                ps.postsolve(&red)
            }
            PresolveOutcome::Infeasible(trigger) => {
                presolve_log.push(format!("Presolve: proved primal infeasible — {trigger}"));
                trivial(pounce_convex::QpStatus::PrimalInfeasible)
            }
            PresolveOutcome::Unbounded => trivial(pounce_convex::QpStatus::DualInfeasible),
        }
    } else {
        solve_socp_ipm(&qp, &cones, &solve_opts(), backend)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    // The conic path returned no verified KKT point. A convex QCQP is still a
    // valid NLP — the same reasoning `SOCP_REFORM_FLOP_BUDGET` already uses to
    // route expensive-to-reformulate ones to the filter-IPM before solving — so
    // hand it to the NLP path
    // rather than reporting a failure with a working solver one branch away.
    //
    // `NumericalFailure` only. The other non-optimal statuses must NOT reroute:
    // `PrimalInfeasible`/`DualInfeasible` are verdicts the conic solver *did*
    // verify, and `IterationLimit` is the budget the caller asked for — it is
    // also what `max_iter=0` returns, whose zero-iteration contract (pounce#186)
    // requires stopping without a solve.
    //
    // Nothing has been printed yet: this sits above the status line, the `.sol`
    // write and the JSON report, so a rerouted solve emits exactly one verdict.
    if allow_nlp_fallback && matches!(sol.status, pounce_convex::QpStatus::NumericalFailure) {
        let res = sol.kkt_residuals_conic(&qp, &cones);
        eprintln!(
            "pounce: note: the conic ({}) solve returned no verified KKT point \
             after {} iterations (KKT error {:.2e}); a convex QCQP is also a \
             valid NLP, so it is being re-solved on the general NLP \
             interior-point path. Use solver_selection=socp to see the conic \
             result instead.",
            class.name(),
            sol.iters,
            res.kkt_error(),
        );
        return None;
    }

    // This solve is the one that reports, so what presolve did belongs on the
    // record after all.
    for line in &presolve_log {
        println!("{line}");
    }

    let reported_obj = sign * sol.obj + obj_const;

    let (msg, ok, srn) = convex_status_report(sol.status);
    println!(
        "POUNCE ({} conic IPM, pounce-convex): {msg}  obj={reported_obj:.8}  iters={}  ({elapsed:.3}s)",
        class.name(),
        sol.iters,
    );

    // Final KKT residuals from pounce-convex; reused for both the Ipopt-style
    // summary block and the JSON report below.
    // Cone-aware: the quadratic rows became SOC blocks, whose individual rows
    // legitimately violate `Gx ≤ h` at a perfectly feasible point, so the
    // orthant-only `kkt_residuals` reported a large bogus constraint violation
    // and NLP error for a solved problem (pounce#209).
    let res = sol.kkt_residuals_conic(&qp, &cones);
    // Ipopt-style summary so the objective/iteration count are scrapable by
    // consumers that parse Ipopt's end-of-run block (see print_convex_summary).
    print::print_convex_summary(
        sol.iters,
        reported_obj,
        res.primal_infeasibility,
        res.dual_infeasibility,
        res.complementarity,
        res.kkt_error(),
    );

    // Per-constraint duals, mapped from the cone multipliers back to `.nl`
    // constraint order (best-effort for the quadratic rows; see
    // `recover_socp_duals`).
    let lambda = pounce_cli::qp_extract::recover_socp_duals(prob, &con_map, &sol.y, &sol.z);

    // Bound multipliers (`ipopt_zL_out`/`ipopt_zU_out`). As on the QP path the
    // variable bounds live in the solver's explicit box — outside the cone
    // partition, which covers only the linear-inequality rows and the SOC
    // blocks — so they come back in `sol.z_lb`/`z_ub`. `sign` restores the
    // user objective sense and the Ipopt output convention is
    // `ipopt_zL_out = +z_l`, `ipopt_zU_out = −z_u` (gh #296).
    let (z_lb_raw, z_ub_raw) = pounce_cli::qp_extract::recover_bound_mults(prob, &sol);
    let z_l_suffix: Vec<f64> = z_lb_raw.iter().map(|&z| sign * z).collect();
    let z_u_suffix: Vec<f64> = z_ub_raw.iter().map(|&z| -sign * z).collect();
    let socp_bound_suffixes = [
        nl_writer::SolSuffix {
            name: "ipopt_zL_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_l_suffix),
        },
        nl_writer::SolSuffix {
            name: "ipopt_zU_out".to_string(),
            target: nl_writer::SolSuffixTarget::Var,
            values: nl_writer::SolSuffixValues::Real(z_u_suffix),
        },
    ];

    if let Some(path) = sol_path {
        let payload = nl_writer::SolutionFile {
            message: &format!("POUNCE {} conic IPM (pounce-convex): {msg}", class.name()),
            x: &sol.x,
            mult_g: &lambda,
            solve_result_num: srn,
            suffixes: &socp_bound_suffixes,
        };
        // Log a `.sol` write failure but do not early-return a distinct exit
        // code: the NLP path (main.rs:1091-1093) only logs, and under `-AMPL`
        // the final exit must still follow the solve-outcome contract.
        if let Err(e) = nl_writer::write_sol_file(path, &payload) {
            eprintln!("pounce: failed to write {}: {e}", path.display());
        }
    }

    if let Some((json_path, detail, input)) = json_cfg {
        let mut builder = ReportBuilder::new(detail, input);
        builder.problem.n_variables = qp.n as _;
        builder.problem.n_constraints = lambda.len() as _;
        builder.problem.n_objectives = 1;
        builder.problem.minimize = prob.minimize;
        builder.solution.status = qp_status_to_ars(sol.status);
        builder.solution.solve_result_num = srn;
        builder.solution.objective = reported_obj;
        builder.solution.x = sol.x.clone();
        builder.solution.lambda = lambda.clone();
        builder.stats.iteration_count = sol.iters as _;
        builder.stats.final_objective = reported_obj;
        builder.stats.total_wallclock_time_secs = elapsed;
        builder.stats.final_constr_viol = res.primal_infeasibility;
        builder.stats.final_dual_inf = res.dual_infeasibility;
        builder.stats.final_compl = res.complementarity;
        builder.stats.final_kkt_error = res.kkt_error();
        if matches!(detail, ReportDetail::Full) {
            builder.iterations = sol
                .iterates
                .iter()
                .map(|it| IterRecord {
                    iter: it.iter as _,
                    objective: it.objective,
                    inf_pr: it.primal_infeasibility,
                    inf_du: it.dual_infeasibility,
                    mu: it.mu,
                    alpha_primal: it.alpha_primal,
                    alpha_dual: it.alpha_dual,
                    ..IterRecord::default()
                })
                .collect();
        }
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

    Some(convex_exit_code(ok, ampl))
}

/// Process exit code for the convex (LP/QP/SOCP) solver paths, honoring the
/// AMPL solver-protocol contract. In `-AMPL` mode the termination is conveyed
/// through the `.sol` file's `solve_result_num`, so the process exits 0 for
/// any non-fatal solve outcome (infeasible, unbounded, iteration limit) just
/// as the NLP path does (main.rs:1103-1118) — a non-zero exit makes Pyomo /
/// the ASL interface raise `ApplicationError` and never read the `.sol`.
/// Genuine startup failures (bad `.nl`/option, unextractable problem) returned
/// non-zero earlier, before any solve, so reaching here in `-AMPL` mode means a
/// verdict was produced. Outside AMPL mode, an unsuccessful solve exits 1.
fn convex_exit_code(ok: bool, ampl: bool) -> ExitCode {
    if ok || ampl {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
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
    for (cat_str, spec_str) in dump_specs {
        let cat = DiagCategory::parse(cat_str)?;
        if cat == DiagCategory::Iterate {
            // `iterate:` accepts an extra `:summary` / `:full` variant
            // suffix after the iter filter. See parse_iterate_spec.
            let (filter, variant) = pounce_common::diagnostics::parse_iterate_spec(spec_str)?;
            config = config
                .with_category(cat, filter)
                .with_iterate_variant(variant);
        } else if cat == DiagCategory::Kkt {
            // `kkt:` accepts `+L` / `+L+Lvals` suffixes that pick up
            // the LDLᵀ factor's pattern (and optionally values). See
            // parse_kkt_spec.
            let (filter, variant) = pounce_common::diagnostics::parse_kkt_spec(spec_str)?;
            config = config.with_category(cat, filter).with_kkt_variant(variant);
        } else {
            let spec = IterSpec::parse(spec_str)?;
            config = config.with_category(cat, spec);
        }
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
///
/// `overall_alg_secs` is always populated. The
/// `linear_system_*` splits are detailed timers gated on
/// `timing_statistics` (default "no", issue #190), so they read `0.0`
/// unless the run set `timing_statistics yes` (or
/// `print_timing_statistics yes`, which implies it).
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

/// `--cite` output: the papers/software a user should cite when
/// publishing pounce results. Always lists the static core (pounce +
/// Wächter-Biegler); when `--cite <report.json>` supplies a solve
/// report, adds solve-aware extras for features the run used. `--bibtex`
/// switches the rendering to BibTeX. See [`pounce_cli::citations`].
fn run_cite(args: &Args) -> ExitCode {
    let report = match &args.cite_report {
        Some(path) => {
            let text = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("pounce: failed to read {}: {e}", path.display());
                    return ExitCode::from(2);
                }
            };
            match serde_json::from_str::<pounce_cli::solve_report::SolveReport>(&text) {
                Ok(r) => Some(r),
                Err(e) => {
                    eprintln!(
                        "pounce: {} is not a valid solve report: {e}",
                        path.display()
                    );
                    // Common mistake: passing the model (`.nl`) instead of a
                    // solve-report JSON. `--cite` takes the report produced by
                    // a prior solve (`--json-output out.json`), not the model;
                    // bare `pounce --cite` prints the static core with no run.
                    if path.extension().and_then(|e| e.to_str()) == Some("nl") {
                        eprintln!(
                            "pounce: --cite expects a solve-report JSON, not a model file. \
                             Run `pounce {} --json-output report.json` first, then \
                             `pounce --cite report.json` — or use bare `pounce --cite` for the core citations.",
                            path.display()
                        );
                    }
                    return ExitCode::from(2);
                }
            }
        }
        None => None,
    };

    let selected = pounce_cli::citations::select(report.as_ref());
    if args.cite_bibtex {
        print!("{}", pounce_cli::citations::render_bibtex(&selected));
    } else {
        print!("{}", pounce_cli::citations::render_human(&selected));
    }
    ExitCode::SUCCESS
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
    println!(
        "  ma57            HSL MA57 via libcoinhsl       (not compiled; resolves to FERAL at runtime)"
    );
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
                LinearSolverChoice::Feral => Box::new(
                    pounce_feral::FeralSolverInterface::with_config(feral_cfg.clone()),
                ),
                LinearSolverChoice::Ma57 => {
                    #[cfg(feature = "ma57")]
                    {
                        Box::new(pounce_hsl::Ma57SolverInterface::new())
                    }
                    #[cfg(not(feature = "ma57"))]
                    {
                        Box::new(pounce_feral::FeralSolverInterface::with_config(
                            feral_cfg.clone(),
                        ))
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod convex_status_tests {
    use super::{convex_status_report, qp_status_to_ars};
    use pounce_convex::QpStatus;
    use pounce_nlp::return_codes::ApplicationReturnStatus;

    /// Code review 2026-06 item M20: the reduced-accuracy convex status
    /// (`OptimalInaccurate`) must surface to the user as a *distinct* outcome —
    /// not silently folded into a clean `Optimal`. It maps to AMPL
    /// `solve_result_num` 1 (Ipopt's own code for an accepted reduced-accuracy
    /// solve) with a distinct message, and onto the NLP-side
    /// `SolvedToAcceptableLevel` status, so callers reading either the `.sol`
    /// terminal line or the JSON report can tell it apart from a full-accuracy
    /// solve.
    ///
    /// The code moved out of the 100 band in gh #591 — see
    /// `pounce_solve_report::status_to_solve_result_num` — and must agree with
    /// the NLP path, which reports the same status.
    #[test]
    fn optimal_inaccurate_is_distinct_from_optimal() {
        let (msg, ok, srn) = convex_status_report(QpStatus::OptimalInaccurate);
        assert_eq!(
            srn,
            pounce_cli::solve_report::status_to_solve_result_num(
                ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "the convex and NLP paths must report one code for one status",
        );
        assert_eq!(
            srn, 1,
            "Ipopt's code for an accepted reduced-accuracy solve"
        );
        assert!(ok, "a reduced-accuracy solve is still a usable success");
        assert!(
            msg.contains("acceptable"),
            "message should signal reduced accuracy, got {msg:?}"
        );

        let (opt_msg, _, opt_srn) = convex_status_report(QpStatus::Optimal);
        assert_eq!(opt_srn, 0);
        assert_ne!(
            srn, opt_srn,
            "OptimalInaccurate must not share Optimal's solve_result_num"
        );
        assert_ne!(msg, opt_msg, "the two must read differently to the user");

        // And on the NLP-side status vocabulary used by the JSON report.
        assert_eq!(
            qp_status_to_ars(QpStatus::OptimalInaccurate),
            ApplicationReturnStatus::SolvedToAcceptableLevel
        );
        assert_eq!(
            qp_status_to_ars(QpStatus::Optimal),
            ApplicationReturnStatus::SolveSucceeded
        );
    }

    /// A declined convex attempt is charged against the wall-clock budget, so
    /// the NLP solve that takes over cannot start a second full one.
    ///
    /// Built on a real `IpoptApplication` rather than a bare `OptionsList`
    /// because the two ways this write can silently do nothing — the option's
    /// *strict* lower bound of zero, and the clobber flag on the stored value —
    /// both live in the registration that only the real one carries.
    #[test]
    fn a_declined_convex_attempt_is_charged_against_the_wall_budget() {
        use std::time::Duration;

        let mut app = super::IpoptApplication::new();
        app.options_mut()
            .set_numeric_value("max_wall_time", 60.0, true, false)
            .unwrap();

        super::charge_wall_budget(app.options_mut(), Duration::from_secs_f64(55.0));
        let (left, set) = app
            .options()
            .get_numeric_value("max_wall_time", "")
            .unwrap();
        assert!(set, "the option must still read as explicitly set");
        assert!(
            (left - 5.0).abs() < 1e-9,
            "60s budget minus a 55s attempt must leave 5s, got {left}"
        );

        // A budget spent outright must not silently write back as the full
        // budget: `max_wall_time` is registered with a strict lower bound, so a
        // literal 0.0 would be rejected and leave 5s standing.
        super::charge_wall_budget(app.options_mut(), Duration::from_secs_f64(600.0));
        let (gone, _) = app
            .options()
            .get_numeric_value("max_wall_time", "")
            .unwrap();
        assert!(
            gone > 0.0 && gone < 1e-6,
            "an exhausted budget must store as positive-but-spent, got {gone}"
        );
    }

    /// The other half: an *unset* budget is left alone. `1e6` is the
    /// effectively-unbounded default, and rewriting it would both say nothing
    /// new and make the option read as user-chosen downstream.
    #[test]
    fn an_unset_wall_budget_is_not_rewritten() {
        use std::time::Duration;

        let mut app = super::IpoptApplication::new();
        let (before, set_before) = app
            .options()
            .get_numeric_value("max_wall_time", "")
            .unwrap();
        assert!(!set_before, "precondition: the option starts unset");

        super::charge_wall_budget(app.options_mut(), Duration::from_secs_f64(3.2));
        let (after, set_after) = app
            .options()
            .get_numeric_value("max_wall_time", "")
            .unwrap();
        assert_eq!(after, before);
        assert!(
            !set_after,
            "an untouched budget must not read as explicitly set"
        );
    }

    #[test]
    fn time_limit_maps_to_wall_clock_status() {
        let (msg, ok, srn) = convex_status_report(QpStatus::TimeLimit);
        assert_eq!(msg, "Maximum wallclock time exceeded.");
        assert!(!ok);
        assert_eq!(srn, 400);
        assert_eq!(
            qp_status_to_ars(QpStatus::TimeLimit),
            ApplicationReturnStatus::MaximumWallTimeExceeded
        );
    }
}

#[cfg(test)]
mod lp_nlp_fallback_tests {
    use super::lp_declines_to_nlp;
    use pounce_cli::dispatch::ProblemClass;
    use pounce_convex::QpStatus;

    const ALL_STATUSES: [QpStatus; 7] = [
        QpStatus::Optimal,
        QpStatus::OptimalInaccurate,
        QpStatus::PrimalInfeasible,
        QpStatus::DualInfeasible,
        QpStatus::IterationLimit,
        QpStatus::TimeLimit,
        QpStatus::NumericalFailure,
    ];

    /// gh #535: the statuses that mean "the convex solve produced no
    /// certificate" are what hands an LP to the NLP path. `OptimalInaccurate`
    /// is the NETLIB `gen`/`gen1` exit (199 of 200 iterations, primal residual
    /// 1.4e-7 against `tol = 1e-8`); `IterationLimit` is the same stall when
    /// the reduced-accuracy band is missed too; `NumericalFailure` (gh #724)
    /// is the post-solve verification refusing the point outright, which is a
    /// stronger statement of the same thing and not a weaker one.
    #[test]
    fn an_uncertified_lp_is_handed_to_the_nlp_path() {
        for status in [
            QpStatus::OptimalInaccurate,
            QpStatus::IterationLimit,
            QpStatus::NumericalFailure,
        ] {
            assert!(
                lp_declines_to_nlp(ProblemClass::Lp, status, true),
                "{status:?} on an LP must reroute"
            );
        }
    }

    /// A certified result is the answer — there is nothing for a second solve
    /// to improve, and running one would double the cost of every LP.
    #[test]
    fn a_certified_lp_is_never_rerouted() {
        assert!(!lp_declines_to_nlp(
            ProblemClass::Lp,
            QpStatus::Optimal,
            true
        ));
    }

    /// `PrimalInfeasible` / `DualInfeasible` are verdicts the convex solver
    /// *verified*. Rerouting them would let a second solve overwrite a proof
    /// with a numerical opinion — the same reason `run_convex_socp` reroutes
    /// only `NumericalFailure` and not these.
    #[test]
    fn verified_verdicts_stand() {
        for status in [QpStatus::PrimalInfeasible, QpStatus::DualInfeasible] {
            assert!(
                !lp_declines_to_nlp(ProblemClass::Lp, status, true),
                "{status:?} must not reroute"
            );
        }
    }

    /// gh #724: the LP gate and the SOCP gate must agree about what an
    /// uncertified convex result means. `run_convex_socp` reroutes exactly
    /// `NumericalFailure`; if the LP gate excludes it, the same failure to
    /// verify is a fallback on one path and a final `InternalError` on the
    /// other. This is the assertion that was inverted before gh #724, so it is
    /// stated as the invariant rather than as one more status in a list.
    #[test]
    fn an_unverified_convex_result_reroutes_on_the_lp_path_as_it_does_on_the_conic_one() {
        assert!(
            lp_declines_to_nlp(ProblemClass::Lp, QpStatus::NumericalFailure, true),
            "NumericalFailure is what the conic path reroutes on; the LP path \
             must not report it as the last word"
        );
    }

    /// A wall-clock budget is a budget, exactly as `max_iter` is: `TimeLimit`
    /// is the answer to the question the user asked. Rerouting it would launch
    /// a *second*, unbudgeted solve on a problem whose whole point was to stop
    /// — the fallback would double the time limit it was told to respect.
    #[test]
    fn a_spent_time_budget_is_not_a_reason_to_solve_again() {
        assert!(!lp_declines_to_nlp(
            ProblemClass::Lp,
            QpStatus::TimeLimit,
            true
        ));
    }

    /// The issue scopes the fallback to `P = 0`. A convex QP that stalls is a
    /// different and unmeasured population, so no status reroutes it — nor
    /// does any class the convex QP driver never sees.
    #[test]
    fn only_the_lp_class_reroutes() {
        for class in [
            ProblemClass::ConvexQp,
            ProblemClass::ConvexQcqp,
            ProblemClass::NonconvexQp,
            ProblemClass::Nlp,
        ] {
            for status in ALL_STATUSES {
                assert!(
                    !lp_declines_to_nlp(class, status, true),
                    "{class:?}/{status:?} must not reroute"
                );
            }
        }
    }

    /// The caller's gate wins outright: an explicitly named engine, a user-set
    /// `max_iter` (including the `max_iter=0` zero-iteration contract,
    /// pounce#186) or an attached debugger clears it, and then nothing
    /// reroutes.
    #[test]
    fn the_callers_gate_suppresses_every_case() {
        for class in [ProblemClass::Lp, ProblemClass::ConvexQp] {
            for status in ALL_STATUSES {
                assert!(
                    !lp_declines_to_nlp(class, status, false),
                    "{class:?}/{status:?} must not reroute when the caller declines"
                );
            }
        }
    }
}

#[cfg(test)]
mod nlp_exit_code_tests {
    //! Code review L27: the module doc claimed exit 0 only on `Solve_Succeeded`,
    //! but the NLP path also (correctly) exits 0 on `SolvedToAcceptableLevel`.
    //! The doc was corrected; these tests lock the actual behavior so the doc
    //! and code can't drift again.
    use super::nlp_solve_succeeded;
    use pounce_nlp::return_codes::ApplicationReturnStatus as A;

    #[test]
    fn acceptable_level_counts_as_success() {
        // The crux of L27: reduced-accuracy convergence is a success.
        assert!(nlp_solve_succeeded(A::SolvedToAcceptableLevel));
        assert!(nlp_solve_succeeded(A::SolveSucceeded));
    }

    #[test]
    fn non_convergent_statuses_are_not_success() {
        for s in [
            A::InfeasibleProblemDetected,
            A::MaximumIterationsExceeded,
            A::RestorationFailed,
            A::DivergingIterates,
            A::MaximumCpuTimeExceeded,
            A::InternalError,
        ] {
            assert!(
                !nlp_solve_succeeded(s),
                "{s:?} must not count as a successful solve"
            );
        }
    }
}
