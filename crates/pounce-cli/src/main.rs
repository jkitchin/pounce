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
use pounce_cli::counting_tnlp::CountingTnlp;
use pounce_cli::nl_reader;
use pounce_cli::nl_writer;
use pounce_cli::print;
use pounce_cli::sens;
use pounce_cli::solve_report::{
    status_to_solve_result_num, write_report_file, InputDescriptor, ReportBuilder, ReportDetail,
    SolutionSuffix,
};
use pounce_common::diagnostics::{
    DiagCategory, DiagnosticsConfig, DiagnosticsState, DumpFormat, IterSpec,
};
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::IterRecord;
use pounce_nlp::tnlp::TNLP;
use pounce_nlp::SolveStatistics;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;

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

    // Register the LP/QP routing option so `solver_selection=...` is
    // accepted by the (validating) options parser. See the dispatch plan
    // (dev-notes/lp-qp-routing.md): `auto` routes classified LP / convex
    // QP problems to the specialized `pounce-convex` IPM and everything
    // else to the NLP filter-IPM; forcing values are validated against
    // the detected class.
    if let Err(e) = app.registered_options().add_string_option(
        "solver_selection",
        "Which solver to route the problem to.",
        "auto",
        &[
            (
                "auto",
                "Most specialized solver matching the detected problem class.",
            ),
            (
                "nlp",
                "Always the filter-IPM NLP solver (current default behavior).",
            ),
            (
                "lp-ipm",
                "Force IPM-LP; errors if the problem is not an LP.",
            ),
            (
                "qp-ipm",
                "Force IPM-QP; errors if the problem is not LP/convex-QP.",
            ),
            (
                "qp-active-set",
                "Force active-set QP; errors if not LP/convex-QP.",
            ),
            (
                "socp",
                "Force the SOCP conic IPM; errors if not a convex LP/QP/QCQP.",
            ),
        ],
        "Selects the solver by problem class. `auto` routes LP and convex \
         QP to the specialized convex interior-point solver (pounce-convex), \
         a convex QCQP to the SOCP conic IPM, and all other classes to the \
         NLP filter-IPM. `qp-active-set` routes an LP / convex QP through the \
         active-set SQP engine (pounce-qp QP subproblems) instead of the IPM; \
         on these classes it converges in essentially one QP solve. `socp` \
         forces the conic IPM (a convex QCQP routes there under `auto`).",
    ) {
        eprintln!("pounce: failed to register solver_selection option: {e}");
        return ExitCode::from(2);
    }

    // Toggle presolve on the convex LP/QP path. Default on.
    if let Err(e) = app.registered_options().add_string_option(
        "qp_presolve",
        "Run presolve before the convex LP/QP interior-point solve.",
        "yes",
        &[
            ("yes", "Reduce the problem (and detect trivial infeasibility / unboundedness) before solving."),
            ("no", "Solve the extracted problem directly, without presolve."),
        ],
        "Only affects the convex LP/QP path (`solver_selection` routing to \
         pounce-convex). When on, presolve removes empty / duplicate / \
         redundant rows, fixes and substitutes structural columns, and may \
         report infeasible / unbounded without invoking the solver.",
    ) {
        eprintln!("pounce: failed to register qp_presolve option: {e}");
        return ExitCode::from(2);
    }

    // ---- Convex LP/QP interior-point knobs (pounce-convex `QpOptions`) ----
    // These only affect the `solver_selection` paths that route to
    // pounce-convex (`lp-ipm` / `qp-ipm` / `auto` on an LP / convex QP, and
    // the SOCP IPM). Each forwards only when the user *explicitly* sets it
    // (see the convex dispatch below); otherwise the driver keeps its own
    // tuned default. The standard `tol` and `max_iter` options feed the
    // convex solve too — these `qp_*` knobs cover the rest of `QpOptions`.
    let convex_knobs: Result<(), pounce_common::SolverException> = (|| {
        let r = app.registered_options();
        // τ ∈ (0,1): fraction-to-boundary step damping (Mehrotra σ is adaptive).
        r.add_bounded_number_option(
            "qp_tau",
            "Convex IPM fraction-to-boundary parameter τ ∈ (0,1).",
            0.0,
            true,
            1.0,
            true,
            0.95,
            "Convex LP/QP interior-point only. Caps each Newton step at a \
             fraction τ of the distance to the cone boundary; nearer 1 is more \
             aggressive. Default 0.95.",
        )?;
        // Static KKT regularization δ ≥ 0.
        r.add_lower_bounded_number_option(
            "qp_reg",
            "Convex IPM static KKT regularization δ ≥ 0.",
            0.0,
            false,
            1e-10,
            "Convex LP/QP interior-point only. Added on the (block) diagonal to \
             keep the reduced KKT quasi-definite for a stable LDLᵀ inertia. Too \
             large freezes the primal residual on badly-scaled LPs; the default \
             1e-10 is centered in the band that converges the LP/QP suites.",
        )?;
        // Certificate value / cone-membership tolerance > 0.
        r.add_lower_bounded_number_option(
            "qp_infeas_tol",
            "Convex IPM infeasibility-certificate value tolerance > 0.",
            0.0,
            true,
            1e-7,
            "Convex LP/QP interior-point only. Relative tolerance on the value \
             and cone-membership parts of an infeasibility / unboundedness \
             certificate. The certificate's defining-equation residual is held \
             to a far tighter internal tolerance; this only governs when a \
             status is backed by a verified proof. Default 1e-7.",
        )?;
        r.add_string_option(
            "qp_hsde",
            "Use the homogeneous self-dual embedding for the convex IPM.",
            "yes",
            &[
                ("yes", "Self-dual embedding: self-starting, native certificates, robust on ill-conditioned data."),
                ("no", "Infeasible-start primal–dual method (the warm-start / build-once substrate)."),
            ],
            "Convex LP/QP interior-point only. HSDE (default) self-starts and \
             produces infeasibility / unboundedness certificates natively; it \
             is also the substrate for non-symmetric cones. Default yes.",
        )?;
        r.add_string_option(
            "qp_equilibrate",
            "Ruiz-equilibrate the data before the direct convex IPM solve.",
            "yes",
            &[
                (
                    "yes",
                    "Apply Ruiz row/column scaling before solving (direct, non-HSDE path).",
                ),
                ("no", "Solve the raw data without equilibration."),
            ],
            "Convex LP/QP interior-point only, and only when `qp_hsde=no` (the \
             direct infeasible-start path): a conditioning aid for the raw KKT \
             factorization. HSDE conditions internally and ignores this. \
             Default yes.",
        )?;
        r.add_string_option(
            "qp_crossover",
            "Run LP crossover to purify the IPM iterate to an exact vertex.",
            "no",
            &[
                ("yes", "After the IPM, pivot the interior iterate to an exact optimal vertex (active-set purification)."),
                ("no", "Return the interior-point iterate directly (default)."),
            ],
            "Convex LP path only (pure LP, P=0); a no-op for genuine QPs. \
             Correct (never-regress) but currently slow on the degenerate / \
             large NETLIB LPs it targets and does not yet reach an exact \
             `Optimal` vertex on the GEN family (issue #133), so it is off by \
             default and offered as an opt-in for small, well-behaved LPs that \
             want exact-vertex refinement. Default no.",
        )?;
        Ok(())
    })();
    if let Err(e) = convex_knobs {
        eprintln!("pounce: failed to register convex LP/QP options: {e}");
        return ExitCode::from(2);
    }

    // ---- Active-set SQP QP-subproblem knobs (pounce-qp `QpOptions`) ----
    // Consulted only on `solver_selection=qp-active-set` (the active-set SQP
    // engine, whose step QPs are solved by pounce-qp). Read into the algorithm
    // builder by `application::apply_qp_subproblem_options`; each forwards only
    // when explicitly set, otherwise the pounce-qp default stands. The outer
    // SQP loop has its own `sqp_*` family (registered with the SQP options);
    // these `sqp_qp_*` knobs tune the inner QP solver specifically.
    let sqp_qp_knobs: Result<(), pounce_common::SolverException> = (|| {
        let r = app.registered_options();
        r.add_lower_bounded_number_option(
            "sqp_qp_feas_tol",
            "Active-set QP-subproblem feasibility tolerance > 0.",
            0.0,
            true,
            1e-9,
            "Active-set SQP only (solver_selection=qp-active-set). Constraint \
             feasibility tolerance for the pounce-qp subproblem solve. \
             Default 1e-9.",
        )?;
        r.add_lower_bounded_number_option(
            "sqp_qp_opt_tol",
            "Active-set QP-subproblem optimality (KKT) tolerance > 0.",
            0.0,
            true,
            1e-9,
            "Active-set SQP only. Optimality / KKT tolerance for the pounce-qp \
             subproblem solve. Default 1e-9.",
        )?;
        r.add_lower_bounded_integer_option(
            "sqp_qp_max_iter",
            "Active-set QP-subproblem iteration cap.",
            1,
            200,
            "Active-set SQP only. Maximum active-set pivots per QP subproblem \
             solve. Default 200.",
        )?;
        r.add_lower_bounded_number_option(
            "sqp_qp_elastic_gamma",
            "Active-set QP-subproblem elastic-mode penalty γ > 0.",
            0.0,
            true,
            1e6,
            "Active-set SQP only. Penalty on the elastic (phase-1) slacks used \
             to recover from an infeasible QP subproblem. Large enough that the \
             slacks vanish at the solution of a feasible QP, small enough not to \
             dominate the Hessian conditioning. Default 1e6.",
        )?;
        r.add_string_option(
            "sqp_qp_anti_cycling",
            "Active-set QP-subproblem anti-cycling rule.",
            "expand",
            &[
                ("expand", "EXPAND tolerance-growth + Harris two-pass (Gill-Murray-Saunders-Wright 1989). Default."),
                ("bland", "Bland's rule: slower but guaranteed finite; mainly for tests."),
                ("none", "No anti-cycling — benchmarking only; may cycle on degenerate QPs."),
            ],
            "Active-set SQP only. Anti-cycling strategy for the pounce-qp \
             subproblem ratio test. Default expand.",
        )?;
        Ok(())
    })();
    if let Err(e) = sqp_qp_knobs {
        eprintln!("pounce: failed to register active-set QP options: {e}");
        return ExitCode::from(2);
    }

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
    // ℓ1-feasibility sub-IPM. Mirrors what upstream's `IpAlgBuilder`
    // does unconditionally for every solve.
    //
    // Capture the feral config off the now-fully-loaded options so the
    // restoration sub-IPM honors the same `feral_*` overrides (e.g.
    // `feral_cascade_break yes` from an `--options-file`) as the main
    // IPM. Snapshot, not borrow: the BFF outlives the option-mutation
    // window we cleanly own here.
    let feral_cfg = pounce_algorithm::application::feral_config_from_options(app.options());
    // Use the multi-pass provider so the ℓ₁ wrapper (`l1_exact_penalty_barrier`)
    // and the auto-fallback (`l1_fallback_on_restoration_failure`) don't
    // panic with "restoration factory invoked more than once" on their
    // second inner solve — see pounce#10 Phase 3 / pounce#24.
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone()))
    };
    // Hand the inner IPM a builder mirroring the outer options so its
    // `mu_strategy` (adaptive vs. monotone) inherits the user's choice —
    // matches upstream `IpAlgBuilder::BuildRestoIpoptAlgorithm`.
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(resto_provider);

    // Branded logo + copyright banner, printed up-front — before the
    // problem is even read — so they head the output. The registered
    // default for `linear_solver` mirrors upstream IPOPT (`"ma57"`), but
    // pounce's actual backend is FERAL; only treat `"ma57"` as user
    // intent when explicitly set, else the banner would always claim
    // "ma57 requested". `sb yes` suppresses both (mirrors upstream
    // `IpoptApplication::Initialize`).
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
    // Problem class captured from the *first* `.nl` parse below, so the
    // LP/QP dispatch never has to re-read the file just to classify it
    // (re-parsing doubled parse time / peak memory on large models — code
    // review L24). `None` for builtins (treated as general NLP).
    let mut nl_class: Option<pounce_cli::dispatch::ProblemClass> = None;
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

    // Multistart / find-minima: when a `--minima` method is set, drive the
    // local solver in a loop over the *raw* problem TNLP (presolve / counting
    // wrappers are intentionally bypassed so coordinates match the original
    // problem and the clean objective is evaluated directly) and return.
    if let Some(mcfg) = &args.minima {
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
        use pounce_cli::dispatch::{resolve_solver, ProblemClass, SolverChoice, SolverSelection};
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

        // Banner-level routing line: report the detected problem class and
        // which of pounce's solvers was selected for it. Gated like the
        // banner (suppressed by `sb yes` and in JSON-debug protocol mode) so
        // stdout stays clean for machine consumers.
        if !suppress_banner && !json_dbg {
            println!(
                "Problem class: {}. Selected solver: {} [solver_selection={}].",
                class.name(),
                choice.describe(),
                sel_str
            );
            println!();
        }

        // Dispatch to the specialized convex solvers when resolved.
        // `LpIpm`/`QpIpm` use the convex QP IPM (LP is P = 0); `SocpIpm`
        // reformulates a convex QCQP to second-order cones and uses the
        // conic IPM. Both live in `pounce-convex`.
        if matches!(
            choice,
            SolverChoice::LpIpm | SolverChoice::QpIpm | SolverChoice::SocpIpm
        ) {
            // The convex solvers need the parsed `NlProblem`, but the initial
            // parse moved it into `NlTnlp`. Re-parse the file here — only on
            // the convex dispatch path (LP / convex-QP / SOCP), never for a
            // general NLP solve. Only `.nl` inputs ever classify as convex, so
            // the builtin arm falls through to NLP. A parse failure surfaces
            // and exits rather than silently mis-routing to NLP (L24).
            if let ProblemSource::NlFile(path) = &args.problem {
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
                // Build the convex IPM options from the registered CLI knobs.
                // Each tunable forwards only when the user *explicitly* set it
                // (the `true` flag from `get_*_value`); otherwise the convex
                // driver keeps its own tuned `QpOptions` default. `max_iter` in
                // particular must not be silently raised to the (far larger)
                // Ipopt default, so it too is forwarded only when set.
                let convex_opts = convex_cli_opts(&app);
                if matches!(choice, SolverChoice::SocpIpm) {
                    return run_convex_socp(
                        &prob,
                        class,
                        sol_path.as_deref(),
                        json_cfg,
                        debug_hook.as_ref(),
                        args.ampl,
                        convex_opts,
                    );
                }
                // Resolve the convex-path presolve switch (#139). See
                // `resolve_convex_presolve` for the aliasing rationale.
                let opts = app.options();
                let presolve_on = resolve_convex_presolve(
                    opts.get_string_value("qp_presolve", "").ok(),
                    opts.get_string_value("presolve", "").ok(),
                );
                return run_convex_qp(
                    &prob,
                    class,
                    sol_path.as_deref(),
                    presolve_on,
                    json_cfg,
                    debug_hook.as_ref(),
                    args.ampl,
                    convex_opts,
                );
            }
            // Builtins never classify as convex; fall through to NLP.
        }
        // `qp-active-set`: route the (convex-QP) problem through the
        // active-set SQP engine instead of the IPM. `resolve_solver`
        // already validated the class is LP / convex QP, so the SQP driver
        // — which solves its step QPs with `pounce-qp` — converges in
        // essentially one QP solve, and the NLP layer recovers the duals
        // and writes the `.sol` exactly as the IPM path does. The
        // application dispatches to that engine whenever the `algorithm`
        // option resolves to "active-set-sqp" (`optimize_tnlp` →
        // `optimize_sqp_tnlp`), so setting the option here is the whole
        // wiring; the solve falls through to the NLP path below unchanged.
        if matches!(choice, SolverChoice::QpActiveSet) {
            if let Err(e) = app
                .options_mut()
                .read_from_str("algorithm active-set-sqp\n", true)
            {
                eprintln!("pounce: failed to select the active-set-sqp algorithm: {e}");
                return ExitCode::from(2);
            }
            // The interactive debugger is a pdb-for-the-IPM: it pauses on
            // barrier-IPM iterations (mu, search direction, fraction-to-the-
            // boundary). The active-set SQP engine is a different algorithm
            // with no such hook, so a `--debug*` request here would otherwise
            // silently no-op. Say so explicitly rather than pretend it engaged.
            if debug_hook.is_some() {
                eprintln!(
                    "pounce: note: the interactive debugger is IPM-only and does \
                     not engage on the active-set QP engine (solver_selection=\
                     qp-active-set); the solve runs without pausing. Use \
                     solver_selection=qp-ipm to debug a convex QP interactively."
                );
            }
        }
        // `nlp` and any unmatched case fall through to the existing NLP
        // solve below unchanged.
        let _ = choice;
    }

    // Does the `.nl` ask for a parametric sensitivity step? When it
    // does, the post-optimal step runs inside `on_converged` below and
    // its result is written back as the `sens_sol_state_1` suffix.
    let sens_active = nl_suffixes
        .as_ref()
        .map(sens::is_sensitivity_input)
        .unwrap_or(false);

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
    let red_hessian_capture: Rc<RefCell<Option<sens::RedHessianResult>>> =
        Rc::new(RefCell::new(None));
    if args.json_output.is_some() || sol_path.is_some() || sens_active || args.compute_red_hessian {
        let cap = Rc::clone(&nominal_capture);
        let sens_cap = Rc::clone(&sens_capture);
        let rh_cap = Rc::clone(&red_hessian_capture);
        let suffixes_cb = nl_suffixes.clone();
        let dims_cb = nl_dims;
        let compute_rh = args.compute_red_hessian;
        let rh_eigen = args.rh_eigendecomp;
        let boundcheck_eps = args.sens_boundcheck.then_some(args.sens_bound_eps);
        app.set_on_converged(Box::new(move |data, cq, nlp, pd| {
            let curr = match data.borrow().curr.clone() {
                Some(c) => c,
                None => return,
            };
            // Lift to full length so a fixed / eliminated variable
            // still occupies its slot — AMPL's `.sol` reader matches
            // the x block against the originating `.nl`'s var count.
            let x = nlp.borrow().lift_x_to_full(&*curr.x);
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
                        &x,
                        boundcheck_eps,
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
    if (sens_active || args.compute_red_hessian) && presolve_opts.enabled {
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
    let post_presolve: Rc<RefCell<dyn TNLP>> = match &presolve_handle {
        Some(p) => Rc::clone(p) as Rc<RefCell<dyn TNLP>>,
        None => Rc::clone(&inner_tnlp),
    };

    // Wrap so we can pull eval-call counts out for the final summary.
    let counting = Rc::new(RefCell::new(CountingTnlp::new(Rc::clone(&post_presolve))));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&counting) as Rc<RefCell<dyn TNLP>>;

    // Problem statistics. (The branded logo + copyright banner print
    // up-front, before the problem is read — see near the top of `run`.)
    // Suppressed in JSON-debug mode so stdout stays a pure protocol stream.
    if !json_dbg {
        // Resolve `fixed_variable_treatment` + bound-infinity exactly as
        // `application.rs` does, so the banner reports the same reduced
        // problem the algorithm solves (#140).
        use pounce_nlp::tnlp_adapter::{
            FixedVarTreatment, DEFAULT_NLP_LOWER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF,
        };
        let opt = app.options();
        let lo_inf = opt
            .get_numeric_value("nlp_lower_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_LOWER_BOUND_INF);
        let up_inf = opt
            .get_numeric_value("nlp_upper_bound_inf", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(DEFAULT_NLP_UPPER_BOUND_INF);
        let fixed_treatment = match opt
            .get_string_value("fixed_variable_treatment", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .as_deref()
        {
            Some("relax_bounds") => FixedVarTreatment::RelaxBounds,
            // `make_constraint` / `make_parameter_nodual` are not yet
            // implemented and fall back to `make_parameter` in the solve.
            _ => FixedVarTreatment::MakeParameter,
        };
        if let Some(stats) = print::collect_stats(&tnlp, lo_inf, up_inf, fixed_treatment) {
            print::print_problem_stats(&stats);
        }
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
        solve_tnlp = Rc::new(RefCell::new(pounce_cli::seeded_tnlp::SeededTnlp::new(
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

    // Hypersensitivity scaling fallback (`feral_infeasibility_scaling_retry`,
    // on by default). Some interior-point KKT trajectories are chaotic: under
    // two equally backward-stable linear-solver scalings the iterates stay
    // bit-identical for many iterations, then diverge by ~1 ULP and fall into
    // different basins — one optimal, the other a spurious stationary point of
    // the constraint violation reported as local infeasibility (discs.nl:
    // InfNorm → infeasible, MC64/Identity/MA57/IPOPT → optimal). It is
    // sensitive dependence, not a bad solve, so the a-priori scaling router
    // can't tell the two apart and no per-factor residual flags it; the only
    // reliable signal is the whole-solve verdict. So: on a local-infeasibility
    // verdict under a non-MC64 effective scaling, re-solve ONCE with MC64
    // before believing it, and promote only if MC64 actually succeeds.
    let scaling_retry_enabled = app
        .options()
        .get_bool_value("feral_infeasibility_scaling_retry", "")
        .map(|(v, _found)| v)
        .unwrap_or(true);
    let already_mc64 = matches!(
        pounce_algorithm::application::feral_config_from_options(app.options()).scaling,
        pounce_feral::ScalingStrategy::Mc64Symmetric
    );
    if scaling_retry_enabled
        && debug_hook.is_none()
        && !already_mc64
        && status == ApplicationReturnStatus::InfeasibleProblemDetected
    {
        eprintln!(
            "pounce: local infeasibility under the current FERAL scaling — re-solving once with \
             MC64 before believing it (discs-class hypersensitivity guard; \
             feral_infeasibility_scaling_retry)…"
        );
        // Flip the scaling for the retry. The main IPM rereads `feral_scaling`
        // fresh each solve, but the restoration sub-IPM uses the provider we
        // snapshotted above at the original scaling — so rebuild it too, or the
        // restoration leg would stay on the failing scaling.
        let _ = app
            .options_mut()
            .read_from_str("feral_scaling mc64\n", true);
        let feral_cfg = pounce_algorithm::application::feral_config_from_options(app.options());
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

        let retry_status = app.optimize_tnlp(Rc::clone(&tnlp));
        let retry_stats = app.statistics();
        if scaling_retry_promoted(retry_status) {
            eprintln!(
                "pounce: MC64 re-solve recovered the problem — promoting ({retry_status:?})."
            );
        } else {
            eprintln!(
                "pounce: MC64 re-solve did not recover ({retry_status:?}); keeping the original \
                 local-infeasibility verdict (now corroborated by a second scaling)."
            );
        }
        // Keep `status` and `solve_stats` in lockstep: on promotion the retry
        // is authoritative (its verdict + its statistics); otherwise both stay
        // the original local-infeasibility verdict and the original solve's
        // statistics. See `resolve_scaling_retry_outcome` (code review L23).
        (status, solve_stats) =
            resolve_scaling_retry_outcome(retry_status, solve_stats, retry_stats);
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
    } else {
        print::print_summary(status, &solve_stats, &counters);
    }
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
    if let Some(p) = &presolve_handle {
        let lifted = if p.borrow().n_dropped_rows() > 0 {
            p.borrow().finalized_full_solution().map(|(_x, lam)| lam)
        } else {
            None
        };
        if let Some(lam_full) = lifted {
            if let Some((_x, lambda)) = nominal_capture.borrow_mut().as_mut() {
                *lambda = lam_full;
            }
        }
    }

    // Reduced Hessian: print to stderr (informational), mirroring
    // upstream sIPOPT's RedHessian / Eigenvalues prints in
    // `SensReducedHessianCalculator.cpp`.
    if let Some(rh) = red_hessian_capture.borrow().as_ref() {
        sens::print_red_hessian_to_stderr(rh);
    } else if args.compute_red_hessian {
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
        builder.solution.solve_result_num = status_to_solve_result_num(status);
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
        let message = format!("POUNCE {}: {status:?}", env!("CARGO_PKG_VERSION"));
        let payload = nl_writer::SolutionFile {
            message: &message,
            x: &x,
            lambda: &lambda,
            solve_result_num: status_to_solve_result_num(status),
            suffixes: &sol_suffixes,
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

/// Did an MC64 hypersensitivity re-solve converge well enough to overturn the
/// original local-infeasibility verdict? Only a clean or acceptable-level solve
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
        QpStatus::NumericalFailure => ApplicationReturnStatus::InternalError,
    }
}

/// Map a convex-solver status onto the AMPL `.sol` terminal line: the message,
/// whether the solve is treated as a success (drives the exit code), and the
/// `solve_result_num`. AMPL convention: 0 solved, 100–199 solved to reduced
/// accuracy, 200–299 infeasible, 300–399 unbounded, 400–499 limit, 500–599
/// failure. Shared by the QP/LP and SOCP report paths so the two cannot drift.
fn convex_status_report(s: pounce_convex::QpStatus) -> (&'static str, bool, i32) {
    use pounce_convex::QpStatus;
    match s {
        QpStatus::Optimal => ("Optimal Solution Found.", true, 0),
        QpStatus::OptimalInaccurate => {
            ("Solved to acceptable level (reduced accuracy).", true, 100)
        }
        QpStatus::PrimalInfeasible => ("Problem is primal infeasible.", false, 200),
        QpStatus::DualInfeasible => ("Problem is unbounded (dual infeasible).", false, 300),
        QpStatus::IterationLimit => ("Maximum iterations exceeded.", false, 400),
        QpStatus::NumericalFailure => ("Numerical failure in KKT factorization.", false, 500),
    }
}

/// Build the convex IPM [`pounce_convex::QpOptions`] from the registered CLI
/// knobs.
///
/// Every field is overridden only when the user *explicitly* set the option
/// (the `true` flag returned by the `OptionsList` accessors); otherwise the
/// `QpOptions` default is kept. The standard `tol` / `max_iter` options feed
/// the convex solve alongside the `qp_*` knobs registered in `main`.
/// `max_iter` is forwarded only when set so the convex driver's own (smaller)
/// cap is never silently raised to the much larger Ipopt default.
fn convex_cli_opts(app: &IpoptApplication) -> pounce_convex::QpOptions {
    let mut o = pounce_convex::QpOptions::default();
    let opt = app.options();
    if let Ok((v, true)) = opt.get_integer_value("max_iter", "") {
        // Forward `max_iter=0` too: AMPL/Ipopt semantics make it a
        // "take no iterations" request that must not reach optimality
        // (pounce#186). Only a negative value (invalid) is ignored so the
        // usize cast can't wrap.
        if v >= 0 {
            o.max_iter = v as usize;
        }
    }
    if let Ok((v, true)) = opt.get_numeric_value("tol", "") {
        o.tol = v;
    }
    if let Ok((v, true)) = opt.get_numeric_value("qp_tau", "") {
        o.tau = v;
    }
    if let Ok((v, true)) = opt.get_numeric_value("qp_reg", "") {
        o.reg = v;
    }
    if let Ok((v, true)) = opt.get_numeric_value("qp_infeas_tol", "") {
        o.infeas_tol = v;
    }
    if let Ok((v, true)) = opt.get_string_value("qp_hsde", "") {
        o.use_hsde = v != "no";
    }
    if let Ok((v, true)) = opt.get_string_value("qp_equilibrate", "") {
        o.equilibrate = v != "no";
    }
    if let Ok((v, true)) = opt.get_string_value("qp_crossover", "") {
        o.crossover = v != "no";
    }
    o
}

/// Resolve the convex LP/QP presolve switch (#139).
///
/// The convex driver is gated by the `qp_presolve` option, but `presolve` is
/// the spelling users carry over from the NLP path; on the convex path it used
/// to be silently ignored. Honor whichever the user *explicitly* set, with the
/// more specific `qp_presolve` winning when both are given; when neither is set
/// keep the driver's default (on).
///
/// Each argument is `Some((value, explicitly_set))` as returned by
/// `OptionsList::get_string_value(..).ok()`, or `None` if the lookup failed.
fn resolve_convex_presolve(
    qp_presolve: Option<(String, bool)>,
    presolve: Option<(String, bool)>,
) -> bool {
    match (qp_presolve, presolve) {
        // `qp_presolve` explicitly set → authoritative.
        (Some((v, true)), _) => v != "no",
        // else alias an explicitly-set `presolve` onto this path.
        (_, Some((v, true))) => v != "no",
        // neither set → keep the driver's default (on).
        _ => true,
    }
}

fn run_convex_qp(
    prob: &nl_reader::NlProblem,
    class: pounce_cli::dispatch::ProblemClass,
    sol_path: Option<&std::path::Path>,
    presolve_on: bool,
    json_cfg: Option<(&std::path::Path, ReportDetail, InputDescriptor)>,
    debug_hook: Option<&Rc<RefCell<pounce_cli::debug_repl::SolverDebugger>>>,
    ampl: bool,
    convex_opts: pounce_convex::QpOptions,
) -> ExitCode {
    use pounce_convex::presolve::{presolve, PresolveOutcome};
    use pounce_convex::{solve_qp_ipm, solve_qp_ipm_debug, QpOptions, QpStatus};

    let (qp, con_map, obj_nl_const) = match pounce_cli::qp_extract::extract_qp_with_map(prob) {
        Some(q) => q,
        None => {
            eprintln!(
                "pounce: internal error: {} not extractable as QP",
                class.name()
            );
            return ExitCode::from(2);
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
    let t0 = std::time::Instant::now();
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
        ..convex_opts
    };
    let sol = if qp_opts.max_iter == 0 {
        // AMPL/Ipopt semantics: `max_iter=0` takes no iterations and so
        // cannot reach optimality. Presolve can otherwise solve a trivial
        // problem (e.g. an unconstrained quadratic) directly — or the IPM's
        // reduced/empty solve can report Optimal — regardless of the cap, so
        // enforce the zero-iteration stop here before any solve runs
        // (pounce#186). Mirrors the NLP path's MaximumIterationsExceeded.
        trivial(QpStatus::IterationLimit)
    } else if let Some(hook) = debug_hook {
        // Interactive debug: step the IPM on the extracted QP directly.
        // Presolve is skipped so the debugger's `x`/`s`/`y`/`z` blocks
        // correspond to the user's problem rather than a reduced one.
        let mut h = hook.borrow_mut();
        solve_qp_ipm_debug(&qp, &qp_opts, &mut *h, backend)
    } else if presolve_on {
        match presolve(&qp) {
            PresolveOutcome::Reduced(ps) => {
                let st = ps.stats();
                if st.reduced_anything() {
                    println!(
                        "Presolve: {} → {} vars, {} → {} rows (fixed {}, \
                         free-fixed {}, substituted {}, forcing {}, dominated {}, tightened {})",
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
                    );
                }
                let red = solve_qp_ipm(&ps.reduced, &qp_opts, backend);
                ps.postsolve(&red)
            }
            PresolveOutcome::Infeasible => trivial(QpStatus::PrimalInfeasible),
            PresolveOutcome::Unbounded => trivial(QpStatus::DualInfeasible),
        }
    } else {
        solve_qp_ipm(&qp, &qp_opts, backend)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    // Report the objective in the user's original sense, including the
    // dropped constant term: f_user = sign * (½xᵀPx + cᵀx) + const.
    let reported_obj = sign * sol.obj + obj_const;

    let (msg, ok, srn) = convex_status_report(sol.status);
    println!(
        "POUNCE ({} IPM, pounce-convex): {msg}  obj={reported_obj:.8}  iters={}  ({elapsed:.3}s)",
        class.name(),
        sol.iters,
    );

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

    // Write a `.sol` if requested: primal x and recovered constraint duals in
    // the AMPL `.sol` convention.
    if let Some(path) = sol_path {
        let payload = nl_writer::SolutionFile {
            message: &format!("POUNCE {} IPM (pounce-convex): {msg}", class.name()),
            x: &sol.x,
            lambda: &lambda,
            solve_result_num: srn,
            suffixes: &[],
        };
        // Log a `.sol` write failure but do not early-return a distinct exit
        // code: the NLP path (main.rs:1091-1093) only logs, and under `-AMPL`
        // the final exit must still follow the solve-outcome contract.
        if let Err(e) = nl_writer::write_sol_file(path, &payload) {
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

    convex_exit_code(ok, ampl)
}

/// Solve a classified **convex QCQP** by reformulating it to a second-order
/// cone program and running the conic IPM (`pounce-convex`). Mirrors
/// [`run_convex_qp`]: same objective-constant fold-back, `.sol`/JSON output,
/// and per-constraint dual recovery, but the constraints carry quadratic rows
/// that become SOC blocks (see `qp_extract::extract_socp_with_map`). Presolve
/// is skipped — it is the QP-path's nonnegative-orthant reducer and is not
/// cone-aware.
fn run_convex_socp(
    prob: &nl_reader::NlProblem,
    class: pounce_cli::dispatch::ProblemClass,
    sol_path: Option<&std::path::Path>,
    json_cfg: Option<(&std::path::Path, ReportDetail, InputDescriptor)>,
    debug_hook: Option<&Rc<RefCell<pounce_cli::debug_repl::SolverDebugger>>>,
    ampl: bool,
    convex_opts: pounce_convex::QpOptions,
) -> ExitCode {
    use pounce_convex::{solve_socp_ipm, solve_socp_ipm_debug, QpOptions};

    let (qp, con_map, obj_nl_const, cones) =
        match pounce_cli::qp_extract::extract_socp_with_map(prob) {
            Some(q) => q,
            None => {
                eprintln!(
                    "pounce: internal error: {} not extractable as SOCP",
                    class.name()
                );
                return ExitCode::from(2);
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
        ..convex_opts
    };
    let t0 = std::time::Instant::now();
    let sol = if qp_opts.max_iter == 0 {
        // `max_iter=0` cannot reach optimality — stop before any solve, the
        // same zero-iteration contract the QP path enforces (pounce#186).
        pounce_convex::QpSolution {
            status: pounce_convex::QpStatus::IterationLimit,
            x: vec![0.0; qp.n],
            y: vec![0.0; qp.m_eq()],
            z: vec![0.0; qp.m_ineq()],
            z_lb: vec![0.0; qp.n],
            z_ub: vec![0.0; qp.n],
            obj: 0.0,
            iters: 0,
            iterates: Vec::new(),
        }
    } else if let Some(hook) = debug_hook {
        let mut h = hook.borrow_mut();
        solve_socp_ipm_debug(&qp, &cones, &qp_opts, &mut *h, backend)
    } else {
        solve_socp_ipm(&qp, &cones, &qp_opts, backend)
    };
    let elapsed = t0.elapsed().as_secs_f64();

    let reported_obj = sign * sol.obj + obj_const;

    let (msg, ok, srn) = convex_status_report(sol.status);
    println!(
        "POUNCE ({} conic IPM, pounce-convex): {msg}  obj={reported_obj:.8}  iters={}  ({elapsed:.3}s)",
        class.name(),
        sol.iters,
    );

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

    // Per-constraint duals, mapped from the cone multipliers back to `.nl`
    // constraint order (best-effort for the quadratic rows; see
    // `recover_socp_duals`).
    let lambda = pounce_cli::qp_extract::recover_socp_duals(prob, &con_map, &sol.y, &sol.z);

    if let Some(path) = sol_path {
        let payload = nl_writer::SolutionFile {
            message: &format!("POUNCE {} conic IPM (pounce-convex): {msg}", class.name()),
            x: &sol.x,
            lambda: &lambda,
            solve_result_num: srn,
            suffixes: &[],
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

    convex_exit_code(ok, ampl)
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
    /// `solve_result_num` 100 (the "solved to acceptable/reduced accuracy"
    /// band) with a distinct message, and onto the NLP-side
    /// `SolvedToAcceptableLevel` status, so callers reading either the `.sol`
    /// terminal line or the JSON report can tell it apart from a full-accuracy
    /// solve.
    #[test]
    fn optimal_inaccurate_is_distinct_from_optimal() {
        let (msg, ok, srn) = convex_status_report(QpStatus::OptimalInaccurate);
        assert_eq!(srn, 100, "reduced-accuracy solve must use the 100 band");
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
}

#[cfg(test)]
mod scaling_retry_tests {
    use super::{resolve_scaling_retry_outcome, scaling_retry_promoted};
    use pounce_nlp::return_codes::ApplicationReturnStatus;
    use pounce_nlp::SolveStatistics;

    fn stats_with_iters(n: i32) -> SolveStatistics {
        SolveStatistics {
            iteration_count: n,
            final_objective: n as f64,
            ..SolveStatistics::default()
        }
    }

    /// Code review L23: when the MC64 hypersensitivity re-solve does **not**
    /// recover, the verdict reverts to the original local-infeasibility status
    /// — and the reported statistics must revert with it, not leak the failed
    /// retry's iteration count / objective.
    #[test]
    fn failed_retry_keeps_original_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::InfeasibleProblemDetected,
            ApplicationReturnStatus::MaximumIterationsExceeded,
            ApplicationReturnStatus::RestorationFailed,
        ] {
            assert!(!scaling_retry_promoted(retry_status));
            let (status, stats) =
                resolve_scaling_retry_outcome(retry_status, original.clone(), retry.clone());
            assert_eq!(
                status,
                ApplicationReturnStatus::InfeasibleProblemDetected,
                "a non-promoting retry ({retry_status:?}) keeps the original verdict"
            );
            assert_eq!(
                stats.iteration_count, 7,
                "stats must stay the original solve's, not the failed retry's"
            );
            assert_eq!(stats.final_objective, 7.0);
        }
    }

    /// On promotion the retry is authoritative: its status AND its statistics
    /// are reported together.
    #[test]
    fn promoted_retry_adopts_retry_status_and_stats() {
        let original = stats_with_iters(7);
        let retry = stats_with_iters(42);
        for retry_status in [
            ApplicationReturnStatus::SolveSucceeded,
            ApplicationReturnStatus::SolvedToAcceptableLevel,
        ] {
            assert!(scaling_retry_promoted(retry_status));
            let (status, stats) =
                resolve_scaling_retry_outcome(retry_status, original.clone(), retry.clone());
            assert_eq!(status, retry_status, "a promoting retry adopts its verdict");
            assert_eq!(
                stats.iteration_count, 42,
                "promoted: stats must be the retry solve's"
            );
            assert_eq!(stats.final_objective, 42.0);
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

#[cfg(test)]
mod convex_presolve_tests {
    //! #139: the convex LP/QP driver is gated by `qp_presolve`, but `presolve`
    //! (the NLP-path spelling) used to be silently ignored on this path. These
    //! lock the aliasing: `presolve` is honored, `qp_presolve` wins ties, and
    //! only explicit settings count.
    use super::resolve_convex_presolve;

    // Helpers mirroring `OptionsList::get_string_value(..).ok()`:
    //   set(v)   → user explicitly set the option to `v`
    //   unset(v) → option carries its default `v`, not user-set
    fn set(v: &str) -> Option<(String, bool)> {
        Some((v.to_string(), true))
    }
    fn unset(v: &str) -> Option<(String, bool)> {
        Some((v.to_string(), false))
    }

    #[test]
    fn defaults_on_when_nothing_set() {
        assert!(resolve_convex_presolve(None, None));
        assert!(resolve_convex_presolve(unset("yes"), unset("yes")));
    }

    #[test]
    fn explicit_presolve_is_honored() {
        // The crux of #139: a bare `presolve no` must turn it off here.
        assert!(!resolve_convex_presolve(unset("yes"), set("no")));
        assert!(resolve_convex_presolve(unset("yes"), set("yes")));
    }

    #[test]
    fn explicit_qp_presolve_is_honored() {
        assert!(!resolve_convex_presolve(set("no"), None));
        assert!(resolve_convex_presolve(set("yes"), None));
    }

    #[test]
    fn qp_presolve_wins_when_both_explicit() {
        // More specific spelling is authoritative when the two conflict.
        assert!(!resolve_convex_presolve(set("no"), set("yes")));
        assert!(resolve_convex_presolve(set("yes"), set("no")));
    }

    #[test]
    fn explicit_presolve_overrides_unset_qp_presolve() {
        assert!(!resolve_convex_presolve(unset("yes"), set("no")));
    }
}
