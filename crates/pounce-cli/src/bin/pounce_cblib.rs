//! `pounce_cblib` — solve a CBLIB Conic Benchmark Format (`.cbf`) instance
//! through POUNCE's convex conic driver and emit a `pounce.solve-report/v1`
//! JSON report (status / iterations / time / objective, and the
//! per-iteration trace at `--json-detail full`).
//!
//! ```text
//! pounce_cblib <file.cbf> [--json-output PATH] [--json-detail summary|full]
//!                         [--max-iter N]
//! ```
//!
//! Used by the `benchmarks/cblib` harness to record per-instance POUNCE
//! results alongside the `.nl`-driven suites. The exit code follows the AMPL
//! convention via [`status_to_solve_result_num`] (0 = solved).

use pounce_cli::cbf;
use pounce_cli::solve_report::{
    InputDescriptor, ReportBuilder, ReportDetail, status_to_solve_result_num, write_report_file,
};
use pounce_convex::{QpOptions, QpStatus, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::IterRecord;
use std::path::PathBuf;
use std::process::ExitCode;

fn qp_status_to_ars(s: QpStatus) -> ApplicationReturnStatus {
    match s {
        QpStatus::Optimal => ApplicationReturnStatus::SolveSucceeded,
        // Reduced-accuracy solve — Ipopt's "Solved To Acceptable Level".
        QpStatus::OptimalInaccurate => ApplicationReturnStatus::SolvedToAcceptableLevel,
        QpStatus::PrimalInfeasible => ApplicationReturnStatus::InfeasibleProblemDetected,
        QpStatus::DualInfeasible => ApplicationReturnStatus::DivergingIterates, // unbounded
        QpStatus::IterationLimit => ApplicationReturnStatus::MaximumIterationsExceeded,
        QpStatus::NumericalFailure => ApplicationReturnStatus::InternalError,
    }
}

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

struct Args {
    file: PathBuf,
    json_output: Option<PathBuf>,
    detail: ReportDetail,
    max_iter: usize,
    debug: Option<pounce_cli::cli::DebugMode>,
    debug_script: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut file = None;
    let mut json_output = None;
    let mut detail = ReportDetail::Summary;
    let mut max_iter = 500;
    let mut debug = None;
    let mut debug_script = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--debug" => debug = Some(pounce_cli::cli::DebugMode::Repl),
            "--debug-json" => debug = Some(pounce_cli::cli::DebugMode::Json),
            "--debug-script" => {
                debug_script = Some(PathBuf::from(
                    it.next().ok_or("--debug-script needs a PATH")?,
                ));
            }
            "--json-output" => {
                json_output = Some(PathBuf::from(
                    it.next().ok_or("--json-output needs a PATH")?,
                ));
            }
            "--json-detail" => {
                let d = it.next().ok_or("--json-detail needs a value")?;
                detail = ReportDetail::parse(&d)?;
            }
            "--max-iter" => {
                max_iter = it
                    .next()
                    .ok_or("--max-iter needs N")?
                    .parse()
                    .map_err(|_| "--max-iter expects an integer")?;
            }
            other if other.starts_with("--") => return Err(format!("unknown flag '{other}'")),
            other => {
                if file.is_some() {
                    return Err(format!("unexpected extra argument '{other}'"));
                }
                file = Some(PathBuf::from(other));
            }
        }
    }
    Ok(Args {
        file: file.ok_or("usage: pounce_cblib <file.cbf> [--json-output PATH] …")?,
        json_output,
        detail,
        max_iter,
        debug,
        debug_script,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("pounce_cblib: {e}");
            return ExitCode::from(2);
        }
    };

    let text = match std::fs::read_to_string(&args.file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("pounce_cblib: cannot read {}: {e}", args.file.display());
            return ExitCode::from(2);
        }
    };
    let model = match cbf::parse(&text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("pounce_cblib: parse {}: {e}", args.file.display());
            return ExitCode::from(2);
        }
    };
    let cp = match model.to_conic() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pounce_cblib: map {}: {e}", args.file.display());
            return ExitCode::from(2);
        }
    };

    let full = matches!(args.detail, ReportDetail::Full);
    let opts = QpOptions {
        max_iter: args.max_iter,
        collect_iterates: full,
        ..QpOptions::default()
    };
    let t0 = std::time::Instant::now();
    let sol = if let Some(mode) = args.debug {
        // Interactive debug of the conic solve (exp/power → non-symmetric
        // HSDE; orthant/SOC/PSD → direct symmetric IPM). A `--debug-script`
        // drives it non-interactively.
        use pounce_cli::debug_repl::SolverDebugger;
        let mut dbg = SolverDebugger::new(mode, None);
        if let Some(p) = &args.debug_script {
            dbg = dbg.with_script(p.to_string_lossy().into_owned());
        }
        pounce_convex::solve_socp_ipm_debug(&cp.prob, &cp.cones, &opts, &mut dbg, backend)
    } else {
        solve_socp_ipm(&cp.prob, &cp.cones, &opts, backend)
    };
    let elapsed = t0.elapsed().as_secs_f64();
    let obj = cp.cbf_objective(sol.obj, model.minimize);
    let status = qp_status_to_ars(sol.status);

    println!(
        "POUNCE (conic HSDE, pounce-convex): {:?}  obj={obj:.8}  iters={}  ({elapsed:.3}s)  [{}]",
        sol.status,
        sol.iters,
        args.file.display(),
    );

    if let Some(path) = &args.json_output {
        let size_bytes = std::fs::metadata(&args.file).ok().map(|m| m.len());
        let mut b = ReportBuilder::new(
            args.detail,
            InputDescriptor::CbfFile {
                path: args.file.clone(),
                size_bytes,
            },
        );
        b.problem.n_variables = cp.prob.n as _;
        b.problem.n_constraints = (cp.prob.m_eq() + cp.prob.m_ineq()) as _;
        b.problem.n_objectives = 1;
        b.problem.minimize = model.minimize;
        b.solution.status = status;
        b.solution.solve_result_num = status_to_solve_result_num(status);
        b.solution.objective = obj;
        b.solution.x = sol.x.clone();
        b.stats.iteration_count = sol.iters as _;
        b.stats.final_objective = obj;
        b.stats.total_wallclock_time_secs = elapsed;
        if full {
            b.iterations = sol
                .iterates
                .iter()
                .map(|it| IterRecord {
                    iter: it.iter as _,
                    objective: it.objective,
                    inf_pr: it.primal_infeasibility,
                    inf_du: it.dual_infeasibility,
                    mu: it.mu,
                    d_norm: 0.0,
                    regularization: 0.0,
                    alpha_dual: it.alpha_dual,
                    alpha_primal: it.alpha_primal,
                    alpha_primal_char: ' ',
                    ls_trials: 0,
                })
                .collect();
        }
        let report = b.finish();
        if let Err(e) = write_report_file(path, &report) {
            eprintln!("pounce_cblib: write {}: {e}", path.display());
            return ExitCode::from(2);
        }
    }

    // A reduced-accuracy solve (`OptimalInaccurate`) still yields a usable
    // solution, so it counts as success for the process exit code.
    if matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
