//! `pounce --minima`: the CLI face of the multistart / find-minima search.
//!
//! The search itself is [`pounce_minima::find_minima`] (gh#966). This module
//! runs it on the CLI's application through [`IpoptLocalSolve`] and turns the
//! result into the console table, the ranked `.sol` files and the JSON
//! report.

// Moved to `pounce-minima` (gh#966); re-exported so `minima::archive` etc.
// keep resolving.
pub use pounce_minima::{archive, penalty_tnlp, sampling};

use crate::cli::{Args, MinimaArgs, ProblemSource};
use crate::solve_report::{InputDescriptor, ReportBuilder, status_to_solve_result_num};
use archive::scaled_distance;
use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_minima::{IpoptLocalSolve, Minimum, Stop, find_minima};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;

/// Entry point: run the `--minima` search on `base` (the raw problem TNLP —
/// presolve / counting wrappers are intentionally bypassed so coordinates
/// match the original problem and the clean objective is evaluated directly).
/// Returns the process exit code.
pub fn run(
    app: &mut IpoptApplication,
    base: &Rc<RefCell<dyn TNLP>>,
    cfg: &MinimaArgs,
    args: &Args,
    sol_path: Option<&Path>,
    dual_sign: Number,
) -> ExitCode {
    let Some(info) = base.borrow_mut().get_nlp_info() else {
        eprintln!("pounce: --minima could not read problem dimensions");
        return ExitCode::from(2);
    };
    let n = info.n as usize;
    let m = info.m as usize;

    println!(
        "Searching for up to {} minima via `{}` (max {} solves, seed {})...",
        cfg.n_minima,
        cfg.method.as_str(),
        cfg.max_solves_or_default(),
        cfg.seed
    );

    let result = {
        let mut solver = IpoptLocalSolve::new(app);
        match find_minima(base, &mut solver, cfg) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("pounce: --minima {e}");
                return ExitCode::from(2);
            }
        }
    };
    for w in &result.warnings {
        eprintln!("pounce: --minima {w}");
    }
    // An empty result caused by the filter is never silent (gh#983 review).
    if result.minima.is_empty() && result.n_kkt_rejected > 0 {
        eprintln!(
            "pounce: --minima: every converged candidate ({}) was rejected by \
             --kkt-tol {} (unscaled dual infeasibility above kkt_tol * max(1, gradient \
             scale)); no minima are returned. Loosen --kkt-tol, or pass --kkt-tol none.",
            result.n_kkt_rejected,
            cfg.kkt_tol.unwrap_or(f64::NAN)
        );
    }

    let minima = &result.minima;
    print_table(
        minima,
        &result.scale,
        result.stop,
        result.n_solves,
        result.n_kkt_rejected,
        dual_sign,
    );

    // Write the per-minimum `.sol` files: best → <stub>.sol, the rest →
    // ranked siblings <stub>.minNNN.sol.
    if let Some(sp) = sol_path {
        write_sol_files(sp, minima, m, dual_sign);
    }

    // JSON report: the standard single-solve report for the best minimum,
    // plus a backward-compatible `minima` section listing all of them.
    if let Some(json_path) = &args.json_output {
        write_json_report(
            json_path,
            args,
            cfg,
            MinimaCounts {
                stop: result.stop,
                n_solves: result.n_solves,
                n_kkt_rejected: result.n_kkt_rejected,
            },
            minima,
            n,
            m,
            &info,
            dual_sign,
        );
    }

    if minima.is_empty() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Print a ranked console table of the distinct optima found.
///
/// `obj_sign` is `-1.0` for a `maximize` `.nl` and `+1.0` otherwise, and it is
/// applied to the printed objective for the same reason the `.sol` duals carry
/// it (gh#959): the search runs on the internal `min -f`, so the number the
/// archive holds is the negated one. The RANKING is unaffected — rank 0 is the
/// best point either way — but for a maximize model the column now reads as
/// the maxima it actually found, in the model's own units, matching what a
/// plain solve of the same file reports.
fn print_table(
    minima: &[Minimum],
    l_scale: &[Number],
    stop: Stop,
    n_solves: usize,
    n_kkt_rejected: usize,
    obj_sign: Number,
) {
    println!();
    println!(
        "find-minima: {} distinct minim{} in {} solves ({})",
        minima.len(),
        if minima.len() == 1 { "um" } else { "a" },
        n_solves,
        stop.as_str()
    );
    if n_kkt_rejected > 0 {
        println!("  {n_kkt_rejected} converged solve(s) rejected by --kkt-tol");
    }
    if minima.is_empty() {
        println!("  (no accepted minima — try raising --max-solves or --patience)");
        return;
    }
    println!("  rank        objective     dist-to-best");
    let best = &minima[0].x;
    for (rank, mn) in minima.iter().enumerate() {
        let d = scaled_distance(&mn.x, best, l_scale);
        println!(
            "  {rank:>4}   {:>16.8e}   {:>14.6e}",
            obj_sign * mn.objective,
            d
        );
    }
}

/// Write `.sol` files: best to `sol_path`, ranked siblings alongside.
///
/// `dual_sign` is `-1.0` for a `maximize` `.nl` and `+1.0` otherwise. The
/// captured `lambda` is a multiplier of the problem the engine solved, which
/// for a maximize model is the negated one, so the sense has to be carried
/// back out here exactly as the single-solve path does (gh#959). Without it a
/// `--minima` run on a maximize model writes marginals opposite in sign to the
/// ones a plain solve of the same file writes. The same factor is applied to
/// the reported objective, in `print_table` and `write_json_report`.
fn write_sol_files(sol_path: &Path, minima: &[Minimum], m: usize, dual_sign: Number) {
    let zeros = vec![0.0; m];
    for (rank, mn) in minima.iter().enumerate() {
        let path = if rank == 0 {
            sol_path.to_path_buf()
        } else {
            sibling_sol_path(sol_path, rank)
        };
        let message = format!(
            "POUNCE {} find-minima rank {rank}: Solve_Succeeded",
            env!("CARGO_PKG_VERSION")
        );
        // Real base-problem duals recovered per minimum (issue #196, related);
        // `recover_duals` guarantees length `m`, but guard defensively.
        let lambda: Vec<Number> = if mn.lambda.len() == m {
            mn.lambda.iter().map(|&v| dual_sign * v).collect()
        } else {
            zeros.clone()
        };
        let payload = crate::nl_writer::SolutionFile {
            message: &message,
            x: &mn.x,
            mult_g: &lambda,
            solve_result_num: status_to_solve_result_num(ApplicationReturnStatus::SolveSucceeded),
            suffixes: &[],
        };
        match crate::nl_writer::write_sol_file(&path, &payload) {
            Ok(_) => eprintln!("pounce: wrote {}", path.display()),
            Err(e) => eprintln!("pounce: failed to write {}: {e}", path.display()),
        }
    }
}

/// `<stub>.sol` → `<stub>.minNNN.sol` for rank ≥ 1.
fn sibling_sol_path(sol_path: &Path, rank: usize) -> PathBuf {
    let mut stub = sol_path.to_path_buf();
    stub.set_extension(""); // drop `.sol`
    let base = stub.to_string_lossy().into_owned();
    PathBuf::from(format!("{base}.min{rank:03}.sol"))
}

/// The search's stop reason and counters, as reported in the JSON `minima`
/// block.
struct MinimaCounts {
    stop: Stop,
    n_solves: usize,
    n_kkt_rejected: usize,
}

/// Build the JSON report (standard best-solution report + `minima` section)
/// and write it.
#[allow(clippy::too_many_arguments)]
fn write_json_report(
    json_path: &Path,
    args: &Args,
    cfg: &MinimaArgs,
    counts: MinimaCounts,
    minima: &[Minimum],
    n: usize,
    m: usize,
    info: &pounce_nlp::tnlp::NlpInfo,
    obj_sign: Number,
) {
    let input = match &args.problem {
        ProblemSource::Builtin(name) => InputDescriptor::Builtin { name: name.clone() },
        ProblemSource::NlFile(p) => InputDescriptor::NlFile {
            path: p.clone(),
            size_bytes: std::fs::metadata(p).ok().map(|md| md.len()),
        },
    };
    let mut builder = ReportBuilder::new(args.json_detail, input);
    builder.problem.n_variables = n as Index;
    builder.problem.n_constraints = m as Index;
    builder.problem.n_objectives = 1;
    builder.problem.nnz_jac_g = Some(info.nnz_jac_g);
    builder.problem.nnz_h_lag = Some(info.nnz_h_lag);
    if let Some(best) = minima.first() {
        builder.solution.status = ApplicationReturnStatus::SolveSucceeded;
        builder.solution.solve_result_num =
            status_to_solve_result_num(ApplicationReturnStatus::SolveSucceeded);
        // The model's own sense, as every other POUNCE report carries it
        // (gh#959). Without it `--minima` on a `maximize` model reports an
        // objective opposite in sign to a plain solve of the same file.
        builder.solution.objective = obj_sign * best.objective;
        builder.solution.x = best.x.clone();
        // Real base-problem duals for the best minimum (issue #196, related);
        // `recover_duals` guarantees length `m`, guard defensively.
        builder.solution.lambda = if best.lambda.len() == m {
            best.lambda.clone()
        } else {
            vec![0.0; m]
        };
    }
    let report = builder.finish();

    // Inject the `minima` section without a schema change: serialize, then
    // splice it into the top-level object.
    let mut value = match serde_json::to_value(&report) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("pounce: failed to serialize minima report: {e}");
            return;
        }
    };
    let minima_json: Vec<serde_json::Value> = minima
        .iter()
        .map(|mn| {
            serde_json::json!({
                "x": mn.x,
                "objective": obj_sign * mn.objective,
            })
        })
        .collect();
    let values: Vec<Number> = minima.iter().map(|mn| obj_sign * mn.objective).collect();
    if let serde_json::Value::Object(map) = &mut value {
        map.insert(
            "minima".to_string(),
            serde_json::json!({
                "method": cfg.method.as_str(),
                "status": counts.stop.as_str(),
                "n_solves": counts.n_solves,
                "n_kkt_rejected": counts.n_kkt_rejected,
                "n_minima": minima.len(),
                "minima": minima_json,
                "values": values,
            }),
        );
    }
    match serde_json::to_string_pretty(&value) {
        Ok(s) => match std::fs::write(json_path, s) {
            Ok(_) => eprintln!("pounce: wrote {}", json_path.display()),
            Err(e) => eprintln!(
                "pounce: failed to write JSON report to {}: {e}",
                json_path.display()
            ),
        },
        Err(e) => eprintln!("pounce: failed to render minima report: {e}"),
    }
}
