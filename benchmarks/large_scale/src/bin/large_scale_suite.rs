//! Synthetic large-scale NLP smoke / benchmark suite.
//!
//! Sweeps the five hand-written problems in
//! [`pounce_large_scale::problems`] through the modern Pounce IPM and prints a
//! single status line per problem plus a one-line summary at the end. This is
//! the pounce-only sweep — solver-vs-solver comparisons against native Ipopt
//! live in the Mittelmann harness.
//!
//! Sizes can be overridden per problem via environment variables:
//! - `LARGE_SCALE_ROSENBROCK_N` (default 2000 — capped lower than the others
//!   because chained Rosenbrock is fundamentally O(n) Newton iterations
//!   regardless of solver, so n=50000 would blow past `max_iter`)
//! - `LARGE_SCALE_BRATU_N`      (default 10000)
//! - `LARGE_SCALE_OC_T`         (default 50000)
//! - `LARGE_SCALE_POISSON_K`    (default 200, ~80k vars)
//! - `LARGE_SCALE_SPARSE_QP_N`  (default 50000)
//!
//! A single `LARGE_SCALE_SIZE_SCALE=<float>` knob multiplies every default
//! size for quick larger/smaller sweeps.
//!
//! To check scaling behaviour, set `LARGE_SCALE_RAMP=<csv-of-scales>`
//! (default `0.1,0.5,1.0`) and the suite runs every problem at each scale,
//! then prints a per-problem scaling table (time vs. n).

use pounce_algorithm::alg_builder::{AlgorithmBuilder, LinearBackendFactory, LinearSolverChoice};
use pounce_algorithm::application::IpoptApplication;
use pounce_linsol::sparse_sym_iface::SparseSymLinearSolverInterface;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::TNLP;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use pounce_large_scale::problems::FinalState;
use pounce_large_scale::{
    BratuProblem, ChainedRosenbrock, OptimalControl, PoissonControl, SparseQP,
};

/// Default linear-solver factory: FERAL (pure-Rust). Mirrors the cutest
/// harness's behaviour when built without the `ma57` feature.
fn default_backend_factory() -> LinearBackendFactory {
    Box::new(
        |choice: LinearSolverChoice| -> Box<dyn SparseSymLinearSolverInterface> {
            match choice {
                LinearSolverChoice::Feral | LinearSolverChoice::Ma57 => {
                    Box::new(pounce_feral::FeralSolverInterface::new())
                }
            }
        },
    )
}

fn app_status_label(s: ApplicationReturnStatus) -> &'static str {
    use ApplicationReturnStatus::*;
    match s {
        SolveSucceeded => "Solve_Succeeded",
        SolvedToAcceptableLevel => "Solved_To_Acceptable_Level",
        InfeasibleProblemDetected => "Infeasible_Problem_Detected",
        SearchDirectionBecomesTooSmall => "Search_Direction_Becomes_Too_Small",
        DivergingIterates => "Diverging_Iterates",
        UserRequestedStop => "User_Requested_Stop",
        FeasiblePointFound => "Feasible_Point_Found",
        MaximumIterationsExceeded => "Maximum_Iterations_Exceeded",
        RestorationFailed => "Restoration_Failed",
        ErrorInStepComputation => "Error_In_Step_Computation",
        MaximumCpuTimeExceeded => "Maximum_CpuTime_Exceeded",
        MaximumWallTimeExceeded => "Maximum_WallTime_Exceeded",
        NotEnoughDegreesOfFreedom => "Not_Enough_Degrees_Of_Freedom",
        InvalidProblemDefinition => "Invalid_Problem_Definition",
        InvalidOption => "Invalid_Option",
        InvalidNumberDetected => "Invalid_Number_Detected",
        UnrecoverableException => "Unrecoverable_Exception",
        NonIpoptExceptionThrown => "NonIpopt_Exception_Thrown",
        InsufficientMemory => "Insufficient_Memory",
        InternalError => "Internal_Error",
    }
}

fn pounce_status_label(s: SolverReturn) -> &'static str {
    match s {
        SolverReturn::Success => "Solve_Succeeded",
        SolverReturn::MaxiterExceeded => "Maximum_Iterations_Exceeded",
        SolverReturn::CpuTimeExceeded => "Maximum_CpuTime_Exceeded",
        SolverReturn::WallTimeExceeded => "Maximum_WallTime_Exceeded",
        SolverReturn::StopAtTinyStep => "Search_Direction_Becomes_Too_Small",
        SolverReturn::StopAtAcceptablePoint => "Solved_To_Acceptable_Level",
        SolverReturn::LocalInfeasibility => "Infeasible_Problem_Detected",
        SolverReturn::UserRequestedStop => "User_Requested_Stop",
        SolverReturn::FeasiblePointFound => "Feasible_Point_Found",
        SolverReturn::DivergingIterates => "Diverging_Iterates",
        SolverReturn::RestorationFailure => "Restoration_Failed",
        SolverReturn::ErrorInStepComputation => "Error_In_Step_Computation",
        SolverReturn::InvalidNumberDetected => "Invalid_Number_Detected",
        SolverReturn::TooFewDegreesOfFreedom => "Not_Enough_Degrees_Of_Freedom",
        SolverReturn::InvalidOption => "Invalid_Option",
        SolverReturn::OutOfMemory => "Insufficient_Memory",
        SolverReturn::InternalError => "Internal_Error",
        SolverReturn::Unassigned => "Internal_Error",
    }
}

fn solved(label: &str) -> bool {
    label == "Solve_Succeeded" || label == "Solved_To_Acceptable_Level"
}

/// Per-problem result captured by `run`.
struct Outcome {
    name: String,
    n: usize,
    m: usize,
    status: String,
    obj: f64,
    iters: usize,
    secs: f64,
}

/// A `TNLP` plus a closure that retrieves its captured `FinalState`. Boxed so
/// the driver can hold a homogeneous list of problems.
struct ProblemEntry {
    name: String,
    n: usize,
    m: usize,
    /// Build the TNLP + a getter for its `FinalState`. The getter consumes the
    /// `Rc<RefCell<dyn TNLP>>` we cloned for `optimize_tnlp`.
    build: Box<dyn FnOnce() -> BuildResult>,
}

struct BuildResult {
    tnlp: Rc<RefCell<dyn TNLP>>,
    /// Pulls the captured `FinalState` out after the solve completes.
    take_final: Box<dyn FnOnce() -> FinalState>,
}

fn boxed_problem<T: TNLP + 'static>(
    name: &str,
    n: usize,
    m: usize,
    make: impl FnOnce() -> T + 'static,
    extract: fn(&mut T) -> FinalState,
) -> ProblemEntry {
    let name_owned = name.to_string();
    ProblemEntry {
        name: name_owned.clone(),
        n,
        m,
        build: Box::new(move || {
            let prob = make();
            let rc: Rc<RefCell<T>> = Rc::new(RefCell::new(prob));
            let rc_clone = rc.clone();
            BuildResult {
                tnlp: rc as Rc<RefCell<dyn TNLP>>,
                take_final: Box::new(move || extract(&mut rc_clone.borrow_mut())),
            }
        }),
    }
}

fn run_problem(entry: ProblemEntry) -> Outcome {
    let ProblemEntry {
        name, n, m, build, ..
    } = entry;
    let BuildResult { tnlp, take_final } = build();

    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_string_value("mu_strategy", "adaptive", true, false);
        let _ = opts.set_numeric_value("tol", 1e-8, true, false);
        let max_iter: i32 = std::env::var("LARGE_SCALE_MAX_ITER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3000);
        let _ = opts.set_integer_value("max_iter", max_iter, true, false);
        let pl: i32 = std::env::var("POUNCE_PRINT_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let _ = opts.set_integer_value("print_level", pl, true, false);
    }
    let bff: InnerBackendFactoryFactory = Box::new(default_backend_factory);
    let resto_factory = make_default_restoration_factory(
        RestoAlgorithmBuilder::new(),
        AlgorithmBuilder::new(),
        bff,
    );
    app.set_restoration_factory(resto_factory);

    if let Err(e) = app.initialize() {
        return Outcome {
            name,
            n,
            m,
            status: format!("Init_Error({:?})", e.kind),
            obj: f64::NAN,
            iters: 0,
            secs: 0.0,
        };
    }

    let t0 = Instant::now();
    let app_status = app.optimize_tnlp(tnlp);
    let secs = t0.elapsed().as_secs_f64();

    let stats = app.statistics();
    let iters = stats.iteration_count as usize;
    let final_state = take_final();
    let status = final_state
        .status
        .map(pounce_status_label)
        .unwrap_or_else(|| app_status_label(app_status))
        .to_string();
    let obj = if final_state.obj.is_finite() {
        final_state.obj
    } else {
        stats.final_objective
    };

    Outcome {
        name,
        n,
        m,
        status,
        obj,
        iters,
        secs,
    }
}

fn env_size(key: &str, default_n: usize, scale: f64) -> usize {
    if let Ok(v) = std::env::var(key) {
        if let Ok(parsed) = v.parse::<usize>() {
            return parsed.max(2);
        }
    }
    let scaled = (default_n as f64 * scale).round() as i64;
    scaled.max(2) as usize
}

fn build_entries(scale: f64) -> Vec<ProblemEntry> {
    let rosenbrock_n = env_size("LARGE_SCALE_ROSENBROCK_N", 2_000, scale);
    let bratu_n = env_size("LARGE_SCALE_BRATU_N", 10_000, scale);
    let oc_t = env_size("LARGE_SCALE_OC_T", 50_000, scale);
    let poisson_k = env_size("LARGE_SCALE_POISSON_K", 200, scale);
    let qp_n = env_size("LARGE_SCALE_SPARSE_QP_N", 50_000, scale);

    vec![
        boxed_problem(
            "ChainedRosenbrock",
            rosenbrock_n,
            0,
            move || ChainedRosenbrock::new(rosenbrock_n),
            |p| std::mem::take(&mut p.final_state),
        ),
        boxed_problem(
            "BratuProblem",
            bratu_n,
            bratu_n - 2,
            move || BratuProblem::new(bratu_n),
            |p| std::mem::take(&mut p.final_state),
        ),
        boxed_problem(
            "OptimalControl",
            2 * oc_t + 1,
            oc_t + 1,
            move || OptimalControl::new(oc_t),
            |p| std::mem::take(&mut p.final_state),
        ),
        boxed_problem(
            "PoissonControl",
            2 * poisson_k * poisson_k,
            poisson_k * poisson_k,
            move || PoissonControl::new(poisson_k),
            |p| std::mem::take(&mut p.final_state),
        ),
        boxed_problem(
            "SparseQP",
            qp_n,
            qp_n,
            move || SparseQP::new(qp_n),
            |p| std::mem::take(&mut p.final_state),
        ),
    ]
}

/// Parse `LARGE_SCALE_RAMP` (csv of f64 multipliers) → list of scales. When
/// unset and `LARGE_SCALE_SIZE_SCALE` is also unset, default to a short ramp
/// so a bare invocation shows scaling. If `LARGE_SCALE_SIZE_SCALE` is set, it
/// wins and forces a single-point sweep.
fn parse_ramp() -> Vec<f64> {
    if let Ok(raw) = std::env::var("LARGE_SCALE_RAMP") {
        let scales: Vec<f64> = raw
            .split(',')
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .filter(|s| *s > 0.0)
            .collect();
        if !scales.is_empty() {
            return scales;
        }
    }
    if let Ok(s) = std::env::var("LARGE_SCALE_SIZE_SCALE") {
        if let Ok(v) = s.parse::<f64>() {
            return vec![v];
        }
    }
    vec![0.1, 0.5, 1.0]
}

fn main() {
    let ramp = parse_ramp();

    eprintln!(
        "Large-scale synthetic suite: ramp = [{}]",
        ramp.iter()
            .map(|s| format!("{:.3}", s))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // outcomes[scale_idx] = per-problem outcomes at that scale (in entry order).
    let mut all_outcomes: Vec<Vec<Outcome>> = Vec::with_capacity(ramp.len());
    let mut problem_names: Vec<String> = Vec::new();

    for &scale in &ramp {
        let entries = build_entries(scale);
        if problem_names.is_empty() {
            problem_names = entries.iter().map(|e| e.name.clone()).collect();
        }
        eprintln!("---- scale = {:.3} ----", scale);
        let mut outcomes: Vec<Outcome> = Vec::with_capacity(entries.len());
        for entry in entries {
            let label = format!("{} (n={}, m={})", entry.name, entry.n, entry.m);
            eprint!("  {} ... ", label);
            let outcome = run_problem(entry);
            eprintln!(
                "{} iters={} obj={:.6e} ({:.1} ms)",
                outcome.status,
                outcome.iters,
                outcome.obj,
                outcome.secs * 1000.0,
            );
            outcomes.push(outcome);
        }
        all_outcomes.push(outcomes);
    }

    // Per-scale tables.
    for (i, scale) in ramp.iter().enumerate() {
        println!();
        println!("=== scale = {:.3} ===", scale);
        println!(
            "{:<22} {:>10} {:>8} {:>8} {:>14} {:>10}",
            "problem", "n", "m", "iters", "objective", "time(s)"
        );
        println!("{}", "-".repeat(80));
        for o in &all_outcomes[i] {
            println!(
                "{:<22} {:>10} {:>8} {:>8} {:>14.6e} {:>10.3}",
                o.name, o.n, o.m, o.iters, o.obj, o.secs
            );
            println!("    status: {}", o.status);
        }
    }

    // Scaling table: rows = problem, cols = scales. Cells show n / time(s);
    // a trailing column shows the empirical exponent fit via log-log slope of
    // time vs. n (only when ≥2 successful points exist).
    if ramp.len() >= 2 {
        println!();
        println!("=== scaling (time in seconds; n in parentheses) ===");
        let mut header = format!("{:<22}", "problem");
        for s in &ramp {
            header.push_str(&format!(" {:>18}", format!("scale={:.3}", s)));
        }
        header.push_str(&format!(" {:>10}", "slope"));
        println!("{}", header);
        println!("{}", "-".repeat(header.len()));
        for (pi, name) in problem_names.iter().enumerate() {
            let mut row = format!("{:<22}", name);
            let mut pts: Vec<(f64, f64)> = Vec::new();
            for outcomes in &all_outcomes {
                let o = &outcomes[pi];
                row.push_str(&format!(" {:>18}", format!("{}/{:.3}", o.n, o.secs)));
                if solved(&o.status) && o.secs > 0.0 && o.n > 0 {
                    pts.push(((o.n as f64).ln(), o.secs.ln()));
                }
            }
            let slope = if pts.len() >= 2 {
                let n = pts.len() as f64;
                let sx: f64 = pts.iter().map(|p| p.0).sum();
                let sy: f64 = pts.iter().map(|p| p.1).sum();
                let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
                let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
                let denom = n * sxx - sx * sx;
                if denom.abs() > 0.0 {
                    Some((n * sxy - sx * sy) / denom)
                } else {
                    None
                }
            } else {
                None
            };
            row.push_str(&match slope {
                Some(s) => format!(" {:>10.2}", s),
                None => format!(" {:>10}", "n/a"),
            });
            println!("{}", row);
        }
    }

    let total_runs: usize = all_outcomes.iter().map(|o| o.len()).sum();
    let n_solved: usize = all_outcomes
        .iter()
        .flatten()
        .filter(|o| solved(&o.status))
        .count();
    println!();
    println!(
        "Large-scale: {}/{} solved across ramp",
        n_solved, total_runs
    );
}
