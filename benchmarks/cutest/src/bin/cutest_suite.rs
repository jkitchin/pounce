//! CUTEst comparison harness — runs each problem with both POUNCE and
//! native Ipopt (linked via the C ABI) and emits a JSON record per
//! solver. Each problem runs in its own subprocess so a crash in one
//! solver doesn't abort the entire batch.
//!
//! Usage:
//!     cargo run --release --bin cutest_suite -- ROSENBR HS71
//!     cargo run --release --bin cutest_suite          # uses problem_list.txt

use pounce_cutest::CutestProblem;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_algorithm::application::IpoptApplication;
use pounce_nlp::tnlp::TNLP;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_void;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

#[derive(Serialize, Deserialize)]
struct CutestResult {
    name: String,
    solver: String,
    n: usize,
    m: usize,
    status: String,
    objective: f64,
    constraint_violation: f64,
    iterations: usize,
    solve_time: f64,
}

// ---- Native Ipopt FFI -------------------------------------------------------

type IpoptProblem = *mut c_void;
type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
type EvalJacGCB =
    extern "C" fn(i32, *const f64, bool, i32, i32, *mut i32, *mut i32, *mut f64, *mut c_void)
        -> bool;
type EvalHCB = extern "C" fn(
    i32,
    *const f64,
    bool,
    f64,
    i32,
    *const f64,
    bool,
    i32,
    *mut i32,
    *mut i32,
    *mut f64,
    *mut c_void,
) -> bool;
type IntermediateCB =
    extern "C" fn(i32, i32, f64, f64, f64, f64, f64, f64, f64, f64, i32, *mut c_void) -> bool;

extern "C" {
    fn CreateIpoptProblem(
        n: i32,
        x_l: *mut f64,
        x_u: *mut f64,
        m: i32,
        g_l: *mut f64,
        g_u: *mut f64,
        nele_jac: i32,
        nele_hess: i32,
        index_style: i32,
        eval_f: EvalFCB,
        eval_g: EvalGCB,
        eval_grad_f: EvalGradFCB,
        eval_jac_g: EvalJacGCB,
        eval_h: EvalHCB,
    ) -> IpoptProblem;
    fn FreeIpoptProblem(problem: IpoptProblem);
    fn AddIpoptStrOption(problem: IpoptProblem, keyword: *const i8, val: *const i8) -> bool;
    fn AddIpoptNumOption(problem: IpoptProblem, keyword: *const i8, val: f64) -> bool;
    fn AddIpoptIntOption(problem: IpoptProblem, keyword: *const i8, val: i32) -> bool;
    fn SetIntermediateCallback(problem: IpoptProblem, cb: IntermediateCB) -> bool;
    fn IpoptSolve(
        problem: IpoptProblem,
        x: *mut f64,
        g: *mut f64,
        obj_val: *mut f64,
        mult_g: *mut f64,
        mult_x_l: *mut f64,
        mult_x_u: *mut f64,
        user_data: *mut c_void,
    ) -> i32;
}

struct IpoptWrapper<'a> {
    problem: &'a mut CutestProblem,
    iterations: i32,
}

extern "C" fn eval_f_cb(
    n: i32,
    x: *const f64,
    new_x: bool,
    obj_value: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let w = &mut *(user_data as *mut IpoptWrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        match w.problem.eval_f(xs, new_x) {
            Some(v) => {
                *obj_value = v;
                true
            }
            None => false,
        }
    }
}

extern "C" fn eval_grad_f_cb(
    n: i32,
    x: *const f64,
    new_x: bool,
    grad_f: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let w = &mut *(user_data as *mut IpoptWrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        let gs = std::slice::from_raw_parts_mut(grad_f, n as usize);
        w.problem.eval_grad_f(xs, new_x, gs)
    }
}

extern "C" fn eval_g_cb(
    n: i32,
    x: *const f64,
    new_x: bool,
    m: i32,
    g: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let w = &mut *(user_data as *mut IpoptWrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        if m > 0 {
            let gs = std::slice::from_raw_parts_mut(g, m as usize);
            w.problem.eval_g(xs, new_x, gs)
        } else {
            true
        }
    }
}

extern "C" fn eval_jac_g_cb(
    n: i32,
    x: *const f64,
    new_x: bool,
    _m: i32,
    nele_jac: i32,
    i_row: *mut i32,
    j_col: *mut i32,
    values: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        use pounce_nlp::tnlp::SparsityRequest;
        let w = &mut *(user_data as *mut IpoptWrapper);
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, nele_jac as usize);
            let cols = std::slice::from_raw_parts_mut(j_col, nele_jac as usize);
            w.problem.eval_jac_g(
                None,
                new_x,
                SparsityRequest::Structure {
                    irow: rows,
                    jcol: cols,
                },
            )
        } else {
            let xs = std::slice::from_raw_parts(x, n as usize);
            let vs = std::slice::from_raw_parts_mut(values, nele_jac as usize);
            w.problem
                .eval_jac_g(Some(xs), new_x, SparsityRequest::Values { values: vs })
        }
    }
}

extern "C" fn eval_h_cb(
    n: i32,
    x: *const f64,
    new_x: bool,
    obj_factor: f64,
    m: i32,
    lambda: *const f64,
    new_lambda: bool,
    nele_hess: i32,
    i_row: *mut i32,
    j_col: *mut i32,
    values: *mut f64,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        use pounce_nlp::tnlp::SparsityRequest;
        let w = &mut *(user_data as *mut IpoptWrapper);
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, nele_hess as usize);
            let cols = std::slice::from_raw_parts_mut(j_col, nele_hess as usize);
            w.problem.eval_h(
                None,
                new_x,
                obj_factor,
                None,
                new_lambda,
                SparsityRequest::Structure {
                    irow: rows,
                    jcol: cols,
                },
            )
        } else {
            let xs = std::slice::from_raw_parts(x, n as usize);
            let lam = if m > 0 {
                Some(std::slice::from_raw_parts(lambda, m as usize))
            } else {
                None
            };
            let vs = std::slice::from_raw_parts_mut(values, nele_hess as usize);
            w.problem.eval_h(
                Some(xs),
                new_x,
                obj_factor,
                lam,
                new_lambda,
                SparsityRequest::Values { values: vs },
            )
        }
    }
}

extern "C" fn intermediate_cb(
    _alg_mod: i32,
    iter_count: i32,
    _obj_value: f64,
    _inf_pr: f64,
    _inf_du: f64,
    _mu: f64,
    _d_norm: f64,
    _regularization_size: f64,
    _alpha_du: f64,
    _alpha_pr: f64,
    _ls_trials: i32,
    user_data: *mut c_void,
) -> bool {
    unsafe {
        let w = &mut *(user_data as *mut IpoptWrapper);
        w.iterations = iter_count;
        true
    }
}

fn set_str(p: IpoptProblem, k: &str, v: &str) {
    let kc = CString::new(k).unwrap();
    let vc = CString::new(v).unwrap();
    unsafe {
        AddIpoptStrOption(p, kc.as_ptr(), vc.as_ptr());
    }
}
fn set_num(p: IpoptProblem, k: &str, v: f64) {
    let kc = CString::new(k).unwrap();
    unsafe {
        AddIpoptNumOption(p, kc.as_ptr(), v);
    }
}
fn set_int(p: IpoptProblem, k: &str, v: i32) {
    let kc = CString::new(k).unwrap();
    unsafe {
        AddIpoptIntOption(p, kc.as_ptr(), v);
    }
}

fn ipopt_status_label(status: i32) -> String {
    match status {
        0 => "Solve_Succeeded".to_string(),
        1 => "Solved_To_Acceptable_Level".to_string(),
        2 => "Infeasible_Problem_Detected".to_string(),
        3 => "Search_Direction_Becomes_Too_Small".to_string(),
        4 => "Diverging_Iterates".to_string(),
        5 => "User_Requested_Stop".to_string(),
        6 => "Feasible_Point_Found".to_string(),
        -1 => "Maximum_Iterations_Exceeded".to_string(),
        -2 => "Restoration_Failed".to_string(),
        -3 => "Error_In_Step_Computation".to_string(),
        -4 => "Maximum_CpuTime_Exceeded".to_string(),
        -5 => "Maximum_WallTime_Exceeded".to_string(),
        -10 => "Not_Enough_Degrees_Of_Freedom".to_string(),
        -11 => "Invalid_Problem_Definition".to_string(),
        -12 => "Invalid_Option".to_string(),
        -13 => "Invalid_Number_Detected".to_string(),
        -100 => "Unrecoverable_Exception".to_string(),
        -101 => "NonIpopt_Exception_Thrown".to_string(),
        -102 => "Insufficient_Memory".to_string(),
        -199 => "Internal_Error".to_string(),
        other => format!("IpoptStatus({})", other),
    }
}

fn app_status_label(s: ApplicationReturnStatus) -> String {
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
    .to_string()
}

fn pounce_status_label(s: SolverReturn) -> String {
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
    .to_string()
}

fn solve_with_ipopt(problem: &mut CutestProblem) -> CutestResult {
    let n = problem.n;
    let m = problem.m;
    let mut x_l = problem.x_l.clone();
    let mut x_u = problem.x_u.clone();
    let mut g_l = if m > 0 { problem.c_l.clone() } else { vec![0.0] };
    let mut g_u = if m > 0 { problem.c_u.clone() } else { vec![0.0] };
    let nele_jac = problem.jac_rows.len() as i32;
    let nele_hess = problem.hess_rows.len() as i32;

    let mut x = problem.x0.clone();
    let mut g = vec![0.0; m.max(1)];
    let mut obj_val = 0.0;
    let mut mult_g = vec![0.0; m.max(1)];
    let mut mult_x_l = vec![0.0; n];
    let mut mult_x_u = vec![0.0; n];

    let mut wrapper = IpoptWrapper {
        problem,
        iterations: 0,
    };

    unsafe {
        let p = CreateIpoptProblem(
            n as i32,
            x_l.as_mut_ptr(),
            x_u.as_mut_ptr(),
            m as i32,
            g_l.as_mut_ptr(),
            g_u.as_mut_ptr(),
            nele_jac,
            nele_hess,
            0,
            eval_f_cb,
            eval_g_cb,
            eval_grad_f_cb,
            eval_jac_g_cb,
            eval_h_cb,
        );
        if p.is_null() {
            return CutestResult {
                name: wrapper.problem.name.clone(),
                solver: "ipopt".to_string(),
                n,
                m,
                status: "Internal_Error".to_string(),
                objective: f64::NAN,
                constraint_violation: f64::NAN,
                iterations: 0,
                solve_time: 0.0,
            };
        }
        set_str(p, "sb", "yes");
        set_str(p, "mu_strategy", "adaptive");
        set_num(p, "tol", 1e-8);
        set_int(p, "max_iter", 3000);
        set_int(p, "print_level", 0);
        SetIntermediateCallback(p, intermediate_cb);

        let user_data = &mut wrapper as *mut _ as *mut c_void;
        let t0 = Instant::now();
        let status = IpoptSolve(
            p,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj_val,
            mult_g.as_mut_ptr(),
            mult_x_l.as_mut_ptr(),
            mult_x_u.as_mut_ptr(),
            user_data,
        );
        let dt = t0.elapsed().as_secs_f64();
        let iters = wrapper.iterations as usize;
        FreeIpoptProblem(p);

        let cv = wrapper.problem.constraint_violation(&x);
        CutestResult {
            name: wrapper.problem.name.clone(),
            solver: "ipopt".to_string(),
            n,
            m,
            status: ipopt_status_label(status),
            objective: obj_val,
            constraint_violation: cv,
            iterations: iters,
            solve_time: dt,
        }
    }
}

fn solve_with_pounce(problem: CutestProblem) -> (CutestResult, CutestProblem) {
    let n = problem.n;
    let m = problem.m;
    let name = problem.name.clone();

    let problem_rc: Rc<RefCell<CutestProblem>> = Rc::new(RefCell::new(problem));

    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_string_value("mu_strategy", "adaptive", true, false);
        let _ = opts.set_numeric_value("tol", 1e-8, true, false);
        let _ = opts.set_integer_value("max_iter", 3000, true, false);
        let pl: i32 = std::env::var("POUNCE_PRINT_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let _ = opts.set_integer_value("print_level", pl, true, false);
    }
    if let Err(e) = app.initialize() {
        let p = Rc::try_unwrap(problem_rc).ok().unwrap().into_inner();
        return (
            CutestResult {
                name,
                solver: "pounce".to_string(),
                n,
                m,
                status: format!("Init_Error({:?})", e.kind),
                objective: f64::NAN,
                constraint_violation: f64::NAN,
                iterations: 0,
                solve_time: 0.0,
            },
            p,
        );
    }

    let t0 = Instant::now();
    let app_status = app.optimize_tnlp(problem_rc.clone() as Rc<RefCell<dyn TNLP>>);
    let dt = t0.elapsed().as_secs_f64();

    let stats = app.statistics();
    let iters = stats.iteration_count as usize;

    let p = Rc::try_unwrap(problem_rc).ok().unwrap().into_inner();
    let final_status = p
        .final_status
        .map(pounce_status_label)
        .unwrap_or_else(|| app_status_label(app_status));
    let final_obj = if p.final_obj.is_finite() {
        p.final_obj
    } else {
        stats.final_objective
    };
    let cv = if !p.final_x.is_empty() {
        p.constraint_violation(&p.final_x)
    } else {
        f64::NAN
    };

    let result = CutestResult {
        name,
        solver: "pounce".to_string(),
        n,
        m,
        status: final_status,
        objective: final_obj,
        constraint_violation: cv,
        iterations: iters,
        solve_time: dt,
    };
    (result, p)
}

fn run_single(name: &str, solver: &str) {
    let suite_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let problems_dir = suite_dir.join("problems");
    let lib_path =
        problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
    let outsdif_path = problems_dir.join(format!("{}_OUTSDIF.d", name));

    let problem = match CutestProblem::load(
        name,
        lib_path.to_str().unwrap(),
        outsdif_path.to_str().unwrap(),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  SKIP {} (load failed: {})", name, e);
            std::process::exit(1);
        }
    };

    match solver {
        "pounce" => {
            let (result, p) = solve_with_pounce(problem);
            println!("{}", serde_json::to_string(&result).unwrap());
            eprintln!(
                "pounce: {} (obj={:.6e}, {:.1}ms)",
                result.status,
                result.objective,
                result.solve_time * 1000.0,
            );
            p.cleanup();
        }
        "ipopt" => {
            let mut p = problem;
            let result = solve_with_ipopt(&mut p);
            println!("{}", serde_json::to_string(&result).unwrap());
            eprintln!(
                "ipopt: {} (obj={:.6e}, {:.1}ms)",
                result.status,
                result.objective,
                result.solve_time * 1000.0,
            );
            p.cleanup();
        }
        other => {
            eprintln!("Unknown solver: {}", other);
            std::process::exit(1);
        }
    }
}

fn problem_list(suite_dir: &Path) -> Vec<String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        return args;
    }
    let list = std::env::var("PROBLEM_LIST")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| suite_dir.join("problem_list.txt"));
    if !list.exists() {
        eprintln!(
            "No problems specified and {} not found.",
            list.display()
        );
        eprintln!("Usage: cutest_suite PROBLEM1 PROBLEM2 ...");
        std::process::exit(1);
    }
    std::fs::read_to_string(&list)
        .expect("read problem_list")
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 5 && args[1] == "--single" && args[3] == "--solver" {
        run_single(&args[2], &args[4]);
        return;
    }

    let suite_dir = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let problems_dir = suite_dir.join("problems");
    let names = problem_list(&suite_dir);

    let max_n: usize = std::env::var("CUTEST_MAX_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let timeout_secs: u64 = std::env::var("CUTEST_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let self_exe = std::env::current_exe().expect("current_exe");
    eprintln!(
        "CUTEst harness: {} problems, max_n={}, timeout={}s",
        names.len(),
        max_n,
        timeout_secs
    );

    let jsonl_path = std::env::var("RESULTS_JSONL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| suite_dir.join("results.jsonl"));
    let mut jsonl = std::fs::File::create(&jsonl_path).ok().map(std::io::BufWriter::new);
    eprintln!("Streaming results to {}", jsonl_path.display());

    let mut all: Vec<CutestResult> = Vec::new();
    for name in &names {
        let lib_path =
            problems_dir.join(format!("lib{}.{}", name, std::env::consts::DLL_EXTENSION));
        let outsdif = problems_dir.join(format!("{}_OUTSDIF.d", name));
        if !lib_path.exists() || !outsdif.exists() {
            eprintln!("  SKIP {} (not prepared)", name);
            continue;
        }

        // Pre-load briefly to read dimensions for the size filter
        let p = match CutestProblem::load(
            name,
            lib_path.to_str().unwrap(),
            outsdif.to_str().unwrap(),
        ) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  SKIP {} (load failed: {})", name, e);
                continue;
            }
        };
        let n = p.n;
        let m = p.m;
        p.cleanup();
        if n > max_n {
            eprintln!("  SKIP {} (n={} > max_n={})", name, n, max_n);
            continue;
        }

        eprint!("  {} (n={}, m={}) ... ", name, n, m);
        for solver in &["pounce", "ipopt"] {
            let out = std::process::Command::new("timeout")
                .arg(format!("{}s", timeout_secs))
                .arg(&self_exe)
                .arg("--single")
                .arg(name)
                .arg("--solver")
                .arg(solver)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            match out {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    if !o.status.success() && stdout.is_empty() {
                        let exit_code = o.status.code();
                        let (status, label) = if exit_code == Some(124) {
                            ("Timeout".to_string(), "TIMEOUT")
                        } else {
                            (format!("Crash({:?})", exit_code), "CRASH")
                        };
                        eprint!("{}: {} ", solver, label);
                        let r = CutestResult {
                            name: name.clone(),
                            solver: solver.to_string(),
                            n,
                            m,
                            status,
                            objective: f64::NAN,
                            constraint_violation: f64::NAN,
                            iterations: 0,
                            solve_time: timeout_secs as f64,
                        };
                        append_jsonl(&mut jsonl, &r);
                        all.push(r);
                        continue;
                    }
                    for line in stdout.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        if let Ok(r) = serde_json::from_str::<CutestResult>(line) {
                            append_jsonl(&mut jsonl, &r);
                            all.push(r);
                        }
                    }
                    for line in stderr.lines() {
                        let t = line.trim();
                        if t.starts_with("pounce:") || t.starts_with("ipopt:") {
                            eprint!("{} ", t);
                        }
                    }
                }
                Err(e) => eprint!("{}: SPAWN_ERROR({}) ", solver, e),
            }
        }
        eprintln!();
    }

    let results_path = std::env::var("RESULTS_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| suite_dir.join("results.json"));
    let json = serde_json::to_string_pretty(&all).unwrap();
    if let Err(e) = std::fs::write(&results_path, &json) {
        eprintln!("WARNING: failed to write {}: {}", results_path.display(), e);
    } else {
        eprintln!("Results written to {}", results_path.display());
    }

    let pounce_solved = all
        .iter()
        .filter(|r| {
            r.solver == "pounce"
                && (r.status == "Solve_Succeeded" || r.status == "Solved_To_Acceptable_Level")
        })
        .count();
    let ipopt_solved = all
        .iter()
        .filter(|r| {
            r.solver == "ipopt"
                && (r.status == "Solve_Succeeded" || r.status == "Solved_To_Acceptable_Level")
        })
        .count();
    let n_problems = all.iter().filter(|r| r.solver == "pounce").count();
    eprintln!("\nSummary: {} problems", n_problems);
    eprintln!("  pounce solved: {}/{}", pounce_solved, n_problems);
    eprintln!("  ipopt  solved: {}/{}", ipopt_solved, n_problems);
}

fn append_jsonl(
    writer: &mut Option<std::io::BufWriter<std::fs::File>>,
    r: &CutestResult,
) {
    use std::io::Write;
    if let Some(w) = writer.as_mut() {
        if let Ok(line) = serde_json::to_string(r) {
            let _ = writeln!(w, "{}", line);
            let _ = w.flush();
        }
    }
}
