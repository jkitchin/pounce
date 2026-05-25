//! Reference-Ipopt runner for the `ChainedRosenbrock` TNLP, ramped at the
//! same scales as `large_scale_suite`. Use this side-by-side with the pounce
//! suite to compare iterations + wall time on identical problem instances.
//!
//! Usage:
//!   ./target/release/large_scale_ipopt
//!   LARGE_SCALE_RAMP=0.1,0.5,1.0 ./target/release/large_scale_ipopt

use pounce_large_scale::ChainedRosenbrock;
use pounce_nlp::tnlp::{BoundsInfo, SparsityRequest, StartingPoint, TNLP};
use std::ffi::{c_void, CString};
use std::time::Instant;

// ---- Native Ipopt FFI (lifted from benchmarks/cutest/src/bin/cutest_suite.rs) ----

type IpoptProblem = *mut c_void;
type EvalFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGradFCB = extern "C" fn(i32, *const f64, bool, *mut f64, *mut c_void) -> bool;
type EvalGCB = extern "C" fn(i32, *const f64, bool, i32, *mut f64, *mut c_void) -> bool;
type EvalJacGCB = extern "C" fn(
    i32,
    *const f64,
    bool,
    i32,
    i32,
    *mut i32,
    *mut i32,
    *mut f64,
    *mut c_void,
) -> bool;
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

struct Wrapper<'a> {
    tnlp: &'a mut dyn TNLP,
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
        let w = &mut *(user_data as *mut Wrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        match w.tnlp.eval_f(xs, new_x) {
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
        let w = &mut *(user_data as *mut Wrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        let gs = std::slice::from_raw_parts_mut(grad_f, n as usize);
        w.tnlp.eval_grad_f(xs, new_x, gs)
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
        let w = &mut *(user_data as *mut Wrapper);
        let xs = std::slice::from_raw_parts(x, n as usize);
        if m > 0 {
            let gs = std::slice::from_raw_parts_mut(g, m as usize);
            w.tnlp.eval_g(xs, new_x, gs)
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
        let w = &mut *(user_data as *mut Wrapper);
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, nele_jac as usize);
            let cols = std::slice::from_raw_parts_mut(j_col, nele_jac as usize);
            w.tnlp.eval_jac_g(
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
            w.tnlp
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
        let w = &mut *(user_data as *mut Wrapper);
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, nele_hess as usize);
            let cols = std::slice::from_raw_parts_mut(j_col, nele_hess as usize);
            w.tnlp.eval_h(
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
            w.tnlp.eval_h(
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
        let w = &mut *(user_data as *mut Wrapper);
        w.iterations = iter_count;
    }
    true
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

fn ipopt_status_label(status: i32) -> &'static str {
    match status {
        0 => "Solve_Succeeded",
        1 => "Solved_To_Acceptable_Level",
        -1 => "Maximum_Iterations_Exceeded",
        -2 => "Restoration_Failed",
        -3 => "Error_In_Step_Computation",
        -4 => "Maximum_CpuTime_Exceeded",
        -5 => "Maximum_WallTime_Exceeded",
        _ => "Other",
    }
}

fn solve_ipopt(tnlp: &mut dyn TNLP, max_iter: i32) -> (String, i32, f64, f64, f64) {
    let info = tnlp.get_nlp_info().expect("nlp_info");
    let n = info.n as usize;
    let m = info.m as usize;

    let mut x_l = vec![0.0f64; n];
    let mut x_u = vec![0.0f64; n];
    let mut g_l = vec![0.0f64; m.max(1)];
    let mut g_u = vec![0.0f64; m.max(1)];
    tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l[..m],
        g_u: &mut g_u[..m],
    });

    let mut x = vec![0.0f64; n];
    let mut z_l = vec![0.0f64; n];
    let mut z_u = vec![0.0f64; n];
    let mut lambda = vec![0.0f64; m.max(1)];
    tnlp.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lambda[..m],
    });

    let problem = unsafe {
        CreateIpoptProblem(
            info.n,
            x_l.as_mut_ptr(),
            x_u.as_mut_ptr(),
            info.m,
            g_l.as_mut_ptr(),
            g_u.as_mut_ptr(),
            info.nnz_jac_g,
            info.nnz_h_lag,
            0, // C-style indexing
            eval_f_cb,
            eval_g_cb,
            eval_grad_f_cb,
            eval_jac_g_cb,
            eval_h_cb,
        )
    };
    assert!(!problem.is_null(), "CreateIpoptProblem returned null");

    set_str(problem, "sb", "yes");
    set_str(problem, "mu_strategy", "adaptive");
    set_num(problem, "tol", 1e-8);
    set_int(problem, "max_iter", max_iter);
    set_int(problem, "print_level", 0);

    let mut wrapper = Wrapper {
        tnlp,
        iterations: 0,
    };
    unsafe {
        SetIntermediateCallback(problem, intermediate_cb);
    }

    let mut obj = 0.0f64;
    let mut g = vec![0.0f64; m.max(1)];
    let mut mult_x_l = vec![0.0f64; n];
    let mut mult_x_u = vec![0.0f64; n];

    let t0 = Instant::now();
    let status = unsafe {
        IpoptSolve(
            problem,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            lambda.as_mut_ptr(),
            mult_x_l.as_mut_ptr(),
            mult_x_u.as_mut_ptr(),
            &mut wrapper as *mut Wrapper as *mut c_void,
        )
    };
    let secs = t0.elapsed().as_secs_f64();

    unsafe {
        FreeIpoptProblem(problem);
    }

    // Final iter count: Ipopt's intermediate_cb fires for k=0..final-1, so
    // iterations at termination is iterations+1 if status indicates success.
    let iters = wrapper.iterations + 1;
    (
        ipopt_status_label(status).to_string(),
        iters,
        obj,
        secs,
        0.0,
    )
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
    let max_iter: i32 = std::env::var("LARGE_SCALE_MAX_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    eprintln!(
        "Reference Ipopt on ChainedRosenbrock: ramp = [{}], max_iter = {}",
        ramp.iter()
            .map(|s| format!("{:.3}", s))
            .collect::<Vec<_>>()
            .join(", "),
        max_iter,
    );

    println!(
        "{:<22} {:>8} {:>8} {:>14} {:>10} {:>10}",
        "problem", "n", "iters", "objective", "time(s)", "iters/n"
    );
    println!("{}", "-".repeat(80));

    let mut pts: Vec<(f64, f64)> = Vec::new();

    for &scale in &ramp {
        let n = env_size("LARGE_SCALE_ROSENBROCK_N", 2_000, scale);
        let mut prob = ChainedRosenbrock::new(n);
        let (status, iters, obj, secs, _) = solve_ipopt(&mut prob, max_iter);
        let iters_per_n = iters as f64 / n as f64;
        println!(
            "{:<22} {:>8} {:>8} {:>14.6e} {:>10.3} {:>10.3}",
            "ChainedRosenbrock", n, iters, obj, secs, iters_per_n,
        );
        println!("    status: {}", status);
        if status == "Solve_Succeeded" && secs > 0.0 {
            pts.push(((n as f64).ln(), secs.ln()));
        }
    }

    if pts.len() >= 2 {
        let n = pts.len() as f64;
        let sx: f64 = pts.iter().map(|p| p.0).sum();
        let sy: f64 = pts.iter().map(|p| p.1).sum();
        let sxx: f64 = pts.iter().map(|p| p.0 * p.0).sum();
        let sxy: f64 = pts.iter().map(|p| p.0 * p.1).sum();
        let denom = n * sxx - sx * sx;
        if denom.abs() > 0.0 {
            let slope = (n * sxy - sx * sy) / denom;
            println!();
            println!("log-log slope of time vs n: {:.2}", slope);
        }
    }
}
