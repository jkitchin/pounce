//! gh#1147 S1-I12 — `IpoptWriteSolveReport` carries the `second_opinion`
//! block (gh#850) after a solve the second-opinion ladder ran on.
//!
//! `IpoptSolve` without pairs runs the same ladder as the CLI and Python
//! (`run_second_opinion_ladder`). It used to keep only the ladder's status
//! and statistics, so the C report of a solve a rung rescued read as a clean,
//! fast solve: `statistics.iteration_count` is the promoted rung's alone, and
//! nothing said the base solve failed. That is gh#850's defect, which the CLI
//! and Python had already fixed by writing the block when `ran()`.
//!
//! One test per branch the writer can take:
//!
//! 1. **Rescued** — `degenerate_start_hs008` (the model the CLI's
//!    `issue850_second_opinion_is_recorded.rs` uses) as callbacks from the
//!    origin: the base solve returns `Infeasible_Problem_Detected` after 9
//!    iterations, three rungs run (9, 7, 5) and the third,
//!    `start_point_perturbation=1e-2`, promotes. The block equals, as a JSON
//!    value, the one Python's `solve(report_path=, report_detail=)` wrote on
//!    the same callbacks and start, at both detail levels.
//! 2. **Every rung fails** — `x² + 1 = 0` (Python's `NoRealSolution`, with
//!    its exact Hessian): the block is present, has no `promoted_by`, and the
//!    shipped status is the base one.
//! 3. **Converged** — the ladder reads the status and returns: no block.
//! 4. **Pairs declared** — the driver replaces the ladder in all three
//!    frontends (the CLI skips it when the driver ran; Python returns
//!    `SecondOpinionOutcome::unchanged`), so no block even on a status that
//!    would open the ladder on the plain path (`infeasible_pair`'s
//!    `Infeasible_Problem_Detected` verdict). Python wrote none either.
//! 5. **No leak** — a rescued solve and then a clean re-solve on the same
//!    handle: the second report has no block.
//!
//! `IpoptSolveWarmStart` reaches the ladder only through `IpoptSolve`, and
//! the session path (`IpoptSolverSolve`) does not run it; neither needs its
//! own branch here.
//!
//! The block has no timing field (`SecondOpinionInfo` is rung labels,
//! statuses and iteration counts), so it is compared whole.
//! `common/second_opinion_python_reports.json` holds Python's values
//! (0.16.0 extension built from this branch; `"full"` and `"summary"`
//! blocks identical).

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/cc_models.rs"]
mod cc_models;
#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use pounce_nlp::ApplicationReturnStatus;
use serde_json::Value;
use std::ffi::{CString, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

const PYTHON: &str = include_str!("common/second_opinion_python_reports.json");

fn python(name: &str) -> Value {
    let all: Value = serde_json::from_str(PYTHON).expect("reference JSON");
    all.get(name)
        .unwrap_or_else(|| panic!("no Python reference {name}"))
        .clone()
}

const INF: Number = 2e19;
const SUCCEEDED: Index = ApplicationReturnStatus::SolveSucceeded as Index;
const INFEASIBLE: Index = ApplicationReturnStatus::InfeasibleProblemDetected as Index;

// ---------------------------------------------------------------------------
// HS008: x0² + x1² = 25, x0·x1 = 9, constant objective, free variables
// ---------------------------------------------------------------------------

unsafe extern "C" fn zero_f(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe { *obj = 0.0 };
    1
}

unsafe extern "C" fn zero_grad(
    n: Index,
    _x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe { std::slice::from_raw_parts_mut(grad, n as usize).fill(0.0) };
    1
}

unsafe extern "C" fn hs008_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    g: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, 2);
        let g = std::slice::from_raw_parts_mut(g, 2);
        g[0] = x[0] * x[0] + x[1] * x[1];
        g[1] = x[0] * x[1];
    }
    1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn hs008_jac(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, 4).copy_from_slice(&[0, 0, 1, 1]);
            std::slice::from_raw_parts_mut(j_col, 4).copy_from_slice(&[0, 1, 0, 1]);
        } else {
            let x = std::slice::from_raw_parts(x, 2);
            std::slice::from_raw_parts_mut(values, 4).copy_from_slice(&[
                2.0 * x[0],
                2.0 * x[1],
                x[1],
                x[0],
            ]);
        }
    }
    1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn hs008_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            std::slice::from_raw_parts_mut(i_row, 3).copy_from_slice(&[0, 1, 1]);
            std::slice::from_raw_parts_mut(j_col, 3).copy_from_slice(&[0, 0, 1]);
        } else {
            let l = std::slice::from_raw_parts(lambda, 2);
            std::slice::from_raw_parts_mut(values, 3).copy_from_slice(&[
                2.0 * l[0],
                l[1],
                2.0 * l[0],
            ]);
        }
    }
    1
}

fn hs008() -> IpoptProblem {
    let x_l = [-INF; 2];
    let x_u = [INF; 2];
    let g = [25.0, 9.0];
    let p = unsafe {
        CreateIpoptProblem(
            2,
            x_l.as_ptr(),
            x_u.as_ptr(),
            2,
            g.as_ptr(),
            g.as_ptr(),
            4,
            3,
            0,
            Some(zero_f),
            Some(hs008_g),
            Some(zero_grad),
            Some(hs008_jac),
            Some(hs008_h),
        )
    };
    assert!(!p.is_null());
    p
}

// ---------------------------------------------------------------------------
// NoRealSolution: min x², x² + 1 = 0, x in [-10, 10]
// ---------------------------------------------------------------------------

unsafe extern "C" fn sq_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe { *obj = *x * *x };
    1
}

unsafe extern "C" fn sq_grad(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe { *grad = 2.0 * *x };
    1
}

unsafe extern "C" fn noreal_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    g: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe { *g = *x * *x + 1.0 };
    1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn noreal_jac(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            *i_row = 0;
            *j_col = 0;
        } else {
            *values = 2.0 * *x;
        }
    }
    1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn noreal_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            *i_row = 0;
            *j_col = 0;
        } else {
            *values = 2.0 * obj_factor + 2.0 * *lambda;
        }
    }
    1
}

fn no_real_solution() -> IpoptProblem {
    let (x_l, x_u, g) = ([-10.0], [10.0], [0.0]);
    let p = unsafe {
        CreateIpoptProblem(
            1,
            x_l.as_ptr(),
            x_u.as_ptr(),
            1,
            g.as_ptr(),
            g.as_ptr(),
            1,
            1,
            0,
            Some(sq_f),
            Some(noreal_g),
            Some(sq_grad),
            Some(noreal_jac),
            Some(noreal_h),
        )
    };
    assert!(!p.is_null());
    p
}

// ---------------------------------------------------------------------------
// Solving and reporting
// ---------------------------------------------------------------------------

/// Frees the handle on drop, so a failed assertion does not leak it.
struct Handle(IpoptProblem);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { FreeIpoptProblem(self.0) };
    }
}

/// Solve quietly from `x0`, as Python's reference did (`print_level 0`,
/// `sb yes`, no other option).
fn solve(p: IpoptProblem, x0: &[Number]) -> Index {
    let pl = CString::new("print_level").unwrap();
    let sb = CString::new("sb").unwrap();
    let yes = CString::new("yes").unwrap();
    unsafe {
        AddIpoptIntOption(p, pl.as_ptr(), 0);
        AddIpoptStrOption(p, sb.as_ptr(), yes.as_ptr());
    }
    let mut x = x0.to_vec();
    unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

fn report(p: IpoptProblem, detail: &str) -> Value {
    static N: AtomicUsize = AtomicUsize::new(0);
    let path: PathBuf = std::env::temp_dir().join(format!(
        "pounce-so-report-{}-{}-{detail}.json",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let cpath = CString::new(path.to_str().unwrap()).unwrap();
    let cdetail = CString::new(detail).unwrap();
    let ok = unsafe { IpoptWriteSolveReport(p, cpath.as_ptr(), cdetail.as_ptr()) };
    assert_eq!(ok, 1, "IpoptWriteSolveReport({detail}) failed");
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    serde_json::from_str(&text).unwrap()
}

fn count(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| panic!("not an integer: {v}"))
}

/// The block at both detail levels: present, identical, and consistent with
/// the report it sits in and with `GetIpoptIterCount`.
fn block_at_both_levels(p: IpoptProblem) -> (Value, Value) {
    let full = report(p, "full");
    let summary = report(p, "summary");
    let block = full
        .get("second_opinion")
        .unwrap_or_else(|| panic!("no second_opinion block in the C report"))
        .clone();
    assert_eq!(
        summary.get("second_opinion"),
        Some(&block),
        "the block is not detail-gated in the CLI and Python"
    );

    let tried = block["tried"].as_array().unwrap();
    let rungs = block["rung_iteration_counts"].as_array().unwrap();
    assert!(!tried.is_empty(), "a written block means a rung ran");
    assert_eq!(rungs.len(), tried.len());
    let base = count(&block["base_iteration_count"]);
    let total = count(&block["total_iteration_count"]);
    assert_eq!(total, base + rungs.iter().map(count).sum::<i64>());

    // The report's own iteration count is the shipped solve's: the promoted
    // rung's on a promotion, the base solve's otherwise. The block is what
    // says which.
    let reported = count(&full["statistics"]["iteration_count"]);
    assert_eq!(reported, i64::from(unsafe { GetIpoptIterCount(p) }));
    match block.get("promoted_by").and_then(Value::as_str) {
        Some(rung) => {
            let k = tried
                .iter()
                .position(|t| t.as_str() == Some(rung))
                .expect("promoted_by names a rung that ran");
            assert_eq!(reported, count(&rungs[k]));
            assert_ne!(
                block["base_status"].as_str(),
                full["solution"]["status_upstream"].as_str()
            );
        }
        None => {
            assert_eq!(reported, base);
            assert_eq!(
                block["base_status"].as_str(),
                full["solution"]["status_upstream"].as_str()
            );
        }
    }
    (full, block)
}

// ---------------------------------------------------------------------------
// (1) rescued
// ---------------------------------------------------------------------------

#[test]
fn a_rescued_solve_carries_pythons_block() {
    let h = Handle(hs008());
    let status = solve(h.0, &[0.0, 0.0]);
    assert_eq!(status, SUCCEEDED, "the promoted rung's verdict ships");
    let (full, block) = block_at_both_levels(h.0);

    let py = python("hs008_rescued");
    assert_eq!(
        block, py["second_opinion"],
        "C block differs from Python's on the same callbacks and start"
    );
    assert_eq!(full["solution"]["status"], py["status"]);
    assert_eq!(full["statistics"]["iteration_count"], py["iteration_count"]);

    // The premise, read off the block rather than assumed: the base solve
    // failed and a rung rescued it, and the reported count is not the cost.
    assert_eq!(block["base_status"], "Infeasible_Problem_Detected");
    assert_eq!(block["promoted_by"], "start_point_perturbation=1e-2");
    assert!(count(&block["total_iteration_count"]) > count(&full["statistics"]["iteration_count"]));
}

// ---------------------------------------------------------------------------
// (2) every rung fails
// ---------------------------------------------------------------------------

#[test]
fn a_ladder_that_rescues_nothing_still_carries_the_block() {
    let h = Handle(no_real_solution());
    let status = solve(h.0, &[0.5]);
    assert_eq!(status, INFEASIBLE, "the base verdict ships");
    let (_full, block) = block_at_both_levels(h.0);
    assert!(block.get("promoted_by").is_none(), "nothing was promoted");
    assert_eq!(block["base_status"], "Infeasible_Problem_Detected");

    // Python's rungs on this platform. Rung 4 opens only when the base
    // solve escalated the pivot quality, which is platform arithmetic
    // (`python/tests/test_second_opinion.py::_expected_rungs`); C and Python
    // ran the same solve, so they agree wherever the reference was taken,
    // and the three fixed rungs lead on every platform.
    let py = python("no_real_solution")["second_opinion"].clone();
    let fixed = [
        "feral_scaling=mc64",
        "mu_strategy=adaptive",
        "start_point_perturbation=1e-2",
    ];
    let tried: Vec<&str> = block["tried"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(&tried[..3], &fixed);
    let py_tried: Vec<&str> = py["tried"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    if tried == py_tried {
        assert_eq!(block, py, "C block differs from Python's");
    }
}

// ---------------------------------------------------------------------------
// (3) converged
// ---------------------------------------------------------------------------

#[test]
fn a_converged_solve_carries_no_block() {
    let h = Handle(hs008());
    assert_eq!(solve(h.0, &[3.0, 2.0]), SUCCEEDED);
    for detail in ["full", "summary"] {
        let r = report(h.0, detail);
        assert!(r.get("second_opinion").is_none(), "{detail}: {r}");
    }
    assert_eq!(
        python("hs008_clean_resolve")["has_second_opinion"],
        Value::Bool(false)
    );
}

// ---------------------------------------------------------------------------
// (4) pairs declared
// ---------------------------------------------------------------------------

#[test]
fn a_driver_solve_carries_no_second_opinion_block() {
    let m = cc_models::Model::infeasible_pair();
    let (status, _) = m.solve();
    // A verdict that opens the ladder on the path without pairs.
    assert_eq!(status, INFEASIBLE);
    let py = python("infeasible_pair_pairs");
    assert_eq!(py["status"], "InfeasibleProblemDetected");
    for detail in ["full", "summary"] {
        let r = report(m.prob, detail);
        assert!(
            r.get("complementarity").is_some(),
            "{detail}: the driver ran"
        );
        assert!(
            r.get("second_opinion").is_none(),
            "{detail}: the driver replaces the ladder, as in the CLI and Python"
        );
    }
    assert_eq!(py["has_second_opinion"], Value::Bool(false));
    assert_eq!(py["has_complementarity"], Value::Bool(true));
}

// ---------------------------------------------------------------------------
// (5) no leak across solves
// ---------------------------------------------------------------------------

#[test]
fn the_block_does_not_survive_a_clean_re_solve() {
    let h = Handle(hs008());
    let first = solve(h.0, &[0.0, 0.0]);
    assert_eq!(first, SUCCEEDED);
    block_at_both_levels(h.0);

    assert_eq!(solve(h.0, &[3.0, 2.0]), SUCCEEDED);
    let py = python("hs008_clean_resolve");
    for detail in ["full", "summary"] {
        let r = report(h.0, detail);
        assert!(
            r.get("second_opinion").is_none(),
            "{detail}: the previous solve's block leaked"
        );
        assert_eq!(r["statistics"]["iteration_count"], py["iteration_count"]);
    }
    assert_eq!(py["has_second_opinion"], Value::Bool(false));
}
