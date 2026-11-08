//! Failure branches (and, where nothing else did, the success branch) of the
//! C entry points that had no test reaching them when the export-surface
//! guard (`export_surface.rs`) was introduced.
//!
//! Each `// error-path: NAME` line marks the test that exercises NAME's
//! failure branch; `export_surface.rs` requires one per export.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_cinterface::fortran::{
    ipaddintoption_, ipaddnumoption_, ipaddstroption_, ipcreate_, ipfree_, ipsetcallback_,
    ipsolve_, ipunsetcallback_,
};
use pounce_cinterface::solver::{
    IpoptCreateSolver, IpoptFreeSolver, IpoptSolver, IpoptSolverGetKktDim, IpoptSolverKktSolve,
    IpoptSolverKktSolveScaled, IpoptSolverParametricStep, IpoptSolverReducedHessian,
    IpoptSolverSolve,
};
use pounce_cinterface::*;
use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

const TRUE: Bool = 1;
const FALSE: Bool = 0;
const NULL_PROBLEM: IpoptProblem = std::ptr::null_mut();
const INTERNAL_ERROR: Index = -199;

// f(x) = (x - 2)^2, unconstrained.
unsafe extern "C" fn quad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        *obj = (*x - 2.0) * (*x - 2.0);
    }
    TRUE
}

unsafe extern "C" fn quad_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        *grad = 2.0 * (*x - 2.0);
    }
    TRUE
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn quad_h(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele: Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            *irow = 0;
            *jcol = 0;
        } else {
            *values = 2.0 * obj_factor;
        }
    }
    TRUE
}

fn create_quad() -> IpoptProblem {
    let xl = [-1.0e20];
    let xu = [1.0e20];
    let p = unsafe {
        CreateIpoptProblem(
            1,
            xl.as_ptr(),
            xu.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            1,
            0,
            Some(quad_f),
            None,
            Some(quad_grad_f),
            None,
            Some(quad_h),
        )
    };
    assert!(!p.is_null());
    set_str(p, "sb", "yes");
    set_int(p, "print_level", 0);
    p
}

fn set_int(p: IpoptProblem, k: &str, v: Index) {
    let k = CString::new(k).unwrap();
    assert_eq!(unsafe { AddIpoptIntOption(p, k.as_ptr(), v) }, TRUE);
}

fn set_str(p: IpoptProblem, k: &str, v: &str) {
    let k = CString::new(k).unwrap();
    let v = CString::new(v).unwrap();
    assert_eq!(
        unsafe { AddIpoptStrOption(p, k.as_ptr(), v.as_ptr()) },
        TRUE
    );
}

fn solve(p: IpoptProblem) -> Index {
    let mut x = [0.0];
    let mut obj = 0.0;
    unsafe {
        IpoptSolve(
            p,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    }
}

// ---------------------------------------------------------------------------
// Problem-handle entry points
// ---------------------------------------------------------------------------

#[test]
fn setters_refuse_a_null_problem() {
    unsafe {
        // error-path: SetIntermediateCallback
        assert_eq!(SetIntermediateCallback(NULL_PROBLEM, None), FALSE);
        // error-path: IpoptClearNonlinearVariables
        assert_eq!(IpoptClearNonlinearVariables(NULL_PROBLEM), FALSE);
        // error-path: IpoptClearWarmStartWorkingSet
        assert_eq!(IpoptClearWarmStartWorkingSet(NULL_PROBLEM), FALSE);
        // error-path: IpoptEnableIterHistory
        assert_eq!(IpoptEnableIterHistory(NULL_PROBLEM), FALSE);
    }
    // …and accept a live one, so the FALSE above is the NULL branch.
    let p = create_quad();
    unsafe {
        assert_eq!(SetIntermediateCallback(p, None), TRUE);
        assert_eq!(IpoptClearNonlinearVariables(p), TRUE);
        assert_eq!(IpoptClearWarmStartWorkingSet(p), TRUE);
        assert_eq!(IpoptEnableIterHistory(p), TRUE);
        FreeIpoptProblem(p);
    }
}

#[test]
fn solve_warm_start_on_a_null_problem_is_internal_error() {
    // error-path: IpoptSolveWarmStart
    let rc = unsafe {
        IpoptSolveWarmStart(
            NULL_PROBLEM,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, INTERNAL_ERROR);
}

#[test]
fn write_solve_report_refuses_each_bad_input() {
    let dir = std::env::temp_dir().join(format!("pounce-ep-report-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = CString::new(dir.join("r.json").to_str().unwrap()).unwrap();
    let full = CString::new("full").unwrap();
    let bogus = CString::new("everything").unwrap();

    // error-path: IpoptWriteSolveReport
    assert_eq!(
        unsafe { IpoptWriteSolveReport(NULL_PROBLEM, path.as_ptr(), std::ptr::null()) },
        FALSE,
        "NULL handle"
    );
    let p = create_quad();
    assert_eq!(
        unsafe { IpoptWriteSolveReport(p, path.as_ptr(), std::ptr::null()) },
        FALSE,
        "no prior solve"
    );
    assert_eq!(solve(p), 0);
    assert_eq!(
        unsafe { IpoptWriteSolveReport(p, std::ptr::null(), std::ptr::null()) },
        FALSE,
        "NULL path"
    );
    assert_eq!(
        unsafe { IpoptWriteSolveReport(p, path.as_ptr(), bogus.as_ptr()) },
        FALSE,
        "invalid detail"
    );
    assert_eq!(
        unsafe { IpoptWriteSolveReport(p, path.as_ptr(), full.as_ptr()) },
        TRUE,
        "the success branch, so the FALSEs above are the refusals"
    );
    unsafe { FreeIpoptProblem(p) };
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn restoration_stats_report_zero_without_data_and_after_a_clean_solve() {
    let mut out: (Index, Index, Index, Number) = (-7, -7, -7, -7.0);
    // error-path: GetPounceRestorationStats
    unsafe {
        GetPounceRestorationStats(NULL_PROBLEM, &mut out.0, &mut out.1, &mut out.2, &mut out.3);
    }
    assert_eq!(out, (0, 0, 0, 0.0), "NULL handle answers 'none'");
    // NULL outputs are skipped, not written.
    unsafe {
        GetPounceRestorationStats(
            NULL_PROBLEM,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
    }
    let p = create_quad();
    out = (-7, -7, -7, -7.0);
    unsafe { GetPounceRestorationStats(p, &mut out.0, &mut out.1, &mut out.2, &mut out.3) };
    assert_eq!(out, (0, 0, 0, 0.0), "no prior solve");
    assert_eq!(solve(p), 0);
    out = (-7, -7, -7, -7.0);
    unsafe { GetPounceRestorationStats(p, &mut out.0, &mut out.1, &mut out.2, &mut out.3) };
    // An unconstrained convex quadratic never needs restoration.
    assert_eq!((out.0, out.1, out.2), (0, 0, 0));
    assert!(out.3 >= 0.0);
    unsafe { FreeIpoptProblem(p) };
}

fn fd_stats(p: IpoptProblem) -> [Index; 7] {
    let mut s = [-7; 7];
    let [a, b, c, d, e, f, g] = &mut s;
    unsafe { GetPounceFdHessianStats(p, a, b, c, d, e, f, g) };
    s
}

#[test]
fn fd_hessian_stats_distinguish_did_not_run_from_ran() {
    // error-path: GetPounceFdHessianStats
    assert_eq!(
        fd_stats(NULL_PROBLEM),
        [-1, 0, 0, 0, 0, 0, 0],
        "NULL handle"
    );

    // Exact Hessian: the census is at its "did not run" values.
    let p = create_quad();
    assert_eq!(solve(p), 0);
    assert_eq!(
        fd_stats(p)[0],
        -1,
        "exact-Hessian solve reports pattern_used = -1"
    );
    unsafe { FreeIpoptProblem(p) };

    // Finite-difference Hessian: the census reports what ran.
    let p = create_quad();
    set_str(p, "hessian_approximation", "finite-difference");
    assert_eq!(solve(p), 0);
    let s = fd_stats(p);
    assert!(s[0] == 0 || s[0] == 1, "pattern_used = {}", s[0]);
    assert_eq!(s[2], 1, "n = 1 column differenced");
    unsafe { FreeIpoptProblem(p) };
}

// ---------------------------------------------------------------------------
// Session entry points
// ---------------------------------------------------------------------------

fn solved_session() -> IpoptSolver {
    let mut p = create_quad();
    let s = unsafe { IpoptCreateSolver(&mut p) };
    assert!(!s.is_null());
    let mut x = [0.0];
    let mut obj = 0.0;
    let rc = unsafe {
        IpoptSolverSolve(
            s,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, 0);
    s
}

#[test]
fn session_solve_on_a_null_solver_is_internal_error() {
    // error-path: IpoptSolverSolve
    let rc = unsafe {
        IpoptSolverSolve(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, INTERNAL_ERROR);
}

#[test]
fn kkt_solves_refuse_without_a_factor_and_agree_with_one() {
    let rhs = [0.0; 8];
    let mut lhs = [1.0; 8];
    unsafe {
        // error-path: IpoptSolverKktSolve
        assert_eq!(
            IpoptSolverKktSolve(std::ptr::null_mut(), rhs.as_ptr(), lhs.as_mut_ptr()),
            FALSE
        );
        // error-path: IpoptSolverKktSolveScaled
        assert_eq!(
            IpoptSolverKktSolveScaled(std::ptr::null_mut(), rhs.as_ptr(), lhs.as_mut_ptr()),
            FALSE
        );
    }
    // A session that has not solved holds no factor.
    let mut p = create_quad();
    let fresh = unsafe { IpoptCreateSolver(&mut p) };
    unsafe {
        assert_eq!(
            IpoptSolverKktSolve(fresh, rhs.as_ptr(), lhs.as_mut_ptr()),
            FALSE
        );
        assert_eq!(
            IpoptSolverKktSolveScaled(fresh, rhs.as_ptr(), lhs.as_mut_ptr()),
            FALSE
        );
        IpoptFreeSolver(fresh);
    }

    // With a factor: both back-solves succeed, and with no NLP scaling on
    // this model (no constraints, gradient-based scaling leaves f alone at
    // |∇f(x0)| = 4 < 100) they give the same answer.
    let s = solved_session();
    let dim = unsafe { IpoptSolverGetKktDim(s) } as usize;
    assert!(dim > 0);
    let rhs: Vec<Number> = (0..dim).map(|i| 1.0 + i as Number).collect();
    let mut a = vec![0.0; dim];
    let mut b = vec![0.0; dim];
    unsafe {
        assert_eq!(IpoptSolverKktSolve(s, rhs.as_ptr(), a.as_mut_ptr()), TRUE);
        assert_eq!(
            IpoptSolverKktSolveScaled(s, rhs.as_ptr(), b.as_mut_ptr()),
            TRUE
        );
        assert_eq!(
            IpoptSolverKktSolveScaled(s, std::ptr::null(), b.as_mut_ptr()),
            FALSE,
            "NULL rhs"
        );
        IpoptFreeSolver(s);
    }
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-12 * (1.0 + x.abs()), "{a:?} vs {b:?}");
    }
}

#[test]
fn parametric_step_and_reduced_hessian_refuse_bad_inputs() {
    let pins: [Index; 1] = [0];
    let deltas = [0.1];
    let mut out = [0.0; 4];
    unsafe {
        // error-path: IpoptSolverParametricStep
        assert_eq!(
            IpoptSolverParametricStep(
                std::ptr::null_mut(),
                1,
                pins.as_ptr(),
                deltas.as_ptr(),
                out.as_mut_ptr()
            ),
            FALSE
        );
        // error-path: IpoptSolverReducedHessian
        assert_eq!(
            IpoptSolverReducedHessian(
                std::ptr::null_mut(),
                1,
                pins.as_ptr(),
                1.0,
                out.as_mut_ptr()
            ),
            FALSE
        );
    }
    let s = solved_session();
    unsafe {
        // Negative pin counts and NULL outputs are refused on a live session.
        assert_eq!(
            IpoptSolverParametricStep(s, -1, pins.as_ptr(), deltas.as_ptr(), out.as_mut_ptr()),
            FALSE
        );
        assert_eq!(
            IpoptSolverParametricStep(
                s,
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut()
            ),
            FALSE
        );
        assert_eq!(
            IpoptSolverReducedHessian(s, 0, std::ptr::null(), 1.0, std::ptr::null_mut()),
            FALSE
        );
        // This model has no constraints, so pin index 0 is out of range.
        assert_eq!(
            IpoptSolverReducedHessian(s, 1, pins.as_ptr(), 1.0, out.as_mut_ptr()),
            FALSE
        );
        IpoptFreeSolver(s);
    }
}

// ---------------------------------------------------------------------------
// Fortran shim
// ---------------------------------------------------------------------------

unsafe extern "C" fn fq_f(
    _n: *const Index,
    x: *mut Number,
    _new_x: *const Index,
    obj: *mut Number,
    _idat: *mut Index,
    _ddat: *mut Number,
    ierr: *mut Index,
) {
    unsafe {
        *obj = (*x - 3.0) * (*x - 3.0);
        *ierr = 0;
    }
}

unsafe extern "C" fn fq_grad(
    _n: *const Index,
    x: *mut Number,
    _new_x: *const Index,
    grad: *mut Number,
    _idat: *mut Index,
    _ddat: *mut Number,
    ierr: *mut Index,
) {
    unsafe {
        *grad = 2.0 * (*x - 3.0);
        *ierr = 0;
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fq_hess(
    task: *const Index,
    _n: *const Index,
    _x: *mut Number,
    _new_x: *const Index,
    obj_factor: *const Number,
    _m: *const Index,
    _lambda: *mut Number,
    _new_lambda: *const Index,
    _nnz: *const Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    _idat: *mut Index,
    _ddat: *mut Number,
    ierr: *mut Index,
) {
    unsafe {
        if *task == 0 {
            *irow = 0;
            *jcol = 0;
        } else {
            *values = 2.0 * *obj_factor;
        }
        *ierr = 0;
    }
}

static F_CB_FIRES: AtomicUsize = AtomicUsize::new(0);

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn f_counting_cb(
    _alg_mode: *const Index,
    _iter: *const Index,
    _obj: *const Number,
    _inf_pr: *const Number,
    _inf_du: *const Number,
    _mu: *const Number,
    _d_norm: *const Number,
    _regu: *const Number,
    _alpha_du: *const Number,
    _alpha_pr: *const Number,
    _ls: *const Index,
    _idat: *mut Index,
    _ddat: *mut Number,
    istop: *mut Index,
) {
    F_CB_FIRES.fetch_add(1, Ordering::SeqCst);
    unsafe { *istop = 0 };
}

fn fcreate(n: Index) -> *mut c_void {
    let (m, nj, nh, sty) = (0, 0, 1, 0);
    let xl = [-1.0e20];
    let xu = [1.0e20];
    unsafe {
        ipcreate_(
            &n,
            xl.as_ptr(),
            xu.as_ptr(),
            &m,
            std::ptr::null(),
            std::ptr::null(),
            &nj,
            &nh,
            &sty,
            fq_f,
            None,
            fq_grad,
            None,
            Some(fq_hess),
        )
    }
}

fn fsolve(fp: &mut *mut c_void) -> (Index, Number) {
    let mut x = [0.0];
    let mut obj = 0.0;
    let mut idat = [0];
    let mut ddat = [0.0];
    let rc = unsafe {
        ipsolve_(
            fp,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            idat.as_mut_ptr(),
            ddat.as_mut_ptr(),
        )
    };
    (rc, x[0])
}

/// Fortran strings: a buffer plus an explicit length, no NUL, blank-padded.
fn fstr(s: &str) -> (Vec<u8>, c_int) {
    let mut v = s.as_bytes().to_vec();
    v.extend_from_slice(b"    ");
    let len = v.len() as c_int;
    (v, len)
}

#[test]
fn fortran_entry_points_refuse_null_and_invalid_handles() {
    // error-path: ipcreate_
    assert!(fcreate(-1).is_null(), "negative n");
    // error-path: ipfree_
    unsafe { ipfree_(std::ptr::null_mut()) };
    let mut null_slot: *mut c_void = std::ptr::null_mut();
    unsafe { ipfree_(&mut null_slot) };
    // error-path: ipsolve_
    assert_eq!(fsolve(&mut null_slot).0, INTERNAL_ERROR);

    let (k, kl) = fstr("print_level");
    let (v, vl) = fstr("yes");
    let one: Number = 1.0;
    let zero: Index = 0;
    let kp = k.as_ptr() as *const c_char;
    unsafe {
        // error-path: ipaddstroption_
        assert_eq!(
            ipaddstroption_(&mut null_slot, kp, v.as_ptr() as *const c_char, kl, vl),
            1
        );
        // error-path: ipaddnumoption_
        assert_eq!(ipaddnumoption_(&mut null_slot, kp, &one, kl), 1);
        // error-path: ipaddintoption_
        assert_eq!(ipaddintoption_(&mut null_slot, kp, &zero, kl), 1);
        // error-path: ipsetcallback_
        ipsetcallback_(&mut null_slot, f_counting_cb);
        ipsetcallback_(std::ptr::null_mut(), f_counting_cb);
        // error-path: ipunsetcallback_
        ipunsetcallback_(&mut null_slot);
        ipunsetcallback_(std::ptr::null_mut());
    }
}

#[test]
fn fortran_options_and_callback_round_trip() {
    let mut fp = fcreate(1);
    assert!(!fp.is_null());
    let (k_pl, l_pl) = fstr("print_level");
    let (k_sb, l_sb) = fstr("sb");
    let (v_yes, l_yes) = fstr("yes");
    let (k_tol, l_tol) = fstr("tol");
    let (k_bad, l_bad) = fstr("no_such_option_at_all");
    let zero: Index = 0;
    let tol: Number = 1e-9;
    unsafe {
        assert_eq!(ipaddintoption_(&mut fp, k_pl.as_ptr() as _, &zero, l_pl), 0);
        assert_eq!(
            ipaddstroption_(
                &mut fp,
                k_sb.as_ptr() as _,
                v_yes.as_ptr() as _,
                l_sb,
                l_yes
            ),
            0
        );
        assert_eq!(
            ipaddnumoption_(&mut fp, k_tol.as_ptr() as _, &tol, l_tol),
            0
        );
        // A live handle with an unknown keyword is refused per setter.
        assert_eq!(
            ipaddintoption_(&mut fp, k_bad.as_ptr() as _, &zero, l_bad),
            1
        );
        assert_eq!(
            ipaddnumoption_(&mut fp, k_bad.as_ptr() as _, &tol, l_bad),
            1
        );
        assert_eq!(
            ipaddstroption_(
                &mut fp,
                k_bad.as_ptr() as _,
                v_yes.as_ptr() as _,
                l_bad,
                l_yes
            ),
            1
        );
    }

    // With the callback set it fires; after unset it does not.
    F_CB_FIRES.store(0, Ordering::SeqCst);
    unsafe { ipsetcallback_(&mut fp, f_counting_cb) };
    let (rc, x) = fsolve(&mut fp);
    assert_eq!(rc, 0);
    assert!((x - 3.0).abs() < 1e-6, "x = {x}");
    let fired = F_CB_FIRES.load(Ordering::SeqCst);
    assert!(fired > 0, "callback never fired");

    unsafe { ipunsetcallback_(&mut fp) };
    let (rc, _) = fsolve(&mut fp);
    assert_eq!(rc, 0);
    assert_eq!(
        F_CB_FIRES.load(Ordering::SeqCst),
        fired,
        "unset callback still fired"
    );

    unsafe { ipfree_(&mut fp) };
    assert!(fp.is_null(), "ipfree_ zeroes the caller's slot");
}
