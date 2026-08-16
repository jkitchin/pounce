//! Regression test for the `GetIpoptCurrent*` inspectors called from
//! inside an intermediate callback — the documented usage.
//!
//! Both functions used to hold a shared borrow of the algorithm-side
//! `IpoptNlp` across an `IpoptCq` accessor that re-enters the NLP
//! mutably (`curr_c` / `curr_d` / `curr_grad_f`). Asking for `g`
//! (`GetIpoptCurrentIterate`) or `grad_lag_x`
//! (`GetIpoptCurrentViolations`) therefore panicked with "RefCell
//! already borrowed" — and because the panic crosses an `extern "C"`
//! boundary, the process *aborted* instead of returning `FALSE`.
//!
//! Any consumer that mirrors Ipopt's callback contract hits this: a
//! CasADi `nlpsol` plugin populating `iteration_callback`, the GAMS C
//! link, or a plain C driver logging its iterates.

use pounce_cinterface::*;
use std::cell::RefCell;
use std::ffi::{CString, c_void};

const N: usize = 2;
const M: usize = 1;

/// What one intermediate-callback fire observed.
#[derive(Default)]
struct Observed {
    iterate_calls: usize,
    iterate_ok: usize,
    violations_calls: usize,
    violations_ok: usize,
    /// `|g - (x0² + x1² - 1.5)|` at the reported iterate, worst case.
    worst_g_mismatch: Number,
    /// Whether any reported `x` was all zeros (a sign we read a stale
    /// or unpopulated buffer rather than the live iterate).
    saw_nonzero_x: bool,
}

thread_local! {
    static OBSERVED: RefCell<Observed> = RefCell::new(Observed::default());
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(grad, N);
        g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
        g[1] = 200.0 * (x[1] - x[0] * x[0]);
        1
    }
}

fn con(x: &[Number]) -> Number {
    x[0] * x[0] + x[1] * x[1] - 1.5
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *gout = con(x);
        1
    }
}

unsafe extern "C" fn ev_jac_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, N);
            let cols = std::slice::from_raw_parts_mut(j_col, N);
            for j in 0..N {
                rows[j] = 0;
                cols[j] = j as Index;
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, N);
        v[0] = 2.0 * x[0];
        v[1] = 2.0 * x[1];
        1
    }
}

/// Ask both inspectors for *every* buffer they can fill, exactly as an
/// `nlpsol`-style plugin does when it forwards the iterate to a user
/// callback.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn on_iter(
    _alg_mod: Index,
    _iter: Index,
    _obj: Number,
    _inf_pr: Number,
    _inf_du: Number,
    _mu: Number,
    _d_norm: Number,
    _regu: Number,
    _alpha_du: Number,
    _alpha_pr: Number,
    _ls_trials: Index,
    user_data: *mut c_void,
) -> Bool {
    unsafe {
        let prob = user_data as IpoptProblem;

        let mut x = [0.0 as Number; N];
        let mut z_l = [0.0 as Number; N];
        let mut z_u = [0.0 as Number; N];
        let mut g = [0.0 as Number; M];
        let mut lambda = [0.0 as Number; M];
        let ok_iterate = GetIpoptCurrentIterate(
            prob,
            0,
            N as Index,
            x.as_mut_ptr(),
            z_l.as_mut_ptr(),
            z_u.as_mut_ptr(),
            M as Index,
            g.as_mut_ptr(),
            lambda.as_mut_ptr(),
        );

        let mut x_l_viol = [0.0 as Number; N];
        let mut x_u_viol = [0.0 as Number; N];
        let mut compl_x_l = [0.0 as Number; N];
        let mut compl_x_u = [0.0 as Number; N];
        let mut grad_lag_x = [0.0 as Number; N];
        let mut g_viol = [0.0 as Number; M];
        let mut compl_g = [0.0 as Number; M];
        let ok_viol = GetIpoptCurrentViolations(
            prob,
            0,
            N as Index,
            x_l_viol.as_mut_ptr(),
            x_u_viol.as_mut_ptr(),
            compl_x_l.as_mut_ptr(),
            compl_x_u.as_mut_ptr(),
            grad_lag_x.as_mut_ptr(),
            M as Index,
            g_viol.as_mut_ptr(),
            compl_g.as_mut_ptr(),
        );

        OBSERVED.with(|o| {
            let mut o = o.borrow_mut();
            o.iterate_calls += 1;
            o.violations_calls += 1;
            if ok_iterate != 0 {
                o.iterate_ok += 1;
                if x.iter().any(|v| *v != 0.0) {
                    o.saw_nonzero_x = true;
                }
                let mismatch = (g[0] - con(&x)).abs();
                if mismatch > o.worst_g_mismatch {
                    o.worst_g_mismatch = mismatch;
                }
            }
            if ok_viol != 0 {
                o.violations_ok += 1;
            }
        });
        1
    }
}

#[test]
fn current_iterate_inspectors_are_usable_from_the_callback() {
    unsafe {
        let x_l = [-5.0 as Number; N];
        let x_u = [5.0 as Number; N];
        let g_l = [-2.0e19 as Number];
        let g_u = [0.0 as Number];

        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            N as Index,
            0,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            None,
        );
        assert!(!prob.is_null(), "CreateIpoptProblem returned NULL");

        let key = CString::new("hessian_approximation").unwrap();
        let val = CString::new("limited-memory").unwrap();
        assert_ne!(
            AddIpoptStrOption(prob, key.as_ptr() as *mut _, val.as_ptr() as *mut _),
            0
        );
        let key = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0), 0);

        assert_ne!(SetIntermediateCallback(prob, Some(on_iter)), 0);

        let mut x = [0.5 as Number, 0.5];
        let mut g = [0.0 as Number; M];
        let mut obj = 0.0 as Number;
        let mut mult_g = [0.0 as Number; M];
        let mut z_l = [0.0 as Number; N];
        let mut z_u = [0.0 as Number; N];
        let status = IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            mult_g.as_mut_ptr(),
            z_l.as_mut_ptr(),
            z_u.as_mut_ptr(),
            prob as *mut c_void,
        );
        FreeIpoptProblem(prob);

        assert_eq!(status as i32, 0, "expected Solve_Succeeded, got {status:?}");

        OBSERVED.with(|o| {
            let o = o.borrow();
            assert!(o.iterate_calls > 0, "intermediate callback never fired");
            assert_eq!(
                o.iterate_ok, o.iterate_calls,
                "GetIpoptCurrentIterate refused inside the callback"
            );
            assert_eq!(
                o.violations_ok, o.violations_calls,
                "GetIpoptCurrentViolations refused inside the callback"
            );
            assert!(o.saw_nonzero_x, "reported iterate was all zeros");
            assert!(
                o.worst_g_mismatch < 1e-9,
                "reported g disagrees with g(x) at the reported iterate by {}",
                o.worst_g_mismatch
            );
        });
    }
}
