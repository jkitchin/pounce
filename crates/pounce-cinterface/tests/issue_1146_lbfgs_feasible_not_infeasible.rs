//! gh#1146 — under `hessian_approximation = limited-memory`, a feasible
//! 1-variable NLP started from `x0 = 0` was reported
//! `Infeasible_Problem_Detected` at a point that satisfied every row.
//!
//! ```text
//! min  0
//! s.t. 0    <= x      <= 1
//!      0.1  <= x      <= 1.1
//!      0.25 <= x      <= 0.75
//!      -1   <= log(x) <= 1        feasible set: x in [1/e, 0.75]
//! ```
//!
//! Gradient scaling samples the Jacobian at the raw `x0 = 0`, where the
//! `log` row's derivative is `1/0 = inf`, so the row is scaled by
//! `nlp_scaling_min_value = 1e-8` (upstream does the same: Ipopt prints
//! `d scaling vector[3] = 1e-08` on this model). At iteration 10 the iterate
//! `x = 0.444` satisfied every declared row — the `inf_pr` column printed
//! `0.00` — but the slack of the `log` row had drifted from `log(x)` by
//! `1.4e-2`. Rapid infeasibility detection reads the internal measure
//! `max(‖c‖, ‖d − s‖)`, unscaled: `1.4e-2` cleared its `1e-2` floor, while
//! the scaled infeasibility gradient (`1.4e-10` of gap) looked stationary.
//! Verdict: infeasible, at a feasible point. Upstream has no such detector;
//! with `recalc_y = no` (pounce's L-BFGS default, see #677) Ipopt walks the
//! identical trajectory through that iterate and converges in 15
//! iterations, as pounce now does.
//!
//! The fix refuses the verdict when the point satisfies the declared rows
//! and box to `constr_viol_tol`. The second-opinion ladder is switched off
//! throughout, so the first solve is what is asserted — with it on, a
//! rescued re-solve would hide a regression.
//!
//! The rule branches on whether the point is declared-feasible, so the
//! genuinely infeasible variant (`log(x) >= 0`, i.e. `x >= 1`, against
//! `x <= 0.75`) is here too: it must still be convicted.
//!
//! **The exact-Hessian leg** fails differently, so it gets its own test. It
//! matches Ipopt digit for digit through iteration 9, and Ipopt's 10th step
//! reaches `inf_pr = 0`. Pounce instead convicted at iteration 9 (`x = 0.349`),
//! where the `log` row really is violated (by `0.057`), so the gate above
//! rightly stays out of it. What failed was the no-descent probe the detector
//! convicts on. It measured `θ = ‖(c, d − s)‖` with the slacks held fixed, so
//! every probe step opened a gap `Δx` on the two linear rows. Their slacks sat
//! exactly on `d(x)`; those rows are satisfied and stay satisfied for any small
//! move. And under row scaling `(1, 1, 1e-8)` the `log` row's real `0.057`
//! counted as `5.7e-10`. A step of `+0.1`, which makes the model feasible,
//! read as a 17 000-fold increase. The probe now measures the model's own
//! violation, per-row in user units, and the solve takes Ipopt's step and
//! lands on Ipopt's answer (`x = 0.612028`).

use pounce_cinterface::*;
use std::ffi::{CString, c_void};

const N: usize = 1;
const M: usize = 3;

unsafe extern "C" fn ev_f(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        *obj = 0.0;
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        *grad = 0.0;
        1
    }
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
        let x = *x;
        let g = std::slice::from_raw_parts_mut(gout, M);
        g[0] = x;
        g[1] = x;
        g[2] = x.ln();
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
            std::slice::from_raw_parts_mut(i_row, M).copy_from_slice(&[0, 1, 2]);
            std::slice::from_raw_parts_mut(j_col, M).copy_from_slice(&[0, 0, 0]);
            return 1;
        }
        // `1/x` at `x = 0` is `inf`: deliberately not guarded, that is the
        // value gradient scaling samples at the raw start.
        let x = *x;
        std::slice::from_raw_parts_mut(values, M).copy_from_slice(&[1.0, 1.0, 1.0 / x]);
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            *i_row = 0;
            *j_col = 0;
            return 1;
        }
        let x = *x;
        let lambda = std::slice::from_raw_parts(lambda, M);
        *values = -lambda[2] / (x * x);
        1
    }
}

fn set_str(prob: IpoptProblem, k: &str, v: &str) {
    let k = CString::new(k).unwrap();
    let v = CString::new(v).unwrap();
    unsafe {
        assert_ne!(
            AddIpoptStrOption(prob, k.as_ptr() as *mut _, v.as_ptr() as *mut _),
            0,
            "option {k:?} rejected"
        );
    }
}

/// Solve from `x0`, with the `log` row's lower bound `log_lo`, returning
/// `(status, x, g)`. Every second-opinion rung is off.
fn solve(hessian: &str, x0: Number, log_lo: Number) -> (i32, Number, [Number; M]) {
    solve_with(hessian, x0, log_lo, &[])
}

/// [`solve`] with extra string options.
fn solve_with(
    hessian: &str,
    x0: Number,
    log_lo: Number,
    extra: &[(&str, &str)],
) -> (i32, Number, [Number; M]) {
    unsafe {
        let x_l = [0.0 as Number];
        let x_u = [1.0 as Number];
        let g_l = [0.1 as Number, 0.25, log_lo];
        let g_u = [1.1 as Number, 0.75, 1.0];
        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            M as Index,
            1,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        );
        assert!(!prob.is_null());
        let key = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0), 0);
        set_str(prob, "hessian_approximation", hessian);
        for rung in [
            "feral_infeasibility_scaling_retry",
            "infeasibility_mu_strategy_retry",
            "infeasibility_perturbed_start_retry",
            "feral_increase_quality_retry",
            "infeasibility_saddle_escape_retry",
        ] {
            set_str(prob, rung, "no");
        }
        for &(k, v) in extra {
            set_str(prob, k, v);
        }
        let mut x = [x0];
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
            std::ptr::null_mut(),
        );
        FreeIpoptProblem(prob);
        (status as i32, x[0], g)
    }
}

const SOLVE_SUCCEEDED: i32 = 0;
const INFEASIBLE_PROBLEM_DETECTED: i32 = 2;

#[test]
fn lbfgs_from_the_log_singularity_is_not_convicted_at_a_feasible_point() {
    let (status, x, g) = solve("limited-memory", 0.0, -1.0);
    assert_eq!(
        status, SOLVE_SUCCEEDED,
        "gh#1146: x = {x}, g = {g:?} (every row holds on [1/e, 0.75])"
    );
    let tol = 1e-6;
    assert!(
        (-1.0f64).exp() - tol <= x && x <= 0.75 + tol,
        "x = {x} outside the feasible set [1/e, 0.75]"
    );
}

#[test]
fn lbfgs_from_a_feasible_start_still_solves() {
    // Control: the start the issue reports as already working.
    let (status, x, _) = solve("limited-memory", 0.5, -1.0);
    assert_eq!(status, SOLVE_SUCCEEDED, "x = {x}");
}

#[test]
fn exact_hessian_from_the_log_singularity_is_not_convicted_one_step_from_feasible() {
    let (status, x, g) = solve("exact", 0.0, -1.0);
    assert_eq!(
        status, SOLVE_SUCCEEDED,
        "gh#1146 exact leg: x = {x}, g = {g:?} (every row holds on [1/e, 0.75])"
    );
    let tol = 1e-6;
    assert!(
        (-1.0f64).exp() - tol <= x && x <= 0.75 + tol,
        "x = {x} outside the feasible set [1/e, 0.75]"
    );
}

/// The third defect gh#1146 surfaced, in restoration. Unscaled, the L-BFGS
/// run enters restoration at outer iteration 7 and matches Ipopt exactly up
/// to there. Pounce's restoration then ran its limited-memory update over
/// the whole compound primal `[x | n_c | p_c | n_d | p_d]`; upstream confines
/// it to the `x` block and leaves the linear slack blocks at zero curvature.
/// The slack motion diluted every curvature pair. The first restoration step
/// came out `‖d‖ = 3.63e3` against Ipopt's `2.10e3`, and the solve ended
/// `Restoration_Failed` at `x = 0.0429`. Confined to the `x` block, it
/// follows Ipopt to the iterate and returns Ipopt's answer, `x = 0.44927`,
/// in 14 iterations.
#[test]
fn unscaled_lbfgs_restoration_recovers_as_upstream_does() {
    let (status, x, g) = solve_with(
        "limited-memory",
        0.0,
        -1.0,
        &[("nlp_scaling_method", "none")],
    );
    assert_eq!(status, SOLVE_SUCCEEDED, "x = {x}, g = {g:?}");
    assert!(
        (x - 0.44927).abs() < 1e-4,
        "x = {x}; Ipopt 3.11.9 returns 0.44927"
    );
}

#[test]
fn the_infeasible_variant_is_still_convicted() {
    // `log(x) >= 0` needs `x >= 1`, but `x <= 0.75`: genuinely infeasible.
    // The point the solver stops at violates a declared row, so neither the
    // gate nor the model-measured descent probe may acquit it — from either
    // start, on either Hessian.
    for hessian in ["limited-memory", "exact"] {
        for x0 in [0.0, 0.5] {
            let (status, x, g) = solve(hessian, x0, 0.0);
            assert_eq!(
                status, INFEASIBLE_PROBLEM_DETECTED,
                "{hessian}, x0 = {x0}: x = {x}, g = {g:?}"
            );
        }
    }
}
