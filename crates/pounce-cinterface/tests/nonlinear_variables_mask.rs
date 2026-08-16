//! gh#624 — nonlinear-variable subsets for the limited-memory Hessian,
//! end to end through the C API.
//!
//! The model has six variables, only two of which (`x0`, `x1`) appear
//! nonlinearly; the other four enter the objective and the constraints
//! linearly. `IpoptSetNonlinearVariables` declares that, which restricts
//! the L-BFGS approximation to `{x0, x1}` and leaves the Hessian exactly
//! zero on the linear block.
//!
//! What the tests pin down:
//!   * masked and unmasked solves agree on the KKT point — the subset is
//!     an approximation-space restriction, not a different problem;
//!   * declaring the subset does not change the exact-Hessian path;
//!   * the argument validation refuses a bad declaration outright rather
//!     than half-applying it.

use pounce_cinterface::*;
use std::ffi::{CString, c_void};

const N: usize = 6;
const M: usize = 2;
/// Variables that actually enter nonlinearly.
const NONLIN: [Index; 2] = [0, 1];

// f(x) = (1-x0)² + 100(x1-x0²)² + 2x2 + 3x3 + x4 + x5
unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let x = std::slice::from_raw_parts(x, N);
        *obj = (1.0 - x[0]).powi(2)
            + 100.0 * (x[1] - x[0] * x[0]).powi(2)
            + 2.0 * x[2]
            + 3.0 * x[3]
            + x[4]
            + x[5];
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
        g[2] = 2.0;
        g[3] = 3.0;
        g[4] = 1.0;
        g[5] = 1.0;
        1
    }
}

// g0(x) = x0 + x1 + x2 + x3        == 4      (linear)
// g1(x) = x0² + x1² - x4 - x5      <= 1      (x4, x5 linear)
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
        let g = std::slice::from_raw_parts_mut(gout, M);
        g[0] = x[0] + x[1] + x[2] + x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] - x[4] - x[5];
        1
    }
}

/// Dense 2×6 Jacobian.
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
            let rows = std::slice::from_raw_parts_mut(i_row, M * N);
            let cols = std::slice::from_raw_parts_mut(j_col, M * N);
            let mut k = 0;
            for i in 0..M {
                for j in 0..N {
                    rows[k] = i as Index;
                    cols[k] = j as Index;
                    k += 1;
                }
            }
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let v = std::slice::from_raw_parts_mut(values, M * N);
        v[..N].copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
        v[N..].copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 0.0, 0.0, -1.0, -1.0]);
        1
    }
}

/// Lower-triangular Hessian of the Lagrangian — only the `(0,0)`,
/// `(1,0)`, `(1,1)` block is ever nonzero, which is the structural fact
/// the mask lets the limited-memory path exploit.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    const NNZ: usize = 3;
    unsafe {
        if values.is_null() {
            let rows = std::slice::from_raw_parts_mut(i_row, NNZ);
            let cols = std::slice::from_raw_parts_mut(j_col, NNZ);
            rows.copy_from_slice(&[0, 1, 1]);
            cols.copy_from_slice(&[0, 0, 1]);
            return 1;
        }
        let x = std::slice::from_raw_parts(x, N);
        let lam = std::slice::from_raw_parts(lambda, M);
        let v = std::slice::from_raw_parts_mut(values, NNZ);
        v[0] = obj_factor * (2.0 - 400.0 * (x[1] - 3.0 * x[0] * x[0])) + lam[1] * 2.0;
        v[1] = obj_factor * (-400.0 * x[0]);
        v[2] = obj_factor * 200.0 + lam[1] * 2.0;
        1
    }
}

struct Solved {
    status: i32,
    x: [Number; N],
    obj: Number,
    iters: Index,
}

/// Solve the model. `mask` declares the nonlinear subset when `Some`.
fn solve(limited_memory: bool, mask: Option<&[Index]>) -> Solved {
    unsafe {
        let x_l = [-5.0 as Number; N];
        let x_u = [5.0 as Number; N];
        let g_l = [4.0 as Number, -2.0e19];
        let g_u = [4.0 as Number, 1.0];

        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            (M * N) as Index,
            if limited_memory { 0 } else { 3 },
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            if limited_memory { None } else { Some(ev_h) },
        );
        assert!(!prob.is_null());

        if limited_memory {
            let key = CString::new("hessian_approximation").unwrap();
            let val = CString::new("limited-memory").unwrap();
            assert_ne!(
                AddIpoptStrOption(prob, key.as_ptr() as *mut _, val.as_ptr() as *mut _),
                0
            );
        }
        let key = CString::new("print_level").unwrap();
        assert_ne!(AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0), 0);
        let key = CString::new("tol").unwrap();
        assert_ne!(AddIpoptNumOption(prob, key.as_ptr() as *mut _, 1e-9), 0);

        if let Some(mask) = mask {
            assert_ne!(
                IpoptSetNonlinearVariables(prob, mask.len() as Index, mask.as_ptr()),
                0,
                "IpoptSetNonlinearVariables refused a valid subset"
            );
        }

        let mut x = [0.0 as Number, 0.0, 1.0, 1.0, 0.0, 0.0];
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
        let iters = GetIpoptIterCount(prob);
        FreeIpoptProblem(prob);
        Solved {
            status: status as i32,
            x,
            obj,
            iters,
        }
    }
}

fn assert_same_solution(a: &Solved, b: &Solved, tol: Number, what: &str) {
    assert_eq!(a.status, 0, "{what}: first solve did not succeed");
    assert_eq!(b.status, 0, "{what}: second solve did not succeed");
    assert!(
        (a.obj - b.obj).abs() < tol,
        "{what}: objectives differ: {} vs {}",
        a.obj,
        b.obj
    );
    for i in 0..N {
        assert!(
            (a.x[i] - b.x[i]).abs() < tol,
            "{what}: x[{i}] differs: {} vs {}",
            a.x[i],
            b.x[i]
        );
    }
}

#[test]
fn masked_and_full_space_lbfgs_reach_the_same_kkt_point() {
    let full = solve(true, None);
    let masked = solve(true, Some(&NONLIN));
    assert_same_solution(&full, &masked, 1e-5, "limited-memory");
    // Not an assertion about speed — just a record that the masked run
    // is a real solve and not a no-op fall-through.
    assert!(masked.iters > 0);
}

#[test]
fn masked_lbfgs_matches_the_exact_hessian_solution() {
    let exact = solve(false, None);
    let masked = solve(true, Some(&NONLIN));
    assert_same_solution(&exact, &masked, 1e-5, "masked vs exact");
}

#[test]
fn declaring_the_subset_does_not_disturb_the_exact_hessian_path() {
    // The declaration is an L-BFGS-only concern; an exact-Hessian solve
    // must be bit-for-bit what it was without it.
    let plain = solve(false, None);
    let declared = solve(false, Some(&NONLIN));
    assert_eq!(plain.iters, declared.iters);
    for i in 0..N {
        assert_eq!(plain.x[i], declared.x[i]);
    }
    assert_eq!(plain.obj, declared.obj);
}

#[test]
fn declaring_every_variable_is_the_same_as_declaring_nothing() {
    let all: Vec<Index> = (0..N as Index).collect();
    let none = solve(true, None);
    let everything = solve(true, Some(&all));
    assert_eq!(none.iters, everything.iters);
    assert_eq!(none.obj, everything.obj);
}

#[test]
fn bad_declarations_are_refused_and_leave_the_problem_untouched() {
    unsafe {
        let x_l = [-5.0 as Number; N];
        let x_u = [5.0 as Number; N];
        let g_l = [4.0 as Number, -2.0e19];
        let g_u = [4.0 as Number, 1.0];
        let prob = CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            (M * N) as Index,
            0,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            None,
        );
        assert!(!prob.is_null());

        let good: [Index; 2] = [0, 1];
        assert_ne!(IpoptSetNonlinearVariables(prob, 2, good.as_ptr()), 0);

        // Out of range, oversized, negative, and NULL-with-count all fail…
        let bad: [Index; 2] = [0, N as Index];
        assert_eq!(IpoptSetNonlinearVariables(prob, 2, bad.as_ptr()), 0);
        let oversized: Vec<Index> = vec![0; N + 1];
        assert_eq!(
            IpoptSetNonlinearVariables(prob, (N + 1) as Index, oversized.as_ptr()),
            0
        );
        assert_eq!(IpoptSetNonlinearVariables(prob, -1, good.as_ptr()), 0);
        assert_eq!(IpoptSetNonlinearVariables(prob, 2, std::ptr::null()), 0);
        assert_eq!(
            IpoptSetNonlinearVariables(std::ptr::null_mut(), 2, good.as_ptr()),
            0
        );

        // …and none of them disturbed the accepted declaration: the
        // solve still runs, and still agrees with the unmasked answer.
        let key = CString::new("hessian_approximation").unwrap();
        let val = CString::new("limited-memory").unwrap();
        AddIpoptStrOption(prob, key.as_ptr() as *mut _, val.as_ptr() as *mut _);
        let key = CString::new("print_level").unwrap();
        AddIpoptIntOption(prob, key.as_ptr() as *mut _, 0);

        let mut x = [0.0 as Number, 0.0, 1.0, 1.0, 0.0, 0.0];
        let mut obj = 0.0 as Number;
        let status = IpoptSolve(
            prob,
            x.as_mut_ptr(),
            std::ptr::null_mut(),
            &mut obj,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        assert_ne!(IpoptClearNonlinearVariables(prob), 0);
        FreeIpoptProblem(prob);
        assert_eq!(status as i32, 0);

        let reference = solve(true, None);
        for i in 0..N {
            assert!((x[i] - reference.x[i]).abs() < 1e-5);
        }
    }
}
