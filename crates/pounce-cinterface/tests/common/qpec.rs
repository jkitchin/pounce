//! `QpecSmall` as C callbacks, shared by the gh#1147 complementarity tests
//! (include with `#[path = "common/qpec.rs"] mod qpec;`).
//!
//! The same model as `python/tests/test_complementarity.py::QpecSmall` and
//! `crates/pounce-cli/tests/fixtures_mpcc/qpec_small.nl`:
//!
//! ```text
//! min (x-1)^2 + (y1-1)^2 + y2^2,   0 <= x <= 2,
//!     0 <= y1  ⟂  2 y1 - 1 - x >= 0,
//!     0 <= y2  ⟂  x + 2 y2 - 1 >= 0.
//! ```
//!
//! Answer `(1, 1, 0)`, the second pair biactive there; S-stationary. The
//! declared pairs are `(row 0, var 1)` and `(row 1, var 2)`, 0-based.
//!
//! The Jacobian and Hessian structure callbacks report indices in the
//! style given as the const parameter (`OFF = 0` for C, `1` for Fortran),
//! so a problem created with `index_style = 1` must pass the `<1>`
//! instances; [`create`] picks them.

#![allow(dead_code)]

use pounce_cinterface::*;
use std::ffi::c_void;

pub const N: usize = 3;
pub const M: usize = 2;
/// Bounds at or beyond `1e19` are infinite; Python's test uses `2e19`.
pub const INF: Number = 2e19;

pub const X_L: [Number; N] = [0.0, 0.0, 0.0];
pub const X_U: [Number; N] = [2.0, INF, INF];
pub const G_L: [Number; M] = [0.0, 0.0];
pub const G_U: [Number; M] = [INF, INF];

/// The two declared pairs, 0-based `(row, var)`.
pub const PAIRS: [(Index, Index); 2] = [(0, 1), (1, 2)];

pub unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let z = std::slice::from_raw_parts(x, N);
        *obj = (z[0] - 1.0).powi(2) + (z[1] - 1.0).powi(2) + z[2] * z[2];
        1
    }
}

pub unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let z = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(grad, N);
        g[0] = 2.0 * (z[0] - 1.0);
        g[1] = 2.0 * (z[1] - 1.0);
        g[2] = 2.0 * z[2];
        1
    }
}

pub unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        let z = std::slice::from_raw_parts(x, N);
        let g = std::slice::from_raw_parts_mut(gout, M);
        g[0] = 2.0 * z[1] - 1.0 - z[0];
        g[1] = z[0] + 2.0 * z[2] - 1.0;
        1
    }
}

pub const NELE_JAC: usize = 4;
pub const NELE_HESS: usize = 3;

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ev_jac_g<const OFF: Index>(
    _n: Index,
    _x: *const Number,
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
            let r = std::slice::from_raw_parts_mut(i_row, NELE_JAC);
            let c = std::slice::from_raw_parts_mut(j_col, NELE_JAC);
            for (k, (i, j)) in [(0, 0), (0, 1), (1, 0), (1, 2)].into_iter().enumerate() {
                r[k] = i + OFF;
                c[k] = j + OFF;
            }
            return 1;
        }
        let v = std::slice::from_raw_parts_mut(values, NELE_JAC);
        v.copy_from_slice(&[-1.0, 2.0, 1.0, 2.0]);
        1
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn ev_h<const OFF: Index>(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    obj_factor: Number,
    _m: Index,
    _lambda: *const Number,
    _new_lambda: Bool,
    _nele_hess: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    _u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let r = std::slice::from_raw_parts_mut(i_row, NELE_HESS);
            let c = std::slice::from_raw_parts_mut(j_col, NELE_HESS);
            for k in 0..NELE_HESS {
                r[k] = k as Index + OFF;
                c[k] = k as Index + OFF;
            }
            return 1;
        }
        let v = std::slice::from_raw_parts_mut(values, NELE_HESS);
        v.fill(2.0 * obj_factor);
        1
    }
}

/// `QpecSmall`'s callbacks with the given bounds (so a test can build a
/// pair of any shape on the same rows and variables) in `index_style`
/// 0 or 1. Panics if the handle is NULL.
pub fn create(
    x_l: &[Number; N],
    x_u: &[Number; N],
    g_l: &[Number; M],
    g_u: &[Number; M],
    index_style: Index,
) -> IpoptProblem {
    let (jac, hess): (Eval_Jac_G_CB, Eval_H_CB) = if index_style == 1 {
        (ev_jac_g::<1>, ev_h::<1>)
    } else {
        (ev_jac_g::<0>, ev_h::<0>)
    };
    let prob = unsafe {
        CreateIpoptProblem(
            N as Index,
            x_l.as_ptr(),
            x_u.as_ptr(),
            M as Index,
            g_l.as_ptr(),
            g_u.as_ptr(),
            NELE_JAC as Index,
            NELE_HESS as Index,
            index_style,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(jac),
            Some(hess),
        )
    };
    assert!(!prob.is_null(), "CreateIpoptProblem returned NULL");
    prob
}

/// `QpecSmall` with its own bounds.
pub fn create_qpec(index_style: Index) -> IpoptProblem {
    create(&X_L, &X_U, &G_L, &G_U, index_style)
}
