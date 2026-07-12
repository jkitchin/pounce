//! Fortran 77 ABI shim — port of `Interfaces/IpStdFInterface.c`.
//!
//! Exposes the gfortran-style `ip<name>_` symbols that the upstream
//! Fortran example programs (`examples/hs071_f`) call. Each function
//! receives all its arguments as pointers (the F77 ABI), translates
//! to the C entry points in [`crate`], and translates back to the
//! Fortran-side `OKRetVal = 0 / NotOKRetVal = 1` convention.
//!
//! Trailing `_` matches gfortran / clang-flang's `F77_FUNC` mangling
//! when no underscores appear in the original name. Names with
//! embedded underscores would need `__`; none of the names exposed
//! here have any.
//!
//! Strings come in as `(char*, len_in_int)` pairs at the *end* of the
//! call (clang-flang / gfortran convention) — Fortran callers must
//! pass the lengths in the order the symbol declares. We accept the
//! length as an extra trailing `c_int` per string argument and copy
//! the buffer with trailing-space stripping ([`f2cstr`]).

use crate::{
    AddIpoptIntOption, AddIpoptNumOption, AddIpoptStrOption, CreateIpoptProblem, Eval_F_CB,
    Eval_G_CB, Eval_Grad_F_CB, Eval_H_CB, Eval_Jac_G_CB, FreeIpoptProblem, Index, Intermediate_CB,
    IpoptProblem, IpoptSolve, Number, SetIntermediateCallback,
};
use std::ffi::{c_char, c_int, c_void};

/// Fortran-side OK status (matches `IpStdFInterface.c::OKRetVal`).
const OK: Index = 0;
/// Fortran-side error status (matches `IpStdFInterface.c::NotOKRetVal`).
const NOT_OK: Index = 1;

/// Holds the Fortran user-data table (`IDAT`, `DDAT` integer/double
/// scratch buffers, plus the user's Fortran callbacks). This is what
/// `IpStdFInterface.c::FUserData` carries; we keep an opaque
/// `Box<FortranUserData>` that the C callbacks dereference.
struct FortranUserData {
    idat: *mut Index,
    ddat: *mut Number,
    eval_f: FEval_F_CB,
    eval_g: Option<FEval_G_CB>,
    eval_grad_f: FEval_Grad_F_CB,
    eval_jac_g: Option<FEval_Jac_G_CB>,
    eval_hess: Option<FEval_Hess_CB>,
    intermediate_cb: Option<FIntermediate_CB>,
    problem: IpoptProblem,
}

// Fortran callback function-pointer types. Each argument is by
// reference, matching `IpStdFInterface.c`.

pub type FEval_F_CB = unsafe extern "C" fn(
    n: *const Index,
    x: *mut Number,
    new_x: *const Index,
    obj_value: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
    ierr: *mut Index,
);

pub type FEval_G_CB = unsafe extern "C" fn(
    n: *const Index,
    x: *mut Number,
    new_x: *const Index,
    m: *const Index,
    g: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
    ierr: *mut Index,
);

pub type FEval_Grad_F_CB = unsafe extern "C" fn(
    n: *const Index,
    x: *mut Number,
    new_x: *const Index,
    grad_f: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
    ierr: *mut Index,
);

pub type FEval_Jac_G_CB = unsafe extern "C" fn(
    task: *const Index,
    n: *const Index,
    x: *mut Number,
    new_x: *const Index,
    m: *const Index,
    nnz_jac: *const Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
    ierr: *mut Index,
);

pub type FEval_Hess_CB = unsafe extern "C" fn(
    task: *const Index,
    n: *const Index,
    x: *mut Number,
    new_x: *const Index,
    obj_factor: *const Number,
    m: *const Index,
    lambda: *mut Number,
    new_lambda: *const Index,
    nnz_hess: *const Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
    ierr: *mut Index,
);

pub type FIntermediate_CB = unsafe extern "C" fn(
    alg_mode: *const Index,
    iter_count: *const Index,
    obj_value: *const Number,
    inf_pr: *const Number,
    inf_du: *const Number,
    mu: *const Number,
    d_norm: *const Number,
    regu_size: *const Number,
    alpha_du: *const Number,
    alpha_pr: *const Number,
    ls_trial: *const Index,
    idat: *mut Index,
    ddat: *mut Number,
    istop: *mut Index,
);

// ------------------------------------------------------------------
// C-side trampolines: these implement the C interface's `Eval_F_CB`
// etc. and unpack the Fortran user_data to call back into Fortran.
// ------------------------------------------------------------------

unsafe extern "C" fn c_eval_f(
    n: Index,
    x: *const Number,
    new_x: c_int,
    obj_value: *mut Number,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let mut ierr: Index = 0;
        let n_local = n;
        let new_x_i: Index = new_x as Index;
        (fud.eval_f)(
            &n_local,
            x as *mut Number,
            &new_x_i,
            obj_value,
            fud.idat,
            fud.ddat,
            &mut ierr,
        );
        if ierr == OK { 1 } else { 0 }
    }
}

unsafe extern "C" fn c_eval_grad_f(
    n: Index,
    x: *const Number,
    new_x: c_int,
    grad_f: *mut Number,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let mut ierr: Index = 0;
        let n_local = n;
        let new_x_i: Index = new_x as Index;
        (fud.eval_grad_f)(
            &n_local,
            x as *mut Number,
            &new_x_i,
            grad_f,
            fud.idat,
            fud.ddat,
            &mut ierr,
        );
        if ierr == OK { 1 } else { 0 }
    }
}

unsafe extern "C" fn c_eval_g(
    n: Index,
    x: *const Number,
    new_x: c_int,
    m: Index,
    g: *mut Number,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let Some(cb) = fud.eval_g else {
            return 0;
        };
        let mut ierr: Index = 0;
        let n_local = n;
        let m_local = m;
        let new_x_i: Index = new_x as Index;
        cb(
            &n_local,
            x as *mut Number,
            &new_x_i,
            &m_local,
            g,
            fud.idat,
            fud.ddat,
            &mut ierr,
        );
        if ierr == OK { 1 } else { 0 }
    }
}

unsafe extern "C" fn c_eval_jac_g(
    n: Index,
    x: *const Number,
    new_x: c_int,
    m: Index,
    nele_jac: Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let Some(cb) = fud.eval_jac_g else {
            return 0;
        };
        let task: Index = if !irow.is_null() && !jcol.is_null() && values.is_null() {
            0
        } else if irow.is_null() && jcol.is_null() && !values.is_null() {
            1
        } else {
            return 0;
        };
        let mut ierr: Index = 0;
        let n_local = n;
        let m_local = m;
        let nele_local = nele_jac;
        let new_x_i: Index = new_x as Index;
        cb(
            &task,
            &n_local,
            x as *mut Number,
            &new_x_i,
            &m_local,
            &nele_local,
            irow,
            jcol,
            values,
            fud.idat,
            fud.ddat,
            &mut ierr,
        );
        if ierr == OK { 1 } else { 0 }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn c_eval_h(
    n: Index,
    x: *const Number,
    new_x: c_int,
    obj_factor: Number,
    m: Index,
    lambda: *const Number,
    new_lambda: c_int,
    nele_hess: Index,
    irow: *mut Index,
    jcol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let Some(cb) = fud.eval_hess else {
            return 0;
        };
        let task: Index = if !irow.is_null() && !jcol.is_null() && values.is_null() {
            0
        } else if irow.is_null() && jcol.is_null() && !values.is_null() {
            1
        } else {
            return 0;
        };
        let mut ierr: Index = 0;
        let n_local = n;
        let m_local = m;
        let nele_local = nele_hess;
        let new_x_i: Index = new_x as Index;
        let new_lam_i: Index = new_lambda as Index;
        cb(
            &task,
            &n_local,
            x as *mut Number,
            &new_x_i,
            &obj_factor,
            &m_local,
            lambda as *mut Number,
            &new_lam_i,
            &nele_local,
            irow,
            jcol,
            values,
            fud.idat,
            fud.ddat,
            &mut ierr,
        );
        if ierr == OK { 1 } else { 0 }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn c_intermediate(
    alg_mod: Index,
    iter_count: Index,
    obj_value: Number,
    inf_pr: Number,
    inf_du: Number,
    mu: Number,
    d_norm: Number,
    regu_size: Number,
    alpha_du: Number,
    alpha_pr: Number,
    ls_trials: Index,
    user_data: *mut c_void,
) -> c_int {
    unsafe {
        let fud = &mut *(user_data as *mut FortranUserData);
        let Some(cb) = fud.intermediate_cb else {
            return 1;
        };
        let mut istop: Index = 0;
        cb(
            &alg_mod,
            &iter_count,
            &obj_value,
            &inf_pr,
            &inf_du,
            &mu,
            &d_norm,
            &regu_size,
            &alpha_du,
            &alpha_pr,
            &ls_trials,
            fud.idat,
            fud.ddat,
            &mut istop,
        );
        if istop == OK { 1 } else { 0 }
    }
}

// ------------------------------------------------------------------
// String marshalling. F77 passes (char*, len) where the buffer is
// blank-padded to `len`; we strip trailing spaces and NUL-terminate.
// ------------------------------------------------------------------

fn f2cstr(buf: *const c_char, slen: c_int) -> Vec<u8> {
    if buf.is_null() || slen <= 0 {
        return vec![0];
    }
    // SAFETY: caller asserts (buf, slen) are a valid Fortran character
    // slice of length `slen` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(buf as *const u8, slen as usize) };
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b' ' {
        end -= 1;
    }
    let mut v = Vec::with_capacity(end + 1);
    v.extend_from_slice(&bytes[..end]);
    v.push(0);
    v
}

// ------------------------------------------------------------------
// Fortran entry points (gfortran trailing-underscore names).
// ------------------------------------------------------------------

/// `ipcreate_(N, X_L, X_U, M, G_L, G_U, NELE_JAC, NELE_HESS, IDX_STY,
///            EVAL_F, EVAL_G, EVAL_GRAD_F, EVAL_JAC_G, EVAL_HESS) -> fptr`
///
/// Returns an opaque handle (a Box<FortranUserData>) cast to a
/// pointer; pass back to [`ipfree_`] / [`ipsolve_`].
///
/// # Safety
/// All pointer arguments must be valid for the lifetime of the
/// returned handle. Bound arrays must hold `*N` / `*M` doubles.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipcreate_(
    n: *const Index,
    x_l: *const Number,
    x_u: *const Number,
    m: *const Index,
    g_l: *const Number,
    g_u: *const Number,
    nele_jac: *const Index,
    nele_hess: *const Index,
    idx_sty: *const Index,
    eval_f: FEval_F_CB,
    eval_g: Option<FEval_G_CB>,
    eval_grad_f: FEval_Grad_F_CB,
    eval_jac_g: Option<FEval_Jac_G_CB>,
    eval_hess: Option<FEval_Hess_CB>,
) -> *mut c_void {
    unsafe {
        let problem = CreateIpoptProblem(
            *n,
            x_l,
            x_u,
            *m,
            g_l,
            g_u,
            *nele_jac,
            *nele_hess,
            *idx_sty,
            Some(c_eval_f),
            Some(c_eval_g),
            Some(c_eval_grad_f),
            Some(c_eval_jac_g),
            Some(c_eval_h),
        );
        if problem.is_null() {
            return std::ptr::null_mut();
        }
        let fud = Box::new(FortranUserData {
            idat: std::ptr::null_mut(),
            ddat: std::ptr::null_mut(),
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_hess,
            intermediate_cb: None,
            problem,
        });
        Box::into_raw(fud) as *mut c_void
    }
}

/// `ipfree_(FProblem)` — frees the handle and zeroes the user's
/// pointer slot, mirroring `IpStdFInterface.c::F77_FUNC(ipfree)`.
///
/// # Safety
/// `fproblem` must point to a slot that holds a handle previously
/// returned by [`ipcreate_`], or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipfree_(fproblem: *mut *mut c_void) {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return;
        }
        let raw = *fproblem as *mut FortranUserData;
        let fud = Box::from_raw(raw);
        FreeIpoptProblem(fud.problem);
        drop(fud);
        *fproblem = std::ptr::null_mut();
    }
}

/// `ipsolve_(FProblem, X, G, OBJ_VAL, MULT_G, MULT_X_L, MULT_X_U, IDAT, DDAT) -> Index`.
///
/// # Safety
/// All pointer arguments must satisfy the contracts documented on
/// [`crate::IpoptSolve`]. `idat`/`ddat` are scratch arrays passed
/// back to the user's Fortran callbacks.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipsolve_(
    fproblem: *mut *mut c_void,
    x: *mut Number,
    g: *mut Number,
    obj_val: *mut Number,
    mult_g: *mut Number,
    mult_x_l: *mut Number,
    mult_x_u: *mut Number,
    idat: *mut Index,
    ddat: *mut Number,
) -> Index {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return -199;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        fud.idat = idat;
        fud.ddat = ddat;
        let fud_ptr = (*fproblem) as *mut c_void;
        IpoptSolve(
            fud.problem,
            x,
            g,
            obj_val,
            mult_g,
            mult_x_l,
            mult_x_u,
            fud_ptr,
        )
    }
}

/// `ipaddstroption_(FProblem, KEYWORD, VALUE, klen, vlen) -> Index`.
/// Returns 0 (`OKRetVal`) on success, 1 on failure.
///
/// # Safety
/// `fproblem` must be valid; the `(KEYWORD, klen)` and `(VALUE, vlen)`
/// pairs must describe valid Fortran character slices.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipaddstroption_(
    fproblem: *mut *mut c_void,
    keyword: *const c_char,
    value: *const c_char,
    klen: c_int,
    vlen: c_int,
) -> Index {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return NOT_OK;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        let k = f2cstr(keyword, klen);
        let v = f2cstr(value, vlen);
        let ok = AddIpoptStrOption(
            fud.problem,
            k.as_ptr() as *const c_char,
            v.as_ptr() as *const c_char,
        );
        if ok != 0 { OK } else { NOT_OK }
    }
}

/// `ipaddnumoption_(FProblem, KEYWORD, VALUE, klen) -> Index`.
///
/// # Safety
/// `fproblem` must be valid; the `(KEYWORD, klen)` pair must describe
/// a valid Fortran character slice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipaddnumoption_(
    fproblem: *mut *mut c_void,
    keyword: *const c_char,
    value: *const Number,
    klen: c_int,
) -> Index {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return NOT_OK;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        let k = f2cstr(keyword, klen);
        let ok = AddIpoptNumOption(fud.problem, k.as_ptr() as *const c_char, *value);
        if ok != 0 { OK } else { NOT_OK }
    }
}

/// `ipaddintoption_(FProblem, KEYWORD, VALUE, klen) -> Index`.
///
/// # Safety
/// `fproblem` must be valid; `(KEYWORD, klen)` must describe a valid
/// Fortran character slice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipaddintoption_(
    fproblem: *mut *mut c_void,
    keyword: *const c_char,
    value: *const Index,
    klen: c_int,
) -> Index {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return NOT_OK;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        let k = f2cstr(keyword, klen);
        let ok = AddIpoptIntOption(fud.problem, k.as_ptr() as *const c_char, *value);
        if ok != 0 { OK } else { NOT_OK }
    }
}

/// `ipsetcallback_(FProblem, INTER_CB)` — install a Fortran-side
/// intermediate callback.
///
/// # Safety
/// `fproblem` must be valid; `inter_cb` must be a valid Fortran
/// callback for the lifetime of the problem.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipsetcallback_(fproblem: *mut *mut c_void, inter_cb: FIntermediate_CB) {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        fud.intermediate_cb = Some(inter_cb);
        let _: Index =
            SetIntermediateCallback(fud.problem, Some(c_intermediate as Intermediate_CB));
    }
}

/// `ipunsetcallback_(FProblem)` — remove the intermediate callback.
///
/// # Safety
/// `fproblem` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ipunsetcallback_(fproblem: *mut *mut c_void) {
    unsafe {
        if fproblem.is_null() || (*fproblem).is_null() {
            return;
        }
        let fud = &mut *(*fproblem as *mut FortranUserData);
        fud.intermediate_cb = None;
        let _: Index = SetIntermediateCallback(fud.problem, None);
    }
}

// Suppress unused import warning when the C ABI types aren't visible
// to dead-code analysis (they're referenced through public type
// aliases above).
const _: Eval_F_CB = c_eval_f;
const _: Eval_Grad_F_CB = c_eval_grad_f;
const _: Eval_G_CB = c_eval_g;
const _: Eval_Jac_G_CB = c_eval_jac_g;
const _: Eval_H_CB = c_eval_h;
const _: Intermediate_CB = c_intermediate;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f2cstr_strips_trailing_spaces() {
        let buf = b"hello    ";
        let v = f2cstr(buf.as_ptr() as *const c_char, buf.len() as c_int);
        assert_eq!(&v[..], b"hello\0");
    }

    #[test]
    fn f2cstr_handles_null_buf() {
        let v = f2cstr(std::ptr::null(), 5);
        assert_eq!(&v[..], &[0]);
    }

    #[test]
    fn f2cstr_keeps_embedded_spaces() {
        let buf = b"a b c    ";
        let v = f2cstr(buf.as_ptr() as *const c_char, buf.len() as c_int);
        assert_eq!(&v[..], b"a b c\0");
    }

    /// Drive a 1-D unconstrained quadratic through the Fortran ABI
    /// path: f(x) = (x - 3)^2.
    unsafe extern "C" fn fquad_eval_f(
        _n: *const Index,
        x: *mut Number,
        _new_x: *const Index,
        obj: *mut Number,
        _idat: *mut Index,
        _ddat: *mut Number,
        ierr: *mut Index,
    ) {
        unsafe {
            let v = *x.offset(0);
            *obj = (v - 3.0) * (v - 3.0);
            *ierr = OK;
        }
    }
    unsafe extern "C" fn fquad_eval_grad_f(
        _n: *const Index,
        x: *mut Number,
        _new_x: *const Index,
        grad: *mut Number,
        _idat: *mut Index,
        _ddat: *mut Number,
        ierr: *mut Index,
    ) {
        unsafe {
            let v = *x.offset(0);
            *grad.offset(0) = 2.0 * (v - 3.0);
            *ierr = OK;
        }
    }
    unsafe extern "C" fn fquad_eval_hess(
        task: *const Index,
        _n: *const Index,
        _x: *mut Number,
        _new_x: *const Index,
        obj_factor: *const Number,
        _m: *const Index,
        _lambda: *mut Number,
        _new_lambda: *const Index,
        _nnz_hess: *const Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        _idat: *mut Index,
        _ddat: *mut Number,
        ierr: *mut Index,
    ) {
        unsafe {
            if *task == 0 {
                *irow.offset(0) = 0;
                *jcol.offset(0) = 0;
            } else {
                *values.offset(0) = 2.0 * *obj_factor;
            }
            *ierr = OK;
        }
    }

    #[test]
    fn fortran_ipsolve_drives_quadratic() {
        let n: Index = 1;
        let m: Index = 0;
        let nele_jac: Index = 0;
        let nele_hess: Index = 1;
        let idx_sty: Index = 0;
        let xl = [-1.0e20];
        let xu = [1.0e20];

        let mut fp: *mut c_void = unsafe {
            ipcreate_(
                &n,
                xl.as_ptr(),
                xu.as_ptr(),
                &m,
                std::ptr::null(),
                std::ptr::null(),
                &nele_jac,
                &nele_hess,
                &idx_sty,
                fquad_eval_f,
                None,
                fquad_eval_grad_f,
                None,
                Some(fquad_eval_hess),
            )
        };
        assert!(!fp.is_null());

        let mut x = [0.0_f64];
        let mut obj: Number = 0.0;
        let mut idat = [0_i32; 1];
        let mut ddat = [0.0_f64; 1];
        let rc = unsafe {
            ipsolve_(
                &mut fp,
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
        assert_eq!(rc, 0); // Solve_Succeeded
        assert!((x[0] - 3.0).abs() < 1e-6, "x[0] = {}", x[0]);
        unsafe { ipfree_(&mut fp) };
        assert!(fp.is_null());
    }
}
