//! CUTEst C API FFI declarations (double precision).
//!
//! Functions with `_c` suffix or `cint_` prefix use 0-based indexing
//! and C-compatible types (bool for logical).

#![allow(dead_code)]

use std::os::raw::c_char;

extern "C" {
    // ---- Dynamic library loading (from cutest_trampoline.f90) ----

    #[link_name = "cutest_load_routines_"]
    pub fn cutest_load_routines(libname: *const c_char);

    #[link_name = "cutest_unload_routines_"]
    pub fn cutest_unload_routines();

    // ---- Fortran I/O ----

    #[link_name = "fortran_open_fixed_"]
    pub fn fortran_open(funit: *const i32, fname: *const c_char, ierr: *mut i32);

    #[link_name = "fortran_close_"]
    pub fn fortran_close(funit: *const i32, ierr: *mut i32);

    // ---- Dimensions ----

    /// Get n and m for any problem (returns m=0 for unconstrained).
    #[link_name = "cutest_cdimen_"]
    pub fn cutest_cdimen(status: *mut i32, funit: *const i32, n: *mut i32, m: *mut i32);

    /// Hessian nnz (constrained).
    #[link_name = "cutest_cdimsh_"]
    pub fn cutest_cdimsh(status: *mut i32, nnzh: *mut i32);

    /// Hessian nnz (unconstrained).
    #[link_name = "cutest_udimsh_"]
    pub fn cutest_udimsh(status: *mut i32, nnzh: *mut i32);

    /// Jacobian nnz upper bound (may include objective gradient entries).
    #[link_name = "cutest_cdimsj_"]
    pub fn cutest_cdimsj(status: *mut i32, nnzj: *mut i32);

    // ---- Setup ----

    /// Constrained setup (0-based indexing via cint wrapper).
    #[link_name = "cutest_cint_csetup_"]
    pub fn cutest_csetup(
        status: *mut i32,
        funit: *const i32,
        iout: *const i32,
        io_buffer: *const i32,
        n: *mut i32,
        m: *mut i32,
        x: *mut f64,
        x_l: *mut f64,
        x_u: *mut f64,
        y: *mut f64,
        c_l: *mut f64,
        c_u: *mut f64,
        equatn: *mut bool,
        linear: *mut bool,
        e_order: *const i32,
        l_order: *const i32,
        v_order: *const i32,
    );

    /// Unconstrained setup.
    #[link_name = "cutest_usetup_"]
    pub fn cutest_usetup(
        status: *mut i32,
        funit: *const i32,
        iout: *const i32,
        io_buffer: *const i32,
        n: *mut i32,
        x: *mut f64,
        x_l: *mut f64,
        x_u: *mut f64,
    );

    // ---- Objective and gradient ----

    /// Constrained: objective f(x) and optionally gradient.
    #[link_name = "cutest_cint_cofg_"]
    pub fn cutest_cofg(
        status: *mut i32,
        n: *const i32,
        x: *const f64,
        f: *mut f64,
        g: *mut f64,
        grad: *const bool,
    );

    /// Unconstrained: objective f(x) and optionally gradient.
    #[link_name = "cutest_cint_uofg_"]
    pub fn cutest_uofg(
        status: *mut i32,
        n: *const i32,
        x: *const f64,
        f: *mut f64,
        g: *mut f64,
        grad: *const bool,
    );

    // ---- Constraints ----

    /// Evaluate constraint values c(x) only.
    #[link_name = "cutest_ccf_"]
    pub fn cutest_ccf(
        status: *mut i32,
        n: *const i32,
        m: *const i32,
        x: *const f64,
        c: *mut f64,
    );

    /// Constraints + sparse Jacobian (0-based indices).
    #[link_name = "cutest_ccfsg_c_"]
    pub fn cutest_ccfsg(
        status: *mut i32,
        n: *const i32,
        m: *const i32,
        x: *const f64,
        c: *mut f64,
        nnzj: *mut i32,
        lcjac: *const i32,
        cjac: *mut f64,
        indvar: *mut i32,
        indfun: *mut i32,
        grad: *const bool,
    );

    /// Jacobian sparsity pattern (0-based indices).
    #[link_name = "cutest_csjp_c_"]
    pub fn cutest_csjp(
        status: *mut i32,
        nnzj: *mut i32,
        lj: *const i32,
        jvar: *mut i32,
        jcon: *mut i32,
    );

    // ---- Hessian (constrained) ----

    /// Hessian sparsity pattern, lower triangle (0-based indices).
    #[link_name = "cutest_cshp_c_"]
    pub fn cutest_cshp(
        status: *mut i32,
        n: *const i32,
        nnzh: *mut i32,
        lh: *const i32,
        irnh: *mut i32,
        icnh: *mut i32,
    );

    /// John function Hessian: y0*H_f + sum(y_i*H_ci), sparse, lower triangle (0-based).
    #[link_name = "cutest_cshj_c_"]
    pub fn cutest_cshj(
        status: *mut i32,
        n: *const i32,
        m: *const i32,
        x: *const f64,
        y0: *const f64,
        y: *const f64,
        nnzh: *mut i32,
        lh: *const i32,
        h: *mut f64,
        irnh: *mut i32,
        icnh: *mut i32,
    );

    // ---- Hessian (unconstrained) ----

    /// Hessian sparsity pattern, lower triangle (0-based indices).
    #[link_name = "cutest_ushp_c_"]
    pub fn cutest_ushp(
        status: *mut i32,
        n: *const i32,
        nnzh: *mut i32,
        lh: *const i32,
        irnh: *mut i32,
        icnh: *mut i32,
    );

    /// Sparse Hessian, lower triangle (0-based indices).
    #[link_name = "cutest_ush_c_"]
    pub fn cutest_ush(
        status: *mut i32,
        n: *const i32,
        x: *const f64,
        nnzh: *mut i32,
        lh: *const i32,
        h: *mut f64,
        irnh: *mut i32,
        icnh: *mut i32,
    );

    // ---- Terminate ----

    #[link_name = "cutest_cterminate_"]
    pub fn cutest_cterminate(status: *mut i32);

    #[link_name = "cutest_uterminate_"]
    pub fn cutest_uterminate(status: *mut i32);
}
