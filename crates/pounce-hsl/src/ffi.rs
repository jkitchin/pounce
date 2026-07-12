//! Raw FFI declarations for MA57's five Fortran entry points.
//!
//! Symbols are the gfortran/double-precision exports
//! `ma57{a,b,c,e,i}d_` from `libcoinhsl.dylib` — confirmed via
//! `nm libcoinhsl.dylib | grep ma57`.
//!
//! Argument layout matches `IPOPT_DECL_MA57{A,B,C,E,I}` in
//! `ref/Ipopt/.../IpMa57TSolverInterface.hpp`. Integers are
//! `c_int` (== `pounce_common::types::Index` on this build —
//! upstream's non-`FUNNY_MA57_FINT` path).

use pounce_common::types::{Index, Number};

unsafe extern "C" {
    /// MA57AD — symbolic analysis.
    ///
    /// ```text
    /// ma57ad_(n, ne, irn, jcn, lkeep, keep, iwork, icntl, info, rinfo)
    /// ```
    pub fn ma57ad_(
        n: *const Index,
        ne: *const Index,
        irn: *const Index,
        jcn: *const Index,
        lkeep: *mut Index,
        keep: *mut Index,
        iwork: *mut Index,
        icntl: *const Index,
        info: *mut Index,
        rinfo: *mut Number,
    );

    /// MA57BD — numerical factorization.
    ///
    /// ```text
    /// ma57bd_(n, ne, a, fact, lfact, ifact, lifact, lkeep, keep,
    ///         iwork, icntl, cntl, info, rinfo)
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn ma57bd_(
        n: *const Index,
        ne: *const Index,
        a: *const Number,
        fact: *mut Number,
        lfact: *const Index,
        ifact: *mut Index,
        lifact: *const Index,
        lkeep: *const Index,
        keep: *mut Index,
        iwork: *mut Index,
        icntl: *const Index,
        cntl: *const Number,
        info: *mut Index,
        rinfo: *mut Number,
    );

    /// MA57CD — solve A * X = B (or transpose variants for `job >= 2`).
    ///
    /// ```text
    /// ma57cd_(job, n, fact, lfact, ifact, lifact, nrhs, rhs, lrhs,
    ///         work, lwork, iwork, icntl, info)
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn ma57cd_(
        job: *const Index,
        n: *const Index,
        fact: *const Number,
        lfact: *const Index,
        ifact: *const Index,
        lifact: *const Index,
        nrhs: *const Index,
        rhs: *mut Number,
        lrhs: *const Index,
        work: *mut Number,
        lwork: *const Index,
        iwork: *mut Index,
        icntl: *const Index,
        info: *mut Index,
    );

    /// MA57ED — copy/grow the `fact` (`ic=0`) or `ifact` (`ic>=1`)
    /// arrays in response to MA57BD's `info[0] = -3 / -4`.
    ///
    /// ```text
    /// ma57ed_(n, ic, keep, fact, lfact, newfac, lnew,
    ///         ifact, lifact, newifc, linew, info)
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn ma57ed_(
        n: *const Index,
        ic: *const Index,
        keep: *mut Index,
        fact: *mut Number,
        lfact: *const Index,
        newfac: *mut Number,
        lnew: *const Index,
        ifact: *mut Index,
        lifact: *const Index,
        newifc: *mut Index,
        linew: *const Index,
        info: *mut Index,
    );

    /// MA57ID — populate default `cntl` / `icntl` arrays.
    ///
    /// ```text
    /// ma57id_(cntl, icntl)
    /// ```
    pub fn ma57id_(cntl: *mut Number, icntl: *mut Index);

    /// MC19AD — symmetric-equivalent row/column scaling. R, C, W are
    /// **single-precision** even in the double-precision MC19AD entry
    /// point, matching `IPOPT_DECL_MC19A` in
    /// `IpMc19TSymScalingMethod.hpp`.
    ///
    /// ```text
    /// mc19ad_(n, nz, a, irn, icn, r, c, w)
    /// ```
    pub fn mc19ad_(
        n: *const Index,
        nz: *const Index,
        a: *mut Number,
        irn: *mut Index,
        icn: *mut Index,
        r: *mut f32,
        c: *mut f32,
        w: *mut f32,
    );

    /// OpenBLAS thread-count setter. MA57's `ma57bd_` calls dgemm
    /// internally on small dense supernodes; with the default thread
    /// count (= core count on M-series Macs) the per-call thread spin-
    /// up dominates the actual flops and slows factorization by 5-10×.
    /// libopenblas is pulled in transitively via libcoinhsl, so this
    /// symbol resolves at link time without an extra `-lopenblas`.
    pub fn openblas_set_num_threads(num: std::os::raw::c_int);
}
