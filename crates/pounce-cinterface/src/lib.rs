//! POUNCE C ABI — port of `Interfaces/IpStdCInterface.{h,cpp}`.
//!
//! Provides the `CreateIpoptProblem / IpoptSolve / FreeIpoptProblem` C
//! entry points that existing PyIpopt / cyipopt / JuMP wrappers link
//! against. Function names and signatures match upstream Ipopt 3.14.x
//! exactly so consumers can swap `libipopt.{dylib,so}` for
//! `libpounce_cinterface` without rebuilding.
//!
//! Surface area (in `IpStdCInterface.h` order):
//!
//! * Lifecycle: [`CreateIpoptProblem`], [`FreeIpoptProblem`].
//! * Options: [`AddIpoptStrOption`], [`AddIpoptNumOption`],
//!   [`AddIpoptIntOption`], [`OpenIpoptOutputFile`],
//!   [`SetIpoptProblemScaling`].
//! * Callbacks: [`SetIntermediateCallback`].
//! * Solve: [`IpoptSolve`].
//! * Introspection (only valid inside an intermediate callback):
//!   [`GetIpoptCurrentIterate`], [`GetIpoptCurrentViolations`].
//! * Library info: [`GetIpoptVersion`].
//!
//! Pounce extensions for post-solve stats (not present in upstream
//! Ipopt's C API): [`GetIpoptIterCount`], [`GetIpoptSolveTime`],
//! [`GetIpoptPrimalInf`], [`GetIpoptDualInf`], [`GetIpoptComplInf`].
//!
//! All entry points are `extern "C"` and `#[no_mangle]`. Pointers are
//! raw and the caller is responsible for lifetime; the `IpoptProblem`
//! handle is opaque (`*mut c_void` from C's perspective). The Fortran
//! 77 ABI shim lives in [`fortran`].

#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn, dead_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod fortran;
pub mod solver;

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::intermediate as ip_intermediate;
use pounce_common::reg_options::OptionType;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_restoration::install::install_default_restoration;
use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::rc::Rc;

/// Mirrors C `Number` typedef in `IpStdCInterface.h`.
pub type Number = f64;
/// Mirrors C `Index`.
pub type Index = c_int;
/// Mirrors C `Bool`, which `pounce.h` — like Ipopt 3.14's
/// `IpStdCInterface.h` — declares as the C99 `bool`. That is **one byte**,
/// so this is `u8` and not `c_int`.
///
/// It was `c_int` until gh#624, i.e. four bytes on the Rust side against
/// one on every C caller's, in both directions:
///
/// * a callback returning `false` sets only `AL`, and the x86-64 psABI
///   leaves the rest of `EAX` unspecified — read as an `i32`, a failed
///   evaluation could come back nonzero, which reads as *success*. The
///   solver would then accept a point it was told it could not evaluate
///   instead of cutting the step. gcc and clang emit `movzbl`, which is
///   why this stayed latent rather than exploding;
/// * an *array* of them would have been a hard stride bug, 1-byte
///   elements read at 4-byte spacing. That is why
///   [`IpoptSetNonlinearVariables`] takes a count plus an index list
///   rather than the `Bool` mask gh#624 originally proposed.
///
/// `u8` rather than Rust's `bool` on purpose: the two have identical
/// layout, but `bool` carries a validity invariant (it *must* hold 0 or
/// 1), and a C caller reaching this boundary with anything else — an
/// older header where `Bool` was `int`, a hand-rolled binding, a value
/// that came through a `memcpy` — would be instant undefined behaviour.
/// `u8` accepts whatever arrives and tests it the way C does, which is
/// the same reason the entry points validate rather than trust.
pub type Bool = u8;

const TRUE: Bool = 1;
const FALSE: Bool = 0;

// The whole point of the alias. `pounce.h` says `typedef bool Bool`, so a
// build where this stops being one byte is one where every C caller
// disagrees with the implementation about every boolean.
const _: () = assert!(
    core::mem::size_of::<Bool>() == 1,
    "Bool must be one byte to match `typedef bool Bool` in pounce.h"
);

/// Run an FFI entry-point body, converting any Rust panic into `fallback`
/// rather than letting it unwind across the `extern "C"` boundary — which is
/// undefined behavior and, in practice, a process abort that takes the
/// embedding application down with it. Upstream Ipopt's C interface likewise
/// wraps the solve in `try { … } catch(…)` and reports `Internal_Error`
/// instead of propagating a C++ exception across the ABI.
///
/// Note: this guards panics that originate in *pounce's own* Rust code (the
/// solver core, the callback bridge, numerical kernels). A panic inside a
/// user-supplied `extern "C"` callback aborts at that callback's own ABI
/// boundary, before unwinding can reach here — that is the caller's
/// responsibility, exactly as in the C/C++ original.
pub(crate) fn ffi_guard<R>(fallback: R, body: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => r,
        Err(_) => fallback,
    }
}

/// C-ABI encoding of [`pounce_qp::BoundStatus`] (§7.2 of the
/// active-set-SQP design note). Stable values:
/// `0 = Inactive`, `1 = AtLower`, `2 = AtUpper`, `3 = Fixed`.
pub type IpoptBoundStatus = c_int;
/// C-ABI encoding of [`pounce_qp::ConsStatus`] (§7.2). Stable values:
/// `0 = Inactive`, `1 = AtLower`, `2 = AtUpper`, `3 = Equality`.
pub type IpoptConsStatus = c_int;

const POUNCE_WS_INACTIVE: c_int = 0;
const POUNCE_WS_AT_LOWER: c_int = 1;
const POUNCE_WS_AT_UPPER: c_int = 2;
const POUNCE_WS_FIXED_OR_EQ: c_int = 3;

/// Internal owned state behind the opaque `IpoptProblem` handle.
/// `#[repr(C)]` is unnecessary because C only sees the pointer.
pub struct IpoptProblemInfo {
    pub(crate) app: IpoptApplication,
    pub(crate) n: Index,
    pub(crate) m: Index,
    pub(crate) nele_jac: Index,
    pub(crate) nele_hess: Index,
    pub(crate) index_style: Index,
    pub(crate) x_l: Vec<Number>,
    pub(crate) x_u: Vec<Number>,
    pub(crate) g_l: Vec<Number>,
    pub(crate) g_u: Vec<Number>,
    pub(crate) eval_f: Option<Eval_F_CB>,
    pub(crate) eval_g: Option<Eval_G_CB>,
    pub(crate) eval_grad_f: Option<Eval_Grad_F_CB>,
    pub(crate) eval_jac_g: Option<Eval_Jac_G_CB>,
    pub(crate) eval_h: Option<Eval_H_CB>,
    pub(crate) intermediate_cb: Option<Intermediate_CB>,
    /// User-provided scaling installed by [`SetIpoptProblemScaling`].
    /// `obj_scaling` defaults to `1.0`. `x_scaling`/`g_scaling` are
    /// `None` when the user passed NULL.
    pub(crate) user_scaling: Option<UserScaling>,
    /// Final iterate and stats from the most recent [`IpoptSolve`].
    /// Used by `GetIpopt{IterCount,SolveTime,...}` accessors. Reset
    /// (cleared) by the next `IpoptSolve` call.
    pub(crate) last_solve: Option<LastSolve>,
    /// Working set staged by [`IpoptSetWarmStartWorkingSet`], pending
    /// until the next [`IpoptSolve`].
    ///
    /// Only the working set is stored here — deliberately *not* a full
    /// `SqpIterates`. The primal/dual iterate is not knowable at set
    /// time; it arrives with the `x` (and, under
    /// `warm_start_init_point=yes`, `mult_g`/`mult_x_L`/`mult_x_U`)
    /// buffers passed to `IpoptSolve`. Building the `SqpIterates`
    /// eagerly here forced the primal to a placeholder — zeros — which
    /// then *became* the starting iterate, because
    /// `SqpAlgorithm::optimize_with_warm_start` uses a supplied warm
    /// iterate instead of querying the NLP for a starting point. The
    /// merge therefore has to happen inside `IpoptSolve` (gh#484).
    pub(crate) pending_working_set: Option<pounce_qp::WorkingSet>,
    /// Nonlinear-variable subset staged by
    /// [`IpoptSetNonlinearVariables`] (gh#624). Stored in the problem's
    /// own index style, exactly as the caller passed it; the bridge
    /// TNLP serves it back through
    /// `get_number_of_nonlinear_variables` /
    /// `get_list_of_nonlinear_variables`, which is where the algorithm
    /// reads it. `None` — the default — means "every variable is
    /// nonlinear".
    pub(crate) nonlinear_vars: Option<Vec<Index>>,
}

/// User-provided NLP scaling stored on the problem until
/// [`IpoptSolve`] copies it into the [`CCallbackTnlp`] bridge.
#[derive(Clone)]
pub(crate) struct UserScaling {
    obj_scaling: Number,
    x_scaling: Option<Vec<Number>>,
    g_scaling: Option<Vec<Number>>,
}

/// Stats and final-iterate snapshot retained between
/// [`IpoptSolve`] and the post-solve accessors. Everything needed to
/// reconstruct a `pounce.solve-report/v1` JSON file lives here so
/// [`IpoptWriteSolveReport`] doesn't have to ask the caller to thread
/// `x`/`lambda`/`obj` back in.
#[derive(Clone)]
pub(crate) struct LastSolve {
    pub(crate) stats: SolveStatistics,
    pub(crate) status: ApplicationReturnStatus,
    pub(crate) linear_solver: Option<pounce_linsol::summary::LinearSolverSummary>,
    pub(crate) final_x: Vec<Number>,
    pub(crate) final_lambda: Vec<Number>,
    pub(crate) final_obj: Number,
}

impl Default for LastSolve {
    fn default() -> Self {
        Self {
            stats: SolveStatistics::default(),
            status: ApplicationReturnStatus::InternalError,
            linear_solver: None,
            final_x: Vec::new(),
            final_lambda: Vec::new(),
            final_obj: 0.0,
        }
    }
}

pub type IpoptProblem = *mut IpoptProblemInfo;

// User-callback function pointer types — match
// `IpStdCInterface.h:Eval_F_CB` etc. byte for byte.

pub type Eval_F_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_value: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_Grad_F_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    grad_f: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_G_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    g: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_Jac_G_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    m: Index,
    nele_jac: Index,
    iRow: *mut Index,
    jCol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Eval_H_CB = unsafe extern "C" fn(
    n: Index,
    x: *const Number,
    new_x: Bool,
    obj_factor: Number,
    m: Index,
    lambda: *const Number,
    new_lambda: Bool,
    nele_hess: Index,
    iRow: *mut Index,
    jCol: *mut Index,
    values: *mut Number,
    user_data: *mut c_void,
) -> Bool;

pub type Intermediate_CB = unsafe extern "C" fn(
    alg_mod: Index,
    iter_count: Index,
    obj_value: Number,
    inf_pr: Number,
    inf_du: Number,
    mu: Number,
    d_norm: Number,
    regularization_size: Number,
    alpha_du: Number,
    alpha_pr: Number,
    ls_trials: Index,
    user_data: *mut c_void,
) -> Bool;

/// Port of `IpStdCInterface.cpp:CreateIpoptProblem`. Returns NULL on
/// invalid arguments (negative n/m, missing required callbacks, NULL
/// bound pointers when the corresponding dimension is positive).
///
/// # Safety
///
/// `x_L`, `x_U` must be valid pointers to `n` `Number`s when `n > 0`.
/// `g_L`, `g_U` must be valid pointers to `m` `Number`s when `m > 0`.
/// The callback function pointers must be valid for the lifetime of
/// the returned [`IpoptProblem`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn CreateIpoptProblem(
    n: Index,
    x_L: *const Number,
    x_U: *const Number,
    m: Index,
    g_L: *const Number,
    g_U: *const Number,
    nele_jac: Index,
    nele_hess: Index,
    index_style: Index,
    eval_f: Option<Eval_F_CB>,
    eval_g: Option<Eval_G_CB>,
    eval_grad_f: Option<Eval_Grad_F_CB>,
    eval_jac_g: Option<Eval_Jac_G_CB>,
    eval_h: Option<Eval_H_CB>,
) -> IpoptProblem {
    unsafe {
        // Install the tracing subscriber on first use so C consumers
        // (cyipopt, AMPL, …) get logging and the iteration collector that
        // backs `IpoptEnableIterHistory` (pounce#71). Idempotent.
        pounce_observability::init_subscriber();

        if n < 0 || m < 0 || nele_jac < 0 || nele_hess < 0 {
            return std::ptr::null_mut();
        }
        if !(0..=1).contains(&index_style) {
            return std::ptr::null_mut();
        }
        if eval_f.is_none() || eval_grad_f.is_none() {
            return std::ptr::null_mut();
        }
        if m > 0 && (eval_g.is_none() || eval_jac_g.is_none()) {
            return std::ptr::null_mut();
        }
        if n > 0 && (x_L.is_null() || x_U.is_null()) {
            return std::ptr::null_mut();
        }
        if m > 0 && (g_L.is_null() || g_U.is_null()) {
            return std::ptr::null_mut();
        }

        let x_l = if n > 0 {
            std::slice::from_raw_parts(x_L, n as usize).to_vec()
        } else {
            Vec::new()
        };
        let x_u = if n > 0 {
            std::slice::from_raw_parts(x_U, n as usize).to_vec()
        } else {
            Vec::new()
        };
        let g_l_vec = if m > 0 {
            std::slice::from_raw_parts(g_L, m as usize).to_vec()
        } else {
            Vec::new()
        };
        let g_u_vec = if m > 0 {
            std::slice::from_raw_parts(g_U, m as usize).to_vec()
        } else {
            Vec::new()
        };

        let info = Box::new(IpoptProblemInfo {
            app: IpoptApplication::new(),
            n,
            m,
            nele_jac,
            nele_hess,
            index_style,
            x_l,
            x_u,
            g_l: g_l_vec,
            g_u: g_u_vec,
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_h,
            intermediate_cb: None,
            user_scaling: None,
            nonlinear_vars: None,
            last_solve: None,
            pending_working_set: None,
        });
        Box::into_raw(info)
    }
}

/// Port of `IpStdCInterface.cpp:FreeIpoptProblem`.
///
/// # Safety
///
/// `ipopt_problem` must be a pointer previously returned by
/// [`CreateIpoptProblem`] and not yet freed, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn FreeIpoptProblem(ipopt_problem: IpoptProblem) {
    unsafe {
        if ipopt_problem.is_null() {
            return;
        }
        drop(Box::from_raw(ipopt_problem));
    }
}

unsafe fn keyword_str<'a>(keyword: *const c_char) -> Option<&'a str> {
    unsafe {
        if keyword.is_null() {
            return None;
        }
        CStr::from_ptr(keyword).to_str().ok()
    }
}

/// Port of `IpStdCInterface.cpp:AddIpoptStrOption`.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. `keyword` and `val`
/// must be valid NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn AddIpoptStrOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: *const c_char,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let Some(k) = keyword_str(keyword) else {
            return FALSE;
        };
        if val.is_null() {
            return FALSE;
        }
        let Ok(v) = CStr::from_ptr(val).to_str() else {
            return FALSE;
        };
        match info.app.options_mut().set_string_value(k, v, true, false) {
            Ok(_) => TRUE,
            Err(_) => FALSE,
        }
    }
}

/// Port of `AddIpoptNumOption`.
///
/// # Safety
///
/// `keyword` must be a valid NUL-terminated string and
/// `ipopt_problem` must be a valid `IpoptProblem`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn AddIpoptNumOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: Number,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let Some(k) = keyword_str(keyword) else {
            return FALSE;
        };
        match info
            .app
            .options_mut()
            .set_numeric_value(k, val, true, false)
        {
            Ok(_) => TRUE,
            Err(_) => FALSE,
        }
    }
}

/// Port of `AddIpoptIntOption`.
///
/// # Safety
///
/// `keyword` must be a valid NUL-terminated string and
/// `ipopt_problem` must be a valid `IpoptProblem`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn AddIpoptIntOption(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
    val: Index,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let Some(k) = keyword_str(keyword) else {
            return FALSE;
        };
        match info.app.options_mut().set_integer_value(
            k,
            val as pounce_common::types::Index,
            true,
            false,
        ) {
            Ok(_) => TRUE,
            Err(_) => FALSE,
        }
    }
}

/// Port of `IpStdCInterface.cpp:OpenIpoptOutputFile`. Opens `file_name`
/// at `print_level` and attaches a journalist `FileJournal` so all
/// solver output is mirrored to disk. Equivalent to setting
/// `output_file` + `file_print_level` options and triggering
/// `IpoptApplication::Initialize`.
///
/// Returns `TRUE` (1) on success, `FALSE` (0) if the file could not
/// be opened or the option store rejected the value.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. `file_name` must
/// be a valid NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn OpenIpoptOutputFile(
    ipopt_problem: IpoptProblem,
    file_name: *const c_char,
    print_level: c_int,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() || file_name.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let Ok(fname) = CStr::from_ptr(file_name).to_str() else {
            return FALSE;
        };
        if info.app.open_output_file(fname, print_level) {
            TRUE
        } else {
            FALSE
        }
    }
}

/// Port of `IpStdCInterface.cpp:SetIpoptProblemScaling`. Stores
/// user-provided NLP scaling on the problem; the scaling is forwarded
/// to the solver via [`TNLP::get_scaling_parameters`] when the option
/// `nlp_scaling_method=user-scaling` is set. Passing NULL for
/// `x_scaling` / `g_scaling` disables scaling on that axis.
///
/// Always returns `TRUE` (the upstream contract). A non-trivial
/// `x_scaling` is applied as a change of variables, so the solution and
/// bound multipliers `IpoptSolve` writes back are in the caller's own
/// units (gh#486). A factor that is not finite and positive is refused
/// at solve time, where [`IpoptSolve`] returns `Invalid_Option` and the
/// journalist explains why: store-time validation is not an option,
/// because the C signature has no way to report it.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. When non-NULL,
/// `x_scaling` must point to `n` doubles and `g_scaling` to `m`
/// doubles; both arrays are copied internally.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetIpoptProblemScaling(
    ipopt_problem: IpoptProblem,
    obj_scaling: Number,
    x_scaling: *const Number,
    g_scaling: *const Number,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let n = info.n as usize;
        let m = info.m as usize;
        let x_vec = if !x_scaling.is_null() && n > 0 {
            Some(std::slice::from_raw_parts(x_scaling, n).to_vec())
        } else {
            None
        };
        let g_vec = if !g_scaling.is_null() && m > 0 {
            Some(std::slice::from_raw_parts(g_scaling, m).to_vec())
        } else {
            None
        };
        info.user_scaling = Some(UserScaling {
            obj_scaling,
            x_scaling: x_vec,
            g_scaling: g_vec,
        });
        TRUE
    }
}

/// Port of `IpStdCInterface.cpp:IpoptSolve`. Returns the
/// `ApplicationReturnStatus` integer.
///
/// Builds a [`CCallbackTnlp`] from the user-supplied callback table
/// and bounds, runs it through [`IpoptApplication::optimize_tnlp`],
/// and writes back the final iterate.
///
/// # Safety
///
/// All pointer arguments are read/written per the
/// `IpStdCInterface.h` contract: `x` is in/out (size `n`); `g`,
/// `mult_g`, `mult_x_L`, `mult_x_U` are out-only (sizes `m, m, n, n`)
/// and may be NULL when the corresponding output is not desired.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptSolve(
    ipopt_problem: IpoptProblem,
    x: *mut Number,
    g: *mut Number,
    obj_val: *mut Number,
    mult_g: *mut Number,
    mult_x_L: *mut Number,
    mult_x_U: *mut Number,
    user_data: *mut c_void,
) -> Index {
    unsafe {
        if ipopt_problem.is_null() {
            return ApplicationReturnStatus::InternalError as Index;
        }
        // Invalidate the retained stats up front, before the solve is attempted.
        // The `last_solve` snapshot is only repopulated at the *end* of a
        // completed solve, so if the guarded body below bails early or a panic is
        // caught (returning `Internal_Error`), the post-solve accessors
        // (`GetIpoptIterCount`, `IpoptWriteSolveReport`, …) must not silently
        // report the *previous* solve's stats. Clearing here makes the
        // failure-consistent state "no data" rather than stale data (F5).
        (*ipopt_problem).last_solve = None;
        // Guard the whole solve: `optimize_tnlp` runs the entire pounce core and
        // callback bridge, any of which could panic on an unexpected internal
        // state. Without this, such a panic would unwind across `extern "C"` and
        // abort the embedding process; instead we report `Internal_Error`,
        // matching upstream Ipopt's exception handling. (See `ffi_guard`.)
        ffi_guard(ApplicationReturnStatus::InternalError as Index, || {
            let info = &mut *ipopt_problem;
            if info.n < 0 || info.m < 0 {
                return ApplicationReturnStatus::InvalidProblemDefinition as Index;
            }
            if info.n > 0 && x.is_null() {
                return ApplicationReturnStatus::InvalidProblemDefinition as Index;
            }

            let n_us = info.n as usize;
            let m_us = info.m as usize;
            let initial_x = if n_us > 0 {
                std::slice::from_raw_parts(x, n_us).to_vec()
            } else {
                Vec::new()
            };

            // Merge any working set staged by
            // `IpoptSetWarmStartWorkingSet` with the iterate the caller
            // actually supplied. This is the point at which the primal
            // starting point is known, so it is the only correct place
            // to build the `SqpIterates` (gh#484).
            //
            // Duals follow upstream Ipopt's `IpoptSolve` contract:
            // `mult_g` / `mult_x_L` / `mult_x_U` are inputs only when
            // `warm_start_init_point=yes`, and out-only otherwise. A
            // caller who has not opted in may pass uninitialized
            // buffers, so reading them unconditionally would seed the
            // SQP with garbage multipliers.
            if let Some(working) = info.pending_working_set.take() {
                let seed_duals = matches!(
                    info.app
                        .options()
                        .get_bool_value("warm_start_init_point", ""),
                    Ok((true, true))
                );
                let read_in = |p: *const Number, len: usize| -> Vec<Number> {
                    if seed_duals && !p.is_null() && len > 0 {
                        std::slice::from_raw_parts(p, len).to_vec()
                    } else {
                        vec![0.0; len]
                    }
                };
                let lambda_g = read_in(mult_g as *const Number, m_us);
                let z_l = read_in(mult_x_L as *const Number, n_us);
                let z_u = read_in(mult_x_U as *const Number, n_us);
                // SQP packs the bound multipliers signed, as
                // `lambda_x = z_l − z_u` (see `sqp::warm_start`).
                let lambda_x = z_l.iter().zip(&z_u).map(|(l, u)| l - u).collect();
                info.app
                    .set_sqp_warm_start(pounce_algorithm::sqp::SqpIterates {
                        x: initial_x.clone(),
                        lambda_g,
                        lambda_x,
                        working: Some(working),
                    });
            }

            let bridge = Rc::new(RefCell::new(CCallbackTnlp {
                n: info.n,
                m: info.m,
                nele_jac: info.nele_jac,
                nele_hess: info.nele_hess,
                index_style: info.index_style,
                x_l: info.x_l.clone(),
                x_u: info.x_u.clone(),
                g_l: info.g_l.clone(),
                g_u: info.g_u.clone(),
                initial_x,
                eval_f: info.eval_f,
                eval_grad_f: info.eval_grad_f,
                eval_g: info.eval_g,
                eval_jac_g: info.eval_jac_g,
                eval_h: info.eval_h,
                user_data,
                intermediate_cb: info.intermediate_cb,
                user_scaling: info.user_scaling.clone(),
                nonlinear_vars: info.nonlinear_vars.clone(),
                final_status: None,
                final_x: vec![0.0; n_us],
                final_z_l: vec![0.0; n_us],
                final_z_u: vec![0.0; n_us],
                final_g: vec![0.0; m_us],
                final_lambda: vec![0.0; m_us],
                final_obj: 0.0,
            }));

            // Wire the restoration phase fresh for this solve. Without it, any
            // line-search failure surfaces as `RestorationFailure` instead of
            // falling back into the ℓ1-feasibility sub-IPM — exactly what the
            // CLI driver does. Re-wire per `IpoptSolve` to stay correct across
            // repeated solves on the same `IpoptProblem`. The feral config is
            // snapshot from the now-fully-populated options so `feral_*`
            // overrides flow into the restoration sub-IPM too. The helper
            // installs the multi-pass provider (so the ℓ₁ wrapper /
            // auto-fallback / second-opinion ladder don't panic on the second
            // inner solve — pounce#10 Phase 3 / pounce#24) and the mint.
            install_default_restoration(&mut info.app);

            let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();
            let status = info.app.optimize_tnlp(bridge_for_solve);
            let bridge_ref = bridge.borrow();
            info.last_solve = Some(LastSolve {
                stats: info.app.statistics(),
                status,
                linear_solver: info.app.linear_solver_summary(),
                final_x: bridge_ref.final_x.clone(),
                final_lambda: bridge_ref.final_lambda.clone(),
                final_obj: bridge_ref.final_obj,
            });
            if !x.is_null() && n_us > 0 {
                std::ptr::copy_nonoverlapping(bridge_ref.final_x.as_ptr(), x, n_us);
            }
            if !g.is_null() && m_us > 0 {
                std::ptr::copy_nonoverlapping(bridge_ref.final_g.as_ptr(), g, m_us);
            }
            if !obj_val.is_null() {
                *obj_val = bridge_ref.final_obj;
            }
            if !mult_g.is_null() && m_us > 0 {
                std::ptr::copy_nonoverlapping(bridge_ref.final_lambda.as_ptr(), mult_g, m_us);
            }
            if !mult_x_L.is_null() && n_us > 0 {
                std::ptr::copy_nonoverlapping(bridge_ref.final_z_l.as_ptr(), mult_x_L, n_us);
            }
            if !mult_x_U.is_null() && n_us > 0 {
                std::ptr::copy_nonoverlapping(bridge_ref.final_z_u.as_ptr(), mult_x_U, n_us);
            }
            status as Index
        })
    }
}

/// Port of `SetIntermediateCallback`.
///
/// # Safety
///
/// `ipopt_problem` must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetIntermediateCallback(
    ipopt_problem: IpoptProblem,
    intermediate_cb: Option<Intermediate_CB>,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        info.intermediate_cb = intermediate_cb;
        TRUE
    }
}

/// Port of `IpStdCInterface.cpp:GetIpoptCurrentIterate` (Ipopt 3.14+).
/// Designed to be called from inside an intermediate callback to
/// inspect `x`, the bound multipliers `z_L/z_U`, the constraint values
/// `g`, and the constraint multipliers `lambda` at the current
/// iterate.
///
/// All output buffers are optional — pass NULL to skip. `n` and `m`
/// must match the dimensions the problem was created with; mismatched
/// sizes cause the function to return `FALSE` without writing.
///
/// `scaled` is currently ignored — quantities are reported in the
/// user TNLP's unscaled space (matching upstream Ipopt's default
/// caller behavior when scaling is unused). Honoring `scaled` for the
/// `gradient-based` scaler is a follow-up.
///
/// Returns `FALSE` when called outside an active intermediate
/// callback (no live iterate to inspect).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. Each output buffer,
/// when non-NULL, must hold at least the declared length.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptCurrentIterate(
    ipopt_problem: IpoptProblem,
    _scaled: Bool,
    n: Index,
    x: *mut Number,
    z_l: *mut Number,
    z_u: *mut Number,
    m: Index,
    g: *mut Number,
    lambda: *mut Number,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        if n != info.n || m != info.m {
            return FALSE;
        }
        let result = ip_intermediate::with_current(|ctx| {
            // Snapshot the iterate handles and release the `data` borrow
            // before touching `cq`: several `IpoptCq` accessors
            // (`curr_c`, `curr_d`, …) evaluate through the NLP and take
            // `nlp.borrow_mut()` internally, so no `nlp`/`data` borrow may
            // be alive across them. Holding one here panicked
            // ("RefCell already borrowed") on every `g != NULL` call, and
            // a panic across this `extern "C"` boundary aborts the
            // process rather than returning `FALSE`.
            let curr = {
                let data = ctx.data.borrow();
                match data.curr.as_ref() {
                    Some(curr) => curr.clone(),
                    None => return false,
                }
            };
            let n_us = n as usize;
            let m_us = m as usize;
            if !x.is_null() && n_us > 0 {
                let full_x = ctx.nlp.borrow().lift_x_to_full(&*curr.x);
                if full_x.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full_x.as_ptr(), x, n_us);
            }
            if !z_l.is_null() && n_us > 0 {
                let full = ctx.nlp.borrow().pack_z_l_for_user(&*curr.z_l);
                if full.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full.as_ptr(), z_l, n_us);
            }
            if !z_u.is_null() && n_us > 0 {
                let full = ctx.nlp.borrow().pack_z_u_for_user(&*curr.z_u);
                if full.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full.as_ptr(), z_u, n_us);
            }
            if !g.is_null() && m_us > 0 {
                // `curr_c` / `curr_d` re-enter the NLP mutably: evaluate
                // them first, *then* borrow `nlp` to pack the result.
                let (c, d) = {
                    let cq = ctx.cq.borrow();
                    (cq.curr_c(), cq.curr_d())
                };
                let full = ctx.nlp.borrow().pack_g_for_user(&*c, &*d);
                if full.len() != m_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full.as_ptr(), g, m_us);
            }
            if !lambda.is_null() && m_us > 0 {
                let full = ctx
                    .nlp
                    .borrow()
                    .pack_lambda_for_user(&*curr.y_c, &*curr.y_d);
                if full.len() != m_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full.as_ptr(), lambda, m_us);
            }
            true
        });
        if result.unwrap_or(false) { TRUE } else { FALSE }
    }
}

/// Port of `IpStdCInterface.cpp:GetIpoptCurrentViolations` (Ipopt 3.14+).
/// Same contract as [`GetIpoptCurrentIterate`]; returns `FALSE` when
/// called outside an active intermediate callback.
///
/// `scaled` is currently ignored — see [`GetIpoptCurrentIterate`].
/// Violations and complementarities are reported in the compressed
/// algorithm-side space scattered out to full-`n`/`m`; this is the
/// shape upstream callers consume (zero-fill for free positions /
/// no-bound positions).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. Each output buffer,
/// when non-NULL, must hold at least the declared length.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptCurrentViolations(
    ipopt_problem: IpoptProblem,
    _scaled: Bool,
    n: Index,
    x_l_violation: *mut Number,
    x_u_violation: *mut Number,
    compl_x_l: *mut Number,
    compl_x_u: *mut Number,
    grad_lag_x: *mut Number,
    m: Index,
    nlp_constraint_violation: *mut Number,
    compl_g: *mut Number,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        if n != info.n || m != info.m {
            return FALSE;
        }
        let result = ip_intermediate::with_current(|ctx| {
            let data = ctx.data.borrow();
            let Some(_curr) = data.curr.as_ref() else {
                return false;
            };
            drop(data);
            let cq = ctx.cq.borrow();
            let n_us = n as usize;
            let m_us = m as usize;
            // No `nlp` borrow may be held across a `cq` accessor that
            // evaluates through the NLP (`curr_grad_lag_x` reaches
            // `curr_grad_f`, which takes `nlp.borrow_mut()`); each branch
            // below therefore borrows `nlp` only to pack a value that has
            // already been computed. See `GetIpoptCurrentIterate`.
            // x_L / x_U violations: scatter the compressed slack-shortfalls
            // up to full-`n`. Upstream defines `x_L_violation_i = max(0, x_L_i
            // - x_i)`; the algorithm tracks `slack_x_l = P_L^T x - x_L`
            // (always non-negative at feasible iterates), so reverse the
            // sign and clamp.
            if !x_l_violation.is_null() && n_us > 0 {
                let slack = cq.curr_slack_x_l();
                let z_l_full = ctx.nlp.borrow().pack_z_l_for_user(&*slack);
                // Guard the scatter length exactly like the sibling branches
                // below: an unexpected packed length would otherwise index
                // `v[i]` out of bounds and panic across this `extern "C"`
                // boundary (an abort, not a recoverable error).
                if z_l_full.len() != n_us {
                    return false;
                }
                // pack_z_l_for_user scatters by the same x_L mapping; the
                // returned vector at full-x positions holds `slack_x_l[i]`
                // which is `x_i - x_L_i`. Clamp the *negative* part to get
                // the violation `max(0, x_L_i - x_i)`.
                let mut v = vec![0.0; n_us];
                for (i, s) in z_l_full.iter().enumerate() {
                    v[i] = (-s).max(0.0);
                }
                std::ptr::copy_nonoverlapping(v.as_ptr(), x_l_violation, n_us);
            }
            if !x_u_violation.is_null() && n_us > 0 {
                let slack = cq.curr_slack_x_u();
                let s_full = ctx.nlp.borrow().pack_z_u_for_user(&*slack);
                if s_full.len() != n_us {
                    return false;
                }
                let mut v = vec![0.0; n_us];
                for (i, s) in s_full.iter().enumerate() {
                    v[i] = (-s).max(0.0);
                }
                std::ptr::copy_nonoverlapping(v.as_ptr(), x_u_violation, n_us);
            }
            if !compl_x_l.is_null() && n_us > 0 {
                let compl = cq.curr_compl_x_l();
                let v = ctx.nlp.borrow().pack_z_l_for_user(&*compl);
                if v.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(v.as_ptr(), compl_x_l, n_us);
            }
            if !compl_x_u.is_null() && n_us > 0 {
                let compl = cq.curr_compl_x_u();
                let v = ctx.nlp.borrow().pack_z_u_for_user(&*compl);
                if v.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(v.as_ptr(), compl_x_u, n_us);
            }
            if !grad_lag_x.is_null() && n_us > 0 {
                let glx = cq.curr_grad_lag_x();
                // Scatter compressed x-var → full-x via lift_x_to_full
                // (treats `glx` as if it were an x-vector). Fixed-variable
                // slots remain zero.
                let full = ctx.nlp.borrow().lift_x_to_full(&*glx);
                if full.len() != n_us {
                    return false;
                }
                std::ptr::copy_nonoverlapping(full.as_ptr(), grad_lag_x, n_us);
            }
            if !nlp_constraint_violation.is_null() && m_us > 0 {
                // Per-row equality and range violation reconstruction in
                // full-g coordinates is a follow-up. The scalar
                // `curr_primal_infeasibility_max` (== `inf_pr` reported in
                // `IterStats`) is the outer summary; populate per-row
                // detail as a future refinement and zero-fill for now.
                let zero = vec![0.0; m_us];
                std::ptr::copy_nonoverlapping(zero.as_ptr(), nlp_constraint_violation, m_us);
            }
            if !compl_g.is_null() && m_us > 0 {
                // Per-row constraint complementarity (`v_L .* s_L` /
                // `v_U .* s_U` mapped back to full-g) is also a follow-up.
                let zero = vec![0.0; m_us];
                std::ptr::copy_nonoverlapping(zero.as_ptr(), compl_g, m_us);
            }
            true
        });
        if result.unwrap_or(false) { TRUE } else { FALSE }
    }
}

/// Port of `IpStdCInterface.cpp:GetIpoptVersion` (Ipopt 3.14.18+).
/// Writes the pounce crate's `major.minor.patch` into the buffers.
/// Any pointer may be NULL to skip that component.
///
/// # Safety
///
/// Each non-NULL pointer must point at a writable `int`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptVersion(
    major: *mut c_int,
    minor: *mut c_int,
    release: *mut c_int,
) {
    unsafe {
        // Read from Cargo at compile time so the symbol always matches the
        // shipped binary. `unwrap_or(0)` keeps the function infallible if a
        // component is missing from the manifest (shouldn't happen in
        // practice — workspace manifest requires SemVer triples).
        let (mj, mn, pt) = parse_pkg_version(env!("CARGO_PKG_VERSION"));
        if !major.is_null() {
            *major = mj;
        }
        if !minor.is_null() {
            *minor = mn;
        }
        if !release.is_null() {
            *release = pt;
        }
    }
}

fn parse_pkg_version(v: &str) -> (c_int, c_int, c_int) {
    let mut it = v.split('.').map(|s| s.parse::<c_int>().unwrap_or(0));
    (
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
        it.next().unwrap_or(0),
    )
}

// ----------------------------------------------------------------------
// Pounce extensions: post-solve statistics accessors.
//
// Convenience accessors not present in upstream Ipopt's C API. Valid
// only after [`IpoptSolve`] has returned; calling them on a
// never-solved problem yields zero. They expose the same
// `SolveStatistics` data the Rust API surfaces via
// [`IpoptApplication::statistics`].
// ----------------------------------------------------------------------

/// Number of IPM iterations in the most recent solve, or `0` if the
/// problem has not been solved yet.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptIterCount(ipopt_problem: IpoptProblem) -> Index {
    unsafe { last_stat(ipopt_problem, |s| s.iteration_count).unwrap_or(0) }
}

/// Wall-clock solve time in seconds for the most recent solve, or
/// `0.0` if the problem has not been solved yet.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptSolveTime(ipopt_problem: IpoptProblem) -> Number {
    unsafe { last_stat(ipopt_problem, |s| s.total_wallclock_time_secs).unwrap_or(0.0) }
}

/// Final primal infeasibility (max constraint violation) for the most
/// recent solve.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptPrimalInf(ipopt_problem: IpoptProblem) -> Number {
    unsafe { last_stat(ipopt_problem, |s| s.final_constr_viol).unwrap_or(0.0) }
}

/// Final dual infeasibility (max gradient-of-Lagrangian norm) for the
/// most recent solve.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptDualInf(ipopt_problem: IpoptProblem) -> Number {
    unsafe { last_stat(ipopt_problem, |s| s.final_dual_inf).unwrap_or(0.0) }
}

/// Final complementarity error for the most recent solve.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptComplInf(ipopt_problem: IpoptProblem) -> Number {
    unsafe { last_stat(ipopt_problem, |s| s.final_compl).unwrap_or(0.0) }
}

unsafe fn last_stat<T, F>(ipopt_problem: IpoptProblem, f: F) -> Option<T>
where
    F: FnOnce(&SolveStatistics) -> T,
{
    unsafe {
        if ipopt_problem.is_null() {
            return None;
        }
        (*ipopt_problem).last_solve.as_ref().map(|ls| f(&ls.stats))
    }
}

/// Restoration-phase activity in the most recent solve. Each output
/// may be NULL to skip it; all yield zero before the first solve.
///
/// Solve-level, and still worth having now that the intermediate
/// callback also fires from restoration with `alg_mod = 1` (gh#645):
/// these counters answer "how much restoration did this solve do?"
/// without the caller having to install a callback and tally fires,
/// and they are the only source for the inner iteration count and
/// wall time.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. Each
/// non-NULL output pointer must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceRestorationStats(
    ipopt_problem: IpoptProblem,
    calls: *mut Index,
    inner_iters: *mut Index,
    outer_iters: *mut Index,
    wall_secs: *mut Number,
) {
    unsafe {
        let stats = last_stat(ipopt_problem, |s| {
            (
                s.restoration_calls,
                s.restoration_inner_iters,
                s.restoration_outer_iters,
                s.restoration_wall_secs,
            )
        });
        let (c, i, o, w) = stats.unwrap_or((0, 0, 0, 0.0));
        if !calls.is_null() {
            *calls = c;
        }
        if !inner_iters.is_null() {
            *inner_iters = i;
        }
        if !outer_iters.is_null() {
            *outer_iters = o;
        }
        if !wall_secs.is_null() {
            *wall_secs = w;
        }
    }
}

/// C mirror of [`pounce_linsol::summary::LinearSolverSummary`], laid
/// out for `pounce.h`'s `PounceLinearSolverStats`. Optional fields
/// carry sentinels rather than a discriminant, because a plain struct
/// of scalars is what a C or C++ caller can consume without an
/// accessor per field: `NaN` for absent reals, `-1` for absent counts.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PounceLinearSolverStats {
    pub solver_name: [c_char; 32],
    pub n_factors: Index,
    pub n_pattern_reuse: Index,
    pub n_pattern_changes: Index,
    pub max_fill_ratio: Number,
    pub min_abs_pivot: Number,
    pub max_abs_pivot: Number,
    pub last_inertia_positive: Index,
    pub last_inertia_negative: Index,
    pub last_inertia_zero: Index,
    pub last_nnz_a: Index,
    pub last_nnz_l: Index,
}

/// Post-mortem of the KKT linear solver for the most recent solve.
///
/// Reports what pounce already collects — factorization counts,
/// pattern reuse, fill, pivot range, final inertia. Timings are not
/// among them: pounce does not instrument the analyse / factor /
/// solve phases separately, so there is nothing honest to report and
/// the struct omits them rather than inventing zeros.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `stats`,
/// when non-NULL, must point at a writable `PounceLinearSolverStats`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceLinearSolverStats(
    ipopt_problem: IpoptProblem,
    stats: *mut PounceLinearSolverStats,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() || stats.is_null() {
            return FALSE;
        }
        let Some(summary) = (*ipopt_problem)
            .last_solve
            .as_ref()
            .and_then(|ls| ls.linear_solver.as_ref())
        else {
            return FALSE;
        };
        // Saturate rather than wrap: these are diagnostics, and a
        // count that does not fit an `Index` is better reported as the
        // largest representable one than as a negative.
        let count = |v: u64| Index::try_from(v).unwrap_or(Index::MAX);
        let size = |x: usize| Index::try_from(x).unwrap_or(Index::MAX);
        let opt_size = |v: Option<usize>| v.map_or(-1, size);
        let inertia = summary.last_inertia;
        let mut out = PounceLinearSolverStats {
            solver_name: [0; 32],
            n_factors: count(summary.n_factors),
            n_pattern_reuse: count(summary.n_pattern_reuse),
            n_pattern_changes: count(summary.n_pattern_changes),
            max_fill_ratio: summary.max_fill_ratio.unwrap_or(Number::NAN),
            min_abs_pivot: summary.min_abs_pivot.unwrap_or(Number::NAN),
            max_abs_pivot: summary.max_abs_pivot.unwrap_or(Number::NAN),
            last_inertia_positive: inertia.map_or(-1, |(p, _, _)| size(p)),
            last_inertia_negative: inertia.map_or(-1, |(_, n, _)| size(n)),
            last_inertia_zero: inertia.map_or(-1, |(_, _, z)| size(z)),
            last_nnz_a: opt_size(summary.last_nnz_a),
            last_nnz_l: opt_size(summary.last_nnz_l),
        };
        // Truncate on a byte boundary and always leave room for the
        // NUL; backend names are ASCII identifiers well under 31 bytes,
        // so this is a guard rather than an expected path.
        let name = summary.solver_name.as_bytes();
        let keep = name.len().min(out.solver_name.len() - 1);
        for (slot, b) in out.solver_name.iter_mut().zip(&name[..keep]) {
            *slot = *b as c_char;
        }
        *stats = out;
        TRUE
    }
}

thread_local! {
    /// Option registry for handle-free [`GetPounceOptionType`] queries.
    ///
    /// Built from a throwaway application rather than by re-running the
    /// registration functions here, so it cannot drift from the set a
    /// real problem carries: it *is* that set, by construction.
    static DEFAULT_REGISTRY: Rc<pounce_common::reg_options::RegisteredOptions> =
        Rc::clone(IpoptApplication::new().registered_options());
}

/// Which `AddIpopt*Option` setter a keyword expects.
///
/// Returns a `PounceOptionType` discriminant: 0 when the keyword is not
/// registered in this build, 1 number, 2 integer, 3 string. Lets a
/// caller forwarding options from an untyped source pick the setter
/// from pounce's registry instead of from the value's own type — the
/// difference between `tol: 1` reaching the solver as `1.0` and being
/// refused as an integer.
///
/// `ipopt_problem` may be NULL: option types are a property of the
/// build, not of a problem, and a caller deciding how to forward
/// options may not have created one yet (code generators, in
/// particular, have no handle at all).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `keyword`,
/// when non-NULL, must be a NUL-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceOptionType(
    ipopt_problem: IpoptProblem,
    keyword: *const c_char,
) -> c_int {
    unsafe {
        if keyword.is_null() {
            return 0;
        }
        let Ok(name) = CStr::from_ptr(keyword).to_str() else {
            return 0;
        };
        let registered = if ipopt_problem.is_null() {
            DEFAULT_REGISTRY.with(|r| r.get_option(name))
        } else {
            (*ipopt_problem).app.registered_options().get_option(name)
        };
        let Some(opt) = registered else {
            return 0;
        };
        match opt.option_type {
            OptionType::OT_Number => 1,
            OptionType::OT_Integer => 2,
            OptionType::OT_String => 3,
            OptionType::OT_Unknown => 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Pounce extension: SQP working-set warm-start C ABI (§7.2 of
// `docs/research/active-set-sqp-warm-start.md`).
//
// Three new entry points; all backward-compatible additions.
// No existing signature changes — existing cyipopt / JuMP /
// AMPL clients are unaffected.
// ─────────────────────────────────────────────────────────────

fn bound_status_to_int(s: pounce_qp::BoundStatus) -> c_int {
    use pounce_qp::BoundStatus::*;
    match s {
        Inactive => POUNCE_WS_INACTIVE,
        AtLower => POUNCE_WS_AT_LOWER,
        AtUpper => POUNCE_WS_AT_UPPER,
        Fixed => POUNCE_WS_FIXED_OR_EQ,
    }
}

fn int_to_bound_status(v: c_int) -> Option<pounce_qp::BoundStatus> {
    use pounce_qp::BoundStatus::*;
    match v {
        POUNCE_WS_INACTIVE => Some(Inactive),
        POUNCE_WS_AT_LOWER => Some(AtLower),
        POUNCE_WS_AT_UPPER => Some(AtUpper),
        POUNCE_WS_FIXED_OR_EQ => Some(Fixed),
        _ => None,
    }
}

fn cons_status_to_int(s: pounce_qp::ConsStatus) -> c_int {
    use pounce_qp::ConsStatus::*;
    match s {
        Inactive => POUNCE_WS_INACTIVE,
        AtLower => POUNCE_WS_AT_LOWER,
        AtUpper => POUNCE_WS_AT_UPPER,
        Equality => POUNCE_WS_FIXED_OR_EQ,
    }
}

fn int_to_cons_status(v: c_int) -> Option<pounce_qp::ConsStatus> {
    use pounce_qp::ConsStatus::*;
    match v {
        POUNCE_WS_INACTIVE => Some(Inactive),
        POUNCE_WS_AT_LOWER => Some(AtLower),
        POUNCE_WS_AT_UPPER => Some(AtUpper),
        POUNCE_WS_FIXED_OR_EQ => Some(Equality),
        _ => None,
    }
}

/// Internal → user row map for the SQP's constraint vector.
///
/// The SQP works on a *reordered* constraint vector: equalities first,
/// inequalities after, each in ascending original index. Its working set
/// is in that order, and both C entry points copied it positionally
/// against the caller's row indices — so on HS071, whose rows are
/// `[x₀x₁x₂x₃ ≥ 25, Σxᵢ² = 40]`, `IpoptGetWorkingSet` reported
/// `[Equality, AtLower]`: exactly reversed. A caller feeding that back
/// through `IpoptSetWarmStartWorkingSet`, which is the documented
/// round-trip, warm-started with the statuses swapped.
///
/// The split is a pure function of the bounds — a row is an equality iff
/// both sides are finite and equal — so the map is reconstructible here,
/// with no need to plumb `BoundClassification` out of `pounce-nlp`.
/// Mirrors `tnlp_adapter::classify_bounds`, sentinel test included.
fn internal_to_user_rows(g_l: &[Number], g_u: &[Number]) -> Vec<usize> {
    let m = g_l.len();
    let is_eq =
        |i: usize| g_l[i] > NLP_LOWER_BOUND_INF && g_u[i] < NLP_UPPER_BOUND_INF && g_l[i] == g_u[i];
    let mut map: Vec<usize> = (0..m).filter(|&i| is_eq(i)).collect();
    map.extend((0..m).filter(|&i| !is_eq(i)));
    map
}

/// Internal → user variable map for the SQP's bound vector.
///
/// Fixed variables (`x_l == x_u`) are removed from the internal problem
/// altogether, so the working set's bound vector is indexed by *non-fixed*
/// position. Same reconstruction argument as `internal_to_user_rows`.
fn internal_to_user_vars(x_l: &[Number], x_u: &[Number]) -> Vec<usize> {
    (0..x_l.len()).filter(|&i| x_l[i] != x_u[i]).collect()
}

/// Retrieve the working set produced by the most recent SQP solve
/// (`algorithm = active-set-sqp`). Buffer sizes are `n` for
/// `bound_status_out` and `m` for `cons_status_out`. Pass `NULL`
/// for either to skip that side.
///
/// Returns `TRUE` (1) on success, `FALSE` (0) if there is no
/// working set to retrieve (e.g. no SQP solve has run, the IPM
/// path was used, or the very first KKT check declared
/// optimality before solving any QP).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. Output
/// buffers (when non-NULL) must be sized at least `n` and `m`
/// respectively.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptGetWorkingSet(
    ipopt_problem: IpoptProblem,
    bound_status_out: *mut IpoptBoundStatus,
    cons_status_out: *mut IpoptConsStatus,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        let ws = match info.app.last_sqp_working_set() {
            Some(w) => w,
            None => return FALSE,
        };
        // Translate out of the SQP's equalities-first ordering, back into
        // the caller's row / variable indices.
        let row_map = internal_to_user_rows(&info.g_l, &info.g_u);
        let var_map = internal_to_user_vars(&info.x_l, &info.x_u);
        if ws.constraints.len() != row_map.len() || ws.bounds.len() != var_map.len() {
            // The stored set does not match this problem's shape. Report
            // nothing rather than something mis-indexed.
            return FALSE;
        }
        if !bound_status_out.is_null() {
            // A fixed variable is absent from the internal problem and so
            // has no stored status. `Fixed` is what it is.
            for i in 0..info.x_l.len() {
                *bound_status_out.add(i) = POUNCE_WS_FIXED_OR_EQ;
            }
            for (internal, &user) in var_map.iter().enumerate() {
                *bound_status_out.add(user) = bound_status_to_int(ws.bounds[internal]);
            }
        }
        if !cons_status_out.is_null() {
            for (internal, &user) in row_map.iter().enumerate() {
                *cons_status_out.add(user) = cons_status_to_int(ws.constraints[internal]);
            }
        }
        TRUE
    }
}

/// Supply a warm-start working set consumed by the next
/// [`IpoptSolve`] on this problem. Pass `NULL` for either side to
/// cold-start it. The caller-owned buffers are copied; reuse
/// across calls is safe.
///
/// Returns `TRUE` on success, `FALSE` on (a) NULL problem, (b)
/// an out-of-range status code in one of the buffers, or
/// (c) both inputs NULL (which would equal a no-op
/// — call [`IpoptClearWarmStartWorkingSet`] instead).
///
/// # Safety
///
/// `ipopt_problem` must be valid. `bound_status_in` (when
/// non-NULL) must be sized `n`; `cons_status_in` (when non-NULL)
/// must be sized `m`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptSetWarmStartWorkingSet(
    ipopt_problem: IpoptProblem,
    bound_status_in: *const IpoptBoundStatus,
    cons_status_in: *const IpoptConsStatus,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        if bound_status_in.is_null() && cons_status_in.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        let n = info.n.max(0) as usize;
        let m = info.m.max(0) as usize;
        // The caller indexes by *their* rows and variables; the SQP works
        // on the reordered, fixed-variables-removed vectors. Translate on
        // the way in, mirroring `IpoptGetWorkingSet` on the way out, so
        // the documented get/set round-trip is actually a round-trip.
        let row_map = internal_to_user_rows(&info.g_l, &info.g_u);
        let var_map = internal_to_user_vars(&info.x_l, &info.x_u);
        // Validation is not just a range check. A status code says
        // where the point sits relative to a bound — but `Fixed` /
        // `Equality`, and `AtLower` / `AtUpper` against an infinite
        // side, additionally assert something about the *problem*:
        // that a variable has `x_l == x_u`, that a row has
        // `b_l == b_u`, that the bound being sat on exists at all.
        // Those are not guesses about the active set, they are claims
        // about the model, and the model is right here to check them
        // against.
        //
        // Accepting them unchecked converted a caller's mistake into a
        // silently wrong answer: `POUNCE_WS_FIXED_OR_EQ` on a variable
        // whose bounds differ pinned it, the solve over-constrained
        // itself, and a *convex* program came back with the wrong
        // optimum — the function having returned TRUE. Rejecting is
        // strictly better: the caller already handles FALSE, and this
        // function already returns it for an out-of-range code, so
        // TRUE reasonably reads as "your working set was accepted".
        let mut bounds = vec![pounce_qp::BoundStatus::Inactive; var_map.len()];
        if !bound_status_in.is_null() {
            // Validate every entry the caller supplied, including those for
            // fixed variables that the internal problem drops: a wrong
            // claim is worth rejecting whether or not it would be used.
            for i in 0..n {
                let v = *bound_status_in.add(i);
                let Some(s) = int_to_bound_status(v) else {
                    return FALSE;
                };
                let lo_finite = info.x_l[i] > NLP_LOWER_BOUND_INF;
                let hi_finite = info.x_u[i] < NLP_UPPER_BOUND_INF;
                let consistent = match s {
                    pounce_qp::BoundStatus::Fixed => info.x_l[i] == info.x_u[i],
                    pounce_qp::BoundStatus::AtLower => lo_finite,
                    pounce_qp::BoundStatus::AtUpper => hi_finite,
                    pounce_qp::BoundStatus::Inactive => true,
                };
                if !consistent {
                    return FALSE;
                }
            }
            for (internal, &user) in var_map.iter().enumerate() {
                // Already range-checked above.
                if let Some(s) = int_to_bound_status(*bound_status_in.add(user)) {
                    bounds[internal] = s;
                }
            }
        }
        let mut constraints = vec![pounce_qp::ConsStatus::Inactive; m];
        if !cons_status_in.is_null() {
            for i in 0..m {
                let v = *cons_status_in.add(i);
                let Some(s) = int_to_cons_status(v) else {
                    return FALSE;
                };
                let lo_finite = info.g_l[i] > NLP_LOWER_BOUND_INF;
                let hi_finite = info.g_u[i] < NLP_UPPER_BOUND_INF;
                let consistent = match s {
                    pounce_qp::ConsStatus::Equality => {
                        lo_finite && hi_finite && info.g_l[i] == info.g_u[i]
                    }
                    pounce_qp::ConsStatus::AtLower => lo_finite,
                    pounce_qp::ConsStatus::AtUpper => hi_finite,
                    pounce_qp::ConsStatus::Inactive => true,
                };
                if !consistent {
                    return FALSE;
                }
            }
            for (internal, &user) in row_map.iter().enumerate() {
                if let Some(s) = int_to_cons_status(*cons_status_in.add(user)) {
                    constraints[internal] = s;
                }
            }
        }
        // Stage the working set only. We do *not* know the
        // primal/dual iterate here, and we must not invent one:
        // `SqpAlgorithm::optimize_with_warm_start` treats a supplied
        // `SqpIterates` as *the* starting iterate and never consults
        // `get_starting_x` on that branch, so a placeholder `x` would
        // silently override the `x` buffer the caller hands to
        // `IpoptSolve`. Zeros here restarted every warm solve from
        // the origin — outside the bounds on any problem with
        // `x_l > 0` — and returned `Infeasible_Problem_Detected` at
        // iteration 0 (gh#484). `IpoptSolve` merges this working set
        // with the real starting point instead.
        info.pending_working_set = Some(pounce_qp::WorkingSet {
            bounds,
            constraints,
        });
        TRUE
    }
}

/// Declare which variables enter the problem **nonlinearly** (gh#624).
///
/// This is the C-API face of Ipopt's `TNLP::get_number_of_nonlinear_variables`
/// / `get_list_of_nonlinear_variables` pair, which upstream exposes only
/// to C++ callers. It exists so a frontend that already knows the
/// structure of its model — CasADi's `pass_nonlinear_variables`, an
/// algebraic modeling language, a hand-written driver — can hand that
/// knowledge to pounce.
///
/// Effect is confined to the **limited-memory** Hessian: curvature is
/// approximated over the declared subset only, and the Hessian is
/// exactly zero for every other variable. Exact-Hessian solves ignore
/// the declaration entirely, and so does any solve that never calls
/// this function — the default remains "all variables are nonlinear".
/// The subset takes precedence over the `num_linear_variables` option,
/// matching Ipopt's own ordering.
///
/// `pos_nonlin_vars` holds `num_nonlin_vars` variable indices **in the
/// problem's index style** (the `index_style` passed to
/// [`CreateIpoptProblem`]). The subset may be arbitrary and
/// noncontiguous; order does not matter. Passing `num_nonlin_vars == n`
/// is equivalent to not calling this at all.
///
/// Returns `FALSE` (leaving any previous declaration untouched) on a
/// NULL problem, a negative or oversized count, a NULL array with a
/// positive count, or an index outside the problem's variable range.
///
/// Note the deliberate signature choice: the issue that requested this
/// suggested a `const Bool*` mask, but `Bool` in the Ipopt C API is a
/// C99 `bool`, and a *array* of those would be a per-element
/// data-layout contract that is easy to get wrong from a caller with a
/// different boolean width. The count-plus-index-list shape is the one
/// the TNLP callbacks already use.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
/// `pos_nonlin_vars`, when non-NULL, must point at `num_nonlin_vars`
/// readable `ipindex` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptSetNonlinearVariables(
    ipopt_problem: IpoptProblem,
    num_nonlin_vars: Index,
    pos_nonlin_vars: *const Index,
) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &mut *ipopt_problem;
        if num_nonlin_vars < 0 || num_nonlin_vars > info.n {
            return FALSE;
        }
        if num_nonlin_vars > 0 && pos_nonlin_vars.is_null() {
            return FALSE;
        }
        let offset = if info.index_style == 1 { 1 } else { 0 };
        let raw = if num_nonlin_vars == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(pos_nonlin_vars, num_nonlin_vars as usize)
        };
        // Validate before storing: a half-applied declaration would be
        // worse than a refused one.
        for &p in raw {
            let zero_based = p - offset;
            if zero_based < 0 || zero_based >= info.n {
                return FALSE;
            }
        }
        info.nonlinear_vars = Some(raw.to_vec());
        TRUE
    }
}

/// Drop a subset declared by [`IpoptSetNonlinearVariables`], restoring
/// the default (every variable treated as nonlinear).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptClearNonlinearVariables(ipopt_problem: IpoptProblem) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        (*ipopt_problem).nonlinear_vars = None;
        TRUE
    }
}

/// Drop any pending warm-start working set without solving. The
/// next [`IpoptSolve`] will cold-start.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptClearWarmStartWorkingSet(ipopt_problem: IpoptProblem) -> Bool {
    unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        (*ipopt_problem).pending_working_set = None;
        (*ipopt_problem).app.clear_sqp_warm_start();
        TRUE
    }
}

/// Convenience one-shot: equivalent to
/// `IpoptSetWarmStartWorkingSet` + `IpoptSolve` +
/// `IpoptGetWorkingSet` in sequence. The input/output working-set
/// buffers are independent (so a caller can read back the new
/// working set into the same array used as input). Pass `NULL`
/// for any in/out buffer to skip that side.
///
/// Returns the `ApplicationReturnStatus` integer, identical to
/// [`IpoptSolve`].
///
/// # Safety
///
/// All pointer arguments follow the same contract as
/// `IpoptSolve` plus the working-set buffer sizes documented on
/// `IpoptSetWarmStartWorkingSet` / `IpoptGetWorkingSet`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptSolveWarmStart(
    ipopt_problem: IpoptProblem,
    x: *mut Number,
    g: *mut Number,
    obj_val: *mut Number,
    mult_g: *mut Number,
    mult_x_L: *mut Number,
    mult_x_U: *mut Number,
    bound_status_in: *const IpoptBoundStatus,
    cons_status_in: *const IpoptConsStatus,
    bound_status_out: *mut IpoptBoundStatus,
    cons_status_out: *mut IpoptConsStatus,
    user_data: *mut c_void,
) -> Index {
    if ipopt_problem.is_null() {
        return ApplicationReturnStatus::InternalError as Index;
    }
    // Guard the working-set set/get helpers too. The inner `IpoptSolve` is
    // independently guarded, but a panic in the warm-start working-set
    // marshalling would otherwise still abort across `extern "C"`.
    ffi_guard(ApplicationReturnStatus::InternalError as Index, || unsafe {
        // Best-effort set. Errors here (e.g. bad status code) are
        // silently treated as cold-start; the caller can probe via
        // `IpoptSetWarmStartWorkingSet` directly if they need to
        // validate the input.
        if !bound_status_in.is_null() || !cons_status_in.is_null() {
            let _ = IpoptSetWarmStartWorkingSet(ipopt_problem, bound_status_in, cons_status_in);
        }
        let status = IpoptSolve(
            ipopt_problem,
            x,
            g,
            obj_val,
            mult_g,
            mult_x_L,
            mult_x_U,
            user_data,
        );
        let _ = IpoptGetWorkingSet(ipopt_problem, bound_status_out, cons_status_out);
        status
    })
}

/// Adapter that bridges the user-supplied C callback table to the
/// in-crate [`TNLP`] trait. Mirrors `Interfaces/IpStdInterfaceTNLP.cpp`
/// (`StdInterfaceTNLP`); each TNLP method forwards to the matching
/// `Eval_*_CB` and propagates `false` returns up so the algorithm
/// layer can map them to `Invalid_Number_Detected`.
///
/// Holds a snapshot of bounds and the initial `x`. After `optimize_tnlp`
/// finishes, `finalize_solution` is called by the algorithm layer; the
/// adapter records the final iterate in `final_*` fields, which the
/// outer [`IpoptSolve`] copies back into the caller's buffers.
pub(crate) struct CCallbackTnlp {
    pub(crate) n: Index,
    pub(crate) m: Index,
    pub(crate) nele_jac: Index,
    pub(crate) nele_hess: Index,
    pub(crate) index_style: Index,
    pub(crate) x_l: Vec<Number>,
    pub(crate) x_u: Vec<Number>,
    pub(crate) g_l: Vec<Number>,
    pub(crate) g_u: Vec<Number>,
    pub(crate) initial_x: Vec<Number>,
    pub(crate) eval_f: Option<Eval_F_CB>,
    pub(crate) eval_grad_f: Option<Eval_Grad_F_CB>,
    pub(crate) eval_g: Option<Eval_G_CB>,
    pub(crate) eval_jac_g: Option<Eval_Jac_G_CB>,
    pub(crate) eval_h: Option<Eval_H_CB>,
    pub(crate) user_data: *mut c_void,
    /// User-installed intermediate callback, copied at solve time so the
    /// TNLP-trait `intermediate_callback` impl can forward through to it.
    pub(crate) intermediate_cb: Option<Intermediate_CB>,
    /// Snapshot of user-provided scaling captured at solve time.
    pub(crate) user_scaling: Option<UserScaling>,
    /// Snapshot of the nonlinear-variable subset (gh#624), in the
    /// problem's index style.
    pub(crate) nonlinear_vars: Option<Vec<Index>>,
    pub(crate) final_status: Option<pounce_nlp::alg_types::SolverReturn>,
    pub(crate) final_x: Vec<Number>,
    pub(crate) final_z_l: Vec<Number>,
    pub(crate) final_z_u: Vec<Number>,
    pub(crate) final_g: Vec<Number>,
    pub(crate) final_lambda: Vec<Number>,
    pub(crate) final_obj: Number,
}

impl TNLP for CCallbackTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as pounce_common::types::Index,
            m: self.m as pounce_common::types::Index,
            nnz_jac_g: self.nele_jac as pounce_common::types::Index,
            nnz_h_lag: self.nele_hess as pounce_common::types::Index,
            index_style: if self.index_style == 1 {
                IndexStyle::Fortran
            } else {
                IndexStyle::C
            },
        })
    }

    /// gh#624 — serve the subset staged by
    /// [`IpoptSetNonlinearVariables`]. `-1` (no subset) keeps the
    /// upstream default of "all variables are nonlinear".
    fn get_number_of_nonlinear_variables(&mut self) -> pounce_common::types::Index {
        match &self.nonlinear_vars {
            Some(v) => v.len() as pounce_common::types::Index,
            None => -1,
        }
    }

    fn get_list_of_nonlinear_variables(
        &mut self,
        pos_nonlin_vars: &mut [pounce_common::types::Index],
    ) -> bool {
        let Some(v) = self.nonlinear_vars.as_ref() else {
            return false;
        };
        if v.len() != pos_nonlin_vars.len() {
            return false;
        }
        pos_nonlin_vars.copy_from_slice(v);
        true
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        if !self.x_l.is_empty() {
            b.x_l.copy_from_slice(&self.x_l);
        }
        if !self.x_u.is_empty() {
            b.x_u.copy_from_slice(&self.x_u);
        }
        if !self.g_l.is_empty() {
            b.g_l.copy_from_slice(&self.g_l);
        }
        if !self.g_u.is_empty() {
            b.g_u.copy_from_slice(&self.g_u);
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if !self.initial_x.is_empty() {
            sp.x.copy_from_slice(&self.initial_x);
        }
        true
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(s) = self.user_scaling.as_ref() else {
            return false;
        };
        *req.obj_scaling = s.obj_scaling;
        if let Some(x) = s.x_scaling.as_ref() {
            if x.len() == req.x_scaling.len() {
                req.x_scaling.copy_from_slice(x);
                *req.use_x_scaling = true;
            }
        } else {
            *req.use_x_scaling = false;
        }
        if let Some(g) = s.g_scaling.as_ref() {
            if g.len() == req.g_scaling.len() {
                req.g_scaling.copy_from_slice(g);
                *req.use_g_scaling = true;
            }
        } else {
            *req.use_g_scaling = false;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        let cb = self.eval_f?;
        let mut obj = 0.0;
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                &mut obj,
                self.user_data,
            )
        };
        if ok != FALSE { Some(obj) } else { None }
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        let Some(cb) = self.eval_grad_f else {
            return false;
        };
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                grad_f.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != FALSE
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        if self.m == 0 {
            return true;
        }
        let Some(cb) = self.eval_g else {
            return false;
        };
        let ok = unsafe {
            cb(
                self.n,
                x.as_ptr() as *mut Number,
                if new_x { TRUE } else { FALSE },
                self.m,
                g.as_mut_ptr(),
                self.user_data,
            )
        };
        ok != FALSE
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        if self.m == 0 || self.nele_jac == 0 {
            return true;
        }
        let Some(cb) = self.eval_jac_g else {
            return false;
        };
        let x_ptr = x
            .map(|s| s.as_ptr() as *mut Number)
            .unwrap_or(std::ptr::null_mut());
        let ok = match mode {
            SparsityRequest::Structure { irow, jcol } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    self.m,
                    self.nele_jac,
                    irow.as_mut_ptr(),
                    jcol.as_mut_ptr(),
                    std::ptr::null_mut(),
                    self.user_data,
                )
            },
            SparsityRequest::Values { values } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    self.m,
                    self.nele_jac,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    values.as_mut_ptr(),
                    self.user_data,
                )
            },
        };
        ok != FALSE
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let Some(cb) = self.eval_h else {
            return false;
        };
        if self.nele_hess == 0 {
            return true;
        }
        let x_ptr = x
            .map(|s| s.as_ptr() as *mut Number)
            .unwrap_or(std::ptr::null_mut());
        let lambda_ptr = lambda
            .map(|s| s.as_ptr() as *mut Number)
            .unwrap_or(std::ptr::null_mut());
        let ok = match mode {
            SparsityRequest::Structure { irow, jcol } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    obj_factor,
                    self.m,
                    lambda_ptr,
                    if new_lambda { TRUE } else { FALSE },
                    self.nele_hess,
                    irow.as_mut_ptr(),
                    jcol.as_mut_ptr(),
                    std::ptr::null_mut(),
                    self.user_data,
                )
            },
            SparsityRequest::Values { values } => unsafe {
                cb(
                    self.n,
                    x_ptr,
                    if new_x { TRUE } else { FALSE },
                    obj_factor,
                    self.m,
                    lambda_ptr,
                    if new_lambda { TRUE } else { FALSE },
                    self.nele_hess,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    values.as_mut_ptr(),
                    self.user_data,
                )
            },
        };
        ok != FALSE
    }

    fn intermediate_callback(
        &mut self,
        stats: pounce_nlp::tnlp::IterStats,
        _ip_data: &IpoptData,
        _ip_cq: &IpoptCq,
    ) -> bool {
        let Some(cb) = self.intermediate_cb else {
            return true;
        };
        let ok = unsafe {
            cb(
                stats.mode as Index,
                stats.iter as Index,
                stats.obj_value,
                stats.inf_pr,
                stats.inf_du,
                stats.mu,
                stats.d_norm,
                stats.regularization_size,
                stats.alpha_du,
                stats.alpha_pr,
                stats.ls_trials as Index,
                self.user_data,
            )
        };
        ok != FALSE
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_status = Some(sol.status);
        if !sol.x.is_empty() {
            self.final_x.copy_from_slice(sol.x);
        }
        if !sol.z_l.is_empty() {
            self.final_z_l.copy_from_slice(sol.z_l);
        }
        if !sol.z_u.is_empty() {
            self.final_z_u.copy_from_slice(sol.z_u);
        }
        if !sol.g.is_empty() {
            self.final_g.copy_from_slice(sol.g);
        }
        if !sol.lambda.is_empty() {
            self.final_lambda.copy_from_slice(sol.lambda);
        }
        self.final_obj = sol.obj_value;
    }
}

/// Enable per-iteration history capture on the underlying
/// `IpoptApplication`. Must be called *before* [`IpoptSolve`] for the
/// trajectory to appear in the report written by
/// [`IpoptWriteSolveReport`]. Off by default — capturing each iterate
/// has a small per-iter cost the IPM core skips otherwise.
///
/// Returns `TRUE` on success, `FALSE` if `ipopt_problem` is NULL.
///
/// # Safety
///
/// `ipopt_problem` must be a valid handle returned by
/// [`CreateIpoptProblem`] (or `NULL`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptEnableIterHistory(ipopt_problem: IpoptProblem) -> Bool {
    if ipopt_problem.is_null() {
        return FALSE;
    }
    let info = unsafe { &mut *ipopt_problem };
    info.app.enable_iter_history();
    TRUE
}

/// Write a `pounce.solve-report/v1` JSON file capturing the most
/// recent [`IpoptSolve`] result. `path` is a NUL-terminated UTF-8
/// filesystem path. `detail` is one of `"summary"` or `"full"`
/// (NUL-terminated); pass `NULL` for the default (`"summary"`).
///
/// When `detail = "full"` and [`IpoptEnableIterHistory`] was called
/// pre-solve, the per-iteration trajectory is embedded so that
/// downstream tools (`diagnose`, `find_stalls`, `convergence_trace`)
/// see the same trace the `pounce` CLI's `--json-output` path
/// produces. The input descriptor is recorded as `tnlp-direct`
/// because the cinterface receives callbacks rather than a file.
///
/// Returns `TRUE` on a successful write, `FALSE` for NULL handle,
/// no prior solve, an invalid `detail`, a bad path, or an I/O error.
///
/// # Safety
///
/// `ipopt_problem` must be a valid handle; `path` must be a valid
/// NUL-terminated UTF-8 string; `detail` must be NULL or a valid
/// NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn IpoptWriteSolveReport(
    ipopt_problem: IpoptProblem,
    path: *const c_char,
    detail: *const c_char,
) -> Bool {
    use pounce_solve_report::{
        InputDescriptor, ReportBuilder, ReportDetail, status_to_solve_result_num, write_report_file,
    };

    // Guard the report build/write: it clones the retained iterate and runs
    // the `pounce-solve-report` serializer + file I/O, any of which could
    // panic on an unexpected state. A panic unwinding across `extern "C"`
    // aborts the embedding process; report `FALSE` instead. (See `ffi_guard`.)
    ffi_guard(FALSE, || unsafe {
        if ipopt_problem.is_null() || path.is_null() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        let Some(last) = info.last_solve.as_ref() else {
            return FALSE;
        };

        let Ok(path_str) = CStr::from_ptr(path).to_str() else {
            return FALSE;
        };

        let detail_choice = if detail.is_null() {
            ReportDetail::Summary
        } else {
            let Ok(detail_str) = CStr::from_ptr(detail).to_str() else {
                return FALSE;
            };
            match ReportDetail::parse(detail_str) {
                Ok(d) => d,
                Err(_) => return FALSE,
            }
        };

        let mut builder = ReportBuilder::new(detail_choice, InputDescriptor::TnlpDirect);
        builder.problem.n_variables = info.n;
        builder.problem.n_constraints = info.m;
        builder.problem.n_objectives = 1;
        builder.problem.nnz_jac_g = Some(info.nele_jac);
        builder.problem.nnz_h_lag = Some(info.nele_hess);

        builder.solution.status = last.status;
        builder.solution.solve_result_num = status_to_solve_result_num(last.status);
        builder.solution.objective = last.final_obj;
        builder.solution.x = last.final_x.clone();
        builder.solution.lambda = last.final_lambda.clone();

        builder.ingest_stats(&last.stats);
        if let Some(linsol) = last.linear_solver.clone() {
            builder.set_linear_solver_summary(linsol);
        }

        let report = builder.finish();
        match write_report_file(std::path::Path::new(path_str), &report) {
            Ok(_) => TRUE,
            Err(_) => FALSE,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    unsafe extern "C" fn dummy_eval_f(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _obj_value: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }
    unsafe extern "C" fn dummy_eval_grad_f(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _grad_f: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }

    fn create_unconstrained() -> IpoptProblem {
        let xl = [-1.0; 4];
        let xu = [1.0; 4];
        unsafe {
            CreateIpoptProblem(
                4,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        }
    }

    #[test]
    fn create_succeeds_for_unconstrained_problem() {
        let p = create_unconstrained();
        assert!(!p.is_null());
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn create_returns_null_on_missing_required_callbacks() {
        let xl = [-1.0; 4];
        let xu = [1.0; 4];
        let p = unsafe {
            CreateIpoptProblem(
                4,
                xl.as_ptr(),
                xu.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                None, // missing eval_f
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn create_returns_null_on_negative_n() {
        let p = unsafe {
            CreateIpoptProblem(
                -1,
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                0,
                10,
                0,
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn create_returns_null_on_invalid_index_style() {
        let xl = [0.0; 1];
        let xu = [1.0; 1];
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
                2, // valid values are 0 and 1
                Some(dummy_eval_f),
                None,
                Some(dummy_eval_grad_f),
                None,
                None,
            )
        };
        assert!(p.is_null());
    }

    #[test]
    fn add_int_option_forwards_to_application() {
        let p = create_unconstrained();
        let key = CString::new("print_level").unwrap();
        let ok = unsafe { AddIpoptIntOption(p, key.as_ptr(), 5) };
        assert_eq!(ok, TRUE);
        let info = unsafe { &*p };
        let (level, found) = info
            .app
            .options()
            .get_integer_value("print_level", "")
            .unwrap();
        assert!(found);
        assert_eq!(level, 5);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn add_str_option_with_invalid_key_returns_false() {
        let p = create_unconstrained();
        let key = CString::new("totally_unknown_option").unwrap();
        let val = CString::new("yes").unwrap();
        let ok = unsafe { AddIpoptStrOption(p, key.as_ptr(), val.as_ptr()) };
        assert_eq!(ok, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn add_options_on_null_problem_returns_false() {
        let key = CString::new("print_level").unwrap();
        let v = CString::new("yes").unwrap();
        unsafe {
            assert_eq!(
                AddIpoptIntOption(std::ptr::null_mut(), key.as_ptr(), 5),
                FALSE
            );
            assert_eq!(
                AddIpoptNumOption(std::ptr::null_mut(), key.as_ptr(), 1.0),
                FALSE
            );
            assert_eq!(
                AddIpoptStrOption(std::ptr::null_mut(), key.as_ptr(), v.as_ptr()),
                FALSE
            );
        }
    }

    unsafe extern "C" fn dummy_intermediate(
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
        _user_data: *mut c_void,
    ) -> Bool {
        TRUE
    }

    #[test]
    fn set_intermediate_callback_stores_pointer() {
        let p = create_unconstrained();
        let ok = unsafe { SetIntermediateCallback(p, Some(dummy_intermediate)) };
        assert_eq!(ok, TRUE);
        let info = unsafe { &*p };
        assert!(info.intermediate_cb.is_some());
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn solve_returns_internal_error_on_null_problem() {
        let rc = unsafe {
            IpoptSolve(
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
        assert_eq!(rc, -199);
    }

    #[test]
    fn free_null_is_safe() {
        unsafe { FreeIpoptProblem(std::ptr::null_mut()) };
    }

    // ---- End-to-end bridge: 1-D unconstrained quadratic ----
    //
    // f(x) = (x - 2)^2, no bounds, no constraints. Newton driver
    // converges in one step.

    unsafe extern "C" fn quad_eval_f(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        obj_value: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            let v = *x.offset(0);
            *obj_value = (v - 2.0) * (v - 2.0);
            TRUE
        }
    }
    unsafe extern "C" fn quad_eval_grad_f(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        grad: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            let v = *x.offset(0);
            *grad.offset(0) = 2.0 * (v - 2.0);
            TRUE
        }
    }
    unsafe extern "C" fn quad_eval_h(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            if !irow.is_null() && !jcol.is_null() && values.is_null() {
                *irow.offset(0) = 0;
                *jcol.offset(0) = 0;
            } else if irow.is_null() && jcol.is_null() && !values.is_null() {
                *values.offset(0) = 2.0 * obj_factor;
            } else {
                return FALSE;
            }
            TRUE
        }
    }

    #[test]
    fn solve_drives_unconstrained_quadratic_through_bridge() {
        // Bounds wide open (kappa1 push won't move us off 0.0 since
        // |0| < 1e19, but the Newton step lands us at 2.0 anyway).
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        assert!(!p.is_null());
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
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
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        assert!((x[0] - 2.0).abs() < 1e-6, "x[0] = {}", x[0]);
        assert!(obj.abs() < 1e-10, "obj = {}", obj);
        unsafe { FreeIpoptProblem(p) };
    }

    /// F5: `IpoptSolve` invalidates the retained `last_solve` stats **up
    /// front**, so a solve that bails — or whose pounce-internal panic
    /// `ffi_guard` catches (returning `Internal_Error`) — does not leave the
    /// post-solve accessors (`GetIpoptIterCount`, `IpoptWriteSolveReport`, …)
    /// silently reporting the *previous* solve's stats.
    ///
    /// A caught panic can't be injected deterministically through the public
    /// C ABI (a panic in a user `extern "C"` callback aborts at its own
    /// boundary; see `ffi_guard`). We drive the equivalent control-flow shape:
    /// after a successful solve we corrupt `n` to a negative value so the next
    /// `IpoptSolve` returns `InvalidProblemDefinition` from inside the guarded
    /// body **without** reaching the trailing `last_solve = Some(..)` write —
    /// exactly where a caught panic also bails. The up-front clear makes the
    /// accessor report "no data" (0) in both cases rather than stale data.
    #[test]
    fn stale_stats_cleared_when_resolve_bails() {
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        assert!(!p.is_null());

        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
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
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        // The successful solve recorded real stats.
        let iters_after_success = unsafe { GetIpoptIterCount(p) };
        assert!(
            iters_after_success >= 1,
            "a converged solve should record >=1 iteration, got {iters_after_success}"
        );
        assert!(unsafe { (*p).last_solve.is_some() });

        // Corrupt the problem so the next solve bails early in the guarded body
        // (the same place a caught panic would land) without recording stats.
        unsafe { (*p).n = -1 };
        let mut x2 = [0.0_f64];
        let rc2 = unsafe {
            IpoptSolve(
                p,
                x2.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            rc2,
            ApplicationReturnStatus::InvalidProblemDefinition as Index
        );

        // Post-fix: the up-front invalidation cleared the retained stats, so
        // the accessor reports "no data" (0), not the previous iteration count.
        // Pre-fix this returned `iters_after_success` (stale).
        assert!(
            unsafe { (*p).last_solve.is_none() },
            "a bailed re-solve must clear stale last_solve (F5)"
        );
        assert_eq!(
            unsafe { GetIpoptIterCount(p) },
            0,
            "stale iteration count must not survive a bailed re-solve (F5)"
        );

        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn solve_invalid_problem_definition_when_x_null() {
        let p = create_unconstrained();
        let rc = unsafe {
            IpoptSolve(
                p,
                std::ptr::null_mut(), // x null but n > 0
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(
            rc,
            ApplicationReturnStatus::InvalidProblemDefinition as Index
        );
        unsafe { FreeIpoptProblem(p) };
    }

    // ---- New entry points (issue #19) ----

    #[test]
    fn get_version_writes_pkg_version() {
        let (mut mj, mut mn, mut pt) = (-1, -1, -1);
        unsafe { GetIpoptVersion(&mut mj, &mut mn, &mut pt) };
        let expected = parse_pkg_version(env!("CARGO_PKG_VERSION"));
        assert_eq!((mj, mn, pt), expected);
    }

    #[test]
    fn get_version_tolerates_null_buffers() {
        // None of these should crash.
        unsafe {
            GetIpoptVersion(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
    }

    #[test]
    fn set_scaling_stores_user_supplied_arrays() {
        let p = create_unconstrained();
        let xs = [2.0, 3.0, 4.0, 5.0];
        let ok = unsafe { SetIpoptProblemScaling(p, 7.0, xs.as_ptr(), std::ptr::null()) };
        assert_eq!(ok, TRUE);
        let info = unsafe { &*p };
        let s = info.user_scaling.as_ref().unwrap();
        assert_eq!(s.obj_scaling, 7.0);
        assert_eq!(s.x_scaling.as_deref(), Some(&xs[..]));
        assert!(s.g_scaling.is_none());
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn set_scaling_on_null_problem_returns_false() {
        let ok = unsafe {
            SetIpoptProblemScaling(
                std::ptr::null_mut(),
                1.0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        assert_eq!(ok, FALSE);
    }

    #[test]
    fn open_output_file_writes_and_attaches_journal() {
        let p = create_unconstrained();
        let dir = std::env::temp_dir().join("pounce-cinterface-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("output.log");
        let cstr = CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let ok = unsafe { OpenIpoptOutputFile(p, cstr.as_ptr(), 5) };
        assert_eq!(ok, TRUE);
        // Option should be reflected in the app.
        let info = unsafe { &*p };
        let (level, found) = info
            .app
            .options()
            .get_integer_value("file_print_level", "")
            .unwrap();
        assert!(found);
        assert_eq!(level, 5);
        unsafe { FreeIpoptProblem(p) };
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_output_file_with_null_inputs_returns_false() {
        let key = CString::new("nope").unwrap();
        unsafe {
            assert_eq!(
                OpenIpoptOutputFile(std::ptr::null_mut(), key.as_ptr(), 0),
                FALSE
            );
        }
        let p = create_unconstrained();
        unsafe {
            assert_eq!(OpenIpoptOutputFile(p, std::ptr::null(), 0), FALSE);
            FreeIpoptProblem(p);
        }
    }

    #[test]
    fn get_current_iterate_returns_false_outside_callback() {
        let p = create_unconstrained();
        let rc = unsafe {
            GetIpoptCurrentIterate(
                p,
                FALSE,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn get_current_violations_returns_false_outside_callback() {
        let p = create_unconstrained();
        let rc = unsafe {
            GetIpoptCurrentViolations(
                p,
                FALSE,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn post_solve_stats_zero_before_solve() {
        let p = create_unconstrained();
        unsafe {
            assert_eq!(GetIpoptIterCount(p), 0);
            assert_eq!(GetIpoptSolveTime(p), 0.0);
            assert_eq!(GetIpoptPrimalInf(p), 0.0);
            assert_eq!(GetIpoptDualInf(p), 0.0);
            assert_eq!(GetIpoptComplInf(p), 0.0);
            FreeIpoptProblem(p);
        }
    }

    #[test]
    fn post_solve_stats_populated_after_solve() {
        // Reuse the same quadratic as the end-to-end solve test.
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
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
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        // After a successful solve, iter count is recorded (>= 0) and
        // wall time is non-negative; primal/dual/compl norms exist.
        unsafe {
            assert!(GetIpoptIterCount(p) >= 0);
            assert!(GetIpoptSolveTime(p) >= 0.0);
            assert!(GetIpoptPrimalInf(p).is_finite());
            assert!(GetIpoptDualInf(p).is_finite());
            assert!(GetIpoptComplInf(p).is_finite());
            FreeIpoptProblem(p);
        }
    }

    /// The linear-solver post-mortem is reported for a real solve, and
    /// reports the backend that actually ran rather than the one the
    /// option asked for — which is the question a caller pinning
    /// `linear_solver` is really asking.
    #[test]
    fn linear_solver_stats_populated_after_solve() {
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        let mut stats = unsafe { std::mem::zeroed::<PounceLinearSolverStats>() };
        // Nothing to report before the first solve.
        assert_eq!(unsafe { GetPounceLinearSolverStats(p, &mut stats) }, FALSE);

        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
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
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        assert_eq!(unsafe { GetPounceLinearSolverStats(p, &mut stats) }, TRUE);

        let name = unsafe { CStr::from_ptr(stats.solver_name.as_ptr()) }
            .to_str()
            .expect("solver name is ASCII");
        assert_eq!(name, "feral", "default backend should report itself");
        assert!(stats.n_factors > 0, "n_factors = {}", stats.n_factors);
        assert_eq!(
            stats.n_pattern_reuse + stats.n_pattern_changes,
            stats.n_factors,
            "every factor is either a pattern reuse or a pattern change"
        );
        // Optional fields are either a real value or the documented
        // sentinel — never a silent zero.
        assert!(stats.max_fill_ratio.is_nan() || stats.max_fill_ratio > 0.0);
        assert!(stats.last_nnz_l == -1 || stats.last_nnz_l > 0);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn option_type_reports_the_setter_a_keyword_expects() {
        let p = create_unconstrained();
        let ty = |s: &str| {
            let c = std::ffi::CString::new(s).unwrap();
            unsafe { GetPounceOptionType(p, c.as_ptr()) }
        };
        assert_eq!(ty("tol"), 1, "tol is a number");
        assert_eq!(ty("max_iter"), 2, "max_iter is an integer");
        assert_eq!(ty("linear_solver"), 3, "linear_solver is a string");
        // `hessian_approximation` is the option the CasADi plugin has to
        // set as a string while the user may well type it as one too.
        assert_eq!(ty("hessian_approximation"), 3);
        // Unregistered, and NULL keyword, both answer "unknown" rather
        // than guessing a type.
        assert_eq!(ty("no_such_option_at_all"), 0);
        assert_eq!(unsafe { GetPounceOptionType(p, std::ptr::null()) }, 0);
        unsafe { FreeIpoptProblem(p) };
    }

    /// A NULL problem handle answers from the same registry — the case a
    /// code generator is in, having no problem to ask.
    #[test]
    fn option_type_answers_without_a_problem_handle() {
        let ty = |s: &str| {
            let c = std::ffi::CString::new(s).unwrap();
            unsafe { GetPounceOptionType(std::ptr::null_mut(), c.as_ptr()) }
        };
        assert_eq!(ty("tol"), 1);
        assert_eq!(ty("max_iter"), 2);
        assert_eq!(ty("linear_solver"), 3);
        assert_eq!(ty("no_such_option_at_all"), 0);

        // …and the same answers a live problem gives, which is the
        // property that keeps the two paths from drifting apart.
        let p = create_unconstrained();
        for name in [
            "tol",
            "max_iter",
            "linear_solver",
            "mu_strategy",
            "print_level",
        ] {
            let c = std::ffi::CString::new(name).unwrap();
            assert_eq!(
                unsafe { GetPounceOptionType(std::ptr::null_mut(), c.as_ptr()) },
                unsafe { GetPounceOptionType(p, c.as_ptr()) },
                "handle-free and problem-bound disagree on {name}"
            );
        }
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn write_solve_report_emits_v1_json_with_iter_history() {
        // Quadratic — Newton driver, single iter; just exercises the
        // post-solve report path end-to-end.
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };

        // Write before any solve must fail.
        let cpath = CString::new("/tmp/pounce_cinterface_no_solve.json").unwrap();
        let bad = unsafe { IpoptWriteSolveReport(p, cpath.as_ptr(), std::ptr::null()) };
        assert_eq!(bad, FALSE);

        // Enable per-iter capture, solve, then write at detail = full.
        assert_eq!(unsafe { IpoptEnableIterHistory(p) }, TRUE);
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
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
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);

        let dir = std::env::temp_dir();
        let path = dir.join("pounce_cinterface_report.json");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let cdetail = CString::new("full").unwrap();
        let ok = unsafe { IpoptWriteSolveReport(p, cpath.as_ptr(), cdetail.as_ptr()) };
        assert_eq!(ok, TRUE);

        // Read it back and check the schema tag + that it parses with
        // the same struct shape pounce-cli uses.
        let txt = std::fs::read_to_string(&path).unwrap();
        assert!(
            txt.contains("\"schema\": \"pounce.solve-report/v1\""),
            "{txt}"
        );
        assert!(txt.contains("\"kind\": \"tnlp-direct\""));
        let parsed: pounce_solve_report::SolveReport = serde_json::from_str(&txt).unwrap();
        assert_eq!(parsed.problem.n_variables, 1);
        assert_eq!(parsed.problem.n_constraints, 0);

        // Invalid detail string is rejected.
        let bad_detail = CString::new("verbose").unwrap();
        let bad = unsafe { IpoptWriteSolveReport(p, cpath.as_ptr(), bad_detail.as_ptr()) };
        assert_eq!(bad, FALSE);

        let _ = std::fs::remove_file(&path);
        unsafe { FreeIpoptProblem(p) };
    }

    // --- Intermediate-callback wiring (issue #19, follow-up) ---
    //
    // The callback only fires on the IPM path (`optimize_constrained`).
    // Unconstrained problems short-circuit through the Newton driver,
    // so these tests use a single-inequality problem to force the IPM.

    unsafe extern "C" fn cb_quad_eval_g(
        _n: Index,
        x: *const Number,
        _new_x: Bool,
        _m: Index,
        g: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            *g.offset(0) = *x.offset(0);
            TRUE
        }
    }
    unsafe extern "C" fn cb_quad_eval_jac_g(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        _m: Index,
        nele_jac: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            assert_eq!(nele_jac, 1);
            if !irow.is_null() {
                *irow.offset(0) = 0;
                *jcol.offset(0) = 0;
            }
            if !values.is_null() {
                *values.offset(0) = 1.0;
            }
            TRUE
        }
    }
    unsafe extern "C" fn cb_quad_eval_h(
        _n: Index,
        _x: *const Number,
        _new_x: Bool,
        obj_factor: Number,
        _m: Index,
        _lambda: *const Number,
        _new_lambda: Bool,
        _nele_hess: Index,
        irow: *mut Index,
        jcol: *mut Index,
        values: *mut Number,
        _user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            if !irow.is_null() {
                *irow.offset(0) = 0;
                *jcol.offset(0) = 0;
            }
            if !values.is_null() {
                *values.offset(0) = 2.0 * obj_factor;
            }
            TRUE
        }
    }

    fn create_callback_test_problem() -> IpoptProblem {
        // min (x - 2)^2  s.t.  -10 <= x <= 10 (single inequality).
        let xl = [-1.0e20];
        let xu = [1.0e20];
        let gl = [-10.0];
        let gu = [10.0];
        unsafe {
            CreateIpoptProblem(
                1,
                xl.as_ptr(),
                xu.as_ptr(),
                1,
                gl.as_ptr(),
                gu.as_ptr(),
                1,
                1,
                0,
                Some(quad_eval_f),
                Some(cb_quad_eval_g),
                Some(quad_eval_grad_f),
                Some(cb_quad_eval_jac_g),
                Some(cb_quad_eval_h),
            )
        }
    }

    static CB_ITER_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static CB_LAST_ITER: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);
    static CB_INSPECTOR_OK: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    unsafe extern "C" fn counting_cb(
        _alg_mod: Index,
        iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
        user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            CB_ITER_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            CB_LAST_ITER.store(iter_count, std::sync::atomic::Ordering::SeqCst);
            // user_data carries the IpoptProblem so we can exercise the
            // inspector from inside the callback.
            let problem = user_data as IpoptProblem;
            let mut x = [0.0_f64];
            let rc = GetIpoptCurrentIterate(
                problem,
                FALSE,
                1,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if rc == TRUE && x[0].is_finite() {
                CB_INSPECTOR_OK.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            TRUE
        }
    }

    #[test]
    fn intermediate_callback_fires_per_iteration_and_inspector_reads_x() {
        CB_ITER_COUNTER.store(0, std::sync::atomic::Ordering::SeqCst);
        CB_LAST_ITER.store(-1, std::sync::atomic::Ordering::SeqCst);
        CB_INSPECTOR_OK.store(false, std::sync::atomic::Ordering::SeqCst);

        let p = create_callback_test_problem();
        assert!(!p.is_null());
        let ok = unsafe { SetIntermediateCallback(p, Some(counting_cb)) };
        assert_eq!(ok, TRUE);
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut obj,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                p as *mut c_void,
            )
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        // At least the iter-0 fire happened, plus one per accepted step.
        let n_fires = CB_ITER_COUNTER.load(std::sync::atomic::Ordering::SeqCst);
        assert!(n_fires >= 2, "callback fired {n_fires} times, want >=2");
        assert!(
            CB_LAST_ITER.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "last iter should be >= 1 after at least one accepted step"
        );
        assert!(
            CB_INSPECTOR_OK.load(std::sync::atomic::Ordering::SeqCst),
            "GetIpoptCurrentIterate did not return a usable x"
        );
        unsafe { FreeIpoptProblem(p) };
    }

    static CB_VIOL_OK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    // Bounded variant of `create_callback_test_problem`: x in [0, 10] with a
    // finite lower bound, so the `x_l_violation` / `x_u_violation` branches of
    // GetIpoptCurrentViolations actually scatter a real `x_L`/`x_U` mapping
    // (not the degenerate "no bound" pack).
    fn create_bounded_callback_test_problem() -> IpoptProblem {
        // min (x - 2)^2  s.t.  -10 <= x <= 10,  x in [0, 10].
        let xl = [0.0];
        let xu = [10.0];
        let gl = [-10.0];
        let gu = [10.0];
        unsafe {
            CreateIpoptProblem(
                1,
                xl.as_ptr(),
                xu.as_ptr(),
                1,
                gl.as_ptr(),
                gu.as_ptr(),
                1,
                1,
                0,
                Some(quad_eval_f),
                Some(cb_quad_eval_g),
                Some(quad_eval_grad_f),
                Some(cb_quad_eval_jac_g),
                Some(cb_quad_eval_h),
            )
        }
    }

    unsafe extern "C" fn violations_inspecting_cb(
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
        user_data: *mut c_void,
    ) -> Bool {
        unsafe {
            let problem = user_data as IpoptProblem;
            // Exercise the bound-violation branches (n=1, m=1) from inside an
            // installed intermediate context. Pre-L51 these branches indexed
            // `v[i]` without a length guard; the fix makes them return FALSE on
            // a packed-length mismatch instead of panicking across `extern "C"`.
            let mut x_l_viol = [f64::NAN];
            let mut x_u_viol = [f64::NAN];
            let rc = GetIpoptCurrentViolations(
                problem,
                FALSE,
                1,
                x_l_viol.as_mut_ptr(),
                x_u_viol.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if rc == TRUE
                && x_l_viol[0].is_finite()
                && x_l_viol[0] >= 0.0
                && x_u_viol[0].is_finite()
                && x_u_viol[0] >= 0.0
            {
                CB_VIOL_OK.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            TRUE
        }
    }

    #[test]
    fn get_current_violations_inside_callback_reports_finite_bounds() {
        CB_VIOL_OK.store(false, std::sync::atomic::Ordering::SeqCst);
        let p = create_bounded_callback_test_problem();
        assert!(!p.is_null());
        let ok = unsafe { SetIntermediateCallback(p, Some(violations_inspecting_cb)) };
        assert_eq!(ok, TRUE);
        let mut x = [5.0_f64];
        let mut obj = 0.0_f64;
        let rc = unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut obj,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                p as *mut c_void,
            )
        };
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        assert!(
            CB_VIOL_OK.load(std::sync::atomic::Ordering::SeqCst),
            "GetIpoptCurrentViolations did not return finite, non-negative \
             bound violations from inside the callback"
        );
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn bound_violation_scatter_rejects_oversized_pack_instead_of_panicking() {
        // L51 fail-first (logic level): reproduce the scatter of the
        // `x_l_violation` / `x_u_violation` branches. The packed vector comes
        // from `pack_z_*_for_user`, whose length must equal the output `n`.
        // Pre-fix the branches scattered it with `for (i, s) in
        // packed.enumerate() { v[i] = ... }` over a `vec![0.0; n]` *without*
        // checking the length; an oversized pack indexes `v[i]` out of bounds
        // and panics — and across the real `extern "C"` boundary that panic
        // aborts the embedding process. The fix adds the same length guard
        // the sibling (`compl_*`, `grad_lag_x`) branches already had.
        let n_us = 1usize;
        let packed = vec![0.5_f64, -0.3]; // len 2 != n_us == 1

        // Pre-fix: the unguarded scatter panics on the oversized pack.
        let unguarded = std::panic::catch_unwind(|| {
            let mut v = vec![0.0; n_us];
            for (i, s) in packed.iter().enumerate() {
                v[i] = (-s).max(0.0);
            }
            v
        });
        assert!(
            unguarded.is_err(),
            "unguarded scatter should panic (→ abort across extern \"C\") on an oversized pack"
        );

        // Post-fix: the length guard returns an error instead of panicking.
        let guarded: Result<Vec<f64>, ()> = (|| {
            if packed.len() != n_us {
                return Err(());
            }
            let mut v = vec![0.0; n_us];
            for (i, s) in packed.iter().enumerate() {
                v[i] = (-s).max(0.0);
            }
            Ok(v)
        })();
        assert!(
            guarded.is_err(),
            "guarded scatter should reject the length mismatch (return FALSE), not panic"
        );
    }

    unsafe extern "C" fn user_stop_cb(
        _alg_mod: Index,
        _iter_count: Index,
        _obj_value: Number,
        _inf_pr: Number,
        _inf_du: Number,
        _mu: Number,
        _d_norm: Number,
        _regularization_size: Number,
        _alpha_du: Number,
        _alpha_pr: Number,
        _ls_trials: Index,
        _user_data: *mut c_void,
    ) -> Bool {
        FALSE
    }

    #[test]
    fn intermediate_callback_false_surfaces_user_requested_stop() {
        let p = create_callback_test_problem();
        assert!(!p.is_null());
        let ok = unsafe { SetIntermediateCallback(p, Some(user_stop_cb)) };
        assert_eq!(ok, TRUE);
        let mut x = [0.0_f64];
        let rc = unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, ApplicationReturnStatus::UserRequestedStop as Index);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn ffi_guard_converts_panic_to_fallback() {
        // L56: a panic in pounce's own Rust code during a solve must be
        // caught at the FFI boundary and reported as `Internal_Error`, never
        // unwound across `extern "C"` (which aborts the embedding process).
        // This exercises the exact mechanism wrapping IpoptSolve /
        // IpoptSolveWarmStart. (The "boom" panic message printing to stderr
        // is expected — the default panic hook still runs before the catch.)
        let fallback = ApplicationReturnStatus::InternalError as Index;
        let got = ffi_guard(fallback, || -> Index {
            panic!("boom inside solver core");
        });
        assert_eq!(got, fallback);
        assert_eq!(got, ApplicationReturnStatus::InternalError as Index);
    }

    #[test]
    fn ffi_guard_is_transparent_on_success() {
        // On the happy path the guard returns the body's value unchanged, so
        // wrapping IpoptSolve does not alter normal solves (the end-to-end
        // solve tests above confirm this at the public-API level).
        let got = ffi_guard(-99, || 7);
        assert_eq!(got, 7);
    }

    #[test]
    fn parse_pkg_version_handles_missing_components() {
        assert_eq!(parse_pkg_version("1.2.3"), (1, 2, 3));
        assert_eq!(parse_pkg_version("4.5"), (4, 5, 0));
        assert_eq!(parse_pkg_version(""), (0, 0, 0));
        assert_eq!(parse_pkg_version("1.x.3"), (1, 0, 3));
    }

    // ---- Solver-session C ABI (crate::solver) ----

    use crate::solver::{
        IpoptCreateSolver, IpoptFreeSolver, IpoptSolverGetKktDim, IpoptSolverKktSolve,
        IpoptSolverSolve,
    };

    #[test]
    fn solver_create_consumes_problem_handle() {
        let mut p = create_unconstrained();
        assert!(!p.is_null());
        let s = unsafe { IpoptCreateSolver(&mut p) };
        assert!(!s.is_null());
        assert!(
            p.is_null(),
            "IpoptCreateSolver should NULL out the caller's handle"
        );
        unsafe { IpoptFreeSolver(s) };
    }

    #[test]
    fn solver_create_null_inputs_return_null() {
        // NULL pointer-to-handle.
        let s = unsafe { IpoptCreateSolver(std::ptr::null_mut()) };
        assert!(s.is_null());
        // Pointer to a NULL handle.
        let mut p: IpoptProblem = std::ptr::null_mut();
        let s = unsafe { IpoptCreateSolver(&mut p) };
        assert!(s.is_null());
    }

    #[test]
    fn solver_free_null_is_safe() {
        unsafe { IpoptFreeSolver(std::ptr::null_mut()) };
    }

    #[test]
    fn solver_solve_drives_quadratic_and_retains_factor() {
        let xl = [-1.0e20];
        let xu = [1.0e20];
        let mut p = unsafe {
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
                Some(quad_eval_f),
                None,
                Some(quad_eval_grad_f),
                None,
                Some(quad_eval_h),
            )
        };
        assert!(!p.is_null());
        let s = unsafe { IpoptCreateSolver(&mut p) };
        assert!(!s.is_null());
        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
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
        assert_eq!(rc, ApplicationReturnStatus::SolveSucceeded as Index);
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!(obj.abs() < 1e-10);

        // After convergence the factor is retained — kkt_dim is positive
        // and a zero RHS back-solves to zero.
        let dim = unsafe { IpoptSolverGetKktDim(s) };
        assert!(dim > 0, "expected positive KKT dim, got {dim}");
        let rhs = vec![0.0_f64; dim as usize];
        let mut lhs = vec![1.0_f64; dim as usize];
        let ok = unsafe { IpoptSolverKktSolve(s, rhs.as_ptr(), lhs.as_mut_ptr()) };
        assert_eq!(ok, TRUE);
        for (i, v) in lhs.iter().enumerate() {
            assert!(v.abs() < 1e-10, "lhs[{i}] = {v} not ~0");
        }
        unsafe { IpoptFreeSolver(s) };
    }

    #[test]
    fn solver_kkt_dim_minus_one_before_solve() {
        let mut p = create_unconstrained();
        let s = unsafe { IpoptCreateSolver(&mut p) };
        assert_eq!(unsafe { IpoptSolverGetKktDim(s) }, -1);
        unsafe { IpoptFreeSolver(s) };
    }

    // ─────────────────────────────────────────────────────────
    // §7.2 SQP working-set warm-start C ABI tests.
    // ─────────────────────────────────────────────────────────

    #[test]
    fn c_get_working_set_returns_false_before_any_solve() {
        let p = create_unconstrained();
        let mut bound_buf = [0; 4];
        let rc = unsafe { IpoptGetWorkingSet(p, bound_buf.as_mut_ptr(), std::ptr::null_mut()) };
        assert_eq!(rc, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn c_set_warm_start_with_both_null_returns_false() {
        let p = create_unconstrained();
        let rc = unsafe { IpoptSetWarmStartWorkingSet(p, std::ptr::null(), std::ptr::null()) };
        assert_eq!(rc, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn c_set_warm_start_with_bad_status_code_returns_false() {
        let p = create_unconstrained();
        // Length n = 4; '7' is out of range (valid: 0..=3).
        let bogus = [
            POUNCE_WS_INACTIVE,
            7,
            POUNCE_WS_AT_LOWER,
            POUNCE_WS_INACTIVE,
        ];
        let rc = unsafe { IpoptSetWarmStartWorkingSet(p, bogus.as_ptr(), std::ptr::null()) };
        assert_eq!(rc, FALSE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn c_set_warm_start_then_clear_succeeds() {
        let p = create_unconstrained();
        let in_buf = [POUNCE_WS_INACTIVE; 4];
        let set_rc = unsafe { IpoptSetWarmStartWorkingSet(p, in_buf.as_ptr(), std::ptr::null()) };
        assert_eq!(set_rc, TRUE);
        let clr_rc = unsafe { IpoptClearWarmStartWorkingSet(p) };
        assert_eq!(clr_rc, TRUE);
        unsafe { FreeIpoptProblem(p) };
    }

    #[test]
    fn c_set_warm_start_on_null_problem_returns_false() {
        let in_buf = [POUNCE_WS_INACTIVE; 1];
        let rc = unsafe {
            IpoptSetWarmStartWorkingSet(std::ptr::null_mut(), in_buf.as_ptr(), std::ptr::null())
        };
        assert_eq!(rc, FALSE);
    }

    #[test]
    fn c_solve_warm_start_round_trips_working_set_on_sqp_path() {
        // Use the 1-D `(x − 2)²` quadratic from
        // `create_callback_test_problem`. Set `algorithm
        // active-set-sqp`, solve, then read the working set
        // through `IpoptGetWorkingSet`. Pass it back via
        // `IpoptSolveWarmStart` for a second solve.
        let p = create_callback_test_problem();
        let key = CString::new("algorithm").unwrap();
        let val = CString::new("active-set-sqp").unwrap();
        let ok = unsafe { AddIpoptStrOption(p, key.as_ptr(), val.as_ptr()) };
        assert_eq!(ok, TRUE);

        let mut x = [0.0_f64];
        let mut obj = 0.0_f64;
        let rc1 = unsafe {
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
        };
        assert_eq!(rc1, ApplicationReturnStatus::SolveSucceeded as Index);

        let mut bound_buf = [-1; 1];
        let mut cons_buf = [-1; 1];
        let got = unsafe { IpoptGetWorkingSet(p, bound_buf.as_mut_ptr(), cons_buf.as_mut_ptr()) };
        assert_eq!(got, TRUE);
        // Status codes must be in 0..=3.
        assert!((0..=3).contains(&bound_buf[0]));
        assert!((0..=3).contains(&cons_buf[0]));

        // Second solve with the just-retrieved working set as
        // input. Resets x to a non-optimal starting point so the
        // SQP loop actually has work to do; the warm-start
        // should still converge to the optimum.
        x[0] = 0.0;
        let mut obj2 = 0.0_f64;
        let mut bound_out = [-1; 1];
        let mut cons_out = [-1; 1];
        let rc2 = unsafe {
            IpoptSolveWarmStart(
                p,
                x.as_mut_ptr(),
                std::ptr::null_mut(),
                &mut obj2,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                bound_buf.as_ptr(),
                cons_buf.as_ptr(),
                bound_out.as_mut_ptr(),
                cons_out.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc2, ApplicationReturnStatus::SolveSucceeded as Index);
        assert!((0..=3).contains(&bound_out[0]));
        assert!((0..=3).contains(&cons_out[0]));

        unsafe { FreeIpoptProblem(p) };
    }
}
