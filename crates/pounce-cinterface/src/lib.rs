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

pub mod bounds;
pub mod fortran;
pub mod qp;
pub mod solver;

pub use bounds::*;
pub use qp::*;

use bounds::{ComplementsLedger, ComplementsRow, ComplementsUse, DeclaredPairs};
use pounce_algorithm::application::{
    IpoptApplication, default_backend_factory, feral_config_from_options, ma57_config_from_options,
};
use pounce_algorithm::intermediate as ip_intermediate;
use pounce_common::reg_options::OptionType;
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_nlp::mpcc::{
    ComplementarityPair, PairErrorKind, PairShape, classify_pair, validate_pairs,
};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use pounce_restoration::second_opinion_driver::run_second_opinion_ladder;
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

/// An *extensible* caller-allocated output struct (gh#1147): its first
/// member is `ipindex struct_size`, which the caller sets to `sizeof(T)`
/// before the call. `PounceComplementarityStats`, `PounceComplementaritySolve`,
/// `PounceQpResiduals` and `PounceQpCertificate` follow it;
/// `PounceLinearSolverStats` does not (it shipped before the convention).
///
/// # The growth rule
///
/// A later release may only **append** fields at the end; nothing is
/// removed, reordered or widened. Every getter then writes
/// `min(caller's struct_size, its own size_of::<T>())` bytes — which is what
/// `write_extensible` does for this release, whose size is the minimum
/// whenever the call is accepted at all.
///
/// # Safety
///
/// Implementors must be `#[repr(C)]` with `struct_size: Index` as the first
/// field, so it sits at offset 0 (asserted by
/// `tests/c_examples_build_and_run.rs`).
pub(crate) unsafe trait ExtensibleOut: Copy {}

/// This release's `sizeof(T)`, as the `struct_size` value a caller passes.
pub(crate) const fn struct_size_of<T: ExtensibleOut>() -> Index {
    std::mem::size_of::<T>() as Index
}

/// Write `value` into the caller's `out` under the `struct_size` contract.
///
/// Returns `false`, writing nothing at all, when the caller's
/// `struct_size` is smaller than this release's `size_of::<T>()` (a
/// forgotten or garbage initializer: `0`, negative, or a size from a
/// release that predates a field of `T` — there is none yet). Otherwise
/// copies **exactly this release's bytes** of `value` after `struct_size`
/// and returns `true`; `struct_size` itself is read, never written, and no
/// byte past `size_of::<T>()` is touched, however large `struct_size` is
/// (a caller built against a newer header keeps its newer fields as they
/// were).
///
/// The copy is a bounded byte copy, not `*out = value`, on purpose: a
/// whole-struct assignment is a contract about *this* release's size that
/// would silently stay one when a field is appended. When appending a field
/// to `T` in a later release, keep this function, and change the refusal
/// below to the size of the oldest layout still accepted (the size *before*
/// the first appended field), writing `min(struct_size, size_of::<T>())`
/// bytes; never lower the floor to accept `struct_size` values a getter
/// cannot tell from an uninitialized one.
///
/// # Safety
///
/// `out` must be non-NULL, aligned for `T`, readable for its first
/// `size_of::<Index>()` bytes and, when the call is accepted, writable for
/// `size_of::<T>()` bytes (the caller promised `struct_size` of them).
pub(crate) unsafe fn write_extensible<T: ExtensibleOut>(out: *mut T, value: &T) -> bool {
    const HEAD: usize = std::mem::size_of::<Index>();
    let size = std::mem::size_of::<T>();
    // SAFETY: `struct_size` is at offset 0 (the trait's contract) and the
    // caller guarantees the first `HEAD` bytes are readable.
    let claimed = unsafe { std::ptr::read(out.cast::<Index>()) };
    if usize::try_from(claimed).map_or(true, |c| c < size) {
        return false;
    }
    // SAFETY: `claimed >= size`, so the caller promised `size` writable
    // bytes; `value` is a distinct Rust value, so the ranges cannot overlap.
    // Only bytes `HEAD..size` are written: `struct_size` stays the caller's.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (value as *const T).cast::<u8>().add(HEAD),
            out.cast::<u8>().add(HEAD),
            size - HEAD,
        );
    }
    true
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
    /// Complementarity pairs declared by [`AddPounceComplementarityPair`] /
    /// [`SetPounceComplementarityPairs`] (gh#1147), **0-based** whatever
    /// the problem's index style, in declaration order. Every stored set
    /// passed [`pounce_nlp::mpcc::validate_pairs`] against this handle's
    /// bounds when it was declared. Empty — the default — means no pairs.
    pub(crate) complementarity: DeclaredPairs,
    /// Every `(row, var)`, 0-based, that [`AddPounceComplementsConstraint`]
    /// accepted on this handle, whether or not it declared a pair. The
    /// helper refuses a row or variable already in here (or in
    /// `complementarity`), so one `F` or one `x` cannot be in two MOI
    /// `Complements` constraints. Never cleared: the bound edits it records
    /// are not undone either.
    pub(crate) complements_used: ComplementsLedger,
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
    /// The complementarity driver's record when the solve declared pairs
    /// (gh#1147); `None` on every other solve, including every session solve.
    pub(crate) mpcc: Option<pounce_restoration::mpcc_driver::MpccOutcome>,
    /// The second-opinion ladder's outcome when the solve took the path
    /// without pairs (gh#850, gh#1147); `None` on a driver solve and on
    /// every session solve, neither of which runs the ladder. Kept whether
    /// or not a rung ran: [`IpoptWriteSolveReport`] writes the block under
    /// the same `ran()` test as the CLI and Python.
    pub(crate) second_opinion:
        Option<pounce_restoration::second_opinion_driver::SecondOpinionOutcome>,
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
            mpcc: None,
            second_opinion: None,
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
            complementarity: DeclaredPairs::default(),
            complements_used: ComplementsLedger::default(),
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
/// With complementarity pairs declared (gh#1147) the bridge goes to
/// [`pounce_restoration::mpcc_driver::solve_mpcc`] instead of one
/// `optimize_tnlp` and the second-opinion ladder, as in the CLI and Python;
/// the driver forwards its reported point to the bridge once, so the
/// write-back is the same. A staged SQP working set plus pairs is refused
/// with `Invalid_Problem_Definition` before any callback.
///
/// # Safety
///
/// All pointer arguments are read/written per the
/// `IpStdCInterface.h` contract: `x` is in/out (size `n`); `g` is
/// out-only (size `m`); `mult_g`, `mult_x_L`, `mult_x_U` (sizes
/// `m, n, n`) are out, and are also *read* as the starting multipliers
/// when `warm_start_init_point=yes` (upstream's `start_lam`, `start_z_L`,
/// `start_z_U`). Any of them may be NULL; a NULL multiplier buffer under
/// a warm start means "not supplied" and that block starts unseeded.
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
            // Every solve states, for its own duration, whether the live
            // iterate on this thread is a lowered model's (gh#1147 review):
            // a pair-free solve nested inside a lowered solve's callback
            // clears the flag, so its own inspectors answer, and the outer
            // value comes back when it returns.
            let _lowered = LoweredSolveGuard::set(!info.complementarity.is_empty());
            if info.n < 0 || info.m < 0 {
                return ApplicationReturnStatus::InvalidProblemDefinition as Index;
            }
            if info.n > 0 && x.is_null() {
                return ApplicationReturnStatus::InvalidProblemDefinition as Index;
            }
            // A staged SQP working set is over the caller's rows and
            // variables; the complementarity driver solves a lowered model
            // with more rows (and possibly lifted variables), so there is no
            // set it could seed. Refused before any callback runs, and
            // neither the working set nor the pairs are consumed (gh#1147).
            if !info.complementarity.is_empty() && info.pending_working_set.is_some() {
                if pounce_algorithm::second_opinion::narration_is_wanted(info.app.options()) {
                    eprintln!(
                        "pounce: a warm-start working set cannot seed a solve with \
                         complementarity pairs ({} declared); clear one of them \
                         (IpoptClearWarmStartWorkingSet or ClearPounceComplementarityPairs).",
                        info.complementarity.len()
                    );
                }
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
                // `lambda_x = z_l − z_u` (see `sqp::warm_start`). All three
                // stay in the caller's space (`g` order, unscaled, full `n`);
                // `set_sqp_warm_start` maps them into the algorithm's.
                let lambda_x = z_l.iter().zip(&z_u).map(|(l, u)| l - u).collect();
                info.app
                    .set_sqp_warm_start(pounce_algorithm::sqp::SqpIterates {
                        x: initial_x.clone(),
                        lambda_g,
                        lambda_x,
                        working: Some(working),
                    });
            }

            let (lower_inf, upper_inf, zero_fixed_grad_lag) =
                bridge_violation_options(info.app.options());
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
                start_lambda: mult_g as *const Number,
                start_z_l: mult_x_L as *const Number,
                start_z_u: mult_x_U as *const Number,
                eval_f: info.eval_f,
                eval_grad_f: info.eval_grad_f,
                eval_g: info.eval_g,
                eval_jac_g: info.eval_jac_g,
                eval_h: info.eval_h,
                user_data,
                intermediate_cb: info.intermediate_cb,
                user_scaling: info.user_scaling.clone(),
                nonlinear_vars: info.nonlinear_vars.clone(),
                lower_inf,
                upper_inf,
                zero_fixed_grad_lag,
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
            // overrides flow into the restoration sub-IPM too. Use the multi-pass
            // provider so the ℓ₁ wrapper / auto-fallback don't panic on the
            // second inner solve (pounce#10 Phase 3 / pounce#24).
            let feral_cfg = feral_config_from_options(info.app.options());
            // The `ma57_*` options under the `"resto."` prefix — dead until
            // gh#825, because nothing threaded any MA57 config into a factory.
            let ma57_cfg = ma57_config_from_options(info.app.options(), "resto.");
            let bff_mint = move || -> InnerBackendFactoryFactory {
                let feral_cfg = feral_cfg.clone();
                let ma57_cfg = ma57_cfg.clone();
                Box::new(move || default_backend_factory(feral_cfg.clone(), ma57_cfg.clone()))
            };
            let resto_provider = make_default_restoration_factory_provider(
                RestoAlgorithmBuilder::new(),
                info.app.algorithm_builder_from_options(),
                bff_mint,
            );
            info.app.set_restoration_factory_provider(resto_provider);

            let bridge_for_solve: Rc<RefCell<dyn TNLP>> = bridge.clone();
            // The run-ending `EXIT:` / `POUNCE <version>:` verdict belongs to the
            // whole run, not to each attempt. Deferred from here through the
            // second-opinion ladder below, which releases it and prints it once
            // with the status that actually ships. Without this, every retry
            // driver's attempt printed its own verdict and a run that recovered
            // reported a mid-run one that read as the final answer.
            info.app.defer_end_verdict();
            // Declared complementarity (gh#1147): the driver chooses the
            // formulation, runs the lowered solves and certifies the answer,
            // as in the CLI and Python. It replaces the single solve and the
            // second-opinion ladder, which both assume one solve of one model,
            // and forwards the reported point to the bridge's
            // `finalize_solution` exactly once, so the copy-back below is the
            // same for both paths.
            let (status, statistics, mpcc, second_opinion) = if !info.complementarity.is_empty() {
                let narrate =
                    pounce_algorithm::second_opinion::narration_is_wanted(info.app.options());
                let result = pounce_restoration::mpcc_driver::solve_mpcc(
                    &mut info.app,
                    bridge_for_solve,
                    info.complementarity.to_vec(),
                    &mut |line| {
                        if narrate {
                            eprintln!("{line}");
                        }
                    },
                );
                let out = match result {
                    Ok(out) => out,
                    // Unreachable from C: the only `Err` is `MpccTnlp::with_lift`
                    // refusing the pairs, and the stored set passed the same
                    // validator against the same bounds at declaration. Kept
                    // as a refusal rather than an assumption.
                    Err(e) => {
                        if narrate {
                            eprintln!(
                                "pounce: cannot solve this model's complementarity constraints: {e}"
                            );
                        }
                        let status = ApplicationReturnStatus::InvalidProblemDefinition;
                        if info.app.release_end_verdict() {
                            info.app.print_end_verdict(status);
                        }
                        return status as Index;
                    }
                };
                if narrate {
                    eprintln!(
                        "{}",
                        pounce_algorithm::mpcc::describe_outcome(
                            out.reported,
                            out.solves.len(),
                            out.total_iterations,
                            out.max_abs_product,
                            out.certificate.as_ref(),
                        )
                    );
                }
                if info.app.release_end_verdict() {
                    info.app.print_end_verdict(out.status);
                }
                (out.status, out.statistics.clone(), Some(out), None)
            } else {
                // The path without pairs, unchanged: one solve, then the ladder.
                let status = info.app.optimize_tnlp(bridge_for_solve);
                let stats = info.app.statistics();
                // Second-opinion ladder, on by default as it is in the CLI and the
                // Python frontend: an `Infeasible_Problem_Detected` or
                // `Invalid_Number_Detected` is re-solved along up to three
                // deliberately different trajectories and a re-solve is promoted
                // only if it converges. A converged solve pays nothing — the
                // ladder reads the status and returns. The three `*_retry`
                // options turn individual rungs off.
                //
                // Narration goes to stderr, where the solver's own banners already
                // go -- but gated on `print_level >= 1`. This crate is the Ipopt
                // drop-in, so `print_level=0 sb=yes` is a caller asking for
                // silence, and eight unexpected `pounce:` lines on a failing
                // solve is exactly what that asks not to happen. The ladder still runs;
                // only the console is quiet.
                let narrate =
                    pounce_algorithm::second_opinion::narration_is_wanted(info.app.options());
                let ladder = run_second_opinion_ladder(
                    &mut info.app,
                    bridge.clone() as Rc<RefCell<dyn TNLP>>,
                    status,
                    stats,
                    &mut |line| {
                        if narrate {
                            eprintln!("{line}");
                        }
                    },
                );
                (ladder.status, ladder.statistics.clone(), None, Some(ladder))
            };
            let bridge_ref = bridge.borrow();
            info.last_solve = Some(LastSolve {
                stats: statistics,
                status,
                linear_solver: info.app.linear_solver_summary(),
                final_x: bridge_ref.final_x.clone(),
                final_lambda: bridge_ref.final_lambda.clone(),
                final_obj: bridge_ref.final_obj,
                mpcc,
                second_opinion,
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
/// `scaled` keeps upstream's meaning (`TNLP::get_curr_iterate`, Ipopt
/// 3.14 `IpTNLP.cpp`), on both the interior-point and the active-set SQP
/// arm:
///
/// * `FALSE` — the user TNLP's unscaled space, the units `IpoptSolve`
///   writes back: objective, constraint and per-variable
///   (`SetIpoptProblemScaling` `x_scaling`) scaling all undone.
/// * `TRUE` — the algorithm's scaled space in the TNLP's shape: `x̃ = d ⊙ x`
///   (a fixed variable at its own value, as upstream's `ResortX` writes
///   it), `z̃ = obj·z/d` (at a fixed variable `obj·z`, `z_L`/`z_U` swapped
///   when `obj < 0`, upstream's manual rescale of `ResortBoundMultipliers`),
///   `g̃ = s ⊙ g` (row factor; equality right-hand sides scaled too) and
///   `λ̃ = obj·λ/s` ([`pounce_nlp::live_iterate::scaled_iterate`]). With
///   presolve on, the scaled values of the problem the algorithm solved,
///   lifted to the user's shape like the unscaled ones: a column or row the
///   algorithm holds carries its factor, one presolve removed carries
///   factor `1` (a multiplier there still carries `obj`).
///
/// Both arms answer from the iterate the algorithm materialises before
/// each fire ([`pounce_algorithm::intermediate::UserIterate`]), built by
/// the conversions `finalize_solution` applies — so at the last fire this
/// reads what `IpoptSolve` writes back. That includes the bound
/// multipliers of fixed variables, recovered from `∇f + Jᵀλ` as upstream's
/// `TNLPAdapter::ResortBoundMultipliers` does (not under
/// `fixed_variable_treatment=make_parameter_nodual`), and, with presolve on,
/// the rows and columns it removed.
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
    scaled: Bool,
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
        // During a complementarity solve the algorithm's iterate is the
        // lowered model's, not the caller's (gh#1147): no data, whatever
        // the request shape. Checked before the handle is read.
        if in_lowered_solve() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        if n != info.n || m != info.m {
            return FALSE;
        }
        // Active-set SQP arm (gh#1015): no `IpoptData`/`IpoptCq`, so the
        // callback window carries a materialised user-space iterate instead.
        if let Some(r) = ip_intermediate::with_current_user_iterate(|it| {
            let scaled_it;
            let it = if scaled != FALSE {
                scaled_it = pounce_nlp::live_iterate::scaled_iterate(it);
                &scaled_it
            } else {
                it
            };
            let n_us = n as usize;
            let m_us = m as usize;
            let copy = |src: &[Number], dst: *mut Number, len: usize| -> bool {
                if dst.is_null() || len == 0 {
                    return true;
                }
                if src.len() != len {
                    return false;
                }
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
                true
            };
            copy(&it.x, x, n_us)
                && copy(&it.z_l, z_l, n_us)
                && copy(&it.z_u, z_u, n_us)
                && copy(&it.g, g, m_us)
                && copy(&it.lambda, lambda, m_us)
        }) {
            return if r { TRUE } else { FALSE };
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
            // Per-variable `user-scaling` substitutes `x̃ = d ⊙ x` in a
            // `ScalingTnlp` below the NLP, so the iterate lifted here is
            // `x̃` / `z̃`. `finalize_solution` undoes that on the way out;
            // so must this, or the callback reads coordinates the user's
            // model never had. The multipliers go through the
            // `finalize_solution_*` lifts, not `pack_*_for_user`, for the
            // same reason: only those divide out `obj_scale_factor`, so a
            // solve with objective scaling (user or gradient-based) read
            // `z` and `λ` that many times too large here — the same
            // convention the SQP arm's iterate and `IpoptSolve` use.
            let d = ctx.nlp.borrow().variable_scaling();
            let d = d.as_deref();
            if !x.is_null() && n_us > 0 {
                let mut full_x = ctx.nlp.borrow().lift_x_to_full(&*curr.x);
                if full_x.len() != n_us {
                    return false;
                }
                pounce_nlp::scaling_tnlp::unscale_iterate(d, &mut full_x, &mut [], &mut []);
                std::ptr::copy_nonoverlapping(full_x.as_ptr(), x, n_us);
            }
            if !z_l.is_null() && n_us > 0 {
                let mut full = ctx.nlp.borrow().finalize_solution_z_l(&*curr.z_l);
                if full.len() != n_us {
                    return false;
                }
                pounce_nlp::scaling_tnlp::unscale_iterate(d, &mut [], &mut full, &mut []);
                std::ptr::copy_nonoverlapping(full.as_ptr(), z_l, n_us);
            }
            if !z_u.is_null() && n_us > 0 {
                let mut full = ctx.nlp.borrow().finalize_solution_z_u(&*curr.z_u);
                if full.len() != n_us {
                    return false;
                }
                pounce_nlp::scaling_tnlp::unscale_iterate(d, &mut [], &mut [], &mut full);
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
                    .finalize_solution_lambda(&*curr.y_c, &*curr.y_d);
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
/// `scaled` keeps upstream's meaning (`TNLP::get_curr_violations`):
///
/// * `FALSE` — the user's problem, unscaled: objective, constraint and
///   per-variable scaling undone, full `n` with fixed variables, rows in
///   `g` order. Computed from the live iterate [`GetIpoptCurrentIterate`]
///   reads (so with presolve on, the user's own rows and columns) against
///   the declared bounds: `x_L_violation = max(0, x_L − x)`,
///   `compl_x_L = (x − x_L^alg)·z_L` with `x_L^alg` the bound the
///   algorithm's slack is measured to (declared, widened by
///   `bound_relax_factor`: upstream's `curr_compl_x_L / obj`, except under
///   per-variable user scaling, where pounce relaxes in the scaled frame and
///   Ipopt in the unscaled one), `grad_lag_x = ∇f + Jᵀλ − z_L + z_U`
///   (`0` at fixed variables, as upstream's `ResortX` leaves them; `∇f` and
///   `J` come from this problem's `eval_grad_f` / `eval_jac_g`, called only
///   when `grad_lag_x` is requested), per-row violation `|g − g_L|` /
///   `max(0, g_L − g, g − g_U)` and `compl_g` with upstream's signs. Both arms.
/// * `TRUE` — upstream's `get_curr_violations(scaled = true)`: the same
///   quantities in the algorithm's scaled space, each a fixed factor of its
///   unscaled counterpart ([`pounce_nlp::live_iterate::scale_violations`]):
///   bound violations `× d`, complementarities `× obj`, `grad_lag_x
///   × obj/d`, row violations `× s` — the factors `GetIpoptCurrentIterate`
///   (`scaled = TRUE`) applies. Both arms (the SQP arm's scaled space is
///   `d` and the `obj_scaling_factor` option: it runs no other scaling),
///   and under presolve, where a removed column or row carries factor 1.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem`. Each output buffer,
/// when non-NULL, must hold at least the declared length.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetIpoptCurrentViolations(
    ipopt_problem: IpoptProblem,
    scaled: Bool,
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
        // During a complementarity solve the algorithm's iterate is the
        // lowered model's, not the caller's (gh#1147): no data, whatever
        // the request shape. Checked before the handle is read.
        if in_lowered_solve() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        if n != info.n || m != info.m {
            return FALSE;
        }
        let n_us = n as usize;
        let m_us = m as usize;
        let Some(it) = ip_intermediate::with_current_user_iterate(|it| it.clone()) else {
            return FALSE;
        };
        let bridge = CURRENT_BRIDGE.with(|c| c.get());
        if bridge.is_null() {
            return FALSE;
        }
        let bridge = &*bridge;
        if it.x.len() != n_us
            || it.z_l.len() != n_us
            || it.z_u.len() != n_us
            || it.g.len() != m_us
            || it.lambda.len() != m_us
        {
            return FALSE;
        }
        let grad_lag = if grad_lag_x.is_null() || n_us == 0 {
            None
        } else if it.grad_lag_x.len() == n_us {
            // The algorithm's own, from the evaluations it made: no
            // `eval_grad_f` / `eval_jac_g` call for it.
            Some(it.grad_lag_x.clone())
        } else {
            match bridge.user_grad_lag(&it) {
                Some(g) => Some(g),
                None => return FALSE,
            }
        };
        let bounds = pounce_nlp::live_iterate::DeclaredBounds {
            x_l: &bridge.x_l,
            x_u: &bridge.x_u,
            g_l: &bridge.g_l,
            g_u: &bridge.g_u,
            lower_inf: bridge.lower_inf,
            upper_inf: bridge.upper_inf,
        };
        let violations = Some(pounce_nlp::live_iterate::current_violations(
            &it,
            &bounds,
            grad_lag,
            bridge.zero_fixed_grad_lag,
            scaled != FALSE,
        ));
        let Some(v) = violations else {
            return FALSE;
        };
        let copy = |src: &[Number], dst: *mut Number, len: usize| -> bool {
            if dst.is_null() || len == 0 {
                return true;
            }
            if src.len() != len {
                return false;
            }
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
            true
        };
        let ok = copy(&v.x_l_violation, x_l_violation, n_us)
            && copy(&v.x_u_violation, x_u_violation, n_us)
            && copy(&v.compl_x_l, compl_x_l, n_us)
            && copy(&v.compl_x_u, compl_x_u, n_us)
            && copy(&v.grad_lag_x, grad_lag_x, n_us)
            && copy(&v.constraint_violation, nlp_constraint_violation, m_us)
            && copy(&v.compl_g, compl_g, m_us);
        if ok { TRUE } else { FALSE }
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

/// Finite-difference Hessian census from the most recent solve. Any
/// pointer may be NULL to skip that component.
///
/// All outputs are left at their "did not run" values on any solve that
/// was not `hessian_approximation=finite-difference`: `pattern_used`
/// is `-1` and the counts are `0`. `pattern_used` is `0` for the
/// declared pattern and `1` for the Jacobian-derived one, and it names
/// what the run **ended up with** — `declared` falls back to `jacobian`
/// when the TNLP declares no Hessian structure, and that fallback is
/// what the number is worth reading for.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. Each
/// non-NULL output pointer must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceFdHessianStats(
    ipopt_problem: IpoptProblem,
    pattern_used: *mut Index,
    nnz: *mut Index,
    n: *mut Index,
    groups: *mut Index,
    rho_max: *mut Index,
    coloring_fell_back: *mut Index,
    objective_clique_widened: *mut Index,
) {
    unsafe {
        let stats = last_stat(ipopt_problem, |s| {
            (
                s.fd_hessian_pattern_used,
                s.fd_hessian_nnz,
                s.fd_hessian_n,
                s.fd_hessian_groups,
                s.fd_hessian_rho_max,
                if s.fd_hessian_coloring_fell_back {
                    1
                } else {
                    0
                },
                if s.fd_hessian_objective_clique_widened {
                    1
                } else {
                    0
                },
            )
        });
        let (p, nz, cols, g, r, f, w) = stats.unwrap_or((-1, 0, 0, 0, 0, 0, 0));
        if !pattern_used.is_null() {
            *pattern_used = p;
        }
        if !nnz.is_null() {
            *nnz = nz;
        }
        if !n.is_null() {
            *n = cols;
        }
        if !groups.is_null() {
            *groups = g;
        }
        if !rho_max.is_null() {
            *rho_max = r;
        }
        if !coloring_fell_back.is_null() {
            *coloring_fell_back = f;
        }
        if !objective_clique_widened.is_null() {
            *objective_clique_widened = w;
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

/// Whether a staged variable status is a true claim about bounds `[lo, hi]`:
/// `Fixed` needs `lo == hi`, `AtLower`/`AtUpper` a finite side. Shared by
/// [`IpoptSetWarmStartWorkingSet`] and [`bounds::SetPounceBounds`], which
/// re-checks a staged set against the bounds it would install.
pub(crate) fn bound_status_consistent(s: pounce_qp::BoundStatus, lo: Number, hi: Number) -> bool {
    match s {
        pounce_qp::BoundStatus::Fixed => lo == hi,
        pounce_qp::BoundStatus::AtLower => lo > NLP_LOWER_BOUND_INF,
        pounce_qp::BoundStatus::AtUpper => hi < NLP_UPPER_BOUND_INF,
        pounce_qp::BoundStatus::Inactive => true,
    }
}

/// The row counterpart of [`bound_status_consistent`]: `Equality` needs two
/// equal finite sides.
pub(crate) fn cons_status_consistent(s: pounce_qp::ConsStatus, lo: Number, hi: Number) -> bool {
    let lo_finite = lo > NLP_LOWER_BOUND_INF;
    let hi_finite = hi < NLP_UPPER_BOUND_INF;
    match s {
        pounce_qp::ConsStatus::Equality => lo_finite && hi_finite && lo == hi,
        pounce_qp::ConsStatus::AtLower => lo_finite,
        pounce_qp::ConsStatus::AtUpper => hi_finite,
        pounce_qp::ConsStatus::Inactive => true,
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

/// Retrieve the working set produced by the most recent SQP solve
/// (`algorithm = active-set-sqp`). Buffer sizes are `n` for
/// `bound_status_out` and `m` for `cons_status_out`. Pass `NULL`
/// for either to skip that side. Entry `i` is variable `i` / constraint
/// row `i` in the problem's own order (a variable fixed by
/// `x_L == x_U` reads `POUNCE_WS_FIXED_OR_EQ`), the same indexing
/// [`IpoptSetWarmStartWorkingSet`] takes.
///
/// Returns `TRUE` (1) on success, `FALSE` (0) if there is no
/// working set to retrieve (e.g. no SQP solve has run, the IPM
/// path was used, or the very first KKT check declared
/// optimality before solving any QP), if the most recent
/// [`IpoptSolve`] did not complete (refused, or failed before it
/// finished), or if complementarity pairs are declared on the
/// problem; nothing is written then.
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
        // No completed solve (none yet, or the last one was refused or
        // failed before it finished — `IpoptSolve` clears `last_solve` up
        // front), or pairs are declared, or the last solve ran the
        // complementarity driver on the lowered model (gh#1147): no working
        // set over the caller's rows belongs to the handle's current state,
        // and one left over from an earlier SQP solve must not read as if it
        // did.
        let Some(last) = info.last_solve.as_ref() else {
            return FALSE;
        };
        if last.mpcc.is_some() || !info.complementarity.is_empty() {
            return FALSE;
        }
        let ws = match info.app.last_sqp_working_set() {
            Some(w) => w,
            None => return FALSE,
        };
        // `last_sqp_working_set` is already in the caller's index spaces —
        // full `n` (fixed variables read `Fixed`) and `g` row order — so it
        // is copied positionally. This used to rebuild the SQP's
        // equalities-first order from the bounds and permute here, which
        // stopped matching once the application lifted bound statuses to
        // full `n` (gh#986): with any fixed variable the length check
        // below failed and this returned FALSE after a successful solve.
        let n = info.x_l.len();
        let m = info.g_l.len();
        if ws.constraints.len() != m || ws.bounds.len() != n {
            // The stored set does not match this problem's shape. Report
            // nothing rather than something mis-indexed.
            return FALSE;
        }
        if !bound_status_out.is_null() {
            for (i, &s) in ws.bounds.iter().enumerate() {
                *bound_status_out.add(i) = bound_status_to_int(s);
            }
        }
        if !cons_status_out.is_null() {
            for (i, &s) in ws.constraints.iter().enumerate() {
                *cons_status_out.add(i) = cons_status_to_int(s);
            }
        }
        TRUE
    }
}

/// Supply a warm-start working set consumed by the next
/// [`IpoptSolve`] on this problem. Pass `NULL` for either side to
/// cold-start it. The caller-owned buffers are copied; reuse
/// across calls is safe. Indexed like [`IpoptGetWorkingSet`]: entry `i`
/// is variable `i` / constraint row `i` in the problem's own order.
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
        // The caller indexes by *their* rows and variables, and so does
        // `IpoptApplication::set_sqp_warm_start`: the solve maps the staged
        // set into the SQP's equalities-first, fixed-variables-removed
        // vectors through the problem's own classification, the inverse of
        // what `IpoptGetWorkingSet` reads out. Staged as given.
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
        let mut bounds = vec![pounce_qp::BoundStatus::Inactive; n];
        if !bound_status_in.is_null() {
            // Validate every entry the caller supplied, including those for
            // fixed variables that the internal problem drops: a wrong
            // claim is worth rejecting whether or not it would be used.
            for i in 0..n {
                let v = *bound_status_in.add(i);
                let Some(s) = int_to_bound_status(v) else {
                    return FALSE;
                };
                if !bound_status_consistent(s, info.x_l[i], info.x_u[i]) {
                    return FALSE;
                }
            }
            for (i, b) in bounds.iter_mut().enumerate() {
                // Already range-checked above.
                if let Some(s) = int_to_bound_status(*bound_status_in.add(i)) {
                    *b = s;
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
                if !cons_status_consistent(s, info.g_l[i], info.g_u[i]) {
                    return FALSE;
                }
            }
            for (i, c) in constraints.iter_mut().enumerate() {
                if let Some(s) = int_to_cons_status(*cons_status_in.add(i)) {
                    *c = s;
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

// ---------------------------------------------------------------------------
// Declared complementarity pairs (gh#1147)
// ---------------------------------------------------------------------------

/// The declaration succeeded.
pub const POUNCE_CC_OK: c_int = 0;
/// NULL handle, or a NULL `rows`/`vars` array with `n_pairs > 0`.
pub const POUNCE_CC_ERR_NULL: c_int = -1;
/// A row or variable index outside `[0, m)` / `[0, n)`, read in the
/// problem's index style.
pub const POUNCE_CC_ERR_OUT_OF_RANGE: c_int = -2;
/// The variable is already in a pair (declared earlier or earlier in the
/// same batch).
pub const POUNCE_CC_ERR_VAR_ALREADY: c_int = -3;
/// The row is already in a pair (declared earlier or earlier in the same
/// batch).
pub const POUNCE_CC_ERR_ROW_ALREADY: c_int = -4;
/// The pair's bounds are none of the lowered shapes: one-sided, one-sided
/// plus a plain bound, or mixed.
pub const POUNCE_CC_ERR_SHAPE: c_int = -5;
/// `n_pairs < 0`.
pub const POUNCE_CC_ERR_COUNT: c_int = -6;
/// A Rust panic inside the library, caught at the boundary, or (from
/// `AddPounceComplementarityPair`) a pair already stored on the handle that
/// no longer validates. Not expected; it exists so neither has to be
/// reported as a diagnosis of the caller's input.
pub const POUNCE_CC_ERR_INTERNAL: c_int = -7;

fn pair_error_code(kind: &PairErrorKind) -> c_int {
    match kind {
        PairErrorKind::OutOfRange => POUNCE_CC_ERR_OUT_OF_RANGE,
        PairErrorKind::VarAlreadyPaired => POUNCE_CC_ERR_VAR_ALREADY,
        PairErrorKind::RowAlreadyPaired => POUNCE_CC_ERR_ROW_ALREADY,
        PairErrorKind::UnsupportedShape => POUNCE_CC_ERR_SHAPE,
    }
}

/// An index in the problem's index style, as a 0-based `usize` when it is
/// in `[0, len)`, else `usize::MAX` — which [`validate_pairs`] refuses as
/// out of range in its own order (range, duplicate variable, duplicate
/// row, shape, pair by pair). `checked_sub` keeps `i32::MIN` in Fortran
/// style from overflowing.
pub(crate) fn zero_based(p: Index, offset: Index, len: Index) -> usize {
    match p.checked_sub(offset) {
        Some(z) if z >= 0 && z < len => z as usize,
        _ => usize::MAX,
    }
}

/// Replace the declared pairs with `new` (pairs in the problem's index
/// style), validated as a whole and on their own by
/// [`validate_pairs`] — the classifier the solve's lowering runs — and
/// stored only if every pair passes. On refusal the declared set is
/// untouched and the error names the refused pair's position in `new`.
///
/// O(n + m + P): `validate_pairs` allocates a flag per variable and row.
/// Fine once per batch; the one-pair path is [`append_pair`].
fn replace_pairs(
    info: &mut IpoptProblemInfo,
    new: &[(Index, Index)],
) -> Result<(), (usize, PairErrorKind)> {
    let offset = if info.index_style == 1 { 1 } else { 0 };
    let candidate: Vec<ComplementarityPair> = new
        .iter()
        .map(|&(row, var)| ComplementarityPair {
            row: zero_based(row, offset, info.m),
            var: zero_based(var, offset, info.n),
        })
        .collect();
    match validate_pairs(
        info.n as usize,
        info.m as usize,
        &info.x_l,
        &info.x_u,
        &info.g_l,
        &info.g_u,
        &candidate,
    ) {
        Ok(_) => {
            info.complementarity.replace(candidate);
            Ok(())
        }
        Err(e) => Err((e.pair, e.kind)),
    }
}

/// Append one pair `(row, var)`, in the problem's index style, to the
/// declared set, after the checks [`validate_pairs`] runs on a pair, in its
/// order: range, variable not in a pair, row not in a pair, a shape
/// [`classify_pair`] accepts. O(1) expected (gh#1147 Stage 2): the stored
/// pairs are not re-validated, because every stored pair passed when it was
/// stored and every change to the bounds re-classifies the pairs it touches
/// ([`bounds`]), and the new pair's variable and row are in no stored pair.
fn append_pair(info: &mut IpoptProblemInfo, row: Index, var: Index) -> Result<(), PairErrorKind> {
    let offset = if info.index_style == 1 { 1 } else { 0 };
    let r = zero_based(row, offset, info.m);
    let v = zero_based(var, offset, info.n);
    if r == usize::MAX || v == usize::MAX {
        return Err(PairErrorKind::OutOfRange);
    }
    if info.complementarity.of_var(v).is_some() {
        return Err(PairErrorKind::VarAlreadyPaired);
    }
    if info.complementarity.of_row(r).is_some() {
        return Err(PairErrorKind::RowAlreadyPaired);
    }
    if classify_pair(info.x_l[v], info.x_u[v], info.g_l[r], info.g_u[r]).is_none() {
        return Err(PairErrorKind::UnsupportedShape);
    }
    info.complementarity
        .push(ComplementarityPair { row: r, var: v });
    Ok(())
}

/// Declare one complementarity pair: constraint row `row` with variable
/// `var`, both **in the problem's index style** (gh#1147). The pair is in
/// POUNCE's native shapes — one-sided, one-sided plus a plain bound, or
/// mixed — exactly as Python's `Problem(complementarity=)` and the `.nl`
/// reader accept, and is checked at once by the same classifier the
/// solve's lowering runs (`pounce_nlp::mpcc::validate_pairs`), against the
/// handle's bounds — those given to [`CreateIpoptProblem`], as
/// [`AddPounceComplementsConstraint`] may have edited them — and the pairs
/// already declared.
///
/// Returns [`POUNCE_CC_OK`] or a negative `POUNCE_CC_ERR_*` code; a refused
/// pair leaves the declared set unchanged. Pairs persist on the handle
/// until [`SetPounceComplementarityPairs`] replaces them or
/// [`ClearPounceComplementarityPairs`] drops them.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn AddPounceComplementarityPair(
    ipopt_problem: IpoptProblem,
    row: Index,
    var: Index,
) -> c_int {
    ffi_guard(POUNCE_CC_ERR_INTERNAL, || unsafe {
        if ipopt_problem.is_null() {
            return POUNCE_CC_ERR_NULL;
        }
        match append_pair(&mut *ipopt_problem, row, var) {
            Ok(()) => POUNCE_CC_OK,
            Err(kind) => pair_error_code(&kind),
        }
    })
}

/// Replace the declared pairs with `n_pairs` pairs `(rows[k], vars[k])`,
/// indices in the problem's index style. All or nothing: the new set is
/// validated as a whole (on its own, not against the pairs it replaces)
/// and stored only if every pair passes; otherwise the previous set stays.
/// `n_pairs == 0` clears the set (the arrays may then be NULL).
///
/// `bad_index`, when non-NULL, receives the position in the batch of the
/// first refused pair (for a duplicate, the second occurrence), or `-1`
/// on success and on the errors that concern no single pair
/// ([`POUNCE_CC_ERR_NULL`], [`POUNCE_CC_ERR_COUNT`]).
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `rows` and
/// `vars`, when non-NULL, must each point at `n_pairs` readable `ipindex`
/// values; `bad_index` must be NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SetPounceComplementarityPairs(
    ipopt_problem: IpoptProblem,
    n_pairs: Index,
    rows: *const Index,
    vars: *const Index,
    bad_index: *mut Index,
) -> c_int {
    ffi_guard(POUNCE_CC_ERR_INTERNAL, || unsafe {
        let report = |k: Index| {
            if !bad_index.is_null() {
                *bad_index = k;
            }
        };
        report(-1);
        if ipopt_problem.is_null() {
            return POUNCE_CC_ERR_NULL;
        }
        if n_pairs < 0 {
            return POUNCE_CC_ERR_COUNT;
        }
        if n_pairs > 0 && (rows.is_null() || vars.is_null()) {
            return POUNCE_CC_ERR_NULL;
        }
        let new: Vec<(Index, Index)> = if n_pairs == 0 {
            Vec::new()
        } else {
            let r = std::slice::from_raw_parts(rows, n_pairs as usize);
            let v = std::slice::from_raw_parts(vars, n_pairs as usize);
            r.iter().copied().zip(v.iter().copied()).collect()
        };
        match replace_pairs(&mut *ipopt_problem, &new) {
            Ok(()) => POUNCE_CC_OK,
            Err((k, kind)) => {
                report(k as Index);
                pair_error_code(&kind)
            }
        }
    })
}

/// Drop every declared complementarity pair. Returns `FALSE` only on a
/// NULL handle.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ClearPounceComplementarityPairs(ipopt_problem: IpoptProblem) -> Bool {
    ffi_guard(FALSE, || unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        (*ipopt_problem).complementarity.clear();
        TRUE
    })
}

/// Number of declared complementarity pairs, or `-1` on a NULL handle.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceComplementarityPairCount(ipopt_problem: IpoptProblem) -> Index {
    ffi_guard(-1, || unsafe {
        if ipopt_problem.is_null() {
            return -1;
        }
        (&*ipopt_problem).complementarity.len() as Index
    })
}

/// Copy the declared pairs, in declaration order and **in the problem's
/// index style**, into `rows` and `vars` (each sized by
/// [`GetPounceComplementarityPairCount`]). Returns `FALSE`, writing
/// nothing, on a NULL handle or a NULL array while pairs are declared;
/// with no pairs it returns `TRUE` and the arrays may be NULL.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `rows` and
/// `vars`, when non-NULL, must each have room for the declared count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceComplementarityPairs(
    ipopt_problem: IpoptProblem,
    rows: *mut Index,
    vars: *mut Index,
) -> Bool {
    ffi_guard(FALSE, || unsafe {
        if ipopt_problem.is_null() {
            return FALSE;
        }
        let info = &*ipopt_problem;
        let k = info.complementarity.len();
        if k == 0 {
            return TRUE;
        }
        if rows.is_null() || vars.is_null() {
            return FALSE;
        }
        let offset = if info.index_style == 1 { 1 } else { 0 };
        let rows = std::slice::from_raw_parts_mut(rows, k);
        let vars = std::slice::from_raw_parts_mut(vars, k);
        for (i, p) in info.complementarity.iter().enumerate() {
            rows[i] = p.row as Index + offset;
            vars[i] = p.var as Index + offset;
        }
        TRUE
    })
}

// ---------------------------------------------------------------------------
// MOI `[F, x] ∈ Complements` semantics (gh#1147)
// ---------------------------------------------------------------------------

/// `x` boxed, `l < u` both finite: a native **mixed** pair was declared;
/// the row's bounds are unchanged (free).
pub const POUNCE_COMPLEMENTS_PAIR_MIXED: c_int = 0;
/// `x >= l` only: the row's lower bound was set to `0` (`F >= 0`) and a
/// native one-sided pair declared.
pub const POUNCE_COMPLEMENTS_PAIR_LOWER: c_int = 1;
/// `x <= u` only: the row's upper bound was set to `0` (`F <= 0`) and a
/// native one-sided pair declared.
pub const POUNCE_COMPLEMENTS_PAIR_UPPER: c_int = 2;
/// `x` free: both row bounds were set to `0` (`F = 0`); no pair.
pub const POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR: c_int = 3;
/// `x` fixed (`l == u`): nothing changed, `F` stays free; no pair.
pub const POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR: c_int = 4;
/// NULL handle.
pub const POUNCE_COMPLEMENTS_ERR_NULL: c_int = -1;
/// Row or variable outside `[0, m)` / `[0, n)`, read in the problem's
/// index style.
pub const POUNCE_COMPLEMENTS_ERR_OUT_OF_RANGE: c_int = -2;
/// The variable is in a declared pair or in an earlier accepted
/// [`AddPounceComplementsConstraint`] call.
pub const POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED: c_int = -3;
/// The row is in a declared pair or in an earlier accepted
/// [`AddPounceComplementsConstraint`] call.
pub const POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED: c_int = -4;
/// The row has a finite bound (`g_L > -1e19` or `g_U < 1e19`), including
/// one an earlier call of this helper set: MOI's `F` is a free function.
/// Declare such a row with [`AddPounceComplementarityPair`].
pub const POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE: c_int = -5;
/// The variable's bounds are NaN or crossed (`l > u`, both finite): no row
/// of the MOI table applies.
pub const POUNCE_COMPLEMENTS_ERR_VAR_BOUNDS: c_int = -6;
/// The handle's state is inconsistent (a declared pair no longer passes
/// validation against its bounds), or a Rust panic was caught at the
/// boundary. Nothing was changed.
pub const POUNCE_COMPLEMENTS_ERR_INTERNAL: c_int = -7;

pub(crate) fn complements_row(xl: Number, xu: Number) -> Option<ComplementsRow> {
    if xl.is_nan() || xu.is_nan() {
        return None;
    }
    match (xl > NLP_LOWER_BOUND_INF, xu < NLP_UPPER_BOUND_INF) {
        (true, true) if xl < xu => Some(ComplementsRow::Mixed),
        (true, true) if xl == xu => Some(ComplementsRow::VarFixed),
        (true, true) => None,
        (true, false) => Some(ComplementsRow::Lower),
        (false, true) => Some(ComplementsRow::Upper),
        (false, false) => Some(ComplementsRow::RowFixed),
    }
}

impl ComplementsRow {
    /// The row bounds the table gives a free row, and whether it declares a
    /// pair.
    pub(crate) fn row_bounds(self, gl: Number, gu: Number) -> (Number, Number, bool) {
        match self {
            ComplementsRow::Mixed => (gl, gu, true),
            ComplementsRow::Lower => (0.0, gu, true),
            ComplementsRow::Upper => (gl, 0.0, true),
            ComplementsRow::RowFixed => (0.0, 0.0, false),
            ComplementsRow::VarFixed => (gl, gu, false),
        }
    }

    fn code(self) -> c_int {
        match self {
            ComplementsRow::Mixed => POUNCE_COMPLEMENTS_PAIR_MIXED,
            ComplementsRow::Lower => POUNCE_COMPLEMENTS_PAIR_LOWER,
            ComplementsRow::Upper => POUNCE_COMPLEMENTS_PAIR_UPPER,
            ComplementsRow::RowFixed => POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR,
            ComplementsRow::VarFixed => POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR,
        }
    }

    /// The native shape the edited bounds must classify as (pair rows).
    fn shape_matches(self, shape: Option<PairShape>) -> bool {
        matches!(
            (self, shape),
            (ComplementsRow::Mixed, Some(PairShape::Mixed { .. }))
                | (ComplementsRow::Lower, Some(PairShape::OneSided { .. }))
                | (ComplementsRow::Upper, Some(PairShape::OneSided { .. }))
        )
    }
}

fn add_complements(info: &mut IpoptProblemInfo, row: Index, var: Index) -> c_int {
    let offset = if info.index_style == 1 { 1 } else { 0 };
    let (r, v) = (
        zero_based(row, offset, info.m),
        zero_based(var, offset, info.n),
    );
    if r == usize::MAX || v == usize::MAX {
        return POUNCE_COMPLEMENTS_ERR_OUT_OF_RANGE;
    }
    let (gl, gu) = (info.g_l[r], info.g_u[r]);
    if !(gl <= NLP_LOWER_BOUND_INF && gu >= NLP_UPPER_BOUND_INF) {
        return POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE;
    }
    if info.complementarity.of_var(v).is_some() || info.complements_used.of_var(v).is_some() {
        return POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED;
    }
    if info.complementarity.of_row(r).is_some() || info.complements_used.of_row(r).is_some() {
        return POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED;
    }
    let Some(table_row) = complements_row(info.x_l[v], info.x_u[v]) else {
        return POUNCE_COMPLEMENTS_ERR_VAR_BOUNDS;
    };
    let (new_gl, new_gu, declares) = table_row.row_bounds(gl, gu);
    debug_assert!(
        !declares
            || table_row.shape_matches(classify_pair(info.x_l[v], info.x_u[v], new_gl, new_gu)),
        "MOI table row {table_row:?} edits to a bound pair that is not its native shape"
    );
    // Tentative edit, then declare the pair (if any) through the path every
    // one-pair declaration takes. The row and the variable are in no pair,
    // so only the shape check can refuse, and the table row's edit makes
    // that shape native by construction; if it does refuse, the row is
    // restored rather than left half-edited.
    info.g_l[r] = new_gl;
    info.g_u[r] = new_gu;
    if declares && append_pair(info, row, var).is_err() {
        info.g_l[r] = gl;
        info.g_u[r] = gu;
        return POUNCE_COMPLEMENTS_ERR_INTERNAL;
    }
    info.complements_used.push(ComplementsUse {
        row: r,
        var: v,
        table: table_row,
        user_gl: gl,
        user_gu: gu,
    });
    table_row.code()
}

/// Declare MOI's `[F, x] ∈ Complements` for constraint row `row` (`F`) and
/// variable `var` (`x`), both **in the problem's index style** (gh#1147).
///
/// MOI writes `F` as a free function and reads the sign it must have off
/// `x`'s bounds; POUNCE's native pairs read both sides' signs off their
/// bounds. This helper transcribes one into the other **by editing the
/// handle's copy of the row bounds**, then declares the native pair, as
/// the table says (bounds at or beyond `±1e19` are infinite):
///
/// | `x` bounds         | row edit            | pair            | returns |
/// |--------------------|---------------------|-----------------|---------|
/// | `l < u`, finite    | none                | native mixed    | [`POUNCE_COMPLEMENTS_PAIR_MIXED`] |
/// | `l` only           | `g_L = 0` (`F ≥ 0`) | one-sided       | [`POUNCE_COMPLEMENTS_PAIR_LOWER`] |
/// | `u` only           | `g_U = 0` (`F ≤ 0`) | one-sided       | [`POUNCE_COMPLEMENTS_PAIR_UPPER`] |
/// | free               | `g_L = g_U = 0`     | none (`F = 0`)  | [`POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR`] |
/// | `l == u`           | none (`F` free)     | none            | [`POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR`] |
///
/// The row must be free (else [`POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE`]:
/// use [`AddPounceComplementarityPair`] for a row that already has a
/// side). Checks, in order: NULL handle, range, row free, variable then
/// row not already used (by a declared pair or by an earlier accepted call
/// of this helper), the variable's bounds neither NaN nor crossed. A
/// refused call changes nothing. Bound edits are permanent on the handle:
/// [`ClearPounceComplementarityPairs`] and
/// [`SetPounceComplementarityPairs`] drop pairs, not edits.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn AddPounceComplementsConstraint(
    ipopt_problem: IpoptProblem,
    row: Index,
    var: Index,
) -> c_int {
    ffi_guard(POUNCE_COMPLEMENTS_ERR_INTERNAL, || unsafe {
        if ipopt_problem.is_null() {
            return POUNCE_COMPLEMENTS_ERR_NULL;
        }
        add_complements(&mut *ipopt_problem, row, var)
    })
}

// ---------------------------------------------------------------------------
// Complementarity outcome getters (gh#1147)
//
// What the complementarity driver did on the most recent `IpoptSolve`: the
// same fields, through the same encodings, as the solve report's
// `complementarity` block (`MpccOutcome::report_info`) and Python's
// `info["complementarity"]`. Every getter answers "nothing" (`FALSE` / `-1`)
// when the most recent solve did not run the driver: no solve yet, a solve
// without pairs, a refused or failed solve, or a session solve.
// ---------------------------------------------------------------------------

/// `certificate_class`: strongly stationary.
pub const POUNCE_CC_CLASS_S: c_int = 0;
/// `certificate_class`: weakly stationary only.
pub const POUNCE_CC_CLASS_W: c_int = 1;
/// `certificate_class`: not a stationary point of the model.
pub const POUNCE_CC_CLASS_NONE: c_int = 2;
/// `certificate_class`: not determined (the multiplier set is too large to
/// search).
pub const POUNCE_CC_CLASS_NOT_SEARCHED: c_int = 3;
/// `certificate_class`: no certificate (the model could not be evaluated at
/// the reported point).
pub const POUNCE_CC_CLASS_ABSENT: c_int = -1;

/// `reported`: a formulation answered.
pub const POUNCE_CC_REPORTED_ANSWER: c_int = 0;
/// `reported`: a weakly stationary point, reported when none answered.
pub const POUNCE_CC_REPORTED_WEAK_FALLBACK: c_int = 1;
/// `reported`: a converged point with every product in tolerance, no
/// stationarity certified.
pub const POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED: c_int = 2;
/// `reported`: no feasible point; the exact-product solve's verdict.
pub const POUNCE_CC_REPORTED_VERDICT: c_int = 3;

/// `b_verdict`: B-stationary.
pub const POUNCE_CC_B_VERDICT_B: c_int = 0;
/// `b_verdict`: a branch descends.
pub const POUNCE_CC_B_VERDICT_NOT_B: c_int = 1;
/// `b_verdict`: not decided (the reason is `POUNCE_CC_TEXT_B_REASON`).
pub const POUNCE_CC_B_VERDICT_NOT_CHECKED: c_int = 2;
/// `b_verdict`: the check did not run.
pub const POUNCE_CC_B_VERDICT_ABSENT: c_int = -1;

/// [`GetPounceComplementarityText`]: the driver's one-line account.
pub const POUNCE_CC_TEXT_SUMMARY: c_int = 0;
/// [`GetPounceComplementarityText`]: why the polish's point was not taken.
pub const POUNCE_CC_TEXT_POLISH_REASON: c_int = 1;
/// [`GetPounceComplementarityText`]: why the B-stationarity check was not
/// decided.
pub const POUNCE_CC_TEXT_B_REASON: c_int = 2;

/// C mirror of the driver's [`MpccOutcome`](pounce_restoration::mpcc_driver::MpccOutcome)
/// for `pounce.h`'s `PounceComplementarityStats`. Field order is the ABI
/// (Pounce.jl mirrors structs field for field); `tests/c_examples_build_and_run.rs`
/// holds every offset to the C compiler's. Sentinels instead of
/// discriminants: `NaN` for a real that was not measured (or is not
/// finite, as the report's `null`), `-1` for a count or flag whose record
/// is absent.
///
/// Extensible (see `ExtensibleOut` in this crate): `struct_size` first, set by the
/// caller to `sizeof(PounceComplementarityStats)`; fields are only ever
/// appended.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PounceComplementarityStats {
    pub struct_size: Index,
    pub strategy: [c_char; 16],
    pub reported: Index,
    pub accepted_by: [c_char; 32],
    pub fallback: [c_char; 32],
    pub n_solves: Index,
    pub total_iterations: Index,
    pub max_abs_product: Number,
    pub certificate_class: Index,
    pub n_pairs: Index,
    pub n_biactive: Index,
    pub n_neither: Index,
    pub multipliers_unique: Index,
    pub residual: Number,
    pub resid_tol: Number,
    pub tight: Index,
    pub feasible: Index,
    pub primal_violation: Number,
    pub b_verdict: Index,
    pub b_n_biactive: Index,
    pub b_branches_solved: Index,
    pub b_enumerated: Index,
    pub escapes: Index,
    pub lifted: Index,
    pub natural_residual: Number,
    pub polish_ran: Index,
    pub polish_taken: Index,
    pub polish_residual_before: Number,
    pub polish_residual_after: Number,
    pub polish_threshold: Number,
}

/// C mirror of one [`MpccSolveRecord`](pounce_restoration::mpcc_driver::MpccSolveRecord),
/// for `pounce.h`'s `PounceComplementaritySolve`. Extensible, like
/// [`PounceComplementarityStats`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PounceComplementaritySolve {
    pub struct_size: Index,
    pub stage: [c_char; 32],
    pub relaxed: Index,
    pub tau: Number,
    pub status: Index,
    pub iterations: Index,
    pub max_abs_product: Number,
    pub seconds: Number,
}

// SAFETY: both are `#[repr(C)]` with `struct_size: Index` first.
unsafe impl ExtensibleOut for PounceComplementarityStats {}
unsafe impl ExtensibleOut for PounceComplementaritySolve {}

/// Copy `s` into a fixed C char array, truncated on a byte boundary so that
/// a NUL always fits; the rest of the array is zeroed.
fn copy_label<const N: usize>(dst: &mut [c_char; N], s: &str) {
    dst.fill(0);
    let keep = s.len().min(N.saturating_sub(1));
    for (slot, b) in dst.iter_mut().zip(&s.as_bytes()[..keep]) {
        *slot = *b as c_char;
    }
}

/// A count as an `Index`, saturating rather than wrapping.
fn cc_count(v: usize) -> Index {
    Index::try_from(v).unwrap_or(Index::MAX)
}

/// The report's `finite` rule, in C's terms: a non-finite or unmeasured
/// real reads `NaN` where the JSON reads `null`.
fn cc_real(v: Number) -> Number {
    if v.is_finite() { v } else { Number::NAN }
}

fn cc_flag(b: bool) -> Index {
    Index::from(b)
}

/// The struct [`GetPounceComplementarityStats`] fills, from the driver's
/// record. The encodings are `MpccOutcome::report_info`'s field for field
/// (strings become the `POUNCE_CC_*` codes, `null` becomes `NaN` or `-1`).
pub(crate) fn complementarity_stats(
    out: &pounce_restoration::mpcc_driver::MpccOutcome,
) -> PounceComplementarityStats {
    use pounce_algorithm::mpcc::{BVerdict, Reported, StationarityClass};
    let mut s = PounceComplementarityStats {
        struct_size: struct_size_of::<PounceComplementarityStats>(),
        strategy: [0; 16],
        reported: match out.reported {
            Reported::Answer(_) => POUNCE_CC_REPORTED_ANSWER,
            Reported::WeakFallback(_) => POUNCE_CC_REPORTED_WEAK_FALLBACK,
            Reported::FeasibleUncertified(_) => POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED,
            Reported::Verdict => POUNCE_CC_REPORTED_VERDICT,
        },
        accepted_by: [0; 32],
        fallback: [0; 32],
        n_solves: cc_count(out.solves.len()),
        total_iterations: cc_count(out.total_iterations),
        max_abs_product: cc_real(out.max_abs_product),
        certificate_class: POUNCE_CC_CLASS_ABSENT,
        n_pairs: -1,
        n_biactive: -1,
        n_neither: -1,
        multipliers_unique: -1,
        residual: Number::NAN,
        resid_tol: Number::NAN,
        tight: -1,
        feasible: -1,
        primal_violation: Number::NAN,
        b_verdict: POUNCE_CC_B_VERDICT_ABSENT,
        b_n_biactive: -1,
        b_branches_solved: -1,
        b_enumerated: -1,
        escapes: cc_count(out.escapes),
        lifted: cc_count(out.lifted),
        natural_residual: out.natural_residual.map_or(Number::NAN, cc_real),
        polish_ran: 0,
        polish_taken: 0,
        polish_residual_before: Number::NAN,
        polish_residual_after: Number::NAN,
        polish_threshold: cc_real(out.polish_threshold),
    };
    copy_label(&mut s.strategy, out.strategy.name());
    if let Some(stage) = out.accepted_by {
        copy_label(&mut s.accepted_by, stage.label());
    }
    if let Some(stage) = out.fallback {
        copy_label(&mut s.fallback, stage.label());
    }
    if let Some(c) = &out.certificate {
        s.certificate_class = match c.class {
            StationarityClass::S => POUNCE_CC_CLASS_S,
            StationarityClass::W => POUNCE_CC_CLASS_W,
            StationarityClass::None => POUNCE_CC_CLASS_NONE,
            StationarityClass::NotSearched => POUNCE_CC_CLASS_NOT_SEARCHED,
        };
        s.n_pairs = cc_count(c.n_pairs);
        s.n_biactive = cc_count(c.n_biactive);
        s.n_neither = cc_count(c.n_neither);
        s.multipliers_unique = cc_flag(c.multipliers_unique);
        s.residual = cc_real(c.residual);
        s.resid_tol = cc_real(c.resid_tol);
        s.tight = cc_flag(c.tight);
        s.feasible = cc_flag(c.feasible);
        s.primal_violation = cc_real(c.primal_violation);
    }
    if let Some(b) = &out.b_check {
        s.b_verdict = match b.verdict {
            BVerdict::B => POUNCE_CC_B_VERDICT_B,
            BVerdict::NotB { .. } => POUNCE_CC_B_VERDICT_NOT_B,
            BVerdict::NotChecked(_) => POUNCE_CC_B_VERDICT_NOT_CHECKED,
        };
        s.b_n_biactive = cc_count(b.n_biactive);
        s.b_branches_solved = cc_count(b.branches_solved);
        s.b_enumerated = cc_flag(b.enumerated);
    }
    if let Some(p) = &out.polish {
        s.polish_ran = 1;
        s.polish_taken = cc_flag(p.taken);
        s.polish_residual_before = cc_real(p.residual_before);
        s.polish_residual_after = cc_real(p.residual_after);
    }
    s
}

/// The record [`GetPounceComplementaritySolve`] fills for one lowered solve.
pub(crate) fn complementarity_solve(
    r: &pounce_restoration::mpcc_driver::MpccSolveRecord,
) -> PounceComplementaritySolve {
    let mut s = PounceComplementaritySolve {
        struct_size: struct_size_of::<PounceComplementaritySolve>(),
        stage: [0; 32],
        relaxed: cc_flag(r.tau.is_some()),
        tau: r.tau.unwrap_or(Number::NAN),
        status: r.status as Index,
        iterations: cc_count(r.iterations),
        max_abs_product: cc_real(r.max_abs_product),
        seconds: cc_real(r.seconds),
    };
    copy_label(&mut s.stage, r.stage.label());
    s
}

/// The text [`GetPounceComplementarityText`] returns for `which`, or `None`
/// when that text is absent on this run (or `which` names none).
pub(crate) fn complementarity_text(
    out: &pounce_restoration::mpcc_driver::MpccOutcome,
    which: c_int,
) -> Option<String> {
    use pounce_algorithm::mpcc::BVerdict;
    match which {
        POUNCE_CC_TEXT_SUMMARY => Some(pounce_algorithm::mpcc::describe_outcome(
            out.reported,
            out.solves.len(),
            out.total_iterations,
            out.max_abs_product,
            out.certificate.as_ref(),
        )),
        POUNCE_CC_TEXT_POLISH_REASON => out.polish.as_ref().and_then(|p| p.reason.clone()),
        POUNCE_CC_TEXT_B_REASON => match out.b_check.as_ref().map(|b| &b.verdict) {
            Some(BVerdict::NotChecked(why)) => Some((*why).to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// The driver's record of the most recent solve, if that solve ran it.
unsafe fn last_mpcc<'a>(
    ipopt_problem: IpoptProblem,
) -> Option<&'a pounce_restoration::mpcc_driver::MpccOutcome> {
    unsafe {
        if ipopt_problem.is_null() {
            return None;
        }
        (*ipopt_problem)
            .last_solve
            .as_ref()
            .and_then(|ls| ls.mpcc.as_ref())
    }
}

/// What the complementarity driver did on the most recent [`IpoptSolve`]
/// (gh#1147): the strategy, what was reported and by which formulation,
/// the iteration totals, the stationarity certificate, the B-stationarity
/// check and the polish — the solve report's `complementarity` block as a
/// struct of scalars. The caller sets `stats->struct_size =
/// sizeof(PounceComplementarityStats)` first. Returns `FALSE`, leaving
/// `stats` untouched, on a NULL argument, a `struct_size` smaller than this
/// release's size, before any solve, and when the most recent solve did not run
/// the driver (no pairs were declared, or it was refused or failed before
/// the driver returned). The record describes the most recent solve: clearing
/// the pairs afterwards does not erase it, the next solve does.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `stats`, when
/// non-NULL, must point at a writable `PounceComplementarityStats` (at
/// least `struct_size` bytes when that is this release's size or more).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceComplementarityStats(
    ipopt_problem: IpoptProblem,
    stats: *mut PounceComplementarityStats,
) -> Bool {
    ffi_guard(FALSE, || unsafe {
        if stats.is_null() {
            return FALSE;
        }
        let Some(out) = last_mpcc(ipopt_problem) else {
            return FALSE;
        };
        // Bounded write of this release's size; FALSE, nothing written,
        // when `stats->struct_size` is short (see `write_extensible`).
        Bool::from(write_extensible(stats, &complementarity_stats(out)))
    })
}

/// Lowered solve `k` (0-based, in the order the driver ran them, `k <
/// n_solves`) of the most recent driver solve; the caller sets
/// `solve->struct_size = sizeof(PounceComplementaritySolve)` first. Returns
/// `FALSE`, leaving `solve` untouched, on a NULL argument, a short
/// `struct_size`, a `k` out of range, and whenever
/// [`GetPounceComplementarityStats`] would.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `solve`, when
/// non-NULL, must point at a writable `PounceComplementaritySolve` (at
/// least `struct_size` bytes when that is this release's size or more).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceComplementaritySolve(
    ipopt_problem: IpoptProblem,
    k: Index,
    solve: *mut PounceComplementaritySolve,
) -> Bool {
    ffi_guard(FALSE, || unsafe {
        if solve.is_null() {
            return FALSE;
        }
        let Some(out) = last_mpcc(ipopt_problem) else {
            return FALSE;
        };
        let Some(r) = usize::try_from(k).ok().and_then(|k| out.solves.get(k)) else {
            return FALSE;
        };
        Bool::from(write_extensible(solve, &complementarity_solve(r)))
    })
}

/// One of the driver's free-form texts for the most recent driver solve,
/// with C's `snprintf` contract: writes at most `buflen - 1` bytes of it
/// plus a NUL into `buf` and returns its full length in bytes, without the
/// NUL — so `buf = NULL, buflen = 0` sizes the buffer, and a return `>=
/// buflen` means it was truncated. Returns `-1`, writing nothing, when
/// there is no such text: a NULL handle, a `which` that names none,
/// whenever [`GetPounceComplementarityStats`] would return `FALSE`, or a
/// text this run does not have (`POUNCE_CC_TEXT_POLISH_REASON` unless the
/// polish ran and was not taken; `POUNCE_CC_TEXT_B_REASON` unless the
/// B-stationarity check ran and was not decided). A NULL `buf` or a
/// `buflen <= 0` writes nothing and still returns the length.
///
/// # Safety
///
/// `ipopt_problem` must be a valid `IpoptProblem` or NULL. `buf`, when
/// non-NULL and `buflen > 0`, must have room for `buflen` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn GetPounceComplementarityText(
    ipopt_problem: IpoptProblem,
    which: c_int,
    buf: *mut c_char,
    buflen: Index,
) -> Index {
    ffi_guard(-1, || unsafe {
        let Some(out) = last_mpcc(ipopt_problem) else {
            return -1;
        };
        let Some(text) = complementarity_text(out, which) else {
            return -1;
        };
        let bytes = text.as_bytes();
        if !buf.is_null() && buflen > 0 {
            let keep = bytes.len().min(buflen as usize - 1);
            let dst = std::slice::from_raw_parts_mut(buf, keep + 1);
            for (slot, b) in dst.iter_mut().zip(&bytes[..keep]) {
                *slot = *b as c_char;
            }
            dst[keep] = 0;
        }
        cc_count(bytes.len())
    })
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
        // Copy a set out only when this solve completed: a refused solve
        // (pairs with a working set, gh#1147) leaves `last_solve` empty, and
        // the handle's last SQP set is then an earlier solve's.
        if (*ipopt_problem).last_solve.is_some() {
            let _ = IpoptGetWorkingSet(ipopt_problem, bound_status_out, cons_status_out);
        }
        status
    })
}

thread_local! {
    /// The bridge whose `intermediate_callback` is running, for
    /// [`GetIpoptCurrentViolations`]; null outside that window.
    static CURRENT_BRIDGE: std::cell::Cell<*const CCallbackTnlp> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// `(nlp_lower_bound_inf, nlp_upper_bound_inf, zero_fixed_grad_lag)` for
/// the bridge fields of the same names.
pub(crate) fn bridge_violation_options(
    o: &pounce_common::options_list::OptionsList,
) -> (Number, Number, bool) {
    let num = |k: &str, d: Number| {
        o.get_numeric_value(k, "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .unwrap_or(d)
    };
    let relax = matches!(
        o.get_string_value("fixed_variable_treatment", "")
            .ok()
            .and_then(|(v, f)| f.then_some(v))
            .as_deref(),
        Some("relax_bounds")
    );
    (
        num("nlp_lower_bound_inf", NLP_LOWER_BOUND_INF),
        num("nlp_upper_bound_inf", NLP_UPPER_BOUND_INF),
        !relax,
    )
}

thread_local! {
    /// `true` while the innermost [`IpoptSolve`] on this thread runs the
    /// complementarity driver (gh#1147). The live inspectors report no data
    /// then: the algorithm's iterate is the lowered model's — product rows
    /// appended, possibly lifted variables — not the caller's, and a prefix
    /// of it is not the caller's iterate either (a lifted row's bounds are
    /// rewritten).
    ///
    /// Per *solve*, not per thread: every `IpoptSolve` sets it for its own
    /// duration, so a pair-free solve of another problem started from inside
    /// a lowered solve's callback reads `false` (its iterate is its own) and
    /// the lowered solve reads `true` again once it returns.
    static IN_LOWERED_SOLVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`IN_LOWERED_SOLVE`] for its lifetime, restoring the previous value
/// on drop (so nesting unwinds in order, and a panic caught by [`ffi_guard`]
/// cannot leave it set).
struct LoweredSolveGuard {
    prev: bool,
}

impl LoweredSolveGuard {
    fn set(lowered: bool) -> Self {
        Self {
            prev: IN_LOWERED_SOLVE.with(|c| c.replace(lowered)),
        }
    }
}

impl Drop for LoweredSolveGuard {
    fn drop(&mut self) {
        IN_LOWERED_SOLVE.with(|c| c.set(self.prev));
    }
}

fn in_lowered_solve() -> bool {
    IN_LOWERED_SOLVE.with(|c| c.get())
}

struct BridgeGuard {
    prev: *const CCallbackTnlp,
}

impl BridgeGuard {
    fn install(bridge: &CCallbackTnlp) -> Self {
        let prev = CURRENT_BRIDGE.with(|c| c.replace(bridge as *const _));
        Self { prev }
    }
}

impl Drop for BridgeGuard {
    fn drop(&mut self) {
        CURRENT_BRIDGE.with(|c| c.set(self.prev));
    }
}

impl CCallbackTnlp {
    /// `∇f + Jᵀλ − z_L + z_U` at the user-space iterate `it`, through the
    /// user's own C callbacks (called directly, not through the TNLP the
    /// callback dispatch holds). `None` if a callback declines.
    fn user_grad_lag(&self, it: &ip_intermediate::UserIterate) -> Option<Vec<Number>> {
        let n = self.n as usize;
        let cb = self.eval_grad_f?;
        let mut grad = vec![0.0; n];
        let x = it.x.as_ptr() as *mut Number;
        if unsafe { cb(self.n, x, TRUE, grad.as_mut_ptr(), self.user_data) } == FALSE {
            return None;
        }
        let nnz = self.nele_jac.max(0) as usize;
        let (mut rows, mut cols, mut vals) = (vec![0; nnz], vec![0; nnz], vec![0.0; nnz]);
        if self.m > 0 && nnz > 0 {
            let jac = self.eval_jac_g?;
            let structure = unsafe {
                jac(
                    self.n,
                    x,
                    FALSE,
                    self.m,
                    self.nele_jac,
                    rows.as_mut_ptr(),
                    cols.as_mut_ptr(),
                    std::ptr::null_mut(),
                    self.user_data,
                )
            };
            let values = unsafe {
                jac(
                    self.n,
                    x,
                    FALSE,
                    self.m,
                    self.nele_jac,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    vals.as_mut_ptr(),
                    self.user_data,
                )
            };
            if structure == FALSE || values == FALSE {
                return None;
            }
        }
        let off = if self.index_style == 1 { 1 } else { 0 };
        let to_usize = |v: &[Index]| -> Vec<usize> {
            v.iter()
                .map(|&k| usize::try_from(k - off).unwrap_or(usize::MAX))
                .collect()
        };
        Some(pounce_nlp::live_iterate::lagrangian_gradient(
            it,
            &grad,
            &to_usize(&rows),
            &to_usize(&cols),
            &vals,
        ))
    }
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
    /// The caller's `mult_g` buffer (length `m`), or NULL. Read only by
    /// [`TNLP::get_starting_point`] under `init_lambda`, i.e. when the
    /// solve was asked for a dual warm start (`warm_start_init_point=yes`)
    /// — upstream `StdInterfaceTNLP`'s `start_lam`. A caller that has not
    /// opted in may pass an uninitialized buffer, which is why this is a
    /// pointer read lazily under the flag rather than a snapshot taken
    /// up front. Valid for the whole `IpoptSolve` call: the buffer is
    /// overwritten with the solution only after the solve (and any
    /// second-opinion re-solve, which calls `get_starting_point` again)
    /// has finished.
    pub(crate) start_lambda: *const Number,
    /// The caller's `mult_x_L` buffer (length `n`), or NULL. Read under
    /// `init_z`; see [`CCallbackTnlp::start_lambda`].
    pub(crate) start_z_l: *const Number,
    /// The caller's `mult_x_U` buffer (length `n`), or NULL. Read under
    /// `init_z`; see [`CCallbackTnlp::start_lambda`].
    pub(crate) start_z_u: *const Number,
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
    /// `nlp_lower_bound_inf` / `nlp_upper_bound_inf` of the solve, for the
    /// violations `GetIpoptCurrentViolations` measures against `x_l` …
    /// `g_u` (a side at or beyond these is absent).
    pub(crate) lower_inf: Number,
    pub(crate) upper_inf: Number,
    /// Whether `grad_lag_x` reads `0` at fixed variables: true unless
    /// `fixed_variable_treatment=relax_bounds`, which keeps them in the
    /// problem (upstream: `ResortX(…, usefixedvals=false)`).
    pub(crate) zero_fixed_grad_lag: bool,
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
        // Dual warm start, as upstream `StdInterfaceTNLP::get_starting_point`:
        // under `init_z` / `init_lambda` the caller's `mult_x_L` / `mult_x_U`
        // / `mult_g` buffers are the starting multipliers. Same sign
        // convention as the values `finalize_solution` writes back into those
        // buffers (L = f + lambda'g, z_L, z_U >= 0), so a solve's output feeds
        // the next solve's input unchanged.
        //
        // One deliberate departure: upstream fails the solve when a requested
        // buffer is NULL. Here NULL means "not supplied" and the slot is left
        // untouched, so `OrigIpoptNlp`'s NaN "unseeded" marker (gh#622) stands
        // and the warm-start initializer resolves it, rather than a primal-only
        // warm start becoming an error.
        let copy_in = |dst: &mut [Number], src: *const Number| {
            if !src.is_null() && !dst.is_empty() {
                // SAFETY: `src` is the caller's buffer of the documented
                // length (`n` for the bound multipliers, `m` for `mult_g`),
                // valid for the duration of the enclosing `IpoptSolve`.
                dst.copy_from_slice(unsafe { std::slice::from_raw_parts(src, dst.len()) });
            }
        };
        if sp.init_z {
            copy_in(sp.z_l, self.start_z_l);
            copy_in(sp.z_u, self.start_z_u);
        }
        if sp.init_lambda {
            copy_in(sp.lambda, self.start_lambda);
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

    /// Only a registered `SetIntermediateCallback` reads the payload, and
    /// the `GetIpoptCurrent*` inspectors are valid only inside it (gh#965).
    fn wants_intermediate_callback(&self) -> bool {
        self.intermediate_cb.is_some()
    }

    /// A registered callback may call `GetIpoptCurrentIterate` /
    /// `GetIpoptCurrentViolations`, and nothing says whether it will, so
    /// the iterate is materialised for every fire it receives.
    fn wants_current_iterate(&self) -> bool {
        self.intermediate_cb.is_some()
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
        // `GetIpoptCurrentViolations` evaluates `∇f` and `J` through this
        // bridge's C callbacks (the TNLP itself is mutably borrowed by the
        // dispatch that called us), so it needs the callbacks, `user_data`
        // and the declared bounds for exactly the duration of `cb`.
        let _bridge = BridgeGuard::install(self);
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
/// When that solve ran the complementarity driver (pairs were declared),
/// the report carries the `complementarity` block at either detail level,
/// built by the same conversion as the CLI's and Python's reports; it is
/// absent otherwise.
///
/// When that solve took the path without pairs and the second-opinion
/// ladder ran at least one rung, the report carries the `second_opinion`
/// block (gh#850) at either detail level, from the same conversion and
/// under the same condition as the CLI's and Python's; absent when no rung
/// ran, after a driver solve, and after a session solve.
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
        // Declared complementarity (gh#1147): what the driver did, through the
        // same conversion the CLI's `--json-output` and Python's `report_path`
        // use, so the three frontends write one block for one driver outcome.
        // `ingest_stats` above wrote the reported lowered solve's statistics;
        // the block carries every lowered solve and the true total. Absent for
        // a solve that did not run the driver (no pairs declared).
        if let Some(out) = &last.mpcc {
            builder.set_complementarity(out.report_info());
        }
        // gh#850: what the second-opinion ladder did, and what the *base*
        // solve did before it, under the CLI's and Python's `ran()` test and
        // through the same `report_info()`. `ingest_stats` above wrote the
        // promoted rung's iteration count as though it were the solve's;
        // without this block a rescued solve reads as a clean, fast one.
        // Absent when no rung ran, and on a driver solve, which does not run
        // the ladder (the CLI and Python skip it there too).
        if let Some(so) = &last.second_opinion
            && so.ran()
        {
            builder.set_second_opinion(so.report_info());
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
        // error-path: CreateIpoptProblem
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
        // error-path: AddIpoptIntOption
        // error-path: AddIpoptNumOption
        // error-path: AddIpoptStrOption
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
        // error-path: IpoptSolve
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
        // error-path: FreeIpoptProblem
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
        // error-path: GetIpoptVersion
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
        // error-path: SetIpoptProblemScaling
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
        // error-path: OpenIpoptOutputFile
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
        // error-path: GetIpoptCurrentIterate
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
        // error-path: GetIpoptCurrentViolations
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
        // error-path: GetIpoptIterCount
        // error-path: GetIpoptSolveTime
        // error-path: GetIpoptPrimalInf
        // error-path: GetIpoptDualInf
        // error-path: GetIpoptComplInf
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
        // error-path: GetPounceLinearSolverStats
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
        // error-path: GetPounceOptionType
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
        let packed = [0.5_f64, -0.3]; // len 2 != n_us == 1

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
        // error-path: IpoptCreateSolver
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
        // error-path: IpoptFreeSolver
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
        // error-path: IpoptSolverGetKktDim
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
        // error-path: IpoptGetWorkingSet
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
        // error-path: IpoptSetWarmStartWorkingSet
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

    // ---- complementarity outcome encodings (gh#1147) ----
    //
    // The integration tests (`tests/complementarity_getters.rs`) reach the
    // S, W, none and not-searched classes, the answer / weak-fallback /
    // verdict reports, the B verdict and a taken polish through real
    // solves. What no in-repo model reaches is held here, on a synthetic
    // driver record: no certificate, a feasible-uncertified report, the
    // not_B and not_checked verdicts, a polish that was not taken, and
    // non-finite reals.

    fn synthetic_outcome() -> pounce_restoration::mpcc_driver::MpccOutcome {
        use pounce_algorithm::mpcc::{
            Certificate, MpccStage, Reported, StationarityClass, Strategy,
        };
        use pounce_restoration::mpcc_driver::{MpccOutcome, MpccSolveRecord};
        MpccOutcome {
            status: ApplicationReturnStatus::SolveSucceeded,
            statistics: SolveStatistics::default(),
            accepted_by: Some(MpccStage::Exact),
            fallback: None,
            reported: Reported::Answer(MpccStage::Exact),
            strategy: Strategy::Continuation,
            solves: vec![MpccSolveRecord {
                stage: MpccStage::Branch,
                tau: None,
                status: ApplicationReturnStatus::MaximumIterationsExceeded,
                iterations: 7,
                budget_iterations: 7,
                max_abs_product: Number::INFINITY,
                seconds: Number::NAN,
            }],
            total_iterations: 7,
            x: vec![0.0],
            max_abs_product: Number::INFINITY,
            certificate: Some(Certificate {
                class: StationarityClass::S,
                n_pairs: 1,
                n_biactive: 0,
                n_neither: 0,
                multipliers_unique: true,
                residual: 0.0,
                resid_tol: Number::INFINITY,
                tight: true,
                primal_violation: 0.0,
                feasible: true,
                worst_biactive_sign: 0.0,
                multipliers: None,
            }),
            b_check: None,
            escapes: 3,
            lifted: 2,
            natural_residual: None,
            polish: None,
            polish_threshold: 1e-7,
        }
    }

    fn c_label(chars: &[c_char]) -> String {
        unsafe { CStr::from_ptr(chars.as_ptr()) }
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn complementarity_encodings_absent_certificate_and_non_finite_reals() {
        let mut out = synthetic_outcome();
        out.certificate = None;
        let s = complementarity_stats(&out);
        assert_eq!(s.certificate_class, POUNCE_CC_CLASS_ABSENT);
        assert_eq!(
            (
                s.n_pairs,
                s.n_biactive,
                s.n_neither,
                s.multipliers_unique,
                s.tight,
                s.feasible
            ),
            (-1, -1, -1, -1, -1, -1)
        );
        assert!(s.residual.is_nan() && s.resid_tol.is_nan() && s.primal_violation.is_nan());
        // Non-finite reals read NaN (the report's `null`), an absent
        // natural residual too.
        assert!(s.max_abs_product.is_nan() && s.natural_residual.is_nan());
        assert_eq!(c_label(&s.strategy), "continuation");
        assert_eq!(c_label(&s.accepted_by), "exact product");
        assert_eq!(c_label(&s.fallback), "");
        assert_eq!((s.escapes, s.lifted), (3, 2));
        // The summary still exists without a certificate.
        assert!(complementarity_text(&out, POUNCE_CC_TEXT_SUMMARY).is_some());

        // With a certificate, a non-finite band reads NaN, the rest as is.
        let s = complementarity_stats(&synthetic_outcome());
        assert_eq!(s.certificate_class, POUNCE_CC_CLASS_S);
        assert!(s.resid_tol.is_nan());
        assert_eq!(
            (s.n_pairs, s.multipliers_unique, s.tight, s.feasible),
            (1, 1, 1, 1)
        );

        let r = complementarity_solve(&out.solves[0]);
        assert_eq!(c_label(&r.stage), "branch NLP");
        assert_eq!((r.relaxed, r.status, r.iterations), (0, -1, 7));
        assert!(r.tau.is_nan() && r.max_abs_product.is_nan() && r.seconds.is_nan());
    }

    #[test]
    fn complementarity_encodings_every_reported_and_class() {
        use pounce_algorithm::mpcc::{MpccStage, Reported, StationarityClass};
        let mut out = synthetic_outcome();
        for (reported, code) in [
            (
                Reported::Answer(MpccStage::Exact),
                POUNCE_CC_REPORTED_ANSWER,
            ),
            (
                Reported::WeakFallback(MpccStage::Penalty),
                POUNCE_CC_REPORTED_WEAK_FALLBACK,
            ),
            (
                Reported::FeasibleUncertified(MpccStage::Continuation),
                POUNCE_CC_REPORTED_FEASIBLE_UNCERTIFIED,
            ),
            (Reported::Verdict, POUNCE_CC_REPORTED_VERDICT),
        ] {
            out.reported = reported;
            assert_eq!(complementarity_stats(&out).reported, code, "{reported:?}");
        }
        for (class, code) in [
            (StationarityClass::S, POUNCE_CC_CLASS_S),
            (StationarityClass::W, POUNCE_CC_CLASS_W),
            (StationarityClass::None, POUNCE_CC_CLASS_NONE),
            (StationarityClass::NotSearched, POUNCE_CC_CLASS_NOT_SEARCHED),
        ] {
            out.certificate.as_mut().unwrap().class = class;
            assert_eq!(
                complementarity_stats(&out).certificate_class,
                code,
                "{class:?}"
            );
        }
    }

    #[test]
    fn complementarity_encodings_b_verdicts_and_b_reason() {
        use pounce_algorithm::mpcc::{BCheck, BVerdict};
        let mut out = synthetic_outcome();
        assert_eq!(
            complementarity_stats(&out).b_verdict,
            POUNCE_CC_B_VERDICT_ABSENT
        );
        assert_eq!(complementarity_text(&out, POUNCE_CC_TEXT_B_REASON), None);
        for (verdict, code, reason) in [
            (BVerdict::B, POUNCE_CC_B_VERDICT_B, None),
            (
                BVerdict::NotB {
                    held: vec![],
                    decrease: -1.0,
                },
                POUNCE_CC_B_VERDICT_NOT_B,
                None,
            ),
            (
                BVerdict::NotChecked("a branch LP did not solve"),
                POUNCE_CC_B_VERDICT_NOT_CHECKED,
                Some("a branch LP did not solve"),
            ),
        ] {
            out.b_check = Some(BCheck {
                verdict,
                n_biactive: 12,
                branches_solved: 1,
                enumerated: false,
            });
            let s = complementarity_stats(&out);
            assert_eq!(s.b_verdict, code);
            assert_eq!(
                (s.b_n_biactive, s.b_branches_solved, s.b_enumerated),
                (12, 1, 0)
            );
            assert_eq!(
                complementarity_text(&out, POUNCE_CC_TEXT_B_REASON).as_deref(),
                reason
            );
        }
    }

    #[test]
    fn complementarity_encodings_polish_not_taken_has_a_reason() {
        use pounce_restoration::mpcc_driver::PolishRecord;
        let mut out = synthetic_outcome();
        out.polish = Some(PolishRecord {
            taken: false,
            residual_before: 1e-5,
            residual_after: Number::INFINITY,
            reason: Some("the re-solve did not converge".to_string()),
        });
        let s = complementarity_stats(&out);
        assert_eq!((s.polish_ran, s.polish_taken), (1, 0));
        assert_eq!(s.polish_residual_before, 1e-5);
        assert!(
            s.polish_residual_after.is_nan(),
            "unmeasured after reads NaN"
        );
        assert_eq!(
            complementarity_text(&out, POUNCE_CC_TEXT_POLISH_REASON).as_deref(),
            Some("the re-solve did not converge")
        );
        // Taken: no reason.
        out.polish = Some(PolishRecord {
            taken: true,
            residual_before: 1e-5,
            residual_after: 0.0,
            reason: None,
        });
        assert_eq!(complementarity_stats(&out).polish_taken, 1);
        assert_eq!(
            complementarity_text(&out, POUNCE_CC_TEXT_POLISH_REASON),
            None
        );
        assert_eq!(complementarity_text(&out, 3), None);
    }

    #[test]
    fn complementarity_labels_truncate_and_terminate() {
        let mut a = [0x7f as c_char; 4];
        copy_label(&mut a, "abcdef");
        assert_eq!(a, [b'a' as c_char, b'b' as c_char, b'c' as c_char, 0]);
        copy_label(&mut a, "x");
        assert_eq!(a, [b'x' as c_char, 0, 0, 0], "the tail is cleared");
        // Every stage label fits the 32-byte fields with room for the NUL.
        use pounce_algorithm::mpcc::MpccStage::*;
        for s in [Exact, Continuation, Penalty, Branch, Polish] {
            assert!(s.label().len() < 32, "{}", s.label());
        }
        use pounce_algorithm::mpcc::Strategy;
        for s in [Strategy::Both, Strategy::Exact, Strategy::Continuation] {
            assert!(s.name().len() < 16, "{}", s.name());
        }
    }

    // ----- AddPounceComplementsConstraint (gh#1147): the mapping and the
    // re-validation branch no public call reaches.

    #[test]
    fn complements_row_reads_every_table_row_and_refuses_nan_and_crossed() {
        use ComplementsRow::*;
        let inf = 2e19;
        assert_eq!(complements_row(0.0, 2.0), Some(Mixed));
        assert_eq!(complements_row(-1.0, inf), Some(Lower));
        assert_eq!(complements_row(-inf, 3.0), Some(Upper));
        assert_eq!(complements_row(-inf, inf), Some(RowFixed));
        assert_eq!(complements_row(1.0, 1.0), Some(VarFixed));
        // The ±1e19 convention of the pair classifier: exactly 1e19 is
        // infinite, just inside it is finite.
        assert_eq!(complements_row(-1e19, 1e19), Some(RowFixed));
        assert_eq!(complements_row(-9.9e18, 1e19), Some(Lower));
        assert_eq!(complements_row(2.0, 1.0), None);
        assert_eq!(complements_row(f64::NAN, 1.0), None);
        assert_eq!(complements_row(0.0, f64::NAN), None);
    }

    #[test]
    fn complements_edits_classify_as_the_native_shape_they_declare() {
        // The debug_assert in `add_complements`, as a test: for each pair
        // row of the table, the edited bounds of a free row are exactly the
        // native shape, and a no-pair row declares nothing.
        let inf = 2e19;
        for (xl, xu, want) in [
            (0.0, 2.0, ComplementsRow::Mixed),
            (-1.0, inf, ComplementsRow::Lower),
            (-inf, 3.0, ComplementsRow::Upper),
        ] {
            let row = complements_row(xl, xu).unwrap();
            assert_eq!(row, want);
            let (gl, gu, declares) = row.row_bounds(-inf, inf);
            assert!(declares);
            let shape = classify_pair(xl, xu, gl, gu);
            assert!(row.shape_matches(shape), "{row:?}: {shape:?}");
        }
        assert_eq!(
            ComplementsRow::RowFixed.row_bounds(-inf, inf),
            (0.0, 0.0, false)
        );
        assert_eq!(
            ComplementsRow::VarFixed.row_bounds(-inf, inf),
            (-inf, inf, false)
        );
        // `shape_matches` refuses a wrong shape, so the assert can fire.
        assert!(!ComplementsRow::Mixed.shape_matches(None));
        assert!(!ComplementsRow::Lower.shape_matches(classify_pair(0.0, 2.0, -inf, inf)));
    }

    /// Two variables, two free rows, no callback is ever called.
    fn create_two_by_two(x_l: [Number; 2], x_u: [Number; 2]) -> IpoptProblem {
        let g_l = [-2e19, -2e19];
        let g_u = [2e19, 2e19];
        let p = unsafe {
            CreateIpoptProblem(
                2,
                x_l.as_ptr(),
                x_u.as_ptr(),
                2,
                g_l.as_ptr(),
                g_u.as_ptr(),
                0,
                0,
                0,
                Some(quad_eval_f),
                Some(cb_quad_eval_g),
                Some(quad_eval_grad_f),
                Some(cb_quad_eval_jac_g),
                Some(cb_quad_eval_h),
            )
        };
        assert!(!p.is_null());
        p
    }

    /// Every (row, var) of the declared set and of the Complements ledger is
    /// found by the maps, and nothing else is.
    fn assert_maps_match(p: IpoptProblem) {
        let i = unsafe { &*p };
        for (k, q) in i.complementarity.iter().enumerate() {
            assert_eq!(i.complementarity.of_var(q.var), Some(k));
            assert_eq!(i.complementarity.of_row(q.row), Some(k));
        }
        for (k, u) in i.complements_used.iter().enumerate() {
            assert_eq!(i.complements_used.of_var(u.var), Some(k));
            assert_eq!(i.complements_used.of_row(u.row), Some(k));
        }
        for j in 0..i.n.max(i.m) as usize {
            assert_eq!(
                i.complementarity.of_var(j).is_some(),
                i.complementarity.iter().any(|q| q.var == j)
            );
            assert_eq!(
                i.complementarity.of_row(j).is_some(),
                i.complementarity.iter().any(|q| q.row == j)
            );
            assert_eq!(
                i.complements_used.of_var(j).is_some(),
                i.complements_used.iter().any(|u| u.var == j)
            );
            assert_eq!(
                i.complements_used.of_row(j).is_some(),
                i.complements_used.iter().any(|u| u.row == j)
            );
        }
    }

    #[test]
    fn pair_maps_follow_every_declaration_path() {
        // error-path: AddPounceComplementarityPair
        // gh#1147 Stage 2 (S2-I1): the one-pair declarations find duplicates
        // through var/row maps instead of re-validating the stored set, so
        // the maps must follow Add, Set (accepted and refused), Clear and
        // the Complements helper, and a refused call must leave them as
        // they were.
        let p = create_two_by_two([0.0, 0.0], [2.0, 2.0]);
        unsafe {
            assert_eq!(AddPounceComplementarityPair(p, 0, 0), POUNCE_CC_OK);
            assert_maps_match(p);
            assert_eq!(
                AddPounceComplementarityPair(p, 1, 0),
                POUNCE_CC_ERR_VAR_ALREADY
            );
            assert_eq!(
                AddPounceComplementarityPair(p, 0, 1),
                POUNCE_CC_ERR_ROW_ALREADY
            );
            assert_maps_match(p);
            let (rows, vars) = ([1, 0], [0, 1]);
            assert_eq!(
                SetPounceComplementarityPairs(
                    p,
                    2,
                    rows.as_ptr(),
                    vars.as_ptr(),
                    std::ptr::null_mut()
                ),
                POUNCE_CC_OK
            );
            assert_maps_match(p);
            assert_eq!((*p).complementarity.of_var(0), Some(0));
            assert_eq!((*p).complementarity.of_row(1), Some(0));
            let (rows, vars) = ([0, 0], [0, 1]);
            assert_eq!(
                SetPounceComplementarityPairs(
                    p,
                    2,
                    rows.as_ptr(),
                    vars.as_ptr(),
                    std::ptr::null_mut()
                ),
                POUNCE_CC_ERR_ROW_ALREADY
            );
            assert_eq!((*p).complementarity.of_row(1), Some(0));
            assert_maps_match(p);
            assert_eq!(ClearPounceComplementarityPairs(p), TRUE);
            assert_maps_match(p);
            assert_eq!((&*p).complementarity.of_var(0), None);
            assert_eq!(
                AddPounceComplementsConstraint(p, 1, 1),
                POUNCE_COMPLEMENTS_PAIR_MIXED
            );
            assert_maps_match(p);
            assert_eq!(
                AddPounceComplementsConstraint(p, 0, 1),
                POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
            );
            assert_maps_match(p);
            FreeIpoptProblem(p);
        }
    }

    #[test]
    fn complements_records_the_table_row_and_the_callers_row_bounds() {
        // gh#1147 Stage 2 (S2-I1): the ledger keeps what the edit replaced,
        // which is the user layer SetPounceBounds / GetPounceBounds serve.
        let p = create_two_by_two([0.0, -2e19], [2e19, 2e19]);
        unsafe {
            assert_eq!(
                AddPounceComplementsConstraint(p, 0, 0),
                POUNCE_COMPLEMENTS_PAIR_LOWER
            );
            assert_eq!(
                AddPounceComplementsConstraint(p, 1, 1),
                POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR
            );
            let i = &*p;
            assert_eq!(
                i.complements_used[..],
                [
                    ComplementsUse {
                        row: 0,
                        var: 0,
                        table: ComplementsRow::Lower,
                        user_gl: -2e19,
                        user_gu: 2e19
                    },
                    ComplementsUse {
                        row: 1,
                        var: 1,
                        table: ComplementsRow::RowFixed,
                        user_gl: -2e19,
                        user_gu: 2e19
                    },
                ]
            );
            assert_eq!((i.g_l[0], i.g_u[0]), (0.0, 2e19));
            assert_eq!((i.g_l[1], i.g_u[1]), (0.0, 0.0));
            FreeIpoptProblem(p);
        }
    }
}
