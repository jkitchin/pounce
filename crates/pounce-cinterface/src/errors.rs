//! One error namespace for the whole C API.
//!
//! Every negative code a POUNCE extension entry point returns is a
//! `POUNCE_ERR_*` value from this module, and each value has exactly one
//! meaning wherever it appears: a wrapper can keep a single code-to-message
//! table, and a log line that says `-3` means "index out of range" whichever
//! function wrote it. [`PounceErrorString`] is that table, so every frontend
//! reports the same text.
//!
//! The values are grouped in blocks:
//!
//! | block        | meaning                                                   |
//! |--------------|-----------------------------------------------------------|
//! | `-1 … -19`   | generic: any function may return these                    |
//! | `-20 … -39`  | complementarity pairs (`*PounceComplementarityPair*`, `AddPounceComplementsConstraint`) |
//! | `-40 … -59`  | bound updates (`SetPounceBounds`, `GetPounceBounds`)      |
//! | `-60 … -79`  | the QP handle (`PounceQp*`)                               |
//!
//! A new code is appended inside its block, and a number is never reused or
//! renumbered once released. A new function family gets a new block. A code
//! from one family's block may be returned by another family when the
//! meaning is the same (`POUNCE_ERR_ROW_NOT_FREE` is returned by both
//! `AddPounceComplementsConstraint` and `SetPounceBounds`).
//!
//! Every function checks in the same order, so a call with several faults
//! reports the same code whichever function it is:
//!
//! 1. the handle ([`POUNCE_ERR_NULL`]);
//! 2. `flags` and enum arguments ([`POUNCE_ERR_FLAGS`], [`POUNCE_ERR_ARGUMENT`]);
//! 3. counts ([`POUNCE_ERR_COUNT`]), then the arrays they size (`NULL`);
//! 4. per entry, over the whole batch in batch order: range, then duplicate
//!    or already paired, then NaN;
//! 5. per entry, over the whole batch in batch order: what the entry would
//!    mean (shape, row not free, variable bounds, `Complements` class, pair
//!    changed);
//! 6. state beyond the batch (the staged working set; `PounceQpSolve`'s
//!    checks).
//!
//! An argument fault anywhere in a batch is therefore reported before a
//! semantic fault anywhere in it. `tests/check_order.rs` holds the batch
//! functions to this.
//!
//! Non-negative returns stay per function: `POUNCE_OK` (0) for success, and
//! the QP statuses (`POUNCE_QP_STATUS_*`) and `AddPounceComplementsConstraint`
//! outcomes (`POUNCE_COMPLEMENTS_*`) are results, not errors.
//!
//! The header (`include/pounce.h`) repeats every value; `tests/header_abi.rs`
//! holds the two to the same names and values, and checks that
//! [`PounceErrorString`] has a message for each.

use std::ffi::{c_char, c_int};

/// Success.
pub const POUNCE_OK: c_int = 0;

// --- generic, -1 … -19 -----------------------------------------------------

/// A NULL handle, or a NULL array the call needed.
pub const POUNCE_ERR_NULL: c_int = -1;
/// A count out of range: negative, larger than the dimension it indexes, or
/// inconsistent with a NULL index array.
pub const POUNCE_ERR_COUNT: c_int = -2;
/// An index outside `[0, n)` / `[0, m)`, read in the problem's index style.
pub const POUNCE_ERR_OUT_OF_RANGE: c_int = -3;
/// The same index given twice where each may appear once.
pub const POUNCE_ERR_DUPLICATE: c_int = -4;
/// A NaN value where a number is required.
pub const POUNCE_ERR_NAN: c_int = -5;
/// `flags` carries a bit this library does not define. An unknown bit is
/// refused, so an older library never silently ignores one.
pub const POUNCE_ERR_FLAGS: c_int = -6;
/// An argument outside its domain: an unknown enum value (a view, a cone
/// kind), a bad cone dimension or exponent, or entries whose sum overflows.
pub const POUNCE_ERR_ARGUMENT: c_int = -7;
/// A Rust panic caught at the boundary, or a handle whose stored state no
/// longer validates. Not expected; it exists so neither is reported as a
/// diagnosis of the caller's input. Nothing was changed.
pub const POUNCE_ERR_INTERNAL: c_int = -8;

// --- complementarity pairs, -20 … -39 --------------------------------------

/// The variable is already in a pair (declared earlier, or earlier in the
/// same batch or an earlier accepted call).
pub const POUNCE_ERR_VAR_ALREADY_PAIRED: c_int = -20;
/// The row is already in a pair (declared earlier, or earlier in the same
/// batch or an earlier accepted call).
pub const POUNCE_ERR_ROW_ALREADY_PAIRED: c_int = -21;
/// The pair's bounds are none of the lowered shapes: one-sided, one-sided
/// plus a plain bound, or mixed.
pub const POUNCE_ERR_SHAPE: c_int = -22;
/// A row that must stay a free function was given (or has) a finite bound:
/// MOI's `F` in `AddPounceComplementsConstraint`, which `SetPounceBounds`
/// then refuses to bound.
pub const POUNCE_ERR_ROW_NOT_FREE: c_int = -23;
/// The variable's bounds are NaN or crossed (`l > u`): no row of the MOI
/// complements table applies.
pub const POUNCE_ERR_VAR_BOUNDS: c_int = -24;

// --- bound updates, -40 … -59 ----------------------------------------------

/// A declared pair would leave its native shapes, or change which variable
/// bound and which row side it pairs.
pub const POUNCE_ERR_PAIR_CHANGED: c_int = -40;
/// A variable used by `AddPounceComplementsConstraint` would move to another
/// row of the MOI table (lower-only, upper-only, boxed, fixed, free).
pub const POUNCE_ERR_COMPLEMENTS_CLASS: c_int = -41;
/// The working set staged by `IpoptSetWarmStartWorkingSet` would claim
/// something the new bounds contradict.
pub const POUNCE_ERR_WORKING_SET: c_int = -42;

// --- QP handle, -60 … -79 --------------------------------------------------

/// The QP is incomplete: no objective, or declared rows never set.
pub const POUNCE_ERR_INCOMPLETE: c_int = -60;
/// The cones do not partition the inequality rows, or PSD is mixed with the
/// exponential or power cone.
pub const POUNCE_ERR_CONES: c_int = -61;
/// An option the selected engine would silently drop.
pub const POUNCE_ERR_OPTIONS: c_int = -62;
/// `method=ipm` and `P` not certified positive semidefinite.
pub const POUNCE_ERR_INDEFINITE: c_int = -63;

/// Every code with its message: the single table [`PounceErrorString`]
/// reads, and the list `tests/header_abi.rs` checks the header against.
pub const POUNCE_ERROR_CODES: &[(&str, c_int, &std::ffi::CStr)] = &[
    ("POUNCE_OK", POUNCE_OK, c"success"),
    ("POUNCE_ERR_NULL", POUNCE_ERR_NULL, c"NULL handle or array"),
    ("POUNCE_ERR_COUNT", POUNCE_ERR_COUNT, c"count out of range"),
    (
        "POUNCE_ERR_OUT_OF_RANGE",
        POUNCE_ERR_OUT_OF_RANGE,
        c"index out of range",
    ),
    (
        "POUNCE_ERR_DUPLICATE",
        POUNCE_ERR_DUPLICATE,
        c"index given more than once",
    ),
    ("POUNCE_ERR_NAN", POUNCE_ERR_NAN, c"NaN value"),
    (
        "POUNCE_ERR_FLAGS",
        POUNCE_ERR_FLAGS,
        c"unknown bit in flags",
    ),
    (
        "POUNCE_ERR_ARGUMENT",
        POUNCE_ERR_ARGUMENT,
        c"argument outside its domain",
    ),
    (
        "POUNCE_ERR_INTERNAL",
        POUNCE_ERR_INTERNAL,
        c"internal error; nothing was changed",
    ),
    (
        "POUNCE_ERR_VAR_ALREADY_PAIRED",
        POUNCE_ERR_VAR_ALREADY_PAIRED,
        c"variable already in a complementarity pair",
    ),
    (
        "POUNCE_ERR_ROW_ALREADY_PAIRED",
        POUNCE_ERR_ROW_ALREADY_PAIRED,
        c"row already in a complementarity pair",
    ),
    (
        "POUNCE_ERR_SHAPE",
        POUNCE_ERR_SHAPE,
        c"complementarity pair has no supported shape",
    ),
    (
        "POUNCE_ERR_ROW_NOT_FREE",
        POUNCE_ERR_ROW_NOT_FREE,
        c"row must stay free (no finite bound)",
    ),
    (
        "POUNCE_ERR_VAR_BOUNDS",
        POUNCE_ERR_VAR_BOUNDS,
        c"variable bounds NaN or crossed",
    ),
    (
        "POUNCE_ERR_PAIR_CHANGED",
        POUNCE_ERR_PAIR_CHANGED,
        c"bound update would change a declared pair",
    ),
    (
        "POUNCE_ERR_COMPLEMENTS_CLASS",
        POUNCE_ERR_COMPLEMENTS_CLASS,
        c"bound update would change a Complements variable's class",
    ),
    (
        "POUNCE_ERR_WORKING_SET",
        POUNCE_ERR_WORKING_SET,
        c"bound update contradicts the staged working set",
    ),
    (
        "POUNCE_ERR_INCOMPLETE",
        POUNCE_ERR_INCOMPLETE,
        c"QP incomplete: no objective, or rows never set",
    ),
    (
        "POUNCE_ERR_CONES",
        POUNCE_ERR_CONES,
        c"cones do not partition the inequality rows",
    ),
    (
        "POUNCE_ERR_OPTIONS",
        POUNCE_ERR_OPTIONS,
        c"option the selected engine would drop",
    ),
    (
        "POUNCE_ERR_INDEFINITE",
        POUNCE_ERR_INDEFINITE,
        c"P not certified positive semidefinite",
    ),
];

/// The message for a `POUNCE_OK` / `POUNCE_ERR_*` code, as a static
/// NUL-terminated string the caller must not free; `"unknown error code"`
/// for any other value. Never NULL.
///
/// Only the shared codes are covered: a non-negative function-specific
/// result (a QP status, an `AddPounceComplementsConstraint` outcome) other
/// than `POUNCE_OK` reads as unknown.
///
/// # Safety
///
/// None required: any `code` is accepted. It is `unsafe` only to match the
/// rest of the exported surface.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PounceErrorString(code: c_int) -> *const c_char {
    POUNCE_ERROR_CODES
        .iter()
        .find(|(_, c, _)| *c == code)
        .map_or(c"unknown error code", |(_, _, msg)| *msg)
        .as_ptr()
}
