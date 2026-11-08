//! gh#1147 — `AddPounceComplementsConstraint`: MOI `[F, x] ∈ Complements`
//! semantics on top of the native pairs.
//!
//! One test per row of the MOI table (`src/lib.rs`, the function's docs),
//! each on its own small model whose answer is worked out by hand in the
//! test's comment — so the expected point does not come from pounce — and
//! solved through `IpoptSolve`. Every row that yields a pair or an equality
//! is solved twice, from two targets chosen so that one solve is decided by
//! the pair (or the equality's one side) and the other by the sign MOI
//! implies (or the equality's other side): a single solve could be right
//! with half the mapping missing (CLAUDE.md, "A test is only evidence about
//! the branch it reaches"). The no-pair row with a fixed variable is solved
//! on both sides of `F = 0` to show `F` stayed free.
//!
//! The lower row is also checked against the **native** formulation of the
//! same model — the row bound `F >= 0` given to `CreateIpoptProblem` and
//! the pair declared with `AddPounceComplementarityPair` — to the bit, and
//! against Python's `Problem(complementarity=)` on it (values in
//! [`PY_LOWER`], from the venv's editable build of this branch).
//!
//! Then every refusal (`// error-path:`), each on a handle that already
//! holds a declaration, asserting nothing moved; and the handling of
//! repeated rows and variables and of rows whose bounds are not free.
//!
//! The model family, through `user_data`: two variables `(x, y)`,
//!
//! ```text
//! min (x - a)^2 + (y - b)^2,   one row  F = cx x + cy y - s,
//! ```
//!
//! the pair being `[F, x]`; `x`'s bounds pick the table row.

// Test code: a panicking unwrap/expect is the test failing, as intended.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use std::ffi::{CString, c_void};
use std::ptr;

const INF: Number = 2e19;

#[derive(Clone, Copy)]
struct Params {
    a: Number,
    b: Number,
    cx: Number,
    cy: Number,
    s: Number,
}

unsafe fn params<'a>(u: *mut c_void) -> &'a Params {
    unsafe { &*(u as *const Params) }
}

unsafe extern "C" fn ev_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    obj: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let p = params(u);
        let z = std::slice::from_raw_parts(x, 2);
        *obj = (z[0] - p.a).powi(2) + (z[1] - p.b).powi(2);
        1
    }
}

unsafe extern "C" fn ev_grad_f(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    grad: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let p = params(u);
        let z = std::slice::from_raw_parts(x, 2);
        let g = std::slice::from_raw_parts_mut(grad, 2);
        g[0] = 2.0 * (z[0] - p.a);
        g[1] = 2.0 * (z[1] - p.b);
        1
    }
}

unsafe extern "C" fn ev_g(
    _n: Index,
    x: *const Number,
    _new_x: Bool,
    _m: Index,
    gout: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        let p = params(u);
        let z = std::slice::from_raw_parts(x, 2);
        *gout = p.cx * z[0] + p.cy * z[1] - p.s;
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_jac_g(
    _n: Index,
    _x: *const Number,
    _new_x: Bool,
    _m: Index,
    _nele_jac: Index,
    i_row: *mut Index,
    j_col: *mut Index,
    values: *mut Number,
    u: *mut c_void,
) -> Bool {
    unsafe {
        if values.is_null() {
            let r = std::slice::from_raw_parts_mut(i_row, 2);
            let c = std::slice::from_raw_parts_mut(j_col, 2);
            r.copy_from_slice(&[0, 0]);
            c.copy_from_slice(&[0, 1]);
            return 1;
        }
        let p = params(u);
        std::slice::from_raw_parts_mut(values, 2).copy_from_slice(&[p.cx, p.cy]);
        1
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn ev_h(
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
            let r = std::slice::from_raw_parts_mut(i_row, 2);
            let c = std::slice::from_raw_parts_mut(j_col, 2);
            r.copy_from_slice(&[0, 1]);
            c.copy_from_slice(&[0, 1]);
            return 1;
        }
        std::slice::from_raw_parts_mut(values, 2).fill(2.0 * obj_factor);
        1
    }
}

/// The model with `x ∈ [xl, xu]`, `y` free and the row bounds `[gl, gu]`
/// (free unless a test gives the native formulation).
fn create_with_row(xl: Number, xu: Number, gl: Number, gu: Number) -> IpoptProblem {
    let x_l = [xl, -INF];
    let x_u = [xu, INF];
    let (g_l, g_u) = ([gl], [gu]);
    let prob = unsafe {
        CreateIpoptProblem(
            2,
            x_l.as_ptr(),
            x_u.as_ptr(),
            1,
            g_l.as_ptr(),
            g_u.as_ptr(),
            2,
            2,
            0,
            Some(ev_f),
            Some(ev_g),
            Some(ev_grad_f),
            Some(ev_jac_g),
            Some(ev_h),
        )
    };
    assert!(!prob.is_null());
    let k = CString::new("print_level").unwrap();
    assert_eq!(unsafe { AddIpoptIntOption(prob, k.as_ptr(), 0) }, 1);
    let (k, v) = (CString::new("sb").unwrap(), CString::new("yes").unwrap());
    assert_eq!(
        unsafe { AddIpoptStrOption(prob, k.as_ptr(), v.as_ptr()) },
        1
    );
    prob
}

fn create(xl: Number, xu: Number) -> IpoptProblem {
    create_with_row(xl, xu, -INF, INF)
}

fn complements(prob: IpoptProblem, row: Index, var: Index) -> i32 {
    unsafe { AddPounceComplementsConstraint(prob, row, var) }
}

fn declared(prob: IpoptProblem) -> Vec<(Index, Index)> {
    unsafe {
        let k = GetPounceComplementarityPairCount(prob);
        assert!(k >= 0);
        let mut rows = vec![-99; k as usize];
        let mut vars = vec![-99; k as usize];
        assert_eq!(
            GetPounceComplementarityPairs(prob, rows.as_mut_ptr(), vars.as_mut_ptr()),
            1
        );
        rows.into_iter().zip(vars).collect()
    }
}

#[derive(Debug)]
struct Solved {
    status: Index,
    x: [Number; 2],
    g: Number,
    obj: Number,
    /// The driver's total iterations over every lowered solve; `None` when
    /// the driver did not run (no pair declared).
    total_iterations: Option<Index>,
}

fn solve(prob: IpoptProblem, p: Params, x0: [Number; 2]) -> Solved {
    let mut x = x0;
    let mut g = [Number::NAN];
    let mut obj = Number::NAN;
    let mut data = p;
    let status = unsafe {
        IpoptSolve(
            prob,
            x.as_mut_ptr(),
            g.as_mut_ptr(),
            &mut obj,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut data as *mut Params as *mut c_void,
        )
    };
    let mut stats = std::mem::MaybeUninit::<PounceComplementarityStats>::zeroed();
    unsafe { (*stats.as_mut_ptr()).struct_size = size_of::<PounceComplementarityStats>() as Index };
    let total_iterations = (unsafe { GetPounceComplementarityStats(prob, stats.as_mut_ptr()) }
        == 1)
        .then(|| unsafe { stats.assume_init() }.total_iterations);
    Solved {
        status,
        x,
        g: g[0],
        obj,
        total_iterations,
    }
}

const SOLVE_SUCCEEDED: Index = 0;

/// The solve succeeded at `x*` (to `1e-6`), with `F(x*) = f_star` and the
/// objective `obj`.
fn assert_answer(s: &Solved, x_star: [Number; 2], f_star: Number, obj: Number) {
    assert_eq!(s.status, SOLVE_SUCCEEDED, "{s:?}");
    for (k, (got, want)) in s.x.iter().zip(x_star).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "x[{k}] = {got} against {want} ({s:?})"
        );
    }
    assert!((s.g - f_star).abs() < 1e-6, "F = {} against {f_star}", s.g);
    assert!((s.obj - obj).abs() < 1e-6, "f = {} against {obj}", s.obj);
}

fn free(prob: IpoptProblem) {
    unsafe { FreeIpoptProblem(prob) }
}

// ---------------------------------------------------------------------------
// The five rows
// ---------------------------------------------------------------------------

/// Row 1 — `x ∈ [0, 2]`, `F = x + y` free: a native mixed pair, the row
/// left free (a second call on it is `ROW_ALREADY_PAIRED`, not
/// `ROW_NOT_FREE`, which it would be had a bound been written).
///
/// Branches: `x = 0, F >= 0` / `0 < x < 2, F = 0` / `x = 2, F <= 0`.
/// * Target `(1, 1)` (unconstrained optimum, `F = 2`, interior `x`, so the
///   pair cuts it off): on `F = 0` the objective `2x^2 + 2` has no
///   interior minimum; on `x = 2, y <= -2` the best is `(2, -2)`, 10;
///   on `x = 0, y >= 0` it is `(0, 1)`, 1, where `F = 1 > 0`. Answer
///   `(0, 1)`: the lower end.
/// * Target `(1.5, -5)` (`F = -3.5`, interior): `x = 2, y <= -2` gives
///   `(2, -5)`, 0.25, `F = -3`; the `F = 0` branch runs to `(2, -2)`, and
///   `x = 0` to `(0, 0)`, both worse and neither a local minimum. Answer
///   `(2, -5)`: the upper end.
#[test]
fn mixed_row_declares_a_native_mixed_pair_and_leaves_the_row_free() {
    let mixed = |a, b| Params {
        a,
        b,
        cx: 1.0,
        cy: 1.0,
        s: 0.0,
    };
    let prob = create(0.0, 2.0);
    assert_eq!(complements(prob, 0, 0), POUNCE_COMPLEMENTS_PAIR_MIXED);
    assert_eq!(declared(prob), vec![(0, 0)]);
    assert_eq!(
        complements(prob, 0, 1),
        POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED
    );
    let s = solve(prob, mixed(1.0, 1.0), [1.0, 0.5]);
    assert_answer(&s, [0.0, 1.0], 1.0, 1.0);
    assert!(s.total_iterations.is_some(), "the driver did not run");
    let s = solve(prob, mixed(1.5, -5.0), [1.0, 0.5]);
    assert_answer(&s, [2.0, -5.0], -3.0, 0.25);
    free(prob);
}

/// Row 2's model: `x >= 1`, `F = x + y - 4`.
fn lower(a: Number, b: Number) -> Params {
    Params {
        a,
        b,
        cx: 1.0,
        cy: 1.0,
        s: 4.0,
    }
}

/// Row 2 — `x >= 1` only, `F = x + y - 4` free: the helper writes
/// `g_L = 0` (`F >= 0`) and declares the one-sided pair
/// `(x - 1) · F = 0`. The edit is observable: a second call on the row is
/// `ROW_NOT_FREE`.
///
/// Branches: `x = 1, F >= 0` / `x > 1, F = 0`.
/// * Target `(2, 5)` (`F = 3 > 0`, so `F >= 0` alone keeps it; the pair
///   decides): on `F = 0` the objective `(x-2)^2 + (x+1)^2` is least at
///   `x = 0.5 < 1`, so that branch runs to `(1, 3)`, 2; on `x = 1` it is
///   `(1, 5)`, 1, `F = 2`. Answer `(1, 5)`.
/// * Target `(0, 0)` (`x < 1`; the plain bound gives `(1, 0)`, `F = -3`,
///   which the pair without the sign would accept on `x = 1`; the sign
///   decides): `x = 1, y >= 3` gives `(1, 3)`, 10, not a local minimum
///   (the objective falls along `F = 0` from it); on `F = 0`, `x^2 +
///   (4-x)^2` is least at `x = 2`. Answer `(2, 2)`, 8, `F = 0`.
#[test]
fn lower_row_writes_f_nonnegative_and_declares_a_one_sided_pair() {
    let prob = create(1.0, INF);
    assert_eq!(complements(prob, 0, 0), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(declared(prob), vec![(0, 0)]);
    assert_eq!(complements(prob, 0, 1), POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE);
    let s = solve(prob, lower(2.0, 5.0), [2.0, 2.0]);
    assert_answer(&s, [1.0, 5.0], 2.0, 1.0);
    let s = solve(prob, lower(0.0, 0.0), [2.0, 2.0]);
    assert_answer(&s, [2.0, 2.0], 0.0, 8.0);
    free(prob);
}

/// Python's `Problem(n=2, m=1, lb=[1, -2e19], ub=[2e19, 2e19], cl=[0],
/// cu=[2e19], complementarity=[(0, 0)]).solve([2, 2])` on the row-2 model
/// with `print_level 0`, both targets: `x`, objective and the driver's
/// `total_iterations`, printed with `repr` (round-trip exact). Both
/// `Solve_Succeeded`, class `S`.
/// `((a, b), x, objective, total_iterations)`.
type PyRow = ((Number, Number), [Number; 2], Number, Index);

const PY_LOWER: [PyRow; 2] = [
    (
        (2.0, 5.0),
        [1.0000000037514962, 4.999999998124025],
        0.9999999924970076,
        43,
    ),
    (
        (0.0, 0.0),
        [2.0000000181543895, 1.9999999818456105],
        8.0,
        40,
    ),
];

/// Row 2 against the native formulation: the same model created with the
/// row bound `F >= 0` and the pair declared through
/// `AddPounceComplementarityPair` solves **bit for bit** like the helper's
/// handle, on both targets; and both equal Python's native
/// `Problem(complementarity=)` ([`PY_LOWER`]).
#[test]
fn lower_row_equals_the_native_formulation_and_python_to_the_bit() {
    for ((a, b), py_x, py_obj, py_iters) in PY_LOWER {
        let moi = create(1.0, INF);
        assert_eq!(complements(moi, 0, 0), POUNCE_COMPLEMENTS_PAIR_LOWER);
        let native = create_with_row(1.0, INF, 0.0, INF);
        assert_eq!(
            unsafe { AddPounceComplementarityPair(native, 0, 0) },
            POUNCE_CC_OK
        );
        let s_moi = solve(moi, lower(a, b), [2.0, 2.0]);
        let s_nat = solve(native, lower(a, b), [2.0, 2.0]);
        assert_eq!(s_moi.status, SOLVE_SUCCEEDED);
        assert_eq!(s_moi.status, s_nat.status);
        assert_eq!(s_moi.x.map(f64::to_bits), s_nat.x.map(f64::to_bits));
        assert_eq!(s_moi.g.to_bits(), s_nat.g.to_bits());
        assert_eq!(s_moi.obj.to_bits(), s_nat.obj.to_bits());
        assert_eq!(s_moi.total_iterations, s_nat.total_iterations);
        assert_eq!(
            s_moi.x.map(f64::to_bits),
            py_x.map(f64::to_bits),
            "target ({a}, {b}): C {:?} against Python {py_x:?}",
            s_moi.x
        );
        assert_eq!(s_moi.obj.to_bits(), py_obj.to_bits());
        assert_eq!(s_moi.total_iterations, Some(py_iters));
        free(moi);
        free(native);
    }
}

/// Row 3 — `x <= 2` only, `F = x - y + 1` free: the mirror of row 2. The
/// helper writes `g_U = 0` (`F <= 0`) and declares the one-sided pair
/// `(2 - x) · F = 0`; a second call on the row is `ROW_NOT_FREE`.
///
/// Branches: `x = 2, F <= 0` (`y >= 3`) / `x < 2, F = 0` (`y = x + 1`).
/// * Target `(1, 5)` (`F = -3 < 0`, so `F <= 0` alone keeps it; the pair
///   decides): on `F = 0`, `(x-1)^2 + (x-4)^2` is least at `x = 2.5 > 2`,
///   so that branch runs to `(2, 3)`, 5; on `x = 2` the best is `(2, 5)`,
///   1, `F = -2`. Answer `(2, 5)`.
/// * Target `(3, 0)` (`x > 2`; the plain bound gives `(2, 0)`, `F = 3`,
///   which the pair without the sign would accept on `x = 2`; the sign
///   decides): `x = 2, y >= 3` gives `(2, 3)`, 10, not a local minimum; on
///   `F = 0`, `(x-3)^2 + (x+1)^2` is least at `x = 1`. Answer `(1, 2)`, 8,
///   `F = 0`.
#[test]
fn upper_row_writes_f_nonpositive_and_declares_a_one_sided_pair() {
    let upper = |a, b| Params {
        a,
        b,
        cx: 1.0,
        cy: -1.0,
        s: -1.0,
    };
    let prob = create(-INF, 2.0);
    assert_eq!(complements(prob, 0, 0), POUNCE_COMPLEMENTS_PAIR_UPPER);
    assert_eq!(declared(prob), vec![(0, 0)]);
    assert_eq!(complements(prob, 0, 1), POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE);
    let s = solve(prob, upper(1.0, 5.0), [1.0, 1.0]);
    assert_answer(&s, [2.0, 5.0], -2.0, 1.0);
    let s = solve(prob, upper(3.0, 0.0), [1.0, 1.0]);
    assert_answer(&s, [1.0, 2.0], 0.0, 8.0);
    free(prob);
}

/// Row 4 — `x` free, `F = x + y - 3` free: the helper fixes the row,
/// `g_L = g_U = 0` (`F = 0`), and declares no pair; a second call on the
/// row is `ROW_NOT_FREE`. With no pair the solve is the plain NLP path.
///
/// The answer is the projection of the target on `x + y = 3`:
/// * target `(1, 5)` (`F = 3 > 0`: `F <= 0` would bind, `F >= 0` would
///   not) → `(-0.5, 3.5)`, 4.5;
/// * target `(1, 1)` (`F = -1 < 0`: the other side) → `(1.5, 1.5)`, 0.5.
#[test]
fn free_variable_fixes_the_row_to_an_equality_and_declares_no_pair() {
    let free_x = |a, b| Params {
        a,
        b,
        cx: 1.0,
        cy: 1.0,
        s: 3.0,
    };
    let prob = create(-INF, INF);
    assert_eq!(
        complements(prob, 0, 0),
        POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR
    );
    assert_eq!(declared(prob), vec![]);
    assert_eq!(complements(prob, 0, 1), POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE);
    let s = solve(prob, free_x(1.0, 5.0), [0.0, 0.0]);
    assert_answer(&s, [-0.5, 3.5], 0.0, 4.5);
    assert_eq!(s.total_iterations, None, "the driver ran without a pair");
    let s = solve(prob, free_x(1.0, 1.0), [0.0, 0.0]);
    assert_answer(&s, [1.5, 1.5], 0.0, 0.5);
    free(prob);
}

/// Row 5 — `x` fixed at `1`, `F = x + y - 3` free: nothing is written, no
/// pair declared. A second call on the row is `ROW_ALREADY_PAIRED` (the
/// helper remembers the row), not `ROW_NOT_FREE` — so no bound was written.
///
/// With `x = 1` the answer is `(1, b)`, objective 0, on either side of
/// `F = 0`:
/// * target `(1, 1)`: `F = -1` (a written `F >= 0` or `F = 0` would move
///   it to `(1, 2)`);
/// * target `(1, 5)`: `F = 3` (a written `F <= 0` would move it to
///   `(1, 2)`).
#[test]
fn fixed_variable_declares_no_pair_and_leaves_f_free() {
    let fixed = |a, b| Params {
        a,
        b,
        cx: 1.0,
        cy: 1.0,
        s: 3.0,
    };
    let prob = create(1.0, 1.0);
    assert_eq!(
        complements(prob, 0, 0),
        POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR
    );
    assert_eq!(declared(prob), vec![]);
    assert_eq!(
        complements(prob, 0, 1),
        POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED
    );
    let s = solve(prob, fixed(1.0, 1.0), [1.0, 0.0]);
    assert_answer(&s, [1.0, 1.0], -1.0, 0.0);
    assert_eq!(s.total_iterations, None);
    let s = solve(prob, fixed(1.0, 5.0), [1.0, 0.0]);
    assert_answer(&s, [1.0, 5.0], 3.0, 0.0);
    free(prob);
}

// ---------------------------------------------------------------------------
// Already-declared pairs stay valid; equivalence on QpecSmall
// ---------------------------------------------------------------------------

/// QpecSmall with row 1 created free and `y2 >= 0`: the native pair
/// `(row 0, y1)` declared first, then the helper on `(row 1, y2)` writes
/// `g_L[1] = 0` — which is QpecSmall exactly. The pair declared before the
/// edit is still listed and still valid (re-declaring the whole set
/// through `SetPounceComplementarityPairs`, which validates against the
/// edited bounds, is accepted), and the solve equals the native QpecSmall
/// solve to the bit.
#[test]
fn an_edit_keeps_earlier_pairs_valid_and_rebuilds_qpec_small_exactly() {
    let mut g_l = qpec::G_L;
    g_l[1] = -INF;
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &g_l, &qpec::G_U, 0);
    assert_eq!(
        unsafe { AddPounceComplementarityPair(prob, 0, 1) },
        POUNCE_CC_OK
    );
    assert_eq!(complements(prob, 1, 2), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(declared(prob), vec![(0, 1), (1, 2)]);
    let rows = [0, 1];
    let vars = [1, 2];
    let mut bad = -42;
    assert_eq!(
        unsafe { SetPounceComplementarityPairs(prob, 2, rows.as_ptr(), vars.as_ptr(), &mut bad) },
        POUNCE_CC_OK
    );
    assert_eq!(bad, -1);

    let native = qpec::create_qpec(0);
    for (r, v) in qpec::PAIRS {
        assert_eq!(
            unsafe { AddPounceComplementarityPair(native, r, v) },
            POUNCE_CC_OK
        );
    }
    let run = |p: IpoptProblem| {
        let k = CString::new("print_level").unwrap();
        unsafe { AddIpoptIntOption(p, k.as_ptr(), 0) };
        let mut x = [0.0, 0.5, 0.5];
        let mut obj = Number::NAN;
        let st = unsafe {
            IpoptSolve(
                p,
                x.as_mut_ptr(),
                ptr::null_mut(),
                &mut obj,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        (st, x.map(f64::to_bits), obj.to_bits(), unsafe {
            GetIpoptIterCount(p)
        })
    };
    let (a, b) = (run(prob), run(native));
    assert_eq!(a.0, SOLVE_SUCCEEDED);
    assert_eq!(a, b);
    let x = a.1.map(f64::from_bits);
    assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6 && x[2].abs() < 1e-6);
    free(prob);
    free(native);
}

// ---------------------------------------------------------------------------
// Refusals. Each runs on a handle that already holds a declaration and
// checks it is unchanged: the pair list, and the row still taking the
// helper (so no bound was written).
// ---------------------------------------------------------------------------

/// A two-row handle on QpecSmall's callbacks (never solved): `x ∈ [0, 2]`,
/// `y1, y2 >= 0`, both rows free, `(row 0, x)` declared mixed by the
/// helper.
fn two_row_handle(index_style: Index) -> IpoptProblem {
    let g = [-INF, -INF];
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &g, &[INF, INF], index_style);
    let base = if index_style == 1 { 1 } else { 0 };
    assert_eq!(complements(prob, base, base), POUNCE_COMPLEMENTS_PAIR_MIXED);
    prob
}

/// After a refusal: the declared set is still `[(row 0, x)]` and row 1 /
/// `y1` (`y1 >= 0`) still takes the helper as the lower row — proving row
/// 1 was not written and `y1` not recorded.
fn assert_untouched(prob: IpoptProblem) {
    assert_eq!(declared(prob), vec![(0, 0)]);
    assert_eq!(complements(prob, 1, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(declared(prob), vec![(0, 0), (1, 1)]);
}

#[test]
fn null_handle_is_refused() {
    // error-path: AddPounceComplementsConstraint
    assert_eq!(
        complements(ptr::null_mut(), 0, 0),
        POUNCE_COMPLEMENTS_ERR_NULL
    );
}

#[test]
fn out_of_range_indices_are_refused() {
    // error-path: AddPounceComplementsConstraint
    for (row, var) in [
        (-1, 1),
        (2, 1),
        (1, -1),
        (1, 3),
        (Index::MIN, 1),
        (Index::MAX, 1),
        (1, Index::MIN),
        (1, Index::MAX),
    ] {
        let prob = two_row_handle(0);
        assert_eq!(
            complements(prob, row, var),
            POUNCE_COMPLEMENTS_ERR_OUT_OF_RANGE,
            "({row}, {var})"
        );
        assert_untouched(prob);
        free(prob);
    }
}

#[test]
fn fortran_style_indices_are_one_based() {
    // error-path: AddPounceComplementsConstraint
    let prob = two_row_handle(1);
    for (row, var) in [(0, 2), (2, 0), (3, 2), (2, 4), (Index::MIN, 2)] {
        assert_eq!(
            complements(prob, row, var),
            POUNCE_COMPLEMENTS_ERR_OUT_OF_RANGE,
            "({row}, {var})"
        );
    }
    assert_eq!(complements(prob, 2, 2), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(declared(prob), vec![(1, 1), (2, 2)]);
    free(prob);
}

/// Every non-free row is refused, whether its bound agrees with the one MOI
/// would imply (`g_L = 0` for `y1 >= 0`), conflicts with it (`g_U = 5`, or
/// `g_L = 1`), or is an equality or a box. Such a row is a native
/// declaration (`AddPounceComplementarityPair`), not MOI semantics; the
/// plan settles this (§3.1).
#[test]
fn a_row_with_any_finite_bound_is_refused_as_not_free() {
    // error-path: AddPounceComplementsConstraint
    for (gl, gu) in [
        (0.0, INF),
        (1.0, INF),
        (-INF, 5.0),
        (-INF, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (Number::NAN, INF),
    ] {
        // Row 1 has the bounds under test; row 0 free, (row 0, x) declared.
        let prob = qpec::create(&qpec::X_L, &qpec::X_U, &[-INF, gl], &[INF, gu], 0);
        assert_eq!(complements(prob, 0, 0), POUNCE_COMPLEMENTS_PAIR_MIXED);
        assert_eq!(
            complements(prob, 1, 1),
            POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE,
            "row [{gl}, {gu}]"
        );
        assert_eq!(declared(prob), vec![(0, 0)]);
        // Nothing recorded: y1 still takes a native pair on that row when
        // its shape allows one.
        if (gl, gu) == (0.0, INF) {
            assert_eq!(
                unsafe { AddPounceComplementarityPair(prob, 1, 1) },
                POUNCE_CC_OK
            );
        }
        free(prob);
    }
}

#[test]
fn a_variable_already_in_a_pair_is_refused() {
    // error-path: AddPounceComplementsConstraint
    // In a native pair (row 0, x), the helper's mixed one here.
    let prob = two_row_handle(0);
    assert_eq!(
        complements(prob, 1, 0),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    assert_untouched(prob);
    free(prob);
    // In a pair declared natively, not through the helper.
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &[0.0, -INF], &[INF, INF], 0);
    assert_eq!(
        unsafe { AddPounceComplementarityPair(prob, 0, 1) },
        POUNCE_CC_OK
    );
    assert_eq!(
        complements(prob, 1, 1),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    assert_eq!(declared(prob), vec![(0, 1)]);
    assert_eq!(complements(prob, 1, 2), POUNCE_COMPLEMENTS_PAIR_LOWER);
    free(prob);
}

#[test]
fn a_row_already_in_a_pair_is_refused() {
    // error-path: AddPounceComplementsConstraint
    // The helper's mixed pair leaves row 0 free; reusing it is refused.
    let prob = two_row_handle(0);
    assert_eq!(
        complements(prob, 0, 1),
        POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED
    );
    assert_untouched(prob);
    free(prob);
    // A native mixed pair on a free row, then the helper on that row.
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &[-INF, -INF], &[INF, INF], 0);
    assert_eq!(
        unsafe { AddPounceComplementarityPair(prob, 1, 0) },
        POUNCE_CC_OK
    );
    assert_eq!(
        complements(prob, 1, 2),
        POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED
    );
    assert_eq!(declared(prob), vec![(1, 0)]);
    free(prob);
}

/// Crossed or NaN variable bounds match no row of the table. Not in the
/// plan's error list (§3.1); refused rather than guessed at.
#[test]
fn crossed_or_nan_variable_bounds_are_refused() {
    // error-path: AddPounceComplementsConstraint
    for (xl, xu) in [(3.0, 1.0), (Number::NAN, 1.0), (0.0, Number::NAN)] {
        let x_l = [xl, 0.0, 0.0];
        let x_u = [xu, INF, INF];
        let prob = qpec::create(&x_l, &x_u, &[-INF, -INF], &[INF, INF], 0);
        assert_eq!(
            complements(prob, 0, 0),
            POUNCE_COMPLEMENTS_ERR_VAR_BOUNDS,
            "x in [{xl}, {xu}]"
        );
        assert_eq!(declared(prob), vec![]);
        // Row 0 was not written, nor x recorded: y1 takes row 0.
        assert_eq!(complements(prob, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
        free(prob);
    }
}

/// Order of the checks: range, then row free, then variable, then row
/// already used, then the variable's bounds.
#[test]
fn refusals_are_reported_in_a_fixed_order() {
    // error-path: AddPounceComplementsConstraint
    // A non-free row and a paired variable: ROW_NOT_FREE.
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &[0.0, 0.0], &[INF, INF], 0);
    assert_eq!(
        unsafe { AddPounceComplementarityPair(prob, 0, 1) },
        POUNCE_CC_OK
    );
    assert_eq!(complements(prob, 1, 1), POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE);
    // Out of range beats not free.
    assert_eq!(complements(prob, 1, 7), POUNCE_COMPLEMENTS_ERR_OUT_OF_RANGE);
    free(prob);
    // A used variable on a used row: VAR_ALREADY_PAIRED.
    let prob = two_row_handle(0);
    assert_eq!(
        complements(prob, 0, 0),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    free(prob);
    // A used row with crossed variable bounds: ROW_ALREADY_PAIRED.
    let x_l = [0.0, 3.0, 0.0];
    let x_u = [2.0, 1.0, INF];
    let prob = qpec::create(&x_l, &x_u, &[-INF, -INF], &[INF, INF], 0);
    assert_eq!(complements(prob, 0, 0), POUNCE_COMPLEMENTS_PAIR_MIXED);
    assert_eq!(
        complements(prob, 0, 1),
        POUNCE_COMPLEMENTS_ERR_ROW_ALREADY_PAIRED
    );
    assert_eq!(complements(prob, 1, 1), POUNCE_COMPLEMENTS_ERR_VAR_BOUNDS);
    free(prob);
}

// ---------------------------------------------------------------------------
// The same row or variable in two calls (the conservative choice: refused,
// whether or not the first call declared a pair)
// ---------------------------------------------------------------------------

#[test]
fn a_variable_used_by_a_no_pair_call_is_refused_in_a_second_call() {
    // error-path: AddPounceComplementsConstraint
    // x free (ROW_FIXED_NO_PAIR on row 0), then x again on free row 1.
    let x_l = [-INF, 0.0, 0.0];
    let x_u = [INF, INF, INF];
    let prob = qpec::create(&x_l, &x_u, &[-INF, -INF], &[INF, INF], 0);
    assert_eq!(
        complements(prob, 0, 0),
        POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR
    );
    assert_eq!(
        complements(prob, 1, 0),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    // Row 1 untouched.
    assert_eq!(complements(prob, 1, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    free(prob);
    // x fixed (VAR_FIXED_NO_PAIR on row 0), then x again on row 1.
    let x_l = [1.0, 0.0, 0.0];
    let x_u = [1.0, INF, INF];
    let prob = qpec::create(&x_l, &x_u, &[-INF, -INF], &[INF, INF], 0);
    assert_eq!(
        complements(prob, 0, 0),
        POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR
    );
    assert_eq!(
        complements(prob, 1, 0),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    assert_eq!(declared(prob), vec![]);
    free(prob);
}

#[test]
fn clearing_the_pairs_does_not_undo_an_edit_or_free_a_row() {
    // error-path: AddPounceComplementsConstraint
    // Lower row: the pair goes, the F >= 0 edit and the record stay.
    let prob = qpec::create(&qpec::X_L, &qpec::X_U, &[-INF, -INF], &[INF, INF], 0);
    assert_eq!(complements(prob, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(unsafe { ClearPounceComplementarityPairs(prob) }, 1);
    assert_eq!(declared(prob), vec![]);
    assert_eq!(complements(prob, 0, 2), POUNCE_COMPLEMENTS_ERR_ROW_NOT_FREE);
    assert_eq!(
        complements(prob, 1, 1),
        POUNCE_COMPLEMENTS_ERR_VAR_ALREADY_PAIRED
    );
    // The edited row still takes the native pair it was lowered to.
    assert_eq!(
        unsafe { AddPounceComplementarityPair(prob, 0, 1) },
        POUNCE_CC_OK
    );
    free(prob);
}
