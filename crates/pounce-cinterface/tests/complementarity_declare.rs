//! gh#1147 — declaring complementarity pairs through the C API.
//!
//! `AddPounceComplementarityPair`, `SetPounceComplementarityPairs`,
//! `ClearPounceComplementarityPairs`, `GetPounceComplementarityPairCount`
//! and `GetPounceComplementarityPairs` on `QpecSmall` (`common/qpec.rs`).
//! No solve here: these pin the declaration contract — indices in the
//! problem's index style, validation at declaration by the classifier the
//! lowering runs, one `POUNCE_CC_*` code per refusal, and a refused call
//! leaving the declared set exactly as it was.
//!
//! Every refusal branch has its own test (CLAUDE.md, "A test is only
//! evidence about the branch it reaches"), and each runs on a handle that
//! already holds a pair, so "unchanged" is a check of something rather
//! than of an empty set.

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use qpec::{G_L, G_U, INF, M, N, X_L, X_U, create, create_qpec};
use std::ptr;

/// The declared pairs, read back through the count and the getter.
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

fn add(prob: IpoptProblem, row: Index, var: Index) -> i32 {
    unsafe { AddPounceComplementarityPair(prob, row, var) }
}

/// `SetPounceComplementarityPairs` over `pairs`; returns `(code, bad_index)`.
fn set(prob: IpoptProblem, pairs: &[(Index, Index)]) -> (i32, Index) {
    let rows: Vec<Index> = pairs.iter().map(|p| p.0).collect();
    let vars: Vec<Index> = pairs.iter().map(|p| p.1).collect();
    let mut bad = -42;
    let code = unsafe {
        SetPounceComplementarityPairs(
            prob,
            pairs.len() as Index,
            rows.as_ptr(),
            vars.as_ptr(),
            &mut bad,
        )
    };
    (code, bad)
}

fn free(prob: IpoptProblem) {
    unsafe { FreeIpoptProblem(prob) }
}

// --- success paths -------------------------------------------------------

#[test]
fn a_fresh_handle_declares_no_pairs() {
    let p = create_qpec(0);
    unsafe {
        assert_eq!(GetPounceComplementarityPairCount(p), 0);
        // No pairs: TRUE, and the arrays may be NULL.
        assert_eq!(
            GetPounceComplementarityPairs(p, ptr::null_mut(), ptr::null_mut()),
            1
        );
    }
    free(p);
}

#[test]
fn add_declares_one_pair_and_reads_back_in_c_style() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    assert_eq!(declared(p), vec![(0, 1)]);
    assert_eq!(add(p, 1, 2), POUNCE_CC_OK);
    assert_eq!(declared(p), vec![(0, 1), (1, 2)]);
    free(p);
}

#[test]
fn fortran_style_indices_are_read_and_returned_one_based() {
    let p = create_qpec(1);
    // (row 1, var 2) in Fortran style is (row 0, var 1) 0-based.
    assert_eq!(add(p, 1, 2), POUNCE_CC_OK);
    assert_eq!(set(p, &[(1, 2), (2, 3)]), (POUNCE_CC_OK, -1));
    assert_eq!(declared(p), vec![(1, 2), (2, 3)]);
    free(p);
}

#[test]
fn fortran_style_refuses_zero_and_accepts_the_last_index() {
    let p = create_qpec(1);
    assert_eq!(add(p, 2, 3), POUNCE_CC_OK); // row m, var n: the last ones
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, 0, 2), POUNCE_CC_ERR_OUT_OF_RANGE); // row 0 is below range
    assert_eq!(add(p, 1, 0), POUNCE_CC_ERR_OUT_OF_RANGE); // var 0 is below range
    assert_eq!(add(p, M as Index + 1, 2), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, 1, N as Index + 1), POUNCE_CC_ERR_OUT_OF_RANGE);
    // `p - 1` on i32::MIN must not overflow.
    assert_eq!(add(p, Index::MIN, 2), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, 1, Index::MIN), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(declared(p), vec![(2, 3)]);
    free(p);
}

#[test]
fn set_replaces_the_declared_set() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // Replaces, so redeclaring (0, 1) is not a duplicate of the old set.
    assert_eq!(set(p, &[(1, 2), (0, 1)]), (POUNCE_CC_OK, -1));
    assert_eq!(declared(p), vec![(1, 2), (0, 1)]);
    assert_eq!(set(p, &[(1, 2)]), (POUNCE_CC_OK, -1));
    assert_eq!(declared(p), vec![(1, 2)]);
    free(p);
}

#[test]
fn add_after_set_appends_and_is_checked_against_the_set() {
    let p = create_qpec(0);
    assert_eq!(set(p, &[(1, 2)]), (POUNCE_CC_OK, -1));
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    assert_eq!(declared(p), vec![(1, 2), (0, 1)]);
    free(p);
}

#[test]
fn set_with_zero_pairs_clears_and_accepts_null_arrays() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    let mut bad = -42;
    let code = unsafe { SetPounceComplementarityPairs(p, 0, ptr::null(), ptr::null(), &mut bad) };
    assert_eq!((code, bad), (POUNCE_CC_OK, -1));
    assert_eq!(declared(p), vec![]);
    free(p);
}

#[test]
fn set_accepts_a_null_bad_index() {
    let p = create_qpec(0);
    let (rows, vars) = ([0], [1]);
    let code = unsafe {
        SetPounceComplementarityPairs(p, 1, rows.as_ptr(), vars.as_ptr(), ptr::null_mut())
    };
    assert_eq!(code, POUNCE_CC_OK);
    // ... and on a refusal too: nothing to write, still a code.
    let (rows, vars) = ([0], [9]);
    let code = unsafe {
        SetPounceComplementarityPairs(p, 1, rows.as_ptr(), vars.as_ptr(), ptr::null_mut())
    };
    assert_eq!(code, POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn clear_empties_the_set() {
    let p = create_qpec(0);
    assert_eq!(set(p, &[(0, 1), (1, 2)]), (POUNCE_CC_OK, -1));
    assert_eq!(unsafe { ClearPounceComplementarityPairs(p) }, 1);
    assert_eq!(declared(p), vec![]);
    // Clearing frees the variables and rows for a new declaration.
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    free(p);
}

/// The three accepted shapes, each on its own handle: the classifier is
/// the one the lowering runs, so each must be accepted here too.
#[test]
fn every_lowered_shape_is_accepted() {
    // one-sided: QpecSmall itself (var 1 lower-bounded, row 0 lower side).
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    free(p);
    // one-sided row, upper orientation, with a variable upper bound.
    let p = create(&X_L, &[2.0, INF, INF], &[0.0, -INF], &[INF, 0.5], 0);
    let p2 = create(&[0.0, -INF, -INF], &[2.0, 1.0, 1.0], &G_L, &G_U, 0);
    assert_eq!(add(p, 1, 1), POUNCE_CC_OK); // x1 >= 0 ⟂ g1 <= 0.5
    assert_eq!(add(p2, 0, 1), POUNCE_CC_OK); // x1 <= 1 ⟂ g0 >= 0
    free(p);
    free(p2);
    // one-sided plus a plain bound: var 0 is boxed [0, 2], row 0 one-sided.
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 0), POUNCE_CC_OK);
    free(p);
    // mixed: var 0 boxed [0, 2] with l < u, row 0 free.
    let p = create(&X_L, &X_U, &[-INF, 0.0], &[INF, INF], 0);
    assert_eq!(add(p, 0, 0), POUNCE_CC_OK);
    free(p);
}

// --- refusals -------------------------------------------------------------

#[test]
fn null_handle_is_refused_by_every_entry_point() {
    let null: IpoptProblem = ptr::null_mut();
    let (rows, vars) = ([0], [1]);
    let mut bad = -42;
    let mut out = [-99; 1];
    unsafe {
        // error-path: AddPounceComplementarityPair
        assert_eq!(AddPounceComplementarityPair(null, 0, 1), POUNCE_CC_ERR_NULL);
        // error-path: SetPounceComplementarityPairs
        assert_eq!(
            SetPounceComplementarityPairs(null, 1, rows.as_ptr(), vars.as_ptr(), &mut bad),
            POUNCE_CC_ERR_NULL
        );
        assert_eq!(bad, -1);
        // error-path: ClearPounceComplementarityPairs
        assert_eq!(ClearPounceComplementarityPairs(null), 0);
        // error-path: GetPounceComplementarityPairCount
        assert_eq!(GetPounceComplementarityPairCount(null), -1);
        // error-path: GetPounceComplementarityPairs
        assert_eq!(
            GetPounceComplementarityPairs(null, out.as_mut_ptr(), out.as_mut_ptr()),
            0
        );
    }
    assert_eq!(out, [-99]);
}

#[test]
fn set_refuses_a_negative_count() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    let (rows, vars) = ([1], [2]);
    let mut bad = -42;
    // error-path: SetPounceComplementarityPairs
    let code =
        unsafe { SetPounceComplementarityPairs(p, -1, rows.as_ptr(), vars.as_ptr(), &mut bad) };
    assert_eq!((code, bad), (POUNCE_CC_ERR_COUNT, -1));
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn set_refuses_null_arrays_with_a_positive_count() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    let a = [1];
    let mut bad = -42;
    // error-path: SetPounceComplementarityPairs
    for (rows, vars) in [
        (ptr::null(), a.as_ptr()),
        (a.as_ptr(), ptr::null()),
        (ptr::null(), ptr::null()),
    ] {
        let code = unsafe { SetPounceComplementarityPairs(p, 1, rows, vars, &mut bad) };
        assert_eq!((code, bad), (POUNCE_CC_ERR_NULL, -1));
        assert_eq!(declared(p), vec![(0, 1)]);
    }
    free(p);
}

#[test]
fn get_refuses_null_arrays_while_pairs_are_declared() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    let mut a = [-99; 1];
    unsafe {
        // error-path: GetPounceComplementarityPairs
        assert_eq!(
            GetPounceComplementarityPairs(p, ptr::null_mut(), a.as_mut_ptr()),
            0
        );
        assert_eq!(
            GetPounceComplementarityPairs(p, a.as_mut_ptr(), ptr::null_mut()),
            0
        );
    }
    assert_eq!(a, [-99], "a refused get must write nothing");
    free(p);
}

#[test]
fn add_refuses_a_variable_out_of_range() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, 1, N as Index), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, 1, -1), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, 1, Index::MAX), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn add_refuses_a_row_out_of_range() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, M as Index, 2), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, -1, 2), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(add(p, Index::MIN, 2), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn a_model_with_no_rows_refuses_every_pair() {
    let x_l = [0.0; 1];
    let x_u = [INF; 1];
    let p = unsafe {
        CreateIpoptProblem(
            1,
            x_l.as_ptr(),
            x_u.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
            0,
            1,
            0,
            Some(qpec::ev_f),
            None,
            Some(qpec::ev_grad_f),
            None,
            None,
        )
    };
    assert!(!p.is_null());
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, 0, 0), POUNCE_CC_ERR_OUT_OF_RANGE);
    assert_eq!(set(p, &[(0, 0)]), (POUNCE_CC_ERR_OUT_OF_RANGE, 0));
    assert_eq!(set(p, &[]), (POUNCE_CC_OK, -1));
    assert_eq!(declared(p), vec![]);
    free(p);
}

#[test]
fn add_refuses_a_variable_already_paired_by_an_earlier_add() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, 1, 1), POUNCE_CC_ERR_VAR_ALREADY);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn add_refuses_a_row_already_paired_by_an_earlier_add() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: AddPounceComplementarityPair
    assert_eq!(add(p, 0, 2), POUNCE_CC_ERR_ROW_ALREADY);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn set_refuses_a_duplicate_variable_inside_the_batch() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: SetPounceComplementarityPairs
    assert_eq!(set(p, &[(1, 2), (0, 2)]), (POUNCE_CC_ERR_VAR_ALREADY, 1));
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn set_refuses_a_duplicate_row_inside_the_batch_and_names_the_second() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: SetPounceComplementarityPairs
    assert_eq!(set(p, &[(1, 2), (1, 1)]), (POUNCE_CC_ERR_ROW_ALREADY, 1));
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn set_refuses_out_of_range_and_names_the_pair() {
    let p = create_qpec(0);
    assert_eq!(add(p, 0, 1), POUNCE_CC_OK);
    // error-path: SetPounceComplementarityPairs
    assert_eq!(set(p, &[(1, 2), (0, 3)]), (POUNCE_CC_ERR_OUT_OF_RANGE, 1));
    assert_eq!(set(p, &[(2, 1)]), (POUNCE_CC_ERR_OUT_OF_RANGE, 0));
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

/// The refusal order is the classifier's, pair by pair: a duplicate in
/// pair 1 is reported before an out-of-range index in pair 2, not after
/// a range pre-pass over the whole batch.
#[test]
fn set_reports_the_first_refused_pair_in_declaration_order() {
    let p = create_qpec(0);
    assert_eq!(
        set(p, &[(0, 1), (1, 1), (9, 9)]),
        (POUNCE_CC_ERR_VAR_ALREADY, 1)
    );
    assert_eq!(declared(p), vec![]);
    free(p);
}

#[test]
fn set_is_all_or_nothing() {
    let p = create_qpec(0);
    assert_eq!(set(p, &[(1, 2)]), (POUNCE_CC_OK, -1));
    // First pair valid, second refused: neither is stored, the old set
    // stays — not replaced by the valid prefix, not emptied.
    assert_eq!(set(p, &[(0, 1), (1, 7)]), (POUNCE_CC_ERR_OUT_OF_RANGE, 1));
    assert_eq!(declared(p), vec![(1, 2)]);
    free(p);
}

#[test]
fn fortran_style_set_reports_the_batch_position_zero_based() {
    let p = create_qpec(1);
    // error-path: SetPounceComplementarityPairs
    assert_eq!(set(p, &[(1, 2), (2, 2)]), (POUNCE_CC_ERR_VAR_ALREADY, 1));
    assert_eq!(set(p, &[(0, 2)]), (POUNCE_CC_ERR_OUT_OF_RANGE, 0));
    assert_eq!(declared(p), vec![]);
    free(p);
}

// --- shape refusals, one per branch the classifier refuses -----------------

/// Declare `(row 0, var 0)` on a handle whose var 0 has `[xl, xu]` and row
/// 0 has `[gl, gu]` (var 1 / row 1 keep QpecSmall's bounds, with `(1, 2)`
/// declared first), and return the code; asserts the set is unchanged.
fn shape(xl: Number, xu: Number, gl: Number, gu: Number) -> i32 {
    let p = create(&[xl, 0.0, 0.0], &[xu, INF, INF], &[gl, 0.0], &[gu, INF], 0);
    assert_eq!(add(p, 1, 2), POUNCE_CC_OK);
    let code = add(p, 0, 0);
    if code != POUNCE_CC_OK {
        assert_eq!(declared(p), vec![(1, 2)]);
    }
    free(p);
    code
}

#[test]
fn shape_refuses_a_boxed_row() {
    // error-path: AddPounceComplementarityPair
    assert_eq!(shape(0.0, INF, 0.0, 1.0), POUNCE_CC_ERR_SHAPE);
}

#[test]
fn shape_refuses_an_equality_row() {
    assert_eq!(shape(0.0, INF, 0.0, 0.0), POUNCE_CC_ERR_SHAPE);
}

#[test]
fn shape_refuses_both_sides_free() {
    assert_eq!(shape(-INF, INF, -INF, INF), POUNCE_CC_ERR_SHAPE);
}

#[test]
fn shape_refuses_a_free_variable_with_a_one_sided_row() {
    assert_eq!(shape(-INF, INF, 0.0, INF), POUNCE_CC_ERR_SHAPE);
}

#[test]
fn shape_refuses_a_one_sided_variable_with_a_free_row() {
    assert_eq!(shape(0.0, INF, -INF, INF), POUNCE_CC_ERR_SHAPE);
}

#[test]
fn shape_refuses_a_fixed_variable_with_a_free_row() {
    // `is_mixed` needs l < u; the MOI helper (S1-I9) maps this to "no pair".
    assert_eq!(shape(1.0, 1.0, -INF, INF), POUNCE_CC_ERR_SHAPE);
}

/// The control for the six refusals above: the same helper, a shape that
/// is lowered, is accepted — so a refusal there is the shape, not the
/// helper.
#[test]
fn shape_helper_accepts_a_lowered_shape() {
    assert_eq!(shape(0.0, 2.0, -INF, INF), POUNCE_CC_OK); // mixed
    assert_eq!(shape(0.0, INF, 0.0, INF), POUNCE_CC_OK); // one-sided
}

#[test]
fn set_refuses_a_shape_and_names_the_pair() {
    // Row 0 boxed [0, 1]: any pair on it is refused.
    let p = create(&X_L, &X_U, &[0.0, 0.0], &[1.0, INF], 0);
    assert_eq!(add(p, 1, 2), POUNCE_CC_OK);
    // error-path: SetPounceComplementarityPairs
    assert_eq!(set(p, &[(1, 2), (0, 1)]), (POUNCE_CC_ERR_SHAPE, 1));
    assert_eq!(declared(p), vec![(1, 2)]);
    free(p);
}

/// Sanity on the fixture, so a typo in the bounds cannot make every test
/// above pass for the wrong reason.
#[test]
fn qpec_bounds_are_the_python_model() {
    assert_eq!(X_L, [0.0, 0.0, 0.0]);
    assert_eq!(X_U, [2.0, INF, INF]);
    assert_eq!(G_L, [0.0, 0.0]);
    assert_eq!(G_U, [INF, INF]);
    assert_eq!(qpec::PAIRS, [(0, 1), (1, 2)]);
    let p = create_qpec(0);
    assert_eq!(set(p, &qpec::PAIRS), (POUNCE_CC_OK, -1));
    free(p);
}
