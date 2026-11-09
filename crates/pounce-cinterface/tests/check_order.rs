//! The C API's shared check order (`src/errors.rs`, `pounce.h`'s error
//! block, `docs/src/c-api.md` "Error codes"): every function checks the
//! handle, then `flags`/enum arguments, then counts, then each entry's
//! argument checks over the whole batch, then each entry's meaning over
//! the whole batch, then state beyond the batch.
//!
//! Each test builds a call with two faults at *different* stages, the later
//! stage's fault placed *first* in the batch, so a function that checks
//! entry by entry through every stage would report the wrong one. Each is
//! paired with a control that has only the later fault, so the test knows
//! that fault really is refused with the code it expects — otherwise a
//! "later" fault that the function accepts would make the test pass
//! vacuously (CLAUDE.md, "A test is only evidence about the branch it
//! reaches"). Every refusal must leave the handle as it was.
//!
//! Model: `QpecSmall` (`common/qpec.rs`), 3 variables, 2 rows.

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use qpec::{G_L, G_U, INF, N, X_L, X_U, create, create_qpec};
use std::ptr;

fn free(p: IpoptProblem) {
    unsafe { FreeIpoptProblem(p) }
}

fn declared(p: IpoptProblem) -> Vec<(Index, Index)> {
    unsafe {
        let k = GetPounceComplementarityPairCount(p);
        let mut rows = vec![-99; k as usize];
        let mut vars = vec![-99; k as usize];
        assert_eq!(
            GetPounceComplementarityPairs(p, rows.as_mut_ptr(), vars.as_mut_ptr()),
            1
        );
        rows.into_iter().zip(vars).collect()
    }
}

fn set_pairs(p: IpoptProblem, pairs: &[(Index, Index)]) -> (i32, Index) {
    let rows: Vec<Index> = pairs.iter().map(|q| q.0).collect();
    let vars: Vec<Index> = pairs.iter().map(|q| q.1).collect();
    let mut bad = -42;
    let code = unsafe {
        SetPounceComplementarityPairs(
            p,
            pairs.len() as Index,
            rows.as_ptr(),
            vars.as_ptr(),
            0,
            &mut bad,
        )
    };
    (code, bad)
}

// --- SetPounceComplementarityPairs ----------------------------------------

/// Variable 0 free against one-sided row 0 is no native shape.
fn with_free_x0() -> IpoptProblem {
    create(&[-INF, 0.0, 0.0], &[INF, INF, INF], &G_L, &G_U, 0)
}

#[test]
fn pairs_batch_reports_an_argument_fault_before_an_earlier_shape_fault() {
    let p = with_free_x0();
    assert_eq!(set_pairs(p, &[(1, 2)]), (POUNCE_OK, -1));
    // Control: pair 0 alone is a shape fault.
    assert_eq!(set_pairs(p, &[(0, 0)]), (POUNCE_ERR_SHAPE, 0));
    // Shape fault at 0, then each argument fault at 1: the argument fault wins.
    for (second, code) in [
        ((5, 1), POUNCE_ERR_OUT_OF_RANGE),
        ((1, 7), POUNCE_ERR_OUT_OF_RANGE),
        ((1, 0), POUNCE_ERR_VAR_ALREADY_PAIRED),
        ((0, 1), POUNCE_ERR_ROW_ALREADY_PAIRED),
    ] {
        assert_eq!(
            set_pairs(p, &[(0, 0), second]),
            (code, 1),
            "second pair {second:?}"
        );
        assert_eq!(declared(p), vec![(1, 2)], "second pair {second:?}");
    }
    free(p);
}

// --- AddPounceComplementsConstraint ---------------------------------------

/// Both rows free, so every row starts usable by `Complements`.
fn with_free_rows() -> IpoptProblem {
    create(&X_L, &X_U, &[-INF, -INF], &[INF, INF], 0)
}

fn complements(p: IpoptProblem, row: Index, var: Index) -> i32 {
    unsafe { AddPounceComplementsConstraint(p, row, var) }
}

#[test]
fn complements_reports_already_paired_before_row_not_free() {
    // A row an earlier call bounded is already paired, not "not free".
    let p = with_free_rows();
    assert_eq!(complements(p, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(complements(p, 0, 2), POUNCE_ERR_ROW_ALREADY_PAIRED);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);

    // A used variable against a row with its own bound: already paired wins.
    let p = create(&X_L, &X_U, &[-INF, 0.0], &[INF, INF], 0);
    // Control: row 1 is refused as not free on its own.
    assert_eq!(complements(p, 1, 2), POUNCE_ERR_ROW_NOT_FREE);
    assert_eq!(complements(p, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(complements(p, 1, 1), POUNCE_ERR_VAR_ALREADY_PAIRED);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}

#[test]
fn complements_reports_already_paired_before_var_bounds() {
    // Variable 0 crossed; row 0 already used by another call.
    let p = create(
        &[3.0, 0.0, 0.0],
        &[2.0, INF, INF],
        &[-INF, -INF],
        &[INF, INF],
        0,
    );
    // Control: variable 0 alone is a bounds fault.
    assert_eq!(complements(p, 1, 0), POUNCE_ERR_VAR_BOUNDS);
    assert_eq!(complements(p, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    assert_eq!(complements(p, 0, 0), POUNCE_ERR_ROW_ALREADY_PAIRED);
    free(p);
}

// --- SetPounceBounds --------------------------------------------------------

type Entry = (Index, Number, Number);

/// `SetPounceBounds` with sparse sides and an explicit `ng` (which may lie
/// about `gs.len()`, to reach the count check); returns `(code, bad_index)`.
fn set_bounds(p: IpoptProblem, xs: &[Entry], ng: Index, gs: &[Entry]) -> (i32, Index) {
    let col = |e: &[Entry]| -> (Vec<Index>, Vec<Number>, Vec<Number>) {
        (
            e.iter().map(|t| t.0).collect(),
            e.iter().map(|t| t.1).collect(),
            e.iter().map(|t| t.2).collect(),
        )
    };
    let (xi, xl, xu) = col(xs);
    let (gi, gl, gu) = col(gs);
    let mut bad = -42;
    let code = unsafe {
        SetPounceBounds(
            p,
            xs.len() as Index,
            xi.as_ptr(),
            xl.as_ptr(),
            xu.as_ptr(),
            ng,
            if gs.is_empty() {
                ptr::null()
            } else {
                gi.as_ptr()
            },
            if gs.is_empty() {
                ptr::null()
            } else {
                gl.as_ptr()
            },
            if gs.is_empty() {
                ptr::null()
            } else {
                gu.as_ptr()
            },
            0,
            &mut bad,
        )
    };
    (code, bad)
}

fn bounds(p: IpoptProblem) -> (Vec<Number>, Vec<Number>) {
    let mut xl = vec![0.0; N];
    let mut xu = vec![0.0; N];
    let mut gl = vec![0.0; 2];
    let mut gu = vec![0.0; 2];
    let ok = unsafe {
        GetPounceBounds(
            p,
            N as Index,
            ptr::null(),
            xl.as_mut_ptr(),
            xu.as_mut_ptr(),
            2,
            ptr::null(),
            gl.as_mut_ptr(),
            gu.as_mut_ptr(),
            POUNCE_BOUNDS_VIEW_USER,
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(ok, POUNCE_OK);
    (
        xl.into_iter().chain(gl).collect(),
        xu.into_iter().chain(gu).collect(),
    )
}

#[test]
fn bounds_reports_a_bad_row_count_before_a_variable_entry_fault() {
    let p = create_qpec(0);
    let before = bounds(p);
    // Control: the NaN entry alone is refused at its position.
    assert_eq!(
        set_bounds(p, &[(0, Number::NAN, 1.0)], 0, &[]),
        (POUNCE_ERR_NAN, 0)
    );
    // NaN at x entry 0, and ng = 3 > m with a NULL g index array: the count
    // is a stage-3 fault, the NaN a stage-4 one.
    assert_eq!(
        set_bounds(p, &[(0, Number::NAN, 1.0)], 3, &[]),
        (POUNCE_ERR_COUNT, -1)
    );
    // Out of range at x entry 0, and a NULL g index array with ng = 1
    // (neither 0 nor m).
    assert_eq!(
        set_bounds(p, &[(9, 0.0, 1.0)], 1, &[]),
        (POUNCE_ERR_COUNT, -1)
    );
    assert_eq!(bounds(p), before);
    free(p);
}

#[test]
fn bounds_reports_a_row_argument_fault_before_a_variable_semantic_fault() {
    // Variable 1 is lower-bounded only and used by `Complements`; giving it
    // an upper bound moves it to another row of the MOI table.
    let p = with_free_rows();
    assert_eq!(complements(p, 0, 1), POUNCE_COMPLEMENTS_PAIR_LOWER);
    let before = bounds(p);
    // Control: the class change alone is refused at entry 0.
    assert_eq!(
        set_bounds(p, &[(1, 0.0, 5.0)], 0, &[]),
        (POUNCE_ERR_COMPLEMENTS_CLASS, 0)
    );
    // The class change at x entry 0, and row index 9 at g entry 0 (batch
    // position nx + 0 = 1): the row's argument fault wins.
    assert_eq!(
        set_bounds(p, &[(1, 0.0, 5.0)], 1, &[(9, 0.0, 1.0)]),
        (POUNCE_ERR_OUT_OF_RANGE, 1)
    );
    // And a NaN row bound.
    assert_eq!(
        set_bounds(p, &[(1, 0.0, 5.0)], 1, &[(1, Number::NAN, 1.0)]),
        (POUNCE_ERR_NAN, 1)
    );
    assert_eq!(bounds(p), before);
    assert_eq!(declared(p), vec![(0, 1)]);
    free(p);
}
