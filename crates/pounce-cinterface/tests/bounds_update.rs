//! gh#1147 Stage 2 — `SetPounceBounds` / `GetPounceBounds` (plan §6.2).
//!
//! No solve here (that is `bounds_update_solve.rs`): these pin the contract.
//! Every refusal code has a test, each placing the refused entry *last* in
//! a batch whose earlier entries are valid, and each followed by a check
//! that the handle — both bound views and the declared pairs — is exactly
//! as it was. Every branch of the pair rule (D3) and of the `Complements`
//! rules (D2) has an accepted and a refused fixture: the second fixture is
//! the test (CLAUDE.md, "A test is only evidence about the branch it
//! reaches").
//!
//! Model: `QpecSmall` (`common/qpec.rs`), 3 variables, 2 rows.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/qpec.rs"]
mod qpec;

use pounce_cinterface::*;
use qpec::{G_L, G_U, INF, M, N, X_L, X_U, create, create_qpec};
use std::ptr;

type Entry = (Index, Number, Number);
type Bounds = (Number, Number);

/// `SetPounceBounds` with sparse sides; returns `(code, bad_index)`.
fn set(p: IpoptProblem, xs: &[Entry], gs: &[Entry]) -> (i32, Index) {
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
            gs.len() as Index,
            gi.as_ptr(),
            gl.as_ptr(),
            gu.as_ptr(),
            &mut bad,
        )
    };
    (code, bad)
}

#[derive(Debug, PartialEq)]
struct Dense {
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
}

/// Every bound, dense, in `view`.
fn get(p: IpoptProblem, view: i32) -> Dense {
    let mut d = Dense {
        x_l: vec![-7.0; N],
        x_u: vec![-7.0; N],
        g_l: vec![-7.0; M],
        g_u: vec![-7.0; M],
    };
    let mut bad = -42;
    let code = unsafe {
        GetPounceBounds(
            p,
            N as Index,
            ptr::null(),
            d.x_l.as_mut_ptr(),
            d.x_u.as_mut_ptr(),
            M as Index,
            ptr::null(),
            d.g_l.as_mut_ptr(),
            d.g_u.as_mut_ptr(),
            view,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_OK, -1));
    d
}

/// Both views and the declared pairs: what a refused call must not move.
fn snapshot(p: IpoptProblem) -> (Dense, Dense, Vec<(Index, Index)>) {
    let k = unsafe { GetPounceComplementarityPairCount(p) } as usize;
    let mut rows = vec![0; k];
    let mut vars = vec![0; k];
    assert_eq!(
        unsafe { GetPounceComplementarityPairs(p, rows.as_mut_ptr(), vars.as_mut_ptr()) },
        1
    );
    (
        get(p, POUNCE_BOUNDS_VIEW_USER),
        get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE),
        rows.into_iter().zip(vars).collect(),
    )
}

/// Assert `set` refuses with `code` at `bad` and changes nothing.
fn refused(p: IpoptProblem, xs: &[Entry], gs: &[Entry], code: i32, bad: Index) {
    let before = snapshot(p);
    assert_eq!(set(p, xs, gs), (code, bad), "xs={xs:?} gs={gs:?}");
    assert_eq!(snapshot(p), before, "a refused update changed the handle");
}

fn free(p: IpoptProblem) {
    unsafe { FreeIpoptProblem(p) }
}

fn with_pairs(p: IpoptProblem, pairs: &[(Index, Index)]) -> IpoptProblem {
    for &(r, v) in pairs {
        assert_eq!(
            unsafe { AddPounceComplementarityPair(p, r, v) },
            POUNCE_CC_OK
        );
    }
    p
}

// --- plain updates --------------------------------------------------------

#[test]
fn sparse_update_writes_only_the_named_entries() {
    let p = create_qpec(0);
    assert_eq!(
        set(p, &[(2, -1.0, 4.0)], &[(1, -3.0, 3.0)]),
        (POUNCE_BOUNDS_OK, -1)
    );
    let d = get(p, POUNCE_BOUNDS_VIEW_USER);
    assert_eq!(d.x_l, vec![X_L[0], X_L[1], -1.0]);
    assert_eq!(d.x_u, vec![X_U[0], X_U[1], 4.0]);
    assert_eq!(d.g_l, vec![G_L[0], -3.0]);
    assert_eq!(d.g_u, vec![G_U[0], 3.0]);
    // No Complements edits: the views agree.
    assert_eq!(d, get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE));
    free(p);
}

#[test]
fn null_index_array_with_full_count_is_dense() {
    let p = create_qpec(0);
    let (xl, xu) = ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let code = unsafe {
        SetPounceBounds(
            p,
            N as Index,
            ptr::null(),
            xl.as_ptr(),
            xu.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, POUNCE_BOUNDS_OK);
    let d = get(p, POUNCE_BOUNDS_VIEW_USER);
    assert_eq!((d.x_l, d.x_u), (xl.to_vec(), xu.to_vec()));
    assert_eq!((d.g_l, d.g_u), (G_L.to_vec(), G_U.to_vec()));
    free(p);
}

#[test]
fn null_value_array_leaves_that_side_unchanged() {
    let p = create_qpec(0);
    let idx = [0];
    let hi = [7.0];
    let code = unsafe {
        SetPounceBounds(
            p,
            1,
            idx.as_ptr(),
            ptr::null(),
            hi.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null_mut(),
        )
    };
    assert_eq!(code, POUNCE_BOUNDS_OK);
    let d = get(p, POUNCE_BOUNDS_VIEW_USER);
    assert_eq!((d.x_l[0], d.x_u[0]), (X_L[0], 7.0));
    free(p);
}

#[test]
fn fortran_style_indices_are_one_based() {
    let p = create_qpec(1);
    assert_eq!(set(p, &[(3, -1.0, 1.0)], &[]), (POUNCE_BOUNDS_OK, -1));
    assert_eq!(get(p, POUNCE_BOUNDS_VIEW_USER).x_l[2], -1.0);
    refused(p, &[(0, 0.0, 1.0)], &[], POUNCE_BOUNDS_ERR_OUT_OF_RANGE, 0);
    free(p);
}

#[test]
fn crossed_bounds_are_left_to_the_solve() {
    // D4: like CreateIpoptProblem, the setter does not refuse l > u on a
    // variable in no pair; the solve reports it.
    let p = create_qpec(0);
    assert_eq!(set(p, &[(2, 3.0, 1.0)], &[]), (POUNCE_BOUNDS_OK, -1));
    free(p);
}

// --- argument refusals ------------------------------------------------------

#[test]
fn null_handle_is_refused() {
    // error-path: SetPounceBounds
    // error-path: GetPounceBounds
    let mut bad = -42;
    let code = unsafe {
        SetPounceBounds(
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_NULL, -1));
    let mut bad = -42;
    let code = unsafe {
        GetPounceBounds(
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            POUNCE_BOUNDS_VIEW_USER,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_NULL, -1));
}

#[test]
fn bad_counts_are_refused() {
    let p = create_qpec(0);
    let before = snapshot(p);
    let lo = [0.0; N];
    for (nx, ng) in [(-1, 0), (0, -1), (1, 0), (0, 1), (N as Index + 1, 0)] {
        // NULL index arrays: only 0 or the full count is meaningful.
        let code = unsafe {
            SetPounceBounds(
                p,
                nx,
                ptr::null(),
                lo.as_ptr(),
                lo.as_ptr(),
                ng,
                ptr::null(),
                lo.as_ptr(),
                lo.as_ptr(),
                ptr::null_mut(),
            )
        };
        assert_eq!(code, POUNCE_BOUNDS_ERR_COUNT, "nx={nx} ng={ng}");
    }
    assert_eq!(snapshot(p), before);
    free(p);
}

/// A count above n (or m) must repeat an index, so it is refused before any
/// entry is read or any space reserved. Before the review of PR #1157 the
/// setter reserved `count` entries first: a garbage count such as this one
/// sized the allocation (an abort, past `ffi_guard`) and then read the
/// one-entry arrays out of bounds.
#[test]
fn a_count_above_the_side_is_refused_before_reading() {
    let p = create_qpec(0);
    let before = snapshot(p);
    let idx: [Index; 1] = [0];
    let val = [0.0; 1];
    for (nx, ng) in [(Index::MAX, 0), (0, Index::MAX), (N as Index + 1, 0)] {
        let mut bad = -7;
        let code = unsafe {
            SetPounceBounds(
                p,
                nx,
                idx.as_ptr(),
                val.as_ptr(),
                val.as_ptr(),
                ng,
                idx.as_ptr(),
                val.as_ptr(),
                val.as_ptr(),
                &mut bad,
            )
        };
        assert_eq!(
            (code, bad),
            (POUNCE_BOUNDS_ERR_COUNT, -1),
            "nx={nx} ng={ng}"
        );
    }
    assert_eq!(snapshot(p), before);
    free(p);
}

#[test]
fn out_of_range_is_refused_at_its_batch_position() {
    let p = create_qpec(0);
    refused(
        p,
        &[(0, 0.0, 1.0), (3, 0.0, 1.0)],
        &[],
        POUNCE_BOUNDS_ERR_OUT_OF_RANGE,
        1,
    );
    // Rows are numbered after the variables in the batch.
    refused(
        p,
        &[(0, 0.0, 1.0)],
        &[(1, 0.0, 1.0), (-1, 0.0, 1.0)],
        POUNCE_BOUNDS_ERR_OUT_OF_RANGE,
        2,
    );
    free(p);
}

#[test]
fn duplicate_index_is_refused_at_its_second_occurrence() {
    let p = create_qpec(0);
    refused(
        p,
        &[(2, 0.0, 1.0), (0, 0.0, 1.0), (2, 0.0, 3.0)],
        &[],
        POUNCE_BOUNDS_ERR_DUPLICATE,
        2,
    );
    // The same index on the two sides is two different things.
    assert_eq!(
        set(p, &[(1, 0.0, 9.0)], &[(1, -1.0, 1.0)]),
        (POUNCE_BOUNDS_OK, -1)
    );
    free(p);
}

#[test]
fn nan_is_refused() {
    let p = create_qpec(0);
    refused(
        p,
        &[(2, 0.0, 1.0)],
        &[(0, 0.0, Number::NAN)],
        POUNCE_BOUNDS_ERR_NAN,
        1,
    );
    free(p);
}

#[test]
fn getter_refusals_write_nothing() {
    let p = create_qpec(0);
    let mut out = [-7.0; 2];
    let idx = [0, 5];
    let mut bad = -42;
    let code = unsafe {
        GetPounceBounds(
            p,
            2,
            idx.as_ptr(),
            out.as_mut_ptr(),
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            POUNCE_BOUNDS_VIEW_USER,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_OUT_OF_RANGE, 1));
    assert_eq!(out, [-7.0; 2]);
    let code = unsafe {
        GetPounceBounds(
            p,
            1,
            idx.as_ptr(),
            out.as_mut_ptr(),
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            2,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_VIEW, -1));
    let code = unsafe {
        GetPounceBounds(
            p,
            1,
            ptr::null(),
            out.as_mut_ptr(),
            ptr::null_mut(),
            0,
            ptr::null(),
            ptr::null_mut(),
            ptr::null_mut(),
            POUNCE_BOUNDS_VIEW_USER,
            &mut bad,
        )
    };
    assert_eq!((code, bad), (POUNCE_BOUNDS_ERR_COUNT, -1));
    assert_eq!(out, [-7.0; 2]);
    free(p);
}

// --- D3: native pairs keep their paired sides ---------------------------------
//
// QpecSmall's pairs: (row 0, var 1) and (row 1, var 2), each one-sided,
// variable lower bound with row lower bound.

#[test]
fn paired_bound_values_may_move() {
    let p = with_pairs(create_qpec(0), &qpec::PAIRS);
    assert_eq!(
        set(p, &[(1, 0.5, INF)], &[(0, 0.25, INF)]),
        (POUNCE_BOUNDS_OK, -1)
    );
    free(p);
}

#[test]
fn gaining_the_unpaired_bound_keeps_the_identity() {
    // y1 >= 0 ⟂ g0 >= 0, then y1 <= 5: OneSided -> OneSidedPlusBound, the
    // row's lower side still pairs the variable's lower bound. Accepted.
    let p = with_pairs(create_qpec(0), &qpec::PAIRS);
    assert_eq!(set(p, &[(1, 0.0, 5.0)], &[]), (POUNCE_BOUNDS_OK, -1));
    // And losing it again.
    assert_eq!(set(p, &[(1, 0.0, INF)], &[]), (POUNCE_BOUNDS_OK, -1));
    free(p);
}

#[test]
fn gaining_an_upper_bound_that_flips_the_orientation_is_refused() {
    // y1 >= 0 ⟂ g0 <= 0 is a legal OneSided pair (the sides are independent).
    // Give y1 a finite upper bound and the bounds classify as
    // OneSidedPlusBound pairing g0 <= 0 with y1 <= u: each state validates,
    // but the complementarity condition is a different one. Refused; the
    // sibling above, where the row's side matches, is accepted.
    let g_l = [-INF, G_L[1]];
    let g_u = [0.0, G_U[1]];
    let p = with_pairs(create(&X_L, &X_U, &g_l, &g_u, 0), &[(0, 1)]);
    refused(
        p,
        &[(2, 0.0, 1.0), (1, 0.0, 5.0)],
        &[],
        POUNCE_BOUNDS_ERR_PAIR_CHANGED,
        1,
    );
    free(p);
}

#[test]
fn moving_the_paired_side_to_the_other_bound_is_refused() {
    // y1 >= 0 becomes y1 <= 3: still OneSided, other orientation.
    let p = with_pairs(create_qpec(0), &qpec::PAIRS);
    refused(p, &[(1, -INF, 3.0)], &[], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    free(p);
}

#[test]
fn losing_the_paired_side_is_refused() {
    let p = with_pairs(create_qpec(0), &qpec::PAIRS);
    refused(p, &[(1, -INF, INF)], &[], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    refused(p, &[], &[(0, -INF, INF)], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    free(p);
}

#[test]
fn a_one_sided_row_gaining_a_second_side_is_refused() {
    let p = with_pairs(create_qpec(0), &qpec::PAIRS);
    refused(
        p,
        &[(0, 0.0, 1.0)],
        &[(1, 0.0, 4.0)],
        POUNCE_BOUNDS_ERR_PAIR_CHANGED,
        1,
    );
    free(p);
}

/// x (var 0, `[0, 2]`) in a mixed pair with row 1 made free.
fn mixed() -> IpoptProblem {
    let g_l = [G_L[0], -INF];
    let g_u = [G_U[0], INF];
    with_pairs(create(&X_L, &X_U, &g_l, &g_u, 0), &[(1, 0)])
}

#[test]
fn a_mixed_box_may_move() {
    let p = mixed();
    assert_eq!(set(p, &[(0, -1.0, 3.0)], &[]), (POUNCE_BOUNDS_OK, -1));
    free(p);
}

#[test]
fn a_mixed_variable_collapsing_or_opening_is_refused() {
    let p = mixed();
    refused(p, &[(0, 1.0, 1.0)], &[], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    refused(p, &[(0, 0.0, INF)], &[], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    free(p);
}

#[test]
fn a_mixed_row_gaining_a_side_is_refused() {
    // A free row with a boxed variable gaining g >= 0 classifies as
    // OneSidedPlusBound: a different pair. Refused.
    let p = mixed();
    refused(p, &[], &[(1, 0.0, INF)], POUNCE_BOUNDS_ERR_PAIR_CHANGED, 0);
    free(p);
}

// --- D2: MOI Complements ------------------------------------------------------

/// QpecSmall with free rows and var 1 given `(xl, xu)`, then
/// `[F_0, x_1] ∈ Complements`; returns the handle and the helper's code.
fn complements(xl: Number, xu: Number) -> (IpoptProblem, i32) {
    let x_l = [X_L[0], xl, X_L[2]];
    let x_u = [X_U[0], xu, X_U[2]];
    let p = create(&x_l, &x_u, &[-INF; M], &[INF; M], 0);
    let code = unsafe { AddPounceComplementsConstraint(p, 0, 1) };
    assert!(code >= 0);
    (p, code)
}

#[test]
fn every_table_row_accepts_a_same_class_update_and_refuses_a_class_change() {
    // (var 1 bounds, helper code, same-class update, class-changing update)
    let cases: [(Bounds, i32, Bounds, Bounds); 5] = [
        (
            (0.0, 2.0),
            POUNCE_COMPLEMENTS_PAIR_MIXED,
            (-1.0, 3.0),
            (1.0, 1.0),
        ),
        (
            (0.0, INF),
            POUNCE_COMPLEMENTS_PAIR_LOWER,
            (1.0, INF),
            (0.0, 5.0),
        ),
        (
            (-INF, 2.0),
            POUNCE_COMPLEMENTS_PAIR_UPPER,
            (-INF, 3.0),
            (-INF, INF),
        ),
        (
            (-INF, INF),
            POUNCE_COMPLEMENTS_ROW_FIXED_NO_PAIR,
            (-1e20, 1e20),
            (0.0, INF),
        ),
        (
            (1.0, 1.0),
            POUNCE_COMPLEMENTS_VAR_FIXED_NO_PAIR,
            (2.0, 2.0),
            (1.0, 2.0),
        ),
    ];
    for (start, code, same, change) in cases {
        let (p, got) = complements(start.0, start.1);
        assert_eq!(got, code, "start {start:?}");
        let effective = get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE);
        refused(
            p,
            &[(0, 0.0, 1.0), (1, change.0, change.1)],
            &[],
            POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS,
            1,
        );
        assert_eq!(
            set(p, &[(1, same.0, same.1)], &[]),
            (POUNCE_BOUNDS_OK, -1),
            "start {start:?} -> {same:?}"
        );
        // The row edit is the table row's, unchanged by a same-class move.
        let after = get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE);
        assert_eq!((after.g_l, after.g_u), (effective.g_l, effective.g_u));
        free(p);
    }
}

#[test]
fn a_complements_row_takes_free_user_bounds_only() {
    let (p, code) = complements(0.0, INF);
    assert_eq!(code, POUNCE_COMPLEMENTS_PAIR_LOWER);
    let user = get(p, POUNCE_BOUNDS_VIEW_USER);
    let eff = get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE);
    assert_eq!((user.g_l[0], user.g_u[0]), (-INF, INF));
    assert_eq!((eff.g_l[0], eff.g_u[0]), (0.0, INF));
    // The edited value is the effective layer's, not an input.
    refused(
        p,
        &[],
        &[(1, -1.0, 1.0), (0, 0.0, INF)],
        POUNCE_BOUNDS_ERR_COMPLEMENTS_ROW,
        1,
    );
    // Another spelling of "free" is accepted, round-trips in the user view,
    // and the edit is re-derived from it.
    assert_eq!(set(p, &[], &[(0, -1e20, 1e30)]), (POUNCE_BOUNDS_OK, -1));
    let user = get(p, POUNCE_BOUNDS_VIEW_USER);
    let eff = get(p, POUNCE_BOUNDS_VIEW_EFFECTIVE);
    assert_eq!((user.g_l[0], user.g_u[0]), (-1e20, 1e30));
    assert_eq!((eff.g_l[0], eff.g_u[0]), (0.0, 1e30));
    free(p);
}

#[test]
fn the_class_rule_outlives_a_clear() {
    // The edits and the ledger survive ClearPounceComplementarityPairs, so
    // the rule does too.
    let (p, _) = complements(0.0, INF);
    assert_eq!(unsafe { ClearPounceComplementarityPairs(p) }, 1);
    refused(
        p,
        &[(1, 0.0, 5.0)],
        &[],
        POUNCE_BOUNDS_ERR_COMPLEMENTS_CLASS,
        0,
    );
    free(p);
}

// --- D4: a staged working set stays true ------------------------------------

#[test]
fn a_staged_working_set_the_update_would_contradict_refuses_it() {
    let p = create_qpec(0);
    // x at its upper bound 2, row 0 at its lower bound 0.
    let bounds = [2, 0, 0]; // POUNCE_WS_AT_UPPER, inactive, inactive
    let cons = [1, 0]; // POUNCE_WS_AT_LOWER, inactive
    assert_eq!(
        unsafe { IpoptSetWarmStartWorkingSet(p, bounds.as_ptr(), cons.as_ptr()) },
        1
    );
    refused(p, &[(0, 0.0, INF)], &[], POUNCE_BOUNDS_ERR_WORKING_SET, 0);
    refused(
        p,
        &[(0, 0.0, 3.0)],
        &[(0, -INF, INF)],
        POUNCE_BOUNDS_ERR_WORKING_SET,
        1,
    );
    // Moving the bound it sits on keeps the claim true.
    assert_eq!(
        set(p, &[(0, 0.0, 3.0)], &[(0, 1.0, INF)]),
        (POUNCE_BOUNDS_OK, -1)
    );
    // Once cleared, nothing is checked.
    assert_eq!(unsafe { IpoptClearWarmStartWorkingSet(p) }, 1);
    assert_eq!(set(p, &[(0, 0.0, INF)], &[]), (POUNCE_BOUNDS_OK, -1));
    free(p);
}
