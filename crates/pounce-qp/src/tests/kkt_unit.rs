//! Focused unit tests for `crate::kkt` helpers — pinned in §8.7
//! of the design note. Integration tests in `tests/analytical.rs`
//! exercise these through the full solver, but the unit-level
//! tests below catch internal bugs faster (e.g. an off-by-one in
//! `add_h_diagonal_shift` would surface here before triggering a
//! cryptic solver failure).

use crate::kkt::KktTriplet;

fn diag_kkt(n: usize, h_diag: &[f64]) -> KktTriplet {
    // Build a trivial `n×n` diagonal-only KKT triplet (no A
    // block). Indices 1-based.
    let mut irn = Vec::with_capacity(n);
    let mut jcn = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    for i in 0..n {
        irn.push((i + 1) as pounce_common::Index);
        jcn.push((i + 1) as pounce_common::Index);
        vals.push(h_diag[i]);
    }
    KktTriplet {
        dim: n,
        irn,
        jcn,
        vals,
    }
}

#[test]
fn add_h_diagonal_shift_zero_delta_is_no_op() {
    let mut k = diag_kkt(3, &[1.0, 2.0, 3.0]);
    let before = (k.irn.clone(), k.jcn.clone(), k.vals.clone());
    k.add_h_diagonal_shift(3, 0.0);
    assert_eq!(k.irn, before.0);
    assert_eq!(k.jcn, before.1);
    assert_eq!(k.vals, before.2);
}

#[test]
fn add_h_diagonal_shift_increments_existing_diagonals() {
    let mut k = diag_kkt(3, &[1.0, 2.0, 3.0]);
    k.add_h_diagonal_shift(3, 0.5);
    assert_eq!(k.vals, vec![1.5, 2.5, 3.5]);
    // Pattern unchanged when all diagonals present.
    assert_eq!(k.irn.len(), 3);
}

#[test]
fn add_h_diagonal_shift_appends_missing_diagonals() {
    // K has explicit (1,1) and (3,3) entries but no (2,2).
    let mut k = KktTriplet {
        dim: 3,
        irn: vec![1, 3],
        jcn: vec![1, 3],
        vals: vec![1.0, 3.0],
    };
    k.add_h_diagonal_shift(3, 0.5);
    // Existing entries incremented; missing (2,2) appended.
    assert_eq!(k.irn.len(), 3);
    // (2,2,0.5) is the appended entry.
    assert!(
        k.irn
            .iter()
            .zip(k.jcn.iter())
            .zip(k.vals.iter())
            .any(|((&r, &c), &v)| r == 2 && c == 2 && v == 0.5)
    );
}

#[test]
fn add_h_diagonal_shift_respects_n_h_rows_boundary() {
    // KKT with dim 5: H block is rows 1..=3; A block is rows
    // 4..=5. Shift on (1..=3) should leave rows 4, 5 untouched
    // even if there are existing entries at (4,4), (5,5).
    let mut k = KktTriplet {
        dim: 5,
        irn: vec![1, 2, 3, 4, 5],
        jcn: vec![1, 2, 3, 4, 5],
        vals: vec![1.0, 2.0, 3.0, 10.0, 20.0],
    };
    k.add_h_diagonal_shift(3, 0.1);
    assert_eq!(k.vals[..3], [1.1, 2.1, 3.1]);
    assert_eq!(k.vals[3], 10.0);
    assert_eq!(k.vals[4], 20.0);
}

#[test]
fn add_h_diagonal_shift_accumulates_across_calls() {
    let mut k = diag_kkt(2, &[1.0, 2.0]);
    k.add_h_diagonal_shift(2, 0.1);
    k.add_h_diagonal_shift(2, 0.2);
    // Float addition isn't bit-exact (0.1 + 0.2 + 2.0 has a ULP
    // of noise); compare within tolerance.
    assert!((k.vals[0] - 1.3).abs() < 1e-12);
    assert!((k.vals[1] - 2.3).abs() < 1e-12);
}

#[test]
fn add_h_diagonal_shift_shifts_a_duplicated_diagonal_once() {
    // gh#1206 B8: a diagonal stored as several triplet entries (they
    // are summed on assembly) must move by exactly `delta`, not by
    // `k·delta`. Row 1 is stored as three entries, row 2 as one.
    let mut k = KktTriplet {
        dim: 2,
        irn: vec![1, 1, 2, 1],
        jcn: vec![1, 1, 2, 1],
        vals: vec![1.0, 0.25, 2.0, 0.25],
    };
    k.add_h_diagonal_shift(2, 0.5);
    let assembled = |row: pounce_common::Index| -> f64 {
        k.irn
            .iter()
            .zip(k.jcn.iter())
            .zip(k.vals.iter())
            .filter(|&((&r, &c), _)| r == row && c == row)
            .map(|(_, &v)| v)
            .sum()
    };
    assert!((assembled(1) - 2.0).abs() < 1e-15);
    assert!((assembled(2) - 2.5).abs() < 1e-15);
    // No entry appended: both diagonals were present.
    assert_eq!(k.irn.len(), 4);
}
