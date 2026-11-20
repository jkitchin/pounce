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

/// A 4-variable QP whose `H` stores no diagonal for columns 2 and 4 (one of
/// them coupled off-diagonally, one structurally zero), with two rows.
fn qp_with_zero_h_columns<R>(f: impl FnOnce(&crate::problem::QpProblem) -> R) -> R {
    use crate::problem::{HessianInertia, QpProblem};
    use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(4, vec![1, 3, 2], vec![1, 3, 1]));
    h.set_values(&[2.0, 1.0, 0.5]);
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(
        2,
        4,
        vec![1, 1, 2, 2],
        vec![1, 2, 3, 4],
    ));
    a.set_values(&[1.0, 1.0, 1.0, -1.0]);
    let g = [0.0; 4];
    let (bl, bu) = ([0.0; 2], [1.0; 2]);
    let (xl, xu) = ([-1.0; 4], [1.0; 4]);
    f(&QpProblem {
        n: 4,
        m: 2,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Indefinite,
    })
}

/// gh#1206 P2: every KKT assembly stores a diagonal for every `H` row, so
/// the §4.5 retry's `add_h_diagonal_shift` moves values only and the δ > 0
/// matrix has the δ = 0 matrix's pattern. Before, the shift appended the
/// missing diagonals and the backend re-analysed the new pattern.
#[test]
fn an_inertia_shift_keeps_every_assembled_kkt_pattern() {
    use crate::kkt::{
        assemble_active_set_kkt, assemble_box_with_active, assemble_equality_plus_bounds,
        assemble_soft_active_set_kkt,
    };
    use crate::schur::SchurState;
    qp_with_zero_h_columns(|qp| {
        let kkts = [
            ("equality_only", KktTriplet::assemble_equality_only(qp)),
            ("box_with_active", assemble_box_with_active(qp, &[0, 3])),
            ("active_set", assemble_active_set_kkt(qp, &[1], &[0])),
            (
                "equality_plus_bounds",
                assemble_equality_plus_bounds(qp, &[2]),
            ),
            (
                "soft_active_set",
                assemble_soft_active_set_kkt(qp, &[0, 1], &[false, true], &[2], &[true]),
            ),
            (
                "k_max",
                SchurState::new(4, 2)
                    .build_k_max_triplet(qp, &[true, false, false, true, false, false]),
            ),
        ];
        for (what, kkt) in kkts {
            let mut shifted = kkt.clone();
            shifted.add_h_diagonal_shift(qp.n, 1e-4);
            assert_eq!(shifted.irn, kkt.irn, "{what}: rows moved");
            assert_eq!(shifted.jcn, kkt.jcn, "{what}: columns moved");
            // The zeros were inert: every H diagonal moved by exactly δ.
            for i in 1..=qp.n as pounce_common::Index {
                let d = |k: &KktTriplet| -> f64 {
                    (0..k.irn.len())
                        .filter(|&p| k.irn[p] == i && k.jcn[p] == i)
                        .map(|p| k.vals[p])
                        .sum()
                };
                let moved = d(&shifted) - d(&kkt);
                assert!((moved - 1e-4).abs() < 1e-15, "{what}: diagonal {i} moved {moved}");
            }
        }
    });
}

/// gh#1206 P2, end to end: a KKT that fails at δ = 0 and factors at δ > 0
/// reuses the δ = 0 attempt's symbolic analysis: FERAL's summary records one
/// pattern change (the first factorization) and a reuse for every rung after
/// it. Before, the first shift appended the missing diagonals, a second
/// pattern change.
#[test]
fn the_shifted_retry_reuses_the_symbolic_analysis() {
    use crate::kkt::assemble_box_with_active;
    use crate::options::QpOptions;
    use crate::solver::ParametricActiveSetSolver;
    use pounce_linsol::summary::LinearSolverSummary;
    use std::sync::{Arc, Mutex};
    qp_with_zero_h_columns(|qp| {
        let sink = Arc::new(Mutex::new(LinearSolverSummary::default()));
        let backend =
            pounce_feral::FeralSolverInterface::new().with_summary_sink(Arc::clone(&sink));
        let mut s = ParametricActiveSetSolver::new(Box::new(backend));
        // No bound active: column 4's zero H column is uncovered, so the
        // unshifted KKT is singular.
        let kkt = assemble_box_with_active(qp, &[]);
        let mut rhs = vec![1.0; kkt.dim];
        let delta = s
            .factorize_with_inertia_control(kkt, &mut rhs, 0, qp.n, &QpOptions::default())
            .unwrap();
        assert!(delta > 0.0, "the unshifted KKT should not have factored");
        let sum = sink.lock().unwrap().clone();
        // Every rung of the ladder is recorded; only the first analysed.
        assert!(sum.n_factors >= 2, "{sum:?}");
        assert_eq!(sum.n_pattern_changes, 1, "{sum:?}");
        assert_eq!(sum.n_pattern_reuse, sum.n_factors - 1, "{sum:?}");
    });
}
