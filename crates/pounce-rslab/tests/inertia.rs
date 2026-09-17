//! Inertia contract for the RSLAB backend.
//!
//! POUNCE's interior-point loop steers on one number from the linear solver:
//! the count of negative eigenvalues of the factored KKT. Everything here is
//! about whether RSLAB's answer to that question is the one POUNCE means, and
//! whether POUNCE can tell when it is not.
//!
//! Each test states which branch of RSLAB's pivot logic its fixture reaches —
//! a 1×1 pivot and a 2×2 Bunch-Kaufman block are different code, and a test
//! that only ever takes one of them is not evidence about the other
//! (`CLAUDE.md`, "A leg is only evidence about the branch its fixture
//! reaches"). Where a fixture is meant to force a 2×2 block, that is asserted
//! rather than assumed.

use pounce_common::types::{Index, Number};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_rslab::scaling::Equilibration;
use pounce_rslab::{InertiaInfo, RslabConfig, RslabSolverInterface};

/// Factor `A` (1-based lower-triangle triplets) and return the backend plus
/// the status of the factorization.
fn factor(
    cfg: RslabConfig,
    n: Index,
    irn: &[Index],
    jcn: &[Index],
    vals: &[Number],
) -> (RslabSolverInterface, ESymSolverStatus) {
    let mut s = RslabSolverInterface::with_config(cfg);
    assert_eq!(
        s.initialize_structure(n, irn.len() as Index, irn, jcn),
        ESymSolverStatus::Success
    );
    s.values_array_mut().copy_from_slice(vals);
    let mut rhs = vec![0.0; n as usize];
    let st = s.multi_solve(true, irn, jcn, 1, &mut rhs, false, 0);
    (s, st)
}

/// Lower-triangle triplets of `diag(d)`, 1-based.
fn diag_triplets(d: &[Number]) -> (Vec<Index>, Vec<Index>, Vec<Number>) {
    let idx: Vec<Index> = (1..=d.len() as Index).collect();
    (idx.clone(), idx, d.to_vec())
}

/// Inertia of the factored matrix, requiring the factorization to succeed.
fn inertia_of(
    n: Index,
    irn: &[Index],
    jcn: &[Index],
    vals: &[Number],
) -> InertiaInfo {
    let (s, st) = factor(RslabConfig::default(), n, irn, jcn, vals);
    assert_eq!(st, ESymSolverStatus::Success, "factorization failed");
    s.inertia_info().clone()
}

// ---------------------------------------------------------------------------
// Diagonal matrices — every pivot is 1×1, so this is the sign-of-`d` branch.
// ---------------------------------------------------------------------------

/// `diag(3, 2, -1, -4)` → two positive, two negative, no zeros.
#[test]
fn diagonal_matrix_inertia() {
    let (irn, jcn, vals) = diag_triplets(&[3.0, 2.0, -1.0, -4.0]);
    let info = inertia_of(4, &irn, &jcn, &vals);
    assert_eq!(info.triple(), (2, 2, 0));
    assert_eq!(
        info.two_by_two_pivots, 0,
        "a diagonal matrix must factor through 1x1 pivots only"
    );
    assert!(info.reliable);
    assert_eq!(info.stable_recount, info.triple());
}

/// The negative count POUNCE actually reads off the trait is the same number.
#[test]
fn number_of_neg_evals_is_the_strict_negative_count() {
    let (irn, jcn, vals) = diag_triplets(&[3.0, 2.0, -1.0, -4.0]);
    let (s, st) = factor(RslabConfig::default(), 4, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Success);
    assert_eq!(s.number_of_neg_evals(), 2);
    assert!(s.provides_inertia());
}

/// `diag(3, -2, 0)` — the singular case.
///
/// The textbook answer is `(1 positive, 1 negative, 1 zero)`. RSLAB under its
/// own default (`ZeroPivotAction::Fail`) never reports it: it aborts the
/// factorization with `NumericallyRankDeficient` the moment it meets the zero
/// pivot, and the adapter maps that abort to `Singular`. That is the *right*
/// answer for POUNCE — `Singular` is what routes the outer loop to `δ_c`,
/// which is where a rank-deficient constraint Jacobian belongs — but it means
/// "inertia of a singular matrix" is a question RSLAB declines rather than
/// answers, and no caller should expect the zero count to be populated.
#[test]
fn singular_diagonal_is_reported_singular_not_counted() {
    let (irn, jcn, vals) = diag_triplets(&[3.0, -2.0, 0.0]);
    let (_, st) = factor(RslabConfig::default(), 3, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Singular);
}

/// Under a perturbing policy the same matrix factors — and reports the inertia
/// of the *perturbed* matrix, which is not `(1, 1, 1)`.
///
/// RSLAB's `ForceAccept` is not feral's. feral accepts the tiny pivot at face
/// value and books it in the `zero` bucket; RSLAB derives an absolute floor
/// `max(‖A‖_max, 1)·ε` and *lifts* the pivot to it, so the zero eigenvalue
/// comes back as a positive one and the reported triple is `(2, 1, 0)`. The
/// adapter cannot make that count mean what POUNCE means, and does not try:
/// `perturbed_pivots > 0` makes the inertia unreliable, so the count is never
/// spent on a `δ_w` retry.
#[test]
fn a_perturbing_policy_reports_the_perturbed_inertia_and_flags_it() {
    let cfg = RslabConfig {
        on_zero_pivot: rslab::ZeroPivotAction::ForceAccept,
        ..Default::default()
    };
    let (irn, jcn, vals) = diag_triplets(&[3.0, -2.0, 0.0]);
    let (s, st) = factor(cfg, 3, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Success);
    let info = s.inertia_info();
    assert!(
        info.perturbed_pivots > 0,
        "the zero pivot must be recorded as perturbed, got {info:?}"
    );
    assert_eq!(
        info.triple(),
        (2, 1, 0),
        "the lifted pivot reads positive, not zero: {info:?}"
    );
    assert!(
        !info.reliable,
        "a perturbed factorization's inertia is not a measurement"
    );
}

// ---------------------------------------------------------------------------
// 2×2 Bunch-Kaufman blocks — the branch a diagonal fixture never reaches.
// ---------------------------------------------------------------------------

/// `[[0, 1], [1, 0]]` — eigenvalues `+1` and `-1`.
///
/// This is the fixture that proves 2×2 pivot blocks are handled by the block's
/// eigenvalues and not by its diagonal: the diagonal is `(0, 0)`, so a
/// sign-of-`d` rule would report two zeros. The Bunch-Kaufman test takes the
/// 2×2 branch here by construction (`|a₁₁| = 0 < α·colmax`, and the candidate
/// diagonal `|a₂₂| = 0 < α·rowmax`), which `two_by_two_pivots` asserts rather
/// than assumes.
#[test]
fn antidiagonal_2x2_block_inertia() {
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![0.0, 1.0, 0.0];
    let info = inertia_of(2, &irn, &jcn, &vals);
    assert_eq!(
        info.two_by_two_pivots, 1,
        "fixture must reach the 2x2 branch, got {info:?}"
    );
    assert_eq!(info.triple(), (1, 1, 0));
    assert_eq!(info.stable_recount, (1, 1, 0));
    assert!(info.reliable);
    // The reported extremes are the block's eigenvalue magnitudes (both 1),
    // not its diagonal entries (both 0).
    let lo = info.min_abs_pivot_or_block_eigenvalue.unwrap();
    let hi = info.max_abs_pivot_or_block_eigenvalue.unwrap();
    assert!((lo - 1.0).abs() < 1e-12, "min |λ| = {lo}");
    assert!((hi - 1.0).abs() < 1e-12, "max |λ| = {hi}");
}

/// The same block embedded in a larger indefinite system, so the 2×2 arrives
/// through the sparse assembly rather than as the whole matrix.
#[test]
fn an_embedded_2x2_block_is_counted_by_its_eigenvalues() {
    // diag(5) ⊕ [[0,1],[1,0]] ⊕ diag(-7), with a weak coupling that does not
    // disturb the antidiagonal block's pivot choice.
    let irn = vec![1, 2, 3, 3, 4, 4];
    let jcn = vec![1, 2, 2, 3, 1, 4];
    let vals = vec![5.0, 0.0, 1.0, 0.0, 1e-3, -7.0];
    let info = inertia_of(4, &irn, &jcn, &vals);
    assert_eq!(
        info.two_by_two_pivots, 1,
        "fixture must reach the 2x2 branch, got {info:?}"
    );
    // Eigenvalues are ≈ (5, +1, -1, -7) — two of each sign.
    assert_eq!(info.triple(), (2, 2, 0));
    assert!(info.reliable);
}

// ---------------------------------------------------------------------------
// Near-singularity and the reliability classification.
// ---------------------------------------------------------------------------

/// A pivot at the working-precision floor: the inertia is *reported*, and
/// flagged as not a measurement.
///
/// The fixture is `[[1, 1], [1, 1 - 2⁻⁵³]]`, run unscaled so the magnitudes are
/// the ones written. Bunch-Kaufman takes a 1×1 pivot here (`|a₁₁| = 1 ≥
/// α·colmax`), leaving the Schur complement `d₂ = (1 - 2⁻⁵³) - 1 = -2⁻⁵³`,
/// which is exact. The inertia is genuinely `(1, 1, 0)` and RSLAB says so, but
/// `1.11e-16` is below the floor `n·ε = 4.44e-16`, so the pivot's *sign* is a
/// rounding artefact and the count built from it is not something to spend a
/// `δ_w` retry on.
///
/// `δ` has to be `2⁻⁵³` and not something rounder: at `1e-17` the subtraction
/// `1 - δ` rounds back to `1.0`, the matrix is exactly rank 1, and RSLAB aborts
/// with `NumericallyRankDeficient` before any of this is reached — a different
/// test.
#[test]
fn a_subfloor_pivot_makes_the_inertia_unreliable() {
    let cfg = RslabConfig {
        equilibration: Equilibration::Identity,
        ..Default::default()
    };
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0 - 2.0_f64.powi(-53)];
    assert_ne!(vals[2], 1.0, "premise: the perturbation must survive rounding");
    let (s, st) = factor(cfg, 2, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Success);
    let info = s.inertia_info();
    let lo = info.min_abs_pivot_or_block_eigenvalue.unwrap();
    assert!(
        lo < pounce_rslab::inertia_trust_floor(None, 2),
        "premise: the smallest pivot {lo} must sit under the floor"
    );
    assert!(
        !info.reliable,
        "a count read off a sub-floor pivot is not a measurement: {info:?}"
    );
    assert_eq!(
        info.perturbed_pivots, 0,
        "nothing was perturbed; the floor alone disqualifies it"
    );
}

/// And the disqualification reaches POUNCE as `Singular`, not `WrongInertia`.
///
/// This is the pounce gh#540 behaviour the adapter exists to preserve:
/// `WrongInertia` sends the IPM up the `δ_w` ladder (×8 per retry), which damps
/// the Newton step to nothing, while the perturbation that repairs a
/// rank-deficient constraint Jacobian is `δ_c` — which is what `Singular`
/// reaches for.
#[test]
fn an_untrustworthy_mismatch_is_singular_not_wrong_inertia() {
    let cfg = RslabConfig {
        equilibration: Equilibration::Identity,
        ..Default::default()
    };
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0 - 2.0_f64.powi(-53)];

    let mut s = RslabSolverInterface::with_config(cfg);
    s.initialize_structure(2, 3, &irn, &jcn);
    s.values_array_mut().copy_from_slice(&vals);
    let mut rhs = vec![0.0; 2];
    // The true inertia is (1, 1, 0); ask for 0 negatives so the counts differ.
    assert_eq!(
        s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 0),
        ESymSolverStatus::Singular
    );
}

/// With the floor pinned off, the same mismatch is `WrongInertia` — so the
/// test above is measuring the floor and not something else.
#[test]
fn with_the_floor_disabled_the_same_mismatch_is_wrong_inertia() {
    let cfg = RslabConfig {
        equilibration: Equilibration::Identity,
        inertia_pivot_floor: Some(0.0),
        ..Default::default()
    };
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0 - 2.0_f64.powi(-53)];

    let mut s = RslabSolverInterface::with_config(cfg);
    s.initialize_structure(2, 3, &irn, &jcn);
    s.values_array_mut().copy_from_slice(&vals);
    let mut rhs = vec![0.0; 2];
    assert_eq!(
        s.multi_solve(true, &irn, &jcn, 1, &mut rhs, true, 0),
        ESymSolverStatus::WrongInertia
    );
}

/// The `singular_pivot_floor` knob (the MA57 `CNTL(2)` analog) fires
/// independently of any inertia comparison.
#[test]
fn the_singular_pivot_floor_fires_without_an_inertia_check() {
    let cfg = RslabConfig {
        equilibration: Equilibration::Identity,
        singular_pivot_floor: 1e-10,
        ..Default::default()
    };
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![1.0, 1.0, 1.0 - 2.0_f64.powi(-53)];
    // `check_neg_evals = false`, so only the floor can produce this status.
    let (_, st) = factor(cfg, 2, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Singular);
}

/// A healthy factor of the same shape is `Success` under the same floor, so
/// the floor is not simply rejecting everything.
#[test]
fn a_healthy_factor_passes_the_singular_pivot_floor() {
    let cfg = RslabConfig {
        equilibration: Equilibration::Identity,
        singular_pivot_floor: 1e-10,
        ..Default::default()
    };
    let irn = vec![1, 2, 2];
    let jcn = vec![1, 1, 2];
    let vals = vec![2.0, 1.0, 3.0];
    let (_, st) = factor(cfg, 2, &irn, &jcn, &vals);
    assert_eq!(st, ESymSolverStatus::Success);
}

// ---------------------------------------------------------------------------
// Invariance.
// ---------------------------------------------------------------------------

/// Inertia is invariant under a symmetric permutation `PᵀAP` (Sylvester), so
/// relabelling the variables must not move the counts — nor the pivot
/// magnitude extent, since the eigenvalues are the same.
///
/// The fixture is indefinite and mixes 1×1 and 2×2 pivot candidates, and each
/// permutation is applied to the *triplets*, which also re-orders the columns
/// RSLAB's `from_triplets` sees and hence the ordering its analysis picks.
#[test]
fn inertia_survives_symmetric_permutation() {
    // 6×6 indefinite: an arrow with an antidiagonal 2×2 in the middle.
    let base_irn: Vec<Index> = vec![1, 2, 3, 3, 4, 5, 6, 6, 6];
    let base_jcn: Vec<Index> = vec![1, 2, 2, 3, 4, 5, 1, 4, 6];
    let base_vals: Vec<Number> = vec![4.0, 0.0, 1.0, 0.0, -3.0, 2.0, 0.5, -0.25, -1.0];

    let reference = inertia_of(6, &base_irn, &base_jcn, &base_vals);
    assert_eq!(
        reference.total_counted(),
        6,
        "the counts must cover the whole matrix"
    );

    // A handful of permutations, including a reversal and two rotations.
    let perms: [[usize; 6]; 4] = [
        [5, 4, 3, 2, 1, 0],
        [2, 0, 4, 1, 5, 3],
        [1, 2, 3, 4, 5, 0],
        [3, 1, 5, 0, 2, 4],
    ];
    for p in perms {
        // p[i] is where original index i lands (0-based).
        let mut irn = Vec::with_capacity(base_irn.len());
        let mut jcn = Vec::with_capacity(base_jcn.len());
        for k in 0..base_irn.len() {
            let i = p[(base_irn[k] - 1) as usize] as Index + 1;
            let j = p[(base_jcn[k] - 1) as usize] as Index + 1;
            // The triplet must stay in the lower triangle after relabelling.
            let (i, j) = if i >= j { (i, j) } else { (j, i) };
            irn.push(i);
            jcn.push(j);
        }
        let info = inertia_of(6, &irn, &jcn, &base_vals);
        assert_eq!(
            info.triple(),
            reference.triple(),
            "permutation {p:?} moved the inertia"
        );
        let a = reference.min_abs_pivot_or_block_eigenvalue.unwrap();
        let b = info.min_abs_pivot_or_block_eigenvalue.unwrap();
        assert!(
            (a - b).abs() <= 1e-9 * a.max(b),
            "permutation {p:?} moved min |λ|: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// The threshold convention is shared with FERAL, not reinvented.
// ---------------------------------------------------------------------------

/// `pounce_rslab::inertia_trust_floor` is `pounce_feral::inertia_trust_floor`.
///
/// The function is reproduced rather than imported (`pounce-feral` is a
/// dev-dependency here, so a build that wants only RSLAB does not drag FERAL
/// in), which makes it a copy that can drift. This is the guard: if FERAL's
/// floor changes and this one does not, the two backends stop being judged
/// against the same threshold and the comparison between them silently stops
/// meaning anything.
#[test]
fn the_trust_floor_matches_ferals() {
    for dim in [0usize, 1, 2, 17, 311, 4503, 100_000, 1 << 20] {
        assert_eq!(
            pounce_rslab::inertia_trust_floor(None, dim),
            pounce_feral::inertia_trust_floor(None, dim),
            "dimension-aware floor diverged at n = {dim}"
        );
        for pinned in [0.0, 1e-12, 1e-8] {
            assert_eq!(
                pounce_rslab::inertia_trust_floor(Some(pinned), dim),
                pounce_feral::inertia_trust_floor(Some(pinned), dim),
                "pinned floor {pinned} diverged at n = {dim}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The adapter's equilibration is RSLAB's own.
// ---------------------------------------------------------------------------

/// The adapter reimplements `rslab::LdltSolver`'s one-pass inf-norm
/// equilibration because RSLAB's own is private. This pins the reimplementation
/// against the original, end to end: on a deliberately badly scaled indefinite
/// system, the adapter and `LdltSolver` must land on the same solution.
///
/// They cannot be bit-identical — `LdltSolver` runs the supernodal panel sweep
/// and the adapter runs the CSC reference sweep — so the comparison is to a
/// tight relative tolerance. A divergent scaling would show up far outside it:
/// the point of equilibrating a system with an `1e8` spread is that it moves
/// the answer.
#[test]
fn the_adapters_equilibration_matches_rslabs() {
    use rslab::{CscMatrix, LdltSolver};

    // Indefinite, with row scales spanning 8 orders of magnitude.
    let n = 5usize;
    let rows_0 = vec![0usize, 1, 2, 2, 3, 4, 4];
    let cols_0 = vec![0usize, 1, 1, 2, 3, 0, 4];
    let vals = vec![1e8, 2.0, -3.0, 1e-4, -5.0, 7e3, 4.0];

    let a = CscMatrix::<f64>::from_triplets(n, &rows_0, &cols_0, &vals).unwrap();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let want = LdltSolver::<f64>::factor(&a).unwrap().solve(&b).unwrap();

    let irn: Vec<Index> = rows_0.iter().map(|&r| r as Index + 1).collect();
    let jcn: Vec<Index> = cols_0.iter().map(|&c| c as Index + 1).collect();
    let mut s = RslabSolverInterface::new();
    s.initialize_structure(n as Index, vals.len() as Index, &irn, &jcn);
    s.values_array_mut().copy_from_slice(&vals);
    let mut got = b.clone();
    assert_eq!(
        s.multi_solve(true, &irn, &jcn, 1, &mut got, false, 0),
        ESymSolverStatus::Success
    );

    for i in 0..n {
        let scale = want[i].abs().max(got[i].abs()).max(1e-300);
        assert!(
            (want[i] - got[i]).abs() <= 1e-10 * scale,
            "entry {i}: LdltSolver {}, adapter {}",
            want[i],
            got[i]
        );
    }
}

/// Running with `Equilibration::Identity` on the same system gives a
/// measurably worse answer — evidence that the equilibration above is doing
/// work, rather than being a no-op the test cannot distinguish.
#[test]
fn turning_the_equilibration_off_is_observable() {
    let n = 5usize;
    let rows_0 = vec![0usize, 1, 2, 2, 3, 4, 4];
    let cols_0 = vec![0usize, 1, 1, 2, 3, 0, 4];
    let vals = vec![1e8, 2.0, -3.0, 1e-4, -5.0, 7e3, 4.0];
    let irn: Vec<Index> = rows_0.iter().map(|&r| r as Index + 1).collect();
    let jcn: Vec<Index> = cols_0.iter().map(|&c| c as Index + 1).collect();

    let mut scaled = RslabSolverInterface::new();
    let mut plain = RslabSolverInterface::with_config(RslabConfig {
        equilibration: Equilibration::Identity,
        ..Default::default()
    });
    let b: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let mut xs = b.clone();
    let mut xp = b.clone();
    for (s, x) in [(&mut scaled, &mut xs), (&mut plain, &mut xp)] {
        s.initialize_structure(n as Index, vals.len() as Index, &irn, &jcn);
        s.values_array_mut().copy_from_slice(&vals);
        assert_eq!(
            s.multi_solve(true, &irn, &jcn, 1, x, false, 0),
            ESymSolverStatus::Success
        );
    }
    // Not asserting which is better — only that the knob changes the answer,
    // so a test comparing against `LdltSolver` can tell the two apart.
    let differs = (0..n).any(|i| xs[i] != xp[i]);
    assert!(differs, "the equilibration knob had no effect at all");
}

trait TotalCounted {
    fn total_counted(&self) -> usize;
}
impl TotalCounted for InertiaInfo {
    fn total_counted(&self) -> usize {
        self.positive + self.negative + self.zero
    }
}
