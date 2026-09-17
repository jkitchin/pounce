//! Cross-solver comparison: FERAL, RSLAB, and (under `--features ma57`) MA57
//! on the same matrices.
//!
//! Two kinds of fixture, and the distinction is the point of the file.
//!
//! **Synthetic** matrices pin the harness itself and the properties every
//! backend must have. They are small, well understood, and prove nothing about
//! a KKT.
//!
//! **Real POUNCE KKT systems**, checked in under `tests/fixtures/`, pin the
//! result that decides whether RSLAB is usable. They were captured by
//! `examples/rslab_kkt_replay.rs` — an actual POUNCE solve with a capturing
//! backend installed through `IpoptApplication::set_linear_backend_factory` —
//! and they carry the negative-eigenvalue count the interior-point loop
//! expected at that iteration, so a backend can be checked against the
//! contract rather than against another backend's opinion.
//!
//! The two fixtures were chosen to reach **different branches** of RSLAB's
//! behaviour, since a fixture that always lands on one side is no evidence
//! about the other:
//!
//! * `airport_kkt_iter0` — RSLAB factors it, using 2×2 pivots, and agrees with
//!   FERAL about the inertia. The "it works" branch.
//! * `eigena2_kkt_iter0` — RSLAB refuses it as `NumericallyRankDeficient`
//!   although a dense eigensolve puts its condition number at 2.6. The
//!   "no delayed pivoting" branch, which is the one that matters.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use pounce_common::types::{Index, Number};
use pounce_linsol::ESymSolverStatus;
use pounce_rslab::compare::{self, SymTriplet};

/// A captured KKT plus the inertia the IPM expected of it.
struct KktFixture {
    matrix: SymTriplet,
    rhs: Vec<Number>,
    expected_neg: Index,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Parse the flat capture format `examples/` writes: `n`, `nnz`,
/// `expected_neg`, then one `t <irn> <jcn> <val>` per triplet and one
/// `b <val>` per right-hand-side entry.
fn load_kkt(name: &str) -> KktFixture {
    let path = fixture_dir().join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let (mut n, mut expected_neg) = (0 as Index, 0 as Index);
    let (mut irn, mut jcn, mut vals, mut rhs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        let mut f = line.split_whitespace();
        match f.next() {
            Some("n") => n = f.next().unwrap().parse().unwrap(),
            Some("expected_neg") => expected_neg = f.next().unwrap().parse().unwrap(),
            Some("t") => {
                irn.push(f.next().unwrap().parse().unwrap());
                jcn.push(f.next().unwrap().parse().unwrap());
                vals.push(f.next().unwrap().parse().unwrap());
            }
            Some("b") => rhs.push(f.next().unwrap().parse().unwrap()),
            _ => {}
        }
    }
    assert!(n > 0 && !vals.is_empty(), "{name}: empty fixture");
    assert_eq!(rhs.len(), n as usize, "{name}: rhs length");
    KktFixture {
        matrix: SymTriplet { n, irn, jcn, vals },
        rhs,
        expected_neg,
    }
}

fn record<'a>(recs: &'a [compare::SolveRecord], solver: &str) -> &'a compare::SolveRecord {
    recs.iter()
        .find(|r| r.solver == solver)
        .unwrap_or_else(|| panic!("no record for {solver}"))
}

// ---------------------------------------------------------------------------
// Synthetic: the properties every backend must have.
// ---------------------------------------------------------------------------

/// A well-conditioned indefinite system with a zero diagonal block — the
/// smallest thing shaped like a KKT. Every backend must factor it, agree about
/// the inertia, and solve it to working precision.
fn small_saddle() -> SymTriplet {
    // [[2, 0, 1], [0, 3, 1], [1, 1, 0]] — eigenvalues straddle zero: (2, 1, 0).
    SymTriplet {
        n: 3,
        irn: vec![1, 2, 3, 3, 3],
        jcn: vec![1, 2, 1, 2, 3],
        vals: vec![2.0, 3.0, 1.0, 1.0, 0.0],
    }
}

#[test]
fn every_backend_agrees_on_a_small_saddle_point() {
    let a = small_saddle();
    let b = a.sample_rhs();

    // The outside number first: if the oracle and the backends disagree, it is
    // the backends that are wrong.
    let (pos, neg, zero, _lo, _hi) =
        compare::dense_inertia_oracle(&a, 64).expect("oracle runs at n = 3");
    assert_eq!((pos, neg, zero), (2, 1, 0), "oracle inertia");

    for rec in compare::run_all(&a, &b) {
        assert_eq!(
            rec.factor_status,
            ESymSolverStatus::Success,
            "{} failed to factor a kappa-small saddle point",
            rec.solver
        );
        assert_eq!(
            rec.negative_evals,
            Some(neg as Index),
            "{} disagrees with the dense oracle",
            rec.solver
        );
        assert!(
            rec.is_acceptable(1e-10),
            "{}: residual_ratio {:?}",
            rec.solver,
            rec.residual_ratio
        );
    }
}

/// An exactly singular system is `Singular` on every backend, not a silent
/// wrong answer and not a panic.
#[test]
fn every_backend_reports_a_singular_system_as_singular() {
    let a = SymTriplet {
        n: 3,
        irn: vec![1, 2, 2, 3, 3],
        jcn: vec![1, 1, 2, 1, 3],
        // Rows 1 and 2 are identical → rank 2, and the third is independent.
        vals: vec![1.0, 1.0, 1.0, 0.0, 0.0],
    };
    let b = a.sample_rhs();
    for rec in compare::run_all(&a, &b) {
        // `rslab-sp` static-pivots rather than failing — that is its job — so
        // it is allowed to succeed, and must then flag the inertia.
        if rec.solver == "rslab-sp" {
            if rec.factor_status == ESymSolverStatus::Success {
                assert_eq!(
                    rec.inertia_reliable,
                    Some(false),
                    "rslab-sp certified a perturbed factor's inertia"
                );
            }
            continue;
        }
        assert_eq!(
            rec.factor_status,
            ESymSolverStatus::Singular,
            "{} did not report a rank-deficient system as singular",
            rec.solver
        );
    }
}

/// Every backend reuses its symbolic analysis across a refactorization.
///
/// This is what the IPM pays for, hundreds of times per solve: the pattern is
/// fixed by `initialize_structure` and only the values change, so a backend
/// that re-orders and re-amalgamates on every call is disqualified on cost
/// before any accuracy question is reached.
///
/// Asserted **structurally**, not on wall-clock. `run_backend` factors twice,
/// and `LinearSolverSummary` counts how many of those reused the cached
/// analysis — an exact number. The first version of this test compared the
/// refactorization's milliseconds against the first factor's, which on a
/// 168-row matrix is a 3 ms measurement against a 5 ms one under a parallel
/// test runner: it passed, then failed, and it was measuring the machine.
#[test]
fn every_backend_reuses_its_symbolic_analysis() {
    let a = load_kkt("airport_kkt_iter0").matrix;
    let b = a.sample_rhs();
    for rec in compare::run_all(&a, &b) {
        if rec.factor_status != ESymSolverStatus::Success {
            continue;
        }
        assert_eq!(
            rec.n_pattern_changes, 1,
            "{}: only the first factorization may build a fresh analysis",
            rec.solver
        );
        assert!(
            rec.n_pattern_reuse >= 1,
            "{}: the refactorization did not reuse the analysis ({} reuses)",
            rec.solver,
            rec.n_pattern_reuse
        );
    }
}

/// And for RSLAB the skipped phase is visible directly: a refactorization
/// spends exactly zero time in the symbolic analysis, because it does not
/// enter it at all.
#[test]
fn an_rslab_refactorization_skips_the_symbolic_phase_entirely() {
    use pounce_linsol::SparseSymLinearSolverInterface as _;

    let a = load_kkt("airport_kkt_iter0").matrix;
    let mut s = pounce_rslab::RslabSolverInterface::new();
    s.initialize_structure(a.n, a.nnz() as Index, &a.irn, &a.jcn);
    s.values_array_mut().copy_from_slice(&a.vals);
    let mut rhs = a.sample_rhs();
    assert_eq!(
        s.multi_solve(true, &a.irn, &a.jcn, 1, &mut rhs, false, 0),
        ESymSolverStatus::Success
    );
    assert!(
        s.phase_timings().symbolic_ms > 0.0,
        "premise: the first factorization runs the analysis"
    );

    s.values_array_mut().copy_from_slice(&a.vals);
    let mut rhs = a.sample_rhs();
    assert_eq!(
        s.multi_solve(true, &a.irn, &a.jcn, 1, &mut rhs, false, 0),
        ESymSolverStatus::Success
    );
    assert_eq!(
        s.phase_timings().symbolic_ms,
        0.0,
        "a refactorization must not re-enter the symbolic analysis"
    );
    assert!(
        s.phase_timings().numeric_ms > 0.0,
        "…but it must still do the numeric work"
    );

    // A pure back-solve reaches no factor phase at all, so a solve-only call
    // cannot be read as a factorization.
    let mut rhs = a.sample_rhs();
    assert_eq!(
        s.multi_solve(false, &a.irn, &a.jcn, 1, &mut rhs, false, 0),
        ESymSolverStatus::Success
    );
    assert_eq!(s.phase_timings().factor_total_ms(), 0.0);
    assert!(s.phase_timings().solve_ms > 0.0);
}

// ---------------------------------------------------------------------------
// Real POUNCE KKT systems.
// ---------------------------------------------------------------------------

/// `airport`, first factorization: the branch where RSLAB works.
///
/// Every backend factors it, every backend returns the negative count the IPM
/// asked for, and RSLAB reaches the 2×2 branch while doing so. The residual
/// assertion is deliberately loose (`1e-8` on POUNCE's own metric) — this test
/// exists to pin *agreement*, and the measured gap between the two is recorded
/// in `dev-notes/rslab-backend-assessment.md`, not asserted here, because it
/// is a number that will move.
#[test]
fn rslab_matches_feral_on_a_kkt_it_can_factor() {
    let f = load_kkt("airport_kkt_iter0");
    let recs = compare::run_all(&f.matrix, &f.rhs);

    let feral = record(&recs, "feral");
    let rslab = record(&recs, "rslab");
    assert_eq!(feral.factor_status, ESymSolverStatus::Success);
    assert_eq!(
        rslab.factor_status,
        ESymSolverStatus::Success,
        "premise: RSLAB factors this one"
    );
    assert_eq!(
        rslab.negative_evals, feral.negative_evals,
        "backends disagree about the inertia"
    );
    assert_eq!(
        rslab.negative_evals,
        Some(f.expected_neg),
        "RSLAB disagrees with the count the IPM expected"
    );
    assert!(
        rslab.two_by_two_pivots.is_some_and(|n| n > 0),
        "fixture must reach RSLAB's 2x2 branch, got {:?}",
        rslab.two_by_two_pivots
    );
    assert!(
        rslab.inertia_reliable == Some(true),
        "a clean factorization's inertia should be trustworthy: {rslab:?}"
    );
    for rec in [feral, rslab] {
        assert!(
            rec.is_acceptable(1e-8),
            "{}: residual_ratio {:?}",
            rec.solver,
            rec.residual_ratio
        );
    }
}

/// `eigena2`, first factorization: the branch that decides the assessment.
///
/// A dense eigensolve puts this matrix at `kappa = 2.6` with inertia
/// `(110, 55, 0)` — full rank, and about as well conditioned as a KKT ever
/// gets. FERAL factors it and solves to `1e-16`. RSLAB in exact mode returns
/// `NumericallyRankDeficient`, which the adapter maps to `Singular`.
///
/// That is not a tuning accident and not conditioning: RSLAB restricts
/// Bunch-Kaufman pivoting to each front's fully-summed block and has no
/// delayed pivoting, which its own module docs state. A saddle-point row
/// factors only when its 2×2 partner lands in the same front.
///
/// This test is written to fail if RSLAB ever grows delayed pivoting — at
/// which point the assessment's central finding is stale and should be redone,
/// which is exactly what a failure here should prompt.
#[test]
fn rslab_refuses_a_well_conditioned_kkt_that_feral_factors() {
    let f = load_kkt("eigena2_kkt_iter0");

    let (pos, neg, zero, lo, hi) =
        compare::dense_inertia_oracle(&f.matrix, 400).expect("oracle runs at n = 165");
    assert_eq!(
        (pos, neg, zero),
        (110, 55, 0),
        "oracle: the matrix is full rank"
    );
    assert!(
        hi / lo < 10.0,
        "oracle: premise is that this matrix is well conditioned, kappa = {}",
        hi / lo
    );

    let recs = compare::run_all(&f.matrix, &f.rhs);
    let feral = record(&recs, "feral");
    assert_eq!(feral.factor_status, ESymSolverStatus::Success);
    assert_eq!(feral.negative_evals, Some(f.expected_neg));
    assert!(
        feral.is_acceptable(1e-12),
        "feral residual_ratio {:?}",
        feral.residual_ratio
    );

    let rslab = record(&recs, "rslab");
    assert_eq!(
        rslab.factor_status,
        ESymSolverStatus::Singular,
        "RSLAB now factors this; the no-delayed-pivoting finding in \
         dev-notes/rslab-backend-assessment.md needs re-measuring"
    );
}

/// Static pivoting is the only RSLAB mode that completes that factorization —
/// and what it produces is a preconditioner, which the adapter must not let
/// POUNCE mistake for a measurement.
#[test]
fn static_pivoting_completes_it_but_the_inertia_is_not_a_measurement() {
    let f = load_kkt("eigena2_kkt_iter0");
    let rec = compare::run_rslab_static_pivoting(&f.matrix, &f.rhs, 1e-12);
    assert_eq!(
        rec.factor_status,
        ESymSolverStatus::Success,
        "static pivoting must not fail"
    );
    assert!(
        rec.perturbed_pivots.is_some_and(|n| n > 0),
        "premise: pivots were perturbed, got {:?}",
        rec.perturbed_pivots
    );
    assert_eq!(
        rec.inertia_reliable,
        Some(false),
        "a perturbed factor's inertia must not be certified"
    );
    // It still gets the right answer here — the point is that POUNCE is not
    // entitled to assume so.
    assert_eq!(rec.negative_evals, Some(f.expected_neg));
}

/// FERAL's residual is smaller than RSLAB's on the KKT they both factor, and
/// the gap is large enough that it is not measurement noise.
///
/// Asserted as a direction and an order of magnitude rather than a value: the
/// measured ratio on this fixture is ~1e3, and pinning that exactly would make
/// the test fail on an unrelated improvement to either solver.
#[test]
fn feral_is_the_more_accurate_of_the_two_where_both_factor() {
    let f = load_kkt("airport_kkt_iter0");
    let recs = compare::run_all(&f.matrix, &f.rhs);
    let feral = record(&recs, "feral").residual_ratio.expect("feral solved");
    let rslab = record(&recs, "rslab").residual_ratio.expect("rslab solved");
    assert!(
        feral < rslab,
        "premise reversed: feral {feral:.3e} is no longer smaller than rslab {rslab:.3e}"
    );
    // Both are still far inside anything POUNCE would reject.
    assert!(rslab < 1e-10, "rslab residual_ratio {rslab:.3e}");
}

/// A synthetic reproducer of the failure, independent of any captured fixture.
///
/// The shape is a saddle point `[[W + Σ, Jᵀ], [J, 0]]` over a grid Laplacian.
/// Two variants, and the pair is the test:
///
/// * a **dense-row** `J`, where each constraint couples a whole grid row —
///   RSLAB factors it at every size tried, up to `n = 160 400`;
/// * a **sparse-row** `J` with one entry per constraint, which is what a slack
///   or simple-bound row contributes and what gives a KKT its arrow signature
///   — RSLAB refuses it at every size.
///
/// FERAL factors both. The first variant alone would have said the failure
/// does not exist, which is why the benchmark carries both and why the real
/// KKT capture was necessary to find it in the first place: a corpus uniform
/// in the dimension a defect acts on reports nothing, however large its
/// models are.
///
/// This is the minimal form of the finding, so it is what a bug report to
/// RSLAB should carry.
#[test]
fn a_sparse_jacobian_saddle_point_reproduces_the_failure() {
    fn grid_saddle(k: usize, dense_rows: bool) -> SymTriplet {
        let nx = k * k;
        let nc = if dense_rows { k } else { nx / 2 };
        let (mut irn, mut jcn, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let at = |i: usize, j: usize| i * k + j;
        for i in 0..k {
            for j in 0..k {
                let r = at(i, j) as Index + 1;
                irn.push(r);
                jcn.push(r);
                vals.push(5.0);
                if i > 0 {
                    irn.push(r);
                    jcn.push(at(i - 1, j) as Index + 1);
                    vals.push(-1.0);
                }
                if j > 0 {
                    irn.push(r);
                    jcn.push(at(i, j - 1) as Index + 1);
                    vals.push(-1.0);
                }
            }
        }
        if dense_rows {
            for c in 0..nc {
                for j in 0..k {
                    irn.push((nx + c) as Index + 1);
                    jcn.push(at(c, j) as Index + 1);
                    vals.push(1.0 + 0.1 * j as f64);
                }
            }
        } else {
            for c in 0..nc {
                irn.push((nx + c) as Index + 1);
                jcn.push((2 * c) as Index + 1);
                vals.push(1.0);
            }
        }
        // The structurally present, numerically zero (2,2) block POUNCE emits.
        for c in 0..nc {
            irn.push((nx + c) as Index + 1);
            jcn.push((nx + c) as Index + 1);
            vals.push(0.0);
        }
        SymTriplet {
            n: (nx + nc) as Index,
            irn,
            jcn,
            vals,
        }
    }

    for k in [8usize, 16, 24] {
        let dense = grid_saddle(k, true);
        let sparse = grid_saddle(k, false);
        for (tag, a) in [("dense-J", &dense), ("sparse-J", &sparse)] {
            let b = a.sample_rhs();
            let recs = compare::run_all(a, &b);
            assert_eq!(
                record(&recs, "feral").factor_status,
                ESymSolverStatus::Success,
                "k={k} {tag}: FERAL must factor both variants"
            );
            assert!(
                record(&recs, "rslab-sp").factor_status == ESymSolverStatus::Success,
                "k={k} {tag}: static pivoting must not fail"
            );
        }
        assert_eq!(
            record(&compare::run_all(&dense, &dense.sample_rhs()), "rslab").factor_status,
            ESymSolverStatus::Success,
            "k={k}: RSLAB factors the dense-row variant — this is the branch that              hides the defect"
        );
        assert_eq!(
            record(&compare::run_all(&sparse, &sparse.sample_rhs()), "rslab").factor_status,
            ESymSolverStatus::Singular,
            "k={k}: RSLAB now factors the sparse-row saddle point; the              no-delayed-pivoting finding needs re-measuring"
        );
    }
}

/// Turning equilibration off on both sides does not rescue RSLAB, so the
/// failure is not an artefact of the adapter's scaling differing from FERAL's.
///
/// The two backends cannot be made to equilibrate identically — FERAL's
/// default is MC64/Knight-Ruiz and RSLAB exposes neither — so this control is
/// the only way to compare the factorization kernels rather than the solvers.
#[test]
fn the_failure_survives_with_equilibration_off_on_both_sides() {
    let f = load_kkt("eigena2_kkt_iter0");
    let recs = compare::run_scaling_neutral_pair(&f.matrix, &f.rhs);
    assert_eq!(
        record(&recs, "feral-ns").factor_status,
        ESymSolverStatus::Success,
        "unscaled FERAL still factors it"
    );
    assert_eq!(
        record(&recs, "rslab-ns").factor_status,
        ESymSolverStatus::Singular,
        "unscaled RSLAB still refuses it"
    );
}
