//! End-to-end semidefinite programs through the PSD cone (PR70 item D).
//!
//! `ConeSpec::Psd(n)` is the least-exercised symmetric cone at the *program*
//! level — the unit tests in `cones/psd.rs` cover the cone primitives (svec /
//! smat / projection / barrier), but nothing drives a full SDP through
//! `solve_socp_ipm`. These tests do, against problems with closed-form optima.
//!
//! svec convention (see `cones/psd.rs`): lower triangle, column by column —
//! `(0,0),(1,0),…,(n-1,0),(1,1),…`, with off-diagonal entries scaled by `√2`
//! so `⟨X,Y⟩_F = svec(X)·svec(Y)`. A program constrains the slack
//! `s = h − G x ∈ PSD`, so `s` must equal `svec(M(x))`.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn opts() -> QpOptions {
    QpOptions {
        max_iter: 200,
        ..QpOptions::default()
    }
}

const R2: f64 = std::f64::consts::SQRT_2;

/// Minimum `t` such that `[[t, 1], [1, t]] ⪰ 0`. Eigenvalues are `t ± 1`, so
/// the matrix is PSD iff `t ≥ 1`; the optimum is `t = 1` (a rank-deficient,
/// on-the-boundary solution — the adversarial case for a PSD IPM).
#[test]
fn sdp_min_diagonal_psd_cone_2x2() {
    // var: t (n=1). svec(M(t)) = (t, √2·1, t).  s = h − G t ∈ PSD₂.
    //   s0 = M00 = t      -> h0=0,  G(0,0) = −1
    //   s1 = √2·M10 = √2  -> h1=√2, G row absent
    //   s2 = M11 = t      -> h2=0,  G(2,0) = −1
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0], // min t
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(2, 0, -1.0)],
        h: vec![0.0, R2, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(2)], &opts(), backend);
    assert_eq!(sol.status, QpStatus::Optimal, "status {:?}", sol.status);
    assert!((sol.x[0] - 1.0).abs() < 1e-5, "t = {} (want 1)", sol.x[0]);
    assert!((sol.obj - 1.0).abs() < 1e-5, "obj = {} (want 1)", sol.obj);
}

/// Maximum-eigenvalue SDP: `min t s.t. t·I − A ⪰ 0` gives `t = λ_max(A)`.
/// For `A = [[2, 1], [1, 2]]`, `λ_max = 3`.  This exercises a non-trivial
/// constant matrix in the constraint and a known spectral optimum.
#[test]
fn sdp_max_eigenvalue_psd_cone() {
    // var: t (n=1).  M(t) = t·I − A = [[t−2, −1], [−1, t−2]].
    // svec(M) = (t−2, √2·(−1), t−2).  s = h − G t ∈ PSD₂.
    //   s0 = t − 2     -> h0=−2,  G(0,0) = −1
    //   s1 = −√2       -> h1=−√2, G row absent
    //   s2 = t − 2     -> h2=−2,  G(2,0) = −1
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(2, 0, -1.0)],
        h: vec![-2.0, -R2, -2.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_socp_ipm(&prob, &[ConeSpec::Psd(2)], &opts(), backend);
    assert_eq!(sol.status, QpStatus::Optimal, "status {:?}", sol.status);
    assert!(
        (sol.x[0] - 3.0).abs() < 1e-5,
        "λ_max = {} (want 3)",
        sol.x[0]
    );
}

/// Infeasibility honesty on the PSD cone: require both `[[t,2],[2,t]] ⪰ 0`
/// (needs `t ≥ 2`) and `t ≤ 1`. Empty feasible set — the solver must NOT
/// report a false optimum.
///
/// LIMITATION (PR70 item D finding): unlike the orthant path — which returns a
/// clean `PrimalInfeasible` Farkas certificate — the symmetric HSDE driver here
/// hits a KKT factorization breakdown (`NumericalFailure`) near the PSD cone
/// boundary *before* the embedding drives τ→0 far enough to extract the
/// certificate. That is a robustness gap, not a wrong-answer bug: the
/// safety-critical property (never a confident wrong `Optimal`) still holds, so
/// we assert exactly that. Tighten to `== PrimalInfeasible` once PSD
/// infeasibility certification is hardened.
#[test]
fn sdp_infeasible_psd_cone_never_reports_optimal() {
    // var: t (n=1).  Rows 0..3: svec of [[t,2],[2,t]] ∈ PSD₂.  Row 3: t ≤ 1.
    //   s0 = t        -> h0=0,   G(0,0) = −1
    //   s1 = 2√2      -> h1=2√2, G row absent
    //   s2 = t        -> h2=0,   G(2,0) = −1
    //   s3 = 1 − t ≥ 0 (Nonneg) -> h3=1, G(3,0) = 1
    let prob = QpProblem {
        n: 1,
        p_lower: vec![],
        c: vec![1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(3, 0, 1.0),
        ],
        h: vec![0.0, 2.0 * R2, 0.0, 1.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve_socp_ipm(
        &prob,
        &[ConeSpec::Psd(2), ConeSpec::Nonneg(1)],
        &opts(),
        backend,
    );
    // Safety property: an empty feasible set must never be reported as solved.
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "infeasible SDP must not report Optimal"
    );
    // With the cone-aware Farkas check (the multiplier `z` is validated against
    // the actual PSD/orthant dual cone, not merely componentwise), the
    // infeasible SDP now yields the clean `PrimalInfeasible` certificate.
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "expected a PrimalInfeasible Farkas certificate, got {:?}",
        sol.status
    );
}
