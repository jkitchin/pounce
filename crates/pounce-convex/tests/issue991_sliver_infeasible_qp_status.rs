//! gh #991 — a convex QP that is primal infeasible by a small exact margin
//! must come back as `PrimalInfeasible` from the active-set driver, not
//! `IterationLimit`.
//!
//! The infeasible counterpart of #969. The reported run spent 3 iterations
//! against an untouched budget and the verdict did not move with `max_iter`,
//! so `iteration_limit` (AMPL `solve_result_num = 400`, "raise the limit and
//! retry") was not describing what happened; the IPM and the NLP arm both
//! report `200` ("the model is infeasible") on identical data.
//!
//! Oracle: closed form. The rows `x₀ + 2x₁ ≤ 2` and `−x₀ − 2x₁ ≤ −(2 + ε)`
//! carry the Farkas pair `z = (1, 1)`: `Gᵀz = 0` and `hᵀz = −ε < 0`, so no `x`
//! satisfies both. Every status assertion is cross-checked against POUNCE's
//! IPM on the same data, so a test that went wrong together with the oracle
//! engine would not pass silently.

use pounce_convex::{
    ActiveSetOverrides, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_active_set, solve_qp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn active_set_with(prob: &QpProblem, opts: &QpOptions) -> QpStatus {
    let mut mk = backend;
    solve_qp_active_set(prob, opts, &ActiveSetOverrides::default(), &mut mk).status
}

fn active_set(prob: &QpProblem) -> QpStatus {
    active_set_with(prob, &QpOptions::default())
}

/// The issue's model: `P = [[2, 1], [1, 2]]`, `c = (−1, 2)`, the sliver
/// `x₀ + 2x₁ ≤ 2`, `x₀ + 2x₁ ≥ 2 + ε`, both variables free.
fn sliver_qp(eps: f64) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![
            Triplet::new(0, 0, 2.0),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 2.0),
        ],
        c: vec![-1.0, 2.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(1, 0, -1.0),
            Triplet::new(1, 1, -2.0),
        ],
        h: vec![2.0, -(2.0 + eps)],
        lb: vec![],
        ub: vec![],
    }
}

#[test]
fn issue_model_is_primal_infeasible() {
    let prob = sliver_qp(1e-5);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::PrimalInfeasible,
        "IPM oracle"
    );
    assert_eq!(
        active_set(&prob),
        QpStatus::PrimalInfeasible,
        "active-set must report the model infeasible, not blame the iteration budget"
    );
}

#[test]
fn a_generous_iteration_budget_does_not_change_the_verdict() {
    let prob = sliver_qp(1e-5);
    let opts = QpOptions {
        max_iter: 100_000,
        ..QpOptions::default()
    };
    assert_eq!(active_set_with(&prob, &opts), QpStatus::PrimalInfeasible);
}

/// Deterministic generator (SplitMix64), so the sweep is reproducible
/// without a dev-dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform on [-1, 1).
    fn sym(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// The adversary's family: `P = QQᵀ + 0.1·I` (strictly convex), random `c`,
/// and a single sliver `a·x ≤ 2`, `a·x ≥ 2 + ε` on free variables.
fn random_sliver(n: usize, seed: u64, eps: f64) -> QpProblem {
    let mut rng = Rng(seed);
    let q: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..n).map(|_| rng.sym()).collect())
        .collect();
    let mut p_lower = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let mut v: f64 = (0..n).map(|k| q[i][k] * q[j][k]).sum();
            if i == j {
                v += 0.1;
            }
            p_lower.push(Triplet::new(i, j, v));
        }
    }
    let c: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
    let a: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
    let mut g = Vec::new();
    for (j, &aj) in a.iter().enumerate() {
        g.push(Triplet::new(0, j, aj));
        g.push(Triplet::new(1, j, -aj));
    }
    QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g,
        h: vec![2.0, -(2.0 + eps)],
        lb: vec![],
        ub: vec![],
    }
}

/// Every gap the adversary swept, 1e-2 down to 1e-6, over `n = 3..7` and four
/// seeds each: 20/20 per gap, and the IPM must agree on every instance.
/// Before the fix every gap below ~1e-4 came back `IterationLimit`.
#[test]
fn random_slivers_above_tolerance_are_all_infeasible() {
    let mut misreports = Vec::new();
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] {
        for n in 3..=7 {
            for seed in 0..4u64 {
                let prob = random_sliver(n, 1000 * n as u64 + seed, eps);
                let ipm = solve_qp_ipm(&prob, &QpOptions::default(), backend).status;
                assert_eq!(
                    ipm,
                    QpStatus::PrimalInfeasible,
                    "IPM oracle eps={eps} n={n} seed={seed}"
                );
                let got = active_set(&prob);
                if got != QpStatus::PrimalInfeasible {
                    misreports.push((eps, n, seed, got));
                }
            }
        }
    }
    assert!(
        misreports.is_empty(),
        "active-set misreports: {misreports:?}"
    );
}

/// Below the solver's own tolerances the answer is *not* "infeasible", and
/// the fix must not start saying so. With `ε/2` per row inside the engine's
/// `feas_tol` (1e-9) the model is feasible to tolerance and both engines
/// report `Optimal`; between that and the infeasibility tolerance (1e-7,
/// relative) the certificate cannot be told from round-off and is not
/// claimed. A fix that called every converged phase-1 exit `Infeasible`
/// fails here.
#[test]
fn gaps_inside_the_tolerances_are_not_certified() {
    for eps in [1e-8, 3e-9] {
        assert_ne!(
            active_set(&sliver_qp(eps)),
            QpStatus::PrimalInfeasible,
            "eps={eps} is within the infeasibility tolerance"
        );
    }
    let prob = sliver_qp(1e-10);
    assert_eq!(active_set(&prob), QpStatus::Optimal);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::Optimal,
        "IPM oracle"
    );
}

/// The feasible mirror (`2 − ε ≤ x₀ + 2x₁ ≤ 2`) solves on both engines.
#[test]
fn the_feasible_sliver_solves() {
    let prob = sliver_qp(-1e-5);
    assert_eq!(active_set(&prob), QpStatus::Optimal);
    assert_eq!(
        solve_qp_ipm(&prob, &QpOptions::default(), backend).status,
        QpStatus::Optimal
    );
}

/// A budget that genuinely runs out is still reported as one: the
/// certificate is only consulted when the phase-1 *converged* infeasible.
#[test]
fn an_exhausted_budget_on_a_feasible_qp_is_still_an_iteration_limit() {
    let n = 12;
    let mut prob = random_sliver(n, 7, -1.0);
    // Add a box that binds on most coordinates so the solve needs many
    // working-set changes.
    prob.lb = vec![0.5; n];
    prob.ub = vec![0.6; n];
    prob.c = vec![-10.0; n];
    prob.h = vec![1e3, 1e3];
    let ok = active_set(&prob);
    assert_eq!(
        ok,
        QpStatus::Optimal,
        "control: the model solves with the default budget"
    );
    let opts = QpOptions {
        max_iter: 1,
        ..QpOptions::default()
    };
    assert_eq!(active_set_with(&prob, &opts), QpStatus::IterationLimit);
}
