//! gh#1143 — an infeasible LP with a near-constant row must not be reported
//! unbounded by the IPM.
//!
//! ```text
//! min  −3·x₀ + 2·x₁
//! s.t. ε·x₀ ≤ −1      (⇒ x₀ ≤ −1/ε)
//!        −x₀ ≤ 0       (⇒ x₀ ≥ 0)
//!        −x₁ ≤ 0
//! ```
//!
//! Rows 0 and 1 contradict each other for every `ε > 0`, so the model is
//! primal infeasible; and it is *not* dual infeasible either — a recession
//! direction needs `ε·d₀ ≤ 0, −d₀ ≤ 0`, i.e. `d₀ = 0`, leaving `cᵀd = 2·d₁ ≥ 0`.
//! An unboundedness verdict is simply wrong.
//!
//! At `ε = 1e-14` the HSDE driver read the iterate `d ≈ (297, 420)` at
//! iteration 2 as a recession ray: its recession slack was absolute,
//! `(Gd)ᵢ ≤ 1e-7·‖d‖`, and `(Gd)₀ = 3e-12` passes that while violating row 0
//! by `3e-12 / 1e-14 = 300` of its own units. The objective-free twin that
//! relabels a false `DualInfeasible` on an infeasible model runs unscaled, and
//! on this row ends `NumericalFailure`, so the wrong verdict survived.
//!
//! The fix scales each row's recession slack by `min(1, ‖Gᵢ‖∞)` — what the
//! active-set engine's `ray_certifies_unbounded` already does (gh#1034).
//!
//! Both branches of the rule are pinned: a tiny row the ray *violates*
//! (rejected — the infeasible model), and a tiny row the ray *satisfies*
//! (`−ε·x₀ ≤ 1`, the same `c`), which is genuinely unbounded and must keep its
//! `DualInfeasible` verdict.
//!
//! The conic entry point is covered too, with an all-orthant cone and with the
//! same rows plus an unrelated SOC block (`(1, x₂) ∈ SOC`, i.e. `|x₂| ≤ 1`).
//! The first version of the fix gated the per-row test on the whole cone being
//! an orthant, so the mixed model kept its false verdict (review of gh#1144).

use pounce_convex::{
    ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm, solve_socp_ipm,
};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn prob(eps: f64, h0: f64, c: [f64; 2]) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: vec![],
        c: c.to_vec(),
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, eps),
            Triplet::new(1, 0, -1.0),
            Triplet::new(2, 1, -1.0),
        ],
        h: vec![h0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    }
}

/// The reported case and its neighbours from the issue table: every one is
/// primal infeasible, on the default options and on the direct driver.
#[test]
fn near_constant_infeasible_row_is_primal_infeasible() {
    let direct = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };
    for opts in [QpOptions::default(), direct] {
        for eps in [1e-14, 1e-12, 1e-10, 1e-6] {
            for c in [[-3.0, 2.0], [1.0, 1.0], [1.0, 0.0]] {
                let sol = solve_qp_ipm(&prob(eps, -1.0, c), &opts, backend);
                assert_eq!(
                    sol.status,
                    QpStatus::PrimalInfeasible,
                    "ε={eps:e}, c={c:?}, use_hsde={}: {:?} after {} iterations",
                    opts.use_hsde,
                    sol.status,
                    sol.iters,
                );
            }
        }
    }
}

/// The other branch: a tiny row the ray satisfies with room to spare.
/// `−ε·x₀ ≤ 1` and `x₀ ≥ 0` leave `x₀ → +∞` feasible, and `−3·x₀` falls without
/// bound along it, so this LP *is* unbounded. The per-row slack must not
/// suppress that verdict.
#[test]
fn near_constant_satisfied_row_stays_unbounded() {
    let direct = QpOptions {
        use_hsde: false,
        ..QpOptions::default()
    };
    for opts in [QpOptions::default(), direct] {
        for eps in [1e-14, 1e-10, 1e-6] {
            let sol = solve_qp_ipm(&prob(-eps, 1.0, [-3.0, 2.0]), &opts, backend);
            assert_eq!(
                sol.status,
                QpStatus::DualInfeasible,
                "ε={eps:e}, use_hsde={}: {:?} after {} iterations",
                opts.use_hsde,
                sol.status,
                sol.iters,
            );
        }
    }
}

/// The issue's rows plus `(1, x₂) ∈ SOC` on a new, cost-free variable.
fn with_soc(p: QpProblem) -> QpProblem {
    let mut m = p;
    m.n = 3;
    m.c.push(0.0);
    m.g.push(Triplet::new(4, 2, -1.0));
    m.h.extend([1.0, 0.0]);
    m
}

const MIXED: [ConeSpec; 2] = [ConeSpec::Nonneg(3), ConeSpec::SecondOrder(2)];

/// Through `solve_socp_ipm`, all-orthant and mixed: never `DualInfeasible`.
///
/// At `ε = 1e-14` both end `NumericalFailure`, not `PrimalInfeasible`: the
/// conic entry has no equilibrated retry (that lives on the LP/QP entry,
/// gh#293), and that retry is what reaches the verdict through `solve_qp_ipm`.
/// Recorded in `dev-notes/known-limitations.md`; pinned here only as "not the
/// wrong verdict" so a future fix can tighten it.
#[test]
fn conic_entry_near_constant_infeasible_row_is_not_unbounded() {
    for eps in [1e-14, 1e-10, 1e-6] {
        let p = prob(eps, -1.0, [-3.0, 2.0]);
        let cases = [
            (
                "orthant",
                solve_socp_ipm(&p, &[ConeSpec::Nonneg(3)], &QpOptions::default(), backend),
            ),
            (
                "mixed",
                solve_socp_ipm(&with_soc(p), &MIXED, &QpOptions::default(), backend),
            ),
        ];
        for (name, sol) in cases {
            if eps == 1e-14 {
                assert_ne!(sol.status, QpStatus::DualInfeasible, "{name}, ε={eps:e}");
            } else {
                assert_eq!(
                    sol.status,
                    QpStatus::PrimalInfeasible,
                    "{name}, ε={eps:e}: {:?} after {} iterations",
                    sol.status,
                    sol.iters,
                );
            }
        }
    }
}

/// Other branch, mixed cone: the tiny row the ray satisfies keeps the
/// genuine unbounded verdict.
#[test]
fn conic_entry_near_constant_satisfied_row_stays_unbounded() {
    for eps in [1e-14, 1e-10, 1e-6] {
        let p = with_soc(prob(-eps, 1.0, [-3.0, 2.0]));
        let sol = solve_socp_ipm(&p, &MIXED, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::DualInfeasible, "ε={eps:e}");
    }
}
