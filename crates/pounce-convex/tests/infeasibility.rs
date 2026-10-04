//! Verified infeasibility / unboundedness detection (the HSDE benefit:
//! clean status instead of exhausting the iteration budget).
//!
//! Each declared status is backed by a checked certificate, so these
//! tests also implicitly confirm there are no false positives — the
//! feasible/optimal problems in the rest of the suite must still report
//! `Optimal`, and a couple of those are re-checked here for contrast.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem) -> pounce_convex::QpSolution {
    solve_qp_ipm(prob, &QpOptions::default(), backend)
}

/// Primal-infeasible: contradictory equalities x0 = 1 and x0 = 2.
/// (min x0² subject to both.) No x satisfies the constraints.
#[test]
fn primal_infeasible_contradictory_equalities() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 0, 1.0)],
        b: vec![1.0, 2.0],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "expected primal infeasible, got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Primal-infeasible via inequalities: x0 ≤ 0 and x0 ≥ 1 (written
/// −x0 ≤ −1). Empty feasible set.
#[test]
fn primal_infeasible_contradictory_inequalities() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 2.0)],
        c: vec![0.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),  // x0 ≤ 0
            Triplet::new(1, 0, -1.0), // −x0 ≤ −1  (x0 ≥ 1)
        ],
        h: vec![0.0, -1.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Unbounded LP: min −x0 with x0 ≥ 0 (no upper bound). Objective → −∞
/// along the recession direction d = (1).
#[test]
fn dual_infeasible_unbounded_lp() {
    let prob = QpProblem {
        n: 1,
        p_lower: vec![], // LP (P = 0)
        c: vec![-1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0)], // −x0 ≤ 0  (x0 ≥ 0)
        h: vec![0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "expected unbounded (dual infeasible), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Unbounded QP: a singular Hessian with a recession direction. min x1²
/// − x0 with x0 free, x1 free. The x0 direction has Pd = 0 and cᵀd < 0,
/// so the objective is unbounded below.
#[test]
fn dual_infeasible_unbounded_qp_singular_hessian() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(1, 1, 2.0)], // only x1 is in P
        c: vec![-1.0, 0.0],                     // −x0
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "got {:?} after {} iters",
        sol.status,
        sol.iters
    );
}

/// Contrast: a feasible, bounded QP must still report Optimal — the
/// detector must not false-positive. min (x0−1)² + (x1−1)², 0 ≤ x ≤ 5.
#[test]
fn feasible_bounded_still_optimal() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(3, 1, -1.0),
        ],
        h: vec![5.0, 5.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6);
    assert!((sol.x[1] - 1.0).abs() < 1e-6);
}

/// gh #293 — a mixed-scale Hessian must NOT be falsely certified unbounded.
/// `min ½(1e6·x0² + 1e-12·x1²) − x1  s.t.  x ≥ 0` is *bounded*: the unique
/// optimum is `x1* = 1e12`, `f* = −5e11`. The descent ray `x1` has genuine
/// (if tiny) curvature `1e-12 > 0`, so it is not a recession ray. Before #293
/// the `‖Pd‖ ≤ rtol·‖d‖·max|P|` test read the `1e-12` curvature as null
/// relative to the `1e6` block and returned a wrong `DualInfeasible`.
#[test]
fn mixed_scale_hessian_is_bounded_not_dual_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e6), Triplet::new(1, 1, 1e-12)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_ne!(
        sol.status,
        QpStatus::DualInfeasible,
        "bounded problem (f* = -5e11) must never get an unboundedness \
         certificate; got a wrong DualInfeasible after {} iters",
        sol.iters
    );
    // The certificate fix also lets it converge to the true optimum.
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal (x1* = 1e12, f* = -5e11), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
    assert!(
        (sol.obj - (-5e11)).abs() <= 1e-3 * 5e11,
        "obj = {} should be ≈ -5e11",
        sol.obj
    );
}

/// gh #293 (symptom 2) — a *uniformly* tiny Hessian must converge, not exhaust
/// the iteration budget. `min ½·1e-12·(x0² + x1²) − x1  s.t.  x ≥ 0` is bounded
/// with the same optimum as above (`x1* = 1e12`, `f* = −5e11`). #290 stopped
/// this from being falsely certified unbounded, but the default HSDE driver
/// then merely ran out of iterations (obj ≈ −4.95e11 at `IterationLimit`)
/// because its per-cone NT scaling never sees the 12-orders-below-O(1)
/// curvature. The fix Ruiz-equilibrates and retries when HSDE hits the cap, so
/// the solve now reports `Optimal` at the true optimum.
#[test]
fn uniform_tiny_hessian_converges_not_iteration_limit() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e-12), Triplet::new(1, 1, 1e-12)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal (x1* = 1e12, f* = -5e11), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
    assert!(
        (sol.obj - (-5e11)).abs() <= 1e-3 * 5e11,
        "obj = {} should be ≈ -5e11",
        sol.obj
    );
}

/// gh #293 (symptom 2, constrained) — the tiny-curvature pathology also
/// surfaces as `OptimalInaccurate` (not `IterationLimit`) when a constraint
/// binds near the far-off optimum: HSDE returns a usable-but-loose iterate at
/// the cap instead of running fully dry. The equilibrated retry must rescue
/// this manifestation too, so the fix is keyed on the *regime* rather than a
/// single status symbol. `min ½·1e-12·(x0²+x1²) − x1  s.t.  x1 ≤ 1e6, x ≥ 0`
/// has optimum `x1* = 1e6`, `f* ≈ −1e6`.
#[test]
fn tiny_hessian_with_binding_inequality_converges_cleanly() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e-12), Triplet::new(1, 1, 1e-12)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 1, 1.0)],
        h: vec![1e6],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected clean Optimal (x1* = 1e6), got {:?} after {} iters",
        sol.status,
        sol.iters
    );
    assert!(
        (sol.obj - (-1e6)).abs() <= 1e-3 * 1e6,
        "obj = {} should be ≈ -1e6",
        sol.obj
    );
}

/// gh #293 (machine-epsilon tail) — a spurious unboundedness certificate on a
/// QP with genuine (if tiny) curvature must be refuted. `min ½·1e-20·(x0²+x1²)
/// − x1  s.t.  x ≥ 0` is bounded (`x1* = 1e20`, `f* = −5e19`), but the raw HSDE
/// solve reads the descent ray as a recession at `P ≈ 1e-20` (the recession
/// curvature floor) and certifies `DualInfeasible`. The direct-driver reverify
/// on the equilibrated problem exposes the true curvature and returns a clean
/// finite optimum, which overrides the bogus certificate.
#[test]
fn extreme_tiny_hessian_not_falsely_unbounded() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e-20), Triplet::new(1, 1, 1e-20)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    let sol = solve(&prob);
    assert_ne!(
        sol.status,
        QpStatus::DualInfeasible,
        "bounded problem (f* = -5e19) must not be certified unbounded; got \
         DualInfeasible after {} iters",
        sol.iters
    );
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "expected Optimal, got {:?}",
        sol.status
    );
    assert!(
        (sol.obj - (-5e19)).abs() <= 1e-3 * 5e19,
        "obj = {} should be ≈ -5e19",
        sol.obj
    );
}

/// gh #293 naive-caller guardrail — when no driver can converge a tiny-curvature
/// problem at the default budget (here a uniformly tiny Hessian coupled through
/// an equality constraint, which stays `IterationLimit`), the honest status must
/// carry a scaling diagnostic that names tiny curvature as the cause. A
/// well-scaled `Optimal` must carry none.
#[test]
fn tiny_curvature_iteration_limit_emits_scaling_warning() {
    let n = 10;
    let mut p_lower = Vec::new();
    let mut c = Vec::new();
    for i in 0..n {
        p_lower.push(Triplet::new(i, i, 1e-13));
        c.push(-1.0 - (i % 7) as f64);
    }
    let prob = QpProblem {
        n,
        p_lower,
        c,
        a: (0..n).map(|j| Triplet::new(0, j, 1.0)).collect(),
        b: vec![1e13],
        g: vec![],
        h: vec![],
        lb: vec![0.0; n],
        ub: vec![f64::INFINITY; n],
    };
    let sol = solve(&prob);
    // This case is genuinely unconvergeable at the default budget; the point is
    // the diagnostic, not the status. Guard the premise, then the diagnostic.
    if sol.status == QpStatus::Optimal {
        // If a future improvement converges it, the guardrail is moot here.
        return;
    }
    let warn = sol.scaling_diagnostic(&prob);
    assert!(
        warn.is_some(),
        "tiny-curvature {:?} must carry a scaling diagnostic",
        sol.status
    );
    assert!(warn.unwrap().contains("scaling warning"));

    // A well-scaled optimal solve carries no diagnostic.
    let good = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![10.0, 10.0],
    };
    let gsol = solve(&good);
    assert_eq!(gsol.status, QpStatus::Optimal);
    assert!(
        gsol.scaling_diagnostic(&good).is_none(),
        "well-scaled Optimal needs no warning"
    );
}

// --- Status / edge-case honesty (PR70 item C) -----------------------------
//
// A solver that stops early for *any* reason must say so. The danger these
// guard against is a confident `Optimal` (or a spurious infeasible/unbounded)
// on a problem the solver did not actually finish or that is degenerate.

/// Iteration-limit honesty: a real, feasible, bounded QP that needs several
/// IPM iterations must report `IterationLimit` — never a premature `Optimal`,
/// and never a false infeasible/unbounded — when starved of iterations.
#[test]
fn iteration_limit_reported_not_optimal() {
    // The same well-posed box QP as `feasible_bounded_still_optimal`, which
    // converges in several iterations at the default cap. With max_iter = 1 it
    // cannot have converged, so the only honest status is IterationLimit.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-2.0, -2.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 0, -1.0),
            Triplet::new(3, 1, -1.0),
        ],
        h: vec![5.0, 5.0, 0.0, 0.0],
        lb: vec![],
        ub: vec![],
    };
    let opts = QpOptions {
        max_iter: 1,
        ..QpOptions::default()
    };
    let sol = solve_qp_ipm(&prob, &opts, backend);
    assert_eq!(
        sol.status,
        QpStatus::IterationLimit,
        "1-iteration solve must report IterationLimit, got {:?}",
        sol.status
    );
    assert_ne!(
        sol.status,
        QpStatus::Optimal,
        "must not claim Optimal after a single iteration"
    );
}

/// Degenerate input — a variable fixed by equal bounds (lb == ub) — must
/// solve honestly to `Optimal` at the fixed value, not trip a spurious
/// infeasible/unbounded or numerical failure.
#[test]
fn fixed_variable_equal_bounds_optimal() {
    // min x0² + x1² − 6x0 − 6x1, x0 fixed to 1 (lb==ub==1), x1 ∈ [0, 10].
    // Unconstrained min is (3, 3); with x0 pinned the optimum is (1, 3).
    // obj = 1 + 9 − 6 − 18 = −14.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, -6.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![1.0, 0.0],
        ub: vec![1.0, 10.0],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - 3.0).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-14.0)).abs() < 1e-6, "obj={}", sol.obj);
}

/// Edge input — a fully unconstrained QP (no equalities, no inequalities, no
/// bounds) — must still solve to its stationary point and report `Optimal`.
#[test]
fn unconstrained_qp_optimal() {
    // min x0² + x1² − 6x0 + 4x1  ->  min at (3, −2), obj = 9 + 4 − 18 − 8 = −13.
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
        c: vec![-6.0, 4.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(sol.status, QpStatus::Optimal, "iters={}", sol.iters);
    assert!((sol.x[0] - 3.0).abs() < 1e-6, "x0={}", sol.x[0]);
    assert!((sol.x[1] - (-2.0)).abs() < 1e-6, "x1={}", sol.x[1]);
    assert!((sol.obj - (-13.0)).abs() < 1e-6, "obj={}", sol.obj);
}

// ---------------------------------------------------------------------------
// Primal- AND dual-infeasible at once: which verdict gets reported.
//
// These two certificates are not alternatives. `DualInfeasible` rests on a
// recession direction `d` with `Pd ≈ 0, Ad ≈ 0, −Gd ∈ K, cᵀd < 0`, which says
// the *dual* has no feasible point — and that is perfectly true of a problem
// whose primal is also empty, because the recession direction of an empty
// feasible set exists just the same. So a model can honestly earn both, and
// then the report is a choice rather than a measurement.
//
// The choice must be `PrimalInfeasible`: it is the actionable one (AMPL
// `solve_result_num=200`, "fix the model", against `300`/`DivergingIterates`),
// it is what pounce's own active-set engine returns on the same data, and it is
// what HiGHS and Gurobi return. Left to the iteration it was a race between two
// residual gates, and the recession gate tended to clear first.
// ---------------------------------------------------------------------------

/// A four-variable LP, found by a randomized sweep, on which this actually went
/// wrong. Hand-crafted two-row instances do *not* reproduce it: the failure is a
/// race between two residual gates, so it needs an instance whose dynamics let
/// the recession gate win, and this is one.
///
/// **Primal infeasible by inspection**, and the test asserts that structurally
/// rather than trusting the numbers: rows 0 and 1 are exact negatives, so they
/// read `w·x ≤ 1` and `w·x ≥ 3`. `scipy`/HiGHS agrees (`status = 2`).
///
/// **Also dual infeasible**: `Gd ≤ 0` forces `w·d = 0`, and within that
/// hyperplane there is a `d` with `row₂·d ≤ 0` and `cᵀd < 0`, which is a genuine
/// recession certificate. Both verdicts are true, so the driver has to *choose*
/// — and before the objective-free-twin check it chose `DualInfeasible`, i.e.
/// `solve_result_num=300` on a model with no feasible point. Measured on this
/// instance: the Farkas value held at `−1.72` with `z ∈ K*` while its residual
/// fell `1.9e-3 → 9.5e-5 → 4.7e-6 → 2.4e-7` toward an `8.6e-11` gate, and the
/// recession gate opened with three orders still to go.
#[test]
fn primal_and_dual_infeasible_reports_primal() {
    // Row 0 is `w`; row 1 is `−w`; row 2 is an extra, satisfiable on its own.
    let w = [
        -1.336972899917,
        -1.045019914384,
        1.450153058953,
        -0.540131207078,
    ];
    let extra = [
        -2.104466010086,
        -0.580699705489,
        1.5099831e-05,
        1.188830678537,
    ];
    let h = [1.0, -3.0, 1.858605832812668];

    // The contradiction, asserted from the data: `w·x ≤ h₀` and `−w·x ≤ h₁`
    // together give `h₁ ≤ −w·x` and `w·x ≤ h₀`, i.e. `−h₁ ≤ w·x ≤ h₀`, which is
    // empty exactly when `h₀ + h₁ < 0`.
    assert!(h[0] + h[1] < 0.0, "rows 0/1 must be contradictory");

    let mut g = Vec::new();
    for (j, &v) in w.iter().enumerate() {
        g.push(Triplet::new(0, j, v));
        g.push(Triplet::new(1, j, -v));
    }
    for (j, &v) in extra.iter().enumerate() {
        g.push(Triplet::new(2, j, v));
    }

    let prob = QpProblem {
        n: 4,
        p_lower: vec![],
        c: vec![
            0.666683325902,
            0.795299099602,
            -0.699388308324,
            -0.187589705319,
        ],
        a: vec![],
        b: vec![],
        g,
        h: h.to_vec(),
        lb: vec![-20.0; 4],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "an infeasible model must not be reported as unbounded (got {:?} after \
         {} iters)",
        sol.status,
        sol.iters
    );
}

/// The guard above must not cost a genuine unboundedness verdict. This LP is
/// **feasible** (`x = (0, 0)` satisfies `x₀ − x₁ ≤ 1`) and unbounded below along
/// `d = (1, 1)`: `Gd = 0`, `cᵀd = −2 < 0`. Its objective-free twin is feasible,
/// so the correction never fires and `DualInfeasible` stands.
///
/// `dual_infeasible_unbounded_lp` above covers the unconstrained-direction case;
/// this one keeps a row *active* in the recession cone, the configuration the
/// correction has to leave alone.
#[test]
fn feasible_unbounded_lp_keeps_dual_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![-1.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
        h: vec![1.0],
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "feasible-and-unbounded must still certify unbounded (iters={})",
        sol.iters
    );
}

/// gh#990 (remaining item l): the `tau`/`kappa` slot must describe the run
/// that produced the returned answer. On the gh #293 tail model the HSDE run's
/// `DualInfeasible` is overridden by the direct driver's `Optimal`, which has
/// no homogeneous scalars — the slot used to keep the discarded run's (`tau`
/// collapsing toward 0, an infeasibility signature beside an `Optimal`).
#[test]
fn a_direct_driver_answer_carries_no_hsde_scalars() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![Triplet::new(0, 0, 1e-20), Triplet::new(1, 1, 1e-20)],
        c: vec![0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0, 0.0],
        ub: vec![f64::INFINITY, f64::INFINITY],
    };
    pounce_convex::hsde_scalars::clear();
    let sol = solve(&prob);
    let scalars = pounce_convex::hsde_scalars::take();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(
        scalars, None,
        "the answer came from the direct driver; tau/kappa {scalars:?} belong \
         to a discarded HSDE run"
    );

    // A plain HSDE solve still reports its own scalars.
    let plain = QpProblem {
        n: 1,
        p_lower: vec![Triplet::new(0, 0, 1.0)],
        c: vec![-1.0],
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![0.0],
        ub: vec![f64::INFINITY],
    };
    pounce_convex::hsde_scalars::clear();
    let sol = solve(&plain);
    assert_eq!(sol.status, QpStatus::Optimal);
    let s = pounce_convex::hsde_scalars::take().expect("HSDE ran");
    assert!(s.tau > 0.0, "{s:?}");
}
