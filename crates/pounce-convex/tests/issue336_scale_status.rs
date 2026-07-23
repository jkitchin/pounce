//! gh #336 regression: the non-symmetric (exp/power) HSDE driver reported
//! `NumericalFailure` on *correct* answers under extreme-but-legitimate data
//! scaling. Its status test keyed off the raw (unnormalized) complementarity
//! gap `s·z`, whose absolute floor grows with the optimal cone magnitudes, so a
//! point that is primal-feasible, dual-feasible, and objective-correct could
//! never reach the absolute tolerance and was discarded as a failure. The
//! post-loop adjudication now scores the recovered point on the scale-relative
//! conic KKT residual, certifying `Optimal` when it is tight and
//! `OptimalInaccurate` (never a spurious `NumericalFailure`) on the accuracy
//! plateau — matching the symmetric SOC driver and ECOS/Clarabel's `*_inacc`.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, cones: &[ConeSpec]) -> pounce_convex::QpSolution {
    let opts = QpOptions::default();
    solve_socp_ipm(prob, cones, &opts, backend)
}

/// Append a 3-row exp block enforcing `t_col ≥ exp(off + sign · x[lin_col])`
/// (the constant `off` folded into the `s0` row's `h`). In the `solve_socp`
/// convention `s = (s0, s1, s2)` with `s1·exp(s0/s1) ≤ s2`, `s1 = 1`, `s2 =
/// x[t_col]`.
fn exp_block(
    lin_col: usize,
    sign: f64,
    off: f64,
    t_col: usize,
    g: &mut Vec<Triplet>,
    h: &mut Vec<f64>,
) {
    let r0 = h.len();
    g.push(Triplet::new(r0, lin_col, -sign));
    h.push(off); // s0 = off + sign·x[lin_col]
    h.push(1.0); // s1 = 1
    let r2 = h.len();
    g.push(Triplet::new(r2, t_col, -1.0));
    h.push(0.0);
}

/// Balanced GP for `min x + K/x`: `min t1 + t2` with `t1 ≥ e^u (=x)` and
/// `t2 ≥ e^{lnK − u} (=K/x)`, so both cone variables sit at `~sqrt(K)` at the
/// optimum. AM-GM optimum `x* = sqrt(K)`, `f* = 2 sqrt(K)`.
fn exp_gp(k: f64) -> QpProblem {
    let ln_k = k.ln();
    let mut g = Vec::new();
    let mut h = Vec::new();
    exp_block(0, 1.0, 0.0, 1, &mut g, &mut h); // t1 ≥ e^u
    exp_block(0, -1.0, ln_k, 2, &mut g, &mut h); // t2 ≥ e^{lnK − u}
    QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 1.0, 1.0],
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

/// Case A, K=1e12: was `NumericalFailure` on the correct answer (obj ≈ 2e6,
/// rel err ~4.9e-6). Must now be a usable solve (`OptimalInaccurate`, since the
/// normalized gap is a little above the default `tol`), never `NumericalFailure`.
#[test]
fn exp_extreme_gp_not_numerical_failure() {
    let k = 1e12_f64;
    let sol = solve(&exp_gp(k), &[ConeSpec::Exponential; 2]);
    let f_star = 2.0 * k.sqrt();
    assert!(
        (sol.obj - f_star).abs() / f_star < 1e-4,
        "objective {:.6e} must match f* = {:.6e}",
        sol.obj,
        f_star
    );
    assert_ne!(
        sol.status,
        QpStatus::NumericalFailure,
        "a correct answer must not be labelled NumericalFailure"
    );
    assert!(
        matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "expected Optimal/OptimalInaccurate, got {:?}",
        sol.status
    );
}

/// K-sweep: the objective stays correct throughout, and the status degrades
/// gracefully with scale — `Optimal` while the scale-relative certificate is
/// tight, then `OptimalInaccurate` on the plateau — but is *never*
/// `NumericalFailure` on a correct answer. Pins the well-scaled end at `Optimal`
/// (guarding against over-relaxing the promotion gate).
#[test]
fn exp_gp_scale_sweep_never_numerical_failure() {
    for kexp in [8, 10, 11, 12, 13, 14] {
        let k = 10f64.powi(kexp);
        let sol = solve(&exp_gp(k), &[ConeSpec::Exponential; 2]);
        let f_star = 2.0 * k.sqrt();
        let rel = (sol.obj - f_star).abs() / f_star;
        assert!(rel < 1e-4, "K=1e{kexp}: obj rel err {rel:.2e} too large");
        assert!(
            matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
            "K=1e{kexp}: expected Optimal/OptimalInaccurate, got {:?}",
            sol.status
        );
        if kexp <= 10 {
            assert_eq!(
                sol.status,
                QpStatus::Optimal,
                "K=1e{kexp}: a well-scaled GP must still certify Optimal"
            );
        }
    }
}

/// Case B: power cone `max x s.t. |x| ≤ (y·w)^{1/2}, y + w ≤ S`, S=2e8.
/// Optimum `x* = S/2 = 1e8`; optimal cone variables ~1e8. Was `NumericalFailure`
/// on the correct answer; must now be a usable solve.
#[test]
fn power_extreme_budget_not_numerical_failure() {
    let s_budget = 2e8_f64;
    // vars [x, y, w]: min −x s.t. (x, y, w) ∈ K_{1/2}, y + w ≤ S.
    let mut g = Vec::new();
    let mut h = Vec::new();
    g.push(Triplet::new(0, 0, -1.0)); // s0 = x
    h.push(0.0);
    g.push(Triplet::new(1, 1, -1.0)); // s1 = y
    h.push(0.0);
    g.push(Triplet::new(2, 2, -1.0)); // s2 = w
    h.push(0.0);
    let r = h.len(); // budget slack = S − y − w ≥ 0
    g.push(Triplet::new(r, 1, 1.0));
    g.push(Triplet::new(r, 2, 1.0));
    h.push(s_budget);
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![-1.0, 0.0, 0.0],
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Power(0.5), ConeSpec::Nonneg(1)]);
    let x_star = s_budget / 2.0;
    assert!(
        (sol.x[0] - x_star).abs() / x_star < 1e-3,
        "x {:.6e} must match x* = {:.6e}",
        sol.x[0],
        x_star
    );
    assert_ne!(sol.status, QpStatus::NumericalFailure);
    assert!(
        matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate),
        "expected Optimal/OptimalInaccurate, got {:?}",
        sol.status
    );
}
