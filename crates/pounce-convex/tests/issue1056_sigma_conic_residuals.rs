//! gh #1056: the `σ` cost normalization must judge a conic answer with the
//! cone's own residuals.
//!
//! `min cᵀx s.t. ‖x‖₂ ≤ 2`, as the SOC row `(2, x₀, x₁) ∈ Q³`, has the closed
//! form `x* = −2c/‖c‖`, `f* = −2‖c‖`. With `c = (g, g)` and `g` past about
//! `4.5e7` the HSDE driver divides the objective by `σ ≈ g` and solves an
//! `O(1)` problem — in 10 iterations, at every scale measured. Its verdict was
//! then asked of `QpSolution::kkt_residuals`, which reads each SOC row as an
//! orthant row (pounce#209): a converged block has `zᵢsᵢ ≠ 0` row by row, and
//! with `z` carrying the full cost scale that "complementarity" is `O(g)`. So
//! every normalized optimum was rejected and the answer fell to the
//! un-normalized embedding — the solve `σ` exists to avoid — which returns
//! `NumericalFailure` from `g = 1e15` and `IterationLimit` at an infeasible
//! point across `1e16 … 1e19`.
//!
//! The isotropic case leaves the variable box out so the cone is the only
//! inequality block: the box would add an orthant block whose rows read
//! correctly either way and say nothing about that defect. The anisotropic
//! case keeps it, because there it is half of what the guards reject.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// `min c₀x₀ + c₁x₁ s.t. (2, x₀, x₁) ∈ Q³`.
fn ball(c: [f64; 2]) -> QpProblem {
    ball_in_box(c, Vec::new(), Vec::new())
}

/// [`ball`] inside the box `[−10, 10]²`, as `qcqp_ball.nl` declares it. The
/// box is inactive, but its multipliers are part of what the `σ` guards
/// read: on the anisotropic cost they sit at `~2e11` on `x₁`, which is
/// `mu`-level at `σ`'s scale and enormous at `c₁`'s.
fn boxed_ball(c: [f64; 2]) -> QpProblem {
    ball_in_box(c, vec![-10.0; 2], vec![10.0; 2])
}

fn ball_in_box(c: [f64; 2], lb: Vec<f64>, ub: Vec<f64>) -> QpProblem {
    QpProblem {
        n: 2,
        p_lower: Vec::new(),
        c: c.to_vec(),
        a: Vec::new(),
        b: Vec::new(),
        // s = h − Gx = (2, x₀, x₁).
        g: vec![Triplet::new(1, 0, -1.0), Triplet::new(2, 1, -1.0)],
        h: vec![2.0, 0.0, 0.0],
        lb,
        ub,
    }
}

#[test]
fn a_cost_scaled_ball_solves_at_every_scale() {
    for g in [1e8, 1e12, 1e15, 1e16, 1e18, 1e19, 1e20, 1e22, 1e24, 1e100] {
        let c = [g, g];
        let sol = solve_socp_ipm(
            &ball(c),
            &[ConeSpec::SecondOrder(3)],
            &QpOptions::default(),
            backend,
        );
        assert_eq!(sol.status, QpStatus::Optimal, "g = {g:e}: {sol:?}");
        let cn = (c[0] * c[0] + c[1] * c[1]).sqrt();
        let f_star = -2.0 * cn;
        assert!(
            ((sol.obj - f_star) / f_star).abs() < 1e-8,
            "g = {g:e}: obj {} vs {f_star}",
            sol.obj
        );
        for (i, (xi, ci)) in sol.x.iter().zip(c).enumerate() {
            let xs = -2.0 * ci / cn;
            assert!((xi - xs).abs() < 1e-6, "g = {g:e}: x[{i}] = {xi} vs {xs}");
        }
        // Feasible, not merely close in objective: at `1e18` the old answer
        // read 22% *below* `f*`, which no point of the ball can.
        let r = sol.x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(r <= 2.0 + 1e-6, "g = {g:e}: ‖x‖ = {r}");
        // And the verdict agrees with the residuals the CLI reports.
        let res = sol.kkt_residuals_conic(&ball(c), &[ConeSpec::SecondOrder(3)]);
        assert!(
            res.primal_infeasibility < 1e-6,
            "g = {g:e}: primal residual {:e}",
            res.primal_infeasibility
        );
    }
}

/// The anisotropic half (gh #1056 follow-up): cost `(g, 1)`. Here the
/// normalized solve is right — `x` within `1e-10` of `x*` — but row `x₁`
/// lives 22 orders below `σ`, so its stationarity and its multipliers cannot
/// resolve and the per-row `σ` guards reject it; the un-normalized re-solve
/// then fails outright. With no third driver for a cone, the cascade used to
/// return that failure. It now returns the closest claimed optimum, and the
/// forward-error arm — reading each SOC block's stiffness as `W⁻²` — decides
/// its status: on `(1e22, 1)` it reads `‖Δ‖ = 8.4e-11`, which is `x₀`'s actual
/// error.
#[test]
fn an_anisotropic_cost_ball_solves_at_every_scale() {
    for g in [1e12, 1e16, 1e20, 1e22, 1e24, 1e100] {
        let c = [g, 1.0];
        let sol = solve_socp_ipm(
            &boxed_ball(c),
            &[ConeSpec::SecondOrder(3)],
            &QpOptions::default(),
            backend,
        );
        assert_eq!(sol.status, QpStatus::Optimal, "g = {g:e}: {sol:?}");
        let cn = (c[0] * c[0] + c[1] * c[1]).sqrt();
        let f_star = -2.0 * cn;
        assert!(
            ((sol.obj - f_star) / f_star).abs() < 1e-8,
            "g = {g:e}: obj {} vs {f_star}",
            sol.obj
        );
        for (i, (xi, ci)) in sol.x.iter().zip(c).enumerate() {
            let xs = -2.0 * ci / cn;
            assert!((xi - xs).abs() < 1e-6, "g = {g:e}: x[{i}] = {xi} vs {xs}");
        }
        let r = sol.x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(r <= 2.0 + 1e-6, "g = {g:e}: ‖x‖ = {r}");
    }
}
