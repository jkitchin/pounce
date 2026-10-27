//! gh#1085, power-cone half: an `Optimal` power-cone solve must come with an
//! accurate minimizer, not only an accurate objective.
//!
//! The adversary logged this gap twice without filing it (2026-08-02
//! quad-over-lin, 2026-08-04 CES utility): the objective agreed with the
//! closed form to `~6e-8` while `x` was `1.5e-4` and `6.6e-3` from it. The
//! cause is the exp cone's: the IPM's dense 3×3 scaling block loses its small
//! eigenvalue near the boundary. The fix is the same active-set polish
//! (`crate::nonsym_polish`), holding an active power block on
//! `ψ(s) = s₁^α s₂^{1−α} − σ·s₀ = 0`, `σ = sign(s₀)`.
//!
//! Both models have closed forms. The sign-flipped quad-over-lin puts every
//! active `s₀` below zero, so the polish runs with `σ = −1` there and `σ = +1`
//! in the other two.

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, cones: &[ConeSpec], tol: f64) -> pounce_convex::QpSolution {
    let opts = QpOptions {
        tol,
        ..QpOptions::default()
    };
    solve_socp_ipm(prob, cones, &opts, backend)
}

fn inf_err(x: &[f64], want: &[f64]) -> f64 {
    x.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// CES utility: `max Σ wᵢ xᵢ^½ s.t. pᵀx = 100`, with `tᵢ ≤ xᵢ^½` as
/// `(tᵢ, xᵢ, 1) ∈ K_½`. Closed form `xᵢ* = C·(wᵢ/pᵢ)²`,
/// `C = 100 / Σⱼ wⱼ²/pⱼ`. Before the polish: `x` off by `6.6e-3`.
#[test]
fn ces_utility_demand_is_accurate() {
    let w = [1.0, 2.0, 3.0];
    let p = [1.0, 1.0, 2.0];
    let n = 3;
    let mut c = vec![0.0; 2 * n];
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for i in 0..n {
        c[n + i] = -w[i];
        let r = 3 * i;
        g.push(Triplet::new(r, n + i, -1.0)); // s₀ = tᵢ
        g.push(Triplet::new(r + 1, i, -1.0)); // s₁ = xᵢ
        h.extend_from_slice(&[0.0, 0.0, 1.0]); // s₂ = 1
    }
    let a = (0..n).map(|i| Triplet::new(0, i, p[i])).collect();
    let prob = QpProblem {
        n: 2 * n,
        p_lower: vec![],
        c,
        a,
        b: vec![100.0],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let cc = 100.0 / (0..n).map(|j| w[j] * w[j] / p[j]).sum::<f64>();
    let want: Vec<f64> = (0..n).map(|i| cc * (w[i] / p[i]).powi(2)).collect();
    for tol in [1e-8, 1e-10] {
        let sol = solve(&prob, &[ConeSpec::Power(0.5); 3], tol);
        assert_eq!(sol.status, QpStatus::Optimal, "tol={tol:e}");
        let err = inf_err(&sol.x[..n], &want);
        assert!(
            err < 1e-11,
            "tol={tol:e}: |x − x*|∞ = {err:e}, x = {:?}",
            &sol.x[..n]
        );
    }
}

/// Quad-over-lin: `min Σ xᵢ²/yᵢ s.t. Σ xᵢ = C`, `y = (1, 2, 3)` fixed in `h`,
/// with `tᵢ ≥ xᵢ²/yᵢ` as `(xᵢ, yᵢ, tᵢ) ∈ K_½`. Closed form
/// `xᵢ* = C·yᵢ/Σy`. `C = 12` gives `s₀ > 0` everywhere (`σ = +1`); `C = −12`
/// gives `s₀ < 0` everywhere (`σ = −1`). Before the polish: `x` off by
/// `1.5e-4`.
#[test]
fn quad_over_lin_is_accurate_with_either_sign() {
    let y = [1.0, 2.0, 3.0];
    let n = 3;
    for total in [12.0, -12.0] {
        let mut c = vec![0.0; 2 * n];
        let (mut g, mut h) = (Vec::new(), Vec::new());
        for i in 0..n {
            c[n + i] = 1.0;
            let r = 3 * i;
            g.push(Triplet::new(r, i, -1.0)); // s₀ = xᵢ
            g.push(Triplet::new(r + 2, n + i, -1.0)); // s₂ = tᵢ
            h.extend_from_slice(&[0.0, y[i], 0.0]); // s₁ = yᵢ
        }
        let prob = QpProblem {
            n: 2 * n,
            p_lower: vec![],
            c,
            a: (0..n).map(|i| Triplet::new(0, i, 1.0)).collect(),
            b: vec![total],
            g,
            h,
            lb: vec![],
            ub: vec![],
        };
        let want: Vec<f64> = y.iter().map(|yi| total * yi / 6.0).collect();
        let sol = solve(&prob, &[ConeSpec::Power(0.5); 3], 1e-8);
        assert_eq!(sol.status, QpStatus::Optimal, "C={total}");
        let err = inf_err(&sol.x[..n], &want);
        assert!(err < 1e-12, "C={total}: |x − x*|∞ = {err:e}");
        let obj_want = total * total / 6.0;
        assert!(
            (sol.obj - obj_want).abs() < 1e-12 * obj_want,
            "C={total}: obj {} vs {obj_want}",
            sol.obj
        );
    }
}
