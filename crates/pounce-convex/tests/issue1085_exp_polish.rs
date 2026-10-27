//! gh#1085: an exp-cone `Optimal` must come with an accurate **minimizer**, not
//! only an accurate objective.
//!
//! The adversary's model is Poisson regression on 8 observations — one exp
//! triple per row, Hessian condition number 18 at the optimum. The
//! non-symmetric HSDE driver reported `optimal` with `β` `1.8e-4` from the
//! minimizer (first-order residual `4.3e-4`), and tightening `tol` stalled at
//! `1.5e-5`: below `μ ~ 1e-8` its dense 3×3 scaling block loses its small
//! eigenvalue to rounding and the step collapses. The driver now polishes the
//! IPM point by Newton on the KKT equations of the active set it identified
//! (`crate::nonsym_polish`).
//!
//! Every check here is oracle-free: the first-order condition of the smooth
//! objective the cone model encodes, evaluated at the returned point. The
//! models are chosen to reach each branch of the active-set guess — active
//! and inactive exp blocks, active and inactive orthant rows, equality rows,
//! and variable bounds (which the driver expands into orthant rows).

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

/// The issue's design matrix (`numpy.random.default_rng(20261027)`, intercept
/// plus two uniform columns) and counts.
const A: [[f64; 3]; 8] = [
    [1.0, 0.9741862345158065, 0.5603341165130173],
    [1.0, 0.9245168008924922, 0.40681600696886977],
    [1.0, -0.9322645479804046, 0.9865698473028668],
    [1.0, -0.11373256193059822, 0.4966348859523133],
    [1.0, -0.23342757629048627, 0.31265845994016206],
    [1.0, -0.716228706101234, 1.673245867135296],
    [1.0, -0.5447643680340517, 1.7407514504344384],
    [1.0, -0.29085912102149547, 1.6333280025874208],
];
const Y: [f64; 8] = [1.0, 0.0, 3.0, 2.0, 5.0, 1.0, 0.0, 4.0];

/// Append the exp block `(sign·xᵀrow + shift, 1, x[t_col]) ∈ K_exp`, i.e.
/// `x[t_col] ≥ exp(sign·xᵀrow + shift)`, in the `s = h − Gx` convention.
fn exp_block(
    row: &[(usize, f64)],
    shift: f64,
    t_col: usize,
    g: &mut Vec<Triplet>,
    h: &mut Vec<f64>,
) {
    let r0 = h.len();
    for &(j, v) in row {
        g.push(Triplet::new(r0, j, -v));
    }
    h.push(shift);
    h.push(1.0); // all-zero row ⇒ s₁ = 1
    let r2 = h.len();
    g.push(Triplet::new(r2, t_col, -1.0));
    h.push(0.0);
}

/// Poisson regression `min Σ exp(aᵢᵀβ) − yᵢaᵢᵀβ` over `x = (β, t₁..t₈)`.
fn poisson() -> (QpProblem, Vec<ConeSpec>) {
    let (p, m) = (3, 8);
    let mut c = vec![0.0; p + m];
    for i in 0..m {
        for j in 0..p {
            c[j] -= A[i][j] * Y[i];
        }
        c[p + i] = 1.0;
    }
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for (i, ai) in A.iter().enumerate() {
        let row: Vec<(usize, f64)> = ai.iter().copied().enumerate().collect();
        exp_block(&row, 0.0, p + i, &mut g, &mut h);
    }
    let prob = QpProblem {
        n: p + m,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    (prob, vec![ConeSpec::Exponential; m])
}

/// `‖Aᵀ(exp(Aβ) − y)‖∞`, the gradient of the Poisson negative log-likelihood.
fn poisson_grad(beta: &[f64]) -> f64 {
    let mut g = [0.0; 3];
    for (ai, yi) in A.iter().zip(Y) {
        let r = (ai[0] * beta[0] + ai[1] * beta[1] + ai[2] * beta[2]).exp() - yi;
        for j in 0..3 {
            g[j] += ai[j] * r;
        }
    }
    g.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Newton's method on the smooth objective, gradient `9e-16`.
const BETA: [f64; 3] = [1.253714079362372, -0.9392093715413039, -0.8202143545134511];

fn beta_err(x: &[f64]) -> f64 {
    (0..3).map(|j| (x[j] - BETA[j]).abs()).fold(0.0, f64::max)
}

/// The reported model, at the default `tol` and tightened: `Optimal` with the
/// minimizer to near machine precision. Before the fix: `1.8e-4` at the
/// default (status `optimal`) and a `1.5e-5` floor however tight `tol` went.
#[test]
fn poisson_regression_minimizer_is_accurate_at_every_tol() {
    let (prob, cones) = poisson();
    for tol in [1e-8, 1e-10, 1e-12] {
        let sol = solve(&prob, &cones, tol);
        assert_eq!(sol.status, QpStatus::Optimal, "tol={tol:e}");
        let err = beta_err(&sol.x);
        let grad = poisson_grad(&sol.x);
        assert!(err < 1e-12, "tol={tol:e}: |β − β*|∞ = {err:e}");
        assert!(grad < 1e-12, "tol={tol:e}: |∇ℓ(β)|∞ = {grad:e}");
    }
}

/// The same model with slack constraints the polish must classify as
/// inactive: an exp block `exp(β₀ − 100) ≤ 1`, an orthant row `β₁ ≤ 50`, and
/// a variable box `−10 ≤ β₂ ≤ 10` (expanded into orthant rows by the driver).
/// None binds, so the minimizer is unchanged and their multipliers vanish.
#[test]
fn inactive_exp_block_orthant_row_and_bounds_leave_the_minimizer() {
    let (mut prob, mut cones) = poisson();
    let n = prob.n;
    // Slack variable for the extra exp block: x[n] = 1 by bounds would bind,
    // so pin it through h instead — the block reads (β₀ − 100, 1, 1).
    let r0 = prob.h.len();
    prob.g.push(Triplet::new(r0, 0, -1.0));
    prob.h.extend_from_slice(&[-100.0, 1.0, 1.0]);
    cones.push(ConeSpec::Exponential);
    // β₁ ≤ 50.
    let r = prob.h.len();
    prob.g.push(Triplet::new(r, 1, 1.0));
    prob.h.push(50.0);
    cones.push(ConeSpec::Nonneg(1));
    prob.lb = vec![f64::NEG_INFINITY; n];
    prob.ub = vec![f64::INFINITY; n];
    prob.lb[2] = -10.0;
    prob.ub[2] = 10.0;

    let sol = solve(&prob, &cones, 1e-8);
    assert_eq!(sol.status, QpStatus::Optimal);
    let err = beta_err(&sol.x);
    assert!(err < 1e-12, "|β − β*|∞ = {err:e}");
    let z_slack = sol.z[r0..r0 + 4].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        z_slack < 1e-12,
        "inactive multipliers {:?}",
        &sol.z[r0..r0 + 4]
    );
    assert!(sol.z_lb[2].abs() < 1e-12 && sol.z_ub[2].abs() < 1e-12);
}

/// Log-sum-exp with an **active** orthant row: `min t − wᵀMx` subject to
/// `Σᵢ exp(mᵢᵀx + bᵢ − t) ≤ 1`, written with `vᵢ ≥ exp(mᵢᵀx + bᵢ − t)` and the
/// orthant row `Σ vᵢ ≤ 1`, which binds at the optimum. Stationarity of the
/// smooth objective `LSE(Mx + b) − wᵀMx` is `Mᵀ(softmax(Mx + b) − w) = 0`.
#[test]
fn log_sum_exp_with_an_active_orthant_row() {
    let m_rows: [[f64; 2]; 5] = [
        [1.0, 0.3],
        [-0.7, 1.2],
        [0.4, -1.1],
        [-1.3, -0.2],
        [0.2, 0.9],
    ];
    let b = [0.1, -0.4, 0.3, 0.0, -0.2];
    let w = [0.3, 0.2, 0.15, 0.25, 0.1];
    let (p, m) = (2, 5);
    let (it, iv) = (p, p + 1); // t, then v₁..v₅
    let n = p + 1 + m;
    let mut c = vec![0.0; n];
    for i in 0..m {
        for j in 0..p {
            c[j] -= w[i] * m_rows[i][j];
        }
    }
    c[it] = 1.0;
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for i in 0..m {
        let row = [(0, m_rows[i][0]), (1, m_rows[i][1]), (it, -1.0)];
        exp_block(&row, b[i], iv + i, &mut g, &mut h);
    }
    let r = h.len();
    for i in 0..m {
        g.push(Triplet::new(r, iv + i, 1.0));
    }
    h.push(1.0);
    let mut cones = vec![ConeSpec::Exponential; m];
    cones.push(ConeSpec::Nonneg(1));
    let prob = QpProblem {
        n,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &cones, 1e-8);
    assert_eq!(sol.status, QpStatus::Optimal);
    let u: Vec<f64> = (0..m)
        .map(|i| m_rows[i][0] * sol.x[0] + m_rows[i][1] * sol.x[1] + b[i])
        .collect();
    let mx = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ez: Vec<f64> = u.iter().map(|v| (v - mx).exp()).collect();
    let sum: f64 = ez.iter().sum();
    let mut grad = [0.0; 2];
    for i in 0..m {
        for j in 0..p {
            grad[j] += m_rows[i][j] * (ez[i] / sum - w[i]);
        }
    }
    let gn = grad[0].abs().max(grad[1].abs());
    assert!(gn < 1e-12, "|∇f|∞ = {gn:e}");
    // The orthant row binds, with a positive multiplier.
    assert!(sol.z[r] > 0.5, "Σv ≤ 1 multiplier {}", sol.z[r]);
}

/// Maximum entropy under **equality** rows: `max −Σ xᵢ log xᵢ` subject to
/// `Σ xᵢ = 1`, `fᵀx = 0.3`, via `(tᵢ, xᵢ, 1) ∈ K_exp` (`tᵢ ≤ −xᵢ log xᵢ`).
/// At the optimum `log xᵢ` is affine in `fᵢ`, so the slope
/// `(log xᵢ − log x₀)/(fᵢ − f₀)` is the same for every `i`.
#[test]
fn max_entropy_with_equality_rows() {
    let f = [-1.0, -0.2, 0.5, 1.0, 2.0];
    let k = f.len();
    let n = 2 * k; // x, then t
    let mut c = vec![0.0; n];
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for i in 0..k {
        c[k + i] = -1.0;
        let r0 = h.len();
        g.push(Triplet::new(r0, k + i, -1.0));
        g.push(Triplet::new(r0 + 1, i, -1.0));
        h.extend_from_slice(&[0.0, 0.0, 1.0]);
    }
    let mut a = Vec::new();
    for (i, fi) in f.iter().enumerate() {
        a.push(Triplet::new(0, i, 1.0));
        a.push(Triplet::new(1, i, *fi));
    }
    let prob = QpProblem {
        n,
        p_lower: vec![],
        c,
        a,
        b: vec![1.0, 0.3],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Exponential; 5], 1e-8);
    assert_eq!(sol.status, QpStatus::Optimal);
    let slope = |i: usize| (sol.x[i].ln() - sol.x[0].ln()) / (f[i] - f[0]);
    let s1 = slope(1);
    for i in 2..k {
        assert!(
            (slope(i) - s1).abs() < 1e-11,
            "slope {i}: {} vs {s1}",
            slope(i)
        );
    }
    let sum: f64 = sol.x[..k].iter().sum();
    let mean: f64 = (0..k).map(|i| f[i] * sol.x[i]).sum();
    assert!((sum - 1.0).abs() < 1e-14 && (mean - 0.3).abs() < 1e-14);
}
