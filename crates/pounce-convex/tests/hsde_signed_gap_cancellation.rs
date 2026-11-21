//! The HSDE stop accepted a duality gap that only *cancellation* had made
//! small (MIPLIB `traininstance6`; exposed when FERAL static pivoting,
//! gh#1193, moved the convex IPM's trajectory).
//!
//! The embedding's gap test reads the signed difference
//! `x̂ᵀPx̂ + cᵀx̂ + bᵀŷ + hᵀẑ`. That equals the complementarity sum `ŝᵀẑ` only
//! at a residual-free point; in between, the two differ by `x̂ᵀρ̂_x` (and the
//! primal-residual couplings), and along the embedding's path those residuals
//! shrink at the same rate as `μ`. With columns in the thousands, the coupling
//! cancels nearly all of `ŝᵀẑ`: `traininstance6`'s reduced LP stopped at a
//! signed gap of `9.8e-9` against `ŝᵀẑ = 3.3e-6`, every individual product
//! below `tol`. Its `(x, λ)` then failed the cpo3 verifier's duality-gap
//! check at `2.5e-6` (`tol = 1e-8`), where `presolve=no`, or the same LP on
//! the pre-static-pivoting trajectory, verified at `~1e-9`.
//!
//! The model here is a purpose-built stand-in with the same shape — objective
//! `min x₀` with optimum `0`, columns boxed in `[0, ~1e4]`, many inactive
//! sparse rows — and nothing else. On 22ae155e it returned `Optimal` at 15
//! iterations with a complementarity sum of `2.0e-6` (signed gap `< 1e-8`).
//!
//! The gate that fixes it (`comp_sum_ok` in `hsde::solve_conic_hsde`) is
//! `ŝᵀẑ ≤ max(tol_gap, tol·|objective|)`, so it has two branches and each
//! needs its own fixture:
//! - objective `≈ 0`: the absolute half binds and refuses the cancelled stop;
//! - objective `300`: the relative half admits a sum above `tol_gap`, so a
//!   large objective is not held to an absolute gap it never was before.

use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

/// xorshift64, so the model is fixed by the seed and needs no data file.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `min x₀` over `Gx ≤ h`, `0 ≤ x ≤ ub` (`ub ~ 1e4`), 150 rows of 2–3
/// integer coefficients, every seventh also carrying `-x₀`. `h` is built from
/// a feasible point with `x₀ = 0` plus a slack (zero on ~30% of the rows), so
/// the optimum is `0`. `lb0` raises `x₀`'s bound, and with it the optimum.
fn model(lb0: f64) -> QpProblem {
    let (n, m, umax) = (60usize, 150usize, 1e4);
    let mut r = Rng(7u64.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let ub: Vec<f64> = (0..n).map(|_| umax * (0.1 + r.unit())).collect();
    let mut xf: Vec<f64> = (0..n).map(|j| ub[j] * r.unit()).collect();
    xf[0] = 0.0;
    let (mut g, mut h) = (Vec::new(), Vec::new());
    for i in 0..m {
        let mut act = 0.0;
        let k = 2 + (r.unit() * 2.0) as usize;
        let mut cols = Vec::new();
        for _ in 0..k {
            let j = (r.unit() * n as f64) as usize;
            if cols.contains(&j) {
                continue;
            }
            cols.push(j);
            let sign = if r.unit() < 0.5 { -1.0 } else { 1.0 };
            let v = sign * (1.0 + 10.0 * r.unit()).floor();
            g.push(Triplet::new(i, j, v));
            act += v * xf[j];
        }
        if i % 7 == 0 && !cols.contains(&0) {
            g.push(Triplet::new(i, 0, -1.0));
        }
        let slack = if r.unit() < 0.3 { 0.0 } else { umax * r.unit() };
        h.push(act + slack);
    }
    let mut c = vec![0.0; n];
    c[0] = 1.0;
    let mut lb = vec![0.0; n];
    lb[0] = lb0;
    QpProblem {
        n,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb,
        ub,
    }
}

/// The returned point's complementarity sum on the *true* slacks,
/// `Σ zᵢ(h − Gx)ᵢ + Σ z_lb(x − lb) + Σ z_ub(ub − x)`: the primal-dual gap an
/// independent checker reads off `(x, z)`.
fn true_comp_sum(p: &QpProblem, s: &QpSolution) -> f64 {
    let mut gx = vec![0.0; p.h.len()];
    for e in &p.g {
        gx[e.row] += e.val * s.x[e.col];
    }
    let rows: f64 = (0..p.h.len()).map(|i| s.z[i] * (p.h[i] - gx[i])).sum();
    let bounds: f64 = (0..p.n)
        .map(|j| s.z_lb[j] * (s.x[j] - p.lb[j]) + s.z_ub[j] * (p.ub[j] - s.x[j]))
        .sum();
    rows + bounds
}

#[test]
fn a_cancelled_signed_gap_does_not_stop_a_zero_objective_lp() {
    let p = model(0.0);
    let o = QpOptions::default();
    let s = solve_qp_ipm(&p, &o, backend);
    assert_eq!(s.status, QpStatus::Optimal);
    assert!(s.obj.abs() < 1e-8, "objective {:e}", s.obj);
    let sum = true_comp_sum(&p, &s);
    // 22ae155e: 2.0e-6 at 15 iterations. Fixed: 2.0e-9 at 18.
    assert!(
        sum.abs() <= 2.0 * o.tol,
        "complementarity sum {sum:e} at {} iterations: the stop accepted a gap \
         that only the residual coupling made small",
        s.iters
    );
}

#[test]
fn a_large_objective_keeps_its_relative_gap_allowance() {
    let p = model(300.0);
    let o = QpOptions::default();
    let s = solve_qp_ipm(&p, &o, backend);
    assert_eq!(s.status, QpStatus::Optimal);
    assert!((s.obj - 300.0).abs() < 1e-6, "objective {:e}", s.obj);
    let sum = true_comp_sum(&p, &s);
    // Measured 1.5e-8 at 16 iterations, both before and after the gate: the
    // relative half (`tol·300 = 3e-6`) admits it. If this lower bound fails,
    // the fixture no longer reaches the relative branch -- re-pick it rather
    // than delete the bound, or the branch goes untested.
    assert!(
        sum > o.tol,
        "complementarity sum {sum:e} no longer above tol_gap: fixture lost its branch"
    );
    assert!(sum <= o.tol * 300.0, "complementarity sum {sum:e}");
}
