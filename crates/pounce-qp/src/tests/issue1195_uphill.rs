//! gh#1195: an unshifted step of the elastic phase 1 must not raise the QP
//! objective. In exact arithmetic it cannot (`∇q(x)ᵀp = −pᵀHp < 0` on a
//! positive definite reduced Hessian); a computed one that does is the
//! solve's error, and the step is taken as a null step instead.
//!
//! The QPs are convex, feasible by construction and infeasible at the cold
//! start, so every solve goes through the elastic phase. The seeds are ones
//! where the solver used to take climbing steps: 5, 4, 9 and 10 of them on
//! seeds 22, 33, 70 and 114, and on seed 50 enough to end `MaxIter` where the
//! guarded solve reaches the optimum. The guard is on the refactor path only
//! (see `raises_the_objective`), and so are these solves.

use crate::solver::uphill_ledger;
use crate::{HessianInertia, ParametricActiveSetSolver, QpOptions, QpProblem, QpSolver, QpStatus};
use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};
use pounce_linalg::triplet::{GenTMatrix, GenTMatrixSpace, SymTMatrix, SymTMatrixSpace};

/// xorshift64, so the generated QPs are the same on every platform.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.next().max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}

/// Solve the seed's QP; returns its status and the number of steps that
/// raised the objective.
fn solve(seed: u64, use_schur_updates: bool) -> (QpStatus, u64) {
    let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let n = 10 + (r.next() * 60.0) as usize;
    let m = n / 2 + (r.next() * 2.0 * n as f64) as usize;
    let scale = [1.0, 1e3, 1e5][(r.next() * 3.0) as usize];
    // H = L Lᵀ (rank k), plus a small diagonal half the time: PSD.
    let k = 1 + (r.next() * n as f64 * 0.5) as usize;
    let l: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..k).map(|_| r.normal()).collect())
        .collect();
    let (mut hi, mut hj, mut hv) = (vec![], vec![], vec![]);
    let diag = if r.next() < 0.5 { 0.0 } else { 1e-3 };
    for i in 0..n {
        for j in 0..=i {
            let mut v: f64 = (0..k).map(|t| l[i][t] * l[j][t]).sum();
            if i == j {
                v += diag;
            }
            if v != 0.0 {
                hi.push(i as i32 + 1);
                hj.push(j as i32 + 1);
                hv.push(v);
            }
        }
    }
    // A feasible point `xs`, sparse rows (a tenth of entries 1000× larger),
    // and row bounds that `xs` satisfies.
    let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
    let (mut ai, mut aj, mut av) = (vec![], vec![], vec![]);
    let mut ax = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            if r.next() < 0.3 {
                let v = r.normal() * if r.next() < 0.1 { 1e3 } else { 1.0 };
                ai.push(i as i32 + 1);
                aj.push(j as i32 + 1);
                av.push(v);
                ax[i] += v * xs[j];
            }
        }
    }
    let mut bl = vec![NLP_LOWER_BOUND_INF; m];
    let mut bu = vec![NLP_UPPER_BOUND_INF; m];
    for i in 0..m {
        let u = r.next();
        let off = 0.5 + r.next();
        if u < 0.15 {
            bl[i] = ax[i];
            bu[i] = ax[i];
        } else if u < 0.6 {
            bl[i] = ax[i] + if r.next() < 0.5 { 0.0 } else { -off };
            bl[i] = bl[i].max(ax[i] - off);
            if bl[i] <= 0.0 {
                bl[i] = (ax[i].abs() + 0.1).min(ax[i]);
            }
        } else {
            bu[i] = ax[i] + if r.next() < 0.5 { 0.0 } else { off };
        }
    }
    let xl: Vec<f64> = xs
        .iter()
        .map(|&v| {
            if r.next() < 0.7 {
                v - r.next() * 2.0
            } else {
                NLP_LOWER_BOUND_INF
            }
        })
        .collect();
    let xu: Vec<f64> = xs
        .iter()
        .map(|&v| {
            if r.next() < 0.7 {
                v + r.next() * 2.0
            } else {
                NLP_UPPER_BOUND_INF
            }
        })
        .collect();
    let g: Vec<f64> = (0..n).map(|_| r.normal() * scale).collect();
    let mut h = SymTMatrix::new(SymTMatrixSpace::new(n as i32, hi, hj));
    h.set_values(&hv);
    let mut a = GenTMatrix::new(GenTMatrixSpace::new(m as i32, n as i32, ai, aj));
    a.set_values(&av);
    let qp = QpProblem {
        n,
        m,
        h: &h,
        g: &g,
        a: &a,
        bl: &bl,
        bu: &bu,
        xl: &xl,
        xu: &xu,
        hessian_inertia: HessianInertia::Psd,
    };
    let opts = QpOptions {
        max_iter: (10 * (n + m)) as u32,
        use_schur_updates,
        ..QpOptions::default()
    };
    let mut solver =
        ParametricActiveSetSolver::new(Box::new(pounce_feral::FeralSolverInterface::new()));
    uphill_ledger::take();
    let sol = solver.solve(&qp, None, &opts).expect("solve");
    (sol.status, uphill_ledger::take())
}

#[test]
fn no_step_climbs_on_the_seeds_that_used_to() {
    for seed in [22, 33, 70, 114] {
        let (status, climbs) = solve(seed, false);
        assert_eq!(
            climbs, 0,
            "seed {seed}: {climbs} steps raised the objective"
        );
        assert_eq!(status, QpStatus::Optimal, "seed {seed}");
    }
}

#[test]
fn a_solve_the_climbs_kept_from_converging_now_does() {
    let (status, climbs) = solve(50, false);
    assert_eq!(status, QpStatus::Optimal, "{climbs} climbing steps");
}
