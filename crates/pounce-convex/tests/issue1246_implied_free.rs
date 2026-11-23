//! gh#1246 finding 3: implied-free columns are substituted out of equality
//! rows.
//!
//! Column `j` in the equality row `r` is implied free when the row, over the
//! other columns' boxes, keeps `x_j` inside `j`'s own box. Then `x_j` is
//! replaced by the row's expression everywhere and both go. One fixture per
//! branch of `implied_free_once`:
//!
//! * the row implies the box: substituted, including into an inequality row;
//! * the box is tighter than the row allows: kept (it can bind);
//! * the column is in `P`: kept;
//! * the pivot is tiny against the row: kept;
//! * a randomized KKT roundtrip over models where substitutions overlap,
//!   so the independence rule decides which go together.
//!
//! Every case solves with and without presolve and checks the *original*
//! problem's KKT conditions at the postsolved point, so a wrong primal
//! expression or a wrong recovered multiplier fails here.

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

const INF: f64 = f64::INFINITY;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn substituted(prob: &QpProblem) -> usize {
    match presolve(prob) {
        PresolveOutcome::Reduced(ps) => ps.stats().implied_free_cols,
        _ => panic!("expected a reduction"),
    }
}

fn assert_kkt(p: &QpProblem, s: &QpSolution, tol: f64) {
    let mut g = p.c.clone();
    p.p_mul(&s.x, &mut g);
    p.at_mul(&s.y, &mut g);
    p.gt_mul(&s.z, &mut g);
    for j in 0..p.n {
        let stat = g[j] - s.z_lb[j] + s.z_ub[j];
        assert!(stat.abs() < tol, "stationarity[{j}] = {stat:e}");
        assert!(
            s.z_lb[j] > -tol && s.z_ub[j] > -tol,
            "bound multiplier sign [{j}]"
        );
        assert!(
            s.x[j] >= p.lb_of(j) - tol && s.x[j] <= p.ub_of(j) + tol,
            "box [{j}]"
        );
        if p.lb_of(j).is_finite() {
            assert!(
                (s.z_lb[j] * (s.x[j] - p.lb_of(j))).abs() < 1e-5,
                "lb comp [{j}]"
            );
        }
        if p.ub_of(j).is_finite() {
            assert!(
                (s.z_ub[j] * (p.ub_of(j) - s.x[j])).abs() < 1e-5,
                "ub comp [{j}]"
            );
        }
    }
    let mut ax = vec![0.0; p.m_eq()];
    p.a_mul(&s.x, &mut ax);
    for (i, (a, b)) in ax.iter().zip(&p.b).enumerate() {
        assert!((a - b).abs() < tol, "row {i}: {a} vs {b}");
    }
    let mut gx = vec![0.0; p.m_ineq()];
    p.g_mul(&s.x, &mut gx);
    for i in 0..p.m_ineq() {
        let slack = p.h[i] - gx[i];
        assert!(slack > -tol, "G row {i}: slack {slack:e}");
        assert!(s.z[i] > -tol, "z[{i}] = {}", s.z[i]);
        assert!((s.z[i] * slack).abs() < 1e-5, "ineq comp row {i}");
    }
}

fn same_answer(prob: &QpProblem) {
    let o = QpOptions::default();
    let plain = solve_qp_ipm(prob, &o, backend);
    let pre = solve_with_presolve(prob, |q| solve_qp_ipm(q, &o, backend));
    assert_eq!(plain.status, QpStatus::Optimal);
    assert_eq!(pre.status, QpStatus::Optimal);
    assert!(
        (plain.obj - pre.obj).abs() <= 1e-6 * (1.0 + plain.obj.abs()),
        "plain {} vs presolved {}",
        plain.obj,
        pre.obj
    );
    assert_kkt(prob, &pre, 1e-6);
}

/// `x₀ = x₁ + x₂ − 1` with `x₁, x₂ ∈ [0, 1]` keeps `x₀` in `[−1, 1]`, and its
/// box is `[−5, 5]`: implied free. `x₀` also enters an inequality row and
/// carries a cost, so the substitution rewrites both. The objective's
/// quadratic is on `x₁`, `x₂` and `x₃` only, as the rule requires.
fn model(x0_box: (f64, f64), p_on_x0: bool, pivot: f64) -> QpProblem {
    let mut p_lower = vec![
        Triplet::new(1, 1, 1.0),
        Triplet::new(2, 2, 2.0),
        Triplet::new(3, 3, 1.0),
    ];
    if p_on_x0 {
        p_lower.push(Triplet::new(0, 0, 1.0));
    }
    QpProblem {
        n: 4,
        p_lower,
        c: vec![1.5, -1.0, 0.5, -2.0],
        // pivot·x₀ − x₁ − x₂ = −1
        a: vec![
            Triplet::new(0, 0, pivot),
            Triplet::new(0, 1, -1.0),
            Triplet::new(0, 2, -1.0),
        ],
        b: vec![-1.0],
        // x₀ + x₃ ≤ 2
        g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 3, 1.0)],
        h: vec![2.0],
        lb: vec![x0_box.0, 0.0, 0.0, 0.0],
        ub: vec![x0_box.1, 1.0, 1.0, 3.0],
    }
}

#[test]
fn an_implied_free_column_is_substituted_into_every_row() {
    let prob = model((-5.0, 5.0), false, 1.0);
    assert_eq!(substituted(&prob), 1);
    same_answer(&prob);
}

#[test]
fn a_box_tighter_than_the_row_is_kept() {
    // x₀ ∈ [0, 0.5] can bind: the row allows [−1, 1].
    let prob = model((0.0, 0.5), false, 1.0);
    assert_eq!(substituted(&prob), 0);
    same_answer(&prob);
}

#[test]
fn a_column_in_the_hessian_is_kept() {
    let prob = model((-5.0, 5.0), true, 1.0);
    assert_eq!(substituted(&prob), 0);
    same_answer(&prob);
}

#[test]
fn a_tiny_pivot_is_kept() {
    // 1e-3·x₀ against unit coefficients: dividing by it would amplify the
    // row's rounding a thousandfold. Its implied box is [−1000, 1000], so
    // give it room to be implied free on the box test alone.
    let prob = model((-5e3, 5e3), false, 1e-3);
    assert_eq!(substituted(&prob), 0);
    same_answer(&prob);
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn coef(&mut self) -> f64 {
        [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0][self.below(6) as usize]
    }
}

/// Random equality rows over boxed columns, with wide boxes on some columns
/// so they are implied free, and rows that share columns so the
/// independence rule has to choose. Each row's right-hand side comes from a
/// point inside every box, so the model is feasible; a strictly convex
/// diagonal on the columns without wide boxes, plus bounded boxes, keeps it
/// bounded.
#[test]
fn randomized_substitutions_roundtrip_to_a_kkt_point() {
    let mut rng = Lcg(0x1246f);
    let mut total = 0;
    for trial in 0..60 {
        let n = 6 + rng.below(5) as usize;
        let m_eq = 2 + rng.below(3) as usize;
        let m_ineq = 1 + rng.below(3) as usize;
        let wide: Vec<bool> = (0..n).map(|_| rng.below(3) == 0).collect();
        let lb: Vec<f64> = (0..n).map(|j| if wide[j] { -100.0 } else { 0.0 }).collect();
        let ub: Vec<f64> = (0..n).map(|j| if wide[j] { 100.0 } else { 1.0 }).collect();
        let point: Vec<f64> = (0..n).map(|j| if wide[j] { 0.3 } else { 0.5 }).collect();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for r in 0..m_eq {
            let mut row = 0.0;
            let mut cols: Vec<usize> = Vec::new();
            while cols.len() < 3 + rng.below(2) as usize {
                let c = rng.below(n as u64) as usize;
                if !cols.contains(&c) {
                    cols.push(c);
                }
            }
            for &c in &cols {
                let v = rng.coef();
                a.push(Triplet::new(r, c, v));
                row += v * point[c];
            }
            b.push(row);
        }
        let mut g = Vec::new();
        let mut h = Vec::new();
        for r in 0..m_ineq {
            let mut row = 0.0;
            for c in 0..n {
                if rng.below(3) == 0 {
                    let v = rng.coef();
                    g.push(Triplet::new(r, c, v));
                    row += v * point[c];
                }
            }
            h.push(row + 1.0);
        }
        let prob = QpProblem {
            n,
            p_lower: (0..n)
                .filter(|&j| !wide[j])
                .map(|j| Triplet::new(j, j, 1.0))
                .collect(),
            c: (0..n).map(|_| rng.coef()).collect(),
            a,
            b,
            g,
            h,
            lb,
            ub,
        };
        total += substituted(&prob);
        let o = QpOptions::default();
        let plain = solve_qp_ipm(&prob, &o, backend);
        let pre = solve_with_presolve(&prob, |q| solve_qp_ipm(q, &o, backend));
        assert_eq!(plain.status, QpStatus::Optimal, "trial {trial}");
        assert_eq!(pre.status, QpStatus::Optimal, "trial {trial}");
        assert!(
            (plain.obj - pre.obj).abs() <= 1e-6 * (1.0 + plain.obj.abs()),
            "trial {trial}: plain {} vs presolved {}",
            plain.obj,
            pre.obj
        );
        assert_kkt(&prob, &pre, 1e-6);
    }
    assert!(total > 20, "only {total} substitutions across trials");
}
