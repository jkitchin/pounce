//! gh#984 — the convex IPM's stopping test depended on the units of `c`, `P`
//! and on a constant objective offset.

use pounce_convex::{QpOptions, QpProblem, QpSolution, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(p: &QpProblem, tol: f64) -> QpSolution {
    let opts = QpOptions {
        tol,
        ..QpOptions::default()
    };
    solve_qp_ipm(p, &opts, backend)
}

/// Item 1: the refinery blending LP of the issue.
fn refinery() -> QpProblem {
    let avail = [3000.0, 6000.0, 9000.0, 12000.0, 4000.0];
    let cost = [45.0, 70.0, 98.0, 92.0, 105.0];
    let ron = [93.0, 70.0, 98.0, 92.0, 96.0];
    let rvp: [f64; 5] = [52.0, 11.0, 3.5, 7.0, 4.6];
    let s = [10.0, 5.0, 1.0, 20.0, 6.0];
    let sg = [0.584, 0.664, 0.810, 0.745, 0.700];
    let price = [110.0, 122.0];
    let mx = [20000.0, 8000.0];
    let ronmin = [91.0, 96.0];
    let col = |i: usize, j: usize| 2 * i + j;
    let mut g = Vec::new();
    let mut h = Vec::new();
    let mut add = |co: Vec<(usize, f64)>, b: f64| {
        let r = h.len();
        for (k, v) in co {
            g.push(Triplet::new(r, k, v));
        }
        h.push(b);
    };
    for i in 0..5 {
        add((0..2).map(|j| (col(i, j), 1.0)).collect(), avail[i]);
    }
    for j in 0..2 {
        add((0..5).map(|i| (col(i, j), 1.0)).collect(), mx[j]);
        add(
            (0..5).map(|i| (col(i, j), -(ron[i] - ronmin[j]))).collect(),
            0.0,
        );
        add(
            (0..5)
                .map(|i| (col(i, j), rvp[i].powf(1.25) - 9f64.powf(1.25)))
                .collect(),
            0.0,
        );
        add(
            (0..5).map(|i| (col(i, j), sg[i] * (s[i] - 10.0))).collect(),
            0.0,
        );
    }
    let mut c = vec![0.0; 10];
    let mut ub = vec![0.0; 10];
    for i in 0..5 {
        for j in 0..2 {
            c[col(i, j)] = -(price[j] - cost[i]);
            ub[col(i, j)] = mx[j];
        }
    }
    QpProblem {
        n: 10,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![0.0; 10],
        ub,
    }
}

/// Item 3: constant objective offset via a shifted variable.
fn shifted_eta() -> QpProblem {
    QpProblem {
        n: 4,
        p_lower: vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(1, 1, 1.0),
            Triplet::new(2, 2, 1.0),
        ],
        c: vec![0.0, 0.0, 0.0, -1.0],
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -64.0),
            Triplet::new(0, 1, -74.0),
            Triplet::new(0, 2, -63.0),
            Triplet::new(0, 3, 1.0),
        ],
        h: vec![1e9 - 1021.0],
        lb: vec![0.0; 4],
        ub: vec![1e6, 1e6, 1e6, 2e9],
    }
}

/// Item 5: per-minute variance portfolio.
fn portfolio(scale: f64) -> QpProblem {
    let mu = [3.0, 4.5, 6.0, 9.0, 14.0];
    let sig = [5.0, 8.0, 15.0, 20.0, 30.0];
    let rho = [
        [1.0, 0.6, -0.2, -0.1, 0.0],
        [0.6, 1.0, 0.3, 0.3, 0.2],
        [-0.2, 0.3, 1.0, 0.3, 0.1],
        [-0.1, 0.3, 0.3, 1.0, 0.5],
        [0.0, 0.2, 0.1, 0.5, 1.0],
    ];
    let mut p = Vec::new();
    for i in 0..5 {
        for j in 0..=i {
            p.push(Triplet::new(
                i,
                j,
                2.0 * rho[i][j] * sig[i] * sig[j] * scale,
            ));
        }
    }
    QpProblem {
        n: 5,
        p_lower: p,
        c: vec![0.0; 5],
        a: (0..5).map(|j| Triplet::new(0, j, 1.0)).collect(),
        b: vec![1.0],
        g: (0..5).map(|j| Triplet::new(0, j, -mu[j])).collect(),
        h: vec![-10.0],
        lb: vec![0.0; 5],
        ub: vec![1.0; 5],
    }
}

/// Deterministic stand-in for numpy's `uniform`.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Item 2: multi-product production LP, `T` weeks, costs scaled by `s`.
fn production_lp(t: usize, s: f64) -> QpProblem {
    let p = 3usize;
    let mut st = 9u64;
    let hold = [0.8, 1.0, 1.2];
    let a = [1.0, 1.5, 2.0];
    let base = [20.0, 26.0, 31.0];
    let ncol = (2 * p + 2) * t;
    let mut trip = Vec::new();
    let mut b = Vec::new();
    let mut c = vec![0.0; ncol];
    let blk = |k: usize, w: usize| k * t + w;
    for i in 0..p {
        for w in 0..t {
            let d = (80.0 + 40.0 * lcg(&mut st))
                * (1.0 - 0.3 * (2.0 * std::f64::consts::PI * w as f64 / 52.0).sin());
            let r = i * t + w;
            trip.push(Triplet::new(r, blk(i, w), 1.0));
            trip.push(Triplet::new(r, blk(p + i, w), -1.0));
            if w > 0 {
                trip.push(Triplet::new(r, blk(p + i, w - 1), 1.0));
            }
            b.push(d);
            c[blk(i, w)] = base[i] * (0.95 + 0.1 * lcg(&mut st)) * s;
            c[blk(p + i, w)] = hold[i] * s;
        }
    }
    for w in 0..t {
        let r = p * t + w;
        for i in 0..p {
            trip.push(Triplet::new(r, blk(i, w), a[i]));
        }
        trip.push(Triplet::new(r, blk(2 * p, w), -1.0));
        trip.push(Triplet::new(r, blk(2 * p + 1, w), 1.0));
        b.push(400.0);
        c[blk(2 * p, w)] = 45.0 * s;
    }
    QpProblem {
        n: ncol,
        p_lower: vec![],
        c,
        a: trip,
        b,
        g: vec![],
        h: vec![],
        lb: vec![0.0; ncol],
        ub: vec![],
    }
}

fn primal_residual(p: &QpProblem, x: &[f64]) -> f64 {
    let mut ax = vec![0.0; p.m_eq()];
    p.a_mul(x, &mut ax);
    ax.iter()
        .zip(&p.b)
        .map(|(u, v)| (u - v).abs())
        .fold(0.0, f64::max)
}

fn kkt(p: &QpProblem, r: &QpSolution) -> f64 {
    r.kkt_residuals(p).kkt_error()
}

/// Item 1. Tightening `tol` must not loosen the answer. On the unfixed code
/// `tol = 1e-10` stopped at `kkt_error 1.2e-6`, ten times *worse* than
/// `tol = 1e-6`'s `1.2e-7`: the relative arm granted `tol·(1 + scale)`.
#[test]
fn tightening_tol_never_loosens_the_answer() {
    let p = refinery();
    let mut prev = f64::INFINITY;
    for tol in [1e-6, 1e-8, 1e-10, 1e-12] {
        let r = solve(&p, tol);
        assert_eq!(r.status, QpStatus::Optimal, "tol={tol:e}");
        let e = kkt(&p, &r);
        // Within a factor of two of non-increasing, i.e. not a 10x regression.
        assert!(
            e <= 2.0 * prev,
            "tol={tol:e}: kkt_error {e:e} vs previous {prev:e}"
        );
        prev = e;
    }
    assert!(prev < 1e-7, "tol=1e-12 left kkt_error {prev:e}");
}

/// Item 2. Primal feasibility is measured against `A` and `b`, not `c`: the
/// same LP with costs in cents must not be left less feasible.
#[test]
fn primal_feasibility_does_not_depend_on_the_cost_unit() {
    for s in [1.0, 100.0, 1e4, 1e-3] {
        let p = production_lp(208, s);
        let r = solve(&p, 1e-10);
        assert_eq!(r.status, QpStatus::Optimal, "s={s}");
        let e = primal_residual(&p, &r.x);
        assert!(e < 1e-8, "s={s}: |Ax-b| = {e:e}");
    }
}

/// Item 3. A constant offset (a shifted variable) must not set the relative
/// scale: the returned point has to be the true optimum, and its own
/// complementarity far below the `1.08` the unfixed code reported.
#[test]
fn a_shifted_variable_does_not_set_the_relative_scale() {
    let p = shifted_eta();
    let r = solve(&p, 1e-8);
    assert_eq!(r.status, QpStatus::Optimal);
    for (got, want) in r.x.iter().zip([64.0, 74.0, 63.0]) {
        assert!((got - want).abs() < 1e-5, "u = {:?}", &r.x[..3]);
    }
    let k = kkt(&p, &r);
    assert!(k < 1e-3, "Optimal with kkt_error {k:e}");
}

/// Item 5. Scaling the objective by a positive constant leaves the minimizer
/// where it is, so it must not move the answer: `P·1e-9` used to stop with the
/// weights off by `1.6e-2`.
#[test]
fn the_answer_does_not_depend_on_the_objective_scale() {
    let w = solve(&portfolio(1.0), 1e-12).x;
    for sc in [1.0, 1e-4, 1e-4 / 252.0, 1e-4 / (252.0 * 390.0)] {
        let r = solve(&portfolio(sc), 1e-8);
        assert_eq!(r.status, QpStatus::Optimal, "scale={sc:e}");
        let e =
            r.x.iter()
                .zip(&w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
        assert!(e < 1e-6, "scale={sc:e}: weights off by {e:e}");
    }
}

/// Item 6. A degenerate tied transportation LP through crossover: the vertex
/// must be exact (`x_B` recomputed after the final pivot), so the returned
/// point satisfies the constraints and complementarity to rounding.
#[test]
fn crossover_returns_an_exact_vertex() {
    let supply = [120.0, 90.0, 70.0];
    let demand = [60.0, 80.0, 50.0, 70.0];
    let cost = [
        [1.2, 2.1, 3.0, 2.4],
        [1.8, 0.9, 1.6, 2.0],
        [2.6, 1.9, 1.1, 1.5],
    ];
    let mut g = Vec::new();
    let mut h = Vec::new();
    for i in 0..3 {
        for j in 0..4 {
            g.push(Triplet::new(i, 4 * i + j, 1.0));
        }
        h.push(supply[i]);
    }
    for j in 0..4 {
        for i in 0..3 {
            g.push(Triplet::new(3 + j, 4 * i + j, -1.0));
        }
        h.push(-demand[j]);
    }
    let p = QpProblem {
        n: 12,
        p_lower: vec![],
        c: (0..12).map(|k| 1000.0 * cost[k / 4][k % 4]).collect(),
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![0.0; 12],
        ub: vec![],
    };
    let opts = QpOptions {
        crossover: true,
        ..QpOptions::default()
    };
    let r = solve_qp_ipm(&p, &opts, backend);
    assert_eq!(r.status, QpStatus::Optimal);
    let res = r.kkt_residuals(&p);
    assert!(
        res.kkt_error() < 1e-9,
        "crossover left kkt_error {:e} ({res:?})",
        res.kkt_error()
    );
}
