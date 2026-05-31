//! Demonstrates the LP/QP presolve reductions and the rayon-parallel
//! duplicate-row detection, reporting the size reduction and the solve.
//!
//! Run: `cargo run -p pounce-convex --release --example presolve_reductions`

use pounce_convex::presolve::{presolve, solve_with_presolve, PresolveOutcome};
use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use std::time::Instant;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn report(name: &str, prob: &QpProblem) {
    print!("{name:<34} {}×{} → ", prob.n, prob.m_eq() + prob.m_ineq());
    match presolve(prob) {
        PresolveOutcome::Infeasible => println!("INFEASIBLE (detected in presolve)"),
        PresolveOutcome::Unbounded => println!("UNBOUNDED (detected in presolve)"),
        PresolveOutcome::Reduced(ps) => {
            let r = &ps.reduced;
            let sol =
                solve_with_presolve(prob, |p| solve_qp_ipm(p, &QpOptions::default(), backend));
            println!(
                "{}×{}   solve: {:?} obj={:.4}",
                r.n,
                r.m_eq() + r.m_ineq(),
                sol.status,
                sol.obj
            );
            assert_eq!(sol.status, QpStatus::Optimal);
        }
    }
}

fn main() {
    println!("=== reduction showcase (original → reduced size) ===");

    // Free column with zero cost: x1 is irrelevant and removed.
    report(
        "free column (dropped)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
        },
    );

    // Free column with nonzero cost: unbounded, detected without solving.
    report(
        "free column (unbounded)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![0.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
        },
    );

    // Fixed variable from a singleton equality row.
    report(
        "fixed variable (singleton eq)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 1, 1.0), // x1 = 1
            ],
            b: vec![3.0, 1.0],
            g: vec![],
            h: vec![],
        },
    );

    // Conflicting duplicate equalities: infeasible.
    report(
        "conflicting duplicate eq",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 1.0),
            ],
            b: vec![2.0, 3.0],
            g: vec![],
            h: vec![],
        },
    );

    println!("\n=== rayon-parallel duplicate-row detection at scale ===");
    for &(n, k) in &[(50usize, 200usize), (100, 1000), (200, 4000)] {
        let mut p_lower = Vec::new();
        for i in 0..n {
            p_lower.push(Triplet::new(i, i, 2.0));
        }
        // K identical equality rows Σx_i = n; presolve collapses to 1.
        let mut a = Vec::new();
        for row in 0..k {
            for i in 0..n {
                a.push(Triplet::new(row, i, 1.0));
            }
        }
        let prob = QpProblem {
            n,
            p_lower,
            c: vec![0.0; n],
            a,
            b: vec![n as f64; k],
            g: vec![],
            h: vec![],
        };
        let t0 = Instant::now();
        let reduced_rows = match presolve(&prob) {
            PresolveOutcome::Reduced(ps) => ps.reduced.m_eq(),
            _ => unreachable!(),
        };
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        println!("n={n:<4} {k} duplicate eq rows → {reduced_rows} kept   (presolve {dt:.2} ms)");
    }
}
