//! Demonstrates the LP/QP presolve reductions and the rayon-parallel
//! duplicate-row detection, reporting the size reduction and the solve.
//!
//! Run: `cargo run -p pounce-convex --release --example presolve_reductions`

use pounce_convex::presolve::{PresolveOutcome, presolve, solve_with_presolve};
use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
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
            lb: vec![],
            ub: vec![],
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
            lb: vec![],
            ub: vec![],
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
            lb: vec![],
            ub: vec![],
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
            lb: vec![],
            ub: vec![],
        },
    );

    // Activity-redundant inequality: with x ∈ [0,1]², `x0+x1 ≤ 5` has
    // max activity 2 ≤ 5, so it is always satisfied and dropped.
    report(
        "redundant ineq (activity)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-1.0, -1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![5.0],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        },
    );

    // Activity-infeasible equality: with x ∈ [0,1]², `x0+x1 = 5` is
    // outside the activity range [0, 2].
    report(
        "infeasible eq (activity)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![5.0],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![1.0, 1.0],
        },
    );

    // Forcing inequality: with x ∈ [0,5]², `x0+x1 ≤ 0` has min activity
    // 0 = h, so it holds only at x0=x1=0 — both variables pinned, row
    // dropped. (Dual recovered exactly in postsolve.)
    report(
        "forcing ineq (pins to bounds)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-2.0, -3.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![0.0],
            lb: vec![0.0, 0.0],
            ub: vec![5.0, 5.0],
        },
    );

    // Parallel inequalities (scalar multiple): `x0+x1 ≤ 3` and
    // `2x0+2x1 ≤ 2` (⟺ x0+x1 ≤ 1). The tighter is kept, the other dropped.
    report(
        "parallel ineq (keep tightest)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-10.0, -10.0],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 0, 2.0),
                Triplet::new(1, 1, 2.0),
            ],
            h: vec![3.0, 2.0],
            lb: vec![],
            ub: vec![],
        },
    );

    // Forcing equality at the max vertex: with x ∈ [0,4]², `x0+x1 = 8`
    // equals the max activity 8, pinning x0=x1=4.
    report(
        "forcing eq (max vertex)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![1.0, 5.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![8.0],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![4.0, 4.0],
        },
    );

    // Bound tightening: `2·x0 ≤ 3` implies x0 ≤ 1.5, tighter than the box
    // [0,10]; the reduced box is shrunk (the variable is kept).
    report(
        "bound tightening (shrink box)",
        &QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-10.0, -10.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 2.0)],
            h: vec![3.0],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 10.0],
        },
    );

    // Dominated column: x2 is not in P, appears only in the `≤` row with a
    // nonnegative coefficient, and has cost ≥ 0 — so x2 = lb is optimal;
    // it is fixed and dropped.
    report(
        "dominated column (→ bound)",
        &QpProblem {
            n: 3,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-4.0, -4.0, 0.5],
            a: vec![],
            b: vec![],
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            h: vec![3.0],
            lb: vec![0.0, 0.0, 0.0],
            ub: vec![5.0, 5.0, 5.0],
        },
    );

    // Free column singleton: x2 (free, only in the equality row) is
    // substituted out, eliminating both the variable and the row.
    report(
        "free col singleton (subst)",
        &QpProblem {
            n: 3,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            b: vec![3.0],
            g: vec![],
            h: vec![],
            lb: vec![f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
            ub: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
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
            lb: vec![],
            ub: vec![],
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
