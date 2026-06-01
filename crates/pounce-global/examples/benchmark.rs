//! Global-optimization benchmark harness (`benchmarks/global` tier).
//!
//! A graduated suite of verifiable nonconvex problems — from quick 2-D
//! classics to a 4-D instance that branches into the thousands — exercising the
//! relaxation suite, the branching rules, and (on the large instance) the
//! parallel node pool. Each row reports the certified optimum against the known
//! global value, the node count, the peak frontier size, and the estimated peak
//! frontier memory.
//!
//! Run: `cargo run --release -p pounce-global --example benchmark`

use pounce_feral::FeralSolverInterface;
use pounce_global::{
    expr::{con, var, Expr},
    solve_global, GlobalOptions, GlobalProblem, GlobalStatus,
};
use pounce_linsol::SparseSymLinearSolverInterface;
use std::time::Instant;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn fmt_bytes(b: usize) -> String {
    let k = 1024.0;
    let x = b as f64;
    if x >= k * k {
        format!("{:.1} MiB", x / (k * k))
    } else if x >= k {
        format!("{:.1} KiB", x / k)
    } else {
        format!("{b} B")
    }
}

struct Row {
    name: String,
    n: usize,
    threads: usize,
    known: f64,
    obj: f64,
    gap: f64,
    nodes: usize,
    peak_frontier: usize,
    peak_mem: usize,
    secs: f64,
    ok: bool,
}

fn bench(name: &str, prob: &GlobalProblem, known: f64, threads: usize) -> Row {
    let opts = GlobalOptions {
        abs_gap: 1e-4,
        rel_gap: 1e-4,
        max_nodes: 500_000,
        threads,
        ..Default::default()
    };
    let t = Instant::now();
    let s = solve_global(prob, &opts, backend);
    let secs = t.elapsed().as_secs_f64();
    let ok = s.status == GlobalStatus::Optimal && (s.objective - known).abs() < 1e-2;
    Row {
        name: name.to_string(),
        n: prob.n_vars,
        threads,
        known,
        obj: s.objective,
        gap: s.gap(),
        nodes: s.nodes,
        peak_frontier: s.peak_frontier,
        peak_mem: s.peak_memory_bytes,
        secs,
        ok,
    }
}

fn camel(a: usize, b: usize) -> Expr {
    let x = var(a);
    let y = var(b);
    4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4)
}

fn allpairs(n: usize) -> GlobalProblem {
    let mut f = con(0.0);
    for i in 0..n {
        for j in i + 1..n {
            f = f + var(i) * var(j);
        }
    }
    GlobalProblem::new(vec![-1.0; n], vec![1.0; n], &f)
}

fn main() {
    let mut rows = Vec::new();

    // 2-D classics.
    rows.push(bench(
        "six-hump camel",
        &GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &camel(0, 1)),
        -1.031_628,
        1,
    ));
    let himmel = (var(0).powi(2) + var(1) - 11.0).powi(2) + (var(0) + var(1).powi(2) - 7.0).powi(2);
    rows.push(bench(
        "himmelblau",
        &GlobalProblem::new(vec![-5.0, -5.0], vec![5.0, 5.0], &himmel),
        0.0,
        1,
    ));
    // Bukin N.6 — non-smooth (|·| + √), forces branching.
    let bukin =
        100.0 * (var(1) - 0.01 * var(0).powi(2)).abs().sqrt() + 0.01 * (var(0) + con(10.0)).abs();
    rows.push(bench(
        "bukin-6",
        &GlobalProblem::new(vec![-15.0, -3.0], vec![-5.0, 3.0], &bukin),
        0.0,
        1,
    ));

    // Scalable all-pairs bilinear: min Σ_{i<j} x_i x_j on [−1,1]ⁿ = −n/2 (n even).
    rows.push(bench("allpairs bilinear", &allpairs(4), -2.0, 1));
    rows.push(bench("allpairs bilinear", &allpairs(6), -3.0, 1));
    rows.push(bench("allpairs bilinear", &allpairs(8), -4.0, 1));

    // Large instance: separable 4-D double camel (global 2×−1.0316), serial then
    // on the parallel node pool — the node count is high enough to saturate it.
    let dc = GlobalProblem::new(
        vec![-2.0, -1.5, -2.0, -1.5],
        vec![2.0, 1.5, 2.0, 1.5],
        &(camel(0, 1) + camel(2, 3)),
    );
    rows.push(bench("double camel", &dc, -2.063_256, 1));
    rows.push(bench("double camel", &dc, -2.063_256, 8));

    // Markdown table.
    println!("| instance | n | threads | status | objective | known | gap | nodes | peak frontier | est. peak mem | time (s) |");
    println!("|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {} | {:+.5} | {:+.5} | {:.1e} | {} | {} | {} | {:.2} |",
            r.name,
            r.n,
            r.threads,
            if r.ok { "Optimal ✓" } else { "MISMATCH ✗" },
            r.obj,
            r.known,
            r.gap,
            r.nodes,
            r.peak_frontier,
            fmt_bytes(r.peak_mem),
            r.secs,
        );
    }
}
