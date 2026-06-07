//! Regression: a badly-scaled OBBT relaxation LP must solve to the true LP
//! optimum, not a wrong vertex.
//!
//! `tests/fixtures/ex4_1_2_relax.lp` is the *exact* first OBBT relaxation LP
//! captured from a `pounce-global` run on GLOBALLib `ex4_1_2` (n=352, m=302,
//! constraint coefficients spanning ~6e8 in magnitude). Before geometric
//! equilibration was built into the engine, the dense basis inverse was
//! ill-conditioned enough that maximizing variable 0 (true LP max = 2) returned
//! a wrong "optimal" vertex with x0 = 1 — which made OBBT collapse the box to
//! [1, 1] and cut the global optimum (certified −539.957 vs. the proven
//! −663.5). Ground truth here is from SciPy/HiGHS on the same system.

// Fixture parsing unwraps freely — a malformed fixture should fail the test loudly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_simplex::{LpProblem, LpStatus, Simplex, Triplet};
use std::path::Path;

fn parse_lp(path: &Path) -> LpProblem {
    let txt = std::fs::read_to_string(path).expect("read fixture");
    let (mut n, mut m) = (0usize, 0usize);
    let (mut c, mut b, mut lb, mut ub) = (vec![], vec![], vec![], vec![]);
    let mut a: Vec<Triplet> = vec![];
    let nums = |s: &str| -> Vec<f64> {
        s.split_whitespace()
            .skip(1)
            .map(|t| t.parse().unwrap())
            .collect()
    };
    for line in txt.lines() {
        if let Some(r) = line.strip_prefix("n ") {
            n = r.trim().parse().unwrap();
        } else if let Some(r) = line.strip_prefix("m ") {
            m = r.trim().parse().unwrap();
        } else if line.starts_with("c ") {
            c = nums(line);
        } else if line.starts_with("b ") {
            b = nums(line);
        } else if line.starts_with("lb ") {
            lb = nums(line);
        } else if line.starts_with("ub ") {
            ub = nums(line);
        } else if line.starts_with("A ") {
            let p: Vec<&str> = line.split_whitespace().collect();
            a.push(Triplet::new(
                p[1].parse().unwrap(),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
            ));
        }
    }
    LpProblem {
        n,
        m,
        c,
        a,
        b,
        lb,
        ub,
    }
}

/// `(min xᵢ, max xᵢ)` reference values from HiGHS for the first three structural
/// variables of the captured relaxation.
const REFERENCE: [(f64, f64); 3] = [(1.0, 2.0), (1.0, 4.0), (2.5, 10.0)];

// Previously known-failing: the dense explicit basis inverse was too
// ill-conditioned on this 6.3e8-dynamic-range LP (max x0 returned the lower
// bound 1 instead of 2), and geometric equilibration alone did not rescue it.
// The Phase 6.2 factored sparse LU (faer) — the representation commercial
// simplex codes use — fixes it, so this is now a live regression guard.
#[test]
fn ex4_1_2_relaxation_min_max_match_highs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ex4_1_2_relax.lp");
    let lp = parse_lp(&path);

    // Warm sweep, exactly as OBBT drives it: prime once, then flip the objective
    // to min/max each variable from the previous optimal basis.
    let mut warm = Simplex::new(&lp);
    assert_eq!(warm.solve().status, LpStatus::Optimal, "prime solve");

    let mut c = vec![0.0; lp.n];
    for (i, &(want_min, want_max)) in REFERENCE.iter().enumerate() {
        c[i] = 1.0;
        let smin = warm.solve_objective(&c);
        c[i] = -1.0;
        let smax = warm.solve_objective(&c);
        c[i] = 0.0;

        assert_eq!(smin.status, LpStatus::Optimal, "var {i} min status");
        assert_eq!(smax.status, LpStatus::Optimal, "var {i} max status");
        // smin.obj = min xᵢ; smax.obj = min(−xᵢ) = −max xᵢ.
        let got_min = smin.obj;
        let got_max = -smax.obj;
        assert!(
            (got_min - want_min).abs() < 1e-5,
            "var {i} min: got {got_min} want {want_min}"
        );
        assert!(
            (got_max - want_max).abs() < 1e-5,
            "var {i} max: got {got_max} want {want_max} (the ill-scaling bug returned the lower bound here)"
        );
    }
}

#[test]
fn ex4_1_2_cold_solves_match_warm() {
    // The bug also reproduced cold (fresh solver per objective), so guard that
    // path too: a cold max of var 0 must reach 2, not the lower bound 1.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ex4_1_2_relax.lp");
    let lp = parse_lp(&path);
    let mut p = lp.clone();
    p.c.iter_mut().for_each(|v| *v = 0.0);
    p.c[0] = -1.0; // maximize x0
    let s = Simplex::new(&p).solve();
    assert_eq!(s.status, LpStatus::Optimal);
    assert!(
        (-s.obj - 2.0).abs() < 1e-5,
        "cold max x0 = {} (want 2)",
        -s.obj
    );
}
