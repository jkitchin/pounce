//! gh #293 — convex-QP scaling regression corpus (ignored by default).
//!
//!   cargo test -p pounce-convex --test bench293 -- --ignored --nocapture
//!
//! Solves a spread of convex QP/LP instances spanning the scaling regimes that
//! stress the HSDE driver — well-scaled, huge-magnitude (#286), mixed-scale
//! (#293 symptom 1), and uniformly/nearly-tiny curvature (#293 symptom 2) —
//! plus random SPD box/equality QPs and pure LPs. The diagonal box QPs carry a
//! solver-independent oracle (`x*_i = clamp(target_i, lb, ub)`), so their
//! optima are checked absolutely.
//!
//! This is the corpus proxy that backed the #293 prototype investigation
//! (proactive P-Ruiz and c-keyed objective scaling vs. the shipped reactive
//! fallback). The proactive variants reached the same 15/15 oracle correctness
//! but perturbed ~75% of already-converged problems (bit-identical objective on
//! only 3–7 of 26 vs. the fallback's 23/26) and inflated the huge-magnitude
//! regime from ~8 to ~18 iterations — so the surgical reactive fallback was
//! kept. This harness remains as a regression guard: the shipped solver must
//! reach every oracle optimum here, and its iteration counts should stay flat.

use pounce_convex::{QpOptions, QpProblem, QpStatus, Triplet, solve_qp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
    let mut px = vec![0.0; prob.n];
    prob.p_mul_add_pub(x, &mut px);
    (0..prob.n)
        .map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i])
        .sum()
}

/// Deterministic LCG so the corpus is reproducible without an external rng.
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn sym(&mut self) -> f64 {
        2.0 * self.next_f64() - 1.0
    }
}

struct Case {
    name: String,
    prob: QpProblem,
    /// Exact optimal objective if a closed-form oracle exists.
    oracle_obj: Option<f64>,
}

/// Diagonal box QP: eig_i = scale·cond^(i/(n-1)), c_i = -eig_i·tgt_i, box
/// [lb,ub]. Separable ⇒ x*_i = clamp(tgt_i, lb, ub) is an exact oracle.
fn diag_box(name: &str, scale: f64, cond: f64, tgt: &[f64], lb: f64, ub: f64) -> Case {
    let n = tgt.len();
    let mut p_lower = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut xopt = vec![0.0; n];
    for (i, &t) in tgt.iter().enumerate() {
        let eig = scale * cond.powf(i as f64 / (n.max(2) - 1) as f64);
        p_lower.push(Triplet::new(i, i, eig));
        c.push(-eig * t);
        xopt[i] = t.clamp(lb, ub);
    }
    let prob = QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![lb; n],
        ub: vec![ub; n],
    };
    let oracle = objective(&prob, &xopt);
    Case {
        name: name.to_string(),
        prob,
        oracle_obj: Some(oracle),
    }
}

/// Dense random SPD QP (P = MᵀM + εI) with a box; no oracle.
fn rand_spd_box(name: &str, n: usize, seed: u64, pscale: f64, cscale: f64) -> Case {
    let mut r = Lcg(seed);
    let m: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| r.sym()).collect()).collect();
    let mut p_lower = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            let mut v: f64 = (0..n).map(|k| m[k][i] * m[k][j]).sum();
            v *= pscale;
            if i == j {
                v += pscale;
            }
            if v != 0.0 {
                p_lower.push(Triplet::new(i, j, v));
            }
        }
    }
    let c: Vec<f64> = (0..n).map(|_| cscale * r.sym()).collect();
    let prob = QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0; n],
        ub: vec![1.0; n],
    };
    Case {
        name: name.to_string(),
        prob,
        oracle_obj: None,
    }
}

/// Random SPD QP with an equality constraint sum(x)=s and x>=0.
fn rand_spd_eq(name: &str, n: usize, seed: u64, s: f64) -> Case {
    let mut base = rand_spd_box(name, n, seed, 1.0, 1.0);
    base.prob.lb = vec![0.0; n];
    base.prob.ub = vec![f64::INFINITY; n];
    base.prob.a = (0..n).map(|j| Triplet::new(0, j, 1.0)).collect();
    base.prob.b = vec![s];
    base
}

/// Pure LP: min cᵀx s.t. box.
fn lp_box(name: &str, n: usize, seed: u64, cscale: f64) -> Case {
    let mut r = Lcg(seed);
    let c: Vec<f64> = (0..n).map(|_| cscale * r.sym()).collect();
    let prob = QpProblem {
        n,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g: vec![],
        h: vec![],
        lb: vec![-1.0; n],
        ub: vec![1.0; n],
    };
    Case {
        name: name.to_string(),
        prob,
        oracle_obj: None,
    }
}

fn corpus() -> Vec<Case> {
    let mut v = Vec::new();
    let tgt6 = [1.5, -1.5, 0.3, -0.7, 2.0, -0.4];
    let tgt3 = [1e11, -0.5, 2.0];

    for &cond in &[1.0, 1e2, 1e4, 1e6] {
        v.push(diag_box(
            &format!("diag_well_cond{cond:e}"),
            1.0,
            cond,
            &tgt6,
            -1.0,
            1.0,
        ));
    }
    for &sc in &[1e6, 1e12, 1e18] {
        v.push(diag_box(
            &format!("diag_huge_{sc:e}"),
            sc,
            1.0,
            &tgt6,
            -1.0,
            1.0,
        ));
    }
    for &sc in &[1e-6, 1e-9, 1e-12, 1e-14, 1e-16] {
        v.push(diag_box(
            &format!("diag_tiny_{sc:e}"),
            sc,
            1.0,
            &tgt3,
            0.0,
            f64::INFINITY,
        ));
    }
    for &cond in &[1e12, 1e18] {
        v.push(diag_box(
            &format!("diag_mixed_cond{cond:e}"),
            1e-6,
            cond,
            &tgt3,
            0.0,
            f64::INFINITY,
        ));
    }
    v.push(diag_box(
        "diag_tiny_boundbox",
        1e-12,
        1.0,
        &[1e11, 1e11],
        0.0,
        1e6,
    ));

    for (i, &n) in [5usize, 10, 25, 50].iter().enumerate() {
        v.push(rand_spd_box(
            &format!("rand_box_n{n}"),
            n,
            1000 + i as u64,
            1.0,
            1.0,
        ));
    }
    v.push(rand_spd_box("rand_box_pbig", 20, 7, 1e6, 1.0));
    v.push(rand_spd_box("rand_box_cbig", 20, 8, 1.0, 1e6));
    v.push(rand_spd_box("rand_box_ptiny", 20, 9, 1e-8, 1.0));

    for (i, &n) in [10usize, 30].iter().enumerate() {
        v.push(rand_spd_eq(
            &format!("rand_eq_n{n}"),
            n,
            2000 + i as u64,
            3.0,
        ));
    }

    v.push(lp_box("lp_small", 10, 11, 1.0));
    v.push(lp_box("lp_cbig", 15, 12, 1e6));

    v
}

#[test]
#[ignore]
fn scaling_corpus_reaches_every_oracle_optimum() {
    let short = |s: QpStatus| match s {
        QpStatus::Optimal => "OPT",
        QpStatus::OptimalInaccurate => "OPT~",
        QpStatus::IterationLimit => "ITER",
        QpStatus::NumericalFailure => "NUMF",
        QpStatus::PrimalInfeasible => "PINF",
        QpStatus::DualInfeasible => "DINF",
    };

    let cases = corpus();
    let mut oracle_n = 0usize;
    let mut oracle_ok = 0usize;
    let mut clean = 0usize;
    let mut iters_tot = 0usize;
    let mut failures: Vec<String> = Vec::new();

    println!(
        "\n{:<26} {:>3} {:>6} {:>10} {:>6}",
        "case", "n", "status", "obj", "iters"
    );
    for c in &cases {
        let sol = solve_qp_ipm(&c.prob, &QpOptions::default(), backend);
        iters_tot += sol.iters;
        if sol.status == QpStatus::Optimal {
            clean += 1;
        }
        println!(
            "{:<26} {:>3} {:>6} {:>10.3e} {:>6}",
            c.name,
            c.prob.n,
            short(sol.status),
            sol.obj,
            sol.iters
        );
        if let Some(o) = c.oracle_obj {
            oracle_n += 1;
            let denom = o.abs().max(1.0);
            if (sol.obj - o).abs() / denom < 1e-4 {
                oracle_ok += 1;
            } else {
                failures.push(format!(
                    "{}: obj {:.6e} != oracle {:.6e} (status {:?})",
                    c.name, sol.obj, o, sol.status
                ));
            }
        }
    }

    println!(
        "\nsummary: oracle-correct {oracle_ok}/{oracle_n}, clean-Optimal {clean}/{}, total-iters {iters_tot}",
        cases.len()
    );
    for f in &failures {
        println!("  FAIL {f}");
    }
    assert!(
        failures.is_empty(),
        "{} oracle instance(s) missed their known optimum",
        failures.len()
    );
    assert_eq!(
        oracle_ok, oracle_n,
        "every oracle instance must be solved exactly"
    );
}
