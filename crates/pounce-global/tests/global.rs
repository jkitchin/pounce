//! End-to-end spatial branch-and-bound on classic nonconvex problems.

use pounce_feral::FeralSolverInterface;
use pounce_global::{expr::var, solve_global, GlobalOptions, GlobalProblem, GlobalStatus};
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

#[test]
fn unconstrained_quartic_two_global_minima() {
    // f(x) = x⁴ − 3x² on [−2, 2]: global minimum −9/4 at x = ±√(3/2).
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 2.25).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0].abs() - 1.224_744_9).abs() < 1e-2,
        "x = {}",
        sol.x[0]
    );
    // Certified bound brackets the optimum.
    assert!(sol.lower_bound <= sol.objective + 1e-6);
}

#[test]
fn bilinear_box_min() {
    // f(x, y) = x·y on [−1, 1]²: global minimum −1 at (1, −1) or (−1, 1).
    let f = var(0) * var(1);
    let prob = GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!((sol.x[0] * sol.x[1] + 1.0).abs() < 1e-2, "x = {:?}", sol.x);
}

#[test]
fn nonconvex_equality_constraint() {
    // min x² + y²  s.t.  x·y = 1,  (x, y) ∈ [0.1, 10]².  Optimum 2 at (1, 1).
    let obj = var(0).powi(2) + var(1).powi(2);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![0.1, 0.1], vec![10.0, 10.0], &obj).equality(&g, 1.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 2.0).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.0).abs() < 5e-2 && (sol.x[1] - 1.0).abs() < 5e-2,
        "x = {:?}",
        sol.x
    );
}

#[test]
fn nonconvex_inequality_feasible_region() {
    // min x + y  s.t.  x·y ≥ 4,  (x, y) ∈ [1, 5]².  Optimum 4 at (2, 2).
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 4.0).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
}

#[test]
fn infeasible_is_detected() {
    // x·y ≥ 100 is unreachable on [0, 1]² (max product 1).
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![0.0, 0.0], vec![1.0, 1.0], &obj).ge(&g, 100.0);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Infeasible, "{sol:?}");
}

#[test]
fn six_hump_camel_global_minimum() {
    // f(x,y) = (4 − 2.1x² + x⁴/3)x² + xy + (−4 + 4y²)y²
    //        = 4x² − 2.1x⁴ + x⁶/3 + xy − 4y² + 4y⁴.
    // Six local minima; two global ones at (±0.0898, ∓0.7126), value ≈ −1.0316.
    let x = var(0);
    let y = var(1);
    let f = 4.0 * x.clone().powi(2) - 2.1 * x.clone().powi(4)
        + (1.0 / 3.0) * x.clone().powi(6)
        + x.clone() * y.clone()
        - 4.0 * y.clone().powi(2)
        + 4.0 * y.powi(4);
    let prob = GlobalProblem::new(vec![-2.0, -1.5], vec![2.0, 1.5], &f);
    let opts = GlobalOptions {
        abs_gap: 1e-4,
        rel_gap: 1e-4,
        max_nodes: 200_000,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    eprintln!(
        "camel: status={:?} obj={} lb={} nodes={} x={:?}",
        sol.status, sol.objective, sol.lower_bound, sol.nodes, sol.x
    );
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - (-1.031_628_5)).abs() < 1e-2,
        "obj = {}",
        sol.objective
    );
    // One of the two global minimizers.
    assert!(
        sol.x[0].abs() < 0.2 && sol.x[1].abs() > 0.5,
        "x = {:?}",
        sol.x
    );
}

#[test]
fn local_nlp_upper_bounds_toggle() {
    // min x + y  s.t.  x·y ≥ 4 on [1, 5]² (optimum 4 at (2, 2)). Solve with the
    // local NLP polish on (default) and off — both must certify the global
    // optimum, exercising the tape→TNLP bridge against the relaxation-only path.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let with_nlp = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(with_nlp.status, GlobalStatus::Optimal, "{with_nlp:?}");
    assert!(
        (with_nlp.objective - 4.0).abs() < 1e-3,
        "obj = {}",
        with_nlp.objective
    );
    // The NLP polish lands essentially on the true minimizer (2, 2).
    assert!(
        (with_nlp.x[0] - 2.0).abs() < 1e-2 && (with_nlp.x[1] - 2.0).abs() < 1e-2,
        "x = {:?}",
        with_nlp.x
    );

    let no_nlp_opts = GlobalOptions {
        local_solve_iters: 0,
        ..GlobalOptions::default()
    };
    let without = solve_global(&prob, &no_nlp_opts, backend);
    assert_eq!(without.status, GlobalStatus::Optimal, "{without:?}");
    assert!(
        (without.objective - 4.0).abs() < 1e-2,
        "obj = {}",
        without.objective
    );
}

#[test]
fn odd_power_straddling_zero() {
    // f(x) = x³ − 3x on [−2, 2]: critical points x = ±1, endpoints ±2.
    // Global minimum −2 (attained at x = 1 and x = −2). The cube term straddles
    // zero, so this needs the single-inflection envelope (previously box-only).
    let f = var(0).powi(3) - 3.0 * var(0);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 2.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
}

#[test]
fn sine_global_minimum() {
    // min sin(x) on [0, 6]: global minimum −1 at x = 3π/2 ≈ 4.712. The root box
    // is wider than π (sin envelope declines, box bound only) — branching must
    // narrow the box until the trig envelope engages, then certify.
    let f = var(0).sin();
    let prob = GlobalProblem::new(vec![0.0], vec![6.0], &f);
    let opts = GlobalOptions {
        max_nodes: 50_000,
        ..GlobalOptions::default()
    };
    let sol = solve_global(&prob, &opts, backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective + 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(
        (sol.x[0] - 1.5 * std::f64::consts::PI).abs() < 1e-2,
        "x = {}",
        sol.x[0]
    );
}

#[test]
fn sandwich_cuts_toggle() {
    // x⁴ − 3x² on [−2, 2] (global min −2.25). Solve with cutting-plane rounds
    // on (default) and off — both must certify the global optimum, exercising
    // the validity of the sandwich tangent cuts.
    let f = var(0).powi(4) - 3.0 * var(0).powi(2);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            sandwich_rounds: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 2.25).abs() < 1e-3,
            "obj = {}",
            sol.objective
        );
    }
}

#[test]
fn obbt_reduces_nodes() {
    // min x + y s.t. x·y ≥ 4 on [1, 5]² (optimum 4 at (2, 2)). OBBT with the
    // incumbent cutoff tightens the box aggressively; both settings certify the
    // optimum and OBBT visits no more nodes.
    let obj = var(0) + var(1);
    let g = var(0) * var(1);
    let prob = GlobalProblem::new(vec![1.0, 1.0], vec![5.0, 5.0], &obj).ge(&g, 4.0);

    let with_obbt = solve_global(&prob, &GlobalOptions::default(), backend);
    let without = solve_global(
        &prob,
        &GlobalOptions {
            obbt_passes: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&with_obbt, &without] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective - 4.0).abs() < 1e-2,
            "obj = {}",
            sol.objective
        );
    }
    assert!(
        with_obbt.nodes <= without.nodes,
        "OBBT nodes {} should be ≤ {} without",
        with_obbt.nodes,
        without.nodes
    );
}

#[test]
fn alphabb_cuts_toggle() {
    // f(x, y) = x·y on [−1, 1]² (global min −1). The objective is nonconvex
    // (indefinite Hessian), so αBB applies a positive spectral shift. Solve with
    // αBB cuts on (default) and off — both certify the optimum, exercising the
    // interval-Hessian / spectral-shift path and the validity of its cuts.
    let f = var(0) * var(1);
    let prob = GlobalProblem::new(vec![-1.0, -1.0], vec![1.0, 1.0], &f);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            alphabb_cuts: 0,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(
            (sol.objective + 1.0).abs() < 1e-3,
            "obj = {}",
            sol.objective
        );
    }
}

#[test]
fn rlt_affine_constraint_toggle() {
    // min x·y  s.t.  x + y = 4 (affine),  (x, y) ∈ [0, 4]². On the segment
    // xy = x(4−x) ∈ [0, 4], so the global minimum is 0 (at a segment end).
    // The affine equality drives RLT (linear constraint × bound factors); both
    // RLT on (default) and off must certify the optimum.
    let obj = var(0) * var(1);
    let g = var(0) + var(1);
    let prob = GlobalProblem::new(vec![0.0, 0.0], vec![4.0, 4.0], &obj).equality(&g, 4.0);

    let on = solve_global(&prob, &GlobalOptions::default(), backend);
    let off = solve_global(
        &prob,
        &GlobalOptions {
            rlt: false,
            ..GlobalOptions::default()
        },
        backend,
    );
    for sol in [&on, &off] {
        assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
        assert!(sol.objective.abs() < 1e-2, "obj = {}", sol.objective);
    }
}

#[test]
fn exp_log_atoms() {
    // min eˣ − x on [−2, 2]: convex, optimum 1 at x = 0 (exercises the exp
    // envelope through the global path).
    let f = var(0).exp() - var(0);
    let prob = GlobalProblem::new(vec![-2.0], vec![2.0], &f);
    let sol = solve_global(&prob, &GlobalOptions::default(), backend);
    assert_eq!(sol.status, GlobalStatus::Optimal, "{sol:?}");
    assert!(
        (sol.objective - 1.0).abs() < 1e-3,
        "obj = {}",
        sol.objective
    );
    assert!(sol.x[0].abs() < 1e-2, "x = {}", sol.x[0]);
}
