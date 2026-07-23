//! Status-labelling regression tests for the non-symmetric (exp / power) HSDE
//! driver (gh #329).
//!
//! Correctly-solved geometric-program / power-cone problems whose barrier
//! Hessian conditions poorly near the cone boundary used to floor their
//! *absolute* KKT residual a hair above `tol` and be reported
//! `OptimalInaccurate` / `success=False`, even though the objective and
//! minimiser matched the reference solvers (ECOS/SCS/Clarabel) to ~1e-7. The
//! driver now re-adjudicates such a stalled iterate against the *scale-relative*
//! true conic KKT residual (with `ẑ ∈ K*` required) and certifies `Optimal`
//! when it is genuinely tight. These tests pin both the promoted `Optimal`
//! labels **and** — critically — that infeasible / unbounded conic solves are
//! unchanged (never falsely promoted).

use pounce_convex::{ConeSpec, QpOptions, QpProblem, QpStatus, Triplet, solve_socp_ipm};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

fn solve(prob: &QpProblem, cones: &[ConeSpec]) -> pounce_convex::QpSolution {
    let opts = QpOptions::default();
    solve_socp_ipm(prob, cones, &opts, backend)
}

/// Append a 3-row exp block enforcing `t_col ≥ exp(sign · x[lin_col])`, in the
/// `solve_socp` convention `s = (s0, s1, s2)` with `s1·exp(s0/s1) ≤ s2`,
/// `s0 = sign·x[lin_col]`, `s1 = 1`, `s2 = x[t_col]`.
fn exp_block(lin_col: usize, sign: f64, t_col: usize, g: &mut Vec<Triplet>, h: &mut Vec<f64>) {
    let r0 = h.len();
    g.push(Triplet::new(r0, lin_col, -sign));
    h.push(0.0);
    h.push(1.0); // row r0+1 all-zero ⇒ s1 = 1
    let r2 = h.len();
    g.push(Triplet::new(r2, t_col, -1.0));
    h.push(0.0);
}

/// gh #329 case 1: separable GP `min x + 4/x + y + 9/y`, four exp cones with
/// non-unit coefficients. Known optimum `10` at `(x, y) = (2, 3)`. Was
/// `OptimalInaccurate`; must now be `Optimal` with the same correct objective.
#[test]
fn exp_gp_four_cones_reports_optimal() {
    // vars z = [u, v, t1, t2, t3, t4], with x = e^u, y = e^v.
    let (iu, iv, it1, it2, it3, it4) = (0, 1, 2, 3, 4, 5);
    let mut c = vec![0.0; 6];
    c[it1] = 1.0;
    c[it2] = 4.0;
    c[it3] = 1.0;
    c[it4] = 9.0;
    let mut g = Vec::new();
    let mut h = Vec::new();
    exp_block(iu, 1.0, it1, &mut g, &mut h); // t1 ≥ e^{u}
    exp_block(iu, -1.0, it2, &mut g, &mut h); // t2 ≥ e^{-u}
    exp_block(iv, 1.0, it3, &mut g, &mut h); // t3 ≥ e^{v}
    exp_block(iv, -1.0, it4, &mut g, &mut h); // t4 ≥ e^{-v}
    let prob = QpProblem {
        n: 6,
        p_lower: vec![],
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Exponential; 4]);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "correctly-solved GP must report Optimal, got {:?}",
        sol.status
    );
    // Objective and minimiser correct to the cone's achievable floor (~1e-7).
    assert!(
        (sol.obj - 10.0).abs() < 1e-5,
        "objective {} must match known optimum 10",
        sol.obj
    );
    let (x_star, y_star) = (sol.x[0].exp(), sol.x[1].exp());
    assert!((x_star - 2.0).abs() < 1e-3, "x* = {x_star} vs 2");
    assert!((y_star - 3.0).abs() < 1e-3, "y* = {y_star} vs 3");
}

/// The simpler two-cone GP `min x + 1/x` (optimum `2`) already reported a clean
/// `Optimal`; guard that the change leaves it — and its objective — untouched.
#[test]
fn exp_gp_two_cones_stays_optimal() {
    // vars [u, t1, t2], x = e^u.
    let mut g = Vec::new();
    let mut h = Vec::new();
    exp_block(0, 1.0, 1, &mut g, &mut h); // t1 ≥ e^{u}
    exp_block(0, -1.0, 2, &mut g, &mut h); // t2 ≥ e^{-u}
    let prob = QpProblem {
        n: 3,
        p_lower: vec![],
        c: vec![0.0, 1.0, 1.0],
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Exponential; 2]);
    assert_eq!(sol.status, QpStatus::Optimal, "got {:?}", sol.status);
    assert!((sol.obj - 2.0).abs() < 1e-6, "objective {} vs 2", sol.obj);
}

/// gh #329 case 2: `max cᵀx s.t. ‖x‖₄ ≤ 1`, four power cones `K_{1/4}`. Known
/// optimum `‖c‖_{4/3}` (Hölder). Was `OptimalInaccurate`; must now be `Optimal`.
#[test]
fn power_dualnorm_p4_reports_optimal() {
    let p = 4.0;
    let alpha = 1.0 / p;
    let c_obj: [f64; 4] = [1.0, -2.0, 3.0, 0.5];
    let q = p / (p - 1.0);
    let known: f64 = c_obj
        .iter()
        .map(|v| v.abs().powf(q))
        .sum::<f64>()
        .powf(1.0 / q); // ‖c‖_{4/3}
    let n = 4;
    let nv = 2 * n; // [x0..3, z0..3]
    let mut c = vec![0.0; nv];
    for i in 0..n {
        c[i] = -c_obj[i]; // min −cᵀx
    }
    let mut g = Vec::new();
    let mut h = Vec::new();
    for i in 0..n {
        let r0 = h.len();
        g.push(Triplet::new(r0, i, -1.0)); // s0 = x_i
        h.push(0.0);
        let r1 = h.len();
        g.push(Triplet::new(r1, n + i, -1.0)); // s1 = z_i
        h.push(0.0);
        h.push(1.0); // s2 = 1 (const)
    }
    // Equality Σ z_i = 1.
    let a: Vec<Triplet> = (0..n).map(|i| Triplet::new(0, n + i, 1.0)).collect();
    let prob = QpProblem {
        n: nv,
        p_lower: vec![],
        c,
        a,
        b: vec![1.0],
        g,
        h,
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Power(alpha); 4]);
    assert_eq!(
        sol.status,
        QpStatus::Optimal,
        "correctly-solved power-ball must report Optimal, got {:?}",
        sol.status
    );
    // Recovered max value = cᵀx = −obj.
    let max_val = -sol.obj;
    assert!(
        (max_val - known).abs() < 1e-5,
        "max value {max_val} must match ‖c‖_(4/3) = {known}"
    );
    // And the recovered point is feasible: ‖x‖₄ ≤ 1.
    let norm4: f64 = (0..n)
        .map(|i| sol.x[i].abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p);
    assert!(norm4 <= 1.0 + 1e-6, "‖x‖₄ = {norm4} must be ≤ 1");
}

// ---- NEGATIVE tests: infeasible / unbounded conic solves are UNCHANGED and
// are never falsely promoted to Optimal by the new adjudication. ----

/// An unbounded exp program (`min u` with `t ≥ e^u`, so `u → −∞`) must still
/// certify `DualInfeasible` — never `Optimal`.
#[test]
fn unbounded_exp_still_dual_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 0.0], // min u
        a: vec![],
        b: vec![],
        g: vec![Triplet::new(0, 0, -1.0), Triplet::new(2, 1, -1.0)],
        h: vec![0.0, 1.0, 0.0], // slack = (u, 1, t)
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Exponential]);
    assert_eq!(
        sol.status,
        QpStatus::DualInfeasible,
        "unbounded exp program must stay DualInfeasible, got {:?}",
        sol.status
    );
    assert_ne!(sol.status, QpStatus::Optimal);
}

/// A power program whose cone `y`-slack is forced strictly negative (domain
/// violation `t − 2 ≤ T − 2 < 0`) must still certify `PrimalInfeasible`.
#[test]
fn infeasible_power_domain_still_primal_infeasible() {
    let prob = QpProblem {
        n: 2,
        p_lower: vec![],
        c: vec![1.0, 0.0], // min w
        a: vec![],
        b: vec![],
        g: vec![
            Triplet::new(0, 0, -1.0), // s0 = w
            Triplet::new(1, 1, -1.0), // s1 = t − 2
            Triplet::new(3, 1, 1.0),  // s3 = T − t ≥ 0
        ],
        h: vec![0.0, -2.0, 1.0, 1.0], // s2 = 1 (const), T = 1
        lb: vec![],
        ub: vec![],
    };
    let sol = solve(&prob, &[ConeSpec::Power(0.5), ConeSpec::Nonneg(1)]);
    assert_eq!(
        sol.status,
        QpStatus::PrimalInfeasible,
        "power domain violation must stay PrimalInfeasible, got {:?}",
        sol.status
    );
    assert_ne!(sol.status, QpStatus::Optimal);
}
