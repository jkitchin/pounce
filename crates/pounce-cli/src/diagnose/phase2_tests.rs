//! Phase 2 (gh#1160) acceptance: each check is driven down every branch it
//! can take, on small in-memory models whose answer is known by hand.

use super::report::{Report, Severity};
use super::*;
use crate::nl_reader::{BinOp, Expr, NlProblemParts, UnaryOp};

fn v(j: usize) -> Expr {
    Expr::Var(j)
}
fn c(x: Number) -> Expr {
    Expr::Const(x)
}
fn add(a: Expr, b: Expr) -> Expr {
    Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
}
fn sub(a: Expr, b: Expr) -> Expr {
    Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b))
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
}
fn sq(a: Expr) -> Expr {
    Expr::Binary(BinOp::Pow, Box::new(a), Box::new(c(2.0)))
}
fn neg(a: Expr) -> Expr {
    Expr::Unary(UnaryOp::Neg, Box::new(a))
}

struct M {
    obj: Expr,
    cons: Vec<Expr>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
}

fn build(m: M) -> NlProblem {
    let n = m.x_l.len();
    let k = m.cons.len();
    NlProblem::from_expressions(NlProblemParts {
        minimize: true,
        objective: m.obj,
        obj_constant: 0.0,
        constraints: m.cons,
        x_l: m.x_l,
        x_u: m.x_u,
        x0: vec![0.3; n],
        g_l: m.g_l,
        g_u: m.g_u,
        var_names: (0..n).map(|j| format!("x{j}")).collect(),
        con_names: (0..k).map(|i| format!("c{i}")).collect(),
    })
    .unwrap()
}

/// Diagnose at a solution `x` with Ipopt-convention multipliers `lambda`.
fn at_solution(p: &NlProblem, x: Vec<Number>, lambda: Vec<Number>) -> Report {
    let inputs = Inputs {
        point: Some(("the test solution".into(), x)),
        multipliers: Some(lambda),
        bound_multipliers: None,
        is_solution: true,
        extra_points: Vec::new(),
    };
    diagnose_problem(p, &inputs, "t".into(), &DiagnoseOptions::default()).unwrap()
}

fn has(r: &Report, code: &str) -> bool {
    r.findings.iter().any(|f| f.code == code)
}

fn codes(r: &Report) -> Vec<&'static str> {
    r.findings.iter().map(|f| f.code).collect()
}

const INF: Number = 1e19;

// ---------------------------------------------------------------- KKT / CQ

#[test]
fn licq_fails_while_mfcq_holds_and_multipliers_are_not_unique() {
    // min x0 + x1  s.t.  x0 + x1 >= 1,  2 x0 + 2 x1 >= 2. At (0.5, 0.5)
    // both are active with gradients (1,1) and (2,2): LICQ fails, but
    // d = (1, 1) strictly increases both, so MFCQ holds. Stationarity:
    // λ0 + 2 λ1 = -1, e.g. λ = (-0.5, -0.25), so μ = (0.5, 0.25). The
    // dependency, normalized to max |c| = 1, is (1, -1/2): μ0 + t ≥ 0 and
    // μ1 - t/2 ≥ 0 give t ∈ [-0.5, 0.5].
    let p = build(M {
        obj: add(v(0), v(1)),
        cons: vec![add(v(0), v(1)), add(mul(c(2.0), v(0)), mul(c(2.0), v(1)))],
        g_l: vec![1.0, 2.0],
        g_u: vec![INF, INF],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&p, vec![0.5, 0.5], vec![-0.5, -0.25]);
    assert!(!has(&r, "not_stationary_at_point"), "{:?}", codes(&r));
    assert!(has(&r, "licq_fails"), "{:?}", codes(&r));
    assert!(has(&r, "mfcq_holds"), "{:?}", codes(&r));
    assert!(!has(&r, "mfcq_fails"));
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "nonunique_multipliers")
        .expect("non-unique");
    let t = &f.evidence["t_range"];
    assert!((t[0].as_f64().unwrap() + 0.5).abs() < 1e-9, "{t}");
    assert!((t[1].as_f64().unwrap() - 0.5).abs() < 1e-9, "{t}");
    // The .sol sign convention (negated) must give the same answer.
    let r2 = at_solution(&p, vec![0.5, 0.5], vec![0.5, 0.25]);
    assert!(has(&r2, "nonunique_multipliers") && !has(&r2, "not_stationary_at_point"));
}

#[test]
fn independent_active_constraints_report_no_licq_finding() {
    // The negative branch: one active row, so LICQ holds and the
    // multiplier is unique; nothing is reported.
    let p = build(M {
        obj: add(v(0), v(1)),
        cons: vec![add(v(0), v(1))],
        g_l: vec![1.0],
        g_u: vec![INF],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&p, vec![0.5, 0.5], vec![-1.0]);
    assert!(
        !has(&r, "licq_fails") && !has(&r, "nonunique_multipliers"),
        "{:?}",
        codes(&r)
    );
    assert!(!has(&r, "mfcq_fails"));
}

#[test]
fn dependent_active_rows_with_pinned_multipliers_are_not_nonunique() {
    // min x0^2 + x1^2 s.t. x0 + x1 >= 0, 2 x0 + 2 x1 >= 0 at the origin:
    // LICQ fails, but both multipliers are 0 and the dependency (1, -1/2)
    // would push one of them negative either way, so t ∈ [0, 0].
    let p = build(M {
        obj: add(sq(v(0)), sq(v(1))),
        cons: vec![add(v(0), v(1)), add(mul(c(2.0), v(0)), mul(c(2.0), v(1)))],
        g_l: vec![0.0, 0.0],
        g_u: vec![INF, INF],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&p, vec![0.0, 0.0], vec![0.0, 0.0]);
    assert!(has(&r, "licq_fails"), "{:?}", codes(&r));
    assert!(!has(&r, "nonunique_multipliers"), "{:?}", codes(&r));
}

#[test]
fn a_one_sided_row_far_from_its_bound_is_inactive() {
    // min (x0 - 3)^2, x0 in [0, 1], row x0 >= 0. At x0 = 1 only the
    // variable's upper bound is active; the row's missing upper side must
    // not count as "active at +inf" (it did, and doubled every bound).
    let p = build(M {
        obj: sq(sub(v(0), c(3.0))),
        cons: vec![v(0)],
        g_l: vec![0.0],
        g_u: vec![INF],
        x_l: vec![0.0],
        x_u: vec![1.0],
    });
    let r = at_solution(&p, vec![1.0], vec![0.0]);
    assert!(!has(&r, "licq_fails"), "{:?}", codes(&r));
    assert_eq!(r.sections["kkt_audit"]["active_inequalities"], 1);
}

#[test]
fn mfcq_fails_where_inequalities_pinch_to_a_point() {
    // x1 >= x0^2 and x1 <= 0: the feasible set is {(0, 0)}. Their oriented
    // gradients there are (0, -1) and (0, 1), which cancel.
    let p = build(M {
        obj: add(sq(v(0)), sq(v(1))),
        cons: vec![sub(v(1), sq(v(0))), v(1)],
        g_l: vec![0.0, -INF],
        g_u: vec![INF, 0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&p, vec![0.0, 0.0], vec![0.0, 0.0]);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "mfcq_fails")
        .expect("MFCQ fails");
    assert_eq!(f.constraints.len(), 2, "{:?}", f.constraints);
}

#[test]
fn not_stationary_and_wrong_sign_are_reported() {
    // min x0 s.t. x0 >= 1 (a row): at x0 = 1 the Ipopt multiplier must be
    // -1. Give +1: stationarity fails for that sign, and the sign picker
    // flips it, so give a value no sign can fix: 0.25.
    let p = build(M {
        obj: v(0),
        cons: vec![v(0)],
        g_l: vec![1.0],
        g_u: vec![INF],
        x_l: vec![-INF],
        x_u: vec![INF],
    });
    let r = at_solution(&p, vec![1.0], vec![0.25]);
    assert!(has(&r, "not_stationary_at_point"), "{:?}", codes(&r));
    let ok = at_solution(&p, vec![1.0], vec![-1.0]);
    assert!(
        !has(&ok, "not_stationary_at_point") && !has(&ok, "multiplier_sign_violation"),
        "{:?}",
        codes(&ok)
    );
}

#[test]
fn zero_objective_sign_ambiguity_resolves_to_valid_signs() {
    // Feasibility problem: min 0 s.t. x0 + x1 <= 1, x0 - x1 <= 1, at the
    // vertex (1, 0) both rows are active. With ∇f = 0 any multipliers and
    // their negation are stationary; the valid (non-negative) reading must
    // win, for either sign the file uses.
    let p = build(M {
        obj: c(0.0),
        cons: vec![add(v(0), v(1)), sub(v(0), v(1))],
        g_l: vec![-INF, -INF],
        g_u: vec![1.0, 1.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    for lam in [vec![0.0, 0.0], vec![0.5, 0.5], vec![-0.5, -0.5]] {
        let r = at_solution(&p, vec![1.0, 0.0], lam.clone());
        assert!(
            !has(&r, "multiplier_sign_violation"),
            "{lam:?}: {:?}",
            codes(&r)
        );
    }
}

#[test]
fn weak_and_near_activity() {
    // min x0^2, x0 >= 0: at 0 the bound is active with a zero multiplier.
    let weak = build(M {
        obj: sq(v(0)),
        cons: vec![],
        g_l: vec![],
        g_u: vec![],
        x_l: vec![0.0],
        x_u: vec![INF],
    });
    let r = at_solution(&weak, vec![0.0], vec![]);
    assert!(has(&r, "weakly_active_constraints"), "{:?}", codes(&r));
    // min (x0+1)^2, x0 >= 0: the bound carries multiplier 2.
    let strong = build(M {
        obj: sq(add(v(0), c(1.0))),
        cons: vec![],
        g_l: vec![],
        g_u: vec![],
        x_l: vec![0.0],
        x_u: vec![INF],
    });
    let r = at_solution(&strong, vec![0.0], vec![]);
    assert!(!has(&r, "weakly_active_constraints"), "{:?}", codes(&r));
    assert!(!has(&r, "near_active_inequalities"));
    // min (x0 - 5e-5)^2, x0 >= 0: the solution sits 5e-5 from the bound,
    // inside the 1e-4 band but outside the activity tolerance.
    let near = build(M {
        obj: sq(sub(v(0), c(5e-5))),
        cons: vec![],
        g_l: vec![],
        g_u: vec![],
        x_l: vec![0.0],
        x_u: vec![INF],
    });
    let r = at_solution(&near, vec![5e-5], vec![]);
    assert!(has(&r, "near_active_inequalities"), "{:?}", codes(&r));
}

#[test]
fn kkt_checks_need_a_solution() {
    let p = build(M {
        obj: sq(v(0)),
        cons: vec![],
        g_l: vec![],
        g_u: vec![],
        x_l: vec![-INF],
        x_u: vec![INF],
    });
    let r = diagnose_problem(
        &p,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    for c in ["kkt_audit", "licq", "mfcq", "reduced_hessian"] {
        assert!(
            r.checks
                .iter()
                .any(|k| k.name == c && matches!(k.status, report::CheckStatus::Unavailable(_))),
            "{c} should be unavailable without --sol"
        );
    }
}

// --------------------------------------------------------- second order

#[test]
fn reduced_hessian_negative_curvature_and_sufficiency() {
    // min -(x0^2 + x1^2) s.t. x0 + x1 = 0 at (0, 0): along (1, -1) the
    // objective falls. λ = 0.
    let saddle = build(M {
        obj: neg(add(sq(v(0)), sq(v(1)))),
        cons: vec![add(v(0), v(1))],
        g_l: vec![0.0],
        g_u: vec![0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&saddle, vec![0.0, 0.0], vec![0.0]);
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "reduced_hessian_negative_curvature")
        .expect("found");
    assert_eq!(f.severity, Severity::Warning);
    // min x0^2 + x1^2 s.t. x0 + x1 = 1 at (0.5, 0.5), λ = -1.
    let bowl = build(M {
        obj: add(sq(v(0)), sq(v(1))),
        cons: vec![add(v(0), v(1))],
        g_l: vec![1.0],
        g_u: vec![1.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = at_solution(&bowl, vec![0.5, 0.5], vec![-1.0]);
    assert!(has(&r, "second_order_sufficient"), "{:?}", codes(&r));
    assert!(!has(&r, "reduced_hessian_negative_curvature"));
}

// ------------------------------------------------------ rank robustness

fn with_extra(p: &NlProblem, x: Vec<Number>, extra: Vec<Number>) -> Report {
    let inputs = Inputs {
        point: Some(("the main point".into(), x)),
        multipliers: None,
        bound_multipliers: None,
        is_solution: false,
        extra_points: vec![("the extra point".into(), extra)],
    };
    diagnose_problem(p, &inputs, "t".into(), &DiagnoseOptions::default()).unwrap()
}

#[test]
fn rank_loss_persistent_versus_point_specific() {
    // Duplicate equalities: deficient everywhere.
    let dup = build(M {
        obj: c(0.0),
        cons: vec![add(v(0), v(1)), add(mul(c(2.0), v(0)), mul(c(2.0), v(1)))],
        g_l: vec![1.0, 2.0],
        g_u: vec![1.0, 2.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = with_extra(&dup, vec![0.5, 0.5], vec![3.0, -1.0]);
    assert!(has(&r, "rank_loss_persistent"), "{:?}", codes(&r));
    // x0*x1 = 0, x0 + x1 = 0: Jacobian [[x1, x0], [1, 1]] is singular only
    // where x0 == x1.
    let val = build(M {
        obj: c(0.0),
        cons: vec![mul(v(0), v(1)), add(v(0), v(1))],
        g_l: vec![0.0, 0.0],
        g_u: vec![0.0, 0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = with_extra(&val, vec![2.0, 2.0], vec![1.0, 3.0]);
    assert!(has(&r, "rank_loss_point_specific"), "{:?}", codes(&r));
    assert!(!has(&r, "rank_loss_persistent"));
}

#[test]
fn rank_loss_that_disappears_under_scaling_or_tolerance() {
    // e1 = 1e-14 (x0 - x1): independent of e0 = x0 + x1, but tiny. Its
    // singular value sits between the default threshold /100 and ×100, and
    // equilibration restores it.
    let p = build(M {
        obj: c(0.0),
        cons: vec![add(v(0), v(1)), mul(c(1e-14), sub(v(0), v(1)))],
        g_l: vec![1.0, 0.0],
        g_u: vec![1.0, 0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &p,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    assert!(has(&r, "rank_tolerance_sensitive"), "{:?}", codes(&r));
    // 1e-20: deficient at every tolerance in the sweep, yet equilibration
    // shows the rows are independent.
    let q = build(M {
        obj: c(0.0),
        cons: vec![add(v(0), v(1)), mul(c(1e-20), sub(v(0), v(1)))],
        g_l: vec![1.0, 0.0],
        g_u: vec![1.0, 0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &q,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    assert!(has(&r, "rank_scaling_sensitive"), "{:?}", codes(&r));
    // A well-conditioned pair is robust and says so.
    let ok = build(M {
        obj: c(0.0),
        cons: vec![add(v(0), v(1)), sub(v(0), v(1))],
        g_l: vec![1.0, 0.0],
        g_u: vec![1.0, 0.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &ok,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "rank_robustness_summary")
        .unwrap();
    assert!(
        f.observation.contains("full rank, robust"),
        "{}",
        f.observation
    );
    assert!(!has(&r, "rank_tolerance_sensitive") && !has(&r, "rank_scaling_sensitive"));
}

// ------------------------------------------------------------ convexity

#[test]
fn convex_model_is_inconclusive_and_nonconvex_has_a_replayable_witness() {
    // min x0^2 + x1^2 s.t. x0^2 + x1^2 <= 4: convex, so no counterexample;
    // the summary must not call it convex.
    let convex = build(M {
        obj: add(sq(v(0)), sq(v(1))),
        cons: vec![add(sq(v(0)), sq(v(1)))],
        g_l: vec![-INF],
        g_u: vec![4.0],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &convex,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    assert!(!has(&r, "convexity_counterexample"), "{:?}", codes(&r));
    let s = r
        .findings
        .iter()
        .find(|f| f.code == "convexity_summary")
        .unwrap();
    assert!(s.observation.contains("inconclusive"), "{}", s.observation);
    // ... s.t. x0^2 + x1^2 >= 1: a convex function bounded below is a
    // nonconvex set.
    let ring = build(M {
        obj: add(sq(v(0)), sq(v(1))),
        cons: vec![add(sq(v(0)), sq(v(1)))],
        g_l: vec![1.0],
        g_u: vec![INF],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &ring,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "convexity_counterexample")
        .expect("witness");
    assert_eq!(f.constraints, vec!["c0".to_string()]);
    // Replay the witness.
    let e = &f.evidence;
    let t = e["t"].as_f64().unwrap();
    let xs: Vec<Number> = e["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let ys: Vec<Number> = e["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let g = |p: &[Number]| p[0] * p[0] + p[1] * p[1];
    let z: Vec<Number> = xs
        .iter()
        .zip(&ys)
        .map(|(a, b)| t * a + (1.0 - t) * b)
        .collect();
    assert!(
        t * g(&xs) + (1.0 - t) * g(&ys) - g(&z) > 0.0,
        "not concave: chord above"
    );
    // Same seed, same witness.
    let r2 = diagnose_problem(
        &ring,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    let f2 = r2
        .findings
        .iter()
        .find(|f| f.code == "convexity_counterexample")
        .unwrap();
    assert_eq!(f.evidence, f2.evidence);
}

#[test]
fn objective_hessian_inertia() {
    let p = build(M {
        obj: sub(sq(v(0)), sq(v(1))),
        cons: vec![],
        g_l: vec![],
        g_u: vec![],
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
    });
    let r = diagnose_problem(
        &p,
        &Inputs::default(),
        "t".into(),
        &DiagnoseOptions::default(),
    )
    .unwrap();
    let f = r
        .findings
        .iter()
        .find(|f| f.code == "hessian_inertia")
        .unwrap();
    assert_eq!(f.evidence["positive"], 1);
    assert_eq!(f.evidence["negative"], 1);
}
