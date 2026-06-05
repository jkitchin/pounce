//! Extract a `pounce_convex::QpProblem` (standard form) from a parsed
//! `.nl` problem, for the LP/QP dispatch path (Phase 2).
//!
//! The classifier (`crate::dispatch`) has already decided the problem is
//! an LP or convex QP; this module marshals the parsed `NlProblem` into
//! the standard form the convex IPM consumes:
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equalities)
//!             G x ≤ h          (inequalities, incl. finite var bounds)
//! ```
//!
//! Mapping from the `.nl` representation:
//! - **Objective.** `P` is the Hessian of the (degree-≤2) objective —
//!   recovered with the same `analyze_quadratic` the classifier uses, so
//!   `P` here is exactly the matrix whose definiteness was tested. `c`
//!   is the objective's linear part. A `maximize` objective is negated
//!   into a minimization.
//! - **Constraints.** Each row has a linear part and bounds `g_l ≤ row ≤
//!   g_u`. An equality (`g_l == g_u`) becomes a row of `A`; a one- or
//!   two-sided inequality becomes one or two rows of `G` (`row ≤ g_u`
//!   and/or `−row ≤ −g_l`).
//! - **Variable bounds.** Finite `x_l`/`x_u` become `G` rows
//!   (`−x_i ≤ −x_l`, `x_i ≤ x_u`); the `.nl` "infinity" sentinel
//!   (`|v| ≥ 1e19`) is treated as no bound.

use crate::dispatch::analyze_quadratic_full;
use crate::nl_reader::NlProblem;
use pounce_convex::{QpProblem, Triplet};

/// The `.nl` infinity sentinel: AMPL writes ±1e20-ish for "no bound";
/// upstream Ipopt treats anything with magnitude ≥ 1e19 as infinite.
const NL_INF: f64 = 1e19;

fn is_finite_bound(v: f64) -> bool {
    v.abs() < NL_INF
}

/// Convert a classified LP/convex-QP `NlProblem` into `QpProblem`
/// standard form. Returns `None` if the objective is not actually a
/// degree-≤2 polynomial (should not happen for a problem the classifier
/// routed here, but the conversion is total and falls back gracefully).
pub fn extract_qp(prob: &NlProblem) -> Option<QpProblem> {
    Some(extract_qp_with_map(prob)?.0)
}

/// Where each `.nl` constraint's rows landed in the standard-form QP, so
/// the QP's multipliers can be mapped back to a per-`.nl`-constraint
/// dual for the `.sol`. One entry per original constraint, in order.
#[derive(Debug, Clone)]
pub enum ConRowMap {
    /// Equality constraint → row `a_row` of `A` (multiplier `y[a_row]`).
    Eq { a_row: usize },
    /// Inequality / range constraint → up to two rows of `G`: the
    /// `row ≤ g_u` upper bound and/or the `−row ≤ −g_l` lower bound
    /// (multipliers `z[..]`, each ≥ 0).
    Ineq {
        upper: Option<usize>,
        lower: Option<usize>,
    },
}

/// Extract the QP and the constraint→row provenance map together.
pub fn extract_qp_with_map(prob: &NlProblem) -> Option<(QpProblem, Vec<ConRowMap>)> {
    let n = prob.n;
    let sign = if prob.minimize { 1.0 } else { -1.0 };

    // --- objective Hessian P (lower triangle) + nonlinear-tree linear part ---
    let (hess, obj_nl_linear) = analyze_quadratic_full(&prob.obj_nonlinear, n)?;
    let mut p_lower: Vec<Triplet> = Vec::with_capacity(hess.len());
    for ((i, j), v) in &hess {
        // analyze_quadratic returns (i ≤ j) upper-ish keys; store as
        // lower triangle (row ≥ col) for the solver.
        let (row, col) = if i >= j { (*i, *j) } else { (*j, *i) };
        p_lower.push(Triplet::new(row, col, sign * v));
    }

    // --- objective linear term c ---
    // Two disjoint sources, exactly as the NLP path's eval_f sums them:
    // the `.nl` linear section (`obj_linear`) and the degree-1 terms AMPL
    // kept inside the nonlinear objective tree (e.g. the `−6·x₀` of
    // `(x₀−3)²`). Dropping the latter silently solves the wrong objective.
    let mut c = vec![0.0; n];
    for (var, coef) in &prob.obj_linear {
        c[*var] += sign * coef;
    }
    for (var, coef) in &obj_nl_linear {
        c[*var] += sign * coef;
    }

    // --- constraints: equalities → A x = b, inequalities → G x ≤ h ---
    let mut a: Vec<Triplet> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut g: Vec<Triplet> = Vec::new();
    let mut h: Vec<f64> = Vec::new();
    let mut con_map: Vec<ConRowMap> = Vec::with_capacity(prob.con_linear.len());

    for (row, lin) in prob.con_linear.iter().enumerate() {
        let lo = prob.g_l[row];
        let hi = prob.g_u[row];
        if lo == hi && is_finite_bound(lo) {
            // Equality row.
            let eq_row = next_row(&b);
            for (var, coef) in lin {
                a.push(Triplet::new(eq_row, *var, *coef));
            }
            b.push(lo);
            con_map.push(ConRowMap::Eq { a_row: eq_row });
        } else {
            // Upper bound: row ≤ hi.
            let upper = if is_finite_bound(hi) {
                let gr = next_row(&h);
                for (var, coef) in lin {
                    g.push(Triplet::new(gr, *var, *coef));
                }
                h.push(hi);
                Some(gr)
            } else {
                None
            };
            // Lower bound: row ≥ lo  ⇔  −row ≤ −lo.
            let lower = if is_finite_bound(lo) {
                let gr = next_row(&h);
                for (var, coef) in lin {
                    g.push(Triplet::new(gr, *var, -*coef));
                }
                h.push(-lo);
                Some(gr)
            } else {
                None
            };
            con_map.push(ConRowMap::Ineq { upper, lower });
        }
    }

    // --- variable bounds as G rows (not part of the constraint map) ---
    for i in 0..n {
        let xl = prob.x_l[i];
        let xu = prob.x_u[i];
        if is_finite_bound(xu) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, 1.0)); // x_i ≤ xu
            h.push(xu);
        }
        if is_finite_bound(xl) {
            let gr = next_row(&h);
            g.push(Triplet::new(gr, i, -1.0)); // −x_i ≤ −xl
            h.push(-xl);
        }
    }

    Some((
        QpProblem {
            n,
            p_lower,
            c,
            a,
            b,
            g,
            h,
            // Variable bounds are currently emitted as `G` rows (see the
            // bound-handling above), so the explicit box is left empty.
            lb: Vec::new(),
            ub: Vec::new(),
        },
        con_map,
    ))
}

/// Map the QP solver's multipliers `(y, z)` back to a per-`.nl`-
/// constraint dual vector (length `prob.m`), in the AMPL `.sol`
/// convention used by POUNCE's NLP path.
///
/// The QP solver enforces stationarity `∇f + Aᵀy + Gᵀz = 0` with
/// `z ≥ 0`, where each inequality `.nl` row contributes a `row ≤ g_u`
/// (`+row`) and/or `−row ≤ −g_l` (`−row`) `G` row. The per-constraint
/// `.nl`/Ipopt multiplier `λ` is recovered as:
/// - equality: `λ = sign · y[a_row]`;
/// - inequality: `λ = sign · (z_upper − z_lower)` — at most one of the
///   two bound rows is active at a solution.
///
/// The inequality sign (`z_upper − z_lower`, *not* `z_lower − z_upper`)
/// is fixed to match POUNCE's NLP path, which is the reference for what
/// a POUNCE `.sol` carries; this is verified empirically against the NLP
/// solve in the crate tests. `sign` undoes the maximize→minimize
/// negation so the reported dual is in the user's original sense.
pub fn recover_duals(prob: &NlProblem, con_map: &[ConRowMap], y: &[f64], z: &[f64]) -> Vec<f64> {
    let sign = if prob.minimize { 1.0 } else { -1.0 };
    con_map
        .iter()
        .map(|m| match m {
            ConRowMap::Eq { a_row } => sign * y[*a_row],
            ConRowMap::Ineq { upper, lower } => {
                let zu = upper.map(|r| z[r]).unwrap_or(0.0);
                let zl = lower.map(|r| z[r]).unwrap_or(0.0);
                sign * (zu - zl)
            }
        })
        .collect()
}

/// The next 0-based row index for a constraint block keyed by its RHS
/// vector's current length.
fn next_row(rhs: &[f64]) -> usize {
    rhs.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr};
    use pounce_convex::{solve_qp_ipm, QpOptions, QpStatus};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    fn pow2(var: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(var)),
            Box::new(Expr::Const(2.0)),
        )
    }

    /// min (x0)^2 + (x1)^2 s.t. x0 + x1 = 2, no var bounds → (1,1), f*=2.
    #[test]
    fn extract_and_solve_equality_qp() {
        let prob = NlProblem {
            n: 2,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Binary(BinOp::Add, Box::new(pow2(0)), Box::new(pow2(1))),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0), (1, 1.0)]],
            x_l: vec![-2e19, -2e19],
            x_u: vec![2e19, 2e19],
            g_l: vec![2.0],
            g_u: vec![2.0],
            x0: vec![0.0, 0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map) = extract_qp_with_map(&prob).expect("extract");
        // P = 2I → two diagonal entries.
        assert_eq!(qp.p_lower.len(), 2);
        assert_eq!(qp.m_eq(), 1);
        assert_eq!(qp.m_ineq(), 0);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        assert!((sol.x[1] - 1.0).abs() < 1e-6, "x1={}", sol.x[1]);
        assert!((sol.obj - 2.0).abs() < 1e-6, "obj={}", sol.obj);

        // KKT for the equality: ∇f + y·∇g = 0 → 2x_i + y = 0 at x=1 → y=−2.
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert_eq!(lambda.len(), 1);
        assert!(
            (lambda[0] - (-2.0)).abs() < 1e-5,
            "equality dual={}",
            lambda[0]
        );
    }

    /// Regression for the dropped-linear-term bug: the objective `(x0-3)²`
    /// lives entirely in the nonlinear tree, so its linear part (`−6·x0`)
    /// must be folded into `c`. Without it the solve minimizes `x0²`
    /// (optimum 0) instead of `(x0-3)²` (optimum 3).
    #[test]
    fn extract_keeps_linear_term_from_nonlinear_tree() {
        // (x0 - 3)^2 = x0^2 - 6 x0 + 9, all in obj_nonlinear.
        let obj = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(3.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: obj,
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert_eq!(qp.c.len(), 1);
        assert!(
            (qp.c[0] - (-6.0)).abs() < 1e-12,
            "c[0]={} — linear term from the nonlinear tree was dropped",
            qp.c[0]
        );
        // P = 2 (one diagonal entry).
        assert_eq!(qp.p_lower.len(), 1);

        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            (sol.x[0] - 3.0).abs() < 1e-6,
            "x0={} (expected 3)",
            sol.x[0]
        );
    }

    /// Inequality dual sign/magnitude. min x0² s.t. x0 ≥ 1 (a one-sided
    /// inequality g_l=1, g_u=+inf). Optimum x0=1, active. The expected
    /// dual −2.0 is the value POUNCE's *NLP* path writes for this exact
    /// problem (verified by running `solver_selection=nlp` on the same
    /// `.nl`); recover_duals must match that reference convention.
    #[test]
    fn inequality_dual_recovered() {
        let prob = NlProblem {
            n: 1,
            m: 1,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![],
            obj_constant: 0.0,
            con_nonlinear: vec![Expr::Const(0.0)],
            con_linear: vec![vec![(0, 1.0)]], // g(x) = x0
            x_l: vec![-2e19],
            x_u: vec![2e19],
            g_l: vec![1.0], // x0 ≥ 1
            g_u: vec![2e19],
            x0: vec![0.0],
            lambda0: vec![0.0],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let (qp, con_map) = extract_qp_with_map(&prob).expect("extract");
        // One inequality row (the lower bound row −x0 ≤ −1); no upper.
        assert_eq!(qp.m_ineq(), 1);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
        let lambda = recover_duals(&prob, &con_map, &sol.y, &sol.z);
        assert!((lambda[0] - (-2.0)).abs() < 1e-5, "ineq dual={}", lambda[0]);
    }

    /// Bound-constrained: min (x0-3)^2 = x0^2 - 6 x0 + 9, 0 ≤ x0 ≤ 1.
    /// Optimum x0 = 1 (upper bound binds). (Constant 9 dropped from c.)
    #[test]
    fn extract_and_solve_bounded_qp() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: pow2(0),
            obj_linear: vec![(0, -6.0)],
            obj_constant: 9.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // Two var-bound rows (x0 ≤ 1, −x0 ≤ 0).
        assert_eq!(qp.m_ineq(), 2);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }

    /// LP: min −x0 − x1, 0 ≤ x ≤ 1 → (1,1).
    #[test]
    fn extract_and_solve_lp() {
        let prob = NlProblem {
            n: 2,
            m: 0,
            num_obj: 1,
            minimize: true,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, -1.0), (1, -1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0, 0.0],
            x_u: vec![1.0, 1.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0, 0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        assert!(qp.p_lower.is_empty(), "LP has no Hessian");
        assert_eq!(qp.m_ineq(), 4); // 2 vars × (upper + lower)
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 1.0).abs() < 1e-6);
    }

    /// maximize x0 s.t. 0 ≤ x0 ≤ 5 → x0 = 5. Tests sign flip on a
    /// maximize objective.
    #[test]
    fn extract_maximize_negates() {
        let prob = NlProblem {
            n: 1,
            m: 0,
            num_obj: 1,
            minimize: false,
            obj_nonlinear: Expr::Const(0.0),
            obj_linear: vec![(0, 1.0)],
            obj_constant: 0.0,
            con_nonlinear: vec![],
            con_linear: vec![],
            x_l: vec![0.0],
            x_u: vec![5.0],
            g_l: vec![],
            g_u: vec![],
            x0: vec![0.0],
            lambda0: vec![],
            suffixes: Default::default(),
            imported_funcs: Vec::new(),
            var_names: Vec::new(),
            con_names: Vec::new(),
        };
        let qp = extract_qp(&prob).expect("extract");
        // minimize −x0.
        assert_eq!(qp.c[0], -1.0);
        let sol = solve_qp_ipm(&qp, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 5.0).abs() < 1e-6, "x0={}", sol.x[0]);
    }
}
