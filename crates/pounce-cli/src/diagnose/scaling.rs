//! Coefficient and scaling ranges.
//!
//! * `coefficient_ranges` — static: the smallest and largest nonzero
//!   magnitudes of the linear coefficients (constraints and objective), the
//!   quadratic-row curvature and right-hand sides read off the `.nl`, the
//!   constraint right-hand sides, and the finite variable bounds. The
//!   extremes are named. A very wide range, or a finite bound so large it is
//!   probably a stand-in for "no bound", is reported. In the style of the
//!   numerical-stability report of SDDP.jl.
//!
//! * `jacobian_scaling` — at the evaluation point: the spread of row and
//!   column scales (largest |entry| per row / per column) of the Jacobian,
//!   and the objective gradient's size against the column scales.
//!
//! These are observations about magnitudes, not a measure of conditioning:
//! a wide range is a reason to look, and a narrow one does not certify a
//! well-conditioned problem.

use super::report::{Basis, Confidence, Finding, Report, Severity, num};
use super::statics::affine_row;
use super::{DiagnoseOptions, PointEval, cname, vname};
use crate::nl_reader::{NlBody, NlProblem};
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};
use pounce_nl::nl_scaling::QuadRowCoef;
use serde_json::{Value, json};

/// A finite bound at or above this magnitude is probably a stand-in for an
/// absent bound. The solver treats only `|b| ≥ 1e19` as absent, so these are
/// real, and very loose, bounds.
const HUGE_BOUND: Number = 1e10;

/// Tracks the smallest and largest nonzero magnitude seen, and where.
#[derive(Default)]
struct Range {
    min: Option<(Number, String)>,
    max: Option<(Number, String)>,
    count: usize,
}

impl Range {
    fn see(&mut self, v: Number, at: impl FnOnce() -> String) {
        let a = v.abs();
        if a == 0.0 || !a.is_finite() {
            return;
        }
        self.count += 1;
        let need_min = self.min.as_ref().is_none_or(|m| a < m.0);
        let need_max = self.max.as_ref().is_none_or(|m| a > m.0);
        if need_min || need_max {
            let where_ = at();
            if need_min {
                self.min = Some((a, where_.clone()));
            }
            if need_max {
                self.max = Some((a, where_));
            }
        }
    }

    fn ratio(&self) -> Option<Number> {
        Some(self.max.as_ref()?.0 / self.min.as_ref()?.0)
    }

    fn json(&self) -> Value {
        json!({
            "count": self.count,
            "min": self.min.as_ref().map(|m| json!({"magnitude": m.0, "at": m.1})),
            "max": self.max.as_ref().map(|m| json!({"magnitude": m.0, "at": m.1})),
            "ratio": self.ratio().map(num),
        })
    }

    fn describe(&self) -> String {
        match (&self.min, &self.max) {
            (Some(lo), Some(hi)) => format!("{:.1e} ({}) to {:.1e} ({})", lo.0, lo.1, hi.0, hi.1),
            _ => "none".to_string(),
        }
    }
}

/// `coefficient_ranges`.
pub fn check_coefficients(
    prob: &NlProblem,
    quad: &[QuadRowCoef],
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    let mut lin = Range::default();
    let mut obj = Range::default();
    let mut rhs = Range::default();
    let mut bnd = Range::default();
    let mut curv = Range::default();
    for k in 0..prob.m {
        // Affine tree rows carry their coefficients in the tree; other rows
        // keep them in the `.nl` linear section.
        let coefs: Vec<(usize, Number)> = match (&prob.con_nonlinear[k], affine_row(prob, k)) {
            (NlBody::Tree(_), Some(a)) => a.coefs.into_iter().collect(),
            _ => prob.con_linear[k].clone(),
        };
        for (j, a) in coefs {
            lin.see(a, || format!("{} in {}", vname(prob, j), cname(prob, k)));
        }
        for b in [prob.g_l[k], prob.g_u[k]] {
            if lower_bound_present(b) && upper_bound_present(b) {
                rhs.see(b, || cname(prob, k));
            }
        }
    }
    if prob.num_obj > 0 {
        for &(j, c) in &prob.obj_linear {
            obj.see(c, || vname(prob, j));
        }
    }
    for q in quad {
        curv.see(q.curvature, || cname(prob, q.index));
    }
    let mut huge: Vec<(usize, Number)> = Vec::new();
    for j in 0..prob.n {
        for b in [prob.x_l[j], prob.x_u[j]] {
            if lower_bound_present(b) && upper_bound_present(b) {
                bnd.see(b, || vname(prob, j));
                if b.abs() >= HUGE_BOUND {
                    huge.push((j, b));
                }
            }
        }
    }
    let table = json!({
        "constraint_linear_coefficients": lin.json(),
        "objective_linear_coefficients": obj.json(),
        "quadratic_row_curvature": curv.json(),
        "constraint_right_hand_sides": rhs.json(),
        "finite_variable_bounds": bnd.json(),
    });
    report.sections.insert("coefficient_ranges", table.clone());
    let mut sink = report.ran("coefficient_ranges");
    sink.push(
        Finding::new("coefficient_ranges", "coefficient_ranges", Severity::Info, Basis::Proof,
            Confidence::Certain,
            format!("Constraint coefficients range {}; objective coefficients {}; right-hand sides {}; finite bounds {}.",
                lin.describe(), obj.describe(), rhs.describe(), bnd.describe()))
        .why("Magnitudes far from 1, or ranges spanning many orders, make the KKT system badly scaled.")
        .next("See sections.coefficient_ranges in the JSON for the full table.")
        .evidence(table),
    );
    for (label, r) in [
        ("constraint coefficients", &lin),
        ("objective coefficients", &obj),
        ("quadratic curvature", &curv),
    ] {
        if let Some(ratio) = r.ratio()
            && ratio > opts.wide_range
        {
            sink.push(
                    Finding::new("coefficient_ranges", "wide_coefficient_range", Severity::Warning,
                        Basis::Proof, Confidence::Certain,
                        format!("The {label} span {ratio:.1e}: from {}.", r.describe()))
                    .why("A range this wide usually means mixed units; the linear solves lose about log10(range) digits, and automatic gradient-based scaling only rescales rows, not columns.")
                    .next("Rescale the variables or rows named at the extremes to comparable units.")
                    .evidence(json!({"which": label, "ratio": num(ratio), "range": r.json(),
                        "threshold": opts.wide_range})),
                );
        }
    }
    for (j, b) in huge {
        sink.push(
            Finding::new("coefficient_ranges", "huge_finite_bound", Severity::Warning, Basis::Proof,
                Confidence::High,
                format!("Variable {} has a finite bound {b:e}, large enough to be a stand-in for \"no bound\".", vname(prob, j)))
            .why("The solver treats only |bound| ≥ 1e19 as absent. A finite bound this large is kept, its barrier term is nearly flat, and the bound multiplier is badly scaled; the initial point is pushed relative to it.")
            .next("If the variable is meant to be unbounded on that side, use an infinite bound (or one ≥ 1e19).")
            .vars(vec![vname(prob, j)])
            .evidence(json!({"bound": b, "threshold": HUGE_BOUND})),
        );
    }
}

/// `jacobian_scaling`.
pub fn check_jacobian(
    prob: &NlProblem,
    pt: &PointEval,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    if !pt.jac_ok || pt.jac_val.iter().any(|v| !v.is_finite()) {
        report.unavailable(
            "jacobian_scaling",
            format!(
                "the Jacobian is not finite at {} (see nonfinite_localization)",
                pt.label
            ),
        );
        return;
    }
    let mut row_max = vec![0.0_f64; prob.m];
    let mut col_max = vec![0.0_f64; prob.n];
    for (k, &v) in pt.jac_val.iter().enumerate() {
        let a = v.abs();
        let (r, c) = (pt.jac_irow[k], pt.jac_jcol[k]);
        row_max[r] = row_max[r].max(a);
        col_max[c] = col_max[c].max(a);
    }
    let mut rows = Range::default();
    for (i, &v) in row_max.iter().enumerate() {
        rows.see(v, || cname(prob, i));
    }
    let mut cols = Range::default();
    for (j, &v) in col_max.iter().enumerate() {
        if prob.x_l[j] != prob.x_u[j] {
            cols.see(v, || vname(prob, j));
        }
    }
    let zero_rows: Vec<String> = (0..prob.m)
        .filter(|&i| row_max[i] == 0.0)
        .map(|i| cname(prob, i))
        .collect();
    let grad_inf = if pt.grad_ok && pt.grad.iter().all(|v| v.is_finite()) {
        Some(pt.grad.iter().fold(0.0_f64, |a, v| a.max(v.abs())))
    } else {
        None
    };
    report.sections.insert(
        "jacobian_scaling",
        json!({"point": pt.label, "row_scales": rows.json(), "column_scales": cols.json(),
               "zero_rows": zero_rows.len(), "objective_gradient_inf_norm": grad_inf.map(num)}),
    );
    let mut sink = report.ran("jacobian_scaling");
    sink.push(
        Finding::new("jacobian_scaling", "jacobian_scale_summary", Severity::Info, Basis::Numerical,
            Confidence::Certain,
            format!("At {}, Jacobian row scales range {}; column scales {}; ‖∇f‖∞ = {}.",
                pt.label, rows.describe(), cols.describe(),
                grad_inf.map(|g| format!("{g:.1e}")).unwrap_or_else(|| "unavailable".into())))
        .why("Row and column scale spreads are what scaling has to undo; a gradient far larger or smaller than the column scales makes the objective dominate, or vanish from, the KKT system.")
        .next("See sections.jacobian_scaling and the embedded check_x0 scaling preview.")
        .evidence(json!({"point": pt.label, "row_scales": rows.json(), "column_scales": cols.json()})),
    );
    for (label, r) in [("row", &rows), ("column", &cols)] {
        if let Some(ratio) = r.ratio()
            && ratio > opts.wide_range
        {
            sink.push(
                    Finding::new("jacobian_scaling", if label == "row" { "wide_jacobian_row_scale_range" } else { "wide_jacobian_column_scale_range" },
                        Severity::Warning, Basis::Numerical, Confidence::High,
                        format!("At {}, Jacobian {label} scales span {ratio:.1e}: from {}.", pt.label, r.describe()))
                    .why(if label == "row" {
                        "Constraints of very different magnitudes are weighted very differently in the feasibility measure; gradient-based scaling caps rows above nlp_scaling_max_gradient but does not raise small ones."
                    } else {
                        "Variables in very different units make the KKT matrix badly scaled; the solver does not rescale columns."
                    })
                    .next("Rescale the extreme rows or variables to comparable units, then compare with another point to see whether the spread persists.")
                    .evidence(json!({"point": pt.label, "ratio": num(ratio), "range": r.json(), "threshold": opts.wide_range})),
                );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlProblemParts};

    fn lin(coefs: &[(usize, Number)]) -> Expr {
        Expr::Sum(
            coefs
                .iter()
                .map(|&(j, a)| {
                    Expr::Binary(BinOp::Mul, Box::new(Expr::Const(a)), Box::new(Expr::Var(j)))
                })
                .collect(),
        )
    }
    fn prob(cons: Vec<Expr>, n: usize) -> NlProblem {
        let m = cons.len();
        NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: cons,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![1.0; n],
            g_l: vec![1.0; m],
            g_u: vec![1.0; m],
            var_names: (0..n).map(|j| format!("x{j}")).collect(),
            con_names: (0..m).map(|i| format!("c{i}")).collect(),
        })
        .unwrap()
    }
    fn codes(r: &Report) -> Vec<&'static str> {
        r.findings.iter().map(|f| f.code).collect()
    }

    #[test]
    fn well_scaled_model_has_only_the_summary() {
        let p = prob(
            vec![lin(&[(0, 1.0), (1, 2.0)]), lin(&[(0, 3.0), (1, -1.0)])],
            2,
        );
        let mut r = Report::new("t".into(), 2, 2, 50);
        check_coefficients(&p, &[], &DiagnoseOptions::default(), &mut r);
        let pt = PointEval::evaluate_nl(&p, vec![1.0, 1.0], "p".into());
        check_jacobian(&p, &pt, &DiagnoseOptions::default(), &mut r);
        assert_eq!(
            codes(&r),
            vec!["coefficient_ranges", "jacobian_scale_summary"]
        );
    }

    #[test]
    fn planted_extreme_coefficient_is_named() {
        let p = prob(
            vec![lin(&[(0, 1.0), (1, 2.0)]), lin(&[(0, 3.0), (1, 1e10)])],
            2,
        );
        let mut r = Report::new("t".into(), 2, 2, 50);
        check_coefficients(&p, &[], &DiagnoseOptions::default(), &mut r);
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "wide_coefficient_range")
            .expect("flagged");
        assert!(f.observation.contains("x1 in c1"), "{}", f.observation);
        let pt = PointEval::evaluate_nl(&p, vec![1.0, 1.0], "p".into());
        check_jacobian(&p, &pt, &DiagnoseOptions::default(), &mut r);
        let f = r
            .findings
            .iter()
            .find(|f| f.code == "wide_jacobian_column_scale_range")
            .expect("flagged");
        assert!(f.observation.contains("x1"), "{}", f.observation);
    }

    #[test]
    fn huge_finite_bound_is_flagged() {
        let mut p = prob(vec![lin(&[(0, 1.0)])], 1);
        p.x_u[0] = 1e15;
        let mut r = Report::new("t".into(), 1, 1, 50);
        check_coefficients(&p, &[], &DiagnoseOptions::default(), &mut r);
        assert!(codes(&r).contains(&"huge_finite_bound"));
        p.x_u[0] = 1e19;
        let mut r = Report::new("t".into(), 1, 1, 50);
        check_coefficients(&p, &[], &DiagnoseOptions::default(), &mut r);
        assert!(!codes(&r).contains(&"huge_finite_bound"));
    }
}
