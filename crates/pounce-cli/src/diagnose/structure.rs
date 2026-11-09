//! Degeneracy of the equality constraints: structural, then numerical.
//!
//! * `structural_rank` — a maximum matching between equality rows and the
//!   (non-fixed) variables they touch, and its Dulmage–Mendelsohn
//!   decomposition (`pounce-presolve`). Unmatched equality rows mean the
//!   equality Jacobian is rank deficient at **every** point: the sparsity
//!   pattern bounds the rank from above. The over-determined block names the
//!   rows and variables involved.
//!
//! * `numerical_rank` — a dense SVD of the equality Jacobian at the
//!   evaluation point, under one absolute singular-value threshold. When it
//!   is rank deficient, a deletion filter shrinks the rows taking part in the
//!   left null space to an **inclusion-minimal dependent set**: a set that is
//!   dependent, while removing any one of its rows leaves an independent set.
//!   That is the "irreducible degenerate set" of Dowling & Biegler's
//!   *Degeneracy Hunter*, here computed numerically at one point and one
//!   threshold. The left-null combination is reported in original row names.
//!   If the deficiency is `d`, up to `max_dependent_sets` such sets are found
//!   by removing one row of each set found and searching again.
//!
//!   A numerical rank below the structural rank means the dependence comes
//!   from the values, not the pattern — it can come and go between points.

use super::report::{Basis, Confidence, Finding, Report, Severity, num};
use super::{DiagnoseOptions, PointEval, cname, vname};
use crate::nl_reader::NlProblem;
use faer::Mat;
use pounce_common::types::Number;
use pounce_presolve::incidence::{EqualityIncidence, ProbeView};
use pounce_presolve::matching::hopcroft_karp;
use pounce_presolve::{DMPart, DulmageMendelsohnPartition};
use serde_json::json;
use std::collections::BTreeMap;

/// Rows of the left null space with participation below this are ignored.
const NULL_WEIGHT_TOL: Number = 1e-3;

/// Names listed in a finding's text before "and N more".
const NAMES_SHOWN: usize = 12;

fn list(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(NAMES_SHOWN).map(String::as_str).collect();
    let more = names.len().saturating_sub(shown.len());
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

fn fixed_mask(prob: &NlProblem) -> Vec<bool> {
    (0..prob.n).map(|j| prob.x_l[j] == prob.x_u[j]).collect()
}

/// Equality rows, in row order.
fn equality_rows(prob: &NlProblem) -> Vec<usize> {
    (0..prob.m)
        .filter(|&i| prob.g_l[i] == prob.g_u[i] && prob.g_l[i].is_finite())
        .collect()
}

/// `structural_rank`.
/// Returns the structural rank, for [`check_numerical`] to compare against.
pub fn check_structural(prob: &NlProblem, pt: &PointEval, report: &mut Report) -> Option<usize> {
    if !pt.jac_structure_ok {
        report.unavailable(
            "structural_rank",
            "the Jacobian sparsity structure could not be obtained",
        );
        return None;
    }
    let irow: Vec<i32> = pt.jac_irow.iter().map(|&r| r as i32).collect();
    let jcol: Vec<i32> = pt.jac_jcol.iter().map(|&c| c as i32).collect();
    let fixed = fixed_mask(prob);
    let inc = EqualityIncidence::from_probe(&ProbeView {
        n_vars: prob.n,
        m_rows: prob.m,
        jac_irow: &irow,
        jac_jcol: &jcol,
        jac_values: None,
        g_l: &prob.g_l,
        g_u: &prob.g_u,
        linearity: None,
        one_based: false,
        eq_tol: 0.0,
        excluded_vars: Some(&fixed),
        excluded_rows: None,
    });
    let n_eq = inc.n_eq_rows();
    let matching = hopcroft_karp(&inc);
    let dm = DulmageMendelsohnPartition::from_matching(&inc, &matching);
    let n_free = fixed.iter().filter(|f| !**f).count();
    report.sections.insert(
        "structural_rank",
        json!({
            "equality_rows": n_eq,
            "free_variables": n_free,
            "structural_rank": matching.size,
            "overdetermined_rows": dm.over_rows.len(),
            "underdetermined_variables": dm.under_cols.iter().filter(|&&j| !fixed[j]).count(),
            "degrees_of_freedom": n_free as i64 - matching.size as i64,
        }),
    );
    let mut sink = report.ran("structural_rank");
    if matching.size < n_eq {
        let rows: Vec<String> = dm
            .over_rows
            .iter()
            .filter(|&&k| dm.row_part[k] == DMPart::Over)
            .map(|&k| cname(prob, inc.eq_row_inner_idx[k]))
            .collect();
        let cols: Vec<String> = dm.over_cols.iter().map(|&j| vname(prob, j)).collect();
        let deficiency = n_eq - matching.size;
        sink.push(
            Finding::new("structural_rank", "structurally_overdetermined_equalities", Severity::Warning,
                Basis::Structural, Confidence::Certain,
                format!("{deficiency} of the {n_eq} equality rows cannot be matched to a distinct free variable. The over-determined block has {} equation(s) in {} variable(s): equations {}; variables {}.",
                    rows.len(), cols.len(), list(&rows), if cols.is_empty() { "(none)".to_string() } else { list(&cols) }))
            .why("The equality Jacobian is rank deficient at every point (its sparsity pattern alone caps the rank), so LICQ fails everywhere: the multipliers are not unique and the solver must regularize. If the extra equations are not consistent, the problem is infeasible.")
            .next("Look for an equation that is implied by the others (a balance written once too often), or a variable that should be free but is fixed.")
            .cons(rows)
            .vars(cols)
            .evidence(json!({"equality_rows": n_eq, "structural_rank": matching.size,
                "structural_deficiency": deficiency, "fixed_variables_excluded": n_free < prob.n})),
        );
    }
    Some(matching.size)
}

/// Dense row-major Jacobian of `rows` restricted to `cols`.
struct Dense {
    rows: Vec<usize>,
    cols: Vec<usize>,
    a: Vec<Number>,
}

impl Dense {
    fn sub(&self, keep: &[usize]) -> Mat<Number> {
        // Only columns some kept row touches matter for the rank.
        let n = self.cols.len();
        let used: Vec<usize> = (0..n)
            .filter(|&c| keep.iter().any(|&r| self.a[r * n + c] != 0.0))
            .collect();
        Mat::from_fn(keep.len(), used.len().max(1), |i, j| {
            used.get(j).map(|&c| self.a[keep[i] * n + c]).unwrap_or(0.0)
        })
    }
}

fn rank_of(m: &Mat<Number>, tau: Number) -> Option<usize> {
    let s = m.singular_values().ok()?;
    Some(s.iter().filter(|&&v| v > tau).count())
}

/// Left-null weights of `keep` rows: Σ over null columns of U of u².
fn null_weights(d: &Dense, keep: &[usize], tau: Number) -> Option<Vec<Number>> {
    let m = d.sub(keep);
    let svd = m.svd().ok()?;
    let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
    let rank = s.iter().filter(|&&v| v > tau).count();
    let u = svd.U();
    let mut w = vec![0.0; keep.len()];
    for k in rank..keep.len() {
        for (i, val) in u.col(k).iter().enumerate() {
            w[i] += val * val;
        }
    }
    Some(w)
}

/// The left-null combination of a minimal dependent set (one dependency).
fn null_combination(d: &Dense, set: &[usize]) -> Option<(Vec<Number>, Number)> {
    let m = d.sub(set);
    let svd = m.svd().ok()?;
    let u = svd.U();
    let col = u.col(set.len() - 1);
    let mut c: Vec<Number> = col.iter().copied().collect();
    let scale = c.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    if scale > 0.0 {
        for v in &mut c {
            *v /= scale;
        }
    }
    // Residual ‖cᵀ A_S‖∞, relative to the largest entry of A_S.
    let ncols = m.ncols();
    let mut resid = 0.0_f64;
    let mut amax = 0.0_f64;
    for j in 0..ncols {
        let mut acc = 0.0;
        for i in 0..set.len() {
            acc += c[i] * m[(i, j)];
            amax = amax.max(m[(i, j)].abs());
        }
        resid = resid.max(acc.abs());
    }
    Some((c, if amax > 0.0 { resid / amax } else { 0.0 }))
}

/// `numerical_rank`.
pub fn check_numerical(
    prob: &NlProblem,
    pt: &PointEval,
    structural: Option<usize>,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    if !pt.jac_ok {
        report.unavailable(
            "numerical_rank",
            format!("the Jacobian could not be evaluated at {}", pt.label),
        );
        return;
    }
    let eq = equality_rows(prob);
    let fixed = fixed_mask(prob);
    let mut row_pos = vec![usize::MAX; prob.m];
    for (p, &r) in eq.iter().enumerate() {
        row_pos[r] = p;
    }
    let mut col_pos: BTreeMap<usize, usize> = BTreeMap::new();
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if row_pos[r] != usize::MAX && !fixed[c] {
            let next = col_pos.len();
            col_pos.entry(c).or_insert(next);
        }
    }
    let (m, n) = (eq.len(), col_pos.len());
    if m == 0 {
        report.ran("numerical_rank");
        return;
    }
    if m.saturating_mul(n.max(1)) > opts.max_dense_entries || m.min(n) > opts.max_dense_dim {
        report.unavailable(
            "numerical_rank",
            format!("the equality Jacobian is {m} × {n}, larger than the dense-SVD limit (--max-dense-entries {}, --max-dense-dim {})", opts.max_dense_entries, opts.max_dense_dim),
        );
        return;
    }
    let mut cols: Vec<usize> = vec![0; n];
    for (&c, &p) in &col_pos {
        cols[p] = c;
    }
    let ncols = n.max(1);
    let mut a = vec![0.0; m * ncols];
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if row_pos[r] == usize::MAX || fixed[c] {
            continue;
        }
        a[row_pos[r] * ncols + col_pos[&c]] += pt.jac_val[k];
    }
    if a.iter().any(|v| !v.is_finite()) {
        report.unavailable(
            "numerical_rank",
            format!(
                "the equality Jacobian has non-finite entries at {} (see nonfinite_localization)",
                pt.label
            ),
        );
        return;
    }
    let d = Dense {
        rows: eq.clone(),
        cols,
        a,
    };
    let all: Vec<usize> = (0..m).collect();
    let full = d.sub(&all);
    let Ok(s) = full.singular_values() else {
        report.unavailable("numerical_rank", "the SVD did not converge");
        return;
    };
    let mut s = s;
    s.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    let smax = s.first().copied().unwrap_or(0.0);
    let rel = opts
        .rank_rel_tol
        .unwrap_or((m.max(n) as Number) * Number::EPSILON);
    let tau = smax * rel;
    report
        .tolerances
        .insert("rank_singular_value_threshold", tau);
    let rank = s.iter().filter(|&&v| v > tau).count();
    let smin_pos = s
        .iter()
        .copied()
        .filter(|&v| v > tau)
        .fold(Number::INFINITY, Number::min);
    report.sections.insert(
        "numerical_rank",
        json!({
            "point": pt.label,
            "equality_rows": m,
            "columns": n,
            "rank": rank,
            "deficiency": m.saturating_sub(rank),
            "sigma_max": num(smax),
            "smallest_retained_sigma": num(smin_pos),
            "threshold": num(tau),
            "relative_threshold": rel,
            "scaling": "none (unscaled Jacobian)",
            "structural_rank": structural,
        }),
    );

    let mut findings: Vec<Finding> = Vec::new();
    if let Some(sr) = structural
        && rank < sr
    {
        findings.push(
                Finding::new("numerical_rank", "numerical_rank_below_structural_rank", Severity::Warning,
                    Basis::Numerical, Confidence::High,
                    format!("At {}, the equality Jacobian has numerical rank {rank}, below its structural rank {sr}, so a deficiency of {} comes from the values, not the sparsity pattern.",
                        pt.label, sr - rank))
                .why("Value-dependent rank loss can appear at some points and not others; where it appears the solver needs dual regularization and the multipliers are not unique.")
                .next("Inspect the dependent sets below, and repeat at another meaningful point (e.g. --sol) to see whether the loss persists.")
                .evidence(json!({"point": pt.label, "rank": rank, "structural_rank": sr, "threshold": num(tau)})),
            );
    }

    // Minimal dependent sets.
    let deficiency = m.saturating_sub(rank);
    let mut working: Vec<usize> = all.clone();
    let mut found = 0usize;
    while found < deficiency.min(opts.max_dependent_sets) {
        let Some(w) = null_weights(&d, &working, tau) else {
            break;
        };
        let mut cand: Vec<(usize, Number)> = working
            .iter()
            .zip(&w)
            .filter(|(_, wt)| **wt > NULL_WEIGHT_TOL)
            .map(|(&r, &wt)| (r, wt))
            .collect();
        let dependent = |set: &[usize]| rank_of(&d.sub(set), tau).map(|r| r < set.len());
        let mut set: Vec<usize> = cand.iter().map(|c| c.0).collect();
        if set.is_empty() || dependent(&set) != Some(true) {
            set = working.clone();
            cand = working.iter().map(|&r| (r, 0.0)).collect();
        }
        if set.len() > opts.max_dependent_search {
            let names: Vec<String> = set.iter().map(|&p| cname(prob, d.rows[p])).collect();
            findings.push(
                Finding::new("numerical_rank", "dependent_equality_rows_unlocalized", Severity::Warning,
                    Basis::Numerical, Confidence::Medium,
                    format!("At {}, the equality Jacobian is rank deficient by {deficiency}; {} rows take part in the null space, more than the minimal-set search limit (--max-dependent-search {}). Rows: {}.",
                        pt.label, set.len(), opts.max_dependent_search, list(&names)))
                .why("LICQ fails at this point; the solver needs dual regularization and the multipliers are not unique.")
                .next("Raise --max-dependent-search, or inspect the listed rows for a redundant equation.")
                .cons(names),
            );
            break;
        }
        // Deletion filter, least-involved rows first.
        cand.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        for (r, _) in &cand {
            if set.len() <= 1 {
                break;
            }
            let trial: Vec<usize> = set.iter().copied().filter(|x| x != r).collect();
            if dependent(&trial) == Some(true) {
                set = trial;
            }
        }
        set.sort_unstable();
        let Some((coef, resid)) = null_combination(&d, &set) else {
            break;
        };
        let names: Vec<String> = set.iter().map(|&p| cname(prob, d.rows[p])).collect();
        let combo: Vec<serde_json::Value> = set
            .iter()
            .zip(&coef)
            .map(|(&p, &c)| json!({"row": cname(prob, d.rows[p]), "coefficient": c}))
            .collect();
        let structural_note = match structural {
            Some(sr) if rank >= sr => {
                "The structural rank shows the same loss, so the sparsity pattern alone forces it."
            }
            Some(_) => {
                "The structural rank does not show this loss, so it comes from the values at this point."
            }
            None => "",
        };
        findings.push(
            Finding::new("numerical_rank", "dependent_equality_rows", Severity::Warning,
                Basis::Numerical, Confidence::High,
                format!("At {}, equalities {} are linearly dependent, and removing any one of them leaves the rest independent (a minimal dependent set). {structural_note}",
                    pt.label, names.join(", ")))
            .why("LICQ fails at this point: the multipliers of these rows are not unique, and the interior-point method must regularize. If the rows' right-hand sides are not consistent with the dependency, no point satisfies them all.")
            .next("Check whether one of these equations is implied by the others; the combination in the evidence shows how they cancel.")
            .cons(names)
            .evidence(json!({"point": pt.label, "left_null_combination": combo,
                "relative_residual": num(resid), "threshold": num(tau),
                "method": "dense SVD, deletion filter at a fixed singular-value threshold"})),
        );
        found += 1;
        // Remove the row carrying the largest coefficient and search again.
        let drop = set
            .iter()
            .zip(&coef)
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&p, _)| p);
        match drop {
            Some(p) => working.retain(|&x| x != p),
            None => break,
        }
    }
    if deficiency > found && found == opts.max_dependent_sets {
        findings.push(
            Finding::new("numerical_rank", "more_dependent_sets_not_searched", Severity::Info,
                Basis::Numerical, Confidence::High,
                format!("The deficiency at {} is {deficiency}; only {found} minimal dependent set(s) were searched (--max-dependent-sets).", pt.label))
            .why("The remaining dependencies are not localized in this report.")
            .next("Raise --max-dependent-sets to localize more."),
        );
    }
    let mut sink = report.ran("numerical_rank");
    for f in findings {
        sink.push(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::{BinOp, Expr, NlProblemParts};

    fn v(j: usize) -> Expr {
        Expr::Var(j)
    }
    fn c(x: Number) -> Expr {
        Expr::Const(x)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Add, Box::new(a), Box::new(b))
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        Expr::Binary(BinOp::Mul, Box::new(a), Box::new(b))
    }
    fn eqs(cons: Vec<Expr>, n: usize) -> NlProblem {
        let m = cons.len();
        NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: cons,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![1.0; n],
            g_l: vec![0.0; m],
            g_u: vec![0.0; m],
            var_names: Vec::new(),
            con_names: (0..m).map(|i| format!("e{i}")).collect(),
        })
        .unwrap()
    }
    fn run(p: &NlProblem, x: Vec<Number>) -> Report {
        let pt = PointEval::evaluate_nl(p, x, "test point".into());
        let mut r = Report::new("t".into(), p.n, p.m, 50);
        let sr = check_structural(p, &pt, &mut r);
        check_numerical(p, &pt, sr, &DiagnoseOptions::default(), &mut r);
        r
    }
    fn dep_sets(r: &Report) -> Vec<Vec<String>> {
        r.findings
            .iter()
            .filter(|f| f.code == "dependent_equality_rows")
            .map(|f| f.constraints.clone())
            .collect()
    }

    #[test]
    fn duplicate_rows_form_a_minimal_pair() {
        // e0: x0 + x1, e1: x2, e2: 2 x0 + 2 x1.
        let p = eqs(
            vec![
                add(v(0), v(1)),
                v(2),
                add(mul(c(2.0), v(0)), mul(c(2.0), v(1))),
            ],
            3,
        );
        let r = run(&p, vec![1.0; 3]);
        assert_eq!(dep_sets(&r), vec![vec!["e0".to_string(), "e2".to_string()]]);
    }

    #[test]
    fn three_row_dependence_with_no_dependent_pair() {
        // e0: x0 + x1, e1: x1 + x2, e2: x0 - x2 (= e0 - e1); e3: x3.
        let p = eqs(
            vec![
                add(v(0), v(1)),
                add(v(1), v(2)),
                add(v(0), mul(c(-1.0), v(2))),
                v(3),
            ],
            4,
        );
        let r = run(&p, vec![1.0; 4]);
        let sets = dep_sets(&r);
        assert_eq!(
            sets,
            vec![vec!["e0".to_string(), "e1".to_string(), "e2".to_string()]]
        );
        // Structurally fine: three rows, three columns, a perfect matching.
        assert!(
            !r.findings
                .iter()
                .any(|f| f.code == "structurally_overdetermined_equalities")
        );
    }

    #[test]
    fn value_dependent_rank_loss_is_numerical_only() {
        // e0: x0*x1, e1: x0 + x1. Jacobian [[x1, x0], [1, 1]] is singular
        // where x0 == x1, and regular elsewhere.
        let p = eqs(vec![mul(v(0), v(1)), add(v(0), v(1))], 2);
        let r = run(&p, vec![2.0, 2.0]);
        let codes: Vec<_> = r.findings.iter().map(|f| f.code).collect();
        assert!(
            codes.contains(&"numerical_rank_below_structural_rank"),
            "{codes:?}"
        );
        assert_eq!(dep_sets(&r), vec![vec!["e0".to_string(), "e1".to_string()]]);
        let r = run(&p, vec![1.0, 3.0]);
        assert!(
            r.findings.is_empty(),
            "regular point: {:?}",
            r.findings.iter().map(|f| f.code).collect::<Vec<_>>()
        );
    }

    #[test]
    fn structural_overdetermination() {
        // Two equations in the single variable x0.
        let p = eqs(vec![v(0), mul(c(3.0), v(0))], 1);
        let r = run(&p, vec![0.0]);
        let codes: Vec<_> = r.findings.iter().map(|f| f.code).collect();
        assert!(
            codes.contains(&"structurally_overdetermined_equalities"),
            "{codes:?}"
        );
        assert!(!codes.contains(&"numerical_rank_below_structural_rank"));
        assert_eq!(dep_sets(&r).len(), 1);
    }

    #[test]
    fn full_rank_reports_nothing() {
        let p = eqs(vec![add(v(0), v(1)), add(v(0), mul(c(-1.0), v(1)))], 2);
        assert!(run(&p, vec![1.0, 1.0]).findings.is_empty());
    }
}
