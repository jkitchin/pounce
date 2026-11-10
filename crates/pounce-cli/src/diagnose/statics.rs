//! Static checks: things the represented algebra and bounds prove before any
//! point is evaluated.
//!
//! * `bounds` — variable and row bounds that are empty on their own.
//! * `linear_rows` — rows whose body is affine: constant rows that can never
//!   hold, single-variable rows that are really bounds (and contradict the
//!   variable's own bounds or each other), and parallel rows that duplicate
//!   or contradict one another.
//! * `incidence` — variables that appear nowhere, and objective terms that
//!   are unbounded along a free direction no constraint touches.
//!
//! Everything here is a [`Basis::Proof`] about the model as written: it holds
//! at every point, so none of it depends on the starting point.

use super::report::{Basis, Confidence, Finding, Report, Severity, fmt_num, interval, num};
use super::{DiagnoseOptions, cname, lo_eff, up_eff, vname};
use crate::nl_reader::{BinOp, Expr, NlBody, NlProblem, UnaryOp, collect_vars, eval_expr};
use pounce_common::types::Number;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// A row whose body is affine: `Σ coefs[j]·x_j + constant`.
#[derive(Debug, Clone)]
pub struct AffineRow {
    pub coefs: BTreeMap<usize, Number>,
    pub constant: Number,
}

/// Node budget for walking one row's tree when looking for an affine body.
/// A row that needs more than this is treated as not affine (the checks that
/// use the result only lose coverage, never claim something false).
const AFFINE_NODE_BUDGET: usize = 100_000;

/// The value of an expression with no variables in it, or `None` if it
/// mentions a variable or an imported function.
pub fn const_fold(e: &Expr) -> Option<Number> {
    match e {
        Expr::Const(c) => Some(*c),
        Expr::Var(_) | Expr::Funcall { .. } => None,
        Expr::Cse(b) => const_fold(b),
        Expr::Unary(op, a) => {
            let v = const_fold(a)?;
            Some(eval_expr(&Expr::Unary(*op, Box::new(Expr::Const(v))), &[]))
        }
        Expr::Binary(op, a, b) => {
            let va = const_fold(a)?;
            let vb = const_fold(b)?;
            Some(eval_expr(
                &Expr::Binary(*op, Box::new(Expr::Const(va)), Box::new(Expr::Const(vb))),
                &[],
            ))
        }
        Expr::Sum(args) => args.iter().map(const_fold).sum(),
        _ => None,
    }
}

fn affine_walk(e: &Expr, scale: Number, out: &mut AffineRow, budget: &mut usize) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    match e {
        Expr::Const(c) => {
            out.constant += scale * c;
            true
        }
        Expr::Var(j) => {
            *out.coefs.entry(*j).or_insert(0.0) += scale;
            true
        }
        Expr::Cse(b) => affine_walk(b, scale, out, budget),
        Expr::Sum(args) => args.iter().all(|a| affine_walk(a, scale, out, budget)),
        Expr::Unary(UnaryOp::Neg, a) => affine_walk(a, -scale, out, budget),
        Expr::Binary(BinOp::Add, a, b) => {
            affine_walk(a, scale, out, budget) && affine_walk(b, scale, out, budget)
        }
        Expr::Binary(BinOp::Sub, a, b) => {
            affine_walk(a, scale, out, budget) && affine_walk(b, -scale, out, budget)
        }
        Expr::Binary(BinOp::Mul, a, b) => {
            if let Some(c) = const_fold(a) {
                affine_walk(b, scale * c, out, budget)
            } else if let Some(c) = const_fold(b) {
                affine_walk(a, scale * c, out, budget)
            } else {
                false
            }
        }
        Expr::Binary(BinOp::Div, a, b) => match const_fold(b) {
            Some(c) if c != 0.0 && c.is_finite() => affine_walk(a, scale / c, out, budget),
            _ => false,
        },
        _ => match const_fold(e) {
            Some(v) => {
                out.constant += scale * v;
                true
            }
            None => false,
        },
    }
}

/// `e` as an affine function, if it is one.
pub fn affine_expr(e: &Expr) -> Option<AffineRow> {
    let mut row = AffineRow {
        coefs: BTreeMap::new(),
        constant: 0.0,
    };
    let mut budget = AFFINE_NODE_BUDGET;
    affine_walk(e, 1.0, &mut row, &mut budget).then_some(row)
}

/// Row `k` as an affine function, if it is one. Combines the `.nl` linear
/// section with an affine nonlinear tree (an in-memory problem built with
/// `from_expressions` keeps its linear terms in the tree). Coefficients that
/// cancel to exactly zero are dropped.
pub fn affine_row(prob: &NlProblem, k: usize) -> Option<AffineRow> {
    let mut row = match &prob.con_nonlinear[k] {
        NlBody::Quad(_) => return None,
        NlBody::Tree(e) => affine_expr(e)?,
    };
    for &(j, a) in &prob.con_linear[k] {
        *row.coefs.entry(j).or_insert(0.0) += a;
    }
    row.coefs.retain(|_, a| *a != 0.0);
    if !row.constant.is_finite() || row.coefs.values().any(|a| !a.is_finite()) {
        return None;
    }
    Some(row)
}

/// Is `[lo, hi]` empty by more than `tol` (scaled by the magnitudes)?
fn gap(lo: Number, hi: Number) -> Number {
    if lo.is_finite() && hi.is_finite() {
        lo - hi
    } else if lo > hi {
        Number::INFINITY
    } else {
        0.0
    }
}

fn empty_beyond(lo: Number, hi: Number, tol: Number) -> bool {
    let g = gap(lo, hi);
    let mag = if lo.is_finite() && hi.is_finite() {
        lo.abs().max(hi.abs())
    } else {
        0.0
    };
    g > tol * (1.0 + mag)
}

/// `bounds`: variable and row bounds that are empty on their own.
///
/// Compared as effective bounds: `±1e19` and beyond is the solver's "no
/// bound" sentinel, so `x ≥ 1e300` has no upper bound rather than a lower
/// bound above its upper one, and `g ≤ -5e20` has no lower bound.
pub fn check_bounds(prob: &NlProblem, report: &mut Report) {
    let mut sink = report.ran("bounds");
    for j in 0..prob.n {
        let (l, u) = (prob.x_l[j], prob.x_u[j]);
        if l.is_nan() || u.is_nan() {
            sink.push(
                Finding::new("bounds", "nan_variable_bound", Severity::Error, Basis::Proof,
                    Confidence::Certain,
                    format!("Variable {} has a NaN bound.", vname(prob, j)))
                .why("A NaN bound makes every comparison false; the solver cannot interpret the box.")
                .next("Trace where the bound value is computed in the model's data.")
                .vars(vec![vname(prob, j)])
                .evidence(json!({"lower": num(l), "upper": num(u)})),
            );
        } else if lo_eff(l) > up_eff(u) {
            sink.push(
                Finding::new("bounds", "inconsistent_variable_bounds", Severity::Error,
                    Basis::Proof, Confidence::Certain,
                    format!("Variable {} has lower bound {} above its upper bound {}.",
                        vname(prob, j), fmt_num(l), fmt_num(u)))
                .why("No value of the variable satisfies its own bounds, so the problem is infeasible.")
                .next("Check the data and units that set both bounds.")
                .vars(vec![vname(prob, j)])
                .evidence(json!({"lower": num(l), "upper": num(u)})),
            );
        }
    }
    for i in 0..prob.m {
        let (l, u) = (prob.g_l[i], prob.g_u[i]);
        if l.is_nan() || u.is_nan() {
            sink.push(
                Finding::new(
                    "bounds",
                    "nan_constraint_bound",
                    Severity::Error,
                    Basis::Proof,
                    Confidence::Certain,
                    format!("Constraint {} has a NaN bound.", cname(prob, i)),
                )
                .why("A NaN right-hand side cannot be satisfied or violated meaningfully.")
                .next("Trace where the right-hand side is computed.")
                .cons(vec![cname(prob, i)])
                .evidence(json!({"lower": num(l), "upper": num(u)})),
            );
        } else if lo_eff(l) > up_eff(u) {
            sink.push(
                Finding::new(
                    "bounds",
                    "inconsistent_constraint_bounds",
                    Severity::Error,
                    Basis::Proof,
                    Confidence::Certain,
                    format!(
                        "Constraint {} has lower bound {} above its upper bound {}.",
                        cname(prob, i),
                        fmt_num(l),
                        fmt_num(u)
                    ),
                )
                .why("No point satisfies the row, so the problem is infeasible.")
                .next("Check how the row's two sides were built.")
                .cons(vec![cname(prob, i)])
                .evidence(json!({"lower": num(l), "upper": num(u)})),
            );
        }
    }
}

/// Parallel-row group key: the support and the coefficients normalized by
/// the first one, printed to 10 significant digits.
type GroupKey = (Vec<usize>, Vec<String>);
/// One row of a group: index, normalized range `[lo, hi]`, normalized
/// coefficients.
type GroupRow = (usize, Number, Number, Vec<Number>);

/// `linear_rows`: constant rows, single-variable rows, parallel rows.
pub fn check_linear_rows(prob: &NlProblem, opts: &DiagnoseOptions, report: &mut Report) {
    let tol = opts.feas_tol;
    let rows: Vec<Option<AffineRow>> = (0..prob.m).map(|k| affine_row(prob, k)).collect();
    let n_affine = rows.iter().filter(|r| r.is_some()).count();
    report.sections.insert(
        "linear_rows",
        json!({"affine_rows": n_affine, "rows": prob.m}),
    );
    let mut sink = report.ran("linear_rows");

    // Per-variable implied bounds from singleton rows: (lo, hi, rows).
    let mut implied: BTreeMap<usize, (Number, Number, Vec<usize>)> = BTreeMap::new();
    // Parallel-row groups keyed by support + normalized coefficients.
    let mut groups: HashMap<GroupKey, Vec<GroupRow>> = HashMap::new();

    for (k, row) in rows.iter().enumerate() {
        let Some(row) = row else { continue };
        let (gl, gu) = (lo_eff(prob.g_l[k]), up_eff(prob.g_u[k]));
        if gl > gu {
            continue; // reported by `bounds`
        }
        match row.coefs.len() {
            0 => {
                let v = row.constant;
                if v < gl - tol * (1.0 + gl.abs()) || v > gu + tol * (1.0 + gu.abs()) {
                    sink.push(
                        Finding::new("linear_rows", "constant_row_infeasible", Severity::Error,
                            Basis::Proof, Confidence::Certain,
                            format!("Constraint {} mentions no variable; its constant value {} lies outside its bounds [{}, {}].", cname(prob, k), fmt_num(v), fmt_num(gl), fmt_num(gu)))
                        .why("No point can satisfy this row, so the problem is infeasible.")
                        .next("Find why every variable dropped out of this row (a fixed or eliminated variable, or a data error).")
                        .cons(vec![cname(prob, k)])
                        .evidence(json!({"value": v, "bounds": interval(gl, gu)})),
                    );
                } else if prob.g_l[k] == prob.g_u[k] {
                    sink.push(
                        Finding::new("linear_rows", "constant_equality_row", Severity::Warning,
                            Basis::Proof, Confidence::Certain,
                            format!("Equality {} mentions no variable (it holds identically, at value {v}).", cname(prob, k)))
                        .why("Its Jacobian row is zero everywhere, so the equality Jacobian is rank deficient at every point (LICQ fails); the interior-point method will need dual regularization.")
                        .next("Drop the row, or find why its variables disappeared.")
                        .cons(vec![cname(prob, k)])
                        .evidence(json!({"value": v})),
                    );
                } else {
                    sink.push(
                        Finding::new(
                            "linear_rows",
                            "constant_row",
                            Severity::Info,
                            Basis::Proof,
                            Confidence::Certain,
                            format!(
                                "Constraint {} mentions no variable and always holds.",
                                cname(prob, k)
                            ),
                        )
                        .why("It has no effect on the feasible set.")
                        .next("Check whether a variable was meant to appear in it.")
                        .cons(vec![cname(prob, k)])
                        .evidence(json!({"value": v, "bounds": interval(gl, gu)})),
                    );
                }
            }
            1 => {
                let Some((&j, &a)) = row.coefs.iter().next() else {
                    continue;
                };
                let (mut lo, mut hi) = ((gl - row.constant) / a, (gu - row.constant) / a);
                if a < 0.0 {
                    std::mem::swap(&mut lo, &mut hi);
                }
                let e = implied.entry(j).or_insert((
                    Number::NEG_INFINITY,
                    Number::INFINITY,
                    Vec::new(),
                ));
                e.0 = e.0.max(lo);
                e.1 = e.1.min(hi);
                e.2.push(k);
                sink.push(
                    Finding::new("linear_rows", "bound_expressed_as_constraint", Severity::Info,
                        Basis::Proof, Confidence::Certain,
                        format!("Constraint {} involves only {}; it is the bound {} ≤ {} ≤ {}.",
                            cname(prob, k), vname(prob, j), fmt_num(lo), vname(prob, j), fmt_num(hi)))
                    .why("As a row it adds a Jacobian row and a multiplier; as a variable bound the interior-point method handles it directly. Both describe the same feasible set.")
                    .next("Keep it if its multiplier means something to you; otherwise a variable bound is cheaper.")
                    .vars(vec![vname(prob, j)])
                    .cons(vec![cname(prob, k)])
                    .evidence(json!({"implied_bound": interval(lo, hi)})),
                );
            }
            _ => {
                let support: Vec<usize> = row.coefs.keys().copied().collect();
                let pivot = row.coefs[&support[0]];
                let norm: Vec<Number> = support.iter().map(|j| row.coefs[j] / pivot).collect();
                let key_coefs: Vec<String> = norm.iter().map(|c| format!("{c:.10e}")).collect();
                let (mut lo, mut hi) = ((gl - row.constant) / pivot, (gu - row.constant) / pivot);
                if pivot < 0.0 {
                    std::mem::swap(&mut lo, &mut hi);
                }
                groups
                    .entry((support, key_coefs))
                    .or_default()
                    .push((k, lo, hi, norm));
            }
        }
    }

    for (j, (lo, hi, ks)) in &implied {
        let (lo, hi, ks) = (*lo, *hi, ks.clone());
        let (xl, xu) = (lo_eff(prob.x_l[*j]), up_eff(prob.x_u[*j]));
        let cons: Vec<String> = ks.iter().map(|&k| cname(prob, k)).collect();
        if empty_beyond(lo, hi, tol) {
            sink.push(
                Finding::new("linear_rows", "inconsistent_single_variable_rows", Severity::Error,
                    Basis::Proof, Confidence::Certain,
                    format!("The single-variable rows {} on {} imply {} ≤ {} ≤ {}, which is empty.",
                        cons.join(", "), vname(prob, *j), fmt_num(lo), vname(prob, *j), fmt_num(hi)))
                .why("No value of the variable satisfies these rows together, so the problem is infeasible.")
                .next("Compare the rows' right-hand sides against the source data and units.")
                .vars(vec![vname(prob, *j)])
                .cons(cons)
                .evidence(json!({"implied_bound": interval(lo, hi)})),
            );
        } else if xl <= xu && empty_beyond(lo.max(xl), hi.min(xu), tol) {
            sink.push(
                Finding::new("linear_rows", "implied_bound_contradicts_variable_bound",
                    Severity::Error, Basis::Proof, Confidence::Certain,
                    format!("Rows {} imply {} ≤ {} ≤ {}, disjoint from the variable's own bounds [{}, {}].",
                        cons.join(", "), fmt_num(lo), vname(prob, *j), fmt_num(hi), fmt_num(xl), fmt_num(xu)))
                .why("No value of the variable satisfies both, so the problem is infeasible.")
                .next("Check which of the two encodes the intended limit.")
                .vars(vec![vname(prob, *j)])
                .cons(cons)
                .evidence(json!({"implied_bound": interval(lo, hi), "variable_bound": interval(xl, xu)})),
            );
        }
    }

    // Deterministic order: by first row index.
    let mut groups: Vec<_> = groups.into_values().filter(|g| g.len() >= 2).collect();
    groups.sort_by_key(|g| g[0].0);
    for g in groups {
        let exact = g.iter().all(|r| r.3 == g[0].3);
        let (lo, hi) = g
            .iter()
            .fold((Number::NEG_INFINITY, Number::INFINITY), |(l, h), r| {
                (l.max(r.1), h.min(r.2))
            });
        let cons: Vec<String> = g.iter().map(|r| cname(prob, r.0)).collect();
        let n_eq = g.iter().filter(|r| prob.g_l[r.0] == prob.g_u[r.0]).count();
        let conf = if exact {
            Confidence::Certain
        } else {
            Confidence::High
        };
        let evidence = json!({
            "rows": g.iter().map(|r| json!({"row": cname(prob, r.0),
                "normalized_range": interval(r.1, r.2)})).collect::<Vec<_>>(),
            "intersection": interval(lo, hi),
            "exactly_proportional": exact,
        });
        if empty_beyond(lo, hi, tol) && exact {
            sink.push(
                Finding::new("linear_rows", "inconsistent_parallel_rows", Severity::Error,
                    Basis::Proof, conf,
                    format!("Rows {} have proportional left-hand sides but right-hand sides no point can meet together.", cons.join(", ")))
                .why("The rows describe the same linear function with contradictory limits, so the problem is infeasible.")
                .next("Check the right-hand sides of these rows against the source data and units.")
                .cons(cons)
                .evidence(evidence),
            );
        } else if empty_beyond(lo, hi, tol) {
            // Proportional only to ~11 significant digits: a tiny
            // difference in direction lets the rows meet far away (x + y ≤ 1
            // and x + (1 + 1e-12)·y ≥ 2 meet at y ≈ 1e12), so this proves
            // nothing.
            sink.push(
                Finding::new("linear_rows", "nearly_parallel_conflicting_rows", Severity::Warning,
                    Basis::Heuristic, Confidence::Medium,
                    format!("Rows {} are nearly proportional (equal to about 11 significant digits) with right-hand sides that conflict along their common direction; they can only both hold far from the origin.", cons.join(", ")))
                .why("If the coefficients were meant to be equal, the problem is infeasible; if not, the only feasible points are extreme, and the near-parallel rows make the Jacobian badly conditioned there.")
                .next("Check whether the coefficients differ by rounding in the source data.")
                .cons(cons)
                .evidence(evidence),
            );
        } else if n_eq >= 2 {
            sink.push(
                Finding::new("linear_rows", "duplicate_equality_rows", Severity::Warning,
                    if exact { Basis::Proof } else { Basis::Heuristic }, conf,
                    format!("Equalities {} are proportional: the same equation written {} times.", cons.join(", "), n_eq))
                .why("Their Jacobian rows are linearly dependent at every point, so LICQ fails and the multipliers are not unique; the interior-point method will need dual regularization and may converge slowly.")
                .next("Keep one of them, or confirm the duplication is intended.")
                .cons(cons)
                .evidence(evidence),
            );
        } else {
            sink.push(
                Finding::new("linear_rows", "parallel_rows", Severity::Info,
                    if exact { Basis::Proof } else { Basis::Heuristic }, conf,
                    format!("Rows {} have proportional left-hand sides; together they bound one linear function.", cons.join(", ")))
                .why("At most one of them can be binding at a time on each side; the others are redundant.")
                .next("Combine them into one ranged row if that matches the intent.")
                .cons(cons)
                .evidence(evidence),
            );
        }
    }
}

/// Every variable that appears in a constraint, from the Jacobian sparsity.
fn constraint_vars(prob: &NlProblem) -> BTreeSet<usize> {
    let (_, jcol) = crate::nl_reader::constraint_jacobian_sparsity(prob);
    jcol.iter().map(|&j| j as usize).collect()
}

/// `incidence`: variables appearing nowhere, and unbounded objective rays.
pub fn check_incidence(prob: &NlProblem, report: &mut Report) {
    let in_cons = constraint_vars(prob);
    let mut obj_nl = BTreeSet::new();
    let mut obj_lin: BTreeMap<usize, Number> = BTreeMap::new();
    if prob.num_obj > 0 {
        match &prob.obj_nonlinear {
            NlBody::Quad(q) => obj_nl.extend(q.vars.iter().map(|&v| v as usize)),
            // An affine tree (an in-memory problem keeps its linear
            // objective terms there) counts as linear terms.
            NlBody::Tree(e) => match affine_expr(e) {
                Some(a) => {
                    for (j, c) in a.coefs {
                        *obj_lin.entry(j).or_insert(0.0) += c;
                    }
                }
                None => collect_vars(e, &mut obj_nl),
            },
        }
        for &(j, c) in &prob.obj_linear {
            *obj_lin.entry(j).or_insert(0.0) += c;
        }
    }
    let mut sink = report.ran("incidence");
    for j in 0..prob.n {
        if in_cons.contains(&j) || obj_nl.contains(&j) {
            continue;
        }
        let c = obj_lin.get(&j).copied().unwrap_or(0.0);
        if c == 0.0 {
            sink.push(
                Finding::new("incidence", "disconnected_variable", Severity::Info, Basis::Proof,
                    Confidence::Certain,
                    format!("Variable {} appears in no constraint and not in the objective.", vname(prob, j)))
                .why("Its value is arbitrary within its bounds; any value the solver returns is meaningless.")
                .next("Check whether it was meant to appear in a constraint.")
                .vars(vec![vname(prob, j)]),
            );
            continue;
        }
        // Objective-only and linear: the objective moves along x_j alone.
        // Improving direction is −sign(c) when minimizing.
        let improving_down = (c > 0.0) == prob.minimize;
        let (l, u) = (lo_eff(prob.x_l[j]), up_eff(prob.x_u[j]));
        let unbounded = if improving_down {
            l == Number::NEG_INFINITY
        } else {
            u == Number::INFINITY
        };
        if unbounded {
            sink.push(
                Finding::new("incidence", "unbounded_objective_ray", Severity::Error, Basis::Proof,
                    Confidence::Certain,
                    format!("Variable {} appears in no constraint, enters the objective linearly with coefficient {c}, and has no bound in the improving direction.", vname(prob, j)))
                .why("Moving it toward its missing bound improves the objective without limit, so if the problem is feasible it is unbounded.")
                .next("Add the missing bound, or check the sign of the objective coefficient and the sense (min/max).")
                .vars(vec![vname(prob, j)])
                .cons(vec!["objective".to_string()])
                .evidence(json!({"objective_coefficient": c, "bounds": interval(l, u),
                    "sense": if prob.minimize {"minimize"} else {"maximize"}})),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_reader::NlProblemParts;

    pub(crate) fn problem(
        cons: Vec<Expr>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        n: usize,
    ) -> NlProblem {
        NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: Expr::Const(0.0),
            obj_constant: 0.0,
            constraints: cons,
            x_l: vec![-1e19; n],
            x_u: vec![1e19; n],
            x0: vec![0.0; n],
            g_l,
            g_u,
            var_names: Vec::new(),
            con_names: Vec::new(),
        })
        .unwrap()
    }

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

    fn codes(r: &Report) -> Vec<&'static str> {
        r.findings.iter().map(|f| f.code).collect()
    }

    fn run_linear(p: &NlProblem) -> Report {
        let mut r = Report::new("t".into(), p.n, p.m, 50);
        check_linear_rows(p, &DiagnoseOptions::default(), &mut r);
        r
    }

    #[test]
    fn affine_row_reads_tree_terms() {
        let p = problem(
            vec![add(mul(c(2.0), v(0)), add(v(1), c(3.0)))],
            vec![0.0],
            vec![0.0],
            2,
        );
        let a = affine_row(&p, 0).unwrap();
        assert_eq!(a.coefs[&0], 2.0);
        assert_eq!(a.coefs[&1], 1.0);
        assert_eq!(a.constant, 3.0);
        let nl = problem(vec![mul(v(0), v(1))], vec![0.0], vec![0.0], 2);
        assert!(
            affine_row(&nl, 0).is_none(),
            "a product of variables is not affine"
        );
    }

    #[test]
    fn parallel_equalities_with_different_rhs_are_inconsistent() {
        // x + y = 1 and 2x + 2y = 4.
        let p = problem(
            vec![add(v(0), v(1)), add(mul(c(2.0), v(0)), mul(c(2.0), v(1)))],
            vec![1.0, 4.0],
            vec![1.0, 4.0],
            2,
        );
        assert_eq!(codes(&run_linear(&p)), vec!["inconsistent_parallel_rows"]);
    }

    #[test]
    fn nearly_parallel_conflicting_rows_are_not_a_proof() {
        // x + y <= 1 and x + (1 + 1e-12) y >= 2 share a key to 11 digits
        // but meet at y ≈ 1e12: feasible, so no proof-level error.
        let p = problem(
            vec![add(v(0), v(1)), add(v(0), mul(c(1.0 + 1e-12), v(1)))],
            vec![-1e19, 2.0],
            vec![1.0, 1e19],
            2,
        );
        let r = run_linear(&p);
        assert_eq!(codes(&r), vec!["nearly_parallel_conflicting_rows"]);
        assert_eq!(r.findings[0].severity, Severity::Warning);
        assert_eq!(r.findings[0].basis, Basis::Heuristic);
    }

    #[test]
    fn parallel_equalities_with_same_rhs_are_duplicates() {
        let p = problem(
            vec![add(v(0), v(1)), add(mul(c(2.0), v(0)), mul(c(2.0), v(1)))],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            2,
        );
        assert_eq!(codes(&run_linear(&p)), vec!["duplicate_equality_rows"]);
    }

    #[test]
    fn parallel_inequalities_that_overlap_are_info() {
        let p = problem(
            vec![add(v(0), v(1)), add(v(0), v(1))],
            vec![0.0, -1e19],
            vec![1e19, 5.0],
            2,
        );
        assert_eq!(codes(&run_linear(&p)), vec!["parallel_rows"]);
    }

    #[test]
    fn independent_rows_report_nothing() {
        let p = problem(
            vec![add(v(0), v(1)), add(v(0), mul(c(-1.0), v(1)))],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            2,
        );
        assert!(run_linear(&p).findings.is_empty());
    }

    #[test]
    fn singleton_rows_are_bounds_and_contradictions_are_found() {
        // 2x >= 4 and x <= 1: implied [2, inf) ∩ (-inf, 1] is empty.
        let p = problem(
            vec![mul(c(2.0), v(0)), v(0)],
            vec![4.0, -1e19],
            vec![1e19, 1.0],
            1,
        );
        let r = run_linear(&p);
        let cs = codes(&r);
        assert_eq!(
            cs.iter()
                .filter(|c| **c == "bound_expressed_as_constraint")
                .count(),
            2
        );
        assert!(cs.contains(&"inconsistent_single_variable_rows"));
    }

    #[test]
    fn singleton_row_against_variable_bound() {
        let mut p = problem(vec![v(0)], vec![5.0], vec![1e19], 1);
        p.x_u[0] = 1.0;
        assert!(codes(&run_linear(&p)).contains(&"implied_bound_contradicts_variable_bound"));
        p.x_u[0] = 10.0;
        assert!(!codes(&run_linear(&p)).contains(&"implied_bound_contradicts_variable_bound"));
    }

    #[test]
    fn constant_rows() {
        let p = problem(
            vec![c(3.0), c(3.0), c(3.0)],
            vec![4.0, 3.0, 0.0],
            vec![5.0, 3.0, 10.0],
            1,
        );
        assert_eq!(
            codes(&run_linear(&p)),
            vec![
                "constant_row_infeasible",
                "constant_equality_row",
                "constant_row"
            ]
        );
    }

    #[test]
    fn bounds_check_finds_crossed_bounds() {
        let mut p = problem(vec![v(0)], vec![2.0], vec![1.0], 1);
        p.x_l[0] = 3.0;
        p.x_u[0] = 1.0;
        let mut r = Report::new("t".into(), 1, 1, 50);
        check_bounds(&p, &mut r);
        assert_eq!(
            codes(&r),
            vec![
                "inconsistent_variable_bounds",
                "inconsistent_constraint_bounds"
            ]
        );
        let ok = problem(vec![v(0)], vec![0.0], vec![1.0], 1);
        let mut r = Report::new("t".into(), 1, 1, 50);
        check_bounds(&ok, &mut r);
        assert!(r.findings.is_empty());
        // A finite bound beyond the absent-bound sentinel on the other side
        // is not crossed: `g >= 1e300` (upper absent at 1e19) and
        // `x <= -5e20` (lower absent at -1e19) are both satisfiable.
        let mut s = problem(vec![v(0)], vec![1e300], vec![1e19], 1);
        s.x_u[0] = -5e20;
        let mut r = Report::new("t".into(), 1, 1, 50);
        check_bounds(&s, &mut r);
        assert!(r.findings.is_empty(), "{:?}", codes(&r));
    }

    #[test]
    fn incidence_finds_disconnected_and_unbounded() {
        // min x1 (x1 free, in no constraint) s.t. x0 = 1; x2 nowhere.
        let mut p = NlProblem::from_expressions(NlProblemParts {
            minimize: true,
            objective: v(1),
            obj_constant: 0.0,
            constraints: vec![v(0)],
            x_l: vec![-1e19; 3],
            x_u: vec![1e19; 3],
            x0: vec![0.0; 3],
            g_l: vec![1.0],
            g_u: vec![1.0],
            var_names: Vec::new(),
            con_names: Vec::new(),
        })
        .unwrap();
        let mut r = Report::new("t".into(), 3, 1, 50);
        check_incidence(&p, &mut r);
        assert_eq!(
            codes(&r),
            vec!["unbounded_objective_ray", "disconnected_variable"]
        );
        // Bounded below: the ray is gone.
        p.x_l[1] = 0.0;
        let mut r = Report::new("t".into(), 3, 1, 50);
        check_incidence(&p, &mut r);
        assert_eq!(codes(&r), vec!["disconnected_variable"]);
    }
}
