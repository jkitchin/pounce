//! Phase 2 (gh#1160): convexity evidence.
//!
//! * `hessian_inertia` — eigenvalue counts (positive / negative / near
//!   zero) of the Hessian of the Lagrangian at the point, over the free
//!   variables. With multipliers from `--sol` it is the Lagrangian's; without
//!   them, the objective's. At a KKT point it also reports the **reduced**
//!   Hessian on the null space of the strongly active constraints. Negative
//!   curvature there means the second-order necessary condition fails: the
//!   point is not a local minimizer.
//!
//! * `convexity_sampling` — seeded Jensen tests. For each nonlinear function,
//!   pairs `x, y` are drawn from a box around the point. The box is one that
//!   the interval walk certifies as free of domain trouble. The test checks
//!   `f(t x + (1-t) y)` against `t f(x) + (1-t) f(y)`. A violation is a
//!   replayable counterexample to convexity (or concavity). Each function is
//!   compared with the curvature its bounds need for a convex feasible set:
//!   convex for `g ≤ u`, concave for `g ≥ l`, affine for equalities and
//!   ranges; the objective must be convex when minimizing. Finding no
//!   counterexample proves nothing, and the report says so.

use super::domain::certified_on_box;
use super::kkt::{Active, KktState};
use super::report::{Basis, Confidence, Finding, Report, Severity, fmt_num};
use super::statics::affine_expr;
use super::structure::fixed_mask;
use super::{DiagnoseOptions, PointEval, cname, lo_eff, up_eff, vname};
use crate::nl_reader::{Expr, NlBody, NlProblem, collect_vars, eval_expr};
use faer::{Mat, Side};
use pounce_common::types::Number;
use pounce_nlp::tnlp::{SparsityRequest, TNLP};
use serde_json::json;
use std::collections::BTreeSet;

/// Rows with more quadratic variables than this are not sampled.
const QUAD_SAMPLE_MAX_VARS: usize = 5_000;

/// Counts of eigenvalues (positive, negative, near zero, zero tolerance).
/// "Near zero" is relative to `scale` or the largest eigenvalue, whichever
/// is larger: a projected matrix's own eigenvalues are no reference for its
/// rounding error, which follows the matrix it was projected from.
fn inertia_scaled(eigs: &[Number], scale: Number) -> (usize, usize, usize, Number) {
    let emax = eigs.iter().fold(scale.abs(), |a, v| a.max(v.abs()));
    let tol = 1e-8 * emax.max(1e-300);
    let pos = eigs.iter().filter(|&&v| v > tol).count();
    let neg = eigs.iter().filter(|&&v| v < -tol).count();
    (pos, neg, eigs.len() - pos - neg, tol)
}

fn inertia(eigs: &[Number]) -> (usize, usize, usize, Number) {
    inertia_scaled(eigs, 0.0)
}

/// `hessian_inertia`.
pub fn check_hessian(
    prob: &NlProblem,
    tnlp: &mut dyn TNLP,
    pt: &PointEval,
    kkt: Option<&KktState>,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    let fixed = fixed_mask(prob);
    let free: Vec<usize> = (0..prob.n).filter(|&j| !fixed[j]).collect();
    let nf = free.len();
    if nf == 0 {
        report.ran("hessian_inertia");
        return;
    }
    if nf.saturating_mul(nf) > opts.max_dense_entries || nf > opts.max_dense_dim {
        report.unavailable(
            "hessian_inertia",
            format!(
                "{nf} free variables exceed the dense limits (--max-dense-entries, --max-dense-dim)"
            ),
        );
        return;
    }
    let Some(info) = tnlp.get_nlp_info() else {
        report.unavailable("hessian_inertia", "get_nlp_info failed");
        return;
    };
    let nnz = info.nnz_h_lag.max(0) as usize;
    let off = usize::from(matches!(
        info.index_style,
        pounce_nlp::tnlp::IndexStyle::Fortran
    ));
    let lambda: Vec<Number> = kkt
        .and_then(|k| k.lambda.clone())
        .unwrap_or_else(|| vec![0.0; prob.m]);
    let which = if kkt.and_then(|k| k.lambda.as_ref()).is_some() {
        "Lagrangian"
    } else {
        "objective"
    };
    let mut hr = vec![0i32; nnz];
    let mut hc = vec![0i32; nnz];
    let mut hv = vec![0.0; nnz];
    let ok = nnz == 0
        || (tnlp.eval_h(
            None,
            false,
            1.0,
            None,
            false,
            SparsityRequest::Structure {
                irow: &mut hr,
                jcol: &mut hc,
            },
        ) && tnlp.eval_h(
            Some(&pt.x),
            true,
            1.0,
            Some(&lambda),
            true,
            SparsityRequest::Values { values: &mut hv },
        ));
    if !ok {
        report.unavailable("hessian_inertia", "the evaluator provides no exact Hessian");
        return;
    }
    if hv.iter().any(|v| !v.is_finite()) {
        report.unavailable(
            "hessian_inertia",
            format!("the Hessian is not finite at {}", pt.label),
        );
        return;
    }
    let mut pos = vec![usize::MAX; prob.n];
    for (p, &j) in free.iter().enumerate() {
        pos[j] = p;
    }
    let mut h = Mat::<Number>::zeros(nf, nf);
    for k in 0..nnz {
        let (r, c) = (
            (hr[k] as usize).wrapping_sub(off),
            (hc[k] as usize).wrapping_sub(off),
        );
        if r >= prob.n || c >= prob.n || pos[r] == usize::MAX || pos[c] == usize::MAX {
            continue;
        }
        let (a, b) = (pos[r], pos[c]);
        h[(a, b)] += hv[k];
        if a != b {
            h[(b, a)] += hv[k];
        }
    }
    let Ok(eigs) = h.self_adjoint_eigenvalues(Side::Lower) else {
        report.unavailable("hessian_inertia", "the eigenvalue solver did not converge");
        return;
    };
    let (np, nn, nz, etol) = inertia(&eigs);
    report.sections.insert(
        "hessian_inertia",
        json!({"point": pt.label, "hessian": which, "free_variables": nf,
            "positive": np, "negative": nn, "zero": nz, "zero_tolerance": etol,
            "min_eigenvalue": eigs.first(), "max_eigenvalue": eigs.last()}),
    );
    let mut sink = report.ran("hessian_inertia");
    sink.push(
        Finding::new("hessian_inertia", "hessian_inertia", Severity::Info, Basis::Numerical, Confidence::High,
            format!("At {}, the {which} Hessian over the {nf} free variables has {np} positive, {nn} negative and {nz} near-zero eigenvalue(s).", pt.label))
        .why(if prob.m == 0 && nn > 0 {
            "With no constraints, a negative eigenvalue means the objective is nonconvex here and the point is not a local minimizer."
        } else {
            "The full Hessian's inertia shows local curvature only; whether the point is a minimizer depends on the reduced Hessian on the constraints' tangent space."
        })
        .next("See reduced_hessian (with --sol) for the second-order condition at a solution.")
        .evidence(json!({"point": pt.label, "hessian": which, "positive": np, "negative": nn, "zero": nz})),
    );

    // Reduced Hessian at a KKT point.
    let Some(k) = kkt else { return };
    let Some(mu) = k.mu.as_ref() else {
        report.unavailable(
            "reduced_hessian",
            k.why_no_multipliers
                .clone()
                .unwrap_or_else(|| "the .sol has no constraint multipliers".into()),
        );
        return;
    };
    let strong: Vec<Active> = k
        .active
        .iter()
        .zip(mu)
        .filter(|(a, m)| !a.is_inequality() || **m > k.mult_tol)
        .map(|(a, _)| *a)
        .collect();
    let weak = k
        .active
        .iter()
        .zip(mu)
        .filter(|(a, m)| a.is_inequality() && **m <= k.mult_tol)
        .count();
    let bound_cols: BTreeSet<usize> = strong
        .iter()
        .filter_map(|a| match a {
            Active::VarLower(j) | Active::VarUpper(j) => Some(*j),
            _ => None,
        })
        .collect();
    let cols: Vec<usize> = free
        .iter()
        .copied()
        .filter(|j| !bound_cols.contains(j))
        .collect();
    let rows: Vec<usize> = strong
        .iter()
        .filter_map(|a| match a {
            Active::Eq(i) | Active::RowLower(i) | Active::RowUpper(i) => Some(*i),
            _ => None,
        })
        .collect();
    let nc = cols.len();
    let mut cpos = vec![usize::MAX; prob.n];
    for (p, &j) in cols.iter().enumerate() {
        cpos[j] = p;
    }
    let mut rpos = vec![usize::MAX; prob.m];
    for (p, &i) in rows.iter().enumerate() {
        rpos[i] = p;
    }
    let mut a = Mat::<Number>::zeros(rows.len(), nc.max(1));
    for (q, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[q];
        if r < prob.m && rpos[r] != usize::MAX && cpos[c] != usize::MAX {
            a[(rpos[r], cpos[c])] += pt.jac_val[q];
        }
    }
    let z: Mat<Number> = if rows.is_empty() {
        Mat::<Number>::identity(nc, nc)
    } else {
        let Ok(svd) = a.svd() else { return };
        let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
        let tau =
            s.first().copied().unwrap_or(0.0) * ((rows.len().max(nc)) as Number) * Number::EPSILON;
        let r = s.iter().filter(|&&v| v > tau).count();
        let v = svd.V();
        Mat::from_fn(nc, nc.saturating_sub(r), |i, j| v[(i, r + j)])
    };
    let zd = z.ncols();
    if zd == 0 {
        report.ran("reduced_hessian").push(
            Finding::new("reduced_hessian", "no_free_directions", Severity::Info, Basis::Numerical, Confidence::High,
                format!("At {}, the strongly active constraints leave no feasible direction: the point is a vertex of the active set.", pt.label))
            .why("The second-order condition is vacuous here; the point is determined by the active constraints alone.")
            .next("Nothing to check."),
        );
        return;
    }
    // H restricted to `cols`.
    let fpos: Vec<usize> = cols.iter().map(|&j| pos[j]).collect();
    let hs = Mat::from_fn(nc, nc, |i, j| h[(fpos[i], fpos[j])]);
    let hz = &hs * &z;
    let zt = z.transpose().to_owned();
    let red = &zt * &hz;
    let Ok(reig) = red.self_adjoint_eigenvalues(Side::Lower) else {
        return;
    };
    let hscale = eigs.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let (rp, rn, rz, _) = inertia_scaled(&reig, hscale);
    report.sections.insert(
        "reduced_hessian",
        json!({"point": pt.label, "dimension": zd, "positive": rp, "negative": rn, "zero": rz,
            "min_eigenvalue": reig.first(), "weakly_active_treated_as_inactive": weak}),
    );
    let mut sink = report.ran("reduced_hessian");
    let conf = if weak > 0 {
        Confidence::Medium
    } else {
        Confidence::High
    };
    let weak_note = if weak > 0 {
        format!(
            " {weak} weakly active constraint(s) were treated as inactive, so the tested subspace may be larger than the critical cone."
        )
    } else {
        String::new()
    };
    if rn > 0 {
        sink.push(
            Finding::new("reduced_hessian", "reduced_hessian_negative_curvature", Severity::Warning, Basis::Numerical, conf,
                format!("At {}, the Lagrangian Hessian has {rn} negative eigenvalue(s) on the tangent space of the active constraints (smallest {}): the second-order necessary condition fails.{weak_note}",
                    pt.label, fmt_num(reig[0])))
            .why("Along those feasible directions the objective decreases: the point is a saddle point or a maximizer, not a local minimizer, even if it is a KKT point.")
            .next("Perturb the point along the negative-curvature direction and re-solve, or try other starts (--minima).")
            .evidence(json!({"point": pt.label, "negative": rn, "min_eigenvalue": reig[0]})),
        );
    } else if rz > 0 {
        sink.push(
            Finding::new("reduced_hessian", "reduced_hessian_singular", Severity::Info, Basis::Numerical, conf,
                format!("At {}, the reduced Hessian has {rz} near-zero eigenvalue(s) and none negative: second-order sufficiency cannot be confirmed.{weak_note}", pt.label))
            .why("Flat directions mean the minimizer may not be isolated: nearby points can be equally optimal, and sensitivities are not unique.")
            .next("Check for a symmetry or an unconstrained combination of variables."),
        );
    } else {
        sink.push(
            Finding::new("reduced_hessian", "second_order_sufficient", Severity::Info, Basis::Numerical, conf,
                format!("At {}, the reduced Hessian is positive definite (smallest eigenvalue {}): the second-order sufficient condition holds.{weak_note}", pt.label, fmt_num(reig[0])))
            .why("Together with first-order stationarity, this makes the point a strict local minimizer.")
            .next("Nothing to check."),
        );
    }
}

/// SplitMix64: a small, fixed, portable generator, so witnesses replay.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> Number {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as Number / (1u64 << 53) as Number
    }
}

/// A Jensen counterexample: x, y (over the function's variables), t,
/// `[f(x), f(y), f(tx + (1-t)y)]`, and the size of the violation.
type Witness = (Vec<Number>, Vec<Number>, Number, [Number; 3], Number);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    Convex,
    Concave,
    Affine,
}

/// `convexity_sampling`.
pub fn check_sampling(
    prob: &NlProblem,
    pt: &PointEval,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    // The functions to test: (label, expression, needed curvature).
    let mut funcs: Vec<(String, std::borrow::Cow<'_, Expr>, Need)> = Vec::new();
    let nonlinear = |b: &NlBody| match b {
        NlBody::Quad(q) => q.vars.len() <= QUAD_SAMPLE_MAX_VARS,
        NlBody::Tree(e) => affine_expr(e).is_none(),
    };
    if prob.num_obj > 0 && nonlinear(&prob.obj_nonlinear) {
        funcs.push((
            "objective".into(),
            prob.obj_expr(),
            if prob.minimize {
                Need::Convex
            } else {
                Need::Concave
            },
        ));
    }
    for i in 0..prob.m {
        if !nonlinear(&prob.con_nonlinear[i]) {
            continue;
        }
        let (gl, gu) = (lo_eff(prob.g_l[i]), up_eff(prob.g_u[i]));
        let need = match (gl.is_finite(), gu.is_finite()) {
            (false, true) => Need::Convex,
            (true, false) => Need::Concave,
            (true, true) => Need::Affine,
            (false, false) => continue,
        };
        funcs.push((cname(prob, i), prob.con_expr(i), need));
    }
    // Sampling box around the point, inside the bounds.
    let mut lo = vec![0.0; prob.n];
    let mut hi = vec![0.0; prob.n];
    for j in 0..prob.n {
        let (l, u) = (lo_eff(prob.x_l[j]), up_eff(prob.x_u[j]));
        let c = pt.x[j].clamp(l.min(u), u.max(l));
        let r = opts.sample_radius * c.abs().max(1.0);
        lo[j] = (c - r).max(l);
        hi[j] = (c + r).min(u);
        if lo[j] > hi[j] {
            lo[j] = c;
            hi[j] = c;
        }
    }
    let mut rng = Rng(opts.seed);
    let mut tested = 0usize;
    let mut clean = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut findings: Vec<Finding> = Vec::new();
    let mut nonconvex = 0usize;
    for (label, e, need) in &funcs {
        if !certified_on_box(e, &lo, &hi, opts.domain_tol) {
            skipped.push(label.clone());
            continue;
        }
        tested += 1;
        let mut vars = BTreeSet::new();
        collect_vars(e, &mut vars);
        let mut x = pt.x.clone();
        let mut y = pt.x.clone();
        let mut zpt = pt.x.clone();
        let mut conv_w: Option<Witness> = None;
        let mut conc_w: Option<Witness> = None;
        for _ in 0..opts.convexity_samples {
            for &j in &vars {
                x[j] = lo[j] + (hi[j] - lo[j]) * rng.next();
                y[j] = lo[j] + (hi[j] - lo[j]) * rng.next();
            }
            let t = 0.1 + 0.8 * rng.next();
            for &j in &vars {
                zpt[j] = t * x[j] + (1.0 - t) * y[j];
            }
            let (fx, fy, fz) = (eval_expr(e, &x), eval_expr(e, &y), eval_expr(e, &zpt));
            if !(fx.is_finite() && fy.is_finite() && fz.is_finite()) {
                continue;
            }
            let chord = t * fx + (1.0 - t) * fy;
            let tol = opts.convexity_tol * (1.0 + fx.abs().max(fy.abs()).max(fz.abs()));
            let gap = fz - chord;
            let pick = |v: &[Number]| vars.iter().map(|&j| v[j]).collect::<Vec<_>>();
            if gap > tol && conv_w.as_ref().is_none_or(|w| gap > w.4) {
                conv_w = Some((pick(&x), pick(&y), t, [fx, fy, fz], gap));
            }
            if -gap > tol && conc_w.as_ref().is_none_or(|w| -gap > w.4) {
                conc_w = Some((pick(&x), pick(&y), t, [fx, fy, fz], -gap));
            }
        }
        let violated = match need {
            Need::Convex => conv_w.as_ref().map(|w| ("convex", w)),
            Need::Concave => conc_w.as_ref().map(|w| ("concave", w)),
            Need::Affine => conv_w
                .as_ref()
                .map(|w| ("convex", w))
                .or(conc_w.as_ref().map(|w| ("concave", w))),
        };
        let Some((what, w)) = violated else {
            clean += 1;
            continue;
        };
        nonconvex += 1;
        let names: Vec<String> = vars.iter().take(20).map(|&j| vname(prob, j)).collect();
        let role = match (label.as_str(), need) {
            ("objective", _) => {
                if prob.minimize {
                    "a minimized objective must be convex"
                } else {
                    "a maximized objective must be concave"
                }
            }
            (_, Need::Convex) => "a row bounded above must be convex",
            (_, Need::Concave) => "a row bounded below must be concave",
            (_, Need::Affine) => "an equality or ranged row must be affine",
        };
        findings.push(
            Finding::new("convexity_sampling", "convexity_counterexample", Severity::Info, Basis::Numerical, Confidence::High,
                format!("{label} is not {what}: at t = {}, f(t·x + (1−t)·y) = {} against the chord's {}, a violation of {} ({role}).",
                    fmt_num(w.2), fmt_num(w.3[2]), fmt_num(w.2 * w.3[0] + (1.0 - w.2) * w.3[1]), fmt_num(w.4)))
            .why("The feasible set or objective is nonconvex, so a local solution need not be global.")
            .next("Use several starts (--minima) or a global solver if the global optimum matters; the witness in the evidence replays exactly.")
            .cons(vec![label.clone()])
            .vars(names.clone())
            .evidence(json!({"function": label, "required": role, "t": w.2, "f_x": w.3[0], "f_y": w.3[1], "f_tx_plus_1mt_y": w.3[2],
                "violation": w.4, "variables": names, "x": w.0, "y": w.1, "seed": opts.seed,
                "box": "bounds ∩ point ± sample_radius·max(1, |x|), certified free of domain trouble"})),
        );
    }
    report.sections.insert(
        "convexity_sampling",
        json!({"functions": funcs.len(), "tested": tested, "counterexamples": nonconvex,
            "no_counterexample": clean, "skipped_domain_not_certified": skipped.len(),
            "samples_per_function": opts.convexity_samples, "seed": opts.seed, "radius": opts.sample_radius}),
    );
    let mut sink = report.ran("convexity_sampling");
    for f in findings {
        sink.push(f);
    }
    if funcs.is_empty() {
        return;
    }
    sink.push(
        Finding::new("convexity_sampling", "convexity_summary", Severity::Info, Basis::Numerical, Confidence::High,
            format!("Of {} nonlinear function(s), {nonconvex} have a counterexample to the curvature their bounds need, {clean} showed none in {} samples (inconclusive: not a proof of convexity), and {} were not sampled (the box around {} is not certified free of domain trouble, or they call imported functions).{}",
                funcs.len(), opts.convexity_samples, skipped.len(), pt.label,
                if nonconvex > 0 { " The model is nonconvex: a local solution need not be global." } else { "" }))
        .why("Convexity makes every local solution global; sampled evidence can disprove it but never prove it.")
        .next("See the counterexamples above; raise --convexity-samples or --sample-radius to search harder.")
        .cons(skipped),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_in_unit_interval() {
        let mut a = Rng(7);
        let mut b = Rng(7);
        for _ in 0..1000 {
            let (x, y) = (a.next(), b.next());
            assert_eq!(x, y);
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn reduced_inertia_is_relative_to_the_full_hessian() {
        // -1.8e-12 alone looks negative; against a Hessian of norm 3.3 it is
        // rounding noise.
        assert_eq!(inertia(&[-1.8e-12]).1, 1);
        assert_eq!(inertia_scaled(&[-1.8e-12], 3.3).1, 0);
        assert_eq!(inertia_scaled(&[-0.9], 1.8).1, 1);
    }

    #[test]
    fn inertia_counts() {
        assert_eq!(inertia(&[-2.0, 0.0, 1e-12, 3.0]).0, 1);
        assert_eq!(inertia(&[-2.0, 0.0, 1e-12, 3.0]).1, 1);
        assert_eq!(inertia(&[-2.0, 0.0, 1e-12, 3.0]).2, 2);
    }
}
