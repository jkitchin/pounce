//! Phase 2 (gh#1160): constraint qualifications and multipliers at a
//! solution.
//!
//! These checks need the point a solve stopped at and, for the multiplier
//! checks, its constraint multipliers; both come from `--sol`.
//!
//! * `kkt_audit` — reconstructs the Lagrangian gradient from the `.sol`
//!   multipliers. The sign convention is resolved by picking the sign that
//!   makes the point stationary, as `pounce verify` does, and bound
//!   multipliers are inferred from the residual at active bounds. It then
//!   reports a residual a valid bound multiplier cannot absorb
//!   (`not_stationary_at_point`), multipliers of the wrong sign, nonzero
//!   multipliers on inactive rows, active constraints with zero multiplier
//!   (strict complementarity fails), and inactive inequalities within a
//!   stated band of their bound.
//!
//! * `licq` — the rank of the active constraint gradients: equality rows,
//!   active inequality rows, and active variable bounds (unit vectors, so
//!   their columns are removed and counted directly). When it is deficient,
//!   the minimal dependent sets are localized as in `numerical_rank`, and,
//!   given multipliers, the range over which they can move along the
//!   dependency without breaking stationarity or sign is reported
//!   (`nonunique_multipliers`).
//!
//! * `mfcq` — Mangasarian–Fromovitz: the equality gradients must be
//!   independent and some direction must keep the equalities and strictly
//!   decrease every active inequality. With the active inequality gradients
//!   projected onto the null space of the equality gradients, that holds
//!   exactly when 0 is outside their convex hull. Wolfe's minimum-norm-point
//!   algorithm finds the closest point of the hull: a nonzero one gives the
//!   direction (verified), a zero one gives the failure witness, nonnegative
//!   weights making the inequality gradients cancel.
//!
//! All of it is a numerical observation at the solution point under the
//! stated tolerances.

use super::report::{Basis, Confidence, Finding, Report, Severity, fmt_num, num};
use super::structure::{Dense, feral_rank, fixed_mask, list, minimal_dependent_sets, sparse_block};
use super::{DiagnoseOptions, PointEval, cname, lo_eff, up_eff, vname};
use crate::nl_reader::NlProblem;
use faer::Mat;
use faer::linalg::solvers::Solve;
use pounce_common::types::Number;
use serde_json::json;
use std::collections::BTreeMap;

/// The reconstructed KKT data: stationarity residual, z_L, z_U, row
/// multipliers (evaluator convention), and the sign applied to the file's.
type Kkt = (Vec<Number>, Vec<Number>, Vec<Number>, Vec<Number>, Number);

/// One entry of the active set, oriented as `a(x) ≤ 0` for inequalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Active {
    Eq(usize),
    RowLower(usize),
    RowUpper(usize),
    VarLower(usize),
    VarUpper(usize),
}

impl Active {
    /// `+1` or `-1`: the oriented gradient is `sign · ∇g` (or `sign · e_j`).
    pub fn sign(self) -> Number {
        match self {
            Active::RowLower(_) | Active::VarLower(_) => -1.0,
            _ => 1.0,
        }
    }
    pub fn is_inequality(self) -> bool {
        !matches!(self, Active::Eq(_))
    }
    pub fn name(self, prob: &NlProblem) -> String {
        match self {
            Active::Eq(i) => cname(prob, i),
            Active::RowLower(i) => format!("{} (at lower bound)", cname(prob, i)),
            Active::RowUpper(i) => format!("{} (at upper bound)", cname(prob, i)),
            Active::VarLower(j) => format!("lower bound of {}", vname(prob, j)),
            Active::VarUpper(j) => format!("upper bound of {}", vname(prob, j)),
        }
    }
}

/// What the KKT reconstruction found, shared with the convexity check.
pub struct KktState {
    /// The active set at the point.
    pub active: Vec<Active>,
    /// Multiplier of each active entry, in the oriented convention
    /// (`≥ 0` for a valid inequality multiplier, free for equalities).
    /// `None` when the `.sol` carries no multipliers.
    pub mu: Option<Vec<Number>>,
    /// Row multipliers in the evaluator's convention
    /// (`∇f + Jᵀλ − z_L + z_U = 0`), for the Lagrangian Hessian.
    pub lambda: Option<Vec<Number>>,
    /// Tolerance under which a multiplier counts as zero.
    pub mult_tol: Number,
}

fn act_tol(b: Number, tol: Number) -> Number {
    tol * (1.0 + b.abs())
}

/// Is a side at distance `dist` from finite bound `b` active? Either it is
/// within the activity tolerance, or it carries a multiplier and multiplier
/// × slack is within the complementarity tolerance — which is where an
/// interior-point solution leaves a binding bound (z·slack ≈ μ).
fn side_active(b: Number, dist: Number, mult: Number, c: &Tols) -> bool {
    b.is_finite() && (dist <= act_tol(b, c.act) || (mult > c.mult && mult * dist <= c.comp))
}

/// The tolerances the active-set decision uses.
struct Tols {
    act: Number,
    mult: Number,
    comp: Number,
}

/// Classify the active set at `pt`, given row multipliers (`|λ_i|`) and
/// bound multipliers.
fn active_set(
    prob: &NlProblem,
    pt: &PointEval,
    t: &Tols,
    row_mult: Option<&[Number]>,
    zl: &[Number],
    zu: &[Number],
) -> Vec<Active> {
    let fixed = fixed_mask(prob);
    let mut out = Vec::new();
    for i in 0..prob.m {
        let (gl, gu) = (lo_eff(prob.g_l[i]), up_eff(prob.g_u[i]));
        let g = pt.g.get(i).copied().unwrap_or(Number::NAN);
        if gl == gu && gl.is_finite() {
            out.push(Active::Eq(i));
            continue;
        }
        let m = row_mult.map(|l| l[i].abs()).unwrap_or(0.0);
        let dl = (g - gl).abs();
        let du = (gu - g).abs();
        // A missing side is never active (its distance and tolerance are
        // both infinite, so it has to be excluded explicitly).
        let al = side_active(gl, dl, m, t);
        let au = side_active(gu, du, m, t);
        if al && (!au || dl <= du) {
            out.push(Active::RowLower(i));
        } else if au {
            out.push(Active::RowUpper(i));
        }
    }
    for j in 0..prob.n {
        if fixed[j] {
            continue;
        }
        let (xl, xu) = (lo_eff(prob.x_l[j]), up_eff(prob.x_u[j]));
        let x = pt.x[j];
        let al = side_active(xl, (x - xl).abs(), zl[j], t);
        let au = side_active(xu, (xu - x).abs(), zu[j], t);
        if al && (!au || (x - xl).abs() <= (xu - x).abs()) {
            out.push(Active::VarLower(j));
        } else if au {
            out.push(Active::VarUpper(j));
        }
    }
    out
}

/// `∇f + σ·Jᵀλ` at the point.
fn lagrangian_gradient(pt: &PointEval, n: usize, lambda: &[Number], sigma: Number) -> Vec<Number> {
    let mut s = pt.grad.clone();
    s.resize(n, 0.0);
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        let c = pt.jac_jcol[k];
        if r < lambda.len() && c < n {
            s[c] += sigma * pt.jac_val[k] * lambda[r];
        }
    }
    s
}

/// Without bound multipliers in the `.sol`: infer them. `s_j > 0` is
/// absorbed by a lower bound and `s_j < 0` by an upper bound, when that bound
/// is active by distance or by complementarity with the multiplier it would
/// need. Returns (residual, z_L, z_U).
fn infer_bounds(
    prob: &NlProblem,
    x: &[Number],
    s: &[Number],
    t: &Tols,
) -> (Vec<Number>, Vec<Number>, Vec<Number>) {
    let fixed = fixed_mask(prob);
    let n = s.len();
    let (mut r, mut zl, mut zu) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for j in 0..n {
        if fixed[j] {
            continue;
        }
        let (xl, xu) = (lo_eff(prob.x_l[j]), up_eff(prob.x_u[j]));
        let v = s[j];
        if v > 0.0 && side_active(xl, (x[j] - xl).abs(), v, t) {
            zl[j] = v;
        } else if v < 0.0 && side_active(xu, (xu - x[j]).abs(), -v, t) {
            zu[j] = -v;
        } else {
            r[j] = v;
        }
    }
    (r, zl, zu)
}

/// How many inequality multipliers have the wrong sign under a candidate
/// convention: rows near a bound whose multiplier points the wrong way, and
/// negative bound multipliers.
fn sign_violations(
    prob: &NlProblem,
    pt: &PointEval,
    t: &Tols,
    lam: &[Number],
    zl: &[Number],
    zu: &[Number],
) -> usize {
    let mut bad = zl.iter().chain(zu).filter(|&&z| z < -t.mult).count();
    for (i, &l) in lam.iter().enumerate() {
        let (gl, gu) = (lo_eff(prob.g_l[i]), up_eff(prob.g_u[i]));
        if gl == gu || l.abs() <= t.mult {
            continue;
        }
        let g = pt.g[i];
        // Evaluator convention: λ ≥ 0 at an upper bound, ≤ 0 at a lower one.
        let near_upper = (gu - g).abs() <= (g - gl).abs();
        if (near_upper && l < 0.0) || (!near_upper && l > 0.0) {
            bad += 1;
        }
    }
    bad
}

fn inf_norm(v: &[Number]) -> Number {
    v.iter().fold(0.0_f64, |a, x| a.max(x.abs()))
}

/// Run `kkt_audit`, `licq`, `mfcq`. Returns the state for the convexity
/// check, or `None` when there is no solution point.
pub fn check_kkt(
    prob: &NlProblem,
    pt: &PointEval,
    lambda_file: Option<&[Number]>,
    bounds_file: Option<(&[Number], &[Number])>,
    opts: &DiagnoseOptions,
    report: &mut Report,
) -> Option<KktState> {
    for c in ["kkt_audit", "licq", "mfcq"] {
        if !pt.is_solution {
            report.unavailable(
                c,
                "needs a solution point: pass the solve's .sol with --sol",
            );
        }
    }
    if !pt.is_solution {
        return None;
    }
    if !pt.jac_ok || !pt.grad_ok || pt.jac_val.iter().chain(&pt.grad).any(|v| !v.is_finite()) {
        for c in ["kkt_audit", "licq", "mfcq"] {
            report.unavailable(c, format!("the derivatives are not finite at {}", pt.label));
        }
        return None;
    }
    let n = prob.n;
    let grad_scale = pt.grad.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let mult_tol = opts.opt_tol * grad_scale;
    let t = Tols {
        act: opts.feas_tol,
        mult: mult_tol,
        comp: opts.opt_tol * grad_scale,
    };
    report.tolerances.insert("activity_tol", opts.feas_tol);
    report.tolerances.insert("multiplier_zero_tol", mult_tol);

    // Row multipliers in the evaluator's convention (∇f + Jᵀλ − z_L + z_U
    // = 0), the bound multipliers, and the stationarity residual. The .sol
    // sign conventions are resolved by taking whichever signs make the
    // point stationary, as `pounce verify` does.
    let row_l: Option<&[Number]> = match lambda_file {
        Some(l) if l.len() == prob.m => Some(l),
        _ if prob.m == 0 => Some(&[]),
        _ => None,
    };
    #[allow(clippy::type_complexity)]
    let mut kkt: Option<(
        Vec<Number>,
        Vec<Number>,
        Vec<Number>,
        Vec<Number>,
        Number,
        usize,
    )> = None;
    if let Some(l) = row_l {
        for sigma in [1.0, -1.0] {
            let s = lagrangian_gradient(pt, n, l, sigma);
            let cands: Vec<(Vec<Number>, Vec<Number>, Vec<Number>)> = match bounds_file {
                Some((zl_f, zu_f)) => [1.0, -1.0]
                    .iter()
                    .map(|&sz| {
                        let zl: Vec<Number> = (0..n)
                            .map(|j| sz * zl_f.get(j).copied().unwrap_or(0.0))
                            .collect();
                        let zu: Vec<Number> = (0..n)
                            .map(|j| -sz * zu_f.get(j).copied().unwrap_or(0.0))
                            .collect();
                        let r: Vec<Number> = (0..n).map(|j| s[j] - zl[j] + zu[j]).collect();
                        (r, zl, zu)
                    })
                    .collect(),
                None => vec![infer_bounds(prob, &pt.x, &s, &t)],
            };
            for (r, zl, zu) in cands {
                let lam: Vec<Number> = l.iter().map(|v| sigma * v).collect();
                let bad = sign_violations(prob, pt, &t, &lam, &zl, &zu);
                let rn = inf_norm(&r);
                // Prefer the smaller residual; among residuals within the
                // stationarity tolerance of each other, the fewer sign
                // violations. With ∇f = 0 (a feasibility problem) a
                // multiplier set and its negation are equally stationary,
                // and only the signs tell the conventions apart.
                let better = match &kkt {
                    None => true,
                    Some(k) => {
                        let (kr, kb) = (inf_norm(&k.0), k.5);
                        if (rn - kr).abs() <= t.comp {
                            bad < kb
                        } else {
                            rn < kr
                        }
                    }
                };
                if better {
                    kkt = Some((r, zl, zu, lam, sigma, bad));
                }
            }
        }
    }
    let kkt = kkt.map(|(r, zl, zu, lam, sigma, _)| (r, zl, zu, lam, sigma));
    let zero = vec![0.0; n];
    let (zl, zu) = match &kkt {
        Some(k) => (
            k.1.iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
            k.2.iter().map(|v| v.max(0.0)).collect::<Vec<_>>(),
        ),
        None => (zero.clone(), zero),
    };
    let lam_abs: Option<Vec<Number>> = kkt.as_ref().map(|k| k.3.clone());
    let active = active_set(prob, pt, &t, lam_abs.as_deref(), &zl, &zu);

    // Oriented multipliers per active entry.
    let mu: Option<Vec<Number>> = kkt.as_ref().map(|(_, zl, zu, lam, _)| {
        active
            .iter()
            .map(|a| match *a {
                Active::Eq(i) => lam[i],
                Active::RowUpper(i) => lam[i],
                Active::RowLower(i) => -lam[i],
                Active::VarLower(j) => zl[j],
                Active::VarUpper(j) => zu[j],
            })
            .collect()
    });

    kkt_audit(
        prob,
        pt,
        &active,
        &kkt,
        &mu,
        mult_tol,
        grad_scale,
        bounds_file.is_some(),
        opts,
        report,
    );
    let licq_ok = licq(prob, pt, &active, mu.as_deref(), mult_tol, opts, report);
    mfcq(prob, pt, &active, licq_ok, opts, report);

    Some(KktState {
        active,
        mu,
        lambda: kkt.map(|k| k.3),
        mult_tol,
    })
}

#[allow(clippy::too_many_arguments)]
fn kkt_audit(
    prob: &NlProblem,
    pt: &PointEval,
    active: &[Active],
    kkt: &Option<Kkt>,
    mu: &Option<Vec<Number>>,
    mult_tol: Number,
    grad_scale: Number,
    bounds_from_file: bool,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    let (Some((resid, _, _, lam, sigma)), Some(mu)) = (kkt, mu) else {
        report.unavailable("kkt_audit", "the .sol has no constraint multipliers");
        return;
    };
    let (worst_j, worst) = resid
        .iter()
        .enumerate()
        .fold((0usize, 0.0_f64), |b, (j, v)| {
            if v.abs() > b.1 { (j, v.abs()) } else { b }
        });
    let n_active_ineq = active.iter().filter(|a| a.is_inequality()).count();
    report.sections.insert(
        "kkt_audit",
        json!({"point": pt.label, "dual_sign_used": sigma,
            "bound_multipliers": if bounds_from_file { "from the .sol suffixes" } else { "inferred from the residual" },
            "stationarity_residual": num(worst), "gradient_scale": num(grad_scale),
            "active_equalities": active.len() - n_active_ineq, "active_inequalities": n_active_ineq}),
    );
    let mut sink = report.ran("kkt_audit");
    if worst > opts.opt_tol * grad_scale {
        sink.push(
            Finding::new("kkt_audit", "not_stationary_at_point", Severity::Warning, Basis::Numerical,
                Confidence::High,
                format!("At {}, the Lagrangian gradient has a residual of {} (largest at {}) that no valid bound multiplier absorbs; the point is not a KKT point at tolerance {}.",
                    pt.label, fmt_num(worst), vname(prob, worst_j), fmt_num(opts.opt_tol * grad_scale)))
            .why("The constraint-qualification and second-order findings below describe the geometry at this point, but it is not a first-order solution, so they say nothing about optimality.")
            .next("Check the solve's status; compare with `pounce verify --require-optimal`.")
            .vars(vec![vname(prob, worst_j)])
            .evidence(json!({"residual": num(worst), "tolerance": num(opts.opt_tol * grad_scale)})),
        );
    }
    // Sign violations on active inequality rows. (Bound multipliers are
    // clipped at zero, so a wrong-sign bound multiplier shows up as
    // stationarity residual instead.)
    let wrong: Vec<(String, Number)> = active
        .iter()
        .zip(mu)
        .filter(|(a, m)| matches!(a, Active::RowLower(_) | Active::RowUpper(_)) && **m < -mult_tol)
        .map(|(a, m)| (a.name(prob), *m))
        .collect();
    if !wrong.is_empty() {
        let names: Vec<String> = wrong.iter().map(|w| w.0.clone()).collect();
        sink.push(
            Finding::new("kkt_audit", "multiplier_sign_violation", Severity::Warning, Basis::Numerical,
                Confidence::High,
                format!("At {}, {} active inequalit{} ha{} a multiplier of the wrong sign: {}.",
                    pt.label, wrong.len(), if wrong.len() == 1 { "y" } else { "ies" },
                    if wrong.len() == 1 { "s" } else { "ve" }, list(&names)))
            .why("A wrong-sign multiplier means moving off that bound would improve the objective: the point is not a minimizer (it may be a saddle or a maximizer along that constraint).")
            .next("Check the solve's status and whether the bound is meant to be binding.")
            .cons(names)
            .evidence(json!({"multipliers": wrong.iter().map(|w| json!({"constraint": w.0, "oriented_multiplier": w.1})).collect::<Vec<_>>()})),
        );
    }
    // Complementarity: nonzero multipliers on inactive inequality rows.
    let mut comp: Vec<(String, Number, Number)> = Vec::new();
    for i in 0..prob.m {
        let (gl, gu) = (lo_eff(prob.g_l[i]), up_eff(prob.g_u[i]));
        if gl == gu {
            continue;
        }
        let is_active = active
            .iter()
            .any(|a| matches!(a, Active::RowLower(k) | Active::RowUpper(k) if *k == i));
        if !is_active && lam[i].abs() > mult_tol {
            let g = pt.g[i];
            let slack = (g - gl).abs().min((gu - g).abs());
            // Inactive here already means |λ|·slack exceeds the
            // complementarity tolerance.
            comp.push((cname(prob, i), lam[i], slack));
        }
    }
    if !comp.is_empty() {
        let names: Vec<String> = comp.iter().map(|c| c.0.clone()).collect();
        sink.push(
            Finding::new("kkt_audit", "complementarity_violation", Severity::Warning, Basis::Numerical,
                Confidence::High,
                format!("At {}, {} inactive row(s) carry a nonzero multiplier: {}.", pt.label, comp.len(), list(&names)))
            .why("A row that is not at its bound should have a zero multiplier; a nonzero one means the point and the multipliers do not belong together (a stale .sol, or a solve that stopped early).")
            .next("Check the solve's status and that the .sol belongs to this model.")
            .cons(names)
            .evidence(json!({"rows": comp.iter().map(|c| json!({"row": c.0, "multiplier": c.1, "slack": c.2})).collect::<Vec<_>>()})),
        );
    }
    // Strict complementarity: active inequalities with zero multiplier.
    let weak: Vec<String> = active
        .iter()
        .zip(mu)
        .filter(|(a, m)| a.is_inequality() && m.abs() <= mult_tol)
        .map(|(a, _)| a.name(prob))
        .collect();
    if !weak.is_empty() {
        sink.push(
            Finding::new("kkt_audit", "weakly_active_constraints", Severity::Info, Basis::Numerical,
                Confidence::Medium,
                format!("At {}, {} active inequalit{} have a zero multiplier (strict complementarity fails): {}.",
                    pt.label, weak.len(), if weak.len() == 1 { "y" } else { "ies" }, list(&weak)))
            .why("Weakly active constraints make the active set ambiguous near the solution: interior-point methods converge more slowly there, and the solution's sensitivity to parameters is not differentiable.")
            .next("Check whether these bounds are meant to bind; see the sensitivity docs for degenerate active sets.")
            .cons(weak),
        );
    }
    near_active(prob, pt, active, opts, &mut sink);
}

/// Inactive inequalities within `weak_band` of their bound.
fn near_active(
    prob: &NlProblem,
    pt: &PointEval,
    active: &[Active],
    opts: &DiagnoseOptions,
    sink: &mut super::report::Sink<'_>,
) {
    let mut near: Vec<(String, Number)> = Vec::new();
    for i in 0..prob.m {
        let (gl, gu) = (lo_eff(prob.g_l[i]), up_eff(prob.g_u[i]));
        if gl == gu
            || active
                .iter()
                .any(|a| matches!(a, Active::RowLower(k) | Active::RowUpper(k) if *k == i))
        {
            continue;
        }
        let g = pt.g[i];
        for b in [gl, gu] {
            if b.is_finite() && (g - b).abs() <= act_tol(b, opts.weak_band) {
                near.push((cname(prob, i), (g - b).abs()));
            }
        }
    }
    let fixed = fixed_mask(prob);
    for j in 0..prob.n {
        if fixed[j]
            || active
                .iter()
                .any(|a| matches!(a, Active::VarLower(k) | Active::VarUpper(k) if *k == j))
        {
            continue;
        }
        for b in [lo_eff(prob.x_l[j]), up_eff(prob.x_u[j])] {
            if b.is_finite() && (pt.x[j] - b).abs() <= act_tol(b, opts.weak_band) {
                near.push((format!("bound of {}", vname(prob, j)), (pt.x[j] - b).abs()));
            }
        }
    }
    if !near.is_empty() {
        let names: Vec<String> = near.iter().map(|n| n.0.clone()).collect();
        sink.push(
            Finding::new(sink.check, "near_active_inequalities", Severity::Info, Basis::Numerical,
                Confidence::High,
                format!("At {}, {} inequalit{} are not active but lie within the band {} of their bound: {}.",
                    pt.label, near.len(), if near.len() == 1 { "y" } else { "ies" }, fmt_num(opts.weak_band), list(&names)))
            .why("Nearly active constraints can switch the active set under small perturbations; activity decisions near the solution are fragile.")
            .next("Widen or tighten --weak-band to see how many constraints sit near their bounds.")
            .cons(names)
            .evidence(json!({"band": opts.weak_band, "entries": near.iter().map(|n| json!({"constraint": n.0, "distance": n.1})).collect::<Vec<_>>()})),
        );
    }
}

/// Dense matrix of the gradients of `rows` (each oriented by `signs`) over
/// `cols` (original variable indices), at `pt`.
fn dense_rows(
    pt: &PointEval,
    rows: &[usize],
    signs: &[Number],
    cols: &[usize],
    m_total: usize,
) -> Dense {
    let mut row_pos = vec![usize::MAX; m_total];
    for (p, &r) in rows.iter().enumerate() {
        row_pos[r] = p;
    }
    let col_pos: BTreeMap<usize, usize> = cols.iter().enumerate().map(|(p, &c)| (c, p)).collect();
    let nc = cols.len().max(1);
    let mut a = vec![0.0; rows.len() * nc];
    for (k, &r) in pt.jac_irow.iter().enumerate() {
        if r >= m_total || row_pos[r] == usize::MAX {
            continue;
        }
        if let Some(&p) = col_pos.get(&pt.jac_jcol[k]) {
            a[row_pos[r] * nc + p] += signs[row_pos[r]] * pt.jac_val[k];
        }
    }
    Dense {
        rows: rows.to_vec(),
        cols: cols.to_vec(),
        a,
    }
}

fn too_big(m: usize, n: usize, opts: &DiagnoseOptions) -> Option<String> {
    (m.saturating_mul(n.max(1)) > opts.max_dense_entries || m.min(n) > opts.max_dense_dim).then(|| {
        format!("the active-gradient block is {m} × {n}, larger than the dense limits (--max-dense-entries, --max-dense-dim)")
    })
}

/// `licq`. Returns whether LICQ holds (`None` when it could not be checked).
fn licq(
    prob: &NlProblem,
    pt: &PointEval,
    active: &[Active],
    mu: Option<&[Number]>,
    mult_tol: Number,
    opts: &DiagnoseOptions,
    report: &mut Report,
) -> Option<bool> {
    let fixed = fixed_mask(prob);
    let bound_cols: Vec<usize> = active
        .iter()
        .filter_map(|a| match *a {
            Active::VarLower(j) | Active::VarUpper(j) => Some(j),
            _ => None,
        })
        .collect();
    // Row entries (position in `active`) and their row indices.
    let row_entries: Vec<usize> = (0..active.len())
        .filter(|&k| {
            matches!(
                active[k],
                Active::Eq(_) | Active::RowLower(_) | Active::RowUpper(_)
            )
        })
        .collect();
    let rows: Vec<usize> = row_entries
        .iter()
        .map(|&k| match active[k] {
            Active::Eq(i) | Active::RowLower(i) | Active::RowUpper(i) => i,
            _ => unreachable!(),
        })
        .collect();
    let signs: Vec<Number> = row_entries.iter().map(|&k| active[k].sign()).collect();
    let free_cols: Vec<usize> = (0..prob.n)
        .filter(|&j| !fixed[j] && !bound_cols.contains(&j))
        .collect();
    let n_free = (0..prob.n).filter(|&j| !fixed[j]).count();
    if let Some(why) = too_big(rows.len(), free_cols.len(), opts) {
        // Too large for the dense SVD: count the rank from feral's inertia
        // instead. Bound columns are removed, as in the dense path, so the
        // count is over the active rows alone.
        let Some(ent) = sparse_block(pt, prob.m, &rows, Some(&signs), &free_cols, prob.n) else {
            report.unavailable(
                "licq",
                format!("the Jacobian is not finite at {}", pt.label),
            );
            return None;
        };
        let rank = match feral_rank(rows.len(), free_cols.len(), &ent) {
            Ok(r) => r,
            Err(e) => {
                report.unavailable(
                    "licq",
                    format!("{why}; and the sparse inertia count failed: {e}"),
                );
                return None;
            }
        };
        let deficiency = rows.len() - rank;
        report.sections.insert(
            "licq",
            json!({"point": pt.label, "active_rows": rows.len(), "active_bounds": bound_cols.len(),
                "rank_of_active_rows_off_bound_columns": rank, "deficiency": deficiency,
                "method": "inertia of [[I, Jᵀ], [J, 0]] (feral)", "dense_svd_skipped": why}),
        );
        let mut sink = report.ran("licq");
        if deficiency > 0 {
            sink.push(
                Finding::new("licq", "licq_fails", Severity::Warning, Basis::Numerical, Confidence::Medium,
                    format!("At {}, the {} active constraint rows are rank deficient by {deficiency} (with the {} active bounds removed), by the inertia of [[I, Jᵀ], [J, 0]]: LICQ fails. The block is too large for the dense search that names the dependent constraints.",
                        pt.label, rows.len(), bound_cols.len()))
                .why("The multipliers of the dependent constraints are not unique and the KKT system is singular.")
                .next("Raise --max-dense-entries / --max-dense-dim to localize the dependent constraints and bound how far the multipliers can move.")
                .evidence(json!({"point": pt.label, "deficiency": deficiency, "method": "feral inertia"})),
            );
        }
        return Some(deficiency == 0);
    }
    let d = dense_rows(pt, &rows, &signs, &free_cols, prob.m);
    let s = d.singular_values().unwrap_or_default();
    let smax = s.first().copied().unwrap_or(0.0);
    let rel = opts
        .rank_rel_tol
        .unwrap_or((rows.len().max(free_cols.len()).max(1) as Number) * Number::EPSILON);
    let tau = smax * rel;
    let rank = s.iter().filter(|&&v| v > tau).count();
    let deficiency = rows.len() - rank;
    report.sections.insert(
        "licq",
        json!({"point": pt.label, "active_rows": rows.len(), "active_bounds": bound_cols.len(),
            "free_variables": n_free, "rank_of_active_rows_off_bound_columns": rank,
            "deficiency": deficiency, "threshold": num(tau)}),
    );
    let mut sink = report.ran("licq");
    if deficiency == 0 {
        return Some(true);
    }
    // Full oriented gradients of the rows over every non-fixed column, for
    // the bound components of each dependency.
    let all_cols: Vec<usize> = (0..prob.n).filter(|&j| !fixed[j]).collect();
    let full = dense_rows(pt, &rows, &signs, &all_cols, prob.m);
    let search = minimal_dependent_sets(
        &d,
        tau,
        deficiency,
        opts.max_dependent_sets,
        opts.max_dependent_search,
    );
    for set in &search.sets {
        // Oriented combination: Σ c_r a_r over rows, plus bound entries that
        // cancel what the rows leave on bound columns.
        let mut entries: Vec<(usize, Number)> = set
            .rows
            .iter()
            .zip(&set.combination)
            .map(|(&p, &c)| (row_entries[p], c))
            .collect();
        let nc = full.ncols();
        for (k, &a) in active.iter().enumerate() {
            let (j, sgn) = match a {
                Active::VarLower(j) => (j, -1.0),
                Active::VarUpper(j) => (j, 1.0),
                _ => continue,
            };
            let Some(p) = all_cols.iter().position(|&c| c == j) else {
                continue;
            };
            let v: Number = set
                .rows
                .iter()
                .zip(&set.combination)
                .map(|(&r, &c)| c * full.a[r * nc + p])
                .sum();
            if v.abs() > 1e-12 {
                entries.push((k, -v * sgn));
            }
        }
        let names: Vec<String> = entries.iter().map(|&(k, _)| active[k].name(prob)).collect();
        sink.push(
            Finding::new("licq", "licq_fails", Severity::Warning, Basis::Numerical, Confidence::High,
                format!("At {}, the active constraints {} have linearly dependent gradients (a minimal dependent set): LICQ fails.",
                    pt.label, list(&names)))
            .why("The multipliers of these constraints are not determined by the point alone; the KKT system is singular, so the solver regularizes and sensitivities are not unique.")
            .next("Check whether one of these constraints is implied by the others at the solution.")
            .cons(names.clone())
            .evidence(json!({"point": pt.label, "combination": entries.iter().map(|&(k, c)| json!({"constraint": active[k].name(prob), "coefficient": c})).collect::<Vec<_>>(),
                "relative_residual": num(set.residual), "threshold": num(tau)})),
        );
        // Non-unique multipliers: μ + t·c stays stationary; keep inequality
        // multipliers non-negative.
        if let Some(mu) = mu {
            let (mut lo, mut hi) = (Number::NEG_INFINITY, Number::INFINITY);
            for &(k, c) in &entries {
                if !active[k].is_inequality() || c == 0.0 {
                    continue;
                }
                let m = mu[k];
                if c > 0.0 {
                    lo = lo.max(-m / c);
                } else {
                    hi = hi.min(m / -c);
                }
            }
            let width = hi - lo;
            let scale = entries
                .iter()
                .map(|&(k, _)| mu[k].abs())
                .fold(mult_tol.max(1e-300), Number::max);
            if width > 1e-8 * scale && lo <= 1e-12 * scale && hi >= -1e-12 * scale {
                sink.push(
                    Finding::new("licq", "nonunique_multipliers", Severity::Warning, Basis::Numerical, Confidence::High,
                        format!("At {}, the multipliers of {} can move along their dependency by t ∈ [{}, {}] without breaking stationarity or sign: the multipliers are not unique.",
                            pt.label, list(&names), fmt_num(lo), fmt_num(hi)))
                    .why("Reported multipliers (shadow prices) for these constraints are one choice among many; any value along the segment is equally valid, so do not interpret them individually.")
                    .next("Interpret the combination, not the individual multipliers, or remove the redundancy.")
                    .cons(names)
                    .evidence(json!({"t_range": [num(lo), num(hi)],
                        "direction": entries.iter().map(|&(k, c)| json!({"constraint": active[k].name(prob), "multiplier": mu[k], "direction": c})).collect::<Vec<_>>()})),
                );
            }
        }
    }
    if search.too_large.is_some() || search.sets.is_empty() {
        sink.push(
            Finding::new("licq", "licq_fails", Severity::Warning, Basis::Numerical, Confidence::Medium,
                format!("At {}, the active constraint gradients are rank deficient by {deficiency}, but the dependent sets were not localized (more rows than --max-dependent-search).", pt.label))
            .why("LICQ fails: multipliers are not unique and the KKT system is singular.")
            .next("Raise --max-dependent-search."),
        );
    }
    Some(false)
}

/// Wolfe's minimum-norm-point algorithm over the convex hull of the rows of
/// `q` (each a point in `R^d`). Returns the weights (summing to 1) and the
/// point.
pub(crate) fn min_norm_point(q: &[Vec<Number>]) -> (Vec<Number>, Vec<Number>) {
    let k = q.len();
    let d = q.first().map(|v| v.len()).unwrap_or(0);
    let dot = |a: &[Number], b: &[Number]| a.iter().zip(b).map(|(x, y)| x * y).sum::<Number>();
    let comb = |s: &[usize], w: &[Number]| {
        let mut x = vec![0.0; d];
        for (&i, &wi) in s.iter().zip(w) {
            for (xj, qj) in x.iter_mut().zip(&q[i]) {
                *xj += wi * qj;
            }
        }
        x
    };
    let scale = q
        .iter()
        .map(|v| dot(v, v))
        .fold(0.0, Number::max)
        .max(1e-300);
    let start = (0..k)
        .min_by(|&a, &b| dot(&q[a], &q[a]).total_cmp(&dot(&q[b], &q[b])))
        .unwrap_or(0);
    let mut s = vec![start];
    let mut w = vec![1.0];
    let mut x = q[start].clone();
    for _major in 0..(10 * k + 100) {
        let (j, best) = (0..k).fold((0, Number::INFINITY), |b, i| {
            let v = dot(&q[i], &x);
            if v < b.1 { (i, v) } else { b }
        });
        if dot(&x, &x) - best <= 1e-12 * scale || s.contains(&j) {
            break;
        }
        s.push(j);
        w.push(0.0);
        for _minor in 0..(10 * k + 100) {
            // Affine minimizer over s: [G 1; 1ᵀ 0][α; ν] = [0; 1].
            let p = s.len();
            let mut a = Mat::<Number>::zeros(p + 1, p + 1);
            let mut rhs = Mat::<Number>::zeros(p + 1, 1);
            for r in 0..p {
                for c in 0..p {
                    a[(r, c)] = dot(&q[s[r]], &q[s[c]]);
                }
                a[(r, r)] += 1e-14 * scale;
                a[(r, p)] = 1.0;
                a[(p, r)] = 1.0;
            }
            rhs[(p, 0)] = 1.0;
            let sol = a.partial_piv_lu().solve(&rhs);
            let alpha: Vec<Number> = (0..p).map(|r| sol[(r, 0)]).collect();
            if alpha.iter().all(|&v| v > 1e-12) {
                w = alpha;
                x = comb(&s, &w);
                break;
            }
            let theta = (0..p)
                .filter(|&r| alpha[r] <= 1e-12)
                .map(|r| {
                    if w[r] - alpha[r] > 0.0 {
                        w[r] / (w[r] - alpha[r])
                    } else {
                        0.0
                    }
                })
                .fold(1.0, Number::min);
            for r in 0..p {
                w[r] = theta * alpha[r] + (1.0 - theta) * w[r];
            }
            let keep: Vec<usize> = (0..p).filter(|&r| w[r] > 1e-12).collect();
            s = keep.iter().map(|&r| s[r]).collect();
            w = keep.iter().map(|&r| w[r]).collect();
            let tot: Number = w.iter().sum();
            for v in &mut w {
                *v /= tot;
            }
            x = comb(&s, &w);
            if s.len() <= 1 {
                break;
            }
        }
    }
    let mut weights = vec![0.0; k];
    for (&i, &wi) in s.iter().zip(&w) {
        weights[i] = wi;
    }
    (weights, x)
}

/// `mfcq`.
fn mfcq(
    prob: &NlProblem,
    pt: &PointEval,
    active: &[Active],
    licq_ok: Option<bool>,
    opts: &DiagnoseOptions,
    report: &mut Report,
) {
    let fixed = fixed_mask(prob);
    let cols: Vec<usize> = (0..prob.n).filter(|&j| !fixed[j]).collect();
    let eq_rows: Vec<usize> = active
        .iter()
        .filter_map(|a| {
            if let Active::Eq(i) = a {
                Some(*i)
            } else {
                None
            }
        })
        .collect();
    let ineq: Vec<Active> = active
        .iter()
        .copied()
        .filter(|a| a.is_inequality())
        .collect();
    if ineq.is_empty() {
        // MFCQ reduces to independent equality gradients, which licq covers.
        report.ran("mfcq");
        return;
    }
    if let Some(why) = too_big(eq_rows.len().max(ineq.len()), cols.len(), opts) {
        report.unavailable("mfcq", why);
        return;
    }
    let e = dense_rows(pt, &eq_rows, &vec![1.0; eq_rows.len()], &cols, prob.m);
    let nc = cols.len();
    // Null-space basis Z of the equality gradients (columns of V past the rank).
    let (z, e_rank): (Mat<Number>, usize) = if eq_rows.is_empty() {
        (Mat::<Number>::identity(nc, nc), 0)
    } else {
        let full = Mat::from_fn(eq_rows.len(), nc, |i, j| e.a[i * e.ncols() + j]);
        let Ok(svd) = full.svd() else {
            report.unavailable("mfcq", "the SVD did not converge");
            return;
        };
        let sv: Vec<Number> = svd.S().column_vector().iter().copied().collect();
        let smax = sv.first().copied().unwrap_or(0.0);
        let tau = smax
            * opts
                .rank_rel_tol
                .unwrap_or((eq_rows.len().max(nc) as Number) * Number::EPSILON);
        let r = sv.iter().filter(|&&v| v > tau).count();
        let v = svd.V();
        (Mat::from_fn(nc, nc - r, |i, j| v[(i, r + j)]), r)
    };
    if e_rank < eq_rows.len() {
        report.ran("mfcq").push(
            Finding::new("mfcq", "mfcq_fails", Severity::Warning, Basis::Numerical, Confidence::High,
                format!("At {}, the equality constraint gradients are linearly dependent, so MFCQ fails (see licq for the dependent sets).", pt.label))
            .why("Without MFCQ the set of multipliers is unbounded or empty: the solver's multiplier estimates can diverge.")
            .next("Remove the redundant equality, or check that the dependency is consistent."),
        );
        return;
    }
    // Oriented inequality gradients projected onto null(E): q = Zᵀ a.
    let col_pos: BTreeMap<usize, usize> = cols.iter().enumerate().map(|(p, &c)| (c, p)).collect();
    let mut a_full: Vec<Vec<Number>> = Vec::new();
    let ineq_rows: Vec<(usize, usize)> = ineq
        .iter()
        .enumerate()
        .filter_map(|(k, a)| match a {
            Active::RowLower(i) | Active::RowUpper(i) => Some((k, *i)),
            _ => None,
        })
        .collect();
    let rows_only: Vec<usize> = ineq_rows.iter().map(|r| r.1).collect();
    let signs: Vec<Number> = ineq_rows.iter().map(|r| ineq[r.0].sign()).collect();
    let rd = dense_rows(pt, &rows_only, &signs, &cols, prob.m);
    let mut row_k = 0;
    for a in &ineq {
        let mut g = vec![0.0; nc];
        match *a {
            Active::RowLower(_) | Active::RowUpper(_) => {
                g.copy_from_slice(&rd.a[row_k * rd.ncols()..row_k * rd.ncols() + nc]);
                row_k += 1;
            }
            Active::VarLower(j) => g[col_pos[&j]] = -1.0,
            Active::VarUpper(j) => g[col_pos[&j]] = 1.0,
            Active::Eq(_) => unreachable!(),
        }
        a_full.push(g);
    }
    let zd = z.ncols();
    let q: Vec<Vec<Number>> = a_full
        .iter()
        .map(|g| {
            (0..zd)
                .map(|c| (0..nc).map(|r| z[(r, c)] * g[r]).sum())
                .collect()
        })
        .collect();
    let qmax = q
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<Number>().sqrt())
        .fold(0.0, Number::max);
    let (weights, p) = if zd == 0 {
        (vec![1.0 / q.len() as Number; q.len()], Vec::new())
    } else {
        min_norm_point(&q)
    };
    let pnorm = p.iter().map(|x| x * x).sum::<Number>().sqrt();
    let tol = 1e-8 * qmax.max(1e-300);
    report.sections.insert("mfcq", json!({"point": pt.label, "active_inequalities": ineq.len(),
        "equality_rank": e_rank, "null_space_dimension": zd,
        "min_norm_of_projected_hull": num(pnorm), "relative": num(if qmax > 0.0 { pnorm / qmax } else { 0.0 })}));
    let mut sink = report.ran("mfcq");
    if pnorm <= tol {
        let witness: Vec<(String, Number)> = ineq
            .iter()
            .zip(&weights)
            .filter(|(_, w)| **w > 1e-10)
            .map(|(a, w)| (a.name(prob), *w))
            .collect();
        let names: Vec<String> = witness.iter().map(|w| w.0.clone()).collect();
        sink.push(
            Finding::new("mfcq", "mfcq_fails", Severity::Warning, Basis::Numerical, Confidence::High,
                format!("At {}, MFCQ fails: no direction keeps the equalities and strictly decreases every active inequality. The gradients of {} cancel with nonnegative weights (modulo the equality gradients).",
                    pt.label, list(&names)))
            .why("Without MFCQ the multiplier set is unbounded or empty: the solver's multipliers can diverge, and the point may be a Fritz–John point that is not a KKT point. Typical causes are inequalities that pinch the feasible set to a point or a lower-dimensional piece, or complementarity-style constraints.")
            .next("Check whether these inequalities force an implicit equality (x ≤ a and x ≥ a written as two inequalities, or a product constraint).")
            .cons(names)
            .evidence(json!({"point": pt.label, "witness_weights": witness.iter().map(|w| json!({"constraint": w.0, "weight": w.1})).collect::<Vec<_>>(),
                "min_norm": num(pnorm), "tolerance": num(tol), "method": "Wolfe minimum-norm point of the projected active inequality gradients"})),
        );
        return;
    }
    // Direction d = −Z p; verify a·d < 0 for every active inequality.
    let dvec: Vec<Number> = (0..nc)
        .map(|r| -(0..zd).map(|c| z[(r, c)] * p[c]).sum::<Number>())
        .collect();
    let dn = dvec.iter().map(|x| x * x).sum::<Number>().sqrt();
    let margin = a_full
        .iter()
        .map(|g| {
            let gn = g.iter().map(|x| x * x).sum::<Number>().sqrt().max(1e-300);
            -g.iter().zip(&dvec).map(|(a, b)| a * b).sum::<Number>() / (gn * dn)
        })
        .fold(Number::INFINITY, Number::min);
    if margin > 0.0 {
        if licq_ok == Some(false) {
            sink.push(
                Finding::new("mfcq", "mfcq_holds", Severity::Info, Basis::Numerical, Confidence::High,
                    format!("At {}, MFCQ holds although LICQ fails: a direction exists that keeps the equalities and strictly decreases every active inequality (worst cosine margin {}).", pt.label, fmt_num(margin)))
                .why("The multipliers are bounded but not unique: the solver converges, but individual multipliers are not meaningful.")
                .next("See licq for which multipliers are free to move.")
                .evidence(json!({"point": pt.label, "cosine_margin": margin, "min_norm": num(pnorm)})),
            );
        }
    } else {
        sink.push(
            Finding::new("mfcq", "mfcq_inconclusive", Severity::Info, Basis::Numerical, Confidence::Low,
                format!("At {}, the MFCQ check is inconclusive: the hull's minimum-norm point is {} but the derived direction does not strictly decrease every active inequality.", pt.label, fmt_num(pnorm)))
            .why("Near the tolerance the check cannot distinguish a weak qualification from a failed one.")
            .next("Re-run with a tighter or looser --feas-tol to change which constraints count as active."),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_norm_point_of_segment_and_of_hull_containing_origin() {
        // Segment from (1, 1) to (1, -1): closest point (1, 0).
        let (w, x) = min_norm_point(&[vec![1.0, 1.0], vec![1.0, -1.0]]);
        assert!((x[0] - 1.0).abs() < 1e-12 && x[1].abs() < 1e-12, "{x:?}");
        assert!((w[0] - 0.5).abs() < 1e-12);
        // Triangle around the origin: closest point is 0.
        let (_, x) = min_norm_point(&[vec![1.0, 0.0], vec![-1.0, 1.0], vec![-1.0, -1.0]]);
        assert!(x.iter().all(|v| v.abs() < 1e-12), "{x:?}");
        // A single point.
        let (_, x) = min_norm_point(&[vec![3.0, 4.0]]);
        assert_eq!(x, vec![3.0, 4.0]);
    }
}
