//! Independent verification of a claimed solution.
//!
//! # Why this exists
//!
//! When a solver is *a tool something else calls*, the caller should never be
//! the thing you trust for "the solution satisfies the constraints". Trust
//! belongs to a small, deterministic checker that re-derives the answer from
//! the canonical problem — not from the caller's narration and not even from
//! the solver's own exit string. Optimization is the rare setting where this
//! is cheap: a claimed `x*` is just numbers, and feasibility is one
//! constraint evaluation (`g_l ≤ g(x*) ≤ g_u`, `x_l ≤ x* ≤ x_u`), `O(nnz)`
//! work with no resolve.
//!
//! [`verify_tnlp`] is that check over any `&mut dyn TNLP`. The `pounce
//! verify` subcommand is this function plus a `.sol` parser, file hashing
//! and a signed receipt; an embedder driving `IpoptApplication` directly has
//! the same reason to want it, and used to have no way to reach it.
//!
//! # What it checks
//!
//! * **Feasibility** — the worst bound and constraint violation, judged
//!   per-row and scale-relative (see [`super::row_is_violated`]): a single
//!   absolute threshold across rows of wildly different magnitude answers a
//!   different question for each of them.
//! * **First-order optimality**, when the claim carries duals — the
//!   bound-projected stationarity residual, the dual sign convention it
//!   holds under, and row complementarity.
//! * **Exact dual infeasibility**, when the claim also carries bound
//!   multipliers `z_L` / `z_U` — strictly sharper than the projected
//!   residual, and what a solver itself reports.

use super::{RowReport, box_violation, name_at, row_is_violated, row_magnitude};
use crate::tnlp::{BoundsInfo, IndexStyle, SparsityRequest, TNLP};
use pounce_common::types::{Number, lower_bound_present, upper_bound_present};

/// A claimed solution, as the checker needs it.
///
/// `lambda` empty means no constraint duals were supplied, which switches the
/// optimality half of the check off rather than treating the duals as zero.
/// Likewise `z_l` / `z_u` are `None` when absent: bound complementarity is
/// then *not checked*, never silently reported as some other quantity.
#[derive(Debug, Clone, Default)]
pub struct SolutionClaim {
    pub x: Vec<Number>,
    pub lambda: Vec<Number>,
    pub z_l: Option<Vec<Number>>,
    pub z_u: Option<Vec<Number>>,
}

/// Tolerances and the strictness of the final verdict.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Scale-relative feasibility tolerance (default 1e-6).
    pub feas_tol: Number,
    /// First-order optimality tolerance (default 1e-6).
    pub opt_tol: Number,
    /// Require first-order optimality, not just feasibility, to report
    /// `verified`.
    pub require_optimal: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions {
            feas_tol: 1e-6,
            opt_tol: 1e-6,
            require_optimal: false,
        }
    }
}

/// Provenance the checker records but does not compute.
///
/// The hashes content-address the receipt the CLI signs; a library caller
/// with no files leaves them empty. Kept as plain `String` rather than
/// `Option<String>` so the signing preimage in `pounce verify` is unchanged.
#[derive(Debug, Clone, Default)]
pub struct VerifyProvenance {
    pub nl_sha256: String,
    pub sol_sha256: String,
    /// The `objno` line's `solve_result_num`, when the claim came from a
    /// `.sol`. Recorded for the receipt; never trusted for the verdict.
    pub solve_result_num: Option<i32>,
}

/// The fully-evaluated verification result. Serialized to the JSON
/// receipt and rendered to the console.
#[derive(Debug)]
pub struct VerifyOutcome {
    pub n_vars: usize,
    pub n_cons: usize,
    pub nl_sha256: String,
    pub sol_sha256: String,
    pub solve_result_num: Option<i32>,
    pub feas_tol: Number,
    pub opt_tol: Number,
    // feasibility
    pub max_con_violation: Number,
    pub worst_con: Option<RowReport>,
    pub max_bound_violation: Number,
    pub worst_bound: Option<RowReport>,
    pub feasible: bool,
    // optimality (only when duals supplied)
    pub objective: Option<Number>,
    pub duals_present: bool,
    pub stationarity: Option<Number>,
    pub dual_sign: Option<i32>,
    /// `max_i |λ_i| · dist(g_i, active side)` over **rows**. NOT the
    /// quantity a solver reports as `Complementarity` — see
    /// [`bound_complementarity`](VerifyOutcome::bound_complementarity).
    pub constraint_complementarity: Option<Number>,
    /// Whether the `.sol` carried `ipopt_zL_out` / `ipopt_zU_out`.
    pub bound_multipliers_present: bool,
    /// `max_j max(|z_L·(x−x_L)|, |z_U·(x_U−x)|)` over **variables** — the
    /// quantity Ipopt prints as `Complementarity`. `None` when the `.sol`
    /// carried no bound multipliers, in which case it is *not checked*
    /// rather than zero.
    pub bound_complementarity: Option<Number>,
    /// Exact (non-projected) dual infeasibility
    /// `‖∇f + sign·Jᵀλ − (z_L^suffix + z_U^suffix)‖∞`, available only when
    /// both duals and bound multipliers are present.
    pub stationarity_with_bound_multipliers: Option<Number>,
    pub optimal: Option<bool>,
    // final
    pub verified: bool,
}

/// Check `claim` against the model `tnlp` presents.
///
/// Returns `Err` only when the model or the claim cannot be read at all (a
/// failed evaluation, a length mismatch). A claim that is simply *wrong*
/// comes back as an `Ok` outcome with `verified == false` — being wrong is a
/// result, not an error.
pub fn verify_tnlp(
    tnlp: &mut dyn TNLP,
    claim: &SolutionClaim,
    var_names: &[String],
    con_names: &[String],
    provenance: &VerifyProvenance,
    opts: &VerifyOptions,
) -> Result<VerifyOutcome, String> {
    let info = tnlp.get_nlp_info().ok_or("get_nlp_info failed")?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    let nnz = info.nnz_jac_g.max(0) as usize;
    let fortran = matches!(info.index_style, IndexStyle::Fortran);

    if claim.x.len() != n {
        return Err(format!(
            "solution has {} primal values but the problem has {n} variables \
             (is this the right solution for this problem?)",
            claim.x.len()
        ));
    }
    let duals_present = !claim.lambda.is_empty();
    if duals_present && claim.lambda.len() != m {
        return Err(format!(
            "solution carries {} dual values but the problem has {m} constraints",
            claim.lambda.len()
        ));
    }
    for (label, z) in [("ipopt_zL_out", &claim.z_l), ("ipopt_zU_out", &claim.z_u)] {
        if let Some(z) = z
            && z.len() != n
        {
            return Err(format!(
                "solution carries {} {label} values but the problem has {n} variables",
                z.len()
            ));
        }
    }

    let x = claim.x.clone();
    let parsed = claim;
    let nl_sha256 = provenance.nl_sha256.clone();
    let sol_sha256 = provenance.sol_sha256.clone();

    // --- bounds ---
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !tnlp.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return Err("get_bounds_info failed".to_string());
    }

    // --- bound feasibility ---
    let mut max_bound_violation = 0.0_f64;
    let mut worst_bound: Option<RowReport> = None;
    let mut any_bound_violated = false;
    for j in 0..n {
        let viol = box_violation(x[j], x_l[j], x_u[j]);
        if row_is_violated(viol, row_magnitude(x[j], x_l[j], x_u[j]), opts.feas_tol) {
            any_bound_violated = true;
        }
        if viol > max_bound_violation {
            max_bound_violation = viol;
            worst_bound = Some(RowReport {
                index: j,
                name: name_at(&var_names, j, 'x'),
                value: x[j],
                lo: x_l[j],
                hi: x_u[j],
                violation: viol,
            });
        }
    }

    // --- constraint feasibility ---
    let mut g = vec![0.0; m];
    if !tnlp.eval_g(&x, true, &mut g) {
        return Err("eval_g failed at the claimed solution".to_string());
    }
    let mut max_con_violation = 0.0_f64;
    let mut worst_con: Option<RowReport> = None;
    let mut any_con_violated = false;
    for i in 0..m {
        let viol = box_violation(g[i], g_l[i], g_u[i]);
        if row_is_violated(viol, row_magnitude(g[i], g_l[i], g_u[i]), opts.feas_tol) {
            any_con_violated = true;
        }
        if viol > max_con_violation {
            max_con_violation = viol;
            worst_con = Some(RowReport {
                index: i,
                name: name_at(&con_names, i, 'c'),
                value: g[i],
                lo: g_l[i],
                hi: g_u[i],
                violation: viol,
            });
        }
    }

    // Per-row and scale-relative: a single absolute threshold across rows of
    // wildly different magnitude answers a different question for each of them.
    let feasible = !any_con_violated && !any_bound_violated;

    // --- objective ---
    let objective = tnlp.eval_f(&x, true);

    // --- bound multipliers, when the `.sol` exported them (gh #516) ---
    //
    // The bound complementarity Ipopt prints as `Complementarity` cannot be
    // computed from the primal and the constraint duals alone; it needs
    // `z_L` / `z_U`, which reach a `.sol` only as the `ipopt_zL_out` /
    // `ipopt_zU_out` variable suffixes. Absent them the quantity is *not
    // checked* — never silently reported as the row quantity.
    let bound_multipliers_present = parsed.z_l.is_some() || parsed.z_u.is_some();
    let z_l_suf = parsed.z_l.clone().unwrap_or_else(|| vec![0.0; n]);
    let z_u_suf = parsed.z_u.clone().unwrap_or_else(|| vec![0.0; n]);
    let bound_complementarity = if bound_multipliers_present {
        Some(bound_complementarity(&x, &x_l, &x_u, &z_l_suf, &z_u_suf))
    } else {
        None
    };

    // --- first-order / KKT stationarity (only when duals are supplied) ---
    let mut stationarity = None;
    let mut dual_sign = None;
    let mut constraint_complementarity = None;
    let mut stationarity_with_bound_multipliers = None;
    let mut optimal = None;
    // A problem with no rows has no constraint duals to carry, so `∇f` alone
    // is the Lagrangian gradient and the residual is available from an empty
    // dual block — which is what a `.sol` for a bounds-only model has.
    if duals_present || m == 0 {
        let lambda = &parsed.lambda;

        // ∇f(x*)
        let mut grad_f = vec![0.0; n];
        tnlp.eval_grad_f(&x, true, &mut grad_f);

        // Jacobian triplets (structure then values).
        let mut irow = vec![0i32; nnz];
        let mut jcol = vec![0i32; nnz];
        tnlp.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            },
        );
        let mut jval = vec![0.0; nnz];
        tnlp.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Values { values: &mut jval },
        );

        // AMPL's dual sign convention can flip relative to ours; rather
        // than guess, compute the bound-projected stationarity residual
        // for both signs and keep the better one. A genuine KKT point is
        // stationary for exactly one of them; we report which.
        let s_pos = lagrangian_gradient(1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda);
        let s_neg = lagrangian_gradient(-1.0, &grad_f, &irow, &jcol, &jval, fortran, lambda);
        let resid_pos = bound_projected_residual(&s_pos, &x, &x_l, &x_u);
        let resid_neg = bound_projected_residual(&s_neg, &x, &x_l, &x_u);
        let (best_resid, sign, s) = if resid_pos <= resid_neg {
            (resid_pos, 1, &s_pos)
        } else {
            (resid_neg, -1, &s_neg)
        };
        stationarity = Some(best_resid);
        dual_sign = Some(sign);
        constraint_complementarity = Some(row_complementarity(lambda, &g, &g_l, &g_u));

        // With the bound multipliers in hand the residual no longer has to
        // be projected: the exact dual infeasibility is available, and it
        // is what a solver reports. It is also the strictly sharper check —
        // the projection can only *remove* residual — so `--require-optimal`
        // gates on it whenever it exists.
        if bound_multipliers_present {
            stationarity_with_bound_multipliers =
                Some(exact_dual_infeasibility(s, &z_l_suf, &z_u_suf));
        }
        let gate = stationarity_with_bound_multipliers.unwrap_or(best_resid);
        optimal = Some(gate <= opts.opt_tol);
    }

    // Verified = feasible (always required) AND, if --require-optimal,
    // also first-order optimal.
    let verified = feasible && (!opts.require_optimal || optimal.unwrap_or(false));

    Ok(VerifyOutcome {
        n_vars: n,
        n_cons: m,
        nl_sha256,
        sol_sha256,
        solve_result_num: provenance.solve_result_num,
        feas_tol: opts.feas_tol,
        opt_tol: opts.opt_tol,
        max_con_violation,
        worst_con,
        max_bound_violation,
        worst_bound,
        feasible,
        objective,
        duals_present,
        stationarity,
        dual_sign,
        constraint_complementarity,
        bound_multipliers_present,
        bound_complementarity,
        stationarity_with_bound_multipliers,
        optimal,
        verified,
    })
}

/// `s = ∇f + sign·Jᵀλ` — the part of the Lagrangian gradient the constraint
/// duals can account for, before any bound multiplier enters.
fn lagrangian_gradient(
    sign: Number,
    grad_f: &[Number],
    irow: &[i32],
    jcol: &[i32],
    jval: &[Number],
    fortran: bool,
    lambda: &[Number],
) -> Vec<Number> {
    let n = grad_f.len();
    let off = if fortran { 1 } else { 0 };
    let mut s = grad_f.to_vec();
    for k in 0..jval.len() {
        let row = (irow[k] as usize).wrapping_sub(off);
        let col = (jcol[k] as usize).wrapping_sub(off);
        if row < lambda.len() && col < n {
            s[col] += sign * jval[k] * lambda[row];
        }
    }
    s
}

/// Bound-**projected** stationarity (a.k.a. "dual infeasibility"): for each
/// variable, the part of `s` that a valid sign-constrained bound multiplier
/// `z_L, z_U ≥ 0` cannot absorb. Returns `‖projected s‖∞`.
///
/// This is a *relaxation*: it projects out exactly the component a bound
/// multiplier would carry, so it cannot see a missing or wrong `z` (gh #495).
/// When the `.sol` exports the multipliers, prefer
/// [`exact_dual_infeasibility`].
fn bound_projected_residual(s: &[Number], x: &[Number], x_l: &[Number], x_u: &[Number]) -> Number {
    let n = s.len();
    // Activity tolerance for "x_j sits on a bound."
    let mut dual_inf = 0.0_f64;
    for j in 0..n {
        let at_lo =
            lower_bound_present(x_l[j]) && (x[j] - x_l[j]).abs() <= 1e-8 * (1.0 + x_l[j].abs());
        let at_hi =
            upper_bound_present(x_u[j]) && (x_u[j] - x[j]).abs() <= 1e-8 * (1.0 + x_u[j].abs());
        let fixed = lower_bound_present(x_l[j])
            && upper_bound_present(x_u[j])
            && (x_u[j] - x_l[j]).abs() <= 1e-12;
        let r = if fixed {
            0.0
        } else if at_lo && !at_hi {
            // need z_L = s_j ≥ 0; leftover is the negative part.
            (-s[j]).max(0.0)
        } else if at_hi && !at_lo {
            // need z_U = -s_j ≥ 0; leftover is the positive part.
            s[j].max(0.0)
        } else {
            s[j].abs()
        };
        dual_inf = dual_inf.max(r);
    }
    dual_inf
}

/// Exact dual infeasibility `‖s − (z_L^suffix + z_U^suffix)‖∞`, with `s` the
/// [`lagrangian_gradient`] at the sign matching the `.sol`'s dual convention.
///
/// Stationarity in pounce's internal convention is
/// `∇f + Jᵀλ − z_L + z_U = 0` with `z_L, z_U ≥ 0`, and the `.sol` suffixes
/// carry `ipopt_zL_out = +z_L`, `ipopt_zU_out = −z_U` — both equal to the
/// objective-gradient component at the bound, matching Ipopt 3.14 (gh #296).
/// So `−z_L + z_U` is exactly `−(zL_out + zU_out)`, and no sign has to be
/// guessed here beyond the one already chosen for `λ`.
///
/// Unlike [`bound_projected_residual`] this sees a bound multiplier that is
/// missing or wrong, because nothing is projected away.
fn exact_dual_infeasibility(s: &[Number], z_l_suf: &[Number], z_u_suf: &[Number]) -> Number {
    let mut dual_inf = 0.0_f64;
    for (j, &s_j) in s.iter().enumerate() {
        let z = z_l_suf.get(j).copied().unwrap_or(0.0) + z_u_suf.get(j).copied().unwrap_or(0.0);
        dual_inf = dual_inf.max((s_j - z).abs());
    }
    dual_inf
}

/// Bound complementarity over **variables**:
/// `max_j max(|z_L·(x−x_L)|, |z_U·(x_U−x)|)` — the quantity Ipopt prints as
/// `Complementarity` (gh #516). Only variables with a finite bound on the
/// side in question contribute.
///
/// Magnitudes throughout, so the result does not depend on which sign
/// convention the writer used for the multipliers, nor on which side of a
/// bound the point sits.
fn bound_complementarity(
    x: &[Number],
    x_l: &[Number],
    x_u: &[Number],
    z_l_suf: &[Number],
    z_u_suf: &[Number],
) -> Number {
    let mut comp = 0.0_f64;
    for j in 0..x.len() {
        if lower_bound_present(x_l[j]) {
            let z = z_l_suf.get(j).copied().unwrap_or(0.0);
            comp = comp.max((z * (x[j] - x_l[j])).abs());
        }
        if upper_bound_present(x_u[j]) {
            let z = z_u_suf.get(j).copied().unwrap_or(0.0);
            comp = comp.max((z * (x_u[j] - x[j])).abs());
        }
    }
    comp
}

/// `max_i |λ_i| · dist(g_i, active side)` over constraints with a finite
/// range — a constraint with a nonzero multiplier should be active.
/// Equalities (`g_l == g_u`) contribute 0. Best-effort, informational.
///
/// This is **constraint** complementarity, over rows, and is not the
/// quantity a solver reports as `Complementarity` — that one is
/// [`bound_complementarity`], over variables. The two are unrelated in
/// magnitude; see the module docs (gh #516).
fn row_complementarity(lambda: &[Number], g: &[Number], g_l: &[Number], g_u: &[Number]) -> Number {
    let mut comp = 0.0_f64;
    for i in 0..lambda.len() {
        // An equality needs *both* bounds present (gh #403): `g_l = g_u = -5e20`
        // is the one-sided `g <= -5e20`, not an equality at `-5e20`, and
        // skipping it here would drop a real complementarity term.
        if lower_bound_present(g_l[i])
            && upper_bound_present(g_u[i])
            && (g_u[i] - g_l[i]).abs() <= 1e-12
        {
            continue; // equality: multiplier is free, no complementarity
        }
        let dl = if lower_bound_present(g_l[i]) {
            (g[i] - g_l[i]).abs()
        } else {
            Number::INFINITY
        };
        let du = if upper_bound_present(g_u[i]) {
            (g_u[i] - g[i]).abs()
        } else {
            Number::INFINITY
        };
        let dist = dl.min(du);
        if dist.is_finite() {
            comp = comp.max(lambda[i].abs() * dist);
        }
    }
    comp
}

#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::types::{NLP_LOWER_BOUND_INF, NLP_UPPER_BOUND_INF};

    /// `min (x−3)² + (y+2)²  s.t.  x ≤ 1, y ≥ −1` — the model whose export
    /// convention is pinned in `main.rs` (gh #296): `ipopt_zL_out = +z_L`,
    /// `ipopt_zU_out = −z_U`, both equal to `∂f/∂x` at the bound.
    ///
    /// At the exact optimum every slack is zero, so bound complementarity is
    /// zero whichever sign convention the writer used — the check is on
    /// magnitudes. Off the optimum it is `|z| · slack`.
    #[test]
    fn bound_complementarity_is_z_times_slack_over_variables() {
        let x_l = [NLP_LOWER_BOUND_INF, -1.0];
        let x_u = [1.0, NLP_UPPER_BOUND_INF];
        // Exactly on both bounds: no slack anywhere.
        assert_eq!(
            bound_complementarity(&[1.0, -1.0], &x_l, &x_u, &[0.0, 2.0], &[-4.0, 0.0]),
            0.0
        );
        // Pull x off its upper bound by 1e-3 while keeping z_U: the product
        // is the residual, and the sign of the multiplier does not enter.
        let c = bound_complementarity(&[0.999, -1.0], &x_l, &x_u, &[0.0, 2.0], &[-4.0, 0.0]);
        assert!((c - 4.0e-3).abs() < 1e-12, "got {c}");
        let flipped = bound_complementarity(&[0.999, -1.0], &x_l, &x_u, &[0.0, 2.0], &[4.0, 0.0]);
        assert_eq!(c, flipped, "magnitudes only — no sign convention assumed");
        // A variable with no bound on the side in question contributes
        // nothing, however large its (meaningless) multiplier.
        assert_eq!(
            bound_complementarity(
                &[0.0],
                &[NLP_LOWER_BOUND_INF],
                &[NLP_UPPER_BOUND_INF],
                &[1e6],
                &[1e6]
            ),
            0.0
        );
    }

    /// The exact residual uses the multipliers the `.sol` actually carries,
    /// so it sees what the bound-projected one projects away — the gh #495
    /// blind spot: a bound multiplier that is missing or wrong leaves the
    /// projected residual at `0.0`.
    #[test]
    fn exact_dual_infeasibility_sees_what_the_projection_hides() {
        // `min (x−3)² s.t. x ≤ 1`: x* = 1, ∇f = −4, so z_U = 4 and the
        // exported suffix is `ipopt_zU_out = −4`.
        let s = [-4.0];
        let x = [1.0];
        let x_l = [NLP_LOWER_BOUND_INF];
        let x_u = [1.0];
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[-4.0]), 0.0);

        // Projection: x sits on its upper bound, so a valid z_U absorbs the
        // whole negative gradient and the residual reads zero — with *no*
        // multiplier supplied at all.
        assert_eq!(bound_projected_residual(&s, &x, &x_l, &x_u), 0.0);
        // The exact check does not get to assume one exists.
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[0.0]), 4.0);
        // Nor that it has the right sign.
        assert_eq!(exact_dual_infeasibility(&s, &[0.0], &[4.0]), 8.0);
    }

    /// **gh #516.** Constraint complementarity (rows) and bound
    /// complementarity (variables) are different quantities at the same
    /// point, and can disagree by orders of magnitude. Printing either under
    /// a bare `complementarity residual` label invites the comparison that
    /// cost two people an afternoon in #505; this test pins the fact that
    /// makes the label matter.
    #[test]
    fn row_and_bound_complementarity_are_different_quantities() {
        // One inequality row `g ≥ 0`, slack 4.5e-2, multiplier 1 — a real
        // row-complementarity residual.
        let rows = row_complementarity(&[1.0], &[4.5e-2], &[0.0], &[NLP_UPPER_BOUND_INF]);
        assert!((rows - 4.5e-2).abs() < 1e-15);
        // The same point's variables sit hard on their bounds: bound
        // complementarity is eleven orders of magnitude smaller.
        let bounds = bound_complementarity(
            &[1.0],
            &[NLP_LOWER_BOUND_INF],
            &[1.0 + 1e-11],
            &[0.0],
            &[-1.0],
        );
        assert!(bounds < 1e-10, "got {bounds}");
        assert!(
            rows / bounds > 1e8,
            "the two must not be read as one number"
        );
    }
}
