//! **Doubleton-equality aggregation** for the convex presolve (gh #494).
//!
//! [`crate::presolve`]'s catalog could fix a variable from a *singleton*
//! equality row and substitute out a *free column singleton*, but had no
//! reduction for the row that carries a flowsheet: `a₁·x + a₂·y = b`
//! linking two otherwise-free interior variables, with no anchoring
//! requirement on either. That is the arc equality, the `Reference` alias,
//! the unit-conversion link. This module adds it.
//!
//! # Sharing the planner, not the wrapper
//!
//! The *which variables can go* half is not restated here.
//! [`pounce_presolve::linear_eq_plan`] already decides it for the general
//! NLP path (Phase 6, gh #487), and its [`PlanInput`] is pure data —
//! per-row `(col, coeff)` lists, right-hand sides, an eligibility mask and
//! a box. No TNLP appears in it. So the convex path feeds its `QpProblem`
//! equality rows straight in and inherits, for free:
//!
//! - the fixed-point iteration over the three shapes (equal bounds,
//!   singleton row, two-variable row), so alias *chains* collapse;
//! - the union-by-size pivot tie-break that keeps those chains shallow;
//! - the fail-closed posture — a contradictory system, or one that removes
//!   every column, returns the identity plan and this pass does nothing.
//!
//! What is written here is the convex half: applying `x = M·y + d` to
//! `P`, `c`, `A`, `G`, and the postsolve in `(y, z)` conventions.
//!
//! # Convexity survives
//!
//! The substitution is a congruence, `P' = Mᵀ P M`, so `P ⪰ 0 ⇒ P' ⪰ 0`.
//! The reduced problem is convex whenever the original was, and the LP/QP
//! classification made upstream stays true of what the solver receives.
//! (Pinned by `presolve_aggregation.rs`, not assumed.)
//!
//! # The dual, and where a bound multiplier lands
//!
//! Row multipliers come from [`recover_dropped_multipliers`] — the same
//! reverse triangular sweep the NLP path uses, in the same `∇f + Jᵀλ − z_l
//! + z_u = 0` convention `crate::presolve` already works in. Only the row
//! block differs: the plan consumes *equality* rows, so `Gᵀz` is folded
//! into the gradient the sweep is handed rather than resolved by it.
//!
//! Bound multipliers take a step of their own. Planning transfers an
//! eliminated variable's box onto its survivor, so a reduced solve reports
//! the *survivor* carrying a bound force that, in the original problem,
//! belongs to the eliminated variable — and may name a bound the survivor
//! does not even have. [`crate::presolve`]'s contract is a valid KKT point
//! of the *original* problem, and a multiplier on a bound the original does
//! not declare is not one. So the leftover reduced cost at each survivor is
//! re-attributed to whichever cluster member is actually sitting on its own
//! declared bound, and the sweep is re-run with that force in place. See
//! [`postsolve`].
//!
//! The NLP path reaches the same answer by a different route (gh #493,
//! fixed by #503): it records during *planning* where each reduced bound
//! came from and rescales by `α` at postsolve. Reading it back off the
//! leftover instead is what suits this side, where the same columns also
//! carry inequality rows and a bound-tightening layer whose own
//! re-attribution has to compose with this one. Where the survivor's own
//! bound is active too, both paths leave the multiplier there — the split
//! is non-unique and either is a valid KKT point.
//!
//! That step is for **library callers**, and it is worth saying which,
//! because the answer is not the obvious one. The CLI's `.nl` extractor
//! lowers every variable bound to a row of `G`, so a model arriving that
//! way reaches this pass with an *empty box* — nothing to transfer, and
//! each bound's force lands on an ordinary inequality multiplier the sweep
//! already accounts for. A box gets here only from a caller who builds a
//! `QpProblem` with `lb`/`ub` directly (`pounce-py`, the batch API, an
//! embedder), or from [`crate::presolve`]'s own bound tightening. The
//! adversary run for gh #494 measured this rather than guessing at it: 0
//! of 114 `.nl` instances reached the re-attribution, against 19 of 400
//! library-level draws in `tests/presolve_aggregation.rs`.

use std::collections::HashMap;

use pounce_common::types::{Index, lower_bound_present, upper_bound_present};
use pounce_presolve::linear_eq_elim::recover_dropped_multipliers;
use pounce_presolve::linear_eq_plan::{
    EliminationPlan, PlanConfig, PlanInput, VarRecovery, build_plan,
};

use crate::presolve::{EMPTY_ROW_TOL, at_bound, group_by_row, merge_sort_coeffs};
use crate::qp::{BOUND_INF, QpProblem, QpSolution, Triplet};

/// A survivor's leftover reduced cost below this is noise, not a bound
/// force worth re-attributing.
const LEFTOVER_TOL: f64 = 1e-12;

/// Plan the aggregation for `prob`, or `None` when there is nothing to do.
///
/// Only the equality block is offered to the planner: an inequality row
/// determines nothing, and handing one over as a candidate would let it be
/// consumed as though it did.
pub(crate) fn plan(prob: &QpProblem) -> Option<EliminationPlan> {
    let n = prob.n;
    let m_eq = prob.m_eq();
    if n == 0 || m_eq == 0 {
        return None;
    }

    // The two crates spell "absent bound" differently: `pounce-convex`
    // calls a bound absent past ±1e20, the planner past ±1e19. A declared
    // bound landing in that band is *present* here and *absent* there, so
    // the planner would drop it from the box it transfers and hand back a
    // reduced problem with a larger feasible set than the original. Rather
    // than lose a constraint, decline the pass outright — the band is far
    // outside anything a real model carries, so this costs nothing real.
    let mut x_l = Vec::with_capacity(n);
    let mut x_u = Vec::with_capacity(n);
    for j in 0..n {
        let lo = prob.lb_of(j);
        let hi = prob.ub_of(j);
        if (lo > -BOUND_INF) != lower_bound_present(lo) {
            return None;
        }
        if (hi < BOUND_INF) != upper_bound_present(hi) {
            return None;
        }
        x_l.push(if lo > -BOUND_INF {
            lo
        } else {
            f64::NEG_INFINITY
        });
        x_u.push(if hi < BOUND_INF { hi } else { f64::INFINITY });
    }

    let rows = group_by_row(&prob.a, m_eq);
    // `QpProblem` equalities read `Σ a_j x_j = b` exactly; there is no
    // separate row constant to fold (the `.nl` path's gh #492 hazard does
    // not arise on this side, where extraction already folded it into `b`).
    let row_const = vec![0.0; m_eq];
    let eligible = vec![true; m_eq];
    let input = PlanInput {
        n_vars: n,
        n_rows: m_eq,
        rows: &rows,
        row_const: &row_const,
        g_l: &prob.b,
        g_u: &prob.b,
        eligible: &eligible,
        x_l: &x_l,
        x_u: &x_u,
    };
    let plan = build_plan(&input, &PlanConfig::default());
    // `build_plan` returns the identity both when it found nothing and when
    // it stood down (a contradiction, or every column removed). Either way
    // there is no reduction to apply, and the model goes on untouched.
    if plan.is_identity() {
        return None;
    }
    Some(plan)
}

/// How one original column reads in the reduced space:
/// `x_j = mult · y[rep] + shift`, with `rep` a *reduced* column index.
struct ColMap {
    /// `None` for a column the plan pinned to a constant.
    rep: Option<usize>,
    mult: f64,
    shift: f64,
}

fn col_maps(plan: &EliminationPlan) -> Option<Vec<ColMap>> {
    let mut reduced_of = vec![usize::MAX; plan.n_full];
    for (red, &full) in plan.vars_kept.iter().enumerate() {
        reduced_of[full] = red;
    }
    let mut out = Vec::with_capacity(plan.n_full);
    for rec in &plan.recovery {
        out.push(match *rec {
            VarRecovery::Kept(red) => ColMap {
                rep: Some(red),
                mult: 1.0,
                shift: 0.0,
            },
            VarRecovery::Constant(c) => ColMap {
                rep: None,
                mult: 0.0,
                shift: c,
            },
            VarRecovery::Affine { rep, coeff, offset } => {
                // The planner guarantees `rep` is a survivor; refuse the
                // whole reduction rather than trust it silently.
                let red = *reduced_of.get(rep)?;
                if red == usize::MAX {
                    return None;
                }
                ColMap {
                    rep: Some(red),
                    mult: coeff,
                    shift: offset,
                }
            }
        });
    }
    Some(out)
}

/// Apply the plan's affine map to `P`, `c`, `A`, `G` and the bounds,
/// returning the reduced problem and the constant it moves into the
/// objective.
pub(crate) fn reduce(prob: &QpProblem, plan: &EliminationPlan) -> Option<(QpProblem, f64)> {
    let maps = col_maps(plan)?;
    let k = plan.n_reduced_vars();
    let mut new_c = vec![0.0; k];
    let mut offset = 0.0;

    // --- objective: 0.5 xᵀPx + cᵀx under x = M y + d ---
    // `p_lower` holds the lower triangle, so a diagonal entry `v` means the
    // term `0.5·v·x_i²` and an off-diagonal entry means `v·x_i·x_j` (both
    // halves at once). Both shapes are expanded below in those terms, so
    // the emitted triangle carries the same convention back.
    let mut pacc: HashMap<(usize, usize), f64> = HashMap::new();
    {
        let mut add_p = |i: usize, j: usize, v: f64| {
            let key = if i >= j { (i, j) } else { (j, i) };
            *pacc.entry(key).or_insert(0.0) += v;
        };
        for t in &prob.p_lower {
            if t.val == 0.0 {
                continue;
            }
            let v = t.val;
            let (a, b) = (&maps[t.row], &maps[t.col]);
            if t.row == t.col {
                // 0.5·v·(m·y + s)² = 0.5·(v m²)·y² + (v m s)·y + 0.5·v s².
                if let Some(p) = a.rep {
                    if a.mult != 0.0 {
                        add_p(p, p, v * a.mult * a.mult);
                    }
                    new_c[p] += v * a.mult * a.shift;
                }
                offset += 0.5 * v * a.shift * a.shift;
            } else {
                // v·(mᵢy_p + sᵢ)(mⱼy_q + s_j).
                if let (Some(p), Some(q)) = (a.rep, b.rep) {
                    let coef = v * a.mult * b.mult;
                    if coef != 0.0 {
                        if p == q {
                            // Both columns folded onto the *same* survivor,
                            // so an off-diagonal term became a diagonal one:
                            // `coef·y²` is `0.5·(2·coef)·y²` in the stored
                            // convention.
                            add_p(p, p, 2.0 * coef);
                        } else {
                            add_p(p, q, coef);
                        }
                    }
                }
                if let Some(p) = a.rep {
                    new_c[p] += v * a.mult * b.shift;
                }
                if let Some(q) = b.rep {
                    new_c[q] += v * b.mult * a.shift;
                }
                offset += v * a.shift * b.shift;
            }
        }
    }
    for j in 0..prob.n {
        let cj = prob.c[j];
        if cj == 0.0 {
            continue;
        }
        let m = &maps[j];
        if let Some(p) = m.rep {
            new_c[p] += cj * m.mult;
        }
        offset += cj * m.shift;
    }
    let mut new_p: Vec<Triplet> = pacc
        .into_iter()
        .filter(|&(_, v)| v != 0.0)
        .map(|((i, j), v)| Triplet::new(i, j, v))
        .collect();
    // `HashMap` iteration order is unspecified; sort so the reduced problem
    // is a deterministic function of its input (the IPM's pivot order, and
    // therefore its iterate trace, would otherwise vary run to run).
    new_p.sort_by_key(|t| (t.row, t.col));

    // --- rows ---
    // A row the substitution empties is left as `0 (cmp) rhs` for the next
    // round's empty-row screen, which compares against exact zero. Its `rhs`
    // is a computed difference of terms up to `scale`, so on a row that is
    // redundant in exact arithmetic it carries rounding of that size — a
    // `0 ≤ -1.3e-15` the next round would report as a proof of
    // infeasibility. Snap a residual within `EMPTY_ROW_TOL × (1 + scale)` to
    // zero (the same test `build_rows` applies to a row emptied by fixing,
    // gh#496); a real conflict is far larger and keeps its value, so the
    // screen still reports it.
    let substitute = |entries: &[(usize, f64)], rhs0: f64| -> (Vec<(usize, f64)>, f64) {
        let mut coeffs: Vec<(usize, f64)> = Vec::with_capacity(entries.len());
        let mut rhs = rhs0;
        let mut scale = rhs0.abs();
        for &(col, a) in entries {
            let m = &maps[col];
            if let Some(p) = m.rep {
                if m.mult != 0.0 {
                    coeffs.push((p, a * m.mult));
                }
            }
            let term = a * m.shift;
            rhs -= term;
            scale = scale.max(term.abs());
        }
        merge_sort_coeffs(&mut coeffs);
        if coeffs.is_empty() && rhs.abs() <= EMPTY_ROW_TOL * (1.0 + scale) {
            rhs = 0.0;
        }
        (coeffs, rhs)
    };

    let a_by_row = group_by_row(&prob.a, prob.m_eq());
    let mut new_a = Vec::new();
    let mut new_b = Vec::with_capacity(plan.rows_kept.len());
    for (newr, &r) in plan.rows_kept.iter().enumerate() {
        let (coeffs, rhs) = substitute(&a_by_row[r], prob.b[r]);
        new_b.push(rhs);
        for (c, v) in coeffs {
            new_a.push(Triplet::new(newr, c, v));
        }
    }
    // Every inequality row survives: the plan is offered the equalities
    // only, so it consumes none of them and their order is unchanged.
    let g_by_row = group_by_row(&prob.g, prob.m_ineq());
    let mut new_g = Vec::new();
    let mut new_h = Vec::with_capacity(prob.m_ineq());
    for (r, entries) in g_by_row.iter().enumerate() {
        let (coeffs, rhs) = substitute(entries, prob.h[r]);
        new_h.push(rhs);
        for (c, v) in coeffs {
            new_g.push(Triplet::new(r, c, v));
        }
    }

    // --- box: the planner's reduced box already carries every bound
    // transferred off an eliminated column ---
    let need_bounds = plan.x_l_red.iter().any(|&v| v > -BOUND_INF && !v.is_nan())
        || plan.x_u_red.iter().any(|&v| v < BOUND_INF && !v.is_nan());
    let (new_lb, new_ub) = if need_bounds {
        (plan.x_l_red.clone(), plan.x_u_red.clone())
    } else {
        (Vec::new(), Vec::new())
    };

    Some((
        QpProblem {
            n: k,
            p_lower: new_p,
            c: new_c,
            a: new_a,
            b: new_b,
            g: new_g,
            h: new_h,
            lb: new_lb,
            ub: new_ub,
        },
        offset,
    ))
}

/// Jacobian of the equality block in the triplet form the shared sweep
/// wants.
struct EqTriplets {
    irow: Vec<Index>,
    jcol: Vec<Index>,
    vals: Vec<f64>,
}

impl EqTriplets {
    fn of(prob: &QpProblem) -> Self {
        Self {
            irow: prob.a.iter().map(|t| t.row as Index).collect(),
            jcol: prob.a.iter().map(|t| t.col as Index).collect(),
            vals: prob.a.iter().map(|t| t.val).collect(),
        }
    }
}

/// Is `x_j` sitting on a declared bound of `orig`? Returns `(at_lb, at_ub)`.
fn on_bounds(orig: &QpProblem, x: &[f64], j: usize) -> (bool, bool) {
    let lb = orig.lb_of(j);
    let ub = orig.ub_of(j);
    (
        lb > -BOUND_INF && at_bound(x[j], lb),
        ub < BOUND_INF && at_bound(x[j], ub),
    )
}

/// Expand a reduced solution back to `orig`'s space.
///
/// Primal is the plan's own lift. Duals are recovered in three moves:
///
/// 1. **A first sweep** with no bound forces at all. It sets every consumed
///    row's multiplier so that stationarity holds exactly at each
///    *eliminated* column, which leaves each survivor holding its cluster's
///    entire reduced cost.
/// 2. **Attribution.** A survivor's leftover is a real bound force. If the
///    survivor is itself on one of its own declared bounds, it keeps it —
///    the same answer a solve without this pass would give. Otherwise the
///    bound was *transferred* during planning, and the force belongs to
///    whichever eliminated cluster member is on its own declared bound:
///    with `x_j = α·x_rep + β`, a multiplier `μ` there enters the
///    survivor's equation as `α·μ`, so `μ = leftover/α` (signs pick which
///    of the two bounds it is).
/// 3. **A second sweep**, with those multipliers in the gradient, so the
///    row multipliers are consistent with where the force ended up. Then
///    each survivor's own bound multiplier is read off the final reduced
///    cost by complementarity, exactly as `Presolve::postsolve_once` does.
///
/// Step 2 is what the NLP path (gh #493) does not do, and does not need to:
/// there the survivor's bound multiplier absorbs the transferred force and
/// full-space stationarity still holds. Here the contract is a valid KKT
/// point of the original problem, and a nonzero multiplier on a bound the
/// original never declared would not be one.
pub(crate) fn postsolve(orig: &QpProblem, plan: &EliminationPlan, red: &QpSolution) -> QpSolution {
    let n = orig.n;
    let m_eq = orig.m_eq();
    let m_ineq = orig.m_ineq();

    let mut x = vec![0.0; n];
    plan.lift_x(&red.x, &mut x);

    // Inequality rows pass through one-for-one.
    let mut z = vec![0.0; m_ineq];
    for (i, slot) in z.iter_mut().enumerate() {
        *slot = red.z.get(i).copied().unwrap_or(0.0);
    }
    let mut y_kept = vec![0.0; m_eq];
    for (newr, &oldr) in plan.rows_kept.iter().enumerate() {
        y_kept[oldr] = red.y.get(newr).copied().unwrap_or(0.0);
    }

    // Everything acting on a column that the sweep does *not* resolve:
    // the objective gradient and the inequality block. `Aᵀy` is added
    // inside the sweep, from `y` (kept rows filled, consumed rows zero).
    let mut base = orig.c.clone();
    orig.p_mul(&x, &mut base);
    orig.gt_mul(&z, &mut base);

    let jac = EqTriplets::of(orig);
    let mut y = y_kept.clone();
    let none: [f64; 0] = [];
    recover_dropped_multipliers(
        plan, &base, &none, &none, &jac.irow, &jac.jcol, &jac.vals, false, &mut y,
    );

    let mut grad = base.clone();
    orig.at_mul(&y, &mut grad);

    // --- attribute each survivor's leftover ---
    let mut z_lb = vec![0.0; n];
    let mut z_ub = vec![0.0; n];
    let mut moved = false;
    for &rep in &plan.vars_kept {
        let leftover = grad[rep];
        if leftover.abs() <= LEFTOVER_TOL {
            continue;
        }
        let (rep_at_lb, rep_at_ub) = on_bounds(orig, &x, rep);
        if (rep_at_lb && leftover > 0.0) || (rep_at_ub && leftover < 0.0) {
            // The survivor's own declared bound carries it; leave it to the
            // complementarity pass below.
            continue;
        }
        for (j, rec) in plan.recovery.iter().enumerate() {
            let VarRecovery::Affine { rep: r, coeff, .. } = *rec else {
                continue;
            };
            if r != rep || coeff == 0.0 {
                continue;
            }
            let mu = leftover / coeff;
            let (at_lb, at_ub) = on_bounds(orig, &x, j);
            if at_lb && mu > 0.0 {
                z_lb[j] = mu;
            } else if at_ub && mu < 0.0 {
                z_ub[j] = -mu;
            } else {
                continue;
            }
            moved = true;
            break;
        }
    }

    if moved {
        y = y_kept;
        recover_dropped_multipliers(
            plan, &base, &z_lb, &z_ub, &jac.irow, &jac.jcol, &jac.vals, false, &mut y,
        );
        grad = base.clone();
        orig.at_mul(&y, &mut grad);
    }

    // Survivors read their own bound multipliers off the final reduced cost
    // by complementarity against their *declared* box. Eliminated columns
    // keep what step 2 gave them (zero, unless they were the owner) — the
    // sweep made their stationarity hold with exactly that.
    for &rep in &plan.vars_kept {
        let (at_lb, at_ub) = on_bounds(orig, &x, rep);
        if at_lb && grad[rep] > 0.0 {
            z_lb[rep] = grad[rep];
        } else if at_ub && grad[rep] < 0.0 {
            z_ub[rep] = -grad[rep];
        }
    }

    let mut px = vec![0.0; n];
    orig.p_mul(&x, &mut px);
    let mut obj = 0.0;
    for i in 0..n {
        obj += 0.5 * x[i] * px[i] + orig.c[i] * x[i];
    }

    QpSolution {
        status: red.status,
        x,
        y,
        z,
        z_lb,
        z_ub,
        obj,
        iters: red.iters,
        iterates: red.iterates.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `0.5 xᵀPx + cᵀx`, from the stored lower triangle.
    fn objective(prob: &QpProblem, x: &[f64]) -> f64 {
        let mut px = vec![0.0; prob.n];
        prob.p_mul(x, &mut px);
        (0..prob.n)
            .map(|i| 0.5 * x[i] * px[i] + prob.c[i] * x[i])
            .sum()
    }

    fn row_activities(triplets: &[Triplet], m: usize, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; m];
        for t in triplets {
            out[t.row] += t.val * x[t.col];
        }
        out
    }

    /// The reduction must be an *identity on the objective and the rows*:
    /// pushing any reduced point through the lift and evaluating in the
    /// original space has to agree, term for term, with evaluating the
    /// reduced problem directly. This is the check that bites when the
    /// Hessian congruence, the shifted linear term or the moved constant
    /// is wrong — none of which a solve-and-compare notices, because a
    /// wrong-but-self-consistent reduced problem still converges.
    fn assert_faithful(prob: &QpProblem, plan: &EliminationPlan, reduced: &QpProblem, off: f64) {
        // A handful of arbitrary reduced points, including a non-symmetric
        // one so a dropped cross term cannot cancel.
        for seed in 0..5 {
            let y: Vec<f64> = (0..reduced.n)
                .map(|i| 0.5 + 0.75 * ((i + seed) as f64) - 0.3 * (seed as f64))
                .collect();
            let mut x = vec![0.0; prob.n];
            plan.lift_x(&y, &mut x);

            let full = objective(prob, &x);
            let red = objective(reduced, &y) + off;
            assert!(
                (full - red).abs() <= 1e-9 * (1.0 + full.abs()),
                "objective seed {seed}: full {full} vs reduced {red}"
            );

            let ax = row_activities(&prob.a, prob.m_eq(), &x);
            let ay = row_activities(&reduced.a, reduced.m_eq(), &y);
            for (newr, &oldr) in plan.rows_kept.iter().enumerate() {
                let full_slack = ax[oldr] - prob.b[oldr];
                let red_slack = ay[newr] - reduced.b[newr];
                assert!(
                    (full_slack - red_slack).abs() <= 1e-9 * (1.0 + full_slack.abs()),
                    "eq row {oldr}→{newr} seed {seed}: {full_slack} vs {red_slack}"
                );
            }
            let gx = row_activities(&prob.g, prob.m_ineq(), &x);
            let gy = row_activities(&reduced.g, reduced.m_ineq(), &y);
            for r in 0..prob.m_ineq() {
                let full_slack = gx[r] - prob.h[r];
                let red_slack = gy[r] - reduced.h[r];
                assert!(
                    (full_slack - red_slack).abs() <= 1e-9 * (1.0 + full_slack.abs()),
                    "ineq row {r} seed {seed}: {full_slack} vs {red_slack}"
                );
            }
            // The lift must also *satisfy* every row the plan consumed —
            // that is the whole claim those rows were dropped on.
            for step in &plan.steps {
                let resid = ax[step.row] - prob.b[step.row];
                assert!(
                    resid.abs() <= 1e-9 * (1.0 + prob.b[step.row].abs()),
                    "consumed row {} seed {seed} residual {resid}",
                    step.row
                );
            }
        }
    }

    fn planned(prob: &QpProblem) -> (EliminationPlan, QpProblem, f64) {
        let plan = plan(prob).expect("a plan");
        let (reduced, off) = reduce(prob, &plan).expect("a reduction");
        assert_faithful(prob, &plan, &reduced, off);
        (plan, reduced, off)
    }

    /// A plain alias `x0 − x1 = 0` collapses two columns to one.
    #[test]
    fn alias_pair_collapses() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0)],
            c: vec![1.0, 3.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let (_, reduced, _) = planned(&prob);
        assert_eq!(reduced.n, 1);
        assert_eq!(reduced.m_eq(), 0);
        // Both costs land on the survivor.
        assert!((reduced.c[0] - 4.0).abs() < 1e-12, "{:?}", reduced.c);
    }

    /// Both signs of the aggregation coefficient, and a non-zero offset:
    /// `x0 = α·x1 + β` for α of either sign must reproduce the objective.
    #[test]
    fn both_signs_and_an_offset() {
        for a1 in [2.0_f64, -2.0] {
            let prob = QpProblem {
                n: 2,
                p_lower: vec![Triplet::new(0, 0, 4.0), Triplet::new(1, 1, 1.0)],
                c: vec![-1.0, 2.0],
                a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, a1)],
                b: vec![3.0],
                g: vec![],
                h: vec![],
                lb: vec![],
                ub: vec![],
            };
            let (_, reduced, _) = planned(&prob);
            assert_eq!(reduced.n, 1, "a1 = {a1}");
        }
    }

    /// A row the plan *keeps* whose columns include an eliminated one must
    /// have that column's affine offset moved into its right-hand side.
    ///
    /// This shape is the one a derivative check is blind to: the reduced
    /// row's coefficients are right either way, since `∂(αy + β)/∂y = α`
    /// regardless of `β`. Only re-evaluating the row at a lifted point sees
    /// the missing constant — which is exactly what `assert_faithful` does.
    #[test]
    fn a_surviving_row_carries_the_substitution_offset() {
        let prob = QpProblem {
            n: 4,
            p_lower: (0..4).map(|i| Triplet::new(i, i, 2.0)).collect(),
            c: vec![1.0, -1.0, 0.5, 0.25],
            a: vec![
                // x0 − 2·x1 = 3  ⇒  one of the pair becomes affine in the
                // other with a non-zero offset.
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, -2.0),
                // Three distinct clusters, so this row survives and has to
                // absorb that offset into its right-hand side.
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 2, 1.0),
                Triplet::new(1, 3, 1.0),
            ],
            b: vec![3.0, 5.0],
            g: vec![
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, -1.0),
                Triplet::new(0, 3, 2.0),
            ],
            h: vec![4.0],
            lb: vec![],
            ub: vec![],
        };
        let (plan, reduced, _) = planned(&prob);
        assert_eq!(reduced.n, 3);
        assert_eq!(plan.rows_kept.len(), 1);
        assert_eq!(
            reduced.m_ineq(),
            1,
            "the inequality is rewritten, not dropped"
        );
    }

    /// Two columns folded onto the *same* survivor turn an off-diagonal
    /// Hessian entry into a diagonal one. The stored triangle counts an
    /// off-diagonal twice and a diagonal once, so this needs the ×2 that
    /// `reduce` applies — without it the reduced objective is off by a
    /// factor of two in exactly this shape.
    #[test]
    fn two_columns_onto_one_survivor_fold_the_cross_term() {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 2.0),
                Triplet::new(1, 0, 3.0), // cross term x0·x1
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![1.0, 1.0, 1.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 2, -1.0), // x0 = x2
                Triplet::new(1, 1, 1.0),
                Triplet::new(1, 2, -2.0), // x1 = 2·x2
            ],
            b: vec![0.0, 0.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let (_, reduced, _) = planned(&prob);
        assert_eq!(reduced.n, 1);
    }

    /// A contradictory alias system stands the whole pass down rather than
    /// declaring the model infeasible from inside an elimination pass.
    #[test]
    fn contradiction_abandons_the_plan() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 1.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, -1.0), // x0 − x1 = 0
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, -1.0), // x0 − x1 = 1
            ],
            b: vec![0.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        assert!(plan(&prob).is_none());
    }

    /// A model whose every column would go is handed back untouched: the
    /// planner refuses to produce a problem with no degrees of freedom.
    #[test]
    fn removing_every_column_abandons_the_plan() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![1.0],
            a: vec![Triplet::new(0, 0, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        assert!(plan(&prob).is_none());
    }

    /// A bound in the band where the two crates disagree about "absent"
    /// declines the pass rather than silently widening the feasible set.
    #[test]
    fn ambiguous_bound_sentinel_declines() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 1.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            // Present to `pounce-convex` (> −1e20), absent to the planner
            // (≤ −1e19).
            lb: vec![-5e19, f64::NEG_INFINITY],
            ub: vec![f64::INFINITY, f64::INFINITY],
        };
        assert!(plan(&prob).is_none());
    }

    /// An eliminated column's box is transferred onto its survivor, and
    /// the transfer follows the sign of α.
    #[test]
    fn the_box_moves_with_the_column() {
        // x0 = −x1, x0 ∈ [1, 4] ⇒ x1 ∈ [−4, −1].
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(1, 1, 2.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![1.0, f64::NEG_INFINITY],
            ub: vec![4.0, f64::INFINITY],
        };
        let (plan, reduced, _) = planned(&prob);
        assert_eq!(reduced.n, 1);
        let survivor = plan.vars_kept[0];
        let (lo, hi) = if survivor == 1 {
            (-4.0, -1.0)
        } else {
            (1.0, 4.0)
        };
        assert!((reduced.lb_of(0) - lo).abs() < 1e-12, "{:?}", reduced.lb);
        assert!((reduced.ub_of(0) - hi).abs() < 1e-12, "{:?}", reduced.ub);
    }
}
