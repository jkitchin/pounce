//! Newton iterations on the barrier system, against the held factor.
//!
//! A parametric step is the first-order prediction of where the
//! solution moves. It leaves a residual in the barrier KKT system at
//! the perturbed parameter values, and that residual can be driven
//! down by Newton iterations that reuse the converged factorization,
//! so each one costs a back-solve rather than a factorization.
//!
//! The operator is the one the solve left behind, evaluated at the
//! base point. That is what makes an iteration cheap and also what
//! bounds what the corrector can do: where the perturbation moves a
//! bound out of the active set, the held barrier diagonal carries that
//! bound's stiffness from the base point, `z / s` at a bound the solve
//! held tightly is `z² / μ`, and no number of iterations against that
//! operator will let the bound go. The iteration then makes no
//! progress at all, which the residual reports on the first step.
//!
//! So this is not a method that converges to a re-solve. It improves
//! the step where the active set the base point settled still fits,
//! and it says how far it got. The residual is the honest measure: the
//! distance to the true solution is not knowable without solving, and
//! the achievable accuracy varies by problem, from the held-μ offset
//! of about `1e-7` on some models to `1e-9` on others.
//!
//! The residual comes from the algorithm's own calculated quantities
//! by way of the trial iterate, so scaling, fixed variables, and the
//! bound expansions are handled exactly as the solve handles them.
//! Setting a trial point leaves `curr` alone, and `curr` is what the
//! held factorization was built from, so nothing here disturbs the
//! factor or any other consumer of the session.

use std::rc::Rc;

use pounce_common::types::Number;

use crate::backsolver::SensBacksolver;
use crate::solver::SolverError;

/// How far a step may travel toward a bound, as a fraction of the
/// distance remaining. The barrier's own rule, at the value the
/// algorithm uses once `μ` is small.
const TAU: Number = 0.9995;

/// What one call to the corrector did.
///
/// `residual` is the primal-dual barrier residual at the returned
/// point, and `initial_residual` the same quantity at the step the
/// caller handed in, so the ratio says what the iterations bought.
/// When they are equal the corrector made no progress, which happens
/// when the perturbation needs a bound the base point held to leave.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectorReport {
    /// Back-solves spent. One per iteration.
    pub iterations: usize,
    /// Residual at the returned point.
    pub residual: Number,
    /// Residual at the point the iterations start from, which is the
    /// step handed in after the active-set decision has been applied
    /// and any coordinate outside a bound has been put back inside. It
    /// is not the residual of the caller's own step, and the two differ
    /// by exactly what that decision and that clamp changed.
    pub initial_residual: Number,
    /// True when the loop stopped because an iteration failed to
    /// improve on the best residual seen, rather than because it ran
    /// out of budget.
    ///
    /// An iteration whose residual is not finite -- the Newton
    /// direction stepped a variable out of the domain of one of the
    /// model's functions -- fails to improve by construction, since a
    /// non-finite residual norms to infinity (gh#845). So it ends the
    /// loop here, and the point handed back is the best *finite* one
    /// seen. It is not reported as a converged correction: `residual`
    /// then equals `initial_residual` and [`Self::improved`] is false.
    pub converged: bool,
    /// Bounds the step took out of the active set, which the corrector
    /// removed from the operator once before iterating.
    pub released: usize,
    /// Bounds the step brought into the active set, whose barrier
    /// diagonal the corrector raised once before iterating.
    pub pinned: usize,
    /// The returned point's residual split the way the compound system
    /// is: how far the Lagrangian's gradient is from zero, how far the
    /// model's own equations are from being satisfied, and how far the
    /// bound multipliers are from complementarity at the held `mu`.
    ///
    /// The three carry different units and the reported `residual` is
    /// the largest of them, so a caller deciding whether an estimate is
    /// usable wants them apart. A correction can leave the equations
    /// nearly satisfied while the multipliers are far off, and only the
    /// first of those affects whether the values can be acted on.
    pub stationarity: Number,
    /// Largest violation of the model's equality and inequality rows.
    pub feasibility: Number,
    /// Largest departure from `z · s = mu` over the bound multipliers.
    pub complementarity: Number,
}

impl CorrectorReport {
    /// Whether the iterations improved on the step handed in.
    ///
    /// False means the returned point is the caller's own step, which
    /// is what happens when the operator cannot represent the
    /// active-set change the perturbation needs.
    pub fn improved(&self) -> bool {
        self.residual < self.initial_residual
    }
}

/// The barrier residual's blocks, in the compound layout's order.
///
/// Assembled from the trial iterate's calculated quantities, with the
/// parametric shift applied to the equality rows the caller pinned:
/// the corrector is aiming at the perturbed problem, whose equality
/// constraints sit at the shifted right-hand side.
/// Sup-norm of a residual block, where a non-finite entry norms to
/// infinity rather than being skipped.
///
/// `f64::max` returns the *other* operand when one of the two is NaN,
/// so a plain fold over it *swallows* NaN: an all-NaN vector norms to
/// `0.0`, the smallest number the stopping rule can see. That is
/// gh#845 -- the NaN iterate was accepted as the best point yet and
/// reported with `residual = 0.0`, all three split residuals `0.0`,
/// `converged = true` and `improved() = true`, while the step handed
/// back was all NaN. A residual that is not a number is not a *small*
/// residual; it is the absence of one, so it norms to the largest
/// value instead of the smallest and can never win the comparison
/// against the best seen.
fn residual_norm(v: &[Number]) -> Number {
    let mut worst = 0.0_f64;
    for &b in v {
        if !b.is_finite() {
            return Number::INFINITY;
        }
        worst = worst.max(b.abs());
    }
    worst
}

pub(crate) fn residual_at(
    bs: &crate::algorithm_backsolver::PdSensBacksolver,
    flat: &[Number],
    pin_rows: &[usize],
    deltas: &[Number],
    mu: Number,
    out: &mut [Number],
) -> Result<(), SolverError> {
    // `flat` is in natural units, the frame the step and the bounds are
    // in. The algorithm's own iterate is scaled, so undo `F` before
    // handing it back. `solve` post-multiplies by the same vector.
    let scaled: Vec<Number> = match bs.natural_units_factor() {
        None => flat.to_vec(),
        Some(f) => flat
            .iter()
            .zip(f)
            .map(|(&v, &s)| if s == 0.0 { v } else { v / s })
            .collect(),
    };
    let iv = bs
        .pack_public(&scaled)
        .map_err(|_| SolverError::SensComputationFailed("corrector: pack failed".into()))?;
    let (data, cq, _) = bs.activity_handles();
    data.borrow_mut().set_trial(iv.freeze());

    let off = bs.offsets_public();
    let cqb = cq.borrow();
    let blocks: [(usize, std::rc::Rc<dyn pounce_linalg::Vector>, Number); 8] = [
        (0, cqb.trial_grad_lag_x(), 0.0),
        (1, cqb.trial_grad_lag_s(), 0.0),
        (2, cqb.trial_c(), 0.0),
        (3, cqb.trial_d_minus_s(), 0.0),
        (4, cqb.trial_compl_x_l(), mu),
        (5, cqb.trial_compl_x_u(), mu),
        (6, cqb.trial_compl_s_l(), mu),
        (7, cqb.trial_compl_s_u(), mu),
    ];
    for (i, v, shift) in blocks {
        let vals = crate::vec_util::dense_to_vec(&*v);
        let (a, b) = (off[i], off[i + 1]);
        if vals.len() != b - a {
            return Err(SolverError::SensComputationFailed(format!(
                "corrector: block {i} is {} long, expected {}",
                vals.len(),
                b - a
            )));
        }
        for (o, val) in out[a..b].iter_mut().zip(vals) {
            *o = val - shift;
        }
    }
    drop(cqb);

    // The perturbation moves the pinned equalities' right-hand sides,
    // so the residual there is measured against the moved value.
    //
    // `pin_rows` are flat KKT rows, already through
    // `pin_rows_and_c_scales`. A user `g` index is NOT that row: the
    // two differ whenever an inequality precedes the pin in `g(x)`
    // (pounce#128). `deltas` arrive multiplied by the same call's row
    // scales, since the residual above is in the algorithm's scaled
    // equality block.
    let (yc_a, yc_b) = (off[2], off[3]);
    for (&r, &d) in pin_rows.iter().zip(deltas) {
        if r < yc_a || r >= yc_b {
            return Err(SolverError::SensComputationFailed(format!(
                "corrector: pin row {r} is outside the equality block"
            )));
        }
        out[r] -= d;
    }
    Ok(())
}

/// The largest step from `val` along `dir` that keeps every entry of a
/// positive quantity at or above `1 - TAU` of where it started.
/// `skip` names entries to leave out, by position in `val`. A released
/// multiplier is held at zero for the whole correction, and zero is not
/// a positive quantity: `-TAU * 0 / d` is exactly zero, so leaving one
/// in with a negative direction sets the step length to zero and
/// freezes every other entry with it.
fn fraction_to_boundary(val: &[Number], dir: &[Number], skip: &[usize]) -> Number {
    let mut a = 1.0;
    for (k, (&v, &d)) in val.iter().zip(dir).enumerate() {
        if d < 0.0 && !skip.contains(&k) {
            let lim = -TAU * v / d;
            if lim < a {
                a = lim;
            }
        }
    }
    a.max(0.0)
}

/// The slacks a primal point sits at, per bound row, and the direction
/// those slacks move under a primal direction.
///
/// The bound rows carry the variable each one constrains and its side,
/// which is what turns a primal vector into the per-bound quantity the
/// fraction rule needs.
fn slacks_and_directions(
    rows: &[crate::backsolver::BoundRow],
    x: &[Number],
    dx: &[Number],
    lo: &[Number],
    hi: &[Number],
    lower: bool,
) -> (Vec<Number>, Vec<Number>) {
    let mut s = Vec::new();
    let mut ds = Vec::new();
    for b in rows.iter().filter(|b| b.lower == lower) {
        let i = b.var_row;
        if lower {
            s.push(x[i] - lo[i]);
            ds.push(dx[i]);
        } else {
            s.push(hi[i] - x[i]);
            ds.push(-dx[i]);
        }
    }
    (s, ds)
}

/// The bounds the step takes out of the active set.
///
/// A bound the solve held tightly contributes `z / s` to the barrier
/// diagonal, which at a small `mu` is a very large number, and the
/// held factorization carries it. If the perturbation moves that
/// variable off its bound, iterating against that operator cannot
/// follow, because the stiffness the base point had is still there.
/// The way out is to take the bound out of the operator once, before
/// iterating, which is what `solve_released` does.
///
/// Which bounds those are is the predictor's answer, read off the step
/// it produced rather than decided here. A bound is released when the
/// solve held it, meaning its barrier diagonal is above one, and the
/// step carries the variable off it by more than roundoff. Reading the
/// primal endpoint works for every mode: `fix_relax` and `path` have
/// already applied their releases by the time the step is handed over,
/// and the plain step shows the same crossing in the coordinate it
/// carries past the bound.
fn released_rows(
    rows: &[crate::backsolver::BoundRow],
    base: &[Number],
    end: &[Number],
    lo: &[Number],
    hi: &[Number],
) -> Vec<usize> {
    let mut out = Vec::new();
    for b in rows {
        let i = b.var_row;
        let (s_base, s_end) = if b.lower {
            (base[i] - lo[i], end[i] - lo[i])
        } else {
            (hi[i] - base[i], hi[i] - end[i])
        };
        let z_base = base[b.row];
        if s_base <= 0.0 || z_base / s_base <= 1.0 {
            continue; // the solve did not hold this bound
        }
        // Off the bound by more than the slack it sat at, and by more
        // than roundoff against the variable's own size.
        if s_end > 10.0 * s_base && s_end > 1e-9 * (1.0 + base[i].abs()) {
            out.push(b.row);
        }
    }
    out
}

/// The bounds the step brings into the active set, with the barrier
/// stiffness each one needs.
///
/// A variable the solve left interior contributes almost nothing to
/// the barrier diagonal, `mu / s²` at a slack of order one, so the
/// held factorization treats it as free. If the step carries it onto a
/// bound, iterating against that operator pushes it straight back out
/// and the fraction-to-boundary rule refuses the step, which is the
/// same failure a released bound causes, in the other direction.
///
/// The stiffness a bound at slack `s` carries is `mu / s²`, which is
/// what the barrier itself would assign there, so that is what goes on
/// the diagonal. The variable sits at the margin the start put it at.
fn pinned_rows(
    rows: &[crate::backsolver::BoundRow],
    base: &[Number],
    end: &[Number],
    lo: &[Number],
    hi: &[Number],
    mu: Number,
) -> Vec<(usize, Number)> {
    let mut out = Vec::new();
    for b in rows {
        let i = b.var_row;
        let (s_base, s_end) = if b.lower {
            (base[i] - lo[i], end[i] - lo[i])
        } else {
            (hi[i] - base[i], hi[i] - end[i])
        };
        let z_base = base[b.row];
        if s_base <= 0.0 || z_base / s_base > 1.0 {
            continue; // the solve already held this bound
        }
        // On the bound now, and it was not before.
        if s_end > 0.0 && s_end < 0.1 * s_base && s_end < 1e-6 * (1.0 + base[i].abs()) {
            out.push((i, mu / (s_end * s_end)));
        }
    }
    out
}

/// Hold every bound multiplier within a band of what its slack
/// implies, the way the algorithm does after each accepted step.
///
/// `kappa_sigma_clamp` in the solver puts each multiplier into
/// `[mu / (kappa · s), kappa · mu / s]`, so one can never drift
/// arbitrarily far from `mu / s`. The corrector needs it in two
/// places the solver does not: the multipliers arrive from the
/// predictor, extrapolated across the whole perturbation and never
/// passed through this, and each iteration produces new ones.
fn clamp_multipliers(
    rows: &[crate::backsolver::BoundRow],
    iterate: &mut [Number],
    lo: &[Number],
    hi: &[Number],
    mu: Number,
) {
    const KAPPA_SIGMA: Number = 1e10;
    for b in rows {
        let i = b.var_row;
        let s = if b.lower {
            iterate[i] - lo[i]
        } else {
            hi[i] - iterate[i]
        };
        if s <= 0.0 || !s.is_finite() {
            continue;
        }
        let z = &mut iterate[b.row];
        if *z <= 0.0 {
            continue; // released, and held there deliberately
        }
        *z = z.clamp(mu / (KAPPA_SIGMA * s), KAPPA_SIGMA * mu / s);
    }
}

/// Run the corrector.
///
/// `start` is the caller's compound step, `base` the converged
/// iterate, both in the compound layout. Returns the corrected step
/// and what the iterations did.
///
/// The loop keeps the iterate with the smallest residual seen and
/// stops when an iteration fails to improve on it. That one comparison
/// covers all three ways the iteration ends: reaching the accuracy the
/// held operator supports, making no progress at all because a bound
/// must leave, and settling into a cycle where the fraction rule
/// alternates between two points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    bs: &crate::algorithm_backsolver::PdSensBacksolver,
    base_scaled: &[Number],
    start: &[Number],
    pin_rows: &[usize],
    deltas: &[Number],
    lo: &[Number],
    hi: &[Number],
    mu: Number,
    max_iter: usize,
    exact_hessian: bool,
) -> Result<(Vec<Number>, CorrectorReport), SolverError> {
    let dim = bs.dim();
    let off = bs.offsets_public();
    let n_x = bs.block_dims()[0];
    let rows = bs
        .bound_rows()
        .ok_or_else(|| SolverError::SensComputationFailed("corrector: no bound rows".into()))?
        .to_vec();

    // The barrier needs every bounded coordinate strictly inside its
    // bound and every multiplier strictly positive. A step that
    // carries one past is put back just inside, since the residual is
    // undefined outside and the fraction rule cannot recover from a
    // point that is already out.
    // Everything below works in natural units: the step, the bounds
    // from `bound_context`, and what `solve` returns are all in that
    // frame, while the algorithm's own iterate is scaled. Converting
    // the base once here is what keeps them addable. `residual_at`
    // converts back before reading the calculated quantities.
    let base: Vec<Number> = match bs.natural_units_factor() {
        None => base_scaled.to_vec(),
        Some(f) => base_scaled.iter().zip(f).map(|(&v, &s)| v * s).collect(),
    };
    let base = &base[..];
    let mut iterate: Vec<Number> = base.iter().zip(start).map(|(&b, &s)| b + s).collect();
    for b in &rows {
        let i = b.var_row;
        let margin = 1e-10 * (1.0 + base[i].abs());
        if b.lower {
            iterate[i] = iterate[i].max(lo[i] + margin);
        } else {
            iterate[i] = iterate[i].min(hi[i] - margin);
        }
    }
    for z in iterate[off[4]..off[8]].iter_mut() {
        *z = z.max(1e-12);
    }

    clamp_multipliers(&rows, &mut iterate, lo, hi, mu);

    // The bounds the step takes out of the active set, decided once
    // here and held for every iteration. Their barrier terms come out
    // of the operator, their multipliers are held at zero, and their
    // complementarity rows leave the system, which is what an inactive
    // bound means. Everything else keeps the base point's barrier term.
    let released = released_rows(&rows, base, &iterate, lo, hi);
    let pinned = pinned_rows(&rows, base, &iterate, lo, hi, mu);
    for &r in &released {
        iterate[r] = 0.0;
    }
    // The operator is assembled at the PREDICTED point: the current
    // iterate is swapped to it for the duration of the correction, so
    // every block evaluates there. The Hessian and the constraint
    // Jacobians at the stepped point with the step's own multipliers
    // as clamped above, the bound rows with the stepped slacks and
    // multipliers, and the barrier diagonal from the same frame
    // (built below, after the swap), so the one factorization the
    // correction pays is of one consistent operator. A chord
    // iteration contracts at the rate the distance between its
    // operator and the true Jacobian sets, and the predicted point is
    // where the truth is. The clamp above bounds every multiplier
    // into the kappa-sigma band at the stepped slacks, and the box
    // put-back leaves a positive margin on every coordinate, so the
    // rebuilt diagonal is large exactly where the step holds a
    // coordinate on a bound and finite everywhere.
    let (data_h, cq_h, _) = bs.activity_handles();
    let predicted = bs.pack_natural(&iterate).ok_or_else(|| {
        SolverError::SensComputationFailed("corrector: packing the predicted point failed".into())
    })?;
    struct RestoreCurr<'a> {
        data: &'a pounce_algorithm::ipopt_data::IpoptDataHandle,
        saved: Option<pounce_algorithm::iterates_vector::IteratesVector>,
        saved_w: Option<Rc<dyn pounce_linalg::SymMatrix>>,
        restore_w: bool,
    }
    impl Drop for RestoreCurr<'_> {
        fn drop(&mut self) {
            if let Some(c) = self.saved.take() {
                self.data.borrow_mut().set_curr(c);
            }
            if self.restore_w {
                self.data.borrow_mut().w = self.saved_w.take();
            }
        }
    }
    let mut restore = RestoreCurr {
        data: data_h,
        saved: data_h.borrow().curr.clone(),
        saved_w: None,
        restore_w: false,
    };
    data_h.borrow_mut().set_curr(predicted);
    // The solve below takes its Hessian from `IpoptData::w`, the
    // matrix the algorithm last stored, so moving the iterate alone
    // moves only the Jacobians and slacks. Under an exact-Hessian
    // solve the matrix is re-evaluated at the predicted point, with
    // the step's own multipliers as clamped above. A limited-memory
    // solve keeps its quasi-Newton matrix: there is no exact Hessian
    // to evaluate anywhere.
    if exact_hessian {
        restore.saved_w = data_h.borrow().w.clone();
        restore.restore_w = true;
        let w = cq_h.borrow().curr_exact_hessian();
        data_h.borrow_mut().w = Some(w);
    }
    let _restore = restore;

    // Built AFTER the swap, so both diagonal blocks read the
    // predicted point: the stored base-point pair, which the ceiling
    // (gh#737) or crossover (gh#654) would otherwise freeze, is never
    // consulted, and the frame rule and the ceiling are re-derived at
    // the predicted point instead. See `corrector_sigma`.
    let op = bs.corrector_sigma(&pinned).ok_or_else(|| {
        SolverError::SensComputationFailed("corrector: operator diagonal unavailable".into())
    })?;

    let solve = |rhs: &[Number], lhs: &mut [Number]| -> bool {
        // The same Rc every call: the factorization cache keys on its
        // tag, so the predicted point's operator is factored once and
        // every later solve is a back-solve.
        // The ratios go with the diagonals: the chord operator
        // eliminates each bound row into a diagonal the gh#737 ceiling
        // may have softened, so the row is read back through the same
        // cap (gh#828). A no-op wherever the ceiling did not bind,
        // which is every corrector fixture that predates it.
        bs.solve_released_prebuilt(
            &released,
            Rc::clone(&op.sigma_x),
            Some(Rc::clone(&op.sigma_s)),
            Some((&op.ratio_x, &op.ratio_s)),
            rhs,
            lhs,
            false,
        )
    };
    // A released bound has no equation left, so its row does not count
    // toward the residual the stopping rule reads.
    let clear = |v: &mut [Number]| {
        for &r in &released {
            v[r] = 0.0;
        }
    };

    let mut resid = vec![0.0; dim];
    residual_at(bs, &iterate, pin_rows, deltas, mu, &mut resid)?;
    clear(&mut resid);
    let norm = residual_norm;
    let initial_residual = norm(&resid);
    if !initial_residual.is_finite() {
        // The starting point itself is outside the domain of the
        // model's own functions, so there is no residual to reduce and
        // nothing to hand back. Say so rather than iterate on NaN.
        return Err(SolverError::SensComputationFailed(
            "corrector: the barrier residual is not finite at the point the \
             iterations start from, so the predicted point is outside the \
             domain of the model's functions (an unbounded variable driven \
             into a log, a sqrt or a reciprocal, say). Bound the variable or \
             take a smaller perturbation."
                .into(),
        ));
    }

    let mut best = iterate.clone();
    let mut best_residual = initial_residual;
    let mut iterations = 0usize;
    let mut converged = false;

    // The released rows as positions inside the dual slice the fraction
    // rule reads, since they are held at zero and must not bound it.
    let released_dual: Vec<usize> = released.iter().map(|&r| r - off[4]).collect();

    let mut rhs = vec![0.0; dim];
    let mut dir = vec![0.0; dim];
    while iterations < max_iter {
        // `resid` came off the calculated quantities, so it is in the
        // algorithm's scaled frame, and `solve` pre-multiplies its
        // right-hand side by `E` because it expects natural units.
        // `r_scaled = E r_nat`, so undo it here; passing the scaled
        // residual straight in applies `E` twice and leaves a
        // diagonally mis-scaled direction that under-corrects
        // stationarity wherever a variable factor is not one.
        match bs.scaled_rhs_factor() {
            None => {
                for (r, s) in rhs.iter_mut().zip(&resid) {
                    *r = -s;
                }
            }
            Some(e) => {
                for ((r, s), &ev) in rhs.iter_mut().zip(&resid).zip(e) {
                    *r = if ev == 0.0 { -s } else { -s / ev };
                }
            }
        }
        if !solve(&rhs, &mut dir) {
            return Err(SolverError::BacksolveFailed);
        }
        iterations += 1;

        let (sl, dsl) = slacks_and_directions(&rows, &iterate[..n_x], &dir[..n_x], lo, hi, true);
        let (su, dsu) = slacks_and_directions(&rows, &iterate[..n_x], &dir[..n_x], lo, hi, false);
        let alpha_p =
            fraction_to_boundary(&sl, &dsl, &[]).min(fraction_to_boundary(&su, &dsu, &[]));
        let alpha_d = fraction_to_boundary(
            &iterate[off[4]..off[8]],
            &dir[off[4]..off[8]],
            &released_dual,
        );

        for i in 0..off[4] {
            iterate[i] += alpha_p * dir[i];
        }
        for i in off[4]..off[8] {
            iterate[i] = (iterate[i] + alpha_d * dir[i]).max(1e-14);
        }
        // A released multiplier stays at zero: the bound is out of the
        // active set for the whole correction, not something the
        // iterations decide again each step.
        for &r in &released {
            iterate[r] = 0.0;
        }
        clamp_multipliers(&rows, &mut iterate, lo, hi, mu);

        residual_at(bs, &iterate, pin_rows, deltas, mu, &mut resid)?;
        clear(&mut resid);
        let now = norm(&resid);
        if now < best_residual {
            best_residual = now;
            best.copy_from_slice(&iterate);
        } else {
            // An iteration that did not improve on the best residual
            // seen ends the loop, and the best point is what the
            // caller gets.
            converged = true;
            break;
        }
    }

    // the returned point's residual, split by what each block means
    residual_at(bs, &best, pin_rows, deltas, mu, &mut resid)?;
    clear(&mut resid);
    let part = |a: usize, b: usize| norm(&resid[off[a]..off[b]]);
    let (stationarity, feasibility, complementarity) = (part(0, 2), part(2, 4), part(4, 8));

    let step: Vec<Number> = best.iter().zip(base).map(|(&v, &b)| v - b).collect();
    // Stated as a promise rather than inferred from the residual: the
    // point handed back is finite. With `residual_norm` in place a NaN
    // iterate can no longer win the `now < best_residual` comparison,
    // so `best` is the caller's own clamped step at worst -- but the
    // whole of gh#845 was a NaN reaching a caller wearing a report that
    // said it had not, and a screen here is one pass over `dim`.
    if !step.iter().all(|v| v.is_finite()) {
        return Err(SolverError::SensComputationFailed(
            "corrector: the corrected step is not finite. The predicted \
             point is outside the domain of the model's functions (an \
             unbounded variable driven into a log, a sqrt or a reciprocal, \
             say). Bound the variable or take a smaller perturbation."
                .into(),
        ));
    }
    Ok((
        step,
        CorrectorReport {
            iterations,
            residual: best_residual,
            initial_residual,
            converged,
            released: released.len(),
            pinned: pinned.len(),
            stationarity,
            feasibility,
            complementarity,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::BoundRow;

    const MU: Number = 1e-8;

    /// gh#845 at the arithmetic. `residual_at` produces the residual
    /// this norms, and a fixture that goes non-finite only *inside* the
    /// loop -- rather than at the point the iterations start from, which
    /// `tests/issue_845_nonfinite_residual.rs` reaches -- is hard to
    /// build on purpose, since the models that leave a domain leave it
    /// at the predicted point. The branch is one line either way, so it
    /// is pinned here instead of through a model.
    #[test]
    fn a_non_finite_entry_norms_to_infinity_rather_than_being_swallowed() {
        // The historical defect: `fold(0.0, f64::max)` returns the other
        // operand at a NaN, so this vector normed to 0.0 and beat every
        // finite residual the loop had seen.
        assert_eq!(residual_norm(&[Number::NAN; 4]), Number::INFINITY);
        assert_eq!(
            residual_norm(&[1e-3, Number::NAN, 2e-3]),
            Number::INFINITY,
            "one NaN among finite entries is still no residual"
        );
        assert_eq!(residual_norm(&[Number::INFINITY, 0.0]), Number::INFINITY);
        assert_eq!(residual_norm(&[Number::NEG_INFINITY]), Number::INFINITY);
        // and the ordinary case is unchanged.
        assert_eq!(residual_norm(&[]), 0.0);
        assert_eq!(residual_norm(&[-3.0, 1.0, 2.0]), 3.0);
        // The consequence the stopping rule cares about: infinity never
        // wins `now < best_residual`, where 0.0 always did.
        let beats_the_best_seen = residual_norm(&[Number::NAN]) < 1e-30;
        assert!(!beats_the_best_seen);
    }

    /// Two variables, one bound row each: a lower bound on `x0` and an
    /// upper bound on `x1`. The compound layout puts the multipliers
    /// after the primal block, which is all `clamp_multipliers` needs.
    fn rows() -> Vec<BoundRow> {
        vec![
            BoundRow {
                row: 2,
                var_row: 0,
                lower: true,
            },
            BoundRow {
                row: 3,
                var_row: 1,
                lower: false,
            },
        ]
    }

    /// `[x0, x1, z0, z1]` with both variables a unit inside their
    /// bounds, so each band is `[mu / kappa, kappa * mu]`.
    fn iterate(z0: Number, z1: Number) -> Vec<Number> {
        vec![1.0, 9.0, z0, z1]
    }

    const LO: [Number; 2] = [0.0, 0.0];
    const HI: [Number; 2] = [10.0, 10.0];

    #[test]
    fn a_multiplier_inside_the_band_is_left_alone() {
        let mut it = iterate(1.0, 2.0);
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        assert_eq!((it[2], it[3]), (1.0, 2.0));
    }

    #[test]
    fn a_multiplier_above_the_band_comes_down_to_it() {
        // the slack is 1.0 on both, so the ceiling is kappa * mu
        let ceiling = 1e10 * MU;
        let mut it = iterate(1e6, 1e6);
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        assert_eq!(
            (it[2], it[3]),
            (ceiling, ceiling),
            "both should land on the ceiling {ceiling}",
        );
    }

    #[test]
    fn a_multiplier_below_the_band_comes_up_to_it() {
        let floor = MU / 1e10;
        let mut it = iterate(1e-30, 1e-30);
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        assert_eq!((it[2], it[3]), (floor, floor));
    }

    #[test]
    fn the_band_moves_with_the_slack() {
        // x0 a hundredth off its lower bound, x1 a hundredth off its
        // upper. Both bands are a hundred times higher than at a slack
        // of one, and the upper bound reads its slack from the other
        // side.
        let mut it = vec![0.01, 9.99, 1e6, 1e6];
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        let ceiling = 1e10 * MU / 0.01;
        assert!(
            (it[2] - ceiling).abs() < 1e-9 * ceiling && (it[3] - ceiling).abs() < 1e-9 * ceiling,
            "want {ceiling} on both, got {} and {}",
            it[2],
            it[3],
        );
    }

    #[test]
    fn a_released_multiplier_stays_at_zero() {
        // A released bound is held at zero for the whole correction,
        // so the clamp must not lift it back into the band.
        let mut it = iterate(0.0, -1.0);
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        assert_eq!((it[2], it[3]), (0.0, -1.0));
    }

    #[test]
    fn a_coordinate_outside_its_bound_is_skipped() {
        // No positive slack means no band to clamp into, and dividing
        // by it would produce a sign the barrier cannot use.
        let mut it = vec![-1.0, 11.0, 1e6, 1e6];
        clamp_multipliers(&rows(), &mut it, &LO, &HI, MU);
        assert_eq!((it[2], it[3]), (1e6, 1e6));
    }

    #[test]
    fn the_fraction_rule_stops_short_of_zero() {
        // A direction that would drive an entry to zero is cut to TAU
        // of the way, and one that only increases is taken whole.
        assert_eq!(fraction_to_boundary(&[1.0], &[-1.0], &[]), TAU);
        assert_eq!(fraction_to_boundary(&[1.0], &[1.0], &[]), 1.0);
        assert_eq!(fraction_to_boundary(&[2.0, 1.0], &[-1.0, -1.0], &[]), TAU);
        // the tightest entry decides
        assert_eq!(fraction_to_boundary(&[1.0], &[-4.0], &[]), TAU / 4.0);
        // a zero entry would set the step to zero, and a skipped one
        // must not: this is what freezes every multiplier once a bound
        // is released (gh#733 review)
        assert_eq!(fraction_to_boundary(&[0.0, 1.0], &[-1.0, -1.0], &[]), 0.0);
        assert_eq!(fraction_to_boundary(&[0.0, 1.0], &[-1.0, -1.0], &[0]), TAU);
    }
}
