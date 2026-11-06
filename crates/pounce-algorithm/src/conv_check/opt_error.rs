//! Optimal-error convergence check — port of
//! `Algorithm/IpOptErrorConvCheck.{hpp,cpp}`.
//!
//! Tolerance state machine over `(nlp_err, iter_count)` plus
//! per-component infeasibilities pulled directly from
//! [`IpoptCalculatedQuantities`]. The scalar
//! [`Self::check_convergence`] entry point only gates on
//! `nlp_err <= tol` (matching upstream when the per-component
//! tolerances are at their `+∞` sentinels); the state-aware
//! [`Self::check_convergence_with_state`] adds the
//! `dual_inf_tol` / `constr_viol_tol` / `compl_inf_tol` gates that
//! mirror upstream `OptimalityErrorConvergenceCheck::CheckConvergence`.

use crate::conv_check::r#trait::{ConvCheck, ConvergenceStatus};
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use pounce_common::types::{Index, Number};

#[derive(Clone)]
pub struct OptErrorConvCheck {
    pub tol: Number,
    pub dual_inf_tol: Number,
    pub constr_viol_tol: Number,
    pub compl_inf_tol: Number,
    pub acceptable_tol: Number,
    pub acceptable_dual_inf_tol: Number,
    pub acceptable_constr_viol_tol: Number,
    pub acceptable_compl_inf_tol: Number,
    pub acceptable_obj_change_tol: Number,
    pub acceptable_iter: Index,
    pub max_iter: Index,
    pub max_cpu_time: Number,
    pub max_wall_time: Number,
    pub acceptable_count: Index,
    /// Objective value at the last iterate the main loop stashed via
    /// `set_curr_acceptable_obj`. Used by the
    /// `acceptable_obj_change_tol` cross-check. `None` until an
    /// acceptable point has been recorded.
    pub last_acceptable_obj: Option<Number>,
    /// Tolerance on the scaled infeasibility stationarity
    /// `‖Jᵀc‖/max(1,‖c‖)`. An iterate counts toward the infeasibility
    /// streak when this ratio is at or below this value while the
    /// constraint violation stays bounded away from zero. Rapid
    /// infeasibility detection is disabled when this is non-positive.
    pub infeas_stationarity_tol: Number,
    /// Multiple of `constr_viol_tol` the constraint violation must
    /// exceed before an iterate can count as infeasible-stationary —
    /// keeps detection from firing on nearly-feasible flat spots. Floored
    /// at [`MIN_INFEAS_VIOL_FLOOR`]; see
    /// [`OptErrorConvCheck::absolute_viol_threshold`].
    pub infeas_viol_kappa: Number,
    /// Consecutive infeasible-stationary iterations required before
    /// terminating with `LocallyInfeasible`. Non-positive disables
    /// rapid infeasibility detection.
    pub infeas_max_streak: Index,
    /// Running count of consecutive infeasible-stationary iterations.
    pub infeas_streak: Index,
    /// Objective-scale floor below which a strict certificate is refused
    /// while the *unscaled* KKT error is still above `acceptable_tol`
    /// (gh #200). See [`certificate_masked`]. `0` disables the mechanism
    /// entirely, restoring bit-for-bit upstream-Ipopt behaviour.
    pub obj_scale_certificate_threshold: Number,
    /// Safety factor on the per-row noise floor the **strict** gate judges the
    /// primal term against (gh #528). `0` disables the floor entirely,
    /// restoring upstream Ipopt's bare-absolute primal residual.
    pub primal_noise_floor_kappa: Number,
    /// Fraction of `acceptable_tol` the KKT error — and, relative to the
    /// objective's own size, the objective — may drift across the
    /// acceptable-level streak's window while the streak still counts as
    /// *settled* (gh #533). See [`Self::streak_has_flattened`]. `0` disables
    /// the progress test, leaving acceptable-level termination the bare
    /// consecutive-count criterion upstream Ipopt uses.
    pub acceptable_progress_kappa: Number,
    /// Trailing `(nlp_err, f)` samples of the current acceptable-level streak,
    /// oldest first, at most [`Self::progress_window_len`] entries. Cleared
    /// whenever the streak breaks — the window describes *this* streak.
    pub acceptable_window: std::collections::VecDeque<(Number, Number)>,
    /// Acceptable-level terminations the gh #533 progress test has refused so
    /// far this solve. Bounded by [`ACCEPTABLE_PROGRESS_MAX_REFUSALS`], past
    /// which the test stands aside and the streak terminates as it would
    /// without it.
    pub acceptable_progress_refusals: Index,
    /// Safety factor on the scale-relative floor the **strict** gate judges
    /// `dual_inf` against (gh #532); see [`Self::dual_inf_bound`]. `0` disables
    /// the floor, restoring upstream Ipopt's bare-absolute `dual_inf_tol`.
    pub dual_inf_scale_kappa: Number,
    /// Whether the gh #532 scale-relative dual floor has already been reported
    /// this solve. Diagnostic only — the certificate below carries a dual
    /// infeasibility above `dual_inf_tol`, which is worth saying once and not
    /// once per iteration.
    pub dual_floor_reported: bool,
    /// Whether a masked **strict** certificate was ever refused this solve.
    pub veto_fired: bool,
    /// Whether a masked **acceptable-level** termination was ever refused.
    ///
    /// Tracked separately because the two refusals must be undone differently:
    /// a refused strict certificate restores as `Success`, a refused
    /// acceptable-level one as `StopAtAcceptablePoint`. Conflating them would
    /// either over-claim a status or, as originally written, leave the
    /// acceptable-level refusal with no safety net at all.
    ///
    /// Set by **both** refusal arms — the gh #200 masked-scale veto and the
    /// gh #533 progress test — because both need the same undo. What the
    /// masked veto's own iteration budget counts is
    /// [`Self::masked_acceptable_veto_fired`].
    pub acceptable_veto_fired: bool,
    /// Whether the *masked-scale* (gh #200) arm specifically refused an
    /// acceptable-level termination.
    ///
    /// [`VETO_MAX_EXTRA_ITERS`] is the masked veto's budget, so only the masked
    /// arms may spend it. Counting the gh #533 progress refusals against it too
    /// would silently disarm the masked veto 60 iterations into any solve whose
    /// acceptable streak was progress-refused — a different mechanism's bug
    /// coming back for reasons having nothing to do with objective scaling.
    pub masked_acceptable_veto_fired: bool,
    /// Iterations spent since the veto first refused a certificate.
    ///
    /// The veto is a bet that continuing reaches a better point. Some problems
    /// never let it pay off — an unscaled error pinned above `acceptable_tol`
    /// by an unbounded direction keeps the veto engaged until `max_iter`,
    /// turning a 40-iteration solve into a 300-iteration one for nothing. Past
    /// [`VETO_MAX_EXTRA_ITERS`] the bet is called off and the run is allowed to
    /// terminate normally; correctness does not depend on the cap, because the
    /// refused certificate is restored either way.
    pub veto_extra_iters: Index,
    /// Iterations on which the scale-relative feasibility veto blocked a
    /// certificate (strict or acceptable) that the absolute tolerances had
    /// passed. Bounded by [`VETO_MAX_EXTRA_ITERS`]; past the budget the veto
    /// disengages and the run terminates as it would have without it, so the
    /// worst case is a bounded number of extra iterations, never a lost
    /// verdict. See [`Self::relative_viol_threshold`].
    ///
    /// Read by the *certificate* arm only. The acceptable-point stash's gate
    /// in [`ConvCheck::current_is_acceptable_with_state`] is deliberately
    /// unbudgeted (gh #693) — declining to stash spends no iterations, so a
    /// budget there bounds nothing and only ever expired, at which point the
    /// point the veto exists to reject became the rollback target.
    pub rel_infeas_extra_iters: Index,
    /// Relative primal infeasibility at the previous
    /// [`Self::note_infeasible_stationary`] call — the progress signal for the
    /// relative arm's streak (see that method). `NAN` until first set, which
    /// compares as "not improving" and lets the first iterate count.
    pub prev_rel_viol: Number,
}

/// How many iterations the veto may spend before its bet is called off.
///
/// Generous relative to what a successful rescue costs — the reported quartics
/// reach the true minimum in 11-15 extra iterations — but bounded, so a veto
/// that can never lift (an unscaled error pinned above `acceptable_tol` by an
/// unbounded direction) cannot run to `max_iter`. Correctness does not rest on
/// this number: whatever happens after the budget is spent, the refused
/// certificate is still restored if the run ends without a better one.
const VETO_MAX_EXTRA_ITERS: Index = 60;

/// How many acceptable-level terminations the gh #533 progress test may refuse
/// before it stands aside for the rest of the solve.
///
/// The test is already self-limiting — it only refuses while the streak's own
/// window shows the solve still moving, and a solve that stops moving flattens
/// the window within `acceptable_iter` iterations — so this bounds only the
/// pathological case: a solve that wanders inside the acceptable band without
/// ever settling and without ever reaching `tol`. Left unbounded that solve
/// would run to `max_iter` (returning the refused point, so no *verdict* is
/// lost, but spending up to 3000 iterations to say what it could have said at
/// 40).
///
/// The number has to clear the widest measured rescue: `kissing` needed 447
/// iterations past the refusal (103 → 550) to reach its strict certificate, so
/// anything below that cannot fix the reported case. `1000` clears it with room
/// to spare and still stops well short of the default `max_iter = 3000`. Note
/// that only iterations on which a termination is actually *refused* are
/// counted, not every iteration after the first refusal — a streak broken by an
/// iterate outside the band costs nothing here.
const ACCEPTABLE_PROGRESS_MAX_REFUSALS: Index = 1000;

/// Longest trailing streak window the progress test will keep samples for.
///
/// The window is `acceptable_iter` long (the streak's own length), which is 15
/// by default. The cap exists because `acceptable_iter` is a user option with no
/// upper bound, and the window is a live allocation. Past the cap the test
/// judges flatness over the trailing `ACCEPTABLE_PROGRESS_WINDOW_MAX` iterates
/// of the streak instead of all of it — a strictly more permissive reading (a
/// shorter window can only contain less movement), so the cap can never make
/// the mechanism fire where the full window would not have.
const ACCEPTABLE_PROGRESS_WINDOW_MAX: usize = 256;

/// Smallest constraint violation rapid infeasibility detection will ever treat
/// as "bounded away from feasible" (gh #519).
///
/// Both arms of [`OptErrorConvCheck::is_infeasible_stationary`] scale their
/// violation floor with `constr_viol_tol`, which is a *feasibility* tolerance:
/// left unclamped, tightening it widens the set of points the detector is
/// willing to convict, so asking for a stricter feasibility standard makes the
/// solver more eager to answer "locally infeasible". That inversion is the bug
/// this floor exists to prevent — at `constr_viol_tol = 1e-6` the absolute arm's
/// floor fell to `1e-4` and @bernalde's `f=1` model (gh #505), plateaued at an
/// unscaled violation of `1.94e-4` with a scaled NLP error of `4.89e-10`, was
/// reported infeasible at iteration 27 instead of "Solved To Acceptable Level"
/// at 37. The flip tracked `100 · constr_viol_tol` to three significant figures.
///
/// `1e-2` is the default `acceptable_constr_viol_tol`, so the floor also states
/// the intended rule directly: never convict a point of infeasibility while its
/// violation sits inside the band the defaults call acceptable. The two forms
/// coincide out of the box, which is why the defect was invisible there.
///
/// Erring loose is the safe direction — a withheld verdict costs iterations and
/// ends at `MaxIterExceeded` or an acceptable point, while a fabricated one is
/// a wrong answer. `infeas_viol_kappa` still raises the floor above this; the
/// disable switch remains `infeas_stationarity_tol = 0` (or
/// `infeas_max_streak = 0`), not a floor small enough to never bind.
const MIN_INFEAS_VIOL_FLOOR: Number = 1e-2;

/// Is a passing strict certificate *masked* by an extreme objective scale
/// (gh #200)?
///
/// Gradient-based scaling picks `df = nlp_scaling_max_gradient / max‖∇f‖`,
/// floored at `nlp_scaling_min_value = 1e-8`. On a flat quartic the initial
/// gradient is enormous (`quartc`: ~4e12 → `df` pinned at the floor), and the
/// strict test then runs on the *scaled* aggregate. Because a quartic's
/// gradient vanishes cubically toward its minimum while `df` stays fixed at its
/// initial value, the scaled error crosses `tol` roughly 30% of the way in: the
/// solver certifies optimality at `quartc` objective 248.88 when the true
/// minimum is ~0, with an unscaled dual infeasibility of 0.84.
///
/// This predicate deliberately does **not** try to decide whether the stop is
/// genuinely false — it only asks whether the conditions that make a false stop
/// *possible* are present. Distinguishing a masked certificate from an honest
/// one at a small scale cannot be done from the residual magnitude: `meyer3`
/// sits at the same 1e-8 scale floor as `quartc` while being genuinely
/// converged, and the unscaled error is a *dimensional* quantity, so any
/// absolute cutoff separating them would move if the objective were rescaled —
/// precisely the sensitivity this bug is about. An earlier revision of this
/// work did exactly that (a 5e-2 bar fitted to the gap in one benchmark suite);
/// it is not defensible and was removed.
///
/// Instead the caller *tests* the hypothesis: it refuses to stop, continues,
/// and sees whether the iterates actually go anywhere. If they do, the stop was
/// false. If they do not, the certificate is honoured unchanged — so the
/// mechanism is never worse than not having it (see `terminate_vetoed_or`).
pub fn certificate_masked(
    obj_scale: Number,
    unscaled_err: Number,
    threshold: Number,
    acceptable_tol: Number,
) -> bool {
    // A non-positive threshold is the documented opt-out; NaN is treated the
    // same way rather than silently enabling the mechanism.
    if threshold.is_nan() || threshold <= 0.0 {
        return false;
    }
    // Magnitude, not signed value: a negative `obj_scaling_factor` (the
    // documented way to maximize) is trivially below any positive threshold,
    // which would arm this on every maximization regardless of scale.
    obj_scale.abs() < threshold && unscaled_err > acceptable_tol
}

impl Default for OptErrorConvCheck {
    fn default() -> Self {
        // Defaults from `IpOptErrorConvCheck.cpp:RegisterOptions`.
        Self {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1e10,
            acceptable_constr_viol_tol: 1e-2,
            acceptable_compl_inf_tol: 1e-2,
            acceptable_obj_change_tol: 1e20,
            acceptable_iter: 15,
            max_iter: 3000,
            max_cpu_time: 1e6,
            max_wall_time: 1e6,
            acceptable_count: 0,
            last_acceptable_obj: None,
            infeas_stationarity_tol: 1e-8,
            infeas_viol_kappa: 1e2,
            infeas_max_streak: 5,
            infeas_streak: 0,
            // 1e-4 separates the falsely-certified problems (objective scale
            // pinned at the 1e-8 floor) from every recorded collateral case
            // (`hs1`/`hs38` at ~4e-2, the 19-problem list at ~1e-2). See
            // [`certificate_masked`].
            obj_scale_certificate_threshold: 1e-4,
            primal_noise_floor_kappa: 64.0,
            // A tenth of the acceptable band. See `streak_has_flattened` for
            // why the band is the right yardstick and why a tenth of it is the
            // conservative end of the range.
            acceptable_progress_kappa: 1e-1,
            acceptable_window: std::collections::VecDeque::new(),
            acceptable_progress_refusals: 0,
            dual_inf_scale_kappa: 1.0,
            dual_floor_reported: false,
            veto_fired: false,
            acceptable_veto_fired: false,
            masked_acceptable_veto_fired: false,
            veto_extra_iters: 0,
            rel_infeas_extra_iters: 0,
            prev_rel_viol: Number::NAN,
        }
    }
}

impl OptErrorConvCheck {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pure helper for the per-component upstream gate. Returns `true`
    /// iff every supplied residual sits at or below its tolerance.
    /// Factored out so tests can exercise the gating logic without
    /// constructing a full `IpoptCq`.
    ///
    /// `dual_scale` is the magnitude of the terms `∇L` is assembled from
    /// ([`IpoptCalculatedQuantities::curr_unscaled_dual_infeasibility_scale_max`]),
    /// which sets the scale-relative floor under `dual_inf_tol` — see
    /// [`Self::dual_inf_bound`]. Pass `0` for the bare absolute bound.
    fn passes_component_tols(
        &self,
        overall: Number,
        dual_inf: Number,
        constr_viol: Number,
        compl_inf: Number,
        dual_scale: Number,
        primal_resolvable: bool,
    ) -> bool {
        overall <= self.tol
            && dual_inf <= self.dual_inf_bound(dual_scale)
            && self.primal_component_passes(constr_viol, primal_resolvable)
            && compl_inf <= self.compl_inf_tol
    }

    /// Whether the unscaled constraint violation clears the **strict** gate's
    /// primal component (gh #590): `constr_viol <= constr_viol_tol`, or the
    /// iterate placed every constraint row at or below the finest residual that
    /// row can represent in floating point.
    ///
    /// `primal_resolvable` is the caller's answer to
    /// [`IpoptCalculatedQuantities::curr_primal_infeasibility_above_noise`] —
    /// `false` only when that accessor returned exactly `0`, meaning *no* row
    /// rose above its own noise floor. It is a boolean rather than a floored
    /// magnitude on purpose. The floor is computed on the internally scaled
    /// residual while `constr_viol` is unscaled, so the two are not comparable
    /// as numbers, and comparing them anyway is precisely the units error
    /// `infeasible_status_tol_invariance` exists to pin down. The *abstention
    /// verdict* is scale-invariant by construction — the floor carries each
    /// row's `dc_i` exactly as the residual and the declared magnitude do — so
    /// the boolean transfers between unit systems even though the magnitude
    /// does not.
    ///
    /// gh #528 gave the strict **aggregate** this floor and deliberately left
    /// the component on the raw residual, reasoning that any realistic quantum
    /// sits far below the tolerances. gh #590 is the model where that premise
    /// fails. LyoPRONTO's pseudosteady lyophilisation OCP is written in Landau
    /// coordinates, so its conduction rows carry `1/(H − S)²` and reach
    /// magnitudes near `1e8`: one ulp of those rows is `~1e-2`, which is not
    /// two decades under `acceptable_constr_viol_tol`, it *is*
    /// `acceptable_constr_viol_tol`. At the converged point the scaled KKT
    /// error is `4.3e-10` against `tol = 1e-6`, every row's residual is at or
    /// below its own floor, and the unscaled violation reads `1.62e-2` — pure
    /// quantisation. Ipopt 3.14.16 lands on the same point with `8.06e-3` and
    /// calls it `Solved To Acceptable Level`; which side of `1e-2` a run falls
    /// on there is arithmetic luck, not a property of the iterate.
    ///
    /// Leaving the component unfloored while the aggregate is floored is also
    /// incoherent on its own terms: the component test is a refinement of the
    /// aggregate, and an unfloored component can veto a certificate the floored
    /// aggregate has already granted. That is exactly what happened here.
    ///
    /// The relaxation is confined to the all-noise case — one resolvable row
    /// anywhere and the raw comparison stands — and it cannot fabricate a
    /// success on a genuinely infeasible model: such a model's violation is
    /// pinned at its infeasibility gap, orders above `eps ·` the row's own
    /// magnitude, so the accessor returns a positive value and nothing here
    /// engages. The scale-relative veto below is untouched and still sees the
    /// raw `rel_viol`. `primal_noise_floor_kappa = 0` opts out, as it does for
    /// the aggregate.
    fn primal_component_passes(&self, constr_viol: Number, primal_resolvable: bool) -> bool {
        constr_viol <= self.constr_viol_tol || !primal_resolvable
    }

    /// The bound the **strict** gate judges the unscaled dual infeasibility
    /// against: `max(dual_inf_tol, dual_inf_scale_kappa · tol · dual_scale)`
    /// (gh #532).
    ///
    /// `dual_inf_tol` is a bare absolute bound on a quantity the aggregate KKT
    /// error normalises. The aggregate's dual term is `‖∇L‖_∞ / s_d`, and `s_d`
    /// grows with the mean magnitude of the multipliers, so on a model whose
    /// gradients live at `1e10` the two are judging one quantity by two
    /// standards ten orders apart: Vanderbei's `orthrds2` reaches `s_d ≈ 1.6e10`
    /// with `‖∇L‖_∞ = 89.7`, an aggregate dual term of `5.6e-09` — comfortably
    /// inside the default `tol = 1e-8` — and the component gate refused it
    /// against `1.0`, so a solve stationary to nine digits exited
    /// `Solved_To_Acceptable_Level` holding the answer. `1.0` is a reasonable
    /// absolute bound when `‖∇f‖` is `O(1)`; it is meaningless when `‖∇f‖` is
    /// `1e10`, and the same LP with its objective multiplied by a positive
    /// constant — which changes no feasible point, no solution and no active
    /// set — crossed it.
    ///
    /// The floor is stated relative to the terms `∇L` is *made of*
    /// (`dual_scale`), not to `s_d`. Both remove the asymmetry the issue
    /// reports, but `s_d` is built from multiplier magnitudes alone and does not
    /// see `∇f`: a model with tiny constraint gradients and huge multipliers
    /// (`‖J‖ ~ 1e-12`, `‖y‖ ~ 1e12`, so every term of `∇L` is `O(1)`) has
    /// `s_d ~ 1e10` and would have its genuinely non-stationary residual
    /// forgiven — exactly the user-space drift the unscaled component gate was
    /// added for (pounce#173). `dual_scale` cannot be fooled that way, because
    /// `dual_inf / dual_scale` is the fraction of the terms that failed to
    /// cancel.
    ///
    /// So the relaxation only ever forgives a residual that is small *relative
    /// to the problem's own scale*, and it is bounded twice over: the aggregate
    /// `overall <= tol` gate still has to pass on the same iterate, and at the
    /// default `kappa = 1` the floor only rises above `dual_inf_tol` once
    /// `dual_scale` exceeds `dual_inf_tol / tol = 1e8`. A genuinely
    /// non-stationary point has `dual_inf ≈ dual_scale` (nothing cancelled) and
    /// is refused by eight orders of magnitude — `min -exp(x) s.t. x >= 0`
    /// reaching `inf_du = 8.8e+47` with `∇f = −8.8e47` stays refused, which is
    /// the case any such rule has to keep rejecting.
    ///
    /// A user who tightens `dual_inf_tol` below the floor is asking for an
    /// absolute standard the floor may override; `dual_inf_scale_kappa = 0`
    /// switches it off and restores upstream's bare comparison. Non-finite or
    /// non-positive scales are read as "nothing can be said", which is the
    /// absolute bound.
    fn dual_inf_bound(&self, dual_scale: Number) -> Number {
        if self.dual_inf_scale_kappa.is_nan()
            || self.dual_inf_scale_kappa <= 0.0
            || !dual_scale.is_finite()
            || dual_scale <= 0.0
        {
            return self.dual_inf_tol;
        }
        self.dual_inf_tol
            .max(self.dual_inf_scale_kappa * self.tol * dual_scale)
    }

    /// The aggregate KKT error the **strict** gate judges against `tol`
    /// (gh #528): [`IpoptCalculatedQuantities::curr_nlp_error_above_primal_noise`],
    /// which is `nlp_err` with each constraint row's residual counted only
    /// where it rises above what that row's residual can represent in floating
    /// point.
    ///
    /// The primal term of the KKT error is the one term Ipopt leaves as a bare
    /// absolute residual (the other two carry `s_d` / `s_c`), and it is
    /// quantised in units of `eps ·` the rows' own magnitude. Once that quantum
    /// exceeds `tol` — constraint values past `~4.5e7` at the `1e-8` default —
    /// `nlp_err <= tol` stops being a statement about the iterate: it asks the
    /// residual to land on an exact `0` rather than on one ulp, which is
    /// arithmetic luck, and every iterate that misses keeps the solve running
    /// at a point it cannot improve until the step collapses
    /// (`Search_Direction_Becomes_Too_Small`, on LPs whose optimum POUNCE
    /// already had to 8 significant figures).
    ///
    /// The per-component `constr_viol` gate reads the same floor, but as a
    /// boolean and only in the all-noise case — see `primal_component_passes`,
    /// which is gh #590's correction to this method's original scope. The
    /// scale-relative veto still sees the raw `rel_viol`, so a row violated by
    /// a meaningful fraction of its own magnitude is refused by that arm no
    /// matter what either floor says. The acceptable-level band is left on the
    /// raw `nlp_err`: once the strict gate understands the floor, a solve that
    /// is converged-to-noise takes the strict exit and never reaches the band.
    ///
    /// A non-finite `nlp_err` is passed through untouched — `f64::min` returns
    /// the *other* operand at `NaN`, which would launder exactly the
    /// `Invalid_Number_Detected` signal gh #292 built `curr_nlp_error`'s
    /// `has_valid_numbers` sweep to raise.
    ///
    /// On finite input the `min` is belt-and-braces rather than a live choice:
    /// `nlp_error(true)` shares its dual and complementarity terms with
    /// `nlp_error(false)` and `amax_above_floor` returns at most the vector's
    /// own `amax` on every path including its fallbacks, so
    /// `above_primal_noise <= nlp_err` always. It is kept so that the gate
    /// cannot be loosened by a future change to either accessor without that
    /// change being deliberate.
    /// Whether the gh #528 primal noise floor is live. `0` (or a negative
    /// value, which the option's lower bound already refuses) is the opt-out
    /// back to upstream Ipopt's bare-absolute primal term; the accessor is not
    /// even called then, so the opt-out costs nothing as well as changing
    /// nothing.
    fn noise_floor_enabled(&self) -> bool {
        self.primal_noise_floor_kappa > 0.0
    }

    fn strict_overall(nlp_err: Number, above_primal_noise: Number) -> Number {
        if !nlp_err.is_finite() {
            return nlp_err;
        }
        nlp_err.min(above_primal_noise)
    }

    /// Pure helper mirroring upstream
    /// `OptimalityErrorConvergenceCheck::CurrentIsAcceptable`. Tests
    /// the per-component `acceptable_*_tol` triplet plus the optional
    /// `acceptable_obj_change_tol` stability cross-check.
    fn passes_acceptable_tols(
        &self,
        overall: Number,
        dual_inf: Number,
        constr_viol: Number,
        compl_inf: Number,
        curr_f: Number,
    ) -> bool {
        // A point is never acceptable if the scaled error metric or the
        // objective itself is non-finite. Without the `curr_f` guard a NaN/Inf
        // objective with otherwise-small infeasibility (e.g. CUTE `himmelbj`,
        // where f evaluates to NaN at a near-feasible point) would be recorded
        // as the acceptable rollback point and reported under
        // `Solved_To_Acceptable_Level` with a `nan` objective.
        if !overall.is_finite() || !curr_f.is_finite() {
            return false;
        }
        let component_ok = overall <= self.acceptable_tol
            && dual_inf <= self.acceptable_dual_inf_tol
            && constr_viol <= self.acceptable_constr_viol_tol
            && compl_inf <= self.acceptable_compl_inf_tol;
        if !component_ok {
            return false;
        }
        // Upstream `IpOptErrorConvCheck.cpp:CurrentIsAcceptable` — when
        // an acceptable point has already been recorded and the user
        // tightened `acceptable_obj_change_tol` below the 1e20
        // sentinel, the iterate is only re-acceptable if `f` has moved
        // by less than `tol * max(1, |f|)` relative to the recorded
        // value. Skipped when no prior point exists or the cross-check
        // is disabled.
        if self.acceptable_obj_change_tol < 1e20 {
            if let Some(prev) = self.last_acceptable_obj {
                let denom = curr_f.abs().max(1.0);
                if (prev - curr_f).abs() >= self.acceptable_obj_change_tol * denom {
                    return false;
                }
            }
        }
        true
    }

    /// Advance the acceptable-level streak, returning whether the run should
    /// terminate with `ConvergedToAcceptable`.
    ///
    /// Acceptable-level termination is **count-based**: it needs
    /// `acceptable_iter` *consecutive* qualifying iterates. The masked-scale
    /// veto (gh #200) suppresses that termination, so the count has to keep
    /// running underneath the suppression — otherwise the mechanism cannot know
    /// where the unvetoed run would have stopped.
    ///
    /// The subtle part, and an earlier bug: `masked` is **not constant over a
    /// run**. `obj_scale` is fixed, but the veto's other condition is
    /// `unscaled_err > acceptable_tol`, and that quantity crosses the bar
    /// during the endgame — the crossing *is* the veto lifting. A streak can
    /// therefore straddle the boundary. Keeping two disjoint counters (a real
    /// one and a shadow), each reset by the other's phase, silently discarded a
    /// streak the unvetoed run would have kept: fourteen unmasked qualifying
    /// iterates followed by one masked qualifying iterate left the real count at
    /// zero, where the baseline would have reached fifteen and stopped. The run
    /// then fell through to `max_iter` — with no snapshot armed, because the
    /// shadow had only just started — and returned a bare failure where the
    /// baseline returned `Solved_To_Acceptable_Level`. That is precisely the
    /// "never worse" guarantee failing.
    ///
    /// So there is **one** counter, advanced on `acceptable_now` regardless of
    /// `masked`. `masked` decides only what happens when it crosses the
    /// threshold: terminate, or record that a termination was refused here —
    /// which is exactly the iterate the unvetoed run would have returned.
    ///
    /// The gh #533 progress test is the second thing that can refuse at the
    /// crossing, and it is undone by the same machinery — see
    /// [`Self::streak_has_flattened`]. Everything about the count is unchanged
    /// by it: the streak advances on the band test alone, so a progress refusal
    /// still records exactly the iterate the unvetoed run would have returned.
    fn note_acceptable(
        &mut self,
        acceptable_now: bool,
        masked: bool,
        nlp_err: Number,
        curr_f: Number,
    ) -> bool {
        if !acceptable_now {
            self.acceptable_count = 0;
            self.acceptable_window.clear();
            return false;
        }
        self.acceptable_count += 1;
        self.push_progress_sample(nlp_err, curr_f);
        if self.acceptable_count < self.acceptable_iter {
            return false;
        }
        if masked {
            self.acceptable_veto_fired = true;
            self.masked_acceptable_veto_fired = true;
            return false;
        }
        // gh #533: the streak says the error has been inside the band for
        // `acceptable_iter` iterations; it says nothing about whether the solve
        // has stopped moving. Refuse the termination while the window shows it
        // has not, and let the run continue — the refusal is recorded, so a run
        // that goes nowhere still ends at this point under this status.
        if !self.streak_has_flattened()
            && self.acceptable_progress_refusals < ACCEPTABLE_PROGRESS_MAX_REFUSALS
        {
            if !self.acceptable_veto_fired {
                tracing::info!(
                    nlp_err,
                    obj = curr_f,
                    acceptable_tol = self.acceptable_tol,
                    window = self.acceptable_window.len(),
                    kappa = self.acceptable_progress_kappa,
                    "refusing an acceptable-level termination: the error has been inside \
                     the acceptable band for the whole streak but is still moving across \
                     it, so the streak has not flattened; continuing \
                     (acceptable_progress_kappa=0 disables)"
                );
            }
            self.acceptable_progress_refusals += 1;
            self.acceptable_veto_fired = true;
            return false;
        }
        true
    }

    /// Length of the streak window the gh #533 progress test judges: the
    /// streak's own length, clamped to `1..=`[`ACCEPTABLE_PROGRESS_WINDOW_MAX`].
    ///
    /// A length of 1 is representable and means the test is inert:
    /// [`Self::streak_has_flattened`] declines to judge a window that short,
    /// because a single iterate carries no progress information. So
    /// `acceptable_iter = 1` never refuses, which is right — the user asked to
    /// stop at the first qualifying iterate.
    fn progress_window_len(&self) -> usize {
        (self.acceptable_iter.max(1) as usize).clamp(1, ACCEPTABLE_PROGRESS_WINDOW_MAX)
    }

    /// Record one qualifying iterate in the streak window, evicting the oldest
    /// sample once the window is full.
    fn push_progress_sample(&mut self, nlp_err: Number, curr_f: Number) {
        let cap = self.progress_window_len();
        self.acceptable_window.push_back((nlp_err, curr_f));
        while self.acceptable_window.len() > cap {
            self.acceptable_window.pop_front();
        }
    }

    /// Has the solve actually *flattened* over the iterates that made up the
    /// acceptable-level streak (gh #533)?
    ///
    /// The streak criterion on its own is a band test repeated
    /// `acceptable_iter` times: it asks whether the KKT error is small, never
    /// whether anything has stopped moving. Those come apart, and when they do
    /// the solve stops at a point that is near-stationary *for the current
    /// barrier subproblem* — a much weaker statement than near-KKT for the NLP —
    /// and returns a worse answer under a weaker status than continuing would
    /// have reached. Measured on two corpus models at `main @ 880b360b`:
    /// `kissing` (Vanderbei) stopped at iteration 103 with objective
    /// `1.00000108` and `Solved_To_Acceptable_Level`, where continuing reaches
    /// `0.84544259` and a strict certificate at 550 — 18% high, and Ipopt's own
    /// answer to eight figures is the lower one; `NARX_CFy` (Mittelmann)
    /// stopped at 565 with both residuals near `1e-7`, where 60 more iterations
    /// (25 s, inside the benchmark's 300 s limit) collapse them by five orders
    /// and beat both its own acceptable answer and Ipopt's.
    ///
    /// So: flat means *neither the error nor the objective moved* across the
    /// window, and the yardstick for both is a fraction
    /// `acceptable_progress_kappa` of `acceptable_tol` —
    ///
    /// - the error's absolute spread `max − min` against
    ///   `kappa · acceptable_tol`;
    /// - the objective's spread against `kappa · acceptable_tol · max(1, |f|)`,
    ///   the same relative form upstream's own `acceptable_obj_change_tol`
    ///   cross-check uses.
    ///
    /// **Spread, not trend, and either one alone is enough to refuse.** Both
    /// choices are load-bearing, and `kissing` is why:
    ///
    /// - Its `inf_du` over the last four iterates of the streak ran `3.35e-08 →
    ///   8.18e-08 → 1.08e-07 → 4.15e-07` — the error the solver stopped on was
    ///   an order of magnitude *worse* than one it had already achieved inside
    ///   the same streak. A trend test reads that as "not improving" and stops;
    ///   a spread test reads it as what it is, an iterate still wandering
    ///   across the band, and keeps going. The same holds in the other
    ///   direction: an error still descending through the band has not settled
    ///   either, and a solve that is still descending is one that may yet
    ///   certify.
    /// - Its objective was flat to all eight printed figures over those same
    ///   iterates (`1.0000011e+00` throughout) while the continued run moved it
    ///   by 15%. Requiring *both* signals to show movement before refusing
    ///   would therefore have stopped exactly where it stopped before.
    ///
    /// The band is the right yardstick because the question is scoped to it:
    /// the point is being certified as good to `acceptable_tol`, so "settled"
    /// has to mean settled on that scale. It also gets the user-intent
    /// monotonicity right in the one direction that matters — a *widened*
    /// `acceptable_tol` widens the flat bar with it, so a user who asked for an
    /// early exit at a loose band keeps getting one. Tightening
    /// `acceptable_tol` makes the test more eager to keep solving, which is the
    /// direction that cannot fabricate a verdict: a refusal is always undone at
    /// the end of a run that fails to do better (see
    /// `IpoptAlgorithm::honour_refused_certificate`), so its worst case is
    /// spent iterations, never a wrong answer.
    ///
    /// Returns `true` — flat, terminate — whenever the test cannot see enough
    /// to judge: `acceptable_progress_kappa <= 0` (the documented opt-out) or
    /// `NaN`, a window not yet full, a window of one, or any non-finite sample.
    /// Refusing on missing evidence would spend iterations for no stated reason.
    fn streak_has_flattened(&self) -> bool {
        if self.acceptable_progress_kappa.is_nan() || self.acceptable_progress_kappa <= 0.0 {
            return true;
        }
        // A partial window is not evidence of movement. (Unreachable from
        // `note_acceptable`, which only asks once the count has reached
        // `acceptable_iter` and pushes one sample per count, but the predicate
        // must not depend on that coincidence.)
        if self.acceptable_window.len() < self.progress_window_len()
            || self.acceptable_window.len() < 2
        {
            return true;
        }
        let bar = self.acceptable_progress_kappa * self.acceptable_tol;
        let (mut err_lo, mut err_hi) = (Number::INFINITY, Number::NEG_INFINITY);
        let (mut f_lo, mut f_hi) = (Number::INFINITY, Number::NEG_INFINITY);
        for &(err, f) in &self.acceptable_window {
            if !err.is_finite() || !f.is_finite() {
                return true;
            }
            err_lo = err_lo.min(err);
            err_hi = err_hi.max(err);
            f_lo = f_lo.min(f);
            f_hi = f_hi.max(f);
        }
        // `f` from the newest sample, matching `passes_acceptable_tols`'
        // `max(1, |f|)` denominator convention.
        let f_curr = self.acceptable_window.back().map_or(0.0, |&(_, f)| f);
        let err_flat = err_hi - err_lo <= bar;
        let obj_flat = f_hi - f_lo <= bar * f_curr.abs().max(1.0);
        err_flat && obj_flat
    }

    /// Fraction of a row's own magnitude a violation must exceed before the
    /// scale-relative machinery treats the row as genuinely violated —
    /// used both to veto a success certificate and as an alternative
    /// violation floor for rapid infeasibility detection.
    ///
    /// `max(100·constr_viol_tol, 1e-2)`: at the default `constr_viol_tol =
    /// 1e-4` this is 1% — a row eaten to 1% of everything it is made of is not
    /// a satisfied row at any scale. The `1e-2` floor is deliberate slack for
    /// the accepting direction: an interior-point run converges inequality
    /// residuals to *absolute* levels, so on a row of magnitude `1e-6` a
    /// converged residual near `1e-9` is a solved row at 0.1% relative — a
    /// tighter relative bar would veto genuine solutions on small-magnitude
    /// rows, the exact failure the clamped form in
    /// `pounce_common::tolerance::is_negligible` exists to avoid. The scale
    /// non-invariance this leaves (`x >= 0.7` at row scale `1e-12` is violated
    /// by 14%, well above any plausible bar; a knife-edge 0.9% violation is
    /// not) is the conservative direction: too-loose withholds a verdict,
    /// too-tight fabricates one.
    fn relative_viol_threshold(&self) -> Number {
        (100.0 * self.constr_viol_tol).max(MIN_INFEAS_VIOL_FLOOR)
    }

    /// Absolute violation floor for rapid infeasibility detection:
    /// `max(infeas_viol_kappa · constr_viol_tol, 1e-2)`.
    ///
    /// The same shape as [`Self::relative_viol_threshold`] and clamped for the
    /// same reason (gh #519): the product alone slides with the user's
    /// feasibility tolerance, so a *tighter* `constr_viol_tol` admitted smaller
    /// and smaller violations as evidence of infeasibility — the one direction
    /// a feasibility tolerance must never move this predicate. See
    /// [`MIN_INFEAS_VIOL_FLOOR`]. Raising `infeas_viol_kappa` still raises the
    /// floor; the clamp only stops it from falling below what the defaults
    /// consider an acceptable violation.
    fn absolute_viol_threshold(&self) -> Number {
        (self.infeas_viol_kappa * self.constr_viol_tol).max(MIN_INFEAS_VIOL_FLOOR)
    }

    /// Pure predicate for a single infeasible-stationary iterate: the
    /// constraint violation is bounded away from zero — absolutely
    /// (`constr_viol` above [`Self::absolute_viol_threshold`]) **or relative to
    /// the violated row's own magnitude** (`rel_viol` above
    /// [`Self::relative_viol_threshold`]; a row violated by 10% of everything
    /// it is made of is bounded away from feasible no matter how small its
    /// numbers are) — and the scaled infeasibility gradient `‖Jᵀc‖/max(1,‖c‖)`
    /// is at or below `infeas_stationarity_tol`. Returns `false` when rapid
    /// infeasibility detection is disabled (either knob non-positive).
    ///
    /// The relative arm changes only this pre-filter; the verdict still
    /// requires the direct no-descent confirmation in
    /// `check_convergence_with_state`, which is what protects against the
    /// false-infeasibility failures the surrogate alone was measured to
    /// produce.
    fn is_infeasible_stationary(
        &self,
        constr_viol: Number,
        rel_viol: Number,
        stationarity: Number,
        primal_resolvable: bool,
    ) -> bool {
        if self.infeas_stationarity_tol <= 0.0 || self.infeas_max_streak <= 0 {
            return false;
        }
        // gh #590: the absolute arm additionally requires the violation to be
        // something the model's own arithmetic can resolve. `primal_resolvable`
        // is `false` only when every constraint row sits at or below its own
        // floating-point noise floor (see `primal_component_passes`), and a
        // residual no iterate could place is not evidence of infeasibility — it
        // is the quantum the row is measured in. Convicting on it is the worst
        // failure this predicate has: `Infeasible_Problem_Detected` is a
        // *confident* verdict, and downstream it becomes an AMPL 200 / Pyomo
        // `infeasible`, indistinguishable from a real proof. The relative arm
        // is deliberately left alone; it is already a ratio against the row's
        // own magnitude, so it cannot mistake a quantum for a violation.
        let absolute_arm = constr_viol > self.absolute_viol_threshold() && primal_resolvable;
        (absolute_arm || rel_viol > self.relative_viol_threshold())
            && stationarity <= self.infeas_stationarity_tol
    }

    /// Advance the rapid-infeasibility-detection streak by one
    /// iteration. An infeasible-stationary iterate (see
    /// [`Self::is_infeasible_stationary`]) increments the streak; any
    /// other iterate resets it to zero. Returns `true` once the streak
    /// reaches `infeas_max_streak`, signalling the caller to terminate
    /// with `ConvergenceStatus::LocallyInfeasible`. The streak guards
    /// against firing on a transient flat spot.
    ///
    /// The **relative** arm additionally requires the relative violation to
    /// have stopped improving — "bounded away from feasible" must mean *not
    /// still converging*. The no-descent confirmation cannot provide that
    /// guard here: it compares violations absolutely, so in the small-scale
    /// regime the relative arm targets (violation ~1e-9 and falling), no
    /// "materially less-violating" point registers and the confirmation is
    /// vacuous. Measured on QSCORPIO: the detector fired at iteration 57 with
    /// the endgame still cutting the violation 16× over its last five
    /// iterations (4.6e-9 → 2.9e-10 relative 4.6e-2 → 2.9e-3); five more
    /// iterations reached `Optimal Solution Found`. An iterate that improved
    /// the relative violation by more than 10% since the previous check
    /// therefore resets the streak; a genuinely infeasible row's violation is
    /// pinned at its infeasibility gap and cannot improve at all.
    fn note_infeasible_stationary(
        &mut self,
        constr_viol: Number,
        rel_viol: Number,
        stationarity: Number,
        primal_resolvable: bool,
    ) -> bool {
        let still_improving = rel_viol < 0.9 * self.prev_rel_viol;
        self.prev_rel_viol = rel_viol;
        // Only the relative arm is progress-gated; the absolute arm keeps its
        // own guard (the direct no-descent confirmation, which is meaningful
        // at absolute violation scales).
        let effective_rel = if still_improving { 0.0 } else { rel_viol };
        if self.is_infeasible_stationary(
            constr_viol,
            effective_rel,
            stationarity,
            primal_resolvable,
        ) {
            self.infeas_streak += 1;
            self.infeas_streak >= self.infeas_max_streak
        } else {
            self.infeas_streak = 0;
            false
        }
    }
}

impl ConvCheck for OptErrorConvCheck {
    /// Snapshot every counter, run the real check, put them back. The
    /// struct is `Clone` for this and only this — the alternative is a
    /// hand-maintained list of the mutable fields, and this type has
    /// grown a new one for roughly every issue it has been through
    /// (`infeas_streak`, `veto_extra_iters`, `acceptable_window`,
    /// `acceptable_progress_refusals`, `rel_infeas_extra_iters`,
    /// `prev_rel_viol`, ...). A list like that is wrong the first time
    /// someone adds a field and does not think of this method.
    fn probe_convergence(
        &mut self,
        nlp_err: Number,
        iter_count: Index,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> ConvergenceStatus {
        let saved = self.clone();
        let status = self.check_convergence_with_state(nlp_err, iter_count, data, cq);
        *self = saved;
        status
    }

    fn certificate_vetoed(&self) -> bool {
        self.veto_fired
    }

    fn acceptable_certificate_vetoed(&self) -> bool {
        self.acceptable_veto_fired
    }

    fn check_convergence(&mut self, nlp_err: Number, iter_count: Index) -> ConvergenceStatus {
        if nlp_err <= self.tol {
            return ConvergenceStatus::Converged;
        }
        // `acceptable_iter == 0` disables acceptable-level termination,
        // mirroring upstream `IpOptErrorConvCheck.cpp:241`
        // (`if( acceptable_iter_ > 0 && CurrentIsAcceptable() )`). Without
        // the `> 0` guard, a zero would make `acceptable_count >= 0` fire on
        // the first acceptable iterate — the opposite of "disabled".
        //
        // The gh #533 progress test deliberately does NOT live here. It needs
        // the objective, which this entry point does not receive, and its two
        // callers do not want it: unit tests exercising the scalar state
        // machine, and `RestoConvCheckAdapter`, whose inner acceptable-level
        // answer feeds the "may the trial point leave restoration" decision
        // rather than a user-facing verdict — and which has no refused-
        // certificate fallback of its own to undo a refusal with.
        if self.acceptable_iter > 0 && nlp_err <= self.acceptable_tol {
            self.acceptable_count += 1;
            if self.acceptable_count >= self.acceptable_iter {
                return ConvergenceStatus::ConvergedToAcceptable;
            }
        } else {
            self.acceptable_count = 0;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
        }
        ConvergenceStatus::Continue
    }

    fn check_convergence_with_state(
        &mut self,
        nlp_err: Number,
        iter_count: Index,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> ConvergenceStatus {
        // Mirror upstream `IpOptErrorConvCheck.cpp::CheckConvergence`:
        // the scaled scalar `nlp_err` must drop below `tol` AND each
        // per-component value must sit under its own tolerance. The
        // component tolerances (`dual_inf_tol`/`constr_viol_tol`/
        // `compl_inf_tol`) are defined on the *unscaled* (user-original)
        // residuals — both upstream and per pounce's own option help text
        // — so we gate on the unscaled accessors. This resolves the former
        // M1 deviation (gating on internally-scaled residuals), which let
        // an ill-conditioned, nlp_scaling-deflated solve report
        // `Solve_Succeeded` while the user-space duals had drifted
        // (pounce#173). When no scaling is active the unscaled accessors
        // return the scaled values unchanged, so behaviour is identical on
        // the common path.
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        let rel_viol = cq_ref.curr_relative_primal_infeasibility_max();
        let curr_f = cq_ref.curr_f();
        let unscaled_err = cq_ref.curr_unscaled_nlp_error();
        // gh #528 — see `strict_overall`. Only the strict gate below reads
        // this; `nlp_err` itself carries on to the acceptable-level band, the
        // rapid-infeasibility pre-filter and everything downstream unchanged.
        //
        // Computed only on the iterations where it can change the verdict.
        // That laziness is doing real work, because the accessor is not two
        // extra Jacobian sweeps on top of a cached number — it is
        // `nlp_error(true)`, a second evaluation of the *whole* KKT error:
        // `optimality_error_scaling`, `curr_grad_lag_x`/`_s` (each a fresh
        // allocation plus two mat-vecs, and uncached — `nlp_error` has no
        // entry among the caches in `ipopt_cq.rs`), all four complementarity
        // vectors and the `has_valid_numbers` sweep, plus the two
        // `compute_row_amax` sweeps the floors need. Anyone reusing this
        // accessor anywhere hotter should read that cost first.
        //
        // The laziness is exact, not an approximation: below `tol` the floored
        // value is smaller still and the gate passes either way, and with any
        // component tolerance already blown `passes_component_tols` is false
        // whatever the aggregate says.
        //
        // gh #532 — the scale-relative floor under `dual_inf_tol`. Computed on
        // the same terms `dual_inf` was assembled from, and only where it can
        // change the verdict: below `dual_inf_tol` the absolute arm has already
        // passed and the floor can only be looser, and with the primal or
        // complementarity component already blown no floor on the dual makes a
        // certificate. That laziness matters because the accessor repeats
        // `curr_grad_lag_x`'s `∇f` and two transpose products.
        // gh #590 — the primal component's noise floor; see
        // `primal_component_passes`. Computed only where it can change the
        // verdict: with `constr_viol` already inside its tolerance the raw
        // comparison has passed, and with complementarity blown no floor on the
        // primal makes a certificate. That leaves the tail of a solve that is
        // complementarity-converged but primal-blown, which is the regime this
        // is for. The accessor is two `compute_row_amax` sweeps — an order
        // cheaper than the `curr_nlp_error_above_primal_noise` below, which is
        // a second evaluation of the whole KKT error — so this guard can afford
        // to be the looser of the two.
        let primal_resolvable = !(self.noise_floor_enabled()
            && constr_viol > self.constr_viol_tol
            && compl_inf <= self.compl_inf_tol
            && cq_ref.curr_primal_infeasibility_above_noise(self.primal_noise_floor_kappa) == 0.0);
        let primal_compl_pass = self.primal_component_passes(constr_viol, primal_resolvable)
            && compl_inf <= self.compl_inf_tol;
        let dual_scale =
            if primal_compl_pass && dual_inf > self.dual_inf_tol && self.dual_inf_scale_kappa > 0.0
            {
                cq_ref.curr_unscaled_dual_infeasibility_scale_max()
            } else {
                0.0
            };
        let components_pass = primal_compl_pass && dual_inf <= self.dual_inf_bound(dual_scale);
        let strict_err = if nlp_err <= self.tol || !components_pass || !self.noise_floor_enabled() {
            nlp_err
        } else {
            Self::strict_overall(
                nlp_err,
                cq_ref.curr_nlp_error_above_primal_noise(self.primal_noise_floor_kappa),
            )
        };
        // The gate asks whether *our* scaling clamped, not how the user chose
        // to scale their objective — see `certificate_masked`.
        let obj_scale = cq_ref.computed_obj_scaling_factor();
        drop(cq_ref);

        // Scale-relative feasibility veto (#385 Step 6; extended to equality
        // rows by #390, which plumbs the pre-fold RHS back so `|c_i|` has a
        // declared magnitude to be relative to). The absolute
        // `constr_viol_tol` gate cannot tell "satisfied" from "violated by 14%
        // of everything the row is" once the row's numbers are small: `x >= 0.7`
        // written as `1e-12·x >= 0.7e-12` has an absolute violation of `1e-13`
        // at `x = 0.6` — under every absolute tolerance, while the same empty
        // feasible set written at unit scale is reported infeasible. Refuse a
        // certificate whose point still has a constraint row violated by more
        // than `relative_viol_threshold` of its own magnitude, and let the run
        // continue: for a genuinely infeasible model the rapid-infeasibility
        // detection below then reaches the honest verdict (its violation floor
        // understands the same relative measure), and for anything else the
        // budget bounds the cost — after `VETO_MAX_EXTRA_ITERS` blocked
        // iterations the veto disengages and the run terminates exactly as it
        // would have, so no verdict is ever lost to it.
        let rel_veto = rel_viol > self.relative_viol_threshold()
            && self.rel_infeas_extra_iters < VETO_MAX_EXTRA_ITERS;
        let mut rel_veto_blocked = false;

        // gh #200: refuse a certificate the objective scaling has masked, and
        // keep iterating. A constant objective scale cancels out of the Newton
        // step and every line-search test is scale-invariant, so the continued
        // run follows exactly the trajectory an unscaled run would and reaches
        // the true minimum — at which point the unscaled error falls under
        // `acceptable_tol`, the veto lifts, and an honest strict certificate is
        // issued. Refusing to stop early is the whole intervention; the strict
        // tolerance in scaled space is untouched.
        // Only the masked arms spend the masked veto's budget — see
        // `masked_acceptable_veto_fired`.
        if self.veto_fired || self.masked_acceptable_veto_fired {
            self.veto_extra_iters += 1;
        }
        // Call the bet off once it has plainly not paid off, so a veto that can
        // never lift cannot cost an unbounded number of iterations. The refused
        // certificate is restored regardless, so this bounds cost, not
        // correctness.
        let budget_spent = self.veto_extra_iters > VETO_MAX_EXTRA_ITERS;
        // A non-finite objective disqualifies the veto outright. `passes_component_tols`
        // never inspects `f`, so a strict certificate can pass at an iterate whose
        // objective is NaN while its residuals are finite and tiny — and the unvetoed
        // run returns exactly that, NaN objective and all. Refusing it would arm a
        // snapshot the restore then declines (`honour_refused_certificate` requires a
        // finite objective), surfacing a failure where the baseline reported success.
        // Declining to engage keeps that case bit-identical to the baseline instead.
        // The acceptable-level side already had this property: finite `f` is a
        // precondition of qualifying there.
        let masked = curr_f.is_finite()
            && !budget_spent
            && certificate_masked(
                obj_scale,
                unscaled_err,
                self.obj_scale_certificate_threshold,
                self.acceptable_tol,
            );
        // Record a refusal only when a strict certificate was genuinely on the
        // table. `masked` alone is far broader — it holds on ordinary iterates
        // long before convergence — and using it would arm the fallback (and
        // snapshot an arbitrary mid-solve iterate) on runs that were never
        // about to stop.
        let refusing_strict = masked
            && self.passes_component_tols(
                strict_err,
                dual_inf,
                constr_viol,
                compl_inf,
                dual_scale,
                primal_resolvable,
            );
        if refusing_strict && !self.veto_fired {
            self.veto_fired = true;
            tracing::info!(
                obj_scale,
                unscaled_kkt_error = unscaled_err,
                scaled_nlp_error = nlp_err,
                threshold = self.obj_scale_certificate_threshold,
                "refusing a termination certificate masked by an extreme objective scale; \
                 continuing toward the true minimum (obj_scale_certificate_threshold=0 disables)"
            );
        }

        if !masked
            && self.passes_component_tols(
                strict_err,
                dual_inf,
                constr_viol,
                compl_inf,
                dual_scale,
                primal_resolvable,
            )
        {
            if rel_veto {
                rel_veto_blocked = true;
                if self.rel_infeas_extra_iters == 0 {
                    tracing::info!(
                        rel_viol,
                        constr_viol,
                        threshold = self.relative_viol_threshold(),
                        "refusing a success certificate: a constraint row is still \
                         violated by more than the scale-relative threshold of its own \
                         magnitude; continuing (bounded by the veto budget)"
                    );
                }
            } else {
                // The certificate is going out with a dual infeasibility above
                // `dual_inf_tol`, which the end-of-run summary will print
                // beside `EXIT: Optimal Solution Found`. Say why, once.
                if dual_inf > self.dual_inf_tol && !self.dual_floor_reported {
                    self.dual_floor_reported = true;
                    tracing::info!(
                        dual_inf,
                        dual_scale,
                        dual_inf_tol = self.dual_inf_tol,
                        bound = self.dual_inf_bound(dual_scale),
                        "certifying with a dual infeasibility above dual_inf_tol: it is \
                         within the scale-relative floor set by the terms the Lagrangian \
                         gradient is built from (dual_inf_scale_kappa=0 disables)"
                    );
                }
                return ConvergenceStatus::Converged;
            }
        }
        // `acceptable_iter == 0` disables acceptable-level termination
        // (upstream `IpOptErrorConvCheck.cpp:241`). See `check_convergence`.
        // The veto covers this branch too, so a refused strict certificate is
        // not merely swapped for an acceptable-level one at the same wrong
        // point. Acceptable-point *storage* is deliberately left un-vetoed —
        // that stashed point is the rollback target if the run later stalls.
        let mut acceptable_now = self.acceptable_iter > 0
            && self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f);
        // The scale-relative veto covers the acceptable band for the same
        // reason the masked-scale veto does: a refused strict certificate must
        // not be swapped for an acceptable-level one at the same wrong point.
        if acceptable_now && rel_veto {
            acceptable_now = false;
            rel_veto_blocked = true;
        }
        if rel_veto_blocked {
            self.rel_infeas_extra_iters += 1;
        }
        if self.note_acceptable(acceptable_now, masked, nlp_err, curr_f) {
            return ConvergenceStatus::ConvergedToAcceptable;
        }
        if iter_count >= self.max_iter {
            return ConvergenceStatus::MaxIterExceeded;
        }
        // Rapid infeasibility detection — recognise an iterate
        // converging to a stationary point of the constraint
        // violation with the violation bounded away from zero, and
        // exit with `LocallyInfeasible` instead of grinding to
        // `max_iter` or thrashing restoration. Gated behind an
        // `infeas_max_streak`-iteration streak to avoid firing on a
        // transient flat spot. The outer guard skips the two
        // transpose-products when detection is disabled.
        if self.infeas_stationarity_tol > 0.0 && self.infeas_max_streak > 0 {
            // The surrogate here is a cheap PRE-FILTER, not the verdict. It is
            // a threshold on `||J^T c|| / max(1, ||c||)`, which is not
            // scale-invariant: under a row scaling `dc` the numerator carries
            // `dc^2` while the denominator clamps at 1, so an aggressive scaling
            // drives it to zero regardless of where the iterate is. That is how
            // HS13 from x0 = (1e4, 1e4) reached `5e-14` at a point whose
            // constraint violation was 0.51, and got reported infeasible.
            //
            // Retuning does not fix it. Measured over 800 corpus models, every
            // tolerance that fires on genuinely infeasible problems also
            // introduces new false infeasibility (>= 3 models at the smallest
            // viable value), and measuring the surrogate unscaled or
            // scale-invariantly does not separate the cases either. So the
            // surrogate stays as-is, and the claim the status actually makes --
            // that no local move reduces the violation -- is confirmed directly
            // before the verdict is issued.
            let stationarity = cq.borrow().curr_infeasibility_stationarity();
            // gh #590 — evaluated only when the absolute arm is what would
            // convict: both its own threshold and the stationarity test have to
            // be armed already, which on a healthy solve is never, and on a
            // genuinely infeasible one the accessor returns a positive value on
            // the first call and the verdict proceeds unchanged.
            let primal_resolvable = !(self.noise_floor_enabled()
                && constr_viol > self.absolute_viol_threshold()
                && stationarity <= self.infeas_stationarity_tol
                && cq
                    .borrow()
                    .curr_primal_infeasibility_above_noise(self.primal_noise_floor_kappa)
                    == 0.0);
            if self.note_infeasible_stationary(
                constr_viol,
                rel_viol,
                stationarity,
                primal_resolvable,
            ) {
                // gh#1146: `constr_viol` is the violation of the internal
                // slack reformulation, `max(‖c‖, ‖d − s‖)`, and the slack can
                // drift from `d(x)` while `d(x)` itself stays inside its
                // declared bounds — `s` is confined to `[d_l, d_u]`, so a gap
                // `d − s` never says `d` left them. An infeasibility verdict
                // is a claim about the *model*, and a point that satisfies
                // every declared row and the declared box to `constr_viol_tol`
                // refutes it outright, whatever the slack says. Measured: a
                // 1-variable `log(x)` model from `x0 = 0` under L-BFGS, whose
                // `log` row was scaled by ~1e-8 at the start, so the scaled
                // gap (1.4e-10) looked stationary while the unscaled one
                // (1.4e-2) cleared the floor — at `x = 0.444`, inside every
                // row, where the `inf_pr` column already printed `0.00`.
                //
                // Feasible means feasible on *both* arms the detector
                // convicts on. `rel_viol` is already a property of the
                // point (`d(x)` against its declared bounds, `c(x)` against
                // its right-hand side — no slack), so it is reused as is: a
                // model whose rows all live below `constr_viol_tol`
                // (`infeasible_square_scaled_1em4`, rows scaled by 1e-4) is
                // infeasible only relative to the rows' own magnitude, and
                // the absolute test alone would acquit it.
                let point_feasible = rel_viol <= self.relative_viol_threshold()
                    && cq.borrow().curr_declared_primal_violation_max() < self.constr_viol_tol;
                if point_feasible {
                    tracing::debug!(
                        target: "pounce::algorithm",
                        constr_viol,
                        "rapid infeasibility detection: refusing the verdict, the iterate \
                         satisfies every declared row and bound (the violation is the \
                         slack gap d(x) - s only); continuing (gh#1146)"
                    );
                    self.infeas_streak = 0;
                } else if cq.borrow().infeasibility_descent_available() {
                    // Descent exists: not a stationary point of the violation,
                    // so the surrogate was wrong here. Drop the streak and keep
                    // solving.
                    self.infeas_streak = 0;
                } else {
                    return ConvergenceStatus::LocallyInfeasible;
                }
            }
        }
        // Time-budget gates. When the application installed a shared
        // [`Deadline`] (pounce#242) it is authoritative: it measures
        // global elapsed time from a fixed start instant, so it fires
        // correctly even inside the restoration inner IPM, whose fresh
        // `timing.overall_alg` is never started. Absent a deadline (the
        // direct-driver / unit-test path), fall back to the `overall_alg`
        // timer, which `IpoptApplication` starts at the top of
        // `optimize_constrained`; `live_*` returns the running elapsed
        // without forcing a `start/end` cycle. Upstream
        // `IpOptErrorConvCheck.cpp::CheckConvergence` reads the
        // application-level start time similarly.
        let d = data.borrow();
        if let Some(deadline) = d.deadline.as_ref() {
            match deadline.exceeded() {
                Some(pounce_common::timing::DeadlineKind::Cpu) => {
                    return ConvergenceStatus::CpuTimeExceeded;
                }
                Some(pounce_common::timing::DeadlineKind::Wall) => {
                    return ConvergenceStatus::WallTimeExceeded;
                }
                None => {}
            }
        } else {
            let timing = &d.timing;
            if timing.overall_alg.live_cpu_time() >= self.max_cpu_time {
                return ConvergenceStatus::CpuTimeExceeded;
            }
            if timing.overall_alg.live_wallclock_time() >= self.max_wall_time {
                return ConvergenceStatus::WallTimeExceeded;
            }
        }
        ConvergenceStatus::Continue
    }

    fn current_passes_strict(
        &self,
        nlp_err: Number,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> bool {
        // The strict per-component gate of `check_convergence_with_state`, minus
        // the masking veto — see the trait doc. Unscaled per-component residuals,
        // matching that method (the `*_tol` triplet is defined on the
        // user-original residuals).
        let cq_ref = cq.borrow();
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        // Same noise-floored aggregate the strict gate uses (gh #528) — this
        // predicate exists to answer "would that gate have passed here?", so it
        // has to ask the same question. Same scale-relative dual floor
        // (gh #532), and lazily for the same reason.
        let strict_err = if self.noise_floor_enabled() {
            Self::strict_overall(
                nlp_err,
                cq_ref.curr_nlp_error_above_primal_noise(self.primal_noise_floor_kappa),
            )
        } else {
            nlp_err
        };
        let dual_scale = if dual_inf > self.dual_inf_tol && self.dual_inf_scale_kappa > 0.0 {
            cq_ref.curr_unscaled_dual_infeasibility_scale_max()
        } else {
            0.0
        };
        // Same primal noise floor the strict gate uses (gh #590), for the same
        // reason this predicate reuses the floored aggregate.
        let primal_resolvable = !(self.noise_floor_enabled()
            && constr_viol > self.constr_viol_tol
            && cq_ref.curr_primal_infeasibility_above_noise(self.primal_noise_floor_kappa) == 0.0);
        drop(cq_ref);
        self.passes_component_tols(
            strict_err,
            dual_inf,
            constr_viol,
            compl_inf,
            dual_scale,
            primal_resolvable,
        )
    }

    fn tol_or_default(&self) -> Number {
        self.tol
    }

    fn constr_viol_tol_or_default(&self) -> Number {
        self.constr_viol_tol
    }

    fn acceptable_constr_viol_tol_or_default(&self) -> Number {
        self.acceptable_constr_viol_tol
    }

    fn set_tolerance(&mut self, name: &str, value: Number) -> bool {
        match name {
            "tol" => self.tol = value,
            "dual_inf_tol" => self.dual_inf_tol = value,
            "constr_viol_tol" => self.constr_viol_tol = value,
            "compl_inf_tol" => self.compl_inf_tol = value,
            "acceptable_tol" => self.acceptable_tol = value,
            "acceptable_dual_inf_tol" => self.acceptable_dual_inf_tol = value,
            "acceptable_constr_viol_tol" => self.acceptable_constr_viol_tol = value,
            "acceptable_compl_inf_tol" => self.acceptable_compl_inf_tol = value,
            "acceptable_obj_change_tol" => self.acceptable_obj_change_tol = value,
            _ => return false,
        }
        true
    }

    fn current_is_acceptable(&self, nlp_err: Number) -> bool {
        // Scalar fallback used when the caller has no `IpoptCq` handle
        // (e.g. unit tests). The state-aware variant
        // [`Self::current_is_acceptable_with_state`] mirrors upstream
        // more faithfully by gating on the per-component
        // `acceptable_*_tol` triplet plus the obj-change cross-check.
        nlp_err.is_finite() && nlp_err <= self.acceptable_tol
    }

    fn current_is_acceptable_with_state(
        &self,
        nlp_err: Number,
        _data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
    ) -> bool {
        let cq_ref = cq.borrow();
        // Unscaled per-component residuals — see `check_convergence_with_state`
        // (the `acceptable_*_tol` triplet is likewise defined on the
        // user-original residuals).
        let dual_inf = cq_ref.curr_unscaled_dual_infeasibility_max();
        let constr_viol = cq_ref.curr_unscaled_primal_infeasibility_max();
        let compl_inf = cq_ref.curr_unscaled_complementarity_max();
        let rel_viol = cq_ref.curr_relative_primal_infeasibility_max();
        let curr_f = cq_ref.curr_f();
        drop(cq_ref);
        // The scale-relative veto reaches acceptable-point *storage* too,
        // unlike the masked-scale (#200) veto above it. That veto refuses a
        // possibly-premature stop at a point that is still genuinely feasible,
        // so the stash stays a legitimate rollback target. Here the point has
        // a constraint row violated by more than the relative threshold of
        // its own magnitude — it is not acceptable in any honest sense, and a
        // stall later in the run must not roll back to it and surface
        // `Solved_To_Acceptable_Level` on an infeasible model (measured: an
        // infeasible row at scale `1e-10`, 100% violated, exited exactly that
        // way through this stash).
        //
        // gh #693: unlike the certificate veto above, this gate is **not**
        // budget-aware, and deliberately so. `VETO_MAX_EXTRA_ITERS` exists to
        // bound the *iterations* a veto can spend refusing to stop — the
        // certificate veto keeps the run going, so it can cost wall clock, and
        // the budget caps that. Declining to stash costs nothing: the stash is
        // a side effect of an iteration the run was taking anyway, so a budget
        // here bounds no cost. What it did do was expire, and once expired the
        // very point the veto exists to reject was written into the rollback
        // target — and `ConvergenceStatus::LocallyInfeasible` consults that
        // stash (gh #505, `ipopt_alg.rs`), so the honest infeasibility verdict
        // came back out as `Solved_To_Acceptable_Level`.
        //
        // Measured on `x >= 2` over `x in [0, 1]` with every row scaled by
        // `1e-8` (`test_scale_invariance.py::_inf_clear`, and `_inf_two`
        // likewise): the `feral_scaling=mc64` leg spends 288 iterations, blows
        // the 60-iteration budget around iteration 60, stashes an iterate whose
        // single row is violated by 99.998% of its own magnitude, then rolls
        // back to it on the infeasibility exit. The comment in `ipopt_alg.rs`
        // asserting this gate makes the stash "inert on genuinely infeasible
        // models" was true only for runs that convict inside 60 iterations.
        if rel_viol > self.relative_viol_threshold() {
            return false;
        }
        self.passes_acceptable_tols(nlp_err, dual_inf, constr_viol, compl_inf, curr_f)
    }

    fn set_curr_acceptable_obj(&mut self, obj: Number) {
        self.last_acceptable_obj = Some(obj);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_at_tol() {
        let mut c = OptErrorConvCheck::new();
        assert_eq!(c.check_convergence(1e-9, 0), ConvergenceStatus::Converged);
    }

    /// The scale-relative arm of rapid infeasibility detection (#385 Step 6):
    /// a row violated by a large fraction of its own magnitude is bounded away
    /// from feasible no matter how small its numbers are, so the pre-filter
    /// must fire even when the absolute violation is far below
    /// `infeas_viol_kappa * constr_viol_tol`.
    #[test]
    fn relative_violation_arms_the_infeasibility_prefilter() {
        let c = OptErrorConvCheck::new();
        // `x >= 0.7` at row scale 1e-12: absolute violation 1e-13 (invisible
        // to the absolute arm, floor is 1e-2), relative violation 0.14.
        assert!(c.is_infeasible_stationary(1e-13, 0.14, 1e-9, true));
        // The same iterate without the relative signal must NOT fire — this
        // is exactly the old behaviour.
        assert!(!c.is_infeasible_stationary(1e-13, 0.0, 1e-9, true));
        // A converged small-magnitude row (residual 1e-9 on a 1e-6-bound row,
        // 0.1% relative) stays under the 1% threshold.
        assert!(!c.is_infeasible_stationary(1e-9, 1e-3, 1e-9, true));
    }

    /// The relative arm's streak resets while the relative violation is still
    /// improving — "bounded away from feasible" must mean *not still
    /// converging*. QSCORPIO's endgame was cutting its violation 16× over
    /// five iterations when the un-guarded arm declared it locally
    /// infeasible; five more iterations reached the optimum.
    #[test]
    fn improving_relative_violation_resets_the_streak() {
        let mut c = OptErrorConvCheck::new();
        c.infeas_max_streak = 3;
        // A pinned relative violation (an infeasibility gap) accumulates.
        assert!(!c.note_infeasible_stationary(1e-13, 0.14, 1e-9, true));
        assert!(!c.note_infeasible_stationary(1e-13, 0.14, 1e-9, true));
        assert!(c.note_infeasible_stationary(1e-13, 0.14, 1e-9, true));
        // A geometrically shrinking one (a converging endgame) never fires.
        let mut c = OptErrorConvCheck::new();
        c.infeas_max_streak = 3;
        let mut rel = 0.5;
        for _ in 0..20 {
            assert!(
                !c.note_infeasible_stationary(1e-13, rel, 1e-9, true),
                "a converging endgame must not be declared infeasible"
            );
            rel *= 0.5;
        }
    }

    /// The relative-violation veto blocks a strict certificate the absolute
    /// tolerances would grant, and its budget bounds the cost: once spent,
    /// the certificate goes through exactly as before.
    #[test]
    fn relative_viol_threshold_is_floored() {
        let mut c = OptErrorConvCheck::new();
        // Default constr_viol_tol = 1e-4 -> threshold 1e-2.
        assert_eq!(c.relative_viol_threshold(), 1e-2);
        // A loosened constr_viol_tol loosens the relative bar with it.
        c.constr_viol_tol = 1e-3;
        assert_eq!(c.relative_viol_threshold(), 1e-1);
        // A tightened one must not push the relative bar below 1% — an
        // interior-point run converges inequality residuals to absolute
        // levels, and a tighter relative bar vetoes genuine solutions on
        // small-magnitude rows.
        c.constr_viol_tol = 1e-8;
        assert_eq!(c.relative_viol_threshold(), 1e-2);
    }

    #[test]
    fn acceptable_iter_count_threshold() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        // nlp_err between tol (1e-8) and acceptable (1e-6).
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 1), ConvergenceStatus::Continue);
        assert_eq!(
            c.check_convergence(1e-7, 2),
            ConvergenceStatus::ConvergedToAcceptable
        );
    }

    #[test]
    fn acceptable_iter_zero_disables_acceptable_termination() {
        // Upstream `IpOptErrorConvCheck.cpp:241` gates the acceptable
        // counter on `acceptable_iter_ > 0`, so a zero disables the
        // acceptable-level exit entirely. Before the guard, `>= 0` made
        // pounce fire on the FIRST acceptable iterate (the opposite).
        let mut c = OptErrorConvCheck {
            acceptable_iter: 0,
            ..Default::default()
        };
        // Many iterates parked between tol (1e-8) and acceptable (1e-6)
        // must never trigger ConvergedToAcceptable; the run continues
        // until tol or max_iter.
        for k in 0..50 {
            assert_eq!(
                c.check_convergence(1e-7, k),
                ConvergenceStatus::Continue,
                "acceptable_iter=0 must not stop at the acceptable level (iter {k})"
            );
        }
        // tol is still honored regardless.
        assert_eq!(c.check_convergence(1e-9, 51), ConvergenceStatus::Converged);
    }

    #[test]
    fn streak_resets_when_above_acceptable() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        assert_eq!(c.check_convergence(1e-7, 0), ConvergenceStatus::Continue);
        // Above acceptable resets the counter.
        assert_eq!(c.check_convergence(1e-3, 1), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 2), ConvergenceStatus::Continue);
        assert_eq!(c.check_convergence(1e-7, 3), ConvergenceStatus::Continue);
        assert_eq!(
            c.check_convergence(1e-7, 4),
            ConvergenceStatus::ConvergedToAcceptable
        );
    }

    #[test]
    fn passes_acceptable_tols_gates_on_per_component_triplet() {
        let c = OptErrorConvCheck {
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1e-3,
            acceptable_constr_viol_tol: 1e-3,
            acceptable_compl_inf_tol: 1e-3,
            ..Default::default()
        };
        assert!(c.passes_acceptable_tols(1e-7, 1e-4, 1e-4, 1e-4, 0.0));
        // dual_inf above its acceptable threshold blocks.
        assert!(!c.passes_acceptable_tols(1e-7, 1.0, 1e-4, 1e-4, 0.0));
        // overall above acceptable_tol blocks.
        assert!(!c.passes_acceptable_tols(1e-5, 1e-4, 1e-4, 1e-4, 0.0));
    }

    #[test]
    fn passes_acceptable_tols_honors_obj_change_tol() {
        let mut c = OptErrorConvCheck {
            acceptable_tol: 1e-6,
            acceptable_dual_inf_tol: 1.0,
            acceptable_constr_viol_tol: 1.0,
            acceptable_compl_inf_tol: 1.0,
            acceptable_obj_change_tol: 0.1,
            ..Default::default()
        };
        // First call always acceptable (no prior obj).
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 10.0));
        c.set_curr_acceptable_obj(10.0);
        // Same f → change well under threshold → still acceptable.
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 10.0));
        // f moved by 2.0 with threshold 0.1 * max(1, |11.0|) = 1.1 →
        // absolute change 1.0 < 1.1: acceptable.
        assert!(c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 11.0));
        // f moved by 5.0 — absolute change 5.0 > 1.5 = 0.1 * 15 →
        // rejected (the stability cross-check fires).
        assert!(!c.passes_acceptable_tols(1e-7, 0.0, 0.0, 0.0, 15.0));
    }

    use crate::conv_check::r#trait::ConvCheck;

    #[test]
    fn set_curr_acceptable_obj_records_for_cross_check() {
        let mut c = OptErrorConvCheck::new();
        assert!(c.last_acceptable_obj.is_none());
        ConvCheck::set_curr_acceptable_obj(&mut c, 4.2);
        assert_eq!(c.last_acceptable_obj, Some(4.2));
    }

    #[test]
    fn a_non_finite_objective_disqualifies_the_veto() {
        // `passes_component_tols` never inspects `f`, so a strict certificate can
        // pass at an iterate whose objective is NaN while its residuals are finite
        // and tiny — and the unvetoed run returns exactly that. Refusing it would
        // arm a snapshot that the restore then declines (it requires a finite
        // objective), surfacing a failure where the baseline reported success:
        // a never-worse violation, on the one path where the objective is not
        // usable as a tiebreak.
        let c = OptErrorConvCheck {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // The residuals alone say "converged"; the objective says nothing usable.
        assert!(c.passes_component_tols(1e-12, 1e-9, 0.0, 0.0, 0.0, true));
        // The masked predicate itself is unchanged — the finiteness gate lives at
        // the call site, where `curr_f` is in hand.
        assert!(certificate_masked(
            1e-8,
            8.4e-1,
            c.obj_scale_certificate_threshold,
            c.acceptable_tol
        ));
        // Both the guard's inputs behave as the call site composes them.
        for bad in [Number::NAN, Number::INFINITY, Number::NEG_INFINITY] {
            assert!(!bad.is_finite(), "{bad} should disqualify the veto");
        }
        assert!((1.0_f64).is_finite());
    }

    #[test]
    fn acceptable_streak_survives_a_masked_boundary_mid_streak() {
        // gh #200. `masked` is not constant over a run: it also depends on the
        // unscaled error crossing `acceptable_tol`, and that crossing is exactly
        // what happens during the endgame. So an acceptable-level streak can
        // straddle the boundary.
        //
        // The earlier implementation kept two disjoint counters, each reset by
        // the other's phase. Fourteen unmasked qualifying iterates followed by
        // one masked qualifying iterate left the real count at 0 while the
        // unvetoed run would have reached 15 and stopped — so the run fell
        // through to `max_iter` and returned a bare failure where the baseline
        // returned `Solved_To_Acceptable_Level`, with no snapshot armed to roll
        // back to. Never-worse, violated.
        //
        // Every iterate here is a *settled* one — same error, same objective —
        // so the gh #533 progress test is flat throughout and this test sees
        // only the masked-veto behaviour it is about. The progress test's own
        // arm is exercised in `a_wandering_streak_refuses_acceptable_termination`.
        const ERR: Number = 1e-7;
        const OBJ: Number = 1.0;
        let mut c = OptErrorConvCheck {
            acceptable_iter: 15,
            ..Default::default()
        };
        // 14 qualifying iterates while unmasked: no termination yet.
        for i in 0..14 {
            assert!(
                !c.note_acceptable(true, false, ERR, OBJ),
                "terminated early at {i}"
            );
        }
        // The 15th qualifies too, but the veto is now engaged. The streak must
        // be honoured — recorded as a refused termination, not discarded.
        assert!(
            !c.note_acceptable(true, true, ERR, OBJ),
            "a masked iterate must not terminate the run"
        );
        assert!(
            c.acceptable_veto_fired,
            "the streak crossed `acceptable_iter` while masked, so a termination was \
             refused here and must be recorded — otherwise the fallback has nothing to \
             restore and the run returns a bare failure"
        );
        assert!(
            c.masked_acceptable_veto_fired,
            "a masked refusal must be attributed to the masked arm — it is what spends \
             the masked veto's iteration budget"
        );

        // The mirror direction: a streak that begins masked and finishes
        // unmasked must terminate on the same iterate the baseline would.
        let mut c = OptErrorConvCheck {
            acceptable_iter: 15,
            ..Default::default()
        };
        for _ in 0..14 {
            assert!(!c.note_acceptable(true, true, ERR, OBJ));
        }
        assert!(
            c.note_acceptable(true, false, ERR, OBJ),
            "the veto lifted with the streak already at 14; the 15th qualifying iterate \
             must terminate exactly as it would without the mechanism"
        );

        // And a non-qualifying iterate still breaks the streak, in either phase.
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        assert!(!c.note_acceptable(true, false, ERR, OBJ));
        assert!(!c.note_acceptable(false, true, ERR, OBJ));
        assert_eq!(
            c.acceptable_count, 0,
            "a non-qualifying iterate resets the streak"
        );
        assert!(
            c.acceptable_window.is_empty(),
            "and clears the streak window"
        );
        assert!(!c.note_acceptable(true, false, ERR, OBJ));
        assert!(!c.note_acceptable(true, false, ERR, OBJ));
        assert!(
            c.note_acceptable(true, false, ERR, OBJ),
            "3 consecutive qualifying iterates terminate"
        );
    }

    /// gh #533. The reported `kissing` streak: fifteen iterates all inside the
    /// acceptable band, but with the KKT error wandering across it — the
    /// iterate the solve stopped on had an error an order of magnitude *worse*
    /// than one it had already reached in the same streak. The count alone
    /// stops there (objective `1.00000108`, `Solved_To_Acceptable_Level`);
    /// continuing reaches `0.84544259` with a strict certificate.
    #[test]
    fn a_wandering_streak_refuses_acceptable_termination() {
        // The tail of the reported trace (`main @ 880b360b`, default options):
        // inf_du 3.35e-08 → 8.18e-08 → 1.08e-07 → 4.15e-07 with the objective
        // flat to all eight printed figures throughout.
        let kissing_tail = [3.35e-08, 8.18e-08, 1.08e-07, 4.15e-07];
        let mut c = OptErrorConvCheck {
            acceptable_iter: 4,
            ..Default::default()
        };
        for (i, &err) in kissing_tail.iter().enumerate() {
            assert!(
                !c.note_acceptable(true, false, err, 1.0000011),
                "the streak must not terminate at iterate {i}: the error is still \
                 wandering across the acceptable band"
            );
        }
        assert!(
            c.acceptable_veto_fired,
            "the refusal must be recorded, or the run has nothing to fall back to"
        );
        assert!(
            !c.masked_acceptable_veto_fired,
            "a progress refusal is not a masked one and must not spend the masked \
             veto's budget"
        );
        // The count keeps running underneath the refusal — it is what identifies
        // the iterate the unvetoed run would have returned.
        assert_eq!(c.acceptable_count, 4);

        // Once the error settles, the window flattens — after the four-iterate
        // window has slid clear of the wandering tail — and the streak
        // terminates exactly as it would have without the mechanism.
        for _ in 0..2 {
            assert!(!c.note_acceptable(true, false, 4.15e-07, 1.0000011));
        }
        assert!(
            c.note_acceptable(true, false, 4.15e-07, 1.0000011),
            "a window of four identical iterates is settled; nothing is left to refuse"
        );
    }

    /// The other reported signal: `NARX_CFy`'s objective was still descending
    /// through the streak (`8.6579696e-03` at the stop, `8.6445195e-03` sixty
    /// iterations later) even where its error spread was small. Either signal
    /// alone must be enough to keep solving.
    #[test]
    fn a_still_descending_objective_refuses_acceptable_termination() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 4,
            ..Default::default()
        };
        // A perfectly steady error — only the objective is moving, by ~3e-6
        // over the window against a bar of 1e-1 · 1e-6 · max(1, |f|) = 1e-7.
        let objs = [8.6592e-03, 8.6588e-03, 8.6584e-03, 8.6580e-03];
        for (i, &f) in objs.iter().enumerate() {
            assert!(
                !c.note_acceptable(true, false, 1.5e-07, f),
                "the streak must not terminate at iterate {i}: the objective is still \
                 descending"
            );
        }
        assert!(c.acceptable_veto_fired);
    }

    /// The opt-out is real: `acceptable_progress_kappa = 0` restores the bare
    /// consecutive-count criterion, wandering error and all.
    #[test]
    fn zero_progress_kappa_restores_the_bare_count() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 4,
            acceptable_progress_kappa: 0.0,
            ..Default::default()
        };
        let kissing_tail = [3.35e-08, 8.18e-08, 1.08e-07, 4.15e-07];
        for (i, &err) in kissing_tail.iter().enumerate() {
            let terminated = c.note_acceptable(true, false, err, 1.0000011);
            assert_eq!(
                terminated,
                i == 3,
                "with the progress test off, iterate {i} must behave exactly as upstream"
            );
        }
        assert!(!c.acceptable_veto_fired);
    }

    /// The refusal budget bounds the cost of a solve that never settles: past
    /// [`ACCEPTABLE_PROGRESS_MAX_REFUSALS`] the test stands aside and the streak
    /// terminates as it would have without it, so the worst case is bounded
    /// extra iterations rather than a run to `max_iter`.
    #[test]
    fn the_progress_refusal_budget_is_bounded() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 2,
            ..Default::default()
        };
        // A permanent two-cycle inside the band: never flat, never converging.
        let mut terminated_at = None;
        for k in 0..(ACCEPTABLE_PROGRESS_MAX_REFUSALS + 10) {
            let err = if k % 2 == 0 { 1e-7 } else { 9e-7 };
            if c.note_acceptable(true, false, err, 1.0) {
                terminated_at = Some(k);
                break;
            }
        }
        assert_eq!(
            c.acceptable_progress_refusals, ACCEPTABLE_PROGRESS_MAX_REFUSALS,
            "the budget must be spent, not exceeded"
        );
        assert!(
            terminated_at.is_some(),
            "a never-settling solve must still terminate at the acceptable level once \
             the budget is spent"
        );
    }

    /// Flatness is judged over the streak's own window, and the window slides:
    /// a transient early in a solve must not block termination forever.
    #[test]
    fn the_flatness_window_slides_past_a_transient() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 3,
            ..Default::default()
        };
        // Entering the band while still descending: refused.
        assert!(!c.note_acceptable(true, false, 9e-7, 1.0));
        assert!(!c.note_acceptable(true, false, 5e-7, 1.0));
        assert!(!c.note_acceptable(true, false, 2e-7, 1.0));
        assert!(c.acceptable_veto_fired);
        // Then it plateaus. Two iterates later the descent has slid out of the
        // three-long window and the solve is judged settled.
        assert!(!c.note_acceptable(true, false, 2e-7, 1.0));
        assert!(
            c.note_acceptable(true, false, 2e-7, 1.0),
            "the window must slide, or an early transient blocks every later termination"
        );
    }

    /// `acceptable_iter = 1` asks to stop at the first qualifying iterate, and
    /// a one-iterate window carries no progress information — so the progress
    /// test must never refuse there.
    #[test]
    fn a_single_iterate_streak_carries_no_progress_signal() {
        let mut c = OptErrorConvCheck {
            acceptable_iter: 1,
            ..Default::default()
        };
        assert!(c.note_acceptable(true, false, 4.15e-07, 1.0));
        assert!(!c.acceptable_veto_fired);
    }

    /// A non-finite sample must not be read as movement — the mechanism spends
    /// iterations, so it may only fire on evidence it actually has.
    #[test]
    fn non_finite_samples_do_not_refuse() {
        for bad in [Number::NAN, Number::INFINITY] {
            let mut c = OptErrorConvCheck {
                acceptable_iter: 2,
                ..Default::default()
            };
            assert!(!c.note_acceptable(true, false, bad, 1.0));
            assert!(
                c.note_acceptable(true, false, 1e-7, 1.0),
                "a {bad} sample in the window must not be treated as a progress signal"
            );
        }
    }

    #[test]
    fn certificate_masked_needs_both_an_extreme_scale_and_a_non_stationary_point() {
        // gh #200. Both conditions are load-bearing, and each was independently
        // shown to be insufficient on the benchmark suite.
        let (th, atol) = (1e-4, 1e-6);

        // The reported failure: scale pinned at the 1e-8 floor, unscaled error
        // 0.84 — the strict test passed in scaled space at `quartc` obj 248.88.
        assert!(certificate_masked(1e-8, 8.4e-1, th, atol));

        // An ordinary objective scale is never second-guessed, however large
        // the unscaled error. Keying on the error alone effectively tightens
        // `tol` by `1/df` and regressed hs1/hs38 (scale ~4e-2).
        assert!(!certificate_masked(4e-2, 8.4e-1, th, atol));
        assert!(!certificate_masked(1.0, 1e3, th, atol));

        // An extreme scale at a point that really is stationary is fine — this
        // is what lifts the veto once the continued run reaches the minimum.
        assert!(!certificate_masked(1e-8, 1e-9, th, atol));

        // Boundaries: strictly below the scale threshold, strictly above the
        // error tolerance.
        assert!(!certificate_masked(th, 1.0, th, atol));
        assert!(!certificate_masked(1e-8, atol, th, atol));

        // `0` disables the mechanism outright (the documented opt-out) — the
        // most extreme possible inputs must not trip it.
        assert!(!certificate_masked(1e-30, 1e30, 0.0, atol));
        // A negative threshold is treated as disabled rather than as "always".
        assert!(!certificate_masked(1e-30, 1e30, -1.0, atol));
    }

    #[test]
    fn veto_blocks_both_strict_and_acceptable_termination() {
        // A refused strict certificate must not simply reappear as an
        // acceptable-level one at the same wrong point, so the veto covers both
        // branches. Exercised through the pure predicates the two branches
        // share, since a full `check_convergence_with_state` needs a live cq.
        let c = OptErrorConvCheck {
            tol: 1e-8,
            acceptable_tol: 1e-6,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // The gh #200 iterate: passes the strict test in scaled space...
        assert!(c.passes_component_tols(1e-9, 8.4e-1, 0.0, 0.0, 0.0, true));
        // ...and the veto is what withholds it.
        assert!(certificate_masked(
            1e-8,
            8.4e-1,
            c.obj_scale_certificate_threshold,
            c.acceptable_tol
        ));
        // Default threshold is the documented 1e-4, and the veto starts clear.
        assert_eq!(c.obj_scale_certificate_threshold, 1e-4);
        assert!(!c.veto_fired);
        assert!(!ConvCheck::certificate_vetoed(&c));
    }

    #[test]
    fn passes_component_tols_requires_all_under_threshold() {
        let c = OptErrorConvCheck {
            tol: 1e-8,
            dual_inf_tol: 1.0,
            constr_viol_tol: 1e-4,
            compl_inf_tol: 1e-4,
            ..Default::default()
        };
        // All under threshold → converged.
        assert!(c.passes_component_tols(1e-9, 0.5, 1e-5, 1e-5, 0.0, true));
        // dual_inf above its tolerance blocks even when nlp_err is tiny.
        assert!(!c.passes_component_tols(1e-12, 2.0, 1e-5, 1e-5, 0.0, true));
        // compl_inf above its tolerance blocks.
        assert!(!c.passes_component_tols(1e-12, 0.0, 0.0, 1e-2, 0.0, true));
        // constr_viol above its tolerance blocks.
        assert!(!c.passes_component_tols(1e-12, 0.0, 1e-2, 0.0, 0.0, true));
    }

    /// gh #590: the strict primal component forgives a violation only when
    /// *no* constraint row rose above its own floating-point noise floor. One
    /// resolvable row anywhere and `constr_viol_tol` is back in charge.
    #[test]
    fn primal_component_passes_only_forgives_an_unresolvable_residual() {
        let c = OptErrorConvCheck {
            constr_viol_tol: 1e-6,
            ..Default::default()
        };
        // Inside the tolerance: passes either way, the floor is not consulted.
        assert!(c.primal_component_passes(1e-9, true));
        assert!(c.primal_component_passes(1e-9, false));
        // The reported point: 1.62e-2 of violation, no row above its floor.
        assert!(c.primal_component_passes(1.620_777e-2, false));
        // The same violation with one row genuinely above its floor is refused.
        assert!(!c.primal_component_passes(1.620_777e-2, true));
        // The forgiveness is not bounded above — that is deliberate. What
        // bounds it is `primal_resolvable`, which no real violation can clear:
        // a row short by 1e6 is short by ~1e22 ulps of itself.
        assert!(c.primal_component_passes(1e6, false));
    }

    /// gh #590: the rapid-infeasibility detector's **absolute** arm carries the
    /// same requirement. Convicting on a residual the model's arithmetic cannot
    /// resolve is the worst failure this predicate has — the verdict is
    /// confident, and downstream it is indistinguishable from a real proof.
    /// The **relative** arm is untouched: it is already a ratio against the
    /// row's own magnitude, so it cannot mistake a quantum for a violation.
    #[test]
    fn the_absolute_infeasibility_arm_needs_a_resolvable_violation() {
        let c = OptErrorConvCheck {
            constr_viol_tol: 1e-6,
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 5,
            ..Default::default()
        };
        assert_eq!(c.absolute_viol_threshold(), MIN_INFEAS_VIOL_FLOOR);

        // The reported iterate: violation above the absolute floor, gradient
        // flat, relative violation negligible — and every row inside its own
        // noise floor. Before the fix this armed the detector.
        assert!(!c.is_infeasible_stationary(1.620_777e-2, 1e-8, 1e-9, false));
        // Identical, except one row is resolvable: the verdict stands.
        assert!(c.is_infeasible_stationary(1.620_777e-2, 1e-8, 1e-9, true));
        // The relative arm does not depend on the flag — a row violated by
        // more than the relative threshold of its own magnitude arms the
        // detector whatever the absolute floor says.
        assert!(c.is_infeasible_stationary(0.0, 1.0, 1e-9, false));
    }

    #[test]
    fn infeasible_stationary_requires_violation_and_flat_gradient() {
        let c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2, // violation threshold = 1e-2
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 5,
            ..Default::default()
        };
        // Violation well above 1e-2 and the infeasibility gradient
        // essentially zero → counts as infeasible-stationary.
        assert!(c.is_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        // Violation above threshold but the gradient is not flat →
        // still making feasibility progress, does not count.
        assert!(!c.is_infeasible_stationary(1e-1, 0.0, 1e-3, true));
        // Gradient flat but violation below threshold → nearly
        // feasible, does not count.
        assert!(!c.is_infeasible_stationary(1e-3, 0.0, 1e-9, true));
    }

    /// gh #519: tightening `constr_viol_tol` must never widen the set of
    /// points the detector is willing to call infeasible. The absolute arm's
    /// floor used to be `infeas_viol_kappa · constr_viol_tol` unclamped, so at
    /// `constr_viol_tol = 1e-6` it fell to `1e-4` — and @bernalde's `f=1`
    /// model (gh #505), plateaued at an unscaled violation of `1.943e-4`, was
    /// convicted at a point its own run reported as acceptable.
    #[test]
    fn tightening_constr_viol_tol_never_arms_the_absolute_arm_lower() {
        let plateau_viol = 1.9430136821e-4; // the measured `f=1` plateau
        // Every value at or below the default: tightening from here must not
        // move the floor at all, and certainly not downward.
        for &cvt in &[1e-4, 1e-5, 1.94e-6, 1e-7, 1e-9, 1e-12] {
            let c = OptErrorConvCheck {
                constr_viol_tol: cvt,
                ..Default::default()
            };
            let floor = c.absolute_viol_threshold();
            assert_eq!(
                floor, MIN_INFEAS_VIOL_FLOOR,
                "constr_viol_tol={cvt} moved the absolute floor to {floor}"
            );
            // The `f=1` plateau is a nearly-feasible flat spot at every one
            // of these tolerances, so no `constr_viol_tol` may arm the
            // absolute arm on it (the relative signal is 0 here: the row is
            // unit-scale, so only the absolute arm is in play).
            assert!(
                !c.is_infeasible_stationary(plateau_viol, 0.0, 1e-9, true),
                "constr_viol_tol={cvt} armed the detector on the {plateau_viol} plateau"
            );
        }
    }

    /// The clamp is a floor, not a cap: `infeas_viol_kappa` still raises the
    /// absolute threshold, and a violation genuinely bounded away from
    /// feasible still arms the detector at any `constr_viol_tol`.
    #[test]
    fn absolute_viol_floor_is_a_floor_not_a_cap() {
        let strict = OptErrorConvCheck {
            constr_viol_tol: 1e-9,
            ..Default::default()
        };
        assert_eq!(strict.absolute_viol_threshold(), 1e-2);
        assert!(strict.is_infeasible_stationary(0.5, 0.0, 1e-9, true));
        // Raising kappa above the floor still moves the threshold.
        let wide = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e4, // 1e0, well above the 1e-2 floor
            ..Default::default()
        };
        assert_eq!(wide.absolute_viol_threshold(), 1.0);
        assert!(!wide.is_infeasible_stationary(0.5, 0.0, 1e-9, true));
        assert!(wide.is_infeasible_stationary(2.0, 0.0, 1e-9, true));
        // Loosening `constr_viol_tol` past the floor moves it too — the floor
        // only binds from below.
        let loose = OptErrorConvCheck {
            constr_viol_tol: 1e-2,
            ..Default::default()
        };
        assert_eq!(loose.absolute_viol_threshold(), 1.0);
    }

    /// gh #508: the status-decision sites that ask "is this violation real"
    /// read `constr_viol_tol` off the policy, so a user setting has to reach
    /// them — the defect was a threshold built from `tol` that no
    /// `constr_viol_tol` value could move. `set_tolerance` is the debugger's
    /// live hot-swap path and must be visible through the accessor too.
    #[test]
    fn constr_viol_tol_accessor_tracks_the_option() {
        let mut c = OptErrorConvCheck {
            tol: 1e-6,
            constr_viol_tol: 1e-3,
            ..Default::default()
        };
        assert_eq!(c.constr_viol_tol_or_default(), 1e-3);
        // Independent of `tol` — retuning convergence must not retune what
        // counts as a violated constraint.
        c.tol = 1e-10;
        assert_eq!(c.constr_viol_tol_or_default(), 1e-3);
        assert!(c.set_tolerance("constr_viol_tol", 1e-7));
        assert_eq!(c.constr_viol_tol_or_default(), 1e-7);
    }

    #[test]
    fn infeasible_stationary_disabled_by_nonpositive_knobs() {
        let off_tol = OptErrorConvCheck {
            infeas_stationarity_tol: 0.0,
            infeas_max_streak: 5,
            ..Default::default()
        };
        assert!(!off_tol.is_infeasible_stationary(1e9, 0.0, 0.0, true));
        let off_streak = OptErrorConvCheck {
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 0,
            ..Default::default()
        };
        assert!(!off_streak.is_infeasible_stationary(1e9, 0.0, 0.0, true));
    }

    #[test]
    fn infeasible_stationary_streak_fires_only_after_max_streak() {
        let mut c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2, // violation threshold = 1e-2
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 3,
            ..Default::default()
        };
        // Infeasible-stationary iterate: violation 1e-1 > 1e-2, flat
        // gradient. Streak accrues but does not fire until the third.
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        assert!(c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
    }

    #[test]
    fn infeasible_stationary_streak_resets_on_feasibility_progress() {
        let mut c = OptErrorConvCheck {
            constr_viol_tol: 1e-4,
            infeas_viol_kappa: 1e2,
            infeas_stationarity_tol: 1e-8,
            infeas_max_streak: 3,
            ..Default::default()
        };
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        // A non-stationary iterate (gradient not flat) resets the streak.
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-3, true));
        assert_eq!(c.infeas_streak, 0);
        // The streak must rebuild from scratch — no carry-over credit.
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        assert!(!c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
        assert!(c.note_infeasible_stationary(1e-1, 0.0, 1e-9, true));
    }

    #[test]
    fn infeasible_stationary_streak_never_fires_when_disabled() {
        let mut c = OptErrorConvCheck {
            infeas_stationarity_tol: 0.0,
            infeas_max_streak: 5,
            ..Default::default()
        };
        for _ in 0..20 {
            assert!(!c.note_infeasible_stationary(1e9, 0.0, 0.0, true));
        }
        assert_eq!(c.infeas_streak, 0);
    }

    /// gh #532. The scale-relative floor under `dual_inf_tol`, on the numbers
    /// that produced the report: `orthrds2` must pass, and the runaway
    /// `min -exp(x) s.t. x >= 0` must not.
    #[test]
    fn dual_inf_bound_forgives_a_relatively_stationary_residual_only() {
        let c = OptErrorConvCheck::new();
        assert_eq!(c.dual_inf_tol, 1.0);
        assert_eq!(c.dual_inf_scale_kappa, 1.0);

        // `orthrds2`: ‖∇L‖_∞ = 89.7 against terms of magnitude ~1.6e12 (the
        // mean multiplier magnitude behind its `s_d ≈ 1.6e10`) — stationary to
        // nine digits relative to what it is made of, and refused by the bare
        // `1.0` before the fix.
        let (orthrds2_dual_inf, orthrds2_scale) = (89.669_051_358_301_67, 1.6e12);
        assert!(orthrds2_dual_inf > c.dual_inf_tol, "the reported refusal");
        assert!(orthrds2_dual_inf <= c.dual_inf_bound(orthrds2_scale));
        assert!(c.passes_component_tols(
            5.537e-9,
            orthrds2_dual_inf,
            1.741e-8,
            0.0,
            orthrds2_scale,
            true
        ));

        // `min -exp(x) s.t. x >= 0` running away: `∇f = −8.8e47` with no
        // multiplier to meet it, so nothing cancelled and the residual IS the
        // scale. Refused by eight orders — the case any such rule has to keep
        // rejecting.
        let runaway = 8.8e47;
        assert!(runaway > c.dual_inf_bound(runaway));
        assert!(!c.passes_component_tols(1e-12, runaway, 1.7e-10, 0.0, runaway, true));

        // The floor is a floor, never a tightening: below `dual_inf_tol` the
        // absolute arm decides, at any scale.
        assert_eq!(c.dual_inf_bound(1.0), c.dual_inf_tol);
        assert_eq!(c.dual_inf_bound(0.0), c.dual_inf_tol);
        assert_eq!(c.dual_inf_bound(1e-30), c.dual_inf_tol);
        // ...and it only lifts off `dual_inf_tol` once the scale passes
        // `dual_inf_tol / (kappa · tol)` = 1e8, so every `O(1)` model keeps the
        // upstream comparison bit for bit.
        assert_eq!(c.dual_inf_bound(1e7), c.dual_inf_tol);
        assert!(c.dual_inf_bound(1e10) > c.dual_inf_tol);

        // Non-finite scales say nothing and must not widen anything.
        for bad in [Number::NAN, Number::INFINITY, Number::NEG_INFINITY] {
            assert_eq!(c.dual_inf_bound(bad), c.dual_inf_tol, "scale {bad}");
        }
    }

    /// The floor tracks `tol`: asking for a stricter solve tightens the dual
    /// component gate in proportion, and `dual_inf_scale_kappa = 0` is the
    /// documented opt-out back to upstream's bare absolute bound.
    #[test]
    fn dual_inf_bound_tracks_tol_and_honours_the_opt_out() {
        let mut c = OptErrorConvCheck::new();
        assert_eq!(c.dual_inf_bound(1e12), 1e4);
        c.tol = 1e-10;
        assert_eq!(c.dual_inf_bound(1e12), 1e2);
        // Kappa scales the floor as advertised.
        c.tol = 1e-8;
        c.dual_inf_scale_kappa = 10.0;
        assert_eq!(c.dual_inf_bound(1e12), 1e5);
        // `0` (and, defensively, a negative or NaN value the option's own lower
        // bound already refuses) disables it outright — the most extreme scale
        // must not move the bound.
        for off in [0.0, -1.0, Number::NAN] {
            c.dual_inf_scale_kappa = off;
            assert_eq!(c.dual_inf_bound(1e30), c.dual_inf_tol, "kappa {off}");
            assert!(!c.passes_component_tols(1e-12, 89.7, 0.0, 0.0, 1.6e12, true));
        }
    }

    /// gh #528. The strict gate reads the noise-floored aggregate when that is
    /// the smaller of the two, and is otherwise untouched — the floored value
    /// can never *raise* the error.
    #[test]
    fn strict_overall_takes_the_noise_floored_aggregate() {
        // The reported case: KKT error pinned one ulp of `|b| ~ 1e8` above
        // `tol`, with the primal residual entirely inside its own resolution.
        assert_eq!(
            OptErrorConvCheck::strict_overall(1.49e-8, 9.09e-10),
            9.09e-10
        );
        // Nothing at its resolution limit: the two agree and the gate is the
        // upstream one, bit for bit.
        assert_eq!(OptErrorConvCheck::strict_overall(1e-9, 1e-9), 1e-9);
    }

    /// A non-finite KKT error must survive the floor untouched. `f64::min`
    /// returns the *other* operand at `NaN`, so a bare `min` would launder the
    /// `Invalid_Number_Detected` signal `curr_nlp_error`'s `has_valid_numbers`
    /// sweep exists to raise (gh #292).
    #[test]
    fn strict_overall_passes_a_non_finite_error_through() {
        assert!(OptErrorConvCheck::strict_overall(Number::NAN, 1e-12).is_nan());
        assert_eq!(
            OptErrorConvCheck::strict_overall(Number::INFINITY, 1e-12),
            Number::INFINITY
        );
    }

    #[test]
    fn max_iter_exceeded() {
        let mut c = OptErrorConvCheck {
            max_iter: 5,
            ..Default::default()
        };
        assert_eq!(
            c.check_convergence(1.0, 5),
            ConvergenceStatus::MaxIterExceeded
        );
    }
}
