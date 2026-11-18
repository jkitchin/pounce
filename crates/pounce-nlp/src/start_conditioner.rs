//! Starting-point conditioners, as transparent [`TNLP`] decorators.
//!
//! Both conditioners here answer the same finding: a large fraction of the
//! solves POUNCE loses are lost at the *starting point*, not in the barrier
//! iteration. On a 244-problem corpus (see `dev-notes/degenerate-starts.md`),
//! fifteen models ended `Infeasible_Problem_Detected` or
//! `Invalid_Number_Detected` from their bundled start; thirteen of the fifteen
//! solve cleanly from a start moved slightly off that point, with no change to
//! any algorithmic option.
//!
//! Neither conditioner touches the algorithm. Each wraps a TNLP and overrides
//! exactly one method — [`TNLP::get_starting_point`] — so the barrier solve
//! that follows is bit-for-bit the solve POUNCE would have run had the user
//! passed the conditioned point in the first place.
//!
//! # The two conditioners
//!
//! * [`StartConditioner::Jitter`] sanitises non-finite entries and applies a
//!   deterministic, seeded, bound-respecting displacement. It is what the
//!   local-infeasibility second-opinion ladder uses, because the measurement
//!   says the iterate does not need to be *better* — it needs to be
//!   *non-degenerate*.
//!
//! * [`StartConditioner::Adam`] runs Adam on the penalised merit
//!   `f(x) + ρ·‖violation(x)‖²` before handing over. This is the stage-0
//!   warm-up from KRONOS (Ahmed & Hasan, 2026), generalised from that paper's
//!   equality-only `ρ‖h(x)‖²` to two-sided constraint bounds so it applies to
//!   an arbitrary NLP rather than only to a squared-slack reformulation.
//!
//! # Why Adam is off by default
//!
//! It is a real preconditioner with a fat tail. Measured on 40 problems POUNCE
//! already solves, it broke none of them and cut the iteration count on 22 —
//! sometimes hard (`rk23` 82 → 11, `bt5` 45 → 9, `chnrosnb` 40 → 10). Median
//! 0.83×, geometric mean 0.79×. But the *total* went up 1.62×, because a few
//! problems blow out: `palmer1c` 71 → 1023, `heart6` 135 → 341, `biggs6`
//! 1906 → 2938. A fixed ρ against an unscaled merit walks a badly-scaled model
//! somewhere the barrier method then has to walk back from.
//!
//! A median win with a 14× tail is an option, not a default. Turning it on is
//! a *trajectory* change in the sense `CLAUDE.md` means: it reorders and
//! rescales every step that follows. Default-off keeps the fixture corpus
//! bit-identical, which is the property that lets this land without a sweep
//! diff to explain.
//!
//! # Determinism
//!
//! [`StartConditioner::Jitter`] draws from SplitMix64 seeded by its own `seed`
//! field and nothing else — no clock, no address, no thread id. The same seed
//! and the same incoming point produce the same conditioned point on every
//! platform, which is what makes a retry rung reproducible and a failure
//! reportable.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::{Index, Number};

use crate::tnlp::{
    BoundsInfo, IterStats, Linearity, MetaData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use crate::{IpoptCq, IpoptData};

/// Default sentinel past which a bound counts as absent. Matches the
/// `nlp_lower_bound_inf` / `nlp_upper_bound_inf` option defaults.
pub const DEFAULT_BOUND_INF: Number = 1e19;

/// Tuning for the Adam warm-up. Defaults are KRONOS's published stage-0
/// settings (Ahmed & Hasan, 2026): 200 iterations, `lr = 5e-2`, `ρ = 10`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamConfig {
    pub iters: usize,
    pub lr: Number,
    /// Penalty weight on the squared constraint violation.
    pub rho: Number,
    pub beta1: Number,
    pub beta2: Number,
    pub eps: Number,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            iters: 200,
            lr: 5e-2,
            rho: 10.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// How to condition the starting point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StartConditioner {
    /// Sanitise non-finite entries, then displace each variable by
    /// `scale · (1 + |xᵢ|) · uᵢ` with `uᵢ` drawn uniformly from `[-1, 1)`,
    /// clipped back into any present bound.
    ///
    /// The `1 +` matters: it is what makes the displacement nonzero at
    /// `xᵢ = 0`, and a start at the origin is the single most common
    /// degenerate start in the corpus.
    Jitter { seed: u64, scale: Number },
    /// Adam on `f(x) + ρ·‖violation(x)‖²`.
    Adam(AdamConfig),
}

/// What a conditioner did, for the log.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConditionerReport {
    /// Variables whose incoming value was not finite and was replaced.
    pub sanitised: Vec<usize>,
    /// Largest absolute displacement applied to any coordinate.
    pub max_shift: Number,
    /// Adam only: iterations actually run, which is fewer than the budget
    /// when the gradient stopped being finite.
    pub iters: usize,
    /// Adam only: merit before and after.
    pub merit_initial: Number,
    pub merit_final: Number,
    /// Adam only: the warm-up bailed on a non-finite gradient and kept the
    /// last good iterate.
    pub stopped_early: bool,
    /// Jitter only: how many times the displacement was halved before the
    /// model evaluated finitely at the displaced point (gh#1176). `0` when the
    /// first draw evaluated, which is every model whose functions are defined
    /// on the whole box.
    pub halvings: u32,
    /// Jitter only: no displacement down to [`MAX_JITTER_HALVINGS`] halvings
    /// evaluated finitely, so the full first draw was kept -- what the
    /// conditioner did before it looked at the point at all.
    pub unevaluable: bool,
}

/// How many times [`StartConditioner::Jitter`] halves a displacement whose
/// point the model cannot evaluate before giving up on it (gh#1176).
///
/// Ten halvings take the default `1e-2` scale to `~1e-5`, below which the
/// displaced point is close enough to the start that the rung would only
/// re-run the solve that already failed.
pub const MAX_JITTER_HALVINGS: u32 = 10;

/// A [`TNLP`] that conditions the inner model's starting point and forwards
/// everything else unchanged.
pub struct ConditionedStartTnlp {
    inner: Rc<RefCell<dyn TNLP>>,
    conditioner: StartConditioner,
    lower_inf: Number,
    upper_inf: Number,
    report: RefCell<Option<ConditionerReport>>,
    /// `evaluates_finitely` has asked for the Jacobian structure once.
    jac_structure_requested: std::cell::Cell<bool>,
    /// The last `(raw start, conditioned start, report)` triple, so a second
    /// `get_starting_point` for the *same* incoming point does not redo the
    /// work. There is a second call on the warm-start path
    /// (`OrigIpoptNlp::fetch_warm_start_snapshot`), and with
    /// `start_point_conditioner=adam` that is a whole extra 200-iteration
    /// warm-up. Both conditioners are deterministic, so this is a pure cost
    /// and never changed the answer -- but the cost is the entire feature.
    ///
    /// Keyed on the raw point rather than merely "have I run yet", because a
    /// `Problem` can be solved again from a start the caller has since
    /// changed, and answering that with the previous solve's conditioned
    /// point would be silently wrong. The comparison is on `to_bits`, so a
    /// `NaN` component matches itself -- a non-finite start is exactly the
    /// case that must hit the cache rather than re-sanitise.
    cache: RefCell<Option<(Vec<Number>, Vec<Number>, ConditionerReport)>>,
}

impl ConditionedStartTnlp {
    pub fn new(inner: Rc<RefCell<dyn TNLP>>, conditioner: StartConditioner) -> Self {
        Self {
            inner,
            conditioner,
            lower_inf: -DEFAULT_BOUND_INF,
            upper_inf: DEFAULT_BOUND_INF,
            report: RefCell::new(None),
            jac_structure_requested: std::cell::Cell::new(false),
            cache: RefCell::new(None),
        }
    }

    /// Override the sentinels past which a bound counts as absent, when the
    /// caller's `nlp_lower_bound_inf` / `nlp_upper_bound_inf` are not the
    /// defaults.
    pub fn with_bound_inf(mut self, lower: Number, upper: Number) -> Self {
        self.lower_inf = lower;
        self.upper_inf = upper;
        self
    }

    /// What the conditioner did, once `get_starting_point` has run.
    pub fn report(&self) -> Option<ConditionerReport> {
        self.report.borrow().clone()
    }

    fn bounds(
        &self,
        n: usize,
        m: usize,
    ) -> Option<(Vec<Number>, Vec<Number>, Vec<Number>, Vec<Number>)> {
        let mut x_l = vec![0.0; n];
        let mut x_u = vec![0.0; n];
        let mut g_l = vec![0.0; m];
        let mut g_u = vec![0.0; m];
        let ok = self.inner.borrow_mut().get_bounds_info(BoundsInfo {
            x_l: &mut x_l,
            x_u: &mut x_u,
            g_l: &mut g_l,
            g_u: &mut g_u,
        });
        ok.then_some((x_l, x_u, g_l, g_u))
    }

    /// Clip into the present bounds. An absent bound (at or past its
    /// sentinel) is not a clip target — clipping to `±1e19` would be a no-op
    /// numerically but would still turn "free" into "bounded" if the sentinel
    /// were ever tightened.
    fn clip(&self, v: Number, lo: Number, hi: Number) -> Number {
        let mut v = v;
        if lo > self.lower_inf && v < lo {
            v = lo;
        }
        if hi < self.upper_inf && v > hi {
            v = hi;
        }
        v
    }

    /// A finite, in-bounds stand-in for a variable whose incoming value was
    /// `NaN` or infinite.
    fn sanitised_value(&self, lo: Number, hi: Number) -> Number {
        let lo_present = lo > self.lower_inf;
        let hi_present = hi < self.upper_inf;
        match (lo_present, hi_present) {
            (true, true) => 0.5 * (lo + hi),
            (true, false) => lo + 1.0,
            (false, true) => hi - 1.0,
            (false, false) => 0.0,
        }
    }
}

/// SplitMix64. Chosen for being seedable, stateless between calls beyond one
/// `u64`, and identical on every platform — the jitter has to reproduce.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[-1, 1)` from 53 random bits.
fn unit_symmetric(state: &mut u64) -> Number {
    let bits = splitmix64(state) >> 11;
    (bits as Number) / ((1u64 << 53) as Number) * 2.0 - 1.0
}

/// Signed constraint-violation residual: how far each row is outside
/// `[g_l, g_u]`, zero inside. Reduces to `g - b` for an equality row, which
/// is the `h(x)` of the KRONOS merit.
pub fn violation(g: &[Number], g_l: &[Number], g_u: &[Number], out: &mut [Number]) {
    for i in 0..g.len() {
        let lo = g_l.get(i).copied().unwrap_or(Number::NEG_INFINITY);
        let hi = g_u.get(i).copied().unwrap_or(Number::INFINITY);
        out[i] = if g[i] < lo {
            g[i] - lo
        } else if g[i] > hi {
            g[i] - hi
        } else {
            0.0
        };
    }
}

impl ConditionedStartTnlp {
    /// Displace the start by a seeded draw, then make sure the model can be
    /// evaluated there, halving the displacement until it can (gh#1176).
    ///
    /// Staying inside the box is not enough. A model's functions can be
    /// undefined inside its own bounds -- a `log` or a fractional power of an
    /// expression in several variables, which no variable bound can protect --
    /// and a start the modeller placed carefully inside that domain is easily
    /// moved out of it by a `1e-2` relative shift. The solve then stops at
    /// iteration 0 with `Invalid_Number_Detected`, and the ladder rung this
    /// conditioner exists for is spent without testing anything. Measured on
    /// gh#1176's catalytic-distillation column: 28 constraint rows NaN at the
    /// displaced point for each of the four seeds tried, while two halvings
    /// (`2.5e-3`) evaluate cleanly and the rung runs a real re-solve.
    ///
    /// The halving keeps the *direction* of the draw, so the result is still
    /// the seeded, reproducible point for that seed, only closer; and it is a
    /// convex combination of two in-bounds points, so it stays in the box. If
    /// no halving evaluates -- typically a start sitting exactly on a domain
    /// boundary, where every point between it and the draw is outside -- the
    /// full draw is kept, which is what the conditioner did before it checked.
    ///
    /// The check is made at the conditioned point, before the algorithm's
    /// `bound_push` moves it into the interior. That push is at most
    /// `bound_push · max(1, |bound|)` toward the inside of a *box*, and a
    /// point whose box-interior neighbourhood leaves the domain is rare; this
    /// is a check against the common failure, not a proof about iteration 0.
    fn apply_jitter(
        &self,
        x: &mut [Number],
        info: &NlpInfo,
        x_l: &[Number],
        x_u: &[Number],
        seed: u64,
        scale: Number,
    ) -> ConditionerReport {
        let (mut report, base) = self.draw_jitter(x, x_l, x_u, seed, scale);
        if self.evaluates_finitely(x, info) {
            return report;
        }
        let drawn = x.to_vec();
        let mut factor = 1.0;
        for k in 1..=MAX_JITTER_HALVINGS {
            factor *= 0.5;
            for i in 0..x.len() {
                x[i] = base[i] + factor * (drawn[i] - base[i]);
            }
            if self.evaluates_finitely(x, info) {
                report.halvings = k;
                report.max_shift = x
                    .iter()
                    .zip(base.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, Number::max);
                return report;
            }
        }
        x.copy_from_slice(&drawn);
        report.unevaluable = true;
        report
    }

    /// Does the model evaluate to finite numbers at `x`? Objective, its
    /// gradient, the constraints and the Jacobian values -- the quantities
    /// iteration 0 needs. A callback that reports failure counts as not
    /// finite, which is how the solve would read it too.
    fn evaluates_finitely(&self, x: &[Number], info: &NlpInfo) -> bool {
        let n = x.len();
        let m = info.m.max(0) as usize;
        let nnz = info.nnz_jac_g.max(0) as usize;
        let mut inner = self.inner.borrow_mut();
        if !inner.eval_f(x, true).is_some_and(Number::is_finite) {
            return false;
        }
        let mut grad = vec![0.0; n];
        if !inner.eval_grad_f(x, false, &mut grad) || !grad.iter().all(|v| v.is_finite()) {
            return false;
        }
        if m == 0 {
            return true;
        }
        let mut g = vec![0.0; m];
        if !inner.eval_g(x, false, &mut g) || !g.iter().all(|v| v.is_finite()) {
            return false;
        }
        // The structure first, as the solve itself (and `apply_adam`) asks
        // for it: a TNLP may build its values layout on that call, and a
        // values request before it is outside the protocol.
        if !self.jac_structure_requested.get() {
            let (mut irow, mut jcol) = (vec![0 as Index; nnz], vec![0 as Index; nnz]);
            if !inner.eval_jac_g(
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut irow,
                    jcol: &mut jcol,
                },
            ) {
                return false;
            }
            self.jac_structure_requested.set(true);
        }
        let mut jac = vec![0.0; nnz];
        inner.eval_jac_g(Some(x), false, SparsityRequest::Values { values: &mut jac })
            && jac.iter().all(|v| v.is_finite())
    }

    /// The seeded draw itself: sanitise, displace, clip. Returns the report
    /// and the sanitised point the draw was displaced from.
    fn draw_jitter(
        &self,
        x: &mut [Number],
        x_l: &[Number],
        x_u: &[Number],
        seed: u64,
        scale: Number,
    ) -> (ConditionerReport, Vec<Number>) {
        let mut report = ConditionerReport::default();
        let mut state = seed;
        let mut bases = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let lo = x_l.get(i).copied().unwrap_or(Number::NEG_INFINITY);
            let hi = x_u.get(i).copied().unwrap_or(Number::INFINITY);
            // Sanitise first. `NaN + jitter` is `NaN`, so a start carrying a
            // `NaN` would survive the displacement untouched and the retry
            // would reproduce the original failure exactly.
            let base = if x[i].is_finite() {
                x[i]
            } else {
                report.sanitised.push(i);
                self.sanitised_value(lo, hi)
            };
            let step = scale * (1.0 + base.abs()) * unit_symmetric(&mut state);
            let moved = self.clip(base + step, lo, hi);
            report.max_shift = report.max_shift.max((moved - base).abs());
            x[i] = moved;
            bases.push(base);
        }
        (report, bases)
    }

    fn apply_adam(
        &self,
        x: &mut [Number],
        info: &NlpInfo,
        x_l: &[Number],
        x_u: &[Number],
        g_l: &[Number],
        g_u: &[Number],
        cfg: &AdamConfig,
    ) -> ConditionerReport {
        let n = x.len();
        let m = info.m.max(0) as usize;
        let nnz = info.nnz_jac_g.max(0) as usize;
        let mut report = ConditionerReport::default();

        // Sanitise before the first evaluation: Adam breaks on a non-finite
        // gradient and keeps the last good iterate, so a NaN start would make
        // the whole warm-up a no-op that reports success.
        for i in 0..n {
            if !x[i].is_finite() {
                report.sanitised.push(i);
                let lo = x_l.get(i).copied().unwrap_or(Number::NEG_INFINITY);
                let hi = x_u.get(i).copied().unwrap_or(Number::INFINITY);
                x[i] = self.sanitised_value(lo, hi);
            }
        }
        // Snapshot *after* sanitising, not before. This is the point the
        // revert below restores and the point `max_shift` measures against,
        // and both have to mean "the start as the solver will see it". Taken
        // before the loop, a warm-up that failed to reduce the merit handed
        // the un-sanitised NaN straight back to the solver while
        // `report.sanitised` said the variable had been repaired — and
        // `max_shift`, folding `(NaN - NaN).abs()`, read 0.0 and hid it.
        // `a_sanitised_start_survives_a_warm_up_that_does_not_help` is the
        // leg that lands in that branch.
        let start = x.to_vec();

        // Jacobian structure, once. Zero-based internally regardless of the
        // inner model's index style.
        let offset = match info.index_style {
            crate::tnlp::IndexStyle::C => 0usize,
            crate::tnlp::IndexStyle::Fortran => 1usize,
        };
        let (mut jrow, mut jcol) = (vec![0 as Index; nnz], vec![0 as Index; nnz]);
        if m > 0
            && !self.inner.borrow_mut().eval_jac_g(
                None,
                true,
                SparsityRequest::Structure {
                    irow: &mut jrow,
                    jcol: &mut jcol,
                },
            )
        {
            report.stopped_early = true;
            return report;
        }

        let mut grad = vec![0.0; n];
        let mut gval = vec![0.0; m];
        let mut resid = vec![0.0; m];
        let mut jval = vec![0.0; nnz];
        let mut mom = vec![0.0; n];
        let mut vel = vec![0.0; n];

        let merit = |slf: &Self, x: &[Number]| -> Number {
            let f = slf
                .inner
                .borrow_mut()
                .eval_f(x, true)
                .unwrap_or(Number::NAN);
            if m == 0 {
                return f;
            }
            let mut gv = vec![0.0; m];
            if !slf.inner.borrow_mut().eval_g(x, false, &mut gv) {
                return Number::NAN;
            }
            let mut r = vec![0.0; m];
            violation(&gv, g_l, g_u, &mut r);
            f + cfg.rho * r.iter().map(|v| v * v).sum::<Number>()
        };
        report.merit_initial = merit(self, x);

        let mut ran = 0usize;
        for t in 1..=cfg.iters {
            if !self.inner.borrow_mut().eval_grad_f(x, true, &mut grad) {
                report.stopped_early = true;
                break;
            }
            if m > 0 {
                if !self.inner.borrow_mut().eval_g(x, false, &mut gval)
                    || !self.inner.borrow_mut().eval_jac_g(
                        Some(x),
                        false,
                        SparsityRequest::Values { values: &mut jval },
                    )
                {
                    report.stopped_early = true;
                    break;
                }
                violation(&gval, g_l, g_u, &mut resid);
                // grad += 2ρ Jᵀr
                for k in 0..nnz {
                    let r = jrow[k].max(0) as usize;
                    let c = jcol[k].max(0) as usize;
                    let (Some(r), Some(c)) = (r.checked_sub(offset), c.checked_sub(offset)) else {
                        continue;
                    };
                    if r < m && c < n {
                        grad[c] += 2.0 * cfg.rho * jval[k] * resid[r];
                    }
                }
            }
            if !grad.iter().all(|v| v.is_finite()) {
                report.stopped_early = true;
                break;
            }

            let bc1 = 1.0 - cfg.beta1.powi(t as i32);
            let bc2 = 1.0 - cfg.beta2.powi(t as i32);
            let mut moved_non_finite = false;
            for i in 0..n {
                mom[i] = cfg.beta1 * mom[i] + (1.0 - cfg.beta1) * grad[i];
                vel[i] = cfg.beta2 * vel[i] + (1.0 - cfg.beta2) * grad[i] * grad[i];
                let step = cfg.lr * (mom[i] / bc1) / ((vel[i] / bc2).sqrt() + cfg.eps);
                let lo = x_l.get(i).copied().unwrap_or(Number::NEG_INFINITY);
                let hi = x_u.get(i).copied().unwrap_or(Number::INFINITY);
                let next = self.clip(x[i] - step, lo, hi);
                if !next.is_finite() {
                    moved_non_finite = true;
                    break;
                }
                x[i] = next;
            }
            if moved_non_finite {
                // Overflowed. Fall back to the last good iterate rather than
                // hand the algorithm a point worse than the one it was given.
                report.stopped_early = true;
                break;
            }
            ran = t;
        }

        report.iters = ran;
        report.merit_final = merit(self, x);
        // The warm-up is only allowed to help. If it did not reduce the merit
        // — or produced something non-finite — the original point stands, so
        // enabling the option can never be worse than the start the user gave
        // beyond the evaluations it spent.
        if !(report.merit_final < report.merit_initial) && report.merit_initial.is_finite() {
            x.copy_from_slice(&start);
            report.merit_final = report.merit_initial;
            report.iters = 0;
        }
        report.max_shift = x
            .iter()
            .zip(start.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, Number::max);
        report
    }
}

impl TNLP for ConditionedStartTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.borrow_mut().get_nlp_info()
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.borrow_mut().get_bounds_info(b)
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let Some(info) = self.inner.borrow_mut().get_nlp_info() else {
            return false;
        };
        let n = info.n.max(0) as usize;
        let m = info.m.max(0) as usize;
        let init_x = sp.init_x;
        let StartingPoint {
            x,
            z_l,
            z_u,
            lambda,
            init_z,
            init_lambda,
            ..
        } = sp;
        if !self.inner.borrow_mut().get_starting_point(StartingPoint {
            init_x,
            x,
            init_z,
            z_l,
            z_u,
            init_lambda,
            lambda,
        }) {
            return false;
        }
        // Only the primal start is conditioned; `z` and `λ` are forwarded
        // exactly as the inner model set them.
        //
        // That is a *caveat*, not an exclusion, and this comment used to read
        // as though it were the latter. Under `warm_start_init_point=yes` the
        // displacement does pair a moved primal point with multipliers
        // certified at the old one. Displacing them alongside is not
        // available — there is no meaningful "same displacement" for a dual —
        // and declining to displace instead would switch the ladder's third
        // rung off for exactly the warm-started runs, which is the wrong
        // trade: the rung only ever runs after a solve has already failed, so
        // stale duals are being compared against a verdict, not a solution.
        // The barrier's first iteration re-derives `z` from the bounds
        // anyway; `λ` is what actually carries over.
        if !init_x || n == 0 {
            return true;
        }
        let Some((x_l, x_u, g_l, g_u)) = self.bounds(n, m) else {
            return true;
        };
        // Same incoming point as last time? Then the conditioned point is the
        // same too -- see `cache`.
        if let Some((raw, conditioned, report)) = self.cache.borrow().as_ref()
            && raw.len() == n
            && raw
                .iter()
                .zip(x[..n].iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            x[..n].copy_from_slice(conditioned);
            *self.report.borrow_mut() = Some(report.clone());
            return true;
        }
        let raw = x[..n].to_vec();
        let report = match self.conditioner {
            StartConditioner::Jitter { seed, scale } => {
                self.apply_jitter(&mut x[..n], &info, &x_l, &x_u, seed, scale)
            }
            StartConditioner::Adam(cfg) => {
                self.apply_adam(&mut x[..n], &info, &x_l, &x_u, &g_l, &g_u, &cfg)
            }
        };
        *self.cache.borrow_mut() = Some((raw, x[..n].to_vec(), report.clone()));
        *self.report.borrow_mut() = Some(report);
        true
    }

    fn eval_f(&mut self, x: &[Number], new_x: bool) -> Option<Number> {
        self.inner.borrow_mut().eval_f(x, new_x)
    }

    fn eval_grad_f(&mut self, x: &[Number], new_x: bool, grad_f: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_grad_f(x, new_x, grad_f)
    }

    fn eval_g(&mut self, x: &[Number], new_x: bool, g: &mut [Number]) -> bool {
        self.inner.borrow_mut().eval_g(x, new_x, g)
    }

    fn eval_jac_g(&mut self, x: Option<&[Number]>, new_x: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.borrow_mut().eval_jac_g(x, new_x, mode)
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner
            .borrow_mut()
            .eval_h(x, new_x, obj_factor, lambda, new_lambda, mode)
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, ip_data: &IpoptData, ip_cq: &IpoptCq) {
        self.inner
            .borrow_mut()
            .finalize_solution(sol, ip_data, ip_cq)
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        self.inner.borrow_mut().get_var_con_metadata(var, con)
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        self.inner.borrow_mut().get_scaling_parameters(req)
    }

    fn get_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_variables_linearity(types)
    }

    fn get_objective_variables_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner
            .borrow_mut()
            .get_objective_variables_linearity(types)
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        self.inner.borrow_mut().get_constraints_linearity(types)
    }

    fn get_number_of_nonlinear_variables(&mut self) -> Index {
        self.inner.borrow_mut().get_number_of_nonlinear_variables()
    }

    fn get_list_of_nonlinear_variables(&mut self, pos: &mut [Index]) -> bool {
        self.inner.borrow_mut().get_list_of_nonlinear_variables(pos)
    }

    fn derivative_proofs(&mut self) -> crate::constant_derivatives::DerivativeProofs {
        // The conditioner changes only where the solve starts, never what the
        // derivatives *are*, so both directions of every proof survive it.
        self.inner.borrow_mut().derivative_proofs()
    }

    fn wants_intermediate_callback(&self) -> bool {
        self.inner.borrow().wants_intermediate_callback()
    }

    fn wants_current_iterate(&self) -> bool {
        self.inner.borrow().wants_current_iterate()
    }

    fn update_with_barrier(&mut self, mu: Number) -> bool {
        self.inner.borrow_mut().update_with_barrier(mu)
    }

    fn intermediate_callback(
        &mut self,
        stats: IterStats,
        ip_data: &IpoptData,
        ip_cq: &IpoptCq,
    ) -> bool {
        self.inner
            .borrow_mut()
            .intermediate_callback(stats, ip_data, ip_cq)
    }

    fn finalize_metadata(&mut self, var: &MetaData, con: &MetaData) {
        self.inner.borrow_mut().finalize_metadata(var, con)
    }

    fn is_presolve_wrapper(&self) -> bool {
        self.inner.borrow().is_presolve_wrapper()
    }

    fn scaling_factors(&self) -> Option<Vec<Number>> {
        self.inner.borrow().scaling_factors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tnlp::IndexStyle;

    /// `min ½‖x − t‖²  s.t.  Σx ∈ [g_l, g_u]`, with a settable start and
    /// bounds so each test can construct exactly the degeneracy it is about.
    struct Toy {
        n: usize,
        start: Vec<Number>,
        x_l: Vec<Number>,
        x_u: Vec<Number>,
        target: Vec<Number>,
        g_l: Number,
        g_u: Number,
        with_constraint: bool,
        init_x: bool,
        /// Count of `eval_f` calls, to show the decorator forwards.
        f_calls: usize,
        /// The model is undefined (evaluates `NaN`) wherever some `x_i` is
        /// below this, the way `log` of an expression is -- with no variable
        /// bound saying so (gh#1176).
        nan_below: Option<Number>,
        /// The Jacobian structure has been asked for. A values request before
        /// it fails, as it would on a TNLP that builds its layout on the
        /// structure call -- the protocol the solve follows.
        jac_structure_seen: bool,
    }

    impl Toy {
        fn new(n: usize) -> Self {
            Self {
                n,
                start: vec![0.0; n],
                x_l: vec![-DEFAULT_BOUND_INF; n],
                x_u: vec![DEFAULT_BOUND_INF; n],
                target: vec![0.0; n],
                g_l: 0.0,
                g_u: 0.0,
                with_constraint: false,
                init_x: true,
                f_calls: 0,
                nan_below: None,
                jac_structure_seen: false,
            }
        }
        fn m(&self) -> usize {
            usize::from(self.with_constraint)
        }
        fn with_start(mut self, v: Number) -> Self {
            self.start = vec![v; self.n];
            self
        }
    }

    impl TNLP for Toy {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: self.n as Index,
                m: self.m() as Index,
                nnz_jac_g: (self.m() * self.n) as Index,
                nnz_h_lag: self.n as Index,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&self.x_l);
            b.x_u.copy_from_slice(&self.x_u);
            if self.with_constraint {
                b.g_l[0] = self.g_l;
                b.g_u[0] = self.g_u;
            }
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            if self.init_x {
                sp.x.copy_from_slice(&self.start);
            }
            true
        }
        fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
            self.f_calls += 1;
            if let Some(t) = self.nan_below
                && x.iter().any(|v| *v < t)
            {
                return Some(Number::NAN);
            }
            Some(
                0.5 * x
                    .iter()
                    .zip(&self.target)
                    .map(|(a, t)| (a - t) * (a - t))
                    .sum::<Number>(),
            )
        }
        fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            for i in 0..self.n {
                g[i] = x[i] - self.target[i];
            }
            true
        }
        fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
            if self.with_constraint {
                g[0] = x.iter().sum();
            }
            true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            if !self.with_constraint {
                return true;
            }
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    for j in 0..self.n {
                        irow[j] = 0;
                        jcol[j] = j as Index;
                    }
                    self.jac_structure_seen = true;
                }
                SparsityRequest::Values { values } => {
                    if !self.jac_structure_seen {
                        return false;
                    }
                    values.fill(1.0);
                }
            }
            true
        }
        fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _n: bool,
            _o: Number,
            _l: Option<&[Number]>,
            _nl: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    for j in 0..self.n {
                        irow[j] = j as Index;
                        jcol[j] = j as Index;
                    }
                }
                SparsityRequest::Values { values } => values.fill(1.0),
            }
            true
        }
    }

    /// Drive `get_starting_point` on a decorated model and hand back the
    /// primal start it produced.
    fn conditioned(toy: Toy, c: StartConditioner) -> (Vec<Number>, ConditionerReport) {
        let n = toy.n;
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(toy));
        let mut wrapped = ConditionedStartTnlp::new(inner, c);
        let (mut x, mut z_l, mut z_u, mut lam) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; 1]);
        assert!(wrapped.get_starting_point(StartingPoint {
            init_x: true,
            x: &mut x,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: false,
            lambda: &mut lam,
        }));
        (x, wrapped.report().unwrap_or_default())
    }

    fn jitter(seed: u64) -> StartConditioner {
        StartConditioner::Jitter { seed, scale: 1e-2 }
    }

    // ---- jitter -----------------------------------------------------------

    #[test]
    fn the_same_seed_produces_the_same_point() {
        let (a, _) = conditioned(Toy::new(5), jitter(7));
        let (b, _) = conditioned(Toy::new(5), jitter(7));
        assert_eq!(a, b, "a retry rung has to be reproducible");
    }

    #[test]
    fn a_different_seed_produces_a_different_point() {
        let (a, _) = conditioned(Toy::new(5), jitter(7));
        let (b, _) = conditioned(Toy::new(5), jitter(8));
        assert_ne!(a, b);
    }

    #[test]
    fn a_start_at_the_origin_actually_moves() {
        // The whole point of the `1 + |x|` factor. A purely relative
        // displacement is identically zero at x = 0, which is the single most
        // common degenerate start in the corpus.
        let (x, report) = conditioned(Toy::new(4), jitter(1));
        assert!(x.iter().all(|v| *v != 0.0), "no coordinate moved: {x:?}");
        assert!(report.max_shift > 0.0);
        assert!(
            report.max_shift <= 1e-2,
            "shift {} out of scale",
            report.max_shift
        );
    }

    /// A start one unit inside a domain the model does not declare as a bound
    /// (`f` is `NaN` wherever any `x_i < 0.999`), displaced at `scale = 1e-2`.
    fn toy_near_a_domain_edge(n: usize) -> Toy {
        let mut toy = Toy::new(n);
        toy.start = vec![1.0; n];
        toy.nan_below = Some(0.999);
        toy
    }

    #[test]
    fn a_draw_the_model_cannot_evaluate_is_halved_until_it_can() {
        // gh#1176: the ladder's perturbed-start rung moved a carefully placed
        // start out of the model's domain and died at iteration 0 with
        // Invalid_Number_Detected, testing nothing.
        let seed = 1;
        // The draw on its own leaves the domain -- otherwise this test is
        // about nothing. A model with no domain edge gets the full draw.
        let (full, _) = conditioned(Toy::new(8).with_start(1.0), jitter(seed));
        assert!(
            full.iter().any(|v| *v < 0.999),
            "seed {seed} never leaves the domain: {full:?}"
        );

        let (x, report) = conditioned(toy_near_a_domain_edge(8), jitter(seed));
        assert!(x.iter().all(|v| *v >= 0.999), "{x:?}");
        assert!(report.halvings >= 1, "{report:?}");
        assert!(!report.unevaluable);
        // Same direction, shorter: each coordinate is the draw scaled toward
        // the start by exactly 2^-halvings.
        let f = 0.5f64.powi(report.halvings as i32);
        for (a, b) in x.iter().zip(&full) {
            assert!((a - (1.0 + f * (b - 1.0))).abs() <= 1e-15, "{a} vs {b}");
        }
        assert!(x.iter().any(|v| *v != 1.0), "the start did not move at all");
    }

    #[test]
    fn the_evaluability_check_asks_for_the_jacobian_structure_first() {
        // gh#1176 review item 2: the check requested Jacobian values on a
        // TNLP that had never been asked for its structure. A model that
        // answers values only after the structure call then read as
        // unevaluable at every point, and the draw was never halved. The
        // halving test above has no constraints, so it cannot see this.
        let mut toy = toy_near_a_domain_edge(8);
        toy.with_constraint = true;
        toy.g_l = -1e3;
        toy.g_u = 1e3;
        let (x, report) = conditioned(toy, jitter(1));
        assert!(!report.unevaluable, "{report:?}");
        assert!(report.halvings >= 1, "{report:?}");
        assert!(x.iter().all(|v| *v >= 0.999), "{x:?}");
    }

    #[test]
    fn a_draw_the_model_can_evaluate_is_kept_bit_for_bit() {
        // The check must not move a draw that was fine: on every model whose
        // functions are defined on the whole box the rung is unchanged.
        let (plain, _) = conditioned(Toy::new(8).with_start(1.0), jitter(1));
        let mut toy = Toy::new(8).with_start(1.0);
        toy.nan_below = Some(-1.0); // a domain edge the draw never reaches
        let (checked, report) = conditioned(toy, jitter(1));
        assert_eq!(plain, checked);
        assert_eq!(report.halvings, 0);
        assert!(!report.unevaluable);
    }

    #[test]
    fn a_start_on_the_domain_edge_keeps_the_full_draw() {
        // Every point between a start exactly on the edge and a draw outside
        // it is outside too, so no halving helps; the conditioner then does
        // what it did before it looked, rather than collapsing to the start.
        let mut toy = Toy::new(8).with_start(0.0);
        toy.nan_below = Some(0.0);
        let (x, report) = conditioned(toy, jitter(1));
        let (full, _) = conditioned(Toy::new(8).with_start(0.0), jitter(1));
        assert_eq!(x, full);
        assert!(report.unevaluable);
        assert_eq!(report.halvings, 0);
    }

    #[test]
    fn a_non_finite_start_is_sanitised_before_it_is_displaced() {
        // `NaN + jitter` is `NaN`. Without the sanitise the retry reproduces
        // the original Invalid_Number_Detected exactly.
        let mut toy = Toy::new(4);
        toy.start = vec![Number::NAN, 1.0, Number::INFINITY, -2.0];
        let (x, report) = conditioned(toy, jitter(3));
        assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
        assert_eq!(report.sanitised, vec![0, 2]);
    }

    #[test]
    fn a_sanitised_variable_lands_inside_its_bounds() {
        let mut toy = Toy::new(3);
        toy.start = vec![Number::NAN; 3];
        toy.x_l = vec![2.0, 5.0, -DEFAULT_BOUND_INF];
        toy.x_u = vec![4.0, DEFAULT_BOUND_INF, -1.0];
        let (x, _) = conditioned(toy, jitter(11));
        assert!((2.0..=4.0).contains(&x[0]), "{x:?}"); // midpoint of a box
        assert!(x[1] >= 5.0, "{x:?}"); // lower + 1, then jitter, clipped
        assert!(x[2] <= -1.0, "{x:?}"); // upper − 1
    }

    #[test]
    fn the_displacement_never_leaves_the_box() {
        let mut toy = Toy::new(3);
        toy.start = vec![1.0, 1.0, 1.0];
        toy.x_l = vec![1.0, 1.0, 1.0];
        toy.x_u = vec![1.0, 1.0, 1.0]; // pinned: every draw must clip back
        let (x, report) = conditioned(
            toy,
            StartConditioner::Jitter {
                seed: 5,
                scale: 10.0,
            },
        );
        assert_eq!(x, vec![1.0, 1.0, 1.0]);
        assert_eq!(report.max_shift, 0.0);
    }

    #[test]
    fn an_absent_bound_is_not_a_clip_target() {
        // A free variable stays free. Clipping to the ±1e19 sentinel would be
        // a numerical no-op today and a silent bound the day the sentinel moves.
        let mut toy = Toy::new(2);
        toy.start = vec![1e30, -1e30];
        let (x, _) = conditioned(toy, jitter(2));
        assert!(x[0] > 1e29 && x[1] < -1e29, "{x:?}");
    }

    #[test]
    fn a_model_that_declines_to_set_a_start_is_left_alone() {
        let mut toy = Toy::new(3);
        toy.init_x = false;
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(toy));
        let mut wrapped = ConditionedStartTnlp::new(inner, jitter(4));
        let (mut x, mut z_l, mut z_u, mut lam) =
            (vec![9.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 1]);
        assert!(wrapped.get_starting_point(StartingPoint {
            init_x: false,
            x: &mut x,
            init_z: false,
            z_l: &mut z_l,
            z_u: &mut z_u,
            init_lambda: false,
            lambda: &mut lam,
        }));
        assert_eq!(
            x,
            vec![9.0; 3],
            "no init_x means there is no start to condition"
        );
        assert!(wrapped.report().is_none());
    }

    // ---- violation --------------------------------------------------------

    #[test]
    fn violation_is_zero_inside_the_band_and_signed_outside() {
        let g = [1.0, -3.0, 7.0, 4.0];
        let g_l = [0.0, 0.0, 0.0, 4.0];
        let g_u = [2.0, 2.0, 2.0, 4.0];
        let mut out = [0.0; 4];
        violation(&g, &g_l, &g_u, &mut out);
        assert_eq!(out, [0.0, -3.0, 5.0, 0.0]);
    }

    #[test]
    fn violation_reduces_to_the_equality_residual() {
        // KRONOS's merit is ρ‖h(x)‖² over equalities only; a two-sided row
        // with g_l == g_u has to give back exactly `g − b` for the
        // generalisation to be one.
        let g = [3.0, -1.0];
        let b = [1.0, 1.0];
        let mut out = [0.0; 2];
        violation(&g, &b, &b, &mut out);
        assert_eq!(out, [2.0, -2.0]);
    }

    // ---- adam -------------------------------------------------------------

    fn adam(iters: usize) -> StartConditioner {
        StartConditioner::Adam(AdamConfig {
            iters,
            ..Default::default()
        })
    }

    /// A second `get_starting_point` for the same point must not redo the
    /// warm-up. The warm-start path makes exactly that second call
    /// (`OrigIpoptNlp::fetch_warm_start_snapshot`), so before the cache
    /// `warm_start_init_point=yes` + `start_point_conditioner=adam` paid for
    /// two full warm-ups and used one.
    #[test]
    fn the_warm_up_does_not_run_twice_for_the_same_start() {
        let toy = Rc::new(RefCell::new(Toy::new(3)));
        toy.borrow_mut().target = vec![1.0, -2.0, 0.5];
        toy.borrow_mut().start = vec![5.0, 5.0, 5.0];
        let inner: Rc<RefCell<dyn TNLP>> = toy.clone();
        let mut wrapped = ConditionedStartTnlp::new(inner, adam(200));

        let mut ask = |w: &mut ConditionedStartTnlp| {
            let (mut x, mut z_l, mut z_u, mut lam) =
                (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0.0; 1]);
            assert!(w.get_starting_point(StartingPoint {
                init_x: true,
                x: &mut x,
                init_z: false,
                z_l: &mut z_l,
                z_u: &mut z_u,
                init_lambda: false,
                lambda: &mut lam,
            }));
            x
        };

        let first = ask(&mut wrapped);
        let after_first = toy.borrow().f_calls;
        assert!(after_first > 0, "the warm-up should have evaluated f");

        let second = ask(&mut wrapped);
        assert_eq!(
            toy.borrow().f_calls,
            after_first,
            "the second call re-ran the warm-up",
        );
        assert_eq!(
            first, second,
            "and it must still answer with the same point"
        );
        assert!(
            wrapped.report().is_some(),
            "the report survives the cache hit"
        );

        // But a start the caller has since *changed* is a different question,
        // and answering it from the cache would be silently wrong.
        toy.borrow_mut().start = vec![-4.0, 0.0, 7.0];
        let third = ask(&mut wrapped);
        assert!(
            toy.borrow().f_calls > after_first,
            "a changed start must re-run the warm-up",
        );
        assert_ne!(third, first);
    }

    #[test]
    fn the_warm_up_walks_downhill_on_an_unconstrained_bowl() {
        let mut toy = Toy::new(3);
        toy.target = vec![1.0, -2.0, 0.5];
        toy.start = vec![5.0, 5.0, 5.0];
        let (x, report) = conditioned(toy, adam(200));
        assert!(report.merit_final < report.merit_initial, "{report:?}");
        assert_eq!(report.iters, 200);
        // Adam's step is size-capped near `lr` regardless of the gradient, so
        // 200 iterations buy about `200·lr = 10` units of travel and the last
        // stretch is slow. Each coordinate here has 4.5-7 units to cover, so
        // the right assertion is "most of the way", not "converged".
        let targets: [Number; 3] = [1.0, -2.0, 0.5];
        for (i, (got, want)) in x.iter().zip(targets).enumerate() {
            let before = (5.0 - want).abs();
            let after = (got - want).abs();
            assert!(
                after < 0.15 * before,
                "x[{i}] went {before} -> {after} ({x:?})"
            );
        }
    }

    #[test]
    fn the_penalty_pulls_a_violated_constraint_back_toward_feasibility() {
        let mut toy = Toy::new(2);
        toy.with_constraint = true;
        toy.g_l = 1.0;
        toy.g_u = 1.0; // x0 + x1 = 1
        toy.target = vec![10.0, 10.0]; // objective alone wants (10, 10)
        toy.start = vec![10.0, 10.0];
        let (x, report) = conditioned(toy, adam(200));
        let before = 20.0 - 1.0;
        let after = (x[0] + x[1] - 1.0).abs();
        assert!(after < before, "sum went {before} -> {after} ({x:?})");
        assert!(report.merit_final < report.merit_initial);
    }

    #[test]
    fn a_warm_up_that_does_not_help_returns_the_original_point() {
        // Starting at the minimum, Adam's first bias-corrected step is a full
        // `lr` regardless of how small the gradient is, so it walks away. The
        // guard is what makes turning the option on safe.
        let mut toy = Toy::new(3);
        toy.target = vec![1.0, 2.0, 3.0];
        toy.start = vec![1.0, 2.0, 3.0];
        let (x, report) = conditioned(toy, adam(50));
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
        assert_eq!(report.iters, 0);
        assert_eq!(report.merit_final, report.merit_initial);
        assert_eq!(report.max_shift, 0.0);
    }

    /// The other branch of the revert guard, and the one that was broken.
    ///
    /// `the_warm_up_sanitises_a_non_finite_start_too` picks a start where
    /// Adam *does* reduce the merit, so it never reaches the revert;
    /// `a_warm_up_that_does_not_help_returns_the_original_point` reverts but
    /// from a finite start. Sanitise-and-revert together was covered by
    /// neither, which is the branch-coverage trap CLAUDE.md describes for the
    /// invariance legs. Here the start is both non-finite and already at the
    /// target, so the warm-up sanitises, fails to help, and reverts — and
    /// what it reverts to must be the *sanitised* point, because that is what
    /// the report claims and what the solver is about to evaluate.
    #[test]
    fn a_sanitised_start_survives_a_warm_up_that_does_not_help() {
        let mut toy = Toy::new(3);
        toy.target = vec![0.0, 2.0, 3.0];
        toy.start = vec![Number::NAN, 2.0, 3.0];
        let (x, report) = conditioned(toy, adam(50));
        assert_eq!(report.sanitised, vec![0]);
        assert!(
            x.iter().all(|v| v.is_finite()),
            "the solver must not receive the un-sanitised start back: {x:?}",
        );
        assert_eq!(&x[1..], &[2.0, 3.0]);
        // `max_shift` folds against the same snapshot, so a NaN there used to
        // read as a shift of exactly 0.0 and corroborate the wrong answer.
        assert!(report.max_shift.is_finite(), "{}", report.max_shift);
    }

    #[test]
    fn a_zero_iteration_budget_is_a_no_op() {
        let mut toy = Toy::new(3);
        toy.start = vec![4.0, 5.0, 6.0];
        let (x, report) = conditioned(toy, adam(0));
        assert_eq!(x, vec![4.0, 5.0, 6.0]);
        assert_eq!(report.iters, 0);
    }

    #[test]
    fn the_warm_up_sanitises_a_non_finite_start_too() {
        // KRONOS's own stage 0 breaks on the first non-finite gradient and
        // keeps the last good iterate, so on a NaN start it hands back the
        // NaN. Sanitising first is what lets the warm-up rescue those.
        let mut toy = Toy::new(3);
        toy.target = vec![1.0, 1.0, 1.0];
        toy.start = vec![Number::NAN, 5.0, 5.0];
        let (x, report) = conditioned(toy, adam(100));
        assert_eq!(report.sanitised, vec![0]);
        assert!(x.iter().all(|v| v.is_finite()), "{x:?}");
    }

    #[test]
    fn the_warm_up_respects_variable_bounds() {
        let mut toy = Toy::new(2);
        toy.target = vec![-100.0, 100.0];
        toy.start = vec![0.0, 0.0];
        toy.x_l = vec![-1.0, -1.0];
        toy.x_u = vec![1.0, 1.0];
        let (x, _) = conditioned(toy, adam(200));
        assert!(x.iter().all(|v| (-1.0..=1.0).contains(v)), "{x:?}");
    }

    // ---- forwarding -------------------------------------------------------

    #[test]
    fn every_other_callback_is_forwarded_untouched() {
        let mut toy = Toy::new(2);
        toy.with_constraint = true;
        toy.target = vec![1.0, 1.0];
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(toy));
        let mut wrapped = ConditionedStartTnlp::new(Rc::clone(&inner), jitter(1));

        let info = wrapped.get_nlp_info().unwrap();
        assert_eq!((info.n, info.m, info.nnz_jac_g), (2, 1, 2));

        let x = [3.0, 4.0];
        assert_eq!(wrapped.eval_f(&x, true), Some(0.5 * (4.0 + 9.0)));
        let mut grad = [0.0; 2];
        assert!(wrapped.eval_grad_f(&x, true, &mut grad));
        assert_eq!(grad, [2.0, 3.0]);
        let mut g = [0.0; 1];
        assert!(wrapped.eval_g(&x, true, &mut g));
        assert_eq!(g, [7.0]);
        let (mut irow, mut jcol) = ([0 as Index; 2], [0 as Index; 2]);
        assert!(wrapped.eval_jac_g(
            None,
            true,
            SparsityRequest::Structure {
                irow: &mut irow,
                jcol: &mut jcol,
            }
        ));
        assert_eq!((irow, jcol), ([0, 0], [0, 1]));
        let mut vals = [0.0; 2];
        assert!(wrapped.eval_jac_g(
            Some(&x),
            true,
            SparsityRequest::Values { values: &mut vals }
        ));
        assert_eq!(vals, [1.0, 1.0]);

        let mut b = ([0.0; 2], [0.0; 2], [0.0; 1], [0.0; 1]);
        assert!(wrapped.get_bounds_info(BoundsInfo {
            x_l: &mut b.0,
            x_u: &mut b.1,
            g_l: &mut b.2,
            g_u: &mut b.3,
        }));
        assert_eq!(b.0, [-DEFAULT_BOUND_INF; 2]);
    }
}
