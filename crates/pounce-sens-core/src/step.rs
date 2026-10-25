//! Parametric steps, and what they do about the bounds.
//!
//! The step is `dx ≈ (dx*/dp) dp` against the held factor: how the solution
//! moves when the parameters do. [`solution`] returns the moved point,
//! [`solution_report`] measures what the step did about the bounds on the
//! way, and [`active_set_changes`] lists the active-set events a path step
//! applies. Parameters are addressed by the model row of the defining
//! equality each is pinned by.
//!
//! Everything here runs over any [`BarrierIterate`] through
//! [`ConvergedSolve`], and measures the moved point with the model's own
//! functions through [`ModelFunctions`]. What a caller should be told comes
//! back as [`StepFinding`]s beside the answer.

use pounce_common::types::{Index, Number};

use crate::activity::{AMBIGUOUS, STRONGLY_ACTIVE, WEAKLY_ACTIVE};
use crate::barrier::BarrierIterate;
use crate::boundcheck::{PathSegment, RefineStop};
use crate::converged::{ConvergedSolve, SolverError};
use crate::corrector::CorrectorReport;
use crate::model::ModelFunctions;

/// What happens when the step leaves a variable bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Take the step and clamp.
    Linear,
    /// Pin each crossing variable at its bound and re-solve.
    FixRelax,
    /// Apply the perturbation a little at a time, changing the active set
    /// where it changes.
    Path,
}

/// What happens when the base point sits at an active-set kink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Degeneracy {
    /// Decide the weakly active bounds by the directional-derivative QP
    /// for the perturbation's own direction.
    Directional,
    /// Take the one-sided value the held factorization leans toward.
    OneSided,
    /// Release every weakly active bound undecided.
    ReleaseAll,
}

/// The options every step shares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOptions {
    /// See [`Mode`].
    pub mode: Mode,
    /// Under [`Mode::FixRelax`] the cap on refinement passes, and under
    /// [`Mode::Path`] the cap on active-set changes.
    pub predictor_iter: usize,
    /// See [`Degeneracy`].
    pub degeneracy: Degeneracy,
    /// The back-solve budget of the directional decision. `None` is 16.
    pub degeneracy_iter: Option<usize>,
    /// Newton iterations on the barrier system after the step.
    pub corrector_iter: usize,
    /// How far outside a bound a step has to end to count as having left
    /// it, under [`Mode::FixRelax`]. `None` is the solve's own margin.
    pub bound_eps: Option<Number>,
    /// Refuse when the held factor carries a larger inertia correction.
    pub max_pdpert: Option<Number>,
}

impl Default for StepOptions {
    fn default() -> Self {
        Self {
            mode: Mode::Linear,
            predictor_iter: 16,
            degeneracy: Degeneracy::Directional,
            degeneracy_iter: None,
            corrector_iter: 0,
            bound_eps: None,
            max_pdpert: None,
        }
    }
}

/// What a caller should be told about a step, beside the answer.
#[derive(Debug, Clone, PartialEq)]
pub enum StepFinding {
    /// A directional budget was given under another degeneracy treatment,
    /// which makes no decision for it to budget.
    DegeneracyIterInert,
    /// A bound margin was given under a mode that runs no refinement.
    BoundEpsInert,
    /// The directional decision failed, and the one-sided step was taken.
    DirectionalFallback {
        /// Why the decision failed.
        error: SolverError,
    },
    /// The corrector did not halve the residual.
    CorrectorWeak {
        /// What its iterations did.
        report: CorrectorReport,
    },
    /// Under [`Mode::FixRelax`] or [`Mode::Path`] the step still leaves the
    /// bounds of these model variables.
    StillOutside {
        /// Full-x indices.
        vars: Vec<usize>,
        /// Under [`Mode::FixRelax`], the variables pinned and why the
        /// refinement stopped.
        pinned: Option<(usize, RefineStop)>,
        /// Under [`Mode::Path`], the active-set changes applied.
        changes: Option<usize>,
        /// Whether the values were clamped.
        clamped: bool,
    },
    /// A linear step left the bounds of these model variables, and the
    /// values were clamped.
    LinearClamped {
        /// Full-x indices.
        vars: Vec<usize>,
    },
}

/// Why a step has no answer.
#[derive(Debug, Clone)]
pub enum StepError {
    /// A calculation over the factor failed.
    Solver(SolverError),
    /// A margin was not a positive number.
    NotPositive {
        /// Which margin.
        name: &'static str,
        /// The value given.
        value: Number,
    },
    /// The held factor carries a larger inertia correction than the cap.
    Perturbed {
        /// The largest entry, in magnitude.
        worst: Number,
        /// `(δ_x, δ_s, δ_c, δ_d)`.
        perturbations: [Number; 4],
        /// The cap.
        limit: Number,
    },
    /// Evaluating the model failed.
    Model(String),
}

impl From<SolverError> for StepError {
    fn from(e: SolverError) -> Self {
        StepError::Solver(e)
    }
}

/// An answer, and everything the caller should be told about it.
#[derive(Debug)]
pub struct StepOutcome<T> {
    /// What the step met, in order.
    pub findings: Vec<StepFinding>,
    /// The answer.
    pub result: Result<T, StepError>,
}

/// Whether a factor's inertia correction is over `limit`, and its largest
/// entry in magnitude. The comparison `sens_max_pdpert` reads too, so the
/// two surfaces agree on what counts as too perturbed.
pub fn pdpert_verdict(perturbations: &[Number; 4], limit: Number) -> (bool, Number) {
    let worst = perturbations
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    (worst > limit, worst)
}

/// Refuses a margin that is not a positive number. Zero reinstates the
/// roundoff pinning the floor prevents, NaN makes every "outside a bound"
/// comparison false, and a negative cap on the inertia correction always
/// refuses.
pub fn check_margins(
    bound_eps: Option<Number>,
    max_pdpert: Option<Number>,
) -> Result<(), StepError> {
    for (name, value) in [("bound_eps", bound_eps), ("max_pdpert", max_pdpert)] {
        if let Some(v) = value
            && !(v > 0.0)
        {
            return Err(StepError::NotPositive { name, value: v });
        }
    }
    Ok(())
}

/// Refuses when the factor carries more inertia correction than
/// `max_pdpert`. An engine that does not report its correction is
/// refused too: the cap cannot be checked against a correction nobody
/// knows, and passing it would answer "within the cap" for a factor that
/// may have been regularized by any amount.
pub fn refuse_on_pdpert<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    max_pdpert: Option<Number>,
) -> Result<(), StepError> {
    let Some(limit) = max_pdpert else {
        return Ok(());
    };
    let Some(p) = conv.kkt_perturbations() else {
        return Err(StepError::Solver(SolverError::BadOptions(format!(
            "max_pdpert = {limit:e} needs the factor's inertia correction, \
             and this engine does not report it"
        ))));
    };
    let (refuse, worst) = pdpert_verdict(&p, limit);
    if refuse {
        return Err(StepError::Perturbed {
            worst,
            perturbations: p,
            limit,
        });
    }
    Ok(())
}

/// What a step needs of the model's variables: each one's factor row, and
/// back.
struct Maps {
    to_var: Vec<Option<usize>>,
    /// The model variable of each factor row of the `x` block.
    to_full: Vec<Option<usize>>,
}

impl Maps {
    fn new<E: BarrierIterate + ?Sized>(conv: &ConvergedSolve<'_, E>) -> Result<Self, SolverError> {
        let n_full = conv.n_full_x()?;
        let rows: Vec<Index> = (0..n_full as Index).collect();
        let to_var: Vec<Option<usize>> = conv
            .x_primal_rows(&rows)?
            .into_iter()
            .map(|r| r.map(|r| r as usize))
            .collect();
        let mut to_full = vec![None; conv.engine().block_dims()[0]];
        for (full, r) in to_var.iter().enumerate() {
            if let Some(slot) = r.and_then(|r| to_full.get_mut(r)) {
                *slot = Some(full);
            }
        }
        Ok(Self { to_var, to_full })
    }

    /// Returns `BadShape` when a vector over the model's variables has
    /// another length. Unchecked, it would be read past its end or paired
    /// with the wrong entries.
    fn check_full(&self, what: &'static str, got: usize) -> Result<(), SolverError> {
        if got == self.to_var.len() {
            Ok(())
        } else {
            Err(SolverError::BadShape {
                what,
                got,
                expected: self.to_var.len(),
            })
        }
    }

    /// A model-space vector from a factor-space (var-x) one: variables the
    /// engine removed get zero, since a fixed variable does not move.
    fn scatter_x(&self, dx_var: &[Number]) -> Vec<Number> {
        self.to_var
            .iter()
            .map(|r| r.map_or(0.0, |r| dx_var[r]))
            .collect()
    }

    /// The model variable of a factor row of the `x` block.
    fn full_of(&self, var_row: usize) -> Option<usize> {
        self.to_full.get(var_row).copied().flatten()
    }
}

/// Returns `BadShape` when the base point or the variable bounds are not
/// over the model's variables, or `deltas` is not one per pin. Unchecked,
/// they would be read past their ends or paired with the wrong entries.
fn check_inputs(
    maps: &Maps,
    model: &dyn ModelFunctions,
    base_x: &[Number],
    pins: &[Index],
    deltas: &[Number],
) -> Result<(), SolverError> {
    maps.check_full("base point", base_x.len())?;
    let (lo, hi) = model.variable_bounds();
    maps.check_full("variable lower bounds", lo.len())?;
    maps.check_full("variable upper bounds", hi.len())?;
    if deltas.len() != pins.len() {
        return Err(SolverError::BadShape {
            what: "perturbations",
            got: deltas.len(),
            expected: pins.len(),
        });
    }
    Ok(())
}

/// The step a mode and a degeneracy treatment take, and what came with it.
struct Dispatched {
    /// The primal step over the factor's `x` block.
    step: Vec<Number>,
    /// Under [`Mode::FixRelax`], the rows pinned and why it stopped.
    refined: Option<(Vec<Index>, RefineStop)>,
    /// Under [`Mode::Path`], the breakpoints crossed.
    segments: Vec<PathSegment>,
}

fn directional_failed(e: &SolverError) -> bool {
    matches!(e, SolverError::SensComputationFailed(m) if m.contains("directional derivative"))
}

fn dispatch<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    budget: usize,
    findings: &mut Vec<StepFinding>,
) -> Result<Dispatched, SolverError> {
    let mut out = Dispatched {
        step: Vec::new(),
        refined: None,
        segments: Vec::new(),
    };
    let mut fell_back = false;
    if o.degeneracy == Degeneracy::Directional {
        // At a degenerate base point the weakly active bounds are decided by
        // the directional-derivative QP for this perturbation's direction.
        // At a clean base point these are the plain calls. A failed decision
        // falls back to the one-sided step and says so.
        let attempt = (|| -> Result<Dispatched, SolverError> {
            let (step, held, _) = conv.parametric_step_directional(pins, deltas, budget)?;
            let held: Vec<Index> = held.iter().map(|&r| r as Index).collect();
            Ok(match o.mode {
                Mode::FixRelax => {
                    let (step, pinned, stop) = conv.parametric_step_bounded_decided(
                        pins,
                        deltas,
                        o.predictor_iter,
                        &held,
                        o.bound_eps,
                    )?;
                    Dispatched {
                        step,
                        refined: Some((pinned, stop)),
                        segments: Vec::new(),
                    }
                }
                Mode::Path => {
                    let (step, segments) =
                        conv.parametric_step_path_decided(pins, deltas, o.predictor_iter, &held)?;
                    Dispatched {
                        step,
                        refined: None,
                        segments,
                    }
                }
                Mode::Linear => Dispatched {
                    step,
                    refined: None,
                    segments: Vec::new(),
                },
            })
        })();
        match attempt {
            Ok(d) => out = d,
            Err(e) if directional_failed(&e) => {
                findings.push(StepFinding::DirectionalFallback { error: e });
                fell_back = true;
            }
            Err(e) => return Err(e),
        }
    }
    if o.degeneracy == Degeneracy::ReleaseAll {
        // Every weakly active bound is released undecided, at one back-solve
        // and no QP: the bounds this perturbation holds come back as
        // crossings for the mode or a correction to handle.
        out = match o.mode {
            Mode::FixRelax => {
                let (step, pinned, stop) = conv.parametric_step_bounded_decided(
                    pins,
                    deltas,
                    o.predictor_iter,
                    &[],
                    o.bound_eps,
                )?;
                Dispatched {
                    step,
                    refined: Some((pinned, stop)),
                    segments: Vec::new(),
                }
            }
            Mode::Path => {
                let (step, segments) =
                    conv.parametric_step_path_decided(pins, deltas, o.predictor_iter, &[])?;
                Dispatched {
                    step,
                    refined: None,
                    segments,
                }
            }
            Mode::Linear => Dispatched {
                step: conv.parametric_step_release_all(pins, deltas)?.0,
                refined: None,
                segments: Vec::new(),
            },
        };
    }
    if o.degeneracy == Degeneracy::OneSided || fell_back {
        out = match o.mode {
            Mode::FixRelax => {
                let (step, pinned, stop) =
                    conv.parametric_step_bounded(pins, deltas, o.predictor_iter, o.bound_eps)?;
                Dispatched {
                    step,
                    refined: Some((pinned, stop)),
                    segments: Vec::new(),
                }
            }
            Mode::Path => {
                let (step, segments) = conv.parametric_step_path(pins, deltas, o.predictor_iter)?;
                Dispatched {
                    step,
                    refined: None,
                    segments,
                }
            }
            Mode::Linear => Dispatched {
                step: conv.parametric_step(pins, deltas)?,
                refined: None,
                segments: Vec::new(),
            },
        };
    }
    Ok(out)
}

/// The budget's default, and a finding when it is inert: only the
/// directional decision spends back-solves.
fn budget(o: &StepOptions, findings: &mut Vec<StepFinding>) -> usize {
    match o.degeneracy_iter {
        None => 16,
        Some(b) => {
            if o.degeneracy != Degeneracy::Directional {
                findings.push(StepFinding::DegeneracyIterInert);
            }
            b
        }
    }
}

/// Refines a step by Newton iterations on the barrier system, returning the
/// refined primal step and what the iterations did.
///
/// The corrector starts from a point, so it needs the multipliers as well
/// as the primal step, and only the plain parametric step reports both.
/// The start is the mode's own primal block carrying the plain step's
/// multipliers, and the iterations correct the multipliers from there.
fn correct<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    pins: &[Index],
    deltas: &[Number],
    step: &[Number],
    corrector_iter: usize,
) -> Result<(Vec<Number>, CorrectorReport), SolverError> {
    let mut full = conv.parametric_step_full(pins, deltas)?;
    let n_x = step.len();
    full[..n_x].copy_from_slice(step);
    let (out, report) = conv.correct_step(pins, deltas, &full, corrector_iter)?;
    Ok((out[..n_x].to_vec(), report))
}

/// How far outside a bound a coordinate has to end to count as having left
/// it: the margin given, or how far outside the solve itself was willing to
/// settle, floored so an unrelaxed solve does not pin on roundoff.
fn bound_margin<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    eps: Option<Number>,
) -> Number {
    match eps {
        Some(e) => e,
        None => conv.bound_relax_factor().abs().max(1e-9),
    }
}

fn prelude<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    o: &StepOptions,
    findings: &mut Vec<StepFinding>,
) -> Result<usize, StepError> {
    let b = budget(o, findings);
    check_margins(o.bound_eps, o.max_pdpert)?;
    if o.bound_eps.is_some() && o.mode != Mode::FixRelax {
        findings.push(StepFinding::BoundEpsInert);
    }
    refuse_on_pdpert(conv, o.max_pdpert)?;
    Ok(b)
}

/// The first-order estimate of the solution at perturbed parameter values,
/// over the model's variables.
///
/// `pins` are the model rows of the defining equalities the parameters are
/// pinned by, and `deltas` how far each one's right-hand side moves from its
/// value at the solve. Under [`Mode::Linear`] the step is taken and, with
/// `clamp`, clamped to the variable bounds. Under [`Mode::FixRelax`] and
/// [`Mode::Path`] the refinement holds the bounds itself, and `clamp`
/// decides what happens to anything still outside at the end. A limit
/// written as a constraint row moves with the perturbation, so it is not
/// clamped against. A base point or variable bounds not over the model's
/// variables, or `deltas` not one per pin, return `BadShape` before any
/// step is taken.
#[allow(clippy::too_many_arguments)]
pub fn solution<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    model: &dyn ModelFunctions,
    base_x: &[Number],
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    clamp: bool,
) -> StepOutcome<Vec<Number>> {
    let mut findings = Vec::new();
    let result = solution_inner(conv, model, base_x, pins, deltas, o, clamp, &mut findings);
    StepOutcome { findings, result }
}

#[allow(clippy::too_many_arguments)]
fn solution_inner<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    model: &dyn ModelFunctions,
    base_x: &[Number],
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    clamp: bool,
    findings: &mut Vec<StepFinding>,
) -> Result<Vec<Number>, StepError> {
    let b = prelude(conv, o, findings)?;
    let maps = Maps::new(conv)?;
    check_inputs(&maps, model, base_x, pins, deltas)?;
    let d = dispatch(conv, pins, deltas, o, b, findings)?;
    let mut step = d.step;
    if o.corrector_iter > 0 {
        let (s, report) = correct(conv, pins, deltas, &step, o.corrector_iter)?;
        step = s;
        // A correction that works drives the residual down by orders, and
        // one that finds nothing to reduce shaves a few percent off.
        // Halving separates them, and a non-finite residual counts as not
        // halved.
        if !(report.residual <= 0.5 * report.initial_residual) {
            findings.push(StepFinding::CorrectorWeak { report });
        }
    }
    let dx = maps.scatter_x(&step);
    let mut x_new: Vec<Number> = base_x.iter().zip(&dx).map(|(a, b)| a + b).collect();
    let (lo, hi) = model.variable_bounds();
    let clip = |x: &mut Vec<Number>| {
        for (i, v) in x.iter_mut().enumerate() {
            *v = v.max(lo[i]).min(hi[i]);
        }
    };
    if o.mode == Mode::FixRelax || o.mode == Mode::Path {
        // The refinement holds each pinned coordinate at its bound, so a
        // coordinate can still be left outside one, because the budget ran
        // out or the pins exhausted the degrees of freedom. One tolerance,
        // the one the refinement pinned against, absolute as its own is.
        let eps = bound_margin(
            conv,
            if o.mode == Mode::FixRelax {
                o.bound_eps
            } else {
                None
            },
        );
        let out: Vec<usize> = (0..x_new.len())
            .filter(|&i| x_new[i] < lo[i] - eps || x_new[i] > hi[i] + eps)
            .collect();
        if !out.is_empty() {
            findings.push(StepFinding::StillOutside {
                vars: out,
                pinned: d.refined.as_ref().map(|(p, s)| (p.len(), *s)),
                changes: (o.mode == Mode::Path).then_some(d.segments.len()),
                clamped: clamp,
            });
            if clamp {
                clip(&mut x_new);
            }
        }
    } else if clamp {
        // a linear step takes the predictor as given, so a crossing shows
        // up as a value outside its bound and clamping is all it can do
        let eps = bound_margin(conv, None);
        let clamped: Vec<usize> = (0..x_new.len())
            .filter(|&i| x_new[i] < lo[i] - eps || x_new[i] > hi[i] + eps)
            .collect();
        if !clamped.is_empty() {
            findings.push(StepFinding::LinearClamped { vars: clamped });
        }
        clip(&mut x_new);
    }
    Ok(x_new)
}

/// The magnitude past which a model's bound is its "no bound" sentinel
/// rather than a bound: `.nl` readers seed every absent bound with `±1e19`,
/// which a finiteness test passes.
pub const NO_BOUND: Number = 1e19;

/// The smallest fraction of `step` that reaches a bound, and the index it
/// happens at, or infinity and `None` when no bound lies along the step.
///
/// Only coordinates strictly inside their bounds take part, since one
/// already on a bound has a gap and a step component of the same size and
/// any quotient. A coordinate the full step carries past a bound is always
/// scored, at a fraction below one, by the same predicate and tolerance
/// that fill the report's crossings. `on_bound` carries the classification
/// per side, and a distance test scaled by `√μ` rules on the coordinates
/// the classifier declined to. The remaining gap is `O(μ)` at a strongly
/// active bound and `O(√μ)` at a weakly active one, against `O(1)` room in
/// the interior, and the threshold was measured four orders of magnitude
/// clear of the interior case. It is capped at `1e-2` because it applies
/// relative to the coordinate's own magnitude, and a loose `μ` at
/// termination would otherwise widen it without limit. A step component
/// below the roundoff of its coordinate takes no part.
///
/// `tol` is the crossing tolerance per entry, and `None` is
/// `1e-9 · max(1, |base + step|)`. `mu` NaN means no classification, and
/// the distance test then falls back to a fixed threshold.
#[allow(clippy::too_many_arguments)]
pub fn ratio_test(
    base: &[Number],
    step: &[Number],
    lo: &[Number],
    hi: &[Number],
    live: Option<&[bool]>,
    on_bound: Option<&[bool]>,
    mu: Number,
    tol: Option<&[Number]>,
) -> (Number, Option<usize>) {
    let floor = if !mu.is_finite() {
        1e-9
    } else {
        1e-2_f64.min(1e-9_f64.max(10.0 * mu.sqrt()))
    };
    let mut best = (Number::INFINITY, None);
    for i in 0..base.len() {
        let scale = base[i].abs().max(1.0);
        let toward_hi = step[i] > 0.0;
        let bound = if toward_hi { hi[i] } else { lo[i] };
        let distance = bound - base[i];
        let present = bound.abs() < NO_BOUND;
        let moving = step[i].abs() > 1e-12 * scale;
        let reached = base[i] + step[i];
        let t = tol.map_or(1e-9 * reached.abs().max(1.0), |t| t[i]);
        let crosses = present
            && moving
            && if toward_hi {
                reached > bound + t
            } else {
                reached < bound - t
            };
        let mut interior = distance.abs() > floor * scale;
        if let Some(ob) = on_bound {
            // the nearer bound is the one the coordinate is held at, and
            // only that side is ruled out
            let at_hi = (hi[i] - base[i]) <= (base[i] - lo[i]);
            interior &= !(ob[i] && (toward_hi == at_hi));
        }
        let mut ok = present && moving && (crosses || interior);
        if let Some(l) = live {
            ok &= l[i];
        }
        if !ok {
            continue;
        }
        // a relaxed solve can settle outside a bound, which leaves no room
        // to move rather than a negative amount of it
        let f = (distance / step[i]).max(0.0);
        if best.1.is_none() || f < best.0 {
            best = (f, Some(i));
        }
    }
    best
}

/// The activity classes that put a coordinate on its bound: strongly and
/// weakly active. `ambiguous` and `unidentified` are not among them, since
/// neither label says whether a coordinate has room left.
pub fn on_bound(statuses: &[i8]) -> Vec<bool> {
    statuses
        .iter()
        .map(|&s| s == STRONGLY_ACTIVE || s == WEAKLY_ACTIVE)
        .collect()
}

/// What [`solution_report`] measures about a step, by model index.
#[derive(Debug, Clone, PartialEq)]
pub struct SolutionReport {
    /// The fraction of the perturbation that can be taken before the first
    /// bound is reached. Infinity when no bound lies along the step.
    pub alpha: Number,
    /// What reaches its bound at `alpha`.
    pub first: Option<BoundTarget>,
    /// `(full-x variable, distance)` for each variable the full step leaves
    /// a bound of.
    pub crossed: Vec<(usize, Number)>,
    /// `(full-g row, distance)` for each inequality row the step leaves a
    /// limit of.
    pub crossed_rows: Vec<(usize, Number)>,
    /// The largest constraint violation at the predicted point, against the
    /// perturbed right-hand sides, zero with no rows, and NaN where the model
    /// cannot evaluate a row there.
    pub violation: Number,
    /// The barrier parameter of the held factor, NaN on a relaxed solve.
    pub mu: Number,
    /// The activity class of every model variable, refined where asked.
    /// Empty on a relaxed solve.
    pub var_status: Vec<i8>,
    /// The activity class of every model row, likewise.
    pub row_status: Vec<i8>,
    /// `(target, before, after)` for every class the reduced normalizer
    /// changed.
    pub refined: Vec<(BoundTarget, i8, i8)>,
    /// Every entry the reduced normalizer classified, at one back-solve
    /// each: the entries the cheap classifier called ambiguous, variables
    /// first. Empty when the refinement was not asked for.
    pub refined_entries: Vec<BoundTarget>,
    /// The held factor's inertia correction, `None` when the engine does
    /// not report it. `None` is not "exact": an engine that does not
    /// report may still have perturbed its factor.
    pub perturbations: Option<[Number; 4]>,
    /// Whether the solve ran with relaxed bounds.
    pub bounds_relaxed: bool,
    /// What the corrector's iterations did, when it ran.
    pub corrector: Option<CorrectorReport>,
    /// Why the [`Mode::FixRelax`] refinement stopped, under that mode.
    pub refine_stop: Option<RefineStop>,
}

/// A model variable or a model row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundTarget {
    /// A full-x variable.
    Variable(usize),
    /// A full-g row.
    Row(usize),
}

/// Re-classifies the entries the cheap classifier called ambiguous with the
/// reduced normalizer, one back-solve each. Returns the classes it changed
/// and every entry it classified.
///
/// The cheap classifier divides by the Hessian's diagonal, or by the
/// curvature along a row's own gradient, while the multiplier is generated
/// by the reduced curvature, so a coupled kink is ambiguous to it at every
/// tolerance (gh#763, gh#804). The reduced normalizer answers it exactly,
/// and the cost is spent only on the entries that raised the question.
fn refine_ambiguous<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    var_status: &mut [i8],
    row_status: &mut [i8],
) -> Result<(Vec<(BoundTarget, i8, i8)>, Vec<BoundTarget>), SolverError> {
    let mut refined = Vec::new();
    let var_idx: Vec<usize> = (0..var_status.len())
        .filter(|&i| var_status[i] == AMBIGUOUS)
        .collect();
    let row_idx: Vec<usize> = (0..row_status.len())
        .filter(|&j| row_status[j] == AMBIGUOUS)
        .collect();
    if !var_idx.is_empty() {
        let red = conv.reduced_activity(&var_idx)?;
        for (k, &i) in var_idx.iter().enumerate() {
            let after = red.status[k];
            if after != var_status[i] {
                refined.push((BoundTarget::Variable(i), var_status[i], after));
                var_status[i] = after;
            }
        }
    }
    if !row_idx.is_empty() {
        let red = conv.reduced_row_activity(&row_idx)?;
        for (k, &j) in row_idx.iter().enumerate() {
            let after = red.status[k];
            if after != row_status[j] {
                refined.push((BoundTarget::Row(j), row_status[j], after));
                row_status[j] = after;
            }
        }
    }
    let entries = var_idx
        .into_iter()
        .map(BoundTarget::Variable)
        .chain(row_idx.into_iter().map(BoundTarget::Row))
        .collect();
    Ok((refined, entries))
}

/// Measures what [`solution`]'s step does about the bounds, for the same
/// options, without changing the estimate.
///
/// The ratio test runs over variables and over inequality rows. Equality
/// rows are held by the step to first order and take no part, and neither
/// do the rows pinning the parameters, which the perturbation moves by
/// construction. Under [`Mode::FixRelax`] and [`Mode::Path`] the step stops
/// at the bounds, so `alpha` is 1 and nothing crosses, unless a margin wide
/// enough to cover the crossing let the refinement pin nothing.
/// `refine_activity` re-classifies the ambiguous entries with the reduced
/// normalizer, at one back-solve each. The inputs that make [`solution`]
/// return an error make this return one too. So do row bounds not over the
/// model's rows, and constraint values or Jacobian entries the model returns
/// outside its rows and variables.
#[allow(clippy::too_many_arguments)]
pub fn solution_report<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    model: &mut dyn ModelFunctions,
    base_x: &[Number],
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    refine_activity: bool,
) -> StepOutcome<SolutionReport> {
    let mut findings = Vec::new();
    let result = report_inner(
        conv,
        model,
        base_x,
        pins,
        deltas,
        o,
        refine_activity,
        &mut findings,
    );
    StepOutcome { findings, result }
}

#[allow(clippy::too_many_arguments)]
fn report_inner<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    model: &mut dyn ModelFunctions,
    base_x: &[Number],
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    refine_activity: bool,
    findings: &mut Vec<StepFinding>,
) -> Result<SolutionReport, StepError> {
    let b = prelude(conv, o, findings)?;
    let maps = Maps::new(conv)?;
    check_inputs(&maps, model, base_x, pins, deltas)?;
    let (g_l, g_u) = model.row_bounds();
    let n_g = conv.n_full_g()?;
    for (what, got) in [
        ("row lower bounds", g_l.len()),
        ("row upper bounds", g_u.len()),
    ] {
        if got != n_g {
            return Err(SolverError::BadShape {
                what,
                got,
                expected: n_g,
            }
            .into());
        }
    }
    let d = dispatch(conv, pins, deltas, o, b, findings)?;
    let refine_stop = d.refined.as_ref().map(|(_, s)| *s);
    let dx = maps.scatter_x(&d.step);
    let x_new: Vec<Number> = base_x.iter().zip(&dx).map(|(a, b)| a + b).collect();
    let (lo, hi) = model.variable_bounds();

    // The margin the refinement pinned against, so `alpha` and `crossed`
    // answer with the number that decided the step. Unset keeps the fixed
    // floor rather than the solve's relaxation, since a coordinate the
    // solve itself left outside a relaxed bound is one the report names.
    let eps = match (o.bound_eps, o.mode) {
        (Some(e), Mode::FixRelax) => e,
        _ => 1e-9,
    };

    // The classifier refuses a solve that relaxed its bounds, so that is
    // asked first and reported as the fact it is, and the rest of the
    // report is still measured.
    let bounds_relaxed = conv.bound_relax_factor() != 0.0;
    let (mut mu, mut var_status, mut row_status) = (Number::NAN, Vec::new(), Vec::new());
    let (mut refined, mut refined_entries) = (Vec::new(), Vec::new());
    let (mut var_on, mut row_on) = (None, None);
    if !bounds_relaxed {
        let act = conv.classify_activity()?;
        mu = act.mu;
        var_status = act.var_status;
        row_status = act.row_status;
        if refine_activity {
            (refined, refined_entries) = refine_ambiguous(conv, &mut var_status, &mut row_status)?;
        }
        // a kink the cheap rule could not call is still a coordinate on its
        // bound
        var_on = Some(on_bound(&var_status));
        row_on = Some(on_bound(&row_status));
    }

    let var_tol = vec![eps; base_x.len()];
    let (mut alpha, first_var) = ratio_test(
        base_x,
        &dx,
        &lo,
        &hi,
        None,
        var_on.as_deref(),
        mu,
        Some(&var_tol),
    );
    let mut first = first_var.map(BoundTarget::Variable);
    // the step's effect on each row, the Jacobian at the base point times
    // the step, summed entry by entry in the model's order
    let m = g_l.len();
    let mut dg = vec![0.0; m];
    for (r, c, v) in model.jacobian(base_x).map_err(StepError::Model)? {
        if r >= m || c >= dx.len() {
            return Err(StepError::Model(format!(
                "the Jacobian has an entry at ({r}, {c}) outside {m} rows and {} columns",
                dx.len()
            )));
        }
        dg[r] += v * dx[c];
    }
    let g_base = model.constraints(base_x).map_err(StepError::Model)?;
    check_rows(&g_base, m)?;
    // an equality is held to first order and cannot change activity, and a
    // pin row moves by construction
    let mut live: Vec<bool> = (0..m).map(|j| g_l[j] < g_u[j]).collect();
    for &p in pins {
        live[p as usize] = false;
    }
    let g_pred: Vec<Number> = g_base.iter().zip(&dg).map(|(a, b)| a + b).collect();
    let gtol_v: Vec<Number> = g_pred.iter().map(|g| 1e-9 * g.abs().max(1.0)).collect();
    let gtol = |j: usize| gtol_v[j];
    let (a_row, f_row) = ratio_test(
        &g_base,
        &dg,
        &g_l,
        &g_u,
        Some(&live),
        row_on.as_deref(),
        mu,
        Some(&gtol_v),
    );
    if a_row < alpha {
        alpha = a_row;
        first = f_row.map(BoundTarget::Row);
    }

    let crossed: Vec<(usize, Number)> = (0..x_new.len())
        .filter(|&i| x_new[i] < lo[i] - eps || x_new[i] > hi[i] + eps)
        .map(|i| (i, (lo[i] - x_new[i]).max(x_new[i] - hi[i])))
        .collect();
    let crossed_rows: Vec<(usize, Number)> = (0..m)
        .filter(|&j| live[j] && (g_pred[j] < g_l[j] - gtol(j) || g_pred[j] > g_u[j] + gtol(j)))
        .map(|j| (j, (g_l[j] - g_pred[j]).max(g_pred[j] - g_u[j])))
        .collect();

    // the perturbation is the shift of the pin rows' right-hand sides, so
    // the violation is measured against the shifted ones
    let (mut gl_p, mut gu_p) = (g_l.clone(), g_u.clone());
    for (&p, &dd) in pins.iter().zip(deltas) {
        gl_p[p as usize] += dd;
        gu_p[p as usize] += dd;
    }
    let g_at = model.constraints(&x_new).map_err(StepError::Model)?;
    check_rows(&g_at, m)?;
    let violation = max_violation(&g_at, &gl_p, &gu_p);

    let corrector = if o.corrector_iter > 0 {
        Some(correct(conv, pins, deltas, &d.step, o.corrector_iter)?.1)
    } else {
        None
    };
    Ok(SolutionReport {
        alpha,
        first,
        crossed,
        crossed_rows,
        violation,
        mu,
        var_status,
        row_status,
        refined,
        refined_entries,
        perturbations: conv.kkt_perturbations(),
        bounds_relaxed,
        corrector,
        refine_stop,
    })
}

/// The largest amount by which `g` lies outside `[lo, hi]`: zero with no
/// rows or with every row inside, and NaN when any row is NaN.
fn max_violation(g: &[Number], lo: &[Number], hi: &[Number]) -> Number {
    // `f64::max` drops a NaN, so a NaN row is returned as NaN here rather
    // than read as satisfied
    if g.iter().any(|v| v.is_nan()) {
        return Number::NAN;
    }
    g.iter()
        .zip(lo.iter().zip(hi))
        .fold(0.0, |acc: Number, (v, (l, u))| acc.max((l - v).max(v - u)))
}

/// Returns a model error when `g` does not hold one value per row.
fn check_rows(g: &[Number], m: usize) -> Result<(), StepError> {
    if g.len() == m {
        Ok(())
    } else {
        Err(StepError::Model(format!(
            "the model returned {} constraint values for {m} rows",
            g.len()
        )))
    }
}

/// One active-set change along a path step.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveSetChange {
    /// How far along the perturbation it happens.
    pub fraction: Number,
    /// The variable whose bound changed, or the row whose limit did: a
    /// limit written as `g(x) <= cap` bounds the row's slack (gh#928).
    pub target: BoundTarget,
    /// Whether the bound is the lower one.
    pub lower: bool,
    /// Whether the quantity reaches the bound and is held there, rather
    /// than leaving it.
    pub reaches: bool,
}

/// The active-set changes [`solution`] under [`Mode::Path`] applies, in
/// order. `o.mode` is read as [`Mode::Path`] whatever it says, and there is
/// no margin, since the path decides from where a multiplier reaches zero.
pub fn active_set_changes<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
) -> StepOutcome<Vec<ActiveSetChange>> {
    let mut findings = Vec::new();
    let result = changes_inner(conv, pins, deltas, o, &mut findings);
    StepOutcome { findings, result }
}

fn changes_inner<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    pins: &[Index],
    deltas: &[Number],
    o: &StepOptions,
    findings: &mut Vec<StepFinding>,
) -> Result<Vec<ActiveSetChange>, StepError> {
    let path = StepOptions {
        mode: Mode::Path,
        bound_eps: None,
        corrector_iter: 0,
        ..*o
    };
    let b = budget(&path, findings);
    check_margins(None, path.max_pdpert)?;
    refuse_on_pdpert(conv, path.max_pdpert)?;
    let d = dispatch(conv, pins, deltas, &path, b, findings)?;
    // A segment names its bounded quantity by primal KKT row, `x` then `s`.
    // Below `n_x` it is a variable in var-x, and at or above it a row's own
    // limit, which resolving against the variable map would misname as a
    // neighbouring variable (the gh#450 hazard).
    let n_x = conv.engine().block_dims()[0];
    let maps = Maps::new(conv)?;
    let mut slack_of: Option<Vec<Option<Index>>> = None;
    let mut out = Vec::with_capacity(d.segments.len());
    for seg in &d.segments {
        let target = if seg.var_row < n_x {
            BoundTarget::Variable(maps.full_of(seg.var_row).ok_or_else(|| {
                StepError::Solver(SolverError::SensComputationFailed(format!(
                    "active_set_changes: factor row {} has no model variable",
                    seg.var_row
                )))
            })?)
        } else {
            if slack_of.is_none() {
                let n_g = conv.n_full_g()?;
                let rows: Vec<Index> = (0..n_g as Index).collect();
                slack_of = Some(conv.d_slack_rows(&rows)?);
            }
            let g = slack_of
                .as_ref()
                .expect("set above")
                .iter()
                .position(|&r| r == Some(seg.var_row as Index))
                .ok_or_else(|| {
                    StepError::Solver(SolverError::SensComputationFailed(format!(
                        "active_set_changes: slack row {} has no model row",
                        seg.var_row
                    )))
                })?;
            BoundTarget::Row(g)
        };
        out.push(ActiveSetChange {
            fraction: seg.at,
            target,
            lower: seg.lower,
            reaches: seg.pinned,
        });
    }
    Ok(out)
}

/// Why an inequality row's multiplier has no sensitivity to read.
#[derive(Debug, Clone)]
pub enum MultiplierRefusal {
    /// The row is in neither multiplier block.
    NoBlock,
    /// The row's activity could not be classified.
    Unclassified(SolverError),
    /// The row is not strictly active: its reduced activity is this class.
    NotStrictlyActive(i8),
}

/// The compound KKT row holding model row `g`'s multiplier, the row a
/// parametric step's column reads `∂λ/∂p` from.
///
/// An equality's row is always the answer. An inequality's is the answer
/// only where the row is strictly active, the regime in which it behaves
/// as an equality over a neighbourhood (gh#910). An inactive row's
/// derivative is a structural zero the row does not carry, and a kink has
/// two one-sided derivatives. The gate reads the reduced classification,
/// not the cheap one, which at strong enough coupling calls a genuine kink
/// inactive (gh#804).
pub fn multiplier_row<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    g: usize,
) -> Result<Result<usize, MultiplierRefusal>, SolverError> {
    if let Some(r) = conv.g_multiplier_rows(&[g as Index])?[0] {
        return Ok(Ok(r as usize));
    }
    let Some(r) = conv.d_multiplier_rows(&[g as Index])?[0] else {
        return Ok(Err(MultiplierRefusal::NoBlock));
    };
    let status = match conv.reduced_row_activity(&[g]) {
        Ok(rep) => rep.status[0],
        Err(e) => return Ok(Err(MultiplierRefusal::Unclassified(e))),
    };
    if status == STRONGLY_ACTIVE {
        Ok(Ok(r as usize))
    } else {
        Ok(Err(MultiplierRefusal::NotStrictlyActive(status)))
    }
}

/// The total derivative `df/dp` for a parametric step's column `col`:
/// `grad · dx`, over the model's variables. A declared parameter is an
/// ordinary coordinate of the solve, pinned by a defining equality, so the
/// gradient carries `∂f/∂p` in the parameter's own slot and the step
/// carries `dp/dp = 1` there, and the one contraction picks up the
/// explicit partial and the implicit sum together (gh#878). `grad` is the
/// objective's gradient in the sense the model states it, over the model's
/// variables. `col` is the factor's `x` block or a whole KKT column, whose
/// `x` block comes first. Either at another length returns `BadShape`.
pub fn total_objective_derivative<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    grad: &[Number],
    col: &[Number],
) -> Result<Number, SolverError> {
    let maps = Maps::new(conv)?;
    maps.check_full("objective gradient", grad.len())?;
    let n_var = conv.engine().block_dims()[0];
    if col.len() != n_var && col.len() != conv.engine().dim() {
        return Err(SolverError::BadShape {
            what: "step column",
            got: col.len(),
            expected: n_var,
        });
    }
    let dx = maps.scatter_x(col);
    Ok(grad.iter().zip(&dx).map(|(a, b)| a * b).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activity::INACTIVE;

    #[test]
    fn the_verdict_compares_the_largest_correction_with_the_cap() {
        let p = [0.0, 1e-3, 0.0, -2e-3];
        assert_eq!(pdpert_verdict(&p, 1e-3), (true, 2e-3));
        assert_eq!(pdpert_verdict(&p, 1e-2), (false, 2e-3));
    }

    #[test]
    fn the_violation_is_zero_without_rows_and_nan_with_a_nan_row() {
        assert_eq!(max_violation(&[], &[], &[]), 0.0);
        let (lo, hi) = ([0.0, 0.0], [1.0, 1.0]);
        assert_eq!(max_violation(&[0.5, 1.0], &lo, &hi), 0.0);
        assert_eq!(max_violation(&[-0.25, 1.5], &lo, &hi), 0.5);
        for g in [[Number::NAN, 3.0], [3.0, Number::NAN]] {
            assert!(max_violation(&g, &lo, &hi).is_nan(), "{g:?}");
        }
    }

    #[test]
    fn a_margin_must_be_a_positive_number() {
        assert!(check_margins(Some(1e-6), Some(1.0)).is_ok());
        assert!(check_margins(None, None).is_ok());
        for bad in [0.0, -1.0, Number::NAN] {
            assert!(matches!(
                check_margins(Some(bad), None),
                Err(StepError::NotPositive {
                    name: "bound_eps",
                    ..
                })
            ));
            assert!(matches!(
                check_margins(None, Some(bad)),
                Err(StepError::NotPositive {
                    name: "max_pdpert",
                    ..
                })
            ));
        }
    }

    #[test]
    fn the_ratio_test_finds_the_first_bound_along_the_step() {
        let r = |base: f64, step: f64, lo: f64, hi: f64, ob: Option<bool>, mu: f64| {
            let ob = ob.map(|b| vec![b]);
            ratio_test(
                &[base],
                &[step],
                &[lo],
                &[hi],
                None,
                ob.as_deref(),
                mu,
                None,
            )
        };
        assert_eq!(r(0.5, 1.0, 0.0, 1.0, None, 1e-8), (0.5, Some(0)));
        // held at its upper bound, the coordinate can still move off it
        // toward the lower one
        assert_eq!(r(1.0, -0.5, 0.0, 1.0, Some(true), 1e-8), (2.0, Some(0)));
        // and pressing into it within the crossing tolerance is not a step
        // toward it
        assert_eq!(r(1.0, 1e-10, 0.0, 1.0, Some(true), 1e-8).1, None);
        // settled outside a bound and moving further out: no room at all
        assert_eq!(r(1.1, 0.1, 0.0, 1.0, None, 1e-8), (0.0, Some(0)));
        // the `.nl` sentinel is no bound
        assert_eq!(r(0.0, 1.0, -NO_BOUND, NO_BOUND, None, 1e-8).1, None);
        // within 10√μ of the bound and not crossing it is on the bound
        assert_eq!(r(0.995, 0.001, 0.0, 1.0, None, 1e-6).1, None);
        assert_eq!(r(0.995, 0.001, 0.0, 1.0, None, 1e-12).1, Some(0));
    }

    #[test]
    fn strongly_and_weakly_active_are_on_their_bound() {
        assert_eq!(
            on_bound(&[STRONGLY_ACTIVE, WEAKLY_ACTIVE, INACTIVE, AMBIGUOUS]),
            vec![true, true, false, false]
        );
    }
}
