//! Second-order certification for the active-set engine (gh #848).
//!
//! # Why a first-order test is not enough
//!
//! Every exit in [`crate::solver`] that returns [`QpStatus::Optimal`] does so
//! on a **first-order** argument: the projected gradient vanishes and the
//! multipliers of the working set carry admissible signs. On a positive
//! semi-definite `H` that is the whole story — a KKT point of a convex QP is a
//! global minimizer. On an indefinite `H` it is not: a saddle point satisfies
//! exactly the same conditions.
//!
//! The failure is not hypothetical and it is not rare. Take
//!
//! ```text
//!     min  ½ xᵀ [[1 5]; [5 1]] x        (eigenvalues −4 and 6)
//!     s.t. −1 ≤ x ≤ 1
//! ```
//!
//! whose minimum is `−4`, attained at `(1, −1)` and `(−1, 1)`.
//! [`crate::solver::ParametricActiveSetSolver`] starts box-constrained solves
//! from the projection of the origin into the box, which here *is* the saddle.
//! The gradient is already zero, so the very first iterate reports
//! `‖p‖∞ ≤ opt_tol` with an empty working set and the solve returns
//! `Optimal` at `x = 0`, `obj = 0` — the *maximum along one eigendirection*,
//! certified as a solution. Inertia control does not save it: §4.5 shifts the
//! `H` block until the reduced KKT factors, which supplies a descent direction
//! for a nonzero right-hand side, and the right-hand side here is zero.
//!
//! # The test
//!
//! At a first-order point with working set `W`, the point is a local minimum
//! iff the reduced Hessian `Zᵀ H Z` is positive semi-definite, where `Z` spans
//! `null(A_W)`. This module answers the complementary question — *is there a
//! feasible direction of strictly negative curvature?* — and answers it with a
//! **witness** rather than an inference:
//!
//! * `d` with `A_W d = 0`, `E_W d = 0` (checked explicitly), and
//! * `dᵀ H d < 0` (evaluated directly against `H`, not read off a factor).
//!
//! A verdict of [`SecondOrder::NegativeCurvature`] therefore never rests on a
//! backend's inertia count being trustworthy, and never rests on an inference
//! from `δ > 0`. That matters because `δ > 0` is *ambiguous*: the §4.5 ladder
//! shifts on `WrongInertia` **and** on `Singular`, and a singular reduced
//! Hessian is a weak minimum — a point this module must not reject. Only a
//! direction in hand distinguishes the two.
//!
//! # How the witness is found
//!
//! By shift-and-invert, on the machinery §4.5 already owns. For a shift `δ`
//! large enough that
//!
//! ```text
//!     K_δ = [ H + δI   A_Wᵀ ]
//!           [ A_W       0   ]
//! ```
//! factors with the inertia of a minimizer, the reduced matrix
//! `Zᵀ(H + δI)Z` is positive definite with eigenvalues `λᵢ(ZᵀHZ) + δ`.
//! Solving `K_δ [d; y] = [v; 0]` yields a `d` automatically in `null(A_W)`,
//! and the map `v ↦ d` is `Z (Zᵀ(H+δI)Z)⁻¹ Zᵀ`. Iterating it is **inverse
//! iteration**: it converges to the eigenvector of the smallest eigenvalue of
//! `Zᵀ(H+δI)Z`, which is the eigenvector of the *most negative* eigenvalue of
//! `ZᵀHZ` — exactly the direction wanted.
//!
//! Its rate is `(λ₂ + δ)/(λ_min + δ)` per step, and that is why this module
//! does **not** simply reuse the shift `§4.5` settled on. The ladder
//! multiplies by `inertia_shift_factor` (100) until a shift works, so it
//! overshoots `|λ_min|` by up to two orders of magnitude, and an overshot
//! shift flattens the spectrum: on `H = diag(1, −1)` the ladder stops at
//! `δ = 100`, giving `diag(101, 99)` and a rate of `1.02` per step. Several
//! hundred back-solves would be needed to separate a direction that is one
//! factorization away from being exact — and the first draft of this module,
//! which did reuse the shift, missed that fixture for precisely that reason.
//!
//! So the shift is refined first. The ladder brackets `|λ_min|` between the
//! step that worked and the one before it, and `neg_curv_shift_refinements`
//! geometric bisections on that bracket — each one factorization, each
//! answering the same yes/no inertia question the ladder asks — narrow it to
//! `|λ_min| · (1 + 100^(2^−r))`. At the default `r = 8` that is a 1.8%
//! overshoot, `λ_min + δ` is under 2% of `|λ_min|`, and the iteration
//! separates the direction in one or two back-solves.
//!
//! Two more consequences worth stating, because both are load-bearing:
//!
//! * **The unshifted factor succeeding is a certificate of the opposite
//!   verdict.** `K₀` having exactly `|W|` negative eigenvalues *is* the
//!   statement that `ZᵀHZ ≻ 0`, so that case returns
//!   [`SecondOrder::Certified`] immediately. On a PSD problem it is the only
//!   branch reachable, so the whole module costs one factorization there —
//!   and the `HessianInertia::Psd` gate means it does not even cost that.
//! * **The seed must not be symmetric.** Inverse iteration from a vector that
//!   happens to *be* an eigenvector is stationary. On the `[[1,5];[5,1]]`
//!   model, `v = (1, 1)` is the eigenvector for `+6`, so an all-ones seed
//!   probes the one direction that can never turn negative. The seed is
//!   therefore a fixed deterministic pseudo-random vector — deterministic
//!   because a solver that reports a different status on a re-run is worse
//!   than one that misses a saddle.
//!
//! # What this module is *not* evidence about
//!
//! Failure to find a witness is reported as [`SecondOrder::NotChecked`], never
//! as `Certified`. The probe is a sufficient test for rejection and nothing
//! more: an exhausted iteration budget, a backend failure, or a rank-deficient
//! constraint block whose masked null direction fails the explicit `A_W d = 0`
//! check all land there, and all leave the engine's first-order verdict
//! standing. That is deliberate — the alternative is downgrading correct
//! answers on weak minima, which is a wrong verdict about the user's model.
//!
//! One class is worth naming because it looks like it should be covered and is
//! not: **negative curvature hidden behind a degenerate active bound**. The
//! test asks about `null(A_W)` for the working set the engine stopped at, so a
//! row or bound sitting in that set with a zero multiplier hides whatever lies
//! on the other side of it. On
//!
//! ```text
//!     min ½x₁² − ½x₂²   s.t.  −1 ≤ x₁ ≤ 1,  x₂ ≥ 0
//! ```
//!
//! the origin is stationary, `x₂`'s bound is active with multiplier exactly
//! zero — so the drop rule, which needs a multiplier that violates its sign
//! condition by more than `opt_tol`, leaves it in — and the null space is the
//! `x₁` axis, on which the curvature is `+1`. The verdict is `Certified`, and
//! the point is not a local minimum. Seeing it requires searching over working
//! sets rather than within one, which for an indefinite QP is the NP-hard part
//! of the problem and not something a certification pass should be pretending
//! to do. What this module removes is the class where the answer is visible
//! from where the engine already stands.

use crate::error::QpError;
use crate::kkt::{KktTriplet, a_times_x, assemble_active_set_kkt, h_times_x};
use crate::options::QpOptions;
use crate::problem::{HessianInertia, QpProblem};
use crate::solver::ParametricActiveSetSolver;
use crate::working_set::WorkingSet;
use pounce_common::Number;

/// The verdict of the second-order test at a first-order point.
///
/// The rejecting variant carries its own witness, so a caller acting on it is
/// acting on a direction it can re-check rather than on this module's word.
#[derive(Debug, Clone, PartialEq)]
pub enum SecondOrder {
    /// No verdict. The engine's first-order finding stands unchanged.
    ///
    /// Reached when the check is switched off, when `H` is claimed PSD (there
    /// is nothing to find), when the probe budget runs out, and when the
    /// backend or the factor declines to cooperate. Never a claim that the
    /// point *is* a minimum.
    NotChecked,
    /// The reduced Hessian on `null(A_W)` is positive definite: the point is a
    /// local minimum of the QP, not merely a stationary point of it.
    Certified,
    /// A feasible direction of strictly negative curvature, `A_W d = 0` and
    /// `dᵀHd < 0`, normalized to `‖d‖₂ = 1`. The point is a saddle or a
    /// maximum along `d`; it is not a local minimum, at any tolerance.
    NegativeCurvature(Vec<Number>),
}

/// A fixed 64-bit-state xorshift, used only to seed the inverse iteration.
///
/// Deterministic on purpose (see the module docs): reproducibility of a
/// *status* outranks the marginally better coverage of a fresh random seed.
/// `Math.random`-grade quality is irrelevant here — the seed only has to avoid
/// being an eigenvector of the reduced Hessian, and any vector with a nonzero
/// component along the sought eigendirection works.
fn probe_seed(n: usize) -> Vec<Number> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to (-1, 1) with a deterministic mantissa.
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        v.push(2.0 * u - 1.0);
    }
    v
}

/// `dᵀHd`, evaluated against the problem's own `H`.
fn curvature(qp: &QpProblem<'_>, d: &[Number]) -> Number {
    let hd = h_times_x(qp.h, d);
    hd.iter().zip(d.iter()).map(|(a, b)| a * b).sum()
}

/// Largest magnitude of any stored entry, the scale a curvature is compared
/// against. `H` stores one triangle only, which does not matter for a max.
fn inf_scale(vals: &[Number]) -> Number {
    vals.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// Normalize in place to `‖d‖₂ = 1`. Returns `false` if `d` is zero or
/// non-finite, in which case it is not a usable direction.
fn normalize(d: &mut [Number]) -> bool {
    let nrm = d.iter().map(|v| v * v).sum::<Number>().sqrt();
    if !nrm.is_finite() || nrm <= 0.0 {
        return false;
    }
    for v in d.iter_mut() {
        *v /= nrm;
    }
    d.iter().all(|v| v.is_finite())
}

/// Is `d` genuinely in the null space of the working set's rows?
///
/// Checked rather than assumed. A KKT whose *constraint* block is rank
/// deficient cannot be repaired by any `H`-block shift, but a large enough
/// shift can stop the backend flagging it — the masking `factor_pinned_primal`
/// documents around `δ ≈ 1e8`. The solve then returns a plausible vector that
/// satisfies nothing, and stepping along it would leave the feasible set. This
/// is the guard that turns that into a `NotChecked`.
fn stays_on_the_working_set(qp: &QpProblem<'_>, working: &WorkingSet, d: &[Number]) -> bool {
    // `d` is normalized, so `‖d‖∞ ≤ 1` and an absolute slack scaled by the
    // data is the right comparison.
    let slack = 1e-7;
    for (i, status) in working.bounds.iter().enumerate() {
        if status.is_active() && d[i].abs() > slack {
            return false;
        }
    }
    if qp.m == 0 {
        return true;
    }
    let ad = a_times_x(qp.a, d, qp.m);
    let a_scale = inf_scale(qp.a.values()).max(1.0);
    for (i, status) in working.constraints.iter().enumerate() {
        if status.is_active() && ad[i].abs() > slack * a_scale {
            return false;
        }
    }
    true
}

impl ParametricActiveSetSolver {
    /// Decide whether the first-order point described by `working` is a local
    /// minimum, and produce a witness when it is not. See the module docs for
    /// the mechanism and for the deliberate asymmetry between the verdicts.
    ///
    /// Costs one factorization of the active-set KKT plus at most
    /// `opts.neg_curv_probe_iters` back-substitutions, and only when
    /// `qp.hessian_inertia` is not [`HessianInertia::Psd`] — so the convex arm
    /// pays nothing by construction.
    ///
    /// Leaves `self.linsol` holding the factor of the probed KKT. Callers that
    /// need their own cached factor afterwards must re-establish it.
    pub(crate) fn second_order_verdict(
        &mut self,
        qp: &QpProblem<'_>,
        working: &WorkingSet,
        opts: &QpOptions,
    ) -> Result<SecondOrder, QpError> {
        if !opts.certify_second_order || qp.hessian_inertia == HessianInertia::Psd {
            return Ok(SecondOrder::NotChecked);
        }
        let n = qp.n;
        if n == 0 {
            return Ok(SecondOrder::Certified);
        }

        let active_cons: Vec<usize> = (0..qp.m)
            .filter(|&i| working.constraints[i].is_active())
            .collect();
        let active_bounds: Vec<usize> = (0..n).filter(|&i| working.bounds[i].is_active()).collect();
        let k = active_cons.len() + active_bounds.len();
        if k >= n {
            // The working set pins every degree of freedom: `null(A_W)` is
            // trivial (or the rows are dependent, which is not this module's
            // business), so there is no direction to move along and the point
            // is a minimum of the QP restricted to it.
            return Ok(SecondOrder::Certified);
        }

        let mut kkt = assemble_active_set_kkt(qp, &active_cons, &active_bounds);
        let dim = n + k;
        let seed = probe_seed(n);
        let expected = k as i32;

        // Attempt zero: unshifted. Success here is the `Certified` verdict —
        // the KKT carrying exactly `k` negative eigenvalues *is* the statement
        // that the reduced Hessian is positive definite.
        let mut rhs = vec![0.0; dim];
        rhs[..n].copy_from_slice(&seed);
        match self
            .linsol
            .factorize_and_solve(&kkt, &mut rhs, Some(expected))
        {
            Ok(()) => return Ok(SecondOrder::Certified),
            Err(ref e) if e.is_recoverable_factorization_failure() => {}
            // A hard backend failure is an absence of evidence, not a verdict.
            Err(_) => return Ok(SecondOrder::NotChecked),
        }

        // Climb §4.5's ladder for a shift that does factor, keeping the
        // bracket: `hi` works, `hi / inertia_shift_factor` did not, so
        // `|λ_min|` lies between them.
        let mut current = 0.0;
        let mut hi = None;
        let mut next = opts.inertia_shift_initial;
        for _ in 0..opts.inertia_max_shifts {
            if crate::deadline::expired() {
                return Err(QpError::DeadlineExpired);
            }
            match self.try_shift(&mut kkt, &mut current, next, n, &seed, expected, dim) {
                Ok(true) => {
                    hi = Some(next);
                    break;
                }
                Ok(false) => next *= opts.inertia_shift_factor,
                Err(e) => return Err(e),
            }
        }
        let Some(mut hi) = hi else {
            // No shift in the ladder's reach makes the reduced Hessian
            // definite. `factorize_with_inertia_control` calls that a hard
            // error; here it is simply no verdict.
            return Ok(SecondOrder::NotChecked);
        };

        if hi <= opts.inertia_shift_initial {
            // The very first rung worked, so `|λ_min| < inertia_shift_initial`
            // — a negative eigenvalue at the level of the rounding noise in
            // the factor. There is nothing here worth rejecting a solve over,
            // and the bracket below has no lower end to bisect against.
            return Ok(SecondOrder::NotChecked);
        }

        let mut lo = hi / opts.inertia_shift_factor;
        for _ in 0..opts.neg_curv_shift_refinements {
            if crate::deadline::expired() {
                return Err(QpError::DeadlineExpired);
            }
            let mid = (lo * hi).sqrt();
            if !(mid > lo && mid < hi) {
                break;
            }
            match self.try_shift(&mut kkt, &mut current, mid, n, &seed, expected, dim) {
                Ok(true) => hi = mid,
                Ok(false) => lo = mid,
                Err(e) => return Err(e),
            }
        }

        // Land on the tightest shift known to work, so `self.linsol` holds
        // the factor the iteration below back-solves against and `rhs` holds
        // the first iterate.
        let mut rhs = vec![0.0; dim];
        rhs[..n].copy_from_slice(&seed);
        kkt.add_h_diagonal_shift(n, hi - current);
        if self
            .linsol
            .factorize_and_solve(&kkt, &mut rhs, Some(expected))
            .is_err()
        {
            return Ok(SecondOrder::NotChecked);
        }

        // A shift was needed at all, so `ZᵀHZ` is indefinite, singular, or the
        // constraint block is rank deficient. Only a witness separates them.
        let h_scale = inf_scale(qp.h.values()).max(1.0);
        let threshold = -opts.neg_curv_tol * h_scale;

        let mut d = rhs[..n].to_vec();
        for _ in 0..opts.neg_curv_probe_iters {
            if crate::deadline::expired() {
                return Err(QpError::DeadlineExpired);
            }
            if !normalize(&mut d) {
                return Ok(SecondOrder::NotChecked);
            }
            if curvature(qp, &d) < threshold && stays_on_the_working_set(qp, working, &d) {
                return Ok(SecondOrder::NegativeCurvature(d));
            }
            // One more step of inverse iteration against the cached factor.
            let mut step = vec![0.0; dim];
            step[..n].copy_from_slice(&d);
            if self.linsol.resolve(&mut step).is_err() {
                return Ok(SecondOrder::NotChecked);
            }
            d.copy_from_slice(&step[..n]);
        }
        Ok(SecondOrder::NotChecked)
    }

    /// Move `kkt`'s `H`-block shift from `*current` to `target` and ask
    /// whether it factors with `expected` negative eigenvalues. `Ok(false)`
    /// is the inertia/singularity answer the caller is bracketing on; a hard
    /// backend failure comes back as `Ok(false)` too, since a shift that
    /// cannot be factored is a shift that does not work.
    #[allow(clippy::too_many_arguments)]
    fn try_shift(
        &mut self,
        kkt: &mut KktTriplet,
        current: &mut Number,
        target: Number,
        n: usize,
        seed: &[Number],
        expected: i32,
        dim: usize,
    ) -> Result<bool, QpError> {
        kkt.add_h_diagonal_shift(n, target - *current);
        *current = target;
        let mut rhs = vec![0.0; dim];
        rhs[..n].copy_from_slice(seed);
        match self
            .linsol
            .factorize_and_solve(kkt, &mut rhs, Some(expected))
        {
            Ok(()) => Ok(true),
            Err(QpError::DeadlineExpired) => Err(QpError::DeadlineExpired),
            Err(_) => Ok(false),
        }
    }
}
