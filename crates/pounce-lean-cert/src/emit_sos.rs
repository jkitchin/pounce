//! The SOS driver: a float SDP relaxation → an exact-rational
//! `verdict = "global-lower-bound"` certificate, or a typed refusal.
//!
//! This is [`crate::emit`]'s sibling for a claim of a different kind. The QP
//! path exhibits a *point* and proves it optimal; this path proves
//! `γ ≤ p(x)` for **every** `x`, with no convexity assumed. That is the
//! interesting case: for a nonconvex polynomial there is no KKT argument to
//! make, because a KKT point is only ever locally optimal.
//!
//! The mechanism is a Positivstellensatz identity,
//!
//! ```text
//!     p(x) − γ = m(x)ᵀ G m(x),   G ⪰ 0
//! ```
//!
//! whose right side is a sum of squares and therefore nonnegative everywhere;
//! `γ ≤ p(x)` follows. Lean closes the identity with `ring` and the PSD claim
//! with an exact `LDLᵀ`, so nothing in the trusted path is floating point.
//!
//! # Constraints
//!
//! With a feasible set `K = {x : gₖ(x) ≥ 0}` the same machinery proves a bound
//! *on `K`*, through the Putinar form
//!
//! ```text
//!     p(x) − γ = σ₀(x) + Σₖ σₖ(x)·gₖ(x),   every σ a Gram form with G ⪰ 0.
//! ```
//!
//! At a feasible `x` every product is a nonnegative times a nonnegative, so the
//! same one-line sign argument closes it. The demand on `p − γ` is far weaker
//! than being a sum of squares outright — it need only be one *modulo the
//! constraints* — which is why this reaches problems the unconstrained form
//! cannot, starting with every one whose objective is unbounded off `K`.
//!
//! The two shapes are one code path here, not two: the unconstrained case is the
//! Putinar case with a single block whose multiplier is the constant `1`. What
//! differs is the *claim*. A constrained bound holds on `K` alone, and reading
//! it as a global bound would be strictly stronger and generally false — so
//! `problem.poly_constraints` travels with the certificate and the codegen keys
//! the theorem it emits off that field rather than off the verdict.
//!
//! # What the float solver is (and is not) allowed to decide
//!
//! The SDP proposes two things: a numeric bound `γ̃` and a Gram matrix `G̃`.
//! Neither is trusted. `γ̃` is only used to *pick candidate rationals* to try,
//! and `G̃` only to choose values for the free parameters of the exact
//! coefficient-matching system (see [`crate::round_gram`]). The emitted `γ` is
//! an exact rational, the emitted `G` satisfies the identity exactly over ℚ,
//! and both are rechecked here before anything is written.
//!
//! So a broken SDP cannot produce a false certificate — only no certificate.

use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};

use crate::emit::{CertMeta, EmitError};
use crate::ldlt::ldlt;
use crate::rational::Rat;
use crate::round_gram::{BlockSpec, RoundError, round_gram_blocks};
use crate::schema::{
    Binding, Candidate, Certificate, Entry, Neighborhood, PolyTerm, PolynomialSpec, Problem,
    SCHEMA_TAG, SosBlock, SparseMatrix, Toolchain, VALIDATED_LEAN, VALIDATED_MATHLIB, Witnesses,
};

/// Neutral input: an SOS relaxation of a polynomial minimization, in `f64`,
/// exactly as POUNCE's SDP produces it. `constraints` empty is the
/// unconstrained case.
#[derive(Clone, Debug)]
pub struct SosInput {
    pub n: usize,
    /// The objective as `(exponent vector, coefficient)` pairs; exponent vectors
    /// of length `n`.
    pub terms: Vec<(Vec<usize>, f64)>,
    /// The monomial lift `m(x)` the SDP used, as exponent vectors of length `n`.
    pub basis: Vec<Vec<usize>>,
    /// The SDP's Gram matrix, `basis.len()` square. A *hint* only.
    pub gram_float: Vec<Vec<f64>>,
    /// The feasible set, one entry per `gₖ(x) ≥ 0`, with the localizing block
    /// the relaxation proposed for each. Empty for an unconstrained problem,
    /// which is then certified over all of `ℝⁿ`.
    pub constraints: Vec<SosConstraint>,
    /// The SDP's numeric lower bound. A *hint* only — it seeds the search for an
    /// exact `γ`, and never becomes the emitted bound itself.
    pub bound_float: f64,
    /// The local solve's `x*`, if there was one. A *hint* only: it seeds the
    /// search for a rational point that attains `γ` exactly, which upgrades the
    /// verdict from a bound to a global minimum. An empty vector, a wrong point,
    /// or a minimizer at irrational coordinates all simply leave the bound
    /// unattained — never a wrong verdict.
    pub x_float: Vec<f64>,
    /// The ball to restrict the claim to, for a *local* certificate. `None`
    /// leaves the claim over all of `K` (or all of `ℝⁿ`, unconstrained).
    pub neighborhood: Option<Ball>,
    /// Try to strengthen a local minimum to a *strict* one by certifying
    /// quadratic growth around it.
    ///
    /// A request, not a demand: the emitter searches a ladder of moduli and
    /// falls back to the plain `local-min` verdict if none of them certifies,
    /// exactly as it falls back from `local-min` to `local-lower-bound` when no
    /// rational point attains the bound. Setting it can only add strength to
    /// the verdict, never change what the certificate is about.
    ///
    /// Ignored without a `neighborhood` or without an attaining point — growth
    /// is measured from the minimizer, so there is nothing to measure from.
    pub growth: bool,
}

/// The closed ball `‖x − center‖² ≤ radius_sq` a local claim is made on, with
/// the localizing block the relaxation proposed for it.
///
/// Unlike `gram_float` and `bound_float`, `center` and `radius_sq` are **not**
/// hints. They are converted losslessly to ℚ and become part of the claim, so a
/// different center is a different theorem — not a failed search. `basis` and
/// `gram_float` are hints like every other Gram.
///
/// The ball is deliberately not passed as one more [`SosConstraint`]. It would
/// then need a term list, which is the same polynomial written a second way,
/// and nothing would force the two spellings to agree; here the expansion has
/// exactly one source ([`ball_terms`]).
#[derive(Clone, Debug)]
pub struct Ball {
    /// Center, length `n`. Need not be the minimizer — only close enough to it
    /// to contain it.
    pub center: Vec<f64>,
    /// The *squared* radius; must be strictly positive. A zero radius makes the
    /// local claim true and vacuous, so the emitter refuses it.
    pub radius_sq: f64,
    /// The localizing monomial lift for this block, as exponent vectors.
    pub basis: Vec<Vec<usize>>,
    /// The SDP's Gram matrix for this block. A hint.
    pub gram_float: Vec<Vec<f64>>,
}

/// One inequality of the feasible set, `g(x) ≥ 0`, with the localizing SOS
/// block the relaxation proposed to multiply it by.
///
/// `g` is part of the *problem*: it is converted losslessly and serialized, and
/// the emitted bound is only claimed where it holds. `basis` and `gram_float`
/// are part of the *witness search*, and are hints — a wrong Gram costs a
/// certificate, never a wrong one.
#[derive(Clone, Debug)]
pub struct SosConstraint {
    /// `g` as `(exponent vector, coefficient)` pairs, exponents of length `n`.
    pub g: Vec<(Vec<usize>, f64)>,
    /// The localizing monomial lift, as exponent vectors of length `n`.
    pub basis: Vec<Vec<usize>>,
    /// The SDP's Gram matrix for this block, `basis.len()` square. A hint.
    pub gram_float: Vec<Vec<f64>>,
}

/// Denominators tried for the free parameters of the coefficient-matching
/// system, coarsest first.
///
/// Coarse grids are tried first because a coarse rational that works is
/// strictly better: it makes a smaller Lean term, and — more to the point — a
/// tight SOS bound puts `G` on the boundary of the PSD cone, where the feasible
/// set is often a single point with small denominators. Fine grids only help
/// when the certificate genuinely has ugly coordinates.
const DENOMS: [i64; 8] = [1, 2, 3, 4, 8, 16, 100, 1000];

/// Denominators tried for `γ` itself, coarsest first. Same reasoning, plus one
/// more: `1/3` is a common exact optimum that no power-of-two grid can express,
/// so the ladder is not purely binary.
const GAMMA_DENOMS: [i64; 9] = [1, 2, 3, 4, 6, 8, 12, 100, 10000];

/// Growth moduli tried when strengthening a local minimum to a strict one,
/// **largest first** — a bigger `μ` is a strictly stronger claim, and every
/// value in the ladder proves the same strict inequality, so the search stops
/// at the first that certifies rather than optimizing further.
///
/// Geometric, and reaching well below 1, because `μ` carries the units of the
/// objective divided by squared distance: a well-scaled problem lands near the
/// top of the ladder and a badly scaled one still finds *something*, which is
/// all strictness needs. The largest certifiable modulus is bounded by the
/// smallest eigenvalue of the reduced Hessian, so a flat-bottomed minimum is
/// expected to exhaust the ladder — a genuine "not strict enough to certify"
/// rather than a search failure, and the plain `local-min` verdict is the
/// honest fallback.
const MU_LADDER: [(i64, i64); 11] = [
    (8, 1),
    (4, 1),
    (2, 1),
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 64),
    (1, 256),
    (1, 4096),
];

/// Why an SOS certificate could not be emitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SosEmitError {
    /// A vector length disagreed with `n` or with the basis size.
    Shape(&'static str),
    /// A coefficient, bound, or Gram entry was `±inf`/`NaN`.
    NonFinite,
    /// No `(γ, grid)` pair in the ladder produced an exact PSD Gram. `last` is
    /// the failure from the final attempt, which is usually the informative one.
    NoExactCertificate { last: RoundError },
    /// Defensive: the assembled certificate failed its own exact recheck.
    SelfCheck(&'static str),
}

impl std::fmt::Display for SosEmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SosEmitError::Shape(s) => write!(f, "shape: {s}"),
            SosEmitError::NonFinite => write!(f, "a value that must be finite was inf/NaN"),
            SosEmitError::NoExactCertificate { last } => write!(
                f,
                "no exact rational SOS certificate near the SDP's bound (last attempt: {last}); \
                 the relaxation may be inexact, or the certificate may need a finer grid"
            ),
            SosEmitError::SelfCheck(s) => write!(f, "self-check failed: {s}"),
        }
    }
}
impl std::error::Error for SosEmitError {}

impl From<SosEmitError> for EmitError {
    /// Collapse into the CLI's error type. The detail survives via `Display`,
    /// which is what the user sees; `EmitError` stays `PartialEq`-simple.
    fn from(_: SosEmitError) -> EmitError {
        EmitError::SelfCheck("SOS certificate could not be emitted exactly")
    }
}

/// f64 → exact `BigRational`, refusing non-finite input.
fn br(x: f64) -> Result<BigRational, SosEmitError> {
    Rat::from_f64(x)
        .map(|r| r.0)
        .map_err(|_| SosEmitError::NonFinite)
}

/// Re-derive the certificate `problem` block from the polynomial and its
/// feasible set — the SOS analogue of [`crate::emit::problem_block`], and for
/// the same reason: `cert-verify` re-derives this from the consumer's own `.nl`
/// and compares it to the certificate's, so producer and consumer must run
/// *identical* code. Two implementations that agree today are a drift waiting to
/// happen.
///
/// The QP half of [`Problem`] stays absent, not zeroed; see the type's docs.
///
/// `constraints` are the `gₖ(x) ≥ 0`; an empty slice leaves `poly_constraints`
/// absent, which is what an unconstrained certificate has always carried and
/// what `canonical_problem` compares against. Their presence is load-bearing
/// here and not merely informational: it is the difference between a bound over
/// `ℝⁿ` and a bound over `K`.
///
/// `neighborhood` is likewise part of the problem block rather than of the
/// witnesses, for the same reason: it changes what region the bound is claimed
/// on, so it has to be inside what `canonical_problem` compares.
pub fn sos_problem_block(
    n: usize,
    terms: &[(Vec<usize>, f64)],
    constraints: &[Vec<(Vec<usize>, f64)>],
    neighborhood: Option<&Neighborhood>,
) -> Result<Problem, SosEmitError> {
    if terms.is_empty() {
        return Err(SosEmitError::Shape("empty polynomial"));
    }
    if constraints.iter().any(|g| g.is_empty()) {
        return Err(SosEmitError::Shape("a constraint polynomial has no terms"));
    }
    if terms
        .iter()
        .chain(constraints.iter().flatten())
        .any(|(e, _)| e.len() != n)
    {
        return Err(SosEmitError::Shape(
            "a term's exponent vector is not length n",
        ));
    }
    let neighborhood = neighborhood
        .map(|nb| {
            if nb.center.len() != n {
                return Err(SosEmitError::Shape("neighborhood center is not length n"));
            }
            // A zero radius would collapse the ball to a point, making the
            // local claim true and empty; a negative one makes the region empty,
            // so *every* bound holds on it. Both are sound and worthless, which
            // is the worst kind of certificate to emit silently.
            if !nb.radius_sq.inner().is_positive() {
                return Err(SosEmitError::Shape(
                    "neighborhood radius² must be strictly positive",
                ));
            }
            Ok(nb.clone())
        })
        .transpose()?;
    Ok(Problem {
        n_vars: n,
        objective: None,
        var_bounds: None,
        constraints: None,
        polynomial: Some(poly_spec(terms)?),
        poly_constraints: (!constraints.is_empty())
            .then(|| constraints.iter().map(|g| poly_spec(g)).collect())
            .transpose()?,
        neighborhood,
    })
}

/// The squared distance `‖x − c‖² = Σⱼ(xⱼ − cⱼ)²` as an exact term list:
///
/// ```text
///     ‖x − c‖² = (Σⱼ cⱼ²)  −  Σⱼ 2cⱼ·xⱼ  +  Σⱼ xⱼ²
/// ```
///
/// **The one place a squared distance becomes a polynomial.** Two different
/// claims are built on it — the ball a *local* certificate restricts to
/// ([`ball_terms`]) and the growth term a *strict* one subtracts
/// ([`growth_shift`]) — and Lean's `sqdist` is a third spelling of the same
/// thing. If any two of them expanded it differently the identity would be
/// about a different region or a different modulus, so all of them route
/// through here; on the Lean side a mismatch surfaces as a failing `ring`.
///
/// Terms are sorted by exponent vector and exact zeros dropped, matching
/// [`poly_spec`]'s convention: a zero coefficient is a real equation to the
/// coefficient-matching system, not a harmless placeholder.
pub fn sqdist_terms(center: &[BigRational]) -> Vec<(Vec<usize>, BigRational)> {
    let n = center.len();
    let mut terms: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();

    let two = BigRational::from_integer(2.into());
    let mut constant = BigRational::zero();
    for (j, cj) in center.iter().enumerate() {
        constant += cj * cj;

        let mut lin = vec![0usize; n];
        lin[j] = 1;
        *terms.entry(lin).or_insert_with(BigRational::zero) -= &two * cj;

        let mut quad = vec![0usize; n];
        quad[j] = 2;
        *terms.entry(quad).or_insert_with(BigRational::zero) += BigRational::from_integer(1.into());
    }
    terms.insert(vec![0usize; n], constant);

    terms.retain(|_, c| !c.is_zero());
    terms.into_iter().collect()
}

/// The ball `‖x − c‖² ≤ r²` as the polynomial inequality
/// `g(x) = r² − ‖x − c‖² ≥ 0`.
///
/// The certificate stores the ball structurally, so the Rust recheck, the exact
/// rounder, and (mirrored) the Lean codegen all have to expand it. Deriving it
/// from [`sqdist_terms`] is what keeps the three spellings one spelling.
pub fn ball_terms(
    center: &[BigRational],
    radius_sq: &BigRational,
) -> Vec<(Vec<usize>, BigRational)> {
    let n = center.len();
    let mut terms: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();
    for (e, c) in sqdist_terms(center) {
        *terms.entry(e).or_insert_with(BigRational::zero) -= c;
    }
    *terms
        .entry(vec![0usize; n])
        .or_insert_with(BigRational::zero) += radius_sq;

    terms.retain(|_, c| !c.is_zero());
    terms.into_iter().collect()
}

/// `μ·‖x − x₀‖²`, the amount a *strict* certificate subtracts from the
/// objective before certifying a bound on what is left.
///
/// The whole of the growth construction is here: certify `p − γ − μ‖x − x₀‖²`
/// nonnegative on `K ∩ B` and it reads back as `p(x) ≥ p(x₀) + μ‖x − x₀‖²`,
/// which is strict away from `x₀` exactly when `μ > 0`.
fn growth_shift(mu: &BigRational, x0: &[BigRational]) -> Vec<(Vec<usize>, BigRational)> {
    sqdist_terms(x0)
        .into_iter()
        .map(|(e, c)| (e, c * mu))
        .filter(|(_, c)| !c.is_zero())
        .collect()
}

/// The σ₀ Gram hint, corrected for a growth shift of `μ` about `x₀`.
///
/// The float SDP proposed its Gram for `p − γ`, not for `p − γ − μ‖x − x₀‖²`,
/// and handing the stale one to the exact rounder is what makes the growth
/// search fail for every modulus worth having: the free parameters get snapped
/// to values that leave σ₀ off the PSD cone, on every grid, for reasons that
/// have nothing to do with whether the shifted certificate exists.
///
/// The correction is exact rather than heuristic. `‖x − x₀‖²` is *itself* a
/// Gram form in the σ₀ basis — it is `Σⱼ(xⱼ − x₀ⱼ)²`, and each `xⱼ − x₀ⱼ` is a
/// linear combination of the constant monomial and `xⱼ` — so subtracting that
/// form's matrix gives a hint that is right for the shifted problem exactly to
/// the degree the original was right for the unshifted one.
///
/// It is also what gives the ladder its meaning: `G₀ − μQ` stops being PSD once
/// `μ` exceeds what σ₀ has left to give, which is the honest reason to come
/// down a rung.
///
/// `None` when the basis lacks the constant monomial or one of the degree-1
/// monomials, so the form is not representable in it. The search then runs on
/// the uncorrected hint — worse, but a hint is never wrong, only unhelpful.
fn shifted_sigma0_hint(
    basis: &[Vec<usize>],
    gram_float: &[Vec<f64>],
    mu: f64,
    x0: &[f64],
) -> Option<Vec<Vec<f64>>> {
    let n = x0.len();
    let idx = |e: &[usize]| basis.iter().position(|b| b.as_slice() == e);
    let konst = idx(&vec![0usize; n])?;
    let lin: Vec<usize> = (0..n)
        .map(|j| {
            let mut e = vec![0usize; n];
            e[j] = 1;
            idx(&e)
        })
        .collect::<Option<_>>()?;

    let mut g: Vec<Vec<f64>> = gram_float.to_vec();
    for (j, &lj) in lin.iter().enumerate() {
        // (xⱼ − x₀ⱼ)² puts x₀ⱼ² at (c,c), −x₀ⱼ at (c,lⱼ) and (lⱼ,c), 1 at
        // (lⱼ,lⱼ) — and the hint subtracts μ times all of it.
        g[konst][konst] -= mu * x0[j] * x0[j];
        g[konst][lj] += mu * x0[j];
        g[lj][konst] += mu * x0[j];
        g[lj][lj] -= mu;
    }
    Some(g)
}

/// `p − shift` as a term list, with exact cancellations dropped.
fn subtract_terms(
    p: &[(Vec<usize>, BigRational)],
    shift: &[(Vec<usize>, BigRational)],
) -> Vec<(Vec<usize>, BigRational)> {
    let mut terms: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();
    for (e, c) in p {
        *terms.entry(e.clone()).or_insert_with(BigRational::zero) += c;
    }
    for (e, c) in shift {
        *terms.entry(e.clone()).or_insert_with(BigRational::zero) -= c;
    }
    terms.retain(|_, c| !c.is_zero());
    terms.into_iter().collect()
}

/// The polynomials the certificate's SOS blocks multiply, in `multiplier` index
/// order.
///
/// That family is the problem's own `gₖ ≥ 0`, **extended by the ball** when the
/// claim is local. So on a local certificate the last index does not name an
/// entry of `poly_constraints` — it names the neighborhood, which is stored
/// separately and derived here. One rule, stated once, and every consumer that
/// resolves a `multiplier` goes through it.
fn multiplier_terms(problem: &Problem) -> Vec<Vec<(Vec<usize>, BigRational)>> {
    let mut out: Vec<Vec<(Vec<usize>, BigRational)>> = problem
        .poly_constraint_specs()
        .iter()
        .map(spec_terms)
        .collect();
    if let Some(nb) = problem.neighborhood.as_ref() {
        let center: Vec<BigRational> = nb.center.iter().map(|r| r.inner().clone()).collect();
        out.push(ball_terms(&center, nb.radius_sq.inner()));
    }
    out
}

/// A float term list as an exact [`PolynomialSpec`]. Lossless — an f64 *is* a
/// rational — so this never approximates and never needs a tolerance.
fn poly_spec(terms: &[(Vec<usize>, f64)]) -> Result<PolynomialSpec, SosEmitError> {
    Ok(PolynomialSpec {
        terms: terms
            .iter()
            .map(|(e, c)| {
                br(*c).map(|r| PolyTerm {
                    exponents: e.clone(),
                    coeff: Rat(r),
                })
            })
            .collect::<Result<_, _>>()?,
    })
}

/// A [`PolynomialSpec`] as the `(exponents, coefficient)` pairs the rounding
/// kernel and the exact evaluator consume.
fn spec_terms(spec: &PolynomialSpec) -> Vec<(Vec<usize>, BigRational)> {
    spec.terms
        .iter()
        .map(|t| (t.exponents.clone(), t.coeff.inner().clone()))
        .collect()
}

/// Candidate exact bounds near `bound_float`, **sharpest first**.
///
/// Each grid contributes the nearest multiple and the one below; the pooled
/// candidates are then ordered by value, largest first. Ordering by sharpness
/// rather than by grid matters because the two disagree: for `x⁴ − 3x² + 2` the
/// SDP bound is ≈ −0.25 and the tight `−1/4` needs a denominator of 4, while the
/// slack `−1` sits on the coarsest grid. Grid-major order certifies `−1` — true,
/// checkable, and four times weaker than what was available.
///
/// Trying a candidate above the true minimum costs only a failed round: the PSD
/// step refuses it, so the search can afford to be greedy about sharpness.
fn gamma_candidates(bound_float: f64) -> Vec<BigRational> {
    let mut out: Vec<BigRational> = Vec::new();
    for d in GAMMA_DENOMS {
        let scaled = bound_float * d as f64;
        for v in [scaled.round(), scaled.floor()] {
            // `as i64` saturates in Rust, which would silently turn an absurd
            // bound into i64::MAX rather than skipping it.
            if !v.is_finite() || v.abs() >= i64::MAX as f64 {
                continue;
            }
            let g = BigRational::new((v as i64).into(), d.into());
            if !out.contains(&g) {
                out.push(g);
            }
        }
    }
    // Descending, so ties in value (`2/4` and `1/2` are one candidate after
    // normalization) cannot reorder between runs.
    out.sort_by(|a, b| b.cmp(a));
    out
}

/// Denominators tried when snapping the float `x*` to a rational point that
/// attains `γ` exactly.
///
/// Deliberately short. A minimizer either has small rational coordinates or, far
/// more often for a nonconvex polynomial, irrational ones — `x⁴ − 3x² + 2`
/// minimizes at `±√(3/2)`, which no grid will ever hit. Searching harder buys
/// almost nothing and costs a slower refusal on exactly the problems where
/// refusal is the right answer.
const ATTAIN_DENOMS: [i64; 9] = [1, 2, 3, 4, 5, 6, 8, 10, 100];

/// The exact values `p` takes at the *feasible* rational points `x*` snaps to.
///
/// These are the `γ`s worth trying first, for a reason the grid ladder cannot
/// reach: if the minimizer really is the rational `x₀`, the sharpest true bound
/// is exactly `p(x₀)` — a rational with whatever denominator `p` and `x₀`
/// happen to produce, `−14/81` say, which no ladder short of luck will hit.
/// Reaching it is the difference between a bound and a minimum.
///
/// **Trying these first cannot cost sharpness**, which is why they may jump the
/// queue. Suppose `γ = p(x₀)` survives the exact rounding: then `γ ≤ p(x)` for
/// every feasible `x`. But `x₀` is itself feasible — that is what the `g` check
/// here is for — so `γ = p(x₀) ≥ inf p`. The two together force `γ = inf p`, the
/// sharpest bound that exists. A candidate from an infeasible point carries no
/// such guarantee and could be provable *and* slack, so those are dropped rather
/// than merely ranked.
///
/// Everything else is free: a `γ` above the infimum fails the coefficient
/// matching or the PSD step, and the search moves on.
fn attained_gamma_candidates(
    p_terms: &[(Vec<usize>, BigRational)],
    g_terms: &[Vec<(Vec<usize>, BigRational)>],
    n: usize,
    x_float: &[f64],
) -> Vec<BigRational> {
    if x_float.len() != n || x_float.iter().any(|v| !v.is_finite()) {
        return Vec::new();
    }
    let mut out: Vec<BigRational> = Vec::new();
    for d in ATTAIN_DENOMS {
        let snapped: Option<Vec<BigRational>> = x_float
            .iter()
            .map(|v| {
                let scaled = (v * d as f64).round();
                (scaled.is_finite() && scaled.abs() < i64::MAX as f64)
                    .then(|| BigRational::new((scaled as i64).into(), d.into()))
            })
            .collect();
        let Some(x) = snapped else { continue };
        if g_terms.iter().any(|g| eval_poly(g, &x).is_negative()) {
            continue;
        }
        let v = eval_poly(p_terms, &x);
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// Look for an exact rational `x₀` with `p(x₀) = γ`, seeded by the float `x*`.
///
/// A proven bound that is *attained* is a global minimum — the Lean step is one
/// rewrite. So this is the whole difference between certifying `γ ≤ p(x)` and
/// certifying that a specific point minimizes a nonconvex polynomial globally,
/// which no KKT argument can deliver.
///
/// Every candidate is checked by evaluating `p` exactly over ℚ and comparing to
/// `γ` for equality — never within a tolerance. A near miss is not a minimizer,
/// and accepting one would emit a certificate that cannot verify. Returning
/// `None` is the honest outcome and costs only sharpness: the bound still holds.
///
/// On a constrained problem the bound holds on `K` only, so a point that attains
/// `γ` outside `K` witnesses nothing — every `gₖ(x₀) ≥ 0` is checked exactly
/// too. Note the asymmetry with the objective: attainment is an equation and
/// feasibility an inequality, so snapping `x*` to the grid can *create*
/// feasibility it did not have, but never attainment.
fn attaining_point(
    p_terms: &[(Vec<usize>, BigRational)],
    g_terms: &[Vec<(Vec<usize>, BigRational)>],
    gamma: &BigRational,
    n: usize,
    x_float: &[f64],
) -> Option<Vec<BigRational>> {
    if x_float.len() != n || x_float.iter().any(|v| !v.is_finite()) {
        return None;
    }
    for d in ATTAIN_DENOMS {
        let snapped: Option<Vec<BigRational>> = x_float
            .iter()
            .map(|v| {
                let scaled = (v * d as f64).round();
                // `as i64` saturates rather than failing, which would snap an
                // absurd coordinate to i64::MAX instead of skipping this grid.
                (scaled.is_finite() && scaled.abs() < i64::MAX as f64)
                    .then(|| BigRational::new((scaled as i64).into(), d.into()))
            })
            .collect();
        // A coordinate that overflows on a fine grid may be fine on a coarser
        // one, so this skips the grid rather than abandoning the search.
        let Some(x) = snapped else { continue };
        if eval_poly(p_terms, &x) == *gamma
            && g_terms.iter().all(|g| !eval_poly(g, &x).is_negative())
        {
            return Some(x);
        }
    }
    None
}

/// Evaluate `p` at a rational point, exactly.
fn eval_poly(p_terms: &[(Vec<usize>, BigRational)], x: &[BigRational]) -> BigRational {
    let mut acc = BigRational::zero();
    for (exps, coeff) in p_terms {
        let mut term = coeff.clone();
        for (i, &e) in exps.iter().enumerate() {
            for _ in 0..e {
                term *= &x[i];
            }
        }
        acc += term;
    }
    acc
}

/// Build an exact SOS certificate, or refuse.
///
/// Searches the `(γ, grid)` ladder for an exact rational certificate, then
/// rechecks the assembled result — identity and PSD — *from the serialized
/// sparse form*, not from the dense intermediate. The recheck is not redundant
/// with [`round_gram`]'s: that one validates the dense matrix it computed, this
/// one validates the bytes that will actually be written, so a bug in the
/// sparsification cannot slip through.
#[allow(clippy::needless_range_loop)] // index loops cross-reference gram/basis
pub fn emit_sos_certificate(
    input: &SosInput,
    meta: &CertMeta,
) -> Result<Certificate, SosEmitError> {
    let n = input.n;
    let shape_ok = |basis: &[Vec<usize>], g: &[Vec<f64>]| {
        let bn = basis.len();
        bn > 0
            && basis.iter().all(|e| e.len() == n)
            && g.len() == bn
            && g.iter().all(|r| r.len() == bn)
    };
    if input.basis.is_empty() {
        return Err(SosEmitError::Shape("empty monomial basis"));
    }
    if !shape_ok(&input.basis, &input.gram_float) {
        return Err(SosEmitError::Shape("σ₀ basis or Gram hint has a bad shape"));
    }
    if input
        .constraints
        .iter()
        .any(|c| !shape_ok(&c.basis, &c.gram_float))
    {
        return Err(SosEmitError::Shape(
            "a localizing basis or Gram hint has a bad shape",
        ));
    }
    if let Some(b) = input.neighborhood.as_ref()
        && !shape_ok(&b.basis, &b.gram_float)
    {
        return Err(SosEmitError::Shape(
            "the neighborhood's basis or Gram hint has a bad shape",
        ));
    }

    // Exact problem. This — not the float term lists — is what the certificate
    // claims a bound for, and the conversion is lossless, so the claim is about
    // the polynomial and feasible set the `.nl` actually encodes. Built through
    // the same `sos_problem_block` the consumer will re-run, then read back, so
    // the witness search and the emitted block cannot describe different
    // problems.
    let g_float: Vec<Vec<(Vec<usize>, f64)>> =
        input.constraints.iter().map(|c| c.g.clone()).collect();
    // The ball's exact form. f64 → ℚ is lossless, so the emitted claim is about
    // precisely the ball the caller named — no rounding, no tolerance.
    let ball = input
        .neighborhood
        .as_ref()
        .map(|b| -> Result<Neighborhood, SosEmitError> {
            Ok(Neighborhood {
                center: b
                    .center
                    .iter()
                    .map(|c| br(*c).map(Rat))
                    .collect::<Result<_, _>>()?,
                radius_sq: Rat(br(b.radius_sq)?),
            })
        })
        .transpose()?;
    let problem = sos_problem_block(n, &input.terms, &g_float, ball.as_ref())?;
    let p_terms = spec_terms(
        problem
            .polynomial
            .as_ref()
            .ok_or(SosEmitError::SelfCheck("problem block lost its polynomial"))?,
    );
    // The multiplier family: the problem's constraints, then the ball if the
    // claim is local. Derived from the *problem block* rather than from `input`
    // so the witness search runs against the same polynomials that get written.
    let g_terms = multiplier_terms(&problem);
    let expected = input.constraints.len() + usize::from(input.neighborhood.is_some());
    if g_terms.len() != expected {
        return Err(SosEmitError::SelfCheck(
            "problem block lost a constraint polynomial",
        ));
    }

    // Block 0 is σ₀ — multiplier the constant 1 — and block `k+1` multiplies
    // `g_terms[k]`. The order is the certificate's `multiplier` indices, so it
    // must not drift from what is serialized below.
    let one = [(vec![0usize; n], BigRational::from_integer(1.into()))];
    let mut blocks = vec![BlockSpec {
        basis: &input.basis,
        multiplier: &one,
        gram_float: &input.gram_float,
    }];
    for (k, c) in input.constraints.iter().enumerate() {
        blocks.push(BlockSpec {
            basis: &c.basis,
            multiplier: &g_terms[k],
            gram_float: &c.gram_float,
        });
    }
    // The ball's block goes last, so its multiplier index is the one past the
    // problem's constraints — which is exactly how `multiplier_terms` numbers it.
    if let Some(b) = input.neighborhood.as_ref() {
        blocks.push(BlockSpec {
            basis: &b.basis,
            multiplier: &g_terms[input.constraints.len()],
            gram_float: &b.gram_float,
        });
    }

    // --- search the ladder for an exact certificate --------------------------
    let mut last = RoundError::Inconsistent;
    let mut found: Option<(BigRational, Vec<Vec<Vec<BigRational>>>)> = None;
    // Exact values at the snapped `x*` first, then the grid ladder. The former
    // are the only way to reach a `γ` with an awkward denominator, and they are
    // precisely the ones that make the bound attained.
    let candidates = attained_gamma_candidates(&p_terms, &g_terms, n, &input.x_float)
        .into_iter()
        .chain(gamma_candidates(input.bound_float));
    'search: for gamma in candidates {
        for denom in DENOMS {
            match round_gram_blocks(&p_terms, &gamma, &blocks, denom) {
                Ok(g) => {
                    found = Some((gamma, g));
                    break 'search;
                }
                Err(e) => last = e,
            }
        }
    }
    let Some((gamma, grams)) = found else {
        return Err(SosEmitError::NoExactCertificate { last });
    };
    if grams.len() != blocks.len() {
        return Err(SosEmitError::SelfCheck(
            "rounding returned the wrong block count",
        ));
    }

    // --- try to upgrade the bound to a minimum ------------------------------
    // If some rational `x₀` hits `γ` exactly — and lies in `K`, which on a
    // constrained problem is half the claim — the bound is attained and the
    // stronger verdict is available; otherwise the bound stands alone. Both are
    // sound, and the emitter never guesses between them: attainment is decided
    // by exact evaluation, not by how close the float solve looked.
    //
    // This runs before the blocks are sparsified because the growth search
    // below may replace them wholesale: a strict certificate's blocks are the
    // ones that close the *shifted* identity, not this one.
    let attained = attaining_point(&p_terms, &g_terms, &gamma, n, &input.x_float);
    let local = input.neighborhood.is_some();

    // --- try to upgrade the minimum to a STRICT one -------------------------
    // Certifying `p − γ − μ‖x − x₀‖²` nonnegative on `K ∩ B` instead of `p − γ`
    // proves quadratic growth, hence strictness away from `x₀`. It is the same
    // Putinar machinery on a shifted objective, so the whole search is the one
    // above with a different polynomial — no new solver, no new witness kind,
    // and the same exact recheck.
    //
    // `γ` does not move: the shift vanishes at `x₀`, so `p(x₀) − γ − 0 = 0`
    // still holds and the bound is still attained there. Only the identity gets
    // harder, which is the entire content of the stronger claim.
    //
    // Restricted to the local case. Growth around a *global* minimum is equally
    // meaningful, but it is a different verdict proved by a different theorem,
    // and emitting `local-min-strict` for an unrestricted claim would
    // misdescribe the region — so the unrestricted case keeps what it earns.
    let mut growth_mu: Option<BigRational> = None;
    let mut grams = grams;
    if input.growth
        && local
        && let Some(x0) = attained.as_ref()
    {
        // For the hint only, so a coordinate that does not round-trip costs
        // sharpness rather than correctness.
        let x0f: Vec<f64> = x0.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
        let usable_hint = x0f.iter().all(|v| v.is_finite());
        'growth: for (num, den) in MU_LADDER {
            let mu = BigRational::new(num.into(), den.into());
            let shifted = subtract_terms(&p_terms, &growth_shift(&mu, x0));
            let hint = usable_hint
                .then(|| {
                    shifted_sigma0_hint(
                        &input.basis,
                        &input.gram_float,
                        num as f64 / den as f64,
                        &x0f,
                    )
                })
                .flatten();
            let mut shifted_blocks = blocks.clone();
            if let Some(h) = hint.as_ref() {
                shifted_blocks[0].gram_float = h;
            }
            for denom in DENOMS {
                if let Ok(g) = round_gram_blocks(&shifted, &gamma, &shifted_blocks, denom) {
                    growth_mu = Some(mu);
                    grams = g;
                    break 'growth;
                }
            }
        }
    }
    if grams.len() != blocks.len() {
        return Err(SosEmitError::SelfCheck(
            "the growth search returned the wrong block count",
        ));
    }

    // --- factor each block for its PSD witness, and sparsify -----------------
    let mut out_blocks: Vec<SosBlock> = Vec::with_capacity(grams.len());
    for (k, gram) in grams.iter().enumerate() {
        // `round_gram_blocks` already refused a non-PSD block via this same
        // factorization, so a failure here would mean the matrix changed
        // underneath us.
        let ldl = ldlt(gram).map_err(|_| SosEmitError::SelfCheck("Gram lost its LDLᵀ"))?;
        if ldl.d.iter().any(|v| v.is_negative()) {
            return Err(SosEmitError::SelfCheck("LDLᵀ diagonal is not nonnegative"));
        }
        let bn = gram.len();
        let mut gram_entries: Vec<Entry> = Vec::new();
        for i in 0..bn {
            for j in 0..=i {
                if !gram[i][j].is_zero() {
                    gram_entries.push(Entry {
                        i,
                        j,
                        val: Rat(gram[i][j].clone()),
                    });
                }
            }
        }
        out_blocks.push(SosBlock {
            monomials: blocks[k].basis.to_vec(),
            gram: SparseMatrix::symmetric(bn, bn, gram_entries),
            l: SparseMatrix::unit_lower(
                bn,
                bn,
                ldl.l_below
                    .iter()
                    .filter(|(_, _, v)| !v.is_zero())
                    .map(|(i, j, v)| Entry {
                        i: *i,
                        j: *j,
                        val: Rat(v.clone()),
                    })
                    .collect(),
            ),
            d: ldl.d.iter().cloned().map(Rat).collect(),
            // Block 0 is σ₀ (no multiplier); block k multiplies constraint k−1.
            multiplier: (k > 0).then(|| k - 1),
        });
    }

    // --- exact self-check of what will be written ---------------------------
    // Checked against the *shifted* left side when the growth search succeeded,
    // since that is the identity the blocks close and the one Lean will be
    // handed. Rechecking against `p − γ` there would reject a correct strict
    // certificate — and, worse, wave through a plain one carrying a stray `μ`.
    let growth_check = growth_mu
        .as_ref()
        .zip(attained.as_ref())
        .map(|(mu, x0)| (mu, x0.as_slice()));
    recheck(&problem, &gamma, growth_check, &out_blocks)?;

    // --- the verdict this earned --------------------------------------------
    // The emitter never guesses between the four: attainment is decided by
    // exact evaluation and growth by whether a modulus certified, and each
    // failure falls back exactly one step. Every step is sound.
    //
    // The `local` prefix is not cosmetic. It travels with `problem.neighborhood`
    // and says the same thing twice on purpose: a reader skimming verdicts and a
    // consumer keying off the problem block should not be able to disagree about
    // whether the claim is restricted to a ball. `-strict` travels with
    // `growth_modulus` for the same reason.
    let (verdict, candidate) = match attained {
        Some(x) => (
            match (local, growth_mu.is_some()) {
                (true, true) => "local-min-strict",
                (true, false) => "local-min",
                (false, _) => "global-min",
            },
            Some(Candidate {
                x: x.into_iter().map(Rat).collect(),
                objective: Rat(gamma.clone()),
            }),
        ),
        None => (
            if local {
                "local-lower-bound"
            } else {
                "global-lower-bound"
            },
            None,
        ),
    };

    Ok(Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: verdict.to_string(),
        problem_class: "sos-poly".to_string(),
        tolerance: Rat(BigRational::zero()),
        bound: Some(Rat(gamma)),
        growth_modulus: growth_mu.map(Rat),
        binding: Binding {
            nl_sha256: meta.nl_sha256.clone(),
            sol_sha256: meta.sol_sha256.clone(),
            solver: meta.solver.clone(),
        },
        toolchain: Toolchain {
            lean: VALIDATED_LEAN.to_string(),
            mathlib: VALIDATED_MATHLIB.to_string(),
        },
        problem,
        candidate,
        witnesses: Witnesses {
            duals: None,
            hessian_psd: None,
            active_set: None,
            farkas: None,
            recession: None,
            sos: Some(out_blocks),
            feasible_witness: None,
        },
    })
}

/// Re-derive every proof obligation from the serialized blocks alone, the way
/// the Lean codegen will read them:
///
/// 1. `p(x) − γ = Σₖ mₖ(x)ᵀ Gₖ mₖ(x)·gₖ(x)` as an identity of coefficient
///    lists, with `g₀ ≡ 1`, and
/// 2. per block, `G = L·diag(D)·Lᵀ` with `D ≥ 0`.
///
/// `growth = Some((μ, x₀))` moves the left side to `p(x) − γ − μ‖x − x₀‖²`, the
/// identity a strict certificate closes. It is the *only* difference between
/// the two kinds here, which is the point: strictness costs a shifted objective
/// and nothing else.
///
/// Checking (2) is what makes shipping both `gram` and `L`/`D` safe rather than
/// merely redundant: they are two independent descriptions of the same matrix,
/// and here they are forced to agree before either is written.
///
/// (1) is checked against the blocks' *own* `multiplier` indices rather than
/// against their position, so a block pointing at the wrong constraint fails
/// here — where the diagnosis is one line — instead of in the kernel.
#[allow(clippy::needless_range_loop)] // index loops cross-reference g, l, d
fn recheck(
    problem: &Problem,
    gamma: &BigRational,
    growth: Option<(&BigRational, &[BigRational])>,
    blocks: &[SosBlock],
) -> Result<(), SosEmitError> {
    let n = problem.n_vars;
    // Resolved through `multiplier_terms`, so the recheck reads a `multiplier`
    // index exactly as the emitter wrote it — including the local case, where
    // the last index names the ball rather than a constraint.
    let cons = multiplier_terms(problem);
    let one = vec![(vec![0usize; n], BigRational::from_integer(1.into()))];

    // The identity's left side, accumulated over every block.
    let mut lhs: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();

    let mut seen: Vec<bool> = vec![false; cons.len()];
    for block in blocks {
        let bn = block.monomials.len();

        // Rebuild the dense Gram from the sparse symmetric block.
        let mut g = vec![vec![BigRational::zero(); bn]; bn];
        for e in &block.gram.entries {
            if e.i >= bn || e.j >= bn {
                return Err(SosEmitError::SelfCheck("Gram entry out of range"));
            }
            g[e.i][e.j] = e.val.inner().clone();
            g[e.j][e.i] = e.val.inner().clone();
        }
        // ... and the dense unit-lower L.
        let mut l = vec![vec![BigRational::zero(); bn]; bn];
        for (k, row) in l.iter_mut().enumerate() {
            row[k] = BigRational::from_integer(1.into());
        }
        for e in &block.l.entries {
            if e.i <= e.j || e.i >= bn {
                return Err(SosEmitError::SelfCheck(
                    "L entry is not strictly below the diagonal",
                ));
            }
            l[e.i][e.j] = e.val.inner().clone();
        }
        let d: Vec<BigRational> = block.d.iter().map(|r| r.inner().clone()).collect();
        if d.len() != bn {
            return Err(SosEmitError::SelfCheck("D length != basis size"));
        }
        if d.iter().any(|v| v.is_negative()) {
            return Err(SosEmitError::SelfCheck("D has a negative entry"));
        }

        // (2) L·diag(D)·Lᵀ == G.
        for i in 0..bn {
            for j in 0..bn {
                let v: BigRational = (0..bn).map(|k| &l[i][k] * &d[k] * &l[j][k]).sum();
                if v != g[i][j] {
                    return Err(SosEmitError::SelfCheck("L·diag(D)·Lᵀ != G"));
                }
            }
        }

        // Which polynomial this block multiplies. A `multiplier` that names a
        // constraint the problem does not have would otherwise silently drop
        // the whole block from the identity.
        let mult = match block.multiplier {
            None => one.clone(),
            Some(k) => {
                let terms = cons
                    .get(k)
                    .ok_or(SosEmitError::SelfCheck("block.multiplier is out of range"))?;
                if seen[k] {
                    return Err(SosEmitError::SelfCheck(
                        "two blocks claim the same constraint",
                    ));
                }
                seen[k] = true;
                terms.clone()
            }
        };

        // (1), this block's contribution: expand m(x)ᵀ G m(x)·g(x).
        for i in 0..bn {
            for j in 0..bn {
                if g[i][j].is_zero() {
                    continue;
                }
                let base: Vec<usize> = block.monomials[i]
                    .iter()
                    .zip(&block.monomials[j])
                    .map(|(a, b)| a + b)
                    .collect();
                for (beta, cbeta) in &mult {
                    if beta.len() != n {
                        return Err(SosEmitError::SelfCheck(
                            "a multiplier term is not length n_vars",
                        ));
                    }
                    let alpha: Vec<usize> = base.iter().zip(beta).map(|(a, b)| a + b).collect();
                    *lhs.entry(alpha).or_insert_with(BigRational::zero) += &g[i][j] * cbeta;
                }
            }
        }
    }
    // Every constraint must carry a multiplier, even a zero one: a missing
    // block is not an error of arithmetic — the identity still closes — but it
    // means the certificate's block list does not describe the problem's, and
    // the codegen builds its `Σᵢ σᵢ·gᵢ` from that list.
    if !seen.iter().all(|s| *s) {
        return Err(SosEmitError::SelfCheck(
            "a constraint has no localizing block",
        ));
    }

    let mut rhs: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();
    let spec = problem
        .polynomial
        .as_ref()
        .ok_or(SosEmitError::SelfCheck("sos-poly cert has no polynomial"))?;
    for t in &spec.terms {
        *rhs.entry(t.exponents.clone())
            .or_insert_with(BigRational::zero) += t.coeff.inner();
    }
    *rhs.entry(vec![0usize; n]).or_insert_with(BigRational::zero) -= gamma;
    if let Some((mu, x0)) = growth {
        if !mu.is_positive() {
            return Err(SosEmitError::SelfCheck("growth modulus is not positive"));
        }
        if x0.len() != n {
            return Err(SosEmitError::SelfCheck(
                "growth center is not length n_vars",
            ));
        }
        for (e, c) in growth_shift(mu, x0) {
            *rhs.entry(e).or_insert_with(BigRational::zero) -= c;
        }
    }

    lhs.retain(|_, v| !v.is_zero());
    rhs.retain(|_, v| !v.is_zero());
    if lhs != rhs {
        return Err(SosEmitError::SelfCheck(if growth.is_some() {
            "p − γ − μ‖x − x₀‖² != σ₀ + Σᵢ σᵢ·gᵢ"
        } else {
            "p − γ != σ₀ + Σᵢ σᵢ·gᵢ"
        }));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn meta() -> CertMeta {
        CertMeta {
            nl_sha256: "0".repeat(64),
            sol_sha256: "1".repeat(64),
            solver: "pounce test".to_string(),
        }
    }

    /// `p = x⁴ − 2x² + 2`, minimum 1, certified by `(x² − 1)²`.
    fn quartic() -> SosInput {
        SosInput {
            n: 1,
            terms: vec![(vec![4], 1.0), (vec![2], -2.0), (vec![0], 2.0)],
            basis: vec![vec![0], vec![1], vec![2]],
            // A plausible interior-point output: near the answer, not equal.
            gram_float: vec![
                vec![1.0000000003, 1e-11, -0.9999999997],
                vec![1e-11, 2.4e-10, -3e-11],
                vec![-0.9999999997, -3e-11, 1.0000000002],
            ],
            bound_float: 0.9999999997,
            // No iterate: the bound path, uncontaminated by attainment.
            x_float: vec![],
            constraints: Vec::new(),
            neighborhood: None,
            growth: false,
        }
    }

    fn rat(c: &Rat) -> BigRational {
        c.inner().clone()
    }

    #[test]
    fn certifies_the_quartic_with_the_tight_exact_bound() {
        let cert = emit_sos_certificate(&quartic(), &meta()).unwrap();
        assert_eq!(cert.verdict, "global-lower-bound");
        assert_eq!(cert.problem_class, "sos-poly");
        // Not 0.9999999997: the emitted bound is exact, and tight.
        assert_eq!(
            rat(cert.bound.as_ref().unwrap()),
            BigRational::from_integer(1.into())
        );
        assert!(rat(&cert.tolerance).is_zero(), "certs claim tolerance = 0");
        assert!(cert.candidate.is_none(), "a bound is not a point claim");
    }

    /// Hand the same quartic an iterate, and the verdict strengthens: `γ = 1` is
    /// a lower bound *and* `p(1) = 1`, so 1 is the global minimum. The iterate is
    /// the solver's, off by ~4e-10 — it is snapped and then checked exactly.
    #[test]
    fn an_iterate_that_attains_the_bound_upgrades_it_to_a_global_minimum() {
        let input = SosInput {
            x_float: vec![0.99999999962],
            ..quartic()
        };
        let cert = emit_sos_certificate(&input, &meta()).unwrap();
        assert_eq!(cert.verdict, "global-min");
        let cand = cert.candidate.as_ref().expect("a minimum exhibits a point");
        assert_eq!(rat(&cand.x[0]), BigRational::from_integer(1.into()));
        // The claimed objective is γ itself, not a re-evaluation that might
        // differ: attainment is what makes the two the same number.
        assert_eq!(rat(&cand.objective), rat(cert.bound.as_ref().unwrap()));
    }

    /// The other basin. `x⁴ − 2x² + 2` minimizes at both `±1`; which one the
    /// local solve reports is arbitrary, and either is a valid minimizer.
    #[test]
    fn the_other_minimizer_certifies_the_same_minimum() {
        let cert = emit_sos_certificate(
            &SosInput {
                x_float: vec![-1.0000000004],
                ..quartic()
            },
            &meta(),
        )
        .unwrap();
        assert_eq!(cert.verdict, "global-min");
        assert_eq!(
            rat(&cert.candidate.as_ref().unwrap().x[0]),
            BigRational::from_integer((-1).into())
        );
    }

    /// A minimizer at `±√(3/2)` is irrational, so no grid attains it and the
    /// verdict must stay a bound. This is the failure mode that would be
    /// dangerous if attainment were tested within a tolerance: `x = 1.2247` is
    /// arbitrarily close and is not a minimizer.
    #[test]
    fn an_irrational_minimizer_leaves_the_verdict_a_bound() {
        // p = x⁴ − 3x² + 2; p + 1/4 = (x² − 3/2)², so γ = −1/4.
        let cert = emit_sos_certificate(
            &SosInput {
                n: 1,
                terms: vec![(vec![4], 1.0), (vec![2], -3.0), (vec![0], 2.0)],
                basis: vec![vec![0], vec![1], vec![2]],
                gram_float: vec![
                    vec![2.2500000003, 1e-11, -1.4999999997],
                    vec![1e-11, 3.1e-10, -2e-11],
                    vec![-1.4999999997, -2e-11, 1.0000000002],
                ],
                bound_float: -0.2500000003,
                x_float: vec![1.2247448713915892],
                constraints: Vec::new(),
                neighborhood: None,
                growth: false,
            },
            &meta(),
        )
        .unwrap();
        assert_eq!(cert.verdict, "global-lower-bound");
        assert!(cert.candidate.is_none());
        assert_eq!(
            rat(cert.bound.as_ref().unwrap()),
            BigRational::new((-1).into(), 4.into())
        );
    }

    /// A hint pointing somewhere else costs sharpness, never soundness: the
    /// bound is still the true minimum, it is just not exhibited as attained.
    #[test]
    fn a_wrong_hint_degrades_to_a_bound_rather_than_a_wrong_point() {
        let cert = emit_sos_certificate(
            &SosInput {
                x_float: vec![7.0],
                ..quartic()
            },
            &meta(),
        )
        .unwrap();
        assert_eq!(cert.verdict, "global-lower-bound");
        assert!(cert.candidate.is_none());
    }

    /// A hint of the wrong length is a caller bug, not a certificate: it must be
    /// ignored rather than indexed into.
    #[test]
    fn a_hint_of_the_wrong_arity_is_ignored() {
        let cert = emit_sos_certificate(
            &SosInput {
                x_float: vec![1.0, 1.0],
                ..quartic()
            },
            &meta(),
        )
        .unwrap();
        assert_eq!(cert.verdict, "global-lower-bound");
    }

    /// The QP half must be *absent*, not zeroed — a zero `Q` would assert the
    /// quartic is a linear program (see [`crate::schema::Problem`]).
    #[test]
    fn the_qp_half_of_the_problem_block_is_absent() {
        let cert = emit_sos_certificate(&quartic(), &meta()).unwrap();
        assert!(cert.problem.objective.is_none());
        assert!(cert.problem.var_bounds.is_none());
        assert!(cert.problem.constraints.is_none());
        assert!(cert.problem.polynomial.is_some());
    }

    #[test]
    fn the_witness_is_the_expected_sos_decomposition() {
        let cert = emit_sos_certificate(&quartic(), &meta()).unwrap();
        let blocks = cert.witnesses.sos.as_ref().unwrap();
        assert_eq!(blocks.len(), 1, "v1 emits the unconstrained one-block case");
        let b = &blocks[0];
        assert_eq!(b.monomials, vec![vec![0], vec![1], vec![2]]);
        // G = [[1,0,-1],[0,0,0],[-1,0,1]], lower triangle, zeros dropped.
        let got: Vec<(usize, usize, BigRational)> = b
            .gram
            .entries
            .iter()
            .map(|e| (e.i, e.j, rat(&e.val)))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, 0, BigRational::from_integer(1.into())),
                (2, 0, BigRational::from_integer((-1).into())),
                (2, 2, BigRational::from_integer(1.into())),
            ]
        );
        // D = (1, 0, 0) — rank one, as (x² − 1)² is a single square.
        assert_eq!(b.d.len(), 3);
        assert_eq!(rat(&b.d[0]), BigRational::from_integer(1.into()));
        assert!(rat(&b.d[1]).is_zero() && rat(&b.d[2]).is_zero());
        assert_eq!(b.l.entries.len(), 1);
        assert_eq!(b.l.entries[0].i, 2);
        assert_eq!(b.l.entries[0].j, 0);
    }

    /// A bound the polynomial does not satisfy must not be certified even if the
    /// float solver insists on it — the exact PSD check is the one that decides.
    #[test]
    fn a_bound_above_the_true_minimum_is_refused() {
        let mut input = quartic();
        // Claim 3 for a polynomial whose minimum is 1. Every grid in the ladder
        // divides 3 exactly, so *every* candidate is γ = 3 — the search cannot
        // rescue the claim by weakening it, and must refuse outright.
        input.bound_float = 3.0;
        let err = emit_sos_certificate(&input, &meta()).unwrap_err();
        assert!(
            matches!(err, SosEmitError::NoExactCertificate { .. }),
            "got {err:?}"
        );
    }

    /// A polynomial whose monomials the basis cannot reach is refused, not
    /// silently certified for a truncation of itself.
    #[test]
    fn a_basis_too_small_for_the_polynomial_is_refused() {
        let mut input = quartic();
        input.basis = vec![vec![0], vec![1]]; // (1, x) cannot form x⁴
        input.gram_float = vec![vec![0.0; 2]; 2];
        let err = emit_sos_certificate(&input, &meta()).unwrap_err();
        assert!(
            matches!(err, SosEmitError::NoExactCertificate { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn mismatched_shapes_are_refused() {
        let mut input = quartic();
        input.n = 2; // exponent vectors are still length 1
        assert!(matches!(
            emit_sos_certificate(&input, &meta()),
            Err(SosEmitError::Shape(_))
        ));
    }

    #[test]
    fn a_non_finite_coefficient_is_refused() {
        let mut input = quartic();
        input.terms[0].1 = f64::NAN;
        assert_eq!(
            emit_sos_certificate(&input, &meta()).unwrap_err(),
            SosEmitError::NonFinite
        );
    }

    /// The ladder must prefer the *tight* bound, not merely a valid one. A
    /// certificate for γ = 0 would be true and useless.
    #[test]
    fn the_ladder_prefers_the_nearest_candidate() {
        let cands = gamma_candidates(0.9999999997);
        assert_eq!(
            cands[0],
            BigRational::from_integer(1.into()),
            "the sharpest candidate comes first"
        );
    }

    /// Sharpness must beat grid coarseness. A bound of ≈ −0.25 puts the tight
    /// `−1/4` on the `d = 4` grid and the slack `−1` on the coarsest one; a
    /// ladder ordered by grid would certify `−1` and never look further.
    #[test]
    fn a_finer_grid_wins_when_it_is_sharper() {
        let cands = gamma_candidates(-0.2500000003);
        let quarter = BigRational::new((-1).into(), 4.into());
        let one = BigRational::from_integer((-1).into());
        let pos = |g: &BigRational| cands.iter().position(|c| c == g).unwrap();
        assert!(pos(&quarter) < pos(&one), "got {cands:?}");
        assert!(
            cands.windows(2).all(|w| w[0] > w[1]),
            "candidates must be strictly descending: {cands:?}"
        );
    }

    /// A `γ` needing a denominator no power-of-two grid reaches.
    #[test]
    fn a_thirds_bound_is_reachable() {
        let cands = gamma_candidates(1.0 / 3.0);
        assert!(
            cands.contains(&BigRational::new(1.into(), 3.into())),
            "1/3 must be a candidate; got {cands:?}"
        );
    }
}
