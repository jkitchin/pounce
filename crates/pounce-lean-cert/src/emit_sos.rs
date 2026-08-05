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
use num_traits::{Signed, Zero};

use crate::emit::{CertMeta, EmitError};
use crate::ldlt::ldlt;
use crate::rational::Rat;
use crate::round_gram::{RoundError, round_gram};
use crate::schema::{
    Binding, Candidate, Certificate, Entry, PolyTerm, PolynomialSpec, Problem, SCHEMA_TAG,
    SosBlock, SparseMatrix, Toolchain, VALIDATED_LEAN, VALIDATED_MATHLIB, Witnesses,
};

/// Neutral input: an SOS relaxation of an unconstrained polynomial minimization,
/// in `f64`, exactly as POUNCE's SDP produces it.
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
    /// The SDP's numeric lower bound. A *hint* only — it seeds the search for an
    /// exact `γ`, and never becomes the emitted bound itself.
    pub bound_float: f64,
    /// The local solve's `x*`, if there was one. A *hint* only: it seeds the
    /// search for a rational point that attains `γ` exactly, which upgrades the
    /// verdict from a bound to a global minimum. An empty vector, a wrong point,
    /// or a minimizer at irrational coordinates all simply leave the bound
    /// unattained — never a wrong verdict.
    pub x_float: Vec<f64>,
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

/// Re-derive the certificate `problem` block from the polynomial alone — the
/// SOS analogue of [`crate::emit::problem_block`], and for the same reason:
/// `cert-verify` re-derives this from the consumer's own `.nl` and compares it
/// to the certificate's, so producer and consumer must run *identical* code.
/// Two implementations that agree today are a drift waiting to happen.
///
/// The QP half of [`Problem`] stays absent, not zeroed; see the type's docs.
pub fn sos_problem_block(n: usize, terms: &[(Vec<usize>, f64)]) -> Result<Problem, SosEmitError> {
    if terms.is_empty() {
        return Err(SosEmitError::Shape("empty polynomial"));
    }
    if terms.iter().any(|(e, _)| e.len() != n) {
        return Err(SosEmitError::Shape(
            "a term's exponent vector is not length n",
        ));
    }
    Ok(Problem {
        n_vars: n,
        objective: None,
        var_bounds: None,
        constraints: None,
        polynomial: Some(PolynomialSpec {
            terms: terms
                .iter()
                .map(|(e, c)| {
                    br(*c).map(|r| PolyTerm {
                        exponents: e.clone(),
                        coeff: Rat(r),
                    })
                })
                .collect::<Result<_, _>>()?,
        }),
    })
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
fn attaining_point(
    p_terms: &[(Vec<usize>, BigRational)],
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
        if eval_poly(p_terms, &x) == *gamma {
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
    let bn = input.basis.len();
    if bn == 0 {
        return Err(SosEmitError::Shape("empty monomial basis"));
    }
    if input.basis.iter().any(|e| e.len() != n) {
        return Err(SosEmitError::Shape("a basis monomial is not length n"));
    }
    if input.gram_float.len() != bn || input.gram_float.iter().any(|r| r.len() != bn) {
        return Err(SosEmitError::Shape("Gram hint is not basis × basis"));
    }

    // Exact objective. This — not the float term list — is what the certificate
    // claims a bound for, and the conversion is lossless, so the claim is about
    // the polynomial the `.nl` actually encodes. Built through the same
    // `sos_problem_block` the consumer will re-run, then read back, so the
    // witness search and the emitted block cannot describe different
    // polynomials.
    let problem = sos_problem_block(n, &input.terms)?;
    let p_terms: Vec<(Vec<usize>, BigRational)> = problem
        .polynomial
        .as_ref()
        .ok_or(SosEmitError::SelfCheck("problem block lost its polynomial"))?
        .terms
        .iter()
        .map(|t| (t.exponents.clone(), t.coeff.inner().clone()))
        .collect();

    // --- search the ladder for an exact certificate --------------------------
    let mut last = RoundError::Inconsistent;
    let mut found: Option<(BigRational, Vec<Vec<BigRational>>)> = None;
    'search: for gamma in gamma_candidates(input.bound_float) {
        for denom in DENOMS {
            match round_gram(&p_terms, &gamma, &input.basis, &input.gram_float, denom) {
                Ok(g) => {
                    found = Some((gamma, g));
                    break 'search;
                }
                Err(e) => last = e,
            }
        }
    }
    let Some((gamma, gram)) = found else {
        return Err(SosEmitError::NoExactCertificate { last });
    };

    // --- factor for the PSD witness -----------------------------------------
    // `round_gram` already refused a non-PSD Gram via this same factorization,
    // so a failure here would mean the matrix changed underneath us.
    let ldl = ldlt(&gram).map_err(|_| SosEmitError::SelfCheck("Gram lost its LDLᵀ"))?;
    if ldl.d.iter().any(|v| v.is_negative()) {
        return Err(SosEmitError::SelfCheck("LDLᵀ diagonal is not nonnegative"));
    }

    // --- sparsify exactly as the schema specifies ----------------------------
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
    let block = SosBlock {
        monomials: input.basis.clone(),
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
    };

    // --- exact self-check of what will be written ---------------------------
    recheck(&problem, &gamma, &block)?;

    // --- try to upgrade the bound to a minimum ------------------------------
    // If some rational `x₀` hits `γ` exactly, the bound is attained and the
    // stronger verdict is available; otherwise the bound stands alone. Both are
    // sound, and the emitter never guesses between them — attainment is decided
    // by exact evaluation, not by how close the float solve looked.
    let attained = attaining_point(&p_terms, &gamma, n, &input.x_float);
    let (verdict, candidate) = match attained {
        Some(x) => (
            "global-min",
            Some(Candidate {
                x: x.into_iter().map(Rat).collect(),
                objective: Rat(gamma.clone()),
            }),
        ),
        None => ("global-lower-bound", None),
    };

    Ok(Certificate {
        schema: SCHEMA_TAG.to_string(),
        verdict: verdict.to_string(),
        problem_class: "sos-poly".to_string(),
        tolerance: Rat(BigRational::zero()),
        bound: Some(Rat(gamma)),
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
            sos: Some(vec![block]),
            feasible_witness: None,
        },
    })
}

/// Re-derive both proof obligations from the serialized blocks alone, the way
/// the Lean codegen will read them:
///
/// 1. `p(x) − γ = m(x)ᵀ G m(x)` as an identity of coefficient lists, and
/// 2. `G = L·diag(D)·Lᵀ` with `D ≥ 0`.
///
/// Checking (2) is what makes shipping both `gram` and `L`/`D` safe rather than
/// merely redundant: they are two independent descriptions of the same matrix,
/// and here they are forced to agree before either is written.
#[allow(clippy::needless_range_loop)] // index loops cross-reference g, l, d
fn recheck(problem: &Problem, gamma: &BigRational, block: &SosBlock) -> Result<(), SosEmitError> {
    let bn = block.monomials.len();
    let n = problem.n_vars;

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

    // (1) Expand m(x)ᵀ G m(x) into a coefficient map and compare with p − γ.
    let mut lhs: std::collections::BTreeMap<Vec<usize>, BigRational> =
        std::collections::BTreeMap::new();
    for i in 0..bn {
        for j in 0..bn {
            if g[i][j].is_zero() {
                continue;
            }
            let alpha: Vec<usize> = block.monomials[i]
                .iter()
                .zip(&block.monomials[j])
                .map(|(a, b)| a + b)
                .collect();
            *lhs.entry(alpha).or_insert_with(BigRational::zero) += &g[i][j];
        }
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

    lhs.retain(|_, v| !v.is_zero());
    rhs.retain(|_, v| !v.is_zero());
    if lhs != rhs {
        return Err(SosEmitError::SelfCheck("p − γ != m(x)ᵀ G m(x)"));
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
