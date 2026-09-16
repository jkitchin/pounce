//! Inertia accounting over RSLAB's block-diagonal `D`.
//!
//! RSLAB hands back `(d_diag, d_subdiag, two_by_two)` and an already-counted
//! [`rslab::Inertia`]. It does **not** expose the two quantities POUNCE's
//! `pounce-feral` path needs on top of that count:
//!
//! * `min_pivot_magnitude()` — the smallest accepted pivot magnitude, which
//!   `pounce_feral::FeralSolverInterface::factor` compares against
//!   [`pounce_feral::inertia_trust_floor`] before it is willing to report
//!   `WrongInertia` (pounce gh#540), and against `singular_pivot_floor` (the
//!   MA57 `CNTL(2)` analog);
//! * any statement about whether the reported count is a *measurement*.
//!
//! Both are recoverable from `D` alone, in one O(n) pass, which is what this
//! module does. Nothing here modifies RSLAB.
//!
//! # Why the 2×2 blocks need care
//!
//! For a 2×2 pivot block `[[a, b], [b, c]]` the two things POUNCE wants are
//! the block's *eigenvalues*, not its diagonal entries:
//!
//! * inertia is `sign(λ₊), sign(λ₋)`, fixed by `sign(det)` and `sign(tr)`;
//! * the near-singularity signal is `min(|λ₊|, |λ₋|)`.
//!
//! `a` and `c` say nothing about either: `[[0, 1], [1, 0]]` has zero diagonal
//! and eigenvalues `±1`.
//!
//! # Two determinants
//!
//! RSLAB classifies a 2×2 block from `det = a*c - b*b` evaluated naively (in
//! `src/numeric/multifrontal_ldlt.rs`, both in the left-looking emit path and
//! in the multifrontal front kernel). That expression loses every significant
//! digit — and can round to exactly `0.0` or flip sign — when the two products
//! are close, which is the normal shape of a borderline KKT pivot. feral 0.17
//! evaluates the same determinant with Kahan's fused difference-of-products
//! (`feral::dense::factor::det_sym2x2`), whose relative error is `≤ 2u` for any
//! inputs, so `sign(det)` is exact unless the block is genuinely singular to
//! working precision.
//!
//! [`stable_recount`] recomputes the inertia from the same `D` using the
//! fused form. Because it keeps RSLAB's *classification rule* (including its
//! exact-zero convention) and changes only the arithmetic, a disagreement
//! between it and [`rslab::Inertia`] means one thing: cancellation in RSLAB's
//! determinant changed a sign. The adapter treats that as an unreliable
//! inertia rather than picking a winner.

use pounce_common::types::Number;

/// Everything the adapter can say about one factorization's inertia.
///
/// `positive` / `negative` / `zero` are **RSLAB's own counts**, reported
/// unchanged — this type is an observation of RSLAB, not a correction of it.
/// The remaining fields are what POUNCE needs in order to decide whether to
/// act on those counts.
#[derive(Debug, Clone, PartialEq)]
pub struct InertiaInfo {
    /// RSLAB's positive-eigenvalue count.
    pub positive: usize,
    /// RSLAB's negative-eigenvalue count.
    pub negative: usize,
    /// RSLAB's zero-eigenvalue count. Note that under the adapter's default
    /// `ZeroPivotAction::Fail` this is always `0` on a *successful* factor:
    /// RSLAB aborts with `NumericallyRankDeficient` rather than emitting a
    /// zero pivot, and the adapter maps that abort to
    /// [`pounce_linsol::ESymSolverStatus::Singular`].
    pub zero: usize,
    /// `min |λ(D)|` over every eliminated 1×1 pivot and 2×2 block, in the
    /// space of the matrix actually factored (i.e. after the adapter's
    /// equilibration). This is the RSLAB analogue of feral's
    /// `Solver::min_pivot_magnitude`, and for a 2×2 block it is the smaller
    /// eigenvalue magnitude, never a diagonal entry.
    ///
    /// `None` when no pivot was eliminated (`n == 0`).
    pub min_abs_pivot_or_block_eigenvalue: Option<Number>,
    /// `max |λ(D)|` over the same blocks. Paired with the minimum so a caller
    /// can form the scale-free ratio `min/max ≈ 1/κ(D)` without recomputing a
    /// norm. `None` on an empty factor.
    pub max_abs_pivot_or_block_eigenvalue: Option<Number>,
    /// Pivots RSLAB statically perturbed (`LdltNumeric::n_perturbed`). Zero
    /// under the adapter's default settings. See [`InertiaInfo::reliable`] for
    /// why a nonzero count is disqualifying.
    pub perturbed_pivots: usize,
    /// Number of 2×2 Bunch-Kaufman blocks in `D`.
    pub two_by_two_pivots: usize,
    /// The inertia recomputed from the same `D` with the cancellation-free
    /// determinant (see the module docs). Equal to
    /// `(positive, negative, zero)` on a well-conditioned factor.
    pub stable_recount: (usize, usize, usize),
    /// `true` when POUNCE may act on `(positive, negative, zero)` as a
    /// measurement. See [`classify_reliability`] for the three disqualifiers.
    pub reliable: bool,
}

impl InertiaInfo {
    /// The empty factorization's inertia — `n == 0`, nothing measured.
    pub fn empty() -> Self {
        Self {
            positive: 0,
            negative: 0,
            zero: 0,
            min_abs_pivot_or_block_eigenvalue: None,
            max_abs_pivot_or_block_eigenvalue: None,
            perturbed_pivots: 0,
            two_by_two_pivots: 0,
            stable_recount: (0, 0, 0),
            reliable: true,
        }
    }

    /// `(positive, negative, zero)` as RSLAB reported it.
    pub fn triple(&self) -> (usize, usize, usize) {
        (self.positive, self.negative, self.zero)
    }

    /// `true` when RSLAB's counts and the cancellation-free recount agree.
    pub fn recount_agrees(&self) -> bool {
        self.stable_recount == self.triple()
    }
}

/// Determinant of the symmetric 2×2 `[[a, b], [b, c]]` without catastrophic
/// cancellation.
///
/// Kahan's fused difference-of-products: with `w = fl(b*b)` and its exact
/// rounding error `e = fma(b, b, -w)`, the determinant is `(a*c - w) + e`, the
/// first term evaluated with one `fma`. The relative error is `≤ 2u` for any
/// inputs (Jeannerod, Louvet & Muller 2013), so the sign is exact whenever the
/// block is not genuinely singular to working precision.
///
/// This is the same expression as `feral::dense::factor::det_sym2x2`; it is
/// reproduced here rather than imported because it is private to feral, and
/// the point of the check is that RSLAB does *not* do this.
#[inline]
pub fn det_sym2x2(a: Number, b: Number, c: Number) -> Number {
    let w = b * b;
    let e = b.mul_add(b, -w); // exact rounding error of b*b
    let f = a.mul_add(c, -w); // a*c - w, fused
    f + e
}

/// `(|λ|_min, |λ|_max)` of the symmetric 2×2 `[[a, b], [b, c]]`.
///
/// The eigenvalues are `λ± = (a+c)/2 ± sqrt(((a-c)/2)² + b²)`. Evaluated in
/// that form the larger-magnitude root is cancellation-free, because the
/// discriminant is a sum of squares and is added to `|tr|` rather than
/// subtracted from it. The smaller root is then recovered as `|det| / |λ|_max`
/// (from `|λ₊·λ₋| = |det|`), which avoids the subtraction `|tr| - s` that
/// annihilates every significant digit exactly when the block is borderline —
/// the case POUNCE cares about.
///
/// This mirrors `feral`'s `SparseFactors::pivot_magnitude_extent`, using
/// [`det_sym2x2`] for the determinant so a borderline block does not report a
/// spurious zero smaller eigenvalue.
#[inline]
pub fn sym2x2_eigenvalue_magnitudes(a: Number, b: Number, c: Number) -> (Number, Number) {
    let trace = a + c;
    // disc = (a - c)² + 4b², algebraically tr² - 4·det but free of
    // cancellation (a sum of squares). Clamped only against a -0.0 from an
    // FMA contraction.
    let diff = a - c;
    let disc = diff.mul_add(diff, 4.0 * b * b).max(0.0);
    let larger = (trace.abs() + disc.sqrt()) * 0.5;
    let det = det_sym2x2(a, b, c);
    let smaller = if larger > 0.0 {
        (det.abs() / larger).min(larger)
    } else {
        0.0
    };
    (smaller, larger)
}

/// Inertia of the symmetric 2×2 `[[a, b], [b, c]]` from the signs of its
/// determinant and trace, as `(positive, negative, zero)`.
///
/// `λ₊·λ₋ = det` and `λ₊+λ₋ = tr` pin the inertia from those two signs alone:
///
/// * `det < 0` → the roots straddle zero → `(1, 1, 0)`;
/// * `det > 0` → both carry the sign of `tr` → `(2, 0, 0)` or `(0, 2, 0)`;
/// * `det == 0` → one root is zero, the other is `tr`.
///
/// This is RSLAB's own classification rule (`multifrontal_ldlt.rs`, both the
/// left-looking emit and the multifrontal front), with two changes:
/// the determinant comes from [`det_sym2x2`] instead of the naive product
/// difference, and `tr >= 0` on a zero determinant is split so an all-zero
/// block reads `(0, 0, 2)` rather than `(1, 0, 1)`. RSLAB's rule assigns the
/// `tr == 0 && det == 0` block one positive and one zero eigenvalue, which is
/// wrong for the zero block; it is unreachable under `ZeroPivotAction::Fail`
/// (a zero-determinant block aborts first), so this is a divergence in an
/// unreachable branch and not a defect the adapter has to route around.
#[inline]
pub fn classify_2x2(a: Number, b: Number, c: Number) -> (usize, usize, usize) {
    let det = det_sym2x2(a, b, c);
    let tr = a + c;
    if det < 0.0 {
        (1, 1, 0)
    } else if det > 0.0 {
        if tr >= 0.0 { (2, 0, 0) } else { (0, 2, 0) }
    } else if tr > 0.0 {
        (1, 0, 1)
    } else if tr < 0.0 {
        (0, 1, 1)
    } else {
        (0, 0, 2)
    }
}

/// One O(n) pass over `D` returning `(min|λ|, max|λ|)` and the
/// cancellation-free inertia recount.
///
/// `d_diag`, `d_subdiag` and `two_by_two` are RSLAB's
/// [`rslab::LdltNumeric`] fields, in elimination order: `two_by_two[k]` marks
/// the *first* column of a 2×2 block, whose off-diagonal entry is
/// `d_subdiag[k]` and whose second column is `k + 1`.
///
/// Returns `(extent, recount)` where `extent` is `None` for `n == 0`.
pub fn scan_d(
    d_diag: &[Number],
    d_subdiag: &[Number],
    two_by_two: &[bool],
) -> (Option<(Number, Number)>, (usize, usize, usize), usize) {
    let n = d_diag.len();
    let mut min_mag = Number::INFINITY;
    let mut max_mag = 0.0_f64;
    let (mut pos, mut neg, mut zero) = (0usize, 0usize, 0usize);
    let mut n_2x2 = 0usize;
    let mut any = false;
    let mut k = 0usize;
    while k < n {
        // `two_by_two` is sized with `d_diag` by RSLAB, but a block start at
        // the last column would index out of bounds; guard rather than trust.
        let is_2x2 = k + 1 < n && two_by_two.get(k).copied().unwrap_or(false);
        if is_2x2 {
            let (a, b, c) = (d_diag[k], d_subdiag[k], d_diag[k + 1]);
            let (smaller, larger) = sym2x2_eigenvalue_magnitudes(a, b, c);
            min_mag = min_mag.min(smaller);
            max_mag = max_mag.max(larger);
            let (p, ng, z) = classify_2x2(a, b, c);
            pos += p;
            neg += ng;
            zero += z;
            n_2x2 += 1;
            k += 2;
        } else {
            let d = d_diag[k];
            let m = d.abs();
            min_mag = min_mag.min(m);
            max_mag = max_mag.max(m);
            // RSLAB's own 1×1 rule: strict sign comparison, no tolerance.
            // Reproduced deliberately — the adapter's job here is to detect a
            // determinant that cancelled, not to re-tolerance RSLAB.
            if d > 0.0 {
                pos += 1;
            } else if d < 0.0 {
                neg += 1;
            } else {
                zero += 1;
            }
            k += 1;
        }
        any = true;
    }
    let extent = if any { Some((min_mag, max_mag)) } else { None };
    (extent, (pos, neg, zero), n_2x2)
}

/// Recompute `(positive, negative, zero)` from `D` with the cancellation-free
/// determinant. Convenience wrapper over [`scan_d`].
pub fn stable_recount(
    d_diag: &[Number],
    d_subdiag: &[Number],
    two_by_two: &[bool],
) -> (usize, usize, usize) {
    scan_d(d_diag, d_subdiag, two_by_two).1
}

/// Effective inertia-trust floor for a factorization of order `dim`.
///
/// `configured` is [`crate::RslabConfig::inertia_pivot_floor`]: `Some(v)` pins
/// an absolute floor (`Some(0.0)` disables the trigger), `None` selects the
/// dimension-aware default.
///
/// That default is `n · ε`, the backward-error bound on the smallest pivot of
/// an equilibrated matrix of order `n`: below it a pivot's sign is a rounding
/// artefact, above it the sign is a measurement. This is
/// `pounce_feral::inertia_trust_floor` reproduced rather than imported —
/// `pounce-feral` is a dev-dependency here, because a build that wants only
/// RSLAB must not drag FERAL in. The two must agree; the guard is
/// `tests/inertia.rs::the_trust_floor_matches_ferals`, which calls both.
///
/// See `dev-notes/issue-592-restart-non-idempotence.md` for why it is not the
/// fixed `1e-12` it used to be.
pub fn inertia_trust_floor(configured: Option<f64>, dim: usize) -> f64 {
    match configured {
        Some(v) => v,
        None => dim as f64 * f64::EPSILON,
    }
}

/// Decide whether POUNCE may treat RSLAB's counts as a measurement.
///
/// Three disqualifiers, each of which makes the reported triple describe
/// something other than "the inertia of the matrix POUNCE handed over":
///
/// 1. **Perturbed pivots.** RSLAB's static pivoting lifts a small 2×2 by
///    adding `lift` to *both* diagonal entries and, failing that, adding to
///    the determinant outright (`multifrontal_ldlt.rs`). Both operations move
///    eigenvalues across zero, so the factor's inertia is that of `A + E` and
///    need not equal `A`'s at all. feral's `n_tiny` is explicitly documented as
///    *not* affecting inertia because its 1×1 perturbation preserves the sign;
///    RSLAB's 2×2 perturbation carries no such guarantee, so the conservative
///    reading is that any perturbation voids the count.
/// 2. **Sub-floor pivot.** `min |λ(D)| < trust_floor` — the pounce gh#540
///    condition. Below the backward-error bound on an equilibrated matrix of
///    this order a pivot's *sign* is a rounding artefact, so the count built
///    from those signs is noise. `trust_floor` is supplied by the caller and
///    is [`pounce_feral::inertia_trust_floor`] in the adapter, so RSLAB and
///    FERAL are judged against exactly the same threshold rather than an
///    invented RSLAB-specific one.
/// 3. **Recount disagreement.** RSLAB's naive determinant and the fused one
///    classified some block differently, i.e. cancellation changed a sign.
///
/// A `trust_floor` of `0.0` disables disqualifier 2, matching
/// `FeralConfig::inertia_pivot_floor = Some(0.0)`.
pub fn classify_reliability(
    rslab_counts: (usize, usize, usize),
    stable: (usize, usize, usize),
    min_abs: Option<Number>,
    perturbed: usize,
    trust_floor: Number,
) -> bool {
    if perturbed > 0 {
        return false;
    }
    if stable != rslab_counts {
        return false;
    }
    if trust_floor > 0.0 {
        if let Some(m) = min_abs {
            if m < trust_floor {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block the prompt singles out: zero diagonal, eigenvalues ±1.
    /// Reading `a` and `c` would report two zeros; the determinant/trace rule
    /// reports one of each.
    #[test]
    fn antidiagonal_2x2_is_one_positive_one_negative() {
        assert_eq!(classify_2x2(0.0, 1.0, 0.0), (1, 1, 0));
        let (lo, hi) = sym2x2_eigenvalue_magnitudes(0.0, 1.0, 0.0);
        assert!((lo - 1.0).abs() < 1e-15, "min |λ| = {lo}");
        assert!((hi - 1.0).abs() < 1e-15, "max |λ| = {hi}");
    }

    /// Definite and indefinite 2×2 blocks against the closed form.
    #[test]
    fn classify_2x2_matches_the_closed_form_eigenvalues() {
        for &(a, b, c) in &[
            (3.0_f64, 1.0_f64, 2.0_f64),
            (-3.0, 1.0, -2.0),
            (1.0, 4.0, 1.0),
            (1e-8, 1.0, -1e-8),
            (5.0, 0.0, -7.0),
            (2.0, 0.0, 0.0),
        ] {
            let tr = a + c;
            let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
            let lp = 0.5 * (tr + disc);
            let lm = 0.5 * (tr - disc);
            let want = (
                usize::from(lp > 0.0) + usize::from(lm > 0.0),
                usize::from(lp < 0.0) + usize::from(lm < 0.0),
                usize::from(lp == 0.0) + usize::from(lm == 0.0),
            );
            assert_eq!(classify_2x2(a, b, c), want, "block ({a}, {b}, {c})");
        }
    }

    /// The smaller eigenvalue magnitude is recovered through `|det|/|λ|_max`,
    /// so a block whose two eigenvalues differ by ~16 orders still reports the
    /// small one to full relative accuracy — the subtraction form
    /// `0.5*(|tr| - disc)` returns 0 here.
    #[test]
    fn small_eigenvalue_survives_a_huge_spread() {
        // [[1e16, 1], [1, 1e-16]] — λ ≈ 1e16 and ≈ -1e-16 (det = -1 + 1e0 …).
        let (a, b, c) = (1e16, 1.0, 1e-16);
        let (lo, hi) = sym2x2_eigenvalue_magnitudes(a, b, c);
        let det = det_sym2x2(a, b, c);
        assert!(hi > 1e15, "max |λ| = {hi}");
        // |λ_min| = |det| / |λ_max|; check consistency to a few ulps.
        let want = det.abs() / hi;
        assert!(
            (lo - want).abs() <= 1e-12 * want.max(f64::MIN_POSITIVE),
            "min |λ| = {lo}, want {want}"
        );
        // The naive subtraction cancels to exactly zero here; ours does not.
        let tr = a + c;
        let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        assert_eq!(0.5 * (tr - disc).abs(), 0.0, "premise: the naive form cancels");
        assert!(lo > 0.0, "the stable form must not report a zero eigenvalue");
    }

    /// The fused determinant keeps a sign the naive product difference loses.
    ///
    /// `a = 1 + 2⁻⁵²`, `c = 1 - 2⁻⁵³`, `b = 1`. The exact `a·c` is
    /// `1 + 2⁻⁵³ - 2⁻¹⁰⁵`, a hair below the midpoint between `1` and the next
    /// double up, so `fl(a·c)` rounds to exactly `1.0` and the naive
    /// `a*c - b*b` evaluates to `0.0`. The true determinant is `+2⁻⁵³`, so the
    /// block is positive definite (`tr = 2 > 0`) — and the naive rule, which
    /// is the rule RSLAB applies, certifies it as singular with one zero
    /// eigenvalue instead. The fused form's single rounding recovers the sign.
    ///
    /// This is not a contrived scale: `a` and `c` are ordinary `O(1)` numbers
    /// one ulp apart, which is what a 2×2 pivot on a converging KKT looks like.
    #[test]
    fn fused_determinant_keeps_a_sign_the_naive_one_loses() {
        let a = 1.0 + 2.0_f64.powi(-52);
        let c = 1.0 - 2.0_f64.powi(-53);
        let b = 1.0_f64;
        let naive = a * c - b * b;
        assert_eq!(naive, 0.0, "premise: the naive difference cancels to zero");

        let fused = det_sym2x2(a, b, c);
        assert!(fused > 0.0, "true determinant is positive: {fused}");
        assert!(
            (fused - 2.0_f64.powi(-53)).abs() <= 2.0_f64.powi(-100),
            "fused determinant {fused} should be ~2^-53"
        );

        // The two rules disagree about the inertia of this block, which is
        // the whole point: RSLAB's naive `det` books a zero eigenvalue.
        assert_eq!(classify_2x2(a, b, c), (2, 0, 0));
        let naive_rule = if naive < 0.0 {
            (1, 1, 0)
        } else if naive > 0.0 {
            (2, 0, 0)
        } else {
            (1, 0, 1) // det == 0, tr > 0 — RSLAB's arm
        };
        assert_eq!(
            naive_rule,
            (1, 0, 1),
            "premise: the naive rule reports a zero eigenvalue"
        );

        // …and the smaller eigenvalue magnitude is likewise nonzero.
        let (lo, _hi) = sym2x2_eigenvalue_magnitudes(a, b, c);
        assert!(lo > 0.0, "min |λ| must not be reported as an exact zero");
    }

    /// `scan_d` walks 1×1 and 2×2 blocks in elimination order and totals both
    /// the inertia and the magnitude extent.
    #[test]
    fn scan_d_walks_mixed_blocks() {
        // D = diag(3) ⊕ [[0,1],[1,0]] ⊕ diag(-4)
        let d_diag = vec![3.0, 0.0, 0.0, -4.0];
        let d_subdiag = vec![0.0, 1.0, 0.0, 0.0];
        let two_by_two = vec![false, true, false, false];
        let (extent, counts, n2) = scan_d(&d_diag, &d_subdiag, &two_by_two);
        assert_eq!(counts, (2, 2, 0));
        assert_eq!(n2, 1);
        let (lo, hi) = extent.unwrap();
        assert!((lo - 1.0).abs() < 1e-15, "min |λ| = {lo}");
        assert!((hi - 4.0).abs() < 1e-15, "max |λ| = {hi}");
    }

    #[test]
    fn scan_d_on_an_empty_factor_measures_nothing() {
        let (extent, counts, n2) = scan_d(&[], &[], &[]);
        assert_eq!(extent, None);
        assert_eq!(counts, (0, 0, 0));
        assert_eq!(n2, 0);
    }

    /// Each disqualifier fires on its own, and a clean factor passes.
    #[test]
    fn reliability_disqualifiers_fire_independently() {
        let good = (5, 3, 0);
        assert!(classify_reliability(good, good, Some(1e-3), 0, 1e-13));
        // 1. a perturbed pivot
        assert!(!classify_reliability(good, good, Some(1e-3), 1, 1e-13));
        // 2. a pivot under the floor
        assert!(!classify_reliability(good, good, Some(1e-16), 0, 1e-13));
        // 2'. …unless the floor is disabled
        assert!(classify_reliability(good, good, Some(1e-16), 0, 0.0));
        // 3. the recount disagrees
        assert!(!classify_reliability(good, (4, 4, 0), Some(1e-3), 0, 1e-13));
    }
}
