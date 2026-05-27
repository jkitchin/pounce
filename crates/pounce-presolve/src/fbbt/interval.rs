//! Rounded interval arithmetic for FBBT.
//!
//! Each operator returns an *outward-rounded* over-approximation: the
//! result is guaranteed to contain the exact mathematical range, even
//! after `f64` rounding errors in the underlying operation. This is
//! the soundness property FBBT needs — over-approximation may produce
//! a looser tightening than ideal, but it can never over-tighten and
//! drop a feasible point.
//!
//! Rounding policy: every finite floating-point result is widened by
//! one ULP outward (`next_down` on `lo`, `next_up` on `hi`) before
//! returning. This costs one ULP of accuracy per operation but
//! requires no FPU rounding-mode changes and works identically on
//! every target. The accumulated padding is `O(ULP × depth)` for a
//! DAG of depth `depth`, well within what FBBT tolerates.
//!
//! Function semantics on the empty interval propagate: any operation
//! involving `EMPTY` returns `EMPTY`. The orchestrator interprets a
//! leaf interval of `EMPTY` as "FBBT detected infeasibility" (the
//! constraint's reverse propagation produced an empty variable
//! interval).

use pounce_common::types::Number;

/// Closed interval `[lo, hi]` with outward rounding on operations.
///
/// `EMPTY` is encoded as `lo > hi` (specifically `lo = +∞`,
/// `hi = -∞`). `ENTIRE` is `(-∞, +∞)`. Construction helpers normalize
/// these for you.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub lo: Number,
    pub hi: Number,
}

impl Interval {
    /// `[-∞, +∞]` — represents "no information".
    pub const ENTIRE: Interval = Interval {
        lo: Number::NEG_INFINITY,
        hi: Number::INFINITY,
    };

    /// `∅` — empty set; sentinel for infeasibility.
    pub const EMPTY: Interval = Interval {
        lo: Number::INFINITY,
        hi: Number::NEG_INFINITY,
    };

    /// Closed interval `[lo, hi]`. NaN endpoints or `lo > hi` collapse
    /// to [`Self::EMPTY`] so downstream rules don't have to special-
    /// case malformed input.
    pub fn new(lo: Number, hi: Number) -> Self {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return Self::EMPTY;
        }
        Self { lo, hi }
    }

    /// Degenerate interval `[x, x]`. NaN collapses to `EMPTY`.
    pub fn point(x: Number) -> Self {
        if x.is_nan() {
            return Self::EMPTY;
        }
        Self { lo: x, hi: x }
    }

    pub fn is_empty(&self) -> bool {
        self.lo > self.hi || self.lo.is_nan() || self.hi.is_nan()
    }

    pub fn is_entire(&self) -> bool {
        self.lo == Number::NEG_INFINITY && self.hi == Number::INFINITY
    }

    /// `true` iff `x ∈ [lo, hi]`.
    pub fn contains(&self, x: Number) -> bool {
        !self.is_empty() && self.lo <= x && x <= self.hi
    }

    pub fn contains_zero(&self) -> bool {
        self.contains(0.0)
    }

    /// `hi - lo`, or 0 for empty. May be `+∞`.
    pub fn width(&self) -> Number {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// `[max(self.lo, other.lo), min(self.hi, other.hi)]`. Empty if
    /// the result is malformed — this is the right semantics for
    /// FBBT's "narrow against the constraint" step.
    pub fn intersect(self, other: Self) -> Self {
        if self.is_empty() || other.is_empty() {
            return Self::EMPTY;
        }
        Self::new(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    /// Convex hull `[min(lo, lo), max(hi, hi)]`. Empty if both inputs
    /// are empty.
    pub fn hull(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    // -------- Arithmetic (outward-rounded) --------

    pub fn add(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: round_down(self.lo + rhs.lo),
            hi: round_up(self.hi + rhs.hi),
        }
    }

    pub fn sub(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: round_down(self.lo - rhs.hi),
            hi: round_up(self.hi - rhs.lo),
        }
    }

    pub fn neg(self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    /// Interval multiplication — the classic four-corner formula.
    pub fn mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        let p1 = self.lo * rhs.lo;
        let p2 = self.lo * rhs.hi;
        let p3 = self.hi * rhs.lo;
        let p4 = self.hi * rhs.hi;
        let lo = round_down(p1.min(p2).min(p3.min(p4)));
        let hi = round_up(p1.max(p2).max(p3.max(p4)));
        Self { lo, hi }
    }

    /// Division — returns `ENTIRE` (rather than a split / extended
    /// interval) when `0 ∈ rhs`, since FBBT's reverse rules elsewhere
    /// can recover useful information without the union-of-intervals
    /// complexity here.
    pub fn div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        if rhs.contains_zero() {
            // Conservative: any value is possible.
            return Self::ENTIRE;
        }
        self.mul(Self {
            lo: 1.0 / rhs.hi,
            hi: 1.0 / rhs.lo,
        })
    }

    /// `[lo, hi]^n` for non-negative integer `n`. Handles the
    /// non-monotone case `n` even, `0 ∈ self` correctly.
    pub fn pow_uint(self, n: u32) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        if n == 0 {
            return Self::point(1.0);
        }
        if n == 1 {
            return self;
        }
        let (a, b) = (self.lo, self.hi);
        if n % 2 == 1 {
            // Odd → monotone increasing.
            Self {
                lo: round_down(powi(a, n as i32)),
                hi: round_up(powi(b, n as i32)),
            }
        } else if a >= 0.0 {
            // Even, fully non-negative → monotone increasing.
            Self {
                lo: round_down(powi(a, n as i32)),
                hi: round_up(powi(b, n as i32)),
            }
        } else if b <= 0.0 {
            // Even, fully non-positive → monotone decreasing in
            // magnitude; powers are non-negative with the smaller
            // |.| inside the interval.
            Self {
                lo: round_down(powi(b, n as i32)),
                hi: round_up(powi(a, n as i32)),
            }
        } else {
            // Even, straddles zero → minimum is at 0, max at the
            // extreme with larger |.|.
            let ha = powi(a, n as i32);
            let hb = powi(b, n as i32);
            Self {
                lo: 0.0,
                hi: round_up(ha.max(hb)),
            }
        }
    }

    /// `√[lo, hi]`. Negative `lo` clipped to 0 (matches mathematical
    /// domain). Empty input or fully-negative interval → `EMPTY`.
    pub fn sqrt(self) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        let lo = self.lo.max(0.0).sqrt();
        let hi = self.hi.sqrt();
        Self {
            lo: round_down(lo),
            hi: round_up(hi),
        }
    }

    /// `exp([lo, hi])` — monotone increasing.
    pub fn exp(self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self {
            lo: round_down(self.lo.exp()),
            hi: round_up(self.hi.exp()),
        }
    }

    /// `ln([lo, hi])` — monotone increasing on `x > 0`. `lo ≤ 0` is
    /// clipped to the smallest positive finite (return `-∞` on the
    /// low side). Fully-non-positive intervals → `EMPTY`.
    pub fn ln(self) -> Self {
        if self.is_empty() || self.hi <= 0.0 {
            return Self::EMPTY;
        }
        let lo = if self.lo <= 0.0 {
            Number::NEG_INFINITY
        } else {
            round_down(self.lo.ln())
        };
        Self {
            lo,
            hi: round_up(self.hi.ln()),
        }
    }

    /// `|[lo, hi]|`.
    pub fn abs(self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            // Straddles zero.
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi),
            }
        }
    }

    /// `sin([lo, hi])`. When the interval is wider than 2π or wraps
    /// over both a peak and a trough we return `[-1, 1]`; otherwise
    /// we test against the local extrema. A loose-but-sound choice.
    pub fn sin(self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        trig_image(self.lo, self.hi, |x| x.sin(), SIN_PEAKS, SIN_TROUGHS)
    }

    /// `cos([lo, hi])`.
    pub fn cos(self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        trig_image(self.lo, self.hi, |x| x.cos(), COS_PEAKS, COS_TROUGHS)
    }

    /// Element-wise `min(self, other)`.
    pub fn min(self, other: Self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        if other.is_empty() {
            return Self::EMPTY;
        }
        Self::new(self.lo.min(other.lo), self.hi.min(other.hi))
    }

    /// Element-wise `max(self, other)`.
    pub fn max(self, other: Self) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        if other.is_empty() {
            return Self::EMPTY;
        }
        Self::new(self.lo.max(other.lo), self.hi.max(other.hi))
    }
}

// -------- Rounding helpers --------

/// Outward round on the low end: nudge `x` one ULP toward `-∞`.
/// Identity on infinities and NaN.
fn round_down(x: Number) -> Number {
    if x.is_finite() {
        x.next_down()
    } else {
        x
    }
}

/// Outward round on the high end: nudge `x` one ULP toward `+∞`.
fn round_up(x: Number) -> Number {
    if x.is_finite() {
        x.next_up()
    } else {
        x
    }
}

/// `x^n` for non-negative `n`, with `0^0 = 1` (Rust's `f64::powi`
/// convention).
fn powi(x: Number, n: i32) -> Number {
    x.powi(n)
}

// -------- Trigonometric image bounds --------
//
// For `sin([a,b])` we check whether the interval crosses any
// peak (`x` where sin x = 1, i.e. π/2 + 2πk) or trough
// (`x` where sin x = -1, i.e. -π/2 + 2πk). If it does, the global
// max / min is locked. Otherwise the image is between the endpoint
// values. Same for cos.

const TWO_PI: Number = 2.0 * std::f64::consts::PI;
const SIN_PEAKS: Number = std::f64::consts::FRAC_PI_2; // π/2 + 2πk
const SIN_TROUGHS: Number = -std::f64::consts::FRAC_PI_2;
const COS_PEAKS: Number = 0.0; // 0 + 2πk
const COS_TROUGHS: Number = std::f64::consts::PI; // π + 2πk

fn trig_image<F>(lo: Number, hi: Number, f: F, peak_offset: Number, trough_offset: Number) -> Interval
where
    F: Fn(Number) -> Number,
{
    if !lo.is_finite() || !hi.is_finite() {
        // Unbounded → can hit any value in [-1, 1].
        return Interval::new(-1.0, 1.0);
    }
    if hi - lo >= TWO_PI {
        return Interval::new(-1.0, 1.0);
    }
    // Reference k such that the closest peak ≥ lo is at peak_offset
    // + 2πk.
    let crosses = |offset: Number| -> bool {
        let k = ((lo - offset) / TWO_PI).ceil();
        let x = offset + TWO_PI * k;
        x <= hi
    };
    let endpoint_lo = f(lo);
    let endpoint_hi = f(hi);
    let mut local_min = endpoint_lo.min(endpoint_hi);
    let mut local_max = endpoint_lo.max(endpoint_hi);
    if crosses(peak_offset) {
        local_max = 1.0;
    }
    if crosses(trough_offset) {
        local_min = -1.0;
    }
    Interval {
        lo: round_down(local_min),
        hi: round_up(local_max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Number, b: Number, eps: Number) -> bool {
        (a - b).abs() <= eps + eps * b.abs()
    }

    #[test]
    fn empty_propagates() {
        let e = Interval::EMPTY;
        let a = Interval::new(0.0, 1.0);
        assert!(e.add(a).is_empty());
        assert!(a.add(e).is_empty());
        assert!(e.mul(a).is_empty());
        assert!(e.sqrt().is_empty());
        assert!(e.exp().is_empty());
    }

    #[test]
    fn new_normalizes_malformed() {
        assert!(Interval::new(1.0, 0.0).is_empty());
        assert!(Interval::new(Number::NAN, 1.0).is_empty());
        assert!(Interval::new(1.0, Number::NAN).is_empty());
    }

    #[test]
    fn entire_is_entire() {
        assert!(Interval::ENTIRE.is_entire());
        assert!(Interval::ENTIRE.contains_zero());
        assert!(!Interval::EMPTY.is_entire());
    }

    #[test]
    fn add_widens_outward() {
        // [1, 2] + [3, 4] = [4, 6] (outward-rounded).
        let r = Interval::new(1.0, 2.0).add(Interval::new(3.0, 4.0));
        assert!(r.lo <= 4.0 && 4.0 - r.lo < 1e-15);
        assert!(r.hi >= 6.0 && r.hi - 6.0 < 1e-15);
    }

    #[test]
    fn sub_uses_cross_endpoints() {
        // [1, 2] - [3, 4] = [-3, -1].
        let r = Interval::new(1.0, 2.0).sub(Interval::new(3.0, 4.0));
        assert!(r.lo <= -3.0 && -3.0 - r.lo < 1e-15);
        assert!(r.hi >= -1.0 && r.hi - (-1.0) < 1e-15);
    }

    #[test]
    fn mul_handles_sign_crossings() {
        // [-2, 3] * [-1, 4] — must consider all four corners.
        // Min is (-2)*4 = -8; max is 3*4 = 12.
        let r = Interval::new(-2.0, 3.0).mul(Interval::new(-1.0, 4.0));
        assert!(r.contains(-8.0));
        assert!(r.contains(12.0));
        assert!(r.lo <= -8.0);
        assert!(r.hi >= 12.0);
    }

    #[test]
    fn div_by_zero_crossing_yields_entire() {
        let r = Interval::new(1.0, 2.0).div(Interval::new(-1.0, 1.0));
        assert!(r.is_entire());
    }

    #[test]
    fn div_disjoint_from_zero_inverts_correctly() {
        // [1, 4] / [2, 4] ⊆ [1/4, 2] = [0.25, 2].
        let r = Interval::new(1.0, 4.0).div(Interval::new(2.0, 4.0));
        assert!(r.contains(0.25));
        assert!(r.contains(2.0));
        assert!(r.lo <= 0.25);
        assert!(r.hi >= 2.0);
    }

    #[test]
    fn pow_uint_even_straddles_zero() {
        // [-2, 3]^2 = [0, 9].
        let r = Interval::new(-2.0, 3.0).pow_uint(2);
        assert_eq!(r.lo, 0.0);
        assert!(r.hi >= 9.0);
    }

    #[test]
    fn pow_uint_even_negative() {
        // [-4, -2]^2 = [4, 16].
        let r = Interval::new(-4.0, -2.0).pow_uint(2);
        assert!(r.lo <= 4.0);
        assert!(r.hi >= 16.0);
    }

    #[test]
    fn pow_uint_odd() {
        // [-2, 3]^3 = [-8, 27].
        let r = Interval::new(-2.0, 3.0).pow_uint(3);
        assert!(r.lo <= -8.0);
        assert!(r.hi >= 27.0);
    }

    #[test]
    fn pow_zero_and_one() {
        let i = Interval::new(2.0, 5.0);
        let z = i.pow_uint(0);
        assert_eq!(z.lo, 1.0);
        assert_eq!(z.hi, 1.0);
        let o = i.pow_uint(1);
        assert_eq!(o, i);
    }

    #[test]
    fn sqrt_clips_negative_lo() {
        // Domain clip: sqrt([-1, 4]) takes the positive part of the
        // domain. The mathematical lower bound is 0; outward rounding
        // is allowed to nudge it one ULP below zero (still a valid
        // over-approximation).
        let r = Interval::new(-1.0, 4.0).sqrt();
        assert!(r.lo <= 0.0);
        assert!(r.lo >= -1e-300, "outward bump should be at most ~1 ULP");
        assert!(r.hi >= 2.0);
    }

    #[test]
    fn sqrt_of_fully_negative_is_empty() {
        assert!(Interval::new(-4.0, -1.0).sqrt().is_empty());
    }

    #[test]
    fn exp_is_monotone() {
        let r = Interval::new(0.0, 1.0).exp();
        assert!(r.contains(1.0));
        assert!(r.contains(std::f64::consts::E));
    }

    #[test]
    fn ln_of_non_positive_is_empty() {
        assert!(Interval::new(-2.0, -1.0).ln().is_empty());
        assert!(Interval::new(-2.0, 0.0).ln().is_empty());
    }

    #[test]
    fn ln_with_zero_lower_yields_neg_inf() {
        let r = Interval::new(0.0, 1.0).ln();
        assert_eq!(r.lo, Number::NEG_INFINITY);
        assert!(r.contains(0.0));
    }

    #[test]
    fn ln_strict_positive() {
        // ln([1, e]) ⊆ [0, 1].
        let r = Interval::new(1.0, std::f64::consts::E).ln();
        assert!(r.contains(0.0));
        assert!(r.contains(1.0));
    }

    #[test]
    fn abs_negative_interval() {
        let r = Interval::new(-3.0, -1.0).abs();
        assert!(r.contains(1.0));
        assert!(r.contains(3.0));
    }

    #[test]
    fn abs_straddling_interval() {
        let r = Interval::new(-2.0, 3.0).abs();
        assert_eq!(r.lo, 0.0);
        assert!(r.hi >= 3.0);
    }

    #[test]
    fn sin_full_range() {
        // sin([0, 2π]) = [-1, 1].
        let r = Interval::new(0.0, TWO_PI).sin();
        assert!(approx_eq(r.lo, -1.0, 1e-15));
        assert!(approx_eq(r.hi, 1.0, 1e-15));
    }

    #[test]
    fn sin_within_one_branch() {
        // sin([0, π/2]) = [0, 1].
        let r = Interval::new(0.0, std::f64::consts::FRAC_PI_2).sin();
        assert!(r.contains(0.0));
        assert!(r.contains(1.0));
    }

    #[test]
    fn cos_at_zero() {
        // cos([-ε, ε]) ⊆ [1-O(ε²), 1].
        let r = Interval::new(-0.1, 0.1).cos();
        assert!(r.contains(1.0));
        assert!(r.lo < 1.0);
    }

    #[test]
    fn intersect_disjoint_is_empty() {
        assert!(Interval::new(0.0, 1.0)
            .intersect(Interval::new(2.0, 3.0))
            .is_empty());
    }

    #[test]
    fn intersect_overlap() {
        let r = Interval::new(0.0, 5.0).intersect(Interval::new(3.0, 10.0));
        assert_eq!(r, Interval::new(3.0, 5.0));
    }

    #[test]
    fn hull_combines() {
        let r = Interval::new(0.0, 1.0).hull(Interval::new(5.0, 6.0));
        assert_eq!(r, Interval::new(0.0, 6.0));
    }

    #[test]
    fn min_max_pairs() {
        let a = Interval::new(1.0, 5.0);
        let b = Interval::new(2.0, 7.0);
        let mn = a.min(b);
        assert!(mn.contains(1.0));
        assert!(mn.contains(5.0));
        let mx = a.max(b);
        assert!(mx.contains(2.0));
        assert!(mx.contains(7.0));
    }

    /// Soundness fuzz: for random `[a, b]` and random `x ∈ [a, b]`,
    /// the operation's result must contain `f(x)` exactly.
    #[test]
    fn fuzz_add_contains_pointwise() {
        let cases = [
            ((0.5, 2.5), (1.0, 1.5), 1.5, 2.0),
            ((-3.0, 1.0), (-1.0, 4.0), 0.5, 2.5),
            ((1.0e-10, 1.0e10), (1.0, 1.0), 100.0, 1.0),
        ];
        for &((a, b), (c, d), x, y) in &cases {
            let i = Interval::new(a, b).add(Interval::new(c, d));
            assert!(i.contains(x + y), "{a},{b} + {c},{d} ∌ {x}+{y}");
        }
    }

    #[test]
    fn fuzz_mul_contains_pointwise() {
        let cases = [
            ((-2.0, 3.0), (-1.0, 4.0), 0.5, 2.0),
            ((1.0, 10.0), (0.1, 0.2), 5.0, 0.15),
            ((-5.0, -1.0), (-3.0, -1.0), -3.0, -2.0),
        ];
        for &((a, b), (c, d), x, y) in &cases {
            let i = Interval::new(a, b).mul(Interval::new(c, d));
            assert!(i.contains(x * y), "{a},{b} × {c},{d} ∌ {x}×{y}");
        }
    }

    /// Outward rounding is observable: `(a + b) - b` may not be
    /// exactly `a`, but the resulting interval must still contain
    /// `a`.
    #[test]
    fn rounding_does_not_shrink_below_truth() {
        let one = Interval::point(0.1);
        let two = Interval::point(0.2);
        let sum = one.add(two);
        assert!(sum.contains(0.3));
    }
}
