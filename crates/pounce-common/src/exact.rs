//! Was this floating-point operation **exact**?
//!
//! Three predicates, shared by everything that folds coefficients into a
//! map and then has to answer whether the fold kept them. They were private
//! to `pounce-nl`'s quadratic recognizer until gh #673 gave the `.nl`
//! pipeline a second such fold — `Σ 2wₖbₖbₖᵀ` in
//! `pounce_nlp::quadratic::QuadraticStructure::push_factored_form` — in a
//! crate that cannot name the first. Duplicating them would have been two
//! copies of the argument in gh #683, gh #685 and gh #687, free to drift.
//!
//! The question they answer is never "is this close enough". It is "did
//! this add or multiply round at all", answered exactly, so that a term
//! that cancelled **exactly** (`x − x`, degree really 0) is told apart from
//! one that was **absorbed** (`2⁵³·x + x − 2⁵³·x`, where the `x` is gone and
//! the read-out would silently be short of it).

/// Was `a + b`, which came out as `s`, computed **exactly**?
///
/// Knuth's two-sum: `err` below is the part of the true sum that `s` could
/// not represent, and it is itself exact for every finite pair — no
/// tolerance, no magnitude test. Two extra flops per merge, which is the
/// whole cost of telling `x − x` apart from `2⁵³·x + x − 2⁵³·x` (the `x` was
/// absorbed by the first add, and the second only made it visible). See
/// gh #687.
///
/// A non-finite operand — or a sum that overflowed — makes `err` a `NaN`,
/// which answers *inexact*: the conservative direction, and the right one,
/// since an infinity has lost every term it swallowed.
pub fn add_is_exact(a: f64, b: f64, s: f64) -> bool {
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    err == 0.0
}

/// Was `a · b`, which came out as `p`, computed **exactly**?
///
/// `fma(a, b, −p)` is the multiply's rounding error, exactly, and `p` is
/// exact iff that error is zero. The `±1` shortcut is not micro-optimizing
/// for its own sake: every variable enters `pounce-nl`'s recognizer with a
/// coefficient of `1.0`, so it is the common case by a wide margin, and
/// `f64::mul_add` is a libm call on targets without a hardware FMA.
///
/// A product that overflowed to an infinity answers *inexact* for the same
/// reason a sum that did.
pub fn mul_is_exact(a: f64, b: f64, p: f64) -> bool {
    if a == 1.0 || a == -1.0 || b == 1.0 || b == -1.0 || a == 0.0 || b == 0.0 {
        // Multiplying by ±1 or by an exact zero cannot round.
        return true;
    }
    // A product that is not *normal* is one the exponent range could not
    // hold: zero (`10⁻²⁰⁰ · 10⁻²⁰⁰`, from two nonzero factors), a subnormal
    // that gave up bits at the bottom, or an infinity from overflow. The
    // `fma` below cannot be asked about those — it rounds onto the same
    // grid, and answers `0` for the underflowed product as readily as for
    // an exact one — so they are answered here, inexact.
    p.is_normal() && a.mul_add(b, -p) == 0.0
}

/// Is this coefficient worth **storing**? Exact zeros are dropped so that
/// degree and constant-folding questions stay `O(1)`, and so that a term
/// that cancelled cannot make a later product look like degree 3. `NaN` is
/// dropped for the same reason.
///
/// This is the *storage* question only. It is applied to a **sum** of
/// coefficients, so a degree-2 body whose coefficients cancel — or whose
/// one coefficient underflows — stores nothing and looks affine. That is
/// gh #683, and it is why every caller weighs the drop against the
/// arithmetic that led to it rather than reading this predicate alone.
pub fn is_live(c: f64) -> bool {
    c.abs() > 0.0
}
