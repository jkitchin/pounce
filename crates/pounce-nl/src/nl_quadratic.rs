//! Degree-≤2 recognition over an [`Expr`] DAG.
//!
//! This is the classifier's "is this row a quadratic, and which one?"
//! question, answered once and reused. `pounce-cli`'s `dispatch` owns the
//! *routing* decision (which `ProblemClass`, which solver); what lives here
//! is only the algebra, so the consumers that are not the CLI —
//! `NlTnlp`'s constant-structure evaluation, the parse-time recognizer, the
//! `QcqpProblem` extractor — can reach it without depending on the
//! command-line driver. It moved out of `pounce-cli/src/dispatch.rs` in
//! Q3 of the #588 series; see
//! `dev-notes/quadratic-structure-exploitation.md`.
//!
//! ## Two properties this module is built around
//!
//! **It is iterative.** The walk carries its own work stack rather than
//! recursing, because the trees it is handed are not shallow: a `.nl`
//! writer that emits `o0` (binary `+`) chains for a long sum — Pyomo does —
//! produces a left-deep `Add` tree one level per term. The recursive
//! predecessor aborted the process somewhere between 4 000 and 6 000 terms
//! on a 2 MB thread (which is what a test gets, and where the crash was
//! first reproduced) and between 16 000 and 24 000 on the CLI's 8 MB main
//! thread. A stack overflow is an abort, not an error return, so the depth a
//! *recognizer* survives must not depend on which thread called it.
//!
//! It is worth knowing what this does **not** fix: `nl_reader`'s parser
//! recurses too, with a fatter frame — it gives out at ~6 000 on that same
//! 8 MB thread — so a deep `.nl` file still fails to load, and it fails
//! before reaching this module. What is fixed is every path where the tree
//! is already built (`NlProblem::from_expressions`, a model handed across
//! threads) and the ceiling this module used to impose on the parser's
//! successor.
//!
//! **It never allocates per monomial.** The predecessor keyed monomials on
//! `BTreeMap<Vec<usize>, f64>`, so every term cost a heap allocation and
//! every merge cloned one (`entry(m.clone())`). A degree-≤2 form has only
//! three shapes of term, so [`Quad2`] stores them in three fields with
//! inline keys and the allocation disappears. The same change removes the
//! `O(N²)` accumulation on `Add` chains: the old `add` re-scanned the whole
//! accumulated map for zeros on *every* merge, which is quadratic down a
//! left-deep chain. Zeros can only appear where a merge touched, so that is
//! all this one looks at.

use crate::nl_reader::{BinOp, Expr, UnaryOp};
use std::collections::{BTreeMap, BTreeSet};

/// The symmetric Hessian of a quadratic form, stored as a sparse upper-
/// triangular (i ≤ j) map of `(i, j) -> ∂²/∂xᵢ∂xⱼ`. Empty means the
/// expression is (at most) linear.
pub type QuadHessian = BTreeMap<(usize, usize), f64>;

/// Full quadratic read-out: `(Hessian, [(var, linear coef), …], constant)`.
/// The linear and constant parts are the pieces AMPL/Pyomo fold into the
/// nonlinear objective tree (see [`analyze_quadratic_full`]).
pub type QuadForm = (QuadHessian, Vec<(usize, f64)>, f64);

/// A polynomial of total degree ≤ 2 in its own shape: a constant, the
/// linear coefficients keyed by variable, and the quadratic coefficients
/// keyed by the (i ≤ j) variable pair.
///
/// This replaces a general `BTreeMap<Vec<usize>, f64>` polynomial. Degree
/// is a property of the *type* here rather than of the data, so the
/// "is it still quadratic?" test is three `is_empty()` calls instead of a
/// scan, and no monomial key is ever allocated or cloned.
///
/// ### Zero coefficients
///
/// Stored coefficients are nonzero: [`add`](Quad2::add) and
/// [`mul`](Quad2::mul) drop any entry they leave at exactly zero, which is
/// what makes [`degree`](Quad2::degree) and [`as_constant`](Quad2::as_constant)
/// answerable in `O(1)`. `constant` is the one exception and needs none:
/// both `0.0` and `-0.0` in that field mean "no constant term", every
/// consumer guards on `!= 0.0`, and [`analyze_quadratic_full`] normalizes
/// the sign on the way out.
///
/// ### …and when dropping one loses something
///
/// Dropping is right for the *storage* question and can be wrong for the
/// *degree* question, and [`lost_terms`](Quad2::lost_terms) is the
/// difference (gh #683, sharpened by gh #687). A coefficient that reaches
/// zero is a coefficient that was **summed**, and it is the arithmetic of
/// that sum — not the drop — that decides whether anything went missing:
///
/// * `x − x` folds `fl(1) + fl(−1)` to `0.0`, and that add is **exact**.
///   The term really is absent, the body really is degree 0, and the maps
///   are not a lower bound on anything.
/// * `2⁵³·x² + x² − 2⁵³·x²` loses the `x²` at `fl(2⁵³ + 1) = 2⁵³`, which is
///   an **inexact** add, and only then does `2⁵³ − 2⁵³` drop the survivor.
///   The body is `x²`; the maps say degree 0.
///
/// The loss happens at the inexact fold; the drop is only where it becomes
/// visible. So the flag is set from the *fold*: a form whose arithmetic
/// never rounded — and never flushed a live coefficient to zero — carries
/// coefficients that are exactly what real arithmetic on the writer's own
/// literals would give, and a term missing from its maps is a term that is
/// genuinely absent. Flagging the drop instead refused both of the above
/// alike, which was sound and cost reach on the first (gh #687).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Quad2 {
    constant: f64,
    linear: BTreeMap<usize, f64>,
    quadratic: QuadHessian,
    /// Set when a term may be **missing** from the maps — see the type
    /// docs and [`lost_terms`](Quad2::lost_terms). Sticky: it survives
    /// every operation the form takes part in, because a term that went
    /// missing three additions ago is no less missing now.
    lost_terms: bool,
    /// Set when some arithmetic in this form's construction **rounded**:
    /// an add that could not represent its sum, or a product that could
    /// not represent itself. Sticky, and for the same reason.
    ///
    /// Clear ⇒ every stored coefficient is exactly the real-arithmetic
    /// value of the terms folded into it, so a coefficient that reached
    /// zero reached it because the real value is zero. That is the whole
    /// warrant for `lost_terms` staying clear across a cancellation, and
    /// it is why an inexact *fold* sets this bit even when the coefficient
    /// it produced is nonzero and stays stored: what it rounded away can
    /// be cancelled out of sight later, by an add that is itself exact.
    inexact: bool,
}

impl Quad2 {
    /// The degree-0 term.
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// The degree-1 terms, ascending by variable index.
    pub fn linear(&self) -> &BTreeMap<usize, f64> {
        &self.linear
    }

    /// The degree-2 terms as *polynomial* coefficients keyed `(i ≤ j)` —
    /// the coefficient of `xᵢxⱼ`, **not** the Hessian entry (they differ by
    /// a factor of 2 on the diagonal; [`analyze_quadratic_full`] applies it).
    pub fn quadratic(&self) -> &QuadHessian {
        &self.quadratic
    }

    /// Whether a term may be **missing** from this form's maps: something
    /// was dropped on the way here *and* the arithmetic that produced it
    /// had already rounded, or a live coefficient was flushed to zero (or
    /// to `NaN`) outright.
    ///
    /// This is what turns the term maps from an answer about degree into a
    /// lower bound on it (gh #683). When it is set, an empty
    /// [`quadratic`](Quad2::quadratic) map is no longer evidence that the
    /// body is affine, and a consumer asking a *degree* question — as
    /// opposed to a storage or evaluation question — must say "not
    /// established" instead of "affine". The two questions used to share
    /// one predicate, which is how a genuinely quadratic row came to be
    /// reported as proved affine and had its Jacobian frozen for a whole
    /// solve.
    ///
    /// It was originally set by the **drop** alone, which conflated an
    /// exact cancellation with a lossy one and refused both (gh #687):
    /// `x − x` is degree 0 by an exact add, and a consumer that treats it
    /// as "not established" gives up a proved degree, a matrix evaluation
    /// and — once the classifier gates on this too — the convex route, for
    /// a body that lost nothing. The gate is now the fold, so the exact
    /// case keeps its fast paths and the lossy one is refused for the
    /// reason it deserves. [`inexact`](Quad2::inexact) is the other half of
    /// that answer.
    ///
    /// It is deliberately one bit for the whole form rather than a record
    /// per monomial: the consumer's question is about the form, and Q3's
    /// point was that a `Quad2` allocates nothing per term (gh #588, Q3).
    /// Two bits, now, and still nothing per term — the cost of a per-key
    /// provenance record is what keeps this a *conservative* answer: an
    /// inexact fold on one monomial makes a cancellation on an unrelated
    /// one count as lossy.
    ///
    /// A *linear* or constant term that goes missing sets it too, even
    /// though losing one cannot understate an affine body's degree by
    /// itself: it can once the form is **multiplied**, because
    /// [`mul`](Quad2::mul)'s degree guard reads the same maps —
    /// `(2⁵³ + 1 − 2⁵³)·x · x²` is degree 3 and folds to an apparent
    /// degree 0. Distinguishing the two would take a third flag to buy
    /// back reach that the corpus does not contain.
    pub fn lost_terms(&self) -> bool {
        self.lost_terms
    }

    /// Whether any arithmetic behind this form's coefficients **rounded**.
    ///
    /// Clear means every stored coefficient is exact, which is what makes a
    /// coefficient of zero a proof of absence rather than a lower bound —
    /// see [`lost_terms`](Quad2::lost_terms), which is this bit read at the
    /// moment a term is dropped. Public because it is the thing a test
    /// about the sharpened gate has to be able to see; no consumer routes
    /// on it.
    pub fn inexact(&self) -> bool {
        self.inexact
    }

    pub(crate) fn of_constant(c: f64) -> Self {
        Quad2 {
            // `-0.0` and `0.0` alike mean "no constant term"; normalizing
            // here keeps `Neg(Const(0.0))` from reporting `-0.0`.
            constant: if c != 0.0 { c } else { 0.0 },
            ..Quad2::default()
        }
    }

    pub(crate) fn of_var(i: usize) -> Self {
        let mut q = Quad2::default();
        q.linear.insert(i, 1.0);
        q
    }

    /// Total degree: 0, 1, or 2.
    pub(crate) fn degree(&self) -> usize {
        if !self.quadratic.is_empty() {
            2
        } else if !self.linear.is_empty() {
            1
        } else {
            0
        }
    }

    /// The value, when this form has no variables in it.
    pub(crate) fn as_constant(&self) -> Option<f64> {
        (self.degree() == 0).then_some(self.constant)
    }

    /// Number of stored (nonzero) variable terms.
    fn width(&self) -> usize {
        self.linear.len() + self.quadratic.len()
    }

    /// `a + b`.
    ///
    /// Two things here are not incidental. **Only the entries the smaller
    /// side contributes are re-checked for zero** — the predecessor
    /// re-scanned the whole accumulated map on every merge, which is `O(k²)`
    /// down a `k`-term `Add` chain, and an `o0` chain is how a long sum
    /// reaches this code. And **the smaller side is merged into the larger**,
    /// whichever operand it is, so the same chain costs `O(k log k)` leaning
    /// either way rather than only when it leans left. Choosing the direction
    /// is free of arithmetic consequence: IEEE addition is commutative bit
    /// for bit, `-0.0` included.
    pub(crate) fn add(a: Quad2, b: Quad2) -> Quad2 {
        let (mut acc, small) = if a.width() >= b.width() {
            (a, b)
        } else {
            (b, a)
        };
        // Whether anything either side already knew it had rounded. Read
        // once, before any merge, so the verdict below does not depend on
        // the order the maps happen to iterate in.
        let carried_inexact = acc.inexact || small.inexact;
        let (mut dropped, mut inexact) = (false, false);
        if small.constant != 0.0 {
            let was = acc.constant;
            acc.constant += small.constant;
            // Same rule as a coefficient, for the same reason: the degree-0
            // term is exempt from the "no stored zeros" invariant but not
            // from arithmetic, and `mul` reads `constant == 0.0` as
            // *annihilates*. A constant that cancelled *inexactly* is not
            // one that was proven absent (gh #683); one that cancelled
            // exactly is (gh #687).
            inexact |= !add_is_exact(was, small.constant, acc.constant);
            dropped |= was != 0.0 && acc.constant == 0.0;
        }
        for (i, c) in &small.linear {
            let m = merge(&mut acc.linear, *i, *c);
            dropped |= m.dropped;
            inexact |= m.inexact;
        }
        for (k, c) in &small.quadratic {
            let m = merge(&mut acc.quadratic, *k, *c);
            dropped |= m.dropped;
            inexact |= m.inexact;
        }
        // A term is only *lost* if something was dropped and some fold on
        // the way here could not represent itself. Neither half is enough
        // alone: an exact cancellation drops a term that was really zero,
        // and an inexact fold that keeps its coefficient has hidden
        // nothing — yet.
        acc.lost_terms |= small.lost_terms || (dropped && (carried_inexact || inexact));
        acc.inexact = carried_inexact || inexact;
        acc
    }

    pub(crate) fn neg(mut self) -> Quad2 {
        self.constant = -self.constant;
        for c in self.linear.values_mut() {
            *c = -*c;
        }
        for c in self.quadratic.values_mut() {
            *c = -*c;
        }
        self
    }

    /// `self · s`, for a scalar `s`.
    ///
    /// The `prune` is not decoration, and neither is the flag it sets. A
    /// nonzero coefficient times a nonzero `s` can still land on zero by
    /// underflow — `1e-300 x0²` divided by `1e300` is reachable from a real
    /// `.nl` body, through [`Op::Div`]'s reciprocal — so this is the
    /// gh #683 shape again, arrived at by scaling rather than by summing.
    /// Leaving the flushed entry stored would make an arithmetically
    /// constant form report degree 2 to [`Quad2::degree`] and be refused as
    /// degree 3 the moment anything multiplied it; dropping it without
    /// recording the loss would make a genuinely degree-2 body look affine
    /// to [`Quad2::lost_terms`]'s consumers. `prune` does the first and
    /// this does the second. (`neg` needs neither: negation cannot reach
    /// zero from a coefficient that was not already there, and it never
    /// rounds.)
    ///
    /// A coefficient here can only reach zero by **underflow** — both
    /// factors are nonzero — so unlike a cancelling sum there is no exact
    /// case to spare (gh #687): `10⁻³⁰⁰ · 10⁻³⁰⁰` is a term the form can no
    /// longer see, whatever the flags said before.
    pub(crate) fn scale(mut self, s: f64) -> Quad2 {
        if s == 0.0 {
            // An exact zero annihilates every term, so this is a proof of
            // degree 0 rather than a loss — nothing is being discarded that
            // survived the multiplication, and `x · 0` is exact. The flags
            // still ride along: the one route here is division by an
            // infinity, and `0 · NaN` is not zero.
            return Quad2 {
                lost_terms: self.lost_terms,
                inexact: self.inexact,
                ..Quad2::default()
            };
        }
        let was = self.constant;
        self.constant *= s;
        if was != 0.0 {
            self.lost_terms |= self.constant == 0.0;
            self.inexact |= !mul_is_exact(was, s, self.constant);
        }
        let mut inexact = false;
        for c in self.linear.values_mut() {
            let was = *c;
            *c *= s;
            inexact |= !mul_is_exact(was, s, *c);
        }
        for c in self.quadratic.values_mut() {
            let was = *c;
            *c *= s;
            inexact |= !mul_is_exact(was, s, *c);
        }
        self.inexact |= inexact;
        // A scale can underflow a live coefficient to zero, which is a lost
        // term and used to be stored as a zero.
        self.lost_terms |= self.prune();
        self
    }

    /// Restore the "no stored zeros" invariant after an operation wrote
    /// coefficients in bulk. Returns whether anything was dropped; what
    /// that *means* is the caller's to say, because it depends on how the
    /// coefficients got there (see [`scale`](Quad2::scale) and
    /// [`mul`](Quad2::mul)).
    fn prune(&mut self) -> bool {
        let before = self.width();
        self.linear.retain(|_, c| is_live(*c));
        self.quadratic.retain(|_, c| is_live(*c));
        self.width() != before
    }

    /// `self / d`, for a constant `d` the caller has already checked is
    /// nonzero.
    ///
    /// Scales by the **reciprocal** (not `c / d`) so the arithmetic matches
    /// what the recursive predecessor produced bit for bit; what is new is
    /// that `fl(1/d)` is the reciprocal only when `d` is a power of two,
    /// and `fma(r, d, −1) == 0` says so exactly. Recording it matters
    /// because a coefficient no real arithmetic produced can still cancel
    /// *exactly* against another one later — and that cancellation would
    /// otherwise be read as a proof of absence (gh #687).
    pub(crate) fn div_by_constant(self, d: f64) -> Quad2 {
        let r = 1.0 / d;
        let exact = r.is_normal() && d.is_normal() && r.mul_add(d, -1.0) == 0.0;
        let mut out = self.scale(r);
        out.inexact |= !exact;
        out
    }

    /// Take on another form's [`lost_terms`](Quad2::lost_terms) and
    /// [`inexact`](Quad2::inexact) history.
    ///
    /// The operand a `Div` or a `Pow` reads as a *constant* is a [`Quad2`]
    /// like any other, and its arithmetic is part of this form's: dividing
    /// by `fl(10²⁰⁰ · 10²⁰⁰)` is dividing by an infinity, and dividing by a
    /// constant that swallowed a variable term is worse than that. The
    /// value is read out through [`as_constant`](Quad2::as_constant), which
    /// keeps neither fact, so the caller hands the form itself over here.
    pub(crate) fn absorb_flags(&mut self, other: &Quad2) {
        self.lost_terms |= other.lost_terms;
        self.inexact |= other.inexact;
    }

    /// `self · other`, or `None` when the product would exceed total
    /// degree 2 — past that the recognizer gives up and the caller routes
    /// to the general NLP path.
    pub(crate) fn mul(&self, other: &Quad2) -> Option<Quad2> {
        if self.degree() + other.degree() > 2 {
            return None;
        }
        let mut out = Quad2::default();
        // Either operand's missing term is missing from the product too,
        // and so is either operand's rounding.
        let mut lost = self.lost_terms || other.lost_terms;
        let carried_inexact = self.inexact || other.inexact;
        let mut inexact = false;
        // A product of two live coefficients that lands on zero underflowed
        // — `(10⁻²⁰⁰·x)·(10⁻²⁰⁰·x)` is one monomial whose coefficient is not
        // representable (gh #683) — and unlike a cancelling *sum* there is
        // no exact case to spare, so `lost` is set on the spot rather than
        // left to the cancellation rule below.
        let product = |a: f64, b: f64, lost: &mut bool, inexact: &mut bool| -> f64 {
            let t = a * b;
            *lost |= !is_live(t);
            *inexact |= !mul_is_exact(a, b, t);
            t
        };
        if self.constant != 0.0 && other.constant != 0.0 {
            // `10⁻²⁰⁰ · 10⁻²⁰⁰` is not zero, and the branches below read a
            // zero constant as an annihilating one — which would take a
            // degree-2 product down to nothing without a trace (gh #683).
            out.constant = product(self.constant, other.constant, &mut lost, &mut inexact);
        }
        let mut dropped = false;
        // constant × (linear, quadratic), both ways round.
        for (a, b) in [(self, other), (other, self)] {
            if a.constant == 0.0 {
                continue;
            }
            for (i, c) in &b.linear {
                let t = product(a.constant, *c, &mut lost, &mut inexact);
                let m = accumulate(&mut out.linear, *i, t);
                dropped |= m.dropped;
                inexact |= m.inexact;
            }
            for (k, c) in &b.quadratic {
                let t = product(a.constant, *c, &mut lost, &mut inexact);
                let m = accumulate(&mut out.quadratic, *k, t);
                dropped |= m.dropped;
                inexact |= m.inexact;
            }
        }
        // linear × linear. The degree guard above means at most one of the
        // two operands carries quadratic terms, so this runs only when
        // neither does and no ordering question arises.
        for (i, a) in &self.linear {
            for (j, b) in &other.linear {
                let key = (*i.min(j), *i.max(j));
                let t = product(*a, *b, &mut lost, &mut inexact);
                let m = accumulate(&mut out.quadratic, key, t);
                dropped |= m.dropped;
                inexact |= m.inexact;
            }
        }
        // What `prune` drops here that `accumulate` did not already see is a
        // key whose *first* contribution was a zero — an underflowed product,
        // and `lost` is set for it above. So the cancellation rule is the
        // same one `add` applies: a drop counts as a loss only when some fold
        // behind it rounded.
        dropped |= out.prune();
        out.lost_terms = lost || (dropped && (carried_inexact || inexact));
        out.inexact = carried_inexact || inexact;
        Some(out)
    }
}

/// What folding one coefficient into a map did — the two facts
/// [`Quad2::lost_terms`] is decided from.
#[derive(Clone, Copy)]
struct Merged {
    /// The key is no longer stored: the fold reached exactly zero, or
    /// `NaN`.
    dropped: bool,
    /// The fold **rounded**: what is stored (or what cancelled) is not what
    /// exact arithmetic on the same two numbers would have left.
    inexact: bool,
}

/// Add `c` to `map[key]`, keeping the "no stored zeros" invariant. Returns
/// what that did, as [`Merged`].
///
/// Only the touched key can have become zero, which is what keeps a merge
/// proportional to what it merged rather than to what it merged *into*.
fn merge<K: Ord>(map: &mut BTreeMap<K, f64>, key: K, c: f64) -> Merged {
    use std::collections::btree_map::Entry;
    match map.entry(key) {
        Entry::Occupied(mut e) => {
            let a = *e.get();
            let v = a + c;
            let inexact = !add_is_exact(a, c, v);
            if is_live(v) {
                e.insert(v);
                Merged {
                    dropped: false,
                    inexact,
                }
            } else {
                e.remove();
                Merged {
                    dropped: true,
                    inexact,
                }
            }
        }
        // Nothing was added to anything: a live `c` is stored as it stands,
        // and a dead one is only reachable from a caller that already knows
        // what its own zero means (`mul`'s underflowed products; `add` only
        // ever passes stored — hence live — coefficients).
        Entry::Vacant(e) => {
            if is_live(c) {
                e.insert(c);
                Merged {
                    dropped: false,
                    inexact: false,
                }
            } else {
                Merged {
                    dropped: true,
                    inexact: c.is_nan(),
                }
            }
        }
    }
}

/// Accumulate `t` into `map[key]`, the way [`Quad2::mul`] builds a product
/// up term by term, and report what the fold did.
///
/// Unlike [`merge`] this leaves a zero stored — `mul` prunes once at the
/// end — because a key it lands on twice must see the first contribution as
/// a plain `0.0 + t`, bit for bit what the `+=` this replaced produced.
/// Only a key that was already nonzero can be said to have *dropped*
/// anything.
fn accumulate<K: Ord>(map: &mut BTreeMap<K, f64>, key: K, t: f64) -> Merged {
    let slot = map.entry(key).or_insert(0.0);
    let was = *slot;
    *slot = was + t;
    Merged {
        dropped: was != 0.0 && !is_live(*slot),
        inexact: !add_is_exact(was, t, *slot),
    }
}

/// The exactness predicates this recognizer's fold is decided from live in
/// `pounce-common` since gh #673, because the `.nl` pipeline grew a second
/// fold — `Σ 2wₖbₖbₖᵀ` in `QuadraticStructure::push_factored_form` — in a
/// crate that cannot name this one. See [`pounce_common::exact`] for what
/// they answer and why it is an exact question rather than a tolerance.
use pounce_common::exact::{add_is_exact, is_live, mul_is_exact};

/// One entry on the recognizer's explicit work stack.
enum Step<'a> {
    /// Lower this subexpression onto the value stack.
    Visit(&'a Expr),
    /// Combine values already on the value stack.
    Apply(Op),
}

/// A pending combination, popped once its operands have been lowered.
enum Op {
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    /// n-ary sum over the top `n` values.
    Sum(usize),
    /// Not an operation: the value now on top of the stack is the lowering
    /// of the `Cse` body with this address, so record it before moving on.
    CacheCse(*const Expr),
}

/// Lower an [`Expr`] to a [`Quad2`], or `None` if it contains anything the
/// recognizer cannot prove is a degree-≤2 polynomial (transcendental ops,
/// division by a non-constant, `Pow` with an exponent ∉ {0, 1, 2}, products
/// of degree > 2, external calls, comparisons, `if-then-else`, `min`/`max`,
/// …). `None` ⇒ treat as general nonlinear.
///
/// `Cse` nodes are inlined: a reference is mathematically its body, and
/// every reference is an independent occurrence. A body is nevertheless
/// lowered **once**, and its [`Quad2`] reused at every later reference
/// (keyed on `Arc` identity). That is what makes the walk `Θ(nodes)` on a
/// shared DAG instead of `Θ(2^depth)`; Q4 had to refuse a re-referenced
/// body outright to bound the cost, and this is the memoization that
/// refusal was waiting for (gh #588, Q5). It is bitwise neutral — the
/// lowering of a body is a function of the body, so the reused value is
/// the one a second lowering would have produced, bit for bit.
///
/// The walk is iterative. See the module docs for why that is a
/// correctness property and not a style choice.
pub fn recognize_expr(e: &Expr) -> Option<Quad2> {
    let mut work: Vec<Step<'_>> = vec![Step::Visit(e)];
    let mut vals: Vec<Quad2> = Vec::new();
    // Lowered `Cse` bodies, by address. Only successes land here: a body
    // that fails aborts the whole walk, so there is no negative result to
    // remember.
    let mut cse: std::collections::HashMap<*const Expr, Quad2> = std::collections::HashMap::new();

    while let Some(step) = work.pop() {
        match step {
            Step::Visit(e) => match e {
                Expr::Const(c) => vals.push(Quad2::of_constant(*c)),
                Expr::Var(i) => vals.push(Quad2::of_var(*i)),
                Expr::Cse(body) => {
                    let key = std::sync::Arc::as_ptr(body);
                    match cse.get(&key) {
                        Some(q) => vals.push(q.clone()),
                        None => {
                            work.push(Step::Apply(Op::CacheCse(key)));
                            work.push(Step::Visit(body));
                        }
                    }
                }
                Expr::Sum(items) => {
                    work.push(Step::Apply(Op::Sum(items.len())));
                    // Pushed forward, so they pop back to front and land on
                    // the value stack with item 0 on top — which puts item
                    // 0 at the *end* of the region `Op::Sum` drains, and
                    // item n-1 at its start. The fold there walks that
                    // region in reverse so the sum still accumulates front
                    // to back, the order the recursive version summed in.
                    for it in items {
                        work.push(Step::Visit(it));
                    }
                }
                Expr::Unary(UnaryOp::Neg, a) => {
                    work.push(Step::Apply(Op::Neg));
                    work.push(Step::Visit(a));
                }
                // Every other unary op is transcendental.
                Expr::Unary(..) => return None,
                Expr::Binary(op, a, b) => {
                    let op = match op {
                        BinOp::Add => Op::Add,
                        BinOp::Sub => Op::Sub,
                        BinOp::Mul => Op::Mul,
                        BinOp::Div => Op::Div,
                        BinOp::Pow => Op::Pow,
                        // atan2 and any other binary opcode.
                        _ => return None,
                    };
                    work.push(Step::Apply(op));
                    // `b` under `a`: `a` pops first and is lowered first.
                    work.push(Step::Visit(b));
                    work.push(Step::Visit(a));
                }
                // External calls are opaque; comparisons, logicals,
                // conditionals and n-ary min/max are the control-flow `.nl`
                // opcodes. None is provably polynomial ⇒ route to NLP.
                _ => return None,
            },
            Step::Apply(Op::CacheCse(key)) => {
                // The body's value is already on the stack and stays there;
                // this only records it for the next reference.
                cse.insert(key, vals.last()?.clone());
            }
            Step::Apply(op) => {
                let combined = match op {
                    Op::CacheCse(_) => unreachable!("handled above"),
                    Op::Sum(n) => {
                        // The items are the top `n` values, item 0 on top,
                        // so item 0 is the *last* thing `drain` yields and
                        // item n-1 the first. Reverse it: floating-point
                        // addition is not associative, and summing back to
                        // front would disagree with the recursive
                        // predecessor — and with the AD tape — by an ulp
                        // whenever two items share a monomial key.
                        let at = vals.len().checked_sub(n)?;
                        let mut acc = Quad2::default();
                        for p in vals.drain(at..).rev() {
                            acc = Quad2::add(acc, p);
                        }
                        acc
                    }
                    Op::Neg => vals.pop()?.neg(),
                    Op::Add => {
                        let (a, b) = pop2(&mut vals)?;
                        Quad2::add(a, b)
                    }
                    Op::Sub => {
                        let (a, b) = pop2(&mut vals)?;
                        Quad2::add(a, b.neg())
                    }
                    Op::Mul => {
                        let (a, b) = pop2(&mut vals)?;
                        a.mul(&b)?
                    }
                    Op::Div => {
                        // Division is polynomial only by a nonzero constant,
                        // and scales by the reciprocal (not `c / d`) so the
                        // arithmetic matches what the recursive predecessor
                        // produced bit for bit.
                        let (a, b) = pop2(&mut vals)?;
                        let d = b.as_constant()?;
                        if d == 0.0 {
                            return None;
                        }
                        let mut out = a.div_by_constant(d);
                        // The divisor's own history comes along: `as_constant`
                        // reads a number out of a form and leaves the flags
                        // behind.
                        out.absorb_flags(&b);
                        out
                    }
                    Op::Pow => {
                        // Polynomial only for constant exponents in {0, 1, 2}.
                        let (a, b) = pop2(&mut vals)?;
                        let exp = b.as_constant()?;
                        let mut out = if exp == 0.0 {
                            Quad2::of_constant(1.0)
                        } else if exp == 1.0 {
                            a
                        } else if exp == 2.0 {
                            a.mul(&a)?
                        } else {
                            return None;
                        };
                        // Same as `Div`: the exponent is read out of a form,
                        // and one that lost a term is not the exponent it
                        // looks like.
                        out.absorb_flags(&b);
                        out
                    }
                };
                vals.push(combined);
            }
        }
    }

    debug_assert_eq!(vals.len(), 1, "one value per lowered expression");
    vals.pop()
}

/// Pop a binary operator's two operands, left first.
fn pop2(vals: &mut Vec<Quad2>) -> Option<(Quad2, Quad2)> {
    let b = vals.pop()?;
    let a = vals.pop()?;
    Some((a, b))
}

/// Attempt to read an expression as a polynomial of total degree ≤ 2 and
/// return its Hessian (constant, since the form is quadratic). `None` if
/// the expression is not provably quadratic ⇒ treat as general nonlinear.
pub fn analyze_quadratic(e: &Expr) -> Option<QuadHessian> {
    analyze_quadratic_full(e).map(|(h, _, _)| h)
}

/// Like [`analyze_quadratic`] but also returns the degree-1 (linear)
/// coefficients *and* the degree-0 (constant) term of the form:
/// `(Hessian, [(var, coef), …], constant)`.
///
/// AMPL folds the linear part of a nonlinear term into the objective's
/// nonlinear expression tree (the `−6·x₀` of `(x₀−3)²`, say) rather than
/// the linear section. Callers building the QP objective vector `c` must
/// add these in, exactly as the NLP path's `eval_f` sums the linear
/// section *and* the nonlinear tree — otherwise the linear shift is
/// silently dropped and the convex solve minimizes the wrong objective.
///
/// The **constant** is returned for the same reason: AMPL/Pyomo also fold
/// the objective's degree-0 term into the nonlinear tree (the `+9` of
/// `(x₀−3)²`), where it does *not* land in `NlProblem::obj_constant`. It
/// is irrelevant to the minimizer but is part of the *reported objective
/// value*; dropping it makes the convex solve report an objective off by
/// that constant versus the NLP path.
pub fn analyze_quadratic_full(e: &Expr) -> Option<QuadForm> {
    Some(quad_form_readout(&recognize_expr(e)?))
}

/// The `(Hessian, linear, constant)` read-out of an already-recognized
/// form — the second half of [`analyze_quadratic_full`], split out so a
/// caller holding a [`Quad2`] the *parser* produced (gh #588, Q5) reaches
/// the identical numbers by the identical route. There is exactly one
/// conversion in the crate; a second one is a second thing to keep in step.
pub fn quad_form_readout(q: &Quad2) -> QuadForm {
    // ∂²(c·xᵢxⱼ)/∂xᵢ∂xⱼ = c for i≠j; ∂²(c·xᵢ²)/∂xᵢ² = 2c.
    let mut h: QuadHessian = q
        .quadratic
        .iter()
        .map(|(&(i, j), c)| ((i, j), if i == j { 2.0 * c } else { *c }))
        .collect();
    // Drop explicit zeros so `is_empty()` means "linear".
    h.retain(|_, v| v.abs() > 0.0);
    let lin: Vec<(usize, f64)> = q.linear.iter().map(|(i, c)| (*i, *c)).collect();
    // `0.0 +` normalizes `-0.0`, which is how this form spells "absent".
    (h, lin, 0.0 + q.constant)
}

/// Is this expression **already** a flat sum of monomials — that is, would
/// reading it as `½xᵀHx + aᵀx + c` reproduce the same additions the writer
/// wrote, rather than algebraically expanding something it did not?
///
/// This is the gate on the constant-structure evaluator (gh #588, Q4), and
/// it exists because expanding a quadratic is not an accuracy-neutral
/// rewrite. `(xᵢ − xⱼ)²` evaluated as written squares a small residual;
/// evaluated as `xᵢ² − 2xᵢxⱼ + xⱼ²` it cancels two large numbers, and on
/// `airport.nl` — 84 coordinates around 10³, every row a squared distance —
/// that difference is enough to take an adaptive-μ solve from stopping at a
/// tiny step in 16 iterations to grinding out the 300-iteration cap at the
/// same objective. Which is precisely the gh #544 failure mode: the right
/// answer, slowly.
///
/// So the rule is *exactness*, not magnitude: a form is admitted only when
/// the recognizer's read-out does no algebra the tape would not have done.
/// A sum of monomials qualifies — which is exactly how AMPL emits the
/// `qcqp*` family (`o54` over `o2 n0.5 o2 o2 n<c> v<i> v<j>`), so the target
/// of the phase is unaffected — and a `Pow` or a `Mul` over a non-atomic
/// operand does not.
///
/// What this predicate is **not** is a verdict on whether a body can be
/// evaluated from constant structure at all. It is a verdict on one
/// *representation*. Q4 shipped as though the two were the same, and the
/// consequence was that a model written in factored form kept its tape and
/// gained nothing — 41 of `airport.nl`'s 42 rows, and every sum of squared
/// residuals ever written. gh #673 is that gap, and
/// [`recognize_factored_quadratic`] closes it by keeping the writer's own
/// grouping instead of expanding it: same constant structure, no algebra
/// the tape did not do. A body this predicate refuses is offered to that
/// one before it falls back to a tape.
///
/// ## A re-referenced `Cse` body is *skipped*, not refused
///
/// This walk and [`recognize_expr`] behind it both inline a `Cse` body at
/// **every** reference, so a DAG whose bodies are shared cost `Θ(2^depth)`
/// rather than `Θ(nodes)` before either was memoized. `nl_reader`'s
/// `shared_dag_walks_are_memoized_not_exponential` builds exactly that shape
/// — 30 levels, each a `Cse` referenced twice — and taking it through this
/// gate in Q4 measured **0.00 s → 172 s** on a model the tape path loads
/// instantly. At depth 40 it would not have returned at all.
///
/// Q4 bounded that by ending the walk in `false` at the second reference,
/// which cost reach to buy termination and was explicitly left for Q5 to
/// replace. It is replaced here: a body reached a second time in the same
/// mode is **skipped**, which is sound for exactly the reason
/// `nl_reader::validate_expr` documents for the same trick — this walk
/// aborts on the first violation, so reaching a body a second time proves
/// the first visit found none. The verdict is unchanged for every DAG that
/// is a tree, and a shared DAG that *is* an expanded quadratic is now
/// admitted instead of refused. The mode is part of the key: a body is
/// legal on the sum spine if it is a sum of monomials, and legal inside a
/// monomial only if it is itself a monomial, so the two answers are cached
/// apart.
///
/// Iterative for the same reason [`recognize_expr`] is.
pub fn is_expanded_quadratic(e: &Expr) -> bool {
    // The sum spine: `Add`/`Sub`/`Neg`/`Sum` may nest freely, and every
    // leaf of that spine must be a monomial.
    let mut seen: BTreeSet<(*const Expr, bool)> = BTreeSet::new();
    let mut spine: Vec<&Expr> = vec![e];
    while let Some(e) = spine.pop() {
        match e {
            Expr::Sum(items) => spine.extend(items.iter()),
            Expr::Binary(BinOp::Add | BinOp::Sub, a, b) => {
                spine.push(a);
                spine.push(b);
            }
            Expr::Unary(UnaryOp::Neg, a) => spine.push(a),
            Expr::Cse(body) => {
                if seen.insert((std::sync::Arc::as_ptr(body), false)) {
                    spine.push(body);
                }
            }
            other => {
                if !is_monomial(other, &mut seen) {
                    return false;
                }
            }
        }
    }
    true
}

/// Is this expression a single monomial — the leaf shape
/// [`is_expanded_quadratic`] admits on its sum spine?
///
/// Public because the parse-time recognizer has to answer the same
/// question about a `V`-segment body it has already parsed, and must
/// answer it with *this* code rather than a second copy of the rule.
pub fn is_monomial_expr(e: &Expr) -> bool {
    let mut seen: BTreeSet<(*const Expr, bool)> = BTreeSet::new();
    is_monomial(e, &mut seen)
}

/// A single product term: constants and variables multiplied together, with
/// no addition anywhere inside it. `xᵢxⱼ`, `0.5·c·xᵢ·xⱼ`, `xᵢ²` and `xᵢ/3`
/// all qualify; `(xᵢ − xⱼ)²` and `(xᵢ + 1)·xⱼ` do not.
///
/// Degree is not checked here — [`recognize_expr`] already refused anything
/// past 2 by the time this runs, and duplicating the rule would only give
/// the two a way to disagree.
fn is_monomial(e: &Expr, seen: &mut BTreeSet<(*const Expr, bool)>) -> bool {
    let mut work: Vec<&Expr> = vec![e];
    while let Some(e) = work.pop() {
        match e {
            Expr::Const(_) | Expr::Var(_) => {}
            // Shared with the spine walk, and skipped on revisit for the
            // same reason — see [`is_expanded_quadratic`]. The `true` in the
            // key is "seen in monomial mode": a body cleared on the spine
            // has not been cleared here.
            Expr::Cse(body) => {
                if seen.insert((std::sync::Arc::as_ptr(body), true)) {
                    work.push(body);
                }
            }
            Expr::Unary(UnaryOp::Neg, a) => work.push(a),
            Expr::Binary(BinOp::Mul | BinOp::Div, a, b) => {
                work.push(a);
                work.push(b);
            }
            // `x^2` is one monomial; `(x - y)^2` is an expansion. The
            // exponent itself may be any constant expression — the
            // recognizer has already restricted it to {0, 1, 2}.
            Expr::Binary(BinOp::Pow, a, b) => {
                if !matches!(
                    a.as_ref(),
                    Expr::Const(_) | Expr::Var(_) | Expr::Unary(UnaryOp::Neg, _)
                ) {
                    return false;
                }
                work.push(a);
                work.push(b);
            }
            _ => return false,
        }
    }
    true
}

/// True if the expression is the literal constant zero the `.nl` reader
/// uses for "no nonlinear part".
pub fn is_trivially_zero(e: &Expr) -> bool {
    matches!(e, Expr::Const(c) if *c == 0.0)
}

// ---------------------------------------------------------------------
// Factored forms — the writer's own grouping, kept
// ---------------------------------------------------------------------

/// One `w·(bᵀx + d)²` term of a [`FactoredQuadratic`].
///
/// `coefs` is `b`, ascending by variable index and free of stored zeros —
/// the same convention [`Quad2::linear`] keeps, because that is where it
/// comes from. It may be **empty**, which is a square of a constant: the
/// term is then `w·d²`, contributes nothing to the gradient or the
/// Hessian, and is kept rather than folded so that the value is still
/// computed the way it was written.
#[derive(Debug, Clone, PartialEq)]
pub struct SquaredAffine {
    /// The constant the square is multiplied by, sign of the sum spine
    /// already folded in.
    pub weight: f64,
    /// `b`, ascending by variable index.
    pub coefs: Vec<(usize, f64)>,
    /// `d`.
    pub constant: f64,
}

/// A degree-2 form kept as `Σ wₖ(bₖᵀx + dₖ)² + aᵀx + c` — the shape the
/// `.nl` writer wrote, rather than its expansion about the origin.
///
/// This is the representation gh #673 asks for and
/// [`is_expanded_quadratic`] exists to refuse the alternative to. Reading
/// `(x − 500000)²` back as `x² − 10⁶x + 2.5·10¹¹` cancels five digits;
/// reading it back as one squared residual repeats exactly the
/// multiplication the tape performs, so admitting it costs no accuracy at
/// all.
///
/// The degree-≤1 leftovers (`linear`, `constant`) are the monomials the
/// writer folded into the same tree — the `+ 3y` of `(x − 1)² + 3y`. They
/// are summed as coefficients rather than kept term by term, which is the
/// same reassociation the expanded path already makes and is gated the
/// same way (see [`Quad2::lost_terms`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FactoredQuadratic {
    /// The squared affine terms, in the order the spine walk met them.
    pub squares: Vec<SquaredAffine>,
    /// `a`, ascending by variable index.
    pub linear: Vec<(usize, f64)>,
    /// `c`.
    pub constant: f64,
}

/// Recognize a degree-2 body as a sum of **squared affine forms** plus
/// degree-≤1 leftovers, keeping the squares factored (gh #673).
///
/// This is the second half of the gate [`is_expanded_quadratic`] opens.
/// That predicate admits a body whose read-out repeats the additions the
/// writer wrote; a factored body fails it, and used to keep its AD tape for
/// good reason — expanding it to the origin cancels. What this returns is
/// the *third* option: a read-out that is not an expansion either, because
/// it stores the writer's own linear forms and squares them at evaluation
/// time exactly as the tape does.
///
/// ## What is admitted
///
/// The sum spine (`Add`/`Sub`/`Neg`/`Sum`) may nest freely. Every leaf of
/// it must be one of:
///
/// * a **square**: a product of constants with exactly one `Pow(base, 2)`,
///   where `base` is itself an [`is_expanded_quadratic`] body of degree
///   ≤ 1 — `(xᵢ − xⱼ)²`, `0.5·(2x − y + 3)²`, `x²`;
/// * a **degree-2 monomial on the diagonal**, `c·xᵢ²`, which is the same
///   term wearing a different opcode and is stored as `c·(xᵢ)²`;
/// * a **degree-≤1 monomial**, which folds into `a` and `c`.
///
/// Anything else — a cross monomial `c·xᵢxⱼ`, a product of two *different*
/// affine forms, a transcendental — refuses the whole body, which then
/// keeps its tape exactly as it did before this function existed. The
/// refusal is deliberate rather than incidental: a mixed form would need
/// both an expanded and a factored quadratic part stored side by side, and
/// nothing in the corpus asks for one.
///
/// At least one square with a variable in it is required. Without that the
/// body is affine or already expanded, and [`is_expanded_quadratic`] is
/// the path that serves it.
///
/// ## Exactness
///
/// Each admitted square evaluates as `w·(d + Σbᵢxᵢ)²`. Against the tape
/// that is a reassociation of an affine sum and a folding of the writer's
/// own constants into `w` — the same latitude [`Quad2::add`] and
/// [`Quad2::scale`] already take on the expanded path — and *not* an
/// algebraic expansion. The `2.4e-5` disagreement gh #673 records for
/// `(x − 500000)²` comes from the expansion; there is none here. Measured
/// end to end: with the outer sum left naive, `airport.nl`'s 42 rows
/// evaluate bit-identically to the 42 tapes they replace.
///
/// The *terms* are the tape's; the sum over them is not, because
/// `QuadraticStructure::value` compensates it (gh #702). That makes the
/// read-out slightly better than the tape rather than equal to it, which is
/// gh #702's deliberate trade and the one line the fixture sweep moves.
///
/// The degree-≤1 leftovers are held to the same [`Quad2::lost_terms`] gate
/// the expanded path uses (gh #685), so a leftover that cancelled inexactly
/// refuses the body rather than dropping a term out of it.
///
/// ## A `Cse` on the sum spine refuses the body
///
/// [`is_expanded_quadratic`] may *skip* a re-referenced body because it
/// answers yes/no and the first visit already decided it. This one
/// accumulates terms, so skipping would silently drop them and visiting
/// every reference is `Θ(2^depth)` on a shared DAG — the shape
/// `nl_reader`'s `shared_dag_walks_are_memoized_not_exponential` builds.
/// So a `Cse` reached on the spine ends the walk, and the body keeps its
/// tape, which is what `ConHybrid` is for. Inside a square's base or a
/// monomial there is no such problem: [`recognize_expr`] memoizes on `Arc`
/// identity and the two predicates answer yes/no.
///
/// Iterative, for the reason the module docs give.
pub fn recognize_factored_quadratic(e: &Expr) -> Option<FactoredQuadratic> {
    let mut seen: BTreeSet<(*const Expr, bool)> = BTreeSet::new();
    // The spine, each entry carrying the sign the enclosing `Sub`/`Neg`
    // chain gives it. `±1` exactly, so folding it into a weight or negating
    // a form is not arithmetic.
    let mut spine: Vec<(&Expr, f64)> = vec![(e, 1.0)];
    let mut squares: Vec<SquaredAffine> = Vec::new();
    let mut rest = Quad2::default();

    while let Some((e, sign)) = spine.pop() {
        match e {
            // Pushed back to front so the leaves *pop* in source order.
            // The degree-≤1 leftovers are folded into one `Quad2` by
            // floating-point addition, and source order is the order a
            // reader can reason about — not, to be exact about what this
            // buys, the same association `recognize_expr` uses: that folds
            // the *tree*, and `Quad2::add` additionally swaps its operands
            // by map width. So the two can still differ in the last ulp on
            // a leaning spine. What keeps that from mattering is
            // `lost_terms`, which refuses any body where the difference
            // could be a dropped term rather than a rounded one.
            Expr::Sum(items) => spine.extend(items.iter().rev().map(|it| (it, sign))),
            Expr::Binary(BinOp::Add, a, b) => {
                spine.push((b, sign));
                spine.push((a, sign));
            }
            Expr::Binary(BinOp::Sub, a, b) => {
                spine.push((b, -sign));
                spine.push((a, sign));
            }
            Expr::Unary(UnaryOp::Neg, a) => spine.push((a, -sign)),
            // See the docs: accumulating terms is what makes skipping
            // unsound here, and visiting is what makes it exponential.
            Expr::Cse(_) => return None,
            leaf => {
                if let Some((weight, base)) = peel_square(leaf) {
                    squares.push(admit_square(sign * weight, base)?);
                    continue;
                }
                if !is_monomial(leaf, &mut seen) {
                    return None;
                }
                let q = recognize_expr(leaf)?;
                match diagonal_square(&q) {
                    Some((i, c)) => squares.push(SquaredAffine {
                        weight: sign * c,
                        coefs: vec![(i, 1.0)],
                        constant: 0.0,
                    }),
                    // A cross term `c·xᵢxⱼ` lands here and refuses: it is
                    // not a square, and storing it expanded alongside the
                    // squares is the mixed representation this refuses.
                    None if !q.quadratic().is_empty() => return None,
                    None => rest = Quad2::add(rest, if sign < 0.0 { q.neg() } else { q }),
                }
            }
        }
    }

    // Nothing factored to keep ⇒ this is not our case. `is_expanded_quadratic`
    // serves an expanded body and an affine one needs no form at all.
    if !squares.iter().any(|t| !t.coefs.is_empty()) {
        return None;
    }
    // The leftovers are held to the expanded path's gate, for its reasons
    // (gh #685): a term that went missing inexactly is a term the form no
    // longer evaluates.
    if !rest.quadratic().is_empty() || rest.lost_terms() {
        return None;
    }
    Some(FactoredQuadratic {
        squares,
        linear: rest.linear().iter().map(|(&i, &c)| (i, c)).collect(),
        // `0.0 +` normalizes `-0.0`, which is how a `Quad2` spells "absent".
        constant: 0.0 + rest.constant(),
    })
}

/// `c·xᵢ²` read off a recognized monomial: the one degree-2 shape that is
/// a square without being written as one. `None` for anything else,
/// including a cross term and anything carrying a linear or constant part.
///
/// The coefficient is the **polynomial** one — `Quad2::quadratic` is not
/// the Hessian — so `c·xᵢ²` is stored with weight `c` and not `2c`.
fn diagonal_square(q: &Quad2) -> Option<(usize, f64)> {
    if q.lost_terms() || !q.linear().is_empty() || q.constant() != 0.0 {
        return None;
    }
    match q.quadratic().iter().next() {
        Some((&(i, j), &c)) if i == j && q.quadratic().len() == 1 => Some((i, c)),
        _ => None,
    }
}

/// Turn `weight · base²` into a stored term, or refuse the body.
///
/// The base has to clear both of the expanded path's gates — the shape gate
/// [`is_expanded_quadratic`] (so reading its coefficients back repeats the
/// writer's own additions) and the [`Quad2::lost_terms`] gate (so none of
/// them went missing) — and be degree ≤ 1, which is what makes the product
/// a square rather than a quartic.
fn admit_square(weight: f64, base: &Expr) -> Option<SquaredAffine> {
    if !is_expanded_quadratic(base) {
        return None;
    }
    let q = recognize_expr(base)?;
    if !q.quadratic().is_empty() || q.lost_terms() {
        return None;
    }
    Some(SquaredAffine {
        weight,
        coefs: q.linear().iter().map(|(&i, &c)| (i, c)).collect(),
        constant: 0.0 + q.constant(),
    })
}

/// Split a leaf into `(constant weight, squared base)`, or `None` when it
/// is not a constant multiple of exactly one square.
///
/// The product tree may nest `Mul`, `Div` and `Neg` freely; every leaf of
/// it must be a constant except for the single `Pow(base, 2)`, which may
/// not sit under a division. Folding several constants into one `weight`
/// reassociates the writer's own constants and nothing else.
///
/// The order is worth stating precisely, because it is not source order:
/// this pops an explicit stack, so `a·(b·(c·s))` folds as `((1·c)·b)·a`.
/// One multiply's worth of rounding either way, on constants the writer
/// wrote next to each other — the same latitude `Quad2::scale` takes on the
/// expanded arm — but a reader reconstructing a last-ulp difference by hand
/// needs the real order, not the written one.
fn peel_square(e: &Expr) -> Option<(f64, &Expr)> {
    let mut work: Vec<(&Expr, bool)> = vec![(e, false)];
    let mut weight = 1.0f64;
    let mut base: Option<&Expr> = None;
    while let Some((e, recip)) = work.pop() {
        match e {
            Expr::Const(c) => weight = if recip { weight / c } else { weight * c },
            Expr::Unary(UnaryOp::Neg, a) => {
                weight = -weight;
                work.push((a, recip));
            }
            Expr::Binary(BinOp::Mul, a, b) => {
                work.push((a, recip));
                work.push((b, recip));
            }
            Expr::Binary(BinOp::Div, a, b) => {
                work.push((a, recip));
                work.push((b, !recip));
            }
            Expr::Binary(BinOp::Pow, a, b) if matches!(b.as_ref(), Expr::Const(c) if *c == 2.0) => {
                // A square under a division is `1/(…)²`, which is not one.
                // A second square makes the leaf degree 4.
                if recip || base.is_some() {
                    return None;
                }
                base = Some(a);
            }
            _ => return None,
        }
    }
    // A weight that is not finite (`x²/0`) is not a form anyone should
    // evaluate from stored coefficients, and one that is exactly zero has
    // annihilated a term the tape still computes.
    let base = base?;
    (weight.is_finite() && weight != 0.0).then_some((weight, base))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(i: usize) -> Expr {
        Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(i)),
            Box::new(Expr::Const(2.0)),
        )
    }

    #[test]
    fn quadratic_diagonal() {
        // (x0 - 1)^2  =>  x0^2 - 2 x0 + 1
        let e = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Binary(
                BinOp::Sub,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Const(2.0)),
        );
        let (h, lin, c) = analyze_quadratic_full(&e).expect("degree-2 polynomial");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
        assert_eq!(lin, vec![(0, -2.0)]);
        assert_eq!(c, 1.0);
    }

    #[test]
    fn cross_term_hessian() {
        // x0 · x1 => H[0,1] = 1
        let e = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 1)), Some(&1.0));
    }

    #[test]
    fn rejects_transcendental_and_cubic() {
        assert!(analyze_quadratic(&Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)))).is_none());
        let cubic = Expr::Binary(
            BinOp::Pow,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Const(3.0)),
        );
        assert!(analyze_quadratic(&cubic).is_none());
        // x0² · x1 — degree 3 by multiplication rather than by exponent.
        let deg3 = Expr::Binary(BinOp::Mul, Box::new(sq(0)), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&deg3).is_none());
    }

    #[test]
    fn division_by_a_constant_scales_by_the_reciprocal() {
        // x0² / 3 — the coefficient must be `2 · (1/3)`, which is what the
        // Hessian of `x0²/3` came out as before this module existed, and is
        // *not* bitwise `2/3`.
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Const(3.0)));
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 0)), Some(&(2.0 * (1.0 / 3.0))));
        // Division by a variable is not polynomial.
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&e).is_none());
    }

    #[test]
    fn scaling_a_coefficient_to_zero_drops_it_like_cancellation_does() {
        // 1e-300·x0² divided by 1e300. Neither the coefficient nor the
        // divisor is zero, so the whole-form `s == 0.0` shortcut does not
        // fire — the product underflows instead, and the entry has to go
        // the same way a cancelled one does.
        let tiny = Expr::Binary(BinOp::Mul, Box::new(Expr::Const(1e-300)), Box::new(sq(0)));
        let flushed = Expr::Binary(BinOp::Div, Box::new(tiny), Box::new(Expr::Const(1e300)));
        let h = analyze_quadratic(&flushed).expect("degree-2 at worst");
        assert!(
            h.is_empty(),
            "underflowed coefficient was kept as a structural nonzero: {h:?}"
        );
        // Storage and degree are the two halves of gh #683 and this route
        // reaches both: the entry is gone from the map, *and* the form
        // says so, so a consumer asking whether the body is affine gets
        // "not established" rather than "yes".
        let q = recognize_expr(&flushed).expect("degree-2 at worst");
        assert!(
            q.lost_terms(),
            "a coefficient that underflowed in `scale` was dropped silently"
        );
        // And the degree has to go with it for the *storage* question:
        // multiplying by another variable must still be recognized rather
        // than refused as degree 3.
        let times_x1 = Expr::Binary(BinOp::Mul, Box::new(flushed), Box::new(Expr::Var(1)));
        assert!(
            analyze_quadratic(&times_x1).is_some(),
            "a form scaled to nothing was refused as degree 3"
        );
    }

    #[test]
    fn cancellation_drops_the_term_and_the_degree_with_it() {
        // x0² − x0² is linear (empty Hessian), not a quadratic with a zero
        // coefficient — otherwise `x0²−x0²` times `x1` would be refused as
        // degree 3.
        let zero = Expr::Binary(BinOp::Sub, Box::new(sq(0)), Box::new(sq(0)));
        let h = analyze_quadratic(&zero).expect("degree-2 at worst");
        assert!(h.is_empty());
        let times_x1 = Expr::Binary(BinOp::Mul, Box::new(zero), Box::new(Expr::Var(1)));
        assert!(analyze_quadratic(&times_x1).is_some());
    }

    /// The gh #687 distinction, at the level it is made: an **exact**
    /// cancellation leaves the form complete, and an absorbed term does
    /// not. Both bodies drop a coefficient; only one of them lost anything
    /// doing it.
    #[test]
    fn an_exact_cancellation_is_not_a_lost_term() {
        // x0² − x0²: `fl(1) + fl(−1)` is exactly `0`, so the body really is
        // degree 0 and the maps say so with nothing held back.
        let zero = Expr::Binary(BinOp::Sub, Box::new(sq(0)), Box::new(sq(0)));
        let q = recognize_expr(&zero).expect("degree-2 at worst");
        assert!(q.quadratic().is_empty());
        assert!(!q.inexact(), "an exact fold reported rounding");
        assert!(
            !q.lost_terms(),
            "x0² − x0² was refused a fast path it is entitled to",
        );

        // 2⁵³·x0² + x0² − 2⁵³·x0², front to back: the `x0²` is absorbed by
        // `fl(2⁵³ + 1) = 2⁵³` — an inexact add — and the exact `− 2⁵³` only
        // makes the loss visible.
        let big = (1u64 << 53) as f64;
        let scaled = |c: f64| Expr::Binary(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(sq(0)));
        let absorbing = Expr::Sum(vec![scaled(big), sq(0), scaled(-big)]);
        let q = recognize_expr(&absorbing).expect("degree-2 at worst");
        assert!(q.quadratic().is_empty());
        assert!(q.inexact(), "the absorbing add was not seen to round");
        assert!(
            q.lost_terms(),
            "a body whose x0² was absorbed was reported complete",
        );
    }

    /// The absorption is recorded where it happens, which is the point of
    /// the sharpened gate: `2⁵³·x0² + x0²` has lost the `x0²` already, and
    /// the form says so before anything cancels — while its coefficient is
    /// still stored and its degree is still 2.
    #[test]
    fn the_inexact_fold_is_flagged_before_anything_drops() {
        let big = (1u64 << 53) as f64;
        let scaled = |c: f64| Expr::Binary(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(sq(0)));
        let e = Expr::Binary(BinOp::Add, Box::new(scaled(big)), Box::new(sq(0)));
        let q = recognize_expr(&e).expect("degree-2");
        assert_eq!(q.quadratic().get(&(0, 0)), Some(&big));
        assert!(q.inexact(), "fl(2⁵³ + 1) = 2⁵³ was called exact");
        // Nothing is missing from the maps *yet*, so the consumers keep
        // their fast paths: the degree is not in question here.
        assert!(!q.lost_terms());
    }

    /// The sticky half of the same rule. The rounding and the drop are in
    /// different subexpressions — `(2⁵³·x0 + x0)` absorbs, `− 2⁵³·x0`
    /// cancels — and the verdict has to survive the trip between them.
    #[test]
    fn rounding_carried_from_a_subexpression_makes_a_later_drop_a_loss() {
        let big = (1u64 << 53) as f64;
        let scaled =
            |c: f64| Expr::Binary(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(Expr::Var(0)));
        let absorbed = Expr::Binary(BinOp::Add, Box::new(scaled(big)), Box::new(Expr::Var(0)));
        let e = Expr::Binary(BinOp::Sub, Box::new(absorbed), Box::new(scaled(big)));
        let q = recognize_expr(&e).expect("degree-1 at worst");
        assert!(q.linear().is_empty());
        assert!(
            q.lost_terms(),
            "the x0 absorbed two additions ago is no less missing now",
        );
    }

    /// Cancellation *inside a product*: `(x0 + x1)·(x0 − x1)` accumulates
    /// `+x0x1` and `−x0x1` onto one key and they cancel exactly. The body
    /// really has no cross term, so the form must not be demoted for
    /// noticing.
    #[test]
    fn an_exact_cancellation_in_a_product_is_not_a_lost_term() {
        let add = |a: Expr, b: Expr| Expr::Binary(BinOp::Add, Box::new(a), Box::new(b));
        let sub = |a: Expr, b: Expr| Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b));
        let e = Expr::Binary(
            BinOp::Mul,
            Box::new(add(Expr::Var(0), Expr::Var(1))),
            Box::new(sub(Expr::Var(0), Expr::Var(1))),
        );
        let q = recognize_expr(&e).expect("degree-2");
        assert_eq!(q.quadratic().get(&(0, 0)), Some(&1.0));
        assert_eq!(q.quadratic().get(&(1, 1)), Some(&-1.0));
        assert_eq!(q.quadratic().get(&(0, 1)), None);
        assert!(!q.lost_terms(), "x0² − x1² was reported incomplete");
    }

    /// An underflowing **multiply** has no exact case to spare: both
    /// factors are nonzero and the product is a monomial the form can no
    /// longer see. `(10⁻²⁰⁰·x0)·(10⁻²⁰⁰·x0)` stays refused, which is gh
    /// #683's second reproduction and gh #687's second acceptance case.
    #[test]
    fn an_underflowing_product_is_still_a_lost_term() {
        let tiny = |i: usize| {
            Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(1e-200)),
                Box::new(Expr::Var(i)),
            )
        };
        let e = Expr::Binary(BinOp::Mul, Box::new(tiny(0)), Box::new(tiny(0)));
        let q = recognize_expr(&e).expect("degree-2 at worst");
        assert!(q.quadratic().is_empty());
        assert!(
            q.lost_terms(),
            "a coefficient that underflowed on the multiply was reported absent",
        );
    }

    /// A constant folded away exactly is degree 0 for real — and `mul`
    /// reads a zero constant as annihilating, so this is the one drop whose
    /// verdict a *product* depends on.
    #[test]
    fn an_exactly_cancelled_constant_is_not_a_lost_term() {
        let e = Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Const(3.0)),
            Box::new(Expr::Const(3.0)),
        );
        let q = recognize_expr(&e).expect("degree 0");
        assert_eq!(q.as_constant(), Some(0.0));
        assert!(!q.lost_terms());

        // The same shape one ulp off: `fl(2⁵³ + 1) − 2⁵³` is `0.0` where the
        // real value is `1`.
        let big = (1u64 << 53) as f64;
        let absorbed = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Const(big)),
            Box::new(Expr::Const(1.0)),
        );
        let e = Expr::Binary(BinOp::Sub, Box::new(absorbed), Box::new(Expr::Const(big)));
        let q = recognize_expr(&e).expect("degree 0");
        assert_eq!(q.as_constant(), Some(0.0));
        assert!(q.lost_terms(), "an absorbed constant was reported absent");
    }

    /// Rounding a coefficient is recorded even when nothing is dropped,
    /// because that is what a later exact cancellation would be hiding:
    /// `x0²/3` cannot be represented, and the form has to remember it.
    #[test]
    fn a_rounded_scale_is_inexact_but_loses_nothing() {
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Const(3.0)));
        let q = recognize_expr(&e).expect("degree-2");
        assert_eq!(q.quadratic().get(&(0, 0)), Some(&(1.0 / 3.0)));
        assert!(q.inexact(), "1 · (1/3) was called exact");
        assert!(
            !q.lost_terms(),
            "a rounded coefficient is not a missing one"
        );

        // Halving is exact, so the same shape by a power of two is not.
        let e = Expr::Binary(BinOp::Div, Box::new(sq(0)), Box::new(Expr::Const(2.0)));
        let q = recognize_expr(&e).expect("degree-2");
        assert!(!q.inexact(), "x0²/2 rounds nothing");
    }

    /// The two-sum and the two-product, against the cases the recognizer
    /// actually meets. Exactness is not a tolerance, so these are `==`.
    #[test]
    fn the_exactness_tests_agree_with_the_arithmetic() {
        let big = (1u64 << 53) as f64;
        assert!(add_is_exact(big, -big, big + -big));
        assert!(add_is_exact(1.0, 2.0, 3.0));
        assert!(!add_is_exact(big, 1.0, big + 1.0), "2⁵³ + 1 loses the 1");
        assert!(!add_is_exact(0.1, 0.2, 0.1 + 0.2));
        assert!(!add_is_exact(f64::INFINITY, 1.0, f64::INFINITY));
        assert!(!add_is_exact(f64::NAN, 1.0, f64::NAN + 1.0));

        assert!(mul_is_exact(3.0, 1.0, 3.0), "the ±1 shortcut");
        assert!(mul_is_exact(3.0, 0.5, 1.5));
        assert!(mul_is_exact(0.1, 4.0, 0.4));
        assert!(!mul_is_exact(3.0, 1.0 / 3.0, 3.0 * (1.0 / 3.0)));
        assert!(!mul_is_exact(1e-200, 1e-200, 1e-200 * 1e-200), "underflow");
        assert!(!mul_is_exact(1e300, 1e300, 1e300 * 1e300), "overflow");
    }

    #[test]
    fn cse_bodies_are_inlined_at_every_reference() {
        // c = x0; c · c is x0².
        let body = std::sync::Arc::new(Expr::Var(0));
        let e = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Cse(body.clone())),
            Box::new(Expr::Cse(body)),
        );
        let h = analyze_quadratic(&e).expect("degree-2");
        assert_eq!(h.get(&(0, 0)), Some(&2.0));
    }

    #[test]
    fn a_wide_nary_sum_does_not_recurse() {
        // Σ xᵢ² over 5000 terms as one `o54` node. Distinct keys, so this
        // says nothing about *order* — see the test below for that.
        const N: usize = 5000;
        let e = Expr::Sum((0..N).map(sq).collect());
        let h = analyze_quadratic(&e).expect("sum of squares is a QP");
        assert_eq!(h.len(), N);
        assert_eq!(h.get(&(N - 1, N - 1)), Some(&2.0));
    }

    /// Summation order is only observable when two items land on the *same*
    /// monomial key with magnitudes far enough apart that the addition
    /// rounds. `1e16·x0² + x0² + x0²` folds to `1e16` front to back (each
    /// `+1` falls below the ulp) and to `1e16 + 2` back to front. The tape
    /// — the reference every non-recognized row is still evaluated with —
    /// sums front to back, so that is the answer required here.
    ///
    /// The 5000-square test above is named for this property but cannot see
    /// it: 5000 distinct keys never re-add anything.
    #[test]
    fn a_repeated_monomial_in_an_nary_sum_accumulates_front_to_back() {
        let scaled =
            |c: f64, i: usize| Expr::Binary(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(sq(i)));
        let e = Expr::Sum(vec![scaled(1.0e16, 0), scaled(1.0, 0), scaled(1.0, 0)]);
        let h = analyze_quadratic(&e).expect("sum of squares is a QP");
        let got = h[&(0, 0)];
        assert_eq!(
            got.to_bits(),
            (2.0 * 1.0e16_f64).to_bits(),
            "expected the front-to-back fold, got {got:e}"
        );
        assert_ne!(got.to_bits(), (2.0 * (1.0e16_f64 + 2.0)).to_bits());
    }

    /// The gate that decides whether a form may be evaluated from its
    /// coefficients. The two directions cost different things, so both are
    /// pinned: admitting a factored form loses digits (and, on
    /// `airport.nl`, 284 iterations), refusing an expanded one loses the
    /// phase's whole point on the family it was built for.
    #[test]
    fn only_already_expanded_forms_are_admitted() {
        // AMPL's `qcqp*` emission: `o54` over `0.5·((c·xᵢ)·xⱼ)`.
        let monomial = |c: f64, i: usize, j: usize| {
            Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Binary(
                        BinOp::Mul,
                        Box::new(Expr::Const(c)),
                        Box::new(Expr::Var(i)),
                    )),
                    Box::new(Expr::Var(j)),
                )),
            )
        };
        let row = Expr::Sum(vec![monomial(2.0, 0, 0), monomial(3.0, 0, 1)]);
        assert!(is_expanded_quadratic(&row));
        // `xᵢ²` is a single monomial, not an expansion.
        assert!(is_expanded_quadratic(&Expr::Sum(vec![sq(0), sq(1)])));
        // A left-deep `Add` chain is still a sum.
        let chain = Expr::Binary(BinOp::Add, Box::new(sq(0)), Box::new(sq(1)));
        assert!(is_expanded_quadratic(&chain));
        // Division by a constant, and a negated term.
        assert!(is_expanded_quadratic(&Expr::Binary(
            BinOp::Div,
            Box::new(sq(0)),
            Box::new(Expr::Const(3.0)),
        )));
        assert!(is_expanded_quadratic(&Expr::Unary(
            UnaryOp::Neg,
            Box::new(monomial(1.0, 0, 1))
        )));

        // `(x₀ − x₁)²` — the `airport.nl` shape. Reading it as
        // `x₀² − 2x₀x₁ + x₁²` cancels, so it stays on the tape.
        let diff = Expr::Binary(BinOp::Sub, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        let factored = Expr::Binary(BinOp::Pow, Box::new(diff), Box::new(Expr::Const(2.0)));
        assert!(analyze_quadratic(&factored).is_some(), "it is quadratic");
        assert!(
            !is_expanded_quadratic(&factored),
            "but not already expanded"
        );

        // `(x₀ + 1)·x₁` — expansion by multiplication rather than by power.
        let product = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::Var(1)),
        );
        assert!(!is_expanded_quadratic(&product));

        // One factored term anywhere in a long expanded sum disqualifies
        // the whole row — the cancellation is in that term, not in the sum.
        let mixed = Expr::Sum(vec![monomial(1.0, 0, 0), factored]);
        assert!(!is_expanded_quadratic(&mixed));

        // A `Cse` referenced once is inlined and judged on its body.
        let once = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Cse(std::sync::Arc::new(Expr::Var(0)))),
            Box::new(Expr::Var(1)),
        );
        assert!(is_expanded_quadratic(&once));
    }

    /// A `Cse` body reached twice is walked once, not `2^depth` times, and
    /// is **admitted** rather than refused.
    ///
    /// Q4 refused it, and refused it for cost rather than algebra — this
    /// walk and `recognize_expr` behind it both inline a body per
    /// reference, so a shared DAG was `Θ(2^depth)`. Q5's memoization is
    /// what makes admitting it affordable, so both halves are asserted
    /// here: the verdict, and the fact that the test returns at all. The
    /// shape is `nl_reader`'s `shared_dag_walks_are_memoized_not_exponential`
    /// scaled up; at depth 60 an exponential walk does not finish inside
    /// the lifetime of this test run, and `2^60` sums do not fit in an
    /// `f64` count either.
    #[test]
    fn a_shared_cse_body_is_walked_once_and_admitted() {
        let mut e = Expr::Var(0);
        for _ in 0..60 {
            let shared = std::sync::Arc::new(e);
            e = Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Cse(std::sync::Arc::clone(&shared))),
                Box::new(Expr::Cse(shared)),
            );
        }
        assert!(is_expanded_quadratic(&e));
        // And the algebra agrees: `x + x` sixty times over is `2^60 · x`.
        let q = recognize_expr(&e).expect("a sum of one monomial is degree 1");
        assert_eq!(q.linear().get(&0).copied(), Some(2.0_f64.powi(60)));
        std::mem::forget(e);
    }

    /// The same shape inside a *monomial*, which is a different question
    /// with a different answer and therefore a separately keyed memo: a
    /// body that is a sum of monomials is legal on the spine and illegal
    /// under a `*`.
    #[test]
    fn a_shared_body_is_judged_per_context_not_once_and_for_all() {
        let sum = std::sync::Arc::new(Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Var(0)),
            Box::new(Expr::Var(1)),
        ));
        // On the spine, twice: fine.
        let spine = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Cse(std::sync::Arc::clone(&sum))),
            Box::new(Expr::Cse(std::sync::Arc::clone(&sum))),
        );
        assert!(is_expanded_quadratic(&spine));
        // The same body on the spine and then under a product: the product
        // is `(x0 + x1) · x1`, a factored form, and the earlier clean visit
        // on the spine must not clear it.
        let mixed = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Cse(std::sync::Arc::clone(&sum))),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Cse(sum)),
                Box::new(Expr::Var(1)),
            )),
        );
        assert!(!is_expanded_quadratic(&mixed));
    }

    /// Deep, and on a default-sized test thread, for the same reason
    /// [`recognize_expr`] is iterative: the gate runs on every row of every
    /// model, including the ones that overflow a recursive walk.
    #[test]
    fn the_expansion_gate_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq(0);
        for i in 1..K {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(sq(i)));
        }
        assert!(is_expanded_quadratic(&e));
        std::mem::forget(e);
    }

    /// The reason this module is iterative.
    ///
    /// A `.nl` writer that emits `o0` (binary `+`) chains for a long sum
    /// hands the recognizer a left-deep `Add` tree one level deep per term.
    /// The recursive predecessor aborted the process — a stack overflow is
    /// not a catchable error — somewhere under 8 000 terms on a 2 MB thread
    /// and under 24 000 on the CLI's 8 MB main thread. The depth below is
    /// far past both, and the test runs on a **default-sized test thread**
    /// deliberately: what a recognizer survives must not depend on who
    /// called it.
    ///
    /// The tree is leaked rather than dropped, because `Expr`'s derived
    /// `Drop` is still recursive and would overflow tearing this down.
    /// That is a real and separate defect (pounce#472 works around it in
    /// the Python bindings with a big-stack worker thread); it is not what
    /// this test is measuring.
    #[test]
    fn deep_add_chain_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq(0);
        for i in 1..K {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(sq(i)));
        }
        let h = analyze_quadratic(&e).expect("a sum of squares is a QP at any depth");
        assert_eq!(h.len(), K, "every xᵢ² contributes one diagonal entry");
        assert_eq!(h.get(&(K - 1, K - 1)), Some(&2.0));
        std::mem::forget(e);
    }

    /// Same shape, right-deep — the value stack, not the work stack, is
    /// what grows here. Both live on the heap.
    #[test]
    fn deep_right_leaning_chain_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq(K - 1);
        for i in (0..K - 1).rev() {
            e = Expr::Binary(BinOp::Add, Box::new(sq(i)), Box::new(e));
        }
        let h = analyze_quadratic(&e).expect("a sum of squares is a QP at any depth");
        assert_eq!(h.len(), K);
        std::mem::forget(e);
    }

    /// A non-quadratic node deep inside a deep tree must return `None`
    /// rather than unwind — the bail-out path drops the work and value
    /// stacks, and neither is recursive.
    #[test]
    fn deep_chain_with_a_transcendental_bails_without_overflowing() {
        const K: usize = 250_000;
        let mut e = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        for i in 1..K {
            e = Expr::Binary(BinOp::Add, Box::new(e), Box::new(sq(i)));
        }
        assert!(analyze_quadratic(&e).is_none());
        std::mem::forget(e);
    }

    // -----------------------------------------------------------------
    // Factored forms (gh #673)
    // -----------------------------------------------------------------

    /// `(base)^2`.
    fn sq_of(base: Expr) -> Expr {
        Expr::Binary(BinOp::Pow, Box::new(base), Box::new(Expr::Const(2.0)))
    }

    fn var_minus(i: usize, c: f64) -> Expr {
        Expr::Binary(BinOp::Sub, Box::new(Expr::Var(i)), Box::new(Expr::Const(c)))
    }

    /// The motivating case. `(x − 500000)²` is exactly what
    /// `feasible_x0_extreme_row.nl` writes, and expanding it is the 2.4e-5
    /// disagreement gh #673 is named after.
    #[test]
    fn a_shifted_square_is_kept_factored() {
        let e = sq_of(var_minus(0, 500_000.0));
        assert!(
            !is_expanded_quadratic(&e),
            "the expanded gate must refuse it"
        );
        let f = recognize_factored_quadratic(&e).expect("a square of an affine form");
        assert_eq!(f.squares.len(), 1);
        assert_eq!(f.squares[0].weight, 1.0);
        assert_eq!(f.squares[0].coefs, vec![(0, 1.0)]);
        assert_eq!(f.squares[0].constant, -500_000.0);
        assert!(f.linear.is_empty());
        assert_eq!(f.constant, 0.0);
    }

    /// The accuracy claim, stated as a number rather than as prose: at
    /// `x = 500000 + 1e-4` the expansion loses five digits and the factored
    /// read-out loses nothing.
    #[test]
    fn the_factored_read_out_does_not_cancel_where_the_expansion_does() {
        let e = sq_of(var_minus(0, 500_000.0));
        let x = 500_000.0 + 1e-4;
        // What the tape computes: square the residual as written.
        let r = x - 500_000.0;
        let taped = r * r;

        let (h, lin, c) = analyze_quadratic_full(&e).expect("degree 2");
        let expanded = 0.5 * h[&(0, 0)] * x * x + lin[0].1 * x + c;
        assert!(
            (expanded - taped).abs() / taped > 1e-6,
            "the expansion is supposed to cancel here, got {expanded} for {taped}"
        );

        let f = recognize_factored_quadratic(&e).expect("a square");
        let t = &f.squares[0];
        let l = t.constant + t.coefs[0].1 * x;
        // Bit for bit, not within a tolerance.
        assert_eq!(t.weight * l * l, taped);
    }

    /// `airport.nl`'s shape: a row that is a sum of squared coordinate
    /// differences, with the writer's grouping kept term by term.
    #[test]
    fn a_sum_of_squared_differences_is_admitted() {
        let diff = |i: usize, j: usize| {
            Expr::Binary(BinOp::Sub, Box::new(Expr::Var(i)), Box::new(Expr::Var(j)))
        };
        let e = Expr::Binary(
            BinOp::Add,
            Box::new(sq_of(diff(0, 1))),
            Box::new(sq_of(diff(2, 3))),
        );
        let f = recognize_factored_quadratic(&e).expect("two squares");
        assert_eq!(f.squares.len(), 2);
        let mut sup: Vec<Vec<(usize, f64)>> = f.squares.iter().map(|t| t.coefs.clone()).collect();
        sup.sort_by_key(|c| c[0].0);
        assert_eq!(
            sup,
            vec![vec![(0, 1.0), (1, -1.0)], vec![(2, 1.0), (3, -1.0)]]
        );
        assert!(
            f.squares
                .iter()
                .all(|t| t.weight == 1.0 && t.constant == 0.0)
        );
    }

    /// The sign of the sum spine reaches the weight, and a constant factor
    /// folds into it.
    #[test]
    fn spine_signs_and_constant_factors_fold_into_the_weight() {
        // 3·(x₀ − 1)² − 0.5·(x₁ + 2)²
        let a = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(3.0)),
            Box::new(sq_of(var_minus(0, 1.0))),
        );
        let b = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(sq_of(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Const(2.0)),
            ))),
        );
        let f = recognize_factored_quadratic(&Expr::Binary(BinOp::Sub, Box::new(a), Box::new(b)))
            .expect("two squares");
        let mut got: Vec<(f64, f64)> = f.squares.iter().map(|t| (t.weight, t.constant)).collect();
        got.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert_eq!(got, vec![(-0.5, 2.0), (3.0, -1.0)]);
    }

    /// A degree-≤1 leftover in the same tree is not a reason to refuse the
    /// row; it folds into `a`/`c` and is evaluated there.
    #[test]
    fn degree_one_leftovers_fold_into_the_linear_part() {
        // (x₀ − 1)² + 3·x₁ + 7
        let e = Expr::Sum(vec![
            sq_of(var_minus(0, 1.0)),
            Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(3.0)),
                Box::new(Expr::Var(1)),
            ),
            Expr::Const(7.0),
        ]);
        let f = recognize_factored_quadratic(&e).expect("a square plus leftovers");
        assert_eq!(f.squares.len(), 1);
        assert_eq!(f.linear, vec![(1, 3.0)]);
        assert_eq!(f.constant, 7.0);
    }

    /// A bare `c·xᵢ²` monomial is the same term wearing a different opcode,
    /// and is stored as `c·(xᵢ)²` so that a row mixing the two shapes is
    /// still served.
    #[test]
    fn a_diagonal_monomial_is_stored_as_a_square() {
        let e = Expr::Binary(
            BinOp::Add,
            Box::new(sq_of(var_minus(0, 1.0))),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Const(4.0)),
                Box::new(Expr::Binary(
                    BinOp::Mul,
                    Box::new(Expr::Var(1)),
                    Box::new(Expr::Var(1)),
                )),
            )),
        );
        let f = recognize_factored_quadratic(&e).expect("square + diagonal monomial");
        assert_eq!(f.squares.len(), 2);
        let diag = f
            .squares
            .iter()
            .find(|t| t.coefs == vec![(1, 1.0)])
            .unwrap();
        // The *polynomial* coefficient, so `4x₁²` and not `8x₁²` —
        // `push_factored_form` is what doubles it into a Hessian entry.
        assert_eq!((diag.weight, diag.constant), (4.0, 0.0));
    }

    /// A cross monomial has no square to be stored as, and storing it
    /// expanded next to the squares is the mixed representation this
    /// deliberately does not have. The row keeps its tape.
    #[test]
    fn a_cross_monomial_alongside_a_square_is_refused() {
        let e = Expr::Binary(
            BinOp::Add,
            Box::new(sq_of(var_minus(0, 1.0))),
            Box::new(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Var(1)),
                Box::new(Expr::Var(2)),
            )),
        );
        assert!(recognize_factored_quadratic(&e).is_none());
    }

    /// An already-expanded body is `is_expanded_quadratic`'s to serve, and
    /// this must not offer a second, slower answer for it.
    #[test]
    fn an_expanded_body_is_not_claimed_here() {
        let e = Expr::Binary(BinOp::Add, Box::new(sq(0)), Box::new(sq(1)));
        assert!(is_expanded_quadratic(&e));
        // `x²` *is* a square, so this one is genuinely recognized — the
        // caller's ordering is what keeps it on the cheaper path.
        assert!(recognize_factored_quadratic(&e).is_some());
        // A body with no square at all is refused outright.
        let cross = Expr::Binary(BinOp::Mul, Box::new(Expr::Var(0)), Box::new(Expr::Var(1)));
        assert!(recognize_factored_quadratic(&cross).is_none());
        assert!(recognize_factored_quadratic(&Expr::Var(0)).is_none());
    }

    /// The product of two *different* affine forms is degree 2 and is not a
    /// square. Admitting it would mean storing `l₁·l₂`, which this
    /// representation cannot express.
    #[test]
    fn a_product_of_two_different_affine_forms_is_refused() {
        let e = Expr::Binary(
            BinOp::Mul,
            Box::new(var_minus(0, 1.0)),
            Box::new(var_minus(1, 2.0)),
        );
        assert!(recognize_factored_quadratic(&e).is_none());
    }

    /// A square of a *quadratic* is degree 4, and a transcendental is not a
    /// polynomial at all.
    #[test]
    fn quartics_and_transcendentals_are_refused() {
        assert!(recognize_factored_quadratic(&sq_of(sq(0))).is_none());
        let s = Expr::Unary(UnaryOp::Sin, Box::new(Expr::Var(0)));
        assert!(recognize_factored_quadratic(&sq_of(s)).is_none());
    }

    /// A base that is not itself a flat sum of monomials is refused: the
    /// coefficients read out of `2·(x + 1)` are not the additions the
    /// writer wrote, which is the same rule `is_expanded_quadratic` states.
    #[test]
    fn a_base_the_expanded_gate_refuses_is_refused_here_too() {
        let inner = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(2.0)),
            Box::new(Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Var(0)),
                Box::new(Expr::Const(1.0)),
            )),
        );
        assert!(!is_expanded_quadratic(&inner));
        assert!(recognize_factored_quadratic(&sq_of(inner)).is_none());
    }

    /// A leftover that lost a term inexactly refuses the body, for the
    /// reason gh #685 gives: the form would evaluate a row its own tape
    /// does not agree with.
    #[test]
    fn a_lost_leftover_refuses_the_body() {
        let big = 9_007_199_254_740_992.0f64; // 2⁵³
        let scaled = |c: f64, i: usize| {
            Expr::Binary(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(Expr::Var(i)))
        };
        // (x₀ − 1)² + 2⁵³·x₁ + x₁ − 2⁵³·x₁ — the `x₁` is folded away.
        let e = Expr::Sum(vec![
            sq_of(var_minus(0, 1.0)),
            scaled(big, 1),
            Expr::Var(1),
            scaled(-big, 1),
        ]);
        assert!(recognize_factored_quadratic(&e).is_none());
    }

    /// A `Cse` on the sum spine ends the walk: skipping a second reference
    /// would drop its terms and visiting every reference is exponential.
    #[test]
    fn a_shared_body_on_the_spine_is_refused() {
        let shared = std::sync::Arc::new(sq_of(var_minus(0, 1.0)));
        let e = Expr::Binary(
            BinOp::Add,
            Box::new(Expr::Cse(shared.clone())),
            Box::new(Expr::Cse(shared)),
        );
        assert!(recognize_factored_quadratic(&e).is_none());
    }

    /// A weight that annihilates or overflows is not evaluated from stored
    /// coefficients: `0·(x−1)²` has dropped a term the tape still walks,
    /// and `(x−1)²/0` is not a number.
    #[test]
    fn degenerate_weights_are_refused() {
        let zero = Expr::Binary(
            BinOp::Mul,
            Box::new(Expr::Const(0.0)),
            Box::new(sq_of(var_minus(0, 1.0))),
        );
        assert!(recognize_factored_quadratic(&zero).is_none());
        let div0 = Expr::Binary(
            BinOp::Div,
            Box::new(sq_of(var_minus(0, 1.0))),
            Box::new(Expr::Const(0.0)),
        );
        assert!(recognize_factored_quadratic(&div0).is_none());
    }

    /// Deep, on a default-sized test thread, for the reason every walk in
    /// this module is iterative: a least-squares model is a long `o0` chain
    /// of squared residuals, which is exactly the shape that aborts a
    /// recursive walk — and now exactly the shape this recognizer is for.
    ///
    /// Leaked rather than dropped: `Expr`'s derived `Drop` is still
    /// recursive. See `deep_add_chain_does_not_overflow_the_stack`.
    #[test]
    fn a_deep_chain_of_squares_does_not_overflow_the_stack() {
        const K: usize = 250_000;
        let mut e = sq_of(var_minus(0, 1.0));
        for i in 1..K {
            e = Expr::Binary(
                BinOp::Add,
                Box::new(e),
                Box::new(sq_of(var_minus(i, i as f64))),
            );
        }
        let f = recognize_factored_quadratic(&e).expect("K squares");
        assert_eq!(f.squares.len(), K);
        std::mem::forget(e);
    }
}
