//! Constant-structure evaluation for degree-≤2 objectives and rows.
//!
//! A quadratic row's value, gradient and Hessian are all determined by one
//! constant matrix. POUNCE nevertheless re-derives them on every call from a
//! reverse-mode AD tape: on Mittelmann's `qcqp500-3c` that is 2.32 M tapes
//! walked twice per `eval_h`, 332 times, to recover ten matrices that never
//! change. [`QuadraticStructure`] is where the recognizer's answer is kept
//! instead of thrown away — see
//! `dev-notes/quadratic-structure-exploitation.md` (gh #588), phase Q4.
//!
//! ## What lives here, and what does not
//!
//! This module is **pure sparse linear algebra**. It never sees an `Expr`, a
//! tape or a `feral` factorization: recognition is
//! `pounce_nl::nl_quadratic`'s job and it hands the result over as plain
//! `(i, j, ∂²/∂xᵢ∂xⱼ)` triplets. That split is what lets the type live in
//! `pounce-nlp`, which every other crate already depends on, so `pounce-py`,
//! `pounce-wasm` and `pounce-rs` inherit it with no manifest change — and it
//! is why the structure stops at the TNLP values array and knows nothing
//! about the KKT system those values end up in.
//!
//! ## Storage
//!
//! Every array is **flat and shared across forms**, indexed by per-form
//! offsets. A `Vec` per form would reintroduce, one level up, exactly the
//! allocation-per-object cost Q3 removed from the recognizer: `qssp180` has
//! 65 341 quadratic rows of three nonzeros each, and a struct-per-row layout
//! would spend ten allocations on each of them to describe 24 bytes of
//! matrix.
//!
//! Each form is stored as
//!
//! * the **full symmetric** `H` in CSR over the form's support — both
//!   triangles, because the value and the gradient are matvecs and a matvec
//!   against half a symmetric matrix costs a scatter it does not need;
//! * the linear coefficients `a` folded into the nonlinear tree by the `.nl`
//!   writer (the `−6x₀` of `(x₀−3)²`), which are *not* the row's `.nl` linear
//!   section — the caller still adds that;
//! * the constant `c` (the `+9`), for the same reason
//!   [`analyze_quadratic_full`](https://docs.rs/pounce-nl) returns it: it does
//!   not move the minimizer but it is part of the reported value;
//! * one scatter slot per lower-triangle entry, so accumulating `∇²L` is a
//!   `values[slot] += w · h` loop with no hashing and no search.
//!
//! The convention on `H` is the Hessian's, `∂²/∂xᵢ∂xⱼ` — so a `c·xᵢ²` term
//! arrives as `2c` on the diagonal, and the form evaluates as
//! `½xᵀHx + aᵀx + c`. Handing it *polynomial* coefficients instead is a
//! silent factor of two on the diagonal, which is why
//! [`QuadraticStructure::push_form`] takes the Hessian triplets that
//! `analyze_quadratic_full` already applies that factor in, rather than a
//! `Quad2`.

use std::collections::{BTreeMap, BTreeSet};

use pounce_common::exact::{add_is_exact, is_live, mul_is_exact};

/// Index into [`QuadraticStructure`]'s form arrays. `u32` because the count
/// is bounded by `m + 1` and the row map is one of these per constraint.
type FormId = u32;

/// No form: the value in [`QuadraticStructure::form_of_row`] for a row that
/// is not (recognized as) quadratic and keeps its tape.
const NO_FORM: FormId = u32::MAX;

/// One `w·(bᵀx + d)²` term on its way into
/// [`QuadraticStructure::push_factored_form`].
///
/// Borrowed rather than owned, and plain tuples rather than a recognizer
/// type, for the reason the module docs give: this crate never sees an
/// `Expr`, and `pounce_nl` hands its answer over as numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SquareTerm<'a> {
    /// `w` — the constant the square is multiplied by.
    pub weight: f64,
    /// `b`, **ascending** by variable index and free of zeros. May be
    /// empty, which makes the term the constant `w·d²`.
    ///
    /// Ascending is load-bearing, not tidiness:
    /// [`QuadraticStructure::push_factored_form`] walks `coefs[a..]` to
    /// build the Hessian's **upper** triangle, so an out-of-order pair
    /// would write `(i, j)` with `i > j` into a map every reader takes as
    /// upper-triangular. Checked by a `debug_assert` there (gh #711).
    pub coefs: &'a [(usize, f64)],
    /// `d`.
    pub constant: f64,
}

/// The recognized degree-≤2 parts of one model: at most one per constraint
/// row, plus at most one for the objective.
///
/// Built once in `NlTnlp::try_new` and then read-only. Cloning it is what
/// `NlTnlp::variant` does — a variation changes bounds and starting points,
/// never structure — so the layout is deliberately a handful of flat `Vec`s
/// rather than a graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuadraticStructure {
    // ---- per form ----
    /// `sup[form_sup[f] .. form_sup[f + 1]]` — the variables with a Hessian
    /// row in form `f`, ascending.
    form_sup: Vec<u32>,
    /// `lin[form_lin[f] .. form_lin[f + 1]]` — indices into `lin_idx`/`lin_val`.
    form_lin: Vec<u32>,
    /// `grad[form_grad[f] .. form_grad[f + 1]]` — the form's gradient
    /// support (Hessian support ∪ linear support), ascending. Precomputed
    /// because `eval_jac_g` needs it once per call per row and it is the
    /// same set every time.
    form_grad: Vec<u32>,
    /// `h_slot[form_h[f] .. form_h[f + 1]]` — one scatter target per
    /// lower-triangle entry of form `f`, in the order
    /// [`Self::lower_triangle`] walks them.
    form_h: Vec<u32>,
    /// The degree-0 term of each form.
    constant: Vec<f64>,

    // ---- concatenated payloads ----
    /// Concatenated Hessian supports.
    sup: Vec<u32>,
    /// CSR row pointers, one per entry of `sup` plus a global terminator:
    /// support entry `k` owns `col[sup_ptr[k] .. sup_ptr[k + 1]]`.
    sup_ptr: Vec<u32>,
    /// Column indices of the **full symmetric** `H`, ascending within a row.
    col: Vec<u32>,
    /// Values of the full symmetric `H`, parallel to `col`.
    val: Vec<f64>,
    /// Concatenated linear supports and coefficients.
    lin_idx: Vec<u32>,
    lin_val: Vec<f64>,
    /// Concatenated gradient supports.
    grad: Vec<u32>,
    /// Concatenated scatter targets; `u32::MAX` until [`Self::bind_slots`].
    h_slot: Vec<u32>,

    // ---- the factored half ----
    /// `sq_*[form_sq[f] .. form_sq[f + 1]]` — the squared affine terms of
    /// form `f`. Empty for an expanded form, which is every form this
    /// module had before gh #673.
    form_sq: Vec<u32>,
    /// `wₖ` and `dₖ` of each stored term.
    sq_w: Vec<f64>,
    sq_d: Vec<f64>,
    /// CSR row pointers over the terms, one per term plus a global
    /// terminator: term `k` owns `sq_idx[sq_ptr[k] .. sq_ptr[k + 1]]`.
    sq_ptr: Vec<u32>,
    /// Concatenated `bₖ`, ascending within a term.
    sq_idx: Vec<u32>,
    sq_val: Vec<f64>,

    // ---- the map back to the model ----
    /// `form_of_row[i]` is the form evaluating row `i`, or [`NO_FORM`] when
    /// row `i` keeps its tape.
    form_of_row: Vec<FormId>,
    /// The objective's form, or [`NO_FORM`].
    obj_form: FormId,
}

/// Neumaier compensated summation.
///
/// Kahan's compensation with Neumaier's fix for the case where the incoming
/// term is larger in magnitude than the running sum — which is the common
/// case here, since the outer accumulator starts at zero.
///
/// This exists for gh#702. See [`QuadraticStructure::value`].
#[derive(Clone, Copy)]
struct Neumaier {
    sum: f64,
    comp: f64,
}

impl Neumaier {
    #[inline]
    fn new() -> Self {
        Self {
            sum: 0.0,
            comp: 0.0,
        }
    }

    #[inline]
    fn add(&mut self, t: f64) {
        let y = self.sum + t;
        self.comp += if self.sum.abs() >= t.abs() {
            (self.sum - y) + t
        } else {
            (t - y) + self.sum
        };
        self.sum = y;
    }

    #[inline]
    fn sum(self) -> f64 {
        self.sum + self.comp
    }
}

/// `Σ 2wₖbₖbₖᵀ` as an upper triangle, or `None` if building it **lost a
/// term** (gh #685, by way of gh #673's review of gh #711).
///
/// # Why this refuses rather than just summing
///
/// gh #685's gate — `is_expanded_quadratic` in `admitted_factored_form` —
/// keeps a body out of the fast path when it dropped a coefficient in the
/// fold. That test is on the *shape* the coefficients came from, and it
/// covers the lossy bodies that are also flat sums of monomials. It does
/// not cover a genuinely **factored** body that loses a term here, in this
/// accumulation, which nothing else looks at:
///
/// ```text
/// (2²⁷x₀)² + (x₀ + x₁)² − (x₀·2²⁷)²
/// ```
///
/// Three honest squares, no monomial spine, so the shape gate has no
/// opinion. But `(0, 0)` accumulates `2·2⁵⁴ + 2 − 2·2⁵⁴`: the `+ 2` is half
/// an ulp at `2⁵⁵` and ties back to even, the third term cancels what is
/// left, and the entry reaches exactly `0.0` and is not stored. The tape
/// declares `(0, 0)` and evaluates the body to `0.0` at `x = (3, 1)`; the
/// factored read-out answers `16.0` and offers a Hessian one entry short.
/// Two routes over the same bytes, disagreeing about a constraint body
/// against its own bound — which is the whole of gh #685, whichever answer
/// is nearer the algebra.
///
/// The mechanism needs about 2⁵³ of dynamic range across the squared terms.
/// On the **diagonal** that takes a negative weight, so difference-of-
/// squares and DC formulations; **off** the diagonal it does not, and
/// `(10⁹x₀ + 10⁹x₁)² + (x₀ + x₁)² + (10⁹x₀ − 10⁹x₁)²` loses `H(0, 1)` with
/// every weight positive. Coefficients of `10⁹` are ordinary in a badly
/// scaled model.
///
/// # The rule
///
/// [`pounce_common::exact`]'s predicates, and gh #687's rule, unchanged: a
/// term is lost when an entry is **dropped** and some fold on the way to
/// dropping it **rounded**. An exact cancellation is not a loss — `x₀² −
/// x₀²` has a genuinely zero Hessian entry and the tape agrees — so this
/// admits it, exactly as `Quad2::merge` does on the expanded arm.
///
/// The products are checked too, not only the sums: `2·w` overflowing to an
/// infinity, or `bᵢ·bⱼ` underflowing to zero out of two nonzero factors, is
/// the same defect arriving by a different route.
fn factored_hessian(squares: &[SquareTerm<'_>]) -> Option<BTreeMap<(usize, usize), f64>> {
    // `coefs` ascending ⇒ `(i, j)` with `i ≤ j` falls out of the double
    // loop. The `bool` is the entry's **carried** inexactness: whether any
    // fold into it has rounded. It has to be sticky, because the fold that
    // rounds and the fold that drops are usually not the same one — in the
    // witness above `(0, 0)` absorbs the `+ 2` on the second term and is
    // cancelled to zero by the third, and *that* add is exact.
    let mut hess: BTreeMap<(usize, usize), (f64, bool)> = BTreeMap::new();
    for t in squares {
        debug_assert!(
            t.coefs.windows(2).all(|w| w[0].0 < w[1].0),
            "SquareTerm::coefs must be ascending and duplicate-free: the \
             upper-triangle walk below depends on it",
        );
        let s = 2.0 * t.weight;
        if !mul_is_exact(2.0, t.weight, s) {
            // `2·w` can only round by overflowing to an infinity, which has
            // swallowed every term it touched.
            return None;
        }
        for (a, &(i, bi)) in t.coefs.iter().enumerate() {
            for &(j, bj) in &t.coefs[a..] {
                let p = s * bi;
                let c = p * bj;
                // A product is a *loss* when it rounds a nonzero to nothing
                // — `10⁻²⁰⁰ · 10⁻²⁰⁰` — or overflows. Merely rounding is the
                // same latitude the squared read-out itself takes.
                let product_lost =
                    (is_live(s) && is_live(bi) && is_live(bj) && !is_live(c)) || !c.is_finite();
                if product_lost {
                    return None;
                }

                let slot = hess.entry((i, j)).or_insert((0.0, false));
                let (was, carried) = *slot;
                let v = was + c;
                let inexact = carried || !add_is_exact(was, c, v);
                if was != 0.0 && !is_live(v) && inexact {
                    // The entry reached zero, and something on the way to
                    // zero rounded: the tape will declare an entry this map
                    // no longer has. gh #687's rule, unchanged — an *exact*
                    // cancellation is not a loss and falls through.
                    return None;
                }
                *slot = (v, inexact);
            }
        }
    }
    Some(
        hess.into_iter()
            .filter(|(_, (c, _))| is_live(*c))
            .map(|(k, (c, _))| (k, c))
            .collect(),
    )
}

impl QuadraticStructure {
    /// An empty structure over `m` constraint rows: nothing recognized, so
    /// every caller behaves exactly as it did before this module existed.
    pub fn new(m: usize) -> Self {
        QuadraticStructure {
            form_sup: vec![0],
            form_lin: vec![0],
            form_grad: vec![0],
            form_h: vec![0],
            form_sq: vec![0],
            sup_ptr: vec![0],
            sq_ptr: vec![0],
            form_of_row: vec![NO_FORM; m],
            obj_form: NO_FORM,
            ..Self::default()
        }
    }

    /// Add a form and return its id.
    ///
    /// `hess` is the **upper-triangular (i ≤ j) Hessian** of the form —
    /// `∂²/∂xᵢ∂xⱼ`, so the factor of two on the diagonal is already applied
    /// — which is exactly `analyze_quadratic_full`'s first return value.
    /// `lin` is its degree-1 part and `constant` its degree-0 part.
    ///
    /// Entries that are exactly zero are dropped: they cost a scatter slot
    /// and a multiply-add to contribute nothing, and keeping them would make
    /// the Hessian pattern depend on how a coefficient happened to cancel.
    pub fn push_form(
        &mut self,
        hess: &BTreeMap<(usize, usize), f64>,
        lin: &[(usize, f64)],
        constant: f64,
    ) -> FormId {
        self.push_matrix(hess);
        self.push_linear(lin);

        // Gradient support = Hessian support ∪ linear support, ascending.
        // Both sides are already ascending, so this is a merge.
        let f = self.constant.len();
        let hs = &self.sup[self.form_sup[f] as usize..];
        let ls = &self.lin_idx[self.form_lin[f] as usize..];
        let (mut a, mut b) = (0usize, 0usize);
        while a < hs.len() || b < ls.len() {
            let take = match (hs.get(a), ls.get(b)) {
                (Some(&x), Some(&y)) => {
                    if x < y {
                        a += 1;
                        x
                    } else if y < x {
                        b += 1;
                        y
                    } else {
                        a += 1;
                        b += 1;
                        x
                    }
                }
                (Some(&x), None) => {
                    a += 1;
                    x
                }
                (None, Some(&y)) => {
                    b += 1;
                    y
                }
                (None, None) => unreachable!("loop guard"),
            };
            self.grad.push(take);
        }
        self.form_grad.push(self.grad.len() as u32);
        // No squared terms: this is the expanded form, evaluated from `H`.
        self.form_sq.push(self.sq_w.len() as u32);
        self.finish_form(constant)
    }

    /// Add a **factored** form — `Σ wₖ(bₖᵀx + dₖ)² + aᵀx + c` — and return
    /// its id (gh #673).
    ///
    /// The difference from [`Self::push_form`] is entirely in how the value
    /// and the gradient are computed. `(x − 500000)²` expanded to
    /// `x² − 10⁶x + 2.5·10¹¹` and read back cancels five digits, which is
    /// why `pounce_nl`'s recognizer refuses to hand a factored body over as
    /// triplets at all; handed over as *squares* it is the tape's own
    /// arithmetic, one multiplication for one multiplication.
    ///
    /// The **Hessian** is not factored and does not need to be: `Σ 2wₖbₖbₖᵀ`
    /// is constant, and it is assembled here once from exactly the products
    /// the tape would accumulate on every call. So
    /// [`Self::accumulate_hessian`], [`Self::add_hessian_vector`] and
    /// [`Self::lower_triangle`] are shared with the expanded path and know
    /// nothing about any of this.
    ///
    /// `lin` and `constant` are the degree-≤1 leftovers of the same tree,
    /// not the row's `.nl` linear section — the same convention
    /// [`Self::push_form`] takes them on.
    ///
    /// ## The gradient support is the terms' support, not the Hessian's
    ///
    /// `∂/∂xᵢ Σ wₖ(bₖᵀx + dₖ)²` is nonzero wherever any `bₖ` is, while an
    /// entry of `Σ 2wₖbₖbₖᵀ` can cancel to exactly zero between two terms
    /// and be dropped. Taking the gradient pattern from the assembled
    /// Hessian would then hand `eval_jac_g` a column list short of a column
    /// the row actually depends on. It is taken from the `bₖ` instead.
    pub fn push_factored_form(
        &mut self,
        squares: &[SquareTerm<'_>],
        lin: &[(usize, f64)],
        constant: f64,
    ) -> Option<FormId> {
        // Refuses before touching `self`, so a rejected body leaves no
        // half-built form behind. See `factored_hessian`.
        let hess = factored_hessian(squares)?;
        self.push_matrix(&hess);
        self.push_linear(lin);

        // Gradient support = ⋃ bₖ support ∪ linear support. See the docs:
        // this is deliberately not read off `hess`.
        let f = self.constant.len();
        let mut sup: BTreeSet<u32> = squares
            .iter()
            .flat_map(|t| t.coefs.iter().map(|&(i, _)| i as u32))
            .collect();
        sup.extend(self.lin_idx[self.form_lin[f] as usize..].iter().copied());
        self.grad.extend(sup);
        self.form_grad.push(self.grad.len() as u32);

        for t in squares {
            self.sq_w.push(t.weight);
            self.sq_d.push(t.constant);
            for &(i, c) in t.coefs {
                self.sq_idx.push(i as u32);
                self.sq_val.push(c);
            }
            self.sq_ptr.push(self.sq_idx.len() as u32);
        }
        self.form_sq.push(self.sq_w.len() as u32);
        Some(self.finish_form(constant))
    }

    /// Scatter a form's upper-triangular Hessian into the full symmetric
    /// CSR the value, gradient and Hessian paths all read.
    fn push_matrix(&mut self, hess: &BTreeMap<(usize, usize), f64>) {
        // A `BTreeMap` keyed by row keeps the support ascending without a
        // sort, and the inner `BTreeMap` keeps each row's columns ascending
        // — which is what makes `lower_triangle` come out in the (row, col)
        // order the TNLP's `lower_pairs` is sorted in.
        let mut rows: BTreeMap<u32, BTreeMap<u32, f64>> = BTreeMap::new();
        for (&(i, j), &v) in hess {
            debug_assert!(
                i <= j,
                "a form's Hessian is the upper triangle, got ({i}, {j})"
            );
            if v == 0.0 {
                continue;
            }
            rows.entry(i as u32).or_default().insert(j as u32, v);
            if i != j {
                rows.entry(j as u32).or_default().insert(i as u32, v);
            }
        }

        for (&r, cols) in &rows {
            self.sup.push(r);
            for (&c, &v) in cols {
                self.col.push(c);
                self.val.push(v);
            }
            self.sup_ptr.push(self.col.len() as u32);
        }
        self.form_sup.push(self.sup.len() as u32);
    }

    /// Append a form's degree-1 coefficients, dropping stored zeros.
    fn push_linear(&mut self, lin: &[(usize, f64)]) {
        for &(i, c) in lin {
            if c == 0.0 {
                continue;
            }
            self.lin_idx.push(i as u32);
            self.lin_val.push(c);
        }
        self.form_lin.push(self.lin_idx.len() as u32);
    }

    /// Close a form: one unbound scatter slot per lower-triangle entry, and
    /// the degree-0 term.
    fn finish_form(&mut self, constant: f64) -> FormId {
        let f = self.constant.len();
        let n_lower = self.lower_triangle(f as FormId).count();
        self.h_slot.resize(self.h_slot.len() + n_lower, NO_FORM);
        self.form_h.push(self.h_slot.len() as u32);
        self.constant.push(constant);
        f as FormId
    }

    /// Attach form `f` to constraint row `i`.
    pub fn assign_row(&mut self, i: usize, f: FormId) {
        self.form_of_row[i] = f;
    }

    /// Attach form `f` to the objective.
    pub fn assign_objective(&mut self, f: FormId) {
        self.obj_form = f;
    }

    /// Is there anything here at all? A model with no recognized part gets
    /// the tape path unchanged, down to the coloring.
    pub fn is_empty(&self) -> bool {
        self.constant.is_empty()
    }

    /// How many forms are stored (objective included).
    pub fn len(&self) -> usize {
        self.constant.len()
    }

    /// The form evaluating constraint row `i`, if it has one.
    pub fn row_form(&self, i: usize) -> Option<FormId> {
        match self.form_of_row.get(i).copied() {
            Some(f) if f != NO_FORM => Some(f),
            _ => None,
        }
    }

    /// The objective's form, if it has one.
    pub fn objective_form(&self) -> Option<FormId> {
        (self.obj_form != NO_FORM).then_some(self.obj_form)
    }

    /// The form's gradient support (Hessian support ∪ linear support),
    /// ascending. This is what a Jacobian row's column list must include.
    pub fn gradient_support(&self, f: FormId) -> &[u32] {
        let f = f as usize;
        &self.grad[self.form_grad[f] as usize..self.form_grad[f + 1] as usize]
    }

    /// The form's lower-triangle Hessian entries as `(row, col, value)` with
    /// `row >= col`, ascending by `(row, col)`.
    ///
    /// The order is load-bearing twice over: it is the order
    /// [`Self::bind_slots`] consumes scatter targets in, and it is the order
    /// [`Self::accumulate_hessian`] replays them in.
    pub fn lower_triangle(&self, f: FormId) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        let f = f as usize;
        let (lo, hi) = (self.form_sup[f] as usize, self.form_sup[f + 1] as usize);
        (lo..hi).flat_map(move |k| {
            let r = self.sup[k];
            let (a, b) = (self.sup_ptr[k] as usize, self.sup_ptr[k + 1] as usize);
            (a..b).filter_map(move |e| {
                let c = self.col[e];
                (c <= r).then(|| (r, c, self.val[e]))
            })
        })
    }

    /// Point each lower-triangle entry at its index in the TNLP's Hessian
    /// `values` array.
    ///
    /// `lookup` is called once per entry, in [`Self::lower_triangle`] order,
    /// and must return the index of `(row, col)` in the model's assembled
    /// lower-triangle pattern.
    pub fn bind_slots(&mut self, mut lookup: impl FnMut(u32, u32) -> usize) {
        let mut k = 0usize;
        for f in 0..self.constant.len() as FormId {
            let entries: Vec<(u32, u32)> = self.lower_triangle(f).map(|(r, c, _)| (r, c)).collect();
            for (r, c) in entries {
                self.h_slot[k] = lookup(r, c) as u32;
                k += 1;
            }
        }
        debug_assert_eq!(k, self.h_slot.len(), "every entry gets a slot");
    }

    /// `½xᵀHx + aᵀx + c` — or `Σ wₖ(bₖᵀx + dₖ)² + aᵀx + c` for a form
    /// [`Self::push_factored_form`] built.
    ///
    /// On the expanded arm the `½` is applied once at the end rather than
    /// per row. That is cheaper and it is exactly equivalent: scaling by
    /// `0.5` only decrements a binary exponent, so it neither rounds nor
    /// reassociates. The factored arm has no `½` to apply — it squares the
    /// writer's own residual, which is the whole point of it (gh #673).
    ///
    /// # Why the outer sums are compensated (gh#702)
    ///
    /// The tape sums AMPL's flat list of `½·c·xᵢ·xⱼ` terms front-to-back in file
    /// order. This walks the coefficient matrix row-major and sums one merged
    /// row at a time. Both are correct; they are different *associations* of
    /// the same real number, so they round differently. On the constraints of
    /// a dense QCQP the outer accumulator takes thousands of same-sign terms
    /// and the two answers separate in the last few ulps.
    ///
    /// That is normally invisible, but the interior-point trajectory is
    /// sensitive to it: on `qcqp1500-1c` the naive row-major sum took 131
    /// iterations where the tape took 103. Compensating the outer accumulator
    /// makes the result far less dependent on the association and takes the
    /// same model to 100 iterations — better than either.
    ///
    /// Only the *outer* sums are compensated. The inner per-row dot product
    /// was measured to contribute nothing (compensating it changes no
    /// trajectory in the corpus, on the Mittelmann QCQP family, or on the
    /// `eigen*` fixtures), and it is the O(nnz) loop — compensating it costs
    /// real time. The outer loop is O(rows), so this is close to free.
    ///
    /// **Both arms, for one reason.** `Σ wₖlₖ²` is the same outer
    /// accumulator wearing a different shape, and a least-squares row is
    /// gh#702's case in its purest form: every term same-signed, hundreds or
    /// thousands of them, a running total that outgrows each one. Leaving it
    /// naive would make the compensation a property of which read-out a body
    /// happened to take. The inner accumulator there is [`Self::affine`] —
    /// the per-residual dot product, `O(nnz)` — and it stays naive for
    /// gh#702's reason.
    ///
    /// The cost is that the factored arm no longer reproduces the tape's
    /// summation bit for bit, only its *terms*. That is gh#702's trade and
    /// not a new one: being less dependent on the association beats matching
    /// one particular association, and it was measured to be worth 31
    /// iterations on `qcqp1500-1c`.
    pub fn value(&self, f: FormId, x: &[f64]) -> f64 {
        let fi = f as usize;
        let quad = match self.squares_of(fi) {
            Some(terms) => {
                let mut acc = Neumaier::new();
                for k in terms {
                    let l = self.affine(k, x);
                    // `w · (l · l)`, and the parentheses are load-bearing:
                    // the tape multiplies the writer's constant into the
                    // *square*, and `(w · l) · l` is a different rounding.
                    // On a body whose terms then cancel — `2⁵³x² + x² −
                    // 2⁵³x²` — that one ulp is the entire answer.
                    acc.add(self.sq_w[k] * (l * l));
                }
                acc.sum()
            }
            None => {
                let (lo, hi) = (self.form_sup[fi] as usize, self.form_sup[fi + 1] as usize);
                let mut quad = Neumaier::new();
                for k in lo..hi {
                    let r = self.sup[k] as usize;
                    let (a, b) = (self.sup_ptr[k] as usize, self.sup_ptr[k + 1] as usize);
                    let mut t = 0.0;
                    for e in a..b {
                        t += self.val[e] * x[self.col[e] as usize];
                    }
                    quad.add(x[r] * t);
                }
                0.5 * quad.sum()
            }
        };
        let mut lin = Neumaier::new();
        for e in self.form_lin[fi] as usize..self.form_lin[fi + 1] as usize {
            lin.add(self.lin_val[e] * x[self.lin_idx[e] as usize]);
        }
        quad + lin.sum() + self.constant[fi]
    }

    /// The stored squared terms of form `fi`, or `None` when it is an
    /// expanded form evaluated from `H`.
    fn squares_of(&self, fi: usize) -> Option<std::ops::Range<usize>> {
        let (lo, hi) = (self.form_sq[fi] as usize, self.form_sq[fi + 1] as usize);
        (lo != hi).then_some(lo..hi)
    }

    /// `dₖ + bₖᵀx` for stored term `k`.
    fn affine(&self, k: usize, x: &[f64]) -> f64 {
        let mut l = self.sq_d[k];
        for e in self.sq_ptr[k] as usize..self.sq_ptr[k + 1] as usize {
            l += self.sq_val[e] * x[self.sq_idx[e] as usize];
        }
        l
    }

    /// `out += w · (Hx + a)`, touching only the form's gradient support —
    /// or `out += w · (Σ 2wₖ(bₖᵀx + dₖ)bₖ + a)` for a factored form, which
    /// differentiates the square rather than its expansion.
    pub fn add_gradient(&self, f: FormId, x: &[f64], w: f64, out: &mut [f64]) {
        let fi = f as usize;
        match self.squares_of(fi) {
            Some(terms) => {
                for k in terms {
                    let d = 2.0 * self.sq_w[k] * self.affine(k, x);
                    for e in self.sq_ptr[k] as usize..self.sq_ptr[k + 1] as usize {
                        out[self.sq_idx[e] as usize] += w * d * self.sq_val[e];
                    }
                }
            }
            None => {
                let (lo, hi) = (self.form_sup[fi] as usize, self.form_sup[fi + 1] as usize);
                for k in lo..hi {
                    let r = self.sup[k] as usize;
                    let (a, b) = (self.sup_ptr[k] as usize, self.sup_ptr[k + 1] as usize);
                    let mut t = 0.0;
                    for e in a..b {
                        t += self.val[e] * x[self.col[e] as usize];
                    }
                    out[r] += w * t;
                }
            }
        }
        for e in self.form_lin[fi] as usize..self.form_lin[fi + 1] as usize {
            out[self.lin_idx[e] as usize] += w * self.lin_val[e];
        }
    }

    /// `values[slot] += w · H[row, col]` over the form's lower triangle.
    ///
    /// This is the whole Hessian contribution of a quadratic row: no forward
    /// sweep, no directional product, no decode — the multipliers are the
    /// only thing that changed since the model was read.
    pub fn accumulate_hessian(&self, f: FormId, w: f64, values: &mut [f64]) {
        let fi = f as usize;
        let base = self.form_h[fi] as usize;
        for (k, (_, _, v)) in self.lower_triangle(f).enumerate() {
            let slot = self.h_slot[base + k] as usize;
            values[slot] += w * v;
        }
    }

    /// `out += w · (H · v)` — the Hessian-vector product form, for the
    /// matrix-free (Newton-Krylov) path.
    pub fn add_hessian_vector(&self, f: FormId, v: &[f64], w: f64, out: &mut [f64]) {
        let fi = f as usize;
        let (lo, hi) = (self.form_sup[fi] as usize, self.form_sup[fi + 1] as usize);
        for k in lo..hi {
            let r = self.sup[k] as usize;
            let (a, b) = (self.sup_ptr[k] as usize, self.sup_ptr[k + 1] as usize);
            let mut t = 0.0;
            for e in a..b {
                t += self.val[e] * v[self.col[e] as usize];
            }
            out[r] += w * t;
        }
    }

    /// Total stored Hessian entries over all forms, both triangles. Reported
    /// by `POUNCE_DBG_TAPE_STATS`; also the thing the memory claim in the
    /// design note is about.
    ///
    /// A factored form's squared terms are **not** counted: this is the
    /// matrix, and they are the residuals it was assembled from.
    pub fn stored_entries(&self) -> usize {
        self.val.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `q(x) = 3x₀² + 5x₀x₁ − 2x₀ + 7`, whose Hessian is
    /// `[[6, 5], [5, 0]]`.
    fn sample() -> (QuadraticStructure, FormId) {
        let mut h = BTreeMap::new();
        h.insert((0, 0), 6.0);
        h.insert((0, 1), 5.0);
        let mut qs = QuadraticStructure::new(1);
        let f = qs.push_form(&h, &[(0, -2.0)], 7.0);
        (qs, f)
    }

    #[test]
    fn value_matches_the_polynomial() {
        let (qs, f) = sample();
        let x = [2.0, -3.0];
        // 3·4 + 5·2·(−3) − 2·2 + 7 = 12 − 30 − 4 + 7
        assert_eq!(qs.value(f, &x), -15.0);
    }

    #[test]
    fn gradient_is_hx_plus_a() {
        let (qs, f) = sample();
        let x = [2.0, -3.0];
        let mut g = [0.0; 2];
        qs.add_gradient(f, &x, 1.0, &mut g);
        // ∂/∂x₀ = 6·2 + 5·(−3) − 2 = −5 ; ∂/∂x₁ = 5·2 = 10
        assert_eq!(g, [-5.0, 10.0]);
        // Weighted, and accumulating rather than overwriting.
        qs.add_gradient(f, &x, 2.0, &mut g);
        assert_eq!(g, [-15.0, 30.0]);
    }

    /// The factor-of-two trap: `push_form` takes *Hessian* entries, so a
    /// `3x₀²` term must arrive as `6`, and the value must come back as `3x₀²`
    /// rather than `6x₀²`.
    #[test]
    fn the_diagonal_is_a_hessian_entry_not_a_polynomial_coefficient() {
        let mut h = BTreeMap::new();
        h.insert((0, 0), 6.0);
        let mut qs = QuadraticStructure::new(0);
        let f = qs.push_form(&h, &[], 0.0);
        assert_eq!(qs.value(f, &[1.0]), 3.0);
        let mut g = [0.0];
        qs.add_gradient(f, &[1.0], 1.0, &mut g);
        assert_eq!(g, [6.0]);
    }

    #[test]
    fn lower_triangle_is_ascending_and_omits_the_upper_half() {
        let (qs, f) = sample();
        let got: Vec<(u32, u32, f64)> = qs.lower_triangle(f).collect();
        assert_eq!(got, vec![(0, 0, 6.0), (1, 0, 5.0)]);
    }

    #[test]
    fn hessian_scatters_through_the_bound_slots() {
        let (mut qs, f) = sample();
        // Pretend the model's assembled pattern is [(1,0), (0,0)] — a
        // deliberately un-sorted order, so a bug that assumed identity
        // mapping shows up.
        let pattern = [(1u32, 0u32), (0, 0)];
        qs.bind_slots(|r, c| {
            pattern
                .iter()
                .position(|&p| p == (r, c))
                .expect("entry in pattern")
        });
        let mut values = [0.0; 2];
        qs.accumulate_hessian(f, 2.0, &mut values);
        assert_eq!(values, [10.0, 12.0]);
    }

    #[test]
    fn hessian_vector_product_agrees_with_the_dense_matrix() {
        let (qs, f) = sample();
        let v = [1.0, 2.0];
        let mut out = [0.0; 2];
        qs.add_hessian_vector(f, &v, 1.0, &mut out);
        // [[6,5],[5,0]] · [1,2] = [16, 5]
        assert_eq!(out, [16.0, 5.0]);
    }

    #[test]
    fn gradient_support_merges_the_two_sides() {
        let mut h = BTreeMap::new();
        h.insert((1, 3), 1.0);
        let mut qs = QuadraticStructure::new(0);
        // Linear part touches 0 and 3; the Hessian touches 1 and 3.
        let f = qs.push_form(&h, &[(0, 1.0), (3, 1.0)], 0.0);
        assert_eq!(qs.gradient_support(f), &[0, 1, 3]);
    }

    #[test]
    fn zero_coefficients_are_not_stored() {
        let mut h = BTreeMap::new();
        h.insert((0, 0), 0.0);
        h.insert((0, 1), 4.0);
        let mut qs = QuadraticStructure::new(0);
        let f = qs.push_form(&h, &[(2, 0.0)], 0.0);
        assert_eq!(qs.lower_triangle(f).count(), 1);
        assert_eq!(qs.gradient_support(f), &[0, 1]);
    }

    /// gh#702: the outer accumulation over Hessian rows is compensated,
    /// so `value` does not depend on where a large row sits in the order.
    ///
    /// The construction is the worst case in miniature. One row carries
    /// `1e18`, a hundred carry `2`, and the last carries `-1e18`. Summed
    /// front-to-back every `2` falls off the bottom of the running total —
    /// `1e18 + 2` *is* `1e18` in binary64 — and the cancellation at the end
    /// leaves zero. The true value is 100.
    ///
    /// A dense QCQP constraint row is the same shape with less contrast:
    /// thousands of same-sign terms, a running total that outgrows them, and
    /// a result that depends on the order they arrive in. That dependence is
    /// what took `qcqp1500-1c` from 103 iterations on the tape to 131 on the
    /// matrix path.
    #[test]
    fn the_outer_row_sum_does_not_lose_small_terms() {
        let mut h = BTreeMap::new();
        h.insert((0, 0), 1e18);
        for r in 1..=100usize {
            h.insert((r, r), 2.0);
        }
        h.insert((101, 101), -1e18);

        let mut qs = QuadraticStructure::new(0);
        let f = qs.push_form(&h, &[], 0.0);

        let x = [1.0; 102];
        // ½·(1e18 + 100·2 − 1e18). Naive summation answers 0.0 here.
        assert_eq!(qs.value(f, &x), 100.0);
    }

    /// The same property for the linear accumulator, which is summed the
    /// same way and compensated the same way (gh#702).
    #[test]
    fn the_linear_sum_does_not_lose_small_terms() {
        let mut lin = vec![(0usize, 1e18)];
        lin.extend((1..=100usize).map(|i| (i, 2.0)));
        lin.push((101, -1e18));

        let mut qs = QuadraticStructure::new(0);
        let f = qs.push_form(&BTreeMap::new(), &lin, 0.0);

        let x = [1.0; 102];
        assert_eq!(qs.value(f, &x), 200.0);
    }

    #[test]
    fn an_empty_structure_reports_itself_as_one() {
        let qs = QuadraticStructure::new(4);
        assert!(qs.is_empty());
        assert_eq!(qs.row_form(2), None);
        assert_eq!(qs.objective_form(), None);
    }

    // -----------------------------------------------------------------
    // Factored forms (gh #673)
    // -----------------------------------------------------------------

    /// `(x₀ − 500000)²` — the form whose *expansion* loses five digits, and
    /// the reason this arm exists.
    #[test]
    fn a_factored_form_squares_the_residual_instead_of_expanding_it() {
        let coefs = [(0usize, 1.0)];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(
                &[SquareTerm {
                    weight: 1.0,
                    coefs: &coefs,
                    constant: -500_000.0,
                }],
                &[],
                0.0,
            )
            .expect("admitted");
        let x = [500_000.0 + 1e-4];
        let r = x[0] - 500_000.0;
        // Bit for bit what the tape computes, not "within a tolerance".
        assert_eq!(qs.value(f, &x), r * r);

        let mut g = [0.0];
        qs.add_gradient(f, &x, 1.0, &mut g);
        assert_eq!(g, [2.0 * r]);

        // The Hessian is constant and is the expansion's — nothing is
        // factored about a second derivative.
        assert_eq!(qs.lower_triangle(f).collect::<Vec<_>>(), vec![(0, 0, 2.0)]);
    }

    /// A form with several terms, a linear leftover and a weight, checked
    /// against the polynomial it is: `2(x₀ − x₁ + 1)² − (x₁ + 3)² + 5x₀ + 7`.
    #[test]
    fn a_multi_term_factored_form_agrees_with_its_polynomial() {
        let a = [(0usize, 1.0), (1usize, -1.0)];
        let b = [(1usize, 1.0)];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(
                &[
                    SquareTerm {
                        weight: 2.0,
                        coefs: &a,
                        constant: 1.0,
                    },
                    SquareTerm {
                        weight: -1.0,
                        coefs: &b,
                        constant: 3.0,
                    },
                ],
                &[(0, 5.0)],
                7.0,
            )
            .expect("admitted");
        let x = [1.5, -0.25];
        let l1 = x[0] - x[1] + 1.0;
        let l2 = x[1] + 3.0;
        assert_eq!(qs.value(f, &x), 2.0 * l1 * l1 - l2 * l2 + 5.0 * x[0] + 7.0);

        let mut g = [0.0; 2];
        qs.add_gradient(f, &x, 1.0, &mut g);
        assert_eq!(g[0], 4.0 * l1 + 5.0);
        assert_eq!(g[1], -4.0 * l1 - 2.0 * l2);

        // ∇² = 2·2·bbᵀ − 2·eeᵀ = [[4, −4], [−4, 4 − 2]]
        assert_eq!(
            qs.lower_triangle(f).collect::<Vec<_>>(),
            vec![(0, 0, 4.0), (1, 0, -4.0), (1, 1, 2.0)]
        );
    }

    /// The gradient support has to come from the terms, not from the
    /// assembled Hessian: `(x₀ + x₁)² − (x₀ − x₁)²` is `4x₀x₁`, whose
    /// Hessian has **no diagonal at all** — while the gradient depends on
    /// both variables, and `eval_jac_g` needs a column for each.
    #[test]
    fn the_gradient_support_survives_a_cancelling_hessian() {
        let p = [(0usize, 1.0), (1usize, 1.0)];
        let m = [(0usize, 1.0), (1usize, -1.0)];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(
                &[
                    SquareTerm {
                        weight: 1.0,
                        coefs: &p,
                        constant: 0.0,
                    },
                    SquareTerm {
                        weight: -1.0,
                        coefs: &m,
                        constant: 0.0,
                    },
                ],
                &[],
                0.0,
            )
            .expect("admitted");
        assert_eq!(qs.gradient_support(f), &[0, 1]);
        // The diagonal cancelled exactly and is not stored.
        assert_eq!(qs.lower_triangle(f).collect::<Vec<_>>(), vec![(1, 0, 4.0)]);
        let x = [2.0, 3.0];
        assert_eq!(qs.value(f, &x), 4.0 * x[0] * x[1]);
        let mut g = [0.0; 2];
        qs.add_gradient(f, &x, 1.0, &mut g);
        assert_eq!(g, [4.0 * x[1], 4.0 * x[0]]);
    }

    /// A square of a constant contributes a value and nothing else.
    #[test]
    fn a_constant_square_is_value_only() {
        let mut qs = QuadraticStructure::new(0);
        let coefs = [(0usize, 1.0)];
        let f = qs
            .push_factored_form(
                &[
                    SquareTerm {
                        weight: 3.0,
                        coefs: &[],
                        constant: 2.0,
                    },
                    SquareTerm {
                        weight: 1.0,
                        coefs: &coefs,
                        constant: 0.0,
                    },
                ],
                &[],
                0.0,
            )
            .expect("admitted");
        assert_eq!(qs.gradient_support(f), &[0]);
        assert_eq!(qs.value(f, &[4.0]), 3.0 * 4.0 + 16.0);
        let mut g = [0.0];
        qs.add_gradient(f, &[4.0], 1.0, &mut g);
        assert_eq!(g, [8.0]);
    }

    /// Expanded and factored forms coexist in one structure, each read
    /// back the way it was stored — the per-form offsets have to stay in
    /// lockstep for that, and an off-by-one here would silently give a
    /// factored form its neighbour's terms.
    #[test]
    fn the_two_kinds_of_form_coexist() {
        let (mut qs, expanded) = sample();
        let coefs = [(0usize, 1.0)];
        let factored = qs
            .push_factored_form(
                &[SquareTerm {
                    weight: 1.0,
                    coefs: &coefs,
                    constant: -1.0,
                }],
                &[],
                0.0,
            )
            .expect("admitted");
        let x = [2.0, -3.0];
        assert_eq!(qs.value(expanded, &x), -15.0);
        assert_eq!(qs.value(factored, &x), 1.0);
        let plain = qs.push_form(&BTreeMap::from([((1, 1), 2.0)]), &[], 0.0);
        assert_eq!(qs.value(plain, &x), 9.0);
        assert_eq!(qs.value(factored, &x), 1.0);

        // …and the scatter slots stay attached to the right entries.
        let pattern = [(0u32, 0u32), (1, 0), (1, 1)];
        qs.bind_slots(|r, c| {
            pattern
                .iter()
                .position(|&p| p == (r, c))
                .expect("in pattern")
        });
        let mut values = [0.0; 3];
        qs.accumulate_hessian(factored, 1.0, &mut values);
        assert_eq!(values, [2.0, 0.0, 0.0]);
        qs.accumulate_hessian(plain, 1.0, &mut values);
        assert_eq!(values, [2.0, 0.0, 2.0]);
    }

    /// The Hessian-vector product reads the assembled matrix, so it is
    /// shared with the expanded path and must agree with the dense form.
    #[test]
    fn a_factored_forms_hessian_vector_product_agrees_with_its_matrix() {
        let a = [(0usize, 1.0), (1usize, -1.0)];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(
                &[SquareTerm {
                    weight: 1.0,
                    coefs: &a,
                    constant: 4.0,
                }],
                &[],
                0.0,
            )
            .expect("admitted");
        // ∇² = 2·[[1, −1], [−1, 1]]
        let mut out = [0.0; 2];
        qs.add_hessian_vector(f, &[1.0, 2.0], 1.0, &mut out);
        assert_eq!(out, [-2.0, 2.0]);
    }

    /// The weight multiplies the **square**, not the residual: `w·(l·l)`,
    /// never `(w·l)·l`. The tape associates the first way — a `.nl` body
    /// `o2 n7 o5 (x) n2` squares first and scales after — and the two
    /// orders are not the same double.
    ///
    /// `7·(1.1)²` is the smallest witness this crate has: `7·(1.1·1.1)` is
    /// `8.47` and `(7·1.1)·1.1` is `8.470000000000002`. That is one ulp on
    /// its own, and on a body whose terms then cancel it is not one ulp of
    /// the answer, it is the answer.
    #[test]
    fn the_weight_multiplies_the_square_not_the_residual() {
        let coefs = [(0usize, 1.0)];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(
                &[SquareTerm {
                    weight: 7.0,
                    coefs: &coefs,
                    constant: 0.0,
                }],
                &[],
                0.0,
            )
            .expect("admitted");
        let x = [1.1];
        assert_eq!(qs.value(f, &x), 7.0 * (x[0] * x[0]));
        // The association this must not have — stated as a number so the
        // test cannot pass by both sides being computed the same wrong way.
        assert_eq!(qs.value(f, &x), 8.47);
        assert_ne!((7.0 * x[0]) * x[0], 8.47);
    }

    /// gh#702's property, on gh#673's arm — and this is the arm where it is
    /// least avoidable. A least-squares row is hundreds of same-signed
    /// `wₖlₖ²`, which is exactly the shape whose naive sum loses its tail.
    ///
    /// Same construction as `the_outer_row_sum_does_not_lose_small_terms`,
    /// written as squares: one term of `1e18`, a hundred of `2`, one of
    /// `-1e18`. Front-to-back every `2` falls off the bottom of the running
    /// total and the cancellation leaves zero; the true value is 200.
    ///
    /// Without this the compensation would be a property of which read-out a
    /// body happened to take, and nothing would say so.
    #[test]
    fn the_factored_outer_sum_does_not_lose_small_terms_either() {
        let coefs: Vec<[(usize, f64); 1]> = (0..102).map(|i| [(i, 1.0)]).collect();
        let mut terms: Vec<SquareTerm<'_>> = Vec::new();
        for (i, c) in coefs.iter().enumerate() {
            let weight = match i {
                0 => 1e18,
                101 => -1e18,
                _ => 2.0,
            };
            terms.push(SquareTerm {
                weight,
                coefs: c,
                constant: 0.0,
            });
        }
        let mut qs = QuadraticStructure::new(0);
        let f = qs.push_factored_form(&terms, &[], 0.0).expect("admitted");
        let x = [1.0; 102];
        // 1e18 + 100·2 − 1e18. Naive summation answers 0.0 here.
        assert_eq!(qs.value(f, &x), 200.0);
    }

    /// gh #711: a factored form whose Hessian loses an entry is refused, so
    /// the body falls back to its tape instead of being evaluated from a
    /// matrix the tape does not agree with.
    ///
    /// `(2²⁷x₀)² + (x₀ + x₁)² − (x₀·2²⁷)²`. `(0, 0)` accumulates
    /// `2·2⁵⁴ + 2 − 2·2⁵⁴`: the `+ 2` is under half an ulp at `2⁵⁵` and
    /// rounds away, then the third term cancels what is left exactly. The
    /// entry reaches `0.0` and is not stored, while the tape declares it.
    ///
    /// The two folds are not the same one — the rounding is on the second
    /// term and the drop on the third, whose add is *exact* — which is why
    /// the inexactness has to be carried per entry rather than tested at
    /// the drop.
    #[test]
    fn a_factored_form_that_loses_a_hessian_entry_is_refused() {
        let big = (1u64 << 27) as f64;
        let wide = [(0usize, big)];
        let unit = [(0usize, 1.0), (1usize, 1.0)];
        let terms = [
            SquareTerm {
                weight: 1.0,
                coefs: &wide,
                constant: 0.0,
            },
            SquareTerm {
                weight: 1.0,
                coefs: &unit,
                constant: 0.0,
            },
            SquareTerm {
                weight: -1.0,
                coefs: &wide,
                constant: 0.0,
            },
        ];

        let mut qs = QuadraticStructure::new(0);
        assert!(
            qs.push_factored_form(&terms, &[], 0.0).is_none(),
            "a form that dropped a Hessian entry was admitted",
        );
        // Refused *before* mutating: the structure is still empty, so a
        // caller that falls back to a tape leaves nothing half-built.
        assert!(qs.is_empty(), "a refused form left state behind");
    }

    /// The same refusal off the diagonal, with **every weight positive** —
    /// which is what makes it more than a curiosity. The diagonal mechanism
    /// needs a negative weight to cancel; here the sign comes from inside a
    /// square, so an ordinary badly scaled least-squares row reaches it.
    ///
    /// `(10⁹x₀ + 10⁹x₁)² + (x₀ + x₁)² + (10⁹x₀ − 10⁹x₁)²` loses `H(0, 1)`.
    #[test]
    fn the_refusal_does_not_need_a_negative_weight() {
        let a = [(0usize, 1e9), (1usize, 1e9)];
        let b = [(0usize, 1.0), (1usize, 1.0)];
        let c = [(0usize, 1e9), (1usize, -1e9)];
        let terms = [
            SquareTerm {
                weight: 1.0,
                coefs: &a,
                constant: 0.0,
            },
            SquareTerm {
                weight: 1.0,
                coefs: &b,
                constant: 0.0,
            },
            SquareTerm {
                weight: 1.0,
                coefs: &c,
                constant: 0.0,
            },
        ];
        let mut qs = QuadraticStructure::new(0);
        assert!(qs.push_factored_form(&terms, &[], 0.0).is_none());
    }

    /// gh #687's rule holds on this arm: a term that cancels **exactly** did
    /// not go missing, and refusing it would give up the fast path for
    /// arithmetic that never rounded.
    ///
    /// `(x₀ + x₁)² − (x₀ − x₁)²` cancels both diagonal entries exactly, and
    /// its true Hessian really is `[[0, 4], [4, 0]]`.
    #[test]
    fn an_exactly_cancelling_factored_form_is_still_admitted() {
        let plus = [(0usize, 1.0), (1usize, 1.0)];
        let minus = [(0usize, 1.0), (1usize, -1.0)];
        let terms = [
            SquareTerm {
                weight: 1.0,
                coefs: &plus,
                constant: 0.0,
            },
            SquareTerm {
                weight: -1.0,
                coefs: &minus,
                constant: 0.0,
            },
        ];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(&terms, &[], 0.0)
            .expect("an exact cancellation is not a loss");
        let x = [2.0, 3.0];
        // 4·x₀·x₁.
        assert_eq!(qs.value(f, &x), 24.0);
        // The cancelled diagonal is genuinely absent, and the gradient
        // support still carries both variables.
        assert_eq!(qs.gradient_support(f), &[0, 1]);
    }

    /// A weight big enough that `2·w` overflows cannot be stored, because
    /// the infinity has swallowed every term it touched. Suspected by
    /// gh #711's review and pinned here rather than left as a maybe.
    #[test]
    fn a_weight_whose_double_overflows_is_refused() {
        let coefs = [(0usize, 1.0)];
        let terms = [SquareTerm {
            weight: 1e308,
            coefs: &coefs,
            constant: 0.0,
        }];
        let mut qs = QuadraticStructure::new(0);
        assert!(qs.push_factored_form(&terms, &[], 0.0).is_none());
    }

    /// The other half of `product_lost`: a coefficient product that
    /// **underflows** out of two nonzero factors.
    ///
    /// `(10⁻²⁰⁰·x₀)²` has `2w·b² = 2·10⁻⁴⁰⁰`, which is not representable and
    /// flushes to zero — so the stored Hessian would claim the row has no
    /// second derivative in `x₀` while its tape squares a nonzero residual.
    /// Exactly `underflowing_body` from gh #685's file, reaching the same
    /// defect through gh #673's door.
    ///
    /// gh #711's review found this branch pinned by nothing: disabling both
    /// early product refusals together failed only the overflow assertion
    /// above. Same shape as the hole the review opened with, one predicate
    /// over.
    #[test]
    fn a_coefficient_product_that_underflows_is_refused() {
        let coefs = [(0usize, 1e-200)];
        let terms = [SquareTerm {
            weight: 1e-200,
            coefs: &coefs,
            constant: 0.0,
        }];
        let mut qs = QuadraticStructure::new(0);
        assert!(
            qs.push_factored_form(&terms, &[], 0.0).is_none(),
            "a form whose Hessian entry underflowed to zero was admitted",
        );
        assert!(qs.is_empty(), "a refused form left state behind");
    }

    /// The underflow refusal is about a product that *lost* something, not
    /// about small numbers. A weight and coefficients that stay
    /// representable are admitted, however small the entry is.
    #[test]
    fn a_small_but_representable_hessian_entry_is_still_admitted() {
        let coefs = [(0usize, 1e-100)];
        let terms = [SquareTerm {
            weight: 1e-100,
            coefs: &coefs,
            constant: 0.0,
        }];
        let mut qs = QuadraticStructure::new(0);
        let f = qs
            .push_factored_form(&terms, &[], 0.0)
            .expect("2e-300 is representable and must not be refused");
        let x = [3.0];
        // `w·(b·x)²` — the coefficient is inside the square, so this is
        // `1e-100 · (3e-100)²`, and the Hessian entry `2·1e-100·(1e-100)²`
        // is `2e-300`: small, representable, nothing lost.
        let l = 1e-100 * 3.0;
        assert_eq!(qs.value(f, &x), 1e-100 * (l * l));
    }
}
