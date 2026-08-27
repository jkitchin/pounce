//! Scoped index newtypes for the two variable spaces this crate mixes.
//!
//! # Why these exist, and why they are not everywhere
//!
//! gh#764 item 3 proposed `VarX`/`FullX`/`KktRow`/`UserG` across the
//! crate, then measured the cost: ~342 touch points across 12 files,
//! `pub trait SensBacksolver` re-exported from the crate root (so
//! changing `solve_released` is a breaking API change), and 31 sites in
//! `pounce-py` where indices cross into Python as `i64`. That sweep is
//! not worth its price, and this module is not it.
//!
//! What survived the measurement is a much sharper rule:
//!
//! > **Table lookup fails loudly; direct array indexing fails
//! > silently.**
//!
//! A released row resolved through `rows.iter().find(|b| b.row == r)?`
//! turns a space-swap into `SensComputationFailed`, immediately and
//! every time. A direct index into a `Vec` does not: it returns a
//! *neighbouring variable's* answer, which is in range, plausible, and
//! wrong. That is gh#450, and gh#672 finding 1 shipped it again as
//! `report.var_status.get(var_row)` — a var-x row indexing a full-x
//! array.
//!
//! So the newtype earns its cost at exactly one shape: **a conversion
//! whose result feeds a direct array index, in a scope where the other
//! space is also live.** Measured over `pounce-sensitivity`, that is
//! two sites, both in `solver.rs`:
//!
//! * the kappa computation's `var_sigma` read, whose `unwrap_or(0.0)`
//!   turns a miss into a zero that silently drops the row from the
//!   engaged set;
//! * `weakly_active_bounds`, whose loop body indexes `ctx.x_curr` /
//!   `ctx.lo` / `ctx.hi` by var-x **and** `report.var_status` by
//!   full-x, with a bound row `b.row` in scope as a third space.
//!
//! The other nine conversion sites do not earn it, and are left alone
//! deliberately:
//!
//! * `activity.rs` has four scatter loops (`var_full[full_of(i)] = e`)
//!   where the index is converted at the point of use and only one
//!   space is live — there is nothing to mix up.
//! * `algorithm_backsolver.rs:871` resolves through a checked
//!   `get_mut(..).ok_or_else(..)` and then sweeps for an unfilled NaN
//!   sentinel; it fails loudly twice over.
//! * `solver.rs:1686` and the `full_x_to_var_x` accessors *produce*
//!   indices for a caller rather than indexing with them.
//!
//! # What this does and does not buy
//!
//! On the typed path the swap is not merely discouraged, it is
//! **unrepresentable**: [`FullX`] has no public constructor, so the
//! only way to obtain one is to put a [`VarX`] through [`VarToFull`],
//! and `FullXSlice::at` accepts nothing else.
//!
//! It does **not** follow that the swap is gone. `ActivityReport`'s
//! `Vec` fields are public and must stay so, and nothing stops a
//! caller writing `report.var_status[row.get()]` — which compiles, and
//! is caught by leg 3 rather than by the compiler. A newtype closes
//! the path that goes through it; it cannot close one whose public API
//! is a bare `Vec`.
//!
//! The guard that closes the known site is still
//! `sens_invariance_legs.rs` leg 3, whose fixture puts a fixed variable
//! *ahead* of the kink so the two spaces actually diverge. This module
//! is what makes the *next* such site a compile error instead of a
//! fixture someone has to think to write.
//!
//! # Mutation evidence
//!
//! What the type actually closes, measured by reintroducing the swap
//! at each converted site rather than by argument:
//!
//! | mutation | outcome |
//! |---|---|
//! | mint a `FullX` from the var-x row at the kappa read | **compile error**, `E0624: associated function 'new' is private` |
//! | the same at `weakly_active_bounds` | **compile error**, same |
//! | bypass the type: `report.var_status.get(row.get())` | **compiles.** Caught at runtime instead, by `sens_invariance_legs.rs` leg 3 -- 3 legs go red (`leg_fixed_the_weak_set_...`, `leg_fixed_the_directional_derivative_...`, `the_legs_compose_at_the_fixed_and_scaled_corner`) |
//!
//! The third row is the honest limit and the reason leg 3 is not
//! retired: the public `Vec` fields are still reachable, so the type
//! covers the typed path and the leg fences the untyped one. Neither
//! alone covers this site.
//!
//! The first draft of this module failed its own first mutation --
//! `FullX::new` was `pub`, so `FullX::new(var_row.get())` typechecked
//! and the swap was one short line away. That is why the constructor
//! is private now, and why the table above exists at all: a newtype
//! whose guarantee is not mutation-checked is a comment with a
//! `struct` around it.

use pounce_common::types::Index;

/// A row in the algorithm's **free-variable** space: the primal block
/// the solver actually optimizes, with `make_parameter`-removed
/// variables absent.
///
/// This is the space of `BoundMultiplier::var_row`, `WeakBound::var_row`,
/// and every array whose length is the primal block width.
///
/// It coincides with [`FullX`] until the first removed variable and
/// diverges after it — which is why a corpus of fixtures with no fixed
/// variables cannot see a swap, and why leg 3 puts one ahead of the
/// kink on purpose.
///
/// ```
/// use pounce_sensitivity::index::VarX;
/// assert_eq!(VarX::new(3).get(), 3);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct VarX(usize);

/// An index into the **model's** variable space, fixed variables
/// included: the space of every `ActivityReport` per-variable array,
/// and the length [`crate::solver::Solver::n_full_x`] reports.
///
/// A `FullX` is obtainable only by converting a [`VarX`] through
/// [`VarToFull`] -- there is no public constructor, by design:
///
/// ```
/// use pounce_sensitivity::index::{VarToFull, VarX};
/// // full-x 0 is `make_parameter`-removed, so var-x k is full-x k+1
/// let map = VarToFull::build(3, |v| v.get() + 1);
/// assert_eq!(map.full_of(VarX::new(0)).unwrap().get(), 1);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FullX(usize);

impl VarX {
    /// Name a var-x row.
    #[must_use]
    pub const fn new(row: usize) -> Self {
        Self(row)
    }

    /// The raw row. Calling this is the explicit act of leaving the
    /// typed path — at a direct array index, check which space the
    /// array is in before you do.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }

    /// The raw row as the FFI-facing [`Index`].
    #[must_use]
    pub const fn as_index(self) -> Index {
        self.0 as Index
    }
}

impl FullX {
    /// Name a full-x index.
    ///
    /// **Deliberately private to this module.** A `FullX` is obtainable
    /// only by converting a [`VarX`] through [`VarToFull`], so the
    /// assertion "this number is in full-x" is made once, where the
    /// map is built, instead of at every read. Making this `pub` is
    /// what made the first draft weaker than its own documentation:
    /// `FullX::new(var_row.get())` typechecked, so the swap the module
    /// exists to prevent was one short line away.
    #[must_use]
    const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// The raw index. See [`VarX::get`] on what calling it means.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// The var-x → full-x map, materialized once so a loop does not
/// re-borrow the NLP per row.
///
/// Built with [`VarToFull::build`] at the point of use, from the
/// NLP's own `var_x_to_full_x`. Lookups are bounds-checked and return
/// [`None`] rather than a neighbouring row's index — the whole point
/// of the exercise.
#[derive(Clone, Debug)]
pub struct VarToFull {
    full_of: Vec<FullX>,
}

impl VarToFull {
    /// Build the map by converting every var-x row, `0..n_var_x`.
    ///
    /// `to_full` is the **one** place a raw index is asserted to be in
    /// full-x. In the solver it is `nlp.var_x_to_full_x`; keeping it to
    /// a single call site per map is the whole protection this type
    /// offers, so do not add a second way to mint a [`FullX`].
    #[must_use]
    pub fn build(n_var_x: usize, mut to_full: impl FnMut(VarX) -> usize) -> Self {
        Self {
            full_of: (0..n_var_x)
                .map(|r| FullX::new(to_full(VarX::new(r))))
                .collect(),
        }
    }

    /// The full-x index of a var-x row, or [`None`] if the row is
    /// outside the primal block.
    #[must_use]
    pub fn full_of(&self, row: VarX) -> Option<FullX> {
        self.full_of.get(row.get()).copied()
    }

    /// Width of the primal block.
    #[must_use]
    pub fn n_var_x(&self) -> usize {
        self.full_of.len()
    }

    /// Every var-x row, in order — so a loop is typed from the start
    /// rather than by converting a bare `usize` at the first use.
    pub fn rows(&self) -> impl Iterator<Item = VarX> + '_ {
        (0..self.full_of.len()).map(VarX::new)
    }
}

/// A per-full-x slice that can only be read with a [`FullX`].
///
/// Used for the `ActivityReport` arrays at the two sites where a var-x
/// row is live in the same scope. It borrows rather than owns, so it
/// costs nothing and does not duplicate the report.
#[derive(Clone, Copy, Debug)]
pub struct FullXSlice<'a, T> {
    inner: &'a [T],
}

impl<'a, T: Copy> FullXSlice<'a, T> {
    /// Wrap a slice asserted to be in full-x order.
    #[must_use]
    pub fn new(inner: &'a [T]) -> Self {
        Self { inner }
    }

    /// Read one entry, or [`None`] when the index is past the end.
    #[must_use]
    pub fn at(&self, idx: FullX) -> Option<T> {
        self.inner.get(idx.get()).copied()
    }

    /// Length, in full-x entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the slice is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// The two spaces do not convert implicitly, in either direction.
///
/// These are the evidence that the newtype does the one job it exists
/// for. Each `compile_fail` is paired with the passing twin directly
/// below it, because a `compile_fail` that fails for the *wrong*
/// reason — a typo, an unresolved import — is a test that passes
/// vacuously, which is the failure mode this crate cares most about.
///
/// A var-x row cannot be used where a full-x index is expected:
///
/// ```compile_fail
/// use pounce_sensitivity::index::{FullX, VarX};
/// fn full_only(_: FullX) {}
/// full_only(VarX::new(0));
/// ```
///
/// and the same call typechecks with the right space, so the failure
/// above is about the types and not about the spelling:
///
/// ```
/// use pounce_sensitivity::index::{FullX, VarToFull, VarX};
/// fn full_only(_: FullX) {}
/// let map = VarToFull::build(1, |v| v.get());
/// full_only(map.full_of(VarX::new(0)).unwrap());
/// ```
///
/// And a `FullX` cannot be minted from a raw row at all, which is what
/// keeps the guarantee above from being one short line wide:
///
/// ```compile_fail
/// use pounce_sensitivity::index::{FullX, VarX};
/// let _ = FullX::new(VarX::new(0).get());
/// ```
///
/// A full-x index cannot be used where a var-x row is expected:
///
/// ```compile_fail
/// use pounce_sensitivity::index::{VarToFull, VarX};
/// fn var_only(_: VarX) {}
/// let map = VarToFull::build(1, |v| v.get());
/// var_only(map.full_of(VarX::new(0)).unwrap());
/// ```
///
/// ```
/// use pounce_sensitivity::index::{VarToFull, VarX};
/// fn var_only(_: VarX) {}
/// let map = VarToFull::build(1, |v| v.get());
/// var_only(VarX::new(0));
/// let _ = map.full_of(VarX::new(0)).unwrap();
/// ```
///
/// A full-x slice cannot be read with a var-x row — this is gh#672
/// finding 1's exact shape, `report.var_status.get(var_row)`, as a
/// type error:
///
/// ```compile_fail
/// use pounce_sensitivity::index::{FullXSlice, VarX};
/// let status = [0i8, 1, 2];
/// let full = FullXSlice::new(&status);
/// full.at(VarX::new(1));
/// ```
///
/// ```
/// use pounce_sensitivity::index::{FullXSlice, VarToFull, VarX};
/// let status = [0i8, 1, 2];
/// let full = FullXSlice::new(&status);
/// let map = VarToFull::build(2, |v| v.get() + 1);
/// assert_eq!(full.at(map.full_of(VarX::new(0)).unwrap()), Some(1));
/// assert_eq!(full.at(map.full_of(VarX::new(1)).unwrap()), Some(2));
/// ```
///
/// And a raw `usize` reaches neither, so an untyped index cannot drift
/// in from a caller:
///
/// ```compile_fail
/// use pounce_sensitivity::index::FullXSlice;
/// let status = [0i8, 1, 2];
/// FullXSlice::new(&status).at(1usize);
/// ```
#[cfg(doctest)]
pub struct IndexSpacesDoNotConvert;

/// A miss returns `None`, never a neighbour.
///
/// The defect this module exists for is not an out-of-bounds panic —
/// it is an in-range read of the wrong row. The bounds check below is
/// the cheap half; the type is the half that matters.
#[cfg(test)]
mod tests {
    use super::*;
    use pounce_common::types::Number;

    #[test]
    fn a_lookup_past_the_end_is_none_not_a_neighbour() {
        let map = VarToFull::build(2, |v| [0usize, 2][v.get()]);
        assert_eq!(map.full_of(VarX::new(0)), Some(FullX::new(0)));
        assert_eq!(map.full_of(VarX::new(1)), Some(FullX::new(2)));
        assert_eq!(map.full_of(VarX::new(2)), None);
        assert_eq!(map.n_var_x(), 2);
    }

    /// The map is where the divergence lives: with a fixed variable
    /// ahead of the rows, var-x `k` is full-x `k + 1`, so reading one
    /// as the other returns a neighbour. This is leg 3's fixture shape
    /// in miniature.
    #[test]
    fn a_fixed_variable_ahead_makes_the_spaces_diverge() {
        // full-x 0 is `make_parameter`-removed, so var-x 0,1,2 are
        // full-x 1,2,3.
        let map = VarToFull::build(3, |v| v.get() + 1);
        for row in map.rows() {
            let full = map.full_of(row).expect("in range");
            assert_eq!(
                full.get(),
                row.get() + 1,
                "the spaces must diverge by the fixed variable"
            );
        }

        // and the untyped read that gh#672 finding 1 shipped would have
        // returned the neighbour rather than failing
        let sigma = [10.0, 20.0, 30.0, 40.0];
        let slice = FullXSlice::new(&sigma);
        let row = VarX::new(1);
        let correct = slice.at(map.full_of(row).unwrap());
        let untyped_swap = slice.at(FullX::new(row.get()));
        assert_eq!(correct, Some(30.0));
        assert_eq!(untyped_swap, Some(20.0));
        assert_ne!(
            correct, untyped_swap,
            "if these agree the fixture has no fixed variable and proves nothing",
        );
    }

    #[test]
    fn rows_are_typed_from_the_start() {
        let map = VarToFull::build(2, |v| v.get() + 1);
        let rows: Vec<VarX> = map.rows().collect();
        assert_eq!(rows, vec![VarX::new(0), VarX::new(1)]);
    }

    #[test]
    fn a_full_x_slice_reports_its_own_length() {
        let v = [1.0, 2.0, 3.0];
        let s = FullXSlice::new(&v);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        let empty: [Number; 0] = [];
        assert!(FullXSlice::new(&empty).is_empty());
    }
}
