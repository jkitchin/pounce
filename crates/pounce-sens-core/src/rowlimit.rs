//! Watching a limit that is written as a **constraint row**, on an
//! engine whose KKT has no slack block (gh#929).
//!
//! # The gap
//!
//! [`crate::boundcheck`]'s walk and refinement decide everything in
//! *primal KKT rows*: the box `lo`/`hi` and the base point `x_curr`
//! index the primal prefix of the compound vector, and a
//! [`BoundRow`](crate::backsolver::BoundRow) ties a multiplier row to
//! the primal row it constrains. That is what lets the NLP arm watch a
//! constraint's own limit — there, `dⱼ(x) = sⱼ` makes the limit a
//! bound on a coordinate the KKT already carries, which is the gh#928
//! fix.
//!
//! An active-set KKT has no such coordinate. `pounce-convex` assembles
//!
//! ```text
//! [ H   Aᵀ  B_aᵀ ] [ dx ]
//! [ A   0   0    ] [ dy ]
//! [ B_a 0   0    ] [ dz ]
//! ```
//!
//! where `B_a` holds the *active* rows only. A row `Gⱼ x ≤ hⱼ` that is
//! **inactive** appears nowhere at all, so a step that drives `Gⱼ x`
//! past `hⱼ` is not a breakpoint the walk can see — it is silence, and
//! the answer comes back infeasible. An **active** row has a
//! multiplier row but still no primal coordinate, so it cannot be
//! reported as a `BoundRow` either, and a perturbation that drives its
//! multiplier negative holds a row the solution has left.
//!
//! # The observer block
//!
//! This wrapper gives each watched row a coordinate of its own. For
//! watched rows `G_w` it presents the augmented system
//!
//! ```text
//! [ Kxx   0   Kxr  -G_wᵀ ] [ dx    ]   [ r_x   ]
//! [ 0     0   0     I    ] [ dt    ] = [ r_t   ]
//! [ Krx   0   Krr   0    ] [ drest ]   [ r_rest]
//! [ -G_w  I   0     0    ] [ dmu   ]   [ r_mu  ]
//! ```
//!
//! which is the base system with `t = G_w x` adjoined through a
//! multiplier `mu` (Lagrangian term `muᵀ(t − G_w x)`). `t` sits
//! immediately after the `x` block, so it lands inside the primal
//! prefix the walk indexes, and carries the box `(−∞, h]`.
//!
//! **It costs no factorization.** The block is triangular in the
//! adjoined variables:
//!
//! ```text
//! dmu = r_t
//! K [dx; drest] = [r_x + G_wᵀ r_t ; r_rest]      <- the base solve
//! dt  = G_w dx + r_mu
//! ```
//!
//! so one base back-solve and two sparse mat-vecs answer it, released
//! or not. Nothing about the base factor changes, and an active row's
//! release is still the base's own row-neutralization: the observer
//! reads `dt = G_w dx + r_mu` whether or not the row it observes is
//! being enforced.
//!
//! # Why the shift needs no special case
//!
//! [`SensBacksolver::solve_released_step`] moves a released
//! multiplier onto the primal row it was acting on. Here that primal
//! row is `tⱼ`, and `r_t` reaches the `x` rows as `G_wᵀ r_t` — which
//! is exactly `zⱼ Gⱼᵀ`, the force the released row was applying. The
//! observer's own algebra performs the conversion, so this file adds
//! the shift in `t` and never writes a `G` row into an `x` right-hand
//! side by hand.
//!
//! # Index spaces
//!
//! Two live here at once, which is the shape gh#450 and gh#764 say to
//! be careful about. [`RowLimitView::to_base`] and
//! [`RowLimitView::from_base`] are the only conversions, both total
//! functions with an explicit `None` for the adjoined rows, and every
//! read of a base-indexed vector goes through one of them.

use pounce_common::types::Number;

use crate::backsolver::{BoundRow, SensBacksolver};

/// One watched limit: a sparse row `Σ coef·x[col] ≤ limit`, and the
/// KKT row of its multiplier when the base system already enforces it.
#[derive(Clone, Debug)]
pub struct WatchedRow {
    /// The row's nonzeros as `(column, coefficient)`, in the `x`
    /// block's index space.
    pub coefficients: Vec<(usize, Number)>,
    /// The row's right-hand side, which becomes the observer's upper
    /// bound. A row written `≥` must be negated by the caller before
    /// it gets here; this type is one-sided on purpose, because a
    /// two-sided row is two watched rows and pretending otherwise
    /// hides which side a breakpoint belongs to.
    pub limit: Number,
    /// The value of `Σ coef·x[col]` at the base point.
    pub base_value: Number,
    /// `Some((multiplier_row, base_multiplier))` when the base system
    /// carries this row in its active set, where `multiplier_row` is
    /// in the **base** index space. `None` for an inactive row, which
    /// the walk can then only reach and hold, never release.
    pub active: Option<(usize, Number)>,
}

/// A [`SensBacksolver`] that adjoins an observer coordinate to each
/// watched row of its base. See the module docs.
#[derive(Clone)]
pub struct RowLimitView<B> {
    base: B,
    n_x: usize,
    rows: Vec<WatchedRow>,
    /// The base's dimension, cached because it appears in every
    /// conversion.
    base_dim: usize,
    /// Every releasable row this view offers the walk, in **this
    /// view's** index space: the base's own bound rows shifted past the
    /// observers, then one per watched row the base holds active.
    ///
    /// The concatenation is built here rather than left to the caller
    /// because the walk reads a single slice and can only release what
    /// it finds in it — a view that reported only its own rows would
    /// silently take variable-bound releases away from the arm it
    /// wraps.
    bound_rows: Vec<BoundRow>,
    /// Per watched row the base holds active: `(multiplier row in this
    /// view's space, observer row, base multiplier)`, for the release
    /// shift. Not parallel to [`Self::bound_rows`], which also carries
    /// the base's.
    shifts: Vec<(usize, usize, Number)>,
}

impl<B: SensBacksolver> RowLimitView<B> {
    /// Wrap `base`, whose `x` block is `n_x` wide, with one observer
    /// per entry of `rows`.
    ///
    /// `None` when the wrapper cannot be built honestly:
    ///
    /// * `n_x` past the base dimension, or a coefficient column past
    ///   `n_x` — the caller's index space is not the one it thinks;
    /// * an `active` multiplier row outside the base's non-`x` rows;
    /// * **the base reports a
    ///   [`natural_units_factor`](SensBacksolver::natural_units_factor)**.
    ///   The observer rows have no entry in it, and inventing one
    ///   would put a scaled right-hand side into an unscaled Schur
    ///   complement. Refusing is the honest answer; the convex arm,
    ///   the only caller, always answers `None` there because its KKT
    ///   is assembled from raw problem data.
    pub fn new(base: B, n_x: usize, rows: Vec<WatchedRow>) -> Option<Self> {
        let base_dim = base.dim();
        if n_x > base_dim || base.natural_units_factor().is_some() {
            return None;
        }
        let n_t = rows.len();
        let mut bound_rows: Vec<BoundRow> = Vec::new();
        // The base's own rows first, shifted into this space. A base
        // bound row whose `var_row` is not in the `x` block cannot be
        // re-expressed here, because the observers were inserted
        // immediately after `x`; refuse rather than mis-map it.
        if let Some(base_rows) = base.bound_rows() {
            for b in base_rows {
                if b.var_row >= n_x || b.row < n_x || b.row >= base_dim {
                    return None;
                }
                bound_rows.push(BoundRow {
                    row: b.row + n_t,
                    var_row: b.var_row,
                    lower: b.lower,
                });
            }
        }
        let mut shifts = Vec::new();
        for (j, r) in rows.iter().enumerate() {
            if r.coefficients.iter().any(|&(c, _)| c >= n_x) {
                return None;
            }
            if let Some((row, mult)) = r.active {
                // An active row's multiplier lives in the base's
                // non-primal rows; one inside the `x` block is a
                // caller confusing the two spaces.
                if row < n_x || row >= base_dim {
                    return None;
                }
                let obs = n_x + j;
                bound_rows.push(BoundRow {
                    row: row + n_t,
                    var_row: obs,
                    // `G x ≤ h` is an upper limit on the observer.
                    lower: false,
                });
                shifts.push((row + n_t, obs, mult));
            }
        }
        Some(Self {
            base,
            n_x,
            rows,
            base_dim,
            bound_rows,
            shifts,
        })
    }

    /// Number of observers, which is also the offset every base row at
    /// or past the `x` block moves by.
    pub fn n_observers(&self) -> usize {
        self.rows.len()
    }

    /// This view's row for a base row. Total, and the only place the
    /// `+ n_t` shift is written.
    pub fn from_base(&self, base_row: usize) -> Option<usize> {
        if base_row < self.n_x {
            Some(base_row)
        } else if base_row < self.base_dim {
            Some(base_row + self.rows.len())
        } else {
            None
        }
    }

    /// The base row behind a row of this view, or `None` for an
    /// adjoined `t` or `mu` row — which is a real answer, not a
    /// failure: those rows exist only here.
    pub fn to_base(&self, row: usize) -> Option<usize> {
        let n_t = self.rows.len();
        if row < self.n_x {
            Some(row)
        } else if row < self.n_x + n_t {
            None
        } else if row < self.base_dim + n_t {
            Some(row - n_t)
        } else {
            None
        }
    }

    /// The walk's box and base point over the primal prefix `x` then
    /// `t`, given the base's own `x`-block box and point.
    ///
    /// Returned rather than assembled by the caller so that the
    /// observer's bounds and the observer's ordering cannot drift
    /// apart from the ones [`Self::new`] built the `BoundRow`s
    /// against.
    pub fn primal_box(
        &self,
        x_curr: &[Number],
        lo: &[Number],
        hi: &[Number],
    ) -> Option<(Vec<Number>, Vec<Number>, Vec<Number>)> {
        if x_curr.len() != self.n_x || lo.len() != self.n_x || hi.len() != self.n_x {
            return None;
        }
        let mut x = x_curr.to_vec();
        let mut l = lo.to_vec();
        let mut h = hi.to_vec();
        for r in &self.rows {
            x.push(r.base_value);
            l.push(Number::NEG_INFINITY);
            h.push(r.limit);
        }
        Some((x, l, h))
    }

    /// A base-space right-hand side lifted into this view's space.
    pub fn lift_rhs(&self, base_rhs: &[Number]) -> Option<Vec<Number>> {
        if base_rhs.len() != self.base_dim {
            return None;
        }
        let mut out = vec![0.0; self.dim()];
        for (i, &v) in base_rhs.iter().enumerate() {
            let row = self.from_base(i)?;
            out[row] = v;
        }
        Some(out)
    }

    /// A base-space **solution** lifted into this view's space, the
    /// left-hand-side counterpart of [`Self::lift_rhs`].
    ///
    /// The observer coordinate of a step is `dt = G_w dx`, and the
    /// adjoined multiplier of a step whose right-hand side came through
    /// `lift_rhs` is zero — that right-hand side puts nothing in the
    /// `t` or `mu` rows. So this is [`Self::unfold`] with both those
    /// pieces zero, written that way rather than open-coded so a lifted
    /// step cannot drift from what [`SensBacksolver::solve`] returns
    /// for the same input. `lifting_a_step_agrees_with_solving_it`
    /// is what holds the two together.
    ///
    /// A caller that already has the base step in hand — every one
    /// does, since the plain step is what the refinement corrects —
    /// pays two sparse mat-vecs here instead of a second back-solve.
    pub fn lift_step(&self, base_lhs: &[Number]) -> Option<Vec<Number>> {
        if base_lhs.len() != self.base_dim {
            return None;
        }
        let zeros = vec![0.0; self.rows.len()];
        let mut out = vec![0.0; self.dim()];
        self.unfold(base_lhs, &zeros, &zeros, &mut out);
        Some(out)
    }

    /// Every releasable row of this view, base and watched alike, in
    /// this view's index space. Same slice
    /// [`SensBacksolver::bound_rows`] returns.
    pub fn all_bound_rows(&self) -> &[BoundRow] {
        &self.bound_rows
    }

    /// The walk's multiplier list for this view: the base's own entries
    /// lifted, then one per watched row the base holds active.
    ///
    /// Built here rather than by the caller so it cannot fall out of
    /// step with [`Self::all_bound_rows`] — the walk releases a row
    /// only when it finds it in *both* lists, so a row present in one
    /// and missing from the other is a release that silently never
    /// happens.
    pub fn lift_multipliers(
        &self,
        base: &[crate::boundcheck::BoundMultiplier],
    ) -> Option<Vec<crate::boundcheck::BoundMultiplier>> {
        let mut out = Vec::with_capacity(base.len() + self.shifts.len());
        for m in base {
            out.push(crate::boundcheck::BoundMultiplier {
                row: self.from_base(m.row)?,
                base: m.base,
            });
        }
        for &(row, _, mult) in &self.shifts {
            out.push(crate::boundcheck::BoundMultiplier { row, base: mult });
        }
        Some(out)
    }

    /// Split a view-space right-hand side into the base solve's
    /// right-hand side and the pieces the observer needs afterwards.
    ///
    /// `dmu = r_t`, and the base sees `r_x + G_wᵀ r_t`.
    fn fold(&self, rhs: &[Number]) -> (Vec<Number>, Vec<Number>, Vec<Number>) {
        let n_t = self.rows.len();
        let r_t = rhs[self.n_x..self.n_x + n_t].to_vec();
        let r_mu = rhs[self.base_dim + n_t..].to_vec();
        let mut base_rhs = vec![0.0; self.base_dim];
        base_rhs[..self.n_x].copy_from_slice(&rhs[..self.n_x]);
        base_rhs[self.n_x..].copy_from_slice(&rhs[self.n_x + n_t..self.base_dim + n_t]);
        for (j, r) in self.rows.iter().enumerate() {
            let v = r_t[j];
            if v != 0.0 {
                for &(c, coef) in &r.coefficients {
                    base_rhs[c] += coef * v;
                }
            }
        }
        (base_rhs, r_t, r_mu)
    }

    /// Scatter a base answer back into this view's space and fill the
    /// observer rows: `dt = G_w dx + r_mu`, `dmu = r_t`.
    fn unfold(&self, base_lhs: &[Number], r_t: &[Number], r_mu: &[Number], lhs: &mut [Number]) {
        let n_t = self.rows.len();
        lhs[..self.n_x].copy_from_slice(&base_lhs[..self.n_x]);
        lhs[self.n_x + n_t..self.base_dim + n_t].copy_from_slice(&base_lhs[self.n_x..]);
        for (j, r) in self.rows.iter().enumerate() {
            let gx: Number = r
                .coefficients
                .iter()
                .map(|&(c, coef)| coef * base_lhs[c])
                .sum();
            lhs[self.n_x + j] = gx + r_mu[j];
            lhs[self.base_dim + n_t + j] = r_t[j];
        }
    }

    /// Common body of the three solves: fold, run `f` on the base
    /// system, unfold.
    fn around<F>(&self, rhs: &[Number], lhs: &mut [Number], f: F) -> bool
    where
        F: FnOnce(&[Number], &mut [Number]) -> bool,
    {
        if rhs.len() != self.dim() || lhs.len() != self.dim() {
            return false;
        }
        let (base_rhs, r_t, r_mu) = self.fold(rhs);
        let mut base_lhs = vec![0.0; self.base_dim];
        if !f(&base_rhs, &mut base_lhs) {
            return false;
        }
        self.unfold(&base_lhs, &r_t, &r_mu, lhs);
        true
    }

    /// `released`, in the base's index space, or `None` if any entry
    /// is an adjoined row — which no caller should produce, since the
    /// only releasable rows this view reports are base multiplier rows.
    fn released_in_base(&self, released: &[usize]) -> Option<Vec<usize>> {
        released.iter().map(|&r| self.to_base(r)).collect()
    }
}

impl<B: SensBacksolver> SensBacksolver for RowLimitView<B> {
    fn dim(&self) -> usize {
        self.base_dim + 2 * self.rows.len()
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.around(rhs, lhs, |b, l| self.base.solve(b, l))
    }

    /// `None`, and guaranteed rather than assumed:
    /// [`RowLimitView::new`] refuses a base that reports one, because
    /// the observer rows have no honest entry in it.
    fn natural_units_factor(&self) -> Option<&[Number]> {
        None
    }

    fn bound_rows(&self) -> Option<&[BoundRow]> {
        Some(&self.bound_rows)
    }

    fn supports_release(&self) -> bool {
        self.base.supports_release()
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        let Some(key) = self.released_in_base(released) else {
            return false;
        };
        self.around(rhs, lhs, |b, l| self.base.solve_released(&key, b, l))
    }

    /// The released step, with the observer carrying the shift.
    ///
    /// A released *row*'s multiplier is moved onto its observer row —
    /// `r_t[j] += z` for the upper-limit orientation every watched row
    /// has — and [`RowLimitView::fold`] then delivers it to the `x`
    /// rows as `z Gⱼᵀ`. A released *variable bound* is not this view's
    /// business and is left to the base's own shift.
    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        if rhs.len() != self.dim() {
            return false;
        }
        let Some(key) = self.released_in_base(released) else {
            return false;
        };
        let mut rhs = rhs.to_vec();
        for &(row, obs, mult) in &self.shifts {
            if !released.contains(&row) {
                continue;
            }
            rhs[row] = 0.0;
            rhs[obs] += mult;
        }
        self.around(&rhs, lhs, |b, l| self.base.solve_released_step(&key, b, l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backsolver::DenseLuBacksolver;

    /// A dense stand-in for a factored KKT that also answers the
    /// release half of the trait, by refactoring with the released
    /// rows neutralized the way a real backsolver does.
    ///
    /// It carries no bound rows of its own, so `solve_released_step`
    /// has nothing to shift and coincides with `solve_released` — which
    /// is the point: every shift these tests observe is one
    /// [`RowLimitView`] put there.
    struct Base {
        n: usize,
        k: Vec<Number>,
        rows: Vec<BoundRow>,
    }

    impl Base {
        fn factor(&self, released: &[usize]) -> Option<DenseLuBacksolver> {
            let mut k = self.k.clone();
            for &r in released {
                for j in 0..self.n {
                    k[r * self.n + j] = 0.0;
                    k[j * self.n + r] = 0.0;
                }
                k[r * self.n + r] = -1.0;
            }
            DenseLuBacksolver::from_dense(self.n, &k).ok()
        }
    }

    impl SensBacksolver for Base {
        fn dim(&self) -> usize {
            self.n
        }
        fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
            self.factor(&[]).is_some_and(|f| f.solve(rhs, lhs))
        }
        fn bound_rows(&self) -> Option<&[BoundRow]> {
            Some(&self.rows)
        }
        fn supports_release(&self) -> bool {
            true
        }
        fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
            self.factor(released).is_some_and(|f| f.solve(rhs, lhs))
        }
        fn solve_released_step(
            &self,
            released: &[usize],
            rhs: &[Number],
            lhs: &mut [Number],
        ) -> bool {
            self.solve_released(released, rhs, lhs)
        }
    }

    /// `n_x = 3`, two further rows, symmetric and nonsingular.
    fn base(rows: Vec<BoundRow>) -> Base {
        #[rustfmt::skip]
        let k = vec![
            2.0, 0.3, -0.4,  1.0, 0.0,
            0.3, 1.7,  0.2,  0.0, 1.0,
           -0.4, 0.2,  2.5,  1.0, 1.0,
            1.0, 0.0,  1.0,  0.0, 0.0,
            0.0, 1.0,  1.0,  0.0, 0.0,
        ];
        Base { n: 5, k, rows }
    }

    fn watched(active: Option<(usize, Number)>, coefficients: Vec<(usize, Number)>) -> WatchedRow {
        WatchedRow {
            coefficients,
            limit: 1.0,
            base_value: 0.25,
            active,
        }
    }

    /// The triangular solve is the augmented system, not merely
    /// something that resembles it.
    ///
    /// The claim is checked against a dense factorization of the whole
    /// `9 × 9` operator the module docs write out, on a right-hand side
    /// with **all four blocks nonzero**. The `r_mu` block is the reason:
    /// nothing on the convex arm ever produces one — `lift_rhs` zeroes
    /// it — so `unfold` dropping `+ r_mu[j]` is invisible to every
    /// integration test in the repo and visible here.
    #[test]
    fn the_triangular_solve_is_the_augmented_system() {
        let n_x = 3;
        let rows = vec![
            watched(None, vec![(0, 1.0), (2, -0.5)]),
            watched(None, vec![(1, 2.0)]),
        ];
        let b = base(Vec::new());
        let base_k = b.k.clone();
        let base_dim = b.n;
        let n_t = rows.len();
        let view = RowLimitView::new(b, n_x, rows.clone()).expect("the view must build");
        let dim = view.dim();
        assert_eq!(dim, base_dim + 2 * n_t);

        // Assemble the augmented operator densely, in this view's row
        // order: x, t, rest, mu.
        let t0 = n_x;
        let mu0 = base_dim + n_t;
        let mut a = vec![0.0; dim * dim];
        let view_of = |i: usize| if i < n_x { i } else { i + n_t };
        for i in 0..base_dim {
            for j in 0..base_dim {
                a[view_of(i) * dim + view_of(j)] = base_k[i * base_dim + j];
            }
        }
        for (j, r) in rows.iter().enumerate() {
            // `I` in the t-row block against mu, and in the mu-row block
            // against t.
            a[(t0 + j) * dim + mu0 + j] = 1.0;
            a[(mu0 + j) * dim + t0 + j] = 1.0;
            for &(c, coef) in &r.coefficients {
                a[c * dim + mu0 + j] = -coef;
                a[(mu0 + j) * dim + c] = -coef;
            }
        }
        let dense =
            DenseLuBacksolver::from_dense(dim, &a).expect("the augmented system is regular");

        let rhs: Vec<Number> = (0..dim).map(|i| 0.7 - 0.31 * (i as Number)).collect();
        let mut want = vec![0.0; dim];
        assert!(dense.solve(&rhs, &mut want));
        let mut got = vec![0.0; dim];
        assert!(view.solve(&rhs, &mut got));
        for i in 0..dim {
            assert!(
                (got[i] - want[i]).abs() < 1e-10,
                "row {i}: view {got:?} vs dense {want:?}",
            );
        }
    }

    /// Each active watched row gets **its own** observer, and the
    /// release shift lands on that one.
    ///
    /// Both halves need two rows with different coefficients to say
    /// anything: with a single watched row every ordering of the
    /// observers is the same ordering, which is exactly the shape the
    /// convex fixture has.
    /// [`RowLimitView::lift_step`] is the cheap route to the augmented
    /// plain step, and cheap is only worth having if it is the same
    /// answer: a caller with the base step in hand skips a back-solve
    /// by using it.
    ///
    /// The mutation this catches is the tempting one — lifting a step
    /// by zero-filling the observers instead of evaluating
    /// `dt = G_w dx`. That reads as a step whose watched rows do not
    /// move at all, so nothing ever reaches a limit and the caller is
    /// back to the defect it was fixing, with no error anywhere.
    #[test]
    fn lifting_a_step_agrees_with_solving_it() {
        let n_x = 3;
        let rows = vec![
            watched(None, vec![(0, 1.0), (2, -0.5)]),
            watched(Some((3, 0.7)), vec![(1, 2.0)]),
        ];
        let b = base(Vec::new());
        let base_dim = b.n;
        let view = RowLimitView::new(b, n_x, rows).expect("the view must build");

        // A base-space right-hand side, lifted, solved in the view.
        let base_rhs: Vec<Number> = vec![0.4, -1.1, 0.9, 0.3, -0.6];
        let lifted_rhs = view.lift_rhs(&base_rhs).expect("the rhs lifts");
        let mut want = vec![0.0; view.dim()];
        assert!(view.solve(&lifted_rhs, &mut want), "the view must solve");

        // The same thing from the base answer alone.
        let mut base_lhs = vec![0.0; base_dim];
        assert!(
            view.base.solve(&base_rhs, &mut base_lhs),
            "the base must solve",
        );
        let got = view.lift_step(&base_lhs).expect("the step lifts");

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-10,
                "row {i}: lift_step {g} against solve {w}\n{got:?}\n{want:?}",
            );
        }
        // Not vacuous: the observers actually moved.
        assert!(
            got[n_x..n_x + 2].iter().any(|v| v.abs() > 1e-6),
            "the observers read {:?}, so this test would pass on a \
             zero-filling lift",
            &got[n_x..n_x + 2],
        );
    }

    /// Length is the only thing `lift_step` can refuse, and refusing is
    /// the point: a base answer of the wrong width is a caller holding
    /// the view's own vector by mistake, which would otherwise scatter
    /// into the observer rows.
    #[test]
    fn lift_step_refuses_a_vector_that_is_not_the_base() {
        let rows = vec![watched(None, vec![(0, 1.0)])];
        let view = RowLimitView::new(base(Vec::new()), 3, rows).expect("the view must build");
        assert!(view.lift_step(&vec![0.0; view.dim()]).is_none());
        assert!(view.lift_step(&[]).is_none());
        assert!(view.lift_step(&vec![0.0; 5]).is_some());
    }

    #[test]
    fn each_active_row_keeps_its_own_observer() {
        let n_x = 3;
        let (m0, m1) = (0.75, 0.2);
        let rows = vec![
            watched(Some((3, m0)), vec![(0, 1.0), (2, -0.5)]),
            watched(Some((4, m1)), vec![(1, 2.0)]),
        ];
        let n_t = rows.len();
        let b = base(Vec::new());
        let reference = base(Vec::new());
        let view = RowLimitView::new(b, n_x, rows.clone()).expect("the view must build");

        // Multiplier row `3 + n_t` observes `n_x + 0`, `4 + n_t` observes
        // `n_x + 1` — not the other way round.
        assert_eq!(
            view.all_bound_rows(),
            &[
                BoundRow {
                    row: 3 + n_t,
                    var_row: n_x,
                    lower: false
                },
                BoundRow {
                    row: 4 + n_t,
                    var_row: n_x + 1,
                    lower: false
                },
            ],
        );
        let lifted = view
            .lift_multipliers(&[])
            .expect("an empty base list still lifts");
        assert_eq!(lifted.len(), 2);
        for (got, want) in lifted.iter().zip([(3 + n_t, m0), (4 + n_t, m1)]) {
            assert_eq!((got.row, got.base), want);
        }

        // Release the *first* row against a zero right-hand side. Its
        // multiplier is the only force left, and `fold` delivers it as
        // `m0 · G₀ᵀ` — so the answer is the released base solve on that
        // vector, and would differ if the shift had landed on the other
        // observer.
        let released = vec![3 + n_t];
        let mut got = vec![0.0; view.dim()];
        assert!(view.solve_released_step(&released, &vec![0.0; view.dim()], &mut got));

        let mut want_rhs = vec![0.0; 5];
        for &(c, coef) in &rows[0].coefficients {
            want_rhs[c] += coef * m0;
        }
        let mut want = vec![0.0; 5];
        assert!(reference.solve_released(&[3], &want_rhs, &mut want));
        for i in 0..n_x {
            assert!(
                (got[i] - want[i]).abs() < 1e-10,
                "x row {i}: {got:?} vs {want:?}",
            );
        }
        assert!(
            got[..n_x].iter().any(|v| v.abs() > 1e-6),
            "the shift must actually move something: {got:?}",
        );
    }
}
