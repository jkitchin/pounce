//! Powell-damped BFGS Hessian approximation for SQP (Powell
//! 1978, *Numerical Analysis Dundee 1977*). Used when
//! `SqpOptions::hessian = DampedBfgs` — the QP subproblem's
//! Hessian comes from this rank-2-updated matrix instead of
//! `nlp.eval_hess_lag`.
//!
//! Powell's damping rule guarantees positive-definiteness of
//! every iterate, so the QP solver doesn't have to engage
//! inertia control to keep `∇²L`-quadratic models PD. The
//! damping factor `θ ∈ [0, 1]` interpolates between the raw
//! BFGS `y = ∇L_new − ∇L_old` and the conservative `B·s`:
//!
//! ```text
//!     if sᵀy ≥ 0.2 · sᵀ B s :  θ = 1            (standard BFGS)
//!     else                  :  θ = 0.8 · sᵀ B s / (sᵀ B s − sᵀy)
//!     y_damp = θ y + (1 − θ) B s
//!     B_new = B − (Bs · sᵀB) / (sᵀ B s)
//!                  + (y_damp · y_dampᵀ) / (sᵀ y_damp)
//! ```
//!
//! Storage is dense `n × n` (lower-triangle row-major); exposed
//! to `pounce-qp` as a fully-populated [`Triplet`] over the upper
//! triangle (1-based row/col).

use crate::sqp::qp_assembly::Triplet;
use pounce_common::types::{Index, Number};

/// A curvature pair with `sᵀy > RESIZE_RATIO · sᵀBs` shifts `B` up (see
/// [`DampedBfgs::update_sy`]).
const RESIZE_RATIO: Number = 10.0;
/// ... and only once the constraint multipliers have moved by more than this
/// fraction of their max-norm (at least 1) since `B` was last sized.
const RESIZE_LAMBDA_MOVE: Number = 0.5;

pub struct DampedBfgs {
    n: usize,
    /// Lower-triangle row-major storage:
    /// `b[i*(i+1)/2 + j] = B[i, j]` for `i ≥ j`.
    b: Vec<Number>,
    /// Previous `x` and ∇L; updated at the end of each `update` call.
    prev_x: Option<Vec<Number>>,
    prev_grad_lag: Option<Vec<Number>>,
    /// Whether the one-time initial sizing has been applied yet (see
    /// [`Self::update`]). `false` until the first `(s, y)` pair with
    /// `sᵀy > 0` arrives, at which point `B` is rescaled from the
    /// identity to `γI` and this flips to `true`.
    sized: bool,
    /// Constraint multipliers at the last sizing, when the caller supplies
    /// them (nonlinear constraints only; see [`Self::update_sy_at`]).
    sized_lambda: Option<Vec<Number>>,
    /// Pre-computed sparsity pattern for `as_triplet`. Fixed:
    /// every (i, j) with `i ≥ j`. 1-based.
    h_irow: Vec<Index>,
    h_jcol: Vec<Index>,
}

impl DampedBfgs {
    pub fn new(n: usize) -> Self {
        let nz = n * (n + 1) / 2;
        let mut b = vec![0.0; nz];
        let mut h_irow = Vec::with_capacity(nz);
        let mut h_jcol = Vec::with_capacity(nz);
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    b[i * (i + 1) / 2 + j] = 1.0;
                }
                h_irow.push((i + 1) as Index);
                h_jcol.push((j + 1) as Index);
            }
        }
        Self {
            n,
            b,
            prev_x: None,
            prev_grad_lag: None,
            sized: false,
            sized_lambda: None,
            h_irow,
            h_jcol,
        }
    }

    /// Have we recorded a previous `(x, ∇L)`? `false` until the
    /// first call to [`Self::update`].
    pub fn has_prev(&self) -> bool {
        self.prev_x.is_some()
    }

    /// Seed `B = γI` directly and mark the one-time sizing done, so the
    /// first [`Self::update`] applies its rank-2 correction on top of
    /// this scale instead of re-seeding from its own `(s, y)`.
    ///
    /// Used by the driver's iteration-0 curvature probe: the internal
    /// sizing in [`Self::update`] cannot fire until a first `(s, y)` pair
    /// exists, i.e. not until iteration 1 — but iteration **0** already
    /// solves a QP against `B`, and with the identity seed that step
    /// overshoots by `~cond(∇²L)` on an ill-conditioned problem. See the
    /// sizing comment in [`Self::update`] for why that is fatal.
    ///
    /// `gamma` must be finite and strictly positive; anything else is
    /// ignored (leaving `B = I`) rather than corrupting the matrix.
    pub fn seed_scale(&mut self, gamma: Number) {
        if !gamma.is_finite() || gamma <= 0.0 {
            return;
        }
        for i in 0..self.n {
            self.set(i, i, gamma);
        }
        self.sized = true;
    }

    /// Discard the accumulated rank-2 curvature and fall back to a
    /// scaled identity `γI`, where `γ` is the current mean diagonal
    /// (a scale the accumulated matrix has already vouched for).
    /// `prev_x` / `prev_grad_lag` are retained, so the next
    /// [`Self::update`] resumes accumulating from the reset base.
    ///
    /// Used as a recovery step when the QP subproblem fails: a
    /// quasi-Newton matrix that has drifted ill-conditioned makes the
    /// step subproblem numerically unsolvable, and that is recoverable
    /// — far better than aborting an otherwise healthy solve.
    /// Off-diagonals are zeroed; the diagonal keeps the problem's scale.
    pub fn reset_to_scale(&mut self) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..self.n {
            let d = self.get(i, i);
            if d.is_finite() && d > 0.0 {
                sum += d;
                count += 1;
            }
        }
        let gamma = if count > 0 {
            sum / count as Number
        } else {
            1.0
        };
        let gamma = if gamma.is_finite() && gamma > 0.0 {
            gamma
        } else {
            1.0
        };
        for v in self.b.iter_mut() {
            *v = 0.0;
        }
        for i in 0..self.n {
            self.set(i, i, gamma);
        }
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.n && j < self.n);
        let (lo, hi) = if i >= j { (j, i) } else { (i, j) };
        hi * (hi + 1) / 2 + lo
    }

    fn get(&self, i: usize, j: usize) -> Number {
        self.b[self.idx(i, j)]
    }

    fn set(&mut self, i: usize, j: usize, v: Number) {
        let k = self.idx(i, j);
        self.b[k] = v;
    }

    /// Apply the Powell-damped BFGS update from the previous
    /// `(x_old, ∇L_old)` to the supplied `(x_new, ∇L_new)`. The
    /// first call just stores the pair; subsequent calls also
    /// modify `B`.
    pub fn update(&mut self, x_new: &[Number], grad_lag_new: &[Number]) {
        // Hard assert (PR #50 review S5): a length mismatch here
        // would silently mis-compute the rank-2 update in release
        // builds with debug_assert.
        assert_eq!(x_new.len(), self.n, "BFGS::update: x_new.len() != n");
        assert_eq!(
            grad_lag_new.len(),
            self.n,
            "BFGS::update: grad_lag_new.len() != n"
        );

        if let (Some(prev_x), Some(prev_grad_lag)) = (self.prev_x.take(), self.prev_grad_lag.take())
        {
            let s: Vec<Number> = x_new
                .iter()
                .zip(prev_x.iter())
                .map(|(a, b)| a - b)
                .collect();
            let y: Vec<Number> = grad_lag_new
                .iter()
                .zip(prev_grad_lag.iter())
                .map(|(a, b)| a - b)
                .collect();
            self.update_sy(&s, &y);
        }

        self.prev_x = Some(x_new.to_vec());
        self.prev_grad_lag = Some(grad_lag_new.to_vec());
    }

    /// Apply the Powell-damped rank-2 update from an explicit curvature
    /// pair `(s, y)`.
    ///
    /// Prefer this over [`Self::update`] when the caller can form `y`
    /// itself: the SQP driver must difference `∇L` at a **single, fixed**
    /// multiplier (see the note in `sqp_alg.rs`), which the `(x, ∇L)` form
    /// of [`Self::update`] cannot express because it stores the previous
    /// `∇L` as evaluated at the previous multiplier.
    pub fn update_sy(&mut self, s: &[Number], y: &[Number]) {
        self.update_sy_at(s, y, None);
    }

    /// [`Self::update_sy`], told the constraint multipliers `y` was formed
    /// at. Pass them only when the constraints are nonlinear: they are what
    /// moves the Lagrangian's curvature, `∇²f + Σλᵢ∇²cᵢ`, and so what can make
    /// the sizing stale (see the re-sizing rule below). `None` never re-sizes.
    pub fn update_sy_at(&mut self, s: &[Number], y: &[Number], lambda_g: Option<&[Number]>) {
        assert_eq!(s.len(), self.n, "BFGS::update_sy: s.len() != n");
        assert_eq!(y.len(), self.n, "BFGS::update_sy: y.len() != n");
        {
            // One-time initial Hessian sizing (Nocedal-Wright §6.1). The
            // identity seed `B_0 = I` is a catastrophic scale on
            // ill-conditioned problems: when `‖∇²L‖ ≫ 1` the first QP
            // step, computed against `B = I`, overshoots the true Newton
            // step by a factor `~cond(∇²L)`, and the filter line search —
            // with an empty filter at the first iterate, where the
            // starting point is near-feasible so `θ_curr` is tiny — accepts
            // the objective-blowing step because it happens to drive the
            // (negligible) constraint violation to zero. The overshoot
            // corrupts the working set and the solve then diverges to
            // `‖x‖ ~ 1e4` before dying with
            // `Search_Direction_Becomes_Too_Small`, on easy convex QPs
            // (issue #358 tail: `cond(P) ≳ 1e3`). Before applying the very
            // first rank-2 update, rescale `B` from `I` to `γI` with the
            // Rayleigh-quotient curvature estimate `γ = sᵀy / sᵀs`, which
            // for a quadratic lies in `[λ_min(∇²L), λ_max(∇²L)]` — a
            // representative scale that keeps the first post-sizing step
            // in range. Applied *once* on the first curvature pair, not
            // every iteration: re-seeding each step would discard the
            // curvature the persistent damped update accumulates (that
            // re-seeding is exactly what makes the L-BFGS path oscillate).
            // Halves the ill-conditioned-QP failure rate on a broad sweep
            // and clears the #358 tail; a fully robust cure for extreme
            // conditioning (`cond ≳ 1e4`) still needs the exact-Hessian or
            // IPM path.
            if !self.sized {
                let s_y: Number = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let s_s: Number = s.iter().map(|v| v * v).sum();
                if s_y > 1e-30 && s_s > 1e-30 {
                    let gamma = s_y / s_s;
                    for i in 0..self.n {
                        self.set(i, i, gamma);
                    }
                }
                // Mark sized once a genuine pair is seen, even if the
                // ratio was degenerate (leave B = I in that rare case) —
                // we only ever size on the first curvature pair.
                self.sized = true;
                self.sized_lambda = lambda_g.map(<[Number]>::to_vec);
            }

            // bs = B · s
            let mut bs: Vec<Number> = (0..self.n)
                .map(|i| (0..self.n).map(|j| self.get(i, j) * s[j]).sum())
                .collect();

            let mut s_bs: Number = s.iter().zip(bs.iter()).map(|(a, b)| a * b).sum();
            let s_y: Number = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

            // Re-sizing. The one-time sizing above is a Rayleigh quotient of
            // the Lagrangian's curvature at the multipliers of the FIRST
            // pair, and a damped BFGS update corrects `B` only along the
            // steps it is shown: every direction no step has explored keeps
            // that first scale. When the curvature moves with the multipliers
            // the scale goes stale, and the unexplored directions overshoot
            // by the ratio. `gilbert` (CUTE, n = 1000, gh#1191): sized at
            // `γ = 0.81` before the sphere's multiplier settled at 17.7, the
            // true curvature is ≈ 18 in every direction, `B`'s smallest
            // diagonal was still ≈ 1 at iteration 200, and 45 of the 204 pairs
            // of a run on #1190's head showed `sᵀy > 10·sᵀBs`. Stationarity
            // bounced between 1e-6 and 1e-4 for 50 iterations, and it
            // certified at 197 (`main`) to 204 (#1190's head) against 25 for
            // L-BFGS (which re-scales from every pair) and 33 for the exact
            // Hessian.
            //
            // So when the multipliers have moved since the sizing (by more
            // than `RESIZE_LAMBDA_MOVE` of their size then), a pair showing
            // `B` short of the curvature along `s` by more than
            // `RESIZE_RATIO` shifts `B` by `δI`, `δ = (sᵀy − sᵀBs)/sᵀs`, the
            // smallest isotropic shift that makes `B` agree with the pair along
            // `s`, and the update then proceeds as usual. A shift only ever
            // raises curvature, which is what the trigger says is missing.
            // Restarting from `γI` instead (`γ = sᵀy/sᵀs`) also discarded
            // what `B` had learned: on `eigenb2` (n = 110, 55 nonlinear rows)
            // it fired after a wildly backtracked early step, the restart's
            // `γ` was far too small in most directions, the following steps
            // ran to `‖p‖ ~ 1e5` at `α ~ 1e-7`, and the solve stopped at
            // `f = 7.52` after 11 iterations instead of reaching 1.6. Both
            // conditions are needed. An ill-conditioned problem shows
            // `sᵀy ≫ sᵀBs` along fresh directions with nothing stale about
            // the scale: on `palmer8e` (bounds only) the ratio test alone threw
            // away the curvature its fit needs and the solve ran out of
            // iterations at `f = 0.067`. Without nonlinear constraints the
            // multipliers do not enter `∇²L` at all, and nothing re-sizes.
            let stale = match (lambda_g, self.sized_lambda.as_deref()) {
                (Some(l), Some(l0)) if l.len() == l0.len() => {
                    let base = l0.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
                    let moved = l
                        .iter()
                        .zip(l0)
                        .fold(0.0_f64, |a, (u, v)| a.max((u - v).abs()));
                    moved > RESIZE_LAMBDA_MOVE * base
                }
                _ => false,
            };
            if stale && s_bs > 1e-14 && s_y > RESIZE_RATIO * s_bs {
                let s_s: Number = s.iter().map(|v| v * v).sum();
                let delta = (s_y - s_bs) / s_s;
                self.sized_lambda = lambda_g.map(<[Number]>::to_vec);
                for i in 0..self.n {
                    self.set(i, i, self.get(i, i) + delta);
                }
                for (b, v) in bs.iter_mut().zip(s) {
                    *b += delta * v;
                }
                s_bs = s_y;
            }

            // Powell damping.
            let theta = if s_y >= 0.2 * s_bs {
                1.0
            } else if s_bs - s_y > 1e-14 {
                0.8 * s_bs / (s_bs - s_y)
            } else {
                // Pathological — fall back to the unmodified
                // identity update (no harm done).
                1.0
            };
            let y_damp: Vec<Number> = y
                .iter()
                .zip(bs.iter())
                .map(|(yi, bsi)| theta * yi + (1.0 - theta) * bsi)
                .collect();
            let s_y_damp: Number = s.iter().zip(y_damp.iter()).map(|(a, b)| a * b).sum();

            if s_bs > 1e-14 && s_y_damp > 1e-14 {
                for i in 0..self.n {
                    for j in 0..=i {
                        let new_val = self.get(i, j) - (bs[i] * bs[j]) / s_bs
                            + (y_damp[i] * y_damp[j]) / s_y_damp;
                        self.set(i, j, new_val);
                    }
                }
            }
        }
    }

    /// Produce the current B as a `Triplet` over the upper
    /// triangle (1-based), ready to feed into `SqpQpData::build`.
    pub fn as_triplet(&self) -> Triplet {
        let mut vals = Vec::with_capacity(self.h_irow.len());
        for i in 0..self.n {
            for j in 0..=i {
                vals.push(self.get(i, j));
            }
        }
        Triplet {
            n_rows: self.n,
            n_cols: self.n,
            irow: self.h_irow.clone(),
            jcol: self.h_jcol.clone(),
            vals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(b: &DampedBfgs, i: usize) -> Number {
        b.get(i, i)
    }

    /// gh#1191: sized at curvature 1 along axis 0 with multiplier 0, then a
    /// pair along the unexplored axis 1 carrying curvature 20.
    fn stale_then_steep(lambda_now: Option<&[Number]>) -> DampedBfgs {
        let mut b = DampedBfgs::new(3);
        b.update_sy_at(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], Some(&[0.0]));
        b.update_sy_at(&[0.0, 1.0, 0.0], &[0.0, 20.0, 0.0], lambda_now);
        b
    }

    #[test]
    fn a_stale_scale_is_shifted_up_when_the_multipliers_have_moved() {
        // λ went 0 → 17: δ = (20 − 1)/1 = 19 lifts the untouched axis 2 to
        // the new scale.
        let b = stale_then_steep(Some(&[17.0]));
        assert!(
            (diag(&b, 2) - 20.0).abs() < 1e-12,
            "axis 2: {}",
            diag(&b, 2)
        );
        assert!(
            (diag(&b, 1) - 20.0).abs() < 1e-12,
            "axis 1: {}",
            diag(&b, 1)
        );
    }

    #[test]
    fn the_shift_keeps_curvature_already_learned() {
        // `eigenb2` (gh#1191 review): restarting from `γI` threw away the
        // large curvature `B` had learned. Axis 2 learns 1e4 while the
        // multipliers are still near the sizing's; then a stale pair along
        // axis 1 fires. A shift keeps axis 2 at ≥ 1e4, where a restart at
        // `γ = 20` would have dropped it to 20.
        let mut b = DampedBfgs::new(3);
        b.update_sy_at(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], Some(&[0.0]));
        b.update_sy_at(&[0.0, 0.0, 1.0], &[0.0, 0.0, 1e4], Some(&[0.2]));
        assert!((diag(&b, 2) - 1e4).abs() < 1e-8, "learned: {}", diag(&b, 2));
        b.update_sy_at(&[0.0, 1.0, 0.0], &[0.0, 20.0, 0.0], Some(&[17.0]));
        assert!(diag(&b, 2) >= 1e4, "axis 2 lowered to {}", diag(&b, 2));
        assert!((diag(&b, 1) - 20.0).abs() < 1e-8, "axis 1: {}", diag(&b, 1));
    }

    #[test]
    fn a_steep_pair_alone_does_not_resize() {
        // Same pair, multipliers unmoved: an ill-conditioned problem's fresh
        // direction, not a stale scale. Axis 2 keeps the first sizing.
        let b = stale_then_steep(Some(&[0.2]));
        assert!((diag(&b, 2) - 1.0).abs() < 1e-12, "axis 2: {}", diag(&b, 2));
        assert!(
            (diag(&b, 1) - 20.0).abs() < 1e-12,
            "axis 1: {}",
            diag(&b, 1)
        );
    }

    #[test]
    fn without_multipliers_nothing_resizes() {
        // Linear constraints or none: `∇²L = ∇²f`, so the caller passes no
        // multipliers and the sizing is never stale.
        let b = stale_then_steep(None);
        assert!((diag(&b, 2) - 1.0).abs() < 1e-12, "axis 2: {}", diag(&b, 2));
    }

    #[test]
    fn first_update_sizes_the_identity_seed() {
        // Curvature pair along axis 0 from the quadratic f = ½·9‖x‖²
        // (∇²f = 9I): s = (1, 0), y = 9 s = (9, 0), so the sizing
        // factor is γ = sᵀy / sᵀs = 9. After the first update the
        // *untouched* direction (axis 1) must carry the sized scale
        // γ = 9, not the identity seed 1 — that is the whole point of
        // sizing on an ill-conditioned problem (issue #358). Along the
        // updated axis the Powell-damped rank-2 term keeps it at 9 too.
        let mut b = DampedBfgs::new(2);
        b.update(&[0.0, 0.0], &[0.0, 0.0]); // record start (no pair yet)
        assert!((diag(&b, 0) - 1.0).abs() < 1e-12, "seed must be I");
        assert!((diag(&b, 1) - 1.0).abs() < 1e-12, "seed must be I");
        b.update(&[1.0, 0.0], &[9.0, 0.0]); // first genuine (s, y): sizes then updates
        assert!(
            (diag(&b, 1) - 9.0).abs() < 1e-9,
            "off-axis diagonal should be sized to γ = 9, got {}",
            diag(&b, 1)
        );
        assert!(
            (diag(&b, 0) - 9.0).abs() < 1e-9,
            "on-axis diagonal should be 9 after sizing + rank-2 update, got {}",
            diag(&b, 0)
        );
    }

    #[test]
    fn seed_scale_sets_the_diagonal_and_marks_sized() {
        let mut b = DampedBfgs::new(3);
        b.seed_scale(25.0);
        assert!(b.sized, "seeding must suppress the later one-time sizing");
        for i in 0..3 {
            assert!((diag(&b, i) - 25.0).abs() < 1e-12);
        }
        // A degenerate scale must be ignored, not written into B.
        let mut c = DampedBfgs::new(2);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            c.seed_scale(bad);
            assert!(!c.sized, "seed_scale({bad}) must be refused");
            assert!((diag(&c, 0) - 1.0).abs() < 1e-12, "B must stay I");
        }
    }

    #[test]
    fn reset_to_scale_drops_curvature_but_keeps_magnitude() {
        // Build up genuine off-diagonal curvature, then reset: the
        // off-diagonals must vanish and the diagonal must retain the
        // matrix's own scale (its mean diagonal), not collapse to 1.
        let mut b = DampedBfgs::new(2);
        b.seed_scale(100.0);
        b.update(&[0.0, 0.0], &[0.0, 0.0]);
        b.update(&[1.0, 1.0], &[150.0, 40.0]); // rank-2 update -> off-diagonals
        assert!(
            b.get(1, 0).abs() > 1e-9,
            "test precondition: expected off-diagonal curvature, got {}",
            b.get(1, 0)
        );
        let mean_diag = (diag(&b, 0) + diag(&b, 1)) / 2.0;
        b.reset_to_scale();
        assert!(b.get(1, 0).abs() < 1e-12, "off-diagonals must be zeroed");
        for i in 0..2 {
            assert!(
                (diag(&b, i) - mean_diag).abs() < 1e-9,
                "diagonal must keep the mean scale {mean_diag}, got {}",
                diag(&b, i)
            );
        }
        assert!(
            mean_diag > 10.0,
            "sanity: the retained scale should reflect the problem, not 1"
        );
    }

    #[test]
    fn update_sy_matches_the_x_grad_lag_form() {
        // `update_sy` is the primitive; `update` is the (s, y)-from-stored-
        // prev convenience wrapper. Feeding the same curvature pair through
        // either path must land on the identical matrix, so the driver's
        // switch to `update_sy` (gh #361) changes only *which* y is formed,
        // never how it is applied.
        let mut via_update = DampedBfgs::new(2);
        via_update.update(&[0.0, 0.0], &[1.0, 2.0]);
        via_update.update(&[1.0, 3.0], &[4.0, 9.0]);

        let mut via_sy = DampedBfgs::new(2);
        via_sy.update_sy(&[1.0, 3.0], &[3.0, 7.0]); // s = x1-x0, y = g1-g0

        for i in 0..2 {
            for j in 0..=i {
                assert!(
                    (via_update.get(i, j) - via_sy.get(i, j)).abs() < 1e-12,
                    "B[{i},{j}]: update={} update_sy={}",
                    via_update.get(i, j),
                    via_sy.get(i, j)
                );
            }
        }
    }

    #[test]
    fn sizing_happens_only_once() {
        // A second pair must NOT re-seed the diagonal; the persistent
        // rank-2 updates accumulate on top of the one-time sized base.
        let mut b = DampedBfgs::new(2);
        b.update(&[0.0, 0.0], &[0.0, 0.0]);
        b.update(&[1.0, 0.0], &[9.0, 0.0]); // sizes to γ = 9
        assert!(b.sized);
        // A second pair along axis 1 with a *different* curvature (4):
        // γ would have been 4 had we re-sized, but we must not.
        b.update(&[1.0, 1.0], &[9.0, 4.0]); // s = (0,1), y = (0,4)
        // Axis-0 diagonal is untouched by this axis-1 pair and must
        // still reflect the first sizing (9), never re-seeded to 4.
        assert!(
            (diag(&b, 0) - 9.0).abs() < 1e-9,
            "second pair must not re-seed; axis-0 diagonal = {}",
            diag(&b, 0)
        );
    }
}
