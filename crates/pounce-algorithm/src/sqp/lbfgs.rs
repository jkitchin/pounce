//! Limited-memory Powell-damped BFGS Hessian approximation for
//! SQP (Nocedal-Wright §7.2; Byrd-Nocedal-Schnabel 1994; Powell
//! 1978). Used when `SqpOptions::hessian = Lbfgs`.
//!
//! Maintains a fixed-length circular buffer of `(s, y)` pairs and,
//! at each `as_triplet` query, reconstructs the dense Hessian B_k
//! by replaying the rank-2 BFGS updates from a scaled-identity
//! seed `B_0 = γ_k I` with `γ_k = (s_{last}·y_{last}) /
//! (y_{last}·y_{last})` (Nocedal-Wright eq. 7.20 — the standard
//! initial-scaling heuristic).
//!
//! Phase 5b commit 15 implements the dense materialization path
//! (output is a full upper-triangular [`Triplet`] consumed by
//! `pounce-qp`'s `SqpQpData`). A future commit can add a
//! matrix-free product interface so the QP can avoid the
//! `O(n²)` storage when `m_history ≪ n`.
//!
//! Powell damping is identical to [`crate::sqp::bfgs::DampedBfgs`]:
//!
//! ```text
//!     if s·y ≥ 0.2 · s·B s :  θ = 1
//!     else                  :  θ = 0.8 · s·B s / (s·B s − s·y)
//!     y_damp = θ y + (1 − θ) B s
//! ```
//!
//! …but `B` here is the rebuilt one, accumulated within
//! `materialize`, so the damping factor for each historical pair
//! is computed at replay time against the running B (matching
//! how a full BFGS would have evolved).

use crate::sqp::qp_assembly::Triplet;
use pounce_common::types::{Index, Number};
use std::collections::VecDeque;

/// Limited-memory Powell-damped BFGS, storing up to `m_history`
/// most-recent `(s, y)` pairs.
pub struct LBfgs {
    n: usize,
    m_history: usize,
    /// Circular buffer of (s_k, y_k) pairs, oldest first.
    pairs: VecDeque<(Vec<Number>, Vec<Number>)>,
    /// Previous `x` and `∇L`; the next call to [`Self::update`]
    /// builds the next `(s, y)` from these.
    prev_x: Option<Vec<Number>>,
    prev_grad_lag: Option<Vec<Number>>,
    /// Seed scale for `B_0 = γI` while no pair is stored, set by the SQP
    /// driver's iteration-0 curvature probe ([`Self::seed_scale`]).
    seed_gamma: Option<Number>,
}

/// `B = σI + Σᵢ signᵢ vᵢ vᵢᵀ` — the L-BFGS matrix in product form, with no
/// `n × n` storage (gh#1054).
///
/// Each stored pair contributes one `−a aᵀ` and one `+b bᵀ` term,
/// `a = B s / √(sᵀB s)` and `b = ȳ / √(sᵀȳ)` with `ȳ` the Powell-damped
/// `y` — the same rank-2 update [`LBfgs::as_triplet`] replays densely, so
/// the two describe one matrix. A pair the dense replay skips (`sᵀBs` or
/// `sᵀȳ` below `1e-14`) is skipped here too.
#[derive(Debug, Clone)]
pub struct LowRankHessian {
    pub sigma: Number,
    /// The `vᵢ`, each of length `n`.
    pub vecs: Vec<Vec<Number>>,
    /// `−1` or `+1` per vector.
    pub signs: Vec<Number>,
}

impl LowRankHessian {
    /// `B v`, in `O(k n)`.
    pub fn apply(&self, v: &[Number]) -> Vec<Number> {
        let mut out: Vec<Number> = v.iter().map(|x| self.sigma * x).collect();
        for (w, &sg) in self.vecs.iter().zip(self.signs.iter()) {
            let c = sg * w.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<Number>();
            for (o, wi) in out.iter_mut().zip(w.iter()) {
                *o += c * wi;
            }
        }
        out
    }

    /// `max |B_ij|`. `B` is positive definite (BFGS with Powell damping),
    /// and for a PSD matrix the largest entry in magnitude sits on the
    /// diagonal (`|B_ij| ≤ √(B_ii B_jj)`), so this is the dense
    /// matrix's max-norm exactly, at `O(k n)`.
    pub fn max_abs_entry(&self) -> Number {
        let n = self.vecs.first().map_or(0, Vec::len);
        let mut diag = vec![self.sigma; n];
        for (w, &sg) in self.vecs.iter().zip(self.signs.iter()) {
            for (d, wi) in diag.iter_mut().zip(w.iter()) {
                *d += sg * wi * wi;
            }
        }
        diag.iter()
            .fold(self.sigma.abs(), |acc, d| acc.max(d.abs()))
    }
}

impl LBfgs {
    /// `m_history` is the number of `(s, y)` pairs to keep
    /// (Nocedal-Wright recommends 3–20; upstream IPOPT defaults to
    /// 6). Must be ≥ 1; a value of 0 would degenerate to plain
    /// identity which is rarely useful.
    pub fn new(n: usize, m_history: usize) -> Self {
        debug_assert!(m_history >= 1);
        Self {
            n,
            m_history,
            pairs: VecDeque::with_capacity(m_history),
            prev_x: None,
            prev_grad_lag: None,
            seed_gamma: None,
        }
    }

    /// Seed `B_0 = γI` from a curvature estimate taken before the first
    /// pair exists — the SQP driver's iteration-0 probe, the same one
    /// [`crate::sqp::bfgs::DampedBfgs::seed_scale`] takes (issue #358).
    /// Without it the first QP solves against `B = I` whatever the
    /// problem's scale. A non-positive or non-finite `gamma` is ignored.
    /// Once a pair is stored the usual `yᵀy / sᵀy` scaling takes over.
    pub fn seed_scale(&mut self, gamma: Number) {
        if gamma.is_finite() && gamma > 0.0 {
            self.seed_gamma = Some(gamma);
        }
    }

    /// `B_0`'s scale: `yᵀy / sᵀy` of the newest pair (Nocedal-Wright
    /// eq. 7.20), else the probe's seed, else `1`.
    fn gamma(&self) -> Number {
        self.pairs
            .back()
            .and_then(|(s, y)| {
                let sy: Number = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
                let yy: Number = y.iter().map(|v| v * v).sum();
                if yy > 1e-30 && sy > 1e-30 {
                    Some(yy / sy)
                } else {
                    None
                }
            })
            .or(self.seed_gamma)
            .unwrap_or(1.0)
    }

    /// The current `B` in product form (see [`LowRankHessian`]). Same
    /// damping, same skip thresholds, same `γ` as [`Self::as_triplet`],
    /// at `O(k² n)` time and `O(k n)` storage instead of `O(k n²)` and
    /// `O(n²)`.
    pub fn low_rank(&self) -> LowRankHessian {
        let mut b = LowRankHessian {
            sigma: self.gamma(),
            vecs: Vec::with_capacity(2 * self.pairs.len()),
            signs: Vec::with_capacity(2 * self.pairs.len()),
        };
        for (s, y) in self.pairs.iter() {
            let bs = b.apply(s);
            let s_bs: Number = s.iter().zip(bs.iter()).map(|(a, c)| a * c).sum();
            let s_y: Number = s.iter().zip(y.iter()).map(|(a, c)| a * c).sum();
            let theta = powell_theta(s_bs, s_y);
            let y_damp: Vec<Number> = y
                .iter()
                .zip(bs.iter())
                .map(|(yi, bsi)| theta * yi + (1.0 - theta) * bsi)
                .collect();
            let s_y_damp: Number = s.iter().zip(y_damp.iter()).map(|(a, c)| a * c).sum();
            if s_bs > 1e-14 && s_y_damp > 1e-14 {
                let ra = 1.0 / s_bs.sqrt();
                let rb = 1.0 / s_y_damp.sqrt();
                b.vecs.push(bs.iter().map(|v| v * ra).collect());
                b.signs.push(-1.0);
                b.vecs.push(y_damp.iter().map(|v| v * rb).collect());
                b.signs.push(1.0);
            }
        }
        b
    }

    pub fn has_prev(&self) -> bool {
        self.prev_x.is_some()
    }

    /// Record a new pair `(s, y) = (x_new − x_prev, ∇L_new −
    /// ∇L_prev)` if a prior iterate exists. The first call merely
    /// stores `(x_new, ∇L_new)`.
    pub fn update(&mut self, x_new: &[Number], grad_lag_new: &[Number]) {
        // Hard assert (PR #50 review S5): see BFGS::update.
        assert_eq!(x_new.len(), self.n, "LBFGS::update: x_new.len() != n");
        assert_eq!(
            grad_lag_new.len(),
            self.n,
            "LBFGS::update: grad_lag_new.len() != n"
        );

        if let (Some(prev_x), Some(prev_grad_lag)) =
            (self.prev_x.as_ref(), self.prev_grad_lag.as_ref())
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

    /// Record an explicit curvature pair `(s, y)`.
    ///
    /// Prefer this over [`Self::update`] when the caller can form `y`
    /// itself: the SQP driver must difference `∇L` at a **single, fixed**
    /// multiplier (see `sqp_alg::curvature_pair`), which the `(x, ∇L)`
    /// form cannot express because it stores the previous `∇L` as
    /// evaluated at the previous multiplier.
    pub fn update_sy(&mut self, s: &[Number], y: &[Number]) {
        assert_eq!(s.len(), self.n, "LBFGS::update_sy: s.len() != n");
        assert_eq!(y.len(), self.n, "LBFGS::update_sy: y.len() != n");
        // Skip degenerate pairs (s ≈ 0).
        let s_norm2: Number = s.iter().map(|v| v * v).sum();
        if s_norm2 > 1e-30 {
            if self.pairs.len() == self.m_history {
                self.pairs.pop_front();
            }
            self.pairs.push_back((s.to_vec(), y.to_vec()));
        }
    }

    /// Materialize the current B_k as a dense `Triplet` over the
    /// upper triangle. Always returns the full `n(n+1)/2` triplets
    /// (the same fixed pattern across iterations, so the QP
    /// solver's symbolic factorization stays valid).
    pub fn as_triplet(&self) -> Triplet {
        let n = self.n;
        let nz = n * (n + 1) / 2;
        let (mut irow, mut jcol) = (Vec::with_capacity(nz), Vec::with_capacity(nz));
        for i in 0..n {
            for j in 0..=i {
                irow.push((i + 1) as Index);
                jcol.push((j + 1) as Index);
            }
        }
        Triplet {
            n_rows: n,
            n_cols: n,
            irow,
            jcol,
            vals: lower_triangle(&self.materialize(), n),
        }
    }

    /// Build the dense `n×n` row-major B_k by seeding `B_0 = γI`
    /// (Nocedal-Wright eq. 7.20) and replaying Powell-damped BFGS
    /// updates for every stored pair. Returned values are
    /// symmetric (only the lower triangle is read by `as_triplet`).
    fn materialize(&self) -> Vec<Number> {
        let n = self.n;
        // Initial scaling γ: most-recent (s, y) ratio, else the probe's
        // seed, else 1.0.
        let gamma = self.gamma();
        let mut b = vec![0.0_f64; n * n];
        for i in 0..n {
            b[i * n + i] = gamma;
        }

        for (s, y) in self.pairs.iter() {
            // bs = B · s
            let mut bs = vec![0.0_f64; n];
            for i in 0..n {
                let mut acc = 0.0_f64;
                let row = &b[i * n..i * n + n];
                for j in 0..n {
                    acc += row[j] * s[j];
                }
                bs[i] = acc;
            }
            let s_bs: Number = s.iter().zip(bs.iter()).map(|(a, b)| a * b).sum();
            let s_y: Number = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

            let theta = powell_theta(s_bs, s_y);
            let y_damp: Vec<Number> = y
                .iter()
                .zip(bs.iter())
                .map(|(yi, bsi)| theta * yi + (1.0 - theta) * bsi)
                .collect();
            let s_y_damp: Number = s.iter().zip(y_damp.iter()).map(|(a, b)| a * b).sum();

            if s_bs > 1e-14 && s_y_damp > 1e-14 {
                for i in 0..n {
                    for j in 0..n {
                        let delta = -(bs[i] * bs[j]) / s_bs + (y_damp[i] * y_damp[j]) / s_y_damp;
                        b[i * n + j] += delta;
                    }
                }
            }
        }
        b
    }
}

/// Row-major dense `n × n` → its lower triangle, row by row.
fn lower_triangle(b: &[Number], n: usize) -> Vec<Number> {
    let mut vals = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            vals.push(b[i * n + j]);
        }
    }
    vals
}

/// Powell's damping factor θ for a pair with curvatures `sᵀBs`, `sᵀy`.
fn powell_theta(s_bs: Number, s_y: Number) -> Number {
    if s_y >= 0.2 * s_bs {
        1.0
    } else if s_bs - s_y > 1e-14 {
        0.8 * s_bs / (s_bs - s_y)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgs_seeds_identity_with_no_pairs() {
        let lb = LBfgs::new(3, 5);
        let t = lb.as_triplet();
        // Diagonal should be 1.0, off-diagonals 0.0.
        for k in 0..t.vals.len() {
            let i = (t.irow[k] - 1) as usize;
            let j = (t.jcol[k] - 1) as usize;
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (t.vals[k] - expected).abs() < 1e-15,
                "B[{i},{j}] = {} but expected {expected}",
                t.vals[k]
            );
        }
    }

    #[test]
    fn lbfgs_first_update_only_records_pair() {
        let mut lb = LBfgs::new(2, 3);
        lb.update(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(lb.has_prev());
        assert!(lb.pairs.is_empty());
        // Without any pairs the matrix is still γI = I.
        let t = lb.as_triplet();
        let diag: Vec<_> = t
            .vals
            .iter()
            .enumerate()
            .filter(|(k, _)| t.irow[*k] == t.jcol[*k])
            .map(|(_, v)| *v)
            .collect();
        assert!((diag[0] - 1.0).abs() < 1e-15);
        assert!((diag[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn lbfgs_quadratic_recovers_exact_hessian_at_convergence() {
        // For the quadratic f(x) = ½ xᵀ A x with A = diag(2, 4),
        // ∇f = Ax, so along iterates x_k = x_{k-1} + s_{k-1}:
        // y_{k-1} = A s_{k-1}. Pumping ≥ 2 linearly independent
        // (s, y) pairs into L-BFGS must rebuild B_2 = A exactly
        // (up to numerical roundoff) because the rank-2 corrections
        // collapse onto A in 2-D.
        let mut lb = LBfgs::new(2, 5);
        // Pair 1: s = (1, 0), y = A·s = (2, 0).
        lb.update(&[0.0, 0.0], &[0.0, 0.0]); // record start
        lb.update(&[1.0, 0.0], &[2.0, 0.0]); // produces (s, y) #1
        lb.update(&[1.0, 1.0], &[2.0, 4.0]); // produces (s, y) #2 with s=(0,1), y=(0,4)
        let t = lb.as_triplet();
        // B should equal A = diag(2, 4).
        let mut b = [[0.0_f64; 2]; 2];
        for k in 0..t.vals.len() {
            let i = (t.irow[k] - 1) as usize;
            let j = (t.jcol[k] - 1) as usize;
            b[i][j] = t.vals[k];
            if i != j {
                b[j][i] = t.vals[k];
            }
        }
        assert!((b[0][0] - 2.0).abs() < 1e-9, "B[0,0] = {}", b[0][0]);
        assert!((b[1][1] - 4.0).abs() < 1e-9, "B[1,1] = {}", b[1][1]);
        assert!(b[0][1].abs() < 1e-9, "B[0,1] = {}", b[0][1]);
    }

    #[test]
    fn lbfgs_history_cap_drops_oldest() {
        let mut lb = LBfgs::new(2, 2);
        lb.update(&[0.0, 0.0], &[0.0, 0.0]);
        lb.update(&[1.0, 0.0], &[1.0, 0.0]);
        lb.update(&[2.0, 0.0], &[2.0, 0.0]);
        lb.update(&[2.0, 1.0], &[2.0, 1.0]);
        // m_history = 2 means only the most recent 2 pairs are
        // retained.
        assert_eq!(lb.pairs.len(), 2);
    }

    /// gh#1054: the product form is the dense replay, term for term —
    /// including a pair Powell damping rewrites (`sᵀy < 0.2 sᵀBs`) and a
    /// seeded `γ`.
    #[test]
    fn low_rank_matches_the_dense_matrix() {
        let n = 5;
        let mut lb = LBfgs::new(n, 3);
        lb.seed_scale(2.5);
        let pairs: [([f64; 5], [f64; 5]); 4] = [
            ([1.0, 0.0, 0.5, 0.0, 0.1], [3.0, 0.2, 1.0, 0.0, 0.4]),
            ([0.0, 1.0, 0.0, -0.3, 0.0], [0.1, 2.0, 0.0, -0.5, 0.2]),
            // Negative curvature along s: damping must engage.
            ([0.2, 0.2, 1.0, 0.0, 0.0], [-0.1, 0.0, -0.2, 0.0, 0.0]),
            ([0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.3, 0.0, 4.0, 3.5]),
        ];
        for (s, y) in &pairs {
            lb.update_sy(s, y);
        }
        let dense = lb.materialize();
        let lr = lb.low_rank();
        assert_eq!(lr.vecs.len(), 6, "three stored pairs, two terms each");
        for k in 0..n {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            let col = lr.apply(&e);
            for i in 0..n {
                assert!(
                    (col[i] - dense[i * n + k]).abs() < 1e-12,
                    "B[{i},{k}]: product form {} vs dense {}",
                    col[i],
                    dense[i * n + k]
                );
            }
        }
        let max_dense = dense.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!((lr.max_abs_entry() - max_dense).abs() < 1e-12);
    }

    #[test]
    fn seed_scale_sets_the_initial_matrix_until_a_pair_arrives() {
        let mut lb = LBfgs::new(2, 3);
        lb.seed_scale(7.0);
        assert_eq!(lb.low_rank().sigma, 7.0);
        lb.seed_scale(-1.0); // ignored
        assert_eq!(lb.low_rank().sigma, 7.0);
        lb.update_sy(&[1.0, 0.0], &[3.0, 0.0]);
        assert_eq!(lb.low_rank().sigma, 3.0, "yᵀy / sᵀy of the newest pair");
    }
}
