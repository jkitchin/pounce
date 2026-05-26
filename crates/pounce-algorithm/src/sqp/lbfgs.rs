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
    /// 1-based upper-triangle sparsity pattern (matches DampedBfgs).
    h_irow: Vec<Index>,
    h_jcol: Vec<Index>,
}

impl LBfgs {
    /// `m_history` is the number of `(s, y)` pairs to keep
    /// (Nocedal-Wright recommends 3–20; upstream IPOPT defaults to
    /// 6). Must be ≥ 1; a value of 0 would degenerate to plain
    /// identity which is rarely useful.
    pub fn new(n: usize, m_history: usize) -> Self {
        debug_assert!(m_history >= 1);
        let nz = n * (n + 1) / 2;
        let mut h_irow = Vec::with_capacity(nz);
        let mut h_jcol = Vec::with_capacity(nz);
        for i in 0..n {
            for j in 0..=i {
                h_irow.push((i + 1) as Index);
                h_jcol.push((j + 1) as Index);
            }
        }
        Self {
            n,
            m_history,
            pairs: VecDeque::with_capacity(m_history),
            prev_x: None,
            prev_grad_lag: None,
            h_irow,
            h_jcol,
        }
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
            // Skip degenerate pairs (s ≈ 0).
            let s_norm2: Number = s.iter().map(|v| v * v).sum();
            if s_norm2 > 1e-30 {
                if self.pairs.len() == self.m_history {
                    self.pairs.pop_front();
                }
                self.pairs.push_back((s, y));
            }
        }

        self.prev_x = Some(x_new.to_vec());
        self.prev_grad_lag = Some(grad_lag_new.to_vec());
    }

    /// Materialize the current B_k as a dense `Triplet` over the
    /// upper triangle. Always returns the full `n(n+1)/2` triplets
    /// (the same fixed pattern across iterations, so the QP
    /// solver's symbolic factorization stays valid).
    pub fn as_triplet(&self) -> Triplet {
        let b_dense = self.materialize();
        let mut vals = Vec::with_capacity(self.h_irow.len());
        for i in 0..self.n {
            for j in 0..=i {
                vals.push(b_dense[i * self.n + j]);
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

    /// Build the dense `n×n` row-major B_k by seeding `B_0 = γI`
    /// (Nocedal-Wright eq. 7.20) and replaying Powell-damped BFGS
    /// updates for every stored pair. Returned values are
    /// symmetric (only the lower triangle is read by `as_triplet`).
    fn materialize(&self) -> Vec<Number> {
        let n = self.n;
        // Initial scaling γ: most-recent (s, y) ratio. Defaults to
        // 1.0 when no pairs exist or sᵀy is tiny.
        let gamma = self
            .pairs
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
            .unwrap_or(1.0);
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

            let theta = if s_y >= 0.2 * s_bs {
                1.0
            } else if s_bs - s_y > 1e-14 {
                0.8 * s_bs / (s_bs - s_y)
            } else {
                1.0
            };
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
}
