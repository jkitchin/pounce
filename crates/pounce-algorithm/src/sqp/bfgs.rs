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

pub struct DampedBfgs {
    n: usize,
    /// Lower-triangle row-major storage:
    /// `b[i*(i+1)/2 + j] = B[i, j]` for `i ≥ j`.
    b: Vec<Number>,
    /// Previous `x` and ∇L; updated at the end of each `update` call.
    prev_x: Option<Vec<Number>>,
    prev_grad_lag: Option<Vec<Number>>,
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
            h_irow,
            h_jcol,
        }
    }

    /// Have we recorded a previous `(x, ∇L)`? `false` until the
    /// first call to [`Self::update`].
    pub fn has_prev(&self) -> bool {
        self.prev_x.is_some()
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

            // bs = B · s
            let bs: Vec<Number> = (0..self.n)
                .map(|i| (0..self.n).map(|j| self.get(i, j) * s[j]).sum())
                .collect();

            let s_bs: Number = s.iter().zip(bs.iter()).map(|(a, b)| a * b).sum();
            let s_y: Number = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();

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

        self.prev_x = Some(x_new.to_vec());
        self.prev_grad_lag = Some(grad_lag_new.to_vec());
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
