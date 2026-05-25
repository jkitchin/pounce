// Indexed loops mirror the tridiagonal Q structure directly.
#![allow(clippy::needless_range_loop, clippy::neg_multiply)]

//! Problem 5: Sparse QP with three-term-sum inequalities and box bounds.
//!
//! ```text
//!   min ½ xᵀ Q x - Σ x_i
//!   s.t. x_j + x_{(j+1) mod n} + x_{(j+2) mod n} ≤ 2.5,   j = 0..n-1
//!        0 ≤ x_i ≤ 10
//! ```
//! `Q` is tridiagonal (`4` on the diagonal, `-1` off-diagonal), SPD.

use crate::problems::FinalState;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

pub struct SparseQP {
    pub n: usize,
    pub final_state: FinalState,
}

impl SparseQP {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            final_state: FinalState::new(),
        }
    }
}

impl TNLP for SparseQP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n as Index;
        Some(NlpInfo {
            n,
            m: n,
            nnz_jac_g: 3 * n,
            nnz_h_lag: 2 * n - 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = 0.0;
        }
        for v in b.x_u.iter_mut() {
            *v = 10.0;
        }
        for v in b.g_l.iter_mut() {
            *v = f64::NEG_INFINITY;
        }
        for v in b.g_u.iter_mut() {
            *v = 2.5;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            for v in sp.x.iter_mut() {
                *v = 0.5;
            }
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let n = self.n;
        let mut f = 0.0;
        for i in 0..n {
            f += 0.5 * 4.0 * x[i] * x[i];
            if i < n - 1 {
                // Both (i,i+1) and (i+1,i) → -1 * x_i * x_{i+1} * 2 * 0.5
                f += 0.5 * (-1.0) * x[i] * x[i + 1] * 2.0;
            }
        }
        for i in 0..n {
            f -= x[i];
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        let n = self.n;
        for i in 0..n {
            grad[i] = 4.0 * x[i] - 1.0;
            if i > 0 {
                grad[i] -= x[i - 1];
            }
            if i < n - 1 {
                grad[i] -= x[i + 1];
            }
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let n = self.n;
        for j in 0..n {
            g[j] = x[j] + x[(j + 1) % n] + x[(j + 2) % n];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..n {
                    let base = 3 * j;
                    let jj = j as Index;
                    irow[base] = jj;
                    jcol[base] = j as Index;
                    irow[base + 1] = jj;
                    jcol[base + 1] = ((j + 1) % n) as Index;
                    irow[base + 2] = jj;
                    jcol[base + 2] = ((j + 2) % n) as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                for j in 0..n {
                    let base = 3 * j;
                    values[base] = 1.0;
                    values[base + 1] = 1.0;
                    values[base + 2] = 1.0;
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                for i in 1..n {
                    let ii = i as Index;
                    irow[2 * i - 1] = ii;
                    jcol[2 * i - 1] = ii - 1;
                    irow[2 * i] = ii;
                    jcol[2 * i] = ii;
                }
                true
            }
            SparsityRequest::Values { values } => {
                // Constraints are linear → no contribution from lambda.
                values[0] = obj_factor * 4.0;
                for i in 1..n {
                    values[2 * i - 1] = obj_factor * (-1.0);
                    values[2 * i] = obj_factor * 4.0;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_state.capture(sol);
    }
}
