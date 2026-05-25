//! Problem 4: 2-D Poisson optimal control on a K×K interior grid.
//!
//! ```text
//!   min ½ h² Σ (u_{ij} - u_d(x_i,y_j))² + (α/2) h² Σ f_{ij}²
//!   s.t. -Δ_h u_{ij} = f_{ij}      (5-point stencil, Dirichlet 0 on the boundary)
//! ```
//! Variables `[u_{0,0},...,u_{K-1,K-1}, f_{0,0},...,f_{K-1,K-1}]`;
//! `n = 2K²`, `m = K²`.

use crate::problems::FinalState;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::f64::consts::PI;

pub struct PoissonControl {
    k: usize,
    h: f64,
    alpha: f64,
    jac_nnz: usize,
    pub final_state: FinalState,
}

impl PoissonControl {
    pub fn new(k: usize) -> Self {
        let h = 1.0 / (k as f64 + 1.0);
        // Pre-compute the Jacobian nonzero count so `get_nlp_info` matches the
        // structure produced by `eval_jac_g` exactly.
        let mut jac_nnz = 0usize;
        for j in 0..k {
            for i in 0..k {
                // center + control = 2
                let mut row = 2usize;
                if i > 0 {
                    row += 1;
                }
                if i < k - 1 {
                    row += 1;
                }
                if j > 0 {
                    row += 1;
                }
                if j < k - 1 {
                    row += 1;
                }
                jac_nnz += row;
            }
        }
        Self {
            k,
            h,
            alpha: 0.01,
            jac_nnz,
            final_state: FinalState::new(),
        }
    }

    #[inline]
    fn idx_u(&self, i: usize, j: usize) -> usize {
        i + j * self.k
    }

    #[inline]
    fn idx_f(&self, i: usize, j: usize) -> usize {
        self.k * self.k + i + j * self.k
    }

    fn u_desired(&self, i: usize, j: usize) -> f64 {
        let x = (i as f64 + 1.0) * self.h;
        let y = (j as f64 + 1.0) * self.h;
        (PI * x).sin() * (PI * y).sin()
    }

    fn n(&self) -> usize {
        2 * self.k * self.k
    }
}

impl TNLP for PoissonControl {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n() as Index;
        let m = (self.k * self.k) as Index;
        Some(NlpInfo {
            n,
            m,
            nnz_jac_g: self.jac_nnz as Index,
            nnz_h_lag: n,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for v in b.x_l.iter_mut() {
            *v = f64::NEG_INFINITY;
        }
        for v in b.x_u.iter_mut() {
            *v = f64::INFINITY;
        }
        for v in b.g_l.iter_mut() {
            *v = 0.0;
        }
        for v in b.g_u.iter_mut() {
            *v = 0.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            for v in sp.x.iter_mut() {
                *v = 0.0;
            }
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let k = self.k;
        let h2 = self.h * self.h;
        let mut f = 0.0;
        for j in 0..k {
            for i in 0..k {
                let u = x[self.idx_u(i, j)];
                let ud = self.u_desired(i, j);
                f += 0.5 * h2 * (u - ud) * (u - ud);

                let fi = x[self.idx_f(i, j)];
                f += 0.5 * self.alpha * h2 * fi * fi;
            }
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        let k = self.k;
        let h2 = self.h * self.h;
        for v in grad.iter_mut() {
            *v = 0.0;
        }
        for j in 0..k {
            for i in 0..k {
                let u = x[self.idx_u(i, j)];
                let ud = self.u_desired(i, j);
                grad[self.idx_u(i, j)] = h2 * (u - ud);
                grad[self.idx_f(i, j)] = self.alpha * h2 * x[self.idx_f(i, j)];
            }
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let k = self.k;
        let h2 = self.h * self.h;
        for j in 0..k {
            for i in 0..k {
                let c = j * k + i;
                let center = x[self.idx_u(i, j)];
                let mut laplacian = 4.0 * center;
                if i > 0 {
                    laplacian -= x[self.idx_u(i - 1, j)];
                }
                if i < k - 1 {
                    laplacian -= x[self.idx_u(i + 1, j)];
                }
                if j > 0 {
                    laplacian -= x[self.idx_u(i, j - 1)];
                }
                if j < k - 1 {
                    laplacian -= x[self.idx_u(i, j + 1)];
                }
                g[c] = laplacian / h2 - x[self.idx_f(i, j)];
            }
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let k = self.k;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let mut idx = 0;
                for j in 0..k {
                    for i in 0..k {
                        let c = (j * k + i) as Index;
                        irow[idx] = c;
                        jcol[idx] = self.idx_u(i, j) as Index;
                        idx += 1;
                        if i > 0 {
                            irow[idx] = c;
                            jcol[idx] = self.idx_u(i - 1, j) as Index;
                            idx += 1;
                        }
                        if i < k - 1 {
                            irow[idx] = c;
                            jcol[idx] = self.idx_u(i + 1, j) as Index;
                            idx += 1;
                        }
                        if j > 0 {
                            irow[idx] = c;
                            jcol[idx] = self.idx_u(i, j - 1) as Index;
                            idx += 1;
                        }
                        if j < k - 1 {
                            irow[idx] = c;
                            jcol[idx] = self.idx_u(i, j + 1) as Index;
                            idx += 1;
                        }
                        irow[idx] = c;
                        jcol[idx] = self.idx_f(i, j) as Index;
                        idx += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let h2 = self.h * self.h;
                let mut idx = 0;
                for j in 0..k {
                    for i in 0..k {
                        values[idx] = 4.0 / h2;
                        idx += 1;
                        if i > 0 {
                            values[idx] = -1.0 / h2;
                            idx += 1;
                        }
                        if i < k - 1 {
                            values[idx] = -1.0 / h2;
                            idx += 1;
                        }
                        if j > 0 {
                            values[idx] = -1.0 / h2;
                            idx += 1;
                        }
                        if j < k - 1 {
                            values[idx] = -1.0 / h2;
                            idx += 1;
                        }
                        values[idx] = -1.0;
                        idx += 1;
                    }
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
        let n = self.n();
        let k = self.k;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for kk in 0..n {
                    irow[kk] = kk as Index;
                    jcol[kk] = kk as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let h2 = self.h * self.h;
                for j in 0..k {
                    for i in 0..k {
                        values[self.idx_u(i, j)] = obj_factor * h2;
                        values[self.idx_f(i, j)] = obj_factor * self.alpha * h2;
                    }
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_state.capture(sol);
    }
}
