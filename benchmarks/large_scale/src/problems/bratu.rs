// Indexed loops mirror the discretization stencils directly; the iterator
// rewrites clippy suggests would obscure the math.
#![allow(clippy::needless_range_loop)]

//! Problem 2: 1-D Bratu BVP discretized with a 3-point stencil.
//!
//! Discretize `-u'' = λ exp(u)` on `[0,1]` with `u(0)=u(1)=0`.
//!
//! ```text
//!   min f = 0
//!   s.t. (-x_{i-1} + 2 x_i - x_{i+1}) / h² - λ exp(x_i) = 0,   i = 1..n-2
//!        x_0 = x_{n-1} = 0   (enforced via bounds)
//! ```

use crate::problems::FinalState;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

pub struct BratuProblem {
    pub n: usize,
    lambda_bratu: f64,
    h: f64,
    pub final_state: FinalState,
}

impl BratuProblem {
    pub fn new(n: usize) -> Self {
        let h = 1.0 / (n as f64 + 1.0);
        Self {
            n,
            lambda_bratu: 1.0,
            h,
            final_state: FinalState::new(),
        }
    }
}

impl TNLP for BratuProblem {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n as Index;
        let m = (self.n - 2) as Index;
        Some(NlpInfo {
            n,
            m,
            // 3-point stencil → 3 entries per row.
            nnz_jac_g: 3 * m,
            // Diagonal (n entries) — the only nonzeros in the Lagrangian Hessian.
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
        // Dirichlet boundary conditions baked into the bounds.
        b.x_l[0] = 0.0;
        b.x_u[0] = 0.0;
        b.x_l[self.n - 1] = 0.0;
        b.x_u[self.n - 1] = 0.0;
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

    fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
        Some(0.0)
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let h2 = self.h * self.h;
        for j in 0..self.n - 2 {
            let i = j + 1;
            g[j] = (-x[i - 1] + 2.0 * x[i] - x[i + 1]) / h2 - self.lambda_bratu * x[i].exp();
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let m = self.n - 2;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..m {
                    let i = j + 1;
                    let base = 3 * j;
                    let jj = j as Index;
                    irow[base] = jj;
                    jcol[base] = (i - 1) as Index;
                    irow[base + 1] = jj;
                    jcol[base + 1] = i as Index;
                    irow[base + 2] = jj;
                    jcol[base + 2] = (i + 1) as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                let h2 = self.h * self.h;
                for j in 0..m {
                    let i = j + 1;
                    let base = 3 * j;
                    values[base] = -1.0 / h2;
                    values[base + 1] = 2.0 / h2 - self.lambda_bratu * x[i].exp();
                    values[base + 2] = -1.0 / h2;
                }
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..self.n {
                    irow[k] = k as Index;
                    jcol[k] = k as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let x = match x {
                    Some(x) => x,
                    None => return false,
                };
                for v in values.iter_mut() {
                    *v = 0.0;
                }
                // Constraint Hessian: d²g_j/dx_{j+1}² = -λ exp(x_{j+1}).
                // Only present if lambda was supplied (m > 0 ⇒ always here).
                if let Some(lambda) = lambda {
                    for j in 0..self.n - 2 {
                        let k = j + 1;
                        values[k] += lambda[j] * (-self.lambda_bratu * x[k].exp());
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
