//! Problem 1: Generalized Rosenbrock (Nash 1984, CUTE `GENROSE`).
//!
//! ```text
//!   min f(x) = 1 + Σ_{i=1}^{n-1} [100*(x_{i+1} - x_i²)² + (1 - x_{i+1})²]
//! ```
//! Unconstrained, tridiagonal Hessian. Optimum at `x = (1, ..., 1)`, `f* = 1`.
//!
//! The (1 - x_{i+1})² anchor on the *target* of each pair (rather than the
//! source) matters for Newton convergence: with the source-anchored
//! variant the algorithm needs ~2× more iterations to climb the chained
//! banana valley. See `dev-notes/rosenbrock-iter-scaling.md`.

use crate::problems::FinalState;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

pub struct ChainedRosenbrock {
    pub n: usize,
    pub final_state: FinalState,
}

impl ChainedRosenbrock {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            final_state: FinalState::new(),
        }
    }
}

impl TNLP for ChainedRosenbrock {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n as Index;
        Some(NlpInfo {
            n,
            m: 0,
            nnz_jac_g: 0,
            // Lower triangle: 1 corner entry + 2 per subsequent row = 2n - 1.
            nnz_h_lag: 2 * n - 1,
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
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            // Nash 1984 / CUTE `GENROSE` canonical start: x_i = i/(n+1).
            // A linear ramp 0…1 that puts the iterate close to the
            // optimum x* = 1. Uniform "bad" starts (e.g. -1, -1.2) drive
            // the iteration count from O(n) → O(n²) because Newton has
            // to climb the curved chain valley step by step. See
            // dev-notes/rosenbrock-iter-scaling.md.
            let np1 = (self.n as f64) + 1.0;
            for (i, v) in sp.x.iter_mut().enumerate() {
                *v = ((i + 1) as f64) / np1;
            }
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut f = 1.0;
        for i in 0..self.n - 1 {
            let a = 1.0 - x[i + 1];
            let b = x[i + 1] - x[i] * x[i];
            f += 100.0 * b * b + a * a;
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        for i in 0..self.n - 1 {
            let xi = x[i];
            let xi1 = x[i + 1];
            let r = xi1 - xi * xi;
            // ∂/∂x_i  of  100*r² + (1 - x_{i+1})²  =  100*2*r*(-2 x_i)
            grad[i] += 200.0 * r * (-2.0 * xi);
            // ∂/∂x_{i+1} = 200*r - 2*(1 - x_{i+1})
            grad[i + 1] += 200.0 * r - 2.0 * (1.0 - xi1);
        }
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
        // No constraints.
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // (0,0)
                irow[0] = 0;
                jcol[0] = 0;
                for i in 1..n {
                    let ii = i as Index;
                    // sub-diagonal (i, i-1)
                    irow[2 * i - 1] = ii;
                    jcol[2 * i - 1] = ii - 1;
                    // diagonal (i, i)
                    irow[2 * i] = ii;
                    jcol[2 * i] = ii;
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
                for i in 0..n - 1 {
                    let xi = x[i];
                    let xi1 = x[i + 1];
                    // Term:  100*(x_{i+1} - x_i²)² + (1 - x_{i+1})²
                    // d²/dx_i²   = 1200*x_i² - 400*x_{i+1}
                    let diag_i_idx = if i == 0 { 0 } else { 2 * i };
                    values[diag_i_idx] += obj_factor * (1200.0 * xi * xi - 400.0 * xi1);
                    // d²/dx_{i+1}dx_i = -400*x_i
                    let sub_idx = 2 * (i + 1) - 1;
                    values[sub_idx] += obj_factor * (-400.0 * xi);
                    // d²/dx_{i+1}² = 200 + 2
                    let diag_i1_idx = 2 * (i + 1);
                    values[diag_i1_idx] += obj_factor * 202.0;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_state.capture(sol);
    }
}
