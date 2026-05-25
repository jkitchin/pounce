// Indexed loops mirror the time-stepping recurrences directly.
#![allow(clippy::needless_range_loop)]

//! Problem 3: Discretized linear-quadratic optimal control.
//!
//! ```text
//!   min h Σ (y_i - 1)² + α h Σ u_i²
//!   s.t. y_0 = 0
//!        y_{i+1} = y_i + h (-y_i + u_i),    i = 0..T-1
//! ```
//! Variables `[y_0, ..., y_T, u_0, ..., u_{T-1}]`; `n = 2T+1`, `m = T+1`.

use crate::problems::FinalState;
use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

pub struct OptimalControl {
    t: usize,
    h: f64,
    alpha: f64,
    pub final_state: FinalState,
}

impl OptimalControl {
    pub fn new(t: usize) -> Self {
        Self {
            t,
            h: 1.0 / t as f64,
            alpha: 0.01,
            final_state: FinalState::new(),
        }
    }

    fn n(&self) -> usize {
        2 * self.t + 1
    }
}

impl TNLP for OptimalControl {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n() as Index;
        let m = (self.t + 1) as Index;
        Some(NlpInfo {
            n,
            m,
            // Constraint 0 has 1 entry, each of the T dynamics constraints has 3.
            nnz_jac_g: 1 + 3 * self.t as Index,
            // Objective is separable quadratic, constraints linear → diagonal Hessian.
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
        let h = self.h;
        let t = self.t;
        let mut f = 0.0;
        for i in 0..=t {
            let dy = x[i] - 1.0;
            f += h * dy * dy;
        }
        for i in 0..t {
            let u = x[t + 1 + i];
            f += self.alpha * h * u * u;
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        let h = self.h;
        let t = self.t;
        for i in 0..=t {
            grad[i] = 2.0 * h * (x[i] - 1.0);
        }
        for i in 0..t {
            grad[t + 1 + i] = 2.0 * self.alpha * h * x[t + 1 + i];
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let h = self.h;
        let t = self.t;
        g[0] = x[0];
        for i in 0..t {
            g[i + 1] = x[i + 1] - (1.0 - h) * x[i] - h * x[t + 1 + i];
        }
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let t = self.t;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                for i in 0..t {
                    let base = 1 + 3 * i;
                    let row = (i + 1) as Index;
                    irow[base] = row;
                    jcol[base] = i as Index;
                    irow[base + 1] = row;
                    jcol[base + 1] = (i + 1) as Index;
                    irow[base + 2] = row;
                    jcol[base + 2] = (t + 1 + i) as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let h = self.h;
                values[0] = 1.0;
                for i in 0..t {
                    let base = 1 + 3 * i;
                    values[base] = -(1.0 - h);
                    values[base + 1] = 1.0;
                    values[base + 2] = -h;
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
        let t = self.t;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..n {
                    irow[k] = k as Index;
                    jcol[k] = k as Index;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let h = self.h;
                for i in 0..=t {
                    values[i] = obj_factor * 2.0 * h;
                }
                for i in 0..t {
                    values[t + 1 + i] = obj_factor * 2.0 * self.alpha * h;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_state.capture(sol);
    }
}
