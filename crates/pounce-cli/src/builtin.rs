//! Built-in TNLP test problems for the CLI. Each problem is a
//! self-contained `impl TNLP` so the CLI can run end-to-end without
//! parsing an `.nl` file.
//!
//! Currently shipped:
//!
//! * `quadratic` — `min (x[0]-3)^2 + (x[1]-4)^2`, unconstrained,
//!   exact Hessian = `2I`. Optimum at `(3, 4)`, `f* = 0`.
//! * `rosenbrock` — `min 100*(x[1]-x[0]^2)^2 + (1-x[0])^2`,
//!   unconstrained, exact Hessian. Optimum at `(1, 1)`, `f* = 0`.

use pounce_common::types::{Index, Number};
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

pub fn list() -> Vec<&'static str> {
    vec![
        "quadratic",
        "rosenbrock",
        "bounded-quadratic",
        "eq-quadratic",
        "circle",
        "infeasible-eq",
    ]
}

pub fn lookup(name: &str) -> Option<Rc<RefCell<dyn TNLP>>> {
    match name {
        "quadratic" => Some(Rc::new(RefCell::new(Quadratic::default()))),
        "rosenbrock" => Some(Rc::new(RefCell::new(Rosenbrock::default()))),
        "bounded-quadratic" => Some(Rc::new(RefCell::new(BoundedQuadratic::default()))),
        "eq-quadratic" => Some(Rc::new(RefCell::new(EqQuadratic::default()))),
        "circle" => Some(Rc::new(RefCell::new(Circle::default()))),
        "infeasible-eq" => Some(Rc::new(RefCell::new(InfeasibleEq::default()))),
        _ => None,
    }
}

// --------------------------------------------------------------------
// Quadratic: min (x0 - 3)^2 + (x1 - 4)^2
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct Quadratic {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for Quadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 2, // diagonal Hessian, lower triangle
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -2e19);
        b.x_u.iter_mut().for_each(|v| *v = 2e19);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * (x[0] - 3.0);
        grad[1] = 2.0 * (x[1] - 4.0);
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
        true
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

// --------------------------------------------------------------------
// Rosenbrock: min 100 (x1 - x0^2)^2 + (1 - x0)^2
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct Rosenbrock {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for Rosenbrock {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 3, // dense 2x2 lower triangle: (0,0), (1,0), (1,1)
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -2e19);
        b.x_u.iter_mut().for_each(|v| *v = 2e19);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-1.2, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let a = x[1] - x[0] * x[0];
        let b = 1.0 - x[0];
        Some(100.0 * a * a + b * b)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        // d/dx0 = -400 x0 (x1 - x0^2) - 2 (1 - x0)
        // d/dx1 =  200 (x1 - x0^2)
        grad[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
        grad[1] = 200.0 * (x[1] - x[0] * x[0]);
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // Lower triangle (row >= col): (0,0), (1,0), (1,1).
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap_or(&[0.0, 0.0]);
                let h00 = -400.0 * (x[1] - 3.0 * x[0] * x[0]) + 2.0;
                let h10 = -400.0 * x[0];
                let h11 = 200.0;
                values[0] = obj_factor * h00;
                values[1] = obj_factor * h10;
                values[2] = obj_factor * h11;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

// Small helper for doc references
#[allow(dead_code)]
fn _ix(_: Index) {}

// --------------------------------------------------------------------
// BoundedQuadratic: min (x0-3)^2 + (x1-4)^2 s.t. 0 <= x0 <= 2, 0 <= x1 <= 2
// Optimum is at the corner (2, 2) with f* = 1 + 4 = 5.
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct BoundedQuadratic {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for BoundedQuadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[2.0, 2.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0).powi(2) + (x[1] - 4.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * (x[0] - 3.0);
        grad[1] = 2.0 * (x[1] - 4.0);
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
        true
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

// --------------------------------------------------------------------
// EqQuadratic: min x0^2 + x1^2  s.t.  x0 + x1 = 1
// Optimum at (1/2, 1/2), f* = 1/2, multiplier y = -1.
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct EqQuadratic {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for EqQuadratic {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -2e19);
        b.x_u.iter_mut().for_each(|v| *v = 2e19);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0;
            }
        }
        true
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

// --------------------------------------------------------------------
// Circle: min  x0  s.t.  x0^2 + x1^2 = 1
// Optimum at (-1, 0), f* = -1, multiplier y = 1/2.
// Tests nonlinear equality constraint with non-trivial Hessian
// contribution from the constraint (∇²g_0 = 2I) into the Lagrangian.
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct Circle {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for Circle {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -2e19);
        b.x_u.iter_mut().for_each(|v| *v = 2e19);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-0.5, 0.5]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 1.0;
        grad[1] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap_or(&[0.0, 0.0]);
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                // ∇²L = obj_factor * ∇²f + λ * ∇²g_0 = 0 + λ * 2I
                let lam = lambda.map(|l| l[0]).unwrap_or(0.0);
                values[0] = 2.0 * lam;
                values[1] = 2.0 * lam;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

// --------------------------------------------------------------------
// InfeasibleEq: min x0^2 + x1^2
//   s.t.  x0 + x1 = 1   (g_0)
//         x0 + x1 = 2   (g_1)
// The two equalities are mutually contradictory, so no feasible point
// exists. The standard solve drives the restoration phase, which also
// cannot achieve feasibility, returning Restoration_Failed. With
// `l1_fallback_on_restoration_failure=yes` (or
// `l1_exact_penalty_barrier=yes`), the CLI then performs a second
// inner solve via the ℓ₁-exact penalty-barrier wrapper. That second
// pass is what exercises the multi-pass restoration factory provider
// path — the very path that previously panicked with
// "restoration factory invoked more than once".
// --------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct InfeasibleEq {
    pub final_x: Option<[Number; 2]>,
    pub final_obj: Number,
}

impl TNLP for InfeasibleEq {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.iter_mut().for_each(|v| *v = -2e19);
        b.x_u.iter_mut().for_each(|v| *v = 2e19);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        b.g_l[1] = 2.0;
        b.g_u[1] = 2.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad[0] = 2.0 * x[0];
        grad[1] = 2.0 * x[1];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] + x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            }
        }
        true
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_obj = sol.obj_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_contains_known_problems() {
        let l = list();
        assert!(l.contains(&"quadratic"));
        assert!(l.contains(&"rosenbrock"));
    }

    #[test]
    fn quadratic_evaluates_correctly() {
        let mut q = Quadratic::default();
        let f = q.eval_f(&[3.0, 4.0], true).unwrap();
        assert_eq!(f, 0.0);
        let mut g = [0.0; 2];
        q.eval_grad_f(&[0.0, 0.0], true, &mut g);
        assert_eq!(g, [-6.0, -8.0]);
    }

    #[test]
    fn rosenbrock_grad_zero_at_optimum() {
        let mut r = Rosenbrock::default();
        let f = r.eval_f(&[1.0, 1.0], true).unwrap();
        assert!(f.abs() < 1e-15);
        let mut g = [0.0; 2];
        r.eval_grad_f(&[1.0, 1.0], true, &mut g);
        assert!(g[0].abs() < 1e-12);
        assert!(g[1].abs() < 1e-12);
    }

    #[test]
    fn lookup_returns_known_and_rejects_unknown() {
        assert!(lookup("quadratic").is_some());
        assert!(lookup("rosenbrock").is_some());
        assert!(lookup("nonsense").is_none());
    }
}
