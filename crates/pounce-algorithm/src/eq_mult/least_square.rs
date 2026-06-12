//! Least-squares multiplier estimate — port of
//! `Algorithm/IpLeastSquareMults.{hpp,cpp}`. Solves the W=0
//! augmented system to get an initial `y_c`/`y_d`.
//!
//! The system, with `delta_x = delta_s = 1.0` and all other
//! perturbations / weights zero (matching upstream `IpLeastSquareMults.cpp:60`):
//!
//! ```text
//!   [ I    0   J_c^T  J_d^T ] [dx ]   [ −∇f + Pₗ z_L − Pᵤ z_U ]
//!   [ 0    I    0      −I   ] [ds ] = [    Pₗ v_L − Pᵤ v_U    ]
//!   [ J_c  0    0       0   ] [dyc]   [          0            ]
//!   [ J_d −I    0       0   ] [dyd]   [          0            ]
//! ```
//!
//! Sign convention from `IpLeastSquareMults.cpp:54-61`. `dyc`, `dyd`
//! are the least-squares estimates we keep as `y_c`, `y_d`; `dx`,
//! `ds` are discarded.

use crate::eq_mult::r#trait::EqMultCalculator;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use pounce_linalg::Vector;
use pounce_linsol::ESymSolverStatus;
use std::cell::RefCell;
use std::rc::Rc;

pub struct LeastSquareMults;

impl LeastSquareMults {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LeastSquareMults {
    fn default() -> Self {
        Self::new()
    }
}

impl EqMultCalculator for LeastSquareMults {
    fn calculate_y_eq(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
        y_c: &mut dyn Vector,
        y_d: &mut dyn Vector,
    ) -> bool {
        let curr = match data.borrow().curr.clone() {
            Some(c) => c,
            None => return false,
        };

        // Pull NLP-evaluated quantities first so the `nlp.borrow_mut()`
        // inside CQ's eval helpers can complete before we take the
        // shared `nlp.borrow()` for the bound-selection matrices.
        let cq_ref = cq.borrow();
        let grad_f = cq_ref.curr_grad_f();
        let j_c = cq_ref.curr_jac_c();
        let j_d = cq_ref.curr_jac_d();
        // Upstream `IpLeastSquareMults.cpp:80` passes a `zeroW` SymMatrix
        // (same sparsity as the real Hessian) with `W_factor=0.0`. This
        // ensures `StdAugSystemSolver` pins its triplet structure with
        // the W slots present, so subsequent calls (with the actual
        // Hessian) write into those slots rather than skipping them.
        let zero_w = cq_ref.curr_exact_hessian();
        drop(cq_ref);

        let nlp_ref = nlp.borrow();

        // rhs_x = −∇f + Pₗ z_L − Pᵤ z_U  (mirrors
        // `IpLeastSquareMults.cpp:54-57` exactly).
        let mut rhs_x = grad_f.make_new();
        rhs_x.copy(&*grad_f);
        nlp_ref
            .px_l()
            .mult_vector(1.0, &*curr.z_l, -1.0, &mut *rhs_x);
        nlp_ref
            .px_u()
            .mult_vector(-1.0, &*curr.z_u, 1.0, &mut *rhs_x);

        // rhs_s = Pₗ v_L − Pᵤ v_U  (zero-init then mult; mirrors
        // `IpLeastSquareMults.cpp:60-61`).
        let mut rhs_s = curr.s.make_new();
        nlp_ref
            .pd_l()
            .mult_vector(1.0, &*curr.v_l, 0.0, &mut *rhs_s);
        nlp_ref
            .pd_u()
            .mult_vector(-1.0, &*curr.v_u, 1.0, &mut *rhs_s);

        // rhs_c = 0, rhs_d = 0.
        let mut rhs_c = curr.y_c.make_new();
        rhs_c.set(0.0);
        let mut rhs_d = curr.y_d.make_new();
        rhs_d.set(0.0);

        // sol_x, sol_s scratch (discarded after solve).
        let mut sol_x = rhs_x.make_new();
        let mut sol_s = rhs_s.make_new();

        let coeffs = AugSysCoeffs {
            w: Some(&*zero_w),
            w_factor: 0.0,
            d_x: None,
            delta_x: 1.0,
            d_s: None,
            delta_s: 1.0,
            j_c: &*j_c,
            d_c: None,
            // Tiny δ_c, δ_d (upstream `IpLeastSquareMults.cpp` uses 0).
            // This is the SAME W=0 / structurally-zero (3,3)/(4,4)-block
            // augmented system the dual initializer solves in
            // `init/default.rs`, where pounce-feral's LDLᵀ mis-reports the
            // inertia (it counted 0 negative eigenvalues on nuffield2_trap
            // where the true count is n_c+n_d, raising WrongInertia). Here
            // a spurious failure makes `calculate_y_eq` return false and the
            // caller (`init/default.rs:388-390`) silently leaves y_c=y_d=0
            // — the iter-0 `inf_du` blow-up this step exists to prevent.
            // Perturbing by 1e-8 keeps the LS solution numerically identical
            // (the constraint Jacobian dominates) while giving the diagonal
            // something nonzero to pivot on. Mirrors the sibling workaround
            // at `init/default.rs:171,174`; keep the two in sync.
            delta_c: 1e-8,
            j_d: &*j_d,
            d_d: None,
            delta_d: 1e-8,
        };
        let aug_rhs = AugSysRhs {
            rhs_x: &*rhs_x,
            rhs_s: &*rhs_s,
            rhs_c: &*rhs_c,
            rhs_d: &*rhs_d,
        };
        let mut sol = AugSysSol {
            sol_x: &mut *sol_x,
            sol_s: &mut *sol_s,
            sol_c: y_c,
            sol_d: y_d,
        };

        let num_eq = aug_rhs.rhs_c.dim() + aug_rhs.rhs_d.dim();
        let check_neg = aug_solver.provides_inertia();
        let status = aug_solver.solve(&coeffs, &aug_rhs, &mut sol, check_neg, num_eq);
        matches!(status, ESymSolverStatus::Success)
    }
}
