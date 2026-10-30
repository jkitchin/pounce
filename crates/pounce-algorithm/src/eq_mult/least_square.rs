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
        drop(cq_ref);

        let nlp_ref = nlp.borrow();
        // Upstream `IpLeastSquareMults.cpp:80` passes a `zeroW` SymMatrix
        // (same sparsity as the real Hessian) with `W_factor=0.0`. This
        // ensures `StdAugSystemSolver` pins its triplet structure with
        // the W slots present, so subsequent calls (with the actual
        // Hessian) write into those slots rather than skipping them.
        //
        // Structure only: upstream takes it from `IpNLP().uninitialized_h()`
        // (`IpLeastSquareMults.cpp:38`), not from an evaluation. Using
        // `curr_exact_hessian()` here — an unmemoized `eval_h` — invoked
        // the user's Hessian callback once per `calculate_y_eq` even under
        // `hessian_approximation = limited-memory`, where the whole point
        // is that the user has declared they are not supplying one
        // (gh#698). The values are never read: `w_factor` is 0.0.
        let zero_w = nlp_ref.uninitialized_h();

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

        // δ_c = δ_d = 0, matching upstream `IpLeastSquareMults.cpp:80-81`,
        // with a perturbed retry only if the unperturbed solve fails (#688).
        //
        // Eliminating `w` from this W=0 system gives
        //
        //     y = −(J Jᵀ + δ I)⁻¹ J · rhs_x
        //
        // so δ=0 returns the least-squares multiplier this calculator is
        // named for, and δ>0 returns a Tikhonov-regularized one, damped
        // by `O(δ / σ_min(J)²)` along the weakest singular directions.
        //
        // Review item M3 introduced δ=1e-8 here — the site previously
        // matched upstream — to mirror the dual initializer's workaround
        // for pounce-feral mis-reporting the inertia of a
        // structurally-zero (3,3)/(4,4) block (0 negatives on
        // nuffield2_trap against a true n_c+n_d, raising WrongInertia).
        // A spurious failure makes this return false and the caller
        // leaves y_c=y_d=0. M3 argued the perturbation was numerically
        // inert because the suite stayed green — true, and true only
        // while `σ_min(J)² ≫ δ`, which holds for every problem in this
        // repo. That is a statement about the covered problems, not a
        // scale-free property.
        //
        // Two things retired it. gh#540 and gh#592 fixed feral's inertia
        // reporting at its source — `inertia_trust_floor` reports
        // `Singular` rather than `WrongInertia` when the count is
        // contradicted by a working-precision pivot, so `δ_c` is reached
        // for where it actually repairs a rank-deficient constraint
        // block — which is the defect M3 was compensating for from a
        // distance. And #688 measured what the compensation costs on a
        // problem that leaves the inert regime: on a ~59,000-variable
        // collocation NLP the damping is `O(1)`, and because `recalc_y`
        // recomputes `y` the same biased way every iteration the error is
        // a fixed point of the estimator rather than a transient. It
        // lands directly in `inf_du`. `recalc_y=yes` stalled at 1.55e-01,
        // *worse* than `recalc_y=no` at 1.73e-02; at δ=0 the model
        // converges, on both MA57 and FERAL.
        //
        // The retry keeps M3's protection without paying for it: the
        // common case is now bit-identical to upstream, and a solve that
        // genuinely fails still gets the perturbation before the caller
        // falls back to zero. `nuffield2_trap` is not in this repo, so
        // the retry cannot be regression-tested here — the same reason
        // M3 shipped without a fail-first test, and why the guard is kept
        // rather than dropped outright.
        //
        // #688 could only reach `recalc_y`, so for one release the three
        // *initializer* sites still took δ=1e-8 unconditionally, behind
        // an `unregularized: bool`. gh#693 retired that flag: the same
        // argument applies to a starting guess that a capped `y0` then
        // carries into iteration 0, and no caller was left asking for the
        // damped estimate. The retry below is now the only behaviour.
        let mut status = ESymSolverStatus::Success;
        let deltas: &[pounce_common::types::Number] = &[0.0, 1e-8];
        for &delta in deltas {
            let coeffs = AugSysCoeffs {
                w: Some(&*zero_w),
                w_factor: 0.0,
                d_x: None,
                delta_x: 1.0,
                d_s: None,
                delta_s: 1.0,
                j_c: &*j_c,
                d_c: None,
                delta_c: delta,
                j_d: &*j_d,
                d_d: None,
                delta_d: delta,
                s_blocks: None,
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
            status = aug_solver.solve(&coeffs, &aug_rhs, &mut sol, check_neg, num_eq);
            if matches!(status, ESymSolverStatus::Success) {
                return true;
            }
        }
        matches!(status, ESymSolverStatus::Success)
    }
}
