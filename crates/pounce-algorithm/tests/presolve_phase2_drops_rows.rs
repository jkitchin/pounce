//! Phase 2 acceptance for pounce-presolve (#20):
//!
//! Build a tiny NLP with TWO linear constraints, one of which is
//! implied by the variable bounds. With the presolve wrapper enabled
//! and `redundant_constraint_removal=yes`, the solver should see only
//! one constraint, and the user's `finalize_solution` should still
//! receive the full-sized `g` / `lambda` (with the dropped row's
//! lambda reinstated as 0).
//!
//! Problem:
//!
//! ```text
//!   min  (x1 - 0.3)^2 + (x2 - 0.3)^2
//!   s.t. x1 + x2 = 1                        (linear, active)
//!        -10 ≤ x1 - x2 ≤ 10                 (linear, redundant on [0,1]^2)
//!        0 ≤ x1, x2 ≤ 1
//! ```

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{wrap_with_presolve, PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct TwoLinears {
    final_g_len: Option<usize>,
    final_lambda: Option<Vec<Number>>,
}

impl TNLP for TwoLinears {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 4, // row 0: (0,0)(0,1); row 1: (1,0)(1,1)
            nnz_h_lag: 2, // diag (only from the objective)
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[1.0, 1.0]);
        b.g_l.copy_from_slice(&[1.0, -10.0]);
        b.g_u.copy_from_slice(&[1.0, 10.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        types[1] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let a = x[0] - 0.3;
        let b = x[1] - 0.3;
        Some(a * a + b * b)
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 0.3);
        g[1] = 2.0 * (x[1] - 0.3);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] - x[1];
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
                values.copy_from_slice(&[1.0, 1.0, 1.0, -1.0]);
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
        self.final_g_len = Some(sol.g.len());
        self.final_lambda = Some(sol.lambda.to_vec());
    }
}

#[test]
fn phase2_drops_one_row_in_info_outer() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(TwoLinears::default()));
    let opts = PresolveOptions {
        enabled: true,
        ..PresolveOptions::defaults()
    };
    let wrapped = wrap_with_presolve(inner, opts).unwrap();
    let info = wrapped.borrow_mut().get_nlp_info().unwrap();
    assert_eq!(info.n, 2);
    assert_eq!(info.m, 1, "expected 1 outer constraint (1 dropped)");
    assert_eq!(info.nnz_jac_g, 2, "kept row contributes 2 nnz");
}

#[test]
fn phase2_end_to_end_solve_succeeds_and_inner_sees_full_g() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    let inner_concrete = Rc::new(RefCell::new(TwoLinears::default()));
    let inner: Rc<RefCell<dyn TNLP>> = Rc::clone(&inner_concrete) as _;
    let opts = PresolveOptions {
        enabled: true,
        ..PresolveOptions::defaults()
    };
    let wrapped = wrap_with_presolve(inner, opts).unwrap();
    let _ = app.optimize_tnlp(wrapped);
    let stats = app.statistics();
    // Optimum: x1 = x2 = 0.5 ⇒ f* = 2*(0.2)^2 = 0.08.
    assert!(
        (stats.final_objective - 0.08).abs() < 1e-6,
        "final_objective = {}",
        stats.final_objective
    );
    let user = inner_concrete.borrow();
    // Even though the solver only ever saw 1 constraint, the
    // user-facing finalize_solution must receive the full m=2 vectors.
    assert_eq!(user.final_g_len, Some(2));
    let lam = user.final_lambda.as_ref().expect("lambda set");
    assert_eq!(lam.len(), 2);
    // The dropped row's lambda was reinstated as 0.
    assert!(lam[1].abs() < 1e-12, "dropped-row lambda should be 0, got {}", lam[1]);
}

#[test]
fn phase2_disabled_leaves_m_alone() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(TwoLinears::default()));
    let opts = PresolveOptions {
        enabled: true,
        redundant_constraint_removal: false,
        ..PresolveOptions::defaults()
    };
    let wrapped = wrap_with_presolve(inner, opts).unwrap();
    let info = wrapped.borrow_mut().get_nlp_info().unwrap();
    assert_eq!(info.m, 2, "with removal disabled, both rows survive");
}

#[allow(dead_code)]
fn _types_compile() -> Option<PresolveTnlp> {
    None
}
