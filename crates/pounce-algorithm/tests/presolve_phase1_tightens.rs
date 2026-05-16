//! Phase 1 acceptance for pounce-presolve (#20):
//!
//! Define a tiny LP-like NLP whose constraints declare themselves
//! linear via `get_constraints_linearity`. With the presolve wrapper
//! enabled and `bound_tightening=yes`, the bounds the solver actually
//! receives must be the tightened ones (not the raw ones the inner
//! TNLP returns).
//!
//! Problem:
//!
//! ```text
//!   min  (x1 - 0.7)^2 + (x2 - 0.7)^2
//!   s.t. x1 + x2 = 1               (linear)
//!        0 ≤ x1 ≤ 10
//!        0 ≤ x2 ≤ 10
//! ```
//!
//! Andersen-style propagation drives both `x_u`s down to 1.

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
struct LinearChain;

impl TNLP for LinearChain {
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
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let a = x[0] - 0.7;
        let b = x[1] - 0.7;
        Some(a * a + b * b)
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 0.7);
        g[1] = 2.0 * (x[1] - 0.7);
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
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
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
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn phase1_tightens_x_upper_bounds() {
    // We exercise PresolveTnlp directly (without a solve) to inspect
    // the cached tightened bounds.
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(LinearChain));
    let opts = PresolveOptions {
        enabled: true,
        ..PresolveOptions::defaults()
    };

    // The wrapper returned by `wrap_with_presolve` is opaque
    // (Rc<RefCell<dyn TNLP>>) — to inspect cached bounds we build
    // PresolveTnlp via the Rust API directly through the same path
    // by serving and re-reading bounds.
    let wrapped = wrap_with_presolve(inner, opts).unwrap();

    // Drive the cache by calling get_nlp_info + get_bounds_info.
    let info = wrapped.borrow_mut().get_nlp_info().unwrap();
    let mut x_l = vec![0.0; info.n as usize];
    let mut x_u = vec![0.0; info.n as usize];
    let mut g_l = vec![0.0; info.m as usize];
    let mut g_u = vec![0.0; info.m as usize];
    assert!(wrapped.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }));

    // Andersen propagation: x1 + x2 = 1, with x ≥ 0 ⇒ each xi ≤ 1.
    assert!(x_u[0] <= 1.0 + 1e-12, "x_u[0] = {}", x_u[0]);
    assert!(x_u[1] <= 1.0 + 1e-12, "x_u[1] = {}", x_u[1]);
    // Lower stays at 0.
    assert!(x_l[0] >= -1e-12 && x_l[0] <= 1e-12);
    assert!(x_l[1] >= -1e-12 && x_l[1] <= 1e-12);
}

#[test]
fn phase1_end_to_end_solve_succeeds() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();

    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(LinearChain));
    let opts = PresolveOptions {
        enabled: true,
        ..PresolveOptions::defaults()
    };
    let wrapped = wrap_with_presolve(inner, opts).unwrap();

    let status = app.optimize_tnlp(wrapped);
    let stats = app.statistics();
    eprintln!(
        "LinearChain: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective
    );
    // Optimum: x1 = x2 = 0.5, f* = 2 * 0.04 = 0.08
    assert!(
        (stats.final_objective - 0.08).abs() < 1e-6,
        "final_objective = {}",
        stats.final_objective
    );
}

#[test]
fn phase1_disabled_leaves_bounds_alone() {
    // Same TNLP, but bound tightening disabled at the option level.
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(LinearChain));
    let opts = PresolveOptions {
        enabled: true,
        bound_tightening: false,
        ..PresolveOptions::defaults()
    };
    let wrapped = wrap_with_presolve(inner, opts).unwrap();
    let info = wrapped.borrow_mut().get_nlp_info().unwrap();
    let mut x_l = vec![0.0; info.n as usize];
    let mut x_u = vec![0.0; info.n as usize];
    let mut g_l = vec![0.0; info.m as usize];
    let mut g_u = vec![0.0; info.m as usize];
    wrapped.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    });
    // Untouched.
    assert_eq!(x_u, vec![10.0, 10.0]);
}

// Marker to ensure we can `use PresolveTnlp` as a type even though
// the public surface is `Rc<RefCell<dyn TNLP>>`. The concrete type is
// exposed so integration tests in pounce-presolve itself (later) can
// downcast or inspect cached bounds — this just compile-checks the
// path here.
#[allow(dead_code)]
fn _types_compile() -> Option<PresolveTnlp> {
    None
}
