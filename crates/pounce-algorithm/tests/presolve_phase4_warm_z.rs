//! Phase 4 acceptance for pounce-presolve (#20):
//!
//! When Phase 1 tightens a variable bound inward, that bound is
//! likely active at the optimum, so the wrapper should publish a
//! non-zero warm-start value for the corresponding bound multiplier.
//! `get_starting_point` should overlay those hints when `init_z` is
//! requested without overwriting user-supplied warm-start values.

use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

/// x1 + x2 = 1, x_i ∈ [0, 10]. Phase 1 must tighten x_u to 1 for
/// both variables, so z_u_warm[0] and z_u_warm[1] become positive.
struct ChainTwoTight;

impl TNLP for ChainTwoTight {
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
        if sp.init_z {
            for v in sp.z_l.iter_mut() {
                *v = 0.0;
            }
            for v in sp.z_u.iter_mut() {
                *v = 0.0;
            }
        }
        if sp.init_lambda {
            sp.lambda[0] = 0.0;
        }
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 0.3).powi(2) + (x[1] - 0.3).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 0.3);
        g[1] = 2.0 * (x[1] - 0.3);
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
                values.copy_from_slice(&[1.0, 1.0]);
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
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn build(opts: PresolveOptions) -> PresolveTnlp {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ChainTwoTight));
    let mut p = PresolveTnlp::new(inner, opts);
    let _ = p.get_nlp_info();
    p
}

#[test]
fn phase4_tightened_upper_bounds_emit_z_u_warm_hints() {
    let p = build(PresolveOptions {
        enabled: true,
        warm_z_bounds: true,
        bound_mult_init_val: 2.5,
        ..PresolveOptions::defaults()
    });
    let (zl, zu) = p.z_warm_starts().expect("init ran");
    assert_eq!(zl, &[0.0, 0.0], "no lower-bound tightening expected");
    assert_eq!(
        zu,
        &[2.5, 2.5],
        "upper bounds got tightened from 10 to 1 ⇒ warm hints"
    );
}

#[test]
fn phase4_starting_point_overlays_hints_when_init_z_requested() {
    let mut p = build(PresolveOptions {
        enabled: true,
        warm_z_bounds: true,
        bound_mult_init_val: 1.0,
        ..PresolveOptions::defaults()
    });
    let mut x = vec![0.0; 2];
    let mut z_l = vec![0.0; 2];
    let mut z_u = vec![0.0; 2];
    let mut lambda = vec![0.0; 1];
    let ok = p.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: true,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: true,
        lambda: &mut lambda,
    });
    assert!(ok);
    assert_eq!(z_l, vec![0.0, 0.0]);
    assert_eq!(z_u, vec![1.0, 1.0], "z_u overlaid with presolve hint");
}

#[test]
fn phase4_starting_point_leaves_user_z_alone_when_warm_off() {
    let mut p = build(PresolveOptions {
        enabled: true,
        warm_z_bounds: false,
        ..PresolveOptions::defaults()
    });
    let mut x = vec![0.0; 2];
    let mut z_l = vec![0.0; 2];
    let mut z_u = vec![0.0; 2];
    let mut lambda = vec![0.0; 1];
    let ok = p.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: true,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: true,
        lambda: &mut lambda,
    });
    assert!(ok);
    assert_eq!(z_u, vec![0.0, 0.0], "warm hints suppressed");
}
