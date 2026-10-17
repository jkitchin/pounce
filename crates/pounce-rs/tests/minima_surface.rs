//! `pounce_rs::minima` (gh#966): the search is reachable from the facade
//! alone, and finds both minima of a double well.

#![cfg(feature = "minima")]

use pounce_rs::minima::{IpoptLocalSolve, Method, MinimaOptions, Stop, find_minima};
use pounce_rs::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

/// (x^2 - 1)^2 + y^2 on [-2,2]^2: minima at (±1, 0), a saddle at the origin.
struct DoubleWell;

impl TNLP for DoubleWell {
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
        b.x_l.copy_from_slice(&[-2.0, -2.0]);
        b.x_u.copy_from_slice(&[2.0, 2.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.3, 0.2]);
        true
    }
    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some((x[0] * x[0] - 1.0).powi(2) + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = 4.0 * x[0] * (x[0] * x[0] - 1.0);
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, _x: &[f64], _new_x: bool, _g: &mut [f64]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[f64]>, _new_x: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[f64]>,
        _new_x: bool,
        of: f64,
        _l: Option<&[f64]>,
        _new_l: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap();
                values.copy_from_slice(&[of * (12.0 * x[0] * x[0] - 4.0), of * 2.0]);
            }
        }
        true
    }
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn the_facade_finds_both_wells() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .read_from_str("print_level 0\nsb yes\n", true)
        .unwrap();
    app.initialize().unwrap();
    let problem: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(DoubleWell));
    let opts = MinimaOptions {
        method: Method::Multistart,
        n_minima: 2,
        ..Default::default()
    };
    let mut solver = IpoptLocalSolve::new(&mut app);
    let r = find_minima(&problem, &mut solver, &opts).unwrap();
    assert_eq!(r.stop, Stop::TargetReached);
    let mut xs: Vec<f64> = r.minima.iter().map(|m| m.x[0]).collect();
    xs.sort_by(f64::total_cmp);
    assert!(
        (xs[0] + 1.0).abs() < 1e-6 && (xs[1] - 1.0).abs() < 1e-6,
        "{xs:?}"
    );
    assert!(r.minima.iter().all(|m| m.objective.abs() < 1e-10));
}
