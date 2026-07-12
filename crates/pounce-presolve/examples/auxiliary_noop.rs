//! Worked example for PR 1 of issue #53.
//!
//! Builds a tiny 2-variable / 1-equality TNLP, wraps it in a
//! `PresolveTnlp` with `presolve_auxiliary=yes`, runs `eval_jac_g`
//! once, and prints the auxiliary-preprocessing diagnostics. In PR 1
//! the orchestrator is a no-op, so every counter prints as zero —
//! the point is to demonstrate that the wiring is in place and the
//! `presolve_auxiliary=yes` path doesn't perturb the inner problem.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example auxiliary_noop
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::cell::RefCell;
use std::rc::Rc;

use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_presolve::{AuxiliaryCouplingPolicy, PresolveOptions, PresolveTnlp, wrap_with_presolve};

/// `min x[0]^2 + x[1]^2  s.t.  x[0] + x[1] = 1`.
struct Mini;

impl TNLP for Mini {
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
        b.x_l.iter_mut().for_each(|v| *v = -1e19);
        b.x_u.iter_mut().for_each(|v| *v = 1e19);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        true
    }
    fn eval_f(&mut self, x: &[f64], _new_x: bool) -> Option<f64> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[f64], _new_x: bool, g: &mut [f64]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[f64]>, _new_x: bool, mode: SparsityRequest<'_>) -> bool {
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
    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn main() {
    let opts = PresolveOptions {
        enabled: true,
        auxiliary: true,
        auxiliary_coupling: AuxiliaryCouplingPolicy::Safe,
        auxiliary_diagnostics: true,
        ..PresolveOptions::defaults()
    };
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mini));
    let wrapped = wrap_with_presolve(inner, opts).expect("wrap ok");

    // Trigger lazy init through any TNLP method.
    let info = wrapped.borrow_mut().get_nlp_info().expect("get_nlp_info");

    // Force one Jacobian-values call so the example exercises the
    // forwarding path (the wrapper does not change anything in PR 1).
    let mut values = vec![0.0; info.nnz_jac_g as usize];
    let ok = wrapped.borrow_mut().eval_jac_g(
        Some(&[0.5, 0.5]),
        true,
        SparsityRequest::Values {
            values: &mut values,
        },
    );
    assert!(ok, "eval_jac_g(Values) must succeed");

    // Downcast back to PresolveTnlp via a typed handle would require
    // a different constructor; instead, build a second wrapper using
    // PresolveTnlp::new so we can read the diagnostics.
    let inner2: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mini));
    let mut typed = PresolveTnlp::new(inner2, opts);
    let _ = typed.get_nlp_info();
    let d = typed.auxiliary_diagnostics();

    println!("auxiliary_noop example — PR 1 of pounce#53");
    println!(
        "inner shape: n={}, m={}, nnz_jac_g={}",
        info.n, info.m, info.nnz_jac_g
    );
    println!("jac values at (0.5, 0.5): {:?}", values);
    println!(
        "diagnostics: blocks_eliminated={}, vars_eliminated={}, rows_eliminated={}, total_time_ms={}",
        d.blocks_eliminated, d.vars_eliminated, d.rows_eliminated, d.total_time_ms,
    );
    println!("rejection_reasons: {:?}", d.rejection_reasons);
}
