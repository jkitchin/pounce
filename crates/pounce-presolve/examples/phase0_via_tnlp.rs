//! End-to-end worked example for PR 8 — Phase 0 orchestrator wired
//! into `PresolveTnlp`.
//!
//! Implements a tiny TNLP whose only constraints are a 2x2 equality
//! block, builds `PresolveTnlp` with `presolve_auxiliary=yes`, and
//! prints what the wrapper exposes to the (would-be) IPM. The
//! orchestrator eliminates the block via clamping `x_l = x_u =
//! fixed_value` on both variables, so the IPM sees zero remaining
//! equality rows.
//!
//! After that we synthesize a "the IPM finished" call to
//! `finalize_solution` and confirm the multiplier recovery puts
//! sensible values at the dropped row indices.
//!
//! Run with:
//! ```bash
//! cargo run -p pounce-presolve --example phase0_via_tnlp
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::SolverReturn;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{AuxiliaryCouplingPolicy, PresolveOptions, wrap_with_presolve};

/// `x + y = 3`, `x - y = 1`. Solution (2, 1). Objective: minimise
/// `(x - 5)^2 + (y - 6)^2` (so the gradient is non-zero at the
/// optimum after elimination — but here both x and y are forced by
/// the equalities, so the objective doesn't actually matter for the
/// solve).
struct Mini;

impl TNLP for Mini {
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
        for v in b.x_l.iter_mut() {
            *v = -1e19;
        }
        for v in b.x_u.iter_mut() {
            *v = 1e19;
        }
        b.g_l[0] = 3.0;
        b.g_u[0] = 3.0;
        b.g_l[1] = 1.0;
        b.g_u[1] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x[0] = 0.0;
            sp.x[1] = 0.0;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 5.0).powi(2) + (x[1] - 6.0).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        // For PureEquality classification, the gradient must be 0 at
        // the block variables at the probe point. We return the
        // *true* gradient — but Phase 0 only inspects the gradient
        // support at the probe (which is x_probe = (0, 0)). At that
        // point grad_f = (-10, -12), making the block
        // ObjectiveCoupled. So we'd need the `Aggressive` policy or a
        // zero gradient to demonstrate PR 8's elimination on this
        // exact problem.
        g[0] = 2.0 * (x[0] - 5.0);
        g[1] = 2.0 * (x[1] - 6.0);
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
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types[0] = Linearity::Linear;
        types[1] = Linearity::Linear;
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        println!("\n[finalize_solution received from PresolveTnlp]");
        println!("   x      = {:?}", sol.x);
        println!("   lambda = {:?}", sol.lambda);
        println!("   g      = {:?}", sol.g);
    }
}

fn main() {
    let opts = PresolveOptions {
        enabled: true,
        auxiliary: true,
        // Aggressive lets us eliminate even when grad_f is non-zero
        // at the block variables at the probe (ObjectiveCoupled).
        auxiliary_coupling: AuxiliaryCouplingPolicy::Aggressive,
        ..PresolveOptions::defaults()
    };
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Mini));
    let wrapped = wrap_with_presolve(inner, opts).expect("wrap ok");

    let info = wrapped.borrow_mut().get_nlp_info().expect("init");
    println!("phase0_via_tnlp — PR 8 of pounce#53");
    println!("====================================");
    println!("Inner TNLP: n=2, m=2 (two equalities).");
    println!("After PresolveTnlp wrapping with presolve_auxiliary=yes:");
    println!("   outer n = {}", info.n);
    println!("   outer m = {}", info.m);
    println!("   outer nnz_jac_g = {}", info.nnz_jac_g);

    // Read the clamped bounds.
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
    println!("\nClamped bounds (from PresolveTnlp):");
    println!("   x_l = {:?}", x_l);
    println!("   x_u = {:?}", x_u);
    println!("   g_l = {:?}", g_l);

    // Simulate the IPM handing back a solution: the eliminated vars
    // are pinned to their clamps; outer lambda is empty (no rows).
    let sol_x = [x_l[0], x_l[1]];
    let sol_z_l = vec![0.0; info.n as usize];
    let sol_z_u = vec![0.0; info.n as usize];
    let sol_lambda: Vec<Number> = Vec::new(); // outer m == 0
    let sol_g: Vec<Number> = Vec::new();
    let ip_data = IpoptData::default();
    let ip_cq = IpoptCq::default();
    wrapped.borrow_mut().finalize_solution(
        Solution {
            status: SolverReturn::Success,
            x: &sol_x,
            z_l: &sol_z_l,
            z_u: &sol_z_u,
            g: &sol_g,
            lambda: &sol_lambda,
            obj_value: 0.0,
        },
        &ip_data,
        &ip_cq,
    );

    println!("\n✓ End-to-end orchestrator path exercised.");
}
