//! Regression test: user-facing bound duals `z_l` / `z_u` must be
//! unscaled (divided by `obj_scale_factor`) at finalize time.
//!
//! Commit 6d713c9 fixed all finalize sites to divide the user-facing
//! multipliers by the objective scale factor — λ via
//! `finalize_solution_lambda` AND the bound duals via
//! `finalize_solution_z_l` / `finalize_solution_z_u`
//! (`crates/pounce-nlp/src/orig_ipopt_nlp.rs`), called from the
//! finalize paths in `application.rs`. The λ half has a CLI regression
//! test (`pounce-cli/tests/json_report.rs`, fixture `dual_scaled.nl`);
//! this test covers the z_l / z_u half at the application level.
//!
//! Problem (n = 2, m = 0):
//!
//! ```text
//! min  (x0 - 3)^2 + 1000*x1
//! s.t. -10 <= x0 <= 2,  0 <= x1 <= 10
//! ```
//!
//! At the start point (0, 5) the objective gradient is (-6, 1000), so
//! its ∞-norm (1000) exceeds the default `nlp_scaling_max_gradient`
//! (100) and gradient-based scaling — the default `nlp_scaling_method`
//! — fires with `obj_scale_factor = 100/1000 = 0.1`.
//!
//! The optimum pins both variables on a bound: `x* = (2, 0)`,
//! `f* = 1`. KKT stationarity `∇f − z_l + z_u = 0` gives the analytic
//! duals
//!
//! ```text
//! z_u[0] = −∂f/∂x0 = −2(2−3) = 2      (upper bound on x0 active)
//! z_l[1] =  ∂f/∂x1 = 1000             (lower bound on x1 active)
//! ```
//!
//! The algorithm-side (scaled) duals are 0.1× those values; the user
//! must never see them. We solve the same TNLP twice — default
//! gradient-based scaling vs `nlp_scaling_method = none` — and assert
//! the captured `z_l` / `z_u` agree between the runs and match the
//! analytic values. If the `finalize_solution_z_*` obj-scale division
//! were reverted, the scaled run would report `z_l[1] ≈ 100` and
//! `z_u[0] ≈ 0.2` (10× off) and both assertions would fail.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Capturing TNLP for `min (x0 - 3)^2 + 1000*x1` with
/// `x0 ∈ [-10, 2]`, `x1 ∈ [0, 10]`. Records what the solver hands the
/// user in `finalize_solution`.
#[derive(Default)]
struct SteepBoundPinned {
    final_x: Option<Vec<Number>>,
    final_obj: Option<Number>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
    final_lambda: Option<Vec<Number>>,
}

impl TNLP for SteepBoundPinned {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, 0.0]);
        b.x_u.copy_from_slice(&[2.0, 10.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // ∇f(0, 5) = (-6, 1000): ∞-norm 1000 > nlp_scaling_max_gradient
        // = 100, so default gradient-based scaling fires (df = 0.1).
        sp.x.copy_from_slice(&[0.0, 5.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 3.0) * (x[0] - 3.0) + 1000.0 * x[1])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 1000.0;
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
                irow.copy_from_slice(&[0]);
                jcol.copy_from_slice(&[0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some(sol.x.to_vec());
        self.final_obj = Some(sol.obj_value);
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
        self.final_lambda = Some(sol.lambda.to_vec());
    }
}

/// Solve the problem with the given `nlp_scaling_method` and return
/// the user-visible finalize payload (x, z_l, z_u).
fn solve_with_scaling(method: &str) -> (Vec<Number>, Vec<Number>, Vec<Number>) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("nlp_scaling_method", method, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(SteepBoundPinned::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "nlp_scaling_method={method}: unexpected status {status:?}",
    );

    let user = tnlp_concrete.borrow();
    let x = user.final_x.clone().expect("finalize_solution not called");
    let obj = user.final_obj.unwrap();
    let z_l = user.final_z_l.clone().unwrap();
    let z_u = user.final_z_u.clone().unwrap();
    let lambda = user.final_lambda.clone().unwrap();

    // Sanity: same primal optimum either way, unscaled objective.
    assert_eq!(x.len(), 2);
    assert!(
        lambda.is_empty(),
        "m = 0 problem reported lambda {lambda:?}"
    );
    // x0 may sit a hair past 2.0 inside the relaxed bound
    // (`bound_relax_factor`), so allow a loose primal tolerance.
    assert!(
        (x[0] - 2.0).abs() < 1e-4 && x[1].abs() < 1e-4,
        "nlp_scaling_method={method}: x = {x:?} (expected (2, 0))",
    );
    assert!(
        (obj - 1.0).abs() < 1e-3,
        "nlp_scaling_method={method}: obj = {obj} (expected 1.0; must be unscaled)",
    );
    (x, z_l, z_u)
}

/// The bound duals handed to `TNLP::finalize_solution` must be
/// invariant under `nlp_scaling_method` (gradient-based vs none) and
/// match the analytic KKT values `z_u[0] = 2`, `z_l[1] = 1000`.
///
/// Before the 6d713c9 fix the scaled run leaked the algorithm-side
/// duals (`z_l[1] ≈ 100`, `z_u[0] ≈ 0.2` — multiplied by
/// `obj_scale_factor = 0.1`), so both the cross-run agreement and the
/// analytic checks below would fail.
#[test]
fn finalize_bound_duals_invariant_under_obj_scaling() {
    let (_, z_l_scaled, z_u_scaled) = solve_with_scaling("gradient-based");
    let (_, z_l_none, z_u_none) = solve_with_scaling("none");

    eprintln!(
        "gradient-based: z_l = {z_l_scaled:?}, z_u = {z_u_scaled:?}\n\
         none:           z_l = {z_l_none:?}, z_u = {z_u_none:?}",
    );

    // The unscaled run is the ground truth; its active duals must be
    // materially non-zero so a missing 1/obj_scale (factor 10 here)
    // cannot hide inside the tolerance.
    assert!(
        z_l_none[1] > 100.0,
        "z_l[1] (none) = {} — expected ~1000, problem no longer pins x1",
        z_l_none[1],
    );

    // Analytic KKT values, both runs.
    for (label, z_l, z_u) in [
        ("gradient-based", &z_l_scaled, &z_u_scaled),
        ("none", &z_l_none, &z_u_none),
    ] {
        assert!(
            (z_l[1] - 1000.0).abs() < 1e-2,
            "{label}: z_l[1] = {} (expected 1000; scaled dual would be 100)",
            z_l[1],
        );
        assert!(
            (z_u[0] - 2.0).abs() < 1e-4,
            "{label}: z_u[0] = {} (expected 2; scaled dual would be 0.2)",
            z_u[0],
        );
        // Inactive sides carry only the O(mu/slack) interior-point
        // residue.
        assert!(z_l[0].abs() < 1e-4, "{label}: z_l[0] = {}", z_l[0]);
        assert!(z_u[1].abs() < 1e-4, "{label}: z_u[1] = {}", z_u[1]);
    }

    // Cross-run agreement, elementwise: |a − b| ≤ atol + rtol·|b|.
    let close = |a: Number, b: Number| (a - b).abs() <= 1e-5 + 1e-6 * b.abs();
    for i in 0..2 {
        assert!(
            close(z_l_scaled[i], z_l_none[i]),
            "z_l[{i}] differs across scaling methods: {} (gradient-based) vs {} (none)",
            z_l_scaled[i],
            z_l_none[i],
        );
        assert!(
            close(z_u_scaled[i], z_u_none[i]),
            "z_u[{i}] differs across scaling methods: {} (gradient-based) vs {} (none)",
            z_u_scaled[i],
            z_u_none[i],
        );
    }
}
