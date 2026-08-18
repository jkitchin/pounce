//! HS071 with `hessian_approximation=limited-memory` (L-BFGS).
//!
//! Phase-8a smoke test for the dense-rebuild quasi-Newton path. We run
//! the same TNLP as `optimize_hs71.rs`, just with L-BFGS routed through
//! `LimMemQuasiNewtonUpdater` instead of `ExactHessianUpdater`. The
//! test still requires `eval_h` because the Phase-8a assembler reuses
//! the user-declared `SymTMatrix` sparsity to publish `data.w` (the
//! `LowRankUpdateSymMatrix` SMW shortcut, which removes the
//! `eval_h` requirement, lands as Phase-8b).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Hs071 {
    final_x: Option<[Number; 4]>,
    final_obj: Option<Number>,
}

impl TNLP for Hs071 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 8,
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[1.0; 4]);
        b.x_u.copy_from_slice(&[5.0; 4]);
        b.g_l.copy_from_slice(&[25.0, 40.0]);
        b.g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
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
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = x[1] * x[2] * x[3];
                values[1] = x[0] * x[2] * x[3];
                values[2] = x[0] * x[1] * x[3];
                values[3] = x[0] * x[1] * x[2];
                values[4] = 2.0 * x[0];
                values[5] = 2.0 * x[1];
                values[6] = 2.0 * x[2];
                values[7] = 2.0 * x[3];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda");
                let of = obj_factor;
                let l0 = lam[0];
                let l1 = lam[1];
                values[0] = of * (2.0 * x[3]) + l1 * 2.0;
                values[1] = of * x[3] + l0 * (x[2] * x[3]);
                values[2] = l1 * 2.0;
                values[3] = of * x[3] + l0 * (x[1] * x[3]);
                values[4] = l0 * (x[0] * x[3]);
                values[5] = l1 * 2.0;
                values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]);
                values[7] = of * x[0] + l0 * (x[0] * x[2]);
                values[8] = of * x[0] + l0 * (x[0] * x[1]);
                values[9] = l1 * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        if sol.x.len() == 4 {
            self.final_x = Some([sol.x[0], sol.x[1], sol.x[2], sol.x[3]]);
        }
        self.final_obj = Some(sol.obj_value);
    }
}

#[test]
fn hs071_solves_with_lbfgs() {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;

    let status = app.optimize_tnlp(tnlp);

    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );

    let stats = app.statistics();
    eprintln!(
        "HS71+LBFGS: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );

    let obj = stats.final_objective;
    assert!(
        (obj - 17.014017).abs() < 1e-4,
        "final_objective = {obj} (expected ~17.014017)",
    );

    let user = tnlp_concrete.borrow();
    let f_user = user.final_obj.unwrap();
    assert!(
        (f_user - 17.014017).abs() < 1e-4,
        "user-side final_obj = {f_user}",
    );
    let xs = user.final_x.unwrap();
    // Optimum is x* ≈ (1, 4.7430, 3.8211, 1.3794). L-BFGS may take more
    // iterations than exact-Hessian but should still converge to the
    // same x*.
    assert!((xs[0] - 1.0).abs() < 1e-3, "x[0]={}", xs[0]);
    assert!((xs[1] - 4.7430).abs() < 1e-2, "x[1]={}", xs[1]);
    assert!((xs[2] - 3.8211).abs() < 1e-2, "x[2]={}", xs[2]);
    assert!((xs[3] - 1.3794).abs() < 1e-2, "x[3]={}", xs[3]);
}

#[test]
fn hs071_solves_with_lbfgs_and_adaptive_mu() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    app.options_mut()
        .set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp_concrete = Rc::new(RefCell::new(Hs071::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&tnlp_concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    let stats = app.statistics();
    eprintln!(
        "HS71+LBFGS+adaptive: status={:?} iter={} obj={}",
        status, stats.iteration_count, stats.final_objective,
    );
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "unexpected status: {status:?}",
    );
    assert!(
        (stats.final_objective - 17.014017).abs() < 1e-4,
        "final_objective = {} (expected ~17.014017)",
        stats.final_objective,
    );
}

/// `recalc_y` must *change the solve*, not merely parse (#677).
///
/// The caution this guards against is spelled out in #551: "A read site
/// that parses a value and discards it is the same silent no-op this
/// whole line of work exists to kill, and it is indistinguishable from a
/// real fix by inspection. That test is the deliverable, not the line
/// that reads the field." `recalc_y` was refused as unimplemented until
/// #677, so nothing downstream of the option existed to test.
///
/// Asserting the *direction* of the change would be asserting a
/// preference — re-estimating `y` by least squares is not uniformly
/// better, which is exactly why pounce leaves it off by default. What
/// must hold is that the switch reaches the algorithm at all, so this
/// pins the trajectory moving, and pins the answer surviving it.
#[test]
fn recalc_y_changes_the_lbfgs_trajectory() {
    fn run(recalc: bool) -> (usize, Number) {
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_string_value("hessian_approximation", "limited-memory", true, true)
            .unwrap();
        app.options_mut()
            .set_string_value("recalc_y", if recalc { "yes" } else { "no" }, true, false)
            .unwrap();
        // Default 1e-6 leaves the gate shut on a short HS071 run; open
        // it so the feature actually fires within the solve.
        app.options_mut()
            .set_numeric_value("recalc_y_feas_tol", 1e2, true, false)
            .unwrap();
        app.initialize().unwrap();
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071::default())) as _;
        let status = app.optimize_tnlp(tnlp);
        let stats = app.statistics();
        eprintln!(
            "HS71+LBFGS recalc_y={recalc}: status={status:?} iter={} obj={}",
            stats.iteration_count, stats.final_objective,
        );
        assert!(
            matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "recalc_y={recalc} did not solve: {status:?}",
        );
        // Whatever the multipliers do, the primal answer must stand.
        assert!(
            (stats.final_objective - 17.014017).abs() < 1e-4,
            "recalc_y={recalc}: objective {} drifted",
            stats.final_objective,
        );
        (stats.iteration_count as usize, stats.final_objective)
    }

    let (it_off, _) = run(false);
    let (it_on, _) = run(true);
    assert_ne!(
        it_off, it_on,
        "recalc_y did not change the trajectory ({it_off} iterations either way) \
         — the option is parsed but not reaching the algorithm",
    );
}
