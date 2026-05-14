//! End-to-end integration tests for the ℓ₁-exact penalty-barrier wrapper
//! through `IpoptApplication::optimize_tnlp` (pounce#10 Phase 2).
//!
//! Confirms the back-projection contract: the user's `finalize_solution`
//! callback receives original-space `(x*, g*, obj_value)` even when the
//! IPM solved the augmented NLP. Multiplier mapping passes the equality-
//! row entries straight through (slack contributions to `c(x)` are
//! linear so the same scalar dual is correct in both spaces).

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

/// Captured per-solve fields from the user's `finalize_solution`. Used
/// by the integration tests to verify back-projection behavior.
#[derive(Default, Clone)]
struct CapturedSolution {
    x: Vec<Number>,
    g: Vec<Number>,
    lambda: Vec<Number>,
    obj_value: Number,
}

/// Minimal equality-only TNLP exercising the wrapper end-to-end.
///
/// `min x[0]^2 + x[1]^2  s.t.  x[0] + x[1] = 1`. Optimum at
/// `(0.5, 0.5)`, `f* = 0.5`. With one equality row and `n_orig = 2`,
/// the wrapper introduces `2 * m_eq = 2` slack variables, taking the
/// augmented variable count to 4.
struct EqOnly {
    captured: Rc<RefCell<Option<CapturedSolution>>>,
}

impl EqOnly {
    fn new() -> (Self, Rc<RefCell<Option<CapturedSolution>>>) {
        let captured = Rc::new(RefCell::new(None));
        (Self { captured: Rc::clone(&captured) }, captured)
    }
}

impl TNLP for EqOnly {
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
        b.x_l.copy_from_slice(&[-1.0e19, -1.0e19]);
        b.x_u.copy_from_slice(&[1.0e19, 1.0e19]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
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
                irow[0] = 0; jcol[0] = 0;
                irow[1] = 0; jcol[1] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = 1.0;
                true
            }
        }
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
                irow[0] = 0; jcol[0] = 0;
                irow[1] = 1; jcol[1] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        *self.captured.borrow_mut() = Some(CapturedSolution {
            x: sol.x.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
            obj_value: sol.obj_value,
        });
    }
}

/// Build an `IpoptApplication` configured for a quiet small-problem
/// solve. Sub-second on the EqOnly fixture.
fn build_app(l1_enabled: bool, rho: Number) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_integer_value("print_level", 0, true, false);
        let _ = opts.set_numeric_value("tol", 1e-10, true, false);
        let _ = opts.set_integer_value("max_iter", 200, true, false);
        if l1_enabled {
            let _ = opts.set_string_value("l1_exact_penalty_barrier", "yes", true, false);
            let _ = opts.set_numeric_value("l1_penalty_init", rho, true, false);
        }
    }
    app.initialize().expect("initialize");
    app
}

#[test]
fn flag_off_solves_eq_only_to_known_optimum() {
    let (tnlp, captured) = EqOnly::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app(false, 1.0);
    let status = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded
                            | ApplicationReturnStatus::SolvedToAcceptableLevel),
        "flag-off status = {:?}", status);
    let cap = captured.borrow().clone().expect("finalize_solution called");
    assert_eq!(cap.x.len(), 2, "flag-off x length");
    assert!((cap.x[0] - 0.5).abs() < 1e-6, "flag-off x[0] = {}", cap.x[0]);
    assert!((cap.x[1] - 0.5).abs() < 1e-6, "flag-off x[1] = {}", cap.x[1]);
    assert!((cap.obj_value - 0.5).abs() < 1e-8, "flag-off obj = {}", cap.obj_value);
    // Diagnostic: pounce#11 (fixed 2026-05-14) wired the multiplier
    // lift on the OrigIpoptNlp path. For this fixture the analytic
    // |λ| = 1.0 (∇L = 2x − λ·1 = 0 with x[0]+x[1] = 1 ⇒ λ = 1);
    // pounce reports λ ≈ −1.0 with `min f + λ·c` sign convention.
    eprintln!("flag-off captured lambda = {:?}", cap.lambda);
    assert!(
        cap.lambda.iter().any(|v| v.abs() > 0.1),
        "post-#11: bare equality solve must report non-zero λ; got {:?}",
        cap.lambda,
    );
}

#[test]
fn flag_on_solution_x_truncated_to_n_orig() {
    let (tnlp, captured) = EqOnly::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app(true, 1.0);
    let _ = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    let cap = captured.borrow().clone().expect("finalize_solution called");
    // Phase-1 wrapper back-projection truncates `x` to n_orig = 2 even
    // though the IPM solved the augmented (n_orig + 2*m_eq = 4)-variable
    // problem. The user must never see slack variables in their
    // finalize_solution callback.
    assert_eq!(cap.x.len(), 2, "x must be truncated to n_orig (got {} entries)", cap.x.len());
}

#[test]
fn flag_on_objective_excludes_penalty_term() {
    let (tnlp, captured) = EqOnly::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let rho = 2.5; // arbitrary > 1 so penalty > optimum f if leaked
    let mut app = build_app(true, rho);
    let _ = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    let cap = captured.borrow().clone().expect("finalize_solution called");
    // The user's reported objective MUST be the original `f(x*) = 0.5`,
    // NOT the augmented `f(x*) + ρ·Σ(p+n)`. With Σ(p+n) ≈ 0 at the
    // optimum (slacks collapse), the augmented and original objectives
    // coincide here, but the back-projection guarantee must hold even
    // when Σ(p+n) > 0 (Phase 3 will exercise that path on infeasible
    // problems via the slack-collapse / honest-infeasibility check).
    assert!((cap.obj_value - 0.5).abs() < 1e-6,
        "reported obj must be original f(x*) = 0.5, got {}", cap.obj_value);
}

#[test]
fn flag_on_constraint_value_excludes_slack_contribution() {
    let (tnlp, captured) = EqOnly::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app(true, 1.0);
    let _ = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    let cap = captured.borrow().clone().expect("finalize_solution called");
    // The reported `g[i]` is `c_i(x_trunc)` ONLY — slack contributions
    // are stripped via the wrapper recomputing inner.eval_g on the
    // truncated x. For EqOnly the augmented IPM converges where
    // `x[0]+x[1]−p+n = 1` exactly; truncating to (x[0], x[1]) drops
    // the `−p+n` correction so the inner constraint can be off by
    // `O(slack)` even at high IPM precision. Tolerance set generously;
    // Phase 3's slack-collapse check will drive Σ(p+n) → 0 on
    // feasible problems and tighten this naturally.
    assert_eq!(cap.g.len(), 1);
    assert!((cap.g[0] - 1.0).abs() < 1e-3,
        "reported g[0] = {} (expected ≈ 1.0; gap = {:.2e})",
        cap.g[0], (cap.g[0] - 1.0).abs());
}

#[test]
fn flag_on_lambda_length_and_passthrough() {
    // The wrapper reports `Solution.lambda` with the same length as
    // the inner constraint vector (the wrapper does not add new
    // constraint rows, only new primal slack variables). pounce#11
    // (fixed 2026-05-14) wired the multiplier lift, so for the bare
    // EqOnly fixture both flag states now report non-zero |λ| ≈ 1.
    //
    // The Phase-2 contract is just that the wrapper preserves
    // whatever pounce reports for the bare un-wrapped solve:
    //  (a) the lambda vector length the user sees is `m_inner`
    //      (wrapper adds no constraint rows);
    //  (b) the values match what the bare flag-off solve reports
    //      element-wise within tolerance.
    //
    // Phase 3's BNW outer loop reads `‖y_eq‖∞` to drive the ρ
    // steering update; with #11 landed that signal is now live
    // rather than driven to zero by the pre-fix stub.
    let (tnlp_off, captured_off) = EqOnly::new();
    let tnlp_off_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_off));
    let mut app_off = build_app(false, 1.0);
    let _ = app_off.optimize_tnlp(Rc::clone(&tnlp_off_rc));
    let cap_off = captured_off.borrow().clone().expect("flag-off finalize");

    let (tnlp_on, captured_on) = EqOnly::new();
    let tnlp_on_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_on));
    let mut app_on = build_app(true, 1.0);
    let _ = app_on.optimize_tnlp(Rc::clone(&tnlp_on_rc));
    let cap_on = captured_on.borrow().clone().expect("flag-on finalize");

    assert_eq!(cap_on.lambda.len(), 1, "wrapper must not add constraint rows");
    assert_eq!(
        cap_on.lambda.len(),
        cap_off.lambda.len(),
        "lambda length must match flag-off"
    );
    for i in 0..cap_on.lambda.len() {
        assert!(
            (cap_on.lambda[i] - cap_off.lambda[i]).abs() < 1e-6,
            "lambda[{}] must pass through (flag-off {}, flag-on {})",
            i, cap_off.lambda[i], cap_on.lambda[i],
        );
    }
}

/// Sanity: the wrapper should not run on a TNLP with no equality rows
/// — there are no slack variables to introduce. The test asserts the
/// user's `n_orig` shows up unchanged in the captured solution.
#[test]
fn flag_on_no_op_when_no_equality_rows() {
    /// Inequality-only TNLP: `min (x − 3)^2  s.t.  x ≤ 10`.
    /// Optimum at `x* = 3`, `f* = 0`.
    struct IneqOnly {
        captured: Rc<RefCell<Option<CapturedSolution>>>,
    }
    impl TNLP for IneqOnly {
        fn get_nlp_info(&mut self) -> Option<NlpInfo> {
            Some(NlpInfo {
                n: 1, m: 1, nnz_jac_g: 1, nnz_h_lag: 1,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l[0] = -1e19; b.x_u[0] = 1e19;
            b.g_l[0] = -1e19; b.g_u[0] = 10.0;
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x[0] = 0.0; true
        }
        fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
            Some((x[0] - 3.0).powi(2))
        }
        fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = 2.0 * (x[0] - 3.0); true
        }
        fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
            g[0] = x[0]; true
        }
        fn eval_jac_g(
            &mut self,
            _x: Option<&[Number]>,
            _: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0; jcol[0] = 0;
                    true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 1.0; true
                }
            }
        }
        fn eval_h(
            &mut self,
            _x: Option<&[Number]>,
            _: bool,
            obj_factor: Number,
            _lambda: Option<&[Number]>,
            _: bool,
            mode: SparsityRequest<'_>,
        ) -> bool {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 0; jcol[0] = 0; true
                }
                SparsityRequest::Values { values } => {
                    values[0] = 2.0 * obj_factor; true
                }
            }
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
            *self.captured.borrow_mut() = Some(CapturedSolution {
                x: sol.x.to_vec(),
                g: sol.g.to_vec(),
                lambda: sol.lambda.to_vec(),
                obj_value: sol.obj_value,
            });
        }
    }
    let captured = Rc::new(RefCell::new(None));
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(IneqOnly {
        captured: Rc::clone(&captured),
    }));
    let mut app = build_app(true, 1.0);
    let _ = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    let cap = captured.borrow().clone().expect("finalize_solution called");
    // n_orig = 1, no equality rows, wrapper should be a no-op.
    assert_eq!(cap.x.len(), 1);
    assert!((cap.x[0] - 3.0).abs() < 1e-4, "x* should be ~3, got {}", cap.x[0]);
}

// ---------- Phase-3 BNW outer loop + honest-infeasibility tests ----------

/// Truly-infeasible TNLP — empty feasible set:
///   min 0  s.t.  x[0] + x[1] = 1,  x[0]² + x[1]² = 0,  x ∈ ℝ²
/// The second equality forces `x = 0`, contradicting the first
/// equality. The Phase-3 honest-infeasibility upgrade should catch
/// this: the BNW loop pushes ρ up to `l1_penalty_max`, the slacks
/// fail to collapse (one of them stays at ~1.0 absorbing the
/// constraint mismatch), and the application status is overridden
/// from `SolveSucceeded` to `InfeasibleProblemDetected`.
struct BurkeHanLike {
    captured: Rc<RefCell<Option<CapturedSolution>>>,
}
impl BurkeHanLike {
    fn new() -> (Self, Rc<RefCell<Option<CapturedSolution>>>) {
        let captured = Rc::new(RefCell::new(None));
        (Self { captured: Rc::clone(&captured) }, captured)
    }
}
impl TNLP for BurkeHanLike {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2, m: 2,
            nnz_jac_g: 4, nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1.0e19, -1.0e19]);
        b.x_u.copy_from_slice(&[1.0e19, 1.0e19]);
        b.g_l.copy_from_slice(&[1.0, 0.0]);
        b.g_u.copy_from_slice(&[1.0, 0.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]); true
    }
    fn eval_f(&mut self, _x: &[Number], _: bool) -> Option<Number> { Some(0.0) }
    fn eval_grad_f(&mut self, _x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = 0.0; g[1] = 0.0; true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        g[1] = x[0] * x[0] + x[1] * x[1];
        true
    }
    fn eval_jac_g(
        &mut self,
        x: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0; jcol[0] = 0;
                irow[1] = 0; jcol[1] = 1;
                irow[2] = 1; jcol[2] = 0;
                irow[3] = 1; jcol[3] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values call needs x");
                values[0] = 1.0;
                values[1] = 1.0;
                values[2] = 2.0 * x[0];
                values[3] = 2.0 * x[1];
                true
            }
        }
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0; jcol[0] = 0;
                irow[1] = 1; jcol[1] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                // f has zero Hessian; only the second constraint contributes
                // 2·λ[1] on each diagonal entry.
                let lam = lambda.expect("values call needs lambda");
                values[0] = 2.0 * lam[1];
                values[1] = 2.0 * lam[1];
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        *self.captured.borrow_mut() = Some(CapturedSolution {
            x: sol.x.to_vec(),
            g: sol.g.to_vec(),
            lambda: sol.lambda.to_vec(),
            obj_value: sol.obj_value,
        });
    }
}

#[test]
fn bnw_outer_loop_runs_to_completion() {
    // Smoke test for the Phase-3 outer loop: with the wrapper on and
    // a feasible problem, the BNW driver should succeed within a few
    // outer iters and the user must receive a back-projected
    // original-space x of length `n_orig`.
    let (tnlp, captured) = EqOnly::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app(true, 1.0);
    let status = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded
                            | ApplicationReturnStatus::SolvedToAcceptableLevel),
        "BNW outer-loop status = {:?}", status);
    let cap = captured.borrow().clone().expect("finalize_solution called");
    assert_eq!(cap.x.len(), 2);
    assert!((cap.x[0] - 0.5).abs() < 1e-4);
    assert!((cap.x[1] - 0.5).abs() < 1e-4);
    assert!((cap.obj_value - 0.5).abs() < 1e-4);
}

#[test]
fn infeasible_problem_upgrades_to_infeasibility_detected() {
    // BurkeHanLike has empty feasible set. The BNW outer loop pushes
    // ρ up to its cap; slacks fail to collapse; the honest-infeasibility
    // upgrade fires and `InfeasibleProblemDetected` is reported.
    let (tnlp, _captured) = BurkeHanLike::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app(true, 1.0);
    // Cap ρ at a moderate value so the test runs in reasonable time;
    // the geometric escalation factor of 8 hits this within ~3 iters.
    {
        let opts = app.options_mut();
        let _ = opts.set_numeric_value("l1_penalty_max", 1.0e4, true, false);
        let _ = opts.set_integer_value("l1_penalty_max_outer_iter", 5, true, false);
    }
    let status = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    assert!(
        matches!(status, ApplicationReturnStatus::InfeasibleProblemDetected),
        "expected InfeasibleProblemDetected, got {:?}",
        status,
    );
}

#[test]
fn flag_on_does_not_regress_well_conditioned_problem() {
    // For a problem where the un-wrapped IPM converges, Phase 3's BNW
    // ρ-escalation should land in the same basin (with slack-collapse
    // termination), not regress to a far point. Phase 2 saw this
    // regression at fixed ρ = 1; Phase 3 fixes it via the BNW loop.
    let (tnlp_off, captured_off) = EqOnly::new();
    let tnlp_off_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_off));
    let mut app_off = build_app(false, 1.0);
    let _ = app_off.optimize_tnlp(Rc::clone(&tnlp_off_rc));
    let cap_off = captured_off.borrow().clone().expect("off");

    let (tnlp_on, captured_on) = EqOnly::new();
    let tnlp_on_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_on));
    let mut app_on = build_app(true, 1.0);
    let _ = app_on.optimize_tnlp(Rc::clone(&tnlp_on_rc));
    let cap_on = captured_on.borrow().clone().expect("on");

    // Same basin: x* and obj match to a reasonable tolerance.
    for i in 0..cap_off.x.len() {
        assert!((cap_off.x[i] - cap_on.x[i]).abs() < 1e-3,
            "x[{}] differs: off {} vs on {}", i, cap_off.x[i], cap_on.x[i]);
    }
    assert!((cap_off.obj_value - cap_on.obj_value).abs() < 1e-3,
        "obj differs: off {} vs on {}", cap_off.obj_value, cap_on.obj_value);
}

// ---------- Phase 3.5 auto-fallback tests ----------

/// Build an `IpoptApplication` with auto-fallback enabled and the
/// wrapper opt-in OFF (the auto-fallback path applies the wrapper on
/// retry without the user having to set it).
fn build_app_with_fallback(rho_init: Number) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_integer_value("print_level", 0, true, false);
        let _ = opts.set_numeric_value("tol", 1e-10, true, false);
        let _ = opts.set_integer_value("max_iter", 200, true, false);
        let _ = opts.set_string_value(
            "l1_fallback_on_restoration_failure",
            "yes",
            true,
            false,
        );
        let _ = opts.set_numeric_value("l1_penalty_init", rho_init, true, false);
        // Keep the retry's outer-iter budget small so the test runs fast.
        let _ = opts.set_integer_value("l1_penalty_max_outer_iter", 5, true, false);
        let _ = opts.set_numeric_value("l1_penalty_max", 1.0e4, true, false);
    }
    app.initialize().expect("initialize");
    app
}

#[test]
fn auto_fallback_no_op_when_first_attempt_succeeds() {
    // EqOnly is feasible; the standard solve returns Solve_Succeeded
    // so the fallback trigger does not fire. Result must be identical
    // to the bare flag-off path.
    let (tnlp_off, captured_off) = EqOnly::new();
    let tnlp_off_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_off));
    let mut app_off = build_app(false, 1.0);
    let status_off = app_off.optimize_tnlp(Rc::clone(&tnlp_off_rc));

    let (tnlp_fb, captured_fb) = EqOnly::new();
    let tnlp_fb_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp_fb));
    let mut app_fb = build_app_with_fallback(1.0);
    let status_fb = app_fb.optimize_tnlp(Rc::clone(&tnlp_fb_rc));

    assert_eq!(
        std::mem::discriminant(&status_off),
        std::mem::discriminant(&status_fb),
        "fallback should not change status on a success path: off {:?} vs fb {:?}",
        status_off, status_fb
    );
    let cap_off = captured_off.borrow().clone().expect("off finalize");
    let cap_fb = captured_fb.borrow().clone().expect("fb finalize");
    for i in 0..cap_off.x.len() {
        assert!((cap_off.x[i] - cap_fb.x[i]).abs() < 1e-6,
            "x[{}] should be identical: off {} vs fb {}",
            i, cap_off.x[i], cap_fb.x[i]);
    }
}

#[test]
fn auto_fallback_preserves_status_on_truly_infeasible_problem() {
    // BurkeHanLike has empty feasible set. The standard solve hits
    // Infeasible_Problem_Detected (or similar non-success). Fallback
    // fires; retry also concludes infeasibility. Promotion rule
    // does NOT fire (retry status != Solve_Succeeded), so the
    // original status is returned.
    let (tnlp, _captured) = BurkeHanLike::new();
    let tnlp_rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let mut app = build_app_with_fallback(1.0);
    let status = app.optimize_tnlp(Rc::clone(&tnlp_rc));
    // The first attempt and the retry both end in non-success on
    // BurkeHanLike. Acceptable terminal statuses are the trigger
    // set members — assert we don't see Solve_Succeeded (which
    // would be a false-positive promotion).
    assert!(
        !matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "fallback must not promote when retry didn't succeed; got {:?}",
        status,
    );
}

// Quiet warnings on Index — kept import for future Phase-3 tests
// that read the equality-row index.
#[allow(dead_code)]
fn _index_marker(_i: Index) {}
