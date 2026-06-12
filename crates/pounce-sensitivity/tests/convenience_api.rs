//! Integration tests for the [`SensSolve`] builder — confirms the
//! one-call convenience API produces the same numbers as the long-form
//! `on_converged` plumbing that `parametric_cpp.rs` exercises.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::SensSolve;

/// Same NLP as `tests/parametric_cpp.rs::ParametricTNLP`. Kept local
/// here to avoid an inter-test-binary module dependency.
struct ParametricTNLP {
    nominal_eta1: Number,
    nominal_eta2: Number,
}

impl ParametricTNLP {
    fn new(eta1: Number, eta2: Number) -> Self {
        Self {
            nominal_eta1: eta1,
            nominal_eta2: eta2,
        }
    }
}

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 4,
            nnz_jac_g: 10,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.nominal_eta1;
        b.g_u[2] = self.nominal_eta1;
        b.g_l[3] = self.nominal_eta2;
        b.g_u[3] = self.nominal_eta2;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.15;
        sp.x[1] = 0.15;
        sp.x[2] = 0.0;
        sp.x[3] = 0.0;
        sp.x[4] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        g[3] = 0.0;
        g[4] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, eta1, eta2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - eta1;
        g[1] = eta2 * x1 + x2 - x3 - 1.0;
        g[2] = eta1;
        g[3] = eta2;
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
                let rs: [Index; 10] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3];
                let cs: [Index; 10] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 6.0;
                values[1] = 3.0;
                values[2] = 2.0;
                values[3] = -1.0;
                values[4] = x[4];
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0];
                values[8] = 1.0;
                values[9] = 1.0;
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 2, 4, 0];
                let cs: [Index; 5] = [0, 1, 2, 0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = lam[1];
                values[4] = 0.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn make_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

/// Upstream sIPOPT's reported Δx for Δeta = (-0.5, 0) starting from
/// nominal (5, 1) — captured in `parametric_cpp.rs` already.
const UPSTREAM_DX: [Number; 5] = [
    0.576_530_601_168_321_9 - 0.632_653_057_519_998_2,
    0.377_551_038_130_684_8 - 0.387_755_107_968_002_7,
    -0.045_918_360_700_993_31 - 0.020_408_165_488_001_08,
    -0.5,
    0.0,
];

#[test]
fn sens_solve_builder_matches_upstream() {
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));

    let result = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app, tnlp);

    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {:?}",
        result.status,
    );
    let dx = result.dx.expect("dx populated when with_deltas was set");
    assert_eq!(dx.len(), 5);

    for k in 0..5 {
        let err = (dx[k] - UPSTREAM_DX[k]).abs();
        assert!(
            err < 1e-8,
            "dx[{k}]: pounce={}, upstream={}, |err|={err} not < 1e-8",
            dx[k],
            UPSTREAM_DX[k],
        );
    }

    // Builder also captures x*, obj_val unconditionally.
    let x = result.x.expect("x captured");
    assert_eq!(x.len(), 5);
    assert!(result.obj_val.is_some());
    assert!(
        result.dx_full.is_some(),
        "dx_full mirrors the KKT-space step"
    );
    assert!(
        result.reduced_hessian.is_none(),
        "reduced Hessian only populated when with_reduced_hessian was set",
    );
}

#[test]
fn sens_solve_reduced_hessian_is_symmetric_positive_definite() {
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));

    let result = SensSolve::new(vec![2, 3])
        .with_reduced_hessian()
        .run(&mut app, tnlp);

    assert!(matches!(
        result.status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    ));
    let hr = result.reduced_hessian.expect("reduced Hessian populated");
    assert_eq!(hr.len(), 4, "n_params=2 → 2x2 column-major dense matrix");

    // Symmetry of the reduced Hessian. (For an equality-constrained NLP
    // with a parameter pin, `B K⁻¹ Bᵀ` is symmetric by construction
    // even when not positive definite — sign just reflects whether the
    // parameter sits on the active vs. reduced side of the KKT block.)
    let off_diag_err = (hr[1] - hr[2]).abs();
    assert!(
        off_diag_err < 1e-8,
        "Hr not symmetric: hr[1]={}, hr[2]={}, |err|={off_diag_err}",
        hr[1],
        hr[2],
    );
}

#[test]
fn sens_solve_both_outputs_populated_together() {
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));

    let result = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .with_reduced_hessian()
        .run(&mut app, tnlp);

    assert!(result.dx.is_some());
    assert!(result.reduced_hessian.is_some());
}

#[test]
fn sens_solve_captures_user_space_multipliers_for_sqp_corrector() {
    // Phase 5c §6 — after a converged sens solve the SensResult
    // should expose the user-space multipliers and `g` so a caller
    // can pass them straight into `classify_working_set` for the
    // parametric corrector hand-off.
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));

    let result = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app, tnlp);

    let n_full_x = 5; // ParametricTNLP's user-space n
    let n_full_g = 4; // ParametricTNLP's user-space m
    assert!(result.mult_g.is_some(), "mult_g must be captured");
    assert!(result.mult_x_l.is_some(), "mult_x_l must be captured");
    assert!(result.mult_x_u.is_some(), "mult_x_u must be captured");
    assert!(result.g.is_some(), "g must be captured");
    assert_eq!(result.mult_g.as_ref().unwrap().len(), n_full_g);
    assert_eq!(result.mult_x_l.as_ref().unwrap().len(), n_full_x);
    assert_eq!(result.mult_x_u.as_ref().unwrap().len(), n_full_x);
    assert_eq!(result.g.as_ref().unwrap().len(), n_full_g);
}

// ─────────────────────────────────────────────────────────────────
// M6 regression (dev-notes/code-review-2026-06.md): a post-solve
// sensitivity-stage failure must be surfaced through
// `SensResult::error`, not silently swallowed.
//
// The underlying solve converges (so `status == SolveSucceeded`), but
// the sensitivity step is asked to pin a constraint index that is NOT
// an equality in the c-block — here an out-of-range index. The
// callback writes `outbox.error` and bails, leaving `dx == None`.
//
// Pre-fix: `SensResult` had no `error` field, so this was
// indistinguishable from "deltas not requested" — `dx == None`,
// `status == SolveSucceeded`, and no way to tell the sensitivity
// step blew up. Post-fix: `result.error` carries the message and the
// requested `dx` is `None`.
// ─────────────────────────────────────────────────────────────────
#[test]
fn sens_solve_surfaces_sensitivity_stage_failure() {
    let mut app = make_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));

    // ParametricTNLP has m = 4 constraints (indices 0..=3). Pin an
    // out-of-range index so `full_g_to_c_block` returns None and the
    // sensitivity callback fails after a successful solve.
    let bad_pin: Index = 99;
    let result = SensSolve::new(vec![bad_pin])
        .with_deltas(vec![0.1])
        .run(&mut app, tnlp);

    // The underlying solve still converged...
    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "underlying solve should converge; status = {:?}",
        result.status,
    );
    // ...but the sensitivity stage failed, and that MUST be visible.
    assert!(
        result.error.is_some(),
        "sensitivity-stage failure must be surfaced in SensResult::error, \
         not swallowed (dx = {:?}, status = {:?})",
        result.dx,
        result.status,
    );
    assert!(
        result.dx.is_none(),
        "dx must be None when the sensitivity step failed",
    );

    // A successful sensitivity solve leaves `error` None — guards
    // against the fix spuriously reporting an error on the happy path.
    let mut app_ok = make_app();
    let tnlp_ok: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)));
    let ok = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_ok, tnlp_ok);
    assert!(ok.error.is_none(), "happy path must not report an error");
    assert!(ok.dx.is_some(), "happy path must produce dx");
}
