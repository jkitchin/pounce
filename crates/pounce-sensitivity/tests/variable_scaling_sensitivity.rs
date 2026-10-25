//! gh#486 stage 3 — the sensitivity accessors under a `user-scaling`
//! change of variables.
//!
//! Stage 2 taught the core to apply a per-variable `scaling_factor` by
//! substituting `x̃ = d ⊙ x` one level below the algorithm. That leaves
//! the converged KKT factor — which the sensitivity layer reads
//! directly, not through the TNLP chain — in scaled coordinates. Stage
//! 3 carries the factors into the natural-units translation, so every
//! accessor answers in the model's own units.
//!
//! The assertion shape throughout is **parity**: solve the same problem
//! twice, once with non-unit factors and once without, and require the
//! two to agree. That makes the property falsifiable without a
//! hand-derived expected value for each accessor, and it is the same
//! criterion issue #486 states for `core.scale_model` parity — the
//! solution and the unscaled duals agree, not the iterates.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use pounce_sensitivity::{SensSolve, Solver};

mod common;
use common::outside_engine::{Relaxation, recorded_solve, same_classification};

/// The `parametric_cpp` / `solver_session` fixture, plus a
/// `get_scaling_parameters` that hands back per-variable factors. Five
/// variables, five rows, two of them the parameter pins; `x[0..3]` are
/// bounded below (so the z_l block is non-empty and the bound-multiplier
/// conjugation is exercised) and row 4 is a one-sided inequality (so the
/// v/d blocks are too).
struct ParametricTNLP {
    nominal_eta1: Number,
    nominal_eta2: Number,
    /// Per-variable factors to report, or `None` to decline scaling.
    x_scaling: Option<[Number; 5]>,
}

impl ParametricTNLP {
    fn new(x_scaling: Option<[Number; 5]>) -> Self {
        Self {
            nominal_eta1: 5.0,
            nominal_eta2: 1.0,
            x_scaling,
        }
    }
}

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 5,
            nnz_jac_g: 11,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(d) = self.x_scaling else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(&d);
        *req.use_g_scaling = false;
        true
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
        b.g_l[4] = 0.0;
        b.g_u[4] = 1.0e19;
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

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::NonLinear);
        types[4] = Linearity::Linear;
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
        g[4] = x1;
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
                let rs: [Index; 11] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 4];
                let cs: [Index; 11] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4, 0];
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
                values[10] = 1.0;
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

/// Factors spanning six orders of magnitude, mixed above and below 1,
/// including an exact 1.0 so the "some variables untagged" case is
/// covered too.
const D: [Number; 5] = [1e3, 1e-2, 1.0, 5.0, 1e-3];

fn make_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    // Both arms run under `user-scaling` so the ONLY difference between
    // them is the variable factors: the option itself, the objective
    // factor and the row factors are identical.
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    // classify_activity refuses a relaxed-bound solve.
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

fn solved_session(x_scaling: Option<[Number; 5]>) -> Solver {
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(x_scaling)));
    let mut solver = Solver::new(make_app(), tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}"
    );
    solver
}

#[track_caller]
fn assert_close(what: &str, got: &[Number], want: &[Number], tol: Number) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(
            err < tol,
            "{what}[{k}]: scaled={g}, plain={w}, rel err {err} not < {tol}"
        );
    }
}

/// The factors reach the sensitivity layer at all — without this the
/// parity assertions below could pass by the wrapper never installing.
#[test]
fn the_session_reports_the_factors_the_solve_ran_under() {
    let plain = solved_session(None);
    assert_eq!(
        plain.variable_scaling().expect("converged"),
        None,
        "an unscaled solve must report no change of variables"
    );

    let scaled = solved_session(Some(D));
    assert_eq!(
        scaled.variable_scaling().expect("converged"),
        Some(D.to_vec()),
        "the factors reported must be the ones the wrapper applied"
    );
}

/// The converged iterate is the model's own `x`, not the algorithm's
/// `x̃`. This capture reads the iterate rather than the
/// `finalize_solution` payload — the predicate #529 identified as the
/// one that matters.
#[test]
fn the_converged_iterate_is_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    let want = plain.converged().expect("converged").x.clone();
    let got = scaled.converged().expect("converged").x.clone();
    assert_close("x", &got, &want, 1e-7);
}

/// `∂x*/∂p` — the accessor pyomo-pounce's `gradient()` / `estimate()`
/// are built on.
#[test]
fn the_parametric_step_is_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    for deltas in [[-0.5, 0.0], [0.0, 0.25], [0.3, -0.2]] {
        let want = plain.parametric_step(&[2, 3], &deltas).expect("plain step");
        let got = scaled
            .parametric_step(&[2, 3], &deltas)
            .expect("scaled step");
        assert_close(&format!("dx for {deltas:?}"), &got, &want, 1e-7);
    }
}

/// The full compound step, so the multiplier sensitivities are covered
/// as well as the primal ones: `λ` is invariant under the substitution
/// and the bound multipliers are not, and only a full-vector comparison
/// distinguishes "both right" from "both untouched".
#[test]
fn the_full_kkt_step_is_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    let want = plain
        .parametric_step_full(&[2, 3], &[-0.5, 0.1])
        .expect("plain full step");
    let got = scaled
        .parametric_step_full(&[2, 3], &[-0.5, 0.1])
        .expect("scaled full step");
    assert_eq!(
        plain.block_dims(),
        scaled.block_dims(),
        "a change of variables must not change the KKT layout"
    );
    assert_close("dx_full", &got, &want, 1e-6);
}

/// `-inv(H_R)` is the parameter covariance, which is what
/// `covariance()` / `information()` return.
#[test]
fn the_reduced_hessian_is_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    let want = plain
        .compute_reduced_hessian(&[2, 3], 1.0)
        .expect("plain H_R");
    let got = scaled
        .compute_reduced_hessian(&[2, 3], 1.0)
        .expect("scaled H_R");
    assert_close("H_R", &got, &want, 1e-6);
}

/// A back-solve against a hand-packed RHS: the same `K⁻¹` the
/// accessors above ride on, asserted directly so a failure localizes to
/// the conjugation rather than to a caller.
#[test]
fn the_back_solve_is_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    let dim = plain.kkt_dim().expect("converged");
    // A RHS that touches every block, and is not symmetric under any
    // permutation of them, so a block-swap would show up.
    let rhs: Vec<Number> = (0..dim).map(|i| 1.0 + 0.25 * (i as Number)).collect();
    let mut want = vec![0.0; dim];
    let mut got = vec![0.0; dim];
    plain.kkt_solve(&rhs, &mut want).expect("plain solve");
    scaled.kkt_solve(&rhs, &mut got).expect("scaled solve");
    assert_close("K^-1 rhs", &got, &want, 1e-6);
}

/// The exact-Hessian product and a constraint row's gradient: both read
/// the model's own matrices out of the converged state rather than
/// riding the factor, so they carry the substitution separately.
#[test]
fn the_model_matrix_accessors_are_in_the_models_units() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));

    for v in [
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.3, -1.2, 0.7, 2.0, -0.4],
    ] {
        let want = plain.hessian_vec(&v).expect("plain Hv");
        let got = scaled.hessian_vec(&v).expect("scaled Hv");
        assert_close("H v", &got, &want, 1e-6);
    }

    for row in 0..5 {
        let want = plain.row_normal(row).expect("plain row_normal");
        let got = scaled.row_normal(row).expect("scaled row_normal");
        assert_close(&format!("row_normal({row})"), &got, &want, 1e-6);
    }
}

/// Activity classification: the statuses must not move, and the
/// exported `Σ` must be the model's own. The status is the part a
/// non-uniform `d` could shift without any single ratio being wrong —
/// the identification floor is one number shared across entries — so
/// asserting the statuses is not redundant with asserting `Σ`.
#[test]
fn activity_classification_is_unmoved_by_the_change_of_variables() {
    let plain = solved_session(None);
    let scaled = solved_session(Some(D));
    let want = plain.classify_activity().expect("plain classify");
    let got = scaled.classify_activity().expect("scaled classify");

    assert_eq!(got.var_status, want.var_status, "variable statuses");
    assert_eq!(got.row_status, want.row_status, "row statuses");
    assert_eq!(got.var_q_sign, want.var_q_sign, "variable curvature signs");
    assert_eq!(got.row_q_sign, want.row_q_sign, "row curvature signs");
    assert_eq!(
        got.var_off_central_path, want.var_off_central_path,
        "off-central-path flags are products s·z, invariant under d"
    );
    // Sigma and the ratio are BARRIER quantities, proportional to the
    // mu the run stopped at (Sigma = z/s, and complementarity pins
    // s.z ~ mu), and the two runs are different Newton trajectories
    // that terminate at different mu -- here 9.1e-10 against 2.5e-9.
    // So the invariant is Sigma/mu, not Sigma: dividing it out
    // compares the geometry both runs found rather than where each
    // one happened to stop. The units are still fully asserted, since
    // a missed `d^2` on the export would land these entries a factor
    // 1e6 / 1e-4 out -- six orders past this tolerance.
    let per_mu = |v: &[Number], mu: Number| -> Vec<Number> { v.iter().map(|s| s / mu).collect() };
    assert_close(
        "var_sigma/mu",
        &per_mu(&got.var_sigma, got.mu),
        &per_mu(&want.var_sigma, want.mu),
        1e-3,
    );
    assert_close(
        "row_sigma/mu",
        &per_mu(&got.row_sigma, got.mu),
        &per_mu(&want.row_sigma, want.mu),
        1e-3,
    );
    // NaN where nothing was classified, so compare the finite entries.
    for (k, (&g, &w)) in got.var_ratio.iter().zip(want.var_ratio.iter()).enumerate() {
        assert_eq!(g.is_nan(), w.is_nan(), "var_ratio[{k}] classified-ness");
        if !g.is_nan() {
            let (g, w) = (g / got.mu, w / want.mu);
            let err = (g - w).abs() / w.abs().max(1.0);
            assert!(err < 1e-3, "var_ratio[{k}]/mu: {g} vs {w}");
        }
    }
}

/// The `SensSolve` builder is the other consumer of the same converged
/// state, and it captures the iterate and the bound multipliers
/// itself — so it needs its own parity check, not just the session's.
#[test]
fn the_sens_solve_builder_reports_in_the_models_units() {
    let run = |x_scaling: Option<[Number; 5]>| {
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ParametricTNLP::new(x_scaling)));
        let mut app = make_app();
        let out = SensSolve::new(vec![2, 3])
            .with_deltas(vec![-0.5, 0.0])
            .with_reduced_hessian()
            .run(&mut app, tnlp);
        assert!(out.error.is_none(), "sensitivity stage: {:?}", out.error);
        out
    };
    let want = run(None);
    let got = run(Some(D));

    assert_close(
        "x",
        got.x.as_ref().expect("x"),
        want.x.as_ref().expect("x"),
        1e-7,
    );
    assert_close(
        "dx",
        got.dx.as_ref().expect("dx"),
        want.dx.as_ref().expect("dx"),
        1e-7,
    );
    assert_close(
        "H_R",
        got.reduced_hessian.as_ref().expect("H_R"),
        want.reduced_hessian.as_ref().expect("H_R"),
        1e-6,
    );
    assert_close(
        "mult_g",
        got.mult_g.as_ref().expect("mult_g"),
        want.mult_g.as_ref().expect("mult_g"),
        1e-6,
    );
    assert_close(
        "mult_x_L",
        got.mult_x_l.as_ref().expect("mult_x_l"),
        want.mult_x_l.as_ref().expect("mult_x_l"),
        1e-6,
    );
    assert_close(
        "mult_x_U",
        got.mult_x_u.as_ref().expect("mult_x_u"),
        want.mult_x_u.as_ref().expect("mult_x_u"),
        1e-6,
    );
}

// ---------------------------------------------------------------------------
// The same classification over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and asserts that the classification over an engine built from
// that report is the same as pounce's.

/// Both arms against the outside engine, which works in the model's units
/// and so gives one classification for both.
#[test]
fn activity_classification_is_unmoved_by_the_change_of_variables_over_an_outside_engine() {
    for scaling in [None, Some(D)] {
        let (solver, model, status) = recorded_solve(make_app(), ParametricTNLP::new(scaling));
        assert!(
            matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "solve failed: {status:?}",
        );
        same_classification(
            &format!("scaling {scaling:?}"),
            &solver,
            &model,
            Relaxation::None,
        );
    }
}
