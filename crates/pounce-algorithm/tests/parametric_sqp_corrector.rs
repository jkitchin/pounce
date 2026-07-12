//! Phase 5c §6/§7 — end-to-end test of the parametric continuation
//! pattern: solve cold IPM at p₀, capture the converged iterate via
//! the user TNLP's `finalize_solution`, perturb p, build an SQP
//! warm-start with [`pounce_algorithm::sqp::classify_working_set`],
//! and run the SQP corrector to first-order KKT at p₀ + Δp.
//!
//! Validates the parametric-corrector handoff documented in the
//! `pounce-sensitivity/src/convenience.rs` module header. No
//! `pounce-sensitivity` dependency needed for this test — the
//! linear predictor step is the user's analytic ∂x*/∂p, not a
//! Schur computation, because the fixture is simple enough to
//! admit a closed-form.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::sqp::{SqpIterates, classify_working_set};
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

/// `min ½‖x − p‖²   s.t.   sum(x) = 1,   x ≥ 0`
///
/// Closed form: water-filling onto the simplex
/// `x_i = max(p_i − λ, 0)` with `λ` such that `sum(x) = 1`.
/// As `p` varies smoothly the active set (which `x_i = 0`) can
/// change discretely; the small-Δp regime keeps it fixed, which
/// is where the SQP-corrector warm-start pays off.
struct SimplexProj {
    p: std::rc::Rc<std::cell::RefCell<Vec<Number>>>,
    finalize_sink: std::rc::Rc<std::cell::RefCell<Option<FinalizedIterate>>>,
}

#[derive(Clone)]
struct FinalizedIterate {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
    g: Vec<Number>,
}

impl SimplexProj {
    fn new(p: Vec<Number>) -> Self {
        Self {
            p: std::rc::Rc::new(std::cell::RefCell::new(p)),
            finalize_sink: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }
    fn n(&self) -> usize {
        self.p.borrow().len()
    }
}

impl TNLP for SimplexProj {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        let n = self.n() as Index;
        Some(NlpInfo {
            n,
            m: 1,
            nnz_jac_g: n,
            nnz_h_lag: n,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for i in 0..self.n() {
            b.x_l[i] = 0.0;
            b.x_u[i] = 1.0e20;
        }
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let inv = 1.0 / self.n() as Number;
        for v in sp.x.iter_mut() {
            *v = inv;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let p = self.p.borrow();
        Some(
            0.5 * x
                .iter()
                .zip(p.iter())
                .map(|(xi, pi)| (xi - pi) * (xi - pi))
                .sum::<Number>(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        let p = self.p.borrow();
        for (g, (xi, pi)) in grad.iter_mut().zip(x.iter().zip(p.iter())) {
            *g = xi - pi;
        }
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x.iter().sum();
        true
    }
    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let n = self.n();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..n {
                    irow[i] = 0;
                    jcol[i] = i as Index;
                }
            }
            SparsityRequest::Values { values, .. } => {
                for v in values.iter_mut() {
                    *v = 1.0;
                }
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
        let n = self.n();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for i in 0..n {
                    irow[i] = i as Index;
                    jcol[i] = i as Index;
                }
            }
            SparsityRequest::Values { values, .. } => {
                for v in values.iter_mut() {
                    *v = obj_factor;
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.finalize_sink.borrow_mut() = Some(FinalizedIterate {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
            g: sol.g.to_vec(),
        });
    }
}

#[test]
fn ipm_then_sqp_corrector_recovers_exact_perturbed_optimum() {
    // 3-D simplex projection. Parameter p₀ = (0.5, 0.4, -0.1) ⇒
    // x* = (0.55, 0.45, 0), x₃ active at lower bound.
    let n = 3;
    let p0 = vec![0.5, 0.4, -0.1];
    let tnlp = std::rc::Rc::new(std::cell::RefCell::new(SimplexProj::new(p0.clone())));
    let p_handle = tnlp.borrow().p.clone();
    let finalize_handle = tnlp.borrow().finalize_sink.clone();

    // --- Step 1: cold IPM solve at p₀. ---
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("print_level 0\n").unwrap();
    let status = app.optimize_tnlp(tnlp.clone());
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let ipm_iter = finalize_handle.borrow().clone().unwrap();
    // Verify x* matches the closed form.
    assert!((ipm_iter.x[0] - 0.55).abs() < 1e-6);
    assert!((ipm_iter.x[1] - 0.45).abs() < 1e-6);
    assert!(ipm_iter.x[2].abs() < 1e-6);

    // --- Step 2: classify the active set at the IPM-converged point. ---
    // Build λ_x = z_l − z_u (the SQP-convention packed bound multiplier).
    let lambda_x: Vec<Number> = ipm_iter
        .z_l
        .iter()
        .zip(ipm_iter.z_u.iter())
        .map(|(l, u)| l - u)
        .collect();
    let x_l = vec![0.0; n];
    let x_u = vec![1.0e20; n];
    let g_l = vec![1.0];
    let g_u = vec![1.0];
    let ws = classify_working_set(
        &lambda_x,
        &ipm_iter.lambda,
        1, // m_eq = 1 (the sum constraint)
        &ipm_iter.x,
        &x_l,
        &x_u,
        &ipm_iter.g,
        &g_l,
        &g_u,
        1e-8,
        1e-6,
    );
    // Bound x_2 must be classified AtLower.
    use pounce_qp::{BoundStatus, ConsStatus};
    assert_eq!(ws.bounds[0], BoundStatus::Inactive);
    assert_eq!(ws.bounds[1], BoundStatus::Inactive);
    assert_eq!(ws.bounds[2], BoundStatus::AtLower);
    assert_eq!(ws.constraints[0], ConsStatus::Equality);

    // --- Step 3: perturb p by Δp; predictor is just x* shifted
    //              by Δp on the interior coordinates (no sensitivity
    //              step needed for this convex quadratic). ---
    let dp = vec![0.02, -0.01, 0.05];
    *p_handle.borrow_mut() = p0.iter().zip(dp.iter()).map(|(a, b)| a + b).collect();

    // Predictor (water-filling closed form): at the active set
    // {x_3 = 0}, the interior variables satisfy x_i = p_i_new − λ,
    // sum_interior(x) = 1 ⇒ λ = (sum_interior(p_new) − 1) / 2.
    let p_new = p_handle.borrow().clone();
    let lambda_eq = (p_new[0] + p_new[1] - 1.0) / 2.0;
    let x_predictor = vec![p_new[0] - lambda_eq, p_new[1] - lambda_eq, 0.0];

    // --- Step 4: SQP corrector with warm-started working set. ---
    let warm = SqpIterates {
        x: x_predictor.clone(),
        lambda_g: vec![-lambda_eq], // sign matches the SQP/QP convention
        lambda_x: lambda_x.clone(),
        working: Some(ws),
    };
    app.set_sqp_warm_start(warm);
    app.options_mut()
        .set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    let status2 = app.optimize_tnlp(tnlp.clone());
    assert_eq!(status2, ApplicationReturnStatus::SolveSucceeded);
    let sqp_iter = finalize_handle.borrow().clone().unwrap();

    // --- Step 5: SQP corrector must match the exact perturbed
    //              closed-form solution. ---
    assert!(
        (sqp_iter.x[0] - x_predictor[0]).abs() < 1e-8,
        "x[0]: corrector={}, expected={}",
        sqp_iter.x[0],
        x_predictor[0]
    );
    assert!(
        (sqp_iter.x[1] - x_predictor[1]).abs() < 1e-8,
        "x[1]: corrector={}, expected={}",
        sqp_iter.x[1],
        x_predictor[1]
    );
    assert!(sqp_iter.x[2].abs() < 1e-8);

    // Confirm the corrector also returns a working set we can use
    // for the next step in the sweep.
    assert!(app.last_sqp_working_set().is_some());
}

/// PR #50 review T3 — exercise the corrector across an
/// **active-set flip**. At `p₀ = (0.5, 0.4, −0.1)` the lower bound
/// `x₃ ≥ 0` is binding; at `p₁ = (0.5, 0.4,  0.1)` it is not. The
/// SQP corrector starts from the previous working set (x₃ at
/// lower) but lands at the new optimum where x₃ > 0 — the QP must
/// detect the wrong-sign bound multiplier and drop x₃ from the
/// active set in the first inner iteration.
#[test]
fn sqp_corrector_handles_active_set_flip_between_perturbations() {
    let p0 = vec![0.5, 0.4, -0.1];
    let tnlp = std::rc::Rc::new(std::cell::RefCell::new(SimplexProj::new(p0.clone())));
    let p_handle = tnlp.borrow().p.clone();
    let finalize_handle = tnlp.borrow().finalize_sink.clone();

    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str("print_level 0\n").unwrap();
    let status = app.optimize_tnlp(tnlp.clone());
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let ipm = finalize_handle.borrow().clone().unwrap();

    let n = 3;
    let lambda_x: Vec<Number> = ipm
        .z_l
        .iter()
        .zip(ipm.z_u.iter())
        .map(|(l, u)| l - u)
        .collect();
    let x_l = vec![0.0; n];
    let x_u = vec![1.0e20; n];
    let g_l = vec![1.0];
    let g_u = vec![1.0];
    let ws = classify_working_set(
        &lambda_x,
        &ipm.lambda,
        1,
        &ipm.x,
        &x_l,
        &x_u,
        &ipm.g,
        &g_l,
        &g_u,
        1e-8,
        1e-6,
    );
    use pounce_qp::BoundStatus;
    assert_eq!(
        ws.bounds[2],
        BoundStatus::AtLower,
        "WS must capture x₃ active"
    );

    // Big perturbation: flip p[2] from negative to positive
    // enough that x₃ becomes interior at the new optimum.
    *p_handle.borrow_mut() = vec![0.5, 0.4, 0.1];

    // Stale predictor: just reuse the IPM primal. Predicting
    // exactly x_ipm puts x₃ at the bound while the new optimum
    // has x₃ > 0; the SQP corrector must detect the
    // wrong-sign multiplier and drop the bound.
    let warm = SqpIterates {
        x: ipm.x.clone(),
        lambda_g: ipm.lambda.clone(),
        lambda_x: lambda_x.clone(),
        working: Some(ws),
    };
    app.set_sqp_warm_start(warm);
    app.options_mut()
        .set_string_value("algorithm", "active-set-sqp", true, false)
        .unwrap();
    let status2 = app.optimize_tnlp(tnlp.clone());
    assert_eq!(status2, ApplicationReturnStatus::SolveSucceeded);
    let sqp = finalize_handle.borrow().clone().unwrap();

    // New optimum: λ = (0.5 + 0.4 + 0.1 - 1)/3 = 0; x = p_new.
    // x₃ = 0.1 > 0 (active set has flipped).
    assert!((sqp.x[0] - 0.5).abs() < 1e-7, "x[0] = {}", sqp.x[0]);
    assert!((sqp.x[1] - 0.4).abs() < 1e-7, "x[1] = {}", sqp.x[1]);
    assert!((sqp.x[2] - 0.1).abs() < 1e-7, "x[2] = {}", sqp.x[2]);
    // The corrector must have produced a new working set
    // reflecting x₃ no longer binding.
    let new_ws = app.last_sqp_working_set().expect("ws produced");
    assert_eq!(new_ws.bounds[2], BoundStatus::Inactive);
}
