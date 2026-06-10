//! Regression tests for pounce#128: the reduced Hessian and the
//! parametric step must come back in **natural (unscaled) units**,
//! independent of the NLP scaling the IPM applied internally.
//!
//! Fixture: a 2-variable least-squares-style NLP
//!
//! ```text
//! min  c0·(x − p)² + c1·(x − 1)²
//! s.t. SCALE·p = SCALE·p̂          (parameter pin, row scaled by SCALE)
//! ```
//!
//! Eliminating x analytically gives `f*(p) = c0·c1·(p − 1)² / (c0 + c1)`,
//! so with the pin's right-hand side `r = SCALE·p̂` as the parameter,
//!
//! ```text
//! ∂²f*/∂r² = 2·c0·c1 / ((c0 + c1)·SCALE²)
//! ```
//!
//! Sign convention: over pin *constraint* rows, `B K⁻¹ Bᵀ` is the
//! multiplier sensitivity `∂y/∂r = −∂²f*/∂r²` (IPOPT's `L = f + yᵀg`
//! gives `y = −∂f*/∂r` at the optimum), so the reported reduced
//! Hessian is the **negative** of `∂²f*/∂r²` — which is why the
//! covariance recipe is `Cov = −inv(H)`.
//!
//! The coefficients are chosen so the objective gradient at the
//! starting point (≈ 2·c1 = 1.2e5) exceeds `nlp_scaling_max_gradient`
//! (100) — the gradient-based objective scaling `df` fires — and the
//! pin row's Jacobian entry SCALE = 1e4 exceeds it too, so the
//! per-row constraint scaling `dc` fires on the pin itself. Before
//! pounce#128 the returned reduced Hessian was off by exactly
//! `df / dc²`.

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

const C0: Number = 4.0e4;
const C1: Number = 6.0e4;
const SCALE: Number = 1.0e4;
const P_HAT: Number = 0.7;

/// `min c0(x−p)² + c1(x−1)²  s.t.  SCALE·p = SCALE·p̂`. Variables
/// `(x, p)`; one equality (the pin). Optionally prepends an inactive
/// inequality `x + p ≤ 10` as g[0] so the pin sits *after* an
/// inequality in the user's g ordering (exercises the full-g → c-block
/// row mapping).
struct ScaledPinTnlp {
    with_leading_inequality: bool,
}

impl ScaledPinTnlp {
    fn m(&self) -> Index {
        if self.with_leading_inequality {
            2
        } else {
            1
        }
    }
    /// Row index of the pin constraint in the user's g ordering.
    fn pin_row(&self) -> Index {
        if self.with_leading_inequality {
            1
        } else {
            0
        }
    }
}

impl TNLP for ScaledPinTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: self.m(),
            nnz_jac_g: if self.with_leading_inequality { 3 } else { 1 },
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        let pin = self.pin_row() as usize;
        if self.with_leading_inequality {
            b.g_l[0] = -1.0e19;
            b.g_u[0] = 10.0;
        }
        b.g_l[pin] = SCALE * P_HAT;
        b.g_u[pin] = SCALE * P_HAT;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (xx, p) = (x[0], x[1]);
        Some(C0 * (xx - p) * (xx - p) + C1 * (xx - 1.0) * (xx - 1.0))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (xx, p) = (x[0], x[1]);
        g[0] = 2.0 * C0 * (xx - p) + 2.0 * C1 * (xx - 1.0);
        g[1] = -2.0 * C0 * (xx - p);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let pin = self.pin_row() as usize;
        if self.with_leading_inequality {
            g[0] = x[0] + x[1];
        }
        g[pin] = SCALE * x[1];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let pin = self.pin_row();
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                if self.with_leading_inequality {
                    irow.copy_from_slice(&[0, 0, pin]);
                    jcol.copy_from_slice(&[0, 1, 1]);
                } else {
                    irow.copy_from_slice(&[pin]);
                    jcol.copy_from_slice(&[1]);
                }
            }
            SparsityRequest::Values { values } => {
                if self.with_leading_inequality {
                    values[0] = 1.0;
                    values[1] = 1.0;
                    values[2] = SCALE;
                } else {
                    values[0] = SCALE;
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
        // Lower triangle of ∇²f: (0,0)=2c0+2c1, (1,0)=−2c0, (1,1)=2c0.
        // Constraints are linear — no Lagrangian contribution.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * 2.0 * (C0 + C1);
                values[1] = obj_factor * (-2.0) * C0;
                values[2] = obj_factor * 2.0 * C0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn make_app(scaling_method: &str) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", scaling_method, true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

/// Analytic reduced Hessian w.r.t. the pin's right-hand side, in
/// pounce's pin-row sign convention (`= −∂²f*/∂r²`, see module doc).
const H_ANALYTIC: Number = -2.0 * C0 * C1 / ((C0 + C1) * SCALE * SCALE);

fn run_reduced_hessian(scaling_method: &str, with_leading_inequality: bool) -> (Number, Number) {
    let mut app = make_app(scaling_method);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ScaledPinTnlp {
        with_leading_inequality,
    }));
    let pin_row = if with_leading_inequality { 1 } else { 0 };
    let result = SensSolve::new(vec![pin_row])
        .with_reduced_hessian()
        .run(&mut app, tnlp);
    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed under nlp_scaling_method={scaling_method}: {:?}",
        result.status,
    );
    let hr = result.reduced_hessian.expect("reduced Hessian populated");
    let hr_scaled = result
        .reduced_hessian_scaled
        .expect("scaled reduced Hessian populated");
    assert_eq!(hr.len(), 1);

    // Cross-check the reported scaling factors reconstruct the scaled
    // value exactly: H̃ = (df / dc²) · H.
    let df = result.obj_scaling_factor.expect("df reported");
    let dc = result.pin_g_scaling.expect("pin scaling reported");
    assert_eq!(dc.len(), 1);
    let reconstructed = hr[0] * df / (dc[0] * dc[0]);
    assert!(
        (reconstructed - hr_scaled[0]).abs() <= 1e-10 * hr_scaled[0].abs().max(1.0),
        "scaled reconstruction mismatch: {} vs {}",
        reconstructed,
        hr_scaled[0],
    );
    (hr[0], hr_scaled[0])
}

#[test]
fn reduced_hessian_is_invariant_to_nlp_scaling() {
    let (h_none, h_none_scaled) = run_reduced_hessian("none", false);
    let (h_grad, h_grad_scaled) = run_reduced_hessian("gradient-based", false);

    let rel = |a: Number, b: Number| (a - b).abs() / b.abs();
    assert!(
        rel(h_none, H_ANALYTIC) < 1e-6,
        "unscaled solve: H = {h_none}, analytic = {H_ANALYTIC}",
    );
    assert!(
        rel(h_grad, H_ANALYTIC) < 1e-6,
        "gradient-based solve: H = {h_grad}, analytic = {H_ANALYTIC} \
         (ratio {} — a ratio ≫ 1 means the pre-#128 scaled value leaked out)",
        H_ANALYTIC / h_grad,
    );

    // With scaling off, the scaled accessor must equal the natural one.
    assert!(rel(h_none_scaled, h_none) < 1e-12);
    // With gradient-based scaling active on this fixture, the two must
    // differ wildly (df ≈ 8.3e-4 fires AND dc ≈ 1e-2 fires on the pin):
    // that's the pre-#128 value preserved for calibrated callers.
    assert!(
        rel(h_grad_scaled, h_grad) > 1.0,
        "expected the scaled accessor to differ when scaling is active: \
         natural {h_grad}, scaled {h_grad_scaled}",
    );
}

#[test]
fn pin_rows_map_through_the_c_d_split() {
    // The pin is g[1], *after* an inactive inequality g[0] — the y_c
    // block holds only the pin (c-block row 0). Before pounce#128 the
    // pin row was mapped as `n_x + n_s + 1`, silently selecting a
    // wrong KKT row; with the `full_g_to_c_block` mapping the reduced
    // Hessian matches the same analytic value as the
    // no-inequality fixture.
    let (h, _) = run_reduced_hessian("none", true);
    let rel = (h - H_ANALYTIC).abs() / H_ANALYTIC;
    assert!(
        rel < 1e-6,
        "with a leading inequality: H = {h}, analytic = {H_ANALYTIC}",
    );
    // And stays right when scaling fires too.
    let (h2, _) = run_reduced_hessian("gradient-based", true);
    let rel2 = (h2 - H_ANALYTIC).abs() / H_ANALYTIC;
    assert!(
        rel2 < 1e-6,
        "leading inequality + gradient-based scaling: H = {h2}, analytic = {H_ANALYTIC}",
    );
}

#[test]
fn parametric_step_is_invariant_to_nlp_scaling() {
    // dx*/dr for the pin RHS r: x*(p) = (c0·p + c1)/(c0+c1) and
    // p = r/SCALE, so dx/dr = c0/((c0+c1)·SCALE), dp/dr = 1/SCALE.
    let delta = 0.5 * SCALE; // Δr (RHS units) ⇒ Δp = 0.5
    let dx_expected = [delta * C0 / ((C0 + C1) * SCALE), delta / SCALE];

    for method in ["none", "gradient-based"] {
        let mut app = make_app(method);
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ScaledPinTnlp {
            with_leading_inequality: false,
        }));
        let result = SensSolve::new(vec![0])
            .with_deltas(vec![delta])
            .run(&mut app, tnlp);
        assert!(matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ));
        let dx = result.dx.expect("dx populated");
        for k in 0..2 {
            let err = (dx[k] - dx_expected[k]).abs() / dx_expected[k].abs();
            assert!(
                err < 1e-6,
                "nlp_scaling_method={method}: dx[{k}] = {}, expected {} (rel err {err})",
                dx[k],
                dx_expected[k],
            );
        }
    }
}
