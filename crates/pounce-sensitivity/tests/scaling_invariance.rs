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
/// inequality `SCALE·(x + p) ≤ SCALE·10` as g[0] so the pin sits
/// *after* an inequality in the user's g ordering (exercises the
/// full-g → c-block row mapping) — and, being badly scaled itself,
/// fires the per-row `dd` inequality scaling, which exercises the
/// s/y_d/v-block entries of the natural-units (E, F) pair, including
/// the `pd_u` expansion lookup for the v rows.
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
            b.g_u[0] = SCALE * 10.0;
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
            g[0] = SCALE * (x[0] + x[1]);
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
                    values[0] = SCALE;
                    values[1] = SCALE;
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
    // Clean quadratic fixture: the final factorization needs no
    // inertia correction, and the all-zero perturbations are the
    // signal that the natural-units covariance reading is exact.
    let pert = result
        .kkt_perturbations
        .expect("KKT perturbations reported at convergence");
    assert_eq!(pert, [0.0; 4], "unexpected KKT regularization: {pert:?}");
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
    let rel = (h - H_ANALYTIC).abs() / H_ANALYTIC.abs();
    assert!(
        rel < 1e-6,
        "with a leading inequality: H = {h}, analytic = {H_ANALYTIC}",
    );
    // And stays right when scaling fires too.
    let (h2, _) = run_reduced_hessian("gradient-based", true);
    let rel2 = (h2 - H_ANALYTIC).abs() / H_ANALYTIC.abs();
    assert!(
        rel2 < 1e-6,
        "leading inequality + gradient-based scaling: H = {h2}, analytic = {H_ANALYTIC}",
    );
}

/// `obj_scaling_factor < 0` is the documented way to maximize. The
/// two-sided (E, F) natural-units scaling needs no square root, so a
/// negative effective `df` is handled and the sensitivity surfaces
/// keep working. The fixture maximizes `−[c0(x−p)² + c1(x−1)²]`
/// (same minimizer as [`ScaledPinTnlp`] under `obj_scaling_factor =
/// −1`). The user-space multipliers satisfy the *declared* problem's
/// Lagrangian `−f + yᵀg`, so the natural-units reduced Hessian is
/// the sign-flip of [`H_ANALYTIC`].
#[test]
fn reduced_hessian_works_under_negative_obj_scaling() {
    let mut app = make_app("gradient-based");
    app.options_mut()
        .set_numeric_value("obj_scaling_factor", -1.0, true, false)
        .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(NegatedScaledPinTnlp));
    let result = SensSolve::new(vec![0])
        .with_reduced_hessian()
        .run(&mut app, tnlp);
    assert!(
        matches!(
            result.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "maximization solve failed: {:?}",
        result.status,
    );
    let hr = result
        .reduced_hessian
        .expect("reduced Hessian populated under negative obj scaling");
    let expected = -H_ANALYTIC;
    let rel = (hr[0] - expected).abs() / expected.abs();
    assert!(
        rel < 1e-6,
        "negative obj_scaling_factor: H = {}, expected = {expected}",
        hr[0],
    );
    let df = result.obj_scaling_factor.expect("df reported");
    assert!(df < 0.0, "effective df must carry the user's sign: {df}");
}

/// `max −[c0(x−p)² + c1(x−1)²]` posed as a TNLP whose `f` is the
/// negated objective; with `obj_scaling_factor = −1` the IPM
/// re-negates it internally and converges to the same minimizer as
/// [`ScaledPinTnlp`].
struct NegatedScaledPinTnlp;

impl TNLP for NegatedScaledPinTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.g_l[0] = SCALE * P_HAT;
        b.g_u[0] = SCALE * P_HAT;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (xx, p) = (x[0], x[1]);
        Some(-(C0 * (xx - p) * (xx - p) + C1 * (xx - 1.0) * (xx - 1.0)))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (xx, p) = (x[0], x[1]);
        g[0] = -(2.0 * C0 * (xx - p) + 2.0 * C1 * (xx - 1.0));
        g[1] = 2.0 * C0 * (xx - p);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = SCALE * x[1];
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
                irow.copy_from_slice(&[0]);
                jcol.copy_from_slice(&[1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = SCALE;
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
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = -obj_factor * 2.0 * (C0 + C1);
                values[1] = obj_factor * 2.0 * C0;
                values[2] = -obj_factor * 2.0 * C0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `min c0(x−p)² + c1(x−1)²` with an **active** inequality
/// `SCALE·x ≥ SCALE·0.95` (g[0]) and the pin `SCALE·p = SCALE·p̂`
/// (g[1]). Variables `(x, p)`. With `p` pinned at `p̂ = 0.7` the
/// unconstrained-in-`x` minimizer is `x* = (c0·p̂ + c1)/(c0+c1) =
/// 0.88 < 0.95`, so the lower bound binds and `x` is clamped at
/// `0.95`. The active inequality makes its slack bound multiplier
/// (`v_l`) nonzero, so — unlike the *inactive* inequality in
/// [`ScaledPinTnlp`] — the v-block of the natural-units (E, F) pair
/// now actually moves the numbers: this is the coverage that proves
/// the `dd`-dependent v-row scaling is applied correctly, not merely
/// present.
struct ActiveIneqTnlp;

impl TNLP for ActiveIneqTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        // g[0] = SCALE·x ≥ SCALE·0.95 (active lower bound)
        b.g_l[0] = SCALE * 0.95;
        b.g_u[0] = 1.0e19;
        // g[1] = SCALE·p = SCALE·p̂ (pin)
        b.g_l[1] = SCALE * P_HAT;
        b.g_u[1] = SCALE * P_HAT;
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
        g[0] = SCALE * x[0];
        g[1] = SCALE * x[1];
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = SCALE;
                values[1] = SCALE;
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

#[test]
fn reduced_hessian_with_active_inequality_is_scaling_invariant() {
    // With x clamped at 0.95 by the active inequality, f*(p) =
    // c0·(0.95 − p)² + c1·(0.95 − 1)², so ∂²f*/∂p² = 2·c0 and, with
    // the pin RHS r = SCALE·p, ∂²f*/∂r² = 2·c0 / SCALE². The reported
    // reduced Hessian carries the pin-row sign flip (= −∂²f*/∂r²):
    const H_ACTIVE: Number = -2.0 * C0 / (SCALE * SCALE);

    let run = |method: &str| -> (Number, Number) {
        let mut app = make_app(method);
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ActiveIneqTnlp));
        let result = SensSolve::new(vec![1])
            .with_reduced_hessian()
            .run(&mut app, tnlp);
        assert!(
            matches!(
                result.status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "solve failed under nlp_scaling_method={method}: {:?}",
            result.status,
        );
        let hr = result.reduced_hessian.expect("reduced Hessian populated");
        let hr_scaled = result
            .reduced_hessian_scaled
            .expect("scaled reduced Hessian populated");
        (hr[0], hr_scaled[0])
    };

    let (h_none, h_none_scaled) = run("none");
    let (h_grad, h_grad_scaled) = run("gradient-based");
    let rel = |a: Number, b: Number| (a - b).abs() / b.abs();

    // The headline guard: the natural-units reduced Hessian over an
    // ACTIVE inequality is the same regardless of NLP scaling. The
    // v-block scaling factors (`dd`-dependent F rows) are now nonzero
    // contributors to the back-solve; if they were dropped or wrong,
    // the gradient-based value would diverge from the unscaled one.
    assert!(
        rel(h_none, h_grad) < 1e-6,
        "active-inequality reduced Hessian not scaling-invariant: \
         none {h_none}, gradient-based {h_grad}",
    );
    // ...and it matches the clamped-x analytic value, which differs
    // from the inactive-inequality H_ANALYTIC (= −4.8e-4): the active
    // constraint removes x as a free variable, so the curvature comes
    // from c0 alone.
    assert!(
        rel(h_none, H_ACTIVE) < 1e-6,
        "active-inequality reduced Hessian: H = {h_none}, analytic = {H_ACTIVE}",
    );
    assert!(
        rel(H_ACTIVE, H_ANALYTIC) > 0.1,
        "fixture sanity: the active-inequality H must differ from the inactive one",
    );
    // With scaling off the two accessors agree; with gradient-based
    // scaling on they must differ (the scaled value is the pre-#128
    // leak preserved for calibrated callers).
    assert!(rel(h_none_scaled, h_none) < 1e-12);
    assert!(
        rel(h_grad_scaled, h_grad) > 1.0,
        "expected the scaled accessor to differ when scaling is active: \
         natural {h_grad}, scaled {h_grad_scaled}",
    );
}

/// Two pins at **different** scales so the off-diagonal of the
/// reported reduced Hessian carries a `dc_i·dc_j` cross term:
///
/// ```text
/// min c0(x − p0)² + c1(x − p1)²
/// s.t. SCALE0·p0 = SCALE0·p̂0   (g[0], pin)
///      SCALE1·p1 = SCALE1·p̂1   (g[1], pin)
/// ```
///
/// Variables `(x, p0, p1)`. Eliminating x gives `f*(p0,p1) =
/// K·(p1 − p0)²` with `K = c0·c1/(c0+c1)`, so the natural-units
/// reduced Hessian over `(r0, r1) = (SCALE0·p0, SCALE1·p1)` is, in
/// the pin-row sign convention (`= −∂²f*/∂r²`):
///
/// ```text
/// H = [ −2K/SCALE0²        +2K/(SCALE0·SCALE1) ]
///     [ +2K/(SCALE0·SCALE1)   −2K/SCALE1²      ]
/// ```
///
/// The two distinct scales (`dc_0 ≠ dc_1`) make every entry carry a
/// different scaling correction; the off-diagonal in particular
/// exercises the `dc_i·dc_j` cross product that a per-pin (diagonal-
/// only) correction would get wrong.
struct MultiPinTnlp;

const SCALE0: Number = 1.0e4;
const SCALE1: Number = 2.0e4;
const P0_HAT: Number = 0.3;
const P1_HAT: Number = 0.8;

impl TNLP for MultiPinTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 2,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        b.g_l[0] = SCALE0 * P0_HAT;
        b.g_u[0] = SCALE0 * P0_HAT;
        b.g_l[1] = SCALE1 * P1_HAT;
        b.g_u[1] = SCALE1 * P1_HAT;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // Nonzero so the objective gradient at the start exceeds
        // nlp_scaling_max_gradient and gradient-based `df` fires.
        sp.x[0] = 1.0;
        sp.x[1] = 0.0;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (xx, p0, p1) = (x[0], x[1], x[2]);
        Some(C0 * (xx - p0) * (xx - p0) + C1 * (xx - p1) * (xx - p1))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (xx, p0, p1) = (x[0], x[1], x[2]);
        g[0] = 2.0 * C0 * (xx - p0) + 2.0 * C1 * (xx - p1);
        g[1] = -2.0 * C0 * (xx - p0);
        g[2] = -2.0 * C1 * (xx - p1);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = SCALE0 * x[1];
        g[1] = SCALE1 * x[2];
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[1, 2]);
            }
            SparsityRequest::Values { values } => {
                values[0] = SCALE0;
                values[1] = SCALE1;
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
        // Lower triangle: (0,0)=2c0+2c1, (1,0)=−2c0, (1,1)=2c0,
        // (2,0)=−2c1, (2,2)=2c1. The p0/p1 cross term is zero.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 2]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 2]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * 2.0 * (C0 + C1);
                values[1] = obj_factor * (-2.0) * C0;
                values[2] = obj_factor * 2.0 * C0;
                values[3] = obj_factor * (-2.0) * C1;
                values[4] = obj_factor * 2.0 * C1;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn multi_pin_reduced_hessian_off_diagonal_is_scaling_invariant() {
    const K: Number = C0 * C1 / (C0 + C1);
    // Column-major 2×2; symmetric so only three distinct values.
    let h00 = -2.0 * K / (SCALE0 * SCALE0);
    let h11 = -2.0 * K / (SCALE1 * SCALE1);
    let h01 = 2.0 * K / (SCALE0 * SCALE1);

    let run = |method: &str| -> Vec<Number> {
        let mut app = make_app(method);
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(MultiPinTnlp));
        let result = SensSolve::new(vec![0, 1])
            .with_reduced_hessian()
            .run(&mut app, tnlp);
        assert!(
            matches!(
                result.status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "solve failed under nlp_scaling_method={method}: {:?}",
            result.status,
        );
        let hr = result.reduced_hessian.expect("reduced Hessian populated");
        assert_eq!(hr.len(), 4, "expected a 2×2 reduced Hessian");
        hr
    };

    let h_none = run("none");
    let h_grad = run("gradient-based");
    let rel = |a: Number, b: Number| (a - b).abs() / b.abs().max(1e-30);

    // Off-diagonal symmetry within each solve.
    assert!(rel(h_none[1], h_none[2]) < 1e-9, "asymmetric: {h_none:?}");

    // Headline guard: the full 2×2 natural-units reduced Hessian —
    // diagonal AND off-diagonal — is invariant to NLP scaling. The
    // off-diagonal mixes the two distinct pin scales (dc_0 ≠ dc_1);
    // a diagonal-only scaling correction would leave it wrong under
    // gradient-based scaling.
    for k in 0..4 {
        assert!(
            rel(h_none[k], h_grad[k]) < 1e-6,
            "entry {k} not scaling-invariant: none {}, gradient-based {}",
            h_none[k],
            h_grad[k],
        );
    }

    // ...and matches the hand-derived analytic block.
    assert!(
        rel(h_none[0], h00) < 1e-6,
        "H[0,0] = {}, analytic {h00}",
        h_none[0]
    );
    assert!(
        rel(h_none[3], h11) < 1e-6,
        "H[1,1] = {}, analytic {h11}",
        h_none[3]
    );
    assert!(
        rel(h_none[1], h01) < 1e-6,
        "H[1,0] = {}, analytic {h01} (the cross-scale off-diagonal)",
        h_none[1],
    );
    // The off-diagonal must be genuinely nonzero — otherwise this
    // would silently pass as a pair of decoupled single-pin tests.
    assert!(h01.abs() > 1e-6, "fixture sanity: off-diagonal is nonzero");
}

/// F1 follow-up (pounce#11): the user-space multipliers `SensSolve`
/// captures must be in natural (unscaled-Lagrangian) units — the
/// `finalize_solution` / Python-info-dict convention — independent of
/// `nlp_scaling_method`. The capture used to go through
/// `pack_lambda_for_user`/`pack_z_*_for_user`, which unwind the
/// per-row constraint scaling but NOT `obj_scale_factor`, so whenever
/// gradient-based objective scaling fired the reported `mult_g` was
/// obj-scaled (off by `df`). [`ScaledPinTnlp`]'s starting objective
/// gradient (≈ 2·c1 = 1.2e5) exceeds `nlp_scaling_max_gradient`
/// (100), so `df` fires under `gradient-based` and this fixture
/// catches the leak.
#[test]
fn captured_multipliers_are_invariant_to_nlp_scaling() {
    // Analytic pin multiplier in the user's convention (L = f + λᵀg):
    // stationarity in p gives −2·c0·(x* − p̂) + λ·SCALE = 0 with
    // x* = (c0·p̂ + c1)/(c0 + c1).
    let x_star = (C0 * P_HAT + C1) / (C0 + C1);
    let lambda_analytic = 2.0 * C0 * (x_star - P_HAT) / SCALE;

    let run = |method: &str| -> (Vec<Number>, Number) {
        let mut app = make_app(method);
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(ScaledPinTnlp {
            with_leading_inequality: false,
        }));
        let result = SensSolve::new(vec![0]).run(&mut app, tnlp);
        assert!(
            matches!(
                result.status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
            ),
            "solve failed under nlp_scaling_method={method}: {:?}",
            result.status,
        );
        assert!(result.error.is_none(), "sens error: {:?}", result.error);
        let mult_g = result.mult_g.expect("mult_g captured");
        let df = result.obj_scaling_factor.expect("df reported");
        // Bound multipliers must also be captured (all-zero here:
        // the fixture has no finite variable bounds) at full-x length.
        assert_eq!(result.mult_x_l.expect("mult_x_l captured").len(), 2);
        assert_eq!(result.mult_x_u.expect("mult_x_u captured").len(), 2);
        (mult_g, df)
    };

    let (m_none, df_none) = run("none");
    let (m_grad, df_grad) = run("gradient-based");

    // Fixture sanity: the gradient-based objective scaling really
    // fired — otherwise this test degenerates into a no-op guard.
    assert!(
        (df_none - 1.0).abs() < 1e-12,
        "df must be 1.0 under nlp_scaling_method=none: {df_none}"
    );
    assert!(
        (df_grad - 1.0).abs() > 0.5,
        "gradient-based df should have fired on this fixture: {df_grad}"
    );

    assert_eq!(m_none.len(), 1);
    assert_eq!(m_grad.len(), 1);
    let rel = |a: Number, b: Number| (a - b).abs() / b.abs();
    assert!(
        rel(m_none[0], lambda_analytic) < 1e-6,
        "unscaled solve: mult_g = {}, analytic = {lambda_analytic}",
        m_none[0],
    );
    // The headline guard: same natural-units value under scaling. A
    // capture that skips the obj_scale division would report
    // λ·df ≈ λ/1200 here instead.
    assert!(
        rel(m_grad[0], m_none[0]) < 1e-6,
        "mult_g not scaling-invariant: none {}, gradient-based {} \
         (ratio {} — a ratio ≪ 1 means the obj-scaled dual leaked out)",
        m_none[0],
        m_grad[0],
        m_grad[0] / m_none[0],
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
