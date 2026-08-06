//! Phase 6 acceptance (issue #493): a bound that was *transferred* off an
//! eliminated column must have its multiplier reported back on that column.
//!
//! The fixture is the smallest model that exhibits the transfer:
//!
//! ```text
//!   min (x0 − 4)²   s.t.   x0 − α·x1 = 0,   x0 ≤ 1
//! ```
//!
//! The row folds `x0` onto `x1` (both coefficients are comparable, so the
//! planner's cluster-size tie-break eliminates the first column), which
//! carries `x0 ≤ 1` onto `x1` as `x1 ≤ 1/α` — or, for `α < 0`, as
//! `x1 ≥ 1/α`, since a negative coefficient reverses the interval. Either
//! way the objective drives `x0` up into that bound, so the solver reports a
//! bound multiplier against a bound `x1` never declared.
//!
//! Attribution is what a no-presolve solve reports, so these tests compare
//! against the bare solve directly — *except* in the degenerate case, where
//! the survivor's own bound is active too and the split between the two is
//! genuinely non-unique. There the assertions are KKT validity and correct
//! signs, the posture `pounce-convex`'s `presolve_forcing.rs` already takes
//! for forcing constraints.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_presolve::{LinearEqElimTnlp, PresolveOptions, PresolveTnlp};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Default)]
struct Captured {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

struct Fixture {
    /// `x0 = α·x1`, published as the row `x0 − α·x1 = 0`.
    alpha: Number,
    x1_lo: Number,
    x1_hi: Number,
    captured: Option<Captured>,
}

impl Fixture {
    fn new(alpha: Number, x1_lo: Number, x1_hi: Number) -> Self {
        Self {
            alpha,
            x1_lo,
            x1_hi,
            captured: None,
        }
    }
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-1e19, self.x1_lo]);
        b.x_u.copy_from_slice(&[1.0, self.x1_hi]);
        b.g_l.copy_from_slice(&[0.0]);
        b.g_u.copy_from_slice(&[0.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.0, 0.0]);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.copy_from_slice(&[Linearity::Linear]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 4.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 4.0);
        g[1] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - self.alpha * x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values[0] = 1.0;
                values[1] = -self.alpha;
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
        self.captured = Some(Captured {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

fn opts(linear_eq_reduction: bool) -> PresolveOptions {
    PresolveOptions {
        enabled: true,
        linear_eq_reduction,
        // Everything else off, so any dual that moves moved for the reason
        // under test.
        bound_tightening: false,
        redundant_constraint_removal: false,
        licq_check: false,
        warm_z_bounds: false,
        ..PresolveOptions::defaults()
    }
}

struct Outcome {
    captured: Captured,
    reduced_n: i32,
}

fn solve(alpha: Number, x1_lo: Number, x1_hi: Number, linear_eq_reduction: bool) -> Outcome {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    // Unscaled, so the reported duals are directly comparable against the
    // fixture's own analytic gradient.
    app.options_mut()
        .set_string_value("nlp_scaling_method", "none", true, false)
        .unwrap();

    let concrete = Rc::new(RefCell::new(Fixture::new(alpha, x1_lo, x1_hi)));
    let presolve = Rc::new(RefCell::new(PresolveTnlp::new(
        Rc::clone(&concrete) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let elim = Rc::new(RefCell::new(LinearEqElimTnlp::new(
        Rc::clone(&presolve) as Rc<RefCell<dyn TNLP>>,
        opts(linear_eq_reduction),
    )));
    let info = elim.borrow_mut().get_nlp_info().expect("dims");
    let _ = app.optimize_tnlp(Rc::clone(&elim) as Rc<RefCell<dyn TNLP>>);

    Outcome {
        captured: concrete.borrow().captured.clone().expect("finalized"),
        reduced_n: info.n,
    }
}

/// `∇f + Jᵀλ − z_l + z_u` at both columns, in POUNCE's `finalize_solution`
/// convention.
fn stationarity(alpha: Number, c: &Captured) -> [Number; 2] {
    let grad = [2.0 * (c.x[0] - 4.0), 0.0];
    let jac = [1.0, -alpha];
    [0, 1].map(|j| grad[j] + jac[j] * c.lambda[0] - c.z_l[j] + c.z_u[j])
}

#[test]
fn a_transferred_upper_bounds_multiplier_lands_on_the_column_that_owns_it() {
    let on = solve(2.0, -100.0, 100.0, true);
    assert_eq!(on.reduced_n, 1, "x0 should be gone");
    let off = solve(2.0, -100.0, 100.0, false);

    // x0 = 1 (its own upper bound), x1 = 0.5.
    assert!((on.captured.x[0] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert!((on.captured.x[1] - 0.5).abs() < 1e-6, "{:?}", on.captured.x);

    // The multiplier belongs to x0's bound, scaled by 1/|α| on the way back.
    assert!(
        (on.captured.z_u[0] - 6.0).abs() < 1e-4,
        "z_u[0] = {}, expected 6 (the bare solve reports {})",
        on.captured.z_u[0],
        off.captured.z_u[0]
    );
    assert!(
        on.captured.z_u[1].abs() < 1e-4,
        "x1 never declared that bound: z_u[1] = {}",
        on.captured.z_u[1]
    );
    // With the bound multiplier where it belongs, the consumed row's own
    // multiplier is zero — as the bare solve reports it.
    assert!(
        (on.captured.lambda[0] - off.captured.lambda[0]).abs() < 1e-4,
        "λ = {} vs bare {}",
        on.captured.lambda[0],
        off.captured.lambda[0]
    );
    for j in 0..2 {
        assert!((on.captured.z_u[j] - off.captured.z_u[j]).abs() < 1e-4);
        assert!((on.captured.z_l[j] - off.captured.z_l[j]).abs() < 1e-4);
    }
}

/// The sign flip is where this breaks: with `α < 0` the survivor's *lower*
/// bound is the eliminated column's *upper* bound, so the multiplier has to
/// change sides on the way back.
#[test]
fn a_negative_coefficient_sends_the_multiplier_to_the_other_side_of_the_box() {
    let on = solve(-2.0, -100.0, 100.0, true);
    assert_eq!(on.reduced_n, 1);
    let off = solve(-2.0, -100.0, 100.0, false);

    assert!((on.captured.x[0] - 1.0).abs() < 1e-6, "{:?}", on.captured.x);
    assert!((on.captured.x[1] + 0.5).abs() < 1e-6, "{:?}", on.captured.x);

    // The reduced problem reports this against x1's *lower* bound; x0's box
    // is bounded above, and that is where it must arrive.
    assert!(
        (on.captured.z_u[0] - 6.0).abs() < 1e-4,
        "z_u[0] = {}, expected 6 (the bare solve reports {})",
        on.captured.z_u[0],
        off.captured.z_u[0]
    );
    assert!(
        on.captured.z_l[0].abs() < 1e-4,
        "x0 has no lower bound to be active: z_l[0] = {}",
        on.captured.z_l[0]
    );
    assert!(on.captured.z_l[1].abs() < 1e-4 && on.captured.z_u[1].abs() < 1e-4);
    assert!((on.captured.lambda[0] - off.captured.lambda[0]).abs() < 1e-4);
}

#[test]
fn both_signs_close_full_space_stationarity() {
    for alpha in [2.0, -2.0, 0.5, -0.5, 3.0] {
        let on = solve(alpha, -100.0, 100.0, true);
        for (j, r) in stationarity(alpha, &on.captured).iter().enumerate() {
            assert!(
                r.abs() < 1e-5,
                "α = {alpha}: stationarity at column {j} = {r}, duals = {:?}",
                on.captured
            );
        }
        for j in 0..2 {
            assert!(on.captured.z_l[j] >= -1e-8 && on.captured.z_u[j] >= -1e-8);
        }
    }
}

/// When the survivor's own bound and a transferred bound are the *same*
/// constraint, the reduced problem has one multiplier where the full problem
/// has two, and any split summing correctly is a valid KKT point. Assert
/// validity, not equality with the bare solve.
#[test]
fn the_degenerate_both_active_case_is_still_a_valid_kkt_point() {
    // x0 ≤ 1 transfers to x1 ≤ 0.5, which x1 already declares.
    let alpha = 2.0;
    let on = solve(alpha, -100.0, 0.5, true);
    assert_eq!(on.reduced_n, 1);
    let c = &on.captured;

    assert!((c.x[0] - 1.0).abs() < 1e-6, "{:?}", c.x);
    assert!((c.x[1] - 0.5).abs() < 1e-6, "{:?}", c.x);
    for (j, r) in stationarity(alpha, c).iter().enumerate() {
        assert!(
            r.abs() < 1e-5,
            "stationarity at column {j} = {r}, duals = {c:?}"
        );
    }
    for j in 0..2 {
        assert!(c.z_l[j] >= -1e-8, "z_l[{j}] = {} is negative", c.z_l[j]);
        assert!(c.z_u[j] >= -1e-8, "z_u[{j}] = {} is negative", c.z_u[j]);
        // Complementarity: no multiplier on a slack bound.
        assert!(c.z_l[j] < 1e-4, "z_l[{j}] on an inactive bound");
    }
    // The tie goes to the incumbent, so the multiplier stays where the
    // solver reported it rather than moving to a bound that is only
    // coincidentally equal.
    assert!(
        c.z_u[1] > 1e-3,
        "expected the survivor to keep it: z_u = {:?}",
        c.z_u
    );
}
