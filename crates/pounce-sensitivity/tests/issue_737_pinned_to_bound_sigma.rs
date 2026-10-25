//! A declared parameter pinned to exactly a bound of the variable it
//! pins (gh#737).
//!
//! # The defect
//!
//! A model can pin a variable to a parameter, `x == p`, and then give
//! the parameter a value that sits exactly on one of `x`'s own bounds.
//! `d x/d p` is then `1` by construction: the equality is linear and
//! says so. What came back instead was `0`, for the whole column
//! through `x` — the parameter's own Var absorbed **half** the
//! requested delta and the other half came back as the residual of a
//! linear equality that the step is supposed to satisfy exactly. An
//! even split is what a solve returns when two rows have stopped being
//! distinguishable to it, and here they had: with `x` gone from the
//! system, `x − p = 0` and `p = p₀ + Δ` are two inconsistent equations
//! in the one remaining unknown.
//!
//! # Why `x` was gone
//!
//! `x` is held between a bound and an equality that both bind, so the
//! force that holds it has no unique split between the two and the
//! solve lands with a bound multiplier far larger than the geometry
//! needs, over a slack near roundoff. `Σ = z/s` is the product of both,
//! and on the fixture below it comes out at `6.9e27` against Jacobian
//! entries of `1`.
//!
//! Eliminating `x` through that diagonal leaves each of its constraint
//! rows holding `a²/Σ ≈ 1e-27` — the whole of what the row still knows
//! about `x`, and far under the roundoff of the row it lands in. The
//! factorization has a row it cannot pivot on, and the answer is
//! whatever the singularity handling substitutes.
//!
//! The fix caps `Σ` at the stiffness the constraint rows can still be
//! seen against (`sigma_pin_caps` in `algorithm_backsolver`). It is not
//! a release: the capped pin still holds `x` to within roundoff of its
//! own scale, so a bound that genuinely holds a variable keeps holding
//! it. What changes is only that the equality is once again
//! representable.
//!
//! # Measured on this fixture, before → after
//!
//! ```text
//!                      Σ          d x/d p    d p_var/d p   requested
//!   interior     6.88e27 → 7.0e13   0 → 1     ½ → 1          1
//!   crossover    1.13e25 → 7.0e13   0 → 1     ½ → 1          1
//! ```
//!
//! # The reported model
//!
//! This fixture is a regression test, not the diagnosis. The diagnosis
//! is notebook 36's CSTR, where `zc_init: zc[0] == zc0` with `zc`
//! bounded `(0, 1)` reads `Σ = 1.9e27` and `d zc[0]/d zc0 = 0` at
//! `zc0 = 1.0`, while sIPOPT solves the same model and perturbation
//! correctly. Every `zc0` short of a bound was already right, which is
//! the same shape the `t` sweep has here. Measured on that model, on
//! the commit adding this file against its parent, `d zc[0]/d zc0` goes
//! `0.00000 → 1.00000` at both `zc0 = 0.00` and `zc0 = 1.00` and is
//! unchanged elsewhere, and `estimate()` to `(0.79, 0.57)` returns
//! `zc[0] = 0.790000` in all three modes rather than the baseline.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

mod common;
use common::outside_engine::{Relaxation, recorded_solve, same_classification};

/// Where the parameter row holds `p`, and equally `x`'s upper bound.
const PIN: Number = 1.0;
/// The perturbation asked of the parameter row.
const DELTA: Number = -0.21;

/// ```text
/// min  ½(x − T)² + ½(w − 1)² + ½(u − 1)²
/// s.t. g₀:  x − p = 0
///      g₁:      p = PIN     ← the parameter row, perturbed
///      g₂:  w − x = 0
///      0 ≤ x ≤ PIN,  p, w and u free
/// ```
///
/// `u` is in no constraint and nothing else reads it. It is here to
/// keep the model **non-square**: at `n == m` the solve takes Ipopt's
/// square-problem path (gh#508, ported in gh#735), which zeroes the
/// bound multipliers because the equalities already determine every
/// variable. That is the right answer for such a model and it makes
/// `Σ = z/s` exactly zero -- so the three-variable version of this
/// fixture stopped reproducing gh#737 the moment that path landed,
/// while still passing every assertion below. One free variable puts
/// the barrier back in play.
///
/// `T > PIN` pulls `x` against the upper bound the equality already
/// holds it on, so bound and equality bind together and the multiplier
/// split between them is not unique. Every coordinate answers the
/// perturbation with the same `Δ`: `x` because `g₀` says so, `w`
/// because `g₂` does.
///
/// `w` and `g₂` are not decoration. Without them the solve reaches the
/// same degenerate point with a `Σ` two hundred times smaller, small
/// enough that the equality survives and the defect does not appear —
/// which is why the two-variable version of this model reported in the
/// issue does not reproduce it.
struct PinnedToBound {
    /// Objective target for `x`, above `PIN`.
    t: Number,
}

impl TNLP for PinnedToBound {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 5,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = PIN;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = PIN;
        b.g_u[1] = PIN;
        b.g_l[2] = 0.0;
        b.g_u[2] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.5;
        sp.x[1] = 0.5;
        sp.x[2] = 0.5;
        sp.x[3] = 0.5;
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            0.5 * (x[0] - self.t).powi(2) + 0.5 * (x[2] - 1.0).powi(2) + 0.5 * (x[3] - 1.0).powi(2),
        )
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - self.t;
        g[1] = 0.0;
        g[2] = x[2] - 1.0;
        g[3] = x[3] - 1.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - x[1];
        g[1] = x[1];
        g[2] = x[2] - x[0];
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
                for (k, &(r, c)) in [(0, 0), (0, 1), (1, 1), (2, 2), (2, 0)].iter().enumerate() {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, -1.0, 1.0, 1.0, -1.0]);
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
                let rs: [Index; 3] = [0, 2, 3];
                for (k, &r) in rs.iter().enumerate() {
                    irow[k] = r;
                    jcol[k] = r;
                }
            }
            SparsityRequest::Values { values } => {
                values.fill(obj_factor);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

/// Solve the fixture and hold the session open for the sensitivity
/// calls. `bound_relax_factor = 0` is what the Pyomo frontend's own
/// sensitivity session sets, and what puts the converged point on the
/// declared bound rather than `1e-8` outside it.
fn solved(t: Number, crossover: bool) -> Solver {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_string_value("sb", "yes", true, false).unwrap();
        o.set_numeric_value("bound_relax_factor", 0.0, true, false)
            .unwrap();
        o.set_string_value(
            "crossover",
            if crossover { "yes" } else { "no" },
            true,
            false,
        )
        .unwrap();
    }
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(PinnedToBound { t }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "fixture must converge (t={t:e}, crossover={crossover}); got {status:?}",
    );
    let x = solver.converged().expect("a converged state").x.clone();
    assert!(
        (x[0] - PIN).abs() < 1e-9,
        "the fixture only says anything with x on its bound; got x = {x:?}",
    );
    solver
}

/// The whole column moves with the parameter, on both the interior and
/// the crossed-over point. `x` reads the equality's `1`, `w` reads it
/// through `g₂`, and the parameter's own Var takes the delta whole
/// rather than half of it.
#[test]
fn a_variable_pinned_to_its_own_bound_takes_the_full_step() {
    for crossover in [false, true] {
        let solver = solved(2.0, crossover);
        let dx = solver
            .parametric_step(&[1 as Index], &[DELTA])
            .expect("the parametric step");
        for (i, name) in ["x", "p", "w"].iter().enumerate() {
            assert!(
                (dx[i] - DELTA).abs() < 1e-9,
                "crossover={crossover}: d {name} = {:e}, want {DELTA:e} (whole step was {dx:?})",
                dx[i],
            );
        }
    }
}

/// The same through the path `estimate()` takes by default: a
/// degenerate base point routes to the directional step, which reaches
/// the factor through the released solves rather than the plain one.
#[test]
fn the_directional_step_reaches_the_same_answer() {
    for crossover in [false, true] {
        let solver = solved(2.0, crossover);
        let (dx, _released, _work) = solver
            .parametric_step_directional(&[1 as Index], &[DELTA], 8)
            .expect("the directional step");
        for (i, name) in ["x", "p", "w"].iter().enumerate() {
            assert!(
                (dx[i] - DELTA).abs() < 1e-9,
                "crossover={crossover}: d {name} = {:e}, want {DELTA:e} (whole step was {dx:?})",
                dx[i],
            );
        }
    }
}

/// The barrier diagonal the sensitivity path reports is the one it
/// factors with, so the ceiling has to be visible here too. Pre-fix
/// this read `6.88e27`; the ceiling for a unit Jacobian coefficient is
/// `7.04e13`, and this fixture's objective scaling leaves the reported
/// (natural-units) value equal to it.
#[test]
fn the_barrier_diagonal_stays_under_the_ceiling() {
    let solver = solved(2.0, false);
    let report = solver.classify_activity().expect("activity report");
    assert!(
        report.var_sigma[0] < 1e20,
        "Sigma on the pinned variable is {:e}; the ceiling should have caught it",
        report.var_sigma[0],
    );
    assert!(
        report.var_sigma[0] > 1e10,
        "Sigma on the pinned variable is {:e}; the bound is still a bound, not a release",
        report.var_sigma[0],
    );
}

/// A `t` far enough above the bound converges through a different
/// multiplier split, lands two hundred times short of the ceiling, and
/// was answering correctly before the fix. It still does: the ceiling
/// binds on nothing here, and the answer is the same to the last bit.
#[test]
fn a_point_that_never_needed_the_ceiling_is_unchanged() {
    let solver = solved(1e3, false);
    let dx = solver
        .parametric_step(&[1 as Index], &[DELTA])
        .expect("the parametric step");
    for (i, name) in ["x", "p", "w"].iter().enumerate() {
        assert!(
            (dx[i] - DELTA).abs() < 1e-12,
            "d {name} = {:e}, want {DELTA:e}",
            dx[i],
        );
    }
}

// ---------------------------------------------------------------------------
// The same classification over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and asserts that the classification over an engine built from
// that report is the same as pounce's.

#[test]
fn the_barrier_diagonal_stays_under_the_ceiling_over_an_outside_engine() {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_string_value("sb", "yes", true, false).unwrap();
        o.set_numeric_value("bound_relax_factor", 0.0, true, false)
            .unwrap();
        o.set_string_value("crossover", "no", true, false).unwrap();
    }
    app.initialize().unwrap();
    let (solver, model, status) = recorded_solve(app, PinnedToBound { t: 2.0 });
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "solve failed: {status:?}",
    );
    same_classification("pinned to its bound", &solver, &model, Relaxation::None);
}
