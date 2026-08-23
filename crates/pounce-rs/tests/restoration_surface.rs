//! The facade ships the restoration phase.
//!
//! # What was wrong
//!
//! The restoration phase is not an optional extra — it is part of the NLP
//! path. When the filter line search cannot make progress, the solver drops
//! into an ℓ₁-feasibility sub-IPM and continues from what it finds. But
//! `pounce-algorithm` cannot install it (`pounce-restoration` depends on *it*,
//! not the reverse), so every frontend has to wire it up, and each one did so
//! by pasting the same ten lines. The CLI, the C interface and the Python
//! extension pasted it. `pounce-rs` — the crate whose own docs call it "the
//! Rust counterpart to the one-import `import pounce` Python API" — did not
//! depend on `pounce-restoration` at all.
//!
//! The result was not a missing feature but a *worse answer* from the same
//! solver: a model that needs restoration stopped at `Restoration_Failed`
//! here while the CLI solved it. 10 of the 71 `.nl` fixtures in the CLI corpus
//! invoke restoration, and most of them — `cresc4`, `deb7`, `eigena2`,
//! `eigmaxa`, `pooling_rt2stp` — succeed through it.
//!
//! These tests pin the fix on a model small enough to state in one line.

use pounce_rs::prelude::*;
use pounce_rs::{ApplicationReturnStatus, IpoptApplication};
use std::cell::RefCell;
use std::rc::Rc;

/// `min x  s.t.  x² = 2,  −10 ≤ x ≤ 10`, started at `x = 1e-8`.
///
/// The start is the whole point. `∇g = 2x` is ~0 there, so the linearised
/// constraint `2x·d = 2 − x²` has no useful solution and the filter line
/// search cannot reduce infeasibility along it. That is exactly the condition
/// restoration exists to handle: it minimises the constraint violation
/// directly, lands somewhere with a usable Jacobian, and the main IPM finishes
/// from there. Without restoration the solve has nowhere to go.
struct NeedsRestoration;

impl TNLP for NeedsRestoration {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = -10.0;
        b.x_u[0] = 10.0;
        b.g_l[0] = 2.0;
        b.g_u[0] = 2.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 1e-8;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 1.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0];
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * x.map(|x| x[0]).unwrap_or(0.0);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = 2.0 * lambda.map(|l| l[0]).unwrap_or(0.0);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn quiet(app: &mut IpoptApplication) {
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
}

/// The regression: `pounce_rs::application()` solves it, and the restoration
/// phase is what got it there.
#[test]
fn the_facade_solves_a_model_that_needs_restoration() {
    let mut app = pounce_rs::application();
    quiet(&mut app);
    let status = app.optimize_tnlp(Rc::new(RefCell::new(NeedsRestoration)));

    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "x² = 2 from x₀ = 1e-8 is solvable, and the CLI solves it"
    );
    assert!(
        app.statistics().restoration_calls > 0,
        "the solve must actually go through restoration — otherwise this model \
         has stopped testing what it was chosen to test"
    );
}

/// The other half, kept explicit so the regression cannot be "fixed" by the
/// model quietly becoming easy: a bare application still fails, because it has
/// no restoration phase to fall into.
///
/// This is not a wish about `IpoptApplication::new()`. `pounce-algorithm`
/// genuinely cannot install restoration itself, so the bare constructor is
/// documented as the lower-level one and [`pounce_rs::application()`] is the
/// facade's answer. What must never happen again is the *facade* handing back
/// the unwired application.
#[test]
fn a_bare_application_still_has_no_restoration() {
    let mut app = IpoptApplication::new();
    quiet(&mut app);
    let status = app.optimize_tnlp(Rc::new(RefCell::new(NeedsRestoration)));

    assert_eq!(
        status,
        ApplicationReturnStatus::RestorationFailed,
        "a bare IpoptApplication has no restoration phase; if this now succeeds, \
         restoration has been made reachable without the install call and the \
         facade's constructor doc needs revisiting"
    );
}

/// `install_default_restoration` reaches the same place as the constructor,
/// for callers who already hold an application.
#[test]
fn installing_on_an_existing_application_works_too() {
    let mut app = IpoptApplication::new();
    pounce_rs::install_default_restoration(&mut app);
    quiet(&mut app);
    let status = app.optimize_tnlp(Rc::new(RefCell::new(NeedsRestoration)));
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(app.statistics().restoration_calls > 0);
}

/// The ergonomic builder path gets it without the caller knowing any of this.
#[test]
fn the_nlp_builder_gets_restoration_by_default() {
    use pounce_rs::builder::{Nlp, Problem};

    struct P;
    impl Problem for P {
        fn objective(&self, x: &[f64]) -> f64 {
            x[0]
        }
        fn n_constraints(&self) -> usize {
            1
        }
        fn constraints(&self, x: &[f64], g: &mut [f64]) {
            g[0] = x[0] * x[0];
        }
    }

    let sol = Nlp::new(P)
        .var_bounds(&[-10.0], &[10.0])
        .constraint_bounds(&[2.0], &[2.0])
        .x0(&[1e-8])
        .option_int("print_level", 0)
        .solve();

    assert!(
        sol.success,
        "the builder must reach the same verdict as the CLI, got {:?}",
        sol.status
    );
    assert!(sol.stats.restoration_calls > 0);
}
