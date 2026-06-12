//! H2 regression: pin-constraint → KKT-row mapping must translate the
//! user's full-g index through the c/d split (`full_g_to_c_block`), not
//! index `y_c` directly.
//!
//! The `y_c` multiplier block holds **equality rows only**. When an
//! inequality precedes a pinned equality in the user's `g(x)` ordering,
//! the inequality lands in the `d` block and every later equality's
//! position in `y_c` is shifted down by one. The pre-fix code computed
//! the KKT row as `n_x + n_s + user_g_index`, silently pinning the wrong
//! constraint (or a `y_d`/slack row) on any problem with an inequality
//! before a pinned equality.
//!
//! Fixture (one inactive inequality, then three equalities):
//!
//! ```text
//!   min x0²
//!   s.t.  g0:  x0 + x1 + x2 <= 1000     (INEQUALITY, inactive ⇒ d block)
//!         g1:  x0 - x1 - x2  = 0        (couples x0 to x1+x2)
//!         g2:  x1            = p1        (fixes parameter x1)
//!         g3:  x2            = p2        (fixes parameter x2)
//! ```
//!
//! Solution: x1 = p1, x2 = p2, x0 = x1 + x2. Pinning g2 (the x1-fixing
//! equality) and perturbing its RHS by Δ moves x1 by Δ and, through g1,
//! x0 by Δ — x2 is untouched: `dx = [Δ, Δ, 0]`.
//!
//! c/d split: g0 → d, {g1,g2,g3} → c-block rows {0,1,2}. So user g-index
//! 2 → c-block row 1. The pre-fix `y_c_offset + 2` instead selects
//! c-block row 2 = g3 (the x2-fixing equality), yielding the WRONG
//! `dx = [Δ, 0, Δ]`. This test asserts the correct `[Δ, Δ, 0]`, so it
//! fails pre-fix and passes post-fix.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::Solver;

/// `p1`, `p2` are the nominal RHS values of the two parameter-fixing
/// equalities g2, g3.
struct LeadingInequalityTNLP {
    p1: Number,
    p2: Number,
}

impl TNLP for LeadingInequalityTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 4,
            nnz_jac_g: 8,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        // g0: one-sided inequality (≤ 1000), inactive at the optimum.
        b.g_l[0] = -1.0e19;
        b.g_u[0] = 1000.0;
        // g1: x0 - x1 - x2 = 0.
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        // g2: x1 = p1.
        b.g_l[2] = self.p1;
        b.g_u[2] = self.p1;
        // g3: x2 = p2.
        b.g_l[3] = self.p2;
        b.g_u[3] = self.p2;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.p1 + self.p2;
        sp.x[1] = self.p1;
        sp.x[2] = self.p2;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 0.0;
        g[2] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[2];
        g[1] = x[0] - x[1] - x[2];
        g[2] = x[1];
        g[3] = x[2];
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
                let rs: [Index; 8] = [0, 0, 0, 1, 1, 1, 2, 3];
                let cs: [Index; 8] = [0, 1, 2, 0, 1, 2, 1, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                // g0: [1, 1, 1]; g1: [1, -1, -1]; g2: [1]; g3: [1].
                values.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]);
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
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                // Only x0² is nonlinear; constraints are all linear.
                values[0] = 2.0 * obj_factor;
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

#[test]
fn parametric_step_translates_pin_index_through_cd_split() {
    let tnlp: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(LeadingInequalityTNLP { p1: 1.0, p2: 1.0 }));
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

    // Pin g2 (user index 2 — the x1-fixing equality) and perturb its RHS
    // by +0.1. Physically: x1 += 0.1, x0 = x1 + x2 += 0.1, x2 unchanged.
    let delta = 0.1;
    let dx = solver
        .parametric_step(&[2], &[delta])
        .expect("parametric_step ok");
    assert_eq!(dx.len(), 3);

    // Correct mapping ⇒ [Δ, Δ, 0]. The pre-fix bug pins g3 instead,
    // giving [Δ, 0, Δ]; the dx[1]/dx[2] asserts below fail in that case.
    assert!(
        (dx[0] - delta).abs() < 1e-7,
        "dx[0] = {} expected ≈ {delta}",
        dx[0]
    );
    assert!(
        (dx[1] - delta).abs() < 1e-7,
        "dx[1] = {} expected ≈ {delta} (pinning x1's constraint must move x1; \
         pre-fix bug pins x2's constraint and leaves this 0)",
        dx[1]
    );
    assert!(
        dx[2].abs() < 1e-7,
        "dx[2] = {} expected ≈ 0 (x2's constraint was NOT pinned; \
         pre-fix bug pins it and moves this by {delta})",
        dx[2]
    );
}

#[test]
fn parametric_step_errors_on_pinned_inequality() {
    let tnlp: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(LeadingInequalityTNLP { p1: 1.0, p2: 1.0 }));
    let mut solver = Solver::new(make_app(), tnlp);
    solver.solve();

    // g0 (user index 0) is an inequality — it lives in the d block, not
    // y_c, so it cannot be pinned. The corrected mapping must reject it
    // rather than silently selecting a y_c row.
    let res = solver.parametric_step(&[0], &[0.1]);
    assert!(
        res.is_err(),
        "pinning an inequality constraint must error, got {res:?}"
    );
}

#[test]
fn reduced_hessian_errors_on_pinned_inequality() {
    let tnlp: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(LeadingInequalityTNLP { p1: 1.0, p2: 1.0 }));
    let mut solver = Solver::new(make_app(), tnlp);
    solver.solve();

    // Same c/d-split guard on the `compute_reduced_hessian` pin path.
    let res = solver.compute_reduced_hessian(&[0], 1.0);
    assert!(
        res.is_err(),
        "reduced Hessian over a pinned inequality must error, got {res:?}"
    );
}
