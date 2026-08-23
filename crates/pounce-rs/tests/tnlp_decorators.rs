//! `CountingTnlp` and `SeededTnlp` are reachable from the library facade.
//!
//! Both wrappers used to live in `crates/pounce-cli/src/`, so the only way to
//! count a solve's evaluations or to warm-start one from a chosen iterate was
//! to be the `pounce` binary — even though neither wrapper has any CLI or
//! `.nl` coupling and both are generic over `dyn TNLP`. This test is the
//! contract that they stay reachable: if the facade stops exporting enough to
//! write these sequences, it stops compiling.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::prelude::*;
use pounce_rs::{CountingTnlp, IpoptApplication, SeededTnlp};

/// min (x₀ − 1)² + (x₁ − 2)²  s.t.  x₀ + x₁ == 3,  0 ≤ x ≤ 5.
/// Optimum (1, 2), where the equality is already satisfied.
struct Quad;

impl TNLP for Quad {
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
        b.x_l.copy_from_slice(&[0.0, 0.0]);
        b.x_u.copy_from_slice(&[5.0, 5.0]);
        b.g_l[0] = 3.0;
        b.g_u[0] = 3.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 2.0);
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values.copy_from_slice(&[1.0, 1.0]),
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[2.0 * obj_factor, 2.0 * obj_factor]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn quiet_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app
}

/// A library caller can read the evaluation counts the CLI prints in its
/// end-of-run summary, and they reflect the solve that actually ran.
#[test]
fn counting_tnlp_reports_evaluation_counts_to_a_library_caller() {
    let counting = Rc::new(RefCell::new(CountingTnlp::new(
        Rc::new(RefCell::new(Quad)) as Rc<RefCell<dyn TNLP>>
    )));

    let mut app = quiet_app();
    let status = app.optimize_tnlp(Rc::clone(&counting) as Rc<RefCell<dyn TNLP>>);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);

    let c = counting.borrow();
    assert!(c.n_obj.get() > 0, "objective evaluations must be counted");
    assert!(c.n_grad_f.get() > 0, "gradient evaluations must be counted");
    assert!(c.n_g.get() > 0, "constraint evaluations must be counted");
    assert!(c.n_jac_g.get() > 0, "Jacobian evaluations must be counted");
    assert!(c.n_h.get() > 0, "Hessian evaluations must be counted");

    // The wrapper also captures the finalized solution, which is the
    // fallback the CLI reads for routes that bypass `on_converged`.
    let (x, _lambda) = c
        .captured_solution()
        .expect("finalize_solution must have been captured");
    assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 2.0).abs() < 1e-6);
}

/// `SeededTnlp` overrides the starting point, so a library caller can
/// re-solve from a chosen iterate. Seeding the optimum itself makes the
/// effect observable without asserting an iteration count: the seeded solve
/// starts feasible and optimal, so it costs strictly fewer objective
/// evaluations than the same solve from the problem's own `x₀`.
#[test]
fn seeded_tnlp_warm_starts_from_a_caller_supplied_iterate() {
    fn eval_count(seed: Option<Vec<Number>>) -> i32 {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Quad));
        let target: Rc<RefCell<dyn TNLP>> = match seed {
            Some(x) => Rc::new(RefCell::new(SeededTnlp::new(inner, x))),
            None => inner,
        };
        let counting = Rc::new(RefCell::new(CountingTnlp::new(target)));
        let mut app = quiet_app();
        let status = app.optimize_tnlp(Rc::clone(&counting) as Rc<RefCell<dyn TNLP>>);
        assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
        let n = counting.borrow().n_obj.get();
        n
    }

    let cold = eval_count(None);
    let warm = eval_count(Some(vec![1.0, 2.0]));
    assert!(
        warm < cold,
        "seeding the optimum must cost fewer objective evaluations \
         than the cold start (warm={warm}, cold={cold})"
    );
}

/// A seed whose length does not match the starting-point buffer is ignored
/// rather than written into the wrong space — the guard that matters when
/// presolve or fixed-variable elimination changed the coordinate count.
#[test]
fn seeded_tnlp_ignores_a_wrong_length_seed() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Quad));
    let seeded: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(SeededTnlp::new(inner, vec![1.0, 2.0, 3.0])));

    let mut app = quiet_app();
    assert_eq!(
        app.optimize_tnlp(seeded),
        ApplicationReturnStatus::SolveSucceeded,
        "a mismatched seed must fall back to the problem's own start"
    );
}
