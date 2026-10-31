//! A stop the intermediate callback asks for is reported as
//! `User_Requested_Stop` wherever it lands, including inside the bets the
//! solver places after its loop has a result (PR #1103 review):
//!
//! * **gh #797's negative-curvature escape.** `honour_neg_curv_floor`
//!   restores the certified point the escape left when the continuation does
//!   not beat it, and reported that point's `Success` even over a user stop,
//!   so a caller driving several solves carried on after being told to stop.
//! * **gh #983's scale audit.** It re-solves only after a converged base. A
//!   stop inside the re-solve first lost the stop (the base `Optimal` was put
//!   back), and then, once honored, returned the re-solve's mid-run iterate
//!   in place of the converged base answer. The base point must ship, under
//!   the stop.
//!
//! Each test sweeps a one-time stop over every callback of the unstopped
//! run, so it reaches the hook whatever the trajectory on a given build.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Copy)]
enum Model {
    /// `min x0·x1  s.t. x0 + x1 = 2, 0 <= x <= 4` from `(0.01, 0.01)`: the
    /// first Newton step lands on the constrained *maximum* `(1, 1)`, which
    /// the gh #797 escape leaves for an endpoint with `f = 0`.
    Saddle,
    /// `min 1e-6·(x0 + 2 x1)  s.t. x0 + x1 >= 1, 0 <= x <= 10` from
    /// `(5, 5)`: a small objective the gh #983 audit re-solves after it
    /// converges.
    SmallLp,
}

struct Stopper {
    model: Model,
    calls: usize,
    stop_at: usize,
    x: Vec<Number>,
}

impl TNLP for Stopper {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: match self.model {
                Model::Saddle => 1,
                Model::SmallLp => 0,
            },
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(0.0);
        match self.model {
            Model::Saddle => {
                b.x_u.fill(4.0);
                b.g_l[0] = 2.0;
                b.g_u[0] = 2.0;
            }
            Model::SmallLp => {
                b.x_u.fill(10.0);
                b.g_l[0] = 1.0;
                b.g_u[0] = 2e19;
            }
        }
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(match self.model {
            Model::Saddle => &[0.01, 0.01],
            Model::SmallLp => &[5.0, 5.0],
        });
        true
    }
    fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
        Some(match self.model {
            Model::Saddle => x[0] * x[1],
            Model::SmallLp => 1e-6 * (x[0] + 2.0 * x[1]),
        })
    }
    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&match self.model {
            Model::Saddle => [x[1], x[0]],
            Model::SmallLp => [1e-6, 2e-6],
        });
        true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
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
        _: Option<&[Number]>,
        _: bool,
        obj_factor: Number,
        _: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        if let Model::Saddle = self.model {
            match mode {
                SparsityRequest::Structure { irow, jcol } => {
                    irow[0] = 1;
                    jcol[0] = 0;
                }
                SparsityRequest::Values { values } => values[0] = obj_factor,
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        self.x = sol.x.to_vec();
    }
    fn wants_intermediate_callback(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, _: IterStats, _: &IpoptData, _: &IpoptCq) -> bool {
        self.calls += 1;
        self.calls != self.stop_at
    }
}

/// `(status, callbacks made, reported x)` with a one-time stop at call
/// `stop_at`.
fn solve(
    model: Model,
    stop_at: usize,
    opts: &[(&str, &str)],
) -> (ApplicationReturnStatus, usize, Vec<Number>) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    for &(k, v) in opts {
        app.options_mut()
            .set_string_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let tnlp = Rc::new(RefCell::new(Stopper {
        model,
        calls: 0,
        stop_at,
        x: Vec::new(),
    }));
    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let t = tnlp.borrow();
    (status, t.calls, t.x.clone())
}

#[test]
fn a_stop_during_the_negative_curvature_escape_is_reported_as_a_stop() {
    let (status, total, x) = solve(Model::Saddle, 0, &[]);
    // The escape ran and paid off: the maximum `(1, 1)` was left for an
    // endpoint, so its continuation is in the swept range.
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(x[0] * x[1] < 1e-6, "no escape: {x:?}");
    for stop_at in 1..=total {
        let (status, calls, _) = solve(Model::Saddle, stop_at, &[]);
        assert_eq!(
            status,
            ApplicationReturnStatus::UserRequestedStop,
            "stop at call {stop_at} of {total}"
        );
        assert_eq!(calls, stop_at, "callbacks after the stop at {stop_at}");
    }
}

#[test]
fn a_stop_during_the_scale_audit_keeps_the_converged_base_answer() {
    let audit_off = [("solve_quality_audit", "no")];
    let (status, base_calls, base_x) = solve(Model::SmallLp, 0, &audit_off);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let (_, total, _) = solve(Model::SmallLp, 0, &[]);
    assert!(
        total > base_calls,
        "the audit did not re-solve: {total} calls"
    );
    // Every stop inside the audit's re-solve: the stop is reported, at the
    // base attempt's converged point, not the re-solve's mid-run iterate.
    for stop_at in base_calls + 1..=total {
        let (status, calls, x) = solve(Model::SmallLp, stop_at, &[]);
        let ctx = format!("stop at call {stop_at} of {total} (base {base_calls})");
        assert_eq!(status, ApplicationReturnStatus::UserRequestedStop, "{ctx}");
        assert_eq!(calls, stop_at, "callbacks after the stop; {ctx}");
        for (a, b) in x.iter().zip(&base_x) {
            assert!(
                (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                "{x:?} vs {base_x:?}; {ctx}"
            );
        }
    }
}
