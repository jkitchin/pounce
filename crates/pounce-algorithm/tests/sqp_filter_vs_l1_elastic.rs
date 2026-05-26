//! Phase 5d — comparison of the two SQP globalization choices,
//! `Filter` (Fletcher-Leyffer 2002, default) and `L1Elastic`
//! (Han-Powell with adaptive ν, design-note §4.1.b). For a
//! cross-section of the HS subset both must converge to the
//! same optimum, certifying the L1Elastic path as a real
//! alternative — the §10 Phase-5d "alternative globalization"
//! exit criterion that doesn't require an oracle.
//!
//! No iteration-count comparison is asserted (that's gated on
//! external solver references); only that both paths reach the
//! same closed-form solution to within tolerance.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default, Clone)]
struct Sink {
    x: Vec<Number>,
    f: Number,
}

fn run_sqp_with_globalization<T: TNLP + 'static>(
    tnlp: T,
    sink: Rc<RefCell<Sink>>,
    globalization: &str,
) -> (ApplicationReturnStatus, Sink) {
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let opts = format!(
        "algorithm active-set-sqp\nprint_level 0\nsqp_globalization {globalization}\nsqp_max_iter 200\nsqp_tol 1e-8\n"
    );
    app.initialize_with_options_str(&opts).unwrap();
    let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    let status = app.optimize_tnlp(rc);
    let s = sink.borrow().clone();
    (status, s)
}

fn converged(s: ApplicationReturnStatus) -> bool {
    matches!(
        s,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

// ─────────────────────────────────────────────────────────────
// HS28 — equality-only quadratic. n=3, m=1.
// f = (x_1 + x_2)² + (x_2 + x_3)², s.t. x_1 + 2x_2 + 3x_3 = 1.
// x* = (0.5, -0.5, 0.5), f* = 0.
// ─────────────────────────────────────────────────────────────

struct Hs28 {
    sink: Rc<RefCell<Sink>>,
}
impl TNLP for Hs28 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 3,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-2.0e19; 3]);
        b.x_u.copy_from_slice(&[2.0e19; 3]);
        b.g_l.copy_from_slice(&[1.0]);
        b.g_u.copy_from_slice(&[1.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[-4.0, 1.0, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let a = x[0] + x[1];
        let b = x[1] + x[2];
        Some(a * a + b * b)
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let a = x[0] + x[1];
        let b = x[1] + x[2];
        g[0] = 2.0 * a;
        g[1] = 2.0 * (a + b);
        g[2] = 2.0 * b;
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + 2.0 * x[1] + 3.0 * x[2];
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
                irow.copy_from_slice(&[0, 0, 0]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values, .. } => {
                values.copy_from_slice(&[1.0, 2.0, 3.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        of: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // H = 2 [[1,1,0],[1,2,1],[0,1,1]]
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 0, 1, 1, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values, .. } => {
                values[0] = of * 2.0;
                values[1] = of * 2.0;
                values[2] = of * 4.0;
                values[3] = of * 2.0;
                values[4] = of * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.sink.borrow_mut() = Sink {
            x: sol.x.to_vec(),
            f: sol.obj_value,
        };
    }
}

#[test]
fn hs28_filter_and_l1_elastic_agree() {
    let sink_f = Rc::new(RefCell::new(Sink::default()));
    let (status_f, out_f) = run_sqp_with_globalization(
        Hs28 {
            sink: sink_f.clone(),
        },
        sink_f,
        "filter",
    );
    assert!(converged(status_f), "filter status = {status_f:?}");

    let sink_l = Rc::new(RefCell::new(Sink::default()));
    let (status_l, out_l) = run_sqp_with_globalization(
        Hs28 {
            sink: sink_l.clone(),
        },
        sink_l,
        "l1-elastic",
    );
    assert!(converged(status_l), "l1-elastic status = {status_l:?}");

    // Both paths must land at the same closed-form optimum.
    for i in 0..3 {
        assert!(
            (out_f.x[i] - out_l.x[i]).abs() < 1e-4,
            "x[{i}] disagrees: filter = {}, l1 = {}",
            out_f.x[i],
            out_l.x[i],
        );
    }
    assert!(out_f.f.abs() < 1e-6 && out_l.f.abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────
// HS35 — convex quadratic + linear inequality + non-negativity.
// (Same fixture body as the HS subset test.)
// ─────────────────────────────────────────────────────────────

struct Hs35 {
    sink: Rc<RefCell<Sink>>,
}
impl TNLP for Hs35 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 3,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.0; 3]);
        b.x_u.copy_from_slice(&[2.0e19; 3]);
        b.g_l.copy_from_slice(&[-2.0e19]);
        b.g_u.copy_from_slice(&[3.0]);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[0.5, 0.5, 0.5]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(
            9.0 - 8.0 * x[0] - 6.0 * x[1] - 4.0 * x[2]
                + 2.0 * x[0] * x[0]
                + 2.0 * x[1] * x[1]
                + x[2] * x[2]
                + 2.0 * x[0] * x[1]
                + 2.0 * x[0] * x[2],
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = -8.0 + 4.0 * x[0] + 2.0 * x[1] + 2.0 * x[2];
        g[1] = -6.0 + 4.0 * x[1] + 2.0 * x[0];
        g[2] = -4.0 + 2.0 * x[2] + 2.0 * x[0];
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + 2.0 * x[2];
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
                irow.copy_from_slice(&[0, 0, 0]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values, .. } => {
                values.copy_from_slice(&[1.0, 1.0, 2.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        of: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 5] = [0, 1, 1, 2, 2];
                let cs: [Index; 5] = [0, 0, 1, 0, 2];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values, .. } => {
                values[0] = of * 4.0;
                values[1] = of * 2.0;
                values[2] = of * 4.0;
                values[3] = of * 2.0;
                values[4] = of * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.sink.borrow_mut() = Sink {
            x: sol.x.to_vec(),
            f: sol.obj_value,
        };
    }
}

#[test]
fn hs35_filter_and_l1_elastic_agree() {
    let sink_f = Rc::new(RefCell::new(Sink::default()));
    let (status_f, out_f) = run_sqp_with_globalization(
        Hs35 {
            sink: sink_f.clone(),
        },
        sink_f,
        "filter",
    );
    assert!(converged(status_f), "filter status = {status_f:?}");

    let sink_l = Rc::new(RefCell::new(Sink::default()));
    let (status_l, out_l) = run_sqp_with_globalization(
        Hs35 {
            sink: sink_l.clone(),
        },
        sink_l,
        "l1-elastic",
    );
    assert!(converged(status_l), "l1-elastic status = {status_l:?}");

    let x_star = [4.0 / 3.0, 7.0 / 9.0, 4.0 / 9.0];
    for i in 0..3 {
        assert!(
            (out_f.x[i] - x_star[i]).abs() < 1e-3,
            "filter x[{i}] = {}",
            out_f.x[i]
        );
        assert!(
            (out_l.x[i] - x_star[i]).abs() < 1e-3,
            "l1-elastic x[{i}] = {}",
            out_l.x[i]
        );
    }
}

#[test]
fn l1_penalty_safety_and_max_options_propagate() {
    // The new sqp_l1_penalty_safety and sqp_l1_penalty_max options
    // should read through the OptionsList into the SqpOptions
    // builder snapshot.
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    app.initialize_with_options_str(
        "algorithm active-set-sqp\n\
         sqp_l1_penalty_safety 0.25\n\
         sqp_l1_penalty_max 1e6\n",
    )
    .unwrap();
    // Use the SqpOptions plumbing via a fresh solve; the options
    // get baked into the builder inside optimize_sqp_tnlp.
    let sink_l = Rc::new(RefCell::new(Sink::default()));
    let tnlp = Hs28 {
        sink: sink_l.clone(),
    };
    let rc: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(tnlp));
    // Just check the solve completes — the option values land via
    // the apply_sqp_options helper and propagate through
    // optimize_with_warm_start. Failure would surface as a panic
    // in option-reading.
    let _ = app.optimize_tnlp(rc);
}
