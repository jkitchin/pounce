//! Head-to-head iteration count: the *same* convex QP solved by the NLP
//! filter-IPM (POUNCE's general solver) and by the specialized
//! convex-QP interior-point method in `pounce-convex`.
//!
//! This is the check behind the plan's central claim
//! (`dev-notes/lp-qp-routing.md`): a specialized convex-QP IPM with
//! Mehrotra predictor-corrector should reach the solution in *fewer*
//! interior-point iterations than routing the same problem through the
//! general NLP path. We solve a scalable equality-constrained convex QP
//! both ways and assert (a) both find the same optimum and (b) the QP
//! path takes no more iterations than the NLP path.
//!
//! The QP is `min ½xᵀPx + cᵀx  s.t.  Ax = b`, with `P` SPD
//! (diagonally dominant) and a handful of dense equality rows, sized by
//! `N`. Large enough that the NLP path needs several iterations, so the
//! comparison is meaningful (unlike the n=2 builtins, where a quadratic
//! is solved almost immediately by either method).

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_convex::{solve_qp_ipm, QpOptions, QpProblem, QpStatus, Triplet};
use pounce_feral::FeralSolverInterface;
use pounce_linsol::SparseSymLinearSolverInterface;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Build a scalable *bound-constrained* convex QP — the regime where the
/// central path is non-trivial and the IPM-QP-vs-IPM-NLP iteration
/// comparison is meaningful. `P = diag(d) + sub-diagonal coupling` (SPD
/// by diagonal dominance). The linear term `c` pushes the unconstrained
/// optimum below the lower bounds, so many bounds are active and the
/// solver must traverse the central path. Bounds `0 ≤ x ≤ ub` are
/// written as inequality rows `−x ≤ 0` and `x ≤ ub`.
fn make_qp(n: usize) -> QpProblem {
    let mut p_lower = Vec::new();
    for i in 0..n {
        p_lower.push(Triplet::new(i, i, 2.0 + (i % 5) as f64));
        if i > 0 {
            p_lower.push(Triplet::new(i, i - 1, 0.5));
        }
    }
    // Negative linear term → unconstrained optimum is positive and large,
    // so the upper bounds bind for many components.
    let c: Vec<f64> = (0..n).map(|i| -2.0 - (i % 7) as f64).collect();

    // Bounds 0 ≤ x_i ≤ 1 as 2n inequality rows.
    let mut g = Vec::new();
    let mut h = Vec::new();
    for i in 0..n {
        g.push(Triplet::new(2 * i, i, 1.0)); // x_i ≤ 1
        h.push(1.0);
        g.push(Triplet::new(2 * i + 1, i, -1.0)); // −x_i ≤ 0
        h.push(0.0);
    }

    QpProblem {
        n,
        p_lower,
        c,
        a: vec![],
        b: vec![],
        g,
        h,
        lb: vec![],
        ub: vec![],
    }
}

/// TNLP adapter wrapping a `QpProblem` so the NLP filter-IPM can solve
/// the identical problem. Only equality constraints are used here.
/// Wraps a bound-constrained convex QP `min ½xᵀPx+cᵀx, 0 ≤ x ≤ ub` as a
/// TNLP. The bounds are expressed as TNLP *variable* bounds (the natural
/// NLP encoding), so the NLP filter-IPM solves exactly the same
/// mathematical problem the `pounce-convex` QP solver sees as bound rows.
struct QpAsTnlp {
    prob: QpProblem,
    /// Variable lower/upper bounds (length n).
    lb: Vec<f64>,
    ub: Vec<f64>,
    /// Lower-triangle Hessian entries (constant) as (row, col, val).
    h_entries: Vec<(usize, usize, f64)>,
    captured_obj: RefCell<Option<f64>>,
    captured_x: RefCell<Option<Vec<f64>>>,
}

impl QpAsTnlp {
    fn new(prob: QpProblem, lb: Vec<f64>, ub: Vec<f64>) -> Self {
        let h_entries: Vec<(usize, usize, f64)> =
            prob.p_lower.iter().map(|t| (t.row, t.col, t.val)).collect();
        QpAsTnlp {
            prob,
            lb,
            ub,
            h_entries,
            captured_obj: RefCell::new(None),
            captured_x: RefCell::new(None),
        }
    }
}

impl TNLP for QpAsTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.prob.n as Index,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: self.h_entries.len() as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.lb);
        b.x_u.copy_from_slice(&self.ub);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.iter_mut().for_each(|v| *v = 0.0);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let mut px = vec![0.0; self.prob.n];
        self.prob.p_mul_add_pub(x, &mut px);
        let mut f = 0.0;
        for i in 0..self.prob.n {
            f += 0.5 * x[i] * px[i] + self.prob.c[i] * x[i];
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad: &mut [Number]) -> bool {
        grad.iter_mut().zip(&self.prob.c).for_each(|(g, c)| *g = *c);
        self.prob.p_mul_add_pub(x, grad);
        true
    }

    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        // No general constraints — bounds are variable bounds.
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _mode: SparsityRequest<'_>,
    ) -> bool {
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
        // Constraints are linear, so the Lagrangian Hessian is just
        // obj_factor * P.
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (i, (r, c, _)) in self.h_entries.iter().enumerate() {
                    irow[i] = *r as Index;
                    jcol[i] = *c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                for (i, (_, _, v)) in self.h_entries.iter().enumerate() {
                    values[i] = obj_factor * v;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        *self.captured_obj.borrow_mut() = Some(sol.obj_value);
        *self.captured_x.borrow_mut() = Some(sol.x.to_vec());
    }
}

fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(FeralSolverInterface::new())
}

#[test]
fn qp_ipm_uses_no_more_iterations_than_nlp() {
    let n = 50;
    let prob = make_qp(n);
    let lb = vec![0.0; n];
    let ub = vec![1.0; n];

    // --- QP path ---
    let qp_sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
    assert_eq!(
        qp_sol.status,
        QpStatus::Optimal,
        "QP IPM failed: {:?}",
        qp_sol.status
    );
    let qp_iters = qp_sol.iters;
    let qp_obj = qp_sol.obj;

    // --- NLP path on the identical problem ---
    let mut app = IpoptApplication::new();
    app.initialize().expect("init");
    let _ = app.options_mut().read_from_str("print_level 0\n", true);
    let tnlp_rc = Rc::new(RefCell::new(QpAsTnlp::new(prob.clone(), lb, ub)));
    let tnlp: Rc<RefCell<dyn TNLP>> = tnlp_rc.clone();
    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "NLP solve failed: {status:?}"
    );
    let nlp_iters = app.statistics().iteration_count as usize;
    let nlp_obj = tnlp_rc
        .borrow()
        .captured_obj
        .borrow()
        .expect("NLP finalize captured objective");

    // --- both reached the same optimum (validates the comparison) ---
    assert!(
        (qp_obj - nlp_obj).abs() < 1e-5,
        "objectives disagree: QP={qp_obj}, NLP={nlp_obj}"
    );

    eprintln!(
        "n={n}: QP IPM iters = {qp_iters}, NLP IPM iters = {nlp_iters} (obj QP={qp_obj:.6}, NLP={nlp_obj:.6})"
    );

    // The specialized QP path should not take more interior-point
    // iterations than the general NLP path on this convex QP.
    assert!(
        qp_iters <= nlp_iters,
        "expected QP iters ({qp_iters}) <= NLP iters ({nlp_iters})"
    );
}
