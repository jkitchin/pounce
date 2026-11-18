//! Variable scaling as a TNLP wrapper (issue #486 stage 2).
//!
//! The derivative tests use finite differences against the wrapper's
//! OWN objective and constraints in scaled coordinates, rather than
//! restating the transform table. A multiply where a divide belongs,
//! or a factor taken from the row where the column was meant, fails
//! here; an implementation that merely agrees with itself does not
//! pass.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_common::types::Number;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::scaling_tnlp::wrap_with_scaling;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_nlp::tnlp_adapter::{DEFAULT_NLP_LOWER_BOUND_INF, DEFAULT_NLP_UPPER_BOUND_INF};

/// The registered defaults, which is what every test here means by
/// "a bound at or past the threshold is absent".
const LO: Number = DEFAULT_NLP_LOWER_BOUND_INF;
const UP: Number = DEFAULT_NLP_UPPER_BOUND_INF;

/// `min x0^2 + 3 x0 x1 + 2 x1^2  s.t.  x0 x1 = 1`, with a dense
/// Jacobian row and a full lower-triangular Hessian, so every row of
/// the transform table has something to act on.
struct Fixture {
    factors: Vec<Number>,
    use_x: bool,
    got: Rc<RefCell<Option<(Vec<Number>, Vec<Number>, Vec<Number>, Vec<Number>)>>>,
}

impl Fixture {
    fn new(factors: Vec<Number>) -> Self {
        Self {
            factors,
            use_x: true,
            got: Rc::new(RefCell::new(None)),
        }
    }
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[0.5, -2.0]);
        b.x_u.copy_from_slice(&[10.0, 4.0]);
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&[2.0, 0.5]);
        }
        if sp.init_z {
            sp.z_l.copy_from_slice(&[0.25, 0.5]);
            sp.z_u.copy_from_slice(&[0.125, 1.0]);
        }
        if sp.init_lambda {
            sp.lambda[0] = 0.75;
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0 * x[0] + 3.0 * x[1];
        grad_f[1] = 3.0 * x[0] + 4.0 * x[1];
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
                true
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("values call carries x");
                values[0] = x[1];
                values[1] = x[0];
                true
            }
        }
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1]);
                jcol.copy_from_slice(&[0, 0, 1]);
                true
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.map_or(0.0, |l| l[0]);
                values[0] = 2.0 * obj_factor;
                values[1] = 3.0 * obj_factor + lam;
                values[2] = 4.0 * obj_factor;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {
        *self.got.borrow_mut() = Some((
            sol.x.to_vec(),
            sol.z_l.to_vec(),
            sol.z_u.to_vec(),
            sol.lambda.to_vec(),
        ));
    }
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = self.use_x;
        req.x_scaling.copy_from_slice(&self.factors);
        *req.use_g_scaling = true;
        req.g_scaling[0] = 7.0;
        true
    }
}

const D: [Number; 2] = [2.0, 100.0];

fn wrapped() -> Rc<RefCell<dyn TNLP>> {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::new(D.to_vec())));
    wrap_with_scaling(inner, LO, UP)
        .expect("valid factors")
        .expect("non-trivial factors wrap")
}

#[test]
fn objective_gradient_is_consistent_with_the_scaled_objective() {
    let w = wrapped();
    let x = [5.0, 60.0];
    let mut g = [0.0; 2];
    assert!(w.borrow_mut().eval_grad_f(&x, true, &mut g));
    for i in 0..2 {
        let h = 1e-6 * x[i].abs().max(1.0);
        let (mut xp, mut xm) = (x, x);
        xp[i] += h;
        xm[i] -= h;
        let fp = w.borrow_mut().eval_f(&xp, true).unwrap();
        let fm = w.borrow_mut().eval_f(&xm, true).unwrap();
        let fd = (fp - fm) / (2.0 * h);
        assert!(
            (g[i] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
            "grad[{i}] = {} but finite difference = {fd}",
            g[i]
        );
    }
}

#[test]
fn jacobian_is_consistent_with_the_scaled_constraints() {
    let w = wrapped();
    let x = [5.0, 60.0];
    let (mut irow, mut jcol) = ([0; 2], [0; 2]);
    assert!(w.borrow_mut().eval_jac_g(
        None,
        true,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ));
    let mut vals = [0.0; 2];
    assert!(w.borrow_mut().eval_jac_g(
        Some(&x),
        true,
        SparsityRequest::Values { values: &mut vals },
    ));
    for k in 0..2 {
        let j = jcol[k] as usize;
        let h = 1e-6 * x[j].abs().max(1.0);
        let (mut xp, mut xm) = (x, x);
        xp[j] += h;
        xm[j] -= h;
        let (mut gp, mut gm) = ([0.0], [0.0]);
        assert!(w.borrow_mut().eval_g(&xp, true, &mut gp));
        assert!(w.borrow_mut().eval_g(&xm, true, &mut gm));
        let fd = (gp[0] - gm[0]) / (2.0 * h);
        assert!(
            (vals[k] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
            "jac[{k}] = {} but finite difference = {fd}",
            vals[k]
        );
    }
}

/// Pins the two-sided division: the Hessian must be the derivative of
/// the scaled Lagrangian gradient, so dividing by the row's factor
/// alone, or the column's alone, fails.
#[test]
fn hessian_is_consistent_with_the_scaled_lagrangian() {
    let w = wrapped();
    let x = [5.0, 60.0];
    let (sigma, lam) = (1.0, 0.75);
    let (mut irow, mut jcol) = ([0; 3], [0; 3]);
    assert!(w.borrow_mut().eval_h(
        None,
        true,
        sigma,
        None,
        true,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ));
    let mut vals = [0.0; 3];
    assert!(w.borrow_mut().eval_h(
        Some(&x),
        true,
        sigma,
        Some(&[lam]),
        true,
        SparsityRequest::Values { values: &mut vals },
    ));
    let lag_grad = |p: &[Number; 2]| -> [Number; 2] {
        let mut gf = [0.0; 2];
        assert!(w.borrow_mut().eval_grad_f(p, true, &mut gf));
        let mut jv = [0.0; 2];
        assert!(w.borrow_mut().eval_jac_g(
            Some(p),
            true,
            SparsityRequest::Values { values: &mut jv },
        ));
        [sigma * gf[0] + lam * jv[0], sigma * gf[1] + lam * jv[1]]
    };
    for k in 0..3 {
        let (r, c) = (irow[k] as usize, jcol[k] as usize);
        let h = 1e-5 * x[c].abs().max(1.0);
        let (mut xp, mut xm) = (x, x);
        xp[c] += h;
        xm[c] -= h;
        let fd = (lag_grad(&xp)[r] - lag_grad(&xm)[r]) / (2.0 * h);
        assert!(
            (vals[k] - fd).abs() <= 1e-4 * fd.abs().max(1.0),
            "hess[{k}] at ({r},{c}) = {} but finite difference = {fd}",
            vals[k]
        );
    }
}

#[test]
fn bounds_and_starting_point_move_into_scaled_space() {
    let w = wrapped();
    let (mut xl, mut xu, mut gl, mut gu) = ([0.0; 2], [0.0; 2], [0.0; 1], [0.0; 1]);
    assert!(w.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut xl,
        x_u: &mut xu,
        g_l: &mut gl,
        g_u: &mut gu,
    }));
    assert_eq!(xl, [0.5 * D[0], -2.0 * D[1]]);
    assert_eq!(xu, [10.0 * D[0], 4.0 * D[1]]);
    assert_eq!(gl, [1.0], "constraint bounds are untouched");

    let (mut x, mut zl, mut zu, mut lam) = ([0.0; 2], [0.0; 2], [0.0; 2], [0.0; 1]);
    assert!(w.borrow_mut().get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: true,
        z_l: &mut zl,
        z_u: &mut zu,
        init_lambda: true,
        lambda: &mut lam,
    }));
    assert_eq!(x, [2.0 * D[0], 0.5 * D[1]]);
    assert_eq!(zl, [0.25 / D[0], 0.5 / D[1]], "bound multipliers divide in");
    assert_eq!(zu, [0.125 / D[0], 1.0 / D[1]]);
    assert_eq!(lam, [0.75], "constraint multipliers are untouched");
}

#[test]
fn finalize_inverts_the_substitution() {
    let inner = Rc::new(RefCell::new(Fixture::new(D.to_vec())));
    let got = inner.borrow().got.clone();
    let dynamic: Rc<RefCell<dyn TNLP>> = inner;
    let w = wrap_with_scaling(dynamic, LO, UP).unwrap().unwrap();
    w.borrow_mut().finalize_solution(
        Solution {
            status: SolverReturn::Success,
            x: &[6.0, 250.0],
            z_l: &[0.5, 0.02],
            z_u: &[0.0, 0.0],
            g: &[1.0],
            lambda: &[0.75],
            obj_value: 42.0,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );
    let (x, zl, _zu, lam) = got.borrow().clone().expect("inner was finalized");
    assert_eq!(x, [3.0, 2.5], "x divides back into user units");
    assert_eq!(zl, [1.0, 2.0], "bound multipliers multiply back");
    assert_eq!(lam, [0.75], "constraint multipliers are untouched");
}

#[test]
fn objective_and_constraint_scaling_pass_through() {
    let w = wrapped();
    let (mut obj, mut ux, mut xs, mut ug, mut gs) = (0.0, true, [0.0; 2], false, [0.0; 1]);
    assert!(w.borrow_mut().get_scaling_parameters(ScalingRequest {
        obj_scaling: &mut obj,
        use_x_scaling: &mut ux,
        x_scaling: &mut xs,
        use_g_scaling: &mut ug,
        g_scaling: &mut gs,
        g_scaling_given: &mut [true],
    }));
    assert!(!ux, "the wrapper consumed the variable factors itself");
    assert!(ug, "constraint factors still reach the core");
    assert_eq!(gs, [7.0]);
    assert_eq!(obj, 1.0);
}

#[test]
fn factors_that_cannot_be_applied_are_refused() {
    for bad in [-2.0, 0.0, 1e-15] {
        let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::new(vec![bad, 1.0])));
        let err = match wrap_with_scaling(inner, LO, UP) {
            Err(e) => e,
            Ok(_) => panic!("factor {bad} must be refused"),
        };
        assert!(err.contains("scaling: variable 0"), "message was: {err}");
    }
}

#[test]
fn values_before_structure_does_not_panic() {
    // The algorithm always asks for structure first, but the wrapper
    // caches the pattern from that call, so a values-first caller must
    // still get scaled numbers rather than an index panic.
    let w = wrapped();
    let x = [5.0, 60.0];
    let mut jv = [0.0; 2];
    assert!(
        w.borrow_mut()
            .eval_jac_g(Some(&x), true, SparsityRequest::Values { values: &mut jv },)
    );
    assert_eq!(jv, [0.6 / D[0], 2.5 / D[1]]);
}

#[test]
fn nothing_wraps_without_non_trivial_factors() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::new(vec![1.0, 1.0])));
    assert!(
        wrap_with_scaling(inner, LO, UP).unwrap().is_none(),
        "all-ones needs no wrapper"
    );
    let mut f = Fixture::new(vec![2.0, 3.0]);
    f.use_x = false;
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(f));
    assert!(
        wrap_with_scaling(inner, LO, UP).unwrap().is_none(),
        "use_x_scaling = false needs no wrapper"
    );
}

/// The presolve flag asks whether a generic presolve wrapper sits
/// below, so that the entry point does not add a second one. Claiming
/// it here would make presolve decline every scaled problem.
#[test]
fn the_presolve_flag_is_forwarded_not_claimed() {
    let w = wrapped();
    assert!(
        !w.borrow().is_presolve_wrapper(),
        "a plain inner TNLP means no presolve wrapper below"
    );
}

/// Two variables with caller-chosen bounds and no constraints, for the
/// bound-side tests. `.nl` models reach the solver with absent bounds
/// written as the `±1e19` sentinel rather than a true infinity, which
/// is the case that matters here.
struct Bounded {
    x_l: [Number; 2],
    x_u: [Number; 2],
    factors: [Number; 2],
}

impl TNLP for Bounded {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&[0.0, 0.0]);
        }
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1])
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, grad_f: &mut [Number]) -> bool {
        grad_f[0] = 2.0 * x[0];
        grad_f[1] = 2.0 * x[1];
        true
    }
    fn eval_g(&mut self, _x: &[Number], _new_x: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _o: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        _m: SparsityRequest<'_>,
    ) -> bool {
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(&self.factors);
        *req.use_g_scaling = false;
        true
    }
}

/// A variable with no bounds must still have none after scaling.
///
/// The sentinel is a finite number, so `x_l *= s` with `s < 1` walks it
/// inside the threshold and the algorithm reads a real bound: the
/// variable picks up a barrier term and `z_L`/`z_U` multipliers that
/// the model never asked for. `s < 1` is the ordinary case for a model
/// whose variables are large.
#[test]
fn absent_bounds_stay_absent_under_scaling() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Bounded {
        x_l: [LO, 1.0],
        x_u: [UP, 2.0],
        factors: [1.0e-6, 2.0],
    }));
    let w = wrap_with_scaling(inner, LO, UP).unwrap().unwrap();

    let mut x_l = [0.0; 2];
    let mut x_u = [0.0; 2];
    assert!(w.borrow_mut().get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut [],
        g_u: &mut [],
    }));

    assert!(
        x_l[0] <= LO && x_u[0] >= UP,
        "the free variable gained bounds: [{}, {}] against thresholds [{LO}, {UP}]",
        x_l[0],
        x_u[0]
    );
    assert_eq!((x_l[1], x_u[1]), (2.0, 4.0), "real bounds still scale");
}

/// The mirror case: a real bound must not be scaled *past* a threshold,
/// where it would read as absent and vanish. Nothing downstream could
/// tell, so it is refused at setup, which is where there is a message.
#[test]
fn a_real_bound_scaled_past_the_threshold_is_refused() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Bounded {
        x_l: [-1.0e14, LO],
        x_u: [1.0, UP],
        factors: [1.0e6, 1.0],
    }));
    let err = match wrap_with_scaling(inner, LO, UP) {
        Err(e) => e,
        Ok(_) => panic!("a bound scaled to -1e20 must not be accepted"),
    };
    assert!(
        err.contains("nlp_lower_bound_inf"),
        "the refusal should name the threshold it crossed: {err}"
    );
}
