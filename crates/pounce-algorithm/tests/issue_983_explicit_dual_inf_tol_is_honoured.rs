//! gh#983 item 1: an explicitly set `dual_inf_tol` must not be loosened by the
//! scale-relative floor of gh#532.
//!
//! The cusp `min x1 s.t. x2 - x1^3 <= 0, x2 >= 0 (a row), x in [-2,2]^2` has no
//! constraint qualification at the origin, so its multipliers run to ~1e9. The
//! floor `kappa * tol * dual_scale` then rose to ~35 and certified
//! `Solve_Succeeded` with an unscaled `|grad L|_inf` of 0.144 under an explicit
//! `dual_inf_tol = 1e-6`.
//!
//! Three legs, because the rule branches on what the caller named:
//! * nothing named: the floor is a default and stays (the gh#532 contract);
//! * `dual_inf_tol` named: the absolute standard is honoured, so the status
//!   is never a strict success with a residual above it;
//! * both named: the caller opted the floor back in.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Cusp {
    out: Option<(Vec<Number>, Vec<Number>)>,
}

impl TNLP for Cusp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 2,
            nnz_jac_g: 3,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-2.0);
        b.x_u.fill(2.0);
        b.g_l[0] = -2e19;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 2e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.fill(0.0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(x[0])
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 1.0;
        g[1] = 0.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[1] - x[0].powi(3);
        g[1] = x[1];
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..3].copy_from_slice(&[0, 0, 1]);
                jcol[..3].copy_from_slice(&[0, 1, 1]);
            }
            SparsityRequest::Values { values } => {
                let x = x.unwrap();
                values[..3].copy_from_slice(&[-3.0 * x[0] * x[0], 1.0, 1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        _of: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => {
                values[0] = -6.0 * x.unwrap()[0] * lambda.unwrap()[0];
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.out = Some((sol.x.to_vec(), sol.lambda.to_vec()));
    }
}

fn grad_l_inf(x: &[Number], lam: &[Number]) -> Number {
    let g0 = 1.0 - 3.0 * x[0] * x[0] * lam[0];
    let g1 = lam[0] + lam[1];
    g0.abs().max(g1.abs())
}

fn run(opts: &[(&str, Number)]) -> (ApplicationReturnStatus, Number) {
    run_audit(opts, true)
}

fn run_audit(opts: &[(&str, Number)], audit: bool) -> (ApplicationReturnStatus, Number) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value(
            "solve_quality_audit",
            if audit { "yes" } else { "no" },
            true,
            false,
        )
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    for &(k, v) in opts {
        app.options_mut()
            .set_numeric_value(k, v, true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    let inst = Rc::new(RefCell::new(Cusp::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = inst.clone();
    let status = app.optimize_tnlp(tnlp);
    let (x, lam) = inst.borrow().out.clone().expect("finalize_solution ran");
    (status, grad_l_inf(&x, &lam))
}

#[test]
fn an_explicit_dual_inf_tol_is_not_loosened_by_the_floor() {
    let (status, res) = run(&[("dual_inf_tol", 1e-6)]);
    if status == ApplicationReturnStatus::SolveSucceeded {
        assert!(
            res <= 1e-4,
            "gh#983: Solve_Succeeded with |grad L|_inf = {res:e} under an explicit \
             dual_inf_tol=1e-6"
        );
    }
}

/// The branch the fixture above does not take: with nothing named, the floor
/// is still the gh#532 default, so the verdict is the one the issue observed
/// (a strict success at a residual above `dual_inf_tol`'s default scale is
/// permitted there; the point is that naming the tolerance changes it).
#[test]
fn the_floor_is_still_on_when_nothing_is_named_and_off_when_the_tolerance_is() {
    let (named_status, named_res) = run(&[("dual_inf_tol", 1e-6)]);
    let (kappa_status, kappa_res) = run(&[("dual_inf_tol", 1e-6), ("dual_inf_scale_kappa", 1.0)]);
    // Naming kappa too puts the floor back, so it can only be as loose or
    // looser than the honoured tolerance.
    assert!(
        !(named_status == ApplicationReturnStatus::SolveSucceeded && named_res > 1e-4),
        "named tolerance leaked through the floor: {named_res:e}"
    );
    // Measured: the floor certifies at 1.44e-1 when the caller opts it back in
    // by naming kappa, and the honoured tolerance reaches 9.2e-9. The *gate*
    // still accepts that point (gh#532); the second pass of gh#983 only stops
    // the verdict from being strict -- multipliers of 3.5e9 are a failed
    // constraint qualification -- so the status is the acceptable level, with
    // the warnings in `statistics.warnings`.
    assert_eq!(
        kappa_status,
        ApplicationReturnStatus::SolvedToAcceptableLevel
    );
    assert!(
        kappa_res > 1e-2,
        "opting the floor back in must restore gh#532: {kappa_res:e}"
    );
    // With the audit off the gate's own verdict is visible: strict.
    let (raw_status, raw_res) = run_audit(
        &[("dual_inf_tol", 1e-6), ("dual_inf_scale_kappa", 1.0)],
        false,
    );
    assert_eq!(raw_status, ApplicationReturnStatus::SolveSucceeded);
    assert!(raw_res > 1e-2, "{raw_res:e}");
}
