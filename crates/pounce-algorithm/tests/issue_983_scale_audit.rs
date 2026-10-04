//! gh#983 items 2 and 3: a success verdict is judged against the model's own
//! scale, not only against a scaled aggregate.
//!
//! Gradient-based scaling measures the objective gradient once, at the
//! starting point, and only ever scales *down*. Two failures follow, one on
//! each side, and the audit (`solve_quality_audit`) is the remedy for both:
//!
//! * **frozen at a huge-gradient start** (`lj7_*`): the factor is tiny, the
//!   scaled test passes, the unscaled stationarity residual is nowhere near
//!   it. Remedy: re-solve from the returned point with the scaling
//!   re-evaluated there.
//! * **small objective** (`small_lp_*`): the gradient is `1e-6`, nothing
//!   scales it up, so the absolute complementarity floor is a visible
//!   fraction of the objective. Remedy: re-solve with the objective
//!   multiplied by `1/max|grad f|`.
//!
//! Every test names its control: the same model with `solve_quality_audit=no`
//! must show the defect, or the test is evidence about nothing. The two
//! models take **different branches** of `scale_audit_trigger`.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

// ---------------------------------------------------------------- LJ7 ----

const NAT: usize = 7;

fn lj_start() -> Vec<Number> {
    // Atoms 0 and 1 a quarter apart: |grad| ~ 3e9, the "gradient 3e8 start".
    let mut x = vec![
        0.0, 0.0, 0.0, //
        0.25, 0.0, 0.0, //
        -0.8, 0.7, 0.2, //
        0.6, -0.9, 0.4, //
        -0.3, -0.6, -0.9, //
        0.9, 0.8, -0.7, //
        -0.9, -0.2, 0.9,
    ];
    x.truncate(3 * NAT);
    x
}

struct Lj;

impl Lj {
    fn grad(x: &[Number], g: &mut [Number]) {
        g.fill(0.0);
        for i in 0..NAT {
            for j in 0..NAT {
                if i == j {
                    continue;
                }
                let mut d = [0.0; 3];
                let mut r2 = 0.0;
                for k in 0..3 {
                    d[k] = x[3 * i + k] - x[3 * j + k];
                    r2 += d[k] * d[k];
                }
                let ir6 = 1.0 / (r2 * r2 * r2);
                let c = 4.0 * (-12.0 * ir6 * ir6 + 6.0 * ir6) / r2;
                for k in 0..3 {
                    g[3 * i + k] += c * d[k];
                }
            }
        }
    }
}

impl TNLP for Lj {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3 * NAT as i32,
            m: 0,
            nnz_jac_g: 0,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(-1.6);
        b.x_u.fill(1.6);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&lj_start());
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let mut e = 0.0;
        for i in 0..NAT {
            for j in (i + 1)..NAT {
                let mut r2 = 0.0;
                for k in 0..3 {
                    let d = x[3 * i + k] - x[3 * j + k];
                    r2 += d * d;
                }
                let ir6 = 1.0 / (r2 * r2 * r2);
                e += 4.0 * (ir6 * ir6 - ir6);
            }
        }
        Some(e)
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        Lj::grad(x, g);
        true
    }
    fn eval_g(&mut self, _x: &[Number], _n: bool, _g: &mut [Number]) -> bool {
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, _m: SparsityRequest<'_>) -> bool {
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        _m: SparsityRequest<'_>,
    ) -> bool {
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve(
    tnlp: Rc<RefCell<dyn TNLP>>,
    opts: &[(&str, &str)],
) -> (ApplicationReturnStatus, SolveStatistics) {
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
    let status = app.optimize_tnlp(tnlp);
    (status, app.statistics())
}

fn lj_opts<'a>(audit: &'a str) -> Vec<(&'a str, &'a str)> {
    vec![
        ("hessian_approximation", "limited-memory"),
        ("solve_quality_audit", audit),
    ]
}

#[test]
fn lj7_start_is_a_huge_gradient_start() {
    let mut g = vec![0.0; 3 * NAT];
    Lj::grad(&lj_start(), &mut g);
    let m = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(m > 1e7, "the premise: gradient at the start is {m:e}");
}

/// The control. If this stops showing a non-stationary success the test below
/// proves nothing; it names what it observed so a platform that happens to
/// converge is visible instead of silently green.
#[test]
fn lj7_control_without_the_audit_may_certify_a_non_stationary_point() {
    let (status, st) = solve(Rc::new(RefCell::new(Lj)), &lj_opts("no"));
    eprintln!(
        "control: {status:?} unscaled dual {:e}, obj scale {:e}",
        st.final_unscaled_dual_inf, st.final_obj_scaling_factor
    );
    assert!(st.final_obj_scaling_factor < 1e-4, "scaling froze small");
    assert!(
        st.final_unscaled_dual_inf > 1e-6,
        "the control no longer shows a non-stationary success: {:e}",
        st.final_unscaled_dual_inf
    );
}

#[test]
fn lj7_audit_leaves_a_stationary_point_in_the_model_s_own_units() {
    let (status, st) = solve(Rc::new(RefCell::new(Lj)), &lj_opts("yes"));
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{status:?}"
    );
    eprintln!("audited: unscaled dual {:e}", st.final_unscaled_dual_inf);
    assert!(
        st.final_unscaled_dual_inf <= 1e-7,
        "unscaled dual infeasibility {:e} (scaled KKT {:e}); the audit must \
         re-scale at the returned point",
        st.final_unscaled_dual_inf,
        st.final_kkt_error
    );
    // The re-solve is a re-scaled run: its factor is not the frozen one.
    assert!(
        st.final_obj_scaling_factor > 1e-4,
        "reported run still carries the frozen factor {:e}",
        st.final_obj_scaling_factor
    );
}

// ------------------------------------------------------------ small LP ----

/// `min fs*(x0 + 2 x1)  s.t. x0 + x1 >= 1, 0 <= x <= 10`. Optimum
/// `x = (1, 0)`, `f* = fs`; `x1` sits on its bound with multiplier `fs`.
struct SmallLp {
    fs: Number,
}

impl TNLP for SmallLp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 0,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.fill(0.0);
        b.x_u.fill(10.0);
        b.g_l[0] = 1.0;
        b.g_u[0] = 2e19;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 1.0]);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(self.fs * (x[0] + 2.0 * x[1]))
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = self.fs;
        g[1] = 2.0 * self.fs;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[..2].copy_from_slice(&[0, 0]);
                jcol[..2].copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => values[..2].copy_from_slice(&[1.0, 1.0]),
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _of: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        _m: SparsityRequest<'_>,
    ) -> bool {
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn rel_err(st: &SolveStatistics, fs: Number) -> Number {
    (st.final_objective - fs).abs() / fs
}

#[test]
fn small_lp_control_loses_digits_to_the_absolute_floor() {
    let fs = 1e-6;
    let (_, st) = solve(
        Rc::new(RefCell::new(SmallLp { fs })),
        &[("solve_quality_audit", "no")],
    );
    eprintln!(
        "control: rel err {:e}, compl {:e}",
        rel_err(&st, fs),
        st.final_unscaled_compl
    );
    assert!(
        rel_err(&st, fs) > 1e-6,
        "the control no longer shows the defect: {:e}",
        rel_err(&st, fs)
    );
}

#[test]
fn small_lp_audit_recovers_the_digits() {
    let fs = 1e-6;
    let (status, st) = solve(Rc::new(RefCell::new(SmallLp { fs })), &[]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(
        rel_err(&st, fs) < 1e-6,
        "relative objective error {:e}",
        rel_err(&st, fs)
    );
}

/// The other side of the trigger: a well-scaled objective is never
/// re-solved, so the iteration count equals the audit-off run's.
#[test]
fn a_well_scaled_model_is_not_audited() {
    let (_, on) = solve(Rc::new(RefCell::new(SmallLp { fs: 1.0 })), &[]);
    let (_, off) = solve(
        Rc::new(RefCell::new(SmallLp { fs: 1.0 })),
        &[("solve_quality_audit", "no")],
    );
    assert_eq!(on.iteration_count, off.iteration_count);
    assert_eq!(on.num_obj_evals, off.num_obj_evals);
    assert!(on.warnings.is_empty(), "{:?}", on.warnings);
}
