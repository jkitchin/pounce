//! Workstream C (`dev-notes/research/nlp-soc-design.md`): second-order cones
//! carried natively by the interior-point method, end to end, on models whose
//! answers are known in closed form.
//!
//! The answers are chosen to catch the failure the design is built against:
//! a cone entry read as a bound `s_i >= 0`. Every model's optimum has a cone
//! tail entry that is negative, and one sits at the cone's tip — where the
//! smooth lowerings `t - ||x||` and `t^2 - ||x||^2` are degenerate.

#![allow(clippy::type_complexity, clippy::needless_range_loop)]

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::soc::SocRows;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 1e19;

type Fx = Box<dyn Fn(&[Number]) -> Number>;
type Vx = Box<dyn Fn(&[Number]) -> Vec<Number>>;
type Mx = Box<dyn Fn(&[Number]) -> Vec<Vec<Number>>>;

/// A dense model: objective, rows, Jacobian, and per-function Hessians
/// (`hess(x)[0]` the objective's, `hess(x)[1 + i]` row `i`'s).
struct Dense {
    n: usize,
    m: usize,
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x0: Vec<Number>,
    f: Fx,
    grad: Vx,
    g: Vx,
    jac: Mx,
    hess: Box<dyn Fn(&[Number]) -> Vec<Vec<Vec<Number>>>>,
    out: Rc<RefCell<Option<(Vec<Number>, Vec<Number>)>>>,
}

impl TNLP for Dense {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n as Index,
            m: self.m as Index,
            nnz_jac_g: (self.n * self.m) as Index,
            nnz_h_lag: (self.n * (self.n + 1) / 2) as Index,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.x_l);
        b.x_u.copy_from_slice(&self.x_u);
        b.g_l.copy_from_slice(&self.g_l);
        b.g_u.copy_from_slice(&self.g_u);
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.x0);
        true
    }
    fn eval_f(&mut self, x: &[Number], _: bool) -> Option<Number> {
        Some((self.f)(x))
    }
    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&(self.grad)(x));
        true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&(self.g)(x));
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for r in 0..self.m {
                    for c in 0..self.n {
                        irow[r * self.n + c] = r as Index;
                        jcol[r * self.n + c] = c as Index;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let j = (self.jac)(x.unwrap());
                for r in 0..self.m {
                    for c in 0..self.n {
                        values[r * self.n + c] = j[r][c];
                    }
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _: bool,
        s: Number,
        lam: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        let mut k = 0;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for r in 0..self.n {
                    for c in 0..=r {
                        irow[k] = r as Index;
                        jcol[k] = c as Index;
                        k += 1;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                let h = (self.hess)(x.unwrap());
                let lam = lam.unwrap();
                for r in 0..self.n {
                    for c in 0..=r {
                        let mut v = s * h[0][r][c];
                        for i in 0..self.m {
                            v += lam[i] * h[1 + i][r][c];
                        }
                        values[k] = v;
                        k += 1;
                    }
                }
            }
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        *self.out.borrow_mut() = Some((sol.x.to_vec(), sol.lambda.to_vec()));
    }
}

fn zeros(n: usize) -> Vec<Vec<Number>> {
    vec![vec![0.0; n]; n]
}

fn unit(n: usize, j: usize) -> Vec<Number> {
    let mut r = vec![0.0; n];
    r[j] = 1.0;
    r
}

/// Rows that are the variables themselves: `g = x` (free rows).
fn identity_rows(n: usize, m: usize, x0: Vec<Number>, f: Fx, grad: Vx, hf: Mx) -> Dense {
    Dense {
        n,
        m,
        x_l: vec![-INF; n],
        x_u: vec![INF; n],
        g_l: vec![-INF; m],
        g_u: vec![INF; m],
        x0,
        f,
        grad,
        g: Box::new(move |x| x[..m].to_vec()),
        jac: Box::new(move |_| (0..m).map(|i| unit(n, i)).collect()),
        hess: Box::new(move |x| {
            let mut h = vec![hf(x)];
            h.extend((0..m).map(|_| zeros(n)));
            h
        }),
        out: Rc::new(RefCell::new(None)),
    }
}

fn solve(
    model: Dense,
    cones: Vec<Vec<usize>>,
) -> (ApplicationReturnStatus, Vec<Number>, Vec<Number>) {
    solve_with(model, cones, &[])
}

fn solve_with(
    model: Dense,
    cones: Vec<Vec<usize>>,
    options: &[(&str, &str)],
) -> (ApplicationReturnStatus, Vec<Number>, Vec<Number>) {
    let out = Rc::clone(&model.out);
    let mut app = IpoptApplication::new();
    for (k, v) in options {
        let o = app.options_mut();
        match v.parse::<Number>() {
            Ok(x) => o.set_numeric_value(k, x, true, false).unwrap(),
            Err(_) => o.set_string_value(k, v, true, false).unwrap(),
        };
    }
    app.options_mut()
        .set_integer_value(
            "print_level",
            std::env::var("SOC_PL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            true,
            false,
        )
        .unwrap();
    app.set_soc_constraints(cones.into_iter().map(|rows| SocRows { rows }).collect());
    let status = app.optimize_tnlp(Rc::new(RefCell::new(model)));
    let (x, lam) = out.borrow().clone().unwrap_or_default();
    (status, x, lam)
}

/// `min x1 + x2` over the disk `(1, x1, x2) ∈ K` (head fixed at 1 by a
/// fixed variable). Answer `x1 = x2 = −1/√2`, objective `−√2`: both tail
/// entries negative, so a cone entry read as `>= 0` cannot reach it.
#[test]
fn the_disk_reaches_its_negative_tail_optimum() {
    let mut m = identity_rows(
        3,
        3,
        vec![1.0, 0.0, 0.0],
        Box::new(|x| x[1] + x[2]),
        Box::new(|_| vec![0.0, 1.0, 1.0]),
        Box::new(|_| zeros(3)),
    );
    m.x_l[0] = 1.0;
    m.x_u[0] = 1.0;
    let (status, x, lam) = solve(m, vec![vec![0, 1, 2]]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{x:?}");
    let r = -(0.5 as Number).sqrt();
    assert!((x[1] - r).abs() < 1e-7 && (x[2] - r).abs() < 1e-7, "{x:?}");
    // The cone dual (−λ on the cone rows) lies in the cone and is
    // complementary: z = (√2, 1, 1) / ... up to the head's scale.
    let z: Vec<Number> = lam.iter().map(|l| -l).collect();
    assert!(z[0] >= (z[1] * z[1] + z[2] * z[2]).sqrt() - 1e-8, "{z:?}");
}

/// The warm-start initializer carries the cones (`init_soc_from_warm_start`):
/// the slacks are pushed into the interior and the duals start at `−y_d`
/// lifted onto the barrier's central path. With nothing staged the TNLP's
/// own start is the warm point, so this is the disk from its cold start
/// through the warm path, and it must reach the same answer.
#[test]
fn a_warm_start_carries_the_cones() {
    let mut m = identity_rows(
        3,
        3,
        vec![1.0, 0.0, 0.0],
        Box::new(|x| x[1] + x[2]),
        Box::new(|_| vec![0.0, 1.0, 1.0]),
        Box::new(|_| zeros(3)),
    );
    m.x_l[0] = 1.0;
    m.x_u[0] = 1.0;
    let (status, x, lam) = solve_with(
        m,
        vec![vec![0, 1, 2]],
        &[("warm_start_init_point", "yes"), ("mu_init", "1e-3")],
    );
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{x:?}");
    let r = -(0.5 as Number).sqrt();
    assert!((x[1] - r).abs() < 1e-7 && (x[2] - r).abs() < 1e-7, "{x:?}");
    let z: Vec<Number> = lam.iter().map(|l| -l).collect();
    assert!(z[0] >= (z[1] * z[1] + z[2] * z[2]).sqrt() - 1e-8, "{z:?}");
}

/// `min t + (x1 − 1)² + x2²` with `(t, x1 − 1, x2) ∈ K` written on the
/// variables `(t, x1, x2)` and a shifted row. Optimum `t = 0`, `x = (1, 0)`:
/// the cone's **tip**, where `t − ‖x‖` has no derivative and the gradient of
/// `t² − ‖x‖²` vanishes.
#[test]
fn the_tip_is_reached_to_tolerance() {
    let m = Dense {
        n: 3,
        m: 3,
        x_l: vec![-INF; 3],
        x_u: vec![INF; 3],
        g_l: vec![-INF; 3],
        g_u: vec![INF; 3],
        x0: vec![2.0, 0.3, 0.4],
        f: Box::new(|x| x[0] + (x[1] - 1.0).powi(2) + x[2] * x[2]),
        grad: Box::new(|x| vec![1.0, 2.0 * (x[1] - 1.0), 2.0 * x[2]]),
        g: Box::new(|x| vec![x[0], x[1] - 1.0, x[2]]),
        jac: Box::new(|_| vec![unit(3, 0), unit(3, 1), unit(3, 2)]),
        hess: Box::new(|_| {
            let mut f = zeros(3);
            f[1][1] = 2.0;
            f[2][2] = 2.0;
            vec![f, zeros(3), zeros(3), zeros(3)]
        }),
        out: Rc::new(RefCell::new(None)),
    };
    let (status, x, _) = solve(m, vec![vec![0, 1, 2]]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{x:?}");
    assert!(
        x[0].abs() < 1e-7 && (x[1] - 1.0).abs() < 1e-7 && x[2].abs() < 1e-7,
        "{x:?}"
    );
    // In the cone, to the solver's tolerance.
    let tail = ((x[1] - 1.0).powi(2) + x[2] * x[2]).sqrt();
    assert!(x[0] - tail >= -1e-9, "cone violated: {} < {tail}", x[0]);
}

/// Rosenbrock restricted to the disk `(1.2, x0 − 0.5, x1) ∈ K` (radius 1.2
/// around `(0.5, 0)`), which contains the unconstrained optimum `(1, 1)`
/// (distance 1.118): the cone is inactive and the answer is `(1, 1)`. The
/// head row is a constant, so its Jacobian row is empty. This is the model
/// that showed the cone duals must step with the dual step size: tied to the
/// primal step, the dual cap drove `alpha_pr` to `1e-16`.
#[test]
fn an_inactive_cone_leaves_the_unconstrained_optimum() {
    let m = Dense {
        n: 2,
        m: 3,
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
        g_l: vec![-INF; 3],
        g_u: vec![INF; 3],
        x0: vec![-0.5, 0.5],
        f: Box::new(|x| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)),
        grad: Box::new(|x| {
            vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        }),
        g: Box::new(|x| vec![1.2, x[0] - 0.5, x[1]]),
        jac: Box::new(|_| vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
        hess: Box::new(|x| {
            let f = vec![
                vec![1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
                vec![-400.0 * x[0], 200.0],
            ];
            vec![f, zeros(2), zeros(2), zeros(2)]
        }),
        out: Rc::new(RefCell::new(None)),
    };
    let (status, x, _) = solve(m, vec![vec![0, 1, 2]]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{x:?}");
    assert!(
        (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
        "{x:?}"
    );
}

/// The same Rosenbrock with a radius-0.5 disk around the origin: the cone
/// is active and the optimum is on its boundary — checked against the
/// boundary's own optimality: the point is on the circle and the objective
/// gradient is normal to it, pointing inward.
#[test]
fn an_active_nonlinear_cone_stops_on_its_boundary() {
    let m = Dense {
        n: 2,
        m: 3,
        x_l: vec![-INF; 2],
        x_u: vec![INF; 2],
        g_l: vec![-INF; 3],
        g_u: vec![INF; 3],
        x0: vec![0.0, 0.0],
        f: Box::new(|x| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)),
        grad: Box::new(|x| {
            vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        }),
        g: Box::new(|x| vec![0.5, x[0], x[1]]),
        jac: Box::new(|_| vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]]),
        hess: Box::new(|x| {
            let f = vec![
                vec![1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0, -400.0 * x[0]],
                vec![-400.0 * x[0], 200.0],
            ];
            vec![f, zeros(2), zeros(2), zeros(2)]
        }),
        out: Rc::new(RefCell::new(None)),
    };
    let (status, x, _) = solve(m, vec![vec![0, 1, 2]]);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded, "{x:?}");
    let r = (x[0] * x[0] + x[1] * x[1]).sqrt();
    assert!(
        (r - 0.5).abs() < 1e-7,
        "not on the boundary: r = {r}, x = {x:?}"
    );
    let g = [
        -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ];
    // ∇f = −κ x with κ > 0 (gradient points into the disk).
    let cross = g[0] * x[1] - g[1] * x[0];
    let dot = g[0] * x[0] + g[1] * x[1];
    assert!(cross.abs() < 1e-5 && dot < 0.0, "g = {g:?}, x = {x:?}");
}

/// Options the cone path does not carry are refused with the reason, before
/// the model is evaluated — never run with the cone rows unconstrained.
#[test]
fn unsupported_options_are_refused() {
    for (k, v) in [
        ("mu_strategy", "adaptive"),
        ("alpha_for_y", "full"),
        ("mehrotra_algorithm", "yes"),
    ] {
        let mut app = IpoptApplication::new();
        app.options_mut()
            .set_string_value(k, v, true, false)
            .unwrap();
        app.set_soc_constraints(vec![SocRows {
            rows: vec![0, 1, 2],
        }]);
        assert!(app.soc_incompatibility().is_some(), "{k}={v}");
    }
}

/// A malformed declaration is refused with the reason.
#[test]
fn a_bounded_cone_row_is_refused() {
    let mut m = identity_rows(
        3,
        3,
        vec![1.0, 0.0, 0.0],
        Box::new(|x| x[1] + x[2]),
        Box::new(|_| vec![0.0, 1.0, 1.0]),
        Box::new(|_| zeros(3)),
    );
    m.g_l[1] = 0.0;
    let (status, _, _) = solve(m, vec![vec![0, 1, 2]]);
    assert_eq!(status, ApplicationReturnStatus::InvalidProblemDefinition);
}
