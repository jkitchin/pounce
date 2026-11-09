//! The multipliers a complementarity solve hands the caller's model are the
//! declared model's (gh#1147 Stage 2 follow-up).
//!
//! The driver solves a lowered model with one product row `p_k = G_k·H_k`
//! per pair (two for a mixed pair), and the caller's model has no such row.
//! Its multiplier `ν_k` belongs on the pair's two sides: `∇p_k = s_G H e_j +
//! s_H G ∇g_i`, so the declared row's multiplier is `λ_i + ν_k s_H G_k` and
//! the paired variable's bound multiplier takes `ν_k s_G H_k`. Dropping `ν_k`
//! (as the forwarding did) left `mult_g` at the lowered model's split, which
//! at a non-degenerate pair (`G > 0`, `H = 0`) is not unique and was reported
//! at 1e7 where the model's multiplier is 0.125, in every frontend (C API,
//! Python, CLI) — they all receive the duals through the caller's
//! `finalize_solution`.
//!
//! What is held, on every path the driver can report from (both
//! formulations, the lifted and unlifted rows, a mixed pair, both Hessian
//! legs): the declared model's dual residual `∇f + Jᵀλ - z_L + z_U` at the
//! forwarded point, from the forwarded multipliers, and the bound
//! multipliers' signs. At a non-degenerate point, where the multipliers are
//! unique, their values too.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory_with_sink, feral_config_from_options,
};
use pounce_common::types::{Index, Number};
use pounce_nlp::mpcc::ComplementarityPair;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::mpcc_driver::solve_mpcc;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 1e19;

/// What `finalize_solution` received.
#[derive(Debug, Clone, Default)]
struct Forwarded {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

/// A small dense model: linear rows `g = A x + a`, objective
/// `Σ (x_j - t_j)^2`.
struct Model {
    x_l: Vec<Number>,
    x_u: Vec<Number>,
    g_l: Vec<Number>,
    g_u: Vec<Number>,
    x0: Vec<Number>,
    target: Vec<Number>,
    a: Vec<Vec<Number>>,
    a0: Vec<Number>,
    /// `+1`: `f = Σ (x - t)^2`; `-1`: `f = -Σ (x - t)^2`, to be solved with
    /// `obj_scaling_factor = -1` (maximize).
    sign: Number,
    out: Rc<RefCell<Vec<Forwarded>>>,
}

impl Model {
    fn n(&self) -> usize {
        self.x_l.len()
    }
    fn m(&self) -> usize {
        self.g_l.len()
    }
    fn grad_f(&self, x: &[Number]) -> Vec<Number> {
        x.iter()
            .zip(&self.target)
            .map(|(x, t)| self.sign * 2.0 * (x - t))
            .collect()
    }
}

impl TNLP for Model {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: self.n() as Index,
            m: self.m() as Index,
            nnz_jac_g: (self.n() * self.m()) as Index,
            nnz_h_lag: self.n() as Index,
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
        Some(
            self.sign
                * x.iter()
                    .zip(&self.target)
                    .map(|(x, t)| (x - t).powi(2))
                    .sum::<Number>(),
        )
    }
    fn eval_grad_f(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        g.copy_from_slice(&self.grad_f(x));
        true
    }
    fn eval_g(&mut self, x: &[Number], _: bool, g: &mut [Number]) -> bool {
        for (i, gi) in g.iter_mut().enumerate() {
            *gi = self.a0[i] + self.a[i].iter().zip(x).map(|(a, x)| a * x).sum::<Number>();
        }
        true
    }
    fn eval_jac_g(&mut self, _: Option<&[Number]>, _: bool, mode: SparsityRequest<'_>) -> bool {
        let (n, m) = (self.n(), self.m());
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for r in 0..m {
                    for c in 0..n {
                        irow[r * n + c] = r as Index;
                        jcol[r * n + c] = c as Index;
                    }
                }
            }
            SparsityRequest::Values { values } => {
                for r in 0..m {
                    for c in 0..n {
                        values[r * n + c] = self.a[r][c];
                    }
                }
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        _: Option<&[Number]>,
        _: bool,
        s: Number,
        _: Option<&[Number]>,
        _: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for j in 0..self.n() {
                    irow[j] = j as Index;
                    jcol[j] = j as Index;
                }
            }
            SparsityRequest::Values { values } => values.fill(self.sign * 2.0 * s),
        }
        true
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _: &IpoptData, _: &IpoptCq) {
        self.out.borrow_mut().push(Forwarded {
            x: sol.x.to_vec(),
            z_l: sol.z_l.to_vec(),
            z_u: sol.z_u.to_vec(),
            lambda: sol.lambda.to_vec(),
        });
    }
}

fn app(options: &[(&str, &str)]) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    for (k, v) in options {
        app.options_mut()
            .read_from_str(&format!("{k} {v}\n"), true)
            .unwrap();
    }
    app.initialize().unwrap();
    let feral_cfg = feral_config_from_options(app.options());
    let sink = app.restoration_summary_sink();
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        let sink = std::sync::Arc::clone(&sink);
        Box::new(move || {
            default_backend_factory_with_sink(
                feral_cfg.clone(),
                Ma57Config::default(),
                std::sync::Arc::clone(&sink),
            )
        })
    };
    let provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(provider);
    app
}

/// QpecSmall: `min (x-1)^2 + (y1-1)^2 + y2^2`, `x` in `[xl, xu]`,
/// `0 <= y1 ⟂ 2 y1 - 1 - x >= 0`, `0 <= y2 ⟂ x + 2 y2 - 1 >= 0`.
fn qpec(xl: Number, xu: Number, out: &Rc<RefCell<Vec<Forwarded>>>) -> Model {
    Model {
        x_l: vec![xl, 0.0, 0.0],
        x_u: vec![xu, INF, INF],
        g_l: vec![0.0, 0.0],
        g_u: vec![INF, INF],
        x0: vec![0.5 * (xl + xu), 0.5, 0.5],
        target: vec![1.0, 1.0, 0.0],
        a: vec![vec![-1.0, 2.0, 0.0], vec![1.0, 0.0, 2.0]],
        a0: vec![-1.0, -1.0],
        sign: 1.0,
        out: Rc::clone(out),
    }
}

const QPEC_PAIRS: [ComplementarityPair; 2] = [
    ComplementarityPair { row: 0, var: 1 },
    ComplementarityPair { row: 1, var: 2 },
];

/// A mixed pair: `0 <= x <= 1 ⟂ y - x` (row free), `min (x - cx)^2 + (y -
/// cy)^2`. With `(cx, cy) = (0.5, 2)` the answer is `(0, 2)`, `x` at its
/// lower bound with `g = 2 > 0`.
fn mixed(cx: Number, cy: Number, out: &Rc<RefCell<Vec<Forwarded>>>) -> Model {
    Model {
        x_l: vec![0.0, -INF],
        x_u: vec![1.0, INF],
        g_l: vec![-INF],
        g_u: vec![INF],
        x0: vec![0.5, 0.5],
        target: vec![cx, cy],
        a: vec![vec![-1.0, 1.0]],
        a0: vec![0.0],
        sign: 1.0,
        out: Rc::clone(out),
    }
}

/// Solve and return the one forwarded solution, with the model's gradient
/// and Jacobian for the residual.
fn solve(
    model: Model,
    pairs: &[ComplementarityPair],
    options: &[(&str, &str)],
) -> (Forwarded, Vec<Number>, Vec<Vec<Number>>) {
    let out = Rc::clone(&model.out);
    let a = model.a.clone();
    let target = model.target.clone();
    let sign = model.sign;
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let mut app = app(options);
    solve_mpcc(&mut app, tnlp, pairs.to_vec(), &mut |_| {}).unwrap();
    let fw = out.borrow();
    assert_eq!(fw.len(), 1, "one forwarded solution");
    let f = fw[0].clone();
    let g: Vec<Number> =
        f.x.iter()
            .zip(&target)
            .map(|(x, t)| sign * 2.0 * (x - t))
            .collect();
    if std::env::var("DUALS_PRINT").is_ok() {
        eprintln!(
            "{options:?}\n   res {:.2e} lam {:?} zl {:?} zu {:?}",
            dual_residual(&f, &g, &a),
            f.lambda,
            f.z_l,
            f.z_u
        );
    }
    (f, g, a)
}

/// `‖∇f + Aᵀλ - z_L + z_U‖∞`.
fn dual_residual(f: &Forwarded, grad: &[Number], a: &[Vec<Number>]) -> Number {
    (0..f.x.len())
        .map(|j| {
            let r = grad[j]
                + a.iter()
                    .zip(&f.lambda)
                    .map(|(row, l)| row[j] * l)
                    .sum::<Number>()
                - f.z_l[j]
                + f.z_u[j];
            r.abs()
        })
        .fold(0.0, Number::max)
}

/// Every option set the driver's reported point can come from.
const PATHS: [&[(&str, &str)]; 6] = [
    &[],
    &[("complementarity_strategy", "exact")],
    &[("complementarity_strategy", "continuation")],
    &[("complementarity_lift", "yes")],
    &[
        ("complementarity_lift", "yes"),
        ("complementarity_strategy", "exact"),
    ],
    &[("hessian_approximation", "limited-memory")],
];

#[test]
fn qpec_with_a_non_degenerate_answer_reports_the_models_multipliers() {
    // x in [0.25, 0.75]: the answer is (3/4, 7/8, 1/8), both rows active
    // and both y > 0, so the multipliers are unique. In Ipopt's convention
    // (∇f + Jᵀλ - z_L + z_U = 0): from y1, 2(y1-1) + 2λ0 = 0, λ0 = 1/8; from
    // y2, 2 y2 + 2λ1 = 0, λ1 = -1/8; from x, 2(x-1) - λ0 + λ1 + z_U = 0,
    // z_U[x] = 3/4.
    for opts in PATHS {
        let out = Rc::new(RefCell::new(Vec::new()));
        let (f, grad, a) = solve(qpec(0.25, 0.75, &out), &QPEC_PAIRS, opts);
        let ctx = format!("{opts:?}: {f:?}");
        for (x, w) in f.x.iter().zip([0.75, 0.875, 0.125]) {
            assert!((x - w).abs() < 1e-6, "x; {ctx}");
        }
        assert!(dual_residual(&f, &grad, &a) < 1e-6, "residual; {ctx}");
        assert!((f.lambda[0] - 0.125).abs() < 1e-5, "λ0; {ctx}");
        assert!((f.lambda[1] + 0.125).abs() < 1e-5, "λ1; {ctx}");
        assert!((f.z_u[0] - 0.75).abs() < 1e-5, "z_U[x]; {ctx}");
        // y > 0 and no upper bounds: every other bound multiplier is zero.
        for (j, z) in f.z_l.iter().enumerate() {
            assert!(z.abs() < 1e-6, "z_L[{j}]; {ctx}");
        }
        assert!(
            f.z_u[1].abs() < 1e-8 && f.z_u[2].abs() < 1e-8,
            "z_U[y]; {ctx}"
        );
    }
}

#[test]
fn qpec_at_its_biactive_answer_reports_multipliers_that_make_it_stationary() {
    // (1, 1, 0): the second pair biactive, so the split between λ1 and
    // z_L[y2] is not unique; the residual is what is held.
    for opts in PATHS {
        let out = Rc::new(RefCell::new(Vec::new()));
        let (f, grad, a) = solve(qpec(0.0, 2.0, &out), &QPEC_PAIRS, opts);
        let ctx = format!("{opts:?}: {f:?}");
        for (x, w) in f.x.iter().zip([1.0, 1.0, 0.0]) {
            assert!((x - w).abs() < 1e-5, "x; {ctx}");
        }
        assert!(dual_residual(&f, &grad, &a) < 1e-6, "residual; {ctx}");
        // Every multiplier is zero at (1, 1, 0) (from y1, λ0 = 1 - y1; then
        // λ1 and z_L[y2] from x and y2). The fit lands within the point's
        // own error of zero (x = 1 + 4e-7 on the lifted path), either side;
        // the certificate's sign band is sqrt(tol) = 1e-4.
        for v in f.lambda.iter().chain(&f.z_l).chain(&f.z_u) {
            assert!(v.abs() < 1e-5, "multipliers; {ctx}");
        }
    }
}

#[test]
fn a_mixed_pair_reports_the_models_multipliers() {
    // (0, 2): x at its lower bound, g = y - x = 2 > 0 (non-degenerate), so
    // the row is inactive and λ = 0 (y's 2(y - 2) + λ = 0). x's
    // stationarity, 2(x - 0.5) - λ - z_L = 0, then gives z_L[x] = -1.
    let pair = [ComplementarityPair { row: 0, var: 0 }];
    let mut pinned = 0;
    for opts in PATHS {
        let out = Rc::new(RefCell::new(Vec::new()));
        let (f, grad, a) = solve(mixed(0.5, 2.0, &out), &pair, opts);
        let ctx = format!("{opts:?}: {f:?}");
        assert!(dual_residual(&f, &grad, &a) < 1e-6, "residual; {ctx}");
        // The exact product alone can stop at (1, 1), a local minimum the
        // driver tests document; only the residual is held there.
        if (f.x[0]).abs() > 1e-6 || (f.x[1] - 2.0).abs() > 1e-6 {
            continue;
        }
        pinned += 1;
        // x is held at 0 by the pair (g = 2 > 0), not by its bound: the
        // multiplier on the pair's bound is sign-free, and here negative,
        // 2(0 - 0.5) - z_L = 0. Splitting it onto z_U (x's other bound, 1
        // away) would break that bound's complementarity.
        assert!(f.lambda[0].abs() < 1e-5, "λ; {ctx}");
        assert!((f.z_l[0] + 1.0).abs() < 1e-5, "z_L[x]; {ctx}");
        assert!(f.z_u[0].abs() < 1e-8, "z_U[x]; {ctx}");
    }
    assert!(
        pinned >= 4,
        "the answer (0, 2) was reached on {pinned} paths only"
    );
}

/// The plain path's convention under `obj_scaling_factor = -1`, measured on
/// the same model without pairs (no driver): the residual the reported
/// multipliers satisfy, against `f` itself.
fn plain_maximize_residual() -> Number {
    use pounce_nlp::return_codes::ApplicationReturnStatus;
    let out = Rc::new(RefCell::new(Vec::new()));
    let mut model = qpec(0.25, 0.75, &out);
    model.sign = -1.0;
    let (a, target) = (model.a.clone(), model.target.clone());
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let mut app = app(&[("obj_scaling_factor", "-1")]);
    let st = app.optimize_tnlp(tnlp);
    assert_eq!(st, ApplicationReturnStatus::SolveSucceeded);
    let f = out.borrow()[0].clone();
    let g: Vec<Number> =
        f.x.iter()
            .zip(&target)
            .map(|(x, t)| -2.0 * (x - t))
            .collect();
    dual_residual(&f, &g, &a)
}

#[test]
fn a_maximized_model_reports_multipliers_in_the_plain_paths_convention() {
    // Upstream un-scales the duals by the objective scaling, so a plain
    // maximize solve reports multipliers for f itself: ∇f + Jᵀλ - z_L + z_U
    // = 0, active bound multipliers <= 0. The driver must do the same;
    // the certificate fits σ·f.
    let plain = plain_maximize_residual();
    assert!(plain < 1e-6, "plain path convention: residual {plain}");
    for opts in PATHS {
        let mut o = opts.to_vec();
        o.push(("obj_scaling_factor", "-1"));
        let out = Rc::new(RefCell::new(Vec::new()));
        let mut model = qpec(0.25, 0.75, &out);
        model.sign = -1.0;
        let (f, grad, a) = solve(model, &QPEC_PAIRS, &o);
        let ctx = format!("{o:?}: {f:?}");
        for (x, w) in f.x.iter().zip([0.75, 0.875, 0.125]) {
            assert!((x - w).abs() < 1e-6, "x; {ctx}");
        }
        assert!(dual_residual(&f, &grad, &a) < 1e-6, "residual; {ctx}");
        assert!((f.lambda[0] + 0.125).abs() < 1e-5, "λ0; {ctx}");
        assert!((f.lambda[1] - 0.125).abs() < 1e-5, "λ1; {ctx}");
        assert!((f.z_u[0] + 0.75).abs() < 1e-5, "z_U[x]; {ctx}");
    }
}
