//! End-to-end tests of `pounce_restoration::mpcc_driver::solve_mpcc`: real
//! solves through `IpoptApplication`, on models whose answers are known in
//! closed form.

// Test-only shapes: a dense model spelled out loop by loop reads closer to
// the math than iterator chains, and the boxed callbacks are the model.
#![allow(clippy::type_complexity, clippy::needless_range_loop)]

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory_with_sink, feral_config_from_options,
};
use pounce_algorithm::mpcc::{
    EXACT_STAGE_TAU, MpccStage, Reported, StationarityClass, TOUCHED_OPTIONS,
};
use pounce_common::types::{Index, Number};
use pounce_nlp::mpcc::ComplementarityPair;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, IterStats, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use pounce_restoration::mpcc_driver::{MpccOutcome, solve_mpcc};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::rc::Rc;

const INF: Number = 1e19;

type Fx = Box<dyn Fn(&[Number]) -> Number>;
type Vx = Box<dyn Fn(&[Number]) -> Vec<Number>>;
type Mx = Box<dyn Fn(&[Number]) -> Vec<Vec<Number>>>;

/// A small dense-derivative model: objective, constraints, their Jacobian,
/// and the per-function Hessians (objective first, then one per row), so the
/// Lagrangian Hessian is assembled exactly.
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
    /// `hess(x)[0]` is ∇²f, `hess(x)[1 + i]` is ∇²g_i, each n×n.
    hess: Box<dyn Fn(&[Number]) -> Vec<Vec<Vec<Number>>>>,
    finalized: Rc<RefCell<Vec<Vec<Number>>>>,
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
        self.finalized.borrow_mut().push(sol.x.to_vec());
    }
}

fn app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
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

/// `benchmarks/mpcc`'s `qpec_small`:
///
/// ```text
/// min (x-1)^2 + (y1-1)^2 + y2^2,   0 <= x <= 2
///     0 <= y1 ⟂ 2 y1 - 1 - x >= 0
///     0 <= y2 ⟂ 2 y2 - 1 + x >= 0
/// ```
///
/// The answer is `(1, 1, 0)`, with the second pair **biactive** there — the
/// case the exact product is recorded failing on from some starts through
/// `benchmarks/mpcc`'s Python path. Through this path the exact-product stage
/// alone reaches it from all three starts (29, 38 and 29 iterations when
/// written), so this test is about the answer and the single finalize, not
/// about which stage produced it; the fallback is the next test's job.
fn qpec_small(x0: [Number; 3], finalized: Rc<RefCell<Vec<Vec<Number>>>>) -> Dense {
    let zero = || vec![vec![0.0; 3]; 3];
    Dense {
        n: 3,
        m: 2,
        x_l: vec![0.0, 0.0, 0.0],
        x_u: vec![2.0, INF, INF],
        g_l: vec![0.0, 0.0],
        g_u: vec![INF, INF],
        x0: x0.to_vec(),
        f: Box::new(|x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2) + x[2] * x[2]),
        grad: Box::new(|x| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0), 2.0 * x[2]]),
        g: Box::new(|x| vec![2.0 * x[1] - 1.0 - x[0], 2.0 * x[2] - 1.0 + x[0]]),
        jac: Box::new(|_| vec![vec![-1.0, 2.0, 0.0], vec![1.0, 0.0, 2.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            for i in 0..3 {
                f[i][i] = 2.0;
            }
            vec![f, zero(), zero()]
        }),
        finalized,
    }
}

fn qpec_pairs() -> Vec<ComplementarityPair> {
    vec![
        ComplementarityPair { row: 0, var: 1 },
        ComplementarityPair { row: 1, var: 2 },
    ]
}

#[test]
fn qpec_small_reaches_the_answer_from_every_gate0_start() {
    for x0 in [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0], [2.0, 1.5, 0.0]] {
        let finalized = Rc::new(RefCell::new(Vec::new()));
        let mut app = app();
        let tnlp: Rc<RefCell<dyn TNLP>> =
            Rc::new(RefCell::new(qpec_small(x0, Rc::clone(&finalized))));
        let mut lines = Vec::new();
        let out = solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |l| {
            lines.push(l.to_string())
        })
        .unwrap();
        assert!(
            out.accepted_by.is_some(),
            "start {x0:?}: no stage accepted: {:?}",
            out.solves
        );
        assert!(
            out.max_abs_product <= 1e-4,
            "start {x0:?}: {}",
            out.max_abs_product
        );
        for (got, want) in out.x.iter().zip([1.0, 1.0, 0.0]) {
            assert!((got - want).abs() < 1e-4, "start {x0:?}: x = {:?}", out.x);
        }
        // Strongly stationary, with the second pair biactive.
        let cert = out.certificate.as_ref().unwrap();
        assert_eq!(
            (cert.class, cert.n_biactive),
            (StationarityClass::S, 1),
            "start {x0:?}: {cert:?}"
        );
        // The model saw exactly one solution, and it is the reported one.
        let seen = finalized.borrow();
        assert_eq!(seen.len(), 1, "start {x0:?}: {} finalizes", seen.len());
        assert_eq!(seen[0], out.x);
    }
}

/// Two variables, one equality and two pairs: the exact product would make
/// three equalities in two unknowns, which the solver refuses before
/// iterating (`Not_Enough_Degrees_Of_Freedom`, as upstream Ipopt does). The
/// continuation's products are inequalities, and its finish falls back to
/// `G·H <= 0` for the same reason. The only feasible point is (0.5, 0.5).
///
/// ```text
/// min (x-0.3)^2 + (y-0.9)^2   s.t.  x + y = 1
///     0 <= x ⟂ y - 0.5 >= 0,   0 <= y ⟂ x - 0.5 >= 0
/// ```
#[test]
fn an_over_determined_model_is_answered_through_the_inequality_finish() {
    let finalized = Rc::new(RefCell::new(Vec::new()));
    let zero = || vec![vec![0.0; 2]; 2];
    let model = Dense {
        n: 2,
        m: 3,
        x_l: vec![0.0, 0.0],
        x_u: vec![INF, INF],
        g_l: vec![1.0, 0.5, 0.5],
        g_u: vec![1.0, INF, INF],
        x0: vec![0.2, 0.2],
        f: Box::new(|x| (x[0] - 0.3).powi(2) + (x[1] - 0.9).powi(2)),
        grad: Box::new(|x| vec![2.0 * (x[0] - 0.3), 2.0 * (x[1] - 0.9)]),
        g: Box::new(|x| vec![x[0] + x[1], x[1], x[0]]),
        jac: Box::new(|_| vec![vec![1.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero(), zero(), zero()]
        }),
        finalized: Rc::clone(&finalized),
    };
    let pairs = vec![
        ComplementarityPair { row: 1, var: 0 },
        ComplementarityPair { row: 2, var: 1 },
    ];
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let out = solve_mpcc(&mut app, tnlp, pairs, &mut |_| {}).unwrap();
    assert_eq!(
        out.accepted_by,
        Some(MpccStage::Continuation),
        "{:?}",
        out.solves
    );
    assert!(
        out.solves
            .iter()
            .any(|s| s.status == ApplicationReturnStatus::NotEnoughDegreesOfFreedom),
        "the equality finish should have been refused: {:?}",
        out.solves
    );
    assert!(
        (out.x[0] - 0.5).abs() < 1e-5 && (out.x[1] - 0.5).abs() < 1e-5,
        "{:?}",
        out.x
    );
    assert_eq!(finalized.borrow().len(), 1);
    // Three active gradients in R^2: MPCC-LICQ fails, the multipliers are a
    // set, and with no biactive pair the set search finds S.
    let cert = out.certificate.unwrap();
    assert_eq!(
        (cert.class, cert.n_biactive, cert.multipliers_unique),
        (StationarityClass::S, 0, false),
        "{cert:?}"
    );
}

/// `ralph1`: `min 2x - y,  x >= 0,  0 <= y ⟂ y - x >= 0`. The solution is the
/// origin, where no S-stationary multiplier exists (it is M-stationary only).
/// The driver converges there and must say so: accepted, but certified W,
/// not S — the case where "the solver succeeded" and "this is a strongly
/// stationary point" part ways. Three active gradients in R^2 make the
/// multipliers a set, so this exercises the certificate's set search; its
/// unique-multiplier path is covered by `pounce_algorithm::mpcc`'s tests.
#[test]
fn a_converged_point_that_is_not_strongly_stationary_is_reported_as_such() {
    let zero = || vec![vec![0.0; 2]; 2];
    let model = Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, 0.0],
        x_u: vec![INF, INF],
        g_l: vec![0.0],
        g_u: vec![INF],
        x0: vec![1.0, 2.0],
        f: Box::new(|x| 2.0 * x[0] - x[1]),
        grad: Box::new(|_| vec![2.0, -1.0]),
        g: Box::new(|x| vec![x[1] - x[0]]),
        jac: Box::new(|_| vec![vec![-1.0, 1.0]]),
        hess: Box::new(move |_| vec![zero(), zero()]),
        finalized: Rc::new(RefCell::new(Vec::new())),
    };
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 1 }],
        &mut |_| {},
    )
    .unwrap();
    // A W point is a fallback, not an answer (PR review): `accepted_by` is
    // reserved for a strongly stationary answer.
    assert_eq!(out.accepted_by, None, "{:?}", out.solves);
    assert!(out.fallback.is_some(), "{:?}", out.solves);
    assert!(
        matches!(out.reported, Reported::WeakFallback(_)),
        "{:?}",
        out.reported
    );
    assert!(out.x.iter().all(|v| v.abs() < 1e-4), "{:?}", out.x);
    let cert = out.certificate.unwrap();
    assert_eq!(
        (cert.class, cert.n_biactive),
        (StationarityClass::W, 1),
        "{cert:?}"
    );
}

/// CALIPSO.jl's `nonconvex/complementarity` example, pairs declared:
///
/// ```text
/// min (x0 - 5)^2 + (2 x1 + 1)^2,   x >= 0,
///   2(x1 - 1) - 1.5 x1 + x2 - 0.5 x3 + x4 = 0
///   3 x0 - x1 - 3 - x5 = 0,  -x0 + 0.5 x1 + 4 - x6 = 0,  -x0 - x1 + 7 - x7 = 0
///   0 <= x_{2+k}  ⟂  x_{5+k} >= 0,   k = 0, 1, 2
/// ```
///
/// From the origin the exact-product stage converges to a W-stationary point
/// at objective 85; the continuation reaches the S-stationary optimum
/// `(1, 0, 2, 0, 0, 0, 3, 6)` at 17. A W point must not end the ladder.
#[test]
fn a_weakly_stationary_point_does_not_stop_the_search_for_a_strong_one() {
    let zero = || vec![vec![0.0; 8]; 8];
    let unit = |j: usize| {
        let mut r = vec![0.0; 8];
        r[j] = 1.0;
        r
    };
    let model = Dense {
        n: 8,
        m: 7,
        x_l: vec![0.0; 8],
        x_u: vec![INF; 8],
        g_l: vec![0.0; 7],
        g_u: vec![0.0, 0.0, 0.0, 0.0, INF, INF, INF],
        x0: vec![0.0; 8],
        f: Box::new(|x| (x[0] - 5.0).powi(2) + (2.0 * x[1] + 1.0).powi(2)),
        grad: Box::new(|x| {
            let mut g = vec![0.0; 8];
            g[0] = 2.0 * (x[0] - 5.0);
            g[1] = 4.0 * (2.0 * x[1] + 1.0);
            g
        }),
        g: Box::new(|x| {
            vec![
                2.0 * (x[1] - 1.0) - 1.5 * x[1] + x[2] - 0.5 * x[3] + x[4],
                3.0 * x[0] - x[1] - 3.0 - x[5],
                -x[0] + 0.5 * x[1] + 4.0 - x[6],
                -x[0] - x[1] + 7.0 - x[7],
                x[5],
                x[6],
                x[7],
            ]
        }),
        jac: Box::new(move |_| {
            vec![
                vec![0.0, 0.5, 1.0, -0.5, 1.0, 0.0, 0.0, 0.0],
                vec![3.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                vec![-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                unit(5),
                unit(6),
                unit(7),
            ]
        }),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 8.0;
            let mut h = vec![f];
            h.extend((0..7).map(|_| zero()));
            h
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    };
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let pairs = (0..3)
        .map(|k| ComplementarityPair {
            row: 4 + k,
            var: 2 + k,
        })
        .collect();
    let out = solve_mpcc(&mut app, tnlp, pairs, &mut |_| {}).unwrap();
    assert_eq!(
        out.accepted_by,
        Some(MpccStage::Continuation),
        "{:?}",
        out.solves
    );
    for (got, want) in out.x.iter().zip([1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 6.0]) {
        assert!((got - want).abs() < 1e-4, "x = {:?}", out.x);
    }
    assert_eq!(out.certificate.unwrap().class, StationarityClass::S);
}

/// Two branches, two objectives:
///
/// ```text
/// min (x - 0.9)^2 + (y - 1)^2   s.t.  0 <= x ⟂ y >= 0
/// ```
///
/// `(0, 1)` at 0.81 is the optimum; `(0.9, 0)` at 1.0 is an S-stationary
/// local minimum. From `(0.9, 0.01)` the cold exact product commits to the
/// near branch and stops at the local minimum, while a relaxation sees both.
/// Every candidate runs (the continuation, and the exact stage's relaxed path
/// and cold solve), and the lower objective is the answer whichever produced
/// it.
#[test]
fn the_continuation_and_the_exact_product_both_run_and_the_lower_objective_wins() {
    let model = two_branch([0.9, 0.01]);
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let mut lines = Vec::new();
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 0 }],
        &mut |l| lines.push(l.to_string()),
    )
    .unwrap();
    let solved: Vec<_> = lines
        .iter()
        .filter(|l| l.contains("solved the model"))
        .collect();
    assert!(
        solved.len() >= 2 && solved.iter().any(|l| l.contains("exact product")),
        "both stages answer: {lines:#?}"
    );
    assert!(
        out.x[0].abs() < 1e-5 && (out.x[1] - 1.0).abs() < 1e-5,
        "{:?} {lines:#?}",
        out.x
    );
    assert_eq!(out.certificate.unwrap().class, StationarityClass::S);
    assert!(out.solves.iter().all(|s| s.stage != MpccStage::Penalty));
}

/// The previous model maximized through `obj_scaling_factor = -1`
/// (upstream's spelling of "maximize"): same answer, certified S. The stage
/// comparison and the certificate work on the minimized `σ·f`; on the raw
/// `f` the certificate's multipliers carry the wrong sign.
#[test]
fn a_negative_obj_scaling_factor_is_maximized_and_certified_on_its_sign() {
    let zero = || vec![vec![0.0; 2]; 2];
    let model = Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![0.0],
        g_u: vec![INF],
        x0: vec![0.9, 0.01],
        f: Box::new(|x| -((x[0] - 0.9).powi(2) + (x[1] - 1.0).powi(2))),
        grad: Box::new(|x| vec![-2.0 * (x[0] - 0.9), -2.0 * (x[1] - 1.0)]),
        g: Box::new(|x| vec![x[1]]),
        jac: Box::new(|_| vec![vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = -2.0;
            f[1][1] = -2.0;
            vec![f, zero()]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    };
    let mut app = app();
    app.options_mut()
        .read_from_str("obj_scaling_factor -1\n", true)
        .unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 0 }],
        &mut |_| {},
    )
    .unwrap();
    assert!(
        out.x[0].abs() < 1e-5 && (out.x[1] - 1.0).abs() < 1e-5,
        "{:?} {:?}",
        out.x,
        out.solves
    );
    assert_eq!(out.certificate.unwrap().class, StationarityClass::S);
    assert!(
        matches!(out.reported, Reported::Answer(_)),
        "{:?}",
        out.reported
    );
}

/// `Dense` with an intermediate callback that stops the solve on its
/// `stop_at`-th call.
struct Stopping {
    inner: Dense,
    calls: Rc<RefCell<usize>>,
    stop_at: usize,
    /// `false` on the `stop_at`-th call only (a stop button), rather than
    /// on every call from there on.
    once: bool,
}

impl TNLP for Stopping {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
        self.inner.eval_f(x, n)
    }
    fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
        self.inner.eval_grad_f(x, n, g)
    }
    fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
        self.inner.eval_g(x, n, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.eval_jac_g(x, n, mode)
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        n: bool,
        s: Number,
        lam: Option<&[Number]>,
        nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner.eval_h(x, n, s, lam, nl, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
        self.inner.finalize_solution(sol, d, q)
    }
    fn wants_intermediate_callback(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, _: IterStats, _: &IpoptData, _: &IpoptCq) -> bool {
        *self.calls.borrow_mut() += 1;
        let k = *self.calls.borrow();
        if self.once {
            k != self.stop_at
        } else {
            k < self.stop_at
        }
    }
}

/// The caller's intermediate callback reaches the lowered solves and can
/// stop them (PR review of #1103: `MpccTnlp` dropped it, so a callback that
/// returned `false` on its 3rd call was never called and the run reported
/// `Solve_Succeeded`). A stop ends the whole ladder: no later stage runs.
#[test]
fn the_callers_intermediate_callback_is_called_and_can_stop_the_solve() {
    let calls = Rc::new(RefCell::new(0));
    let model = Stopping {
        inner: qpec_small([0.0, 0.5, 0.5], Rc::new(RefCell::new(Vec::new()))),
        calls: Rc::clone(&calls),
        stop_at: 3,
        once: false,
    };
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let out = solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |_| {}).unwrap();
    assert_eq!(*calls.borrow(), 3, "{:?}", out.solves);
    assert_eq!(out.status, ApplicationReturnStatus::UserRequestedStop);
    assert_eq!(out.solves.len(), 1, "the ladder went on: {:?}", out.solves);
    assert_eq!(out.accepted_by, None);
}

/// A stop wherever it lands — a relaxation, the primal-only rung, the
/// continuation's finish, the exact stage's retry — ends the run there
/// (PR review of #1103: a stop in an unchecked solve was dropped, and one
/// returned once during the finish was overridden by a success). Swept over
/// every callback of the unstopped run, for a callback that returns `false`
/// once and for one that keeps returning it: the run reports
/// `User_Requested_Stop`, the stopped solve is the last one started, and no
/// callback is made after the stop.
#[test]
fn a_stop_anywhere_in_the_ladder_ends_it() {
    let solve = |stop_at: usize, once: bool| {
        let calls = Rc::new(RefCell::new(0));
        let model = Stopping {
            inner: qpec_small([0.0, 0.5, 0.5], Rc::new(RefCell::new(Vec::new()))),
            calls: Rc::clone(&calls),
            stop_at,
            once,
        };
        let mut app = app();
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
        let out = solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |_| {}).unwrap();
        let n = *calls.borrow();
        (out, n)
    };
    let (free, total) = solve(usize::MAX, false);
    assert_ne!(free.status, ApplicationReturnStatus::UserRequestedStop);
    assert!(
        free.solves.len() > 3 && total > 20,
        "{total} {:?}",
        free.solves
    );
    // Every call for a one-time stop; a persistent stop lands in the same
    // places, so every tenth call covers it.
    for (once, stride) in [(true, 1), (false, 10)] {
        for stop_at in (1..=total).step_by(stride) {
            let (out, n) = solve(stop_at, once);
            let ctx = format!("stop at call {stop_at} (once = {once}): {:?}", out.solves);
            assert_eq!(
                out.status,
                ApplicationReturnStatus::UserRequestedStop,
                "{ctx}"
            );
            assert_eq!(n, stop_at, "callbacks after the stop; {ctx}");
            assert_eq!(
                out.solves.last().map(|s| s.status),
                Some(ApplicationReturnStatus::UserRequestedStop),
                "a solve started after the stop; {ctx}"
            );
        }
    }
}

/// An infeasible complementarity model: `x >= 1`, `y >= 1` and
/// `x + y <= 1.5` cannot hold together whatever the pair does. Every stage
/// runs and fails, so every option any stage writes is written.
///
/// ```text
/// min x^2 + y^2   s.t.  x + y <= 1.5,   0 <= x - 1 ⟂ y - 1 >= 0
/// ```
fn infeasible_pair(finalized: Rc<RefCell<Vec<Vec<Number>>>>) -> Dense {
    let zero = || vec![vec![0.0; 2]; 2];
    Dense {
        n: 2,
        m: 2,
        x_l: vec![1.0, 0.0],
        x_u: vec![INF, INF],
        g_l: vec![-INF, 1.0],
        g_u: vec![1.5, INF],
        x0: vec![1.5, 1.5],
        f: Box::new(|x| x[0] * x[0] + x[1] * x[1]),
        grad: Box::new(|x| vec![2.0 * x[0], 2.0 * x[1]]),
        g: Box::new(|x| vec![x[0] + x[1], x[1]]),
        jac: Box::new(|_| vec![vec![1.0, 1.0], vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero(), zero()]
        }),
        finalized,
    }
}

/// No stage can succeed: the driver reports the exact-product stage's
/// verdict — what a plain solve of the lowered model says — forwards that
/// one point to the model, and has run every stage.
#[test]
fn when_no_stage_succeeds_the_exact_verdict_is_reported() {
    let finalized = Rc::new(RefCell::new(Vec::new()));
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(infeasible_pair(Rc::clone(&finalized))));
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 1, var: 0 }],
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.accepted_by, None, "{:?}", out.solves);
    let exact = out.solves.iter().find(|s| s.stage == MpccStage::Exact);
    assert_eq!(
        Some(out.status),
        exact.map(|s| s.status),
        "{:?}",
        out.solves
    );
    assert!(
        out.solves
            .iter()
            .any(|s| s.stage == MpccStage::Continuation)
    );
    assert!(out.solves.iter().any(|s| s.stage == MpccStage::Penalty));
    assert_eq!(finalized.borrow().len(), 1);
}

/// Gate 0's `infeasible_pair`: two equalities force `(1/2, 1/2)`, where the
/// product is 1/4, so the MPCC has no feasible point; and the model is
/// square, so the exact product's equality row is refused before iteration
/// 0. Declared, this once reported a relaxed solve's `Solve_Succeeded`
/// (continuation first, nothing answering) and, before that, forwarded the
/// refused stage's empty point. The exact stage's relaxation `G·H <= 1e-6`
/// is infeasible too (the product is pinned at 1/4). Its cold solve is
/// refused as `G·H = 0` (the square model) and retried as `G·H <= 0`, which
/// reaches the same verdict with a point of its own; that verdict —
/// infeasible — is what is reported.
///
/// ```text
/// min x^2 + y^2   s.t.  x + y = 1,  x - y = 0,   0 <= x ⟂ y >= 0
/// ```
#[test]
fn a_square_infeasible_model_reports_the_exact_verdict_with_a_point() {
    let zero = || vec![vec![0.0; 2]; 2];
    let finalized = Rc::new(RefCell::new(Vec::new()));
    let model = Dense {
        n: 2,
        m: 3,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![1.0, 0.0, 0.0],
        g_u: vec![1.0, 0.0, INF],
        x0: vec![0.0, 0.0],
        f: Box::new(|x| x[0] * x[0] + x[1] * x[1]),
        grad: Box::new(|x| vec![2.0 * x[0], 2.0 * x[1]]),
        g: Box::new(|x| vec![x[0] + x[1], x[0] - x[1], x[1]]),
        jac: Box::new(|_| vec![vec![1.0, 1.0], vec![1.0, -1.0], vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero(), zero(), zero()]
        }),
        finalized: Rc::clone(&finalized),
    };
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 2, var: 0 }],
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(out.accepted_by, None, "{:?}", out.solves);
    let exact: Vec<_> = out
        .solves
        .iter()
        .filter(|s| s.stage == MpccStage::Exact)
        .collect();
    assert_eq!(exact.len(), 3, "{:?}", out.solves);
    assert_eq!(exact[0].tau, Some(EXACT_STAGE_TAU), "{:?}", out.solves);
    assert_eq!(
        exact[1].status,
        ApplicationReturnStatus::NotEnoughDegreesOfFreedom,
        "the cold G·H = 0 is refused: {:?}",
        out.solves
    );
    assert_eq!(exact[2].tau, None, "the G·H <= 0 retry: {:?}", out.solves);
    assert_eq!(
        exact[0].status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "{:?}",
        out.solves
    );
    assert_eq!(out.status, exact[0].status);
    // The continuation's finish is refused before iteration 0 (the square
    // model's equality row) and has no point of its own: it must not report
    // the schedule's last point (PR review: `finalize_solution` never runs for
    // a solve refused before iteration 0, so an uncleared capture carried the
    // previous solve's x and products).
    let refused = out
        .solves
        .iter()
        .find(|s| s.status == ApplicationReturnStatus::NotEnoughDegreesOfFreedom)
        .expect("the finish's equality row is refused");
    assert!(
        refused.max_abs_product.is_infinite(),
        "the refused solve reports a point: {:?}",
        out.solves
    );
    assert!(
        !matches!(
            out.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{:?}",
        out.status
    );
    assert_eq!(out.x.len(), 2);
    assert_eq!(finalized.borrow().len(), 1);
    assert_eq!(finalized.borrow()[0].len(), 2);
}

/// Every option a stage writes reads afterwards exactly as before, set-ness
/// included. On the infeasible model every stage runs, so the warm-start
/// options and `l1_exact_penalty_barrier` are all written during the solve;
/// none of it may leak into the caller's next solve.
#[test]
fn options_are_restored_to_set_ness() {
    let mut app = app();
    // One option the caller set, the rest left unset.
    app.options_mut()
        .set_numeric_value("mu_init", 0.05, true, false)
        .unwrap();
    let show = |app: &IpoptApplication| -> Vec<String> {
        TOUCHED_OPTIONS
            .iter()
            .map(|k| {
                let set = app.options().names().any(|n| n == *k);
                let v = app
                    .options()
                    .get_string_value(k, "")
                    .map(|(v, _)| v)
                    .unwrap_or_default();
                format!("{k} set={set} {v}")
            })
            .collect()
    };
    let before = show(&app);
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(infeasible_pair(Rc::new(
        RefCell::new(Vec::new()),
    ))));
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 1, var: 0 }],
        &mut |_| {},
    )
    .unwrap();
    assert!(
        out.solves.iter().any(|s| s.stage == MpccStage::Penalty),
        "{:?}",
        out.solves
    );
    assert_eq!(before, show(&app));
}

#[test]
fn a_pair_it_cannot_lower_is_refused_with_the_reason() {
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(mixed(false)));
    // y has no finite bound and row 0 is free: neither one-sided nor mixed.
    let e = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 1 }],
        &mut |_| {},
    )
    .err()
    .unwrap();
    assert!(e.contains("one-sided") && e.contains("mixed"), "{e}");
}

/// The equality-split model, where the two formulations reach different
/// S-stationary points (PR review, round 2):
///
/// ```text
/// min x^2 + 2 y^2   s.t.  x + y = 1,   0 <= x ⟂ y >= 0
/// ```
///
/// The branches are `(1, 0)` with `f = 1` and `(0, 1)` with `f = 2`; from
/// `(0.5, 0.5)` both formulations reach the first.
fn equality_split() -> Dense {
    let zero = || vec![vec![0.0; 2]; 2];
    Dense {
        n: 2,
        m: 2,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![1.0, 0.0],
        g_u: vec![1.0, INF],
        x0: vec![0.5, 0.5],
        f: Box::new(|x| x[0] * x[0] + 2.0 * x[1] * x[1]),
        grad: Box::new(|x| vec![2.0 * x[0], 4.0 * x[1]]),
        g: Box::new(|x| vec![x[0] + x[1], x[1]]),
        jac: Box::new(|_| vec![vec![1.0, 1.0], vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 4.0;
            vec![f, zero(), zero()]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

fn solve_with_strategy(
    model: Dense,
    strategy: Option<&str>,
    pair: ComplementarityPair,
) -> MpccOutcome {
    let mut app = app();
    if let Some(s) = strategy {
        app.options_mut()
            .read_from_str(&format!("complementarity_strategy {s}\n"), true)
            .unwrap();
    }
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    solve_mpcc(&mut app, tnlp, vec![pair], &mut |_| {}).unwrap()
}

fn stages(out: &MpccOutcome) -> Vec<MpccStage> {
    let mut s: Vec<MpccStage> = out.solves.iter().map(|r| r.stage).collect();
    s.dedup();
    s
}

/// `complementarity_strategy` (PR review, round 4): `both` (the default)
/// runs the two formulations and keeps the better answer; `exact` and
/// `continuation` run only theirs, so a user who knows which works pays for
/// one.
#[test]
fn complementarity_strategy_picks_the_formulations_that_run() {
    let pair = ComplementarityPair { row: 1, var: 0 };
    let f = |out: &MpccOutcome| out.x[0] * out.x[0] + 2.0 * out.x[1] * out.x[1];

    let both = solve_with_strategy(equality_split(), None, pair);
    assert_eq!(stages(&both), [MpccStage::Continuation, MpccStage::Exact]);
    assert!((f(&both) - 1.0).abs() < 1e-6, "{:?}", both.x);

    let exact = solve_with_strategy(equality_split(), Some("exact"), pair);
    assert_eq!(stages(&exact), [MpccStage::Exact], "{:?}", exact.solves);
    assert!((f(&exact) - 1.0).abs() < 1e-6, "{:?}", exact.x);
    assert_eq!(exact.accepted_by, Some(MpccStage::Exact));

    let cont = solve_with_strategy(equality_split(), Some("continuation"), pair);
    assert_eq!(
        stages(&cont),
        [MpccStage::Continuation],
        "{:?}",
        cont.solves
    );
    assert!((f(&cont) - 1.0).abs() < 1e-6, "{:?}", cont.x);
    assert_eq!(cont.accepted_by, Some(MpccStage::Continuation));

    // The ladder's iterations are the single route's, not both.
    let iters = |o: &MpccOutcome| o.total_iterations;
    assert!(iters(&exact) < iters(&both) && iters(&cont) < iters(&both));
}

/// With `continuation` alone and nothing answering, the verdict is never a
/// relaxed solve's: on `infeasible_pair` the first relaxation is already
/// infeasible, so no finish runs, the penalty runs as the last resort, and
/// the infeasibility verdict is reported. The exact stage never runs.
#[test]
fn continuation_alone_still_reports_an_infeasible_verdict() {
    let finalized = Rc::new(RefCell::new(Vec::new()));
    let out = solve_with_strategy(
        infeasible_pair(Rc::clone(&finalized)),
        Some("continuation"),
        ComplementarityPair { row: 1, var: 0 },
    );
    assert_eq!(out.accepted_by, None, "{:?}", out.solves);
    assert!(out.solves.iter().all(|s| s.stage != MpccStage::Exact));
    assert!(out.solves.iter().any(|s| s.stage == MpccStage::Penalty));
    assert_eq!(
        out.status,
        ApplicationReturnStatus::InfeasibleProblemDetected,
        "{:?}",
        out.solves
    );
    assert_eq!(finalized.borrow().len(), 1);
}

/// A model whose relaxations are feasible but whose MPCC is not:
///
/// ```text
/// min x^2 + y^2   s.t.  x + y = 1,  x >= 0.3,  y >= 0.3,   0 <= x ⟂ y >= 0
/// ```
///
/// `x·y >= 0.21` on the feasible segment, so `τ = 1` solves and `τ = 0.1` is
/// already infeasible, the schedule stops there, and the exact finish from
/// the `τ = 1` point is infeasible too; the relaxed point's products are far
/// outside tolerance, so nothing answers.
fn relaxation_only_feasible() -> Dense {
    let zero = || vec![vec![0.0; 2]; 2];
    Dense {
        n: 2,
        m: 4,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![1.0, 0.0, 0.3, 0.3],
        g_u: vec![1.0, INF, INF, INF],
        x0: vec![0.5, 0.5],
        f: Box::new(|x| x[0] * x[0] + x[1] * x[1]),
        grad: Box::new(|x| vec![2.0 * x[0], 2.0 * x[1]]),
        g: Box::new(|x| vec![x[0] + x[1], x[1], x[0], x[1]]),
        jac: Box::new(|_| {
            vec![
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ]
        }),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero(), zero(), zero(), zero()]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

/// With `continuation` alone, the exact stage is not there to supply the
/// verdict, so the continuation's own exact finish does: on a model whose
/// relaxations solve and whose MPCC is infeasible, the reported status is the
/// finish's, never the `τ = 1` relaxation's `Solve_Succeeded`.
#[test]
fn continuation_alone_reports_its_exact_finish_as_the_verdict() {
    let out = solve_with_strategy(
        relaxation_only_feasible(),
        Some("continuation"),
        ComplementarityPair { row: 1, var: 0 },
    );
    assert_eq!(out.accepted_by, None, "{:?}", out.solves);
    assert!(
        out.solves
            .iter()
            .any(|s| s.tau.is_some() && s.status == ApplicationReturnStatus::SolveSucceeded),
        "no relaxation solved: {:?}",
        out.solves
    );
    let finish = out
        .solves
        .iter()
        .rfind(|s| s.stage == MpccStage::Continuation && s.tau.is_none())
        .expect("the finish ran");
    assert_ne!(finish.status, ApplicationReturnStatus::SolveSucceeded);
    assert_eq!(out.status, finish.status, "{:?}", out.solves);
}

/// A mixed pair, `0 <= x <= 1 ⟂ g = y - x` with `y` and the row free, whose
/// optimum is at the bound named: `min (x - 0.5)² + (y - 2)²` at `(0, 2)`,
/// `g = 2 >= 0` (f = 0.25); `min (x - 2)² + y²` at `(1, 0)`, `g = -1 <= 0`
/// (f = 1). The interior branch `y = x` is worse on both.
fn mixed(at_upper: bool) -> Dense {
    let zero = || vec![vec![0.0; 2]; 2];
    let (cx, cy) = if at_upper { (2.0, 0.0) } else { (0.5, 2.0) };
    Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, -INF],
        x_u: vec![1.0, INF],
        g_l: vec![-INF],
        g_u: vec![INF],
        x0: vec![0.5, 0.5],
        f: Box::new(move |x| (x[0] - cx).powi(2) + (x[1] - cy).powi(2)),
        grad: Box::new(move |x| vec![2.0 * (x[0] - cx), 2.0 * (x[1] - cy)]),
        g: Box::new(|x| vec![x[1] - x[0]]),
        jac: Box::new(|_| vec![vec![-1.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero()]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

/// A mixed (two-sided) pair is lowered and solved, not refused. Every
/// strategy returns a strongly stationary point that satisfies the pair (in
/// the box; `g >= 0` at 0, `g <= 0` at 1, `g = 0` between); the default runs
/// both formulations and reaches the optimum at whichever bound it lies. A
/// single formulation can stop at a local minimum: from `(0.5, 0.5)` on the
/// lower-bound model the exact product reaches `(1, 1)`, `f = 1.25` — the
/// interior branch `y = x` pressed against `x <= 1`, a genuine local minimum.
#[test]
fn a_mixed_pair_is_solved_at_either_bound() {
    for (at_upper, want, f_want) in [(false, [0.0, 2.0], 0.25), (true, [1.0, 0.0], 1.0)] {
        let (cx, cy) = if at_upper { (2.0, 0.0) } else { (0.5, 2.0) };
        for strategy in [None, Some("exact"), Some("continuation")] {
            let model = mixed(at_upper);
            let fin = Rc::clone(&model.finalized);
            let out = solve_with_strategy(model, strategy, ComplementarityPair { row: 0, var: 0 });
            let ctx = format!(
                "upper={at_upper} {strategy:?}: x = {:?}, {:?}",
                out.x, out.solves
            );
            assert!(out.accepted_by.is_some(), "{ctx}");
            assert_eq!(
                out.certificate.as_ref().map(|c| c.class),
                Some(StationarityClass::S),
                "{ctx}"
            );
            assert_eq!(fin.borrow().len(), 1, "one forwarded answer; {ctx}");
            // The mixed conditions, to the solve's accuracy.
            let (x, g) = (out.x[0], out.x[1] - out.x[0]);
            assert!((-1e-8..=1.0 + 1e-8).contains(&x), "{ctx}");
            let resid = (x.min(g.max(0.0))).max((1.0 - x).min((-g).max(0.0)));
            assert!(resid < 1e-6, "mixed residual {resid}; {ctx}");
            if strategy.is_none() {
                for (a, b) in out.x.iter().zip(want) {
                    assert!((a - b).abs() < 1e-6, "want {want:?}; {ctx}");
                }
                let f = (out.x[0] - cx).powi(2) + (out.x[1] - cy).powi(2);
                assert!((f - f_want).abs() < 1e-6, "f = {f}; {ctx}");
            }
        }
    }
}

/// The branch escape's pieces, as the driver chains them, from a point no
/// formulation stops at on its own: the corner `(0, 0)` of
/// `min (x - 1)² + (y - 1)²`, `0 <= x ⟂ y >= 0` (row `y >= 0`). The check
/// finds a descending branch; the branch NLP -- the lowered model with the
/// held side pinned and the product rows free, warm-started at the corner --
/// reaches that branch's optimum, `(0, 1)` or `(1, 0)` with `f = 1`, below
/// the corner's `f = 2` and certified S.
#[test]
fn a_branch_nlp_leaves_a_corner_that_is_not_b_stationary() {
    use pounce_algorithm::mpcc::{
        BVerdict, CertificateInput, branch_bounds, certify, check_b_stationarity,
    };
    use pounce_nlp::mpcc::{Lowering, MpccTnlp, StartOverride};

    let zero = || vec![vec![0.0; 2]; 2];
    let finalized = Rc::new(RefCell::new(Vec::new()));
    let corner = || Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![0.0],
        g_u: vec![INF],
        x0: vec![0.5, 0.5],
        f: Box::new(|x| (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2)),
        grad: Box::new(|x| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)]),
        g: Box::new(|x| vec![x[1]]),
        jac: Box::new(|_| vec![vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero()]
        }),
        finalized: Rc::clone(&finalized),
    };
    let pairs = vec![ComplementarityPair { row: 0, var: 0 }];
    let jac = [(0usize, 1usize, 1.0)];
    let (xl, xu, gl, gu) = ([0.0, -INF], [INF, INF], [0.0], [INF]);
    let inp =
        |x: &'static [f64; 2], gf: &'static [f64; 2], g: &'static [f64; 1]| CertificateInput {
            x,
            grad_f: gf,
            g,
            jac: &jac,
            x_l: &xl,
            x_u: &xu,
            g_l: &gl,
            g_u: &gu,
            pairs: &pairs,
            solver_tol: 1e-8,
            feas_tol: 1e-4,
        };
    let at_corner = inp(&[0.0, 0.0], &[-2.0, -2.0], &[0.0]);
    let BVerdict::NotB { held, .. } = check_b_stationarity(&at_corner).verdict else {
        panic!("the corner should not be B-stationary")
    };
    let bounds = branch_bounds(&at_corner, &held);

    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(corner()));
    let w = Rc::new(RefCell::new(
        MpccTnlp::new(Rc::clone(&inner), pairs.clone(), Lowering::Ineq).unwrap(),
    ));
    w.borrow_mut().set_defer_inner_finalize(true);
    w.borrow_mut().set_branch(Some(bounds));
    w.borrow_mut().set_start_override(Some(StartOverride {
        x: vec![0.0, 0.0],
        ..Default::default()
    }));
    let mut app = app();
    let lowered: Rc<RefCell<dyn TNLP>> = w.clone();
    let status = app.optimize_tnlp(lowered);
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{status:?}"
    );
    let x = w.borrow().last_x().to_vec();
    let f = (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2);
    assert!((f - 1.0).abs() < 1e-6, "x = {x:?}, f = {f}");
    assert!(
        x[0].abs() < 1e-6 || x[1].abs() < 1e-6,
        "complementary: {x:?}"
    );
    let (gf, g) = ([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 1.0)], [x[1]]);
    let cert = certify(&CertificateInput {
        x: &x,
        grad_f: &gf,
        g: &g,
        ..inp(&[0.0, 0.0], &[0.0, 0.0], &[0.0])
    });
    assert_eq!(cert.class, StationarityClass::S, "{cert:?}");
    // The inner model saw no finalize: the branch solve is deferred like
    // every lowered solve, so only the driver forwards an answer.
    assert!(finalized.borrow().is_empty());
}

/// The exact stage's and the finish's iteration budgets never raise the
/// caller's own `max_iter`, and the option reads afterwards as the caller
/// left it -- set, at their value -- even though a budgeted stage wrote it.
/// qpec_small's continuation answers, so both budgets are written here.
#[test]
fn iteration_budgets_respect_and_restore_the_callers_max_iter() {
    for caller in [None, Some(7)] {
        let mut app = app();
        if let Some(it) = caller {
            app.options_mut()
                .set_integer_value("max_iter", it, true, false)
                .unwrap();
        }
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(qpec_small(
            [0.0, 0.5, 0.5],
            Rc::new(RefCell::new(Vec::new())),
        )));
        let out = solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |_| {}).unwrap();
        if let Some(it) = caller {
            // The budgeted solves: the exact stage and the continuation's
            // finish. (A relaxation's spent count can exceed `max_iter`: it
            // includes the solver's own re-solves, which no budget governs.)
            let budgeted = out.solves.iter().filter(|s| {
                s.stage == MpccStage::Exact
                    || (s.stage == MpccStage::Continuation && s.tau.is_none())
            });
            for s in budgeted {
                assert!(
                    s.iterations <= it as usize,
                    "a solve ran past the caller's max_iter {it}: {:?}",
                    out.solves
                );
            }
        } else {
            assert!(out.accepted_by.is_some(), "{:?}", out.solves);
            assert!(
                out.solves.iter().any(|s| s.stage == MpccStage::Exact),
                "the exact stage did not run: {:?}",
                out.solves
            );
        }
        let set = app.options().names().any(|n| n == "max_iter");
        let v = app
            .options()
            .get_integer_value("max_iter", "")
            .map(|(v, _)| v)
            .unwrap();
        assert_eq!(set, caller.is_some(), "max_iter set-ness leaked");
        assert_eq!(v, caller.unwrap_or(3000), "max_iter value leaked");
    }
}

/// The exact stage's relaxed path by what its relaxed solve returns --
/// converged (tighten to the continuation's last `τ`, then the exact finish),
/// infeasible (that verdict is the model's: the square infeasible model
/// above), or failed some other way (here out of iterations, when it
/// contributes no candidate) -- and, whichever it is, the cold exact product
/// after it.
#[test]
fn the_exact_stage_runs_its_relaxed_path_and_the_cold_exact_product() {
    let run = |max_iter: Option<i32>| {
        let mut app = app();
        app.options_mut()
            .set_string_value("complementarity_strategy", "exact", true, false)
            .unwrap();
        if let Some(it) = max_iter {
            app.options_mut()
                .set_integer_value("max_iter", it, true, false)
                .unwrap();
        }
        let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(qpec_small(
            [0.0, 0.5, 0.5],
            Rc::new(RefCell::new(Vec::new())),
        )));
        solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |_| {}).unwrap()
    };
    let taus = |out: &MpccOutcome| -> Vec<Option<f64>> {
        out.solves
            .iter()
            .filter(|s| s.stage == MpccStage::Exact)
            .map(|s| s.tau)
            .collect()
    };
    // Converged: relaxed, tightened, finished; then the cold solve. The
    // answer is S.
    let ok = run(None);
    assert_eq!(
        taus(&ok),
        [Some(EXACT_STAGE_TAU), Some(1e-8), None, None],
        "{:?}",
        ok.solves
    );
    assert_eq!(ok.accepted_by, Some(MpccStage::Exact), "{:?}", ok.solves);
    // Out of iterations: the relaxation fails without a verdict, and the
    // cold solve still runs.
    let failed = run(Some(2));
    let exact: Vec<_> = failed
        .solves
        .iter()
        .filter(|s| s.stage == MpccStage::Exact)
        .collect();
    assert_eq!(exact[0].tau, Some(EXACT_STAGE_TAU), "{:?}", failed.solves);
    assert_eq!(
        exact[0].status,
        ApplicationReturnStatus::MaximumIterationsExceeded,
        "{:?}",
        failed.solves
    );
    assert_eq!(exact.len(), 2, "{:?}", failed.solves);
    assert_eq!(exact[1].tau, None, "no cold solve: {:?}", failed.solves);
}

/// The exact stage's cold `G·H = 0` solve runs only when the cheap paths
/// disagree. On qpec_small the continuation and the exact stage's relaxed
/// path land on the same answer, so the cold solve is skipped; on the
/// two-branch model below them, from `(0.9, 0.01)`, the relaxed path settles
/// on the near branch (1.0) and the continuation on the optimum (0.81), so
/// it runs. Either way the reported answer is the optimum.
#[test]
fn the_cold_exact_product_runs_only_when_the_cheap_paths_disagree() {
    let exact_taus = |out: &MpccOutcome| -> Vec<Option<f64>> {
        out.solves
            .iter()
            .filter(|s| s.stage == MpccStage::Exact)
            .map(|s| s.tau)
            .collect()
    };
    // Agree: relaxed, 1e-8 step, finish -- and no cold solve.
    let mut app = app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(qpec_small(
        [0.0, 0.5, 0.5],
        Rc::new(RefCell::new(Vec::new())),
    )));
    let mut lines = Vec::new();
    let out = solve_mpcc(&mut app, tnlp, qpec_pairs(), &mut |l| {
        lines.push(l.to_string())
    })
    .unwrap();
    assert_eq!(
        exact_taus(&out),
        [Some(EXACT_STAGE_TAU), Some(1e-8), None],
        "{:?} {lines:#?}",
        out.solves
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("skipping the cold exact product")),
        "{lines:#?}"
    );
    for (got, want) in out.x.iter().zip([1.0, 1.0, 0.0]) {
        assert!((got - want).abs() < 1e-4, "x = {:?}", out.x);
    }

    // Disagree: the continuation reaches the optimum (0.81) and the relaxed
    // path the near branch (1.0), so after the relaxed path's three solves
    // the cold exact product runs as well.
    let mut app = crate::app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(two_branch([0.9, 0.01])));
    let mut lines = Vec::new();
    let out = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 0 }],
        &mut |l| lines.push(l.to_string()),
    )
    .unwrap();
    assert_eq!(
        exact_taus(&out),
        [Some(EXACT_STAGE_TAU), Some(1e-8), None, None],
        "{:?} {lines:#?}",
        out.solves
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.contains("skipping the cold exact product")),
        "{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("exact product solved the model") && l.contains("1.000000e0")),
        "the relaxed path's answer is the near branch: {lines:#?}"
    );
    assert!(
        out.x[0].abs() < 1e-5 && (out.x[1] - 1.0).abs() < 1e-5,
        "{:?} {lines:#?}",
        out.x
    );
}

/// MacMPEC `scholtes4` with its objective multiplied by `c`:
///
/// ```text
/// min c·(z1 + z2 - z3)   s.t.  z3 - 4 z1 <= 0,  z3 - 4 z2 <= 0,
///                              0 <= z1 ⟂ z2 >= 0
/// ```
///
/// `f* = 0` at the origin, a biactive corner. Relaxed to `z1·z2 <= τ` the
/// optimum is `z1 = z2 = sqrt(τ)`, `z3 = 4·sqrt(τ)`, `f = -2c·sqrt(τ)`: below
/// `f*` by an amount that scales with `c`.
fn scholtes4(c: Number) -> Dense {
    let zero = || vec![vec![0.0; 3]; 3];
    Dense {
        n: 3,
        m: 3,
        x_l: vec![0.0, 0.0, -INF],
        x_u: vec![INF, INF, INF],
        g_l: vec![-INF, -INF, 0.0],
        g_u: vec![0.0, 0.0, INF],
        x0: vec![1.0, 1.0, 0.0],
        f: Box::new(move |x| c * (x[0] + x[1] - x[2])),
        grad: Box::new(move |_| vec![c, c, -c]),
        g: Box::new(|x| vec![x[2] - 4.0 * x[0], x[2] - 4.0 * x[1], x[1]]),
        jac: Box::new(|_| {
            vec![
                vec![-4.0, 0.0, 1.0],
                vec![0.0, -4.0, 1.0],
                vec![0.0, 1.0, 0.0],
            ]
        }),
        hess: Box::new(move |_| vec![zero(), zero(), zero(), zero()]),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

/// Two branches, two objectives:
///
/// ```text
/// min (x - 0.9)^2 + (y - 1)^2   s.t.  0 <= x ⟂ y >= 0
/// ```
///
/// `(0, 1)` at 0.81 is the optimum; `(0.9, 0)` at 1.0 is an S-stationary
/// local minimum.
fn two_branch(x0: [Number; 2]) -> Dense {
    let zero = || vec![vec![0.0; 2]; 2];
    Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![0.0],
        g_u: vec![INF],
        x0: x0.to_vec(),
        f: Box::new(|x| (x[0] - 0.9).powi(2) + (x[1] - 1.0).powi(2)),
        grad: Box::new(|x| vec![2.0 * (x[0] - 0.9), 2.0 * (x[1] - 1.0)]),
        g: Box::new(|x| vec![x[1]]),
        jac: Box::new(|_| vec![vec![0.0, 1.0]]),
        hess: Box::new(move |_| {
            let mut f = zero();
            f[0][0] = 2.0;
            f[1][1] = 2.0;
            vec![f, zero()]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

/// A corner where the relaxation buys far more than its first-order value:
///
/// ```text
/// min -z3   s.t.  z3^k <= z1·z2,   0 <= z1 ⟂ z2 >= 0,   z in [0, 1]^3
/// ```
///
/// The exact model forces `z3 = 0`. Relaxed to `z1·z2 <= τ`, `z3 = τ^(1/k)`
/// and `f = -τ^(1/k)`, so the product row's multiplier is `τ^(1/k - 1)/k` and
/// the first-order value of the slack is `τ^(1/k)/k`: `1/k` of what it
/// actually buys. At `k = 20`, `τ = 1e-8` that is `0.02` against `0.398`.
fn concave_corner(k: i32) -> Dense {
    let kf = k as Number;
    Dense {
        n: 3,
        m: 2,
        x_l: vec![0.0, 0.0, 0.0],
        x_u: vec![1.0, 1.0, 1.0],
        g_l: vec![-INF, 0.0],
        g_u: vec![0.0, INF],
        x0: vec![0.5, 0.5, 0.5],
        f: Box::new(|x| -x[2]),
        grad: Box::new(|_| vec![0.0, 0.0, -1.0]),
        g: Box::new(move |x| vec![x[2].powi(k) - x[0] * x[1], x[1]]),
        jac: Box::new(move |x| {
            vec![
                vec![-x[1], -x[0], kf * x[2].powi(k - 1)],
                vec![0.0, 1.0, 0.0],
            ]
        }),
        hess: Box::new(move |x| {
            let z = vec![vec![0.0; 3]; 3];
            let mut h0 = z.clone();
            h0[0][1] = -1.0;
            h0[1][0] = -1.0;
            h0[2][2] = kf * (kf - 1.0) * x[2].powi(k - 2);
            vec![z.clone(), h0, z]
        }),
        finalized: Rc::new(RefCell::new(Vec::new())),
    }
}

fn run(
    model: Dense,
    pairs: Vec<ComplementarityPair>,
    max_iter: Option<Index>,
) -> (MpccOutcome, Vec<String>) {
    run_tnlp(model, pairs, max_iter)
}

fn run_tnlp<T: TNLP + 'static>(
    model: T,
    pairs: Vec<ComplementarityPair>,
    max_iter: Option<Index>,
) -> (MpccOutcome, Vec<String>) {
    let mut app = app();
    if let Some(mi) = max_iter {
        app.options_mut()
            .set_integer_value("max_iter", mi, true, false)
            .unwrap();
    }
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(model));
    let mut lines = Vec::new();
    let out = solve_mpcc(&mut app, tnlp, pairs, &mut |l| lines.push(l.to_string())).unwrap();
    (out, lines)
}

/// gh#1128: the finish-drift test must not depend on the objective's units.
/// On `scholtes4` the continuation's relaxed point sits at `-2c·sqrt(τ)` and
/// its exact finish reaches the corner at `0`, which is the answer. The old
/// allowance, `10·sqrt(p)·max(1, |f|)`, stayed at `1e-3` while the slack
/// grew with `c`: at `c = 10` the finish read as drifted and the relaxed
/// point, below `f*`, was kept. The allowance now comes from the relaxed
/// point's multipliers, which scale with `c`.
#[test]
fn the_finish_drift_verdict_does_not_change_with_the_objectives_scale() {
    for c in [0.1, 1.0, 10.0, 1e3] {
        let (out, lines) = run(
            scholtes4(c),
            vec![ComplementarityPair { row: 2, var: 0 }],
            None,
        );
        assert!(
            !lines.iter().any(|l| l.contains("drifted")),
            "scholtes4 x{c}: {lines:#?}"
        );
        assert_eq!(
            out.reported,
            Reported::WeakFallback(MpccStage::Continuation),
            "scholtes4 x{c}: {lines:#?}"
        );
        assert!(
            out.x.iter().all(|v| v.abs() < 1e-6),
            "scholtes4 x{c}: the finish (the origin) is reported, not the relaxed point: {:?} {lines:#?}",
            out.x
        );
    }
}

/// The other branch of the same test: a finish worse than its relaxed start
/// by more than the allowance is not taken, and the relaxed point -- every
/// product within its `τ` -- stands in for it. On `concave_corner(20)` the
/// relaxation at `τ = 1e-8` buys `0.398` against a first-order value of
/// `0.02`, so both the continuation's finish and the relaxed path's read as
/// drifted and the continuation's relaxed point (`z3 = τ^(1/20)`) is
/// reported.
#[test]
fn a_finish_that_drifts_past_the_allowance_leaves_the_relaxed_point() {
    let (out, lines) = run(
        concave_corner(20),
        vec![ComplementarityPair { row: 1, var: 0 }],
        None,
    );
    for label in ["Scholtes continuation", "exact product"] {
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("{label}'s exact finish drifted"))),
            "{label}: {lines:#?}"
        );
    }
    assert_eq!(
        out.reported,
        Reported::FeasibleUncertified(MpccStage::Continuation)
    );
    assert!(
        (out.x[2] - (1e-8f64).powf(0.05)).abs() < 1e-6,
        "the relaxed point at tau = 1e-8: {:?} {lines:#?}",
        out.x
    );
    assert!(
        out.max_abs_product <= 1e-8 * (1.0 + 1e-6),
        "{}",
        out.max_abs_product
    );
}

/// `Dense` recording `mu` at every solve's iteration 0 (the barrier the solve
/// started from).
struct MuAtStart {
    inner: Dense,
    mu0: Rc<RefCell<Vec<Number>>>,
}

impl TNLP for MuAtStart {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        self.inner.get_nlp_info()
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        self.inner.get_bounds_info(b)
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        self.inner.get_starting_point(sp)
    }
    fn eval_f(&mut self, x: &[Number], n: bool) -> Option<Number> {
        self.inner.eval_f(x, n)
    }
    fn eval_grad_f(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
        self.inner.eval_grad_f(x, n, g)
    }
    fn eval_g(&mut self, x: &[Number], n: bool, g: &mut [Number]) -> bool {
        self.inner.eval_g(x, n, g)
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, n: bool, mode: SparsityRequest<'_>) -> bool {
        self.inner.eval_jac_g(x, n, mode)
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        n: bool,
        s: Number,
        lam: Option<&[Number]>,
        nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        self.inner.eval_h(x, n, s, lam, nl, mode)
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, d: &IpoptData, q: &IpoptCq) {
        self.inner.finalize_solution(sol, d, q)
    }
    fn wants_intermediate_callback(&self) -> bool {
        true
    }
    fn intermediate_callback(&mut self, st: IterStats, _: &IpoptData, _: &IpoptCq) -> bool {
        if st.iter == 0 {
            self.mu0.borrow_mut().push(st.mu);
        }
        true
    }
}

/// A continuation rung whose warm-started solve fails is retried once from
/// the previous rung's `x` alone, at the barrier that rung ended on rather
/// than `mu_init`'s default `0.1`, on a budget of twice its iterations (at
/// least `pounce_algorithm::mpcc::RUNG_RETRY_FLOOR`, never above the caller's `max_iter`). On
/// `concave_corner(2)` under `max_iter = 12` the warm `τ = 1e-2` rung runs
/// out of iterations and the retry converges in 7 (the same holds for every
/// `max_iter` from 10 to 15), after which the schedule moves on to `1e-3`.
/// The retry's failing arm -- the schedule ending at the last converged
/// rung -- is reached by `scholtes4` under `max_iter = 6`, below.
#[test]
fn a_failed_warm_rung_is_retried_from_its_primal_point_at_its_barrier() {
    let mu0 = Rc::new(RefCell::new(Vec::new()));
    let model = MuAtStart {
        inner: concave_corner(2),
        mu0: Rc::clone(&mu0),
    };
    let (out, lines) = run_tnlp(
        model,
        vec![ComplementarityPair { row: 1, var: 0 }],
        Some(12),
    );
    let mu0 = mu0.borrow();
    assert_eq!(
        mu0.len(),
        out.solves.len(),
        "one iteration 0 per solve: {:?}",
        out.solves
    );
    let cont: Vec<(usize, &_)> = out
        .solves
        .iter()
        .enumerate()
        .filter(|(_, s)| s.stage == MpccStage::Continuation)
        .collect();
    let k = cont
        .windows(2)
        .position(|w| w[0].1.tau == w[1].1.tau && w[0].1.tau.is_some())
        .unwrap_or_else(|| panic!("no rung was retried: {:?} {lines:#?}", out.solves));
    let ((_, failed), (retry_idx, retry)) = (cont[k], cont[k + 1]);
    assert_eq!(failed.tau, Some(1e-2), "{:?}", out.solves);
    assert_eq!(
        failed.status,
        ApplicationReturnStatus::MaximumIterationsExceeded
    );
    assert!(
        matches!(
            retry.status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{:?}",
        out.solves
    );
    // The barrier the previous converged rung ended on, not `mu_init`'s 0.1.
    assert!(
        mu0[retry_idx] < 1e-2,
        "retry started at mu = {}",
        mu0[retry_idx]
    );
    // The schedule goes on past the retried rung: the next `τ` runs, warm
    // from the retry.
    let taus: Vec<_> = cont.iter().map(|(_, s)| s.tau).collect();
    assert_eq!(taus.get(k + 2), Some(&Some(1e-3)), "{taus:?}");
}

/// The retry's other arm: on `scholtes4` under `max_iter = 6` the warm
/// `τ = 1e-5` rung and its retry both run out of iterations, and the
/// schedule ends there -- the finish starts from the last converged rung,
/// `τ = 1e-4`, and no later rung runs.
#[test]
fn a_failed_rung_retry_ends_the_schedule() {
    let (out, lines) = run(
        scholtes4(1.0),
        vec![ComplementarityPair { row: 2, var: 0 }],
        Some(6),
    );
    let taus: Vec<_> = out
        .solves
        .iter()
        .filter(|s| s.stage == MpccStage::Continuation)
        .map(|s| (s.tau, s.status))
        .collect();
    let mi = ApplicationReturnStatus::MaximumIterationsExceeded;
    let k = taus
        .iter()
        .position(|t| t.0 == Some(1e-5))
        .unwrap_or_else(|| panic!("{taus:?} {lines:#?}"));
    assert_eq!(
        &taus[k..k + 2],
        &[(Some(1e-5), mi), (Some(1e-5), mi)],
        "{taus:?}"
    );
    assert_eq!(
        taus[k + 2].0,
        None,
        "the finish follows the failed retry: {taus:?}"
    );
    assert_eq!(taus.len(), k + 3, "{taus:?}");
}
