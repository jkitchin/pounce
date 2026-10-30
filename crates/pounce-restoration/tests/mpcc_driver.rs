//! End-to-end tests of `pounce_restoration::mpcc_driver::solve_mpcc`: real
//! solves through `IpoptApplication`, on models whose answers are known in
//! closed form.

// Test-only shapes: a dense model spelled out loop by loop reads closer to
// the math than iterator chains, and the boxed callbacks are the model.
#![allow(clippy::type_complexity, clippy::needless_range_loop)]

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory_with_sink, feral_config_from_options,
};
use pounce_algorithm::mpcc::{MpccStage, Reported, StationarityClass, TOUCHED_OPTIONS};
use pounce_common::types::{Index, Number};
use pounce_nlp::mpcc::ComplementarityPair;
use pounce_nlp::return_codes::ApplicationReturnStatus;
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
/// local minimum. From `(0.9, 0.01)` the exact product commits to the near
/// branch and stops at the local minimum, while the relaxation sees both.
/// Both stages run, and the lower objective is the answer whichever stage
/// produced it.
#[test]
fn the_continuation_and_the_exact_product_both_run_and_the_lower_objective_wins() {
    let zero = || vec![vec![0.0; 2]; 2];
    let model = Dense {
        n: 2,
        m: 1,
        x_l: vec![0.0, -INF],
        x_u: vec![INF, INF],
        g_l: vec![0.0],
        g_u: vec![INF],
        x0: vec![0.9, 0.01],
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
    };
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
    assert_eq!(solved.len(), 2, "both stages answer: {lines:#?}");
    assert!(
        out.x[0].abs() < 1e-5 && (out.x[1] - 1.0).abs() < 1e-5,
        "{:?} {lines:#?}",
        out.x
    );
    assert_eq!(out.certificate.unwrap().class, StationarityClass::S);
    assert!(out.solves.iter().all(|s| s.stage != MpccStage::Penalty));
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
/// refused stage's empty point. The exact stage must fall back to
/// `G·H <= 0`, and its verdict — infeasible — is what is reported.
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
    assert_eq!(
        exact.first().map(|s| s.status),
        Some(ApplicationReturnStatus::NotEnoughDegreesOfFreedom),
        "{:?}",
        out.solves
    );
    assert_eq!(exact.len(), 2, "the G·H <= 0 retry: {:?}", out.solves);
    // The refused solve ran after the continuation and has no point of its
    // own: it must not report the continuation's last point (PR review:
    // `finalize_solution` never runs for a solve refused before iteration 0,
    // so an uncleared capture carried the previous stage's x and products).
    assert!(
        exact[0].max_abs_product.is_infinite(),
        "the refused solve reports a point: {:?}",
        out.solves
    );
    assert_eq!(out.status, exact[1].status);
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
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(qpec_small(
        [0.0, 0.5, 0.5],
        Rc::new(RefCell::new(Vec::new())),
    )));
    // x0 is bounded above (two-sided): not a pair this lowering handles.
    let e = solve_mpcc(
        &mut app,
        tnlp,
        vec![ComplementarityPair { row: 0, var: 0 }],
        &mut |_| {},
    )
    .err()
    .unwrap();
    assert!(e.contains("one-sided"), "{e}");
}
