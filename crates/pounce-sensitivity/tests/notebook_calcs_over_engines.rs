//! The statistics and the steps of notebooks 26, 31 and 36 over an engine
//! outside pounce.
//!
//! The models are in `fixtures/notebook_calcs/`, written by `models.py`
//! there, which says what each one is for. Three carry the notebooks'
//! calculations:
//!
//! - `decay`, the residual-variable fit of notebook 26
//! - `bounded`, a fit with a parameter held at its bound and a binding
//!   limit row
//! - `parametric`, a model with two parameters pinned by equality rows, a
//!   variable at a kink and an inactive limit row that a step makes active,
//!   the cases of notebook 36
//!
//! Each calculation on them runs three times: over pounce's own solve, over
//! the engine of `tests/common/outside_engine.rs` built from what pounce's
//! solve returned to its caller, and over the same engine built from Ipopt
//! 3.13.2's `.sol` with MA57, both solvers with `bound_relax_factor = 0`.
//! Each run is compared with pounce's, everything discrete exactly and every
//! value to the accuracy given at the comparison, and with the closed form
//! where the model has one.
//!
//! Five more models reach cases the notebooks do not, over pounce's solve
//! and the engine built from it: `coupled`, `unidentified`, `held`, `concave`
//! and `kinkrow`.
//!
//! The `.sol` files are committed, so this runs anywhere. With
//! `POUNCE_TEST_IPOPT` naming an Ipopt that has MA57,
//! `ipopt_still_produces_the_committed_results` re-solves each `.nl` that has
//! one and checks the committed files are still its answer.

use std::cell::RefCell;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nl::nl_reader::{NlTnlp, read_nl_file};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_sens_core::backsolver::SensBacksolver;
use pounce_sens_core::barrier::BarrierIterate;
use pounce_sens_core::converged::ConvergedSolve;
use pounce_sens_core::dense::{Dense, sym_eigen};
use pounce_sens_core::model::{Declarations, ModelFunctions, SolveRecord};
use pounce_sens_core::stats::{self, HessianForm};
use pounce_sens_core::step::{self, Degeneracy, Mode, StepOptions};
use pounce_sensitivity::Solver;

mod common;
use common::compare::RATIO_RTOL;
use common::model_functions::TnlpModel;
use common::outside_engine::{
    Bounds, Fault, Layout, OutsideEngine, Relaxation, outside_engine, outside_engine_with_layout,
    recorded_solve,
};
use common::sol_file::{SolFile, ipopt_final_mu, ipopt_objective_scaling};

const DECAY: &str = "decay";
const BOUNDED: &str = "bounded";
const PARAMETRIC: &str = "parametric";

/// The runs each comparison names, in the order [`over_engines`] returns
/// them.
const RUNS: [&str; ENGINES] = [
    "pounce",
    "pounce's iterate, outside engine",
    "Ipopt",
    "pounce's iterate, outside engine, blocks reversed",
];

/// How many runs [`over_engines`] makes.
const ENGINES: usize = 4;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/notebook_calcs")
        .join(name)
}

fn read_fixture(name: &str) -> String {
    std::fs::read_to_string(fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn nl_model(stem: &str) -> NlTnlp {
    NlTnlp::new(read_nl_file(&fixture(&format!("{stem}.nl"))).expect("the fixture .nl parses"))
}

fn model(stem: &str) -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(nl_model(stem)))
}

/// One model solved by pounce, and the two outside engines over it.
struct Solved {
    stem: &'static str,
    pounce: Solver,
    recorded: OutsideEngine,
    ipopt: OutsideEngine,
    /// pounce's iterate again, with every block in the reverse of the
    /// model's order, so the maps are all that relates engine positions to
    /// model indices.
    reversed: OutsideEngine,
}

fn solved(stem: &'static str) -> Solved {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model(stem));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{stem}: pounce's solve: {st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let reversed = outside_engine_with_layout(&recording, Relaxation::None, Layout::Reversed);

    let out = read_fixture(&format!("{stem}.ipopt.txt"));
    let mu = ipopt_final_mu(&out).expect("Ipopt printed its barrier parameter");
    let df = ipopt_objective_scaling(&out).expect("Ipopt printed its objective scaling");
    let reported = SolFile::parse(&read_fixture(&format!("{stem}.sol")))
        .expect("the .sol parses")
        .ipopt_iterate(mu);
    let m = model(stem);
    let declared = Bounds::declared(&mut *m.borrow_mut());
    let ipopt = OutsideEngine::with_objective_factor(m, &reported, &declared, &declared, df);
    Solved {
        stem,
        pounce,
        recorded,
        ipopt,
        reversed,
    }
}

/// `$body` evaluated over each run's converged solve, bound to `$conv`, and
/// the point that solve returned, bound to `$x`. No variable is fixed in
/// these models, so the engines' `x` blocks are the model's variables.
macro_rules! over_engines {
    ($s:expr, |$conv:ident, $x:ident| $body:expr) => {{
        let s: &Solved = &$s;
        let a = {
            let $x: Vec<Number> = s.pounce.converged().expect("converged").x.clone();
            s.pounce
                .with_converged(|$conv| Ok($body))
                .expect("pounce converged")
        };
        let b = {
            let $x: Vec<Number> = s.recorded.converged()[..s.recorded.offsets()[1]].to_vec();
            let $conv = &ConvergedSolve::new(&s.recorded, 0.0, true);
            $body
        };
        let c = {
            let $x: Vec<Number> = s.ipopt.converged()[..s.ipopt.offsets()[1]].to_vec();
            let $conv = &ConvergedSolve::new(&s.ipopt, 0.0, true);
            $body
        };
        let d = {
            let $x: Vec<Number> = s.reversed.model_x();
            let $conv = &ConvergedSolve::new(&s.reversed, 0.0, true);
            $body
        };
        [a, b, c, d]
    }};
}

// ---------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------

/// The smallest magnitude [`rel_diff`] divides by, so a quantity that is
/// zero in pounce's run is compared absolutely.
const FLOOR: Number = 1e-10;

/// The largest difference between `a` and `b` relative to `a`'s largest
/// magnitude, the measure every value comparison below is held to.
fn rel_diff(a: &[Number], b: &[Number]) -> Number {
    assert_eq!(a.len(), b.len(), "lengths");
    let scale = a.iter().fold(FLOOR, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()))
        / scale
}

fn assert_close(label: &str, a: &[Number], b: &[Number], tol: Number) {
    let d = rel_diff(a, b);
    assert!(
        d <= tol,
        "{label}: differ by {d:e} relative, over {tol:e}\n{a:?}\n{b:?}"
    );
}

fn assert_close_dense(label: &str, a: &Dense, b: &Dense, tol: Number) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols), "{label}: shape");
    assert_close(label, &a.data, &b.data, tol);
}

/// The variant names of `findings`. These have to agree exactly, and the
/// values some findings carry are compared where the finding matters.
fn kinds<T: Debug>(findings: &[T]) -> Vec<String> {
    findings
        .iter()
        .map(|f| {
            let s = format!("{f:?}");
            s.split([' ', '(', '{']).next().unwrap_or("").to_string()
        })
        .collect()
}

/// `v` over the factor's `x` block, put in the model's variable order by
/// the maps the engine reports. A parametric step comes back in the
/// engine's own order, which is the model's only when the layouts agree.
fn in_model_order<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    v: &[Number],
) -> Vec<Number> {
    let maps = conv.engine().model_maps().expect("maps");
    let mut out = vec![0.0; maps.n_full_x];
    for (k, &i) in maps.var_x_to_full_x.iter().enumerate() {
        out[i] = v[k];
    }
    out
}

/// The model equality row whose multiplier sits at KKT row `r`, for a
/// row [`step::multiplier_row`] returns in the engine's own layout.
fn model_row_of<E: BarrierIterate + ?Sized>(conv: &ConvergedSolve<'_, E>, r: usize) -> usize {
    let at = r - conv.engine().offsets()[2];
    let maps = conv.engine().model_maps().expect("maps");
    maps.full_g_to_c
        .iter()
        .position(|&c| c == Some(at))
        .expect("a multiplier row of an equality")
}

/// [`step::multiplier_row`] for model rows `0..3`, with each row found
/// put back as the model row it belongs to.
fn multiplier_rows<E: BarrierIterate + ?Sized>(conv: &ConvergedSolve<'_, E>) -> Vec<String> {
    (0..3)
        .map(
            |g| match step::multiplier_row(conv, g).expect("multiplier row") {
                Ok(r) => format!("Ok(row {})", model_row_of(conv, r)),
                Err(e) => format!("Err({e:?})"),
            },
        )
        .collect()
}

/// Each run's results against pounce's, by `cmp`.
fn against_pounce<T>(what: &str, runs: &[T; ENGINES], cmp: impl Fn(&str, &T, &T)) {
    for (k, r) in runs.iter().enumerate().skip(1) {
        cmp(&format!("{what}, {}", RUNS[k]), &runs[0], r);
    }
}

// ---------------------------------------------------------------------------
// Notebooks 26 and 31: covariance and information
// ---------------------------------------------------------------------------

/// The fit's declarations: the fitted parameters and the residuals, by
/// their positions in the `.col` file.
fn declarations(stem: &str) -> (Vec<usize>, Vec<Vec<usize>>) {
    match stem {
        DECAY => (vec![0, 1], vec![(2..14).collect()]),
        BOUNDED => (vec![0, 1, 14], vec![(2..14).collect()]),
        _ => unreachable!(),
    }
}

/// The objective at `x`, which the noise estimate is built from.
fn objective(stem: &str, x: &[Number]) -> Number {
    nl_model(stem)
        .eval_f(x, true)
        .expect("the objective evaluates")
}

type CovRun = (Vec<String>, Result<stats::Covariance, String>);

fn covariance_runs(
    s: &Solved,
    known: Option<&[Number]>,
    hessian: HessianForm,
) -> [CovRun; ENGINES] {
    let (fit, groups) = declarations(s.stem);
    over_engines!(s, |conv, x| {
        let decl = Declarations {
            pins: &[],
            fit_rows: &fit,
            res_groups: &groups,
        };
        let rec = SolveRecord {
            x: &x,
            obj: objective(s.stem, &x),
            obj_sign: 1.0,
        };
        let out = stats::covariance(conv, decl, rec, None, known, None, hessian, None);
        (
            kinds(&out.findings),
            out.result.map_err(|e| format!("{e:?}")),
        )
    })
}

fn same_covariance(label: &str, a: &CovRun, b: &CovRun, tol: Number) {
    assert_eq!(a.0, b.0, "{label}: findings");
    match (&a.1, &b.1) {
        (Ok(ca), Ok(cb)) => {
            assert_close_dense(&format!("{label}: covariance"), &ca.matrix, &cb.matrix, tol);
            assert_close(
                &format!("{label}: sigma_sq"),
                &ca.sigma_sq,
                &cb.sigma_sq,
                tol,
            );
            let (sa, ra) = stats::correlation(&ca.matrix);
            let (sb, rb) = stats::correlation(&cb.matrix);
            assert_close(&format!("{label}: standard deviations"), &sa, &sb, tol);
            assert_close_dense(&format!("{label}: correlation"), &ra, &rb, tol);
        }
        (ea, eb) => assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{label}: result"),
    }
}

type InfoRun = (Vec<String>, Result<stats::Information, String>);

fn information_runs(
    s: &Solved,
    block: Option<&[usize]>,
    hessian: HessianForm,
) -> [InfoRun; ENGINES] {
    let (fit, groups) = declarations(s.stem);
    over_engines!(s, |conv, _x| {
        let decl = Declarations {
            pins: &[],
            fit_rows: &fit,
            res_groups: &groups,
        };
        let out = stats::information(conv, decl, block, hessian, None);
        (
            kinds(&out.findings),
            out.result.map_err(|e| format!("{e:?}")),
        )
    })
}

fn same_information(label: &str, a: &InfoRun, b: &InfoRun, tol: Number) {
    assert_eq!(a.0, b.0, "{label}: findings");
    match (&a.1, &b.1) {
        (Ok(ia), Ok(ib)) => {
            assert_eq!(
                ia.conditioned_on, ib.conditioned_on,
                "{label}: conditioned on"
            );
            let (ia, ib) = (&ia.matrix, &ib.matrix);
            assert_close_dense(&format!("{label}: information"), ia, ib, tol);
            let (va, ea) = sym_eigen(ia).expect("eigen");
            let (vb, eb) = sym_eigen(ib).expect("eigen");
            assert_close(&format!("{label}: eigenvalues"), &va, &vb, tol);
            assert_close_dense(&format!("{label}: eigenvectors"), &ea, &eb, tol);
        }
        (ea, eb) => assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{label}: result"),
    }
}

/// How closely the statistics agree with pounce's.
///
/// Over both fits the matrices, the noise variances, the eigenvalues and
/// the correlations agree to `2.4e-8`, and the eigenvectors to `3e-16`. The
/// covariance and the information reach the factor through a back-solve
/// for each fitted parameter, and the two engines' factorizations of the
/// same KKT matrix give those to that accuracy.
const STATS: Number = 1e-7;

/// How closely the information of a parameter held at its bound agrees
/// with pounce's: [`RATIO_RTOL`], the accuracy the reduced classification
/// is compared to.
///
/// `k` in `bounded` sits on its lower bound with `Σ = 3.0e7`, and its
/// information is the reduced curvature `1/(K⁻¹)_kk − Σ = 0.111`, a
/// difference that cancels eight digits. The two engines agree on it to
/// `5.3e-5`, over pounce's iterate and over Ipopt's alike, so the
/// difference comes from the factorizations and not from the point.
const HELD: Number = RATIO_RTOL;

fn the_statistics_over_each_engine_are_pounces(stem: &'static str) {
    let s = solved(stem);
    for (form, name) in [
        (HessianForm::Lagrangian, "Lagrangian"),
        (HessianForm::GaussNewton, "Gauss-Newton"),
    ] {
        let runs = covariance_runs(&s, None, form);
        against_pounce(&format!("{stem}: covariance, {name}"), &runs, |l, a, b| {
            same_covariance(l, a, b, STATS)
        });
        let runs = information_runs(&s, None, form);
        against_pounce(&format!("{stem}: information, {name}"), &runs, |l, a, b| {
            same_information(l, a, b, STATS)
        });
    }
    let runs = covariance_runs(&s, Some(&[0.0025]), HessianForm::Lagrangian);
    against_pounce(
        &format!("{stem}: covariance, known noise"),
        &runs,
        |l, a, b| same_covariance(l, a, b, STATS),
    );
    let held = if stem == BOUNDED { HELD } else { STATS };
    let runs = information_runs(&s, Some(&[1]), HessianForm::Lagrangian);
    against_pounce(&format!("{stem}: information of k"), &runs, |l, a, b| {
        same_information(l, a, b, held)
    });
}

#[test]
fn the_statistics_of_the_decay_fit_over_each_engine_are_pounces() {
    the_statistics_over_each_engine_are_pounces(DECAY);
}

#[test]
fn the_statistics_of_the_bounded_fit_over_each_engine_are_pounces() {
    the_statistics_over_each_engine_are_pounces(BOUNDED);
}

// ---------------------------------------------------------------------------
// Notebook 36: the steps
// ---------------------------------------------------------------------------

/// `pin_p` and `pin_q`, the rows holding `p` and `q`.
const PINS: [Index; 2] = [0, 1];

/// The perturbations each step is taken at. `q` growing makes `cap` active.
/// `p` moving down takes `x1` off its upper bound, and `p` moving up presses
/// it into that bound.
const DELTAS: [[Number; 2]; 3] = [[0.05, 0.4], [-0.05, 0.4], [0.0, 0.8]];

const MODES: [Mode; 3] = [Mode::Linear, Mode::FixRelax, Mode::Path];
const DEGENERACIES: [Degeneracy; 3] = [
    Degeneracy::Directional,
    Degeneracy::OneSided,
    Degeneracy::ReleaseAll,
];

/// How closely the steps agree with pounce's.
///
/// The estimates, the step fractions, the crossings and the derivative
/// columns agree to `4.1e-12` relative to their largest entry, under every
/// mode, degeneracy treatment and perturbation below. A step is a few
/// back-solves, the bounded modes and the directional decision through the
/// released solves of each engine, and the factorizations agree to that.
const STEPS: Number = 1e-10;

type StepRun = (Vec<String>, Result<Vec<Number>, String>);

#[test]
fn the_solution_estimates_over_each_engine_are_pounces() {
    let s = solved(PARAMETRIC);
    for mode in MODES {
        for degeneracy in DEGENERACIES {
            for deltas in DELTAS {
                let o = StepOptions {
                    mode,
                    degeneracy,
                    ..StepOptions::default()
                };
                let runs: [StepRun; ENGINES] = over_engines!(s, |conv, x| {
                    let model = TnlpModel::new(model(PARAMETRIC));
                    let out = step::solution(conv, &model, &x, &PINS, &deltas, &o, true);
                    (
                        kinds(&out.findings),
                        out.result.map_err(|e| format!("{e:?}")),
                    )
                });
                let what = format!("solution, {mode:?}, {degeneracy:?}, {deltas:?}");
                against_pounce(&what, &runs, |l, a, b| {
                    assert_eq!(a.0, b.0, "{l}: findings");
                    match (&a.1, &b.1) {
                        (Ok(xa), Ok(xb)) => assert_close(l, xa, xb, STEPS),
                        (ea, eb) => assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{l}"),
                    }
                });
            }
        }
    }
}

#[test]
fn the_solution_reports_over_each_engine_are_pounces() {
    let s = solved(PARAMETRIC);
    for mode in MODES {
        for deltas in DELTAS {
            let o = StepOptions {
                mode,
                ..StepOptions::default()
            };
            let runs = over_engines!(s, |conv, x| {
                let mut model = TnlpModel::new(model(PARAMETRIC));
                let out = step::solution_report(conv, &mut model, &x, &PINS, &deltas, &o, true);
                (
                    kinds(&out.findings),
                    out.result.map_err(|e| format!("{e:?}")),
                )
            });
            let what = format!("solution report, {mode:?}, {deltas:?}");
            against_pounce(&what, &runs, |l, a, b| {
                assert_eq!(a.0, b.0, "{l}: findings");
                let (ra, rb) = match (&a.1, &b.1) {
                    (Ok(ra), Ok(rb)) => (ra, rb),
                    (ea, eb) => {
                        assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{l}");
                        return;
                    }
                };
                assert_close(&format!("{l}: alpha"), &[ra.alpha], &[rb.alpha], STEPS);
                assert_eq!(ra.first, rb.first, "{l}: first");
                let idx = |v: &[(usize, Number)]| v.iter().map(|p| p.0).collect::<Vec<_>>();
                let val = |v: &[(usize, Number)]| v.iter().map(|p| p.1).collect::<Vec<_>>();
                assert_eq!(idx(&ra.crossed), idx(&rb.crossed), "{l}: crossed");
                assert_close(
                    &format!("{l}: crossed"),
                    &val(&ra.crossed),
                    &val(&rb.crossed),
                    STEPS,
                );
                assert_eq!(
                    idx(&ra.crossed_rows),
                    idx(&rb.crossed_rows),
                    "{l}: crossed rows"
                );
                assert_close(
                    &format!("{l}: crossed rows"),
                    &val(&ra.crossed_rows),
                    &val(&rb.crossed_rows),
                    STEPS,
                );
                // A distance past a bound in the model's own units, zero on
                // one run where roundoff leaves 4.4e-16 on another, so it is
                // compared absolutely.
                let dv = (ra.violation - rb.violation).abs();
                assert!(dv <= STEPS, "{l}: violation differs by {dv:e}");
                assert_eq!(ra.var_status, rb.var_status, "{l}: variable statuses");
                assert_eq!(ra.row_status, rb.row_status, "{l}: row statuses");
                assert_eq!(ra.refined, rb.refined, "{l}: refined");
                assert_eq!(ra.bounds_relaxed, rb.bounds_relaxed, "{l}: bounds relaxed");
                assert_eq!(ra.refine_stop, rb.refine_stop, "{l}: refinement stop");
            });
        }
    }
}

#[test]
fn the_active_set_changes_over_each_engine_are_pounces() {
    let s = solved(PARAMETRIC);
    for degeneracy in DEGENERACIES {
        for deltas in DELTAS {
            let o = StepOptions {
                degeneracy,
                ..StepOptions::default()
            };
            let runs = over_engines!(s, |conv, _x| {
                let out = step::active_set_changes(conv, &PINS, &deltas, &o);
                (
                    kinds(&out.findings),
                    out.result.map_err(|e| format!("{e:?}")),
                )
            });
            let what = format!("active-set changes, {degeneracy:?}, {deltas:?}");
            against_pounce(&what, &runs, |l, a, b| {
                assert_eq!(a.0, b.0, "{l}: findings");
                match (&a.1, &b.1) {
                    (Ok(ca), Ok(cb)) => {
                        let shape = |c: &[step::ActiveSetChange]| {
                            c.iter()
                                .map(|c| (c.target, c.lower, c.reaches))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(shape(ca), shape(cb), "{l}: changes");
                        let frac = |c: &[step::ActiveSetChange]| {
                            c.iter().map(|c| c.fraction).collect::<Vec<_>>()
                        };
                        assert_close(&format!("{l}: fractions"), &frac(ca), &frac(cb), STEPS);
                    }
                    (ea, eb) => assert_eq!(format!("{ea:?}"), format!("{eb:?}"), "{l}"),
                }
            });
        }
    }
}

/// `sens_jacobian`'s calculations: the multiplier row each model row's
/// sensitivity is read from, the step's column for each parameter, and the
/// total derivative of the objective along it.
#[test]
fn the_jacobian_over_each_engine_is_pounces() {
    let s = solved(PARAMETRIC);
    let rows = over_engines!(s, |conv, _x| multiplier_rows(conv));
    against_pounce("multiplier rows", &rows, |l, a, b| assert_eq!(a, b, "{l}"));
    for (k, pin) in PINS.iter().enumerate() {
        let cols = over_engines!(s, |conv, x| {
            let col = conv.parametric_step(&[*pin], &[1.0]).expect("the column");
            let grad = TnlpModel::new(model(PARAMETRIC))
                .objective_gradient(&x)
                .expect("the gradient");
            let total = step::total_objective_derivative(conv, &grad, &col).expect("df/dp");
            (in_model_order(conv, &col), total)
        });
        against_pounce(&format!("column {k}"), &cols, |l, a, b| {
            assert_close(&format!("{l}: dx/dp"), &a.0, &b.0, STEPS);
            assert_close(&format!("{l}: df/dp"), &[a.1], &[b.1], STEPS);
        });
    }
}

// ---------------------------------------------------------------------------
// The answers in closed form
// ---------------------------------------------------------------------------
//
// Every run above shares the calculations, so a defect in one passes the
// comparisons with pounce's run. These hold each run to the answer the model
// has in closed form.

/// How closely a statistic agrees with its closed form, by [`rel_diff`].
///
/// Every comparison below agrees to `4.3e-8` or closer over the three runs.
/// The largest is on pounce's own run of the decay fit, in the sandwich
/// covariance of two residual groups.
const EXACT: Number = 1e-7;

/// The twelve sample times of both fits, `numpy.linspace(0, 3, 12)`.
fn times() -> Vec<Number> {
    (0..12).map(|i| 3.0 * i as Number / 11.0).collect()
}

/// The residual Jacobian `J` of `r = y − a exp(−k t) − b` over the
/// parameters given by position, `0` for `a`, `1` for `k` and `2` for `b`,
/// and `Σ rᵢ ∇²rᵢ` over them, at amplitude `a` and rate `k`.
fn fit_derivatives(a: Number, k: Number, r: &[Number], params: &[usize]) -> (Dense, Dense) {
    let t = times();
    let n = params.len();
    let jac = Dense::from_fn(12, n, |i, p| {
        let e = (-k * t[i]).exp();
        [-e, a * t[i] * e, -1.0][params[p]]
    });
    let curv = Dense::from_fn(n, n, |p, q| {
        (0..12)
            .map(|i| {
                let e = (-k * t[i]).exp();
                let h = match (params[p], params[q]) {
                    (0, 1) | (1, 0) => t[i] * e,
                    (1, 1) => -a * t[i] * t[i] * e,
                    _ => 0.0,
                };
                r[i] * h
            })
            .sum()
    });
    (jac, curv)
}

fn inv(m: &Dense) -> Dense {
    pounce_sens_core::dense::inv(m).expect("invertible")
}

fn ok<T: Debug, E: Debug>(label: &str, r: Result<T, E>) -> T {
    r.unwrap_or_else(|e| panic!("{label}: {e:?}"))
}

fn declared<'a>(fit: &'a [usize], groups: &'a [Vec<usize>]) -> Declarations<'a> {
    Declarations {
        pins: &[],
        fit_rows: fit,
        res_groups: groups,
    }
}

/// The decay fit's covariance at `x`, with the objective reported as
/// `obj_sign` times the sum of squared residuals there.
#[allow(clippy::too_many_arguments)]
fn decay_covariance<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    x: &[Number],
    groups: &[Vec<usize>],
    known: Option<&[Number]>,
    n_data: Option<Number>,
    obj_sign: Number,
    hessian: HessianForm,
) -> stats::Outcome<stats::Covariance> {
    let ssr: Number = x[2..14].iter().map(|r| r * r).sum();
    let rec = SolveRecord {
        x,
        obj: obj_sign * ssr,
        obj_sign,
    };
    stats::covariance(
        conv,
        declared(&[0, 1], groups),
        rec,
        None,
        known,
        n_data,
        hessian,
        None,
    )
}

/// The decay fit `r = y − A exp(−k t)`, `f = Σ r²`, has `∇²f = 2(JᵀJ +
/// Σ rᵢ ∇²rᵢ)` over `(A, k)`. The Gauss-Newton forms drop the second term,
/// and every noise estimate is a sum of squared residuals over its degrees
/// of freedom.
#[test]
fn the_decay_fits_statistics_are_their_closed_forms() {
    let s = solved(DECAY);
    let (fit, one) = declarations(DECAY);
    let none: Vec<Vec<usize>> = Vec::new();
    let two: Vec<Vec<usize>> = vec![(2..8).collect(), (8..14).collect()];
    let small: Vec<Vec<usize>> = vec![vec![2, 3]];
    let (lag, gn) = (HessianForm::Lagrangian, HessianForm::GaussNewton);
    let runs = over_engines!(s, |conv, x| {
        let cov = |groups, known, n_data, sign, h| {
            decay_covariance(conv, &x, groups, known, n_data, sign, h)
        };
        (
            x.clone(),
            [
                cov(&one, None, None, 1.0, lag),
                cov(&one, None, None, 1.0, gn),
                cov(&one, Some(&[0.0025][..]), None, 1.0, lag),
                cov(&none, None, Some(12.0), 1.0, lag),
                cov(&none, None, Some(12.0), -1.0, lag),
                cov(&one, None, Some(12.0), 1.0, lag),
                cov(&two, None, None, 1.0, lag),
                cov(&two, None, None, 1.0, gn),
                cov(&small, None, None, 1.0, lag),
            ],
            [lag, gn].map(|h| stats::information(conv, declared(&fit, &one), None, h, None)),
        )
    });
    for (k, (x, covs, infos)) in runs.into_iter().enumerate() {
        let run = RUNS[k];
        let r = &x[2..14];
        let (j, c) = fit_derivatives(x[0], x[1], r, &[0, 1]);
        let g = j.t().mul(&j);
        let l = g.add(&c);
        let ssr: Number = r.iter().map(|v| v * v).sum();
        let sigma = ssr / 10.0;
        let [
            lag,
            gn,
            known,
            n_data,
            maximized,
            ignored,
            two_l,
            two_gn,
            small,
        ] = covs;
        let expect =
            |label: &str, out: stats::Outcome<stats::Covariance>, m: &Dense, s2: &[Number]| {
                assert!(
                    out.findings.is_empty(),
                    "{run}: {label}: {:?}",
                    out.findings
                );
                let c = ok(label, out.result);
                assert_close_dense(&format!("{run}: {label}"), m, &c.matrix, EXACT);
                assert_close(&format!("{run}: {label}: sigma_sq"), s2, &c.sigma_sq, EXACT);
            };
        expect("Lagrangian", lag, &inv(&l).scale(sigma), &[sigma]);
        expect("Gauss-Newton", gn, &inv(&g).scale(sigma), &[sigma]);
        expect("known noise", known, &inv(&l).scale(0.0025), &[0.0025]);
        expect("n_data", n_data, &inv(&l).scale(sigma), &[sigma]);
        expect(
            "n_data, maximized",
            maximized,
            &inv(&l).scale(sigma),
            &[sigma],
        );
        // the declared residuals take precedence over n_data, and the
        // covariance reports that it ignored n_data
        assert_eq!(
            ignored.findings,
            vec![stats::StatsFinding::NDataIgnored],
            "{run}"
        );
        assert_close_dense(
            &format!("{run}: n_data ignored"),
            &inv(&l).scale(sigma),
            &ok("n_data ignored", ignored.result).matrix,
            EXACT,
        );
        // two groups of different noise: the sandwich
        let rows = |g: usize| -> Vec<usize> { (6 * g..6 * g + 6).collect() };
        let s_g: Vec<Number> = (0..2)
            .map(|g| rows(g).iter().map(|&i| r[i] * r[i]).sum::<Number>() / 4.0)
            .collect();
        let mut b = Dense::zeros(2, 2);
        for (gi, &sg) in s_g.iter().enumerate() {
            let jg = j.select(&rows(gi), &[0, 1]);
            b = b.add(&jg.t().mul(&jg).scale(sg));
        }
        expect(
            "two groups, Lagrangian",
            two_l,
            &inv(&l).mul(&b).mul(&inv(&l)),
            &s_g,
        );
        expect(
            "two groups, Gauss-Newton",
            two_gn,
            &inv(&g).mul(&b).mul(&inv(&g)),
            &s_g,
        );
        assert!(
            matches!(
                small.result,
                Err(stats::StatsError::GroupTooSmall {
                    group: 0,
                    members: 2,
                    n_fit: 2
                })
            ),
            "{run}: a group no larger than the fit: {:?}",
            small.result
        );
        let [il, ign] = infos;
        for (label, out, m) in [
            ("Lagrangian", il, l.scale(2.0)),
            ("Gauss-Newton", ign, g.scale(2.0)),
        ] {
            assert!(out.findings.is_empty(), "{run}: information, {label}");
            let info = ok(label, out.result).matrix;
            assert_close_dense(&format!("{run}: information, {label}"), &m, &info, EXACT);
            // each eigenvector is one, with its largest component positive
            let (w, v) = sym_eigen(&info).expect("eigen");
            for col in 0..2 {
                let vc: Vec<Number> = (0..2).map(|i| v.at(i, col)).collect();
                let lead = if vc[0].abs() >= vc[1].abs() { 0 } else { 1 };
                assert!(
                    vc[lead] > 0.0,
                    "{run}: eigenvector {col} is not sign-pinned"
                );
                assert_close(
                    &format!("{run}: eigenpair {col}"),
                    &info.mul_vec(&vc),
                    &vc.iter().map(|e| w[col] * e).collect::<Vec<_>>(),
                    EXACT,
                );
            }
        }
    }
}

/// The bounded fit holds `k` at its lower bound and `a + b = 2` binding, so
/// with `b = 2 − a` the fit is linear in `a` alone, with regressor
/// `gᵢ = exp(−k tᵢ) − 1`. Its covariance over `(a, k, b)` is
/// `c [[1, 0, −1], [0, 0, 0], [−1, 0, 1]]` with `c = σ² / Σ gᵢ²`, and its
/// information is the full one projected off `(1, 1)` on the free pair, with
/// `k`'s entry the reduction onto `k`.
#[test]
fn the_bounded_fits_statistics_are_their_closed_forms() {
    let s = solved(BOUNDED);
    let (fit, groups) = declarations(BOUNDED);
    let runs = over_engines!(s, |conv, x| {
        let decl = Declarations {
            pins: &[],
            fit_rows: &fit,
            res_groups: &groups,
        };
        let rec = SolveRecord {
            x: &x,
            obj: objective(BOUNDED, &x),
            obj_sign: 1.0,
        };
        (
            x.clone(),
            [HessianForm::Lagrangian, HessianForm::GaussNewton]
                .map(|h| stats::covariance(conv, decl, rec, None, None, None, h, None)),
            [HessianForm::Lagrangian, HessianForm::GaussNewton]
                .map(|h| stats::information(conv, decl, None, h, None)),
        )
    });
    let conditioned = over_engines!(s, |conv, _x| stats::conditioned_on(conv, &[0])
        .map_err(|e| format!("{e:?}")));
    for (k, c) in conditioned.into_iter().enumerate() {
        assert_eq!(
            c,
            Ok(vec![1]),
            "{}: a statistic of a is conditioned on k",
            RUNS[k]
        );
    }
    for (k, (x, covs, infos)) in runs.into_iter().enumerate() {
        let run = RUNS[k];
        let (a, kk) = (x[0], x[1]);
        let r = &x[2..14];
        let t = times();
        let gsum: Number = t.iter().map(|ti| ((-kk * ti).exp() - 1.0).powi(2)).sum();
        let sigma = r.iter().map(|v| v * v).sum::<Number>() / 9.0;
        let c = sigma / gsum;
        let want = Dense::from_fn(3, 3, |i, j| match (i, j) {
            (0, 0) | (2, 2) => c,
            (0, 2) | (2, 0) => -c,
            _ => 0.0,
        });
        for (label, out) in ["Lagrangian", "Gauss-Newton"].into_iter().zip(covs) {
            let kinds_found = kinds(&out.findings);
            assert_eq!(kinds_found, ["MemberHeld", "RowPins"], "{run}: {label}");
            assert!(
                matches!(
                    out.findings[0],
                    stats::StatsFinding::MemberHeld { member: 1 }
                ),
                "{run}: {label}: {:?}",
                out.findings[0]
            );
            assert!(
                matches!(
                    out.findings[1],
                    stats::StatsFinding::RowPins { row: 12, .. }
                ),
                "{run}: {label}: {:?}",
                out.findings[1]
            );
            let cov = ok(label, out.result);
            assert_close_dense(
                &format!("{run}: covariance, {label}"),
                &want,
                &cov.matrix,
                EXACT,
            );
            assert_close(
                &format!("{run}: sigma_sq, {label}"),
                &[sigma],
                &cov.sigma_sq,
                EXACT,
            );
            let (se, corr) = stats::correlation(&cov.matrix);
            assert_eq!(se[1], 0.0, "{run}: k is held");
            assert!(
                (corr.at(0, 2) + 1.0).abs() < 1e-9,
                "{run}: corr(a, b) {}",
                corr.at(0, 2)
            );
            assert_eq!(
                (corr.at(0, 1), corr.at(1, 1)),
                (0.0, 0.0),
                "{run}: k's correlations"
            );
        }
        let (j, curv) = fit_derivatives(a, kk, r, &[0, 1, 2]);
        let gn = j.t().mul(&j).scale(2.0);
        let lag = gn.add(&curv.scale(2.0));
        let (free, held) = ([0usize, 2], [1usize]);
        let z = Dense::from_fn(2, 1, |i, _| {
            if i == 0 {
                0.5_f64.sqrt()
            } else {
                -(0.5_f64.sqrt())
            }
        });
        for ((label, out), full) in ["Lagrangian", "Gauss-Newton"]
            .into_iter()
            .zip(infos)
            .zip([lag, gn])
        {
            assert_eq!(
                kinds(&out.findings),
                ["MemberHeld", "RowPins"],
                "{run}: information, {label}"
            );
            let info = ok(label, out.result).matrix;
            let rff = full.select(&free, &free);
            let proj = z.mul(&z.t().mul(&rff).mul(&z)).mul(&z.t());
            let s_k = full.select(&held, &held).sub(
                &full
                    .select(&held, &free)
                    .mul(&inv(&rff))
                    .mul(&full.select(&free, &held)),
            );
            let want = Dense::from_fn(3, 3, |i, j| match (i, j) {
                (1, 1) => s_k.at(0, 0),
                (1, _) | (_, 1) => 0.0,
                _ => proj.at(i / 2, j / 2),
            });
            assert_close_dense(&format!("{run}: information, {label}"), &want, &info, EXACT);
        }
    }
}

// ---------------------------------------------------------------------------
// The steps' answers
// ---------------------------------------------------------------------------

/// What the steps answer on `parametric`, checked against their definitions.
/// The linear step is the base point plus the parametric step of the
/// treatment asked for, clamping is reported, the bounded modes stay on the
/// bounds and reach `cap`, and the report's fraction is where the linear
/// prediction of `cap`'s row reaches its limit.
#[test]
fn the_steps_answer_what_they_are_defined_as() {
    let s = solved(PARAMETRIC);
    let o = |mode, degeneracy| StepOptions {
        mode,
        degeneracy,
        ..StepOptions::default()
    };
    let down = [-0.05, 0.4];
    let runs = over_engines!(s, |conv, x| {
        let functions = TnlpModel::new(model(PARAMETRIC));
        let sol = |deltas: &[Number], o: &StepOptions, clamp| {
            step::solution(conv, &functions, &x, &PINS, deltas, o, clamp)
        };
        let plain = in_model_order(conv, &conv.parametric_step(&PINS, &down).expect("step"));
        let released = in_model_order(
            conv,
            &conv
                .parametric_step_release_all(&PINS, &down)
                .expect("step")
                .0,
        );
        let decided = in_model_order(
            conv,
            &conv
                .parametric_step_directional(&PINS, &down, 16)
                .expect("step")
                .0,
        );
        let mut m = TnlpModel::new(model(PARAMETRIC));
        let report = step::solution_report(
            conv,
            &mut m,
            &x,
            &PINS,
            &[0.0, 0.8],
            &StepOptions {
                degeneracy: Degeneracy::OneSided,
                degeneracy_iter: Some(4),
                bound_eps: Some(1e-6),
                ..StepOptions::default()
            },
            true,
        );
        let mult = multiplier_rows(conv);
        let col = conv
            .parametric_step(&[PINS[0]], &[1.0])
            .expect("the column");
        let col_model = in_model_order(conv, &col);
        let grad = TnlpModel::new(model(PARAMETRIC))
            .objective_gradient(&x)
            .expect("gradient");
        let total = step::total_objective_derivative(conv, &grad, &col).expect("df/dp");
        (
            x.clone(),
            [
                (
                    plain,
                    sol(&down, &o(Mode::Linear, Degeneracy::OneSided), true),
                ),
                (
                    released,
                    sol(&down, &o(Mode::Linear, Degeneracy::ReleaseAll), true),
                ),
                (
                    decided,
                    sol(&down, &o(Mode::Linear, Degeneracy::Directional), true),
                ),
            ],
            [
                sol(&[0.05, 0.4], &o(Mode::Linear, Degeneracy::OneSided), true),
                sol(&[0.05, 0.4], &o(Mode::Linear, Degeneracy::OneSided), false),
                sol(
                    &[0.05, 0.4],
                    &o(Mode::FixRelax, Degeneracy::Directional),
                    true,
                ),
                sol(&[0.0, 0.8], &o(Mode::Path, Degeneracy::Directional), true),
            ],
            in_model_order(
                conv,
                &conv.parametric_step(&PINS, &[0.0, 0.8]).expect("step"),
            ),
            report,
            step::active_set_changes(
                conv,
                &PINS,
                &[0.0, 0.8],
                &o(Mode::Linear, Degeneracy::Directional),
            ),
            (mult, col_model, grad, total),
        )
    });
    for (k, (x, linear, bounded, dx_cap, report, changes, jac)) in runs.into_iter().enumerate() {
        let run = RUNS[k];
        for (label, (dx, out)) in ["one-sided", "release-all", "directional"]
            .into_iter()
            .zip(linear)
        {
            assert!(
                out.findings.is_empty(),
                "{run}: {label}: {:?}",
                out.findings
            );
            let want: Vec<Number> = x.iter().zip(&dx).map(|(a, b)| a + b).collect();
            assert_close(
                &format!("{run}: linear, {label}"),
                &want,
                &ok(label, out.result),
                1e-14,
            );
        }
        let [clamped, unclamped, fix_relax, path] = bounded;
        assert_eq!(
            clamped.findings,
            vec![step::StepFinding::LinearClamped { vars: vec![0] }],
            "{run}"
        );
        assert_eq!(
            ok("clamped", clamped.result)[0],
            1.0,
            "{run}: x1 clamped to its bound"
        );
        assert!(
            unclamped.findings.is_empty(),
            "{run}: {:?}",
            unclamped.findings
        );
        assert!(
            ok("unclamped", unclamped.result)[0] > 1.0 + 1e-6,
            "{run}: x1 past its bound"
        );
        let xf = ok("fix_relax", fix_relax.result);
        assert!(
            fix_relax.findings.is_empty(),
            "{run}: {:?}",
            fix_relax.findings
        );
        assert!(
            xf[0] <= 1.0 + 1e-9,
            "{run}: fix_relax leaves x1 at {}",
            xf[0]
        );
        assert!(
            (xf[1] - 1.24).abs() < 1e-12 && (xf[3] - 0.9).abs() < 1e-12,
            "{run}: the pins moved"
        );
        let xp = ok("path", path.result);
        assert!(
            (xp[2] + xp[4] - 2.3).abs() < 1e-9,
            "{run}: path holds cap at {}",
            xp[2] + xp[4]
        );

        // the report under the linear one-sided step
        assert_eq!(
            kinds(&report.findings),
            ["DegeneracyIterInert", "BoundEpsInert"],
            "{run}"
        );
        let rep = ok("report", report.result);
        let g_base = x[2] + x[4];
        let dg = dx_cap[2] + dx_cap[4];
        assert_eq!(rep.first, Some(step::BoundTarget::Row(2)), "{run}");
        assert_close(
            &format!("{run}: alpha"),
            &[(2.3 - g_base) / dg],
            &[rep.alpha],
            1e-12,
        );
        let over = g_base + dg - 2.3;
        assert_eq!(rep.crossed_rows.len(), 1, "{run}");
        assert_eq!(rep.crossed_rows[0].0, 2, "{run}");
        assert_close(
            &format!("{run}: crossed row"),
            &[over],
            &[rep.crossed_rows[0].1],
            1e-12,
        );
        assert_close(
            &format!("{run}: violation"),
            &[over],
            &[rep.violation],
            1e-12,
        );
        assert!(rep.crossed.is_empty(), "{run}: {:?}", rep.crossed);

        let ch = ok("changes", changes.result);
        let shape: Vec<_> = ch.iter().map(|c| (c.target, c.lower, c.reaches)).collect();
        assert_eq!(
            shape,
            [
                (step::BoundTarget::Variable(0), false, false),
                (step::BoundTarget::Row(2), false, true)
            ],
            "{run}: x1 leaves its upper bound and cap reaches its limit"
        );

        let (mult, col, grad, total) = jac;
        assert_eq!(
            mult,
            [
                "Ok(row 0)".to_string(),
                "Ok(row 1)".to_string(),
                "Err(NotStrictlyActive(0))".to_string()
            ],
            "{run}: the pins' rows, and cap inactive"
        );
        let want: Number = grad.iter().zip(&col).map(|(a, b)| a * b).sum();
        assert_close(&format!("{run}: df/dp"), &[want], &[total], 1e-14);
    }
}

type CoupledReports = (i8, step::SolutionReport, step::SolutionReport);

/// The classifier's call on `k`, and the report with the refinement and
/// without it.
fn coupled_reports<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    x: &[Number],
) -> CoupledReports {
    let o = StepOptions::default();
    let report = |refine| {
        let mut m = TnlpModel::new(model("coupled"));
        ok(
            "report",
            step::solution_report(conv, &mut m, x, &[0], &[1e-3], &o, refine).result,
        )
    };
    let act = conv.classify_activity().expect("classification");
    (act.var_status[0], report(true), report(false))
}

/// `coupled`'s `k` is a kink the classifier calls ambiguous, and the
/// report's refinement classifies it as weakly active by the reduced
/// curvature, at one back-solve for `k` and none for anything else. Without
/// the refinement the report keeps the classifier's call. A statistic of `y`
/// alone is not conditioned on `k`'s bound, which only a held bound would
/// be.
#[test]
fn the_report_refines_a_coupled_kink() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("tol", 1e-10, true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model("coupled"));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let x = pounce.converged().expect("converged").x.clone();
    let a = pounce
        .with_converged(|c| Ok(coupled_reports(c, &x)))
        .expect("pounce converged");
    let b = coupled_reports(&ConvergedSolve::new(&recorded, 0.0, true), &x);
    // `k` is weakly active, not held, so a statistic of `y` alone is not
    // conditioned on its bound
    let conditioned = pounce
        .with_converged(|c| Ok(stats::conditioned_on(c, &[1])))
        .expect("pounce converged");
    assert_eq!(
        ok("conditioned on", conditioned),
        Vec::<usize>::new(),
        "pounce"
    );
    let conditioned = stats::conditioned_on(&ConvergedSolve::new(&recorded, 0.0, true), &[1]);
    assert_eq!(
        ok("conditioned on", conditioned),
        Vec::<usize>::new(),
        "{}",
        RUNS[1]
    );
    for (run, (cheap, refined, plain)) in [("pounce", a), (RUNS[1], b)] {
        use pounce_sens_core::activity::{AMBIGUOUS, WEAKLY_ACTIVE};
        assert_eq!(
            cheap, AMBIGUOUS,
            "{run}: the fixture must reach the refinement"
        );
        assert_eq!(refined.var_status[0], WEAKLY_ACTIVE, "{run}");
        assert_eq!(
            refined.refined,
            [(step::BoundTarget::Variable(0), AMBIGUOUS, WEAKLY_ACTIVE)],
            "{run}"
        );
        assert_eq!(
            refined.refined_entries,
            [step::BoundTarget::Variable(0)],
            "{run}: only the ambiguous entry is classified again"
        );
        assert_eq!(plain.var_status[0], AMBIGUOUS, "{run}");
        assert!(
            plain.refined.is_empty() && plain.refined_entries.is_empty(),
            "{run}"
        );
        assert_eq!(refined.alpha, plain.alpha, "{run}: the same step otherwise");
    }
}

// ---------------------------------------------------------------------------
// The cases the notebooks do not reach
// ---------------------------------------------------------------------------

/// `unidentified`'s `u` has curvature far below the noise scale and a
/// barrier weight `2μ/0.3² = 5.6e-8` against a curvature floor of `5.8e-8`,
/// so its ratio is about one. A member with curvature below the floor is
/// unidentified only while its barrier weight is below scale too, at a ratio
/// of at most ten. Above that ratio its bound holds it, and the ratio bands
/// classify it.
#[test]
fn a_member_with_curvature_below_the_floor_is_unidentified() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model("unidentified"));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let x = pounce.converged().expect("converged").x.clone();
    let a = pounce
        .with_converged(|c| Ok(unidentified_findings(c, &x)))
        .expect("pounce converged");
    let b = unidentified_findings(&ConvergedSolve::new(&recorded, 0.0, true), &x);
    for (run, (cov, info)) in [("pounce", a), (RUNS[1], b)] {
        let want = vec![stats::StatsFinding::MemberUnidentified { member: 2 }];
        assert_eq!(cov, want, "{run}: covariance");
        assert_eq!(info, want, "{run}: information");
    }
}

/// The findings of the covariance and the information of `unidentified`'s
/// three fitted parameters `A`, `k` and `u`.
fn unidentified_findings<E: BarrierIterate + ?Sized>(
    conv: &ConvergedSolve<'_, E>,
    x: &[Number],
) -> (Vec<stats::StatsFinding>, Vec<stats::StatsFinding>) {
    let fit = [0, 1, 14];
    let groups: Vec<Vec<usize>> = vec![(2..14).collect()];
    let decl = declared(&fit, &groups);
    let rec = SolveRecord {
        x,
        obj: objective("unidentified", x),
        obj_sign: 1.0,
    };
    let cov = stats::covariance(
        conv,
        decl,
        rec,
        None,
        None,
        None,
        HessianForm::Lagrangian,
        None,
    );
    let info = stats::information(conv, decl, None, HessianForm::Lagrangian, None);
    ok("covariance", cov.result.map(|_| ()));
    ok("information", info.result.map(|_| ()));
    (cov.findings, info.findings)
}

/// `held` is `bounded` without the limit row, so `k` sits on its lower bound
/// and no row binds. The information of `a` alone is the Schur complement
/// that profiles out `b` and conditions on `k`: `R_aa − R_ab²/R_bb` with
/// `R = 2(JᵀJ + Σ rᵢ ∇²rᵢ)` over `(a, b)` at the bound. Profiling `k` out as
/// well gives `1.44` where the answer is `1.98`.
#[test]
fn a_sub_blocks_information_conditions_on_the_held_members() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model("held"));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let x = pounce.converged().expect("converged").x.clone();
    // the information of `a` and its covariance both hold the bounds
    // `conditioned_on` finds
    fn info<E: BarrierIterate + ?Sized>(c: &ConvergedSolve<'_, E>, x: &[Number]) -> Dense {
        let fit = [0, 1, 14];
        let groups: Vec<Vec<usize>> = vec![(2..14).collect()];
        let info = ok(
            "information of a",
            stats::information(
                c,
                declared(&fit, &groups),
                Some(&[0]),
                HessianForm::Lagrangian,
                None,
            )
            .result,
        );
        let cov = ok(
            "covariance of a",
            stats::covariance(
                c,
                declared(&fit, &groups),
                SolveRecord {
                    x,
                    obj: objective("held", x),
                    obj_sign: 1.0,
                },
                Some(&[0]),
                None,
                None,
                HessianForm::Lagrangian,
                None,
            )
            .result,
        );
        let rows = stats::conditioned_on(c, &[0]).expect("conditioned on");
        assert_eq!(rows, [1], "a is conditional on k's bound");
        assert_eq!(info.conditioned_on, rows, "the information");
        assert_eq!(cov.conditioned_on, rows, "the covariance");
        info.matrix
    }
    let a = pounce
        .with_converged(|c| Ok(info(c, &x)))
        .expect("pounce converged");
    let b = info(&ConvergedSolve::new(&recorded, 0.0, true), &x);
    let (j, curv) = fit_derivatives(x[0], x[1], &x[2..14], &[0, 2]);
    let r = j.t().mul(&j).add(&curv).scale(2.0);
    let want = r.at(0, 0) - r.at(0, 1) * r.at(1, 0) / r.at(1, 1);
    for (run, got) in [("pounce", a), (RUNS[1], b)] {
        assert_close(
            &format!("{run}: information of a"),
            &[want],
            &got.data,
            EXACT,
        );
    }
}

/// A block of `(A, k, r[0])` has one more coordinate than the decay fit has
/// degrees of freedom. Its covariance is the marginal `2σ² M`, which with
/// `r[0]` a function of `(A, k)` is `σ² [[L⁻¹, L⁻¹j], [jᵀL⁻¹, jᵀL⁻¹j]]`, `j`
/// being `r[0]`'s gradient. The Gauss-Newton form has no inverse to profile
/// through, and the information matrix does not exist. Both errors name the
/// count as the reason.
#[test]
fn a_block_larger_than_the_fit_gets_its_marginal() {
    let s = solved(DECAY);
    let (fit, one) = declarations(DECAY);
    let block = [0, 1, 2];
    let runs = over_engines!(s, |conv, x| {
        let ssr: Number = x[2..14].iter().map(|r| r * r).sum();
        let rec = SolveRecord {
            x: &x,
            obj: ssr,
            obj_sign: 1.0,
        };
        let cov = |h| {
            stats::covariance(
                conv,
                declared(&fit, &one),
                rec,
                Some(&block[..]),
                None,
                None,
                h,
                None,
            )
        };
        let info = stats::information(
            conv,
            declared(&fit, &one),
            Some(&block[..]),
            HessianForm::Lagrangian,
            None,
        );
        (
            x.clone(),
            cov(HessianForm::Lagrangian),
            cov(HessianForm::GaussNewton),
            info,
        )
    });
    for (k, (x, lag, gn, info)) in runs.into_iter().enumerate() {
        let run = RUNS[k];
        let r = &x[2..14];
        let (j, c) = fit_derivatives(x[0], x[1], r, &[0, 1]);
        let li = inv(&j.t().mul(&j).add(&c));
        let sigma = r.iter().map(|v| v * v).sum::<Number>() / 10.0;
        let g0 = j.select(&[0], &[0, 1]);
        let want = Dense::from_fn(3, 3, |p, q| {
            let side = |i: usize| -> Dense {
                if i < 2 {
                    Dense::from_fn(2, 1, |a, _| if a == i { 1.0 } else { 0.0 })
                } else {
                    g0.t()
                }
            };
            side(p).t().mul(&li).mul(&side(q)).at(0, 0) * sigma
        });
        assert!(lag.findings.is_empty(), "{run}: {:?}", lag.findings);
        assert_close_dense(
            &format!("{run}: marginal"),
            &want,
            &ok("marginal", lag.result).matrix,
            EXACT,
        );
        assert!(
            matches!(
                gn.result,
                Err(stats::StatsError::DeficientNeedsLagrangian {
                    reason: stats::Deficiency::Count
                })
            ),
            "{run}: {:?}",
            gn.result
        );
        // and it carries no information matrix, for the same reason
        assert!(
            matches!(
                info.result,
                Err(stats::StatsError::InformationDeficient {
                    reason: stats::Deficiency::Count
                })
            ),
            "{run}: {:?}",
            info.result
        );
    }
}

/// `concave`'s `u` has curvature `−2.6e-7` under a barrier weight of
/// `2.6e-5`, so the factor is positive definite and the classification
/// keeps `u` with its barrier weight removed. Its variance is then
/// `2σ²/(−2.6e-7) = −σ²/1.3e-7`, which the covariance returns with a
/// finding that the point is not a least-squares minimum.
#[test]
fn a_negative_variance_is_reported() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model("concave"));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let x = pounce.converged().expect("converged").x.clone();
    let fit = [0, 1, 14];
    let groups: Vec<Vec<usize>> = vec![(2..14).collect()];
    let rec = SolveRecord {
        x: &x,
        obj: objective("concave", &x),
        obj_sign: 1.0,
    };
    let cov = |out: stats::Outcome<stats::Covariance>| (out.findings, ok("covariance", out.result));
    let a = pounce
        .with_converged(|c| {
            Ok(cov(stats::covariance(
                c,
                declared(&fit, &groups),
                rec,
                None,
                None,
                None,
                HessianForm::Lagrangian,
                None,
            )))
        })
        .expect("pounce converged");
    let b = cov(stats::covariance(
        &ConvergedSolve::new(&recorded, 0.0, true),
        declared(&fit, &groups),
        rec,
        None,
        None,
        None,
        HessianForm::Lagrangian,
        None,
    ));
    let sigma = x[2..14].iter().map(|v| v * v).sum::<Number>() / 9.0;
    for (run, (findings, c)) in [("pounce", a), (RUNS[1], b)] {
        assert_eq!(
            findings.last(),
            Some(&stats::StatsFinding::NegativeVariance),
            "{run}: {findings:?}"
        );
        assert_close(
            &format!("{run}: variance of u"),
            &[-sigma / 1.3e-7],
            &[c.matrix.at(2, 2)],
            EXACT,
        );
    }
    // the information is the reduced Hessian itself, with `u`'s entry the
    // curvature `−2.6e-7`. It is indefinite under the Lagrangian form and
    // positive semidefinite under Gauss-Newton, whose residuals ignore `u`
    let info = |conv_info: stats::Outcome<stats::Information>| {
        (
            conv_info.findings,
            ok("information", conv_info.result).matrix,
        )
    };
    for h in [HessianForm::Lagrangian, HessianForm::GaussNewton] {
        let (findings, m) = pounce
            .with_converged(|c| {
                Ok(info(stats::information(
                    c,
                    declared(&fit, &groups),
                    None,
                    h,
                    None,
                )))
            })
            .expect("pounce converged");
        let indefinite = findings.contains(&stats::StatsFinding::Indefinite);
        assert_eq!(
            indefinite,
            h == HessianForm::Lagrangian,
            "{h:?}: {findings:?}"
        );
        if h == HessianForm::Lagrangian {
            assert_close("information of u", &[-2.6e-7], &[m.at(2, 2)], 1e-6);
        }
    }
}

/// Under `fix_relax` at `(0.3, 1.5)`, one corrector iteration takes the
/// residual to `0.83` of where it started and a second to `9e-7`. A
/// correction that does not halve the residual is reported, and one that
/// reduces it by orders is not.
#[test]
fn a_correction_that_does_not_halve_the_residual_is_reported() {
    let s = solved(PARAMETRIC);
    let runs = over_engines!(s, |conv, x| {
        let functions = TnlpModel::new(model(PARAMETRIC));
        [1, 2].map(|corrector_iter| {
            let o = StepOptions {
                mode: Mode::FixRelax,
                corrector_iter,
                ..StepOptions::default()
            };
            kinds(&step::solution(conv, &functions, &x, &PINS, &[0.3, 1.5], &o, true).findings)
        })
    });
    for (k, [one, two]) in runs.into_iter().enumerate() {
        assert!(
            one.contains(&"CorrectorWeak".to_string()),
            "{}: {one:?}",
            RUNS[k]
        );
        assert!(
            !two.contains(&"CorrectorWeak".to_string()),
            "{}: {two:?}",
            RUNS[k]
        );
    }
}

/// `bound_eps` is the margin `fix_relax` pins against, and the report reads
/// it under that mode only. Under the linear step at `(0.05, 0.4)`, `x1`
/// ends past its upper bound, and a margin of one, which would hide that,
/// changes nothing about what the report names.
#[test]
fn the_report_reads_the_margin_under_fix_relax_only() {
    let s = solved(PARAMETRIC);
    let runs = over_engines!(s, |conv, x| {
        let mut functions = TnlpModel::new(model(PARAMETRIC));
        let o = StepOptions {
            degeneracy: Degeneracy::OneSided,
            bound_eps: Some(1.0),
            ..StepOptions::default()
        };
        let rep = step::solution_report(conv, &mut functions, &x, &PINS, &[0.05, 0.4], &o, true);
        ok("report", rep.result).crossed
    });
    for (k, crossed) in runs.into_iter().enumerate() {
        assert_eq!(crossed.len(), 1, "{}: {crossed:?}", RUNS[k]);
        assert_eq!(crossed[0].0, 0, "{}: x1 crosses", RUNS[k]);
        assert!(crossed[0].1 > 1e-6, "{}: {crossed:?}", RUNS[k]);
    }
}

/// A crossing is measured from the bound it crosses. Lowering `p` by `2.5`
/// carries `x1` below its lower bound of zero, and the report gives the
/// distance below it.
#[test]
fn a_crossing_is_measured_from_the_bound_it_crosses() {
    let s = solved(PARAMETRIC);
    let deltas = [-2.5, 0.0];
    let runs = over_engines!(s, |conv, x| {
        let mut functions = TnlpModel::new(model(PARAMETRIC));
        let o = StepOptions {
            degeneracy: Degeneracy::OneSided,
            ..StepOptions::default()
        };
        let rep = step::solution_report(conv, &mut functions, &x, &PINS, &deltas, &o, true);
        let dx = in_model_order(conv, &conv.parametric_step(&PINS, &deltas).expect("step"));
        (x[0] + dx[0], ok("report", rep.result).crossed)
    });
    for (k, (x1, crossed)) in runs.into_iter().enumerate() {
        assert!(x1 < -0.1, "{}: x1 lands at {x1}", RUNS[k]);
        assert_eq!(crossed.len(), 1, "{}: {crossed:?}", RUNS[k]);
        assert_eq!(crossed[0].0, 0, "{}", RUNS[k]);
        assert_close(
            &format!("{}: distance below", RUNS[k]),
            &[-x1],
            &[crossed[0].1],
            1e-14,
        );
    }
}

/// The classifier refuses a solve that relaxed its bounds, so the report
/// says the bounds were relaxed and measures the rest. Under pounce's
/// default `bound_relax_factor` the report comes back with no barrier
/// level and no statuses, over pounce and over the engine built from its
/// relaxed solve.
#[test]
fn the_report_over_a_relaxed_solve_skips_the_classification() {
    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model(PARAMETRIC));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::POUNCE_DEFAULT);
    let x = pounce.converged().expect("converged").x.clone();
    let report = |conv_rep: step::StepOutcome<step::SolutionReport>| ok("report", conv_rep.result);
    let o = StepOptions::default();
    let a = pounce
        .with_converged(|c| {
            let mut functions = TnlpModel::new(model(PARAMETRIC));
            Ok(report(step::solution_report(
                c,
                &mut functions,
                &x,
                &PINS,
                &[0.05, 0.4],
                &o,
                true,
            )))
        })
        .expect("pounce converged");
    let mut functions = TnlpModel::new(model(PARAMETRIC));
    let b = report(step::solution_report(
        &ConvergedSolve::new(&recorded, 1e-8, true),
        &mut functions,
        &x,
        &PINS,
        &[0.05, 0.4],
        &o,
        true,
    ));
    for (run, rep) in [("pounce", a), (RUNS[1], b)] {
        assert!(rep.bounds_relaxed, "{run}");
        assert!(rep.mu.is_nan(), "{run}: mu {}", rep.mu);
        assert!(
            rep.var_status.is_empty() && rep.row_status.is_empty(),
            "{run}"
        );
    }
}

/// An inequality row's multiplier has a derivative to read only where the
/// row is strictly active. `bounded`'s `cap` is, and its multiplier sits in
/// the first row of the `y_d` block. `kinkrow`'s `cap` is weakly active, a
/// kink, and the row is refused with its class.
#[test]
fn an_inequality_rows_multiplier_is_read_only_where_strictly_active() {
    use pounce_sens_core::activity::WEAKLY_ACTIVE;
    let s = solved(BOUNDED);
    let rows = over_engines!(s, |conv, _x| {
        let r = step::multiplier_row(conv, 12).expect("multiplier row");
        (conv.engine().offsets()[3], format!("{r:?}"))
    });
    for (k, (y_d, r)) in rows.into_iter().enumerate() {
        assert_eq!(r, format!("Ok({y_d})"), "{}", RUNS[k]);
    }

    let mut app = IpoptApplication::new();
    let opts = app.options_mut();
    opts.set_integer_value("print_level", 0, true, false)
        .unwrap();
    opts.set_string_value("sb", "yes", true, false).unwrap();
    opts.set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let (pounce, recording, st) = recorded_solve(app, nl_model("kinkrow"));
    assert!(
        matches!(st, ApplicationReturnStatus::SolveSucceeded),
        "{st:?}"
    );
    let recorded = outside_engine(&recording, Relaxation::None);
    let want = format!("Err(NotStrictlyActive({WEAKLY_ACTIVE}))");
    let a = pounce
        .with_converged(|c| Ok(format!("{:?}", step::multiplier_row(c, 1).expect("row"))))
        .expect("pounce converged");
    let b = format!(
        "{:?}",
        step::multiplier_row(&ConvergedSolve::new(&recorded, 0.0, true), 1).expect("row")
    );
    assert_eq!(a, want, "pounce");
    assert_eq!(b, want, "{}", RUNS[1]);
}

/// `ConvergedSolve` checks what an engine reports before computing with
/// it. An engine whose variable map names a variable twice, whose scale
/// factors are one short, or whose primal bounds are one short is refused
/// with an error naming what is wrong, not answered from a neighbouring
/// entry.
#[test]
fn what_an_engine_reports_is_checked() {
    let s = solved(PARAMETRIC);
    let maps = s.recorded.clone().with_fault(Fault::Maps);
    let e = ConvergedSolve::new(&maps, 0.0, true)
        .g_multiplier_rows(&[0])
        .unwrap_err();
    assert!(format!("{e:?}").contains("model maps"), "{e:?}");
    let scaling = s.recorded.clone().with_fault(Fault::Scaling);
    let e = ConvergedSolve::new(&scaling, 0.0, true)
        .nlp_scaling()
        .unwrap_err();
    assert!(format!("{e:?}").contains("scal"), "{e:?}");
    let bounds = s.recorded.clone().with_fault(Fault::PrimalBounds);
    let e = ConvergedSolve::new(&bounds, 0.0, true)
        .parametric_step_bounded(&PINS, &[0.05, 0.4], 16, None)
        .unwrap_err();
    assert!(format!("{e:?}").contains("primal bounds"), "{e:?}");
}

/// A buffer of the wrong length returns an error before anything reads
/// past its end or pairs it with the wrong entries. The buffers are the
/// base point and the perturbations of a step, the gradient and the column
/// of the total derivative, and both buffers of the barrier residual.
#[test]
fn a_buffer_of_the_wrong_length_is_refused() {
    fn shape(r: &Result<impl Debug, impl Debug>, what: &str) -> bool {
        format!("{r:?}").contains(&format!("BadShape {{ what: \"{what}\""))
    }
    let s = solved(PARAMETRIC);
    let runs = over_engines!(s, |conv, x| {
        let o = StepOptions::default();
        let functions = TnlpModel::new(model(PARAMETRIC));
        let short = &x[..x.len() - 1];
        let both = [0.05, 0.4];
        let solution = |x: &[Number], deltas: &[Number]| {
            step::solution(conv, &functions, x, &PINS, deltas, &o, true).result
        };
        let mut m = TnlpModel::new(model(PARAMETRIC));
        let report = step::solution_report(conv, &mut m, short, &PINS, &both, &o, true).result;
        let col = conv
            .parametric_step(&[PINS[0]], &[1.0])
            .expect("the column");
        let grad = TnlpModel::new(model(PARAMETRIC))
            .objective_gradient(&x)
            .expect("the gradient");
        let total =
            |grad: &[Number], col: &[Number]| step::total_objective_derivative(conv, grad, col);
        let dim = conv.engine().dim();
        let residual = |point: usize, out: usize| {
            conv.engine()
                .barrier_residual(&vec![0.0; point], &mut vec![0.0; out])
                .is_err()
        };
        [
            (
                "short base point",
                shape(&solution(short, &both), "base point"),
            ),
            (
                "one perturbation for two pins",
                shape(&solution(&x, &both[..1]), "perturbations"),
            ),
            (
                "short base point in the report",
                shape(&report, "base point"),
            ),
            (
                "short gradient",
                shape(&total(&grad[1..], &col), "objective gradient"),
            ),
            (
                "short column",
                shape(&total(&grad, &col[1..]), "step column"),
            ),
            ("short residual point", residual(dim - 1, dim)),
            ("short residual buffer", residual(dim, dim - 1)),
            (
                "the right lengths",
                total(&grad, &col).is_ok() && !residual(dim, dim),
            ),
        ]
    });
    for (k, checks) in runs.into_iter().enumerate() {
        for (label, holds) in checks {
            assert!(holds, "{}: {label}", RUNS[k]);
        }
    }
}

/// What [`MalformedModel`] gets wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Malformed {
    VariableLower,
    VariableUpper,
    RowLower,
    RowUpper,
    /// one constraint value short at the base point
    BaseConstraints,
    /// one constraint value short at the moved point
    StepConstraints,
    JacobianRow,
    JacobianColumn,
    /// a NaN row at the moved point
    NanAtTheStep,
}

/// The parametric model with one of its functions wrong.
struct MalformedModel {
    inner: TnlpModel,
    fault: Malformed,
    constraint_calls: usize,
}

impl ModelFunctions for MalformedModel {
    fn variable_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let (mut lo, mut hi) = self.inner.variable_bounds();
        match self.fault {
            Malformed::VariableLower => drop(lo.pop()),
            Malformed::VariableUpper => drop(hi.pop()),
            _ => {}
        }
        (lo, hi)
    }

    fn row_bounds(&self) -> (Vec<Number>, Vec<Number>) {
        let (mut lo, mut hi) = self.inner.row_bounds();
        match self.fault {
            Malformed::RowLower => drop(lo.pop()),
            Malformed::RowUpper => drop(hi.pop()),
            _ => {}
        }
        (lo, hi)
    }

    // the report evaluates the base point first and the moved point second
    fn constraints(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        let mut g = self.inner.constraints(x)?;
        self.constraint_calls += 1;
        match (self.fault, self.constraint_calls) {
            (Malformed::BaseConstraints, 1) | (Malformed::StepConstraints, 2) => drop(g.pop()),
            (Malformed::NanAtTheStep, 2) => g[0] = Number::NAN,
            _ => {}
        }
        Ok(g)
    }

    fn jacobian(&mut self, x: &[Number]) -> Result<Vec<(usize, usize, Number)>, String> {
        let mut j = self.inner.jacobian(x)?;
        match self.fault {
            Malformed::JacobianRow => j.push((self.inner.row_bounds().0.len(), 0, 1.0)),
            Malformed::JacobianColumn => j.push((0, x.len(), 1.0)),
            _ => {}
        }
        Ok(j)
    }

    fn objective_gradient(&mut self, x: &[Number]) -> Result<Vec<Number>, String> {
        self.inner.objective_gradient(x)
    }
}

/// A model whose bounds, constraint values or Jacobian do not match its
/// own rows and variables makes the report return an error, and its
/// variable bounds make the solution return one too. A row the model
/// returns as NaN at the moved point makes the violation NaN rather than a
/// number that reads as satisfied.
#[test]
fn a_malformed_model_is_refused_and_a_nan_row_is_reported() {
    let s = solved(PARAMETRIC);
    let runs = over_engines!(s, |conv, x| {
        let o = StepOptions::default();
        let deltas = [0.05, 0.4];
        let malformed = |fault| MalformedModel {
            inner: TnlpModel::new(model(PARAMETRIC)),
            fault,
            constraint_calls: 0,
        };
        let report = |fault| {
            step::solution_report(conv, &mut malformed(fault), &x, &PINS, &deltas, &o, true).result
        };
        let refused = |fault, says: &str| format!("{:?}", report(fault)).contains(says);
        let solution = |fault| {
            format!(
                "{:?}",
                step::solution(conv, &malformed(fault), &x, &PINS, &deltas, &o, true).result
            )
        };
        let violation = report(Malformed::NanAtTheStep).map(|r| r.violation);
        [
            (
                "variable lower bounds in the solution",
                solution(Malformed::VariableLower).contains("variable lower bounds"),
            ),
            (
                "variable upper bounds in the solution",
                solution(Malformed::VariableUpper).contains("variable upper bounds"),
            ),
            (
                "variable lower bounds",
                refused(Malformed::VariableLower, "variable lower bounds"),
            ),
            (
                "variable upper bounds",
                refused(Malformed::VariableUpper, "variable upper bounds"),
            ),
            (
                "row lower bounds",
                refused(Malformed::RowLower, "row lower bounds"),
            ),
            (
                "row upper bounds",
                refused(Malformed::RowUpper, "row upper bounds"),
            ),
            (
                "constraints at the base point",
                refused(Malformed::BaseConstraints, "constraint values for"),
            ),
            (
                "constraints at the moved point",
                refused(Malformed::StepConstraints, "constraint values for"),
            ),
            ("Jacobian row", refused(Malformed::JacobianRow, "outside")),
            (
                "Jacobian column",
                refused(Malformed::JacobianColumn, "outside"),
            ),
            ("NaN row", matches!(violation, Ok(v) if v.is_nan())),
        ]
    });
    for (k, checks) in runs.into_iter().enumerate() {
        for (label, holds) in checks {
            assert!(holds, "{}: {label}", RUNS[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// The fixtures
// ---------------------------------------------------------------------------

/// Re-solves each committed `.nl` with the Ipopt `POUNCE_TEST_IPOPT` names
/// and checks the committed `.sol`, objective scaling and barrier parameter
/// are still its answer, so the fixtures cannot drift from what they say
/// they are.
#[test]
#[ignore = "needs an Ipopt with MA57: set POUNCE_TEST_IPOPT to it and run with --ignored"]
fn ipopt_still_produces_the_committed_results() {
    let ipopt = std::env::var("POUNCE_TEST_IPOPT").expect("POUNCE_TEST_IPOPT names an ipopt");
    for stem in [DECAY, BOUNDED, PARAMETRIC] {
        let dir = std::env::temp_dir().join(format!("pounce-{stem}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let nl = format!("{stem}.nl");
        std::fs::copy(fixture(&nl), dir.join(&nl)).unwrap();
        let run = std::process::Command::new(&ipopt)
            .current_dir(&dir)
            .args([
                nl.as_str(),
                "-AMPL",
                "linear_solver=ma57",
                "bound_relax_factor=0",
                "print_level=8",
            ])
            .output()
            .expect("ipopt runs");
        let printed = String::from_utf8_lossy(&run.stdout).replace('\r', "");
        assert!(
            printed.contains("running with linear solver ma57"),
            "{stem}: Ipopt did not run MA57"
        );
        let committed_out = read_fixture(&format!("{stem}.ipopt.txt"));
        assert_eq!(
            ipopt_final_mu(&printed),
            ipopt_final_mu(&committed_out),
            "{stem}: the barrier parameter moved"
        );
        assert_eq!(
            ipopt_objective_scaling(&printed),
            ipopt_objective_scaling(&committed_out),
            "{stem}: the objective scaling moved"
        );
        let fresh =
            SolFile::parse(&std::fs::read_to_string(dir.join(format!("{stem}.sol"))).unwrap())
                .unwrap();
        let committed = SolFile::parse(&read_fixture(&format!("{stem}.sol"))).unwrap();
        for (what, a, b) in [
            ("primals", &fresh.primals, &committed.primals),
            ("duals", &fresh.duals, &committed.duals),
        ] {
            assert_eq!(a.len(), b.len(), "{stem}: {what}: lengths");
            for (k, (&u, &v)) in a.iter().zip(b).enumerate() {
                assert!(
                    (u - v).abs() <= 1e-12 * u.abs().max(1.0),
                    "{stem}: {what}[{k}]: {u:e} vs {v:e}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// An engine that does not report its inertia correction is not one that
/// applied none. `max_pdpert` cannot be checked against a correction
/// nobody knows, so it refuses there rather than passing, and the report
/// says the correction is unknown rather than zero. pounce reports its
/// correction, so its run takes the cap and reports four numbers.
#[test]
fn an_unreported_inertia_correction_is_neither_zero_nor_within_the_cap() {
    let s = solved(PARAMETRIC);
    let runs = over_engines!(s, |conv, x| {
        let reports = conv.kkt_perturbations().is_some();
        let capped = StepOptions {
            max_pdpert: Some(1.0),
            ..StepOptions::default()
        };
        let deltas = [0.05, 0.4];
        let functions = TnlpModel::new(model(PARAMETRIC));
        let step = format!(
            "{:?}",
            step::solution(conv, &functions, &x, &PINS, &deltas, &capped, true).result
        );
        let mut functions = TnlpModel::new(model(PARAMETRIC));
        let o = StepOptions::default();
        let report = step::solution_report(conv, &mut functions, &x, &PINS, &deltas, &o, true)
            .result
            .expect("an uncapped report");
        (reports, step, report.perturbations)
    });
    for (k, (reports, step, perturbations)) in runs.into_iter().enumerate() {
        let l = RUNS[k];
        assert_eq!(perturbations.is_some(), reports, "{l}: report");
        if reports {
            assert!(
                step.starts_with("Ok"),
                "{l}: a reported zero correction is within the cap: {step}"
            );
        } else {
            assert!(
                step.contains("max_pdpert") && step.contains("does not report"),
                "{l}: the cap is refused, not passed: {step}"
            );
        }
    }
}
