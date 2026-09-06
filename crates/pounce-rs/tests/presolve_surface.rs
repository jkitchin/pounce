//! The presolve and restoration surfaces through the facade only (gh#561).
//!
//! Both modules are re-exports, and a re-export has exactly two failure modes:
//! it stops resolving, or it keeps resolving and means something else. Neither
//! shows up in `pounce-presolve`'s own tests, which import from the crate
//! directly. Everything below therefore goes through `pounce_rs::presolve` /
//! `pounce_rs::restoration` paths, so a rename upstream breaks this file
//! rather than breaking a downstream user silently.
//!
//! Two of these are regression pins rather than compile checks:
//!
//! * `the_wrapper_entry_point_reaches_the_reduced_problem` — `PresolveTnlp::new`
//!   is the one construction path that does not stack Phase 6, so a facade that
//!   offered only that shape would turn `presolve_linear_eq_reduction=yes` into
//!   a silent no-op. It pins the difference at the facade, which is where the
//!   choice is now offered.
//! * `every_report_type_can_be_bound_not_just_printed` — the accessors return
//!   types that must themselves be exported. Drop one from the re-export list
//!   and the annotations below stop compiling, which is the point; without
//!   them a caller can `{:?}` a verdict and nothing else.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_rs::Linearity;
use pounce_rs::prelude::*;
use pounce_rs::presolve::{
    AuxiliaryPreprocessingDiagnostics, CachedBounds, LicqVerdict, PresolveOptions, PresolveTnlp,
    TightenReport, wrap_from_options, wrap_with_presolve,
};
use pounce_rs::restoration::{SecondOpinionOutcome, run_second_opinion_ladder};

/// min (x₀ − 1)² + (x₂ − 3)²
/// s.t. x₀ − 2·x₁ = 0        (linear, two free variables — Phase 6 folds x₀ onto x₁)
///      x₁² + x₂² = 2        (nonlinear — survives, re-presented over the reduced columns)
///      −10 ≤ x₀, x₁ ≤ 10,  0 ≤ x₂ ≤ 10
///
/// The linear row also tightens a bound: x₀ ∈ [−10, 10] implies x₁ ∈ [−5, 5],
/// so Phase 1 has something to report.
#[derive(Default)]
struct Fixture {
    x_star: Option<Vec<Number>>,
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 2,
            nnz_jac_g: 4,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-10.0, -10.0, 0.0]);
        b.x_u.copy_from_slice(&[10.0, 10.0, 10.0]);
        b.g_l.copy_from_slice(&[0.0, 2.0]);
        b.g_u.copy_from_slice(&[0.0, 2.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 0.5, 1.0]);
        true
    }

    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.copy_from_slice(&[Linearity::Linear, Linearity::NonLinear]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[2] - 3.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 0.0;
        g[2] = 2.0 * (x[2] - 3.0);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] - 2.0 * x[1];
        g[1] = x[1] * x[1] + x[2] * x[2];
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
                irow.copy_from_slice(&[0, 0, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                values[0] = 1.0;
                values[1] = -2.0;
                values[2] = 2.0 * x[1];
                values[3] = 2.0 * x[2];
            }
        }
        true
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
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.map(|l| l[1]).unwrap_or(0.0);
                values[0] = obj_factor * 2.0;
                values[1] = lam * 2.0;
                values[2] = obj_factor * 2.0 + lam * 2.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.x_star = Some(sol.x.to_vec());
    }
}

fn options(linear_eq_reduction: bool) -> PresolveOptions {
    PresolveOptions {
        enabled: true,
        linear_eq_reduction,
        ..PresolveOptions::defaults()
    }
}

fn quiet_app() -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.initialize().unwrap();
    app
}

/// `n` and `m` of whatever the wrapper re-presents to the solver.
fn dims(tnlp: &Rc<RefCell<dyn TNLP>>) -> (Index, Index) {
    let info = tnlp.borrow_mut().get_nlp_info().expect("dims");
    (info.n, info.m)
}

#[test]
fn the_wrapper_entry_point_reaches_the_reduced_problem() {
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::default()));
    let wrapped = wrap_with_presolve(inner, options(true)).expect("wrap");
    assert_eq!(
        dims(&wrapped),
        (2, 1),
        "wrap_with_presolve must stack Phase 6: x₀ folds onto x₁ and its row goes"
    );

    // The same options through the wrapper that does *not* stack it — the trap
    // the facade would set if `wrap_with_presolve` were not exported.
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::default()));
    let bare: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(PresolveTnlp::new(inner, options(true))));
    assert_eq!(
        dims(&bare),
        (3, 2),
        "PresolveTnlp::new alone must not reduce columns — if it starts to, \
         this test has stopped saying anything"
    );

    // And with the option off, the entry point leaves the dimensions alone.
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::default()));
    let unreduced = wrap_with_presolve(inner, options(false)).expect("wrap");
    assert_eq!(dims(&unreduced), (3, 2));
}

#[test]
fn every_report_type_can_be_bound_not_just_printed() {
    // The accessors hang off the concrete wrapper, which is why the facade
    // exports `PresolveTnlp` and not only the `dyn TNLP` entry points.
    let inner: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::default()));
    let mut presolve = PresolveTnlp::new(inner, options(false));

    // Building the state is what populates the reports.
    let info = presolve.get_nlp_info().expect("dims");
    let mut x_l = vec![0.0; info.n as usize];
    let mut x_u = vec![0.0; info.n as usize];
    let mut g_l = vec![0.0; info.m as usize];
    let mut g_u = vec![0.0; info.m as usize];
    assert!(presolve.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }));

    // Each annotation is the assertion: the type has to be nameable here.
    let tighten: TightenReport = presolve.tighten_report();
    assert!(
        tighten.n_tightened >= 1,
        "x₀ − 2·x₁ = 0 with x₀ ∈ [−10, 10] tightens x₁ to [−5, 5]: {tighten:?}"
    );
    assert!(!tighten.infeasible);

    let licq: Option<&LicqVerdict> = presolve.licq_verdict();
    assert!(
        matches!(licq, Some(LicqVerdict::Full)),
        "two independent equality rows in three columns have full structural rank: {licq:?}"
    );

    let aux: AuxiliaryPreprocessingDiagnostics = presolve.auxiliary_diagnostics();
    assert_eq!(
        aux.blocks_eliminated, 0,
        "presolve_auxiliary is off by default"
    );

    let cached: Option<&CachedBounds> = presolve.cached_bounds();
    let cached = cached.expect("bounds are cached once the state is built");
    assert_eq!(cached.x_l.len(), 3);
    assert!(
        cached.x_l[1] > -10.0 && cached.x_u[1] < 10.0,
        "the cached box carries Phase 1's tightening: {:?}..{:?}",
        cached.x_l[1],
        cached.x_u[1]
    );

    // `certified_infeasible` returns the proof type; this model is feasible.
    assert!(presolve.certified_infeasible().is_none());
}

#[test]
fn wrap_from_options_composes_with_the_already_applied_flag() {
    let mut app = quiet_app();
    app.options_mut()
        .set_string_value("presolve", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("presolve_linear_eq_reduction", "yes", true, false)
        .unwrap();

    let concrete = Rc::new(RefCell::new(Fixture::default()));
    let wrapped = wrap_from_options(Rc::clone(&concrete) as Rc<RefCell<dyn TNLP>>, app.options())
        .expect("options materialize");
    assert_eq!(
        dims(&wrapped),
        (2, 1),
        "wrap_from_options reads presolve_linear_eq_reduction off the list"
    );

    // Declared, so `optimize_tnlp` does not wrap a second time.
    app.set_presolve_already_applied(true);
    let status = app.optimize_tnlp(wrapped);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);

    // Postsolve put the eliminated column back before the user's TNLP saw it.
    let x = concrete.borrow().x_star.clone().expect("finalize_solution");
    assert_eq!(x.len(), 3, "the caller is answered in its own variables");
    assert!(
        (x[0] - 2.0 * x[1]).abs() < 1e-6,
        "the eliminated row is satisfied at the recovered point: {x:?}"
    );
    assert!(
        (x[1] * x[1] + x[2] * x[2] - 2.0).abs() < 1e-6,
        "the surviving nonlinear row holds: {x:?}"
    );
}

#[test]
fn the_second_opinion_ladder_is_reachable_and_stays_out_of_the_way() {
    let mut app = quiet_app();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture::default()));

    let status = app.optimize_tnlp(Rc::clone(&tnlp));
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let stats = app.statistics();
    let base_iterations = stats.iteration_count;

    let outcome: SecondOpinionOutcome =
        run_second_opinion_ladder(&mut app, tnlp, status, stats, &mut |_line| {});

    // A converged solve opens no ladder, so the verdict and the cost are the
    // base solve's — which is also the contract a caller relies on to call
    // this unconditionally after every solve.
    assert!(!outcome.ran(), "tried = {:?}", outcome.tried);
    assert!(outcome.promoted_by.is_none());
    assert_eq!(outcome.status, ApplicationReturnStatus::SolveSucceeded);
    assert_eq!(outcome.base_status, outcome.status);
    assert_eq!(outcome.rung_iteration_counts, Vec::<usize>::new());
    assert_eq!(
        outcome.total_iteration_count(),
        base_iterations.max(0) as usize
    );
}
