//! A `LinearSolverProvider` registered by the embedding build (gh#1175
//! follow-up, PR #1177 review): the augmented-system level of Ipopt's
//! custom-solver hook. Checks that the provider
//!
//! * is selected by `linear_solver=<name>` and solves HS071 to its optimum;
//! * receives the KKT layout and the integer metadata in KKT order;
//! * registers typed, bounded options that the registry validates;
//! * is used for the limited-memory bypass solver (restoration, which this
//!   crate cannot wire, is covered by `pounce-restoration`'s
//!   `linear_solver_provider_restoration.rs`);
//! * can own iterative refinement, after which `PdFullSpaceSolver` does not
//!   back-solve against the unmodified system;
//! * is refused, never replaced by FERAL, when it cannot be registered,
//!   validated or built.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::kkt::AugSystemSolver;
use pounce_algorithm::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol};
use pounce_algorithm::linear_solver_provider::{
    AugSolverContext, LinearSolverProvider, register_linear_solver_provider,
};
use pounce_common::exception::SolverException;
use pounce_common::options_list::OptionsList;
use pounce_common::reg_options::RegisteredOptions;
use pounce_common::types::{Index, Number};
use pounce_linsol::ESymSolverStatus;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, MetaData, NlpInfo, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

#[derive(Default)]
struct Hs071 {
    final_x: Option<[Number; 4]>,
    final_obj: Option<Number>,
    final_lambda: Option<Vec<Number>>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
}

impl TNLP for Hs071 {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 2,
            nnz_jac_g: 8,
            nnz_h_lag: 10,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[1.0; 4]);
        b.x_u.copy_from_slice(&[5.0; 4]);
        // g0: x1*x2*x3*x4 >= 25  (inequality, finite lower only)
        // g1: sum xi^2 == 40     (equality)
        b.g_l.copy_from_slice(&[25.0, 40.0]);
        b.g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
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
                irow.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
                jcol.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = x[1] * x[2] * x[3];
                values[1] = x[0] * x[2] * x[3];
                values[2] = x[0] * x[1] * x[3];
                values[3] = x[0] * x[1] * x[2];
                values[4] = 2.0 * x[0];
                values[5] = 2.0 * x[1];
                values[6] = 2.0 * x[2];
                values[7] = 2.0 * x[3];
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 2, 2, 3, 3, 3, 3]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2, 0, 1, 2, 3]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda");
                let of = obj_factor;
                let l0 = lam[0];
                let l1 = lam[1];
                values[0] = of * (2.0 * x[3]) + l1 * 2.0;
                values[1] = of * x[3] + l0 * (x[2] * x[3]);
                values[2] = l1 * 2.0;
                values[3] = of * x[3] + l0 * (x[1] * x[3]);
                values[4] = l0 * (x[0] * x[3]);
                values[5] = l1 * 2.0;
                values[6] = of * (2.0 * x[0] + x[1] + x[2]) + l0 * (x[1] * x[2]);
                values[7] = of * x[0] + l0 * (x[0] * x[2]);
                values[8] = of * x[0] + l0 * (x[0] * x[1]);
                values[9] = l1 * 2.0;
            }
        }
        true
    }

    fn get_var_con_metadata(&mut self, var: &mut MetaData, con: &mut MetaData) -> bool {
        // A block label per variable and per constraint, as a structured
        // solver would read them.
        var.integers.insert("block".into(), vec![1, 1, 2, 2]);
        con.integers.insert("block".into(), vec![7, 8]);
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        if sol.x.len() == 4 {
            self.final_x = Some([sol.x[0], sol.x[1], sol.x[2], sol.x[3]]);
        }
        self.final_obj = Some(sol.obj_value);
        self.final_lambda = Some(sol.lambda.to_vec());
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
    }
}

// ---------------------------------------------------------------------------
// The provider: wraps POUNCE's standard solver and records what it sees.
// ---------------------------------------------------------------------------

#[derive(Default, Clone, Debug)]
struct Stats {
    builds: usize,
    solves: usize,
    resolves: usize,
    prefixes: BTreeSet<String>,
    layout: Option<[Index; 4]>,
    labels: Option<Vec<Index>>,
    penalty: Option<Number>,
}

fn stats() -> &'static Mutex<HashMap<&'static str, Stats>> {
    static S: OnceLock<Mutex<HashMap<&'static str, Stats>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn stats_of(name: &'static str) -> Stats {
    stats()
        .lock()
        .unwrap()
        .get(name)
        .cloned()
        .unwrap_or_default()
}

fn record(name: &'static str, f: impl FnOnce(&mut Stats)) {
    f(stats().lock().unwrap().entry(name).or_default());
}

struct Recording {
    name: &'static str,
    registers_options: bool,
    owns_refinement: bool,
    fail_build: bool,
    fail_validate: bool,
}

impl LinearSolverProvider for Recording {
    fn name(&self) -> &'static str {
        self.name
    }

    fn register_options(&self, r: &RegisteredOptions) -> Result<(), SolverException> {
        if !self.registers_options {
            return Ok(());
        }
        r.set_registering_category("Recording Linear Solver");
        r.add_lower_bounded_number_option(
            "rec_penalty",
            "Primal penalty per block.",
            0.0,
            false,
            0.0,
            "",
        )?;
        r.add_string_option(
            "rec_partition",
            "Partition route.",
            "border",
            &[
                ("border", "on the border"),
                ("stage", "one block per stage"),
            ],
            "",
        )?;
        r.add_lower_bounded_integer_option("rec_refinement_steps", "Refinement steps.", 0, 3, "")?;
        Ok(())
    }

    fn validate(&self, _options: &OptionsList) -> Result<(), String> {
        if self.fail_validate {
            Err("rec_invalid refuses every configuration".into())
        } else {
            Ok(())
        }
    }

    fn aug_system_solver(
        &self,
        ctx: &AugSolverContext<'_>,
    ) -> Result<Box<dyn AugSystemSolver>, String> {
        if self.fail_build {
            return Err("no structure found".into());
        }
        let penalty = ctx
            .options
            .get_numeric_value("rec_penalty", ctx.prefix)
            .ok()
            .map(|(v, _)| v);
        let name = self.name;
        let prefix = ctx.prefix.to_string();
        record(name, |s| {
            s.builds += 1;
            s.prefixes.insert(prefix);
            s.penalty = penalty;
        });
        Ok(Box::new(Recorder {
            name,
            owns_refinement: self.owns_refinement,
            metadata: std::sync::Arc::clone(ctx.metadata),
            inner: ctx.standard(ctx.default_backend()),
        }))
    }
}

struct Recorder {
    name: &'static str,
    owns_refinement: bool,
    metadata: pounce_algorithm::custom_linsol::KktMetadataCell,
    inner: Box<dyn AugSystemSolver>,
}

impl AugSystemSolver for Recorder {
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }
    fn singularity_certified(&self) -> bool {
        self.inner.singularity_certified()
    }
    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
    }
    fn system_dim(&self) -> Index {
        self.inner.system_dim()
    }
    fn increase_quality(&mut self) -> bool {
        self.inner.increase_quality()
    }
    fn last_solve_status(&self) -> ESymSolverStatus {
        self.inner.last_solve_status()
    }
    fn owns_refinement(&self) -> bool {
        self.owns_refinement
    }
    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        let meta = self.metadata.lock().unwrap().clone();
        record(self.name, |s| {
            s.solves += 1;
            if let Some(m) = meta {
                s.layout = Some([m.n_x, m.n_s, m.n_c, m.n_d]);
                s.labels = m.integers.get("block").cloned();
            }
        });
        self.inner
            .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals)
    }
    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        record(self.name, |s| s.resolves += 1);
        self.inner.resolve(coeffs, rhs, sol)
    }
}

const fn rec(name: &'static str) -> Recording {
    Recording {
        name,
        registers_options: false,
        owns_refinement: false,
        fail_build: false,
        fail_validate: false,
    }
}

static BASIC: Recording = Recording {
    registers_options: true,
    ..rec("rec_basic")
};
static LBFGS: Recording = rec("rec_lbfgs");
static REFINED: Recording = rec("rec_refined");
static OWNS: Recording = Recording {
    owns_refinement: true,
    ..rec("rec_owns")
};
static OWNS_QF: Recording = Recording {
    owns_refinement: true,
    ..rec("rec_owns_qf")
};
static FAIL_BUILD: Recording = Recording {
    fail_build: true,
    ..rec("rec_fail_build")
};
static FAIL_VALIDATE: Recording = Recording {
    fail_validate: true,
    ..rec("rec_fail_validate")
};

/// Registration happens once per process, before any application exists.
fn setup() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        for p in [
            &BASIC as &'static dyn LinearSolverProvider,
            &LBFGS,
            &REFINED,
            &OWNS,
            &OWNS_QF,
            &FAIL_BUILD,
            &FAIL_VALIDATE,
        ] {
            register_linear_solver_provider(p).unwrap();
        }
    });
}

fn app_with(solver: &str) -> IpoptApplication {
    setup();
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("linear_solver", solver, true, false)
        .unwrap();
    app
}

fn solve(app: &mut IpoptApplication) -> (ApplicationReturnStatus, Option<Number>) {
    let tnlp = Rc::new(RefCell::new(Hs071::default()));
    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let obj = tnlp.borrow().final_obj;
    (status, obj)
}

fn assert_optimal(status: ApplicationReturnStatus, obj: Option<Number>) {
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    let f = obj.expect("no objective");
    assert!((f - 17.014_017_3).abs() < 1e-6, "f = {f}");
}

#[test]
fn provider_solves_hs071_and_sees_layout_and_labels() {
    let mut app = app_with("rec_basic");
    app.options_mut()
        .set_numeric_value("rec_penalty", 0.25, true, false)
        .unwrap();
    let (status, obj) = solve(&mut app);
    assert_optimal(status, obj);
    let s = stats_of("rec_basic");
    assert_eq!(s.builds, 1, "exact Hessian: one solver");
    assert!(s.solves > 0);
    assert_eq!(s.prefixes, BTreeSet::from([String::new()]));
    assert_eq!(
        s.penalty,
        Some(0.25),
        "typed option read back from the context"
    );
    // [x | slacks | eq duals | ineq duals]; g0 is the inequality, g1 the equality.
    assert_eq!(s.layout, Some([4, 1, 1, 1]));
    assert_eq!(s.labels, Some(vec![1, 1, 2, 2, 7, 8, 7]));
}

#[test]
fn provider_options_are_typed_and_validated() {
    setup();
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    assert!(
        o.set_numeric_value("rec_penalty", -1.0, true, false)
            .map_or(true, |ok| !ok)
    );
    assert!(
        o.set_string_value("rec_partition", "diagonal", true, false)
            .map_or(true, |ok| !ok)
    );
    assert!(
        o.set_string_value("rec_partition", "stage", true, false)
            .unwrap()
    );
    assert!(
        o.set_integer_value("rec_refinement_steps", -2, true, false)
            .map_or(true, |ok| !ok)
    );
    assert!(
        o.set_integer_value("rec_refinement_steps", 2, true, false)
            .unwrap()
    );
    // The value is offered by `linear_solver` itself.
    assert!(
        o.set_string_value("linear_solver", "rec_basic", true, false)
            .unwrap()
    );
}

#[test]
fn limited_memory_wraps_two_provider_solvers() {
    let mut app = app_with("rec_lbfgs");
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, false)
        .unwrap();
    let (status, obj) = solve(&mut app);
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!(obj.is_some());
    // Main inner solver and the low-rank wrapper's bypass solver.
    assert_eq!(stats_of("rec_lbfgs").builds, 2);
}

#[test]
fn owned_refinement_skips_the_refinement_back_solves() {
    // Same provider, refinement owned or not. `min_refinement_steps = 1`
    // (the default) makes PdFullSpaceSolver back-solve at least once per
    // step for a solver that does not own refinement.
    let mut app = app_with("rec_refined");
    let (status, obj) = solve(&mut app);
    assert_optimal(status, obj);
    let refined = stats_of("rec_refined");
    assert!(refined.resolves > 0, "{refined:?}");

    let mut app = app_with("rec_owns");
    let (status, obj) = solve(&mut app);
    assert_optimal(status, obj);
    let owns = stats_of("rec_owns");
    assert_eq!(owns.resolves, 0, "{owns:?}");
    assert!(owns.solves > 0);
}

#[test]
fn owned_refinement_answers_an_improve_request_with_a_fresh_solve() {
    // The quality-function oracle computes a step and then asks
    // PdFullSpaceSolver to *improve* it (`data.delta` already set). A solver
    // that owns refinement is not refined, so that request must re-solve
    // with the new right-hand side, not return the given step unchanged.
    let mut app = app_with("rec_owns_qf");
    let o = app.options_mut();
    o.set_string_value("mu_strategy", "adaptive", true, false)
        .unwrap();
    o.set_string_value("mu_oracle", "quality-function", true, false)
        .unwrap();
    // Without the fresh solve the oracle's step repeats and the run ends in
    // restoration failure (measured: `RestorationFailed`). The oracle's own
    // same-matrix back-solves for its other right-hand sides still happen;
    // they are not refinement.
    let (status, obj) = solve(&mut app);
    assert_optimal(status, obj);
}

#[test]
fn a_failing_provider_is_refused_not_replaced() {
    let mut app = app_with("rec_fail_validate");
    let (status, _) = solve(&mut app);
    assert_eq!(status, ApplicationReturnStatus::InvalidOption);

    let mut app = app_with("rec_fail_build");
    let (status, _) = solve(&mut app);
    assert_ne!(status, ApplicationReturnStatus::SolveSucceeded);
    assert_ne!(status, ApplicationReturnStatus::SolvedToAcceptableLevel);
}

#[test]
fn registration_refuses_taken_names_and_options() {
    setup();
    static FERAL: Recording = rec("feral");
    static MUMPS: Recording = rec("MUMPS");
    static DUP: Recording = rec("rec_basic");
    static CLASH: Recording = Recording {
        registers_options: true,
        ..rec("rec_clash")
    };
    static EMPTY: Recording = rec("");
    for p in [&FERAL, &MUMPS, &DUP, &CLASH, &EMPTY] {
        assert!(register_linear_solver_provider(p).is_err(), "{}", p.name);
    }
    // Registering the same provider object again is a no-op.
    assert!(register_linear_solver_provider(&BASIC).is_ok());
}

#[test]
fn unregistered_names_are_still_refused() {
    let mut app = app_with("mumps");
    let (status, _) = solve(&mut app);
    assert_eq!(status, ApplicationReturnStatus::InvalidOption);
}
