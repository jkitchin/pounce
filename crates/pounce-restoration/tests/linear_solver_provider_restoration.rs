//! A `LinearSolverProvider` in the restoration phase (PR #1177 review).
//!
//! The restoration sub-solve builds its augmented-system solver from a clone
//! of the main builder and wraps it in `AugRestoSystemSolver`. These tests
//! force restoration (`start_with_resto`) and check that
//!
//! * the provider builds that solver too, with the context prefix `resto.`;
//! * `owns_refinement()` passes through `AugRestoSystemSolver`, so the
//!   restoration phase's `PdFullSpaceSolver` does not refine the provider's
//!   step against the unmodified system: the `resto.` solver is solved fewer
//!   times than the same provider's without ownership. (Removing the
//!   forwarding in `AugRestoSystemSolver` fails this test.)

use pounce_algorithm::application::{
    IpoptApplication, Ma57Config, default_backend_factory, feral_config_from_options,
};
use pounce_algorithm::kkt::AugSystemSolver;
use pounce_algorithm::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol};
use pounce_algorithm::linear_solver_provider::{
    AugSolverContext, LinearSolverProvider, register_linear_solver_provider,
};
use pounce_common::types::{Index, Number};
use pounce_linsol::ESymSolverStatus;
use pounce_nlp::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    InnerBackendFactoryFactory, make_default_restoration_factory_provider,
};
use std::cell::RefCell;
use std::collections::HashMap;
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

#[derive(Default, Clone, Copy, Debug)]
struct Counts {
    builds: usize,
    solves: usize,
    resolves: usize,
}

type Key = (&'static str, String);

fn counts() -> &'static Mutex<HashMap<Key, Counts>> {
    static C: OnceLock<Mutex<HashMap<Key, Counts>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

fn bump(name: &'static str, prefix: &str, f: impl FnOnce(&mut Counts)) {
    f(counts()
        .lock()
        .unwrap()
        .entry((name, prefix.to_string()))
        .or_default());
}

fn counts_of(name: &'static str, prefix: &str) -> Counts {
    counts()
        .lock()
        .unwrap()
        .get(&(name, prefix.to_string()))
        .copied()
        .unwrap_or_default()
}

struct Wrapping {
    name: &'static str,
    owns_refinement: bool,
}

impl LinearSolverProvider for Wrapping {
    fn name(&self) -> &'static str {
        self.name
    }
    fn aug_system_solver(
        &self,
        ctx: &AugSolverContext<'_>,
    ) -> Result<Box<dyn AugSystemSolver>, String> {
        bump(self.name, ctx.prefix, |c| c.builds += 1);
        Ok(Box::new(Counted {
            name: self.name,
            prefix: ctx.prefix.to_string(),
            owns_refinement: self.owns_refinement,
            inner: ctx.standard(ctx.default_backend()),
        }))
    }
}

struct Counted {
    name: &'static str,
    prefix: String,
    owns_refinement: bool,
    inner: Box<dyn AugSystemSolver>,
}

impl AugSystemSolver for Counted {
    fn provides_inertia(&self) -> bool {
        self.inner.provides_inertia()
    }
    fn number_of_neg_evals(&self) -> Index {
        self.inner.number_of_neg_evals()
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
    fn handles_low_rank_w(&self) -> bool {
        self.inner.handles_low_rank_w()
    }
    fn solve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
        check_neg_evals: bool,
        num_neg_evals: Index,
    ) -> ESymSolverStatus {
        bump(self.name, &self.prefix, |c| c.solves += 1);
        self.inner
            .solve(coeffs, rhs, sol, check_neg_evals, num_neg_evals)
    }
    fn resolve(
        &mut self,
        coeffs: &AugSysCoeffs<'_>,
        rhs: &AugSysRhs<'_>,
        sol: &mut AugSysSol<'_>,
    ) -> ESymSolverStatus {
        bump(self.name, &self.prefix, |c| c.resolves += 1);
        self.inner.resolve(coeffs, rhs, sol)
    }
}

static REFINED: Wrapping = Wrapping {
    name: "resto_test_refined",
    owns_refinement: false,
};
static OWNS: Wrapping = Wrapping {
    name: "resto_test_owns",
    owns_refinement: true,
};

/// Register the providers, then wire restoration the way every frontend
/// does (a bare `IpoptApplication` has no restoration factory). The
/// restoration builder is snapshotted from the options, so they are set
/// first.
fn solve_with_resto(solver: &'static str) -> (ApplicationReturnStatus, Option<Number>) {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        register_linear_solver_provider(&REFINED).unwrap();
        register_linear_solver_provider(&OWNS).unwrap();
    });
    let mut app = IpoptApplication::new();
    app.initialize().unwrap();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, true).unwrap();
    o.set_string_value("linear_solver", solver, true, true)
        .unwrap();
    o.set_string_value("start_with_resto", "yes", true, true)
        .unwrap();
    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone(), Ma57Config::default()))
    };
    let provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(provider);
    let tnlp = Rc::new(RefCell::new(Hs071::default()));
    let status = app.optimize_tnlp(Rc::clone(&tnlp) as Rc<RefCell<dyn TNLP>>);
    let obj = tnlp.borrow().final_obj;
    (status, obj)
}

#[test]
fn restoration_builds_the_provider_and_honours_owned_refinement() {
    let (status, obj) = solve_with_resto("resto_test_refined");
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!((obj.unwrap() - 17.014_017_3).abs() < 1e-6);
    let main = counts_of("resto_test_refined", "");
    let resto = counts_of("resto_test_refined", "resto.");
    assert!(main.builds >= 1 && main.solves > 0, "{main:?}");
    assert!(
        resto.builds >= 1 && resto.solves > 0,
        "restoration did not use the provider: {resto:?}"
    );
    // `AugRestoSystemSolver` re-solves through its inner `solve` (it has no
    // cached-factor `resolve`), so restoration's refinement shows up as
    // extra solves of the provider's solver.
    let refined_resto_solves = resto.solves;

    let (status, obj) = solve_with_resto("resto_test_owns");
    assert_eq!(status, ApplicationReturnStatus::SolveSucceeded);
    assert!((obj.unwrap() - 17.014_017_3).abs() < 1e-6);
    let main = counts_of("resto_test_owns", "");
    let resto = counts_of("resto_test_owns", "resto.");
    assert!(resto.builds >= 1 && resto.solves > 0, "{resto:?}");
    assert!(
        resto.solves < refined_resto_solves,
        "AugRestoSystemSolver must pass ownership through: {resto:?} vs {refined_resto_solves} solves"
    );
    assert_eq!(main.resolves, 0, "{main:?}");
}
