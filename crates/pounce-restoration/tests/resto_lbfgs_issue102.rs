//! Repro for pounce#102: a constrained limited-memory solve that enters
//! feasibility restoration used to panic. The limited-memory updater
//! published the restoration Hessian `W` as a *flat dense*
//! `LowRankUpdateSymMatrix`, but the resto sub-IPM operates on 5-block
//! `CompoundVector` iterates, so every `W·x` panicked — first in
//! `AugRestoSystemSolver` (which expected a `SymTMatrix`) and, even past
//! that, in the inner solver's residual computation. The fix builds `W`'s
//! diagonal and curvature columns in the primal iterates' native vector
//! space. Mirrors the Python `Problem`/`minimize` path (dense
//! lower-triangle `eval_h` sparsity with zero values + L-BFGS).

use pounce_algorithm::application::{
    default_backend_factory, feral_config_from_options, IpoptApplication,
};
use pounce_common::types::Number;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_nlp::ApplicationReturnStatus;
use pounce_restoration::resto_alg_builder::RestoAlgorithmBuilder;
use pounce_restoration::resto_inner_solver::{
    make_default_restoration_factory_provider, InnerBackendFactoryFactory,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Toy;

impl TNLP for Toy {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 3,
            nnz_jac_g: 9, // dense
            nnz_h_lag: 6, // dense lower triangle of 3x3
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-5.0; 3]);
        b.x_u.copy_from_slice(&[5.0; 3]);
        // three equality constraints (g_l == g_u)
        b.g_l.copy_from_slice(&[0.0, 0.0, 0.0]);
        b.g_u.copy_from_slice(&[0.0, 0.0, 0.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        // The bad random-in-box start that triggered restoration in the
        // Python repro.
        sp.x.copy_from_slice(&[1.37, -2.302, -4.59]);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 3.0;
        g[1] = x[0] * x[1] * x[2] - 1.0;
        g[2] = x[0].exp() + x[1] - x[2] - 2.0;
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
                irow.copy_from_slice(&[0, 0, 0, 1, 1, 1, 2, 2, 2]);
                jcol.copy_from_slice(&[0, 1, 2, 0, 1, 2, 0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 2.0 * x[0];
                values[1] = 2.0 * x[1];
                values[2] = 2.0 * x[2];
                values[3] = x[1] * x[2];
                values[4] = x[0] * x[2];
                values[5] = x[0] * x[1];
                values[6] = x[0].exp();
                values[7] = 1.0;
                values[8] = -1.0;
            }
        }
        true
    }

    // Mirror the Python bridge's no-user-Hessian path: declare a dense
    // lower-triangle sparsity, hand back zeros (the L-BFGS updater owns
    // the values).
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 1, 2, 2, 2]);
                jcol.copy_from_slice(&[0, 0, 1, 0, 1, 2]);
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

#[test]
fn resto_lbfgs_does_not_panic_issue102() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_string_value("hessian_approximation", "limited-memory", true, true)
        .unwrap();
    app.options_mut()
        .set_integer_value("print_level", 0, true, true)
        .unwrap();
    app.initialize().unwrap();

    // Mirror the Python `Problem.solve` wiring: install a restoration
    // factory provider built from the (limited-memory) options builder,
    // so the resto sub-IPM inherits limited-memory too (pounce#102).
    let feral_cfg = feral_config_from_options(app.options());
    let bff_mint = move || -> InnerBackendFactoryFactory {
        let feral_cfg = feral_cfg.clone();
        Box::new(move || default_backend_factory(feral_cfg.clone()))
    };
    let resto_provider = make_default_restoration_factory_provider(
        RestoAlgorithmBuilder::new(),
        app.algorithm_builder_from_options(),
        bff_mint,
    );
    app.set_restoration_factory_provider(resto_provider);

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Toy));
    // Before the pounce#102 fix this call panicked inside the resto
    // sub-IPM the moment restoration triggered (the limited-memory `W`
    // was a flat dense `LowRankUpdateSymMatrix` that could not multiply
    // the resto compound iterates). It must now run to a clean terminal
    // status. This particular toy is genuinely infeasible (the three
    // equalities have no common root — scipy SLSQP also fails it), so we
    // only assert the solve *finished* without panicking and did not come
    // back with an internal-error verdict.
    let status = app.optimize_tnlp(tnlp);
    eprintln!("issue102 repro status = {status:?}");
    assert!(
        !matches!(
            status,
            ApplicationReturnStatus::InternalError
                | ApplicationReturnStatus::UnrecoverableException
        ),
        "resto limited-memory solve returned an error verdict: {status:?}"
    );
}
