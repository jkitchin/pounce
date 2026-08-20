//! The registered sIPOPT option names change what a sensitivity solve
//! computes (gh#551 / gh#677).
//!
//! `compute_red_hessian`, `rh_eigendecomp`, `run_sens`,
//! `sens_boundcheck`, `sens_bound_eps` and `sens_max_pdpert` were
//! registered — an `ipopt.opt` carrying them parsed — and every one of
//! them then did nothing: the same work was reachable only through the
//! [`SensSolve`] builder or the CLI's own flags. These tests set the
//! option and check the *output changes*, which is the only thing that
//! separates a read site from the no-op it replaced.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use pounce_sensitivity::SensSolve;

/// Same NLP as `tests/parametric_cpp.rs::ParametricTNLP` and
/// `tests/convenience_api.rs`. Kept local here to avoid an
/// inter-test-binary module dependency.
struct ParametricTNLP {
    nominal_eta1: Number,
    nominal_eta2: Number,
}

impl ParametricTNLP {
    fn new(eta1: Number, eta2: Number) -> Self {
        Self {
            nominal_eta1: eta1,
            nominal_eta2: eta2,
        }
    }
}

impl TNLP for ParametricTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 4,
            nnz_jac_g: 10,
            nnz_h_lag: 5,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..3 {
            b.x_l[k] = 0.0;
            b.x_u[k] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        b.g_l[2] = self.nominal_eta1;
        b.g_u[2] = self.nominal_eta1;
        b.g_l[3] = self.nominal_eta2;
        b.g_u[3] = self.nominal_eta2;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.15;
        sp.x[1] = 0.15;
        sp.x[2] = 0.0;
        sp.x[3] = 0.0;
        sp.x[4] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0];
        g[1] = 2.0 * x[1];
        g[2] = 2.0 * x[2];
        g[3] = 0.0;
        g[4] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, x3, eta1, eta2) = (x[0], x[1], x[2], x[3], x[4]);
        g[0] = 6.0 * x1 + 3.0 * x2 + 2.0 * x3 - eta1;
        g[1] = eta2 * x1 + x2 - x3 - 1.0;
        g[2] = eta1;
        g[3] = eta2;
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
                let rs: [Index; 10] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3];
                let cs: [Index; 10] = [0, 1, 2, 3, 0, 1, 2, 4, 3, 4];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                values[0] = 6.0;
                values[1] = 3.0;
                values[2] = 2.0;
                values[3] = -1.0;
                values[4] = x[4];
                values[5] = 1.0;
                values[6] = -1.0;
                values[7] = x[0];
                values[8] = 1.0;
                values[9] = 1.0;
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
                let rs: [Index; 5] = [0, 1, 2, 4, 0];
                let cs: [Index; 5] = [0, 1, 2, 0, 0];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                let lam = lambda.expect("eval_h(Values) without lambda");
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
                values[2] = 2.0 * obj_factor;
                values[3] = lam[1];
                values[4] = 0.0;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// A quiet application. `opts` are applied the way an `ipopt.opt` (or a
/// `key=value` on the pounce command line) applies them — through the
/// options list, which is the channel these tests are about.
fn app_with(opts: &[(&str, &str)]) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    for (k, v) in opts {
        app.options_mut()
            .read_from_str(&format!("{k} {v}\n"), true)
            .unwrap_or_else(|e| panic!("`{k} {v}` must parse: {e:?}"));
    }
    app.initialize().unwrap();
    app
}

fn tnlp() -> Rc<RefCell<dyn TNLP>> {
    Rc::new(RefCell::new(ParametricTNLP::new(5.0, 1.0)))
}

fn converged(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// `compute_red_hessian=yes` alone produces the reduced Hessian, on a
/// builder that never called `with_reduced_hessian()`. Without the
/// option the same call returns `None` — that difference is the point.
#[test]
fn compute_red_hessian_option_produces_the_reduced_hessian() {
    let off = SensSolve::new(vec![2, 3]).run(&mut app_with(&[]), tnlp());
    assert!(converged(off.status), "{:?}", off.status);
    assert!(
        off.reduced_hessian.is_none(),
        "unset: the builder asked for nothing"
    );

    let on =
        SensSolve::new(vec![2, 3]).run(&mut app_with(&[("compute_red_hessian", "yes")]), tnlp());
    assert!(converged(on.status), "{:?}", on.status);
    let hr = on
        .reduced_hessian
        .expect("`compute_red_hessian=yes` must produce the reduced Hessian");
    assert_eq!(hr.len(), 4, "2 pins → 2x2 column-major");
    assert!(
        (hr[1] - hr[2]).abs() < 1e-8,
        "H_R must be symmetric: {hr:?}"
    );
    assert!(on.error.is_none(), "{:?}", on.error);
}

/// `rh_eigendecomp=yes` implies the reduced Hessian and adds its
/// eigenpairs — the same implication `--rh-eigendecomp` and
/// `with_reduced_hessian_eigen()` carry.
#[test]
fn rh_eigendecomp_option_adds_the_eigenpairs() {
    let off =
        SensSolve::new(vec![2, 3]).run(&mut app_with(&[("compute_red_hessian", "yes")]), tnlp());
    assert!(
        off.reduced_hessian_eigenvalues.is_none(),
        "the matrix alone must not decompose it"
    );

    let on = SensSolve::new(vec![2, 3]).run(&mut app_with(&[("rh_eigendecomp", "yes")]), tnlp());
    assert!(converged(on.status), "{:?}", on.status);
    let hr = on
        .reduced_hessian
        .expect("`rh_eigendecomp=yes` implies the reduced Hessian");
    let w = on
        .reduced_hessian_eigenvalues
        .expect("`rh_eigendecomp=yes` must produce eigenvalues");
    let v = on
        .reduced_hessian_eigenvectors
        .expect("...and eigenvectors");
    assert_eq!(w.len(), 2);
    assert!(w[0] <= w[1], "ascending: {w:?}");
    // H_R · v_j = λ_j · v_j, so the pair really describes this matrix.
    for j in 0..2 {
        let (v0, v1) = (v[2 * j], v[2 * j + 1]);
        let av0 = hr[0] * v0 + hr[2] * v1;
        let av1 = hr[1] * v0 + hr[3] * v1;
        assert!((av0 - w[j] * v0).abs() < 1e-8, "eigenpair {j}");
        assert!((av1 - w[j] * v1).abs() < 1e-8, "eigenpair {j}");
    }
}

/// `run_sens=no` is upstream's "solve, but do not take the step". It
/// has to actually take the step away — a caller that asked for deltas
/// gets none.
#[test]
fn run_sens_no_suppresses_the_step() {
    let on = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_with(&[]), tnlp());
    assert!(converged(on.status), "{:?}", on.status);
    assert!(on.dx.is_some(), "the step runs when nothing suppresses it");

    let off = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_with(&[("run_sens", "no")]), tnlp());
    assert!(converged(off.status), "{:?}", off.status);
    assert!(
        off.dx.is_none() && off.dx_full.is_none(),
        "`run_sens=no` must suppress the sensitivity step"
    );
    // Suppressed on request is not a failure: the solve's own outputs
    // are still there and nothing is reported as broken.
    assert!(off.error.is_none(), "{:?}", off.error);
    assert!(off.x.is_some() && off.obj_val.is_some());
}

/// `sens_boundcheck=yes` refines the step onto the declared box. On
/// this model the unrefined step carries x₂ below its `x_l = 0`, so the
/// option is the difference between a perturbed primal outside the
/// user's own bounds and one on them.
#[test]
fn sens_boundcheck_option_refines_the_step() {
    let plain = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_with(&[]), tnlp());
    let x = plain.x.clone().expect("x*");
    let dx = plain.dx.clone().expect("dx");
    let crossed = x[2] + dx[2];
    assert!(
        crossed < -1e-3,
        "fixture check: the unrefined step must leave x_l=0 (got {crossed})"
    );

    let refined = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_with(&[("sens_boundcheck", "yes")]), tnlp());
    assert!(refined.error.is_none(), "{:?}", refined.error);
    let rdx = refined.dx.clone().expect("dx with the refinement on");
    assert!(
        (rdx[2] - dx[2]).abs() > 1e-6,
        "the refinement must move the step: {} vs {}",
        rdx[2],
        dx[2]
    );
    assert!(
        x[2] + rdx[2] > -1e-3,
        "the refined step must hold x_2 at its bound, got {}",
        x[2] + rdx[2]
    );
}

/// `sens_bound_eps` is the margin the refinement measures a crossing
/// against, so a margin wider than the crossing leaves the step alone
/// and a tight one pins it. Same model, same builder, two answers.
#[test]
fn sens_bound_eps_sets_the_margin_the_refinement_uses() {
    let tight = SensSolve::new(vec![2, 3]).with_deltas(vec![-0.5, 0.0]).run(
        &mut app_with(&[("sens_boundcheck", "yes"), ("sens_bound_eps", "1e-9")]),
        tnlp(),
    );
    let slack = SensSolve::new(vec![2, 3]).with_deltas(vec![-0.5, 0.0]).run(
        &mut app_with(&[("sens_boundcheck", "yes"), ("sens_bound_eps", "10.0")]),
        tnlp(),
    );
    let plain = SensSolve::new(vec![2, 3])
        .with_deltas(vec![-0.5, 0.0])
        .run(&mut app_with(&[]), tnlp());

    let (t, s, p) = (
        tight.dx.expect("dx"),
        slack.dx.expect("dx"),
        plain.dx.expect("dx"),
    );
    // A margin of 10 is wider than the ~0.07 crossing, so nothing is
    // out of bounds *by more than eps* and the step is untouched.
    for k in 0..s.len() {
        assert!(
            (s[k] - p[k]).abs() < 1e-12,
            "eps=10 must leave the step alone at {k}: {} vs {}",
            s[k],
            p[k]
        );
    }
    assert!(
        (t[2] - s[2]).abs() > 1e-6,
        "eps=1e-9 must pin what eps=10 tolerates: {} vs {}",
        t[2],
        s[2]
    );
}

/// A rank-deficient constraint set: `g1` and `g2` are the same row, so
/// the KKT matrix is singular and the solve only gets a factor at all
/// because the Jacobian regularization perturbs it. That perturbation
/// is exactly what `sens_max_pdpert` is a cap on.
///
/// `min x₀² + x₁² + x₂²` subject to `x₀ + x₁ + x₂ + p = 1` and
/// `p = 0.25` (stated twice).
struct RankDeficientTNLP;

impl TNLP for RankDeficientTNLP {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 4,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        for k in 0..4 {
            b.x_l[k] = -1.0e19;
            b.x_u[k] = 1.0e19;
        }
        b.g_l[0] = 1.0;
        b.g_u[0] = 1.0;
        for k in 1..3 {
            b.g_l[k] = 0.25;
            b.g_u[k] = 0.25;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        for k in 0..4 {
            sp.x[k] = 0.5;
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for k in 0..3 {
            g[k] = 2.0 * x[k];
        }
        g[3] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1] + x[2] + x[3];
        g[1] = x[3];
        g[2] = x[3];
        true
    }

    fn eval_jac_g(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let rs: [Index; 6] = [0, 0, 0, 0, 1, 2];
                let cs: [Index; 6] = [0, 1, 2, 3, 3, 3];
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0; 6]);
            }
        }
        true
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        obj_factor: Number,
        _lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for k in 0..3 {
                    irow[k] = k as Index;
                    jcol[k] = k as Index;
                }
            }
            SparsityRequest::Values { values } => {
                for v in values.iter_mut() {
                    *v = 2.0 * obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// `sens_max_pdpert` refuses a step taken through a factor the inertia
/// correction had to perturb past the cap — and returns one when the
/// cap is above the perturbation. Unset, the step is returned however
/// perturbed the factor is, which is pounce's behaviour today and the
/// one this option does not change on its own.
#[test]
fn sens_max_pdpert_caps_the_perturbation_of_the_factor() {
    let rd = || -> Rc<RefCell<dyn TNLP>> { Rc::new(RefCell::new(RankDeficientTNLP)) };

    let uncapped = SensSolve::new(vec![1])
        .with_deltas(vec![0.01])
        .run(&mut app_with(&[]), rd());
    assert!(converged(uncapped.status), "{:?}", uncapped.status);
    let perts = uncapped
        .kkt_perturbations
        .expect("perturbations are reported on every converged sens solve");
    let worst = perts.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        worst > 0.0,
        "fixture check: this model's factor must be regularized, got {perts:?}"
    );
    assert!(
        uncapped.dx.is_some() && uncapped.error.is_none(),
        "unset `sens_max_pdpert` must keep returning the step: {:?}",
        uncapped.error
    );

    let below = format!("{:e}", worst / 10.0);
    let capped = SensSolve::new(vec![1])
        .with_deltas(vec![0.01])
        .run(&mut app_with(&[("sens_max_pdpert", &below)]), rd());
    assert!(converged(capped.status), "the solve itself still succeeds");
    assert!(
        capped.dx.is_none(),
        "a cap below the perturbation must withhold the step"
    );
    let msg = capped.error.expect("...and say why");
    assert!(msg.contains("sens_max_pdpert"), "{msg}");

    let above = format!("{:e}", worst * 10.0);
    let allowed = SensSolve::new(vec![1])
        .with_deltas(vec![0.01])
        .run(&mut app_with(&[("sens_max_pdpert", &above)]), rd());
    assert!(
        allowed.dx.is_some(),
        "a cap above the perturbation must let it through: {:?}",
        allowed.error
    );
}
