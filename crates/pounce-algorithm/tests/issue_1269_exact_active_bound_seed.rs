//! gh#1269 — an exact warm start must not have its active-bound
//! multipliers refused, and `warm_start_recentering=none` must say what it
//! did with the seeds.
//!
//! The model is the issue's own:
//!
//! ```text
//! min 1000((x0-2)^2 + (x1-2)^2)  s.t.  x0 + x1 = 2,  -10 <= x0 <= 0.5,  0 <= x1 <= 10
//! ```
//!
//! with optimum `(0.5, 1.5)`, `y = 1000` and `z_U[0] = 2000` at the active
//! upper bound. Restarted from its own solution under the recipe
//! `pounce.WarmStart` emits (`mu_init` = the captured μ clamped to `1e-9`,
//! every push `1e-9`), the gh#617 coherence test read the active bound's
//! `z · s` — the exact multiplier times a slack the *push* had set to
//! `1e-9` — as `6.7e-8` against `μ̂ = 1e-9`, refused the block, and the
//! restart started at `inf_du = 66.7` and took 4 iterations where the same
//! seeds taken as-is need 1-2.
//!
//! The rule under test branches on whether the push moved an entry, so
//! both branches have a fixture: the exact seed (the active entry, pushed,
//! must not be judged) and corrupted seeds whose bad entry the push did
//! not move (must still be refused) — including a tight box with the
//! default pushes, where a slack-size window would have excused the whole
//! box. The review of the first version added three more: a negative
//! active-bound multiplier, a pushed multiplier large enough to drag μ to
//! the ceiling, and a pushed multiplier large enough to vouch for a
//! corrupted `y`.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::init::warm_start::{BlockVerdict, WarmStartDiagnostics};
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// The issue's model, generalized: `min w((x0-c0)^2 + (x1-c1)^2)` s.t.
/// `x0 + x1 = rhs` inside a box. [`ActiveBound::issue`] is the issue's
/// instance.
struct ActiveBound {
    weight: Number,
    center: [Number; 2],
    rhs: Number,
    lb: [Number; 2],
    ub: [Number; 2],
    x0: Option<[Number; 2]>,
    lambda0: Option<Vec<Number>>,
    z_l0: Option<Vec<Number>>,
    z_u0: Option<Vec<Number>>,
    final_x: Option<[Number; 2]>,
    final_lambda: Option<Vec<Number>>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
}

impl ActiveBound {
    fn issue() -> Self {
        Self::with(1000.0, [2.0, 2.0], 2.0, [-10.0, 0.0], [0.5, 10.0])
    }

    fn with(
        weight: Number,
        center: [Number; 2],
        rhs: Number,
        lb: [Number; 2],
        ub: [Number; 2],
    ) -> Self {
        Self {
            weight,
            center,
            rhs,
            lb,
            ub,
            x0: None,
            lambda0: None,
            z_l0: None,
            z_u0: None,
            final_x: None,
            final_lambda: None,
            final_z_l: None,
            final_z_u: None,
        }
    }
}

impl TNLP for ActiveBound {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 1,
            nnz_jac_g: 2,
            nnz_h_lag: 2,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&self.lb);
        b.x_u.copy_from_slice(&self.ub);
        b.g_l.copy_from_slice(&[self.rhs]);
        b.g_u.copy_from_slice(&[self.rhs]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            let mid = [
                0.5 * (self.lb[0] + self.ub[0]),
                0.5 * (self.lb[1] + self.ub[1]),
            ];
            sp.x.copy_from_slice(&self.x0.unwrap_or(mid));
        }
        if sp.init_lambda {
            if let Some(l) = &self.lambda0 {
                sp.lambda.copy_from_slice(l);
            }
        }
        if sp.init_z {
            if let Some(z) = &self.z_l0 {
                sp.z_l.copy_from_slice(z);
            }
            if let Some(z) = &self.z_u0 {
                sp.z_u.copy_from_slice(z);
            }
        }
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (w, c) = (self.weight, self.center);
        Some(w * ((x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2)))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (w, c) = (self.weight, self.center);
        g[0] = 2.0 * w * (x[0] - c[0]);
        g[1] = 2.0 * w * (x[1] - c[1]);
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0] + x[1];
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
                irow.copy_from_slice(&[0, 0]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0]);
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
                irow.copy_from_slice(&[0, 1]);
                jcol.copy_from_slice(&[0, 1]);
            }
            SparsityRequest::Values { values } => {
                let h = 2.0 * self.weight * obj_factor;
                values.copy_from_slice(&[h, h]);
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        self.final_x = Some([sol.x[0], sol.x[1]]);
        self.final_lambda = Some(sol.lambda.to_vec());
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
    }
}

struct Captured {
    x: [Number; 2],
    lambda: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    mu: Number,
}

/// Solver options for one fixture: the issue's are `tol = 1e-10` and the
/// default scaling; the large-multiplier fixture turns scaling off so the
/// multiplier reaches the solver at its own size.
#[derive(Clone, Copy)]
struct Setup {
    model: fn() -> ActiveBound,
    unscaled: bool,
    /// `warm_start_*` push value; `None` keeps the registered defaults.
    push: Option<Number>,
}

const ISSUE: Setup = Setup {
    model: ActiveBound::issue,
    unscaled: false,
    push: Some(1e-9),
};

fn quiet(app: &mut IpoptApplication, setup: Setup) {
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_numeric_value("tol", 1e-10, true, false).unwrap();
    if setup.unscaled {
        o.set_string_value("nlp_scaling_method", "none", true, false)
            .unwrap();
    }
}

fn cold_solve() -> Captured {
    let c = cold_solve_of(ISSUE);
    // The fixture has to be the regime the issue describes: an active
    // upper bound with a multiplier far above the converged barrier.
    assert!((c.x[0] - 0.5).abs() < 1e-6, "x = {:?}", c.x);
    assert!((c.z_u[0] - 2000.0).abs() < 1e-3, "z_U = {:?}", c.z_u);
    c
}

fn cold_solve_of(setup: Setup) -> Captured {
    let mut app = IpoptApplication::new();
    quiet(&mut app, setup);
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new((setup.model)()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    let b = concrete.borrow();
    Captured {
        x: b.final_x.unwrap(),
        lambda: b.final_lambda.clone().unwrap(),
        z_l: b.final_z_l.clone().unwrap(),
        z_u: b.final_z_u.clone().unwrap(),
        mu: app.statistics().final_mu,
    }
}

/// One warm re-solve under the recipe `pounce.WarmStart.from_info` emits.
fn warm_solve(
    seed: &Captured,
    recentering: &str,
) -> (ApplicationReturnStatus, i32, WarmStartDiagnostics) {
    warm_solve_of(ISSUE, seed, recentering)
}

fn warm_solve_of(
    setup: Setup,
    seed: &Captured,
    recentering: &str,
) -> (ApplicationReturnStatus, i32, WarmStartDiagnostics) {
    let mut app = IpoptApplication::new();
    quiet(&mut app, setup);
    let o = app.options_mut();
    o.set_string_value("warm_start_init_point", "yes", true, false)
        .unwrap();
    o.set_string_value("warm_start_recentering", recentering, true, false)
        .unwrap();
    o.set_numeric_value("mu_init", seed.mu.clamp(1e-9, 1e-1), true, false)
        .unwrap();
    if let Some(push) = setup.push {
        for k in [
            "warm_start_bound_push",
            "warm_start_bound_frac",
            "warm_start_slack_bound_push",
            "warm_start_slack_bound_frac",
            "warm_start_mult_bound_push",
        ] {
            o.set_numeric_value(k, push, true, false).unwrap();
        }
    }
    app.initialize().unwrap();
    let mut model = (setup.model)();
    model.x0 = Some(seed.x);
    model.lambda0 = Some(seed.lambda.clone());
    model.z_l0 = Some(seed.z_l.clone());
    model.z_u0 = Some(seed.z_u.clone());
    let concrete = Rc::new(RefCell::new(model));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    (
        status,
        app.statistics().iteration_count,
        app.warm_start_diagnostics()
            .expect("a warm solve must report diagnostics"),
    )
}

/// The headline: the exact multipliers of an exact restart are kept, the
/// restart starts stationary, and it is no slower than the same seeds
/// taken unmeasured.
#[test]
fn exact_active_bound_multipliers_are_not_refused() {
    let seed = cold_solve();
    let (status, iters, diag) = warm_solve(&seed, "residual");
    let (st_none, it_none, _) = warm_solve(&seed, "none");
    eprintln!(
        "gh#1269 exact restart: residual={iters} ({status:?}) none={it_none} ({st_none:?}) \
         bound_duals={:?} rejected={} inf_du={:e} mu {:e} -> {:e}",
        diag.bound_duals, diag.bound_duals_rejected, diag.dual_residual, diag.mu_in, diag.mu_out
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(matches!(st_none, ApplicationReturnStatus::SolveSucceeded));
    assert_ne!(diag.bound_duals, BlockVerdict::Rejected);
    assert_eq!(diag.bound_duals_rejected, 0);
    assert!(!diag.eq_duals_rejected);
    assert!(
        diag.dual_residual < 1e-6,
        "kept exact multipliers leave the start stationary, got inf_du={:e} \
         (66.7 is the dropped z_U)",
        diag.dual_residual
    );
    assert!(
        iters <= it_none,
        "an exact restart must not cost more than its seeds taken as-is: \
         residual={iters} none={it_none}"
    );
}

/// The other branch of the push-floor exclusion: a corrupted multiplier at
/// the *inactive* bound, whose slack the point really determines (8.5),
/// is still refused. Without this the exclusion could have switched the
/// gh#617 guard off wholesale and the test above would stay green.
#[test]
fn a_corrupted_inactive_bound_multiplier_is_still_refused() {
    let mut seed = cold_solve();
    seed.z_u[1] = 100.0;
    let (status, _iters, diag) = warm_solve(&seed, "residual");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert_eq!(
        diag.bound_duals,
        BlockVerdict::Rejected,
        "z_U[1] = 100 at slack 8.5 implies complementarity 850 at a point \
         feasible to round-off"
    );
    assert!(diag.bound_duals_rejected > 0);
}

/// Part 2 of the issue: under the kill switch the seeds are used, and the
/// report has to say so rather than `absent`, with the start's residuals
/// measured even though nothing acts on them.
#[test]
fn recentering_none_reports_the_seeds_it_used() {
    let seed = cold_solve();
    let (status, _iters, diag) = warm_solve(&seed, "none");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(diag.recentering_disabled);
    assert_eq!(diag.bound_duals, BlockVerdict::Used);
    assert_eq!(diag.eq_duals, BlockVerdict::Used);
    // Not measured: measuring would cost the kill switch a round of
    // model evaluations it never made before.
    assert!(diag.dual_residual.is_nan());
    assert!(diag.complementarity.is_nan());
    assert!(diag.primal_residual.is_nan());
    assert_eq!(diag.mu_in, diag.mu_out, "the kill switch must not move mu");
}

/// Review item: the skip is by what the push did, not by slack size. A
/// box `[1000, 1001]` under the *default* pushes (`1e-3`) has
/// `push · max(1,|b|) = 1`, so a slack-size window of twice that covered
/// the whole box and excused a corrupted multiplier at an inactive bound.
/// The optimum sits mid-box, the push moves nothing, and the corruption
/// must be refused.
#[test]
fn a_tight_box_under_default_pushes_still_checks_its_inactive_bounds() {
    fn mid_box() -> ActiveBound {
        ActiveBound::with(1.0, [1000.5, 1.5], 1002.0, [1000.0, 0.0], [1001.0, 10.0])
    }
    let setup = Setup {
        model: mid_box,
        unscaled: false,
        push: None,
    };
    let mut seed = cold_solve_of(setup);
    assert!((seed.x[0] - 1000.5).abs() < 1e-6, "x = {:?}", seed.x);
    seed.z_l[0] = 100.0;
    seed.z_u[0] = 100.0;
    let (status, _iters, diag) = warm_solve_of(setup, &seed, "residual");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert_eq!(diag.bound_duals, BlockVerdict::Rejected);
}

/// Review item: a negative multiplier at an active bound. The push moves
/// that entry, so it is skipped by the complementarity average — but no
/// barrier multiplier can be negative, and the sign alone refuses it.
#[test]
fn a_negative_active_bound_multiplier_is_refused() {
    let mut seed = cold_solve();
    seed.z_u[0] = -2000.0;
    let (status, _iters, diag) = warm_solve(&seed, "residual");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert_eq!(diag.bound_duals, BlockVerdict::Rejected);
}

/// Review item: a kept block with a large active-bound multiplier must not
/// lift μ through `avrg_compl`. Unscaled, `z_U` is `2e9`, the order the
/// issue's NMPC model reports; `warm_start_mult_init_max` caps it at `1e6`,
/// so `z_U · push` is `1e-3` and the iterate's `avrg_compl` is `2.5e-4`.
/// Read at face value that is a stale-point measurement five orders above
/// `mu_init`, and escalates μ to it.
#[test]
fn a_large_pushed_multiplier_does_not_restart_at_a_cold_barrier() {
    fn big() -> ActiveBound {
        ActiveBound::with(1e9, [2.0, 2.0], 2.0, [-10.0, 0.0], [0.5, 10.0])
    }
    let setup = Setup {
        model: big,
        unscaled: true,
        push: Some(1e-9),
    };
    let seed = cold_solve_of(setup);
    assert!(seed.z_u[0] > 1e9, "z_U = {:?}", seed.z_u);
    let (status, iters, diag) = warm_solve_of(setup, &seed, "residual");
    let (_st, it_none, _) = warm_solve_of(setup, &seed, "none");
    eprintln!(
        "gh#1269 large z: residual={iters} none={it_none} bound_duals={:?} \
         compl={:e} mu {:e} -> {:e}",
        diag.bound_duals, diag.complementarity, diag.mu_in, diag.mu_out
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert_ne!(diag.bound_duals, BlockVerdict::Rejected);
    assert!(
        diag.complementarity > 1e3 * diag.mu_in,
        "the fixture must carry a large pushed z*s, got {:e}",
        diag.complementarity
    );
    assert!(
        diag.mu_out <= 10.0 * diag.mu_in,
        "a pushed z*s must not escalate mu: {:e} -> {:e}",
        diag.mu_in,
        diag.mu_out
    );
    assert!(iters <= it_none, "residual={iters} none={it_none}");
}

/// Review item: an unjudged (pushed) multiplier must not set the scale the
/// `y` check measures against. `z_U[0] = 1e8` at the pushed active bound,
/// and a `y` of `1e6` that misses stationarity by about that much: against
/// a scale of `1e8` the `y` passes, against the gradient (`3e3`) it is
/// refused. `z_L` is left unseeded so the stationarity split, and with it
/// the `y` check, runs.
#[test]
fn a_pushed_multiplier_does_not_vouch_for_a_corrupted_lagrange_seed() {
    let mut seed = cold_solve();
    seed.z_u[0] = 1e8;
    seed.z_l = vec![0.0, 0.0];
    seed.lambda = vec![1e6];
    let (status, _iters, diag) = warm_solve(&seed, "residual");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(
        diag.eq_duals_rejected,
        "y = 1e6 against grad f ~ 3e3 must be refused as an input"
    );
}
