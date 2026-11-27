//! Residual-adaptive warm-start recentering — gh#606.
//!
//! Three things are asserted here.
//!
//! 1. **The reconstruction pays.** A caller who hands over a primal
//!    point and no multipliers used to get bound multipliers of
//!    `warm_start_mult_bound_push` — a constant with no relation to
//!    the slacks it is paired against — and equality multipliers of
//!    zero. Under `warm_start_recentering=residual` those blocks are
//!    rebuilt from `μ̂ / slack` and from the stationarity least-squares
//!    solve, and the re-solve is shorter for it.
//! 2. **A stale point is recentered rather than trusted.** A warm
//!    point whose residuals are large gets a looser barrier than
//!    `mu_init` asked for, which is what stops it costing more than a
//!    cold start.
//! 3. **The two `GetWarmStartIterate` flags are refused, not
//!    half-served.** `warm_start_same_structure` and
//!    `warm_start_entire_iterate` parsed and set fields nothing read;
//!    they now fail with a message. Plus the registry invariant that
//!    would have caught that in the first place: every registered
//!    `Warm Start` option is read by the solver or explicitly refused.

use pounce_algorithm::application::IpoptApplication;
use pounce_algorithm::init::warm_start::BlockVerdict;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// HS071 with a settable starting point and settable dual seeds, so a
/// test can hand the solver exactly as much of a warm start as it
/// wants to measure.
#[derive(Default)]
struct Hs071Seeded {
    x0: Option<[Number; 4]>,
    lambda0: Option<Vec<Number>>,
    z_l0: Option<Vec<Number>>,
    z_u0: Option<Vec<Number>>,
    final_x: Option<[Number; 4]>,
    final_obj: Option<Number>,
    final_lambda: Option<Vec<Number>>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
}

impl TNLP for Hs071Seeded {
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
        b.g_l.copy_from_slice(&[25.0, 40.0]);
        b.g_u.copy_from_slice(&[2.0e19, 40.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&self.x0.unwrap_or([1.0, 5.0, 5.0, 1.0]));
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

struct Captured {
    x: [Number; 4],
    lambda: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    mu: Number,
    iters: i32,
}

fn cold_solve() -> Captured {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(Hs071Seeded::default()));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "cold HS071 must solve: {status:?}"
    );
    let b = concrete.borrow();
    Captured {
        x: b.final_x.unwrap(),
        lambda: b.final_lambda.clone().unwrap(),
        z_l: b.final_z_l.clone().unwrap(),
        z_u: b.final_z_u.clone().unwrap(),
        mu: app.statistics().final_mu,
        iters: app.statistics().iteration_count,
    }
}

/// How much of the captured dual state a warm re-solve hands back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seeded {
    /// Everything `TNLP::get_starting_point` can carry.
    All,
    /// The bound multipliers only — `lagrange` left out, the shape a
    /// caller who kept `info["mult_x_L"]` but not `info["mult_g"]`
    /// produces.
    BoundsOnly,
    /// The primal point and nothing else.
    Nothing,
}

/// One warm re-solve of the same model.
fn warm_solve(
    seed: &Captured,
    seeded: Seeded,
    recentering: &str,
    x0: Option<[Number; 4]>,
) -> (
    ApplicationReturnStatus,
    i32,
    Option<pounce_algorithm::init::warm_start::WarmStartDiagnostics>,
) {
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("warm_start_init_point", "yes", true, false)
        .unwrap();
    o.set_string_value("warm_start_recentering", recentering, true, false)
        .unwrap();
    // The recipe `docs/src/initialization.md` prescribes, and the one
    // `pounce.WarmStart` emits: tight pushes plus the captured μ.
    o.set_numeric_value("mu_init", seed.mu.clamp(1e-9, 1e-1), true, false)
        .unwrap();
    for k in [
        "warm_start_bound_push",
        "warm_start_bound_frac",
        "warm_start_slack_bound_push",
        "warm_start_slack_bound_frac",
        "warm_start_mult_bound_push",
    ] {
        o.set_numeric_value(k, 1e-9, true, false).unwrap();
    }
    app.initialize().unwrap();

    let concrete = Rc::new(RefCell::new(Hs071Seeded {
        x0: Some(x0.unwrap_or(seed.x)),
        lambda0: (seeded == Seeded::All).then(|| seed.lambda.clone()),
        z_l0: (seeded != Seeded::Nothing).then(|| seed.z_l.clone()),
        z_u0: (seeded != Seeded::Nothing).then(|| seed.z_u.clone()),
        ..Default::default()
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    (
        status,
        app.statistics().iteration_count,
        app.warm_start_diagnostics(),
    )
}

/// The headline claim, on the block the acceptance criteria are about.
/// An exact same-model restart hands back every multiplier the TNLP
/// surface can carry — but not the *slack*-bound multipliers, which
/// `TNLP::get_starting_point` has no field for and which therefore
/// arrive as zero on every warm start there has ever been. Filling
/// them from `warm_start_mult_bound_push` leaves a stationarity
/// residual the solver then has to work off; reconstructing them from
/// the seeded `y_d` does not.
#[test]
fn an_exact_restart_is_not_degraded_by_the_blocks_the_caller_cannot_seed() {
    let seed = cold_solve();
    let (st_legacy, it_legacy, _) = warm_solve(&seed, Seeded::All, "none", None);
    let (st_resid, it_resid, diag) = warm_solve(&seed, Seeded::All, "residual", None);
    let diag = diag.expect("a warm solve must report diagnostics");

    eprintln!(
        "HS071 exact restart: none={it_legacy} ({st_legacy:?}) \
         residual={it_resid} ({st_resid:?}) cold={} \
         mu {:e} -> {:e} inf_du_after={:e}",
        seed.iters, diag.mu_in, diag.mu_out, diag.dual_residual
    );
    assert!(matches!(st_legacy, ApplicationReturnStatus::SolveSucceeded));
    assert!(matches!(st_resid, ApplicationReturnStatus::SolveSucceeded));

    // The equality/inequality multipliers *were* seeded, so they are
    // kept; the slack-bound block was not, so it is rebuilt. That
    // asymmetry is the whole content of this test.
    assert_eq!(diag.eq_duals, BlockVerdict::Accepted);
    assert_eq!(
        diag.bound_duals,
        BlockVerdict::Reconstructed,
        "v_L/v_U have no TNLP seed field, so they must be rebuilt"
    );
    assert!(diag.bound_duals_reconstructed > 0);

    // Reconstructed from stationarity, the supplied point *is* a KKT
    // point, and the measured dual residual says so.
    assert!(
        diag.dual_residual < 1e-6,
        "reconstruction must leave the point stationary, got {:e}",
        diag.dual_residual
    );
    assert!(
        diag.mu_out <= 1e-6,
        "a KKT-quality point must keep a tight barrier, got mu_out={:e}",
        diag.mu_out
    );
    assert!(
        it_resid < it_legacy,
        "an exact restart must not be degraded by the unseedable block: \
         residual={it_resid} vs none={it_legacy} (cold={})",
        seed.iters
    );
}

/// The acceptance criterion's own case: a *partial* warm start, with
/// the bound multipliers supplied and `lagrange` left out, must beat
/// the zero-filled equality multipliers it used to get.
///
/// This is where the reconstruction is well-posed — the supplied
/// blocks pin the missing one through stationarity — and it is the
/// only place it runs.
#[test]
fn a_partial_warm_start_reconstructs_the_block_it_is_missing() {
    let seed = cold_solve();
    let (st_legacy, it_legacy, _) = warm_solve(&seed, Seeded::BoundsOnly, "none", None);
    let (st_resid, it_resid, diag) = warm_solve(&seed, Seeded::BoundsOnly, "residual", None);
    let diag = diag.expect("a warm solve must report diagnostics");

    eprintln!(
        "HS071 partial (bounds-only) warm start: none={it_legacy} ({st_legacy:?}) \
         residual={it_resid} ({st_resid:?}) cold={} eq_duals={:?} inf_du_after={:e}",
        seed.iters, diag.eq_duals, diag.dual_residual
    );
    assert!(matches!(st_legacy, ApplicationReturnStatus::SolveSucceeded));
    assert!(matches!(st_resid, ApplicationReturnStatus::SolveSucceeded));

    assert_eq!(
        diag.eq_duals,
        BlockVerdict::Reconstructed,
        "an all-zero y block alongside real bound multipliers must go \
         through the stationarity least-squares solve"
    );
    assert_eq!(diag.bound_duals, BlockVerdict::Reconstructed);
    assert!(
        it_resid < it_legacy,
        "completing the seed must cost fewer iterations: residual={it_resid} \
         vs none={it_legacy}"
    );
}

/// A seed with no dual information gets its *bound* blocks filled and
/// nothing else derived (gh#606 as amended by gh#622).
///
/// The two halves are gated differently because they need different
/// things. `z = mu / slack` needs only the slacks the supplied point
/// already determines, so it runs here; the least-squares `y` needs a
/// dual to complete, and from a primal point alone it is the cold
/// path's estimate wearing the warm path's barrier — measured over
/// `benchmarks/warmstart` at 1102 -> 1211 iterations across 27
/// parametric paths when it was allowed to run. So: bound duals
/// reconstructed, equality duals reported unseeded, mu untouched.
#[test]
fn a_primal_only_seed_gets_its_bound_blocks_and_nothing_derived() {
    let seed = cold_solve();
    let (st_legacy, it_legacy, _) = warm_solve(&seed, Seeded::Nothing, "none", None);
    let (st_resid, it_resid, diag) = warm_solve(&seed, Seeded::Nothing, "residual", None);
    let diag = diag.expect("a warm solve must report diagnostics");

    eprintln!(
        "HS071 primal-only warm start: none={it_legacy} ({st_legacy:?}) \
         residual={it_resid} ({st_resid:?}) cold={}",
        seed.iters
    );
    assert!(matches!(st_legacy, ApplicationReturnStatus::SolveSucceeded));
    assert!(matches!(st_resid, ApplicationReturnStatus::SolveSucceeded));
    assert_eq!(diag.bound_duals, BlockVerdict::Reconstructed);
    assert!(
        diag.bound_duals_reconstructed > 0,
        "every bound block arrived unseeded, so every entry is a fill"
    );
    assert_eq!(
        diag.eq_duals,
        BlockVerdict::Unseeded,
        "there is no dual to complete y from; it keeps the constant fill"
    );
    assert!(!diag.stationarity_split);
    assert_eq!(
        diag.mu_in, diag.mu_out,
        "the complementarity of a point this initializer just filled to \
         mu / slack measures mu; escalating off it is the barrier \
         arguing with itself"
    );
    assert!(
        it_resid < it_legacy,
        "filling the bound blocks from the barrier relation must beat \
         the constant: residual={it_resid} vs none={it_legacy}"
    );
}

/// A stale point — the converged duals of this model pinned onto a
/// primal point that is nowhere near feasible — must be recentered.
/// `mu_init` asked for the converged barrier; the measurement
/// overrides it, which is the mechanism that stops a stale warm start
/// costing more than a cold one.
#[test]
fn a_stale_warm_point_is_recentered_above_mu_init() {
    let seed = cold_solve();
    let stale = [5.0, 1.0, 1.0, 5.0];
    let (status, iters, diag) = warm_solve(&seed, Seeded::All, "residual", Some(stale));
    let diag = diag.expect("diagnostics");
    let (st_legacy, it_legacy, _) = warm_solve(&seed, Seeded::All, "none", Some(stale));
    eprintln!(
        "HS071 stale warm point: residual={iters} ({status:?}) none={it_legacy} \
         ({st_legacy:?}) cold={} mu {:e} -> {:e} inf_pr={:e}",
        seed.iters, diag.mu_in, diag.mu_out, diag.primal_residual
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(
        diag.primal_residual > 1.0,
        "the fixture must actually be stale: inf_pr={:e}",
        diag.primal_residual
    );
    assert!(
        diag.mu_out > diag.mu_in,
        "a point this far off must not keep the converged barrier: \
         mu_in={:e} mu_out={:e}",
        diag.mu_in,
        diag.mu_out
    );
}

/// `warm_start_recentering=none` is the kill switch, and it has to be
/// exactly that: with it set, nothing in gh#606 runs.
#[test]
fn recentering_none_leaves_every_block_alone() {
    let seed = cold_solve();
    let (_, _, diag) = warm_solve(&seed, Seeded::Nothing, "none", None);
    let diag = diag.expect("diagnostics");
    assert!(diag.recentering_disabled);
    // Nothing was seeded, and the blocks exist: `Unseeded`, not `Absent`,
    // which is reserved for a block of dimension zero (gh#1269).
    assert_eq!(diag.bound_duals, BlockVerdict::Unseeded);
    assert_eq!(diag.eq_duals, BlockVerdict::Unseeded);
    assert_eq!(diag.bound_duals_reconstructed, 0);
    assert_eq!(
        diag.mu_in, diag.mu_out,
        "μ must be untouched when recentering is off"
    );
}

/// gh#606 scope item 5, the refusal half. Both flags name Ipopt's
/// `TNLP::GetWarmStartIterate`, which pounce does not expose; they used
/// to parse into `WarmStartOptions` fields the initializer never read.
#[test]
fn the_get_warm_start_iterate_flags_are_refused_not_half_served() {
    for name in ["warm_start_same_structure", "warm_start_entire_iterate"] {
        let mut app = IpoptApplication::new();
        assert_eq!(app.unimplemented_option_refusal(), None, "unset: {name}");

        app.options_mut()
            .set_string_value(name, "no", true, false)
            .expect("upstream's default must still parse");
        assert_eq!(
            app.unimplemented_option_refusal(),
            None,
            "`{name}=no` asks for nothing and must keep working"
        );

        app.options_mut()
            .set_string_value(name, "yes", true, false)
            .expect("upstream's value must still parse");
        let msg = app
            .unimplemented_option_refusal()
            .unwrap_or_else(|| panic!("`{name}=yes` must be refused"));
        assert!(msg.contains(name), "{msg}");
        assert!(msg.contains("GetWarmStartIterate"), "{msg}");
        assert!(msg.contains("606"), "message should name the issue: {msg}");
    }
}

/// The invariant that would have caught the two dead flags without
/// anyone noticing them by hand: a registered `Warm Start` option is
/// either read by the solver or explicitly refused. Mirrors
/// `every_registered_initialization_option_is_consumed_or_refused` in
/// `init_options_wiring.rs` (gh#604) one category over.
#[test]
fn every_registered_warm_start_option_is_consumed_or_refused() {
    use pounce_algorithm::unimplemented_options::{UNIMPLEMENTED_FEATURES, UNIMPLEMENTED_VALUES};

    let app = IpoptApplication::new();
    let refused: BTreeSet<&str> = UNIMPLEMENTED_FEATURES
        .iter()
        .flat_map(|g| g.options.iter().copied())
        .chain(UNIMPLEMENTED_VALUES.iter().map(|v| v.option))
        .collect();
    let sources = solver_sources();

    let mut dangling = Vec::new();
    for opt in app.registered_options().registered_options_in_order() {
        if opt.category != "Warm Start" {
            continue;
        }
        if refused.contains(opt.name.as_str()) {
            continue;
        }
        let quoted = format!("\"{}\"", opt.name);
        if sources.iter().any(|s| s.contains(&quoted)) {
            continue;
        }
        dangling.push(opt.name.clone());
    }
    assert!(
        dangling.is_empty(),
        "these Warm Start options are registered but neither read nor \
         refused — setting one does nothing, silently: {dangling:?}"
    );

    let n = app
        .registered_options()
        .registered_options_in_order()
        .iter()
        .filter(|o| o.category == "Warm Start")
        .count();
    assert!(n >= 10, "the category must be non-empty, found {n}");
}

/// Every `.rs` under the solver crates, concatenated, minus the
/// registry itself — the same trick `init_options_wiring.rs` uses to
/// ask "does anything read this name?".
fn solver_sources() -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs")
                && p.file_name().is_some_and(|f| f != "upstream_options.rs")
            {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    out.push(s);
                }
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

// ---------------------------------------------------------------
// gh#606 review regressions.
//
// The model above has one equality row and one inequality row, so
// both `y` blocks are non-empty in every test written against it.
// That is exactly the shape that hid the first defect below, so
// these use a two-variable model with only *one* of the two blocks.
// ---------------------------------------------------------------

/// Which constraint block the single row lands in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Row {
    /// `x0 + x1 == 2` — `y_c` has one entry, `y_d` is empty.
    Equality,
    /// `x0 + x1 <= 2`, active at the optimum — `y_c` is empty.
    Inequality,
}

/// `min (x0−3)² + (x1−3)²` over `0 ≤ x ≤ 10` with a single row, whose
/// optimum sits on that row at `x = (1, 1)`.
struct OneBlock {
    row: Row,
    /// With `free`, `x` has no finite bounds — so the four bound
    /// multiplier blocks are all zero-dimension and there is nothing
    /// for the stationarity split to rebuild.
    free: bool,
    x0: Option<[Number; 2]>,
    lambda0: Option<Vec<Number>>,
    z_l0: Option<Vec<Number>>,
    z_u0: Option<Vec<Number>>,
    final_x: Option<[Number; 2]>,
    final_obj: Option<Number>,
    final_lambda: Option<Vec<Number>>,
    final_z_l: Option<Vec<Number>>,
    final_z_u: Option<Vec<Number>>,
}

impl OneBlock {
    fn new(row: Row) -> Self {
        Self {
            row,
            free: false,
            x0: None,
            lambda0: None,
            z_l0: None,
            z_u0: None,
            final_x: None,
            final_obj: None,
            final_lambda: None,
            final_z_l: None,
            final_z_u: None,
        }
    }
}

impl TNLP for OneBlock {
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
        if self.free {
            b.x_l.copy_from_slice(&[-2.0e19, -2.0e19]);
            b.x_u.copy_from_slice(&[2.0e19, 2.0e19]);
        } else {
            b.x_l.copy_from_slice(&[0.0, 0.0]);
            b.x_u.copy_from_slice(&[10.0, 10.0]);
        }
        match self.row {
            Row::Equality => {
                b.g_l.copy_from_slice(&[2.0]);
                b.g_u.copy_from_slice(&[2.0]);
            }
            Row::Inequality => {
                b.g_l.copy_from_slice(&[-2.0e19]);
                b.g_u.copy_from_slice(&[2.0]);
            }
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        if sp.init_x {
            sp.x.copy_from_slice(&self.x0.unwrap_or([0.5, 0.5]));
        }
        // Left unserved by default: the "bound multipliers kept,
        // `lagrange` dropped" shape. The free-variable fixture is the
        // mirror image and seeds it.
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
        Some((x[0] - 3.0).powi(2) + (x[1] - 3.0).powi(2))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 3.0);
        g[1] = 2.0 * (x[1] - 3.0);
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
                values[0] = 1.0;
                values[1] = 1.0;
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
                // The row is linear, so `lambda` contributes nothing.
                values[0] = 2.0 * obj_factor;
                values[1] = 2.0 * obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        if sol.x.len() == 2 {
            self.final_x = Some([sol.x[0], sol.x[1]]);
        }
        self.final_obj = Some(sol.obj_value);
        self.final_lambda = Some(sol.lambda.to_vec());
        self.final_z_l = Some(sol.z_l.to_vec());
        self.final_z_u = Some(sol.z_u.to_vec());
    }
}

/// Cold-solve a `OneBlock`, then warm-restart it from its own answer
/// with the bound multipliers seeded and `lagrange` left out.
fn one_block_restart(
    row: Row,
    mu_init: Number,
) -> (
    ApplicationReturnStatus,
    pounce_algorithm::init::warm_start::WarmStartDiagnostics,
) {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(OneBlock::new(row)));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    assert!(
        matches!(status, ApplicationReturnStatus::SolveSucceeded),
        "cold OneBlock must solve: {status:?}"
    );
    let (x, z_l, z_u, obj) = {
        let b = concrete.borrow();
        (
            b.final_x.unwrap(),
            b.final_z_l.clone().unwrap(),
            b.final_z_u.clone().unwrap(),
            b.final_obj.unwrap(),
        )
    };
    // The row is active at the optimum in both variants, which is what
    // makes the missing `y` worth reconstructing.
    assert!(
        (x[0] + x[1] - 2.0).abs() < 1e-6,
        "fixture must sit on its row: x={x:?} obj={obj}"
    );

    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("warm_start_init_point", "yes", true, false)
        .unwrap();
    o.set_string_value("warm_start_recentering", "residual", true, false)
        .unwrap();
    o.set_numeric_value("mu_init", mu_init, true, false)
        .unwrap();
    for k in [
        "warm_start_bound_push",
        "warm_start_bound_frac",
        "warm_start_slack_bound_push",
        "warm_start_slack_bound_frac",
        "warm_start_mult_bound_push",
    ] {
        o.set_numeric_value(k, 1e-9, true, false).unwrap();
    }
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(OneBlock {
        x0: Some(x),
        z_l0: Some(z_l),
        z_u0: Some(z_u),
        ..OneBlock::new(row)
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    (
        status,
        app.warm_start_diagnostics()
            .expect("a warm solve must report diagnostics"),
    )
}

/// gh#606 review defect 1. `is_identically_zero` answered `false` for a
/// zero-dimension block, so on a model with only equality rows (or only
/// inequality rows) the `seeded` test short-circuited to `true`, the
/// least-squares reconstruction never ran, and the diagnostics reported
/// `eq_duals: Accepted` for a block nothing had seeded.
///
/// Equality-only and inequality-only models are most of the feature's
/// reach, and `Hs071Seeded` cannot see this: it has one row of each.
#[test]
fn a_single_constraint_block_still_reaches_the_reconstruction() {
    for row in [Row::Equality, Row::Inequality] {
        let (status, diag) = one_block_restart(row, 1e-7);
        let which = if row == Row::Equality { "eq" } else { "ineq" };
        eprintln!(
            "OneBlock {which}-only: status={status:?} eq_duals={:?} \
             bound_duals={:?} reconstructed={} inf_du={:e} split={}",
            diag.eq_duals,
            diag.bound_duals,
            diag.bound_duals_reconstructed,
            diag.dual_residual,
            diag.stationarity_split
        );
        assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
        assert_eq!(
            diag.eq_duals,
            BlockVerdict::Reconstructed,
            "{which}-only model: the y block was never seeded, so it must \
             be rebuilt rather than reported as kept"
        );
    }
}

/// gh#606 review defect 2. `final_mu` applied the `[MU_FLOOR,
/// MU_CEILING]` clamp to the pass-through value as well as to the
/// escalated one, so an explicit `mu_init` above the ceiling was
/// silently lowered on every warm start — and silently, because the
/// `wmu` info token only fires when μ goes *up*.
///
/// The function's own contract is "`mu_in` is a floor, never a
/// ceiling".
#[test]
fn an_explicit_mu_init_above_the_ceiling_survives_when_nothing_escalates() {
    // 1.0 is ten times `MU_CEILING`. A restart from the model's own
    // answer measures a complementarity far below `10 · mu_in`, so no
    // escalation fires and μ must come out exactly as it went in.
    let (status, diag) = one_block_restart(Row::Equality, 1.0);
    eprintln!(
        "OneBlock mu pass-through: status={status:?} mu {:e} -> {:e} compl={:e}",
        diag.mu_in, diag.mu_out, diag.complementarity
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(
        diag.complementarity <= 10.0 * diag.mu_in,
        "fixture must not escalate, or it tests the wrong branch: \
         compl={:e} mu_in={:e}",
        diag.complementarity,
        diag.mu_in
    );
    assert_eq!(
        diag.mu_out, diag.mu_in,
        "an explicit mu_init the caller chose must not be capped: \
         mu_in={:e} mu_out={:e}",
        diag.mu_in, diag.mu_out
    );
}

/// gh#606 review, the cosmetic one: `stationarity_split` was set from
/// the fact that the split was *called*, not from whether it did
/// anything. It returns early when there is no unseeded bound
/// multiplier to rebuild, and claiming otherwise made
/// `info["warm_start"]` untrue.
///
/// Reaching that early return takes a model with **no bound
/// multipliers at all** and a seeded `y`: free variables make the four
/// `z`/`v` blocks zero-dimension, and a supplied `lagrange` keeps
/// `eq_duals` at `Accepted` so the split is still called. On HS071 it
/// is unreachable — the slack-bound block is never seedable, so
/// something is always rebuilt there and the flag was accidentally
/// right.
#[test]
fn stationarity_split_is_reported_only_when_it_runs() {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(OneBlock {
        free: true,
        ..OneBlock::new(Row::Equality)
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    assert!(matches!(
        app.optimize_tnlp(tnlp),
        ApplicationReturnStatus::SolveSucceeded
    ));
    let (x, lambda) = {
        let b = concrete.borrow();
        (b.final_x.unwrap(), b.final_lambda.clone().unwrap())
    };

    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("warm_start_init_point", "yes", true, false)
        .unwrap();
    o.set_string_value("warm_start_recentering", "residual", true, false)
        .unwrap();
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(OneBlock {
        free: true,
        x0: Some(x),
        lambda0: Some(lambda),
        ..OneBlock::new(Row::Equality)
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    let diag = app.warm_start_diagnostics().expect("diagnostics");
    eprintln!(
        "OneBlock free+seeded-y: status={status:?} bound_duals={:?} \
         eq_duals={:?} reconstructed={} split={}",
        diag.bound_duals, diag.eq_duals, diag.bound_duals_reconstructed, diag.stationarity_split
    );
    assert_eq!(
        diag.bound_duals_reconstructed, 0,
        "fixture must have no bound multiplier to rebuild, or it tests \
         the wrong branch"
    );
    assert!(
        !diag.stationarity_split,
        "the flag must track the work, not the call site"
    );
}

/// A warm start that carries only a point must not be charged for
/// multipliers it never claimed to have (gh#622).
///
/// `TNLP::get_starting_point` is asked for `init_z` and may leave the
/// blocks untouched; the buffer behind them used to arrive as a
/// literal zero, which is a *legal multiplier value* and so skipped
/// the unseeded resolution and was merely floored at
/// `warm_start_mult_bound_push`. Under the tightened pushes
/// `pounce.WarmStart` ships that is 1e-9 — a start declaring every
/// bound inactive, with complementarity broken against μ before the
/// first iteration. It cost this fixture 11 iterations against 7, and
/// the tighter the pushes the worse it got, which is the opposite of
/// what a push is for.
///
/// Three statements, in the order they pin the fix: an absent block is
/// the same thing as an explicitly unseeded one; tightening the pushes
/// does not move a start that seeded no multipliers; and the switch is
/// not a penalty box relative to a plain solve from the same point.
#[test]
fn a_values_only_warm_start_is_not_taxed_by_the_tightened_pushes() {
    let seed = cold_solve();
    // A transferred point: the solution nudged, as a horizon shift
    // would leave it. Every arm below starts from exactly this x.
    let x0 = [
        seed.x[0] * 1.05,
        seed.x[1] * 0.95,
        seed.x[2] * 1.02,
        seed.x[3] * 0.98,
    ];
    let (st_cold, it_cold, _) = values_only_solve(x0, Switch::Off, Pushes::Default);
    let (st_tight, it_tight, diag) = values_only_solve(x0, Switch::On, Pushes::Tight);
    let (_, it_loose, _) = values_only_solve(x0, Switch::On, Pushes::Default);
    let (_, it_nan, _) = values_only_solve(x0, Switch::OnSeedingNan, Pushes::Tight);

    eprintln!(
        "HS071 values-only warm start: cold={it_cold} ({st_cold:?}) \
         switch+tight={it_tight} ({st_tight:?}) switch+default={it_loose} \
         switch+tight, NaN-seeded={it_nan}"
    );
    assert!(matches!(st_cold, ApplicationReturnStatus::SolveSucceeded));
    assert!(matches!(st_tight, ApplicationReturnStatus::SolveSucceeded));

    assert_eq!(
        it_tight, it_nan,
        "a block the caller never wrote and one written as NaN are the \
         same statement — 'I have no multipliers for you'"
    );
    assert_eq!(
        it_tight, it_loose,
        "with no multipliers seeded there is nothing for the multiplier \
         push to push, so tightening it must not reroute the solve"
    );
    assert!(
        it_tight <= it_cold + 2,
        "the warm-start switch must not turn into a penalty box for a \
         caller who only has a point: switch={it_tight} cold={it_cold} \
         (it was 11 against 6 before gh#622)"
    );

    // The other half of the fix: the blocks are filled from the
    // barrier relation rather than from a push, and the measurement
    // taken against that fill is reported but never allowed to move mu.
    let diag = diag.expect("a warm solve must report diagnostics");
    assert_eq!(diag.bound_duals, BlockVerdict::Reconstructed);
    assert_eq!(
        diag.mu_in, diag.mu_out,
        "a point this initializer just filled to mu / slack measures \
         mu; it must not escalate off its own fill"
    );
}

/// Is `warm_start_init_point` on, and does the caller mark its (absent)
/// multipliers explicitly?
#[derive(Clone, Copy, PartialEq, Eq)]
enum Switch {
    Off,
    On,
    /// The switch, plus `z_l`/`z_u` written as all-NaN — the explicit
    /// spelling of what `On` leaves implicit.
    OnSeedingNan,
}

/// `warm_start_*_push` at the defaults, or at the 1e-9 the
/// `pounce.WarmStart` recipe sets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pushes {
    Default,
    Tight,
}

fn values_only_solve(
    x0: [Number; 4],
    switch: Switch,
    pushes: Pushes,
) -> (
    ApplicationReturnStatus,
    i32,
    Option<pounce_algorithm::init::warm_start::WarmStartDiagnostics>,
) {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        if switch != Switch::Off {
            o.set_string_value("warm_start_init_point", "yes", true, false)
                .unwrap();
        }
        if pushes == Pushes::Tight {
            for k in [
                "warm_start_bound_push",
                "warm_start_bound_frac",
                "warm_start_slack_bound_push",
                "warm_start_slack_bound_frac",
                "warm_start_mult_bound_push",
            ] {
                o.set_numeric_value(k, 1e-9, true, false).unwrap();
            }
        }
    }
    app.initialize().unwrap();
    let nan = vec![Number::NAN; 4];
    let concrete = Rc::new(RefCell::new(Hs071Seeded {
        x0: Some(x0),
        z_l0: (switch == Switch::OnSeedingNan).then(|| nan.clone()),
        z_u0: (switch == Switch::OnSeedingNan).then(|| nan.clone()),
        ..Default::default()
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    (
        status,
        app.statistics().iteration_count,
        app.warm_start_diagnostics(),
    )
}

// ---------------------------------------------------------------------
// gh#617 / gh#618 — what the recentering pass does with a seed it should
// not believe.
// ---------------------------------------------------------------------

/// gh#617's regime, deterministically: the exact primal solution paired
/// with multipliers that carry no signal.
///
/// The issue draws the noise from `N(0, 1e4)`; a fixed alternating
/// `±100` is the same magnitude with no RNG in a regression test, and
/// the alternation matters — a block that is half negative is what made
/// a signed average of `z · s` cancel to nothing and hide the
/// corruption from a mean-based test.
fn corrupt(seed: &Captured) -> Captured {
    let flip = |n: usize, first: Number| -> Vec<Number> {
        (0..n)
            .map(|i| if i % 2 == 0 { first } else { -first })
            .collect()
    };
    Captured {
        x: seed.x,
        lambda: flip(seed.lambda.len(), 100.0),
        z_l: flip(seed.z_l.len(), 100.0),
        z_u: flip(seed.z_u.len(), -100.0),
        mu: seed.mu,
        iters: seed.iters,
    }
}

/// gh#617, the behavioural claim, and the one that does not need any of
/// this change's new API to state: a dual seed that cannot belong to
/// the primal point it arrived with must not drag μ up to the cold
/// ceiling.
///
/// Before this change the corrupted multipliers were taken at face
/// value, `avrg_compl` measured them, and the escalation in
/// [`final_mu`] moved μ eight orders — from the converged barrier to
/// `MU_CEILING` — while keeping `x` at the exact solution, which is the
/// most off-centre a point can be for that barrier. The result was a
/// warm start that cost about what a cold solve costs, against the
/// three iterations the pre-gh#606 constants happened to get.
#[test]
fn a_corrupted_dual_seed_does_not_escalate_mu_to_the_ceiling() {
    let seed = cold_solve();
    let bad = corrupt(&seed);
    let (status, iters, diag) = warm_solve(&bad, Seeded::All, "residual", None);
    let diag = diag.expect("diagnostics");
    let (st_legacy, it_legacy, _) = warm_solve(&bad, Seeded::All, "none", None);
    eprintln!(
        "HS071 corrupted seed: residual={iters} ({status:?}) none={it_legacy} \
         ({st_legacy:?}) cold={} mu {:e} -> {:e} inf_pr={:e} compl={:e}",
        seed.iters, diag.mu_in, diag.mu_out, diag.primal_residual, diag.complementarity
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    // The fixture has to actually be the regime the issue describes:
    // the primal point is the answer, so nothing about it is stale.
    assert!(
        diag.primal_residual < 1e-6,
        "the primal seed must be exact: inf_pr={:e}",
        diag.primal_residual
    );
    assert!(
        diag.mu_out <= diag.mu_in,
        "a refused dual seed must not move the barrier: mu_in={:e} mu_out={:e}",
        diag.mu_in,
        diag.mu_out
    );
    assert!(
        iters <= it_legacy,
        "reconstructing off a corrupted seed must not cost more than the \
         pre-gh#606 constants it replaced: residual={iters} none={it_legacy}"
    );
    assert!(
        iters < seed.iters,
        "an exact primal point must still beat a cold solve however bad its \
         duals are: warm={iters} cold={}",
        seed.iters
    );
}

/// gh#617, the reporting half. Needs the `Rejected` verdict and the two
/// new counters, so unlike the test above it cannot be compiled against
/// the parent commit at all — it pins the diagnostics, not a behaviour.
#[test]
fn a_refused_dual_seed_says_so_in_the_diagnostics() {
    let seed = cold_solve();
    let bad = corrupt(&seed);
    let (_status, _iters, diag) = warm_solve(&bad, Seeded::All, "residual", None);
    let diag = diag.expect("diagnostics");
    assert_eq!(
        diag.bound_duals,
        BlockVerdict::Rejected,
        "a refused block is the loudest verdict and must win the summary"
    );
    assert!(
        diag.bound_duals_rejected > 0,
        "the count must say how much was refused"
    );
    // The verdicts are per block, and HS071 carries both kinds at
    // once: the two seeded variable-bound blocks are refused, while the
    // slack-bound blocks no frontend can seed are still rebuilt from
    // `μ̂ / slack` as gh#606 intends. The summary reports the refusal
    // because that is the one a caller has to act on; the counters keep
    // both numbers so the mixture stays legible.
    assert!(
        diag.bound_duals_reconstructed > 0,
        "the blocks nobody seeded are still reconstructed"
    );
    // A refused `z` leaves nothing for the split to stand on, and the
    // split is what would launder the corruption back in.
    assert!(!diag.stationarity_split);
}

/// gh#617, the other block. A caller who keeps `info["mult_g"]` but not
/// the bound multipliers is the common partial-seed shape, and a
/// corrupted `y` reaches the reconstruction through a different door:
/// there is no seeded `z` to refuse, so the damage is the stationarity
/// split turning `y`'s miss into bound multipliers.
#[test]
fn a_corrupted_lagrange_seed_is_not_split_into_bound_multipliers() {
    let seed = cold_solve();
    let bad = Captured {
        lambda: vec![100.0, -100.0],
        ..corrupt(&seed)
    };
    // `BoundsOnly` inverted: seed `lagrange` and leave `z` out. The
    // helper's three modes do not cover it, so drive it directly.
    let (status, iters, diag) = warm_solve_lagrange_only(&bad, "residual");
    let (_st, it_legacy, _) = warm_solve_lagrange_only(&bad, "none");
    let diag = diag.expect("diagnostics");
    eprintln!(
        "HS071 corrupted lagrange: residual={iters} none={it_legacy} \
         mu {:e} -> {:e} compl={:e} split={}",
        diag.mu_in, diag.mu_out, diag.complementarity, diag.stationarity_split
    );
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(
        diag.mu_out <= diag.mu_in,
        "mu_in={:e} mu_out={:e}",
        diag.mu_in,
        diag.mu_out
    );
    assert!(iters <= it_legacy, "residual={iters} none={it_legacy}");
}

/// `warm_solve` with `lagrange` seeded and the bound multipliers left
/// out — the mirror image of [`Seeded::BoundsOnly`].
fn warm_solve_lagrange_only(
    seed: &Captured,
    recentering: &str,
) -> (
    ApplicationReturnStatus,
    i32,
    Option<pounce_algorithm::init::warm_start::WarmStartDiagnostics>,
) {
    let mut app = IpoptApplication::new();
    let o = app.options_mut();
    o.set_integer_value("print_level", 0, true, false).unwrap();
    o.set_string_value("warm_start_init_point", "yes", true, false)
        .unwrap();
    o.set_string_value("warm_start_recentering", recentering, true, false)
        .unwrap();
    o.set_numeric_value("mu_init", seed.mu.clamp(1e-9, 1e-1), true, false)
        .unwrap();
    for k in [
        "warm_start_bound_push",
        "warm_start_bound_frac",
        "warm_start_slack_bound_push",
        "warm_start_slack_bound_frac",
        "warm_start_mult_bound_push",
    ] {
        o.set_numeric_value(k, 1e-9, true, false).unwrap();
    }
    app.initialize().unwrap();
    let concrete = Rc::new(RefCell::new(Hs071Seeded {
        x0: Some(seed.x),
        lambda0: Some(seed.lambda.clone()),
        ..Default::default()
    }));
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::clone(&concrete) as _;
    let status = app.optimize_tnlp(tnlp);
    (
        status,
        app.statistics().iteration_count,
        app.warm_start_diagnostics(),
    )
}

/// gh#618. A good seed must be untouched by everything above: the whole
/// point of the conservatism is that an exact restart's numbers are the
/// ones gh#606 earned and this change may not spend them.
///
/// Stated as an equality against the parent's own behaviour rather than
/// a bound, because "no worse" would pass on a change that quietly
/// rerouted the trajectory somewhere else that happened to cost the
/// same.
#[test]
fn an_exact_restart_is_unchanged_by_the_seed_guards() {
    let seed = cold_solve();
    let (status, iters, diag) = warm_solve(&seed, Seeded::All, "residual", None);
    let diag = diag.expect("diagnostics");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    // Nothing was refused, and the reconstruction still ran.
    assert_ne!(diag.bound_duals, BlockVerdict::Rejected);
    assert_eq!(diag.bound_duals_rejected, 0);
    assert!(!diag.eq_duals_rejected);
    assert!(diag.bound_duals_reconstructed > 0);
    assert!(diag.stationarity_split, "the split earns gh#606's win here");
    let (_st, it_legacy, _) = warm_solve(&seed, Seeded::All, "none", None);
    assert!(
        iters < it_legacy,
        "the exact restart must still beat the constants: residual={iters} \
         none={it_legacy}"
    );
}

/// gh#618. The stale point gh#606 built the escalation for still gets
/// it. The staleness fallback added here works on the *reconstruction*,
/// not on μ, and the two must not be confused: a point this far out
/// wants the loose barrier.
#[test]
fn a_strongly_stale_point_still_gets_its_looser_barrier() {
    let seed = cold_solve();
    let stale = [5.0, 1.0, 1.0, 5.0];
    let (status, _iters, diag) = warm_solve(&seed, Seeded::All, "residual", Some(stale));
    let diag = diag.expect("diagnostics");
    assert!(matches!(status, ApplicationReturnStatus::SolveSucceeded));
    assert!(diag.primal_residual > 1.0);
    assert!(
        diag.mu_out > diag.mu_in,
        "mu_in={:e} mu_out={:e}",
        diag.mu_in,
        diag.mu_out
    );
    // And the seed was not mistaken for a corrupted one: a point that
    // misses feasibility by `inf_pr` is *allowed* complementarity of
    // that order, which is the `max(mu_hat, inf_pr)` half of the
    // rejection test.
    assert_ne!(diag.bound_duals, BlockVerdict::Rejected);
}
