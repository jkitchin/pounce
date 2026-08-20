//! The barrier / KKT options of #551's "registered, never read" list:
//! `tau_min`, `s_max`, `neg_curv_test_tol` and `neg_curv_test_reg`.
//!
//! Each had somewhere to land — `tau_min` a field on both μ updaters,
//! `s_max` a field on the calculated quantities, `neg_curv_test_tol` a
//! field on `PdFullSpaceSolver` — and no line anywhere that read the
//! option, so setting one did nothing and said nothing.
//!
//! Wiring each is two lines. The point of this file is the other half,
//! which #551 is explicit about:
//!
//!   "A read site that parses a value and discards it is the same silent
//!    no-op this whole line of work exists to kill, and it is
//!    indistinguishable from a real fix by inspection. That test is the
//!    deliverable, not the line that reads the field."
//!
//! So each option is pinned twice: once on the builder (the value
//! survives the parse) and once on a solve (the value moves the
//! algorithm). Directions are asserted only where the mechanism fixes
//! one — a different `tau_min` is not a better `tau_min`, so that test
//! asserts movement, while `s_max`'s effect on the KKT error scaling has
//! a sign and is asserted with it.
//!
//! Every struct default already equalled the registered default, so none
//! of this moves a run that leaves the options alone: the fixture sweep
//! for the wiring commit is empty. Setting them is what changes, which
//! is the whole point.
//!
//! `neg_curv_test_reg` is the one option here whose effect this crate
//! cannot reach end-to-end: the flag only matters when the curvature
//! test runs at a *nonzero* regularization, i.e. when a factorization
//! still shows the wrong inertia after δ_x has already been escalated,
//! which none of the small TNLPs available here provoke. Its solve-level
//! evidence is `crates/pounce-cli/tests/neg_curv_test_options.rs`, which
//! runs a corpus fixture that does.

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::solve_statistics::SolveStatistics;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
struct Hs071 {
    final_x: Option<[Number; 4]>,
    final_obj: Option<Number>,
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
    }
}

/// Five-variable double well:
///
/// ```text
/// min  Σ (−2 xᵢ² + xᵢ⁴)   s.t.  Σ xᵢ = 1,  Σ xᵢ² ≤ 4,  −5 ≤ x ≤ 5
/// ```
///
/// Nonconvex on purpose: the Hessian of the Lagrangian is indefinite at
/// the start, so the solve runs the inertia-correction path that the
/// negative-curvature test replaces — which HS071 never does.
struct DoubleWell;

const DW_N: usize = 5;
const DW_START: [Number; DW_N] = [0.2, 0.3, -0.1, 0.4, 0.25];
/// The local minimum this start converges to, at every setting below.
const DW_OPT: Number = -4.792610610;

impl TNLP for DoubleWell {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: DW_N as i32,
            m: 2,
            nnz_jac_g: (2 * DW_N) as i32,
            nnz_h_lag: DW_N as i32,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l.copy_from_slice(&[-5.0; DW_N]);
        b.x_u.copy_from_slice(&[5.0; DW_N]);
        // g0: Σ xᵢ = 1 (equality), g1: Σ xᵢ² ≤ 4 (inequality ⇒ a slack).
        b.g_l.copy_from_slice(&[1.0, -2.0e19]);
        b.g_u.copy_from_slice(&[1.0, 4.0]);
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&DW_START);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(x.iter().map(|v| -2.0 * v * v + v.powi(4)).sum())
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        for (gi, xi) in g.iter_mut().zip(x) {
            *gi = -4.0 * xi + 4.0 * xi.powi(3);
        }
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x.iter().sum();
        g[1] = x.iter().map(|v| v * v).sum();
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
                for i in 0..DW_N {
                    irow[i] = 0;
                    jcol[i] = i as i32;
                    irow[DW_N + i] = 1;
                    jcol[DW_N + i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_jac_g(Values) without x");
                for i in 0..DW_N {
                    values[i] = 1.0;
                    values[DW_N + i] = 2.0 * x[i];
                }
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
                for i in 0..DW_N {
                    irow[i] = i as i32;
                    jcol[i] = i as i32;
                }
            }
            SparsityRequest::Values { values } => {
                let x = x.expect("eval_h(Values) without x");
                let lam = lambda.expect("eval_h(Values) without lambda");
                for i in 0..DW_N {
                    values[i] = obj_factor * (-4.0 + 12.0 * x[i] * x[i]) + lam[1] * 2.0;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solve_hs071(
    setup: impl FnOnce(&mut IpoptApplication),
) -> (SolveStatistics, ApplicationReturnStatus) {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Hs071::default())) as _;
    let status = app.optimize_tnlp(tnlp);
    (app.statistics(), status)
}

fn solve_double_well(
    setup: impl FnOnce(&mut IpoptApplication),
) -> (SolveStatistics, ApplicationReturnStatus) {
    let mut app = IpoptApplication::new();
    setup(&mut app);
    app.initialize().unwrap();
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(DoubleWell)) as _;
    let status = app.optimize_tnlp(tnlp);
    (app.statistics(), status)
}

fn assert_solved(
    label: &str,
    expected_obj: Number,
    stats: &SolveStatistics,
    status: ApplicationReturnStatus,
) {
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{label}: did not solve: {status:?}",
    );
    assert!(
        (stats.final_objective - expected_obj).abs() < 1e-6,
        "{label}: objective {} drifted from {expected_obj}",
        stats.final_objective,
    );
}

// ---------------------------------------------------------------
// tau_min — floor on the fraction-to-the-boundary parameter
// ---------------------------------------------------------------

/// The value reaches `MuOptions`, which `build_inner` copies onto
/// whichever μ updater `mu_strategy` selects.
#[test]
fn tau_min_reaches_the_builder() {
    let mut app = IpoptApplication::new();
    assert_eq!(
        app.algorithm_builder_from_options().mu.tau_min,
        0.99,
        "the untouched default must stay at upstream's 0.99 — wiring an \
         option must not change what an unset option does",
    );
    app.options_mut()
        .set_numeric_value("tau_min", 0.5, true, false)
        .unwrap();
    assert_eq!(app.algorithm_builder_from_options().mu.tau_min, 0.5);
}

/// And it must move the solve. τ = max(tau_min, 1 − μ) caps how far
/// toward its bound a step may travel, so with μ = 0.1 early on
/// (1 − μ = 0.9 < 0.99) a lower floor lets the fraction-to-the-boundary
/// rule bind differently and the trajectory changes.
///
/// The adaptive strategy is the one asserted here because it re-derives
/// τ after every oracle call, which HS071 is sensitive to; the monotone
/// branch of `build_inner` takes the same field two lines earlier and its
/// effect shows on the fixture corpus (`deb7` 154 → 139 iterations at
/// `tau_min=0.5`, `hs13_bigstart` 29 → 33) rather than on a four-variable
/// problem whose steps are never boundary-limited.
#[test]
fn tau_min_changes_the_trajectory() {
    let run = |tau_min: Option<Number>| {
        solve_hs071(|app| {
            app.options_mut()
                .set_string_value("mu_strategy", "adaptive", true, false)
                .unwrap();
            if let Some(t) = tau_min {
                app.options_mut()
                    .set_numeric_value("tau_min", t, true, false)
                    .unwrap();
            }
        })
    };
    let (default, st_default) = run(None);
    let (lowered, st_low) = run(Some(0.5));
    eprintln!(
        "HS071 adaptive: tau_min default -> {} iters, tau_min=0.5 -> {} iters",
        default.iteration_count, lowered.iteration_count,
    );
    assert_solved("tau_min/default", 17.014017, &default, st_default);
    assert_solved("tau_min=0.5", 17.014017, &lowered, st_low);
    assert_ne!(
        default.iteration_count, lowered.iteration_count,
        "tau_min did not change the trajectory ({} iterations either way) — \
         the option is parsed but is not reaching the μ updater",
        default.iteration_count,
    );
}

// ---------------------------------------------------------------
// s_max — the (s_d, s_c) cap in the KKT error test
// ---------------------------------------------------------------

#[test]
fn s_max_reaches_the_builder() {
    let mut app = IpoptApplication::new();
    assert_eq!(
        app.algorithm_builder_from_options().s_max,
        100.0,
        "the untouched default must stay at upstream's 100",
    );
    app.options_mut()
        .set_numeric_value("s_max", 1e-3, true, false)
        .unwrap();
    assert_eq!(app.algorithm_builder_from_options().s_max, 1e-3);
}

/// `s_max` caps the mean multiplier magnitude that builds `s_d` / `s_c`,
/// the two factors the KKT error test divides its dual and
/// complementarity terms by. Both are exactly 1 whenever the multipliers
/// average below the cap — which is why the default 100 is invisible on
/// a small problem — and grow as `mean / s_max` once the cap is under
/// them.
///
/// So the assertion is not "the number moved" but the mechanism itself:
/// at the default the scaled and unscaled KKT errors coincide, and under
/// a cap far below the multipliers the scaled error drops well below the
/// unscaled one. `final_unscaled_kkt_error` is by definition the same
/// quantity without `s_d` / `s_c`, so the gap between the two *is* the
/// option's effect, and it can only appear if the value is read.
#[test]
fn s_max_scales_the_kkt_error() {
    let (default, st_default) = solve_double_well(|_| {});
    let (capped, st_capped) = solve_double_well(|app| {
        app.options_mut()
            .set_numeric_value("s_max", 1e-3, true, false)
            .unwrap();
    });
    eprintln!(
        "double well: s_max default -> {} iters, kkt {:e} (unscaled {:e}); \
         s_max=1e-3 -> {} iters, kkt {:e} (unscaled {:e})",
        default.iteration_count,
        default.final_kkt_error,
        default.final_unscaled_kkt_error,
        capped.iteration_count,
        capped.final_kkt_error,
        capped.final_unscaled_kkt_error,
    );
    assert_solved("s_max/default", DW_OPT, &default, st_default);
    assert_solved("s_max=1e-3", DW_OPT, &capped, st_capped);

    assert_eq!(
        default.final_kkt_error, default.final_unscaled_kkt_error,
        "at s_max=100 this problem's multipliers are far below the cap, so \
         s_d = s_c = 1 and the two errors must be the same number",
    );
    assert!(
        capped.final_kkt_error < default.final_unscaled_kkt_error / 10.0,
        "s_max did not reach the KKT error scaling: scaled error {:e} against \
         an unscaled {:e}, but a cap of 1e-3 puts s_d/s_c far above 1",
        capped.final_kkt_error,
        capped.final_unscaled_kkt_error,
    );
}

// ---------------------------------------------------------------
// neg_curv_test_tol / neg_curv_test_reg — the inertia-free
// curvature test (Zavala & Chiang 2014)
// ---------------------------------------------------------------

#[test]
fn neg_curv_options_reach_the_builder() {
    let mut app = IpoptApplication::new();
    let d = app.algorithm_builder_from_options().refinement;
    assert_eq!(d.neg_curv_test_tol, 0.0, "default: the heuristic is off");
    assert!(d.neg_curv_test_reg, "default: `yes`, per the registry");

    app.options_mut()
        .set_numeric_value("neg_curv_test_tol", 1e-11, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("neg_curv_test_reg", "no", true, false)
        .unwrap();
    let r = app.algorithm_builder_from_options().refinement;
    assert_eq!(r.neg_curv_test_tol, 1e-11);
    assert!(!r.neg_curv_test_reg);
}

/// A positive `neg_curv_test_tol` turns the inertia check off and puts
/// the curvature test in its place: a factorization whose inertia is
/// wrong is kept anyway when the direction it produced has sufficient
/// positive curvature, and otherwise drives the same primal-regularization
/// escalation a wrong inertia would.
///
/// The double well needs that correction — its Lagrangian Hessian is
/// indefinite at the start — so switching the criterion changes which
/// factorizations are accepted and the solve takes a different path to
/// the same minimum. `1e-11` is the value upstream's own option text
/// recommends, so this is the feature as a user would turn it on, not a
/// contrived setting.
#[test]
fn neg_curv_test_tol_changes_the_trajectory() {
    let (off, st_off) = solve_double_well(|_| {});
    let (on, st_on) = solve_double_well(|app| {
        app.options_mut()
            .set_numeric_value("neg_curv_test_tol", 1e-11, true, false)
            .unwrap();
    });
    eprintln!(
        "double well: neg_curv_test_tol off -> {} iters, 1e-11 -> {} iters",
        off.iteration_count, on.iteration_count,
    );
    assert_solved("neg_curv/off", DW_OPT, &off, st_off);
    assert_solved("neg_curv/1e-11", DW_OPT, &on, st_on);
    assert_ne!(
        off.iteration_count, on.iteration_count,
        "neg_curv_test_tol did not change the trajectory ({} iterations either \
         way) — the option is parsed but is not reaching `PdFullSpaceSolver`",
        off.iteration_count,
    );
}
