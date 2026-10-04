//! gh#794 P1: the ℓ₁ penalty-barrier wrapper must report the *user's*
//! constraint violation and judge the exit by the *caller's* tolerances.
//!
//! The wrapper solves an augmented problem, `c(x) − p + n = target` with
//! `p, n ≥ 0`, whose equality rows the slacks satisfy to machine precision
//! by construction. Before this fix the reported statistics carried that
//! augmented residual and the exit verdict argued from `Σ(p + n)` against
//! `l1_slack_tol` — the wrong quantity judged by the wrong number. The
//! result was `Solve_Succeeded` at a point that violates the model's own
//! row, with every field in the result agreeing it was feasible.
//!
//! The fixture is `ralph1` from `benchmarks/mpcc/` under the exact-product
//! (`ncp_eq` / `prod_eq`) lowering, which is the model the defect was found
//! on:
//!
//! ```text
//! min 2·x − y   s.t.   x ≥ 0,   0 ≤ y ⊥ (y − x) ≥ 0
//! ```
//!
//! lowered to rows `G = y ≥ 0`, `H = y − x ≥ 0`, `G·H = 0`, with `f* = 0`
//! at the origin.
//!
//! What each test pins, and how to check it still bites. Every row was
//! run, not asserted: each mutation reddens the named test **and only**
//! that test.
//!
//! | test | mutation that reddens it |
//! |---|---|
//! | `the_reported_violation_is_the_users_not_the_augmented_problems` | the whole P1 fix — verified on the parent commit, where it reports `9.636e-15` against a true violation of `2.501e-7` |
//! | `a_violation_above_tol_does_not_report_plain_success` | the same — the parent returns `Solve_Succeeded` at a `2.500e-7` violation under `tol = 2.501e-11` |
//! | `an_eval_g_failure_does_not_fabricate_feasibility` | `let _ = tnlp.eval_g(..)` at the exit-verdict site: the reported violation becomes exactly `0.0` instead of the fallback's `4.585e-19` |
//! | `an_original_units_violation_stays_out_of_the_scaled_family` | mirror `max_violation` into `final_constr_viol` unconditionally |
//!
//! The first two are the "fails on the parent" evidence gh#794 asks for;
//! the last two cover defects the fix itself could introduce, so they are
//! checked against this file's own code rather than the parent.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

const INF: Number = 2e19;

/// `ralph1` under the exact-product lowering. Rows, in order:
/// `0: G = y`, `1: H = y − x`, `2: G·H = y·(y − x)`, with `G, H ≥ 0` and
/// the product row an equality at `0`.
struct Ralph1ProdEq {
    start: [Number; 2],
    captured: Rc<RefCell<Option<Vec<Number>>>>,
    /// When set, `eval_g` returns `false` from this call onward. Used by
    /// the evaluation-failure test; `0` means "never fail".
    fail_eval_g_after: Rc<Cell<usize>>,
    eval_g_calls: Rc<Cell<usize>>,
}

/// Handles onto a fixture's `eval_g` counters, readable after the TNLP
/// itself has been moved into the solver's `Rc<RefCell<_>>`.
#[derive(Clone)]
struct EvalGProbe {
    calls: Rc<Cell<usize>>,
    fail_after: Rc<Cell<usize>>,
}

impl Ralph1ProdEq {
    fn new(start: [Number; 2]) -> (Self, Rc<RefCell<Option<Vec<Number>>>>) {
        let (me, captured, _) = Self::with_probe(start);
        (me, captured)
    }

    fn with_probe(start: [Number; 2]) -> (Self, Rc<RefCell<Option<Vec<Number>>>>, EvalGProbe) {
        let captured = Rc::new(RefCell::new(None));
        let probe = EvalGProbe {
            calls: Rc::new(Cell::new(0)),
            fail_after: Rc::new(Cell::new(0)),
        };
        (
            Self {
                start,
                captured: Rc::clone(&captured),
                fail_eval_g_after: Rc::clone(&probe.fail_after),
                eval_g_calls: Rc::clone(&probe.calls),
            },
            captured,
            probe,
        )
    }
}

/// Violation of the model's own rows and bounds at `x`, in the model's
/// units — computed here independently of the solver so a test can compare
/// what was *reported* against what is *true*.
fn true_violation(x: &[Number]) -> Number {
    let g = [x[1], x[1] - x[0], x[1] * (x[1] - x[0])];
    let mut v: Number = 0.0;
    // Rows 0 and 1 are `≥ 0`; row 2 is `= 0`.
    v = v.max((-g[0]).max(0.0));
    v = v.max((-g[1]).max(0.0));
    v = v.max(g[2].abs());
    // The single finite bound, `x ≥ 0`.
    v.max((-x[0]).max(0.0))
}

impl TNLP for Ralph1ProdEq {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 2,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_l[1] = -INF;
        b.x_u[0] = INF;
        b.x_u[1] = INF;
        b.g_l[0] = 0.0;
        b.g_u[0] = INF;
        b.g_l[1] = 0.0;
        b.g_u[1] = INF;
        b.g_l[2] = 0.0;
        b.g_u[2] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x.copy_from_slice(&self.start);
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some(2.0 * x[0] - x[1])
    }

    fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0;
        g[1] = -1.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let n = self.eval_g_calls.get() + 1;
        self.eval_g_calls.set(n);
        let fail_after = self.fail_eval_g_after.get();
        if fail_after != 0 && n >= fail_after {
            return false;
        }
        g[0] = x[1];
        g[1] = x[1] - x[0];
        g[2] = x[1] * (x[1] - x[0]);
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
                let mut k = 0;
                for i in 0..3 {
                    for j in 0..2 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                // row 0: G = y
                values[0] = 0.0;
                values[1] = 1.0;
                // row 1: H = y − x
                values[2] = -1.0;
                values[3] = 1.0;
                // row 2: G·H = y² − x·y
                values[4] = -x[1];
                values[5] = 2.0 * x[1] - x[0];
                true
            }
        }
    }

    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _new_x: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _new_lambda: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // Lower triangle of the 2×2: (0,0), (1,0), (1,1).
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 0;
                irow[2] = 1;
                jcol[2] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                // Only the product row carries curvature: ∇²(y² − x·y).
                // The objective is linear, so `obj_factor` contributes
                // nothing.
                let l2 = lambda.and_then(|l| l.get(2).copied()).unwrap_or(0.0);
                values[0] = 0.0;
                values[1] = -l2;
                values[2] = 2.0 * l2;
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        *self.captured.borrow_mut() = Some(sol.x.to_vec());
    }
}

/// An ℓ₁-wrapped application at the given tolerances.
fn l1_app(tol: Number, acceptable_tol: Number) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_integer_value("print_level", 0, true, false);
        let _ = opts.set_string_value("l1_exact_penalty_barrier", "yes", true, false);
        let _ = opts.set_numeric_value("tol", tol, true, false);
        let _ = opts.set_numeric_value("acceptable_tol", acceptable_tol, true, false);
        let _ = opts.set_integer_value("max_iter", 300, true, false);
    }
    app.initialize().expect("initialize");
    app
}

/// The core P1 claim: the violation the caller reads is the violation of
/// the caller's own rows.
///
/// Before the fix `final_constr_viol` and `final_unscaled_constr_viol`
/// both carried the *augmented* problem's residual, which the `p`/`n`
/// slacks drive to machine precision by construction — so on this model
/// they read `~1e-15` while the model's own product row was violated
/// several orders above that. No field in the result disclosed it.
#[test]
fn the_reported_violation_is_the_users_not_the_augmented_problems() {
    let (tnlp, captured) = Ralph1ProdEq::new([1.0, 1.0]);
    let mut app = l1_app(1e-8, 1e-6);
    let _ = app.optimize_tnlp(Rc::new(RefCell::new(tnlp)));

    let x = captured.borrow().clone().expect("finalize_solution ran");
    let truth = true_violation(&x);
    let reported = app.statistics().final_unscaled_constr_viol;

    // The reported number must be the measured one. A tiny absolute floor
    // keeps the comparison meaningful when both are essentially zero.
    assert!(
        (reported - truth).abs() <= 1e-9 + 1e-6 * truth,
        "reported unscaled constraint violation {reported:.3e} does not match \
         the model's own violation {truth:.3e} at x = {x:?}",
    );

    // And the aggregate must not sit below the violation it contains:
    // that is the inequality the old code broke, reporting a converged
    // KKT error beside a point that was not feasible.
    let agg = app.statistics().final_unscaled_kkt_error;
    assert!(
        agg + 1e-12 >= truth,
        "aggregate KKT error {agg:.3e} is below the constraint violation \
         {truth:.3e} it should dominate",
    );
}

/// The exit verdict is judged by the caller's tolerances, not by
/// `l1_slack_tol`.
///
/// Same model and same returned point, read against two different
/// standards. Whatever residual this model lands on, a `tol` far *below*
/// it must not yield a plain `Solve_Succeeded`: that is exactly the
/// false success P1 is about, and `l1_slack_tol`'s `1e-6` default is four
/// orders looser than the `tol = 1e-8` such a solve asked for.
#[test]
fn a_violation_above_tol_does_not_report_plain_success() {
    let (tnlp, captured) = Ralph1ProdEq::new([1.0, 1.0]);
    let mut app = l1_app(1e-8, 1e-6);
    let _ = app.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let x = captured.borrow().clone().expect("finalize_solution ran");
    let achieved = true_violation(&x);

    // Re-solve demanding far more accuracy than the point achieved. The
    // strict standard is now unmeetable at that residual, so the verdict
    // must not be plain success.
    let strict_tol = (achieved / 1e4).max(1e-14);
    let (tnlp2, captured2) = Ralph1ProdEq::new([1.0, 1.0]);
    let mut strict = l1_app(strict_tol, strict_tol);
    let status = strict.optimize_tnlp(Rc::new(RefCell::new(tnlp2)));
    let x2 = captured2.borrow().clone().expect("finalize_solution ran");
    let achieved2 = true_violation(&x2);

    if achieved2 > strict_tol * 10.0 {
        assert_ne!(
            status,
            ApplicationReturnStatus::SolveSucceeded,
            "reported plain success at violation {achieved2:.3e} against tol \
             {strict_tol:.3e} — the ℓ₁ exit is being judged by something \
             other than the caller's tolerance",
        );
    }
}

/// gh#794 review: an `eval_g` that fails must not be read as "every row
/// is zero".
///
/// The exit-verdict measurement allocates `g` zero-filled and hands it to
/// `eval_g`. Discarding the success flag lets a TNLP whose final
/// evaluation fails present a zero vector as a measurement — a violation
/// of exactly `0`, i.e. fabricated feasibility, which then flows into both
/// the status and the reported residuals. The solve must not come back
/// claiming a perfectly feasible point.
/// gh#794 review: a failed `eval_g` must not be read as "every row is
/// zero".
///
/// The exit-verdict measurement allocates `g` zero-filled and hands it to
/// `eval_g`. Discarding the success flag lets a TNLP whose final
/// evaluation fails present that untouched buffer as a measurement — a
/// violation of exactly `0`, i.e. fabricated feasibility, which then
/// decides both the status and the reported residuals.
///
/// The failure has to land on the *exit-verdict* call specifically: fail
/// earlier and the solve dies in restoration, never reaching the code
/// under test, and the assertion below would pass vacuously. So the
/// fixture is calibrated rather than guessed — one clean solve counts the
/// evaluations, and the second fails only the last of them, which is the
/// exit-verdict call.
///
/// Signature of the defect: the reported violation is *exactly* `0.0`,
/// the zero-filled buffer's own answer. With the flag honoured the
/// wrapper instead falls back to the documented `l1_slack_tol` path and
/// reports the augmented residual (`~4.6e-19` on this model). Measured
/// both ways while writing this test.
#[test]
fn an_eval_g_failure_does_not_fabricate_feasibility() {
    // Pass 1: a clean solve, to count the evaluations.
    let (tnlp, _captured, probe) = Ralph1ProdEq::with_probe([1.0, 1.0]);
    let mut app = l1_app(1e-8, 1e-6);
    let clean_status = app.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let total_calls = probe.calls.get();
    assert!(
        total_calls > 1 && clean_status == ApplicationReturnStatus::SolveSucceeded,
        "calibration solve did not behave as expected: {clean_status:?} after \
         {total_calls} eval_g calls",
    );

    // Pass 2: identical solve, with only the final evaluation failing.
    let (tnlp2, _c2, probe2) = Ralph1ProdEq::with_probe([1.0, 1.0]);
    probe2.fail_after.set(total_calls);
    let mut app2 = l1_app(1e-8, 1e-6);
    let _ = app2.optimize_tnlp(Rc::new(RefCell::new(tnlp2)));

    let reported = app2.statistics().final_unscaled_constr_viol;
    assert_ne!(
        reported, 0.0,
        "the exit verdict reported an exactly-zero constraint violation after \
         eval_g failed: the zero-filled buffer was read as a measurement \
         instead of the failure being honoured",
    );
}

/// gh#794 review: the original-units measurement stays out of the scaled
/// field family.
///
/// `SolveStatistics` documents `final_*` as the residuals in the
/// internally scaled NLP space and `final_unscaled_*` as the same
/// quantities in the model's units, equal only when no scaling is active.
/// The measurement the wrapper makes is in the model's units, so under an
/// active row scaling it may appear only in the unscaled family.
///
/// The lever is `gradient-based` scaling (the default method) with
/// `nlp_scaling_max_gradient` pulled below this model's row-gradient
/// norms, which forces per-row factors strictly below `1`. It has to be
/// this lever rather than `user-scaling`: the ℓ₁ wrapper declines
/// `get_scaling_parameters` outright (`pounce-l1penalty/src/wrapper.rs`,
/// "Phase 2"), so user-supplied factors never reach the augmented
/// problem, while gradient-based factors are computed by the NLP layer
/// from the augmented problem itself and do engage.
#[test]
fn an_original_units_violation_stays_out_of_the_scaled_family() {
    let (tnlp, captured) = Ralph1ProdEq::new([1.0, 1.0]);
    let mut app = IpoptApplication::new();
    {
        let opts = app.options_mut();
        let _ = opts.set_string_value("sb", "yes", true, false);
        let _ = opts.set_integer_value("print_level", 0, true, false);
        let _ = opts.set_string_value("l1_exact_penalty_barrier", "yes", true, false);
        let _ = opts.set_string_value("nlp_scaling_method", "gradient-based", true, false);
        // Far below the ‖∇row‖ of this model, so every row gets a factor
        // strictly less than one and the two families must diverge.
        let _ = opts.set_numeric_value("nlp_scaling_max_gradient", 1e-2, true, false);
        let _ = opts.set_numeric_value("tol", 1e-8, true, false);
        let _ = opts.set_integer_value("max_iter", 300, true, false);
    }
    app.initialize().expect("initialize");
    let _ = app.optimize_tnlp(Rc::new(RefCell::new(tnlp)));

    let x = captured.borrow().clone().expect("finalize_solution ran");
    let truth = true_violation(&x);
    let stats = app.statistics();

    // The unscaled family carries the model-units measurement.
    assert!(
        (stats.final_unscaled_constr_viol - truth).abs() <= 1e-9 + 1e-6 * truth,
        "unscaled constraint violation {:.3e} is not the model's own {truth:.3e}",
        stats.final_unscaled_constr_viol,
    );

    // The scaled family must not have been overwritten with that same
    // original-units number while a non-unit row scaling is active. Were
    // it mirrored unconditionally the two would be bit-identical here,
    // which is exactly the contract violation this pins.
    assert_ne!(
        stats.final_constr_viol, stats.final_unscaled_constr_viol,
        "final_constr_viol and final_unscaled_constr_viol are bit-identical \
         under active gradient-based row scaling, so an original-units \
         number was written into the scaled family",
    );
}

/// A model that is **infeasible by construction, by exactly `GAP`**, with
/// its one row living at magnitude `K`.
///
/// `min (x/K)²  s.t.  x == K,  0 ≤ x ≤ K − GAP`
///
/// The row cannot be satisfied: the bound excludes its target. Every
/// returned point violates the declared equality by at least `GAP`, so any
/// success status is wrong, and no oracle is needed to say so.
struct InfeasibleLargeRow {
    k: Number,
    gap: Number,
}

impl TNLP for InfeasibleLargeRow {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 1,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 1,
            index_style: IndexStyle::C,
        })
    }
    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = self.k - self.gap;
        b.g_l[0] = self.k;
        b.g_u[0] = self.k;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = self.k - self.gap;
        true
    }
    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] / self.k).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * x[0] / (self.k * self.k);
        true
    }
    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[0];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, m: SparsityRequest<'_>) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
        }
        true
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        m: SparsityRequest<'_>,
    ) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
            }
            SparsityRequest::Values { values } => values[0] = obj_factor * 2.0 / (self.k * self.k),
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// gh#794 adversary follow-up: **a large-magnitude row must not buy a
/// success on an infeasible model.**
///
/// `original_space_feasibility` judged each row with `is_negligible(viol,
/// scale, tol)` alone — `viol ≤ tol · max(|row|, 1)`. That is a relative
/// test with no absolute floor, so the allowance grows without bound with
/// the row's magnitude: at the default `tol = 1e-8` a row near `1e10` buys
/// `1e2` of "feasible". This model is infeasible by exactly `50` with its
/// row at `1e10`, and it came back `Solve_Succeeded`.
///
/// That was a regression this branch introduced, not an inherited gap: on
/// the parent the same model exits `Error_In_Step_Computation`, because
/// the `Σ(p + n) > l1_slack_tol` argument it replaced was crude but
/// *absolute*. The fix conjoins the standard the rest of the solver uses —
/// `OptErrorConvCheck::primal_component_passes`, an absolute
/// `constr_viol_tol` with a noise-floor abstention (gh#528/gh#590) — so
/// scale-awareness comes from the row's floating-point floor rather than
/// from multiplying the tolerance by the row's size. Here `50` is `~2e7 ×`
/// that floor, so nothing abstains.
///
/// **Mutation that reddens it:** drop the `&& absolute_ok(..)` conjuncts
/// from `judge` in `original_space_feasibility` — this test then reports
/// `Solve_Succeeded` at a violation of `4.999990e1`. Verified.
///
/// The sweep behind the `k` values: with the violation held at `50`, the
/// pre-fix verdict tracked `tol · k` exactly — `Infeasible_Problem_Detected`
/// at `k = 1e6`, `Solved_To_Acceptable_Level` from `1e8` to `4.9e9`
/// (threshold `49 < 50`), and `Solve_Succeeded` from `5.1e9` (threshold
/// `51 > 50`) up. The two `k` values below straddle that crossover.
#[test]
fn a_large_row_magnitude_does_not_buy_success_on_an_infeasible_model() {
    for k in [5.1e9, 1e10, 1e12] {
        let gap = 50.0;
        let mut app = IpoptApplication::new();
        {
            let opts = app.options_mut();
            let _ = opts.set_string_value("sb", "yes", true, false);
            let _ = opts.set_integer_value("print_level", 0, true, false);
            let _ = opts.set_string_value("l1_exact_penalty_barrier", "yes", true, false);
            // Both arms on the NLP path, so the comparison is like for like.
            let _ = opts.set_string_value("solver_selection", "nlp", true, false);
            let _ = opts.set_integer_value("max_iter", 500, true, false);
        }
        app.initialize().expect("initialize");
        let status = app.optimize_tnlp(Rc::new(RefCell::new(InfeasibleLargeRow { k, gap })));

        assert!(
            !matches!(
                status,
                ApplicationReturnStatus::SolveSucceeded
                    | ApplicationReturnStatus::SolvedToAcceptableLevel
                    | ApplicationReturnStatus::FeasiblePointFound
            ),
            "k = {k:.1e}: reported {status:?} on a model infeasible by exactly {gap}; \
             a row at this magnitude bought {:.1e} of allowance from the \
             scale-relative test (gh#794 adversary)",
            1e-8 * k,
        );

        // And the number stays honest — the P1 property must survive the
        // stricter verdict rather than be traded against it.
        let reported = app.statistics().final_unscaled_constr_viol;
        assert!(
            (reported - gap).abs() <= 1e-3 * gap,
            "k = {k:.1e}: reported violation {reported:.6e} does not match the \
             model's own violation of {gap}",
        );
    }
}

/// gh#987 (remaining item b): under `l1_exact_penalty_barrier=yes`
/// `final_declared_constr_viol` described the wrapper's augmented rows, which
/// the elastic slacks satisfy by construction — `~1e-12` on a model
/// infeasible by `O(1)`. It must be the user's own violation, at
/// `bound_relax_factor = 0` as much as anywhere, and stay small on a feasible
/// variant of the same model.
#[test]
fn the_declared_violation_under_the_l1_penalty_is_the_users() {
    for (gap, infeasible) in [(0.5, true), (-0.5, false)] {
        let mut app = IpoptApplication::new();
        {
            let opts = app.options_mut();
            let _ = opts.set_string_value("sb", "yes", true, false);
            let _ = opts.set_integer_value("print_level", 0, true, false);
            let _ = opts.set_string_value("l1_exact_penalty_barrier", "yes", true, false);
            let _ = opts.set_numeric_value("bound_relax_factor", 0.0, true, false);
            let _ = opts.set_string_value("solver_selection", "nlp", true, false);
            let _ = opts.set_integer_value("max_iter", 500, true, false);
        }
        app.initialize().expect("initialize");
        let _ = app.optimize_tnlp(Rc::new(RefCell::new(InfeasibleLargeRow { k: 1.0, gap })));
        let declared = app.statistics().final_declared_constr_viol;
        if infeasible {
            assert!(
                (declared - gap).abs() <= 1e-3,
                "infeasible by {gap}: final_declared_constr_viol reads {declared:.3e}, \
                 the augmented problem's residual rather than the model's"
            );
        } else {
            assert!(
                declared.is_finite() && declared <= 1e-6,
                "feasible variant: final_declared_constr_viol reads {declared:.3e}"
            );
        }
    }
}
