//! gh#884: a biactive complementarity pair makes the exact-product
//! lowering's multipliers diverge, and the run fails next to the primal
//! solution.
//!
//! **Provenance, by measurement group.** Almost everything here is
//! *defect*-era, measured on `87402274` before the fix existed: the
//! multiplier table below, the base attempt's returned point, the
//! `3.25e11` and `7.9e+04` residuals, the `4.64e-4` objective-gradient
//! residual it leaves, the `9.96e-8` an honest solve reaches, and — both
//! of them, from one run — `ralph1`'s objectives under
//! `perturb_always_cd` and the `5.25e-7` that run certifies.
//! `ralph1`'s `7.2e-3` step floor belongs to that group too: it is the
//! minimum `||d||` of its *base* `RestorationFailed` trajectory, which
//! the retry never touches (`dev-notes/mpcc-biactive-dual-divergence.md`,
//! the `||d||` separation table).
//!
//! Only the **promoted** point and its violation, in
//! `qpec_small_returns_a_near_optimal_point_and_never_claims_a_false_success`,
//! are *retry*-era — they could not exist before the fix — and they come
//! from its own commits (`ef1dd0b`, `198173b`; PR #885).
//!
//! Both groups were re-run on `d32204e0` and reproduce, with one
//! exception that was **not** re-derived there: the `7.2e-3` floor, which
//! is a minimum over a trajectory rather than a quantity the solve
//! returns, and whose source is the dev-note's table. Each site names the
//! group it belongs to; re-measure rather than trusting any of them
//! across a commit that touches the IPM.
//!
//! **The first four are invariants rather than a description of the
//! defect.** Each is written so that it stays correct after a fix: it
//! fails when the solver gets *worse*, never when it gets better. In
//! particular none of them asserts that the failure persists — a test
//! that pinned the bug would go red on a genuine fix, which is
//! backwards, and would train the next reader to expect red here. They
//! were written while gh#884 was open and are unchanged by its closing,
//! except for `qpec_small`'s feasibility bar, which moved `1e-12` ->
//! `1e-10` because the retry reaches the answer along a different
//! trajectory (see that test).
//!
//! **The last four are about the mechanism that closed it**, and each
//! exists because the rule has a branch a green fixture would otherwise
//! never execute. See the block comment above them.
//!
//! The measurements — the runaway multipliers, the `mu` trace, and the
//! four approaches ruled out — live in
//! `dev-notes/mpcc-biactive-dual-divergence.md`. That is where history
//! belongs; this file holds contracts. If you are fixing gh#884, read the
//! note first: the obvious remedy is measured and rejected there.
//!
//! The fixture is `qpec_small` from `benchmarks/mpcc/` under the exact
//! product (`ncp_eq` / `prod_eq`) lowering:
//!
//! ```text
//! min (x-1)^2 + (y1-1)^2 + y2^2
//! s.t. 0 <= x <= 2
//!      0 <= y1  _|_  (2 y1 - 1 - x) >= 0
//!      0 <= y2  _|_  (2 y2 - 1 + x) >= 0
//! ```
//!
//! with `f* = 0` at `(1, 1, 0)`, where pair 1 is strict and **pair 2 is
//! biactive** (`G2 = H2 = 0`). There `grad f` is exactly `0` and the
//! product row's gradient is exactly `(0, 0, 0)`, so `lambda = 0`
//! certifies stationarity with residual `0` **at that exact point** — so
//! the answer exists and is reachable.
//!
//! What happens instead — the *base* attempt, which is what gh#884 is
//! about, and which the default path still runs first: it reaches
//! `(1.0002321, 1.0001161, 2.67e-15)`, feasible to `2.2e-16` with
//! `f = 6.73e-8`, and before #885 stopped there. (Since #885 the default
//! path does not stop there: it detects the runaway, retries cold, and
//! returns the promoted point instead. Everything from here to the end of
//! this section describes the attempt that fails.)
//! That is *near* the optimum, not at it — `lambda = 0` does not certify
//! the returned iterate, where it leaves an objective-gradient residual of
//! `4.64e-4`. Meanwhile the multipliers on the *linearly dependent* rows
//! run away in near-cancelling pairs; rows 2 and 5 both restrict only
//! `y2`, and rows 1 and 4 are likewise parallel:
//!
//! | row | `|grad|_inf` | `lambda` |
//! |---|---:|---:|
//! | 1 (`H1 >= 0`) | `2.0` | `-2.089e10` |
//! | 4 (`G1·H1 = 0`) | `2.0` | `+2.089e10` |
//! | 2 (`G2 >= 0`) | `1.0` | `-7.283e11` |
//! | 5 (`G2·H2 = 0`) | `2.3e-4` | `+1.737e15` |
//!
//! `-7.283e11 + 1.737e15 · 2.3e-4` is the `3.25e11` residual it fails
//! on. That table is measurement, not an assertion — see the note.
//!
//! **The asymmetry between the two fixtures is the point.** `qpec_small`
//! failing is the *bug*, so nothing here asserts that it fails; what is
//! asserted is that the point it returns is feasible and lands *near* the
//! optimum — not at it, in either era; see that test — and that it never
//! claims a success it cannot back. `ralph1` failing is *correct* —
//! no sign-feasible multiplier exists at its origin — so that one is
//! asserted directly, and it is what catches a `perturb_always_cd`
//! default flip.
//!
//! | test | what it pins | red when |
//! |---|---|---|
//! | `qpec_small_returns_a_near_optimal_point_and_never_claims_a_false_success` | the returned point is feasible and within `1e-3` of `(1, 1, 0)` with `|f|` under `1e-6` — near `f*`, not at it; any claimed success must hold in the model's own units | a verdict-only "fix" reports success at a `1e11` residual |
//! | `a_structurally_zero_hessian_entry_does_not_change_the_solve` | declaring an identically-zero Hessian entry is a no-op (and refutes gh#884's stated prerequisite hypothesis) | the declared sparsity pattern starts changing the answer |
//! | `dual_regularization_reaches_a_certificate_honestly` | an honest solve of this model exists at `9.96e-8` unscaled, so gh#884 is a POUNCE gap | that configuration stops reaching the answer |
//! | `ralph1_must_not_claim_success_where_no_multiplier_certifies_it` | a model with no sign-feasible multiplier must not report success, and never below `f*` | `perturb_always_cd` is turned on by default — measured: plain `Solve_Succeeded` at `-2.71e-5` |
//! | `the_kill_switch_restores_the_pre_884_verdict` | `dual_divergence_retry=no` returns the base attempt's verdict | the kill switch stops killing |
//! | `a_zero_step_tolerance_holds_the_detector_off` | a `0` step tolerance is a second, finer off switch | the detector stops consulting its threshold |
//! | `the_detector_must_not_fire_on_ralph1` | **the safety property** — the detector is the only barrier between `ralph1` and a success below `f*` | the step conjunct is widened past `7.2e-3` |
//! | `a_retry_that_does_not_promote_leaves_the_base_answer_in_place` | a refused retry leaves status, point and residuals as the base attempt left them | the promotion gate's conjunct 4 stops refusing, or the floor stops restoring |

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
    TNLP,
};

const INF: Number = 2e19;

/// What `finalize_solution` hands back, so a test can measure the
/// returned point for itself rather than trusting a reported number.
///
/// The multipliers are deliberately *not* carried here. The runaway
/// values in the module table above are a measurement of behaviour
/// gh#884 is open about, and asserting on them would pin the bug; the
/// table and `dev-notes/mpcc-biactive-dual-divergence.md` are their home.
#[derive(Clone, Default)]
struct Solved {
    x: Vec<Number>,
}

/// `qpec_small` under the exact-product lowering. Rows, in order:
/// `0: G1 = y1`, `1: H1 = 2y1 − 1 − x`, `2: G2 = y2`, `3: H2 = 2y2 − 1 + x`,
/// `4: G1·H1`, `5: G2·H2`, the last two equalities at `0`.
struct QpecSmallProdEq {
    captured: Rc<RefCell<Option<Solved>>>,
    /// The MPCC harness hands back a *dense* lower triangle (6 nonzeros)
    /// where the structurally correct pattern has 5 — `(2, 1)` is
    /// identically zero. gh#884 names that difference as the leading
    /// hypothesis for its two paths' different exits, so it is a switch.
    dense_hessian: bool,
}

impl QpecSmallProdEq {
    fn new(dense_hessian: bool) -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
                dense_hessian,
            },
            captured,
        )
    }
}

/// Violation of the model's own rows and bounds at `x`, in the model's
/// units, computed independently of the solver.
fn true_violation(x: &[Number]) -> Number {
    let g = [x[1], 2.0 * x[1] - 1.0 - x[0], x[2], 2.0 * x[2] - 1.0 + x[0]];
    let mut v: Number = 0.0;
    for gi in &g {
        v = v.max((-gi).max(0.0));
    }
    v = v.max((g[0] * g[1]).abs());
    v = v.max((g[2] * g[3]).abs());
    v = v.max((-x[0]).max(0.0));
    v.max((x[0] - 2.0).max(0.0))
}

impl TNLP for QpecSmallProdEq {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 6,
            nnz_jac_g: 18,
            nnz_h_lag: if self.dense_hessian { 6 } else { 5 },
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 2.0;
        b.x_l[1] = -INF;
        b.x_u[1] = INF;
        b.x_l[2] = -INF;
        b.x_u[2] = INF;
        for i in 0..4 {
            b.g_l[i] = 0.0;
            b.g_u[i] = INF;
        }
        for i in 4..6 {
            b.g_l[i] = 0.0;
            b.g_u[i] = 0.0;
        }
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.5;
        sp.x[2] = 0.5;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        Some((x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2) + x[2] * x[2])
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 2.0 * (x[1] - 1.0);
        g[2] = 2.0 * x[2];
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let g1 = x[1];
        let h1 = 2.0 * x[1] - 1.0 - x[0];
        let g2 = x[2];
        let h2 = 2.0 * x[2] - 1.0 + x[0];
        g[0] = g1;
        g[1] = h1;
        g[2] = g2;
        g[3] = h2;
        g[4] = g1 * h1;
        g[5] = g2 * h2;
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
                for i in 0..6 {
                    for j in 0..3 {
                        irow[k] = i as Index;
                        jcol[k] = j as Index;
                        k += 1;
                    }
                }
                true
            }
            SparsityRequest::Values { values } => {
                let Some(x) = x else { return false };
                let rows = jac_rows(x);
                for (i, row) in rows.iter().enumerate() {
                    values[3 * i..3 * i + 3].copy_from_slice(row);
                }
                true
            }
        }
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
        let dense = self.dense_hessian;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                let pattern: &[(Index, Index)] = if dense {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
                } else {
                    &[(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)]
                };
                for (k, (i, j)) in pattern.iter().enumerate() {
                    irow[k] = *i;
                    jcol[k] = *j;
                }
                true
            }
            SparsityRequest::Values { values } => {
                let l4 = lambda.and_then(|l| l.get(4).copied()).unwrap_or(0.0);
                let l5 = lambda.and_then(|l| l.get(5).copied()).unwrap_or(0.0);
                // ∇²f = diag(2,2,2); row 4 adds ∂²/∂y1² = 4, ∂²/∂x∂y1 = −1;
                // row 5 adds ∂²/∂y2² = 4, ∂²/∂x∂y2 = +1. `(2,1)` is
                // identically zero, which is what makes it optional.
                values[0] = 2.0 * obj_factor;
                values[1] = -l4;
                values[2] = 2.0 * obj_factor + 4.0 * l4;
                values[3] = l5;
                if dense {
                    values[4] = 0.0;
                    values[5] = 2.0 * obj_factor + 4.0 * l5;
                } else {
                    values[4] = 2.0 * obj_factor + 4.0 * l5;
                }
                true
            }
        }
    }

    fn finalize_solution(&mut self, sol: Solution<'_>, _ip_data: &IpoptData, _ip_cq: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved { x: sol.x.to_vec() });
    }
}

/// The six constraint gradients at `x`.
fn jac_rows(x: &[Number]) -> [[Number; 3]; 6] {
    [
        [0.0, 1.0, 0.0],
        [-1.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [-x[1], 4.0 * x[1] - 1.0 - x[0], 0.0],
        [x[2], 0.0, 4.0 * x[2] - 1.0 + x[0]],
    ]
}

/// `ralph1` under the `direct` lowering (`G·H <= 0`):
/// `min 2x − y  s.t.  x >= 0, G = y >= 0, H = y − x >= 0, G·H <= 0`,
/// with `f* = 0` at the origin.
///
/// The origin is M-stationary but **not** S-stationary, and NLP KKT is
/// S-stationarity, so no sign-feasible multiplier exists there and
/// failing is the correct outcome. gh#884 makes that an explicit
/// acceptance criterion: a fix that greens this has over-fired.
struct Ralph1Direct {
    captured: Rc<RefCell<Option<Solved>>>,
}

impl Ralph1Direct {
    fn new() -> (Self, Rc<RefCell<Option<Solved>>>) {
        let captured = Rc::new(RefCell::new(None));
        (
            Self {
                captured: Rc::clone(&captured),
            },
            captured,
        )
    }
}

impl TNLP for Ralph1Direct {
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
        // the `direct` lowering: `G·H <= 0`
        b.g_l[2] = -INF;
        b.g_u[2] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.0;
        sp.x[1] = 0.0;
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        Some(2.0 * x[0] - x[1])
    }
    fn eval_grad_f(&mut self, _x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = 2.0;
        g[1] = -1.0;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[1];
        g[1] = x[1] - x[0];
        g[2] = x[1] * (x[1] - x[0]);
        true
    }
    fn eval_jac_g(&mut self, x: Option<&[Number]>, _n: bool, m: SparsityRequest<'_>) -> bool {
        match m {
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
                values[0] = 0.0;
                values[1] = 1.0;
                values[2] = -1.0;
                values[3] = 1.0;
                values[4] = -x[1];
                values[5] = 2.0 * x[1] - x[0];
                true
            }
        }
    }
    fn eval_h(
        &mut self,
        _x: Option<&[Number]>,
        _n: bool,
        _obj_factor: Number,
        lambda: Option<&[Number]>,
        _nl: bool,
        m: SparsityRequest<'_>,
    ) -> bool {
        match m {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 0;
                irow[1] = 1;
                jcol[1] = 0;
                irow[2] = 1;
                jcol[2] = 1;
                true
            }
            SparsityRequest::Values { values } => {
                let l2 = lambda.and_then(|l| l.get(2).copied()).unwrap_or(0.0);
                values[0] = 0.0;
                values[1] = -l2;
                values[2] = 2.0 * l2;
                true
            }
        }
    }
    fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {
        *self.captured.borrow_mut() = Some(Solved { x: sol.x.to_vec() });
    }
}

/// The `.nl`-path options gh#884 measures under.
fn app(always_cd: bool, max_iter: i32) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        let _ = o.set_string_value("sb", "yes", true, false);
        let _ = o.set_integer_value("print_level", 0, true, false);
        let _ = o.set_numeric_value("tol", 1e-8, true, false);
        let _ = o.set_numeric_value("bound_relax_factor", 0.0, true, false);
        let _ = o.set_string_value("honor_original_bounds", "yes", true, false);
        let _ = o.set_integer_value("max_iter", max_iter, true, false);
        if always_cd {
            let _ = o.set_string_value("perturb_always_cd", "yes", true, false);
        }
    }
    app.initialize().expect("initialize");
    app
}

/// Statuses POUNCE offers as "we solved it". A claim of success made
/// under any of these is a claim the user's tooling acts on — Pyomo maps
/// both into its solved family.
fn claims_success(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded | ApplicationReturnStatus::SolvedToAcceptableLevel
    )
}

/// **If it claims success, the claim has to be true in the model's own
/// units** — asserted for *these two fixtures*, at a bar each of them
/// earns individually.
///
/// This is deliberately **not** stated as a general POUNCE invariant, and
/// it is worth being precise about why, because the earlier draft of this
/// comment claimed exactly that and was wrong. `acceptable_dual_inf_tol`
/// defaults to `1e10`: an exit at `Solved_To_Acceptable_Level` with a
/// large unscaled dual residual is *by design*, not a defect, and POUNCE
/// produces them routinely. `CLAUDE.md` now records a measured one — an
/// unconstrained ill-conditioned quadratic exiting acceptable at an
/// unscaled dual residual of `87.5`, which is correct behaviour and which
/// this helper would reject. Applied workspace-wide it would fail POUNCE
/// across the whole unconstrained class.
///
/// What makes `1e-6` the right bar *here* is a property of these two
/// models, not of the solver:
///
/// - `qpec_small` — `grad f` is exactly `0` at `(1, 1, 0)` and the
///   product row's gradient is exactly `(0, 0, 0)`, so `lambda = 0`
///   certifies with residual `0`; and an honest solve reaching `9.96e-8`
///   unscaled is measured in
///   `dual_regularization_reaches_a_certificate_honestly`. A success claim
///   on this model at a residual above `1e-6` is therefore *known* to be
///   avoidable, which is what makes it suspect rather than merely loose.
/// - `ralph1` — no sign-feasible multiplier exists at its origin at all,
///   so any success claim is false at any bar.
///
/// Vacuous while a model is failing, which is the point — it is a guard,
/// not a description. It becomes load-bearing the moment anything starts
/// reporting success on the model it is applied to, which is exactly when
/// someone needs catching.
///
/// The corollary for a fixer: this does **not** say a fix must land
/// `qpec_small` at `Solve_Succeeded`. It says that whatever verdict it
/// lands on, the residual behind it has to be one this model is known to
/// be able to reach. Refusing the verdict outright satisfies it too — and
/// that is `gh#884`'s criterion 1, which asks that the model "must not
/// report success", not that it must converge.
fn a_claimed_success_must_be_real(
    what: &str,
    status: ApplicationReturnStatus,
    unscaled_kkt: Number,
    violation: Number,
) {
    if !claims_success(status) {
        return;
    }
    assert!(
        unscaled_kkt <= 1e-6,
        "{what}: reported {status:?} at an unscaled KKT error of {unscaled_kkt:.3e}. \
         A verdict that passes while the residual it stands for is orders larger \
         is the symptom-only fix gh#884 warns against. (The bar is this \
         fixture's own — see the helper's doc; it is not a general POUNCE \
         invariant.)",
    );
    assert!(
        violation <= 1e-8,
        "{what}: reported {status:?} at a constraint violation of {violation:.3e}",
    );
}

/// `qpec_small` returns a near-optimal point, and must never claim a
/// false success.
///
/// Two halves, and the split is deliberate.
///
/// The **unconditional** half — the returned point is feasible and lands
/// within `1e-3` of `(1, 1, 0)` with `|f|` under `1e-6` — is true today
/// *and* after any fix, so it never needs revisiting. The tolerances are
/// what they are because the run stops *near* the optimum, not at it, and
/// that is still true after the retry: the base attempt returned
/// `(1.0002321, 1.0001161, 2.67e-15)` with `f = 6.73e-8` (measured on
/// `87402274`), and the promoted retry returns
/// `(0.9999940, 0.9999970, 3.72e-6)` with `f = 5.84e-11` (measured with
/// the fix in place; reproduces on `d32204e0`). Neither is `(1, 1, 0)`.
/// The interesting fact about the first is that it gives up that close;
/// the second is three orders nearer in objective (and about one and a
/// half in `x`) and still not the point itself, which is why nothing here
/// asserts `f*` exactly.
///
/// The **conditional** half is the guard, and it is no longer vacuous.
/// Before gh#884 closed, this model exited `RestorationFailed` at an
/// unscaled KKT error of `3.25e11` (and, from the acceptable-level start
/// the issue filed, `Solved_To_Acceptable_Level` at `7.9e+04`), so the
/// guard had nothing to check. The dual-divergence retry now takes it to
/// `Solve_Succeeded` with a residual that holds in the model's own units,
/// which is what the guard has always demanded of a claimed success. What
/// it still catches is the tempting bad fix gh#884 names — relaxing the
/// exit verdict so the model reports success while the residual stays at
/// `1e11`. That fix passes a status assertion and fails this one.
///
/// The feasibility bar moved `1e-12` → `1e-10` when the retry landed, and
/// the direction is worth naming: the *base* attempt violates by
/// `2.2e-16`, the promoted one by `5.5e-12`. The retry is not a tighter
/// solve of the same trajectory — it is a different one, run with
/// `perturb_always_cd`, and it buys eighteen orders of unscaled dual
/// residual (`3.25e11` → `9.96e-8`) for four of primal. Both are far
/// inside `constr_viol_tol`; the point of the number here is that the
/// trade is recorded rather than absorbed. All four reproduce on
/// `d32204e0`: the base pair is the pre-fix behaviour, re-measured there
/// with `dual_divergence_retry=no`, and the promoted pair is retry-era
/// and could not have been measured on `87402274` at all.
#[test]
fn qpec_small_returns_a_near_optimal_point_and_never_claims_a_false_success() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();
    let violation = true_violation(&sol.x);

    // Forward-compatible: true while gh#884 is open, and after it closes.
    // Deliberately loose in `x` and `f` — the run stops near the optimum,
    // not at it (see the doc comment); tightening these would pin the
    // distance gh#884 leaves on the table.
    assert!(
        violation <= 1e-10,
        "expected a feasible point, got violation {violation:.3e} at {:?}",
        sol.x,
    );
    for (i, want) in [1.0, 1.0, 0.0].iter().enumerate() {
        assert!(
            (sol.x[i] - want).abs() <= 1e-3,
            "x[{i}] = {:.6e} is not within 1e-3 of the optimum {want}",
            sol.x[i],
        );
    }
    assert!(
        s.final_objective.abs() <= 1e-6,
        "objective {:.3e} is not within 1e-6 of f* = 0",
        s.final_objective,
    );

    a_claimed_success_must_be_real(
        "qpec_small/prod_eq/origin",
        status,
        s.final_unscaled_kkt_error,
        violation,
    );

    // gh#884, criterion 1, in its strongest available form. The issue
    // asks only that this model not report success at an unscaled
    // `7.9e+04`; the retry does better and reaches a certificate, so pin
    // that instead — a regression that returns the honest failure would
    // still satisfy criterion 1 and would still be a regression.
    assert_eq!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "the dual-divergence retry should reach a real certificate here \
         (unscaled KKT {:.3e})",
        s.final_unscaled_kkt_error,
    );
    assert!(
        s.dual_divergence_retry_promoted,
        "the certificate should have come from the gh#884 retry, not from \
         the base attempt; signature={} promoted={}",
        s.dual_divergence_signature, s.dual_divergence_retry_promoted,
    );
}

/// Declaring a structurally-zero Hessian entry must not change the solve.
///
/// A permanent contract in its own right: `(2, 1)` is identically zero
/// here, so a caller that declares it — as the MPCC harness does, handing
/// back a dense lower triangle with 6 nonzeros where the structural
/// pattern has 5 — must get the same answer as one that does not.
///
/// It also settles gh#884's stated prerequisite. The issue records that
/// the same model exits differently through the harness's Python path
/// (`Error_In_Step_Computation`, 118 iters) than through `.nl`/CLI
/// (`Solved_To_Acceptable_Level`, 41), and names this sparsity difference
/// as the leading hypothesis, explicitly not established. It is
/// **refuted**: both patterns give bit-identical iterates, the same
/// iteration count and the same status. Whatever separates those two
/// paths is somewhere else.
#[test]
fn a_structurally_zero_hessian_entry_does_not_change_the_solve() {
    let mut seen = Vec::new();
    for dense in [false, true] {
        let (tnlp, captured) = QpecSmallProdEq::new(dense);
        let mut a = app(false, 300);
        let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
        let sol = captured.borrow().clone().expect("finalize_solution ran");
        let s = a.statistics();
        seen.push((status, s.iteration_count, sol.x.clone()));
    }
    assert_eq!(
        seen[0].0, seen[1].0,
        "status differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].1, seen[1].1,
        "iteration count differs between the 5- and 6-nonzero Hessian patterns",
    );
    assert_eq!(
        seen[0].2, seen[1].2,
        "the returned iterate differs between the 5- and 6-nonzero Hessian \
         patterns — bit-identical is the measured result",
    );
}

/// An honest solve of this model exists, so gh#884 is a POUNCE gap and
/// not a property of the reformulation.
///
/// With dual regularization on from the start, `qpec_small` converges to
/// an **unscaled** KKT error of `~1e-7` — in the model's own units, not
/// by normalising the residual away.
///
/// The name says *certificate*, not *optimum*, and the distinction is the
/// same one the `qpec_small_returns_a_near_optimal_point_…` rename draws.
/// This run lands on `(0.999994, 0.999997, 3.7e-6)` — bit-identical to
/// the point the retry promotes, measured on `d32204e0`, because the
/// retry *is* this configuration restarted cold — which is near
/// `(1, 1, 0)` and not at it. What is pinned here is that the residual
/// behind the verdict holds in the model's own units — not that the
/// iterate is the optimum.
///
/// This is evidence that there is something to fix. It is **not** an
/// argument for turning that option on by default: see
/// `ralph1_must_not_claim_success_where_no_multiplier_certifies_it`, and
/// `dev-notes/mpcc-biactive-dual-divergence.md` ("Ruled out 2") for why
/// engaging it *in flight*, after the runaway is detected mid-solve, does
/// not work either — by the time the pattern is visible, regularization
/// can no longer recover *that iterate*, non-monotonically so. That
/// negative is about the in-flight switch specifically. What shipped does
/// engage this option conditionally, but by restarting the solve **cold**
/// from the original starting point, which is a different mechanism and
/// is not what those measurements rule out.
#[test]
fn dual_regularization_reaches_a_certificate_honestly() {
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(true, 300);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    assert!(
        claims_success(status),
        "dual regularization no longer reaches the answer on qpec_small: {status:?}",
    );
    assert!(
        s.final_unscaled_kkt_error <= 1e-6,
        "converged, but not in the model's own units: unscaled KKT {:.3e}",
        s.final_unscaled_kkt_error,
    );
    assert!(
        true_violation(&sol.x) <= 1e-9,
        "converged to an infeasible point: violation {:.3e}",
        true_violation(&sol.x),
    );
}

/// `ralph1`'s origin admits no sign-feasible multiplier, so **failing
/// there is correct** — and a reported success must never sit below `f*`.
///
/// This is gh#884's own acceptance criterion as a contract, and unlike
/// the qpec_small guard it is not vacuous: failing is the *right* answer
/// on this model, today and after any fix. NLP KKT is S-stationarity;
/// the origin is M-stationary but not S-stationary, and the best
/// achievable residual over sign-feasible multipliers there is `0.707`,
/// three orders above any tolerance in play.
///
/// **What this catches.** The obvious way to "fix" gh#884 is to turn
/// `perturb_always_cd` on by default — dual regularization is the one
/// thing measured to reach qpec_small's answer. Do that and this test
/// goes red, because it runs at *default* options: with regularization
/// always on, this model reports plain `Solve_Succeeded` at an objective
/// of `-2.71e-5`, below `f* = 0`, at a point the MPCC does not contain
/// (measured at `max_iter = 3000`; at `300` the cap hides it as
/// `Solved_To_Acceptable_Level` at `-3.81e-5`).
///
/// So the trap is guarded in the direction that stays correct: this test
/// fails when the solver gets *worse*, never when it gets better. It
/// deliberately does not assert that the bad outcome still happens under
/// the non-default option — that measurement is history, and history
/// belongs in `dev-notes/mpcc-biactive-dual-divergence.md`.
///
/// Mutation-checked by applying `a_claimed_success_must_be_real` and the
/// objective bound below to a `perturb_always_cd=yes` run of this same
/// fixture: both fire, on the objective bound, at `-2.71e-5`.
#[test]
fn ralph1_must_not_claim_success_where_no_multiplier_certifies_it() {
    // A generous cap, so a pass here is not the cap hiding an outcome.
    let (tnlp, captured) = Ralph1Direct::new();
    let mut a = app(false, 3000);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let sol = captured.borrow().clone().expect("finalize_solution ran");
    let s = a.statistics();

    assert!(
        !claims_success(status),
        "reported {status:?} at {:?} on a model whose only candidate point \
         admits no sign-feasible multiplier (best residual there is 0.707). \
         If this went red on a `perturb_always_cd` default flip, that is the \
         test working: see dev-notes/mpcc-biactive-dual-divergence.md.",
        sol.x,
    );

    // And whatever it reports, never a success below the true optimum.
    if claims_success(status) {
        assert!(
            s.final_objective >= -1e-9,
            "reported {status:?} at objective {:.6e}, below f* = 0 — a point \
             outside the MPCC",
            s.final_objective,
        );
    }
    a_claimed_success_must_be_real(
        "ralph1/direct/origin",
        status,
        s.final_unscaled_kkt_error,
        true_violation_ralph1(&sol.x),
    );
}

/// Violation of `ralph1`'s own rows and bounds at `x`, in the model's
/// units: `G = y >= 0`, `H = y − x >= 0`, `G·H <= 0`, `x >= 0`.
fn true_violation_ralph1(x: &[Number]) -> Number {
    let g = x[1];
    let h = x[1] - x[0];
    let mut v: Number = 0.0;
    v = v.max((-g).max(0.0));
    v = v.max((-h).max(0.0));
    v = v.max((g * h).max(0.0));
    v.max((-x[0]).max(0.0))
}

// ---------------------------------------------------------------------
// gh#884's fix, and the branches it can take.
//
// The four tests above are the *invariants* — they were written while the
// issue was open and are worded to survive it. The four below are about
// the mechanism that closed it, and each one exists because the rule has
// a branch a green fixture would otherwise never execute (CLAUDE.md, "a
// leg is only evidence about the branch its fixture reaches"):
//
// | test | branch |
// |---|---|
// | `the_kill_switch_restores_the_pre_884_verdict` | detector never consulted |
// | `a_zero_step_tolerance_holds_the_detector_off` | detector consulted, never fires |
// | `the_detector_must_not_fire_on_ralph1` | detector consulted, refuses — the safety property |
// | `a_retry_that_does_not_promote_leaves_the_base_answer_in_place` | detector fires, retry loses, floor restores |
// ---------------------------------------------------------------------

/// `dual_divergence_retry=no` restores the pre-gh#884 behaviour outright.
///
/// The kill switch is not a formality here. The retry's remedy is
/// `perturb_always_cd`, which `ralph1_must_not_claim_success_where_no_
/// multiplier_certifies_it` measures reporting success below `f*`, so a
/// user who finds the detector firing where it should not needs a way to
/// turn the whole thing off — not merely to widen a threshold.
///
/// Pins the *verdict*, not a residual: what the base attempt reports on
/// this model is a property of gh#884 and belongs in the dev-note, so
/// this asserts only that the answer is no longer the retry's.
#[test]
fn the_kill_switch_restores_the_pre_884_verdict() {
    let (tnlp, _captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let _ = a
        .options_mut()
        .set_string_value("dual_divergence_retry", "no", true, false);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let s = a.statistics();

    assert!(
        !s.dual_divergence_retry_promoted,
        "the kill switch did not stop the retry: {status:?}",
    );
    assert_ne!(
        status,
        ApplicationReturnStatus::SolveSucceeded,
        "with the retry off this model has no certificate to reach — a \
         Solve_Succeeded here means the base attempt started producing one, \
         which would make the retry dead code and this test the place that \
         says so",
    );
}

/// `dual_divergence_retry_step_tol = 0` holds the detector off with the
/// feature still enabled.
///
/// A different branch from the kill switch above: there the wrapper is
/// never entered, here it is entered and the *detector* declines, so the
/// base attempt's verdict comes back through the "signature never set"
/// early return rather than through the option gate. Both paths have to
/// leave the answer alone and only one of them is exercised by the other
/// test.
#[test]
fn a_zero_step_tolerance_holds_the_detector_off() {
    let (tnlp, _captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let _ = a
        .options_mut()
        .set_numeric_value("dual_divergence_retry_step_tol", 0.0, true, false);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let s = a.statistics();

    assert!(
        !s.dual_divergence_signature,
        "the detector fired at a step tolerance of 0",
    );
    assert!(!s.dual_divergence_retry_promoted);
    assert_ne!(status, ApplicationReturnStatus::SolveSucceeded);
}

/// **The safety property.** The detector must not fire on `ralph1`.
///
/// `ralph1_must_not_claim_success_where_no_multiplier_certifies_it` above
/// pins the *outcome* — no false success — and would stay green if the
/// detector fired and the promotion gate happened to catch it. That is
/// not the guarantee this feature rests on, because on this model the
/// gate does **not** catch it: `perturb_always_cd=yes` reaches
/// `Solve_Succeeded` at an unscaled KKT error of `5.25e-7` and an
/// objective of `-2.71e-5`, which is a *better* certified residual than
/// the base attempt's and would therefore promote — a wrong answer,
/// reported as success, through a gate that saw nothing wrong.
///
/// So the detector is the entire barrier, and this is the test that says
/// so. It fires red the moment `dual_divergence_retry_step_tol` is
/// widened past `ralph1`'s floor, which is the one change to this feature
/// that turns an honest failure into a lie.
///
/// Mutation-checked: set `dual_divergence_retry_step_tol` to `1e-1` (past
/// `ralph1`'s measured `7.2e-3`) and this test goes red on the signature
/// assertion, while everything else in this file stays green.
#[test]
fn the_detector_must_not_fire_on_ralph1() {
    let (tnlp, _captured) = Ralph1Direct::new();
    let mut a = app(false, 3000);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let s = a.statistics();

    assert!(
        !s.dual_divergence_signature,
        "the gh#884 detector fired on ralph1 ({status:?}). Its primal never \
         settles — the scale-relative step bottoms out at 7.2e-3, five \
         orders above the 1e-5 default — so this means the step conjunct \
         moved. The retry it authorizes reaches Solve_Succeeded at \
         f = -2.71e-5, below f* = 0, and the promotion gate does not catch \
         that. See dev-notes/mpcc-biactive-dual-divergence.md.",
    );
    assert!(!s.dual_divergence_retry_promoted);
}

/// A retry that does not promote leaves the base attempt's answer,
/// status and statistics exactly where they were.
///
/// The branch is reached through **conjunct 4** of the promotion gate —
/// the retry runs, reaches `Solve_Succeeded`, and is refused anyway
/// because its residual does not clear the bar in the model's own units.
/// An `acceptable_tol` of `1e-30` is what sets that bar out of reach; no
/// one should run that, and it is not the point. The point is that this
/// is the conjunct with no natural fixture: `qpec_small` promotes and
/// `ralph1` never fires, so without a test here the one check standing
/// between a status and a wrong answer would never execute.
///
/// What it pins is the floor, which is the same three-sink floor
/// `mu_strategy_fallback` uses (pounce#870): a status describing one
/// attempt attached to a point from another is the failure mode it
/// exists to prevent, and a refused retry has already run
/// `finalize_solution` with its own iterate by the time the gate says no.
#[test]
fn a_retry_that_does_not_promote_leaves_the_base_answer_in_place() {
    // Base run with the retry off: the answer the refused run below has
    // to reproduce exactly.
    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    {
        let o = a.options_mut();
        let _ = o.set_numeric_value("acceptable_tol", 1e-30, true, false);
        let _ = o.set_string_value("dual_divergence_retry", "no", true, false);
    }
    let base_status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let base_x = captured.borrow().clone().expect("finalize_solution ran").x;
    let base_obj = a.statistics().final_objective;
    let base_kkt = a.statistics().final_unscaled_kkt_error;

    let (tnlp, captured) = QpecSmallProdEq::new(false);
    let mut a = app(false, 300);
    let _ = a
        .options_mut()
        .set_numeric_value("acceptable_tol", 1e-30, true, false);
    let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
    let x = captured.borrow().clone().expect("finalize_solution ran").x;
    let s = a.statistics();

    assert!(
        s.dual_divergence_signature,
        "the detector should still fire here — only the promotion gate is \
         out of reach ({status:?})",
    );
    assert!(
        !s.dual_divergence_retry_promoted,
        "conjunct 4 was supposed to refuse this retry, and did not \
         ({status:?})",
    );
    assert_eq!(
        status, base_status,
        "the refused retry changed the reported status",
    );
    assert_eq!(x, base_x, "the refused retry changed the returned point");
    assert_eq!(
        s.final_objective, base_obj,
        "the refused retry changed the reported objective",
    );
    assert_eq!(
        s.final_unscaled_kkt_error, base_kkt,
        "the refused retry changed the reported residual",
    );
}

/// gh#945's retry is **gated** because of *this model*, and this test is
/// the measurement that keeps the gate where it is.
///
/// gh#945 is a filter defect: on a model feasible to round-off, every filter
/// entry records a `theta` of a few ulp and later trials are ranked against
/// it on which way the last constraint sum rounded. Allowing the round-off
/// floor of evaluating `c` fixes it. `qpec_small` needs the opposite — the
/// `theta` axis ENFORCED exactly where gh#945 needs it forgiven — and as a
/// *threshold on theta* the two requirements overlap with no gap between
/// them. Swept over a floor applied to every filter decision, this fixture
/// reports `Solve_Succeeded` at an unscaled KKT of 9.96e-8 (the right
/// answer) at floors of 0 and 1e-16, is already lost at **4.44e-16** — which
/// is precisely the floor gh#945's model requires, `5.551115123125783e-16`
/// against an entry's `1.110211922394910e-16` — and at 1e-14 comes back as
/// `Solve_Succeeded` at 5.74e-2, a *false* certificate. Both models sit at
/// `‖x‖∞ ≈ 1` with `theta` at round-off, so a scale-aware floor hands them
/// the same number too.
///
/// What separates them is not a threshold on `theta` at all. The retry as
/// shipped asks a question about the **iterate**: is the point the α-loop
/// gave up at feasible to its own evaluation noise — i.e. does the
/// restoration phase the driver is about to call have anything to minimize?
/// On gh#945's model the answer is no, at every failure: `theta` is 0.0 or
/// one ulp against a floor of 1.2e-15. On this fixture the answer is yes.
/// It has exactly **one** line-search failure in the whole solve, at
///
/// ```text
///   theta = 1.746646e-15      noise floor = 6.664715e-16
/// ```
///
/// 2.6× above its own noise — so the gate declines, no retry runs, and the
/// trajectory is byte-identical. Widened to a fixed floor the gate holds
/// this fixture at 1.2e-15 and 1.5e-15 and loses it at 1.8e-15, 2.0e-15 and
/// above; the shipped floor reads 6.66e-16 here, a factor of 2.6 under the
/// nearest value that costs anything.
///
/// That margin is what this test pins, and it pins it the only way that
/// cannot be satisfied by a coincidence: the retry option off against the
/// retry option on, asserting the solve is **identical** — status, point,
/// objective, residual and iteration count. Anything that widens the gate,
/// raises the floor, or moves this fixture into the band turns one of those
/// five red. (`issue_945_the_retry_is_the_mechanism` in
/// `pounce-rs/tests/issue_945_eq_constrained_lbfgs_filter_noise.rs` is the
/// other half of the pair: it fails if the retry ever stops firing.)
///
/// `ralph1` is not in the same position and does not need the gate: its
/// iterate sits at `3.8e-8`, its `theta` differences are seven orders above
/// their own noise floor, and the floor is scale-aware for that reason. The
/// model that forces the gate is this one.
///
/// **What this is not.** It is a claim about *this* fixture — `qpec_small`
/// under the `prod_eq` lowering, at this file's options — and not about
/// every MPCC. The same model as a `.nl` file under `bound_relax_factor=0
/// mu_strategy_fallback=no` does reach the retry, and there the effect runs
/// the other way: its base solve stops needing gh#884's `perturb_always_cd`
/// promotion at all, converging on its own at `Optimal`/100 with a KKT error
/// of `5.65e-10` (unscaled dual infeasibility `6.26e-10`) against the
/// `9.96e-8` the promotion used to buy. That is
/// pinned in `pounce-cli/tests/issue884_promotion_gate_reads_the_answer.rs`,
/// `the_reproducer_no_longer_needs_the_retry`, alongside
/// `the_reproducer_still_promotes`, which now pins
/// `filter_theta_roundoff_retry=no` so the promotion gate keeps an input to
/// be tested on.
#[test]
fn the_gh945_retry_gate_is_what_this_fixture_relies_on() {
    let run = |retry_on: bool| {
        let (tnlp, captured) = QpecSmallProdEq::new(false);
        let mut a = IpoptApplication::new();
        {
            let o = a.options_mut();
            let _ = o.set_string_value("sb", "yes", true, false);
            let _ = o.set_integer_value("print_level", 0, true, false);
            let _ = o.set_numeric_value("tol", 1e-8, true, false);
            let _ = o.set_numeric_value("bound_relax_factor", 0.0, true, false);
            let _ = o.set_string_value("honor_original_bounds", "yes", true, false);
            let _ = o.set_integer_value("max_iter", 300, true, false);
            let _ = o.set_string_value(
                "filter_theta_roundoff_retry",
                if retry_on { "yes" } else { "no" },
                true,
                false,
            );
        }
        a.initialize().expect("initialize");
        let status = a.optimize_tnlp(Rc::new(RefCell::new(tnlp)));
        let x = captured.borrow().clone().expect("finalize_solution ran").x;
        let s = a.statistics();
        (
            status,
            x,
            s.final_objective,
            s.final_unscaled_kkt_error,
            s.iteration_count,
        )
    };

    // Without the retry — i.e. the pre-gh#945 line search exactly — this
    // fixture reaches a certificate that holds in the model's own units.
    // That is the thing there is something to lose here.
    let (off_status, off_x, off_obj, off_kkt, off_iters) = run(false);
    assert!(
        matches!(off_status, ApplicationReturnStatus::SolveSucceeded),
        "this fixture is supposed to reach a real certificate without the \
         retry; got {off_status:?} at unscaled KKT {off_kkt:.3e}"
    );
    assert!(
        off_kkt <= 1e-6,
        "that certificate is supposed to be real in the model's own units; \
         got {off_kkt:.3e}"
    );

    // With the retry at the shipped default, the gate declines and nothing
    // moves. Five separate reads, because a trajectory change that left the
    // verdict alone is exactly what `scripts/sweep-fixtures.sh` exists to
    // catch and exactly what a status-only assertion would miss.
    let (on_status, on_x, on_obj, on_kkt, on_iters) = run(true);
    assert_eq!(
        on_status, off_status,
        "the gh#945 retry changed this fixture's verdict — the gate is \
         supposed to decline here (theta 1.746646e-15 against a floor of \
         6.664715e-16). Re-measure the table in this test's doc before \
         changing it"
    );
    assert_eq!(on_x, off_x, "the gh#945 retry moved this fixture's point");
    assert_eq!(
        on_obj, off_obj,
        "the gh#945 retry moved this fixture's objective"
    );
    assert_eq!(
        on_kkt, off_kkt,
        "the gh#945 retry moved this fixture's residual"
    );
    assert_eq!(
        on_iters, off_iters,
        "the gh#945 retry changed this fixture's trajectory length \
         ({off_iters} -> {on_iters}) without changing its verdict; that is \
         the class of regression the fixture sweep exists for"
    );
}
