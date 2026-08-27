//! The re-solve verification oracle: check the sensitivity step
//! against an INDEPENDENT warm re-solve, over a span of step sizes.
//!
//! # Why an oracle, when the invariance legs exist
//!
//! `sens_invariance_legs.rs` sweeps three dimensions a shipped defect
//! hid in, and every one of its assertions is an *internal* one: two
//! arms of the same machinery must agree, and both must match a
//! closed-form derivative. That catches a rule that is not invariant.
//! It cannot catch a step that is self-consistently wrong.
//!
//! The defect class with the worst blast radius is exactly that one,
//! and `205bb67`'s own framing names it:
//!
//! > All three turn an essentially exact step into a wrong one while
//! > reporting `improved()` and converged … the residual halved so
//! > nothing warned.
//!
//! Every guard in the crate today reads a number the same machinery
//! produced. This file reads one it did not: a full re-solve at the
//! perturbed parameter, warm-started from the base iterate, at a
//! tolerance two orders tighter than the base solve. Nothing in the
//! sensitivity layer participates in computing it.
//!
//! # The two truth regimes
//!
//! An oracle cannot be hung on every step size, and the reason is not
//! a tolerance to be tuned. Below the barrier width the warm re-solve
//! and the directional-derivative contract genuinely diverge, by an
//! amount of order the row's slack, and **both are correct**. The
//! converged base point sits `O(sqrt(mu))` off the exact solution --
//! at a kink `s · z = mu` with both factors vanishing, so the slack is
//! `sqrt(mu)` -- and the step is measured from there.
//!
//! So the composition is:
//!
//! | regime | truth | owned by |
//! |---|---|---|
//! | `delta >> sqrt(mu)` | the warm re-solve | this file |
//! | `delta << sqrt(mu)` | linearity in the step | `sens_invariance_legs.rs` |
//!
//! A sub-width oracle arm would fail on correct code.
//! [`the_oracle_and_the_step_diverge_below_the_barrier_width`] pins
//! that boundary as a measured fact rather than a comment, so a future
//! reader who tries to extend the oracle downward finds out why it
//! stops where it does.
//!
//! # Why the span, and not a representative step
//!
//! gh#756's defect was correct at every perturbation the out-of-repo
//! 62k-model validation measured (rms at 1% and 10%) and wrong far
//! below them. An oracle placed only at large perturbations would have
//! missed it, the same way the exact-Hessian-only fixture corpus
//! missed gh#677. The legs below sweep the whole span from the width
//! up, and assert first-order agreement at every point of it.
//!
//! # The two fixtures take different branches
//!
//! Per CLAUDE.md: a leg is only evidence about the branch its fixture
//! reaches. These two are chosen to land in different activity classes
//! *and* on different halves of fix-relax:
//!
//! * [`KinkTnlp`] -- a weakly active bound, the degenerate base point.
//!   Exercises the directional decision.
//! * [`ReleaseTnlp`] -- a **strongly** active bound with a multiplier
//!   of 0.6, which a large enough perturbation drives negative. That is
//!   upstream's equation 18, the *release* half of fix-relax, which no
//!   fixture in `sens_invariance_legs.rs` reaches at all.
//!
//! # The oracle is checked before it is trusted
//!
//! An oracle that shares the machinery's defect is worse than no
//! oracle. Both fixtures have a closed-form solution at every
//! parameter, and [`the_resolve_is_an_independent_oracle`] pins the
//! re-solve against it before any leg compares a step to it.
//!
//! # What this file is NOT evidence about
//!
//! Stated because CLAUDE.md's rule cuts both ways: a leg is only
//! evidence about the branch its fixture reaches, and naming the
//! branches these two do not reach is the difference between a guard
//! and a false sense of one.
//!
//! * **Index spaces.** Both fixtures have `m = 1`, so the pin's user-g
//!   index and its KKT row coincide and a defect that confuses them is
//!   invisible here. That dimension is owned by
//!   `cd_split_pin_mapping.rs`, whose fixture puts an inactive
//!   inequality ahead of three equalities on purpose.
//! * **Scaling.** Both fixtures run unit-scaled, so a step that mixes
//!   the algorithm's scaled frame with the model's units reads correct
//!   here. `sens_invariance_legs.rs` leg 1 and
//!   `variable_scaling_sensitivity.rs` own that.
//! * **`improved() == false`.** No measured point on either fixture
//!   reaches it, so the arm asserting that branch in
//!   [`leg_oracle_improved_means_closer_to_the_resolve`] is written but
//!   unexercised. A fixture whose corrector declines to improve is the
//!   missing one.
//! * **Scale.** Every fixture here is three variables. The resource
//!   paths that only appear at 62k (gh#672 f2's `2^n` masks, gh#708
//!   f4's full-length basis columns) are out of reach by construction,
//!   as gh#764 says explicitly.
//! * **O1 at the fine end of its span, taken alone.** The budget is
//!   flat -- `4 · max(floor, width)`, or `2.17e-4` -- while the step
//!   being predicted shrinks with `delta`, so the two entries at
//!   `delta = ±1e-4` do not by themselves establish that the step is
//!   right: the whole exact displacement there is `1.0e-4`, inside the
//!   budget, so a predictor that returned the base point unmoved would
//!   pass them. It is only those two: at `±1e-3` a null step misses by
//!   `1.0e-3`, five budgets out, and the margin grows from there.
//!   O1's power at the fine end therefore comes from the coarse
//!   entries and from
//!   [`leg_oracle_the_plain_step_is_the_two_sided_average_at_a_kink`],
//!   which separates a correct step from the nearest wrong one at the
//!   same magnitudes. Tightening the budget to make `1e-4` bite on its
//!   own would mean squeezing it between the measured error `6.3e-5`
//!   and the null step's `1.0e-4` -- at most `1.58×` of headroom,
//!   trading a robust leg for a flaky one to duplicate power the file
//!   already has.
//!
//! # Mutation evidence
//!
//! These legs pass on `main`, which is what a guard does, so the
//! evidence that they work is a mutation table rather than a red
//! parent commit -- the same discipline `sens_invariance_legs.rs`
//! uses. Reintroduce a defect and only the matching legs go red:
//!
//! | mutation | reintroduces | red here | red elsewhere |
//! |---|---|---|---|
//! | `releasable` returns empty | fix-relax before its release half, "the difference between returning 0.0 and 1.667" | O3 and the corrector-release leg | `boundcheck` unit tests (3), `corrector.rs` (4) |
//! | `parametric_step_bounded` passes `&[]` for the multipliers | a wiring defect: the refinement called without the information a release needs | O3 and the corrector-release leg | `corrector.rs` (4); the `boundcheck` unit tests stay GREEN, since they supply their own multipliers |
//! | `KAPPA_MIN` `1e-3` -> `1e2` | gh#756's class: a genuine kink dropped, so the two-sided step stands | O1 and the two-sided-average leg | 7 of 12 invariance legs, 5 `degenerate_*` files |
//!
//! The separation is the point: the release mutations move only the
//! release legs and the kink mutation moves only the kink legs, so a
//! red leg here names the dimension that broke.
//!
//! Worth recording honestly: **none of the three is caught by this
//! file alone.** The argument for it is not that it finds what nothing
//! else can, but that it is the only guard in the crate that reads a
//! number the sensitivity layer did not produce -- and the defect
//! class gh#764 filed it for is precisely the one that survives every
//! internal check. The second row is the closest demonstration: a
//! defect in the option plumbing, where the algorithm is correct and
//! the call site is not, passes every unit test in the module it
//! breaks.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::Number;
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, Solution, SparsityRequest, StartingPoint,
};
use pounce_sensitivity::Solver;

/// Tolerance of the base solve -- the one whose factor the steps are
/// taken against.
const BASE_TOL: Number = 1e-8;
/// Tolerance of the oracle re-solve. Two orders tighter than the base
/// solve, so the offset the legs measure is the base point's own and
/// not the oracle's. Matches the tolerance the out-of-repo 62k-model
/// validation uses for its truth.
const ORACLE_TOL: Number = 1e-10;
/// Back-solve budget for the directional decision and for the
/// fix-relax refinement. Both fixtures engage at most one row.
const ITER: usize = 16;

// ===============================================================
// Warm-start plumbing
// ===============================================================

/// A converged iterate, captured on the way out of a solve so the next
/// one can start from it.
#[derive(Clone, Debug, Default)]
struct Seed {
    x: Vec<Number>,
    z_l: Vec<Number>,
    z_u: Vec<Number>,
    lambda: Vec<Number>,
}

/// Written by `finalize_solution`, read by the caller.
type SeedOut = Rc<RefCell<Option<Seed>>>;

// ===============================================================
// Fixture 1 -- the kink (gh#762's model, parameterized)
// ===============================================================

/// Coupling between the pin and the kink variable. The one-sided
/// derivative on the leaving side is exactly this.
const A: Number = 1.10;
/// Where the interior variable sits.
const W_STAR: Number = 2.0;

/// ```text
/// min  0.5 k^2 - A p k + 0.5 (w - W_STAR)^2
/// s.t. p = p_nominal,   0 <= k <= 10,   0 <= w <= 10
/// ```
///
/// gh#762's fixture 1, with the pin's right-hand side lifted to a
/// field so the oracle can re-solve at a moved parameter. At
/// `p_nominal = 0` the kink variable `k` sits at its lower bound with
/// a multiplier that vanishes with `mu`: a genuine kink, certified
/// `WEAKLY_ACTIVE`.
///
/// Closed form: `k* = max(0, A p)`, `w* = W_STAR`, `p* = p_nominal`.
struct KinkTnlp {
    p_nominal: Number,
    seed_in: Option<Seed>,
    seed_out: SeedOut,
}

impl KinkTnlp {
    /// The exact solution at `p`, in var-x order `[k, w, p]`.
    fn exact(p: Number) -> [Number; 3] {
        [(A * p).max(0.0), W_STAR, p]
    }
}

impl TNLP for KinkTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 3,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = 0.0;
        b.x_u[1] = 10.0;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = self.p_nominal;
        b.g_u[0] = self.p_nominal;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        seed_or(sp, self.seed_in.as_ref(), &[0.3, 0.5, self.p_nominal])
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (k, w, p) = (x[0], x[1], x[2]);
        Some(0.5 * k * k - A * p * k + 0.5 * (w - W_STAR) * (w - W_STAR))
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (k, w, p) = (x[0], x[1], x[2]);
        g[0] = k - A * p;
        g[1] = w - W_STAR;
        g[2] = -A * k;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
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
                // lower triangle: (k,k), (w,w), (p,k)
                irow.copy_from_slice(&[0, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = -obj_factor * A;
            }
        }
        true
    }

    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        capture(&self.seed_out, &s);
    }
}

// ===============================================================
// Fixture 2 -- a STRONGLY active bound the perturbation releases
// ===============================================================

/// Off-diagonal weight. Nonzero so that releasing `x1` moves `x2`,
/// which is what separates the refinement from a clamp.
const C: Number = 0.5;
/// Linear pull on the free variable.
const B: Number = 2.0;
/// Constant part of the pull on the bounded variable.
const A0: Number = 0.4;
/// The parameter at which `x1`'s bound multiplier reaches zero:
/// `A0 + p = C * B`. Below it `x1` holds at zero; above it `x1` leaves.
const RELEASE_AT: Number = C * B - A0; // 0.6

/// ```text
/// min  0.5 x1^2 + 0.5 x2^2 + C x1 x2 - (A0 + p) x1 - B x2
/// s.t. p = p_nominal,   x1 >= 0,   x2 free
/// ```
///
/// At `p = 0` the stationary `x1` wants to be negative, so `x1` sits
/// on its lower bound with multiplier `z1 = C B - A0 = 0.6`: a
/// **strongly** active bound, nothing like a kink. The linear step
/// preserves complementarity, so it holds `x1` at zero however hard
/// the parameter pulls -- while `dz1/dp = -1` drives the multiplier
/// negative once `p` passes [`RELEASE_AT`]. That is upstream's
/// equation 18, the release half of fix-relax.
///
/// Closed form: below the breakpoint `x1 = 0, x2 = B`; above it
/// `x1 = (A0 + p - C B) / (1 - C^2)` and `x2 = B - C x1`.
struct ReleaseTnlp {
    p_nominal: Number,
    seed_in: Option<Seed>,
    seed_out: SeedOut,
}

impl ReleaseTnlp {
    /// The exact solution at `p`, in var-x order `[x1, x2, p]`.
    fn exact(p: Number) -> [Number; 3] {
        let x1 = ((A0 + p) - C * B) / (1.0 - C * C);
        if x1 <= 0.0 {
            [0.0, B, p]
        } else {
            [x1, B - C * x1, p]
        }
    }
}

impl TNLP for ReleaseTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = self.p_nominal;
        b.g_u[0] = self.p_nominal;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        seed_or(sp, self.seed_in.as_ref(), &[0.2, 1.0, self.p_nominal])
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        Some(0.5 * x1 * x1 + 0.5 * x2 * x2 + C * x1 * x2 - (A0 + p) * x1 - B * x2)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (x1, x2, p) = (x[0], x[1], x[2]);
        g[0] = x1 + C * x2 - (A0 + p);
        g[1] = x2 + C * x1 - B;
        g[2] = -x1;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = 2;
            }
            SparsityRequest::Values { values } => values[0] = 1.0,
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
                // lower triangle: (x1,x1), (x2,x2), (x2,x1), (p,x1)
                irow.copy_from_slice(&[0, 1, 1, 2]);
                jcol.copy_from_slice(&[0, 1, 0, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = obj_factor * C;
                values[3] = -obj_factor;
            }
        }
        true
    }

    fn finalize_solution(&mut self, s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
        capture(&self.seed_out, &s);
    }
}

// ===============================================================
// Shared harness
// ===============================================================

/// Seed the starting point from a converged iterate when one is
/// supplied, otherwise use the fixture's cold start.
///
/// The multipliers matter as much as `x` here: this is what keeps the
/// re-solve in the base point's basin, which is the whole reason
/// gh#764 asks for a warm start rather than a fresh solve. The
/// `init_*` flags are the initializer's request, not the TNLP's
/// choice -- `init_z` and `init_lambda` come back true exactly under
/// `warm_start_init_point=yes` -- so a seed is only fully consumed on
/// an application configured for it.
fn seed_or(sp: StartingPoint<'_>, seed: Option<&Seed>, cold: &[Number]) -> bool {
    match seed {
        Some(s) => {
            sp.x.copy_from_slice(&s.x);
            if sp.init_z {
                sp.z_l.copy_from_slice(&s.z_l);
                sp.z_u.copy_from_slice(&s.z_u);
            }
            if sp.init_lambda {
                sp.lambda.copy_from_slice(&s.lambda);
            }
        }
        None => sp.x.copy_from_slice(cold),
    }
    true
}

fn capture(out: &SeedOut, s: &Solution<'_>) {
    *out.borrow_mut() = Some(Seed {
        x: s.x.to_vec(),
        z_l: s.z_l.to_vec(),
        z_u: s.z_u.to_vec(),
        lambda: s.lambda.to_vec(),
    });
}

/// An application configured the way every arm here runs it.
///
/// `bound_relax_factor = 0` because `weakly_active_bounds` refuses a
/// relaxed-bound solve, and because a relaxed bound would put the
/// oracle's own answer outside the box the step is refined onto --
/// the two must be measured against the same feasible set.
fn configured(tol: Number, warm: bool) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", tol, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    if warm {
        app.options_mut()
            .set_string_value("warm_start_init_point", "yes", true, false)
            .unwrap();
    }
    app.initialize().unwrap();
    app
}

fn run(app: IpoptApplication, tnlp: Rc<RefCell<dyn TNLP>>, what: &str) -> Solver {
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "{what} failed: {status:?}",
    );
    solver
}

/// Which fixture a leg is running. Keeps the harness generic without
/// making the fixtures generic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fx {
    Kink,
    Release,
}

impl Fx {
    fn exact(self, p: Number) -> [Number; 3] {
        match self {
            Fx::Kink => KinkTnlp::exact(p),
            Fx::Release => ReleaseTnlp::exact(p),
        }
    }

    fn build(self, p: Number, seed_in: Option<Seed>, seed_out: SeedOut) -> Rc<RefCell<dyn TNLP>> {
        match self {
            Fx::Kink => Rc::new(RefCell::new(KinkTnlp {
                p_nominal: p,
                seed_in,
                seed_out,
            })),
            Fx::Release => Rc::new(RefCell::new(ReleaseTnlp {
                p_nominal: p,
                seed_in,
                seed_out,
            })),
        }
    }
}

/// The base solve, at `p = 0`, plus its converged iterate.
fn base(fx: Fx) -> (Solver, Seed) {
    let out: SeedOut = Rc::new(RefCell::new(None));
    let solver = run(
        configured(BASE_TOL, false),
        fx.build(0.0, None, Rc::clone(&out)),
        &format!("{fx:?} base solve"),
    );
    let seed = out.borrow().clone().expect("base seed");
    (solver, seed)
}

/// **The oracle.** A full re-solve at the perturbed parameter,
/// warm-started from the base iterate, at [`ORACLE_TOL`].
///
/// Nothing in `pounce-sensitivity` takes part in producing this: it is
/// the IPM solving a different problem from a good starting point. That
/// independence is the entire value of the leg.
fn oracle(fx: Fx, seed: &Seed, delta: Number) -> Vec<Number> {
    let out: SeedOut = Rc::new(RefCell::new(None));
    let solver = run(
        configured(ORACLE_TOL, true),
        fx.build(delta, Some(seed.clone()), Rc::clone(&out)),
        &format!("{fx:?} oracle re-solve at delta={delta:e}"),
    );
    let x = solver.converged().expect("oracle converged").x.clone();
    x
}

/// Barrier parameter the base solve stopped at, and the width
/// `sqrt(mu)` that separates the two truth regimes.
fn mu_and_width(s: &Solver) -> (Number, Number) {
    let mu = s.classify_activity().expect("classify").mu;
    (mu, mu.max(0.0).sqrt())
}

/// Largest absolute difference over the primal block.
fn dist(a: &[Number], b: &[Number]) -> Number {
    assert_eq!(a.len(), b.len(), "length mismatch: {a:?} vs {b:?}");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, Number::max)
}

fn add(base: &[Number], step: &[Number]) -> Vec<Number> {
    base.iter().zip(step.iter()).map(|(&a, &b)| a + b).collect()
}

/// The predicted point under each mode, as `base_x + step`.
fn predicted_linear(s: &Solver, delta: Number) -> Vec<Number> {
    let base_x = s.converged().expect("converged").x.clone();
    let d = s.parametric_step(&[0], &[delta]).expect("linear step");
    add(&base_x, &d)
}

fn predicted_directional(s: &Solver, delta: Number) -> Vec<Number> {
    let base_x = s.converged().expect("converged").x.clone();
    let (d, _held, _work) = s
        .parametric_step_directional(&[0], &[delta], ITER)
        .expect("directional step");
    add(&base_x, &d)
}

fn predicted_fix_relax(s: &Solver, delta: Number) -> Vec<Number> {
    let base_x = s.converged().expect("converged").x.clone();
    let (d, _pinned, _stop) = s
        .parametric_step_bounded(&[0], &[delta], ITER, None)
        .expect("fix-relax step");
    add(&base_x, &d)
}

// ===============================================================
// Preconditions -- an oracle that is not independent proves nothing
// ===============================================================

/// The oracle must be right before anything is compared against it.
///
/// Both fixtures have a closed-form solution at every parameter, so
/// this is checkable rather than assumed. If the re-solve were subtly
/// wrong -- warm start landing in a different basin, the pin not
/// carrying the parameter -- every leg below would still pass by
/// agreeing with it, which is the failure mode that makes a bad oracle
/// worse than none.
/// The oracle is a barrier solution too, so it is exact only to its
/// own slack -- `mu_oracle / z` on a bound whose multiplier `z` is
/// healthy, and `sqrt(mu_oracle)` where `z` vanishes. On the kink
/// fixture that slack is a function of how far the parameter is from
/// the kink: 9.4e-6 sitting exactly on it, 2.2e-8 at `|delta| = 1e-3`,
/// and 1e-10 by `|delta| = 0.1`. Measured, not assumed -- hence two
/// bounds rather than one.
const ORACLE_EXACT_AT_KINK: Number = 1e-5;
/// Away from the kink, and everywhere on the release fixture, whose
/// bound multiplier never vanishes.
const ORACLE_EXACT_AWAY: Number = 1e-7;

#[test]
fn the_resolve_is_an_independent_oracle() {
    for fx in [Fx::Kink, Fx::Release] {
        let (_s, seed) = base(fx);
        for &delta in &[-2.0, -0.5, -1e-1, -1e-3, 0.0, 1e-3, 1e-1, 0.5, 1.0, 2.0] {
            let got = oracle(fx, &seed, delta);
            let want = fx.exact(delta);
            let err = dist(&got, &want);
            // The loose bound applies only where the fixture's own
            // multiplier is vanishing, i.e. the kink fixture at the
            // kink. Everywhere else the tight one holds, so a
            // regression cannot hide behind the loose number.
            let at_kink = fx == Fx::Kink && delta.abs() < 1e-2;
            let tol = if at_kink {
                ORACLE_EXACT_AT_KINK
            } else {
                ORACLE_EXACT_AWAY
            };
            assert!(
                err <= tol,
                "{fx:?}: the oracle must reproduce the closed form at \
                 delta={delta:e} to {tol:e}, off by {err:e}\n  \
                 got  {got:?}\n  want {want:?}",
            );
        }
    }
}

/// The oracle has to out-resolve what the legs measure with it, or an
/// agreement between step and re-solve says nothing.
///
/// The kink fixture's base point sits `5.4e-5` off the oracle -- that
/// is the `floor` every leg budgets against, and it is a distance
/// between two barrier solutions, not from the exact one (the base
/// point is `6.4e-5` from exact, the oracle `9.4e-6`).
///
/// The oracle stays at least **two** orders finer than that floor
/// across the span, which is what the assert enforces. The margin is
/// not uniform and the fine end is the binding one: measured
/// `floor / err` runs `6.0e5` at `delta = 1e-1`, `4.4e4` at `1e-2`,
/// `2.5e3` at `1e-3`, and `5.4e2` at `1e-4` -- the re-solve has less
/// parameter distance to work with as `delta` shrinks. If that margin
/// ever closes, the legs stop being able to tell a correct step from a
/// wrong one and this test says so first.
#[test]
fn the_oracle_out_resolves_the_base_offset() {
    let (s, seed) = base(Fx::Kink);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Kink, &seed, 0.0));

    for &mag in &SPAN {
        for sign in [1.0, -1.0] {
            let delta = sign * mag;
            let err = dist(&oracle(Fx::Kink, &seed, delta), &KinkTnlp::exact(delta));
            assert!(
                err < floor / 100.0,
                "the oracle must out-resolve the base offset by two \
                 orders at delta={delta:e}: oracle off by {err:e}, base \
                 offset {floor:e}",
            );
        }
    }
}

/// The two fixtures must land in different activity classes, or the
/// second one is a duplicate of the first and the release half of
/// fix-relax stays untested.
#[test]
fn the_fixtures_take_different_branches() {
    let (kink, _) = base(Fx::Kink);
    let weak = kink.weakly_active_bounds().expect("weak set");
    assert!(
        weak.iter().any(|b| b.var_row == 0 && b.lower),
        "the kink fixture's bound must be weakly active: {weak:?}",
    );

    let (rel, _) = base(Fx::Release);
    let weak = rel.weakly_active_bounds().expect("weak set");
    assert!(
        weak.is_empty(),
        "the release fixture's bound is strongly active and must NOT be \
         weak -- otherwise this fixture reaches the same branch as the \
         kink and proves nothing new: {weak:?}",
    );

    // and it really is on its bound, with a multiplier far from zero
    let x = rel.converged().expect("converged").x.clone();
    assert!(
        x[0].abs() < 1e-6,
        "x1 sits on its lower bound, got {}",
        x[0]
    );
    let report = rel.classify_activity().expect("classify");
    assert_eq!(
        report.var_status[0],
        pounce_sensitivity::activity::STRONGLY_ACTIVE,
        "x1's bound must be certified strongly active, ratio {}",
        report.var_ratio[0],
    );
}

/// The base point's offset **from the oracle** is the floor every
/// comparison below is measured against, and it is `O(sqrt(mu))` at a
/// kink because `s · z = mu` with both factors vanishing.
///
/// Note which two points that is. `floor` is the distance between two
/// *barrier* solutions at different `mu`, not the distance from the
/// exact one: measured, the base point sits `6.4e-5` from exact, the
/// oracle `9.4e-6`, and the floor between them is `5.4e-5`. The legs
/// budget against the floor because the oracle is what they compare a
/// step to.
///
/// Pinned here so a leg never fails for a reason its name does not
/// describe: if the base solve ever stops landing this close, the legs
/// would read as sensitivity defects.
#[test]
fn the_base_offset_is_the_barrier_width() {
    let (s, seed) = base(Fx::Kink);
    let (mu, width) = mu_and_width(&s);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Kink, &seed, 0.0));

    assert!(
        floor <= 20.0 * width,
        "the base point should sit within a small multiple of the \
         barrier width of the oracle: floor {floor:e}, \
         width {width:e}, mu {mu:e}",
    );
    assert!(
        floor > 0.1 * width,
        "and the offset should BE the barrier width, not something \
         smaller -- if it is, the two regimes do not divide where this \
         file says they do: floor {floor:e}, width {width:e}",
    );
}

// ===============================================================
// Leg O1 -- first-order agreement above the barrier width
// ===============================================================

/// Step sizes spanning the region where the re-solve is the truth,
/// four orders of it. The leg asserts every entry sits above the
/// measured width rather than assuming it, so a change that widens the
/// barrier (a looser `BASE_TOL`, say) fails loudly here instead of
/// quietly moving an arm into the regime this file does not own.
const SPAN: [Number; 4] = [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4];

/// The step must reproduce the re-solve to first order, on both sides
/// of the kink, at every magnitude above the barrier width.
///
/// The comparison is `|predicted - resolved| <= 4 · max(floor, width)`
/// -- a flat budget, with no term in `delta`. Both fixtures are QPs
/// whose exact solution is affine in the parameter on each side, so a
/// correct first-order step carries no second-order error to allow
/// for, and the only slack it needs is the base point's own offset
/// from the oracle. A budget growing with `delta` would forgive
/// exactly the first-order error this leg exists to catch.
///
/// Measured, a correct directional step spends none of that budget on
/// the step and all of it on the floor: its error is `6.36e-5` at every
/// magnitude here -- constant, because it *is* the base offset --
/// against a budget of `2.17e-4`.
#[test]
fn leg_oracle_the_step_reproduces_the_resolve_above_the_barrier_width() {
    let (s, seed) = base(Fx::Kink);
    let (_mu, width) = mu_and_width(&s);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Kink, &seed, 0.0));
    let budget = 4.0 * floor.max(width);

    for sign in [1.0, -1.0] {
        for &mag in &SPAN {
            let delta = sign * mag;
            assert!(
                mag > width,
                "the span must stay above the width: {mag:e} vs {width:e}",
            );
            let got = predicted_directional(&s, delta);
            let want = oracle(Fx::Kink, &seed, delta);
            let err = dist(&got, &want);
            assert!(
                err <= budget,
                "directional step vs re-solve at delta={delta:e}: off by \
                 {err:e}, budget {budget:e} (floor {floor:e}, width \
                 {width:e})\n  predicted {got:?}\n  resolved  {want:?}",
            );
        }
    }
}

/// The separation: the plain step is the **two-sided average** at a
/// kink, so the oracle must be able to see it being wrong.
///
/// This leg is what keeps the one above from being vacuous. An oracle
/// that agreed with every mode would prove nothing; here the plain
/// predictor is off by half the step -- `A/2 · delta` against the
/// exact `A · delta` on the leaving side and `0` on the holding one --
/// because the barrier system's linearization at a weakly active bound
/// sits strictly between the two one-sided values at every `mu`.
/// Tightening `tol` does not move it toward either. That is the
/// documented reason the directional mode exists, and measuring it
/// against the re-solve turns it from a claim into a number.
///
/// Restricted to `delta >= 1e-3`. Below that the two-sided average and
/// the one-sided answer are separated by less than the base point's
/// own slack, so the characterization stops holding -- at `1e-4` the
/// plain step's error is `1.3e-6` where `A/2 · delta` would be
/// `5.5e-5`. That crossover is the same barrier width leg O2 maps, and
/// it is a fact about the fixture, not about the predictor.
#[test]
fn leg_oracle_the_plain_step_is_the_two_sided_average_at_a_kink() {
    let (s, seed) = base(Fx::Kink);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Kink, &seed, 0.0));

    for sign in [1.0, -1.0] {
        for &mag in &[1.0e-1, 1.0e-2, 1.0e-3] {
            let delta = sign * mag;
            let err = dist(
                &predicted_linear(&s, delta),
                &oracle(Fx::Kink, &seed, delta),
            );
            let want = 0.5 * A * mag;
            assert!(
                (err - want).abs() <= 2.0 * floor,
                "the plain step should miss the re-solve by half the \
                 step at a kink (delta={delta:e}): off by {err:e}, \
                 expected about {want:e} +/- {:e}",
                2.0 * floor,
            );
            // and the directional decision must actually repair it,
            // or the pair above is measuring one mode twice.
            //
            // The two errors are different *kinds* of quantity, which
            // is the sharper statement than any ratio: the plain
            // step's tracks `delta`, while the directional step's is
            // the constant base offset. So the separation is widest at
            // the coarse end (864x at 1e-1) and narrowest at the fine
            // end (7.8x at 1e-3), and asserting a flat ratio would
            // encode whichever end the list happens to stop at.
            let fixed = dist(
                &predicted_directional(&s, delta),
                &oracle(Fx::Kink, &seed, delta),
            );
            assert!(
                fixed <= 2.0 * floor,
                "the directional step's error should be the base offset \
                 and nothing more, at every delta: delta={delta:e}, \
                 directional {fixed:e}, floor {floor:e}",
            );
            assert!(
                err > 5.0 * fixed,
                "the directional decision must repair what the plain \
                 step misses at delta={delta:e}: plain {err:e}, \
                 directional {fixed:e}",
            );
        }
    }
}

// ===============================================================
// Leg O2 -- the boundary between the two truth regimes
// ===============================================================

/// Below the barrier width the oracle stops being the truth, and this
/// pins that as a measured fact.
///
/// Two things are asserted, and the second is the one that matters:
///
/// 1. The predicted point never wanders: `|predicted - resolved|`
///    stays at the floor at every magnitude, including far below the
///    width. A step that drifts would show here.
/// 2. The *first-order* content is swamped below the width --
///    `err / |delta|` grows without bound as `delta` shrinks, while
///    above the width it stays small. So an oracle arm placed below
///    the width would fail on correct code, which is why this file
///    does not place one there.
///
/// The invariance legs own that region instead, comparing slopes,
/// which cancels the floor exactly.
#[test]
fn the_oracle_and_the_step_diverge_below_the_barrier_width() {
    let (s, seed) = base(Fx::Kink);
    let (mu, width) = mu_and_width(&s);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Kink, &seed, 0.0));

    let mut ratios: Vec<(Number, Number, Number)> = Vec::new();
    for &mag in &[1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10] {
        let got = predicted_directional(&s, mag);
        let want = oracle(Fx::Kink, &seed, mag);
        let err = dist(&got, &want);
        // (1) the point itself never wanders past the floor
        assert!(
            err <= 4.0 * floor.max(width),
            "the predicted point must stay at the floor at every \
             magnitude, including below the width: delta={mag:e}, err \
             {err:e}, floor {floor:e}",
        );
        ratios.push((mag, err, err / mag));
    }

    // (2) the relative content crosses over near the width.
    //
    // The crossover is not a cliff: `err / delta` is essentially
    // `floor / delta`, so it passes 1 where `delta` passes the floor
    // and the two regimes are separated by a band, not a point. That
    // band is left deliberately unasserted -- claiming a sharp edge
    // where the measurement shows a gradual one is how a threshold
    // ends up encoding a fixture instead of a fact. `MARGIN` is how
    // far outside it a step size has to sit to be claimed for either
    // regime.
    const MARGIN: Number = 100.0;
    let above = ratios
        .iter()
        .filter(|&&(m, _, _)| m > MARGIN * width)
        .map(|&(_, _, r)| r)
        .fold(0.0, Number::max);
    let below = ratios
        .iter()
        .filter(|&&(m, _, _)| m < width)
        .map(|&(_, _, r)| r)
        .fold(0.0, Number::max);
    assert!(
        ratios.iter().any(|&(m, _, _)| m > MARGIN * width)
            && ratios.iter().any(|&(m, _, _)| m < width),
        "the sweep must reach both regimes, or this leg asserts \
         nothing: width {width:e}\n{ratios:?}",
    );
    assert!(
        above < 1.0e-1,
        "above the width the re-solve is the truth and the step should \
         track it to first order: worst err/delta {above:e}\n{ratios:?}",
    );
    assert!(
        below > 1.0,
        "below the width the two genuinely diverge by the row's slack, \
         so an oracle arm placed here would fail on correct code. If \
         this assertion ever goes red, the regimes no longer divide at \
         sqrt(mu) and this file's boundary needs re-deriving \
         (mu {mu:e}, width {width:e}): worst err/delta {below:e}\n{ratios:?}",
    );
}

// ===============================================================
// Leg O3 -- the strongly active release
// ===============================================================

/// The release half of fix-relax, checked against the re-solve on both
/// sides of the breakpoint.
///
/// Below [`RELEASE_AT`] nothing changes and every mode agrees. Above
/// it the exact answer leaves the bound, the linear step cannot follow
/// (it preserves complementarity), and fix-relax must. Both halves are
/// asserted: the agreement below, and the *separation* above -- the
/// leg fails if the linear step ever stops being wrong there, because
/// then the fixture has stopped exercising the release path and the
/// fix-relax assertion has gone vacuous.
#[test]
fn leg_oracle_fix_relax_reproduces_the_resolve_across_a_strongly_active_release() {
    let (s, seed) = base(Fx::Release);
    let base_x = s.converged().expect("converged").x.clone();
    let floor = dist(&base_x, &oracle(Fx::Release, &seed, 0.0));
    let budget = 100.0 * floor.max(1e-9);

    // --- below the breakpoint: the bound stays on, everything agrees
    for &delta in &[0.1, 0.3, 0.5] {
        assert!(delta < RELEASE_AT);
        let want = oracle(Fx::Release, &seed, delta);
        for (what, got) in [
            ("linear", predicted_linear(&s, delta)),
            ("fix_relax", predicted_fix_relax(&s, delta)),
        ] {
            let err = dist(&got, &want);
            assert!(
                err <= budget,
                "{what} vs re-solve below the breakpoint at \
                 delta={delta:e}: off by {err:e}, budget {budget:e}\n  \
                 predicted {got:?}\n  resolved  {want:?}",
            );
        }
    }

    // --- above it: the bound releases
    for &delta in &[0.7, 1.0, 2.0] {
        assert!(delta > RELEASE_AT);
        let want = oracle(Fx::Release, &seed, delta);
        let exact_x1 = ((A0 + delta) - C * B) / (1.0 - C * C);

        // the separation: the linear step holds x1 at its bound, so it
        // is wrong by the whole released distance
        let lin = predicted_linear(&s, delta);
        let lin_err = dist(&lin, &want);
        assert!(
            lin_err > 0.5 * exact_x1,
            "precondition: above the breakpoint the linear step must be \
             visibly wrong, or this fixture is not exercising the \
             release path: delta={delta:e}, linear err {lin_err:e}, \
             released distance {exact_x1:e}",
        );

        // The directional decision does NOT cover this: with no
        // weakly active row there is nothing for it to decide, so it
        // returns the plain step. Pinned because it is the reason
        // this fixture had to exist -- gh#762's legs all run through
        // the directional path, and none of them reaches the release.
        let dir = predicted_directional(&s, delta);
        assert!(
            dist(&dir, &lin) <= 1e-12,
            "with no weak rows the directional mode must be the plain \
             step: delta={delta:e}, directional {dir:?}, linear {lin:?}",
        );

        // and fix-relax must reproduce the re-solve
        let got = predicted_fix_relax(&s, delta);
        let err = dist(&got, &want);
        assert!(
            err <= budget,
            "fix_relax vs re-solve across the release at delta={delta:e}: \
             off by {err:e}, budget {budget:e}\n  predicted {got:?}\n  \
             resolved  {want:?}",
        );
    }
}

// ===============================================================
// Leg O4 -- what the corrector REPORTS, against the oracle
// ===============================================================

/// `improved()` must mean improved *toward the re-solve*, not merely
/// toward a smaller residual.
///
/// This is the leg that reaches the defect class the file exists for.
/// `205bb67` turned an essentially exact step into a wrong one "while
/// reporting `improved()` and converged … the residual halved so
/// nothing warned". Residual is the machinery's own opinion of itself;
/// distance to an independent re-solve is not. Where the two disagree
/// the report is misleading, and only a leg holding both can see it.
///
/// The comparison carries a noise term. Below the base point's own
/// offset, "closer" and "further" are not distinguishable -- the
/// measured case is `Fx::Release` at `delta = 0.3`, where the
/// corrector drops the residual 300-fold and moves the point from
/// `2.1e-9` to `8.3e-9`, both far under the `4.2e-9` floor. Asserting
/// on that difference would be asserting on noise.
///
/// **Which branch this leg reaches:** every measured point on both
/// fixtures reports `improved() == true`, so the `false` arm below is
/// written for correctness but is *not* exercised by this corpus, and
/// this file is no evidence about it. Per CLAUDE.md that is worth
/// saying rather than leaving for the next reader to discover: a
/// fixture whose corrector declines to improve is the missing one.
#[test]
fn leg_oracle_improved_means_closer_to_the_resolve() {
    let mut saw_improved = false;

    for (fx, deltas) in [
        (Fx::Kink, vec![-1.0e-1, -1.0e-2, 1.0e-2, 1.0e-1]),
        (Fx::Release, vec![0.3, 0.7, 1.0, 2.0]),
    ] {
        let (s, seed) = base(fx);
        let base_x = s.converged().expect("converged").x.clone();
        let n_x = base_x.len();
        let floor = dist(&base_x, &oracle(fx, &seed, 0.0));
        let noise = 2.0 * floor.max(1e-10);

        for delta in deltas {
            let want = oracle(fx, &seed, delta);
            let step = s.parametric_step_full(&[0], &[delta]).expect("full step");
            let before = dist(&add(&base_x, &step[..n_x]), &want);

            let (refined, report) = s.correct_step(&[0], &[delta], &step, 8).expect("corrector");
            let after = dist(&add(&base_x, &refined[..n_x]), &want);

            if report.improved() {
                saw_improved = true;
                assert!(
                    after <= before * 1.05 + noise,
                    "{fx:?} at delta={delta:e}: the corrector reported an \
                     improvement (residual {:e} -> {:e}) while moving the \
                     point AWAY from the re-solve ({before:e} -> \
                     {after:e}, noise {noise:e}). That is the gh#764 \
                     defect class: the residual is the machinery's own \
                     opinion of itself, the re-solve is not.",
                    report.initial_residual,
                    report.residual,
                );
            } else {
                // Documented on `correct_step`: the step handed back is
                // then the caller's own, so the distance cannot move.
                assert!(
                    (after - before).abs() <= before * 1e-6 + noise,
                    "{fx:?} at delta={delta:e}: improved() is false, so \
                     correct_step must hand back the caller's own step \
                     unchanged, but the distance to the re-solve moved \
                     {before:e} -> {after:e}",
                );
            }
        }
    }

    assert!(
        saw_improved,
        "no measured point reported improved(), so this leg asserted \
         nothing about the branch it exists for",
    );
}

/// A falling residual is not a converging answer, and here is the
/// measured proof.
///
/// Across a strongly active release the corrector cannot reach the
/// re-solve: the held barrier diagonal has no way to represent a bound
/// leaving the active set, which `correct_step`'s own doc says. What
/// it *does* do is report `improved()` and drop the residual, because
/// the residual it measures is the one it can still reduce. On this
/// fixture at `delta = 0.7` that is a residual falling by `3e-8` while
/// the point stays `0.1333` from the truth -- and `0.1333` is the
/// entire quantity at stake, the whole distance `x1` should have
/// travelled off its bound.
///
/// This is the gh#764 thesis stated as a number: `improved()` plus a
/// converged residual does **not** imply the answer is close, and no
/// internal guard can tell you that. Only the re-solve can.
///
/// Pinned deliberately, in the manner of gh#762's
/// `the_coupled_fixture_carries_an_ambiguous_kink`: this asserts
/// *current* behaviour, so a corrector that learns to cross a release
/// makes it fail, and that failure is the signal to update it on
/// purpose rather than a regression.
#[test]
fn the_corrector_reports_improved_without_crossing_a_release() {
    let (s, seed) = base(Fx::Release);
    let base_x = s.converged().expect("converged").x.clone();
    let n_x = base_x.len();

    for &delta in &[0.7, 1.0, 2.0] {
        assert!(delta > RELEASE_AT);
        let released = ((A0 + delta) - C * B) / (1.0 - C * C);
        let want = oracle(Fx::Release, &seed, delta);
        let step = s.parametric_step_full(&[0], &[delta]).expect("full step");
        let (refined, report) = s.correct_step(&[0], &[delta], &step, 8).expect("corrector");
        let after = dist(&add(&base_x, &refined[..n_x]), &want);

        assert!(
            report.improved(),
            "delta={delta:e}: the corrector is expected to report an \
             improvement here -- that is what makes the gap below \
             worth pinning",
        );
        assert!(
            after > 0.5 * released,
            "delta={delta:e}: the corrector is not expected to cross a \
             strongly active release. If it now does, this is the good \
             kind of red: the held barrier diagonal has gained a way to \
             represent a bound leaving the active set, and this test \
             should be updated deliberately. Distance to the re-solve \
             {after:e}, released distance {released:e}, residual {:e} -> \
             {:e}",
            report.initial_residual,
            report.residual,
        );
        // and the mode that CAN cross it gets there, so the gap above
        // is the corrector's and not the fixture's
        let fixed = dist(&predicted_fix_relax(&s, delta), &want);
        assert!(
            fixed < 1e-6,
            "delta={delta:e}: fix_relax must reach the re-solve here, \
             off by {fixed:e}",
        );
    }
}
