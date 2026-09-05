//! Three invariance legs for the sensitivity layer, over one fixture
//! carrying a genuine kink.
//!
//! Each leg exists because a shipped defect got through a corpus that
//! was uniform in exactly the dimension the defect lived in. The legs
//! sweep that dimension the way `sweep-fixtures.sh` sweeps the exact
//! and L-BFGS Hessians.
//!
//! 1. **Scaling.** The corrector mixed the algorithm's scaled frame
//!    with the model's own units, and the mix is invisible at unit
//!    scaling -- "which coincide only at unit scaling, which is every
//!    fixture it had" (`205bb67`). The same gap reaches *membership*
//!    rules: under `x̃ = d ⊙ x` the barrier diagonal carries `d^-2`, so
//!    a threshold compared against a bare `Sigma` calls the same bound
//!    a different kind of bound in different units.
//!    `variable_scaling_sensitivity.rs` already runs this leg for
//!    `classify_activity`, but its fixture has no weakly active bound,
//!    so the degeneracy surface is untested there.
//!
//! 2. **Perturbation magnitude.** gh#672 finding 4: an acceptance
//!    tolerance was absolute on a quantity that scales with the
//!    perturbation, so a step of `1e-10` cleared feasibility
//!    everywhere and the holding side's derivative read `-1` instead
//!    of `0`. The leg sweeps `delta` over eight orders on both sides.
//!    It runs over TWO fixtures, because the rule that engages a weak
//!    row branches on the classifier's verdict: the certified
//!    `WEAKLY_ACTIVE` kink below, and a coupled kink that lands in
//!    `AMBIGUOUS` ([`CoupledKinkTnlp`]). A rule exact in one class and
//!    length-based in the other is invisible to a corpus carrying only
//!    the first -- which is the same shape as every entry above.
//!
//! 3. **Fixed variable ahead of the kink.** gh#672 finding 1, the
//!    gh#450 hazard: full-x and var-x diverge from the first
//!    `make_parameter`-removed variable on, and reading one as the
//!    other returns a NEIGHBORING variable's answer -- plausible and
//!    wrong. The leg puts a fixed variable in front of the kink and
//!    requires every var-x answer to be unmoved by it.
//!
//! Every public accessor the sensitivity layer grows gets a row in
//! each leg it has a dimension in. `reduced_activity` (gh#763) is in
//! legs 1 and 3 -- it converts frames and it maps full-x to var-x --
//! and its own contract, including the class disagreement that
//! motivates it, is in
//! [`the_reduced_normalizer_certifies_a_coupled_kink_at_every_coupling`]
//! and in `reduced_activity.rs`.
//!
//! # What the legs compare
//!
//! Not `dx / delta`. The parametric step is affine in `delta`, not
//! linear: it carries a base-point term of order `mu`, because the
//! converged iterate sits that far off the exact solution and the step
//! corrects it (see [`the_step_is_affine_in_delta`], which pins the
//! size of that term). Dividing by `delta` therefore inflates it as
//! `delta` shrinks -- at `1e-10` it is the whole answer -- so the
//! invariant is the *slope*, taken as a difference quotient between
//! two perturbations on the same side of the kink. The constant
//! cancels and what remains is the one-sided derivative, which is what
//! every leg here asserts.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::TNLP;
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, NlpInfo, ScalingRequest, Solution, SparsityRequest,
    StartingPoint,
};
use pounce_sensitivity::activity::{AMBIGUOUS, FIXED, INACTIVE, STRONGLY_ACTIVE, WEAKLY_ACTIVE};
use pounce_sensitivity::{Solver, SolverError};

/// Coupling between the pin and the kink variable. The one-sided
/// derivative on the leaving side is exactly this.
const A: Number = 1.10;
/// Where the interior variable sits. Far enough from either bound that
/// no membership rule should ever reach it.
const W_STAR: Number = 2.0;
/// The value the leading fixed variable is pinned to.
const FIXED_AT: Number = 1.5;
/// Back-solve budget for the directional decision. The fixture engages
/// at most one weak row, so this is generous.
const DEGENERACY_ITER: usize = 16;

/// ```text
/// min  0.5 k^2 - A p k + 0.5 (w - W_STAR)^2  [+ 0.5 (xf - 0.7)^2]
/// s.t. p = 0,   0 <= k <= 10,   0 <= w <= 10  [, xf == FIXED_AT]
/// ```
///
/// At `p = 0` the kink variable `k` sits at its lower bound with a
/// multiplier that vanishes with `mu`: a genuine kink. Moving the pin
/// up lets `k` follow at `A`; moving it down would drive `k` through
/// its bound, so `k` holds and the derivative is `0`. `w` is interior
/// and decoupled from the pin -- it must never be called weak and must
/// never move.
///
/// `leading_fixed` prepends `xf`, whose equal bounds make
/// `fixed_variable_treatment=make_parameter` (the default) remove its
/// column, so full-x and var-x diverge in front of everything
/// interesting.
struct KinkTnlp {
    /// Per-variable factors to report, or `None` to decline scaling.
    x_scaling: Option<Vec<Number>>,
    leading_fixed: bool,
}

impl KinkTnlp {
    fn new(x_scaling: Option<Vec<Number>>, leading_fixed: bool) -> Self {
        Self {
            x_scaling,
            leading_fixed,
        }
    }

    /// full-x offset of the logical block: 1 when a fixed variable
    /// leads, 0 otherwise.
    fn off(&self) -> usize {
        usize::from(self.leading_fixed)
    }
}

impl TNLP for KinkTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: (3 + self.off()) as Index,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: (3 + self.off()) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(d) = self.x_scaling.as_ref() else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = true;
        req.x_scaling.copy_from_slice(d);
        *req.use_g_scaling = false;
        true
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            b.x_l[0] = FIXED_AT;
            b.x_u[0] = FIXED_AT;
        }
        b.x_l[o] = 0.0;
        b.x_u[o] = 10.0;
        b.x_l[o + 1] = 0.0;
        b.x_u[o + 1] = 10.0;
        b.x_l[o + 2] = -1.0e19;
        b.x_u[o + 2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            sp.x[0] = FIXED_AT;
        }
        sp.x[o] = 0.3;
        sp.x[o + 1] = 0.5;
        sp.x[o + 2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let o = self.off();
        let (k, w, p) = (x[o], x[o + 1], x[o + 2]);
        let mut f = 0.5 * k * k - A * p * k + 0.5 * (w - W_STAR) * (w - W_STAR);
        if self.leading_fixed {
            f += 0.5 * (x[0] - 0.7) * (x[0] - 0.7);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        let (k, w, p) = (x[o], x[o + 1], x[o + 2]);
        if self.leading_fixed {
            g[0] = x[0] - 0.7;
        }
        g[o] = k - A * p;
        g[o + 1] = w - W_STAR;
        g[o + 2] = -A * k;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        g[0] = x[self.off() + 2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow[0] = 0;
                jcol[0] = (self.off() + 2) as Index;
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
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (k,k), (w,w), (p,k) [, (xf,xf)]
                let mut rs: Vec<Index> = vec![o, o + 1, o + 2];
                let mut cs: Vec<Index> = vec![o, o + 1, o];
                if self.leading_fixed {
                    rs.push(0);
                    cs.push(0);
                }
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                values[2] = -obj_factor * A;
                if self.leading_fixed {
                    values[3] = obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// Both arms of every leg run under `user-scaling`, so the ONLY
/// difference between them is whether the TNLP hands factors back --
/// the option, the objective factor and the row factors are identical.
/// `weakly_active_bounds` refuses a relaxed-bound solve, so the relax
/// factor is off.
fn solved(x_scaling: Option<Vec<Number>>, leading_fixed: bool) -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> =
        Rc::new(RefCell::new(KinkTnlp::new(x_scaling, leading_fixed)));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "base solve failed: {status:?}",
    );
    solver
}

/// The weak set as `(var_row, lower)` pairs, sorted so the comparison
/// does not depend on the order the rows happen to be walked in.
fn weak_set(s: &Solver) -> Vec<(usize, bool)> {
    let mut v: Vec<(usize, bool)> = s
        .weakly_active_bounds()
        .expect("weak set")
        .iter()
        .map(|b| (b.var_row, b.lower))
        .collect();
    v.sort();
    v
}

/// The reduced-normalizer verdict for one **user-space** variable, as
/// `(status, ratio, q_reduced)` (gh#763). The index is full-x, like
/// every other report index; the accessor maps it to its factor row.
fn reduced(s: &Solver, full_x: usize) -> (i8, Number, Number) {
    let r = s.reduced_activity(&[full_x]).expect("reduced activity");
    (r.status[0], r.ratio[0], r.q_reduced[0])
}

/// `parametric_step_directional` over the single pin, at `delta`.
fn step(s: &Solver, delta: Number) -> Vec<Number> {
    let (d, _held, _work) = s
        .parametric_step_directional(&[0], &[delta], DEGENERACY_ITER)
        .unwrap_or_else(|e| panic!("directional step at delta={delta:e}: {e:?}"));
    d
}

/// The one-sided derivative on `sign`'s side, as the difference
/// quotient between two perturbations there. See the module docs: the
/// step is affine in `delta`, so this cancels the base-point constant
/// and `step(delta) / delta` would not.
fn derivative(s: &Solver, sign: Number) -> Vec<Number> {
    slope(s, sign * 1.0e-3, sign * 1.0e-6)
}

fn slope(s: &Solver, d_hi: Number, d_lo: Number) -> Vec<Number> {
    let hi = step(s, d_hi);
    let lo = step(s, d_lo);
    hi.iter()
        .zip(lo.iter())
        .map(|(&a, &b)| (a - b) / (d_hi - d_lo))
        .collect()
}

/// The exact one-sided derivative of this fixture, in var-x order
/// `[k, w, p]`. The pin moves `p` at unit rate; `k` follows at `A`
/// where that is feasible and holds at its bound where it is not; `w`
/// is decoupled.
fn exact_derivative(sign: Number) -> [Number; 3] {
    [if sign > 0.0 { A } else { 0.0 }, 0.0, 1.0]
}

#[track_caller]
fn assert_close(what: &str, got: &[Number], want: &[Number], tol: Number) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(
            err < tol,
            "{what}[{k}]: got {g:e}, want {w:e}, rel err {err:e} not < {tol:e}"
        );
    }
}

// ---------------------------------------------------------------
// Preconditions -- without these every leg below is vacuous
// ---------------------------------------------------------------

/// The fixture has to actually carry a kink. Asserted once and named
/// so a failure here reads as "the fixture stopped being degenerate"
/// rather than as a defect in a leg.
#[test]
fn the_fixture_carries_a_kink_and_an_untouchable_interior_variable() {
    let s = solved(None, false);
    let x = s.converged().expect("converged").x.clone();
    assert!(x[0].abs() < 1e-3, "k sits on its lower bound, got {}", x[0]);
    assert!(
        (x[1] - W_STAR).abs() < 1e-6,
        "w sits at W_STAR, got {}",
        x[1]
    );

    let weak = weak_set(&s);
    assert!(
        weak.contains(&(0, true)),
        "k's lower bound is the kink and must be weak: {weak:?}"
    );
    assert!(
        !weak.iter().any(|&(r, _)| r == 1),
        "w is {W_STAR} from either bound and must never be weak: {weak:?}"
    );

    // The two sides must genuinely disagree, or the magnitude leg
    // would pass on any linear map.
    assert_close(
        "derivative up",
        &derivative(&s, 1.0),
        &exact_derivative(1.0),
        1e-6,
    );
    assert_close(
        "derivative down",
        &derivative(&s, -1.0),
        &exact_derivative(-1.0),
        1e-6,
    );
}

/// The step is `J·delta + c`, not `J·delta`. `c` is the correction of
/// the base point's own barrier displacement -- order `mu`, the same
/// vector at every `delta` -- which is why the legs compare slopes and
/// not ratios. Pinned here so that if `c` ever stops being negligible,
/// or stops being constant, this test says so rather than a leg
/// failing for a reason its name does not describe.
#[test]
fn the_step_is_affine_in_delta() {
    let s = solved(None, false);
    let j = exact_derivative(1.0);

    let mut constants = Vec::new();
    for &delta in &[1.0e-2, 1.0e-4, 1.0e-7, 1.0e-10] {
        let d = step(s_ref(&s), delta);
        let c: Vec<Number> = d
            .iter()
            .zip(j.iter())
            .map(|(&v, &jj)| v - jj * delta)
            .collect();
        for (i, &ci) in c.iter().enumerate() {
            assert!(
                ci.abs() < 1e-6,
                "the base-point term must stay of order mu: c[{i}] = {ci:e} at delta={delta:e}"
            );
        }
        constants.push(c);
    }
    let first = &constants[0];
    for (n, c) in constants.iter().enumerate().skip(1) {
        assert_close(&format!("base-point term at magnitude {n}"), c, first, 1e-6);
    }
}

/// Borrow helper: keeps the loop above readable without cloning the
/// solver.
fn s_ref(s: &Solver) -> &Solver {
    s
}

// ---------------------------------------------------------------
// Leg 1 -- scaling
// ---------------------------------------------------------------

/// Factors spanning five orders, mixed above and below one, with the
/// kink variable's factor well away from 1: under `x̃ = d ⊙ x` the
/// barrier diagonal carries `d^-2`, so a rule comparing a bare `Sigma`
/// against a fixed band sees this kink as a different kind of bound
/// than the unscaled arm does.
const D_PLAIN: [Number; 3] = [1.0e-2, 5.0, 1.0e3];
const D_FIXED: [Number; 4] = [2.0, 1.0e-2, 5.0, 1.0e3];

#[test]
fn leg_scaling_the_weak_set_is_unmoved_by_the_change_of_variables() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);

    let want = weak_set(&plain);
    assert!(
        !want.is_empty(),
        "precondition: the unscaled arm must find the kink"
    );
    assert_eq!(
        weak_set(&scaled),
        want,
        "membership is a fact about the bound, not about the units the \
         model is written in: Sigma carries d^-2, so a threshold on a \
         bare Sigma moves here"
    );
}

/// Leg 1 for `reduced_activity` (gh#763). The reduced curvature is a
/// natural-units quantity, like `var_sigma`: it is a fact about the
/// model, not about the coordinates the solve ran in. Under
/// `x̃ = d ⊙ x` the factor's own `(K⁻¹)_ii` carries `d²` and `Sigma`
/// carries `d²` as well, so a refinement that forgot either -- or
/// subtracted one frame's `Sigma` from the other frame's reciprocal --
/// moves here and nowhere else.
#[test]
fn leg_scaling_the_reduced_curvature_is_unmoved_by_the_change_of_variables() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);

    for (i, want_status) in [(0usize, WEAKLY_ACTIVE), (1, INACTIVE)] {
        let (sp, rp, qp) = reduced(&plain, i);
        let (ss, rs, qs) = reduced(&scaled, i);
        // `k`'s only Hessian coupling is to the pin `p`, which the
        // equality holds fixed so nothing re-optimizes along it, and
        // `w` is decoupled outright: both reduced curvatures are the
        // model's own 1.0. Checked against that exact value as well,
        // so a shared error cannot pass the pair off as agreement.
        assert!(
            (qp - 1.0).abs() < 1e-6 && (qs - 1.0).abs() < 1e-6,
            "var {i}: the reduced curvature is 1 in both arms, got \
             plain {qp:e} scaled {qs:e}"
        );
        assert!(
            (rp - rs).abs() <= 1e-6 * rp.abs().max(1.0),
            "var {i}: the reduced ratio is unmoved by the change of \
             variables, got plain {rp:e} scaled {rs:e}"
        );
        assert_eq!(
            (sp, ss),
            (want_status, want_status),
            "var {i}: reduced class in both arms"
        );
    }
}

#[test]
fn leg_scaling_the_directional_derivative_is_unmoved_by_the_change_of_variables() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);

    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        // Both arms are checked against the exact derivative as well,
        // so a shared error cannot pass the pair off as agreement.
        let want = derivative(&plain, sign);
        let got = derivative(&scaled, sign);
        assert_close(
            &format!("plain derivative (sign {sign})"),
            &want,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("scaled derivative (sign {sign})"),
            &got,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("derivative parity (sign {sign})"),
            &got,
            &want,
            1e-6,
        );
    }
}

// ---------------------------------------------------------------
// Leg 2 -- perturbation magnitude
// ---------------------------------------------------------------

/// Eight orders. An absolute tolerance anywhere on the path shows up
/// as one end of this range disagreeing with the other.
const DELTAS: [Number; 4] = [1.0e-2, 1.0e-4, 1.0e-7, 1.0e-10];

fn magnitude_sweep(s: &Solver, what: &str) {
    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("{what}: slope over [{lo:e}, {hi:e}]"),
                &slope(s, hi, lo),
                &exact,
                1e-6,
            );
        }
        // and the full span, so a defect that only shows between
        // distant magnitudes is not stepped over
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("{what}: slope over the full span [{lo:e}, {hi:e}]"),
            &slope(s, hi, lo),
            &exact,
            1e-6,
        );
    }
}

#[test]
fn leg_magnitude_the_directional_derivative_does_not_depend_on_the_step_size() {
    magnitude_sweep(&solved(None, false), "plain");
}

#[test]
fn leg_magnitude_holds_under_the_change_of_variables_too() {
    // The legs compose: gh#672 finding 4 was an absolute tolerance on
    // a perturbation-scaled quantity, and a scaled frame moves the
    // scale such a tolerance is implicitly calibrated against.
    magnitude_sweep(&solved(Some(D_PLAIN.to_vec()), false), "scaled");
}

// ---------------------------------------------------------------
// Leg 3 -- a fixed variable ahead of the kink
// ---------------------------------------------------------------

#[test]
fn leg_fixed_the_index_spaces_actually_diverge() {
    // Without this the leg proves nothing: `make_parameter` has to
    // have removed the leading column, so that full-x is one longer
    // than var-x and every var-x row of interest is shifted.
    let plain = solved(None, false);
    let fixed = solved(None, true);

    assert_eq!(plain.n_full_x().expect("n_full_x"), 3);
    assert_eq!(fixed.n_full_x().expect("n_full_x"), 4);
    assert_eq!(
        plain.x_primal_rows(&[0, 1, 2]).expect("plain rows"),
        vec![Some(0), Some(1), Some(2)],
        "with nothing removed the two spaces coincide"
    );
    assert_eq!(
        fixed.x_primal_rows(&[0, 1, 2, 3]).expect("fixed rows"),
        vec![None, Some(0), Some(1), Some(2)],
        "the fixed variable has no row at all, and the kink's full-x \
         index 1 is var-x row 0: reading the one as the other lands on \
         the NEIGHBORING variable"
    );
}

#[test]
fn leg_fixed_the_weak_set_is_unmoved_by_a_fixed_variable_ahead_of_the_kink() {
    let plain = solved(None, false);
    let fixed = solved(None, true);

    let want = weak_set(&plain);
    assert!(
        !want.is_empty(),
        "precondition: the plain arm must find the kink"
    );
    assert_eq!(
        weak_set(&fixed),
        want,
        "the weak set is var-x indexed, so removing a column ahead of \
         the kink must not move it (gh#450, gh#672 finding 1)"
    );
}

/// Leg 3 for `reduced_activity` (gh#763). The accessor takes full-x
/// indices and back-solves against a var-x factor, so it is exactly
/// the shape gh#450 bites: read the user index as a factor row and the
/// kink's answer becomes the interior variable's, plausible and wrong.
/// The two are one class and ten orders of ratio apart here, so the
/// mix-up cannot pass.
#[test]
fn leg_fixed_the_reduced_curvature_is_unmoved_by_a_fixed_variable_ahead_of_the_kink() {
    let plain = solved(None, false);
    let fixed = solved(None, true);

    // full-x 0 in the plain arm is full-x 1 in the fixed one; both are
    // var-x row 0.
    for (p_idx, f_idx, want) in [(0usize, 1usize, WEAKLY_ACTIVE), (1, 2, INACTIVE)] {
        let (sp, rp, qp) = reduced(&plain, p_idx);
        let (sf, rf, qf) = reduced(&fixed, f_idx);
        assert_eq!(
            (sp, sf),
            (want, want),
            "full-x {p_idx}/{f_idx}: removing a column ahead of the kink \
             must not move the class (ratios {rp:e} / {rf:e})"
        );
        assert!(
            (rp - rf).abs() <= 1e-6 * rp.abs().max(1.0) && (qp - qf).abs() < 1e-6,
            "full-x {p_idx}/{f_idx}: ratio {rp:e} vs {rf:e}, curvature \
             {qp:e} vs {qf:e}"
        );
    }
    // The removed column itself has no factor row to back-solve
    // against, and says so instead of answering about its neighbour.
    let (st, ratio, q) = reduced(&fixed, 0);
    assert_eq!(st, FIXED, "the make_parameter-removed variable");
    assert!(
        ratio.is_nan() && q.is_nan(),
        "a removed column carries no curvature: ratio {ratio:e}, q {q:e}"
    );
}

#[test]
fn leg_fixed_the_directional_derivative_is_unmoved_by_a_fixed_variable_ahead_of_the_kink() {
    let plain = solved(None, false);
    let fixed = solved(None, true);

    for sign in [1.0, -1.0] {
        let exact = exact_derivative(sign);
        let want = derivative(&plain, sign);
        let got = derivative(&fixed, sign);
        assert_close(
            &format!("plain derivative (sign {sign})"),
            &want,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("fixed derivative (sign {sign})"),
            &got,
            &exact,
            1e-6,
        );
        assert_close(
            &format!("derivative parity (sign {sign})"),
            &got,
            &want,
            1e-6,
        );
    }
}

// ---------------------------------------------------------------
// The corners -- where a defect surviving each leg alone shows up
// ---------------------------------------------------------------

#[test]
fn the_legs_compose_at_the_fixed_and_scaled_corner() {
    let plain = solved(None, false);
    let both = solved(Some(D_FIXED.to_vec()), true);

    assert_eq!(
        weak_set(&both),
        weak_set(&plain),
        "weak set at the fixed-and-scaled corner"
    );
    for sign in [1.0, -1.0] {
        assert_close(
            &format!("fixed+scaled derivative (sign {sign})"),
            &derivative(&both, sign),
            &exact_derivative(sign),
            1e-6,
        );
    }
    magnitude_sweep(&both, "fixed+scaled");

    // The reduced normalizer at the same corner: full-x 1 in the
    // fixed+scaled arm is full-x 0 in the plain one, and neither the
    // removed column nor the change of variables may move the answer.
    for (p_idx, b_idx, want) in [(0usize, 1usize, WEAKLY_ACTIVE), (1, 2, INACTIVE)] {
        let (sp, rp, qp) = reduced(&plain, p_idx);
        let (sb, rb, qb) = reduced(&both, b_idx);
        assert_eq!((sp, sb), (want, want), "reduced class at the corner");
        assert!(
            (rp - rb).abs() <= 1e-6 * rp.abs().max(1.0) && (qp - qb).abs() < 1e-6,
            "full-x {p_idx}/{b_idx} at the corner: ratio {rp:e} vs {rb:e}, \
             curvature {qp:e} vs {qb:e}"
        );
    }
}

// ---------------------------------------------------------------
// Leg 2, second fixture: the magnitude sweep over an AMBIGUOUS kink
// ---------------------------------------------------------------
//
// The engagement rule that decides a weak row branches on the
// classifier's verdict, so the magnitude leg has to run in BOTH
// classes. The fixture above is certified `WEAKLY_ACTIVE`; a rule
// that treats the certified class exactly and the ambiguous class by
// a length is invisible to it.
//
// A genuine kink lands in the ambiguous class whenever it is coupled.
// `classify` divides `sigma` by the Hessian's DIAGONAL `H_ii`, but the
// multiplier at a kink is generated by the curvature *reduced* along
// that coordinate. Eliminating a free partner `y` from
// `[[h, c], [c, m]]` leaves
//
//     reduced = h - c^2/m,   sigma = reduced,   ratio = reduced / h
//
// so the ratio is 1 only when the coordinate is DECOUPLED. Drive
// `c^2/(h*m)` toward 1 and the ratio falls below the band edge at
// `1e-1`: a genuine kink, classified `AMBIGUOUS`. On a collocation
// model -- the kind that motivated the degeneracy work -- strong
// coupling between neighbouring coordinates is the normal case, not a
// corner.

/// Reduced curvature along `k` in [`CoupledKinkTnlp`]. Chosen so the
/// classifier's ratio is `1e-2`, a decade inside the ambiguous class.
const RHO: Number = 1.0e-2;
/// Diagonal curvature of both coordinates in [`CoupledKinkTnlp`].
const H_DIAG: Number = 1.0;
const M_DIAG: Number = 1.0;

/// Cross term giving reduced curvature `rho`: `h - c^2/m = rho`.
fn cross(rho: Number) -> Number {
    (M_DIAG * (H_DIAG - rho)).sqrt()
}

/// ```text
/// min  0.5*h*k^2 + c*k*y + 0.5*m*y^2 - A*p*k
/// s.t. p = 0,  0 <= k <= 10,  y free
/// ```
///
/// At `p = 0` the reduced gradient at `k = 0` is zero and the
/// multiplier vanishes with `mu`: a kink by construction, exactly as
/// in [`KinkTnlp`], but coupled. Moving the pin up lets `k` follow at
/// `A/RHO`; moving it down would drive `k` through its bound, so `k`
/// holds and the derivative is `0` -- at EVERY step size.
///
/// `rho` is the reduced curvature along `k`, i.e. how strongly the
/// coordinate is coupled: `rho = H_DIAG` is decoupled, and driving it
/// toward `0` drives `c^2/(h*m)` toward `1`.
struct CoupledKinkTnlp {
    rho: Number,
    /// `g_scaling[0]` under `nlp_scaling_method=user-scaling`; `None`
    /// declines scaling entirely, as every other use of this fixture
    /// does.
    g_scale: Option<Number>,
}

impl TNLP for CoupledKinkTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 3,
            m: 1,
            nnz_jac_g: 1,
            nnz_h_lag: 4,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        let Some(dg) = self.g_scale else {
            return false;
        };
        *req.obj_scaling = 1.0;
        *req.use_x_scaling = false;
        *req.use_g_scaling = true;
        req.g_scaling[0] = dg;
        true
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        b.x_l[0] = 0.0;
        b.x_u[0] = 10.0;
        b.x_l[1] = -1.0e19;
        b.x_u[1] = 1.0e19;
        b.x_l[2] = -1.0e19;
        b.x_u[2] = 1.0e19;
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        sp.x[0] = 0.3;
        sp.x[1] = 0.0;
        sp.x[2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let (k, y, p) = (x[0], x[1], x[2]);
        Some(0.5 * H_DIAG * k * k + cross(self.rho) * k * y + 0.5 * M_DIAG * y * y - A * p * k)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let (k, y, p) = (x[0], x[1], x[2]);
        g[0] = H_DIAG * k + cross(self.rho) * y - A * p;
        g[1] = cross(self.rho) * k + M_DIAG * y;
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
                // lower triangle: (k,k), (y,k), (y,y), (p,k)
                irow.copy_from_slice(&[0 as Index, 1, 1, 2]);
                jcol.copy_from_slice(&[0 as Index, 0, 1, 0]);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor * H_DIAG;
                values[1] = obj_factor * cross(self.rho);
                values[2] = obj_factor * M_DIAG;
                values[3] = -obj_factor * A;
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

fn solved_coupled() -> Solver {
    solved_coupled_at(RHO)
}

/// [`solved_coupled`] at a chosen coupling strength.
fn solved_coupled_at(rho: Number) -> Solver {
    solved_coupled_scaled(rho, None)
}

/// [`solved_coupled_at`] with an optional `user-scaling` row factor on
/// the single constraint row.
fn solved_coupled_scaled(rho: Number, g_scale: Option<Number>) -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-8, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    if g_scale.is_some() {
        app.options_mut()
            .set_string_value("nlp_scaling_method", "user-scaling", true, false)
            .unwrap();
    }
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CoupledKinkTnlp { rho, g_scale }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "coupled base solve failed: {status:?}",
    );
    solver
}

/// `k` follows the pin at `A/RHO` on the leaving side and holds at `0`
/// on the other; `y` follows `k` at `-c/m`; the pin moves at `1`.
fn exact_coupled(sign: Number) -> [Number; 3] {
    if sign > 0.0 {
        let dk = A / RHO;
        [dk, -(cross(RHO) / M_DIAG) * dk, 1.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// The precondition the leg rests on, asserted rather than assumed:
/// the coupled fixture's kink really is a kink, really is weak, and
/// really is in the AMBIGUOUS class rather than the certified one. If
/// a classifier change ever moves it, this fails rather than letting
/// the leg pass vacuously against the wrong branch.
///
/// AMBIGUOUS here is `classify_activity`'s verdict, i.e. the DIAGONAL
/// normalizer's, and it is a mislabeling of a genuine kink (gh#763) --
/// pinned, not endorsed. `reduced_activity` certifies the same kink;
/// see [`the_reduced_normalizer_certifies_a_coupled_kink_at_every_coupling`].
#[test]
fn the_coupled_fixture_carries_an_ambiguous_kink() {
    let s = solved_coupled();
    let report = s.classify_activity().expect("activity report");

    assert_eq!(
        report.var_status[0], AMBIGUOUS,
        "the coupled kink must land in the AMBIGUOUS class (got {}, WEAKLY_ACTIVE is {}); \
         the ratio is {:e}",
        report.var_status[0], WEAKLY_ACTIVE, report.var_ratio[0]
    );
    // The ratio is `reduced/diagonal` up to the barrier's own finite
    // `mu`; the point is the decade it sits in, not the last digit.
    assert!(
        (report.var_ratio[0] - RHO).abs() < 1e-3 * RHO,
        "the ratio is reduced/diagonal = {RHO:e}, got {:e}",
        report.var_ratio[0]
    );
    assert!(
        weak_set(&s).contains(&(0, true)),
        "the coupled kink's lower bound must still be weak: {:?}",
        weak_set(&s)
    );
    // The two sides must genuinely disagree, or the leg would pass on
    // any linear map.
    assert_close(
        "coupled derivative up",
        &slope(&s, 1.0e-3, 1.0e-6),
        &exact_coupled(1.0),
        1e-6,
    );
}

/// The headline of gh#763, as the issue's own table: four solves that
/// are the SAME kink -- same one-sided derivatives, `A/reduced`
/// leaving and `0` holding -- differing only in how strongly the kink
/// coordinate is coupled to its free partner.
///
/// `classify_activity` divides `Sigma` by the Hessian diagonal, so its
/// ratio is `reduced/diagonal` and tracks the coupling: the bottom two
/// rows fall out of the `[1e-1, 1e1]` band and read AMBIGUOUS. No
/// tolerance recovers them -- the ratio is `mu`-independent, so a
/// tighter solve reports the same thing. `reduced_activity` divides by
/// the curvature that generates the multiplier, so its ratio is `1` at
/// every coupling and all four certify.
///
/// This is the test the guard above anticipates: whichever way the
/// default normalizer is decided, one of the two halves has to be
/// updated deliberately.
#[test]
fn the_reduced_normalizer_certifies_a_coupled_kink_at_every_coupling() {
    for rho in [1.0, 1.0e-1, 1.0e-2, 1.0e-3] {
        let s = solved_coupled_at(rho);
        let report = s.classify_activity().expect("activity report");
        let (status, ratio, q) = reduced(&s, 0);

        // What the diagonal normalizer says: the ratio IS the coupling.
        assert!(
            (report.var_ratio[0] - rho).abs() < 1e-3 * rho,
            "rho {rho:e}: the diagonal ratio is reduced/diagonal, got {:e}",
            report.var_ratio[0]
        );
        let diagonal_class = if rho >= 1.0e-1 {
            WEAKLY_ACTIVE
        } else {
            AMBIGUOUS
        };
        assert_eq!(
            report.var_status[0], diagonal_class,
            "rho {rho:e}: the diagonal class follows the band edge at 1e-1, \
             not the geometry (ratio {:e})",
            report.var_ratio[0]
        );

        // What the reduced normalizer says: the same kink, every time.
        assert!(
            (q - rho).abs() < 1e-3 * rho,
            "rho {rho:e}: the reduced curvature is rho itself, got {q:e}"
        );
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "rho {rho:e}: Sigma at a kink IS the reduced curvature, so the \
             reduced ratio is 1, got {ratio:e}"
        );
        assert_eq!(
            status, WEAKLY_ACTIVE,
            "rho {rho:e}: a kink is a kink at every coupling (reduced ratio \
             {ratio:e}, diagonal ratio {:e})",
            report.var_ratio[0]
        );

        // It really is the same kink at every row: the one-sided
        // derivatives are what they are for a kink, not for a bound
        // that merely looks like one.
        let dk = A / rho;
        let exact_up = [dk, -(cross(rho) / M_DIAG) * dk, 1.0];
        assert_close(
            &format!("rho {rho:e}: derivative up"),
            &slope(&s, 1.0e-3, 1.0e-6),
            &exact_up,
            1e-6 * dk.max(1.0),
        );
        assert_close(
            &format!("rho {rho:e}: derivative down"),
            &slope(&s, -1.0e-3, -1.0e-6),
            &[0.0, 0.0, 1.0],
            1e-6,
        );
    }
}

/// Leg 2 over the ambiguous class. The holding side's derivative is
/// `0` at every step size; a rule that engages the row only once the
/// step exceeds a fixed base-point length reads the LEAVING side's
/// answer below that length instead, which is a first-order error and
/// a step straight through the bound.
///
/// This is gh#672 finding 4's shape: a length compared against a
/// quantity that scales with the perturbation. The length being
/// *measured* rather than *chosen* does not change that -- it is still
/// fixed while the step shrinks.
#[test]
fn leg_magnitude_an_ambiguous_kink_decides_the_same_way_at_every_step_size() {
    let s = solved_coupled();
    for sign in [1.0, -1.0] {
        let exact = exact_coupled(sign);
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("ambiguous kink: slope over [{lo:e}, {hi:e}]"),
                &slope(&s, hi, lo),
                &exact,
                1e-6,
            );
        }
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("ambiguous kink: slope over the full span [{lo:e}, {hi:e}]"),
            &slope(&s, hi, lo),
            &exact,
            1e-6,
        );
    }
}

/// Leg 1, row-scaling arm (gh#763 follow-up). The change of variables
/// is not the only scaling axis a `user-scaling` solve moves: the
/// constraint rows carry their own factors, and every fixture in this
/// file pins that axis at the identity — [`KinkTnlp`] sets
/// `use_g_scaling = false` outright, and the coupled fixture declined
/// scaling entirely until this test.
///
/// The row scale is safe by construction: the natural-units
/// conjugation carries no `dg` into the `x` block, so `Sigma`, the
/// reduced curvature and the classification are all unmoved by it.
/// That is a property worth *pinning* rather than re-deriving —
/// gh#763's own defect was an untested scaling axis (`obj_scaling`),
/// measured safe by the same kind of argument and wrong.
///
/// What "unmoved" means here, measured rather than assumed: the
/// statuses and curvature signs are bit-identical across three decades
/// of `dg`, and the magnitudes agree to 7–27 ULP (relative ~5e-15) —
/// the solver takes a slightly different path in scaled space, so this
/// is solver precision, not bit-equality. The tolerance below is three
/// decades looser than that and still eleven decades tighter than any
/// `dg` leak could hide in: a factor of `dg` or `dg²` reaching the `x`
/// block would move these by `1e3` or `1e6`.
#[test]
fn leg_scaling_the_reduced_curvature_is_unmoved_by_a_row_scaling() {
    const DG: Number = 1.0e3;
    /// Three decades above the 5e-15 actually observed, eleven below a
    /// one-factor-of-`dg` leak.
    const RTOL: Number = 1.0e-12;

    let base = solved_coupled_scaled(RHO, Some(1.0));
    let scaled = solved_coupled_scaled(RHO, Some(DG));

    let vars = [0usize, 1, 2];
    let rb = base.reduced_activity(&vars).expect("reduced activity");
    let rs = scaled.reduced_activity(&vars).expect("reduced activity");
    let pb = base.classify_activity().expect("activity report");
    let ps = scaled.classify_activity().expect("activity report");

    // the kink really is the coupled one this file is about, so the
    // test cannot pass by classifying nothing
    assert_eq!(
        rb.status[0], WEAKLY_ACTIVE,
        "fixture drifted: k should certify as a kink on the reduced normalizer",
    );
    assert_eq!(
        pb.var_status[0], AMBIGUOUS,
        "fixture drifted: the report should still read the coupled kink as ambiguous",
    );

    for (k, &i) in vars.iter().enumerate() {
        // classification and curvature SIGN are exact
        assert_eq!(
            rb.status[k], rs.status[k],
            "x{i}: reduced status moved under a row scaling of {DG:e}",
        );
        assert_eq!(
            rb.q_sign[k], rs.q_sign[k],
            "x{i}: reduced curvature sign moved under a row scaling of {DG:e}",
        );
        assert_eq!(
            pb.var_status[i], ps.var_status[i],
            "x{i}: report status moved under a row scaling of {DG:e}",
        );

        // magnitudes to solver precision
        assert_close_scaled(
            &format!("x{i} reduced ratio"),
            rb.ratio[k],
            rs.ratio[k],
            RTOL,
        );
        assert_close_scaled(
            &format!("x{i} reduced curvature"),
            rb.q_reduced[k],
            rs.q_reduced[k],
            RTOL,
        );
        assert_close_scaled(&format!("x{i} sigma"), rb.sigma[k], rs.sigma[k], RTOL);
        assert_close_scaled(
            &format!("x{i} report ratio"),
            pb.var_ratio[i],
            ps.var_ratio[i],
            RTOL,
        );
    }
}

/// Relative comparison that treats the non-finite entries this fixture
/// genuinely produces as values to match rather than as failures: an
/// unbounded variable's ratio is `NaN` and the equality-pinned
/// coordinate's reduced curvature is `+inf`, and a leak that moved
/// either INTO or OUT OF those states must fail.
fn assert_close_scaled(what: &str, base: Number, scaled: Number, rtol: Number) {
    assert_eq!(
        base.is_nan(),
        scaled.is_nan(),
        "{what}: NaN-ness moved with the row scaling ({base:e} vs {scaled:e})",
    );
    if base.is_nan() {
        return;
    }
    assert_eq!(
        base.is_infinite(),
        scaled.is_infinite(),
        "{what}: infiniteness moved with the row scaling ({base:e} vs {scaled:e})",
    );
    if base.is_infinite() {
        assert_eq!(
            base.signum(),
            scaled.signum(),
            "{what}: sign of the infinity moved ({base:e} vs {scaled:e})",
        );
        return;
    }
    let err = (base - scaled).abs() / base.abs().max(1.0);
    assert!(
        err <= rtol,
        "{what}: moved with the row scaling -- {base:.17e} vs {scaled:.17e} (rel {err:e} > {rtol:e})",
    );
}

// ---------------------------------------------------------------
// The all-released step -- the same three rows
// ---------------------------------------------------------------

/// [`slope`] over `parametric_step_release_all`, cancelling the
/// affine constant the way [`derivative`] does.
fn released_slope(s: &Solver, d_hi: Number, d_lo: Number) -> Vec<Number> {
    let step_at = |d: Number| -> Vec<Number> {
        let (v, released) = s
            .parametric_step_release_all(&[0], &[d])
            .unwrap_or_else(|e| panic!("released step at delta={d:e}: {e:?}"));
        assert!(released > 0, "the fixture's kink must be in the weak set");
        v
    };
    let hi = step_at(d_hi);
    let lo = step_at(d_lo);
    hi.iter()
        .zip(lo.iter())
        .map(|(&a, &b)| (a - b) / (d_hi - d_lo))
        .collect()
}

/// The all-released step is one linear map with no decision in it,
/// so its derivative is the releasing side's exact value on BOTH
/// signs, and it gets the same rows the directional step has: unmoved
/// by the change of variables, by the perturbation magnitude, and by
/// a fixed variable ahead of the kink.
#[test]
fn leg_release_all_is_the_releasing_sides_map_in_every_frame() {
    let plain = solved(None, false);
    let scaled = solved(Some(D_PLAIN.to_vec()), false);
    let fixed = solved(None, true);
    let exact = exact_derivative(1.0);
    for sign in [1.0, -1.0] {
        for (s, what) in [(&plain, "plain"), (&scaled, "scaled"), (&fixed, "fixed")] {
            assert_close(
                &format!("released {what} (sign {sign})"),
                &released_slope(s, sign * 1.0e-3, sign * 1.0e-6),
                &exact,
                1e-6,
            );
        }
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("released slope over [{lo:e}, {hi:e}]"),
                &released_slope(&plain, hi, lo),
                &exact,
                1e-6,
            );
        }
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("released slope over the full span [{lo:e}, {hi:e}]"),
            &released_slope(&plain, hi, lo),
            &exact,
            1e-6,
        );
    }
}

// ===================================================================
// Fixture 3 -- a strictly active INEQUALITY row, for
// `d_multiplier_rows` (pounce#910)
// ===================================================================
//
// The two fixtures above have `m = 1`, and that one row is the pin
// equality: neither of them has an inequality row at all, so neither
// says anything about the `y_d` block. `d_multiplier_rows` maps a
// user-space row to the KKT row of its inequality multiplier, and it
// has a dimension in all three legs:
//
// 1. it is read through the frame conversions -- the multiplier it
//    points at is scaled by the row's own factor and by the
//    objective's, and the pin's delta is scaled too, so three factors
//    meet in one number (the same shape as
//    `leg_scaling_the_reduced_row_curvature_is_unmoved_by_a_row_scaling`);
// 2. the step it is read out of is affine in `delta` like every other
//    block, so the invariant is a slope;
// 3. it is an index map, and the space it maps out of is not the space
//    it maps into. `full_to_c` and `full_to_d` are complements, so the
//    d-block position of a row is its g index MINUS the equalities
//    ahead of it -- which is exactly the shape of the full-x/var-x
//    hazard leg 3 exists for, one block over. The fixture puts the pin
//    equality FIRST and an inactive inequality SECOND so that the
//    active row's g index (2), its d-block position (1) and its
//    c-block position (which does not exist) are three different
//    things.
//
// ```text
// min  0.5 x^2 + 0.5 y^2 - x - y     [+ 0.5 (xf - 0.7)^2]
// s.t. g0:  p == 0                   (the pin)
//      g1:  x - y <= SLACK_UB        (never active)
//      g2:  x + y - p <= P_CAP       (strictly active)
// ```
//
// With `p = delta` the cap reads `x + y <= P_CAP + delta`, and the
// unconstrained optimum `x = y = 1` violates it for `P_CAP < 2`. On
// the cap, `x = y = (P_CAP + delta)/2` and the multiplier is
//
//     lambda = 1 - (P_CAP + delta)/2,   d lambda / d delta = -1/2
//
// at `P_CAP = 1`, i.e. `lambda = 1/2`: bounded away from zero on both
// sides, which is what makes the derivative two-sided and the row
// answerable at all. `SLACK_UB` keeps `g1` slack by a wide margin so
// its own multiplier stays at the barrier floor.
//
// Mutation table -- each row was introduced, the suite run, and the
// mutation reverted:
//
// | mutation                                            | result |
// |-----------------------------------------------------|--------|
// | `full_g_to_d_block` returns `Some(full_idx)` instead | 18 passed, 6 failed |
// | `d_multiplier_rows`' `y_d_offset` drops `dims[2]`    | 18 passed, 6 failed -- the same six |
//
// The six are every test in this section except
// `the_scaled_arm_of_the_multiplier_leg_is_actually_scaled`, which is
// the liveness witness and stays green under both.
//
// It stays green by construction: it reads `nlp_scaling()` and
// `variable_scaling()`, which do not go through `full_to_d` at all, so
// it can still testify that the scaled arm really was scaled while the
// map under test is broken. A leg asserting "this number did not move"
// passes identically when the mechanism never engaged, which is what
// that separation is for.
//
// What this fixture is NOT evidence about, so that the next reader does
// not over-read a green run:
//
// * **the gate.** Nothing here is weakly active, by design -- the cap's
//   multiplier is `1/2`. The three-regime refusal lives in
//   `python/pounce/sensitivity/_session.py`'s `mult_entry` and is
//   covered by `pyomo-pounce/tests/test_sens.py`, including a coupled
//   row kink that `classify_activity` misreads as INACTIVE. Per the
//   branch rule in CLAUDE.md, a leg is evidence only about the branch
//   its fixture reaches, and this one reaches the strictly complementary
//   branch only.
// * **more than one active inequality.** `m = 3` with exactly one
//   active cap, so no leg here compares two live multiplier
//   derivatives against each other. The ordering half of that gap is
//   closed by hand -- the precondition test pins both d rows to their
//   exact flat positions, so a permuted `d_map` fails even though only
//   one of the two rows is active -- but a defect that needs two
//   *simultaneously active* rows to appear would not show.
// * **magnitude.** Three variables and three rows; nothing here says
//   anything about the cost of the extra map at benchmark scale (it is
//   an `O(1)` vector read, but that is an argument, not a measurement).

/// The active cap's right-hand side. Below 2 so the cap binds, and far
/// enough below that the multiplier `1 - P_CAP/2` is nowhere near the
/// barrier floor: this fixture is the STRICTLY complementary branch,
/// deliberately, since the kink branch is what the two fixtures above
/// already carry.
const P_CAP: Number = 1.0;
/// Upper bound on the decoy inequality `x - y`. The solution has
/// `x = y`, so the row sits this far inside its bound.
const SLACK_UB: Number = 5.0;
/// `d lambda / d delta` for the active cap, exactly.
const EXACT_DLAMBDA: Number = -0.5;

/// User-space g index of each row of [`IneqMultTnlp`]. Named because
/// the whole point of the fixture is that these are not the block
/// positions.
const G_PIN: Index = 0;
const G_SLACK: Index = 1;
const G_CAP: Index = 2;

struct IneqMultTnlp {
    /// Per-variable factors under `user-scaling`, or `None` to decline
    /// scaling. Length `3 + off`.
    x_scaling: Option<Vec<Number>>,
    /// Per-row factors, same convention. Length 3.
    g_scaling: Option<Vec<Number>>,
    leading_fixed: bool,
}

impl IneqMultTnlp {
    fn off(&self) -> usize {
        usize::from(self.leading_fixed)
    }
}

impl TNLP for IneqMultTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: (3 + self.off()) as Index,
            m: 3,
            nnz_jac_g: 6,
            nnz_h_lag: (2 + self.off()) as Index,
            index_style: IndexStyle::C,
        })
    }

    fn get_scaling_parameters(&mut self, req: ScalingRequest<'_>) -> bool {
        if self.x_scaling.is_none() && self.g_scaling.is_none() {
            return false;
        }
        *req.obj_scaling = 1.0;
        if let Some(d) = self.x_scaling.as_ref() {
            *req.use_x_scaling = true;
            req.x_scaling.copy_from_slice(d);
        } else {
            *req.use_x_scaling = false;
        }
        if let Some(d) = self.g_scaling.as_ref() {
            *req.use_g_scaling = true;
            req.g_scaling.copy_from_slice(d);
        } else {
            *req.use_g_scaling = false;
        }
        true
    }

    fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            b.x_l[0] = FIXED_AT;
            b.x_u[0] = FIXED_AT;
        }
        for j in 0..3 {
            b.x_l[o + j] = -1.0e19;
            b.x_u[o + j] = 1.0e19;
        }
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = -1.0e19;
        b.g_u[1] = SLACK_UB;
        b.g_l[2] = -1.0e19;
        b.g_u[2] = P_CAP;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            sp.x[0] = FIXED_AT;
        }
        sp.x[o] = 0.2;
        sp.x[o + 1] = 0.2;
        sp.x[o + 2] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let o = self.off();
        let (a, b) = (x[o], x[o + 1]);
        let mut f = 0.5 * a * a + 0.5 * b * b - a - b;
        if self.leading_fixed {
            f += 0.5 * (x[0] - 0.7) * (x[0] - 0.7);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        if self.leading_fixed {
            g[0] = x[0] - 0.7;
        }
        g[o] = x[o] - 1.0;
        g[o + 1] = x[o + 1] - 1.0;
        g[o + 2] = 0.0;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        g[0] = x[o + 2];
        g[1] = x[o] - x[o + 1];
        g[2] = x[o] + x[o + 1] - x[o + 2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0 as Index, 1, 1, 2, 2, 2]);
                jcol.copy_from_slice(&[o + 2, o, o + 1, o, o + 1, o + 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, 1.0, -1.0, 1.0, 1.0, -1.0]);
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
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (x,x), (y,y) [, (xf,xf)]. Both
                // constraints are linear and `p` carries no curvature,
                // so nothing else is nonzero.
                let mut rs: Vec<Index> = vec![o, o + 1];
                let mut cs: Vec<Index> = vec![o, o + 1];
                if self.leading_fixed {
                    rs.push(0);
                    cs.push(0);
                }
                irow.copy_from_slice(&rs);
                jcol.copy_from_slice(&cs);
            }
            SparsityRequest::Values { values } => {
                values[0] = obj_factor;
                values[1] = obj_factor;
                if self.leading_fixed {
                    values[2] = obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// As [`solved`], for [`IneqMultTnlp`]. Both arms of leg 1 run under
/// `user-scaling`; only whether the TNLP hands factors back differs.
fn solved_ineq(
    x_scaling: Option<Vec<Number>>,
    g_scaling: Option<Vec<Number>>,
    leading_fixed: bool,
) -> Solver {
    let mut app = IpoptApplication::new();
    app.options_mut()
        .set_integer_value("print_level", 0, true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("sb", "yes", true, false)
        .unwrap();
    app.options_mut()
        .set_string_value("nlp_scaling_method", "user-scaling", true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("tol", 1e-10, true, false)
        .unwrap();
    app.options_mut()
        .set_numeric_value("bound_relax_factor", 0.0, true, false)
        .unwrap();
    app.initialize().unwrap();

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(IneqMultTnlp {
        x_scaling,
        g_scaling,
        leading_fixed,
    }));
    let mut solver = Solver::new(app, tnlp);
    let status = solver.solve();
    assert!(
        matches!(
            status,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "inequality-multiplier base solve failed: {status:?}",
    );
    solver
}

/// The KKT row `d_multiplier_rows` gives for user row `g`, or a panic
/// naming the row if it refuses.
fn d_row(s: &Solver, g: Index) -> usize {
    s.d_multiplier_rows(&[g])
        .expect("d_multiplier_rows")
        .into_iter()
        .next()
        .unwrap()
        .unwrap_or_else(|| panic!("row {g} has no inequality multiplier")) as usize
}

/// `d lambda / d delta` for user row `g`, as a slope between two
/// perturbations on the same side -- see the module docs on why this
/// is not `step / delta`.
fn mult_slope(s: &Solver, g: Index, d_hi: Number, d_lo: Number) -> Number {
    let row = d_row(s, g);
    let hi = s
        .parametric_step_full(&[G_PIN], &[d_hi])
        .unwrap_or_else(|e| panic!("full step at {d_hi:e}: {e:?}"));
    let lo = s
        .parametric_step_full(&[G_PIN], &[d_lo])
        .unwrap_or_else(|e| panic!("full step at {d_lo:e}: {e:?}"));
    (hi[row] - lo[row]) / (d_hi - d_lo)
}

// ---------------------------------------------------------------
// Preconditions -- without these the three legs below are vacuous
// ---------------------------------------------------------------

/// The fixture has to carry a STRICTLY active inequality, an INACTIVE
/// one, and an equality ahead of both. A leg that asserts "the row's
/// multiplier derivative is unmoved" passes identically when the row
/// stopped being active, so the branch each leg runs on is asserted
/// here and named, exactly as
/// [`the_fixture_carries_a_kink_and_an_untouchable_interior_variable`]
/// does for fixture 1.
#[test]
fn the_ineq_fixture_carries_one_strictly_active_and_one_inactive_row() {
    let s = solved_ineq(None, None, false);

    // the pin is an equality: in the c block, not the d block
    assert_eq!(
        s.d_multiplier_rows(&[G_PIN]).unwrap()[0],
        None,
        "the pin row must have no inequality multiplier",
    );
    assert!(
        s.g_multiplier_rows(&[G_PIN]).unwrap()[0].is_some(),
        "the pin row must have an equality multiplier",
    );
    // and the two inequalities are the other way round
    for g in [G_SLACK, G_CAP] {
        assert_eq!(
            s.g_multiplier_rows(&[g]).unwrap()[0],
            None,
            "row {g} must have no equality multiplier",
        );
        assert!(
            s.d_multiplier_rows(&[g]).unwrap()[0].is_some(),
            "row {g} must have an inequality multiplier",
        );
    }

    // The two d rows must come back in g order, and adjacent: the
    // fixture has n_x = 3, n_s = 2 and n_c = 1, so the `y_d` block
    // starts at flat row 6 and the slack row (d position 0) precedes
    // the cap row (d position 1). Spelled out rather than derived so a
    // permuted `d_map` -- which every other test in this section would
    // survive, since only one row is ever active -- fails here.
    assert_eq!(
        (d_row(&s, G_SLACK), d_row(&s, G_CAP)),
        (6, 7),
        "the d block must start at n_x + n_s + n_c = 6 and hold the          two inequalities in g order",
    );

    // the classifier -- the same gate `pounce.sensitivity`'s
    // `mult_entry` uses -- has to agree about which is which
    let rep = s
        .reduced_row_activity(&[G_SLACK as usize, G_CAP as usize])
        .unwrap();
    assert_eq!(
        rep.status[0], INACTIVE,
        "g1 must be inactive, got status {}",
        rep.status[0],
    );
    assert_eq!(
        rep.status[1], STRONGLY_ACTIVE,
        "g2 must be strictly active, got status {}",
        rep.status[1],
    );

    // ... and the exact answer must be the one the legs then compare
    // against, at unit scaling and with no fixed variable in front
    let got = mult_slope(&s, G_CAP, 1.0e-3, 1.0e-6);
    assert!(
        (got - EXACT_DLAMBDA).abs() < 1e-6,
        "d lambda/d delta = {got:e}, want {EXACT_DLAMBDA:e}",
    );
}

/// An out-of-range g index is an ERROR from `d_multiplier_rows`, not a
/// `None`.
///
/// With both accessors present, `None` from `g_multiplier_rows` means
/// "ask the other one" -- which is exactly the fall-through
/// `pounce.sensitivity`'s `mult_entry` performs. If the second call
/// also answered `None` for a row that does not exist, a caller could
/// not tell "no multiplier row" from "no such row", and the message it
/// would print names the wrong reason. `g_multiplier_rows` keeps its
/// historical `None` (nothing reads it as a complement), so the pair is
/// asymmetric on purpose and the asymmetry is what closes the case.
///
/// The same shape as `x_primal_rows`, whose comment says it directly:
/// out of range must not masquerade as "removed as fixed".
#[test]
fn an_out_of_range_row_is_an_error_not_an_equality() {
    let s = solved_ineq(None, None, false);
    let past_the_end = 3 as Index; // m = 3, so 0..=2 are the rows
    assert_eq!(
        s.g_multiplier_rows(&[past_the_end]).unwrap()[0],
        None,
        "g_multiplier_rows keeps its historical None for an unknown row",
    );
    let err = s
        .d_multiplier_rows(&[past_the_end])
        .expect_err("d_multiplier_rows must refuse an out-of-range row");
    match err {
        SolverError::BadShape { got, expected, .. } => {
            assert_eq!((got, expected), (3, 3), "wrong BadShape payload: {err:?}");
        }
        other => panic!("wrong error for an out-of-range row: {other:?}"),
    }
    // and a negative index, which reaches the same guard by the other
    // side of the comparison rather than by the `as usize` wrap
    assert!(
        s.d_multiplier_rows(&[-1]).is_err(),
        "a negative row index must refuse too",
    );
}

/// The `y_d` block is affine in `delta` for the same reason the primal
/// block is, and by the same order of constant. The companion of
/// [`the_step_is_affine_in_delta`] for the multiplier row: if the
/// base-point term ever stops being negligible or stops being
/// constant, this says so rather than leg 2 failing under a name that
/// does not describe it.
#[test]
fn the_multiplier_step_is_affine_in_delta() {
    let s = solved_ineq(None, None, false);
    let row = d_row(&s, G_CAP);
    let mut consts = Vec::new();
    for d in [1.0e-3, 1.0e-5, 1.0e-7] {
        let step = s.parametric_step_full(&[G_PIN], &[d]).unwrap();
        consts.push(step[row] - EXACT_DLAMBDA * d);
    }
    let spread = consts.iter().fold(Number::NEG_INFINITY, |a, &b| a.max(b))
        - consts.iter().fold(Number::INFINITY, |a, &b| a.min(b));
    assert!(
        consts.iter().all(|c| c.abs() < 1e-6),
        "base-point term is not negligible: {consts:?}",
    );
    assert!(
        spread < 1e-9,
        "base-point term is not constant across delta: {consts:?}",
    );
}

// ---------------------------------------------------------------
// Leg 1 -- scaling
// ---------------------------------------------------------------

/// Leg 1 for `d_multiplier_rows`. The multiplier this row points at
/// lives in the algorithm's frame: the row's own factor `dg` divides
/// it, the objective factor multiplies it, and the pin's `delta` is
/// scaled by the PIN row's factor -- three factors meeting in one
/// ratio, the same shape as
/// [`leg_scaling_the_reduced_row_curvature_is_unmoved_by_a_row_scaling`].
/// The model's `d lambda / d delta` is a property of the model, so it
/// must come out the same in either frame.
///
/// The factors are deliberately unequal across the three rows and the
/// three columns: a conversion that used the wrong row's factor, or
/// the pin's where the cap's belongs, is invisible when they agree.
#[test]
fn leg_scaling_the_multiplier_derivative_is_unmoved_by_the_change_of_variables() {
    let base = solved_ineq(None, None, false);
    let scaled = solved_ineq(
        Some(vec![4.0, 0.25, 1.0e2]),
        Some(vec![1.0e-1, 8.0, 5.0e1]),
        false,
    );

    for (what, s) in [("base", &base), ("scaled", &scaled)] {
        let got = mult_slope(s, G_CAP, 1.0e-3, 1.0e-6);
        assert!(
            (got - EXACT_DLAMBDA).abs() < 1e-6,
            "{what}: d lambda/d delta = {got:e}, want {EXACT_DLAMBDA:e}",
        );
    }
}

/// The scaled arm has to be genuinely scaled, or the leg above is two
/// runs of the same solve -- and nothing the sensitivity layer returns
/// can show that, because every accessor on `Solver` already reports
/// natural units. That is the property leg 1 asserts, so it cannot
/// also be the evidence that leg 1 is live. Measured: `row_sigma`,
/// documented as RAW, differs between the two arms by 4e-6 relative,
/// which is floating-point divergence and not a factor of 50.
///
/// [`Solver::nlp_scaling`] is the honest witness: it reports the
/// factors the IPM actually applied. It also cross-checks
/// `full_to_d`'s ordering against a source that does not go through
/// it -- the cap's factor has to land in the SECOND `d_scale` slot,
/// the same position [`Solver::d_multiplier_rows`] claims for it.
#[test]
fn the_scaled_arm_of_the_multiplier_leg_is_actually_scaled() {
    let base = solved_ineq(None, None, false);
    let scaled = solved_ineq(
        Some(vec![4.0, 0.25, 1.0e2]),
        Some(vec![1.0e-1, 8.0, 5.0e1]),
        false,
    );

    let (_, cb, db) = base.nlp_scaling().expect("nlp scaling");
    assert!(
        cb.is_none() && db.is_none() && base.variable_scaling().unwrap().is_none(),
        "the base arm must run unscaled, got c={cb:?} d={db:?}",
    );

    let (_, cs, ds) = scaled.nlp_scaling().expect("nlp scaling");
    let cs = cs.expect("scaled arm must carry equality row factors");
    let ds = ds.expect("scaled arm must carry inequality row factors");
    assert_eq!(cs, vec![1.0e-1], "the pin's factor, in the c block");
    assert_eq!(
        ds,
        vec![8.0, 5.0e1],
        "the two inequality factors in d-block order: the slack row \
         first, the CAP second -- the position d_multiplier_rows claims \
         for it, reached without going through full_to_d",
    );
    assert_eq!(
        scaled.variable_scaling().unwrap(),
        Some(vec![4.0, 0.25, 1.0e2]),
        "the column factors must reach the algorithm too",
    );
}

// ---------------------------------------------------------------
// Leg 2 -- perturbation magnitude
// ---------------------------------------------------------------

/// Leg 2 for `d_multiplier_rows`, over eight orders on both sides.
/// The cap is strictly active, so unlike a kink the two sides agree --
/// which is the property that makes the row answerable at all, and is
/// therefore the thing to assert rather than to assume.
#[test]
fn leg_magnitude_the_multiplier_derivative_does_not_depend_on_the_step_size() {
    let s = solved_ineq(None, None, false);
    for sign in [1.0, -1.0] {
        for (hi, lo) in [
            (1.0e-2, 1.0e-3),
            (1.0e-4, 1.0e-5),
            (1.0e-6, 1.0e-7),
            (1.0e-8, 1.0e-9),
        ] {
            let got = mult_slope(&s, G_CAP, sign * hi, sign * lo);
            assert!(
                (got - EXACT_DLAMBDA).abs() < 1e-5,
                "sign {sign:+}, delta {hi:e}..{lo:e}: d lambda/d delta = \
                 {got:e}, want {EXACT_DLAMBDA:e}",
            );
        }
    }
}

// ---------------------------------------------------------------
// Leg 3 -- a fixed variable ahead of the row
// ---------------------------------------------------------------

/// Leg 3 for `d_multiplier_rows`, and the leg the accessor is most
/// exposed to. Its answer is `n_x + n_s + n_c + full_to_d[g]`, so it
/// reads THREE block dimensions and one index map, and a fixed
/// variable removed by `make_parameter` moves `n_x` under it. Getting
/// `n_x` from the wrong place -- the user's variable count rather than
/// the factor's -- returns a neighbouring block's row, which is
/// gh#450's hazard one block over.
///
/// The map itself is checked in the same breath: `full_to_d` is the
/// complement of `full_to_c`, so the cap's d-block position is its g
/// index minus the one equality ahead of it. Asserting the row's
/// numeric VALUE rather than its index is what makes that check bite
/// -- an off-by-one lands in the slack row's multiplier, whose
/// derivative is zero, not in an out-of-range panic.
#[test]
fn leg_fixed_the_multiplier_derivative_is_unmoved_by_a_fixed_variable_ahead() {
    let plain = solved_ineq(None, None, false);
    let fixed = solved_ineq(None, None, true);

    // the index spaces really do diverge, or the leg is two runs of
    // the same solve
    let dims_plain = plain.block_dims().expect("block dims");
    let dims_fixed = fixed.block_dims().expect("block dims");
    assert_eq!(
        dims_plain[0] + 1,
        dims_fixed[0] + 1,
        "sanity: both solves keep three free columns",
    );
    assert_eq!(
        dims_plain[0], dims_fixed[0],
        "make_parameter must remove the fixed column, leaving n_x equal",
    );

    for (what, s) in [("plain", &plain), ("fixed", &fixed)] {
        // the cap's d-block position is 1, not its g index 2
        let row = d_row(s, G_CAP);
        let dims = s.block_dims().expect("block dims");
        assert_eq!(
            row,
            dims[0] + dims[1] + dims[2] + 1,
            "{what}: the cap's row must be the SECOND entry of the y_d \
             block (one equality is ahead of it in g)",
        );
        let got = mult_slope(s, G_CAP, 1.0e-3, 1.0e-6);
        assert!(
            (got - EXACT_DLAMBDA).abs() < 1e-6,
            "{what}: d lambda/d delta = {got:e}, want {EXACT_DLAMBDA:e}",
        );
    }
}

/// The three legs compose: scaled AND with a fixed variable in front.
/// Each leg alone can pass while the two conversions cancel only in
/// isolation; fixture 1 has [`the_legs_compose_at_the_fixed_and_scaled_corner`]
/// for the same reason.
#[test]
fn the_multiplier_legs_compose_at_the_fixed_and_scaled_corner() {
    let s = solved_ineq(
        Some(vec![2.0, 4.0, 0.25, 1.0e2]),
        Some(vec![1.0e-1, 8.0, 5.0e1]),
        true,
    );
    let got = mult_slope(&s, G_CAP, 1.0e-3, 1.0e-6);
    assert!(
        (got - EXACT_DLAMBDA).abs() < 1e-6,
        "corner: d lambda/d delta = {got:e}, want {EXACT_DLAMBDA:e}",
    );
}
