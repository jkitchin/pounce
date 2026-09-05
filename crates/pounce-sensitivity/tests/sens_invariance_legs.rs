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
use pounce_sensitivity::Solver;
use pounce_sensitivity::activity::{AMBIGUOUS, FIXED, INACTIVE, STRONGLY_ACTIVE, WEAKLY_ACTIVE};

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

// ---------------------------------------------------------------
// Third fixture: the shadow price of a cap (gh#910)
// ---------------------------------------------------------------
//
// `d_multiplier_rows` is the `y_d` counterpart of `g_multiplier_rows`,
// and it is a *map*: it says which row of the compound KKT vector
// carries a given inequality's multiplier. Two things can go wrong
// with a map, and this fixture is built so neither can hide.
//
// 1. **It can address the wrong row.** The d-block position of a
//    user constraint is neither its `g` index nor its `g` index minus
//    a constant -- every equality ahead of it shifts it by one -- and
//    the flat row is that position offset by three block widths. So
//    the fixture puts the pin equality at `g0` and an INACTIVE decoy
//    inequality at `g1`, which makes the cap's three coordinates three
//    different numbers: `g` index 2, d-block position 1, and no
//    c-block position at all. A `full_to_c`-shaped confusion, an
//    off-by-one in the scan, and reading `g` as the d position each
//    land somewhere else.
//
// 2. **It can address the right row of a number that is not a
//    derivative.** `y_d` holds an entry for every inequality in all
//    three activity regimes, and only the strictly active one has a
//    two-sided `dlambda/dp`. So `g3` is a genuine row kink, and the
//    fixture asserts that all three regimes are present and
//    distinguishable -- which is what makes the gate in
//    `pounce.sensitivity`'s `mult_entry` testable at all.
//
// The three legs then run on the cap's multiplier row: its position is
// unmoved by a change of variables (leg 1), the derivative it carries
// does not depend on the step size (leg 2), and neither moves when a
// `make_parameter`-removed variable leads the model (leg 3). Leg 3 is
// the one that would look vacuous -- a fixed variable is an `x`-block
// question and this is a `y_d` row -- and it is exactly the leg that
// fails if the offset is assembled from `n_full_x` instead of the
// factor's own `x` width, which is the gh#450 hazard two blocks over.

/// Where the cap sits, and the row's gradient scale. Not 1, so the
/// multiplier carries a factor and an answer of `-1` cannot pass by
/// coincidence.
const CAP_SCALE: Number = 2.0;
const CAP: Number = 2.0;
/// The unconstrained optimum of `u`, comfortably past the cap so the
/// row is held hard against it: `lambda = (U0 - u)/CAP_SCALE = 2`.
const U0: Number = 5.0;
/// The decoy row: `DECOY_SCALE * w <= DECOY_CAP`, with `w` sitting at
/// `W_STAR`, decades of slack away.
const DECOY_SCALE: Number = 3.0;
const DECOY_CAP: Number = 30.0;
/// The kink row `KINK_SCALE * z >= 0`, and the pin's coupling into
/// `z`. At `p = 0` both `z` and the row's multiplier vanish.
const KINK_SCALE: Number = 1.0;
const A_KINK: Number = 0.6;

/// ```text
/// min  0.5 (u - U0)^2 + 0.5 (w - W_STAR)^2 + 0.5 z^2 - A_KINK p z
///                                            [+ 0.5 (xf - 0.7)^2]
/// s.t. g0:                    p == 0       (the pin; the whole c block)
///      g1: DECOY_SCALE * w <= DECOY_CAP    (inactive decoy; d position 0)
///      g2: CAP_SCALE (u - p) <= CAP        (strictly active; d position 1)
///      g3: KINK_SCALE * z  >= 0            (weakly active; d position 2)
/// ```
///
/// Every variable is free, so the ONLY activity in the model is in the
/// rows -- a variable-bound answer cannot stand in for a row one.
///
/// The cap moves with the pin, which is what gives its shadow price a
/// derivative: at the solution `CAP_SCALE (u - p) = CAP`, so
/// `u = CAP/CAP_SCALE + p`, and stationarity in `u` gives
/// `lambda_2 = (U0 - u)/CAP_SCALE`. Hence
/// `dlambda_2/dp = -1/CAP_SCALE`, exactly, on both sides.
///
/// `leading_fixed` prepends `xf` exactly as [`KinkTnlp`] does, so
/// full-x and var-x diverge in front of everything.
struct CapTnlp {
    x_scaling: Option<Vec<Number>>,
    leading_fixed: bool,
}

impl CapTnlp {
    fn off(&self) -> usize {
        usize::from(self.leading_fixed)
    }
}

impl TNLP for CapTnlp {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: (4 + self.off()) as Index,
            m: 4,
            nnz_jac_g: 5,
            nnz_h_lag: (4 + self.off()) as Index,
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
        for i in o..o + 4 {
            b.x_l[i] = -1.0e19;
            b.x_u[i] = 1.0e19;
        }
        b.g_l[0] = 0.0;
        b.g_u[0] = 0.0;
        b.g_l[1] = -1.0e19;
        b.g_u[1] = DECOY_CAP;
        b.g_l[2] = -1.0e19;
        b.g_u[2] = CAP;
        b.g_l[3] = 0.0;
        b.g_u[3] = 1.0e19;
        true
    }

    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        let o = self.off();
        if self.leading_fixed {
            sp.x[0] = FIXED_AT;
        }
        sp.x[o] = 0.5;
        sp.x[o + 1] = 0.5;
        sp.x[o + 2] = 0.5;
        sp.x[o + 3] = 0.0;
        true
    }

    fn eval_f(&mut self, x: &[Number], _new_x: bool) -> Option<Number> {
        let o = self.off();
        let (u, w, z, p) = (x[o], x[o + 1], x[o + 2], x[o + 3]);
        let mut f = 0.5 * (u - U0) * (u - U0) + 0.5 * (w - W_STAR) * (w - W_STAR) + 0.5 * z * z
            - A_KINK * p * z;
        if self.leading_fixed {
            f += 0.5 * (x[0] - 0.7) * (x[0] - 0.7);
        }
        Some(f)
    }

    fn eval_grad_f(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        let (u, w, z, p) = (x[o], x[o + 1], x[o + 2], x[o + 3]);
        if self.leading_fixed {
            g[0] = x[0] - 0.7;
        }
        g[o] = u - U0;
        g[o + 1] = w - W_STAR;
        g[o + 2] = z - A_KINK * p;
        g[o + 3] = -A_KINK * z;
        true
    }

    fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
        let o = self.off();
        g[0] = x[o + 3];
        g[1] = DECOY_SCALE * x[o + 1];
        g[2] = CAP_SCALE * (x[o] - x[o + 3]);
        g[3] = KINK_SCALE * x[o + 2];
        true
    }

    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _nx: bool, mode: SparsityRequest<'_>) -> bool {
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                irow.copy_from_slice(&[0, 1, 2, 2, 3]);
                jcol.copy_from_slice(&[o + 3, o + 1, o, o + 3, o + 2]);
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, DECOY_SCALE, CAP_SCALE, -CAP_SCALE, KINK_SCALE]);
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
        // every row is linear, so the Lagrangian Hessian is the
        // objective's alone and `lambda` never enters
        let o = self.off() as Index;
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                // lower triangle: (u,u), (w,w), (z,z), (p,z) [, (xf,xf)]
                let mut rs: Vec<Index> = vec![o, o + 1, o + 2, o + 3];
                let mut cs: Vec<Index> = vec![o, o + 1, o + 2, o + 2];
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
                values[2] = obj_factor;
                values[3] = -obj_factor * A_KINK;
                if self.leading_fixed {
                    values[4] = obj_factor;
                }
            }
        }
        true
    }

    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {}
}

/// User-space `g` indices of the four rows, by role.
const G_PIN: Index = 0;
const G_DECOY: Index = 1;
const G_CAP: Index = 2;
const G_KINK: Index = 3;

/// `d(lambda_cap)/dp`, exact and two-sided: see [`CapTnlp`].
const CAP_MULT_SLOPE: Number = -1.0 / CAP_SCALE;

/// Same option set the other two fixtures solve under, so the only
/// difference between the arms of a leg is the fixture's own knob.
fn solved_cap(x_scaling: Option<Vec<Number>>, leading_fixed: bool) -> Solver {
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

    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(CapTnlp {
        x_scaling,
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
        "cap fixture base solve failed: {status:?}",
    );
    solver
}

/// The flat KKT row carrying `g`'s multiplier, through whichever block
/// claims it. Panics if neither does, which the c/d split forbids.
fn mult_row(s: &Solver, g: Index) -> usize {
    let c = s.g_multiplier_rows(&[g]).expect("y_c rows")[0];
    let d = s.d_multiplier_rows(&[g]).expect("y_d rows")[0];
    assert!(
        c.is_none() || d.is_none(),
        "g{g} claimed by both multiplier blocks: y_c {c:?}, y_d {d:?}"
    );
    c.or(d)
        .unwrap_or_else(|| panic!("g{g} claimed by neither multiplier block")) as usize
}

/// `reduced_row_activity`'s verdict for one user row.
fn row_class(s: &Solver, g: Index) -> i8 {
    s.reduced_row_activity(&[g as usize])
        .expect("reduced row")
        .status[0]
}

/// Slope of a flat KKT row of `parametric_step_full` in `delta`,
/// between two perturbations on the same side. The step is affine in
/// `delta` (see [`the_step_is_affine_in_delta`]), so a slope is the
/// derivative and `row / delta` would not be.
fn full_slope(s: &Solver, row: usize, d_hi: Number, d_lo: Number) -> Number {
    let at = |d: Number| -> Number {
        s.parametric_step_full(&[G_PIN], &[d])
            .unwrap_or_else(|e| panic!("full step at delta={d:e}: {e:?}"))[row]
    };
    (at(d_hi) - at(d_lo)) / (d_hi - d_lo)
}

// ---------------- preconditions ----------------

/// Without this every leg below is vacuous: the fixture has to put its
/// three inequalities in three different regimes, and it has to make
/// the cap's `g` index, its d-block position and its flat KKT row
/// three genuinely different numbers.
#[test]
fn the_cap_fixture_carries_all_three_inequality_regimes_at_shifted_indices() {
    let s = solved_cap(None, false);
    let x = s.converged().expect("converged").x.clone();
    assert!(
        (x[0] - CAP / CAP_SCALE).abs() < 1e-6,
        "u is held at the cap, got {}",
        x[0]
    );
    assert!(
        (x[1] - W_STAR).abs() < 1e-6,
        "w sits at W_STAR, decades inside the decoy, got {}",
        x[1]
    );
    assert!(x[2].abs() < 1e-3, "z sits at its kink row, got {}", x[2]);

    assert_eq!(row_class(&s, G_CAP), STRONGLY_ACTIVE, "the cap");
    assert_eq!(row_class(&s, G_DECOY), INACTIVE, "the decoy");
    assert_eq!(row_class(&s, G_KINK), WEAKLY_ACTIVE, "the kink row");

    // the pin is the whole c block; the three inequalities are the
    // whole d block, in g order
    assert_eq!(
        s.g_multiplier_rows(&[G_PIN, G_DECOY, G_CAP, G_KINK])
            .expect("y_c rows")
            .iter()
            .map(|r| r.is_some())
            .collect::<Vec<_>>(),
        vec![true, false, false, false],
        "only the pin lives in y_c",
    );
    let d_rows = s
        .d_multiplier_rows(&[G_PIN, G_DECOY, G_CAP, G_KINK])
        .expect("y_d rows");
    assert!(d_rows[0].is_none(), "the pin is an equality, not in y_d");
    let dims = s.block_dims().expect("block dims");
    let y_d_offset = (dims[0] + dims[1] + dims[2]) as Index;
    assert_eq!(
        d_rows[1..].to_vec(),
        vec![Some(y_d_offset), Some(y_d_offset + 1), Some(y_d_offset + 2)],
        "the d block is the three inequalities in g order",
    );

    // the three numbers the map exists to keep apart
    let cap_row = mult_row(&s, G_CAP);
    assert_ne!(
        cap_row, G_CAP as usize,
        "the cap's flat row must not equal its g index, or a raw g \
         index would pass as a row",
    );
    assert_ne!(
        d_rows[2].unwrap() - y_d_offset,
        G_CAP,
        "the cap's d-block position must differ from its g index, or an \
         equality-blind scan would pass",
    );

    // and the derivative the whole fixture is about
    assert!(
        (full_slope(&s, cap_row, 1.0e-3, 1.0e-6) - CAP_MULT_SLOPE).abs() < 1e-6,
        "d(lambda_cap)/dp should be {CAP_MULT_SLOPE}, got {}",
        full_slope(&s, cap_row, 1.0e-3, 1.0e-6),
    );
}

/// A kink row's `y_d` entry is addressable -- the map does not gate,
/// and must not: a caller asking for a one-sided answer needs the row.
/// What the entry holds there is not a two-sided derivative, which is
/// the classifier's business and not the map's. Pinned so that a
/// future "helpfully" gating map is a deliberate change.
#[test]
fn the_map_addresses_every_inequality_including_the_ones_with_no_derivative() {
    let s = solved_cap(None, false);
    for g in [G_DECOY, G_CAP, G_KINK] {
        assert!(
            s.d_multiplier_rows(&[g]).expect("y_d rows")[0].is_some(),
            "g{g} ({}) must be addressable in y_d",
            row_class(&s, g),
        );
    }
    // and the classifier is what separates them, at classes the
    // DIRECTIONAL normalizer would not have to agree with
    assert_ne!(row_class(&s, G_CAP), row_class(&s, G_KINK));
    assert_ne!(row_class(&s, G_DECOY), row_class(&s, G_KINK));
}

// ---------------- leg 1: scaling ----------------

/// Per-variable factors for the cap fixture, spread over decades.
const D_CAP: [Number; 4] = [1.0e-2, 5.0, 0.25, 1.0e3];
const D_CAP_FIXED: [Number; 5] = [2.0, 1.0e-2, 5.0, 0.25, 1.0e3];

/// The row a constraint's multiplier lives in is a property of the c/d
/// split, not of the units the solve ran in -- and the derivative it
/// carries is in natural units, so the change of variables must not
/// reach it either.
///
/// `y_d`'s natural-units conversion is `E = dd`, `F = dd/df`, exactly
/// `y_c`'s one block over. A conversion that used the `x` block's
/// factors, or that dropped `df`, moves this number without moving any
/// row index -- so the leg asserts the derivative, not just the row.
#[test]
fn leg_scaling_the_cap_multiplier_row_is_unmoved_by_the_change_of_variables() {
    let plain = solved_cap(None, false);
    let scaled = solved_cap(Some(D_CAP.to_vec()), false);

    for g in [G_PIN, G_DECOY, G_CAP, G_KINK] {
        assert_eq!(
            plain.d_multiplier_rows(&[g]).expect("y_d rows"),
            scaled.d_multiplier_rows(&[g]).expect("y_d rows"),
            "g{g}: the y_d row moved under the change of variables",
        );
        assert_eq!(
            row_class(&plain, g),
            row_class(&scaled, g),
            "g{g}: the activity regime moved under the change of variables",
        );
    }

    for sign in [1.0, -1.0] {
        let (hi, lo) = (sign * 1.0e-3, sign * 1.0e-6);
        assert_close(
            "scaled d(lambda_cap)/dp",
            &[full_slope(&scaled, mult_row(&scaled, G_CAP), hi, lo)],
            &[CAP_MULT_SLOPE],
            1e-5,
        );
        assert_close(
            "plain d(lambda_cap)/dp",
            &[full_slope(&plain, mult_row(&plain, G_CAP), hi, lo)],
            &[CAP_MULT_SLOPE],
            1e-5,
        );
    }
}

// ---------------- leg 2: perturbation magnitude ----------------

/// The cap is strictly active, so its multiplier derivative is
/// two-sided and must be the same number over eight orders of `delta`
/// and on either side -- unlike the kink's, which has no two-sided
/// value at all. An absolute tolerance anywhere on the path to this
/// number reads a `1e-10` step as no step, which is gh#672 finding 4.
#[test]
fn leg_magnitude_the_cap_multiplier_derivative_does_not_depend_on_the_step_size() {
    let s = solved_cap(None, false);
    let row = mult_row(&s, G_CAP);
    for sign in [1.0, -1.0] {
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("d(lambda_cap)/dp over [{lo:e}, {hi:e}]"),
                &[full_slope(&s, row, hi, lo)],
                &[CAP_MULT_SLOPE],
                1e-5,
            );
        }
        let (hi, lo) = (sign * DELTAS[0], sign * DELTAS[DELTAS.len() - 1]);
        assert_close(
            &format!("d(lambda_cap)/dp over the full span [{lo:e}, {hi:e}]"),
            &[full_slope(&s, row, hi, lo)],
            &[CAP_MULT_SLOPE],
            1e-5,
        );
    }
}

/// ...and the same sweep under the change of variables, so leg 2's
/// answer is not an artifact of unit scaling.
#[test]
fn leg_magnitude_the_cap_multiplier_derivative_holds_under_scaling_too() {
    let s = solved_cap(Some(D_CAP.to_vec()), false);
    let row = mult_row(&s, G_CAP);
    for sign in [1.0, -1.0] {
        for w in DELTAS.windows(2) {
            let (hi, lo) = (sign * w[0], sign * w[1]);
            assert_close(
                &format!("scaled d(lambda_cap)/dp over [{lo:e}, {hi:e}]"),
                &[full_slope(&s, row, hi, lo)],
                &[CAP_MULT_SLOPE],
                1e-5,
            );
        }
    }
}

// ---------------- leg 3: a fixed variable ahead of the rows ----------------

/// A `make_parameter`-removed variable shortens the factor's `x`
/// block, and the `y_d` block's offset is the sum of the `x`, `s` and
/// `y_c` widths -- so an offset assembled from the USER's `n` instead
/// of the factor's puts every `y_d` row one too far along. Nothing
/// about the two lengths is distinguishable on a model without a fixed
/// variable, which is the gh#450 hazard two blocks over.
///
/// The row itself is *expected* to move (the block ahead of it got
/// shorter). What must not move is the answer it carries, or the
/// d-block position within the block.
#[test]
fn leg_fixed_the_cap_multiplier_is_unmoved_by_a_fixed_variable_ahead_of_the_rows() {
    let plain = solved_cap(None, false);
    let fixed = solved_cap(None, true);

    // precondition: the spaces genuinely diverge, or the leg is
    // vacuous
    let n_plain = plain.n_full_x().expect("n_full_x");
    let n_fixed = fixed.n_full_x().expect("n_full_x");
    assert_eq!((n_plain, n_fixed), (4, 5), "fixture drifted");
    assert_eq!(
        plain.block_dims().expect("block dims")[0],
        fixed.block_dims().expect("block dims")[0],
        "the fixed variable must be removed from the x block, or the \
         leg tests nothing",
    );

    // the constraint side is untouched by a fixed VARIABLE, so the
    // d-block positions and the regimes must be identical
    let dims_p = plain.block_dims().expect("block dims");
    let dims_f = fixed.block_dims().expect("block dims");
    let off_p = (dims_p[0] + dims_p[1] + dims_p[2]) as Index;
    let off_f = (dims_f[0] + dims_f[1] + dims_f[2]) as Index;
    for g in [G_DECOY, G_CAP, G_KINK] {
        let rp = plain.d_multiplier_rows(&[g]).expect("y_d rows")[0].unwrap();
        let rf = fixed.d_multiplier_rows(&[g]).expect("y_d rows")[0].unwrap();
        assert_eq!(
            rp - off_p,
            rf - off_f,
            "g{g}: the d-block position moved with a fixed variable",
        );
        assert_eq!(
            row_class(&plain, g),
            row_class(&fixed, g),
            "g{g}: the activity regime moved with a fixed variable",
        );
    }

    for sign in [1.0, -1.0] {
        let (hi, lo) = (sign * 1.0e-3, sign * 1.0e-6);
        assert_close(
            "d(lambda_cap)/dp with a fixed variable ahead of the rows",
            &[full_slope(&fixed, mult_row(&fixed, G_CAP), hi, lo)],
            &[CAP_MULT_SLOPE],
            1e-5,
        );
    }
}

/// The corner: a fixed variable AND a change of variables, the same
/// composition [`the_legs_compose_at_the_fixed_and_scaled_corner`]
/// runs for the kink fixture.
#[test]
fn the_cap_legs_compose_at_the_fixed_and_scaled_corner() {
    let s = solved_cap(Some(D_CAP_FIXED.to_vec()), true);
    assert_eq!(row_class(&s, G_CAP), STRONGLY_ACTIVE);
    assert_eq!(row_class(&s, G_DECOY), INACTIVE);
    assert_eq!(row_class(&s, G_KINK), WEAKLY_ACTIVE);
    let row = mult_row(&s, G_CAP);
    for sign in [1.0, -1.0] {
        assert_close(
            "d(lambda_cap)/dp at the fixed-and-scaled corner",
            &[full_slope(&s, row, sign * 1.0e-3, sign * 1.0e-6)],
            &[CAP_MULT_SLOPE],
            1e-5,
        );
    }
}
