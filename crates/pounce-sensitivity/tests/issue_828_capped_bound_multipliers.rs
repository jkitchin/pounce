//! gh#828 — the parametric step's bound-multiplier block, and the
//! corrector's own, read back in the frame the gh#737 ceiling actually
//! built the operator in.
//!
//! `PdFullSpaceSolver` eliminates each bound row into the `x` / `s`
//! barrier diagonal and recovers it afterwards as
//! `dz = r_z/s − (z/s)·dx`, re-deriving `z/s` straight off the iterate.
//! That is right exactly when the diagonal it eliminated into *was*
//! `Σ = Σ z/s`. Under gh#737's ceiling it is not: the solve holds the
//! capped bound softly and reads its multiplier back stiffly, and the
//! two disagree by the cap ratio.
//!
//! # The fixture
//!
//! ```text
//! min exp(x0) − p·x0 + ½(x1 − (2.5 − 2p))² + ½(x2 + 5)²
//! s.t.  p = p0,   A·x2 + x3 = 0,   x1 ≥ 0,  x2 ≥ 0
//! ```
//!
//! with the closed form `x0 = ln p`, `x1 = max(0, 2.5 − 2p)`,
//! `x2 = x3 = 0`, so the truth is independent of the machinery under
//! test. `x2` sits hard on its bound at `Σ = z/s ≈ 2.5e12`; `A` is its
//! only Jacobian coefficient, so it sets that variable's ceiling
//! through `sigma_pin_cap(a) = (a/eps)·(a/64)` and sweeping `A` walks
//! the cap in and out. `x1` is a near-bound row whose slack shrinks
//! 5× over the step and `x0` carries the curvature, so the correction
//! has real work to do in the uncapped coordinates.
//!
//! # Mutation table
//!
//! Measured at `delta = 0.2`, `tol = 1e-10`, `bound_relax_factor = 0`,
//! budget 8. `dz` is the returned step's entry for `x2`'s lower-bound
//! multiplier, whose true value is `0` — `z2 = 5` at both parameter
//! values, the `½(x2+5)²` term not depending on `p`.
//!
//! | `A` | `dz` before | `dz` after | corrected error before | after |
//! |---|---|---|---|---|
//! | 1e-4 | 1.776e7  | −7.10e-6  | 1.7678e-2 (no progress) | 1.553e-4 |
//! | 1e-3 | 1.776e5  | −7.10e-8  | 1.7678e-2 (no progress) | 4.736e-8 |
//! | 1e-2 | 1.771e3  | −7.10e-10 | 1.7678e-2 (no progress) | 4.817e-10 |
//! | 1e-1 | 1.276e1  | −7.11e-12 | 1.0000e-10              | 1.000e-10 |
//! | 1.0  | −2.0e-12 | −2.0e-12  | 1.0000e-10              | 1.000e-10 |
//!
//! The ceiling does not bind at `A = 1.0` (`Σ = 2.5e12` against a cap
//! of `7.0e13`), which is the arm that pins the fix as a no-op off the
//! capped path — every number in that row is bit-identical across the
//! change. Reintroduce the defect by dropping either call to
//! `rescale_bound_multipliers` and the four capped rows go back to the
//! "before" column, and each mutation is caught by the test whose name
//! describes it: dropping the predictor's restores the `A⁻²` blow-up
//! in `dz` and the corrector then opens on it, while dropping only the
//! corrector's leaves it opening on the right `2.14e-2` and still
//! unable to reduce it — `improved() == false` at `A ≤ 1e-2`, from a
//! chord that softens the bound and reads it back stiff.
//!
//! Two more mutations, each caught by exactly one leg and by no
//! other: turning over the sign in the `px_u` arm — silent above,
//! because every rung in the table is bounded *below* only, so the
//! `z_u` block never executes — reddens
//! [`the_upper_bound_branch_is_read_back_through_its_cap_too`]; and
//! moving the rescale to *after* the natural-units conjugation, which
//! is invisible at unit scaling, reddens
//! [`the_capped_correction_survives_a_change_of_variables`].
//!
//! # Which branch this reaches
//!
//! Every case here corrects with `released == 0` and `pinned == 0` —
//! asserted below, so a later reader does not mistake it for
//! evidence about the active-set branches. Those diagonals are not a
//! rescaling of the base one and deliberately carry no ratio;
//! `corrector_ceiling.rs` is the file that drives the pinned branch
//! under the same ceiling.
//!
//! Nor is it evidence about the `s` block. Every fixture here has
//! `dims[1] == 0` — no inequality slacks — so `ratio_s` is empty and
//! the `pd_l` / `pd_u` arms of the rescale are never entered. They
//! carry the same two signs as the `x` arms one block over, and a
//! model with a bounded `d(x)` row stiff enough to reach
//! `sigma_pin_cap(1.0) = 7.0e13` is what would exercise them.

use std::cell::RefCell;
use std::rc::Rc;

use pounce_algorithm::application::IpoptApplication;
use pounce_common::types::{Index, Number};
use pounce_nlp::return_codes::ApplicationReturnStatus;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, Linearity, NlpInfo, ScalingRequest, Solution,
    SparsityRequest, StartingPoint, TNLP,
};
use pounce_sensitivity::Solver;

mod common;
use common::compare::{Frame, assert_same_correction};
use common::outside_engine::{RecordingTnlp, Relaxation, correct_outside, outside_engine};

const P0: Number = 1.0;
const DELTA: Number = 0.2;

/// min exp(x0) - p*x0 + ½(x1 - (2.5-2p))² + ½(x2+5)²
/// s.t. g0: p = p0,  g1: A*x2 + x3 = 0,  x1 >= 0, x2 >= 0
/// vars: x0, x1, x2, x3, p
struct Fixture {
    a: Number,
    p0: Number,
    start: Option<Vec<Number>>,
    /// Reflect `x2` through its bound: the objective term becomes
    /// `½(x2 − 5)²` and the bound an **upper** one at `0`, which is the
    /// same model under `x2 → −x2` and the same `Σ = 5 / s`. It reaches
    /// the other sign of the fold — see
    /// [`the_upper_bound_branch_is_read_back_through_its_cap_too`].
    mirror: bool,
    /// Per-variable `user-scaling` factors, or `None` for an unscaled
    /// solve.
    x_scaling: Option<Vec<Number>>,
}

impl TNLP for Fixture {
    fn get_nlp_info(&mut self) -> Option<NlpInfo> {
        Some(NlpInfo {
            n: 5,
            m: 2,
            nnz_jac_g: 3,
            nnz_h_lag: 6,
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
        b.x_l[0] = -1.0e19;
        b.x_u[0] = 1.0e19;
        b.x_l[1] = 0.0;
        b.x_u[1] = 1.0e19;
        if self.mirror {
            b.x_l[2] = -1.0e19;
            b.x_u[2] = 0.0;
        } else {
            b.x_l[2] = 0.0;
            b.x_u[2] = 1.0e19;
        }
        b.x_l[3] = -1.0e19;
        b.x_u[3] = 1.0e19;
        b.x_l[4] = -1.0e19;
        b.x_u[4] = 1.0e19;
        b.g_l[0] = self.p0;
        b.g_u[0] = self.p0;
        b.g_l[1] = 0.0;
        b.g_u[1] = 0.0;
        true
    }
    fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
        match &self.start {
            Some(x0) => sp.x.copy_from_slice(x0),
            None => {
                sp.x[0] = 0.0;
                sp.x[1] = 0.5;
                sp.x[2] = if self.mirror { -0.5 } else { 0.5 };
                sp.x[3] = 0.0;
                sp.x[4] = self.p0;
            }
        }
        true
    }
    fn get_constraints_linearity(&mut self, types: &mut [Linearity]) -> bool {
        types.fill(Linearity::Linear);
        true
    }
    fn eval_f(&mut self, x: &[Number], _n: bool) -> Option<Number> {
        let c = 2.5 - 2.0 * x[4];
        let pull = if self.mirror { -5.0 } else { 5.0 };
        Some(x[0].exp() - x[4] * x[0] + 0.5 * (x[1] - c).powi(2) + 0.5 * (x[2] + pull).powi(2))
    }
    fn eval_grad_f(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        let r = x[1] - (2.5 - 2.0 * x[4]);
        g[0] = x[0].exp() - x[4];
        g[1] = r;
        g[2] = x[2] + if self.mirror { -5.0 } else { 5.0 };
        g[3] = 0.0;
        g[4] = -x[0] + 2.0 * r;
        true
    }
    fn eval_g(&mut self, x: &[Number], _n: bool, g: &mut [Number]) -> bool {
        g[0] = x[4];
        g[1] = self.a * x[2] + x[3];
        true
    }
    fn eval_jac_g(&mut self, _x: Option<&[Number]>, _n: bool, mode: SparsityRequest<'_>) -> bool {
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0usize, 4usize), (1, 2), (1, 3)].iter().enumerate() {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                values.copy_from_slice(&[1.0, self.a, 1.0]);
            }
        }
        true
    }
    fn eval_h(
        &mut self,
        x: Option<&[Number]>,
        _n: bool,
        obj_factor: Number,
        _l: Option<&[Number]>,
        _nl: bool,
        mode: SparsityRequest<'_>,
    ) -> bool {
        // lower triangle: (0,0) (1,1) (2,2) (4,4) (4,0) (4,1)
        match mode {
            SparsityRequest::Structure { irow, jcol } => {
                for (k, &(r, c)) in [(0usize, 0usize), (1, 1), (2, 2), (4, 4), (4, 0), (4, 1)]
                    .iter()
                    .enumerate()
                {
                    irow[k] = r as Index;
                    jcol[k] = c as Index;
                }
            }
            SparsityRequest::Values { values } => {
                let x0 = x.map_or(0.0, |x| x[0]);
                values[0] = obj_factor * x0.exp();
                values[1] = obj_factor;
                values[2] = obj_factor;
                values[3] = obj_factor * 4.0;
                values[4] = -obj_factor;
                values[5] = obj_factor * 2.0;
            }
        }
        true
    }
    fn finalize_solution(&mut self, _s: Solution<'_>, _d: &IpoptData, _c: &IpoptCq) {}
}

fn solved(a: Number, p0: Number, start: Option<Vec<Number>>) -> Solver {
    solved_with(a, p0, start, false, None)
}

fn solved_with(
    a: Number,
    p0: Number,
    start: Option<Vec<Number>>,
    mirror: bool,
    x_scaling: Option<Vec<Number>>,
) -> Solver {
    let app = app(x_scaling.is_some());
    let tnlp: Rc<RefCell<dyn TNLP>> = Rc::new(RefCell::new(Fixture {
        a,
        p0,
        start,
        mirror,
        x_scaling,
    }));
    let mut solver = Solver::new(app, tnlp);
    let st = solver.solve();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "a={a:e}: {st:?}"
    );
    solver
}

/// The options every solve of this fixture runs with.
fn app(user_scaling: bool) -> IpoptApplication {
    let mut app = IpoptApplication::new();
    {
        let o = app.options_mut();
        o.set_integer_value("print_level", 0, true, false).unwrap();
        o.set_string_value("sb", "yes", true, false).unwrap();
        o.set_numeric_value("bound_relax_factor", 0.0, true, false)
            .unwrap();
        o.set_numeric_value("tol", 1e-10, true, false).unwrap();
        if user_scaling {
            o.set_string_value("nlp_scaling_method", "user-scaling", true, false)
                .unwrap();
        }
    }
    app.initialize().unwrap();
    app
}

/// [`solved_with`] from the default start, with the model wrapped to record
/// what the solve returns.
fn solved_recorded(
    a: Number,
    mirror: bool,
    x_scaling: Option<Vec<Number>>,
) -> (Solver, Rc<RefCell<RecordingTnlp<Fixture>>>) {
    let user_scaling = x_scaling.is_some();
    let model = Rc::new(RefCell::new(RecordingTnlp::new(Fixture {
        a,
        p0: P0,
        start: None,
        mirror,
        x_scaling,
    })));
    let tnlp: Rc<RefCell<dyn TNLP>> = model.clone();
    let mut solver = Solver::new(app(user_scaling), tnlp);
    let st = solver.solve();
    assert!(
        matches!(
            st,
            ApplicationReturnStatus::SolveSucceeded
                | ApplicationReturnStatus::SolvedToAcceptableLevel
        ),
        "a={a:e}: {st:?}"
    );
    (solver, model)
}

fn truth(p: Number) -> Vec<Number> {
    vec![p.ln(), (2.5 - 2.0 * p).max(0.0), 0.0, 0.0, p]
}

fn dist(a: &[Number], b: &[Number]) -> Number {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (&x, &y)| m.max((x - y).abs()))
}

/// Row coefficients that walk the gh#737 ceiling from biting hard to
/// not biting at all.
const COEFFS: [Number; 5] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0];

/// `x2`'s lower-bound multiplier row inside the compound KKT vector,
/// checked against the block dimensions rather than assumed: the `z_l`
/// block holds the bounded-below variables in order, and `x1` and `x2`
/// are the only two.
fn capped_dz_row(dims: &[usize; 8]) -> usize {
    assert_eq!(
        dims,
        &[5, 0, 2, 0, 2, 0, 0, 0],
        "fixture shape moved; the z_l row below is read off it"
    );
    dims[0] + dims[1] + dims[2] + dims[3] + 1
}

/// The predictor's multiplier block stays in the capped frame.
///
/// Before the fix this entry grew as `A⁻²` — 1.776e7 at `A = 1e-4` —
/// against a true `0`, because the elimination used the capped `Σ`
/// and the recovery divided by the uncapped `s`.
#[test]
fn the_parametric_step_reads_a_capped_bound_back_through_its_cap() {
    for a in COEFFS {
        let s = solved(a, P0, None);
        let dims = s.block_dims().expect("block dims");
        let row = capped_dz_row(&dims);
        let step = s.parametric_step_full(&[0], &[DELTA]).expect("step");
        let dz = step[row];
        assert!(
            dz.abs() < 1e-3,
            "a={a:e}: dz={dz:e} on a bound whose multiplier does not move"
        );
    }
}

/// `correct_step` improves on its input at every rung of the sweep,
/// and the improvement is not cosmetic: the error against the closed
/// form drops by at least two orders from the predictor's.
///
/// Before the fix the four capped rungs stopped after one iteration
/// with `improved() == false` and handed back the uncorrected step —
/// `estimate(corrector_iter = ...)` silently unavailable in exactly
/// the stiff, tightly-bounded regime a caller reaches for it in.
#[test]
fn the_corrector_makes_progress_under_the_ceiling() {
    let want = truth(P0 + DELTA);
    for a in COEFFS {
        let s = solved(a, P0, None);
        let base = s.converged().expect("converged").x.clone();
        let n = base.len();
        let step = s.parametric_step_full(&[0], &[DELTA]).expect("step");
        let plain: Vec<Number> = base.iter().zip(&step[..n]).map(|(&b, &d)| b + d).collect();
        let plain_err = dist(&plain, &want);

        let (out, rep) = s.correct_step(&[0], &[DELTA], &step, 8).expect("correct");
        let corrected: Vec<Number> = base.iter().zip(&out[..n]).map(|(&b, &d)| b + d).collect();
        let err = dist(&corrected, &want);

        assert!(
            rep.improved(),
            "a={a:e}: no progress, r0={:e}",
            rep.initial_residual
        );
        assert!(
            err < plain_err / 100.0,
            "a={a:e}: corrected {err:e} against predicted {plain_err:e}"
        );
        // The residual the corrector opens on is the model's own
        // second-order term, not the cap's: it is the same 2.14e-2 at
        // every rung, where before the fix it ran 1.776e7 down to
        // 2.14e-2 as the ceiling let go.
        assert!(
            (rep.initial_residual - 2.14e-2).abs() < 1e-3,
            "a={a:e}: initial residual {:e}",
            rep.initial_residual
        );
        // See the module header: this leg says nothing about the
        // release or pin branches.
        assert_eq!((rep.released, rep.pinned), (0, 0), "a={a:e}");
    }
}

/// Off the capped path nothing moves. `A = 1.0` leaves `Σ = 2.5e12`
/// under a ceiling of `7.0e13`, and the whole correction has to be
/// bit-identical to what it was before the ratio existed — the
/// `1.0` row of the module's mutation table.
#[test]
fn an_uncapped_bound_is_untouched_by_the_ratio() {
    let want = truth(P0 + DELTA);
    let s = solved(1.0, P0, None);
    let base = s.converged().expect("converged").x.clone();
    let n = base.len();
    let dims = s.block_dims().expect("block dims");
    let step = s.parametric_step_full(&[0], &[DELTA]).expect("step");
    // `dz` here is the barrier standoff `−s2`, the whole of what the
    // equation-11 correction takes off a bound at `mu`; it is what the
    // capped rows should approach and never did.
    assert!(
        (step[capped_dz_row(&dims)] + 2.0e-12).abs() < 1e-14,
        "dz={:e}",
        step[capped_dz_row(&dims)]
    );
    let (out, rep) = s.correct_step(&[0], &[DELTA], &step, 8).expect("correct");
    let corrected: Vec<Number> = base.iter().zip(&out[..n]).map(|(&b, &d)| b + d).collect();
    assert_eq!(rep.iterations, 8);
    // 1e-10 is the corrector's own box put-back margin,
    // `1e-10 * (1 + |base_i|)`, on `x2` — the floor this fixture can
    // reach, not a tolerance chosen to pass.
    assert!(dist(&corrected, &want) < 1.1e-10);
}

/// The corrected error, as a fraction of the closed form, for one
/// solved arm of the sweep. The two legs below both work by comparing
/// this number across a transformation that must not move it.
fn corrected_error(s: &Solver, row: usize) -> (Number, Number, bool) {
    let base = s.converged().expect("converged").x.clone();
    let n = base.len();
    let want = truth(P0 + DELTA);
    let step = s.parametric_step_full(&[0], &[DELTA]).expect("step");
    let (out, rep) = s.correct_step(&[0], &[DELTA], &step, 8).expect("correct");
    let corrected: Vec<Number> = base.iter().zip(&out[..n]).map(|(&b, &d)| b + d).collect();
    assert_eq!((rep.released, rep.pinned), (0, 0));
    (dist(&corrected, &want), step[row], rep.improved())
}

/// The **upper**-bound sign of the fold.
///
/// A lower bound folds into the diagonal as `−z·dx` and an upper one
/// as `+z·dx`, so the correction that puts the row back in the capped
/// frame carries the opposite sign on each. Every fixture above is
/// bounded below only, and a sign error on the other block is silent
/// there: the whole `z_u` path would never execute.
///
/// The mirror is the same model under `x2 → −x2`, so the answer is
/// the *same number*, and that is what is asserted — not a loose
/// bound. Flip the sign in `rescale_bound_multipliers`'s `px_u` arm
/// and the row comes back at roughly `−2·(z/s)·dx`, i.e. the defect's
/// own `1.8e7` with the sign turned over.
#[test]
fn the_upper_bound_branch_is_read_back_through_its_cap_too() {
    for a in COEFFS {
        let lower = solved(a, P0, None);
        let l_dims = lower.block_dims().expect("block dims");
        let (l_err, l_dz, l_imp) = corrected_error(&lower, capped_dz_row(&l_dims));

        let upper = solved_with(a, P0, None, true, None);
        let u_dims = upper.block_dims().expect("block dims");
        assert_eq!(
            u_dims,
            [5, 0, 2, 0, 1, 1, 0, 0],
            "mirror shape moved; the z_u row below is read off it"
        );
        // `x1`'s lower-bound row is the whole of `z_l`; `x2`'s
        // upper-bound row is the whole of `z_u`, which starts after it.
        let u_row = u_dims[0] + u_dims[1] + u_dims[2] + u_dims[3] + u_dims[4];
        let (u_err, u_dz, u_imp) = corrected_error(&upper, u_row);

        assert!(l_imp && u_imp, "a={a:e}: no progress");
        assert!(
            (u_dz - l_dz).abs() <= 1e-6 * l_dz.abs(),
            "a={a:e}: dz {u_dz:e} on the upper bound against {l_dz:e} on the lower"
        );
        assert!(
            (u_err - l_err).abs() <= 1e-6 * l_err,
            "a={a:e}: corrected error {u_err:e} against the mirror's {l_err:e}"
        );
    }
}

/// Invariance leg 1 for the capped multiplier rows: the correction is
/// read off `z / s` in the solver's **scaled** frame and applied to a
/// scaled `dx`, before the natural-units conjugation, so a frame slip
/// there is invisible at unit scaling — which is every arm above.
///
/// Under `x̃ = d ⊙ x` the barrier diagonal carries `d⁻²` and so does
/// the ceiling, because `sigma_pin_cap` is quadratic in a Jacobian
/// coefficient that carries `d⁻¹`: the cap therefore binds on exactly
/// the same rungs in both arms, which is what makes this leg a fair
/// comparison rather than a comparison of a capped run to an uncapped
/// one. The corrected answer is in natural units and must not move.
///
/// `initial_residual` is deliberately **not** compared: it is measured
/// in the scaled frame and legitimately halves here.
#[test]
fn the_capped_correction_survives_a_change_of_variables() {
    let d = vec![2.0, 0.5, 4.0, 0.25, 1.0];
    for a in COEFFS {
        let plain = solved(a, P0, None);
        let p_dims = plain.block_dims().expect("block dims");
        let (p_err, p_dz, p_imp) = corrected_error(&plain, capped_dz_row(&p_dims));

        let scaled = solved_with(a, P0, None, false, Some(d.clone()));
        let s_dims = scaled.block_dims().expect("block dims");
        let (s_err, s_dz, s_imp) = corrected_error(&scaled, capped_dz_row(&s_dims));

        assert!(p_imp && s_imp, "a={a:e}: no progress");
        // Measured spread across the sweep is 0 to 7.5e-4 relative —
        // the two arms converge to slightly different points, and this
        // row is the barrier standoff read at whichever one. The
        // budget is 13x that rather than 1.3x on purpose: both
        // mutations this leg exists for (a frame slip, a sign flip)
        // move it by order one or more, so there is nothing to buy by
        // tightening it into flakiness.
        assert!(
            (s_dz - p_dz).abs() <= 1e-2 * p_dz.abs(),
            "a={a:e}: dz {s_dz:e} scaled against {p_dz:e} unscaled"
        );
        assert!(
            (s_err - p_err).abs() <= 1e-6 * p_err,
            "a={a:e}: corrected error {s_err:e} scaled against {p_err:e} unscaled"
        );
    }
    // Not a vacuous leg: at the stiffest rung the ceiling really does
    // bind in the scaled arm. Its `dz` is the capped standoff, six
    // orders off the `−2.0e-12` an uncapped row returns.
    let s = solved_with(1e-4, P0, None, false, Some(d));
    let dims = s.block_dims().expect("block dims");
    let dz = s.parametric_step_full(&[0], &[DELTA]).expect("step")[capped_dz_row(&dims)];
    assert!(dz.abs() > 1e-7, "the ceiling did not bind: dz={dz:e}");
}

// ---------------------------------------------------------------------------
// The same corrections over an outside engine
// ---------------------------------------------------------------------------
//
// Each test below repeats the setup of the test above it with the same name
// minus `_over_an_outside_engine`, on a solve whose model records what pounce
// returned, and hands the same step to the corrector over an engine built
// from that report.
//
// The sweep is a sweep of the gh#737 ceiling, and the ceiling belongs to
// pounce's factorization, not to the point. pounce factors the augmented
// system, where `x2`'s barrier diagonal `z/s` sits beside a constraint row of
// coefficient `a`, and above the ceiling that factorization loses the step,
// so pounce caps the diagonal. The outside engine factors the eight-block
// matrix, which keeps `z` and `s` apart, and needs no cap. Measured over
// eight iterations, distance from the re-solved truth:
//
// ```text
// a        pounce     outside
// 1e-4     1.553e-4   1.000e-10
// 1e-3     4.736e-8   1.000e-10
// 1e-2     4.817e-10  1.000e-10
// 1e-1     1.000e-10  1.000e-10
// 1e0      1.000e-10  1.000e-10
// ```
//
// So where the ceiling does not bind the two are the same correction, and
// where it binds the outside engine is the closer to the truth, by what the
// cap costs. The mirrored fixture measures the same to every digit.

/// The coefficients at and above which the ceiling leaves `x2`'s diagonal
/// alone.
const UNCAPPED_FROM: Number = 1e-1;

/// pounce's correction and the outside engine's from the same step, with
/// what each is distant from the truth.
fn compare_at(a: Number, mirror: bool, x_scaling: Option<Vec<Number>>) {
    let scaled = x_scaling.is_some();
    let (s, model) = solved_recorded(a, mirror, x_scaling);
    let base = s.converged().expect("converged").x.clone();
    let n = base.len();
    let want = truth(P0 + DELTA);
    let step = s.parametric_step_full(&[0], &[DELTA]).expect("step");
    let plain: Vec<Number> = base.iter().zip(&step[..n]).map(|(&b, &d)| b + d).collect();
    let plain_err = dist(&plain, &want);
    let pounce = s
        .correct_step(&[0], &[DELTA], &step, 8)
        .expect("pounce's corrector");
    let eng = outside_engine(&model, Relaxation::None);
    let outside = correct_outside(&eng, &[0], &[DELTA], &step, 8).expect("outside corrector");
    let err = |o: &[Number]| -> Number {
        let x: Vec<Number> = base.iter().zip(&o[..n]).map(|(&b, &d)| b + d).collect();
        dist(&x, &want)
    };
    let (ea, eb) = (err(&pounce.0), err(&outside.0));
    let label = format!("a={a:e}, mirror {mirror}, scaled {scaled}");
    if a >= UNCAPPED_FROM {
        let frame = if scaled { Frame::Scaled } else { Frame::Same };
        assert_same_correction(&label, &pounce, &outside, frame);
    } else {
        assert_eq!(
            (pounce.1.released, pounce.1.pinned),
            (outside.1.released, outside.1.pinned),
            "{label}: the active-set decision differs",
        );
        assert!(
            eb <= ea,
            "{label}: the outside engine lands {eb:e} from the truth, pounce {ea:e}",
        );
    }
    assert!(
        outside.1.improved() && eb < plain_err / 100.0,
        "{label}: outside {eb:e} against predicted {plain_err:e}",
    );
}

#[test]
fn the_corrector_makes_progress_under_the_ceiling_over_an_outside_engine() {
    for a in COEFFS {
        compare_at(a, false, None);
    }
}

#[test]
fn an_uncapped_bound_is_untouched_by_the_ratio_over_an_outside_engine() {
    compare_at(1.0, false, None);
}

#[test]
fn the_upper_bound_branch_is_read_back_through_its_cap_too_over_an_outside_engine() {
    for a in COEFFS {
        compare_at(a, true, None);
    }
}

#[test]
fn the_capped_correction_survives_a_change_of_variables_over_an_outside_engine() {
    let d = vec![2.0, 0.5, 4.0, 0.25, 1.0];
    for a in COEFFS {
        compare_at(a, false, Some(d.clone()));
    }
}
