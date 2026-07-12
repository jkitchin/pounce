//! Default iterate initializer — port of
//! `Algorithm/IpDefaultIterateInitializer.{hpp,cpp}`.
//!
//! Bound push, slack init, multiplier init (constant / mu-based /
//! least-square via the `EqMultCalculator`). Constants below match
//! upstream's defaults from `RegisterOptions`.
//!
//! `set_initial_iterates` ports the upstream sequence:
//!
//! 1. Pull `x` from `nlp.get_starting_x` and push each component
//!    into the interior of `[x_l, x_u]` per
//!    [`DefaultIterateInitializer::push_to_interior`].
//! 2. Set `s = d(x)` (evaluated through CQ on a transient iterate)
//!    and push it into the interior of `[d_l, d_u]`.
//! 3. Initialize `y_c`, `y_d`:
//!    * `bound_mult_init_method == "constant"` — leave at zero (the
//!      default y-targets that the linear-solver sweep will refine).
//!    * `bound_mult_init_method == "least-square"` — call
//!      [`crate::eq_mult::least_square::LeastSquareMults`] via the
//!      provided `aug_solver`. **Phase 7 default is "constant"** to
//!      avoid pulling the aug-system through this path on bring-up.
//! 4. Initialize `z_l`, `z_u`, `v_l`, `v_u` to `bound_mult_init_val`
//!    (component-wise).
//!
//! The fully-loaded mu-based / least-square multiplier paths land
//! once `LeastSquareMults` and the iterate-trace gold tests are
//! online.

use crate::eq_mult::r#trait::EqMultCalculator;
use crate::init::r#trait::IterateInitializer;
use crate::ipopt_cq::IpoptCqHandle;
use crate::ipopt_data::IpoptDataHandle;
use crate::ipopt_nlp::IpoptNlp;
use crate::iterates_vector::IteratesVector;
use crate::kkt::aug_system_solver::{AugSysCoeffs, AugSysRhs, AugSysSol, AugSystemSolver};
use pounce_common::types::{Index, Number};
use pounce_linalg::Vector;
use pounce_linalg::dense_vector::{DenseVector, DenseVectorSpace};
use std::cell::RefCell;
use std::rc::Rc;

pub struct DefaultIterateInitializer {
    pub bound_push: Number,
    pub bound_frac: Number,
    pub slack_bound_push: Number,
    pub slack_bound_frac: Number,
    pub constr_mult_init_max: Number,
    pub bound_mult_init_val: Number,
    /// "constant" / "mu-based" / "least-square".
    pub bound_mult_init_method: String,
    /// Equality-multiplier calculator used by the
    /// `least_square_mults` step at the end of `set_initial_iterates`,
    /// matching upstream `IpDefaultIterateInitializer.cpp:334-341`. If
    /// `None`, the LS step is skipped (y_c, y_d remain at zero).
    pub eq_mult_calculator: Option<Box<dyn EqMultCalculator>>,
    /// `least_square_init_primal` — port of
    /// `IpDefaultIterateInitializer.cpp:200-222`. When on, the
    /// initializer replaces the user's starting `x` with the min-norm
    /// solution of the linearized equality + inequality constraints,
    /// then pushes that to the interior. Used by the Mehrotra cascade
    /// (`IpIpoptAlg.cpp:182`) to dramatically reduce iter-0 primal
    /// infeasibility on LP-shaped problems.
    pub least_square_init_primal: bool,
}

impl Default for DefaultIterateInitializer {
    fn default() -> Self {
        Self {
            bound_push: 1e-2,
            bound_frac: 1e-2,
            slack_bound_push: 1e-2,
            slack_bound_frac: 1e-2,
            constr_mult_init_max: 1e3,
            bound_mult_init_val: 1.0,
            bound_mult_init_method: "constant".into(),
            eq_mult_calculator: None,
            least_square_init_primal: false,
        }
    }
}

impl DefaultIterateInitializer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_eq_mult_calculator(eq_mult: Box<dyn EqMultCalculator>) -> Self {
        Self {
            eq_mult_calculator: Some(eq_mult),
            ..Self::default()
        }
    }

    /// Per-element bound-push formula from upstream
    /// `IpDefaultIterateInitializer.cpp:473-666`. Given a primal value
    /// `x` and optional bounds `(lower, upper)`, return a value
    /// shifted to the interior:
    ///
    /// * Two-sided bounds: clamp into `[lo + p_l, hi - p_u]` where
    ///   `p_l = min(bound_push * max(|lo|, 1), bound_frac * (hi-lo))`,
    ///   `p_u = min(bound_push * max(|hi|, 1), bound_frac * (hi-lo))`.
    /// * Lower-only: return `max(x, lo + bound_push * max(|lo|, 1))`.
    /// * Upper-only: return `min(x, hi - bound_push * max(|hi|, 1))`.
    /// * Free: return `x`.
    ///
    /// The `Px_L`/`Px_U` selection-matrix dance in upstream collapses
    /// to exactly this per-coordinate formula once the bounds are
    /// expanded to the full primal space.
    /// Port of `IpDefaultIterateInitializer.cpp:CalculateLeastSquarePrimals`.
    /// Solves the augmented system with `W=0`, `D_x=I`, `D_s=I` and
    /// `rhs=(0, 0, curr_c, curr_d)`; on success returns the min-norm
    /// `x_ls` (negated per upstream `x_ls.Scal(-1)`). `s_ls` is
    /// discarded — upstream overwrites it with `trial_d(trial_x)`
    /// after pushing `x_ls` to the interior, so we save the allocation
    /// and re-evaluate `d` later in `set_initial_iterates`. Assumes
    /// `data.curr.x` already holds the point at which the constraints
    /// and Jacobians should be linearized.
    fn calculate_least_square_primals(
        &self,
        cq: &IpoptCqHandle,
        _nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
        n_x: Index,
    ) -> Option<Rc<dyn Vector>> {
        let cq_ref = cq.borrow();
        let curr_c = cq_ref.curr_c();
        let curr_d = cq_ref.curr_d();
        let j_c = cq_ref.curr_jac_c();
        let j_d = cq_ref.curr_jac_d();
        // `zeroW` pins the W triplet structure in the linsol so later
        // calls with the real Hessian write into the right slots
        // (mirrors `IpLeastSquareMults`).
        let zero_w = cq_ref.curr_exact_hessian();
        drop(cq_ref);

        let n_s = curr_d.dim();
        let n_c = curr_c.dim();
        let n_d = curr_d.dim();

        let mut rhs_x = DenseVectorSpace::new(n_x).make_new_dense();
        rhs_x.set(0.0);
        let mut rhs_s = DenseVectorSpace::new(n_s).make_new_dense();
        rhs_s.set(0.0);
        let mut rhs_c_v = curr_c.make_new();
        rhs_c_v.copy(&*curr_c);
        let mut rhs_d_v = curr_d.make_new();
        rhs_d_v.copy(&*curr_d);

        let mut sol_x = DenseVectorSpace::new(n_x).make_new_dense();
        let mut sol_s = DenseVectorSpace::new(n_s).make_new_dense();
        let mut sol_c = DenseVectorSpace::new(n_c).make_new_dense();
        let mut sol_d = DenseVectorSpace::new(n_d).make_new_dense();

        let coeffs = AugSysCoeffs {
            w: Some(&*zero_w),
            w_factor: 0.0,
            d_x: None,
            delta_x: 1.0,
            d_s: None,
            delta_s: 1.0,
            j_c: &*j_c,
            d_c: None,
            // Tiny δ_c, δ_d (upstream uses 0). pounce-feral's LDL^T
            // mis-reports the inertia of an augmented system with a
            // structurally-zero (3,3)/(4,4) block — it counted 0
            // negative eigenvalues on nuffield2_trap where the true
            // count is n_c+n_d, triggering WrongInertia. Perturbing
            // by 1e-8 keeps the LS solution numerically identical
            // (the constraint Jacobian dominates this term) while
            // giving the diagonal something nonzero to pivot on.
            delta_c: 1e-8,
            j_d: &*j_d,
            d_d: None,
            delta_d: 1e-8,
        };
        let aug_rhs = AugSysRhs {
            rhs_x: &rhs_x,
            rhs_s: &rhs_s,
            rhs_c: &*rhs_c_v,
            rhs_d: &*rhs_d_v,
        };
        let mut sol = AugSysSol {
            sol_x: &mut sol_x,
            sol_s: &mut sol_s,
            sol_c: &mut sol_c,
            sol_d: &mut sol_d,
        };

        // Upstream `IpDefaultIterateInitializer.cpp:381` passes
        // check_NegEVals=true, numberOfNegEVals=n_c+n_d (matches the
        // expected inertia of the W=0,Dx=I,Ds=I augmented system).
        let num_eq = n_c + n_d;
        let check_neg = aug_solver.provides_inertia();
        let status = aug_solver.solve(&coeffs, &aug_rhs, &mut sol, check_neg, num_eq);
        if !matches!(status, pounce_linsol::ESymSolverStatus::Success) {
            return None;
        }
        // Upstream `IpDefaultIterateInitializer.cpp:386-387`:
        // x_ls.Scal(-1); s_ls.Scal(-1).
        sol_x.scal(-1.0);
        Some(Rc::new(sol_x))
    }

    pub fn push_to_interior(
        bound_push: Number,
        bound_frac: Number,
        x: Number,
        lower: Option<Number>,
        upper: Option<Number>,
    ) -> Number {
        match (lower, upper) {
            (Some(lo), Some(hi)) => {
                let span = hi - lo;
                let p_l = (bound_push * lo.abs().max(1.0)).min(bound_frac * span);
                let p_u = (bound_push * hi.abs().max(1.0)).min(bound_frac * span);
                x.max(lo + p_l).min(hi - p_u)
            }
            (Some(lo), None) => {
                let p_l = bound_push * lo.abs().max(1.0);
                x.max(lo + p_l)
            }
            (None, Some(hi)) => {
                let p_u = bound_push * hi.abs().max(1.0);
                x.min(hi - p_u)
            }
            (None, None) => x,
        }
    }
}

impl IterateInitializer for DefaultIterateInitializer {
    fn set_initial_iterates(
        &mut self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
    ) -> bool {
        let curr_template = match data.borrow().curr.clone() {
            Some(c) => c,
            None => return false,
        };

        let n_x = curr_template.x.dim();
        let n_s = curr_template.s.dim();
        let n_yc = curr_template.y_c.dim();
        let n_yd = curr_template.y_d.dim();
        let n_zl = curr_template.z_l.dim();
        let n_zu = curr_template.z_u.dim();
        let n_vl = curr_template.v_l.dim();
        let n_vu = curr_template.v_u.dim();

        // Step 1: pull x from NLP and push each finite-bounded
        // component into the interior. Bound vectors `x_l`, `x_u` are
        // packed (only finite entries); we expand via `Px_L^T` masks
        // by walking the dense slot.
        let mut x = DenseVectorSpace::new(n_x).make_new_dense();
        nlp.borrow_mut().get_starting_x(&mut x);

        // Step 1.5 (optional): replace `x` with the min-norm solution
        // of the linearized equality + inequality constraints. Port of
        // `IpDefaultIterateInitializer.cpp:200-222`. The Mehrotra
        // cascade in `application.rs` turns this on; it is the iter-0
        // feasibility correction that lets Mehrotra LPs land on the
        // central path on the first solve. Failure leaves `x` as-is.
        if self.least_square_init_primal && (n_yc + n_yd) > 0 {
            // Stage a partial iterate with the user's starting `x` and
            // zeros for everything else, so `cq.curr_*` evaluates at
            // the right point.
            let mut x_stage = DenseVectorSpace::new(n_x).make_new_dense();
            x_stage.copy(&x);
            let mut s_zero = DenseVectorSpace::new(n_s).make_new_dense();
            s_zero.set(0.0);
            let mut y_c_zero = DenseVectorSpace::new(n_yc).make_new_dense();
            y_c_zero.set(0.0);
            let mut y_d_zero = DenseVectorSpace::new(n_yd).make_new_dense();
            y_d_zero.set(0.0);
            let mut z_l_zero = DenseVectorSpace::new(n_zl).make_new_dense();
            z_l_zero.set(0.0);
            let mut z_u_zero = DenseVectorSpace::new(n_zu).make_new_dense();
            z_u_zero.set(0.0);
            let mut v_l_zero = DenseVectorSpace::new(n_vl).make_new_dense();
            v_l_zero.set(0.0);
            let mut v_u_zero = DenseVectorSpace::new(n_vu).make_new_dense();
            v_u_zero.set(0.0);
            let stage_iv = IteratesVector::new(
                Rc::new(x_stage),
                Rc::new(s_zero),
                Rc::new(y_c_zero),
                Rc::new(y_d_zero),
                Rc::new(z_l_zero),
                Rc::new(z_u_zero),
                Rc::new(v_l_zero),
                Rc::new(v_u_zero),
            );
            data.borrow_mut().set_curr(stage_iv);

            if let Some(x_ls) = self.calculate_least_square_primals(cq, nlp, aug_solver, n_x) {
                x.copy(&*x_ls);
            }
        }

        {
            let nlp_ref = nlp.borrow();
            push_x_into_interior(
                &mut x,
                &*nlp_ref.px_l(),
                nlp_ref.x_l(),
                &*nlp_ref.px_u(),
                nlp_ref.x_u(),
                self.bound_push,
                self.bound_frac,
            );
        }

        // Step 2: s = d(x), then push into [d_l, d_u].
        let mut s = DenseVectorSpace::new(n_s).make_new_dense();
        nlp.borrow_mut().eval_d(&x, &mut s);
        {
            let nlp_ref = nlp.borrow();
            push_x_into_interior(
                &mut s,
                &*nlp_ref.pd_l(),
                nlp_ref.d_l(),
                &*nlp_ref.pd_u(),
                nlp_ref.d_u(),
                self.slack_bound_push,
                self.slack_bound_frac,
            );
        }

        // Step 3: y_c, y_d initial guesses. `constant` mode leaves
        // them at zero (the algorithm refines on the first KKT solve).
        let mut y_c = DenseVectorSpace::new(n_yc).make_new_dense();
        let mut y_d = DenseVectorSpace::new(n_yd).make_new_dense();
        if self.bound_mult_init_method == "constant" {
            // Materialize as homogeneous-zero so callers' asum / values
            // probes don't trip the `initialized` debug-assert.
            y_c.set(0.0);
            y_d.set(0.0);
        } else {
            // Other modes (mu-based, least-square) require the
            // aug-system path; fall back to NLP's own y-init for now.
            nlp.borrow_mut().get_starting_y(&mut y_c, &mut y_d);
            cap_constraint_multipliers(&mut y_c, self.constr_mult_init_max);
            cap_constraint_multipliers(&mut y_d, self.constr_mult_init_max);
        }

        // Step 4: bound multipliers — constant init.
        let mut z_l = DenseVectorSpace::new(n_zl).make_new_dense();
        let mut z_u = DenseVectorSpace::new(n_zu).make_new_dense();
        let mut v_l = DenseVectorSpace::new(n_vl).make_new_dense();
        let mut v_u = DenseVectorSpace::new(n_vu).make_new_dense();
        z_l.set(self.bound_mult_init_val);
        z_u.set(self.bound_mult_init_val);
        v_l.set(self.bound_mult_init_val);
        v_u.set(self.bound_mult_init_val);

        let iv = IteratesVector::new(
            Rc::new(x),
            Rc::new(s),
            Rc::new(y_c),
            Rc::new(y_d),
            Rc::new(z_l),
            Rc::new(z_u),
            Rc::new(v_l),
            Rc::new(v_u),
        );
        let n_x_dim = iv.x.dim();
        data.borrow_mut().set_curr(iv);

        // Step 5: least-square equality multipliers — port of
        // `IpDefaultIterateInitializer.cpp:285-341` /
        // `least_square_mults` (lines 669-743). Upstream always runs
        // this after the constant-init y_c/y_d=0, unless the full
        // `least_square_init_duals` path succeeded. Without it the
        // initial gradient-of-Lagrangian residual is computed against
        // y_c=y_d=0, blowing up `inf_du` on iter 0.
        if n_yc != n_x_dim
            && self.constr_mult_init_max > 0.0
            && (n_yc + n_yd) > 0
            && self.eq_mult_calculator.is_some()
        {
            let mut new_y_c = DenseVectorSpace::new(n_yc).make_new_dense();
            let mut new_y_d = DenseVectorSpace::new(n_yd).make_new_dense();
            let calc = self.eq_mult_calculator.as_mut().unwrap();
            let ok = calc.calculate_y_eq(data, cq, nlp, aug_solver, &mut new_y_c, &mut new_y_d);
            if !ok {
                // Solver failed → leave at zero (already the case).
                data.borrow_mut().append_info_string("y0");
            } else {
                let yinitnrm = new_y_c.amax().max(new_y_d.amax());
                if yinitnrm > self.constr_mult_init_max {
                    // Cap exceeded → upstream zeros them out
                    // (`IpDefaultIterateInitializer.cpp:723-727`).
                    data.borrow_mut().append_info_string("yc");
                } else {
                    // Accept LS estimates. Build a fresh iterate
                    // sharing the existing x/s/z/v Rcs and replacing
                    // y_c, y_d with the LS values.
                    let curr = data.borrow().curr.clone();
                    if let Some(c) = curr {
                        let new_iv = IteratesVector::new(
                            c.x.clone(),
                            c.s.clone(),
                            Rc::new(new_y_c),
                            Rc::new(new_y_d),
                            c.z_l.clone(),
                            c.z_u.clone(),
                            c.v_l.clone(),
                            c.v_u.clone(),
                        );
                        let mut d = data.borrow_mut();
                        d.set_curr(new_iv);
                        d.append_info_string("y");
                    }
                }
            }
        }

        true
    }
}

/// Apply [`DefaultIterateInitializer::push_to_interior`] to every
/// component of `x` using the lower/upper bound vectors expanded
/// through the `P_L`/`P_U` selection matrices. Bounds are packed
/// (lower-bound vector `x_l` has dim equal to the number of
/// lower-bounded components; `Px_L: n × n_lo` selects them).
pub(crate) fn push_x_into_interior(
    x: &mut DenseVector,
    px_l: &dyn pounce_linalg::Matrix,
    x_l: &dyn Vector,
    px_u: &dyn pounce_linalg::Matrix,
    x_u: &dyn Vector,
    bound_push: Number,
    bound_frac: Number,
) {
    // Use `dim()` (not `values().len()`): the iterate initializer is
    // called before any user `x0` has been written, so `x` is still in
    // its default homogeneous-zero state. `values()` carries a
    // `debug_assert!(!self.homogeneous)` and trips in debug builds on
    // clnlbeam.nl-class problems (n=59999, x_L/x_U packed). `values_mut()`
    // below materializes the dense buffer before the per-element write.
    let n = x.dim() as usize;
    // Expand x_l and x_u into full-length sentinel vectors:
    //   lower[i] = Some(x_l_packed[k]) if i is the k-th lower-bounded slot
    //   upper[i] = Some(x_u_packed[k]) similarly.
    let mut lower = vec![None; n];
    let mut upper = vec![None; n];
    expand_packed_into_dense(px_l, x_l, &mut lower);
    expand_packed_into_dense(px_u, x_u, &mut upper);

    let xs = x.values_mut();
    for (i, xi) in xs.iter_mut().enumerate() {
        *xi = DefaultIterateInitializer::push_to_interior(
            bound_push, bound_frac, *xi, lower[i], upper[i],
        );
    }
}

/// Apply `P` to a packed bound vector `b_packed` (dim `n_pack`) to
/// produce a sparse marking of `out` (dim `P.n_rows`). For each
/// `k = 0..n_pack`, `out[P_rows[k]] = Some(b_packed[k])`. Falls back
/// to a column-by-column probe via `mult_vector` if downcast to
/// `ExpansionMatrix` is unavailable.
fn expand_packed_into_dense(
    p: &dyn pounce_linalg::Matrix,
    b_packed: &dyn Vector,
    out: &mut [Option<Number>],
) {
    use pounce_linalg::expansion_matrix::ExpansionMatrix;
    let dim_packed = b_packed.dim() as usize;
    if dim_packed == 0 {
        return;
    }

    if let Some(em) = p.as_any().downcast_ref::<ExpansionMatrix>() {
        let rows = em.expanded_pos_indices();
        let Some(packed) = b_packed.as_any().downcast_ref::<DenseVector>() else {
            unreachable!("expansion-matrix bound vec must be DenseVector")
        };
        let vals = packed.values();
        for k in 0..dim_packed {
            let row = rows[k] as usize;
            out[row] = Some(vals[k]);
        }
    } else {
        // Generic fallback: probe via mult_vector with unit input
        // vectors. Quadratic; fine for tiny problems and tests.
        let n_full = out.len() as i32;
        let mut tmp = DenseVectorSpace::new(n_full).make_new_dense();
        for k in 0..dim_packed {
            let mut e_k = DenseVectorSpace::new(b_packed.dim()).make_new_dense();
            e_k.values_mut()[k] = 1.0;
            tmp.set(0.0);
            p.mult_vector(1.0, &e_k, 0.0, &mut tmp);
            // tmp is the k-th expansion column: a single 1.0 at the
            // expanded position. Read the value we want into the
            // matching slot.
            let Some(packed) = b_packed.as_any().downcast_ref::<DenseVector>() else {
                unreachable!("packed bound vec must be DenseVector")
            };
            for (i, &t) in tmp.values().iter().enumerate() {
                if t == 1.0 {
                    out[i] = Some(packed.values()[k]);
                }
            }
        }
    }
}

/// Clamp every component of `y` to `[-max, max]`. Mirrors the
/// upstream `constr_mult_init_max` cap.
fn cap_constraint_multipliers(y: &mut DenseVector, max: Number) {
    for v in y.values_mut() {
        if *v > max {
            *v = max;
        } else if *v < -max {
            *v = -max;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_point_left_alone() {
        // x=5 strictly inside [0, 10] with bound_push=1e-2 →
        // p_l = min(1e-2 * max(0,1), 1e-2 * 10) = 1e-2; same for p_u.
        // 5 is well inside [0.01, 9.9].
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 5.0, Some(0.0), Some(10.0));
        assert!((v - 5.0).abs() < 1e-15);
    }

    #[test]
    fn point_at_lower_bound_pushed_in() {
        // x=0 at the lower bound. Should become lo + p_l = 0.01.
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 0.0, Some(0.0), Some(10.0));
        assert!((v - 0.01).abs() < 1e-15);
    }

    #[test]
    fn point_at_upper_bound_pushed_in() {
        // x=10 at the upper bound. Should become hi - p_u = 9.9.
        let v =
            DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 10.0, Some(0.0), Some(10.0));
        assert!((v - 9.9).abs() < 1e-15);
    }

    #[test]
    fn point_below_lower_bound_clamped() {
        // x=-5 → lo + p_l = 0.01.
        let v =
            DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, -5.0, Some(0.0), Some(10.0));
        assert!((v - 0.01).abs() < 1e-15);
    }

    #[test]
    fn lower_only_pushed_by_max_abs() {
        // Lower-only with lo=-100: p_l = bound_push * max(|-100|, 1) = 1e-2 * 100 = 1.
        // x=-100 → -100 + 1 = -99.
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, -100.0, Some(-100.0), None);
        assert!((v - -99.0).abs() < 1e-13);
    }

    #[test]
    fn upper_only_pushed_by_max_abs() {
        // Upper-only with hi=50, x=50 → 50 - 1e-2 * 50 = 49.5.
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 50.0, None, Some(50.0));
        assert!((v - 49.5).abs() < 1e-13);
    }

    #[test]
    fn free_variable_unchanged() {
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 42.0, None, None);
        assert_eq!(v, 42.0);
    }

    #[test]
    fn narrow_interval_uses_bound_frac_branch() {
        // Tiny span [0, 1e-4]: p_l = min(1e-2 * 1, 1e-2 * 1e-4) = 1e-6.
        // x=0 → 0 + 1e-6 = 1e-6.
        let v = DefaultIterateInitializer::push_to_interior(1e-2, 1e-2, 0.0, Some(0.0), Some(1e-4));
        assert!((v - 1e-6).abs() < 1e-18);
    }
}
