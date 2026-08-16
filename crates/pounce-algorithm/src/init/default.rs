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
//! 3. Initialize `y_c`, `y_d` to zero, then revise them with the
//!    least-square estimate from
//!    [`crate::eq_mult::least_square::LeastSquareMults`] when an
//!    `EqMultCalculator` is wired and `constr_mult_init_max > 0`
//!    (an estimate above that cap is discarded, per upstream).
//! 4. Initialize `z_l`, `z_u`, `v_l`, `v_u` to `bound_mult_init_val`
//!    (component-wise) — i.e. `bound_mult_init_method = "constant"`,
//!    the only mode pounce implements. `"mu-based"` is registered for
//!    `ipopt.opt` compatibility and refused rather than silently
//!    served as `"constant"` (gh#604).

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

/// What the safeguarded least-square initializer did. Returned by
/// [`DefaultIterateInitializer::safeguarded_least_square_x`] and
/// readable after the solve through
/// [`crate::application::IpoptApplication::least_square_init_report`],
/// so a starting point that silently got worse is visible somewhere
/// other than the iteration count. Nothing prints it: it is a
/// programmatic accessor, not a log line.
#[derive(Debug, Clone, Default)]
pub struct LeastSquareInitReport {
    /// Nonlinear violation at the user's `x0`, after the interior push.
    pub violation_initial: Number,
    /// Nonlinear violation at the point actually handed to the
    /// algorithm. Equal to `violation_initial` when nothing was
    /// accepted.
    pub violation_final: Number,
    /// `alpha * ||x_ls - x0||_2` for the accepted trial; 0 if none was.
    pub step_norm: Number,
    /// Step fraction of the accepted trial; 0 if none was.
    pub alpha: Number,
    /// Backtracking trials whose true violation failed the test.
    pub rejected_trials: Index,
    /// Why the safeguard stopped.
    pub termination: &'static str,
}

pub struct DefaultIterateInitializer {
    pub bound_push: Number,
    pub bound_frac: Number,
    pub slack_bound_push: Number,
    pub slack_bound_frac: Number,
    pub constr_mult_init_max: Number,
    pub bound_mult_init_val: Number,
    /// `bound_mult_init_method`. Must be `"constant"` — upstream's
    /// `"mu-based"` is registered for `ipopt.opt` compatibility but not
    /// implemented, and `set_initial_iterates` returns `false` on it
    /// rather than serving a third, undocumented behaviour (gh#604).
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
    /// `least_square_init_primal_max_trials` — how many backtracking
    /// trials the safeguard in [`Self::safeguarded_least_square_x`] may
    /// take before it gives up and keeps the user's point. Each trial
    /// costs one constraint evaluation (`c` and `d`); none of them
    /// costs a Jacobian or a KKT solve, because the step direction is
    /// computed once and only its length changes.
    pub least_square_init_max_trials: Index,
    /// Armijo-style acceptance ratio for the safeguard: a trial at
    /// step fraction `alpha` is accepted when the true nonlinear
    /// violation satisfies `theta(alpha) <= (1 - eta*alpha) * theta_0`,
    /// i.e. when the *actual* feasibility reduction is at least `eta`
    /// times the reduction the linearization *predicted*.
    pub least_square_init_accept_ratio: Number,
    /// Diagnostics from the most recent safeguarded least-square
    /// initialization: initial/final violation, accepted step norm,
    /// rejected trial count, termination reason. `None` when the step
    /// was never attempted.
    pub last_least_square_report: Option<LeastSquareInitReport>,
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
            least_square_init_max_trials: 4,
            least_square_init_accept_ratio: 1e-2,
            last_least_square_report: None,
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

    /// Stage `x_cand` as the current iterate and return the true
    /// nonlinear constraint violation there.
    ///
    /// The merit is `curr_unscaled_nlp_constraint_violation_max()` —
    /// `max(||c(x)||_inf, ||max(d_l - d(x), d(x) - d_u, 0)||_inf)` in
    /// unscaled NLP units. It is the same quantity the CLI reports as
    /// the model's constraint violation, so "the initializer improved
    /// feasibility" means the number a user can read improved.
    ///
    /// Costs one `c`/`d` evaluation. No Jacobian, no KKT solve.
    fn violation_at(
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        template: &IteratesVector,
        x_cand: &dyn Vector,
    ) -> Number {
        let mut x_stage = DenseVectorSpace::new(x_cand.dim()).make_new_dense();
        x_stage.copy(x_cand);
        let staged = template.with_x(Rc::new(x_stage));
        data.borrow_mut().set_curr(staged);
        cq.borrow().curr_unscaled_nlp_constraint_violation_max()
    }

    /// Safeguarded least-square normal step.
    ///
    /// `calculate_least_square_primals` returns the minimum-norm
    /// solution of the *linearized* constraints. That is a local model
    /// step, not automatically a better NLP starting point: where the
    /// Jacobian is small relative to the residual the linearization
    /// asks for a huge correction, and the true nonlinear violation at
    /// the far end can be orders of magnitude worse than where it
    /// started. Accepting it unconditionally (which is what upstream
    /// `IpDefaultIterateInitializer.cpp:200-222` does, and what pounce
    /// did through 0.10.0) hands the algorithm a worse starting point
    /// than the user supplied.
    ///
    /// So: compute the direction once, then walk it back.
    ///
    /// * Trial `k` uses `alpha = 2^-k` for `k` in `0..max_trials`.
    /// * Every candidate is pushed into the bound interior *before*
    ///   its violation is measured, so the accepted merit is the merit
    ///   of the point the algorithm will actually start from, and
    ///   bound interiority is preserved by construction.
    /// * A trial is accepted when
    ///   `theta(alpha) <= (1 - eta*alpha) * theta_0`. The linear model
    ///   predicts `theta -> 0` at `alpha = 1`, so the predicted
    ///   reduction at `alpha` is `alpha * theta_0` and this test is
    ///   exactly "actual reduction is at least `eta` times predicted".
    /// * The first accepted trial wins (they are tried longest-first,
    ///   so that is also the best available reduction on this ray).
    /// * If no trial is accepted the user's `x` is returned unchanged.
    ///
    /// Returns `(accepted_x, diagnostics)`.
    #[allow(clippy::too_many_arguments)]
    fn safeguarded_least_square_x(
        &self,
        data: &IpoptDataHandle,
        cq: &IpoptCqHandle,
        nlp: &Rc<RefCell<dyn IpoptNlp>>,
        aug_solver: &mut dyn AugSystemSolver,
        template: &IteratesVector,
        x0: &dyn Vector,
        n_x: Index,
    ) -> (Option<Box<dyn Vector>>, LeastSquareInitReport) {
        let mut report = LeastSquareInitReport::default();

        // theta_0 at the user's point, measured after the interior
        // push so it is comparable with every trial below.
        let mut x_base = DenseVectorSpace::new(n_x).make_new_dense();
        x_base.copy(x0);
        self.push_into_bounds(nlp, &mut x_base);
        let theta_0 = Self::violation_at(data, cq, template, &x_base);
        report.violation_initial = theta_0;
        report.violation_final = theta_0;

        if !theta_0.is_finite() {
            report.termination = "x0 violation is not finite";
            return (None, report);
        }
        if theta_0 == 0.0 {
            report.termination = "x0 already feasible";
            return (None, report);
        }

        // The direction. Computed at the user's point, so re-stage it
        // first: `calculate_least_square_primals` linearizes at
        // whatever `data.curr` holds.
        let staged = template.with_x({
            let mut xs = DenseVectorSpace::new(n_x).make_new_dense();
            xs.copy(x0);
            Rc::new(xs)
        });
        data.borrow_mut().set_curr(staged);
        let x_ls = match self.calculate_least_square_primals(cq, nlp, aug_solver, n_x) {
            Some(v) => v,
            None => {
                report.termination = "augmented system solve failed";
                return (None, report);
            }
        };

        // d = x_ls - x0, formed once and reused at every trial.
        let mut dir = DenseVectorSpace::new(n_x).make_new_dense();
        dir.copy(&*x_ls);
        dir.axpy(-1.0, x0);
        let dir_norm = dir.nrm2();
        if !dir_norm.is_finite() {
            report.termination = "least-square step is not finite";
            return (None, report);
        }

        let mut alpha = 1.0;
        for _ in 0..self.least_square_init_max_trials.max(1) {
            let mut cand = DenseVectorSpace::new(n_x).make_new_dense();
            if alpha == 1.0 {
                // Use `x_ls` itself rather than `x0 + 1.0*(x_ls - x0)`.
                // The two differ in the last bit, and that is enough to
                // move a borderline model by an iteration — so the
                // full-length trial stays bit-identical to what the
                // unsafeguarded path produced, and the only trajectory
                // change is on the models where the step is actually
                // rejected.
                cand.copy(&*x_ls);
            } else {
                cand.copy(x0);
                cand.axpy(alpha, &dir);
            }
            self.push_into_bounds(nlp, &mut cand);

            let theta = Self::violation_at(data, cq, template, &cand);
            let predicted = alpha * theta_0;
            let actual = theta_0 - theta;
            if theta.is_finite()
                && actual >= self.least_square_init_accept_ratio * predicted
                && theta < theta_0
            {
                report.violation_final = theta;
                report.step_norm = alpha * dir_norm;
                report.alpha = alpha;
                report.termination = "accepted";
                return (Some(Box::new(cand)), report);
            }
            report.rejected_trials += 1;
            alpha *= 0.5;
        }

        report.termination = "no trial improved the nonlinear violation";
        (None, report)
    }

    /// Push `x` into the interior of `[x_l, x_u]` with this
    /// initializer's `bound_push` / `bound_frac`. Split out so the
    /// safeguard can measure candidates at the point the algorithm
    /// would actually use.
    fn push_into_bounds(&self, nlp: &Rc<RefCell<dyn IpoptNlp>>, x: &mut DenseVector) {
        let nlp_ref = nlp.borrow();
        push_x_into_interior(
            x,
            &*nlp_ref.px_l(),
            nlp_ref.x_l(),
            &*nlp_ref.px_u(),
            nlp_ref.x_u(),
            self.bound_push,
            self.bound_frac,
        );
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
    fn least_square_report(&self) -> Option<LeastSquareInitReport> {
        self.last_least_square_report.clone()
    }

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
            data.borrow_mut().set_curr(stage_iv.clone());

            // The linearized least-squares point is only accepted when
            // it actually reduces the *true* nonlinear violation; see
            // `safeguarded_least_square_x`. Rejecting it leaves `x` as
            // the user gave it, which is the pre-0.11 behaviour minus
            // the cases where the linearization sent the starting
            // point somewhere worse.
            let (accepted, report) =
                self.safeguarded_least_square_x(data, cq, nlp, aug_solver, &stage_iv, &x, n_x);
            if let Some(x_new) = accepted {
                x.copy(&*x_new);
            }
            self.last_least_square_report = Some(report);
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

        // `bound_mult_init_method` — pounce implements `constant` only
        // (gh#604). The refusal that a caller actually sees is raised at
        // the application layer, before any work
        // (`unimplemented_options::UNIMPLEMENTED_VALUES`); this is the
        // backstop for a caller who builds the initializer directly.
        //
        // It used to fall through to `nlp.get_starting_y` here, which is
        // neither of the documented modes — an unsupported value silently
        // bought a *third* behaviour. Failing is the honest answer.
        if !self.bound_mult_init_method.eq_ignore_ascii_case("constant") {
            tracing::error!(
                target: "pounce::algorithm",
                method = %self.bound_mult_init_method,
                "pounce: bound_mult_init_method must be \"constant\"; \
                 \"mu-based\" is registered for ipopt.opt compatibility but \
                 not implemented (gh#604)."
            );
            return false;
        }

        // Step 3: y_c, y_d initial guesses. `constant` mode leaves
        // them at zero (the algorithm refines on the first KKT solve),
        // and the least-square step below revises them when an
        // `EqMultCalculator` is wired.
        let mut y_c = DenseVectorSpace::new(n_yc).make_new_dense();
        let mut y_d = DenseVectorSpace::new(n_yd).make_new_dense();
        // Materialize as homogeneous-zero so callers' asum / values
        // probes don't trip the `initialized` debug-assert.
        y_c.set(0.0);
        y_d.set(0.0);

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

/// Behavior tests for the cold-start options (gh#604).
///
/// The wiring tests in `tests/init_options_wiring.rs` prove each option
/// reaches [`crate::alg_builder::InitOptions`]; these prove the value
/// then changes the iterate the initializer produces. An option that is
/// registered, read, threaded onto the strategy and then ignored is the
/// same silent no-op with more steps.
#[cfg(test)]
mod option_behavior {
    use super::*;
    use crate::ipopt_cq::IpoptCalculatedQuantities;
    use crate::ipopt_data::IpoptData;
    use pounce_linalg::{IdentityMatrix, Matrix, SymMatrix};
    use pounce_linsol::status::ESymSolverStatus;

    const N_X: Index = 2;
    const N_S: Index = 1;
    const N_C: Index = 1;

    /// `x0 = [0, 10]` sits on both bounds of `[0, 10]^2`, and
    /// `d(x) = -5` sits on the lower inequality bound, so every push
    /// knob has something visible to move.
    struct StubNlp {
        x_l: DenseVector,
        x_u: DenseVector,
        d_l: DenseVector,
        d_u: DenseVector,
        p2: Rc<dyn Matrix>,
        p1: Rc<dyn Matrix>,
    }

    impl StubNlp {
        fn new() -> Self {
            let s2 = DenseVectorSpace::new(N_X);
            let mut x_l = DenseVector::new(Rc::clone(&s2));
            x_l.set_values(&[0.0, 0.0]);
            let mut x_u = DenseVector::new(Rc::clone(&s2));
            x_u.set_values(&[10.0, 10.0]);
            let s1 = DenseVectorSpace::new(N_S);
            let mut d_l = DenseVector::new(Rc::clone(&s1));
            d_l.set_values(&[-5.0]);
            let mut d_u = DenseVector::new(Rc::clone(&s1));
            d_u.set_values(&[5.0]);
            Self {
                x_l,
                x_u,
                d_l,
                d_u,
                p2: Rc::new(IdentityMatrix::new(N_X)),
                p1: Rc::new(IdentityMatrix::new(N_S)),
            }
        }
    }

    impl crate::ipopt_nlp::Nlp for StubNlp {
        fn n(&self) -> Index {
            N_X
        }
        fn m_eq(&self) -> Index {
            N_C
        }
        fn m_ineq(&self) -> Index {
            N_S
        }
        fn eval_f(&mut self, _x: &dyn Vector) -> Number {
            0.0
        }
        fn eval_grad_f(&mut self, _x: &dyn Vector, g: &mut dyn Vector) {
            g.set(0.0);
        }
        /// `c(x) = x0 + x1 - 4`, which [`FixedAugSolver`]'s `x_ls =
        /// [1, 3]` satisfies exactly.
        ///
        /// This was a constant `1.0` when gh#604 wrote these tests, and
        /// a constant will not do since gh#605: the least-square step is
        /// now taken only when it reduces the *true* nonlinear
        /// violation, and against a constant `c` no step ever can, so
        /// `least_square_init_primal` would be correctly declined and
        /// the option untestable here. An `x`-dependent row also makes
        /// the stub honest — `x_ls` is supposed to be the point that
        /// solves the linearized constraints, and now it is one.
        fn eval_c(&mut self, x: &dyn Vector, c: &mut dyn Vector) {
            let v = x
                .as_any()
                .downcast_ref::<DenseVector>()
                .expect("dense x")
                .expanded_values();
            c.set(v[0] + v[1] - 4.0);
        }
        fn eval_d(&mut self, _x: &dyn Vector, d: &mut dyn Vector) {
            d.set(-5.0);
        }
        fn eval_jac_c(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            Rc::new(IdentityMatrix::new(N_X))
        }
        fn eval_jac_d(&mut self, _x: &dyn Vector) -> Rc<dyn Matrix> {
            Rc::new(IdentityMatrix::new(N_X))
        }
        fn eval_h(
            &mut self,
            _x: &dyn Vector,
            _obj_factor: Number,
            _y_c: &dyn Vector,
            _y_d: &dyn Vector,
        ) -> Rc<dyn SymMatrix> {
            let s = pounce_linalg::DenseSymMatrixSpace::new(N_X);
            Rc::new(pounce_linalg::DenseSymMatrix::new(s))
        }
    }

    impl crate::ipopt_nlp::IpoptNlp for StubNlp {
        fn x_l(&self) -> &dyn Vector {
            &self.x_l
        }
        fn x_u(&self) -> &dyn Vector {
            &self.x_u
        }
        fn d_l(&self) -> &dyn Vector {
            &self.d_l
        }
        fn d_u(&self) -> &dyn Vector {
            &self.d_u
        }
        fn px_l(&self) -> Rc<dyn Matrix> {
            self.p2.clone()
        }
        fn px_u(&self) -> Rc<dyn Matrix> {
            self.p2.clone()
        }
        fn pd_l(&self) -> Rc<dyn Matrix> {
            self.p1.clone()
        }
        fn pd_u(&self) -> Rc<dyn Matrix> {
            self.p1.clone()
        }
        fn get_starting_x(&mut self, x: &mut dyn Vector) -> bool {
            let dense = x
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .expect("dense x");
            dense.set_values(&[0.0, 10.0]);
            true
        }
    }

    /// Fails every solve, so the least-square primal step is a no-op.
    /// Nothing else in the tested paths touches the aug solver.
    struct FailingAugSolver;
    /// Returns a fixed `sol_x`, so `least_square_init_primal=yes` lands
    /// on a point the bound push alone could never produce.
    struct FixedAugSolver;

    macro_rules! aug_solver_boilerplate {
        () => {
            fn provides_inertia(&self) -> bool {
                false
            }
            fn number_of_neg_evals(&self) -> Index {
                0
            }
            fn increase_quality(&mut self) -> bool {
                false
            }
            fn last_solve_status(&self) -> ESymSolverStatus {
                ESymSolverStatus::Success
            }
        };
    }

    impl AugSystemSolver for FailingAugSolver {
        aug_solver_boilerplate!();
        fn solve(
            &mut self,
            _coeffs: &AugSysCoeffs<'_>,
            _rhs: &AugSysRhs<'_>,
            _sol: &mut AugSysSol<'_>,
            _check_neg_evals: bool,
            _num_neg_evals: Index,
        ) -> ESymSolverStatus {
            ESymSolverStatus::Singular
        }
    }

    impl AugSystemSolver for FixedAugSolver {
        aug_solver_boilerplate!();
        fn solve(
            &mut self,
            _coeffs: &AugSysCoeffs<'_>,
            _rhs: &AugSysRhs<'_>,
            sol: &mut AugSysSol<'_>,
            _check_neg_evals: bool,
            _num_neg_evals: Index,
        ) -> ESymSolverStatus {
            // The initializer negates this, so `x_ls = [1, 3]`.
            sol.sol_x
                .as_any_mut()
                .downcast_mut::<DenseVector>()
                .expect("dense sol_x")
                .set_values(&[-1.0, -3.0]);
            sol.sol_s.set(0.0);
            sol.sol_c.set(0.0);
            sol.sol_d.set(0.0);
            ESymSolverStatus::Success
        }
    }

    /// Hands back a fixed equality-multiplier estimate so the
    /// `constr_mult_init_max` cap has something to accept or discard.
    struct FixedEqMults(Number);
    impl EqMultCalculator for FixedEqMults {
        fn calculate_y_eq(
            &mut self,
            _data: &IpoptDataHandle,
            _cq: &IpoptCqHandle,
            _nlp: &Rc<RefCell<dyn IpoptNlp>>,
            _aug_solver: &mut dyn AugSystemSolver,
            y_c: &mut dyn Vector,
            y_d: &mut dyn Vector,
        ) -> bool {
            y_c.set(self.0);
            y_d.set(self.0);
            true
        }
    }

    fn zeros(n: Index) -> Rc<DenseVector> {
        let mut v = DenseVectorSpace::new(n).make_new_dense();
        v.set(0.0);
        Rc::new(v)
    }

    /// A data/cq pair over [`StubNlp`] with a correctly-shaped `curr`
    /// installed — the initializer reads the block dimensions off it.
    fn fixture() -> (IpoptDataHandle, IpoptCqHandle, Rc<RefCell<dyn IpoptNlp>>) {
        let nlp: Rc<RefCell<dyn IpoptNlp>> = Rc::new(RefCell::new(StubNlp::new()));
        let data: IpoptDataHandle = Rc::new(RefCell::new(IpoptData::new()));
        let cq: IpoptCqHandle = Rc::new(RefCell::new(IpoptCalculatedQuantities::new(
            Rc::clone(&data),
            Rc::clone(&nlp),
        )));
        let template = IteratesVector::new(
            zeros(N_X),
            zeros(N_S),
            zeros(N_C),
            zeros(N_S),
            zeros(N_X),
            zeros(N_X),
            zeros(N_S),
            zeros(N_S),
        );
        data.borrow_mut().set_curr(template);
        (data, cq, nlp)
    }

    /// Run the initializer and return the installed `curr`.
    fn run(init: &mut DefaultIterateInitializer, aug: &mut dyn AugSystemSolver) -> IteratesVector {
        let (data, cq, nlp) = fixture();
        assert!(
            init.set_initial_iterates(&data, &cq, &nlp, aug),
            "initializer should succeed"
        );
        data.borrow().curr.clone().expect("curr installed")
    }

    /// `expanded_values` rather than `values`: a block the initializer
    /// filled with `set` stays in the homogeneous representation, which
    /// `values` refuses to hand out.
    fn values_of(v: &Rc<dyn Vector>) -> Vec<Number> {
        v.as_any()
            .downcast_ref::<DenseVector>()
            .expect("dense")
            .expanded_values()
    }
    fn x_of(iv: &IteratesVector) -> Vec<Number> {
        values_of(&iv.x)
    }

    /// `bound_push` moves `x0` off its lower bound by
    /// `min(bound_push * max(|lo|, 1), bound_frac * span)`.
    #[test]
    fn bound_push_changes_the_initial_primal() {
        let mut aug = FailingAugSolver;

        let mut d = DefaultIterateInitializer::new();
        let base = x_of(&run(&mut d, &mut aug));
        assert!((base[0] - 1e-2).abs() < 1e-15, "default: {base:?}");

        let mut pushed = DefaultIterateInitializer {
            bound_push: 5e-2,
            ..DefaultIterateInitializer::new()
        };
        let moved = x_of(&run(&mut pushed, &mut aug));
        assert!(
            (moved[0] - 5e-2).abs() < 1e-15,
            "bound_push=5e-2: {moved:?}"
        );
        assert_ne!(base[0], moved[0]);
    }

    /// `bound_frac` is the other arm of the same min, and it binds when
    /// the interval is narrow relative to `bound_push`.
    #[test]
    fn bound_frac_changes_the_initial_primal() {
        let mut aug = FailingAugSolver;
        let mut init = DefaultIterateInitializer {
            bound_frac: 5e-4,
            ..DefaultIterateInitializer::new()
        };
        // span = 10, so the frac arm gives 5e-3 < bound_push's 1e-2.
        let x = x_of(&run(&mut init, &mut aug));
        assert!((x[0] - 5e-3).abs() < 1e-15, "bound_frac=5e-4: {x:?}");
    }

    /// The slack knobs do the same job for `s`, which starts on the
    /// lower inequality bound `-5`.
    #[test]
    fn slack_bound_push_and_frac_change_the_initial_slack() {
        let mut aug = FailingAugSolver;

        let mut d = DefaultIterateInitializer::new();
        // p_l = min(1e-2 * max(|-5|, 1), 1e-2 * 10) = 5e-2.
        let base = values_of(&run(&mut d, &mut aug).s);
        assert!((base[0] - -4.95).abs() < 1e-14, "default: {base:?}");

        let mut pushed = DefaultIterateInitializer {
            slack_bound_push: 1e-1,
            ..DefaultIterateInitializer::new()
        };
        // p_l = min(1e-1 * 5, 1e-2 * 10) = 1e-1 — the frac arm still binds.
        let s = values_of(&run(&mut pushed, &mut aug).s);
        assert!((s[0] - -4.9).abs() < 1e-14, "slack_bound_push=1e-1: {s:?}");

        let mut fracced = DefaultIterateInitializer {
            slack_bound_frac: 1e-3,
            ..DefaultIterateInitializer::new()
        };
        // p_l = min(1e-2 * 5, 1e-3 * 10) = 1e-2.
        let s = values_of(&run(&mut fracced, &mut aug).s);
        assert!((s[0] - -4.99).abs() < 1e-14, "slack_bound_frac=1e-3: {s:?}");
    }

    /// `bound_mult_init_val` is the value every bound multiplier takes.
    #[test]
    fn bound_mult_init_val_changes_the_bound_multipliers() {
        let mut aug = FailingAugSolver;

        let base = run(&mut DefaultIterateInitializer::new(), &mut aug);
        assert_eq!(values_of(&base.z_l), vec![1.0, 1.0]);
        assert_eq!(values_of(&base.v_u), vec![1.0]);

        let mut init = DefaultIterateInitializer {
            bound_mult_init_val: 7.5,
            ..DefaultIterateInitializer::new()
        };
        let iv = run(&mut init, &mut aug);
        assert_eq!(values_of(&iv.z_l), vec![7.5, 7.5]);
        assert_eq!(values_of(&iv.z_u), vec![7.5, 7.5]);
        assert_eq!(values_of(&iv.v_l), vec![7.5]);
        assert_eq!(values_of(&iv.v_u), vec![7.5]);
    }

    /// `constr_mult_init_max` caps the least-square equality-multiplier
    /// estimate: above the cap upstream discards it and leaves zeros.
    #[test]
    fn constr_mult_init_max_gates_the_equality_multipliers() {
        let mut aug = FailingAugSolver;

        let mut accepted = DefaultIterateInitializer {
            constr_mult_init_max: 1e3,
            ..DefaultIterateInitializer::with_eq_mult_calculator(Box::new(FixedEqMults(2.0)))
        };
        assert_eq!(values_of(&run(&mut accepted, &mut aug).y_c), vec![2.0]);

        let mut capped = DefaultIterateInitializer {
            constr_mult_init_max: 1.0,
            ..DefaultIterateInitializer::with_eq_mult_calculator(Box::new(FixedEqMults(2.0)))
        };
        assert_eq!(
            values_of(&run(&mut capped, &mut aug).y_c),
            vec![0.0],
            "an estimate above the cap is discarded, not clamped"
        );

        // 0 switches the least-square step off entirely.
        let mut off = DefaultIterateInitializer {
            constr_mult_init_max: 0.0,
            ..DefaultIterateInitializer::with_eq_mult_calculator(Box::new(FixedEqMults(2.0)))
        };
        assert_eq!(values_of(&run(&mut off, &mut aug).y_c), vec![0.0]);
    }

    /// `least_square_init_primal=yes` replaces the user's `x0` with the
    /// solution of the linearized-constraint system.
    #[test]
    fn least_square_init_primal_replaces_the_starting_point() {
        let mut fixed = FixedAugSolver;

        let mut off = DefaultIterateInitializer::new();
        let base = x_of(&run(&mut off, &mut fixed));
        assert!((base[0] - 1e-2).abs() < 1e-15, "user x0, pushed: {base:?}");

        let mut on = DefaultIterateInitializer {
            least_square_init_primal: true,
            ..DefaultIterateInitializer::new()
        };
        let ls = x_of(&run(&mut on, &mut fixed));
        assert_eq!(
            ls,
            vec![1.0, 3.0],
            "the least-square point, already interior"
        );
    }

    /// The one mode pounce implements runs; anything else fails rather
    /// than quietly running a third behaviour (gh#604). The refusal a
    /// caller actually sees is raised earlier, at the application layer.
    #[test]
    fn an_unsupported_bound_mult_init_method_fails_instead_of_falling_back() {
        let (data, cq, nlp) = fixture();
        let mut aug = FailingAugSolver;
        let mut init = DefaultIterateInitializer {
            bound_mult_init_method: "mu-based".into(),
            ..DefaultIterateInitializer::new()
        };
        assert!(
            !init.set_initial_iterates(&data, &cq, &nlp, &mut aug),
            "`mu-based` is not implemented and must not be served as `constant`"
        );

        // Spelling is the only thing that varies — case does not.
        let (data, cq, nlp) = fixture();
        let mut cased = DefaultIterateInitializer {
            bound_mult_init_method: "CONSTANT".into(),
            ..DefaultIterateInitializer::new()
        };
        assert!(cased.set_initial_iterates(&data, &cq, &nlp, &mut aug));
    }
}
