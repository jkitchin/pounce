//! Convex QP problem data in standard form.
//!
//! ```text
//! minimize    ½ xᵀP x + cᵀx
//! subject to  A x = b          (equality,   m_eq rows)
//!             G x ≤ h          (inequality, m_ineq rows)
//! ```
//!
//! `x` is free; variable bounds are expressed as rows of `G`. `P` must
//! be symmetric positive semidefinite (convexity); it is supplied as its
//! **lower triangle** in triplet form. `A` and `G` are general sparse
//! triplets. This is the form the IPM in [`crate::ipm`] consumes, and
//! the form the `.nl` → QP extraction (Phase 2 dispatch) will target.

/// A sparse matrix entry `(row, col, val)`, 0-based.
#[derive(Debug, Clone, Copy)]
pub struct Triplet {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

impl Triplet {
    pub fn new(row: usize, col: usize, val: f64) -> Self {
        Triplet { row, col, val }
    }
}

/// Convex QP in the standard form documented at the module level.
#[derive(Debug, Clone)]
pub struct QpProblem {
    /// Number of decision variables.
    pub n: usize,
    /// Lower triangle (row ≥ col) of the symmetric PSD Hessian `P`.
    pub p_lower: Vec<Triplet>,
    /// Linear objective coefficient `c` (length `n`).
    pub c: Vec<f64>,
    /// Equality matrix `A` (m_eq × n), full triplets.
    pub a: Vec<Triplet>,
    /// Equality right-hand side `b` (length m_eq).
    pub b: Vec<f64>,
    /// Inequality matrix `G` (m_ineq × n), full triplets.
    pub g: Vec<Triplet>,
    /// Inequality right-hand side `h` (length m_ineq).
    pub h: Vec<f64>,
    /// Per-variable lower bounds `lb ≤ x`. Either empty (all `-∞`) or
    /// length `n`. Use [`NEG_INF`] for an unbounded entry. Bounds are a
    /// first-class part of the problem (not encoded as `G` rows), so
    /// presolve can reason about variable boxes; the solver expands the
    /// finite ones into internal inequality rows.
    pub lb: Vec<f64>,
    /// Per-variable upper bounds `x ≤ ub`. Either empty (all `+∞`) or
    /// length `n`. Use [`POS_INF`] for an unbounded entry.
    pub ub: Vec<f64>,
}

/// Sentinel for an absent lower bound (`-∞`). Anything `≤ -BOUND_INF` is
/// treated as no bound.
pub const NEG_INF: f64 = f64::NEG_INFINITY;
/// Sentinel for an absent upper bound (`+∞`). Anything `≥ BOUND_INF` is
/// treated as no bound.
pub const POS_INF: f64 = f64::INFINITY;
/// Magnitude past which a bound is considered infinite.
pub(crate) const BOUND_INF: f64 = 1e20;

impl QpProblem {
    pub fn m_eq(&self) -> usize {
        self.b.len()
    }

    pub fn m_ineq(&self) -> usize {
        self.h.len()
    }

    /// Lower bound of variable `i` (`-∞` when `lb` is empty).
    pub fn lb_of(&self, i: usize) -> f64 {
        self.lb.get(i).copied().unwrap_or(NEG_INF)
    }

    /// Upper bound of variable `i` (`+∞` when `ub` is empty).
    pub fn ub_of(&self, i: usize) -> f64 {
        self.ub.get(i).copied().unwrap_or(POS_INF)
    }

    /// Whether the problem carries any finite variable bound.
    pub fn has_bounds(&self) -> bool {
        self.lb.iter().any(|&v| v > -BOUND_INF) || self.ub.iter().any(|&v| v < BOUND_INF)
    }

    /// Public `y += P x` (full symmetric product from the stored lower
    /// triangle). Exposed so external callers — e.g. a TNLP adapter
    /// reusing the same problem data — can evaluate the objective
    /// gradient consistently with the solver.
    pub fn p_mul_add_pub(&self, x: &[f64], y: &mut [f64]) {
        self.p_mul_add(x, y);
    }

    /// Public `y += A x`.
    pub fn a_mul_add_pub(&self, x: &[f64], y: &mut [f64]) {
        self.a_mul_add(x, y);
    }

    /// `y += P x` using the stored lower triangle (mirrors the implicit
    /// upper triangle for off-diagonal entries).
    pub(crate) fn p_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.p_lower {
            y[t.row] += t.val * x[t.col];
            if t.row != t.col {
                y[t.col] += t.val * x[t.row];
            }
        }
    }

    /// `y += A x`.
    pub(crate) fn a_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.a {
            y[t.row] += t.val * x[t.col];
        }
    }

    /// `y += Aᵀ v`.
    pub(crate) fn at_mul_add(&self, v: &[f64], y: &mut [f64]) {
        for t in &self.a {
            y[t.col] += t.val * v[t.row];
        }
    }

    /// `y += G x`.
    pub(crate) fn g_mul_add(&self, x: &[f64], y: &mut [f64]) {
        for t in &self.g {
            y[t.row] += t.val * x[t.col];
        }
    }

    /// `y += Gᵀ v`.
    pub(crate) fn gt_mul_add(&self, v: &[f64], y: &mut [f64]) {
        for t in &self.g {
            y[t.col] += t.val * v[t.row];
        }
    }

    /// Public `y += A x` (alias of [`Self::a_mul_add`]).
    pub fn a_mul(&self, x: &[f64], y: &mut [f64]) {
        self.a_mul_add(x, y);
    }

    /// Public `y += G x` (alias of [`Self::g_mul_add`]).
    pub fn g_mul(&self, x: &[f64], y: &mut [f64]) {
        self.g_mul_add(x, y);
    }

    /// Public `y += Aᵀ v` (alias of [`Self::at_mul_add`]).
    pub fn at_mul(&self, v: &[f64], y: &mut [f64]) {
        self.at_mul_add(v, y);
    }

    /// Public `y += Gᵀ v` (alias of [`Self::gt_mul_add`]).
    pub fn gt_mul(&self, v: &[f64], y: &mut [f64]) {
        self.gt_mul_add(v, y);
    }

    /// Public `y += P x` (alias of [`Self::p_mul_add`]).
    pub fn p_mul(&self, x: &[f64], y: &mut [f64]) {
        self.p_mul_add(x, y);
    }
}

/// Termination status of an IPM solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// Converged: KKT residuals and duality gap below tolerance.
    Optimal,
    /// Primal infeasible: no `x` satisfies `Ax = b, Gx ≤ h`. A Farkas
    /// certificate `(y, z ≥ 0)` with `Aᵀy + Gᵀz ≈ 0` and `bᵀy + hᵀz < 0`
    /// was detected and verified.
    PrimalInfeasible,
    /// Dual infeasible / unbounded below: a recession direction `d` with
    /// `Pd ≈ 0, Ad = 0, Gd ≤ 0, cᵀd < 0` was detected and verified.
    DualInfeasible,
    /// Iteration limit reached before convergence.
    IterationLimit,
    /// The KKT factorization failed (e.g. structurally singular system).
    NumericalFailure,
}

/// Result of an IPM solve: the primal/dual solution and status.
#[derive(Debug, Clone)]
pub struct QpSolution {
    pub status: QpStatus,
    /// Primal solution `x` (length `n`).
    pub x: Vec<f64>,
    /// Equality multipliers `y` (length m_eq).
    pub y: Vec<f64>,
    /// Inequality multipliers `z ≥ 0` (length m_ineq).
    pub z: Vec<f64>,
    /// Lower-bound multipliers `z_lb ≥ 0` for `lb ≤ x` (length `n`; zero
    /// where there is no finite lower bound or it is inactive).
    pub z_lb: Vec<f64>,
    /// Upper-bound multipliers `z_ub ≥ 0` for `x ≤ ub` (length `n`).
    pub z_ub: Vec<f64>,
    /// Objective value `½ xᵀP x + cᵀx`.
    pub obj: f64,
    /// Iterations taken.
    pub iters: usize,
    /// Per-iteration convergence trace, populated only when
    /// [`crate::QpOptions::collect_iterates`] was set (otherwise empty, with
    /// no per-solve overhead). Each entry is one interior-point iteration.
    pub iterates: Vec<QpIterate>,
}

/// One interior-point iteration's convergence record — the per-iteration data
/// a solve report or benchmark harness wants (residuals, the duality measure,
/// and the step lengths). Collected by the convex IPM when
/// [`crate::QpOptions::collect_iterates`] is set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpIterate {
    /// Iteration index (0-based).
    pub iter: usize,
    /// Objective `½ xᵀP x + cᵀx` at the start of this iteration.
    pub objective: f64,
    /// Primal infeasibility `max(‖Ax − b‖∞, ‖(Gx + s − h)‖∞)`.
    pub primal_infeasibility: f64,
    /// Dual infeasibility `‖Px + c + Aᵀy + Gᵀz‖∞`.
    pub dual_infeasibility: f64,
    /// Duality measure `μ = ⟨s, z⟩ / degree`.
    pub mu: f64,
    /// Primal step length taken this iteration.
    pub alpha_primal: f64,
    /// Dual step length taken this iteration.
    pub alpha_dual: f64,
}

/// Final KKT residuals of a [`QpSolution`] with respect to its [`QpProblem`]
/// — the convergence quantities a caller (e.g. a solve report or benchmark
/// harness) needs but that aren't otherwise carried on the solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpResiduals {
    /// Primal infeasibility: `max(|Ax − b|, max(0, Gx − h), bound violations)`.
    pub primal_infeasibility: f64,
    /// Dual infeasibility (stationarity):
    /// `‖Px + c + Aᵀy + Gᵀz − z_lb + z_ub‖∞`.
    pub dual_infeasibility: f64,
    /// Complementarity: `max |zᵢ · slackᵢ|` over inequalities and finite bounds.
    pub complementarity: f64,
}

impl QpResiduals {
    /// Overall KKT error — the max of the three components.
    pub fn kkt_error(&self) -> f64 {
        self.primal_infeasibility
            .max(self.dual_infeasibility)
            .max(self.complementarity)
    }
}

impl QpSolution {
    /// Recompute the final KKT residuals of this solution against `prob`.
    ///
    /// Uses the convex solver's standard-form conventions —
    /// `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ≤ h, lb ≤ x ≤ ub`, with equality dual
    /// `y`, inequality dual `z ≥ 0`, and bound duals `z_lb, z_ub ≥ 0`. The
    /// stationarity residual is `∇ₓL = Px + c + Aᵀy + Gᵀz − z_lb + z_ub`, the
    /// `−z_lb + z_ub` matching how variable bounds expand into `G`-rows and
    /// split back into the bound multipliers.
    pub fn kkt_residuals(&self, prob: &QpProblem) -> QpResiduals {
        let n = prob.n;

        // Dual infeasibility (stationarity).
        let mut r = vec![0.0; n];
        prob.p_mul(&self.x, &mut r);
        for (((ri, &ci), &lb), &ub) in r.iter_mut().zip(&prob.c).zip(&self.z_lb).zip(&self.z_ub) {
            *ri += ci - lb + ub;
        }
        prob.at_mul(&self.y, &mut r);
        prob.gt_mul(&self.z, &mut r);
        let dual_infeasibility = r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

        // Primal infeasibility.
        let mut primal_infeasibility = 0.0_f64;
        let mut ax = vec![0.0; prob.m_eq()];
        prob.a_mul(&self.x, &mut ax);
        for (&axi, &bi) in ax.iter().zip(&prob.b) {
            primal_infeasibility = primal_infeasibility.max((axi - bi).abs());
        }
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&self.x, &mut gx);
        for (&gxi, &hi) in gx.iter().zip(&prob.h) {
            primal_infeasibility = primal_infeasibility.max((gxi - hi).max(0.0));
        }
        for i in 0..n {
            primal_infeasibility = primal_infeasibility.max((prob.lb_of(i) - self.x[i]).max(0.0));
            primal_infeasibility = primal_infeasibility.max((self.x[i] - prob.ub_of(i)).max(0.0));
        }

        // Complementarity.
        let mut complementarity = 0.0_f64;
        for ((&zi, &hi), &gxi) in self.z.iter().zip(&prob.h).zip(&gx) {
            complementarity = complementarity.max((zi * (hi - gxi)).abs());
        }
        for i in 0..n {
            let (lb, ub) = (prob.lb_of(i), prob.ub_of(i));
            if lb > -1e19 {
                complementarity = complementarity.max((self.z_lb[i] * (self.x[i] - lb)).abs());
            }
            if ub < 1e19 {
                complementarity = complementarity.max((self.z_ub[i] * (ub - self.x[i])).abs());
            }
        }

        QpResiduals {
            primal_infeasibility,
            dual_infeasibility,
            complementarity,
        }
    }
}

#[cfg(test)]
mod residual_tests {
    use super::*;
    use crate::ipm::{solve_qp_ipm, QpOptions};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// KKT residuals vanish at the optimum even when **variable bounds are
    /// active** — the sharp check of the `−z_lb + z_ub` stationarity sign.
    /// `min x0²+x1² −3x0 −4x1 s.t. 0 ≤ x ≤ 0.5` clamps to the upper bounds
    /// `(0.5, 0.5)` (unconstrained optimum is `(1.5, 2)`), so `z_ub > 0` and
    /// the stationarity term must carry it with the right sign.
    #[test]
    fn kkt_residuals_vanish_with_active_bounds() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![0.0, 0.0],
            ub: vec![0.5, 0.5],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 0.5).abs() < 1e-5 && (sol.x[1] - 0.5).abs() < 1e-5);
        let res = sol.kkt_residuals(&prob);
        assert!(
            res.kkt_error() < 1e-6,
            "active-bound residuals not small: {res:?}"
        );
    }

    /// The opt-in iterate trace is populated only when requested, records one
    /// entry per interior-point iteration *plus* a terminal record at the
    /// converged iterate (the NLP path's N+1 convention), and reflects
    /// convergence (μ and the residuals shrink toward the optimum).
    #[test]
    fn iterate_trace_is_opt_in_and_records_convergence() {
        // A bounded QP (inequalities ⇒ a non-trivial central path, μ > 0).
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        };
        // Off by default: no trace, no overhead.
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert!(
            sol.iterates.is_empty(),
            "default solve must not collect a trace"
        );

        // On: one record per iteration, μ and residuals decreasing to the end.
        let opts = QpOptions {
            collect_iterates: true,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(!sol.iterates.is_empty(), "trace should be populated");
        let first = &sol.iterates[0];
        let last = sol.iterates.last().unwrap();
        assert!(first.iter == 0);
        assert!(first.mu > 0.0, "early μ should be positive");
        assert!(
            last.mu < first.mu,
            "μ should decrease: {} -> {}",
            first.mu,
            last.mu
        );
        // The trace ends at a (near-)converged iterate (this problem starts
        // primal-feasible, so μ — not primal infeasibility — is the signal).
        assert!(last.mu < 1e-6, "final traced μ {} should be tiny", last.mu);
        assert!(
            last.dual_infeasibility < 1e-5,
            "final traced dual infeasibility {} should be small",
            last.dual_infeasibility
        );
        // Every stepping iterate has positive fraction-to-boundary lengths;
        // the terminal converged record takes no step, so its α's are zero.
        let (term, stepping) = sol.iterates.split_last().unwrap();
        for r in stepping {
            assert!(r.alpha_primal > 0.0 && r.alpha_primal <= 1.0);
            assert!(r.alpha_dual > 0.0 && r.alpha_dual <= 1.0);
        }
        assert_eq!(term.alpha_primal, 0.0, "converged record takes no step");
        assert_eq!(term.alpha_dual, 0.0, "converged record takes no step");
    }

    /// Inequality complementarity: a binding general inequality must show
    /// `z·slack ≈ 0`, and stationarity must vanish with the `Gᵀz` term.
    /// `min x0²+x1² −3x0 −4x1 s.t. x0+x1 ≤ 1` → optimum on the face (0.25, 0.75).
    #[test]
    fn kkt_residuals_vanish_with_binding_inequality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 2.0)],
            c: vec![-3.0, -4.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let res = sol.kkt_residuals(&prob);
        assert!(
            res.kkt_error() < 1e-6,
            "binding-inequality residuals not small: {res:?}"
        );
    }
}
