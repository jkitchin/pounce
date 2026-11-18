//! Square-system recovery: when the interior-point solve of a square,
//! constant-objective model fails, solve `c(x) = 0` directly.
//!
//! # Why
//!
//! A model with as many free variables as equality rows, no inequality rows
//! and a constant objective is not an optimization problem. It is a system of
//! nonlinear equations wearing an NLP's clothes, and every feasible point is a
//! global optimum. That is the class an equation-oriented flowsheet
//! initialization hands a solver (IDAES-style: fix the degrees of freedom,
//! solve the square block), and it is the class where an interior-point
//! failure has the least to do with the model: the barrier, the filter and
//! the Hessian approximation all exist to trade objective against
//! feasibility, and here there is no objective to trade.
//!
//! Measured on `square_flowsheet_resto` (32×32, `f ≡ 0`, feasible by
//! construction). Its Jacobian is nearly rank-deficient even at the solution
//! (`σ_min ≈ 3.6e-9` against `σ_max ≈ 3e6`; the near-null direction is a
//! uniform shift of the eight temperatures, pinned only by `exp(-6000/T)`),
//! so the solution set is nearly non-isolated. Under
//! `hessian_approximation=limited-memory` POUNCE on FERAL PR#222 runs to the
//! 3000-iteration cap and every rung of the second-opinion ladder fails too;
//! upstream Ipopt 3.14.19 declares local infeasibility at iteration 50. A
//! Levenberg–Marquardt least-squares solve that uses only `J` solves it from
//! the same start in 8 iterations. The two figures are from two FERALs and
//! both are named: on FERAL PR#222 at default options, `max|c| = 1.9e-10`
//! (recorded when the stage was written); on the published FERAL 0.20 this
//! crate builds against, where rung 4 (`feral_increase_quality=no`) rescues
//! the leg first by default and the stage is reached only with
//! `feral_increase_quality_retry=no`, `max|c| = 2.874e-10` (scaled
//! `1.710e-10`), at `max_iter` 500 and 3000 alike.
//!
//! # The gate
//!
//! [`probe`] opens this stage only when all of the following hold of the TNLP
//! the ladder was handed:
//!
//! * every row is an equality (`g_l == g_u`, finite) and there is at least
//!   one;
//! * the number of **free** variables — `x_l < x_u`, i.e. after removing the
//!   variables `fixed_variable_treatment` would remove — equals the row count;
//! * the objective is constant: the model does not *prove* `∇f` varies
//!   ([`TNLP::derivative_proofs`]), and `∇f` evaluates to exactly zero, every
//!   component, at the starting point.
//!
//! The last test is evaluated, not proved, and that is a deliberate choice.
//! A structural proof would be the stronger gate, but it is not available on
//! the path that matters: presolve wrappers decline `derivative_proofs` (they
//! change what the rows are), and the C, Python and CasADi surfaces hand
//! POUNCE callbacks, not algebra. What makes an evaluated gate safe is that
//! **nothing reported rests on it**. The gate only decides whether to spend
//! the attempt. The verdict rests on the certificate checked at the returned
//! point `x̂` itself (below), which includes `∇f(x̂) = 0` evaluated *there*:
//! at a feasible `x̂` with `∇f(x̂) = 0`, `(λ, z) = (0, 0)` satisfies
//! stationarity and complementarity exactly, so `x̂` is a first-order KKT
//! point by the same definition the IPM's convergence test uses — whether or
//! not `f` is constant anywhere else.
//!
//! # The algorithm
//!
//! A bounded, scaled trust-region Levenberg–Marquardt method on
//! `min ½‖c(x)‖²` over the free variables:
//!
//! * **Step.** Moré's trust-region form (Moré 1978, MINPACK `lmder`/`lmpar`):
//!   with column scaling `D`, the step `p̃ = D p` solves
//!   `(ÃᵀÃ + λI) p̃ = −Ãᵀc`, `Ã = J D⁻¹`, with `λ ≥ 0` chosen so
//!   `‖p̃‖ ≈ Δ` (within 10%) by safeguarded Newton iteration on the secular
//!   equation `1/‖p̃(λ)‖ = 1/Δ`. The Gauss–Newton step is taken when it
//!   already fits.
//! * **Linear algebra.** Each `λ` trial factors the quasi-definite augmented
//!   system `[[λI, Ãᵀ], [Ã, −I]]` with the workspace's sparse symmetric
//!   backend (FERAL) rather than forming `JᵀJ`, which squares the condition
//!   number and densifies. One extra back-solve against the same factor gives
//!   `(ÃᵀÃ + λI)⁻¹ p̃` for the Newton update. At most
//!   [`MAX_LAMBDA_TRIALS`] factorizations per iteration.
//! * **Scaling.** `D_j = max over iterates of ‖J_{:,j}‖₂` (Moré's
//!   non-decreasing column norms), which makes the method invariant to
//!   rescaling a variable — the fixture's variables span `1e-6` to `9e6`.
//!   The rows are **not** reweighted: equilibrating them by `‖J_i(x₀)‖∞` was
//!   measured on the fixture and lost it from every start tried, while the
//!   unweighted merit (MINPACK's, and scipy's `method='lm'`) solves it.
//! * **Bounds.** The trial point is the projection of `x + p` onto the box,
//!   and the gain ratio is computed for the *projected* step against the
//!   linear model, so a projection that destroys the predicted decrease
//!   shrinks `Δ` — which drives `λ` up and the step toward the scaled
//!   (projected) gradient. Every iterate is inside the box by construction.
//!   This is the projected Levenberg–Marquardt idea of Kanzow, Yamashita &
//!   Fukushima (2004), in a trust-region rather than a line-search
//!   globalization.
//! * **Update.** Gain ratio `ρ = ared/pred`; accept when `ρ ≥ 1e-4`; halve
//!   `Δ` (to at most the step taken) when `ρ < 0.25`; set `Δ = 2‖D s‖` when
//!   `ρ > 0.75` or the step was Gauss–Newton. A non-finite residual at the
//!   trial point is a rejection with `Δ` quartered.
//!
//! # Termination and honesty
//!
//! The LM stops as soon as the iterate passes the acceptance test, or fails
//! on an iteration cap (`min(max_iter, 200)`), on a vanishing projected
//! gradient of `‖c‖²` at a non-root ([`STATIONARITY_TOL`]), on `Δ` collapsing to
//! `1e-15·max(‖Dx‖, 1)` (a projected stationary point of `‖c‖²` that is not a
//! root — a *local* least-squares minimum, so the original verdict stands),
//! or on an evaluation / factorization failure.
//!
//! The acceptance test is the IPM's own, at zero multipliers: Ipopt's
//! convergence test asks for the *scaled* overall error below `tol` and the
//! *unscaled* constraint violation below `constr_viol_tol`, and at
//! `(λ, z) = (0, 0)` with `∇f = 0` the dual and complementarity terms vanish,
//! leaving `max_i s_i|c_i| ≤ tol` and `max_i |c_i| ≤ constr_viol_tol`. `s_i`
//! is the row scaling the NLP arm computes for the same model, by the same
//! `pounce_nlp::orig_ipopt_nlp` helpers at the same point (`ipm_scaling`
//! has the details): `none` is `1`; `user-scaling` and `curvature-based`
//! are the TNLP's `g_scaling` floored at `nlp_scaling_min_value`;
//! `gradient-based` — and `equilibration-based`, which the NLP arm runs as
//! gradient-based — is the row-max rule over the full Jacobian sampled at
//! the TNLP's starting point with only fixed variables lifted, applied only
//! when some row exceeds `nlp_scaling_max_gradient` (or a constraint target
//! gradient is set). The one thing not reproduced is a wrapper the
//! application inserts *inside* `optimize_tnlp` that moves the start; that
//! can change `s_i`, but the unscaled half of the test is independent of
//! `s_i`, so no choice of `s_i` admits a point with
//! `max|c_i| > constr_viol_tol`. On top of that
//! the box is satisfied **exactly** (projection; no `bound_relax_factor`
//! widening is borrowed), and `∇f(x̂)` must be exactly zero at the returned
//! point. If any check fails the stage reports failure and the caller keeps
//! the original verdict untouched.
//!
//! # Budget and callback
//!
//! The LM honours `max_wall_time` and `max_cpu_time`, as a budget measured
//! from the start of the stage — the per-solve budget each option rung's
//! `optimize_tnlp` also gets — and reads the clock once per iterate, after
//! the convergence test. It fires the TNLP's `intermediate_callback` at
//! every iterate, the start included, with `mode = RegularMode`, `inf_pr`
//! the scaled violation, `inf_du = max|∇f|`, `mu = 0`, `d_norm` the
//! accepted step's `‖s‖∞` and `regularization_size` the LM parameter; a
//! `false` stops it. A stop or an exhausted budget ends the stage without a
//! promotion. No live iterate is installed for the callback
//! (`GetIpoptCurrentIterate` finds none), because there is no IPM iterate.
//!
//! # What is reported
//!
//! `Solve_Succeeded` with `x̂`, `λ = 0` and `z_l = z_u = 0`, and statistics
//! that describe the LM: `iteration_count` is its iteration count and, when
//! the application records iteration history, `iterations` holds one record
//! per iterate (the start included, the IPM's convention; `alpha_primal_char`
//! `L`); `final_scaled_objective` is `f` times the objective multiplier the
//! NLP arm would have applied. The post-solve computations that read the
//! IPM's converged KKT factor (sIPOPT, the reduced Hessian) have nothing to
//! read and are not run; the CLI says so. Those are true
//! KKT multipliers, not placeholders: with `∇f(x̂) = 0` they satisfy
//! `∇f + Jᵀλ − z_l + z_u = 0` exactly. They are unique when `J(x̂)` is
//! nonsingular and no bound is active; when `J` is singular (as on the
//! fixture, numerically) the multiplier set is a cone containing zero and
//! zero is its minimum-norm member. They are **not** the multipliers an IPM
//! would have reported for the same point, because the IPM never reached it.
//!
//! The root may differ from the one an IPM would have reached: a square
//! system can have several, and the fixture's nearly non-isolated solution
//! set puts the LM root about 11% (relative, in the temperature block) from
//! the generator's `x*`. With a constant objective each is a global optimum.

use pounce_common::options_list::OptionsList;
use pounce_common::types::{Index, Number};
use pounce_linsol::{ESymSolverStatus, SparseSymLinearSolverInterface};
use pounce_nlp::SolveStatistics;
use pounce_nlp::alg_types::SolverReturn;
use pounce_nlp::constant_derivatives::DerivativeProof;
use pounce_nlp::tnlp::{
    BoundsInfo, IndexStyle, IpoptCq, IpoptData, ScalingRequest, Solution, SparsityRequest,
    StartingPoint, TNLP,
};
use std::cell::RefCell;
use std::rc::Rc;

/// The label this stage carries in the second-opinion record (`tried`,
/// `promoted_by`, the sweep's `2nd=` column).
pub const SQUARE_SYSTEM_RUNG_LABEL: &str = "square_system_lm";

/// Factorizations allowed per LM iteration while solving the secular
/// equation for `λ` (MINPACK `lmpar` allows 10 Newton steps; the extra one
/// is the Gauss–Newton trial).
pub const MAX_LAMBDA_TRIALS: usize = 11;

/// The iteration cap is `min(max_iter, this)`. Measured on the fixture from
/// eight starts (`x*` scaled by 0.1 … 10 per block): 5 to 143 iterations.
pub const LM_ITERATION_CAP: usize = 200;

/// Smallest `λ` the augmented system is factored at. `λ = 0` makes the
/// Schur complement `ÃᵀÃ` exactly singular on a rank-deficient Jacobian,
/// and `Ã` is column-normalized, so `1e-16` is an `ε`-sized regularization.
const LAMBDA_FLOOR: Number = 1e-16;

/// MINPACK's `gtol` test: stop when every column of `Ã` is orthogonal to
/// `c` to within this cosine (the columns are normalized by `D`, so each
/// `|Ã_jᵀc|/‖c‖` lies in `[0, 1]`), counting only directions the box lets
/// the iterate move in. Set at the noise level on purpose: it exists for the
/// exact case — `degenerate_start_infeasible` starts where `J = 0`, and
/// without it the method spends 57 iterations halving `Δ` at a point no
/// step can leave — and must never fire on the near-null directions of a
/// nearly singular but consistent system like `square_flowsheet_resto`.
pub const STATIONARITY_TOL: Number = 1e-12;

/// A residual system `c: R^n → R^m` with a fixed sparse Jacobian pattern.
pub trait ResidualSystem {
    /// `c(x)` into `c`. `false` when the model cannot be evaluated at `x`.
    fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool;
    /// Jacobian values at `x`, in the order of [`LmProblem::jac_rows`].
    fn eval_jac(&mut self, x: &[Number], vals: &mut [Number]) -> bool;
}

/// Everything the LM core needs that is not a callback.
#[derive(Debug, Clone)]
pub struct LmProblem {
    pub n: usize,
    pub m: usize,
    /// Jacobian triplet pattern, 0-based. Duplicates are summed.
    pub jac_rows: Vec<usize>,
    pub jac_cols: Vec<usize>,
    /// Box; `±∞` for an absent bound.
    pub lb: Vec<Number>,
    pub ub: Vec<Number>,
}

/// The acceptance test, in the NLP arm's terms — see the module docs.
#[derive(Debug, Clone)]
pub struct FeasibilityTest {
    /// Per-row scaling `s_i` the scaled test multiplies by.
    pub row_scale: Vec<Number>,
    /// `tol`, against `max_i s_i |c_i|`.
    pub tol: Number,
    /// `constr_viol_tol`, against `max_i |c_i|`.
    pub constr_viol_tol: Number,
}

impl FeasibilityTest {
    /// `(scaled, unscaled)` max-norm violations.
    pub fn violations(&self, c: &[Number]) -> (Number, Number) {
        let mut scaled: Number = 0.0;
        let mut unscaled: Number = 0.0;
        for (i, &ci) in c.iter().enumerate() {
            let a = ci.abs();
            // A NaN must not read as feasible: `max` would drop it.
            if !a.is_finite() {
                return (Number::INFINITY, Number::INFINITY);
            }
            unscaled = unscaled.max(a);
            scaled = scaled.max(self.row_scale.get(i).copied().unwrap_or(1.0) * a);
        }
        (scaled, unscaled)
    }

    pub fn passes(&self, c: &[Number]) -> bool {
        let (s, u) = self.violations(c);
        s <= self.tol && u <= self.constr_viol_tol
    }
}

/// How an LM run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LmStatus {
    /// The iterate passed the [`FeasibilityTest`].
    Converged,
    /// Iteration cap reached first.
    IterationLimit,
    /// The projected gradient of `½‖c‖²` vanished at a point that is not a
    /// root: a stationary point of the least-squares problem on the box
    /// (MINPACK's `gtol` test, at [`STATIONARITY_TOL`]). No step this method
    /// can take leaves it, so it stops rather than collapse `Δ` to get there.
    Stationary,
    /// `Δ` collapsed: a projected stationary point of `‖c‖²` that is not a
    /// root, as far as this method can tell.
    TrustRegionCollapsed,
    /// The residual or Jacobian could not be evaluated at an accepted point,
    /// or was not finite there.
    EvaluationFailed,
    /// The augmented system could not be factored at any `λ` tried.
    LinearSolverFailed,
    /// `max_cpu_time` ran out before the iterate passed the test.
    CpuTimeExceeded,
    /// `max_wall_time` ran out before the iterate passed the test.
    WallTimeExceeded,
    /// The TNLP's `intermediate_callback` returned `false`.
    UserRequestedStop,
}

/// What an LM run returns.
#[derive(Debug, Clone)]
pub struct LmResult {
    pub status: LmStatus,
    /// Last accepted iterate (inside the box).
    pub x: Vec<Number>,
    /// `c(x)` at that iterate.
    pub c: Vec<Number>,
    pub iterations: usize,
    pub residual_evals: usize,
    pub jacobian_evals: usize,
    pub factorizations: usize,
}

/// Merged Jacobian pattern plus the augmented-system triplets built on it.
struct Augmented {
    /// `slot[k]` = merged entry of TNLP triplet `k`.
    slot: Vec<usize>,
    /// Merged `(row, col)` pairs.
    uniq: Vec<(usize, usize)>,
    /// 1-based lower-triangle triplets of `[[λI, Ãᵀ], [Ã, −I]]`, laid out
    /// `n` diagonal, then the merged `Ã` entries, then `m` diagonal.
    irn: Vec<Index>,
    jcn: Vec<Index>,
}

impl Augmented {
    fn new(p: &LmProblem) -> Self {
        let mut index = std::collections::HashMap::with_capacity(p.jac_rows.len());
        let mut uniq = Vec::new();
        let mut slot = Vec::with_capacity(p.jac_rows.len());
        for (&i, &j) in p.jac_rows.iter().zip(&p.jac_cols) {
            let next = uniq.len();
            let s = *index.entry((i, j)).or_insert(next);
            if s == next {
                uniq.push((i, j));
            }
            slot.push(s);
        }
        let nnz = p.n + uniq.len() + p.m;
        let mut irn = Vec::with_capacity(nnz);
        let mut jcn = Vec::with_capacity(nnz);
        for j in 0..p.n {
            irn.push((j + 1) as Index);
            jcn.push((j + 1) as Index);
        }
        for &(i, j) in &uniq {
            irn.push((p.n + i + 1) as Index);
            jcn.push((j + 1) as Index);
        }
        for i in 0..p.m {
            irn.push((p.n + i + 1) as Index);
            jcn.push((p.n + i + 1) as Index);
        }
        Self {
            slot,
            uniq,
            irn,
            jcn,
        }
    }

    fn merge(&self, raw: &[Number], out: &mut [Number]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for (k, &v) in raw.iter().enumerate() {
            out[self.slot[k]] += v;
        }
    }
}

fn norm2(v: &[Number]) -> Number {
    v.iter().map(|a| a * a).sum::<Number>().sqrt()
}

fn project(x: &mut [Number], lb: &[Number], ub: &[Number]) {
    for j in 0..x.len() {
        if x[j] < lb[j] {
            x[j] = lb[j];
        }
        if x[j] > ub[j] {
            x[j] = ub[j];
        }
    }
}

/// The factored-system half of the LM: owns the backend and the scaled
/// Jacobian values for the current iterate.
struct StepSolver {
    backend: Box<dyn SparseSymLinearSolverInterface>,
    aug: Augmented,
    n: usize,
    m: usize,
    /// Merged `Ã = J D⁻¹` values, aligned with `aug.uniq`.
    a: Vec<Number>,
    factorizations: usize,
}

impl StepSolver {
    /// Factor `[[λI, Ãᵀ], [Ã, −I]]` and return `(p̃, (ÃᵀÃ+λI)⁻¹p̃)` for the
    /// right-hand side `[0; −c]`. `None` on a failed or non-finite solve.
    fn solve(&mut self, lambda: Number, c: &[Number]) -> Option<(Vec<Number>, Vec<Number>)> {
        let (n, m) = (self.n, self.m);
        {
            let vals = self.backend.values_array_mut();
            for v in vals.iter_mut().take(n) {
                *v = lambda;
            }
            vals[n..n + self.a.len()].copy_from_slice(&self.a);
            for v in vals.iter_mut().skip(n + self.a.len()) {
                *v = -1.0;
            }
        }
        let mut rhs = vec![0.0; n + m];
        for i in 0..m {
            rhs[n + i] = -c[i];
        }
        self.factorizations += 1;
        let st =
            self.backend
                .multi_solve(true, &self.aug.irn, &self.aug.jcn, 1, &mut rhs, false, 0);
        if st != ESymSolverStatus::Success || rhs.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let p: Vec<Number> = rhs[..n].to_vec();
        let mut rhs2 = vec![0.0; n + m];
        rhs2[..n].copy_from_slice(&p);
        let st =
            self.backend
                .multi_solve(false, &self.aug.irn, &self.aug.jcn, 1, &mut rhs2, false, 0);
        if st != ESymSolverStatus::Success || rhs2.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some((p, rhs2[..n].to_vec()))
    }

    /// `‖Ãᵀc‖₂`, the scaled gradient norm — MINPACK's upper bound on the
    /// `λ` that fits the trust region is `‖Ãᵀc‖/Δ`.
    fn scaled_gradient_norm(&self, c: &[Number]) -> Number {
        let mut g = vec![0.0; self.n];
        for (k, &(i, j)) in self.aug.uniq.iter().enumerate() {
            g[j] += self.a[k] * c[i];
        }
        norm2(&g)
    }

    /// Is `x` a stationary point of `½‖c‖²` on the box? See
    /// [`STATIONARITY_TOL`].
    fn is_stationary(&self, c: &[Number], x: &[Number], lb: &[Number], ub: &[Number]) -> bool {
        let cn = norm2(c);
        // A root (or a NaN norm) is not a stationary non-root.
        if cn.is_nan() || cn <= 0.0 {
            return false;
        }
        let mut g = vec![0.0; self.n];
        for (k, &(i, j)) in self.aug.uniq.iter().enumerate() {
            g[j] += self.a[k] * c[i];
        }
        g.iter().enumerate().all(|(j, &gj)| {
            // Descent moves along −g: blocked at a lower bound when g > 0,
            // at an upper bound when g < 0.
            let blocked = (gj > 0.0 && x[j] <= lb[j]) || (gj < 0.0 && x[j] >= ub[j]);
            blocked || gj.abs() <= STATIONARITY_TOL * cn
        })
    }

    /// Moré's trust-region step: `λ` such that `‖p̃(λ)‖ ≈ Δ`.
    /// Returns `(p̃, λ)`, with `λ = 0` reporting a Gauss–Newton step.
    fn trust_region_step(
        &mut self,
        c: &[Number],
        delta: Number,
        lambda_prev: Number,
    ) -> Option<(Vec<Number>, Number)> {
        // Gauss–Newton first: if it fits, it is the step.
        let mut lo: Number = 0.0;
        if let Some((p, _)) = self.solve(LAMBDA_FLOOR, c)
            && norm2(&p) <= 1.1 * delta
        {
            return Some((p, 0.0));
        }
        let gnorm = self.scaled_gradient_norm(c);
        if !(gnorm.is_finite() && gnorm > 0.0) {
            return None;
        }
        let mut hi = gnorm / delta;
        let mut lambda = if lambda_prev > LAMBDA_FLOOR {
            lambda_prev.clamp(LAMBDA_FLOOR, hi)
        } else {
            (1e-3 * hi).max(LAMBDA_FLOOR)
        };
        let mut best: Option<(Vec<Number>, Number)> = None;
        for _ in 0..MAX_LAMBDA_TRIALS - 1 {
            let Some((p, u)) = self.solve(lambda, c) else {
                // A failed factorization at this λ: more regularization is
                // the only direction that can help.
                lo = lambda;
                lambda = (10.0 * lambda).max(1e-12);
                continue;
            };
            let pn = norm2(&p);
            let fits = (pn - delta).abs() <= 0.1 * delta;
            // Keep the last inside-the-region candidate as a fallback; any
            // `p̃` with `‖p̃‖ ≤ 1.1Δ` is an acceptable (if conservative) step.
            if pn <= 1.1 * delta {
                best = Some((p.clone(), lambda));
            }
            if fits {
                return Some((p, lambda));
            }
            if pn > delta {
                lo = lo.max(lambda);
            } else {
                hi = hi.min(lambda);
            }
            let pu: Number = p.iter().zip(&u).map(|(a, b)| a * b).sum();
            let newton = if pu > 0.0 {
                lambda + (pn * pn / pu) * ((pn - delta) / delta)
            } else {
                Number::NAN
            };
            lambda = if newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                // Outside the bracket: MINPACK's safeguard.
                (lo * hi).sqrt().max(1e-3 * hi).max(LAMBDA_FLOOR)
            };
        }
        if best.is_none() {
            // Out of trials without a fitting step. At `hi` the step is
            // inside the region by construction; take it.
            if let Some((p, _)) = self.solve(hi, c) {
                best = Some((p, hi));
            }
        }
        best
    }
}

/// One iterate of the LM, as handed to [`LmHooks::on_iterate`].
#[derive(Debug)]
pub struct LmIterate<'a> {
    /// Iteration index: `0` is the (projected) start, `k` the iterate after
    /// the `k`-th step — accepted or not; a rejected step leaves `x` as it
    /// was and reports `accepted = false`.
    pub iter: usize,
    pub x: &'a [Number],
    pub c: &'a [Number],
    /// `‖s‖∞` of the step that produced this iterate (`0` at the start and
    /// after a rejected step).
    pub step_inf_norm: Number,
    /// The Levenberg–Marquardt parameter of that step (`0` for a
    /// Gauss–Newton step and at the start).
    pub lambda: Number,
    /// Whether that step was accepted (`false` at the start).
    pub accepted: bool,
}

/// What the LM consults besides the model: a time budget and a per-iterate
/// observer. [`LmHooks::default`] is neither.
#[derive(Default)]
pub struct LmHooks<'a> {
    /// Checked once per iterate, after the convergence test and before the
    /// next step, so an iterate that already passes is never thrown away
    /// for being late — the same order as the IPM's convergence check.
    pub deadline: Option<pounce_common::timing::Deadline>,
    /// Called at every iterate, the start included, *before* the
    /// convergence test (the IPM's order: upstream's
    /// `OptimalityErrorConvergenceCheck` fires the intermediate callback
    /// first). Returning `false` stops the method with
    /// [`LmStatus::UserRequestedStop`].
    pub on_iterate: Option<&'a mut dyn FnMut(&LmIterate<'_>) -> bool>,
}

impl LmHooks<'_> {
    fn fire(&mut self, it: &LmIterate<'_>) -> bool {
        match self.on_iterate.as_mut() {
            Some(f) => f(it),
            None => true,
        }
    }

    fn time_exceeded(&self) -> Option<LmStatus> {
        use pounce_common::timing::DeadlineKind;
        match self.deadline.as_ref().and_then(|d| d.exceeded()) {
            Some(DeadlineKind::Cpu) => Some(LmStatus::CpuTimeExceeded),
            Some(DeadlineKind::Wall) => Some(LmStatus::WallTimeExceeded),
            None => None,
        }
    }
}

/// Solve `c(x) = 0` over the box by bounded trust-region LM. See the module
/// docs for the method, the scaling and the termination rules.
pub fn bounded_levenberg_marquardt(
    model: &mut dyn ResidualSystem,
    prob: &LmProblem,
    x0: &[Number],
    test: &FeasibilityTest,
    max_iter: usize,
    mut backend: Box<dyn SparseSymLinearSolverInterface>,
    mut hooks: LmHooks<'_>,
) -> LmResult {
    let (n, m) = (prob.n, prob.m);
    let mut x = x0.to_vec();
    project(&mut x, &prob.lb, &prob.ub);
    let mut c = vec![0.0; m];
    let mut out = LmResult {
        status: LmStatus::EvaluationFailed,
        x: x.clone(),
        c: c.clone(),
        iterations: 0,
        residual_evals: 0,
        jacobian_evals: 0,
        factorizations: 0,
    };
    out.residual_evals += 1;
    if !model.eval_c(&x, &mut c) || c.iter().any(|v| !v.is_finite()) {
        return out;
    }
    out.c = c.clone();
    // Iterate 0, in the order every later iterate is judged in below:
    // observer, convergence, clock.
    let start = LmIterate {
        iter: 0,
        x: &x,
        c: &c,
        step_inf_norm: 0.0,
        lambda: 0.0,
        accepted: false,
    };
    if !hooks.fire(&start) {
        out.status = LmStatus::UserRequestedStop;
        return out;
    }
    if test.passes(&c) {
        out.status = LmStatus::Converged;
        return out;
    }
    if let Some(st) = hooks.time_exceeded() {
        out.status = st;
        return out;
    }

    let aug = Augmented::new(prob);
    let nnz = aug.irn.len();
    if backend.initialize_structure((n + m) as Index, nnz as Index, &aug.irn, &aug.jcn)
        != ESymSolverStatus::Success
    {
        out.status = LmStatus::LinearSolverFailed;
        return out;
    }
    let n_uniq = aug.uniq.len();
    let mut solver = StepSolver {
        backend,
        aug,
        n,
        m,
        a: vec![0.0; n_uniq],
        factorizations: 0,
    };

    let mut raw = vec![0.0; prob.jac_rows.len()];
    let mut jac = vec![0.0; n_uniq];
    let mut d = vec![0.0; n];
    // Evaluate J at `x`, merge duplicates, raise the column norms, and
    // refresh `Ã`. `false` on a failed or non-finite Jacobian.
    let refresh = |model: &mut dyn ResidualSystem,
                   x: &[Number],
                   raw: &mut [Number],
                   jac: &mut [Number],
                   d: &mut [Number],
                   solver: &mut StepSolver|
     -> bool {
        if !model.eval_jac(x, raw) || raw.iter().any(|v| !v.is_finite()) {
            return false;
        }
        solver.aug.merge(raw, jac);
        let mut col = vec![0.0; d.len()];
        for (k, &(_, j)) in solver.aug.uniq.iter().enumerate() {
            col[j] += jac[k] * jac[k];
        }
        for j in 0..d.len() {
            d[j] = d[j].max(col[j].sqrt());
        }
        for (k, &(_, j)) in solver.aug.uniq.iter().enumerate() {
            let dj = if d[j] > 0.0 { d[j] } else { 1.0 };
            solver.a[k] = jac[k] / dj;
        }
        true
    };
    out.jacobian_evals += 1;
    if !refresh(model, &x, &mut raw, &mut jac, &mut d, &mut solver) {
        out.factorizations = solver.factorizations;
        return out;
    }
    let dscale = |d: &[Number], j: usize| if d[j] > 0.0 { d[j] } else { 1.0 };
    let scaled_norm = |d: &[Number], v: &[Number]| -> Number {
        v.iter()
            .enumerate()
            .map(|(j, &vj)| (dscale(d, j) * vj).powi(2))
            .sum::<Number>()
            .sqrt()
    };
    // MINPACK's `factor = 100` initial radius.
    let xnorm = scaled_norm(&d, &x);
    let mut delta = if xnorm > 0.0 { 100.0 * xnorm } else { 100.0 };
    let mut lambda: Number = 0.0;
    let mut ct = vec![0.0; m];
    let status = loop {
        // `x` is iterate `out.iterations`, and iterate 0 was judged above.
        let iter = out.iterations;
        if iter >= max_iter {
            break LmStatus::IterationLimit;
        }
        if solver.is_stationary(&c, &x, &prob.lb, &prob.ub) {
            break LmStatus::Stationary;
        }
        out.iterations = iter + 1;
        let Some((pt, lam)) = solver.trust_region_step(&c, delta, lambda) else {
            break LmStatus::LinearSolverFailed;
        };
        lambda = lam;
        let mut xt: Vec<Number> = (0..n).map(|j| x[j] + pt[j] / dscale(&d, j)).collect();
        project(&mut xt, &prob.lb, &prob.ub);
        let s: Vec<Number> = (0..n).map(|j| xt[j] - x[j]).collect();
        let snorm = scaled_norm(&d, &s);

        out.residual_evals += 1;
        let ok = model.eval_c(&xt, &mut ct) && ct.iter().all(|v| v.is_finite());
        let rho = if ok {
            let phi: Number = c.iter().map(|v| v * v).sum();
            let phit: Number = ct.iter().map(|v| v * v).sum();
            // Linear model at the *projected* step.
            let mut lin = c.clone();
            for (k, &(i, j)) in solver.aug.uniq.iter().enumerate() {
                lin[i] += jac[k] * s[j];
            }
            let pred = phi - lin.iter().map(|v| v * v).sum::<Number>();
            if pred > 0.0 {
                (phi - phit) / pred
            } else {
                Number::NEG_INFINITY
            }
        } else {
            Number::NEG_INFINITY
        };

        if !ok {
            delta *= 0.25;
        } else if rho < 0.25 {
            delta = 0.5 * if snorm > 0.0 { delta.min(snorm) } else { delta };
        } else if rho > 0.75 || lambda == 0.0 {
            delta = delta.max(2.0 * snorm);
        }

        let accepted = rho >= 1e-4;
        if accepted {
            x = xt;
            c.copy_from_slice(&ct);
        }
        let here = LmIterate {
            iter: out.iterations,
            x: &x,
            c: &c,
            step_inf_norm: if accepted {
                s.iter().fold(0.0, |a: Number, v| a.max(v.abs()))
            } else {
                0.0
            },
            lambda,
            accepted,
        };
        if !hooks.fire(&here) {
            break LmStatus::UserRequestedStop;
        }
        if accepted && test.passes(&c) {
            break LmStatus::Converged;
        }
        if let Some(st) = hooks.time_exceeded() {
            break st;
        }
        if accepted {
            out.jacobian_evals += 1;
            if !refresh(model, &x, &mut raw, &mut jac, &mut d, &mut solver) {
                break LmStatus::EvaluationFailed;
            }
        }
        let floor = 1e-15 * scaled_norm(&d, &x).max(1.0);
        // Written to catch a NaN radius too.
        if delta.is_nan() || delta <= floor {
            break LmStatus::TrustRegionCollapsed;
        }
    };
    out.status = status;
    out.x = x;
    out.c = c;
    out.factorizations = solver.factorizations;
    out
}

/// Why [`probe`] did not open the stage.
#[derive(Debug, Clone, PartialEq)]
pub enum SquareSystemDecline {
    /// `square_system_recovery=no`.
    Disabled,
    /// The TNLP could not be queried (info, bounds, start, gradient).
    Unavailable,
    /// No rows, or a row that is not an equality.
    NotEqualityConstrained,
    /// Free variables and equality rows differ in number.
    NotSquare { free: usize, rows: usize },
    /// `∇f` is proved varying or is nonzero at the start.
    ObjectiveNotConstant,
}

/// A square, equality-only, constant-objective model, read off the TNLP and
/// ready to hand to [`run`].
#[derive(Debug, Clone)]
pub struct SquareSystemPlan {
    n_full: usize,
    m: usize,
    /// Full-space index of each free variable.
    free: Vec<usize>,
    /// Full-space starting point, projected into the box; fixed variables
    /// hold their fixed value.
    x_full: Vec<Number>,
    /// The TNLP's starting point with only the fixed variables lifted to
    /// their value and **no** projection — the point
    /// `OrigIpoptNlp::determine_scaling_from_starting_point` samples the
    /// gradient-based row scaling at, so the acceptance test's `s_i` is the
    /// IPM's own rather than one measured somewhere nearby.
    x_scaling: Vec<Number>,
    /// Full Jacobian row of every TNLP triplet, fixed columns included
    /// (the IPM's row maximum runs over all of them).
    jac_row_full: Vec<usize>,
    g_rhs: Vec<Number>,
    /// `(TNLP triplet index, row, free column)` for every Jacobian entry in
    /// a free column.
    kept: Vec<(usize, usize, usize)>,
    nnz_full: usize,
    lb: Vec<Number>,
    ub: Vec<Number>,
}

/// A numeric option as the IPM reads it: the value when set, else the
/// registered default spelled out here — the `f.then_some(v)` idiom of
/// `IpoptApplication`, so an options list without a registry (or one whose
/// tag reads as an unset `0`) cannot hand the stage a different number than
/// the interior-point method got.
fn opt_num(options: &OptionsList, tag: &str, default: Number) -> Number {
    options
        .get_numeric_value(tag, "")
        .ok()
        .and_then(|(v, found)| found.then_some(v))
        .unwrap_or(default)
}

/// Is the stage switched on? Read as a literal tag so the option-wiring
/// scans see it.
pub fn is_enabled(options: &OptionsList) -> bool {
    options
        .get_bool_value("square_system_recovery", "")
        .map(|(v, _found)| v)
        .unwrap_or(true)
}

/// Decide whether the model the ladder was handed is in this stage's class.
/// Costs one `get_nlp_info`, one `get_bounds_info` and — only on a model that
/// is square and equality-only — one starting point, one Jacobian structure
/// query and one `∇f` evaluation.
pub fn probe(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
) -> Result<SquareSystemPlan, SquareSystemDecline> {
    if !is_enabled(options) {
        return Err(SquareSystemDecline::Disabled);
    }
    let lo_inf = opt_num(options, "nlp_lower_bound_inf", -1e19);
    let up_inf = opt_num(options, "nlp_upper_bound_inf", 1e19);
    let mut t = tnlp.borrow_mut();
    let info = t.get_nlp_info().ok_or(SquareSystemDecline::Unavailable)?;
    let n = info.n.max(0) as usize;
    let m = info.m.max(0) as usize;
    if m == 0 {
        return Err(SquareSystemDecline::NotEqualityConstrained);
    }
    let mut x_l = vec![0.0; n];
    let mut x_u = vec![0.0; n];
    let mut g_l = vec![0.0; m];
    let mut g_u = vec![0.0; m];
    if !t.get_bounds_info(BoundsInfo {
        x_l: &mut x_l,
        x_u: &mut x_u,
        g_l: &mut g_l,
        g_u: &mut g_u,
    }) {
        return Err(SquareSystemDecline::Unavailable);
    }
    if (0..m).any(|i| !(g_l[i].is_finite() && g_l[i] == g_u[i])) {
        return Err(SquareSystemDecline::NotEqualityConstrained);
    }
    let lb: Vec<Number> = x_l
        .iter()
        .map(|&v| if v <= lo_inf { Number::NEG_INFINITY } else { v })
        .collect();
    let ub: Vec<Number> = x_u
        .iter()
        .map(|&v| if v >= up_inf { Number::INFINITY } else { v })
        .collect();
    if (0..n).any(|j| lb[j] > ub[j]) {
        return Err(SquareSystemDecline::Unavailable);
    }
    let free: Vec<usize> = (0..n).filter(|&j| lb[j] < ub[j]).collect();
    if free.len() != m {
        return Err(SquareSystemDecline::NotSquare {
            free: free.len(),
            rows: m,
        });
    }
    // Cheapest objective test first: a model that proves `∇f` varies is out
    // without an evaluation.
    if t.derivative_proofs().grad_f == DerivativeProof::Varying {
        return Err(SquareSystemDecline::ObjectiveNotConstant);
    }
    let mut x = vec![0.0; n];
    let mut z_l = vec![0.0; n];
    let mut z_u = vec![0.0; n];
    let mut lam = vec![0.0; m];
    if !t.get_starting_point(StartingPoint {
        init_x: true,
        x: &mut x,
        init_z: false,
        z_l: &mut z_l,
        z_u: &mut z_u,
        init_lambda: false,
        lambda: &mut lam,
    }) || x.iter().any(|v| !v.is_finite())
    {
        return Err(SquareSystemDecline::Unavailable);
    }
    for j in 0..n {
        if lb[j] == ub[j] {
            x[j] = lb[j];
        }
    }
    let x_scaling = x.clone();
    project(&mut x, &lb, &ub);
    let mut grad = vec![0.0; n];
    if !t.eval_grad_f(&x, true, &mut grad) {
        return Err(SquareSystemDecline::Unavailable);
    }
    // Exactly zero, every component: "constant" is the claim, and a small
    // gradient is a different class (the stationarity term would then not
    // vanish at λ = 0).
    if grad.iter().any(|&v| v != 0.0) {
        return Err(SquareSystemDecline::ObjectiveNotConstant);
    }
    let nnz = info.nnz_jac_g.max(0) as usize;
    let mut irow = vec![0 as Index; nnz];
    let mut jcol = vec![0 as Index; nnz];
    if !t.eval_jac_g(
        None,
        false,
        SparsityRequest::Structure {
            irow: &mut irow,
            jcol: &mut jcol,
        },
    ) {
        return Err(SquareSystemDecline::Unavailable);
    }
    let index_base: Index = match info.index_style {
        IndexStyle::Fortran => 1,
        _ => 0,
    };
    let mut free_of = vec![usize::MAX; n];
    for (k, &j) in free.iter().enumerate() {
        free_of[j] = k;
    }
    let mut kept = Vec::with_capacity(nnz);
    let mut jac_row_full = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let i = (irow[k] - index_base) as usize;
        let j = (jcol[k] - index_base) as usize;
        if i >= m || j >= n {
            return Err(SquareSystemDecline::Unavailable);
        }
        jac_row_full.push(i);
        if free_of[j] != usize::MAX {
            kept.push((k, i, free_of[j]));
        }
    }
    Ok(SquareSystemPlan {
        n_full: n,
        m,
        lb: free.iter().map(|&j| lb[j]).collect(),
        ub: free.iter().map(|&j| ub[j]).collect(),
        free,
        x_full: x,
        x_scaling,
        jac_row_full,
        g_rhs: g_l,
        kept,
        nnz_full: nnz,
    })
}

/// The TNLP seen as a [`ResidualSystem`] over the free variables.
struct TnlpResidual<'a> {
    tnlp: &'a Rc<RefCell<dyn TNLP>>,
    plan: &'a SquareSystemPlan,
    x_full: Vec<Number>,
    g: Vec<Number>,
    jac_full: Vec<Number>,
}

impl TnlpResidual<'_> {
    fn scatter(&mut self, x: &[Number]) {
        for (k, &j) in self.plan.free.iter().enumerate() {
            self.x_full[j] = x[k];
        }
    }
}

impl ResidualSystem for TnlpResidual<'_> {
    fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool {
        self.scatter(x);
        if !self
            .tnlp
            .borrow_mut()
            .eval_g(&self.x_full, true, &mut self.g)
        {
            return false;
        }
        for ((ci, gi), ri) in c.iter_mut().zip(&self.g).zip(&self.plan.g_rhs) {
            *ci = gi - ri;
        }
        true
    }

    fn eval_jac(&mut self, x: &[Number], vals: &mut [Number]) -> bool {
        self.scatter(x);
        if !self.tnlp.borrow_mut().eval_jac_g(
            Some(&self.x_full),
            true,
            SparsityRequest::Values {
                values: &mut self.jac_full,
            },
        ) {
            return false;
        }
        for (q, &(k, _, _)) in self.plan.kept.iter().enumerate() {
            vals[q] = self.jac_full[k];
        }
        true
    }
}

/// The scaling the NLP arm would have run this model under, as far as the
/// acceptance test and the reported statistics need it.
struct IpmScaling {
    /// Per-row `s_i` of the scaled constraint violation.
    row_scale: Vec<Number>,
    /// The gradient-based objective factor alone (`1` when none applies) —
    /// `SolveStatistics::final_obj_scaling_factor`.
    computed_obj_scale: Number,
    /// The whole objective multiplier, signed: the caller's
    /// `obj_scaling_factor` times the method's own factor.
    total_obj_scale: Number,
    /// `max |∇f|` over the free variables where gradient-based scaling
    /// sampled it; `NaN` when it did not run.
    start_obj_grad_max: Number,
    /// Evaluations spent measuring it: `(∇f, J)`.
    evals: (usize, usize),
}

/// The scaling `OrigIpoptNlp::determine_scaling_from_starting_point`
/// computes for this model, reproduced from the same `pounce_nlp` helpers
/// at the same point.
///
/// * The method is resolved the way `IpoptApplication` resolves it:
///   `none`; `user-scaling` and `curvature-based` both read the TNLP's
///   `get_scaling_parameters`; anything else — `gradient-based` and the
///   registered-but-unimplemented `equilibration-based` — runs
///   gradient-based.
/// * Gradient-based samples `∇f` and the **full** Jacobian (fixed columns
///   included) at the TNLP's starting point with only the fixed variables
///   lifted — not at the projected start the LM iterates from.
///   [`gradient_scaling_fires`] decides whether the rows get a vector at
///   all, and [`gradient_row_scale`] gives each row its factor.
/// * User scaling floors every `g_scaling` entry at
///   `nlp_scaling_min_value`, and falls back to no scaling when the TNLP
///   declines, as the IPM does. The row factors come from
///   [`user_row_scale`], the function the IPM uses, with the Jacobian
///   sampled at the same point, so `nlp_scaling_untagged_constr` gives the
///   same factors here.
///
/// What is *not* reproduced: a wrapper the application inserts inside
/// `optimize_tnlp` (the start conditioner, the gh#486 variable-scaling
/// substitution) can move the start the IPM sampled. The unscaled
/// `constr_viol_tol` half of the test does not depend on any of this, so a
/// residual difference in `s_i` can change only how far below `tol` the
/// scaled half is met, never admit a point the unscaled half rejects.
///
/// [`gradient_scaling_fires`]: pounce_nlp::orig_ipopt_nlp::gradient_scaling_fires
/// [`gradient_row_scale`]: pounce_nlp::orig_ipopt_nlp::gradient_row_scale
/// [`user_row_scale`]: pounce_nlp::orig_ipopt_nlp::user_row_scale
fn ipm_scaling(
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
    plan: &SquareSystemPlan,
) -> IpmScaling {
    use pounce_nlp::orig_ipopt_nlp::{
        UntaggedConstrScaling, gradient_obj_scale, gradient_row_scale, gradient_scaling_fires,
        user_row_scale,
    };
    let m = plan.m;
    let user_obj = opt_num(options, "obj_scaling_factor", 1.0);
    let min_value = opt_num(options, "nlp_scaling_min_value", 1e-8);
    let unscaled = IpmScaling {
        row_scale: vec![1.0; m],
        computed_obj_scale: 1.0,
        total_obj_scale: user_obj,
        start_obj_grad_max: Number::NAN,
        evals: (0, 0),
    };
    let method = options
        .get_string_value("nlp_scaling_method", "")
        .ok()
        .and_then(|(v, found)| found.then_some(v))
        .unwrap_or_else(|| "gradient-based".to_string());
    let max_grad = opt_num(options, "nlp_scaling_max_gradient", 100.0);
    let constr_target = opt_num(options, "nlp_scaling_constr_target_gradient", 0.0);
    // Each row's Jacobian ∞-norm at the gradient-based sampling point, with
    // upstream's `dbl_min` seed for a row with no nonzero.
    let jac_row_max = || -> Option<Vec<Number>> {
        let mut vals = vec![0.0; plan.nnz_full];
        if !tnlp.borrow_mut().eval_jac_g(
            Some(&plan.x_scaling),
            true,
            SparsityRequest::Values { values: &mut vals },
        ) {
            return None;
        }
        let mut row_max = vec![Number::MIN_POSITIVE; m];
        for (k, &i) in plan.jac_row_full.iter().enumerate() {
            row_max[i] = row_max[i].max(vals[k].abs());
        }
        Some(row_max)
    };
    match method.as_str() {
        "none" => unscaled,
        "user-scaling" | "curvature-based" => {
            let rule = crate::application::untagged_constr_scaling(options);
            let mut obj = 1.0;
            let mut use_x = false;
            let mut xs = vec![1.0; plan.n_full];
            let mut use_g = false;
            let mut gs = vec![1.0; m];
            let mut given = vec![true; m];
            let ok = tnlp.borrow_mut().get_scaling_parameters(ScalingRequest {
                obj_scaling: &mut obj,
                use_x_scaling: &mut use_x,
                x_scaling: &mut xs,
                use_g_scaling: &mut use_g,
                g_scaling: &mut gs,
                g_scaling_given: &mut given,
            });
            if !ok && rule == UntaggedConstrScaling::None {
                return unscaled;
            }
            if !ok {
                obj = 1.0;
            }
            let sampled = std::cell::Cell::new(false);
            let row_scale = user_row_scale(
                ok,
                use_g,
                &gs,
                &given,
                rule,
                || {
                    sampled.set(true);
                    jac_row_max()
                },
                max_grad,
                min_value,
                constr_target,
            );
            let df = if obj.abs() < min_value {
                min_value
            } else {
                obj
            };
            IpmScaling {
                row_scale,
                computed_obj_scale: 1.0,
                total_obj_scale: df * user_obj,
                start_obj_grad_max: Number::NAN,
                evals: (0, usize::from(sampled.get())),
            }
        }
        _ => {
            let obj_target = opt_num(options, "nlp_scaling_obj_target_gradient", 0.0);
            let x = &plan.x_scaling;
            let mut grad = vec![0.0; plan.n_full];
            let mut df = 1.0;
            let mut grad_max = Number::NAN;
            if tnlp.borrow_mut().eval_grad_f(x, true, &mut grad) {
                let gm = plan
                    .free
                    .iter()
                    .fold(0.0 as Number, |a, &j| a.max(grad[j].abs()));
                grad_max = gm;
                if gm.is_finite() {
                    df = gradient_obj_scale(gm, max_grad, min_value, obj_target);
                }
            }
            let row_scale = match jac_row_max() {
                Some(row_max) if gradient_scaling_fires(&row_max, max_grad, constr_target) => {
                    row_max
                        .iter()
                        .map(|&g| gradient_row_scale(g, max_grad, min_value, constr_target))
                        .collect()
                }
                _ => vec![1.0; m],
            };
            IpmScaling {
                row_scale,
                computed_obj_scale: df,
                total_obj_scale: df * user_obj,
                start_obj_grad_max: grad_max,
                evals: (1, 1),
            }
        }
    }
}

/// What [`run`] reports.
#[derive(Debug, Clone)]
pub struct SquareSystemOutcome {
    pub lm: LmResult,
    /// The certificate held and the solution was handed to the TNLP's
    /// `finalize_solution`.
    pub accepted: bool,
    /// One line for the console when not accepted.
    pub reason: String,
    /// Statistics for the promoted verdict, populated on acceptance.
    pub statistics: SolveStatistics,
    /// `(scaled, unscaled)` violation at the returned point.
    pub violation: (Number, Number),
}

/// How [`run`] is to treat its caller's per-iteration surfaces.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions {
    /// Fill `SolveStatistics::iterations` with one record per LM iterate
    /// (the start included, so `iterations.len() == iteration_count + 1`,
    /// the IPM's convention) — set it when the application records
    /// iteration history, so a promoted verdict keeps the
    /// `iterations`/`iteration_count` pairing every other verdict has.
    pub capture_iterations: bool,
}

/// Run the LM on a probed model and, if the result passes every check in the
/// module docs, finalize it through `tnlp` as `Solve_Succeeded`.
///
/// On any failure `tnlp` receives no `finalize_solution` call, so the caller's
/// captured solution stays the one its original verdict describes.
///
/// The LM honours `max_cpu_time` / `max_wall_time` as a budget of its own,
/// measured from this call — the same per-solve budget each option rung's
/// `optimize_tnlp` gets — and fires the TNLP's `intermediate_callback` at
/// every iterate (unless the TNLP says it does not want one), stopping on a
/// `false`. Running out of time and a stop request both end the stage
/// without a promotion.
pub fn run(
    plan: &SquareSystemPlan,
    tnlp: &Rc<RefCell<dyn TNLP>>,
    options: &OptionsList,
    backend: Box<dyn SparseSymLinearSolverInterface>,
    run_opts: RunOptions,
) -> SquareSystemOutcome {
    use pounce_nlp::solve_statistics::IterRecord;
    let wall0 = pounce_common::utils::wallclock_time();
    let cpu0 = pounce_common::utils::cpu_time();
    let deadline = pounce_common::timing::Deadline::new(
        opt_num(options, "max_wall_time", 1e6),
        opt_num(options, "max_cpu_time", 1e6),
    );
    let max_iter = options
        .get_integer_value("max_iter", "")
        .ok()
        .and_then(|(v, found)| found.then_some(v.max(0) as usize))
        .unwrap_or(3000)
        .min(LM_ITERATION_CAP);
    let prob = LmProblem {
        n: plan.free.len(),
        m: plan.m,
        jac_rows: plan.kept.iter().map(|&(_, i, _)| i).collect(),
        jac_cols: plan.kept.iter().map(|&(_, _, j)| j).collect(),
        lb: plan.lb.clone(),
        ub: plan.ub.clone(),
    };
    let x0: Vec<Number> = plan.free.iter().map(|&j| plan.x_full[j]).collect();
    let mut model = TnlpResidual {
        tnlp,
        plan,
        x_full: plan.x_full.clone(),
        g: vec![0.0; plan.m],
        jac_full: vec![0.0; plan.nnz_full],
    };
    let fail = |lm: LmResult, reason: String| SquareSystemOutcome {
        violation: (Number::NAN, Number::NAN),
        lm,
        accepted: false,
        reason,
        statistics: SolveStatistics::default(),
    };
    let scaling = ipm_scaling(tnlp, options, plan);
    let test = FeasibilityTest {
        row_scale: scaling.row_scale.clone(),
        tol: opt_num(options, "tol", 1e-8),
        constr_viol_tol: opt_num(options, "constr_viol_tol", 1e-4),
    };

    // The per-iterate observer: the user's `intermediate_callback` and the
    // iteration record. Each needs `f` and `∇f` at the iterate, which the LM
    // itself never evaluates, so they are evaluated only when one of the two
    // is wanted — and counted.
    let wants_callback = tnlp.borrow().wants_intermediate_callback();
    let mut records: Vec<IterRecord> = Vec::new();
    let mut extra_f_evals = 0usize;
    let mut extra_grad_evals = 0usize;
    let mut observe = |it: &LmIterate<'_>| -> bool {
        if !(wants_callback || run_opts.capture_iterations) {
            return true;
        }
        let mut xf = plan.x_full.clone();
        for (k, &j) in plan.free.iter().enumerate() {
            xf[j] = it.x[k];
        }
        extra_f_evals += 1;
        let obj = tnlp.borrow_mut().eval_f(&xf, true).unwrap_or(Number::NAN);
        let mut grad = vec![0.0; plan.n_full];
        extra_grad_evals += 1;
        let inf_du = if tnlp.borrow_mut().eval_grad_f(&xf, false, &mut grad) {
            plan.free
                .iter()
                .fold(0.0 as Number, |a, &j| a.max(grad[j].abs()))
        } else {
            Number::NAN
        };
        let (scaled, unscaled) = test.violations(it.c);
        let alpha = if it.accepted { 1.0 } else { 0.0 };
        if run_opts.capture_iterations {
            records.push(IterRecord {
                iter: it.iter as Index,
                objective: obj,
                inf_pr: unscaled,
                inf_pr_internal: scaled,
                inf_du,
                mu: 0.0,
                d_norm: it.step_inf_norm,
                regularization: it.lambda,
                alpha_dual: 0.0,
                alpha_primal: alpha,
                alpha_primal_char: if it.iter == 0 { ' ' } else { 'L' },
                ls_trials: 0,
                phase: Default::default(),
            });
        }
        if wants_callback {
            return tnlp.borrow_mut().intermediate_callback(
                pounce_nlp::tnlp::IterStats {
                    mode: pounce_nlp::AlgorithmMode::RegularMode,
                    iter: it.iter as Index,
                    obj_value: obj,
                    inf_pr: scaled,
                    inf_du,
                    mu: 0.0,
                    d_norm: it.step_inf_norm,
                    regularization_size: it.lambda,
                    alpha_du: 0.0,
                    alpha_pr: alpha,
                    ls_trials: 0,
                },
                &IpoptData::default(),
                &IpoptCq::default(),
            );
        }
        true
    };
    let lm = bounded_levenberg_marquardt(
        &mut model,
        &prob,
        &x0,
        &test,
        max_iter,
        backend,
        LmHooks {
            deadline: Some(deadline),
            on_iterate: Some(&mut observe),
        },
    );
    if lm.status != LmStatus::Converged {
        let reason = match lm.status {
            LmStatus::CpuTimeExceeded => format!(
                "max_cpu_time reached after {} iteration(s), before a root",
                lm.iterations
            ),
            LmStatus::WallTimeExceeded => format!(
                "max_wall_time reached after {} iteration(s), before a root",
                lm.iterations
            ),
            LmStatus::UserRequestedStop => format!(
                "the intermediate callback requested a stop after {} iteration(s)",
                lm.iterations
            ),
            st => format!("{st:?} after {} iteration(s)", lm.iterations),
        };
        return fail(lm, reason);
    }

    // The certificate, re-checked at the returned point from fresh
    // evaluations rather than read off the LM's bookkeeping.
    model.scatter(&lm.x);
    let x_full = model.x_full.clone();
    let mut g = vec![0.0; plan.m];
    let mut c = vec![0.0; plan.m];
    if !tnlp.borrow_mut().eval_g(&x_full, true, &mut g) {
        return fail(lm, "eval_g failed at the returned point".into());
    }
    for i in 0..plan.m {
        c[i] = g[i] - plan.g_rhs[i];
    }
    if !test.passes(&c) {
        return fail(
            lm,
            "the returned point does not pass the feasibility test".into(),
        );
    }
    if (0..prob.n).any(|k| !(lm.x[k] >= prob.lb[k] && lm.x[k] <= prob.ub[k])) {
        return fail(lm, "the returned point is outside the variable box".into());
    }
    let mut grad = vec![0.0; plan.n_full];
    if !tnlp.borrow_mut().eval_grad_f(&x_full, false, &mut grad) || grad.iter().any(|&v| v != 0.0) {
        return fail(
            lm,
            "the objective gradient is nonzero at the returned point".into(),
        );
    }
    let Some(obj) = tnlp.borrow_mut().eval_f(&x_full, false) else {
        return fail(lm, "eval_f failed at the returned point".into());
    };
    if !obj.is_finite() {
        return fail(
            lm,
            "the objective is not finite at the returned point".into(),
        );
    }

    let (scaled, unscaled) = test.violations(&c);
    let zeros_n = vec![0.0; plan.n_full];
    let zeros_m = vec![0.0; plan.m];
    tnlp.borrow_mut().finalize_solution(
        Solution {
            status: SolverReturn::Success,
            x: &x_full,
            z_l: &zeros_n,
            z_u: &zeros_n,
            g: &g,
            lambda: &zeros_m,
            obj_value: obj,
        },
        &IpoptData::default(),
        &IpoptCq::default(),
    );

    // Recorded iterates go to any enclosing capture as well, the way
    // `optimize_tnlp` forwards its own.
    pounce_observability::extend_active_capture(&records);
    let statistics = SolveStatistics {
        iteration_count: lm.iterations as Index,
        total_cpu_time_secs: pounce_common::utils::cpu_time() - cpu0,
        total_wallclock_time_secs: pounce_common::utils::wallclock_time() - wall0,
        // The certificate's `f` and `∇f` plus the observer's and the
        // scaling measurement's, and the certificate's `g` on top of the
        // LM's own.
        num_obj_evals: (1 + extra_f_evals) as Index,
        num_constr_evals: (lm.residual_evals + 1) as Index,
        num_obj_grad_evals: (1 + extra_grad_evals + scaling.evals.0) as Index,
        num_constr_jac_evals: (lm.jacobian_evals + scaling.evals.1) as Index,
        final_objective: obj,
        // In the units the IPM reports it in: `total · f`.
        final_scaled_objective: scaling.total_obj_scale * obj,
        final_obj_scaling_factor: scaling.computed_obj_scale,
        final_total_obj_scaling_factor: scaling.total_obj_scale.abs(),
        start_obj_grad_max: scaling.start_obj_grad_max,
        final_dual_inf: 0.0,
        final_constr_viol: scaled,
        final_compl: 0.0,
        final_kkt_error: scaled,
        final_unscaled_dual_inf: 0.0,
        final_unscaled_constr_viol: unscaled,
        final_unscaled_compl: 0.0,
        final_unscaled_kkt_error: unscaled,
        final_kkt_error_above_noise: scaled,
        final_declared_constr_viol: unscaled,
        final_declared_box_viol: 0.0,
        final_mu: 0.0,
        iterations: records,
        ..SolveStatistics::default()
    };
    SquareSystemOutcome {
        lm,
        accepted: true,
        reason: String::new(),
        statistics,
        violation: (scaled, unscaled),
    }
}

/// The backend the stage factors with: FERAL, configured from the same
/// `feral_*` options as the IPM, and with **no** summary sink, so the
/// linear-solver summary a caller reads keeps describing the interior-point
/// solves rather than absorbing this stage's factorizations.
pub fn default_backend(options: &OptionsList) -> Box<dyn SparseSymLinearSolverInterface> {
    Box::new(pounce_feral::FeralSolverInterface::with_config(
        crate::application::feral_config_from_options(options),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense-pattern test system from a closure pair.
    struct Sys<F, J>
    where
        F: FnMut(&[Number], &mut [Number]),
        J: FnMut(&[Number], &mut [Number]),
    {
        f: F,
        j: J,
    }

    impl<F, J> ResidualSystem for Sys<F, J>
    where
        F: FnMut(&[Number], &mut [Number]),
        J: FnMut(&[Number], &mut [Number]),
    {
        fn eval_c(&mut self, x: &[Number], c: &mut [Number]) -> bool {
            (self.f)(x, c);
            true
        }
        fn eval_jac(&mut self, x: &[Number], v: &mut [Number]) -> bool {
            (self.j)(x, v);
            true
        }
    }

    fn dense2(lb: [Number; 2], ub: [Number; 2]) -> LmProblem {
        LmProblem {
            n: 2,
            m: 2,
            jac_rows: vec![0, 0, 1, 1],
            jac_cols: vec![0, 1, 0, 1],
            lb: lb.to_vec(),
            ub: ub.to_vec(),
        }
    }

    fn strict() -> FeasibilityTest {
        FeasibilityTest {
            row_scale: vec![1.0, 1.0],
            tol: 1e-10,
            constr_viol_tol: 1e-10,
        }
    }

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(pounce_feral::FeralSolverInterface::with_config(
            pounce_feral::FeralConfig::default(),
        ))
    }

    /// The root sits **on** a bound, and the unconstrained Newton steps from
    /// this start overshoot it: `x0² + x1² = 2`, `x0 = x1`, `x0 ≤ 1`, from
    /// `(0.2, 3)`. Projection must clip the overshoot onto the bound and the
    /// method must still converge — to the root on the bound, not to the
    /// mirror root `(−1, −1)` and not to a point outside the box.
    #[test]
    fn converges_to_a_root_on_an_active_bound() {
        let prob = dense2([-10.0, -10.0], [1.0, 10.0]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] * x[0] + x[1] * x[1] - 2.0;
                c[1] = x[0] - x[1];
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 1.0, -1.0]);
            },
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[0.2, 3.0],
            &strict(),
            200,
            backend(),
            LmHooks::default(),
        );
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!(r.x[0] <= 1.0, "x0 = {} left the box", r.x[0]);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-9 && (r.x[1] - 1.0).abs() < 1e-9,
            "{:?}",
            r.x
        );
    }

    /// The honesty half of the bound case: the only root is outside the box,
    /// so the method must say so rather than report the clipped point.
    /// `x0 = 2`, `x1 = 0` over `x0 ≤ 1` has no root in the box; the
    /// least-squares minimum on the box is `(1, 0)` with `|c| = 1`.
    #[test]
    fn a_root_outside_the_box_is_not_reported_as_converged() {
        let prob = dense2([-10.0, -10.0], [1.0, 10.0]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] - 2.0;
                c[1] = x[1];
            },
            j: |_x: &[Number], v: &mut [Number]| v.copy_from_slice(&[1.0, 0.0, 0.0, 1.0]),
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[0.0, 0.5],
            &strict(),
            200,
            backend(),
            LmHooks::default(),
        );
        assert_eq!(r.status, LmStatus::Stationary, "{r:?}");
        assert!(r.x[0] <= 1.0);
        assert!(
            (r.c[0] + 1.0).abs() < 1e-8,
            "should stall at the box's nearest point: {r:?}"
        );
    }

    /// `degenerate_start_infeasible`'s shape: `x0² + x1² = −25`, `x0·x1 = 9`
    /// from the origin, where `J = 0`. No step leaves a point whose gradient
    /// is exactly zero, so the method must stop at once on stationarity —
    /// not report the origin, and not spend its budget halving `Δ`.
    #[test]
    fn a_zero_jacobian_start_stops_on_stationarity_at_once() {
        let prob = dense2([Number::NEG_INFINITY; 2], [Number::INFINITY; 2]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] * x[0] + x[1] * x[1] + 25.0;
                c[1] = x[0] * x[1] - 9.0;
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], x[1], x[0]]);
            },
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[0.0, 0.0],
            &strict(),
            200,
            backend(),
            LmHooks::default(),
        );
        assert_eq!(r.status, LmStatus::Stationary, "{r:?}");
        assert_eq!(r.iterations, 0);
    }

    /// Rank-deficient everywhere on the solution set: `x0 = x1` and
    /// `x0² = x1²` share the root set `{x0 = x1}`, and the Jacobian
    /// `[[1, −1], [2x0, −2x1]]` is singular at every root. Gauss–Newton's
    /// system is singular there; the LM's `λ` keeps the augmented factor
    /// well-posed and the method still drives `‖c‖` to zero (the
    /// error-bound regime of Yamashita & Fukushima). The root it lands on
    /// is not unique, so only feasibility is asserted.
    #[test]
    fn converges_on_a_rank_deficient_non_isolated_root_set() {
        let prob = dense2([Number::NEG_INFINITY; 2], [Number::INFINITY; 2]);
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] - x[1];
                c[1] = x[0] * x[0] - x[1] * x[1];
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[1.0, -1.0, 2.0 * x[0], -2.0 * x[1]]);
            },
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[3.0, -1.0],
            &strict(),
            200,
            backend(),
            LmHooks::default(),
        );
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!((r.x[0] - r.x[1]).abs() < 1e-10, "{:?}", r.x);
    }

    /// The same deficiency, structurally: a duplicated row (`J` has rank 1
    /// at every point) with duplicate triplets in the pattern, which must be
    /// summed rather than double-placed in the augmented system.
    #[test]
    fn duplicate_triplets_are_summed_and_a_repeated_row_is_tolerated() {
        let prob = LmProblem {
            n: 2,
            m: 2,
            // Row 0 lists (0, 0) twice, half the coefficient each time.
            jac_rows: vec![0, 0, 0, 1, 1],
            jac_cols: vec![0, 0, 1, 0, 1],
            lb: vec![Number::NEG_INFINITY; 2],
            ub: vec![Number::INFINITY; 2],
        };
        let mut sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] + x[1] - 2.0;
                c[1] = 2.0 * (x[0] + x[1] - 2.0);
            },
            j: |_x: &[Number], v: &mut [Number]| v.copy_from_slice(&[0.5, 0.5, 1.0, 2.0, 2.0]),
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[5.0, 7.0],
            &strict(),
            200,
            backend(),
            LmHooks::default(),
        );
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert!((r.x[0] + r.x[1] - 2.0).abs() < 1e-10);
    }

    /// The bound-root system of `converges_to_a_root_on_an_active_bound`,
    /// shared by the budget and callback tests below so each differs from a
    /// converging run in exactly the hook it exercises.
    fn bound_root_system() -> (
        LmProblem,
        Sys<impl FnMut(&[Number], &mut [Number]), impl FnMut(&[Number], &mut [Number])>,
    ) {
        let prob = dense2([-10.0, -10.0], [1.0, 10.0]);
        let sys = Sys {
            f: |x: &[Number], c: &mut [Number]| {
                c[0] = x[0] * x[0] + x[1] * x[1] - 2.0;
                c[1] = x[0] - x[1];
            },
            j: |x: &[Number], v: &mut [Number]| {
                v.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], 1.0, -1.0]);
            },
        };
        (prob, sys)
    }

    /// An exhausted time budget stops the method before its first step, and
    /// says which budget. `Deadline::new(0, ·)` is exceeded on its first
    /// read whatever the clock resolution (`elapsed ≥ 0`), so this is
    /// deterministic. Mutation-checked: with the `time_exceeded` checks
    /// removed from `bounded_levenberg_marquardt` both arms run to
    /// `Converged` and the test fails.
    #[test]
    fn an_exhausted_time_budget_stops_the_method_and_names_the_budget() {
        use pounce_common::timing::Deadline;
        for (deadline, want) in [
            (Deadline::new(0.0, 1e6), LmStatus::WallTimeExceeded),
            (Deadline::new(1e6, 0.0), LmStatus::CpuTimeExceeded),
        ] {
            let (prob, mut sys) = bound_root_system();
            let r = bounded_levenberg_marquardt(
                &mut sys,
                &prob,
                &[0.2, 3.0],
                &strict(),
                200,
                backend(),
                LmHooks {
                    deadline: Some(deadline),
                    on_iterate: None,
                },
            );
            assert_eq!(r.status, want, "{r:?}");
            assert_eq!(r.iterations, 0, "{r:?}");
        }
    }

    /// The clock is read after the convergence test, so a start that is
    /// already a root is reported as one even with no time left — the same
    /// order as the IPM's check. Mutation-checked: moving the iterate-0
    /// clock check above `test.passes` turns this `WallTimeExceeded`.
    #[test]
    fn a_root_is_not_discarded_for_being_late() {
        let (prob, mut sys) = bound_root_system();
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[1.0, 1.0],
            &strict(),
            200,
            backend(),
            LmHooks {
                deadline: Some(pounce_common::timing::Deadline::new(0.0, 0.0)),
                on_iterate: None,
            },
        );
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert_eq!(r.iterations, 0);
    }

    /// The observer sees every iterate from the start on, in order, and a
    /// `false` stops the method on that iterate. Mutation-checked: ignoring
    /// the observer's return value runs to `Converged` and fails here. (Every
    /// step of this system is accepted, so the test cannot tell an observer
    /// fired on every iterate from one fired on accepted steps only; that
    /// mutation was tried and survives.)
    #[test]
    fn the_observer_sees_every_iterate_and_can_stop_the_method() {
        let (prob, mut sys) = bound_root_system();
        let mut seen: Vec<usize> = Vec::new();
        let mut obs = |it: &LmIterate<'_>| {
            seen.push(it.iter);
            it.iter < 2
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[0.2, 3.0],
            &strict(),
            200,
            backend(),
            LmHooks {
                deadline: None,
                on_iterate: Some(&mut obs),
            },
        );
        assert_eq!(r.status, LmStatus::UserRequestedStop, "{r:?}");
        assert_eq!(r.iterations, 2);
        assert_eq!(seen, [0, 1, 2]);

        // Unstopped, it sees iterates 0..=iterations: one per iteration
        // plus the start, the IPM's record convention.
        let (prob, mut sys) = bound_root_system();
        let mut seen: Vec<usize> = Vec::new();
        let mut obs = |it: &LmIterate<'_>| {
            seen.push(it.iter);
            true
        };
        let r = bounded_levenberg_marquardt(
            &mut sys,
            &prob,
            &[0.2, 3.0],
            &strict(),
            200,
            backend(),
            LmHooks {
                deadline: None,
                on_iterate: Some(&mut obs),
            },
        );
        assert_eq!(r.status, LmStatus::Converged, "{r:?}");
        assert_eq!(seen, (0..=r.iterations).collect::<Vec<_>>());
    }

    /// A 3-variable TNLP for [`run`]: `x1` fixed at 3 in the **middle**
    /// slot (so a full/free index mix-up lands a value in the wrong place),
    /// `x0 ≤ 2` active at the root `(2, 3, 2.5)`, rows `x0·x2 − x1 = 2` and
    /// `x0 + x1/2 − x2 = 1`, objective the constant 5. Counts what the
    /// stage does to it.
    struct Fxm {
        finalized: Vec<Vec<Number>>,
        callbacks: usize,
        /// Callback returns `false` from this iterate on.
        stop_at: Option<Index>,
    }

    impl Fxm {
        fn new() -> Self {
            Self {
                finalized: Vec::new(),
                callbacks: 0,
                stop_at: None,
            }
        }
    }

    impl TNLP for Fxm {
        fn get_nlp_info(&mut self) -> Option<pounce_nlp::tnlp::NlpInfo> {
            Some(pounce_nlp::tnlp::NlpInfo {
                n: 3,
                m: 2,
                nnz_jac_g: 6,
                nnz_h_lag: 0,
                index_style: IndexStyle::C,
            })
        }
        fn get_bounds_info(&mut self, b: BoundsInfo<'_>) -> bool {
            b.x_l.copy_from_slice(&[-1e20, 3.0, -1e20]);
            b.x_u.copy_from_slice(&[2.0, 3.0, 1e20]);
            b.g_l.copy_from_slice(&[2.0, 1.0]);
            b.g_u.copy_from_slice(&[2.0, 1.0]);
            true
        }
        fn get_starting_point(&mut self, sp: StartingPoint<'_>) -> bool {
            sp.x.copy_from_slice(&[1.5, 7.0, 1.5]);
            true
        }
        fn eval_f(&mut self, _x: &[Number], _new_x: bool) -> Option<Number> {
            Some(5.0)
        }
        fn eval_grad_f(&mut self, _x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g.iter_mut().for_each(|v| *v = 0.0);
            true
        }
        fn eval_g(&mut self, x: &[Number], _new_x: bool, g: &mut [Number]) -> bool {
            g[0] = x[0] * x[2] - x[1];
            g[1] = x[0] + 0.5 * x[1] - x[2];
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
                    irow.copy_from_slice(&[0, 0, 0, 1, 1, 1]);
                    jcol.copy_from_slice(&[0, 1, 2, 0, 1, 2]);
                }
                SparsityRequest::Values { values } => {
                    let x = x.unwrap();
                    values.copy_from_slice(&[x[2], -1.0, x[0], 1.0, 0.5, -1.0]);
                }
            }
            true
        }
        fn intermediate_callback(
            &mut self,
            stats: pounce_nlp::tnlp::IterStats,
            _d: &IpoptData,
            _q: &IpoptCq,
        ) -> bool {
            self.callbacks += 1;
            self.stop_at.is_none_or(|k| stats.iter < k)
        }
        fn finalize_solution(&mut self, sol: Solution<'_>, _d: &IpoptData, _q: &IpoptCq) {
            self.finalized.push(sol.x.to_vec());
        }
    }

    fn run_fxm(
        fxm: Fxm,
        opts: &[(&str, Number)],
        capture: bool,
    ) -> (SquareSystemOutcome, Rc<RefCell<Fxm>>) {
        let mut options = OptionsList::new();
        for &(tag, v) in opts {
            options.set_numeric_value(tag, v, true, true).unwrap();
        }
        let cell = Rc::new(RefCell::new(fxm));
        let tnlp: Rc<RefCell<dyn TNLP>> = cell.clone();
        let plan = probe(&tnlp, &options).expect("Fxm is in the class");
        let out = run(
            &plan,
            &tnlp,
            &options,
            backend(),
            RunOptions {
                capture_iterations: capture,
            },
        );
        (out, cell)
    }

    /// The baseline the two tests below each break one hook of: promoted,
    /// finalized once, with the fixed variable in its own (middle) slot, a
    /// callback per iterate, `iterations.len() == iteration_count + 1`, and
    /// `final_scaled_objective` carrying `obj_scaling_factor`.
    #[test]
    fn run_promotes_and_reports_in_the_ipm_conventions() {
        let (out, fxm) = run_fxm(Fxm::new(), &[("obj_scaling_factor", 2.0)], true);
        assert!(out.accepted, "{}", out.reason);
        let fxm = fxm.borrow();
        assert_eq!(fxm.finalized.len(), 1);
        let x = &fxm.finalized[0];
        assert_eq!(x[1], 3.0, "the fixed variable keeps its slot: {x:?}");
        assert!(
            (x[0] - 2.0).abs() < 1e-12 && (x[2] - 2.5).abs() < 1e-12,
            "{x:?}"
        );
        let st = &out.statistics;
        assert_eq!(st.iterations.len(), st.iteration_count as usize + 1);
        assert_eq!(fxm.callbacks, st.iteration_count as usize + 1);
        assert_eq!(st.final_objective, 5.0);
        assert_eq!(st.final_scaled_objective, 10.0);
    }

    /// `max_wall_time` reaches the LM through [`run`], and running out
    /// means no promotion and no `finalize_solution`. `0` is exceeded on the
    /// first read of the clock, so this is deterministic. Mutation-checked:
    /// passing `deadline: None` in `run` promotes and the test fails.
    #[test]
    fn run_honours_max_wall_time_and_does_not_promote() {
        for (tag, want) in [
            ("max_wall_time", LmStatus::WallTimeExceeded),
            ("max_cpu_time", LmStatus::CpuTimeExceeded),
        ] {
            let (out, fxm) = run_fxm(Fxm::new(), &[(tag, 0.0)], true);
            assert!(!out.accepted, "{tag}");
            assert_eq!(out.lm.status, want, "{tag}");
            assert!(out.reason.contains(tag), "{tag}: {}", out.reason);
            assert!(fxm.borrow().finalized.is_empty(), "{tag}");
        }
    }

    /// The TNLP's intermediate callback is fired at every iterate and its
    /// stop request is honoured: no promotion, no `finalize_solution`.
    /// Mutation-checked: discarding the callback's return value in `run`'s
    /// observer promotes and the test fails.
    #[test]
    fn run_honours_an_intermediate_callback_stop() {
        let mut fxm = Fxm::new();
        fxm.stop_at = Some(1);
        let (out, fxm) = run_fxm(fxm, &[], false);
        assert!(!out.accepted);
        assert_eq!(out.lm.status, LmStatus::UserRequestedStop);
        assert_eq!(out.lm.iterations, 1);
        assert_eq!(fxm.borrow().callbacks, 2, "iterates 0 and 1");
        assert!(fxm.borrow().finalized.is_empty());
    }

    /// The scaled test is the NLP arm's: a row's violation is judged after
    /// its `s_i`, and the unscaled violation separately against
    /// `constr_viol_tol`. A NaN never passes.
    #[test]
    fn feasibility_test_applies_both_halves_and_rejects_nan() {
        let t = FeasibilityTest {
            row_scale: vec![1e-3, 1.0],
            tol: 1e-8,
            constr_viol_tol: 1e-4,
        };
        assert!(t.passes(&[5e-6, 1e-9]), "5e-6 · 1e-3 = 5e-9 ≤ tol");
        assert!(
            !t.passes(&[5e-4, 0.0]),
            "scaled passes but unscaled 5e-4 > constr_viol_tol"
        );
        assert!(!t.passes(&[0.0, 2e-8]), "unscaled row over tol");
        assert!(!t.passes(&[Number::NAN, 0.0]));
    }
}
