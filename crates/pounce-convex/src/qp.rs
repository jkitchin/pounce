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

use crate::cones::ConeSpec;

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

    /// True when the variable box admits **no finite point**: a *present*
    /// lower bound at `+∞` (`lb ≥ BOUND_INF`) or a *present* upper bound at
    /// `−∞` (`ub ≤ −BOUND_INF`). Such a bound is genuinely primal-infeasible
    /// (gh #295) — the same class as a finite reversed box (`lb > ub`).
    ///
    /// Deliberately distinct from an *absent* bound (`lb ≤ −BOUND_INF` /
    /// `ub ≥ +BOUND_INF`, the normal one-sided `±∞` encoding), which is
    /// feasible and left to the solver. The interior-point setup
    /// ([`crate::ipm`]'s `expand_bounds`) is sign-agnostic — it tests only a
    /// bound's *presence* by magnitude (`ub < BOUND_INF`, `lb > -BOUND_INF`)
    /// — so without this screen a `+∞` lower / `−∞` upper bound is silently
    /// mishandled and a point violating it can be reported `Optimal`. Every
    /// solve entry point (and presolve) consults this so the rejection holds
    /// on every surface — the raw `_pounce` bindings and direct Rust callers
    /// included — mirroring the Python-layer `_validate` guard (#275 / #291).
    pub(crate) fn bounds_admit_no_point(&self) -> bool {
        (0..self.n).any(|i| self.lb_of(i) >= BOUND_INF || self.ub_of(i) <= -BOUND_INF)
    }

    /// True when some variable's box is **reversed** (`lb > ub`), so the
    /// feasible set is empty for a reason that needs no arithmetic to see.
    ///
    /// Kept separate from [`Self::bounds_admit_no_point`] on purpose: this is
    /// the cheap *exact* test, and what a crossing of a given width should
    /// mean is [`screen_variable_box`]'s decision, not this predicate's. Do
    /// not fold the two together — presolve consults
    /// `bounds_admit_no_point` on every pass, and its own bound tightening
    /// legitimately leaves a box crossed by up to `BOUND_FEAS_TOL`, so an
    /// exact reversal test there would turn a tolerated near-crossing into a
    /// false infeasibility claim.
    pub(crate) fn bounds_are_reversed(&self) -> bool {
        (0..self.n).any(|i| self.lb_of(i) > self.ub_of(i))
    }

    /// A copy of this problem with the **objective** data scaled by `factor`
    /// (`P ← factor·P`, `c ← factor·c`); constraints, bounds, and the feasible
    /// set are untouched. Scaling the objective by a positive constant leaves
    /// the minimizer `x*` unchanged, so this is used to renormalize a
    /// badly-scaled objective before an interior-point solve and map the result
    /// back afterward (the dual multipliers and objective value scale by the
    /// same `factor`). See the HSDE cost-normalization in
    /// [`crate::ipm`] (gh #286).
    pub(crate) fn scaled_objective(&self, factor: f64) -> QpProblem {
        QpProblem {
            n: self.n,
            p_lower: self
                .p_lower
                .iter()
                .map(|t| Triplet::new(t.row, t.col, t.val * factor))
                .collect(),
            c: self.c.iter().map(|v| v * factor).collect(),
            a: self.a.clone(),
            b: self.b.clone(),
            g: self.g.clone(),
            h: self.h.clone(),
            lb: self.lb.clone(),
            ub: self.ub.clone(),
        }
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

/// Verdict on the variable box, decided before any solve is attempted.
///
/// Public because [`ActiveSetQp`] is: a caller translating and solving the
/// native problem itself is a solve entry point, and this is the one step
/// ahead of the translation that is not optional (gh #769). See
/// [`screen_variable_box`].
///
/// [`ActiveSetQp`]: crate::ActiveSetQp
#[derive(Debug, Clone)]
pub enum BoxScreen {
    /// Every `[lb, ub]` is a non-empty interval; solve the problem as posed.
    Feasible,
    /// Some variable's box is empty by inspection — report `PrimalInfeasible`.
    Empty,
    /// Some box was crossed by no more than [`CROSSED_BOX_TOL`]; solve this
    /// repaired copy, in which each such variable is fixed at its midpoint.
    Snapped(QpProblem),
}

/// Widest box crossing (`lb − ub`) treated as a numerical artifact rather than
/// an empty feasible set.
///
/// Matched to presolve's own `BOUND_FEAS_TOL`, which is the number that decides
/// what can reach a solve entry point: bound tightening applies an update only
/// when it improves a bound by more than that tolerance and declares
/// infeasibility only when `tlb > tub + BOUND_FEAS_TOL`, so a *reduced* problem
/// handed on from presolve can carry a box crossed by up to exactly this much —
/// with presolve having already ruled that crossing tolerable. Calling it
/// infeasible downstream would overturn that ruling on the strength of
/// arithmetic presolve had already discounted.
///
/// It is also where the interior-point method drew this line of its own accord,
/// which is why snapping here is a repair rather than a change of answer.
/// Measured on `min ½x² s.t. 0 ≤ x ≤ −gap`, before this screen existed:
///
/// | crossing        | IPM verdict                       |
/// |-----------------|-----------------------------------|
/// | `1e-14 … 1e-9`  | `Optimal`, at the box midpoint    |
/// | `1e-8 … 1e-7`   | `NumericalFailure`, `x = NaN`     |
/// | `1e-6 … 1e0`    | `PrimalInfeasible`                |
///
/// So the screen preserves the top row (that midpoint is exactly what
/// [`BoxScreen::Snapped`] hands back) and the bottom row, and replaces the
/// middle one — a `NaN` iterate from a box the IPM's own arithmetic could not
/// resolve either way — with the verdict the rows on both sides of it imply.
pub(crate) const CROSSED_BOX_TOL: f64 = 1e-9;

/// Decide what an ill-formed variable box means before a solver sees it.
///
/// Two classes of box admit no point, and neither survives contact with a
/// solver intact:
///
/// * **Impossible** — a *present* `+∞` lower or `−∞` upper bound (gh #295).
///   The bound-presence tests are sign-agnostic, so such a bound was dropped as
///   if *absent* and a point violating it came back `Optimal`.
/// * **Reversed** — `lb > ub` (gh #491). On the active-set path this **aborted
///   the process**: the engine's `validate` rejects `xl > xu` with
///   `QpError::InvertedBounds`, so the driver's first attempts failed and it
///   fell through to its simplex-seeded last resort, where the seed was clamped
///   into the inverted interval and `f64::clamp` panicked (`min > max`). Across
///   the PyO3 boundary that surfaced as a `pyo3_runtime.PanicException`, which
///   derives from `BaseException` and so tore down callers that had defensively
///   wrapped the solve in `except Exception`.
///
/// A wide crossing is [`BoxScreen::Empty`], and saying so needs no arithmetic —
/// which matters because the active-set driver otherwise refuses to report
/// `PrimalInfeasible` without a certificate it re-derived itself. The
/// certificate for this class *is* the bound pair: `x_i ≥ lb_i > ub_i ≥ x_i`
/// has no solution, so there is nothing numerical to prove or to demote.
///
/// A hairline crossing is repaired instead, for the reason given on
/// [`CROSSED_BOX_TOL`]: it is the residue of a tolerance decision made upstream,
/// not a statement that the problem is infeasible.
///
/// **Every solve entry point runs this**, both engines' alike — which is the
/// point. The two methods are alternatives for the same problem, so an
/// input-domain question like "is this box empty?" must not be answered by
/// whichever engine happens to be selected. Screening in the core rather than in
/// the Python `_validate` pass also keeps the answer the same on the raw
/// `_pounce` bindings, the CLI, and direct Rust callers.
///
/// "Every solve entry point runs this" now includes one outside this crate.
/// [`ActiveSetQp::from_convex`] translates the problem *as posed* — it is a
/// translation, not a repair, and a `from_convex` that quietly snapped a
/// crossed box would hide the decision — so an external caller driving the
/// engine directly runs this first, exactly as
/// [`solve_qp_active_set`](crate::solve_qp_active_set) and
/// [`ActiveSetSession`] do. Skipping it is not a missing optimization: an
/// `Empty` box reaches `pounce_qp`'s `validate` as `InvertedBounds` — a hard
/// `Err` where the driver reports a certified `PrimalInfeasible` — and the
/// *impossible* class, a present `+∞` lower bound, is dropped as if absent and
/// comes back `Optimal` at a point violating it. That is the wrong answer,
/// silently, which is why this is not left for callers to rediscover
/// (gh #295, gh #491).
///
/// [`ActiveSetQp::from_convex`]: crate::ActiveSetQp::from_convex
/// [`ActiveSetSession`]: crate::ActiveSetSession
pub fn screen_variable_box(prob: &QpProblem) -> BoxScreen {
    if prob.bounds_admit_no_point() {
        return BoxScreen::Empty;
    }
    if !prob.bounds_are_reversed() {
        return BoxScreen::Feasible;
    }
    if (0..prob.n).any(|i| prob.lb_of(i) > prob.ub_of(i) + CROSSED_BOX_TOL) {
        return BoxScreen::Empty;
    }
    // Hairline: fix each crossed variable at its midpoint. Both bound vectors
    // are necessarily length `n` here — an absent bound reads as `∓∞` and
    // cannot be the crossed side — but index defensively anyway, since a
    // partially-filled vector must not silently drop the repair and hand the
    // solver the inverted box after all.
    let mut repaired = prob.clone();
    for i in 0..prob.n {
        let (l, u) = (prob.lb_of(i), prob.ub_of(i));
        if l > u {
            let mid = 0.5 * (l + u);
            if let (Some(lo), Some(hi)) = (repaired.lb.get_mut(i), repaired.ub.get_mut(i)) {
                *lo = mid;
                *hi = mid;
            } else {
                return BoxScreen::Empty;
            }
        }
    }
    BoxScreen::Snapped(repaired)
}

/// Termination status of an IPM solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// Converged: KKT residuals and duality gap below tolerance.
    Optimal,
    /// Solved to *reduced* accuracy: the KKT factorization or a back-solve
    /// broke down (or the iteration cap / a stalled step was hit) while the
    /// best KKT residual reached was already small — within `~1e3·tol` for the
    /// symmetric HSDE driver, `√tol` for the non-symmetric one — but never as
    /// tight as the clean `tol` convergence test. The returned iterate is
    /// usable but its residual sits above `tol`. This is the analogue of
    /// ECOS/Clarabel's `*_INACC` ("solved to inaccurate") and Ipopt's
    /// "Solved To Acceptable Level": callers that need full accuracy (e.g.
    /// sensitivity, SOS exactness certification) should treat it as *not*
    /// [`Optimal`](Self::Optimal). Previously these cases were reported as a
    /// bare `Optimal`, indistinguishable from a genuinely-converged solve.
    /// (Code review 2026-06 item M20.)
    OptimalInaccurate,
    /// Primal infeasible: no `x` satisfies `Ax = b, Gx ≤ h`. A Farkas
    /// certificate `(y, z ≥ 0)` with `Aᵀy + Gᵀz ≈ 0` and `bᵀy + hᵀz < 0`
    /// was detected and verified.
    PrimalInfeasible,
    /// Dual infeasible / unbounded below: a recession direction `d` with
    /// `Pd ≈ 0, Ad = 0, Gd ≤ 0, cᵀd < 0` was detected and verified.
    DualInfeasible,
    /// Iteration limit reached before convergence.
    IterationLimit,
    /// The solve-wide wall-clock budget expired. The latest finite iterate is
    /// returned when one is available.
    TimeLimit,
    /// The KKT factorization failed (e.g. structurally singular system).
    NumericalFailure,
}

/// Terminal status for a mid-iteration breakdown (factorization / back-solve
/// failure, or a non-positive step). When the best KKT residual reached so far
/// is already within the reduced-accuracy band (`near_opt`), the iterate is
/// usable and we report [`QpStatus::OptimalInaccurate`] rather than discarding
/// it as a [`QpStatus::NumericalFailure`]. Centralized so the symmetric and
/// non-symmetric HSDE drivers cannot drift apart, and so the "a near-`tol`
/// breakdown is *not* a bare `Optimal`" rule is unit-testable. (Code review
/// 2026-06 item M20.)
pub(crate) fn breakdown_status(near_opt: bool) -> QpStatus {
    if near_opt {
        QpStatus::OptimalInaccurate
    } else {
        QpStatus::NumericalFailure
    }
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
    /// Objective `½ xᵀP x + cᵀx` at the start of this iteration, in the
    /// **original problem's coordinates** — consistent with
    /// [`QpSolution::obj`]. When the solve was Ruiz-equilibrated (the default
    /// direct path), the inner iteration records the scaled objective and the
    /// unscaling pass divides it by the cost-scaling factor σ to recover this
    /// value (see [`crate::equilibrate::Scaling::unscale_solution`]).
    pub objective: f64,
    /// Primal infeasibility `max(‖Ax − b‖∞, ‖(Gx + s − h)‖∞)`.
    ///
    /// On an equilibrated solve this (and the two fields below) is reported in
    /// the solver's **internal scaled coordinates**, not the original problem's:
    /// an ∞-norm of a per-row/per-column diagonally-scaled residual has no exact
    /// scalar inverse, so unlike [`Self::objective`] it cannot be mapped back
    /// exactly. It is a monotone convergence indicator that vanishes at the
    /// optimum in either coordinate system.
    pub primal_infeasibility: f64,
    /// Dual infeasibility `‖Px + c + Aᵀy + Gᵀz‖∞` (scaled coordinates on an
    /// equilibrated solve; see [`Self::primal_infeasibility`]).
    pub dual_infeasibility: f64,
    /// Duality measure `μ = ⟨s, z⟩ / degree` (scaled coordinates on an
    /// equilibrated solve; see [`Self::primal_infeasibility`]).
    pub mu: f64,
    /// Primal step length taken this iteration.
    pub alpha_primal: f64,
    /// Dual step length taken this iteration.
    pub alpha_dual: f64,
}

/// How far `s` is outside the cone `spec`, in the cone's own defining
/// inequality (`0` when `s ∈ K`). Not a Euclidean distance — a violation
/// magnitude, which is what a convergence report wants and what vanishes
/// exactly at a feasible point.
///
/// * orthant — `max(0, −sᵢ)`
/// * second-order — `max(0, ‖s₁‖ − s₀)`
/// * PSD — `max(0, −λ_min(smat s))`
/// * exponential — `max(0, −ψ)` for `ψ = y·log(z/y) − x`, plus `y, z ≥ 0`
/// * power — `max(0, |x| − y^α z^{1−α})`, plus `y, z ≥ 0`
fn cone_violation(spec: &ConeSpec, s: &[f64]) -> f64 {
    let neg = |v: f64| (-v).max(0.0);
    match spec {
        ConeSpec::Nonneg(_) => s.iter().fold(0.0_f64, |m, &si| m.max(neg(si))),
        ConeSpec::SecondOrder(_) => {
            let tail = s[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
            (tail - s[0]).max(0.0)
        }
        ConeSpec::Psd(k) => neg(crate::cones::PsdCone::new(*k).min_eig(s)),
        ConeSpec::Exponential => {
            let (x, y, z) = (s[0], s[1], s[2]);
            let bounds = neg(y).max(neg(z));
            if y <= 0.0 || z <= 0.0 {
                // ψ is undefined here; the sign violation is the whole story.
                return bounds.max(0.0_f64.max(x));
            }
            bounds.max(neg(y * (z / y).ln() - x))
        }
        ConeSpec::Power(alpha) => {
            let (x, y, z) = (s[0], s[1], s[2]);
            let bounds = neg(y).max(neg(z));
            if y < 0.0 || z < 0.0 {
                return bounds.max(x.abs());
            }
            bounds.max((x.abs() - y.powf(*alpha) * z.powf(1.0 - *alpha)).max(0.0))
        }
    }
}

/// Final KKT residuals of a [`QpSolution`] with respect to its [`QpProblem`]
/// — the convergence quantities a caller (e.g. a solve report or benchmark
/// harness) needs but that aren't otherwise carried on the solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpResiduals {
    /// Primal infeasibility: `max(|Ax − b|, max(0, Gx − h), bound violations)`.
    pub primal_infeasibility: f64,
    /// The variable-box half of [`Self::primal_infeasibility`] on its own:
    /// `max_i max(lb_i − x_i, x_i − ub_i, 0)`, how far the returned `x` sits
    /// outside the box of the problem it is measured against.
    ///
    /// Separated because the console summary reports it as its own line —
    /// Ipopt's `Variable bound violation` — and the aggregate above cannot
    /// answer that question: a row violation and a box violation are
    /// indistinguishable once maxed together, and on this arm they have
    /// different causes. Measured against `prob`, so pairing it with a
    /// re-extraction at `BoundRelax::NONE`
    /// (`pounce_cli::qp_extract::declared_residuals_qp`) is what makes it the
    /// violation of the box the *caller* declared rather than the widened one
    /// the solver was handed.
    pub bound_violation: f64,
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

/// How many times the stopping rule's finite-precision cap the reported (and
/// adjudicated) residuals are read above; see `kkt_residuals_above_floor`.
const REPORT_FLOOR_FACTOR: f64 = 4.0;

/// Smallest objective unit the stopping rule normalizes by.
pub(crate) const COST_UNIT_FLOOR: f64 = 1e-15;

/// The objective's own unit `max(‖P‖∞, ‖c‖∞)`.
pub(crate) fn cost_unit(prob: &QpProblem) -> f64 {
    let nc = prob.c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    prob.p_lower.iter().fold(nc, |m, t| m.max(t.val.abs()))
}

/// `(unit_d, unit_g)`: the factors the stopping rule multiplies `tol` by for
/// the dual residual and for the duality gap / complementarity (gh#984).
///
/// Both are in the units of `(P, c)`, so an absolute `tol` on them is a
/// statement about the caller's choice of units: they tighten `tol` with the
/// objective's unit when it is below 1 (`min(unit, 1)`, floored). A large unit
/// is deliberately *not* used to loosen anything -- that is `hsde_cost_scale`'s
/// regime, the presolve amplifies slack in the dual residual, and a stiff `P`
/// makes a unit-scaled tolerance loose in `x` (gh#846). A zero objective has no
/// unit: both are 1.
pub(crate) fn objective_units(prob: &QpProblem) -> (f64, f64) {
    let u = cost_unit(prob);
    if u > 0.0 {
        let g = u.clamp(COST_UNIT_FLOOR, 1.0);
        (g, g)
    } else {
        (1.0, 1.0)
    }
}

/// The units [`QpSolution::kkt_residuals_above_floor`] divides stationarity
/// and complementarity by: the objective's unit, up as well as down.
fn report_units(prob: &QpProblem) -> (f64, f64) {
    let u = cost_unit(prob);
    if u > 0.0 {
        let g = u.max(COST_UNIT_FLOOR);
        (g, g)
    } else {
        (1.0, 1.0)
    }
}

/// A tiny-curvature warning fires when `‖P‖∞` is more than this many orders of
/// magnitude below the rest of the problem data. Six orders is comfortably past
/// where a well-scaled convex QP sits (curvature commensurate with the linear
/// and constraint coefficients) yet well above the gh #293 regime (`‖P‖`
/// 8–20 orders below `‖c‖`), so a healthy hard problem is never blamed on
/// curvature and an ill-scaled one always is.
const CURV_DATA_RATIO: f64 = 1e-6;

impl QpSolution {
    /// Diagnose whether an *unconverged or possibly-spurious* status is likely
    /// caused by tiny objective curvature relative to the rest of the problem
    /// data — the ill-scaling regime of gh #293. There an interior-point solve
    /// can exhaust its budget (`IterationLimit` / `OptimalInaccurate`) short of
    /// a far-off optimum, or, in the machine-epsilon tail, return a
    /// scaling-artifact certificate. Returns a human-readable warning when the
    /// status is one of those *and* the curvature is tiny relative to the data;
    /// otherwise `None` — so a caller can surface it without ever second-guessing
    /// a clean `Optimal` (including one the solver recovered via its own
    /// equilibrated retry) or an exact infeasibility certificate.
    ///
    /// This is the naive-caller guardrail for the residual cases no driver can
    /// converge at the default budget (e.g. a uniformly tiny Hessian coupled
    /// through an equality constraint): the status is already honest, and this
    /// says *why*, with an actionable remedy, instead of leaving a truncated
    /// objective to be mistaken for the optimum.
    pub fn scaling_diagnostic(&self, prob: &QpProblem) -> Option<String> {
        let suspect = matches!(
            self.status,
            QpStatus::IterationLimit
                | QpStatus::OptimalInaccurate
                | QpStatus::NumericalFailure
                | QpStatus::DualInfeasible
        );
        if !suspect {
            return None;
        }
        let maxabs_t = |it: &[Triplet]| it.iter().fold(0.0f64, |m, t| m.max(t.val.abs()));
        let maxabs_v = |v: &[f64]| v.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        let p_norm = maxabs_t(&prob.p_lower);
        if p_norm == 0.0 {
            // A pure LP has no curvature; its unboundedness/limits are not a
            // tiny-Hessian artifact, so this diagnostic does not apply.
            return None;
        }
        let data = maxabs_v(&prob.c)
            .max(maxabs_t(&prob.a))
            .max(maxabs_t(&prob.g))
            .max(1.0);
        if p_norm >= data * CURV_DATA_RATIO {
            return None;
        }
        Some(format!(
            "scaling warning: objective curvature ‖P‖∞ = {p_norm:.1e} is tiny \
             relative to the problem data (‖c,A,G‖∞ ≈ {data:.1e}); the {:?} \
             result may be inaccurate. Rescale the objective (e.g. divide P and \
             c by ‖P‖∞) or cross-check with a reference solver.",
            self.status
        ))
    }

    /// Recompute the final KKT residuals of this solution against `prob`.
    ///
    /// Uses the convex solver's standard-form conventions —
    /// `min ½xᵀPx + cᵀx s.t. Ax = b, Gx ≤ h, lb ≤ x ≤ ub`, with equality dual
    /// `y`, inequality dual `z ≥ 0`, and bound duals `z_lb, z_ub ≥ 0`. The
    /// stationarity residual is `∇ₓL = Px + c + Aᵀy + Gᵀz − z_lb + z_ub`, the
    /// `−z_lb + z_ub` matching how variable bounds expand into `G`-rows and
    /// split back into the bound multipliers.
    pub fn kkt_residuals(&self, prob: &QpProblem) -> QpResiduals {
        self.kkt_residuals_inner(prob, None)
    }

    /// Recompute the final KKT residuals of a **conic** solve, where the
    /// inequality block is not `Gx ≤ h` row-by-row but `h − Gx ∈ K` for the
    /// product cone `cones` (the form [`crate::solve_socp_ipm`] consumes).
    ///
    /// [`Self::kkt_residuals`] assumes the nonnegative orthant, so on a solve
    /// with second-order / PSD / exponential / power blocks it reports garbage:
    /// individual rows of a converged SOC block legitimately have `Gx > h`
    /// (only the *cone* membership `s₀ ≥ ‖s₁‖` must hold) and `zᵢsᵢ ≠ 0` (only
    /// the *block* product `⟨s, z⟩` vanishes). Feeding those per-row numbers to
    /// a convergence report made a feasible, optimal QCQP look badly infeasible
    /// (pounce#209). This variant measures each block with its own cone:
    /// membership violation for the primal residual, the block inner product
    /// for complementarity. Equalities, variable bounds and stationarity are
    /// unchanged.
    ///
    /// `cones` must cover `prob.m_ineq()` rows in order; any trailing rows it
    /// does not cover are treated as orthant rows.
    pub fn kkt_residuals_conic(&self, prob: &QpProblem, cones: &[ConeSpec]) -> QpResiduals {
        self.kkt_residuals_inner(prob, Some(cones))
    }

    /// [`Self::kkt_residuals`] with each residual read **above its own
    /// finite-precision floor** (gh#984 item 3), the measure a status of
    /// `Optimal` is judged on.
    ///
    /// An absolute residual cannot be driven below the rounding error of
    /// *evaluating* it, `~ε·(the magnitude of the terms it is a difference
    /// of)`. A variable shifted by `1e9` makes its row slack `h − Gx` a
    /// difference of `1e9`-sized numbers, quantized at `1.2e-7`, and a
    /// multiplier of `1` times that slack reads as complementarity `1.2e-7` at
    /// the exact optimum: `kkt_error > tol = 1e-8` is *unreachable* in doubles,
    /// not a sign of a bad point. Reporting that number beside `optimal` is
    /// either a false alarm or, if the verdict is made to follow it, a
    /// solver that can never succeed on a model with a large offset.
    ///
    /// So each residual *entry* is reduced by `4·REL_FLOOR_KAPPA·ε` times the
    /// magnitude of the terms **it** is a difference of, and floored at zero:
    /// a stationarity component by `|c_j| + Σ|P_jk x_k| + Σ|a_ij y_i| +
    /// Σ|g_ij z_i| + z_lb_j + z_ub_j`, an equality row by `|b_i| + Σ|a_ij x_j|`,
    /// an inequality row by `|h_i| + Σ|g_ij x_j|`, a bound by `|x_i| +
    /// |bound|`, a complementarity product by `z_i` times its slack's
    /// magnitude. The constant is the stopping rule's
    /// (`hsde::REL_FLOOR_KAPPA`), so a point the relative arm stopped on is
    /// read the way it was judged.
    ///
    /// Per entry, not per block (gh#984 review): the second pass used global
    /// norms -- `max|x_i|` over every boxed variable, `|cᵀx| + |hᵀz| + ...` for
    /// every product -- so one large term excused the whole model (`5.7e-5`,
    /// 500 ulps, on the item-3 model) and a well-scaled solve with raw
    /// `kkt_error 1.2e-9` reported `0.0`. Now a row of ordinary magnitude has a
    /// floor of `~1e-13` and reads as its raw residual.
    ///
    /// The stationarity and complementarity terms are then divided by the
    /// objective's unit exactly as the stopping rule scales `tol` by it
    /// ([`objective_units`]): they are in the units of `(P, c)`, so an absolute
    /// bound on them is a statement about the caller's choice of units. The
    /// stop rule, the `Optimal` verdict and this number are one measurement.
    ///
    /// Orthant/box problems only; a conic solve is returned unadjusted.
    #[allow(clippy::needless_range_loop)]
    pub fn kkt_residuals_above_floor(&self, prob: &QpProblem) -> QpResiduals {
        // Four times the stopping rule's cap: the loop judges the homogeneous
        // residuals, this recomputes them from the returned point, and the
        // two differ by a few ulps of the same scales.
        let k = REPORT_FLOOR_FACTOR * crate::hsde::REL_FLOOR_KAPPA * f64::EPSILON;
        let n = prob.n;
        let x = &self.x;
        let excess = |r: f64, mag: f64| (r - k * mag).max(0.0);

        // Stationarity, per column: `r_j` against the magnitude of the terms
        // it is a sum of, `|c_j| + Σ|P_jk x_k| + Σ|a_ij y_i| + Σ|g_ij z_i| +
        // z_lb_j + z_ub_j`.
        let mut r = vec![0.0; n];
        prob.p_mul(x, &mut r);
        for j in 0..n {
            r[j] += prob.c[j] - self.z_lb[j] + self.z_ub[j];
        }
        prob.at_mul(&self.y, &mut r);
        prob.gt_mul(&self.z, &mut r);
        let mut col_mag: Vec<f64> = (0..n)
            .map(|j| prob.c[j].abs() + self.z_lb[j].abs() + self.z_ub[j].abs())
            .collect();
        for t in &prob.p_lower {
            col_mag[t.row] += (t.val * x[t.col]).abs();
            if t.row != t.col {
                col_mag[t.col] += (t.val * x[t.row]).abs();
            }
        }
        for t in &prob.a {
            col_mag[t.col] += (t.val * self.y[t.row]).abs();
        }
        for t in &prob.g {
            col_mag[t.col] += (t.val * self.z[t.row]).abs();
        }
        let dual = (0..n).fold(0.0_f64, |m, j| m.max(excess(r[j].abs(), col_mag[j])));

        // Primal, per row: `|Ax - b|_i` against `|b_i| + Σ|a_ij x_j|`, the
        // violation `(Gx - h)_i⁺` against `|h_i| + Σ|g_ij x_j|`, a bound
        // violation against `|x_i| + |bound|`.
        let mut ax = vec![0.0; prob.m_eq()];
        prob.a_mul(x, &mut ax);
        let mut eq_mag: Vec<f64> = prob.b.iter().map(|v| v.abs()).collect();
        for t in &prob.a {
            eq_mag[t.row] += (t.val * x[t.col]).abs();
        }
        let mut primal = (0..prob.m_eq()).fold(0.0_f64, |m, i| {
            m.max(excess((ax[i] - prob.b[i]).abs(), eq_mag[i]))
        });
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(x, &mut gx);
        let mut row_mag: Vec<f64> = prob.h.iter().map(|v| v.abs()).collect();
        for t in &prob.g {
            row_mag[t.row] += (t.val * x[t.col]).abs();
        }
        for i in 0..prob.m_ineq() {
            primal = primal.max(excess((gx[i] - prob.h[i]).max(0.0), row_mag[i]));
        }
        let mut bound = 0.0_f64;
        for i in 0..n {
            let (lb, ub, xi) = (prob.lb_of(i), prob.ub_of(i), x[i]);
            if lb > -1e19 {
                bound = bound.max(excess((lb - xi).max(0.0), xi.abs() + lb.abs()));
            }
            if ub < 1e19 {
                bound = bound.max(excess((xi - ub).max(0.0), xi.abs() + ub.abs()));
            }
        }
        primal = primal.max(bound);

        // Complementarity, per product: `z_i·slack_i` against the noise of
        // that slack, `z_i·(|h_i| + Σ|g_ij x_j|)` (resp. `|x_i| + |bound|`).
        // No global allowance: the second pass also excused every product by
        // `k` times the duality-gap scale `|cᵀx| + |hᵀz| + ...`, which let one
        // large term excuse every product in the model.
        let mut comp = 0.0_f64;
        for (i, &zi) in self.z.iter().enumerate() {
            let slack = prob.h[i] - gx[i];
            comp = comp.max(excess((zi * slack).abs(), zi.abs() * row_mag[i]));
        }
        for i in 0..n {
            let (lb, ub, xi) = (prob.lb_of(i), prob.ub_of(i), x[i]);
            // A pinned column (`lb == ub`) is the equality `x_i = v`: its
            // multiplier is the free `z_lb - z_ub`, there is no
            // complementarity condition to meet, and feasibility is the bound
            // violation above. Reading `z_lb·(x - v)` there charged the
            // multiplier split (`z_lb, z_ub` both ~600 on a pinned production
            // column within `5e-10` of its pin: `3e-7`) as if it were a
            // complementarity error.
            if lb == ub {
                continue;
            }
            if lb > -1e19 {
                let zl = self.z_lb[i];
                comp = comp.max(excess(
                    (zl * (xi - lb)).abs(),
                    zl.abs() * (xi.abs() + lb.abs()),
                ));
            }
            if ub < 1e19 {
                let zu = self.z_ub[i];
                comp = comp.max(excess(
                    (zu * (ub - xi)).abs(),
                    zu.abs() * (xi.abs() + ub.abs()),
                ));
            }
        }
        // The objective-normalized problem's reading: both terms divided by the
        // objective's unit, up as well as down. The stopping rule is stricter
        // than this for a QP with a large unit (it keeps `tol` absolute there,
        // gh#846), so this can only ever *excuse* a label, never move a point;
        // when it is what excuses one, [`Self::unit_scaling_note`] says so.
        let (unit_d, unit_g) = report_units(prob);
        QpResiduals {
            primal_infeasibility: primal,
            bound_violation: bound,
            dual_infeasibility: dual / unit_d,
            complementarity: comp / unit_g,
        }
    }

    /// A note for a result whose stationarity / complementarity are within
    /// `tol` only *after* division by a large objective unit (gh#984 review).
    ///
    /// [`Self::kkt_residuals_above_floor`] reads those two residuals in the
    /// objective's own unit `max(‖P‖∞, ‖c‖∞)`, up as well as down, so a
    /// portfolio with `P` scaled by `1e9` reports `kkt_error 2e-9` beside a raw
    /// dual residual of `4041`. That is the right relative statement -- the
    /// weights are within `1.3e-7` -- but a caller comparing the raw residual
    /// with another solver's must not have it hidden. `None` unless the status
    /// is a success, the raw residual is above `tol`, and the unit-normalized
    /// one is not.
    pub fn unit_scaling_note(&self, prob: &QpProblem, tol: f64) -> Option<String> {
        if !matches!(self.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
            return None;
        }
        let (unit_d, _) = report_units(prob);
        if unit_d <= 1.0 {
            return None;
        }
        let raw = self.kkt_residuals(prob);
        let raw_dc = raw.dual_infeasibility.max(raw.complementarity);
        if raw_dc <= tol {
            return None;
        }
        let adj = self.kkt_residuals_above_floor(prob);
        if adj.dual_infeasibility.max(adj.complementarity) > tol {
            return None;
        }
        Some(format!(
            "scaling note: stationarity/complementarity are within tol only in \
             the objective's unit max(‖P‖∞, ‖c‖∞) = {unit_d:.1e}; the raw \
             residuals are dual {:.1e}, complementarity {:.1e} (kkt_error_raw). \
             Rescale the objective if you need them absolute.",
            raw.dual_infeasibility, raw.complementarity
        ))
    }

    fn kkt_residuals_inner(&self, prob: &QpProblem, cones: Option<&[ConeSpec]>) -> QpResiduals {
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
        // Inequality slack `s = h − Gx`, the vector that must lie in the cone
        // (in the orthant `s ≥ 0`, i.e. the familiar `Gx ≤ h`).
        let s: Vec<f64> = prob.h.iter().zip(&gx).map(|(&hi, &gxi)| hi - gxi).collect();
        let mut complementarity = 0.0_f64;
        let mut off = 0usize;
        for spec in cones.unwrap_or(&[]) {
            let dim = spec.dim().min(s.len() - off);
            if dim == 0 {
                break;
            }
            let (sb, zb) = (&s[off..off + dim], &self.z[off..off + dim]);
            primal_infeasibility = primal_infeasibility.max(cone_violation(spec, sb));
            complementarity = match spec {
                // Orthant rows complement one-for-one; keep the sharper
                // per-row measure rather than the block sum.
                ConeSpec::Nonneg(_) => sb
                    .iter()
                    .zip(zb)
                    .fold(complementarity, |m, (&si, &zi)| m.max((si * zi).abs())),
                _ => complementarity.max(
                    sb.iter()
                        .zip(zb)
                        .map(|(&si, &zi)| si * zi)
                        .sum::<f64>()
                        .abs(),
                ),
            };
            off += dim;
        }
        // Rows past the cone list (all of them when `cones` is `None`) are the
        // nonnegative orthant.
        for i in off..s.len() {
            primal_infeasibility = primal_infeasibility.max((-s[i]).max(0.0));
            complementarity = complementarity.max((self.z[i] * s[i]).abs());
        }
        // Accumulated separately as well as folded in: the summary block
        // reports the box term on its own line (Ipopt's `Variable bound
        // violation`), and once maxed into the aggregate above it can no
        // longer be told apart from a row violation.
        let mut bound_violation = 0.0_f64;
        for i in 0..n {
            bound_violation = bound_violation.max((prob.lb_of(i) - self.x[i]).max(0.0));
            bound_violation = bound_violation.max((self.x[i] - prob.ub_of(i)).max(0.0));
        }
        primal_infeasibility = primal_infeasibility.max(bound_violation);

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
            bound_violation,
            dual_infeasibility,
            complementarity,
        }
    }
}

#[cfg(test)]
mod residual_tests {
    use super::*;
    use crate::ipm::{QpOptions, solve_qp_ipm};
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

    /// Code review L38: on a Ruiz-equilibrated solve the per-iteration trace was
    /// recorded in scaled coordinates while the returned solution was unscaled,
    /// so the trace's objective disagreed with `sol.obj`. The unscaling pass now
    /// maps the per-iterate objective back (÷σ), so the converged trace point
    /// reports the same objective as the solution.
    ///
    /// A pure LP triggers the cost scaling σ = 1/max|ĉ| ≠ 1 (a QP keeps σ = 1,
    /// so the discrepancy is invisible there). With a large linear term the
    /// scaled objective is off by ~σ, which this test would catch.
    #[test]
    fn equilibrated_trace_objective_is_in_original_coordinates() {
        // min 1000·x0 + 500·x1  s.t.  x0 + x1 ≥ 2,  0 ≤ x ≤ 10.
        // Pure LP (empty P) ⇒ σ ≠ 1. Optimum loads the cheaper variable:
        // x = (0, 2), obj = 1000.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1000.0, 500.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0), Triplet::new(0, 1, -1.0)],
            h: vec![-2.0],
            lb: vec![0.0, 0.0],
            ub: vec![10.0, 10.0],
        };
        // Direct equilibrated path (use_hsde = false ⇒ Ruiz is applied), with
        // the trace turned on.
        let opts = QpOptions {
            use_hsde: false,
            equilibrate: true,
            collect_iterates: true,
            ..QpOptions::default()
        };
        let sol = solve_qp_ipm(&prob, &opts, backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.obj - 1000.0).abs() < 1e-3, "obj {} ≠ 1000", sol.obj);
        assert!(!sol.iterates.is_empty(), "trace should be populated");

        // The converged (final) trace point's objective must agree with the
        // unscaled solution objective — not the σ-scaled value the inner solve
        // recorded. (Before the fix this was ≈ σ·1000, off by orders of
        // magnitude.)
        let last = sol.iterates.last().unwrap();
        assert!(
            (last.objective - sol.obj).abs() < 1e-2,
            "final traced objective {} should match unscaled sol.obj {}",
            last.objective,
            sol.obj
        );
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

    /// Code review 2026-06 item M20: a mid-iteration breakdown whose best KKT
    /// residual is already within the reduced-accuracy band must be reported as
    /// the distinct `OptimalInaccurate`, *not* a bare `Optimal`. Before the fix
    /// both the symmetric and non-symmetric HSDE drivers re-labeled these
    /// breakdowns plain `Optimal`, so callers could not tell a residual sitting
    /// at ~1e3·tol apart from a genuinely converged solve. `breakdown_status`
    /// centralizes that decision; this pins it.
    #[test]
    fn breakdown_status_marks_near_opt_as_inaccurate_not_optimal() {
        // Near-optimal breakdown: usable iterate, reduced accuracy.
        assert_eq!(breakdown_status(true), QpStatus::OptimalInaccurate);
        assert_ne!(
            breakdown_status(true),
            QpStatus::Optimal,
            "a near-tol breakdown must be distinguishable from a clean Optimal"
        );
        // Genuine breakdown with a large residual: still a hard failure.
        assert_eq!(breakdown_status(false), QpStatus::NumericalFailure);
    }
}

#[cfg(test)]
mod conic_residual_tests {
    use super::*;
    use crate::ipm::{QpOptions, solve_socp_ipm};
    use pounce_feral::FeralSolverInterface;
    use pounce_linsol::SparseSymLinearSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    /// pounce#209: the residuals of a *conic* solve must be measured with the
    /// solve's own cones. `min x₀ s.t. ‖x‖ ≤ 1` in SOC form
    /// (`s = (1, x₀, x₁) ∈ K_soc`) has the optimum `x = (−1, 0)`, at which the
    /// cone is satisfied exactly — but its second SOC row reads `Gx = 1 > h = 0`
    /// and its rows are individually non-complementary. The orthant-only
    /// [`QpSolution::kkt_residuals`] therefore reports a large violation for a
    /// perfectly feasible point, which is what leaked into the CLI's
    /// end-of-run summary and made a solved QCQP look infeasible.
    #[test]
    fn conic_residuals_vanish_where_orthant_residuals_do_not() {
        // Rows of `s = h − Gx`: s₀ = 1 (the radius), s₁ = x₀, s₂ = x₁.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![],
            c: vec![1.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(1, 0, -1.0), Triplet::new(2, 1, -1.0)],
            h: vec![1.0, 0.0, 0.0],
            lb: vec![-5.0, -5.0],
            ub: vec![5.0, 5.0],
        };
        let cones = [ConeSpec::SecondOrder(3)];
        let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
        assert!(
            (sol.x[0] - -1.0).abs() < 1e-6 && sol.x[1].abs() < 1e-6,
            "expected the optimum (−1, 0), got {:?}",
            sol.x
        );

        let conic = sol.kkt_residuals_conic(&prob, &cones);
        assert!(
            conic.kkt_error() < 1e-6,
            "cone-aware residuals must vanish at the optimum: {conic:?}"
        );

        // The orthant reading of the same point is badly wrong — `s₁ = x₀ = −1`
        // looks like a violation of ~1 even though `s` is squarely in the cone.
        // (Pinned so the two measures cannot silently converge and make the
        // test above vacuous.)
        let orthant = sol.kkt_residuals(&prob);
        assert!(
            orthant.primal_infeasibility > 0.5,
            "the orthant metric should misread this feasible point (that is the \
             bug); got {orthant:?}"
        );
    }

    /// The trailing rows a cone list does not cover fall back to the orthant, so
    /// a plain QP's residuals are identical whether or not a (nonneg) cone list
    /// is supplied. Guards the shared implementation against drift.
    #[test]
    fn conic_residuals_match_orthant_on_a_cone_free_qp() {
        // min ½(x₀² + x₁²) − x₀ s.t. x₀ + x₁ ≤ 1, 0 ≤ x ≤ 2.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-1.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![0.0, 0.0],
            ub: vec![2.0, 2.0],
        };
        let cones = [ConeSpec::Nonneg(1)];
        let sol = solve_socp_ipm(&prob, &cones, &QpOptions::default(), backend);
        let orthant = sol.kkt_residuals(&prob);
        let conic = sol.kkt_residuals_conic(&prob, &cones);
        assert_eq!(orthant, conic, "orthant rows must measure identically");
        // And with no cone list at all: every row is orthant by default.
        assert_eq!(orthant, sol.kkt_residuals_conic(&prob, &[]));
    }
}
