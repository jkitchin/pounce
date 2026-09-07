//! Post-optimal sensitivity for the convex QP — the sIPOPT analog.
//!
//! Given a converged [`QpSolution`] to
//!
//! ```text
//!   min ½xᵀPx + cᵀx  s.t.  Ax = b,  Gx ≤ h,  lb ≤ x ≤ ub,
//! ```
//!
//! the first-order change of the primal–dual solution under a small
//! perturbation of the problem data — *holding the active set fixed* — is
//! the solution of the **active-set KKT system**
//!
//! ```text
//!   ⎡ P    Aᵀ   B_aᵀ ⎤ ⎡ dx  ⎤   ⎡ −dc                  ⎤
//!   ⎢ A    0    0    ⎥ ⎢ dy  ⎥ = ⎢  db                  ⎥
//!   ⎣ B_a  0    0    ⎦ ⎣ dz_a⎦   ⎣  dr_a                ⎦
//! ```
//!
//! where `B_a` stacks the **active** inequality rows of `G` and the active
//! variable-bound rows (`eⱼᵀ`), and the right-hand side is the parameter
//! derivative of the KKT residual. This is exactly the predictor used by
//! Ipopt's sIPOPT (Pirnay, López-Negrete & Biegler 2012) specialized to a
//! quadratic program, where the Lagrangian Hessian is the constant `P`.
//!
//! [`QpSensitivity`] assembles and factors this symmetric, indefinite
//! system **once** at the optimum; each [`QpSensitivity::parametric_step`]
//! is then a single back-substitution, so a parametric sweep costs one
//! solve per query (the build-once / solve-many idiom of the NLP
//! `Solver`). A tiny static regularization `δ` (the QP solver's own `reg`,
//! default `1e-10`) is placed on the diagonal so the indefinite factor is
//! stable.
//!
//! # Near-singular (near-LICQ) KKT: refinement + a conditioning diagnostic
//!
//! When the active-constraint gradients are *nearly* rank-deficient (LICQ
//! almost fails — e.g. two nearly-parallel equality rows) the KKT matrix is
//! near-singular. A single regularized back-solve then **over-damps**
//! `dx/db` toward a smooth but badly wrong value, silently, because the
//! static `δ` floors the smallest KKT singular value (gh #284). Two
//! defenses close that gap:
//!
//! 1. **Iterative refinement against the *unregularized* KKT.** Each solve
//!    refines its back-substitution against the true (`δ`-free) KKT matrix,
//!    so the `O(δ)` regularization bias is removed wherever the information
//!    is still present in double precision — recovering LU-quality `dx/db`.
//!    On a well-conditioned KKT the first residual is already at round-off
//!    and refinement is a no-op, so this never perturbs the good cases.
//! 2. **A two-part conditioning diagnostic.**
//!    [`QpSensitivity::kkt_cond_estimate`] is a cheap Hager 1-norm estimate of
//!    `κ₁` of the factored KKT; [`QpSensitivity::ill_conditioned`] fires when it
//!    is huge **or** when the most recent step's refinement residual is large.
//!    The condition estimate alone has a blind spot: it measures the
//!    *regularized* factor, whose smallest singular value is floored at `δ`, so
//!    on a well-scaled `P` (e.g. `P = I`, `‖K‖₁ ≈ O(1)`) it saturates near
//!    `‖K‖₁ / δ` and never reaches its threshold, even when the true KKT is
//!    numerically singular — so a purely near-parallel *constraint* Jacobian
//!    slips past it (gh #328). The per-step residual closes that gap: refinement
//!    against the true KKT *cannot* solve an unrecoverable step, so it stalls at
//!    a large relative residual ([`QpSensitivity::last_step_residual`]), which
//!    fires the flag. Between the two, a caller can always *detect* that `dx/db`
//!    is untrustworthy instead of consuming a silently-damped value — whether
//!    the near-singularity shows up in the condition estimate (badly-scaled `P`)
//!    or only in the stalled residual (well-scaled `P`, near-LICQ constraints).

use crate::activity::{ConvexActivityReport, classify_all, curvature_floor};
use crate::cones::ConeSpec;
use crate::cones::psd::smat;
use crate::ipm::QpOptions;
use crate::qp::{BOUND_INF, QpProblem, QpSolution, QpStatus, Triplet};
use pounce_common::types::{Index, Number};
use pounce_linalg::symmetric_eigen;
use pounce_linsol::{Factorization, SparseSymLinearSolverInterface};
use pounce_sens_core::backsolver::{BoundRow, SensBacksolver};
use pounce_sens_core::boundcheck::{
    BoundMultiplier, PathSegment, RefineStop, refine_step_onto_bounds, step_along_path,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

/// Group a constraint matrix's triplets by row, so an active-set assembly
/// can read a row's `(col, val)` entries directly. Without this, both the
/// KKT build and the reduced-Hessian assembly re-scanned *all* of `G` once
/// per active row (`O(n_active · nnz(G))`); the grouping is a single
/// `O(nnz(G))` pass and each lookup is then proportional to that row's
/// own nonzeros.
///
/// `n_rows` must bound every `t.row` — that is what makes the raw `rows[t.row]`
/// safe. **Two callers rely on it and they are not the same shape**: the
/// inequality path passes `prob.g` with `m_ineq`, and [`apex_can_absorb_db`]'s
/// caller passes `prob.a` with `m_eq = prob.b.len()`. A `QpProblem` whose `a`
/// carried a triplet with `row >= b.len()` would be malformed — an equality
/// with no right-hand side — so it holds on both, but it is the *problem's*
/// invariant rather than this function's, and the second caller arrived in #889
/// long after this sentence said "the number of inequality rows".
pub(crate) fn group_rows_by_index(triplets: &[Triplet], n_rows: usize) -> Vec<Vec<(usize, f64)>> {
    let mut rows = vec![Vec::new(); n_rows];
    for t in triplets {
        rows[t.row].push((t.col, t.val));
    }
    rows
}

/// A reason a [`QpSensitivity`] could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SensError {
    /// The solution was not optimal, so the active set is undefined.
    NotOptimal,
    /// The active-set KKT factorization failed (e.g. the active constraint
    /// gradients are rank-deficient, so the parametric step is not unique).
    FactorizationFailed,
    /// A symmetric eigensolve did not converge while forming the reduced
    /// Hessian, so its rank / null-space (and hence the result) cannot be
    /// trusted. Only [`reduced_hessian`](QpSensitivity::reduced_hessian) can
    /// raise this; the parametric step does not eigendecompose.
    EigenFailed,
    /// The solution's inequality block does not complement **row by row**, so
    /// it is not a solution of an orthant-only problem and
    /// [`build`](QpSensitivity::build) must not read it as one.
    ///
    /// [`solve_socp_ipm`](crate::solve_socp_ipm) returns the same
    /// [`QpSolution`] type as [`solve_qp_ipm`](crate::solve_qp_ipm), and the
    /// cone partition travels beside it as a separate `&[ConeSpec]` argument
    /// that `build` never sees. Without this check a solved SOCP is accepted
    /// and every cone row is silently read as an orthant row — a wrong
    /// `dx/db` reported as a good one. A conic solution complements only as a
    /// *block* inner product `⟨s, z⟩ = 0`; row-wise `sᵢ·zᵢ` is generally
    /// nonzero and `z` generally has negative entries, which is what this
    /// detects. Use [`build_conic`](QpSensitivity::build_conic) for a problem
    /// that carries cones.
    NotOrthantComplementary {
        /// The inequality row whose evidence triggered the refusal.
        row: usize,
        /// Which test failed, for a diagnosable message.
        what: &'static str,
    },
    /// The bound refinement could not run — a shape mismatch or a back-solve
    /// failure inside `pounce-sens-core`, carrying that layer's own message.
    Refinement(String),
    /// The solution sits at a point where the cone's face is not differentiable
    /// — an apex reached along the boundary, a collapsed normal — so there is no
    /// single `dx/db` to report.
    NonsmoothConePoint {
        /// Index into the `&[ConeSpec]` slice.
        block: usize,
        /// Which condition fired.
        what: &'static str,
    },
    /// The active set has no room left for `dx`, so no step can satisfy
    /// `A·dx = db` — but a derivative may well **exist**.
    ///
    /// This is the sibling of [`NonsmoothConePoint`](Self::NonsmoothConePoint)
    /// and the distinction is the whole point of having two variants. That one
    /// means *there is no single `dx/db` here*: a kink, a collapsed normal, a
    /// two-valued derivative. This one means *the derivative exists and this
    /// active set cannot express it*.
    ///
    /// The case it was introduced for is exactly that: an apex-pinned block on
    /// a problem that is **smooth on both sides of the classification cliff**
    /// — at `‖s‖` a decade larger the boundary face finds the true derivative,
    /// and only the classifier changed its mind. A caller who matched
    /// `NonsmoothConePoint` to decide "genuinely nondifferentiable, fall back
    /// to a subgradient" would make the wrong call on a smooth model, which is
    /// why these are not the same error. Raised in review of #889.
    ///
    /// A re-solve at a looser apex tolerance, or a perturbation small enough to
    /// keep the block off its tip, will generally be answerable.
    ActiveSetOverdetermined {
        /// Index into the `&[ConeSpec]` slice.
        block: usize,
        /// Which condition fired.
        what: &'static str,
    },
    /// The cone partition handed to
    /// [`build_conic`](QpSensitivity::build_conic) does not cover the
    /// inequality block exactly, so the caller and the builder disagree about
    /// which rows are which.
    ConePartitionMismatch {
        /// Rows the partition accounts for (`Σ ConeSpec::dim`).
        covered: usize,
        /// Rows the problem actually has.
        m_ineq: usize,
    },
}

/// Refuse a solution whose inequality block does not complement row by row.
///
/// `gx` is `G·x` at the solution, so `sᵢ = hᵢ − gxᵢ` is row `i`'s slack. For a
/// nonnegative-orthant row at an optimum, `sᵢ ≥ 0`, `zᵢ ≥ 0` and `sᵢzᵢ = μ ≈ 0`
/// — all three hold *per row*. A conic block satisfies only `⟨s, z⟩ = 0` over
/// the block, so a second-order-cone row on its boundary has `s = (t, u)` with
/// `t = ‖u‖ > 0` and `z ∝ (t, −u)`: the tail entries of `z` are negative and
/// `s₀z₀ = c·t² > 0`. Either signal is decisive.
///
/// # What this cannot catch
///
/// A second-order cone at its **apex** with `z_{1:} = 0` — `s = 0` and
/// `z = (z₀, 0, …, 0)` — passes every test here, because row-wise it is
/// indistinguishable from a degenerate orthant block, which is a legitimate
/// input. That case needs the cone partition, which is what
/// [`QpSensitivity::build_conic`] takes. This guard is the safety net for a
/// caller who never mentions cones; it is not a substitute for telling the
/// builder what the problem is.
fn check_orthant_complementarity(
    prob: &QpProblem,
    sol: &QpSolution,
    gx: &[f64],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<(), SensError> {
    let sign_tol = ORTHANT_GUARD_REL * dual_scale;
    let slack_tol = ORTHANT_GUARD_REL * primal_scale;
    let comp_tol = ORTHANT_GUARD_REL * primal_scale * dual_scale;
    for (i, (&h_i, &gx_i)) in prob.h.iter().zip(gx.iter()).enumerate() {
        let s = h_i - gx_i;
        let z = sol.z[i];
        // A dual entry that is negative by more than round-off is not an
        // orthant multiplier at all — the SOC dual's tail is the common case.
        if z < -sign_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "the inequality multiplier is negative, which the nonnegative orthant \
                       forbids; a second-order or exponential cone's dual has negative entries",
            });
        }
        // Likewise a slack outside the orthant.
        if s < -slack_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "the inequality slack is negative beyond the solve's own tolerance, so \
                       this row is not a satisfied orthant row",
            });
        }
        if (s * z).abs() > comp_tol {
            return Err(SensError::NotOrthantComplementary {
                row: i,
                what: "slack and multiplier are both away from zero, so the row does not \
                       complement; a conic block complements only as a block inner product",
            });
        }
    }
    Ok(())
}

/// What a second-order cone block is doing at the solution.
///
/// A cone's "active set" is not a set of rows. Its slack `s` sits somewhere on
/// the cone, and what the sensitivity needs is the *face* it sits on — the
/// tangent/normal decomposition there. Every family in [`ConeSpec`] splits the
/// same three ways, which is why one enum serves all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConeBlockKind {
    /// `s` strictly inside the cone, so `z = 0`: the block constrains nothing
    /// locally and contributes no rows.
    Interior,
    /// `s` at the apex (`s ≈ 0`) with `z` in the dual interior. The whole block
    /// is active — every row of `G` for this block enters `B_a`, because `ds`
    /// must keep `s = 0`. The face is a single point, hence **flat**, so the
    /// predictor is exact here in the same way an orthant row's is.
    ///
    /// For a PSD block this is `S = 0`, the rank-zero end of the same
    /// constant-rank stratification [`ConeBlockKind::Boundary`] covers.
    Apex,
    /// `s` on the relative boundary away from the apex, in the interior of a
    /// face the cone is *smooth* along. What that face is, and how many rows it
    /// contributes, is per family:
    ///
    /// | family | face | rows |
    /// |---|---|---|
    /// | `SecondOrder(k)` | `s₀ = ‖s₁‖ > 0` | 1, `wᵀG` with `w = (1, −s₁/s₀)` |
    /// | `Psd(n)` at rank `r` | the constant-rank manifold | `q(q+1)/2`, `q = n − r` |
    /// | `Exponential`, `Power(α)` | the smooth facet `φ(s) = 0` | 1, `∇φᵀG` |
    ///
    /// Unlike the orthant and apex cases every one of these faces is
    /// **curved**, so the rows are a linearization and the step is first-order
    /// rather than exact — the same status an active nonlinear constraint has
    /// on the NLP arm. The curvature is not optional; see [`assemble_kkt`].
    Boundary,
}

/// Numbers a cone block is classified against, all relative to the
/// problem-wide `primal_scale` / `dual_scale` the orthant guard already uses,
/// so a verdict does not move when the model is rescaled.
///
/// Those two scales are floored at `1.0` (see [`QpSensitivity::build`]), which
/// makes them absolute for a model whose data is smaller than one. That is a
/// deliberate convention inherited from the orthant guard rather than an
/// oversight — changing it here alone would leave the two guards disagreeing
/// about what "zero" means on the same solution.
const CONE_APEX_REL: f64 = 1e-8;
/// How close the face's defining function must sit to zero for the block to
/// count as *on* that face rather than strictly inside the cone —
/// `s₀ − ‖s₁‖` for a second-order block, `φ(s)` for a non-symmetric one. The
/// PSD arm uses it as the eigenvalue threshold that decides the rank.
const CONE_BOUNDARY_REL: f64 = 1e-8;
/// Strict-complementarity screen. A boundary block whose dual has collapsed to
/// this level is the conic analogue of a weakly active row: slack and
/// multiplier vanish together, `dx/db` is two-valued, and there is no single
/// answer to return.
const CONE_STRICT_COMP_REL: f64 = 1e-8;

/// Classify one cone block and return the rows it contributes, plus the
/// lower-triangle `(x,x)` curvature triplets its face carries.
///
/// This is the whole of the conic arm's decision. `s` and `z` are the block's
/// slices of the slack and the dual; `g_rows` holds the block's rows of `G` in
/// block order.
///
/// The `match` below is **exhaustive over [`ConeSpec`]**, and that is the
/// guard: a family added to `ConeSpec` without a face decomposition breaks the
/// build rather than reaching a runtime refusal. That is why there is no
/// longer an "unsupported cone" error — an empty error category is a
/// documentation hazard, and a compile error is a stronger promise than a
/// message.
///
/// [`ConeSpec::Nonneg`] never arrives here: an orthant block is per-row, not
/// per-block, and [`QpSensitivity::build_conic`] applies the plain path's rule
/// to its rows directly.
fn cone_block_face(
    block: usize,
    spec: &ConeSpec,
    s: &[f64],
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    // The apex is family-independent: `s = 0` is the cone's tip whatever the
    // cone, `ds` must keep it there, so every row of the block enters and the
    // face — a single point — is flat.
    if inf_norm_of(s) <= CONE_APEX_REL * primal_scale {
        if dual_has_collapsed(z, dual_scale) {
            return Err(ConeError::new(
                block,
                "the slack is at the cone apex and the dual has collapsed too, so the \
                 block is weakly active: dx/db is two-valued there",
            ));
        }
        return Ok((ConeBlockKind::Apex, g_rows.to_vec(), Vec::new()));
    }
    match spec {
        ConeSpec::Nonneg(_) => unreachable!("orthant blocks are classified per row"),
        ConeSpec::SecondOrder(_) => soc_face(block, s, z, g_rows, primal_scale, dual_scale),
        ConeSpec::Psd(n) => psd_face(block, *n, s, z, g_rows, primal_scale, dual_scale),
        ConeSpec::Exponential => exp_face(block, s, z, g_rows, primal_scale, dual_scale),
        ConeSpec::Power(alpha) => power_face(block, *alpha, s, z, g_rows, primal_scale, dual_scale),
    }
}

/// A refusal under construction — the block index is filled in by the caller
/// that knows it, so a face routine only supplies the reason.
struct ConeError {
    block: usize,
    what: &'static str,
}

impl ConeError {
    fn new(block: usize, what: &'static str) -> Self {
        ConeError { block, what }
    }
}

impl From<ConeError> for SensError {
    fn from(e: ConeError) -> SensError {
        SensError::NonsmoothConePoint {
            block: e.block,
            what: e.what,
        }
    }
}

fn inf_norm_of(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn dual_has_collapsed(z: &[f64], dual_scale: f64) -> bool {
    inf_norm_of(z) <= CONE_STRICT_COMP_REL * dual_scale
}

/// The block contributes nothing — but only if it really is complementary. A
/// slack strictly inside the cone with a live dual is not the optimum it is
/// being read as, whatever the status field says.
fn interior_face(
    block: usize,
    s: &[f64],
    z: &[f64],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    let comp: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
    if comp.abs() > ORTHANT_GUARD_REL * primal_scale * dual_scale {
        return Err(ConeError::new(
            block,
            "the slack is strictly inside the cone yet the block does not \
             complement, so this is not the optimum it is being read as",
        ));
    }
    Ok((ConeBlockKind::Interior, Vec::new(), Vec::new()))
}

/// `wᵀG` over one block's rows, as a sparse row in `x` coordinates. This is
/// also `Gᵀw`, which is the form the curvature wants — the same vector either
/// way.
fn combine_rows(w: &[f64], g_rows: &[Vec<(usize, f64)>]) -> Vec<(usize, f64)> {
    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
    for (r, row) in g_rows.iter().enumerate() {
        if w[r] == 0.0 {
            continue;
        }
        for &(col, val) in row {
            *acc.entry(col).or_insert(0.0) += w[r] * val;
        }
    }
    acc.into_iter().filter(|&(_, v)| v != 0.0).collect()
}

/// `Σ_k κ_k · v_k v_kᵀ` as lower-triangle triplets, for sparse `v_k`.
fn outer_triplets(terms: &[(f64, Vec<(usize, f64)>)]) -> Vec<(usize, usize, f64)> {
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for (kappa, v) in terms {
        if *kappa == 0.0 {
            continue;
        }
        for &(r, vr) in v {
            for &(c, vc) in v {
                if r < c {
                    continue; // lower triangle only
                }
                *acc.entry((r, c)).or_insert(0.0) += kappa * vr * vc;
            }
        }
    }
    acc.into_iter()
        .filter(|&(_, v)| v != 0.0)
        .map(|((r, c), v)| (r, c, v))
        .collect()
}

/// Rank of a set of sparse rows over `n` columns, by Gaussian elimination with
/// partial pivoting on the dense restriction to their combined support.
///
/// Called only by [`apex_can_absorb_db`], on a build that carries an apex block
/// — twice per such build: once on the active rows that cannot be released, and
/// once on `A` to get `rank(A)` rather than its row count.
///
/// # Cost, stated honestly
///
/// `O(r² · w)` for `r` rows over a combined support of width `w`, as a dense
/// `r × w` array.
///
/// Three calls per apex-carrying build, and they are not the same size. On `B`
/// a cone face contributes one row (second-order, exponential, power) or
/// `q(q+1)/2` (PSD), each touching a handful of columns — though a **mixed
/// partition's orthant rows are also in there**, and there can be many. On `A`,
/// and on the stacked `[A; B]`, the support is typically **every column**, so
/// an apex build on a model with many equalities allocates an `m_eq × n` dense
/// array. That is the honest cost of the exact criterion, and it is not what
/// the cone-face reasoning alone would suggest.
///
/// The apex gate is what keeps all of it off the common path, not the row
/// count: an apex is rare, and a build without one never calls this.
///
/// That is a statement about **frequency, not magnitude** — the gate does not
/// bound `m_eq`. Once you are on the path, the two `A`-width eliminations are
/// `~20 GB` each at 50 000 equalities over 50 000 columns (round 5 of #889,
/// recording the number rather than proposing a change). Unreachable today,
/// since the conic CLI route reroutes before it. If it ever is reachable the
/// fix is a rank-revealing factorization or a gate on `m_eq`, not the
/// short-circuit round 4 proposed — that one is only valid for the weaker
/// criterion this replaced.
///
/// # The tolerance biases toward *refusing a servable model*
///
/// The guard refuses unless `rank([A;B]) == rank(A) + rank(B)`. Undercounting
/// the stacked rank breaks that equality and **refuses**; undercounting either
/// part restores it and **passes**. The stacked elimination has the most rows,
/// and so the most opportunity to drop a marginal pivot — which biases the
/// guard toward **inventing** a deficiency, not missing one.
///
/// Measured, on `A = [x₀]`, `B = [x₀ + ε·x₁]`, `n = 2`, where
/// `ker(B) = span(−ε, 1)` and `A(ker B) = range(A)` for every `ε > 0`, so the
/// truth is *serve* throughout:
///
/// ```text
///   ε = 1e-6, 1e-7    rank([A;B]) = 2      serve
///   ε ≤ 1e-8          rank([A;B]) = 1      REFUSE   (truth: serve)
/// ```
///
/// That is acceptable, and the reason is specific rather than general: the
/// near-dependency an undercount detects is between `A` and `B`, so at `ε ≤
/// 1e-8` the true `dx/db` is `O(1/ε) ≥ 1e8` and a refusal is the honest
/// answer to a question whose answer is numerically meaningless. It is **not**
/// acceptable for the reason the previous three versions of this comment gave
/// — "a missed deficiency is left to `ill_conditioned()`" — because missing is
/// no longer the direction this errs in. A reader who budgeted for that would
/// budget for an error the code does not make.
///
/// The complementary misfire exists and goes the other way: undercount a
/// *part* and a genuinely deficient model passes (`A = [x₁]`,
/// `B = [x₀, x₀ + 1e-12·x₁]`, where `rank(B)` reads 1 against a true 2). There
/// `ill_conditioned()` is the fallback, measured across three shapes at
/// residual `0.5`, `0.333` and `0.8` against a `1e-6` threshold, with no
/// overlap against the `~1e-13` of a correct step.
///
/// (This direction has now been recorded wrongly three times: twice with both
/// halves backwards, and once — here — because changing the criterion from
/// `≥` to `==` inverted the bias and the prose did not move. The threshold has
/// been `√ε` throughout. `row_rank_drops_a_round_off_pivot` now asserts the
/// guard's *verdict* on both misfires, not just this function's rank, so a
/// fourth flip has a test that can see it.)
fn row_rank(rows: &[Vec<(usize, f64)>], n: usize) -> usize {
    if rows.is_empty() || n == 0 {
        return 0;
    }
    // Dense over the combined support: a cone block touches a handful of
    // columns even when the model has thousands.
    let mut cols: Vec<usize> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&(c, _)| c))
        .collect();
    cols.sort_unstable();
    cols.dedup();
    let w = cols.len();
    let index_of = |c: usize| cols.binary_search(&c).expect("column is in the support");
    let mut m: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            let mut row = vec![0.0; w];
            for &(c, v) in r {
                row[index_of(c)] += v;
            }
            row
        })
        .collect();

    // Equilibrate: divide each row by its own largest entry, and drop rows that
    // are entirely zero. Only then is a single scalar tolerance meaningful.
    //
    // A *global* max was enough while these rows were cone faces and active
    // orthant rows, whose entries the cone geometry and `G` fix within an order
    // or two of each other. It stopped being enough when `A` joined them
    // (#889, fourth review): `A`'s row scaling is the user's, so one equality
    // carrying `1e10` coefficients would set `tol = √ε · 1e10 ≈ 0.15` for every
    // pivot in the elimination, and two genuinely independent `O(1)` equalities
    // elsewhere in `A` would collapse to rank 1 — the guard passing where it
    // must refuse, on a model that is merely badly scaled rather than
    // degenerate. Row-wise, `tol` means the same thing on `A` that it always
    // meant on the face rows.
    let mut any = false;
    for row in &mut m {
        let s = row.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if s > 0.0 {
            for v in row.iter_mut() {
                *v /= s;
            }
            any = true;
        }
    }
    if !any {
        return 0;
    }
    // `√ε` on rows whose largest entry is now exactly 1: below that a pivot is
    // round-off, and counting it inflates the rank. Rank up ⇒ refuse, so
    // dropping small pivots deflates the rank and biases the guard toward
    // *missing* a deficiency rather than inventing one — the safe direction for
    // a refusal, since a miss is left to `ill_conditioned()` while an invention
    // breaks a working model at build time with no recourse.
    let tol = f64::EPSILON.sqrt();

    let mut rank = 0;
    for col in 0..w {
        let Some(piv) = (rank..m.len()).max_by(|&a, &b| {
            m[a][col]
                .abs()
                .partial_cmp(&m[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            break;
        };
        if m[piv][col].abs() <= tol {
            continue;
        }
        m.swap(rank, piv);
        let inv = 1.0 / m[rank][col];
        for r in rank + 1..m.len() {
            let f = m[r][col] * inv;
            if f == 0.0 {
                continue;
            }
            for c in col..w {
                m[r][c] -= f * m[rank][c];
            }
        }
        rank += 1;
        if rank == m.len() {
            break;
        }
    }
    rank
}

/// Can an apex-pinned active set still absorb an arbitrary `db`?
///
/// An apex block is the one face that pins its **whole** block unconditionally:
/// `s = 0` is the cone's tip, so `ds_block = 0` and every row of the block
/// enters the active set. Every other face pins one row, or none. That
/// difference is what makes this check apex-gated.
///
/// **That is a statement about faces, not about rows, and it is not a
/// completeness claim** (round 5 of #889). Active orthant rows are in `B` too
/// and have no release path either — [`release_slots`] is built for variable
/// bounds only — so a purely polyhedral degenerate QP can have
/// `A(ker B) ⊊ range(A)`, never reach this gate, and be served the same
/// least-squares answer the gate exists to refuse. That is pre-existing
/// plain-path behaviour rather than something this guard introduced, and
/// widening the gate to every build is a separate change with its own cost;
/// but the reader should not take "apex-gated" to mean "the only place this
/// can happen".
///
/// The step then lives in `ker(B)` for the active rows `B`, while feasibility
/// of the perturbed problem needs `A·dx = db`. So the apex model can answer
/// every `db` it may be asked for only if `A` restricted to that kernel still
/// covers `range(A)` — which holds exactly when
///
/// ```text
///   rank([A; B])  ==  rank(A) + rank(B)
/// ```
///
/// # Why this, and not a dimension count
///
/// The quantity that matters is `dim A(ker B)`: the space of perturbations the
/// step can actually reach. The rank identity gives it directly —
/// `dim A(ker B) = rank([A; B]) − rank(B)` — and we need that to be all of
/// `range(A)`, i.e. `rank(A)`. Rearranged, that is the line above.
///
/// `rank(A)` and not `A`'s **row count**: a redundant equality does not shrink
/// the space a step has to reach. The reachable perturbations are `range(A)`,
/// and a `db` outside it makes the *perturbed problem* infeasible rather than
/// the derivative unrepresentable — a different failure, and not this guard's
/// to report. `an_apex_with_a_redundant_equality_is_served` is the model where
/// those two readings disagree.
///
/// This condition is **exact**, where the earlier dimension count
/// `n − rank(B) ≥ rank(A)` was only necessary. It implies that count strictly
/// (`rank([A;B]) ≤ n`), so it refuses a superset — and it refuses exactly the
/// models the count let through with a wrong answer.
///
/// The fourth review of #889 is why. Move [`apex`]'s equality onto coordinates
/// the apex pins, `x₀ + x₁ = b`: `rank(A) = 1`, `rank(B) = 3`, `4 − 3 ≥ 1`, so
/// the count passes — but `ker(B) = span(e₂)` and `A e₂ = 0`, so `A(ker B)` is
/// `{0}` and **no** nonzero `db` is reachable at all. Measured, it was served,
/// and returned `dx/db = (⅓, ⅓, 0, 0)` against `|A·dx − db|/δ = 0.333` at every
/// step size — the guard's own motivating defect, one model past the guard.
/// `an_equality_inside_the_pinned_coordinates_is_refused` pins it.
///
/// That model is not a "subtle dependency" of the kind a coarse rule may fairly
/// leave to a residual flag: the equality lives *entirely inside* the pinned
/// coordinates. The earlier doc claimed the count "already covers" this case
/// and priced the exact test as costing "far more". Neither survived
/// measurement: it did not cover it, and the exact test is one more
/// elimination, on rows already assembled.
///
/// # `B` is the rows that cannot be released, and that is the whole subtlety
///
/// `B` is `active_rows`: the cone faces and the active orthant rows. It
/// deliberately **excludes active variable bounds**, even though a bound does
/// pin its coordinate for the plain [`parametric_step`](QpSensitivity::parametric_step).
///
/// The reason is [`release_slots`], which exists precisely so that fix-relax
/// can *open* an active bound. Counting bounds here would refuse the build for
/// a model [`parametric_step_bounded`](QpSensitivity::parametric_step_bounded)
/// could serve — the refusal is at build time, so it takes the release path
/// away too. A cone face row has no such escape: `release_slots` is built for
/// variable bounds only. Ranking exactly the un-releasable rows is what keeps
/// the guard from over-refusing.
///
/// The cost of that choice is a bound-pinned model whose *plain* step is then a
/// least-squares compromise, left to `ill_conditioned()` — the same division of
/// labour the dimension count already uses: refuse what no mode can serve, flag
/// what some mode can.
///
/// That transfer was **measured** rather than assumed, at the third review of
/// #889, because the 33-probe separation behind it was taken on the cone-apex
/// case and this is a different row type on a different path. On the
/// discriminating model (`a_bound_pinned_apex_is_served_by_fix_relax_and_flagged_without_it`
/// in `tests/convex_soc_sensitivity.rs`) the plain step misses `A·dx = db` by a
/// third, `parametric_step_bounded` reproduces the re-solve exactly, and
/// `ill_conditioned()` is `true` — residual `0.333` against a `1e-6` threshold,
/// not a marginal call.
///
/// **But it is the *second* clause that fires, and only after a step.** At
/// build time `kkt_cond_estimate()` reads `3.0e10` there, comfortably under
/// `KKT_ILL_CONDITIONED_THRESHOLD`, so a caller who checks `ill_conditioned()`
/// straight after `build_conic` — which that accessor's own doc invites — gets
/// `false`, and then takes the wrong step. The assembled KKT genuinely *is*
/// well conditioned; what is wrong is that it carries a row the perturbation
/// forces off, and only the residual can see that. So "`ill_conditioned()`
/// covers the remainder" means *after the step*, never at build time.
///
/// # What the condition does and does not promise
///
/// **Exact for the question it asks, and still a build-time decision.** When it
/// refuses, `A(ker B)` really is a proper subspace of `range(A)`: some `db`
/// direction the caller may later ask for is unreachable, and the step for it
/// would be a least-squares compromise rather than a derivative.
///
/// It is not a promise that *every* `db` would have failed. A build serves
/// every later perturbation and cannot know which are coming, so it refuses on
/// the existence of one bad direction — deliberate, and a stronger action than
/// "no answer exists here", which is worth stating rather than implying.
///
/// The complementary case — a `db` the caller asks for that is outside
/// `range(A)` entirely, on a build this guard *served* — is not a refusal at
/// all: the perturbed problem is infeasible, so there is no derivative to
/// report. The step comes back a least-squares answer to an unanswerable
/// question, and [`ill_conditioned`](QpSensitivity::ill_conditioned) is what
/// tells the caller so (measured on the redundant-equality fixture: residual
/// `0.8` against a `1e-6` threshold). After a step, never at build time.
///
/// Found by adversarial review of #889: `min t s.t. u = b₀, v = b₁,
/// (t,u,v) ∈ Q₃` — the parametric distance function — classifies `Apex` once
/// `‖b‖` drops under `CONE_APEX_REL`, and returned `du/db₀ = 0.5` where primal
/// feasibility *alone* forces `1`. The problem is smooth on both sides of that
/// cliff: at `‖b‖ = 1.12e-8` the true derivative still exists and the boundary
/// branch finds it. Only the classifier changed its mind — which is why the
/// error this raises is [`SensError::ActiveSetOverdetermined`] and **not**
/// `NonsmoothConePoint`.
fn apex_can_absorb_db(
    n: usize,
    eq_rows: &[Vec<(usize, f64)>],
    active_rows: &[Vec<(usize, f64)>],
) -> bool {
    let rank_a = row_rank(eq_rows, n);
    if rank_a == 0 {
        return true;
    }
    let rank_b = row_rank(active_rows, n);
    let stacked: Vec<Vec<(usize, f64)>> =
        eq_rows.iter().chain(active_rows.iter()).cloned().collect();
    row_rank(&stacked, n) == rank_a + rank_b
}

/// A point on a face defined by one smooth concave inequality `φ(s) ≥ 0`,
/// active at `φ(s) = 0`.
///
/// This is the shape the exponential and power cones present, and it is the
/// general form the second-order case is a hand-optimized instance of: with
/// `z = ν∇φ`, the active row is `∇φᵀG` and the Lagrangian carries `−νφ(h−Gx)`,
/// whose `x`-Hessian is `−ν·Gᵀ∇²φ G`. Because `φ` is concave, `∇²φ ⪯ 0` and
/// that term is positive semidefinite, exactly as the second-order case's is.
///
/// `hess_factors` supplies `∇²φ = −Σ κ_k v_k v_kᵀ` in factored form, which
/// costs `k` rank-one updates rather than a dense `d × d` product. Both cones
/// here have `k = 1`.
struct SmoothFacet {
    grad: Vec<f64>,
    hess_factors: Vec<(f64, Vec<f64>)>,
}

/// How close `φ(s)` must sit to zero, relative to the block's primal scale,
/// for the facet to count as active.
///
/// Separate from [`CONE_BOUNDARY_REL`] because it is calibrated against a
/// different solver. The exponential and power cones route to the
/// **non-symmetric** HSDE driver, whose achieved primal accuracy is well short
/// of the symmetric one's. Measured across four fixtures (exponential ×2,
/// `Power(0.6)`, `Power(0.3)`) at `tol` `1e-9` and `1e-11`, `|φ|/primal_scale`
/// runs `4.1e-10` to `2.1e-9` — under `CONE_BOUNDARY_REL`'s `1e-8`, but by
/// only a factor of five. `1e-6` keeps a factor of ~500 while a strictly
/// interior point sits `O(1)` above it.
const FACET_ACTIVE_REL: f64 = 1e-6;

/// How far `z` may sit off the ray `ℝ₊∇φ` before the point is refused.
///
/// At the interior of a smooth facet the normal cone **is** that ray, so
/// `z = ν∇φ` is not an approximation — it is the optimality condition. A `z`
/// that is not parallel to `∇φ` means the solution is not the one it claims to
/// be, or is on a lower-dimensional face where the normal cone is wider, and
/// building `ν` from it would answer for the wrong face.
///
/// This is a converged-solution check of the same kind as
/// [`ORTHANT_GUARD_REL`], and is calibrated the same way — off measured
/// populations rather than a round number that looks safe. On the four
/// non-symmetric fixtures above, `‖z − ν∇φ‖∞ / ‖z‖∞` runs `2.8e-8` to
/// `3.4e-5`; a dual deliberately tilted off the ray is `O(1)`. So `1e-3` sits
/// ~30× above everything converged and ~1000× below a genuine mismatch.
/// `1e-6`, the first value tried, refused two of the four correct solutions.
///
/// # The denominator is the block's own `‖z‖∞`, with no problem-scale floor
///
/// It used to be `max(‖z‖∞, dual_scale)`, and `dual_scale` floors at `1.0`. So
/// below unit scale the test went **absolute** and admitted exactly what it
/// exists to refuse: round 6 of #889 measured `s = 1e-6·(1, 0.6, 0.8)` against
/// `z = 1e-6·(1, 0.8, −0.6)` — the tail rotated 90°, maximally
/// non-complementary — served as a `Boundary` face with `ν = 1e-6` fed into the
/// curvature. On a mixed model the same hole opens with no global rescaling at
/// all: one block's small dual beside another's large one.
///
/// Dropping the floor makes the check **scale-equivariant** — scale `(s, z)` by
/// `c` and residual and threshold scale together — which is what a ratio of two
/// dual-space quantities should be. It is safe only because a genuinely
/// collapsed dual is refused *earlier and separately*: [`dual_has_collapsed`]
/// on the second-order path and at `exp_face`/`power_face` before they delegate
/// here, so `‖z‖∞` is meaningfully nonzero by the time this runs. Without that
/// ordering the denominator could vanish and the test would reject round-off.
///
/// The PSD analogue (`⟨S, Z⟩` against `‖S‖∞·‖Z‖∞`) is the same convention one
/// degree up: its residual is quadratic in the scale, and so is its threshold.
const FACET_DUAL_REL: f64 = 1e-3;

fn smooth_facet_face(
    block: usize,
    facet: &SmoothFacet,
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
) -> Result<ConeBlockFace, ConeError> {
    let grad = &facet.grad;
    let gn2: f64 = grad.iter().map(|v| v * v).sum();
    if !gn2.is_finite() || gn2 <= 0.0 {
        return Err(ConeError::new(
            block,
            "the face's normal is zero or not finite at this point, so there is no \
             direction to differentiate along",
        ));
    }
    let nu: f64 = z.iter().zip(grad).map(|(a, b)| a * b).sum::<f64>() / gn2;
    let resid = inf_norm_of(
        &z.iter()
            .zip(grad)
            .map(|(a, b)| a - nu * b)
            .collect::<Vec<_>>(),
    );
    if nu <= 0.0 || resid > FACET_DUAL_REL * inf_norm_of(z) {
        return Err(ConeError::new(
            block,
            "the dual is not on the ray normal to this face, so the block is not in \
             the facet's interior — the face being linearized is not the face the \
             solution is on",
        ));
    }
    let row = combine_rows(grad, g_rows);
    if row.is_empty() {
        return Err(ConeError::new(
            block,
            "the cone normal projects to zero against this block's rows, so there \
             is no direction to differentiate along",
        ));
    }
    let terms: Vec<(f64, Vec<(usize, f64)>)> = facet
        .hess_factors
        .iter()
        .map(|(kappa, v)| (nu * kappa, combine_rows(v, g_rows)))
        .collect();
    Ok((ConeBlockKind::Boundary, vec![row], outer_triplets(&terms)))
}

/// The second-order cone's face, given that `s` is not at the apex.
///
/// `SOC(k) = { (s₀, s₁) : s₀ ≥ ‖s₁‖ }`. Away from the apex the boundary is the
/// smooth hypersurface `φ(s) = s₀ − ‖s₁‖ = 0`, so this is
/// [`smooth_facet_face`]'s shape — but it is written out by hand because
/// `∇²φ = −(0 ⊕ (I − ŝ₁ŝ₁ᵀ))/‖s₁‖` has rank `k − 2`, and factoring it would
/// mean building an orthonormal basis of `ŝ₁`'s complement to say what the
/// closed form ([`soc_boundary_curvature`]) says in `k` rank-one updates.
///
/// # Why the near-apex band is narrow
///
/// On a second-order cone "on the boundary and close to the apex" *is* "close
/// to the apex" — `s₀ = ‖s₁‖`, so one is small exactly when the other is. The
/// near-apex refusal therefore covers only the residue between the two relative
/// tests (`‖s‖∞ > APEX_REL·scale` while `s₀ ≤ APEX_REL·scale`), a thin band by
/// construction rather than by accident. It is still reachable, and
/// `a_boundary_point_too_close_to_the_apex_is_refused` reaches it, because the
/// alternative — a normal `(1, −s₁/s₀)` built by dividing by a number that is
/// all round-off — is the silently-wrong class this crate refuses on principle.
type ConeBlockFace = (
    ConeBlockKind,
    Vec<Vec<(usize, f64)>>,
    Vec<(usize, usize, f64)>,
);

fn soc_face(
    block: usize,
    s: &[f64],
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    let s_tail = s[1..].iter().map(|x| x * x).sum::<f64>().sqrt();
    let s0 = s[0];
    let gap = s0 - s_tail;

    if gap > CONE_BOUNDARY_REL * primal_scale {
        return interior_face(block, s, z, primal_scale, dual_scale);
    }
    if gap < -CONE_BOUNDARY_REL * primal_scale {
        return Err(ConeError::new(
            block,
            "the slack lies outside the cone beyond the solve tolerance, so the \
             block has no face to differentiate along",
        ));
    }
    if dual_has_collapsed(z, dual_scale) {
        return Err(ConeError::new(
            block,
            "the slack is on the cone boundary with a collapsed dual — the conic \
             analogue of a weakly active row, where dx/db is two-valued",
        ));
    }
    if s0 <= CONE_APEX_REL * primal_scale {
        return Err(ConeError::new(
            block,
            "the slack is on the cone boundary but too close to the apex for its \
             normal to be meaningful; dx/db is not single-valued there",
        ));
    }

    // The boundary normal `w = (1, −s₁/s₀)`, as one combined row `wᵀG`.
    let mut w = vec![1.0; s.len()];
    for (r, wr) in w.iter_mut().enumerate().skip(1) {
        *wr = -s[r] / s0;
    }

    // **The dual must lie on that normal ray**, `z = ν·w` with `ν = z₀ > 0`.
    //
    // At a boundary point the normal cone *is* `ℝ₊w`, so this is the optimality
    // condition rather than a refinement — and `ν` is fed straight into the
    // curvature below, so a tilted `z` does not degrade the answer, it makes up
    // a different problem's. [`smooth_facet_face`] has refused exactly this for
    // the exponential and power cones since they were written, with the same
    // constant; the second-order cone did not, and round 5 of #889 reached it
    // with `s = (1, 0.6, 0.8)`, `z = (1, 0.8, −0.6)` — `⟨s, z⟩ = 1.0`, wildly
    // non-complementary — and was served.
    //
    // Reachable whenever the caller's `sol` does not match the partition it
    // hands alongside: a stale solution, a mislabeled block. That is the same
    // mismatched-caller-input class [`SensError::NotOrthantComplementary`]
    // exists to refuse, one cone family over.
    let nu = z[0];
    let dual_resid = inf_norm_of(
        &z.iter()
            .zip(&w)
            .map(|(zi, wi)| zi - nu * wi)
            .collect::<Vec<_>>(),
    );
    if nu <= 0.0 || dual_resid > FACET_DUAL_REL * inf_norm_of(z) {
        return Err(ConeError::new(
            block,
            "the dual is not on the ray normal to this boundary point, so the face \
             being linearized is not the face the solution is on",
        ));
    }

    let combined = combine_rows(&w, g_rows);
    if combined.is_empty() {
        return Err(ConeError::new(
            block,
            "the cone normal projects to zero against this block's rows, so there \
             is no direction to differentiate along",
        ));
    }
    let curvature = soc_boundary_curvature(s, z[0], s0, g_rows);
    Ok((ConeBlockKind::Boundary, vec![combined], curvature))
}

// ---------------------------------------------------------------------------
// The positive-semidefinite cone, at constant rank.
// ---------------------------------------------------------------------------

/// Eigenvalue threshold, relative to the scale of the quantity it is applied
/// to, below which an eigenvalue is zero and the matrix has lost a rank.
const PSD_RANK_REL: f64 = 1e-8;

/// The PSD cone's face: the **constant-rank manifold** through `S = smat(s)`.
///
/// `{ X ⪰ 0 : rank X = r }` is a smooth manifold of codimension `q(q+1)/2` with
/// `q = n − r`. Writing `X` in the (range, kernel) basis of `S` as
/// `[[A, B], [Bᵀ, C]]`, membership is the vanishing of the Schur complement
/// `C − Bᵀ A⁻¹ B`, and at `S` itself (`B = C = 0`) that gives:
///
/// * **the tangent** `Vᵀ dX V = 0` — one row per pair `(a ≤ b)` of kernel
///   vectors, `svec(sym(v_a v_bᵀ))ᵀ G`;
/// * **the curvature** `−2 dBᵀ A⁻¹ dB` with `dB = Uᵀ dX V`, whose contribution
///   to the Lagrangian's `x`-Hessian (multiplier `Λ = Vᵀ Z V`, i.e. `Z = VΛVᵀ`)
///   is
///
///   ```text
///     2 · Σ_{l ≤ r} Σ_{k ≤ q} (λ_k / a_l) · c_lk c_lkᵀ,
///     c_lk = Gᵀ svec(sym(ũ_l w̃_kᵀ))
///   ```
///
///   where `a_l, ũ_l` are `S`'s positive eigenpairs and `λ_k, w̃_k` are `Z`'s.
///   `r·q` rank-one updates, and positive semidefinite as every face's
///   curvature in this file is.
///
/// # Strict complementarity is required, not assumed
///
/// `rank Z = n − rank S` is what makes the kernel of `S` the *whole* normal
/// direction. Where it fails the block is weakly active in the PSD sense —
/// a direction along which slack and multiplier vanish together — and `dx/db`
/// is two-valued, so the block is refused rather than answered on a guess about
/// which side.
#[allow(clippy::too_many_arguments)]
fn psd_face(
    block: usize,
    n: usize,
    s: &[f64],
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    let (s_vals, s_vecs) = sym_spectrum(block, s, n, "primal")?;
    let (z_vals, z_vecs) = sym_spectrum(block, z, n, "dual")?;
    let s_tol = PSD_RANK_REL * primal_scale;
    let z_tol = PSD_RANK_REL * dual_scale;

    if s_vals.iter().any(|&v| v < -s_tol) {
        return Err(ConeError::new(
            block,
            "the slack matrix has a negative eigenvalue beyond the solve tolerance, \
             so it is outside the PSD cone and has no face to differentiate along",
        ));
    }
    if z_vals.iter().any(|&v| v < -z_tol) {
        return Err(ConeError::new(
            block,
            "the dual matrix has a negative eigenvalue beyond the solve tolerance, \
             so it is outside the PSD cone's dual",
        ));
    }

    let kernel: Vec<usize> = (0..n).filter(|&j| s_vals[j] <= s_tol).collect();
    let range: Vec<usize> = (0..n).filter(|&j| s_vals[j] > s_tol).collect();
    let q = kernel.len();
    if q == 0 {
        // `S ≻ 0`: strictly inside the cone.
        return interior_face(block, s, z, primal_scale, dual_scale);
    }
    let dual_range: Vec<usize> = (0..n).filter(|&j| z_vals[j] > z_tol).collect();
    if dual_range.len() != q {
        return Err(ConeError::new(
            block,
            "the PSD block does not satisfy strict complementarity (rank Z ≠ n − rank S), \
             so a direction exists along which slack and multiplier vanish together and \
             dx/db is two-valued",
        ));
    }

    // **The ranks matching is not the same as the subspaces matching.**
    //
    // Complementarity needs `range(Z) ⊆ ker(S)`; counting eigenvalues only says
    // the two have the right *dimensions*. A `Z` of exactly the right rank in a
    // mismatched eigenbasis passes the count and is not complementary at all —
    // and the tangent rows below are built from `S`'s kernel vectors, so a `Z`
    // living somewhere else silently linearizes a face the solution is not on.
    //
    // Both matrices are PSD here (checked above), so `⟨S, Z⟩ = 0` is equivalent
    // to `SZ = 0` and hence to the range containment — one inner product rather
    // than a subspace comparison. `svec` is an isometry, so the dot product of
    // the vectorized forms IS the trace inner product.
    //
    // Round 5 of #889 raised this alongside the second-order cone's missing
    // dual-ray check; the same mismatched-caller-input class, and the same
    // reason it is worth a refusal rather than a caveat.
    let sz_inner: f64 = s.iter().zip(z).map(|(a, b)| a * b).sum();
    if sz_inner.abs() > FACET_DUAL_REL * inf_norm_of(s) * inf_norm_of(z) {
        return Err(ConeError::new(
            block,
            "the PSD block's ranks complement but its subspaces do not — ⟨S, Z⟩ is \
             not zero, so range(Z) is not inside ker(S) and the face being \
             linearized is not the face the solution is on",
        ));
    }

    // Tangent rows: `v_aᵀ dX v_b = 0` for every pair from the kernel.
    let mut rows = Vec::with_capacity(q * (q + 1) / 2);
    let mut buf = vec![0.0; n * (n + 1) / 2];
    for (ai, &a) in kernel.iter().enumerate() {
        for &b in &kernel[ai..] {
            svec_sym_outer(
                &s_vecs[a * n..(a + 1) * n],
                &s_vecs[b * n..(b + 1) * n],
                n,
                &mut buf,
            );
            let row = combine_rows(&buf, g_rows);
            if !row.is_empty() {
                rows.push(row);
            }
        }
    }
    if rows.is_empty() {
        return Err(ConeError::new(
            block,
            "the PSD face's normals all project to zero against this block's rows, so \
             there is no direction to differentiate along",
        ));
    }

    // Curvature: one rank-one update per (range, dual-range) pair.
    let mut terms = Vec::with_capacity(range.len() * dual_range.len());
    for &l in &range {
        for &k in &dual_range {
            svec_sym_outer(
                &s_vecs[l * n..(l + 1) * n],
                &z_vecs[k * n..(k + 1) * n],
                n,
                &mut buf,
            );
            terms.push((2.0 * z_vals[k] / s_vals[l], combine_rows(&buf, g_rows)));
        }
    }
    let kind = if range.is_empty() {
        // `S = 0` reached without tripping the inf-norm apex test upstream —
        // the same face, and it carries no curvature because there is no
        // range block for `A⁻¹` to come from.
        ConeBlockKind::Apex
    } else {
        ConeBlockKind::Boundary
    };
    Ok((kind, rows, outer_triplets(&terms)))
}

/// Eigendecompose `smat(v)`. Returns `(values, vectors)` with eigenvector `j`
/// at `vectors[j*n .. (j+1)*n]`, matching [`pounce_linalg::symmetric_eigen`]'s
/// column-major output.
fn sym_spectrum(
    block: usize,
    v: &[f64],
    n: usize,
    which: &'static str,
) -> Result<(Vec<f64>, Vec<f64>), ConeError> {
    let mut m = vec![0.0; n * n];
    smat(v, n, &mut m);
    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    if !symmetric_eigen(&m, n, &mut vals, &mut vecs) {
        return Err(ConeError::new(
            block,
            match which {
                "primal" => {
                    "the eigensolver did not converge on the PSD block's slack, so \
                             its rank — and therefore its face — cannot be determined"
                }
                _ => {
                    "the eigensolver did not converge on the PSD block's dual, so strict \
                      complementarity cannot be checked"
                }
            },
        ));
    }
    Ok((vals, vecs))
}

/// `svec((u vᵀ + v uᵀ)/2)` into `out`, in [`crate::cones::psd::svec`]'s
/// ordering and `√2` scaling — so `⟨out, w⟩ = uᵀ smat(w) v` for any symmetric
/// `smat(w)`, which is the identity both the rows and the curvature are built
/// on.
fn svec_sym_outer(u: &[f64], v: &[f64], n: usize, out: &mut [f64]) {
    let r2 = std::f64::consts::SQRT_2;
    let mut k = 0;
    for j in 0..n {
        for i in j..n {
            let m = 0.5 * (u[i] * v[j] + u[j] * v[i]);
            out[k] = if i == j { m } else { r2 * m };
            k += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// The non-symmetric cones: exponential and power.
// ---------------------------------------------------------------------------

/// The exponential cone's smooth facet.
///
/// `K_exp = cl { (x, y, z) : y·log(z/y) ≥ x, y > 0, z > 0 }`. Its boundary has
/// two pieces: the smooth surface `φ(s) = y·log(z/y) − x = 0` with `y, z > 0`,
/// and the ray `{ (x, 0, z) : x ≤ 0, z ≥ 0 }` where the cone has no tangent
/// plane. Only the first is answered; the ray is refused.
///
/// ```text
///   ∇φ  = (−1,  log(z/y) − 1,  y/z)
///   ∇²φ = −(1/y) · v vᵀ,   v = (0, 1, −y/z)
/// ```
///
/// The rank-one form of `∇²φ` is exact, not an approximation: `φ` is the
/// perspective of a one-dimensional function, so its Hessian has rank one
/// everywhere on the facet.
fn exp_face(
    block: usize,
    s: &[f64],
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    let (sx, sy, sz) = (s[0], s[1], s[2]);
    let eps = FACET_ACTIVE_REL * primal_scale;
    if sy <= eps || sz <= eps {
        return Err(ConeError::new(
            block,
            "the slack sits on the exponential cone's y = 0 (or z = 0) ray, where the \
             boundary is not a smooth facet and has no single normal",
        ));
    }
    let phi = sy * (sz / sy).ln() - sx;
    if phi > eps {
        return interior_face(block, s, z, primal_scale, dual_scale);
    }
    if phi < -eps {
        return Err(ConeError::new(
            block,
            "the slack lies outside the cone beyond the solve tolerance, so the \
             block has no face to differentiate along",
        ));
    }
    if dual_has_collapsed(z, dual_scale) {
        return Err(ConeError::new(
            block,
            "the slack is on the cone boundary with a collapsed dual — the conic \
             analogue of a weakly active row, where dx/db is two-valued",
        ));
    }
    let facet = SmoothFacet {
        grad: vec![-1.0, (sz / sy).ln() - 1.0, sy / sz],
        hess_factors: vec![(1.0 / sy, vec![0.0, 1.0, -sy / sz])],
    };
    smooth_facet_face(block, &facet, z, g_rows)
}

/// The power cone's smooth facet.
///
/// `K_α = { (x, y, z) : |x| ≤ y^α z^{1−α}, y, z ≥ 0 }`. The smooth piece is
/// `φ(s) = y^α z^{1−α} − |x| = 0` with `y, z > 0`; `y = 0` and `z = 0` are the
/// degenerate faces and are refused.
///
/// There is deliberately **no** guard for the `|x| = 0` kink, and that is a
/// statement about the cone rather than an omission. `φ` is non-differentiable
/// at `x = 0`, but a boundary point with `x = 0` needs `g = y^α z^{1−α} = 0`,
/// which means `y = 0` or `z = 0` — already refused above. With `y, z > ε` the
/// two sheets `x = ±g` are each smooth and never meet. A guard here would be
/// unreachable code that reads like coverage, which is worse than no guard:
/// see `the_power_cones_x_kink_is_not_on_the_facet`, which asserts the
/// geometry rather than the branch.
///
/// With `g = y^α z^{1−α}`,
///
/// ```text
///   ∇φ  = (−sign x,  α g / y,  (1−α) g / z)
///   ∇²φ = −α(1−α) g · v vᵀ,   v = (0, 1/y, −1/z)
/// ```
///
/// Rank one for the same reason the exponential cone's is: `g` is a geometric
/// mean, homogeneous of degree one, so its Hessian annihilates the ray through
/// the point.
#[allow(clippy::too_many_arguments)]
fn power_face(
    block: usize,
    alpha: f64,
    s: &[f64],
    z: &[f64],
    g_rows: &[Vec<(usize, f64)>],
    primal_scale: f64,
    dual_scale: f64,
) -> Result<ConeBlockFace, ConeError> {
    let (sx, sy, sz) = (s[0], s[1], s[2]);
    let eps = FACET_ACTIVE_REL * primal_scale;
    if sy <= eps || sz <= eps {
        return Err(ConeError::new(
            block,
            "the slack sits on the power cone's y = 0 or z = 0 face, where the boundary \
             is not a smooth facet and has no single normal",
        ));
    }
    let g = sy.powf(alpha) * sz.powf(1.0 - alpha);
    let phi = g - sx.abs();
    if phi > eps {
        return interior_face(block, s, z, primal_scale, dual_scale);
    }
    if phi < -eps {
        return Err(ConeError::new(
            block,
            "the slack lies outside the cone beyond the solve tolerance, so the \
             block has no face to differentiate along",
        ));
    }
    if dual_has_collapsed(z, dual_scale) {
        return Err(ConeError::new(
            block,
            "the slack is on the cone boundary with a collapsed dual — the conic \
             analogue of a weakly active row, where dx/db is two-valued",
        ));
    }
    let k = alpha * (1.0 - alpha) * g;
    let facet = SmoothFacet {
        grad: vec![-sx.signum(), alpha * g / sy, (1.0 - alpha) * g / sz],
        hess_factors: vec![(k, vec![0.0, 1.0 / sy, -1.0 / sz])],
    };
    smooth_facet_face(block, &facet, z, g_rows)
}

/// The `(x,x)` Hessian a second-order block on its boundary contributes, as
/// lower-triangle triplets.
///
/// The active constraint is `φ(s) = s₀ − ‖s₁‖ = 0` with `s = h − Gx`, and its
/// multiplier is `ν = z₀` (the dual sits on the dual cone's boundary, so
/// `z = ν∇φ` and `∇φ₀ = 1`). Since `∇²φ` is `0 ⊕ −(I − ŝ₁ŝ₁ᵀ)/‖s₁‖` and the
/// Lagrangian carries `−νφ`, the contribution is
///
/// ```text
///   (ν/‖s₁‖) · Gᵀ (0 ⊕ (I − ŝ₁ŝ₁ᵀ)) G   =   (ν/s₀) · (Σ_{r≥1} gᵣgᵣᵀ − u uᵀ)
/// ```
///
/// with `u = Σ_{r≥1} ŝᵣ gᵣ` and `ŝᵣ = sᵣ/s₀` (on the boundary `‖s₁‖ = s₀`). The
/// right-hand form is used because it is `d` rank-one updates rather than `d²`,
/// and because `u` is the same vector the active row is built from: that row is
/// `g₀ − u`.
///
/// # Cost
///
/// The result is as dense as the outer products make it — up to the square of
/// the block's column support. That is affordable at the block sizes a
/// second-order cone is normally posed at and is *not* a claim about a cone
/// spanning thousands of columns.
fn soc_boundary_curvature(
    s: &[f64],
    nu: f64,
    s0: f64,
    g_rows: &[Vec<(usize, f64)>],
) -> Vec<(usize, usize, f64)> {
    let scale = nu / s0;
    if scale == 0.0 {
        return Vec::new();
    }
    let mut acc: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut outer = |a: &[(usize, f64)], b: &[(usize, f64)], w: f64| {
        for &(r, vr) in a {
            for &(c, vc) in b {
                if r < c {
                    continue; // lower triangle only
                }
                *acc.entry((r, c)).or_insert(0.0) += w * vr * vc;
            }
        }
    };
    // Σ_{r≥1} gᵣ gᵣᵀ
    for row in &g_rows[1..] {
        outer(row, row, scale);
    }
    // −u uᵀ, with u = Σ_{r≥1} ŝᵣ gᵣ
    let mut u: BTreeMap<usize, f64> = BTreeMap::new();
    for (r, row) in g_rows.iter().enumerate().skip(1) {
        let sr = s[r] / s0;
        for &(col, val) in row {
            *u.entry(col).or_insert(0.0) += sr * val;
        }
    }
    let u: Vec<(usize, f64)> = u.into_iter().filter(|&(_, v)| v != 0.0).collect();
    outer(&u, &u, -scale);
    acc.into_iter()
        .filter(|&(_, v)| v != 0.0)
        .map(|((r, c), v)| (r, c, v))
        .collect()
}

/// Post-optimal sensitivity for a solved convex QP.
///
/// Holds the factored active-set KKT system at the optimum. Build it once
/// from a [`QpProblem`] and its [`QpSolution`], then call
/// [`parametric_step`](Self::parametric_step) for each parameter
/// perturbation — the factorization is reused across queries.
pub struct QpSensitivity {
    n: usize,
    m_eq: usize,
    /// KKT dimension `n + m_eq + n_active`.
    dim: usize,
    /// Shared so [`backsolver`](QpSensitivity::backsolver) can hand the same
    /// factorization to the core's machinery — see [`QpKktBacksolver`].
    fact: Rc<RefCell<Factorization>>,
    /// Problem data, retained for the reduced-Hessian projection.
    prob: QpProblem,
    /// Active inequality rows (indices into `G`). On the conic path this names
    /// the underlying `G` rows a cone block contributed, which is provenance
    /// rather than the active object itself — see [`active_rows`].
    active_ineq: Vec<usize>,
    /// The active inequality-side rows of `B_a`, as sparse `(col, val)` rows.
    ///
    /// On the orthant path these are just `G`'s active rows. A cone block's
    /// contribution is not a row of `G` at all — it is a combination of them
    /// (the cone's normal at the boundary point), or the whole block at an apex
    /// — so the assembly and the reduced Hessian both read this rather than
    /// re-deriving from indices.
    active_rows: Vec<Vec<(usize, f64)>>,
    /// Variables whose bound is active (one row each).
    active_bound_vars: Vec<usize>,
    /// The solution this sensitivity was built at, retained because the bound
    /// refinement measures the step against the base point and the duality
    /// measure needs the slacks.
    base_x: Vec<f64>,
    base_z: Vec<f64>,
    base_z_lb: Vec<f64>,
    base_z_ub: Vec<f64>,
    /// The same set, carrying which side is active: `true` = lower bound.
    /// The orientation is what `assemble_kkt` signs the row by and what the
    /// recovered `dz_a` block is read against.
    active_bounds: Vec<(usize, bool)>,
    /// Each active bound's multiplier at the base point, oriented so it is the
    /// non-negative multiplier of the row `assemble_kkt` emitted. Needed by the
    /// release half of fix-relax, which has to move a released multiplier onto
    /// its variable's `x` row.
    bound_base_mult: Vec<f64>,
    /// `true` when this is a pure LP (`P = 0`) whose solve did not run
    /// crossover — see [`lp_without_crossover`](QpSensitivity::lp_without_crossover).
    lp_without_crossover: bool,
    /// Per second-order block, what it was doing at the solution. Empty on the
    /// orthant path.
    cone_kinds: Vec<(usize, ConeBlockKind)>,
    /// The active faces' curvature, as lower-triangle `(row, col, val)` into
    /// the `(x,x)` block. Empty on the orthant path, where every active face is
    /// a hyperplane and carries none.
    ///
    /// Retained rather than consumed by [`assemble_kkt`] because
    /// [`reduced_hessian`](QpSensitivity::reduced_hessian) needs it too: on a
    /// curved face the second-order object is the Hessian of the **Lagrangian**
    /// restricted to the tangent space, and projecting bare `P` there is the
    /// same omission that made the first draft of the step converge to the
    /// wrong derivative. Round 5 of #889 found `reduced_hessian` still doing
    /// it.
    curvature: Vec<(usize, usize, f64)>,
    /// Inequality rows at which strict complementarity fails (gh #219).
    weakly_active_ineq: Vec<usize>,
    /// Variables whose bound is weakly active.
    weakly_active_bound_vars: Vec<usize>,
    /// Lower-triangle KKT pattern (1-based), shared by the factored
    /// (regularized) matrix and the unregularized values below.
    kkt_airn: Vec<Index>,
    kkt_ajcn: Vec<Index>,
    /// Unregularized KKT values (the `δ`-free matrix) for the refinement
    /// residual — see [`solve_refined`] (gh #284).
    kkt_vals_true: Vec<f64>,
    /// Hager 1-norm estimate of `κ₁` of the factored KKT (gh #284).
    kkt_cond_estimate: f64,
    /// Relative KKT residual of the most recent parametric step, or `None`
    /// before any step has been taken (gh #284).
    last_residual: Option<f64>,
    /// Reusable iterative-refinement buffers.
    ir_scratch: IrScratch,
    /// Factored (regularized) KKT values; a release starts from these.
    kkt_vals_reg: Rc<Vec<f64>>,
    /// Per active bound, the value slots a release neutralizes.
    release_slots: Rc<Vec<(usize, usize)>>,
    /// One spare linear-solver instance, drawn from the caller's factory at
    /// build, reserved for the released system.
    ///
    /// Storing an *instance* rather than the factory is what keeps
    /// [`build`](QpSensitivity::build)'s signature unchanged — boxing the
    /// factory would have needed an `F: 'static` bound on a published method.
    /// One instance is enough because at most one released `Factorization` is
    /// ever created: a different released set refactors it in place, the
    /// sparsity pattern being identical.
    release_backend: Rc<RefCell<Option<Box<dyn SparseSymLinearSolverInterface>>>>,
}

/// Relative threshold below which a slack or a multiplier counts as zero for
/// the weak-activity screen (see [`QpSensitivity::weakly_active_ineq`]).
///
/// Deliberately loose, because it is the *conjunction* that carries the signal:
/// a constraint must be binding in the primal **and** carry a negligible dual
/// at the same time. Either alone is ordinary — every active constraint has
/// zero slack, every inactive one has zero multiplier — while both at once is
/// exactly the non-strict complementarity that makes `dx/db` one-sided.
///
/// The magnitude is set by how these quantities actually behave. At a
/// degenerate optimum both collapse together at roughly `√tol`: on gh #219's
/// QP the multiplier and slack measure `(3.8e-5, 1.7e-4)` at `tol = 1e-8`,
/// `(2.0e-7, 9.0e-7)` at `1e-12`, and `(2.9e-8, 1.3e-7)` at `1e-14` — their
/// ratio pinned near 0.22 across six orders of magnitude. A threshold tight
/// enough to look precise would simply miss the default-tolerance case, which
/// is the one users hit.
const WEAK_ACTIVE_REL: f64 = 1e-3;

/// Relative margin for the orthant guard's three row-wise tests
/// ([`SensError::NotOrthantComplementary`]).
///
/// Deliberately loose. The guard separates two *categorically* different
/// inputs, not two nearby numbers: a converged orthant row has `sᵢ·zᵢ ≈ μ`,
/// which is at round-off relative to `‖s‖∞·‖z‖∞`, while a second-order-cone
/// row on its boundary has `s₀z₀ = c·s₀²`, which is `O(1)` on the same scale.
///
/// # The measured populations on each side
///
/// Worst row of each fixture, as the ratio this constant is compared against:
///
/// | fixture | `|sᵢzᵢ| / (‖s‖∞‖z‖∞)` | `max(−zᵢ) / ‖z‖∞` | verdict |
/// |---|---|---|---|
/// | `SecondOrder(3)` through `solve_socp_ipm` | **1.3e-1** | **1.2e-1** | must refuse |
/// | convex QP, one active inequality | 1.7e-9 | 0 | must accept |
/// | weakly-active (non-strictly-complementary) QP | 6.5e-9 | 0 | must accept |
///
/// The gap is **7.9 orders wide** and `1e-4` sits near its centre — about five
/// orders below anything that must be refused and about five above anything
/// that must be accepted. The weakly-active row is the one worth checking,
/// since non-strict complementarity is the accept-side case that comes closest
/// to the boundary, and it is still five orders clear.
///
/// The asymmetry is deliberate: a false *positive* is the failure that would
/// matter, because this guard sits on the path of every convex sensitivity
/// build, so the threshold is placed far from the accept side.
const ORTHANT_GUARD_REL: f64 = 1e-4;

/// Above this 1-norm condition estimate of the factored KKT the parametric
/// step is reported [`ill_conditioned`](QpSensitivity::ill_conditioned).
///
/// Calibrated against gh #284's near-LICQ sweep. With the static `δ = 1e-10`
/// flooring the smallest KKT singular value, `κ₁` saturates near `1e16` on a
/// numerically singular KKT while the genuinely well-conditioned sensitivity
/// cases sit at `κ₁ ≈ 3–8e9`. Iterative refinement (see the module doc)
/// recovers a correct `dx/db` up to `κ₁ ≈ 6e13`; past that the information is
/// below the double-precision floor and refinement cannot help. The threshold
/// sits in the wide gap between those regimes, so it fires exactly on the
/// unrecoverable cases and stays quiet on every case refinement rescues — no
/// false alarm on the well-conditioned equality-only or active-set paths.
///
/// This condition estimate is a *build-time* screen, and by itself it has a
/// blind spot: it is the `κ₁` of the **regularized** factor, whose smallest
/// singular value is floored at `δ`, so on a *well-scaled* `P` (e.g. `P = I`,
/// `‖K‖₁ ≈ O(1)`) it saturates near `‖K‖₁ / δ ≈ 3e10` no matter how nearly
/// parallel the active rows become — never reaching this threshold even when
/// the true KKT is numerically singular (gh #328). The per-step residual gate
/// below closes that gap.
const KKT_ILL_CONDITIONED_THRESHOLD: f64 = 1e14;

/// Relative KKT residual above which the most recent parametric step is treated
/// as *unreliable*, so [`ill_conditioned`](QpSensitivity::ill_conditioned)
/// fires on it (gh #328).
///
/// This is the companion signal to the build-time condition estimate and covers
/// its blind spot. When the active-constraint Jacobian is near-LICQ but `P` is
/// well scaled, the saturating [`KKT_ILL_CONDITIONED_THRESHOLD`] never trips,
/// yet iterative refinement against the true (`δ`-free) KKT *cannot* solve the
/// step — it stalls at a large relative residual (`≈ 3e-2` at `κ(A) ≈ 2e5`,
/// `≈ 0.25` at `κ(A) ≈ 2e7`). A well-solved step, by contrast, refines to
/// round-off (`≲ 1e-8`). The two regimes are separated by many orders of
/// magnitude, so this threshold sits comfortably in the gap: it flags exactly
/// the steps whose returned `dx/db` does not satisfy the true KKT, and stays
/// quiet on every accurately recovered step.
const STEP_UNRELIABLE_RESIDUAL: f64 = 1e-6;

/// Iterative-refinement passes for a parametric step (mirrors the HSDE
/// solve's `IR_MAX_PASSES`). A handful suffices: refinement against the
/// unregularized KKT converges geometrically until it hits the near-singular
/// floor, where it stagnates and stops.
/// Margin below which a bound multiplier counts as driven negative by a step.
///
/// The NLP arm derives this from `bound_relax_factor`, since there the solve's
/// own bound relaxation sets the scale. The convex path does not relax bounds
/// in that sense, so this is the floor that derivation bottoms out at: `1e-9`,
/// which upstream also uses when `bound_relax_factor` is unset or unreadable.
const RELEASE_FLOOR: f64 = 1e-9;

const IR_MAX_PASSES: usize = 5;

/// Relative-residual target below which refinement stops early (the KKT step
/// is solved to working precision).
const IR_RELTOL: f64 = 1e-12;

/// Hager/Higham 1-norm power iterations for the `‖K⁻¹‖₁` estimate. Five is
/// LAPACK's `dlacon` default and is more than enough here (the estimate
/// matched the exact 1-norm condition on gh #284's sweep).
const HAGER_ITERS: usize = 5;

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Symmetric matvec `y ← K x` for lower-triangle KKT triplets (`airn`/`ajcn`
/// 1-based, `row ≥ col`). Each strictly-lower entry hits both `y[i]` and
/// `y[j]`; the diagonal once. Mirrors the HSDE solver's `kkt_matvec` — used
/// to form the residual `rhs − K u` that drives refinement.
fn kkt_matvec(airn: &[Index], ajcn: &[Index], vals: &[f64], x: &[f64], y: &mut [f64]) {
    for v in y.iter_mut() {
        *v = 0.0;
    }
    for k in 0..vals.len() {
        let i = (airn[k] - 1) as usize;
        let j = (ajcn[k] - 1) as usize;
        let v = vals[k];
        y[i] += v * x[j];
        if i != j {
            y[j] += v * x[i];
        }
    }
}

/// 1-norm `‖K‖₁ = maxⱼ Σᵢ |Kᵢⱼ|` of a symmetric matrix from its lower-triangle
/// triplets (equal to `‖K‖∞`). Each off-diagonal entry contributes to both its
/// row and column absolute sums.
fn one_norm_sym(dim: usize, airn: &[Index], ajcn: &[Index], vals: &[f64]) -> f64 {
    let mut colsum = vec![0.0_f64; dim];
    for k in 0..vals.len() {
        let i = (airn[k] - 1) as usize;
        let j = (ajcn[k] - 1) as usize;
        let a = vals[k].abs();
        colsum[i] += a;
        if i != j {
            colsum[j] += a;
        }
    }
    colsum.into_iter().fold(0.0_f64, f64::max)
}

/// Hager/Higham lower-bound estimate of `‖K⁻¹‖₁` using only back-solves
/// against the cached factor. `K` is symmetric, so `K⁻ᵀ = K⁻¹` and a single
/// factor drives both half-steps. Returns `∞` if a back-solve fails (the
/// caller then reports an infinite condition estimate — the safe direction).
fn estimate_inv_norm1(fact: &mut Factorization, dim: usize) -> f64 {
    if dim == 0 {
        return 0.0;
    }
    let mut x = vec![1.0 / dim as f64; dim];
    let mut est = 0.0_f64;
    let mut prev_j = usize::MAX;
    for _ in 0..HAGER_ITERS {
        // y = K⁻¹ x; the 1-norm of y is the running estimate of ‖K⁻¹‖₁.
        let mut y = x.clone();
        if fact.solve_one(&mut y).is_err() {
            return f64::INFINITY;
        }
        est = y.iter().map(|v| v.abs()).sum();
        // z = K⁻¹ sign(y)  (K symmetric ⇒ K⁻ᵀ = K⁻¹).
        let mut z: Vec<f64> = y
            .iter()
            .map(|v| if *v >= 0.0 { 1.0 } else { -1.0 })
            .collect();
        if fact.solve_one(&mut z).is_err() {
            return f64::INFINITY;
        }
        let (j, zmax) = z
            .iter()
            .enumerate()
            .fold((0usize, 0.0_f64), |(bi, bm), (i, v)| {
                if v.abs() > bm { (i, v.abs()) } else { (bi, bm) }
            });
        let ztx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
        // Higham's stopping test: no coordinate of z beats the current
        // direction, or we would revisit a unit vector (a cycle).
        if zmax <= ztx || j == prev_j {
            break;
        }
        prev_j = j;
        x = vec![0.0; dim];
        x[j] = 1.0;
    }
    est
}

/// Solve `K u = rhs` against the cached (regularized) factor, then refine `u`
/// against the **unregularized** KKT triplets `(airn, ajcn, vals_true)` to
/// strip the `O(δ)` regularization bias. Overwrites `rhs` with `u` and returns
/// the final **relative** residual `‖rhs₀ − K u‖∞ / ‖rhs₀‖∞` — the reliability
/// signal a caller reads back as
/// [`last_step_residual`](QpSensitivity::last_step_residual).
///
/// The residual is normalized by `‖rhs₀‖∞` (not `1 + ‖rhs₀‖∞`) so it is a true
/// *relative* residual, invariant to the magnitude of the perturbation. The
/// `1 +` floor of the earlier form (gh #284) silently masked a failed solve
/// whenever the perturbation was small: a step scaled by e.g. `1e-6` shrank
/// both `‖r‖` and `‖rhs₀‖` by `1e-6`, but the `1 +` left the denominator at
/// `≈ 1`, so a fully over-damped step (true relative residual `≈ 0.25`) read
/// back as `≈ 2.5e-7` — small enough to look solved (gh #328).
/// The assembled lower triangle of the active-set KKT, in the 1-based triplet
/// form [`Factorization`] takes.
///
/// `vals_reg` is what gets factored (the `δ`-stabilized, indefinite matrix);
/// `vals_true` is the same pattern with the regularization removed and is what
/// iterative refinement measures its residual against (gh #284). They share one
/// sparsity pattern by construction, which is what lets a release re-use the
/// symbolic factor.
struct KktPattern {
    airn: Vec<Index>,
    ajcn: Vec<Index>,
    vals_true: Vec<f64>,
    vals_reg: Vec<f64>,
    dim: usize,
}

/// Assemble the active-set KKT
///
/// ```text
///   ⎡ H    Aᵀ   B_aᵀ ⎤
///   ⎢ A    0    0    ⎥
///   ⎣ B_a  0    0    ⎦
/// ```
///
/// `B_a` stacks the active inequality rows of `G` then one row per active
/// variable bound, in the order given. Every diagonal slot is materialized
/// (even where `P` is zero) so `vals_true` and `vals_reg` share one pattern.
///
/// # `H` is `P` plus the active set's own curvature
///
/// For an orthant row or a variable bound, `curvature` is empty and `H = P`:
/// the active face is a hyperplane, so it contributes no second derivative and
/// the KKT's `(x,x)` block is the objective's alone. That is why the parameter
/// did not exist before the conic arm.
///
/// A **curved** active face does contribute. A second-order block sitting on
/// its boundary is active through `φ(s) = s₀ − ‖s₁‖ = 0`, and the Lagrangian
/// carries `−ν φ(h − Gx)` whose `x`-Hessian is `(ν/‖s₁‖)·Gᵀ(0 ⊕ (I − ŝ₁ŝ₁ᵀ))G`
/// — nonzero, positive semidefinite, and **not** optional. Dropping it does not
/// make the step approximate; it makes it a first-order-wrong number that
/// converges to the wrong derivative as `δ → 0`. Measured on the fixture in
/// `crates/pounce-convex/tests/convex_soc_sensitivity.rs`: `dx/db` reads
/// `(0.348, 0.652)` against a true `(0.5, 0.5)`, at every `δ`. It is the
/// re-solve oracle that says so — every internal residual is happy, because the
/// step solves the KKT it was *given* exactly. `/sens-review` entry 5.
///
/// # Bound-row orientation
///
/// `active_bounds` carries `(variable, is_lower)`, and the sign matters. In the
/// `Gx ≤ h` orientation the convex form uses, a lower bound `lb ≤ xⱼ` is the row
/// `−eⱼᵀ` (its multiplier is `z_lb ≥ 0`, entering stationarity as `−z_lb`), and
/// an upper bound `xⱼ ≤ ub` is `+eⱼᵀ`. Emitting `+1` for both — as this code did
/// before the sign was needed — is invisible while the active block's
/// right-hand side is zero, because `eⱼᵀ dx = 0` and `−eⱼᵀ dx = 0` are the same
/// constraint. It stops being invisible the moment anything *reads* the
/// recovered multiplier block `dz_a`, where the lower-bound entries come back
/// negated. `the_recovered_bound_multipliers_carry_the_solutions_sign` pins it.
fn assemble_kkt(
    prob: &QpProblem,
    active_rows: &[Vec<(usize, f64)>],
    active_bounds: &[(usize, bool)],
    curvature: &[(usize, usize, f64)],
    reg: f64,
) -> KktPattern {
    let n = prob.n;
    let m_eq = prob.m_eq();
    let n_active = active_rows.len() + active_bounds.len();
    let dim = n + m_eq + n_active;

    let mut entries: BTreeMap<(usize, usize), (f64, f64)> = BTreeMap::new();
    let mut add = |r: usize, c: usize, v: f64, reg_off: f64| {
        let (r, c) = if r >= c { (r, c) } else { (c, r) };
        let e = entries.entry((r, c)).or_insert((0.0, 0.0));
        e.0 += v;
        e.1 += reg_off;
    };

    // (x,x): P plus the active set's curvature, with +δ on the diagonal for
    // the factor only.
    for t in &prob.p_lower {
        add(t.row, t.col, t.val, 0.0);
    }
    for &(r, c, v) in curvature {
        debug_assert!(r >= c, "curvature triplets are the lower triangle");
        add(r, c, v, 0.0);
    }
    for i in 0..n {
        add(i, i, 0.0, reg);
    }
    // (y,x): A; (y,y): −δI (factor only).
    for t in &prob.a {
        add(n + t.row, t.col, t.val, 0.0);
    }
    for i in 0..m_eq {
        add(n + i, n + i, 0.0, -reg);
    }
    // Active-row block `B_a`: active inequality rows, then active bound rows.
    let abase = n + m_eq;
    for (k, row) in active_rows.iter().enumerate() {
        for &(col, val) in row {
            add(abase + k, col, val, 0.0);
        }
    }
    for (k, &(j, is_lower)) in active_bounds.iter().enumerate() {
        let sign = if is_lower { -1.0 } else { 1.0 };
        add(abase + active_rows.len() + k, j, sign, 0.0);
    }
    for k in 0..n_active {
        add(abase + k, abase + k, 0.0, -reg);
    }

    let nnz = entries.len();
    let mut airn = Vec::with_capacity(nnz);
    let mut ajcn = Vec::with_capacity(nnz);
    let mut vals_true = Vec::with_capacity(nnz);
    let mut vals_reg = Vec::with_capacity(nnz);
    for ((r, c), (v_true, v_reg_off)) in entries {
        airn.push((r + 1) as Index);
        ajcn.push((c + 1) as Index);
        vals_true.push(v_true);
        vals_reg.push(v_true + v_reg_off);
    }
    KktPattern {
        airn,
        ajcn,
        vals_true,
        vals_reg,
        dim,
    }
}

/// For each active bound, the two value-array slots a release has to touch:
/// the `±1` coupling the multiplier row to its variable, and that row's own
/// diagonal.
///
/// Precomputed once at build so a release is an array rewrite plus a
/// [`Factorization::refactor`] — the sparsity pattern is unchanged by
/// neutralizing a row, so the symbolic factorization is reused.
fn release_slots(
    pat: &KktPattern,
    abase: usize,
    n_ineq_active: usize,
    active_bounds: &[(usize, bool)],
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(active_bounds.len());
    for (k, &(j, _)) in active_bounds.iter().enumerate() {
        let row = abase + n_ineq_active + k;
        // 1-based in the pattern arrays.
        let (r1, j1) = ((row + 1) as Index, (j + 1) as Index);
        let mut coupling = usize::MAX;
        let mut diagonal = usize::MAX;
        for (idx, (&a, &b)) in pat.airn.iter().zip(pat.ajcn.iter()).enumerate() {
            if a == r1 && b == j1 {
                coupling = idx;
            } else if a == r1 && b == r1 {
                diagonal = idx;
            }
        }
        debug_assert!(
            coupling != usize::MAX && diagonal != usize::MAX,
            "every active bound row has a coupling entry and a diagonal"
        );
        out.push((coupling, diagonal));
    }
    out
}

/// Reusable buffers for [`solve_refined`].
///
/// Hoisted out of the function because [`QpKktBacksolver::solve`] is called
/// once per candidate row per refinement pass — `refine_step_onto_bounds` makes
/// `k + 1` back-solves per pass — so a per-call `Vec` allocation turns `O(1)`
/// into `O(k)` for no reason.
#[derive(Default, Clone)]
struct IrScratch {
    b: Vec<f64>,
    r: Vec<f64>,
}

impl IrScratch {
    fn ready(&mut self, dim: usize) {
        self.b.clear();
        self.b.resize(dim, 0.0);
        self.r.clear();
        self.r.resize(dim, 0.0);
    }
}

fn solve_refined(
    fact: &mut Factorization,
    airn: &[Index],
    ajcn: &[Index],
    vals_true: &[f64],
    rhs: &mut [f64],
    scratch: &mut IrScratch,
) -> Result<f64, ()> {
    let dim = rhs.len();
    scratch.ready(dim);
    let b = &mut scratch.b;
    b.copy_from_slice(rhs);
    let b = &*b;
    fact.solve_one(rhs).map_err(|_| ())?;
    // True relative residual: divide by ‖rhs₀‖, flooring only a genuinely zero
    // RHS (whose exact solution is the zero step, residual zero) to avoid 0/0.
    let bnorm = {
        let bn = inf_norm(&b);
        if bn > 0.0 { bn } else { 1.0 }
    };
    let r = &mut scratch.r;
    let mut res = f64::INFINITY;
    for _ in 0..IR_MAX_PASSES {
        kkt_matvec(airn, ajcn, vals_true, rhs, r);
        for k in 0..dim {
            r[k] = b[k] - r[k];
        }
        let new_res = inf_norm(r) / bnorm;
        // Stop when solved to working precision, or when refinement stops
        // making progress — the latter is the near-singular floor and its
        // residual is exactly the "step is unreliable" signal.
        if new_res <= IR_RELTOL || new_res >= res {
            res = new_res;
            break;
        }
        res = new_res;
        fact.solve_one(r).map_err(|_| ())?;
        for k in 0..dim {
            rhs[k] += r[k];
        }
    }
    Ok(res)
}

impl QpSensitivity {
    /// Build the active-set sensitivity for `sol` (a solution of `prob`).
    ///
    /// The active set is read from the dual certificate: an inequality row
    /// `i` is active when `zᵢ > active_tol`, a lower bound on `xⱼ` when
    /// `z_lbⱼ > active_tol`, an upper bound when `z_ubⱼ > active_tol`. A
    /// good default for `active_tol` is `1e-7` (see
    /// [`build_default`](Self::build_default)).
    ///
    /// Returns [`SensError::NotOptimal`] unless `sol` is `Optimal` or
    /// `OptimalInaccurate` (see gh #880), or
    /// [`SensError::FactorizationFailed`] if the active-set KKT is singular.
    pub fn build<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        // gh #880 admits `OptimalInaccurate` here. That status now also means
        // "the `σ` cascade could not certify this point to `tol`", and such a
        // point previously arrived as a clean `Optimal` and built a
        // sensitivity object. The issue is about a wrong *label*; narrowing
        // what the library will do for that population would be a second,
        // unrelated change, and it is the one the repo has already ruled on
        // in the other direction — `_curve_fit.py` (gh #119 / #123) records
        // that treating `Solved_To_Acceptable_Level` as a non-success
        // "reported `success=False` at a verified optimum ... and callers
        // gating on `.success` discarded valid fits". The derivative at a
        // less accurate point is less accurate, exactly as it was before.
        if !matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
            return Err(SensError::NotOptimal);
        }
        let n = prob.n;
        let reg = opts.reg;

        // Active set: which inequality rows and which variable bounds bind.
        let active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| sol.z[i] > active_tol)
            .collect();
        // A bound contributes one row per variable, oriented by which side is
        // active: `−eⱼᵀ` for a lower bound, `+eⱼᵀ` for an upper one (see
        // `assemble_kkt`). A variable active on both sides is a fixed variable
        // (`lb == ub`); it gets one row, and the larger multiplier decides the
        // orientation, which is the side the solve actually leaned on.
        let active_bounds: Vec<(usize, bool)> = (0..n)
            .filter(|&j| sol.z_lb[j] > active_tol || sol.z_ub[j] > active_tol)
            .map(|j| (j, sol.z_lb[j] >= sol.z_ub[j]))
            .collect();

        // Weak activity (gh #219): binding in the primal *and* negligible in
        // the dual, i.e. non-strict complementarity. Classical post-optimal
        // sensitivity (Fiacco) assumes this never happens; where it does, the
        // perturbation changes the active set and `dx/db` is a one-sided
        // derivative with another, equally valid, value on the other side.
        //
        // Both tests are relative to the natural scale of their own quantity,
        // so the screen is invariant to a rescaling of the problem data.
        let inf_norm = |v: &[f64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let dual_scale = inf_norm(&sol.y)
            .max(inf_norm(&sol.z))
            .max(inf_norm(&sol.z_lb))
            .max(inf_norm(&sol.z_ub))
            .max(1.0);
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&sol.x, &mut gx);
        let primal_scale = inf_norm(&prob.h).max(inf_norm(&gx)).max(1.0);

        // Before reading a single row as an orthant row, check that it is one.
        // `solve_socp_ipm` hands back this very `QpSolution` type and the cone
        // partition travels beside it, so without this a solved SOCP is
        // accepted here and answered wrongly in silence.
        check_orthant_complementarity(prob, sol, &gx, primal_scale, dual_scale)?;

        let dual_zero = WEAK_ACTIVE_REL * dual_scale;
        let primal_zero = WEAK_ACTIVE_REL * primal_scale;

        let weakly_active_ineq: Vec<usize> = (0..prob.m_ineq())
            .filter(|&i| (prob.h[i] - gx[i]).abs() <= primal_zero && sol.z[i] <= dual_zero)
            .collect();
        let x_scale = inf_norm(&sol.x).max(1.0);
        let bound_zero = WEAK_ACTIVE_REL * x_scale;
        let weakly_active_bound_vars: Vec<usize> = (0..n)
            .filter(|&j| {
                // `lb`/`ub` may be empty (= unbounded), and a "present" bound
                // is one inside the `BOUND_INF` sentinel band, matching
                // `QpProblem::has_bounds`.
                let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
                let lb_weak = lb > -BOUND_INF
                    && (sol.x[j] - lb).abs() <= bound_zero
                    && sol.z_lb[j] <= dual_zero;
                let ub_weak = ub < BOUND_INF
                    && (ub - sol.x[j]).abs() <= bound_zero
                    && sol.z_ub[j] <= dual_zero;
                lb_weak || ub_weak
            })
            .collect();

        let all_g_rows = group_rows_by_index(&prob.g, prob.m_ineq());
        let active_rows: Vec<Vec<(usize, f64)>> =
            active_ineq.iter().map(|&i| all_g_rows[i].clone()).collect();
        Self::finish(
            prob,
            sol,
            reg,
            active_rows,
            active_ineq,
            active_bounds,
            weakly_active_ineq,
            weakly_active_bound_vars,
            Vec::new(),
            // An orthant row and a variable bound are both hyperplanes: no
            // curvature, which is why this parameter did not exist before the
            // conic arm.
            Vec::new(),
            opts.crossover,
            make_backend,
        )
    }

    /// The conic entry's construction path, after cone classification has
    /// produced the active rows.
    #[allow(clippy::too_many_arguments)]
    fn build_from_rows<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        active_rows: Vec<Vec<(usize, f64)>>,
        active_ineq: Vec<usize>,
        cone_kinds: Vec<(usize, ConeBlockKind)>,
        curvature: Vec<(usize, usize, f64)>,
        orthant_rows: Vec<usize>,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        let n = prob.n;
        let active_bounds: Vec<(usize, bool)> = (0..n)
            .filter(|&j| sol.z_lb[j] > active_tol || sol.z_ub[j] > active_tol)
            .map(|j| (j, sol.z_lb[j] >= sol.z_ub[j]))
            .collect();

        // ---- weak-activity screens on a conic build -----------------------
        //
        // These were both `Vec::new()`, so `weakly_active_ineq()` and
        // `weakly_active_bound_vars()` returned empty on *every* conic build
        // while their docs promised a "deliberately conservative" screen.
        // Round 5 of #889.
        //
        // The two halves are not the same story:
        //
        // * **Variable bounds are ordinary bounds.** A cone in the inequality
        //   block does not change what `x_j` sitting on `lb_j` with a vanishing
        //   multiplier means, so this screen is simply run, exactly as `build`
        //   runs it.
        // * **Rows are screened only where the row is an orthant row** — the
        //   `Nonneg` blocks of a mixed partition. A *cone* block has no weak
        //   row to report, and that is by construction rather than by omission:
        //   `cone_block_face` **refuses** a block whose dual has collapsed at
        //   the apex or on the boundary, which is the conic analogue of weak
        //   activity, so no built conic sensitivity carries one. The refusal is
        //   the report.
        let inf_norm = |v: &[f64]| v.iter().fold(0.0_f64, |mx, x| mx.max(x.abs()));
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&sol.x, &mut gx);
        let primal_scale = inf_norm(&prob.h).max(inf_norm(&gx)).max(1.0);
        let dual_scale = inf_norm(&sol.y)
            .max(inf_norm(&sol.z))
            .max(inf_norm(&sol.z_lb))
            .max(inf_norm(&sol.z_ub))
            .max(1.0);
        let dual_zero = WEAK_ACTIVE_REL * dual_scale;
        let primal_zero = WEAK_ACTIVE_REL * primal_scale;
        let weakly_active_ineq: Vec<usize> = orthant_rows
            .iter()
            .copied()
            .filter(|&i| (prob.h[i] - gx[i]).abs() <= primal_zero && sol.z[i] <= dual_zero)
            .collect();
        let x_scale = inf_norm(&sol.x).max(1.0);
        let bound_zero = WEAK_ACTIVE_REL * x_scale;
        let weakly_active_bound_vars: Vec<usize> = (0..n)
            .filter(|&j| {
                let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
                let lb_weak = lb > -BOUND_INF
                    && (sol.x[j] - lb).abs() <= bound_zero
                    && sol.z_lb[j] <= dual_zero;
                let ub_weak = ub < BOUND_INF
                    && (ub - sol.x[j]).abs() <= bound_zero
                    && sol.z_ub[j] <= dual_zero;
                lb_weak || ub_weak
            })
            .collect();
        Self::finish(
            prob,
            sol,
            opts.reg,
            active_rows,
            active_ineq,
            active_bounds,
            weakly_active_ineq,
            weakly_active_bound_vars,
            cone_kinds,
            curvature,
            opts.crossover,
            make_backend,
        )
    }

    /// Assemble, factor, and package — shared by both entry points so they
    /// cannot diverge on anything but which rows are active.
    #[allow(clippy::too_many_arguments)]
    fn finish<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        reg: f64,
        active_rows: Vec<Vec<(usize, f64)>>,
        active_ineq: Vec<usize>,
        active_bounds: Vec<(usize, bool)>,
        weakly_active_ineq: Vec<usize>,
        weakly_active_bound_vars: Vec<usize>,
        cone_kinds: Vec<(usize, ConeBlockKind)>,
        curvature: Vec<(usize, usize, f64)>,
        crossover: bool,
        mut make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        let n = prob.n;
        let m_eq = prob.m_eq();
        let n_active = active_rows.len() + active_bounds.len();
        let dim = n + m_eq + n_active;
        let active_bound_vars: Vec<usize> = active_bounds.iter().map(|&(j, _)| j).collect();

        // An apex pins its whole block, which can leave the equality block
        // unable to absorb `db` at all. Gated on an apex being present because
        // that is the only face that pins unconditionally, and because the
        // rank costs something; see `apex_can_absorb_db`.
        if let Some(&(block, _)) = cone_kinds.iter().find(|&&(_, k)| k == ConeBlockKind::Apex)
            && !apex_can_absorb_db(n, &group_rows_by_index(&prob.a, m_eq), &active_rows)
        {
            return Err(SensError::ActiveSetOverdetermined {
                block,
                what: "the block is pinned at the cone apex, which leaves the equality \
                       rows unable to absorb an arbitrary perturbation: the active set \
                       has no room left for dx, so the step would be a least-squares \
                       compromise rather than a derivative. The derivative may still \
                       exist — a solve that keeps the block off its tip will find it",
            });
        }

        let pat = assemble_kkt(prob, &active_rows, &active_bounds, &curvature, reg);
        // `KktPattern::dim` is the assembler's own count of the same quantity
        // this function derives independently. It existed as a field nothing
        // read — one dead-code warning, and the reviewer of #889 asked the
        // right question about it: a dimension check IS the natural reason for
        // it to be there, so here it is. A mismatch means the assembler and the
        // caller disagree about the KKT's shape, which would corrupt every
        // index into it.
        // `assert!`, not `debug_assert!`: the stated consequence of a mismatch
        // is that every index into the KKT is corrupt, and a debug-only check
        // does not protect the builds where that would happen — every wheel and
        // every CLI binary is a release build. One `usize` comparison once per
        // build is not a cost worth trading for it. (Raised in review of #889.)
        assert_eq!(
            pat.dim, dim,
            "assemble_kkt and finish must agree on the KKT dimension"
        );
        let release_slots = release_slots(&pat, n + m_eq, active_rows.len(), &active_bounds);
        let KktPattern {
            airn: kkt_airn,
            ajcn: kkt_ajcn,
            vals_true: kkt_vals_true,
            vals_reg: values_reg,
            ..
        } = pat;
        // Kept for the release path, which starts from the unreleased values.
        let values_reg_kept = values_reg.clone();

        // 1-norm of the factored (regularized) KKT, for the condition estimate.
        let kkt_norm1 = one_norm_sym(dim, &kkt_airn, &kkt_ajcn, &values_reg);

        let mut fact = Factorization::new(
            dim as Index,
            kkt_airn.clone(),
            kkt_ajcn.clone(),
            values_reg,
            make_backend(),
        )
        .map_err(|_| SensError::FactorizationFailed)?;

        // Hager estimate of κ₁ = ‖K‖₁·‖K⁻¹‖₁ (gh #284). Reuses the factor, so
        // it costs only a handful of back-solves.
        let inv_norm1 = estimate_inv_norm1(&mut fact, dim);
        let kkt_cond_estimate = kkt_norm1 * inv_norm1;

        // Oriented base multipliers, in the row order `assemble_kkt` used.
        let bound_base_mult: Vec<f64> = active_bounds
            .iter()
            .map(|&(j, is_lower)| if is_lower { sol.z_lb[j] } else { sol.z_ub[j] })
            .collect();
        let lp_without_crossover = prob.p_lower.is_empty() && !crossover;

        Ok(QpSensitivity {
            n,
            m_eq,
            base_x: sol.x.clone(),
            base_z: sol.z.clone(),
            base_z_lb: sol.z_lb.clone(),
            base_z_ub: sol.z_ub.clone(),
            active_bounds,
            bound_base_mult,
            lp_without_crossover,
            dim,
            prob: prob.clone(),
            active_ineq,
            active_rows,
            active_bound_vars,
            weakly_active_ineq,
            weakly_active_bound_vars,
            kkt_airn,
            kkt_ajcn,
            kkt_vals_true,
            kkt_cond_estimate,
            last_residual: None,
            ir_scratch: IrScratch::default(),
            kkt_vals_reg: Rc::new(values_reg_kept),
            release_slots: Rc::new(release_slots),
            release_backend: Rc::new(RefCell::new(Some(make_backend()))),
            cone_kinds,
            curvature,
            fact: Rc::new(RefCell::new(fact)),
        })
    }

    /// [`build`](Self::build) with the QP's default options and an active-set
    /// tolerance of `1e-7`.
    pub fn build_default<F>(
        prob: &QpProblem,
        sol: &QpSolution,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        Self::build(prob, sol, &QpOptions::default(), 1e-7, make_backend)
    }

    /// [`build`](Self::build) for a problem that carries a cone partition —
    /// the entry point for a solution produced by
    /// [`solve_socp_ipm`](crate::solve_socp_ipm).
    ///
    /// `cones` is the same slice handed to the solve; its blocks stack in
    /// order to cover the `m_ineq` inequality rows.
    ///
    /// An all-[`ConeSpec::Nonneg`] partition *is* the orthant problem, so it
    /// delegates to [`build`](Self::build) and behaves identically. Every other
    /// block goes through [`cone_block_face`], which classifies it as
    /// [`ConeBlockKind`] and returns the rows and curvature its face carries.
    /// Every family in `ConeSpec` is covered, and the `match` there is
    /// exhaustive, so a family added later is a compile error rather than a
    /// wrong answer.
    ///
    /// What is refused is not a *family* but a *point*. A cone's active object
    /// is the tangent/normal decomposition of the face its slack sits on;
    /// where that decomposition does not exist — a kink, a collapsed dual, a
    /// rank that strict complementarity does not pin down — the answer is
    /// [`SensError::NonsmoothConePoint`] rather than a linearization against
    /// noise. Refusing names the gap; answering hides it.
    pub fn build_conic<F>(
        prob: &QpProblem,
        cones: &[ConeSpec],
        sol: &QpSolution,
        opts: &QpOptions,
        active_tol: f64,
        make_backend: F,
    ) -> Result<Self, SensError>
    where
        F: FnMut() -> Box<dyn SparseSymLinearSolverInterface>,
    {
        // The partition must cover the inequality block exactly, or the caller
        // and the builder disagree about which rows are which. A short
        // partition is the dangerous direction: `[Nonneg(2)]` on a problem
        // whose rows 2.. are really a cone would pass the family check below
        // and hand a conic solution to the orthant path. `build`'s row-wise
        // guard would still catch that, but a caller error deserves to be
        // reported as one rather than as a complementarity failure.
        let covered: usize = cones.iter().map(ConeSpec::dim).sum();
        if covered != prob.m_ineq() {
            return Err(SensError::ConePartitionMismatch {
                covered,
                m_ineq: prob.m_ineq(),
            });
        }
        // The **same** admissible statuses as [`build`](Self::build), which
        // takes `OptimalInaccurate` per gh#880. This check read
        // `!= QpStatus::Optimal` while `build` admitted both, so an all-`Nonneg`
        // partition carrying an `OptimalInaccurate` solution was served by one
        // entry point and refused by the other — with the comment below
        // asserting that could not happen. Round 5 of #889.
        //
        // gh#880's reasoning applies unchanged here: `OptimalInaccurate` now
        // also carries the `σ` cascade's "could not certify to `tol`" verdict,
        // and that population previously arrived as a clean `Optimal` and built
        // a sensitivity object. The derivative at a less accurate point is less
        // accurate, which is what it was before; narrowing what the library
        // will do for it is a separate change, and one the repo has already
        // ruled on in the other direction.
        if !matches!(sol.status, QpStatus::Optimal | QpStatus::OptimalInaccurate) {
            return Err(SensError::NotOptimal);
        }
        // All-orthant IS the orthant problem; take the plain path so the two
        // entry points cannot answer differently on the same input — which is
        // only true because the status check above matches `build`'s.
        if cones.iter().all(|c| matches!(c, ConeSpec::Nonneg(_))) {
            return Self::build(prob, sol, opts, active_tol, make_backend);
        }

        let m = prob.m_ineq();
        let g_rows = group_rows_by_index(&prob.g, m);
        let mut gx = vec![0.0; m];
        prob.g_mul(&sol.x, &mut gx);
        let slack: Vec<f64> = (0..m).map(|i| prob.h[i] - gx[i]).collect();
        // The same two scales `build` derives for the orthant guard, so both
        // paths agree about what "zero" means on the same solution.
        let inf_norm = |v: &[f64]| v.iter().fold(0.0_f64, |mx, x| mx.max(x.abs()));
        let dual_scale = inf_norm(&sol.y)
            .max(inf_norm(&sol.z))
            .max(inf_norm(&sol.z_lb))
            .max(inf_norm(&sol.z_ub))
            .max(1.0);
        let primal_scale = inf_norm(&prob.h).max(inf_norm(&gx)).max(1.0);

        let mut active_rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut active_ineq: Vec<usize> = Vec::new();
        let mut kinds: Vec<(usize, ConeBlockKind)> = Vec::new();
        let mut curvature: Vec<(usize, usize, f64)> = Vec::new();
        // Which rows are ordinary orthant rows, so the weak-activity screen can
        // still run over them on a mixed partition — see `build_from_rows`.
        let mut orthant_rows: Vec<usize> = Vec::new();
        let mut offset = 0usize;
        for (block, spec) in cones.iter().enumerate() {
            let dim = spec.dim();
            let rows = &g_rows[offset..offset + dim];
            match spec {
                ConeSpec::Nonneg(_) => {
                    // Orthant rows inside a mixed partition: the same rule the
                    // plain path uses, applied to this block's slice.
                    for r in 0..dim {
                        orthant_rows.push(offset + r);
                        if sol.z[offset + r] > active_tol {
                            active_rows.push(rows[r].clone());
                            active_ineq.push(offset + r);
                        }
                    }
                }
                _ => {
                    let (kind, contributed, curved) = cone_block_face(
                        block,
                        spec,
                        &slack[offset..offset + dim],
                        &sol.z[offset..offset + dim],
                        rows,
                        primal_scale,
                        dual_scale,
                    )?;
                    for (r, row) in contributed.into_iter().enumerate() {
                        active_rows.push(row);
                        // Provenance only; a curved face's rows are
                        // combinations and have no one `G` row behind them.
                        active_ineq.push(offset + r.min(dim - 1));
                    }
                    curvature.extend(curved);
                    kinds.push((block, kind));
                }
            }
            offset += dim;
        }
        Self::build_from_rows(
            prob,
            sol,
            opts,
            active_tol,
            active_rows,
            active_ineq,
            kinds,
            curvature,
            orthant_rows,
            make_backend,
        )
    }

    /// First-order primal step `dx ≈ x*(b + Δb) − x*(b)` for a perturbation
    /// of the **equality right-hand side** `b`, the direct QP analog of
    /// sIPOPT's "pin a constraint, perturb its value". Constraint
    /// `pin_constraint_indices[k]` (an index into `b`) is perturbed by
    /// `deltas[k]`; all others are held fixed.
    ///
    /// Returns the length-`n` primal sensitivity, so `x* + dx` predicts the
    /// solution of the perturbed QP (exact to first order while the active
    /// set is unchanged). The factorization is reused, so repeated calls
    /// (e.g. a continuation sweep) cost one back-substitution each.
    ///
    /// # Panics
    ///
    /// Panics if `pin_constraint_indices` and `deltas` differ in length, or
    /// if any pin index is `≥ m_eq`.
    pub fn parametric_step(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
    ) -> Vec<f64> {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut db = vec![0.0; self.m_eq];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(
                i < self.m_eq,
                "pin constraint index {i} out of range (m_eq = {})",
                self.m_eq
            );
            db[i] += d;
        }
        self.step_from_db(&db)
    }

    /// Primal sensitivity for a full equality-RHS perturbation `db` (length
    /// `m_eq`): solves the active-set KKT with right-hand side `[0; db; 0]`
    /// and returns `dx = step[0..n]`.
    ///
    /// The back-solve is refined against the **unregularized** KKT
    /// ([`solve_refined`]) so the `O(δ)` regularization bias is stripped
    /// wherever the information survives in double precision; the achieved
    /// relative residual is recorded for
    /// [`last_step_residual`](Self::last_step_residual) (gh #284).
    ///
    /// # The step is meaningful only when `ill_conditioned()` is false
    ///
    /// This returns a bare `Vec` and cannot signal failure. Check
    /// [`ill_conditioned`](Self::ill_conditioned) after the call, or the step
    /// may be a least-squares compromise rather than a derivative — the
    /// active set can be rank-deficient in ways the build cannot always rule
    /// out cheaply (see `apex_can_absorb_db` for the one it does).
    ///
    /// Note **which** clause of that flag does the work here: on a
    /// rank-deficient active set the *regularized* matrix is perfectly well
    /// conditioned — `kkt_cond_estimate` reads `~2e10`, far under its
    /// threshold — and it is the step's **residual** that fires. That is
    /// exactly the trap gh#328 named, and review of #889 confirmed the
    /// separation independently over 33 conic probes: residual `0.5` on every
    /// wrong step, `~1e-13` on every correct one, no overlap.
    pub fn step_from_db(&mut self, db: &[f64]) -> Vec<f64> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        // A singular factor would have been caught at build; a back-solve
        // failure here is not recoverable, so surface a zero step.
        let mut fact = self.fact.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.kkt_airn,
            &self.kkt_ajcn,
            &self.kkt_vals_true,
            &mut rhs,
            &mut self.ir_scratch,
        ) {
            Ok(res) => self.last_residual = Some(res),
            Err(()) => return vec![0.0; self.n],
        }
        rhs.truncate(self.n);
        rhs
    }

    /// The full compound KKT step for an equality-RHS perturbation `db`,
    /// length [`kkt_dim`](Self::kkt_dim) rather than truncated to `n`.
    ///
    /// [`step_from_db`](Self::step_from_db) returns the primal block; the
    /// bound refinement needs the whole vector, because the multiplier rows are
    /// what tell it which bounds the step drives negative.
    fn full_step_from_db(&mut self, db: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        assert_eq!(db.len(), self.m_eq, "db must have length m_eq");
        let mut rhs = vec![0.0 as Number; self.dim];
        rhs[self.n..self.n + self.m_eq].copy_from_slice(db);
        let rhs_plain = rhs.clone();
        let mut fact = self.fact.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.kkt_airn,
            &self.kkt_ajcn,
            &self.kkt_vals_true,
            &mut rhs,
            &mut self.ir_scratch,
        ) {
            Ok(res) => {
                drop(fact);
                self.last_residual = Some(res);
                Some((rhs, rhs_plain))
            }
            Err(()) => None,
        }
    }

    /// The active-set KKT as a [`SensBacksolver`], sharing this object's
    /// factorization.
    ///
    /// This is the seam onto `pounce-sens-core`: anything there that is generic
    /// over the trait works on a convex QP through this handle.
    pub fn backsolver(&self) -> QpKktBacksolver {
        let abase = self.n + self.m_eq + self.active_rows.len();
        let bound_rows: Vec<BoundRow> = self
            .active_bounds
            .iter()
            .enumerate()
            .map(|(k, &(j, is_lower))| BoundRow {
                row: abase + k,
                var_row: j,
                lower: is_lower,
            })
            .collect();
        QpKktBacksolver {
            fact: Rc::clone(&self.fact),
            airn: Rc::new(self.kkt_airn.clone()),
            ajcn: Rc::new(self.kkt_ajcn.clone()),
            vals_true: Rc::new(self.kkt_vals_true.clone()),
            scratch: Rc::new(RefCell::new(IrScratch::default())),
            dim: self.dim,
            bound_rows: Rc::new(bound_rows),
            last_residual: Rc::new(Cell::new(f64::NAN)),
            vals_reg: Rc::clone(&self.kkt_vals_reg),
            slots: Rc::clone(&self.release_slots),
            base_mult: Rc::new(self.bound_base_mult.clone()),
            released: Rc::new(RefCell::new(None)),
            release_backend: Rc::clone(&self.release_backend),
        }
    }

    /// [`parametric_step`](Self::parametric_step), refined to respect the
    /// variable bounds — the convex arm's `fix_relax`.
    ///
    /// The plain step is a linear predictor and can point outside the box.
    /// Clipping the offending coordinate is cheap but leaves every other one at
    /// its unclipped value, so the result satisfies the bounds and no longer
    /// satisfies the constraints. This instead pins the crossing coordinate at
    /// its bound and re-solves, so the others move with it.
    ///
    /// Returns `(dx, pinned_variables, stop_reason)`. The computation is
    /// `pounce_sens_core::boundcheck::refine_step_onto_bounds` — the same code
    /// the NLP arm runs, reached through [`backsolver`](Self::backsolver)
    /// rather than reimplemented.
    ///
    /// In this phase the refinement **pins only**: a bound whose multiplier the
    /// step drives negative is not released (see
    /// [`QpKktBacksolver::supports_release`]), so a perturbation pulling a
    /// variable off a bound is still held there.
    pub fn parametric_step_bounded(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
        bound_eps: f64,
        max_iter: usize,
    ) -> Result<(Vec<f64>, Vec<usize>, RefineStop), SensError> {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut db = vec![0.0; self.m_eq];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(i < self.m_eq, "pin constraint index {i} out of range");
            db[i] += d;
        }
        let (dx_full, rhs_plain) = self
            .full_step_from_db(&db)
            .ok_or(SensError::FactorizationFailed)?;

        let n = self.n;
        let (_, x_curr, lo, hi, multipliers) = self.path_inputs(pin_constraint_indices, deltas);
        let bs = self.backsolver();
        let (dx, pinned, stop) = refine_step_onto_bounds(
            &bs,
            &dx_full,
            &x_curr,
            &lo,
            &hi,
            &multipliers,
            &rhs_plain,
            bound_eps,
            RELEASE_FLOOR,
            max_iter,
        )
        .map_err(SensError::Refinement)?;
        self.last_residual = Some(bs.last_residual());
        Ok((dx[..n].to_vec(), pinned, stop))
    }

    /// The perturbation applied **a little at a time**, stopping at each point
    /// where the active set changes.
    ///
    /// `parametric_step_bounded` repairs one linear predictor. This instead
    /// walks the perturbation, re-solving at every breakpoint where a variable
    /// reaches a bound or a bound's multiplier hits zero, so the answer follows
    /// the piecewise-affine solution path rather than extrapolating across its
    /// kinks. For a QP the path *is* piecewise affine, so within a segment the
    /// walk is exact.
    ///
    /// Returns `(dx, segments)`; each [`PathSegment`] records the fraction of
    /// the perturbation at which a bound was reached or released, and which
    /// variable it belonged to.
    ///
    /// Runs `pounce_sens_core::boundcheck::step_along_path` — the NLP arm's
    /// code, reached through [`backsolver`](Self::backsolver).
    pub fn parametric_step_path(
        &mut self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
        max_iter: usize,
    ) -> Result<(Vec<f64>, Vec<PathSegment>), SensError> {
        let (rhs_plain, x_curr, lo, hi, multipliers) =
            self.path_inputs(pin_constraint_indices, deltas);
        let bs = self.backsolver();
        let (dx, segments) = step_along_path(
            &bs,
            &rhs_plain,
            &x_curr,
            &lo,
            &hi,
            &multipliers,
            max_iter,
            &[],
            &[],
            &[],
            // How far outside its box the answer has to land before the
            // walk treats a base-active bound as one the factorization
            // never enforced. The same floor the release decision uses;
            // this arm has no bound relaxation to widen it.
            RELEASE_FLOOR,
        )
        .map_err(SensError::Refinement)?;
        self.last_residual = Some(bs.last_residual());
        Ok((dx[..self.n].to_vec(), segments))
    }

    /// The shared inputs the path and bounded modes both need: the compound
    /// right-hand side, the base point, its bounds, and the active bounds'
    /// multiplier rows.
    fn path_inputs(
        &self,
        pin_constraint_indices: &[usize],
        deltas: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<BoundMultiplier>) {
        assert_eq!(
            pin_constraint_indices.len(),
            deltas.len(),
            "pin_constraint_indices and deltas must have equal length"
        );
        let mut rhs_plain = vec![0.0 as Number; self.dim];
        for (&i, &d) in pin_constraint_indices.iter().zip(deltas) {
            assert!(i < self.m_eq, "pin constraint index {i} out of range");
            rhs_plain[self.n + i] += d;
        }
        let n = self.n;
        let lo: Vec<f64> = (0..n).map(|j| self.prob.lb_of(j)).collect();
        let hi: Vec<f64> = (0..n).map(|j| self.prob.ub_of(j)).collect();
        let abase = n + self.m_eq + self.active_rows.len();
        let multipliers: Vec<BoundMultiplier> = self
            .bound_base_mult
            .iter()
            .enumerate()
            .map(|(k, &base)| BoundMultiplier {
                row: abase + k,
                base,
            })
            .collect();
        (rhs_plain, self.base_x.clone(), lo, hi, multipliers)
    }

    /// The achieved complementarity `⟨s, z⟩ / degree` at the solution.
    ///
    /// The μ-scaled activity classification the NLP arm uses needs a barrier
    /// parameter, and `QpSolution` does not carry one. It does not need to:
    /// everything required is derivable from `(prob, sol)`, which is why this
    /// is a method rather than a new field — the type is all-public, not
    /// `#[non_exhaustive]`, and constructed by literal in dozens of places.
    ///
    /// Caveat worth knowing before comparing arms: this is the *achieved*
    /// complementarity at the returned point, not the barrier parameter the
    /// last interior-point iteration actually ran at. They agree to within the
    /// centering ratio at convergence, and the classification bands are decades
    /// wide, so it is fine for classifying — but a cross-arm test must not
    /// demand bit-agreement with the NLP's `barrier_mu()`.
    pub fn duality_measure(&self) -> f64 {
        let prob = &self.prob;
        let mut gx = vec![0.0; prob.m_ineq()];
        prob.g_mul(&self.base_x, &mut gx);
        let mut total = 0.0;
        let mut degree = 0usize;
        for (i, &gx_i) in gx.iter().enumerate() {
            total += (prob.h[i] - gx_i) * self.base_z[i];
            degree += 1;
        }
        for j in 0..prob.n {
            let (lb, ub) = (prob.lb_of(j), prob.ub_of(j));
            if lb > -BOUND_INF {
                total += (self.base_x[j] - lb) * self.base_z_lb[j];
                degree += 1;
            }
            if ub < BOUND_INF {
                total += (ub - self.base_x[j]) * self.base_z_ub[j];
                degree += 1;
            }
        }
        if degree == 0 {
            0.0
        } else {
            total / degree as f64
        }
    }

    /// What each bound is doing at this optimum: holding its coordinate,
    /// inactive, or vanishing together with its multiplier at a kink.
    ///
    /// The decision is `pounce_sens_core::activity_kernel`, the same rule the
    /// NLP arm applies, so the two arms cannot drift on what a kink is. What
    /// differs is the derivation of the inputs, and that is
    /// [`crate::activity`]'s subject.
    ///
    /// # Read the caveat before reading a status
    ///
    /// [`AMBIGUOUS`](pounce_sens_core::activity_kernel::AMBIGUOUS) does **not**
    /// mean "probably not a kink". A genuine kink lands there whenever its
    /// coordinate is coupled, because the cheap curvature is a diagonal (for a
    /// variable) or a directional one (for a row) while the multiplier is
    /// generated by the *reduced* curvature. The ratio is then
    /// `reduced/diagonal`, which is μ-independent — so re-solving tighter does
    /// not separate it. Treating the class as a proxy for kink-ness is the
    /// error that shipped gh#763.
    ///
    /// # On a conic build, only the orthant rows and the bounds mean anything
    ///
    /// The classifier is the **orthant** rule: it reads each row of `G` as an
    /// inequality with its own slack and multiplier. A cone block does not work
    /// that way — it complements as a *block* inner product `⟨s, z⟩ = 0`, and
    /// row-wise `sᵢ·zᵢ` is generally nonzero with `z` generally carrying
    /// negative entries. So a cone row's entry here is not a wrong reading of a
    /// real quantity; it is a reading of a quantity that does not exist.
    ///
    /// Measured on this crate's `boundary` fixture, where `z = (6.38, −4.86,
    /// −4.14)`: the two negative tail rows come back classified **inactive**,
    /// which is exactly the plausible-and-wrong answer — the block is on its
    /// boundary and binding. Round 5 of #889.
    ///
    /// This is not refused, because the report is still correct for every
    /// orthant row and every variable bound of a mixed partition, and those are
    /// the entries a caller on a mixed model actually asks about. Use
    /// [`cone_block_kinds`](Self::cone_block_kinds) for what a cone block is
    /// doing; there is no per-row answer for it to give.
    pub fn activity(&self) -> ConvexActivityReport {
        let floor = curvature_floor(&self.prob);
        let sol = QpSolution {
            status: QpStatus::Optimal,
            x: self.base_x.clone(),
            y: vec![],
            z: self.base_z.clone(),
            z_lb: self.base_z_lb.clone(),
            z_ub: self.base_z_ub.clone(),
            obj: 0.0,
            iters: 0,
            iterates: vec![],
        };
        classify_all(&self.prob, &sol, self.duality_measure(), floor)
    }

    /// What each second-order block was doing at the solution: interior, at its
    /// apex, or on its boundary.
    ///
    /// Empty on the orthant path. Worth reading before trusting a conic step,
    /// because the three regimes do not carry the same guarantee: an apex block
    /// contributes a **flat** face and the predictor is exact there, exactly as
    /// an orthant row's is, while a boundary block's single row is a
    /// linearization of a **curved** face — first-order, like an active
    /// nonlinear constraint on the NLP arm.
    pub fn cone_block_kinds(&self) -> &[(usize, ConeBlockKind)] {
        &self.cone_kinds
    }

    /// `true` when this is a pure LP (`P = 0`) whose solve did not run
    /// crossover, so a **degenerate** optimal vertex is a real hazard here.
    ///
    /// At a degenerate LP vertex more constraints are active than there are
    /// variables, the active-set KKT is rank-deficient, and `dx/db` is not
    /// single-valued — measured on a two-variable example, the step comes back
    /// summing to half the perturbation it should. That case *is* caught, by
    /// [`ill_conditioned`](Self::ill_conditioned); this flag names the cause and
    /// the remedy, which is to solve with `qp_crossover=yes` so the interior
    /// point is pivoted to an exact vertex basis first.
    ///
    /// Reading `opts.crossover` means the `opts` handed to
    /// [`build`](Self::build) must be **the options the solve actually ran
    /// with**. Passing a fresh `QpOptions::default()` to `build` after solving
    /// with crossover on would report `true` here and be wrong.
    pub fn lp_without_crossover(&self) -> bool {
        self.lp_without_crossover
    }

    /// Hager 1-norm estimate of the condition number `κ₁` of the (factored,
    /// regularized) active-set KKT.
    ///
    /// A large value warns that the sensitivity system is near-singular — the
    /// active-constraint gradients are nearly rank-deficient (near-LICQ) — so
    /// the parametric step may be untrustworthy even though the solve reports
    /// success (gh #284). This is the quantitative companion to the boolean
    /// [`ill_conditioned`](Self::ill_conditioned); see also the per-step
    /// [`last_step_residual`](Self::last_step_residual). Well-conditioned
    /// sensitivities report a modest `κ₁` (a few `×10⁹` on the badly-scaled
    /// gh #284 QPs); a numerically singular one saturates near `1e16`.
    pub fn kkt_cond_estimate(&self) -> f64 {
        self.kkt_cond_estimate
    }

    /// Whether the KKT/sensitivity system is ill-conditioned enough that
    /// [`parametric_step`](Self::parametric_step) may be unreliable even after
    /// refinement.
    ///
    /// `true` when **either**
    ///
    /// * the build-time [`kkt_cond_estimate`](Self::kkt_cond_estimate) exceeds
    ///   [`KKT_ILL_CONDITIONED_THRESHOLD`] (gh #284) — catches a numerically
    ///   singular KKT before any step is taken; **or**
    /// * the most recent [`parametric_step`](Self::parametric_step) refined to a
    ///   relative KKT residual above [`STEP_UNRELIABLE_RESIDUAL`] (gh #328) —
    ///   catches the near-LICQ case the saturating condition estimate misses
    ///   (well-scaled `P`, near-parallel active rows), where the returned
    ///   `dx/db` does not actually satisfy the true sensitivity system.
    ///
    /// The second clause is what makes the diagnostic honest across the whole
    /// near-LICQ family: on a well-scaled `P` the condition estimate saturates
    /// below its threshold (see [`KKT_ILL_CONDITIONED_THRESHOLD`]), so before
    /// gh #328 an over-damped, ~3300×-wrong step reported `ill_conditioned =
    /// false`. Now the stalled refinement residual fires the flag instead of
    /// letting a silently-damped value pass. On the well-conditioned
    /// equality-only and active-set cases both clauses stay quiet.
    pub fn ill_conditioned(&self) -> bool {
        self.kkt_cond_estimate > KKT_ILL_CONDITIONED_THRESHOLD
            || self
                .last_residual
                .is_some_and(|r| r > STEP_UNRELIABLE_RESIDUAL)
    }

    /// Relative KKT residual `‖rhs − K·step‖∞ / ‖rhs‖∞` achieved by the most
    /// recent [`parametric_step`](Self::parametric_step) /
    /// [`step_from_db`](Self::step_from_db), or `None` before any step.
    ///
    /// Measured against the **unregularized** KKT, so it reflects how well the
    /// returned step actually satisfies the true sensitivity system. A tiny
    /// value (round-off level) means the step is trustworthy; a large one
    /// means refinement could not solve the near-singular system and the step
    /// is unreliable (gh #284). Because it is a true *relative* residual it is
    /// invariant to the magnitude of the perturbation, so it exposes a stalled
    /// solve even for a small `db` — the case the earlier `1 + ‖rhs‖` floor
    /// masked (gh #328). A value above [`STEP_UNRELIABLE_RESIDUAL`] fires
    /// [`ill_conditioned`](Self::ill_conditioned).
    pub fn last_step_residual(&self) -> Option<f64> {
        self.last_residual
    }

    /// The active-set KKT dimension `n + m_eq + n_active`.
    pub fn kkt_dim(&self) -> usize {
        self.dim
    }

    /// Provenance for the active inequality rows — **not** always usable as an
    /// index into `G`.
    ///
    /// For an orthant row it *is* the row of `G`, which is what this accessor
    /// originally documented. A **curved cone face** contributes rows that are
    /// linear combinations of the block's rows — one `wᵀG` for a second-order
    /// or non-symmetric boundary, `q(q+1)/2` of them for a PSD face — and no
    /// single `G` row stands behind any of them. Those entries carry the
    /// block's first row as provenance instead.
    ///
    /// So `prob.g[sens.active_ineq()[k]]` is a real, plausible, **wrong** row
    /// whenever a cone block is active: the gh#450 shape, in this arm's own
    /// index space. Use [`cone_block_kinds`](Self::cone_block_kinds) to learn
    /// which blocks contributed faces, and read this as "which block a row came
    /// from" rather than "which `G` row it is". Raised in review of #889.
    pub fn active_ineq(&self) -> &[usize] {
        &self.active_ineq
    }

    /// Variables whose bound is in the active set.
    pub fn active_bound_vars(&self) -> &[usize] {
        &self.active_bound_vars
    }

    /// Inequality rows at which **strict complementarity fails**: binding in
    /// the primal while carrying a negligible multiplier (gh #219).
    ///
    /// This is the precondition check for
    /// [`parametric_step`](Self::parametric_step). That predictor is exact only
    /// while the active set is unchanged; at a weakly active constraint the
    /// perturbation changes it, so `dx/db` is a genuine one-sided derivative
    /// and the opposite direction has a different — equally correct — value.
    /// On gh #219's QP the two branches differ by 33%, and which one is
    /// reported turns on the solver's `tol`.
    ///
    /// A non-empty result does not invalidate anything already returned: both
    /// branches are real derivatives. It means the caller should not assume the
    /// predictor extrapolates in both directions, and should probe the
    /// direction it actually cares about. The screen is deliberately
    /// conservative (see `WEAK_ACTIVE_REL`) — a near-degenerate constraint is
    /// flagged too, which is the useful behaviour for a diagnostic.
    ///
    /// # On a conic build this screens the orthant rows only
    ///
    /// A `Nonneg` block inside a mixed partition is screened exactly as the
    /// plain path screens it. A **cone** block never appears here, and that is
    /// by construction rather than omission: `cone_block_face` *refuses* a
    /// block whose dual has collapsed at the apex or on the boundary, which is
    /// the conic analogue of a weakly active row — so no built conic
    /// sensitivity carries one, and the refusal is the report.
    ///
    /// Both screens returned empty on every conic build until round 5 of #889,
    /// which is a different thing from "there was nothing to report".
    pub fn weakly_active_ineq(&self) -> &[usize] {
        &self.weakly_active_ineq
    }

    /// Variables whose bound is weakly active — the bound analog of
    /// [`weakly_active_ineq`](Self::weakly_active_ineq).
    pub fn weakly_active_bound_vars(&self) -> &[usize] {
        &self.weakly_active_bound_vars
    }

    /// Reduced Hessian of the QP at the optimum: the **Hessian of the
    /// Lagrangian** projected onto the null space of the active constraints
    /// `B = [A; the active inequality-side rows; active bound rows]`. If `Z` is
    /// an orthonormal basis of `null(B)` (the feasible directions / degrees of
    /// freedom), the reduced Hessian is `H_R = Zᵀ (P + C) Z`, where `C` is the
    /// active faces' curvature. Its eigenvalues are the curvatures along
    /// feasible directions: all positive ⟺ a strict second-order minimizer, and
    /// their spread is the conditioning of the QP on the active manifold.
    ///
    /// **On the orthant path `C` is empty and this is exactly `Zᵀ P Z`** — the
    /// classical definition, unchanged, because a linear constraint has no
    /// Hessian and the Lagrangian's is the objective's. `C` is nonzero only on a
    /// conic build with a *curved* face, where projecting bare `P` is the same
    /// omission that made the first draft of the parametric step converge to the
    /// wrong derivative (round 5 of #889 found this method still doing it; this
    /// doc described the buggy behaviour before then, and neither behaviour
    /// after the fix until round 6 caught it).
    ///
    /// A second correction in the same breath: `B`'s inequality side is the
    /// **active rows**, which on a conic build are a face's combined row `wᵀG`
    /// rather than rows of `G`. "Active G rows" was never true there.
    ///
    /// This mirrors the NLP `Solver.reduced_hessian` /
    /// `solve_with_sens(compute_reduced_hessian=True)`.
    ///
    /// The basis `Z` is the null space of `B`, obtained from the
    /// eigenvectors of `BᵀB` whose eigenvalue is below `rank_tol · λ_max`
    /// (squared singular values; the count above the threshold is
    /// `rank(B)`, so the degrees of freedom are `n − rank(B)`). The
    /// computation densifies `B` and `P`, so it is `O(n³)` — intended, like
    /// sIPOPT's reduced Hessian, for QPs with a modest number of variables
    /// (the parametric step stays sparse and is the workhorse for large
    /// problems).
    ///
    /// # Errors
    ///
    /// Returns [`SensError::EigenFailed`] if either symmetric eigensolve (the
    /// one that extracts `Z` from `BᵀB`, or the final one on `H_R`) does not
    /// converge — its rank / null-space, and hence the result, cannot be
    /// trusted, so a wrong answer is never returned silently.
    pub fn reduced_hessian(&self, rank_tol: f64) -> Result<ReducedHessian, SensError> {
        let n = self.n;

        // Active Jacobian B (m_act × n), dense row-major: equality rows,
        // then active inequality rows, then active variable-bound rows.
        let m_act = self.m_eq + self.active_rows.len() + self.active_bound_vars.len();
        let mut b = vec![0.0; m_act * n];
        for t in &self.prob.a {
            b[t.row * n + t.col] += t.val;
        }
        // `active_rows`, NOT `active_ineq`. On the orthant path they say the
        // same thing; on a conic build `active_ineq` is **provenance** — the
        // `G` rows a block contributed — while the active object is the face's
        // own row, a combination of them. Indexing raw `G` with a provenance
        // index returns a real, plausible, wrong row: gh#450's shape, and the
        // `active_rows` field doc has claimed this loop reads it since the row
        // was introduced. It did not until round 5 of #889 measured it.
        let mut row = self.m_eq;
        for r in &self.active_rows {
            for &(col, val) in r {
                b[row * n + col] += val;
            }
            row += 1;
        }
        for &j in &self.active_bound_vars {
            b[row * n + j] += 1.0;
            row += 1;
        }

        // Null space of B from the eigenvectors of BᵀB (symmetric, n×n,
        // column-major for `symmetric_eigen`). BᵀB[a,c] = Σ_r B[r,a]·B[r,c].
        let mut btb = vec![0.0; n * n];
        for r in 0..m_act {
            for a in 0..n {
                let bra = b[r * n + a];
                if bra == 0.0 {
                    continue;
                }
                for c in 0..n {
                    btb[a * n + c] += bra * b[r * n + c];
                }
            }
        }
        let mut sv = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        // Ascending eigenvalues. A failed eigensolve makes the rank/null-space
        // count below meaningless, so refuse rather than return garbage.
        if !symmetric_eigen(&btb, n, &mut sv, &mut vecs) {
            return Err(SensError::EigenFailed);
        }

        // rank(B) = # squared-singular-values above the relative threshold;
        // the null space is spanned by the eigenvectors of the rest (the
        // smallest, ≈ 0). With ascending order those are the first columns.
        let max_sv = sv.last().copied().unwrap_or(0.0).max(0.0);
        let thresh = rank_tol * max_sv;
        let rank = sv.iter().filter(|&&l| l > thresh).count();
        let n_dof = n - rank;

        // Dense symmetric Hessian (n×n) from `P`'s lower triangle, **plus the
        // active faces' curvature** — i.e. the Hessian of the Lagrangian, not
        // of the objective alone.
        //
        // On the orthant path `curvature` is empty and this is exactly `P`,
        // which is what the classical definition wants: a linear constraint has
        // no Hessian, so the Lagrangian's is the objective's. A conic boundary
        // face is *curved*, and there the second-order behaviour along the
        // active manifold is not `P`'s. Projecting bare `P` there is the same
        // omission that made the first draft of the parametric step converge to
        // the wrong derivative, one method over. Round 5 of #889.
        let mut p = vec![0.0; n * n];
        for t in &self.prob.p_lower {
            p[t.row * n + t.col] += t.val;
            if t.row != t.col {
                p[t.col * n + t.row] += t.val;
            }
        }
        for &(r, c, v) in &self.curvature {
            p[r * n + c] += v;
            if r != c {
                p[c * n + r] += v;
            }
        }

        // H_R = Zᵀ (P + curvature) Z, with Z = first `n_dof` columns of `vecs` (the null
        // space). Column-major throughout: column j of Z is vecs[j*n + ·].
        let z = |j: usize, r: usize| vecs[j * n + r];
        // PZ (n × n_dof), column-major.
        let mut pz = vec![0.0; n * n_dof];
        for j in 0..n_dof {
            for (r, pzr) in pz[j * n..(j + 1) * n].iter_mut().enumerate() {
                let mut acc = 0.0;
                for c in 0..n {
                    acc += p[r * n + c] * z(j, c);
                }
                *pzr = acc;
            }
        }
        // H_R (n_dof × n_dof), column-major: H_R[i,j] = z_iᵀ (P z_j).
        let mut hr = vec![0.0; n_dof * n_dof];
        for j in 0..n_dof {
            for i in 0..n_dof {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += z(i, r) * pz[j * n + r];
                }
                hr[j * n_dof + i] = acc;
            }
        }

        // Eigendecompose the (small) reduced Hessian.
        let mut eigenvalues = vec![0.0; n_dof];
        let mut eigenvectors = vec![0.0; n_dof * n_dof];
        if !symmetric_eigen(&hr, n_dof, &mut eigenvalues, &mut eigenvectors) {
            return Err(SensError::EigenFailed);
        }

        Ok(ReducedHessian {
            n_dof,
            matrix: hr,
            eigenvalues,
            eigenvectors,
        })
    }

    /// [`reduced_hessian`](Self::reduced_hessian) with a relative rank
    /// tolerance of `1e-9`.
    pub fn reduced_hessian_default(&self) -> Result<ReducedHessian, SensError> {
        self.reduced_hessian(1e-9)
    }
}

/// The reduced Hessian `H_R = Zᵀ (P + C) Z` of a QP on its active manifold —
/// `C` being the active faces' curvature, empty on the orthant path — with
/// its eigendecomposition. All matrices are column-major and `n_dof × n_dof`
/// (`n_dof` = degrees of freedom = `n − rank` of the active Jacobian).
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedHessian {
    /// Degrees of freedom: the dimension of every field here.
    pub n_dof: usize,
    /// The reduced Hessian `H_R`, column-major `n_dof × n_dof` (symmetric).
    pub matrix: Vec<f64>,
    /// Eigenvalues of `H_R`, ascending (length `n_dof`).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors, column-major `n_dof × n_dof`; column `j` pairs with
    /// `eigenvalues[j]`. Signs are pinned (largest-magnitude component
    /// positive) so a column read as a direction reproduces. Note these
    /// live in the null-space basis `Z`, which is itself only fixed up
    /// to a rotation within any degenerate eigenspace of `BᵀB`.
    pub eigenvectors: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipm::solve_qp_ipm;
    use crate::qp::Triplet;
    use pounce_feral::FeralSolverInterface;

    fn backend() -> Box<dyn SparseSymLinearSolverInterface> {
        Box::new(FeralSolverInterface::new())
    }

    // -----------------------------------------------------------------------
    // `row_rank` / `apex_can_absorb_db` — the apex absorbability guard's
    // arithmetic, tested without a solver.
    //
    // These are here rather than in `convex_soc_sensitivity.rs` because the
    // case that separates the right rule from the wrong one cannot be reached
    // by an ordinary model: see
    // `an_active_bound_does_not_count_as_an_apex_pin` for the counting
    // argument (a discriminating integration fixture must be primal
    // degenerate). The integration fixture pins that the line is *reached*;
    // these pin what it computes.
    // -----------------------------------------------------------------------

    fn row(cols: &[(usize, f64)]) -> Vec<(usize, f64)> {
        cols.to_vec()
    }

    #[test]
    fn row_rank_counts_independent_rows() {
        assert_eq!(row_rank(&[], 4), 0, "no rows is rank zero");
        assert_eq!(row_rank(&[row(&[(0, 1.0)])], 4), 1);
        assert_eq!(
            row_rank(&[row(&[(0, 1.0)]), row(&[(2, 3.0)]), row(&[(3, -1.0)])], 4),
            3,
            "unit rows on distinct columns are independent"
        );
        assert_eq!(
            row_rank(
                &[row(&[(0, 1.0), (1, 1.0)]), row(&[(0, 1.0), (1, -1.0)])],
                4
            ),
            2,
            "partial pivoting must not lose a rank on a well-conditioned pair"
        );
    }

    #[test]
    fn row_rank_sees_through_a_dependency() {
        // The third row is the sum of the first two, so the rank is 2 — this
        // is the case the guard exists to detect, since a dependent active row
        // does not remove a further dimension from `ker(B)`.
        let rows = [
            row(&[(0, 1.0), (1, 1.0)]),
            row(&[(1, 1.0), (2, 1.0)]),
            row(&[(0, 1.0), (1, 2.0), (2, 1.0)]),
        ];
        assert_eq!(row_rank(&rows, 3), 2);
    }

    /// The tolerance's bias direction, as behaviour rather than prose.
    ///
    /// `row_rank` equilibrates each row and drops pivots under `√ε`, so a row
    /// that differs from a dependent one only at round-off is **not** counted.
    ///
    /// What that does to the *guard* is the part worth pinning, and it is not
    /// what this test used to say. Under `rank([A;B]) == rank(A) + rank(B)`,
    /// undercounting the stacked rank **refuses** and undercounting a part
    /// **passes**; the stacked elimination has the most rows, so the dominant
    /// misfire is refusing a model that could have been served. Both are
    /// asserted below.
    ///
    /// This test's own doc claimed to be "what would catch it flipping" while
    /// calling only `row_rank` and never `apex_can_absorb_db` — so it could
    /// not, and the direction then flipped once with it green (round 4's `≥` →
    /// `==` change, caught in round 5 of #889). The guard's verdict is in the
    /// test now.
    #[test]
    fn row_rank_drops_a_round_off_pivot() {
        let eps = 1e-14;
        let near = [row(&[(0, 1.0), (1, 1.0)]), row(&[(0, 1.0), (1, 1.0 + eps)])];
        assert_eq!(
            row_rank(&near, 2),
            1,
            "a difference at round-off must not buy a rank"
        );

        // …and a genuinely distinct second row still does, so the tolerance is
        // not simply swallowing everything.
        let real = [row(&[(0, 1.0), (1, 1.0)]), row(&[(0, 1.0), (1, 1.001)])];
        assert_eq!(row_rank(&real, 2), 2);

        // ---- and what that does to the guard's verdict ----
        //
        // (a) Stacked undercount -> REFUSE a servable model. `ker(B)` is
        //     `span(-eps, 1)`, so `A(ker B) = range(A)` for every eps > 0 and
        //     the truth is serve; at eps <= 1e-8 the stacked rank reads 1
        //     against a true 2 and the guard refuses. Acceptable, because the
        //     near-dependency is between A and B: the true dx/db there is
        //     O(1/eps) >= 1e8. Recorded as the *dominant* misfire, which is
        //     the opposite of what this file said for three revisions.
        let a = [row(&[(0, 1.0)])];
        let b_tight = [row(&[(0, 1.0), (1, 1e-12)])];
        assert!(
            !apex_can_absorb_db(2, &a, &b_tight),
            "a stacked undercount must refuse — that is the direction this errs in"
        );
        let b_loose = [row(&[(0, 1.0), (1, 1e-6)])];
        assert!(
            apex_can_absorb_db(2, &a, &b_loose),
            "…and well clear of the tolerance the same shape is served"
        );

        // (b) Part undercount -> PASS a deficient model. B's two rows are
        //     near-dependent, so rank(B) reads 1 against a true 2; with the
        //     true rank `ker(B) = {0}` and `A(ker B) = {0}`, so the truth is
        //     refuse. This is the direction `ill_conditioned()` covers.
        let a2 = [row(&[(1, 1.0)])];
        let b2 = [row(&[(0, 1.0)]), row(&[(0, 1.0), (1, 1e-12)])];
        assert_eq!(row_rank(&b2, 2), 1, "the two rows read as one");
        assert!(
            apex_can_absorb_db(2, &a2, &b2),
            "a part undercount passes a model whose true rank would refuse"
        );
    }

    #[test]
    fn apex_can_absorb_db_is_a_dimension_count() {
        let e0 = row(&[(0, 1.0)]);
        let e1 = row(&[(1, 1.0)]);
        let e2 = row(&[(2, 1.0)]);

        // No equalities: nothing to absorb, so nothing to refuse.
        assert!(apex_can_absorb_db(3, &[], &[e0.clone(), e1.clone()]));

        // The reviewer's minimal case, as rows: n = 3, two independent
        // equalities, and an apex pinning all three coordinates leaves
        // `ker(B) = {0}`.
        let apex_rows = [e0.clone(), e1.clone(), e2.clone()];
        assert!(!apex_can_absorb_db(
            3,
            &[e0.clone(), e1.clone()],
            &apex_rows
        ));

        // One free coordinate is enough for one equality — but only if the
        // equality can *reach* it. On `e₂`, the free one, it can:
        let two_pinned = [e0.clone(), e1.clone()];
        assert!(apex_can_absorb_db(3, &[e2.clone()], &two_pinned));

        // On `e₀`, which `B` pins, it cannot: `A(ker B) = {0}`. The dimension
        // count passed this (`3 − 2 = 1 ≥ 1`) and it is exactly the wrong
        // answer the fourth review of #889 measured; the exact criterion
        // refuses it. `the_criterion_is_exact_not_a_dimension_count` owns the
        // full argument.
        assert!(!apex_can_absorb_db(3, &[e0.clone()], &two_pinned));

        // Two equalities against one free coordinate: refused either way.
        assert!(!apex_can_absorb_db(
            3,
            &[e0.clone(), e1.clone()],
            &two_pinned
        ));
    }

    /// **`row_rank` equilibrates, so one badly scaled row cannot erase the
    /// others.**
    ///
    /// Raised in the fourth review of #889. The tolerance was `√ε` times a
    /// *global* maximum, which was fine while these rows were cone faces and
    /// active orthant rows — the cone geometry and `G` keep those within an
    /// order or two of each other. `A` joined them in the third review, and
    /// `A`'s row scaling is the user's: one equality carrying `1e10`
    /// coefficients sets `tol ≈ 0.15` for every pivot, and two genuinely
    /// independent `O(1)` rows collapse to rank 1. `rank(A)` deflated that way
    /// makes the guard pass where it must refuse — silently, on a model that is
    /// merely badly scaled rather than degenerate.
    ///
    /// Mutation: restore the global `scale`. Red here, and nowhere else — every
    /// other fixture in the crate is unit-scaled, which is exactly why this
    /// needed writing rather than assuming.
    #[test]
    fn row_rank_is_not_fooled_by_one_huge_row() {
        // Two independent rows whose independence shows up only in a modest
        // pivot (0.05 after eliminating column 0), plus one row 1e10 larger on
        // a column of its own. True rank 3.
        //
        // Equilibrated, that pivot is 0.048 against a `√ε` tolerance and
        // survives. Against a *global* max the tolerance is `√ε · 1e10 ≈ 0.15`,
        // the pivot is dropped, and the two independent rows read as one —
        // rank 2. That is the mutation, and this arithmetic is what makes it
        // bite; a fixture whose rows are all the same size cannot.
        let three = [
            row(&[(0, 1.0), (1, 1.0)]),
            row(&[(0, 1.0), (1, 1.05)]),
            row(&[(2, 1e10)]),
        ];
        assert_eq!(
            row_rank(&three, 3),
            3,
            "a huge row on its own column must not raise the tolerance for the others"
        );

        // And a big row that really is dependent still does not buy a rank.
        let dependent = [
            row(&[(0, 1.0), (2, 1.0)]),
            row(&[(1, 1.0), (2, 1.0)]),
            row(&[(0, 1e10), (1, 1e10), (2, 2e10)]),
        ];
        assert_eq!(row_rank(&dependent, 3), 2);
    }

    /// **The criterion is exact, not a dimension count**, and this is the input
    /// where the two disagree.
    ///
    /// `dim A(ker B) = rank([A;B]) − rank(B)`, and the step can reach every
    /// `db` only when that equals `rank(A)`. The dimension count
    /// `n − rank(B) ≥ rank(A)` is implied by it and strictly weaker.
    ///
    /// Here `A` lives entirely inside the coordinates `B` pins: `n = 4`,
    /// `rank(A) = 1`, `rank(B) = 3`, so `4 − 3 ≥ 1` passes — while
    /// `ker(B) = span(e₂)`, `A e₂ = 0`, and `A(ker B) = {0}`, so no nonzero
    /// `db` is reachable at all. `rank([A;B]) = 3 ≠ 1 + 3` refuses it.
    ///
    /// `an_equality_inside_the_pinned_coordinates_is_refused` in
    /// `tests/convex_soc_sensitivity.rs` is the same shape as a solved model,
    /// and carries the measurement (served, `dx/db = (⅓, ⅓, 0, 0)`, off by a
    /// third at every step size) that made this worth fixing rather than
    /// documenting.
    ///
    /// Mutation: `n.saturating_sub(rank_b) >= rank_a`. Red here and there, and
    /// on nothing else — the exact rule reproduces every other verdict in the
    /// crate.
    #[test]
    fn the_criterion_is_exact_not_a_dimension_count() {
        // B pins x₀, x₁, x₃; A is x₀ + x₁ = b, inside the pinned set.
        let b_rows = [row(&[(0, 1.0)]), row(&[(1, 1.0)]), row(&[(3, 1.0)])];
        let a_rows = [row(&[(0, 1.0), (1, 1.0)])];

        assert_eq!(row_rank(&a_rows, 4), 1);
        assert_eq!(row_rank(&b_rows, 4), 3);
        assert!(
            4_usize.saturating_sub(3) >= 1,
            "the dimension count passes this, which is the whole point"
        );
        assert!(
            !apex_can_absorb_db(4, &a_rows, &b_rows),
            "…and the exact criterion refuses it: A(ker B) = {{0}}"
        );

        // Move the equality onto a free coordinate and it is answerable again.
        let free = [row(&[(0, 1.0), (2, 1.0)])];
        assert!(apex_can_absorb_db(4, &free, &b_rows));
    }

    /// **The equality side is a rank, not a row count.**
    ///
    /// Raised in the third review of #889. `A`'s redundant rows do not shrink
    /// the space a step must reach: the reachable perturbations are `range(A)`,
    /// of dimension `rank(A)`, and a `db` outside that makes the *perturbed
    /// problem* infeasible rather than the derivative unrepresentable. Reading
    /// the row count instead over-refuses by exactly the redundancy.
    ///
    /// Two independent equalities against one free coordinate must refuse;
    /// the same two rows made proportional must not, because they are one
    /// equality written twice.
    ///
    /// Mutation: compare against `eq_rows.len()`. Red here, and on
    /// `an_apex_with_a_redundant_equality_is_served` in
    /// `tests/convex_soc_sensitivity.rs`, which reaches the same branch
    /// through a solved model.
    #[test]
    fn a_redundant_equality_does_not_cost_a_dimension() {
        let two_pinned = [row(&[(0, 1.0)]), row(&[(1, 1.0)])];

        let independent = [row(&[(0, 1.0), (2, 1.0)]), row(&[(1, 1.0), (2, 1.0)])];
        assert_eq!(row_rank(&independent, 3), 2);
        assert!(
            !apex_can_absorb_db(3, &independent, &two_pinned),
            "one free coordinate cannot absorb two independent equalities"
        );

        let redundant = [row(&[(0, 1.0), (2, 1.0)]), row(&[(0, 2.0), (2, 2.0)])];
        assert_eq!(
            row_rank(&redundant, 3),
            1,
            "the second row is the first, doubled"
        );
        assert!(
            apex_can_absorb_db(3, &redundant, &two_pinned),
            "…but two copies of one equality are still one equality"
        );
    }

    /// **The bound-exclusion, at the only shape that can convict it.**
    ///
    /// `apex_can_absorb_db` ranks the active rows that cannot be *released*.
    /// Active variable bounds are excluded on purpose: `release_slots` builds
    /// one releasable slot per active bound, so counting a bound as a hard pin
    /// would refuse the whole build — fix-relax included — for a model
    /// `parametric_step_bounded` could serve. The refusal is at build time,
    /// which takes the release path away with it.
    ///
    /// Here `n = 4`, `m_eq = 2`, and the apex pins two coordinates: rank 2,
    /// `n − 2 = 2 ≥ 2`, served. Stack a unit row for an active bound on a
    /// third coordinate and the rank is 3, `n − 3 = 1 < 2`, refused. So this
    /// input separates the two rules exactly, which no solved model can do
    /// without being primal degenerate (the argument is written out at
    /// `an_active_bound_does_not_count_as_an_apex_pin` in
    /// `tests/convex_soc_sensitivity.rs`).
    ///
    /// Mutation: re-add the `active_bounds` parameter and stack
    /// `vec![(j, 1.0)]` per bound. That changes the signature, so this call
    /// and the one in `finish` both move with it — the mutation is still
    /// compile-checkable, it is just not a one-line edit.
    #[test]
    fn an_active_bound_is_not_stacked_into_the_apex_rank() {
        // Two independent equalities, on coordinates the apex does not pin.
        let eqs = [row(&[(0, 1.0), (2, 1.0)]), row(&[(1, 1.0), (3, 1.0)])];
        let apex_rows = [row(&[(0, 1.0)]), row(&[(1, 1.0)])];
        assert!(
            apex_can_absorb_db(4, &eqs, &apex_rows),
            "two free coordinates absorb two equalities"
        );

        // The rank the discarded rule would have computed, spelled out so the
        // separation is visible rather than asserted: adding the bound row
        // takes it to 3, and `4 − 3 = 1 < 2`.
        let with_bound = [row(&[(0, 1.0)]), row(&[(1, 1.0)]), row(&[(2, 1.0)])];
        assert_eq!(row_rank(&with_bound, 4), 3);
        assert!(
            !apex_can_absorb_db(4, &eqs, &with_bound),
            "…so had the bound been stacked, this build would have been refused"
        );
    }

    /// gh #219's degenerate QP: `min ½‖x‖² s.t. x₀ + x₁ = 1, x₀ − 2x₁ ≤ h`.
    /// At `h = −½` the equality-only optimum `(½, ½)` hits the inequality
    /// *exactly*, so strict complementarity fails; other `h` give a strictly
    /// active (`h = −0.9`) or strictly inactive (`h = 0.5`) constraint.
    fn weakly_active_qp(h: f64) -> QpProblem {
        QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, -2.0)],
            h: vec![h],
            lb: vec![f64::NEG_INFINITY; 2],
            ub: vec![f64::INFINITY; 2],
        }
    }

    #[test]
    fn weak_activity_is_detected_independently_of_solver_tol() {
        // The point of the flag (gh #219). At this optimum `dx/db` is
        // two-valued — (2/3, 1/3) on the minus side, (1/2, 1/2) on the plus
        // side, 33% apart — and *which* one `parametric_step` reports turns on
        // `tol`, an otherwise unrelated setting: the multiplier and the slack
        // both collapse at ~√tol, so `active_tol` slices the pair differently
        // at different `tol`.
        //
        // `kkt_dim` therefore flips 4 → 3 across this sweep while the geometry
        // does not change at all. The weak-activity flag is the stable signal:
        // it must fire at every tolerance, including the ones where the
        // constraint *is* in the active set.
        let prob = weakly_active_qp(-0.5);
        let mut saw_in_active_set = false;
        let mut saw_out_of_active_set = false;
        for tol in [1e-8, 1e-12, 1e-14] {
            let opts = QpOptions {
                tol,
                ..QpOptions::default()
            };
            let sol = solve_qp_ipm(&prob, &opts, backend);
            assert_eq!(sol.status, QpStatus::Optimal);
            let sens = QpSensitivity::build(&prob, &sol, &opts, 1e-7, backend).unwrap();
            assert_eq!(
                sens.weakly_active_ineq(),
                &[0],
                "tol {tol:e}: weak activity missed (kkt_dim {})",
                sens.kkt_dim()
            );
            match sens.active_ineq() {
                [] => saw_out_of_active_set = true,
                [0] => saw_in_active_set = true,
                other => panic!("tol {tol:e}: unexpected active set {other:?}"),
            }
        }
        // Guards the premise: if the sweep stopped straddling the active-set
        // boundary the test would still pass while no longer testing anything.
        assert!(
            saw_in_active_set && saw_out_of_active_set,
            "sweep no longer straddles the active-set boundary, so this test \
             no longer demonstrates tol-independence"
        );
    }

    #[test]
    fn strictly_complementary_constraints_are_not_flagged_weak() {
        // The false-positive guard. A screen that fired on every active
        // constraint, or on every constraint with a small multiplier, would
        // pass the test above while being useless.
        //
        // `h = −0.9`: the constraint binds with multiplier ~8.9e-2 — strictly
        // active, `dx/db` two-sided. `h = 0.5`: the constraint is slack at the
        // optimum — strictly inactive. Neither is degenerate.
        for (h, expect_active) in [(-0.9, true), (0.5, false)] {
            let prob = weakly_active_qp(h);
            let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
            assert_eq!(sol.status, QpStatus::Optimal);
            let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
            assert!(
                sens.weakly_active_ineq().is_empty(),
                "h = {h}: strictly complementary constraint flagged as weakly active"
            );
            assert_eq!(
                !sens.active_ineq().is_empty(),
                expect_active,
                "h = {h}: active set {:?}",
                sens.active_ineq()
            );
        }
    }

    /// `min ½‖x‖²  s.t.  x₀ + x₁ = b` (b = 2). The optimum is the projection
    /// of the origin onto the line: `x = (b/2, b/2)`, so `dx/db = (½, ½)`
    /// exactly. The parametric step for `Δb` must reproduce that.
    #[test]
    fn parametric_step_matches_closed_form_equality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-7 && (sol.x[1] - 1.0).abs() < 1e-7);

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[1.0]); // Δb = +1
        assert!((dx[0] - 0.5).abs() < 1e-6, "dx0 = {}", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);

        // Predictor lands on the exact re-solve for the perturbed b.
        let mut prob2 = prob.clone();
        prob2.b = vec![3.0];
        let sol2 = solve_qp_ipm(&prob2, &QpOptions::default(), backend);
        assert!((sol.x[0] + dx[0] - sol2.x[0]).abs() < 1e-6);
        assert!((sol.x[1] + dx[1] - sol2.x[1]).abs() < 1e-6);
    }

    /// With an **active inequality** in the active set, the predictor must
    /// still match the re-solve. `min ½‖x‖² s.t. x₀+x₁ = b, x₀ ≥ 1`. At
    /// b = 1 the unconstrained projection would be (0.5, 0.5) but `x₀ ≥ 1`
    /// binds, giving `x = (1, 0)`. Perturbing b shifts along the active
    /// face: `x = (1, b−1)`, so `dx/db = (0, 1)`.
    #[test]
    fn parametric_step_with_active_inequality() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![1.0],
            g: vec![Triplet::new(0, 0, -1.0)], // −x₀ ≤ −1  ⇔  x₀ ≥ 1
            h: vec![-1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!((sol.x[0] - 1.0).abs() < 1e-6 && sol.x[1].abs() < 1e-6);
        assert!(sol.z[0] > 1e-6, "inequality should be active");

        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let dx = sens.parametric_step(&[0], &[0.5]);
        assert!(dx[0].abs() < 1e-6, "dx0 = {} (should stay on x₀=1)", dx[0]);
        assert!((dx[1] - 0.5).abs() < 1e-6, "dx1 = {}", dx[1]);
    }

    // ---- gh #284: near-LICQ conditioning diagnostic + refinement ----------

    /// The gh #284 Hessian `P = D·H₆·D` (Hilbert matrix `H₆`, `D =
    /// diag(1e3,…,1e-2)`; `cond(P) ≈ 7e15`) and its linear term. Shared by the
    /// conditioning tests below.
    fn hilbert_p_and_c() -> (Vec<Triplet>, Vec<f64>) {
        let d = [1e3, 1e2, 1e1, 1.0, 1e-1, 1e-2];
        let mut p_lower = Vec::new();
        for i in 0..6 {
            for j in 0..=i {
                let hij = 1.0 / ((i + j + 1) as f64);
                p_lower.push(Triplet::new(i, j, d[i] * hij * d[j]));
            }
        }
        (p_lower, vec![1.0, -2.0, 3.0, -1.0, 0.5, -0.25])
    }

    /// Two equality rows that are all-ones except the last entry of row 1,
    /// which differs by `eps` — nearly parallel, so LICQ nearly fails.
    fn near_parallel_rows(eps: f64) -> Vec<Triplet> {
        let mut a = Vec::new();
        for j in 0..6 {
            a.push(Triplet::new(0, j, 1.0));
            a.push(Triplet::new(1, j, if j == 5 { 1.0 + eps } else { 1.0 }));
        }
        a
    }

    /// Dense LU with partial pivoting — the float64 reference the gh #284 issue
    /// uses (`numpy.linalg.solve`) to show `dx/db` survives in double precision.
    /// `a` is row-major `n×n`; solves `a x = b`.
    fn dense_lu_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for k in 0..n {
            let mut piv = k;
            for i in (k + 1)..n {
                if a[i][k].abs() > a[piv][k].abs() {
                    piv = i;
                }
            }
            a.swap(k, piv);
            b.swap(k, piv);
            for i in (k + 1)..n {
                let f = a[i][k] / a[k][k];
                for j in k..n {
                    a[i][j] -= f * a[k][j];
                }
                b[i] -= f * b[k];
            }
        }
        let mut x = vec![0.0; n];
        for k in (0..n).rev() {
            let mut s = b[k];
            for j in (k + 1)..n {
                s -= a[k][j] * x[j];
            }
            x[k] = s / a[k][k];
        }
        x
    }

    /// Dense true (δ-free) equality KKT `[[P, Aᵀ], [A, 0]]`, row-major.
    fn dense_eq_kkt(p_lower: &[Triplet], a: &[Triplet], n: usize, m: usize) -> Vec<Vec<f64>> {
        let dim = n + m;
        let mut k = vec![vec![0.0; dim]; dim];
        for t in p_lower {
            k[t.row][t.col] += t.val;
            if t.row != t.col {
                k[t.col][t.row] += t.val;
            }
        }
        for t in a {
            k[n + t.row][t.col] += t.val;
            k[t.col][n + t.row] += t.val;
        }
        k
    }

    /// dx/db reference for the equality-only KKT: the x-block of the true-KKT
    /// solve with rhs `[0; e_{pin}]`, by dense float64 LU (independent of the
    /// factored/regularized path).
    fn dxdb_reference(prob: &QpProblem, pin: usize) -> Vec<f64> {
        let (n, m) = (prob.n, prob.m_eq());
        let kkt = dense_eq_kkt(&prob.p_lower, &prob.a, n, m);
        let mut rhs = vec![0.0; n + m];
        rhs[n + pin] = 1.0;
        dense_lu_solve(kkt, rhs)[..n].to_vec()
    }

    fn rel_err(a: &[f64], b: &[f64]) -> f64 {
        let scale = b.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        a.iter()
            .zip(b)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// A near-LICQ sensitivity (two equality rows differing by `1e-9`) is
    /// **detectably** untrustworthy (gh #284). Before the fix, `dx/db`
    /// collapsed to a smoothly over-damped, ~98%-wrong value while every
    /// existing signal (`weakly_active`, `kkt_dim`, status) looked ordinary and
    /// no exception was raised — the caller had no way to know. The
    /// conditioning diagnostic must fire, and the refinement residual must
    /// expose that the near-singular step was not solved.
    #[test]
    fn near_licq_sensitivity_is_flagged_ill_conditioned() {
        let (p_lower, c) = hilbert_p_and_c();
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a: near_parallel_rows(1e-10),
            b: vec![1.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // The old signals stay silent: nothing weakly active, full KKT dim.
        assert!(sens.weakly_active_ineq().is_empty());
        assert!(sens.weakly_active_bound_vars().is_empty());
        // The new diagnostic fires.
        assert!(
            sens.ill_conditioned(),
            "near-LICQ KKT must be flagged (κ₁ = {:.3e})",
            sens.kkt_cond_estimate()
        );
        assert!(
            sens.kkt_cond_estimate() > KKT_ILL_CONDITIONED_THRESHOLD,
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        // The step's residual against the true KKT is large: refinement could
        // not solve the near-singular system, so the step is unreliable.
        let _ = sens.parametric_step(&[0], &[1.0]);
        let res = sens.last_step_residual().expect("a step was taken");
        assert!(
            res > 1e-6,
            "expected a large refinement residual, got {res:.3e}"
        );
    }

    /// The false-alarm guard: the gh #284 *well-conditioned* case — the same
    /// badly-scaled `P = D·H₆·D` with a full-rank (if badly scaled) `A` whose
    /// rows differ by orders of magnitude, `cond(KKT) ≈ 5e9`. The diagnostic
    /// must stay quiet, and `dx/db` must match a dense float64 LU reference to
    /// ~1e-7 (the regularization introduces no detectable bias here).
    #[test]
    fn well_conditioned_sensitivity_not_flagged_and_accurate() {
        let (p_lower, c) = hilbert_p_and_c();
        // Rows: [1,1,1,1,1,1] and [1e4,1,1,1,1,1e-4] — badly scaled, full rank.
        let a = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 1.0),
            Triplet::new(0, 2, 1.0),
            Triplet::new(0, 3, 1.0),
            Triplet::new(0, 4, 1.0),
            Triplet::new(0, 5, 1.0),
            Triplet::new(1, 0, 1e4),
            Triplet::new(1, 1, 1.0),
            Triplet::new(1, 2, 1.0),
            Triplet::new(1, 3, 1.0),
            Triplet::new(1, 4, 1.0),
            Triplet::new(1, 5, 1e-4),
        ];
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a,
            b: vec![1.0, 2.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        assert!(
            !sens.ill_conditioned(),
            "well-conditioned KKT must NOT be flagged (κ₁ = {:.3e})",
            sens.kkt_cond_estimate()
        );
        assert!(
            sens.kkt_cond_estimate() < 1e12,
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        let dx = sens.parametric_step(&[0], &[1.0]);
        let reference = dxdb_reference(&prob, 0);
        let err = rel_err(&dx, &reference);
        assert!(err < 1e-7, "dx/db rel err vs float64 LU = {err:.3e}");
        assert!(
            sens.last_step_residual().unwrap() < 1e-8,
            "residual = {:?}",
            sens.last_step_residual()
        );
    }

    /// Refinement recovers accuracy where the information survives in double
    /// precision (gh #284). At `eps = 1e-6` the KKT is near-LICQ enough that a
    /// single regularized back-solve over-damps `dx/db` to ~4e-5 relative
    /// error, yet a plain float64 LU recovers it. Refinement against the
    /// unregularized KKT must close that gap — matching the LU reference far
    /// better than the un-refined solve could — while the conditioning flag
    /// stays quiet (the step *is* reliable here).
    #[test]
    fn refinement_recovers_dxdb_where_information_survives() {
        let (p_lower, c) = hilbert_p_and_c();
        let prob = QpProblem {
            n: 6,
            p_lower,
            c,
            a: near_parallel_rows(1e-6),
            b: vec![1.0, 1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // Recoverable, so not flagged.
        assert!(
            !sens.ill_conditioned(),
            "κ₁ = {:.3e}",
            sens.kkt_cond_estimate()
        );

        let dx = sens.parametric_step(&[0], &[1.0]);
        let reference = dxdb_reference(&prob, 0);
        let err = rel_err(&dx, &reference);
        // Comfortably better than the ~4e-5 an un-refined regularized solve
        // yields here: refinement did its job.
        assert!(
            err < 1e-6,
            "refined dx/db rel err vs float64 LU = {err:.3e}"
        );
        assert!(
            sens.last_step_residual().unwrap() < 1e-6,
            "residual = {:?}",
            sens.last_step_residual()
        );
    }

    /// gh #328: a **well-scaled** `P` (`P = I`) with a near-LICQ *constraint*
    /// Jacobian must never return a silently-wrong `dx/db`. `A = [[1,0],[1,ε]]`
    /// fully pins `x`, so the exact sensitivity is `dx/db = A⁻¹` with no
    /// truncation — `dx/db[:,0] = [1, −1/ε]`. As `ε → 0` the two equality rows
    /// become parallel and the KKT goes numerically singular, but because `P`
    /// is well scaled the build-time condition estimate saturates (`κ₁ ≈ 3e10`)
    /// and never reaches its threshold — the blind spot the old
    /// `ill_conditioned` had. The regression bar: for **every** `ε` the step is
    /// either accurate to a reasonable relative tolerance **or**
    /// `ill_conditioned` is `true`; a catastrophically over-damped step with
    /// `ill_conditioned == false` (the gh #328 failure — `−2999` where the truth
    /// is `−1e7`) must never happen. The well-conditioned `ε = 1e-3` end must
    /// stay accurate *and* unflagged.
    #[test]
    fn near_licq_constraint_jacobian_never_silently_wrong() {
        // Perturbing b0 by a small δb, then dividing out δb, is exactly how a
        // caller reads dx/db — and the small δb is what the old (1 + ‖rhs‖)
        // residual floor masked, so exercise that path here (δb = 1e-6).
        let db = 1e-6;
        let mut saw_flagged = false;
        for eps in [1e-3, 1e-5, 1e-7, 1e-9] {
            let prob = QpProblem {
                n: 2,
                p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
                c: vec![0.0, 0.0],
                // Rows [1,0] and [1,ε]: near-parallel as ε → 0.
                a: vec![
                    Triplet::new(0, 0, 1.0),
                    Triplet::new(1, 0, 1.0),
                    Triplet::new(1, 1, eps),
                ],
                b: vec![1.0, 1.0],
                g: vec![],
                h: vec![],
                lb: vec![],
                ub: vec![],
            };
            let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
            assert_eq!(sol.status, QpStatus::Optimal, "eps {eps:e}");
            let mut sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

            // dx/db[:,0] via a pinned step, divided back out.
            let step = sens.parametric_step(&[0], &[db]);
            let dxdb = [step[0] / db, step[1] / db];
            // Exact reference: A⁻¹[:,0] = [1, −1/ε].
            let exact = [1.0, -1.0 / eps];
            let rel = rel_err(&dxdb, &exact);
            let accurate = rel < 1e-3;

            // The acceptance bar: accurate OR honestly flagged. The forbidden
            // state is exactly the gh #328 bug — wrong AND unflagged.
            assert!(
                accurate || sens.ill_conditioned(),
                "eps {eps:e}: silently wrong dx/db = {dxdb:?} (exact {exact:?}, \
                 rel err {rel:.3e}) with ill_conditioned = false, \
                 kkt_cond = {:.3e}, residual = {:?}",
                sens.kkt_cond_estimate(),
                sens.last_step_residual(),
            );

            if eps == 1e-3 {
                // The well-conditioned end must be accurate *and* trusted.
                assert!(accurate, "eps {eps:e}: dx/db = {dxdb:?} rel err {rel:.3e}");
                assert!(
                    !sens.ill_conditioned(),
                    "eps {eps:e}: well-conditioned case falsely flagged \
                     (kkt_cond = {:.3e}, residual = {:?})",
                    sens.kkt_cond_estimate(),
                    sens.last_step_residual(),
                );
            } else {
                // Whenever the step is *not* accurate here, the flag must fire —
                // this is the clause that was silently false before the fix.
                if !accurate {
                    assert!(
                        sens.ill_conditioned(),
                        "eps {eps:e}: over-damped dx/db = {dxdb:?} (rel err \
                         {rel:.3e}) not flagged; residual = {:?}",
                        sens.last_step_residual(),
                    );
                    saw_flagged = true;
                }
            }
        }
        // Guards the premise: the sweep must actually reach the unrecoverable
        // regime, otherwise it no longer exercises the #328 flag path.
        assert!(
            saw_flagged,
            "sweep never hit an over-damped step, so the ill-conditioned flag \
             path was not exercised"
        );
    }

    /// A non-optimal solution has no well-defined active set.
    #[test]
    fn build_rejects_non_optimal() {
        let prob = QpProblem {
            n: 1,
            p_lower: vec![],
            c: vec![-1.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, -1.0)],
            h: vec![0.0], // x ≥ 0, min −x ⇒ unbounded
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_ne!(sol.status, QpStatus::Optimal);
        assert!(matches!(
            QpSensitivity::build_default(&prob, &sol, backend),
            Err(SensError::NotOptimal)
        ));
    }

    /// Unconstrained-direction reduced Hessian equals `P` itself: with no
    /// active constraints the null space is all of ℝⁿ, so `H_R = ZᵀPZ = P`
    /// (up to an orthonormal rotation, hence the eigenvalues match `P`).
    /// `min ½(2x₀² + 3x₁²)` has no binding constraints; eigenvalues = {2, 3}.
    #[test]
    fn reduced_hessian_unconstrained_is_p() {
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 2.0), Triplet::new(1, 1, 3.0)],
            c: vec![0.0, 0.0],
            a: vec![],
            b: vec![],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 2);
        assert!(
            (rh.eigenvalues[0] - 2.0).abs() < 1e-9,
            "{:?}",
            rh.eigenvalues
        );
        assert!(
            (rh.eigenvalues[1] - 3.0).abs() < 1e-9,
            "{:?}",
            rh.eigenvalues
        );
    }

    /// One equality constraint removes one degree of freedom. `min ½‖x‖²`
    /// (P = I) on the 3-D space with `x₀ + x₁ + x₂ = b` leaves a 2-D null
    /// space; the reduced Hessian is the 2×2 identity (both curvatures = 1).
    #[test]
    fn reduced_hessian_drops_one_dof_per_active_constraint() {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 2, 1.0),
            ],
            c: vec![0.0, 0.0, 0.0],
            a: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(0, 2, 1.0),
            ],
            b: vec![1.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 2, "one equality ⇒ 2 DOF");
        for &ev in &rh.eigenvalues {
            assert!((ev - 1.0).abs() < 1e-9, "eig {ev}");
        }
    }

    /// A non-identity reduced Hessian: `min ½xᵀPx` with a coupled `P` and an
    /// equality that pins the sum, cross-checked against the hand-computed
    /// `ZᵀPZ` for the unit null-space direction `z = (1,−1)/√2`.
    #[test]
    fn reduced_hessian_value_matches_hand_projection() {
        // P = [[3, 1], [1, 2]]; constraint x₀ + x₁ = 0 ⇒ Z = (1,−1)/√2.
        // zᵀPz = (3 − 1 − 1 + 2)/2 = 3/2.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![
                Triplet::new(0, 0, 3.0),
                Triplet::new(1, 0, 1.0),
                Triplet::new(1, 1, 2.0),
            ],
            c: vec![0.0, 0.0],
            a: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            b: vec![0.0],
            g: vec![],
            h: vec![],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 1);
        assert!(
            (rh.eigenvalues[0] - 1.5).abs() < 1e-9,
            "H_R = {:?}",
            rh.eigenvalues
        );
        assert!((rh.matrix[0] - 1.5).abs() < 1e-9);
    }

    /// Two **simultaneously active** inequality rows, each with *multiple*
    /// nonzeros and a **shared column**, so both the KKT build and the
    /// reduced-Hessian assembly must read each active row's full set of
    /// `(col, val)` entries — and must not let one row's entries leak into
    /// the other (col 1 appears in both). The single-triplet active-row
    /// fixtures elsewhere never exercise the per-row grouping; this is the
    /// guard for the `group_rows_by_index` assembly.
    ///
    /// `min ½‖x‖² − 2·𝟙ᵀx` (unconstrained min at `(2,2,2)`) with
    /// `x₀+x₁ ≤ 1` and `x₁+x₂ ≤ 1`. Both bind at the optimum `(1,0,1)`
    /// with equal positive multipliers (λ = 1), so `B = [[1,1,0],[0,1,1]]`
    /// has rank 2 → one degree of freedom. The null space is spanned by
    /// `(−1,1,−1)/√3`, so `H_R = ZᵀIZ = 1`.
    #[test]
    fn reduced_hessian_two_active_multi_triplet_rows() {
        let prob = QpProblem {
            n: 3,
            p_lower: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(2, 2, 1.0),
            ],
            c: vec![-2.0, -2.0, -2.0],
            a: vec![],
            b: vec![],
            // Row 0: x₀ + x₁ (cols 0,1); row 1: x₁ + x₂ (cols 1,2).
            g: vec![
                Triplet::new(0, 0, 1.0),
                Triplet::new(0, 1, 1.0),
                Triplet::new(1, 1, 1.0),
                Triplet::new(1, 2, 1.0),
            ],
            h: vec![1.0, 1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        assert!(
            (sol.x[0] - 1.0).abs() < 1e-5 && sol.x[1].abs() < 1e-5 && (sol.x[2] - 1.0).abs() < 1e-5,
            "x = {:?} (expected (1, 0, 1))",
            sol.x,
        );
        assert!(
            sol.z[0] > 1e-6 && sol.z[1] > 1e-6,
            "both inequalities should be active: z = {:?}",
            sol.z,
        );

        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();
        let rh = sens
            .reduced_hessian_default()
            .expect("eigensolve converges");
        assert_eq!(rh.n_dof, 1, "rank-2 active Jacobian on n=3 ⇒ 1 DOF");
        assert!(
            (rh.eigenvalues[0] - 1.0).abs() < 1e-7,
            "H_R = {:?} (expected eigenvalue 1)",
            rh.eigenvalues,
        );

        // The build's KKT must also see both active rows: a free RHS over
        // the (empty) equality block leaves dx = 0, but the factorization
        // having succeeded with dim = n + 0 + 2 confirms both rows entered.
        assert_eq!(sens.kkt_dim(), 3 + 0 + 2);
    }

    /// `reduced_hessian` now *returns* an eigensolve-convergence verdict
    /// instead of silently ignoring it: on a well-formed QP both internal
    /// symmetric eigensolves (the `BᵀB` rank/null-space split and the final
    /// `H_R` decomposition) converge, so the call must yield `Ok` with the
    /// hand-checked reduced Hessian.
    ///
    /// The `Err(EigenFailed)` branch is a defensive consistency guard: it can
    /// only trip if `symmetric_eigen` exhausts its sweeps, which a modest,
    /// well-conditioned reduced Hessian like this one never does — so the
    /// failure path is not reachable through the public solver here and is not
    /// exercised by a fixture (the same limitation noted for the underlying
    /// `symmetric_eigen` convergence flag). This test pins the `Ok` contract;
    /// before the fix the function returned a bare `ReducedHessian` and a
    /// non-converged solve would have been published as if trustworthy.
    #[test]
    fn reduced_hessian_returns_ok_on_convergent_eigensolve() {
        // min ½‖x‖² − 2·𝟙ᵀx with x₀ + x₁ ≤ 1 (active at (0.5, 0.5)); the
        // single active row has rank 1 on n=2 ⇒ 1 DOF, null space (1,−1)/√2,
        // so H_R = ZᵀIZ = 1.
        let prob = QpProblem {
            n: 2,
            p_lower: vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, 1.0)],
            c: vec![-2.0, -2.0],
            a: vec![],
            b: vec![],
            g: vec![Triplet::new(0, 0, 1.0), Triplet::new(0, 1, 1.0)],
            h: vec![1.0],
            lb: vec![],
            ub: vec![],
        };
        let sol = solve_qp_ipm(&prob, &QpOptions::default(), backend);
        assert_eq!(sol.status, QpStatus::Optimal);
        let sens = QpSensitivity::build_default(&prob, &sol, backend).unwrap();

        // The verdict is surfaced, not discarded: matching on the Result is
        // the behavior L40 introduced.
        let rh = match sens.reduced_hessian_default() {
            Ok(rh) => rh,
            Err(e) => panic!("convergent eigensolve must yield Ok, got {e:?}"),
        };
        assert_eq!(rh.n_dof, 1, "rank-1 active Jacobian on n=2 ⇒ 1 DOF");
        assert!(
            (rh.eigenvalues[0] - 1.0).abs() < 1e-7,
            "H_R = {:?} (expected eigenvalue 1)",
            rh.eigenvalues,
        );
        // The explicit-tolerance entry point carries the same contract.
        assert!(sens.reduced_hessian(1e-9).is_ok());
    }

    /// Pin the regularization value the module doc cites. The default-built
    /// sensitivity (`build_default` → `QpOptions::default()`) places
    /// `opts.reg` on the KKT diagonal (see `build`, `let reg = opts.reg`), so
    /// the "default `δ`" the module-level doc names *is* `QpOptions::default()
    /// .reg`. That default was retuned `1e-8 → 1e-10` (ipm.rs: `1e-8` stalls
    /// `adlittle`), but the doc kept saying `1e-8` (L42). This guards the doc
    /// against silent drift: if the default reg changes again, this fails and
    /// forces the module doc to be updated in lockstep.
    #[test]
    fn module_doc_regularization_matches_qp_options_default() {
        assert_eq!(
            QpOptions::default().reg,
            1e-10,
            "module doc names this as the default sensitivity regularization δ",
        );
    }
}

// ---------------------------------------------------------------------------
// The `SensBacksolver` adapter.
// ---------------------------------------------------------------------------

/// The convex active-set KKT, as a [`SensBacksolver`].
///
/// This is what lets the convex arm reach the parametric machinery in
/// `pounce-sens-core` — fix-relax refinement, and in due course path following
/// and the directional derivative — instead of reimplementing it. That
/// machinery is already generic over this trait, derives its variable count
/// from slice lengths, and reads bounds only through [`BoundRow`], so the only
/// thing it needed was an implementation over a different KKT.
///
/// # Why `Rc<RefCell<…>>`
///
/// Not a style choice. [`Factorization::solve_one`] takes `&mut self` while
/// [`SensBacksolver::solve`] takes `&self`, and `boundcheck` additionally
/// requires `B: Clone` while `Factorization` owns a non-`Clone`
/// `TSymLinearSolver`. Shared interior mutability is the only shape that
/// satisfies all three — and it is exactly the shape the NLP arm's
/// `PdSensBacksolver` already uses for the same reasons.
///
/// # Block layout
///
/// The compound vector is `(x, y, z_a)`: `n` primal rows, `m_eq` equality
/// multipliers, then one row per active constraint (inequality rows first, then
/// bound rows). The NLP arm's is an eight-block iterate instead, which is fine —
/// the shared machinery assumes only that block 0 is `x`.
#[derive(Clone)]
pub struct QpKktBacksolver {
    fact: Rc<RefCell<Factorization>>,
    airn: Rc<Vec<Index>>,
    ajcn: Rc<Vec<Index>>,
    vals_true: Rc<Vec<f64>>,
    scratch: Rc<RefCell<IrScratch>>,
    dim: usize,
    bound_rows: Rc<Vec<BoundRow>>,
    /// Relative residual of the most recent `solve`, so the caller's
    /// `ill_conditioned` reporting keeps working across boundcheck-driven
    /// solves. A `Cell` because `solve` takes `&self`.
    last_residual: Rc<Cell<f64>>,
    // --- release support ---
    /// The factored (regularized) values of the *unreleased* system; a release
    /// starts from these and neutralizes the released rows.
    vals_reg: Rc<Vec<f64>>,
    /// Per active bound, the `(coupling, diagonal)` value slots to neutralize.
    slots: Rc<Vec<(usize, usize)>>,
    /// Per active bound, its oriented base multiplier — what
    /// `solve_released_step` moves onto the variable's `x` row.
    base_mult: Rc<Vec<f64>>,
    /// The most recent released system, kept so a repeat costs nothing and a
    /// *different* release costs one numeric refactorization rather than a new
    /// symbolic one (the pattern never changes).
    released: Rc<RefCell<Option<ReleasedFactor>>>,
    release_backend: Rc<RefCell<Option<Box<dyn SparseSymLinearSolverInterface>>>>,
}

/// A factored released system, with the released row set it belongs to.
struct ReleasedFactor {
    key: Vec<usize>,
    fact: Factorization,
    vals_true: Vec<f64>,
}

impl QpKktBacksolver {
    /// Relative KKT residual of the most recent [`SensBacksolver::solve`],
    /// or `f64::NAN` if none has run.
    pub fn last_residual(&self) -> f64 {
        self.last_residual.get()
    }
}

impl SensBacksolver for QpKktBacksolver {
    fn dim(&self) -> usize {
        self.dim
    }

    fn solve(&self, rhs: &[Number], lhs: &mut [Number]) -> bool {
        if rhs.len() != self.dim || lhs.len() != self.dim {
            return false;
        }
        lhs.copy_from_slice(rhs);
        let mut fact = self.fact.borrow_mut();
        let mut scratch = self.scratch.borrow_mut();
        match solve_refined(
            &mut fact,
            &self.airn,
            &self.ajcn,
            &self.vals_true,
            lhs,
            &mut scratch,
        ) {
            Ok(res) => {
                self.last_residual.set(res);
                true
            }
            Err(()) => false,
        }
    }

    /// `None`, and that is correct rather than unimplemented.
    ///
    /// The factor this trait answers in and the multipliers a caller reads off
    /// `QpSolution` are already in one frame: Ruiz equilibration lives inside
    /// `solve_qp_ipm` and is undone by `Scaling::unscale_solution` before the
    /// solution is returned, and [`assemble_kkt`] builds from the raw
    /// `QpProblem`. So the per-row factor is the identity.
    ///
    /// That is an **invariant, not a fact of nature**. If anyone later builds
    /// the sensitivity KKT from equilibrated data — a plausible "make the
    /// factorization faster" change — `F` silently becomes the Ruiz diagonal,
    /// and every pin and release would read a mis-scaled multiplier against a
    /// natural-units step.
    ///
    /// **And it is currently unguarded, measured rather than assumed.** Making
    /// this return a non-identity vector turns *nothing* in the crate red,
    /// because the only consumer is the release half of `refine_step_onto_bounds`
    /// and [`supports_release`](Self::supports_release) is `false` here. So the
    /// value is correct and untested at the same time. The phase that turns
    /// release on is the phase that makes it load-bearing, and it owes this a
    /// guard — a leg comparing a released step against a re-solve, which cannot
    /// pass with a mis-scaled `F`.
    ///
    /// `the_step_is_unmoved_by_internal_equilibration` guards the neighbouring
    /// and weaker claim that `dx/db` itself does not depend on whether the
    /// solve equilibrated internally. That is worth having, but it exercises
    /// `parametric_step`, which never reads this factor.
    fn natural_units_factor(&self) -> Option<&[Number]> {
        None
    }

    fn bound_rows(&self) -> Option<&[BoundRow]> {
        Some(&self.bound_rows)
    }

    /// Both halves of fix-relax are available on this arm.
    ///
    /// Releasing is cheaper here than on the NLP one, and for a structural
    /// reason worth stating: the NLP's release *must* re-factor because an
    /// active bound puts `σ = z/s` on the `x` diagonal, and on a tightly
    /// converged bound that term destroys the released system's information in
    /// the converged factor — the better the solve converged, the worse a
    /// recovered answer would be. The convex active-set KKT carries no barrier
    /// term at all; the bound is an explicit row. Releasing it is exact, and
    /// costs one numeric refactorization against an unchanged sparsity pattern.
    fn supports_release(&self) -> bool {
        true
    }

    fn solve_released(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, false)
    }

    fn solve_released_step(&self, released: &[usize], rhs: &[Number], lhs: &mut [Number]) -> bool {
        self.solve_released_inner(released, rhs, lhs, true)
    }
}

impl QpKktBacksolver {
    /// Index of the active bound whose multiplier lives at compound row `row`.
    fn bound_at(&self, row: usize) -> Option<usize> {
        self.bound_rows.iter().position(|b| b.row == row)
    }

    /// Make the released system current, factoring it if the cached one is for
    /// a different released set.
    ///
    /// Neutralizing a row means zeroing its `±1` coupling to the variable and
    /// setting its diagonal to `−1`: the row then reads `−dz_a = rhs`, so the
    /// multiplier decouples and the variable is free. The **sparsity pattern is
    /// untouched**, which is the point — `refactor` reuses the symbolic
    /// factorization, so a release costs a numeric factorization rather than a
    /// fresh analyse.
    fn ensure_released(&self, key: &[usize]) -> bool {
        {
            let cached = self.released.borrow();
            if cached.as_ref().is_some_and(|rf| rf.key == key) {
                return true;
            }
        }
        let mut vals_reg = (*self.vals_reg).clone();
        let mut vals_true = (*self.vals_true).clone();
        for &row in key {
            let Some(k) = self.bound_at(row) else {
                return false;
            };
            let (coupling, diagonal) = self.slots[k];
            vals_reg[coupling] = 0.0;
            vals_true[coupling] = 0.0;
            vals_reg[diagonal] = -1.0;
            vals_true[diagonal] = -1.0;
        }
        let mut slot = self.released.borrow_mut();
        match slot.as_mut() {
            Some(rf) => {
                if rf.fact.refactor(&vals_reg).is_err() {
                    return false;
                }
                rf.key = key.to_vec();
                rf.vals_true = vals_true;
            }
            None => {
                let Some(backend) = self.release_backend.borrow_mut().take() else {
                    // Already consumed, which cannot happen: the `None` arm
                    // runs at most once, and the factor it builds is refactored
                    // thereafter.
                    return false;
                };
                let Ok(fact) = Factorization::new(
                    self.dim as Index,
                    (*self.airn).clone(),
                    (*self.ajcn).clone(),
                    vals_reg,
                    backend,
                ) else {
                    return false;
                };
                *slot = Some(ReleasedFactor {
                    key: key.to_vec(),
                    fact,
                    vals_true,
                });
            }
        }
        true
    }

    fn solve_released_inner(
        &self,
        released: &[usize],
        rhs: &[Number],
        lhs: &mut [Number],
        shift: bool,
    ) -> bool {
        if rhs.len() != self.dim || lhs.len() != self.dim {
            return false;
        }
        if released.is_empty() {
            return self.solve(rhs, lhs);
        }
        let mut key = released.to_vec();
        key.sort_unstable();
        key.dedup();
        if !self.ensure_released(&key) {
            return false;
        }

        lhs.copy_from_slice(rhs);
        if shift {
            // The released multiplier no longer acts on its variable, so its
            // base value moves onto that variable's `x` row with the sign that
            // row carries the bound's side with — `−z` for a lower bound,
            // `+z` for an upper one. Mirrors the NLP arm's
            // `shift_released_rhs`, which is the reference for this convention.
            for &row in &key {
                let Some(k) = self.bound_at(row) else {
                    return false;
                };
                let b = &self.bound_rows[k];
                if b.var_row >= self.dim {
                    return false;
                }
                lhs[row] = 0.0;
                let z = self.base_mult[k];
                lhs[b.var_row] += if b.lower { -z } else { z };
            }
        }

        let mut slot = self.released.borrow_mut();
        let Some(rf) = slot.as_mut() else {
            return false;
        };
        let mut scratch = self.scratch.borrow_mut();
        let vals_true = rf.vals_true.clone();
        match solve_refined(
            &mut rf.fact,
            &self.airn,
            &self.ajcn,
            &vals_true,
            lhs,
            &mut scratch,
        ) {
            Ok(res) => {
                self.last_residual.set(res);
                true
            }
            Err(()) => false,
        }
    }
}
