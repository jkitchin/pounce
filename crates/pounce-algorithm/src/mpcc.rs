//! Policy for solving a model with declared complementarity pairs: which
//! lowerings of [`pounce_nlp::mpcc::MpccTnlp`] to try, in what order, and
//! what counts as an answer.
//!
//! Workstream B of `dev-notes/research/calipso-contact-plan.md`. Like
//! [`crate::second_opinion`], this module holds the policy and runs no
//! solves; the driver that runs them is
//! `pounce_restoration::mpcc_driver`, so every entry point shares one
//! implementation.
//!
//! ## Why a ladder and not one route
//!
//! Measured on `benchmarks/calipso/` (seven complementarity models, POUNCE
//! and Ipopt), no single lowering wins: the exact product is fast and
//! solves most models from most starts but fails where a pair is biactive
//! at the answer; the Scholtes continuation with full warm starts and an
//! exact finish was the only route physical on 5/5 starts on bunny-hop and
//! the drifting car, at several times the iterations; the ℓ₁ penalty was the
//! only route that solved the quadruped on every seed. So the driver tries
//! them in [`STAGES`] order: the continuation and the exact product both
//! run and the better certified answer wins ([`compared`]); the penalty
//! runs only if neither answers.
//!
//! ## Why the continuation goes first
//!
//! Measured on the Gate 0 corpus (`benchmarks/mpcc/declared.py`, 60 feasible
//! cells: every case, both scaling legs, every start). Exact first: 60
//! solved, 0 below `f*`, but only **48** at `f*` — the exact product
//! converges fast (median 19 iterations) to S-stationary local minima that
//! are not the optimum (`regular_strict` 4 for 1, the selectors 0.75 for
//! 0.25). Continuation first: 60 solved, **60** at `f*`, 0 below, median 60
//! iterations — against 57 at `f*` for `scholtes_then_ncp`, the route Gate 0
//! chose as supported. The relaxation at `τ = 1` sees the whole feasible
//! set before it commits to a branch; the exact product commits from the
//! start. The cost is about 3× the iterations where the exact product
//! would have found the optimum anyway. The exact product stays as the
//! second stage, for models whose relaxation path fails.
//!
//! ## What an answer is
//!
//! A converged status is not enough: a lowered solve "succeeds" on its own
//! reformulation, and the reformulation's tolerance is not the model's. A
//! stage is accepted only if it converged, every product `G·H` is within
//! `constr_viol_tol` — the bar the model's own rows are held to — **and**
//! [`certify`] finds the point stationary ([`answers`]). The last test is
//! not redundant: through the Python callbacks, qpec_small's exact-product
//! stage exits "acceptable" with every product at 0 at a point 9e-4 short of
//! the answer, which is not stationary (residual 1.7e-3 against 2e-6), and
//! the continuation reaches the answer.

use faer::Mat;
use pounce_common::types::Number;
use pounce_nlp::mpcc::ComplementarityPair;
use pounce_nlp::return_codes::ApplicationReturnStatus;

/// The stages, in the order the driver tries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpccStage {
    /// `G·H = 0` from the model's own start (`G·H <= 0` where the equality
    /// would leave the model with more equalities than variables).
    Exact,
    /// `G·H <= τ` for τ down [`SCHOLTES_SCHEDULE`], each stage warm-started
    /// from the last, then one `G·H = 0` solve from the end of it
    /// (`G·H <= 0` where the equality would leave the model with more
    /// equalities than variables).
    Continuation,
    /// `G·H = 0` through the ℓ₁ exact-penalty barrier.
    Penalty,
}

impl MpccStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact product",
            Self::Continuation => "Scholtes continuation",
            Self::Penalty => "l1 penalty",
        }
    }
}

pub const STAGES: [MpccStage; 3] = [
    MpccStage::Continuation,
    MpccStage::Exact,
    MpccStage::Penalty,
];

/// The relaxation schedule, as in `benchmarks/mpcc/routes.py`: it starts at
/// 1 so a relaxation whose crossover lies in `[0.25, 1]` is seen, and ends
/// four orders below the default `constr_viol_tol`.
pub const SCHOLTES_SCHEDULE: [Number; 9] = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];

/// The options a warm-started continuation stage sets, as
/// `pounce.WarmStart.options()` does: the seeds are honoured, the barrier
/// starts where the last stage stopped instead of re-walking from 0.1, and
/// the pushes are tightened so a point at a bound stays there. Each is an
/// upstream Ipopt option. The driver restores every one of them to the
/// caller's set-ness afterwards.
pub fn warm_start_assignments(final_mu: Number) -> Vec<String> {
    let mu = if final_mu.is_finite() && final_mu > 0.0 {
        final_mu.clamp(1e-9, 1e-1)
    } else {
        1e-6
    };
    let p = 1e-9;
    vec![
        "warm_start_init_point yes\n".to_string(),
        format!("mu_init {mu:e}\n"),
        format!("warm_start_bound_push {p:e}\n"),
        format!("warm_start_bound_frac {p:e}\n"),
        format!("warm_start_slack_bound_push {p:e}\n"),
        format!("warm_start_slack_bound_frac {p:e}\n"),
        format!("warm_start_mult_bound_push {p:e}\n"),
    ]
}

/// Every option any stage writes; the driver snapshots these before the
/// first stage and restores them after the last.
pub const TOUCHED_OPTIONS: [&str; 10] = [
    "warm_start_init_point",
    "mu_init",
    "warm_start_bound_push",
    "warm_start_bound_frac",
    "warm_start_slack_bound_push",
    "warm_start_slack_bound_frac",
    "warm_start_mult_bound_push",
    "l1_exact_penalty_barrier",
    "bound_relax_factor",
    "honor_original_bounds",
];

/// Defaults every stage runs under unless the caller set the option
/// themselves. `bound_relax_factor`'s default of `1e-8` relaxes every bound
/// and row before the solve, which on a complementarity pair means accepting
/// `G >= -1e-8`, and at a cone tip or a pair's corner turns into a violation
/// of its square root: on `benchmarks/calipso/` every reported success that
/// violated the model was a run under that default. `honor_original_bounds`
/// moves the answer back inside the bounds the model stated.
pub const SAFE_DEFAULTS: [(&str, &str); 2] = [
    ("bound_relax_factor", "0"),
    ("honor_original_bounds", "yes"),
];

/// One line for the console: what answered, and what the answer is.
pub fn describe_outcome(
    accepted_by: Option<MpccStage>,
    n_solves: usize,
    total_iterations: usize,
    max_abs_product: Number,
    certificate: Option<&Certificate>,
) -> String {
    let head = match accepted_by {
        Some(stage) => format!(
            "pounce: complementarity: answered by the {} formulation ({n_solves} lowered \
             solve{}, {total_iterations} iterations in all, max |G·H| = {max_abs_product:.1e})",
            stage.label(),
            if n_solves == 1 { "" } else { "s" },
        ),
        None => format!(
            "pounce: complementarity: no formulation produced an answer ({n_solves} lowered \
             solves, {total_iterations} iterations); reporting the exact-product solve"
        ),
    };
    let tail = match certificate {
        None => "; stationarity not evaluated".to_string(),
        Some(c) => {
            let class = match c.class {
                StationarityClass::S => "S (strongly stationary)",
                StationarityClass::W => {
                    "W only (a biactive pair has a wrong-sign multiplier: a branch may still descend)"
                }
                StationarityClass::None => "none (not a stationary point of the model)",
                StationarityClass::NotSearched => {
                    "not determined (multiplier set too large to search)"
                }
            };
            format!(
                "; {} pair{}, {} biactive; stationarity: {class}.",
                c.n_pairs,
                if c.n_pairs == 1 { "" } else { "s" },
                c.n_biactive
            )
        }
    };
    head + &tail
}

/// Statuses a stage can converge with.
pub fn converged(status: ApplicationReturnStatus) -> bool {
    matches!(
        status,
        ApplicationReturnStatus::SolveSucceeded
            | ApplicationReturnStatus::SolvedToAcceptableLevel
            | ApplicationReturnStatus::FeasiblePointFound
    )
}

/// Are a stage's S answers compared with the other compared stage's, the
/// lower objective winning, rather than ending the ladder? The continuation
/// and the exact product: on a nonconvex model each reaches local minima the
/// other avoids, and neither order wins on both corpora we have. Gate 0
/// (60 feasible cells) at `f*`: continuation first 60, exact first 48.
/// CALIPSO bunny-hop (5 seeds) at the best known objective: exact first 5,
/// continuation first 2 — the continuation settles on S-stationary
/// trajectories at 1111–1169 against 462.6. Running both and keeping the
/// lower objective gets every one of those, for the iterations of both.
/// The penalty runs only when neither answers.
pub fn compared(stage: MpccStage) -> bool {
    matches!(stage, MpccStage::Continuation | MpccStage::Exact)
}

/// Does a certified class make a converged, physical point the answer, ending
/// the ladder? S, the goal; and `NotSearched`, which claims nothing either way
/// and only arises on models large enough that more stages are costly. A W
/// point is not: it admits a descent direction along some branch, and a later
/// stage can reach an S point below it ([`kept_if_nothing_better`]). A point
/// that is not stationary at all is a stall, and the driver moves on.
pub fn answers(class: StationarityClass) -> bool {
    matches!(class, StationarityClass::S | StationarityClass::NotSearched)
}

/// Is a converged, physical point that does not [`answers`] still worth
/// reporting if no later stage certifies? W is: it is the honest answer
/// where no S point exists nearby (ralph1's origin). Measured on the CALIPSO
/// corpus's `complementarity` model, stopping at the first W point returned
/// objective 85 where the continuation reaches the S optimum at 17.
pub fn kept_if_nothing_better(class: StationarityClass) -> bool {
    class == StationarityClass::W
}

/// Is a stage's answer an answer to the *model*, not just to the lowering?
pub fn accepts(
    status: ApplicationReturnStatus,
    max_abs_product: Number,
    constr_viol_tol: Number,
) -> bool {
    converged(status) && max_abs_product.is_finite() && max_abs_product <= constr_viol_tol
}

// ---------------------------------------------------------------------------
// The certificate: which stationarity class the reported point satisfies.
// ---------------------------------------------------------------------------

/// The stationarity class a point was shown to satisfy; see [`certify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StationarityClass {
    /// Strong stationarity: a KKT point of the tightened NLP, B-stationary
    /// under MPCC-LICQ. The strongest first-order statement there is.
    S,
    /// Weak stationarity only: multipliers reproduce `∇f`, but every
    /// admissible choice puts a negative sign on some biactive pair, so the
    /// point admits a first-order descent direction along a branch.
    W,
    /// No multipliers reproduce `∇f` to the tolerance: not a stationary
    /// point of the model (or a pair is not complementary at all).
    None,
    /// The multipliers are not unique and the model is too large to search
    /// the set; nothing is claimed.
    NotSearched,
}

/// What [`certify`] measured. Everything the verdict rests on is here.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub class: StationarityClass,
    pub n_pairs: usize,
    pub n_biactive: usize,
    /// Pairs with neither side at zero — complementarity does not hold.
    pub n_neither: usize,
    /// Columns of the stationarity system are linearly dependent, so the
    /// multipliers form a set rather than a point (MPCC-LICQ fails).
    pub multipliers_unique: bool,
    /// `‖A θ − ∇f‖∞` at the multipliers that decided the class.
    pub residual: Number,
    pub resid_tol: Number,
    /// Largest-magnitude negative sign on a biactive pair, when the class
    /// is decided by a sign (`W` with unique multipliers).
    pub worst_biactive_sign: Number,
}

/// The point and model data [`certify`] reads. All indices 0-based, the
/// Jacobian as `(row, col, value)` triplets of the *inner* model's rows.
pub struct CertificateInput<'a> {
    pub x: &'a [Number],
    pub grad_f: &'a [Number],
    pub g: &'a [Number],
    pub jac: &'a [(usize, usize, Number)],
    pub x_l: &'a [Number],
    pub x_u: &'a [Number],
    pub g_l: &'a [Number],
    pub g_u: &'a [Number],
    pub pairs: &'a [ComplementarityPair],
    /// The tolerance the answer was solved to (`tol`).
    pub solver_tol: Number,
}

/// Above this many sign-constrained columns, a rank-deficient system is not
/// searched (each NNLS step is an SVD of the passive columns).
pub const MAX_SEARCHED_COLUMNS: usize = 600;

const INF: Number = 1e19;

#[derive(Clone, Copy, PartialEq)]
enum Sign {
    Free,
    NonNeg,
    /// Non-negative under S, free under W.
    Biactive,
}

/// Which stationarity class `x` satisfies, as `benchmarks/calipso/
/// stationarity.py` decides it — same columns, same thresholds — so the two
/// can be held to each other on the same points.
///
/// The stationarity system is
/// `∇f = Σ λ ∇g + Σ ν ∇G + Σ w ∇H + z_L − z_U` with: equality rows free;
/// active inequality rows and bounds sign-constrained (inactive ones absent);
/// a pair side free when only it is active, absent when inactive, and on a
/// biactive pair non-negative under S and free under W.
///
/// * Activity: a pair side counts as zero within `sqrt(solver_tol)` times
///   the family's largest magnitude (a product driven to `tol` pins each
///   side only to its square root); rows and bounds within
///   `sqrt(solver_tol)`.
/// * Residual: judged against `max(1e-6, solver_tol)` times the dual scale
///   — the largest entry of `∇f` and of every usable column.
/// * Signs: a multiplier counts as negative only below `-sqrt(solver_tol)`
///   times the dual scale. An answer near a biactive pair is itself only
///   `sqrt(tol)`-accurate, so smaller signs are noise, not descent. This is
///   the one place this departs from `stationarity.py`, whose BVLS has no
///   band; on the points it was run on (exact or `tol=1e-8` answers away
///   from biactive pairs) the two agree.
///
/// When the columns have full rank the multipliers are unique and the
/// class is a sign check. When they do not (MPCC-LICQ fails) the class is a
/// property of the multiplier *set* — reading one vector can report W for a
/// point that is S — so the set is searched with NNLS, up to
/// [`MAX_SEARCHED_COLUMNS`].
pub fn certify(inp: &CertificateInput<'_>) -> Certificate {
    let n = inp.x.len();
    let m = inp.g.len();
    let corner = inp.solver_tol.max(0.0).sqrt();
    let row_tol = corner;

    // Pair sides and their family scales.
    // Each side's orientation from its one finite bound, as MpccTnlp reads
    // it: (+1, lower) or (-1, upper).
    let side = |lo: Number, hi: Number| -> (Number, Number) {
        if lo > -INF { (1.0, lo) } else { (-1.0, hi) }
    };
    let g_side: Vec<(Number, Number)> = inp
        .pairs
        .iter()
        .map(|p| side(inp.x_l[p.var], inp.x_u[p.var]))
        .collect();
    let h_side: Vec<(Number, Number)> = inp
        .pairs
        .iter()
        .map(|p| side(inp.g_l[p.row], inp.g_u[p.row]))
        .collect();
    let gv: Vec<Number> = inp
        .pairs
        .iter()
        .zip(&g_side)
        .map(|(p, &(s, b))| s * (inp.x[p.var] - b))
        .collect();
    let hv: Vec<Number> = inp
        .pairs
        .iter()
        .zip(&h_side)
        .map(|(p, &(s, c))| s * (inp.g[p.row] - c))
        .collect();
    let gs = gv.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let hs = hv.iter().fold(1.0_f64, |a, v| a.max(v.abs()));
    let g_act: Vec<bool> = gv.iter().map(|v| v.abs() <= corner * gs).collect();
    let h_act: Vec<bool> = hv.iter().map(|v| v.abs() <= corner * hs).collect();
    let n_biactive = g_act.iter().zip(&h_act).filter(|(a, b)| **a && **b).count();
    let n_neither = g_act
        .iter()
        .zip(&h_act)
        .filter(|(a, b)| !**a && !**b)
        .count();

    let mut pair_row = vec![None; m];
    let mut pair_var = vec![None; n];
    for (k, p) in inp.pairs.iter().enumerate() {
        pair_row[p.row] = Some(k);
        pair_var[p.var] = Some(k);
    }
    let mut rows_of: Vec<Vec<(usize, Number)>> = vec![Vec::new(); m];
    for &(r, c, v) in inp.jac {
        rows_of[r].push((c, v));
    }

    // Columns: (sparse entries, sign class).
    let mut cols: Vec<(Vec<(usize, Number)>, Sign)> = Vec::new();
    for i in 0..m {
        if let Some(k) = pair_row[i] {
            let sh = h_side[k].0;
            let grad_h: Vec<(usize, Number)> =
                rows_of[i].iter().map(|&(c, v)| (c, sh * v)).collect();
            match (g_act[k], h_act[k]) {
                (_, false) => {}
                (true, true) => cols.push((grad_h, Sign::Biactive)),
                (false, true) => cols.push((grad_h, Sign::Free)),
            }
            continue;
        }
        let (lo, hi) = (inp.g_l[i], inp.g_u[i]);
        let at_lo = lo > -INF && inp.g[i] - lo <= row_tol;
        let at_hi = hi < INF && hi - inp.g[i] <= row_tol;
        let neg: Vec<(usize, Number)> = rows_of[i].iter().map(|&(c, v)| (c, -v)).collect();
        match (at_lo, at_hi) {
            (true, true) => cols.push((rows_of[i].clone(), Sign::Free)),
            (true, false) => cols.push((rows_of[i].clone(), Sign::NonNeg)),
            (false, true) => cols.push((neg, Sign::NonNeg)),
            (false, false) => {}
        }
    }
    for (j, pv) in pair_var.iter().enumerate() {
        if let Some(k) = *pv {
            let sg = g_side[k].0;
            match (g_act[k], h_act[k]) {
                (false, _) => {}
                (true, true) => cols.push((vec![(j, sg)], Sign::Biactive)),
                (true, false) => cols.push((vec![(j, sg)], Sign::Free)),
            }
            // A pair variable's other bound is infinite (MpccTnlp refuses
            // anything else), so nothing more to add.
            continue;
        }
        let (lo, hi) = (inp.x_l[j], inp.x_u[j]);
        let at_lo = lo > -INF && inp.x[j] - lo <= row_tol;
        let at_hi = hi < INF && hi - inp.x[j] <= row_tol;
        match (at_lo, at_hi) {
            (true, true) => cols.push((vec![(j, 1.0)], Sign::Free)),
            (true, false) => cols.push((vec![(j, 1.0)], Sign::NonNeg)),
            (false, true) => cols.push((vec![(j, -1.0)], Sign::NonNeg)),
            (false, false) => {}
        }
    }

    let b = inp.grad_f.to_vec();
    let k = cols.len();
    let a = Mat::<Number>::from_fn(n, k, |_, _| 0.0);
    let mut a = a;
    let mut col_max: Number = 0.0;
    for (c, (ents, _)) in cols.iter().enumerate() {
        for &(r, v) in ents {
            a[(r, c)] += v;
            col_max = col_max.max(v.abs());
        }
    }
    let dual_scale = 1.0_f64
        .max(b.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
        .max(col_max);
    let resid_tol = inp.solver_tol.max(1e-6) * dual_scale;
    // The noise band on a multiplier's sign. An answer near a biactive pair
    // is accurate only to `sqrt(tol)` (that is where the activity threshold
    // comes from), so `∇f` there — and every multiplier reproducing it —
    // carries an error of that order. qpec_small's converged point, accepted
    // at products <= 1e-4, has a biactive multiplier of -1.6e-4 while the
    // exact point's is 0. A sign inside the band is not evidence of descent.
    let sign_tol = corner * dual_scale;
    let signs: Vec<Sign> = cols.iter().map(|(_, s)| *s).collect();

    let mut cert = Certificate {
        class: StationarityClass::None,
        n_pairs: inp.pairs.len(),
        n_biactive,
        n_neither,
        multipliers_unique: true,
        residual: Number::INFINITY,
        resid_tol,
        worst_biactive_sign: 0.0,
    };
    if n_neither > 0 {
        // Complementarity itself fails at this point: no class applies.
        return cert;
    }

    // Sign-free least squares: the W test's lower bound on the residual,
    // and the unique multipliers when the columns have full rank.
    let (theta, rank) = lstsq(&a, &b);
    let r_free = resid_inf(&a, &theta, &b);
    cert.residual = r_free;
    if r_free > resid_tol {
        return cert; // not stationary for any multipliers, signs aside
    }
    if rank == k {
        let ordinary_ok = signs
            .iter()
            .zip(&theta)
            .all(|(s, t)| *s != Sign::NonNeg || *t >= -sign_tol);
        if !ordinary_ok {
            return cert;
        }
        let worst = signs
            .iter()
            .zip(&theta)
            .filter(|(s, _)| **s == Sign::Biactive)
            .fold(0.0_f64, |w, (_, t)| w.min(*t));
        cert.worst_biactive_sign = worst;
        cert.class = if worst >= -sign_tol {
            StationarityClass::S
        } else {
            StationarityClass::W
        };
        return cert;
    }

    // Non-unique multipliers: search the set.
    cert.multipliers_unique = false;
    let constrained = signs.iter().filter(|s| **s != Sign::Free).count();
    if constrained > MAX_SEARCHED_COLUMNS {
        cert.class = StationarityClass::NotSearched;
        return cert;
    }
    for (class, biactive_signed) in [(StationarityClass::S, true), (StationarityClass::W, false)] {
        let nonneg: Vec<bool> = signs
            .iter()
            .map(|s| *s == Sign::NonNeg || (biactive_signed && *s == Sign::Biactive))
            .collect();
        // Forcing a multiplier inside the noise band to zero moves the
        // residual by up to that band; allow it.
        let r = signed_lstsq_residual(&a, &b, &nonneg);
        if r <= resid_tol + sign_tol {
            cert.class = class;
            cert.residual = r;
            return cert;
        }
    }
    cert
}

/// Minimum-norm least squares by thin SVD; returns the solution and the
/// numerical rank (LAPACK's threshold).
fn lstsq(a: &Mat<Number>, b: &[Number]) -> (Vec<Number>, usize) {
    let (n, k) = (a.nrows(), a.ncols());
    if k == 0 {
        return (Vec::new(), 0);
    }
    let Ok(svd) = a.thin_svd() else {
        return (vec![0.0; k], 0);
    };
    let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
    let smax = s.first().copied().unwrap_or(0.0);
    let tol = smax * (n.max(k) as Number) * Number::EPSILON;
    let u = svd.U();
    let v = svd.V();
    let mut theta = vec![0.0; k];
    let mut rank = 0;
    for (i, &si) in s.iter().enumerate() {
        if si <= tol {
            continue;
        }
        rank += 1;
        let coef = (0..n).map(|r| u[(r, i)] * b[r]).sum::<Number>() / si;
        for (c, t) in theta.iter_mut().enumerate() {
            *t += coef * v[(c, i)];
        }
    }
    (theta, rank)
}

fn resid_inf(a: &Mat<Number>, theta: &[Number], b: &[Number]) -> Number {
    (0..a.nrows())
        .map(|r| ((0..a.ncols()).map(|c| a[(r, c)] * theta[c]).sum::<Number>() - b[r]).abs())
        .fold(0.0, Number::max)
}

/// `min ‖A θ − b‖₂` with `θ_c >= 0` where `nonneg[c]`, the rest free; returns
/// the ∞-norm residual. The free columns are projected out (their least-
/// squares part is exact), then Lawson–Hanson NNLS on the rest.
fn signed_lstsq_residual(a: &Mat<Number>, b: &[Number], nonneg: &[bool]) -> Number {
    let n = a.nrows();
    let free: Vec<usize> = (0..a.ncols()).filter(|&c| !nonneg[c]).collect();
    let cons: Vec<usize> = (0..a.ncols()).filter(|&c| nonneg[c]).collect();
    // Orthonormal basis of range(A_free).
    let basis: Vec<Vec<Number>> = if free.is_empty() {
        Vec::new()
    } else {
        let af = Mat::<Number>::from_fn(n, free.len(), |r, c| a[(r, free[c])]);
        match af.thin_svd() {
            Ok(svd) => {
                let s: Vec<Number> = svd.S().column_vector().iter().copied().collect();
                let smax = s.first().copied().unwrap_or(0.0);
                let tol = smax * (n.max(free.len()) as Number) * Number::EPSILON;
                let u = svd.U();
                (0..s.len())
                    .filter(|&i| s[i] > tol)
                    .map(|i| (0..n).map(|r| u[(r, i)]).collect())
                    .collect()
            }
            Err(_) => return Number::INFINITY,
        }
    };
    let project = |v: &mut Vec<Number>| {
        for q in &basis {
            let d: Number = q.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            for (vi, qi) in v.iter_mut().zip(q) {
                *vi -= d * qi;
            }
        }
    };
    let mut pb = b.to_vec();
    project(&mut pb);
    let pa: Vec<Vec<Number>> = cons
        .iter()
        .map(|&c| {
            let mut v: Vec<Number> = (0..n).map(|r| a[(r, c)]).collect();
            project(&mut v);
            v
        })
        .collect();
    let x = nnls(&pa, &pb, n);
    (0..n)
        .map(|r| {
            (pb[r]
                - pa.iter()
                    .zip(&x)
                    .map(|(col, xi)| col[r] * xi)
                    .sum::<Number>())
            .abs()
        })
        .fold(0.0, Number::max)
}

/// Lawson–Hanson non-negative least squares, columns given explicitly.
fn nnls(cols: &[Vec<Number>], b: &[Number], n: usize) -> Vec<Number> {
    let k = cols.len();
    let mut x = vec![0.0; k];
    if k == 0 {
        return x;
    }
    let mut passive = vec![false; k];
    let resid = |x: &[Number]| -> Vec<Number> {
        (0..n)
            .map(|r| b[r] - cols.iter().zip(x).map(|(c, xi)| c[r] * xi).sum::<Number>())
            .collect()
    };
    let grad = |res: &[Number]| -> Vec<Number> {
        cols.iter()
            .map(|c| c.iter().zip(res).map(|(a, r)| a * r).sum())
            .collect()
    };
    let scale = cols
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(1.0)
        * b.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
    let wtol = 1e-12 * scale * (n.max(k) as Number);
    for _outer in 0..3 * k + 10 {
        let w = grad(&resid(&x));
        let Some((t, wt)) = (0..k)
            .filter(|&j| !passive[j])
            .map(|j| (j, w[j]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        else {
            break;
        };
        if wt <= wtol {
            break;
        }
        passive[t] = true;
        for _inner in 0..3 * k + 10 {
            let p: Vec<usize> = (0..k).filter(|&j| passive[j]).collect();
            let ap = Mat::<Number>::from_fn(n, p.len(), |r, c| cols[p[c]][r]);
            let (z, _) = lstsq(&ap, b);
            if z.iter().all(|&v| v > 0.0) {
                for (i, &j) in p.iter().enumerate() {
                    x[j] = z[i];
                }
                break;
            }
            let mut alpha = 1.0_f64;
            for (i, &j) in p.iter().enumerate() {
                if z[i] <= 0.0 {
                    let d = x[j] - z[i];
                    if d > 0.0 {
                        alpha = alpha.min(x[j] / d);
                    }
                }
            }
            for (i, &j) in p.iter().enumerate() {
                x[j] += alpha * (z[i] - x[j]);
                if x[j] <= 1e-15 * scale {
                    x[j] = 0.0;
                    passive[j] = false;
                }
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_converged_lowering_with_a_large_product_is_not_an_answer() {
        let ok = ApplicationReturnStatus::SolveSucceeded;
        assert!(accepts(ok, 1e-9, 1e-4));
        assert!(!accepts(ok, 1e-3, 1e-4));
        assert!(!accepts(ok, Number::NAN, 1e-4));
        assert!(!accepts(
            ApplicationReturnStatus::InfeasibleProblemDetected,
            0.0,
            1e-4
        ));
        assert!(accepts(
            ApplicationReturnStatus::FeasiblePointFound,
            0.0,
            1e-4
        ));
    }

    #[test]
    fn warm_mu_is_clamped_and_never_zero() {
        let mu = |v| {
            warm_start_assignments(v)
                .into_iter()
                .find(|a| a.starts_with("mu_init"))
                .unwrap()
        };
        assert_eq!(mu(1e-12), "mu_init 1e-9\n");
        assert_eq!(mu(3.0), "mu_init 1e-1\n");
        assert_eq!(mu(0.0), "mu_init 1e-6\n");
        assert_eq!(mu(Number::NAN), "mu_init 1e-6\n");
    }

    #[test]
    fn every_assignment_is_restored() {
        for a in warm_start_assignments(1e-5) {
            let name = a.split_whitespace().next().unwrap();
            assert!(
                TOUCHED_OPTIONS.contains(&name),
                "{name} is written but not restored"
            );
        }
    }

    // ---- certify: the same points `benchmarks/calipso/stationarity.py`'s
    // selftest and `benchmarks/mpcc/cases.py` decide, so the two
    // implementations are held to each other. Python's M (and C) is "not S"
    // here, reported as W.

    const NI: Number = -1e20;
    const PI: Number = 1e20;

    #[derive(Clone)]
    struct Model {
        x_l: Vec<Number>,
        x_u: Vec<Number>,
        g_l: Vec<Number>,
        g_u: Vec<Number>,
        pairs: Vec<ComplementarityPair>,
        jac: Vec<(usize, usize, Number)>,
    }

    fn run(m: &Model, x: &[Number], grad_f: &[Number], g: &[Number]) -> Certificate {
        certify(&CertificateInput {
            x,
            grad_f,
            g,
            jac: &m.jac,
            x_l: &m.x_l,
            x_u: &m.x_u,
            g_l: &m.g_l,
            g_u: &m.g_u,
            pairs: &m.pairs,
            solver_tol: 1e-8,
        })
    }

    /// CALIPSO.jl `nonconvex/complementarity.jl`, its x-x pairs written as
    /// rows `g_{4+k} = x_{5+k} >= 0` paired with `x_{2+k}` (and those
    /// variables' own bounds dropped, so the row is the only statement of
    /// them — the same system the Python classifier builds).
    #[allow(clippy::type_complexity)]
    fn complementarity() -> (Model, impl Fn(&[Number]) -> (Vec<Number>, Vec<Number>)) {
        let mut jac = vec![
            (0, 1, 0.5),
            (0, 2, 1.0),
            (0, 3, -0.5),
            (0, 4, 1.0),
            (1, 0, 3.0),
            (1, 1, -1.0),
            (1, 5, -1.0),
            (2, 0, -1.0),
            (2, 1, 0.5),
            (2, 6, -1.0),
            (3, 0, -1.0),
            (3, 1, -1.0),
            (3, 7, -1.0),
        ];
        for k in 0..3 {
            jac.push((4 + k, 5 + k, 1.0));
        }
        let m = Model {
            x_l: vec![0.0, 0.0, 0.0, 0.0, 0.0, NI, NI, NI],
            x_u: vec![PI; 8],
            g_l: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            g_u: vec![0.0, 0.0, 0.0, 0.0, PI, PI, PI],
            pairs: (0..3)
                .map(|k| ComplementarityPair {
                    row: 4 + k,
                    var: 2 + k,
                })
                .collect(),
            jac,
        };
        let eval = |x: &[Number]| {
            let grad = vec![
                2.0 * (x[0] - 5.0),
                4.0 * (2.0 * x[1] + 1.0),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            let g = vec![
                2.0 * (x[1] - 1.0) - 1.5 * x[1] + x[2] - 0.5 * x[3] + x[4],
                3.0 * x[0] - x[1] - 3.0 - x[5],
                -x[0] + 0.5 * x[1] + 4.0 - x[6],
                -x[0] - x[1] + 7.0 - x[7],
                x[5],
                x[6],
                x[7],
            ];
            (grad, g)
        };
        (m, eval)
    }

    #[test]
    fn certify_matches_the_python_classifier_on_complementarity() {
        let (m, eval) = complementarity();
        for (x, want, nb) in [
            (
                [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 6.0],
                StationarityClass::S,
                0,
            ), // f = 17
            (
                [5.0, 2.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0],
                StationarityClass::S,
                1,
            ), // f = 25
            (
                [3.0, 4.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0],
                StationarityClass::W,
                1,
            ), // f = 85, Python: M
        ] {
            let (grad, g) = eval(&x);
            assert!(
                g[..4].iter().all(|v| v.abs() < 1e-12),
                "{x:?} is not feasible: {g:?}"
            );
            let c = run(&m, &x, &grad, &g);
            assert_eq!((c.class, c.n_biactive), (want, nb), "{x:?}: {c:?}");
        }
    }

    #[test]
    fn certify_reads_kkt_on_a_plain_nlp() {
        // maratos: min 2(x0^2 + x1^2 - 1) - x0  s.t.  x0^2 + x1^2 = 1.
        let m = Model {
            x_l: vec![NI; 2],
            x_u: vec![PI; 2],
            g_l: vec![1.0],
            g_u: vec![1.0],
            pairs: vec![],
            jac: vec![],
        };
        let at = |x: [Number; 2]| {
            let mut mm = Model {
                jac: vec![(0, 0, 2.0 * x[0]), (0, 1, 2.0 * x[1])],
                ..m.clone()
            };
            mm.jac.retain(|e| e.2 != 0.0);
            let grad = [4.0 * x[0] - 1.0, 4.0 * x[1]];
            run(&mm, &x, &grad, &[x[0] * x[0] + x[1] * x[1]]).class
        };
        assert_eq!(at([1.0, 0.0]), StationarityClass::S);
        assert_eq!(at([0.0, 1.0]), StationarityClass::None); // feasible, not stationary
    }

    #[test]
    fn certify_ralph1_ctrap_and_qpec_small() {
        // ralph1: min 2x - y,  x >= 0,  0 <= y ⟂ y - x >= 0. Origin: M, not S.
        let m = Model {
            x_l: vec![0.0, 0.0],
            x_u: vec![PI; 2],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 1 }],
            jac: vec![(0, 0, -1.0), (0, 1, 1.0)],
        };
        let c = run(&m, &[0.0, 0.0], &[2.0, -1.0], &[0.0]);
        assert_eq!((c.class, c.n_biactive), (StationarityClass::W, 1), "{c:?}");

        // ralph1 mirrored through x -> -x, y -> -y (both sides upper-bounded:
        // y <= 0 ⟂ y - x <= 0): the same point, the same verdict.
        let mirrored = Model {
            x_l: vec![NI, NI],
            x_u: vec![0.0, 0.0],
            g_l: vec![NI],
            g_u: vec![0.0],
            pairs: vec![ComplementarityPair { row: 0, var: 1 }],
            jac: vec![(0, 0, -1.0), (0, 1, 1.0)],
        };
        let c = run(&mirrored, &[0.0, 0.0], &[-2.0, 1.0], &[0.0]);
        assert_eq!((c.class, c.n_biactive), (StationarityClass::W, 1), "{c:?}");

        // ctrap: min x1^2 + x2^2 - x1 - x2,  0 <= x1 ⟂ x2 >= 0. Origin: C, not S.
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![PI; 2],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let c = run(&m, &[0.0, 0.0], &[-1.0, -1.0], &[0.0]);
        assert_eq!(c.class, StationarityClass::W, "{c:?}");
        // ... and its minimizer (1/2, 0) is S, strictly complementary.
        let c = run(&m, &[0.5, 0.0], &[0.0, -1.0], &[0.0]);
        assert_eq!((c.class, c.n_biactive), (StationarityClass::S, 0), "{c:?}");

        // qpec_small at (1, 1, 0): grad f = 0, one biactive pair, S.
        let m = Model {
            x_l: vec![0.0, 0.0, 0.0],
            x_u: vec![2.0, PI, PI],
            g_l: vec![0.0, 0.0],
            g_u: vec![PI, PI],
            pairs: vec![
                ComplementarityPair { row: 0, var: 1 },
                ComplementarityPair { row: 1, var: 2 },
            ],
            jac: vec![(0, 0, -1.0), (0, 1, 2.0), (1, 0, 1.0), (1, 2, 2.0)],
        };
        let c = run(&m, &[1.0, 1.0, 0.0], &[0.0, 0.0, 0.0], &[0.0, 0.0]);
        assert_eq!((c.class, c.n_biactive), (StationarityClass::S, 1), "{c:?}");
        // ctrap's minimizer mirrored (x1 <= 0 ⟂ x2 <= 0, at (-1/2, 0)): S,
        // reading the orientation from the finite bounds.
        let m = Model {
            x_l: vec![NI, NI],
            x_u: vec![0.0, PI],
            g_l: vec![NI],
            g_u: vec![0.0],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let c = run(&m, &[-0.5, 0.0], &[0.0, 1.0], &[0.0]);
        assert_eq!((c.class, c.n_biactive), (StationarityClass::S, 0), "{c:?}");
        let c = run(&m, &[0.0, 0.0], &[1.0, 1.0], &[0.0]);
        assert_eq!(c.class, StationarityClass::W, "{c:?}");
    }

    /// The class is a property of the multiplier *set*. Here an equality row
    /// is parallel to the pair's `H` side, so the multipliers are not unique;
    /// the minimum-norm vector splits the `-0.1` between them and puts a
    /// negative sign on the biactive `w`, but `w = 0, λ = -0.1, ν = 1` is an
    /// S vector. Reading one vector would report W.
    ///
    /// `min x0 - 0.1 x1   s.t.  x1 = 0,   0 <= x0 ⟂ x1 >= 0`, at the origin.
    #[test]
    fn certify_searches_the_multiplier_set_when_it_is_not_a_point() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![PI; 2],
            g_l: vec![0.0, 0.0],
            g_u: vec![PI, 0.0],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0), (1, 1, 1.0)],
        };
        let c = run(&m, &[0.0, 0.0], &[1.0, -0.1], &[0.0, 0.0]);
        assert!(!c.multipliers_unique, "{c:?}");
        assert_eq!((c.class, c.n_biactive), (StationarityClass::S, 1), "{c:?}");
        // The minimum-norm vector alone would have said W.
        let a = Mat::<Number>::from_fn(2, 3, |r, col| match (r, col) {
            (0, 0) => 1.0,
            (1, 1) | (1, 2) => 1.0,
            _ => 0.0,
        });
        let (theta, _) = lstsq(&a, &[1.0, -0.1]);
        assert!(theta[1] < 0.0, "{theta:?}");
    }

    #[test]
    fn certify_refuses_a_point_where_a_pair_is_not_complementary() {
        let m = Model {
            x_l: vec![0.0, NI],
            x_u: vec![PI; 2],
            g_l: vec![0.0],
            g_u: vec![PI],
            pairs: vec![ComplementarityPair { row: 0, var: 0 }],
            jac: vec![(0, 1, 1.0)],
        };
        let c = run(&m, &[0.5, 0.5], &[0.0, 0.0], &[0.5]);
        assert_eq!(
            (c.class, c.n_neither),
            (StationarityClass::None, 1),
            "{c:?}"
        );
    }
}
